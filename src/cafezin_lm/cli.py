"""Cafezin LM command line interface.

Commands:
  spec        write a fresh spec.json from a built model
  verify      compare a model against the reference fingerprint (offline)
  prepare     tokenize a corpus into train/val memmaps
  train       run training (resumable)
  export      write a trained model in the HF layout (safetensors + index + config)
  info        parameter count and memory estimate
  generate    sample from a checkpoint (or a fresh model)
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

from .checkpoint import (
    checkpoint_assets,
    compare_specs,
    import_config,
    load_hf_checkpoint,
    model_spec,
    remote_spec,
    save_hf_checkpoint,
    spec_digest,
    verify_offline,
)
from .config import PRESET_NAMES, ModelConfig, preset, tiny_config
from .data import load_meta, prepare
from .model import (
    CafezinLM,
    TextRotaryEmbedding,
    VisionRotaryEmbedding,
    build_model,
    parameter_count_breakdown,
    parameter_count_estimate_full,
)
from .safetensors_io import nbytes_of
from .tokenizer import Tokenizer
from .train import TrainConfig, train


def _checkpoint_model(src: Path, fallback_preset: str) -> CafezinLM:
    directory = src if src.is_dir() else src.parent
    cfg = (
        import_config(directory)
        if (directory / "config.json").is_file()
        else preset(fallback_preset)
    )
    with torch.device("meta"):
        model = build_model(cfg)
    load_hf_checkpoint(model, src, assign=True)
    # These small, non-persistent RoPE buffers are absent from the weight files.
    model.model.language_model.rotary_emb = TextRotaryEmbedding(cfg.text)
    model.model.visual.rotary_pos_emb = VisionRotaryEmbedding(cfg.vision)
    return model


def cmd_spec(args: argparse.Namespace) -> int:
    with torch.device("meta"):
        model = build_model(preset(args.preset), seed=args.seed)
    spec = model_spec(model, dtype_override=args.dtype)
    total = sum(nbytes_of(dtype, shape) for dtype, shape in spec.values())
    digest = spec_digest(spec, total)
    out = {
        "tensors": len(spec),
        "total_bytes": total,
        "digest": digest,
        "spec": {k: [v, list(s)] for k, (v, s) in spec.items()},
    }
    Path(args.output).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"spec written to {args.output} (tensors={len(spec)}, bytes={total:,}, digest={digest})")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    with torch.device("meta"):
        model = build_model(preset(args.preset), seed=args.seed)
    if args.offline:
        report = verify_offline(model)
    else:
        remote = remote_spec(cache_dir=args.cache_dir)
        local = model_spec(model)
        report = compare_specs(local, remote.spec, remote.total_size)
    print(report.summary())
    if args.verbose:
        print(report.details())
    return 0 if report.ok else 1


def cmd_prepare(args: argparse.Namespace) -> int:
    tok = Tokenizer(args.tokenizer) if args.tokenizer else Tokenizer.from_hub()
    meta = prepare(
        tok,
        args.out,
        val_fraction=args.val_fraction,
        dtype=args.dtype,
        inputs=args.input,
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        text_field=args.text_field,
        chat_jsonl=args.chat_jsonl,
        max_docs=args.max_docs,
    )
    print(
        f"prepared: {meta['n_docs']} docs, "
        f"{meta['train_tokens']:,} train, {meta['val_tokens']:,} val tokens"
    )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    model_cfg = None
    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        cfg = TrainConfig.from_dict(config.get("train", config))
        if "model" in config:
            model_cfg = ModelConfig.from_dict(config["model"])
        if args.resume is not None:
            cfg.resume = args.resume
        if args.resume_step is not None:
            cfg.resume_step = args.resume_step
    else:
        cfg = TrainConfig(
            out_dir=args.out_dir,
            data_dir=args.data_dir,
            seed=args.seed,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            steps=args.steps,
            eval_every=args.eval_every,
            eval_batches=args.eval_batches,
            lr=args.lr,
            min_lr=args.min_lr,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            beta1=args.beta1,
            beta2=args.beta2,
            grad_clip=args.grad_clip,
            gradient_checkpointing=args.gradient_checkpointing,
            mtp_weight=args.mtp_weight,
            compile_model=args.compile,
            log_every=args.log_every,
            save_every=args.save_every,
            keep_last=args.keep_last,
            device=args.device,
            dtype=args.dtype or TrainConfig.dtype,
            torch_seed=args.torch_seed,
            resume=args.resume,
            resume_step=args.resume_step,
        )

    if model_cfg is None:
        if cfg.resume:
            model_cfg = import_config(cfg.resume)
            if not args.config and args.dtype is None:
                cfg.dtype = model_cfg.text.dtype
        else:
            meta = load_meta(cfg.data_dir)
            model_cfg = preset(args.preset)
            if args.preset == "tiny":
                required_vocab = meta.get("max_token_id", meta["vocab_size"] - 1) + 1
                model_cfg = tiny_config(vocab_size=max(model_cfg.text.vocab_size, required_vocab))
            if meta.get("eos_token_id") is not None:
                model_cfg.text.eos_token_id = meta["eos_token_id"]

    train(model_cfg, cfg)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    src = Path(args.checkpoint)
    model = _checkpoint_model(src, args.preset)
    directory = src if src.is_dir() else src.parent
    assets = checkpoint_assets(directory)
    if args.assets_dir:
        extra_assets = checkpoint_assets(args.assets_dir)
        existing_tokenizer = assets.get("tokenizer.json")
        supplied_tokenizer = extra_assets.get("tokenizer.json")
        if (
            existing_tokenizer is not None
            and supplied_tokenizer is not None
            and json.loads(existing_tokenizer) != json.loads(supplied_tokenizer)
        ):
            raise ValueError("--assets-dir contains a different tokenizer from the checkpoint")
        assets.update(extra_assets)
    gen_file = directory / "generation_config.json"
    generation_config = (
        json.loads(gen_file.read_text(encoding="utf-8")) if gen_file.is_file() else None
    )
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"export destination already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".export-", dir=out.parent) as tmp:
        staged = Path(tmp) / "model"
        manifest = save_hf_checkpoint(
            model,
            staged,
            dtype_override=args.dtype_override,
            tokenizer_files=assets,
            generation_config=generation_config,
        )
        staged.rename(out)
    print(f"exported to {out} (shards={len(manifest['shards'])}, tensors={manifest['tensors']})")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    cfg = preset(args.preset)
    with torch.device("meta"):
        model = build_model(cfg)
    breakdown = parameter_count_breakdown(model)
    estimate = parameter_count_estimate_full(cfg)
    print(f"parameters: {model.num_params():,} (estimate {estimate:,})")
    for k, v in breakdown.items():
        print(f"  {k}: {v:,}")
    dtype = getattr(torch, cfg.text.dtype)
    bytes_per = dtype.itemsize
    print(f"  dtype: {dtype}")
    print(f"  ~model weights: {model.num_params() * bytes_per / 1e9:.2f} GB")
    print(f"  ~adam states (x2): {model.num_params() * 8 / 1e9:.2f} GB")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    src = Path(args.checkpoint) if args.checkpoint else None
    tokenizer_path = Path(args.tokenizer) if args.tokenizer else None
    if src is not None and tokenizer_path is None:
        directory = src if src.is_dir() else src.parent
        candidate = directory / "tokenizer.json"
        if candidate.is_file():
            tokenizer_path = candidate
    tok = Tokenizer(tokenizer_path) if tokenizer_path else Tokenizer.from_hub()
    if src is not None:
        model = _checkpoint_model(src, args.preset)
    else:
        cfg = preset(args.preset)
        if args.preset == "tiny":
            cfg = tiny_config(vocab_size=max(cfg.text.vocab_size, tok.max_token_id + 1))
        model = build_model(cfg, seed=args.seed)
    if tok.max_token_id >= model.cfg.text.vocab_size:
        raise ValueError("tokenizer IDs exceed the checkpoint's vocabulary; supply --tokenizer")
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    model.to(device)

    prompt_ids = tok.encode(args.prompt, eos=False)
    eos = tok.eos_token_id if tok.eos_token_id is not None else model.cfg.text.eos_token_id
    if not prompt_ids:
        if tok.eos_token_id is None:
            raise ValueError("provide a nonempty prompt when the tokenizer has no EOS token")
        prompt_ids = [eos]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(
        input_ids,
        max_new_tokens=args.max_new,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=eos,
        vocab_limit=tok.max_token_id + 1,
    )
    print(tok.decode(out[0].tolist()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cafezin-lm",
        description="Cafezin LM: hybrid linear/full-attention multimodal decoder",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # spec
    p = sub.add_parser("spec", help="write a weight spec from a built model")
    p.add_argument("--preset", choices=PRESET_NAMES, default="tiny")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", choices=["BF16", "F32"], help="override the reference dtype policy")
    p.add_argument("--output", default="spec.json")
    p.set_defaults(func=cmd_spec)

    # verify
    p = sub.add_parser("verify", help="compare against the reference fingerprint")
    p.add_argument("--preset", choices=PRESET_NAMES, default="base")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--offline", action="store_true", help="use the recorded fingerprint")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--cache-dir", help="directory for downloaded tensor metadata")
    p.set_defaults(func=cmd_verify)

    # prepare
    p = sub.add_parser("prepare", help="tokenize a corpus into train/val memmaps")
    p.add_argument("--out", default="data")
    p.add_argument(
        "--tokenizer", help="local tokenizer.json; defaults to the pinned S5-nano tokenizer"
    )
    p.add_argument("--val-fraction", type=float, default=0.01)
    p.add_argument("--dtype", choices=["uint32", "uint16"])
    p.add_argument("--input", nargs="*")
    p.add_argument("--hf-dataset")
    p.add_argument("--hf-split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--chat-jsonl")
    p.add_argument("--max-docs", type=int)
    p.set_defaults(func=cmd_prepare)

    # train
    p = sub.add_parser("train", help="run training")
    p.add_argument("--config", help="JSON with 'model' and 'train' sections")
    p.add_argument("--preset", choices=PRESET_NAMES, default="tiny")
    p.add_argument("--out-dir", default="out")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--eval-every", type=int, default=1_000)
    p.add_argument("--eval-batches", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup-steps", type=int, default=2_000)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument(
        "--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false"
    )
    p.add_argument("--mtp-weight", type=float, default=0.3)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=5_000)
    p.add_argument("--keep-last", type=int, default=3)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        help="training precision (defaults to bfloat16, or the checkpoint dtype when resuming)",
    )
    p.add_argument("--torch-seed", type=int)
    p.add_argument("--resume")
    p.add_argument("--resume-step", type=int)
    p.set_defaults(func=cmd_train)

    # export
    p = sub.add_parser("export", help="write a trained model in HF layout")
    p.add_argument("--preset", choices=PRESET_NAMES, default="tiny")
    p.add_argument("--checkpoint", required=True, help="checkpoint directory or safetensors file")
    p.add_argument("--out", required=True)
    p.add_argument("--dtype-override", choices=["BF16", "F32", "F16"], default="native")
    p.add_argument("--assets-dir", help="optional directory with tokenizer and processor assets")
    p.set_defaults(func=cmd_export)

    # info
    p = sub.add_parser("info", help="parameter count and memory estimate")
    p.add_argument("--preset", choices=PRESET_NAMES, default="tiny")
    p.set_defaults(func=cmd_info)

    # generate
    p = sub.add_parser("generate", help="sample from a model")
    p.add_argument("--preset", choices=PRESET_NAMES, default="tiny")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", help="checkpoint directory or safetensors file")
    p.add_argument("--tokenizer", help="local tokenizer.json; defaults to the checkpoint tokenizer")
    p.add_argument("--prompt", default="")
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=cmd_generate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
