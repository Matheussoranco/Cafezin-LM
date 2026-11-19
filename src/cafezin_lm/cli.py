"""Cafezin LM command line interface.

Commands:
  spec             write a fresh spec.json from a built model
  verify           compare a model against the reference fingerprint (offline)
  prepare          tokenize a corpus into train/val memmaps
  train            run training (resumable)
  export           write a trained model in the HF layout (safetensors + index + config)
  info             parameter count and memory estimate
  generate         sample from a checkpoint (or a fresh model)
  tokenizer-extend extend tokenizer with new tokens and grow embeddings
  tokenizer-stats  measure tokenization statistics on a corpus
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
from .tokenizer_extension import (
    compute_tokenization_gain,
    extend_tokenizer,
    grow_embeddings,
    measure_tokenization_stats,
    mine_candidate_tokens,
)
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


def cmd_tokenizer_stats(args: argparse.Namespace) -> int:
    tok = Tokenizer(args.tokenizer)
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    texts = corpus_path.read_text(encoding="utf-8").split("\n\n")
    texts = [t.strip() for t in texts if t.strip()]

    stats = measure_tokenization_stats(tok, texts)
    print(f"Total chars (UTF-8 bytes): {stats['total_chars']:,}")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"Chars per token: {stats['chars_per_token']:.3f}")
    print(f"Tokens per char: {stats['tokens_per_char']:.5f}")
    return 0


def cmd_tokenizer_extend(args: argparse.Namespace) -> int:
    # Load base tokenizer
    base_tok = Tokenizer(args.tokenizer)

    # Read mining corpus
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    mining_texts = corpus_path.read_text(encoding="utf-8").split("\n\n")
    mining_texts = [t.strip() for t in mining_texts if t.strip()]

    print(f"Mining candidates from {len(mining_texts)} documents...")
    candidates = mine_candidate_tokens(
        base_tok,
        mining_texts,
        min_frequency=args.min_frequency,
        max_candidates=args.max_candidates,
    )
    print(f"Found {len(candidates)} candidate tokens")

    if not candidates:
        print("No candidates found meeting frequency threshold")
        return 0

    # Show top candidates
    print("\nTop 20 candidates (token, estimated gain):")
    for token, gain in candidates[:20]:
        print(f"  {token!r}: {gain}")

    # Read heldout corpus if provided
    heldout_texts = None
    if args.heldout:
        heldout_path = Path(args.heldout)
        if not heldout_path.exists():
            raise FileNotFoundError(f"Heldout corpus not found: {heldout_path}")
        heldout_texts = heldout_path.read_text(encoding="utf-8").split("\n\n")
        heldout_texts = [t.strip() for t in heldout_texts if t.strip()]
        print(f"\nLoaded {len(heldout_texts)} heldout documents for gain measurement")

    if args.dry_run:
        if heldout_texts:
            # Simulate extension to measure gain
            new_tokens = [c[0] for c in candidates]
            # We can't easily simulate without actually extending, so just report candidates
            print("\nDry run: would add", len(new_tokens), "tokens")
        return 0

    # Extend tokenizer
    new_tokens = [c[0] for c in candidates]
    print(f"\nExtending tokenizer with {len(new_tokens)} new tokens...")
    extended_tok = extend_tokenizer(base_tok, new_tokens, args.out_tokenizer)
    print(f"Extended tokenizer saved to {args.out_tokenizer}")
    print(f"  Original vocab size: {base_tok.vocab_size}")
    print(f"  Extended vocab size: {extended_tok.vocab_size}")
    print(f"  New tokens added: {extended_tok.vocab_size - base_tok.vocab_size}")

    # Measure gain on heldout if provided
    if heldout_texts:
        print("\nMeasuring tokenization gain on heldout corpus...")
        gain = compute_tokenization_gain(base_tok, extended_tok, heldout_texts)
        print(f"  Original tokens: {gain['original_tokens']:,}")
        print(f"  Extended tokens: {gain['extended_tokens']:,}")
        print(f"  Token reduction: {gain['token_reduction']:,} ({gain['pct_reduction']:.2f}%)")
        print(f"  Original chars/token: {gain['original_chars_per_token']:.3f}")
        print(f"  Extended chars/token: {gain['extended_chars_per_token']:.3f}")

    # Grow checkpoint embeddings if requested
    if args.out_checkpoint:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required when --out-checkpoint is specified")
        print(f"\nGrowing embeddings for checkpoint at {args.checkpoint}...")

        # Load checkpoint model
        checkpoint_path = Path(args.checkpoint)
        model = _checkpoint_model(checkpoint_path, "base")

        # Grow embeddings
        grow_embeddings(model, base_tok, extended_tok, new_tokens)

        # Export extended checkpoint
        from .checkpoint import save_hf_checkpoint, checkpoint_assets
        assets = {"tokenizer.json": extended_tok.path.read_bytes()}
        manifest = save_hf_checkpoint(
            model=model,
            out_dir=args.out_checkpoint,
            cfg=model.cfg,
            tokenizer_files=assets,
        )
        print(f"Extended checkpoint saved to {args.out_checkpoint}")
        print(f"  Shards: {len(manifest['shards'])}, Tensors: {manifest['tensors']}")

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

    # tokenizer-extend
    p = sub.add_parser(
        "tokenizer-extend", help="extend tokenizer with new tokens and grow embeddings"
    )
    p.add_argument("--tokenizer", required=True, help="base tokenizer.json to extend")
    p.add_argument("--corpus", required=True, help="corpus file(s) to mine candidates from")
    p.add_argument("--heldout", help="held-out corpus file for gain measurement")
    p.add_argument("--out-tokenizer", required=True, help="output path for extended tokenizer.json")
    p.add_argument("--out-checkpoint", help="output directory for extended checkpoint (optional)")
    p.add_argument("--checkpoint", help="checkpoint directory to grow embeddings for (required with --out-checkpoint)")
    p.add_argument("--min-frequency", type=int, default=10, help="minimum frequency for candidate tokens")
    p.add_argument("--max-candidates", type=int, default=500, help="maximum number of candidate tokens to add")
    p.add_argument("--dry-run", action="store_true", help="report candidates and gains without writing files")
    p.set_defaults(func=cmd_tokenizer_extend)

    # tokenizer-stats
    p = sub.add_parser("tokenizer-stats", help="measure tokenization statistics on a corpus")
    p.add_argument("--tokenizer", required=True, help="tokenizer.json to evaluate")
    p.add_argument("--corpus", required=True, help="corpus file to measure")
    p.set_defaults(func=cmd_tokenizer_stats)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
