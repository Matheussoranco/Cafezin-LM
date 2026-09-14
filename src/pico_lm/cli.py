"""Pico-LM command line.

tokenizer / prepare / train / eval / generate / chat / info / presets / runs / quantize.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _source_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input", nargs="*", help="text files or globs (e.g. corpus/*.txt)")
    p.add_argument("--hf", help="HuggingFace dataset name (e.g. roneneldan/TinyStories)")
    p.add_argument("--hf-split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--chat-jsonl", help='JSONL with {"messages": [{"role", "content"}, ...]}')
    p.add_argument("--max-docs", type=int, help="cap the number of documents")


def _source_kwargs(args: argparse.Namespace) -> dict:
    return dict(
        inputs=args.input,
        hf_dataset=args.hf,
        hf_split=args.hf_split,
        text_field=args.text_field,
        chat_jsonl=args.chat_jsonl,
        max_docs=args.max_docs,
    )


def cmd_tokenizer(args: argparse.Namespace) -> None:
    from .data import iter_texts
    from .tokenizer import PicoTokenizer

    tok = PicoTokenizer.train(iter_texts(**_source_kwargs(args)), args.vocab_size, args.out)
    print(f"trained tokenizer: vocab={tok.vocab_size} -> {args.out}")


def cmd_prepare(args: argparse.Namespace) -> None:
    from .data import prepare
    from .tokenizer import PicoTokenizer

    tok = PicoTokenizer.load(args.tokenizer)
    meta = prepare(tok, args.out, val_fraction=args.val_fraction, **_source_kwargs(args))
    print(json.dumps(meta, indent=2))


TRAIN_OVERRIDES = (
    "batch_size",
    "grad_accum_steps",
    "max_steps",
    "lr",
    "min_lr",
    "eval_interval",
    "eval_iters",
    "warmup_steps",
    "log_interval",
    "seed",
)


def cmd_train(args: argparse.Namespace) -> None:
    from .config import TrainConfig, preset
    from .train import train

    data_dir = Path(args.data)
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} not found — run `pico-lm prepare --out {args.data}` first")

    model_cfg = preset(args.preset)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model_cfg.vocab_size = meta["vocab_size"]
    if args.seq_len:
        model_cfg.max_seq_len = args.seq_len

    train_cfg = TrainConfig()
    if args.resume and (Path(args.out) / "last.pt").exists():
        import torch

        from .config import ModelConfig

        saved = torch.load(Path(args.out) / "last.pt", map_location="cpu", weights_only=True)
        model_cfg = ModelConfig.from_dict(saved["model_cfg"])
        train_cfg = TrainConfig.from_dict(saved["train_cfg"])
    if args.seq_len:
        train_cfg.seq_len = args.seq_len
    for name in TRAIN_OVERRIDES:
        v = getattr(args, name)
        if v is not None:
            setattr(train_cfg, name, v)
    if args.compile is not None:
        train_cfg.compile = args.compile
    if args.gradient_checkpointing is not None:
        train_cfg.gradient_checkpointing = args.gradient_checkpointing

    # explicit flag > tokenizer recorded by `prepare` > conventional location
    tokenizer_path = args.tokenizer or str(data_dir / meta.get("tokenizer", "tokenizer.json"))
    best = train(
        model_cfg,
        train_cfg,
        args.data,
        args.out,
        resume=args.resume,
        init_from=args.init_from,
        teacher_ckpt=args.teacher,
        tokenizer_path=tokenizer_path,
    )
    print(f"done; best val loss {best:.4f}; checkpoints in {args.out}")


def cmd_generate(args: argparse.Namespace) -> None:
    from .generate import complete, load_for_inference

    model, tok, device = load_for_inference(args.ckpt, args.tokenizer)
    text = complete(
        model,
        tok,
        args.prompt,
        device,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.rep_penalty,
    )
    print(args.prompt + text)


def cmd_chat(args: argparse.Namespace) -> None:
    from .generate import chat_repl, load_for_inference

    model, tok, device = load_for_inference(args.ckpt, args.tokenizer)
    chat_repl(
        model,
        tok,
        device,
        system=args.system,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.rep_penalty,
    )


def cmd_info(args: argparse.Namespace) -> None:
    from .train import load_model

    model, ckpt = load_model(args.ckpt, device="cpu")
    cfg = model.cfg
    best_val = ckpt.get("best_val")
    best_str = f"{best_val:.4f}" if isinstance(best_val, (int, float)) else "n/a"
    print(f"params:      {model.num_params() / 1e6:.2f}M")
    print(
        f"arch:        dim={cfg.dim} layers={cfg.n_layers} "
        f"heads={cfg.n_heads} kv_heads={cfg.n_kv_heads}"
    )
    print(f"vocab:       {cfg.vocab_size}  max_seq_len: {cfg.max_seq_len}")
    print(f"step:        {ckpt.get('step')}  best_val: {best_str}")
    print(f"tokenizer:   {ckpt.get('tokenizer_path')}")


SFT_OVERRIDES = (
    "batch_size",
    "grad_accum_steps",
    "max_steps",
    "lr",
    "min_lr",
    "warmup_steps",
    "eval_interval",
    "eval_iters",
    "log_interval",
    "max_len",
    "bench_examples",
    "replay_weight",
    "seed",
)


def cmd_prepare_sft(args: argparse.Namespace) -> None:
    from .sft import prepare_sft
    from .tokenizer import PicoTokenizer

    tok = PicoTokenizer.load(args.tokenizer)
    meta = prepare_sft(
        args.jsonl,
        tok,
        args.out,
        val_fraction=args.val_fraction,
        max_len=args.max_len,
        seed=args.seed,
    )
    print(json.dumps(meta, indent=2))


def cmd_finetune(args: argparse.Namespace) -> None:
    from .config import SFTConfig
    from .sft import finetune

    cfg = SFTConfig()
    if args.resume and (Path(args.out) / "last.pt").exists():
        import torch

        saved = torch.load(Path(args.out) / "last.pt", map_location="cpu", weights_only=True)
        cfg = SFTConfig.from_dict(saved["sft_cfg"])
    for name in SFT_OVERRIDES:
        v = getattr(args, name, None)
        if v is not None:
            setattr(cfg, name, v)
    if args.select_by is not None:
        cfg.select_by = args.select_by
    if args.compile is not None:
        cfg.compile = args.compile
    if args.gradient_checkpointing is not None:
        cfg.gradient_checkpointing = args.gradient_checkpointing

    best = finetune(
        args.sft_data,
        args.out,
        cfg,
        init_from=args.init_from,
        resume=args.resume,
        tokenizer_path=args.tokenizer,
        bench_jsonl=args.bench_jsonl,
        replay_data=args.replay_data,
    )
    score = best.get("score", -1.0)
    tail = f"; best held-out exact match {score:.1%}" if score >= 0 else ""
    print(f"done; best val loss {best['loss']:.4f}{tail}; checkpoints in {args.out}")


def cmd_bench(args: argparse.Namespace) -> None:
    """Behavioural evaluation — the number that decides whether a model is good."""
    from .bench import format_report, run_bench
    from .generate import load_for_inference

    model, tok, device = load_for_inference(args.ckpt, args.tokenizer, device=args.device or None)
    val_bin = Path(args.data) / "val.bin" if args.data else None
    report = run_bench(
        model,
        tok,
        device,
        val_bin=val_bin,
        instructions=args.instructions,
        n_pairs=args.pairs,
        n_cloze=args.cloze,
        n_gen=args.gen,
        n_instruct=args.instruct_examples,
        seed=args.seed,
        speed=not args.no_speed,
    )
    report["checkpoint"] = str(args.ckpt)
    from .provenance import manifest, tokenizer_hash

    report["provenance"] = manifest(
        [args.ckpt, val_bin, args.instructions],
        seed=args.seed,
        model=model.cfg.to_dict(),
        device=device,
        tokenizer_sha256=tokenizer_hash(tok),
        pairs=args.pairs,
        cloze=args.cloze,
        gen=args.gen,
        instruct_examples=args.instruct_examples,
    )
    print(format_report(report, title=f"bench: {args.ckpt}"))

    if report.get("genhealth", {}).get("samples"):
        print("\nsample generations (temperature 0.8):")
        for s in report["genhealth"]["samples"][:3]:
            print(f"  > {s['prompt']!r}\n    {s['completion'][:220]!r}")
    if report.get("instruct", {}).get("samples"):
        print("\nsample instruction responses (greedy):")
        for s in report["instruct"]["samples"]:
            mark = "OK " if s["correct"] else "XX "
            print(f"  {mark} {s['instruction']!r}\n      want {s['expected']!r} got {s['got']!r}")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")


def cmd_quantize(args: argparse.Namespace) -> None:
    src_mb = Path(args.ckpt).stat().st_size / 1e6
    if args.method == "dynamic":
        from .export import quantize_int8

        out = quantize_int8(args.ckpt, args.out)
        print(f"dynamic int8 (CPU only): {src_mb:.1f}MB -> {out.stat().st_size / 1e6:.1f}MB")
        return

    from .quant import quantize_checkpoint

    out, meta = quantize_checkpoint(
        args.ckpt,
        args.out,
        bits=args.bits,
        group_size=args.group_size,
        quantize_embedding=args.quantize_embedding,
    )
    dst_mb = out.stat().st_size / 1e6
    print(f"int{meta['bits']} weight-only, group={meta['group_size']}, ", end="")
    print(f"embedding={'quantized' if meta['quantize_embedding'] else 'fp32'}")
    print(f"{src_mb:.1f}MB -> {dst_mb:.1f}MB  ({src_mb / max(dst_mb, 1e-9):.2f}x smaller)  {out}")
    print("\nsize alone proves nothing — measure it:")
    print(f"  pico-lm bench --ckpt {out} --data data")


def cmd_presets(args: argparse.Namespace) -> None:
    from .config import PRESETS, preset
    from .model import Transformer

    names = [args.name] if args.name else list(PRESETS)
    unknown = [n for n in names if n not in PRESETS]
    if unknown:
        raise SystemExit(f"unknown preset {unknown[0]!r}; choose from {sorted(PRESETS)}")

    if args.json:
        payload = {n: PRESETS[n].to_dict() for n in names}
        print(json.dumps(payload, indent=2))
        return

    header = f"{'preset':8s} {'params':>9s}  {'dim':>4s} {'layers':>6s} {'heads':>5s} "
    header += f"{'kv':>3s} {'seq':>5s}"
    print(header)
    print("-" * len(header))
    for name in names:
        cfg = preset(name)
        # Instantiate on the meta device: parameter count without allocating.
        try:
            import torch

            with torch.device("meta"):
                params = Transformer(cfg).num_params()
            params_str = f"{params / 1e6:.1f}M"
        except Exception:
            params_str = "?"
        print(
            f"{name:8s} {params_str:>9s}  {cfg.dim:>4d} {cfg.n_layers:>6d} "
            f"{cfg.n_heads:>5d} {cfg.n_kv_heads:>3d} {cfg.max_seq_len:>5d}"
        )
    if not args.name:
        print("\nTrain one with:  pico-lm train --preset micro --data data --out runs/micro")


def cmd_eval(args: argparse.Namespace) -> None:
    import torch

    from .config import TrainConfig
    from .data import TokenDataset
    from .provenance import manifest
    from .train import estimate_loss, load_model, pick_device, pick_dtype

    data_dir = Path(args.data)
    device = args.device or pick_device()
    model, ckpt = load_model(args.ckpt, device=device)
    model.eval()

    cfg = TrainConfig()
    cfg.batch_size = args.batch_size
    cfg.eval_iters = args.iters
    cfg.seq_len = min(args.seq_len or model.cfg.max_seq_len, model.cfg.max_seq_len)

    datasets: dict[str, TokenDataset] = {}
    for split in ("train", "val"):
        bin_path = data_dir / f"{split}.bin"
        if bin_path.exists():
            datasets[split] = TokenDataset(bin_path)
    if not datasets:
        raise SystemExit(f"no train.bin/val.bin under {data_dir} — run `pico-lm prepare` first")

    dtype = pick_dtype(device)
    cfg.validate()
    torch.manual_seed(args.seed)
    losses = estimate_loss(model, datasets, cfg, device, dtype)

    import math

    print(f"checkpoint:  {args.ckpt}")
    print(f"device:      {device}  seq_len={cfg.seq_len}  iters={cfg.eval_iters}")
    print(f"step:        {ckpt.get('step')}")
    for split, loss in losses.items():
        print(f"{split + ' loss:':12s} {loss:.4f}   perplexity: {math.exp(min(loss, 20)):.2f}")

    if args.json:
        print(
            json.dumps(
                {
                    "checkpoint": str(args.ckpt),
                    "step": ckpt.get("step"),
                    "provenance": manifest(
                        [args.ckpt, data_dir / "train.bin", data_dir / "val.bin"],
                        seed=args.seed,
                        config=cfg.to_dict(),
                        model=model.cfg.to_dict(),
                        device=device,
                    ),
                    "losses": losses,
                    "perplexity": {k: math.exp(min(v, 20)) for k, v in losses.items()},
                },
                indent=2,
            )
        )


def cmd_runs(args: argparse.Namespace) -> None:
    runs_dir = Path(args.dir)
    if not runs_dir.is_dir():
        raise SystemExit(f"{runs_dir} not found")

    # A run is any directory holding a checkpoint.
    runs = sorted(d for d in runs_dir.iterdir() if d.is_dir() and any(d.glob("*.pt")))
    if not runs:
        print(f"no runs under {runs_dir} (a run is a directory containing *.pt)")
        return

    print(f"{'run':16s} {'ckpts':>5s} {'size':>9s} {'step':>8s} {'best_val':>9s}  modified")
    print("-" * 72)
    for run in runs:
        ckpts = sorted(run.glob("*.pt"))
        size_mb = sum(c.stat().st_size for c in ckpts) / 1e6
        step, best = "?", "?"
        # Read the metadata from the best checkpoint without building the model.
        probe = run / "best.pt"
        if not probe.exists():
            probe = ckpts[0]
        try:
            import torch

            meta = torch.load(probe, map_location="cpu", weights_only=True)
            step = str(meta.get("step", "?"))
            best_val = meta.get("best_val")
            best = f"{best_val:.4f}" if isinstance(best_val, (int, float)) else "?"
        except Exception:
            pass
        import datetime

        mtime = datetime.datetime.fromtimestamp(probe.stat().st_mtime)
        print(
            f"{run.name:16s} {len(ckpts):>5d} {size_mb:>8.1f}M {step:>8s} {best:>9s}  "
            f"{mtime:%Y-%m-%d %H:%M}"
        )

    log = runs_dir / "logs"
    if log.is_dir():
        print(f"\nlogs: {log} ({len(list(log.iterdir()))} files)")


def build_parser() -> argparse.ArgumentParser:
    from .config import PRESETS

    p = argparse.ArgumentParser(
        prog="pico-lm", description="train and run a small language model locally"
    )
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("tokenizer", help="train a BPE tokenizer on your corpus")
    _source_args(t)
    t.add_argument("--vocab-size", type=int, default=16384)
    t.add_argument("--out", default="data/tokenizer.json")
    t.set_defaults(fn=cmd_tokenizer)

    pr = sub.add_parser("prepare", help="tokenize the corpus into train/val bins")
    _source_args(pr)
    pr.add_argument("--tokenizer", default="data/tokenizer.json")
    pr.add_argument("--out", default="data")
    pr.add_argument("--val-fraction", type=float, default=0.01)
    pr.set_defaults(fn=cmd_prepare)

    tr = sub.add_parser("train", help="train a model")
    tr.add_argument("--preset", default="small", choices=sorted(PRESETS))
    tr.add_argument("--data", default="data")
    tr.add_argument("--out", default="runs/small")
    tr.add_argument("--tokenizer", help="tokenizer to copy into the run dir")
    tr.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    tr.add_argument(
        "--init-from",
        help="start with compatible weights but a fresh optimizer (supports context extension)",
    )
    tr.add_argument("--teacher", help="checkpoint of a larger Pico-LM model to distill from")
    tr.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)
    tr.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="recompute transformer blocks to reduce activation memory (slower)",
    )
    tr.add_argument("--seq-len", type=int)
    tr.add_argument("--batch-size", type=int)
    tr.add_argument("--grad-accum-steps", type=int)
    tr.add_argument("--max-steps", type=int)
    tr.add_argument("--lr", type=float)
    tr.add_argument("--min-lr", type=float, help="cosine floor (default 6e-5)")
    tr.add_argument("--warmup-steps", type=int)
    tr.add_argument("--eval-interval", type=int)
    tr.add_argument("--eval-iters", type=int)
    tr.add_argument("--log-interval", type=int)
    tr.add_argument("--seed", type=int)
    tr.set_defaults(fn=cmd_train)

    def _gen_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--ckpt", required=True)
        sp.add_argument("--tokenizer")
        sp.add_argument("--tokens", type=int, default=200)
        sp.add_argument("--temperature", type=float, default=0.8)
        sp.add_argument("--top-k", type=int, default=50)
        sp.add_argument("--top-p", type=float, default=0.95)
        sp.add_argument("--rep-penalty", type=float, default=1.1)

    g = sub.add_parser("generate", help="complete a prompt")
    _gen_args(g)
    g.add_argument("--prompt", required=True)
    g.set_defaults(fn=cmd_generate)

    c = sub.add_parser("chat", help="interactive chat REPL")
    _gen_args(c)
    c.add_argument("--system", help="system prompt")
    c.set_defaults(fn=cmd_chat)

    i = sub.add_parser("info", help="inspect a checkpoint")
    i.add_argument("--ckpt", required=True)
    i.set_defaults(fn=cmd_info)

    ps = sub.add_parser("prepare-sft", help="tokenize instruction pairs for fine-tuning")
    ps.add_argument("--jsonl", required=True, help='{"instruction","response"} or {"messages"}')
    ps.add_argument("--tokenizer", default="data/tokenizer.json", help="MUST match pretraining")
    ps.add_argument("--out", default="data_sft")
    ps.add_argument("--val-fraction", type=float, default=0.05)
    ps.add_argument("--max-len", type=int, default=512)
    ps.add_argument("--seed", type=int, default=0)
    ps.set_defaults(fn=cmd_prepare_sft)

    ft = sub.add_parser("finetune", help="supervised fine-tune a pretrained checkpoint")
    ft.add_argument("--sft-data", default="data_sft", help="dir holding sft_train/val.npz")
    ft.add_argument("--out", default="runs/sft")
    ft.add_argument(
        "--init-from", help="pretrained checkpoint to start from (required unless --resume)"
    )
    ft.add_argument("--resume", action="store_true", help="continue an interrupted SFT run")
    ft.add_argument("--tokenizer")
    ft.add_argument(
        "--bench-jsonl",
        help="held-out instruction JSONL scored by exact match at every eval; "
        "without it, checkpoint selection falls back to loss",
    )
    ft.add_argument(
        "--select-by",
        default=None,
        choices=("score", "loss"),
        help="metric that decides best.pt (default: the behavioural score)",
    )
    ft.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)
    ft.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="recompute transformer blocks to reduce activation memory",
    )
    ft.add_argument(
        "--replay-data",
        help="pretraining data dir containing train.bin; mixes LM loss to reduce forgetting",
    )
    ft.add_argument(
        "--replay-weight",
        type=float,
        default=None,
        help="fraction of the SFT objective assigned to replay loss (default 0.1)",
    )
    ft.add_argument("--batch-size", type=int)
    ft.add_argument("--grad-accum-steps", type=int)
    ft.add_argument("--max-steps", type=int)
    ft.add_argument("--lr", type=float)
    ft.add_argument("--min-lr", type=float)
    ft.add_argument("--warmup-steps", type=int)
    ft.add_argument("--eval-interval", type=int)
    ft.add_argument("--eval-iters", type=int)
    ft.add_argument("--log-interval", type=int)
    ft.add_argument("--max-len", type=int)
    ft.add_argument("--bench-examples", type=int)
    ft.add_argument("--seed", type=int)
    ft.set_defaults(fn=cmd_finetune)

    b = sub.add_parser("bench", help="behavioural evaluation (task scores, not loss)")
    b.add_argument("--ckpt", required=True)
    b.add_argument("--tokenizer")
    b.add_argument("--data", default="data", help="dir holding val.bin ('' to skip corpus probes)")
    b.add_argument("--instructions", help="held-out instruction JSONL for exact-match scoring")
    b.add_argument("--pairs", type=int, default=400, help="minimal-pair items")
    b.add_argument("--cloze", type=int, default=200, help="cloze items")
    b.add_argument("--gen", type=int, default=24, help="free-generation prompts")
    b.add_argument("--instruct-examples", type=int, default=200)
    b.add_argument("--device", default="")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--no-speed", action="store_true")
    b.add_argument("--json", help="also write the full report here")
    b.set_defaults(fn=cmd_bench)

    q = sub.add_parser("quantize", help="post-training quantization of a checkpoint")
    q.add_argument("--ckpt", required=True)
    q.add_argument("--out", default="runs/quantized.pt")
    q.add_argument("--bits", type=int, default=8, choices=(4, 8))
    q.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="weights per scale along the input dim; 0 = per output channel",
    )
    q.add_argument(
        "--quantize-embedding",
        action="store_true",
        help="also quantize the (tied) embedding — most of the parameters in these "
        "presets, and most of the quality loss",
    )
    q.add_argument(
        "--method",
        default="weight-only",
        choices=("weight-only", "dynamic"),
        help="weight-only: any device, smaller on disk. dynamic: torch int8, CPU only",
    )
    q.set_defaults(fn=cmd_quantize)

    pp = sub.add_parser("presets", help="list model presets and their parameter counts")
    pp.add_argument("name", nargs="?", default="", help=f"show one preset ({', '.join(PRESETS)})")
    pp.add_argument("--json", action="store_true")
    pp.set_defaults(fn=cmd_presets)

    e = sub.add_parser("eval", help="measure loss / perplexity of a checkpoint on a dataset")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--data", default="data", help="directory holding train.bin / val.bin")
    e.add_argument("--iters", type=int, default=100, help="batches per split")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--batch-size", type=int, default=16)
    e.add_argument("--seq-len", type=int, default=0, help="0 = the model's max_seq_len")
    e.add_argument("--device", default="", help="cuda / cpu (default: auto)")
    e.add_argument("--json", action="store_true")
    e.set_defaults(fn=cmd_eval)

    r = sub.add_parser("runs", help="list training runs and their checkpoints")
    r.add_argument("--dir", default="runs", help="runs directory")
    r.set_defaults(fn=cmd_runs)

    return p


def main(argv: list[str] | None = None) -> None:
    # Windows consoles default to cp1252; force UTF-8 so any corpus text prints
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
