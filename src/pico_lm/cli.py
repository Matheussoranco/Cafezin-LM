"""Pico-LM command line: tokenizer / prepare / train / generate / chat / info / quantize."""
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
    "batch_size", "grad_accum_steps", "max_steps", "lr", "min_lr",
    "eval_interval", "eval_iters", "warmup_steps", "log_interval", "seed",
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
    if args.seq_len:
        train_cfg.seq_len = args.seq_len
    for name in TRAIN_OVERRIDES:
        v = getattr(args, name)
        if v is not None:
            setattr(train_cfg, name, v)
    train_cfg.compile = args.compile

    # explicit flag > tokenizer recorded by `prepare` > conventional location
    tokenizer_path = args.tokenizer or str(data_dir / meta.get("tokenizer", "tokenizer.json"))
    best = train(
        model_cfg,
        train_cfg,
        args.data,
        args.out,
        resume=args.resume,
        teacher_ckpt=args.teacher,
        tokenizer_path=tokenizer_path,
    )
    print(f"done; best val loss {best:.4f}; checkpoints in {args.out}")


def cmd_generate(args: argparse.Namespace) -> None:
    from .generate import complete, load_for_inference

    model, tok, device = load_for_inference(args.ckpt, args.tokenizer)
    text = complete(
        model, tok, args.prompt, device,
        max_new_tokens=args.tokens, temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p, repetition_penalty=args.rep_penalty,
    )
    print(args.prompt + text)


def cmd_chat(args: argparse.Namespace) -> None:
    from .generate import chat_repl, load_for_inference

    model, tok, device = load_for_inference(args.ckpt, args.tokenizer)
    chat_repl(
        model, tok, device, system=args.system,
        max_new_tokens=args.tokens, temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p, repetition_penalty=args.rep_penalty,
    )


def cmd_info(args: argparse.Namespace) -> None:
    from .train import load_model

    model, ckpt = load_model(args.ckpt, device="cpu")
    cfg = model.cfg
    best_val = ckpt.get("best_val")
    best_str = f"{best_val:.4f}" if isinstance(best_val, (int, float)) else "n/a"
    print(f"params:      {model.num_params()/1e6:.2f}M")
    print(
        f"arch:        dim={cfg.dim} layers={cfg.n_layers} "
        f"heads={cfg.n_heads} kv_heads={cfg.n_kv_heads}"
    )
    print(f"vocab:       {cfg.vocab_size}  max_seq_len: {cfg.max_seq_len}")
    print(f"step:        {ckpt.get('step')}  best_val: {best_str}")
    print(f"tokenizer:   {ckpt.get('tokenizer_path')}")


def cmd_quantize(args: argparse.Namespace) -> None:
    from .export import quantize_int8

    out = quantize_int8(args.ckpt, args.out)
    src = Path(args.ckpt).stat().st_size / 1e6
    dst = out.stat().st_size / 1e6
    print(f"quantized {src:.1f}MB -> {dst:.1f}MB at {out}")


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
    tr.add_argument("--teacher", help="checkpoint of a larger Pico-LM model to distill from")
    tr.add_argument("--compile", action="store_true")
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

    q = sub.add_parser("quantize", help="int8-quantize a checkpoint for CPU inference")
    q.add_argument("--ckpt", required=True)
    q.add_argument("--out", default="runs/quantized.pt")
    q.set_defaults(fn=cmd_quantize)

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
