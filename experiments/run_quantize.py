"""Measure post-training quantization: size, speed, and behavioural quality.

Run over two model sizes on purpose. The interesting result at this scale is not
"int4 is worse than int8" but *how much* worse, and how that depends on the
model being small — a 2.9M-parameter model has far less redundancy to spend on
rounding error than a 15.7M one.

Reports, per configuration:
  * weights-only checkpoint size on disk (the fp32 baseline is re-saved without
    optimizer state, otherwise the comparison is against a 3x larger file and
    every compression ratio is a lie)
  * decode tokens/sec
  * held-out behavioural scores: minimal pairs, 4-way cloze, generation health
  * val loss, for reference only

    python experiments/run_quantize.py --out results/quantization.json
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from pico_lm.bench import (
    build_cloze,
    build_minimal_pairs,
    generation_health,
    measure_speed,
    score_cloze,
    score_minimal_pairs,
    val_documents,
    val_loss,
)
from pico_lm.quant import quantize_model, save_quantized
from pico_lm.tokenizer import PicoTokenizer
from pico_lm.train import load_model

CONFIGS = [
    ("fp32", None),
    ("int8 g64 linear", {"bits": 8, "group_size": 64, "quantize_embedding": False}),
    ("int8 g64 +emb", {"bits": 8, "group_size": 64, "quantize_embedding": True}),
    ("int8 perchan +emb", {"bits": 8, "group_size": 0, "quantize_embedding": True}),
    ("int4 g64 linear", {"bits": 4, "group_size": 64, "quantize_embedding": False}),
    ("int4 g64 +emb", {"bits": 4, "group_size": 64, "quantize_embedding": True}),
    ("int4 g32 +emb", {"bits": 4, "group_size": 32, "quantize_embedding": True}),
    ("int4 perchan +emb", {"bits": 4, "group_size": 0, "quantize_embedding": True}),
]


def save_fp32_weights_only(model, ckpt, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_cfg": ckpt["model_cfg"],
            "step": ckpt.get("step"),
            "tokenizer_path": ckpt.get("tokenizer_path"),
        },
        path,
    )
    return path


def evaluate(model, tok, device, probes, n_gen: int) -> dict:
    model.to(device).eval()
    out = {
        "val_loss": val_loss(model, probes["val_bin"], device, iters=30)["loss"],
        "minimal_pairs": score_minimal_pairs(model, tok, probes["pairs"], device)["accuracy"],
        "cloze": score_cloze(model, tok, probes["cloze"], device)["accuracy"],
    }
    gh = generation_health(model, tok, probes["gen_prompts"][:n_gen], device, seed=0)
    out["distinct_2"] = gh["distinct_2"]
    out["repetition_4gram"] = gh["repetition_4gram"]
    out["sample"] = gh["samples"][0]["completion"][:200] if gh["samples"] else ""
    out["decode_tok_per_s"] = measure_speed(model, device, new_tokens=64, repeats=2)[
        "decode_tok_per_s"
    ]
    return out


def run_model(name: str, ckpt_path: str, tmp: Path, device: str, args) -> dict:
    print(f"\n{'=' * 70}\n{name}  ({ckpt_path})\n{'=' * 70}", flush=True)
    base, ckpt = load_model(ckpt_path, device="cpu")
    tok = PicoTokenizer.load(Path(ckpt_path).parent / "tokenizer.json")

    docs = val_documents(args.val_bin, tok, max_docs=1200)
    probes = {
        "val_bin": args.val_bin,
        "pairs": build_minimal_pairs(docs, args.pairs, seed=0),
        "cloze": build_cloze(docs, tok, args.cloze, seed=0),
        "gen_prompts": [" ".join(d.split()[:6]) for d in docs[: args.gen]],
    }

    rows = []
    for label, qcfg in CONFIGS:
        t0 = time.time()
        model = copy.deepcopy(base)
        path = tmp / f"{name}-{label.replace(' ', '_')}.pt"
        if qcfg is None:
            save_fp32_weights_only(model, ckpt, path)
        else:
            quantize_model(model, **qcfg)
            save_quantized(model, {**qcfg, "method": "weight-only-symmetric"}, ckpt, path)
        size_mb = path.stat().st_size / 1e6
        metrics = evaluate(model, tok, device, probes, args.gen)
        row = {"config": label, "size_mb": size_mb, **metrics}
        rows.append(row)
        print(
            f"{label:20s} {size_mb:7.2f} MB  loss {row['val_loss']:.4f}  "
            f"pairs {row['minimal_pairs']:.1%}  cloze {row['cloze']:.1%}  "
            f"d2 {row['distinct_2']:.3f}  {row['decode_tok_per_s']:.1f} tok/s  "
            f"[{time.time() - t0:.0f}s]",
            flush=True,
        )
        del model
        path.unlink(missing_ok=True)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return {"checkpoint": ckpt_path, "params_m": base.num_params() / 1e6, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/quantization.json")
    ap.add_argument("--val-bin", default="data/val.bin")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pairs", type=int, default=300)
    ap.add_argument("--cloze", type=int, default=150)
    ap.add_argument("--gen", type=int, default=8)
    ap.add_argument(
        "--models",
        nargs="*",
        default=["micro=runs/micro/best.pt", "nano=runs/nano-kd/best.pt"],
        help="name=checkpoint pairs",
    )
    args = ap.parse_args()

    tmp = Path("runs/_quant_tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    report = {"device": args.device, "models": {}}
    for spec in args.models:
        name, path = spec.split("=", 1)
        report["models"][name] = run_model(name, path, tmp, args.device, args)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
