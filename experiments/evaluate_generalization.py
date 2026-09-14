"""Seeded passkey retrieval and external-corpus loss with content provenance.

External text must be a UTF-8 JSONL file with one {"text": ...} document per line.
It is scored document by document, without training or joining document boundaries.
"""

import argparse
import json
import math
import random
import urllib.request
from pathlib import Path

import torch

from pico_lm.generate import load_for_inference
from pico_lm.provenance import manifest, tokenizer_hash


@torch.inference_mode()
def external_loss(model, tok, path, device, seq_len=512):
    if seq_len < 1:
        raise ValueError("external seq_len must be positive")
    seq_len = min(seq_len, model.cfg.max_seq_len)
    total, count, documents = 0.0, 0, 0
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            ids = tok.encode(json.loads(line)["text"], bos=True, eos=True)
            documents += 1
            for start in range(0, len(ids) - 1, seq_len):
                window = ids[start : start + seq_len + 1]
                x = torch.tensor([window[:-1]], device=device)
                y = torch.tensor([window[1:]], device=device)
                _, loss = model(x, y)
                total += loss.item() * y.numel()
                count += y.numel()
    if count == 0:
        raise ValueError("external corpus contains no targets")
    return {
        "loss": total / count,
        "perplexity": math.exp(total / count),
        "tokens": count,
        "documents": documents,
        "seq_len": seq_len,
        "protocol": "nonoverlapping windows within documents; all targets scored once",
    }


@torch.inference_mode()
def retrieval(model, tok, device, lengths, seeds, trials):
    results = []
    filler = tok.encode(" The garden has flowers and trees. The children play outside.")
    for seed in seeds:
        rng = random.Random(seed)
        for length in lengths:
            if length + 16 > model.cfg.max_seq_len:
                raise ValueError("retrieval length plus 16 output tokens exceeds model context")
            for trial in range(trials):
                key = f"{rng.randrange(100000):05d}"
                needle = tok.encode(f" The secret code is {key}. Remember this code. ")
                suffix = tok.encode(" What is the secret code? The secret code is")
                budget = length - 1 - len(needle) - len(suffix)
                if budget < 0:
                    raise ValueError("retrieval length is too short")
                background = (filler * (budget // len(filler) + 1))[:budget]
                fraction = (0.1, 0.5, 0.9)[trial % 3]
                offset = int(budget * fraction)
                ids = [tok.bos_id, *background[:offset], *needle, *background[offset:], *suffix]
                y = model.generate(
                    torch.tensor([ids], device=device), 16, temperature=0, eos_id=tok.eos_id
                )
                output = tok.decode(y[0, len(ids) :].tolist()).strip()
                # Prefix match permits punctuation after the code, but not extra digits.
                hit = output.startswith(key) and (len(output) == 5 or not output[5].isdigit())
                results.append(
                    {
                        "seed": seed,
                        "length": length,
                        "trial": trial,
                        "needle_offset": offset + 1,
                        "expected": key,
                        "output": output,
                        "correct": hit,
                        "prompt_ids": ids,
                    }
                )
    return {
        "accuracy": sum(r["correct"] for r in results) / len(results),
        "n": len(results),
        "chance": 0.00001,
        "items": results,
        "interpretation": "synthetic base-model completion probe; not general reasoning",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--external-jsonl")
    ap.add_argument("--external-url", help="HTTPS UTF-8 plain-text source; saves a bounded sample")
    ap.add_argument("--external-chars", type=int, default=20000)
    ap.add_argument("--external-seq-len", type=int, default=512)
    ap.add_argument("--lengths", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.trials < 1:
        ap.error("trials must be positive")
    if args.external_url:
        if args.external_jsonl or not args.external_url.startswith("https://"):
            ap.error("use either external-jsonl or an HTTPS external-url")
        if args.external_chars < 1:
            ap.error("external-chars must be positive")
        with urllib.request.urlopen(args.external_url, timeout=30) as response:
            text = response.read(2_000_000).decode("utf-8-sig")
        if "*** START OF" in text:
            text = text.split("*** START OF", 1)[1].split("***", 1)[1]
        text = text[: args.external_chars]
        sample = Path(args.out).with_suffix(".external.jsonl")
        sample.parent.mkdir(parents=True, exist_ok=True)
        sample.write_text(json.dumps({"text": text}) + "\n", encoding="utf-8")
        args.external_jsonl = str(sample)
    model, tok, device = load_for_inference(args.ckpt, device=args.device)
    report = {
        "provenance": manifest(
            [args.ckpt, args.external_jsonl, __file__],
            **vars(args),
            tokenizer_sha256=tokenizer_hash(tok),
        )
    }
    report["retrieval"] = retrieval(model, tok, device, args.lengths, args.seeds, args.trials)
    if args.external_jsonl:
        report["external"] = external_loss(
            model, tok, args.external_jsonl, device, args.external_seq_len
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: {a: b for a, b in v.items() if a != "items"}
                for k, v in report.items()
                if k != "provenance"
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
