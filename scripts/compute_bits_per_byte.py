#!/usr/bin/env python
"""Compute bits-per-UTF-8-byte metric for Portuguese text.

This script computes the bits-per-UTF-8-byte metric as specified in the
acceptance criteria. It uses loglikelihood_rolling to compute per-token
perplexity, then converts to bits per UTF-8 byte.

The metric is computed as:
    bits_per_utf8_byte = (log_perplexity * log2(e)) / avg_utf8_bytes_per_token

Where:
- log_perplexity is the natural log of perplexity from loglikelihood_rolling
- avg_utf8_bytes_per_token is computed from the tokenizer on the evaluation corpus

This accounts for the fact that Portuguese tokens may encode different numbers
of UTF-8 bytes, making the metric comparable across tokenizers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer as HFTokenizer

from cafezin_lm import Tokenizer, build_model, tiny_config


def load_portuguese_corpus(data_path: str, max_chars: int | None = None) -> str:
    """Load Portuguese evaluation corpus."""
    path = Path(data_path)
    if path.is_file():
        text = path.read_text(encoding="utf-8")
    elif path.is_dir():
        # Concatenate all .txt files
        texts = []
        for txt_file in path.glob("*.txt"):
            texts.append(txt_file.read_text(encoding="utf-8"))
        text = "\n".join(texts)
    else:
        raise ValueError(f"Data path not found: {data_path}")

    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
    return text


def compute_utf8_bytes_per_token(tokenizer: Tokenizer, text: str) -> float:
    """Compute average UTF-8 bytes per token for the given text."""
    ids = tokenizer.encode(text, eos=False)
    if not ids:
        return 1.0

    # Decode each token individually and measure UTF-8 bytes
    total_bytes = 0
    for token_id in ids:
        token_text = tokenizer.decode([token_id])
        total_bytes += len(token_text.encode("utf-8"))

    return total_bytes / len(ids)


def compute_loglikelihood_rolling(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    text: str,
    max_seq_len: int = 2048,
    device: str = "cuda",
) -> float:
    """Compute log-likelihood rolling perplexity on text.

    This mimics lm_eval's loglikelihood_rolling approach: slide a window
    over the tokenized text and compute the likelihood of each token
    given the previous context.
    """
    model.eval()
    model.to(device)

    ids = tokenizer.encode(text, eos=False)
    if len(ids) < 2:
        return float("inf")

    # Convert to tensor
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    # Process in chunks of max_seq_len
    total_log_likelihood = 0.0
    total_tokens = 0

    with torch.no_grad():
        for start in range(0, input_ids.size(1) - 1, max_seq_len):
            end = min(start + max_seq_len, input_ids.size(1))
            chunk = input_ids[:, start:end]

            # Forward pass
            logits, _ = model(chunk[:, :-1])  # Predict next token
            targets = chunk[:, 1:]

            # Compute log-likelihood
            log_probs = torch.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

            total_log_likelihood += token_log_probs.sum().item()
            total_tokens += targets.numel()

    # Average negative log-likelihood (cross-entropy)
    avg_nll = -total_log_likelihood / total_tokens
    perplexity = torch.exp(torch.tensor(avg_nll)).item()

    return perplexity


def compute_bits_per_utf8_byte(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    text: str,
    device: str = "cuda",
) -> dict[str, float]:
    """Compute bits-per-UTF-8-byte metric."""
    # Get perplexity
    perplexity = compute_loglikelihood_rolling(model, tokenizer, text, device=device)

    # Get average UTF-8 bytes per token
    avg_bytes_per_token = compute_utf8_bytes_per_token(tokenizer, text)

    # Convert to bits per UTF-8 byte
    # perplexity = exp(cross_entropy) where cross_entropy is in nats
    # bits_per_token = cross_entropy * log2(e) = ln(perplexity) * log2(e)
    # bits_per_utf8_byte = bits_per_token / avg_bytes_per_token
    import math
    bits_per_token = math.log(perplexity) * math.log2(math.e)
    bits_per_utf8_byte = bits_per_token / avg_bytes_per_token

    return {
        "perplexity": perplexity,
        "avg_utf8_bytes_per_token": avg_bytes_per_token,
        "bits_per_token": bits_per_token,
        "bits_per_utf8_byte": bits_per_utf8_byte,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute bits-per-UTF-8-byte metric")
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer-path", required=True, help="Path to tokenizer.json")
    parser.add_argument("--data-path", required=True, help="Path to Portuguese corpus")
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--max-chars", type=int, help="Maximum characters to evaluate")
    parser.add_argument("--config", choices=["tiny", "base"], default="tiny", help="Model config")

    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = Tokenizer(args.tokenizer_path)

    # Load model
    print(f"Loading model from {args.model_path}")
    if args.config == "tiny":
        cfg = tiny_config()
    else:
        from cafezin_lm import base_config
        cfg = base_config()

    model = build_model(cfg)
    state_dict = torch.load(args.model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)

    # Load corpus
    print(f"Loading corpus from {args.data_path}")
    text = load_portuguese_corpus(args.data_path, args.max_chars)
    print(f"Corpus length: {len(text)} chars")

    # Compute metric
    print("Computing bits-per-UTF-8-byte...")
    metrics = compute_bits_per_utf8_byte(model, tokenizer, text, device=args.device)

    print("\n=== Results ===")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())