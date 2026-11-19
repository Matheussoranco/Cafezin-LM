"""Tokenizer extension infrastructure for Cafezin LM.

This module provides utilities to:
1. Mine candidate tokens from a corpus (e.g., Brazilian Portuguese text)
2. Extend a tokenizer with new tokens while preserving existing IDs
3. Initialize new embedding rows from mean decomposition under the original tokenizer
4. Export/import extended tokenizer + checkpoint pairs
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .tokenizer import Tokenizer

if TYPE_CHECKING:
    from .config import TextConfig


def count_token_frequencies(tokenizer: Tokenizer, texts: list[str]) -> Counter[int]:
    """Count token frequencies in a list of texts using the given tokenizer."""
    freq: Counter[int] = Counter()
    for text in texts:
        ids = tokenizer.encode(text, eos=False)
        freq.update(ids)
    return freq


def mine_candidate_tokens(
    tokenizer: Tokenizer,
    corpus_texts: list[str],
    min_frequency: int = 10,
    max_candidates: int = 1000,
) -> list[tuple[str, int]]:
    """Mine candidate tokens from a corpus that would improve tokenization.

    This uses the tokenizer's BPE model to find frequent character/byte sequences
    that are currently split into multiple tokens. Candidates are ranked by
    estimated tokenization gain (frequency * (num_subtokens - 1)).

    Args:
        tokenizer: The base tokenizer to analyze.
        corpus_texts: List of text documents to mine from.
        min_frequency: Minimum frequency for a candidate to be considered.
        max_candidates: Maximum number of candidates to return.

    Returns:
        List of (candidate_string, estimated_gain) tuples, sorted by gain descending.
    """
    # Get the underlying HF tokenizer to access BPE internals
    hf_tok = tokenizer._tokenizer
    vocab = hf_tok.get_vocab(with_added_tokens=True)
    id_to_token = {v: k for k, v in vocab.items()}

    # Encode all texts and count subtoken sequences
    # For each position, look at n-grams of tokens that could be merged
    pair_counts: Counter[tuple[int, int]] = Counter()
    triplet_counts: Counter[tuple[int, int, int]] = Counter()

    for text in corpus_texts:
        ids = tokenizer.encode(text, eos=False)
        for i in range(len(ids) - 1):
            pair_counts[(ids[i], ids[i + 1])] += 1
        for i in range(len(ids) - 2):
            triplet_counts[(ids[i], ids[i + 1], ids[i + 2])] += 1

    # Score candidates: frequency * (num_tokens - 1) = tokenization gain
    candidates: list[tuple[str, int]] = []

    for (a, b), freq in pair_counts.items():
        if freq >= min_frequency:
            token_a = id_to_token.get(a, "")
            token_b = id_to_token.get(b, "")
            merged = token_a + token_b
            # Only consider if the merged form isn't already a single token
            if hf_tok.token_to_id(merged) is None:
                gain = freq * 1  # Merging 2 tokens into 1 saves 1 token per occurrence
                candidates.append((merged, gain))

    for (a, b, c), freq in triplet_counts.items():
        if freq >= min_frequency:
            token_a = id_to_token.get(a, "")
            token_b = id_to_token.get(b, "")
            token_c = id_to_token.get(c, "")
            merged = token_a + token_b + token_c
            if hf_tok.token_to_id(merged) is None:
                gain = freq * 2  # Merging 3 tokens into 1 saves 2 tokens per occurrence
                candidates.append((merged, gain))

    # Sort by gain descending and deduplicate
    seen = set()
    unique_candidates = []
    for cand, gain in sorted(candidates, key=lambda x: -x[1]):
        if cand not in seen:
            seen.add(cand)
            unique_candidates.append((cand, gain))
            if len(unique_candidates) >= max_candidates:
                break

    return unique_candidates


def measure_tokenization_stats(
    tokenizer: Tokenizer, texts: list[str]
) -> dict[str, float]:
    """Measure tokenization statistics on a list of texts.

    Returns:
        Dict with: total_chars, total_tokens, chars_per_token, tokens_per_char
    """
    total_chars = 0
    total_tokens = 0
    for text in texts:
        total_chars += len(text.encode("utf-8"))
        ids = tokenizer.encode(text, eos=False)
        total_tokens += len(ids)
    return {
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "chars_per_token": total_chars / max(1, total_tokens),
        "tokens_per_char": total_tokens / max(1, total_chars),
    }


def compute_tokenization_gain(
    original_tokenizer: Tokenizer,
    extended_tokenizer: Tokenizer,
    heldout_texts: list[str],
) -> dict[str, float]:
    """Compute tokenization gains and regressions on held-out texts.

    Args:
        original_tokenizer: The original (unextended) tokenizer.
        extended_tokenizer: The tokenizer with added tokens.
        heldout_texts: Evaluation texts (should not overlap with mining corpus).

    Returns:
        Dict with gain metrics.
    """
    orig_stats = measure_tokenization_stats(original_tokenizer, heldout_texts)
    ext_stats = measure_tokenization_stats(extended_tokenizer, heldout_texts)

    token_reduction = orig_stats["total_tokens"] - ext_stats["total_tokens"]
    pct_reduction = 100.0 * token_reduction / max(1, orig_stats["total_tokens"])

    return {
        "original_tokens": orig_stats["total_tokens"],
        "extended_tokens": ext_stats["total_tokens"],
        "token_reduction": token_reduction,
        "pct_reduction": pct_reduction,
        "original_chars_per_token": orig_stats["chars_per_token"],
        "extended_chars_per_token": ext_stats["chars_per_token"],
    }


def extend_tokenizer(
    tokenizer: Tokenizer,
    new_tokens: list[str],
    output_path: str | Path,
) -> Tokenizer:
    """Extend a tokenizer with new tokens, preserving all existing IDs.

    Args:
        tokenizer: The base tokenizer to extend.
        new_tokens: List of new token strings to add. Must not exist in vocab.
        output_path: Path to save the extended tokenizer.json.

    Returns:
        A new Tokenizer instance wrapping the extended tokenizer.
    """

    hf_tok = tokenizer._tokenizer
    vocab = hf_tok.get_vocab(with_added_tokens=True)

    # Verify no collisions
    for token in new_tokens:
        if token in vocab:
            raise ValueError(f"Token {token!r} already exists in vocabulary")

    # Add tokens using the HF tokenizer's add_tokens method
    # This preserves all existing token IDs and adds new ones at the end
    num_added = hf_tok.add_tokens(new_tokens)
    if num_added != len(new_tokens):
        raise RuntimeError(f"Expected to add {len(new_tokens)} tokens, added {num_added}")

    # Save the extended tokenizer
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hf_tok.save(str(output_path))

    # Return a new Tokenizer wrapper
    return Tokenizer(output_path, eos_token_id=tokenizer.eos_token_id)


def decompose_token(tokenizer: Tokenizer, token: str) -> list[int]:
    """Decompose a token string into its subtoken IDs under the original tokenizer.

    Args:
        tokenizer: The original (unextended) tokenizer.
        token: The token string to decompose.

    Returns:
        List of token IDs that the original tokenizer produces for this string.
    """
    ids = tokenizer.encode(token, eos=False)
    # Filter out any IDs that exceed the tokenizer's vocab size
    # This can happen with HF tokenizers that have byte fallback or special tokens
    max_id = tokenizer.max_token_id
    return [tid for tid in ids if tid <= max_id]


def mean_decomposition_init(
    original_embeddings: torch.Tensor,
    tokenizer: Tokenizer,
    new_tokens: list[str],
) -> torch.Tensor:
    """Initialize new embedding rows from mean decomposition under original tokenizer.

    For each new token, encode it with the original tokenizer to get a sequence
    of subtoken IDs, then average the corresponding embedding rows.

    Args:
        original_embeddings: The original embedding matrix (vocab_size, hidden_size).
        tokenizer: The original tokenizer (before extension).
        new_tokens: List of new token strings that were added.

    Returns:
        Tensor of shape (len(new_tokens), hidden_size) with initialized rows.
    """
    hidden_size = original_embeddings.shape[1]
    new_rows = torch.zeros(len(new_tokens), hidden_size, dtype=original_embeddings.dtype)

    for i, token in enumerate(new_tokens):
        subtoken_ids = decompose_token(tokenizer, token)
        if subtoken_ids:
            # Mean of subtoken embeddings
            new_rows[i] = original_embeddings[subtoken_ids].mean(dim=0)
        else:
            # Fallback: global mean of all embeddings
            new_rows[i] = original_embeddings.mean(dim=0)

    return new_rows


def grow_embeddings(
    model: torch.nn.Module,
    original_tokenizer: Tokenizer,
    extended_tokenizer: Tokenizer,
    new_tokens: list[str],
    embedding_attr: str = "model.language_model.embed_tokens",
) -> torch.nn.Module:
    """Grow the model's token embedding table for new tokens.

    Preserves existing embedding rows, initializes new rows from mean decomposition
    under the original tokenizer, and preserves the tied output projection (lm_head).

    Args:
        model: The model to modify in-place.
        original_tokenizer: The tokenizer before extension.
        extended_tokenizer: The tokenizer after extension.
        new_tokens: List of new token strings that were added.
        embedding_attr: Dot-path to the embedding module in the model.

    Returns:
        The modified model (same object).
    """
    # Navigate to the embedding module
    embed_module = model
    for part in embedding_attr.split("."):
        embed_module = getattr(embed_module, part)

    if not isinstance(embed_module, torch.nn.Embedding):
        raise TypeError(f"{embedding_attr} is not an nn.Embedding")

    # Use the model's actual embedding size as the old vocab size
    old_vocab_size = embed_module.weight.shape[0]
    new_vocab_size = extended_tokenizer.vocab_size

    # Verify consistency with tokenizer
    if original_tokenizer.vocab_size != old_vocab_size:
        raise ValueError(
            f"Tokenizer vocab size ({original_tokenizer.vocab_size}) "
            f"does not match model embedding size ({old_vocab_size})"
        )
    if new_vocab_size <= old_vocab_size:
        raise ValueError("Extended tokenizer vocab size must be larger than original")

    hidden_size = embed_module.weight.shape[1]
    device = embed_module.weight.device
    dtype = embed_module.weight.dtype

    # Check if lm_head is tied to embed_tokens BEFORE we replace the weight
    lm_head_tied = hasattr(model, "lm_head") and model.lm_head.weight is embed_module.weight

    # Create new embedding matrix
    new_weight = torch.empty(new_vocab_size, hidden_size, device=device, dtype=dtype)

    # Copy existing rows
    with torch.no_grad():
        new_weight[:old_vocab_size] = embed_module.weight.data

        # Initialize new rows from mean decomposition
        new_rows = mean_decomposition_init(embed_module.weight.data, original_tokenizer, new_tokens)
        new_weight[old_vocab_size:] = new_rows.to(device=device, dtype=dtype)

    # Replace the embedding weight
    embed_module.weight = torch.nn.Parameter(new_weight)

    # Update the tied lm_head if it shares the same weight
    if lm_head_tied:
        # lm_head was tied, update it to point to the new embedding weight
        model.lm_head.weight = embed_module.weight
        model.lm_head.out_features = new_vocab_size
    elif hasattr(model, "lm_head"):
        # Not tied but separate - need to grow it too
        lm_head = model.lm_head
        new_lm_weight = torch.empty(new_vocab_size, hidden_size, device=device, dtype=dtype)
        with torch.no_grad():
            new_lm_weight[:old_vocab_size] = lm_head.weight.data
            new_lm_weight[old_vocab_size:] = new_rows.to(device=device, dtype=dtype)
        lm_head.weight = torch.nn.Parameter(new_lm_weight)
        lm_head.out_features = new_vocab_size

    # Update config vocab_size
    if hasattr(model, "cfg"):
        model.cfg.text.vocab_size = new_vocab_size

    return model


def export_extended_checkpoint(
    model: torch.nn.Module,
    extended_tokenizer: Tokenizer,
    output_dir: str | Path,
    config: TextConfig | None = None,
    dtype_override: str | None = None,
) -> dict:
    """Export a model with extended tokenizer as a matched tokenizer/checkpoint pair.

    This saves the model weights and tokenizer together so they can be reloaded
    as a coherent unit.

    Args:
        model: The model with grown embeddings.
        extended_tokenizer: The extended tokenizer.
        output_dir: Directory to write the checkpoint.
        config: Model config (uses model.cfg if not provided).
        dtype_override: Optional dtype override for export.

    Returns:
        Export manifest dictionary.
    """
    from .checkpoint import save_hf_checkpoint

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save tokenizer as part of checkpoint assets
    tokenizer_bytes = extended_tokenizer.path.read_bytes()
    assets = {"tokenizer.json": tokenizer_bytes}

    # Use the model's config if not provided
    if config is None and hasattr(model, "cfg"):
        config = model.cfg

    manifest = save_hf_checkpoint(
        model=model,
        out_dir=output_dir,
        cfg=config,
        dtype_override=dtype_override,
        tokenizer_files=assets,
    )

    return manifest


def load_extended_checkpoint(
    checkpoint_dir: str | Path,
    model_class,
    config: TextConfig | None = None,
    device: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, Tokenizer]:
    """Load a checkpoint with its matched extended tokenizer.

    Args:
        checkpoint_dir: Directory containing the exported checkpoint.
        model_class: The model class to instantiate (e.g., CafezinLM).
        config: Optional config override (loads from checkpoint if not provided).
        device: Device to load the model on.

    Returns:
        Tuple of (model, tokenizer).
    """
    from .checkpoint import import_config, load_hf_checkpoint
    from .tokenizer import Tokenizer

    checkpoint_dir = Path(checkpoint_dir)

    # Load config
    if config is None:
        config = import_config(checkpoint_dir)

    # Load tokenizer
    tokenizer_path = checkpoint_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer.json not found in {checkpoint_dir}")
    tokenizer = Tokenizer(tokenizer_path)

    # Build model with extended vocab size
    model = model_class(config)
    model.to(device)

    # Load weights
    load_hf_checkpoint(model, checkpoint_dir, assign=True)

    return model, tokenizer


def verify_round_trip(
    original_tokenizer: Tokenizer,
    extended_tokenizer: Tokenizer,
    original_model: torch.nn.Module,
    extended_model: torch.nn.Module,
    test_texts: list[str],
    tolerance: float = 1e-5,
) -> dict[str, bool]:
    """Verify encode/decode and embedding-growth round trip.

    Checks:
    1. Extended tokenizer encodes/decodes original tokens identically
    2. New tokens encode to their new IDs and decode back
    3. Original embedding rows are preserved exactly
    4. New embedding rows match mean decomposition initialization
    5. Tied lm_head is preserved
    6. Forward pass on original vocabulary texts matches (within tolerance)

    Args:
        original_tokenizer: The tokenizer before extension.
        extended_tokenizer: The tokenizer after extension.
        original_model: The model before extension (for comparison).
        extended_model: The model after extension.
        test_texts: Texts to test forward pass consistency.
        tolerance: Numerical tolerance for forward pass comparison.

    Returns:
        Dict of check names to pass/fail booleans.
    """
    results = {}

    # 1. Original token IDs preserved
    orig_vocab = original_tokenizer._tokenizer.get_vocab(with_added_tokens=True)
    ext_vocab = extended_tokenizer._tokenizer.get_vocab(with_added_tokens=True)
    results["original_token_ids_preserved"] = all(
        ext_vocab.get(tok) == tid for tok, tid in orig_vocab.items()
    )

    # 2. Encode/decode consistency for original vocabulary texts
    encode_decode_ok = True
    for text in test_texts[:10]:  # Sample
        orig_ids = original_tokenizer.encode(text, eos=False)
        ext_ids = extended_tokenizer.encode(text, eos=False)
        if orig_ids != ext_ids:
            encode_decode_ok = False
            break
        # Decode should match
        if extended_tokenizer.decode(ext_ids) != original_tokenizer.decode(orig_ids):
            encode_decode_ok = False
            break
    results["encode_decode_consistency"] = encode_decode_ok

    # 3. New tokens encode to new IDs and decode back
    new_token_ids = set(ext_vocab.values()) - set(orig_vocab.values())
    new_token_decode_ok = True
    for token, tid in ext_vocab.items():
        if tid in new_token_ids:
            decoded = extended_tokenizer.decode([tid])
            if decoded != token:
                new_token_decode_ok = False
                break
    results["new_token_encode_decode"] = new_token_decode_ok

    # 4. Original embedding rows preserved
    orig_embed = original_model.model.language_model.embed_tokens.weight.data
    ext_embed = extended_model.model.language_model.embed_tokens.weight.data
    orig_vocab_size = original_tokenizer.vocab_size
    results["original_embeddings_preserved"] = torch.allclose(
        orig_embed, ext_embed[:orig_vocab_size], atol=tolerance
    )

    # 5. New embedding rows match mean decomposition
    new_tokens = [tok for tok, tid in ext_vocab.items() if tid >= orig_vocab_size]
    expected_new_rows = mean_decomposition_init(orig_embed, original_tokenizer, new_tokens)
    actual_new_rows = ext_embed[orig_vocab_size:]
    results["new_embeddings_match_mean_decomp"] = torch.allclose(
        expected_new_rows.to(device=actual_new_rows.device, dtype=actual_new_rows.dtype),
        actual_new_rows,
        atol=tolerance,
    )

    # 6. Tied lm_head preserved
    results["lm_head_tied"] = (
        extended_model.lm_head.weight is extended_model.model.language_model.embed_tokens.weight
    )

    # 7. Forward pass consistency on original vocabulary texts
    forward_ok = True
    original_model.eval()
    extended_model.eval()
    with torch.no_grad():
        for text in test_texts[:5]:
            ids = original_tokenizer.encode(text, eos=False)
            input_ids = torch.tensor([ids], dtype=torch.long)
            orig_logits, _ = original_model(input_ids)
            ext_logits, _ = extended_model(input_ids)
            # Compare only on original vocab logits
            if not torch.allclose(
                orig_logits[..., :orig_vocab_size],
                ext_logits[..., :orig_vocab_size],
                atol=tolerance,
            ):
                forward_ok = False
                break
    results["forward_pass_consistency"] = forward_ok

    return results
