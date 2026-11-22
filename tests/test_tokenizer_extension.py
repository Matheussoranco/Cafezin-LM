"""Tests for tokenizer extension infrastructure."""

import tempfile
from pathlib import Path

import pytest
import torch

from cafezin_lm import (
    CafezinLM,
    Tokenizer,
    build_model,
    compute_tokenization_gain,
    extend_tokenizer,
    grow_embeddings,
    measure_tokenization_stats,
    mine_candidate_tokens,
    tiny_config,
    verify_round_trip,
)


def test_tokenizer_stats_basic():
    """Test basic tokenization statistics measurement."""
    with tempfile.TemporaryDirectory() as tmp:
        # Create a simple tokenizer for testing
        from tokenizers import Tokenizer as HFTokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.trainers import BpeTrainer

        tok = HFTokenizer(BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=500, special_tokens=["[UNK]", "[EOS]"])
        tok.train_from_iterator(["hello world", "hello there", "world peace"], trainer=trainer)

        tok_path = f"{tmp}/tok.json"
        tok.save(tok_path)

        cafezin_tok = Tokenizer(tok_path)

        # Create test corpus
        corpus_path = f"{tmp}/corpus.txt"
        with open(corpus_path, "w") as f:
            f.write("hello world\n\nhello there\n\nworld peace")

        stats = measure_tokenization_stats(
            cafezin_tok, ["hello world", "hello there", "world peace"]
        )
        assert stats["total_chars"] > 0
        assert stats["total_tokens"] > 0
        assert stats["chars_per_token"] > 0


def test_mine_candidate_tokens():
    """Test candidate token mining from corpus."""
    with tempfile.TemporaryDirectory() as tmp:
        from tokenizers import Tokenizer as HFTokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.trainers import BpeTrainer

        tok = HFTokenizer(BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=500, special_tokens=["[UNK]", "[EOS]"])
        # Train on data where "hello" and "world" appear frequently together
        corpus = ["hello world " * 50, "hello there " * 30, "world peace " * 20]
        tok.train_from_iterator(corpus, trainer=trainer)

        tok_path = f"{tmp}/tok.json"
        tok.save(tok_path)

        cafezin_tok = Tokenizer(tok_path)

        candidates = mine_candidate_tokens(cafezin_tok, corpus, min_frequency=5, max_candidates=10)
        assert len(candidates) > 0
        # Check candidates are sorted by gain
        gains = [g for _, g in candidates]
        assert gains == sorted(gains, reverse=True)


def test_extend_tokenizer_preserves_ids():
    """Test that extending tokenizer preserves existing token IDs."""
    with tempfile.TemporaryDirectory() as tmp:
        from tokenizers import Tokenizer as HFTokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.trainers import BpeTrainer

        tok = HFTokenizer(BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=300, special_tokens=["[UNK]", "[EOS]"])
        tok.train_from_iterator(["hello world", "hello there"], trainer=trainer)

        tok_path = f"{tmp}/tok.json"
        tok.save(tok_path)

        cafezin_tok = Tokenizer(tok_path)
        original_vocab = cafezin_tok._tokenizer.get_vocab(with_added_tokens=True)
        original_max_id = max(original_vocab.values())

        # Extend with new tokens
        new_tokens = ["helloworld", "hellothere"]
        extended_tok = extend_tokenizer(cafezin_tok, new_tokens, f"{tmp}/extended.json")

        # Check original IDs preserved
        extended_vocab = extended_tok._tokenizer.get_vocab(with_added_tokens=True)
        for token, tid in original_vocab.items():
            assert extended_vocab[token] == tid, (
                f"Token {token} ID changed from {tid} to {extended_vocab[token]}"
            )

        # Check new tokens added at the end
        for token in new_tokens:
            assert token in extended_vocab
            assert extended_vocab[token] > original_max_id

        assert extended_tok.vocab_size == cafezin_tok.vocab_size + len(new_tokens)


def test_compute_tokenization_gain():
    """Test tokenization gain computation."""
    with tempfile.TemporaryDirectory() as tmp:
        from tokenizers import Tokenizer as HFTokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.trainers import BpeTrainer

        tok = HFTokenizer(BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=300, special_tokens=["[UNK]", "[EOS]"])
        tok.train_from_iterator(["hello world " * 10, "hello there " * 10], trainer=trainer)

        tok_path = f"{tmp}/tok.json"
        tok.save(tok_path)

        cafezin_tok = Tokenizer(tok_path)

        # Extend with a merged token
        extended_tok = extend_tokenizer(cafezin_tok, ["helloworld"], f"{tmp}/extended.json")

        heldout = ["hello world hello world", "hello there hello there"]
        gain = compute_tokenization_gain(cafezin_tok, extended_tok, heldout)

        assert "original_tokens" in gain
        assert "extended_tokens" in gain
        assert "token_reduction" in gain
        assert "pct_reduction" in gain
        # Should have some reduction since "helloworld" merges two tokens
        assert gain["token_reduction"] >= 0


def test_mean_decomposition_init():
    """Test mean decomposition initialization for new embeddings."""
    from cafezin_lm.tokenizer_extension import mean_decomposition_init

    # Create dummy embeddings
    vocab_size = 100
    hidden_size = 64
    embeddings = torch.randn(vocab_size, hidden_size)

    with tempfile.TemporaryDirectory() as tmp:
        from tokenizers import Tokenizer as HFTokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.trainers import BpeTrainer

        tok = HFTokenizer(BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=200, special_tokens=["[UNK]", "[EOS]"])
        tok.train_from_iterator(["hello world", "hello there", "world peace"], trainer=trainer)

        tok_path = f"{tmp}/tok.json"
        tok.save(tok_path)

        cafezin_tok = Tokenizer(tok_path)

        new_tokens = ["helloworld", "worldpeace"]
        new_rows = mean_decomposition_init(embeddings, cafezin_tok, new_tokens)

        assert new_rows.shape == (2, hidden_size)
        # Each new row should be the mean of its decomposition
        for i, token in enumerate(new_tokens):
            subtoken_ids = cafezin_tok.encode(token, eos=False)
            expected = embeddings[subtoken_ids].mean(dim=0)
            assert torch.allclose(new_rows[i], expected, atol=1e-5)


def test_grow_embeddings_preserves_original():
    """Test that growing embeddings preserves original rows and ties lm_head."""
    with tempfile.TemporaryDirectory() as tmp:
        from tokenizers import Tokenizer as HFTokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.trainers import BpeTrainer

        tok = HFTokenizer(BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=256, special_tokens=["[UNK]", "[EOS]"])
        # Use training data that won't produce the new tokens we'll add
        tok.train_from_iterator(["hello world foo", "hello there bar"], trainer=trainer)

        tok_path = f"{tmp}/tok.json"
        tok.save(tok_path)

        base_tok = Tokenizer(tok_path)
        vocab_size = base_tok.vocab_size

        cfg = tiny_config(vocab_size=vocab_size)
        model = build_model(cfg, seed=42)

        # Extend tokenizer with tokens that won't be in base vocab
        new_tokens = ["helloworldfoobar", "hellotherefoobar"]
        extended_tok = extend_tokenizer(base_tok, new_tokens, f"{tmp}/extended.json")

        # Save original embeddings
        orig_embed = model.model.language_model.embed_tokens.weight.data.clone()
        orig_vocab_size = base_tok.vocab_size

        # Grow embeddings
        grow_embeddings(model, base_tok, extended_tok, new_tokens)

        # Check original embeddings preserved
        new_embed = model.model.language_model.embed_tokens.weight.data
        assert torch.allclose(new_embed[:orig_vocab_size], orig_embed, atol=1e-5)

        # Check new vocab size
        assert model.model.language_model.embed_tokens.weight.shape[0] == extended_tok.vocab_size
        assert model.cfg.text.vocab_size == extended_tok.vocab_size

        # Check lm_head is tied
        assert model.lm_head.weight is model.model.language_model.embed_tokens.weight
        assert model.lm_head.out_features == extended_tok.vocab_size


def test_verify_round_trip():
    """Test the full round-trip verification."""
    with tempfile.TemporaryDirectory() as tmp:
        from tokenizers import Tokenizer as HFTokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.trainers import BpeTrainer

        tok = HFTokenizer(BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=256, special_tokens=["[UNK]", "[EOS]"])
        # Use training data that won't produce the new tokens we'll add
        tok.train_from_iterator(
            ["hello world foo", "hello there bar", "world peace baz"], trainer=trainer
        )

        tok_path = f"{tmp}/tok.json"
        tok.save(tok_path)

        base_tok = Tokenizer(tok_path)
        vocab_size = base_tok.vocab_size

        # Create original model with base vocab
        original_model = build_model(tiny_config(vocab_size=vocab_size), seed=42)
        original_model.eval()

        # Extend tokenizer with tokens that won't be in base vocab
        new_tokens = ["helloworldfoobar"]
        extended_tok = extend_tokenizer(base_tok, new_tokens, f"{tmp}/extended.json")

        # Create extended model with BASE vocab (will grow it)
        extended_model = build_model(tiny_config(vocab_size=vocab_size), seed=42)
        extended_model.eval()

        # Copy original weights to extended model
        with torch.no_grad():
            extended_model.model.language_model.embed_tokens.weight.data[:] = (
                original_model.model.language_model.embed_tokens.weight.data
            )

        # Grow embeddings on extended model
        grow_embeddings(extended_model, base_tok, extended_tok, new_tokens)

        test_texts = ["hello world foo", "hello there bar", "world peace baz"]
        results = verify_round_trip(
            base_tok, extended_tok, original_model, extended_model, test_texts
        )

        assert results["original_token_ids_preserved"]
        assert results["encode_decode_consistency"]
        assert results["new_token_encode_decode"]
        assert results["original_embeddings_preserved"]
        assert results["new_embeddings_match_mean_decomp"]
        assert results["lm_head_tied"]
        assert results["forward_pass_consistency"]


def test_export_extended_checkpoint():
    """Test exporting and loading an extended checkpoint."""
    with tempfile.TemporaryDirectory() as tmp:
        from tokenizers import Tokenizer as HFTokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.trainers import BpeTrainer

        tok = HFTokenizer(BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=256, special_tokens=["[UNK]", "[EOS]"])
        # Use training data that won't produce the new tokens we'll add
        tok.train_from_iterator(["hello world foo", "hello there bar"], trainer=trainer)

        tok_path = f"{tmp}/tok.json"
        tok.save(tok_path)

        base_tok = Tokenizer(tok_path)
        vocab_size = base_tok.vocab_size

        cfg = tiny_config(vocab_size=vocab_size)
        model = build_model(cfg, seed=42)

        # Extend tokenizer and grow embeddings
        new_tokens = ["helloworldfoobar"]
        extended_tok = extend_tokenizer(base_tok, new_tokens, f"{tmp}/extended.json")
        grow_embeddings(model, base_tok, extended_tok, new_tokens)

        # Export
        from cafezin_lm.tokenizer_extension import export_extended_checkpoint

        export_dir = f"{tmp}/export"
        manifest = export_extended_checkpoint(model, extended_tok, export_dir)

        assert manifest["tensors"] > 0
        assert (Path(export_dir) / "tokenizer.json").exists()
        assert (Path(export_dir) / "config.json").exists()

        # Load back
        from cafezin_lm.tokenizer_extension import load_extended_checkpoint

        loaded_model, loaded_tok = load_extended_checkpoint(export_dir, CafezinLM)

        assert loaded_tok.vocab_size == extended_tok.vocab_size
        assert loaded_model.cfg.text.vocab_size == extended_tok.vocab_size

        # Verify weights match (use allclose for dtype differences)
        for name in model.state_dict():
            if name in loaded_model.state_dict():
                a = model.state_dict()[name]
                b = loaded_model.state_dict()[name]
                # Convert to same dtype for comparison
                if a.dtype != b.dtype:
                    b = b.to(dtype=a.dtype)
                assert torch.allclose(a, b, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
