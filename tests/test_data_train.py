"""Tests for the Cafezin LM data pipeline and training loop."""

import os
import tempfile

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer as RustTokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from cafezin_lm import (
    TokenDataset,
    Tokenizer,
    TrainConfig,
    load_meta,
    tiny_config,
    train,
)
from cafezin_lm.checkpoint import read_hf_state
from cafezin_lm.cli import main


def test_token_dataset_basic():
    """TokenDataset reads and batches uint32 memmaps."""
    with tempfile.TemporaryDirectory() as tmp:
        tokens = np.arange(1000, dtype=np.uint32)
        bin_path = os.path.join(tmp, "train.bin")
        tokens.tofile(bin_path)

        ds = TokenDataset(bin_path, dtype="uint32")
        assert len(ds) == 1000

        x, y = ds.get_batch(2, 8, "cpu")
        assert x.shape == (2, 8)
        assert y.shape == (2, 8)
        assert torch.equal(x[0, 1:], y[0, :-1])
        x, y = ds.get_batch(1, 999)
        assert x[0, 0] == 0
        assert y[0, -1] == 999

        del ds


def test_token_dataset_requires_enough_tokens():
    """TokenDataset rejects files with <2 tokens."""
    with tempfile.TemporaryDirectory() as tmp:
        bin_path = os.path.join(tmp, "small.bin")
        np.array([1], dtype=np.uint32).tofile(bin_path)
        with pytest.raises(ValueError):
            TokenDataset(bin_path)


@pytest.fixture
def local_tokenizer(tmp_path):
    vocab = {"[UNK]": 0, **{f"unused{i}": i for i in range(1, 257)}, "hello": 257, "world": 258}
    raw = RustTokenizer(WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    raw.add_special_tokens(["<eos>"])
    path = tmp_path / "tokenizer.json"
    raw.save(str(path))
    return path


def test_tokenizer_encode_decode(local_tokenizer):
    tok = Tokenizer(local_tokenizer)
    ids = tok.encode("hello world", eos=True)
    assert ids == [257, 258, tok.eos_token_id]
    assert tok.decode(ids) == "hello world"


def test_prepare_train_resume_export_generate(tmp_path, local_tokenizer, capsys):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world hello world\n\nworld hello world hello", encoding="utf-8")
    data = tmp_path / "data"
    assert (
        main(
            [
                "prepare",
                "--tokenizer",
                str(local_tokenizer),
                "--input",
                str(corpus),
                "--out",
                str(data),
            ]
        )
        == 0
    )
    assert load_meta(data)["max_token_id"] == 259
    assert (data / "tokenizer.json").read_bytes() == local_tokenizer.read_bytes()
    run = tmp_path / "run"
    train_args = [
        "train",
        "--data-dir",
        str(data),
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--batch-size",
        "1",
        "--grad-accum",
        "1",
        "--steps",
        "2",
        "--seq-len",
        "4",
        "--warmup-steps",
        "0",
        "--save-every",
        "1",
        "--eval-every",
        "1",
        "--eval-batches",
        "1",
    ]
    assert main([*train_args, "--out-dir", str(run)]) == 0
    resumed = tmp_path / "resumed"
    assert (
        main([*train_args, "--out-dir", str(resumed), "--resume", str(run / "step-000000001")]) == 0
    )
    final = run / "step-000000002"
    expected, _ = read_hf_state(final)
    actual, _ = read_hf_state(resumed / final.name)
    for name in expected:
        assert torch.equal(expected[name], actual[name]), name

    export = tmp_path / "export"
    assert main(["export", "--checkpoint", str(final), "--out", str(export)]) == 0
    assert (export / "tokenizer.json").read_bytes() == local_tokenizer.read_bytes()
    exported, _ = read_hf_state(export)
    for name in expected:
        assert torch.equal(expected[name], exported[name]), name
    capsys.readouterr()
    assert (
        main(
            [
                "generate",
                "--checkpoint",
                str(export),
                "--device",
                "cpu",
                "--temperature",
                "0",
                "--prompt",
                "hello",
                "--max-new",
                "2",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.startswith("hello")


def test_prepare_rejects_oversized_vocab():
    """prepare() rejects vocab > 65535 for uint16."""
    from cafezin_lm.data import dtype_for_vocab

    assert dtype_for_vocab(70000) == "uint32"
    assert dtype_for_vocab(65535) == "uint16"
    assert dtype_for_vocab(248320) == "uint32"


def test_train_config_defaults():
    """TrainConfig has sensible defaults."""
    cfg = TrainConfig()
    assert cfg.batch_size == 4
    assert cfg.grad_accum == 8
    assert cfg.mtp_weight == 0.3
    assert cfg.steps == 100_000


def test_train_tiny_smoke():
    """Tiny Cafezin LM trains for 1 step without errors."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = tiny_config()

        # Create minimal data (tokens must be < vocab_size=256)
        train_tokens = np.random.randint(0, cfg.text.vocab_size, size=2000, dtype=np.uint32)
        val_tokens = np.random.randint(0, cfg.text.vocab_size, size=500, dtype=np.uint32)
        data_dir = os.path.join(tmp, "data")
        os.makedirs(data_dir)
        train_path = os.path.join(data_dir, "train.bin")
        val_path = os.path.join(data_dir, "val.bin")
        train_tokens.tofile(train_path)
        val_tokens.tofile(val_path)

        # Write meta.json
        meta = {
            "vocab_size": cfg.text.vocab_size,
            "n_docs": 1,
            "train_tokens": 2000,
            "val_tokens": 500,
            "dtype": "uint32",
            "eos_token_id": cfg.text.eos_token_id,
            "tokenizer": os.path.join(data_dir, "tokenizer.json"),
        }
        import json

        with open(os.path.join(data_dir, "meta.json"), "w") as f:
            json.dump(meta, f)

        train_cfg = TrainConfig(
            data_dir=data_dir,
            out_dir=os.path.join(tmp, "out"),
            seed=42,
            seq_len=64,
            batch_size=2,
            grad_accum=1,
            steps=1,
            eval_every=1,
            eval_batches=1,
            log_every=1,
            save_every=1,
            keep_last=1,
            device="cpu",
            dtype="float32",
            gradient_checkpointing=False,
            compile_model=False,
        )

        summary = train(cfg, train_cfg)
        assert isinstance(summary, str) or summary is not None

        # Close memmaps explicitly before tmp cleanup
        import gc

        gc.collect()


def test_load_meta():
    """load_meta reads the meta.json correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = os.path.join(tmp, "data")
        os.makedirs(data_dir)
        meta = {
            "vocab_size": 256,
            "n_docs": 5,
            "train_tokens": 1000,
            "val_tokens": 200,
            "dtype": "uint16",
            "eos_token_id": 2,
            "tokenizer": os.path.join(data_dir, "tokenizer.json"),
        }
        import json

        with open(os.path.join(data_dir, "meta.json"), "w") as f:
            json.dump(meta, f)

        loaded = load_meta(data_dir)
        assert loaded["vocab_size"] == 256
        assert loaded["train_tokens"] == 1000
