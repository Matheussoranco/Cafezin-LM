"""Regression checks for checkpoint trust, evaluation semantics and continuation."""

import io
import json
import pickle
import random
from unittest.mock import patch

import numpy as np
import pytest
import torch

from pico_lm.bench import build_cloze, load_instruction_examples, normalize_answer
from pico_lm.config import ModelConfig, SFTConfig
from pico_lm.data import _val_stride
from pico_lm.model import Transformer
from pico_lm.provenance import capture_rng, restore_rng
from pico_lm.train import load_model


def test_unsafe_load_is_never_retried():
    with (
        patch("torch.load", side_effect=pickle.UnpicklingError("untrusted")) as loader,
        pytest.raises(pickle.UnpicklingError),
    ):
        load_model("untrusted.pt")
    assert loader.call_count == 1
    assert loader.call_args.kwargs["weights_only"] is True


def test_rng_roundtrip_safe_serialization():
    state = capture_rng()
    expected = (torch.rand(4), np.random.rand(4), random.random())
    stream = io.BytesIO()
    torch.save(state, stream)
    stream.seek(0)
    restore_rng(torch.load(stream, weights_only=True))
    assert torch.equal(expected[0], torch.rand(4))
    np.testing.assert_array_equal(expected[1], np.random.rand(4))
    assert expected[2] == random.random()


@pytest.mark.parametrize("fraction", [-1, 2, float("nan"), float("inf")])
def test_invalid_split(fraction):
    with pytest.raises(ValueError):
        _val_stride(fraction)


def test_too_short_sft():
    with pytest.raises(ValueError):
        SFTConfig(max_len=1).validate()


def test_cloze_insufficient_distinct_choices():
    class Tokenizer:
        def encode(self, text):
            return [1] * 80

    with pytest.raises(ValueError, match="distinct"):
        build_cloze(["same", "same"], Tokenizer(), 2)


def test_multiturn_evaluation_uses_final_response(tmp_path):
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    path = tmp_path / "messages.jsonl"
    path.write_text(json.dumps({"messages": messages}), encoding="utf-8")
    item = load_instruction_examples(path)[0]
    assert item["messages"] == messages[:3]
    assert item["target"] == "a2"


def test_normalization_preserves_case():
    assert normalize_answer("CAT") != normalize_answer("cat")


def test_interrupted_training_matches_uninterrupted(tmp_path):
    import importlib

    from pico_lm.config import TrainConfig

    training = importlib.import_module("pico_lm.train")
    data = tmp_path / "data"
    data.mkdir()
    (data / "meta.json").write_text(json.dumps({"vocab_size": 64}), encoding="utf-8")
    for split in ("train", "val"):
        np.asarray(list(range(64)) * 4, dtype=np.uint16).tofile(data / f"{split}.bin")
    model_cfg = ModelConfig(
        vocab_size=64, dim=16, n_layers=1, n_heads=2, n_kv_heads=1, max_seq_len=16, dropout=0.1
    )
    config = TrainConfig(
        batch_size=2,
        grad_accum_steps=1,
        seq_len=16,
        max_steps=4,
        warmup_steps=1,
        eval_interval=2,
        eval_iters=1,
    )
    original_save = training.save_checkpoint

    def interrupted_save(path, *args, **kwargs):
        original_save(path, *args, **kwargs)
        if path.name == "last.pt":
            raise InterruptedError("simulated interruption after checkpoint")

    with patch.object(training, "pick_device", return_value="cpu"):
        training.train(model_cfg, config, data, tmp_path / "full")
        with (
            patch.object(training, "save_checkpoint", side_effect=interrupted_save),
            pytest.raises(InterruptedError),
        ):
            training.train(model_cfg, config, data, tmp_path / "resumed")
        training.train(model_cfg, config, data, tmp_path / "resumed", resume=True)
    full = torch.load(tmp_path / "full" / "last.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed" / "last.pt", weights_only=True)
    for name in full["model"]:
        assert torch.equal(full["model"][name], resumed["model"][name]), name
    assert full["best_val"] == resumed["best_val"]


def test_interrupted_sft_matches_uninterrupted(tmp_path):
    import importlib

    from pico_lm.sft import prepare_sft
    from pico_lm.tokenizer import PicoTokenizer

    sft = importlib.import_module("pico_lm.sft")
    tok = PicoTokenizer.train(["fox dog bird cat"], 280, tmp_path / "tokenizer.json")
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(
        "\n".join(json.dumps({"instruction": "repeat fox", "response": "fox"}) for _ in range(8)),
        encoding="utf-8",
    )
    prepare_sft(pairs, tok, tmp_path / "data", max_len=32)
    base = Transformer(
        ModelConfig(
            vocab_size=tok.vocab_size,
            dim=16,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            max_seq_len=32,
            dropout=0.1,
        )
    )
    source = tmp_path / "base.pt"
    torch.save({"model": base.state_dict(), "model_cfg": base.cfg.to_dict()}, source)
    cfg = SFTConfig(
        max_steps=4,
        max_len=32,
        batch_size=2,
        eval_interval=2,
        eval_iters=1,
        warmup_steps=1,
        select_by="loss",
    )
    original = sft.save_sft_checkpoint

    def interrupt(path, *args, **kwargs):
        original(path, *args, **kwargs)
        if path.name == "last.pt":
            raise InterruptedError

    with patch.object(sft, "pick_device", return_value="cpu"):
        sft.finetune(tmp_path / "data", tmp_path / "full", cfg, init_from=str(source))
        with (
            patch.object(sft, "save_sft_checkpoint", side_effect=interrupt),
            pytest.raises(InterruptedError),
        ):
            sft.finetune(tmp_path / "data", tmp_path / "resumed", cfg, init_from=str(source))
        sft.finetune(tmp_path / "data", tmp_path / "resumed", cfg, resume=True)
    full = torch.load(tmp_path / "full" / "last.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed" / "last.pt", weights_only=True)
    for name in full["model"]:
        assert torch.equal(full["model"][name], resumed["model"][name]), name


def test_cli_resume_preserves_saved_schedule(tmp_path):
    from pico_lm.cli import main
    from pico_lm.config import TrainConfig

    (tmp_path / "meta.json").write_text(json.dumps({"vocab_size": 64}), encoding="utf-8")
    cfg = TrainConfig(lr=1e-4, min_lr=1e-5, warmup_steps=11, gradient_checkpointing=True)
    torch.save(
        {"train_cfg": cfg.to_dict(), "model_cfg": ModelConfig().to_dict()}, tmp_path / "last.pt"
    )
    with patch("pico_lm.train.train", return_value=1.0) as train:
        main(["train", "--data", str(tmp_path), "--out", str(tmp_path), "--resume"])
    assert train.call_args.args[1].to_dict() == cfg.to_dict()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_long_context_backward_and_cache():
    cfg = ModelConfig(vocab_size=64, dim=32, n_layers=1, n_heads=4, n_kv_heads=2, max_seq_len=4096)
    model = Transformer(cfg).cuda().train()
    model.set_gradient_checkpointing(True)
    x = torch.randint(0, 64, (1, 4096), device="cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.autocast("cuda", dtype=dtype):
        _, loss = model(x, x)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.generate(x[:, :8], 4, temperature=0).shape == (1, 12)
