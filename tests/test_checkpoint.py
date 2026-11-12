"""Tests for the Cafezin LM checkpoint I/O and verification."""

import tempfile

import pytest
import torch

from cafezin_lm import (
    REFERENCE_SPEC,
    ModelConfig,
    build_model,
    load_hf_checkpoint,
    model_spec,
    save_hf_checkpoint,
    tiny_config,
    verify_offline,
)
from cafezin_lm.checkpoint import spec_digest
from cafezin_lm.safetensors_io import load_file, load_header, save_file


def test_safetensors_round_trip():
    tmp = tempfile.mkdtemp()
    tensors = {
        "a": torch.randn(4, 8, dtype=torch.bfloat16),
        "b": torch.arange(5, dtype=torch.float32),
        "c": torch.arange(6, dtype=torch.int64),
        "d": torch.zeros(2, dtype=torch.bool),
    }
    path = f"{tmp}/x.safetensors"
    save_file(tensors, path, metadata={"format": "pt"})
    back = load_file(path)
    for k in tensors:
        assert back[k].dtype == tensors[k].dtype
        assert torch.equal(back[k], tensors[k])


def test_safetensors_header_only():
    tmp = tempfile.mkdtemp()
    tensors = {"x": torch.randn(3, 4, dtype=torch.bfloat16)}
    path = f"{tmp}/hdr.safetensors"
    save_file(tensors, path)
    hdr = load_header(path)
    assert "x" in hdr
    assert hdr["x"]["dtype"] == "BF16"
    assert hdr["x"]["shape"] == [3, 4]


def test_safetensors_subset_preserves_offsets(tmp_path):
    path = tmp_path / "weights.safetensors"
    tensors = {"a": torch.tensor([11]), "b": torch.tensor([22]), "c": torch.tensor([33])}
    save_file(tensors, path)
    subset = load_file(path, keys=["c", "b"])
    assert set(subset) == {"b", "c"}
    assert torch.equal(subset["b"], tensors["b"])
    assert torch.equal(subset["c"], tensors["c"])


def test_spec_digest_stability():
    spec = {"a": ("BF16", (2, 3)), "b": ("F32", (5,))}
    d1 = spec_digest(spec, total_bytes=100)
    d2 = spec_digest(spec, total_bytes=100)
    assert d1 == d2
    assert spec_digest(spec, total_bytes=200) != d1


def test_export_round_trip_f32():
    cfg = tiny_config()
    model = build_model(cfg, seed=7)
    model.eval()

    tmp = tempfile.mkdtemp()
    save_hf_checkpoint(model, f"{tmp}/hf", cfg, dtype_override="F32", max_shard_bytes=400_000)

    model2 = build_model(cfg, seed=1)
    model2.eval()
    load_hf_checkpoint(model2, f"{tmp}/hf")

    for name in model.state_dict():
        if name in model2.state_dict():
            assert torch.equal(model.state_dict()[name], model2.state_dict()[name])

    assert model2.lm_head.weight is model2.model.language_model.embed_tokens.weight


def test_export_bf16_round_trip_lossy():
    """bf16 round-trip preserves values to bf16 precision only."""
    cfg = tiny_config()
    model = build_model(cfg, seed=7)
    model.eval()

    tmp = tempfile.mkdtemp()
    save_hf_checkpoint(model, f"{tmp}/hf", cfg, dtype_override="BF16", max_shard_bytes=400_000)

    model2 = build_model(cfg, seed=1)
    model2.eval()
    load_hf_checkpoint(model2, f"{tmp}/hf")

    for name in model.state_dict():
        a = model.state_dict()[name].to(torch.bfloat16)
        b = model2.state_dict()[name].to(torch.bfloat16)
        assert torch.equal(a, b), f"{name} differs after bf16 round-trip"


def test_export_matches_reference_spec():
    """Export of the full preset with reference policy must match the spec."""
    cfg = ModelConfig()
    with torch.device("meta"):
        model = build_model(cfg)
    report = verify_offline(model, REFERENCE_SPEC)
    assert report.ok


def test_load_hf_preserves_tied_weights():
    cfg = tiny_config()
    model = build_model(cfg, seed=1)
    model.eval()

    tmp = tempfile.mkdtemp()
    save_hf_checkpoint(model, f"{tmp}/hf", cfg, dtype_override="F32")
    model2 = build_model(cfg, seed=2)
    load_hf_checkpoint(model2, f"{tmp}/hf")

    assert model2.lm_head.weight is model2.model.language_model.embed_tokens.weight


def test_load_hf_rejects_mismatched_config():
    cfg = tiny_config()
    model = build_model(cfg, seed=1)

    tmp = tempfile.mkdtemp()
    save_hf_checkpoint(model, f"{tmp}/hf", cfg, dtype_override="F32")

    other = tiny_config(hidden_size=96, num_hidden_layers=4)
    other_model = build_model(other)
    with pytest.raises(RuntimeError):
        load_hf_checkpoint(other_model, f"{tmp}/hf")


def test_model_spec_excludes_lm_head():
    cfg = tiny_config()
    with torch.device("meta"):
        model = build_model(cfg)
    spec = model_spec(model)
    assert "lm_head.weight" not in spec
    assert "model.language_model.embed_tokens.weight" in spec


def test_offline_verify_parity():
    """The full preset's weight spec matches the recorded reference fingerprint."""
    cfg = ModelConfig()
    with torch.device("meta"):
        model = build_model(cfg)
    report = verify_offline(model, REFERENCE_SPEC)
    assert report.ok
    assert report.actual_tensors == 738


def test_export_rejects_partial_checkpoint(tmp_path):
    from cafezin_lm.cli import main

    path = tmp_path / "partial.safetensors"
    save_file({"mtp.fc.weight": torch.zeros(64, 128)}, path)
    out = tmp_path / "export"
    with pytest.raises(RuntimeError, match="missing:"):
        main(["export", "--checkpoint", str(path), "--out", str(out)])
    assert not out.exists()
