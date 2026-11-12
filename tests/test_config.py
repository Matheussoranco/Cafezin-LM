"""Tests for the Cafezin LM configuration."""

import pytest

from cafezin_lm.config import (
    ModelConfig,
    TextConfig,
    VisionConfig,
    preset,
    tiny_config,
)


def test_text_config_defaults():
    cfg = TextConfig()
    assert cfg.vocab_size == 248320
    assert cfg.hidden_size == 2560
    assert cfg.num_hidden_layers == 32
    assert cfg.num_attention_heads == 16
    assert cfg.num_key_value_heads == 4
    assert cfg.head_dim == 256
    assert cfg.rotary_dim == 64
    assert cfg.num_key_value_groups == 4
    assert len(cfg.layer_types) == 32
    assert cfg.num_full_attention_layers == 8
    assert cfg.num_linear_attention_layers == 24


def test_text_config_derives_layer_types():
    cfg = TextConfig(
        num_hidden_layers=12,
        full_attention_interval=4,
        layer_types=(),
    )
    assert len(cfg.layer_types) == 12
    assert cfg.layer_types[3] == "full_attention"
    assert cfg.layer_types[7] == "full_attention"
    assert cfg.layer_types[11] == "full_attention"
    assert all(
        t == "linear_attention" for i, t in enumerate(cfg.layer_types) if i not in (3, 7, 11)
    )


def test_text_config_validates_layer_types():
    with pytest.raises(ValueError):
        TextConfig(num_hidden_layers=4, layer_types=("linear_attention", "full_attention"))


def test_text_config_validates_gqa():
    with pytest.raises(ValueError):
        TextConfig(num_attention_heads=8, num_key_value_heads=3)


def test_text_config_mrope_validation():
    TextConfig(
        head_dim=256,
        partial_rotary_factor=0.25,
        mrope_section=(11, 11, 10),
    )
    with pytest.raises(ValueError):
        TextConfig(
            head_dim=64,
            partial_rotary_factor=0.25,
            mrope_section=(11, 11, 10),
        )


def test_vision_config_defaults():
    cfg = VisionConfig()
    assert cfg.depth == 24
    assert cfg.hidden_size == 1024
    assert cfg.num_heads == 16
    assert cfg.head_dim == 64
    assert cfg.num_grid_per_side == 48
    assert cfg.spatial_merge_unit == 4


def test_full_config_roundtrip():
    cfg = preset("base")
    hf = cfg.to_hf_dict()
    cfg2 = ModelConfig.from_hf_dict(hf)
    assert cfg2.to_hf_dict() == hf


def test_config_to_from_dict():
    cfg = preset("base")
    d = cfg.to_dict()
    cfg2 = ModelConfig.from_dict(d)
    assert cfg2.text.hidden_size == cfg.text.hidden_size
    assert cfg2.vision.depth == cfg.vision.depth


def test_tiny_preset_shape():
    cfg = tiny_config()
    assert cfg.text.hidden_size == 64
    assert cfg.text.num_hidden_layers == 8
    assert cfg.text.num_attention_heads == 4
    assert cfg.text.rotary_dim == 16
    assert cfg.text.full_attention_interval == 4
    assert len(cfg.text.layer_types) == 8
    assert cfg.text.layer_types[3] == "full_attention"
    assert cfg.text.layer_types[7] == "full_attention"
    assert all(
        t == "linear_attention" for i, t in enumerate(cfg.text.layer_types) if i not in (3, 7)
    )
    assert cfg.vision.depth == 2
    assert cfg.vision.hidden_size == 32
    assert cfg.mtp_num_hidden_layers == 1


def test_text_config_derived_properties():
    cfg = TextConfig()
    assert cfg.linear_key_dim == 2048
    assert cfg.linear_value_dim == 4096
    assert cfg.linear_conv_dim == 8192
    assert cfg.linear_num_value_groups == 2
