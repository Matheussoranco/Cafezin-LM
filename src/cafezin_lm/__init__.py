"""Cafezin LM: hybrid linear/full-attention multimodal decoder.

Experimental implementation of dense multimodal Qwen3.5, with a checkpoint
layout based on ``maritaca-ai/S5-nano``.
"""

from .checkpoint import (
    REFERENCE_SPEC,
    import_config,
    load_hf_checkpoint,
    model_spec,
    remote_spec,
    save_hf_checkpoint,
    verify_offline,
)
from .config import (
    PRESET_NAMES,
    PRESETS,
    ModelConfig,
    TextConfig,
    VisionConfig,
    base_config,
    preset,
    tiny_config,
)
from .data import TokenDataset, dtype_for_vocab, load_meta, prepare
from .model import (
    CafezinLM,
    GenerationCache,
    build_model,
    chunk_gated_delta_rule,
    parameter_count_breakdown,
    parameter_count_estimate,
    parameter_count_estimate_full,
    recurrent_gated_delta_rule,
)
from .safetensors_io import load_file, load_header, save_file
from .tokenizer import Tokenizer
from .train import TrainConfig, train

__all__ = [
    "PRESETS",
    "PRESET_NAMES",
    "REFERENCE_SPEC",
    "CafezinLM",
    "GenerationCache",
    "ModelConfig",
    "TextConfig",
    "TokenDataset",
    "Tokenizer",
    "TrainConfig",
    "VisionConfig",
    "base_config",
    "build_model",
    "chunk_gated_delta_rule",
    "dtype_for_vocab",
    "import_config",
    "load_file",
    "load_header",
    "load_hf_checkpoint",
    "load_meta",
    "model_spec",
    "parameter_count_breakdown",
    "parameter_count_estimate",
    "parameter_count_estimate_full",
    "prepare",
    "preset",
    "recurrent_gated_delta_rule",
    "remote_spec",
    "save_file",
    "save_hf_checkpoint",
    "tiny_config",
    "train",
    "verify_offline",
]

__version__ = "0.3.0a1"
