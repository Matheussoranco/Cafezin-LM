"""Cafezin LM configuration: a hybrid linear/full-attention multimodal decoder.

The model is a Qwen3.5-VL-class architecture — a 32-layer text decoder that
interleaves *linear* attention (gated delta rule) with *full* attention every
fourth layer, a learned-position vision tower, and a one-layer multi-token
prediction (MTP) head.

`from_hf_dict` / `to_hf_dict` use the reference configuration layout. The
``base`` preset describes the pinned S5-nano architecture, and ``cafezin-lm verify``
compares its tensor metadata with the published checkpoint.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

# The published checkpoint this implementation is verified against.
REFERENCE_REPO = "maritaca-ai/S5-nano"
REFERENCE_REVISION = "fe3a3581c33cb22917770ef1987b7c6e30929425"

# Special-token ids the reference config pins for multimodal plumbing.
IMAGE_TOKEN_ID = 248056
VIDEO_TOKEN_ID = 248057
VISION_START_TOKEN_ID = 248053
VISION_END_TOKEN_ID = 248054

# Absent from the published vision_config; Hugging Face fills the RoPE base from
# its default when `rope_parameters` is missing. Recorded explicitly so the
# assumption is visible and overridable instead of buried.
VISION_ROPE_THETA_FALLBACK = 10000.0


def _finite(value: object) -> bool:
    return not isinstance(value, float) or math.isfinite(value)


def _check_finite(cfg: object) -> None:
    for name, value in asdict(cfg).items():  # type: ignore[arg-type]
        if not _finite(value):
            raise ValueError(f"{name} must be finite")


@dataclass
class TextConfig:
    """Text decoder shape. Defaults are the reference model's text_config."""

    vocab_size: int = 248320
    hidden_size: int = 2560
    intermediate_size: int = 9216
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    hidden_act: str = "silu"
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    attention_bias: bool = False
    attention_dropout: float = 0.0
    attn_output_gate: bool = True
    tie_word_embeddings: bool = True
    use_cache: bool = True
    dtype: str = "bfloat16"
    # linear attention (gated delta rule)
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    mamba_ssm_dtype: str = "float32"
    # attention schedule
    full_attention_interval: int = 4
    layer_types: tuple[str, ...] = ()
    mlp_only_layers: tuple[int, ...] = ()
    # rotary
    rope_theta: float = 10000000.0
    partial_rotary_factor: float = 0.25
    mrope_section: tuple[int, int, int] = (11, 11, 10)
    mrope_interleaved: bool = True
    # multi-token prediction
    mtp_num_hidden_layers: int = 1
    mtp_use_dedicated_embeddings: bool = False
    # token ids recorded by the reference text_config
    eos_token_id: int = 248044
    model_type: str = "qwen3_5_text"

    def __post_init__(self) -> None:
        _check_finite(self)
        if not self.layer_types:
            self.layer_types = self.derive_layer_types()
        elif len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries but num_hidden_layers "
                f"is {self.num_hidden_layers}"
            )
        self.validate()

    def derive_layer_types(self) -> tuple[str, ...]:
        """`linear_attention`, except every `full_attention_interval`-th layer.

        Full attention lands on layers 3, 7, 11, ... — 8 of 32 by default.
        """
        return tuple(
            "full_attention" if (i + 1) % self.full_attention_interval == 0 else "linear_attention"
            for i in range(self.num_hidden_layers)
        )

    def validate(self) -> None:
        if self.vocab_size <= 0 or self.hidden_size <= 0 or self.num_hidden_layers <= 0:
            raise ValueError("vocab_size, hidden_size and num_hidden_layers must be positive")
        if self.num_attention_heads <= 0 or self.num_key_value_heads <= 0:
            raise ValueError("attention head counts must be positive")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads {self.num_attention_heads} not divisible by "
                f"num_key_value_heads {self.num_key_value_heads}"
            )
        if self.head_dim <= 0 or self.head_dim % 2:
            raise ValueError(f"head_dim {self.head_dim} must be positive and even")
        if self.linear_num_value_heads % self.linear_num_key_heads != 0:
            raise ValueError(
                f"linear_num_value_heads {self.linear_num_value_heads} not divisible by "
                f"linear_num_key_heads {self.linear_num_key_heads}"
            )
        if not 0.0 < self.partial_rotary_factor <= 1.0:
            raise ValueError("partial_rotary_factor must be in (0, 1]")
        rope_dim = self.rotary_dim
        if rope_dim % 2:
            raise ValueError(f"rotary dim {rope_dim} must be even")
        # mRoPE writes section[d] *interleaved* slots for grids H and W: for grid
        # `d` the slice is (d, section[d] * 3, 3), so its last index is
        # d + 3 * (section[d] - 1) and must land inside the half-rotary table.
        if len(self.mrope_section) != 3:
            raise ValueError(f"mrope_section must have 3 entries, got {self.mrope_section}")
        half = rope_dim // 2
        for grid, count in enumerate(self.mrope_section):
            if grid == 0:
                continue
            last = grid + 3 * (count - 1)
            if count <= 0 or last >= half:
                raise ValueError(
                    f"mrope_section {self.mrope_section} overruns the {half}-entry rotary "
                    f"table for grid {grid} (last index {last})"
                )
        unknown = {t for t in self.layer_types} - {"linear_attention", "full_attention"}
        if unknown:
            raise ValueError(f"unknown layer types {sorted(unknown)}")
        if self.linear_conv_kernel_dim <= 0:
            raise ValueError("linear_conv_kernel_dim must be positive")

    # -- derived ---------------------------------------------------------
    @property
    def rotary_dim(self) -> int:
        """Rotary width: a partial slice of `head_dim` (64 of 256 by default)."""
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def linear_key_dim(self) -> int:
        return self.linear_key_head_dim * self.linear_num_key_heads

    @property
    def linear_value_dim(self) -> int:
        return self.linear_value_head_dim * self.linear_num_value_heads

    @property
    def linear_conv_dim(self) -> int:
        return self.linear_key_dim * 2 + self.linear_value_dim

    @property
    def linear_num_value_groups(self) -> int:
        return self.linear_num_value_heads // self.linear_num_key_heads

    @property
    def num_full_attention_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == "full_attention")

    @property
    def num_linear_attention_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == "linear_attention")

    def to_hf_dict(self) -> dict:
        """The exact ``text_config`` block of the reference model."""
        return {
            "attention_bias": self.attention_bias,
            "attention_dropout": self.attention_dropout,
            "attn_output_gate": self.attn_output_gate,
            "dtype": self.dtype,
            "eos_token_id": self.eos_token_id,
            "full_attention_interval": self.full_attention_interval,
            "head_dim": self.head_dim,
            "hidden_act": self.hidden_act,
            "hidden_size": self.hidden_size,
            "initializer_range": self.initializer_range,
            "intermediate_size": self.intermediate_size,
            "layer_types": list(self.layer_types),
            "linear_conv_kernel_dim": self.linear_conv_kernel_dim,
            "linear_key_head_dim": self.linear_key_head_dim,
            "linear_num_key_heads": self.linear_num_key_heads,
            "linear_num_value_heads": self.linear_num_value_heads,
            "linear_value_head_dim": self.linear_value_head_dim,
            "max_position_embeddings": self.max_position_embeddings,
            "mlp_only_layers": list(self.mlp_only_layers),
            "model_type": self.model_type,
            "mtp_num_hidden_layers": self.mtp_num_hidden_layers,
            "mtp_use_dedicated_embeddings": self.mtp_use_dedicated_embeddings,
            "num_attention_heads": self.num_attention_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "num_key_value_heads": self.num_key_value_heads,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_parameters": {
                "mrope_interleaved": self.mrope_interleaved,
                "mrope_section": list(self.mrope_section),
                "rope_type": "default",
                "rope_theta": self.rope_theta,
                "partial_rotary_factor": self.partial_rotary_factor,
            },
            "tie_word_embeddings": self.tie_word_embeddings,
            "use_cache": self.use_cache,
            "vocab_size": self.vocab_size,
            "mamba_ssm_dtype": self.mamba_ssm_dtype,
        }

    @classmethod
    def from_hf_dict(cls, d: dict) -> TextConfig:
        rope = d.get("rope_parameters", {}) or {}
        known = {
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "hidden_act",
            "max_position_embeddings",
            "rms_norm_eps",
            "initializer_range",
            "attention_bias",
            "attention_dropout",
            "attn_output_gate",
            "tie_word_embeddings",
            "use_cache",
            "dtype",
            "linear_conv_kernel_dim",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "mamba_ssm_dtype",
            "full_attention_interval",
            "mtp_num_hidden_layers",
            "mtp_use_dedicated_embeddings",
            "eos_token_id",
            "model_type",
        }
        kwargs = {k: v for k, v in d.items() if k in known}
        if d.get("layer_types"):
            kwargs["layer_types"] = tuple(d["layer_types"])
        kwargs["mlp_only_layers"] = tuple(d.get("mlp_only_layers") or ())
        kwargs["rope_theta"] = float(rope.get("rope_theta", 10000000.0))
        kwargs["partial_rotary_factor"] = float(rope.get("partial_rotary_factor", 0.25))
        if rope.get("mrope_section"):
            kwargs["mrope_section"] = tuple(int(x) for x in rope["mrope_section"])
        kwargs["mrope_interleaved"] = bool(rope.get("mrope_interleaved", True))
        return cls(**kwargs)


@dataclass
class VisionConfig:
    """Vision tower shape. Defaults are the reference model's vision_config.

    ``zero_centered_norm`` records a real ambiguity. The published Hugging Face
    port builds the vision blocks with plain ``nn.LayerNorm`` (scale used as
    stored), but the published weights do not look like that: every
    ``blocks.*.norm*`` tensor has the same signature as the checkpoint's *other*
    norms, which that same port computes as ``(1 + weight)``. See
    the reference configuration. Default ``False`` follows the reference port.
    """

    zero_centered_norm: bool = False
    depth: int = 24
    hidden_size: int = 1024
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 4096
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 2560
    num_position_embeddings: int = 2304
    initializer_range: float = 0.02
    deepstack_visual_indexes: tuple[int, ...] = ()
    rope_theta: float = VISION_ROPE_THETA_FALLBACK
    model_type: str = "qwen3_5"

    def __post_init__(self) -> None:
        _check_finite(self)
        self.validate()

    def validate(self) -> None:
        if self.depth <= 0 or self.hidden_size <= 0 or self.num_heads <= 0:
            raise ValueError("depth, hidden_size and num_heads must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError(
                f"hidden_size {self.hidden_size} not divisible by num_heads {self.num_heads}"
            )
        if self.head_dim % 4:
            raise ValueError(
                f"vision head_dim {self.head_dim} must be divisible by 4 for axial RoPE"
            )
        if self.patch_size <= 0 or self.temporal_patch_size <= 0:
            raise ValueError("patch sizes must be positive")
        if self.spatial_merge_size <= 0:
            raise ValueError("spatial_merge_size must be positive")
        if self.num_position_embeddings <= 0:
            raise ValueError("num_position_embeddings must be positive")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def num_grid_per_side(self) -> int:
        side = round(self.num_position_embeddings**0.5)
        if side * side != self.num_position_embeddings:
            raise ValueError(
                f"num_position_embeddings {self.num_position_embeddings} is not a perfect square"
            )
        return side

    @property
    def spatial_merge_unit(self) -> int:
        return self.spatial_merge_size**2

    def to_hf_dict(self) -> dict:
        return {
            "deepstack_visual_indexes": list(self.deepstack_visual_indexes),
            "depth": self.depth,
            "hidden_act": self.hidden_act,
            "hidden_size": self.hidden_size,
            "in_channels": self.in_channels,
            "initializer_range": self.initializer_range,
            "intermediate_size": self.intermediate_size,
            "model_type": self.model_type,
            "num_heads": self.num_heads,
            "num_position_embeddings": self.num_position_embeddings,
            "out_hidden_size": self.out_hidden_size,
            "patch_size": self.patch_size,
            "spatial_merge_size": self.spatial_merge_size,
            "temporal_patch_size": self.temporal_patch_size,
        }

    @classmethod
    def from_hf_dict(cls, d: dict) -> VisionConfig:
        known = {
            "depth",
            "hidden_size",
            "hidden_act",
            "intermediate_size",
            "num_heads",
            "in_channels",
            "patch_size",
            "spatial_merge_size",
            "temporal_patch_size",
            "out_hidden_size",
            "num_position_embeddings",
            "initializer_range",
        }
        kwargs = {k: v for k, v in d.items() if k in known}
        kwargs["deepstack_visual_indexes"] = tuple(d.get("deepstack_visual_indexes") or ())
        rope = d.get("rope_parameters") or {}
        kwargs["rope_theta"] = float(rope.get("rope_theta", VISION_ROPE_THETA_FALLBACK))
        # An older checkpoint records model_type "qwen3_5" for the vision tower,
        # which collides with the top-level type; normalise it.
        kwargs["model_type"] = "qwen3_5"
        return cls(**kwargs)


@dataclass
class ModelConfig:
    """Top-level configuration: vision tower + text decoder + MTP head."""

    text: TextConfig = field(default_factory=TextConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    image_token_id: int = IMAGE_TOKEN_ID
    video_token_id: int = VIDEO_TOKEN_ID
    vision_start_token_id: int = VISION_START_TOKEN_ID
    vision_end_token_id: int = VISION_END_TOKEN_ID
    tie_word_embeddings: bool = True
    architectures: tuple[str, ...] = ("Qwen3_5ForConditionalGeneration",)
    model_type: str = "qwen3_5"

    @property
    def mtp_num_hidden_layers(self) -> int:
        return self.text.mtp_num_hidden_layers

    @property
    def lm_head_tied(self) -> bool:
        """True when ``lm_head`` mirrors the token embedding and is not stored."""
        return bool(self.tie_word_embeddings or self.text.tie_word_embeddings)

    def to_hf_dict(self) -> dict:
        return {
            "architectures": list(self.architectures),
            "image_token_id": self.image_token_id,
            "model_type": self.model_type,
            "text_config": self.text.to_hf_dict(),
            "tie_word_embeddings": self.tie_word_embeddings,
            "video_token_id": self.video_token_id,
            "vision_config": self.vision.to_hf_dict(),
            "vision_end_token_id": self.vision_end_token_id,
            "vision_start_token_id": self.vision_start_token_id,
        }

    @classmethod
    def from_hf_dict(cls, d: dict) -> ModelConfig:
        return cls(
            text=TextConfig.from_hf_dict(d.get("text_config") or {}),
            vision=VisionConfig.from_hf_dict(d.get("vision_config") or {}),
            image_token_id=d.get("image_token_id", IMAGE_TOKEN_ID),
            video_token_id=d.get("video_token_id", VIDEO_TOKEN_ID),
            vision_start_token_id=d.get("vision_start_token_id", VISION_START_TOKEN_ID),
            vision_end_token_id=d.get("vision_end_token_id", VISION_END_TOKEN_ID),
            tie_word_embeddings=bool(d.get("tie_word_embeddings", True)),
            architectures=tuple(d.get("architectures") or ("Qwen3_5ForConditionalGeneration",)),
            model_type=d.get("model_type", "qwen3_5"),
        )

    def to_dict(self) -> dict:
        """Flat, JSON-safe view used inside training checkpoints."""
        return {
            "text": asdict(self.text),
            "vision": asdict(self.vision),
            "image_token_id": self.image_token_id,
            "video_token_id": self.video_token_id,
            "vision_start_token_id": self.vision_start_token_id,
            "vision_end_token_id": self.vision_end_token_id,
            "tie_word_embeddings": self.tie_word_embeddings,
            "architectures": list(self.architectures),
            "model_type": self.model_type,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        if isinstance(d.get("text"), dict):
            as_tuple = lambda v: tuple(v) if isinstance(v, list) else v  # noqa: E731
            cfg = cls(
                text=TextConfig(
                    **{
                        k: as_tuple(v)
                        for k, v in d["text"].items()
                        if k in TextConfig.__dataclass_fields__
                    }
                ),
                vision=VisionConfig(
                    **{
                        k: as_tuple(v)
                        for k, v in d["vision"].items()
                        if k in VisionConfig.__dataclass_fields__
                    }
                ),
            )
            for name in (
                "image_token_id",
                "video_token_id",
                "vision_start_token_id",
                "vision_end_token_id",
            ):
                if name in d:
                    setattr(cfg, name, d[name])
            return cfg
        return cls.from_hf_dict(d)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_hf_dict(), indent=4), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> ModelConfig:
        return cls.from_hf_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def base_config() -> ModelConfig:
    """The full 4.66 B parameter configuration."""
    return ModelConfig()


def tiny_config(
    *,
    hidden_size: int = 64,
    num_hidden_layers: int = 8,
    vocab_size: int = 256,
    vision_depth: int = 2,
) -> ModelConfig:
    """A small same-shape model for tests and CPU smoke runs.

    Every structural relationship of the full config is preserved (full
    attention every 4th layer, a 2:1 linear value/key head ratio, spatial merge
    2, one MTP layer) so the code paths exercised are the production ones.
    """
    text = TextConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=512,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        full_attention_interval=4,
        rope_theta=10000.0,
        # rotary width 16, i.e. an 8-entry table: sections (3, 3, 2) fill it the
        # same way the real (11, 11, 10) fills its 32 entries.
        partial_rotary_factor=0.5,
        mrope_section=(3, 3, 2),
        mtp_num_hidden_layers=1,
        dtype="float32",
        eos_token_id=0,
    )
    vision = VisionConfig(
        depth=vision_depth,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        patch_size=8,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=hidden_size,
        num_position_embeddings=64,
    )
    return ModelConfig(text=text, vision=vision)


PRESETS: dict[str, ModelConfig] = {
    # 4.66 B parameters — the full model, needs a real GPU
    "base": base_config(),
    # ~0.5 M parameters — CI, CPU smoke runs, shape tests
    "tiny": tiny_config(),
}

PRESET_NAMES = tuple(PRESETS)


def preset(name: str) -> ModelConfig:
    """A fresh copy of a named preset (never the shared instance)."""
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return ModelConfig.from_dict(PRESETS[name].to_dict())
