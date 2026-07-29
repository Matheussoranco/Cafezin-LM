"""Model and training configuration with presets sized for local hardware."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 16384
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4  # grouped-query attention; must divide n_heads
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_weights: bool = True
    mlp_multiple_of: int = 64  # SwiGLU hidden dim rounded up to this

    def __post_init__(self) -> None:
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim {self.dim} not divisible by n_heads {self.n_heads}")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads {self.n_heads} not divisible by n_kv_heads {self.n_kv_heads}"
            )

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainConfig:
    batch_size: int = 16  # micro-batch (per forward pass)
    grad_accum_steps: int = 4  # effective batch = batch_size * grad_accum_steps
    seq_len: int = 1024
    max_steps: int = 20000
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 500
    eval_iters: int = 100
    log_interval: int = 20
    seed: int = 1337
    compile: bool = False  # torch.compile (Triton is unavailable on Windows CUDA)
    # knowledge distillation (teacher must share the tokenizer/vocab)
    kd_alpha: float = 0.5  # weight of KD loss vs cross-entropy when a teacher is set
    kd_temperature: float = 2.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# Presets sized for a 6 GB laptop GPU (RTX 3050-class).
# Params counted with the default 16k vocab, weights tied.
PRESETS: dict[str, ModelConfig] = {
    # 2.9M params — smoke tests and CI
    "nano": ModelConfig(dim=128, n_layers=4, n_heads=4, n_kv_heads=2, max_seq_len=256),
    # 15.7M params — trains in a few hours, TinyStories-class fluency
    "micro": ModelConfig(dim=384, n_layers=6, n_heads=6, n_kv_heads=2, max_seq_len=512),
    # 32.0M params — good quality/speed balance for overnight runs
    "small": ModelConfig(dim=512, n_layers=8, n_heads=8, n_kv_heads=4, max_seq_len=1024),
    # 88.1M params — GPT-2-class capacity, the most this GPU trains comfortably
    "base": ModelConfig(dim=768, n_layers=12, n_heads=12, n_kv_heads=4, max_seq_len=1024),
}


def preset(name: str) -> ModelConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    cfg = PRESETS[name]
    return ModelConfig(**cfg.to_dict())
