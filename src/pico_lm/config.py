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
        if self.vocab_size <= 0 or self.dim <= 0 or self.n_layers <= 0:
            raise ValueError("vocab_size, dim, and n_layers must be positive")
        if self.n_heads <= 0 or self.n_kv_heads <= 0:
            raise ValueError("n_heads and n_kv_heads must be positive")
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim {self.dim} not divisible by n_heads {self.n_heads}")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads {self.n_heads} not divisible by n_kv_heads {self.n_kv_heads}"
            )
        if self.head_dim % 2:
            raise ValueError(f"head_dim {self.head_dim} must be even for RoPE")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.mlp_multiple_of <= 0:
            raise ValueError("mlp_multiple_of must be positive")

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
    gradient_checkpointing: bool = False  # recompute blocks to make long contexts fit
    # knowledge distillation (teacher must share the tokenizer/vocab)
    kd_alpha: float = 0.5  # weight of KD loss vs cross-entropy when a teacher is set
    kd_temperature: float = 2.0

    def validate(self) -> None:
        for name in ("batch_size", "grad_accum_steps", "seq_len", "max_steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.lr <= 0 or self.min_lr < 0 or self.min_lr > self.lr:
            raise ValueError("learning rates must satisfy 0 <= min_lr <= lr")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.eval_interval <= 0 or self.eval_iters <= 0 or self.log_interval <= 0:
            raise ValueError("evaluation and logging intervals must be positive")
        if not 0.0 <= self.kd_alpha <= 1.0:
            raise ValueError("kd_alpha must be in [0, 1]")
        if self.kd_temperature <= 0:
            raise ValueError("kd_temperature must be positive")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SFTConfig:
    """Supervised fine-tuning schedule — deliberately *not* TrainConfig.

    Pretraining LRs (6e-4) blow a pretrained model apart in a few hundred SFT
    steps. Fine-tuning wants roughly an order of magnitude less, a short warmup,
    and far fewer steps, because the dataset is small and the model is already
    trained. Field names mirror TrainConfig so `cosine_lr` works on both.
    """

    batch_size: int = 16
    grad_accum_steps: int = 1
    max_len: int = 512  # per-example cap; longer examples are left-truncated
    max_steps: int = 2000
    lr: float = 1e-4
    min_lr: float = 1e-5
    warmup_steps: int = 50
    weight_decay: float = 0.0  # small dataset: decay fights the pretrained weights
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 200
    eval_iters: int = 40
    log_interval: int = 20
    seed: int = 1337
    compile: bool = False
    gradient_checkpointing: bool = False
    # With --replay-data, retain this fraction of ordinary language-model loss.
    # Rehearsing pretraining text reduces catastrophic forgetting on narrow SFT data.
    replay_weight: float = 0.1
    # behavioural evaluation during training (see pico_lm.bench)
    bench_examples: int = 120  # held-out prompts scored by exact match at each eval
    select_by: str = "score"  # "score" (behavioural) or "loss" — see finetune()

    def validate(self) -> None:
        for name in ("batch_size", "grad_accum_steps", "max_len", "max_steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.lr <= 0 or self.min_lr < 0 or self.min_lr > self.lr:
            raise ValueError("learning rates must satisfy 0 <= min_lr <= lr")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.eval_interval <= 0 or self.eval_iters <= 0 or self.log_interval <= 0:
            raise ValueError("evaluation and logging intervals must be positive")
        if not 0.0 <= self.replay_weight < 1.0:
            raise ValueError("replay_weight must be in [0, 1)")
        if self.select_by not in {"score", "loss"}:
            raise ValueError("select_by must be 'score' or 'loss'")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SFTConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# Presets sized for a 6 GB laptop GPU (RTX 3050-class).
# Params counted with the default 16k vocab, weights tied.
PRESETS: dict[str, ModelConfig] = {
    # 2.9M params — smoke tests and CI
    "nano": ModelConfig(dim=128, n_layers=4, n_heads=4, n_kv_heads=2, max_seq_len=256),
    # 15.7M params — trains in a couple of hours, TinyStories-class fluency
    "micro": ModelConfig(dim=384, n_layers=6, n_heads=6, n_kv_heads=2, max_seq_len=512),
    # 32.0M params — good quality/speed balance for overnight runs
    "small": ModelConfig(dim=512, n_layers=8, n_heads=8, n_kv_heads=4, max_seq_len=1024),
    # 88.1M params — GPT-2-class capacity, the most this GPU trains comfortably
    "base": ModelConfig(dim=768, n_layers=12, n_heads=12, n_kv_heads=4, max_seq_len=1024),
    # 15.7M params with a 4096-token window. Use activation checkpointing and
    # a small micro-batch on 6 GB GPUs; parameter count is unchanged by context.
    "micro-long": ModelConfig(
        dim=384,
        n_layers=6,
        n_heads=6,
        n_kv_heads=2,
        max_seq_len=4096,
        rope_theta=100000.0,
    ),
}


def preset(name: str) -> ModelConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    cfg = PRESETS[name]
    return ModelConfig(**cfg.to_dict())
