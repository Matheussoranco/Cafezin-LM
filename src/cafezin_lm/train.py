"""Training loop: AdamW, cosine decay with warm-up, gradient checkpointing.

Designed to be reproducible from a seed and resumable from a sharded
checkpoint. The MTP head (when present) is trained with an auxiliary
cross-entropy loss against targets shifted by two positions.

No external trainers: ~300 lines of standard PyTorch so the exact behaviour is
auditable and the checkpoint layout is explicit.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from .checkpoint import checkpoint_assets, load_hf_checkpoint, save_hf_checkpoint
from .config import ModelConfig, tiny_config
from .data import TokenDataset, load_meta
from .model import (
    build_model,
    chunk_gated_delta_rule,
)


@dataclass
class TrainConfig:
    out_dir: str = "out"
    data_dir: str = "data"
    seed: int = 42
    seq_len: int = 4096
    batch_size: int = 4
    grad_accum: int = 8
    steps: int = 100_000
    eval_every: int = 1_000
    eval_batches: int = 32
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 2_000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    gradient_checkpointing: bool = True
    mtp_weight: float = 0.3
    compile_model: bool = False
    log_every: int = 50
    save_every: int = 5_000
    keep_last: int = 3
    device: str = "cuda"
    dtype: str = "bfloat16"
    torch_seed: int | None = None
    # resume
    resume: str | None = None
    resume_step: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> TrainConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class TrainingState:
    step: int = 0
    best_val_loss: float = float("inf")
    tokens_seen: int = 0
    rng_state: dict | None = None


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    use_fused = next(model.parameters()).is_cuda
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), fused=use_fused)


def build_scheduler(
    optimizer: torch.optim.AdamW, cfg: TrainConfig
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return cfg.min_lr / cfg.lr + (1 - cfg.min_lr / cfg.lr) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    state: TrainingState,
    cfg: TrainConfig,
    train_cfg: ModelConfig,
) -> Path:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    step_tag = f"step-{state.step:09d}"
    checkpoint_dir = out / step_tag
    if checkpoint_dir.exists():
        raise FileExistsError(f"checkpoint already exists: {checkpoint_dir}")
    with tempfile.TemporaryDirectory(prefix=".checkpoint-", dir=out) as tmp:
        staged = Path(tmp) / step_tag
        save_hf_checkpoint(
            model,
            staged,
            cfg=train_cfg,
            metadata={"step": str(state.step), "tokens_seen": str(state.tokens_seen)},
            tokenizer_files=checkpoint_assets(cfg.data_dir),
            dtype_override="native",
        )
        torch.save(
            {
                "step": state.step,
                "tokens_seen": state.tokens_seen,
                "best_val_loss": state.best_val_loss,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else None
                ),
            },
            staged / "trainer_state.pt",
        )
        staged.rename(checkpoint_dir)
    if cfg.keep_last > 0:
        checkpoints = sorted(
            p for p in out.glob("step-" + "[0-9]" * 9) if (p / "trainer_state.pt").is_file()
        )
        for old in checkpoints[: -cfg.keep_last]:
            shutil.rmtree(old, ignore_errors=True)
    return checkpoint_dir


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    path: str | Path,
    device: str,
) -> TrainingState:
    path = Path(path)
    trainer_file = path / "trainer_state.pt"
    if not trainer_file.exists():
        raise FileNotFoundError(f"{trainer_file} not found")
    blob = torch.load(trainer_file, map_location="cpu", weights_only=True)
    load_hf_checkpoint(model, path)
    if optimizer is not None:
        optimizer.load_state_dict(blob["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(blob["scheduler"])
    torch.set_rng_state(blob["rng_state"])
    if blob["cuda_rng_state"] is not None and torch.device(device).type == "cuda":
        torch.cuda.set_rng_state_all(blob["cuda_rng_state"])
    return TrainingState(
        step=blob["step"],
        tokens_seen=blob["tokens_seen"],
        best_val_loss=blob["best_val_loss"],
        rng_state=blob["rng_state"],
    )


def evaluate(
    model: nn.Module,
    val_data: TokenDataset,
    cfg: TrainConfig,
    train_cfg: ModelConfig,
    device: str,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(cfg.eval_batches):
            x, y = val_data.get_batch(cfg.batch_size, cfg.seq_len, device)
            _, loss = model(x, targets=y, mtp_weight=cfg.mtp_weight)
            total_loss += loss.item()
    model.train()
    return total_loss / cfg.eval_batches


def train(
    model_cfg: ModelConfig | None = None,
    train_cfg: TrainConfig | None = None,
) -> Path:
    model_cfg = model_cfg or tiny_config()
    train_cfg = train_cfg or TrainConfig()
    if min(train_cfg.seq_len, train_cfg.batch_size, train_cfg.grad_accum) < 1:
        raise ValueError("seq_len, batch_size and grad_accum must be positive")

    _set_seed(train_cfg.seed)

    device = train_cfg.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    meta = load_meta(train_cfg.data_dir)
    required_vocab = meta.get("max_token_id", meta["vocab_size"] - 1) + 1
    if required_vocab > model_cfg.text.vocab_size:
        raise ValueError(
            f"data requires {required_vocab} embedding rows; model has {model_cfg.text.vocab_size}"
        )

    train_data = TokenDataset(Path(train_cfg.data_dir) / "train.bin", dtype=meta["dtype"])
    val_data = None
    if train_cfg.eval_every > 0 and meta["val_tokens"] > 0:
        if train_cfg.eval_batches < 1:
            raise ValueError("eval_batches must be positive when validation is enabled")
        val_data = TokenDataset(Path(train_cfg.data_dir) / "val.bin", dtype=meta["dtype"])

    torch_dtype = _get_dtype(train_cfg.dtype)
    model = build_model(model_cfg, seed=train_cfg.torch_seed).to(device=device, dtype=torch_dtype)

    if train_cfg.gradient_checkpointing:
        model.set_gradient_checkpointing(True)

    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)

    state = TrainingState()
    if train_cfg.resume:
        print(f"resuming from {train_cfg.resume}")
        state = load_checkpoint(model, optimizer, scheduler, train_cfg.resume, device)
        if train_cfg.resume_step is not None:
            state.step = train_cfg.resume_step
        if state.step > train_cfg.steps:
            raise ValueError("resume step exceeds the requested total steps")
        if (
            state.step == train_cfg.steps
            and train_cfg.resume_step is None
            and Path(train_cfg.resume).parent.resolve() == Path(train_cfg.out_dir).resolve()
        ):
            return Path(train_cfg.resume)

    train_model = model
    if train_cfg.compile_model:
        print("compiling model...")
        train_model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

    print(
        f"model params: {model.num_params():,} | "
        f"train tokens: {meta['train_tokens']:,} | "
        f"val tokens: {meta['val_tokens']:,} | "
        f"seq_len: {train_cfg.seq_len} | "
        f"batch: {train_cfg.batch_size} x {train_cfg.grad_accum} | "
        f"device: {device} | "
        f"dtype: {train_cfg.dtype} | "
        f"start step: {state.step}"
    )

    micro_batch = train_cfg.batch_size
    grad_accum = train_cfg.grad_accum
    seq_len = min(train_cfg.seq_len, len(train_data) - 1)
    tokens_per_step = micro_batch * seq_len * grad_accum

    while state.step < train_cfg.steps:
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        for _ in range(grad_accum):
            x, y = train_data.get_batch(micro_batch, seq_len, device)
            _, loss = train_model(
                x,
                targets=y,
                mtp_weight=train_cfg.mtp_weight,
                delta_rule=chunk_gated_delta_rule,
            )
            loss = loss / grad_accum
            loss.backward()

        if train_cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

        optimizer.step()
        scheduler.step()

        state.step += 1
        state.tokens_seen += tokens_per_step

        if train_cfg.log_every > 0 and state.step % train_cfg.log_every == 0:
            lr = scheduler.get_last_lr()[0]
            dt = time.perf_counter() - t0
            tps = tokens_per_step / dt
            print(
                f"step {state.step:9d} | loss {loss.item() * grad_accum:.4f} | "
                f"lr {lr:.2e} | {tps:,.0f} tok/s | "
                f"{state.tokens_seen / 1e9:.3f}B tokens"
            )

        if val_data is not None and state.step % train_cfg.eval_every == 0:
            val_loss = evaluate(model, val_data, train_cfg, model_cfg, device)
            print(f"  val loss {val_loss:.4f} (best {state.best_val_loss:.4f})")
            if val_loss < state.best_val_loss:
                state.best_val_loss = val_loss

        if (
            train_cfg.save_every > 0
            and state.step < train_cfg.steps
            and state.step % train_cfg.save_every == 0
        ):
            save_checkpoint(model, optimizer, scheduler, state, train_cfg, model_cfg)

    save_checkpoint(model, optimizer, scheduler, state, train_cfg, model_cfg)
    return Path(train_cfg.out_dir) / f"step-{state.step:09d}"


def train_from_config(config_path: str | Path) -> Path:
    """Load both configs from a single JSON (convenience for scripts)."""
    data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    model_cfg = ModelConfig.from_dict(data["model"])
    train_cfg = TrainConfig.from_dict(data["train"])
    return train(model_cfg, train_cfg)
