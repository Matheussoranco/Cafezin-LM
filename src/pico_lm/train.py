"""Single-GPU training loop: AMP, grad accumulation, cosine LR, checkpoints, KD."""

from __future__ import annotations

import contextlib
import json
import math
import pickle
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import ModelConfig, TrainConfig
from .data import TokenDataset
from .model import Transformer


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def pick_dtype(device: str) -> torch.dtype:
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device == "cuda":
        return torch.float16
    return torch.float32


def autocast_ctx(device: str, dtype: torch.dtype):
    """Autocast on an accelerator; a no-op on CPU (fp32 autocast only warns)."""
    if device == "cpu" or dtype == torch.float32:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.split(":")[0], dtype=dtype)


def cosine_lr(step: int, cfg: TrainConfig) -> float:
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


def make_optimizer(model: Transformer, cfg: TrainConfig, device: str) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), fused=(device == "cuda")
    )


@torch.no_grad()
def estimate_loss(
    model: Transformer,
    datasets: dict[str, TokenDataset],
    cfg: TrainConfig,
    device: str,
    dtype: torch.dtype,
) -> dict[str, float]:
    model.eval()
    out = {}
    for split, ds in datasets.items():
        losses = torch.zeros(cfg.eval_iters)
        for i in range(cfg.eval_iters):
            x, y = ds.get_batch(cfg.batch_size, cfg.seq_len, device)
            with autocast_ctx(device, dtype):
                _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def kd_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    t = temperature
    return (
        F.kl_div(
            F.log_softmax(student_logits / t, dim=-1),
            F.log_softmax(teacher_logits / t, dim=-1),
            log_target=True,
            reduction="batchmean",
        )
        * (t * t)
        / student_logits.size(1)  # batchmean over (B, T, V) counts B only; normalize T
    )


def save_checkpoint(
    path: Path,
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    step: int,
    best_val: float,
    tokenizer_path: str | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_cfg": model_cfg.to_dict(),
            "train_cfg": train_cfg.to_dict(),
            "step": step,
            "best_val": best_val,
            "tokenizer_path": tokenizer_path,
        },
        path,
    )


def load_model(ckpt_path: str | Path, device: str = "cpu") -> tuple[Transformer, dict]:
    """Load a checkpoint, quantized or not.

    Quantized checkpoints carry a "quant" block describing how the Linear /
    embedding modules were replaced; dispatching here means `generate`, `chat`,
    `eval`, `info` and `bench` all accept them without knowing they exist.

    `--method dynamic` is the exception: torch.ao returns a live module, so that
    path pickles the module itself and cannot be read under weights_only=True.
    Handled below, so both quantization methods are loadable by every consumer.
    """
    try:
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    except pickle.UnpicklingError:
        # `quantize --method dynamic` hands torch.ao a live module and pickles it
        # whole, which weights_only=True refuses. It is our own artefact written
        # by our own CLI, so retry -- but only after the safe load has refused a
        # pickled global, never for a truncated or otherwise unreadable file.
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model = ckpt.pop("quantized_model", None)
        if model is None:
            raise
        return model.eval(), ckpt
    if isinstance(ckpt.get("quant"), dict):
        from .quant import build_quantized_shell

        cfg = ModelConfig.from_dict(ckpt["model_cfg"])
        model = build_quantized_shell(cfg, ckpt["quant"])
        state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
        model.load_state_dict(state, strict=False)
        return model.to(device).eval(), ckpt
    cfg = ModelConfig.from_dict(ckpt["model_cfg"])
    model = Transformer(cfg)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.to(device)
    return model, ckpt


def _stage_tokenizer(tokenizer_path: str | None, data_dir: Path, out_dir: Path) -> str | None:
    """Copy the tokenizer next to the checkpoints; return an absolute path to it.

    Checkpoints record this path, so `generate`/`chat` keep working after the
    data directory is deleted or the process runs from another directory.
    """
    src = Path(tokenizer_path) if tokenizer_path else data_dir / "tokenizer.json"
    if not src.exists():
        print(f"warning: tokenizer {src} not found; `generate` will need --tokenizer")
        return str(src.resolve())
    dst = out_dir / "tokenizer.json"
    if src.resolve() != dst.resolve():
        shutil.copyfile(src, dst)
    return str(dst.resolve())


def train(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    data_dir: str | Path,
    out_dir: str | Path,
    resume: bool = False,
    teacher_ckpt: str | None = None,
    tokenizer_path: str | None = None,
) -> float:
    device = pick_device()
    dtype = pick_dtype(device)
    torch.manual_seed(train_cfg.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(train_cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    if meta["vocab_size"] > model_cfg.vocab_size:
        raise ValueError(
            f"data vocab {meta['vocab_size']} exceeds model vocab {model_cfg.vocab_size}"
        )
    train_cfg.seq_len = min(train_cfg.seq_len, model_cfg.max_seq_len)
    train_cfg.validate()
    datasets = {
        "train": TokenDataset(data_dir / "train.bin"),
        "val": TokenDataset(data_dir / "val.bin"),
    }

    # Keep a copy of the tokenizer beside the checkpoints and record an absolute
    # path, so `generate`/`chat` work from any working directory later on.
    tokenizer_path = _stage_tokenizer(tokenizer_path, data_dir, out_dir)

    step, best_val = 0, float("inf")
    ckpt_path = out_dir / "last.pt"
    if resume and ckpt_path.exists():
        model, ckpt = load_model(ckpt_path, device)
        optimizer = make_optimizer(model, train_cfg, device)
        optimizer.load_state_dict(ckpt["optimizer"])
        step, best_val = ckpt["step"], ckpt.get("best_val", float("inf"))
        # the weights come from the checkpoint, so its architecture is the truth;
        # writing the caller's config here would silently desync cfg from weights
        model_cfg = model.cfg
        print(f"resumed from step {step} (best val {best_val:.4f})")
        if step >= train_cfg.max_steps:
            print(
                f"already at step {step} >= max_steps {train_cfg.max_steps}; "
                "nothing to do (raise --max-steps to continue)"
            )
            return best_val
    else:
        if resume:
            print(f"--resume requested but {ckpt_path} does not exist; starting from scratch")
        model = Transformer(model_cfg).to(device)
        optimizer = make_optimizer(model, train_cfg, device)
    train_cfg.seq_len = min(train_cfg.seq_len, model.cfg.max_seq_len)
    model.set_gradient_checkpointing(train_cfg.gradient_checkpointing)
    model.train()

    teacher = None
    if teacher_ckpt:
        teacher, _ = load_model(teacher_ckpt, device)
        if teacher.cfg.vocab_size != model_cfg.vocab_size:
            raise ValueError("teacher and student must share the same vocabulary")
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"distilling from teacher: {teacher.num_params() / 1e6:.1f}M params")

    if train_cfg.compile:
        model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    tokens_per_step = train_cfg.batch_size * train_cfg.grad_accum_steps * train_cfg.seq_len
    print(
        f"device={device} dtype={dtype} params={n_params / 1e6:.1f}M "
        f"tokens/step={tokens_per_step:,} checkpointing={train_cfg.gradient_checkpointing}"
    )

    scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16))
    log_path = out_dir / "log.csv"
    if not log_path.exists():
        log_path.write_text("step,loss,lr,tok_per_s\n", encoding="utf-8")

    t0 = time.time()
    running_loss = 0.0
    while step < train_cfg.max_steps:
        lr = cosine_lr(step, train_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(train_cfg.grad_accum_steps):
            x, y = datasets["train"].get_batch(train_cfg.batch_size, train_cfg.seq_len, device)
            with autocast_ctx(device, dtype):
                logits, ce = model(x, y)
                loss = ce
                if teacher is not None:
                    with torch.no_grad():
                        t_logits, _ = teacher(x)
                    loss = (1 - train_cfg.kd_alpha) * ce + train_cfg.kd_alpha * kd_loss(
                        logits, t_logits, train_cfg.kd_temperature
                    )
                loss = loss / train_cfg.grad_accum_steps
            scaler.scale(loss).backward()
            loss_accum += loss.item()
        if train_cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        step += 1
        running_loss += loss_accum

        if step % train_cfg.log_interval == 0:
            dt = time.time() - t0
            tok_s = tokens_per_step * train_cfg.log_interval / dt
            avg = running_loss / train_cfg.log_interval
            print(f"step {step:6d} | loss {avg:.4f} | lr {lr:.2e} | {tok_s:,.0f} tok/s")
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"{step},{avg:.6f},{lr:.6e},{tok_s:.0f}\n")
            running_loss, t0 = 0.0, time.time()

        if step % train_cfg.eval_interval == 0 or step == train_cfg.max_steps:
            losses = estimate_loss(model, datasets, train_cfg, device, dtype)
            print(f"eval @ {step}: train {losses['train']:.4f} | val {losses['val']:.4f}")
            raw = getattr(model, "_orig_mod", model)
            improved = losses["val"] < best_val
            if improved:
                best_val = losses["val"]
            # best_val must be updated *before* last.pt is written, otherwise a
            # resume reloads a stale best and can overwrite best.pt with a worse model
            save_checkpoint(
                out_dir / "last.pt",
                raw,
                optimizer,
                model_cfg,
                train_cfg,
                step,
                best_val,
                tokenizer_path,
            )
            if improved:
                save_checkpoint(
                    out_dir / "best.pt",
                    raw,
                    optimizer,
                    model_cfg,
                    train_cfg,
                    step,
                    best_val,
                    tokenizer_path,
                )
                print(f"  new best val {best_val:.4f} -> best.pt")
            t0 = time.time()

    return best_val
