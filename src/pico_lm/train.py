"""Single-GPU training loop: AMP, grad accumulation, cosine LR, checkpoints, KD."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import ModelConfig, TrainConfig
from .data import TokenDataset
from .model import Transformer
from .provenance import capture_rng, manifest, restore_rng

logger = logging.getLogger(__name__)


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


def resolve_fused(device: str, option: str | bool | None) -> bool:
    """Map a ``--fused auto/on/off`` option to an AdamW ``fused`` flag.

    ``auto`` means fused on CUDA, plain elsewhere. Booleans are accepted for
    backward compatibility (True -> "on", False -> "off").
    """
    if option is None:
        option = "auto"
    if isinstance(option, bool):
        return option
    opt = str(option).lower()
    if opt in {"off", "false", "0", "no"}:
        return False
    if opt in {"on", "true", "1", "yes"}:
        return True
    if opt == "auto":
        return device.split(":")[0] == "cuda"
    raise ValueError(f"fused must be one of 'auto', 'on', 'off', got {option!r}")


def _build_adamw(groups, cfg: TrainConfig, want_fused: bool) -> torch.optim.AdamW:
    try:
        return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), fused=want_fused)
    except TypeError as exc:
        # Old torch builds have no `fused` kwarg at all — retry plain, even
        # when want_fused is False, since the kwarg itself is unknown.
        logger.warning("AdamW fused kwarg unsupported (%s); using unfused AdamW", exc)
        return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    except ValueError as exc:
        # Torch knows `fused` but rejects the value for this device/dtype.
        if want_fused:
            logger.warning("AdamW fused kernel unavailable (%s); using fused=False", exc)
            return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
        raise


def make_optimizer(
    model: Transformer,
    cfg: TrainConfig,
    device: str,
    fused: str | bool | None = None,
) -> torch.optim.AdamW:
    if fused is None:
        fused = getattr(cfg, "fused", "auto")
    want_fused = resolve_fused(device, fused)
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return _build_adamw(groups, cfg, want_fused)


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
            "rng_state": capture_rng(),
            "scaler": getattr(model, "_training_scaler", None).state_dict()
            if getattr(model, "_training_scaler", None) is not None
            else {},
            "provenance": getattr(model, "_provenance", None),
        },
        path,
    )


def load_model(ckpt_path: str | Path, device: str = "cpu") -> tuple[Transformer, dict]:
    """Load a checkpoint, quantized or not.

    Quantized checkpoints carry a "quant" block describing how the Linear /
    embedding modules were replaced; dispatching here means `generate`, `chat`,
    `eval`, `info` and `bench` all accept them without knowing they exist.

    Dynamic exports store a state dictionary and rebuild their CPU shell locally.
    Legacy pickled modules are deliberately rejected; re-export original FP weights.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    if ckpt.get("format") == "dynamic-int8-state-v1":
        if device != "cpu":
            raise ValueError("dynamic int8 checkpoints require device='cpu'")
        model = Transformer(ModelConfig.from_dict(ckpt["model_cfg"])).eval()
        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        model.load_state_dict(ckpt["model"])
        return model.eval(), ckpt
    if isinstance(ckpt.get("quant"), dict):
        from .quant import build_quantized_shell

        cfg = ModelConfig.from_dict(ckpt["model_cfg"])
        model = build_quantized_shell(cfg, ckpt["quant"])
        state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
        model.load_state_dict(state)
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
    init_from: str | Path | None = None,
    teacher_ckpt: str | None = None,
    tokenizer_path: str | None = None,
) -> float:
    """Run the pretraining loop.

    ``resume=True`` is an *exact* resume from ``<out>/last.pt``: model,
    optimizer state, step counter, best-val, AMP scaler and RNG states are all
    restored (checkpoints already save them; ``--help`` says so). ``init_from``
    is a *warm-start, not an exact resume*: compatible weights are copied into
    a fresh model/schedule at step 0 with a fresh optimizer — intended for
    context extension (larger ``max_seq_len`` / ``rope_theta``).
    """
    if resume and init_from:
        raise ValueError("--resume and --init-from are mutually exclusive")
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
    if train_cfg.seq_len > model_cfg.max_seq_len:
        import warnings as _warnings

        _msg = (
            f"train seq_len {train_cfg.seq_len} exceeds model max_seq_len "
            f"{model_cfg.max_seq_len}; clamping to {model_cfg.max_seq_len}"
        )
        _warnings.warn(_msg, UserWarning, stacklevel=2)
        logger.warning(_msg)
        print(f"warning: {_msg}")
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
            raise SystemExit(
                f"already at step {step} >= max_steps {train_cfg.max_steps}; nothing to do. "
                f"Raise --max-steps to continue (e.g. --resume --max-steps {step + 1}) "
                "or start a new --out dir."
            )
    elif init_from:
        source, source_ckpt = load_model(init_from, device)
        source_cfg = source.cfg
        architecture = ("vocab_size", "dim", "n_layers", "n_heads", "n_kv_heads")
        mismatches = [
            f"{name}: {getattr(source_cfg, name)} != {getattr(model_cfg, name)}"
            for name in architecture
            if getattr(source_cfg, name) != getattr(model_cfg, name)
        ]
        if mismatches:
            raise ValueError(
                "--init-from checkpoint is not weight-compatible with the requested model: "
                + ", ".join(mismatches)
            )
        # RoPE buffers are deliberately non-persistent. Building the requested
        # shell first lets a trained model adopt a larger context window and a
        # different RoPE base while retaining every learned parameter.
        model = Transformer(model_cfg).to(device)
        model.load_state_dict(source.state_dict())
        del source
        optimizer = make_optimizer(model, train_cfg, device)
        print(
            f"initialised from {init_from} (source step {source_ckpt.get('step')}); "
            f"fresh optimizer, context {source_cfg.max_seq_len} -> {model_cfg.max_seq_len}, "
            f"rope_theta {source_cfg.rope_theta:g} -> {model_cfg.rope_theta:g}"
        )
    else:
        if resume:
            print(f"--resume requested but {ckpt_path} does not exist; starting from scratch")
        model = Transformer(model_cfg).to(device)
        optimizer = make_optimizer(model, train_cfg, device)
    if train_cfg.seq_len > model.cfg.max_seq_len:
        import warnings as _warnings2

        _msg2 = (
            f"train seq_len {train_cfg.seq_len} exceeds model max_seq_len "
            f"{model.cfg.max_seq_len}; clamping to {model.cfg.max_seq_len}"
        )
        _warnings2.warn(_msg2, UserWarning, stacklevel=2)
        logger.warning(_msg2)
        print(f"warning: {_msg2}")
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
    raw_model = getattr(model, "_orig_mod", model)
    raw_model._training_scaler = scaler
    raw_model._provenance = manifest(
        [data_dir / "train.bin", data_dir / "val.bin", tokenizer_path, init_from, teacher_ckpt],
        model=model_cfg.to_dict(),
        training=train_cfg.to_dict(),
        device=device,
    )
    if resume and ckpt_path.exists():
        if ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        restore_rng(ckpt.get("rng_state"))
    log_path = out_dir / "log.csv"
    if not log_path.exists():
        log_path.write_text("step,loss,lr,tok_per_s\n", encoding="utf-8")

    t0 = time.perf_counter()
    n_logged = 0
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
        n_logged += 1

        if step % train_cfg.log_interval == 0:
            dt = max(time.perf_counter() - t0, 1e-9)
            tok_s = tokens_per_step * n_logged / dt
            avg = running_loss / n_logged
            print(f"step {step:6d} | loss {avg:.4f} | lr {lr:.2e} | {tok_s:,.0f} tok/s")
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"{step},{avg:.6f},{lr:.6e},{tok_s:.0f}\n")
            running_loss, n_logged, t0 = 0.0, 0, time.perf_counter()

        if step % train_cfg.eval_interval == 0 or step == train_cfg.max_steps:
            eval_start = time.perf_counter()
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
            t0 += time.perf_counter() - eval_start

    return best_val
