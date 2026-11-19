"""Continued pretraining pipeline for Cafezin LM.

Implements MS-001: Load S5-nano weights, grow embeddings for new tokens,
initialize fresh Adam with peak LR 3e-4 and 2k warmup, train on single 48GB GPU
with seq_len=2048 baseline / 4096 stretch, grad_accum=32, gradient checkpointing,
MTP weight=0.3. Checkpoints compatible with existing export command.
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .checkpoint import (
    checkpoint_assets,
    download_file,
    load_hf_checkpoint,
    save_hf_checkpoint,
)
from .config import REFERENCE_REPO, REFERENCE_REVISION, ModelConfig
from .data import TokenDataset, load_meta
from .model import (
    CafezinLM,
    build_model,
    chunk_gated_delta_rule,
)
from .tokenizer import Tokenizer
from .tokenizer_extension import grow_embeddings
from .train import (
    TrainConfig,
    TrainingState,
    _get_dtype,
    _set_seed,
    build_optimizer,
    build_scheduler,
    evaluate,
    load_checkpoint,
    save_checkpoint,
)

if TYPE_CHECKING:
    pass


@dataclass
class ContinuedPretrainConfig:
    """Configuration for continued pretraining from S5-nano."""

    # Source checkpoint (S5-nano)
    source_repo: str = REFERENCE_REPO
    source_revision: str = REFERENCE_REVISION
    source_cache_dir: str | None = None

    # Extended tokenizer and data
    tokenizer_path: str | None = None  # Path to extended tokenizer.json
    data_dir: str = "data"

    # Training hyperparameters (MS-001 spec)
    out_dir: str = "out-continued"
    seed: int = 42
    seq_len: int = 2048  # baseline; 4096 stretch
    batch_size: int = 1
    grad_accum: int = 32
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

    # Resume
    resume: str | None = None
    resume_step: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ContinuedPretrainConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> ContinuedPretrainConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_train_config(self) -> TrainConfig:
        """Convert to standard TrainConfig for the training loop."""
        return TrainConfig(
            out_dir=self.out_dir,
            data_dir=self.data_dir,
            seed=self.seed,
            seq_len=self.seq_len,
            batch_size=self.batch_size,
            grad_accum=self.grad_accum,
            steps=self.steps,
            eval_every=self.eval_every,
            eval_batches=self.eval_batches,
            lr=self.lr,
            min_lr=self.min_lr,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            beta1=self.beta1,
            beta2=self.beta2,
            grad_clip=self.grad_clip,
            gradient_checkpointing=self.gradient_checkpointing,
            mtp_weight=self.mtp_weight,
            compile_model=self.compile_model,
            log_every=self.log_every,
            save_every=self.save_every,
            keep_last=self.keep_last,
            device=self.device,
            dtype=self.dtype,
            torch_seed=self.torch_seed,
            resume=self.resume,
            resume_step=self.resume_step,
        )


def download_s5nano_checkpoint(
    cache_dir: str | Path | None = None,
    repo: str = REFERENCE_REPO,
    revision: str = REFERENCE_REVISION,
) -> Path:
    """Download the S5-nano checkpoint to local cache."""
    cache = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "cafezin-lm" / "s5nano"
    cache.mkdir(parents=True, exist_ok=True)

    # Download all safetensors files and config
    files = [
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]

    # Also download all shards
    from .checkpoint import hf_files
    all_files = hf_files(repo, revision)
    safetensors_files = [f for f in all_files if f.endswith(".safetensors")]

    for f in files + safetensors_files:
        download_file(f, cache / f, repo=repo, revision=revision)

    return cache


def load_s5nano_into_base(
    checkpoint_dir: str | Path,
    extended_tokenizer: Tokenizer | None = None,
    device: str = "cpu",
) -> tuple[CafezinLM, ModelConfig]:
    """Load S5-nano weights into base preset, optionally growing embeddings.

    Args:
        checkpoint_dir: Path to S5-nano checkpoint directory
        extended_tokenizer: Optional extended tokenizer for embedding growth
        device: Device to load model on

    Returns:
        Tuple of (model, config)
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Load config from checkpoint
    from .checkpoint import import_config
    model_cfg = import_config(checkpoint_dir)

    # Build model on meta device first
    with torch.device("meta"):
        model = build_model(model_cfg)

    # Load weights with assignment (materializes on device)
    load_hf_checkpoint(model, checkpoint_dir, assign=True)

    # Recreate RoPE buffers (not in checkpoint)
    from .model import TextRotaryEmbedding, VisionRotaryEmbedding
    model.model.language_model.rotary_emb = TextRotaryEmbedding(model_cfg.text)
    model.model.visual.rotary_pos_emb = VisionRotaryEmbedding(model_cfg.vision)

    # Move to target device
    model.to(device=device)

    # Grow embeddings if extended tokenizer provided
    if extended_tokenizer is not None:
        # Need original tokenizer - load from checkpoint
        original_tokenizer_path = checkpoint_dir / "tokenizer.json"
        if original_tokenizer_path.exists():
            original_tokenizer = Tokenizer(original_tokenizer_path)
        else:
            # Fallback: download from hub
            original_tokenizer = Tokenizer.from_hub()

        # Get new tokens
        orig_vocab = original_tokenizer._tokenizer.get_vocab(with_added_tokens=True)
        ext_vocab = extended_tokenizer._tokenizer.get_vocab(with_added_tokens=True)
        new_tokens = [tok for tok, tid in ext_vocab.items() if tid not in orig_vocab.values()]

        if new_tokens:
            print(f"Growing embeddings for {len(new_tokens)} new tokens...")
            grow_embeddings(model, original_tokenizer, extended_tokenizer, new_tokens)
            print(
                f"Embeddings grown: {model.cfg.text.vocab_size} "
                f"-> {extended_tokenizer.vocab_size}"
            )

    return model, model_cfg


def continued_pretrain(
    cfg: ContinuedPretrainConfig | None = None,
    model_cfg: ModelConfig | None = None,
) -> Path:
    """Run continued pretraining from S5-nano.

    This is the main entry point for MS-001. It:
    1. Downloads/loads S5-nano checkpoint
    2. Loads extended tokenizer (if provided)
    3. Grows embeddings for new tokens
    4. Initializes fresh Adam optimizer with peak LR 3e-4, warmup 2k
    5. Trains with seq_len=2048/4096, grad_accum=32, gradient checkpointing, MTP=0.3
    6. Saves checkpoints compatible with export command

    Args:
        cfg: ContinuedPretrainConfig with all settings
        model_cfg: Optional ModelConfig override (loaded from S5-nano if not provided)

    Returns:
        Path to final checkpoint directory
    """
    cfg = cfg or ContinuedPretrainConfig()
    _set_seed(cfg.seed)

    device = cfg.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # Step 1: Get S5-nano checkpoint
    if cfg.source_cache_dir:
        s5nano_dir = Path(cfg.source_cache_dir)
    else:
        print("Downloading S5-nano checkpoint...")
        s5nano_dir = download_s5nano_checkpoint(
            cache_dir=cfg.source_cache_dir,
            repo=cfg.source_repo,
            revision=cfg.source_revision,
        )
    print(f"Using S5-nano from: {s5nano_dir}")

    # Step 2: Load extended tokenizer if provided
    extended_tokenizer = None
    if cfg.tokenizer_path:
        extended_tokenizer = Tokenizer(cfg.tokenizer_path)
        print(f"Loaded extended tokenizer: vocab_size={extended_tokenizer.vocab_size}")

    # Step 3: Load S5-nano into base preset, grow embeddings
    model, loaded_model_cfg = load_s5nano_into_base(
        s5nano_dir,
        extended_tokenizer=extended_tokenizer,
        device=device,
    )

    # Use loaded config if not overridden
    model_cfg = model_cfg or loaded_model_cfg

    # Verify vocab size matches tokenizer if extended
    if extended_tokenizer and model_cfg.text.vocab_size != extended_tokenizer.vocab_size:
        raise ValueError(
            f"Model vocab size ({model_cfg.text.vocab_size}) != "
            f"extended tokenizer vocab size ({extended_tokenizer.vocab_size})"
        )

    # Step 4: Prepare data
    meta = load_meta(cfg.data_dir)
    required_vocab = meta.get("max_token_id", meta["vocab_size"] - 1) + 1
    if required_vocab > model_cfg.text.vocab_size:
        raise ValueError(
            f"Data requires {required_vocab} embedding rows; model has {model_cfg.text.vocab_size}"
        )

    train_data = TokenDataset(Path(cfg.data_dir) / "train.bin", dtype=meta["dtype"])
    val_data = None
    if cfg.eval_every > 0 and meta["val_tokens"] > 0:
        if cfg.eval_batches < 1:
            raise ValueError("eval_batches must be positive when validation is enabled")
        val_data = TokenDataset(Path(cfg.data_dir) / "val.bin", dtype=meta["dtype"])

    # Step 5: Setup model for training
    torch_dtype = _get_dtype(cfg.dtype)
    model = model.to(device=device, dtype=torch_dtype)

    if cfg.gradient_checkpointing:
        model.set_gradient_checkpointing(True)

    # Fresh Adam optimizer (MS-001 requirement)
    optimizer = build_optimizer(model, cfg.to_train_config())
    scheduler = build_scheduler(optimizer, cfg.to_train_config())

    state = TrainingState()

    # Resume handling
    if cfg.resume:
        print(f"Resuming from {cfg.resume}")
        state = load_checkpoint(model, optimizer, scheduler, cfg.resume, device)
        if cfg.resume_step is not None:
            state.step = cfg.resume_step
        if state.step > cfg.steps:
            raise ValueError("resume step exceeds the requested total steps")

    train_model = model
    if cfg.compile_model:
        print("Compiling model...")
        train_model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

    print(
        f"model params: {model.num_params():,} | "
        f"train tokens: {meta['train_tokens']:,} | "
        f"val tokens: {meta['val_tokens']:,} | "
        f"seq_len: {cfg.seq_len} | "
        f"batch: {cfg.batch_size} x {cfg.grad_accum} | "
        f"device: {device} | "
        f"dtype: {cfg.dtype} | "
        f"start step: {state.step} | "
        f"mtp_weight: {cfg.mtp_weight} | "
        f"lr: {cfg.lr} | "
        f"warmup: {cfg.warmup_steps}"
    )

    # Record run settings
    run_settings = {
        "continued_pretrain_config": cfg.to_dict(),
        "model_config": model_cfg.to_dict(),
        "data_meta": meta,
        "start_step": state.step,
        "peak_memory_gb": None,  # Will be filled during training
        "throughput_tokens_per_sec": None,
    }
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_settings.json").write_text(json.dumps(run_settings, indent=2), encoding="utf-8")

    micro_batch = cfg.batch_size
    grad_accum = cfg.grad_accum
    seq_len = min(cfg.seq_len, len(train_data) - 1)
    tokens_per_step = micro_batch * seq_len * grad_accum

    peak_memory = 0.0
    total_tokens = 0
    step_times = []

    while state.step < cfg.steps:
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        for _ in range(grad_accum):
            x, y = train_data.get_batch(micro_batch, seq_len, device)
            _, loss = train_model(
                x,
                targets=y,
                mtp_weight=cfg.mtp_weight,
                delta_rule=chunk_gated_delta_rule,
            )
            loss = loss / grad_accum
            loss.backward()

        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()
        scheduler.step()

        state.step += 1
        state.tokens_seen += tokens_per_step
        total_tokens += tokens_per_step

        dt = time.perf_counter() - t0
        step_times.append(dt)

        # Track peak memory
        if device.startswith("cuda"):
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated() / 1e9)

        if cfg.log_every > 0 and state.step % cfg.log_every == 0:
            lr = scheduler.get_last_lr()[0]
            tps = tokens_per_step / dt
            avg_tps = tokens_per_step * len(step_times) / sum(step_times) if step_times else tps
            print(
                f"step {state.step:9d} | loss {loss.item() * grad_accum:.4f} | "
                f"lr {lr:.2e} | {tps:,.0f} tok/s (avg {avg_tps:,.0f}) | "
                f"{state.tokens_seen / 1e9:.3f}B tokens | "
                f"peak mem: {peak_memory:.2f} GB"
            )

            # Update run settings with latest metrics
            run_settings["peak_memory_gb"] = round(peak_memory, 2)
            run_settings["throughput_tokens_per_sec"] = round(avg_tps, 1)
            run_settings["current_step"] = state.step
            run_settings["tokens_seen"] = state.tokens_seen
            (out_dir / "run_settings.json").write_text(
                json.dumps(run_settings, indent=2), encoding="utf-8"
            )

        if val_data is not None and state.step % cfg.eval_every == 0:
            val_loss = evaluate(model, val_data, cfg.to_train_config(), model_cfg, device)
            print(f"  val loss {val_loss:.4f} (best {state.best_val_loss:.4f})")
            if val_loss < state.best_val_loss:
                state.best_val_loss = val_loss

        if (
            cfg.save_every > 0
            and state.step < cfg.steps
            and state.step % cfg.save_every == 0
        ):
            save_checkpoint(model, optimizer, scheduler, state, cfg.to_train_config(), model_cfg)

    # Final checkpoint
    final_checkpoint = save_checkpoint(
        model, optimizer, scheduler, state, cfg.to_train_config(), model_cfg
    )
    print(f"Training complete. Final checkpoint: {final_checkpoint}")

    # Update final run settings
    run_settings["final_checkpoint"] = str(final_checkpoint)
    run_settings["end_step"] = state.step
    run_settings["total_tokens_seen"] = state.tokens_seen
    run_settings["peak_memory_gb"] = round(peak_memory, 2)
    avg_tps = total_tokens / sum(step_times) if step_times else 0
    run_settings["avg_throughput_tokens_per_sec"] = round(avg_tps, 1)
    (out_dir / "run_settings.json").write_text(json.dumps(run_settings, indent=2), encoding="utf-8")

    return final_checkpoint


def smoke_test_cpu(
    checkpoint_dir: str | Path,
    tokenizer_path: str | Path,
    prompt: str = "Olá, mundo!",
    max_new: int = 32,
) -> str:
    """Smoke test: clean reload and text generation on CPU fixture.

    Exercises: init -> train -> checkpoint -> export -> clean reload -> generate
    """
    checkpoint_dir = Path(checkpoint_dir)
    tokenizer_path = Path(tokenizer_path)

    # Export checkpoint to HF format
    from .cli import _checkpoint_model

    with tempfile.TemporaryDirectory() as tmp:
        export_dir = Path(tmp) / "export"
        model = _checkpoint_model(checkpoint_dir, "base")
        assets = checkpoint_assets(
            checkpoint_dir.parent if checkpoint_dir.is_dir() else checkpoint_dir
        )
        # Use extended tokenizer
        assets["tokenizer.json"] = tokenizer_path.read_bytes()

        save_hf_checkpoint(
            model,
            export_dir,
            cfg=model.cfg,
            tokenizer_files=assets,
            dtype_override="native",
        )

        # Clean reload on CPU
        tok = Tokenizer(tokenizer_path)
        from .checkpoint import import_config, load_hf_checkpoint
        cfg = import_config(export_dir)
        model2 = build_model(cfg).to("cpu")
        load_hf_checkpoint(model2, export_dir, assign=True)

        # Generate
        model2.eval()
        input_ids = torch.tensor([tok.encode(prompt, eos=False)], dtype=torch.long)
        eos = tok.eos_token_id if tok.eos_token_id is not None else cfg.text.eos_token_id
        out = model2.generate(input_ids, max_new_tokens=max_new, temperature=0.0, eos_token_id=eos)
        generated = tok.decode(out[0].tolist())
        print(f"Smoke test generation: {generated}")
        return generated


def continued_pretrain_from_config(config_path: str | Path) -> Path:
    """Load config from JSON and run continued pretraining."""
    data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    cfg = ContinuedPretrainConfig.from_dict(data.get("continued_pretrain", data))
    model_cfg = None
    if "model" in data:
        from .config import ModelConfig
        model_cfg = ModelConfig.from_dict(data["model"])
    return continued_pretrain(cfg, model_cfg)

