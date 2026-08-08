"""Supervised fine-tuning on instruction/response pairs.

Three things make this a real SFT path rather than "continue pretraining on
chat-shaped text", which is what `prepare --chat-jsonl` + `train --resume` did:

1. **Example-aligned batching.** Pretraining draws random contiguous windows out
   of one long token stream, so a sample routinely starts mid-answer and ends
   mid-question. Here one row of the batch is one complete example.
2. **Loss masked to the response.** Prompt positions get target ``-1``, which
   :meth:`Transformer.forward` already treats as ``ignore_index``. Training on
   the prompt teaches the model to *write* instructions, not follow them.
3. **Its own schedule and its own optimizer state.** ``--init-from`` loads the
   pretrained weights and starts a fresh low-LR cosine schedule at step 0.
   ``--resume`` is a different operation and continues an interrupted SFT run.

Checkpoint selection is behavioural by default (``select_by="score"``): every
eval, held-out prompts are greedy-decoded and exact-matched, and ``best.pt``
tracks *that*, not the loss. A fine-tune can reach an excellent masked loss
while producing unusable generations; selecting on loss ships that model.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .config import ModelConfig, SFTConfig
from .model import Transformer
from .tokenizer import PicoTokenizer
from .train import autocast_ctx, cosine_lr, load_model, pick_device, pick_dtype

IGNORE_INDEX = -1


# --------------------------------------------------------------------------- #
# dataset construction
# --------------------------------------------------------------------------- #
def read_pairs(path: str | Path) -> list[dict]:
    """Read JSONL of {"messages": [...]} or {"instruction", "response"} records."""
    records = []
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "messages" in rec:
                msgs = rec["messages"]
                if not any(m["role"] == "assistant" for m in msgs):
                    raise ValueError(f"{path}:{lineno} has no assistant message")
                # everything up to the final assistant turn is the prompt
                last = max(i for i, m in enumerate(msgs) if m["role"] == "assistant")
                prompt_msgs, response = msgs[:last], msgs[last]["content"]
            elif "instruction" in rec and "response" in rec:
                prompt_msgs = [{"role": "user", "content": rec["instruction"]}]
                if rec.get("system"):
                    prompt_msgs.insert(0, {"role": "system", "content": rec["system"]})
                response = rec["response"]
            else:
                raise ValueError(f"{path}:{lineno} needs 'messages' or 'instruction'+'response'")
            records.append(
                {"messages": prompt_msgs, "response": response, "task": rec.get("task", "all")}
            )
    return records


def encode_pair(rec: dict, tok: PicoTokenizer, max_len: int) -> tuple[list[int], int] | None:
    """-> (token ids, prompt length). ``None`` if no response survives truncation."""
    prompt = tok.format_chat(rec["messages"], add_generation_prompt=True)
    prompt_ids = tok.encode(prompt, bos=True)
    response_ids = [*tok.encode(rec["response"]), tok.eos_id]
    if len(response_ids) >= max_len:
        response_ids = response_ids[: max_len - 1]
        response_ids[-1] = tok.eos_id
    # left-truncate the prompt: the response and the generation cue must survive
    budget = max_len - len(response_ids)
    if len(prompt_ids) > budget:
        prompt_ids = prompt_ids[-budget:]
    if not response_ids or budget <= 0:
        return None
    return prompt_ids + response_ids, len(prompt_ids)


def prepare_sft(
    jsonl: str | Path,
    tok: PicoTokenizer,
    out_dir: str | Path,
    val_fraction: float = 0.05,
    max_len: int = 512,
    seed: int = 0,
) -> dict:
    """Tokenize instruction pairs into ``sft_train.npz`` / ``sft_val.npz``.

    Unlike `prepare`, examples keep their boundaries: each is stored with the
    offset where its response begins, which is what the loss mask needs.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = read_pairs(jsonl)
    encoded = []
    for rec in records:
        item = encode_pair(rec, tok, max_len)
        if item is not None:
            encoded.append((item[0], item[1], rec["task"]))
    if len(encoded) < 4:
        raise ValueError(f"{jsonl} produced only {len(encoded)} usable examples")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(encoded))
    n_val = max(1, int(len(encoded) * val_fraction))
    splits = {"val": order[:n_val], "train": order[n_val:]}

    meta = {"n_examples": len(encoded), "max_len": max_len, "vocab_size": tok.vocab_size}
    for name, idx in splits.items():
        tokens, offsets, prompt_lens, tasks = [], [0], [], []
        for i in idx:
            ids, plen, task = encoded[i]
            tokens.extend(ids)
            offsets.append(len(tokens))
            prompt_lens.append(plen)
            tasks.append(task)
        np.savez(
            out / f"sft_{name}.npz",
            tokens=np.asarray(tokens, dtype=np.uint16),
            offsets=np.asarray(offsets, dtype=np.int64),
            prompt_lens=np.asarray(prompt_lens, dtype=np.int32),
        )
        meta[f"{name}_examples"] = len(idx)
        meta[f"{name}_tokens"] = len(tokens)
        meta[f"{name}_response_tokens"] = int(
            sum(offsets[j + 1] - offsets[j] - prompt_lens[j] for j in range(len(idx)))
        )
    tok.save(out / "tokenizer.json")
    meta["tokenizer"] = "tokenizer.json"
    (out / "sft_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


class SFTDataset:
    """Batches of complete examples with the prompt masked out of the loss."""

    def __init__(self, npz_path: str | Path, pad_id: int = 0) -> None:
        path = Path(npz_path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — run `pico-lm prepare-sft` first")
        with np.load(path) as z:
            self.tokens = z["tokens"]
            self.offsets = z["offsets"]
            self.prompt_lens = z["prompt_lens"]
        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.prompt_lens)

    def example(self, i: int) -> tuple[np.ndarray, int]:
        return self.tokens[self.offsets[i] : self.offsets[i + 1]], int(self.prompt_lens[i])

    def get_batch(
        self, batch_size: int, device: str | torch.device = "cpu", generator=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = len(self)
        idx = torch.randint(n, (min(batch_size, n),), generator=generator).tolist()
        return self.collate(idx, device)

    def collate(self, idx: list[int], device: str | torch.device = "cpu"):
        items = [self.example(i) for i in idx]
        width = max(len(ids) for ids, _ in items)
        x = torch.full((len(items), width - 1), self.pad_id, dtype=torch.long)
        y = torch.full((len(items), width - 1), IGNORE_INDEX, dtype=torch.long)
        for b, (ids, plen) in enumerate(items):
            t = torch.from_numpy(ids.astype(np.int64))
            n_tok = len(t)
            x[b, : n_tok - 1] = t[:-1]
            y[b, : n_tok - 1] = t[1:]
            # position i predicts token i+1, so the prompt occupies i < plen - 1
            y[b, : max(0, plen - 1)] = IGNORE_INDEX
        # right padding + causal attention: padded positions can never influence
        # a real one, and their targets are already IGNORE_INDEX
        return x.to(device), y.to(device)

    def iter_batches(self, batch_size: int):
        for start in range(0, len(self), batch_size):
            yield list(range(start, min(start + batch_size, len(self))))


# --------------------------------------------------------------------------- #
# fine-tuning loop
# --------------------------------------------------------------------------- #
@torch.no_grad()
def masked_val_loss(
    model: Transformer, ds: SFTDataset, cfg: SFTConfig, device: str, dtype: torch.dtype
) -> float:
    model.eval()
    gen = torch.Generator().manual_seed(0)
    losses = []
    for _ in range(cfg.eval_iters):
        x, y = ds.get_batch(cfg.batch_size, device, generator=gen)
        with autocast_ctx(device, dtype):
            _, loss = model(x, y)
        if not math.isnan(loss.item()):
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def _bench_examples(jsonl: str | Path | None, limit: int) -> list[dict]:
    if not jsonl or not Path(jsonl).exists():
        return []
    from .bench import load_instruction_examples

    return load_instruction_examples(jsonl)[:limit]


def save_sft_checkpoint(
    path: Path,
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    model_cfg: ModelConfig,
    sft_cfg: SFTConfig,
    step: int,
    best: dict,
    tokenizer_path: str | None,
    init_from: str | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_cfg": model_cfg.to_dict(),
            "sft_cfg": sft_cfg.to_dict(),
            "step": step,
            "best_val": best.get("loss", float("inf")),
            "best_score": best.get("score", 0.0),
            "tokenizer_path": tokenizer_path,
            "init_from": init_from,
            "stage": "sft",
        },
        path,
    )


def finetune(
    sft_dir: str | Path,
    out_dir: str | Path,
    cfg: SFTConfig,
    init_from: str | None = None,
    resume: bool = False,
    tokenizer_path: str | None = None,
    bench_jsonl: str | Path | None = None,
) -> dict:
    """Fine-tune a pretrained checkpoint on instruction pairs.

    Returns the best metrics seen. ``bench_jsonl`` supplies held-out prompts for
    the behavioural score; without it selection silently falls back to loss, and
    a warning says so — because selecting a fine-tune on loss alone is exactly
    how a 0.05-loss model that answers nothing correctly gets shipped.
    """
    device = pick_device()
    dtype = pick_dtype(device)
    torch.manual_seed(cfg.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    sft_dir, out_dir = Path(sft_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_tok = Path(tokenizer_path) if tokenizer_path else sft_dir / "tokenizer.json"
    if src_tok.exists() and src_tok.resolve() != (out_dir / "tokenizer.json").resolve():
        shutil.copyfile(src_tok, out_dir / "tokenizer.json")
    tok = PicoTokenizer.load(out_dir / "tokenizer.json")

    train_ds = SFTDataset(sft_dir / "sft_train.npz", pad_id=tok.pad_id)
    val_ds = SFTDataset(sft_dir / "sft_val.npz", pad_id=tok.pad_id)

    step = 0
    best = {"loss": float("inf"), "score": -1.0}
    last_path = out_dir / "last.pt"
    if resume and last_path.exists():
        model, ckpt = load_model(last_path, device)
        optimizer = _make_optimizer(model, cfg, device)
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        best = {"loss": ckpt.get("best_val", float("inf")), "score": ckpt.get("best_score", -1.0)}
        init_from = ckpt.get("init_from", init_from)
        print(f"resumed SFT from step {step}")
    elif init_from:
        model, base_ckpt = load_model(init_from, device)
        optimizer = _make_optimizer(model, cfg, device)
        print(
            f"initialised from {init_from} "
            f"({model.num_params() / 1e6:.2f}M params, pretrain step {base_ckpt.get('step')})"
        )
    else:
        raise SystemExit("SFT needs pretrained weights: pass --init-from <checkpoint>")
    model_cfg = model.cfg
    model.train()

    bench_set = _bench_examples(bench_jsonl, cfg.bench_examples)
    select_by = cfg.select_by
    if select_by == "score" and not bench_set:
        print(
            "warning: --select-by score but no --bench-jsonl held-out prompts were given; "
            "falling back to loss-based selection (this is the failure mode this "
            "pipeline exists to avoid)"
        )
        select_by = "loss"

    if cfg.compile:
        model = torch.compile(model)

    print(
        f"device={device} dtype={dtype} train={len(train_ds)} val={len(val_ds)} "
        f"lr={cfg.lr:g} steps={cfg.max_steps} select_by={select_by}"
    )

    log_path = out_dir / "sft_log.csv"
    if not log_path.exists():
        log_path.write_text("step,train_loss,val_loss,score,lr\n", encoding="utf-8")

    scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16))
    t0, running, n_logged = time.time(), 0.0, 0
    history: list[dict] = []

    while step < cfg.max_steps:
        lr = cosine_lr(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum = 0.0
        for _ in range(cfg.grad_accum_steps):
            x, y = train_ds.get_batch(cfg.batch_size, device)
            with autocast_ctx(device, dtype):
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum_steps
            scaler.scale(loss).backward()
            accum += loss.item()
        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        step += 1
        running += accum
        n_logged += 1

        if step % cfg.log_interval == 0:
            dt = time.time() - t0
            print(
                f"step {step:5d} | loss {running / n_logged:.4f} | lr {lr:.2e} | "
                f"{cfg.log_interval / dt:.1f} steps/s"
            )
            running, n_logged, t0 = 0.0, 0, time.time()

        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            raw = getattr(model, "_orig_mod", model)
            vloss = masked_val_loss(raw, val_ds, cfg, device, dtype)
            score = float("nan")
            if bench_set:
                from .bench import score_instructions

                score = score_instructions(raw, tok, bench_set, device, n_samples=0)["exact_match"]
                raw.train()
            msg = f"eval @ {step}: val loss {vloss:.4f}"
            if bench_set:
                msg += f" | held-out exact match {score:.1%}"
            print(msg)
            history.append({"step": step, "val_loss": vloss, "score": score})
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"{step},{accum:.6f},{vloss:.6f},{score:.6f},{lr:.6e}\n")

            improved = score > best["score"] if select_by == "score" else vloss < best["loss"]
            if vloss < best["loss"]:
                best["loss"] = vloss
            if bench_set and score > best["score"]:
                best["score"] = score
            save_sft_checkpoint(
                last_path,
                raw,
                optimizer,
                model_cfg,
                cfg,
                step,
                best,
                str((out_dir / "tokenizer.json").resolve()),
                init_from,
            )
            if improved:
                save_sft_checkpoint(
                    out_dir / "best.pt",
                    raw,
                    optimizer,
                    model_cfg,
                    cfg,
                    step,
                    best,
                    str((out_dir / "tokenizer.json").resolve()),
                    init_from,
                )
                metric = f"{best['score']:.1%} EM" if select_by == "score" else f"{vloss:.4f} loss"
                print(f"  new best ({metric}) -> best.pt")
            t0 = time.time()

    (out_dir / "sft_history.json").write_text(
        json.dumps({"history": history, "best": best, "select_by": select_by}, indent=2),
        encoding="utf-8",
    )
    return best


def _make_optimizer(model: Transformer, cfg: SFTConfig, device: str) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (decay if p.dim() >= 2 else no_decay).append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        fused=(device == "cuda"),
    )
