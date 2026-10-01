"""Corpus ingestion: text sources -> tokenized uint16 memmaps (nanoGPT-style)."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

from .tokenizer import PicoTokenizer


def iter_texts(
    inputs: list[str] | None = None,
    hf_dataset: str | None = None,
    hf_split: str = "train",
    text_field: str = "text",
    chat_jsonl: str | None = None,
    max_docs: int | None = None,
) -> Iterator[str]:
    """Yield one document at a time from local files, a HF dataset, or chat JSONL."""
    count = 0

    def _limit_reached() -> bool:
        return max_docs is not None and count >= max_docs

    if chat_jsonl:
        with Path(chat_jsonl).open(encoding="utf-8") as fh:
            for line in fh:
                if _limit_reached():
                    return
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                messages = record.get("messages", record) if isinstance(record, dict) else record
                yield _format_chat_text(messages)
                count += 1
        return

    if inputs:
        for pattern in inputs:
            p = Path(pattern)
            files = sorted(p.parent.glob(p.name)) if any(c in pattern for c in "*?[") else [p]
            for f in files:
                if _limit_reached():
                    return
                text = f.read_text(encoding="utf-8", errors="replace")
                # split large files on blank lines into paragraph-ish documents
                for doc in text.split("\n\n"):
                    doc = doc.strip()
                    if doc:
                        yield doc
                        count += 1
                        if _limit_reached():
                            return
        return

    if hf_dataset:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError("pip install 'pico-lm[data]' to stream HuggingFace datasets") from e
        ds = load_dataset(hf_dataset, split=hf_split, streaming=True)
        for row in ds:
            if _limit_reached():
                return
            text = row.get(text_field, "")
            if text:
                yield text
                count += 1
        return

    raise ValueError("provide --input files, --hf dataset, or --chat-jsonl")


def _format_chat_text(messages: list[dict[str, str]]) -> str:
    from .tokenizer import ASSISTANT, EOS, SYSTEM, USER

    tags = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT}
    return "".join(f"{tags[m['role']]}{m['content']}{EOS}" for m in messages)


def _val_stride(val_fraction: float) -> int:
    """Every Nth document goes to the validation split (0 disables the split).

    A random per-document coin flip leaves `val.bin` empty on small corpora
    (30 documents at 1% has a ~74% chance of producing no validation data at
    all), which only surfaces much later as an unreadable memmap at train time.
    A deterministic stride is reproducible, stratifies evenly over the corpus,
    and — together with always holding out the first document — guarantees a
    non-empty validation split for any corpus with at least two documents.
    """
    if not math.isfinite(val_fraction) or not 0 <= val_fraction <= 0.5:
        raise ValueError("val_fraction must be finite and in [0, 0.5]")
    if val_fraction == 0:
        return 0
    return max(2, round(1.0 / min(val_fraction, 0.5)))


def prepare(
    tokenizer: PicoTokenizer,
    out_dir: str | Path,
    val_fraction: float = 0.01,
    **source_kwargs,
) -> dict:
    """Tokenize the corpus into train.bin / val.bin uint16 memmaps + meta.json.

    Also drops a copy of the tokenizer next to the bins so that `train` and
    `generate` can always find it from the data directory alone.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if tokenizer.vocab_size > 65535:
        raise ValueError("uint16 token storage requires vocab_size <= 65535")

    stride = _val_stride(val_fraction)
    buffers: dict[str, list[int]] = {"train": [], "val": []}
    files = {
        "train": (out / "train.bin").open("wb"),
    }
    val_file = None
    if stride:
        val_file = (out / "val.bin").open("wb")
        files["val"] = val_file
    counts = {"train": 0, "val": 0}
    n_docs = 0
    try:
        for doc in iter_texts(**source_kwargs):
            ids = tokenizer.encode(doc, eos=True)
            # hold out the first document too, so even sub-stride corpora get a val set
            is_val = bool(stride) and (n_docs == 0 or n_docs % stride == stride - 1)
            split = "val" if is_val else "train"
            buffers[split].extend(ids)
            counts[split] += len(ids)
            n_docs += 1
            if len(buffers[split]) >= 1_000_000:
                files[split].write(np.asarray(buffers[split], dtype=np.uint16).tobytes())
                buffers[split] = []
            if n_docs % 20000 == 0:
                print(f"  {n_docs} docs, {counts['train'] + counts['val']:,} tokens")
        for split in ("train", "val"):
            if split in files and buffers[split]:
                files[split].write(np.asarray(buffers[split], dtype=np.uint16).tobytes())
    finally:
        for f in files.values():
            f.close()
        # If val was empty, remove the file
        if not stride:
            val_path = out / "val.bin"
            if val_path.exists():
                val_path.unlink()

    if n_docs == 0:
        raise ValueError("the corpus produced no documents; check --input / --hf / --chat-jsonl")
    if counts["train"] < 2:
        raise ValueError(
            f"corpus too small: {n_docs} document(s) left only {counts['train']} training "
            "tokens after the validation split. Add more text."
        )
    if stride and counts["val"] < 2:
        raise ValueError(
            f"corpus too small: {n_docs} document(s) produced no validation tokens at "
            f"val_fraction={val_fraction}. Add more text or raise --val-fraction."
        )

    # keep the tokenizer with the data so checkpoints stay self-describing
    tokenizer.save(out / "tokenizer.json")

    from .provenance import manifest

    meta = {
        "vocab_size": tokenizer.vocab_size,
        "n_docs": n_docs,
        "train_tokens": counts["train"],
        "val_tokens": counts["val"],
        "dtype": "uint16",
        "tokenizer": "tokenizer.json",
        "provenance": manifest(
            [out / "train.bin", out / "val.bin", out / "tokenizer.json"],
            source=source_kwargs,
            val_fraction=val_fraction,
        ),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


class TokenDataset:
    """Random contiguous windows over a token memmap."""

    def __init__(self, bin_path: str | Path) -> None:
        path = Path(bin_path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — run `pico-lm prepare` first")
        # np.memmap raises an opaque "cannot mmap an empty file" on a 0-byte bin
        if path.stat().st_size < 2 * np.dtype(np.uint16).itemsize:
            raise ValueError(
                f"{path} holds fewer than 2 tokens; re-run `pico-lm prepare` with a "
                "larger corpus or a different --val-fraction"
            )
        self.data = np.memmap(str(path), dtype=np.uint16, mode="r")

    def __len__(self) -> int:
        return len(self.data)

    def get_batch(
        self, batch_size: int, seq_len: int, device: str | torch.device = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = min(seq_len, len(self.data) - 1)
        ix = torch.randint(len(self.data) - seq_len, (batch_size,))
        x = torch.stack([torch.from_numpy(self.data[i : i + seq_len].astype(np.int64)) for i in ix])
        y = torch.stack(
            [torch.from_numpy(self.data[i + 1 : i + 1 + seq_len].astype(np.int64)) for i in ix]
        )
        if str(device).startswith("cuda"):
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
