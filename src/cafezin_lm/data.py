"""Corpus ingestion: text sources -> tokenized memmaps.

The vocabulary is 248,320 tokens, so the token stream is stored as **uint32**
rather than the uint16 a small model would use — a uint16 stream would wrap
silently. ``dtype_for_vocab`` picks the width, and ``prepare`` refuses a dtype
that cannot hold the vocabulary.

Sources are local files, a Hugging Face dataset, or chat JSONL; all three feed
the same deterministic train/validation split.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

from .tokenizer import Tokenizer

TOKEN_DTYPES = {"uint32": np.uint32, "uint16": np.uint16, "uint64": np.uint64}


def dtype_for_vocab(vocab_size: int) -> str:
    if vocab_size <= 2**16:
        return "uint16"
    if vocab_size <= 2**32:
        return "uint32"
    return "uint64"


def _format_chat(messages: list[dict[str, str]]) -> str:
    """Flatten chat messages to plain text with role markers."""
    parts = []
    for m in messages:
        role = m.get("role", "user")
        parts.append(f"<|{role}|>{m.get('content', '')}")
    return "\n".join(parts)


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
                yield _format_chat(messages)
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
            raise ImportError(
                "pip install 'cafezin-lm[data]' to stream HuggingFace datasets"
            ) from e
        ds = load_dataset(hf_dataset, split=hf_split, streaming=True)
        for row in ds:
            if _limit_reached():
                return
            text = row.get(text_field, "")
            if text:
                yield text
                count += 1
        return

    raise ValueError("provide --input files, --hf-dataset, or --chat-jsonl")


def _val_stride(val_fraction: float) -> int:
    """Every Nth document goes to the validation split (0 disables the split).

    A random per-document coin flip leaves ``val.bin`` empty on small corpora
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
    tokenizer: Tokenizer,
    out_dir: str | Path,
    val_fraction: float = 0.01,
    dtype: str | None = None,
    **source_kwargs,
) -> dict:
    """Tokenize a corpus into train/val memmaps plus meta.json.

    The tokenizer's file is copied next to the bins so checkpoints stay
    self-contained.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dtype = dtype or dtype_for_vocab(tokenizer.max_token_id + 1)
    if dtype not in TOKEN_DTYPES:
        raise ValueError(f"unknown token dtype {dtype!r}")
    np_dtype = TOKEN_DTYPES[dtype]
    if tokenizer.max_token_id > np.iinfo(np_dtype).max:
        raise ValueError(f"token ID {tokenizer.max_token_id} does not fit in {dtype}")

    stride = _val_stride(val_fraction)
    buffers: dict[str, list[int]] = {"train": [], "val": []}
    handles: dict[str, object] = {"train": (out / "train.bin").open("wb")}
    if stride:
        handles["val"] = (out / "val.bin").open("wb")
    counts = {"train": 0, "val": 0}
    docs = 0
    try:
        for text in iter_texts(**source_kwargs):
            ids = tokenizer.encode(text, eos=True)
            split = "val" if stride and (docs == 0 or docs % stride == stride - 1) else "train"
            buffers[split].extend(ids)
            counts[split] += len(ids)
            docs += 1
            if len(buffers[split]) >= 4_000_000:
                handles[split].write(np.asarray(buffers[split], dtype=np_dtype).tobytes())
                buffers[split] = []
            if docs % 20000 == 0:
                print(f"  {docs} docs, {counts['train'] + counts['val']:,} tokens")
        for split, handle in handles.items():
            if buffers[split]:
                handle.write(np.asarray(buffers[split], dtype=np_dtype).tobytes())
    finally:
        for handle in handles.values():
            handle.close()

    if docs == 0:
        raise ValueError("the corpus produced no documents")
    if counts["train"] < 2:
        raise ValueError(f"corpus too small: only {counts['train']} training tokens")
    if stride and counts["val"] < 2:
        raise ValueError("corpus too small: the validation split is empty")

    tokenizer_dest = out / "tokenizer.json"
    tokenizer.save(tokenizer_dest)

    meta = {
        "vocab_size": tokenizer.vocab_size,
        "max_token_id": tokenizer.max_token_id,
        "n_docs": docs,
        "train_tokens": counts["train"],
        "val_tokens": counts["val"],
        "dtype": dtype,
        "eos_token_id": tokenizer.eos_token_id,
        "tokenizer": str(tokenizer_dest),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


class TokenDataset:
    """Random contiguous windows over a token memmap."""

    def __init__(self, bin_path: str | Path, dtype: str = "uint32") -> None:
        path = Path(bin_path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — run `cafezin-lm prepare` first")
        if dtype not in TOKEN_DTYPES:
            raise ValueError(f"unknown token dtype {dtype!r}")
        self.dtype = dtype
        item = np.dtype(TOKEN_DTYPES[dtype]).itemsize
        if path.stat().st_size < 2 * item:
            raise ValueError(
                f"{path} holds fewer than 2 tokens; re-run `cafezin-lm prepare` with a "
                "larger corpus or a different --val-fraction"
            )
        self.data = np.memmap(str(path), dtype=TOKEN_DTYPES[dtype], mode="r")

    def __len__(self) -> int:
        return len(self.data)

    def get_batch(
        self, batch_size: int, seq_len: int, device: str | torch.device = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = min(seq_len, len(self.data) - 1)
        starts = torch.randint(len(self.data) - seq_len, (batch_size,))
        x = torch.stack(
            [torch.from_numpy(self.data[i : i + seq_len].astype(np.int64)) for i in starts]
        )
        y = torch.stack(
            [torch.from_numpy(self.data[i + 1 : i + 1 + seq_len].astype(np.int64)) for i in starts]
        )
        if str(device).startswith("cuda"):
            return (
                x.pin_memory().to(device, non_blocking=True),
                y.pin_memory().to(device, non_blocking=True),
            )
        return x.to(device), y.to(device)


def load_meta(data_dir: str | Path) -> dict:
    meta_path = Path(data_dir) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found — run `cafezin-lm prepare --out {data_dir}` first"
        )
    return json.loads(meta_path.read_text(encoding="utf-8"))
