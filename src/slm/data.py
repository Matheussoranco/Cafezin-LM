"""Corpus ingestion: text sources -> tokenized uint16 memmaps (nanoGPT-style)."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from .tokenizer import SLMTokenizer


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
                messages = record["messages"] if "messages" in record else record
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
            raise ImportError("pip install 'slm[data]' to stream HuggingFace datasets") from e
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
    from .tokenizer import SYSTEM, USER, ASSISTANT, EOS

    tags = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT}
    return "".join(f"{tags[m['role']]}{m['content']}{EOS}" for m in messages)


def prepare(
    tokenizer: SLMTokenizer,
    out_dir: str | Path,
    val_fraction: float = 0.01,
    seed: int = 1337,
    **source_kwargs,
) -> dict:
    """Tokenize the corpus into train.bin / val.bin uint16 memmaps + meta.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if tokenizer.vocab_size > 65535:
        raise ValueError("uint16 token storage requires vocab_size <= 65535")

    rng = random.Random(seed)
    buffers: dict[str, list[int]] = {"train": [], "val": []}
    files = {
        "train": (out / "train.bin").open("wb"),
        "val": (out / "val.bin").open("wb"),
    }
    counts = {"train": 0, "val": 0}
    n_docs = 0
    try:
        for doc in iter_texts(**source_kwargs):
            ids = tokenizer.encode(doc, eos=True)
            split = "val" if rng.random() < val_fraction else "train"
            buffers[split].extend(ids)
            counts[split] += len(ids)
            n_docs += 1
            if len(buffers[split]) >= 1_000_000:
                files[split].write(np.asarray(buffers[split], dtype=np.uint16).tobytes())
                buffers[split] = []
            if n_docs % 20000 == 0:
                print(f"  {n_docs} docs, {counts['train'] + counts['val']:,} tokens")
        for split in ("train", "val"):
            if buffers[split]:
                files[split].write(np.asarray(buffers[split], dtype=np.uint16).tobytes())
    finally:
        for f in files.values():
            f.close()

    meta = {
        "vocab_size": tokenizer.vocab_size,
        "n_docs": n_docs,
        "train_tokens": counts["train"],
        "val_tokens": counts["val"],
        "dtype": "uint16",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


class TokenDataset:
    """Random contiguous windows over a token memmap."""

    def __init__(self, bin_path: str | Path) -> None:
        self.data = np.memmap(str(bin_path), dtype=np.uint16, mode="r")
        if len(self.data) < 2:
            raise ValueError(f"{bin_path} is empty")

    def __len__(self) -> int:
        return len(self.data)

    def get_batch(
        self, batch_size: int, seq_len: int, device: str | torch.device = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = min(seq_len, len(self.data) - 1)
        ix = torch.randint(len(self.data) - seq_len, (batch_size,))
        x = torch.stack([torch.from_numpy(self.data[i : i + seq_len].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1 : i + 1 + seq_len].astype(np.int64)) for i in ix])
        if str(device).startswith("cuda"):
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
