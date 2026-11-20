"""Tokenizer utilities for corpus preparation and token counting."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cafezin_lm.tokenizer import Tokenizer


def _utcnow_iso() -> str:
    """Return current UTC time as ISO format string with Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class TokenizationReport:
    """Report of tokenization statistics for a corpus."""

    corpus_name: str
    tokenizer_path: str
    tokenizer_vocab_size: int
    created_at: str = field(default_factory=_utcnow_iso)
    num_documents: int = 0
    total_chars: int = 0
    total_tokens: int = 0
    chars_per_token: float = 0.0
    tokens_per_char: float = 0.0
    avg_doc_tokens: float = 0.0
    min_doc_tokens: int = 0
    max_doc_tokens: int = 0
    # Curriculum tracking
    pt_tokens: int = 0
    multilingual_tokens: int = 0
    pt_ratio: float = 0.0
    curriculum_stage: str = ""
    # Per-document stats (optional, for detailed analysis)
    doc_token_counts: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dictionary (excluding per-doc counts for brevity)."""
        return {
            "corpus_name": self.corpus_name,
            "tokenizer_path": self.tokenizer_path,
            "tokenizer_vocab_size": self.tokenizer_vocab_size,
            "created_at": self.created_at,
            "num_documents": self.num_documents,
            "total_chars": self.total_chars,
            "total_tokens": self.total_tokens,
            "chars_per_token": self.chars_per_token,
            "tokens_per_char": self.tokens_per_char,
            "avg_doc_tokens": self.avg_doc_tokens,
            "min_doc_tokens": self.min_doc_tokens,
            "max_doc_tokens": self.max_doc_tokens,
            "pt_tokens": self.pt_tokens,
            "multilingual_tokens": self.multilingual_tokens,
            "pt_ratio": self.pt_ratio,
            "curriculum_stage": self.curriculum_stage,
        }

    def save(self, path: str | Path) -> None:
        """Save report to JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> TokenizationReport:
        """Load report from JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


def tokenize_corpus(
    documents: list[str],
    tokenizer: "Tokenizer",
    corpus_name: str = "corpus",
    eos: bool = True,
) -> tuple[list[list[int]], TokenizationReport]:
    """Tokenize a corpus and generate statistics.

    Args:
        documents: List of document texts.
        tokenizer: Tokenizer to use.
        corpus_name: Name for the report.
        eos: Whether to append EOS token.

    Returns:
        Tuple of (tokenized documents as list of token ID lists, report).
    """
    tokenized = []
    doc_token_counts = []
    total_chars = 0
    total_tokens = 0

    for doc in documents:
        total_chars += len(doc.encode("utf-8"))
        ids = tokenizer.encode(doc, eos=eos)
        tokenized.append(ids)
        count = len(ids)
        doc_token_counts.append(count)
        total_tokens += count

    num_docs = len(documents)
    report = TokenizationReport(
        corpus_name=corpus_name,
        tokenizer_path=str(tokenizer.path),
        tokenizer_vocab_size=tokenizer.vocab_size,
        num_documents=num_docs,
        total_chars=total_chars,
        total_tokens=total_tokens,
        chars_per_token=total_chars / max(1, total_tokens),
        tokens_per_char=total_tokens / max(1, total_chars),
        avg_doc_tokens=total_tokens / max(1, num_docs),
        min_doc_tokens=min(doc_token_counts) if doc_token_counts else 0,
        max_doc_tokens=max(doc_token_counts) if doc_token_counts else 0,
        doc_token_counts=doc_token_counts,
    )

    return tokenized, report


def record_token_counts(
    pt_documents: list[str],
    multilingual_documents: list[str],
    tokenizer: "Tokenizer",
    schedule: "CurriculumSchedule | None" = None,
    tokens_seen: int = 0,
) -> tuple[TokenizationReport, TokenizationReport, TokenizationReport]:
    """Tokenize PT and multilingual corpora separately and record counts.

    Args:
        pt_documents: Portuguese documents.
        multilingual_documents: Multilingual documents.
        tokenizer: Tokenizer to use.
        schedule: Optional curriculum schedule for stage tracking.
        tokens_seen: Tokens already trained on (for curriculum stage).

    Returns:
        Tuple of (pt_report, multilingual_report, combined_report).
    """
    from .curriculum import CurriculumStage

    # Tokenize PT corpus
    pt_tokenized, pt_report = tokenize_corpus(pt_documents, tokenizer, "portuguese")
    pt_report.pt_tokens = pt_report.total_tokens
    pt_report.pt_ratio = 1.0

    # Tokenize multilingual corpus
    multi_tokenized, multi_report = tokenize_corpus(multilingual_documents, tokenizer, "multilingual")
    multi_report.multilingual_tokens = multi_report.total_tokens
    multi_report.pt_ratio = 0.0

    # Combined report
    _ = pt_tokenized + multi_tokenized
    total_tokens = pt_report.total_tokens + multi_report.total_tokens
    total_chars = pt_report.total_chars + multi_report.total_chars
    total_docs = pt_report.num_documents + multi_report.num_documents

    pt_ratio = pt_report.total_tokens / max(1, total_tokens)
    curriculum_stage = ""
    if schedule:
        curriculum_stage = schedule.stage_at(tokens_seen).value

    combined_report = TokenizationReport(
        corpus_name="combined",
        tokenizer_path=str(tokenizer.path),
        tokenizer_vocab_size=tokenizer.vocab_size,
        num_documents=total_docs,
        total_chars=total_chars,
        total_tokens=total_tokens,
        chars_per_token=total_chars / max(1, total_tokens),
        tokens_per_char=total_tokens / max(1, total_chars),
        avg_doc_tokens=total_tokens / max(1, total_docs),
        min_doc_tokens=min(pt_report.min_doc_tokens, multi_report.min_doc_tokens) if pt_report.min_doc_tokens and multi_report.min_doc_tokens else 0,
        max_doc_tokens=max(pt_report.max_doc_tokens, multi_report.max_doc_tokens),
        pt_tokens=pt_report.total_tokens,
        multilingual_tokens=multi_report.total_tokens,
        pt_ratio=pt_ratio,
        curriculum_stage=curriculum_stage,
    )

    return pt_report, multi_report, combined_report


def prepare_curriculum_data(
    pt_train_docs: list[str],
    pt_eval_docs: list[str],
    multi_train_docs: list[str],
    multi_eval_docs: list[str],
    tokenizer: "Tokenizer",
    schedule: "CurriculumSchedule",
    output_dir: str | Path,
    curriculum_name: str = "portuguese-curriculum",
) -> dict:
    """Prepare full curriculum data package with train/eval splits and tokenization.

    Creates:
    - train.bin / val.bin (tokenized memmaps)
    - meta.json (data metadata)
    - provenance.json (corpus provenance)
    - curriculum_schedule.json (schedule config)
    - tokenization_reports/ (PT, multilingual, combined reports)

    Args:
        pt_train_docs: Portuguese training documents.
        pt_eval_docs: Portuguese eval documents.
        multi_train_docs: Multilingual training documents.
        multi_eval_docs: Multilingual eval documents.
        tokenizer: Tokenizer to use.
        schedule: Curriculum schedule.
        output_dir: Output directory.
        curriculum_name: Name for the curriculum package.

    Returns:
        Dictionary with paths and summary statistics.
    """
    import numpy as np
    from cafezin_lm.data import dtype_for_vocab, TOKEN_DTYPES

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reports_dir = output_dir / "tokenization_reports"
    reports_dir.mkdir(exist_ok=True)

    # Tokenize all splits
    pt_train_tokens, pt_train_report = tokenize_corpus(pt_train_docs, tokenizer, "pt_train")
    pt_eval_tokens, pt_eval_report = tokenize_corpus(pt_eval_docs, tokenizer, "pt_eval")
    multi_train_tokens, multi_train_report = tokenize_corpus(multi_train_docs, tokenizer, "multi_train")
    multi_eval_tokens, multi_eval_report = tokenize_corpus(multi_eval_docs, tokenizer, "multi_eval")

    # Save tokenization reports
    pt_train_report.save(reports_dir / "pt_train_report.json")
    pt_eval_report.save(reports_dir / "pt_eval_report.json")
    multi_train_report.save(reports_dir / "multi_train_report.json")
    multi_eval_report.save(reports_dir / "multi_eval_report.json")

    # Combine train tokens (curriculum will be applied during training)
    # For now, we save them separately so the training loop can apply curriculum
    train_tokens = {"pt": pt_train_tokens, "multilingual": multi_train_tokens}
    val_tokens = {"pt": pt_eval_tokens, "multilingual": multi_eval_tokens}

    # Save as separate memmaps for curriculum-aware training
    dtype = dtype_for_vocab(tokenizer.max_token_id + 1)
    np_dtype = TOKEN_DTYPES[dtype]

    for split_name, split_data in [("train", train_tokens), ("val", val_tokens)]:
        for lang_name, docs_tokens in split_data.items():
            if not docs_tokens:
                continue
            # Flatten
            flat = np.concatenate([np.array(t, dtype=np_dtype) for t in docs_tokens])
            out_path = output_dir / f"{split_name}_{lang_name}.bin"
            flat.tofile(out_path)
            print(f"Saved {len(flat):,} tokens to {out_path}")

    # Save meta.json
    meta = {
        "vocab_size": tokenizer.vocab_size,
        "max_token_id": tokenizer.max_token_id,
        "dtype": dtype,
        "eos_token_id": tokenizer.eos_token_id,
        "tokenizer": str(tokenizer.path),
        "curriculum": schedule.to_dict(),
        "splits": {
            "train": {
                "pt_tokens": pt_train_report.total_tokens,
                "pt_docs": pt_train_report.num_documents,
                "multilingual_tokens": multi_train_report.total_tokens,
                "multilingual_docs": multi_train_report.num_documents,
            },
            "val": {
                "pt_tokens": pt_eval_report.total_tokens,
                "pt_docs": pt_eval_report.num_documents,
                "multilingual_tokens": multi_eval_report.total_tokens,
                "multilingual_docs": multi_eval_report.num_documents,
            },
        },
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Save curriculum schedule
    (output_dir / "curriculum_schedule.json").write_text(
        json.dumps(schedule.to_dict(), indent=2), encoding="utf-8"
    )

    return {
        "output_dir": str(output_dir),
        "meta": meta,
        "pt_train_report": pt_train_report.to_dict(),
        "pt_eval_report": pt_eval_report.to_dict(),
        "multi_train_report": multi_train_report.to_dict(),
        "multi_eval_report": multi_eval_report.to_dict(),
    }
