"""Portuguese corpus assembly with documented provenance."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    """Return current UTC time as ISO format string with Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class CorpusSource:
    """Specification for a corpus source."""

    name: str
    """Human-readable name (e.g., 'mc4-pt', 'oscar-pt', 'br-common-crawl', 'pt-wikipedia')."""

    hf_dataset_id: str
    """Hugging Face dataset identifier (e.g., 'mc4', 'oscar-corpus/OSCAR-2301')."""

    hf_config: str | None = None
    """Dataset config name (e.g., 'pt' for mc4, 'pt' for OSCAR)."""

    hf_split: str = "train"
    """Dataset split to use."""

    text_field: str = "text"
    """Field containing the text content."""

    language_filter: str | None = None
    """Optional language code to filter (e.g., 'pt' for Portuguese)."""

    revision: str | None = None
    """Dataset revision/commit hash for reproducibility."""

    # Brazilian Portuguese selection criteria
    br_pt_criteria: dict | None = None
    """Criteria for Brazilian Portuguese selection:
    - For Common Crawl: {'country_code': 'BR', 'language_score_threshold': 0.9}
    - For Wikipedia: {'language': 'pt', 'variant': 'br'}"""

    max_docs: int | None = None
    """Maximum documents to load (for testing/fixtures)."""

    trust_remote_code: bool = False
    """Whether to trust remote code when loading."""

    def provenance_dict(self) -> dict:
        """Return dict suitable for provenance recording."""
        return {
            "name": self.name,
            "hf_dataset_id": self.hf_dataset_id,
            "hf_config": self.hf_config,
            "hf_split": self.hf_split,
            "text_field": self.text_field,
            "language_filter": self.language_filter,
            "revision": self.revision,
            "br_pt_criteria": self.br_pt_criteria,
            "max_docs": self.max_docs,
            "trust_remote_code": self.trust_remote_code,
        }


@dataclass
class CorpusProvenance:
    """Complete provenance record for a curated corpus."""

    corpus_name: str
    created_at: str = field(default_factory=_utcnow_iso)
    sources: list[CorpusSource] = field(default_factory=list)
    total_docs_before_filter: int = 0
    total_docs_after_filter: int = 0
    total_tokens_before_dedup: int = 0
    total_tokens_after_dedup: int = 0
    filtering_steps: list[dict] = field(default_factory=list)
    deduplication_method: str | None = None
    eval_split_config: dict | None = None
    tokenizer_info: dict | None = None
    curriculum_schedule: dict | None = None
    notes: list[str] = field(default_factory=list)

    def add_source(self, source: CorpusSource) -> None:
        """Add a source to the provenance."""
        self.sources.append(source)

    def add_filtering_step(self, step_name: str, docs_before: int, docs_after: int, params: dict | None = None) -> None:
        """Record a filtering step."""
        self.filtering_steps.append({
            "step": step_name,
            "docs_before": docs_before,
            "docs_after": docs_after,
            "docs_removed": docs_before - docs_after,
            "removal_rate": (docs_before - docs_after) / max(1, docs_before),
            "params": params or {},
            "timestamp": _utcnow_iso(),
        })

    def add_note(self, note: str) -> None:
        """Add a provenance note."""
        self.notes.append(f"[{_utcnow_iso()}] {note}")

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "corpus_name": self.corpus_name,
            "created_at": self.created_at,
            "sources": [s.provenance_dict() for s in self.sources],
            "total_docs_before_filter": self.total_docs_before_filter,
            "total_docs_after_filter": self.total_docs_after_filter,
            "total_tokens_before_dedup": self.total_tokens_before_dedup,
            "total_tokens_after_dedup": self.total_tokens_after_dedup,
            "filtering_steps": self.filtering_steps,
            "deduplication_method": self.deduplication_method,
            "eval_split_config": self.eval_split_config,
            "tokenizer_info": self.tokenizer_info,
            "curriculum_schedule": self.curriculum_schedule,
            "notes": self.notes,
        }

    def save(self, path: str | Path) -> None:
        """Save provenance to JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CorpusProvenance:
        """Load provenance from JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        prov = cls(corpus_name=data["corpus_name"])
        prov.created_at = data["created_at"]
        prov.sources = [CorpusSource(**s) for s in data["sources"]]
        prov.total_docs_before_filter = data["total_docs_before_filter"]
        prov.total_docs_after_filter = data["total_docs_after_filter"]
        prov.total_tokens_before_dedup = data["total_tokens_before_dedup"]
        prov.total_tokens_after_dedup = data["total_tokens_after_dedup"]
        prov.filtering_steps = data["filtering_steps"]
        prov.deduplication_method = data.get("deduplication_method")
        prov.eval_split_config = data.get("eval_split_config")
        prov.tokenizer_info = data.get("tokenizer_info")
        prov.curriculum_schedule = data.get("curriculum_schedule")
        prov.notes = data.get("notes", [])
        return prov


def default_portuguese_sources(max_docs: int | None = None) -> list[CorpusSource]:
    """Get default Portuguese corpus sources with documented revisions.

    These revisions are pinned for reproducibility. Update with actual
    revision hashes when datasets are frozen.
    """
    return [
        CorpusSource(
            name="mc4-pt",
            hf_dataset_id="mc4",
            hf_config="pt",
            hf_split="train",
            text_field="text",
            language_filter="pt",
            revision="main",  # TODO: pin to specific commit when available
            max_docs=max_docs,
            trust_remote_code=True,
        ),
        CorpusSource(
            name="oscar-pt",
            hf_dataset_id="oscar-corpus/OSCAR-2301",
            hf_config="pt",
            hf_split="train",
            text_field="text",
            language_filter="pt",
            revision="main",  # TODO: pin to specific commit
            max_docs=max_docs,
            trust_remote_code=True,
        ),
        CorpusSource(
            name="br-common-crawl",
            hf_dataset_id="common-crawl/common-crawl",
            hf_config="pt-BR",
            hf_split="train",
            text_field="text",
            language_filter="pt",
            revision="main",  # TODO: pin to specific snapshot
            br_pt_criteria={
                "country_code": "BR",
                "language_score_threshold": 0.9,
                "content_type": "text/html",
            },
            max_docs=max_docs,
            trust_remote_code=True,
        ),
        CorpusSource(
            name="pt-wikipedia",
            hf_dataset_id="wikimedia/wikipedia",
            hf_config="20231101.pt",
            hf_split="train",
            text_field="text",
            language_filter="pt",
            revision="main",  # TODO: pin to specific dump date
            br_pt_criteria={
                "language": "pt",
                "include_br_variant": True,
            },
            max_docs=max_docs,
            trust_remote_code=True,
        ),
    ]


def multilingual_sources(max_docs: int | None = None) -> list[CorpusSource]:
    """Get multilingual replay sources (non-Portuguese).

    Used for the multilingual portion of the curriculum.
    """
    return [
        CorpusSource(
            name="mc4-multilingual",
            hf_dataset_id="mc4",
            hf_config=None,  # all languages
            hf_split="train",
            text_field="text",
            language_filter=None,  # no filter - we'll sample non-PT
            revision="main",
            max_docs=max_docs,
            trust_remote_code=True,
        ),
        CorpusSource(
            name="oscar-multilingual",
            hf_dataset_id="oscar-corpus/OSCAR-2301",
            hf_config=None,
            hf_split="train",
            text_field="text",
            language_filter=None,
            revision="main",
            max_docs=max_docs,
            trust_remote_code=True,
        ),
    ]


def assemble_portuguese_corpus(
    sources: list[CorpusSource] | None = None,
    provenance: CorpusProvenance | None = None,
    streaming: bool = True,
) -> tuple["Iterator[str]", CorpusProvenance]:
    """Assemble Portuguese corpus from multiple sources.

    Args:
        sources: List of CorpusSource specifications. If None, uses defaults.
        provenance: Optional CorpusProvenance to record to. Creates new if None.
        streaming: Whether to use streaming mode (memory efficient).

    Returns:
        Tuple of (document iterator, provenance record).
    """
    from datasets import load_dataset

    if sources is None:
        sources = default_portuguese_sources()

    if provenance is None:
        provenance = CorpusProvenance(corpus_name="portuguese-pretrain")

    for source in sources:
        provenance.add_source(source)

    def document_iterator() -> "Iterator[str]":
        total_docs = 0
        for source in sources:
            logger.info(f"Loading source: {source.name} ({source.hf_dataset_id})")

            try:
                if source.hf_config:
                    ds = load_dataset(
                        source.hf_dataset_id,
                        source.hf_config,
                        split=source.hf_split,
                        streaming=streaming,
                        revision=source.revision,
                        trust_remote_code=source.trust_remote_code,
                    )
                else:
                    ds = load_dataset(
                        source.hf_dataset_id,
                        split=source.hf_split,
                        streaming=streaming,
                        revision=source.revision,
                        trust_remote_code=source.trust_remote_code,
                    )

                count = 0
                for row in ds:
                    if source.max_docs and count >= source.max_docs:
                        break

                    text = row.get(source.text_field, "")
                    if not text or not isinstance(text, str):
                        continue

                    # Apply language filter if specified
                    if source.language_filter:
                        # Check if row has language field
                        lang = row.get("language", row.get("lang", None))
                        if lang and lang != source.language_filter:
                            continue

                    # Apply Brazilian Portuguese criteria
                    if source.br_pt_criteria:
                        if not _matches_br_pt_criteria(row, source.br_pt_criteria):
                            continue

                    yield text
                    count += 1
                    total_docs += 1

                logger.info(f"  Loaded {count} documents from {source.name}")
            except Exception as e:
                logger.error(f"Failed to load {source.name}: {e}")
                provenance.add_note(f"Failed to load {source.name}: {e}")
                continue

        provenance.total_docs_before_filter = total_docs
        provenance.add_note(f"Total documents loaded: {total_docs}")

    return document_iterator(), provenance


def _matches_br_pt_criteria(row: dict, criteria: dict) -> bool:
    """Check if a row matches Brazilian Portuguese selection criteria."""
    # Country code check (for Common Crawl)
    if "country_code" in criteria:
        if row.get("country_code", "").upper() != criteria["country_code"].upper():
            return False

    # Language score threshold
    if "language_score_threshold" in criteria:
        score = row.get("language_score", row.get("lang_score", 1.0))
        if score < criteria["language_score_threshold"]:
            return False

    # Content type
    if "content_type" in criteria:
        if row.get("content_type", "") != criteria["content_type"]:
            return False

    # Wikipedia variant
    if criteria.get("include_br_variant"):
        # For Wikipedia, we accept all pt but could filter by region
        pass

    return True


def filter_and_deduplicate(
    documents: "Iterator[str]",
    provenance: CorpusProvenance,
    min_doc_chars: int = 100,
    max_doc_chars: int = 1_000_000,
    deduplicate: bool = True,
    dedup_method: Literal["exact", "minhash"] = "exact",
) -> tuple[list[str], CorpusProvenance]:
    """Filter and deduplicate documents.

    Args:
        documents: Iterator of document texts.
        provenance: Provenance record to update.
        min_doc_chars: Minimum document length in characters.
        max_doc_chars: Maximum document length in characters.
        deduplicate: Whether to deduplicate.
        dedup_method: Deduplication method ('exact' or 'minhash').

    Returns:
        Tuple of (filtered documents list, updated provenance).
    """
    docs = list(documents)
    provenance.add_filtering_step(
        "initial_load", 0, len(docs), {"total_loaded": len(docs)}
    )

    # Length filtering
    before = len(docs)
    docs = [d for d in docs if min_doc_chars <= len(d) <= max_doc_chars]
    provenance.add_filtering_step(
        "length_filter", before, len(docs), {"min_chars": min_doc_chars, "max_chars": max_doc_chars}
    )

    # Deduplication
    if deduplicate and docs:
        before = len(docs)
        if dedup_method == "exact":
            # Exact deduplication using content hashes
            seen = set()
            unique_docs = []
            for doc in docs:
                doc_hash = hashlib.sha256(doc.encode("utf-8")).hexdigest()[:16]
                if doc_hash not in seen:
                    seen.add(doc_hash)
                    unique_docs.append(doc)
            docs = unique_docs
        else:
            # MinHash deduplication (placeholder - would need datasketch)
            provenance.add_note("MinHash deduplication not implemented; falling back to exact")
            seen = set()
            unique_docs = []
            for doc in docs:
                doc_hash = hashlib.sha256(doc.encode("utf-8")).hexdigest()[:16]
                if doc_hash not in seen:
                    seen.add(doc_hash)
                    unique_docs.append(doc)
            docs = unique_docs

        provenance.add_filtering_step(
            "deduplication", before, len(docs), {"method": dedup_method}
        )
        provenance.deduplication_method = dedup_method

    provenance.total_docs_after_filter = len(docs)
    return docs, provenance


def split_train_eval(
    documents: list[str],
    provenance: CorpusProvenance,
    eval_fraction: float = 0.01,
    seed: int = 42,
) -> tuple[list[str], list[str], CorpusProvenance]:
    """Split documents into train and eval sets.

    Uses deterministic shuffling with fixed seed for reproducibility.
    First document always goes to eval to guarantee non-empty eval.

    Args:
        documents: List of document texts.
        provenance: Provenance record to update.
        eval_fraction: Fraction of documents for eval (0-0.5).
        seed: Random seed for shuffling.

    Returns:
        Tuple of (train_docs, eval_docs, updated provenance).
    """
    import random

    if not 0 <= eval_fraction <= 0.5:
        raise ValueError("eval_fraction must be in [0, 0.5]")

    # Deterministic shuffle
    rng = random.Random(seed)
    shuffled = documents.copy()
    rng.shuffle(shuffled)

    # Stride-based split (like data.py's _val_stride)
    if eval_fraction == 0:
        train_docs = shuffled
        eval_docs = []
    else:
        stride = max(2, round(1.0 / min(eval_fraction, 0.5)))
        eval_docs = [shuffled[i] for i in range(len(shuffled)) if i == 0 or i % stride == stride - 1]
        train_docs = [d for i, d in enumerate(shuffled) if i not in [0] + [j for j in range(stride - 1, len(shuffled), stride)]]

    provenance.eval_split_config = {
        "eval_fraction": eval_fraction,
        "seed": seed,
        "stride": stride if eval_fraction > 0 else 0,
        "train_docs": len(train_docs),
        "eval_docs": len(eval_docs),
    }
    provenance.add_note(f"Train/eval split: {len(train_docs)} train, {len(eval_docs)} eval (fraction={eval_fraction}, seed={seed})")

    return train_docs, eval_docs, provenance