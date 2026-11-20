"""Curriculum scheduling for Portuguese pretraining.

Implements the 50/50 -> 90/10 PT/multilingual curriculum by token count.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class CurriculumStage(StrEnum):
    """Stages of the curriculum."""

    INITIAL = "initial"  # 50/50 PT/multilingual
    TRANSITION = "transition"  # Annealing from 50/50 to 90/10
    FINAL = "final"  # 90/10 PT/multilingual


@dataclass(frozen=True)
class CurriculumSchedule:
    """Configuration for the PT/multilingual curriculum.

    The curriculum anneals from 50% PT / 50% multilingual to
    90% PT / 10% multilingual over the course of training.

    Args:
        initial_pt_ratio: Starting PT ratio (default 0.5 = 50/50).
        final_pt_ratio: Final PT ratio (default 0.9 = 90/10).
        transition_tokens: Number of tokens over which to anneal.
        total_training_tokens: Total training tokens (for scheduling).
    """

    initial_pt_ratio: float = 0.5
    final_pt_ratio: float = 0.9
    transition_tokens: int = 100_000_000_000  # 100B tokens
    total_training_tokens: int = 200_000_000_000  # 200B tokens

    def __post_init__(self) -> None:
        if not 0 <= self.initial_pt_ratio <= 1:
            raise ValueError("initial_pt_ratio must be in [0, 1]")
        if not 0 <= self.final_pt_ratio <= 1:
            raise ValueError("final_pt_ratio must be in [0, 1]")
        if self.transition_tokens < 0:
            raise ValueError("transition_tokens must be non-negative")
        if self.total_training_tokens <= 0:
            raise ValueError("total_training_tokens must be positive")
        if self.transition_tokens > self.total_training_tokens:
            raise ValueError("transition_tokens cannot exceed total_training_tokens")

    @property
    def stage_boundaries(self) -> dict[CurriculumStage, tuple[int, int]]:
        """Return token ranges for each stage."""
        return {
            CurriculumStage.INITIAL: (0, self.transition_tokens // 3 if self.transition_tokens > 0 else 0),
            CurriculumStage.TRANSITION: (
                self.transition_tokens // 3 if self.transition_tokens > 0 else 0,
                self.transition_tokens,
            ),
            CurriculumStage.FINAL: (self.transition_tokens, self.total_training_tokens),
        }

    def pt_ratio_at(self, tokens_seen: int) -> float:
        """Compute PT ratio at a given token count.

        Linear annealing from initial_pt_ratio to final_pt_ratio
        over transition_tokens, then stays at final_pt_ratio.
        """
        if tokens_seen >= self.transition_tokens:
            return self.final_pt_ratio
        if self.transition_tokens == 0:
            return self.final_pt_ratio

        progress = tokens_seen / self.transition_tokens
        return self.initial_pt_ratio + progress * (self.final_pt_ratio - self.initial_pt_ratio)

    def multilingual_ratio_at(self, tokens_seen: int) -> float:
        """Compute multilingual ratio at a given token count."""
        return 1.0 - self.pt_ratio_at(tokens_seen)

    def stage_at(self, tokens_seen: int) -> CurriculumStage:
        """Determine curriculum stage at given token count."""
        if tokens_seen < self.transition_tokens:
            if tokens_seen < self.transition_tokens / 3:
                return CurriculumStage.INITIAL
            return CurriculumStage.TRANSITION
        return CurriculumStage.FINAL

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "initial_pt_ratio": self.initial_pt_ratio,
            "final_pt_ratio": self.final_pt_ratio,
            "transition_tokens": self.transition_tokens,
            "total_training_tokens": self.total_training_tokens,
            "stage_boundaries": {
                stage.value: {"start": start, "end": end}
                for stage, (start, end) in self.stage_boundaries.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> CurriculumSchedule:
        """Create from dictionary."""
        return cls(
            initial_pt_ratio=data.get("initial_pt_ratio", 0.5),
            final_pt_ratio=data.get("final_pt_ratio", 0.9),
            transition_tokens=data.get("transition_tokens", 100_000_000_000),
            total_training_tokens=data.get("total_training_tokens", 200_000_000_000),
        )


def compute_curriculum_mix(
    pt_tokens: int,
    multilingual_tokens: int,
    target_pt_ratio: float,
) -> tuple[int, int]:
    """Compute how many tokens to sample from each corpus to achieve target ratio.

    Args:
        pt_tokens: Available PT tokens.
        multilingual_tokens: Available multilingual tokens.
        target_pt_ratio: Target PT ratio (0-1).

    Returns:
        Tuple of (pt_tokens_to_use, multilingual_tokens_to_use).
    """
    if target_pt_ratio <= 0:
        return 0, min(multilingual_tokens, pt_tokens + multilingual_tokens)
    if target_pt_ratio >= 1:
        return min(pt_tokens, pt_tokens + multilingual_tokens), 0

    # We want: pt_used / (pt_used + multi_used) = target_pt_ratio
    # So: pt_used = target_pt_ratio * (pt_used + multi_used)
    # pt_used * (1 - target_pt_ratio) = target_pt_ratio * multi_used
    # multi_used = pt_used * (1 - target_pt_ratio) / target_pt_ratio

    # Use as much PT as possible, then compute multilingual needed
    max_pt = pt_tokens
    needed_multi = int(max_pt * (1 - target_pt_ratio) / target_pt_ratio)

    if needed_multi <= multilingual_tokens:
        return max_pt, needed_multi
    else:
        # Limited by multilingual, compute PT needed
        max_multi = multilingual_tokens
        needed_pt = int(max_multi * target_pt_ratio / (1 - target_pt_ratio))
        return min(needed_pt, pt_tokens), max_multi


def apply_curriculum(
    pt_documents: list[str],
    multilingual_documents: list[str],
    schedule: CurriculumSchedule,
    tokenizer,
    tokens_seen: int = 0,
) -> tuple[list[str], dict]:
    """Apply curriculum to select documents for current training step.

    This is a simplified version that selects documents based on
    the target PT ratio at the current token count. In practice,
    this would be used during training to sample batches.

    Args:
        pt_documents: Portuguese documents.
        multilingual_documents: Multilingual documents.
        schedule: Curriculum schedule.
        tokenizer: Tokenizer to count tokens.
        tokens_seen: Tokens already trained on.

    Returns:
        Tuple of (selected_documents, mix_report).
    """
    pt_ratio = schedule.pt_ratio_at(tokens_seen)
    stage = schedule.stage_at(tokens_seen)

    # Tokenize to count tokens (could be cached)
    pt_token_counts = [len(tokenizer.encode(doc, eos=True)) for doc in pt_documents]
    multi_token_counts = [len(tokenizer.encode(doc, eos=True)) for doc in multilingual_documents]

    total_pt_tokens = sum(pt_token_counts)
    total_multi_tokens = sum(multi_token_counts)

    target_pt_tokens, target_multi_tokens = compute_curriculum_mix(
        total_pt_tokens, total_multi_tokens, pt_ratio
    )

    # Select documents to meet targets (greedy by token count)
    selected = []
    pt_tokens_used = 0
    multi_tokens_used = 0

    # Sort by token count descending to use larger docs first
    pt_sorted = sorted(zip(pt_documents, pt_token_counts, strict=False), key=lambda x: -x[1])
    multi_sorted = sorted(zip(multilingual_documents, multi_token_counts, strict=False), key=lambda x: -x[1])

    for doc, count in pt_sorted:
        if pt_tokens_used + count <= target_pt_tokens:
            selected.append(doc)
            pt_tokens_used += count

    for doc, count in multi_sorted:
        if multi_tokens_used + count <= target_multi_tokens:
            selected.append(doc)
            multi_tokens_used += count

    report = {
        "stage": stage.value,
        "tokens_seen": tokens_seen,
        "target_pt_ratio": pt_ratio,
        "actual_pt_ratio": pt_tokens_used / max(1, pt_tokens_used + multi_tokens_used),
        "pt_tokens_used": pt_tokens_used,
        "multilingual_tokens_used": multi_tokens_used,
        "total_tokens_selected": pt_tokens_used + multi_tokens_used,
        "pt_docs_selected": sum(1 for d in selected if d in pt_documents),
        "multilingual_docs_selected": sum(1 for d in selected if d in multilingual_documents),
    }

    return selected, report


def curriculum_token_iterator(
    pt_documents: list[str],
    multilingual_documents: list[str],
    schedule: CurriculumSchedule,
    tokenizer,
    batch_tokens: int = 4096,
) -> Iterator[tuple[str, Literal["pt", "multilingual"]]]:
    """Infinite iterator yielding documents according to curriculum schedule.

    This is designed for use during training. It yields (document, source_type)
    pairs where source_type is "pt" or "multilingual".

    Args:
        pt_documents: Portuguese documents.
        multilingual_documents: Multilingual documents.
        schedule: Curriculum schedule.
        tokenizer: Tokenizer for counting.
        batch_tokens: Target tokens per yield (approximate).

    Yields:
        (document_text, source_type) pairs.
    """
    import random

    # Shuffle once per epoch
    rng = random.Random(42)
    pt_idx = 0
    multi_idx = 0
    pt_shuffled = pt_documents.copy()
    multi_shuffled = multilingual_documents.copy()
    rng.shuffle(pt_shuffled)
    rng.shuffle(multi_shuffled)

    tokens_seen = 0

    while True:
        pt_ratio = schedule.pt_ratio_at(tokens_seen)

        # Decide which corpus to sample from
        if rng.random() < pt_ratio and pt_shuffled:
            doc = pt_shuffled[pt_idx % len(pt_shuffled)]
            pt_idx += 1
            source = "pt"
        elif multi_shuffled:
            doc = multi_shuffled[multi_idx % len(multi_shuffled)]
            multi_idx += 1
            source = "multilingual"
        else:
            # Fallback to PT if multilingual exhausted
            doc = pt_shuffled[pt_idx % len(pt_shuffled)]
            pt_idx += 1
            source = "pt"

        yield doc, source
        tokens_seen += len(tokenizer.encode(doc, eos=True))

        # Reshuffle at epoch boundaries
        if pt_idx % len(pt_shuffled) == 0 and pt_shuffled:
            rng.shuffle(pt_shuffled)
        if multi_idx % len(multi_shuffled) == 0 and multi_shuffled:
            rng.shuffle(multi_shuffled)
