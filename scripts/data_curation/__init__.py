"""Data curation for Portuguese pretraining corpus."""

from .corpus_assembler import (
    CorpusSource,
    CorpusProvenance,
    assemble_portuguese_corpus,
    filter_and_deduplicate,
    split_train_eval,
    default_portuguese_sources,
    multilingual_sources,
)
from .curriculum import (
    CurriculumSchedule,
    compute_curriculum_mix,
    apply_curriculum,
    CurriculumStage,
)
from .tokenizer_utils import (
    tokenize_corpus,
    record_token_counts,
    TokenizationReport,
    prepare_curriculum_data,
)

__all__ = [
    "CorpusSource",
    "CorpusProvenance",
    "assemble_portuguese_corpus",
    "filter_and_deduplicate",
    "split_train_eval",
    "default_portuguese_sources",
    "multilingual_sources",
    "CurriculumSchedule",
    "compute_curriculum_mix",
    "apply_curriculum",
    "CurriculumStage",
    "tokenize_corpus",
    "record_token_counts",
    "TokenizationReport",
    "prepare_curriculum_data",
]