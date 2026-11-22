"""Tests for Portuguese data curation pipeline."""

import tempfile
from pathlib import Path

import pytest
from scripts.data_curation import (
    CorpusProvenance,
    CorpusSource,
    CurriculumSchedule,
    CurriculumStage,
    filter_and_deduplicate,
    prepare_curriculum_data,
    record_token_counts,
    split_train_eval,
    tokenize_corpus,
)

from cafezin_lm.tokenizer import Tokenizer
from cafezin_lm.tokenizer_extension import extend_tokenizer, mine_candidate_tokens


def create_test_tokenizer(tmp_path: Path) -> Tokenizer:
    """Create a simple test tokenizer."""
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import BpeTrainer

    tok = HFTokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=500, special_tokens=["[UNK]", "[EOS]"])
    corpus = [
        "hello world " * 50,
        "hello there " * 30,
        "world peace " * 20,
        "O café é bom " * 30,
        "Brasil é grande " * 20,
        "português brasileiro " * 25,
    ]
    tok.train_from_iterator(corpus, trainer=trainer)

    tok_path = tmp_path / "test_tok.json"
    tok.save(str(tok_path))
    return Tokenizer(tok_path)


def test_corpus_source_provenance():
    """Test CorpusSource provenance serialization."""
    source = CorpusSource(
        name="test-source",
        hf_dataset_id="test/dataset",
        hf_config="pt",
        revision="abc123",
        br_pt_criteria={"country_code": "BR"},
    )
    prov_dict = source.provenance_dict()
    assert prov_dict["name"] == "test-source"
    assert prov_dict["hf_dataset_id"] == "test/dataset"
    assert prov_dict["revision"] == "abc123"
    assert prov_dict["br_pt_criteria"]["country_code"] == "BR"


def test_corpus_provenance_serialization(tmp_path: Path):
    """Test CorpusProvenance save/load."""
    prov = CorpusProvenance(corpus_name="test-corpus")
    prov.add_source(CorpusSource(name="src1", hf_dataset_id="test/ds1"))
    prov.add_filtering_step("test_step", 100, 80, {"param": "value"})
    prov.add_note("Test note")

    save_path = tmp_path / "provenance.json"
    prov.save(save_path)

    loaded = CorpusProvenance.load(save_path)
    assert loaded.corpus_name == "test-corpus"
    assert len(loaded.sources) == 1
    assert loaded.sources[0].name == "src1"
    assert len(loaded.filtering_steps) == 1
    assert loaded.filtering_steps[0]["step"] == "test_step"
    # Note includes timestamp prefix
    assert len(loaded.notes) == 1
    assert loaded.notes[0].endswith("Test note")


def test_curriculum_schedule():
    """Test CurriculumSchedule configuration and methods."""
    schedule = CurriculumSchedule(
        initial_pt_ratio=0.5,
        final_pt_ratio=0.9,
        transition_tokens=100,
        total_training_tokens=200,
    )

    # Test stage boundaries
    boundaries = schedule.stage_boundaries
    assert boundaries[CurriculumStage.INITIAL] == (0, 33)
    assert boundaries[CurriculumStage.TRANSITION] == (33, 100)
    assert boundaries[CurriculumStage.FINAL] == (100, 200)

    # Test PT ratio at different points
    assert schedule.pt_ratio_at(0) == 0.5
    assert schedule.pt_ratio_at(50) == pytest.approx(0.7, rel=0.01)
    assert schedule.pt_ratio_at(100) == 0.9
    assert schedule.pt_ratio_at(150) == 0.9

    # Test stage at
    assert schedule.stage_at(0) == CurriculumStage.INITIAL
    assert schedule.stage_at(50) == CurriculumStage.TRANSITION
    assert schedule.stage_at(150) == CurriculumStage.FINAL

    # Test serialization
    data = schedule.to_dict()
    loaded = CurriculumSchedule.from_dict(data)
    assert loaded.initial_pt_ratio == 0.5
    assert loaded.final_pt_ratio == 0.9


def test_compute_curriculum_mix():
    """Test curriculum mix computation."""
    from scripts.data_curation.curriculum import compute_curriculum_mix

    # 50/50 target with equal available tokens
    pt, multi = compute_curriculum_mix(1000, 1000, 0.5)
    assert pt == 1000
    assert multi == 1000

    # 90/10 target
    pt, multi = compute_curriculum_mix(1000, 1000, 0.9)
    assert pt == 1000
    assert multi == pytest.approx(111, rel=0.05)  # 1000 * 0.1 / 0.9

    # Limited multilingual
    pt, multi = compute_curriculum_mix(1000, 50, 0.5)
    assert multi == 50
    assert pt == 50


def test_tokenize_corpus():
    """Test tokenize_corpus function."""
    with tempfile.TemporaryDirectory() as tmp:
        tok = create_test_tokenizer(Path(tmp))
        documents = ["hello world", "hello there", "world peace"]

        tokenized, report = tokenize_corpus(documents, tok, "test-corpus")

        assert len(tokenized) == 3
        assert report.num_documents == 3
        assert report.total_tokens > 0
        assert report.chars_per_token > 0
        assert report.tokenizer_vocab_size == tok.vocab_size


def test_record_token_counts():
    """Test record_token_counts for PT and multilingual corpora."""
    with tempfile.TemporaryDirectory() as tmp:
        tok = create_test_tokenizer(Path(tmp))
        pt_docs = ["O café é bom", "Brasil é grande", "português brasileiro"]
        multi_docs = ["hello world", "hello there", "world peace"]

        schedule = CurriculumSchedule()
        pt_report, multi_report, combined_report = record_token_counts(
            pt_docs, multi_docs, tok, schedule, tokens_seen=0
        )

        assert pt_report.pt_tokens > 0
        assert multi_report.multilingual_tokens > 0
        assert combined_report.total_tokens == pt_report.total_tokens + multi_report.total_tokens
        assert combined_report.pt_ratio == pt_report.total_tokens / combined_report.total_tokens
        assert combined_report.curriculum_stage == "initial"


def test_split_train_eval():
    """Test train/eval split."""
    documents = [f"doc {i}" for i in range(100)]
    prov = CorpusProvenance(corpus_name="test")

    train, eval_docs, prov = split_train_eval(documents, prov, eval_fraction=0.1, seed=42)

    assert len(train) + len(eval_docs) == 100
    assert len(eval_docs) >= 1  # At least first doc
    assert prov.eval_split_config is not None
    assert prov.eval_split_config["eval_fraction"] == 0.1


def test_filter_and_deduplicate():
    """Test filtering and deduplication."""
    documents = [
        "a" * 50,  # Too short (min=100)
        "b" * 200,
        "c" * 200,  # Duplicate of b after content (different content)
        "b" * 200,  # Exact duplicate
        "d" * 500000,  # Too long (max=100000)
    ]
    prov = CorpusProvenance(corpus_name="test")

    filtered, prov = filter_and_deduplicate(
        iter(documents), prov, min_doc_chars=100, max_doc_chars=100000
    )

    # Should keep: "b" * 200, "c" * 200 (dedup keeps first), but "b" * 200 is duplicate
    # So: "b" * 200, "c" * 200 = 2 docs
    assert len(filtered) == 2
    assert prov.deduplication_method == "exact"


def test_extend_tokenizer_with_portuguese_corpus():
    """Test tokenizer extension with Portuguese-like corpus."""
    with tempfile.TemporaryDirectory() as tmp:
        tok = create_test_tokenizer(Path(tmp))
        pt_corpus = [
            "O café é uma bebida popular no Brasil",
            "Os brasileiros gostam de café coado",
            "São Paulo é a maior cidade do Brasil",
            "O português do Brasil tem sotaque único",
        ] * 10  # Repeat to get frequency

        candidates = mine_candidate_tokens(tok, pt_corpus, min_frequency=5, max_candidates=20)
        assert len(candidates) > 0

        # Extend tokenizer
        new_tokens = [c[0] for c in candidates]
        extended_tok = extend_tokenizer(tok, new_tokens, f"{tmp}/extended.json")

        assert extended_tok.vocab_size == tok.vocab_size + len(new_tokens)
        # Original IDs preserved
        orig_vocab = tok._tokenizer.get_vocab(with_added_tokens=True)
        ext_vocab = extended_tok._tokenizer.get_vocab(with_added_tokens=True)
        for token, tid in orig_vocab.items():
            assert ext_vocab[token] == tid


def test_prepare_curriculum_data():
    """Test full curriculum data preparation."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        tok = create_test_tokenizer(tmp_path)

        pt_train = ["O café é bom"] * 10
        pt_eval = ["Brasil é grande"] * 2
        multi_train = ["hello world"] * 10
        multi_eval = ["world peace"] * 2

        schedule = CurriculumSchedule(
            initial_pt_ratio=0.5,
            final_pt_ratio=0.9,
            transition_tokens=1000,
            total_training_tokens=2000,
        )

        result = prepare_curriculum_data(
            pt_train, pt_eval, multi_train, multi_eval, tok, schedule, tmp_path / "curriculum_out"
        )

        assert "output_dir" in result
        assert "meta" in result
        assert (tmp_path / "curriculum_out" / "meta.json").exists()
        assert (tmp_path / "curriculum_out" / "curriculum_schedule.json").exists()
        assert (tmp_path / "curriculum_out" / "train_pt.bin").exists()
        assert (tmp_path / "curriculum_out" / "train_multilingual.bin").exists()

        # Check meta contains curriculum info
        import json

        meta = json.loads((tmp_path / "curriculum_out" / "meta.json").read_text())
        assert "curriculum" in meta
        assert meta["curriculum"]["initial_pt_ratio"] == 0.5
        assert meta["curriculum"]["final_pt_ratio"] == 0.9


def test_curriculum_token_iterator():
    """Test curriculum token iterator."""
    from scripts.data_curation.curriculum import curriculum_token_iterator

    with tempfile.TemporaryDirectory() as tmp:
        tok = create_test_tokenizer(Path(tmp))
        pt_docs = ["O café é bom"] * 5
        multi_docs = ["hello world"] * 5
        schedule = CurriculumSchedule(
            initial_pt_ratio=0.5,
            final_pt_ratio=0.9,
            transition_tokens=1000,
            total_training_tokens=2000,
        )

        iterator = curriculum_token_iterator(pt_docs, multi_docs, schedule, tok)
        samples = []
        for _ in range(20):
            doc, source = next(iterator)
            samples.append((doc, source))

        # Should have both sources
        sources = [s for _, s in samples]
        assert "pt" in sources
        assert "multilingual" in sources


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
