"""Tests for generative evaluation (Brazilian Portuguese)."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from cafezin_lm import Tokenizer, build_model, tiny_config
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

# Add scripts to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from eval_generative_br import (
    GenerationConfig,
    Prompt,
    load_prompts,
    load_rubric,
    heuristic_judge,
    compute_aggregate,
    EvaluationResult,
    JudgeScore,
    ModelResponse,
)


def create_test_tokenizer(tmpdir: Path) -> Path:
    """Create a minimal test tokenizer."""
    tok = HFTokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=500, special_tokens=["[UNK]", "[EOS]", "[PAD]"])
    corpus = [
        "olá mundo como vai você " * 20,
        "brasil brasileiro são paulo rio " * 15,
        "carnaval samba futebol praia " * 15,
    ]
    tok.train_from_iterator(corpus, trainer=trainer)
    tok_path = tmpdir / "tokenizer.json"
    tok.save(str(tok_path))
    return tok_path


def create_test_model(tokenizer_path: Path, model_path: Path, seed: int = 42) -> None:
    """Create a tiny test model checkpoint."""
    tokenizer = Tokenizer(str(tokenizer_path))
    cfg = tiny_config(vocab_size=tokenizer.vocab_size)
    model = build_model(cfg, seed=seed)
    torch.save(model.state_dict(), model_path)


class TestGenerativeEvaluation:
    """Test generative evaluation components."""

    def test_load_prompts(self):
        """Test loading prompts from JSON."""
        prompts_path = Path(__file__).parent.parent / "eval" / "generative" / "prompts_v1.json"
        prompts = load_prompts(prompts_path)
        
        assert len(prompts) == 12
        assert all(isinstance(p, Prompt) for p in prompts)
        assert prompts[0].id == "culture_001"
        assert prompts[0].category == "culture"
        assert "Carnaval" in prompts[0].prompt

    def test_load_rubric(self):
        """Test loading rubric from JSON."""
        rubric_path = Path(__file__).parent.parent / "eval" / "generative" / "rubric_v1.json"
        rubric = load_rubric(rubric_path)
        
        assert rubric["version"] == "1.0"
        assert "dimensions" in rubric
        assert len(rubric["dimensions"]) == 5
        assert "judge_protocol" in rubric

    def test_heuristic_judge(self):
        """Test heuristic judge produces valid scores."""
        rubric_path = Path(__file__).parent.parent / "eval" / "generative" / "rubric_v1.json"
        rubric = load_rubric(rubric_path)
        
        prompt = "Explique o Carnaval do Rio de Janeiro."
        response = "O Carnaval do Rio é famoso pelos desfiles das escolas de samba no Sambódromo, com samba-enredo e arquibancadas. É um espetáculo competitivo."
        
        score = heuristic_judge(prompt, response, rubric, "heuristic")
        
        assert isinstance(score, JudgeScore)
        assert all(k in score.scores for k in ["factual_accuracy", "cultural_competence", "language_quality", "reasoning_depth", "completeness"])
        assert all(1 <= v <= 5 for v in score.scores.values())
        assert 1.0 <= score.final_score <= 5.0
        assert score.judge_model == "heuristic"

    def test_heuristic_judge_portuguese_response(self):
        """Test heuristic judge with Portuguese response."""
        rubric_path = Path(__file__).parent.parent / "eval" / "generative" / "rubric_v1.json"
        rubric = load_rubric(rubric_path)
        
        prompt = "Descreva o SUS."
        response = "O SUS (Sistema Único de Saúde) é o sistema de saúde público do Brasil. Princípios: universalidade, integralidade, equidade. Conquistas: vacinação, redução mortalidade infantil. Desafios: subfinanciamento, filas."
        
        score = heuristic_judge(prompt, response, rubric, "heuristic")
        
        # Should score well on cultural competence and language quality
        assert score.scores["cultural_competence"] >= 3
        assert score.scores["language_quality"] >= 3
        assert score.scores["factual_accuracy"] >= 3

    def test_compute_aggregate(self):
        """Test aggregate computation."""
        # Create mock results
        prompts = [
            Prompt("culture_001", "culture", "prompt", [], "easy"),
            Prompt("knowledge_001", "knowledge", "prompt", [], "medium"),
            Prompt("writing_001", "writing", "prompt", [], "hard"),
        ]
        
        results = []
        for i, p in enumerate(prompts):
            # Baseline scores
            b_score = JudgeScore(
                model_name="baseline", prompt_id=p.id,
                scores={"factual_accuracy": 4, "cultural_competence": 3, "language_quality": 4, "reasoning_depth": 3, "completeness": 4},
                justifications={}, final_score=3.6, judge_model="test", judge_timestamp="2024-01-01"
            )
            # Candidate scores (slightly better)
            c_score = JudgeScore(
                model_name="candidate", prompt_id=p.id,
                scores={"factual_accuracy": 4, "cultural_competence": 4, "language_quality": 4, "reasoning_depth": 4, "completeness": 4},
                justifications={}, final_score=3.8, judge_model="test", judge_timestamp="2024-01-01"
            )
            
            results.append(EvaluationResult(
                prompt=p,
                baseline_response=ModelResponse("baseline", p.id, p.prompt, "resp", GenerationConfig(), 100, 50),
                candidate_response=ModelResponse("candidate", p.id, p.prompt, "resp", GenerationConfig(), 100, 50),
                baseline_scores=b_score,
                candidate_scores=c_score
            ))
        
        aggregate = compute_aggregate(results)
        
        assert aggregate["overall"]["baseline_mean"] == 3.6
        assert aggregate["overall"]["candidate_mean"] == 3.8
        assert aggregate["overall"]["difference"] == 0.2
        assert aggregate["win_rate"]["candidate_wins"] == 3
        assert aggregate["win_rate"]["baseline_wins"] == 0
        assert "culture" in aggregate["by_category"]
        assert "knowledge" in aggregate["by_category"]
        assert "writing" in aggregate["by_category"]

    def test_generation_config_defaults(self):
        """Test GenerationConfig default values."""
        config = GenerationConfig()
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 50
        assert config.max_tokens == 1024
        assert config.repetition_penalty == 1.1
        assert config.do_sample is True

    def test_prompt_categories_covered(self):
        """Test that all four categories are present in prompts."""
        prompts_path = Path(__file__).parent.parent / "eval" / "generative" / "prompts_v1.json"
        prompts = load_prompts(prompts_path)
        
        categories = {p.category for p in prompts}
        assert categories == {"culture", "knowledge", "reasoning", "writing"}
        
        # Each category should have 3 prompts
        for cat in categories:
            cat_prompts = [p for p in prompts if p.category == cat]
            assert len(cat_prompts) == 3

    def test_difficulty_levels(self):
        """Test that all difficulty levels are present."""
        prompts_path = Path(__file__).parent.parent / "eval" / "generative" / "prompts_v1.json"
        prompts = load_prompts(prompts_path)
        
        difficulties = {p.difficulty for p in prompts}
        assert difficulties == {"easy", "medium", "hard"}

    def test_rubric_weights_sum_to_one(self):
        """Test that rubric dimension weights sum to 1.0."""
        rubric_path = Path(__file__).parent.parent / "eval" / "generative" / "rubric_v1.json"
        rubric = load_rubric(rubric_path)
        
        total_weight = sum(d["weight"] for d in rubric["dimensions"].values())
        assert abs(total_weight - 1.0) < 0.001

    def test_rubric_has_all_dimensions(self):
        """Test that rubric has all required dimensions."""
        rubric_path = Path(__file__).parent.parent / "eval" / "generative" / "rubric_v1.json"
        rubric = load_rubric(rubric_path)
        
        expected_dims = {"factual_accuracy", "cultural_competence", "language_quality", "reasoning_depth", "completeness"}
        assert set(rubric["dimensions"].keys()) == expected_dims

    def test_limitations_documented(self):
        """Test that limitations are documented."""
        from eval_generative_br import get_limitations
        
        limitations = get_limitations()
        assert len(limitations) >= 10
        assert any("heuristic" in lim.lower() for lim in limitations)
        assert any("statistical" in lim.lower() for lim in limitations)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])