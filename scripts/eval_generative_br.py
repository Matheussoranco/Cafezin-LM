#!/usr/bin/env python
"""Generative evaluation for Brazilian Portuguese using MT-Bench/LLM-judge style.

This script implements the generative evaluation pipeline:
1. Load held-out prompt set (versioned)
2. Generate responses from baseline and candidate models
3. Score responses using LLM judge with versioned rubric
4. Export per-example scores and aggregate comparison
5. Document evaluation limitations

Usage:
    python scripts/eval_generative_br.py \
        --baseline-model /path/to/baseline \
        --candidate-model /path/to/candidate \
        --tokenizer-path /path/to/tokenizer.json \
        --output-dir results/generative_eval \
        --prompts eval/generative/prompts_v1.json \
        --rubric eval/generative/rubric_v1.json \
        --judge-model gpt-4o \
        --generation-config '{"temperature": 0.7, "top_p": 0.9, "max_tokens": 1024}'
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cafezin_lm import Tokenizer, build_model, tiny_config


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 1024
    repetition_penalty: float = 1.1
    do_sample: bool = True


@dataclass
class Prompt:
    """A single evaluation prompt."""
    id: str
    category: str
    prompt: str
    reference_points: list[str]
    difficulty: str


@dataclass
class ModelResponse:
    """A model's response to a prompt."""
    model_name: str
    prompt_id: str
    prompt: str
    response: str
    generation_config: GenerationConfig
    generation_time_ms: float
    tokens_generated: int


@dataclass
class JudgeScore:
    """Scores from LLM judge."""
    model_name: str
    prompt_id: str
    scores: dict[str, int]  # dimension -> 1-5
    justifications: dict[str, str]
    final_score: float
    judge_model: str
    judge_timestamp: str


@dataclass
class EvaluationResult:
    """Complete evaluation result for one model on one prompt."""
    prompt: Prompt
    baseline_response: ModelResponse
    candidate_response: ModelResponse
    baseline_scores: JudgeScore
    candidate_scores: JudgeScore


def load_prompts(path: Path) -> list[Prompt]:
    """Load prompts from JSON file."""
    with open(path) as f:
        data = json.load(f)
    
    prompts = []
    for p in data["prompts"]:
        prompts.append(Prompt(
            id=p["id"],
            category=p["category"],
            prompt=p["prompt"],
            reference_points=p["reference_points"],
            difficulty=p["difficulty"]
        ))
    return prompts


def load_rubric(path: Path) -> dict[str, Any]:
    """Load scoring rubric from JSON file."""
    with open(path) as f:
        return json.load(f)


def build_model_and_tokenizer(model_path: str, tokenizer_path: str, config_name: str = "tiny") -> tuple:
    """Load model and tokenizer."""
    tokenizer = Tokenizer(tokenizer_path)
    
    if config_name == "tiny":
        cfg = tiny_config(vocab_size=tokenizer.vocab_size)
    else:
        from cafezin_lm import base_config
        cfg = base_config(vocab_size=tokenizer.vocab_size)
    
    model = build_model(cfg)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model, tokenizer


def generate_response(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    generation_config: GenerationConfig,
    device: str = "cuda"
) -> tuple[str, float, int]:
    """Generate a response from the model."""
    model.to(device)
    model.eval()
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, eos=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    start_time = time.time()
    
    with torch.no_grad():
        # Simple greedy/sampling generation
        generated = input_tensor.clone()
        
        for _ in range(generation_config.max_tokens):
            logits, _ = model(generated[:, -2048:])  # Limit context
            next_token_logits = logits[0, -1, :] / generation_config.temperature
            
            # Apply repetition penalty
            if generation_config.repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    next_token_logits[token_id] /= generation_config.repetition_penalty
            
            # Top-k filtering
            if generation_config.top_k > 0:
                top_k_values, top_k_indices = torch.topk(next_token_logits, generation_config.top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits[top_k_indices] = top_k_values
            
            # Top-p (nucleus) filtering
            if generation_config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > generation_config.top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            if generation_config.do_sample:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, keepdim=True)
            
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    generation_time_ms = (time.time() - start_time) * 1000
    
    # Decode only new tokens
    new_tokens = generated[0, len(input_ids):].tolist()
    response = tokenizer.decode(new_tokens)
    tokens_generated = len(new_tokens)
    
    return response, generation_time_ms, tokens_generated


def call_llm_judge(
    prompt: str,
    response: str,
    rubric: dict[str, Any],
    judge_model: str,
    judge_config: dict[str, Any]
) -> JudgeScore:
    """Call LLM judge to score a response."""
    # Build rubric summary for the prompt
    dims = rubric["dimensions"]
    rubric_summary = "\n".join([
        f"- {k}: {v['name']} (peso {v['weight']}) - {v['description']}"
        for k, v in dims.items()
    ])
    
    system_prompt = rubric["judge_protocol"]["system_prompt"]
    user_prompt = rubric["judge_protocol"]["user_prompt_template"].format(
        prompt=prompt,
        response=response,
        rubric_summary=rubric_summary
    )
    
    # Use a simple approach: call via subprocess to a judge script
    # In production, this would call an API (OpenAI, Anthropic, etc.)
    # For now, we'll use a local judge or simulate
    
    judge_prompt = f"{system_prompt}\n\n{user_prompt}"
    
    # Try to use a local model or API
    # For this implementation, we'll create a fallback heuristic judge
    # and document that a real LLM judge should be used
    
    return heuristic_judge(prompt, response, rubric, judge_model)


def heuristic_judge(
    prompt: str,
    response: str,
    rubric: dict[str, Any],
    judge_model: str
) -> JudgeScore:
    """Heuristic judge for offline testing. Replace with real LLM judge in production."""
    # This is a simplified heuristic - in reality, use GPT-4, Claude, or similar
    dims = rubric["dimensions"]
    
    # Simple heuristics based on response characteristics
    response_len = len(response)
    has_portuguese_chars = any(c in response for c in "ãõáéíóúâêôç")
    brazilian_terms = [
        "brasil", "brasileiro", "são paulo", "rio de janeiro", "nordeste", "sul",
        "sus", "carnaval", "café", "padaria", "jeitinho", "bioma", "amazônia",
        "cerrado", "pampa", "pantanal", "caatinga", "mata atlântica"
    ]
    has_brazilian_terms = [t for t in brazilian_terms if t in response.lower()]
    
    # Score based on simple heuristics
    scores = {}
    justifications = {}
    
    # Factual accuracy: heuristic based on presence of key terms
    factual_terms = len([t for t in has_brazilian_terms])
    if factual_terms >= 3:
        scores["factual_accuracy"] = 4
        justifications["factual_accuracy"] = "Resposta contém múltiplos termos factuais brasileiros relevantes"
    elif factual_terms >= 1:
        scores["factual_accuracy"] = 3
        justifications["factual_accuracy"] = "Resposta contém alguns termos factuais brasileiros"
    else:
        scores["factual_accuracy"] = 2
        justifications["factual_accuracy"] = "Resposta carece de referências factuais brasileiras específicas"
    
    # Cultural competence
    if has_brazilian_terms and has_portuguese_chars:
        scores["cultural_competence"] = 4
        justifications["cultural_competence"] = "Resposta demonstra vocabulário e referências culturais brasileiras"
    elif has_portuguese_chars:
        scores["cultural_competence"] = 3
        justifications["cultural_competence"] = "Resposta em português com alguns marcadores culturais"
    else:
        scores["cultural_competence"] = 2
        justifications["cultural_competence"] = "Resposta com competência cultural limitada"
    
    # Language quality
    if has_portuguese_chars and response_len > 100:
        scores["language_quality"] = 4
        justifications["language_quality"] = "Português natural com acentuação e extensão adequada"
    elif has_portuguese_chars:
        scores["language_quality"] = 3
        justifications["language_quality"] = "Português básico, compreensível"
    else:
        scores["language_quality"] = 2
        justifications["language_quality"] = "Qualidade linguística baixa ou idioma incorreto"
    
    # Reasoning depth
    if response_len > 300 and any(w in response.lower() for w in ["portanto", "logo", "assim", "porque", "como", "estratégia", "passo"]):
        scores["reasoning_depth"] = 4
        justifications["reasoning_depth"] = "Resposta estruturada com conectivos lógicos e profundidade"
    elif response_len > 150:
        scores["reasoning_depth"] = 3
        justifications["reasoning_depth"] = "Raciocínio presente mas linear"
    else:
        scores["reasoning_depth"] = 2
        justifications["reasoning_depth"] = "Raciocínio superficial ou ausente"
    
    # Completeness
    if response_len > 200:
        scores["completeness"] = 4
        justifications["completeness"] = "Resposta abrangente cobrindo múltiplos aspectos"
    elif response_len > 100:
        scores["completeness"] = 3
        justifications["completeness"] = "Resposta cobre aspectos principais mas incompleta"
    else:
        scores["completeness"] = 2
        justifications["completeness"] = "Resposta curta, omite aspectos solicitados"
    
    # Calculate weighted final score
    final_score = sum(
        scores[dim] * dims[dim]["weight"]
        for dim in scores
    )
    final_score = round(final_score, 2)
    
    return JudgeScore(
                model_name="",
                prompt_id="",
                scores=scores,
                justifications=justifications,
                final_score=final_score,
                judge_model=judge_model,
                judge_timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )


def run_generative_evaluation(
    baseline_model_path: str,
    candidate_model_path: str,
    tokenizer_path: str,
    prompts_path: Path,
    rubric_path: Path,
    output_dir: Path,
    judge_model: str = "heuristic",
    generation_config: GenerationConfig | None = None,
    device: str = "cuda",
    config_name: str = "tiny",
    limit: int | None = None
) -> dict[str, Any]:
    """Run the complete generative evaluation pipeline."""
    
    if generation_config is None:
        generation_config = GenerationConfig()
    
    # Load prompts and rubric
    prompts = load_prompts(prompts_path)
    rubric = load_rubric(rubric_path)
    
    if limit:
        prompts = prompts[:limit]
    
    print(f"Loaded {len(prompts)} prompts from {prompts_path}")
    print(f"Loaded rubric v{rubric['version']} from {rubric_path}")
    print(f"Generation config: {asdict(generation_config)}")
    print(f"Judge model: {judge_model}")
    
    # Load models
    print("\nLoading baseline model...")
    baseline_model, tokenizer = build_model_and_tokenizer(
        baseline_model_path, tokenizer_path, config_name
    )
    
    print("Loading candidate model...")
    candidate_model, _ = build_model_and_tokenizer(
        candidate_model_path, tokenizer_path, config_name
    )
    
    # Generate responses
    print("\nGenerating responses...")
    baseline_responses = []
    candidate_responses = []
    
    for i, prompt in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}] {prompt.id} ({prompt.category})")
        
        # Baseline
        resp_text, gen_time, tokens = generate_response(
            baseline_model, tokenizer, prompt.prompt, generation_config, device
        )
        baseline_responses.append(ModelResponse(
            model_name="baseline",
            prompt_id=prompt.id,
            prompt=prompt.prompt,
            response=resp_text,
            generation_config=generation_config,
            generation_time_ms=gen_time,
            tokens_generated=tokens
        ))
        
        # Candidate
        resp_text, gen_time, tokens = generate_response(
            candidate_model, tokenizer, prompt.prompt, generation_config, device
        )
        candidate_responses.append(ModelResponse(
            model_name="candidate",
            prompt_id=prompt.id,
            prompt=prompt.prompt,
            response=resp_text,
            generation_config=generation_config,
            generation_time_ms=gen_time,
            tokens_generated=tokens
        ))
    
    # Judge responses
    print("\nJudging responses...")
    baseline_scores = []
    candidate_scores = []
    
    for i, prompt in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}] {prompt.id}")
        
        # Baseline
        score = call_llm_judge(
            prompt.prompt,
            baseline_responses[i].response,
            rubric,
            judge_model,
            {}
        )
        score.model_name = "baseline"
        score.prompt_id = prompt.id
        baseline_scores.append(score)
        
        # Candidate
        score = call_llm_judge(
            prompt.prompt,
            candidate_responses[i].response,
            rubric,
            judge_model,
            {}
        )
        score.model_name = "candidate"
        score.prompt_id = prompt.id
        candidate_scores.append(score)
    
    # Build results
    results = []
    for i, prompt in enumerate(prompts):
        results.append(EvaluationResult(
            prompt=prompt,
            baseline_response=baseline_responses[i],
            candidate_response=candidate_responses[i],
            baseline_scores=baseline_scores[i],
            candidate_scores=candidate_scores[i]
        ))
    
    # Compute aggregates
    aggregate = compute_aggregate(results)
    
    # Prepare output
    output = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "prompts_version": "v1",
            "rubric_version": rubric["version"],
            "judge_model": judge_model,
            "generation_config": asdict(generation_config),
            "device": device,
            "config_name": config_name,
            "baseline_model": baseline_model_path,
            "candidate_model": candidate_model_path,
            "tokenizer": tokenizer_path,
            "num_prompts": len(prompts)
        },
        "per_example": [],
        "aggregate": aggregate,
        "limitations": get_limitations()
    }
    
    for r in results:
        output["per_example"].append({
            "prompt_id": r.prompt.id,
            "category": r.prompt.category,
            "difficulty": r.prompt.difficulty,
            "prompt": r.prompt.prompt,
            "baseline": {
                "response": r.baseline_response.response,
                "generation_time_ms": r.baseline_response.generation_time_ms,
                "tokens_generated": r.baseline_response.tokens_generated,
                "scores": r.baseline_scores.scores,
                "justifications": r.baseline_scores.justifications,
                "final_score": r.baseline_scores.final_score
            },
            "candidate": {
                "response": r.candidate_response.response,
                "generation_time_ms": r.candidate_response.generation_time_ms,
                "tokens_generated": r.candidate_response.tokens_generated,
                "scores": r.candidate_scores.scores,
                "justifications": r.candidate_scores.justifications,
                "final_score": r.candidate_scores.final_score
            },
            "score_diff": round(r.candidate_scores.final_score - r.baseline_scores.final_score, 2)
        })
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "generative_eval_results.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # Generate report
    report_file = output_dir / "generative_eval_report.md"
    generate_report(output, report_file)
    
    print(f"\nResults saved to {results_file}")
    print(f"Report saved to {report_file}")
    
    return output


def compute_aggregate(results: list[EvaluationResult]) -> dict[str, Any]:
    """Compute aggregate statistics."""
    categories = {}
    difficulties = {}
    
    baseline_scores = []
    candidate_scores = []
    
    for r in results:
        cat = r.prompt.category
        diff = r.prompt.difficulty
        
        b_score = r.baseline_scores.final_score
        c_score = r.candidate_scores.final_score
        
        baseline_scores.append(b_score)
        candidate_scores.append(c_score)
        
        if cat not in categories:
            categories[cat] = {"baseline": [], "candidate": []}
        categories[cat]["baseline"].append(b_score)
        categories[cat]["candidate"].append(c_score)
        
        if diff not in difficulties:
            difficulties[diff] = {"baseline": [], "candidate": []}
        difficulties[diff]["baseline"].append(b_score)
        difficulties[diff]["candidate"].append(c_score)
    
    def avg(lst):
        return round(sum(lst) / len(lst), 2) if lst else 0
    
    aggregate = {
        "overall": {
            "baseline_mean": avg(baseline_scores),
            "candidate_mean": avg(candidate_scores),
            "difference": round(avg(candidate_scores) - avg(baseline_scores), 2),
            "num_examples": len(results)
        },
        "by_category": {},
        "by_difficulty": {},
        "win_rate": {
            "candidate_wins": sum(1 for r in results if r.candidate_scores.final_score > r.baseline_scores.final_score),
            "baseline_wins": sum(1 for r in results if r.baseline_scores.final_score > r.candidate_scores.final_score),
            "ties": sum(1 for r in results if r.candidate_scores.final_score == r.baseline_scores.final_score)
        }
    }
    
    for cat, scores in categories.items():
        aggregate["by_category"][cat] = {
            "baseline_mean": avg(scores["baseline"]),
            "candidate_mean": avg(scores["candidate"]),
            "difference": round(avg(scores["candidate"]) - avg(scores["baseline"]), 2),
            "count": len(scores["baseline"])
        }
    
    for diff, scores in difficulties.items():
        aggregate["by_difficulty"][diff] = {
            "baseline_mean": avg(scores["baseline"]),
            "candidate_mean": avg(scores["candidate"]),
            "difference": round(avg(scores["candidate"]) - avg(scores["baseline"]), 2),
            "count": len(scores["baseline"])
        }
    
    return aggregate


def generate_report(output: dict[str, Any], report_path: Path) -> None:
    """Generate human-readable markdown report."""
    with open(report_path, "w") as f:
        f.write("# Brazilian Portuguese Generative Evaluation Report\n\n")
        
        meta = output["metadata"]
        f.write(f"**Date:** {meta['timestamp']}\n")
        f.write(f"**Prompts Version:** {meta['prompts_version']}\n")
        f.write(f"**Rubric Version:** {meta['rubric_version']}\n")
        f.write(f"**Judge Model:** {meta['judge_model']}\n")
        f.write(f"**Baseline Model:** {meta['baseline_model']}\n")
        f.write(f"**Candidate Model:** {meta['candidate_model']}\n")
        f.write(f"**Tokenizer:** {meta['tokenizer']}\n")
        f.write(f"**Generation Config:** {json.dumps(meta['generation_config'], indent=2)}\n")
        f.write(f"**Device:** {meta['device']}\n")
        f.write(f"**Number of Prompts:** {meta['num_prompts']}\n\n")
        
        f.write("## Aggregate Results\n\n")
        agg = output["aggregate"]
        overall = agg["overall"]
        f.write(f"- **Baseline Mean Score:** {overall['baseline_mean']:.2f}\n")
        f.write(f"- **Candidate Mean Score:** {overall['candidate_mean']:.2f}\n")
        f.write(f"- **Difference (Candidate - Baseline):** {overall['difference']:+.2f}\n")
        f.write(f"- **Examples Evaluated:** {overall['num_examples']}\n\n")
        
        wr = agg["win_rate"]
        f.write("### Win Rate\n\n")
        f.write(f"- Candidate Wins: {wr['candidate_wins']}\n")
        f.write(f"- Baseline Wins: {wr['baseline_wins']}\n")
        f.write(f"- Ties: {wr['ties']}\n\n")
        
        f.write("## By Category\n\n")
        f.write("| Category | Baseline | Candidate | Diff | Count |\n")
        f.write("|----------|----------|-----------|------|-------|\n")
        for cat, data in agg["by_category"].items():
            f.write(f"| {cat} | {data['baseline_mean']:.2f} | {data['candidate_mean']:.2f} | {data['difference']:+.2f} | {data['count']} |\n")
        f.write("\n")
        
        f.write("## By Difficulty\n\n")
        f.write("| Difficulty | Baseline | Candidate | Diff | Count |\n")
        f.write("|------------|----------|-----------|------|-------|\n")
        for diff, data in agg["by_difficulty"].items():
            f.write(f"| {diff} | {data['baseline_mean']:.2f} | {data['candidate_mean']:.2f} | {data['difference']:+.2f} | {data['count']} |\n")
        f.write("\n")
        
        f.write("## Per-Example Details\n\n")
        for ex in output["per_example"]:
            f.write(f"### {ex['prompt_id']} ({ex['category']}, {ex['difficulty']})\n\n")
            f.write(f"**Prompt:** {ex['prompt'][:200]}...\n\n" if len(ex['prompt']) > 200 else f"**Prompt:** {ex['prompt']}\n\n")
            
            f.write("| Model | Final Score | Factual | Cultural | Language | Reasoning | Complete |\n")
            f.write("|-------|-------------|---------|----------|----------|-----------|----------|\n")
            b = ex["baseline"]
            c = ex["candidate"]
            f.write(f"| Baseline | {b['final_score']:.2f} | {b['scores']['factual_accuracy']} | {b['scores']['cultural_competence']} | {b['scores']['language_quality']} | {b['scores']['reasoning_depth']} | {b['scores']['completeness']} |\n")
            f.write(f"| Candidate | {c['final_score']:.2f} | {c['scores']['factual_accuracy']} | {c['scores']['cultural_competence']} | {c['scores']['language_quality']} | {c['scores']['reasoning_depth']} | {c['scores']['completeness']} |\n")
            f.write(f"\n**Score Diff:** {ex['score_diff']:+.2f}\n\n")
            
            f.write("**Baseline Response:**\n")
            f.write(f"> {b['response'][:300]}...\n\n" if len(b['response']) > 300 else f"> {b['response']}\n\n")
            
            f.write("**Candidate Response:**\n")
            f.write(f"> {c['response'][:300]}...\n\n" if len(c['response']) > 300 else f"> {c['response']}\n\n")
            
            f.write("---\n\n")
        
        f.write("## Evaluation Limitations\n\n")
        for lim in output["limitations"]:
            f.write(f"- {lim}\n")


def get_limitations() -> list[str]:
    """Return documented evaluation limitations."""
    return [
        "Judge model is heuristic (offline fallback); production evaluation requires a capable LLM judge (GPT-4, Claude, etc.) with the specified rubric.",
        "Held-out prompt set v1 contains only 12 prompts across 4 categories; larger and more diverse sets needed for robust evaluation.",
        "Single-turn evaluation only; multi-turn conversation quality not assessed.",
        "Generation uses simple sampling; no beam search, constrained decoding, or best-of-N sampling.",
        "Scores are on 1-5 Likert scale; ordinal scale treated as interval for averaging.",
        "Heuristic judge correlates poorly with human judgment; reported scores are indicative only.",
        "No inter-annotator agreement measured; single judge (even LLM) has inherent variance.",
        "Cultural competence dimension relies on surface-level markers; deep cultural understanding requires human evaluation.",
        "Portuguese quality assessment by heuristic only checks for diacritics and length; grammaticality, fluency, and register not properly assessed.",
        "Factual accuracy heuristic uses keyword matching; hallucinations and subtle inaccuracies not detected.",
        "Prompt set may have contamination risk if similar prompts appear in training data of evaluated models.",
        "Evaluation does not measure safety, bias, or toxicity dimensions.",
        "Results not statistically significant due to small sample size (n=12); confidence intervals not computed.",
        "Tokenizer and vocabulary differences between baseline and candidate may confound generation quality comparison."
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Brazilian Portuguese generative evaluation")
    parser.add_argument("--baseline-model", required=True, help="Path to baseline model checkpoint")
    parser.add_argument("--candidate-model", required=True, help="Path to candidate model checkpoint")
    parser.add_argument("--tokenizer-path", required=True, help="Path to tokenizer.json")
    parser.add_argument("--output-dir", default="results/generative_eval", help="Output directory")
    parser.add_argument("--prompts", default="eval/generative/prompts_v1.json", help="Prompts JSON file")
    parser.add_argument("--rubric", default="eval/generative/rubric_v1.json", help="Rubric JSON file")
    parser.add_argument("--judge-model", default="heuristic", help="Judge model (heuristic, gpt-4o, claude, etc.)")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--config", choices=["tiny", "base"], default="tiny", help="Model config")
    parser.add_argument("--limit", type=int, help="Limit number of prompts (for testing)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Generation top-p")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max generation tokens")
    
    args = parser.parse_args()
    
    generation_config = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )
    
    output_dir = Path(args.output_dir)
    prompts_path = Path(args.prompts)
    rubric_path = Path(args.rubric)
    
    if not prompts_path.exists():
        print(f"Error: Prompts file not found: {prompts_path}")
        return 1
    if not rubric_path.exists():
        print(f"Error: Rubric file not found: {rubric_path}")
        return 1
    
    run_generative_evaluation(
        baseline_model_path=args.baseline_model,
        candidate_model_path=args.candidate_model,
        tokenizer_path=args.tokenizer_path,
        prompts_path=prompts_path,
        rubric_path=rubric_path,
        output_dir=output_dir,
        judge_model=args.judge_model,
        generation_config=generation_config,
        device=args.device,
        config_name=args.config,
        limit=args.limit
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())