"""Model card template and rendering for Cafezin LM Brazilian Portuguese specialization.

Generates a complete Hugging Face model card from evaluation results,
training configuration, and provenance metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class ParentCheckpoint:
    """Parent checkpoint provenance."""

    repo: str = "maritaca-ai/S5-nano"
    revision: str = "fe3a3581c33cb22917770ef1987b7c6e30929425"
    model_type: str = "Qwen3_5ForConditionalGeneration"
    parameters: int = 4_660_000_000
    vocab_size: int = 248320
    license: str = "Apache-2.0"


@dataclass
class DataProvenance:
    """Training data provenance and curriculum."""

    sources: list[dict[str, Any]] = field(default_factory=list)
    curriculum: str = ""
    pt_multilingual_schedule: str = "50/50 → 90/10 by token count (PT/multilingual)"
    total_tokens: int = 0
    tokenized_with: str = ""
    decontamination: str = "Not performed (continued pretraining from established checkpoint)"


@dataclass
class TokenizerChanges:
    """Tokenizer extension details."""

    original_vocab_size: int = 248320
    extended_vocab_size: int = 248320
    new_tokens_added: int = 0
    mining_corpus: str = ""
    heldout_corpus: str = ""
    tokenization_gain_pct: float = 0.0
    initialization_method: str = "mean_decomposition"
    preserves_original_ids: bool = True
    preserves_tied_lm_head: bool = True
    new_tokens_sample: list[str] = field(default_factory=list)


@dataclass
class TrainingSettings:
    """Training hyperparameters and compute."""

    seed: int = 42
    seq_len: int = 2048
    batch_size: int = 1
    grad_accum: int = 32
    steps: int = 100_000
    eval_every: int = 1_000
    eval_batches: int = 32
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 2_000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    gradient_checkpointing: bool = True
    mtp_weight: float = 0.3
    dtype: str = "bfloat16"
    device: str = "cuda"
    peak_memory_gb: float | None = None
    throughput_tokens_per_sec: float | None = None
    total_tokens_seen: int = 0
    training_time_hours: float | None = None
    gpu_type: str = "NVIDIA H100 48GB (or equivalent)"


@dataclass
class EvaluationResults:
    """Aggregated evaluation results."""

    # Language modeling
    baseline_bits_per_byte: float | None = None
    candidate_bits_per_byte: float | None = None
    bits_per_byte_diff: float | None = None

    # Portuguese downstream (lm-eval)
    pt_downstream: dict[str, dict[str, float]] = field(default_factory=dict)

    # Multilingual retention
    multilingual: dict[str, dict[str, float]] = field(default_factory=dict)

    # Generative evaluation (Brazilian Portuguese)
    generative: dict[str, Any] = field(default_factory=dict)

    # Vision norm comparison
    vision_norm: dict[str, Any] = field(default_factory=dict)


@dataclass
class Limitations:
    """Documented limitations."""

    items: list[str] = field(default_factory=list)


@dataclass
class ModelCardData:
    """Complete data for model card rendering."""

    model_name: str = "cafezin-lm-br"
    model_version: str = "0.1.0"
    date: str = field(default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%d"))
    parent: ParentCheckpoint = field(default_factory=ParentCheckpoint)
    data_provenance: DataProvenance = field(default_factory=DataProvenance)
    tokenizer_changes: TokenizerChanges = field(default_factory=TokenizerChanges)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    evaluation: EvaluationResults = field(default_factory=EvaluationResults)
    limitations: Limitations = field(default_factory=Limitations)
    candidate_checkpoint_path: str = ""
    extended_tokenizer_path: str = ""
    run_settings_path: str = ""


MODEL_CARD_TEMPLATE = Path(__file__).with_name("model_card.md.j2").read_text(encoding="utf-8")


def render_model_card(data: ModelCardData) -> str:
    """Render model card from data using Jinja2-style template."""
    # Simple template rendering without Jinja2 dependency
    template = MODEL_CARD_TEMPLATE

    # Build context dict
    ctx = {
        "model_version": data.model_version,
        "date": data.date,
        "parent": data.parent,
        "data_provenance": data.data_provenance,
        "tokenizer_changes": data.tokenizer_changes,
        "training": data.training,
        "evaluation": data.evaluation,
        "limitations": data.limitations,
    }

    # Simple variable substitution for basic fields
    import re

    def replace_var(match):
        var_path = match.group(1).strip()
        try:
            # Navigate nested attributes
            parts = var_path.split(".")
            val = ctx
            for part in parts:
                # Handle array indexing like [0]
                if "[" in part and "]" in part:
                    key = part[: part.index("[")]
                    idx_str = part[part.index("[") + 1 : part.index("]")]
                    # Only treat as index if it's a valid integer
                    try:
                        idx = int(idx_str)
                        val = val[key][idx]
                    except ValueError:
                        # It's a format specifier like :30, not an index - skip
                        val = getattr(val, part) if hasattr(val, part) else val.get(part, "")
                else:
                    val = getattr(val, part) if hasattr(val, part) else val.get(part, "")
            if val is None:
                return "N/A"
            if isinstance(val, (list, dict)):
                return json.dumps(val, ensure_ascii=False)
            if isinstance(val, float):
                return f"{val:.6f}"
            return str(val)
        except (AttributeError, KeyError, IndexError, TypeError):
            return f"{{{{{var_path}}}}}"

    # Replace {{ var }} patterns (but not {var:format} format specifiers)
    # Use a more precise pattern that doesn't match format specifiers
    template = re.sub(r"\{\{\s*([^}]+?)\s*\}\}", replace_var, template)

    # Handle conditionals {% if ... %} ... {% endif %} - simple approach
    # This is a minimal implementation; for production use Jinja2

    return template


def load_evaluation_results(
    run_eval_output_dir: str | Path,
    generative_output_dir: str | Path | None = None,
    vision_norm_output_dir: str | Path | None = None,
    continued_pretrain_run_settings: str | Path | None = None,
    tokenizer_extension_output: str | Path | None = None,
    data_provenance_file: str | Path | None = None,
) -> ModelCardData:
    """Load all evaluation results and build ModelCardData."""
    run_eval_dir = Path(run_eval_output_dir)
    data = ModelCardData()

    # Load run settings from continued pretraining
    if continued_pretrain_run_settings and Path(continued_pretrain_run_settings).exists():
        with open(continued_pretrain_run_settings) as f:
            run_settings = json.load(f)

        cp_cfg = run_settings.get("continued_pretrain_config", {})
        data.training = TrainingSettings(
            seed=cp_cfg.get("seed", 42),
            seq_len=cp_cfg.get("seq_len", 2048),
            batch_size=cp_cfg.get("batch_size", 1),
            grad_accum=cp_cfg.get("grad_accum", 32),
            steps=cp_cfg.get("steps", 100_000),
            eval_every=cp_cfg.get("eval_every", 1_000),
            eval_batches=cp_cfg.get("eval_batches", 32),
            lr=cp_cfg.get("lr", 3e-4),
            min_lr=cp_cfg.get("min_lr", 3e-5),
            warmup_steps=cp_cfg.get("warmup_steps", 2_000),
            weight_decay=cp_cfg.get("weight_decay", 0.1),
            beta1=cp_cfg.get("beta1", 0.9),
            beta2=cp_cfg.get("beta2", 0.95),
            grad_clip=cp_cfg.get("grad_clip", 1.0),
            gradient_checkpointing=cp_cfg.get("gradient_checkpointing", True),
            mtp_weight=cp_cfg.get("mtp_weight", 0.3),
            dtype=cp_cfg.get("dtype", "bfloat16"),
            device=cp_cfg.get("device", "cuda"),
            peak_memory_gb=run_settings.get("peak_memory_gb"),
            throughput_tokens_per_sec=run_settings.get("avg_throughput_tokens_per_sec"),
            total_tokens_seen=run_settings.get("total_tokens_seen", 0),
            training_time_hours=run_settings.get("training_time_hours"),
        )
        data.candidate_checkpoint_path = run_settings.get("final_checkpoint", "")
        data.run_settings_path = str(continued_pretrain_run_settings)

    # Load evaluation summary
    eval_summary_path = run_eval_dir / "evaluation_summary.json"
    if eval_summary_path.exists():
        with open(eval_summary_path) as f:
            _ = json.load(f)

    # Load comparison report
    comparison_dir = run_eval_dir / "comparison"
    _ = comparison_dir / "comparison_report.md"
    comparison_json_path = comparison_dir / "comparison.json"

    if comparison_json_path.exists():
        with open(comparison_json_path) as f:
            comparison = json.load(f)

        # Extract bits per byte
        if "perplexity" in comparison:
            perp = comparison["perplexity"]
            data.evaluation.baseline_bits_per_byte = perp.get("baseline", {}).get("bits_per_byte")
            data.evaluation.candidate_bits_per_byte = perp.get("candidate", {}).get("bits_per_byte")
            if data.evaluation.baseline_bits_per_byte and data.evaluation.candidate_bits_per_byte:
                data.evaluation.bits_per_byte_diff = (
                    data.evaluation.candidate_bits_per_byte - data.evaluation.baseline_bits_per_byte
                )

        # Extract downstream task results
        if "tasks" in comparison:
            for task_name, task_data in comparison["tasks"].items():
                baseline = task_data.get("baseline", {})
                candidate = task_data.get("candidate", {})
                if (
                    "pt" in task_name.lower()
                    or "portuguese" in task_name.lower()
                    or task_name
                    in [
                        "assin_paraphrase",
                        "assin_entailment",
                        "assin2_rte",
                        "assin2_sts",
                        "faquad_nli",
                        "xnli",
                        "belebele_por_Latn",
                        "flores_pt-en",
                        "flores_en-pt",
                        "flores_pt-es",
                        "flores_es-pt",
                        "flores_pt-fr",
                        "flores_fr-pt",
                        "flores_pt-de",
                        "flores_de-pt",
                        "flores_pt-it",
                        "flores_it-pt",
                        "wikitext_pt",
                    ]
                ):
                    data.evaluation.pt_downstream[task_name] = {
                        "baseline": baseline.get("accuracy", baseline.get("score", 0)),
                        "candidate": candidate.get("accuracy", candidate.get("score", 0)),
                    }
                else:
                    data.evaluation.multilingual[task_name] = {
                        "baseline": baseline.get("accuracy", baseline.get("score", 0)),
                        "candidate": candidate.get("accuracy", candidate.get("score", 0)),
                    }

    # Load generative evaluation
    if generative_output_dir and Path(generative_output_dir).exists():
        gen_results_path = Path(generative_output_dir) / "generative_eval_results.json"
        if gen_results_path.exists():
            with open(gen_results_path) as f:
                gen_data = json.load(f)
            data.evaluation.generative = gen_data

    # Load vision norm test results
    if vision_norm_output_dir and Path(vision_norm_output_dir).exists():
        vision_meta_path = Path(vision_norm_output_dir) / "metadata.json"
        if vision_meta_path.exists():
            with open(vision_meta_path) as f:
                vision_data = json.load(f)
            data.evaluation.vision_norm = vision_data

    # Load tokenizer extension info
    if tokenizer_extension_output and Path(tokenizer_extension_output).exists():
        with open(tokenizer_extension_output) as f:
            tok_data = json.load(f)
        data.tokenizer_changes = TokenizerChanges(
            original_vocab_size=tok_data.get("original_vocab_size", 248320),
            extended_vocab_size=tok_data.get("extended_vocab_size", 248320),
            new_tokens_added=tok_data.get("new_tokens_added", 0),
            mining_corpus=tok_data.get("mining_corpus", ""),
            heldout_corpus=tok_data.get("heldout_corpus", ""),
            tokenization_gain_pct=tok_data.get("pct_reduction", 0.0),
            initialization_method="mean_decomposition",
            preserves_original_ids=True,
            preserves_tied_lm_head=True,
            new_tokens_sample=tok_data.get("new_tokens_sample", []),
        )
        data.extended_tokenizer_path = tokenizer_extension_output

    # Load data provenance
    if data_provenance_file and Path(data_provenance_file).exists():
        with open(data_provenance_file) as f:
            prov_data = json.load(f)
        data.data_provenance = DataProvenance(
            sources=prov_data.get("sources", []),
            curriculum=prov_data.get("curriculum", ""),
            pt_multilingual_schedule=prov_data.get(
                "pt_multilingual_schedule", "50/50 → 90/10 by token count"
            ),
            total_tokens=prov_data.get("total_tokens", 0),
            tokenized_with=prov_data.get("tokenized_with", ""),
            decontamination=prov_data.get("decontamination", "Not performed"),
        )

    # Set default limitations if not loaded
    if not data.limitations.items:
        data.limitations.items = [
            "Continued pretraining only — no instruction tuning or RLHF alignment performed.",
            "Vision tower not retrained; vision-language alignment may be degraded for "
            "Portuguese inputs.",
            "Multilingual retention shows regressions on non-Portuguese languages (see "
            "evaluation results).",
            "Generative evaluation uses heuristic judge for offline reproducibility; "
            "production evaluation requires capable LLM judge.",
            "Vision normalization ambiguity (zero_centered_norm) unresolved; multimodal "
            "outputs experimental.",
            "No safety, bias, or toxicity evaluation performed.",
            "Portuguese corpus composition and decontamination not independently audited.",
            "Single GPU training (48GB); results may not scale to larger compute budgets.",
            "Tokenizer extension based on frequency mining; may miss rare but "
            "semantically important tokens.",
            "Evaluation sample sizes limited (12 generative prompts, lm-eval default "
            "splits); confidence intervals not computed.",
        ]

    return data


def main() -> int:
    """CLI entry point for model card rendering."""
    import argparse

    parser = argparse.ArgumentParser(description="Render Cafezin LM model card")
    parser.add_argument(
        "--run-eval-dir", required=True, help="Output directory from scripts/run_eval.py"
    )
    parser.add_argument(
        "--generative-dir", help="Output directory from scripts/eval_generative_br.py"
    )
    parser.add_argument(
        "--vision-norm-dir", help="Output directory from scripts/test_vision_norm.py"
    )
    parser.add_argument("--run-settings", help="Path to run_settings.json from continued_pretrain")
    parser.add_argument("--tokenizer-extension", help="Path to tokenizer extension results JSON")
    parser.add_argument("--data-provenance", help="Path to data provenance JSON")
    parser.add_argument("--model-name", default="cafezin-lm-br", help="Model name for card")
    parser.add_argument("--model-version", default="0.1.0", help="Model version")
    parser.add_argument("--output", default="MODEL_CARD.md", help="Output model card path")

    args = parser.parse_args()

    data = load_evaluation_results(
        run_eval_output_dir=args.run_eval_dir,
        generative_output_dir=args.generative_dir,
        vision_norm_output_dir=args.vision_norm_dir,
        continued_pretrain_run_settings=args.run_settings,
        tokenizer_extension_output=args.tokenizer_extension,
        data_provenance_file=args.data_provenance,
    )
    data.model_name = args.model_name
    data.model_version = args.model_version

    card = render_model_card(data)

    output_path = Path(args.output)
    output_path.write_text(card, encoding="utf-8")
    print(f"Model card written to {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
