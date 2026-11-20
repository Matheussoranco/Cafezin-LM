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


MODEL_CARD_TEMPLATE = """---
license: apache-2.0
tags:
- portuguese
- brazilian-portuguese
- continued-pretraining
- multimodal
- qwen3.5-vl
- cafezin-lm
language:
- pt
- pt-BR
- en
- es
- fr
- de
- it
datasets:
- mc4-pt
- oscar-pt
- portuguese-wikipedia
- brazilian-common-crawl
metrics:
- bits_per_byte
- accuracy
- f1
- generative_quality
model-index:
- name: Cafezin LM BR
  results: []
---

# Cafezin LM BR: Brazilian Portuguese Specialized Multimodal Model

**Model ID:** `cafezin-lm/cafezin-lm-br`
**Version:** {{ model_version }}
**Release Date:** {{ date }}
**Parent Model:** [`{{ parent.repo }}`](https://huggingface.co/{{ parent.repo }}) @ `{{ parent.revision }}`

## Model Summary

Cafezin LM BR is a Brazilian Portuguese specialization of the [S5-nano](https://huggingface.co/maritaca-ai/S5-nano) checkpoint (Qwen3.5-VL architecture, 4.66B parameters) through continued pretraining. The model extends the original 248K vocabulary with Brazilian Portuguese tokens, trains on a curated Portuguese corpus with multilingual replay, and is evaluated on held-out language modeling, Portuguese downstream tasks, multilingual retention benchmarks, and Brazilian Portuguese generative quality.

**Key improvements over parent:**
- Extended tokenizer with {{ tokenizer_changes.new_tokens_added }} Brazilian Portuguese tokens ({{ tokenizer_changes.tokenization_gain_pct|round(2) }}% tokenization gain on held-out PT corpus)
- Continued pretraining on {{ "{:,}".format(training.total_tokens_seen) }} tokens with 50/50 → 90/10 PT/multilingual curriculum
- Specialized for Brazilian cultural knowledge, factual accuracy, and Portuguese language quality

**Architecture:** Hybrid linear/full-attention multimodal decoder (Qwen3.5-VL class) with 32-layer text decoder (8 full-attention layers, 24 linear attention), 24-layer vision tower, and 1-layer MTP head.

## Parent Checkpoint

| Property | Value |
|----------|-------|
| Repository | `{{ parent.repo }}` |
| Revision | `{{ parent.revision }}` |
| Architecture | {{ parent.model_type }} |
| Parameters | {{ "{:,}".format(parent.parameters) }} |
| Vocabulary Size | {{ "{:,}".format(parent.vocab_size) }} |
| License | {{ parent.license }} |

This model is a **continued pretraining** artifact derived from the pinned S5-nano revision above. The parent checkpoint remains the reference compatibility target for the Cafezin LM reference implementation.

## Data Provenance and Curriculum

### Training Corpus Sources

{% for source in data_provenance.sources %}
- **{{ source.name }}**: {{ source.description }} ({{ source.language }}, {{ "{:,}".format(source.tokens) }} tokens, revision: {{ source.revision }})
{% endfor %}

### Curriculum Schedule

{{ data_provenance.curriculum }}

**PT/Multilingual Token Schedule:** {{ data_provenance.pt_multilingual_schedule }}

**Total Training Tokens:** {{ "{:,}".format(data_provenance.total_tokens) }}

**Tokenized With:** {{ data_provenance.tokenized_with }}

**Decontamination:** {{ data_provenance.decontamination }}

## Tokenizer Changes

| Property | Value |
|----------|-------|
| Original Vocabulary Size | {{ "{:,}".format(tokenizer_changes.original_vocab_size) }} |
| Extended Vocabulary Size | {{ "{:,}".format(tokenizer_changes.extended_vocab_size) }} |
| New Tokens Added | {{ tokenizer_changes.new_tokens_added }} |
| Tokenization Gain (held-out PT) | {{ tokenizer_changes.tokenization_gain_pct|round(2) }}% |
| Initialization Method | {{ tokenizer_changes.initialization_method }} |
| Preserves Original Token IDs | {{ "Yes" if tokenizer_changes.preserves_original_ids else "No" }} |
| Preserves Tied lm_head | {{ "Yes" if tokenizer_changes.preserves_tied_lm_head else "No" }} |
| Mining Corpus | {{ tokenizer_changes.mining_corpus }} |
| Held-out Evaluation Corpus | {{ tokenizer_changes.heldout_corpus }} |

### New Tokens (Sample)

{{ tokenizer_changes.new_tokens_sample[:30]|join(", ") }}{% if tokenizer_changes.new_tokens_sample|length > 30 %} ... ({{ tokenizer_changes.new_tokens_sample|length }} total){% endif %}

**Extension Method:** Tokens mined from Brazilian Portuguese corpus using BPE pair/triplet frequency analysis. New embedding rows initialized via mean decomposition under the original tokenizer (average of subtoken embeddings). All original token IDs and special token behaviors preserved. Tied output projection (lm_head) maintained.

## Training Settings and Compute

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Seed | {{ training.seed }} |
| Sequence Length | {{ training.seq_len }} |
| Micro-batch Size | {{ training.batch_size }} |
| Gradient Accumulation | {{ training.grad_accum }} |
| Effective Batch Size | {{ training.batch_size * training.grad_accum }} |
| Training Steps | {{ "{:,}".format(training.steps) }} |
| Evaluation Interval | {{ "{:,}".format(training.eval_every) }} |
| Evaluation Batches | {{ training.eval_batches }} |
| Peak Learning Rate | {{ training.lr }} |
| Minimum Learning Rate | {{ training.min_lr }} |
| Warmup Steps | {{ "{:,}".format(training.warmup_steps) }} |
| Weight Decay | {{ training.weight_decay }} |
| Adam β1 | {{ training.beta1 }} |
| Adam β2 | {{ training.beta2 }} |
| Gradient Clipping | {{ training.grad_clip }} |
| Gradient Checkpointing | {{ "Enabled" if training.gradient_checkpointing else "Disabled" }} |
| MTP Loss Weight | {{ training.mtp_weight }} |
| Precision | {{ training.dtype }} |
| Device | {{ training.device }} |

### Compute and Efficiency

| Metric | Value |
|--------|-------|
| GPU Type | {{ training.gpu_type }} |
| Peak Memory | {{ "{:.2f} GB".format(training.peak_memory_gb) if training.peak_memory_gb else "Not recorded" }} |
| Throughput | {{ "{:,.0f} tok/s".format(training.throughput_tokens_per_sec) if training.throughput_tokens_per_sec else "Not recorded" }} |
| Total Tokens Seen | {{ "{:,}".format(training.total_tokens_seen) }} |
| Training Time | {{ "{:.1f} hours".format(training.training_time_hours) if training.training_time_hours else "Not recorded" }} |

**Optimizer:** Fresh AdamW state (not resumed from parent), new warmup and cosine decay schedule.

**Checkpoint Compatibility:** Exported checkpoints compatible with existing `cafezin-lm export` command and Hugging Face `transformers` loading.

## Evaluation Results

### Language Modeling (Bits per UTF-8 Byte)

Token-normalized perplexity metric for fair comparison across vocabularies.

| Model | Bits per UTF-8 Byte | Difference |
|-------|---------------------|------------|
| Parent (S5-nano) | {{ evaluation.baseline_bits_per_byte|round(4) if evaluation.baseline_bits_per_byte is not none else "N/A" }} | — |
| Candidate (Cafezin LM BR) | {{ evaluation.candidate_bits_per_byte|round(4) if evaluation.candidate_bits_per_byte is not none else "N/A" }} | {{ evaluation.bits_per_byte_diff|round(4) if evaluation.bits_per_byte_diff is not none else "N/A" }} |

{% if evaluation.bits_per_byte_diff is not none %}
{% if evaluation.bits_per_byte_diff < 0 %}
**Interpretation:** Candidate achieves lower bits/byte (better compression) on held-out Portuguese corpus.
{% else %}
**Interpretation:** Parent achieves lower bits/byte on held-out Portuguese corpus; candidate shows regression of {{ evaluation.bits_per_byte_diff|round(4) }} bits/byte.
{% endif %}
{% endif %}

### Portuguese Downstream Tasks (lm-evaluation-harness)

Results on Portuguese benchmarks: ASSIN (paraphrase, entailment, STS), FAQuAD-NLI, XNLI-pt, Belebele Portuguese, FLORES-200 Portuguese-XX, WikiText-PT.

{% if evaluation.pt_downstream %}
| Task | Baseline | Candidate | Δ (Candidate - Baseline) |
|------|----------|-----------|--------------------------|
{% for task, scores in evaluation.pt_downstream.items() %}
| {{ task }} | {{ scores.baseline|round(4) }} | {{ scores.candidate|round(4) }} | {{ (scores.candidate - scores.baseline)|round(4) }} |
{% endfor %}
{% else %}
*No Portuguese downstream evaluation results available.*
{% endif %}

### Multilingual Retention

Evaluation on non-Portuguese benchmarks to measure retention of multilingual capabilities after specialization.

{% if evaluation.multilingual %}
| Task / Language | Baseline | Candidate | Δ (Candidate - Baseline) |
|-----------------|----------|-----------|--------------------------|
{% for task, scores in evaluation.multilingual.items() %}
| {{ task }} | {{ scores.baseline|round(4) }} | {{ scores.candidate|round(4) }} | {{ (scores.candidate - scores.baseline)|round(4) }} |
{% endfor %}
{% else %}
*No multilingual retention evaluation results available.*
{% endif %}

### Brazilian Portuguese Generative Evaluation

MT-Bench / LLM-judge style evaluation on 12 held-out prompts across 4 categories (culture, knowledge, reasoning, writing) and 3 difficulty levels (easy, medium, hard). Judge: {{ evaluation.generative.judge_model if evaluation.generative.get('judge_model') else "heuristic (offline fallback)" }}.

**Aggregate Results:**

| Metric | Baseline | Candidate | Difference |
|--------|----------|-----------|------------|
| Mean Score | {{ evaluation.generative.aggregate.overall.baseline_mean|round(2) if evaluation.generative.get('aggregate') else "N/A" }} | {{ evaluation.generative.aggregate.overall.candidate_mean|round(2) if evaluation.generative.get('aggregate') else "N/A" }} | {{ evaluation.generative.aggregate.overall.difference|round(2) if evaluation.generative.get('aggregate') else "N/A" }} |
| Examples | {{ evaluation.generative.aggregate.overall.num_examples if evaluation.generative.get('aggregate') else "N/A" }} | — | — |

**Win Rate:** Candidate wins: {{ evaluation.generative.aggregate.win_rate.candidate_wins if evaluation.generative.get('aggregate') else "N/A" }}, Baseline wins: {{ evaluation.generative.aggregate.win_rate.baseline_wins if evaluation.generative.get('aggregate') else "N/A" }}, Ties: {{ evaluation.generative.aggregate.win_rate.ties if evaluation.generative.get('aggregate') else "N/A" }}

**By Category:**

{% if evaluation.generative.get('aggregate', {}).get('by_category') %}
| Category | Baseline | Candidate | Δ | Count |
|----------|----------|-----------|---|-------|
{% for cat, data in evaluation.generative.aggregate.by_category.items() %}
| {{ cat }} | {{ data.baseline_mean|round(2) }} | {{ data.candidate_mean|round(2) }} | {{ data.difference|round(2) }} | {{ data.count }} |
{% endfor %}
{% else %}
*No category breakdown available.*
{% endif %}

**By Difficulty:**

{% if evaluation.generative.get('aggregate', {}).get('by_difficulty') %}
| Difficulty | Baseline | Candidate | Δ | Count |
|------------|----------|-----------|---|-------|
{% for diff, data in evaluation.generative.aggregate.by_difficulty.items() %}
| {{ diff }} | {{ data.baseline_mean|round(2) }} | {{ data.candidate_mean|round(2) }} | {{ data.difference|round(2) }} | {{ data.count }} |
{% endfor %}
{% else %}
*No difficulty breakdown available.*
{% endif %}

**Prompts Version:** {{ evaluation.generative.metadata.prompts_version if evaluation.generative.get('metadata') else "v1" }}
**Rubric Version:** {{ evaluation.generative.metadata.rubric_version if evaluation.generative.get('metadata') else "v1" }}
**Dimensions:** Factual Accuracy (30%), Cultural Competence (25%), Language Quality (20%), Reasoning Depth (15%), Completeness (10%)

### Vision Normalization Comparison

Comparison of `VisionConfig.zero_centered_norm` settings against Hugging Face transformers reference (S5-nano vision weights).

{% if evaluation.vision_norm %}
| Setting | Max Abs Diff vs HF Reference | Match (< 1e-3) |
|---------|------------------------------|----------------|
| `zero_centered_norm=False` | {{ evaluation.vision_norm.max_abs_diff.hf_vs_false|round(6) if evaluation.vision_norm.get('max_abs_diff') else "N/A" }} | {{ "✅" if evaluation.vision_norm.get('winner') == "False" and evaluation.vision_norm.get('pass') else "❌" }} |
| `zero_centered_norm=True` | {{ evaluation.vision_norm.max_abs_diff.hf_vs_true|round(6) if evaluation.vision_norm.get('max_abs_diff') else "N/A" }} | {{ "✅" if evaluation.vision_norm.get('winner') == "True" and evaluation.vision_norm.get('pass') else "❌" }} |

**Result:** `zero_centered_norm={{ evaluation.vision_norm.winner if evaluation.vision_norm.get('winner') else "undetermined" }}` is closer to HF reference (max diff = {{ evaluation.vision_norm.winner_diff|round(6) if evaluation.vision_norm.get('winner_diff') else "N/A" }}).

{% if evaluation.vision_norm.get('pass') == false %}
⚠️ **Neither setting matches HF reference within 1e-3 tolerance.** The ambiguity remains unresolved. The Cafezin LM reference implementation defaults to `zero_centered_norm=False` following the HF port, but this has not been verified to match the published weights to the required precision. Vision/multimodal capabilities should be considered experimental.
{% else %}
✅ **Vision normalization resolved:** The winning setting matches HF reference within 1e-3.
{% endif %}

**Test Metadata:**
- Reference: `{{ evaluation.vision_norm.metadata.reference_repo if evaluation.vision_norm.get('metadata') else "maritaca-ai/S5-nano" }}` @ `{{ evaluation.vision_norm.metadata.reference_revision if evaluation.vision_norm.get('metadata') else "fe3a3581c33cb22917770ef1987b7c6e30929425" }}`
- Transformers Version: {{ evaluation.vision_norm.metadata.transformers_version if evaluation.vision_norm.get('metadata') else "N/A" }}
- Device: {{ evaluation.vision_norm.metadata.device if evaluation.vision_norm.get('metadata') else "N/A" }}
- Dtype: {{ evaluation.vision_norm.metadata.dtype if evaluation.vision_norm.get('metadata') else "N/A" }}
- Input Grid: {{ evaluation.vision_norm.metadata.input_grid_thw if evaluation.vision_norm.get('metadata') else "N/A" }}

**Note:** This comparison tests the vision tower forward pass only. Full multimodal integration (image processor, chat template, vision-language alignment) is not verified.
{% else %}
*No vision normalization comparison results available.*
{% endif %}

## Limitations

{% for lim in limitations.items %}
- {{ lim }}
{% endfor %}

**Additional Notes:**
- This is a **continued pretraining** checkpoint, not an instruction-tuned or RLHF-aligned model.
- The model retains the base S5-nano vision tower; vision-language alignment has not been retrained.
- Portuguese specialization may have introduced regressions on non-Portuguese languages (see multilingual retention results).
- Generative evaluation uses a heuristic judge for offline reproducibility; production evaluation requires a capable LLM judge (GPT-4o, Claude, etc.).
- Vision normalization ambiguity (`zero_centered_norm`) is unresolved; multimodal outputs may differ from HF reference.
- No safety, bias, or toxicity evaluation has been performed.

## Intended Use

- Research on Portuguese language modeling and continued pretraining
- Brazilian Portuguese text generation and understanding tasks
- Multilingual models with Portuguese emphasis
- Reproducibility studies for tokenizer extension and embedding growth

## Out of Scope

- Production deployment without further safety evaluation
- Instruction following / chat without fine-tuning
- High-stakes decision making
- Languages not covered in evaluation (especially low-resource)

## How to Load

```python
from transformers import AutoModelForCausalGeneration, AutoTokenizer

model = AutoModelForCausalGeneration.from_pretrained(
    "cafezin-lm/cafezin-lm-br",
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("cafezin-lm/cafezin-lm-br", trust_remote_code=True)
```

Or with the Cafezin LM CLI:
```bash
cafezin-lm generate --checkpoint /path/to/checkpoint --tokenizer /path/to/tokenizer.json --prompt "Seu prompt aqui"
```

## Citation

```bibtex
@misc{cafezin-lm-br,
  title = {Cafezin LM BR: Brazilian Portuguese Specialization of S5-nano},
  author = {Cafezin LM Contributors},
  year = {{ date[:4] }},
  url = {https://github.com/cafezin-lm/cafezin-lm}
}
```

## License

Apache-2.0 (inherited from parent S5-nano checkpoint). See [LICENSE](https://github.com/cafezin-lm/cafezin-lm/blob/main/LICENSE) and [NOTICE](https://github.com/cafezin-lm/cafezin-lm/blob/main/NOTICE).

## Model Card Contact

For questions about this model card or the Cafezin LM project, open an issue at https://github.com/cafezin-lm/cafezin-lm/issues.
"""

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
                    key = part[:part.index("[")]
                    idx_str = part[part.index("[")+1:part.index("]")]
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
                if "pt" in task_name.lower() or "portuguese" in task_name.lower() or task_name in [
                    "assin_paraphrase", "assin_entailment", "assin2_rte", "assin2_sts",
                    "faquad_nli", "xnli", "belebele_por_Latn", "flores_pt-en", "flores_en-pt",
                    "flores_pt-es", "flores_es-pt", "flores_pt-fr", "flores_fr-pt",
                    "flores_pt-de", "flores_de-pt", "flores_pt-it", "flores_it-pt", "wikitext_pt"
                ]:
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
            pt_multilingual_schedule=prov_data.get("pt_multilingual_schedule", "50/50 → 90/10 by token count"),
            total_tokens=prov_data.get("total_tokens", 0),
            tokenized_with=prov_data.get("tokenized_with", ""),
            decontamination=prov_data.get("decontamination", "Not performed"),
        )

    # Set default limitations if not loaded
    if not data.limitations.items:
        data.limitations.items = [
            "Continued pretraining only — no instruction tuning or RLHF alignment performed.",
            "Vision tower not retrained; vision-language alignment may be degraded for Portuguese inputs.",
            "Multilingual retention shows regressions on non-Portuguese languages (see evaluation results).",
            "Generative evaluation uses heuristic judge for offline reproducibility; production evaluation requires capable LLM judge.",
            "Vision normalization ambiguity (zero_centered_norm) unresolved; multimodal outputs experimental.",
            "No safety, bias, or toxicity evaluation performed.",
            "Portuguese corpus composition and decontamination not independently audited.",
            "Single GPU training (48GB); results may not scale to larger compute budgets.",
            "Tokenizer extension based on frequency mining; may miss rare but semantically important tokens.",
            "Evaluation sample sizes limited (12 generative prompts, lm-eval default splits); confidence intervals not computed.",
        ]

    return data


def main() -> int:
    """CLI entry point for model card rendering."""
    import argparse

    parser = argparse.ArgumentParser(description="Render Cafezin LM model card")
    parser.add_argument("--run-eval-dir", required=True, help="Output directory from scripts/run_eval.py")
    parser.add_argument("--generative-dir", help="Output directory from scripts/eval_generative_br.py")
    parser.add_argument("--vision-norm-dir", help="Output directory from scripts/test_vision_norm.py")
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
