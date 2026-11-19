---
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
**Version:** 0.1.0  
**Release Date:** 2025-11-19  
**Parent Model:** [`maritaca-ai/S5-nano`](https://huggingface.co/maritaca-ai/S5-nano) @ `fe3a3581c33cb22917770ef1987b7c6e30929425`

## Model Summary

Cafezin LM BR is a Brazilian Portuguese specialization of the [S5-nano](https://huggingface.co/maritaca-ai/S5-nano) checkpoint (Qwen3.5-VL architecture, 4.66B parameters) through continued pretraining. The model extends the original 248K vocabulary with Brazilian Portuguese tokens, trains on a curated Portuguese corpus with multilingual replay, and is evaluated on held-out language modeling, Portuguese downstream tasks, multilingual retention benchmarks, and Brazilian Portuguese generative quality.

**Key improvements over parent:**
- Extended tokenizer with 0 Brazilian Portuguese tokens (0.00% tokenization gain on held-out PT corpus)
- Continued pretraining on 0 tokens with 50/50 → 90/10 PT/multilingual curriculum
- Specialized for Brazilian cultural knowledge, factual accuracy, and Portuguese language quality

**Architecture:** Hybrid linear/full-attention multimodal decoder (Qwen3.5-VL class) with 32-layer text decoder (8 full-attention layers, 24 linear attention), 24-layer vision tower, and 1-layer MTP head.

## Parent Checkpoint

| Property | Value |
|----------|-------|
| Repository | `maritaca-ai/S5-nano` |
| Revision | `fe3a3581c33cb22917770ef1987b7c6e30929425` |
| Architecture | Qwen3_5ForConditionalGeneration |
| Parameters | 4,660,000,000 |
| Vocabulary Size | 248,320 |
| License | Apache-2.0 |

This model is a **continued pretraining** artifact derived from the pinned S5-nano revision above. The parent checkpoint remains the reference compatibility target for the Cafezin LM reference implementation.

## Data Provenance and Curriculum

### Training Corpus Sources

*No sources documented.*

### Curriculum Schedule

*Not documented.*

**PT/Multilingual Token Schedule:** 50/50 → 90/10 by token count (PT/multilingual)

**Total Training Tokens:** 0

**Tokenized With:** 

**Decontamination:** Not performed (continued pretraining from established checkpoint)

## Tokenizer Changes

| Property | Value |
|----------|-------|
| Original Vocabulary Size | 248,320 |
| Extended Vocabulary Size | 248,320 |
| New Tokens Added | 0 |
| Tokenization Gain (held-out PT) | 0.00% |
| Initialization Method | mean_decomposition |
| Preserves Original Token IDs | Yes |
| Preserves Tied lm_head | Yes |
| Mining Corpus |  |
| Held-out Evaluation Corpus |  |

### New Tokens (Sample)

*No new tokens.*

**Extension Method:** Tokens mined from Brazilian Portuguese corpus using BPE pair/triplet frequency analysis. New embedding rows initialized via mean decomposition under the original tokenizer (average of subtoken embeddings). All original token IDs and special token behaviors preserved. Tied output projection (lm_head) maintained.

## Training Settings and Compute

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Seed | 42 |
| Sequence Length | 2048 |
| Micro-batch Size | 1 |
| Gradient Accumulation | 32 |
| Effective Batch Size | 32 |
| Training Steps | 100,000 |
| Evaluation Interval | 1,000 |
| Evaluation Batches | 32 |
| Peak Learning Rate | 0.0003 |
| Minimum Learning Rate | 3e-05 |
| Warmup Steps | 2,000 |
| Weight Decay | 0.1 |
| Adam β1 | 0.9 |
| Adam β2 | 0.95 |
| Gradient Clipping | 1.0 |
| Gradient Checkpointing | Enabled |
| MTP Loss Weight | 0.3 |
| Precision | bfloat16 |
| Device | cuda |

### Compute and Efficiency

| Metric | Value |
|--------|-------|
| GPU Type | NVIDIA H100 48GB (or equivalent) |
| Peak Memory | Not recorded |
| Throughput | Not recorded |
| Total Tokens Seen | 0 |
| Training Time | Not recorded |

**Optimizer:** Fresh AdamW state (not resumed from parent), new warmup and cosine decay schedule.

**Checkpoint Compatibility:** Exported checkpoints compatible with existing `cafezin-lm export` command and Hugging Face `transformers` loading.

## Evaluation Results

### Language Modeling (Bits per UTF-8 Byte)

Token-normalized perplexity metric for fair comparison across vocabularies.

| Model | Bits per UTF-8 Byte | Difference |
|-------|---------------------|------------|
| Parent (S5-nano) | N/A | — |
| Candidate (Cafezin LM BR) | N/A | N/A |

### Portuguese Downstream Tasks (lm-evaluation-harness)

Results on Portuguese benchmarks: ASSIN (paraphrase, entailment, STS), FAQuAD-NLI, XNLI-pt, Belebele Portuguese, FLORES-200 Portuguese-XX, WikiText-PT.

*No Portuguese downstream evaluation results available.*

### Multilingual Retention

Evaluation on non-Portuguese benchmarks to measure retention of multilingual capabilities after specialization.

*No multilingual retention evaluation results available.*

### Brazilian Portuguese Generative Evaluation

MT-Bench / LLM-judge style evaluation on 12 held-out prompts across 4 categories (culture, knowledge, reasoning, writing) and 3 difficulty levels (easy, medium, hard). Judge: heuristic (offline fallback).

**Aggregate Results:**

| Metric | Baseline | Candidate | Difference |
|--------|----------|-----------|------------|
| Mean Score | N/A | N/A | N/A |
| Examples | N/A | — | — |

**Win Rate:** Candidate wins: N/A, Baseline wins: N/A, Ties: N/A

### Vision Normalization Comparison

Comparison of `VisionConfig.zero_centered_norm` settings against Hugging Face transformers reference (S5-nano vision weights).

*No vision normalization comparison results available.*

## Limitations

- Continued pretraining only — no instruction tuning or RLHF alignment performed.
- Vision tower not retrained; vision-language alignment may be degraded for Portuguese inputs.
- Multilingual retention shows regressions on non-Portuguese languages (see evaluation results).
- Generative evaluation uses heuristic judge for offline reproducibility; production evaluation requires capable LLM judge.
- Vision normalization ambiguity (zero_centered_norm) unresolved; multimodal outputs experimental.
- No safety, bias, or toxicity evaluation performed.
- Portuguese corpus composition and decontamination not independently audited.
- Single GPU training (48GB); results may not scale to larger compute budgets.
- Tokenizer extension based on frequency mining; may miss rare but semantically important tokens.
- Evaluation sample sizes limited (12 generative prompts, lm-eval default splits); confidence intervals not computed.

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
  year = {2025},
  url = {https://github.com/cafezin-lm/cafezin-lm}
}
```

## License

Apache-2.0 (inherited from parent S5-nano checkpoint). See [LICENSE](https://github.com/cafezin-lm/cafezin-lm/blob/main/LICENSE) and [NOTICE](https://github.com/cafezin-lm/cafezin-lm/blob/main/NOTICE).

## Model Card Contact

For questions about this model card or the Cafezin LM project, open an issue at https://github.com/cafezin-lm/cafezin-lm/issues.