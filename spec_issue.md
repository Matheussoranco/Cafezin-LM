## Summary
Spec review against SPEC.md reveals several gaps between requirements and implementation.

## Missing/Partial Requirements

### AS-001: Vision normalization ambiguity unresolved (Known Gap)
- **Spec line 48**: "record which setting matches within 1e-3 max absolute difference"
- **Implementation**: `scripts/test_vision_norm.py` runs and finds `zero_centered_norm=False` winner with max_diff=0.625 vs HF reference
- **Status**: NOT within 1e-3 precision. Explicitly noted as UNRESOLVED in SPEC.md line 66.
- **Impact**: Cannot claim vision parity with S5-nano.

### DS-002: Tokenizer extension not validated on real Brazilian corpus
- **Spec line 40**: "measure the vocabulary gap between S5-nano's 248K tokenizer and Portuguese corpora, add only high-value tokens"
- **Implementation**: `tokenizer_extension.py` has `mine_candidate_tokens` and `measure_tokenization_stats` but they haven't been run on actual Brazilian corpora (mc4-pt, OSCAR-pt, etc.)
- **Gap**: Tokenizer extension infrastructure exists but not validated on real data.

### MS-001: Continued pretraining not run on 48GB GPU
- **Spec line 44**: "train with `seq_len=2048` baseline / `4096` stretch on single 48 GB GPU"
- **Implementation**: CLI `continue-pretrain` exists with correct defaults, but only smoke test on CPU (`--smoke-test`) has been run.
- **Gap**: No actual training run on 48GB GPU to validate memory/throughput.

### AS-002: Generative evaluation uses heuristic judge only
- **Spec line 52**: "add a generative eval (MT-Bench style or LLM-judge)"
- **Implementation**: `eval_generative_br.py` with `heuristic_judge` fallback. No LLM judge (GPT-4, Claude) integration.
- **Limitation**: Noted in model card limitations.

### AS-003: Model card has placeholder values
- **Spec line 56**: "render a model card from template recording parent revision, data provenance/curriculum, tokenizer changes, hyperparameters, compute, evaluation results, and limitations"
- **Implementation**: `model_card.py` template exists but renders with placeholder values since evaluations haven't been run.

## Scope Creep (Not in Spec)
- `bandit_report.json` (3866 lines)
- `uv.lock` (3021 lines)
- `test_model_card.md` (255 lines)
- `tickets.json` (internal tracking)
- `AGENTS.md`, `docs/agents/`, `docs/cafezin-open-weight-spec.md` (SAIL skills setup)
- `scripts/__init__.py`, `corpus.txt`, `test_vision_norm_outputs/metadata.json`