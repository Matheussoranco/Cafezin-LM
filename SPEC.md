## Problem Statement

Cafezin LM is a from-scratch PyTorch implementation of the Qwen3.5-VL architecture (Maritaca AI's S5-nano checkpoint layout). The repo currently provides:
- Exact tensor-metadata compatibility with the pinned S5-nano revision (738 tensors, 9.3 GB)
- Working CLI: `prepare`, `train`, `export`, `verify`, `generate`, `info`
- CPU smoke tests passing; full-size inference with S5-nano weights unverified
- Vision/multimodal experimental; `zero_centered_norm` ambiguity unresolved

The goal is to produce a **Brazilian Portuguese specialized open-weight model** by continuing pretraining from S5-nano, extending the tokenizer for Brazilian Portuguese, evaluating against the parent checkpoint, and publishing a reproducible release — while keeping the pinned reference implementation intact as a compatibility target.

## Solution

Execute the Portuguese specialization track as a distinct Cafezin artifact:
1. **Continued pretraining** from the pinned S5-nano weights (ADR-0002)
2. **Tokenizer extension** preserving existing IDs and embedding rows (ADR-0003)
3. **Evaluation** against S5-nano on Portuguese LM, downstream, multilingual, and generative benchmarks (ADR-0004)
4. **Reproducible release** with model card, configs, and provenance (ADR-0005)
5. **Architecture stability**: no new experiments; stabilize current hybrid linear/full-attention implementation first (ADR-0001)

## Stories

### US-001 — Train a Portuguese-specialized model
**Type:** User Story  
As a researcher, I want to continue pretraining S5-nano on Brazilian Portuguese data so that the resulting model is specialized for Portuguese while retaining multilingual capabilities.

### US-002 — Evaluate the specialized model against S5-nano
**Type:** User Story  
As a researcher, I want to compare the Portuguese-specialized model against the pinned S5-nano checkpoint on held-out language modeling, downstream tasks, multilingual benchmarks, and Brazilian Portuguese generative evaluation so that specialization gains and regressions are quantified.

### US-003 — Publish a reproducible open-weight release
**Type:** User Story  
As a researcher, I want to publish the specialized model weights, tokenizer, configs, and a complete model card on Hugging Face so that the community can reproduce, audit, and use the model.

### DS-001 — Curate Brazilian Portuguese training corpus
**Type:** Data Story  
As a data engineer, I need to assemble a Portuguese corpus (mc4-pt, OSCAR-pt, Brazilian Common Crawl, Portuguese Wikipedia) with documented provenance, apply a 50/50 → 90/10 PT/multilingual curriculum by token count, and tokenize with the extended S5-nano tokenizer.

### DS-002 — Extend S5-nano tokenizer for Brazilian Portuguese
**Type:** Data Story  
As a data engineer, I need to measure the vocabulary gap between S5-nano's 248K tokenizer and Portuguese corpora, add only high-value tokens, initialize new embedding rows from the mean of their decomposition under the original tokenizer, and export the extended tokenizer with matching weights.

### MS-001 — Implement continued pretraining pipeline
**Type:** Model Story  
As an ML engineer, I need to load S5-nano weights into the base preset, grow the embedding table for new tokens, initialize a fresh Adam optimizer with peak LR (3e-4) and new warmup (2k steps), train with `seq_len=2048` baseline / `4096` stretch on single 48 GB GPU using gradient accumulation=32, gradient checkpointing, and MTP weight=0.3, and save checkpoints compatible with the existing `export` command.

### AS-001 — Resolve vision normalization ambiguity
**Type:** Agent Story  
As an agent, I need to complete the `zero_centered_norm` comparison by installing transformers from git, running the test script against HF reference on the same inputs/weights, and recording which setting matches within 1e-3 max absolute difference.

### AS-002 — Build eval harness on lm-evaluation-harness
**Type:** Agent Story  
As an agent, I need to add `lm-eval` to dev deps, configure Portuguese tasks (FAQUAD, ASSIN, XNLI-pt, XQuAD-pt, Flores-200 pt-X), add a generative eval (MT-Bench style or LLM-judge), report bits-per-UTF-8-byte for tokenizer-normalized perplexity, and produce a comparison table against S5-nano.

### AS-003 — Create model card and upload script
**Type:** Agent Story  
As an agent, I need to render a model card from template recording parent revision, data provenance/curriculum, tokenizer changes, hyperparameters, compute, evaluation results, and limitations; write a Python upload script using `huggingface_hub`; verify reload in clean environment; and publish to HF.

## Implementation Decisions

1. **Training recipe**: Continued pretraining from S5-nano (not from-scratch, not LoRA). Fresh Adam state, peak LR 3e-4, warmup 2k steps, cosine decay. MTP weight stays 0.3.
2. **Data curriculum**: Start 50% Portuguese / 50% multilingual replay, anneal to 90/10 by token count.
3. **Compute target**: Single 48 GB GPU. `seq_len=2048`, `batch_size=1`, `grad_accum=32`, gradient checkpointing. `seq_len=4096` as stretch if memory permits.
4. **Tokenizer extension**: Preserve all existing token IDs and special-token behavior. Add tokens only after measuring vocabulary gap and held-out tokenization gain. Initialize new embedding rows from mean of their original-tokenizer decomposition. Export extended tokenizer with adapted checkpoint.
5. **Reference separation**: The pinned `base` preset remains the exact S5-nano compatibility target. All specialization work produces a separate Cafezin artifact with its own versioned release.
6. **Python/numpy**: `requires-python >=3.12`, `numpy` moved to optional `data` and `dev` extras with lazy import. **DONE** in codebase.
7. **Vision norm**: `zero_centered_norm` setting unresolved. Test script written at `scripts/test_vision_norm.py`. Requires transformers from git for HF reference comparison.

## Testing Decisions

- **Unit/integration**: Existing test suite (43 tests) must pass before and after any changes. CI runs on Ubuntu and Windows.
- **Training smoke**: `test_prepare_train_resume_export_generate` validates the full CLI round-trip on CPU with `tiny` preset.
- **Continued pretraining verification** (MS-001):
  - Seam: `cafezin-lm train` CLI with extended tokenizer + grown embeddings + S5-nano init
  - Verify: Checkpoint loads in clean env, `export` produces valid HF layout, `generate` produces coherent text
- **Tokenizer extension** (DS-002):
  - Seam: Extended tokenizer round-trip encode/decode on PT text; new tokens map to grown embedding rows
  - Verify: `cafezin-lm verify --offline` still passes on unchanged `base` preset
- **Evaluation** (AS-002):
  - Seam: `lm-eval-harness` with custom Portuguese tasks
  - Verify: Perplexity, downstream accuracy, multilingual retention, generative quality vs S5-nano
- **Vision norm** (AS-001):
  - Seam: `scripts/test_vision_norm.py` against HF reference
  - Verify: One setting matches within 1e-3 max absolute difference

## Out of Scope

- Full pretraining from scratch
- LoRA/QLoRA fine-tuning as primary specialization method
- New architecture experiments (attention schedule variants, MoE, different linear kernels)
- Multi-GPU or distributed training infrastructure
- Vision preprocessing pipeline (image/video processors, chat templates)
- Instruction tuning / RLHF (this is continued pretraining only)

## Further Notes

- **ADRs**: Five ADRs recorded in `docs/adr/` (0001–0005) capture the binding decisions.
- **SAIL skills**: Repo configured with `AGENTS.md`, `docs/agents/issue-tracker.md`, `docs/agents/triage-labels.md`, `docs/agents/domain.md`. Issue tracker = GitHub Issues on `Matheussoranco/Pico-LM`.
- **Codebase state**: All 43 tests pass, ruff clean, CLI verified end-to-end. `pyproject.toml` updated for Python 3.12+ and optional numpy. `data.py` uses lazy numpy import.
- **Vision norm test**: Script exists but blocked on transformers git install. Run separately to resolve ADR-0001.
- **Release repo**: Git remotes still point to `Pico-LM` (legacy name). Update remote or publish to new HF repo when releasing.