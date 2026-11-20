## Summary
Model card template (AS-003) renders with placeholder values since actual evaluations haven't been run.

## Spec Requirement
SPEC.md line 56: "render a model card from template recording parent revision, data provenance/curriculum, tokenizer changes, hyperparameters, compute, evaluation results, and limitations"

## Current State
- **Template complete**: `src/cafezin_lm/model_card.py` with comprehensive `ModelCardData` dataclasses
- **CLI**: `model-card` command loads results from:
  - `run_eval.py` output (lm-eval harness)
  - `eval_generative_br.py` output (generative eval)
  - `test_vision_norm.py` output (vision norm comparison)
  - Continued pretraining `run_settings.json`
  - Tokenizer extension output
  - Data provenance JSON

- **Template sections**: Model Summary, Parent Checkpoint, Data Provenance, Tokenizer Changes, Training Settings, Evaluation Results (bits-per-byte, PT downstream, multilingual retention, generative, vision norm), Limitations, License

## Gap
Template renders with placeholder/default values because:
1. lm-eval harness not run on actual models
2. Generative eval not run with LLM judge
3. Vision norm test unresolved (diff=0.625 vs 1e-3 target)
4. Continued pretraining not run on 48GB GPU
5. Tokenizer extension not run on real Brazilian corpus

## Template Behavior
`model_card.py` gracefully handles missing data by:
- Using default/empty values for missing sections
- Including "Not performed" or "Not available" notes
- Limitations section populated with known caveats

## Required Work
Complete prerequisite evaluations:
1. #22 - Tokenizer extension on real corpus
2. #23 - Continued pretraining on 48GB GPU
3. #21 - Vision norm resolution
4. #22 - Generative eval with LLM judge

Then run `model-card` CLI with actual result directories to generate final card.

## Dependencies
Blocks: HF release upload (#17)