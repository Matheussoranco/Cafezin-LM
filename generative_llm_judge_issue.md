## Summary
Generative evaluation (AS-002) uses heuristic judge fallback instead of LLM judge (GPT-4, Claude).

## Spec Requirement
SPEC.md line 52: "add a generative eval (MT-Bench style or LLM-judge)"

## Current State
- **Infrastructure complete**: `scripts/eval_generative_br.py` with:
  - Versioned prompts: `eval/generative/prompts_v1.json` (12 prompts across culture, knowledge, reasoning, writing)
  - Versioned rubric: `eval/generative/rubric_v1.json` (5 dimensions: factual_accuracy 30%, cultural_competence 25%, language_quality 20%, reasoning_depth 15%, completeness 10%)
  - `heuristic_judge()` - rule-based fallback scoring
  - `scripts/smoke_test_generative.py` for testing
  - Integration into `scripts/run_eval.py`

- **Tests**: 4 tests in `tests/test_generative_eval.py` passing

## Gap
No LLM judge integration (GPT-4, Claude, etc.). Current `heuristic_judge` is rule-based:
- Scores based on keyword matching, length, simple heuristics
- Not semantically aware
- Cannot evaluate nuance, cultural competence, reasoning depth properly

## Required Work
1. Add LLM judge support (OpenAI GPT-4, Anthropic Claude, or local LLM)
2. Implement judge protocol: prompt template, response parsing, score extraction
3. Add calibration examples for judge alignment
4. Compare heuristic vs LLM judge on same prompts
5. Document judge model, version, temperature, prompt template

## Limitations Noted
Model card limitations already include: "Generative evaluation uses heuristic judge for offline reproducibility; production evaluation requires capable LLM judge."

## Dependencies
- Requires API keys for LLM judges
- Blocks: Model card rendering (#16), HF release (#17)