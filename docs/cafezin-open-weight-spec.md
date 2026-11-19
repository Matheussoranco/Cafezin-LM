# Cafezin LM: Brazilian Portuguese specialization and open-weight release

Status: Draft — test seams pending confirmation; not published.

## Problem Statement

Cafezin LM provides a reference implementation and a working small-model training and export pipeline, but those checks do not demonstrate that a Cafezin-trained checkpoint improves on S5-nano in Brazilian Portuguese, reflects Brazilian culture and knowledge, or retains multilingual capability. There is no verified Cafezin release package for users to download and reproduce.

## Solution

Produce and evaluate a distinct Cafezin checkpoint by continuing pretraining from the pinned S5-nano weights. Preserve the upstream reference preset for compatibility work, then publish the adapted model with the artifacts and evidence needed to load it and understand its strengths and limitations.

## Stories

### DS-001 — Build a Brazilian Portuguese curriculum with multilingual replay

**Type:** Data Story

- Goal: Increase exposure to Brazilian Portuguese and Brazilian culture and knowledge while retaining multilingual replay.
- Source: Existing Portuguese corpora curated for Brazilian relevance and documented provenance, plus multilingual replay. Unresolved: exact sources and revisions.
- Data type: Text corpora for continued pretraining.
- Approximate quantity: Unresolved: corpus sizes and total token budget.

### MS-001 — Adapt the pinned S5-nano checkpoint

**Type:** Model Story

- Learning goal: Improve Brazilian Portuguese and Brazilian cultural and factual coverage while retaining multilingual capability.
- Learning type: Continued pretraining from pinned S5-nano weights, with measured tokenizer extension.
- Data source: The curriculum in DS-001.
- Expected performance: Compare against unadapted S5-nano on held-out language modeling, Portuguese downstream tasks, multilingual benchmarks, and Brazilian Portuguese generation. Unresolved: exact scores, improvement thresholds, and acceptable multilingual regressions.

### US-001 — Download and load the adapted checkpoint

**Type:** User Story

As a Cafezin model user, I want a versioned checkpoint with matching tokenizer and configuration assets and a model card, so that I can load it and understand how it was trained, what was evaluated, and what remains unsupported.

## Implementation Decisions

- Keep the pinned upstream preset as the reference compatibility target. Treat Cafezin weights and tokenizer as a distinct artifact. A metadata fingerprint is not a numerical parity check.
- Schedule the training mixture by token count from 50/50 Portuguese/multilingual toward 90/10. Record provenance and document the realized mix.
- Mine candidate Brazilian Portuguese pieces and measure held-out tokenization gains and multilingual regressions before choosing the extension size. Preserve original token IDs, special-token behavior, and existing embedding rows; initialize added rows from the mean of their original-tokenizer decomposition and preserve the tied output projection.
- Initialize from model weights rather than resuming upstream trainer state. Start fresh Adam state and a new warmup and decay schedule; retain the experimental MTP loss weight of `0.3`. Treat 2,048-token and 4,096-token contexts as single-GPU baseline and stretch targets, not verified fits.
- Use `lm-evaluation-harness` as the evaluation foundation and add coverage where required. The minimum evidence is held-out language modeling, Portuguese downstream evaluation, and multilingual evaluation; success also requires Brazilian Portuguese generative evaluation. Report cross-lingual retention separately and use a tokenizer-normalized metric for comparisons across vocabularies.
- Release versioned SafeTensors weights, tokenizer assets, model and generation configs, and a model card. Record the parent revision, corpus provenance and curriculum, tokenizer changes, compute, hyperparameters, results, and limitations. Check redistribution terms and notices before publishing. Use an explicitly invoked Python uploader, not automatic publishing from CI.
- Compare both vision normalization settings against a versioned reference using the same weights and inputs. Save all three outputs and use the agreed `1e-3` maximum absolute difference for reference agreement. No setting has been established as the match.

## Testing Decisions

Proposed seams, pending confirmation. Assert observable inputs, outputs, and artifacts rather than private functions or implementation order.

- **Training and release:** Extend the existing CLI prepare → train/resume → export → reload/generate smoke test. Use a tiny parent fixture to verify original token IDs and embedding rows at the initialization artifact boundary, then exercise training and load the exported checkpoint in a clean environment. Retain the existing offline fingerprint and checkpoint round-trip tests for the unchanged reference path.
- **Evaluation:** Exercise the evaluation runner against the parent and adapted releases and inspect the resulting comparison report. Use a bounded fixture for integration coverage; real model-quality evidence needs the selected held-out datasets and scoring rubrics.
- **Vision:** Compare the vision tower's public forward outputs with the versioned reference. Verify the saved outputs and ensure a missing reference cannot count as passing. The existing vision forward and output-shape tests are prior art, not evidence of reference parity.
- **Compute:** Measure peak memory and throughput at the actual training entry point on the target GPU before committing the run budget.

## Out of Scope

Pretraining a new base model from scratch, replacing the upstream tokenizer, new architecture variants during stabilization, automatic Hub publishing from CI, and claiming full end-to-end multimodal support from vision-tower parity alone.

## Further Notes

- Unresolved: exact datasets, split and decontamination rules, vocabulary expansion size, run budget, peak learning rate, benchmark thresholds, and successful reference vision comparison.
- The `bdd-scenarios` skill was unavailable; agreed behavior was not written as Gherkin scenarios.
- Tracker and triage mapping were not configured for this repository when drafted; no issue was published or labeled.
