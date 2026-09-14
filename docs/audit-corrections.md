# September audit corrections

The implementation addresses checkpoint trust, continuation state, input validation,
measurement semantics and experiment provenance. Existing training weights and
historical reports are preserved. New reports use the `audit-` prefix.

## Checkpoint compatibility

Every loader now uses `weights_only=True`. Dynamic CPU int8 exports store a state
dictionary and reconstruct their module locally; generation automatically selects
CPU for this format. Old exports containing `quantized_model` pickle objects are
rejected. Re-export the original floating-point checkpoint with `quantize --method
dynamic`; there is no automatic unsafe fallback. Quantized state dictionaries
must match their shell exactly. The underlying torch.ao API still issues deprecation
warnings on the validated environment; migration to another backend is separate work.

## Continuation and provenance

New pretraining and SFT checkpoints save CPU/CUDA, Python and NumPy RNG states,
AMP scaler state, optimizer, step and configuration. RNG restoration happens after
model construction. CLI resume inherits saved schedules, with explicit flags taking
precedence. Match the original data, teacher/replay options, evaluation schedule and
training horizon when testing exact continuation. Raising `max_steps` changes the
cosine schedule; changing hardware or kernels may change numerical results.
Legacy checkpoints without RNG state remain usable with an explicit warning.

New manifests record commit, dirty state, tracked-diff digest, hashes of source
files, data and checkpoint files, tokenizer identity and runtime versions. `bench`
includes the exact ranking items. These fingerprints establish identity, not public
availability: users still need the referenced data and weights. Historical reports
cannot be retroactively assigned trustworthy provenance. Keep raw artifacts outside
Git and retain them alongside their manifests when sharing an experiment.

## Evaluation corrections

- Cloze distractors are distinct; insufficient diversity raises a clear error
  instead of looping. Scores from the old item generator are not directly comparable.
- Multi-turn instruction evaluation targets the final assistant answer and keeps
  the preceding conversation, matching SFT preparation.
- Matching preserves case, so lowercase output cannot pass an uppercase task.
- Four-gram repetition includes only outputs with at least four tokens and reports
  the number of eligible samples. Zero with no eligible samples is not evidence of
  healthy generation. Empty/short answers must be assessed through length and EOS.
- Speed uses the median of recorded timings, counts the effective output budget,
  and explicitly includes prefill. Training timing uses a monotonic clock and excludes
  evaluation time without discarding the accumulated training interval.

## Measured external checks

The passkey probe uses five-digit random codes, seeds 0/1/2, and insertion positions
near 10%, 50% and 90% of the filler. Each stored item includes prompt token IDs and
the generated answer. The long model uses 128/496/1008/2032/4080 input tokens, reserving
16 tokens for output: **0/45 correct**. The base model uses 128/496: **0/18 correct**.
This is a completion-format retrieval test; failure does not isolate instruction
understanding from retrieval, and multiple evaluation seeds are not training seeds.

An external sample contains the first 20,000 characters after Project Gutenberg's
start marker for *Alice's Adventures in Wonderland* (ebook 11), including front matter.
The saved sample contains 6,662 scored tokens. Both models use identical 512-token
windows and token-weighted loss. Base loss is **7.26349** (PPL **1427.23**); long-model
loss is **7.40825** (PPL **1649.54**). One literary sample is a domain-shift smoke test,
not a representative external benchmark or a causal test of context extension.

Reports: `results/audit-20260904-generalization-long.json`,
`results/audit-20260906-generalization-base.json`; the exact external sample is
`results/audit-20260904-generalization-long.external.jsonl`.

The corrected SFT report (`results/audit-20260906-micro-sft.json`) evaluates all 400
test instructions: **97.5%** overall, **88.6%** uppercase. Corpus PPL remains **903.20**,
mean free generation remains approximately **2.7 tokens**, and EOS rate is **100%**.
New cloze accuracy is **71.5%** on newly generated items. Four-gram repetition is **0**
among eligible outputs; the historical **0.833** was a metric defect, not measured looping.

## Run the checks

```powershell
python -m pytest -m ""
ruff check src tests experiments
ruff format --check src tests experiments
python experiments/evaluate_generalization.py --ckpt runs/micro-long-continued/best.pt --device cuda --lengths 128 496 1008 2032 4080 --seeds 0 1 2 --trials 3 --external-jsonl results/audit-20260904-generalization-long.external.jsonl --out results/new-generalization.json
```

CI runs CPU end-to-end tests on Windows and Linux. CUDA is validated locally and
remains experimental across other GPUs/drivers; hosted CI does not validate GPU
support. These corrections do not establish production or paper readiness.
