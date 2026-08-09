# Changelog

All notable changes to Pico-LM are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-08-09 — Post-training: fine-tune, quantize, distil

0.1.0 pretrained a Llama-style SLM from scratch. 0.2.0 adds the three things you
do to a model afterwards, and — because the point was to find out whether they
help — measures every one of them on real checkpoints rather than shipping the
plumbing alone.

### Added

- **`pico-lm bench`** — the behavioural harness the rest of this release is judged
  by. Four probes with explicit chance baselines: grammaticality minimal pairs
  (50%), 4-way cloze (25%), free-generation degeneracy, and greedy exact-match
  instruction following. This is load-bearing rather than decorative; see below.
- **`pico-lm prepare-sft` / `pico-lm finetune`** — supervised fine-tuning with
  response-masked loss, example-aligned batches, its own LR schedule,
  `--init-from`, resume, and `best.pt` selected on a held-out behavioural score.
  Dev and test splits are disjoint, so the reported number is not the selection
  metric.
- **`pico-lm quantize --bits 8|4`** — weight-only post-training quantization,
  group-wise or per-channel, optionally including the tied embedding.
- **Distillation** — KL-on-logits plus the hard-label term, temperature
  configurable, with an equal-budget from-scratch control to compare against.

### Measured

**Fine-tuning taught the task and destroyed the language model.** micro (15.7M),
held-out test:

| probe | before | after |
|---|---|---|
| instruction exact match | 0.0% | **97.5%** |
| cloze | 98.0% | **62.0%** |
| 4-gram repetition | 0.011 | **0.833** |
| generation length | 82 tok | **2.7 tok** |
| perplexity | 4.7 | **903** |

`"One day, a little girl named"` now completes to `" Jack"` and stops. **Training
loss was 8e-6 and dev accuracy 100%** — this is exactly the failure a loss curve
cannot see, and the reason `bench` exists. Shipped as a documented warning rather
than tuned away.

**int8 is free; quantization does not make decoding faster.** 62.96 → 16.28 MB
(3.9×) with validation loss unchanged to four decimals and no probe moving. But
throughput goes **22.4 → 17.1 (int8) → 9.7 tok/s (int4)**: dequantization per
matmul costs more than the bandwidth it saves. It buys disk and VRAM, never speed.
Most of the size is the tied embedding — Linears alone give 1.8×, and nano saves
only 20% without it. Default stays int8.

**int4 did not wreck these models** (97.0% vs 97.5% EM). Stated narrowly: these are
from-scratch TinyStories models with very peaked output distributions, which is not
the same claim as int4 being safe for a general instruct model of similar size.
Granularity matters — per-channel int4 is clearly worse than g64.

**Distillation, micro (15.7M) → nano (2.9M)**, equal token budget and seed against a
from-scratch control: cloze +3.0pp (z = 1.83, marginal), minimal pairs +0.5pp
(z = 1.26, **not** significant), 4-gram repetition 0.045 → 0.015 — the one
unambiguous win. The capacity gap does not close (85.5% vs the teacher's 97.4%), and
KD costs 2.3× wall-clock per token, so at equal *wall-clock* the control may win.
The teacher was `micro`, not `base`; `base` needs roughly a day on the target GPU.

**A hardware note worth keeping.** KD at `--batch-size 64` peaks at 8.7 GB on a 6 GB
card and the run *succeeds*, at 7.2k tok/s — Windows spills to system RAM instead of
raising. The same tokens/step at `bs 32 × accum 2` peaks 4.4 GB and runs 34.6k tok/s,
**4.8× faster**. Throughput was the only symptom; nothing errored and nothing warned.

### Fixed

- **`quantize --method dynamic` wrote a checkpoint nothing could read.** That path
  hands `torch.ao` a live module and pickles it whole, while `load_model` reads with
  `weights_only=True` — so `generate`, `chat`, `info`, `eval` and `bench` all refused
  a file the CLI had just advertised as "roughly 4× smaller, CPU-only".
  `export.load_quantized` existed for precisely this and was called by nothing.
  `load_model` now retries with `weights_only=False` only after the safe load has
  specifically refused a pickled global *and* the payload is one of our own exports;
  a corrupt file still raises.
- **`info` understated dynamic-quantized checkpoints.** `torch.ao` keeps int8 weights
  in `_packed_params`, which is not a registered `Parameter`, so `num_params()`
  reported 6.30M for a 15.73M model.
- **Preset parameter counts in `config.py` had drifted** — the comments read
  ~3M/~22M/~42M/~110M against a measured 2.88M/15.73M/31.99M/88.10M, overstating
  micro by 40% and base by 25%. A test now asserts the comments against the built
  models so they cannot drift apart again.

## [0.1.0] — 2026-08-01

From-scratch Llama-style small language model: RoPE, grouped-query attention,
SwiGLU, RMSNorm, tied embeddings. Presets `nano` (2.9M) through `base` (88.1M),
a byte-level BPE tokenizer, a single-GPU training loop with AMP, gradient
accumulation and cosine decay, and a prepared TinyStories corpus.
