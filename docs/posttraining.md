# Post-training: fine-tuning, quantization, distillation

Historical measurements: the September audit corrected case-insensitive instruction
scoring, short-output repetition, and duplicate cloze distractors. In particular,
the old SFT repetition value 0.833 is an invalid repetition estimate: outputs shorter
than four tokens were counted as fully repetitive. The loss increase and premature
EOS remain evidence of forgetting. See [audit corrections](audit-corrections.md)
for new reports; do not compare old and new cloze scores as model improvements.

Three things you can do to a pretrained Pico-LM checkpoint, and what each one
actually measured on this hardware (RTX 3050 6 GB, Windows).

Every number below is a **behavioural score on held-out data**, not a training
loss. That is not a stylistic preference. A fine-tune in this repo reached a
training loss of 8e-6 and a dev exact match of 100% while quietly destroying
the model's ability to write a sentence; only the general-ability probes caught
it. Loss tells you how well you fit the thing you optimized. It cannot tell you
what you broke.

---

## The evaluation harness first

`pico-lm bench` runs four probes, each with an explicit chance baseline.

| probe | what it measures | chance |
|---|---|---|
| **minimal pairs** | prefers a real sentence over a minimally corrupted copy (adjacent-word swap, subject–verb agreement, pronoun case, a/an). Word count is preserved so mean-per-token normalization is not deciding the answer. | 50% |
| **cloze, 4-way** | prefers the true continuation of a held-out context over three continuations lifted from elsewhere in the split | 25% |
| **generation health** | free-generation degeneracy: distinct-2, 4-gram repetition, whether generation terminates, mean length | — |
| **instruction EM** | greedy-decoded exact match on held-out instruction prompts | ~0% |

```powershell
pico-lm bench --ckpt runs\micro\best.pt --data data `
    --instructions data_sft\instruct_test.jsonl --json results\micro.json
```

The first two are *ranking* probes: the model only has to put the right option
above the wrong ones, which tolerates a lot of damage. They are the least
sensitive thing here and should never be the only evidence. Generation health
and exact match are far sharper.

---

## 1. Fine-tuning (SFT)

```powershell
# a verifiable instruction set: train / dev / test, mutually disjoint
python examples\make_instruct_data.py --out data_sft --train 6000 --dev 200 --test 400

pico-lm prepare-sft --jsonl data_sft\instruct_train.jsonl `
    --tokenizer data\tokenizer.json --out data_sft --max-len 256

pico-lm finetune --sft-data data_sft --init-from runs\micro\best.pt `
    --out runs\micro-sft --bench-jsonl data_sft\instruct_dev.jsonl `
    --select-by score --max-steps 2000 --lr 3e-4
```

What makes this SFT rather than "more pretraining on chat-shaped text":

- **Example-aligned batches.** Pretraining draws random windows from one long
  token stream, so a sample routinely starts mid-answer. Here one row of the
  batch is one complete example.
- **Loss masked to the response.** Prompt positions get target `-1`, which
  `Transformer.forward` already treats as `ignore_index`. Training on the prompt
  teaches the model to write instructions, not follow them.
- **Its own schedule.** `--init-from` loads pretrained weights and starts a
  fresh low-LR cosine schedule at step 0. `--resume` is the different operation
  that continues an interrupted SFT run.
- **Behavioural checkpoint selection.** Every eval greedy-decodes the *dev*
  prompts; `best.pt` tracks that exact match, not the loss. Dev and test are
  disjoint, so the reported test number is not a selection metric.

### Measured: micro 15.7M, 2000 steps, lr 3e-4

Held-out test split, 200 prompts never seen in training or selection.

| metric | base | after SFT |
|---|---|---|
| **instruction EM** | **0.0%** | **97.5%** |
| minimal pairs | 98.8% | 91.2% |
| cloze 4-way | 98.0% | 62.0% |
| distinct-2 | 0.900 | 0.424 |
| 4-gram repetition | 0.011 | 0.833 |
| mean generation length | 82 tok | 2.7 tok |
| eos rate | 29% | 100% |
| corpus val loss | 1.555 | 6.806 (ppl 4.7 → 903) |

Per task: `first_word`, `last_word`, `repeat`, `word_count` 100%; `reverse` 95%;
`uppercase` 90%.

**The fine-tune both worked and was a disaster, depending which number you
read.** Instruction following went from nothing to nearly perfect. Everything
else collapsed: the model now answers in about three tokens and emits EOS, so
`"One day, a little girl named"` completes to `" Jack"` and stops. Corpus
perplexity rose 190x.

This is catastrophic forgetting, and it is visible only because the eval scores
general ability alongside the target task. Note that *every single number
available from inside the run looked perfect*: training loss 8e-6, dev exact
match 100% by step 1500. Reporting only the target-task score would have been
the same mistake as reporting only the loss — just a more flattering version.

Cause is not mysterious: six narrow synthetic tasks, ~5.6 epochs, loss driven to
zero, no pretraining data mixed in. The standard mitigation is replay — mixing
a fraction of the pretraining corpus into the SFT batches — plus stopping much
earlier. If you fine-tune with this pipeline, **bench the general probes, not
just your task.**

Replay is now implemented directly. It mixes response-masked SFT loss with an
ordinary next-token batch from the original prepared corpus:

```powershell
pico-lm finetune --sft-data data_sft --init-from runs\micro\best.pt `
    --out runs\micro-sft-replay --bench-jsonl data_sft\instruct_dev.jsonl `
    --replay-data data --replay-weight 0.1 --lr 1e-4
```

The destructive result above is the SFT-only baseline, not a measured claim
about the new replay setting. Run both, benchmark both, and treat replay weight
and stopping step as experimental variables.

---

## 2. Quantization

```powershell
pico-lm quantize --ckpt runs\micro\best.pt --out runs\micro-int8.pt `
    --bits 8 --group-size 64 --quantize-embedding
pico-lm bench --ckpt runs\micro-int8.pt --data data
```

Weight-only symmetric absmax quantization. int8 is per-output-channel or
grouped; int4 is grouped along the input dimension and packed two per byte.
Weights are dequantized to the compute dtype inside `forward`, so quantized
checkpoints run on any device and load through the normal `load_model` path —
`generate`, `chat`, `eval`, `info` and `bench` all accept them.

### Measured: micro 15.7M and nano 2.9M, CUDA

fp32 baselines are re-saved **without optimizer state**, otherwise every
compression ratio is inflated ~3x by Adam moments.

**micro 15.7M**

| config | size | val loss | pairs | cloze | distinct-2 | tok/s |
|---|---|---|---|---|---|---|
| fp32 | 62.96 MB | 1.5447 | 99.0% | 95.3% | 0.928 | 22.4 |
| int8 g64 linear only | 34.95 MB | 1.5447 | 99.0% | 95.3% | 0.917 | 16.9 |
| **int8 g64 + embedding** | **16.28 MB** | **1.5447** | 99.0% | 95.3% | 0.894 | 17.1 |
| int8 per-channel + emb | 15.86 MB | 1.5449 | 99.3% | 95.3% | 0.895 | 17.1 |
| int4 g64 linear only | 30.23 MB | 1.5799 | 99.7% | 94.0% | 0.908 | 10.1 |
| int4 g64 + embedding | 8.41 MB | 1.6003 | 98.7% | 95.3% | 0.868 | 9.7 |
| int4 g32 + embedding | 8.90 MB | 1.5920 | 98.3% | 95.3% | 0.884 | 8.8 |
| int4 per-channel + emb | 7.99 MB | 1.6345 | 98.7% | 93.3% | 0.837 | 9.7 |

**nano 2.9M**

| config | size | val loss | pairs | cloze | distinct-2 | tok/s |
|---|---|---|---|---|---|---|
| fp32 | 11.55 MB | 2.2461 | 98.7% | 86.0% | 0.902 | 39.4 |
| **int8 g64 + embedding** | **3.00 MB** | **2.2467** | 98.7% | 86.0% | 0.897 | 26.2 |
| int8 per-channel + emb | 2.96 MB | 2.2463 | 98.7% | 86.0% | 0.893 | 26.5 |
| int4 g64 linear only | 8.84 MB | 2.3142 | 99.0% | 86.0% | 0.882 | 14.9 |
| int4 g64 + embedding | 1.56 MB | 2.3571 | 98.0% | 85.3% | 0.866 | 14.2 |
| int4 g32 + embedding | 1.65 MB | 2.3452 | 98.3% | 84.0% | 0.859 | 13.9 |
| int4 per-channel + emb | 1.51 MB | 2.3762 | 99.3% | 87.3% | 0.886 | 14.1 |

### Three things worth stating plainly

**int8 has small loss changes in these runs.** Micro loss rounds to the same
four decimals; nano changes from 2.2461 to 2.2467. Generation diversity also changes.
Storage is about 3.9x smaller. This is the default.

**Quantization here makes inference *slower*, not faster.** micro goes
22.4 → 17.1 (int8) → 9.7 tok/s (int4). Weights are dequantized on every matmul,
and at these sizes that costs more than the memory traffic it saves. Weight-only
quantization buys disk and VRAM, not throughput. For a real CPU speedup use
`--method dynamic`, which does genuine int8 matmuls via `torch.ao` (CPU only).

**Almost all the size is the embedding.** int8 on the Linear layers alone takes
micro from 62.96 to 34.95 MB (1.8x); including the tied embedding takes it to
16.28 MB (3.9x). On nano, linear-only saves 20%, because 2.1M of its 2.9M
parameters are the 16k-vocab table. Quantizing the projections of a
small-vocab-dominated model is close to pointless.

### Does int4 break these models? Measured on an exact-match task

The ranking probes above barely move at int4, but they tolerate a lot of noise.
Exact-match generation does not. Scoring the *fine-tuned* model, held-out test:

| config | instruction EM | pairs | cloze | tok/s |
|---|---|---|---|---|
| fp32 | 97.5% | 91.2% | 62.0% | 22.1 |
| int8 g64 + emb | 97.5% | 92.0% | 64.0% | 17.2 |
| int4 g64 + emb | 97.0% | 90.7% | 66.0% | 9.6 |
| int4 g32 + emb | 97.0% | 90.0% | 64.0% | 9.7 |

int8 is exactly lossless on the task. int4 costs 0.5pp, all of it from
`uppercase` (90.0% → 87.5%) — the task with the longest, least predictable
output, which is where rounding error should bite first.

**This does not reproduce the expectation that 4-bit wrecks sub-1B models**, and
the reason matters more than the result. These are narrow models: TinyStories
pretraining, six synthetic tasks, a very peaked output distribution. A peaked
distribution survives weight noise because the argmax has a wide margin. A
general-purpose instruct model choosing among many plausible continuations has
no such margin. Granularity is also often the real culprit: int4 per-channel on
micro gives loss 1.6345 / distinct-2 0.837, versus 1.6003 / 0.868 at
group_size 64.

So the defensible claim is narrow: **at 3–16M parameters, on peaked
narrow-domain tasks, int4 with group_size 64 costs a few percent perplexity and
about 0.5pp exact match.** That is not evidence that int4 is safe for general
small models. The default stays int8, which costs nothing measurable.

---

## 3. Distillation

```powershell
pico-lm train --preset nano --data data --out runs\nano-kd `
    --teacher runs\micro\best.pt `
    --seq-len 256 --batch-size 32 --grad-accum-steps 2 --max-steps 4000
```

KL divergence on temperature-softened logits plus the ordinary cross-entropy
term: `loss = (1 - kd_alpha) * CE + kd_alpha * T^2 * KL(teacher || student)`,
with `kd_alpha` and `kd_temperature` in `TrainConfig` (defaults 0.5 and 2.0).

**Their training losses are not comparable.** The KD student optimizes a blend
of cross-entropy and KL, so its loss curve sits on a different scale from the
control's. Only the behavioural probes can rank these three models.

### VRAM: watch throughput, not exit status

On a 6 GB card a KD step at `--batch-size 64` peaks at 8.7 GB. Windows serves
the overflow from system RAM instead of failing, so the run *succeeds* — at
7.2k tok/s. `--batch-size 32 --grad-accum-steps 2` is the same 16,384
tokens/step, peaks at 4.4 GB, and runs at 34.6k tok/s. **4.8x faster for the
same work, and the only symptom of the bad configuration was the throughput
number.**

Measured on this machine:

| config | peak VRAM | throughput |
|---|---|---|
| bs 64, KD | 8.68 GB | 7,217 tok/s |
| bs 32 × accum 2, KD | 4.41 GB | 34,617 tok/s |
| bs 16 × accum 4, KD | 2.28 GB | 20,609 tok/s |

### Measured: micro 15.7M → nano 2.9M

Both students: 4000 steps, 65.5M tokens, seed 1337, identical data. Probes are
sized for power — n=2000 minimal pairs, n=1000 cloze — because the gaps here are
a couple of points and at n=200 a 2.5pp difference is one standard error.

| model | val loss | ppl | minimal pairs | cloze 4-way | distinct-2 | 4-gram rep | tok/s |
|---|---|---|---|---|---|---|---|
| **teacher** micro 15.7M | 1.5547 | 4.73 | 98.90% ±0.23 | 97.40% ±0.50 | 0.887 | 0.012 | 25.2 |
| **student, distilled** nano 2.9M | 2.2524 | 9.51 | 98.65% ±0.26 | 85.50% ±1.11 | 0.883 | 0.015 | 36.9 |
| **student, from scratch** nano 2.9M | 2.2675 | 9.66 | 98.15% ±0.30 | 82.50% ±1.20 | 0.875 | 0.045 | 37.2 |

Distillation minus from-scratch:

| probe | difference | significance |
|---|---|---|
| minimal pairs | +0.50pp | z = +1.26 — not significant |
| cloze 4-way | +3.00pp | z = +1.83 — marginal |
| 4-gram repetition | 0.045 → 0.015 | **3x less degenerate** |
| val loss | 2.2675 → 2.2524 | −0.7% |

**Distillation helps, modestly, and the honest headline is the repetition
number rather than the accuracy numbers.** On the ranking probes the gain is
one to two sigma — suggestive, not established. The unambiguous effect is on
free generation: the from-scratch student loops on itself three times as often.
That shows up qualitatively too. Same prompt, same budget:

> **KD:** *"…went to the park to play. She saw a little girl named Lily with a
> big smile on her face. Lily wanted to play too, so she asked her mom…"*
>
> **scratch:** *"…went to the park to play. She saw a little girl named Lily
> with a new dress. Lily wanted to make a show, but Lily said, 'I don't want to
> go to the park to play with me!'"*

Two caveats that matter more than the gain:

**The capacity gap does not close.** The student reaches cloze 85.5% against
the teacher's 97.4%. Distillation makes a 2.9M model a slightly better 2.9M
model; it does not make it a small 15.7M model.

**At equal wall-clock, from-scratch may well win.** These runs are matched on
tokens, not time. The KD run trains at 32.8k tok/s against the control's 74k,
because every step also pays for a teacher forward pass — 2.3x the wall-clock
for the same data. Spending that 2.3x on more from-scratch steps instead is a
real alternative, and this experiment does not rule it out. Distillation is
clearly worth it when the teacher already exists and you are compute-bound at
*inference*; it is not obviously worth it when you are compute-bound at
training.

**`base` (88M) was not used as the teacher.** Training it to convergence is
roughly a day on this GPU, which did not fit the window. `micro` at 15.7M is a
well-trained teacher (20k steps, 655M tokens, val 1.474) and gives a 5.4x
compression to nano, so the pipeline is demonstrated on a real teacher/student
pair — but the teacher is 15.7M, not 88M.

---

## Reproducing

```powershell
experiments\run_sft.ps1            # fine-tune + behavioural eval
experiments\run_quantize.py        # size / speed / quality sweep, two model sizes
experiments\run_quantize_sft.ps1   # quantized fine-tuned model, exact-match task
experiments\run_distill.ps1        # KD student + from-scratch control
experiments\run_distill_eval.ps1   # score teacher / student / control
```

Anything longer than a couple of minutes should be launched detached and polled,
not run in the foreground:

```powershell
Start-Process powershell -ArgumentList '-NoProfile','-File','experiments\run_sft.ps1' -WindowStyle Hidden
```

Raw reports land in `results/*.json`.
