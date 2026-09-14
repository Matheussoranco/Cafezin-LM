# Long-context continuation results

Historical experiment. The new [audit evaluation](audit-corrections.md) adds a
passkey probe: the extended checkpoint scored 0/45, including short and long
windows. Its ability to accept 4096 tokens is not evidence of successful retrieval.

Measured on 2026-08-20 on an RTX 3050 Laptop GPU (6 GB). Raw behavioral
reports are in `results/micro-base-current.json` and `results/micro-long.json`.

## Experiment

The 15.73M-parameter `micro` checkpoint was continued into the weight-compatible
`micro-long` architecture:

```powershell
pico-lm train --preset micro-long --data data `
    --out runs\micro-long-continued --init-from runs\micro\best.pt `
    --seq-len 4096 --batch-size 1 --grad-accum-steps 8 `
    --gradient-checkpointing --max-steps 5000 `
    --lr 1e-4 --min-lr 1e-5 --warmup-steps 100
```

This is 163.84M nominal continuation tokens. The model keeps every learned
parameter, expands its RoPE cache from 512 to 4,096 positions, changes the RoPE
base from 10k to 100k, and starts a fresh optimizer. The best checkpoint was
selected at continuation step 3,500 (`best_val=1.3773` during training).

## Loss evaluation

The following measurements use 100 random evaluation batches. Both models were
evaluated at 512 tokens for the controlled comparison.

| checkpoint | evaluation context | train loss | validation loss | validation perplexity |
|---|---:|---:|---:|---:|
| micro | 512 | 1.3884 | 1.5012 | 4.49 |
| micro-long | 512 | **1.3813** | **1.4845** | **4.41** |
| micro-long | 4,096 | **1.2845** | **1.4271** | **4.17** |

The continuation did not trade away short-context quality: at the original
512-token window, validation loss improved by 0.0167. With its full window, the
model uses additional left context and reaches lower loss. The 4,096-token row
is not an architecture-controlled comparison with `micro`, which cannot run at
that length; it measures the capability unlocked by the extended checkpoint.

## Behavioral evaluation

Both checkpoints were evaluated in the same environment and seed:

| metric | micro | micro-long |
|---|---:|---:|
| validation loss (bench sample) | 1.5547 | **1.5515** |
| grammatical minimal pairs | 98.8% | 98.8% |
| 4-way cloze | 98.0% | 98.0% |
| distinct-2 | **0.900** | 0.879 |
| 4-gram repetition | **0.010** | 0.016 |
| EOS rate | 29% | 25% |
| mean generation length | 82 | 81 |
| instruction exact match | 0.0% | 0.0% |
| local decode speed | 158.4 tok/s | 190.7 tok/s |

The behavioral ranking probes are unchanged. Free-generation diversity regressed
slightly, so the result is not an across-the-board quality win. The decode speed
measurement is a local observation and should not be attributed to context
length: the trainable architecture and parameter count are unchanged.

Instruction exact match remains zero because this is a base language model, not
an instruction-tuned checkpoint. Replay-regularized SFT is the next stage.

## What this demonstrates—and what it does not

This run demonstrates checkpoint-compatible context extension, stable 4,096-token
training on a 6 GB GPU, preserved short-context behavior, and lower long-window
loss. It does not yet prove long-range retrieval: TinyStories examples are much
shorter than 4,096 tokens. A public long-context claim should add a synthetic
passkey/retrieval benchmark and training data containing real long dependencies.

Likewise, replay provides a mechanism for continual learning, but meta-learning
requires a task-distribution training objective and held-out fast-adaptation
evaluation. Those should be implemented and measured before using either term as
an unqualified capability claim.
