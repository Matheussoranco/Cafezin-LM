# Learning guide

Pico-LM is intentionally small enough to read end to end. It is a real causal
language model, but it is not meant to compete with a production model trained
on trillions of tokens. Its value is that every important stage is visible and
hackable.

## The path through the code

Read these files in this order:

1. `config.py` defines the model and training experiments.
2. `tokenizer.py` turns text into byte-level BPE token IDs.
3. `data.py` streams documents into memory-mapped token arrays.
4. `model.py` maps `(batch, time)` token IDs to next-token logits.
5. `train.py` optimizes cross-entropy and optionally distils a teacher.
6. `generate.py` performs cached autoregressive decoding.
7. `sft.py` masks prompt loss and fine-tunes on instruction responses.
8. `quant.py` replaces floating-point weights with grouped int8/int4 weights.
9. `bench.py` tests behavior that validation loss alone cannot describe.

## One transformer pass

For a batch of `B` sequences, length `T`, model width `D`, and vocabulary `V`:

```text
token IDs (B,T)
  -> embedding (B,T,D)
  -> [RMSNorm -> causal GQA -> residual
      RMSNorm -> SwiGLU MLP -> residual] x N layers
  -> RMSNorm
  -> tied language-model head (B,T,V)
  -> cross-entropy against the next token
```

RoPE rotates query and key pairs as a function of position, so no learned
position table is required. GQA uses fewer key/value heads than query heads,
which shrinks the decoding cache. PyTorch SDPA selects an efficient attention
kernel when the hardware supports one.

## What makes the model intelligent

Architecture matters, but at this scale data and evaluation matter more.

- Train a tokenizer on the same domain as the corpus.
- Prefer coherent, deduplicated, age-appropriate text over a large noisy scrape.
- Use enough tokens. A 15.7M-parameter model is still undertrained after a few
  million tokens.
- Hold out documents, not random token fragments copied from the same document.
- Judge samples and behavioral probes alongside loss.
- Add instruction data only after the base model can complete ordinary prose.

TinyStories is a good first corpus because its language and world model fit a
small network. A broad assistant corpus does not magically give a 15M model the
knowledge or reasoning capacity of a billion-parameter model.

## Long context

`micro-long` has the same 15.7M trainable parameters as `micro`, but uses a
4,096-token window and a larger RoPE base. Context length changes activation and
attention cost, not parameter count. Attention compute grows approximately with
`T^2`, so 4,096 tokens is far more expensive than 512.

Start with:

```powershell
pico-lm train --preset micro-long --data data --out runs\micro-long `
    --init-from runs\micro\best.pt `
    --seq-len 4096 --batch-size 1 --grad-accum-steps 32 `
    --gradient-checkpointing --max-steps 20000
```

Activation checkpointing discards intermediate block activations and recomputes
them during backward. It lowers memory use but increases step time. If 4,096
still does not fit, reduce `--seq-len`; gradient accumulation recovers the token
batch size but cannot reduce the memory of one sequence.

`--init-from` is different from `--resume`: it keeps compatible learned weights
but builds the requested context/RoPE configuration and resets the optimizer and
schedule. `--resume` restores the exact prior architecture, optimizer, and step.

Long context only becomes useful when training examples contain dependencies at
that distance. Padding short stories to 4,096 tokens spends compute without
teaching long-range recall.

## Fine-tuning without erasing the base model

SFT loss applies only to assistant response tokens. A narrow dataset can still
cause catastrophic forgetting: every update points toward the small SFT domain
and away from general language modeling.

Replay keeps a fraction of the original objective active:

```powershell
pico-lm finetune --sft-data data_sft --init-from runs\micro\best.pt `
    --out runs\micro-sft-replay --bench-jsonl data_sft\instruct_dev.jsonl `
    --replay-data data --replay-weight 0.1 --lr 1e-4
```

The mixed objective is:

```text
loss = (1 - replay_weight) * response_masked_SFT_loss
     + replay_weight       * ordinary_pretraining_loss
```

Replay costs a second forward pass. Compare several weights and stop based on
both instruction accuracy and the general probes in `pico-lm bench`.

## Experiments worth trying

- Compare `micro` and `micro-long` at the same token budget.
- Change `n_kv_heads` and measure KV-cache size and decoding speed.
- Distil `micro` into `nano`, then compare against a scratch `nano` trained for
  the same number of tokens and the same wall-clock time.
- Sweep SFT replay weights `0`, `0.05`, `0.1`, and `0.2`.
- Quantize the same checkpoint to int8 and int4, then compare file size, loss,
  behavioral scores, and tokens per second.
- Add a new behavioral probe before changing the architecture. A metric often
  teaches more than another layer.

Keep a control run for every experiment. Without one, a falling loss can make a
broken model look successful.
