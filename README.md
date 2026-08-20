# Pico-LM — a small, highly capable language model, trained and run locally

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](LICENSE)

A from-scratch small language model stack in PyTorch, sized for a laptop GPU
(tuned for an RTX 3050 6 GB). Same architecture family as Llama/Mistral, scaled down:

New to language models? Follow the **[code-reading and experiment guide](docs/learning-guide.md)**
to understand the tensor flow, long-context tradeoffs, and useful controlled experiments.

- **Decoder-only transformer** — RMSNorm (pre-norm), rotary position embeddings (RoPE),
  grouped-query attention (GQA), SwiGLU MLP, weight tying, flash attention via PyTorch SDPA
- **Your own byte-level BPE tokenizer** trained on your corpus (16k vocab by default)
- **Long-context option** — `micro-long` uses a 4,096-token window, GQA, a
  100k RoPE base, and activation checkpointing to stay practical on small GPUs
- **Training loop** — bf16 mixed precision, gradient accumulation, cosine LR with warmup,
  gradient clipping, checkpoint/resume, CSV logging
- **Post-training** — supervised fine-tuning with response-masked loss, post-training
  quantization (int8 / int4), and knowledge distillation from a larger teacher
- **Behavioural evaluation** — `pico-lm bench` scores what the model *does* on held-out
  data (grammaticality, cloze, generation health, exact-match instruction following),
  with explicit chance baselines
- **Fast inference** — preallocated KV cache, top-k / top-p / temperature / repetition penalty
- **Chat** — special-token chat template + a real SFT path

## Why small models can be smart

Capability at small scale comes from three levers, all supported here:

1. **Data quality over data quantity** (the TinyStories / phi lesson): a 25M-param model
   trained on clean, simple, coherent text is dramatically more fluent than the same model
   on raw web scrape. Curate your corpus.
2. **Overtraining past Chinchilla**: for inference-bound local use, train on far more
   tokens than the "compute-optimal" ratio — small models keep improving.
3. **Distillation**: train the biggest model that fits (`base`, ~88M), then distill it
   into a smaller, faster student with `--teacher`.

## Setup

```powershell
cd C:\Users\mathe\Documents\Code\Pico-LM
.venv\Scripts\activate            # venv has CUDA PyTorch installed
pip install -e ".[data,dev]"
```

The CLI is installed as `pico-lm`; the importable package is `pico_lm`.

## Quickstart: train on TinyStories (recommended first run)

```powershell
# 1. Train a tokenizer (streams the dataset; ~2M docs)
pico-lm tokenizer --hf roneneldan/TinyStories --vocab-size 16384 --out data\tokenizer.json --max-docs 500000

# 2. Tokenize into train/val memmaps (also copies the tokenizer into data\)
pico-lm prepare --hf roneneldan/TinyStories --tokenizer data\tokenizer.json --out data --max-docs 500000

# 3. Train (micro ≈ 15.7M params; a few hours on the 3050)
pico-lm train --preset micro --data data --out runs\micro --batch-size 32 --grad-accum-steps 2 --seq-len 512

# 4. Generate
pico-lm generate --ckpt runs\micro\best.pt --prompt "Once upon a time"
```

Or use your own text files: `--input corpus\*.txt` instead of `--hf ...`. Local files are
split into documents on blank lines.

Training copies the tokenizer into the run directory, so `runs\micro\` is self-contained:
`generate` and `chat` keep working even if `data\` is deleted.

Interrupted? `--resume` continues from `<out>\last.pt` with the optimizer state, step
counter and best-val bookkeeping intact — raise `--max-steps` to train further.

## Presets (16k vocab, sized for 6 GB VRAM)

| preset | params | dim | layers | heads (kv) | context |
|--------|--------|-----|--------|------------|---------|
| nano   | ~2.9M  | 128 | 4      | 4 (2)      | 256     |
| micro  | ~15.7M | 384 | 6      | 6 (2)      | 512     |
| small  | ~32.0M | 512 | 8      | 8 (4)      | 1024    |
| base   | ~88.1M | 768 | 12     | 12 (4)     | 1024    |
| micro-long | ~15.7M | 384 | 6 | 6 (2) | 4096 |

(`pico-lm presets` prints these live, computed from the actual model.)

For `base` on 6 GB, use `--batch-size 8 --grad-accum-steps 8`. For a real
long-context experiment, start conservatively:

```powershell
pico-lm train --preset micro-long --data data --out runs\micro-long `
    --seq-len 4096 --batch-size 1 --grad-accum-steps 32 --gradient-checkpointing
```

To extend an already-trained compatible `micro` instead of starting over, add
`--init-from runs\micro\best.pt`. This copies learned weights into the long-context
architecture and intentionally starts a fresh optimizer and step counter.

Context length is not free: attention compute grows quadratically with sequence
length. Train at 512–1024 first to validate a corpus and schedule before spending
compute on 4096-token windows.

### Training flags

`train` takes `--preset --data --out --tokenizer --resume --init-from --teacher --compile --seq-len
--batch-size --grad-accum-steps --max-steps --lr --min-lr --warmup-steps --eval-interval
--eval-iters --log-interval --seed`. Anything you leave out falls back to `TrainConfig`
in `config.py`. Add `--gradient-checkpointing` when sequence activations do not fit.
`pico-lm train --help` is the authority.

## Post-training

Full write-up with every measured number: **[docs/posttraining.md](docs/posttraining.md)**.

### Evaluate behaviour, not loss

```powershell
pico-lm bench --ckpt runs\micro\best.pt --data data
```

Four probes with explicit chance baselines — grammaticality minimal pairs (50%),
4-way cloze (25%), free-generation degeneracy, and greedy exact-match instruction
following. Use this for every before/after comparison. A fine-tune in this repo hit a
training loss of 8e-6 and 100% dev accuracy while destroying the model's ability to
write a sentence; only these probes caught it.

### Fine-tuning

```powershell
python examples\make_instruct_data.py --out data_sft --train 6000 --dev 200 --test 400
pico-lm prepare-sft --jsonl data_sft\instruct_train.jsonl --tokenizer data\tokenizer.json --out data_sft
pico-lm finetune --sft-data data_sft --init-from runs\micro\best.pt --out runs\micro-sft `
    --bench-jsonl data_sft\instruct_dev.jsonl --replay-data data --replay-weight 0.1 `
    --max-steps 2000 --lr 1e-4
pico-lm chat --ckpt runs\micro-sft\best.pt
```

Loss is masked to response tokens only, batches are example-aligned, and `best.pt` is
selected on a held-out behavioural score rather than val loss. Replay mixes 10% ordinary
pretraining loss into the objective so a narrow instruction dataset is less likely to
erase the base model's language ability. It costs an extra forward pass; omit
`--replay-data` for the original SFT-only experiment.

**Measured (micro 15.7M, held-out test):** instruction exact match **0.0% → 97.5%** —
and cloze 98.0% → 62.0%, 4-gram repetition 0.011 → 0.833, corpus perplexity 4.7 → 903.
The fine-tune taught the task and destroyed the language model. Bench the general
probes, not just your task.

### Quantization

```powershell
pico-lm quantize --ckpt runs\micro\best.pt --out runs\micro-int8.pt `
    --bits 8 --group-size 64 --quantize-embedding
pico-lm bench --ckpt runs\micro-int8.pt --data data
```

**Measured (micro 15.7M):** int8 is free — 62.96 MB → 16.28 MB (3.9x), val loss
unchanged to four decimals, no probe moves. int4 g64 reaches 8.41 MB (7.5x) for +3.6%
perplexity and −0.5pp exact match. Two warnings: quantization here makes decoding
*slower* (22.4 → 17.1 → 9.7 tok/s), because weights are dequantized on every matmul —
it buys disk and VRAM, not speed. And most of the size is the tied embedding, so
`--quantize-embedding` is where the win is.

### Distillation

```powershell
pico-lm train --preset nano --data data --out runs\nano-kd --teacher runs\micro\best.pt `
    --seq-len 256 --batch-size 32 --grad-accum-steps 2 --max-steps 4000
```

**Measured (micro 15.7M → nano 2.9M, equal token budget vs from-scratch):** cloze
82.5% → 85.5% (z = 1.8, marginal), 4-gram repetition 0.045 → 0.015 (3x less
degenerate). The gain is real but small, the teacher's 97.4% cloze is not approached,
and KD costs 2.3x the wall-clock per token.

> **Sizing on a 6 GB card:** a KD step at `--batch-size 64` peaks at 8.7 GB. Windows
> serves the overflow from system RAM instead of failing, so the run *succeeds* at
> 7.2k tok/s. `--batch-size 32 --grad-accum-steps 2` is the same tokens/step, peaks at
> 4.4 GB, and runs at 34.6k tok/s. Watch throughput, not exit status.

## Other commands

```powershell
pico-lm info --ckpt runs\micro\best.pt   # params, config, training step
pico-lm eval --ckpt runs\micro\best.pt   # loss / perplexity (reference only — prefer bench)
pico-lm runs                             # list training runs
```

## Tests

```powershell
pytest                  # 51 fast tests: model, cache, tokenizer, data, SFT, quantization
pytest -m slow          # 5 end-to-end tokenizer -> prepare -> train -> resume -> generate runs
pytest -m ""            # everything (56)
ruff check .
```

## Layout

```
src/pico_lm/
  config.py     presets + ModelConfig/TrainConfig/SFTConfig
  model.py      Transformer (RMSNorm, RoPE, GQA, SwiGLU, KVCache, generate)
  tokenizer.py  byte-level BPE + chat template
  data.py       corpus -> uint16 memmaps; local files / HF datasets / chat JSONL
  train.py      AMP training loop, cosine LR, checkpoints, distillation
  sft.py        supervised fine-tuning: response-masked loss, behavioural selection
  quant.py      post-training int8/int4 weight-only quantization
  bench.py      behavioural probes (minimal pairs, cloze, generation health, exact match)
  generate.py   completion + chat REPL
  export.py     int8 dynamic quantization (torch.ao, CPU-only, real speedup)
  cli.py        the `pico-lm` command
examples/       make_instruct_data.py — verifiable instruction dataset
experiments/    the scripts that produced every number in docs/posttraining.md
results/        their raw JSON reports
```

## License

[**CC BY-NC-SA 4.0**](https://creativecommons.org/licenses/by-nc-sa/4.0/) — see [LICENSE](LICENSE).

Copyright © 2026 Matheus Soranzo <matheussoranco@gmail.com>
`SPDX-License-Identifier: CC-BY-NC-SA-4.0`

Free to use, study, modify and share for any noncommercial purpose, on three
conditions:

- **NonCommercial** — not primarily for or directed towards commercial advantage
  or monetary compensation (§1(k)).
- **ShareAlike** — anything you share onward must carry these same terms. This
  is a copyleft licence, and it reaches modified versions *and* models you train
  from this code (§3(b)).
- **Attribution** — keep the copyright notice, the licence reference and the
  warranty disclaimer, say if you changed it, and link back where practicable (§3(a)).

Commercial use requires a separate licence from the copyright holder. Note the
licence grants no patent or trademark rights (§2(b)(2)).
