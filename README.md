# Pico-LM — a small, highly capable language model, trained and run locally

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](LICENSE)

A from-scratch small language model stack in PyTorch, sized for a laptop GPU
(tuned for an RTX 3050 6 GB). Same architecture family as Llama/Mistral, scaled down:

- **Decoder-only transformer** — RMSNorm (pre-norm), rotary position embeddings (RoPE),
  grouped-query attention (GQA), SwiGLU MLP, weight tying, flash attention via PyTorch SDPA
- **Your own byte-level BPE tokenizer** trained on your corpus (16k vocab by default)
- **Training loop** — bf16 mixed precision, gradient accumulation, cosine LR with warmup,
  gradient clipping, checkpoint/resume, CSV logging
- **Knowledge distillation** — train a `base` teacher, distill it into a smaller student
- **Fast inference** — preallocated KV cache, top-k / top-p / temperature / repetition penalty,
  int8 dynamic quantization for CPU
- **Chat** — special-token chat template + SFT via chat-formatted JSONL

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

(`pico-lm presets` prints these live, computed from the actual model.)

For `base` on 6 GB, use `--batch-size 8 --grad-accum-steps 8`.

### Training flags

`train` takes `--preset --data --out --tokenizer --resume --teacher --compile --seq-len
--batch-size --grad-accum-steps --max-steps --lr --min-lr --warmup-steps --eval-interval
--eval-iters --log-interval --seed`. Anything you leave out falls back to `TrainConfig`
in `config.py`. `pico-lm train --help` is the authority.

## Distillation

```powershell
pico-lm train --preset base  --data data --out runs\base
pico-lm train --preset small --data data --out runs\small-kd --teacher runs\base\best.pt
```

## Chat / SFT

Fine-tune on conversations (JSONL, one `{"messages": [{"role": ..., "content": ...}]}` per line),
then chat. Pass the pretraining tokenizer explicitly so the vocabulary matches:

```powershell
pico-lm prepare --chat-jsonl chats.jsonl --tokenizer data\tokenizer.json --out data_sft
copy runs\micro\best.pt runs\micro-sft\last.pt
pico-lm train --preset micro --data data_sft --out runs\micro-sft --resume
pico-lm chat --ckpt runs\micro-sft\best.pt --system "You are a helpful assistant."
```

## Other commands

```powershell
pico-lm info --ckpt runs\micro\best.pt          # params, config, training step
pico-lm quantize --ckpt runs\micro\best.pt --out runs\micro-int8.pt   # 4x smaller, CPU inference
```

## Tests

```powershell
pytest                  # 22 fast tests: model math, KV-cache equivalence, tokenizer, data
pytest -m slow          # 4 more: full tokenizer -> prepare -> train -> resume -> generate runs
pytest -m ""            # everything (26)
ruff check .
```

## Layout

```
src/pico_lm/
  config.py     presets + ModelConfig/TrainConfig
  model.py      Transformer (RMSNorm, RoPE, GQA, SwiGLU, KVCache, generate)
  tokenizer.py  byte-level BPE + chat template
  data.py       corpus -> uint16 memmaps; local files / HF datasets / chat JSONL
  train.py      AMP training loop, cosine LR, checkpoints, distillation
  generate.py   completion + chat REPL
  export.py     int8 dynamic quantization
  cli.py        the `pico-lm` command
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
