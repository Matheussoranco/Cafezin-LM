# slm — a small, highly capable language model, trained and run locally

A from-scratch small language model (SLM) stack in PyTorch, sized for a laptop GPU
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
3. **Distillation**: train the biggest model that fits (`base`, ~110M), then distill it
   into a smaller, faster student with `--teacher`.

## Setup

```powershell
cd C:\Users\mathe\Documents\Code\LLM
.venv\Scripts\activate            # venv has CUDA PyTorch installed
pip install -e .[data,dev]
```

## Quickstart: train on TinyStories (recommended first run)

```powershell
# 1. Train a tokenizer (streams the dataset; ~2M docs)
slm tokenizer --hf roneneldan/TinyStories --vocab-size 16384 --out data\tokenizer.json --max-docs 500000

# 2. Tokenize into train/val memmaps
slm prepare --hf roneneldan/TinyStories --tokenizer data\tokenizer.json --out data --max-docs 500000

# 3. Train (micro ≈ 22M params; a few hours on the 3050)
slm train --preset micro --data data --out runs\micro --batch-size 32 --grad-accum-steps 2 --seq-len 512

# 4. Generate
slm generate --ckpt runs\micro\best.pt --prompt "Once upon a time"
```

Or use your own text files: `--input corpus\*.txt` instead of `--hf ...`.

## Presets (16k vocab, sized for 6 GB VRAM)

| preset | params | dim | layers | heads (kv) | context |
|--------|--------|-----|--------|------------|---------|
| nano   | ~3M    | 128 | 4      | 4 (2)      | 256     |
| micro  | ~22M   | 384 | 6      | 6 (2)      | 512     |
| small  | ~42M   | 512 | 8      | 8 (4)      | 1024    |
| base   | ~110M  | 768 | 12     | 12 (4)     | 1024    |

For `base` on 6 GB, use `--batch-size 8 --grad-accum-steps 8`.

## Distillation

```powershell
slm train --preset base  --data data --out runs\base
slm train --preset small --data data --out runs\small-kd --teacher runs\base\best.pt
```

## Chat / SFT

Fine-tune on conversations (JSONL, one `{"messages": [{"role": ..., "content": ...}]}` per line),
then chat:

```powershell
slm prepare --chat-jsonl chats.jsonl --tokenizer data\tokenizer.json --out data_sft
slm train --preset micro --data data_sft --out runs\micro-sft --resume   # start from a pretrained ckpt copied to runs\micro-sft\last.pt
slm chat --ckpt runs\micro-sft\best.pt --system "You are a helpful assistant."
```

## Other commands

```powershell
slm info --ckpt runs\micro\best.pt          # params, config, training step
slm quantize --ckpt runs\micro\best.pt --out runs\micro-int8.pt   # 4x smaller, CPU inference
```

## Tests

```powershell
pytest            # 20 tests: model math, KV-cache equivalence, tokenizer, data, end-to-end
```

## Layout

```
src/slm/
  config.py     presets + ModelConfig/TrainConfig
  model.py      Transformer (RMSNorm, RoPE, GQA, SwiGLU, KVCache, generate)
  tokenizer.py  byte-level BPE + chat template
  data.py       corpus -> uint16 memmaps; local files / HF datasets / chat JSONL
  train.py      AMP training loop, cosine LR, checkpoints, distillation
  generate.py   completion + chat REPL
  export.py     int8 dynamic quantization
  cli.py        the `slm` command
```
