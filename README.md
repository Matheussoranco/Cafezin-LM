# Cafezin LM

Cafezin LM is an experimental PyTorch implementation of dense multimodal Qwen3.5.
The full preset follows Maritaca AI's S5-nano, which is based on Qwen3.5-4B.
[Qwen3.5 architecture](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5)
· [S5-nano](https://huggingface.co/maritaca-ai/S5-nano)

The decoder alternates three Gated DeltaNet layers with one gated grouped-query
attention layer. It uses multimodal rotary position embeddings (mRoPE), a vision
transformer with a 2 × 2 spatial patch merger, and an experimental multi-token
prediction (MTP) head.

| Preset | Parameters | Decoder layers | Vision blocks | Purpose |
| --- | ---: | ---: | ---: | --- |
| `tiny` | 455,808 with 256 embedding rows | 8 | 2 | CPU tests and small text experiments |
| `base` | 4.66 billion | 32 | 24 | S5-nano checkpoint layout and full-size experiments |

## Installation

Use Python 3.11 or 3.12 and run these commands from the repository root.

```sh
python -m pip install -e ".[dev]"
cafezin-lm info --preset tiny
python -m pytest
```

## Train and export a small text checkpoint

Put at least two short paragraphs separated by a blank line in `corpus.txt`.
The preparation command downloads the pinned S5-nano tokenizer. Pass
`--tokenizer path/to/tokenizer.json` to use a local tokenizer instead.

```sh
cafezin-lm prepare --input corpus.txt --out data/cafezin-smoke
cafezin-lm train --preset tiny --data-dir data/cafezin-smoke --out-dir out/cafezin-smoke --device cpu --dtype float32 --seq-len 8 --batch-size 1 --grad-accum 1 --warmup-steps 0 --steps 2
cafezin-lm export --checkpoint out/cafezin-smoke/step-000000002 --out out/cafezin-smoke-export
cafezin-lm generate --checkpoint out/cafezin-smoke-export --device cpu --prompt "Once upon a time" --temperature 0 --max-new 8
```

The `tiny` training command grows the embedding table to fit the prepared
tokenizer. The example checks the workflow with a two-step checkpoint. For a
longer experiment, set `--save-every` to keep intermediate checkpoints. To
continue a run, pass `--resume` the training checkpoint directory and
`--steps` the desired total step count. Resuming with the same training
settings restores the weights, optimizer, scheduler, and PyTorch RNG state.

Export accepts a checkpoint directory or a single SafeTensors file. It
preserves the stored tensor dtypes by default and writes a new directory with
the weight shards, their index, model configuration, generation settings, and
any tokenizer and processor files from the checkpoint. Use `--assets-dir` to
supply assets from the matching tokenizer or processor. Point export at a
fresh output path.

## Reference compatibility

```sh
cafezin-lm verify --offline
cafezin-lm verify
```

Both commands build the `base` preset on the PyTorch meta device, so they
require no full-size weight allocation. The online check fetches the index and
headers from the pinned S5-nano revision. It matches 738 tensor names, shapes,
and storage dtypes, with a tensor payload of 9,319,737,856 bytes.
[Reference revision](https://huggingface.co/maritaca-ai/S5-nano/tree/fe3a3581c33cb22917770ef1987b7c6e30929425)

A `tiny` export was also loaded in Transformers 5.22.2. With identical
weights, the text logits and vision encoder features matched in synthetic FP32
checks. Full-size inference with the published S5-nano weights remains
unverified.

The CLI supports text training and generation. The Python vision forward pass
accepts raster-ordered patch tensors, grid dimensions, and caller-supplied
multimodal positions. Image and video preprocessing and end-to-end multimodal
generation remain experimental.

## Weights and license

Cafezin LM source is licensed under [Apache License 2.0](LICENSE). See [NOTICE](NOTICE)
for the project copyright. This repository contains code and tests. Maritaca AI
publishes the S5-nano checkpoint and tokenizer separately under Apache 2.0.
[Upstream model card](https://huggingface.co/maritaca-ai/S5-nano)

Checkpoints from the pre-Qwen decoder use a different architecture.
For a Cafezin LM weight release, export the trained checkpoint with its matching
assets, reload it in a clean environment, and document the source of the
weights, training data, license, evaluation, and limitations in a model card.
[Hugging Face model cards](https://huggingface.co/docs/hub/model-cards)
