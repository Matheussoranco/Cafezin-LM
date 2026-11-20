## Summary
Continued pretraining pipeline (MS-001) implemented but not validated on target 48GB GPU hardware.

## Spec Requirement
SPEC.md line 44: "train with `seq_len=2048` baseline / `4096` stretch on single 48 GB GPU using gradient accumulation=32, gradient checkpointing, and MTP weight=0.3"

## Current State
- **Pipeline complete**: `src/cafezin_lm/continued_pretrain.py` with:
  - `ContinuedPretrainConfig` with MS-001 defaults
  - `continued_pretrain()` function: loads S5-nano → grows embeddings → fresh Adam → cosine decay with warmup
  - `smoke_test_cpu()` for clean reload + generation on CPU
  - CLI commands: `continue-pretrain`, `download-s5nano`, `smoke-test`

- **Tests**: Existing test suite passes (73 tests)

## Gap
Only CPU smoke test (`--smoke-test`) has been run. No training on target 48GB GPU to validate:
- Peak memory usage with `seq_len=2048`, `batch_size=1`, `grad_accum=32`
- Throughput (tokens/sec)
- Whether `seq_len=4096` stretch target fits in 48GB
- Actual training stability over multiple steps

## Required Work
1. Access 48GB GPU (A100 40GB or A6000 48GB)
2. Run `download-s5nano` to cache checkpoint
3. Run `tokenizer-extend` to create extended tokenizer (requires Brazilian corpus first)
4. Run `continue-pretrain` with baseline config for several thousand steps
5. Monitor peak memory, throughput, loss curves
6. Validate checkpoint export + reload + generation

## Dependencies
- Blocked by: #22 (tokenizer extension on real corpus)
- Blocks: Model card rendering (#16), HF release (#17)