## Summary
Vision normalization ambiguity (AS-001) remains unresolved - cannot claim vision parity with S5-nano.

## Details
- **Spec requirement** (SPEC.md line 48): "record which setting matches within 1e-3 max absolute difference"
- **Implementation**: `scripts/test_vision_norm.py` runs against HF transformers reference
- **Result**: `zero_centered_norm=False` winner with max_diff=0.625 vs HF reference
- **Status**: NOT within 1e-3 precision threshold

## Test Output
```
>>> WINNER: zero_centered_norm=False (diff=0.625000)
>>> FAIL: diff > 1e-3
```

## Root Cause
The difference between `zero_centered_norm=False` (default, matches HF port) and `zero_centered_norm=True` (matches checkpoint storage pattern) is 23.75 max_diff between them. Neither matches HF reference within 1e-3.

## Possible Causes
1. HF transformers Qwen3.5-VL implementation may have different vision tower forward pass
2. Input preprocessing difference (pixel_values format, grid_thw handling)
3. Weight loading mismatch for vision tower
4. GELU approximation difference (`gelu_pytorch_tanh` vs default)

## Blockers
- Cannot claim vision parity with S5-nano checkpoint
- Model card must state "vision normalization ambiguity unresolved"
- Multimodal outputs marked experimental

## Next Steps
1. Investigate HF transformers Qwen3.5-VL vision forward pass implementation
2. Compare pixel_values preprocessing pipeline
3. Try different GELU approximations
4. Consider reaching out to Maritaca AI for clarification