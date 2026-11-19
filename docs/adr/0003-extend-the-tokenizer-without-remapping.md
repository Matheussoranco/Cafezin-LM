# Extend the tokenizer without remapping existing IDs

Keep the pinned tokenizer as the base instead of training an incompatible replacement. Mine candidate pieces from Brazilian Portuguese data, but choose the extension budget only after measuring held-out tokenization gains and checking for regressions on multilingual text; a vocabulary diff alone does not justify adding a token.

The checkpoint initialization tool should preserve existing token IDs, special-token behavior, and embedding rows; initialize new rows from the mean of their decomposition under the original tokenizer; and preserve the tied output projection. Export the adapted tokenizer and weights together. An expanded embedding table is not an exact match for the upstream fingerprint: verify upstream compatibility against the unchanged reference preset, and verify the adapted checkpoint against its own configuration and tokenizer.
