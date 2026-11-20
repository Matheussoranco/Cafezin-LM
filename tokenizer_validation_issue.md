## Summary
Tokenizer extension infrastructure exists but not validated on real Brazilian Portuguese corpora (DS-002 gap).

## Spec Requirement
SPEC.md line 40: "measure the vocabulary gap between S5-nano's 248K tokenizer and Portuguese corpora, add only high-value tokens, initialize new embedding rows from the mean of their decomposition under the original tokenizer, and export the extended tokenizer with matching weights"

## Current State
- **Infrastructure complete**: `src/cafezin_lm/tokenizer_extension.py` has:
  - `mine_candidate_tokens()` - mines candidate tokens from corpus
  - `measure_tokenization_stats()` - measures tokenization statistics
  - `extend_tokenizer()` - extends tokenizer preserving existing IDs
  - `mean_decomposition_init()` - initializes new embeddings from mean decomposition
  - `grow_embeddings()` - grows model embeddings preserving tied lm_head
  - `export_extended_checkpoint()` - exports matched tokenizer/checkpoint pair
  - `verify_round_trip()` - verifies consistency

- **CLI commands**: `tokenizer-extend`, `tokenizer-stats`

- **Tests**: 8 tests in `tests/test_tokenizer_extension.py` passing

## Gap
The infrastructure has only been tested on small fixture corpora (3-4 short paragraphs). Not run on actual Brazilian Portuguese corpora:
- mc4-pt
- OSCAR-pt  
- Brazilian Common Crawl
- Portuguese Wikipedia

## Required Work
1. Download/prepare actual Brazilian Portuguese corpora
2. Run `tokenizer-stats` to measure vocabulary gap and tokenization gains
3. Run `tokenizer-extend` with appropriate `--min-frequency` and `--max-candidates`
4. Validate extended tokenizer on held-out Portuguese text
5. Export extended tokenizer and checkpoint for continued pretraining

## Dependencies
- Blocked by: Data curation (#11) for corpus availability
- Blocks: Continued pretraining (#12) needs extended tokenizer