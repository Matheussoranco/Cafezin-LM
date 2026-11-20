## Summary
Fowler code smells identified in the diff (judgement calls, not hard violations). The repo's ruff config overrides baseline where applicable.

## Identified Smells

### 1. Duplicated Code
- **Location**: `scripts/data_curation/` modules - provenance tracking, tokenization reporting patterns repeated across `corpus_assembler.py`, `curriculum.py`, `tokenizer_utils.py`
- **Fix**: Extract common provenance/tokenization reporting into shared utilities

### 2. Feature Envy
- **Location**: `src/cafezin_lm/tokenizer_extension.py` - `grow_embeddings()` reaches into `model.language_model.embed_tokens` and `model.lm_head`
- **Fix**: Consider adding model methods for embedding growth

### 3. Data Clumps
- **Location**: `CurriculumSchedule`, `CurriculumStage`, `CurriculumStage` appear together across `curriculum.py`, `tokenizer_utils.py`, `curate_portuguese_corpus.py`
- **Fix**: Already partially addressed by `CurriculumSchedule` dataclass

### 4. Primitive Obsession
- **Location**: Heavy use of raw `dict` for provenance (`CorpusProvenance.to_dict()`), curriculum config instead of typed classes throughout `data_curation/`
- **Fix**: Already partially addressed by dataclasses, but provenance still uses dict serialization

### 5. Repeated Switches
- **Location**: `scripts/data_curation/corpus_assembler.py` lines 318, 341, 352 - repeated `if key in criteria:` patterns
- **Fix**: Extract criteria matching logic

### 6. Shotgun Surgery
- **Location**: Adding tokenizer extension required edits across `cli.py`, `tokenizer_extension.py`, `tests/`, `__init__.py`, `model.py`
- **Fix**: Consider plugin architecture for CLI commands

### 7. Divergent Change
- **Location**: `src/cafezin_lm/cli.py` handles 14 different commands (spec, verify, prepare, train, export, info, generate, tokenizer-extend, tokenizer-stats, continue-pretrain, download-s5nano, smoke-test, model-card)
- **Fix**: Consider subcommand modules

### 8. Speculative Generality
- **Location**: `scripts/data_curation/corpus_assembler.py` - `CorpusSource`, `CorpusProvenance` abstract classes with single implementation
- **Fix**: Simplify to concrete classes if no extension planned

### 9. Message Chains
- **Location**: `src/cafezin_lm/model_card.py` - `data.evaluation.baseline_bits_per_byte`, `data.evaluation.candidate_bits_per_byte`, etc.
- **Fix**: Add accessor methods or properties

### 10. Middle Man
- **Location**: `src/cafezin_lm/cli.py` - thin wrapper over module functions (14 commands)
- **Fix**: Consider direct module entry points or plugin system

## Action
These are judgement calls. Prioritize fixes that affect maintainability: #6 (shotgun surgery for CLI), #7 (divergent change in cli.py), #4 (primitive obsession in provenance).