## Summary
Ruff reports 387 linting errors across the codebase introduced in the Portuguese specialization track. Many are fixable with `ruff check --fix`.

## Categories of Violations

### E501 Line too long (>100 chars) - 100+ occurrences
Files affected:
- scripts/compare_models.py (5 violations)
- scripts/curate_portuguese_corpus.py (1)
- scripts/data_curation/corpus_assembler.py (5)
- scripts/data_curation/curriculum.py (1)
- scripts/data_curation/tokenizer_utils.py (3)
- src/cafezin_lm/cli.py (8+)
- src/cafezin_lm/model_card.py (20+)
- tests/test_data_curation.py (1)
- tests/test_generative_eval.py (8)
- tests/test_tokenizer_extension.py (4)

### E402 Module level import not at top of file
- scripts/curate_portuguese_corpus.py: 3 violations (lines 25, 26, 28)

### I001 Import block un-sorted/un-formatted
- scripts/curate_portuguese_corpus.py
- scripts/data_curation/__init__.py
- scripts/data_curation/tokenizer_utils.py
- src/cafezin_lm/cli.py
- src/cafezin_lm/model_card.py
- tests/test_data_curation.py
- tests/test_generative_eval.py
- tests/test_tokenizer_extension.py

### E401 Unused imports
- Multiple files (F401): json, tempfile, sys, CurriculumStage, assemble_portuguese_corpus, etc.

### W292/W293 Missing trailing newline / whitespace on blank lines
- Many files missing trailing newline
- src/cafezin_lm/model_card.py: 15+ W293 violations (whitespace on blank lines)

### UP037 Remove quotes from type annotations
- Multiple files using quoted type annotations that could be unquoted (Python 3.12+)

### UP017/UP035/UP042 Modern Python improvements
- datetime.UTC alias instead of timezone.utc
- Import Iterator from collections.abc
- Use StrEnum instead of str + Enum

### Other violations
- SIM102: Nested if statements could be combined
- B905: zip() without explicit strict=
- B007: Unused loop variable
- F841: Unused variable assignment
- F821: Undefined name (CurriculumSchedule)
- RUF022: __all__ not sorted
- F401: Unused imports

## Action
Run `ruff check --fix` to auto-fix 216 violations. Remaining 171 require manual fixes.