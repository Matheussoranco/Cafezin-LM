## Code Review Summary: cafezin-lm Portuguese Specialization Track

**Fixed point**: `d3a51e6` (Add Model Tensor Specification) → `HEAD` (bc14b87)
**Diff size**: 15,261 insertions, 16 deletions across 46 files

---

## Standards Review (387 ruff violations)

### Hard Violations (auto-fixable: 216)
- **E501 Line too long** (>100 chars): 100+ occurrences across 15 files
- **E402** Import not at top: 3 in `curate_portuguese_corpus.py`
- **I001** Import blocks un-sorted: 8 files
- **E401** Unused imports: 15+ across tests/scripts
- **W292/W293** Missing newlines/whitespace: 30+ files
- **UP037** Quoted type annotations: 20+ (Python 3.12+ supports unquoted)
- **UP017/UP035/UP042**: Modern Python improvements needed
- **RUF022**: `__all__` not sorted in 2 files
- **F841/F821/F401**: Unused/undefined names

### Fowler Smells (Judgement Calls)
1. **Shotgun Surgery** - CLI changes touch 8+ files per new command
2. **Divergent Change** - `cli.py` handles 14 commands
3. **Primitive Obsession** - Raw dicts for provenance/curriculum
4. **Duplicated Code** - Provenance/tokenization patterns in 3 data_curation modules
5. **Feature Envy** - `grow_embeddings()` reaches into model internals
6. **Message Chains** - `model_card.py` long attribute chains
7. **Middle Man** - `cli.py` thin wrapper over 14 commands

**Action**: Run `ruff check --fix` (fixes 216). Remaining 171 need manual attention.

---

## Spec Review (SPEC.md compliance)

### ✅ Fully Implemented
- Continued pretraining pipeline (MS-001) - CLI complete
- Tokenizer extension infrastructure (DS-002) - complete with tests
- Data curation pipeline (DS-001) - curriculum, provenance, tokenization
- lm-eval harness with PT tasks (AS-002) - complete
- Generative eval framework (AS-002) - prompts, rubric, heuristic judge
- Model card template (AS-003) - complete with all sections
- HF upload script (AS-003) - validation, smoke test, upload
- Vision norm test (AS-001) - script complete, runs against HF

### ⚠️ Partial / Known Gaps
| Requirement | Spec Line | Status |
|-------------|-----------|--------|
| Vision norm within 1e-3 | 48 | **FAIL** - diff=0.625 vs HF |
| Tokenizer on real BR corpus | 40 | Infrastructure only, not run |
| GPU training validation | 44 | CPU smoke test only |
| LLM judge for generative | 52 | Heuristic fallback only |
| Model card with real data | 56 | Placeholder values only |

### 📋 Out of Scope (Correctly Not Implemented)
- Full pretraining from scratch
- LoRA/QLoRA fine-tuning
- New architecture experiments
- Multi-GPU/distributed training
- Vision preprocessing pipeline
- Instruction tuning / RLHF

### 📦 Scope Creep (Not in Spec)
- `bandit_report.json` (3866 lines)
- `uv.lock` (3021 lines)
- `test_model_card.md`, `tickets.json`
- `AGENTS.md`, `docs/agents/` (SAIL skills)
- `docs/cafezin-open-weight-spec.md` (duplicate)
- Test outputs: `test_vision_norm_outputs/`, `test_model_card.md`

---

## Issues Created
1. [#18](https://github.com/Matheussoranco/Cafezin-LM/issues/18) - Standards: Fix ruff linting violations
2. [#19](https://github.com/Matheussoranco/Cafezin-LM/issues/19) - Spec: Gaps between SPEC.md and implementation
3. [#20](https://github.com/Matheussoranco/Cafezin-LM/issues/20) - Standards: Fowler code smells
4. [#21](https://github.com/Matheussoranco/Cafezin-LM/issues/21) - Spec: Vision norm unresolved (AS-001)
5. [#22](https://github.com/Matheussoranco/Cafezin-LM/issues/22) - Spec: Tokenizer not validated on real corpus (DS-002)
6. [#23](https://github.com/Matheussoranco/Cafezin-LM/issues/23) - Spec: Pretraining not on 48GB GPU (MS-001)
7. [#24](https://github.com/Matheussoranco/Cafezin-LM/issues/24) - Spec: Generative eval heuristic only (AS-002)
8. [#25](https://github.com/Matheussoranco/Cafezin-LM/issues/25) - Spec: Model card placeholders (AS-003)
9. [#26](https://github.com/Matheussoranco/Cafezin-LM/issues/26) - Spec: Out-of-scope artifacts committed

---

## Next Steps Priority
1. **Run `ruff check --fix`** - fixes 216 lint violations immediately
2. **Resolve vision norm** (Issue #21) - blocks vision parity claim
3. **Run tokenizer on real corpus** (Issue #22) - prerequisite for training
4. **GPU training run** (Issue #23) - validate 48GB config
5. **Add LLM judge** (Issue #24) - complete generative eval
6. **Clean up scope creep** (Issue #26) - remove artifacts, add .gitignore