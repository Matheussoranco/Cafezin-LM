## Summary
Several artifacts were added during implementation that are not in the original SPEC.md scope.

## Out-of-Scope Artifacts Added

| File | Lines | Description |
|------|-------|-------------|
| `bandit_report.json` | 3866 | Security scan output - not in spec |
| `uv.lock` | 3021 | uv lock file - not in spec (uses pip) |
| `test_model_card.md` | 255 | Test output - not in spec |
| `tickets.json` | 42 | Internal tracking - not in spec |
| `AGENTS.md` | 17 | SAIL skills config - not in spec |
| `docs/agents/` (3 files) | 98 | SAIL skills setup - not in spec |
| `docs/cafezin-open-weight-spec.md` | 66 | Duplicate of SPEC.md |
| `scripts/__init__.py` | 1 | Empty package init |
| `corpus.txt` | 3 | Test fixture |
| `test_vision_norm_outputs/metadata.json` | 62 | Test output artifact |

## Spec Reference
SPEC.md "Out of Scope" section (lines 85-92) explicitly lists:
- Full pretraining from scratch
- LoRA/QLoRA fine-tuning
- New architecture experiments
- Multi-GPU/distributed training
- Vision preprocessing pipeline
- Instruction tuning / RLHF

But does NOT mention: SAIL skills setup, bandit reports, uv lock files, test outputs, internal tracking files.

## Action Items
1. **Keep**: ADRs (0001-0005) - explicitly required by spec
2. **Keep**: SPEC.md - the spec itself
3. **Review**: SAIL skills setup (AGENTS.md, docs/agents/) - useful for agents but not in spec
3. **Remove/Clean**: bandit_report.json, uv.lock, test_model_card.md, tickets.json, scripts/__init__.py, corpus.txt, test_vision_norm_outputs/
4. **Consolidate**: docs/cafezin-open-weight-spec.md duplicates SPEC.md

## Recommendation
Add a `.gitignore` entry for build/test artifacts:
```
bandit_report.json
uv.lock
test_model_card.md
tickets.json
test_vision_norm_outputs/
corpus.txt
*.lock
__pycache__/
.pytest_cache/
.ruff_cache/
.mypy_cache/
```

And document SAIL skills setup in CONTRIBUTING.md or README.md instead of committing config files.