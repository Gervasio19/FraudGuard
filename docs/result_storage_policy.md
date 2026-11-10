# Result Storage and Git Tracking Policy

> **Status:** approved by the project owner in the Phase 5B-0 briefing (2026-11-11). Applies to every result file written from Phase 5B onward. Phase 4 files are unchanged (see "Existing files").

## What is tracked in Git

| Tracked | Examples |
|---|---|
| Result summaries (compact JSON) | `reports/phase4b_baseline_results.json`, `reports/phase4c_threshold_results.json`, future `reports/phase5b_*.json` |
| Environment manifest | `reports/environment_manifest.json` |
| Freeze manifests and their template | `docs/templates/phase5_freeze_manifest.template.json`; a frozen copy committed before any test scoring (protocol Section 4.2, step 9) |
| Requirements and reproducibility documentation | `requirements.txt`, `requirements-notebook.txt`, `README.md`, `data/README.md`, this file |
| Source code, tests, documentation | `src/`, `tests/`, `docs/` |
| Empty-directory placeholders | `data/raw/.gitkeep`, `data/processed/.gitkeep`, `artifacts/.gitkeep`, `reports/figures/.gitkeep` |

## What is never tracked

| Not tracked | How it is enforced |
|---|---|
| Raw dataset (`data/raw/creditcard.csv`) and processed data | `.gitignore`: `/data/raw/*`, `/data/processed/*` |
| Fitted models | `.gitignore`: `/artifacts/*`, `*.pkl`, `*.pickle`, `*.joblib`, `*.h5`, `*.keras`, `*.pt`, `*.pth`, `*.onnx` |
| Large intermediate arrays (prediction or score vectors) | `.gitignore`: `*.npy`, `*.npz`, `*.parquet`, `*.feather` |
| Local caches, environments, machine-specific files | `.gitignore`: `.venv/`, `venv/`, `env/`, `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `.ipynb_checkpoints/`, `.env*`, IDE and OS files |

The dataset is never committed, uploaded or published as a downloadable artifact. Users obtain it from Kaggle (`data/README.md`).

## Content rules for result JSON

1. **Strict JSON:** NaN and infinity are written as `null`. Use `src.reproducibility.manifest.write_result_json`, which enforces this rule and the next two.
2. **No absolute paths:** paths are repository-relative (POSIX style). The writer refuses absolute or home-relative paths.
3. **Compact summaries:** store metrics, counts, configurations, decisions and the uncertainty summaries the protocol requires. Do **not** store per-row prediction or score vectors, per-replicate bootstrap arrays, or full threshold curves unless a protocol section needs them. Scores are recomputed deterministically from code (protocol Section 12.2).
4. **Provenance:** each file records the dataset SHA-256, the Git commit, the environment versions and the seeds (as Phase 4 files do).
5. **Separation:** a result file holds results. A freeze manifest holds frozen configuration and decisions, and points to result files; it never holds metric values.

## Size limits

1 MB = 1,000,000 bytes.

| Size | Status | Action |
|---|---|---|
| ≤ 5 MB | `ok` (target) | none |
| > 5 MB and ≤ 10 MB | `over_target` | allowed; the relevant phase report must explain why the size is needed |
| > 10 MB | `over_hard_limit` | refused by `write_result_json` unless an explicit `size_exception_reason` is given; the report must state and justify the exception |

Check files with `python -m src.reproducibility.manifest check-size reports/*.json` (run from the repository root).

## Existing files

- `reports/phase4b_baseline_results.json` (5,554 bytes): within the target.
- `reports/phase4c_threshold_results.json` (9,495,967 bytes at Phase 5B-0): **above the 5 MB target, below the 10 MB hard limit.** It predates this policy, stores three full threshold curves, and stays tracked by the project-owner decision of 2026-11-08. It is not rewritten.
