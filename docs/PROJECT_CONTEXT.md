# FraudGuard Project Context

## Verified repository facts

- FraudGuard is an educational credit-card fraud detection project.
- Current project phase: **Phase 4 — Model Development & Evaluation, in progress.** Phase 4a (reusable chronological partitioning module and tests) is implemented and awaiting human review. Phase 4b (baseline modeling) has not begun and requires explicit approval.
- Main documentation: `README.md`, `docs/business_understanding.md`, and `docs/validation_strategy.md`.
- Data documentation expects `data/raw/creditcard.csv`. A local CSV was inspected in the prior work; raw data is ignored by Git and must remain unchanged.
- Requirements listed in `requirements.txt`: pandas, NumPy, Matplotlib, Seaborn, Jupyter, ipykernel, and scikit-learn (added in Phase 4a for the upcoming modeling work; version 1.9.1 is installed in the designated environment).
- Partitioning code: `src/data/split.py`. Tests: `tests/test_split.py`, run from the repository root with `/Users/duc/Desktop/env/bin/python -m unittest discover -s tests -v` (standard-library `unittest`; pytest is not installed or required).
- The designated environment used for Phase 3 checks is `/Users/duc/Desktop/env/bin/python` (Python 3.13.6, pandas 2.3.1 in the recorded run).

## Confirmed validation decisions

- Primary validation: chronological 70/15/15, with all rows sharing a timestamp assigned together.
- Train: `Time` 0–132928; 199,364 rows; 384 frauds.
- Validation: `Time` 132929–151328; 42,722 rows; 56 frauds.
- Test: `Time` 151329–172792; 42,721 rows; 52 frauds.
- Keep all duplicate records. Identical feature groups must remain within a single partition. The inspected data had 773 repeated feature groups, 1,081 excess rows, consistent labels, and zero groups crossing the confirmed chronological partitions.
- Secondary benchmark: stratified random split that keeps duplicate feature groups together.
- No classification threshold is selected.

## Scope and limits

No feature engineering or model training has been performed. The secondary random benchmark split is not implemented (out of scope for Phase 4a). The data covers about 48 hours and has few fraud cases in validation and test. `V1`–`V28` are anonymized PCA components. Label availability timing and the intended operational prediction point remain unspecified. Refer to `docs/validation_strategy.md` for detailed design and leakage controls, and `docs/business_understanding.md` for business framing and evaluation considerations.

## Handoff state and next task

The handoff documentation consists of exactly two files: this file and `docs/WORK_LOG.md`. They were committed in `c0ecd3c` ("Add AI handoff protocol and project continuity docs"); despite that commit message, no separate protocol file exists in the repository.

Phase 4a was implemented on top of `c0ecd3c` on branch `main` and is uncommitted, pending human review. Recommended next task (only after approval): Phase 4b — establish a simple baseline using the training partition only and report class-sensitive validation metrics; keep the test partition untouched until final evaluation. Threshold selection requires later operational criteria. See `docs/WORK_LOG.md` for details.

## Assumptions

No additional project assumptions are asserted here. Dataset counts and split behavior above are verified results documented by the prior inspection; recheck against the local CSV before implementing Phase 4 if data availability or content differs.
