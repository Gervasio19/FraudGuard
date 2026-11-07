# FraudGuard Project Context

## Verified repository facts

- FraudGuard is an educational credit-card fraud detection project.
- Current project phase: **Phase 4 — Model Development & Evaluation, in progress.** Phase 4a (chronological partitioning) is committed in `37d4521`. Phase 4b (baseline modeling) is implemented, tested and uncommitted. Its three open decisions were approved by the mentor on 2026-11-07, and the final verification has passed. **Phase 4b completion awaits independent mentor approval.** Phase 4c has not begun.
- Documentation: `README.md` (overview and usage), this file (architecture and methodology), `docs/WORK_LOG.md` (chronological history and experiment results), `docs/TECHNICAL_DOCUMENTATION.md` (component reference), and `docs/FINAL_REPORT.md` (pending placeholder). Topic documents: `docs/business_understanding.md` and `docs/validation_strategy.md`.
- Data documentation expects `data/raw/creditcard.csv`. Raw data is ignored by Git and must remain unchanged.
- Requirements listed in `requirements.txt`: pandas, NumPy, Matplotlib, Seaborn, Jupyter, ipykernel and scikit-learn (1.9.1 installed in the designated environment).
- Tests run from the repository root with `/Users/duc/Desktop/env/bin/python -m unittest discover -s tests -v`. They use the standard-library `unittest`; pytest is not installed or required.
- The designated environment is `/Users/duc/Desktop/env/bin/python` (Python 3.13.6, pandas 2.3.1, NumPy 2.3.2, scikit-learn 1.9.1).

## Architecture (as of Phase 4b)

| Concern | Module |
|---|---|
| Dataset column roles | `src/config.py`: `DatasetConfig` and the `CREDITCARD` instance |
| Chronological split (primary) | `src/data/split.py` (Phase 4a, unchanged) |
| Stratified random split with duplicate groups kept together (secondary) | `src/data/random_split.py` |
| Preprocessing | `src/features/preprocessing.py`: `StandardScaler` inside the model pipeline |
| Baseline models | `src/models/baselines.py` |
| Metrics | `src/evaluation/metrics.py` |
| Train/evaluate runner | `src/models/experiment.py`; CLI `python -m src.models.run_baselines` |

Both splits return the same ordered categorical `train`/`validation`/`test` labels, so they plug into the same runner. Preprocessing lives inside each scikit-learn `Pipeline`, so `fit` on the training rows is the only place any statistic is learned. The runner refuses to evaluate on the test partition. Details: `docs/TECHNICAL_DOCUMENTATION.md`.

## Confirmed validation decisions

- Primary validation: chronological 70/15/15, with all rows sharing a timestamp assigned together.
- Train: `Time` 0–132928; 199,364 rows; 384 frauds.
- Validation: `Time` 132929–151328; 42,722 rows; 56 frauds.
- Test: `Time` 151329–172792; 42,721 rows; 52 frauds.
- Keep all duplicate records. Identical feature groups must remain within a single partition. The inspected data had 773 repeated feature groups, 1,081 excess rows, consistent labels, and zero groups crossing the confirmed chronological partitions.
- Secondary benchmark: a stratified random split that keeps duplicate feature groups together. It is reported separately and never pooled with chronological results.
- No classification threshold is selected.
- Fit preprocessing on training data only. Do not use validation or test outcomes to select models, thresholds or hyperparameters. The test partition is reserved for final evaluation.

## Approved Phase 4b decisions (mentor approval, 2026-11-07)

These were proposed in the Phase 4b implementation report and approved by the project mentor. They are now project methodology.

1. **Raw `Time` is excluded from the baseline model features.** `Time` is still used for chronological splitting and duplicate grouping. Reason: raw `Time` is elapsed seconds within this specific 48-hour dataset, and the chronological validation period lies entirely outside the training time range. This is *not* a claim that temporal information has no predictive value. Temporal feature engineering (for example, derived time-of-day features) remains a possible future research direction. The baseline features are `V1`–`V28` and `Amount`.
2. **Stratified random split (secondary benchmark):**
   - 70/15/15 train/validation/test, `random_state=42`, stratified by the target class.
   - Duplicate groups are defined by all columns except `Class` and never cross partitions.
   - All original rows are preserved; no raw duplicates are removed.
   - Chronological splitting remains the primary evaluation strategy.
3. **Default 0.5 rule for diagnostic reporting only.**
   - Average precision and ROC-AUC are *ranking metrics*, computed from predicted probabilities.
   - Precision, recall and the confusion matrix are *diagnostic metrics* at each estimator's default `predict()` rule: probability ≥ 0.5 for Logistic Regression, and always the majority class for the majority baseline.
   - 0.5 is not an optimal or production threshold, and no threshold has been selected.

Phase 4b results are recorded in `docs/WORK_LOG.md` and `reports/phase4b_baseline_results.json`.

## Scope and limits

No feature engineering or hyperparameter tuning has been performed. The data covers about 48 hours and has few fraud cases in validation (56 chronological, 74 random) and test. No confidence intervals are reported yet. `V1`–`V28` are anonymized PCA components. Label availability timing and the intended operational prediction point remain unspecified. The chronological boundaries in `src/data/split.py` are specific to this dataset. Refer to `docs/validation_strategy.md` for detailed design and leakage controls, and `docs/business_understanding.md` for business framing.

## Handoff state and next task

Phase 4b is implemented on top of `37d4521` on branch `main` and is uncommitted. Its decisions are approved and its final verification has passed (see `docs/WORK_LOG.md`). Next: independent mentor approval of Phase 4b completion. Do not begin Phase 4c until Phase 4b is approved. Keep the test partition untouched until final evaluation. Threshold selection requires later operational criteria. See `docs/WORK_LOG.md` for details.

## Assumptions

No additional project assumptions are asserted here. Dataset counts and split behavior above are verified by automated real-data tests (`tests/test_split.py`, `tests/test_random_split.py`, `tests/test_baselines.py`). Recheck them if the local CSV changes; its SHA-256 is recorded in `docs/WORK_LOG.md`.
