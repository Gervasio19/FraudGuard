# FraudGuard Project Context

## Verified repository facts

- FraudGuard is an educational credit-card fraud detection project.
- Current project phase: **Phase 5 — methodology and design briefing complete; the design decisions are approved (recorded 2026-11-08) but not implemented.** The Phase 5 design documentation has been reviewed and is committed in `4c463f4`. Phase 5 implementation has not started; no Phase 5 code, experiments or results exist yet. See "Approved Phase 5 design decisions" below.
- Phase 4 — Model Development & Evaluation is complete and mentor-approved (2026-11-08). Phase 4a (chronological partitioning) is committed in `37d4521`. Phase 4b (baseline modeling) is committed in `4b5b94b`, and the Phase 4c specification treats it as a completed prerequisite. **Phase 4c (threshold analysis and temporal stability) is approved and complete, committed and pushed in `cffc32c`.** The retrospective documentation remediation of Phases 1–4 was completed on 2026-11-08: Phase 1 is committed in `0459921`, the Phase 2 report in `9b52a8a`, and the Phase 3 report in `ac178c3`. The Phase 4c completion work (common-reference comparison) and the Phase 4 retrospective report are committed in `1f510f7` (the Phase 4 completion commit).
- Documentation: `README.md` (overview and usage), this file (architecture and methodology), `docs/WORK_LOG.md` (chronological history and experiment results), `docs/TECHNICAL_DOCUMENTATION.md` (component reference), and `docs/FINAL_REPORT.md` (pending placeholder). Topic documents: `docs/business_understanding.md` and `docs/validation_strategy.md`. Retrospective phase reports: `docs/reports/` (Phase 1: `docs/reports/phase1_foundation.md`; Phase 2: `docs/reports/phase2_data_understanding.md`; Phase 3: `docs/reports/phase3_data_splitting.md`; Phase 4: `docs/reports/phase4_model_development.md`, committed in `1f510f7`).
- Data documentation expects `data/raw/creditcard.csv`. Raw data is ignored by Git and must remain unchanged.
- Requirements listed in `requirements.txt`: pandas, NumPy, Matplotlib, Seaborn, Jupyter, ipykernel and scikit-learn (1.9.1 installed in the designated environment).
- Tests run from the repository root with `/Users/duc/Desktop/env/bin/python -m unittest discover -s tests -v`. They use the standard-library `unittest`; pytest is not installed or required.
- The designated environment is `/Users/duc/Desktop/env/bin/python` (Python 3.13.6, pandas 2.3.1, NumPy 2.3.2, scikit-learn 1.9.1).

## Architecture (as of Phase 4c)

| Concern | Module |
|---|---|
| Dataset column roles | `src/config.py`: `DatasetConfig` and the `CREDITCARD` instance |
| Chronological split (primary) | `src/data/split.py` (Phase 4a, unchanged) |
| Stratified random split with duplicate groups kept together (secondary) | `src/data/random_split.py` |
| Preprocessing | `src/features/preprocessing.py`: `StandardScaler` inside the model pipeline |
| Baseline models | `src/models/baselines.py` |
| Metrics | `src/evaluation/metrics.py` |
| Train/evaluate runner | `src/models/experiment.py` (`fit_and_score`, `fit_and_evaluate`); CLI `python -m src.models.run_baselines` |
| Threshold analysis (Phase 4c) | `src/evaluation/threshold.py`; validation time windows in `src/data/temporal_windows.py`; CLI `python -m src.models.run_threshold_analysis` |

Both splits return the same ordered categorical `train`/`validation`/`test` labels, so they plug into the same runner. Preprocessing lives inside each scikit-learn `Pipeline`, so `fit` on the training rows is the only place any statistic is learned. The runner refuses to evaluate on the test partition. Phase 4c fits once, scores validation once, and reuses those scores for every threshold and window analysis. Details: `docs/TECHNICAL_DOCUMENTATION.md`.

## Confirmed validation decisions

- Primary validation: chronological 70/15/15, with all rows sharing a timestamp assigned together.
- Train: `Time` 0–132928; 199,364 rows; 384 frauds.
- Validation: `Time` 132929–151328; 42,722 rows; 56 frauds.
- Test: `Time` 151329–172792; 42,721 rows; 52 frauds.
- Keep all duplicate records. Identical feature groups must remain within a single partition. The inspected data had 773 repeated feature groups, 1,081 excess rows, consistent labels, and zero groups crossing the confirmed chronological partitions.
  - *Retrospective qualification (Phase 3 audit, 2026-11-08; report committed in `ac178c3`):* this duplicate key is all columns except `Class`, so it includes `Time`. Because the chronological split is a function of `Time`, zero crossing groups holds by construction and is not independent evidence. Over the model inputs (`V1`–`V28` + `Amount`), 1,335 groups cross the chronological partitions and 2,391 cross the random split. All are non-fraud with consistent labels. They are documented as potential leakage of unmeasured effect, not as leakage or future-data contamination. The decision itself is unchanged. See `docs/reports/phase3_data_splitting.md`, Section 5.
  - *Disclosure:* test fraud counts were known during Phase 3 design. The boundary rule uses only row counts per timestamp, and no model or threshold has been selected on test performance (Phase 3 report, Section 3.4).
- Secondary benchmark: a stratified random split that keeps duplicate feature groups together. It is reported separately and never pooled with chronological results.
- No classification threshold is selected. *Refined in Phase 4c (per the mentor's specification):* F1-maximizing thresholds are now selected on validation data as an **analytical evaluation**. No operational or production threshold has been selected; that still requires agreed costs and review capacity.
- **Partition usage policy** (updated 2026-11-08 to match the approved methodology; it replaces an earlier blanket rule against using validation or test outcomes to select models, thresholds or hyperparameters):
  - **Train:** model fitting and training-related operations. Preprocessing is fitted on training data only.
  - **Validation:** may be used for model selection, hyperparameter tuning, threshold selection and calibration decisions, following the approved Phase 5 protocol. Results selected on validation are development results, not independent performance estimates.
  - **Test:** reserved for final evaluation, after model selection, threshold selection and calibration decisions are frozen. Test outcomes must not influence later model development or selection.

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
   - Precision, recall and the confusion matrix are *diagnostic metrics* at each estimator's default `predict()` rule: probability strictly greater than 0.5 for Logistic Regression (scikit-learn's binary `predict` flags a row when its decision function is > 0; no validation score equals exactly 0.5, so the counts are the same under ≥), and always the majority class for the majority baseline.
   - 0.5 is not an optimal or production threshold, and no threshold has been selected.

Phase 4b results are recorded in `docs/WORK_LOG.md` and `reports/phase4b_baseline_results.json`.

## Phase 4c methodology (mentor specification, 2026-11-08)

- **Model and data:** the balanced Logistic Regression only, on the chronological split only. It is trained once on train and scored once on validation with `predict_proba`. The test set is never scored, and the random split is not used for thresholds.
- **Threshold candidates:** every unique validation score. The rule `score >= threshold` is inclusive.
- **Selection:** maximize F1, breaking ties by the lowest threshold. This is an evaluation convention.
- **Full-validation optimum:** a retrospective diagnostic, not an independently validated threshold.
- **Temporal stability:** validation is split into two consecutive windows of about 50% of rows each. The boundary is the one between unique timestamps that minimizes the size imbalance (the earlier boundary on ties), so equal timestamps stay together. The Window 1 max-F1 threshold is frozen and applied unchanged to Window 2, and compared with Window 2's retrospective optimum.
- **Common-reference comparison (descriptive, approved in the Phase 4c design discussion; implemented 2026-11-08):** a separate experiment from the transfer above. The fixed threshold 0.5 and the full-validation max-F1 threshold are each applied independently to Window 1 and Window 2, using the same scores and the inclusive `>=` rule. The full-validation threshold is retrospective, because it was selected using the labels of both windows. Output key `common_reference_thresholds` in `reports/phase4c_threshold_results.json`.
- **Interpretation limits:** 56 validation frauds (23 / 33 per window); the dataset covers about 48 hours; balanced-weighted probabilities are uncalibrated and saturate near 1.0; F1 is not a cost function. Results are in `docs/WORK_LOG.md` and `reports/phase4c_threshold_results.json`.

## Approved Phase 5 design decisions (methodology briefing, recorded 2026-11-08)

> **Status: approved design, not implemented.** These items are planned methodology. No Phase 5 code has been written, no Phase 5 experiment has been run, and no Phase 5 result exists. The briefing's full text is not stored in this repository; this section records the approved decisions at the level of detail agreed.
>
> **Awaiting detailed specification** (not set; must not be assumed): K values for fixed-K capacity; the definition of time-based review capacity; cost ratios for the cost-sensitive and break-even analysis; the temporal-block bootstrap block length; the hyperparameter search space.

- **Validation design**
  - A temporal validation and tuning protocol.
  - A validation and test-set usage policy.
  - Window 1 → Window 2 transfer evaluation (the Phase 4c temporal windows).
- **Models**
  - HistGradientBoosting is the challenger model. The Phase 4 balanced Logistic Regression is the existing baseline.
- **Calibration**
  - Model calibration is part of Phase 5.
  - A temporal calibration holdout has priority. Ensemble investigation is the fallback, if needed.
- **Statistical comparison**
  - Paired bootstrap evaluation.
  - Temporal-block bootstrap for the online queue evaluation, when feasible. Otherwise: point estimates with sensitivity analysis.
- **Operational evaluation**
  - Use case: the Analyst Review Queue (see `docs/business_understanding.md`).
  - Cost-sensitive evaluation with break-even analysis.
  - Review capacity is evaluated both as fixed-K capacity and as time-based capacity.
  - Precision@K and Recall@K.
  - SLA scenarios of 1 hour, 2 hours and 4 hours.

The existing rules still apply: the test partition stays untouched until final evaluation, and the test-partition scope (below) must be acknowledged before it is used. The Phase 1–4 results above are unchanged.

## Scope and limits

No feature engineering or hyperparameter tuning has been performed. The data covers about 48 hours and has few fraud cases in validation (56 chronological, 74 random) and test. Independent bootstrap confidence intervals (audit computations, seed 0, 2,000 stratified resamples) for the Phase 4b average precision and the Phase 4c threshold transfer are documented in `docs/reports/phase4_model_development.md` (Sections 3.2 and 5.1). `V1`–`V28` are anonymized PCA components. Label availability timing and the intended operational prediction point remain unspecified. The chronological boundaries in `src/data/split.py` are specific to this dataset. **Test-partition scope (to be acknowledged before final test evaluation):** the test set covers only hour values 18–23 (elapsed hour modulo 24, not clock time) of the second day of one approximately 48-hour dataset, with 42,721 transactions, 52 frauds and prevalence 0.1217%. Validation and test contain no hour values 00–11 (Phase 3 report, Sections 4 and 7). Refer to `docs/validation_strategy.md` for detailed design and leakage controls, and `docs/business_understanding.md` for business framing.

## Handoff state and next task

Phase 4 is complete, mentor-approved and committed on branch `main` (Phase 4c in `cffc32c`; completion work and Phase 4 report in `1f510f7`). The 9.5 MB results file `reports/phase4c_threshold_results.json` stays tracked in Git (project-owner decision, 2026-11-08), and Git history is not rewritten. The retrospective documentation remediation of Phases 1–4 is complete. A model-input duplicate checker is a recorded future improvement and needs separate approval.

The Phase 5 methodology and design briefing is complete, and its decisions are recorded above as approved design. The design documentation review is complete; the documentation is committed in `4c463f4`. **Next task:** Phase 5 experimental protocol preparation. Phase 5 implementation has not started; it starts only on explicit approval and proceeds with review between stages. Do not write the final report until the project is complete. Keep the test partition untouched until final evaluation. The Phase 4c thresholds are analytical; no operational threshold has been chosen. See `docs/WORK_LOG.md` for details.

## Assumptions

No additional project assumptions are asserted here. Dataset counts and split behavior above are verified by automated real-data tests (`tests/test_split.py`, `tests/test_random_split.py`, `tests/test_baselines.py`). Recheck them if the local CSV changes; its SHA-256 is recorded in `docs/WORK_LOG.md`.
