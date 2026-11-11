# FraudGuard Project Context

## Verified repository facts

- FraudGuard is an educational credit-card fraud detection project.
- Current project phase: **Phase 5B-2 part II: protocol Section 4.2 step 2, implemented and awaiting mentor review (2026-11-11); not committed.**
  - All 28 frozen configurations were fitted once on `T_fit`, and `T_cal` and validation were scored once each.
  - Step-3 candidates under the Section 5.4 rule: **LR-10** (C = 1, balanced; validation AP 0.8536) and **HGB-14** (0.1 / 31 / 20 / balanced; 0.8686).
  - Results: `reports/phase5b2_grid_results.json`; report: `docs/reports/PHASE_5B_2_REPORT.md`, Part II.
  - No calibration, criterion E1/E2 statistic, champion decision or test evaluation has been run.
- *Part I of Phase 5B-2, mentor-approved with a scope clarification (E0 and experiment E2 protocol-aligned; E1 and E3 exploratory deviations only, never influencing champion selection):* controlled experiments E0–E3 (LR / HGB × baseline features / + `log1p(Amount)`). Implemented and awaiting mentor review (2026-11-11); not committed.** Report: `docs/reports/PHASE_5B_2_REPORT.md`; results: `reports/phase5b2_results.json`. One a-priori HGB grid point was fitted. No hyperparameter grid, selection, calibrator, criterion E1/E2 statistic or test evaluation has been run. `log1p(Amount)` (E1, E3) is a logged, owner-instructed exploratory deviation from protocol Section 5.5, outside the champion pathway.
  - *Previous phase:* Phase 5B-1 (E2 usability, `T_fit`/`T_cal` partition helper, three-track Phase 4 continuity), committed in `aa4e63a`. Report: `docs/reports/PHASE_5B_1_REPORT.md`; results: `reports/phase5b1_results.json`.
  - *Previous phase:* **Phase 5B-0 — protocol and documentation remediation and reproducibility infrastructure, complete and mentor-approved (2026-11-11).** Phase 5B-0 is committed: implementation in `7c4edcd` (`7c4edcdc400b1f4332c7d961f845662f65685ea9`), environment manifest in `aa67802` (`aa67802353f3646e7b5eb53237f3da8e8d263626`); see `git log` for the current push state.
  - **Phase 5A** (experimental protocol) is approved by the project owner and closed (2026-11-09). The approved protocol is `docs/phase_5a_experimental_protocol.md`, Revision 4 plus closure, committed in `9b6d8e8` (originally `760ccc3`; renamed by the 2026-11-11 attribution rewrite, files identical; see WORK_LOG).
  - Phase 5B-0 adds the implementation clarification addendum (protocol Appendix C), frozen settings, the ranking-score interface, the E2 block and buffer convention, reproducibility tooling and the onboarding documentation. See `docs/reports/PHASE_5B_0_COMPLETION_REPORT.md`.
  - *Superseded by Phase 5B-1 (2026-11-11):* "No Phase 5 model has been fitted or scored, and no Phase 5 experiment or result exists." Phase 5B-1 fitted one model, the Phase 4 configuration on `T_fit`, for the continuity comparison only.
  - Earlier status: the Phase 5 design briefing was approved on 2026-11-08 (committed in `a84f94a`, originally `4c463f4`); see "Approved Phase 5 design decisions" below.
- Phase 4 — Model Development & Evaluation is complete and mentor-approved (2026-11-08). Phase 4a (chronological partitioning) is committed in `37d4521`. Phase 4b (baseline modeling) is committed in `4b5b94b`, and the Phase 4c specification treats it as a completed prerequisite. **Phase 4c (threshold analysis and temporal stability) is approved and complete, committed and pushed in `cffc32c`.** The retrospective documentation remediation of Phases 1–4 was completed on 2026-11-08: Phase 1 is committed in `0459921`, the Phase 2 report in `9b52a8a`, and the Phase 3 report in `ac178c3`. The Phase 4c completion work (common-reference comparison) and the Phase 4 retrospective report are committed in `1f510f7` (the Phase 4 completion commit).
- Documentation: `README.md` (overview and usage), this file (architecture and methodology), `docs/WORK_LOG.md` (chronological history and experiment results), `docs/TECHNICAL_DOCUMENTATION.md` (component reference), and `docs/FINAL_REPORT.md` (pending placeholder). Topic documents: `docs/business_understanding.md` and `docs/validation_strategy.md`. Phase 5 protocol: `docs/phase_5a_experimental_protocol.md` (Revision 4; Appendix C = Phase 5B-0 clarifications). Policies and templates: `docs/result_storage_policy.md`, `docs/templates/phase5_freeze_manifest.template.json`. Dataset instructions: `data/README.md`. Retrospective phase reports: `docs/reports/` (Phase 1: `docs/reports/phase1_foundation.md`; Phase 2: `docs/reports/phase2_data_understanding.md`; Phase 3: `docs/reports/phase3_data_splitting.md`; Phase 4: `docs/reports/phase4_model_development.md`, committed in `1f510f7`; Phase 5B-0: `docs/reports/PHASE_5B_0_COMPLETION_REPORT.md`).
- Data documentation expects `data/raw/creditcard.csv`. Raw data is ignored by Git and must remain unchanged.
- Requirements (Phase 5B-0): `requirements.txt` pins NumPy 2.3.2, pandas 2.3.1, scikit-learn 1.9.1, SciPy 1.17.1, Matplotlib 3.10.5 and seaborn 0.13.2. `requirements-notebook.txt` (optional) adds ipykernel 7.1.0 and an unpinned `jupyter` front end. *Superseded:* the earlier unpinned list included `jupyter`, which is not installed in the verified environment.
- Tests run from the repository root with `python -m unittest discover -s tests -v`, using the activated project environment. They use the standard-library `unittest`; pytest is not installed or required. Setup for new users: `README.md`, "Getting started".
- Verified environment: Python 3.13.6 with the pinned versions above, recorded in `reports/environment_manifest.json`. It is the original developer's local environment, which must not be modified. Other environments are not verified.

## Architecture (as of Phase 5B-2)

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
| Frozen Phase 5 settings (Phase 5B-0) | `src/models/phase5_settings.py`: seeds, partitions, grids and protocol constants; declarations only |
| Ranking-score interface (Phase 5B-0) | `src/models/ranking_score.py`: `decision_function` score, Section 9.2 ordering key, Platt sign mapping, constancy guard |
| E2 blocks and buffer (Phase 5B-0) | `src/data/validation_blocks.py` |
| `T_fit`/`T_cal` sub-partitions (Phase 5B-1) | `src/data/train_subpartitions.py`: frozen boundary 112,553 \| 112,555; train rows only; integrity and estimability checks |
| E2 usable-split count (Phase 5B-1) | `src/evaluation/e2_usability.py`: validation timestamps and labels only; frozen rules (≥ 10 held-out frauds; ≥ 1,000 usable splits) |
| Phase 4 continuity (Phase 5B-1) | `src/models/phase4_continuity.py`: three score tracks with a required `score_representation`; CLI `python -m src.models.run_phase5b1` |
| `log1p(Amount)` feature (Phase 5B-2) | `src/features/amount_transform.py`: stateless, row-wise; appends `log1p_Amount`, keeps `Amount`; refuses negative or non-finite amounts |
| Fixed-K capacity metrics (Phase 5B-2) | `src/evaluation/capacity.py`: Precision@K, Recall@K, TP/FP, oracle ceiling, Section 9.2 tie-break and boundary-tie report |
| Experiments E0–E3 (Phase 5B-2) | `src/models/phase5b2_experiments.py`: definitions, the a-priori HGB grid point, fitting on `T_fit`, tracks A/B/C with a required `score_representation`, held-out-half stability (descriptive, not criterion E2); CLI `python -m src.models.run_phase5b2` |
| 28-configuration grid (Phase 5B-2, protocol 4.2 step 2) | `src/models/phase5_grid.py`: frozen grid enumeration, builders, single `T_fit` fit with `T_cal` and validation scoring, Section 5.4 selection rule; CLI `python -m src.models.run_phase5b2_grid` (`compute_grid` returns the in-memory score cache) |
| Reproducibility (Phase 5B-0) | `src/reproducibility/dataset.py` (fingerprint check); `src/reproducibility/manifest.py` (environment manifest, freeze-manifest template, result-size policy) |

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

> **Status: approved design, now fully specified by the approved protocol** (`docs/phase_5a_experimental_protocol.md`, Revision 4, closed 2026-11-09; clarifications in Appendix C, 2026-11-11). No Phase 5 experiment has been run and no Phase 5 result exists. The briefing's full text is not stored in this repository; this section records the briefing-level decisions. Where the protocol refines them, the protocol governs.
>
> *Superseded (2026-11-09):* the list formerly headed "Awaiting detailed specification" is now specified in the protocol:
> - K values for fixed-K capacity: Section 9.2.
> - The time-based capacity definition: Section 9.3.
> - Cost ratios: Section 8.2.
> - Block length: Sections 7.2 and 7.4.
> - Hyperparameter search space: Section 5.
> - Calibration holdout: Section 6.2 (C-A).
>
> The final decisions are in Section 13.0.

- **Validation design**
  - A temporal validation and tuning protocol.
  - A validation and test-set usage policy.
  - Window 1 → Window 2 transfer evaluation (the Phase 4c temporal windows).
- **Models**
  - HistGradientBoosting is the challenger model. The Phase 4 balanced Logistic Regression is the existing baseline.
  - Champion rule (protocol Section 4.3, Section 13.0):
    - Logistic Regression is the default champion.
    - HGB replaces it only if both E1 and E2 pass.
    - If E2 has fewer than 1,000 usable splits of 2,000, it is not estimable and Logistic Regression is retained.
- **Calibration**
  - Model calibration is part of Phase 5.
  - A temporal calibration holdout has priority (protocol Section 6, C-A: the last ~20% of train, sigmoid).
  - *Superseded by the approved protocol (decision #1b, Section 13.0):* the briefing's "ensemble investigation is the fallback, if needed" no longer applies.
    - On a technical calibration failure, the affected model uses its **uncalibrated scores**. That is the safe alternative.
    - The cross-fitted ensemble (C-F) is a **retrospective sensitivity analysis only**: never a calibration fallback and never primary evidence.
    - Calibration status is decided separately for each model.
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

No hyperparameter tuning has been performed. *Qualified in Phase 5B-2:* the only feature engineering is the exploratory `log1p(Amount)` column of experiments E1/E3 (a logged, owner-instructed deviation from protocol Section 5.5). It is not adopted into any model or into the champion pathway. The data covers about 48 hours and has few fraud cases in validation (56 chronological, 74 random) and test. Independent bootstrap confidence intervals (audit computations, seed 0, 2,000 stratified resamples) for the Phase 4b average precision and the Phase 4c threshold transfer are documented in `docs/reports/phase4_model_development.md` (Sections 3.2 and 5.1). `V1`–`V28` are anonymized PCA components. Label availability timing and the intended operational prediction point remain unspecified. The chronological boundaries in `src/data/split.py` are specific to this dataset. **Test-partition scope (to be acknowledged before final test evaluation):** the test set covers only hour values 18–23 (elapsed hour modulo 24, not clock time) of the second day of one approximately 48-hour dataset, with 42,721 transactions, 52 frauds and prevalence 0.1217%. Validation and test contain no hour values 00–11 (Phase 3 report, Sections 4 and 7). Refer to `docs/validation_strategy.md` for detailed design and leakage controls, and `docs/business_understanding.md` for business framing.

## Handoff state and next task

Phase 4 is complete, mentor-approved and committed on branch `main` (Phase 4c in `cffc32c`; completion work and Phase 4 report in `1f510f7`). The 9.5 MB results file `reports/phase4c_threshold_results.json` stays tracked in Git (project-owner decision, 2026-11-08), and Git history is not rewritten. The retrospective documentation remediation of Phases 1–4 is complete. A model-input duplicate checker is a recorded future improvement and needs separate approval.

The Phase 5 methodology and design briefing is complete, and its decisions are recorded above as approved design. The design documentation is committed in `a84f94a` (originally `4c463f4`). Phase 5A (experimental protocol) is approved and closed (committed in `760ccc3`). *Historical:* the next task recorded here before the protocol was "Phase 5 experimental protocol preparation"; that is done.

*Historical (end of Phase 5B-0):* Phase 5B-0 (protocol and documentation remediation, reproducibility infrastructure) is complete and mentor-approved (2026-11-11), committed in `7c4edcd`, with the environment manifest in `aa67802`. **Next task:** Phase 5B-1, the first implementation and experiment stage of the protocol. It starts only on explicit approval and proceeds with review between stages. Seeds, the ranking score (`decision_function`), the E2 buffer convention and the result-storage policy are frozen before it (protocol Appendix C). Do not write the final report until the project is complete. Keep the test partition untouched until final evaluation. The Phase 4c thresholds are analytical; no operational threshold has been chosen. See `docs/WORK_LOG.md` for details.

**Current state (2026-11-11):** Phase 5B-2 is implemented and awaiting mentor review; it is not committed. Development results (validation only; 56 frauds; track B = `decision_function` on `T_fit`):
- AP: E0 0.8536, E1 0.8539, E2 = E3 0.8686.
- HGB − LR ΔAP: +0.003 in W1 and +0.024 in W2. On the held-out halves of the frozen E2 splits, the median is +0.016 and the difference is positive in 80% of splits (descriptive).
- At the illustrative K grid, all models catch the same frauds within one.
- `log1p(Amount)` has a negligible effect on LR and exactly none on HGB: as a monotone duplicate, it is never used in a split.
- Track C (`predict_proba`) equals track B for all four experiments.
- No model is selected; the champion rule (criteria E1 + E2 after the 28-configuration grid) is unchanged. See `docs/reports/PHASE_5B_2_REPORT.md`.

*Historical (end of Phase 5B-1):* Phase 5B-1 was implemented and awaiting mentor review; it is now committed (`aa4e63a`). Results (development, validation only):
- `T_fit` 159,491 rows / 356 frauds and `T_cal` 39,873 / 28, exactly as in protocol Section 3.1; no timestamp spans both.
- E2: 1,984 of 2,000 splits usable, so E2 is estimable under the pre-registered rule. No E2 statistic has been computed.
- Continuity, validation AP: 0.8397 historical (`predict_proba`, full train, quoted); 0.8536 Phase 5 primary (`decision_function`, `T_fit`); 0.8536 diagnostic (`predict_proba`, same fit).

**Next task:** mentor review of Phase 5B-2 part II (the grid and the recorded step-3 candidates LR-10 and HGB-14). Then, only on approval, step 4 (sigmoid calibration on `T_cal`). *Superseded:* "mentor review of Phase 5B-2, including the logged `log1p(Amount)` deviation" (approved; step 2 done). *(Earlier text: mentor review of Phase 5B-1.)* Then, only on approval, protocol Section 4.2 step 2 (fit the 28 configurations on `T_fit`, score `T_cal` and validation once each). The test partition stays untouched until final evaluation. Do not write the final report until the project is complete.

## Assumptions

No additional project assumptions are asserted here. Dataset counts and split behavior above are verified by automated real-data tests (`tests/test_split.py`, `tests/test_random_split.py`, `tests/test_baselines.py`). Recheck them if the local CSV changes; its SHA-256 is recorded in `docs/WORK_LOG.md`.
