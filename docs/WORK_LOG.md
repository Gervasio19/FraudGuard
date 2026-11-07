# FraudGuard Work Log

## Current status

- **Current phase:** Phase 4 — Model Development & Evaluation, in progress. Phase 4a (chronological partitioning) is committed in `37d4521`. Phase 4b (baseline modeling) is implemented and tested. Its three open decisions were approved by the mentor, and the final verification passed (see the decision finalization session below). **Phase 4b completion awaits independent mentor approval; it is not marked complete.** Phase 4c has not started.
- **Git state:** branch `main`, `HEAD` `37d4521` (`Complete Phase 4a chronological data splitting`). Phase 4b changes are uncommitted in the working tree (see the Phase 4b session below).
- **Documentation set:** `README.md`, `docs/PROJECT_CONTEXT.md`, `docs/WORK_LOG.md`, `docs/TECHNICAL_DOCUMENTATION.md` (created in Phase 4b), and `docs/FINAL_REPORT.md` (a pending placeholder, created in Phase 4b). Topic documents: `docs/business_understanding.md`, `docs/validation_strategy.md`.
- **Raw data:** `data/raw/creditcard.csv` is local and ignored by Git. It has not been modified (SHA-256 `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`, checked before and after Phase 4a and Phase 4b).

## Phase 4b decision finalization session (2026-11-07)

The mentor approved the three decisions raised in the Phase 4b report. No models, tuning, threshold optimization, Phase 4c work, raw-data edits, commits or pushes were done in this session.

### Approved decisions

1. **Raw `Time` is excluded from the baseline model features.** It is still used for the chronological split and duplicate grouping. Reason: it is elapsed seconds within this 48-hour dataset, and the chronological validation period lies entirely outside the training time range. This does not claim that temporal information has no predictive value; temporal feature engineering remains a possible future direction.
2. **Stratified random split:** 70/15/15, seed 42, stratified by `Class`. Duplicate groups are defined by all columns except `Class` and never cross partitions. All rows are preserved. This split remains a secondary benchmark; chronological evaluation stays primary.
3. **The default 0.5 rule is used for diagnostic reporting only.** Average precision and ROC-AUC are ranking metrics; precision, recall and the confusion matrix at 0.5 are diagnostic metrics. 0.5 is not an optimal or production threshold.

The code already implemented all three decisions, so no model, split or metric code changed. The only code change is the `decision_rule_note` text in `src/models/run_baselines.py`, which now uses the "ranking" and "diagnostic" wording. `reports/phase4b_baseline_results.json` was regenerated: a diff against the previous file showed that only that note changed, and every metric is identical.

### Final verification

I wrote an independent script in the session scratchpad, outside the repository. It recomputes values directly from the raw CSV where possible.

1. **Data hash:** SHA-256 of `data/raw/creditcard.csv` (the path in `CREDITCARD.path` and in the results JSON) is `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`. It matches the JSON and the value recorded since Phase 4a. The file has 284,807 rows, 492 frauds and 31 columns.
2. **Class proportions in the random split** (overall fraud rate 0.1727%):

   | Partition | Rows (share) | Frauds (share of all frauds) | Fraud rate |
   |---|---:|---:|---:|
   | Train | 199,364 (70.000%) | 344 (69.92%) | 0.1725% |
   | Validation | 42,722 (15.000%) | 74 (15.04%) | 0.1732% |
   | Test | 42,721 (15.000%) | 74 (15.04%) | 0.1732% |

3. **Duplicate isolation:** I computed it independently by hashing all non-`Class` columns. Both splits contain 773 repeated groups and 1,081 excess rows, all 284,807 rows are retained, and 0 groups cross partitions.
4. **Average precision uses probabilities:** I recomputed it from the captured `predict_proba` outputs, and it matches the reported values exactly (A: 0.839660, B: 0.752375). Logistic Regression produced 41,807 and 42,104 distinct scores. Average precision computed from the 0.5 labels instead would be 0.0370 and 0.0572, so the reported values are not label-based. The majority baseline has one constant score, so its average precision equals the prevalence.
5. **Test isolation:** I wrapped every pipeline's `fit`, `predict_proba` and `predict`. In both experiments and for both models, `fit` received only the 199,364 training rows and prediction received only the 42,722 validation rows; 0 test rows were touched. No model selection takes place. The two baselines are fixed and are reported side by side.
6. **Reproducibility:** two fresh CLI runs were byte-identical to each other and to the recorded results file, before the note text changed.

Tests: `/Users/duc/Desktop/env/bin/python -W always -m unittest discover -s tests -v` → 97 tests, OK, with no warnings and none skipped.

## Phase 4b session — baseline modeling (2026-11-07)

Scope: two fixed baselines, evaluated on the validation partition of two separate experiments. No threshold selection, hyperparameter tuning, test-set scoring, feature engineering, Phase 4c work, raw-data edits, commits or pushes.

- **Files created:**
  - `src/config.py`: the `DatasetConfig` dataclass (target, positive label, feature selection or exclusion, optional time column, optional duplicate-group columns) and the `CREDITCARD` instance.
  - `src/data/random_split.py`: `assign_stratified_group_partitions`, the secondary benchmark split (stratified by label and keeping duplicate groups together, seeded). This split was specified in Phase 3 but had not been implemented; it returns the same partition labels and dtype as `src/data/split.py`.
  - `src/features/preprocessing.py`: `build_preprocessor` (a `StandardScaler` on the configured features inside a `ColumnTransformer` that drops all other columns) and `validate_feature_frame`.
  - `src/models/baselines.py`: `build_majority_baseline` (`DummyClassifier(strategy="prior")`) and `build_logistic_regression` (`LogisticRegression(class_weight="balanced", C=1.0, solver="lbfgs", max_iter=1000)`). Each returns an unfitted preprocess→model `Pipeline`.
  - `src/evaluation/metrics.py`: `compute_binary_metrics` (average precision, ROC-AUC, precision, recall, confusion matrix; undefined values are returned as NaN).
  - `src/models/experiment.py`: `fit_and_evaluate`, which fits on `train` rows only and evaluates on `validation` (passing `evaluation_partition="test"` raises an error); `summarize_partitions`.
  - `src/models/run_baselines.py`: the CLI `python -m src.models.run_baselines`, which runs Experiments A and B and writes `reports/phase4b_baseline_results.json`.
  - `reports/phase4b_baseline_results.json`: the recorded results, including configuration, data SHA-256 and library versions.
  - `tests/test_random_split.py` (23 tests) and `tests/test_baselines.py` (38 tests).
  - `docs/TECHNICAL_DOCUMENTATION.md` and `docs/FINAL_REPORT.md` (pending placeholder).
- **Files modified:** `docs/WORK_LOG.md`, `docs/PROJECT_CONTEXT.md`, `README.md`, and `docs/validation_strategy.md` (an appended "Implementation note" for the secondary split; the Phase 3 text is unchanged). `src/data/split.py` and `tests/test_split.py` were **not** modified.

### Experiment configuration

- **Model features:** `V1`–`V28` and `Amount` (29 columns). `Time` is **excluded from the model features** but still drives the chronological split and duplicate grouping. Reason: `Time` counts seconds since the first transaction in this 48-hour extract, so in Experiment A every validation value lies outside the training range. This was proposed as a default needing approval. *Resolved:* approved in the decision finalization session above.
- **Preprocessing:** a `StandardScaler` on all 29 features, fitted inside the pipeline on the training partition only. There is no imputation, resampling or feature engineering.
- **Experiment A (primary):** the chronological split from `src/data/split.py`, unchanged (Phase 3 boundaries).
- **Experiment B (secondary):** the stratified random split with duplicate groups kept together. Fractions are 70/15/15 within each class, `random_state=42`, and groups are defined by all columns except `Class`.
- **Evaluation:** validation partition only. The test partition was never passed to any model. Partition label counts for test are reported for descriptive purposes only.
- **Decision rule for precision, recall and the confusion matrix:** each estimator's default `predict()`. For Logistic Regression this is probability ≥ 0.5; for the majority baseline it means always predicting non-fraud. **This is not a selected or optimized operating threshold.** *Approved wording, from the finalization session:* these are diagnostic metrics, and 0.5 is not an optimal or production threshold.
- **Environment:** Python 3.13.6, pandas 2.3.1, NumPy 2.3.2, scikit-learn 1.9.1 (`/Users/duc/Desktop/env/bin/python`).

### Split verification

| Experiment | Train rows / frauds | Validation rows / frauds | Test rows / frauds | Duplicate groups crossing partitions |
|---|---:|---:|---:|---:|
| A: chronological | 199,364 / 384 | 42,722 / 56 | 42,721 / 52 | 0 |
| B: stratified random (seed 42) | 199,364 / 344 | 42,722 / 74 | 42,721 / 74 | 0 |

The B partition sizes coincide with A's because both target 70/15/15 of 284,807 rows. A's sizes come from the Phase 3 timestamp boundaries; B's are computed per class (199,020 + 344 train rows).

### Validation results (verified by running `python -m src.models.run_baselines`)

The two experiments are reported separately and must not be pooled.

**Experiment A: chronological (primary)**, validation: 42,722 rows, 56 frauds, prevalence 0.131%.

| Model | Average precision | ROC-AUC | Precision* | Recall* | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Majority class | 0.0013 | 0.5000 | undefined | 0.0000 | 42,666 | 0 | 56 | 0 |
| Logistic Regression (balanced) | 0.8397 | 0.9815 | 0.0397 | 0.9286 | 41,409 | 1,257 | 4 | 52 |

**Experiment B: stratified random (secondary benchmark)**, validation: 42,722 rows, 74 frauds, prevalence 0.173%.

| Model | Average precision | ROC-AUC | Precision* | Recall* | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Majority class | 0.0017 | 0.5000 | undefined | 0.0000 | 42,648 | 0 | 74 | 0 |
| Logistic Regression (balanced) | 0.7524 | 0.9725 | 0.0649 | 0.8784 | 41,711 | 937 | 9 | 65 |

Average precision and ROC-AUC are ranking metrics. \* These columns are diagnostic metrics at the default 0.5 rule (Logistic Regression probability ≥ 0.5); 0.5 is not an optimal or production threshold. Precision for the majority baseline is undefined because it makes no positive predictions.

- Logistic Regression converged in both experiments (32 and 22 lbfgs iterations; no `ConvergenceWarning`).
- As expected, the majority baseline's average precision equals the validation prevalence and its ROC-AUC is 0.5.
- With class weighting, the 0.5 rule flags 1,309 (A) and 1,002 (B) validation transactions, about 2–3% of rows. That gives high recall and low precision. Choosing an operating point is deferred to later work with agreed operational criteria.
- Experiment A's average precision is higher than B's. The validation sets have only 56 and 74 frauds and cover different transactions, so this gap is within the range that a small number of cases can move. It should not be read as evidence that chronological generalization is easier.

### Tests executed

- `/Users/duc/Desktop/env/bin/python -m unittest discover -s tests -v` → **97 tests, OK** (36 existing from Phase 4a, 23 new random-split tests, 38 new baseline tests). No tests were skipped because the raw CSV is present, and `-W always` showed no warnings.
- Coverage: majority-baseline behavior; Logistic Regression configuration, training, probability outputs and reproducibility; the scaler fitted on training rows only; models unchanged when validation/test rows are altered; NaN-corrupted test rows never touched; refusal to evaluate on test; `Time` excluded from model inputs; hand-computed metric values and NaN for undefined metrics; compatibility with both split outputs; deterministic results; a CLI run end to end on synthetic data; and regression checks on the real data against the tables above.
- Mutation check (in a scratch copy, outside the repository): fitting on train+validation made 5 tests fail, including both leakage tests.
- Reproducibility: two CLI runs produced byte-identical JSON output. `git diff --check` passed.

### Outstanding issues (Phase 4b)

- Mentor approval was needed for: excluding `Time` from the model features; the seed and procedure of the random split; and reporting precision, recall and the confusion matrix at the default 0.5 rule. *Resolved:* all three were approved (see the decision finalization session).
- No confidence intervals are reported. With 56 and 74 validation frauds, the metrics have substantial sampling uncertainty.
- The chronological boundaries in `src/data/split.py` remain specific to this dataset. A new dataset would need its own documented boundaries, although the runner accepts any partition Series.

## Phase 4a session — chronological partitioning module (2026-11-07)

Scope was authorized for Phase 4a only. No model training, feature engineering, threshold selection, test-set evaluation, random benchmark split, raw-data edits, commits, or pushes.

- **Files created:**
  - `src/data/split.py` — `assign_partitions(df)` returns an ordered categorical `train`/`validation`/`test` Series aligned to `df.index`, using only `Time` and the frozen boundaries (`TRAIN_END_TIME = 132928`, `VALIDATION_END_TIME = 151328`, both inclusive). `split_partitions(df)` returns the three frames via boolean `.loc` selection, keeping original index labels, values and row order. `count_cross_partition_duplicate_groups(df, partitions)` counts identical-feature groups (all columns except `Class`) that span partitions, as a leakage check.
  - `tests/test_split.py` — 36 `unittest` tests: boundary inclusivity, documented constants, equal-timestamp and duplicate-record grouping, the leakage detector, preservation of records/index/order, no input mutation, determinism, row-order and label independence, invalid inputs, and real-data checks of the documented row, fraud, time-range and duplicate-group figures. Real-data tests are skipped if the raw CSV is absent.
- **Files modified:** `requirements.txt` (added `scikit-learn`), `docs/PROJECT_CONTEXT.md` and `docs/WORK_LOG.md` (status and handoff references only; earlier decisions and history unchanged).
- **Input validation:** rejects non-DataFrame input, a missing or repeated `Time` column, non-numeric or boolean `Time`, missing/infinite values, negative values, and non-integer values. Non-integer times are rejected because the documented inclusive integer ranges leave values such as `132928.5` undefined; the raw data contains only whole seconds. Values above `172792` are assigned to test.
- **Verification:** `/Users/duc/Desktop/env/bin/python -m unittest discover -s tests -v` → 36 tests, OK. Real-data results matched the documentation exactly: train 199,364 rows / 384 frauds, validation 42,722 / 56, test 42,721 / 52; 284,807 rows retained; 773 duplicate groups and 1,081 excess rows retained; zero duplicate groups or timestamps crossing partitions. As a sanity check, moving `TRAIN_END_TIME` by one second in memory made 4 real-data tests fail.
- **Methodology:** unchanged. Boundaries, duplicate retention, and the undecided threshold are as approved in Phase 3.

## Tasks completed in the Phase 3 finalization session

The latest Phase 3 commit finalized the validation design and aligned the project status text.

- **Committed files changed:**
  - `README.md` — updated roadmap and current status through Phase 3.
  - `docs/business_understanding.md` — aligned the evaluation plan with the finalized validation strategy and undecided threshold.
  - `docs/validation_strategy.md` — created the validation strategy, split diagnostics, leakage mitigation, limitations, and Phase 3 completion summary.
- **Files deleted:** none.
- **Commit:** `fadebd7 Complete Phase 3: Data Preparation and Validation`.
- **Decisions:** chronological 70/15/15 is primary; boundaries are train `Time 0–132928`, validation `132929–151328`, test `151329–172792`; equal timestamps stay together; all duplicate rows are retained; identical feature groups must not cross partitions; a duplicate-group-aware stratified random split is secondary; no classification threshold is selected.
- **Implementation scope:** documentation and validation diagnostics only. No data preparation, feature engineering, split implementation, model training, raw-data edits, commits, or pushes were performed in that finalization session.

## Verified diagnostics and execution

- **Environment used for the split checks:** `/Users/duc/Desktop/env/bin/python` (Python 3.13.6, pandas 2.3.1 as reported in the prior inspection).
- **Dataset counts:** 284,807 rows, 492 fraud rows. `Time` is nondecreasing and spans 0–172,792 seconds, approximately 48 hours.
- **Frozen chronological partitions:**

  | Partition | Time range | Rows | Non-fraud | Fraud | Fraud rate |
  |---|---:|---:|---:|---:|---:|
  | Train | 0–132,928 | 199,364 | 198,980 | 384 | 0.1926% |
  | Validation | 132,929–151,328 | 42,722 | 42,666 | 56 | 0.1311% |
  | Test | 151,329–172,792 | 42,721 | 42,669 | 52 | 0.1217% |

- Equal timestamps are kept in one partition. Grouping the 30 feature columns excluding `Class` identified 773 repeated feature groups and 1,081 excess rows; labels were consistent and no repeated feature group crossed the finalized partitions.
- The specified Python environment reran these partition/count and duplicate-group checks during finalization; assertions passed.
- **Notebook execution evidence from Phase 2:** all code cells in `notebooks/01_data_understanding.ipynb` were executed in order under Python 3.13.6 with pandas 2.3.1 and NumPy 2.3.2; they completed successfully. No models were trained.
- **Other checks:** `git diff --check` passed for the Phase 3 finalization changes. No conventional automated test suite was run or requested.

## Known issues, limitations, and unresolved questions

### Verified limitations

- Validation and test contain only 56 and 52 fraud cases, so metrics can be sensitive to a small number of observations.
- The dataset covers only about 48 hours in 2013; chronological evaluation cannot establish robustness across longer time periods or current production traffic.
- `V1`–`V28` are anonymized PCA components; individual original business meanings are unavailable from this dataset documentation.
- The dataset documentation does not establish label availability timing or the production prediction point.
- The chronological split is implemented as reusable code in `src/data/split.py` (Phase 4a). The secondary random benchmark split was not implemented in Phase 4a; *superseded:* it was implemented in Phase 4b (`src/data/random_split.py`).
- No feature engineering has been performed. *Superseded in Phase 4b:* two baseline models were trained (Phase 4b session above). The classification threshold remains intentionally undecided.

### Unresolved for later project work

- Confirm the intended prediction point and which fields would be available then.
- Decide operational costs, review capacity, and evaluation criteria before any threshold selection.
- Choose models and preprocessing only after the split is implemented with leakage controls and the Phase 3 boundaries remain fixed.

No blocking implementation errors are known from the completed Phase 3 validation checks. An earlier notebook cell-execution attempt lacked Jupyter's `display` binding; the notebook was corrected, and the full code-cell execution subsequently passed. The default Python environment also lacked pandas; use the designated environment listed under "Verified diagnostics and execution" above.

## Exact files created for this handoff

- `docs/WORK_LOG.md` — this chronological project handoff log.
- `docs/PROJECT_CONTEXT.md` — concise verified project context and continuation notes.

No files were deleted. No raw data or notebook was changed during handoff preparation. These two files were later committed in `c0ecd3c`; they are the complete handoff documentation, and no separate protocol file exists.

## Next recommended task

*Historical (end of Phase 4a):* begin Phase 4b after approval. That work has now been done (see the Phase 4b session).

**Current:** independent mentor approval of Phase 4b completion. Its decisions are approved and the final verification has passed. Do not begin Phase 4c until Phase 4b is approved. The test partition stays untouched until final evaluation, and no classification threshold is selected until operational criteria and costs are agreed.
