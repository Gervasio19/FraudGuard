# FraudGuard Work Log

## Current status

- **Current phase:** Phase 4 — Model Development & Evaluation, in progress. Phase 4a (chronological partitioning module and tests) is implemented and awaiting human review. Phase 4b has not started and requires explicit approval.
- **Git state:** branch `main`, `HEAD` `c0ecd3c` (`Add AI handoff protocol and project continuity docs`), which committed this log and `docs/PROJECT_CONTEXT.md`. Phase 4a changes are uncommitted in the working tree (see the Phase 4a session below).
- **Handoff documentation:** consists of exactly `docs/PROJECT_CONTEXT.md` and `docs/WORK_LOG.md`. Despite the `c0ecd3c` commit message, no separate handoff protocol file exists.
- **Raw data:** `data/raw/creditcard.csv` is local and ignored by Git. It has not been modified (SHA-256 `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`, checked before and after Phase 4a).

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
- The chronological split is implemented as reusable code in `src/data/split.py` (Phase 4a). The secondary random benchmark split is not yet implemented.
- No feature engineering or model training has been performed; the classification threshold remains intentionally undecided.

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

Phase 4a is complete pending human review. After explicit approval, begin Phase 4b: fit a simple baseline on the training partition only (preprocessing fitted on train), and evaluate on validation with class-sensitive, threshold-free metrics. Keep the test partition untouched until final evaluation. Do not select a classification threshold until the operational criteria and costs are agreed. The Phase 4b baseline choice and threshold-reporting approach still need human decisions.
