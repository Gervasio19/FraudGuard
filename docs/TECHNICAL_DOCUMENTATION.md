# FraudGuard Technical Documentation

Reference for implemented and tested components. Methodology and status live in `docs/PROJECT_CONTEXT.md`, and experiment history and results in `docs/WORK_LOG.md`. This file describes only code that exists and is covered by tests.

Status: covers Phase 4a (chronological split), Phase 4b (baseline modeling, committed in `4b5b94b`) and Phase 4c (threshold analysis and temporal stability, approved and committed in `cffc32c`).

## Running

From the repository root, using the designated environment (`/Users/duc/Desktop/env/bin/python`):

```bash
python -m unittest discover -s tests -v          # all tests
python -m src.models.run_baselines               # Phase 4b experiments
python -m src.models.run_baselines --data PATH --output PATH --random-state 42
python -m src.models.run_threshold_analysis     # Phase 4c threshold analysis (same options)
```

Both runners read `data/raw/creditcard.csv` by default. They write `reports/phase4b_baseline_results.json` (about 5 seconds) and `reports/phase4c_threshold_results.json` (about 9.5 MB, about 3 seconds).

## Data flow

```
CSV ──► DatasetConfig (column roles)
          │
          ├─► partitions: assign_partitions (chronological)  ─┐  same labels/dtype:
          │               assign_stratified_group_partitions ─┘  train / validation / test
          ▼
   fit_and_evaluate: Pipeline[preprocess ► model].fit(train rows only)
          │          predict_proba / predict on validation rows
          ▼
   compute_binary_metrics ──► results dict / JSON
```

Test rows are never passed to a model.

Phase 4c (chronological split only):

```
fit_and_score(builder=build_logistic_regression) ──► one fit on train, one predict_proba on validation
          │  y_true, y_score (validation index and order preserved)
          ├─► threshold_curve + select_max_f1_threshold        ── full validation (retrospective)
          ├─► split_two_time_windows(validation Time)
          │      ├─ window_1: curve + max-F1 ──► frozen threshold ──► evaluate_threshold(window_2)
          │      └─ window_2: curve + max-F1 (retrospective diagnostic only)
          ▼
   dumps() ──► reports/phase4c_threshold_results.json
```

## `src/config.py`

`DatasetConfig` is a frozen dataclass that holds the column roles for one dataset:

| Field | Meaning |
|---|---|
| `name`, `path` | Identifier and CSV location |
| `target_column`, `positive_label` | Binary label and its fraud value |
| `feature_columns` | Explicit features; `None` means all columns except the target and `excluded_feature_columns` |
| `excluded_feature_columns` | Columns never used as model features |
| `time_column` | Optional timestamp used for chronological splitting |
| `duplicate_group_columns` | `None` means all columns except the target (the Phase 3 definition); `()` disables grouping |

`resolve_feature_columns(df)` and `resolve_duplicate_group_columns(df)` validate the configuration against the data. The target can never be a feature, and a column cannot be both selected and excluded. `CREDITCARD` uses target `Class` and time `Time`, and excludes raw `Time` from the features, giving 29 features: `V1`–`V28` and `Amount`. The exclusion was approved because raw `Time` is elapsed seconds within this 48-hour dataset, so the chronological validation period lies entirely outside the training time range. It is not a claim that temporal information has no predictive value. `Time` is still used for splitting and duplicate grouping.

## `src/data/split.py` (Phase 4a)

- `assign_partitions(df, time_column="Time")`: train for `Time ≤ 132928`, validation for `132929–151328`, test for `≥ 151329`. It returns an ordered categorical Series (`PARTITION_DTYPE`) aligned to `df.index`. Time values must be numeric, finite, non-negative whole seconds.
- `split_partitions(df)`: returns a dict of three frames that keep the original index and order.
- `count_cross_partition_duplicate_groups(df, partitions, feature_columns=None)`: the number of identical-feature groups that span partitions. It should be 0.

The boundaries are constants specific to this dataset.

## `src/data/random_split.py` (Phase 4b)

`assign_stratified_group_partitions(df, target_column, group_columns=None, fractions=(0.70, 0.15, 0.15), random_state=42)`:

1. Group rows by identical `group_columns` values. Group IDs come from `groupby(sort=True)`, so they do not depend on row order.
2. Reject any group containing more than one label.
3. For each class separately, shuffle that class's groups with `numpy.random.default_rng(random_state)`. Then allocate groups in shuffled order by cumulative row count: train up to `round(f_train·n)`, validation up to `round((f_train+f_val)·n)`, and test for the remainder.

The approved configuration is 70/15/15, seed 42, stratified by `Class`, with groups defined by all columns except `Class`. All rows are preserved. The output has the same format as `assign_partitions`. The function is deterministic for a given input and seed and does not modify its input. On the real data with seed 42: train 199,364/344, validation 42,722/74 and test 42,721/74 (rows/frauds), with 0 duplicate groups crossing partitions.

## `src/features/preprocessing.py`

- `build_preprocessor(feature_columns)`: an unfitted `ColumnTransformer` with a `StandardScaler` on the features and `remainder="drop"`. Any column that is not a feature, such as the target or `Time`, cannot reach the model.
- `validate_feature_frame(X, feature_columns)`: raises on non-numeric or boolean features and on NaN or infinite values. There is no imputation or encoding, so such data must be handled explicitly.

## `src/models/baselines.py`

Builders have the signature `(feature_columns, random_state) -> unfitted Pipeline[("preprocess", …), ("model", …)]`. They are registered in `BASELINE_BUILDERS`:

| Key | Estimator | Settings |
|---|---|---|
| `majority_class` | `DummyClassifier` | `strategy="prior"`: always predicts the training majority class; its probability is the training prior |
| `logistic_regression_balanced` | `LogisticRegression` | `class_weight="balanced"`, `C=1.0`, `solver="lbfgs"`, `max_iter=1000`, `random_state` |

The settings are fixed and untuned.

## `src/evaluation/metrics.py`

`compute_binary_metrics(y_true, y_score, y_pred, positive_label=1)` returns `n_rows`, `n_positive`, `prevalence`, `average_precision`, `roc_auc`, `precision`, `recall`, `n_predicted_positive`, and `confusion_matrix` (`tn`/`fp`/`fn`/`tp`). Undefined values are NaN (`null` in JSON):

- precision when there are no positive predictions;
- recall and average precision when there are no positives;
- ROC-AUC when only one class is present.

The metrics fall into two groups:

- **Ranking metrics:** `average_precision` and `roc_auc`, computed from `y_score`, the positive-class probability.
- **Diagnostic metrics:** `precision`, `recall` and `confusion_matrix`, computed from `y_pred`, the estimator's default `predict()`. That is probability strictly greater than 0.5 for Logistic Regression (scikit-learn's binary `predict` uses decision function > 0), and always the majority class for the majority baseline.

0.5 is not an optimal or production threshold.

## `src/models/experiment.py`

- `fit_and_score(df, partitions, config, builder, random_state, evaluation_partition="validation")` fits one pipeline on the `train` rows and calls `predict_proba` once on the evaluation rows. It returns a `ScoredModel` with the fitted `pipeline`, `evaluation_rows`, `y_true`, `y_score` (a Series aligned to the evaluation index, in original row order), `convergence_warnings` and `n_iter`. It applies the same validation and test-partition refusal as `fit_and_evaluate`. Added in Phase 4c by extracting code from `fit_and_evaluate`; Phase 4b output is unchanged.
- `fit_and_evaluate(df, partitions, config, builders, random_state, evaluation_partition="validation")` calls `fit_and_score` for each builder, then `predict` for the diagnostic metrics. It resolves the features from `config` and validates them. It fits each pipeline on the `train` rows and scores the evaluation rows. It returns `{model: {"metrics", "convergence_warnings", "n_iter"}}`. It raises if the evaluation partition is `test` or `train`, if the partitions are misaligned or use unknown labels, or if training contains no positives.
- `summarize_partitions(df, partitions, config)` reports rows, positives and prevalence per partition, plus the cross-partition duplicate-group count.

## `src/models/run_baselines.py`

`run(config, random_state)` runs Experiment A (chronological) and Experiment B (stratified random) and returns a JSON-ready dict. The dict contains the dataset SHA-256, feature list, model settings, split settings, library versions, partition summaries and per-model validation metrics; the two experiments are kept separate. `main()` is the CLI.

## `src/evaluation/threshold.py` (Phase 4c)

All functions validate their inputs. They reject empty inputs, length mismatches, non-numeric or non-finite scores, scores outside [0, 1], and missing labels. They accept a `positive_label`.

- `threshold_candidates(y_score)`: unique scores, ascending (`numpy.unique`).
- `threshold_curve(y_true, y_score)`: one row per candidate, returned as columns (`CURVE_COLUMNS`): `threshold`, `tp`, `fp`, `tn`, `fn`, `precision`, `recall`, `f1`, `n_predicted_positive`, `alert_rate`. Counts are exact: TP at threshold `t` is the number of positives with score ≥ `t`, found by binary search over the sorted scores, in O(n log n).
- `evaluate_threshold(y_true, y_score, threshold)`: the same row for one fixed threshold, which need not be a candidate.
- `select_max_f1_threshold(curve)`: the row with maximal F1, choosing the lowest threshold on ties. It raises if F1 is undefined everywhere.

Definitions:

- Decision rule: `score >= threshold`.
- `F1 = 2TP / (2TP + FP + FN)`. This equals `2PR/(P+R)` whenever both are defined. Because it is one division of integers, equal F1 values compare exactly equal, so ties are detected reliably.
- Undefined values are NaN (JSON `null`): precision when there are no predicted positives, recall when there are no positives, and F1 when either is undefined. F1 is 0 when TP = 0 and both precision and recall are defined.

## `src/data/temporal_windows.py` (Phase 4c)

`split_two_time_windows(times)` returns a `TimeWindows` object with these fields:

- `labels`: ordered categorical `window_1`/`window_2`, aligned to `times.index`;
- `window_1_end_time`, the inclusive end of window 1, and `window_2_start_time`;
- `n_window_1` and `n_window_2`.

The boundary is chosen among the gaps between consecutive unique timestamps to minimize `|n_W1 − n_W2|`, taking the earliest on ties. Rows with equal timestamps always share a window. The function raises on non-Series or non-numeric input, on NaN or infinite values, and on fewer than two distinct timestamps.

## `src/models/run_threshold_analysis.py` (Phase 4c)

- `analyse(df, config, random_state, builder)` runs the whole protocol in memory. The `builder` parameter lets tests record which rows reach each call.
- `run(config, random_state)` loads `config.path` and calls `analyse`.
- `dumps(output)` writes strict JSON: NaN becomes `null`, `allow_nan=False` is enforced, and floats keep full precision. Each threshold-curve column is written on one line.
- `main()` is the CLI.

Top-level output keys:

- `dataset`, `model`, `evaluation_protocol`, `environment`;
- `full_validation`: `partition_summary`, `selected_threshold`, `metrics`, `threshold_curve`;
- `temporal_windows`: `boundary_time` (inclusive end of window 1), `window_2_start_time`, the window sizes, and `window_1`/`window_2` analyses;
- `threshold_transfer`: `transferred_threshold`, `source_selection_f1`, `metrics`, `retrospective_optimum`, and `f1_difference` (transferred − retrospective).
- `common_reference_thresholds` (added 2026-11-08; descriptive, separate from `threshold_transfer`): `role`, `note`, `decision_rule` (`y_score >= threshold`), `window_prevalence`, and `thresholds` with two entries, `fixed_0_5` and `full_validation_max_f1` (retrospective). Each has `threshold`, `description`, and `window_1`/`window_2` rows in the `evaluate_threshold` format. The rows are computed from the scores already in memory; nothing is refitted or rescored.

## Adding a dataset

1. Add a `DatasetConfig` with the target, the features or exclusions, and optionally a time column and duplicate-group columns.
2. Provide partitions. The random split works for any config. A chronological split for a new dataset needs its own documented boundaries, because the `split.py` constants are specific to the credit-card data.
3. Call `fit_and_evaluate(df, partitions, config, BASELINE_BUILDERS, seed)`. `run_baselines.run` currently wires up both credit-card experiments: A (chronological split) and B (stratified random split).

Feature columns must be numeric and finite. Categorical encoding and imputation are not implemented.

## Tests

| File | Tests | Covers |
|---|---:|---|
| `tests/test_split.py` | 36 | Chronological boundaries, grouping, preservation, input validation, documented real-data counts |
| `tests/test_random_split.py` | 23 | Proportions, stratification, duplicate groups, conflicting labels, determinism and row-order independence, input validation, real-data counts for seed 42 |
| `tests/test_baselines.py` | 38 | Config resolution, train-only scaler fitting, model invariance to validation/test changes, test rows never scored, refusal to evaluate on test, baseline behavior, LR probabilities and reproducibility, hand-checked metrics, compatibility with both splits, end-to-end CLI run, real-data result regression |
| `tests/test_threshold.py` | 29 | Unique candidates, inclusive `>=` rule, hand-calculated precision/recall/F1/confusion counts, brute-force count check, undefined values, lowest-threshold tie-break, rejection of invalid inputs; window boundary balance, earliest-boundary ties, timestamp integrity, alignment, determinism, invalid timestamps |
| `tests/test_threshold_analysis.py` | 21 | One fit on train only; one `predict_proba` on validation only; test rows never used; windows partition validation chronologically; Window 1 threshold transferred unchanged and independent of Window 2 labels; transfer metrics recomputed independently; common-reference thresholds (0.5 and full-validation max-F1) recomputed independently per window, with window counts summing to full-validation counts; reproducibility; strict JSON with NaN as null; full float precision; end-to-end CLI run; real-data result regression |

Real-data tests are skipped when `data/raw/creditcard.csv` is absent.
