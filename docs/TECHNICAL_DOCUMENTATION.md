# FraudGuard Technical Documentation

Reference for implemented and tested components. Methodology and status live in `docs/PROJECT_CONTEXT.md`, and experiment history and results in `docs/WORK_LOG.md`. This file describes only code that exists and is covered by tests.

Status: covers Phase 4a (chronological split) and Phase 4b (baseline modeling). The Phase 4b decisions were approved by the mentor on 2026-11-07, and Phase 4b completion awaits independent mentor approval.

## Running

From the repository root, using the designated environment (`/Users/duc/Desktop/env/bin/python`):

```bash
python -m unittest discover -s tests -v          # all tests
python -m src.models.run_baselines               # Phase 4b experiments
python -m src.models.run_baselines --data PATH --output PATH --random-state 42
```

The runner reads `data/raw/creditcard.csv` by default and writes `reports/phase4b_baseline_results.json`. It finishes in about 5 seconds.

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
- **Diagnostic metrics:** `precision`, `recall` and `confusion_matrix`, computed from `y_pred`, the estimator's default `predict()`. That is probability ≥ 0.5 for Logistic Regression, and always the majority class for the majority baseline.

0.5 is not an optimal or production threshold.

## `src/models/experiment.py`

- `fit_and_evaluate(df, partitions, config, builders, random_state, evaluation_partition="validation")` resolves the features from `config` and validates them. It fits each pipeline on the `train` rows and scores the evaluation rows. It returns `{model: {"metrics", "convergence_warnings", "n_iter"}}`. It raises if the evaluation partition is `test` or `train`, if the partitions are misaligned or use unknown labels, or if training contains no positives.
- `summarize_partitions(df, partitions, config)` reports rows, positives and prevalence per partition, plus the cross-partition duplicate-group count.

## `src/models/run_baselines.py`

`run(config, random_state)` runs Experiment A (chronological) and Experiment B (stratified random) and returns a JSON-ready dict. The dict contains the dataset SHA-256, feature list, model settings, split settings, library versions, partition summaries and per-model validation metrics; the two experiments are kept separate. `main()` is the CLI.

## Adding a dataset

1. Add a `DatasetConfig` with the target, the features or exclusions, and optionally a time column and duplicate-group columns.
2. Provide partitions. The random split works for any config. A chronological split for a new dataset needs its own documented boundaries, because the `split.py` constants are specific to the credit-card data.
3. Call `fit_and_evaluate(df, partitions, config, BASELINE_BUILDERS, seed)`. `run_baselines.run` currently wires up the credit-card chronological split.

Feature columns must be numeric and finite. Categorical encoding and imputation are not implemented.

## Tests

| File | Tests | Covers |
|---|---:|---|
| `tests/test_split.py` | 36 | Chronological boundaries, grouping, preservation, input validation, documented real-data counts |
| `tests/test_random_split.py` | 23 | Proportions, stratification, duplicate groups, conflicting labels, determinism and row-order independence, input validation, real-data counts for seed 42 |
| `tests/test_baselines.py` | 38 | Config resolution, train-only scaler fitting, model invariance to validation/test changes, test rows never scored, refusal to evaluate on test, baseline behavior, LR probabilities and reproducibility, hand-checked metrics, compatibility with both splits, end-to-end CLI run, real-data result regression |

Real-data tests are skipped when `data/raw/creditcard.csv` is absent.
