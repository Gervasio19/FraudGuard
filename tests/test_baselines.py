"""Tests for Phase 4b baseline modeling: config, preprocessing, models, metrics, runner.

Run from the repository root:

    python -m unittest discover -s tests -v

Real-data tests read ``data/raw/creditcard.csv`` (read-only) and are skipped
when the file is not present.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src.config import CREDITCARD, DatasetConfig
from src.data.random_split import assign_stratified_group_partitions
from src.data.split import PARTITION_DTYPE, TEST, TRAIN, VALIDATION, assign_partitions
from src.evaluation.metrics import compute_binary_metrics
from src.features.preprocessing import build_preprocessor, validate_feature_frame
from src.models.baselines import (
    BASELINE_BUILDERS,
    LOGISTIC_REGRESSION,
    MAJORITY_BASELINE,
    build_logistic_regression,
    build_majority_baseline,
)
from src.models.experiment import fit_and_evaluate, summarize_partitions
from src.models import run_baselines

RAW_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "creditcard.csv"

CONFIG = DatasetConfig(
    name="synthetic",
    path=Path("unused.csv"),
    target_column="Class",
    excluded_feature_columns=("Time",),
    time_column="Time",
)
FEATURES = ["V1", "V2", "Amount"]


def _frame(n=3000, positive_rate=0.05, seed=0) -> pd.DataFrame:
    """Synthetic data with Time spanning the chronological boundaries and a learnable signal."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < positive_rate).astype(int)
    return pd.DataFrame(
        {
            "Time": np.sort(rng.integers(0, 172_793, size=n)),
            "V1": rng.normal(size=n) + 2.5 * y,
            "V2": rng.normal(size=n),
            "Amount": rng.exponential(100.0, size=n),
            "Class": y,
        }
    )


def _manual_partitions(df: pd.DataFrame) -> pd.Series:
    labels = np.where(np.arange(len(df)) < 0.6 * len(df), TRAIN,
                      np.where(np.arange(len(df)) < 0.8 * len(df), VALIDATION, TEST))
    return pd.Series(pd.Categorical(labels, dtype=PARTITION_DTYPE), index=df.index, name="partition")


def _capturing_builders(store: dict):
    """Wrap the baseline builders so tests can inspect the fitted pipelines."""
    def wrap(name, builder):
        def build(feature_columns, random_state):
            store[name] = builder(feature_columns, random_state)
            return store[name]
        return build
    return {name: wrap(name, builder) for name, builder in BASELINE_BUILDERS.items()}


class TestDatasetConfig(unittest.TestCase):
    def test_default_features_exclude_target_and_excluded_columns(self):
        self.assertEqual(CONFIG.resolve_feature_columns(_frame()), FEATURES)

    def test_creditcard_config_features(self):
        columns = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]
        df = pd.DataFrame(columns=columns)
        features = CREDITCARD.resolve_feature_columns(df)
        self.assertEqual(features, [f"V{i}" for i in range(1, 29)] + ["Amount"])
        self.assertEqual(CREDITCARD.resolve_duplicate_group_columns(df), columns[:-1])

    def test_explicit_features(self):
        config = replace(CONFIG, feature_columns=("V2",), excluded_feature_columns=())
        self.assertEqual(config.resolve_feature_columns(_frame()), ["V2"])

    def test_target_as_feature_rejected(self):
        with self.assertRaises(ValueError):
            replace(CONFIG, feature_columns=("V1", "Class")).resolve_feature_columns(_frame())

    def test_selected_and_excluded_overlap_rejected(self):
        with self.assertRaises(ValueError):
            replace(CONFIG, feature_columns=("Time",)).resolve_feature_columns(_frame())

    def test_missing_feature_rejected(self):
        with self.assertRaises(KeyError):
            replace(CONFIG, feature_columns=("V9",)).resolve_feature_columns(_frame())

    def test_other_dataset_column_names(self):
        df = _frame().rename(columns={"Class": "is_fraud", "Time": "event_ts"})
        config = DatasetConfig("other", Path("x.csv"), "is_fraud", excluded_feature_columns=("event_ts",))
        self.assertEqual(config.resolve_feature_columns(df), FEATURES)
        self.assertEqual(config.resolve_duplicate_group_columns(df), ["event_ts", *FEATURES])

    def test_duplicate_grouping_can_be_disabled(self):
        self.assertEqual(replace(CONFIG, duplicate_group_columns=()).resolve_duplicate_group_columns(_frame()), [])


class TestPreprocessing(unittest.TestCase):
    def test_scaler_fitted_on_training_rows_only(self):
        df = _frame()
        train, other = df.iloc[:2000], df.iloc[2000:].copy()
        other[FEATURES] *= 1000.0  # wildly different evaluation data
        preprocessor = build_preprocessor(FEATURES).fit(train)
        scaler = preprocessor.named_transformers_["standardize"]
        np.testing.assert_allclose(scaler.mean_, train[FEATURES].mean().to_numpy())
        np.testing.assert_allclose(scaler.scale_, train[FEATURES].std(ddof=0).to_numpy())
        before = scaler.mean_.copy()
        preprocessor.transform(other)
        np.testing.assert_array_equal(scaler.mean_, before)

    def test_unlisted_columns_dropped(self):
        preprocessor = build_preprocessor(FEATURES).fit(_frame())
        self.assertEqual(list(preprocessor.get_feature_names_out()), FEATURES)

    def test_no_features_rejected(self):
        with self.assertRaises(ValueError):
            build_preprocessor([])

    def test_non_numeric_and_missing_features_rejected(self):
        df = _frame()
        with self.assertRaises(TypeError):
            validate_feature_frame(df.assign(V2="a"), FEATURES)
        with self.assertRaises(ValueError):
            validate_feature_frame(df.assign(V2=np.nan), FEATURES)
        with self.assertRaises(ValueError):
            validate_feature_frame(df.assign(V2=np.inf), FEATURES)


class TestBaselineModels(unittest.TestCase):
    def setUp(self):
        self.df = _frame()
        self.train, self.valid = self.df.iloc[:2000], self.df.iloc[2000:]

    def test_majority_baseline_predicts_majority_with_prior_probabilities(self):
        model = build_majority_baseline(FEATURES, 0).fit(self.train[FEATURES], self.train["Class"])
        self.assertIsInstance(model.named_steps["model"], DummyClassifier)
        self.assertTrue((model.predict(self.valid[FEATURES]) == 0).all())
        proba = model.predict_proba(self.valid[FEATURES])[:, 1]
        np.testing.assert_allclose(proba, self.train["Class"].mean())

    def test_logistic_regression_configuration(self):
        model = build_logistic_regression(FEATURES, 5).named_steps["model"]
        self.assertIsInstance(model, LogisticRegression)
        self.assertEqual(model.class_weight, "balanced")
        self.assertEqual(model.random_state, 5)

    def test_logistic_regression_trains_and_produces_probabilities(self):
        model = build_logistic_regression(FEATURES, 0).fit(self.train[FEATURES], self.train["Class"])
        proba = model.predict_proba(self.valid[FEATURES])
        self.assertEqual(proba.shape, (len(self.valid), 2))
        self.assertTrue(((proba >= 0) & (proba <= 1)).all())
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)
        # The synthetic signal is in V1, so it must receive the largest positive weight.
        coef = model.named_steps["model"].coef_[0]
        self.assertEqual(int(np.argmax(coef)), FEATURES.index("V1"))

    def test_logistic_regression_is_reproducible(self):
        fits = [
            build_logistic_regression(FEATURES, 0).fit(self.train[FEATURES], self.train["Class"])
            for _ in range(2)
        ]
        np.testing.assert_array_equal(
            fits[0].predict_proba(self.valid[FEATURES]), fits[1].predict_proba(self.valid[FEATURES])
        )

    def test_builders_return_unfitted_independent_pipelines(self):
        a, b = build_logistic_regression(FEATURES, 0), build_logistic_regression(FEATURES, 0)
        self.assertIsNot(a.named_steps["model"], b.named_steps["model"])
        self.assertFalse(hasattr(a.named_steps["model"], "coef_"))


class TestMetrics(unittest.TestCase):
    def test_hand_computed_example(self):
        m = compute_binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], [0, 0, 0, 1])
        self.assertAlmostEqual(m["average_precision"], (1.0 + 2 / 3) / 2)
        self.assertAlmostEqual(m["roc_auc"], 0.75)
        self.assertEqual(m["precision"], 1.0)
        self.assertEqual(m["recall"], 0.5)
        self.assertEqual(m["confusion_matrix"], {"tn": 2, "fp": 0, "fn": 1, "tp": 1})
        self.assertEqual((m["n_rows"], m["n_positive"], m["n_predicted_positive"]), (4, 2, 1))
        self.assertEqual(m["prevalence"], 0.5)

    def test_constant_scores_give_prevalence_ap_and_half_auc(self):
        m = compute_binary_metrics([0, 0, 0, 1], [0.2] * 4, [0] * 4)
        self.assertAlmostEqual(m["average_precision"], 0.25)
        self.assertAlmostEqual(m["roc_auc"], 0.5)

    def test_undefined_values_reported_as_nan(self):
        m = compute_binary_metrics([0, 0, 1], [0.1, 0.2, 0.3], [0, 0, 0])
        self.assertTrue(math.isnan(m["precision"]))
        self.assertEqual(m["recall"], 0.0)
        m = compute_binary_metrics([0, 0], [0.1, 0.2], [0, 1])
        self.assertTrue(math.isnan(m["recall"]))
        self.assertTrue(math.isnan(m["average_precision"]))
        self.assertTrue(math.isnan(m["roc_auc"]))

    def test_custom_positive_label(self):
        m = compute_binary_metrics(["ok", "fraud"], [0.1, 0.9], ["ok", "fraud"], positive_label="fraud")
        self.assertEqual(m["confusion_matrix"], {"tn": 1, "fp": 0, "fn": 0, "tp": 1})

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            compute_binary_metrics([0, 1], [0.1], [0, 1])
        with self.assertRaises(ValueError):
            compute_binary_metrics([], [], [])
        with self.assertRaises(ValueError):
            compute_binary_metrics([0, 1], [0.1, np.nan], [0, 1])


class TestExperimentRunner(unittest.TestCase):
    def setUp(self):
        self.df = _frame()
        self.partitions = _manual_partitions(self.df)

    def test_returns_metrics_for_each_baseline(self):
        results = fit_and_evaluate(self.df, self.partitions, CONFIG, BASELINE_BUILDERS, 0)
        self.assertEqual(set(results), {MAJORITY_BASELINE, LOGISTIC_REGRESSION})
        n_valid = int((self.partitions == VALIDATION).sum())
        for result in results.values():
            self.assertEqual(result["metrics"]["n_rows"], n_valid)
        self.assertEqual(results[LOGISTIC_REGRESSION]["convergence_warnings"], [])
        self.assertGreater(results[LOGISTIC_REGRESSION]["metrics"]["average_precision"],
                           results[MAJORITY_BASELINE]["metrics"]["average_precision"])

    def test_models_fit_on_training_rows_only(self):
        store: dict = {}
        fit_and_evaluate(self.df, self.partitions, CONFIG, _capturing_builders(store), 0)
        scaler = store[LOGISTIC_REGRESSION].named_steps["preprocess"].named_transformers_["standardize"]
        train = self.df.loc[self.partitions == TRAIN, FEATURES]
        np.testing.assert_allclose(scaler.mean_, train.mean().to_numpy())
        prior = store[MAJORITY_BASELINE].named_steps["model"].class_prior_[1]
        self.assertAlmostEqual(prior, self.df.loc[self.partitions == TRAIN, "Class"].mean())

    def test_changing_validation_and_test_rows_does_not_change_fitted_model(self):
        altered = self.df.copy()
        not_train = (self.partitions != TRAIN).to_numpy()
        altered.loc[not_train, FEATURES] *= 50.0
        altered.loc[not_train, "Class"] = 1 - altered.loc[not_train, "Class"]
        coefs = []
        for frame in (self.df, altered):
            store: dict = {}
            fit_and_evaluate(frame, self.partitions, CONFIG, _capturing_builders(store), 0)
            lr = store[LOGISTIC_REGRESSION]
            coefs.append(np.r_[lr.named_steps["model"].coef_.ravel(), lr.named_steps["model"].intercept_])
        np.testing.assert_array_equal(coefs[0], coefs[1])

    def test_test_rows_are_never_scored(self):
        # Corrupt every test row; the run must still succeed because test rows are unused.
        df = self.df.copy()
        df.loc[(self.partitions == TEST).to_numpy(), FEATURES] = np.nan
        results = fit_and_evaluate(df, self.partitions, CONFIG, BASELINE_BUILDERS, 0)
        self.assertIn(LOGISTIC_REGRESSION, results)

    def test_evaluating_on_test_partition_refused(self):
        with self.assertRaises(ValueError):
            fit_and_evaluate(self.df, self.partitions, CONFIG, BASELINE_BUILDERS, 0, evaluation_partition=TEST)
        with self.assertRaises(ValueError):
            fit_and_evaluate(self.df, self.partitions, CONFIG, BASELINE_BUILDERS, 0, evaluation_partition=TRAIN)

    def test_excluded_time_column_not_used_as_feature(self):
        store: dict = {}
        fit_and_evaluate(self.df, self.partitions, CONFIG, _capturing_builders(store), 0)
        self.assertEqual(list(store[LOGISTIC_REGRESSION].feature_names_in_), FEATURES)

    def test_misaligned_or_unknown_partitions_rejected(self):
        with self.assertRaises(ValueError):
            fit_and_evaluate(self.df, self.partitions.reset_index(drop=True).iloc[::-1], CONFIG, BASELINE_BUILDERS, 0)
        bad = pd.Series(["train"] * len(self.df), index=self.df.index).astype(object)
        bad.iloc[0] = "holdout"
        with self.assertRaises(ValueError):
            fit_and_evaluate(self.df, bad, CONFIG, BASELINE_BUILDERS, 0)

    def test_training_without_positives_rejected(self):
        df = self.df.copy()
        df.loc[(self.partitions == TRAIN).to_numpy(), "Class"] = 0
        with self.assertRaises(ValueError):
            fit_and_evaluate(df, self.partitions, CONFIG, BASELINE_BUILDERS, 0)

    def test_compatible_with_chronological_split_output(self):
        partitions = assign_partitions(self.df, "Time")
        results = fit_and_evaluate(self.df, partitions, CONFIG, BASELINE_BUILDERS, 0)
        self.assertEqual(results[MAJORITY_BASELINE]["metrics"]["n_rows"], int((partitions == VALIDATION).sum()))

    def test_compatible_with_random_split_output(self):
        partitions = assign_stratified_group_partitions(self.df, "Class", random_state=0)
        results = fit_and_evaluate(self.df, partitions, CONFIG, BASELINE_BUILDERS, 0)
        summary = summarize_partitions(self.df, partitions, CONFIG)
        self.assertEqual(summary["cross_partition_duplicate_groups"], 0)
        self.assertEqual(results[LOGISTIC_REGRESSION]["metrics"]["n_positive"],
                         summary["partitions"][VALIDATION]["n_positive"])

    def test_results_reproducible(self):
        first = fit_and_evaluate(self.df, self.partitions, CONFIG, BASELINE_BUILDERS, 0)
        second = fit_and_evaluate(self.df, self.partitions, CONFIG, BASELINE_BUILDERS, 0)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))

    def test_end_to_end_cli_on_synthetic_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, out = Path(tmp) / "data.csv", Path(tmp) / "out.json"
            self.df.to_csv(data, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                run_baselines.main(["--data", str(data), "--output", str(out)])
            output = json.loads(out.read_text())
        self.assertEqual(set(output["experiments"]), {"A_chronological", "B_stratified_random"})
        self.assertEqual(output["evaluation_partition"], "validation")
        for experiment in output["experiments"].values():
            self.assertEqual(set(experiment["results"]), {MAJORITY_BASELINE, LOGISTIC_REGRESSION})


@unittest.skipUnless(RAW_DATA_PATH.exists(), f"raw dataset not found at {RAW_DATA_PATH}")
class TestBaselinesOnRealData(unittest.TestCase):
    """Regression check of the documented Phase 4b validation results (docs/WORK_LOG.md)."""

    @classmethod
    def setUpClass(cls):
        cls.output = run_baselines.run(CREDITCARD, random_state=42)

    def _metrics(self, experiment, model):
        return self.output["experiments"][experiment]["results"][model]["metrics"]

    def test_partitions_match_phase3_and_have_no_duplicate_leakage(self):
        chrono = self.output["experiments"]["A_chronological"]
        self.assertEqual(chrono["partitions"][TRAIN]["n_rows"], 199_364)
        self.assertEqual(chrono["partitions"][VALIDATION]["n_positive"], 56)
        for experiment in self.output["experiments"].values():
            self.assertEqual(experiment["cross_partition_duplicate_groups"], 0)

    def test_documented_chronological_results(self):
        lr = self._metrics("A_chronological", LOGISTIC_REGRESSION)
        self.assertAlmostEqual(lr["average_precision"], 0.8397, places=4)
        self.assertAlmostEqual(lr["roc_auc"], 0.9815, places=4)
        self.assertEqual(lr["confusion_matrix"], {"tn": 41409, "fp": 1257, "fn": 4, "tp": 52})
        majority = self._metrics("A_chronological", MAJORITY_BASELINE)
        self.assertAlmostEqual(majority["average_precision"], 56 / 42_722)
        self.assertEqual(majority["roc_auc"], 0.5)

    def test_documented_random_benchmark_results(self):
        lr = self._metrics("B_stratified_random", LOGISTIC_REGRESSION)
        self.assertAlmostEqual(lr["average_precision"], 0.7524, places=4)
        self.assertAlmostEqual(lr["roc_auc"], 0.9725, places=4)
        self.assertEqual(lr["confusion_matrix"], {"tn": 41711, "fp": 937, "fn": 9, "tp": 65})

    def test_logistic_regression_converged(self):
        for experiment in self.output["experiments"].values():
            self.assertEqual(experiment["results"][LOGISTIC_REGRESSION]["convergence_warnings"], [])


if __name__ == "__main__":
    unittest.main()
