"""Integration tests for the Phase 4c runner (``src/models/run_threshold_analysis.py``).

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
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CREDITCARD, DatasetConfig
from src.data.split import TEST, TRAIN, VALIDATION, assign_partitions
from src.data.temporal_windows import WINDOW_1, WINDOW_2, split_two_time_windows
from src.evaluation.threshold import evaluate_threshold
from src.models.baselines import build_logistic_regression
from src.models.experiment import fit_and_score
from src.models import run_threshold_analysis as runner

RAW_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "creditcard.csv"

CONFIG = DatasetConfig(
    name="synthetic",
    path=Path("unused.csv"),
    target_column="Class",
    excluded_feature_columns=("Time",),
    time_column="Time",
)
FEATURES = ["V1", "V2", "Amount"]


def _frame(n=8000, positive_rate=0.05, seed=0) -> pd.DataFrame:
    """Synthetic data spanning the chronological boundaries, with a noisy signal in V1."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < positive_rate).astype(int)
    return pd.DataFrame(
        {
            "Time": np.sort(rng.integers(0, 172_793, size=n)),
            "V1": rng.normal(size=n) + 1.5 * y,
            "V2": rng.normal(size=n),
            "Amount": rng.exponential(100.0, size=n),
            "Class": y,
        }
    )


class _Recorder:
    """Builder wrapper recording which rows reach fit / predict_proba / predict."""

    def __init__(self):
        self.calls: list[tuple[str, pd.Index]] = []

    def __call__(self, feature_columns, random_state):
        pipeline = build_logistic_regression(feature_columns, random_state)
        for method in ("fit", "predict_proba", "predict"):
            original = getattr(pipeline, method)

            def wrapped(X, *args, _original=original, _method=method, **kwargs):
                self.calls.append((_method, X.index))
                return _original(X, *args, **kwargs)

            setattr(pipeline, method, wrapped)
        return pipeline


class TestProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = _frame()
        cls.partitions = assign_partitions(cls.df).astype(str)
        cls.recorder = _Recorder()
        cls.output = runner.analyse(cls.df, CONFIG, 42, builder=cls.recorder)

    def _rows(self, partition):
        return set(self.df.index[(self.partitions == partition).to_numpy()])

    def test_model_fitted_once_on_train_only(self):
        fits = [idx for method, idx in self.recorder.calls if method == "fit"]
        self.assertEqual(len(fits), 1)
        self.assertEqual(set(fits[0]), self._rows(TRAIN))

    def test_validation_scored_exactly_once_and_test_never(self):
        scored = [idx for method, idx in self.recorder.calls if method == "predict_proba"]
        self.assertEqual(len(scored), 1)
        self.assertEqual(set(scored[0]), self._rows(VALIDATION))
        self.assertFalse(set(scored[0]) & self._rows(TEST))
        self.assertEqual([m for m, _ in self.recorder.calls], ["fit", "predict_proba"])

    def test_full_validation_curve_covers_unique_validation_scores(self):
        fv = self.output["full_validation"]
        n_valid = len(self._rows(VALIDATION))
        self.assertEqual(fv["partition_summary"]["n_rows"], n_valid)
        self.assertEqual(self.output["evaluation_protocol"]["n_validation_rows_scored"], n_valid)
        curve = fv["threshold_curve"]
        self.assertEqual(curve["threshold"], sorted(set(curve["threshold"])))
        self.assertEqual(curve["n_predicted_positive"][0], n_valid)  # lowest score flags all
        self.assertEqual(fv["metrics"]["threshold"], fv["selected_threshold"])
        self.assertEqual(fv["metrics"]["f1"], np.nanmax(curve["f1"]))

    def test_windows_partition_validation_chronologically(self):
        tw = self.output["temporal_windows"]
        w1, w2 = tw[WINDOW_1]["partition_summary"], tw[WINDOW_2]["partition_summary"]
        self.assertEqual(w1["n_rows"] + w2["n_rows"], len(self._rows(VALIDATION)))
        self.assertEqual((w1["n_rows"], w2["n_rows"]), (tw["n_window_1"], tw["n_window_2"]))
        self.assertLessEqual(w1["max_time"], tw["boundary_time"])
        self.assertGreater(w2["min_time"], tw["boundary_time"])
        self.assertEqual(w2["min_time"], tw["window_2_start_time"])
        self.assertEqual(w1["n_positive"] + w2["n_positive"],
                         self.output["full_validation"]["partition_summary"]["n_positive"])

    def test_window1_threshold_applied_unchanged_to_window2(self):
        tw, tt = self.output["temporal_windows"], self.output["threshold_transfer"]
        self.assertEqual(tt["transferred_threshold"], tw[WINDOW_1]["selected_threshold"])
        self.assertEqual(tt["metrics"]["threshold"], tw[WINDOW_1]["selected_threshold"])
        self.assertEqual(tt["source_selection_f1"], tw[WINDOW_1]["metrics"]["f1"])
        self.assertEqual(tt["retrospective_optimum"], tw[WINDOW_2]["metrics"])
        self.assertAlmostEqual(tt["f1_difference"], tt["metrics"]["f1"] - tt["retrospective_optimum"]["f1"])
        self.assertLessEqual(tt["f1_difference"], 0.0)  # a frozen threshold cannot beat the optimum

    def test_transferred_metrics_recomputed_independently(self):
        validation = self.df.loc[(self.partitions == VALIDATION).to_numpy()]
        scored = fit_and_score(self.df, assign_partitions(self.df), CONFIG, build_logistic_regression, 42)
        windows = split_two_time_windows(validation["Time"])
        mask = (windows.labels == WINDOW_2).to_numpy()
        expected = evaluate_threshold(
            validation["Class"].to_numpy()[mask], scored.y_score.to_numpy()[mask],
            self.output["threshold_transfer"]["transferred_threshold"],
        )
        self.assertEqual(self.output["threshold_transfer"]["metrics"], expected)


class TestIsolation(unittest.TestCase):
    def setUp(self):
        self.df = _frame()
        self.baseline = runner.analyse(self.df, CONFIG, 42)

    def _window2_mask(self, df):
        partitions = assign_partitions(df).to_numpy()
        boundary = self.baseline["temporal_windows"]["boundary_time"]
        return (partitions == VALIDATION) & (df["Time"].to_numpy() > boundary)

    def test_window2_outcomes_do_not_influence_transferred_threshold(self):
        altered = self.df.copy()
        mask = self._window2_mask(altered)
        altered.loc[mask, "Class"] = 1 - altered.loc[mask, "Class"]
        result = runner.analyse(altered, CONFIG, 42)
        self.assertEqual(result["threshold_transfer"]["transferred_threshold"],
                         self.baseline["threshold_transfer"]["transferred_threshold"])
        self.assertEqual(result["temporal_windows"][WINDOW_1], self.baseline["temporal_windows"][WINDOW_1])
        self.assertNotEqual(result["temporal_windows"][WINDOW_2]["metrics"],
                            self.baseline["temporal_windows"][WINDOW_2]["metrics"])

    def test_test_rows_never_used(self):
        corrupted = self.df.copy()
        test_mask = assign_partitions(corrupted).to_numpy() == TEST
        corrupted.loc[test_mask, FEATURES] = np.nan
        corrupted.loc[test_mask, "Class"] = 1 - corrupted.loc[test_mask, "Class"]
        result = runner.analyse(corrupted, CONFIG, 42)
        for key in ("full_validation", "temporal_windows", "threshold_transfer", "model"):
            self.assertEqual(result[key], self.baseline[key], key)

    def test_results_reproducible(self):
        again = runner.analyse(self.df, CONFIG, 42)
        self.assertEqual(runner.dumps(again), runner.dumps(self.baseline))

    def test_time_column_required(self):
        config = DatasetConfig("x", Path("x.csv"), "Class", excluded_feature_columns=("Time",))
        with self.assertRaises(ValueError):
            runner.analyse(self.df, config, 42)


class TestSerialization(unittest.TestCase):
    def test_nan_serialized_as_null_and_output_is_strict_json(self):
        output = {
            "a": float("nan"),
            "nested": {"threshold_curve": {"threshold": [0.1, 0.2], "f1": [float("nan"), 0.5]}},
        }
        text = runner.dumps(output)

        def reject(constant):
            raise ValueError(f"non-standard JSON constant {constant}")

        parsed = json.loads(text, parse_constant=reject)
        self.assertIsNone(parsed["a"])
        self.assertEqual(parsed["nested"]["threshold_curve"]["f1"], [None, 0.5])
        self.assertTrue(math.isnan(output["a"]))  # input not modified

    def test_full_precision_preserved(self):
        value = 0.9999978229787393
        parsed = json.loads(runner.dumps({"threshold_curve": {"threshold": [value]}}))
        self.assertEqual(parsed["threshold_curve"]["threshold"][0], value)

    def test_cli_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, out = Path(tmp) / "data.csv", Path(tmp) / "out.json"
            _frame().to_csv(data, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                runner.main(["--data", str(data), "--output", str(out)])
            parsed = json.loads(out.read_text())
        self.assertEqual(
            set(parsed),
            {"phase", "dataset", "model", "evaluation_protocol", "environment",
             "full_validation", "temporal_windows", "threshold_transfer"},
        )
        self.assertEqual(parsed["full_validation"]["selection_objective"], "maximize_f1")
        self.assertEqual(parsed["full_validation"]["tie_breaking"], "lowest_threshold")
        self.assertEqual(parsed["evaluation_protocol"]["test_partition"], "not scored")


@unittest.skipUnless(RAW_DATA_PATH.exists(), f"raw dataset not found at {RAW_DATA_PATH}")
class TestThresholdAnalysisOnRealData(unittest.TestCase):
    """Regression check of the documented Phase 4c results (docs/WORK_LOG.md)."""

    @classmethod
    def setUpClass(cls):
        cls.output = runner.run(CREDITCARD, 42)

    def test_scored_partition_matches_phase3(self):
        s = self.output["full_validation"]["partition_summary"]
        self.assertEqual((s["n_rows"], s["n_positive"]), (42_722, 56))
        self.assertEqual(self.output["full_validation"]["n_thresholds"], 41_807)
        self.assertEqual(self.output["model"]["convergence_warnings"], [])

    def test_documented_full_validation_result(self):
        m = self.output["full_validation"]["metrics"]
        self.assertAlmostEqual(m["threshold"], 0.999995978456591, places=12)
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (43, 4, 13, 42_662))

    def test_documented_windows(self):
        tw = self.output["temporal_windows"]
        self.assertEqual(tw["boundary_time"], 142_280)
        self.assertEqual((tw["n_window_1"], tw["n_window_2"]), (21_362, 21_360))
        self.assertEqual(tw[WINDOW_1]["partition_summary"]["n_positive"], 23)
        self.assertEqual(tw[WINDOW_2]["partition_summary"]["n_positive"], 33)

    def test_documented_transfer(self):
        tt = self.output["threshold_transfer"]
        self.assertAlmostEqual(tt["transferred_threshold"], 0.9999978229787393, places=12)
        m, r = tt["metrics"], tt["retrospective_optimum"]
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (24, 3, 9, 21_324))
        self.assertEqual((r["tp"], r["fp"], r["fn"], r["tn"]), (24, 1, 9, 21_326))
        # The transfer uses the Window 1 threshold, not the full-validation optimum.
        self.assertNotEqual(tt["transferred_threshold"], self.output["full_validation"]["selected_threshold"])


if __name__ == "__main__":
    unittest.main()
