"""Tests for Phase 5B-1: T_fit/T_cal partition, E2 usability, Phase 4 continuity.

Synthetic tests use generated data only. Real-data tests are skipped when
``data/raw/creditcard.csv`` is absent; they process train and validation rows
only (test rows are removed before any label is accessed).
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.config import CREDITCARD, PROJECT_ROOT, DatasetConfig
from src.data.split import TRAIN_END_TIME, VALIDATION_END_TIME
from src.data.train_subpartitions import (
    T_CAL, T_FIT, assign_train_subpartitions, check_temporal_integrity, require_estimable, summarize_subpartitions,
)
from src.data.validation_blocks import block_edges, draw_selection_blocks
from src.evaluation.e2_usability import e2_usability, split_counts, usability_decision
from src.models import phase4_continuity as pc
from src.models.phase5_settings import E2_SPLIT_SEED, T_CAL_START_TIME, T_FIT_END_TIME, VALIDATION_START_TIME
from src.models.run_phase5b1 import compute, drop_test_rows

DATA_AVAILABLE = CREDITCARD.path.is_file()
RESULTS_FILE = PROJECT_ROOT / "reports" / "phase5b1_results.json"
PHASE4B_FILE = PROJECT_ROOT / "reports" / "phase4b_baseline_results.json"
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]


def synthetic_frame(n=6000, seed=0) -> pd.DataFrame:
    """Rows spread over the full 0-172,792 s range with a learnable label."""
    rng = np.random.default_rng(seed)
    times = np.sort(rng.integers(0, 172_793, size=n)).astype(float)
    times = times[(times <= T_FIT_END_TIME) | (times >= T_CAL_START_TIME)]  # no row at 112,554
    X = rng.normal(size=(len(times), len(FEATURES)))
    logit = -4.0 + 2.5 * X[:, 0]
    y = (rng.random(len(times)) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame(X, columns=FEATURES)
    df.insert(0, "Time", times)
    df["Class"] = y
    return df


def assert_nested_close(case, a, b, tol=1e-12, path=""):
    """Exact for everything except floats, which must agree to ``tol``."""
    if isinstance(a, dict):
        case.assertEqual(set(a), set(b), path)
        for key in a:
            assert_nested_close(case, a[key], b[key], tol, f"{path}/{key}")
    elif isinstance(a, list):
        case.assertEqual(len(a), len(b), path)
        for i, (x, y) in enumerate(zip(a, b)):
            assert_nested_close(case, x, y, tol, f"{path}[{i}]")
    elif isinstance(a, float) and not isinstance(b, bool):
        case.assertAlmostEqual(a, b, delta=tol, msg=path)
    else:
        case.assertEqual(a, b, path)


def fake_phase4b(path: Path, n_rows: int, n_positive: int) -> None:
    path.write_text(json.dumps({
        "models": {"logistic_regression_balanced": {"C": 1.0, "class_weight": "balanced"}},
        "experiments": {"A_chronological": {"results": {"logistic_regression_balanced": {"metrics": {
            "n_rows": n_rows, "n_positive": n_positive, "average_precision": 0.5, "roc_auc": 0.9}}}}},
    }))


# ---------------------------------------------------------------------------
# 5B-1B: T_fit / T_cal
# ---------------------------------------------------------------------------

class TestTrainSubpartitionAssignment(unittest.TestCase):
    def test_boundaries_are_inclusive_as_frozen(self):
        labels = assign_train_subpartitions(pd.Series([0, T_FIT_END_TIME, T_CAL_START_TIME, TRAIN_END_TIME]))
        self.assertEqual(list(labels.astype(str)), [T_FIT, T_FIT, T_CAL, T_CAL])
        self.assertEqual((T_FIT_END_TIME, T_CAL_START_TIME), (112_553, 112_555))

    def test_gap_timestamp_rejected(self):
        with self.assertRaises(ValueError):
            assign_train_subpartitions(pd.Series([100, 112_554]))

    def test_validation_and_test_rows_refused(self):
        for t in (TRAIN_END_TIME + 1, VALIDATION_END_TIME, VALIDATION_END_TIME + 1):
            with self.assertRaises(ValueError):
                assign_train_subpartitions(pd.Series([0, t]))

    def test_invalid_timestamps_rejected(self):
        for bad in ([0.5], [-1], [np.nan], [np.inf]):
            with self.assertRaises(ValueError):
                assign_train_subpartitions(pd.Series(bad, dtype=float))
        with self.assertRaises(TypeError):
            assign_train_subpartitions(pd.Series(["a"]))
        with self.assertRaises(ValueError):
            assign_train_subpartitions(np.zeros((2, 2)))

    def test_order_index_and_equal_timestamps_preserved(self):
        times = pd.Series([120_000, 5, 5, 112_553, 120_000], index=[10, 3, 7, 1, 2])
        labels = assign_train_subpartitions(times)
        self.assertTrue(labels.index.equals(times.index))
        self.assertEqual(list(labels.astype(str)), [T_CAL, T_FIT, T_FIT, T_FIT, T_CAL])

    def test_deterministic_and_label_free(self):
        times = pd.Series(np.arange(0, TRAIN_END_TIME, 997))
        a, b = assign_train_subpartitions(times), assign_train_subpartitions(times.copy())
        self.assertTrue(a.equals(b))
        # Accepts a bare array: no label is needed to assign.
        np.testing.assert_array_equal(assign_train_subpartitions(times.to_numpy()).to_numpy(), a.to_numpy())


class TestTrainSubpartitionChecks(unittest.TestCase):
    def setUp(self):
        self.times = pd.Series([0, 50, 112_553, 112_555, 130_000, 132_928])
        self.sub = assign_train_subpartitions(self.times)

    def test_integrity_passes_on_frozen_rule(self):
        record = check_temporal_integrity(self.times, self.sub)
        self.assertEqual(record["t_fit_time_range"], [0, 112_553])
        self.assertEqual(record["t_cal_time_range"], [112_555, 132_928])
        self.assertEqual(record["gap_seconds_between_last_t_fit_and_first_t_cal"], 2)

    def test_integrity_detects_order_violation_and_shared_timestamp(self):
        bad = pd.Series(pd.Categorical([T_FIT, T_CAL, T_FIT, T_CAL, T_CAL, T_CAL], categories=[T_FIT, T_CAL]))
        with self.assertRaises(ValueError):
            check_temporal_integrity(self.times, bad)
        times = pd.Series([0, 0, 120_000])
        shared = pd.Series(pd.Categorical([T_FIT, T_CAL, T_CAL], categories=[T_FIT, T_CAL]))
        with self.assertRaises(ValueError):
            check_temporal_integrity(times, shared)

    def test_integrity_requires_both_non_empty(self):
        times = pd.Series([0, 10])
        with self.assertRaises(ValueError):
            check_temporal_integrity(times, assign_train_subpartitions(times))

    def test_counts_and_estimability(self):
        y = [0, 1, 0, 1, 0, 0]
        summary = summarize_subpartitions(self.sub, y)
        self.assertEqual(summary[T_FIT], {"n_rows": 3, "n_positive": 1, "n_negative": 2,
                                          "prevalence": 1 / 3, "estimable": True})
        self.assertTrue(summary["all_estimable"])
        require_estimable(summary)

    def test_partition_without_positive_or_negative_is_not_estimable(self):
        for y in ([0, 1, 0, 0, 0, 0], [0, 1, 0, 1, 1, 1]):
            summary = summarize_subpartitions(self.sub, y)
            self.assertFalse(summary[T_CAL]["estimable"])
            with self.assertRaises(ValueError):
                require_estimable(summary)


# ---------------------------------------------------------------------------
# 5B-1A: E2 usability
# ---------------------------------------------------------------------------

class TestUsabilityRules(unittest.TestCase):
    def test_ten_frauds_is_usable_nine_is_not(self):
        d = usability_decision([9, 10, 11])
        self.assertEqual((d["n_usable"], d["n_skipped"]), (2, 1))

    def test_minimum_usable_splits_boundary(self):
        below = usability_decision([10] * 999 + [0] * 1001)
        at = usability_decision([10] * 1000 + [0] * 1000)
        self.assertEqual((below["n_usable"], below["e2_estimable"]), (999, False))
        self.assertIn("not estimable", below["consequence"])
        self.assertEqual((at["n_usable"], at["e2_estimable"]), (1000, True))


class TestSplitCounts(unittest.TestCase):
    """Hand-computed case: blocks [0,900) [900,1800) [1800,2700); selection = block 1."""

    def test_buffer_applied_before_counting(self):
        edges = np.array([0, 900, 1800, 2700])
        times = np.array([10, 599, 600, 899, 900, 1799, 1800, 2099, 2100])
        y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        c = split_counts(times, y, edges, [[1]])
        # Held out and kept: 10, 599 (before a-300 = 600) and 2100 (>= b+300). Dropped: 600, 899, 1800, 2099.
        self.assertEqual(int(c["heldout_frauds"][0]), 3)
        self.assertEqual(int(c["selection_frauds"][0]), 2)
        self.assertEqual(int(c["buffer_dropped_frauds"][0]), 4)

    def test_counts_only_frauds_and_conserve_total(self):
        rng = np.random.default_rng(1)
        edges = block_edges(0, 18_899)  # 21 blocks of 900 s
        times = np.sort(rng.integers(0, 18_900, 3000))
        y = (rng.random(3000) < 0.02).astype(int)
        draws = draw_selection_blocks(50, 21, 10, 0)
        c = split_counts(times, y, edges, draws)
        np.testing.assert_array_equal(c["selection_frauds"] + c["heldout_frauds"] + c["buffer_dropped_frauds"],
                                      np.full(50, y.sum()))
        np.testing.assert_array_equal(c["selection_rows"] + c["heldout_rows"] + c["buffer_dropped_rows"],
                                      np.full(50, 3000))

    def test_rejects_rows_outside_edges(self):
        with self.assertRaises(ValueError):
            split_counts([0, 2700], [0, 1], np.array([0, 900, 1800, 2700]), [[1]])


class TestE2UsabilityOnValidationPeriod(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        self.times = np.sort(rng.integers(VALIDATION_START_TIME, VALIDATION_END_TIME + 1, 5000))
        self.y = (rng.random(5000) < 0.02).astype(int)

    def test_uses_frozen_draws_and_is_deterministic(self):
        a, b = e2_usability(self.times, self.y), e2_usability(self.times, self.y)
        self.assertEqual(a, b)
        self.assertEqual(a["diagnostics"]["first_three_splits"],
                         draw_selection_blocks(3, 21, 10, E2_SPLIT_SEED).tolist())
        self.assertEqual(a["decision"]["n_attempted"], 2000)
        self.assertEqual(a["settings"]["block_edges"][-1], VALIDATION_END_TIME + 1)

    def test_test_period_timestamp_refused(self):
        with self.assertRaises(ValueError):
            e2_usability(np.append(self.times, VALIDATION_END_TIME + 1), np.append(self.y, 0))

    def test_train_period_timestamp_refused(self):
        with self.assertRaises(ValueError):
            e2_usability(np.append(self.times, TRAIN_END_TIME), np.append(self.y, 0))


# ---------------------------------------------------------------------------
# 5B-1C: three-track continuity
# ---------------------------------------------------------------------------

class TestScoreTrackValidation(unittest.TestCase):
    def tracks(self):
        base = {"role": "r", "fit_rows": "f", "source": "s", "metrics": {}}
        return {pc.PHASE4_HISTORICAL: {**base, "score_representation": "predict_proba"},
                pc.PHASE5_PRIMARY: {**base, "score_representation": "decision_function"},
                pc.PHASE5_DIAGNOSTIC: {**base, "score_representation": "predict_proba"}}

    def test_valid_tracks_pass(self):
        pc.validate_score_tracks(self.tracks())

    def test_missing_score_representation_rejected(self):
        tracks = self.tracks()
        del tracks[pc.PHASE5_PRIMARY]["score_representation"]
        with self.assertRaises(ValueError):
            pc.validate_score_tracks(tracks)

    def test_wrong_or_swapped_representation_rejected(self):
        for name, rep in ((pc.PHASE5_PRIMARY, "predict_proba"), (pc.PHASE4_HISTORICAL, "decision_function"),
                          (pc.PHASE5_DIAGNOSTIC, "decision_function"), (pc.PHASE5_PRIMARY, "rank")):
            tracks = self.tracks()
            tracks[name]["score_representation"] = rep
            with self.assertRaises(ValueError):
                pc.validate_score_tracks(tracks)

    def test_missing_track_rejected(self):
        tracks = self.tracks()
        del tracks[pc.PHASE5_DIAGNOSTIC]
        with self.assertRaises(ValueError):
            pc.validate_score_tracks(tracks)


class TestHistoricalTrackIsQuotedReadOnly(unittest.TestCase):
    def test_quote_matches_phase4b_file_and_leaves_it_unchanged(self):
        before = hashlib.sha256(PHASE4B_FILE.read_bytes()).hexdigest()
        track = pc.quote_phase4_historical(PHASE4B_FILE)
        self.assertEqual(hashlib.sha256(PHASE4B_FILE.read_bytes()).hexdigest(), before)
        self.assertEqual(track["score_representation"], "predict_proba")
        self.assertEqual(track["metrics"]["average_precision"], 0.8396604620788174)
        self.assertEqual(track["source"], "reports/phase4b_baseline_results.json")


class TestPhase5TracksSynthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = synthetic_frame()
        cls.config = DatasetConfig(name="synthetic", path=Path("unused.csv"), target_column="Class",
                                   excluded_feature_columns=("Time",), time_column="Time")
        cls.fit_rows = df.loc[df["Time"] <= T_FIT_END_TIME]
        cls.validation = df.loc[(df["Time"] > TRAIN_END_TIME) & (df["Time"] <= VALIDATION_END_TIME)]
        cls.result = pc.fit_phase5_tracks(cls.fit_rows, cls.validation, cls.config)

    def test_fit_uses_t_fit_rows_only(self):
        self.assertEqual(self.result["fit"]["n_fit_rows"], len(self.fit_rows))
        self.assertEqual(self.result["fit"]["params"]["C"], 1.0)
        self.assertEqual(self.result["fit"]["params"]["class_weight"], "balanced")

    def test_primary_is_decision_function_and_diagnostic_is_same_fit_proba(self):
        primary, diagnostic = self.result[pc.PHASE5_PRIMARY], self.result[pc.PHASE5_DIAGNOSTIC]
        self.assertEqual(primary["score_representation"], "decision_function")
        self.assertEqual(diagnostic["score_representation"], "predict_proba")
        self.assertEqual(self.result["fit"]["proba_equals_expit_decision_max_abs_diff"], 0.0)
        self.assertGreater(primary["score_diagnostics"]["max"], 1.0)  # a logit, not a probability
        self.assertLessEqual(diagnostic["score_diagnostics"]["max"], 1.0)

    def test_primary_metric_recomputes(self):
        builder = pc.build_continuity_lr(pc.MAX_ITER)
        pipe = builder(FEATURES, 42).fit(self.fit_rows[FEATURES], self.fit_rows["Class"])
        ap = average_precision_score(self.validation["Class"], pipe.decision_function(self.validation[FEATURES]))
        self.assertEqual(self.result[pc.PHASE5_PRIMARY]["metrics"]["average_precision"], ap)

    def test_overlapping_rows_rejected(self):
        with self.assertRaises(ValueError):
            pc.fit_phase5_tracks(self.fit_rows, self.fit_rows.iloc[:10], self.config)


# ---------------------------------------------------------------------------
# Runner: test-set isolation
# ---------------------------------------------------------------------------

class TestRunnerTestIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = DatasetConfig(name="synthetic", path=Path("unused.csv"), target_column="Class",
                                   excluded_feature_columns=("Time",), time_column="Time")
        cls.df = synthetic_frame()
        val = cls.df.loc[(cls.df["Time"] > TRAIN_END_TIME) & (cls.df["Time"] <= VALIDATION_END_TIME)]
        cls.tmp = tempfile.TemporaryDirectory()
        cls.phase4b = Path(cls.tmp.name) / "reports" / "phase4b.json"
        cls.phase4b.parent.mkdir()
        fake_phase4b(cls.phase4b, len(val), int(val["Class"].sum()))
        cls.baseline = compute(cls.df, cls.config, cls.phase4b)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_drop_test_rows_removes_every_test_period_row(self):
        dev, parts, removed = drop_test_rows(self.df, self.config)
        self.assertEqual(removed, int((self.df["Time"] > VALIDATION_END_TIME).sum()))
        self.assertLessEqual(dev["Time"].max(), VALIDATION_END_TIME)
        self.assertNotIn("test", set(parts.astype(str)))

    def test_results_invariant_to_test_labels_and_features(self):
        tampered = self.df.copy()
        test = tampered["Time"] > VALIDATION_END_TIME
        tampered.loc[test, "Class"] = 1 - tampered.loc[test, "Class"]
        tampered.loc[test, FEATURES] = 1e6
        result = compute(tampered, self.config, self.phase4b)
        # Counts, partitions and E2 are exact. Fitted floats are compared to 1e-12:
        # copying a frame changes its memory layout, which changes BLAS summation
        # order at the last bit (~1e-15) even for identical values.
        for key in ("test_isolation", "t_fit_t_cal_partition", "e2_usability"):
            self.assertEqual(result[key], self.baseline[key])
        assert_nested_close(self, result["phase4_continuity"], self.baseline["phase4_continuity"])

    def test_results_change_when_validation_labels_change(self):
        tampered = self.df.copy()
        val = (tampered["Time"] > TRAIN_END_TIME) & (tampered["Time"] <= VALIDATION_END_TIME)
        tampered.loc[val, "Class"] = 1 - tampered.loc[val, "Class"]
        fake = Path(self.tmp.name) / "reports" / "flipped.json"
        fake_phase4b(fake, int(val.sum()), int(tampered.loc[val, "Class"].sum()))
        self.assertNotEqual(compute(tampered, self.config, fake)["e2_usability"], self.baseline["e2_usability"])

    def test_record_states_isolation(self):
        record = self.baseline["test_isolation"]
        self.assertEqual(record["partitions_processed"], ["train", "validation"])
        self.assertLessEqual(record["max_time_in_processed_rows"], VALIDATION_END_TIME)


# ---------------------------------------------------------------------------
# Real data (train and validation rows only)
# ---------------------------------------------------------------------------

@unittest.skipUnless(DATA_AVAILABLE, "data/raw/creditcard.csv not present")
class TestPhase5B1OnRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dev, parts, _ = drop_test_rows(pd.read_csv(CREDITCARD.path), CREDITCARD)
        cls.train = dev.loc[(parts == "train").to_numpy()]
        cls.validation = dev.loc[(parts == "validation").to_numpy()]

    def test_subpartition_counts_match_protocol(self):
        sub = assign_train_subpartitions(self.train["Time"])
        summary = summarize_subpartitions(sub, self.train["Class"])
        self.assertEqual((summary[T_FIT]["n_rows"], summary[T_FIT]["n_positive"]), (159_491, 356))
        self.assertEqual((summary[T_CAL]["n_rows"], summary[T_CAL]["n_positive"]), (39_873, 28))
        check_temporal_integrity(self.train["Time"], sub)

    @unittest.skipUnless(RESULTS_FILE.is_file(), "reports/phase5b1_results.json not generated")
    def test_e2_usability_reproduces_recorded_result(self):
        recorded = json.loads(RESULTS_FILE.read_text())["e2_usability"]
        current = e2_usability(self.validation["Time"].to_numpy(), self.validation["Class"].to_numpy())
        self.assertEqual(json.loads(json.dumps(current)), recorded)


if __name__ == "__main__":
    unittest.main()
