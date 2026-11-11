"""Tests for Phase 5B-2: experiments E0-E3, fixed-K metrics, score tracks, stability.

Synthetic tests use generated data only. Real-data tests are skipped when
``data/raw/creditcard.csv`` is absent; they process train and validation rows
only (test rows are removed before any label is accessed).
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from src.config import CREDITCARD, PROJECT_ROOT
from src.data.split import TRAIN_END_TIME, VALIDATION_END_TIME
from src.evaluation.capacity import fixed_k_metrics, fixed_k_table
from src.evaluation.e2_usability import e2_usability
from src.features.amount_transform import LOG1P_AMOUNT_COLUMN, add_log1p_amount, log1p_amount
from src.models import phase5b2_experiments as px
from src.models.baselines import LOGISTIC_REGRESSION_PARAMS
from src.models.phase4_continuity import fit_phase5_tracks
from src.models.phase5_settings import (
    T_CAL_START_TIME, T_FIT_END_TIME, VALIDATION_START_TIME, pre_registered_configuration,
)
from src.models.run_phase5b2 import RUNTIME_KEY, compute

DATA_AVAILABLE = CREDITCARD.path.is_file()
RESULTS_FILE = PROJECT_ROOT / "reports" / "phase5b2_results.json"
PHASE4B_FILE = PROJECT_ROOT / "reports" / "phase4b_baseline_results.json"
PHASE5B1_FILE = PROJECT_ROOT / "reports" / "phase5b1_results.json"
BASE = [f"V{i}" for i in range(1, 29)] + ["Amount"]


def synthetic_frame(n=8000, seed=0) -> pd.DataFrame:
    """Rows over the full 0-172,792 s range; non-negative Amount; learnable label."""
    rng = np.random.default_rng(seed)
    times = np.sort(rng.integers(0, 172_793, size=n)).astype(float)
    times = times[(times <= T_FIT_END_TIME) | (times >= T_CAL_START_TIME)]
    X = rng.normal(size=(len(times), 28))
    amount = np.round(np.exp(rng.normal(3.0, 1.5, size=len(times))), 2)
    amount[:50] = 0.0
    logit = -4.0 + 2.5 * X[:, 0] + 0.4 * np.log1p(amount) - 0.8
    y = (rng.random(len(times)) < expit(logit)).astype(int)
    df = pd.DataFrame(X, columns=BASE[:-1])
    df["Amount"] = amount
    df.insert(0, "Time", times)
    df["Class"] = y
    return df


def fake_phase4b(path: Path, n_rows: int, n_positive: int) -> None:
    path.write_text(json.dumps({
        "models": {"logistic_regression_balanced": {"C": 1.0, "class_weight": "balanced"}},
        "experiments": {"A_chronological": {"results": {"logistic_regression_balanced": {"metrics": {
            "n_rows": n_rows, "n_positive": n_positive, "average_precision": 0.5, "roc_auc": 0.9}}}}},
    }))


def split_frames(df):
    fit_rows = df[df.Time <= T_FIT_END_TIME]
    validation = df[(df.Time >= VALIDATION_START_TIME) & (df.Time <= VALIDATION_END_TIME)]
    return fit_rows, validation


def strip_runtime(result: dict) -> dict:
    return {k: v for k, v in result.items() if k not in (RUNTIME_KEY, "hgb_determinism")}


def assert_nested_close(case, a, b, tol=1e-12, path=""):
    if isinstance(a, dict):
        case.assertEqual(set(a), set(b), path)
        for key in a:
            assert_nested_close(case, a[key], b[key], tol, f"{path}/{key}")
    elif isinstance(a, (list, tuple)):
        case.assertEqual(len(a), len(b), path)
        for i, (x, y) in enumerate(zip(a, b)):
            assert_nested_close(case, x, y, tol, f"{path}[{i}]")
    elif isinstance(a, float) and not isinstance(b, bool):
        if math.isnan(a):
            case.assertTrue(b is None or math.isnan(b), path)
        else:
            case.assertAlmostEqual(a, b, delta=tol, msg=path)
    else:
        case.assertEqual(a, b, path)


# ---------------------------------------------------------------------------
# Feature transformation
# ---------------------------------------------------------------------------

class TestLog1pAmount(unittest.TestCase):
    def test_values(self):
        np.testing.assert_array_equal(log1p_amount([0.0, math.e - 1]), [0.0, 1.0])
        self.assertEqual(log1p_amount([0.0])[0], 0.0)  # zero amounts map exactly to 0

    def test_appends_column_and_preserves_input(self):
        df = pd.DataFrame({"Time": [3.0, 1.0], "Amount": [0.0, 9.0], "Class": [0, 1]}, index=[10, 4])
        before = df.copy()
        out = add_log1p_amount(df)
        pd.testing.assert_frame_equal(df, before)  # input not modified
        self.assertEqual(list(out.columns), ["Time", "Amount", "Class", LOG1P_AMOUNT_COLUMN])
        self.assertEqual(list(out.index), [10, 4])
        pd.testing.assert_series_equal(out["Amount"], df["Amount"])
        np.testing.assert_allclose(out[LOG1P_AMOUNT_COLUMN], np.log1p([0.0, 9.0]), rtol=0, atol=0)

    def test_row_wise_and_stateless(self):
        """Each row's value depends on that row only (no fitted statistic)."""
        df = pd.DataFrame({"Amount": [1.0, 2.0, 3.0]})
        full = add_log1p_amount(df)[LOG1P_AMOUNT_COLUMN].to_numpy()
        parts = np.concatenate([add_log1p_amount(df.iloc[[i]])[LOG1P_AMOUNT_COLUMN].to_numpy() for i in range(3)])
        np.testing.assert_array_equal(full, parts)

    def test_refusals(self):
        for bad in ([-0.01], [np.nan], [np.inf]):
            with self.assertRaises(ValueError):
                log1p_amount(bad)
        with self.assertRaises(KeyError):
            add_log1p_amount(pd.DataFrame({"x": [1.0]}))
        with self.assertRaises(ValueError):
            add_log1p_amount(pd.DataFrame({"Amount": [1.0], LOG1P_AMOUNT_COLUMN: [0.0]}))
        with self.assertRaises(TypeError):
            add_log1p_amount(pd.DataFrame({"Amount": ["1"]}))


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

class TestExperimentDefinitions(unittest.TestCase):
    def test_four_experiments(self):
        self.assertEqual(list(px.EXPERIMENTS), ["E0", "E1", "E2", "E3"])
        fams = {e: d["family"] for e, d in px.EXPERIMENTS.items()}
        self.assertEqual(fams, {"E0": px.LOGISTIC_REGRESSION, "E1": px.LOGISTIC_REGRESSION,
                                "E2": px.HIST_GRADIENT_BOOSTING, "E3": px.HIST_GRADIENT_BOOSTING})

    def test_feature_sets(self):
        frozen = pre_registered_configuration()["features"]["model_features"]
        self.assertEqual(list(px.experiment_features("E0")), frozen)
        self.assertEqual(px.experiment_features("E0"), px.experiment_features("E2"))
        self.assertEqual(px.experiment_features("E1"), px.experiment_features("E3"))
        # the pairs differ only by the appended log1p(Amount) column
        self.assertEqual(list(px.experiment_features("E1")), frozen + [LOG1P_AMOUNT_COLUMN])
        for e in px.EXPERIMENTS:
            self.assertNotIn("Time", px.experiment_features(e))
            self.assertNotIn("Class", px.experiment_features(e))

    def test_lr_is_phase4_configuration(self):
        from src.models.phase4_continuity import CONTINUITY_LR_PARAMS
        for key, value in CONTINUITY_LR_PARAMS.items():
            self.assertEqual(LOGISTIC_REGRESSION_PARAMS[key], value)

    def test_hgb_is_a_frozen_grid_point(self):
        hgb = pre_registered_configuration()["models"]["hist_gradient_boosting"]
        for key, values in hgb["grid"].items():
            self.assertIn(px.HGB_PARAMS[key], values, key)
        for key, value in hgb["fixed"].items():
            if key in px.HGB_PARAMS:
                self.assertEqual(px.HGB_PARAMS[key], value, key)
        self.assertIs(px.HGB_PARAMS["early_stopping"], False)  # no hidden random split
        model = px.build_hgb(BASE, 42).named_steps["model"]
        self.assertEqual(model.random_state, 42)
        self.assertEqual(model.max_iter, 300)

    def test_contrasts_isolate_one_factor(self):
        for a, b, _ in px.CONTRASTS.values():
            same_family = px.EXPERIMENTS[a]["family"] == px.EXPERIMENTS[b]["family"]
            same_features = px.EXPERIMENTS[a]["feature_set"] == px.EXPERIMENTS[b]["feature_set"]
            self.assertTrue(same_family != same_features)


# ---------------------------------------------------------------------------
# Training/validation separation and fitting (synthetic)
# ---------------------------------------------------------------------------

class TestFitExperiment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = synthetic_frame()
        cls.fit_rows, cls.validation = split_frames(cls.df)
        cls.fitted = {e: px.fit_experiment(e, cls.fit_rows, cls.validation, CREDITCARD) for e in px.EXPERIMENTS}

    def test_refuses_rows_outside_t_fit_or_validation(self):
        t_cal = self.df[(self.df.Time >= T_CAL_START_TIME) & (self.df.Time <= TRAIN_END_TIME)]
        test = self.df[self.df.Time > VALIDATION_END_TIME]
        with self.assertRaises(ValueError):
            px.prepare_frames("E0", pd.concat([self.fit_rows, t_cal]), self.validation, "Time")
        with self.assertRaises(ValueError):
            px.prepare_frames("E0", self.fit_rows, pd.concat([self.validation, test]), "Time")
        with self.assertRaises(ValueError):
            px.prepare_frames("E0", self.fit_rows, pd.concat([t_cal, self.validation]), "Time")

    def test_inputs_not_modified(self):
        before = self.fit_rows.copy()
        px.prepare_frames("E1", self.fit_rows, self.validation, "Time")
        pd.testing.assert_frame_equal(self.fit_rows, before)
        self.assertNotIn(LOG1P_AMOUNT_COLUMN, self.fit_rows.columns)

    def test_preprocessing_fitted_on_t_fit_only(self):
        for e in ("E0", "E1"):
            scaler = self.fitted[e]["pipeline"].named_steps["preprocess"].named_transformers_["standardize"]
            self.assertEqual(scaler.n_samples_seen_, len(self.fit_rows))
            self.assertEqual(list(scaler.feature_names_in_), list(px.experiment_features(e)))
            mean_amount = scaler.mean_[list(scaler.feature_names_in_).index("Amount")]
            self.assertAlmostEqual(mean_amount, self.fit_rows["Amount"].mean(), places=8)

    def test_n_features(self):
        for e in px.EXPERIMENTS:
            model = self.fitted[e]["pipeline"].named_steps["model"]
            self.assertEqual(model.n_features_in_, len(px.experiment_features(e)))

    def test_same_validation_rows_for_all(self):
        for e in px.EXPERIMENTS:
            np.testing.assert_array_equal(self.fitted[e]["validation_index"], self.validation.index.to_numpy())

    def test_score_representations(self):
        for e, f in self.fitted.items():
            X = px.prepare_frames(e, self.fit_rows, self.validation, "Time")[1][list(f["features"])]
            np.testing.assert_array_equal(f["decision_function"], f["pipeline"].decision_function(X))
            np.testing.assert_array_equal(f["predict_proba"], f["pipeline"].predict_proba(X)[:, 1])
            np.testing.assert_allclose(f["predict_proba"], expit(f["decision_function"]), rtol=0, atol=1e-15)

    def test_e0_matches_phase5b1_primary_track(self):
        """Phase 5B-1 primary-track behavior is preserved exactly by E0."""
        ref = fit_phase5_tracks(self.fit_rows, self.validation, CREDITCARD)
        e0 = px.ranking_metrics(self.validation["Class"], self.fitted["E0"]["decision_function"])
        self.assertEqual(e0["average_precision"], ref["phase5_primary"]["metrics"]["average_precision"])

    def test_hgb_deterministic(self):
        again = px.fit_experiment("E2", self.fit_rows, self.validation, CREDITCARD)
        np.testing.assert_array_equal(again["decision_function"], self.fitted["E2"]["decision_function"])

    def test_hgb_invariant_to_monotone_log1p_duplicate(self):
        """log1p(Amount) bins the training rows exactly like Amount, so gain ties keep the earlier
        Amount column: E3 scores equal E2 scores. The log1p column is never used in a split."""
        np.testing.assert_array_equal(self.fitted["E3"]["decision_function"], self.fitted["E2"]["decision_function"])
        model = self.fitted["E3"]["pipeline"].named_steps["model"]
        log_idx = list(px.experiment_features("E3")).index(LOG1P_AMOUNT_COLUMN)
        used = {int(i) for trees in model._predictors for tree in trees  # private attribute; test only
                for i in tree.nodes["feature_idx"][tree.nodes["is_leaf"] == 0]}
        self.assertNotIn(log_idx, used)

    def test_lr_log1p_changes_scores(self):
        self.assertFalse(np.array_equal(self.fitted["E1"]["decision_function"], self.fitted["E0"]["decision_function"]))


# ---------------------------------------------------------------------------
# Fixed-K metrics
# ---------------------------------------------------------------------------

class TestFixedK(unittest.TestCase):
    def setUp(self):
        self.y = np.array([1, 0, 1, 0, 0, 1])
        self.s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        self.t = np.arange(6.0)
        self.pos = np.arange(6)

    def test_hand_computed(self):
        m = fixed_k_metrics(self.y, self.s, self.t, self.pos, 3)
        self.assertEqual((m["tp"], m["fp"]), (2, 1))
        self.assertAlmostEqual(m["precision_at_k"], 2 / 3)
        self.assertAlmostEqual(m["recall_at_k"], 2 / 3)
        self.assertEqual(m["oracle_recall_ceiling"], 1.0)
        self.assertFalse(m["ties"]["tie_sensitive"])
        full = fixed_k_metrics(self.y, self.s, self.t, self.pos, 6)
        self.assertEqual((full["tp"], full["recall_at_k"], full["precision_at_k"]), (3, 1.0, 0.5))
        one = fixed_k_metrics(self.y, self.s, self.t, self.pos, 1)
        self.assertEqual(one["oracle_recall_ceiling"], 1 / 3)

    def test_ties_use_time_then_position_and_report_range(self):
        y = np.array([0, 1, 0, 1])
        s = np.array([0.5, 0.5, 0.5, 0.9])
        t = np.array([2.0, 5.0, 2.0, 0.0])
        pos = np.array([2, 0, 1, 3])  # rows 0 and 2 share Time 2; position decides
        m = fixed_k_metrics(y, s, t, pos, 2)
        # order: row 3 (0.9), then tied rows by Time: rows 0,2 (Time 2) by position -> row 2 first
        self.assertEqual(m["tp"], 1)
        ties = m["ties"]
        self.assertEqual((ties["n_tied_at_boundary"], ties["n_tied_inside_top_k"], ties["frauds_in_tied_group"]),
                         (3, 1, 1))
        self.assertEqual((ties["tp_min"], ties["tp_max"]), (1, 2))
        self.assertAlmostEqual(ties["expected_tp_random_tie_order"], 1 + 1 / 3)
        self.assertTrue(ties["tie_sensitive"])

    def test_order_ignores_labels(self):
        rng = np.random.default_rng(1)
        s = rng.integers(0, 5, size=200).astype(float)
        t, pos = np.arange(200.0), np.arange(200)
        a = fixed_k_metrics(rng.integers(0, 2, 200), s, t, pos, 37)["ties"]
        b = fixed_k_metrics(rng.integers(0, 2, 200), s, t, pos, 37)["ties"]
        for key in ("n_tied_at_boundary", "n_tied_inside_top_k", "n_tied_outside_top_k"):
            self.assertEqual(a[key], b[key])

    def test_edge_cases(self):
        for bad in (0, -1, 2.5, True, 7):
            with self.assertRaises(ValueError):
                fixed_k_metrics(self.y, self.s, self.t, self.pos, bad)
        none = fixed_k_metrics(np.zeros(6), self.s, self.t, self.pos, 2)
        self.assertTrue(math.isnan(none["recall_at_k"]))
        self.assertEqual(none["precision_at_k"], 0.0)
        with self.assertRaises(ValueError):
            fixed_k_metrics(self.y[:5], self.s, self.t, self.pos, 2)

    def test_table(self):
        rows = fixed_k_table(self.y, self.s, self.t, self.pos, [1, 3, 6])
        self.assertEqual([r["k"] for r in rows], [1, 3, 6])
        self.assertEqual([r["tp"] for r in rows], [1, 2, 3])


# ---------------------------------------------------------------------------
# Score tracks
# ---------------------------------------------------------------------------

class TestTrackValidation(unittest.TestCase):
    def tracks(self):
        def block(track):
            return {"score_representation": px.TRACK_REPRESENTATION[track], "role": "r", "fit_rows": "f"}
        out = {e: {"tracks": {px.TRACK_A: {"status": px.NOT_APPLICABLE}, px.TRACK_B: block(px.TRACK_B),
                              px.TRACK_C: block(px.TRACK_C)}} for e in px.EXPERIMENTS}
        out["E0"]["tracks"][px.TRACK_A] = block(px.TRACK_A)
        return out

    def test_valid(self):
        px.validate_experiment_tracks(self.tracks())

    def test_wrong_or_missing_representation(self):
        t = self.tracks()
        t["E2"]["tracks"][px.TRACK_B]["score_representation"] = "predict_proba"
        with self.assertRaises(ValueError):
            px.validate_experiment_tracks(t)
        t = self.tracks()
        del t["E1"]["tracks"][px.TRACK_C]["score_representation"]
        with self.assertRaises(ValueError):
            px.validate_experiment_tracks(t)
        t = self.tracks()
        t["E3"]["tracks"][px.TRACK_C]["score_representation"] = "decision_function"
        with self.assertRaises(ValueError):
            px.validate_experiment_tracks(t)

    def test_not_applicable_rules(self):
        t = self.tracks()
        t["E0"]["tracks"][px.TRACK_A] = {"status": px.NOT_APPLICABLE}
        with self.assertRaises(ValueError):
            px.validate_experiment_tracks(t)
        t = self.tracks()
        t["E1"]["tracks"][px.TRACK_B] = {"status": px.NOT_APPLICABLE}
        with self.assertRaises(ValueError):
            px.validate_experiment_tracks(t)
        t = self.tracks()
        del t["E3"]
        with self.assertRaises(ValueError):
            px.validate_experiment_tracks(t)


# ---------------------------------------------------------------------------
# Metrics and temporal stability
# ---------------------------------------------------------------------------

class TestEvaluation(unittest.TestCase):
    def test_ranking_metrics_undefined_cases(self):
        m = px.ranking_metrics([0, 0, 0], [0.1, 0.2, 0.3])
        self.assertTrue(math.isnan(m["average_precision"]) and math.isnan(m["roc_auc"]))
        m = px.ranking_metrics([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
        self.assertEqual((m["average_precision"], m["roc_auc"]), (1.0, 1.0))

    def test_default_rule_diagnostic(self):
        d = px.default_rule_diagnostic([1, 0, 1, 0], [1, 1, 0, 0])
        self.assertEqual((d["n_alerts"], d["tp"], d["fp"]), (2, 1, 1))
        self.assertEqual((d["precision"], d["recall"]), (0.5, 0.5))
        self.assertTrue(math.isnan(px.default_rule_diagnostic([1, 0], [0, 0])["precision"]))

    def test_evaluate_track_windows_partition_rows(self):
        rng = np.random.default_rng(3)
        n = 400
        y = (rng.random(n) < 0.1).astype(int)
        s = rng.normal(size=n) + 2 * y
        labels = np.array(["window_1"] * 200 + ["window_2"] * 200)
        out = px.evaluate_track(s, y, np.arange(n, dtype=float), np.arange(n), labels, [10, 50], [5, 25])
        w = out["windows"]
        self.assertEqual(w["window_1"]["n_rows"] + w["window_2"]["n_rows"], n)
        self.assertEqual(w["window_1"]["n_positive"] + w["window_2"]["n_positive"], int(y.sum()))
        self.assertEqual([r["k"] for r in w["window_1"]["fixed_k"]], [5, 25])

    def test_heldout_halves(self):
        df = synthetic_frame(seed=4)
        _, validation = split_frames(df)
        rng = np.random.default_rng(0)
        s = rng.normal(size=len(validation)) + 2 * validation["Class"].to_numpy()
        out = px.heldout_half_stability(validation.Time.to_numpy(), validation["Class"].to_numpy(),
                                        {"E0": s, "E1": s.copy(), "E2": -s})
        ref = e2_usability(validation.Time.to_numpy(), validation["Class"].to_numpy())
        self.assertEqual(out["n_usable"], ref["decision"]["n_usable"])  # same frozen splits and rule
        self.assertEqual(out["n_splits_attempted"], 2000)
        if out["n_usable"]:
            self.assertEqual(out["contrasts"]["E1_minus_E0"]["delta_ap"]["max"], 0.0)
            self.assertEqual(out["contrasts"]["E2_minus_E0"]["share_positive"], 0.0)
            self.assertAlmostEqual(sum(out["share_of_splits_where_best"].values()), 1.0)
        self.assertNotIn("E3_minus_E2", out["contrasts"])  # contrasts with missing experiments skipped
        self.assertIn("NOT the protocol criterion E2", out["scope"])

    def test_heldout_halves_refuse_out_of_period(self):
        with self.assertRaises(ValueError):
            px.heldout_half_stability(np.array([1.0, VALIDATION_START_TIME]), np.array([0, 1]),
                                      {"E0": np.array([0.1, 0.2])})


# ---------------------------------------------------------------------------
# End-to-end (synthetic): isolation, reproducibility, historical track
# ---------------------------------------------------------------------------

class TestComputeSynthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.df = synthetic_frame(seed=5)
        _, validation = split_frames(cls.df)
        cls.p4 = Path(cls.tmp.name) / "p4b.json"
        fake_phase4b(cls.p4, len(validation), int(validation["Class"].sum()))
        cls.result = compute(cls.df, CREDITCARD, cls.p4, None, check_hgb_determinism=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_all_experiments_and_tracks(self):
        px.validate_experiment_tracks(self.result["experiments"])
        self.assertEqual(self.result["experiments"]["E0"]["tracks"][px.TRACK_A]["metrics"]["average_precision"], 0.5)
        for e in ("E1", "E2", "E3"):
            self.assertEqual(self.result["experiments"][e]["tracks"][px.TRACK_A]["status"], px.NOT_APPLICABLE)

    def test_no_test_rows_processed(self):
        iso = self.result["test_isolation"]
        self.assertLessEqual(iso["max_time_in_processed_rows"], VALIDATION_END_TIME)
        self.assertEqual(iso["test_rows_removed_before_label_access"], int((self.df.Time > VALIDATION_END_TIME).sum()))

    def test_invariant_to_test_labels_and_features(self):
        corrupt = self.df.copy()
        test = corrupt.Time > VALIDATION_END_TIME
        corrupt.loc[test, "Class"] = 1 - corrupt.loc[test, "Class"]
        corrupt.loc[test, BASE] = 1e6
        again = compute(corrupt, CREDITCARD, self.p4, None, check_hgb_determinism=False)
        assert_nested_close(self, strip_runtime(self.result), strip_runtime(again))

    def test_sensitivity_control(self):
        """Flipping validation labels changes the results, so the invariance test is not vacuous."""
        corrupt = self.df.copy()
        val = (corrupt.Time >= VALIDATION_START_TIME) & (corrupt.Time <= VALIDATION_END_TIME)
        corrupt.loc[val, "Class"] = 1 - corrupt.loc[val, "Class"]
        p4 = Path(self.tmp.name) / "p4b_flip.json"
        fake_phase4b(p4, int(val.sum()), int(corrupt.loc[val, "Class"].sum()))
        again = compute(corrupt, CREDITCARD, p4, None, check_hgb_determinism=False)
        self.assertNotEqual(again["experiments"]["E0"]["tracks"][px.TRACK_B]["validation"]["average_precision"],
                            self.result["experiments"]["E0"]["tracks"][px.TRACK_B]["validation"]["average_precision"])

    def test_track_b_and_c_are_separate_blocks(self):
        for e, block in self.result["experiments"].items():
            b, c = block["tracks"][px.TRACK_B], block["tracks"][px.TRACK_C]
            self.assertEqual((b["score_representation"], c["score_representation"]),
                             ("decision_function", "predict_proba"))
            self.assertIn("diagnostic only", c["role"])
            self.assertNotIn("default_rule_diagnostic", c)


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------

@unittest.skipUnless(DATA_AVAILABLE and RESULTS_FILE.is_file(), "raw dataset or result file absent")
class TestRealDataResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recorded = json.loads(RESULTS_FILE.read_text())
        cls.phase4b_sha = hashlib.sha256(PHASE4B_FILE.read_bytes()).hexdigest()

    def test_historical_track_quoted_unchanged(self):
        p4 = json.loads(PHASE4B_FILE.read_text())
        quoted = p4["experiments"]["A_chronological"]["results"]["logistic_regression_balanced"]["metrics"]
        a = self.recorded["experiments"]["E0"]["tracks"][px.TRACK_A]
        self.assertEqual(a["score_representation"], "predict_proba")
        self.assertEqual(a["metrics"]["average_precision"], quoted["average_precision"])
        self.assertEqual(a["metrics"]["roc_auc"], quoted["roc_auc"])

    def test_e0_equals_phase5b1_primary(self):
        ref = json.loads(PHASE5B1_FILE.read_text())["phase4_continuity"]["tracks"]["phase5_primary"]["metrics"]
        e0 = self.recorded["experiments"]["E0"]["tracks"][px.TRACK_B]["validation"]
        self.assertEqual(e0["average_precision"], ref["average_precision"])
        self.assertEqual(e0["roc_auc"], ref["roc_auc"])

    def test_counts_and_settings(self):
        d = self.recorded["data"]
        self.assertEqual((d["t_fit"]["n_rows"], d["t_fit"]["n_positive"]), (159_491, 356))
        self.assertEqual((d["validation"]["n_rows"], d["validation"]["n_positive"]), (42_722, 56))
        self.assertTrue(d["windows"]["matches_frozen_window_boundary"])
        self.assertEqual(self.recorded["temporal_stability_heldout_halves"]["n_usable"], 1984)
        self.assertTrue(all(v["second_fit_identical_decision_function"]
                            for v in self.recorded["hgb_determinism"].values()))

    def test_recompute_matches_recorded(self):
        df = pd.read_csv(CREDITCARD.path)
        again = compute(df, CREDITCARD, check_hgb_determinism=False)
        recorded = {k: v for k, v in self.recorded.items() if k in again}
        assert_nested_close(self, strip_runtime(recorded), json.loads(json.dumps(strip_runtime(again), default=float)))
        self.assertEqual(hashlib.sha256(PHASE4B_FILE.read_bytes()).hexdigest(), self.phase4b_sha)


if __name__ == "__main__":
    unittest.main()
