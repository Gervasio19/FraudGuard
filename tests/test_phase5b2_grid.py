"""Tests for protocol Section 4.2 step 2: the 28-configuration grid (Phase 5B-2).

Synthetic tests use generated data only. Real-data tests are skipped when the
dataset or the recorded result file is absent; they read the recorded file only.
"""

from __future__ import annotations

import itertools
import json
import math
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from scipy.special import expit

from src.config import CREDITCARD, PROJECT_ROOT
from src.data.split import VALIDATION_END_TIME
from src.models import phase5_grid as grid
from src.models import phase5b2_experiments as px
from src.models.phase4_continuity import build_continuity_lr
from src.models.phase5_settings import (
    T_CAL_START_TIME, T_FIT_END_TIME, VALIDATION_START_TIME, pre_registered_configuration,
)
from src.models.run_phase5b2_grid import RUNTIME_KEY, compute_grid, summarize

DATA_AVAILABLE = CREDITCARD.path.is_file()
RESULTS_FILE = PROJECT_ROOT / "reports" / "phase5b2_grid_results.json"
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]


def synthetic_frame(n=6000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = np.sort(rng.integers(0, 172_793, size=n)).astype(float)
    times = times[(times <= T_FIT_END_TIME) | (times >= T_CAL_START_TIME)]
    X = rng.normal(size=(len(times), 28))
    amount = np.round(np.exp(rng.normal(3.0, 1.5, size=len(times))), 2)
    y = (rng.random(len(times)) < expit(-4.0 + 2.5 * X[:, 0])).astype(int)
    df = pd.DataFrame(X, columns=FEATURES[:-1])
    df["Amount"] = amount
    df.insert(0, "Time", times)
    df["Class"] = y
    return df


def roles(df):
    fit = df[df.Time <= T_FIT_END_TIME]
    cal = df[(df.Time >= T_CAL_START_TIME) & (df.Time < VALIDATION_START_TIME)]
    val = df[(df.Time >= VALIDATION_START_TIME) & (df.Time <= VALIDATION_END_TIME)]
    return fit, cal, val


def fake_phase4b(path: Path, n_rows: int, n_positive: int) -> None:
    path.write_text(json.dumps({
        "models": {"logistic_regression_balanced": {"C": 1.0, "class_weight": "balanced"}},
        "experiments": {"A_chronological": {"results": {"logistic_regression_balanced": {"metrics": {
            "n_rows": n_rows, "n_positive": n_positive, "average_precision": 0.5, "roc_auc": 0.9}}}}},
    }))


class TestGridDefinition(unittest.TestCase):
    def test_28_configurations_equal_the_frozen_product(self):
        configs = grid.grid_configurations()
        self.assertEqual(len(configs), 28)
        self.assertEqual(len({c["config_id"] for c in configs}), 28)
        models = pre_registered_configuration()["models"]
        for family, n in ((grid.LR, 12), (grid.HGB, 16)):
            spec = models[family]["grid"]
            expected = [dict(zip(spec, v)) for v in itertools.product(*spec.values())]
            got = [c["params"] for c in configs if c["family"] == family]
            self.assertEqual(got, expected)
            self.assertEqual(len(got), n)

    def test_fixed_values_applied(self):
        fixed = pre_registered_configuration()["models"]
        for c in grid.grid_configurations():
            with warnings.catch_warnings():
                warnings.simplefilter("error", FutureWarning)  # no deprecated penalty= argument
                model = grid.build_pipeline(c["family"], c["params"], FEATURES, 42).named_steps["model"]
            self.assertEqual(model.random_state, 42)
            self.assertEqual(model.class_weight, c["params"]["class_weight"])
            if c["family"] == grid.LR:
                self.assertEqual((model.C, model.solver, model.max_iter), (c["params"]["C"], "lbfgs", 1000))
            else:
                f = fixed[grid.HGB]["fixed"]
                self.assertIs(model.early_stopping, False)
                self.assertEqual((model.max_iter, model.l2_regularization, model.max_bins),
                                 (f["max_iter"], f["l2_regularization"], f["max_bins"]))
                for key in ("learning_rate", "max_leaf_nodes", "min_samples_leaf"):
                    self.assertEqual(getattr(model, key), c["params"][key])

    def test_features_are_frozen_and_exclude_time(self):
        self.assertEqual(list(grid.model_features()), FEATURES)
        self.assertNotIn("Time", grid.model_features())


class TestFitConfiguration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = synthetic_frame()
        cls.fit, cls.cal, cls.val = roles(cls.df)
        configs = {(c["family"], tuple(c["params"].values())): c for c in grid.grid_configurations()}
        cls.lr = configs[(grid.LR, (1.0, "balanced"))]
        cls.hgb = configs[(grid.HGB, (0.1, 31, 20, "balanced"))]
        cls.f_lr = grid.fit_configuration(cls.lr, cls.fit, cls.cal, cls.val, CREDITCARD)
        cls.f_hgb = grid.fit_configuration(cls.hgb, cls.fit, cls.cal, cls.val, CREDITCARD)

    def test_row_roles_refused(self):
        test = self.df[self.df.Time > VALIDATION_END_TIME]
        bad = [(pd.concat([self.fit, self.cal]), self.cal, self.val),
               (self.fit, pd.concat([self.cal, self.val]), self.val),
               (self.fit, self.cal, pd.concat([self.val, test])),
               (self.fit, self.fit, self.val)]
        for fit, cal, val in bad:
            with self.assertRaises(ValueError):
                grid.check_row_roles(fit, cal, val, "Time")

    def test_fit_on_t_fit_only(self):
        scaler = self.f_lr["pipeline"].named_steps["preprocess"].named_transformers_["standardize"]
        self.assertEqual(scaler.n_samples_seen_, len(self.fit))
        self.assertEqual(self.f_lr["fit_info"]["n_fit_rows"], len(self.fit))

    def test_scores_both_partitions_and_representations(self):
        for f in (self.f_lr, self.f_hgb):
            p = f["pipeline"]
            for part, rows in (("validation", self.val), ("t_cal", self.cal)):
                s = f["scores"][part]
                np.testing.assert_array_equal(s["decision_function"], p.decision_function(rows[FEATURES]))
                np.testing.assert_array_equal(s["predict_proba"], p.predict_proba(rows[FEATURES])[:, 1])
                np.testing.assert_allclose(s["predict_proba"], expit(s["decision_function"]), rtol=0, atol=1e-15)
            np.testing.assert_array_equal(f["t_cal_index"], self.cal.index.to_numpy())

    def test_grid_points_reproduce_5b2_experiments(self):
        """LR C=1 balanced = the Phase 4 / E0 builder; the HGB point = experiment E2's builder."""
        e0 = build_continuity_lr(1000)(FEATURES, 42).fit(self.fit[FEATURES], self.fit["Class"])
        np.testing.assert_array_equal(self.f_lr["scores"]["validation"]["decision_function"],
                                      e0.decision_function(self.val[FEATURES]))
        e2 = px.build_hgb(FEATURES, 42).fit(self.fit[FEATURES], self.fit["Class"])
        np.testing.assert_array_equal(self.f_hgb["scores"]["validation"]["decision_function"],
                                      e2.decision_function(self.val[FEATURES]))

    def test_convergence_refit_rule(self):
        settings = pre_registered_configuration()
        settings["models"][grid.LR]["fixed"]["max_iter"] = 1  # forces a ConvergenceWarning
        with mock.patch.object(grid, "pre_registered_configuration", return_value=settings):
            f = grid.fit_configuration(self.lr, self.fit, self.cal, self.val, CREDITCARD)
        self.assertTrue(f["fit_info"]["convergence_warnings_first_fit"])
        self.assertTrue(f["fit_info"]["refit_with_max_iter_5000"])
        self.assertEqual(f["fit_info"]["effective_params"]["max_iter"], 5000)
        self.assertEqual(f["pipeline"].named_steps["model"].max_iter, 5000)


class TestSelectionRule(unittest.TestCase):
    @staticmethod
    def row(cid, family, ap, **params):
        return {"config_id": cid, "family": family, "average_precision": ap, "params": params}

    def test_max_ap_wins(self):
        rows = [self.row("a", grid.LR, 0.80, C=0.1, class_weight=None),
                self.row("b", grid.LR, 0.81, C=10.0, class_weight="balanced")]
        sel = grid.select_within_family(rows, grid.LR)
        self.assertEqual(sel["selected_config_id"], "b")
        self.assertAlmostEqual(sel["spread_max_minus_median"], 0.005)

    def test_lr_ties_prefer_smaller_c_then_unweighted(self):
        rows = [self.row("big", grid.LR, 0.8, C=1.0, class_weight=None),
                self.row("bal", grid.LR, 0.8 - 5e-13, C=0.1, class_weight="balanced"),
                self.row("none", grid.LR, 0.8, C=0.1, class_weight=None),
                self.row("far", grid.LR, 0.8 - 2e-12, C=1e-4, class_weight=None)]
        sel = grid.select_within_family(rows, grid.LR)
        self.assertEqual(sel["selected_config_id"], "none")
        self.assertEqual(sel["n_within_tie_tolerance"], 3)

    def test_hgb_tie_order(self):
        def h(cid, lr, leaves, msl, cw):
            return self.row(cid, grid.HGB, 0.9, learning_rate=lr, max_leaf_nodes=leaves,
                            min_samples_leaf=msl, class_weight=cw)
        rows = [h("a", 0.05, 31, 200, None), h("b", 0.1, 15, 20, None), h("c", 0.1, 15, 200, "balanced"),
                h("d", 0.05, 15, 200, "balanced"), h("e", 0.05, 15, 200, None)]
        self.assertEqual(grid.select_within_family(rows, grid.HGB)["selected_config_id"], "e")
        self.assertEqual(grid.select_within_family(rows[:4], grid.HGB)["selected_config_id"], "d")
        self.assertEqual(grid.select_within_family(rows[:3], grid.HGB)["selected_config_id"], "c")

    def test_refusals(self):
        with self.assertRaises(ValueError):
            grid.select_within_family([], grid.LR)
        with self.assertRaises(ValueError):
            grid.select_within_family([self.row("a", grid.LR, math.nan, C=1.0, class_weight=None)], grid.LR)


class TestGridSynthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.df = synthetic_frame(seed=3)
        _, _, val = roles(cls.df)
        cls.p4 = Path(cls.tmp.name) / "p4b.json"
        fake_phase4b(cls.p4, len(val), int(val["Class"].sum()))
        cls.result = summarize(compute_grid(cls.df, CREDITCARD, check_hgb_determinism=False),
                               CREDITCARD, cls.p4, None)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_all_configurations_and_tracks(self):
        self.assertEqual(len(self.result["grid_table_track_b"]), 28)
        with_a = [cid for cid, b in self.result["configurations"].items()
                  if b["tracks"][px.TRACK_A].get("status") != px.NOT_APPLICABLE]
        self.assertEqual(len(with_a), 1)
        self.assertEqual(self.result["configurations"][with_a[0]]["params"], {"C": 1.0, "class_weight": "balanced"})
        for block in self.result["configurations"].values():
            self.assertEqual(block["tracks"][px.TRACK_B]["score_representation"], "decision_function")
            self.assertEqual(block["tracks"][px.TRACK_C]["score_representation"], "predict_proba")
        for fam in grid.FAMILIES:
            self.assertIn(self.result["step3_selection_record"][fam]["selected_config_id"],
                          [r["config_id"] for r in self.result["grid_table_track_b"] if r["family"] == fam])

    def test_invariant_to_test_labels_and_features(self):
        corrupt = self.df.copy()
        test = corrupt.Time > VALIDATION_END_TIME
        corrupt.loc[test, "Class"] = 1 - corrupt.loc[test, "Class"]
        corrupt.loc[test, FEATURES] = 1e6
        again = summarize(compute_grid(corrupt, CREDITCARD, check_hgb_determinism=False), CREDITCARD, self.p4, None)
        strip = lambda r: json.dumps({k: v for k, v in r.items() if k != RUNTIME_KEY}, sort_keys=True, default=str)
        self.assertEqual(strip(self.result), strip(again))


@unittest.skipUnless(DATA_AVAILABLE and RESULTS_FILE.is_file(), "raw dataset or result file absent")
class TestRealDataGridResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = json.loads(RESULTS_FILE.read_text())

    def test_counts_and_isolation(self):
        d = self.r["data"]
        self.assertEqual((d["t_fit"]["n_rows"], d["t_fit"]["n_positive"]), (159_491, 356))
        self.assertEqual(d["t_cal"]["n_rows"], 39_873)
        self.assertEqual((d["validation"]["n_rows"], d["validation"]["n_positive"]), (42_722, 56))
        self.assertLessEqual(self.r["test_isolation"]["max_time_in_processed_rows"], VALIDATION_END_TIME)
        self.assertEqual(len(self.r["grid_table_track_b"]), 28)

    def test_continuity_with_5b2_experiments(self):
        for exp in ("E0", "E2"):
            self.assertEqual(self.r["continuity_with_phase5b2_experiments"][exp]["abs_difference"], 0.0)

    def test_hgb_determinism_recorded(self):
        det = self.r["hgb_determinism"]
        self.assertEqual(len(det), 16)
        self.assertTrue(all(v["second_fit_identical_scores"] for v in det.values()))

    def test_selection_reproducible_from_table(self):
        for fam in grid.FAMILIES:
            again = grid.select_within_family(self.r["grid_table_track_b"], fam)
            self.assertEqual(again["selected_config_id"], self.r["step3_selection_record"][fam]["selected_config_id"])


if __name__ == "__main__":
    unittest.main()
