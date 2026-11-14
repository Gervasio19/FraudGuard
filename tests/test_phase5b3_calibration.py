"""Tests for the Phase 5B-3 calibration module (protocol Sections 6.3-6.5).

Synthetic data only; the FraudGuard dataset is never read here.
"""

from __future__ import annotations

import inspect
import math
import unittest

import numpy as np
from scipy.special import expit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from src.evaluation import calibration as cal
from src.models.phase5_settings import RELIABILITY_BIN_EDGES


def sample(n=20_000, slope=1.7, intercept=-3.0, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.normal(0.0, 2.0, size=n)
    y = rng.random(n) < expit(slope * s + intercept)
    return s, y


class TestPlattFit(unittest.TestCase):
    def test_recovers_a_known_sigmoid(self):
        s, y = sample(n=200_000)
        fit = cal.fit_platt(s, y, positive_label=True)
        self.assertEqual(fit["status"], "ok")
        self.assertTrue(fit["converged"])
        self.assertAlmostEqual(fit["A"], 1.7, delta=0.05)
        self.assertAlmostEqual(fit["B"], -3.0, delta=0.08)

    def test_is_the_minimum_of_the_platt_objective(self):
        s, y = sample(n=3_000, seed=3)
        fit = cal.fit_platt(s, y, positive_label=True)
        n1, n0 = y.sum(), (~y).sum()
        target = np.where(y, (n1 + 1) / (n1 + 2), 1 / (n0 + 2))

        def objective(A, B):
            z = A * s + B
            return float(np.sum(np.logaddexp(0, z) - target * z))

        best = objective(fit["A"], fit["B"])
        for dA, dB in ((1e-3, 0), (-1e-3, 0), (0, 1e-3), (0, -1e-3)):
            self.assertGreater(objective(fit["A"] + dA, fit["B"] + dB), best)

    def test_equivalent_to_the_public_scikit_learn_calibrator(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(12_000, 4))
        y = (rng.random(12_000) < expit(-4.0 + 2.0 * X[:, 0] - X[:, 1])).astype(int)
        for estimator in (LogisticRegression(class_weight="balanced"),
                          HistGradientBoostingClassifier(max_iter=30, random_state=0)):
            with self.subTest(estimator=type(estimator).__name__):
                estimator.fit(X[:6_000], y[:6_000])
                X_cal, y_cal = X[6_000:], y[6_000:]
                s = estimator.decision_function(X_cal)
                own = cal.fit_platt(s, y_cal)
                public = CalibratedClassifierCV(FrozenEstimator(estimator), method="sigmoid").fit(X_cal, y_cal)
                self.assertEqual(len(public.calibrated_classifiers_), 1)
                p_public = public.predict_proba(X_cal)[:, 1]
                z_public = np.log(p_public) - np.log1p(-p_public)
                # same convex objective; the difference is the optimizers' tolerance
                self.assertLess(np.abs(own["A"] * s + own["B"] - z_public).max(), 1e-5)
                self.assertGreater(own["A"], 0)

    def test_weights_equal_an_expanded_sample(self):
        s, y = sample(n=2_000, seed=5)
        w = np.random.default_rng(6).integers(0, 4, size=len(s))
        weighted = cal.fit_platt(s, y, sample_weight=w, positive_label=True)
        expanded = cal.fit_platt(np.repeat(s, w), np.repeat(y, w), positive_label=True)
        self.assertAlmostEqual(weighted["A"], expanded["A"], places=9)
        self.assertAlmostEqual(weighted["B"], expanded["B"], places=9)

    def test_scale_equivariance_and_extreme_scores(self):
        s, y = sample(n=5_000, seed=7)
        base = cal.fit_platt(s, y, positive_label=True)
        for factor in (1e-6, 1e4):
            scaled = cal.fit_platt(s * factor, y, positive_label=True)
            self.assertEqual(scaled["status"], "ok")
            self.assertAlmostEqual(scaled["A"] * factor, base["A"], places=6)
            self.assertAlmostEqual(scaled["B"], base["B"], places=6)
        shifted = cal.fit_platt(s + 1e6, y, positive_label=True)
        self.assertAlmostEqual(shifted["A"], base["A"], places=5)

    def test_perfect_separation_stays_finite(self):
        s = np.r_[np.linspace(-60, -40, 500), np.linspace(40, 60, 20)]
        y = np.r_[np.zeros(500, bool), np.ones(20, bool)]
        fit = cal.fit_platt(s, y, positive_label=True)
        self.assertEqual(fit["status"], "ok")  # Platt's smoothed targets keep the optimum finite
        self.assertTrue(math.isfinite(fit["A"]) and fit["A"] > 0)
        self.assertEqual(cal.classify_calibrator(fit, s), cal.VALID)

    def test_failure_cases_are_reported_not_raised(self):
        s, y = sample(n=500, seed=8)
        self.assertEqual(cal.fit_platt(s, np.zeros(500, bool), positive_label=True)["status"], "no positives")
        self.assertEqual(cal.fit_platt(s, np.ones(500, bool), positive_label=True)["status"], "no negatives")
        self.assertEqual(cal.fit_platt(np.full(500, 2.0), y, positive_label=True)["status"], "constant scores")
        bad = s.copy()
        bad[3] = np.inf
        self.assertEqual(cal.fit_platt(bad, y, positive_label=True)["status"], "non-finite scores")
        with self.assertRaises(ValueError):
            cal.fit_platt(s, y, sample_weight=-np.ones(500), positive_label=True)
        with self.assertRaises(ValueError):
            cal.fit_platt(s[:-1], y, positive_label=True)

    def test_technical_failure_classification(self):
        s, y = sample(n=4_000, seed=9)
        good = cal.fit_platt(s, y, positive_label=True)
        self.assertEqual(cal.classify_calibrator(good, s), cal.VALID)
        reversed_fit = cal.fit_platt(-s, y, positive_label=True)
        self.assertLess(reversed_fit["A"], 0)
        self.assertEqual(cal.classify_calibrator(reversed_fit, -s), cal.INVALID_NON_POSITIVE_SLOPE)
        self.assertEqual(cal.classify_calibrator({"status": "ok", "A": 0.0, "B": -3.0}, s),
                         cal.INVALID_NON_POSITIVE_SLOPE)
        self.assertEqual(cal.classify_calibrator({"status": "ok", "A": 1e-10, "B": -3.0}, s),
                         cal.INVALID_DEGENERATE)  # logit SD about 2e-10 < 1e-8
        self.assertEqual(cal.classify_calibrator({"status": "ok", "A": 1e-7, "B": -3.0}, s), cal.VALID)
        self.assertEqual(cal.classify_calibrator({"status": "no positives"}, s), cal.INVALID_NO_POSITIVES)
        self.assertEqual(cal.classify_calibrator({"status": "constant scores"}, s), cal.INVALID_FIT_FAILURE)
        failed = cal.primary_calibrator(-s, y, True)
        self.assertFalse(failed["calibrated"])
        self.assertIn("uncalibrated scores", failed["consequence"])


class TestMetrics(unittest.TestCase):
    def test_garwood_matches_the_protocol_worked_values(self):
        lower, upper = cal.garwood_interval(56)
        self.assertAlmostEqual(lower, 42.3, delta=0.05)
        self.assertAlmostEqual(upper, 72.7, delta=0.05)
        self.assertEqual(cal.garwood_interval(0)[0], 0.0)
        self.assertAlmostEqual(cal.garwood_interval(0)[1], -math.log(0.025), places=9)

    def test_clopper_pearson_edges(self):
        self.assertEqual(cal.clopper_pearson(0, 100)[0], 0.0)
        self.assertAlmostEqual(cal.clopper_pearson(0, 100)[1], 1 - 0.025 ** (1 / 100), places=9)
        self.assertEqual(cal.clopper_pearson(100, 100)[1], 1.0)
        self.assertTrue(all(math.isnan(v) for v in cal.clopper_pearson(0, 0)))

    def test_brier_and_log_loss_match_scikit_learn(self):
        s, y = sample(n=5_000, seed=11)
        z = 1.2 * s - 2.5
        metrics = cal.calibration_metrics(y, expit(z), z, with_intervals=False)
        self.assertAlmostEqual(metrics["brier_score"], brier_score_loss(y, expit(z)), places=12)
        self.assertAlmostEqual(metrics["log_loss"], log_loss(y, expit(z)), places=10)
        prevalence = y.mean()
        self.assertAlmostEqual(metrics["brier_skill"],
                               1 - metrics["brier_score"] / (prevalence * (1 - prevalence)), places=12)

    def test_log_loss_from_logits_is_finite_when_probabilities_saturate(self):
        y = np.array([True, False])
        z = np.array([-800.0, 800.0])  # expit gives exactly 0 and 1
        terms = cal.log_loss_terms(y, logits=z)
        self.assertTrue(np.isfinite(terms).all())
        np.testing.assert_allclose(terms, [800.0, 800.0])

    def test_intercept_and_slope_on_well_specified_data(self):
        s, y = sample(n=60_000, seed=12)
        z = 1.7 * s - 3.0
        metrics = cal.calibration_metrics(y, expit(z), z)
        a, b = metrics["calibration_intercept"], metrics["calibration_slope"]
        self.assertLess(a["profile_likelihood_95"][0], 0.0)
        self.assertGreater(a["profile_likelihood_95"][1], 0.0)
        self.assertLess(b["profile_likelihood_95"][0], 1.0)
        self.assertGreater(b["profile_likelihood_95"][1], 1.0)
        for m in (a, b):
            self.assertLess(m["profile_likelihood_95"][0], m["estimate"])
            self.assertGreater(m["profile_likelihood_95"][1], m["estimate"])
        # profile interval is close to Wald when the sample is large
        self.assertAlmostEqual(a["profile_likelihood_95"][1] - a["estimate"], 1.96 * a["wald_standard_error"],
                               delta=0.01)

    def test_intercept_and_slope_detect_miscalibration(self):
        s, y = sample(n=60_000, seed=13)
        z = 1.7 * s - 3.0
        self.assertAlmostEqual(cal.calibration_intercept(y, z + 1.0)["estimate"], -1.0, delta=0.05)
        self.assertAlmostEqual(cal.calibration_slope(y, 2.0 * z)["estimate"], 0.5, delta=0.02)
        negative = cal.calibration_slope(y, -z)
        self.assertEqual(negative["status"], "ok")  # a negative slope is a result, not a failure
        self.assertLess(negative["estimate"], 0)

    def test_fitted_metrics_not_estimable(self):
        z = np.linspace(-5, 1, 200)
        none = np.zeros(200, bool)
        self.assertIn("zero frauds", cal.calibration_intercept(none, z)["status"])
        self.assertIn("zero frauds", cal.calibration_slope(none, z)["status"])
        separated = z > 0.5
        self.assertIn("fit failure", cal.calibration_slope(separated, z)["status"])
        self.assertIn("non-finite", cal.calibration_slope(separated, np.where(separated, np.inf, z))["status"])

    def test_isotonic_sensitivity_reports_zero_probabilities(self):
        s, y = sample(n=4_000, seed=14)
        p = cal.isotonic_probabilities(s[:2_000], y[:2_000], s[2_000:], True)
        self.assertTrue(((p >= 0) & (p <= 1)).all())
        self.assertTrue(np.all(np.diff(p[np.argsort(s[2_000:])]) >= 0))
        metrics = cal.calibration_metrics(y[2_000:], p, None, with_intervals=False)
        self.assertGreater(metrics["n_probability_exactly_zero"], 0)
        self.assertTrue(math.isfinite(metrics["log_loss"]))
        self.assertIn("clipped", metrics["log_loss_source"])
        self.assertIn("not estimable", metrics["calibration_slope"]["status"])


class TestReliability(unittest.TestCase):
    def test_decade_bins_partition_every_row(self):
        rng = np.random.default_rng(15)
        p = np.r_[10 ** rng.uniform(-9, 0, size=5_000), [0.0, 1.0, 1e-6, 0.1]]
        y = rng.random(len(p)) < p
        table = cal.decade_reliability(y, p)
        self.assertEqual(len(table), len(RELIABILITY_BIN_EDGES) - 1)
        self.assertEqual(sum(r["n_rows"] for r in table), len(p))
        self.assertEqual(sum(r["observed_frauds"] for r in table), int(y.sum()))
        self.assertAlmostEqual(sum(r["expected_frauds"] for r in table), p.sum(), places=9)
        self.assertEqual(table[-1]["bin"], "[0.1, 1]")

    def test_interpretable_flag_needs_five_observed_and_five_expected(self):
        y = np.r_[np.ones(5, bool), np.zeros(75, bool)]
        five_expected = np.full(80, 0.0625)  # 80 / 16 = 5 exactly in float64
        one_bin = np.zeros(80, int)
        self.assertTrue(cal.reliability_table(y, five_expected, one_bin, ["all"])[0]["interpretable"])
        self.assertFalse(cal.reliability_table(y, five_expected * 0.99, one_bin, ["all"])[0]["interpretable"])
        y4 = np.r_[np.ones(4, bool), np.zeros(76, bool)]
        self.assertFalse(cal.reliability_table(y4, five_expected, one_bin, ["all"])[0]["interpretable"])
        empty = cal.reliability_table(y, five_expected, one_bin, ["all", "none"])[1]
        self.assertEqual(empty["n_rows"], 0)
        self.assertFalse(empty["interpretable"])

    def test_k_derived_edges_are_label_free_and_bands_have_the_right_sizes(self):
        self.assertNotIn("y_true", inspect.signature(cal.k_derived_score_edges).parameters)
        self.assertEqual(list(inspect.signature(cal.k_derived_score_edges).parameters),
                         ["scores", "times", "positions", "ks"])
        rng = np.random.default_rng(16)
        scores = rng.normal(size=2_000)  # continuous: no ties
        times, positions = np.arange(2_000), np.arange(2_000)
        edges = cal.k_derived_score_edges(scores, times, positions, [25, 50, 100])
        index, labels = cal.k_band_index(scores, edges)
        self.assertEqual(np.bincount(index, minlength=4).tolist(), [25, 25, 50, 1_900])
        self.assertEqual(len(labels), 4)


def clustered_period(n=6_000, start=1_000, seconds=20_000, seed=0, fraud_seconds=None):
    rng = np.random.default_rng(seed)
    times = np.sort(rng.integers(start, start + seconds, size=n))
    s = rng.normal(0, 2, size=n)
    y = rng.random(n) < expit(1.5 * s - 3.0)
    if fraud_seconds is not None:  # keep frauds only inside a short span of the period
        y &= (times >= fraud_seconds[0]) & (times < fraud_seconds[1])
    return times, s, y


class TestBootstraps(unittest.TestCase):
    def test_tcal_bootstrap_is_reproducible_and_shared_between_models(self):
        times, s, y = clustered_period()
        args = (y, times, 1_000, 20_999, 900, 60, 42, True)
        first = cal.tcal_calibrator_bootstrap(s, *args)
        second = cal.tcal_calibrator_bootstrap(s, *args)
        np.testing.assert_array_equal(first["parameters"], second["parameters"])
        self.assertEqual(first["counts"][cal.VALID], 60)
        self.assertTrue(first["estimable"])
        self.assertLess(first["A"]["percentile_95"][0], first["A"]["percentile_95"][1])
        # a second "model" (monotone transform of the scores) sees the same resampled rows
        other = cal.tcal_calibrator_bootstrap(2.0 * s, *args)
        np.testing.assert_allclose(other["parameters"][:, 0] * 2.0, first["parameters"][:, 0], rtol=1e-7)

    def test_tcal_bootstrap_not_estimable_when_too_few_replicates_fit(self):
        times, s, y = clustered_period(fraud_seconds=(5_000, 5_300))  # frauds in one 5-minute span
        out = cal.tcal_calibrator_bootstrap(s, y, times, 1_000, 20_999, 900, 80, 42, True)
        self.assertGreater(out["counts"][cal.INVALID_NO_POSITIVES], 8)
        self.assertEqual(sum(out["counts"].values()), 80)
        self.assertFalse(out["estimable"])
        self.assertIn("not estimable under the pre-registered rule", out["status"])
        self.assertNotIn("A", out)  # no interval and no retry

    def test_negative_slope_and_reversed_calibrators_are_excluded_from_fit_bootstrap(self):
        times, s, y = clustered_period()
        out = cal.tcal_calibrator_bootstrap(-s, y, times, 1_000, 20_999, 900, 30, 42, True)
        self.assertEqual(out["counts"][cal.INVALID_NON_POSITIVE_SLOPE], 30)
        self.assertFalse(out["estimable"])

    def test_evaluation_bootstrap_keeps_negative_slopes_and_never_drops_direct_metrics(self):
        times, s, y = clustered_period(seed=2)
        z = 1.5 * s - 3.0
        vectors = {"good": {"probabilities": expit(z), "logits": z, "fit_metrics": True},
                   "reversed": {"probabilities": expit(-z - 6.0), "logits": -z - 6.0, "fit_metrics": True},
                   "isotonic_like": {"probabilities": np.clip(expit(z), 0, 1), "logits": None, "fit_metrics": False}}
        out = cal.calibration_evaluation_bootstrap(vectors, y, times, 1_000, 20_999, 900, 40, 0,
                                                   pairs=[("good", "reversed")], positive_label=True)
        for name in vectors:
            for metric in ("observed_expected", "brier_score", "log_loss"):
                self.assertEqual(out["by_vector"][name][metric]["n_valid"], 40)
        reversed_slope = out["by_vector"]["reversed"]["calibration_slope"]
        self.assertEqual(reversed_slope["status"], "ok")
        self.assertEqual(reversed_slope["share_slope_at_or_below_zero"], 1.0)
        good_slope = out["by_vector"]["good"]["calibration_slope"]
        self.assertLess(good_slope["percentile_95"][0], 1.0)
        self.assertGreater(good_slope["percentile_95"][1], 1.0)
        self.assertIn("not available", out["by_vector"]["isotonic_like"]["calibration_slope"]["status"])
        paired = out["paired"]["good minus reversed"]["brier_score"]
        self.assertLess(paired["percentile_95"][1], 0)  # the good vector has the lower Brier score

    def test_evaluation_bootstrap_counts_zero_fraud_replicates(self):
        times, s, y = clustered_period(fraud_seconds=(5_000, 5_300))
        z = 1.5 * s - 3.0
        out = cal.calibration_evaluation_bootstrap(
            {"v": {"probabilities": expit(z), "logits": z, "fit_metrics": True}}, y, times, 1_000, 20_999,
            900, 80, 0, positive_label=True)
        self.assertGreater(out["n_zero_fraud_replicates"], 8)
        block = out["by_vector"]["v"]
        self.assertEqual(block["observed_expected"]["n_valid"], 80)  # O = 0 gives O/E = 0, a valid value
        self.assertEqual(block["brier_skill"]["n_valid"], 80 - out["n_zero_fraud_replicates"])
        self.assertIn("not estimable under the pre-registered rule", block["calibration_slope"]["status"])


if __name__ == "__main__":
    unittest.main()
