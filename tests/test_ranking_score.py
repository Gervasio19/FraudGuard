"""Tests for the Phase 5 ranking-score interface (protocol Addendum C, items C5 and C8).

Most tests fit nothing: estimators are stubs with fixed scores, real
scikit-learn classes are only inspected unfitted, and the sigmoid calibrator's
sign convention is first checked with parameters set by hand.

Exception: ``TestSyntheticFittedIntegration`` (mentor-requested, Appendix C7)
fits small LogisticRegression and HistGradientBoosting models and sigmoid
calibrators on *generated* arrays, to confirm positive-class orientation and
the Platt sign convention on fitted objects. It never reads the FraudGuard
dataset and evaluates no project model.
"""

from __future__ import annotations

import unittest

import numpy as np
from scipy.special import expit
from sklearn.calibration import _SigmoidCalibration
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from src.models.ranking_score import (
    calibrated_logits,
    is_numerically_constant,
    protocol_platt_parameters,
    ranking_order,
    ranking_scores,
    reliability_bin_index,
)


class _Stub:
    """A 'fitted' binary estimator returning fixed decision scores."""

    def __init__(self, scores, classes=(0, 1)):
        self.classes_ = np.asarray(classes)
        self._scores = scores

    def decision_function(self, X):
        return self._scores


class _ProbaOnly:
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        return np.array([[0.5, 0.5]])


class TestRankingScoreInterface(unittest.TestCase):
    def test_returns_float64_copy(self):
        raw = np.array([1, -2, 3], dtype=np.float32)
        scores = ranking_scores(_Stub(raw), None)
        self.assertEqual(scores.dtype, np.float64)
        scores[0] = 99
        self.assertEqual(raw[0], 1)

    def test_column_vector_is_flattened(self):
        scores = ranking_scores(_Stub(np.array([[0.1], [0.2]])), None)
        np.testing.assert_array_equal(scores, [0.1, 0.2])

    def test_two_column_output_rejected(self):
        with self.assertRaises(ValueError):
            ranking_scores(_Stub(np.zeros((3, 2))), None)

    def test_no_silent_predict_proba_fallback(self):
        with self.assertRaises(TypeError):
            ranking_scores(_ProbaOnly(), None)

    def test_non_binary_or_unknown_label_rejected(self):
        with self.assertRaises(ValueError):
            ranking_scores(_Stub(np.zeros(3), classes=(0, 1, 2)), None)
        with self.assertRaises(ValueError):
            ranking_scores(_Stub(np.zeros(3)), None, positive_label=7)

    def test_non_finite_rejected(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.assertRaises(ValueError):
                ranking_scores(_Stub(np.array([0.0, bad])), None)

    def test_positive_label_first_class_reverses_order_exactly(self):
        raw = np.array([0.3, -1.2, 5.0, 0.0])
        pos_second = ranking_scores(_Stub(raw, classes=("legit", "fraud")), None, "fraud")
        pos_first = ranking_scores(_Stub(raw, classes=("fraud", "legit")), None, "fraud")
        np.testing.assert_array_equal(pos_first, -pos_second)
        np.testing.assert_array_equal(np.argsort(pos_first, kind="stable"),
                                      np.argsort(-pos_second, kind="stable"))


class TestInstalledScikitLearnApi(unittest.TestCase):
    """Class-level API checks only; nothing is fitted."""

    def test_both_families_expose_decision_function(self):
        for estimator in (LogisticRegression(), HistGradientBoostingClassifier()):
            self.assertTrue(hasattr(estimator, "decision_function"))
            self.assertTrue(hasattr(Pipeline([("model", estimator)]), "decision_function"))

    def test_reference_baseline_has_no_decision_function(self):
        # The majority-class reference is reported only; its AP equals prevalence
        # for any constant score, so it is exempt from this interface.
        with self.assertRaises(TypeError):
            ranking_scores(DummyClassifier(strategy="prior"), None)


class TestSaturationEvidence(unittest.TestCase):
    """Why probabilities are not used for ranking (pure arithmetic, no models)."""

    def test_probability_saturates_to_exactly_one(self):
        self.assertLess(expit(36.0), 1.0)
        self.assertEqual(expit(37.0), 1.0)
        self.assertEqual(expit(50.0), 1.0)

    def test_distinct_logits_collapse_to_ties_after_expit(self):
        logits = np.linspace(30.0, 31.0, 1001)
        self.assertEqual(len(np.unique(logits)), 1001)
        self.assertLess(len(np.unique(expit(logits))), 1001)

    def test_ap_unchanged_by_expit_without_saturation(self):
        rng = np.random.default_rng(0)
        y = rng.random(400) < 0.05
        z = rng.normal(size=400) * 3 + 2 * y
        self.assertEqual(len(np.unique(expit(z))), len(np.unique(z)))
        self.assertAlmostEqual(average_precision_score(y, z), average_precision_score(y, expit(z)), places=15)

    def test_saturation_ties_change_ap(self):
        # The positive ranks above a negative by logit, but both become p = 1.0.
        z = np.array([40.0, 38.0, 0.0, -1.0])
        y = np.array([1, 0, 0, 0])
        self.assertEqual(expit(z[0]), expit(z[1]))
        self.assertEqual(average_precision_score(y, z), 1.0)
        self.assertEqual(average_precision_score(y, expit(z)), 0.5)


class TestRankingOrder(unittest.TestCase):
    def test_key_score_then_time_then_position(self):
        scores = [0.5, 0.9, 0.5, 0.5, 0.9]
        times = [10, 20, 5, 10, 20]
        positions = [3, 1, 4, 2, 0]
        # 0.9 group: same time -> position 0 (idx 4) then 1 (idx 1);
        # 0.5 group: time 5 (idx 2), then time 10 by position 2 (idx 3), 3 (idx 0).
        np.testing.assert_array_equal(ranking_order(scores, times, positions), [4, 1, 2, 3, 0])

    def test_negative_zero_ties_with_zero(self):
        order = ranking_order([0.0, -0.0], [5, 1], [0, 1])
        np.testing.assert_array_equal(order, [1, 0])  # decided by Time, not by the sign of zero

    def test_exact_comparison_no_tolerance(self):
        a = 0.1 + 0.2
        order = ranking_order([0.3, a], [1, 2], [0, 1])  # 0.30000000000000004 > 0.3
        np.testing.assert_array_equal(order, [1, 0])

    def test_ap_is_invariant_to_tie_order(self):
        scores = np.array([2.0, 1.0, 1.0, 1.0, 0.0])
        y = np.array([1, 1, 0, 0, 0])
        permuted = np.array([0, 3, 2, 1, 4])
        self.assertEqual(average_precision_score(y, scores),
                         average_precision_score(y[permuted], scores[permuted]))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            ranking_order([1.0, np.nan], [0, 1], [0, 1])
        with self.assertRaises(ValueError):
            ranking_order([1.0, 2.0], [0, 1], [0, 0])
        with self.assertRaises(ValueError):
            ranking_order([1.0], [0, 1], [0, 1])


class TestCalibrationConventions(unittest.TestCase):
    def test_sign_mapping(self):
        self.assertEqual(protocol_platt_parameters(-2.0, 0.5), (2.0, -0.5))
        with self.assertRaises(ValueError):
            protocol_platt_parameters(np.nan, 0.0)

    def test_mapping_matches_installed_sklearn_sigmoid(self):
        calibrator = _SigmoidCalibration()
        calibrator.a_, calibrator.b_ = -1.7, 3.2  # set by hand, not fitted
        s = np.linspace(-12, 12, 97)
        A, B = protocol_platt_parameters(calibrator.a_, calibrator.b_)
        np.testing.assert_allclose(calibrator.predict(s), expit(calibrated_logits(s, A, B)), rtol=0, atol=0)
        self.assertGreater(A, 0)  # an increasing calibrator has sklearn a_ < 0

    def test_numerical_constancy_guard(self):
        self.assertTrue(is_numerically_constant(np.full(10, -6.0)))
        self.assertTrue(is_numerically_constant(-6.0 + 1e-10 * np.arange(10)))
        self.assertFalse(is_numerically_constant(-6.0 + 0.01 * np.random.default_rng(0).normal(size=50)))
        self.assertTrue(is_numerically_constant([0.0, np.inf]))
        with self.assertRaises(ValueError):
            is_numerically_constant([])

    def test_constancy_guard_below_threshold(self):
        z = np.array([1.0 - 1e-8, 1.0 + 1e-8])
        self.assertLess(np.std(z), 1e-8)  # 9.99999999e-09 in float64
        self.assertTrue(is_numerically_constant(z))

    def test_constancy_guard_exactly_at_threshold_is_not_constant(self):
        z = np.array([-1e-8, 1e-8])
        self.assertEqual(np.std(z), 1e-8)  # population SD is exactly 1e-8 in float64
        self.assertFalse(is_numerically_constant(z))  # strict "<": equality is not constant

    def test_constancy_guard_above_threshold(self):
        z = np.array([-2e-8, 2e-8])
        self.assertGreater(np.std(z), 1e-8)
        self.assertFalse(is_numerically_constant(z))

    def test_constancy_guard_uses_population_sd(self):
        # ddof changes the outcome here: population SD < 1e-8 <= sample SD.
        z = np.array([5.0, 5.0 + 2e-8])
        self.assertLess(np.std(z, ddof=0), 1e-8)
        self.assertGreaterEqual(np.std(z, ddof=1), 1e-8)
        self.assertTrue(is_numerically_constant(z))


class TestReliabilityBins(unittest.TestCase):
    """Bin assignment of Addendum C, item C8 (synthetic probabilities; no models)."""

    def test_every_edge_value_starts_its_bin(self):
        edges = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        np.testing.assert_array_equal(reliability_bin_index(edges), [0, 1, 2, 3, 4, 5, 6])

    def test_values_just_below_an_edge_stay_in_the_lower_bin(self):
        below = np.nextafter(np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]), 0.0)
        np.testing.assert_array_equal(reliability_bin_index(below), [0, 1, 2, 3, 4, 5])

    def test_lower_tail_bin_holds_zero_and_tiny_probabilities(self):
        np.testing.assert_array_equal(reliability_bin_index([0.0, 5e-324, 1e-300, 9.99e-7]), [0, 0, 0, 0])

    def test_last_bin_is_closed_and_includes_one(self):
        np.testing.assert_array_equal(reliability_bin_index([0.1, 0.5, np.nextafter(1.0, 0.0), 1.0]),
                                      [6, 6, 6, 6])

    def test_bins_partition_the_unit_interval(self):
        p = np.concatenate([np.random.default_rng(0).random(10_000), np.logspace(-12, 0, 2_000), [0.0, 1.0]])
        index = reliability_bin_index(p)
        self.assertTrue(np.all((index >= 0) & (index <= 6)))
        edges = np.array([0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])
        lower, upper = edges[index], edges[index + 1]
        self.assertTrue(np.all(p >= lower))
        self.assertTrue(np.all((p < upper) | ((index == 6) & (p <= 1.0))))

    def test_out_of_range_and_non_finite_rejected(self):
        for bad in ([-1e-12], [1.0 + 1e-12], [np.nan], [np.inf]):
            with self.assertRaises(ValueError):
                reliability_bin_index(bad)
        with self.assertRaises(ValueError):
            reliability_bin_index([0.5], edges=[0.0, 0.5, 0.5, 1.0])


class TestSyntheticFittedIntegration(unittest.TestCase):
    """Mentor-requested integration check on *synthetic* data only (Phase 5B-0).

    Tiny models are fitted on generated arrays to confirm, on fitted objects,
    the positive-class orientation of ``decision_function`` and the Platt sign
    convention. The FraudGuard dataset is never read, and nothing here is an
    experiment, a selection step or an evaluation of a project model.
    """

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)

        def sample(n, direction=1.0):
            X = rng.normal(size=(n, 2))
            p = expit(direction * (2.0 * X[:, 0] - 1.0 * X[:, 1]) - 2.0)
            return X, (rng.random(n) < p).astype(int)

        cls.X, cls.y = sample(3000)
        cls.X_cal, cls.y_cal = sample(3000)
        cls.X_cal_rev, cls.y_cal_rev = sample(3000, direction=-1.0)  # reversed association
        cls.lr = LogisticRegression(max_iter=1000).fit(cls.X, cls.y)
        cls.hgb = HistGradientBoostingClassifier(max_iter=30, early_stopping=False,
                                                 random_state=0).fit(cls.X, cls.y)

    def assert_positive_orientation(self, estimator, X, y, positive_label):
        s = ranking_scores(estimator, X, positive_label)
        self.assertGreater(s[y == positive_label].mean(), s[y != positive_label].mean())

    def test_positive_class_is_classes_1_for_both_families(self):
        for model in (self.lr, self.hgb):
            np.testing.assert_array_equal(model.classes_, [0, 1])
            s = ranking_scores(model, self.X)
            np.testing.assert_array_equal(s, model.decision_function(self.X))
            self.assert_positive_orientation(model, self.X, self.y, 1)

    def test_predict_proba_is_expit_of_decision_function(self):
        for model in (self.lr, self.hgb):
            s = model.decision_function(self.X)
            np.testing.assert_allclose(model.predict_proba(self.X)[:, 1], expit(s), rtol=1e-15, atol=0)

    def test_positive_label_sorting_first_is_negated_correctly(self):
        labels = np.where(self.y == 1, "a_fraud", "b_legit")  # positive sorts first: classes_[0]
        model = LogisticRegression(max_iter=1000).fit(self.X, labels)
        np.testing.assert_array_equal(model.classes_, ["a_fraud", "b_legit"])
        s = ranking_scores(model, self.X, positive_label="a_fraud")
        np.testing.assert_array_equal(s, -model.decision_function(self.X))
        self.assert_positive_orientation(model, self.X, labels, "a_fraud")

    def _calibrate(self, X_cal, y_cal):
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.frozen import FrozenEstimator

        calibrated = CalibratedClassifierCV(FrozenEstimator(self.lr), method="sigmoid").fit(X_cal, y_cal)
        self.assertEqual(len(calibrated.calibrated_classifiers_), 1)  # FrozenEstimator: no CV ensemble
        sigmoid = calibrated.calibrated_classifiers_[0].calibrators[0]
        return calibrated, sigmoid

    def test_platt_sign_convention_on_fitted_calibrator(self):
        calibrated, sigmoid = self._calibrate(self.X_cal, self.y_cal)
        A, B = protocol_platt_parameters(sigmoid.a_, sigmoid.b_)
        self.assertLess(sigmoid.a_, 0)  # sklearn slope negative for a positive association
        self.assertGreater(A, 0)        # protocol slope positive
        s = ranking_scores(self.lr, self.X)
        p_cal = calibrated.predict_proba(self.X)[:, 1]
        # The calibrator consumed decision_function: p_cal == expit(A * s + B).
        np.testing.assert_allclose(p_cal, expit(calibrated_logits(s, A, B)), rtol=1e-12, atol=0)
        # A positive slope preserves the ranking of the scores.
        order = np.argsort(s, kind="stable")
        self.assertTrue(np.all(np.diff(p_cal[order]) >= 0))
        self.assertFalse(is_numerically_constant(calibrated_logits(s, A, B)))

    def test_reversed_association_gives_non_positive_protocol_slope(self):
        _, sigmoid = self._calibrate(self.X_cal_rev, self.y_cal_rev)
        A, _ = protocol_platt_parameters(sigmoid.a_, sigmoid.b_)
        self.assertGreater(sigmoid.a_, 0)
        self.assertLessEqual(A, 0)  # would be a technical calibration failure (protocol Section 6.4)


if __name__ == "__main__":
    unittest.main()
