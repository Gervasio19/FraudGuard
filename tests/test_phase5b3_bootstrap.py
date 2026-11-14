"""Tests for the block bootstrap, criterion E1 and the champion rule (Phase 5B-3). Synthetic data only."""

from __future__ import annotations

import inspect
import math
import unittest

import numpy as np
from sklearn.metrics import average_precision_score

from src.evaluation import block_bootstrap as bb
from src.evaluation.capacity import fixed_k_metrics
from src.models.phase5_grid import HGB, LR
from src.models.phase5b3_champion import champion_decision, e1_gate

T = 18_400  # validation period length in seconds


def scored_period(n=5_000, seed=0, round_to=None):
    rng = np.random.default_rng(seed)
    times = np.sort(rng.integers(0, T, size=n))
    y = rng.random(n) < 0.03
    a = rng.normal(size=n) + 2.0 * y
    b = rng.normal(size=n) + 1.2 * y
    if round_to is not None:
        a, b = np.round(a, round_to), np.round(b, round_to)
    return times, y, a, b


class TestReplicateWeights(unittest.TestCase):
    def test_block_lengths(self):
        lengths = bb.fixed_block_lengths(T, 900)
        self.assertEqual(len(lengths), 21)
        self.assertEqual(int(lengths[-1]), 400)
        self.assertEqual(int(lengths.sum()), T)
        self.assertEqual(bb.fixed_block_lengths(1_800, 900).tolist(), [900, 900])
        with self.assertRaises(ValueError):
            bb.fixed_block_lengths(T, 0)

    def test_second_counts_wrap_circularly(self):
        counts = bb.second_counts([8, 2], [4, 3], 10)
        self.assertEqual(counts.tolist(), [1, 1, 1, 1, 1, 0, 0, 0, 1, 1])
        self.assertEqual(bb.second_counts([3], [10], 10).tolist(), [1] * 10)
        with self.assertRaises(ValueError):
            bb.second_counts([10], [1], 10)
        with self.assertRaises(ValueError):
            bb.second_counts([0], [11], 10)

    def test_every_scheme_resamples_exactly_the_period_duration(self):
        offsets = np.arange(T)
        for scheme in bb.SCHEMES:
            for minutes in (5, 10, 15, 30):
                for w in bb.iter_replicate_weights(offsets, T, scheme, minutes * 60, 5, 0):
                    self.assertEqual(int(w.sum()), T, (scheme, minutes))

    def test_seeded_draws_are_pinned(self):
        # NumPy 2.3.2; one fresh default_rng(0) per analysis (Addendum C3)
        offsets = np.arange(T)
        expected = {"cbb": (5_940, 11_949, 100, 0), "mbb": (6_188, 11_365, 0, 200),
                    "stationary": (7_727, 11_720, 112, 0)}
        for scheme, (n_zero, first_max, head, middle) in expected.items():
            w = next(bb.iter_replicate_weights(offsets, T, scheme, 900, 2, 0))
            self.assertEqual((int((w == 0).sum()), int(np.flatnonzero(w == w.max())[0]),
                              int(w[:100].sum()), int(w[9_000:9_100].sum())),
                             (n_zero, first_max, head, middle), scheme)
        again = [w.copy() for w in bb.iter_replicate_weights(offsets, T, "stationary", 900, 3, 0)]
        for a, b in zip(again, bb.iter_replicate_weights(offsets, T, "stationary", 900, 3, 0)):
            np.testing.assert_array_equal(a, b)

    def test_circular_weights_are_uniform_and_moving_blocks_underweight_the_ends(self):
        offsets = np.arange(1_000)
        cbb = sum(bb.iter_replicate_weights(offsets, 1_000, "cbb", 100, 4_000, 0)) / 4_000
        mbb = sum(bb.iter_replicate_weights(offsets, 1_000, "mbb", 100, 4_000, 0)) / 4_000
        self.assertAlmostEqual(cbb[:20].mean(), 1.0, delta=0.05)
        self.assertAlmostEqual(cbb[490:510].mean(), 1.0, delta=0.05)
        self.assertLess(mbb[:20].mean(), 0.3)
        self.assertGreater(mbb[490:510].mean(), 1.0)

    def test_rows_sharing_a_timestamp_share_a_weight(self):
        offsets, period = bb.period_offsets([5, 5, 5, 9, 9], 0, 9)
        for w in bb.iter_replicate_weights(offsets, period, "cbb", 3, 20, 1):
            self.assertEqual(len(set(w[:3])), 1)
            self.assertEqual(len(set(w[3:])), 1)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            bb.period_offsets([1.5], 0, 10)
        with self.assertRaises(ValueError):
            bb.period_offsets([11], 0, 10)
        with self.assertRaises(ValueError):
            next(bb.iter_replicate_weights(np.arange(10), 10, "other", 3, 1, 0))
        with self.assertRaises(ValueError):
            next(bb.iter_replicate_weights(np.arange(10), 10, "mbb", 30, 1, 0))


class TestWeightedStatistics(unittest.TestCase):
    def test_weighted_ap_and_tp_equal_the_expanded_resample(self):
        times, y, a, _ = scored_period(round_to=1)  # rounding creates many tied scores
        positions = np.arange(len(a))
        ranked = bb.RankedScores(a, y, times, positions, True)
        self.assertAlmostEqual(ranked.average_precision(), average_precision_score(y, a), places=12)
        offsets, period = bb.period_offsets(times, 0, T - 1)
        for w in bb.iter_replicate_weights(offsets, period, "cbb", 900, 5, 3):
            rows = np.repeat(np.arange(len(a)), w)
            self.assertAlmostEqual(ranked.average_precision(w), average_precision_score(y[rows], a[rows]), places=12)
            expanded = [fixed_k_metrics(y[rows], a[rows], times[rows], np.arange(len(rows)), k, True)["tp"]
                        for k in (25, 50, 100)]
            self.assertEqual(ranked.tp_at_k((25, 50, 100), w).tolist(), expanded)

    def test_zero_fraud_sample_gives_nan_ap(self):
        times, y, a, _ = scored_period()
        ranked = bb.RankedScores(a, y, times, np.arange(len(a)), True)
        self.assertTrue(math.isnan(ranked.average_precision((~y).astype(float))))

    def test_k_larger_than_the_sample_is_refused(self):
        times, y, a, _ = scored_period(n=100)
        with self.assertRaises(ValueError):
            bb.RankedScores(a, y, times, np.arange(100), True).tp_at_k((500,))


class TestPairedBootstrap(unittest.TestCase):
    def setUp(self):
        self.times, self.y, a, b = scored_period(seed=4)
        positions = np.arange(len(a))
        self.ranked = {HGB: bb.RankedScores(a, self.y, self.times, positions, True),
                       LR: bb.RankedScores(b, self.y, self.times, positions, True)}

    def run_bootstrap(self, ranked=None, seed=0, scheme="cbb"):
        return bb.paired_block_bootstrap(ranked or self.ranked, self.y, self.times, 0, T - 1, scheme, 900,
                                         200, seed, (50, 100), True)

    def test_reproducible_and_seed_dependent(self):
        first, second, other = self.run_bootstrap(), self.run_bootstrap(), self.run_bootstrap(seed=1)
        np.testing.assert_array_equal(first["ap"][HGB], second["ap"][HGB])
        self.assertFalse(np.array_equal(first["ap"][HGB], other["ap"][HGB]))

    def test_models_share_every_replicate(self):
        same = {HGB: self.ranked[HGB], LR: self.ranked[HGB]}
        rep = self.run_bootstrap(same)
        np.testing.assert_array_equal(rep["ap"][HGB], rep["ap"][LR])  # identical rows -> delta is exactly 0
        np.testing.assert_array_equal(rep["tp_at_k"][HGB], rep["tp_at_k"][LR])
        summary = bb.interval_summary(rep["ap"][HGB] - rep["ap"][LR], 0.0)
        self.assertEqual((summary["lower"], summary["upper"]), (0.0, 0.0))
        self.assertEqual(summary["share_at_or_below_zero"], 1.0)

    def test_better_model_has_a_positive_interval(self):
        rep = self.run_bootstrap()
        observed = self.ranked[HGB].average_precision() - self.ranked[LR].average_precision()
        summary = bb.interval_summary(rep["ap"][HGB] - rep["ap"][LR], observed)
        self.assertGreater(summary["lower"], 0)
        self.assertLess(abs(summary["centring_bias_mean_minus_observed"]), 0.05)
        self.assertEqual(summary["n_dropped"], 0)

    def test_zero_fraud_replicates_are_dropped_and_counted(self):
        y = np.zeros(len(self.y), bool)
        y[(self.times >= 3_000) & (self.times < 3_200)] = True  # every fraud inside one short span
        positions = np.arange(len(y))
        scores = np.random.default_rng(0).normal(size=len(y)) + y
        ranked = {HGB: bb.RankedScores(scores, y, self.times, positions, True),
                  LR: bb.RankedScores(-scores, y, self.times, positions, True)}
        rep = bb.paired_block_bootstrap(ranked, y, self.times, 0, T - 1, "cbb", 900, 200, 0, (), True)
        n_zero = int((rep["n_frauds"] == 0).sum())
        self.assertGreater(n_zero, 20)
        summary = bb.interval_summary(rep["ap"][HGB] - rep["ap"][LR])
        self.assertEqual(summary["n_dropped"], n_zero)
        self.assertEqual(summary["n_valid"], 200 - n_zero)

    def test_interval_summary(self):
        summary = bb.interval_summary(np.arange(1, 101, dtype=float), 50.0)
        self.assertAlmostEqual(summary["lower"], np.percentile(np.arange(1, 101), 2.5))
        self.assertAlmostEqual(summary["upper"], np.percentile(np.arange(1, 101), 97.5))
        self.assertEqual(summary["share_at_or_below_zero"], 0.0)
        self.assertIn("not estimable", bb.interval_summary([np.nan, 1.0])["status"])

    def test_jackknife_and_iid_continuity(self):
        jack = bb.block_jackknife(self.ranked, (HGB, LR), self.times, 0, T - 1, 900)
        self.assertEqual(jack["n_blocks"], 21)
        self.assertGreater(jack["standard_error"], 0)
        iid = bb.stratified_iid_bootstrap(self.ranked, self.y, 50, 0, True)
        self.assertEqual(len(iid["ap"][HGB]), 50)
        self.assertTrue(np.isfinite(iid["ap"][HGB]).all())  # stratified: every replicate keeps the frauds

    def test_status_dependence(self):
        fraud_times = np.arange(0, T, 200)
        clustered = fraud_times < 3_000
        out = bb.status_dependence(fraud_times, clustered, 0, T - 1, (5, 15), 300, 0)
        self.assertLess(out["by_bin_minutes"]["15"]["permutation_p"], 0.05)
        self.assertGreater(out["by_bin_minutes"]["15"]["dispersion_index"],
                           out["by_bin_minutes"]["15"]["null_95th_percentile"])
        constant = bb.status_dependence(fraud_times, np.ones(len(fraud_times), bool), 0, T - 1, (5,), 10, 0)
        self.assertIn("not computable", constant["status"])


class TestGatesAndChampion(unittest.TestCase):
    def test_e1_requires_every_envelope_length_strictly_above_zero(self):
        self.assertTrue(e1_gate({5: -1.0, 10: 1e-12, 15: 0.01, 30: 0.02})["passed"])  # 5 min is not in the envelope
        self.assertFalse(e1_gate({10: 0.01, 15: 0.0, 30: 0.02})["passed"])  # exactly 0 is not above 0
        self.assertFalse(e1_gate({10: 0.01, 15: 0.01, 30: -1e-9})["passed"])
        self.assertFalse(e1_gate({10: 0.01, 15: None, 30: 0.02})["passed"])
        self.assertFalse(e1_gate({10: 0.01, 15: math.nan, 30: 0.02})["passed"])
        self.assertEqual(e1_gate({10: 1, 15: 1, 30: 1})["envelope_block_minutes"], [10, 15, 30])
        with self.assertRaises(ValueError):
            e1_gate({10: 0.01, 15: 0.01})

    def test_champion_truth_table(self):
        for e1_pass in (True, False):
            for e2_pass in (True, False):
                decision = champion_decision({"passed": e1_pass}, {"passed": e2_pass, "estimable": True})
                self.assertEqual(decision["champion"], HGB if (e1_pass and e2_pass) else LR)
                self.assertEqual(decision["challenger"], LR if (e1_pass and e2_pass) else HGB)
        retained = champion_decision({"passed": False}, {"passed": False, "estimable": True})
        self.assertIn("no sufficient evidence for replacement", retained["statement"])
        self.assertIn("not a finding that LR is better", retained["statement"])
        not_estimable = champion_decision({"passed": True}, {"passed": False, "estimable": False})
        self.assertEqual(not_estimable["champion"], LR)
        self.assertIn("E2 was not estimable", not_estimable["statement"])
        self.assertIn("optimistic development estimate",
                      champion_decision({"passed": True}, {"passed": True, "estimable": True})["statement"])

    def test_champion_rule_accepts_only_the_two_gates(self):
        # Calibration, Precision@K, business scenarios and test data cannot enter the decision.
        self.assertEqual(list(inspect.signature(champion_decision).parameters), ["e1", "e2"])
        decision = champion_decision({"passed": False, "calibration": "excellent", "net_saving": 1e9},
                                     {"passed": True, "estimable": True})
        self.assertEqual(decision["champion"], LR)
        self.assertIn("calibration results", decision["inputs_not_used"])
        self.assertIn("business scenarios", decision["inputs_not_used"])
        self.assertIn("test partition", decision["inputs_not_used"])


if __name__ == "__main__":
    unittest.main()
