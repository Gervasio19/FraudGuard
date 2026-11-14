"""Tests for criterion E2, the selection-aware held-out block splits (Phase 5B-3). Synthetic data only."""

from __future__ import annotations

import unittest

import numpy as np

from src.data.split import VALIDATION_END_TIME
from src.data.validation_blocks import block_edges, draw_selection_blocks, heldout_keep_mask
from src.evaluation.e2_stability import FREQUENCY_NAME, NOT_ESTIMABLE, e2_gate, e2_selection_aware
from src.models.phase5_grid import HGB, LR
from src.models.phase5_settings import (
    E2_MIN_USABLE_SPLITS, E2_N_SPLITS, E2_SPLIT_SEED, E2_STABILITY_THRESHOLD, VALIDATION_START_TIME,
)

CONFIGS = [
    {"config_id": "LR-01", "family": LR, "params": {"C": 0.1, "class_weight": None}},
    {"config_id": "LR-02", "family": LR, "params": {"C": 1.0, "class_weight": None}},
    {"config_id": "HGB-01", "family": HGB,
     "params": {"learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 20, "class_weight": None}},
    {"config_id": "HGB-02", "family": HGB,
     "params": {"learning_rate": 0.1, "max_leaf_nodes": 31, "min_samples_leaf": 20, "class_weight": None}},
]


def period(n=4_000, prevalence=0.03, seed=0):
    rng = np.random.default_rng(seed)
    times = np.sort(rng.integers(VALIDATION_START_TIME, VALIDATION_END_TIME + 1, size=n))
    y = rng.random(n) < prevalence
    return rng, times, y


def scores(rng, y, strength):
    return rng.normal(size=len(y)) + strength * y


class TestGate(unittest.TestCase):
    def test_frozen_constants(self):
        self.assertEqual((E2_N_SPLITS, E2_MIN_USABLE_SPLITS, E2_STABILITY_THRESHOLD, E2_SPLIT_SEED),
                         (2_000, 1_000, 0.90, 42))

    def test_boundaries(self):
        self.assertTrue(e2_gate(1_000, 0.90, 1e-9, 1_000)["passed"])  # exactly 90% passes (>=)
        self.assertFalse(e2_gate(1_000, 0.8999, 0.01, 1_000)["passed"])
        self.assertFalse(e2_gate(1_984, 0.95, 0.0, 1_000)["passed"])  # median must be strictly > 0
        self.assertFalse(e2_gate(1_984, 0.95, -0.001, 1_000)["passed"])
        gate = e2_gate(1_984, 0.95, 0.01, 1_000)
        self.assertTrue(gate["frequency_pass"] and gate["median_pass"] and gate["estimable"])

    def test_not_estimable_below_the_minimum_usable_splits(self):
        gate = e2_gate(999, None, None, 1_000)
        self.assertEqual((gate["estimable"], gate["passed"], gate["status"]), (False, False, NOT_ESTIMABLE))
        self.assertNotIn("frequency", gate)
        self.assertTrue(e2_gate(1_000, 0.95, 0.01, 1_000)["estimable"])


class TestSelectionAware(unittest.TestCase):
    def run_e2(self, times, y, by_config, **kwargs):
        return e2_selection_aware(times, y, by_config, CONFIGS, "LR-02", True, **kwargs)

    def test_clear_challenger_advantage_passes(self):
        rng, times, y = period()
        by_config = {"LR-01": scores(rng, y, 0.5), "LR-02": scores(rng, y, 1.0),
                     "HGB-01": scores(rng, y, 3.0), "HGB-02": scores(rng, y, 6.0)}
        out = self.run_e2(times, y, by_config)
        self.assertEqual(out["splits"]["n_attempted"], 2_000)
        self.assertTrue(out["gate"]["passed"])
        self.assertGreaterEqual(out["gate"]["frequency"], 0.90)
        self.assertGreater(out["heldout_delta_ap"]["median"], 0)
        self.assertEqual(out["frequency_name"], FREQUENCY_NAME)
        self.assertIn("NOT a probability", out["frequency_meaning"])
        for family in (LR, HGB):
            self.assertAlmostEqual(sum(out["selection_frequency"][family].values()), 1.0)
        self.assertEqual(max(out["selection_frequency"][HGB], key=out["selection_frequency"][HGB].get), "HGB-02")
        self.assertIn("NOT an unbiased estimate", out["optimism_indicator"]["label"])

    def test_no_advantage_fails_and_incumbent_is_never_displaced(self):
        rng, times, y = period(seed=1)
        by_config = {"LR-01": scores(rng, y, 4.0), "LR-02": scores(rng, y, 4.0),
                     "HGB-01": scores(rng, y, 1.0), "HGB-02": scores(rng, y, 1.0)}
        out = self.run_e2(times, y, by_config)
        self.assertFalse(out["gate"]["passed"])
        self.assertLess(out["gate"]["frequency"], 0.5)

    def test_identical_scores_give_zero_difference_which_is_not_positive(self):
        rng, times, y = period(seed=2)
        shared = scores(rng, y, 2.0)
        out = self.run_e2(times, y, {c["config_id"]: shared for c in CONFIGS})
        self.assertEqual(out["heldout_delta_ap"]["share_exactly_zero"], 1.0)
        self.assertEqual(out["gate"]["frequency"], 0.0)  # delta AP = 0 is not positive (D7)
        self.assertFalse(out["gate"]["passed"])

    def test_reselection_happens_on_the_selection_half_only(self):
        rng, times, y = period(seed=3)
        edges = block_edges(VALIDATION_START_TIME, VALIDATION_END_TIME)
        early = times < edges[10]
        # HGB-01 ranks well only in the early blocks, HGB-02 only in the late ones
        by_config = {"LR-01": scores(rng, y, 1.0), "LR-02": scores(rng, y, 1.0),
                     "HGB-01": rng.normal(size=len(y)) + 5.0 * (y & early),
                     "HGB-02": rng.normal(size=len(y)) + 5.0 * (y & ~early)}
        out = self.run_e2(times, y, by_config, n_splits=300, min_usable=50)
        frequency = out["selection_frequency"][HGB]
        self.assertEqual(set(frequency), {"HGB-01", "HGB-02"})  # the choice follows the selection half
        self.assertGreater(min(frequency.values()), 0.1)
        # a configuration picked where it is strong does worse on the held-out half
        self.assertGreater(out["optimism_indicator"][HGB]["unadjusted_mean_selection_minus_heldout_ap"], 0)

    def test_heldout_labels_do_not_influence_the_selection(self):
        rng, times, y = period(seed=4)
        by_config = {c["config_id"]: scores(rng, y, 1.0 + i) for i, c in enumerate(CONFIGS)}
        base = self.run_e2(times, y, by_config, n_splits=40, min_usable=5)
        # flip labels only on rows that are held out in the first split: its selection must not change,
        # while its held-out result does
        edges = block_edges(VALIDATION_START_TIME, VALIDATION_END_TIME)
        first = draw_selection_blocks(1, 21, 10, E2_SPLIT_SEED)[0]
        keep = heldout_keep_mask(times, edges, first)
        tampered = y.copy()
        tampered[keep] = ~tampered[keep]
        one = self.run_e2(times, y, by_config, n_splits=1, min_usable=1)
        other = self.run_e2(times, tampered, by_config, n_splits=1, min_usable=1)
        self.assertEqual(one["selection_frequency"], other["selection_frequency"])
        self.assertNotEqual(one["heldout_delta_ap"]["median"], other["heldout_delta_ap"]["median"])
        self.assertEqual(base["splits"]["n_attempted"], 40)

    def test_not_estimable_computes_nothing(self):
        rng, times, y = period(n=3_000, prevalence=0.004, seed=5)  # about 12 frauds: held-out halves < 10
        by_config = {c["config_id"]: scores(rng, y, 2.0) for c in CONFIGS}
        out = self.run_e2(times, y, by_config)
        self.assertLess(out["splits"]["n_usable"], 1_000)
        self.assertEqual(out["gate"]["status"], NOT_ESTIMABLE)
        self.assertFalse(out["gate"]["passed"])
        for key in ("heldout_delta_ap", "selection_frequency", "optimism_indicator"):
            self.assertNotIn(key, out)

    def test_selection_half_without_fraud_is_refused_not_guessed(self):
        rng, times, _ = period(seed=6)
        edges = block_edges(VALIDATION_START_TIME, VALIDATION_END_TIME)
        y = (times >= edges[5] + 300) & (times < edges[6] - 300)  # every fraud in the middle of block 5
        y &= rng.random(len(times)) < 0.5
        self.assertGreaterEqual(int(y.sum()), 10)
        by_config = {c["config_id"]: scores(rng, y, 2.0) for c in CONFIGS}
        with self.assertRaises(RuntimeError):
            self.run_e2(times, y, by_config, n_splits=50, min_usable=1)

    def test_input_validation(self):
        rng, times, y = period(seed=7)
        by_config = {c["config_id"]: scores(rng, y, 2.0) for c in CONFIGS}
        with self.assertRaises(ValueError):
            e2_selection_aware(times, y, by_config, CONFIGS, "LR-99", True)
        with self.assertRaises(ValueError):
            e2_selection_aware(times, y, {"LR-01": by_config["LR-01"]}, CONFIGS, "LR-01", True)


if __name__ == "__main__":
    unittest.main()
