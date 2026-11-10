"""Tests for the frozen Phase 5 settings and seed handling (Phase 5B-0).

Synthetic only: nothing here reads the dataset, fits a model or scores anything.
"""

from __future__ import annotations

import json
import unittest

import numpy as np

from src.data import split
from src.data.validation_blocks import block_edges, draw_selection_blocks
from src.models import phase5_settings as s


class TestFrozenSeeds(unittest.TestCase):
    def test_approved_seed_values(self):
        self.assertEqual(s.E2_SPLIT_SEED, 42)
        self.assertEqual(s.TCAL_BOOTSTRAP_SEED, 42)
        self.assertEqual(s.CONDITIONAL_BOOTSTRAP_SEED, 0)
        self.assertEqual(s.MODEL_RANDOM_STATE, 42)
        self.assertEqual(s.OTHER_BOOTSTRAP_SEED, 0)

    def test_seeds_exposed_in_configuration(self):
        seeds = s.pre_registered_configuration()["seeds"]
        self.assertEqual(seeds["e2_split_seed"], 42)
        self.assertEqual(seeds["tcal_bootstrap_seed"], 42)
        self.assertEqual(seeds["conditional_bootstrap_seed"], 0)
        self.assertEqual(seeds["generator"], "numpy.random.default_rng")


class TestConfiguration(unittest.TestCase):
    def test_strict_json_serializable(self):
        text = json.dumps(s.pre_registered_configuration(), allow_nan=False)
        self.assertEqual(json.loads(text), s.pre_registered_configuration())

    def test_returns_independent_copies(self):
        first = s.pre_registered_configuration()
        first["seeds"]["e2_split_seed"] = -1
        self.assertEqual(s.pre_registered_configuration()["seeds"]["e2_split_seed"], 42)

    def test_grid_sizes_match_protocol(self):
        models = s.pre_registered_configuration()["models"]
        for family, expected in (("logistic_regression", 12), ("hist_gradient_boosting", 16)):
            grid = models[family]["grid"]
            size = int(np.prod([len(v) for v in grid.values()]))
            self.assertEqual(size, expected)
            self.assertEqual(models[family]["n_configurations"], expected)
        self.assertFalse(models["hist_gradient_boosting"]["fixed"]["early_stopping"])

    def test_partition_boundaries_consistent_with_phase4_split(self):
        self.assertEqual(s.T_CAL_END_TIME, split.TRAIN_END_TIME)
        self.assertEqual(s.VALIDATION_START_TIME, split.TRAIN_END_TIME + 1)
        self.assertLess(s.T_FIT_END_TIME, s.T_CAL_START_TIME)
        self.assertLess(s.VALIDATION_START_TIME, s.WINDOW_1_END_TIME)
        self.assertLess(s.WINDOW_1_END_TIME, split.VALIDATION_END_TIME)

    def test_e2_block_count_matches_validation_period(self):
        edges = block_edges(s.VALIDATION_START_TIME, split.VALIDATION_END_TIME, s.E2_BLOCK_SECONDS)
        self.assertEqual(len(edges) - 1, s.E2_N_BLOCKS)
        self.assertEqual(int(edges[-1] - edges[-2]), 400)

    def test_scope_decisions_recorded(self):
        config = s.pre_registered_configuration()
        self.assertEqual(config["ranking_score"]["method"], "decision_function")
        self.assertEqual(config["calibration"]["input"], "decision_function")
        self.assertIn("not required", config["calibration"]["amount_tercile_calibration"])
        self.assertIn("never a fallback", config["calibration"]["c_f"])
        self.assertEqual(config["champion_rule"]["default"], "logistic_regression")
        self.assertEqual(config["uncertainty"]["e2_heldout_splits"]["min_usable_splits"], 1000)


class TestReliabilityBinSettings(unittest.TestCase):
    def test_edges_cover_unit_interval_with_lower_tail_bin(self):
        edges = s.RELIABILITY_BIN_EDGES
        self.assertEqual(edges, (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0))
        self.assertEqual(edges[0], 0.0)
        self.assertEqual(edges[-1], 1.0)
        self.assertTrue(all(a < b for a, b in zip(edges, edges[1:])))
        self.assertEqual(len(edges) - 1, 7)

    def test_configuration_records_edges_and_closed_last_bin(self):
        bins = s.pre_registered_configuration()["calibration"]["reliability_bins"]
        self.assertEqual(bins["edges"], list(s.RELIABILITY_BIN_EDGES))
        self.assertIn("[0.1, 1.0]", bins["rule"])
        self.assertIn("[0, 1]", bins["covers"])
        self.assertNotIn("reliability_decade_edges", s.pre_registered_configuration()["calibration"])


class TestSplitDrawDeterminism(unittest.TestCase):
    def test_same_seed_same_draws(self):
        a = draw_selection_blocks(50, 21, 10, s.E2_SPLIT_SEED)
        b = draw_selection_blocks(50, 21, 10, s.E2_SPLIT_SEED)
        np.testing.assert_array_equal(a, b)

    def test_different_seed_different_draws(self):
        a = draw_selection_blocks(50, 21, 10, 42)
        b = draw_selection_blocks(50, 21, 10, 43)
        self.assertFalse(np.array_equal(a, b))

    def test_shape_sorted_unique_in_range(self):
        draws = draw_selection_blocks(200, 21, 10, s.E2_SPLIT_SEED)
        self.assertEqual(draws.shape, (200, 10))
        for row in draws:
            self.assertTrue(np.all(np.diff(row) > 0))
            self.assertTrue(row.min() >= 0 and row.max() <= 20)

    def test_prefix_stable(self):
        """Fewer splits are a prefix of more splits (sequential single generator)."""
        np.testing.assert_array_equal(draw_selection_blocks(10, 21, 10, 42),
                                      draw_selection_blocks(30, 21, 10, 42)[:10])

    @unittest.skipUnless(np.__version__ == "2.3.2", "snapshot recorded with NumPy 2.3.2")
    def test_snapshot_with_verified_numpy(self):
        """Regression guard: the frozen seed gives these first draws under NumPy 2.3.2."""
        expected = _SNAPSHOT_FIRST_THREE
        np.testing.assert_array_equal(draw_selection_blocks(3, 21, 10, 42), np.array(expected))

    def test_invalid_arguments(self):
        for args in ((0, 21, 10, 42), (5, 21, 0, 42), (5, 21, 21, 42)):
            with self.assertRaises(ValueError):
                draw_selection_blocks(*args)


#: First three selection halves for seed 42 (NumPy 2.3.2). Block indices only;
#: computed without labels or scores during Phase 5B-0.
_SNAPSHOT_FIRST_THREE = [
    [1, 4, 6, 9, 10, 13, 14, 15, 17, 20],
    [2, 5, 6, 7, 11, 13, 14, 16, 19, 20],
    [1, 2, 3, 6, 8, 11, 15, 18, 19, 20],
]


if __name__ == "__main__":
    unittest.main()
