"""Tests for the E2 block and buffer convention (protocol Addendum C, item C4).

Synthetic timestamps only; no dataset, labels, models or scores.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.data.split import TRAIN_END_TIME, VALIDATION_END_TIME
from src.data.validation_blocks import assign_blocks, block_edges, heldout_keep_mask

VAL_START, VAL_END = TRAIN_END_TIME + 1, VALIDATION_END_TIME  # 132,929 and 151,328


class TestBlockEdges(unittest.TestCase):
    def test_validation_partition_blocks(self):
        edges = block_edges(VAL_START, VAL_END, 900)
        self.assertEqual(len(edges), 22)  # 21 blocks
        self.assertEqual(int(edges[0]), 132_929)
        self.assertEqual(int(edges[-1]), 151_329)  # one second after the last timestamp
        self.assertTrue(np.all(np.diff(edges)[:-1] == 900))
        self.assertEqual(int(edges[-1] - edges[-2]), 400)  # final partial block
        self.assertEqual(int(edges[-2]), 150_929)

    def test_exact_multiple_has_no_empty_block(self):
        edges = block_edges(0, 1799, 900)
        np.testing.assert_array_equal(edges, [0, 900, 1800])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            block_edges(10, 9, 900)
        with self.assertRaises(ValueError):
            block_edges(0, 100, 0)
        with self.assertRaises(ValueError):
            block_edges(0.5, 100, 900)


class TestAssignBlocks(unittest.TestCase):
    def setUp(self):
        self.edges = block_edges(VAL_START, VAL_END, 900)

    def test_edges_are_half_open(self):
        times = [132_929, 133_828, 133_829, 150_928, 150_929, 151_328]
        np.testing.assert_array_equal(assign_blocks(times, self.edges), [0, 0, 1, 19, 20, 20])

    def test_out_of_period_and_fractional_rejected(self):
        for bad in ([132_928], [151_329], [133_000.5], [np.nan]):
            with self.assertRaises(ValueError):
                assign_blocks(bad, self.edges)

    def test_equal_timestamps_share_a_block(self):
        times = np.repeat([132_929, 140_000, 151_328], 4)
        blocks = assign_blocks(times, self.edges)
        for i in range(0, 12, 4):
            self.assertEqual(len(set(blocks[i:i + 4])), 1)


class TestBufferConvention(unittest.TestCase):
    """Toy period [0, 3000): blocks [0,900) [900,1800) [1800,2700) [2700,3000) (partial)."""

    def setUp(self):
        self.edges = block_edges(0, 2999, 900)
        self.times = np.arange(0, 3000)

    def keep(self, selection, buffer=300):
        return heldout_keep_mask(self.times, self.edges, selection, buffer)

    def test_boundary_seconds_around_one_selection_block(self):
        keep = self.keep([1])  # selection block [900, 1800)
        self.assertTrue(keep[599])    # 301 s before the block start: kept
        self.assertFalse(keep[600])   # a - 300: inside the buffer (inclusive start)
        self.assertFalse(keep[899])   # last second before the block: buffered
        self.assertFalse(keep[900])   # selection row: never held out
        self.assertFalse(keep[1799])  # selection row
        self.assertFalse(keep[1800])  # b: buffered (inclusive start of the right zone)
        self.assertFalse(keep[2099])  # b + 299: buffered
        self.assertTrue(keep[2100])   # b + 300: kept (zone is half-open)

    def test_exactly_buffer_seconds_removed_per_side(self):
        keep = self.keep([1])
        heldout_seconds = 3000 - 900
        self.assertEqual(int(keep.sum()), heldout_seconds - 2 * 300)

    def test_minimum_separation_is_buffer_plus_one(self):
        rng = np.random.default_rng(0)
        for _ in range(25):
            selection = np.sort(rng.choice(4, size=2, replace=False))
            keep = self.keep(selection)
            selection_times = self.times[np.isin(assign_blocks(self.times, self.edges), selection)]
            kept = self.times[keep]
            if kept.size:
                gaps = np.abs(kept[:, None] - selection_times[None, :]).min()
                self.assertGreaterEqual(int(gaps), 301)

    def test_kept_rows_never_overlap_selection(self):
        keep = self.keep([0, 2])
        blocks = assign_blocks(self.times, self.edges)
        self.assertFalse(np.any(keep & np.isin(blocks, [0, 2])))

    def test_heldout_block_between_two_selection_blocks(self):
        keep = self.keep([0, 2])  # held-out block 1 = [900, 1800) loses 300 s at each end
        self.assertEqual(int(keep[900:1800].sum()), 300)
        self.assertEqual(int(np.flatnonzero(keep[900:1800])[0]) + 900, 1200)

    def test_partial_final_block_fully_buffered_when_short(self):
        keep = self.keep([2])  # partial block [2700, 3000) is 300 s = exactly the buffer
        self.assertEqual(int(keep[2700:3000].sum()), 0)

    def test_no_buffer_beyond_outer_edges(self):
        keep = self.keep([3])  # selection = partial final block; only its left side buffers
        self.assertTrue(keep[:2400].all())
        self.assertFalse(keep[2400:].any())
        keep = self.keep([0])
        self.assertFalse(keep[:1200].any())
        self.assertTrue(keep[1200:].all())

    def test_zero_buffer_is_plain_block_membership(self):
        keep = self.keep([1], buffer=0)
        np.testing.assert_array_equal(keep, ~((self.times >= 900) & (self.times < 1800)))

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            self.keep([4])
        with self.assertRaises(ValueError):
            self.keep([1], buffer=-1)


class TestValidationPartialBlock(unittest.TestCase):
    """The real validation geometry: final block [150,929, 151,329) holds 400 s."""

    def setUp(self):
        self.edges = block_edges(VAL_START, VAL_END, 900)
        self.times = np.arange(150_000, VAL_END + 1)

    def test_partial_heldout_block_keeps_last_100_seconds(self):
        keep = heldout_keep_mask(self.times, self.edges, [19])
        partial = (self.times >= 150_929)
        kept = self.times[keep & partial]
        self.assertEqual(int(kept.min()), 151_229)
        self.assertEqual(int(kept.max()), 151_328)
        self.assertEqual(kept.size, 100)

    def test_partial_selection_block_buffers_only_its_left_side(self):
        keep = heldout_keep_mask(self.times, self.edges, [20])
        block_19 = (self.times >= 150_029) & (self.times < 150_929)
        self.assertEqual(int(keep[block_19].sum()), 600)
        self.assertFalse(keep[self.times >= 150_629].any())

    def test_equal_timestamps_get_identical_decisions(self):
        times = np.repeat([150_628, 150_629, 151_228, 151_229], 3)
        keep = heldout_keep_mask(times, self.edges, [20])
        for i in range(0, 12, 3):
            self.assertEqual(len(set(keep[i:i + 3])), 1)


if __name__ == "__main__":
    unittest.main()
