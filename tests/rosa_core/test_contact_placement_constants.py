"""Sanity tests for ``rosa_core.contact_placement.constants``.

The point isn't to verify arithmetic — it's to pin the calibrated values so
an accidental edit (e.g. a global search-replace) trips a fast unit test
before someone has to discover it via a multi-subject regression run.
"""
from __future__ import annotations

import unittest

from rosa_core.contact_placement import constants as c


class ConstantsPinTests(unittest.TestCase):
    """Pin the 2026-05-09 calibrated values."""

    def test_walker_disk_geometry(self):
        self.assertEqual(c.WALK_N_RADII, 3)
        self.assertEqual(c.WALK_N_ANGLES, 12)
        self.assertEqual(c.WALK_DISK_RADIUS_MM, 1.0)
        self.assertEqual(c.WALK_STEP_MM, 0.25)
        self.assertEqual(c.WALK_TIP_PAD_MM, 3.0)
        self.assertEqual(c.WALK_AGGREGATOR, "p90")

    def test_compound_weights_sum_to_one(self):
        # The weighted sum is clipped at the end, but the design intent is
        # weights ≈ 1.0 (with bolt_only_penalty subtracted post-sum).
        s = sum(c.COMPOUND_WEIGHTS.values())
        self.assertAlmostEqual(s, 1.00, places=2)

    def test_compound_weights_keys(self):
        # Score components match what stage_f_score writes.
        expected = {"corr", "fft", "tube", "margin", "walker", "cc_overlap", "seeder"}
        self.assertEqual(set(c.COMPOUND_WEIGHTS.keys()), expected)

    def test_band_thresholds(self):
        self.assertEqual(c.COMPOUND_BANDS["high"], 0.70)
        self.assertEqual(c.COMPOUND_BANDS["medium"], 0.45)

    def test_seeder_label_mapping(self):
        self.assertEqual(c.SEEDER_LABEL_TO_SCORE["high"], 1.0)
        self.assertEqual(c.SEEDER_LABEL_TO_SCORE["medium"], 0.6)
        self.assertEqual(c.SEEDER_LABEL_TO_SCORE["low"], 0.3)
        self.assertEqual(c.SEEDER_LABEL_TO_SCORE[""], 0.5)

    def test_validator_thresholds(self):
        self.assertEqual(c.MIN_CORR_FOR_REAL_SHANK, 0.35)
        self.assertEqual(c.MIN_SLOT_HU_MEAN, 1500.0)
        self.assertEqual(c.MAX_SLOT_CC_VOLUME_P90_MM3, 150.0)

    def test_degenerate_zone_threshold(self):
        self.assertEqual(c.DEGENERATE_CONTACT_ZONE_MM, 5.0)


if __name__ == "__main__":
    unittest.main()
