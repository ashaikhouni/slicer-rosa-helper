"""Smoke tests for rosa_core.contact_placement.bolt_end — the bolt-end /
metal-mass landmark estimators relocated 2026-05-30 out of the retired
contact_placement_legacy.py (which they were the only live survivors of).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))

try:
    import numpy as np
    from rosa_core.contact_placement.bolt_end import (
        entry_arc_from_metal_mass,
        estimate_bolt_end_from_metal_mass,
        median_library_pitch_mm,
    )
    HAVE_DEPS = True
except Exception:  # noqa: BLE001
    HAVE_DEPS = False


@unittest.skipUnless(HAVE_DEPS, "numpy / rosa_core unavailable")
class BoltEndTests(unittest.TestCase):
    def test_entry_arc_detects_mass_dropoff(self):
        """Bolt mass (high) → contacts (low): the bolt-end arc lands near the
        drop-off, not at the start or end."""
        arcs = np.arange(0.0, 30.0, 0.3)
        signal = np.where(arcs < 10.0, 2000.0, 100.0)
        be = entry_arc_from_metal_mass(arcs, signal, smooth_sigma_mm=3.5)
        self.assertIsNotNone(be)
        self.assertGreater(be, 6.0)
        self.assertLess(be, 14.0)

    def test_estimate_bolt_end_on_synthetic_bolt(self):
        """A bright proximal mass in a synthetic CT yields a finite bolt-end arc
        and a refined centerline."""
        ct = np.zeros((40, 40, 40), dtype=np.float32)
        ct[18:22, 18:22, 5:15] = 3000.0           # bright bolt mass, proximal
        feats = {"ct_arr_kji": ct, "ras_to_ijk_mat": np.eye(4)}
        out = estimate_bolt_end_from_metal_mass(
            np.array([20.0, 20.0, 2.0]), np.array([20.0, 20.0, 30.0]),
            features=feats,
        )
        self.assertIn("bolt_end_arc_mm", out)
        self.assertIsNotNone(out["bolt_end_arc_mm"])
        self.assertIsNotNone(out["centerline"])

    def test_median_library_pitch_none_without_library(self):
        self.assertIsNone(median_library_pitch_mm(None))


if __name__ == "__main__":
    unittest.main()
