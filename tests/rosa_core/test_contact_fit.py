import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))

from rosa_core.contact_fit import (  # noqa: E402
    angle_deg,
    fit_electrode_axis_and_tip,
    point_line_distance,
    unit,
)


class ContactFitTests(unittest.TestCase):
    def test_unit_and_angle(self):
        u = unit([3, 0, 0])
        self.assertAlmostEqual(float(np.linalg.norm(u)), 1.0, places=7)
        self.assertAlmostEqual(angle_deg([1, 0, 0], [0, 1, 0]), 90.0, places=6)

    def test_point_line_distance(self):
        points = np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
        dist = point_line_distance(points, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        self.assertAlmostEqual(float(dist[0]), 1.0, places=6)
        self.assertAlmostEqual(float(dist[1]), 2.0, places=6)

    def test_fit_electrode_axis_and_tip_success(self):
        rng = np.random.default_rng(7)
        # Planned trajectory along +X from entry to target.
        planned_entry = np.array([0.0, 0.0, 0.0])
        planned_target = np.array([50.0, 0.0, 0.0])

        # Actual points are near a slightly shifted axis.
        x = rng.uniform(0.0, 50.0, size=2000)
        y = rng.normal(loc=0.5, scale=0.15, size=2000)
        z = rng.normal(loc=-0.2, scale=0.15, size=2000)
        pts = np.column_stack([x, y, z])

        result = fit_electrode_axis_and_tip(
            candidate_points_lps=pts,
            planned_entry_lps=planned_entry,
            planned_target_lps=planned_target,
            contact_offsets_mm=[0.0, 3.5, 7.0, 10.5],
            tip_at="target",
            roi_radius_mm=3.0,
            max_angle_deg=12.0,
            max_depth_shift_mm=20.0,
        )

        self.assertTrue(result["success"], msg=str(result))
        self.assertLess(result["angle_deg"], 3.0)
        self.assertGreater(result["points_in_roi"], 100)
        self.assertIn("entry_lps", result)
        self.assertIn("target_lps", result)

    def test_fit_electrode_axis_and_tip_failure_no_points(self):
        result = fit_electrode_axis_and_tip(
            candidate_points_lps=[],
            planned_entry_lps=[0.0, 0.0, 0.0],
            planned_target_lps=[10.0, 0.0, 0.0],
            contact_offsets_mm=[0.0, 3.5],
        )
        self.assertFalse(result["success"])
        self.assertIn("No candidate points", result["reason"])


if __name__ == "__main__":
    unittest.main()
