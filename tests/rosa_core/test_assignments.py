import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))

from rosa_core.assignments import (  # noqa: E402
    electrode_length_mm,
    suggest_model_id_for_trajectory,
    trajectory_length_mm,
)


class AssignmentsTests(unittest.TestCase):
    def test_trajectory_and_electrode_length(self):
        traj = {"name": "R01", "start": [0, 0, 0], "end": [3, 4, 0]}
        model = {"total_exploration_length_mm": 61.5}
        self.assertAlmostEqual(trajectory_length_mm(traj), 5.0)
        self.assertAlmostEqual(electrode_length_mm(model), 61.5)

    def test_suggest_model_id_for_trajectory_prefers_closest(self):
        traj = {"name": "R01", "start": [0, 0, 0], "end": [50, 0, 0]}
        models = {
            "A": {"total_exploration_length_mm": 40.0, "contact_count": 8},
            "B": {"total_exploration_length_mm": 51.0, "contact_count": 15},
            "C": {"total_exploration_length_mm": 61.5, "contact_count": 18},
        }
        self.assertEqual(suggest_model_id_for_trajectory(traj, models, tolerance_mm=5.0), "B")

    def test_suggest_model_id_tie_breaks_by_contact_count_then_length(self):
        traj = {"name": "R01", "start": [0, 0, 0], "end": [40.5, 0, 0]}
        models = {
            "A": {"total_exploration_length_mm": 40.0, "contact_count": 8},
            "B": {"total_exploration_length_mm": 41.0, "contact_count": 12},
            "C": {"total_exploration_length_mm": 41.0, "contact_count": 12},
        }
        # B and C are tied on delta/contact/length; lexical ID decides.
        self.assertEqual(suggest_model_id_for_trajectory(traj, models, tolerance_mm=1.0), "B")

    def test_suggest_model_id_returns_empty_when_no_candidate_in_tolerance(self):
        traj = {"name": "R01", "start": [0, 0, 0], "end": [100, 0, 0]}
        models = {"A": {"total_exploration_length_mm": 40.0, "contact_count": 8}}
        self.assertEqual(suggest_model_id_for_trajectory(traj, models, tolerance_mm=5.0), "")


if __name__ == "__main__":
    unittest.main()
