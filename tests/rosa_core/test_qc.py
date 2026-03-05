import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))

from rosa_core.qc import compute_qc_metrics, sorted_contacts_by_trajectory  # noqa: E402


class QCTests(unittest.TestCase):
    def test_sorted_contacts_by_trajectory(self):
        contacts = [
            {"trajectory": "RHH", "index": 3},
            {"trajectory": "RHH", "index": 1},
            {"trajectory": "LHH", "index": 2},
        ]
        by_traj = sorted_contacts_by_trajectory(contacts)
        self.assertEqual([c["index"] for c in by_traj["RHH"]], [1, 3])
        self.assertEqual([c["index"] for c in by_traj["LHH"]], [2])

    def test_compute_qc_metrics(self):
        planned_traj = {"RHH": {"start": [0, 0, 0], "end": [10, 0, 0]}}
        final_traj = {"RHH": {"start": [0, 1, 0], "end": [10, 1, 0]}}
        planned_contacts = [
            {"trajectory": "RHH", "index": 1, "position_lps": [2, 0, 0]},
            {"trajectory": "RHH", "index": 2, "position_lps": [5, 0, 0]},
            {"trajectory": "RHH", "index": 3, "position_lps": [8, 0, 0]},
        ]
        final_contacts = [
            {"trajectory": "RHH", "index": 1, "position_lps": [2, 2, 0]},
            {"trajectory": "RHH", "index": 2, "position_lps": [5, 2, 0]},
            {"trajectory": "RHH", "index": 3, "position_lps": [8, 2, 0]},
        ]
        rows = compute_qc_metrics(planned_traj, final_traj, planned_contacts, final_contacts)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["trajectory"], "RHH")
        self.assertAlmostEqual(row["entry_radial_mm"], 1.0, places=6)
        self.assertAlmostEqual(row["target_radial_mm"], 1.0, places=6)
        self.assertAlmostEqual(row["mean_contact_radial_mm"], 2.0, places=6)
        self.assertAlmostEqual(row["max_contact_radial_mm"], 2.0, places=6)
        self.assertAlmostEqual(row["rms_contact_radial_mm"], 2.0, places=6)
        self.assertAlmostEqual(row["angle_deg"], 0.0, places=6)
        self.assertEqual(row["matched_contacts"], 3)

    def test_compute_qc_metrics_includes_unmatched_planned(self):
        planned_traj = {
            "RHH": {"start": [0, 0, 0], "end": [10, 0, 0]},
            "LHH": {"start": [0, 0, 0], "end": [0, 10, 0]},
        }
        final_traj = {"RHH": {"start": [0, 1, 0], "end": [10, 1, 0]}}
        planned_contacts = [
            {"trajectory": "RHH", "index": 1, "position_lps": [2, 0, 0]},
            {"trajectory": "RHH", "index": 2, "position_lps": [5, 0, 0]},
            {"trajectory": "LHH", "index": 1, "position_lps": [0, 2, 0]},
        ]
        final_contacts = [
            {"trajectory": "RHH", "index": 1, "position_lps": [2, 1, 0]},
            {"trajectory": "RHH", "index": 2, "position_lps": [5, 1, 0]},
        ]
        rows = compute_qc_metrics(
            planned_traj,
            final_traj,
            planned_contacts,
            final_contacts,
            include_unmatched_planned=True,
        )
        self.assertEqual(len(rows), 2)
        by_name = {row["trajectory"]: row for row in rows}
        self.assertIn("LHH", by_name)
        self.assertIsNone(by_name["LHH"]["entry_radial_mm"])
        self.assertIsNone(by_name["LHH"]["angle_deg"])
        self.assertEqual(by_name["LHH"]["matched_contacts"], 0)


if __name__ == "__main__":
    unittest.main()
