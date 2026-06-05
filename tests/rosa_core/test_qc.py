import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))

from rosa_core.qc import (  # noqa: E402
    compute_plan_vs_placement_qc,
    compute_qc_metrics,
    sorted_contacts_by_trajectory,
    summarize_plan_vs_placement_qc,
)


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


class PlanVsPlacementQCTests(unittest.TestCase):
    def test_basic_metrics(self):
        # Planned RHH along +x. Fitted entry off by 1 (y), tip off by 2 (y),
        # contacts at lateral 3 and 4 from the planned line.
        planned = {"RHH": {"start": [0, 0, 0], "end": [10, 0, 0]}}
        fitted = {
            "RHH": {
                "start": [0, 1, 0], "end": [10, 2, 0],
                "contacts": [[2, 3, 0], [5, 4, 0]],
            }
        }
        rows = compute_plan_vs_placement_qc(planned, fitted)
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["trajectory"], "RHH")
        self.assertAlmostEqual(r["entry_error_mm"], 1.0, places=6)
        self.assertAlmostEqual(r["target_error_mm"], 2.0, places=6)
        self.assertAlmostEqual(r["mean_contact_radial_mm"], 3.5, places=6)
        self.assertAlmostEqual(r["max_contact_radial_mm"], 4.0, places=6)
        self.assertAlmostEqual(r["rms_contact_radial_mm"], (12.5) ** 0.5, places=6)
        self.assertEqual(r["n_contacts"], 2)
        self.assertGreater(r["angle_deg"], 0.0)  # axes differ

    def test_contact_radial_is_distance_from_planned_line(self):
        # A perfectly on-axis fit (different tip depth) → zero lateral deviation,
        # confirming we measure distance from the LINE, not nominal contacts.
        planned = {"L": {"start": [0, 0, 0], "end": [20, 0, 0]}}
        fitted = {"L": {"start": [0, 0, 0], "end": [20, 0, 0],
                        "contacts": [[3, 0, 0], [9, 0, 0], [14, 0, 0]]}}
        r = compute_plan_vs_placement_qc(planned, fitted)[0]
        self.assertAlmostEqual(r["mean_contact_radial_mm"], 0.0, places=6)
        self.assertAlmostEqual(r["angle_deg"], 0.0, places=6)

    def test_unmatched_plan_gets_none(self):
        planned = {"A": {"start": [0, 0, 0], "end": [10, 0, 0]},
                   "B": {"start": [0, 0, 0], "end": [0, 10, 0]}}
        fitted = {"A": {"start": [0, 0, 0], "end": [10, 0, 0], "contacts": []}}
        rows = compute_plan_vs_placement_qc(planned, fitted)
        by = {r["trajectory"]: r for r in rows}
        self.assertIsNone(by["B"]["entry_error_mm"])
        self.assertEqual(by["B"]["n_contacts"], 0)
        # A has no contacts → contact stats None but entry/target present.
        self.assertIsNotNone(by["A"]["entry_error_mm"])
        self.assertIsNone(by["A"]["mean_contact_radial_mm"])

    def test_summary(self):
        rows = [
            {"entry_error_mm": 1.0, "target_error_mm": 2.0,
             "mean_contact_radial_mm": 0.5, "max_contact_radial_mm": 1.0, "angle_deg": 1.0},
            {"entry_error_mm": 3.0, "target_error_mm": 4.0,
             "mean_contact_radial_mm": 1.5, "max_contact_radial_mm": 2.0, "angle_deg": 2.0},
            {"entry_error_mm": None, "target_error_mm": None,
             "mean_contact_radial_mm": None, "max_contact_radial_mm": None, "angle_deg": None},
        ]
        s = summarize_plan_vs_placement_qc(rows)
        self.assertAlmostEqual(s["entry_error_mm"]["median"], 2.0, places=6)
        self.assertAlmostEqual(s["entry_error_mm"]["max"], 3.0, places=6)
        self.assertIsNotNone(s["angle_deg"]["p95"])


if __name__ == "__main__":
    unittest.main()
