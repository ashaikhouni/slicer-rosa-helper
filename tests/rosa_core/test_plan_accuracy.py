"""``rosa_core.plan_accuracy`` — plan-vs-actual trajectory accuracy from the case
TSVs, reusing rosa_core.qc. Locks: exact entry/target errors, the SEEG bolt↔tip
axis-flip resolution, and the "no plan → None" contract.
"""
from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "CommonLib"))


def _deps() -> bool:
    try:
        from rosa_core.plan_accuracy import compute_plan_accuracy  # noqa: F401
        return True
    except Exception:
        return False


DEPS = _deps()


def _tsv(path: Path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader(); w.writerows(rows)


TCOL = ["name", "start_x", "start_y", "start_z", "end_x", "end_y", "end_z"]
CCOL = ["trajectory", "x", "y", "z"]


@unittest.skipUnless(DEPS, "rosa_core.plan_accuracy not importable.")
class PlanAccuracyTests(unittest.TestCase):
    def _case(self, root: Path, *, flip_plan_B=False):
        plan, traj, cont = root / "ros_plan.tsv", root / "trajectories.tsv", root / "contacts.tsv"
        # A: plan entry (0,0,0)→target (0,0,30). B: plan (10,0,0)→(10,0,30).
        pB = ["B", 10, 0, 30, 10, 0, 0] if flip_plan_B else ["B", 10, 0, 0, 10, 0, 30]
        _tsv(plan, TCOL, [
            dict(zip(TCOL, ["A", 0, 0, 0, 0, 0, 30])),
            dict(zip(TCOL, pB)),
        ])
        # actual A: entry off by 3mm (x), tip off by 5mm (x); B: exact.
        _tsv(traj, TCOL, [
            dict(zip(TCOL, ["A", 3, 0, 0, 5, 0, 30])),
            dict(zip(TCOL, ["B", 10, 0, 0, 10, 0, 30])),
        ])
        _tsv(cont, CCOL, [
            {"trajectory": "A", "x": 4, "y": 0, "z": 10},   # ~1mm off the plan line (x=0)... actually 4mm
            {"trajectory": "A", "x": 4, "y": 0, "z": 20},
            {"trajectory": "B", "x": 10, "y": 0, "z": 10},
        ])
        return plan, traj, cont

    def test_errors_and_summary(self):
        from rosa_core.plan_accuracy import compute_plan_accuracy
        root = Path(tempfile.mkdtemp())
        plan, traj, cont = self._case(root)
        res = compute_plan_accuracy(plan, traj, cont)
        self.assertIsNotNone(res)
        by = {r["trajectory"]: r for r in res["rows"]}
        self.assertAlmostEqual(by["A"]["entry_error_mm"], 3.0, places=5)   # |(3,0,0)-(0,0,0)|
        self.assertAlmostEqual(by["A"]["target_error_mm"], 5.0, places=5)  # |(5,0,30)-(0,0,30)|
        self.assertAlmostEqual(by["B"]["target_error_mm"], 0.0, places=5)
        self.assertEqual(res["n_matched"], 2)
        # headline = median target error over shanks (0 and 5 → 2.5)
        self.assertAlmostEqual(res["headline_tpe_mm"], 2.5, places=5)

    def test_axis_flip_resolved(self):
        from rosa_core.plan_accuracy import compute_plan_accuracy
        root = Path(tempfile.mkdtemp())
        plan, traj, cont = self._case(root, flip_plan_B=True)   # plan B stored tip→bolt
        res = compute_plan_accuracy(plan, traj, cont)
        by = {r["trajectory"]: r for r in res["rows"]}
        # Without flip resolution B would read ~30mm entry/target error; resolved → ~0.
        self.assertAlmostEqual(by["B"]["entry_error_mm"], 0.0, places=5)
        self.assertAlmostEqual(by["B"]["target_error_mm"], 0.0, places=5)

    def test_no_plan_returns_none(self):
        from rosa_core.plan_accuracy import compute_plan_accuracy
        root = Path(tempfile.mkdtemp())
        _tsv(root / "trajectories.tsv", TCOL, [dict(zip(TCOL, ["A", 0, 0, 0, 0, 0, 30]))])
        self.assertIsNone(compute_plan_accuracy(
            root / "ros_plan.tsv", root / "trajectories.tsv", root / "contacts.tsv"))


if __name__ == "__main__":
    unittest.main()
