"""``rosa_core.trajectory_fit`` — make a trajectory line carry its contacts.

Placement snaps contacts onto the metal, 1–2 mm off the detected seed line, so
the shaft reads as "slightly off" its own contacts in the 3-D scene. These tests
lock the invariant: the refit line is the PCA axis through the contacts (they lie
on it), oriented entry→tip, spanning both the bolt/entry extent and every
contact — and shanks with <2 contacts are left untouched.
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
        import numpy  # noqa: F401
        from rosa_core.trajectory_fit import fit_line_through_points  # noqa: F401
        return True
    except Exception:
        return False


DEPS = _deps()


def _dist_point_to_line(p, a, b) -> float:
    import numpy as np
    p, a, b = np.asarray(p, float), np.asarray(a, float), np.asarray(b, float)
    d = b - a
    d = d / np.linalg.norm(d)
    return float(np.linalg.norm((p - a) - np.dot(p - a, d) * d))


@unittest.skipUnless(DEPS, "numpy/rosa_core not importable.")
class FitLineThroughPointsTests(unittest.TestCase):
    def test_carries_contacts_offset_from_seed(self):
        """Contacts on a line at x=+2 (seed sits at x=0): the refit line must
        pass through the contacts, not the seed."""
        import numpy as np
        from rosa_core.trajectory_fit import fit_line_through_points
        contacts = [[2, 0, 5], [2, 0, 10], [2, 0, 15], [2, 0, 20]]
        seed_start, seed_end = [0, 0, 0], [0, 0, 30]
        start, end = fit_line_through_points(contacts, seed_start, seed_end)
        # Every contact now lies on the fitted line (was 2 mm off the seed).
        for c in contacts:
            self.assertLess(_dist_point_to_line(c, start, end), 1e-6)
            self.assertAlmostEqual(_dist_point_to_line(c, seed_start, seed_end), 2.0, places=6)
        # Line landed on the contact axis (x≈2), entry→tip orientation preserved.
        self.assertAlmostEqual(float(start[0]), 2.0, places=4)
        self.assertAlmostEqual(float(end[0]), 2.0, places=4)
        self.assertLess(float(start[2]), float(end[2]))

    def test_spans_both_bolt_and_contacts(self):
        """Endpoints reach the entry (bolt) extent AND cover the deepest contact
        even when the seed tip stops short."""
        import numpy as np
        from rosa_core.trajectory_fit import fit_line_through_points
        contacts = [[0, 0, 10], [0, 0, 14], [0, 0, 18], [0, 0, 25]]  # deepest at z=25
        start, end = fit_line_through_points(contacts, [0, 0, 0], [0, 0, 20])  # seed tip short (z=20)
        t = [float(np.asarray(c)[2]) for c in contacts]
        self.assertLessEqual(float(start[2]), min(t) + 1e-6)   # reaches entry (z≈0)
        self.assertGreaterEqual(float(end[2]), max(t) - 1e-6)  # covers deepest contact (z≥25)
        self.assertAlmostEqual(float(start[2]), 0.0, places=4)
        self.assertAlmostEqual(float(end[2]), 25.0, places=4)

    def test_orientation_follows_reference(self):
        """PCA sign is arbitrary; the fit must orient entry→tip like the seed."""
        from rosa_core.trajectory_fit import fit_line_through_points
        contacts = [[0, 0, 5], [0, 0, 10], [0, 0, 15]]
        # reference points the OTHER way (tip shallow, entry deep)
        start, end = fit_line_through_points(contacts, [0, 0, 30], [0, 0, 0])
        self.assertGreater(float(start[2]), float(end[2]))     # start deep, end shallow — matches ref

    def test_too_few_points_returns_reference(self):
        import numpy as np
        from rosa_core.trajectory_fit import fit_line_through_points
        for pts in ([], [[1, 2, 3]]):
            start, end = fit_line_through_points(pts, [0, 0, 0], [0, 0, 10])
            self.assertTrue(np.allclose(start, [0, 0, 0]))
            self.assertTrue(np.allclose(end, [0, 0, 10]))

    def test_outlier_contact_does_not_tilt_or_overshoot(self):
        """A single diverged / past-tip contact must NOT rotate the axis or drag
        an endpoint out to itself — the well-placed contacts stay on the line."""
        import numpy as np
        from rosa_core.trajectory_fit import fit_line_through_points
        good = [[0, 0, float(z)] for z in range(0, 35, 4)]     # 9 contacts on x=0, z=0..32
        outlier = [6.0, 0.0, 60.0]                             # 6 mm off-axis, well past the tip
        start, end = fit_line_through_points(good + [outlier], [0, 0, 0], [0, 0, 32])
        # The 9 good contacts remain on the fitted line (would be up to ~1.6 mm
        # off, and the end would overshoot toward z=60, without outlier rejection).
        for c in good:
            self.assertLess(_dist_point_to_line(c, start, end), 0.25)
        self.assertLess(float(end[2]), 40.0)                   # endpoint not dragged to z=60
        self.assertLess(abs(float(end[0])), 0.5)               # axis not tilted toward x=6

    def test_already_fitted_is_stable(self):
        """A line already through its contacts is (near-)unchanged — idempotent."""
        import numpy as np
        from rosa_core.trajectory_fit import fit_line_through_points
        contacts = [[1, 1, 4], [1, 1, 8], [1, 1, 12], [1, 1, 16]]
        s1, e1 = fit_line_through_points(contacts, [1, 1, 0], [1, 1, 20])
        s2, e2 = fit_line_through_points(contacts, s1, e1)
        self.assertTrue(np.allclose(s1, s2, atol=1e-6))
        self.assertTrue(np.allclose(e1, e2, atol=1e-6))


@unittest.skipUnless(DEPS, "numpy/rosa_core not importable.")
class RefitTrajectoriesFileTests(unittest.TestCase):
    TRAJ_COLS = ["name", "start_x", "start_y", "start_z", "end_x", "end_y", "end_z",
                 "confidence_label", "electrode_model", "bolt_source", "length_mm"]
    CONTACT_COLS = ["trajectory", "label", "contact_index", "x", "y", "z",
                    "peak_detected", "electrode_model"]

    def _write(self, path, cols, rows):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
            w.writeheader()
            w.writerows(rows)

    def _make_case(self, root: Path):
        traj = root / "trajectories.tsv"
        contacts = root / "contacts.tsv"
        # A: 4 contacts on x=+2 (seed at x=0) — should refit. B: 1 contact — untouched.
        self._write(traj, self.TRAJ_COLS, [
            {"name": "A", "start_x": "0", "start_y": "0", "start_z": "0",
             "end_x": "0", "end_y": "0", "end_z": "30", "confidence_label": "high",
             "electrode_model": "DIXI-15AM", "bolt_source": "real", "length_mm": "30.000"},
            {"name": "B", "start_x": "5", "start_y": "5", "start_z": "0",
             "end_x": "5", "end_y": "5", "end_z": "20", "confidence_label": "low",
             "electrode_model": "", "bolt_source": "", "length_mm": "20.000"},
        ])
        crows = [{"trajectory": "A", "label": f"A{i+1}", "contact_index": i + 1,
                  "x": "2", "y": "0", "z": z, "peak_detected": "1",
                  "electrode_model": "DIXI-15AM"} for i, z in enumerate((5, 10, 15, 20))]
        crows.append({"trajectory": "B", "label": "B1", "contact_index": 1,
                      "x": "9", "y": "9", "z": "9", "peak_detected": "1", "electrode_model": ""})
        self._write(contacts, self.CONTACT_COLS, crows)
        return traj, contacts

    def _rows(self, path):
        with open(path, newline="") as f:
            return {r["name"]: r for r in csv.DictReader(f, delimiter="\t")}

    def test_refit_moves_line_onto_contacts_preserving_columns(self):
        traj, contacts = self._make_case(Path(tempfile.mkdtemp()))
        from rosa_core.trajectory_fit import refit_trajectories_file
        n = refit_trajectories_file(traj, contacts)
        self.assertEqual(n, 1)                                  # only A refit
        rows = self._rows(traj)
        A = rows["A"]
        # A's line landed on the contact axis (x≈2), length + provenance intact.
        self.assertAlmostEqual(float(A["start_x"]), 2.0, places=3)
        self.assertAlmostEqual(float(A["end_x"]), 2.0, places=3)
        self.assertEqual(A["bolt_source"], "real")
        self.assertEqual(A["confidence_label"], "high")
        self.assertAlmostEqual(float(A["length_mm"]), 30.0, places=2)
        # B (1 contact) untouched.
        B = rows["B"]
        self.assertEqual((B["start_x"], B["end_z"]), ("5", "20"))

    def test_unfamiliar_schema_left_alone(self):
        root = Path(tempfile.mkdtemp())
        traj = root / "traj.tsv"
        self._write(traj, ["name", "entry_x", "tip_x"], [{"name": "A", "entry_x": "1", "tip_x": "2"}])
        (root / "contacts.tsv").write_text("trajectory\tx\ty\tz\n")
        from rosa_core.trajectory_fit import refit_trajectories_file
        self.assertEqual(refit_trajectories_file(traj, root / "contacts.tsv"), 0)

    def test_contacts_with_leading_comment_still_refit(self):
        """A contacts TSV that carries a leading '# ...' provenance comment (as
        fit-rosa stamps) must still be parsed — not treated as the header."""
        traj, contacts = self._make_case(Path(tempfile.mkdtemp()))
        contacts.write_text("# reference_frame: ct\n" + contacts.read_text())
        from rosa_core.trajectory_fit import refit_trajectories_file
        self.assertEqual(refit_trajectories_file(traj, contacts), 1)     # A still refit
        A = self._rows(traj)["A"]
        self.assertAlmostEqual(float(A["start_x"]), 2.0, places=3)

    def test_idempotent_on_disk(self):
        traj, contacts = self._make_case(Path(tempfile.mkdtemp()))
        from rosa_core.trajectory_fit import refit_trajectories_file
        refit_trajectories_file(traj, contacts)
        first = traj.read_text()
        refit_trajectories_file(traj, contacts)
        self.assertEqual(first, traj.read_text())               # second pass = no drift


@unittest.skipUnless(DEPS, "numpy/rosa_core not importable.")
class RefitInplaceTests(unittest.TestCase):
    """The CLI `pipeline` path refits in-memory dict rows before writing."""

    def test_mutates_endpoints_and_skips_sparse(self):
        import numpy as np
        from rosa_core.trajectory_fit import refit_trajectories_inplace
        trajs = [
            {"name": "A", "start_ras": [0, 0, 0], "end_ras": [0, 0, 30]},
            {"name": "B", "start_ras": [9, 9, 0], "end_ras": [9, 9, 20]},   # 1 contact → untouched
        ]
        groups = [
            {"trajectory": "A", "positions_ras": [[2, 0, 5], [2, 0, 10], [2, 0, 15], [2, 0, 20]]},
            {"trajectory": "B", "positions_ras": [[9, 9, 9]]},
        ]
        n = refit_trajectories_inplace(trajs, groups)
        self.assertEqual(n, 1)
        self.assertAlmostEqual(trajs[0]["start_ras"][0], 2.0, places=3)   # A moved onto contacts
        self.assertAlmostEqual(trajs[0]["end_ras"][0], 2.0, places=3)
        self.assertEqual(trajs[1]["start_ras"], [9, 9, 0])               # B unchanged
        # endpoints returned as plain float lists (TSV-writable)
        self.assertTrue(all(isinstance(v, float) for v in trajs[0]["start_ras"]))


if __name__ == "__main__":
    unittest.main()
