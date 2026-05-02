"""Curry .pom export — round-trip + format-marker tests."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))

from rosa_core.curry_export import (  # noqa: E402
    contacts_to_pom_points,
    trajectory_endpoints_to_pom_points,
    write_curry_pom,
)


class CurryPomWriterTests(unittest.TestCase):
    def test_per_line_label_and_lps_flip(self):
        # Two RAS points; expect each row as `label x y z` in LPS.
        points = [
            ("RAH1", 10.0, 20.0, 30.0),
            ("RAH2", -5.5, 12.0, 0.0),
        ]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "test.pom"
            n = write_curry_pom(out, points, coords_in="ras")
            self.assertEqual(n, 2)
            text = out.read_text()
        # Per-line layout: label x y z, LPS = [-x_ras, -y_ras, z_ras]
        self.assertIn("RAH1 -10.000000 -20.000000 30.000000", text)
        self.assertIn("RAH2 5.500000 -12.000000 0.000000", text)
        # Each output line should split into exactly 4 whitespace tokens
        for line in text.strip().splitlines():
            tokens = line.split()
            self.assertEqual(len(tokens), 4, f"row {line!r} not 4 tokens")

    def test_lps_input_no_flip(self):
        points = [("X1", 1.0, 2.0, 3.0)]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "test.pom"
            write_curry_pom(out, points, coords_in="lps")
            text = out.read_text()
        # LPS in -> LPS out unchanged
        self.assertIn("X1 1.000000 2.000000 3.000000", text)

    def test_label_with_spaces_collapsed(self):
        points = [("multi  word name", 1.0, 2.0, 3.0)]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "test.pom"
            write_curry_pom(out, points, coords_in="lps")
            text = out.read_text()
        # Spaces inside label get replaced so the row still splits
        # into exactly 4 whitespace-separated tokens.
        self.assertIn("multi__word_name 1.000000 2.000000 3.000000", text)
        line = text.strip().splitlines()[0]
        self.assertEqual(len(line.split()), 4)

    def test_no_points_raises(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "empty.pom"
            with self.assertRaises(ValueError):
                write_curry_pom(out, [])

    def test_invalid_coord_system_raises(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "x.pom"
            with self.assertRaises(ValueError):
                write_curry_pom(out, [("a", 0, 0, 0)], coords_in="mni")


class ContactsToPomPointsTests(unittest.TestCase):
    def test_label_format_traj_plus_index(self):
        contacts = [
            {"trajectory": "RAH", "index": 1, "position_ras": [1.0, 2.0, 3.0]},
            {"trajectory": "RAH", "index": 10, "position_ras": [4.0, 5.0, 6.0]},
        ]
        out = contacts_to_pom_points(contacts)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0][0], "RAH1")
        self.assertEqual(out[1][0], "RAH10")
        self.assertEqual(out[0][1:], (1.0, 2.0, 3.0))

    def test_lps_input_converted_to_ras(self):
        contacts = [
            {"trajectory": "X", "index": 1, "position_lps": [-1.0, -2.0, 3.0]},
        ]
        out = contacts_to_pom_points(contacts)
        # LPS [-1, -2, 3] -> RAS [1, 2, 3]
        self.assertEqual(out[0][1:], (1.0, 2.0, 3.0))

    def test_missing_position_skipped(self):
        contacts = [
            {"trajectory": "A", "index": 1},  # no position
            {"trajectory": "B", "index": 2, "position_ras": [0, 0, 0]},
        ]
        out = contacts_to_pom_points(contacts)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][0], "B2")


class TrajectoryEndpointsToPomPointsTests(unittest.TestCase):
    def test_two_points_per_trajectory(self):
        trajs = [
            {"name": "RAH", "start_ras": [1, 2, 3], "end_ras": [10, 11, 12]},
            {"name": "LAH", "start_ras": [-1, -2, -3], "end_ras": [-10, -11, -12]},
        ]
        out = trajectory_endpoints_to_pom_points(trajs)
        self.assertEqual(len(out), 4)
        self.assertEqual([p[0] for p in out], ["RAH_E", "RAH_T", "LAH_E", "LAH_T"])
        self.assertEqual(out[0][1:], (1, 2, 3))
        self.assertEqual(out[1][1:], (10, 11, 12))


if __name__ == "__main__":
    unittest.main()
