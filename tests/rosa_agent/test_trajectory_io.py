"""Sanity tests for the trajectory_io TSV reader/writers.

The TSV column names are the rosa_agent CLI's public contract — these
tests pin both the column set and the loose-format seed parsing.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "cli"))


class TrajectoryIoTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_trajectory_roundtrip(self):
        from rosa_agent.io.trajectory_io import (
            TRAJECTORY_COLUMNS,
            read_seeds_tsv,
            write_trajectories_tsv,
        )

        trajs = [
            {
                "name": "L_AC",
                "start_ras": [-0.78, 11.63, 28.91],
                "end_ras": [-18.54, 3.90, 53.75],
                "confidence": 0.91,
                "confidence_label": "high",
                "electrode_model": "DIXI-15AM",
                "bolt": {"source": "metal"},
            }
        ]
        out = self.tmp / "trajectories.tsv"
        n = write_trajectories_tsv(out, trajs)
        self.assertEqual(n, 1)

        seeds = read_seeds_tsv(out)
        self.assertEqual(len(seeds), 1)
        self.assertEqual(seeds[0]["name"], "L_AC")
        self.assertAlmostEqual(seeds[0]["start_ras"][0], -0.78, places=2)
        self.assertAlmostEqual(seeds[0]["end_ras"][2], 53.75, places=2)

        # Confirm the column set was honored.
        header = out.read_text().splitlines()[0].split("\t")
        self.assertEqual(tuple(header), TRAJECTORY_COLUMNS)

    def test_endpoint_pair_form(self):
        from rosa_agent.io.trajectory_io import read_seeds_tsv

        path = self.tmp / "seeds.tsv"
        path.write_text(
            "label\tx\ty\tz\n"
            "L_AC\t-0.78\t11.63\t28.91\n"
            "L_AC\t-18.54\t3.90\t53.75\n"
            "L_PH\t1.0\t2.0\t3.0\n"
            "L_PH\t10.0\t11.0\t12.0\n"
        )
        seeds = read_seeds_tsv(path)
        self.assertEqual([s["name"] for s in seeds], ["L_AC", "L_PH"])
        self.assertAlmostEqual(seeds[1]["end_ras"][1], 11.0)


if __name__ == "__main__":
    unittest.main()
