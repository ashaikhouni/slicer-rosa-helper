"""Integration tests for the ``rosa-agent place`` CLI subcommand.

Two test classes:

* ``PlaceCommandShapeTests`` — runs ``place.main(...)`` on a tiny synthetic
  CT (works on any machine, no dataset needed). Validates: argparse
  surface, mode dispatch, output directory layout, manifest schema.

* ``PlaceCommandAmc88Tests`` — gated on the AMC88 dataset env var. Runs
  the full mode-1 pipeline end-to-end and checks the output contains
  the expected 8 GT-matched trajectories in the high band.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "cli"))
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))

AMC_ROOT = Path(os.environ.get("ROSA_AMC_TESTING_ROOT",
                                "/Users/ammar/Documents/testing"))


def _try_imports():
    try:
        import SimpleITK  # noqa: F401
        from rosa_agent.commands import place  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(_try_imports(),
                     "needs SimpleITK + rosa_agent.commands.place")
class PlaceCommandShapeTests(unittest.TestCase):
    """Run ``place`` on a synthetic CT — exercises the full CLI surface
    without depending on real datasets."""

    @classmethod
    def setUpClass(cls):
        # Build a minimal synthetic NIfTI: identity 1mm IJK→RAS, all zero
        # (the candidate-seed generator will emit nothing, but the
        # dispatcher should still produce the empty-output directory).
        import SimpleITK as sitk
        cls.tmp = tempfile.TemporaryDirectory()
        arr = np.zeros((32, 32, 32), dtype=np.float32)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((1.0, 1.0, 1.0))
        img.SetOrigin((0.0, 0.0, 0.0))
        img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        cls.ct_path = Path(cls.tmp.name) / "synthetic_ct.nii.gz"
        sitk.WriteImage(img, str(cls.ct_path))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_mode_1_writes_directory_layout(self):
        from rosa_agent.commands.place import main as place_main
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "qc"
            rc = place_main([
                "--ct", str(self.ct_path),
                "--output", str(out_dir),
                "--no-figures",
                "--quiet",
            ])
            self.assertEqual(rc, 0)
            # Directory layout.
            self.assertTrue((out_dir / "manifest.json").exists())
            self.assertTrue((out_dir / "trajectories.tsv").exists())
            self.assertTrue((out_dir / "contacts.tsv").exists())
            self.assertTrue((out_dir / "diagnostics" / "cmp.tsv").exists())
            self.assertFalse((out_dir / "figures").exists())
            # Manifest schema.
            data = json.loads((out_dir / "manifest.json").read_text())
            self.assertEqual(data["mode"], 1)
            self.assertIn("pipeline_version", data)
            self.assertIn("timestamp", data)
            self.assertIn("runtime_seconds", data)
            self.assertIsInstance(data["n_trajectories"], int)

    def test_mode_2_n_expected(self):
        from rosa_agent.commands.place import main as place_main
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "qc"
            rc = place_main([
                "--ct", str(self.ct_path),
                "--output", str(out_dir),
                "--n-expected", "5",
                "--no-figures", "--quiet",
            ])
            self.assertEqual(rc, 0)
            data = json.loads((out_dir / "manifest.json").read_text())
            self.assertEqual(data["mode"], 2)

    def test_rejects_seeds_plus_expected(self):
        # Mode-arg conflict: pre-validated by the CLI before any work.
        from rosa_agent.commands.place import main as place_main
        with tempfile.TemporaryDirectory() as tmp:
            seeds = Path(tmp) / "seeds.tsv"
            seeds.write_text(
                "name\tstart_x\tstart_y\tstart_z\tend_x\tend_y\tend_z\n"
                "L1\t0\t0\t0\t10\t0\t0\n",
            )
            expected = Path(tmp) / "expected.tsv"
            expected.write_text("name\tmodel_id\nL1\tDIXI-15\n")
            rc = place_main([
                "--ct", str(self.ct_path),
                "--output", str(Path(tmp) / "qc"),
                "--seeds", str(seeds),
                "--expected", str(expected),
                "--no-figures", "--quiet",
            ])
            self.assertEqual(rc, 2)  # error exit


@unittest.skipUnless(_try_imports() and (AMC_ROOT / "AMC88").is_dir(),
                     f"AMC88 not found at {AMC_ROOT}")
class PlaceCommandAmc88Tests(unittest.TestCase):
    """Live end-to-end on AMC88 — pin the same notebook numbers the
    integration suite already covers, but through the CLI surface."""

    def test_amc88_mode_1_via_cli(self):
        from rosa_agent.commands.place import main as place_main
        amc_dir = AMC_ROOT / "AMC88"
        ct = next(amc_dir.glob("*_CT.nii.gz"), None) or next(amc_dir.glob("*.nii.gz"))
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "qc"
            rc = place_main([
                "--ct", str(ct),
                "--output", str(out_dir),
                "--library", "pmt_35",
                "--no-figures",  # save time; figures tested elsewhere
                "--quiet",
            ])
            self.assertEqual(rc, 0)

            # Manifest counts: 8 GT shanks land in 'high' (notebook).
            # The CLI defaults band_floor="medium", so medium-band orphans
            # come through too — verify n_high pins the GT count and the
            # total count matches what the dispatcher reports on stdout.
            data = json.loads((out_dir / "manifest.json").read_text())
            self.assertEqual(data["mode"], 1)
            self.assertEqual(data["library_id"], "pmt_35")
            self.assertEqual(data["n_high"], 8,
                             "AMC88 should have 8 GT shanks in 'high'")
            self.assertEqual(data["n_low"], 0,
                             "band_floor='medium' (default) drops 'low' orphans")
            # trajectories.tsv: header + (n_high + n_medium) rows.
            traj_text = (out_dir / "trajectories.tsv").read_text()
            n_data_rows = len(traj_text.splitlines()) - 1
            self.assertEqual(
                n_data_rows, data["n_high"] + data["n_medium"],
                f"trajectories.tsv row count should match high+medium "
                f"manifest counts; got {n_data_rows} rows for "
                f"{data['n_high']}+{data['n_medium']}",
            )


if __name__ == "__main__":
    unittest.main()
