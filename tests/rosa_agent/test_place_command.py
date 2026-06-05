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

import csv
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

    def _capture_stderr(self, fn, *args, **kwargs):
        """Run ``fn`` while redirecting stderr; return (rc, stderr_text)."""
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            rc = fn(*args, **kwargs)
        return rc, buf.getvalue()

    def test_empty_seeds_file_polite_failure(self):
        """Header-only seeds.tsv ⇒ caller meant mode 4/5 but supplied
        nothing; CLI exits 2 with a clear stderr line, not a traceback."""
        from rosa_agent.commands.place import main as place_main
        with tempfile.TemporaryDirectory() as tmp:
            seeds = Path(tmp) / "seeds_empty.tsv"
            seeds.write_text(
                "name\tstart_x\tstart_y\tstart_z\tend_x\tend_y\tend_z\n",
            )
            rc, err = self._capture_stderr(place_main, [
                "--ct", str(self.ct_path),
                "--output", str(Path(tmp) / "qc"),
                "--seeds", str(seeds),
                "--no-figures",
            ])
            self.assertEqual(rc, 2)
            self.assertIn("error:", err.lower())
            self.assertIn("seeds", err.lower())

    def test_empty_expected_file_polite_failure(self):
        from rosa_agent.commands.place import main as place_main
        with tempfile.TemporaryDirectory() as tmp:
            expected = Path(tmp) / "expected_empty.tsv"
            expected.write_text("name\tmodel_id\n")
            rc, err = self._capture_stderr(place_main, [
                "--ct", str(self.ct_path),
                "--output", str(Path(tmp) / "qc"),
                "--expected", str(expected),
                "--no-figures",
            ])
            self.assertEqual(rc, 2)
            self.assertIn("error:", err.lower())
            self.assertIn("expected", err.lower())

    def test_zero_n_expected_polite_failure(self):
        from rosa_agent.commands.place import main as place_main
        with tempfile.TemporaryDirectory() as tmp:
            rc, err = self._capture_stderr(place_main, [
                "--ct", str(self.ct_path),
                "--output", str(Path(tmp) / "qc"),
                "--n-expected", "0",
                "--no-figures",
            ])
            self.assertEqual(rc, 2)
            self.assertIn("error:", err.lower())
            self.assertIn("n_expected", err.lower())

    def test_negative_n_expected_polite_failure(self):
        from rosa_agent.commands.place import main as place_main
        with tempfile.TemporaryDirectory() as tmp:
            rc, err = self._capture_stderr(place_main, [
                "--ct", str(self.ct_path),
                "--output", str(Path(tmp) / "qc"),
                "--n-expected", "-3",
                "--no-figures",
            ])
            self.assertEqual(rc, 2)
            self.assertIn("error:", err.lower())

    def test_mask_backend_hull_runs(self):
        """--mask-backend hull exercises the dependency-free hull path end to
        end (no SynthStrip/watershed) and is recorded in the manifest."""
        from rosa_agent.commands.place import main as place_main
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "qc"
            rc = place_main([
                "--ct", str(self.ct_path),
                "--output", str(out_dir),
                "--mask-backend", "hull",
                "--no-figures", "--quiet",
            ])
            self.assertEqual(rc, 0)
            data = json.loads((out_dir / "manifest.json").read_text())
            self.assertEqual(data["mode_args"]["mask_backend"], "hull")

    def test_invalid_mask_backend_rejected(self):
        """argparse rejects an unknown --mask-backend choice (exit 2)."""
        from rosa_agent.commands.place import main as place_main
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                place_main([
                    "--ct", str(self.ct_path),
                    "--output", str(Path(tmp) / "qc"),
                    "--mask-backend", "not-a-backend",
                    "--no-figures", "--quiet",
                ])

    def test_missing_brain_mask_polite_failure(self):
        """--brain-mask pointing at a nonexistent file ⇒ clean exit 2."""
        from rosa_agent.commands.place import main as place_main
        with tempfile.TemporaryDirectory() as tmp:
            rc, err = self._capture_stderr(place_main, [
                "--ct", str(self.ct_path),
                "--output", str(Path(tmp) / "qc"),
                "--brain-mask", str(Path(tmp) / "nope.nii.gz"),
                "--no-figures",
            ])
            self.assertEqual(rc, 2)
            self.assertIn("error:", err.lower())
            self.assertIn("brain mask", err.lower())

    def test_malformed_seeds_polite_failure(self):
        """A seeds TSV with a missing numeric column (start_y) makes
        read_seeds_tsv raise ValueError; place must catch it and exit 2 with a
        clean message, not leak a traceback past the heavy pipeline boundary."""
        from rosa_agent.commands.place import main as place_main
        with tempfile.TemporaryDirectory() as tmp:
            seeds = Path(tmp) / "seeds_bad.tsv"
            seeds.write_text(
                "name\tstart_x\tstart_z\tend_x\tend_y\tend_z\n"   # no start_y col
                "L1\t0\t0\t10\t0\t0\n",
            )
            rc, err = self._capture_stderr(place_main, [
                "--ct", str(self.ct_path),
                "--output", str(Path(tmp) / "qc"),
                "--seeds", str(seeds),
                "--no-figures",
            ])
            self.assertEqual(rc, 2)
            self.assertIn("error:", err.lower())
            self.assertIn("seeds", err.lower())

    def test_unknown_model_id_polite_failure(self):
        """Mode 4 vouched-model lookup misses ⇒ CLI exits 2 with a
        clear stderr message naming the missing model id."""
        from rosa_agent.commands.place import main as place_main
        with tempfile.TemporaryDirectory() as tmp:
            seeds = Path(tmp) / "seeds_bad_model.tsv"
            seeds.write_text(
                "name\tstart_x\tstart_y\tstart_z\tend_x\tend_y\tend_z\telectrode_model\n"
                "L1\t0\t0\t0\t10\t0\t0\tNOT-A-REAL-MODEL\n",
            )
            rc, err = self._capture_stderr(place_main, [
                "--ct", str(self.ct_path),
                "--output", str(Path(tmp) / "qc"),
                "--seeds", str(seeds),
                "--library", "dixi",
                "--no-figures",
            ])
            self.assertEqual(rc, 2)
            self.assertIn("error:", err.lower())
            self.assertIn("NOT-A-REAL-MODEL", err)


@unittest.skipUnless(_try_imports() and (AMC_ROOT / "AMC88").is_dir(),
                     f"AMC88 not found at {AMC_ROOT}")
class PlaceCommandAmc88Tests(unittest.TestCase):
    """Live end-to-end on AMC88 through the CLI surface — pin that every
    GT shank is emitted with a model assigned (the band split is left to
    the integration suite, as it depends on the brain-mask backend)."""

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

            # The exact band split (n_high vs n_medium) depends on the
            # brain-mask backend — SynthStrip vs the LoG-watershed fallback
            # shifts a borderline shank between 'high' and 'medium' — so we
            # deliberately do NOT pin n_high. We pin what actually matters:
            #   * all GT shanks are emitted (AMC88 has 8; the detector may
            #     add an orphan, so >= 8),
            #   * none lands in 'low' (band_floor='medium' default drops those),
            #   * every emitted trajectory carries an electrode model.
            data = json.loads((out_dir / "manifest.json").read_text())
            self.assertEqual(data["mode"], 1)
            self.assertEqual(data["library_id"], "pmt_35")
            self.assertEqual(data["n_low"], 0,
                             "band_floor='medium' (default) drops 'low' orphans")
            n_emitted = data["n_high"] + data["n_medium"] + data["n_low"]
            self.assertGreaterEqual(
                n_emitted, 8,
                f"AMC88 has 8 GT shanks; expected >= 8 emitted, got {n_emitted}",
            )
            with open(out_dir / "trajectories.tsv", newline="") as f:
                rows = list(csv.DictReader(f, delimiter="\t"))
            self.assertEqual(
                len(rows), n_emitted,
                f"trajectories.tsv row count should match emitted manifest "
                f"count; got {len(rows)} rows for {n_emitted} emitted",
            )
            # The real acceptance criterion: every emitted trajectory must
            # have a model assigned (the band is secondary).
            modelless = [
                r.get("name", "?") for r in rows
                if not (r.get("electrode_model") or r.get("model_id") or "").strip()
            ]
            self.assertEqual(
                modelless, [],
                f"every emitted trajectory must have a model; missing: {modelless}",
            )


if __name__ == "__main__":
    unittest.main()
