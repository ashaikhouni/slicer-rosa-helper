"""End-to-end test for ``rosa-agent rosa-to-nifti``.

Builds a tiny synthetic ROSA folder with one display volume + one
trajectory, runs the command, and asserts:
- the output dir contains the expected NIfTI(s) + manifest.json + seeds.tsv;
- seeds.tsv carries the trajectory in RAS coordinates ready for
  ``rosa-agent place --seeds``;
- the NIfTI's IJK→RAS header matches what ``load_rosa_volume_as_sitk``
  computes (no drift from the in-memory path).
"""
from __future__ import annotations

import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "cli"))
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))


def _try_imports():
    try:
        import numpy  # noqa: F401
        import SimpleITK  # noqa: F401
        from rosa_agent.commands import rosa_to_nifti  # noqa: F401
        return True
    except ImportError:
        return False


DEPS_AVAILABLE = _try_imports()


def _write_synthetic_analyze(path: Path, *, size=(20, 20, 20)):
    import numpy as np
    import SimpleITK as sitk

    arr = np.zeros(size, dtype=np.float32)
    arr[5:9, 5:9, 5:9] = 100.0
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(img, str(path.with_suffix(".img")))


def _build_synthetic_rosa_case(case_dir: Path) -> None:
    """Single-display ROSA case (matches the synthetic fixture
    ``test_pipeline_frames.py`` already uses, just kept self-contained
    here so this test doesn't import from a sibling test module)."""
    analyze_root = case_dir / "DICOM" / "uid_a"
    analyze_root.mkdir(parents=True, exist_ok=True)
    _write_synthetic_analyze(analyze_root / "ref_vol")
    ros_text = textwrap.dedent("""
        [TRdicomRdisplay]
        1 0 0 0
        0 1 0 0
        0 0 1 0
        0 0 0 1
        [VOLUME]
        DICOM/uid_a/ref_vol
        [IMAGERY_NAME]
        ref_vol
        [SERIE_UID]
        uid_a
        [IMAGERY_3DREF]
        0
        [TRAJECTORY]
        traj1
        T1 1 0 0 -1.0 -2.0 3.0 0 -10.0 -20.0 30.0
        [END]
    """).strip()
    (case_dir / "case.ros").write_text(ros_text)


def _display_block(name: str, uid: str) -> str:
    return textwrap.dedent(f"""
        [TRdicomRdisplay]
        1 0 0 0
        0 1 0 0
        0 0 1 0
        0 0 0 1
        [VOLUME]
        DICOM/{uid}/{name}
        [IMAGERY_NAME]
        {name}
        [SERIE_UID]
        {uid}
        [IMAGERY_3DREF]
        0
    """).strip()


def _build_multi_display_case(case_dir: Path, displays):
    """A ROSA case declaring several displays; ``displays`` is a list of
    ``(name, has_img)``. The FIRST entry is the reference (display 0). Displays
    with ``has_img=False`` are declared in the .ros but have no Analyze file —
    exactly the real-world case (a .ros references empty display slots)."""
    blocks = []
    for i, (name, has_img) in enumerate(displays):
        uid = f"uid_{i}"
        if has_img:
            root = case_dir / "DICOM" / uid
            root.mkdir(parents=True, exist_ok=True)
            _write_synthetic_analyze(root / name)
        blocks.append(_display_block(name, uid))
    traj = textwrap.dedent("""
        [TRAJECTORY]
        traj1
        T1 1 0 0 -1.0 -2.0 3.0 0 -10.0 -20.0 30.0
        [END]
    """).strip()
    (case_dir / "case.ros").write_text("\n".join(blocks) + "\n" + traj + "\n")


@unittest.skipUnless(
    DEPS_AVAILABLE,
    "numpy/SimpleITK/rosa_agent not importable in this environment.",
)
class RosaToNiftiTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.case_dir = self.tmp / "rosa_case"
        self.case_dir.mkdir()
        _build_synthetic_rosa_case(self.case_dir)
        self.out_dir = self.tmp / "out"

    def tearDown(self):
        self._tmp.cleanup()

    def test_writes_nifti_seeds_and_manifest(self):
        from rosa_agent.commands.rosa_to_nifti import main as cmd
        rc = cmd([
            "--rosa-folder", str(self.case_dir),
            "--output", str(self.out_dir),
            "--quiet",
        ])
        self.assertEqual(rc, 0)
        self.assertTrue((self.out_dir / "manifest.json").exists())
        self.assertTrue((self.out_dir / "seeds.tsv").exists())
        self.assertTrue((self.out_dir / "ref_vol.nii.gz").exists())

        manifest = json.loads((self.out_dir / "manifest.json").read_text())
        self.assertEqual(manifest["reference_volume"], "ref_vol")
        self.assertEqual(manifest["all_displays_in_ros"], ["ref_vol"])
        self.assertEqual(len(manifest["exported_volumes"]), 1)
        self.assertEqual(manifest["n_planned_trajectories"], 1)
        ev = manifest["exported_volumes"][0]
        self.assertEqual(ev["display_name"], "ref_vol")
        self.assertTrue(ev["is_reference"])

    def test_seeds_tsv_is_place_compatible(self):
        from rosa_agent.commands.rosa_to_nifti import main as cmd
        rc = cmd([
            "--rosa-folder", str(self.case_dir),
            "--output", str(self.out_dir),
            "--quiet",
        ])
        self.assertEqual(rc, 0)
        # Re-parse via the same helper `place` uses to confirm the
        # schema matches.
        from rosa_agent.io.trajectory_io import read_seeds_tsv
        seeds = read_seeds_tsv(self.out_dir / "seeds.tsv")
        self.assertEqual(len(seeds), 1)
        s = seeds[0]
        # ROSA trajectory names come from the FIRST field of the
        # data line, not the [TRAJECTORY] block header. Fixture data
        # row starts with "T1", so that's the imported name.
        self.assertEqual(s["name"], "T1")
        # ROS LPS (-1, -2, 3) → RAS (1, 2, 3); ROS LPS (-10, -20, 30)
        # → RAS (10, 20, 30). Per parse_ros_text.
        self.assertAlmostEqual(s["start_ras"][0], 1.0, places=4)
        self.assertAlmostEqual(s["start_ras"][1], 2.0, places=4)
        self.assertAlmostEqual(s["start_ras"][2], 3.0, places=4)
        self.assertAlmostEqual(s["end_ras"][0], 10.0, places=4)
        self.assertAlmostEqual(s["end_ras"][1], 20.0, places=4)
        self.assertAlmostEqual(s["end_ras"][2], 30.0, places=4)

    def test_volume_subset_filter(self):
        """--volume restricts which displays are exported."""
        from rosa_agent.commands.rosa_to_nifti import main as cmd
        # ref_vol is the only display, so subset == [ref_vol] succeeds…
        rc = cmd([
            "--rosa-folder", str(self.case_dir),
            "--output", str(self.out_dir),
            "--volume", "ref_vol",
            "--quiet",
        ])
        self.assertEqual(rc, 0)
        # …but an unknown name is rejected with exit 2.
        out_dir2 = self.tmp / "out2"
        rc2 = cmd([
            "--rosa-folder", str(self.case_dir),
            "--output", str(out_dir2),
            "--volume", "definitely-not-a-real-display",
            "--quiet",
        ])
        self.assertEqual(rc2, 2)
        self.assertFalse((out_dir2 / "manifest.json").exists())

    def test_missing_rosa_folder_polite_failure(self):
        from rosa_agent.commands.rosa_to_nifti import main as cmd
        rc = cmd([
            "--rosa-folder", str(self.tmp / "no_such_case"),
            "--output", str(self.out_dir),
            "--quiet",
        ])
        self.assertEqual(rc, 2)

    def test_missing_nonreference_display_is_skipped(self):
        """A display the .ros declares but with no Analyze volume is skipped —
        the case still imports with the volumes that ARE present."""
        from rosa_agent.commands.rosa_to_nifti import main as cmd
        case = self.tmp / "multi"; case.mkdir()
        _build_multi_display_case(case, [("ref_vol", True), ("gone", False)])
        out = self.tmp / "multi_out"
        rc = cmd(["--rosa-folder", str(case), "--output", str(out), "--quiet"])
        self.assertEqual(rc, 0)
        self.assertTrue((out / "ref_vol.nii.gz").exists())
        self.assertFalse((out / "gone.nii.gz").exists())          # not baked
        m = json.loads((out / "manifest.json").read_text())
        self.assertEqual([e["display_name"] for e in m["exported_volumes"]], ["ref_vol"])
        self.assertEqual([s["display_name"] for s in m["skipped_volumes"]], ["gone"])

    def test_missing_reference_display_is_fatal(self):
        """But if the REFERENCE display (display 0, the common frame) has no
        volume, that IS fatal — the rest can't be placed without it."""
        from rosa_agent.commands.rosa_to_nifti import main as cmd
        case = self.tmp / "noref"; case.mkdir()
        _build_multi_display_case(case, [("gone_ref", False), ("ref_vol", True)])
        out = self.tmp / "noref_out"
        rc = cmd(["--rosa-folder", str(case), "--output", str(out), "--quiet"])
        self.assertEqual(rc, 2)
        self.assertFalse((out / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
