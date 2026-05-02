"""Pin pipeline frame-handling on a synthetic ROSA folder.

Real-world ROSA flows can't be exercised against the regression dataset
(no .ros file per subject — the dataset ships a CT NIfTI only). This
test builds a tiny synthetic ROSA folder + Analyze volumes + a one-line
.ros file and pins the pipeline's frame model:

  * Mode B (ROSA folder, no external CT): the chosen ROSA reference is
    loaded as the working CT. Output dir contains a ct.nii.gz (Analyze
    converted to NIfTI for the user) plus trajectories/contacts in the
    ROSA reference frame.

  * Dataset-mode regression: doesn't write ct.nii.gz (no value in
    copying a CT that's already a NIfTI on disk).

We don't gate on detected trajectory count — the synthetic phantom
isn't realistic enough for detection — only on the I/O contract.
"""

from __future__ import annotations

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
        from rosa_agent.commands import pipeline  # noqa: F401
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
    """Single-display ROSA case so we don't need IMAGERY_3DREF chains."""
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


@unittest.skipUnless(
    DEPS_AVAILABLE,
    "numpy/SimpleITK/rosa_agent not importable in this environment.",
)
class PipelineRosaFolderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.case_dir = self.tmp / "rosa_case"
        self.case_dir.mkdir()
        _build_synthetic_rosa_case(self.case_dir)
        self.out_dir = self.tmp / "out"

    def tearDown(self):
        self._tmp.cleanup()

    def test_rosa_folder_no_ct_writes_working_ct_nifti(self):
        """Mode B: working CT comes from Analyze inside the .ros folder.
        We must emit ct.nii.gz so detection has a NIfTI to read AND the
        user gets a portable copy of the converted volume.
        """
        from rosa_agent.commands.pipeline import run_pipeline

        try:
            summary = run_pipeline(
                str(self.case_dir),
                out_dir=self.out_dir,
                ref_volume="ref_vol",
            )
        except SystemExit:
            # Detection may not find anything on the toy phantom; that's
            # OK — we only care about the I/O contract before/around
            # detection. SystemExit is the only thing we truly can't
            # absorb (means a fatal validation error before detection).
            self.fail("pipeline raised SystemExit before detection ran")
        except Exception:
            # Anything during detection is acceptable — we still need
            # the manifest + working CT to have been written.
            pass

        self.assertTrue((self.out_dir / "ct.nii.gz").exists(),
                        "ROSA-folder mode must emit ct.nii.gz")
        self.assertTrue((self.out_dir / "manifest.json").exists())

    def test_rosa_folder_with_external_ct_resolves_frame(self):
        """Mode C: ROSA folder + --ct external.nii.gz with
        --skip-registration. Pin the resolver — this exercises the
        external-CT branch of _resolve_pipeline_frame, which is where
        the working_ct_path = out_ct UnboundLocalError lived. We pass
        ``--skip-registration`` so the test stays fast (no MI run on a
        toy phantom).
        """
        import SimpleITK as sitk
        import numpy as np
        from rosa_agent.commands.pipeline import _resolve_pipeline_frame

        # Synthesize an external CT NIfTI (any frame — skip-registration
        # short-circuits the actual alignment check).
        external_ct = self.tmp / "external_ct.nii.gz"
        arr = np.zeros((20, 20, 20), dtype=np.float32)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((1.0, 1.0, 1.0))
        sitk.WriteImage(img, str(external_ct))

        out = self.tmp / "ext_out"
        out.mkdir()
        frame = _resolve_pipeline_frame(
            str(self.case_dir),
            out_dir=out,
            ct_override=str(external_ct),
            ref_volume="ref_vol",
            skip_registration=True,
        )
        # The fix: working CT is the external CT, NOT the ROSA-reference
        # NIfTI written into out_dir.
        self.assertEqual(
            Path(frame.working_ct_path).resolve(),
            external_ct.resolve(),
            "external CT path must be returned as the working CT",
        )
        # We must NOT have written a ct.nii.gz copy under out_dir for
        # the external-CT path (per user's "no value duplicating a
        # NIfTI the user already has on disk").
        self.assertFalse(
            (out / "ct.nii.gz").exists(),
            "external CT must not be copied into out_dir",
        )
        # The ROSA-derived seeds should have been transformed into the
        # working frame. With skip_registration the transform is
        # identity, so seeds equal the ROSA-frame planned trajectories.
        self.assertEqual(len(frame.seeds), 1)
        self.assertEqual(frame.seeds[0]["name"], "T1")
        np.testing.assert_allclose(frame.seeds[0]["start_ras"], [1.0, 2.0, 3.0])

    def test_rosa_folder_with_registered_external_ct_pushes_seeds_correctly(self):
        """Mode C with REAL registration: pin that ROSA-frame seeds end
        up close to where they belong in the CT frame after registration.

        Catches the direction error the user flagged: when the pipeline
        mistakenly used ``fixed_to_moving_ras_4x4`` to push ROSA seeds
        into the CT frame, seeds landed off by ~2× the true translation.
        With the corrected ``moving_to_fixed_ras_4x4`` they land within
        sub-voxel.
        """
        import numpy as np
        import SimpleITK as sitk
        from rosa_agent.commands.pipeline import _resolve_pipeline_frame
        from rosa_core import load_rosa_volume_as_sitk

        # Replace the synthetic Analyze ref_vol from setUp with a 3-cube
        # phantom that has enough structure for Mattes MI to converge.
        size = 80
        arr = np.zeros((size, size, size), dtype=np.float32)
        arr[16:28, 16:28, 16:28] = 1000.0
        arr[48:64, 24:36, 44:56] = 1500.0
        arr[40:52, 56:72, 12:24] = 800.0
        ref_phantom = sitk.GetImageFromArray(arr)
        ref_phantom.SetSpacing((1.0, 1.0, 1.0))
        analyze_img = self.case_dir / "DICOM" / "uid_a" / "ref_vol.img"
        sitk.WriteImage(ref_phantom, str(analyze_img))

        # Load the ROSA-centered reference image so we know its frame.
        # The external CT we synthesize must live in the SAME centered
        # frame (so registration recovers JUST the known +5 LPS shift,
        # not the centering offset). Take the centered ref image, apply
        # an additional +5 LPS-X translation to the SITK image's origin
        # (no resample — pure affine relabel), and that's the external CT.
        ref_img, _ref_meta = load_rosa_volume_as_sitk(
            str(self.case_dir), volume_name="ref_vol",
        )
        ext_origin_lps = list(ref_img.GetOrigin())
        ext_origin_lps[0] += 5.0   # shift in LPS-X
        external_ct = sitk.GetImageFromArray(
            sitk.GetArrayFromImage(ref_img)
        )
        external_ct.SetSpacing(ref_img.GetSpacing())
        external_ct.SetDirection(ref_img.GetDirection())
        external_ct.SetOrigin(tuple(float(v) for v in ext_origin_lps))
        external_ct_path = self.tmp / "external_ct.nii.gz"
        sitk.WriteImage(external_ct, str(external_ct_path))

        out = self.tmp / "ext_reg_out"
        out.mkdir()

        frame = _resolve_pipeline_frame(
            str(self.case_dir),
            out_dir=out,
            ct_override=str(external_ct_path),
            ref_volume="ref_vol",
            skip_registration=False,   # the actual registration path
        )

        # The .ros TRAJECTORY had start LPS=(-1, -2, 3) -> RAS=(1, 2, 3),
        # end LPS=(0, -10, -20) -> RAS=(0, 10, -20). Wait — re-checking
        # _build_synthetic_rosa_case: start_lps=(-1,-2,3), end_lps
        # depends on the numbers. Recompute from the ros file:
        # T1 1 0 0 -1.0 -2.0 3.0 0 -10.0 -20.0 30.0
        # fields[4..6] = start = (-1, -2, 3) LPS -> RAS (1, 2, 3)
        # fields[8..10] = end = (-10, -20, 30) LPS -> RAS (10, 20, 30)
        rosa_start_ras = np.array([1.0, 2.0, 3.0])
        rosa_end_ras = np.array([10.0, 20.0, 30.0])

        # Translation +5 LPS-X = -5 RAS-X. ROSA-RAS X=1 → CT-RAS X=1-5=-4.
        expected_ct_start = rosa_start_ras + np.array([-5.0, 0.0, 0.0])
        expected_ct_end = rosa_end_ras + np.array([-5.0, 0.0, 0.0])

        self.assertEqual(len(frame.seeds), 1)
        seed = frame.seeds[0]
        got_start = np.asarray(seed["start_ras"], dtype=float)
        got_end = np.asarray(seed["end_ras"], dtype=float)
        # Sub-voxel recovery on a clean synthetic phantom. If this fails
        # > 1 mm, the registration direction is reversed (the pre-fix
        # bug pushed seeds in the wrong direction by ~2× the offset,
        # which would land ~10 mm away on this test).
        self.assertLess(
            float(np.linalg.norm(got_start - expected_ct_start)),
            1.5,
            f"ROSA seed start pushed to wrong CT-frame location: "
            f"got {got_start}, expected {expected_ct_start} "
            f"(if off by ~10 mm, the registration direction is reversed)",
        )
        self.assertLess(
            float(np.linalg.norm(got_end - expected_ct_end)),
            1.5,
            f"ROSA seed end pushed to wrong CT-frame location: "
            f"got {got_end}, expected {expected_ct_end}",
        )

    def test_dataset_mode_does_not_copy_ct(self):
        """Dataset-id mode: CT is already a NIfTI on disk; no copy."""
        import os
        if not os.environ.get("ROSA_SEEG_DATASET"):
            self.skipTest("ROSA_SEEG_DATASET not set; skipping dataset-mode check")

        from rosa_agent.commands.pipeline import run_pipeline

        # T22 is the pinned regression subject.
        try:
            summary = run_pipeline(
                "T22",
                out_dir=self.out_dir,
            )
        except SystemExit as exc:
            self.skipTest(f"T22 not available: {exc}")
        self.assertFalse(
            (self.out_dir / "ct.nii.gz").exists(),
            "Dataset mode must NOT copy the CT — there's no value in "
            "duplicating a NIfTI the user already has on disk",
        )
        self.assertTrue((self.out_dir / "trajectories.tsv").exists())
        self.assertTrue((self.out_dir / "contacts.tsv").exists())
        self.assertEqual(summary["n_trajectories"], 9)


if __name__ == "__main__":
    unittest.main()
