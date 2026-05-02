"""Pin load_rosa_volume_as_sitk on a synthetic ROSA case.

Builds a tiny .ros file pointing at two synthetic Analyze volumes (one
'reference', one 'native') with a known TRdicomRdisplay matrix between
them. Loading the native volume must produce a SITK image whose
IJK->RAS matrix carries the composed display->reference transform.
"""

from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))


def _try_imports() -> bool:
    try:
        import numpy  # noqa: F401
        import SimpleITK  # noqa: F401
        from rosa_core import load_rosa_volume_as_sitk  # noqa: F401
        return True
    except ImportError:
        return False


DEPS_AVAILABLE = _try_imports()


def _write_synthetic_analyze(path: Path, *, size=(8, 9, 10), spacing=(1.0, 1.0, 1.0)):
    """Write a tiny Analyze (.img/.hdr) volume at ``path.with_suffix('.img')``."""
    import numpy as np
    import SimpleITK as sitk

    arr = np.zeros(size, dtype=np.float32)
    arr[2:5, 3:6, 4:7] = 100.0  # Some signal so the volume isn't pure zero.
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(tuple(float(s) for s in spacing))
    sitk.WriteImage(img, str(path.with_suffix(".img")))
    return img


def _build_synthetic_rosa_case(case_dir: Path) -> Path:
    """Lay out the minimum on-disk shape load_rosa_volume_as_sitk needs.

    Two displays:
      * ``ref_vol``: identity TRdicomRdisplay.
      * ``native_vol``: TRdicomRdisplay = translate +5/+0/+0 in LPS.

    Returns the .ros path.
    """
    analyze_root = case_dir / "DICOM" / "uid_a"
    analyze_root.mkdir(parents=True, exist_ok=True)
    _write_synthetic_analyze(analyze_root / "ref_vol")
    _write_synthetic_analyze(analyze_root / "native_vol")

    # Minimal ROS file: two TRdicomRdisplay entries, two VOLUMES, two
    # IMAGERY_NAME entries, two IMAGERY_3DREF entries (native_vol points
    # at ref_vol as its parent display).
    ros_text = textwrap.dedent("""
        [TRdicomRdisplay]
        1 0 0 0
        0 1 0 0
        0 0 1 0
        0 0 0 1
        [VOLUME]
        DICOM/uid_a/ref_vol
        [TRdicomRdisplay]
        1 0 0 5
        0 1 0 0
        0 0 1 0
        0 0 0 1
        [VOLUME]
        DICOM/uid_a/native_vol
        [IMAGERY_NAME]
        ref_vol
        [IMAGERY_NAME]
        native_vol
        [SERIE_UID]
        uid_a
        [SERIE_UID]
        uid_a
        [IMAGERY_3DREF]
        0
        [IMAGERY_3DREF]
        0
        [TRAJECTORY]
        traj1
        T1 1 0 0 -1.0 -2.0 3.0 0 -10.0 -20.0 30.0
        [END]
    """).strip()
    ros_path = case_dir / "case.ros"
    ros_path.write_text(ros_text)
    return ros_path


@unittest.skipUnless(
    DEPS_AVAILABLE,
    "numpy/SimpleITK/rosa_core not importable in this environment.",
)
class LoadRosaVolumeAsSitkTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.case_dir = Path(self._tmp.name)
        _build_synthetic_rosa_case(self.case_dir)

    def tearDown(self):
        self._tmp.cleanup()

    def test_loads_reference_volume_with_centered_geometry(self):
        """Reference volume — no display transform, but the IJK->RAS
        gets centered (image center voxel sits at world origin) to
        match Slicer's CaseLoaderService behavior. ROSA-planned
        trajectories live in this brain-centered frame, NOT the
        on-disk image-corner-origin frame.
        """
        import numpy as np
        from rosa_core import load_rosa_volume_as_sitk
        from shank_core.io import image_ijk_ras_matrices

        img, meta = load_rosa_volume_as_sitk(
            str(self.case_dir), volume_name="ref_vol",
        )
        self.assertEqual(meta["loaded_volume"], "ref_vol")
        self.assertEqual(meta["reference_volume"], "ref_vol")
        self.assertTrue(meta["is_reference"])
        self.assertEqual(meta["display_index"], 0)
        np.testing.assert_allclose(
            np.asarray(meta["display_to_reference_ras"]), np.eye(4),
        )

        # _write_synthetic_analyze uses size=(8, 9, 10) at unit spacing.
        # Centered IJK->RAS: image-center IJK = ((size-1)/2). With
        # SITK's LPS-flipped on-disk geometry the post-centering RAS
        # origin lands at (-(size-1)/2 * spacing, ...). The image center
        # sampled in the new frame must be exactly (0, 0, 0).
        ijk_to_ras, _ = image_ijk_ras_matrices(img)
        size = img.GetSize()
        center_ijk_h = np.array([
            (size[0] - 1) * 0.5,
            (size[1] - 1) * 0.5,
            (size[2] - 1) * 0.5,
            1.0,
        ])
        center_ras = ijk_to_ras @ center_ijk_h
        np.testing.assert_allclose(
            center_ras[:3], [0.0, 0.0, 0.0], atol=1e-6,
            err_msg=(
                "Reference volume's image-center voxel should land at "
                "world origin after centering; got {}".format(center_ras[:3])
            ),
        )

    def test_loads_native_volume_with_composed_transform(self):
        """Native (non-reference) volume — centered first, then the
        display->reference transform is composed on top.
        """
        import numpy as np
        from rosa_core import load_rosa_volume_as_sitk
        from shank_core.io import image_ijk_ras_matrices

        img, meta = load_rosa_volume_as_sitk(
            str(self.case_dir), volume_name="native_vol",
        )
        self.assertEqual(meta["loaded_volume"], "native_vol")
        self.assertFalse(meta["is_reference"])
        self.assertEqual(meta["display_index"], 1)

        # The .ros file specified TRdicomRdisplay = translate +5 in
        # LPS-X for native_vol relative to ref_vol. In RAS that's a -5
        # in R (LPS-L = +X). The composed display->ref matrix in RAS
        # form must reflect that flipped sign.
        m_ras = np.asarray(meta["display_to_reference_ras"])
        np.testing.assert_allclose(m_ras[:3, 3], [-5.0, 0.0, 0.0])
        np.testing.assert_allclose(m_ras[:3, :3], np.eye(3))

        # native_vol is the same size as ref_vol with unit spacing. After
        # centering its image-center voxel sits at native-RAS (0, 0, 0).
        # Then the display transform shifts that point by (-5, 0, 0).
        # So in the post-stamp IJK->RAS the image-center voxel should
        # land at world (-5, 0, 0).
        ijk_to_ras, _ = image_ijk_ras_matrices(img)
        size = img.GetSize()
        center_ijk_h = np.array([
            (size[0] - 1) * 0.5,
            (size[1] - 1) * 0.5,
            (size[2] - 1) * 0.5,
            1.0,
        ])
        center_ras = ijk_to_ras @ center_ijk_h
        np.testing.assert_allclose(
            center_ras[:3], [-5.0, 0.0, 0.0], atol=1e-6,
            err_msg=(
                "Native vol's center should land at the display->ref "
                "translation in world frame; got {}".format(center_ras[:3])
            ),
        )

    def test_invert_flag_inverts_composed_transform(self):
        import numpy as np
        from rosa_core import load_rosa_volume_as_sitk

        _, meta_native = load_rosa_volume_as_sitk(
            str(self.case_dir), volume_name="native_vol",
        )
        _, meta_inverted = load_rosa_volume_as_sitk(
            str(self.case_dir), volume_name="native_vol", invert=True,
        )
        m1 = np.asarray(meta_native["display_to_reference_ras"])
        m2 = np.asarray(meta_inverted["display_to_reference_ras"])
        # m2 should equal inverse(m1).
        np.testing.assert_allclose(m1 @ m2, np.eye(4), atol=1e-6)

    def test_trajectories_are_lps_and_ras_converted(self):
        from rosa_core import load_rosa_volume_as_sitk

        _, meta = load_rosa_volume_as_sitk(str(self.case_dir))
        trajs = meta["trajectories"]
        self.assertEqual(len(trajs), 1)
        t = trajs[0]
        # The .ros TRAJECTORY had start LPS=(-1, -2, 3) → RAS=(1, 2, 3).
        self.assertEqual(t["name"], "T1")
        self.assertEqual(t["start_lps"], [-1.0, -2.0, 3.0])
        self.assertEqual(t["start_ras"], [1.0, 2.0, 3.0])
        self.assertEqual(t["end_ras"], [10.0, 20.0, 30.0])

    def test_unknown_volume_name_errors(self):
        from rosa_core import load_rosa_volume_as_sitk

        with self.assertRaises(ValueError) as ctx:
            load_rosa_volume_as_sitk(str(self.case_dir), volume_name="not_a_volume")
        self.assertIn("not_a_volume", str(ctx.exception))

    def test_non_unit_spacing_volume_stamps_correct_geometry(self):
        """Pin the bug the s57 ROSA case caught: when a non-reference
        display volume has non-unit spacing (e.g. 0.415mm CT), the
        composed display->reference matrix must be applied to the
        IJK->RAS matrix BEFORE stamping — not directly substituted as
        the IJK->RAS matrix. The previous version conflated the two
        and produced direction columns off by 1/spacing.
        """
        import numpy as np
        import SimpleITK as sitk
        import textwrap
        from shank_core.io import image_ijk_ras_matrices
        from rosa_core import load_rosa_volume_as_sitk

        # Tiny separate case with a non-unit-spacing native volume.
        analyze_root = self.case_dir / "DICOM" / "uid_b"
        analyze_root.mkdir(parents=True, exist_ok=True)
        # Reference vol: unit spacing (matches the existing setUp ref_vol
        # path we already use for the simpler tests).
        _write_synthetic_analyze(analyze_root / "ref_vol2", size=(8, 9, 10))
        # Native vol: 0.5 mm spacing (so the spacing-divide bug would
        # produce direction columns 2x what they should be).
        _write_synthetic_analyze(analyze_root / "fine_vol", size=(8, 9, 10), spacing=(0.5, 0.5, 0.5))

        ros_text = textwrap.dedent("""
            [TRdicomRdisplay]
            1 0 0 0
            0 1 0 0
            0 0 1 0
            0 0 0 1
            [VOLUME]
            DICOM/uid_b/ref_vol2
            [TRdicomRdisplay]
            1 0 0 7
            0 1 0 0
            0 0 1 0
            0 0 0 1
            [VOLUME]
            DICOM/uid_b/fine_vol
            [IMAGERY_NAME]
            ref_vol2
            [IMAGERY_NAME]
            fine_vol
            [SERIE_UID]
            uid_b
            [SERIE_UID]
            uid_b
            [IMAGERY_3DREF]
            0
            [IMAGERY_3DREF]
            0
        """).strip()
        (self.case_dir / "case.ros").write_text(ros_text)

        img, meta = load_rosa_volume_as_sitk(
            str(self.case_dir), volume_name="fine_vol",
        )

        # The post-stamp IJK->RAS for fine_vol applies BOTH centering
        # (image-center voxel -> world origin) AND the display->ref
        # translation (-7 in R axis). Verify two invariants:
        #   1. Direction columns reflect spacing correctly (the
        #      conflation bug would produce 2x magnitudes here).
        #   2. The image-center voxel lands at the display->ref world
        #      translation point (-7, 0, 0) — proves both centering AND
        #      composition are wired correctly.
        ijk_to_ras_after, _ = image_ijk_ras_matrices(img)

        # Direction columns: SITK on-disk header is +x/+y/+z LPS at
        # 0.5mm. After LPS->RAS flip, native IJK->RAS direction is
        # diag(-0.5, -0.5, +0.5). The display->ref transform is pure
        # translation, so direction is unchanged.
        expected_dir = np.diag([-0.5, -0.5, 0.5])
        np.testing.assert_allclose(
            ijk_to_ras_after[:3, :3], expected_dir, atol=1e-6,
            err_msg=(
                "Stamped IJK->RAS direction drifted. Old buggy code "
                "produced columns 2x the correct magnitude (1/spacing "
                "scaling error)."
            ),
        )

        # Image-center voxel location in world after the full
        # centering + display->ref composition.
        size = img.GetSize()
        center_ijk_h = np.array([
            (size[0] - 1) * 0.5,
            (size[1] - 1) * 0.5,
            (size[2] - 1) * 0.5,
            1.0,
        ])
        center_ras = ijk_to_ras_after @ center_ijk_h
        np.testing.assert_allclose(
            center_ras[:3], [-7.0, 0.0, 0.0], atol=1e-6,
            err_msg=(
                "Image-center voxel should land at display->ref world "
                "translation; got {}".format(center_ras[:3])
            ),
        )


if __name__ == "__main__":
    unittest.main()
