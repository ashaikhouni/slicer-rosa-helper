"""Pin atlas-base registration in LabelmapAtlasProvider.

The 'FreeSurfer parcellation lives in T1 RAS but contacts live in CT
RAS' problem: contacts sampled directly against the parcellation hit
the wrong labels. The provider's atlas_base_path / target_volume_path
flags rigidly register the atlas-base T1 to the target CT and resample
the labelmap, so contacts in CT RAS get correctly-labeled samples.

Synthetic test:
  * Build a target volume with two distinct labeled regions (synthetic
    contacts at known positions inside each).
  * Build a 'parcellation' labelmap in the SAME world frame with the
    same labels at those positions.
  * Build an 'atlas base T1' that lives in a SHIFTED frame — origin
    offset by a known translation. Apply the SAME shift to the
    parcellation so atlas_base ↔ parcellation stay paired.
  * Construct the provider WITHOUT atlas_base/target args → labels at
    the contact positions should be wrong (because the parcellation
    is in the shifted frame).
  * Construct the provider WITH atlas_base/target args → registration
    should recover the shift; labels at the contact positions should
    match the target frame's intent.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "cli"))
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))


def _try_imports():
    try:
        import numpy  # noqa: F401
        import nibabel  # noqa: F401
        import SimpleITK  # noqa: F401
        from rosa_agent.services.atlas_provider_headless import LabelmapAtlasProvider  # noqa: F401
        return True
    except ImportError:
        return False


DEPS_AVAILABLE = _try_imports()


def _write_nifti(arr_ijk, affine, path: Path):
    import nibabel as nib
    import numpy as np
    nib.Nifti1Image(np.asarray(arr_ijk), np.asarray(affine, dtype=float)).to_filename(str(path))


@unittest.skipUnless(
    DEPS_AVAILABLE,
    "numpy / nibabel / SimpleITK / rosa_agent not importable",
)
class AtlasBaseRegistrationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

        import numpy as np
        # Target volume: 80³ with two high-contrast cubes at known
        # positions. Treat these positions as where the synthetic
        # contacts live. 80³ leaves headroom for the +20 mm shift below
        # so both cubes stay in-bounds in the shifted frame too —
        # otherwise the registration loses gradient information.
        size = 80
        target_arr = np.zeros((size, size, size), dtype=np.float32)
        target_arr[15:25, 15:25, 15:25] = 1000.0
        target_arr[40:50, 40:50, 40:50] = 1500.0
        # Identity affine → world coords match voxel indices.
        target_affine = np.eye(4)
        target_affine[3, 3] = 1.0
        self.target_path = self.tmp / "target.nii.gz"
        _write_nifti(target_arr, target_affine, self.target_path)

        # Parcellation: same shape, label 17 in the first region, 18 in
        # the second. Lives in the SHIFTED frame: origin offset +20 in i.
        # 20 mm is twice the 10-mm cube extent, so the query point at
        # world (20, 20, 20) falls cleanly OUTSIDE the labeled region
        # in the un-registered frame (proves the test exercises real
        # misalignment).
        label_arr = np.zeros((size, size, size), dtype=np.int16)
        label_arr[15:25, 15:25, 15:25] = 17
        label_arr[40:50, 40:50, 40:50] = 18
        shift_affine = np.eye(4)
        shift_affine[0, 3] = 20.0
        shift_affine[3, 3] = 1.0
        self.parc_path = self.tmp / "parcellation.nii.gz"
        _write_nifti(label_arr, shift_affine, self.parc_path)

        # Atlas-base T1: same intensity pattern as the target so MI
        # registration has signal to match, but in the same shifted
        # frame as the parcellation.
        self.t1_path = self.tmp / "t1_base.nii.gz"
        _write_nifti(target_arr, shift_affine, self.t1_path)

        # Sample point: the center of the first labeled cube. In the
        # target world frame that's RAS=(20, 20, 20).
        self.point_in_target_frame = (20.0, 20.0, 20.0)
        self.expected_label_value = 17

    def tearDown(self):
        self._tmp.cleanup()

    def test_without_registration_misses_label(self):
        """Sampling the un-registered (shifted) parcellation at a
        target-frame contact position should hit the WRONG label —
        proves the test setup is actually exercising the misalignment.
        """
        from rosa_agent.services.atlas_provider_headless import LabelmapAtlasProvider

        prov = LabelmapAtlasProvider(
            source_id="freesurfer", display_name="FS",
            label_path=self.parc_path,
            label_names={17: "Left-Hippocampus", 18: "Left-Amygdala"},
        )
        sample = prov.sample_contact(self.point_in_target_frame)
        # Without registration, the +5-mm shift means the queried point
        # (20, 20, 20) doesn't sit at the center of cube 17 in the
        # parcellation's frame. distance_to_voxel_mm SHOULD be > 0
        # (we're querying a point that's offset from the cube).
        self.assertGreater(
            sample["distance_to_voxel_mm"], 1.0,
            "Without registration the labelmap should be misaligned — "
            "if this passes we lost the +5 mm shift in setUp",
        )

    def test_with_registration_recovers_label(self):
        """Sampling the registered+resampled parcellation at the same
        contact position should hit the CORRECT label (label 17,
        distance well under 1 voxel).
        """
        from rosa_agent.services.atlas_provider_headless import LabelmapAtlasProvider

        log_lines: list[str] = []
        prov = LabelmapAtlasProvider(
            source_id="freesurfer", display_name="FS",
            label_path=self.parc_path,
            label_names={17: "Left-Hippocampus", 18: "Left-Amygdala"},
            atlas_base_path=self.t1_path,
            target_volume_path=self.target_path,
            logger=log_lines.append,
        )
        sample = prov.sample_contact(self.point_in_target_frame)
        self.assertEqual(
            sample["label_value"], self.expected_label_value,
            f"Expected label 17 after registration; got "
            f"{sample['label_value']!r}. Log: {log_lines}",
        )
        self.assertEqual(sample["label"], "Left-Hippocampus")
        self.assertLess(
            sample["distance_to_voxel_mm"], 1.0,
            f"Distance to nearest labeled voxel {sample['distance_to_voxel_mm']:.2f} "
            f"mm > 1 voxel — registration didn't recover the shift cleanly. "
            f"Log: {log_lines}",
        )

        # Logger should have surfaced both registration milestones and
        # the resample sink. Pin the basic shape so a regression in the
        # message format gets caught.
        log_text = "\n".join(log_lines)
        self.assertIn("registering", log_text)
        self.assertIn("resampled labelmap", log_text)


@unittest.skipUnless(
    DEPS_AVAILABLE,
    "numpy / nibabel / SimpleITK / rosa_agent not importable",
)
class NoScipyFallbackTests(unittest.TestCase):
    """Pin the no-scipy nearest-neighbor fallback in atlas_provider_headless.

    The provider's ``_build_kdtree`` prefers ``scipy.spatial.cKDTree`` and
    falls back to brute-force NumPy when scipy isn't installed. A 2026-05-02
    refactor (extracting shared helpers into rosa_core.atlas_index) dropped
    the ``import math`` that the fallback's ``math.sqrt`` call still needed,
    so any sample_contact() call in a no-scipy environment would NameError.

    We force the fallback path by hiding scipy from the import system, then
    exercise sample_contact() to confirm it produces a sensible answer.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

        import numpy as np
        # Tiny labelmap with two labels at known positions.
        arr = np.zeros((10, 10, 10), dtype=np.int16)
        arr[2:5, 2:5, 2:5] = 17
        arr[6:9, 6:9, 6:9] = 18
        affine = np.eye(4, dtype=float)
        affine[3, 3] = 1.0
        self.label_path = self.tmp / "labelmap.nii.gz"
        _write_nifti(arr, affine, self.label_path)

    def tearDown(self):
        self._tmp.cleanup()

    def test_fallback_query_works_without_scipy(self):
        import sys
        from rosa_agent.services import atlas_provider_headless

        # Block scipy so ``_build_kdtree``'s ``from scipy.spatial import
        # cKDTree`` raises ImportError and the fallback path runs. Save
        # and restore any pre-imported scipy modules so we don't poison
        # the test process.
        saved = {
            name: sys.modules[name]
            for name in list(sys.modules)
            if name == "scipy" or name.startswith("scipy.")
        }
        for name in saved:
            del sys.modules[name]
        sys.modules["scipy"] = None  # type: ignore[assignment]
        sys.modules["scipy.spatial"] = None  # type: ignore[assignment]
        try:
            prov = atlas_provider_headless.LabelmapAtlasProvider(
                source_id="freesurfer", display_name="FS",
                label_path=self.label_path,
                label_names={17: "Left-Hippo", 18: "Left-Amyg"},
            )
            sample = prov.sample_contact([3.0, 3.0, 3.0])
        finally:
            sys.modules.pop("scipy", None)
            sys.modules.pop("scipy.spatial", None)
            sys.modules.update(saved)

        self.assertEqual(sample["label_value"], 17)
        self.assertEqual(sample["label"], "Left-Hippo")
        # Querying at the cube center should give zero distance.
        self.assertAlmostEqual(sample["distance_to_voxel_mm"], 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
