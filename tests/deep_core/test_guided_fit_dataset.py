"""End-to-end regression for the guided-fit engine.

The guided-fit code path (``rosa_detect.guided_fit_engine``) is what
PostopCT's "Guided Fit" button + the ROSA-folder mode of the CLI agent
both call. It shares ``compute_features`` and the SITK-from-volume
geometry helpers with Auto Fit, so any silent drift in the LPS-flip
machinery (``rosa_detect.service._apply_slicer_geometry_to_sitk`` /
``_load_image_and_matrices``) would mis-localize every guided fit even
when Auto Fit still passes.

Per ``feedback_cli_slicer_parity.md``: same input must give the same
detection. This test pins:

  - that ``compute_features`` runs cleanly on a real subject CT and
    emits all the feature volumes the fit engine needs,
  - that ``fit_trajectory`` recovers each GT shank when seeded with the
    GT axis itself (the easiest possible case — anything below this
    means the geometry path is broken),
  - that the fit endpoints are within tolerance of the seed (so an
    LPS-vs-RAS bug surfaces immediately).

T22 is the pinned regression subject (mirrors
``test_pipeline_dataset_contact_pitch_v1.test_T22``). Runtime ≈ 4 s.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

DATASET_ROOT = Path(
    os.environ.get(
        "ROSA_SEEG_DATASET",
        "/Users/ammar/Dropbox/thalamus_subjects/seeg_localization",
    )
)
SUBJECTS_TSV = DATASET_ROOT / "contact_label_dataset" / "subjects.tsv"
DATASET_AVAILABLE = SUBJECTS_TSV.exists()


def _try_imports() -> bool:
    try:
        import numpy  # noqa: F401
        import SimpleITK  # noqa: F401
        from rosa_detect import guided_fit_engine  # noqa: F401
        from rosa_detect.service import run_contact_pitch_v1  # noqa: F401
        return True
    except ImportError:
        return False


DEPS_AVAILABLE = _try_imports()


@unittest.skipUnless(
    DATASET_AVAILABLE,
    f"SEEG dataset not found at {SUBJECTS_TSV}. Set ROSA_SEEG_DATASET to override.",
)
@unittest.skipUnless(
    DEPS_AVAILABLE,
    "numpy/SimpleITK/rosa_detect not importable in this environment.",
)
class GuidedFitDatasetRegressionTests(unittest.TestCase):
    """Pins guided-fit behavior for the LPS-flip/feature-volume surface."""

    @classmethod
    def setUpClass(cls):
        import SimpleITK as sitk
        from eval_seeg_localization import (
            iter_subject_rows,
            load_reference_ground_truth_shanks,
        )
        from rosa_detect import guided_fit_engine as gfe
        from shank_core.io import image_ijk_ras_matrices

        rows = iter_subject_rows(DATASET_ROOT, {"T22"})
        assert len(rows) == 1, "T22 missing from dataset manifest"
        row = rows[0]
        cls.subject_id = "T22"
        cls.gt_shanks, _ = load_reference_ground_truth_shanks(row)
        assert len(cls.gt_shanks) == 9, (
            f"T22 GT count drifted from 9 (got {len(cls.gt_shanks)})"
        )

        ct_path = row["ct_path"]
        img = sitk.ReadImage(str(ct_path))
        ijk_to_ras, ras_to_ijk = image_ijk_ras_matrices(img)
        cls.features = gfe.compute_features(img, ijk_to_ras, ras_to_ijk)
        # ``compute_features`` may resample to canonical 1 mm; the
        # canonical matrices are stamped back on the dict and downstream
        # callers MUST use those, not the originals.
        cls.ijk_to_ras = cls.features["ijk_to_ras_mat"]
        cls.ras_to_ijk = cls.features["ras_to_ijk_mat"]

    def test_features_shape_and_keys(self):
        """compute_features must hand back every volume fit_trajectory consumes."""
        feats = self.features
        for key in (
            "img", "ijk_to_ras_mat", "ras_to_ijk_mat",
            "log", "frangi", "ct_arr_kji",
            "hull", "intracranial", "head_distance",
            "blob_pts_ras", "blob_amps", "bolts", "bolt_mask",
        ):
            self.assertIn(key, feats, f"compute_features missing {key!r}")
        # Blobs must be non-empty for a real subject CT.
        self.assertGreater(
            int(feats["blob_pts_ras"].shape[0]), 100,
            "blob cloud collapsed — likely LoG / canonicalization regression",
        )
        # T22 has 9 metallic bolts; we should detect most of them.
        self.assertGreaterEqual(
            len(list(feats["bolts"])), 7,
            "bolt extraction recall regressed on T22",
        )

    def test_each_gt_shank_recoverable_from_self_seed(self):
        """Seeding fit_trajectory with the GT axis must recover the shank.

        This is the strongest pin against silent geometry drift: if the
        canonical-grid IJK<->RAS matrices were broken, the wide-cylinder
        ROI would land on the wrong voxels and zero shanks would
        recover.
        """
        import numpy as np
        from rosa_detect import guided_fit_engine as gfe

        n_success = 0
        n_bolt_anchored = 0
        worst_lateral_mm = 0.0
        worst_angle_deg = 0.0

        for gt in self.gt_shanks:
            seed_start = np.asarray(gt.start_ras, dtype=float)
            seed_end = np.asarray(gt.end_ras, dtype=float)
            fit = gfe.fit_trajectory(
                seed_start, seed_end, self.features,
                self.ijk_to_ras, self.ras_to_ijk,
            )
            if not fit.get("success"):
                continue
            n_success += 1
            if bool(fit.get("bolt_anchored")):
                n_bolt_anchored += 1
            # Endpoint geometry should stay near the seed (axis prior is
            # strong; the fit should snap to nearby contacts, not jump).
            fit_start = np.asarray(fit["start_ras"], dtype=float)
            fit_end = np.asarray(fit["end_ras"], dtype=float)
            fit_axis = (fit_end - fit_start)
            fit_axis = fit_axis / max(1e-9, float(np.linalg.norm(fit_axis)))
            seed_axis = (seed_end - seed_start)
            seed_axis = seed_axis / max(1e-9, float(np.linalg.norm(seed_axis)))
            cos = float(np.clip(abs(np.dot(seed_axis, fit_axis)), 0.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(cos)))
            seed_mid = 0.5 * (seed_start + seed_end)
            fit_mid = 0.5 * (fit_start + fit_end)
            d = seed_mid - fit_mid
            perp = d - float(np.dot(d, fit_axis)) * fit_axis
            lateral_mm = float(np.linalg.norm(perp))
            worst_lateral_mm = max(worst_lateral_mm, lateral_mm)
            worst_angle_deg = max(worst_angle_deg, angle_deg)

        # Recovery budget: 8/9 on T22 today. The miss is LMFG, whose
        # manually-snapped GT axis is offset > 2 mm from the real shank
        # (see project_gt_snapping_probe.md) so the wide-cylinder ROI
        # doesn't pick up enough blobs. A regression below 8 means the
        # geometry path is broken, not the GT.
        self.assertGreaterEqual(
            n_success, 8,
            f"guided fit self-seed regressed: {n_success}/9 succeeded",
        )
        self.assertGreaterEqual(
            n_bolt_anchored, 7,
            f"guided-fit bolt anchor regressed: {n_bolt_anchored}/9 anchored",
        )
        # Geometry budget: self-seed fit must stay close to the seed.
        # Today's worst on T22: lateral 5.11 mm (LIFG), angle 5.10° (RCMN).
        # Both numbers reflect contact-cloud PCA drift, not bugs. The
        # threshold has to absorb that drift but stay tight enough that
        # an LPS<->RAS sign flip — which would put the fit on the
        # opposite hemisphere (60+ mm shift) — explodes the test.
        self.assertLess(
            worst_lateral_mm, 6.0,
            f"guided-fit lateral midpoint shift regressed: {worst_lateral_mm:.2f} mm",
        )
        self.assertLess(
            worst_angle_deg, 8.0,
            f"guided-fit axis tilt regressed: {worst_angle_deg:.2f} deg",
        )


class SitkGeometryParityTests(unittest.TestCase):
    """Pin the LPS-flip helper that bridges Slicer node matrices and SITK.

    ``_apply_slicer_geometry_to_sitk`` rewrites the SITK origin/direction
    so an image read from disk inherits a Slicer volume node's IJK->RAS
    matrix instead of the on-disk header's. Three callers depend on this
    being correct:

      - Auto Fit (Slicer side, when a registered ROSA volume is in scene)
      - Guided Fit (Slicer side)
      - ContactsTrajectoryView LoG cache

    The parity invariant: when the supplied ijk_to_ras matrix EQUALS what
    the SITK header already implies, the helper must be a no-op (the
    derived matrices round-trip back to the original within numeric
    tolerance). Anything else is a bug. This test exercises that on a
    synthetic image so it runs without the dataset.
    """

    def test_apply_geometry_is_noop_when_matrix_matches_header(self):
        try:
            import numpy as np
            import SimpleITK as sitk
            from shank_core.io import image_ijk_ras_matrices
            from rosa_detect.service import _apply_slicer_geometry_to_sitk
        except ImportError:
            self.skipTest("numpy/SimpleITK/rosa_detect not importable")

        # Synthetic 5x6x7 volume with a non-trivial spacing + LPS origin
        # and direction (positive identity in LPS).
        img = sitk.GetImageFromArray(
            np.zeros((7, 6, 5), dtype=np.float32)
        )
        img.SetSpacing((0.7, 0.8, 1.1))
        img.SetOrigin((-50.0, 30.0, -10.0))
        img.SetDirection((1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0))

        ijk_to_ras_before, _ = image_ijk_ras_matrices(img)

        # Apply the helper using the matrix the header already implies.
        _apply_slicer_geometry_to_sitk(img, ijk_to_ras_before)

        ijk_to_ras_after, _ = image_ijk_ras_matrices(img)
        np.testing.assert_allclose(
            ijk_to_ras_before, ijk_to_ras_after, atol=1e-6,
            err_msg=(
                "apply_slicer_geometry_to_sitk(self_matrix) must round-trip "
                "to the same IJK->RAS — found drift."
            ),
        )

    def test_apply_geometry_round_trips_through_node_matrix(self):
        """Stamp a custom ijk_to_ras, then derive matrices back; must match."""
        try:
            import numpy as np
            import SimpleITK as sitk
            from shank_core.io import image_ijk_ras_matrices
            from rosa_detect.service import _apply_slicer_geometry_to_sitk
        except ImportError:
            self.skipTest("numpy/SimpleITK/rosa_detect not importable")

        img = sitk.GetImageFromArray(np.zeros((4, 5, 6), dtype=np.float32))
        img.SetSpacing((1.0, 1.0, 1.0))

        # A "Slicer node" matrix with non-axis-aligned RAS direction.
        custom = np.array([
            [0.9, 0.1, 0.0, 12.5],
            [-0.1, 0.9, 0.0, -7.0],
            [0.0, 0.0, 1.0, 3.25],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)

        _apply_slicer_geometry_to_sitk(img, custom)

        derived, _ = image_ijk_ras_matrices(img)
        np.testing.assert_allclose(
            derived, custom, atol=1e-6,
            err_msg=(
                "After _apply_slicer_geometry_to_sitk(custom), the SITK "
                "image's derived IJK->RAS must equal the custom matrix."
            ),
        )


if __name__ == "__main__":
    unittest.main()
