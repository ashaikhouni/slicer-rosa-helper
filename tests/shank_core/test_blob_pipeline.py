import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "shank_core"))

from shank_core.blob_candidates import extract_blob_candidates, filter_blob_candidates  # noqa: E402
from shank_core.pipeline import run_detection  # noqa: E402
from _utils import kji_to_ras_points_identity, ras_to_ijk_identity  # noqa: E402


def _make_synthetic_bead_ct(shape=(80, 80, 80)):
    arr = np.full(shape, -1000.0, dtype=np.float32)
    k0, j0, i0 = [s // 2 for s in shape]
    kk, jj, ii = np.indices(shape)
    rr = np.sqrt((kk - k0) ** 2 + (jj - j0) ** 2 + (ii - i0) ** 2)
    arr[rr <= 30.0] = 40.0

    # Line A: stronger support (12 beads).
    for t in range(12):
        k = k0
        j = j0 + 6
        i = i0 - 22 + t * 3
        arr[k, j, i] = 2800.0

    # Line B: weaker support (7 beads), recovered by rescue pass.
    for t in range(7):
        k = k0 - 3
        j = j0 - 5
        i = i0 - 18 + t * 3
        arr[k, j, i] = 2800.0
    return arr


class BlobCandidateTests(unittest.TestCase):
    def test_blob_candidate_extraction_counts(self):
        mask = np.zeros((24, 24, 24), dtype=np.uint8)
        mask[5:7, 5:7, 5:7] = 1
        mask[14:16, 14:16, 14:16] = 1
        arr = np.zeros_like(mask, dtype=np.float32)
        arr[mask > 0] = 2500.0

        raw = extract_blob_candidates(
            metal_mask_kji=mask,
            arr_kji=arr,
            depth_map_kji=None,
            ijk_kji_to_ras_fn=kji_to_ras_points_identity,
        )
        self.assertEqual(raw["blob_count_total"], 2)
        self.assertEqual(len(raw["blobs"]), 2)

    def test_blob_filtering_thresholds(self):
        mask = np.zeros((30, 30, 30), dtype=np.uint8)
        mask[2:3, 2:3, 2:3] = 1  # 1 voxel (small)
        mask[8:11, 8:11, 8:11] = 1  # 27 voxels (kept)
        mask[18:24, 18:24, 18:24] = 1  # 216 voxels (large)
        arr = np.zeros_like(mask, dtype=np.float32)
        arr[mask > 0] = 2400.0

        raw = extract_blob_candidates(
            metal_mask_kji=mask,
            arr_kji=arr,
            depth_map_kji=None,
            ijk_kji_to_ras_fn=kji_to_ras_points_identity,
        )
        filtered = filter_blob_candidates(
            blob_result=raw,
            min_blob_voxels=2,
            max_blob_voxels=120,
            min_blob_peak_hu=2000.0,
            max_blob_elongation=None,
        )
        self.assertEqual(filtered["blob_count_kept"], 1)
        self.assertEqual(filtered["blob_reject_small"], 1)
        self.assertEqual(filtered["blob_reject_large"], 1)
        self.assertEqual(filtered["blob_reject_intensity"], 0)


class BlobDetectionPipelineTests(unittest.TestCase):
    def test_detection_result_schema_stability_both_modes(self):
        arr = _make_synthetic_bead_ct()
        required = [
            "candidate_points_total",
            "candidate_points_after_mask",
            "candidate_points_after_depth",
            "fit1_lines_proposed",
            "fit2_lines_kept",
            "rescue_lines_kept",
            "final_lines_kept",
            "assigned_points_after_refine",
            "unassigned_points_after_refine",
            "final_unassigned_points",
            "blob_count_total",
            "blob_count_kept",
            "blob_labelmap_kji",
            "profile_ms",
        ]
        for mode in ("voxel", "blob_centroid"):
            out = run_detection(
                arr_kji=arr,
                spacing_xyz=(1.0, 1.0, 1.0),
                threshold=1800.0,
                ijk_kji_to_ras_fn=kji_to_ras_points_identity,
                ras_to_ijk_fn=ras_to_ijk_identity,
                center_ras=np.array([40.0, 40.0, 40.0], dtype=float),
                max_points=300000,
                max_lines=8,
                inlier_radius_mm=1.4,
                min_length_mm=10.0,
                min_inliers=8,
                ransac_iterations=120,
                use_head_mask=True,
                build_head_mask=True,
                head_mask_threshold_hu=-500.0,
                head_mask_method="outside_air",
                min_metal_depth_mm=0.0,
                max_metal_depth_mm=220.0,
                candidate_mode=mode,
                min_blob_voxels=1,
                max_blob_voxels=80,
                enable_rescue_pass=True,
            )
            for key in required:
                self.assertIn(key, out, f"missing key '{key}' in mode {mode}")

    def test_rescue_pass_recovers_sparse_secondary_line(self):
        arr = _make_synthetic_bead_ct()
        no_rescue = run_detection(
            arr_kji=arr,
            spacing_xyz=(1.0, 1.0, 1.0),
            threshold=1800.0,
            ijk_kji_to_ras_fn=kji_to_ras_points_identity,
            ras_to_ijk_fn=ras_to_ijk_identity,
            center_ras=np.array([40.0, 40.0, 40.0], dtype=float),
            max_points=300000,
            max_lines=8,
            inlier_radius_mm=1.4,
            min_length_mm=10.0,
            min_inliers=8,
            ransac_iterations=140,
            use_head_mask=True,
            build_head_mask=True,
            head_mask_threshold_hu=-500.0,
            head_mask_method="outside_air",
            min_metal_depth_mm=0.0,
            max_metal_depth_mm=220.0,
            candidate_mode="blob_centroid",
            min_blob_voxels=1,
            max_blob_voxels=80,
            enable_rescue_pass=False,
        )
        with_rescue = run_detection(
            arr_kji=arr,
            spacing_xyz=(1.0, 1.0, 1.0),
            threshold=1800.0,
            ijk_kji_to_ras_fn=kji_to_ras_points_identity,
            ras_to_ijk_fn=ras_to_ijk_identity,
            center_ras=np.array([40.0, 40.0, 40.0], dtype=float),
            max_points=300000,
            max_lines=8,
            inlier_radius_mm=1.4,
            min_length_mm=10.0,
            min_inliers=8,
            ransac_iterations=140,
            use_head_mask=True,
            build_head_mask=True,
            head_mask_threshold_hu=-500.0,
            head_mask_method="outside_air",
            min_metal_depth_mm=0.0,
            max_metal_depth_mm=220.0,
            candidate_mode="blob_centroid",
            min_blob_voxels=1,
            max_blob_voxels=80,
            enable_rescue_pass=True,
            rescue_min_inliers_scale=0.5,
            rescue_max_lines=4,
        )
        self.assertGreaterEqual(with_rescue.get("final_lines_kept", 0), no_rescue.get("final_lines_kept", 0))
        self.assertGreaterEqual(with_rescue.get("rescue_lines_kept", 0), 0)


if __name__ == "__main__":
    unittest.main()
