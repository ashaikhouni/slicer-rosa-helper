import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "shank_core"))

from shank_core.detect import detect_from_preview  # noqa: E402
from shank_core.masking import build_preview_masks  # noqa: E402
from shank_core.pipeline import run_detection  # noqa: E402
from _utils import (  # noqa: E402
    kji_to_ras_points_identity,
    make_synthetic_ct_with_line,
    ras_to_ijk_identity,
)


class ShankDetectPipelineTests(unittest.TestCase):
    def test_detect_from_preview_depth_values_follow_subsampling(self):
        arr = make_synthetic_ct_with_line(shape=(64, 64, 64))
        preview = build_preview_masks(
            arr_kji=arr,
            spacing_xyz=(1.0, 1.0, 1.0),
            threshold=1800.0,
            use_head_mask=True,
            build_head_mask=True,
            head_mask_threshold_hu=-500.0,
            head_mask_method="outside_air",
            min_metal_depth_mm=0.0,
            max_metal_depth_mm=220.0,
        )
        result = detect_from_preview(
            arr_kji=arr,
            spacing_xyz=(1.0, 1.0, 1.0),
            preview=preview,
            ijk_kji_to_ras_fn=kji_to_ras_points_identity,
            ras_to_ijk_fn=ras_to_ijk_identity,
            center_ras=np.array([32.0, 32.0, 32.0], dtype=float),
            max_points=10,
            max_lines=2,
            min_inliers=1000,  # force no line fit
            ransac_iterations=50,
        )
        self.assertEqual(result["in_mask_points_ras"].shape[0], 10)
        self.assertEqual(result["in_mask_depth_values_mm"].shape[0], 10)
        self.assertEqual(len(result["lines"]), 0)

    def test_run_detection_smoke(self):
        arr = make_synthetic_ct_with_line(shape=(64, 64, 64))
        result = run_detection(
            arr_kji=arr,
            spacing_xyz=(1.0, 1.0, 1.0),
            threshold=1800.0,
            ijk_kji_to_ras_fn=kji_to_ras_points_identity,
            ras_to_ijk_fn=ras_to_ijk_identity,
            center_ras=np.array([32.0, 32.0, 32.0], dtype=float),
            max_points=300000,
            max_lines=5,
            inlier_radius_mm=1.2,
            min_length_mm=10.0,
            min_inliers=20,
            ransac_iterations=120,
            use_head_mask=True,
            build_head_mask=True,
            head_mask_threshold_hu=-500.0,
            head_mask_method="outside_air",
            min_metal_depth_mm=0.0,
            max_metal_depth_mm=220.0,
        )
        self.assertIn("lines", result)
        self.assertIn("profile_ms", result)
        self.assertGreaterEqual(len(result["lines"]), 1)
        line = result["lines"][0]
        self.assertIn("start_ras", line)
        self.assertIn("end_ras", line)
        self.assertGreater(line["length_mm"], 10.0)


if __name__ == "__main__":
    unittest.main()
