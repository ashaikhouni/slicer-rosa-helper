import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "shank_core"))

from shank_core.masking import (  # noqa: E402
    build_head_mask_kji,
    build_outside_air_mask_kji,
    build_preview_masks,
)
from _utils import make_synthetic_ct_with_line  # noqa: E402


class ShankMaskingTests(unittest.TestCase):
    def test_outside_air_excludes_internal_air_pocket(self):
        arr = np.full((32, 32, 32), -1000.0, dtype=np.float32)
        arr[6:26, 6:26, 6:26] = 40.0
        arr[14:18, 14:18, 14:18] = -1000.0  # enclosed air pocket

        outside = build_outside_air_mask_kji(arr, air_threshold_hu=-500.0)
        self.assertTrue(outside[0, 0, 0])  # border air is outside
        self.assertFalse(outside[15, 15, 15])  # enclosed pocket is not outside-air

    def test_head_mask_outside_air_method(self):
        arr = np.full((40, 40, 40), -1000.0, dtype=np.float32)
        arr[8:32, 8:32, 8:32] = 40.0
        mask = build_head_mask_kji(
            arr_kji=arr,
            spacing_xyz=(1.0, 1.0, 1.0),
            threshold_hu=-500.0,
            method="outside_air",
        )
        self.assertTrue(mask[20, 20, 20])
        self.assertFalse(mask[0, 0, 0])

    def test_preview_masks_depth_gating_reduces_points(self):
        arr = make_synthetic_ct_with_line(shape=(64, 64, 64))
        spacing = (1.0, 1.0, 1.0)
        preview_no_depth = build_preview_masks(
            arr_kji=arr,
            spacing_xyz=spacing,
            threshold=1800.0,
            use_head_mask=True,
            build_head_mask=True,
            head_mask_threshold_hu=-500.0,
            head_mask_method="outside_air",
            min_metal_depth_mm=0.0,
            max_metal_depth_mm=220.0,
        )
        preview_depth = build_preview_masks(
            arr_kji=arr,
            spacing_xyz=spacing,
            threshold=1800.0,
            use_head_mask=True,
            build_head_mask=True,
            head_mask_threshold_hu=-500.0,
            head_mask_method="outside_air",
            min_metal_depth_mm=5.0,
            max_metal_depth_mm=220.0,
        )
        self.assertGreater(preview_no_depth["candidate_count"], 0)
        self.assertLessEqual(preview_depth["depth_kept_count"], preview_no_depth["depth_kept_count"])
        self.assertLessEqual(preview_no_depth["depth_kept_count"], preview_no_depth["metal_in_head_count"])


if __name__ == "__main__":
    unittest.main()
