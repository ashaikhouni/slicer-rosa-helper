import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))

from shank_core.io import (  # noqa: E402
    image_ijk_ras_matrices,
    kji_to_ras_points,
    kji_to_ras_points_matrix,
    ras_to_ijk_float_matrix,
    read_volume,
    write_mask_like,
    write_points_csv,
)


class ShankIOTests(unittest.TestCase):
    def _make_reference_image(self):
        arr = np.zeros((8, 9, 10), dtype=np.int16)  # KJI
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((0.5, 1.0, 2.0))
        img.SetOrigin((10.0, 20.0, 30.0))  # LPS
        img.SetDirection((1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0))
        return img

    def test_kji_ras_matrix_conversions(self):
        img = self._make_reference_image()
        kji = np.array([[3, 4, 5]], dtype=float)

        ras1 = kji_to_ras_points(img, kji)[0]
        # kji=[3,4,5] -> ijk=[5,4,3] -> lps=[12.5,24,36] -> ras=[-12.5,-24,36]
        self.assertAlmostEqual(float(ras1[0]), -12.5, places=6)
        self.assertAlmostEqual(float(ras1[1]), -24.0, places=6)
        self.assertAlmostEqual(float(ras1[2]), 36.0, places=6)

        m_ijk_to_ras, m_ras_to_ijk = image_ijk_ras_matrices(img)
        ras2 = kji_to_ras_points_matrix(kji, m_ijk_to_ras)[0]
        self.assertTrue(np.allclose(ras1, ras2, atol=1e-6))

        ijk = ras_to_ijk_float_matrix(ras1, m_ras_to_ijk)
        self.assertTrue(np.allclose(ijk, [5.0, 4.0, 3.0], atol=1e-6))

    def test_read_and_write_helpers(self):
        img = self._make_reference_image()
        with tempfile.TemporaryDirectory() as td:
            ct_path = Path(td) / "ct.nii.gz"
            sitk.WriteImage(img, str(ct_path))

            ref_img, arr_kji, spacing = read_volume(str(ct_path))
            self.assertEqual(arr_kji.shape, (8, 9, 10))
            self.assertEqual(tuple(float(v) for v in spacing), (0.5, 1.0, 2.0))

            mask = np.zeros_like(arr_kji, dtype=np.uint8)
            mask[1:3, 2:4, 3:5] = 1
            mask_path = Path(td) / "mask.nii.gz"
            write_mask_like(ref_img, mask, str(mask_path))
            self.assertTrue(mask_path.exists())

            points_path = Path(td) / "pts.csv"
            write_points_csv(str(points_path), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            with points_path.open("r", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows[0], ["x", "y", "z"])
            self.assertEqual(rows[1], ["1.000000", "2.000000", "3.000000"])


if __name__ == "__main__":
    unittest.main()
