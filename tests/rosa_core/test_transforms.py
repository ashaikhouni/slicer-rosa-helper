import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))

from rosa_core.transforms import (  # noqa: E402
    apply_affine,
    identity_4x4,
    invert_4x4,
    is_identity_4x4,
    lps_to_ras_matrix,
    lps_to_ras_point,
    matmul_4x4,
    to_itk_affine_text,
)


class TransformTests(unittest.TestCase):
    def test_apply_affine(self):
        m = [
            [1.0, 0.0, 0.0, 5.0],
            [0.0, 2.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        self.assertEqual(apply_affine(m, [1.0, 2.0, 3.0]), [6.0, 3.0, 6.0])

    def test_invert_matrix_round_trip(self):
        m = [
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 2.0, 0.0, -4.0],
            [0.0, 0.0, 0.5, 6.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        inv = invert_4x4(m)
        prod = matmul_4x4(m, inv)
        self.assertTrue(is_identity_4x4(prod, tol=1e-8))

    def test_invert_singular_matrix_raises(self):
        singular = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        with self.assertRaisesRegex(ValueError, "singular"):
            invert_4x4(singular)

    def test_lps_to_ras_helpers(self):
        self.assertEqual(lps_to_ras_point([1.0, 2.0, 3.0]), [-1.0, -2.0, 3.0])
        m_lps = [
            [1.0, 0.0, 0.0, 5.0],
            [0.0, 1.0, 0.0, 6.0],
            [0.0, 0.0, 1.0, 7.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        m_ras = lps_to_ras_matrix(m_lps)
        # Translation x/y flips sign in RAS.
        self.assertEqual(m_ras[0][3], -5.0)
        self.assertEqual(m_ras[1][3], -6.0)
        self.assertEqual(m_ras[2][3], 7.0)

    def test_itk_affine_text(self):
        text = to_itk_affine_text(identity_4x4())
        self.assertIn("Transform: AffineTransform_double_3_3", text)
        self.assertIn("Parameters:", text)
        self.assertIn("FixedParameters: 0 0 0", text)


if __name__ == "__main__":
    unittest.main()
