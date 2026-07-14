"""Unit tests for ``rosa_core.thomas_import.build_thomas_labelmap``.

Builds a tiny synthetic THOMAS output tree (``left/`` + ``right/`` per-nucleus
masks + a reference ``T1.nii.gz``) and checks the combined labelmap + LUT:
left = THOMAS#, right = THOMAS# + 100, with the ``-L`` / ``-R`` naming; and the
aggregate masks (whole THALAMUS, VL, GP) are excluded so sub-nuclei aren't
enclosed.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


def _write_mask(path: Path, affine: np.ndarray, corner=(1, 1, 1)) -> None:
    import nibabel as nib
    x, y, z = corner
    m = np.zeros((16, 10, 10), dtype=np.uint8)
    m[x:x + 2, y:y + 2, z:z + 2] = 1
    nib.save(nib.Nifti1Image(m, affine), str(path))


class BuildThomasLabelmapTests(unittest.TestCase):
    def _synthetic_dir(self, root: Path) -> None:
        import nibabel as nib
        aff = np.diag([1.0, 1.0, 1.0, 1.0])
        # left hemisphere at low x, right at high x — so they don't overwrite.
        for hemi, x0 in (("left", 1), ("right", 10)):
            d = root / hemi
            d.mkdir(parents=True)
            _write_mask(d / "11-CM.nii.gz", aff, corner=(x0, 1, 1))
            _write_mask(d / "8-Pul.nii.gz", aff, corner=(x0, 5, 1))
            # aggregates that must be EXCLUDED (they enclose sub-nuclei)
            _write_mask(d / "1-THALAMUS.nii.gz", aff, corner=(x0, 1, 1))
            _write_mask(d / "4567-VL.nii.gz", aff, corner=(x0, 5, 1))
            _write_mask(d / "6_VLPd.nii.gz", aff, corner=(x0, 3, 1))   # '_' subdivision, not '-'
        nib.save(nib.Nifti1Image(np.zeros((16, 10, 10), np.int16), aff),
                 str(root / "T1.nii.gz"))

    def test_combined_labelmap_and_lut(self):
        from rosa_core.thomas_import import build_thomas_labelmap
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._synthetic_dir(root)
            img, lut, t1 = build_thomas_labelmap(root)

            labels = {int(v) for v in np.unique(np.asanyarray(img.dataobj)) if v}
            # only the canonical nuclei: CM(11) + Pul(8), left=#, right=#+100.
            # The aggregates (1, 4567) and the '_' subdivision (6) are excluded.
            self.assertEqual(labels, {8, 11, 108, 111})

            self.assertEqual(lut[11][0], "CM-L")
            self.assertEqual(lut[111][0], "CM-R")
            self.assertEqual(lut[8][0], "Pul-L")
            # left/right of a nucleus share a color (anatomy convention)
            self.assertEqual(lut[11][1], lut[111][1])

            self.assertIsNotNone(t1)
            self.assertEqual(Path(t1).name, "T1.nii.gz")

    def test_missing_masks_raises(self):
        from rosa_core.thomas_import import build_thomas_labelmap
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                build_thomas_labelmap(Path(td))   # empty dir, no left/ or right/


if __name__ == "__main__":
    unittest.main()
