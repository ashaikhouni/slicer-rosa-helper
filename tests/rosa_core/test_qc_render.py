"""Tests for the registration-QC slice renderer (rosa_core.qc_render).

Synthetic CT + MRI on a shared grid → each mode yields a valid PNG; a grid
mismatch or bad axis raises. Guarded on nibabel (an engine dep).
"""
from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import nibabel as nib
    from rosa_core.qc_render import render_registration_qc
    HAVE = True
except Exception:  # noqa: BLE001
    HAVE = False


def _png_dims(b: bytes):
    assert b[:8] == b"\x89PNG\r\n\x1a\n", "bad PNG signature"
    # IHDR width/height are the first two big-endian uint32 after the 8-byte
    # signature + 4-byte length + 4-byte "IHDR".
    w, h = struct.unpack(">II", b[16:24])
    return w, h


@unittest.skipUnless(HAVE, "nibabel/numpy unavailable")
class QcRenderTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        d = Path(self.td.name)
        shape = (40, 48, 44)
        rng = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
        ct = np.zeros(shape, np.float32); ct[10:30, 12:36, 8:38] = 1000.0 + rng[10:30, 12:36, 8:38] % 200
        mri = np.zeros(shape, np.float32); mri[10:30, 12:36, 8:38] = 300.0 + rng[10:30, 12:36, 8:38] % 90
        aff = np.eye(4)
        self.ct = d / "ct.nii.gz"; nib.save(nib.Nifti1Image(ct, aff), self.ct)
        self.mri = d / "mri.nii.gz"; nib.save(nib.Nifti1Image(mri, aff), self.mri)

    def tearDown(self):
        self.td.cleanup()

    def test_each_mode_and_axis_renders_valid_png(self):
        for mode in ("color", "opacity", "wipe", "checker", "ct", "mri"):
            for axis in (0, 1, 2):
                png = render_registration_qc(self.ct, self.mri, axis=axis,
                                             frac=0.5, mode=mode)
                w, h = _png_dims(png)
                self.assertGreater(w, 0)
                self.assertGreater(h, 0)

    def test_opacity_value_changes_output(self):
        a = render_registration_qc(self.ct, self.mri, mode="opacity", value=0.0)
        b = render_registration_qc(self.ct, self.mri, mode="opacity", value=1.0)
        self.assertNotEqual(a, b)   # 0 = CT only, 1 = MRI only

    def test_wipe_direction_changes_output(self):
        h = render_registration_qc(self.ct, self.mri, mode="wipe", value=0.5, direction="h")
        v = render_registration_qc(self.ct, self.mri, mode="wipe", value=0.5, direction="v")
        self.assertNotEqual(h, v)

    def test_frac_clamped(self):
        # Out-of-range fracs should not raise (clamped to the end slices).
        for f in (-1.0, 0.0, 1.0, 2.0):
            self.assertTrue(render_registration_qc(self.ct, self.mri, frac=f))

    def test_grid_mismatch_raises(self):
        d = Path(self.td.name)
        other = d / "small.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((10, 10, 10), np.float32), np.eye(4)), other)
        with self.assertRaises(ValueError):
            render_registration_qc(self.ct, other)

    def test_bad_axis_raises(self):
        with self.assertRaises(ValueError):
            render_registration_qc(self.ct, self.mri, axis=5)


if __name__ == "__main__":
    unittest.main()
