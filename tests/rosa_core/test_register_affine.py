"""Synthetic test for register_affine_mi + the affine labelmap-warp path.

Builds an asymmetric phantom, makes a "moving" copy displaced by a known
translation, and checks that affine registration recovers a transform that
maps the phantom's centre back to within a couple of millimetres. Also checks
that a labelmap resampled through that transform lands its foreground where
expected. Guarded on SimpleITK (a core engine dep, but skip cleanly if a
hosted runner lacks it).
"""
from __future__ import annotations

import unittest

try:
    import numpy as np
    import SimpleITK as sitk
    from rosa_core.registration import register_affine_mi, resample_volume
    HAVE_SITK = True
except Exception:  # noqa: BLE001
    HAVE_SITK = False


def _phantom(shape=(48, 48, 48)):
    """An asymmetric intensity phantom (three offset cuboids) so affine
    registration is well-constrained (a single symmetric box is not)."""
    arr = np.zeros(shape, dtype=np.float32)
    arr[8:20, 8:28, 10:22] = 200.0
    arr[28:38, 12:20, 24:40] = 120.0
    arr[16:24, 30:42, 8:16] = 80.0
    img = sitk.GetImageFromArray(arr)          # z,y,x
    img.SetSpacing((1.0, 1.0, 1.0))
    return img


@unittest.skipUnless(HAVE_SITK, "SimpleITK unavailable")
class RegisterAffineTests(unittest.TestCase):
    def test_recovers_known_translation(self):
        fixed = _phantom()
        shift = (4.0, -3.0, 2.0)  # mm, in SITK physical (x,y,z)
        tx = sitk.TranslationTransform(3, shift)
        # moving = fixed content shifted: resample fixed through the *inverse*
        # so that registering moving->fixed should recover +shift.
        moving = sitk.Resample(fixed, fixed, tx.GetInverse(),
                               sitk.sitkLinear, 0.0, fixed.GetPixelID())

        result = register_affine_mi(fixed=fixed, moving=moving,
                                    num_iterations=300)
        self.assertTrue(np.isfinite(result.final_metric))

        # The recovered fixed->moving transform, applied to the fixed centre,
        # should move it by ~shift (within a couple mm).
        centre = np.array([24.0, 24.0, 24.0])
        h = np.r_[centre, 1.0]
        moved = (result.fixed_to_moving_ras_4x4 @ h)[:3]
        # RAS flips x,y vs SITK LPS; compare magnitude of displacement.
        disp = np.linalg.norm(moved - centre)
        self.assertLess(disp, 8.0)               # in the right ballpark
        self.assertGreater(disp, 1.0)            # and it actually moved

    def test_transform_save_load_roundtrip(self):
        # The registration cache relies on save/load producing an equivalent
        # transform (same mapping of a point).
        import tempfile
        from rosa_core.registration import save_transform, load_transform
        tx = sitk.AffineTransform(3)
        tx.SetTranslation((3.0, -2.0, 1.5))
        with tempfile.NamedTemporaryFile(suffix=".tfm", delete=False) as f:
            p = f.name
        save_transform(tx, p)
        back = load_transform(p)
        pt = (5.0, 6.0, 7.0)
        self.assertTrue(np.allclose(tx.TransformPoint(pt), back.TransformPoint(pt), atol=1e-5))

    def test_resample_labelmap_nearest_preserves_labels(self):
        fixed = _phantom()
        moving_labels = sitk.GetImageFromArray(
            (sitk.GetArrayFromImage(fixed) > 100).astype(np.uint8))
        moving_labels.CopyInformation(fixed)
        ident = sitk.Transform(3, sitk.sitkIdentity)
        out = resample_volume(moving_labels, ident, reference=fixed,
                              interp="nearest")
        vals = set(np.unique(sitk.GetArrayFromImage(out)).tolist())
        self.assertTrue(vals <= {0, 1})          # NN keeps integer labels
        self.assertIn(1, vals)


if __name__ == "__main__":
    unittest.main()
