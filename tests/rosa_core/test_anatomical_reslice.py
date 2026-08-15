"""``rosa_core.anatomical_reslice`` — reformat a volume along anatomical axes
*in place* (same world, rotated grid) so overlays stay consistent.

Locks the geometry: the rotation is stripped of scale/shear, the output grid's
voxel axes are the anatomical axes (Rᵀ), and the physical world is preserved so a
bright landmark keeps its world position.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "CommonLib"))


def _deps() -> bool:
    try:
        import numpy  # noqa: F401
        import SimpleITK  # noqa: F401
        from rosa_core.anatomical_reslice import reslice_along_anatomical_axes  # noqa: F401
        return True
    except Exception:
        return False


DEPS = _deps()


def _vox_to_ras(img):
    """vox→RAS for a SITK image, mirroring export_view._slice_volume_meta."""
    import numpy as np
    sp = img.GetSpacing(); o = img.GetOrigin()
    d = np.asarray(img.GetDirection(), float).reshape(3, 3)
    d_ras = d.copy(); d_ras[0, :] *= -1; d_ras[1, :] *= -1
    M = np.eye(4)
    for k in range(3):
        M[:3, k] = d_ras[:, k] * sp[k]
    M[:3, 3] = [-o[0], -o[1], o[2]]
    return M


@unittest.skipUnless(DEPS, "numpy/SimpleITK/rosa_core not importable.")
class RotationTests(unittest.TestCase):
    def test_strips_scale_and_shear_keeps_rotation(self):
        import numpy as np
        from rosa_core.anatomical_reslice import rigid_rotation_from_affine
        th = 0.3; c, s = np.cos(th), np.sin(th)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        M = np.eye(4); M[:3, :3] = Rz @ np.diag([1.3, 0.7, 1.1])   # rotation + anisotropic scale
        R = rigid_rotation_from_affine(M)
        self.assertTrue(np.allclose(R @ R.T, np.eye(3), atol=1e-6))   # orthonormal
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=6)
        self.assertTrue(np.allclose(R, Rz, atol=0.05))               # ≈ the true rotation
        # a pure rotation is returned exactly
        M2 = np.eye(4); M2[:3, :3] = Rz
        self.assertTrue(np.allclose(rigid_rotation_from_affine(M2), Rz, atol=1e-6))


@unittest.skipUnless(DEPS, "numpy/SimpleITK/rosa_core not importable.")
class ResliceTests(unittest.TestCase):
    def _blob(self, direction=None):
        import numpy as np, SimpleITK as sitk
        arr = np.zeros((30, 30, 30), np.int16); arr[10, 12, 14] = 1000   # z,y,x
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((1.0, 1.0, 1.0))
        if direction:
            img.SetDirection(direction)
        return img

    def test_identity_axis_aligns_and_preserves_world(self):
        import numpy as np, SimpleITK as sitk
        from rosa_core.anatomical_reslice import reslice_along_anatomical_axes
        th = 0.26; c, s = np.cos(th), np.sin(th)
        img = self._blob(direction=[c, -s, 0, s, c, 0, 0, 0, 1])       # tilted acquisition
        W = img.TransformIndexToPhysicalPoint((14, 12, 10))            # world (LPS) of the bright voxel
        out = reslice_along_anatomical_axes(img, np.eye(4))            # already MNI-aligned → standard axes
        d = np.asarray(out.GetDirection(), float).reshape(3, 3)
        self.assertTrue(np.allclose(np.abs(d), np.eye(3), atol=1e-6))  # output grid is axis-aligned
        oa = sitk.GetArrayFromImage(out)
        z, y, x = np.unravel_index(int(np.argmax(oa)), oa.shape)
        W2 = out.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
        self.assertLess(float(np.linalg.norm(np.asarray(W2) - np.asarray(W))), 2.0)  # world preserved

    def test_output_axes_equal_R_transpose(self):
        import numpy as np
        from rosa_core.anatomical_reslice import reslice_along_anatomical_axes, rigid_rotation_from_affine
        img = self._blob()                                            # identity direction
        th = 0.4; c, s = np.cos(th), np.sin(th)
        M = np.eye(4); M[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]  # CT→MNI rotation
        out = reslice_along_anatomical_axes(img, M)
        R = rigid_rotation_from_affine(M)
        vt = _vox_to_ras(out)[:3, :3]                                 # spacing 1 → unit columns
        self.assertTrue(np.allclose(vt, R.T, atol=1e-5))              # anatomical axes = Rᵀ


if __name__ == "__main__":
    unittest.main()
