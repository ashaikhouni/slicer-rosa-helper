"""Unit tests for rosa_core.brain_mesh — mask -> RAS-mm surface via marching cubes.

The synthetic-sphere tests are self-contained (a ball written to a temp NIfTI),
so they run wherever scikit-image is installed and skip cleanly where it isn't
(the [mesh] extra). They check the load-bearing properties: watertightness,
correct world placement via the affine, unit normals, and in-range faces. An
opt-in T22 test runs the real cohort mask when the (gitignored) volume is local.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))

try:
    import numpy as np
    import nibabel as nib
    import skimage  # noqa: F401  — the [mesh] extra
    from rosa_core.brain_mesh import (
        surface_from_mask, gyral_mask_from_mri, BrainSurface,
    )
    HAVE_DEPS = True
except Exception:  # noqa: BLE001
    HAVE_DEPS = False


def _boundary_edge_count(faces):
    """0 for a closed (watertight) surface: every edge shared by exactly 2 faces."""
    e = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _uniq, cnt = np.unique(e, axis=0, return_counts=True)
    return int((cnt != 2).sum())


def _write_ball(path, *, n=64, center=(32, 32, 32), radius=18, affine=None):
    """Write a solid binary ball to a NIfTI and return its affine."""
    if affine is None:
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
        affine[:3, 3] = [-32.0, -40.0, -20.0]  # arbitrary non-trivial origin
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    ball = ((xx - center[0]) ** 2 + (yy - center[1]) ** 2
            + (zz - center[2]) ** 2) <= radius ** 2
    nib.save(nib.Nifti1Image(ball.astype(np.uint8), affine), str(path))
    return affine


@unittest.skipUnless(HAVE_DEPS, "numpy/nibabel/scikit-image (the [mesh] extra) unavailable")
class SyntheticBallTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.mask = Path(self.td.name) / "ball.nii.gz"
        self.center = (32, 30, 34)
        self.radius = 18
        self.affine = _write_ball(self.mask, n=64, center=self.center, radius=self.radius)

    def tearDown(self):
        self.td.cleanup()

    def test_returns_watertight_placed_surface(self):
        surf = surface_from_mask(self.mask)
        self.assertIsInstance(surf, BrainSurface)
        self.assertGreater(surf.n_vertices, 100)
        self.assertGreater(surf.n_faces, 100)

        # Watertight — a solid ball meshes to a closed surface.
        self.assertEqual(_boundary_edge_count(surf.faces), 0)

        # Faces index valid vertices; arrays are the right dtype/shape.
        self.assertEqual(surf.vertices_ras.shape[1], 3)
        self.assertEqual(surf.faces.shape[1], 3)
        self.assertTrue(surf.faces.min() >= 0)
        self.assertTrue(surf.faces.max() < surf.n_vertices)

        # Normals are (approximately) unit length.
        lens = np.linalg.norm(surf.vertex_normals, axis=1)
        self.assertTrue(np.allclose(lens, 1.0, atol=1e-3))

    def test_world_placement_matches_affine(self):
        surf = surface_from_mask(self.mask, smooth_sigma=1.0, taubin_iterations=8)
        world_center = nib.affines.apply_affine(self.affine, np.array(self.center, float))
        centroid = surf.vertices_ras.mean(axis=0)
        # Centroid of a ball's surface sits at its center; smoothing keeps it
        # centered. Tolerance covers voxel/smoothing slack.
        self.assertLess(float(np.linalg.norm(centroid - world_center)), 3.0)
        # Extent along each axis ~ 2*radius (mm; 1 mm spacing), within slack.
        extent = surf.vertices_ras.max(0) - surf.vertices_ras.min(0)
        self.assertTrue(np.all(np.abs(extent - 2 * self.radius) < 6.0))

    def test_largest_component_drops_speckle(self):
        # Add a tiny detached blob far from the ball; largest_component should
        # keep only the ball, so the centroid stays put.
        img = nib.load(str(self.mask))
        arr = np.asanyarray(img.dataobj).copy()
        arr[2:4, 2:4, 2:4] = 1  # speckle in the corner
        nib.save(nib.Nifti1Image(arr, img.affine), str(self.mask))
        surf = surface_from_mask(self.mask, largest_component=True)
        world_center = nib.affines.apply_affine(self.affine, np.array(self.center, float))
        centroid = surf.vertices_ras.mean(axis=0)
        self.assertLess(float(np.linalg.norm(centroid - world_center)), 3.0)

    def test_empty_mask_raises(self):
        empty = Path(self.td.name) / "empty.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((16, 16, 16), np.uint8), np.eye(4)), str(empty))
        with self.assertRaises(ValueError):
            surface_from_mask(empty)


@unittest.skipUnless(HAVE_DEPS, "numpy/nibabel/scikit-image (the [mesh] extra) unavailable")
class GyralMaskTests(unittest.TestCase):
    """gyral_mask_from_mri: Otsu-drop CSF from a T1 inside the brain mask.

    A synthetic 3-tissue phantom (dark CSF shell / mid GM / bright WM core)
    inside a ball mask must keep GM+WM and drop the CSF, so the result is a
    subset of the mask that still contains the bright core.
    """
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        n = 48
        zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
        r = np.sqrt((xx - 24) ** 2 + (yy - 24) ** 2 + (zz - 24) ** 2)
        # Concentric intensity shells: CSF (dark) outside, GM mid, WM bright core.
        t1 = np.zeros((n, n, n), np.float32)
        t1[r <= 20] = 30.0    # CSF-like rim
        t1[r <= 15] = 120.0   # GM-like
        t1[r <= 8] = 240.0    # WM-like core
        mask = (r <= 20).astype(np.uint8)
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
        affine[:3, 3] = [-24.0, -24.0, -24.0]
        self.vol = Path(self.td.name) / "t1.nii.gz"
        self.mask = Path(self.td.name) / "mask.nii.gz"
        nib.save(nib.Nifti1Image(t1, affine), str(self.vol))
        nib.save(nib.Nifti1Image(mask, affine), str(self.mask))
        self.affine = affine

    def tearDown(self):
        self.td.cleanup()

    def test_drops_csf_keeps_gm_wm(self):
        img = gyral_mask_from_mri(self.vol, self.mask)
        gm = np.asanyarray(img.dataobj) > 0
        full = np.asanyarray(nib.load(str(self.mask)).dataobj) > 0
        # A strict subset of the brain mask (the dark CSF rim is dropped)...
        self.assertTrue(gm[full].all() or gm.sum() < full.sum())
        self.assertLess(int(gm.sum()), int(full.sum()))
        # ...that still contains the bright WM core (never dropped).
        n = gm.shape[0]
        zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
        core = np.sqrt((xx - 24) ** 2 + (yy - 24) ** 2 + (zz - 24) ** 2) <= 6
        self.assertTrue(gm[core].all())
        # Meshable with folds preserved (low smoothing).
        surf = surface_from_mask(img, smooth_sigma=0.5, taubin_iterations=4)
        self.assertGreater(surf.n_vertices, 100)

    def test_accepts_nibabel_images(self):
        # The viewer passes nibabel images (no temp file); path and image agree.
        vimg = nib.load(str(self.vol))
        mimg = nib.load(str(self.mask))
        a = np.asanyarray(gyral_mask_from_mri(self.vol, self.mask).dataobj)
        b = np.asanyarray(gyral_mask_from_mri(vimg, mimg).dataobj)
        self.assertTrue(np.array_equal(a, b))


_T22_MASK = REPO_ROOT / "tests" / "data" / "T22" / "T22_brain_mask_ct.nii.gz"


@unittest.skipUnless(HAVE_DEPS and _T22_MASK.is_file(),
                     "needs scikit-image and the (gitignored) T22 mask — local only")
class RealMaskTests(unittest.TestCase):
    def test_t22_mask_yields_watertight_brain_surface(self):
        surf = surface_from_mask(_T22_MASK)
        self.assertEqual(_boundary_edge_count(surf.faces), 0)
        # A whole-brain surface: tens of thousands of verts, ~14 cm extent.
        self.assertGreater(surf.n_vertices, 5000)
        extent = surf.vertices_ras.max(0) - surf.vertices_ras.min(0)
        self.assertTrue(np.all(extent > 80.0) and np.all(extent < 220.0))


if __name__ == "__main__":
    unittest.main()
