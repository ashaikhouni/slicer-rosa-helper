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
        surface_from_mask, gyral_mask_from_mri, gyral_surface_from_mri,
        transform_surface, BrainSurface, atlas_vertex_colors,
    )
    HAVE_DEPS = True
except Exception:  # noqa: BLE001
    HAVE_DEPS = False


@unittest.skipUnless(HAVE_DEPS, "numpy/nibabel/scikit-image (the [mesh] extra) unavailable")
class AtlasVertexColorsTests(unittest.TestCase):
    """Color a surface by a warped atlas labelmap: distinct color per region,
    neutral gray where the atlas has no label (honest coverage, no fill)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        arr = np.zeros((20, 20, 20), np.int32)
        arr[2:8, 2:8, 2:8] = 5
        arr[12:18, 12:18, 12:18] = 42
        self.lm = Path(self._tmp.name) / "atlas.nii.gz"
        nib.save(nib.Nifti1Image(arr, np.eye(4)), str(self.lm))
        # vertices: in region 5, in region 42, in unlabeled space.
        self.verts = np.array([[4., 4., 4.], [15., 15., 15.], [10., 0., 0.]], float)

    def tearDown(self):
        self._tmp.cleanup()

    def test_distinct_colors_and_gray_where_uncovered(self):
        cols = atlas_vertex_colors(self.verts, str(self.lm))
        self.assertEqual(cols.shape, (3, 4))
        self.assertEqual(cols.dtype, np.uint8)
        self.assertEqual(tuple(cols[2][:3]), (150, 150, 150))  # uncovered → gray
        self.assertFalse(np.array_equal(cols[0][:3], cols[1][:3]))  # regions differ
        self.assertFalse(np.array_equal(cols[0][:3], (150, 150, 150)))  # labeled ≠ gray

    def test_palette_override(self):
        cols = atlas_vertex_colors(self.verts, str(self.lm), colors={5: (255, 0, 0)})
        self.assertEqual(tuple(cols[0][:3]), (255, 0, 0))


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
        a = np.asanyarray(gyral_mask_from_mri(self.vol, self.mask, bias_correct=False).dataobj)
        b = np.asanyarray(gyral_mask_from_mri(vimg, mimg, bias_correct=False).dataobj)
        self.assertTrue(np.array_equal(a, b))

    def test_gyral_surface_iso_is_watertight_and_placed(self):
        # The grayscale iso-surface path: a valid placed BrainSurface whose
        # largest-component filter leaves a single closed shell (the phantom's
        # bright core), not internal fragments. Bias correction off (uniform
        # phantom → N4 is a no-op but slow/edge-casey on a synthetic).
        surf = gyral_surface_from_mri(
            self.vol, self.mask, step_size=1, taubin_iterations=2, bias_correct=False)
        self.assertIsInstance(surf, BrainSurface)
        self.assertGreater(surf.n_vertices, 100)
        self.assertEqual(_boundary_edge_count(surf.faces), 0)   # single closed shell
        self.assertTrue(surf.faces.max() < surf.n_vertices)
        # Surface sits around the GM/WM core (centered on the phantom).
        centroid = surf.vertices_ras.mean(axis=0)
        world_center = nib.affines.apply_affine(self.affine, np.array([24, 24, 24], float))
        self.assertLess(float(np.linalg.norm(centroid - world_center)), 6.0)

    def test_fastsurfer_tissue_support(self):
        from rosa_core.brain_mesh import brain_tissue_from_fastsurfer_aseg
        # Synthetic aseg: WM core (2), GM shell (ctx 1001), CSF rim (24), bg (0).
        n = 48
        zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
        r = np.sqrt((xx - 24) ** 2 + (yy - 24) ** 2 + (zz - 24) ** 2)
        aseg = np.zeros((n, n, n), np.int32)
        aseg[r <= 20] = 24      # CSF rim
        aseg[r <= 17] = 1001    # cortex (GM)
        aseg[r <= 10] = 2       # WM core
        aff = np.diag([1.0, 1.0, 1.0, 1.0]); aff[:3, 3] = [-24, -24, -24]
        aimg = nib.Nifti1Image(aseg, aff)
        # A matching T1 so gyral_surface_from_mri can mesh the intensity iso.
        t1 = np.where(aseg == 2, 240.0, np.where(aseg == 1001, 120.0,
             np.where(aseg == 24, 30.0, 0.0))).astype(np.float32)
        td = tempfile.TemporaryDirectory(); self.addCleanup(td.cleanup)
        vol = Path(td.name) / "t1.nii.gz"; mask = Path(td.name) / "m.nii.gz"
        nib.save(nib.Nifti1Image(t1, aff), str(vol))
        nib.save(nib.Nifti1Image((r <= 20).astype(np.uint8), aff), str(mask))

        tissue = brain_tissue_from_fastsurfer_aseg(aimg, aimg)
        self.assertTrue(tissue[aseg == 2].all() and tissue[aseg == 1001].all())  # GM+WM kept
        self.assertFalse(tissue[aseg == 24].any())                               # CSF dropped
        self.assertFalse(tissue[aseg == 0].any())                                # bg dropped
        # Feeding it as the support yields a valid placed surface (N4 off — the
        # phantom is uniform so N4 can be edge-casey on a synthetic).
        surf = gyral_surface_from_mri(vol, mask, step_size=1, taubin_iterations=2,
                                      bias_correct=False, brain_tissue=tissue)
        self.assertGreater(surf.n_vertices, 100)
        self.assertEqual(_boundary_edge_count(surf.faces), 0)

        # aparc_vertex_colors: color the surface by the (synthetic) parcellation.
        from rosa_core.brain_mesh import aparc_vertex_colors
        lut = {1001: {"rgba": (160, 100, 50, 0)}, 2: {"rgba": (245, 245, 245, 0)}}
        cols = aparc_vertex_colors(surf.vertices_ras, aimg, lut)
        self.assertEqual(cols.shape, (surf.n_vertices, 4))
        self.assertEqual(cols.dtype, np.uint8)
        self.assertTrue((cols[:, 3] == 255).all())          # opaque
        # A real LUT color (cortex or WM) lands on the surface — not all default.
        present = {tuple(c) for c in cols[:, :3].tolist()}
        self.assertTrue((160, 100, 50) in present or (245, 245, 245) in present)

    def test_deepmriprep_tissue_support(self):
        # deepmriprep p0 tissue label (0=bg,1=CSF,2=GM,3=WM) → GM+WM support, the
        # same brain_tissue hook the FastSurfer aseg feeds.
        from rosa_core.brain_mesh import brain_tissue_from_tissue_labelmap
        n = 48
        zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
        r = np.sqrt((xx - 24) ** 2 + (yy - 24) ** 2 + (zz - 24) ** 2)
        p0 = np.zeros((n, n, n), np.float32)
        p0[r <= 20] = 1.0    # CSF
        p0[r <= 17] = 2.0    # GM
        p0[r <= 10] = 3.0    # WM
        aff = np.diag([1.0, 1.0, 1.0, 1.0]); aff[:3, 3] = [-24, -24, -24]
        pimg = nib.Nifti1Image(p0, aff)
        tissue = brain_tissue_from_tissue_labelmap(pimg, pimg)   # default gm_wm_min=1.5
        self.assertTrue(tissue[p0 == 3.0].all() and tissue[p0 == 2.0].all())  # GM+WM kept
        self.assertFalse(tissue[p0 == 1.0].any())                            # CSF dropped
        self.assertFalse(tissue[p0 == 0.0].any())                            # bg dropped
        # Resamples onto a differing reference grid without error.
        ref = nib.Nifti1Image(np.zeros((n, n, n), np.uint8),
                              np.diag([1.0, 1.0, 1.0, 1.0]))
        t2 = brain_tissue_from_tissue_labelmap(pimg, ref)
        self.assertEqual(t2.shape, (n, n, n))


@unittest.skipUnless(HAVE_DEPS, "numpy/nibabel/scikit-image (the [mesh] extra) unavailable")
class TransformSurfaceTests(unittest.TestCase):
    """transform_surface: push a native-frame surface into the CT frame."""
    def _ball_surface(self):
        td = tempfile.TemporaryDirectory(); self.addCleanup(td.cleanup)
        p = Path(td.name) / "b.nii.gz"
        _write_ball(p, n=48, center=(24, 24, 24), radius=12)
        return surface_from_mask(p)

    def test_rigid_transform_moves_and_preserves_shape(self):
        surf = self._ball_surface()
        # Rigid: 90° about z + translation. Shape (extent) must be preserved,
        # centroid must move by the translation (after rotation).
        c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        t = np.array([10.0, -5.0, 3.0])
        M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t
        moved = transform_surface(surf, M)
        # Same vertex/face count; extent (a rotation-invariant of a ball) preserved.
        self.assertEqual(moved.n_vertices, surf.n_vertices)
        self.assertTrue(np.array_equal(moved.faces, surf.faces))
        ext0 = surf.vertices_ras.max(0) - surf.vertices_ras.min(0)
        ext1 = moved.vertices_ras.max(0) - moved.vertices_ras.min(0)
        self.assertTrue(np.allclose(np.sort(ext0), np.sort(ext1), atol=1e-3))
        # Centroid maps by M.
        c0 = surf.vertices_ras.mean(0)
        expect = R @ c0 + t
        self.assertTrue(np.allclose(moved.vertices_ras.mean(0), expect, atol=1e-2))
        # Normals stay unit length in the new frame.
        lens = np.linalg.norm(moved.vertex_normals, axis=1)
        self.assertTrue(np.allclose(lens, 1.0, atol=1e-3))

    def test_identity_is_noop(self):
        surf = self._ball_surface()
        same = transform_surface(surf, np.eye(4))
        self.assertTrue(np.allclose(same.vertices_ras, surf.vertices_ras, atol=1e-4))


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
