"""Unit tests for ``rosa_core.contact_placement_v2``.

Covers the polyline + centerline helpers and the ``PlacementV2Result``
dataclass. Full-pipeline behavior (matched filter on dataset CT volumes)
is validated by the dataset regression in
``project_v2_pipeline_2026-05-07.md``.
"""
from __future__ import annotations

import unittest

try:
    import numpy as np
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


@unittest.skipUnless(DEPS_AVAILABLE, "numpy not available")
class PolylineHelperTests(unittest.TestCase):
    """Tests for the private polyline + centerline primitives."""

    def setUp(self):
        from rosa_core import contact_placement_v2 as v2
        self.v2 = v2

    def test_polyline_segments_drops_zero_length(self):
        """Consecutive duplicate points should be dropped from the polyline."""
        cl = np.array([[0, 0, 0], [0, 0, 0], [10, 0, 0]], dtype=float)
        starts, dirs, lens, _ = self.v2._polyline_segments(cl)
        self.assertEqual(len(starts), 1)
        np.testing.assert_allclose(dirs[0], [1, 0, 0])
        self.assertAlmostEqual(lens[0], 10.0)

    def test_polyline_segments_rejects_short_input(self):
        with self.assertRaises(ValueError):
            self.v2._polyline_segments(np.array([[0, 0, 0]], dtype=float))

    def test_polyline_segments_rejects_zero_length(self):
        cl = np.array([[1, 2, 3], [1, 2, 3]], dtype=float)
        with self.assertRaises(ValueError):
            self.v2._polyline_segments(cl)

    def test_polyline_pos_at_arc_endpoints_and_midpoint(self):
        """A 10mm straight polyline along +x: arc=0 → start, arc=5 → midpoint, arc=10 → end."""
        cl = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
        np.testing.assert_allclose(self.v2._polyline_pos_at_arc(cl, 0.0), [0, 0, 0])
        np.testing.assert_allclose(self.v2._polyline_pos_at_arc(cl, 5.0), [5, 0, 0])
        np.testing.assert_allclose(self.v2._polyline_pos_at_arc(cl, 10.0), [10, 0, 0])

    def test_polyline_pos_at_arc_clamps_to_endpoints(self):
        """arc < 0 returns start; arc > total returns end (no extrapolation)."""
        cl = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
        np.testing.assert_allclose(self.v2._polyline_pos_at_arc(cl, -1.0), [0, 0, 0])
        # Clamps past end (matched-filter caller may pass slot_arcs slightly
        # past cl_max; clamping keeps placement on the centerline).
        np.testing.assert_allclose(self.v2._polyline_pos_at_arc(cl, 12.0), [10, 0, 0])

    def test_polyline_pos_at_arc_curved(self):
        """Two-segment polyline: 0-10mm along +x, then 10-20mm along +y."""
        cl = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0]], dtype=float)
        np.testing.assert_allclose(self.v2._polyline_pos_at_arc(cl, 5.0), [5, 0, 0])
        np.testing.assert_allclose(self.v2._polyline_pos_at_arc(cl, 10.0), [10, 0, 0], atol=1e-9)
        np.testing.assert_allclose(self.v2._polyline_pos_at_arc(cl, 15.0), [10, 5, 0])

    def test_project_to_polyline_arc_orthogonal_drop(self):
        """A point off-axis projects to the closest arc on the centerline."""
        cl = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
        arc = self.v2._project_to_polyline_arc(cl, np.array([5.0, 1.0, 0.0]))
        self.assertAlmostEqual(arc, 5.0, places=4)

    def test_project_to_polyline_arc_clamps_to_endpoints(self):
        cl = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
        arc_before = self.v2._project_to_polyline_arc(cl, np.array([-5.0, 0.0, 0.0]))
        self.assertAlmostEqual(arc_before, 0.0)
        arc_after = self.v2._project_to_polyline_arc(cl, np.array([20.0, 0.0, 0.0]))
        self.assertAlmostEqual(arc_after, 10.0)

    def test_straight_centerline_has_n_points(self):
        cl = self.v2._straight_centerline(np.array([0, 0, 0]),
                                           np.array([10, 0, 0]),
                                           n_points=11)
        self.assertEqual(cl.shape, (11, 3))
        np.testing.assert_allclose(cl[0], [0, 0, 0])
        np.testing.assert_allclose(cl[-1], [10, 0, 0])
        np.testing.assert_allclose(cl[5], [5, 0, 0])

    def test_extend_centerline_tail_lengthens_by_extra_mm(self):
        """A 10mm centerline + 3mm extension → 13mm total along same direction."""
        cl = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
        cl2 = self.v2._extend_centerline_tail(cl, extra_mm=3.0)
        self.assertEqual(cl2.shape, (3, 3))
        # Total length is now 13mm
        diffs = np.diff(cl2, axis=0)
        total = float(np.linalg.norm(diffs, axis=1).sum())
        self.assertAlmostEqual(total, 13.0, places=5)
        np.testing.assert_allclose(cl2[-1], [13, 0, 0])

    def test_extend_centerline_tail_zero_extra_returns_input(self):
        cl = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
        cl2 = self.v2._extend_centerline_tail(cl, extra_mm=0.0)
        # Should return the same array (or a no-op equivalent)
        np.testing.assert_array_equal(cl, cl2)

    def test_ortho_uv_generates_orthonormal_basis(self):
        """For any tangent unit vector, _ortho_uv returns u, v perpendicular
        to tangent and to each other."""
        for tangent in (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.6, 0.8, 0.0]),  # 30°
            np.array([0.5, 0.5, np.sqrt(0.5)]),  # general
        ):
            tangent = tangent / np.linalg.norm(tangent)
            u, v = self.v2._ortho_uv(tangent)
            # All three orthogonal
            self.assertAlmostEqual(float(u @ tangent), 0.0, places=6)
            self.assertAlmostEqual(float(v @ tangent), 0.0, places=6)
            self.assertAlmostEqual(float(u @ v), 0.0, places=6)
            # u, v are unit vectors
            self.assertAlmostEqual(float(np.linalg.norm(u)), 1.0, places=6)
            self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=6)


@unittest.skipUnless(DEPS_AVAILABLE, "numpy not available")
class PlacementV2ResultTests(unittest.TestCase):
    """Test the public ``PlacementV2Result`` dataclass."""

    def test_failure_construction(self):
        from rosa_core import PlacementV2Result
        r = PlacementV2Result(
            success=False, model_id=None, placed_ras=[], centerline_ras=None,
            corr_score=0.0, bolt_end_arc_mm=0.0, bolt_source="none",
            n_placed=0, rejected_reason="seed_zero_length",
        )
        self.assertFalse(r.success)
        self.assertEqual(r.bolt_source, "none")
        self.assertEqual(r.placed_ras, [])
        self.assertEqual(r.diagnostics, {})  # default factory

    def test_success_construction(self):
        from rosa_core import PlacementV2Result
        placed = [[0, 0, 0], [3.5, 0, 0], [7, 0, 0]]
        r = PlacementV2Result(
            success=True, model_id="PMT-8",
            placed_ras=placed, centerline_ras=None,
            corr_score=0.85, bolt_end_arc_mm=12.0, bolt_source="metal",
            n_placed=3, rejected_reason="",
        )
        self.assertTrue(r.success)
        self.assertEqual(r.model_id, "PMT-8")
        self.assertEqual(r.placed_ras, placed)
        self.assertAlmostEqual(r.corr_score, 0.85)


@unittest.skipUnless(DEPS_AVAILABLE, "numpy not available")
class PlaceContactsForSeedV2Tests(unittest.TestCase):
    """Behavioral tests on the public entry point that don't require a full
    CT volume (zero-length seed, edge cases). Full-pipeline behavior is
    validated by the dataset regression in
    ``project_v2_pipeline_2026-05-07.md``.
    """

    def test_zero_length_seed_returns_failure(self):
        from rosa_core import place_contacts_for_seed_v2
        # Seed start == end: zero length. Should fail before touching features.
        result = place_contacts_for_seed_v2(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
            features={},   # not accessed
            library_models=[],
        )
        self.assertFalse(result.success)
        self.assertEqual(result.rejected_reason, "seed_zero_length")
        self.assertEqual(result.bolt_source, "none")
        self.assertEqual(result.n_placed, 0)


@unittest.skipUnless(DEPS_AVAILABLE, "numpy not available")
class SlotCCVolumeTests(unittest.TestCase):
    """Tests for the per-slot saturating-HU CC-volume helper used by the
    optional ``max_slot_cc_volume_p90_mm3`` filter.
    """

    def setUp(self):
        from rosa_core import contact_placement_v2 as v2
        self.v2 = v2
        # 41×41×41 voxel cube at 1mm isotropic spacing → easy to reason
        # about. RAS = IJK in this synthetic frame.
        self.r2i = np.eye(4)
        self.spacing = (1.0, 1.0, 1.0)

    def _empty_volume(self):
        return np.full((41, 41, 41), -1000, dtype=np.int16)

    def test_zero_volume_when_no_metal(self):
        vol = self._empty_volume()
        v = self.v2._slot_cc_volume_mm3(
            vol, self.r2i, np.array([20.0, 20.0, 20.0]), self.spacing,
        )
        self.assertEqual(v, 0.0)

    def test_isolated_2mm3_blob(self):
        """A 2-voxel above-threshold blob at the slot → CC volume = 2 mm³."""
        vol = self._empty_volume()
        # Place 2 saturating voxels at the slot (k=20, j=20, i=20-21).
        vol[20, 20, 20] = 2000
        vol[20, 20, 21] = 2000
        v = self.v2._slot_cc_volume_mm3(
            vol, self.r2i, np.array([20.0, 20.0, 20.0]), self.spacing,
        )
        self.assertAlmostEqual(v, 2.0)

    def test_large_bone_extends_beyond_roi(self):
        """A 9×9×9 bone CC inside a 5mm half-extent ROI gives 729 mm³."""
        vol = self._empty_volume()
        vol[16:25, 16:25, 16:25] = 2000
        v = self.v2._slot_cc_volume_mm3(
            vol, self.r2i, np.array([20.0, 20.0, 20.0]), self.spacing,
            roi_half_mm=5.0,
        )
        # ROI is 11×11×11 at 1mm spacing; the 9×9×9 bone fits entirely.
        self.assertAlmostEqual(v, 9 * 9 * 9)

    def test_two_disjoint_blobs_only_slot_cc_counted(self):
        """Two separate above-threshold blobs in the ROI: only the one
        containing the slot voxel contributes to the volume."""
        vol = self._empty_volume()
        # CC #1: 1 voxel at slot.
        vol[20, 20, 20] = 2000
        # CC #2: 27 voxels far across the ROI (3x3x3 cluster).
        vol[15:18, 15:18, 15:18] = 2000
        v = self.v2._slot_cc_volume_mm3(
            vol, self.r2i, np.array([20.0, 20.0, 20.0]), self.spacing,
        )
        self.assertAlmostEqual(v, 1.0)

    def test_slot_offset_finds_nearest_above_threshold(self):
        """If the slot voxel is below threshold but a CC is nearby in the
        ROI, the helper measures the volume of the nearest CC."""
        vol = self._empty_volume()
        # CC near the slot but not at it.
        vol[19, 20, 20] = 2000
        vol[19, 20, 21] = 2000
        v = self.v2._slot_cc_volume_mm3(
            vol, self.r2i, np.array([20.0, 20.0, 20.0]), self.spacing,
        )
        self.assertAlmostEqual(v, 2.0)


@unittest.skipUnless(DEPS_AVAILABLE, "numpy not available")
class LazyExportsTests(unittest.TestCase):
    """The v2 module's public symbols are reachable via rosa_core's lazy export."""

    def test_imports_via_rosa_core(self):
        from rosa_core import (
            place_contacts_for_seed_v2,
            PlacementV2Result,
            MIN_CORR_FOR_REAL_SHANK,
            MAX_SLOT_CC_VOLUME_P90_MM3,
            MIN_SLOT_HU_MEAN,
        )
        # Attribute checks
        self.assertTrue(callable(place_contacts_for_seed_v2))
        self.assertEqual(MIN_CORR_FOR_REAL_SHANK, 0.35)
        self.assertEqual(MAX_SLOT_CC_VOLUME_P90_MM3, 150.0)
        self.assertEqual(MIN_SLOT_HU_MEAN, 1500.0)
        self.assertTrue(hasattr(PlacementV2Result, "__dataclass_fields__"))


if __name__ == "__main__":
    unittest.main()
