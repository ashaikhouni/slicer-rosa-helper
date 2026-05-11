"""Unit tests for ``rosa_core.contact_placement.polyline``.

These cover the geometry helpers extracted from the v2 monolith. Mirrors the
existing ``test_contact_placement_v2.py`` polyline-helper tests but uses the
new public names (no leading underscore).
"""
from __future__ import annotations

import unittest

import numpy as np

from rosa_core.contact_placement import polyline as pl


class PolylineSegmentsTests(unittest.TestCase):
    def test_straight_line_three_points(self):
        cl = np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0]], dtype=float)
        starts, dirs, lens, cum = pl.polyline_segments(cl)
        self.assertEqual(starts.shape, (2, 3))
        np.testing.assert_allclose(lens, [5, 5])
        np.testing.assert_allclose(cum, [0, 5])
        np.testing.assert_allclose(dirs, [[1, 0, 0], [1, 0, 0]])

    def test_drops_zero_length_segments(self):
        # The middle two points are coincident.
        cl = np.array([[0, 0, 0], [5, 0, 0], [5, 0, 0], [10, 0, 0]], dtype=float)
        starts, dirs, lens, cum = pl.polyline_segments(cl)
        self.assertEqual(starts.shape[0], 2)
        np.testing.assert_allclose(lens, [5, 5])

    def test_rejects_bad_shape(self):
        with self.assertRaises(ValueError):
            pl.polyline_segments(np.zeros((1, 3)))   # only 1 point
        with self.assertRaises(ValueError):
            pl.polyline_segments(np.zeros((3, 2)))   # 2D coords
        with self.assertRaises(ValueError):
            # All zero-length segments.
            pl.polyline_segments(np.zeros((3, 3)))


class PolylinePosAtArcTests(unittest.TestCase):
    def test_endpoints_and_midpoint(self):
        cl = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
        np.testing.assert_allclose(pl.polyline_pos_at_arc(cl, 0.0), [0, 0, 0])
        np.testing.assert_allclose(pl.polyline_pos_at_arc(cl, 5.0), [5, 0, 0])
        np.testing.assert_allclose(pl.polyline_pos_at_arc(cl, 10.0), [10, 0, 0])

    def test_clamps_to_endpoints(self):
        cl = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
        np.testing.assert_allclose(pl.polyline_pos_at_arc(cl, -1.0), [0, 0, 0])
        np.testing.assert_allclose(pl.polyline_pos_at_arc(cl, 12.0), [10, 0, 0])

    def test_curved(self):
        cl = np.array([[0, 0, 0], [10, 0, 0], [10, 5, 0]], dtype=float)
        np.testing.assert_allclose(pl.polyline_pos_at_arc(cl, 5.0), [5, 0, 0])
        np.testing.assert_allclose(pl.polyline_pos_at_arc(cl, 10.0), [10, 0, 0], atol=1e-9)
        np.testing.assert_allclose(pl.polyline_pos_at_arc(cl, 15.0), [10, 5, 0])


class PolylineAtArcTests(unittest.TestCase):
    """``polyline_at_arc`` is the alternate impl used by stage_e_place."""

    def test_matches_polyline_pos_at_arc(self):
        cl = np.array([[0, 0, 0], [10, 0, 0], [10, 5, 0]], dtype=float)
        for arc in [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0]:
            a = pl.polyline_at_arc(cl, arc)
            b = pl.polyline_pos_at_arc(cl, arc)
            np.testing.assert_allclose(a, b, atol=1e-9, err_msg=f"arc={arc}")

    def test_clamps_to_endpoints(self):
        cl = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
        np.testing.assert_allclose(pl.polyline_at_arc(cl, -5.0), [0, 0, 0])
        np.testing.assert_allclose(pl.polyline_at_arc(cl, 50.0), [10, 0, 0])


class ProjectToPolylineTests(unittest.TestCase):
    def test_arc_orthogonal_drop(self):
        cl = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
        arc = pl.project_to_polyline_arc(cl, np.array([5.0, 1.0, 0.0]))
        self.assertAlmostEqual(arc, 5.0, places=6)

    def test_arc_clamps_to_endpoints(self):
        cl = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
        before = pl.project_to_polyline_arc(cl, np.array([-5.0, 0.0, 0.0]))
        self.assertAlmostEqual(before, 0.0, places=6)
        after = pl.project_to_polyline_arc(cl, np.array([20.0, 0.0, 0.0]))
        self.assertAlmostEqual(after, 10.0, places=6)

    def test_arc_perp_pair(self):
        cl = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
        arc, perp = pl.project_to_polyline(np.array([5.0, 3.0, 0.0]), cl)
        self.assertAlmostEqual(arc, 5.0, places=6)
        self.assertAlmostEqual(perp, 3.0, places=6)


class ExtendCenterlineTailTests(unittest.TestCase):
    def test_lengthens_by_extra_mm(self):
        cl = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
        cl2 = pl.extend_centerline_tail(cl, extra_mm=3.0)
        self.assertEqual(cl2.shape[0], 3)
        np.testing.assert_allclose(cl2[-1], [13, 0, 0])
        # First two points unchanged.
        np.testing.assert_allclose(cl2[:2], cl)

    def test_zero_extra_returns_input(self):
        cl = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
        cl2 = pl.extend_centerline_tail(cl, extra_mm=0.0)
        self.assertIs(cl2, cl)


class StraightCenterlineTests(unittest.TestCase):
    def test_endpoints_and_count(self):
        cl = pl.straight_centerline(np.array([0, 0, 0]), np.array([10, 0, 0]), n_points=8)
        self.assertEqual(cl.shape, (8, 3))
        np.testing.assert_allclose(cl[0], [0, 0, 0])
        np.testing.assert_allclose(cl[-1], [10, 0, 0])
        # Uniform spacing.
        diffs = np.diff(cl, axis=0)
        np.testing.assert_allclose(np.linalg.norm(diffs, axis=1),
                                    np.full(7, 10.0 / 7))


class MinDistPtsToPolylineTests(unittest.TestCase):
    def test_empty_polyline(self):
        pts = np.array([[1, 2, 3]], dtype=float)
        d = pl.min_dist_pts_to_polyline(pts, np.zeros((1, 3)))
        self.assertTrue(np.isinf(d[0]))

    def test_axis_perpendicular_distance(self):
        cl = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
        pts = np.array([[5, 4, 0], [5, 0, 3], [-1, 0, 0]], dtype=float)
        d = pl.min_dist_pts_to_polyline(pts, cl)
        np.testing.assert_allclose(d[0], 4.0, atol=1e-9)
        np.testing.assert_allclose(d[1], 3.0, atol=1e-9)
        np.testing.assert_allclose(d[2], 1.0, atol=1e-9)  # past start


class OrthoUVTests(unittest.TestCase):
    def test_orthogonal_to_tangent(self):
        for tangent in [
            np.array([1, 0, 0], dtype=float),
            np.array([0, 1, 0], dtype=float),
            np.array([1, 1, 1], dtype=float) / np.sqrt(3),
        ]:
            u, v = pl.ortho_uv(tangent)
            np.testing.assert_allclose(u @ tangent, 0.0, atol=1e-9)
            np.testing.assert_allclose(v @ tangent, 0.0, atol=1e-9)
            np.testing.assert_allclose(u @ v, 0.0, atol=1e-9)
            np.testing.assert_allclose(np.linalg.norm(u), 1.0, atol=1e-9)
            np.testing.assert_allclose(np.linalg.norm(v), 1.0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
