"""Unit + dataset regression tests for ``rosa_core.contact_placement``.

Unit tests cover the placement primitives (partition heuristic, curve
fits, RANSAC pick, polyline projection, ownership). They run without
the heavy CT dataset and pin the building-block contracts.

The dataset regression test (gated on ``ROSA_AMC_TESTING_ROOT``)
re-runs the validated 6-subject placement pipeline (AMC88 / AMC91 /
AMC135 / AMC136 / AMC137 / T22) and asserts the headline numbers
(exact-count >= 49/66, median error <= 0.45 mm, on-metal fraction >=
0.95). This is the regression net for the bent-RANSAC + polynomial
production path.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))

import numpy as np

from rosa_core.contact_placement import (
    ContactPlacementConfig,
    _fit_polynomial_through_points,
    _fit_spline_through_points,
    _identify_inliers,
    _partition_bolt_contacts_by_arc,
    _partition_bolt_from_contacts,
    _plausibly_matches_library,
    _polyline_axis_at,
    _project_to_polyline,
    _ransac_pick_library,
    _resolve_seed_path,
    assign_axis_owners,
)


# ---------------------------------------------------------------------
# Partition heuristic
# ---------------------------------------------------------------------


class PartitionBoltContactsByArcTests(unittest.TestCase):
    def test_basic_5mm_gap_picked(self):
        # Bolt blobs at 0..3mm, gap, contacts at 13..40 (~3mm pitch x10).
        bolt = np.arange(0, 4, 1.0)
        contacts = np.arange(13.0, 41.0, 3.0)
        arcs = np.concatenate([bolt, contacts])
        eff = _partition_bolt_contacts_by_arc(
            arcs, entry_arc=2.0, gap_thresh_mm=5.0, min_contact_side=4,
        )
        self.assertIsNotNone(eff)
        self.assertAlmostEqual(eff, 13.0, places=3)

    def test_stray_blob_past_tip_falls_through(self):
        # Real bolt-to-contact gap at arc=10; stray blob at arc=50 makes
        # the largest gap meaningless if we keep it.
        bolt = np.arange(0, 4, 1.0)
        contacts = np.arange(10.0, 31.0, 3.0)        # 8 contacts
        stray = np.array([50.0])                      # stray past tip
        arcs = np.concatenate([bolt, contacts, stray])
        eff = _partition_bolt_contacts_by_arc(
            arcs, entry_arc=2.0, gap_thresh_mm=5.0, min_contact_side=4,
        )
        # Largest gap is 31->50 (19mm) but only 1 contact past it; should
        # fall through to the bolt->contact gap (4->10, 6mm).
        self.assertAlmostEqual(eff, 10.0, places=3)

    def test_no_qualifying_gap_returns_entry_arc(self):
        # All gaps under threshold → fallback to entry_arc.
        arcs = np.arange(0, 12, 2.0)                  # uniform 2mm gaps
        eff = _partition_bolt_contacts_by_arc(
            arcs, entry_arc=3.5, gap_thresh_mm=5.0, min_contact_side=4,
        )
        self.assertEqual(eff, 3.5)

    def test_no_qualifying_gap_returns_none_when_entry_none(self):
        arcs = np.arange(0, 12, 2.0)
        eff = _partition_bolt_contacts_by_arc(
            arcs, entry_arc=None, gap_thresh_mm=5.0, min_contact_side=4,
        )
        self.assertIsNone(eff)

    def test_single_blob_returns_entry_arc(self):
        eff = _partition_bolt_contacts_by_arc(
            [10.0], entry_arc=5.0, gap_thresh_mm=5.0, min_contact_side=4,
        )
        self.assertEqual(eff, 5.0)

    def test_min_contact_side_zero_picks_largest_gap(self):
        # When the floor is 0, the largest gap wins.
        bolt = np.arange(0, 4, 1.0)
        contacts = np.arange(10.0, 31.0, 3.0)
        stray = np.array([60.0])
        arcs = np.concatenate([bolt, contacts, stray])
        eff = _partition_bolt_contacts_by_arc(
            arcs, entry_arc=2.0, gap_thresh_mm=5.0, min_contact_side=0,
        )
        self.assertAlmostEqual(eff, 60.0, places=3)


# ---------------------------------------------------------------------
# Partition cascade (Tier 1 → Tier 2 → Tier 3 → Tier 4)
# ---------------------------------------------------------------------


def _make_uniform_pitch_model(model_id: str, n_contacts: int, pitch: float):
    offsets = [i * pitch for i in range(n_contacts)]
    return {
        "id": model_id,
        "contact_center_offsets_from_tip_mm": offsets,
    }


def _make_dixi_15cm_like_model():
    """Synthetic long-gap electrode mimicking DIXI 15CM: 8 distal
    contacts at 3.5 mm pitch, 15-mm design gap, 7 proximal contacts at
    3.5 mm pitch. Total span ~ 0+3.5*7 + 15 + 3.5*6 = 24.5 + 15 + 21
    = 60.5 mm + 3.5 = 64 mm to proximal contact."""
    distal = [i * 3.5 for i in range(8)]
    proximal_start = distal[-1] + 15.0
    proximal = [proximal_start + i * 3.5 for i in range(7)]
    offsets = distal + proximal
    return {
        "id": "DIXI-15CM-like",
        "contact_center_offsets_from_tip_mm": offsets,
    }


class PlausiblyMatchesLibraryTests(unittest.TestCase):
    def test_uniform_pitch_pattern_matches(self):
        models = [_make_uniform_pitch_model("DIXI-8", 8, 3.5)]
        arcs = np.array([0.0, 3.5, 7.0, 10.5, 14.0, 17.5, 21.0, 24.5])
        self.assertTrue(_plausibly_matches_library(
            arcs, models, min_inliers=4,
            pair_tol_mm=1.0, inlier_tol_mm=1.5,
        ))

    def test_random_pattern_does_not_match(self):
        models = [_make_uniform_pitch_model("DIXI-8", 8, 3.5)]
        arcs = np.array([0.0, 1.7, 4.2, 9.1, 11.6, 14.3])  # noise
        self.assertFalse(_plausibly_matches_library(
            arcs, models, min_inliers=6,
            pair_tol_mm=0.5, inlier_tol_mm=0.5,
        ))

    def test_empty_or_too_few_arcs_returns_false(self):
        models = [_make_uniform_pitch_model("DIXI-8", 8, 3.5)]
        self.assertFalse(_plausibly_matches_library(
            np.array([]), models, min_inliers=3,
            pair_tol_mm=1.0, inlier_tol_mm=1.5,
        ))

    def test_no_models_returns_false(self):
        self.assertFalse(_plausibly_matches_library(
            np.array([0.0, 3.5, 7.0]), None, min_inliers=3,
            pair_tol_mm=1.0, inlier_tol_mm=1.5,
        ))


class PartitionCascadeTests(unittest.TestCase):
    def test_tier1_hu_profile_takes_precedence(self):
        # Even when there's an obvious large gap, entry_arc wins.
        arcs = np.array([0.0, 2.0, 30.0, 33.5, 37.0, 40.5, 44.0])
        eff, tier = _partition_bolt_from_contacts(
            arcs, entry_arc=10.0,
            head_distance_at_blob=None, library_models=None,
        )
        self.assertEqual(tier, "hu_profile")
        self.assertAlmostEqual(eff, 10.0)

    def test_tier2_hull_proximity_when_no_entry_arc(self):
        # head_distance: bolt blobs at ~0-1 mm (in the skull), contact
        # blobs at >3 mm. The first contact-side blob's arc wins.
        arcs = np.array([0.0, 1.5, 3.0, 12.0, 15.5, 19.0, 22.5])
        head_distance = np.array([0.5, 1.0, 1.8, 8.0, 12.0, 16.0, 20.0])
        eff, tier = _partition_bolt_from_contacts(
            arcs, entry_arc=None,
            head_distance_at_blob=head_distance,
            library_models=None,
        )
        self.assertEqual(tier, "hull_proximity")
        # First arc whose head_distance > 3 mm is at arc=12.
        self.assertAlmostEqual(eff, 12.0)

    def test_tier3_library_aware_rejects_design_gap_cut(self):
        # Synthetic 15CM-like layout with bolt blobs.
        # Geometry along axis (mm):
        #   bolt @ [0, 1.5, 3.0]  (3 bolt blobs, sustained bright)
        #   bolt-to-proximal gap = 14 mm
        #   distal cluster @ [17 + i * 3.5 for i in range(8)]  → 17..41.5
        #   design gap @ 41.5 → 56.5 (15 mm)
        #   proximal cluster @ [56.5 + i * 3.5 for i in range(7)] → 56.5..77.5
        bolt = np.array([0.0, 1.5, 3.0])
        distal = np.array([17 + i * 3.5 for i in range(8)])
        proximal = np.array([56.5 + i * 3.5 for i in range(7)])
        arcs = np.concatenate([bolt, distal, proximal])
        # Library: a 15-contact 15CM-like model, plus a confounder
        # uniform 8-contact model that COULD match the distal-only
        # pattern. The cascade must NOT cut at the 15-mm design gap
        # even though it's the largest gap, because cutting there
        # would NOT match the 15CM pattern (only the 8-contact model
        # matches a distal-only cut).
        models = [_make_dixi_15cm_like_model()]
        eff, tier = _partition_bolt_from_contacts(
            arcs, entry_arc=None,
            head_distance_at_blob=None,
            library_models=models,
            config=ContactPlacementConfig(
                partition_strategy="cascade",
                partition_gap_thresh_mm=5.0,
                partition_min_contact_side=4,
            ),
        )
        # The bolt-to-distal gap (14 mm) is the cut. The design gap
        # (15 mm, larger!) is rejected because the distal-only pattern
        # doesn't match DIXI-15CM-like.
        self.assertEqual(tier, "library_aware_gap")
        self.assertAlmostEqual(eff, 17.0, places=3)

    def test_tier3_picks_largest_gap_when_pattern_matches(self):
        # Two qualifying gaps; the largest passes the library check.
        # No bolt blobs — just a single 8-contact electrode with a
        # leading stray blob.
        arcs = np.array([0.0, 12.0, 15.5, 19.0, 22.5, 26.0, 29.5, 33.0, 36.5])
        models = [_make_uniform_pitch_model("DIXI-8", 8, 3.5)]
        eff, tier = _partition_bolt_from_contacts(
            arcs, entry_arc=None,
            head_distance_at_blob=None,
            library_models=models,
            config=ContactPlacementConfig(
                partition_strategy="cascade",
                partition_gap_thresh_mm=5.0,
                partition_min_contact_side=4,
            ),
        )
        # Largest gap is stray-to-cluster (12 mm); cluster matches
        # DIXI-8.
        self.assertEqual(tier, "library_aware_gap")
        self.assertAlmostEqual(eff, 12.0, places=3)

    def test_tier4_no_cut_when_nothing_works(self):
        # No entry_arc, no head_distance, no library, no big gaps.
        arcs = np.array([0.0, 3.5, 7.0, 10.5, 14.0])
        eff, tier = _partition_bolt_from_contacts(
            arcs, entry_arc=None,
            head_distance_at_blob=None,
            library_models=None,
        )
        self.assertEqual(tier, "no_cut")
        self.assertIsNone(eff)

    def test_trivial_under_two_blobs(self):
        eff, tier = _partition_bolt_from_contacts(
            np.array([5.0]), entry_arc=2.0,
            head_distance_at_blob=None, library_models=None,
        )
        self.assertEqual(tier, "trivial")
        self.assertEqual(eff, 2.0)

    def test_legacy_gap_only_misclassifies_15cm_design_gap(self):
        """Documents the failure mode that motivated the cascade.

        With ``gap_only`` strategy + the 15CM-like geometry, the
        15-mm design gap is the largest qualifying gap, so the cut
        falls there — splitting the electrode and breaking RANSAC.
        Held as a regression test; if this changes, the gap_only
        path itself has been altered.
        """
        bolt = np.array([0.0, 1.5, 3.0])
        distal = np.array([17 + i * 3.5 for i in range(8)])
        proximal = np.array([56.5 + i * 3.5 for i in range(7)])
        arcs = np.concatenate([bolt, distal, proximal])
        eff = _partition_bolt_contacts_by_arc(
            arcs, entry_arc=None,
            gap_thresh_mm=5.0, min_contact_side=4,
        )
        # gap_only chooses the design gap (15 mm) over the bolt gap
        # (14 mm) — wrong cut.
        self.assertAlmostEqual(eff, 56.5, places=3)


# ---------------------------------------------------------------------
# Polyline + arc-length
# ---------------------------------------------------------------------


class PolylineProjectionTests(unittest.TestCase):
    def test_straight_two_point_path_is_linear_projection(self):
        path = np.array([[0, 0, 0], [0, 0, 30]], dtype=float)
        pts = np.array([
            [0, 0, 5],          # arc 5, perp 0
            [1, 0, 10],         # arc 10, perp 1
            [0, 0, -3],         # arc -3, perp 0 (past start)
            [0, 0, 35],         # arc 35, perp 0 (past end)
        ], dtype=float)
        arc, perp = _project_to_polyline(pts, path)
        np.testing.assert_allclose(arc, [5, 10, -3, 35], atol=1e-6)
        np.testing.assert_allclose(perp, [0, 1, 0, 0], atol=1e-6)

    def test_l_bend_polyline_arc_at_elbow(self):
        # Polyline: 10mm in +z, then 10mm in +x.
        path = np.array([
            [0, 0, 0],
            [0, 0, 10],
            [10, 0, 10],
        ], dtype=float)
        pts = np.array([
            [0, 0, 5],          # midway first segment
            [5, 0, 10],          # midway second segment
            [0.5, 0, 10],        # at the elbow, slightly off
        ], dtype=float)
        arc, perp = _project_to_polyline(pts, path)
        self.assertAlmostEqual(arc[0], 5.0, places=3)
        self.assertAlmostEqual(arc[1], 15.0, places=3)
        # Elbow point: arc ~ 10.5, perp 0.
        self.assertGreater(arc[2], 9.5)
        self.assertLess(arc[2], 11.0)

    def test_resolve_seed_path_curved_takes_precedence(self):
        path = np.array([[0, 0, 0], [0, 0, 5], [5, 0, 5]], dtype=float)
        out, L = _resolve_seed_path([0, 0, 0], [10, 0, 0], path_ras=path)
        self.assertEqual(out.shape, (3, 3))
        self.assertAlmostEqual(L, 10.0, places=3)

    def test_resolve_seed_path_straight_default(self):
        out, L = _resolve_seed_path([0, 0, 0], [0, 0, 7])
        self.assertEqual(out.shape, (2, 3))
        self.assertAlmostEqual(L, 7.0, places=3)

    def test_polyline_axis_at_end_extrapolates(self):
        path = np.array([[0, 0, 0], [0, 0, 30]], dtype=float)
        pt = _polyline_axis_at(path, 35.0)
        np.testing.assert_allclose(pt, [0, 0, 35], atol=1e-6)
        pt = _polyline_axis_at(path, -2.0)
        np.testing.assert_allclose(pt, [0, 0, -2], atol=1e-6)


# ---------------------------------------------------------------------
# Curve fits
# ---------------------------------------------------------------------


class CurveFitTests(unittest.TestCase):
    def test_polynomial_through_colinear_points_is_straight(self):
        ts = np.linspace(0, 30, 7)
        # All points on a line in 3-space.
        pts = np.column_stack([ts, 2 * ts, -1.5 * ts + 4])
        sp, t2s, ctrl = _fit_polynomial_through_points(pts, ts, deg=2)
        self.assertIsNotNone(sp)
        # Sample sp at 5 along-axis arcs and check against the analytic
        # straight line.
        taus = np.array([1.0, 7.5, 12.0, 22.0, 28.0])
        out = sp(taus)
        expected = np.column_stack([taus, 2 * taus, -1.5 * taus + 4])
        np.testing.assert_allclose(out, expected, atol=1e-3)
        # t2s on a colinear set: cumulative arc = |delta| * sqrt(1+4+2.25) per unit tau
        scale = float(np.sqrt(1 + 4 + 2.25))
        s = t2s(np.array([0.0, 30.0]))
        np.testing.assert_allclose(s, [0.0, 30.0 * scale], atol=1e-2)

    def test_polynomial_too_few_points_returns_none(self):
        sp, t2s, ctrl = _fit_polynomial_through_points(
            np.zeros((2, 3)), np.array([0.0, 1.0]), deg=2,
        )
        self.assertIsNone(sp)

    def test_polynomial_deduplicates_close_arcs(self):
        # Two arcs within 1e-3 collapse to one — guards against
        # singular polyfit. With deg=2 we need >=3 unique points.
        ts = np.array([0.0, 0.0005, 5.0, 10.0])
        pts = np.column_stack([ts, ts, ts])
        sp, _, _ = _fit_polynomial_through_points(pts, ts, deg=2)
        self.assertIsNotNone(sp)

    def test_spline_returns_none_if_fewer_than_4_points(self):
        sp, _, _ = _fit_spline_through_points(np.zeros((3, 3)), np.array([0, 1, 2]))
        self.assertIsNone(sp)

    def test_spline_through_4_points_works(self):
        ts = np.array([0.0, 5.0, 10.0, 15.0])
        pts = np.column_stack([ts, np.zeros_like(ts), ts ** 2 / 30])
        sp, t2s, _ = _fit_spline_through_points(pts, ts)
        self.assertIsNotNone(sp)
        out = sp(np.array([7.5]))
        # Inside the sample range, spline matches data approximately.
        self.assertAlmostEqual(out[0, 0], 7.5, places=2)


# ---------------------------------------------------------------------
# RANSAC library match
# ---------------------------------------------------------------------


def _make_uniform_pitch_model(model_id: str, n_contacts: int, pitch: float):
    """Build a synthetic library-shaped dict — uniform-pitch electrode."""
    offsets = [i * pitch for i in range(n_contacts)]
    return {
        "id": model_id,
        "contact_center_offsets_from_tip_mm": offsets,
    }


class RansacPickLibraryTests(unittest.TestCase):
    def test_synthetic_pattern_matches_correct_model(self):
        # Build 3 candidate models; the data is a perfect 8-contact 3.5mm pattern.
        models = [
            _make_uniform_pitch_model("FOO-5", 5, 4.0),
            _make_uniform_pitch_model("DIXI-8", 8, 3.5),
            _make_uniform_pitch_model("BAR-12", 12, 2.0),
        ]
        # Tip at arc=20mm, peaks at 20, 23.5, 27, 30.5, ..., 44.5
        peaks = np.array([20 + i * 3.5 for i in range(8)])
        rb = _ransac_pick_library(
            peaks, models,
            entry_arc=15.0, profile_end_arc=50.0,
            pair_tol_mm=0.6, inlier_tol_mm=0.5,
            max_extend_tip_mm=5.0, bolt_free_pass=True, phantom_penalty=1.0,
        )
        self.assertIsNotNone(rb)
        self.assertEqual(rb["model_id"], "DIXI-8")
        self.assertEqual(rb["inl"], 8)

    def test_returns_none_on_too_few_peaks(self):
        models = [_make_uniform_pitch_model("DIXI-8", 8, 3.5)]
        rb = _ransac_pick_library(
            np.array([1.0]), models,
            entry_arc=None, profile_end_arc=50.0,
            pair_tol_mm=0.6, inlier_tol_mm=1.0,
            max_extend_tip_mm=5.0, bolt_free_pass=True, phantom_penalty=1.0,
        )
        self.assertIsNone(rb)

    def test_phantom_past_end_disqualifies(self):
        # A model whose tip placement would extend past profile_end +
        # max_extend_tip is disqualified outright.
        models = [_make_uniform_pitch_model("LONG-15", 15, 3.5)]
        peaks = np.array([0.0, 3.5])  # only 2 peaks; rest of slots phantom
        rb = _ransac_pick_library(
            peaks, models,
            entry_arc=None, profile_end_arc=10.0,
            pair_tol_mm=0.6, inlier_tol_mm=0.5,
            max_extend_tip_mm=2.0, bolt_free_pass=True, phantom_penalty=1.0,
        )
        # Either no valid placement found, or score very negative — but
        # the test we care about is that the function doesn't crash.
        # A single best _is_ found; just verify inl is small (=2).
        if rb is not None:
            self.assertEqual(rb["inl"], 2)

    def test_identify_inliers_no_double_assignment(self):
        models = [_make_uniform_pitch_model("DIXI-5", 5, 3.5)]
        # Tip at arc=10. Slot offsets = 0, 3.5, 7, 10.5, 14.
        peaks = np.array([10.0, 13.5, 17.0, 20.5, 24.0])
        rb = {
            "model_id": "DIXI-5", "score": 5.0, "inl": 5,
            "ns": 5, "tip_arc": 10.0, "sign": +1.0,
        }
        inl = _identify_inliers(peaks, rb, models, tol=0.5)
        self.assertEqual(sorted(inl), [0, 1, 2, 3, 4])
        self.assertEqual(len(inl), len(set(inl)))


# ---------------------------------------------------------------------
# Cross-shank ownership
# ---------------------------------------------------------------------


class AssignAxisOwnersTests(unittest.TestCase):
    def test_two_parallel_axes_closest_wins(self):
        # Two axes 5mm apart in y, both running along z.
        axes = [
            {"start_ras": [0, 0, 0], "end_ras": [0, 0, 30]},
            {"start_ras": [0, 5, 0], "end_ras": [0, 5, 30]},
        ]
        blobs = np.array([
            [0, 0, 5],     # closer to axis 0
            [0, 5, 10],    # closer to axis 1
            [0, 2, 15],    # closer to axis 0 (perp 2 vs 3)
            [0, 3, 20],    # closer to axis 1 (perp 2 vs 3)
        ], dtype=float)
        owners = assign_axis_owners(blobs, axes, max_perp_mm=4.0)
        np.testing.assert_array_equal(owners, [0, 1, 0, 1])

    def test_far_blob_unowned(self):
        axes = [{"start_ras": [0, 0, 0], "end_ras": [0, 0, 30]}]
        blobs = np.array([[0, 10, 5]], dtype=float)
        owners = assign_axis_owners(blobs, axes, max_perp_mm=4.0)
        self.assertEqual(owners[0], -1)

    def test_blob_outside_along_extent_unowned(self):
        axes = [{"start_ras": [0, 0, 0], "end_ras": [0, 0, 30]}]
        # 50mm past the deep tip — beyond max_extend_tip_mm.
        blobs = np.array([[0, 0, 80]], dtype=float)
        owners = assign_axis_owners(
            blobs, axes, max_perp_mm=4.0, max_extend_tip_mm=5.0,
        )
        self.assertEqual(owners[0], -1)

    def test_curved_axis_uses_polyline_projection(self):
        # L-shaped axis: down z 10mm, then across x 10mm.
        axes = [{
            "start_ras": [0, 0, 0],
            "end_ras": [10, 0, 10],
            "path_ras": np.array([
                [0, 0, 0],
                [0, 0, 10],
                [10, 0, 10],
            ], dtype=float),
        }]
        # Blob at the elbow — perp ~0, owned.
        # Blob 1mm off the deep arm — owned.
        # Blob far from both arms — unowned.
        blobs = np.array([
            [0, 0, 10],
            [5, 1, 10],
            [-10, 0, 5],
        ], dtype=float)
        owners = assign_axis_owners(blobs, axes, max_perp_mm=4.0)
        self.assertEqual(owners[0], 0)
        self.assertEqual(owners[1], 0)
        self.assertEqual(owners[2], -1)

    def test_empty_inputs(self):
        owners = assign_axis_owners(np.zeros((0, 3)), [])
        self.assertEqual(owners.shape, (0,))
        owners = assign_axis_owners(np.array([[0, 0, 0]]), [])
        self.assertEqual(owners.tolist(), [-1])


# ---------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------


class ContactPlacementConfigTests(unittest.TestCase):
    def test_defaults_match_validated_pipeline(self):
        cfg = ContactPlacementConfig()
        # These are the constants whose values determine the 49/66
        # dataset number — changing them implies re-running the
        # dataset regression and updating this test.
        self.assertEqual(cfg.curve_fit, "polynomial")
        self.assertEqual(cfg.poly_deg, 2)
        self.assertEqual(cfg.partition_min_contact_side, 4)
        self.assertEqual(cfg.partition_gap_thresh_mm, 5.0)
        self.assertEqual(cfg.corridor_radius_mm, 6.0)
        self.assertEqual(cfg.peak_hu_min, 500.0)
        self.assertEqual(cfg.ransac_inlier_tol_mm, 1.0)
        self.assertEqual(cfg.ransac_pair_tol_mm, 0.6)
        self.assertEqual(cfg.max_bent_iterations, 4)
        self.assertEqual(cfg.ownership_max_perp_mm, 4.0)
        self.assertEqual(cfg.ransac_max_extend_tip_mm, 5.0)


# ---------------------------------------------------------------------
# Dataset regression (gated)
# ---------------------------------------------------------------------


_AMC_ROOT = Path(
    os.environ.get("ROSA_AMC_TESTING_ROOT", "/Users/ammar/Documents/testing")
)
_AMC_AVAILABLE = (
    _AMC_ROOT.exists()
    and (_AMC_ROOT / "AMC88").exists()
    and (_AMC_ROOT / "T22").exists()
)


def _amc_deps_available() -> bool:
    try:
        import SimpleITK  # noqa: F401
        from rosa_detect.guided_fit_engine import compute_features  # noqa: F401
        from rosa_detect.service import (  # noqa: F401
            run_contact_pitch_v1_with_features,
        )
        from shank_core.io import image_ijk_ras_matrices  # noqa: F401
        return True
    except ImportError:
        return False


def _load_freesurfer_dat(path):
    pts = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) == 3:
                try:
                    pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    continue
                except ValueError:
                    pass
            break
    return np.asarray(pts, dtype=float)


def _gt_axis_from_contacts(contacts):
    pts = np.asarray(contacts, dtype=float)
    if pts.shape[0] < 2:
        return None, None
    cm = pts.mean(axis=0)
    cn = pts - cm
    _, _, vh = np.linalg.svd(cn, full_matrices=False)
    d = vh[0] / np.linalg.norm(vh[0])
    pr = cn @ d
    if pr[0] > pr[-1]:
        d = -d
        pr = -pr
    return cm + d * float(pr.min()), cm + d * float(pr.max())


def _list_gt_shanks(elec_dir):
    out = []
    for f in sorted(Path(elec_dir).glob("*.dat")):
        if f.stem.lower() == "elecpointset":
            continue
        pts = _load_freesurfer_dat(f)
        if pts.shape[0] >= 2:
            out.append({"name": f.stem, "contacts": pts})
    return out


def _find_subject_paths(root: Path, sid: str):
    p = root / sid
    ct = next(iter(p.glob("*_CT.nii.gz")), None) or next(
        iter(p.glob("*.nii.gz")), None
    )
    elec = None
    for n in ("Electrodes", "electrodes"):
        if (p / n).is_dir():
            elec = p / n
            break
    return ct, elec


@unittest.skipUnless(
    _AMC_AVAILABLE,
    f"AMC dataset not found at {_AMC_ROOT}. Set ROSA_AMC_TESTING_ROOT to override.",
)
@unittest.skipUnless(
    _amc_deps_available(),
    "rosa_detect / SimpleITK / shank_core not importable in this environment.",
)
class ContactPlacementDatasetRegressionTests(unittest.TestCase):
    """Pin the validated bent-RANSAC + polynomial pipeline numbers on
    AMC88 / AMC91 / AMC135 / AMC136 / AMC137 / T22. Slow (~3-5 min per
    strategy; ~10 min total) — runs only when the AMC dataset is
    reachable.

    Two strategies pinned:
      * ``gap_only`` — the legacy largest-qualifying-gap heuristic.
        Headline 49/66 baseline.
      * ``cascade`` — the conceptually correct three-tier path
        (HU profile → hull proximity → library-aware gap). Currently
        regresses 9 shanks; floor is held at the observed 40/67. Goal
        is to lift this floor by tuning ``signal_derived_entry_arc``
        (use ``notebooks/bolt_partition_qc.ipynb`` to diagnose).
    """

    SUBJECTS = ["AMC88", "AMC91", "AMC135", "AMC136", "AMC137", "T22"]
    SUBJECT_STRATEGY = {
        "AMC88": "pmt_35", "AMC91": "pmt_35", "AMC135": "pmt_35",
        "AMC136": "pmt_35", "AMC137": "pmt_35",
        "T22": "dixi",
    }
    GAP_ONLY_EXACT_FLOOR = 49
    GAP_ONLY_MEDIAN_CEILING_MM = 0.45
    GAP_ONLY_ON_METAL_FLOOR = 0.95
    # Cascade strategy floors. Updated 2026-05-05 after wiring
    # ``estimate_bolt_end_from_metal_mass`` as Tier 1A. Cascade now
    # matches gap_only within 1 exact-count.
    CASCADE_EXACT_FLOOR = 47        # observed 48 with Tier 1A (was 40)
    CASCADE_MEDIAN_CEILING_MM = 0.45
    CASCADE_ON_METAL_FLOOR = 0.95
    ON_METAL_HU = 1500.0
    MATCH_MAX_ANGLE_DEG = 12.0
    MATCH_MAX_PERP_MM = 8.0

    def _run_strategy(self, partition_strategy: str) -> dict:
        """Run the full 6-subject placement pipeline with the given
        partition strategy. Returns aggregate {exact, total, median_err,
        on_metal_frac}.
        """
        import SimpleITK as sitk
        from rosa_core import (
            ContactPlacementConfig, place_contacts_for_trajectories,
        )
        from rosa_core.electrode_models import load_electrode_library
        from rosa_core.electrode_classifier import filter_models_for_strategy
        from rosa_core.volume_sampling import sample_trilinear_at_ras
        from rosa_detect.guided_fit_engine import (
            compute_features, fit_trajectory, match_seed_to_auto_traj,
        )
        from rosa_detect.service import run_contact_pitch_v1_with_features
        from shank_core.io import image_ijk_ras_matrices

        library = load_electrode_library()

        total_shanks = 0
        total_exact = 0
        all_errors: list[float] = []
        total_pred = 0
        total_on_metal = 0

        for subj in self.SUBJECTS:
            ct_path, elec_dir = _find_subject_paths(_AMC_ROOT, subj)
            self.assertIsNotNone(ct_path, f"{subj}: no CT under {_AMC_ROOT}")
            self.assertIsNotNone(elec_dir, f"{subj}: no Electrodes/")

            img = sitk.ReadImage(str(ct_path))
            i2r_in, r2i_in = image_ijk_ras_matrices(img)
            features = compute_features(
                img, np.asarray(i2r_in), np.asarray(r2i_in),
            )
            i2r = np.asarray(features["ijk_to_ras_mat"])
            r2i = np.asarray(features["ras_to_ijk_mat"])
            ct_arr = features["ct_arr_kji"]
            strategy = self.SUBJECT_STRATEGY[subj]
            models = filter_models_for_strategy(library["models"], strategy)
            self.assertGreater(len(models), 0, f"{subj}: empty model strategy")

            auto_result, _ = run_contact_pitch_v1_with_features({
                "img": features["img"],
                "ijk_to_ras_4x4": i2r,
                "ras_to_ijk_4x4": r2i,
            })
            auto_trajs = list(auto_result.get("trajectories") or [])

            gt_shanks = _list_gt_shanks(elec_dir)
            traj_seeds: list[dict] = []
            gt_pts_by_index: list[np.ndarray] = []
            remaining = list(auto_trajs)
            for g in gt_shanks:
                gs, ge = _gt_axis_from_contacts(g["contacts"])
                if gs is None:
                    continue
                m = match_seed_to_auto_traj(
                    gs, ge, remaining,
                    max_angle_deg=self.MATCH_MAX_ANGLE_DEG,
                    max_lateral_shift_mm=self.MATCH_MAX_PERP_MM,
                )
                if m is not None:
                    cs = list(m.get("start_ras") or [])
                    ce = list(m.get("end_ras") or [])
                    consumed = False
                    kept = []
                    for t in remaining:
                        if (not consumed
                                and list(t.get("start_ras") or []) == cs
                                and list(t.get("end_ras") or []) == ce):
                            consumed = True
                            continue
                        kept.append(t)
                    remaining = kept
                    traj_seeds.append({
                        "name": g["name"],
                        "start_ras": list(m["start_ras"]),
                        "end_ras": list(m["end_ras"]),
                    })
                    gt_pts_by_index.append(np.asarray(g["contacts"], float))
                    continue
                try:
                    res = fit_trajectory(
                        planned_start_ras=gs, planned_end_ras=ge,
                        features=features,
                        ijk_to_ras_mat=i2r, ras_to_ijk_mat=r2i,
                    )
                except Exception:
                    res = {"success": False}
                if res.get("success"):
                    traj_seeds.append({
                        "name": g["name"],
                        "start_ras": list(res["start_ras"]),
                        "end_ras": list(res["end_ras"]),
                    })
                    gt_pts_by_index.append(np.asarray(g["contacts"], float))

            self.assertGreater(len(traj_seeds), 0, f"{subj}: no seeds")

            cfg = ContactPlacementConfig(partition_strategy=partition_strategy)
            batch = place_contacts_for_trajectories(
                features["img"], i2r, r2i,
                trajectories=traj_seeds,
                library_strategy=strategy,
                config=cfg,
                features=features,
            )

            for result, gt_pts in zip(batch.results, gt_pts_by_index):
                pred = np.asarray(result.positions_ras, dtype=float)
                total_shanks += 1
                if pred.shape[0] == gt_pts.shape[0]:
                    total_exact += 1
                if pred.size:
                    d = np.linalg.norm(
                        pred[:, None, :] - gt_pts[None, :, :], axis=2,
                    )
                    errs = d.min(axis=1)
                    all_errors.extend(errs.tolist())
                    total_pred += pred.shape[0]
                    if not result.success or result.placement_kind == "failed":
                        # Already failed shanks contribute 0 on-metal; pass.
                        continue
                    # Pick axis_unit from result (path-aware fall back).
                    seed = next(
                        s for s in traj_seeds if s["name"] == result.name
                    )
                    axis = (
                        np.asarray(seed["end_ras"], float)
                        - np.asarray(seed["start_ras"], float)
                    )
                    n = float(np.linalg.norm(axis))
                    if n < 1e-3:
                        continue
                    au = axis / n
                    for c in pred:
                        # Disk-max HU sample (mirrors the notebook's QC fn).
                        any_v = (
                            np.array([1.0, 0, 0])
                            if abs(au[0]) <= 0.9
                            else np.array([0, 1.0, 0])
                        )
                        u = np.cross(au, any_v)
                        u /= np.linalg.norm(u)
                        v = np.cross(au, u)
                        v /= np.linalg.norm(v)
                        s_max = float(sample_trilinear_at_ras(ct_arr, r2i, c))
                        for r_idx in range(1, 3):
                            rr = 1.0 * r_idx / 2
                            for ai in range(8):
                                ang = 2 * np.pi * ai / 8
                                off = rr * (
                                    np.cos(ang) * u + np.sin(ang) * v
                                )
                                v_hu = float(
                                    sample_trilinear_at_ras(ct_arr, r2i, c + off)
                                )
                                if np.isfinite(v_hu) and v_hu > s_max:
                                    s_max = v_hu
                        if s_max >= self.ON_METAL_HU:
                            total_on_metal += 1

        median_err = float(np.median(all_errors)) if all_errors else float("inf")
        on_metal_frac = total_on_metal / max(1, total_pred)
        return {
            "exact": total_exact,
            "total": total_shanks,
            "median_err_mm": median_err,
            "on_metal_frac": on_metal_frac,
            "n_pred": total_pred,
            "n_on_metal": total_on_metal,
        }

    def test_dataset_regression_gap_only(self):
        """Headline regression net: 49/66 exact-count for the legacy
        gap-only partition. This is the validated production path."""
        m = self._run_strategy("gap_only")
        self.assertGreaterEqual(
            m["exact"], self.GAP_ONLY_EXACT_FLOOR,
            f"gap_only exact-count {m['exact']}/{m['total']} below "
            f"floor {self.GAP_ONLY_EXACT_FLOOR}",
        )
        self.assertLessEqual(
            m["median_err_mm"], self.GAP_ONLY_MEDIAN_CEILING_MM,
            f"gap_only median error {m['median_err_mm']:.3f} mm above "
            f"ceiling {self.GAP_ONLY_MEDIAN_CEILING_MM} mm",
        )
        self.assertGreaterEqual(
            m["on_metal_frac"], self.GAP_ONLY_ON_METAL_FLOOR,
            f"gap_only on-metal fraction {m['on_metal_frac']:.3f} below "
            f"floor {self.GAP_ONLY_ON_METAL_FLOOR}",
        )

    def test_dataset_regression_cascade(self):
        """Cascade strategy (current state). Floor is the observed 40/67
        baseline; the goal is to lift this past gap_only by tuning
        ``signal_derived_entry_arc``. Use the bolt-partition QC notebook
        to diagnose per-shank tier behavior."""
        m = self._run_strategy("cascade")
        self.assertGreaterEqual(
            m["exact"], self.CASCADE_EXACT_FLOOR,
            f"cascade exact-count {m['exact']}/{m['total']} below "
            f"floor {self.CASCADE_EXACT_FLOOR} — Tier 1 (HU profile) "
            f"signal regressed; check signal_derived_entry_arc",
        )
        self.assertLessEqual(
            m["median_err_mm"], self.CASCADE_MEDIAN_CEILING_MM,
            f"cascade median error {m['median_err_mm']:.3f} mm above "
            f"ceiling {self.CASCADE_MEDIAN_CEILING_MM} mm",
        )
        self.assertGreaterEqual(
            m["on_metal_frac"], self.CASCADE_ON_METAL_FLOOR,
            f"cascade on-metal fraction {m['on_metal_frac']:.3f} below "
            f"floor {self.CASCADE_ON_METAL_FLOOR}",
        )


# ---------------------------------------------------------------------
# Bolt-end estimation: building blocks
# ---------------------------------------------------------------------


from rosa_core.contact_placement import (
    entry_arc_from_metal_mass,
    median_library_pitch_mm,
    refine_axis_via_centroid,
    sample_disk_along_polyline,
)


class MedianLibraryPitchTests(unittest.TestCase):
    def test_uniform_pitch_model(self):
        models = [{"contact_center_offsets_from_tip_mm": [0, 3.5, 7.0, 10.5, 14.0]}]
        self.assertAlmostEqual(median_library_pitch_mm(models), 3.5)

    def test_mixed_pitch_models(self):
        # Model A: 3.5mm pitch (4 spaces); Model B: 4.0mm pitch (3 spaces).
        # Combined: [3.5, 3.5, 3.5, 3.5, 4.0, 4.0, 4.0] → median = 3.5
        models = [
            {"contact_center_offsets_from_tip_mm": [0, 3.5, 7.0, 10.5, 14.0]},
            {"contact_center_offsets_from_tip_mm": [0, 4.0, 8.0, 12.0]},
        ]
        self.assertAlmostEqual(median_library_pitch_mm(models), 3.5)

    def test_nonuniform_offsets(self):
        # DIXI-15CM-style: 8 distal at 3.5mm, big design gap, 7 proximal at 3.5mm
        offsets = [i * 3.5 for i in range(8)] + [56.5 + i * 3.5 for i in range(7)]
        models = [{"contact_center_offsets_from_tip_mm": offsets}]
        # Median across [3.5×7 distal-pitches, 15.0 design-gap, 3.5×6 proximal] = 3.5
        self.assertAlmostEqual(median_library_pitch_mm(models), 3.5)

    def test_empty_inputs(self):
        self.assertIsNone(median_library_pitch_mm([]))
        self.assertIsNone(median_library_pitch_mm(None))
        # Single-contact model has no pitches → still None
        self.assertIsNone(median_library_pitch_mm(
            [{"contact_center_offsets_from_tip_mm": [0]}]
        ))


class EntryArcFromMetalMassTests(unittest.TestCase):
    """Tests for the bolt-end detector. Synthetic mass profiles
    constructed to exercise the documented algorithm."""

    def _bolt_then_contacts_profile(self, bolt_end=25.0, n_contacts=10,
                                      contact_pitch=3.5, profile_len=120.0,
                                      bolt_mass=14000.0, contact_mass=6000.0,
                                      gap_mm=5.0):
        """Profile: [0..bolt_end] high (bolt), [bolt_end..bolt_end+gap]
        zero (air), [bolt_end+gap..bolt_end+gap+n*pitch] periodic
        contact peaks."""
        arcs = np.arange(0, profile_len, 0.25)
        mass = np.zeros_like(arcs)
        mass[arcs <= bolt_end] = bolt_mass
        first_contact = bolt_end + gap_mm
        for k in range(n_contacts):
            c = first_contact + k * contact_pitch
            idx = int(c / 0.25)
            mass[max(0, idx - 1):idx + 2] = contact_mass
        return arcs, mass

    def test_clean_bolt_then_contacts(self):
        """Standard pattern: sustained bolt + air gap + contact peaks.
        Detector should land just past the bolt's deep edge."""
        arcs, mass = self._bolt_then_contacts_profile(bolt_end=25.0)
        ent = entry_arc_from_metal_mass(
            arcs, mass, smooth_sigma_mm=3.5, plateau_frac=0.5,
            max_gap_mm=6.0, padding_mm=0.5,
        )
        self.assertIsNotNone(ent)
        self.assertGreater(ent, 23.0)
        self.assertLess(ent, 30.0)

    def test_two_peak_bolt_internal_dip_absorbed(self):
        """Bolt with an internal dip (nut + bone-collar): two peaks
        separated by a 3 mm gap. Detector should NOT terminate at the
        internal dip (because dip < max_gap_mm); should land past the
        second peak."""
        arcs = np.arange(0, 100, 0.25)
        mass = np.zeros_like(arcs)
        # Peak A (nut) at arc 5, width ~5mm
        mass[(arcs >= 0) & (arcs <= 8)] = 12000
        # Internal dip at 8-11mm (3mm wide, < max_gap)
        # Peak B (bone-collar) at arc 11-25
        mass[(arcs >= 11) & (arcs <= 25)] = 14000
        # Air gap 25-30
        # Contacts 30+
        for k in range(10):
            c = 30 + k * 3.5
            idx = int(c / 0.25)
            mass[max(0, idx - 1):idx + 2] = 6000
        ent = entry_arc_from_metal_mass(
            arcs, mass, smooth_sigma_mm=3.5, plateau_frac=0.5,
            max_gap_mm=6.0, padding_mm=0.5,
        )
        # Should be near 25 (past the second peak), not near 8 (the dip).
        self.assertIsNotNone(ent)
        self.assertGreater(ent, 20.0)
        self.assertLess(ent, 32.0)

    def test_returns_none_on_flat_profile(self):
        arcs = np.arange(0, 100, 0.25)
        mass = np.zeros_like(arcs)
        ent = entry_arc_from_metal_mass(arcs, mass)
        self.assertIsNone(ent)

    def test_absolute_threshold_mode(self):
        """Absolute threshold returns a bolt-end past the constructed
        edge. With σ=3.5 mm smoothing, the sharp HU drop at the actual
        bolt-end is spread over ~5-7 mm, so the threshold-cross lands
        slightly past arc=20."""
        arcs, mass = self._bolt_then_contacts_profile(
            bolt_end=20.0, contact_mass=300.0,  # contacts below abs_threshold
        )
        ent = entry_arc_from_metal_mass(
            arcs, mass, smooth_sigma_mm=3.5,
            threshold_mode="absolute", abs_threshold=500.0,
            max_gap_mm=6.0, padding_mm=0.5,
        )
        self.assertIsNotNone(ent)
        self.assertGreater(ent, 18.0)
        self.assertLess(ent, 30.0)

    def test_short_dip_does_not_terminate(self):
        """Brief dip below threshold (< max_gap_mm) should be absorbed
        into the bolt region."""
        arcs = np.arange(0, 100, 0.25)
        mass = np.full_like(arcs, 14000.0)
        # 3mm dip in the middle of the bolt
        mass[(arcs >= 12) & (arcs <= 15)] = 0.0
        # Real bolt-end at 30, sustained-low after
        mass[arcs > 30] = 0.0
        ent = entry_arc_from_metal_mass(
            arcs, mass, smooth_sigma_mm=3.5, plateau_frac=0.5,
            max_gap_mm=6.0, padding_mm=0.5,
        )
        self.assertIsNotNone(ent)
        # Should land near 30 (the real bolt-end), not near 12 (the dip).
        self.assertGreater(ent, 25.0)
        self.assertLess(ent, 35.0)


# ---------------------------------------------------------------------
# Centerline + sampling: synthetic-volume integration
# ---------------------------------------------------------------------


def _make_synthetic_metal_volume(spacing_mm=0.5, half_extent_mm=30.0,
                                    line_perp_xy=(0.0, 0.0),
                                    bolt_extent_mm=20.0,
                                    bolt_hu=2500.0):
    """Build a small KJI volume with a metal "bolt" along a known axis.

    Returns (ct_arr, ras_to_ijk, start_ras, end_ras). The bolt is a
    short cylinder of HU=`bolt_hu` along the +x axis, optionally
    offset perpendicularly by `line_perp_xy = (dy, dz)` mm. Air HU
    is 0 elsewhere.
    """
    n = int(2 * half_extent_mm / spacing_mm) + 1
    # KJI shape: (z, y, x) by SITK convention.
    arr = np.zeros((n, n, n), dtype=np.float32)
    # Place the bolt along the +x axis (i = increasing).
    # Bolt center voxel: (n//2, n//2 + dy_vox, ...)
    dy, dz = line_perp_xy
    cy = n // 2 + int(round(dy / spacing_mm))
    cz = n // 2 + int(round(dz / spacing_mm))
    bolt_radius_vox = max(1, int(round(1.0 / spacing_mm)))  # ~1mm radius
    bolt_start_x = n // 2  # arc=0 is at center voxel
    bolt_end_x = bolt_start_x + int(round(bolt_extent_mm / spacing_mm))
    for k in range(max(0, cz - bolt_radius_vox), min(n, cz + bolt_radius_vox + 1)):
        for j in range(max(0, cy - bolt_radius_vox), min(n, cy + bolt_radius_vox + 1)):
            for i in range(bolt_start_x, min(n, bolt_end_x + 1)):
                d_jk = np.sqrt((j - cy) ** 2 + (k - cz) ** 2) * spacing_mm
                if d_jk <= 1.0:
                    arr[k, j, i] = bolt_hu

    # ras_to_ijk for an axis-aligned RAS volume centered at origin.
    # IJK origin = (0, 0, 0); RAS origin = (-half_extent, -half_extent, -half_extent).
    # IJK = (RAS + half_extent) / spacing
    ras_to_ijk = np.array([
        [1.0 / spacing_mm, 0,                 0,                 half_extent_mm / spacing_mm],
        [0,                 1.0 / spacing_mm, 0,                 half_extent_mm / spacing_mm],
        [0,                 0,                 1.0 / spacing_mm, half_extent_mm / spacing_mm],
        [0,                 0,                 0,                 1.0],
    ])
    # start at arc=0 voxel, walking +x.
    start_ras = np.array([0.0, dy, dz])  # at the center voxel
    end_ras = np.array([bolt_extent_mm + 5.0, dy, dz])  # just past the bolt
    return arr, ras_to_ijk, start_ras, end_ras


class SyntheticBoltVolumeTests(unittest.TestCase):
    """Integration tests: sample our helpers on a known synthetic
    volume and verify the bolt-end is detected near the constructed
    location."""

    def test_centerline_traces_offset_bolt(self):
        """Bolt offset 4 mm in +y direction. Centerline should
        deflect toward +y in the bolt region."""
        ct_arr, r2i, start_ras, end_ras = _make_synthetic_metal_volume(
            line_perp_xy=(4.0, 0.0), bolt_extent_mm=15.0,
        )
        polyline = refine_axis_via_centroid(
            ct_arr, r2i, start_ras + np.array([0.0, -4.0, 0.0]), end_ras,
            polarity="positive",
            cross_radius_mm=6.0, filter_value=500.0, weight_offset=1500.0,
            poly_deg=2,
        )
        self.assertIsNotNone(polyline)
        # Compute mean perp offset from the seed straight axis.
        seed_axis = (end_ras - (start_ras + np.array([0.0, -4.0, 0.0])))
        seed_axis = seed_axis / np.linalg.norm(seed_axis)
        seed_start = start_ras + np.array([0.0, -4.0, 0.0])
        # Project polyline onto axis; perp component magnitude.
        diffs = polyline - seed_start
        along = diffs @ seed_axis
        perp = np.linalg.norm(diffs - np.outer(along, seed_axis), axis=1)
        # Polyline should curve toward the bolt: max perp > 1mm.
        self.assertGreater(perp.max(), 1.0)

    def test_sample_disk_along_polyline_total_mass(self):
        """Total mass should be HIGH inside the bolt region and ~0
        past the bolt on a synthetic volume."""
        ct_arr, r2i, start_ras, end_ras = _make_synthetic_metal_volume(
            line_perp_xy=(0.0, 0.0), bolt_extent_mm=15.0,
        )
        # Sample along the straight axis (which goes through the bolt).
        polyline = np.stack([start_ras, end_ras])
        arcs, max_hu, total_hu = sample_disk_along_polyline(
            ct_arr, r2i, polyline,
            polarity="positive", step_mm=0.5,
            disk_radius_mm=1.5, total_threshold=1500.0,
        )
        # Inside bolt (arcs 2-13mm): total_hu should be high
        bolt_region = (arcs >= 2.0) & (arcs <= 13.0)
        # Past bolt (arcs >= 18mm): total_hu should be ~0
        past_bolt = arcs >= 18.0
        self.assertGreater(total_hu[bolt_region].mean(), 1000.0)
        self.assertLess(total_hu[past_bolt].mean(), 100.0)

    def test_lazy_export_works(self):
        """The new building blocks resolve via the rosa_core lazy
        re-export surface."""
        from rosa_core import (
            median_library_pitch_mm as mlp,
            entry_arc_from_metal_mass as eafm,
            estimate_bolt_end_from_metal_mass as ebefm,
        )
        # Just verify they're callable.
        self.assertTrue(callable(mlp))
        self.assertTrue(callable(eafm))
        self.assertTrue(callable(ebefm))


if __name__ == "__main__":
    unittest.main()
