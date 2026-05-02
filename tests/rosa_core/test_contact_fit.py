import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))


# Wrap the scientific-stack imports so the module loads cleanly in
# minimal envs (no numpy / rosa_core). The class skipUnless below
# prevents test bodies from running when names are unbound.
try:
    import numpy as np  # noqa: E402
    from rosa_core.contact_fit import (  # noqa: E402
        _axis_support_run_from_clusters,
        _configured_gap_priors_mm,
        angle_deg,
        fit_electrode_axis_and_tip,
        point_line_distance,
        unit,
    )
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


@unittest.skipUnless(
    DEPS_AVAILABLE,
    "numpy / rosa_core.contact_fit not importable in this environment.",
)
class ContactFitTests(unittest.TestCase):
    def test_unit_and_angle(self):
        u = unit([3, 0, 0])
        self.assertAlmostEqual(float(np.linalg.norm(u)), 1.0, places=7)
        self.assertAlmostEqual(angle_deg([1, 0, 0], [0, 1, 0]), 90.0, places=6)

    def test_point_line_distance(self):
        points = np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
        dist = point_line_distance(points, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        self.assertAlmostEqual(float(dist[0]), 1.0, places=6)
        self.assertAlmostEqual(float(dist[1]), 2.0, places=6)

    def test_fit_electrode_axis_and_tip_success(self):
        rng = np.random.default_rng(7)
        # Planned trajectory along +X from entry to target.
        planned_entry = np.array([0.0, 0.0, 0.0])
        planned_target = np.array([50.0, 0.0, 0.0])

        # Actual points are near a slightly shifted axis.
        x = rng.uniform(0.0, 50.0, size=2000)
        y = rng.normal(loc=0.5, scale=0.15, size=2000)
        z = rng.normal(loc=-0.2, scale=0.15, size=2000)
        pts = np.column_stack([x, y, z])

        result = fit_electrode_axis_and_tip(
            candidate_points_lps=pts,
            planned_entry_lps=planned_entry,
            planned_target_lps=planned_target,
            contact_offsets_mm=[0.0, 3.5, 7.0, 10.5],
            tip_at="target",
            roi_radius_mm=3.0,
            max_angle_deg=12.0,
            max_depth_shift_mm=20.0,
        )

        self.assertTrue(result["success"], msg=str(result))
        self.assertLess(result["angle_deg"], 3.0)
        self.assertGreater(result["points_in_roi"], 100)
        self.assertIn("entry_lps", result)
        self.assertIn("target_lps", result)

    def test_fit_electrode_axis_and_tip_failure_no_points(self):
        result = fit_electrode_axis_and_tip(
            candidate_points_lps=[],
            planned_entry_lps=[0.0, 0.0, 0.0],
            planned_target_lps=[10.0, 0.0, 0.0],
            contact_offsets_mm=[0.0, 3.5],
        )
        self.assertFalse(result["success"])
        self.assertIn("No candidate points", result["reason"])

    def test_fit_electrode_axis_and_tip_retries_with_wider_roi(self):
        rng = np.random.default_rng(11)
        planned_entry = np.array([0.0, 0.0, 0.0])
        planned_target = np.array([50.0, 0.0, 0.0])

        x = rng.uniform(0.0, 50.0, size=1600)
        y = rng.normal(loc=4.2, scale=0.25, size=1600)
        z = rng.normal(loc=0.0, scale=0.20, size=1600)
        pts = np.column_stack([x, y, z])

        result = fit_electrode_axis_and_tip(
            candidate_points_lps=pts,
            planned_entry_lps=planned_entry,
            planned_target_lps=planned_target,
            contact_offsets_mm=[0.0, 3.5, 7.0, 10.5],
            tip_at="target",
            roi_radius_mm=3.0,
            max_angle_deg=12.0,
            max_depth_shift_mm=2.0,
        )

        self.assertTrue(result["success"], msg=str(result))
        self.assertEqual(int(result["fit_attempt_index"]), 2)
        self.assertAlmostEqual(float(result["roi_radius_mm_used"]), 5.0, places=6)
        self.assertGreaterEqual(int(result["slab_centroids"]), 8)

    def test_fit_electrode_axis_and_tip_deep_anchor_v2_uses_bead_chain(self):
        rng = np.random.default_rng(17)
        planned_entry = np.array([0.0, 0.0, 0.0])
        planned_target = np.array([50.0, 0.0, 0.0])

        bead_centers = [np.array([x, 4.0, 0.0]) for x in np.linspace(4.0, 48.0, 12)]
        bead_points = []
        for center in bead_centers:
            bead_points.append(center + rng.normal(scale=[0.35, 0.20, 0.20], size=(60, 3)))
        shallow_blob = np.column_stack(
            [
                rng.normal(loc=2.0, scale=1.2, size=350),
                rng.normal(loc=4.0, scale=0.35, size=350),
                rng.normal(loc=0.0, scale=0.25, size=350),
            ]
        )
        pts = np.vstack(bead_points + [shallow_blob])

        result = fit_electrode_axis_and_tip(
            candidate_points_lps=pts,
            planned_entry_lps=planned_entry,
            planned_target_lps=planned_target,
            contact_offsets_mm=[0.0, 3.5, 7.0, 10.5],
            tip_at="target",
            roi_radius_mm=5.0,
            max_angle_deg=12.0,
            max_depth_shift_mm=2.0,
            fit_mode="deep_anchor_v2",
        )

        self.assertTrue(result["success"], msg=str(result))
        self.assertEqual(str(result["fit_mode_used"]), "deep_anchor_v2")
        self.assertGreaterEqual(int(result["compact_cluster_count"]), 3)
        self.assertGreaterEqual(int(result["bead_cluster_inliers"]), 3)
        self.assertIn("terminal_blob_diagnostics_json", result)
        self.assertIn("terminal_blob_source", result)
        self.assertIn("terminal_anchor_mode", result)
        self.assertLess(abs(float(result["entry_lps"][1]) - 4.0), 0.8)
        self.assertLess(abs(float(result["target_lps"][1]) - 4.0), 0.8)

    def test_fit_electrode_axis_and_tip_deep_anchor_v2_ignores_far_terminal_outlier(self):
        rng = np.random.default_rng(23)
        planned_entry = np.array([0.0, 0.0, 0.0])
        planned_target = np.array([50.0, 0.0, 0.0])

        main_centers = [np.array([x, 3.5, 0.0]) for x in np.arange(6.0, 42.1, 3.5)]
        main_points = [center + rng.normal(scale=[0.25, 0.18, 0.18], size=(50, 3)) for center in main_centers]
        outlier_center = np.array([49.0, 3.5, 0.0])
        outlier_points = outlier_center + rng.normal(scale=[0.20, 0.16, 0.16], size=(45, 3))
        pts = np.vstack(main_points + [outlier_points])

        result = fit_electrode_axis_and_tip(
            candidate_points_lps=pts,
            planned_entry_lps=planned_entry,
            planned_target_lps=planned_target,
            contact_offsets_mm=[0.0, 3.5, 7.0, 10.5, 14.0, 17.5],
            tip_at="target",
            roi_radius_mm=5.0,
            max_angle_deg=12.0,
            max_depth_shift_mm=10.0,
            fit_mode="deep_anchor_v2",
        )

        self.assertTrue(result["success"], msg=str(result))
        self.assertIn(str(result["deep_anchor_source"]), {"terminal_blob_centroid", "terminal_bead_chain", "axis_support_run"})
        self.assertLess(float(result["target_lps"][0]), 45.0)
        self.assertGreater(float(result["target_lps"][0]), 41.0)

    def test_fit_electrode_axis_and_tip_deep_anchor_v2_stops_on_off_chain_terminal_centroid(self):
        rng = np.random.default_rng(29)
        planned_entry = np.array([0.0, 0.0, 0.0])
        planned_target = np.array([50.0, 0.0, 0.0])

        main_centers = [np.array([x, 2.5, 0.0]) for x in np.arange(6.0, 42.1, 3.5)]
        main_points = [center + rng.normal(scale=[0.22, 0.16, 0.16], size=(45, 3)) for center in main_centers]
        crossing_center = np.array([44.0, 4.8, 0.0])
        crossing_points = crossing_center + rng.normal(scale=[0.18, 0.14, 0.14], size=(40, 3))
        pts = np.vstack(main_points + [crossing_points])

        result = fit_electrode_axis_and_tip(
            candidate_points_lps=pts,
            planned_entry_lps=planned_entry,
            planned_target_lps=planned_target,
            contact_offsets_mm=[0.0, 3.5, 7.0, 10.5, 14.0, 17.5],
            tip_at="target",
            roi_radius_mm=5.0,
            max_angle_deg=12.0,
            max_depth_shift_mm=10.0,
            fit_mode="deep_anchor_v2",
        )

        self.assertTrue(result["success"], msg=str(result))
        self.assertIn(str(result["deep_anchor_source"]), {"terminal_blob_centroid", "axis_support_run"})
        self.assertLess(float(result["target_lps"][0]), 43.0)

    def test_fit_electrode_axis_and_tip_deep_anchor_v2_handles_single_string_cluster(self):
        rng = np.random.default_rng(31)
        planned_entry = np.array([0.0, 0.0, 0.0])
        planned_target = np.array([50.0, 0.0, 0.0])

        x = rng.uniform(3.0, 48.0, size=1800)
        y = rng.normal(loc=2.0, scale=0.20, size=1800)
        z = rng.normal(loc=0.0, scale=0.18, size=1800)
        pts = np.column_stack([x, y, z])

        result = fit_electrode_axis_and_tip(
            candidate_points_lps=pts,
            planned_entry_lps=planned_entry,
            planned_target_lps=planned_target,
            contact_offsets_mm=[0.0, 3.5, 7.0, 10.5],
            tip_at="target",
            roi_radius_mm=5.0,
            max_angle_deg=12.0,
            max_depth_shift_mm=2.0,
            fit_mode="deep_anchor_v2",
        )

        self.assertTrue(result["success"], msg=str(result))
        self.assertEqual(str(result["terminal_blob_source"]), "single_string_cluster")
        self.assertEqual(str(result["terminal_anchor_mode"]), "tight_extent")
        self.assertEqual(str(result["local_terminal_morphology"]), "string_like")
        self.assertLess(abs(float(result["target_lps"][1]) - 2.0), 0.8)

    def test_fit_electrode_axis_and_tip_em_v1_handles_single_string_cluster(self):
        rng = np.random.default_rng(37)
        planned_entry = np.array([0.0, 0.0, 0.0])
        planned_target = np.array([50.0, 0.0, 0.0])

        x = rng.uniform(2.0, 48.0, size=1800)
        y = rng.normal(loc=1.8, scale=0.20, size=1800)
        z = rng.normal(loc=0.0, scale=0.18, size=1800)
        pts = np.column_stack([x, y, z])

        result = fit_electrode_axis_and_tip(
            candidate_points_lps=pts,
            planned_entry_lps=planned_entry,
            planned_target_lps=planned_target,
            contact_offsets_mm=[0.0, 3.5, 7.0, 10.5],
            tip_at="target",
            roi_radius_mm=5.0,
            max_angle_deg=12.0,
            max_depth_shift_mm=2.0,
            fit_mode="em_v1",
        )

        self.assertTrue(result["success"], msg=str(result))
        self.assertEqual(str(result["fit_mode_used"]), "em_v1")
        self.assertEqual(str(result["deep_anchor_source"]), "em_support_run")
        self.assertIn("axis_support_selected_run_json", result)
        self.assertLess(abs(float(result["target_lps"][1]) - 1.8), 0.8)

    def test_fit_electrode_axis_and_tip_em_v1_handles_bead_chain(self):
        rng = np.random.default_rng(41)
        planned_entry = np.array([0.0, 0.0, 0.0])
        planned_target = np.array([50.0, 0.0, 0.0])

        centers = [np.array([x, 3.0, 0.0]) for x in np.linspace(5.0, 45.0, 10)]
        pts = np.vstack([center + rng.normal(scale=[0.30, 0.20, 0.20], size=(55, 3)) for center in centers])

        result = fit_electrode_axis_and_tip(
            candidate_points_lps=pts,
            planned_entry_lps=planned_entry,
            planned_target_lps=planned_target,
            contact_offsets_mm=[0.0, 3.5, 7.0, 10.5, 14.0, 17.5],
            tip_at="target",
            roi_radius_mm=5.0,
            max_angle_deg=12.0,
            max_depth_shift_mm=4.0,
            fit_mode="em_v1",
        )

        self.assertTrue(result["success"], msg=str(result))
        self.assertEqual(str(result["fit_mode_used"]), "em_v1")
        self.assertEqual(str(result["terminal_anchor_mode"]), "support_run_interval")
        self.assertLess(abs(float(result["entry_lps"][1]) - 3.0), 0.8)
        self.assertLess(abs(float(result["target_lps"][1]) - 3.0), 0.8)

    def test_configured_gap_priors_detect_large_programmed_gaps(self):
        gaps = _configured_gap_priors_mm([0.0, 3.5, 7.0, 10.5, 14.0, 27.0, 30.5])
        self.assertEqual(gaps, [13.0])

    def test_axis_support_run_prefers_run_near_proposed_segment(self):
        def make_summary(cluster_id: int, t_center: float) -> dict[str, object]:
            center = np.array([t_center, 0.0, 0.0], dtype=float)
            offsets = np.linspace(-0.7, 0.7, 9)
            pts = np.column_stack([t_center + offsets, np.zeros_like(offsets), np.zeros_like(offsets)])
            return {
                "cluster_id": cluster_id,
                "center": center,
                "axis": np.array([1.0, 0.0, 0.0], dtype=float),
                "count": int(pts.shape[0]),
                "span_mm": 1.4,
                "radial_rms_mm": 0.0,
                "points": pts,
            }

        run_a = [make_summary(i, t) for i, t in enumerate([-28.0, -22.0, -16.0])]
        run_b = [make_summary(i + 10, t) for i, t in enumerate([18.0, 24.0, 30.0])]
        summaries = run_a + run_b

        axis_support = _axis_support_run_from_clusters(
            summaries,
            core_cluster_ids={item["cluster_id"] for item in summaries},
            line_point=np.zeros(3, dtype=float),
            line_axis=np.array([1.0, 0.0, 0.0], dtype=float),
            planned_entry_t_mm=16.0,
            planned_target_t_mm=32.0,
            radial_limit_mm=1.0,
            max_gap_mm=7.0,
        )

        self.assertIsNotNone(axis_support["selected_run"])
        self.assertEqual(axis_support["selected_run"]["cluster_ids"], [10, 11, 12])
        self.assertGreater(float(axis_support["selected_run"]["score"]), 0.0)


if __name__ == "__main__":
    unittest.main()
