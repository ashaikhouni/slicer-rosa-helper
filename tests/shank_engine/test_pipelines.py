import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))

from shank_engine import BlobRecord, PipelineRegistry, register_builtin_pipelines  # noqa: E402
from shank_engine.contracts import ShankModel  # noqa: E402
from shank_engine.pipelines.blob_em_v2 import (  # noqa: E402
    _EMBlobExtractor,
    _EMBlobScorer,
    _EMInitializer,
    _EMSelector,
    _supports_for_line,
)



def _kji_to_ras_points_identity(idx):
    arr = np.asarray(idx, dtype=float).reshape(-1, 3)
    return np.stack([arr[:, 2], arr[:, 1], arr[:, 0]], axis=1)



def _ras_to_ijk_identity(ras_xyz):
    xyz = np.asarray(ras_xyz, dtype=float).reshape(-1)
    return np.asarray([xyz[0], xyz[1], xyz[2]], dtype=float)


class _OverrideBlobExtractor:
    def extract(self, ctx, state):
        return [
            BlobRecord(
                blob_id=1,
                centroid_ras=(0.0, 0.0, 0.0),
                centroid_kji=(0.0, 0.0, 0.0),
                voxel_count=7,
            )
        ]


class _OverrideGating:
    def compute(self, ctx, state):
        return {
            "candidate_count": 1,
            "metal_in_head_count": 1,
            "depth_kept_count": 1,
            "gating_mask_type": "test",
            "inside_method": "test",
        }


class _MaskSourceGating:
    def compute(self, ctx, state):
        metal_in_head = np.zeros((6, 6, 6), dtype=np.uint8)
        metal_in_head[1, 1, 1] = 1
        metal_in_head[4, 4, 4] = 1
        # Depth-pass intentionally different so source choice is testable.
        metal_depth_pass = np.zeros((6, 6, 6), dtype=np.uint8)
        metal_depth_pass[2, 2, 2] = 1
        return {
            "candidate_count": 3,
            "metal_in_head_count": 2,
            "depth_kept_count": 1,
            "gating_mask_type": "test",
            "inside_method": "test",
            "metal_in_head_mask_kji": metal_in_head,
            "metal_depth_pass_mask_kji": metal_depth_pass,
            "head_distance_map_kji": np.ones((6, 6, 6), dtype=np.float32),
        }


class _NoopInitializer:
    def initialize(self, ctx, blobs, state):
        return []


def _make_bar_points(length=24, thickness=3, k0=10, j0=10, i0=4):
    pts = []
    half = int(thickness // 2)
    for i in range(i0, i0 + int(length)):
        for dk in range(-half, half + 1):
            for dj in range(-half, half + 1):
                pts.append((k0 + dk, j0 + dj, i))
    return np.asarray(pts, dtype=float).reshape(-1, 3)


def _make_wavy_bar_points(length=24, thickness=3, k0=12, j0=12, i0=4):
    pts = []
    half = int(thickness // 2)
    for di in range(int(length)):
        i = i0 + di
        j_center = j0 + int(round(2.0 * np.sin(2.0 * np.pi * float(di) / max(6.0, float(length)))))
        for dk in range(-half, half + 1):
            for dj in range(-half, half + 1):
                pts.append((k0 + dk, j_center + dj, i))
    return np.asarray(pts, dtype=float).reshape(-1, 3)


class PipelinesTests(unittest.TestCase):
    def test_all_pipelines_share_run_signature(self):
        reg = PipelineRegistry()
        register_builtin_pipelines(reg)
        for key in reg.keys():
            pipe = reg.create_pipeline(key)
            self.assertTrue(callable(getattr(pipe, "run", None)))

    def test_blob_em_v2_scaffold_schema(self):
        reg = PipelineRegistry()
        register_builtin_pipelines(reg)
        pipe = reg.create_pipeline("blob_em_v2")
        out = pipe.run({"run_id": "r1", "config": {}})
        self.assertEqual(out["pipeline_id"], "blob_em_v2")
        self.assertEqual(out["status"], "ok")
        self.assertIn("diagnostics", out)
        self.assertIn("warnings", out)
        self.assertIn("timing", out["diagnostics"])

    def test_blob_persistence_v1_schema(self):
        reg = PipelineRegistry()
        register_builtin_pipelines(reg)
        pipe = reg.create_pipeline("blob_persistence_v1")
        out = pipe.run({"run_id": "r-persist-empty", "config": {}})
        self.assertEqual(out["pipeline_id"], "blob_persistence_v1")
        self.assertEqual(out["status"], "ok")
        self.assertIn("diagnostics", out)
        self.assertIsInstance(out.get("trajectories"), list)
        self.assertIsInstance(out.get("contacts"), list)

    def test_blob_consensus_v1_schema(self):
        reg = PipelineRegistry()
        register_builtin_pipelines(reg)
        pipe = reg.create_pipeline("blob_consensus_v1")
        out = pipe.run({"run_id": "r-consensus-empty", "config": {}})
        self.assertEqual(out["pipeline_id"], "blob_consensus_v1")
        self.assertEqual(out["status"], "ok")
        self.assertIn("diagnostics", out)
        self.assertIsInstance(out.get("trajectories"), list)
        self.assertIsInstance(out.get("contacts"), list)

    def test_blob_persistence_v2_schema(self):
        reg = PipelineRegistry()
        register_builtin_pipelines(reg)
        pipe = reg.create_pipeline("blob_persistence_v2")
        out = pipe.run({"run_id": "r-persist-v2-empty", "config": {}})
        self.assertEqual(out["pipeline_id"], "blob_persistence_v2")
        self.assertEqual(out["status"], "ok")
        self.assertIn("diagnostics", out)
        self.assertIsInstance(out.get("trajectories"), list)
        self.assertIsInstance(out.get("contacts"), list)

    def test_hybrid_bead_string_v1_schema(self):
        reg = PipelineRegistry()
        register_builtin_pipelines(reg)
        pipe = reg.create_pipeline("hybrid_bead_string_v1")
        out = pipe.run({"run_id": "r-hybrid-empty", "config": {}})
        self.assertEqual(out["pipeline_id"], "hybrid_bead_string_v1")
        self.assertEqual(out["status"], "ok")
        self.assertIn("diagnostics", out)
        self.assertIsInstance(out.get("trajectories"), list)
        self.assertIsInstance(out.get("contacts"), list)

    def test_shank_grow_v1_schema(self):
        reg = PipelineRegistry()
        register_builtin_pipelines(reg)
        pipe = reg.create_pipeline("shank_grow_v1")
        out = pipe.run({"run_id": "r-grow-empty", "config": {}})
        self.assertEqual(out["pipeline_id"], "shank_grow_v1")
        self.assertEqual(out["status"], "ok")
        self.assertIn("diagnostics", out)
        self.assertIsInstance(out.get("trajectories"), list)
        self.assertIsInstance(out.get("contacts"), list)

    def test_shank_cluster_v1_schema(self):
        reg = PipelineRegistry()
        register_builtin_pipelines(reg)
        pipe = reg.create_pipeline("shank_cluster_v1")
        out = pipe.run({"run_id": "r-cluster-empty", "config": {}})
        self.assertEqual(out["pipeline_id"], "shank_cluster_v1")
        self.assertEqual(out["status"], "ok")
        self.assertIn("diagnostics", out)
        self.assertIsInstance(out.get("trajectories"), list)
        self.assertIsInstance(out.get("contacts"), list)

    def test_de_novo_seed_extend_v2_schema(self):
        reg = PipelineRegistry()
        register_builtin_pipelines(reg)
        pipe = reg.create_pipeline("de_novo_seed_extend_v2")
        out = pipe.run({"run_id": "r-denovo-seed-extend-empty", "config": {}})
        self.assertEqual(out["pipeline_id"], "de_novo_seed_extend_v2")
        self.assertEqual(out["status"], "ok")
        self.assertIn("diagnostics", out)
        self.assertIsInstance(out.get("trajectories"), list)
        self.assertIsInstance(out.get("contacts"), list)

    def test_de_novo_hypothesis_select_v1_schema(self):
        reg = PipelineRegistry()
        register_builtin_pipelines(reg)
        pipe = reg.create_pipeline("de_novo_hypothesis_select_v1")
        out = pipe.run({"run_id": "r-denovo-empty", "config": {}})
        self.assertEqual(out["pipeline_id"], "de_novo_hypothesis_select_v1")
        self.assertEqual(out["status"], "ok")
        self.assertIn("diagnostics", out)
        self.assertIsInstance(out.get("trajectories"), list)
        self.assertIsInstance(out.get("contacts"), list)

    def test_shank_graph_v1_schema(self):
        reg = PipelineRegistry()
        register_builtin_pipelines(reg)
        pipe = reg.create_pipeline("shank_graph_v1")
        out = pipe.run({"run_id": "r-graph-empty", "config": {}})
        self.assertEqual(out["pipeline_id"], "shank_graph_v1")
        self.assertEqual(out["status"], "ok")
        self.assertIn("diagnostics", out)
        self.assertIsInstance(out.get("trajectories"), list)
        self.assertIsInstance(out.get("contacts"), list)

    def test_shank_hypothesis_v1_schema(self):
        reg = PipelineRegistry()
        register_builtin_pipelines(reg)
        pipe = reg.create_pipeline("shank_hypothesis_v1")
        out = pipe.run({"run_id": "r-hyp-empty", "config": {}})
        self.assertEqual(out["pipeline_id"], "shank_hypothesis_v1")
        self.assertEqual(out["status"], "ok")
        self.assertIn("diagnostics", out)
        self.assertIsInstance(out.get("trajectories"), list)
        self.assertIsInstance(out.get("contacts"), list)

    def test_shank_stitch_v1_schema(self):
        reg = PipelineRegistry()
        register_builtin_pipelines(reg)
        pipe = reg.create_pipeline("shank_stitch_v1")
        out = pipe.run({"run_id": "r-stitch-empty", "config": {}})
        self.assertEqual(out["pipeline_id"], "shank_stitch_v1")
        self.assertEqual(out["status"], "ok")
        self.assertIn("diagnostics", out)
        self.assertIsInstance(out.get("trajectories"), list)
        self.assertIsInstance(out.get("contacts"), list)

    def test_blob_em_v2_component_override(self):
        reg = PipelineRegistry()
        register_builtin_pipelines(reg)
        pipe = reg.create_pipeline("blob_em_v2")
        out = pipe.run(
            {
                "run_id": "r1-override",
                "config": {},
                "arr_kji": np.zeros((2, 2, 2), dtype=np.float32),
                "ijk_kji_to_ras_fn": _kji_to_ras_points_identity,
                "components": {
                    "gating": _OverrideGating(),
                    "blob_extractor": _OverrideBlobExtractor(),
                },
            }
        )
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["diagnostics"]["counts"].get("blob_count_total"), 1)

    def test_blob_em_v2_uses_metal_in_head_blob_source(self):
        reg = PipelineRegistry()
        register_builtin_pipelines(reg)
        pipe = reg.create_pipeline("blob_em_v2")
        out = pipe.run(
            {
                "run_id": "r-metal-source",
                "config": {"use_distance_mask_for_blob_candidates": False},
                "arr_kji": np.full((6, 6, 6), 2500.0, dtype=np.float32),
                "ijk_kji_to_ras_fn": _kji_to_ras_points_identity,
                "components": {
                    "gating": _MaskSourceGating(),
                    "initializer": _NoopInitializer(),
                },
            }
        )
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["diagnostics"]["extras"].get("blob_source"), "metal_in_head")
        self.assertEqual(out["diagnostics"]["counts"].get("blob_count_total"), 2)

    def test_blob_em_v2_can_use_distance_pass_blob_source(self):
        reg = PipelineRegistry()
        register_builtin_pipelines(reg)
        pipe = reg.create_pipeline("blob_em_v2")
        out = pipe.run(
            {
                "run_id": "r-depth-source",
                "config": {},
                "arr_kji": np.full((6, 6, 6), 2500.0, dtype=np.float32),
                "ijk_kji_to_ras_fn": _kji_to_ras_points_identity,
                "components": {
                    "gating": _MaskSourceGating(),
                    "initializer": _NoopInitializer(),
                },
            }
        )
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["diagnostics"]["extras"].get("blob_source"), "metal_depth_pass")
        self.assertEqual(out["diagnostics"]["counts"].get("blob_count_total"), 1)

    def test_blob_em_v2_initializer_segment_first(self):
        init = _EMInitializer()
        blobs = []
        for i in range(8):
            blobs.append(
                BlobRecord(
                    blob_id=i + 1,
                    centroid_ras=(float(i), 0.0, 0.0),
                    centroid_kji=(0.0, 0.0, float(i)),
                    voxel_count=15,
                    length_mm=8.0,
                    pca_axis_ras=(1.0, 0.0, 0.0),
                    scores={"p_segment": 0.9, "p_bead": 0.1, "support_weight": 2.0},
                )
            )
        state = {}
        models = init.initialize(
            {"config": {"max_lines": 6, "min_inliers": 100, "inlier_radius_mm": 1.2, "min_length_mm": 5.0}},
            blobs,
            state,
        )
        self.assertGreaterEqual(len(models), 1)
        self.assertEqual(state["seed_stats"]["seed_count_bead_pair"], 0)
        self.assertGreater(state["seed_stats"]["seed_count_segment"], 0)

    def test_blob_em_v2_seed_nms_removes_collinear_duplicates(self):
        init = _EMInitializer()
        seeds = [
            {"point": np.array([0.0, 0.0, 0.0]), "direction": np.array([1.0, 0.0, 0.0]), "score": 10.0, "seed_type": "segment", "blob_i": 1, "blob_j": 1},
            {"point": np.array([0.5, 0.2, 0.0]), "direction": np.array([1.0, 0.02, 0.0]), "score": 9.0, "seed_type": "segment", "blob_i": 2, "blob_j": 2},
            {"point": np.array([0.0, 6.0, 0.0]), "direction": np.array([0.0, 1.0, 0.0]), "score": 8.0, "seed_type": "segment", "blob_i": 3, "blob_j": 3},
        ]
        kept = init._nms_seeds(seeds, {"seed_nms_angle_deg": 8.0, "seed_nms_line_distance_mm": 1.0}, max_keep=10)
        self.assertEqual(len(kept), 2)

    def test_blob_em_v2_bead_seeds_go_through_nms(self):
        init = _EMInitializer()
        blobs = []
        for i in range(10):
            blobs.append(
                BlobRecord(
                    blob_id=i + 1,
                    centroid_ras=(float(i), 0.0, 0.0),
                    centroid_kji=(0.0, 0.0, float(i)),
                    voxel_count=8,
                    scores={"p_bead": 0.95, "p_segment": 0.05, "support_weight": 1.5},
                )
            )
        points = np.asarray([b.centroid_ras for b in blobs], dtype=float)
        raw = init._bead_pair_seeds(blobs, points, {"max_lines": 8})
        kept = init._nms_seeds(raw, {"seed_nms_angle_deg": 8.0, "seed_nms_line_distance_mm": 1.0}, max_keep=16)
        self.assertGreater(len(raw), 0)
        self.assertLess(len(kept), len(raw))

    def test_blob_em_v2_coverage_triggered_bead_rescue(self):
        init = _EMInitializer()
        blobs = []
        # Strong segment corridor.
        for i in range(6):
            blobs.append(
                BlobRecord(
                    blob_id=i + 1,
                    centroid_ras=(float(i * 5), 0.0, 0.0),
                    centroid_kji=(0.0, 0.0, float(i)),
                    voxel_count=16,
                    length_mm=10.0,
                    pca_axis_ras=(1.0, 0.0, 0.0),
                    scores={"p_segment": 0.95, "p_bead": 0.05, "p_junk": 0.0, "support_weight": 3.0, "depth_prior": 1.0},
                )
            )
        # Bead-only residual corridor.
        for i in range(6):
            blobs.append(
                BlobRecord(
                    blob_id=100 + i,
                    centroid_ras=(40.0, float(i * 5), 0.0),
                    centroid_kji=(0.0, 0.0, float(i)),
                    voxel_count=10,
                    length_mm=2.0,
                    pca_axis_ras=(0.0, 1.0, 0.0),
                    scores={"p_segment": 0.01, "p_bead": 0.95, "p_junk": 0.0, "support_weight": 2.0, "depth_prior": 1.0},
                )
            )
        state = {}
        _ = init.initialize(
            {
                "config": {
                    "max_lines": 10,
                    "min_inliers": 50,
                    "inlier_radius_mm": 1.2,
                    "min_length_mm": 5.0,
                    "coverage_min_ratio_for_no_rescue": 0.80,
                    "rescue_only_unassigned_min_p_bead": 0.60,
                    "seed_nms_angle_deg": 6.0,
                    "seed_nms_line_distance_mm": 1.0,
                }
            },
            blobs,
            state,
        )
        ss = state["seed_stats"]
        self.assertEqual(int(ss["bead_rescue_triggered"]), 1)
        self.assertGreater(int(ss["seed_count_bead_rescue"]), 0)
        self.assertGreater(int(ss["shanks_after_second_pass"]), int(ss["shanks_after_first_pass"]))

    def test_blob_em_v2_mixed_evidence_uses_segment_alignment(self):
        p0 = np.asarray([0.0, 0.0, 0.0], dtype=float)
        direction = np.asarray([1.0, 0.0, 0.0], dtype=float)
        points = np.asarray([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float)
        blobs = [
            BlobRecord(
                blob_id=1,
                centroid_ras=(10.0, 0.0, 0.0),
                centroid_kji=(0.0, 0.0, 0.0),
                voxel_count=10,
                pca_axis_ras=(1.0, 0.0, 0.0),
                scores={"p_bead": 0.0, "p_segment": 1.0, "p_junk": 0.0, "support_weight": 1.0, "depth_prior": 1.0},
            ),
            BlobRecord(
                blob_id=2,
                centroid_ras=(10.0, 0.0, 0.0),
                centroid_kji=(0.0, 0.0, 0.0),
                voxel_count=10,
                pca_axis_ras=(0.0, 1.0, 0.0),
                scores={"p_bead": 0.0, "p_segment": 1.0, "p_junk": 0.0, "support_weight": 1.0, "depth_prior": 1.0},
            ),
        ]
        stat = _supports_for_line(points, blobs, p0, direction, {"inlier_radius_mm": 1.2}, active_weights=np.ones((2,), dtype=float))
        self.assertGreater(float(stat["contrib"][0]), float(stat["contrib"][1]))

    def test_blob_em_v2_selector_accepts_sparse_weighted_shank(self):
        selector = _EMSelector()
        model = ShankModel(
            shank_id="S1",
            kind="line",
            params={"point_ras": [0.0, 0.0, 0.0], "direction_ras": [1.0, 0.0, 0.0], "t_min": -10.0, "t_max": 30.0},
            support={"support_mass": 2.0, "span_mm": 40.0, "bead_support_mass": 0.8, "segment_support_mass": 1.2},
            assigned_blob_ids=(1,),
        )
        state = {}
        kept = selector.select({"config": {"min_length_mm": 20.0, "selector_min_support_mass": 1.6}}, [], [model], state)
        self.assertEqual(len(kept), 1)

    def test_blob_em_v2_contacts_deferred_but_valid(self):
        reg = PipelineRegistry()
        register_builtin_pipelines(reg)
        pipe = reg.create_pipeline("blob_em_v2")
        out = pipe.run(
            {
                "run_id": "r-contacts",
                "config": {"max_lines": 8},
                "arr_kji": np.zeros((4, 4, 4), dtype=np.float32),
                "ijk_kji_to_ras_fn": _kji_to_ras_points_identity,
                "components": {
                    "gating": _OverrideGating(),
                    "blob_extractor": _OverrideBlobExtractor(),
                },
            }
        )
        self.assertEqual(out["status"], "ok")
        self.assertIsInstance(out.get("contacts"), list)
        self.assertEqual(len(out.get("contacts") or []), 0)
        self.assertIn("contact_detection_not_implemented", out.get("warnings") or [])
        counts = out["diagnostics"]["counts"]
        for key in ("blobs_total", "seed_count_segment", "seed_count_bead_pair", "shanks_before_select", "shanks_final"):
            self.assertIn(key, counts)
        for key in (
            "seed_count_segment_raw",
            "seed_count_segment_after_nms",
            "seed_count_bead_raw",
            "seed_count_bead_after_nms",
            "bead_rescue_triggered",
            "seed_count_bead_rescue",
            "shanks_after_first_pass",
            "shanks_after_second_pass",
        ):
            self.assertIn(key, counts)
        self.assertIn("coverage_metrics", out["diagnostics"]["extras"])

    def test_blob_em_v2_compact_blob_produces_one_support_observation(self):
        extractor = _EMBlobExtractor()
        arr = np.zeros((32, 32, 32), dtype=np.float32)
        points = np.asarray(
            [[k, j, i] for k in range(12, 15) for j in range(12, 15) for i in range(12, 15)],
            dtype=float,
        )
        arr[points[:, 0].astype(int), points[:, 1].astype(int), points[:, 2].astype(int)] = 2400.0
        blob = extractor._blob_from_points(
            points,
            arr_kji=arr,
            depth_map_kji=np.ones(arr.shape, dtype=np.float32) * 10.0,
            ijk_kji_to_ras_fn=_kji_to_ras_points_identity,
            blob_id=1,
            meta={},
        )
        self.assertIsNotNone(blob)
        priors = extractor._electrode_priors_from_cfg({}, {})
        supports = extractor._support_observations_for_blob(
            blob=blob,
            points_kji=points,
            arr_kji=arr,
            depth_map_kji=np.ones(arr.shape, dtype=np.float32) * 10.0,
            ijk_kji_to_ras_fn=_kji_to_ras_points_identity,
            cfg={},
            priors=priors,
        )
        self.assertEqual(len(supports), 1)
        self.assertEqual(str((supports[0].get("meta") or {}).get("support_kind")), "blob_centroid")

    def test_blob_em_v2_elongated_blob_produces_multiple_axial_samples(self):
        extractor = _EMBlobExtractor()
        arr = np.zeros((48, 48, 48), dtype=np.float32)
        points = _make_bar_points(length=28, thickness=3, k0=14, j0=14, i0=6)
        arr[points[:, 0].astype(int), points[:, 1].astype(int), points[:, 2].astype(int)] = 2500.0
        blob = extractor._blob_from_points(
            points,
            arr_kji=arr,
            depth_map_kji=np.ones(arr.shape, dtype=np.float32) * 12.0,
            ijk_kji_to_ras_fn=_kji_to_ras_points_identity,
            blob_id=2,
            meta={},
        )
        self.assertIsNotNone(blob)
        cfg = {
            "axial_support_min_elongation": 2.0,
            "axial_support_min_length_mm": 6.0,
            "axial_support_spacing_mm": 3.0,
            "axial_support_window_mm": 2.0,
            "axial_support_min_window_voxels": 3,
            "axial_support_min_diameter_mm": 0.3,
            "axial_support_max_diameter_mm": 5.0,
        }
        priors = extractor._electrode_priors_from_cfg(cfg, {})
        supports = extractor._support_observations_for_blob(
            blob=blob,
            points_kji=points,
            arr_kji=arr,
            depth_map_kji=np.ones(arr.shape, dtype=np.float32) * 12.0,
            ijk_kji_to_ras_fn=_kji_to_ras_points_identity,
            cfg=cfg,
            priors=priors,
        )
        self.assertGreater(len(supports), 1)
        self.assertTrue(all(str((s.get("meta") or {}).get("support_kind")) == "axial_sample" for s in supports))

    def test_blob_em_v2_support_weight_conservation_across_axial_samples(self):
        extractor = _EMBlobExtractor()
        scorer = _EMBlobScorer()
        arr = np.zeros((48, 48, 48), dtype=np.float32)
        points = _make_bar_points(length=24, thickness=3, k0=16, j0=16, i0=8)
        arr[points[:, 0].astype(int), points[:, 1].astype(int), points[:, 2].astype(int)] = 2400.0
        blob = extractor._blob_from_points(
            points,
            arr_kji=arr,
            depth_map_kji=np.ones(arr.shape, dtype=np.float32) * 14.0,
            ijk_kji_to_ras_fn=_kji_to_ras_points_identity,
            blob_id=7,
            meta={},
        )
        self.assertIsNotNone(blob)
        cfg = {"axial_support_spacing_mm": 3.0, "axial_support_window_mm": 2.5, "axial_support_max_diameter_mm": 5.0}
        priors = extractor._electrode_priors_from_cfg(cfg, {})
        supports = extractor._support_observations_for_blob(
            blob=blob,
            points_kji=points,
            arr_kji=arr,
            depth_map_kji=np.ones(arr.shape, dtype=np.float32) * 14.0,
            ijk_kji_to_ras_fn=_kji_to_ras_points_identity,
            cfg=cfg,
            priors=priors,
        )
        records = [
            BlobRecord(
                blob_id=i + 1,
                centroid_ras=tuple(float(v) for v in s["centroid_ras"]),
                centroid_kji=tuple(float(v) for v in s["centroid_kji"]),
                voxel_count=int(s["voxel_count"]),
                peak_hu=float(s["hu_max"]),
                q95_hu=float(s["hu_q95"]),
                mean_hu=float(s["hu_mean"]),
                pca_axis_ras=tuple(float(v) for v in blob["pca_axis_ras"]),
                pca_evals=tuple(float(v) for v in blob["pca_evals"]),
                length_mm=float(blob["length_mm"]),
                diameter_mm=float(blob["diameter_mm"]),
                elongation=float(blob["elongation"]),
                depth_min_mm=float(s["depth_min"]),
                depth_mean_mm=float(s["depth_mean"]),
                depth_max_mm=float(s["depth_max"]),
                meta=dict(s.get("meta") or {}),
            )
            for i, s in enumerate(supports)
        ]
        scored = scorer.score({"config": {"min_metal_depth_mm": 5.0}}, records, {})
        total_weight = float(sum(float(b.scores.get("support_weight", 0.0)) for b in scored))
        base_weight = float(scored[0].scores.get("support_weight_base", 0.0))
        self.assertAlmostEqual(total_weight, base_weight, places=4)

    def test_blob_em_v2_support_samples_use_local_voxel_windows(self):
        extractor = _EMBlobExtractor()
        arr = np.zeros((48, 48, 48), dtype=np.float32)
        points = _make_wavy_bar_points(length=26, thickness=3, k0=20, j0=18, i0=6)
        arr[points[:, 0].astype(int), points[:, 1].astype(int), points[:, 2].astype(int)] = 2500.0
        blob = extractor._blob_from_points(
            points,
            arr_kji=arr,
            depth_map_kji=np.ones(arr.shape, dtype=np.float32) * 11.0,
            ijk_kji_to_ras_fn=_kji_to_ras_points_identity,
            blob_id=12,
            meta={},
        )
        self.assertIsNotNone(blob)
        cfg = {"axial_support_spacing_mm": 2.5, "axial_support_window_mm": 2.0, "axial_support_max_diameter_mm": 5.0}
        priors = extractor._electrode_priors_from_cfg(cfg, {})
        supports = extractor._support_observations_for_blob(
            blob=blob,
            points_kji=points,
            arr_kji=arr,
            depth_map_kji=np.ones(arr.shape, dtype=np.float32) * 11.0,
            ijk_kji_to_ras_fn=_kji_to_ras_points_identity,
            cfg=cfg,
            priors=priors,
        )
        self.assertGreater(len(supports), 2)
        j_values = [float(s["centroid_kji"][1]) for s in supports]
        self.assertGreater(float(np.max(j_values) - np.min(j_values)), 0.5)

    def test_blob_em_v2_ineligible_elongated_blob_falls_back_to_single_support(self):
        extractor = _EMBlobExtractor()
        arr = np.zeros((48, 48, 48), dtype=np.float32)
        points = _make_bar_points(length=22, thickness=5, k0=18, j0=18, i0=8)
        arr[points[:, 0].astype(int), points[:, 1].astype(int), points[:, 2].astype(int)] = 2400.0
        blob = extractor._blob_from_points(
            points,
            arr_kji=arr,
            depth_map_kji=np.ones(arr.shape, dtype=np.float32) * 10.0,
            ijk_kji_to_ras_fn=_kji_to_ras_points_identity,
            blob_id=22,
            meta={},
        )
        self.assertIsNotNone(blob)
        cfg = {"axial_support_min_length_mm": 4.0, "axial_support_max_diameter_mm": 0.6}
        priors = extractor._electrode_priors_from_cfg(cfg, {})
        supports = extractor._support_observations_for_blob(
            blob=blob,
            points_kji=points,
            arr_kji=arr,
            depth_map_kji=np.ones(arr.shape, dtype=np.float32) * 10.0,
            ijk_kji_to_ras_fn=_kji_to_ras_points_identity,
            cfg=cfg,
            priors=priors,
        )
        self.assertEqual(len(supports), 1)
        self.assertEqual(str((supports[0].get("meta") or {}).get("support_kind")), "blob_centroid")

    def test_blob_em_v2_support_sampling_diagnostics_counters_updated(self):
        reg = PipelineRegistry()
        register_builtin_pipelines(reg)
        pipe = reg.create_pipeline("blob_em_v2")
        out = pipe.run(
            {
                "run_id": "r-support-counters",
                "arr_kji": np.zeros((8, 8, 8), dtype=np.float32),
                "ijk_kji_to_ras_fn": _kji_to_ras_points_identity,
                "config": {"min_blob_voxels": 1},
                "components": {"gating": _MaskSourceGating(), "initializer": _NoopInitializer()},
            }
        )
        self.assertEqual(out["status"], "ok")
        counts = out["diagnostics"]["counts"]
        for key in (
            "blobs_total_cc",
            "blobs_eligible_for_axial_sampling",
            "axial_support_samples_generated",
            "support_points_total",
            "support_points_from_compact_blobs",
            "support_points_from_elongated_blobs",
        ):
            self.assertIn(key, counts)

    def test_blob_ransac_v1_adapter_runs_minimal_case(self):
        reg = PipelineRegistry()
        register_builtin_pipelines(reg)
        pipe = reg.create_pipeline("blob_ransac_v1")

        arr = np.full((24, 24, 24), -1000.0, dtype=np.float32)
        arr[6:18, 6:18, 6:18] = 40.0
        arr[12, 12, 10] = 2500.0

        ctx = {
            "run_id": "r2",
            "arr_kji": arr,
            "spacing_xyz": (1.0, 1.0, 1.0),
            # Regression coverage: numpy arrays must not be evaluated as bool.
            "center_ras": np.asarray([12.0, 12.0, 12.0], dtype=float),
            "ijk_kji_to_ras_fn": _kji_to_ras_points_identity,
            "ras_to_ijk_fn": _ras_to_ijk_identity,
            "config": {
                "threshold": 1800.0,
                "max_lines": 5,
                "min_inliers": 2,
                "candidate_mode": "blob_centroid",
                "head_mask_method": "outside_air",
                "min_metal_depth_mm": 0.0,
            },
        }

        out = pipe.run(ctx)
        self.assertEqual(out["pipeline_id"], "blob_ransac_v1")
        self.assertIn(out["status"], ("ok", "error"))
        self.assertIn("diagnostics", out)

    def test_shank_axis_v1_runs_minimal_case(self):
        reg = PipelineRegistry()
        register_builtin_pipelines(reg)
        pipe = reg.create_pipeline("shank_axis_v1")

        arr = np.full((24, 24, 24), -1000.0, dtype=np.float32)
        arr[4:20, 4:20, 4:20] = 40.0
        for t in range(6, 18):
            arr[t, 12, 12] = 2500.0

        ctx = {
            "run_id": "r-shank-axis",
            "arr_kji": arr,
            "spacing_xyz": (1.0, 1.0, 1.0),
            "center_ras": np.asarray([12.0, 12.0, 12.0], dtype=float),
            "ijk_kji_to_ras_fn": _kji_to_ras_points_identity,
            "ras_to_ijk_fn": _ras_to_ijk_identity,
            "config": {
                "threshold": 1800.0,
                "use_head_mask": False,
                "build_head_mask": False,
                "max_lines": 4,
                "proposal_max_lines": 8,
                "min_inliers": 2,
                "proposal_min_inliers": 2,
                "min_length_mm": 4.0,
                "start_zone_window_mm": 12.0,
            },
        }

        out = pipe.run(ctx)
        self.assertEqual(out["pipeline_id"], "shank_axis_v1")
        self.assertIn(out["status"], ("ok", "error"))
        self.assertIn("diagnostics", out)
        self.assertIn("trajectories", out)

    def test_de_novo_hypothesis_select_v1_detects_clean_bar(self):
        reg = PipelineRegistry()
        register_builtin_pipelines(reg)
        pipe = reg.create_pipeline("de_novo_hypothesis_select_v1")

        arr = np.full((32, 32, 48), -1000.0, dtype=np.float32)
        bar = _make_bar_points(length=28, thickness=3, k0=15, j0=15, i0=8).astype(int)
        arr[bar[:, 0], bar[:, 1], bar[:, 2]] = 850.0

        out = pipe.run(
            {
                "run_id": "r-denovo-bar",
                "arr_kji": arr,
                "spacing_xyz": (1.0, 1.0, 1.0),
                "center_ras": np.asarray([16.0, 16.0, 24.0], dtype=float),
                "ijk_kji_to_ras_fn": _kji_to_ras_points_identity,
                "ras_to_ijk_fn": _ras_to_ijk_identity,
                "config": {
                    "use_head_mask": False,
                    "build_head_mask": False,
                    "min_metal_depth_mm": 0.0,
                    "target_candidate_points": 1,
                    "ablation_stage": "global_selection",
                },
            }
        )
        self.assertEqual(out["status"], "ok")
        self.assertGreaterEqual(len(list(out.get("trajectories") or [])), 1)



if __name__ == "__main__":
    unittest.main()
