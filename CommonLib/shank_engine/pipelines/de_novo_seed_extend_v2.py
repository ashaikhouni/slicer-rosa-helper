"""De novo shank detection by local linelet seeds plus straight proposal extension.

This benchmark-only pipeline separates proposal generation into two support pools:
- a selective seed pool for reliable short local axes
- a permissive extension pool for growing those seeds into long straight proposals

Selection and late guided refinement are intentionally reused from the de novo
hypothesis/select pipeline so proposal recall can be improved independently.
"""

from __future__ import annotations

import math
import time
from typing import Any, Iterable

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:  # pragma: no cover
    sitk = None

from shank_core.blob_candidates import extract_blob_candidates
from shank_core.masking import build_not_air_lcc_gate_kji, compute_head_distance_map_kji, largest_component_binary

from ..artifacts import add_artifact, write_standard_artifacts
from ..contracts import DetectionContext, DetectionResult
from .de_novo_hypothesis_select_v1 import (
    DeNovoHypothesisSelectV1Pipeline,
    _Hypothesis,
    _SelectedHypothesis,
    _SupportToken,
)


class DeNovoSeedExtendV2Pipeline(DeNovoHypothesisSelectV1Pipeline):
    pipeline_id = "de_novo_seed_extend_v2"
    display_name = "De Novo Seed+Extend v2"
    pipeline_version = "2.0.0"
    scaffold = False
    expose_in_ui = False

    def _defaults(self, cfg: dict[str, Any]) -> dict[str, Any]:
        out = super()._defaults(cfg)
        out["seed_threshold_schedule_hu"] = [900.0, 800.0, 700.0]
        out["seed_core_shrink_schedule_mm"] = [15.0, 12.0, 10.0]
        out["seed_target_candidate_points"] = 800
        out["seed_metal_grow_vox"] = 1
        out["seed_hull_threshold_hu"] = float(out.get("head_mask_threshold_hu", -500.0))
        out["seed_hull_erode_vox"] = 1
        out["seed_hull_dilate_vox"] = 1
        out["seed_hull_clip_hu"] = 1200.0
        out["seed_hull_gaussian_sigma_mm"] = 4.0
        out["seed_hull_open_vox"] = 7
        out["seed_hull_close_vox"] = 0
        out["seed_hull_extend_cap_mm"] = 40.0
        out["seed_hull_extend_step_mm"] = 0.5
        out["seed_ransac_use_bridge"] = True
        out["seed_ransac_sample_budget"] = 6000
        out["seed_ransac_iterations"] = 1600
        out["seed_ransac_random_seed"] = 0
        out["seed_ransac_min_pair_separation_mm"] = 8.0
        out["seed_ransac_inlier_radius_mm"] = 2.0
        out["seed_ransac_min_inliers"] = 14
        out["seed_ransac_min_span_mm"] = 10.0
        out["seed_ransac_max_seeds"] = 220
        out["seed_ransac_max_width_mm"] = 2.2
        out["extension_threshold_schedule_hu"] = [600.0, 550.0, 500.0]
        out["extension_target_candidate_points"] = int(out.get("target_candidate_points", 300000))
        out["extension_min_metal_depth_mm"] = 5.0
        out["extension_max_metal_depth_mm"] = float(out.get("max_metal_depth_mm", 220.0))
        out["extension_dilate_vox"] = 1
        out["seed_sample_budget"] = 7000
        out["seed_random_seed"] = 0
        out["linelet_neighbor_radius_mm"] = 3.0
        out["linelet_min_neighbors"] = 6
        out["linelet_max_width_mm"] = 1.8
        out["linelet_min_linearity"] = 3.0
        out["linelet_cluster_radius_mm"] = 4.0
        out["linelet_cluster_angle_deg"] = 18.0
        out["linelet_min_cluster_samples"] = 2
        out["linelet_max_count"] = 220
        out["linelet_min_confidence"] = 0.5
        out["seed_blob_axis_augmentation"] = True
        out["seed_blob_min_length_mm"] = 6.0
        out["seed_blob_min_elongation"] = 4.5
        out["extension_corridor_radius_mm"] = 2.2
        out["extension_bin_size_mm"] = 1.0
        out["extension_min_bin_points"] = 2
        out["extension_normal_gap_mm"] = 6.0
        out["extension_large_gap_mm"] = 12.0
        out["extension_max_large_gaps"] = 1
        out["extension_refit_iters"] = 3
        out["extension_max_width_mm"] = 2.8
        out["extension_max_radial_rms_mm"] = 1.5
        out["extension_min_points"] = 10
        out["extension_assignment_radius_mm"] = 2.5
        out["extension_assignment_min_affinity"] = 0.05
        out["min_proposal_length_mm"] = 12.0
        out["selection_redundancy_weight"] = 2.5
        out["selection_shared_token_weight"] = 0.6
        out["selection_complexity_cost"] = 2.0
        out["selection_min_gain"] = 0.25
        out.setdefault("ablation_stage", "late_model_priors")
        return out

    def run(self, ctx: DetectionContext) -> DetectionResult:
        t_start = time.perf_counter()
        result = self.make_result(ctx)
        diagnostics = self.diagnostics(result)
        cfg = self._defaults(self._config(ctx))
        stage = str(cfg.get("ablation_stage") or "late_model_priors").strip().lower()
        valid_stages = {"support_pool", "hypothesis_generation", "global_selection", "late_model_priors"}
        if stage not in valid_stages:
            stage = "late_model_priors"
        self._reset_run_state()
        try:
            if "arr_kji" not in ctx or "ijk_kji_to_ras_fn" not in ctx:
                result["warnings"].append("de_novo_seed_extend_v2 missing volume context; returning empty result")
                diagnostics.note("de_novo_seed_extend_v2 requires arr_kji and ijk_kji_to_ras_fn")
                return self.finalize(result, diagnostics, t_start)

            support = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="support_pool",
                fn=lambda: self._build_support_bundle(ctx, cfg),
            )
            self._record_support_diagnostics(diagnostics, support)
            self._attach_support_artifact(result, ctx, support)
            seed_support = dict(support.get("seed_pool") or {})
            if stage == "support_pool" or not list(seed_support.get("tokens") or []):
                result["trajectories"] = []
                result["contacts"] = []
                return self.finalize(result, diagnostics, t_start)

            proposals_before_nms = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="hypothesis_generation",
                fn=lambda: self._generate_hypotheses(ctx, support, cfg),
            )
            diagnostics.set_count("linelet_count", int(len(self._last_linelets)))
            diagnostics.set_count("seed_count", int(len(self._last_seeds)))
            diagnostics.set_count("proposal_count_before_nms", int(len(proposals_before_nms)))
            diagnostics.set_extra("linelets", list(self._last_linelets))
            diagnostics.set_extra("seeds", list(self._last_seeds))
            diagnostics.set_extra("proposals_before_nms", [self._hypothesis_json(h) for h in proposals_before_nms])
            hypotheses = self._nms_hypotheses(proposals_before_nms, cfg)
            diagnostics.set_count("hypothesis_count_generated", int(len(hypotheses)))
            diagnostics.set_extra("hypotheses", [self._hypothesis_json(h) for h in hypotheses])
            if stage == "hypothesis_generation":
                result["trajectories"] = [self._trajectory_from_hypothesis(h) for h in hypotheses]
                result["contacts"] = []
                result["warnings"].append("contact_detection_not_implemented")
                return self.finalize(result, diagnostics, t_start)

            selected, conflicts = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="global_selection",
                fn=lambda: self._select_hypotheses(hypotheses, support, cfg),
            )
            diagnostics.set_count("selected_count", int(len(selected)))
            diagnostics.set_count("conflict_edge_count", int(len(conflicts)))
            diagnostics.set_extra("conflicts", conflicts)
            diagnostics.set_extra("selected_proposals", [self._hypothesis_json(h) for h in selected])
            if stage == "global_selection":
                result["trajectories"] = [self._trajectory_from_hypothesis(h) for h in selected]
                result["contacts"] = []
                result["warnings"].append("contact_detection_not_implemented")
                return self.finalize(result, diagnostics, t_start)

            refinement_support = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="refinement_support_pool",
                fn=lambda: self._build_support_pool(ctx, cfg, pool_name="refinement"),
            )
            self._record_refinement_support_diagnostics(diagnostics, refinement_support)
            final_selected = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="late_model_priors",
                fn=lambda: self._refine_and_rerank_selected(ctx, seed_support, refinement_support, selected, cfg),
            )
            result["trajectories"] = [self._trajectory_from_selected(h) for h in final_selected]
            result["contacts"] = []
            result["warnings"].append("contact_detection_not_implemented")
            diagnostics.set_extra("selected_hypotheses", [self._selected_json(h) for h in final_selected])
        except Exception as exc:  # pragma: no cover
            self.fail(ctx=ctx, result=result, diagnostics=diagnostics, stage=str(getattr(exc, "stage", "pipeline")), exc=exc)
        return self.finalize(result, diagnostics, t_start)

    def _reset_run_state(self) -> None:
        self._last_linelets: list[dict[str, Any]] = []
        self._last_seeds: list[dict[str, Any]] = []

    def _build_support_bundle(self, ctx: DetectionContext, cfg: dict[str, Any]) -> dict[str, Any]:
        seed_support = self._build_seed_pool(ctx, cfg)
        seed_support["pool_name"] = "seed_pool"
        extension_support = self._build_extension_pool(ctx, cfg)
        return {
            "seed_pool": seed_support,
            "extension_pool": extension_support,
        }

    def _build_seed_pool(self, ctx: DetectionContext, cfg: dict[str, Any]) -> dict[str, Any]:
        arr_kji = np.asarray(ctx["arr_kji"], dtype=float)
        spacing_xyz = tuple(ctx.get("spacing_xyz") or (1.0, 1.0, 1.0))
        hull_mask_kji, head_distance_map_kji, hull_stats = self._build_smoothed_hull_mask(
            arr_kji=arr_kji,
            spacing_xyz=spacing_xyz,
            threshold_hu=float(cfg.get("seed_hull_threshold_hu", cfg.get("head_mask_threshold_hu", -500.0))),
            clip_hu=float(cfg.get("seed_hull_clip_hu", 1200.0)),
            sigma_mm=float(cfg.get("seed_hull_gaussian_sigma_mm", 4.0)),
            open_vox=int(cfg.get("seed_hull_open_vox", 7)),
            close_vox=int(cfg.get("seed_hull_close_vox", 0)),
        )
        thresholds = [float(v) for v in list(cfg.get("seed_threshold_schedule_hu") or [900.0])]
        shrink_schedule = [float(v) for v in list(cfg.get("seed_core_shrink_schedule_mm") or [15.0, 12.0, 10.0])]
        target_points = max(0, int(cfg.get("seed_target_candidate_points", 800)))
        grow_vox = int(cfg.get("seed_metal_grow_vox", 1))

        chosen_threshold = thresholds[-1]
        chosen_shrink = shrink_schedule[-1]
        chosen_candidate_mask = np.zeros(arr_kji.shape, dtype=bool)
        chosen_metal_mask = np.zeros(arr_kji.shape, dtype=bool)
        chosen_bridge_mask = np.zeros(arr_kji.shape, dtype=bool)
        chosen_deep_core_mask = np.zeros(arr_kji.shape, dtype=bool)
        preview_stats: list[dict[str, Any]] = []
        best_count = -1

        for threshold in thresholds:
            metal_mask_kji = np.asarray(arr_kji >= float(threshold), dtype=bool)
            metal_grown_kji = self._dilate_mask(metal_mask_kji, grow_vox)
            for shrink_mm in shrink_schedule:
                deep_core_mask_kji = np.logical_and(hull_mask_kji, np.asarray(head_distance_map_kji, dtype=float) >= float(shrink_mm))
                candidate_mask_kji = np.logical_and(metal_mask_kji, deep_core_mask_kji)
                bridge_mask_kji = np.logical_and(metal_grown_kji, deep_core_mask_kji)
                count = int(np.count_nonzero(candidate_mask_kji))
                preview_stats.append(
                    {
                        "threshold_hu": float(threshold),
                        "core_shrink_mm": float(shrink_mm),
                        "candidate_count": int(count),
                        "bridge_count": int(np.count_nonzero(bridge_mask_kji)),
                    }
                )
                if count > best_count:
                    best_count = count
                    chosen_threshold = float(threshold)
                    chosen_shrink = float(shrink_mm)
                    chosen_candidate_mask = candidate_mask_kji
                    chosen_metal_mask = metal_grown_kji
                    chosen_bridge_mask = bridge_mask_kji
                    chosen_deep_core_mask = deep_core_mask_kji
                if target_points <= 0 or count >= target_points:
                    break
            else:
                continue
            break

        if not np.any(chosen_candidate_mask) and thresholds and shrink_schedule:
            metal_mask_kji = np.asarray(arr_kji >= float(chosen_threshold), dtype=bool)
            chosen_metal_mask = metal_mask_kji
            metal_grown_kji = self._dilate_mask(metal_mask_kji, grow_vox)
            chosen_deep_core_mask = np.logical_and(hull_mask_kji, np.asarray(head_distance_map_kji, dtype=float) >= float(chosen_shrink))
            chosen_candidate_mask = np.logical_and(chosen_metal_mask, chosen_deep_core_mask)
            chosen_bridge_mask = np.logical_and(metal_grown_kji, chosen_deep_core_mask)

        ijk_kji = np.argwhere(chosen_candidate_mask)
        points_ras = self._ijk_kji_to_ras(ctx, ijk_kji.astype(float)) if ijk_kji.size else np.zeros((0, 3), dtype=float)
        points_lps = self._ras_to_lps(points_ras)
        bridge_ijk_kji = np.argwhere(chosen_bridge_mask)
        bridge_points_ras = self._ijk_kji_to_ras(ctx, bridge_ijk_kji.astype(float)) if bridge_ijk_kji.size else np.zeros((0, 3), dtype=float)
        bridge_points_lps = self._ras_to_lps(bridge_points_ras)
        candidate_depth_mm = None
        if ijk_kji.size:
            candidate_depth_mm = np.asarray(
                head_distance_map_kji[ijk_kji[:, 0], ijk_kji[:, 1], ijk_kji[:, 2]],
                dtype=float,
            ).reshape(-1)
        bridge_depth_mm = None
        if bridge_ijk_kji.size:
            bridge_depth_mm = np.asarray(
                head_distance_map_kji[bridge_ijk_kji[:, 0], bridge_ijk_kji[:, 1], bridge_ijk_kji[:, 2]],
                dtype=float,
            ).reshape(-1)
        blob_source_mask = chosen_bridge_mask if np.any(chosen_bridge_mask) else chosen_candidate_mask
        blob_result = extract_blob_candidates(
            blob_source_mask,
            arr_kji=arr_kji,
            depth_map_kji=head_distance_map_kji,
            ijk_kji_to_ras_fn=ctx["ijk_kji_to_ras_fn"],
            fully_connected=True,
        )
        blob_samples = self._blob_samples(blob_result, ctx, arr_kji)
        tokens, token_summary = self._tokens_from_blobs(blob_result, blob_samples, cfg)
        preview = {
            "gating_mask_kji": hull_mask_kji.astype(np.uint8),
            "head_mask_kji": hull_mask_kji.astype(np.uint8),
            "head_distance_map_kji": head_distance_map_kji,
            "metal_mask_kji": chosen_metal_mask.astype(np.uint8),
            "metal_in_head_mask_kji": np.logical_and(chosen_metal_mask, hull_mask_kji).astype(np.uint8),
            "metal_depth_pass_mask_kji": chosen_candidate_mask.astype(np.uint8),
            "deep_core_mask_kji": chosen_deep_core_mask.astype(np.uint8),
            "seed_bridge_mask_kji": chosen_bridge_mask.astype(np.uint8),
            "head_mask_kept_count": int(np.count_nonzero(hull_mask_kji)),
            "metal_in_head_count": int(np.count_nonzero(np.logical_and(chosen_metal_mask, hull_mask_kji))),
            "depth_kept_count": int(np.count_nonzero(chosen_candidate_mask)),
        }
        return {
            "pool_name": "seed_pool",
            "support_mode": "deep_core_seed",
            "threshold_hu": float(chosen_threshold),
            "preview_stats": preview_stats,
            "preview": preview,
            "candidate_mask_kji": chosen_candidate_mask,
            "candidate_points_after_mask": int(np.count_nonzero(chosen_candidate_mask)),
            "candidate_points_ras": points_ras,
            "candidate_points_lps": points_lps,
            "candidate_depth_mm": candidate_depth_mm,
            "bridge_candidate_points_ras": bridge_points_ras,
            "bridge_candidate_points_lps": bridge_points_lps,
            "bridge_candidate_depth_mm": bridge_depth_mm,
            "blob_result": blob_result,
            "blob_samples": blob_samples,
            "tokens": tokens,
            "token_summary": token_summary,
            "ras_to_ijk_fn": ctx.get("ras_to_ijk_fn"),
            "deep_core_shrink_mm": float(chosen_shrink),
            "hull_mask_kji": hull_mask_kji,
            "hull_distance_map_kji": head_distance_map_kji,
            "hull_stats": hull_stats,
            "bridge_candidate_points_after_mask": int(np.count_nonzero(chosen_bridge_mask)),
        }

    def _build_smoothed_hull_mask(
        self,
        *,
        arr_kji: np.ndarray,
        spacing_xyz: tuple[float, float, float],
        threshold_hu: float,
        clip_hu: float,
        sigma_mm: float,
        open_vox: int,
        close_vox: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        arr = np.asarray(arr_kji, dtype=np.float32)
        if sitk is None:
            gate = build_not_air_lcc_gate_kji(
                arr_kji=arr,
                spacing_xyz=spacing_xyz,
                air_threshold_hu=float(threshold_hu),
                erode_radius_vox=1,
                dilate_radius_vox=1,
                gate_margin_mm=0.0,
            )
            hull_mask_kji = np.asarray(gate.get("head_gate_mask_kji"), dtype=bool)
            dist_map = np.asarray(gate.get("head_distance_map_kji"), dtype=np.float32)
            return hull_mask_kji, dist_map, {"fallback": "not_air_lcc"}

        arr_clip = np.asarray(arr, dtype=np.float32)
        if math.isfinite(float(clip_hu)):
            arr_clip = np.minimum(arr_clip, float(clip_hu))
        img = sitk.GetImageFromArray(arr_clip.astype(np.float32))
        img.SetSpacing((float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])))
        if float(sigma_mm) > 0.0:
            img = sitk.SmoothingRecursiveGaussian(img, float(sigma_mm))
        binary = sitk.BinaryThreshold(
            img,
            lowerThreshold=float(threshold_hu),
            upperThreshold=float(np.max(arr_clip) + 1.0),
            insideValue=1,
            outsideValue=0,
        )
        opened = binary
        if int(open_vox) > 0:
            opened = sitk.BinaryMorphologicalOpening(binary, [int(open_vox)] * 3, sitk.sitkBall)
        hull_img = largest_component_binary(opened)
        if hull_img is None:
            hull_img = largest_component_binary(binary)
        if hull_img is None:
            empty = np.zeros(arr.shape, dtype=bool)
            return empty, np.zeros(arr.shape, dtype=np.float32), {"fallback": "empty"}
        if int(close_vox) > 0:
            hull_img = sitk.BinaryMorphologicalClosing(hull_img, [int(close_vox)] * 3, sitk.sitkBall)
        hull_mask_kji = sitk.GetArrayFromImage(hull_img).astype(bool)
        dist_map = compute_head_distance_map_kji(hull_mask_kji, spacing_xyz=spacing_xyz)
        return hull_mask_kji, np.asarray(dist_map, dtype=np.float32), {
            "clip_hu": float(clip_hu),
            "sigma_mm": float(sigma_mm),
            "open_vox": int(open_vox),
            "close_vox": int(close_vox),
        }

    def _choose_preview(self, ctx: DetectionContext, cfg: dict[str, Any], pool_name: str) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
        arr_kji = np.asarray(ctx["arr_kji"], dtype=float)
        spacing_xyz = tuple(ctx.get("spacing_xyz") or (1.0, 1.0, 1.0))
        pool_cfg = self._support_pool_cfg(cfg, pool_name)
        thresholds = [float(v) for v in list(pool_cfg.get("threshold_schedule_hu") or [600.0, 550.0, 500.0])]
        chosen_threshold = thresholds[-1]
        chosen_preview: dict[str, Any] | None = None
        preview_stats: list[dict[str, Any]] = []
        target_points = max(0, int(pool_cfg.get("target_candidate_points", 300000)))
        use_masking = bool(cfg.get("use_head_mask", True)) or bool(cfg.get("build_head_mask", True))
        for threshold in thresholds:
            preview = (
                self._simple_preview_masks(arr_kji=arr_kji, threshold=float(threshold))
                if not use_masking
                else __import__("shank_core.masking", fromlist=["build_preview_masks"]).build_preview_masks(
                    arr_kji=arr_kji,
                    spacing_xyz=spacing_xyz,
                    threshold=float(threshold),
                    use_head_mask=bool(cfg.get("use_head_mask", True)),
                    build_head_mask=bool(cfg.get("build_head_mask", True)),
                    head_mask_threshold_hu=float(cfg.get("head_mask_threshold_hu", -500.0)),
                    head_mask_aggressive_cleanup=bool(cfg.get("head_mask_aggressive_cleanup", True)),
                    head_mask_close_mm=float(cfg.get("head_mask_close_mm", 2.0)),
                    head_mask_method=str(cfg.get("head_mask_method", "outside_air")),
                    head_mask_metal_dilate_mm=float(cfg.get("head_mask_metal_dilate_mm", 1.0)),
                    head_gate_erode_vox=int(cfg.get("head_gate_erode_vox", 1)),
                    head_gate_dilate_vox=int(cfg.get("head_gate_dilate_vox", 1)),
                    head_gate_margin_mm=float(cfg.get("head_gate_margin_mm", 0.0)),
                    min_metal_depth_mm=float(pool_cfg.get("min_metal_depth_mm", 5.0)),
                    max_metal_depth_mm=float(pool_cfg.get("max_metal_depth_mm", 220.0)),
                    include_debug_masks=False,
                )
            )
            count = int(preview.get("depth_kept_count") or 0)
            preview_stats.append({"threshold_hu": float(threshold), "depth_kept_count": count})
            if target_points <= 0 or count >= target_points:
                chosen_threshold = float(threshold)
                chosen_preview = preview
                break
        if chosen_preview is None:
            chosen_threshold = thresholds[-1]
            chosen_preview = (
                self._simple_preview_masks(arr_kji=arr_kji, threshold=float(chosen_threshold))
                if not use_masking
                else __import__("shank_core.masking", fromlist=["build_preview_masks"]).build_preview_masks(
                    arr_kji=arr_kji,
                    spacing_xyz=spacing_xyz,
                    threshold=float(chosen_threshold),
                    use_head_mask=bool(cfg.get("use_head_mask", True)),
                    build_head_mask=bool(cfg.get("build_head_mask", True)),
                    head_mask_threshold_hu=float(cfg.get("head_mask_threshold_hu", -500.0)),
                    head_mask_aggressive_cleanup=bool(cfg.get("head_mask_aggressive_cleanup", True)),
                    head_mask_close_mm=float(cfg.get("head_mask_close_mm", 2.0)),
                    head_mask_method=str(cfg.get("head_mask_method", "outside_air")),
                    head_mask_metal_dilate_mm=float(cfg.get("head_mask_metal_dilate_mm", 1.0)),
                    head_gate_erode_vox=int(cfg.get("head_gate_erode_vox", 1)),
                    head_gate_dilate_vox=int(cfg.get("head_gate_dilate_vox", 1)),
                    head_gate_margin_mm=float(cfg.get("head_gate_margin_mm", 0.0)),
                    min_metal_depth_mm=float(pool_cfg.get("min_metal_depth_mm", 5.0)),
                    max_metal_depth_mm=float(pool_cfg.get("max_metal_depth_mm", 220.0)),
                    include_debug_masks=False,
                )
            )
        return float(chosen_threshold), chosen_preview, preview_stats

    def _build_extension_pool(self, ctx: DetectionContext, cfg: dict[str, Any]) -> dict[str, Any]:
        chosen_threshold, chosen_preview, preview_stats = self._choose_preview(ctx, cfg, pool_name="extension")
        gate_mask = np.asarray(chosen_preview.get("metal_depth_pass_mask_kji"), dtype=bool)
        gating_mask = chosen_preview.get("gating_mask_kji")
        if gating_mask is None:
            gating_mask = chosen_preview.get("head_mask_kji")
        gating_mask = np.asarray(gating_mask, dtype=bool)
        dilated_mask = self._dilate_mask(gate_mask, int(cfg.get("extension_dilate_vox", 1)))
        candidate_mask_kji = np.logical_and(dilated_mask, gating_mask)
        ijk_kji = np.argwhere(candidate_mask_kji)
        points_ras = self._ijk_kji_to_ras(ctx, ijk_kji.astype(float)) if ijk_kji.size else np.zeros((0, 3), dtype=float)
        points_lps = self._ras_to_lps(points_ras)
        depth_map_kji = chosen_preview.get("head_distance_map_kji")
        candidate_depth_mm = None
        if depth_map_kji is not None and ijk_kji.size:
            candidate_depth_mm = np.asarray(depth_map_kji[ijk_kji[:, 0], ijk_kji[:, 1], ijk_kji[:, 2]], dtype=float).reshape(-1)
        return {
            "pool_name": "extension_pool",
            "support_mode": "depth_mask_dilated",
            "threshold_hu": float(chosen_threshold),
            "preview_stats": preview_stats,
            "preview": chosen_preview,
            "candidate_mask_kji": candidate_mask_kji,
            "candidate_points_after_mask": int(np.count_nonzero(candidate_mask_kji)),
            "candidate_points_ras": points_ras,
            "candidate_points_lps": points_lps,
            "candidate_depth_mm": candidate_depth_mm,
            "ras_to_ijk_fn": ctx.get("ras_to_ijk_fn"),
            "extension_point_ids": np.arange(points_ras.shape[0], dtype=np.int32),
        }

    @staticmethod
    def _dilate_mask(mask_kji: np.ndarray, grow_vox: int) -> np.ndarray:
        mask = np.asarray(mask_kji, dtype=bool)
        if int(grow_vox) <= 0 or int(np.count_nonzero(mask)) <= 0:
            return mask
        if sitk is None:
            out = mask.copy()
            idx = np.argwhere(mask)
            for k, j, i in idx.tolist():
                k0 = max(0, int(k) - int(grow_vox))
                k1 = min(mask.shape[0], int(k) + int(grow_vox) + 1)
                j0 = max(0, int(j) - int(grow_vox))
                j1 = min(mask.shape[1], int(j) + int(grow_vox) + 1)
                i0 = max(0, int(i) - int(grow_vox))
                i1 = min(mask.shape[2], int(i) + int(grow_vox) + 1)
                out[k0:k1, j0:j1, i0:i1] = True
            return out
        img = sitk.GetImageFromArray(mask.astype(np.uint8))
        out = sitk.BinaryDilate(img, [int(grow_vox)] * 3, sitk.sitkBall)
        return sitk.GetArrayFromImage(out).astype(bool)

    def _generate_hypotheses(self, ctx: DetectionContext, support_bundle: dict[str, Any], cfg: dict[str, Any]) -> list[_Hypothesis]:
        seed_support = dict(support_bundle.get("seed_pool") or {})
        extension_support = dict(support_bundle.get("extension_pool") or {})
        seed_points = np.asarray(seed_support.get("candidate_points_ras"), dtype=float).reshape(-1, 3)
        if seed_points.size == 0:
            self._last_linelets = []
            self._last_seeds = []
            return []
        seeds = self._build_ransac_seeds(seed_support, cfg)
        self._last_linelets = []
        self._last_seeds = list(seeds)
        proposals: list[_Hypothesis] = []
        next_hyp_id = 1
        for seed in seeds:
            hyp = self._proposal_from_seed(next_hyp_id, seed, seed_support, extension_support, cfg)
            if hyp is None:
                continue
            proposals.append(hyp)
            next_hyp_id += 1
        return sorted(proposals, key=lambda h: float(h.local_score), reverse=True)

    def _build_ransac_seeds(self, seed_support: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
        use_bridge = bool(cfg.get("seed_ransac_use_bridge", True))
        points = np.asarray(
            seed_support.get("bridge_candidate_points_ras") if use_bridge else seed_support.get("candidate_points_ras"),
            dtype=float,
        ).reshape(-1, 3)
        if points.shape[0] < 2:
            return self._make_seeds_from_linelets_and_blobs([], seed_support, cfg)
        rng = np.random.default_rng(int(cfg.get("seed_ransac_random_seed", 0)))
        budget = int(cfg.get("seed_ransac_sample_budget", 6000))
        if points.shape[0] > budget > 0:
            sample_idx = np.sort(rng.choice(points.shape[0], size=budget, replace=False))
            sample_points = points[sample_idx]
        else:
            sample_points = points
        if sample_points.shape[0] < 2:
            return self._make_seeds_from_linelets_and_blobs([], seed_support, cfg)
        min_sep = float(cfg.get("seed_ransac_min_pair_separation_mm", 8.0))
        inlier_radius = float(cfg.get("seed_ransac_inlier_radius_mm", 2.0))
        min_inliers = int(cfg.get("seed_ransac_min_inliers", 14))
        min_span = float(cfg.get("seed_ransac_min_span_mm", 10.0))
        max_width = float(cfg.get("seed_ransac_max_width_mm", 2.2))
        max_iters = int(cfg.get("seed_ransac_iterations", 1600))
        seeds: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()

        def add_seed(point: np.ndarray, direction: np.ndarray, score: float, blob_ids: Iterable[int] = ()) -> None:
            direction = self._canonical_direction(direction)
            key = self._seed_key(point, direction)
            if key in seen:
                return
            seen.add(key)
            seeds.append(
                {
                    "seed_kind": "deep_ransac",
                    "point_ras": tuple(float(v) for v in point),
                    "direction_ras": tuple(float(v) for v in direction),
                    "seed_score": float(score),
                    "seed_blob_ids": tuple(sorted(int(v) for v in blob_ids if int(v) > 0)),
                }
            )

        for _ in range(max(0, max_iters)):
            pair = rng.choice(sample_points.shape[0], size=2, replace=False)
            p0 = sample_points[int(pair[0])]
            p1 = sample_points[int(pair[1])]
            seg = p1 - p0
            seg_len = float(np.linalg.norm(seg))
            if seg_len < min_sep:
                continue
            axis = self._canonical_direction(seg)
            center = 0.5 * (p0 + p1)
            d = self._line_distances(sample_points, center, axis)
            inlier_idx = np.where(d <= inlier_radius)[0]
            if inlier_idx.size < min_inliers:
                continue
            inlier_pts = sample_points[inlier_idx]
            try:
                center_fit, axis_fit = self._weighted_pca_line(inlier_pts, np.ones((inlier_pts.shape[0],), dtype=float))
            except Exception:
                continue
            axis_fit = self._canonical_direction(axis_fit)
            if self._angle_deg(axis, axis_fit) > 20.0:
                continue
            d_fit = self._line_distances(sample_points, center_fit, axis_fit)
            fit_idx = np.where(d_fit <= inlier_radius)[0]
            if fit_idx.size < min_inliers:
                continue
            fit_pts = sample_points[fit_idx]
            proj = (fit_pts - center_fit.reshape(1, 3)) @ axis_fit.reshape(3)
            span = float(np.ptp(proj)) if proj.size else 0.0
            if span < min_span:
                continue
            width = float(np.sqrt(np.mean(self._line_distances(fit_pts, center_fit, axis_fit) ** 2)))
            if width > max_width:
                continue
            score = float(fit_idx.size) + 0.35 * float(span) - 1.5 * float(width)
            add_seed(center_fit, axis_fit, score)

        seeds = self._augment_ransac_seeds_with_blobs(seeds, seed_support, cfg)
        seeds = sorted(seeds, key=lambda item: float(item.get("seed_score", 0.0)), reverse=True)
        return seeds[: int(cfg.get("seed_ransac_max_seeds", 220))]

    def _augment_ransac_seeds_with_blobs(self, seeds: list[dict[str, Any]], seed_support: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
        seen = {self._seed_key(np.asarray(seed["point_ras"], dtype=float), np.asarray(seed["direction_ras"], dtype=float)) for seed in seeds}
        out = list(seeds)
        if not bool(cfg.get("seed_blob_axis_augmentation", True)):
            return out
        for blob_id, sample in dict(seed_support.get("blob_samples") or {}).items():
            blob = dict(sample.get("blob") or {})
            length_mm = float(blob.get("length_mm") or 0.0)
            elongation = float(blob.get("elongation") or 1.0)
            if length_mm < float(cfg.get("seed_blob_min_length_mm", 6.0)):
                continue
            if elongation < float(cfg.get("seed_blob_min_elongation", 4.5)):
                continue
            centroid = np.asarray(blob.get("centroid_ras") if blob.get("centroid_ras") is not None else np.mean(np.asarray(sample.get("points_ras"), dtype=float), axis=0), dtype=float).reshape(3)
            axis = self._canonical_direction(np.asarray(blob.get("pca_axis_ras") if blob.get("pca_axis_ras") is not None else [0.0, 0.0, 1.0], dtype=float))
            key = self._seed_key(centroid, axis)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "seed_kind": "elongated_blob",
                    "point_ras": tuple(float(v) for v in centroid),
                    "direction_ras": tuple(float(v) for v in axis),
                    "seed_score": float(np.sqrt(max(1.0, float(sample.get("voxel_count") or 1)))) + 0.1 * float(length_mm),
                    "seed_blob_ids": (int(blob_id),),
                }
            )
        return out

    def _build_linelets(self, seed_support: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
        points = np.asarray(seed_support.get("candidate_points_ras"), dtype=float).reshape(-1, 3)
        if points.shape[0] == 0:
            return []
        budget = int(cfg.get("seed_sample_budget", 7000))
        rng = np.random.default_rng(int(cfg.get("seed_random_seed", 0)))
        if points.shape[0] > budget > 0:
            choose = rng.choice(points.shape[0], size=budget, replace=False)
            sample_idx = np.sort(choose)
        else:
            sample_idx = np.arange(points.shape[0], dtype=np.int32)
        radius = float(cfg.get("linelet_neighbor_radius_mm", 3.0))
        grid = self._grid_from_points(points, radius)
        min_neighbors = int(cfg.get("linelet_min_neighbors", 6))
        max_width = float(cfg.get("linelet_max_width_mm", 1.8))
        min_linearity = float(cfg.get("linelet_min_linearity", 3.0))
        retained: list[dict[str, Any]] = []
        for idx in sample_idx.tolist():
            nbr_ids = self._query_grid_neighbors(points[int(idx)], points, grid, radius)
            if len(nbr_ids) < min_neighbors:
                continue
            nbr_pts = points[np.asarray(nbr_ids, dtype=np.int32)]
            center = np.mean(nbr_pts, axis=0)
            centered = nbr_pts - center.reshape(1, 3)
            cov = np.cov(centered, rowvar=False)
            evals, evecs = np.linalg.eigh(np.asarray(cov, dtype=float))
            order = np.argsort(evals)[::-1]
            evals = np.maximum(evals[order], 0.0)
            axis = self._canonical_direction(evecs[:, order[0]])
            width = float(np.sqrt(max(1e-6, 0.5 * (float(evals[1]) + float(evals[2])))))
            linearity = float(evals[0] / max(float(evals[1]), 1e-6))
            if width > max_width or linearity < min_linearity:
                continue
            confidence = float((linearity / max(min_linearity, 1e-6)) * (len(nbr_ids) / max(1, min_neighbors)))
            retained.append(
                {
                    "sample_index": int(idx),
                    "point_ras": tuple(float(v) for v in points[int(idx)]),
                    "center_ras": tuple(float(v) for v in center),
                    "direction_ras": tuple(float(v) for v in axis),
                    "neighbor_count": int(len(nbr_ids)),
                    "width_mm": float(width),
                    "linearity": float(linearity),
                    "confidence": float(confidence),
                }
            )
        if not retained:
            return []
        cluster_radius = float(cfg.get("linelet_cluster_radius_mm", 4.0))
        cluster_angle = float(cfg.get("linelet_cluster_angle_deg", 18.0))
        min_cluster_samples = int(cfg.get("linelet_min_cluster_samples", 2))
        retained_points = np.asarray([item["center_ras"] for item in retained], dtype=float).reshape(-1, 3)
        retained_axes = np.asarray([item["direction_ras"] for item in retained], dtype=float).reshape(-1, 3)
        retained_conf = np.asarray([float(item["confidence"]) for item in retained], dtype=float)
        rgrid = self._grid_from_points(retained_points, cluster_radius)
        visited = np.zeros((len(retained),), dtype=bool)
        linelets: list[dict[str, Any]] = []
        next_id = 1
        for root in range(len(retained)):
            if bool(visited[root]):
                continue
            queue = [int(root)]
            component: list[int] = []
            visited[root] = True
            while queue:
                cur = int(queue.pop())
                component.append(cur)
                nbr_ids = self._query_grid_neighbors(retained_points[cur], retained_points, rgrid, cluster_radius)
                for nbr in nbr_ids:
                    nbr = int(nbr)
                    if nbr == cur or bool(visited[nbr]):
                        continue
                    if self._angle_deg(retained_axes[cur], retained_axes[nbr]) > cluster_angle:
                        continue
                    visited[nbr] = True
                    queue.append(nbr)
            if len(component) < min_cluster_samples:
                continue
            comp_points = retained_points[np.asarray(component, dtype=np.int32)]
            comp_weights = retained_conf[np.asarray(component, dtype=np.int32)]
            try:
                center, axis = self._weighted_pca_line(comp_points, np.maximum(comp_weights, 1e-6))
            except Exception:
                continue
            projections = (comp_points - center.reshape(1, 3)) @ axis.reshape(3)
            linelets.append(
                {
                    "linelet_id": int(next_id),
                    "point_ras": tuple(float(v) for v in center),
                    "direction_ras": tuple(float(v) for v in axis),
                    "sample_count": int(len(component)),
                    "support_mass": float(np.sum(comp_weights)),
                    "confidence": float(np.mean(comp_weights)),
                    "length_mm": float(np.ptp(projections) if projections.size else 0.0),
                }
            )
            next_id += 1
        linelets = sorted(linelets, key=lambda item: (float(item["confidence"]), float(item["support_mass"])), reverse=True)
        return linelets[: int(cfg.get("linelet_max_count", 220))]

    def _make_seeds_from_linelets_and_blobs(self, linelets: list[dict[str, Any]], seed_support: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
        seeds: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()

        def add_seed(seed_kind: str, point: np.ndarray, direction: np.ndarray, seed_score: float, blob_ids: Iterable[int] = ()) -> None:
            direction = self._canonical_direction(direction)
            key = self._seed_key(point, direction)
            if key in seen:
                return
            seen.add(key)
            seeds.append(
                {
                    "seed_kind": str(seed_kind),
                    "point_ras": tuple(float(v) for v in point),
                    "direction_ras": tuple(float(v) for v in direction),
                    "seed_score": float(seed_score),
                    "seed_blob_ids": tuple(sorted(int(v) for v in blob_ids if int(v) > 0)),
                }
            )

        min_conf = float(cfg.get("linelet_min_confidence", 0.5))
        for item in linelets:
            if float(item.get("confidence", 0.0)) < min_conf:
                continue
            add_seed(
                "linelet",
                np.asarray(item["point_ras"], dtype=float),
                np.asarray(item["direction_ras"], dtype=float),
                float(item.get("support_mass", 0.0)) + 2.0 * float(item.get("confidence", 0.0)),
            )

        if bool(cfg.get("seed_blob_axis_augmentation", True)):
            for blob_id, sample in dict(seed_support.get("blob_samples") or {}).items():
                blob = dict(sample.get("blob") or {})
                length_mm = float(blob.get("length_mm") or 0.0)
                elongation = float(blob.get("elongation") or 1.0)
                if length_mm < float(cfg.get("seed_blob_min_length_mm", 6.0)):
                    continue
                if elongation < float(cfg.get("seed_blob_min_elongation", 4.5)):
                    continue
                centroid = np.asarray(blob.get("centroid_ras") if blob.get("centroid_ras") is not None else np.mean(np.asarray(sample.get("points_ras"), dtype=float), axis=0), dtype=float).reshape(3)
                axis = np.asarray(blob.get("pca_axis_ras") if blob.get("pca_axis_ras") is not None else [0.0, 0.0, 1.0], dtype=float)
                add_seed(
                    "elongated_blob",
                    centroid,
                    axis,
                    float(np.sqrt(max(1.0, float(sample.get("voxel_count") or 1)))) + 0.1 * float(length_mm),
                    [int(blob_id)],
                )
        seeds = sorted(seeds, key=lambda item: float(item.get("seed_score", 0.0)), reverse=True)
        return seeds[: int(cfg.get("max_seed_count", 400))]

    @staticmethod
    def _grid_from_points(points: np.ndarray, cell_size_mm: float) -> dict[tuple[int, int, int], list[int]]:
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        size = max(float(cell_size_mm), 1e-3)
        out: dict[tuple[int, int, int], list[int]] = {}
        for idx, p in enumerate(pts):
            key = (
                int(math.floor(float(p[0]) / size)),
                int(math.floor(float(p[1]) / size)),
                int(math.floor(float(p[2]) / size)),
            )
            out.setdefault(key, []).append(int(idx))
        return out

    @staticmethod
    def _query_grid_neighbors(point: np.ndarray, all_points: np.ndarray, grid: dict[tuple[int, int, int], list[int]], radius_mm: float) -> list[int]:
        p = np.asarray(point, dtype=float).reshape(3)
        pts = np.asarray(all_points, dtype=float).reshape(-1, 3)
        size = max(float(radius_mm), 1e-3)
        key = (
            int(math.floor(float(p[0]) / size)),
            int(math.floor(float(p[1]) / size)),
            int(math.floor(float(p[2]) / size)),
        )
        radius2 = float(radius_mm) * float(radius_mm)
        out: list[int] = []
        for dk in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for di in (-1, 0, 1):
                    cell = (key[0] + dk, key[1] + dj, key[2] + di)
                    for idx in grid.get(cell, []):
                        d2 = float(np.sum((pts[int(idx)] - p) ** 2))
                        if d2 <= radius2:
                            out.append(int(idx))
        return out

    def _proposal_from_seed(
        self,
        hyp_id: int,
        seed: dict[str, Any],
        seed_support: dict[str, Any],
        extension_support: dict[str, Any],
        cfg: dict[str, Any],
    ) -> _Hypothesis | None:
        points = np.asarray(extension_support.get("candidate_points_ras"), dtype=float).reshape(-1, 3)
        if points.shape[0] == 0:
            return None
        depths = None
        if extension_support.get("candidate_depth_mm") is not None:
            depths = np.asarray(extension_support.get("candidate_depth_mm"), dtype=float).reshape(-1)
        center = np.asarray(seed["point_ras"], dtype=float).reshape(3)
        axis = self._canonical_direction(np.asarray(seed["direction_ras"], dtype=float).reshape(3))
        point_ids: np.ndarray | None = None
        support_stats: dict[str, Any] | None = None
        max_iters = int(cfg.get("extension_refit_iters", 3))
        for _ in range(max(1, max_iters)):
            proposal = self._proposal_support_from_axis(
                center=center,
                axis=axis,
                points=points,
                depths=depths,
                extension_support=extension_support,
                seed_support=seed_support,
                seed_confidence=float(seed.get("seed_score", 0.0)),
                cfg=cfg,
            )
            if proposal is None:
                return None
            new_ids = np.asarray(proposal["owned_extension_point_ids"], dtype=np.int32)
            if point_ids is not None and new_ids.shape == point_ids.shape and np.array_equal(new_ids, point_ids):
                support_stats = proposal
                break
            point_ids = new_ids
            owned_points = points[point_ids]
            dist = self._line_distances(owned_points, center, axis)
            weights = 1.0 / (1.0 + (dist / max(float(cfg.get("extension_corridor_radius_mm", 2.2)), 1e-3)) ** 2)
            try:
                center_new, axis_new = self._weighted_pca_line(owned_points, weights)
            except Exception:
                support_stats = proposal
                break
            axis_new = self._canonical_direction(axis_new)
            if float(np.dot(axis_new, axis)) < 0.0:
                axis_new = -axis_new
            if self._angle_deg(axis_new, axis) <= 1.0 and float(np.linalg.norm(center_new - center)) <= 0.75:
                center = center_new
                axis = axis_new
                support_stats = proposal
                break
            center = center_new
            axis = axis_new
            support_stats = proposal
        if support_stats is None:
            return None
        if float(support_stats.get("covered_length_mm", 0.0)) < float(cfg.get("min_proposal_length_mm", 12.0)):
            return None
        if float(support_stats.get("width_mm", 999.0)) > float(cfg.get("extension_max_width_mm", 2.8)):
            return None
        if float(support_stats.get("radial_rms_mm", 999.0)) > float(cfg.get("extension_max_radial_rms_mm", 1.5)):
            return None
        start = np.asarray(support_stats["start_ras"], dtype=float)
        end = np.asarray(support_stats["end_ras"], dtype=float)
        axis = self._canonical_direction(end - start)
        owned_tokens = self._nearby_seed_tokens(seed_support, start, end, axis, cfg)
        owned_token_ids = tuple(int(t.token_id) for t in owned_tokens)
        token_blob_ids = tuple(sorted({int(t.blob_id) for t in owned_tokens}))
        return _Hypothesis(
            hyp_id=int(hyp_id),
            seed_kind=str(seed.get("seed_kind") or "seed"),
            seed_blob_ids=tuple(int(v) for v in list(seed.get("seed_blob_ids") or [])),
            point_ras=tuple(float(v) for v in center),
            direction_ras=tuple(float(v) for v in axis),
            start_ras=tuple(float(v) for v in start),
            end_ras=tuple(float(v) for v in end),
            owned_token_ids=owned_token_ids,
            token_blob_ids=token_blob_ids,
            support_mass=float(support_stats.get("support_mass", 0.0)),
            covered_length_mm=float(support_stats.get("covered_length_mm", 0.0)),
            depth_span_mm=float(support_stats.get("depth_span_mm", 0.0)),
            internal_gap_mm=float(support_stats.get("internal_gap_mm", 0.0)),
            width_mm=float(support_stats.get("width_mm", 0.0)),
            in_head_fraction=float(support_stats.get("in_head_fraction", 0.0)),
            superficial_penalty=float(support_stats.get("superficial_penalty", 0.0)),
            degeneracy_penalty=float(support_stats.get("degeneracy_penalty", 0.0)),
            local_score=float(support_stats.get("score", 0.0)),
            extras={
                "seed_score": float(seed.get("seed_score") or 0.0),
                "seed_point_ras": [float(v) for v in seed.get("point_ras") or ()],
                "owned_extension_point_ids": [int(v) for v in support_stats.get("owned_extension_point_ids") or []],
                "owned_extension_bin_ids": [int(v) for v in support_stats.get("owned_extension_bin_ids") or []],
                "owned_extension_point_count": int(support_stats.get("owned_extension_point_count", 0)),
                "linelet_length_mm": float(support_stats.get("covered_length_mm", 0.0)),
                "radial_rms_mm": float(support_stats.get("radial_rms_mm", 0.0)),
            },
        )

    def _proposal_support_from_axis(
        self,
        *,
        center: np.ndarray,
        axis: np.ndarray,
        points: np.ndarray,
        depths: np.ndarray | None,
        extension_support: dict[str, Any],
        seed_support: dict[str, Any],
        seed_confidence: float,
        cfg: dict[str, Any],
    ) -> dict[str, Any] | None:
        radius = float(cfg.get("extension_corridor_radius_mm", 2.2))
        d = self._line_distances(points, center, axis)
        corridor_ids = np.where(d <= radius)[0]
        if corridor_ids.size < int(cfg.get("extension_min_points", 10)):
            return None
        corridor_points = points[corridor_ids]
        corridor_proj = (corridor_points - center.reshape(1, 3)) @ axis.reshape(3)
        bin_size = max(float(cfg.get("extension_bin_size_mm", 1.0)), 1e-3)
        bin_idx = np.floor(corridor_proj / bin_size).astype(int)
        unique_bins, counts = np.unique(bin_idx, return_counts=True)
        min_bin_points = int(cfg.get("extension_min_bin_points", 2))
        supported_bins = unique_bins[counts >= min_bin_points]
        if supported_bins.size <= 0:
            return None
        seed_bin = 0
        supported_bins = np.sort(supported_bins)
        nearest = int(np.argmin(np.abs(supported_bins - seed_bin)))
        seed_bin = int(supported_bins[nearest])
        interval = self._grow_supported_interval(supported_bins, seed_bin, cfg, bin_size)
        if interval is None:
            return None
        keep_bins = set(int(v) for v in interval["supported_bins"])
        keep_local = np.asarray([int(b) in keep_bins for b in bin_idx], dtype=bool)
        owned_ids = corridor_ids[keep_local]
        if owned_ids.size < int(cfg.get("extension_min_points", 10)):
            return None
        owned_points = points[owned_ids]
        owned_dist = self._line_distances(owned_points, center, axis)
        owned_proj = (owned_points - center.reshape(1, 3)) @ axis.reshape(3)
        t_min = float(np.min(owned_proj))
        t_max = float(np.max(owned_proj))
        start = np.asarray(center, dtype=float) + np.asarray(axis, dtype=float) * t_min
        end = np.asarray(center, dtype=float) + np.asarray(axis, dtype=float) * t_max
        start, end = self._orient_shallow_to_deep(
            start,
            end,
            extension_support.get("preview", {}).get("head_distance_map_kji"),
            extension_support.get("candidate_mask_kji"),
            extension_support,
        )
        hull_mask_kji = seed_support.get("hull_mask_kji")
        if hull_mask_kji is not None:
            start = self._extend_shallow_endpoint_to_mask(
                start_ras=np.asarray(start, dtype=float),
                end_ras=np.asarray(end, dtype=float),
                mask_kji=np.asarray(hull_mask_kji, dtype=bool),
                support=seed_support,
                max_extend_mm=float(cfg.get("seed_hull_extend_cap_mm", 40.0)),
                step_mm=float(cfg.get("seed_hull_extend_step_mm", 0.5)),
            )
        axis = self._canonical_direction(end - start)
        owned_depth = None if depths is None else np.asarray(depths[owned_ids], dtype=float)
        width_mm = float(np.sqrt(np.mean(owned_dist ** 2))) if owned_dist.size else 0.0
        radial_rms = width_mm
        depth_span = 0.0
        superficial_penalty = 0.0
        if owned_depth is not None and owned_depth.size:
            depth_span = float(np.nanmax(owned_depth) - np.nanmin(owned_depth))
            superficial_penalty = float(np.sum(owned_depth < float(cfg.get("superficial_depth_mm", 8.0))))
        in_head_fraction = self._segment_inside_fraction(
            start,
            end,
            extension_support.get("preview", {}).get("gating_mask_kji")
            if extension_support.get("preview", {}).get("gating_mask_kji") is not None
            else extension_support.get("preview", {}).get("head_mask_kji"),
            extension_support,
        )
        covered_length = float(np.linalg.norm(np.asarray(end, dtype=float) - np.asarray(start, dtype=float)))
        short_penalty = max(0.0, float(cfg.get("min_proposal_length_mm", 12.0)) - covered_length)
        support_mass = float(owned_ids.size)
        score = (
            (0.12 * covered_length)
            + (0.03 * support_mass)
            + (0.02 * depth_span)
            + (1.5 * in_head_fraction)
            + (0.08 * seed_confidence)
            - (0.40 * float(interval["internal_gap_mm"]))
            - (1.75 * width_mm)
            - (0.08 * superficial_penalty)
            - (0.50 * short_penalty)
        )
        return {
            "start_ras": tuple(float(v) for v in start),
            "end_ras": tuple(float(v) for v in end),
            "t_min_mm": float(t_min),
            "t_max_mm": float(t_max),
            "covered_length_mm": float(covered_length),
            "support_mass": float(support_mass),
            "depth_span_mm": float(depth_span),
            "internal_gap_mm": float(interval["internal_gap_mm"]),
            "width_mm": float(width_mm),
            "radial_rms_mm": float(radial_rms),
            "in_head_fraction": float(in_head_fraction),
            "superficial_penalty": float(superficial_penalty),
            "degeneracy_penalty": 0.0,
            "owned_extension_point_ids": [int(v) for v in owned_ids.tolist()],
            "owned_extension_point_count": int(owned_ids.size),
            "owned_extension_bin_ids": [int(v) for v in interval["supported_bins"]],
            "score": float(score),
        }

    def _extend_shallow_endpoint_to_mask(
        self,
        *,
        start_ras: np.ndarray,
        end_ras: np.ndarray,
        mask_kji: np.ndarray,
        support: dict[str, Any],
        max_extend_mm: float,
        step_mm: float,
    ) -> np.ndarray:
        ras_to_ijk_fn = support.get("ras_to_ijk_fn") if isinstance(support, dict) else None
        if ras_to_ijk_fn is None or mask_kji is None:
            return np.asarray(start_ras, dtype=float)
        axis = self._canonical_direction(np.asarray(end_ras, dtype=float) - np.asarray(start_ras, dtype=float))
        dims = np.asarray(mask_kji.shape, dtype=int)
        out = np.asarray(start_ras, dtype=float)
        max_steps = max(1, int(math.floor(float(max_extend_mm) / max(float(step_mm), 1e-3))))
        for step_idx in range(1, max_steps + 1):
            p = np.asarray(start_ras, dtype=float) - axis * (float(step_idx) * float(step_mm))
            ijk = ras_to_ijk_fn(p)
            i = int(round(float(ijk[0])))
            j = int(round(float(ijk[1])))
            k = int(round(float(ijk[2])))
            if not (0 <= k < dims[0] and 0 <= j < dims[1] and 0 <= i < dims[2]):
                break
            if not bool(mask_kji[k, j, i]):
                break
            out = p
        return out

    @staticmethod
    def _grow_supported_interval(supported_bins: np.ndarray, seed_bin: int, cfg: dict[str, Any], bin_size_mm: float) -> dict[str, Any] | None:
        bins = np.sort(np.asarray(supported_bins, dtype=int).reshape(-1))
        if bins.size == 0:
            return None
        anchor = int(bins[int(np.argmin(np.abs(bins - int(seed_bin))))])
        anchor_idx = int(np.where(bins == anchor)[0][0])
        normal_gap_bins = int(math.floor(float(cfg.get("extension_normal_gap_mm", 6.0)) / max(bin_size_mm, 1e-3)))
        large_gap_bins = int(math.floor(float(cfg.get("extension_large_gap_mm", 12.0)) / max(bin_size_mm, 1e-3)))
        max_large = int(cfg.get("extension_max_large_gaps", 1))
        left = anchor
        right = anchor
        internal_gap = 0.0
        large_used = 0
        prev = anchor
        idx = anchor_idx - 1
        while idx >= 0:
            b = int(bins[idx])
            gap = int(prev - b - 1)
            if gap <= normal_gap_bins:
                internal_gap += float(max(0, gap)) * float(bin_size_mm)
                left = b
                prev = b
                idx -= 1
                continue
            if large_used < max_large and gap <= large_gap_bins:
                internal_gap += float(max(0, gap)) * float(bin_size_mm)
                large_used += 1
                left = b
                prev = b
                idx -= 1
                continue
            break
        prev = anchor
        idx = anchor_idx + 1
        while idx < bins.size:
            b = int(bins[idx])
            gap = int(b - prev - 1)
            if gap <= normal_gap_bins:
                internal_gap += float(max(0, gap)) * float(bin_size_mm)
                right = b
                prev = b
                idx += 1
                continue
            if large_used < max_large and gap <= large_gap_bins:
                internal_gap += float(max(0, gap)) * float(bin_size_mm)
                large_used += 1
                right = b
                prev = b
                idx += 1
                continue
            break
        keep = [int(v) for v in bins.tolist() if int(left) <= int(v) <= int(right)]
        return {
            "supported_bins": keep,
            "internal_gap_mm": float(internal_gap),
        }

    def _nearby_seed_tokens(self, seed_support: dict[str, Any], start_ras: np.ndarray, end_ras: np.ndarray, axis: np.ndarray, cfg: dict[str, Any]) -> list[_SupportToken]:
        tokens = list(seed_support.get("tokens") or [])
        if not tokens:
            return []
        points = np.asarray([t.point_ras for t in tokens], dtype=float).reshape(-1, 3)
        origin = np.asarray(start_ras, dtype=float).reshape(3)
        axis = self._canonical_direction(axis)
        seg_len = float(np.linalg.norm(np.asarray(end_ras, dtype=float) - np.asarray(start_ras, dtype=float)))
        dist = self._line_distances(points, origin, axis)
        proj = (points - origin.reshape(1, 3)) @ axis.reshape(3)
        radius = float(cfg.get("corridor_radius_mm", 2.2))
        keep = (dist <= radius) & (proj >= -3.0) & (proj <= seg_len + 3.0)
        owned = [tokens[int(i)] for i in np.where(keep)[0].tolist()]
        if owned:
            return owned
        nearest = np.argsort(dist)[: min(6, len(tokens))]
        return [tokens[int(i)] for i in nearest.tolist()]

    def _conflicts(self, hypotheses: list[_Hypothesis], support_bundle: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
        redundancy_weight = float(cfg.get("selection_redundancy_weight", 2.5))
        shared_weight = float(cfg.get("selection_shared_token_weight", 0.6))
        out: list[dict[str, Any]] = []
        for i, a in enumerate(hypotheses):
            owned_a = set(int(v) for v in list(dict(a.extras or {}).get("owned_extension_point_ids") or []))
            p0 = 0.5 * (np.asarray(a.start_ras, dtype=float) + np.asarray(a.end_ras, dtype=float))
            d0 = self._canonical_direction(np.asarray(a.direction_ras, dtype=float))
            for b in hypotheses[i + 1 :]:
                owned_b = set(int(v) for v in list(dict(b.extras or {}).get("owned_extension_point_ids") or []))
                shared_ids = sorted(owned_a & owned_b)
                shared_mass = float(len(shared_ids))
                p1 = 0.5 * (np.asarray(b.start_ras, dtype=float) + np.asarray(b.end_ras, dtype=float))
                d1 = self._canonical_direction(np.asarray(b.direction_ras, dtype=float))
                angle = self._angle_deg(d0, d1)
                line_dist = self._line_to_line_distance(p0, d0, p1, d1)
                overlap = self._segment_overlap_fraction(a.start_ras, a.end_ras, b.start_ras, b.end_ras, d1)
                redundant = angle <= 10.0 and line_dist <= 2.5 and overlap >= 0.3
                if shared_mass <= 0.0 and not redundant:
                    continue
                penalty = (shared_weight * shared_mass) + (redundancy_weight * overlap if redundant else 0.0)
                out.append(
                    {
                        "a": int(a.hyp_id),
                        "b": int(b.hyp_id),
                        "shared_extension_point_ids": shared_ids,
                        "shared_mass": float(shared_mass),
                        "angle_deg": float(angle),
                        "line_distance_mm": float(line_dist),
                        "segment_overlap_fraction": float(overlap),
                        "penalty": float(penalty),
                    }
                )
        return out

    def _select_hypotheses(self, hypotheses: list[_Hypothesis], support_bundle: dict[str, Any], cfg: dict[str, Any]) -> tuple[list[_Hypothesis], list[dict[str, Any]]]:
        if not hypotheses:
            return [], []
        hypotheses = list(hypotheses)
        conflicts = self._conflicts(hypotheses, support_bundle, cfg)
        penalty_pairs = {(int(item["a"]), int(item["b"])): float(item["penalty"]) for item in conflicts}
        base_scores = {int(h.hyp_id): float(h.local_score) for h in hypotheses}
        complexity = float(cfg.get("selection_complexity_cost", 2.0))
        min_gain = float(cfg.get("selection_min_gain", 0.25))
        hyp_by_id = {int(h.hyp_id): h for h in hypotheses}
        ordered_ids = [int(h.hyp_id) for h in sorted(hypotheses, key=lambda item: float(item.local_score), reverse=True)]

        def objective(selected_ids: set[int]) -> float:
            total = 0.0
            ordered = sorted(selected_ids)
            for hyp_id in ordered:
                total += float(base_scores[hyp_id]) - complexity
            for i, a in enumerate(ordered):
                for b in ordered[i + 1 :]:
                    total -= float(penalty_pairs.get((a, b), penalty_pairs.get((b, a), 0.0)))
            return total

        selected_ids: set[int] = set()
        current_obj = 0.0
        for hyp_id in ordered_ids:
            trial = set(selected_ids)
            trial.add(hyp_id)
            trial_obj = objective(trial)
            if trial_obj > current_obj + min_gain:
                selected_ids = trial
                current_obj = trial_obj
        passes = int(cfg.get("selection_refine_passes", 2))
        for _ in range(max(1, passes)):
            improved = False
            for hyp_id in list(selected_ids):
                trial = set(selected_ids)
                trial.remove(hyp_id)
                trial_obj = objective(trial)
                if trial_obj > current_obj + 1e-6:
                    selected_ids = trial
                    current_obj = trial_obj
                    improved = True
            for hyp_id in ordered_ids:
                if hyp_id in selected_ids:
                    continue
                trial = set(selected_ids)
                trial.add(hyp_id)
                trial_obj = objective(trial)
                if trial_obj > current_obj + min_gain:
                    selected_ids = trial
                    current_obj = trial_obj
                    improved = True
            rejected = [hyp_id for hyp_id in ordered_ids if hyp_id not in selected_ids]
            for add_id in rejected:
                for drop_id in list(selected_ids):
                    trial = set(selected_ids)
                    trial.remove(drop_id)
                    trial.add(add_id)
                    trial_obj = objective(trial)
                    if trial_obj > current_obj + min_gain:
                        selected_ids = trial
                        current_obj = trial_obj
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        selected = [hyp_by_id[hyp_id] for hyp_id in ordered_ids if hyp_id in selected_ids]
        selected = self._extension_ownership_cleanup(selected, support_bundle, cfg)
        return selected, conflicts

    def _extension_ownership_cleanup(self, selected: list[_Hypothesis], support_bundle: dict[str, Any], cfg: dict[str, Any]) -> list[_Hypothesis]:
        if not selected:
            return []
        extension_support = dict(support_bundle.get("extension_pool") or {})
        points = np.asarray(extension_support.get("candidate_points_ras"), dtype=float).reshape(-1, 3)
        if points.shape[0] == 0:
            return selected
        claim_ids = sorted({int(v) for hyp in selected for v in list(dict(hyp.extras or {}).get("owned_extension_point_ids") or [])})
        if not claim_ids:
            return selected
        claim_points = points[np.asarray(claim_ids, dtype=np.int32)]
        radius = float(cfg.get("extension_assignment_radius_mm", 2.5))
        min_affinity = float(cfg.get("extension_assignment_min_affinity", 0.05))
        affinities = np.zeros((claim_points.shape[0], len(selected)), dtype=float)
        for j, hyp in enumerate(selected):
            start = np.asarray(hyp.start_ras, dtype=float)
            end = np.asarray(hyp.end_ras, dtype=float)
            axis = self._canonical_direction(end - start)
            seg_len = float(np.linalg.norm(end - start))
            dist = self._line_distances(claim_points, start, axis)
            proj = (claim_points - start.reshape(1, 3)) @ axis.reshape(3)
            window = (proj >= -3.0) & (proj <= seg_len + 3.0)
            affinities[:, j] = np.exp(-0.5 * (dist / max(radius, 1e-3)) ** 2) * window.astype(float)
        best_idx = np.argmax(affinities, axis=1)
        best_val = np.max(affinities, axis=1)
        assigned: dict[int, list[int]] = {int(h.hyp_id): [] for h in selected}
        for local_idx, point_id in enumerate(claim_ids):
            if float(best_val[local_idx]) < min_affinity:
                continue
            assigned[int(selected[int(best_idx[local_idx])].hyp_id)].append(int(point_id))
        hyp_by_id = {int(h.hyp_id): h for h in selected}
        out: list[_Hypothesis] = []
        seed_support = dict(support_bundle.get("seed_pool") or {})
        depths = None
        if extension_support.get("candidate_depth_mm") is not None:
            depths = np.asarray(extension_support.get("candidate_depth_mm"), dtype=float).reshape(-1)
        for hyp in selected:
            owned_ids = assigned.get(int(hyp.hyp_id)) or list(dict(hyp.extras or {}).get("owned_extension_point_ids") or [])
            if len(owned_ids) < int(cfg.get("extension_min_points", 10)):
                continue
            owned_points = points[np.asarray(owned_ids, dtype=np.int32)]
            weights = np.ones((owned_points.shape[0],), dtype=float)
            try:
                center, axis = self._weighted_pca_line(owned_points, weights)
            except Exception:
                out.append(hyp)
                continue
            support_stats = self._proposal_support_from_axis(
                center=center,
                axis=axis,
                points=points,
                depths=depths,
                extension_support=extension_support,
                seed_support=seed_support,
                seed_confidence=float(dict(hyp.extras or {}).get("seed_score", 0.0)),
                cfg=cfg,
            )
            if support_stats is None:
                out.append(hyp)
                continue
            start = np.asarray(support_stats["start_ras"], dtype=float)
            end = np.asarray(support_stats["end_ras"], dtype=float)
            axis = self._canonical_direction(end - start)
            owned_tokens = self._nearby_seed_tokens(seed_support, start, end, axis, cfg)
            new_extras = dict(hyp.extras or {})
            new_extras["owned_extension_point_ids"] = [int(v) for v in support_stats.get("owned_extension_point_ids") or []]
            new_extras["owned_extension_bin_ids"] = [int(v) for v in support_stats.get("owned_extension_bin_ids") or []]
            new_extras["owned_extension_point_count"] = int(support_stats.get("owned_extension_point_count", 0))
            new_extras["ownership_cleanup"] = True
            out.append(
                _Hypothesis(
                    hyp_id=int(hyp.hyp_id),
                    seed_kind=str(hyp.seed_kind),
                    seed_blob_ids=tuple(hyp.seed_blob_ids),
                    point_ras=tuple(float(v) for v in center),
                    direction_ras=tuple(float(v) for v in axis),
                    start_ras=tuple(float(v) for v in start),
                    end_ras=tuple(float(v) for v in end),
                    owned_token_ids=tuple(int(t.token_id) for t in owned_tokens),
                    token_blob_ids=tuple(sorted({int(t.blob_id) for t in owned_tokens})),
                    support_mass=float(support_stats.get("support_mass", hyp.support_mass)),
                    covered_length_mm=float(support_stats.get("covered_length_mm", hyp.covered_length_mm)),
                    depth_span_mm=float(support_stats.get("depth_span_mm", hyp.depth_span_mm)),
                    internal_gap_mm=float(support_stats.get("internal_gap_mm", hyp.internal_gap_mm)),
                    width_mm=float(support_stats.get("width_mm", hyp.width_mm)),
                    in_head_fraction=float(support_stats.get("in_head_fraction", hyp.in_head_fraction)),
                    superficial_penalty=float(support_stats.get("superficial_penalty", hyp.superficial_penalty)),
                    degeneracy_penalty=float(support_stats.get("degeneracy_penalty", hyp.degeneracy_penalty)),
                    local_score=float(support_stats.get("score", hyp.local_score)),
                    extras=new_extras,
                )
            )
        return sorted(out, key=lambda item: float(item.local_score), reverse=True)

    def _record_support_diagnostics(self, diagnostics, support: dict[str, Any]) -> None:
        seed_support = dict(support.get("seed_pool") or {})
        extension_support = dict(support.get("extension_pool") or {})
        super()._record_support_diagnostics(diagnostics, seed_support)
        diagnostics.set_extra("seed_support_pool_name", str(seed_support.get("pool_name") or "seed_pool"))
        diagnostics.set_extra("seed_chosen_threshold_hu", float(seed_support.get("threshold_hu") or 0.0))
        diagnostics.set_extra("seed_deep_core_shrink_mm", float(seed_support.get("deep_core_shrink_mm") or 0.0))
        diagnostics.set_extra("seed_hull_stats", dict(seed_support.get("hull_stats") or {}))
        diagnostics.set_extra("extension_support_pool_name", str(extension_support.get("pool_name") or "extension_pool"))
        diagnostics.set_extra("extension_chosen_threshold_hu", float(extension_support.get("threshold_hu") or 0.0))
        diagnostics.set_extra("extension_threshold_trials", list(extension_support.get("preview_stats") or []))
        diagnostics.set_count("extension_candidate_points_after_depth", int(extension_support.get("preview", {}).get("depth_kept_count") or 0))
        diagnostics.set_count("extension_candidate_points_total", int(np.asarray(extension_support.get("candidate_points_ras"), dtype=float).reshape(-1, 3).shape[0]))
        diagnostics.set_count("seed_bridge_candidate_points_after_depth", int(seed_support.get("bridge_candidate_points_after_mask") or 0))

    def _attach_support_artifact(self, result: DetectionResult, ctx: DetectionContext, support: dict[str, Any]) -> None:
        writer = self.get_artifact_writer(ctx, result)
        seed_support = dict(support.get("seed_pool") or {})
        token_rows = [
            [
                int(token.token_id),
                int(token.blob_id),
                str(token.blob_kind),
                float(token.point_ras[0]),
                float(token.point_ras[1]),
                float(token.point_ras[2]),
                float(token.weight),
                float(token.purity),
                int(bool(token.seedable)),
                float(token.depth_mm),
            ]
            for token in list(seed_support.get("tokens") or [])
        ]
        token_path = writer.write_csv_rows(
            "de_novo_v2_seed_tokens.csv",
            ["token_id", "blob_id", "blob_kind", "x", "y", "z", "weight", "purity", "seedable", "depth_mm"],
            token_rows,
        )
        add_artifact(result["artifacts"], kind="token_csv", path=token_path, description="De novo v2 seed support tokens", stage="support_pool")
        result["artifacts"].extend(
            write_standard_artifacts(
                writer,
                result,
                blobs=[],
                pipeline_payload={
                    "pipeline_id": self.pipeline_id,
                    "pipeline_version": self.pipeline_version,
                    "seed_support": {
                        "threshold_hu": float(seed_support.get("threshold_hu") or 0.0),
                        "preview_stats": list(seed_support.get("preview_stats") or []),
                        "token_summary": dict(seed_support.get("token_summary") or {}),
                    },
                    "extension_support": {
                        "threshold_hu": float(dict(support.get("extension_pool") or {}).get("threshold_hu") or 0.0),
                        "preview_stats": list(dict(support.get("extension_pool") or {}).get("preview_stats") or []),
                    },
                },
            )
        )
