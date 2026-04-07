"""De novo shank detection by hypothesis generation plus global selection.

This pipeline is benchmark-only in v1. It does not require an initial shank count
or seed trajectory. It over-generates generic straight-shaft hypotheses from a
subject-level support pool, then selects a non-redundant subset by maximizing a
simple global objective before running the stable guided-fit refiner on each
selected hypothesis.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from rosa_core.contact_fit import fit_electrode_axis_and_tip
from shank_core.blob_candidates import extract_blob_candidates
from shank_core.masking import build_preview_masks

try:
    import SimpleITK as sitk
except ImportError:  # pragma: no cover
    sitk = None

from ..artifacts import add_artifact, write_standard_artifacts
from ..contracts import DetectionContext, DetectionResult
from .base import BaseDetectionPipeline


@dataclass(frozen=True)
class _SupportToken:
    token_id: int
    point_ras: tuple[float, float, float]
    weight: float
    purity: float
    blob_id: int
    blob_kind: str
    local_axis_ras: tuple[float, float, float]
    depth_mm: float
    seedable: bool
    meta: dict[str, Any]


@dataclass(frozen=True)
class _Hypothesis:
    hyp_id: int
    seed_kind: str
    seed_blob_ids: tuple[int, ...]
    point_ras: tuple[float, float, float]
    direction_ras: tuple[float, float, float]
    start_ras: tuple[float, float, float]
    end_ras: tuple[float, float, float]
    owned_token_ids: tuple[int, ...]
    token_blob_ids: tuple[int, ...]
    support_mass: float
    covered_length_mm: float
    depth_span_mm: float
    internal_gap_mm: float
    width_mm: float
    in_head_fraction: float
    superficial_penalty: float
    degeneracy_penalty: float
    local_score: float
    extras: dict[str, Any]


@dataclass(frozen=True)
class _SelectedHypothesis:
    hyp_id: int
    start_ras: tuple[float, float, float]
    end_ras: tuple[float, float, float]
    direction_ras: tuple[float, float, float]
    owned_token_ids: tuple[int, ...]
    local_score: float
    global_score: float
    late_model_prior_score: float
    best_model_id: str
    refined_start_ras: tuple[float, float, float] | None
    refined_end_ras: tuple[float, float, float] | None
    refinement_success: bool
    refinement_reason: str
    extras: dict[str, Any]


class DeNovoHypothesisSelectV1Pipeline(BaseDetectionPipeline):
    pipeline_id = "de_novo_hypothesis_select_v1"
    display_name = "De Novo Hypothesis+Select v1"
    pipeline_version = "1.0.0"
    scaffold = False
    expose_in_ui = False

    def _defaults(self, cfg: dict[str, Any]) -> dict[str, Any]:
        out = dict(cfg or {})
        out.setdefault("threshold_schedule_hu", [600.0, 550.0, 500.0])
        out.setdefault("target_candidate_points", 300000)
        out.setdefault("use_head_mask", True)
        out.setdefault("build_head_mask", True)
        out.setdefault("head_mask_threshold_hu", -500.0)
        out.setdefault("head_mask_aggressive_cleanup", True)
        out.setdefault("head_mask_close_mm", 2.0)
        out.setdefault("head_mask_method", "outside_air")
        out.setdefault("head_mask_metal_dilate_mm", 1.0)
        out.setdefault("head_gate_erode_vox", 1)
        out.setdefault("head_gate_dilate_vox", 1)
        out.setdefault("head_gate_margin_mm", 0.0)
        out.setdefault("min_metal_depth_mm", 5.0)
        out.setdefault("max_metal_depth_mm", 220.0)
        out.setdefault("proposal_threshold_schedule_hu", [1500.0])
        out.setdefault("proposal_target_candidate_points", 0)
        out.setdefault("proposal_min_metal_depth_mm", 10.0)
        out.setdefault("proposal_max_metal_depth_mm", float(out.get("max_metal_depth_mm", 220.0)))
        out.setdefault("proposal_allow_low_threshold_fallback", True)
        out.setdefault("proposal_support_mode", "depth_mask")
        out.setdefault("proposal_support_mode_union_members", ["depth_mask", "gaussian_residual"])
        out.setdefault("proposal_objectness_candidate_mode", "metal_in_head")
        out.setdefault("proposal_objectness_gaussian_sigma_mm", 1.0)
        out.setdefault("proposal_objectness_alpha", 0.2)
        out.setdefault("proposal_objectness_beta", 0.2)
        out.setdefault("proposal_objectness_gamma", 5.0)
        out.setdefault("proposal_objectness_scale_measure", True)
        out.setdefault("proposal_objectness_bright_object", True)
        out.setdefault("proposal_objectness_dimension", 1)
        out.setdefault("proposal_objectness_response_percentiles", [99.8, 99.5, 99.0, 98.5, 98.0, 97.5, 97.0, 96.0, 95.0, 92.5, 90.0])
        out.setdefault("proposal_objectness_target_points", 2500)
        out.setdefault("proposal_objectness_min_points", 300)
        out.setdefault("proposal_objectness_max_points", 6000)
        out.setdefault("proposal_objectness_min_fraction_of_max", 0.10)
        out.setdefault("proposal_residual_grow_vox", 1)
        out.setdefault("proposal_residual_sigma_mm", 4.0)
        out.setdefault("proposal_residual_blur_threshold", 0.2)
        out.setdefault("proposal_ellipsoid_shrink_factor", 0.9)
        out.setdefault("proposal_ellipsoid_fit_max_points", 200000)
        out.setdefault("proposal_ellipsoid_candidate_mode", "metal_in_head")
        out.setdefault("refinement_threshold_schedule_hu", list(out.get("threshold_schedule_hu") or [600.0, 550.0, 500.0]))
        out.setdefault("refinement_target_candidate_points", int(out.get("target_candidate_points", 300000)))
        out.setdefault("refinement_min_metal_depth_mm", float(out.get("min_metal_depth_mm", 5.0)))
        out.setdefault("refinement_max_metal_depth_mm", float(out.get("max_metal_depth_mm", 220.0)))
        out.setdefault("compact_max_length_mm", 4.5)
        out.setdefault("compact_max_diameter_mm", 3.8)
        out.setdefault("compact_max_elongation", 4.0)
        out.setdefault("elongated_min_length_mm", 6.0)
        out.setdefault("elongated_min_elongation", 4.5)
        out.setdefault("token_spacing_mm", 3.0)
        out.setdefault("max_tokens_per_blob", 10)
        out.setdefault("max_complex_tokens_per_blob", 4)
        out.setdefault("hough_vote_min_depth_mm", 8.0)
        out.setdefault("hough_vote_max_depth_mm", 60.0)
        out.setdefault("hough_vote_min_pair_distance_mm", 10.0)
        out.setdefault("hough_vote_max_pair_distance_mm", 90.0)
        out.setdefault("hough_direction_bin_deg", 6.0)
        out.setdefault("hough_direction_consensus_deg", 8.0)
        out.setdefault("hough_top_directions", 24)
        out.setdefault("hough_offset_bin_mm", 4.0)
        out.setdefault("hough_min_bin_pair_count", 2)
        out.setdefault("hough_min_offset_points", 3)
        out.setdefault("hough_max_offset_bins_per_direction", 8)
        out.setdefault("hough_max_vote_points", 100000)
        out.setdefault("hough_pair_samples", 160000)
        out.setdefault("hough_random_seed", 0)
        out.setdefault("hough_max_bins", 240)
        out.setdefault("hough_max_seed_count", 180)
        out.setdefault("seed_neighbor_radius_mm", 16.0)
        out.setdefault("seed_pair_max_per_token", 6)
        out.setdefault("seed_triple_max_per_token", 3)
        out.setdefault("max_seed_count", 400)
        out.setdefault("corridor_radius_mm", 2.2)
        out.setdefault("seed_corridor_radius_mm", 8.0)
        out.setdefault("min_support_tokens", 4)
        out.setdefault("min_unique_blobs", 3)
        out.setdefault("projection_gap_mm", 6.0)
        out.setdefault("expected_internal_spacing_mm", 3.0)
        out.setdefault("superficial_depth_mm", 8.0)
        out.setdefault("nms_angle_deg", 8.0)
        out.setdefault("nms_line_distance_mm", 2.0)
        out.setdefault("nms_overlap_fraction", 0.55)
        out.setdefault("max_hypotheses_after_nms", 160)
        out.setdefault("selection_complexity_cost", 2.5)
        out.setdefault("selection_shared_token_weight", 1.1)
        out.setdefault("selection_redundancy_weight", 3.0)
        out.setdefault("selection_min_gain", 0.5)
        out.setdefault("selection_refine_passes", 2)
        out.setdefault("ownership_radius_mm", 2.5)
        out.setdefault("ownership_min_affinity", 0.08)
        out.setdefault("guided_fit_mode", "deep_anchor_v2")
        out.setdefault("guided_roi_radius_mm", 5.0)
        out.setdefault("guided_max_angle_deg", 12.0)
        out.setdefault("guided_max_depth_shift_mm", 6.0)
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
        try:
            if "arr_kji" not in ctx or "ijk_kji_to_ras_fn" not in ctx:
                result["warnings"].append("de_novo_hypothesis_select_v1 missing volume context; returning empty result")
                diagnostics.note("de_novo_hypothesis_select_v1 requires arr_kji and ijk_kji_to_ras_fn")
                return self.finalize(result, diagnostics, t_start)

            support = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="support_pool",
                fn=lambda: self._build_support_pool(ctx, cfg, pool_name="proposal"),
            )
            self._record_support_diagnostics(diagnostics, support)
            self._attach_support_artifact(result, ctx, support)
            if stage == "support_pool" or not support["tokens"]:
                result["trajectories"] = []
                result["contacts"] = []
                return self.finalize(result, diagnostics, t_start)

            hypotheses = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="hypothesis_generation", fn=lambda: self._generate_hypotheses(ctx, support, cfg))
            hypotheses = self._nms_hypotheses(hypotheses, cfg)
            diagnostics.set_count("hypothesis_count_generated", int(len(hypotheses)))
            diagnostics.set_extra("hypotheses", [self._hypothesis_json(h) for h in hypotheses[: int(cfg.get("max_hypotheses_after_nms", 160))]])
            if stage == "hypothesis_generation":
                result["trajectories"] = [self._trajectory_from_hypothesis(h) for h in hypotheses]
                result["contacts"] = []
                result["warnings"].append("contact_detection_not_implemented")
                return self.finalize(result, diagnostics, t_start)

            selected, conflicts = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="global_selection", fn=lambda: self._select_hypotheses(hypotheses, support, cfg))
            diagnostics.set_count("selected_count", int(len(selected)))
            diagnostics.set_count("conflict_edge_count", int(len(conflicts)))
            diagnostics.set_extra("conflicts", conflicts)
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
                fn=lambda: self._refine_and_rerank_selected(ctx, support, refinement_support, selected, cfg),
            )
            result["trajectories"] = [self._trajectory_from_selected(h) for h in final_selected]
            result["contacts"] = []
            result["warnings"].append("contact_detection_not_implemented")
            diagnostics.set_extra("selected_hypotheses", [self._selected_json(h) for h in final_selected])
        except Exception as exc:  # pragma: no cover - base handles shaping
            self.fail(ctx=ctx, result=result, diagnostics=diagnostics, stage=str(getattr(exc, "stage", "pipeline")), exc=exc)
        return self.finalize(result, diagnostics, t_start)

    def _support_pool_cfg(self, cfg: dict[str, Any], pool_name: str) -> dict[str, Any]:
        prefix = f"{str(pool_name).strip().lower()}_"
        return {
            "threshold_schedule_hu": list(cfg.get(f"{prefix}threshold_schedule_hu") or cfg.get("threshold_schedule_hu") or [600.0, 550.0, 500.0]),
            "target_candidate_points": int(cfg.get(f"{prefix}target_candidate_points", cfg.get("target_candidate_points", 300000))),
            "min_metal_depth_mm": float(cfg.get(f"{prefix}min_metal_depth_mm", cfg.get("min_metal_depth_mm", 5.0))),
            "max_metal_depth_mm": float(cfg.get(f"{prefix}max_metal_depth_mm", cfg.get("max_metal_depth_mm", 220.0))),
        }

    def _build_support_pool(self, ctx: DetectionContext, cfg: dict[str, Any], pool_name: str = "proposal") -> dict[str, Any]:
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
                build_preview_masks(
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
                if use_masking
                else self._simple_preview_masks(arr_kji=arr_kji, threshold=float(threshold))
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
                build_preview_masks(
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
                if use_masking
                else self._simple_preview_masks(arr_kji=arr_kji, threshold=float(chosen_threshold))
            )

        candidate_mask_kji = np.asarray(chosen_preview.get("metal_depth_pass_mask_kji"), dtype=bool)
        support_mode_stats: dict[str, Any] | None = None
        if pool_name == "proposal":
            support_mode = str(cfg.get("proposal_support_mode") or "depth_mask").strip().lower()
            if support_mode == "union_depth_and_gaussian":
                return self._build_union_proposal_support_pool(
                    ctx=ctx,
                    cfg=cfg,
                    arr_kji=arr_kji,
                    spacing_xyz=spacing_xyz,
                    chosen_preview=chosen_preview,
                    chosen_threshold=float(chosen_threshold),
                    preview_stats=preview_stats,
                )
            if support_mode == "objectness_ridge":
                candidate_source_mode = str(cfg.get("proposal_objectness_candidate_mode") or "metal_in_head").strip().lower()
                source_mask = (
                    np.asarray(chosen_preview.get("metal_depth_pass_mask_kji"), dtype=bool)
                    if candidate_source_mode == "depth_mask"
                    else np.asarray(chosen_preview.get("metal_in_head_mask_kji"), dtype=bool)
                )
                candidate_mask_kji, support_mode_stats = self._proposal_objectness_mask(
                    arr_kji=arr_kji,
                    candidate_mask_kji=source_mask,
                    head_mask_kji=np.asarray(chosen_preview.get("head_mask_kji"), dtype=bool),
                    spacing_xyz=spacing_xyz,
                    gaussian_sigma_mm=float(cfg.get("proposal_objectness_gaussian_sigma_mm", 1.0)),
                    alpha=float(cfg.get("proposal_objectness_alpha", 0.2)),
                    beta=float(cfg.get("proposal_objectness_beta", 0.2)),
                    gamma=float(cfg.get("proposal_objectness_gamma", 5.0)),
                    scale_measure=bool(cfg.get("proposal_objectness_scale_measure", True)),
                    bright_object=bool(cfg.get("proposal_objectness_bright_object", True)),
                    object_dimension=int(cfg.get("proposal_objectness_dimension", 1)),
                    response_percentiles=list(cfg.get("proposal_objectness_response_percentiles") or [99.8, 99.5, 99.0, 98.5, 98.0, 97.5, 97.0, 96.0, 95.0, 92.5, 90.0]),
                    target_points=int(cfg.get("proposal_objectness_target_points", 2500)),
                    min_points=int(cfg.get("proposal_objectness_min_points", 300)),
                    max_points=int(cfg.get("proposal_objectness_max_points", 6000)),
                    min_fraction_of_max=float(cfg.get("proposal_objectness_min_fraction_of_max", 0.10)),
                )
            if support_mode == "gaussian_residual":
                candidate_mask_kji = self._proposal_gaussian_residual_mask(
                    arr_kji=arr_kji,
                    gate_mask_kji=np.asarray(chosen_preview.get("metal_depth_pass_mask_kji"), dtype=bool),
                    spacing_xyz=spacing_xyz,
                    threshold_hu=float(chosen_threshold),
                    grow_vox=int(cfg.get("proposal_residual_grow_vox", 1)),
                    sigma_mm=float(cfg.get("proposal_residual_sigma_mm", 4.0)),
                    blur_threshold=float(cfg.get("proposal_residual_blur_threshold", 0.2)),
                )
            elif support_mode == "ellipsoid_core":
                candidate_source_mode = str(cfg.get("proposal_ellipsoid_candidate_mode") or "metal_in_head").strip().lower()
                source_mask = (
                    np.asarray(chosen_preview.get("metal_depth_pass_mask_kji"), dtype=bool)
                    if candidate_source_mode == "depth_mask"
                    else np.asarray(chosen_preview.get("metal_in_head_mask_kji"), dtype=bool)
                )
                candidate_mask_kji = self._proposal_ellipsoid_core_mask(
                    head_mask_kji=np.asarray(chosen_preview.get("head_mask_kji"), dtype=bool),
                    candidate_mask_kji=source_mask,
                    spacing_xyz=spacing_xyz,
                    shrink_factor=float(cfg.get("proposal_ellipsoid_shrink_factor", 0.9)),
                    max_fit_points=int(cfg.get("proposal_ellipsoid_fit_max_points", 200000)),
                    seed=int(cfg.get("hough_random_seed", 0)),
                )
        if (
            pool_name == "proposal"
            and int(np.count_nonzero(candidate_mask_kji)) <= 0
            and bool(cfg.get("proposal_allow_low_threshold_fallback", True))
        ):
            fallback_cfg = dict(cfg)
            fallback_cfg["proposal_allow_low_threshold_fallback"] = False
            fallback_cfg["proposal_threshold_schedule_hu"] = list(cfg.get("threshold_schedule_hu") or [600.0, 550.0, 500.0])
            fallback_cfg["proposal_target_candidate_points"] = int(cfg.get("target_candidate_points", 300000))
            fallback_cfg["proposal_min_metal_depth_mm"] = float(cfg.get("min_metal_depth_mm", 5.0))
            fallback_cfg["proposal_max_metal_depth_mm"] = float(cfg.get("max_metal_depth_mm", 220.0))
            fallback_support = self._build_support_pool(ctx, fallback_cfg, pool_name="proposal")
            fallback_support["pool_name"] = "proposal_fallback"
            return fallback_support
        ijk_kji = np.argwhere(candidate_mask_kji)
        points_ras = self._ijk_kji_to_ras(ctx, ijk_kji.astype(float)) if ijk_kji.size else np.zeros((0, 3), dtype=float)
        points_lps = self._ras_to_lps(points_ras)
        depth_map_kji = chosen_preview.get("head_distance_map_kji")
        candidate_depth_mm = None
        if depth_map_kji is not None and ijk_kji.size:
            candidate_depth_mm = np.asarray(depth_map_kji[ijk_kji[:, 0], ijk_kji[:, 1], ijk_kji[:, 2]], dtype=float).reshape(-1)
        blob_result = extract_blob_candidates(
            candidate_mask_kji,
            arr_kji=arr_kji,
            depth_map_kji=depth_map_kji,
            ijk_kji_to_ras_fn=ctx["ijk_kji_to_ras_fn"],
            fully_connected=True,
        )
        blob_samples = self._blob_samples(blob_result, ctx, arr_kji)
        tokens, token_summary = self._tokens_from_blobs(blob_result, blob_samples, cfg)
        return {
            "pool_name": str(pool_name),
            "support_mode": str(cfg.get("proposal_support_mode") or "depth_mask") if pool_name == "proposal" else "depth_mask",
            "threshold_hu": float(chosen_threshold),
            "preview_stats": preview_stats,
            "preview": chosen_preview,
            "candidate_mask_kji": candidate_mask_kji,
            "candidate_points_after_mask": int(np.count_nonzero(candidate_mask_kji)),
            "candidate_points_ras": points_ras,
            "candidate_points_lps": points_lps,
            "candidate_depth_mm": candidate_depth_mm,
            "blob_result": blob_result,
            "blob_samples": blob_samples,
            "tokens": tokens,
            "token_summary": token_summary,
            "ras_to_ijk_fn": ctx.get("ras_to_ijk_fn"),
            "support_mode_stats": support_mode_stats,
        }

    def _build_union_proposal_support_pool(
        self,
        *,
        ctx: DetectionContext,
        cfg: dict[str, Any],
        arr_kji: np.ndarray,
        spacing_xyz: tuple[float, float, float],
        chosen_preview: dict[str, Any],
        chosen_threshold: float,
        preview_stats: list[dict[str, Any]],
    ) -> dict[str, Any]:
        base_depth_mask = np.asarray(chosen_preview.get("metal_depth_pass_mask_kji"), dtype=bool)
        gaussian_mask = self._proposal_gaussian_residual_mask(
            arr_kji=arr_kji,
            gate_mask_kji=base_depth_mask,
            spacing_xyz=spacing_xyz,
            threshold_hu=float(chosen_threshold),
            grow_vox=int(cfg.get("proposal_residual_grow_vox", 1)),
            sigma_mm=float(cfg.get("proposal_residual_sigma_mm", 4.0)),
            blur_threshold=float(cfg.get("proposal_residual_blur_threshold", 0.2)),
        )
        member_masks = [
            ("depth_mask", base_depth_mask),
            ("gaussian_residual", gaussian_mask),
        ]
        combined_mask_kji = np.zeros_like(base_depth_mask, dtype=bool)
        merged_blob_samples: dict[int, dict[str, Any]] = {}
        merged_tokens: list[_SupportToken] = []
        next_blob_id = 1
        next_token_id = 1
        merged_blob_count = 0
        merged_token_summary: dict[str, int] = {
            "token_count_total": 0,
            "token_count_compact": 0,
            "token_count_elongated": 0,
            "token_count_complex": 0,
            "blob_count_compact": 0,
            "blob_count_elongated": 0,
            "blob_count_complex": 0,
        }
        mode_counts: list[dict[str, Any]] = []
        for member_mode, member_mask in member_masks:
            member_mask = np.asarray(member_mask, dtype=bool)
            if int(np.count_nonzero(member_mask)) <= 0:
                mode_counts.append({"mode": str(member_mode), "candidate_points_after_mask": 0, "blob_count_total": 0, "token_count_total": 0})
                continue
            combined_mask_kji |= member_mask
            member_blob_result = extract_blob_candidates(
                member_mask,
                arr_kji=arr_kji,
                depth_map_kji=chosen_preview.get("head_distance_map_kji"),
                ijk_kji_to_ras_fn=ctx["ijk_kji_to_ras_fn"],
                fully_connected=True,
            )
            member_blob_samples = self._blob_samples(member_blob_result, ctx, arr_kji)
            member_tokens, member_token_summary = self._tokens_from_blobs(member_blob_result, member_blob_samples, cfg)
            blob_id_map: dict[int, int] = {}
            for old_blob_id in sorted(member_blob_samples.keys()):
                sample = dict(member_blob_samples[old_blob_id])
                new_blob_id = next_blob_id
                next_blob_id += 1
                blob_id_map[int(old_blob_id)] = int(new_blob_id)
                blob = dict(sample.get("blob") or {})
                blob["blob_id"] = int(new_blob_id)
                blob["proposal_mode"] = str(member_mode)
                sample["blob"] = blob
                sample["proposal_mode"] = str(member_mode)
                merged_blob_samples[int(new_blob_id)] = sample
            for token in member_tokens:
                new_token_id = next_token_id
                next_token_id += 1
                new_blob_id = int(blob_id_map.get(int(token.blob_id), int(token.blob_id)))
                meta = dict(token.meta or {})
                meta["proposal_mode"] = str(member_mode)
                merged_tokens.append(
                    _SupportToken(
                        token_id=int(new_token_id),
                        point_ras=tuple(float(v) for v in token.point_ras),
                        weight=float(token.weight),
                        purity=float(token.purity),
                        blob_id=int(new_blob_id),
                        blob_kind=str(token.blob_kind),
                        local_axis_ras=tuple(float(v) for v in token.local_axis_ras),
                        depth_mm=float(token.depth_mm),
                        seedable=bool(token.seedable),
                        meta=meta,
                    )
                )
            merged_blob_count += int(member_blob_result.get("blob_count_total") or 0)
            for key, value in dict(member_token_summary or {}).items():
                merged_token_summary[key] = int(merged_token_summary.get(key, 0)) + int(value)
            mode_counts.append(
                {
                    "mode": str(member_mode),
                    "candidate_points_after_mask": int(np.count_nonzero(member_mask)),
                    "blob_count_total": int(member_blob_result.get("blob_count_total") or 0),
                    "token_count_total": int(member_token_summary.get("token_count_total") or 0),
                }
            )
        ijk_kji = np.argwhere(combined_mask_kji)
        points_ras = self._ijk_kji_to_ras(ctx, ijk_kji.astype(float)) if ijk_kji.size else np.zeros((0, 3), dtype=float)
        points_lps = self._ras_to_lps(points_ras)
        depth_map_kji = chosen_preview.get("head_distance_map_kji")
        candidate_depth_mm = None
        if depth_map_kji is not None and ijk_kji.size:
            candidate_depth_mm = np.asarray(depth_map_kji[ijk_kji[:, 0], ijk_kji[:, 1], ijk_kji[:, 2]], dtype=float).reshape(-1)
        return {
            "pool_name": "proposal",
            "support_mode": "union_depth_and_gaussian",
            "threshold_hu": float(chosen_threshold),
            "preview_stats": preview_stats,
            "preview": chosen_preview,
            "candidate_mask_kji": combined_mask_kji,
            "candidate_points_after_mask": int(np.count_nonzero(combined_mask_kji)),
            "candidate_points_ras": points_ras,
            "candidate_points_lps": points_lps,
            "candidate_depth_mm": candidate_depth_mm,
            "blob_result": {"blob_count_total": int(merged_blob_count)},
            "blob_samples": merged_blob_samples,
            "tokens": merged_tokens,
            "token_summary": merged_token_summary,
            "ras_to_ijk_fn": ctx.get("ras_to_ijk_fn"),
            "support_mode_counts": mode_counts,
        }

    @staticmethod
    def _simple_preview_masks(arr_kji: np.ndarray, threshold: float) -> dict[str, Any]:
        candidate_mask_kji = np.asarray(arr_kji >= float(threshold), dtype=bool)
        count = int(np.count_nonzero(candidate_mask_kji))
        ones_mask = np.ones_like(candidate_mask_kji, dtype=bool)
        return {
            "metal_in_head_mask_kji": candidate_mask_kji,
            "metal_depth_pass_mask_kji": candidate_mask_kji,
            "head_mask_kji": ones_mask,
            "gating_mask_kji": ones_mask,
            "head_distance_map_kji": None,
            "candidate_count": count,
            "metal_in_head_count": count,
            "depth_kept_count": count,
        }

    @staticmethod
    def _proposal_gaussian_residual_mask(
        arr_kji: np.ndarray,
        gate_mask_kji: np.ndarray,
        spacing_xyz: tuple[float, float, float],
        threshold_hu: float,
        grow_vox: int,
        sigma_mm: float,
        blur_threshold: float,
    ) -> np.ndarray:
        base_mask = np.asarray(arr_kji >= float(threshold_hu), dtype=bool)
        if int(np.count_nonzero(base_mask)) <= 0:
            return np.zeros_like(base_mask, dtype=bool)
        if sitk is None:
            return np.logical_and(base_mask, np.asarray(gate_mask_kji, dtype=bool))
        img = sitk.GetImageFromArray(base_mask.astype(np.uint8))
        if int(grow_vox) > 0:
            img = sitk.BinaryDilate(img, [int(grow_vox)] * 3, sitk.sitkBall)
        smooth = sitk.SmoothingRecursiveGaussian(sitk.Cast(img, sitk.sitkFloat32), sigma=float(max(sigma_mm, 0.0)))
        opened_like = sitk.GetArrayFromImage(smooth) >= float(blur_threshold)
        residual = np.logical_and(sitk.GetArrayFromImage(img).astype(bool), np.logical_not(np.asarray(opened_like, dtype=bool)))
        return np.logical_and(residual, np.asarray(gate_mask_kji, dtype=bool))

    @staticmethod
    def _proposal_objectness_mask(
        *,
        arr_kji: np.ndarray,
        candidate_mask_kji: np.ndarray,
        head_mask_kji: np.ndarray,
        spacing_xyz: tuple[float, float, float],
        gaussian_sigma_mm: float,
        alpha: float,
        beta: float,
        gamma: float,
        scale_measure: bool,
        bright_object: bool,
        object_dimension: int,
        response_percentiles: list[float],
        target_points: int,
        min_points: int,
        max_points: int,
        min_fraction_of_max: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        source_mask = np.asarray(candidate_mask_kji, dtype=bool)
        if int(np.count_nonzero(source_mask)) <= 0:
            return np.zeros_like(source_mask, dtype=bool), {"reason": "empty_source_mask"}
        if sitk is None:
            return source_mask, {"reason": "sitk_unavailable"}
        head_mask = np.asarray(head_mask_kji, dtype=bool)
        roi_mask = head_mask if int(np.count_nonzero(head_mask)) > 0 else source_mask
        roi_idx = np.argwhere(roi_mask)
        if roi_idx.size == 0:
            return np.zeros_like(source_mask, dtype=bool), {"reason": "empty_roi_mask"}
        sx, sy, sz = [max(float(v), 1e-6) for v in (spacing_xyz or (1.0, 1.0, 1.0))]
        margin_mm = max(2.0, 3.0 * float(max(gaussian_sigma_mm, 0.0)))
        margin_k = int(math.ceil(margin_mm / sz))
        margin_j = int(math.ceil(margin_mm / sy))
        margin_i = int(math.ceil(margin_mm / sx))
        k0 = max(0, int(np.min(roi_idx[:, 0])) - margin_k)
        k1 = min(int(source_mask.shape[0]), int(np.max(roi_idx[:, 0])) + margin_k + 1)
        j0 = max(0, int(np.min(roi_idx[:, 1])) - margin_j)
        j1 = min(int(source_mask.shape[1]), int(np.max(roi_idx[:, 1])) + margin_j + 1)
        i0 = max(0, int(np.min(roi_idx[:, 2])) - margin_i)
        i1 = min(int(source_mask.shape[2]), int(np.max(roi_idx[:, 2])) + margin_i + 1)
        slices = (slice(k0, k1), slice(j0, j1), slice(i0, i1))
        arr_crop = np.asarray(arr_kji[slices], dtype=np.float32)
        head_crop = np.asarray(head_mask[slices], dtype=bool)
        source_crop = np.asarray(source_mask[slices], dtype=bool)
        inside_vals = arr_crop[head_crop] if int(np.count_nonzero(head_crop)) > 0 else arr_crop.reshape(-1)
        fill_value = float(np.percentile(inside_vals, 5.0)) if inside_vals.size else float(np.min(arr_crop))
        work_crop = np.where(head_crop, arr_crop, fill_value).astype(np.float32, copy=False)
        img = sitk.GetImageFromArray(work_crop)
        img.SetSpacing(tuple(float(v) for v in spacing_xyz))
        sigma = float(max(gaussian_sigma_mm, 0.0))
        if sigma > 0.0:
            img = sitk.SmoothingRecursiveGaussian(img, sigma=sigma)
        filt = sitk.ObjectnessMeasureImageFilter()
        filt.SetAlpha(float(alpha))
        filt.SetBeta(float(beta))
        filt.SetGamma(float(gamma))
        filt.SetScaleObjectnessMeasure(bool(scale_measure))
        filt.SetBrightObject(bool(bright_object))
        filt.SetObjectDimension(int(max(0, object_dimension)))
        response_crop = sitk.GetArrayFromImage(filt.Execute(sitk.Cast(img, sitk.sitkFloat32))).astype(np.float32)
        response_vals = response_crop[source_crop]
        response_vals = response_vals[np.isfinite(response_vals) & (response_vals > 0.0)]
        if response_vals.size <= 0:
            return np.zeros_like(source_mask, dtype=bool), {"reason": "no_positive_response"}
        percentiles = [float(np.clip(v, 0.0, 100.0)) for v in list(response_percentiles or [99.8, 99.5, 99.0, 98.5, 98.0, 97.5, 97.0, 96.0, 95.0, 92.5, 90.0])]
        if not percentiles:
            percentiles = [90.0]
        target_points = max(1, int(target_points))
        min_points = max(1, int(min_points))
        max_points = max(min_points, int(max_points))
        max_thr = float(max(0.0, min_fraction_of_max)) * float(np.max(response_vals))
        threshold = max_thr
        best_keep_crop: np.ndarray | None = None
        best_stats: dict[str, Any] | None = None
        for percentile in percentiles:
            percentile_thr = float(np.percentile(response_vals, percentile))
            trial_threshold = max(percentile_thr, max_thr)
            trial_keep_crop = np.logical_and(source_crop, np.isfinite(response_crop) & (response_crop >= float(trial_threshold)))
            kept_count = int(np.count_nonzero(trial_keep_crop))
            stats = {
                "percentile": float(percentile),
                "threshold": float(trial_threshold),
                "kept_count": int(kept_count),
            }
            score = abs(int(kept_count) - int(target_points))
            in_band = int(min_points) <= int(kept_count) <= int(max_points)
            enough = int(kept_count) >= int(min_points)
            if best_stats is None:
                best_keep_crop = trial_keep_crop
                best_stats = dict(stats, score=float(score), in_band=bool(in_band), enough=bool(enough))
                threshold = float(trial_threshold)
                continue
            best_in_band = bool(best_stats.get("in_band", False))
            best_enough = bool(best_stats.get("enough", False))
            take = False
            if in_band and not best_in_band:
                take = True
            elif in_band == best_in_band:
                if enough and not best_enough:
                    take = True
                elif enough == best_enough:
                    best_score = float(best_stats.get("score", float("inf")))
                    if float(score) < best_score - 1e-6:
                        take = True
                    elif abs(float(score) - best_score) <= 1e-6 and float(percentile) > float(best_stats.get("percentile", -1.0)):
                        take = True
            if take:
                best_keep_crop = trial_keep_crop
                best_stats = dict(stats, score=float(score), in_band=bool(in_band), enough=bool(enough))
                threshold = float(trial_threshold)
        if best_keep_crop is None or best_stats is None:
            return np.zeros_like(source_mask, dtype=bool), {"reason": "threshold_selection_failed"}
        keep_crop = np.logical_and(source_crop, np.isfinite(response_crop) & (response_crop >= float(threshold)))
        out = np.zeros_like(source_mask, dtype=bool)
        out[slices] = keep_crop
        return out, {
            "reason": "ok",
            "chosen_percentile": float(best_stats.get("percentile", percentiles[-1])),
            "chosen_threshold": float(threshold),
            "kept_count": int(np.count_nonzero(keep_crop)),
            "target_points": int(target_points),
            "min_points": int(min_points),
            "max_points": int(max_points),
            "response_max": float(np.max(response_vals)),
            "response_min_positive": float(np.min(response_vals)),
        }

    @staticmethod
    def _kji_to_phys_xyz(ijk_kji: np.ndarray, spacing_xyz: tuple[float, float, float]) -> np.ndarray:
        pts = np.asarray(ijk_kji, dtype=float).reshape(-1, 3)
        sx, sy, sz = [float(v) for v in (spacing_xyz or (1.0, 1.0, 1.0))]
        xyz = np.empty((pts.shape[0], 3), dtype=float)
        xyz[:, 0] = pts[:, 2] * sx
        xyz[:, 1] = pts[:, 1] * sy
        xyz[:, 2] = pts[:, 0] * sz
        return xyz

    def _proposal_ellipsoid_core_mask(
        self,
        *,
        head_mask_kji: np.ndarray,
        candidate_mask_kji: np.ndarray,
        spacing_xyz: tuple[float, float, float],
        shrink_factor: float,
        max_fit_points: int,
        seed: int,
    ) -> np.ndarray:
        head_mask = np.asarray(head_mask_kji, dtype=bool)
        candidate_mask = np.asarray(candidate_mask_kji, dtype=bool)
        if int(np.count_nonzero(head_mask)) <= 8 or int(np.count_nonzero(candidate_mask)) <= 0:
            return np.zeros_like(candidate_mask, dtype=bool)
        head_idx = np.argwhere(head_mask)
        if head_idx.shape[0] > int(max_fit_points):
            rng = np.random.default_rng(int(seed))
            choose = rng.choice(head_idx.shape[0], size=int(max_fit_points), replace=False)
            head_idx = head_idx[np.sort(choose)]
        head_xyz = self._kji_to_phys_xyz(head_idx, spacing_xyz)
        center = np.mean(head_xyz, axis=0)
        centered = head_xyz - center.reshape(1, 3)
        cov = np.cov(centered, rowvar=False)
        evals, evecs = np.linalg.eigh(np.asarray(cov, dtype=float))
        order = np.argsort(evals)[::-1]
        evals = np.maximum(evals[order], 1e-6)
        evecs = evecs[:, order]
        # For a filled ellipsoid, variance along each principal axis is a^2/5.
        radii = np.sqrt(5.0 * evals) * float(max(0.1, shrink_factor))
        cand_idx = np.argwhere(candidate_mask)
        cand_xyz = self._kji_to_phys_xyz(cand_idx, spacing_xyz)
        proj = (cand_xyz - center.reshape(1, 3)) @ evecs
        ellipsoid_val = np.sum((proj / radii.reshape(1, 3)) ** 2, axis=1)
        keep = ellipsoid_val <= 1.0
        out = np.zeros_like(candidate_mask, dtype=bool)
        if int(np.count_nonzero(keep)) > 0:
            kept_idx = cand_idx[keep]
            out[kept_idx[:, 0], kept_idx[:, 1], kept_idx[:, 2]] = True
        return out

    def _blob_samples(self, blob_result: dict[str, Any], ctx: DetectionContext, arr_kji: np.ndarray) -> dict[int, dict[str, Any]]:
        labels = np.asarray(blob_result.get("labels_kji"), dtype=np.int32)
        coords = np.argwhere(labels > 0)
        if coords.size == 0:
            return {}
        label_vals = labels[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.int32)
        order = np.argsort(label_vals, kind="mergesort")
        coords = coords[order]
        label_vals = label_vals[order]
        ras_pts = self._ijk_kji_to_ras(ctx, coords.astype(float))
        blobs_by_id = {int(blob.get("blob_id")): dict(blob) for blob in list(blob_result.get("blobs") or [])}
        out: dict[int, dict[str, Any]] = {}
        start = 0
        while start < coords.shape[0]:
            blob_id = int(label_vals[start])
            end = start + 1
            while end < coords.shape[0] and int(label_vals[end]) == blob_id:
                end += 1
            blob_coords = coords[start:end]
            blob_ras = ras_pts[start:end]
            blob = dict(blobs_by_id.get(blob_id) or {})
            out[blob_id] = {
                "blob": blob,
                "coords_kji": blob_coords,
                "points_ras": blob_ras,
                "points_lps": self._ras_to_lps(blob_ras),
                "voxel_count": int(blob_coords.shape[0]),
            }
            start = end
        return out

    def _tokens_from_blobs(self, blob_result: dict[str, Any], blob_samples: dict[int, dict[str, Any]], cfg: dict[str, Any]) -> tuple[list[_SupportToken], dict[str, Any]]:
        tokens: list[_SupportToken] = []
        compact = elongated = complex_count = 0
        next_token_id = 1
        spacing = float(cfg.get("token_spacing_mm", 3.0))
        max_tokens_per_blob = int(cfg.get("max_tokens_per_blob", 10))
        max_complex_tokens = int(cfg.get("max_complex_tokens_per_blob", 4))
        for blob in list(blob_result.get("blobs") or []):
            blob_id = int(blob.get("blob_id") or 0)
            sample = blob_samples.get(blob_id)
            if sample is None:
                continue
            points_ras = np.asarray(sample.get("points_ras"), dtype=float).reshape(-1, 3)
            if points_ras.shape[0] == 0:
                continue
            axis_val = blob.get("pca_axis_ras")
            axis = self._normalize(np.asarray(axis_val if axis_val is not None else [0.0, 0.0, 1.0], dtype=float))
            centroid_val = blob.get("centroid_ras")
            centroid = np.asarray(centroid_val if centroid_val is not None else np.mean(points_ras, axis=0), dtype=float).reshape(3)
            length_mm = float(blob.get("length_mm") or 0.0)
            diameter_mm = float(blob.get("diameter_mm") or 0.0)
            elongation = float(blob.get("elongation") or 1.0)
            depth_mm = float(blob.get("depth_mean") or blob.get("depth_max") or 0.0)
            base_weight = max(1.0, float(np.sqrt(max(1.0, float(sample.get("voxel_count") or 1)))))
            compact_like = length_mm <= float(cfg.get("compact_max_length_mm", 4.5)) and diameter_mm <= float(cfg.get("compact_max_diameter_mm", 3.8)) and elongation <= float(cfg.get("compact_max_elongation", 4.0))
            elongated_like = length_mm >= float(cfg.get("elongated_min_length_mm", 6.0)) and elongation >= float(cfg.get("elongated_min_elongation", 4.5))
            if compact_like:
                compact += 1
                tokens.append(
                    _SupportToken(
                        token_id=next_token_id,
                        point_ras=tuple(float(v) for v in centroid),
                        weight=float(base_weight),
                        purity=1.0,
                        blob_id=blob_id,
                        blob_kind="compact",
                        local_axis_ras=tuple(float(v) for v in axis),
                        depth_mm=float(depth_mm),
                        seedable=True,
                        meta={"voxel_count": int(sample.get("voxel_count") or 0)},
                    )
                )
                next_token_id += 1
                continue

            projections = (points_ras - centroid.reshape(1, 3)) @ axis.reshape(3)
            t_min = float(np.min(projections))
            t_max = float(np.max(projections))
            span = max(0.0, t_max - t_min)
            if elongated_like:
                elongated += 1
                token_count = int(min(max_tokens_per_blob, max(3, int(math.ceil(span / max(spacing, 1e-3))) + 1)))
                centers = np.linspace(t_min, t_max, token_count)
                seed_positions = {0, token_count // 2, token_count - 1}
                for idx, center_t in enumerate(centers):
                    mask = np.abs(projections - center_t) <= max(spacing * 0.5, 1.5)
                    if int(np.count_nonzero(mask)) > 0:
                        point = np.mean(points_ras[mask], axis=0)
                    else:
                        point = centroid + axis * center_t
                    tokens.append(
                        _SupportToken(
                            token_id=next_token_id,
                            point_ras=tuple(float(v) for v in point),
                            weight=float(base_weight / float(token_count)),
                            purity=1.0,
                            blob_id=blob_id,
                            blob_kind="elongated",
                            local_axis_ras=tuple(float(v) for v in axis),
                            depth_mm=float(depth_mm),
                            seedable=idx in seed_positions,
                            meta={"span_mm": float(span), "center_t": float(center_t)},
                        )
                    )
                    next_token_id += 1
                continue

            complex_count += 1
            token_count = int(min(max_complex_tokens, max(1, int(math.ceil(span / max(spacing * 1.5, 1e-3))))))
            if token_count <= 1:
                complex_centers = [0.5 * (t_min + t_max)]
            else:
                complex_centers = np.linspace(t_min, t_max, token_count)
            for idx, center_t in enumerate(complex_centers):
                mask = np.abs(projections - center_t) <= max(spacing * 0.75, 2.0)
                if int(np.count_nonzero(mask)) > 0:
                    point = np.mean(points_ras[mask], axis=0)
                else:
                    point = centroid + axis * center_t
                tokens.append(
                    _SupportToken(
                        token_id=next_token_id,
                        point_ras=tuple(float(v) for v in point),
                        weight=float(0.4 * base_weight / float(max(1, token_count))),
                        purity=0.45,
                        blob_id=blob_id,
                        blob_kind="complex",
                        local_axis_ras=tuple(float(v) for v in axis),
                        depth_mm=float(depth_mm),
                        seedable=idx == (token_count // 2),
                        meta={"span_mm": float(span), "center_t": float(center_t)},
                    )
                )
                next_token_id += 1
        summary = {
            "token_count_total": int(len(tokens)),
            "token_count_compact": int(compact),
            "token_count_elongated": int(sum(1 for t in tokens if t.blob_kind == "elongated")),
            "token_count_complex": int(sum(1 for t in tokens if t.blob_kind == "complex")),
            "blob_count_compact": int(compact),
            "blob_count_elongated": int(elongated),
            "blob_count_complex": int(complex_count),
        }
        return tokens, summary

    def _generate_hypotheses(self, ctx: DetectionContext, support: dict[str, Any], cfg: dict[str, Any]) -> list[_Hypothesis]:
        tokens = list(support["tokens"])
        if not tokens:
            return []
        seeds: list[dict[str, Any]] = []
        seen_seed_keys: set[tuple[Any, ...]] = set()
        next_seed_id = 1

        def add_seed(seed_kind: str, point: np.ndarray, direction: np.ndarray, blob_ids: Iterable[int], seed_score: float) -> None:
            nonlocal next_seed_id
            direction = self._normalize(direction)
            key = self._seed_key(point, direction)
            if key in seen_seed_keys:
                return
            seen_seed_keys.add(key)
            seeds.append(
                {
                    "seed_id": int(next_seed_id),
                    "seed_kind": str(seed_kind),
                    "point_ras": point.astype(float),
                    "direction_ras": direction.astype(float),
                    "seed_blob_ids": tuple(sorted({int(v) for v in blob_ids if int(v) > 0})),
                    "seed_score": float(seed_score),
                }
            )
            next_seed_id += 1

        for blob_id, sample in support["blob_samples"].items():
            blob = dict(sample.get("blob") or {})
            length_mm = float(blob.get("length_mm") or 0.0)
            elongation = float(blob.get("elongation") or 1.0)
            if length_mm >= float(cfg.get("elongated_min_length_mm", 6.0)) and elongation >= float(cfg.get("elongated_min_elongation", 4.5)):
                add_seed(
                    "elongated_blob",
                    np.asarray(blob.get("centroid_ras") if blob.get("centroid_ras") is not None else np.mean(np.asarray(sample.get("points_ras"), dtype=float), axis=0), dtype=float).reshape(3),
                    np.asarray(blob.get("pca_axis_ras") if blob.get("pca_axis_ras") is not None else [0.0, 0.0, 1.0], dtype=float),
                    [blob_id],
                    float(np.sqrt(max(1.0, float(sample.get("voxel_count") or 1)))) + 0.1 * float(length_mm),
                )

        seeds = []
        seen_seed_keys.clear()
        next_seed_id = 1
        for vote_seed in self._hough_vote_seeds(support, cfg):
            add_seed(
                str(vote_seed["seed_kind"]),
                np.asarray(vote_seed["point_ras"], dtype=float).reshape(3),
                np.asarray(vote_seed["direction_ras"], dtype=float).reshape(3),
                list(vote_seed.get("seed_blob_ids") or []),
                float(vote_seed.get("seed_score") or 0.0),
            )
        seeds = sorted(seeds, key=lambda seed: float(seed["seed_score"]), reverse=True)[: int(cfg.get("max_seed_count", 400))]
        hypotheses: list[_Hypothesis] = []
        next_hyp_id = 1
        for seed in seeds:
            hyp = self._hypothesis_from_seed(next_hyp_id, seed, support, cfg)
            if hyp is None:
                continue
            hypotheses.append(hyp)
            next_hyp_id += 1
        return sorted(hypotheses, key=lambda h: float(h.local_score), reverse=True)

    def _hough_vote_seeds(self, support: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
        points = np.asarray(support.get("candidate_points_ras"), dtype=float).reshape(-1, 3)
        if points.shape[0] < 2:
            return []
        vote_min_depth = float(cfg.get("hough_vote_min_depth_mm", 8.0))
        vote_max_depth = float(cfg.get("hough_vote_max_depth_mm", 60.0))
        depths = np.asarray(support.get("candidate_depth_mm"), dtype=float).reshape(-1) if support.get("candidate_depth_mm") is not None else np.asarray([], dtype=float)
        if depths.size == points.shape[0]:
            depth_mask = np.isfinite(depths) & (depths >= vote_min_depth) & (depths <= vote_max_depth)
            if int(np.count_nonzero(depth_mask)) >= 2:
                points = points[depth_mask]
        max_points = int(cfg.get("hough_max_vote_points", 100000))
        rng = np.random.default_rng(int(cfg.get("hough_random_seed", 0)))
        if points.shape[0] > max_points:
            choose = rng.choice(points.shape[0], size=max_points, replace=False)
            points = points[np.sort(choose)]
        if points.shape[0] < 2:
            return []
        min_pair = float(cfg.get("hough_vote_min_pair_distance_mm", 10.0))
        max_pair = float(cfg.get("hough_vote_max_pair_distance_mm", 90.0))
        dir_bin_deg = max(float(cfg.get("hough_direction_bin_deg", 6.0)), 1.0)
        offset_bin_mm = max(float(cfg.get("hough_offset_bin_mm", 4.0)), 0.5)
        pair_samples = int(cfg.get("hough_pair_samples", 160000))
        n_points = int(points.shape[0])
        if n_points < 2:
            return []
        token_points = np.asarray([t.point_ras for t in list(support.get("tokens") or [])], dtype=float).reshape(-1, 3)
        token_weights = np.asarray([float(t.weight) * float(t.purity) for t in list(support.get("tokens") or [])], dtype=float).reshape(-1)

        direction_bins: dict[tuple[Any, ...], dict[str, Any]] = {}
        batch_size = 40000
        remaining = pair_samples
        while remaining > 0:
            batch = min(batch_size, remaining)
            i = rng.integers(0, n_points, size=batch, endpoint=False)
            j = rng.integers(0, n_points, size=batch, endpoint=False)
            valid = i != j
            if not np.any(valid):
                remaining -= batch
                continue
            i = i[valid]
            j = j[valid]
            p0 = points[i]
            p1 = points[j]
            diff = p1 - p0
            dist = np.linalg.norm(diff, axis=1)
            valid = (dist >= min_pair) & (dist <= max_pair)
            if not np.any(valid):
                remaining -= batch
                continue
            p0 = p0[valid]
            p1 = p1[valid]
            diff = diff[valid]
            dist = dist[valid]
            dirs = diff / np.maximum(dist.reshape(-1, 1), 1e-6)
            dirs = np.asarray([self._canonical_direction(d) for d in dirs], dtype=float).reshape(-1, 3)
            votes = 0.5 + 0.5 * np.minimum(dist / max_pair, 1.0)
            for pair_idx in range(dirs.shape[0]):
                key = self._direction_key(dirs[pair_idx], dir_bin_deg=dir_bin_deg)
                bucket = direction_bins.setdefault(
                    key,
                    {
                        "vote": 0.0,
                        "pair_count": 0,
                        "dir_sum": np.zeros((3,), dtype=float),
                    },
                )
                bucket["vote"] += float(votes[pair_idx])
                bucket["pair_count"] += 1
                bucket["dir_sum"] += float(votes[pair_idx]) * dirs[pair_idx]
            remaining -= batch

        if not direction_bins:
            return []
        min_pairs = int(cfg.get("hough_min_bin_pair_count", 2))
        max_dirs = int(cfg.get("hough_top_directions", 24))
        max_seeds = int(cfg.get("hough_max_seed_count", 180))
        max_offset_bins = int(cfg.get("hough_max_offset_bins_per_direction", 8))
        min_offset_points = int(cfg.get("hough_min_offset_points", 8))
        seeds: list[dict[str, Any]] = []
        direction_peaks: list[np.ndarray] = []
        for bucket in sorted(direction_bins.values(), key=lambda item: (float(item["vote"]), int(item["pair_count"])), reverse=True)[:max_dirs]:
            if int(bucket["pair_count"]) < min_pairs or float(bucket["vote"]) <= 0.0:
                continue
            direction = self._canonical_direction(bucket["dir_sum"])
            duplicate = any(self._angle_deg(direction, prev) <= float(cfg.get("hough_direction_bin_deg", 6.0)) for prev in direction_peaks)
            if duplicate:
                continue
            direction_peaks.append(direction)
            basis_u, basis_v = self._orthonormal_basis(direction)
            if token_points.size == 0:
                continue
            offsets = token_points - np.outer(token_points @ direction.reshape(3), direction.reshape(3))
            c0 = offsets @ basis_u.reshape(3)
            c1 = offsets @ basis_v.reshape(3)
            offset_bins: dict[tuple[int, int], dict[str, Any]] = {}
            for idx in range(offsets.shape[0]):
                key = (
                    int(round(float(c0[idx]) / offset_bin_mm)),
                    int(round(float(c1[idx]) / offset_bin_mm)),
                )
                bucket2 = offset_bins.setdefault(
                    key,
                    {
                        "vote": 0.0,
                        "point_count": 0,
                        "offset_sum": np.zeros((3,), dtype=float),
                    },
                )
                bucket2["vote"] += float(token_weights[idx]) if token_weights.size == offsets.shape[0] else 1.0
                bucket2["point_count"] += 1
                bucket2["offset_sum"] += (float(token_weights[idx]) if token_weights.size == offsets.shape[0] else 1.0) * offsets[idx]
            for bucket2 in sorted(offset_bins.values(), key=lambda item: (float(item["vote"]), int(item["point_count"])), reverse=True)[:max_offset_bins]:
                if int(bucket2["point_count"]) < min_offset_points or float(bucket2["vote"]) <= 0.0:
                    continue
                point = np.asarray(bucket2["offset_sum"], dtype=float) / float(bucket2["point_count"])
                seeds.append(
                    {
                        "seed_kind": "hough_voxel_vote",
                        "point_ras": point.astype(float),
                        "direction_ras": direction.astype(float),
                        "seed_blob_ids": (),
                        "seed_score": float(float(bucket["vote"]) + 0.1 * float(bucket2["point_count"])),
                    }
                )
                if len(seeds) >= max_seeds:
                    return seeds
        return seeds

    def _hypothesis_from_seed(self, hyp_id: int, seed: dict[str, Any], support: dict[str, Any], cfg: dict[str, Any]) -> _Hypothesis | None:
        tokens = list(support["tokens"])
        if not tokens:
            return None
        points = np.asarray([t.point_ras for t in tokens], dtype=float).reshape(-1, 3)
        weights = np.asarray([float(t.weight) for t in tokens], dtype=float)
        purity = np.asarray([float(t.purity) for t in tokens], dtype=float)
        local_axes = np.asarray([t.local_axis_ras for t in tokens], dtype=float).reshape(-1, 3)
        line_point = np.asarray(seed["point_ras"], dtype=float).reshape(3)
        line_axis = self._normalize(np.asarray(seed["direction_ras"], dtype=float).reshape(3))
        d0 = self._line_distances(points, line_point, line_axis)
        axis_align = np.clip(np.abs(np.sum(local_axes * line_axis.reshape(1, 3), axis=1)), 0.0, 1.0)
        token_is_compact = np.asarray([t.blob_kind == "compact" for t in tokens], dtype=bool)
        initial_corridor_radius = float(cfg.get("seed_corridor_radius_mm", 8.0))
        compatible = (d0 <= initial_corridor_radius) & ((axis_align >= 0.55) | token_is_compact)
        if int(np.count_nonzero(compatible)) < int(cfg.get("min_support_tokens", 4)):
            return None
        seed_weights = weights[compatible] * purity[compatible] * np.exp(-0.5 * (d0[compatible] / max(initial_corridor_radius, 1e-3)) ** 2)
        fit_center, fit_axis = self._weighted_pca_line(points[compatible], seed_weights)
        d1 = self._line_distances(points, fit_center, fit_axis)
        axis_align = np.clip(np.abs(np.sum(local_axes * fit_axis.reshape(1, 3), axis=1)), 0.0, 1.0)
        compatible = (d1 <= float(cfg.get("corridor_radius_mm", 2.2))) & ((axis_align >= 0.55) | token_is_compact)
        if int(np.count_nonzero(compatible)) < int(cfg.get("min_support_tokens", 4)):
            return None
        projections = (points[compatible] - fit_center.reshape(1, 3)) @ fit_axis.reshape(3)
        compatible_tokens = [tokens[idx] for idx in np.where(compatible)[0].tolist()]
        compatible_dist = d1[compatible]
        run = self._best_support_run(
            fit_center,
            fit_axis,
            compatible_tokens,
            projections,
            compatible_dist,
            cfg,
            support.get("preview", {}),
            support.get("candidate_mask_kji"),
            support.get("preview", {}).get("head_distance_map_kji"),
            support.get("preview", {}).get("gating_mask_kji") if support.get("preview", {}).get("gating_mask_kji") is not None else support.get("preview", {}).get("head_mask_kji"),
            support.get("preview", {}).get("head_distance_map_kji"),
            support,
        )
        if run is None:
            return None
        start, end = self._segment_from_run(fit_center, fit_axis, run)
        start, end = self._orient_shallow_to_deep(start, end, support.get("preview", {}).get("head_distance_map_kji"), support.get("preview", {}).get("head_distance_map_kji"), support)
        axis = self._normalize(end - start)
        return _Hypothesis(
            hyp_id=int(hyp_id),
            seed_kind=str(seed.get("seed_kind") or "seed"),
            seed_blob_ids=tuple(int(v) for v in list(seed.get("seed_blob_ids") or [])),
            point_ras=tuple(float(v) for v in fit_center),
            direction_ras=tuple(float(v) for v in axis),
            start_ras=tuple(float(v) for v in start),
            end_ras=tuple(float(v) for v in end),
            owned_token_ids=tuple(int(v) for v in run["token_ids"]),
            token_blob_ids=tuple(int(v) for v in run["blob_ids"]),
            support_mass=float(run["support_mass"]),
            covered_length_mm=float(run["covered_length_mm"]),
            depth_span_mm=float(run["depth_span_mm"]),
            internal_gap_mm=float(run["internal_gap_mm"]),
            width_mm=float(run["width_mm"]),
            in_head_fraction=float(run["in_head_fraction"]),
            superficial_penalty=float(run["superficial_penalty"]),
            degeneracy_penalty=float(run["degeneracy_penalty"]),
            local_score=float(run["score"]),
            extras={
                "seed_score": float(seed.get("seed_score") or 0.0),
                "t_min_mm": float(run["t_min_mm"]),
                "t_max_mm": float(run["t_max_mm"]),
                "token_count": int(run["token_count"]),
                "unique_blob_count": int(run["unique_blob_count"]),
            },
        )

    def _best_support_run(
        self,
        center: np.ndarray,
        axis: np.ndarray,
        tokens: list[_SupportToken],
        projections: np.ndarray,
        distances: np.ndarray,
        cfg: dict[str, Any],
        preview: dict[str, Any],
        candidate_mask_kji: np.ndarray | None,
        depth_map_kji: np.ndarray | None,
        gating_mask_kji: np.ndarray | None,
        _unused_depth_map_again: np.ndarray | None,
        support: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not tokens:
            return None
        order = np.argsort(projections)
        sorted_tokens = [tokens[int(i)] for i in order.tolist()]
        sorted_proj = projections[order]
        sorted_dist = distances[order]
        gap_break = float(cfg.get("projection_gap_mm", 6.0))
        runs: list[tuple[int, int]] = []
        start = 0
        for idx in range(1, len(sorted_tokens)):
            if float(sorted_proj[idx] - sorted_proj[idx - 1]) > gap_break:
                runs.append((start, idx))
                start = idx
        runs.append((start, len(sorted_tokens)))
        best: dict[str, Any] | None = None
        expected_spacing = float(cfg.get("expected_internal_spacing_mm", 3.0))
        min_unique_blobs = int(cfg.get("min_unique_blobs", 3))
        for lo, hi in runs:
            run_tokens = sorted_tokens[lo:hi]
            run_proj = sorted_proj[lo:hi]
            run_dist = sorted_dist[lo:hi]
            if len(run_tokens) < int(cfg.get("min_support_tokens", 4)):
                continue
            token_weights = np.asarray([float(t.weight) * float(t.purity) for t in run_tokens], dtype=float)
            support_mass = float(np.sum(token_weights * np.exp(-0.5 * (run_dist / max(float(cfg.get("corridor_radius_mm", 2.2)), 1e-3)) ** 2)))
            t_min = float(run_proj[0])
            t_max = float(run_proj[-1])
            covered_length = max(0.0, t_max - t_min)
            gaps = np.diff(run_proj) if len(run_proj) >= 2 else np.zeros((0,), dtype=float)
            internal_gap = float(np.sum(np.maximum(0.0, gaps - expected_spacing))) if gaps.size else 0.0
            width_mm = float(np.sqrt(np.average(run_dist ** 2, weights=np.maximum(token_weights, 1e-6)))) if run_dist.size else 0.0
            depth_vals = np.asarray([float(t.depth_mm) for t in run_tokens], dtype=float)
            depth_span = float(np.max(depth_vals) - np.min(depth_vals)) if depth_vals.size else 0.0
            superficial_penalty = float(np.sum(token_weights[depth_vals < float(cfg.get("superficial_depth_mm", 8.0))])) if depth_vals.size else 0.0
            unique_blobs = {int(t.blob_id) for t in run_tokens}
            degeneracy_penalty = max(0.0, float(min_unique_blobs - len(unique_blobs)))
            start_ras, end_ras = self._segment_from_run(center, axis, {"t_min_mm": t_min, "t_max_mm": t_max})
            start_ras, end_ras = self._orient_shallow_to_deep(start_ras, end_ras, depth_map_kji, candidate_mask_kji, support)
            in_head_fraction = self._segment_inside_fraction(start_ras, end_ras, gating_mask_kji, support)
            entry_depth = self._depth_at_ras(start_ras, depth_map_kji, support)
            target_depth = self._depth_at_ras(end_ras, depth_map_kji, support)
            depth_span = max(depth_span, 0.0 if entry_depth is None or target_depth is None else float(target_depth - entry_depth))
            score = (
                (1.15 * support_mass)
                + (0.08 * covered_length)
                + (0.03 * depth_span)
                + (1.5 * in_head_fraction)
                - (0.35 * internal_gap)
                - (1.75 * width_mm)
                - (0.18 * superficial_penalty)
                - (0.75 * degeneracy_penalty)
            )
            run_payload = {
                "token_ids": [int(t.token_id) for t in run_tokens],
                "blob_ids": [int(t.blob_id) for t in run_tokens],
                "token_count": int(len(run_tokens)),
                "unique_blob_count": int(len(unique_blobs)),
                "t_min_mm": float(t_min),
                "t_max_mm": float(t_max),
                "covered_length_mm": float(covered_length),
                "support_mass": float(support_mass),
                "depth_span_mm": float(depth_span),
                "internal_gap_mm": float(internal_gap),
                "width_mm": float(width_mm),
                "in_head_fraction": float(in_head_fraction),
                "superficial_penalty": float(superficial_penalty),
                "degeneracy_penalty": float(degeneracy_penalty),
                "score": float(score),
            }
            if best is None or float(run_payload["score"]) > float(best["score"]):
                best = run_payload
        return best

    def _nms_hypotheses(self, hypotheses: list[_Hypothesis], cfg: dict[str, Any]) -> list[_Hypothesis]:
        kept: list[_Hypothesis] = []
        for hyp in sorted(hypotheses, key=lambda item: float(item.local_score), reverse=True):
            duplicate = False
            p0 = 0.5 * (np.asarray(hyp.start_ras, dtype=float) + np.asarray(hyp.end_ras, dtype=float))
            d0 = self._normalize(np.asarray(hyp.direction_ras, dtype=float))
            for prev in kept:
                p1 = 0.5 * (np.asarray(prev.start_ras, dtype=float) + np.asarray(prev.end_ras, dtype=float))
                d1 = self._normalize(np.asarray(prev.direction_ras, dtype=float))
                angle = self._angle_deg(d0, d1)
                line_dist = self._line_to_line_distance(p0, d0, p1, d1)
                overlap = self._segment_overlap_fraction(hyp.start_ras, hyp.end_ras, prev.start_ras, prev.end_ras, d1)
                if angle <= float(cfg.get("nms_angle_deg", 8.0)) and line_dist <= float(cfg.get("nms_line_distance_mm", 2.0)) and overlap >= float(cfg.get("nms_overlap_fraction", 0.55)):
                    duplicate = True
                    break
            if duplicate:
                continue
            kept.append(hyp)
            if len(kept) >= int(cfg.get("max_hypotheses_after_nms", 160)):
                break
        return kept

    def _select_hypotheses(self, hypotheses: list[_Hypothesis], support: dict[str, Any], cfg: dict[str, Any]) -> tuple[list[_Hypothesis], list[dict[str, Any]]]:
        if not hypotheses:
            return [], []
        hypotheses = list(hypotheses)
        conflicts = self._conflicts(hypotheses, support, cfg)
        penalty_pairs = {(int(item["a"]), int(item["b"])): float(item["penalty"]) for item in conflicts}
        base_scores = {int(h.hyp_id): float(h.local_score) for h in hypotheses}
        complexity = float(cfg.get("selection_complexity_cost", 2.5))
        min_gain = float(cfg.get("selection_min_gain", 0.5))
        hyp_by_id = {int(h.hyp_id): h for h in hypotheses}
        ordered_ids = [int(h.hyp_id) for h in sorted(hypotheses, key=lambda item: float(item.local_score), reverse=True)]

        def objective(selected_ids: set[int]) -> float:
            total = 0.0
            for hyp_id in selected_ids:
                total += float(base_scores[hyp_id]) - complexity
            pairs = sorted(selected_ids)
            for i, a in enumerate(pairs):
                for b in pairs[i + 1 :]:
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
        selected = self._ownership_cleanup(selected, support, cfg)
        return selected, conflicts

    def _ownership_cleanup(self, selected: list[_Hypothesis], support: dict[str, Any], cfg: dict[str, Any]) -> list[_Hypothesis]:
        if not selected:
            return []
        tokens = list(support["tokens"])
        token_points = np.asarray([t.point_ras for t in tokens], dtype=float).reshape(-1, 3)
        ownership_radius = float(cfg.get("ownership_radius_mm", 2.5))
        min_affinity = float(cfg.get("ownership_min_affinity", 0.08))
        selected_axes = [self._normalize(np.asarray(h.direction_ras, dtype=float)) for h in selected]
        selected_centers = [0.5 * (np.asarray(h.start_ras, dtype=float) + np.asarray(h.end_ras, dtype=float)) for h in selected]
        affinities = np.zeros((len(tokens), len(selected)), dtype=float)
        for j, hyp in enumerate(selected):
            dist = self._line_distances(token_points, selected_centers[j], selected_axes[j])
            align = np.asarray([1.0 if t.blob_kind == "compact" else max(0.2, abs(float(np.dot(np.asarray(t.local_axis_ras, dtype=float), selected_axes[j])))) for t in tokens], dtype=float)
            affinities[:, j] = np.asarray([float(t.weight) * float(t.purity) for t in tokens], dtype=float) * np.exp(-0.5 * (dist / max(ownership_radius, 1e-3)) ** 2) * align
        best_idx = np.argmax(affinities, axis=1)
        best_val = np.max(affinities, axis=1)
        owned_by_hyp: dict[int, list[_SupportToken]] = {int(h.hyp_id): [] for h in selected}
        for token_idx, token in enumerate(tokens):
            if float(best_val[token_idx]) < min_affinity:
                continue
            owned_by_hyp[int(selected[int(best_idx[token_idx])].hyp_id)].append(token)
        refined: list[_Hypothesis] = []
        support_map = support
        for hyp in selected:
            owned_tokens = owned_by_hyp.get(int(hyp.hyp_id)) or []
            if len(owned_tokens) < int(cfg.get("min_support_tokens", 4)):
                continue
            center = 0.5 * (np.asarray(hyp.start_ras, dtype=float) + np.asarray(hyp.end_ras, dtype=float))
            axis = self._normalize(np.asarray(hyp.direction_ras, dtype=float))
            points = np.asarray([t.point_ras for t in owned_tokens], dtype=float).reshape(-1, 3)
            weights = np.asarray([float(t.weight) * float(t.purity) for t in owned_tokens], dtype=float)
            center, axis = self._weighted_pca_line(points, weights)
            dist = self._line_distances(points, center, axis)
            projections = (points - center.reshape(1, 3)) @ axis.reshape(3)
            run = self._best_support_run(
                center,
                axis,
                owned_tokens,
                projections,
                dist,
                cfg,
                support_map.get("preview", {}),
                support_map.get("candidate_mask_kji"),
                support_map.get("preview", {}).get("head_distance_map_kji"),
                support_map.get("preview", {}).get("gating_mask_kji") if support_map.get("preview", {}).get("gating_mask_kji") is not None else support_map.get("preview", {}).get("head_mask_kji"),
                support_map.get("preview", {}).get("head_distance_map_kji"),
                support_map,
            )
            if run is None:
                continue
            start, end = self._segment_from_run(center, axis, run)
            start, end = self._orient_shallow_to_deep(start, end, support_map.get("preview", {}).get("head_distance_map_kji"), support_map.get("candidate_mask_kji"), support_map)
            refined.append(
                _Hypothesis(
                    hyp_id=int(hyp.hyp_id),
                    seed_kind=str(hyp.seed_kind),
                    seed_blob_ids=tuple(hyp.seed_blob_ids),
                    point_ras=tuple(float(v) for v in center),
                    direction_ras=tuple(float(v) for v in self._normalize(end - start)),
                    start_ras=tuple(float(v) for v in start),
                    end_ras=tuple(float(v) for v in end),
                    owned_token_ids=tuple(int(v) for v in run["token_ids"]),
                    token_blob_ids=tuple(int(v) for v in run["blob_ids"]),
                    support_mass=float(run["support_mass"]),
                    covered_length_mm=float(run["covered_length_mm"]),
                    depth_span_mm=float(run["depth_span_mm"]),
                    internal_gap_mm=float(run["internal_gap_mm"]),
                    width_mm=float(run["width_mm"]),
                    in_head_fraction=float(run["in_head_fraction"]),
                    superficial_penalty=float(run["superficial_penalty"]),
                    degeneracy_penalty=float(run["degeneracy_penalty"]),
                    local_score=float(run["score"]),
                    extras={**dict(hyp.extras), "ownership_cleanup": True},
                )
            )
        return sorted(refined, key=lambda item: float(item.local_score), reverse=True)

    def _refine_and_rerank_selected(
        self,
        ctx: DetectionContext,
        support: dict[str, Any],
        refinement_support: dict[str, Any],
        selected: list[_Hypothesis],
        cfg: dict[str, Any],
    ) -> list[_SelectedHypothesis]:
        if not selected:
            return []
        candidate_points_lps = np.asarray(refinement_support.get("candidate_points_lps"), dtype=float).reshape(-1, 3)
        models_by_id = dict(ctx.get("extras") or {}).get("models_by_id")
        selected_final: list[_SelectedHypothesis] = []
        for hyp in selected:
            start_ras = np.asarray(hyp.start_ras, dtype=float)
            end_ras = np.asarray(hyp.end_ras, dtype=float)
            fit = fit_electrode_axis_and_tip(
                candidate_points_lps=candidate_points_lps,
                planned_entry_lps=self._ras_to_lps(start_ras),
                planned_target_lps=self._ras_to_lps(end_ras),
                contact_offsets_mm=None,
                tip_at="target",
                roi_radius_mm=float(cfg.get("guided_roi_radius_mm", 5.0)),
                max_angle_deg=float(cfg.get("guided_max_angle_deg", 12.0)),
                max_depth_shift_mm=float(cfg.get("guided_max_depth_shift_mm", 6.0)),
                fit_mode=str(cfg.get("guided_fit_mode", "deep_anchor_v2")),
            )
            refined_start = refined_end = None
            refinement_success = bool(fit.get("success"))
            refinement_reason = str(fit.get("reason") or "")
            if refinement_success:
                refined_start = tuple(float(v) for v in self._lps_to_ras(np.asarray(fit["entry_lps"], dtype=float)))
                refined_end = tuple(float(v) for v in self._lps_to_ras(np.asarray(fit["target_lps"], dtype=float)))
            late_model_prior_score, best_model_id = self._late_model_prior_score(hyp, support, models_by_id)
            global_score = float(hyp.local_score) + 0.35 * float(late_model_prior_score) + (0.5 if refinement_success else 0.0)
            selected_final.append(
                _SelectedHypothesis(
                    hyp_id=int(hyp.hyp_id),
                    start_ras=tuple(hyp.start_ras),
                    end_ras=tuple(hyp.end_ras),
                    direction_ras=tuple(hyp.direction_ras),
                    owned_token_ids=tuple(hyp.owned_token_ids),
                    local_score=float(hyp.local_score),
                    global_score=float(global_score),
                    late_model_prior_score=float(late_model_prior_score),
                    best_model_id=str(best_model_id or ""),
                    refined_start_ras=refined_start,
                    refined_end_ras=refined_end,
                    refinement_success=bool(refinement_success),
                    refinement_reason=str(refinement_reason),
                    extras={
                        **dict(hyp.extras),
                        "fit_mode_used": str(fit.get("fit_mode_used") or ""),
                        "deep_anchor_source": str(fit.get("deep_anchor_source") or ""),
                        "residual_mm": float(fit.get("residual_mm") or 0.0),
                    },
                )
            )
        return sorted(selected_final, key=lambda item: float(item.global_score), reverse=True)

    def _late_model_prior_score(self, hyp: _Hypothesis, support: dict[str, Any], models_by_id: Any) -> tuple[float, str]:
        tokens_by_id = {int(t.token_id): t for t in list(support["tokens"])}
        run_tokens = [tokens_by_id[token_id] for token_id in hyp.owned_token_ids if int(token_id) in tokens_by_id]
        if len(run_tokens) < 3 or not isinstance(models_by_id, dict) or not models_by_id:
            return 0.0, ""
        axis = self._normalize(np.asarray(hyp.direction_ras, dtype=float))
        origin = np.asarray(hyp.start_ras, dtype=float)
        blob_points: dict[int, list[float]] = {}
        for token in run_tokens:
            t = float(np.dot(np.asarray(token.point_ras, dtype=float) - origin, axis))
            blob_points.setdefault(int(token.blob_id), []).append(t)
        centers = np.asarray(sorted(float(np.mean(vals)) for vals in blob_points.values()), dtype=float)
        if centers.size < 3:
            return 0.0, ""
        gaps = np.diff(centers)
        small_gaps = gaps[gaps <= 6.0]
        large_gaps = gaps[gaps > 6.0]
        median_pitch = float(np.median(small_gaps)) if small_gaps.size else float(np.median(gaps))
        best_score = 0.0
        best_model_id = ""
        for model_id, model in list(models_by_id.items()):
            offsets = np.asarray(list(model.get("contact_center_offsets_from_tip_mm") or []), dtype=float)
            if offsets.size < 2:
                continue
            model_gaps = np.diff(np.sort(offsets))
            model_small = model_gaps[model_gaps <= 6.0]
            model_large = model_gaps[model_gaps > 6.0]
            if model_small.size:
                pitch = float(np.median(model_small))
            else:
                pitch = float(np.median(model_gaps))
            pitch_score = math.exp(-(((median_pitch - pitch) / max(0.75, 0.25 * max(pitch, 1.0))) ** 2))
            if large_gaps.size and model_large.size:
                gap_delta = float(np.min(np.abs(large_gaps.reshape(-1, 1) - model_large.reshape(1, -1))))
                gap_score = math.exp(-((gap_delta / 1.5) ** 2))
            elif not large_gaps.size and not model_large.size:
                gap_score = 0.5
            else:
                gap_score = 0.0
            score = float(pitch_score + 0.5 * gap_score)
            if score > best_score:
                best_score = score
                best_model_id = str(model_id)
        return float(best_score), best_model_id

    def _conflicts(self, hypotheses: list[_Hypothesis], support: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
        shared_weight = float(cfg.get("selection_shared_token_weight", 1.1))
        redundancy_weight = float(cfg.get("selection_redundancy_weight", 3.0))
        token_weight = {int(t.token_id): float(t.weight) * float(t.purity) for t in list(support["tokens"])}
        out: list[dict[str, Any]] = []
        for i, a in enumerate(hypotheses):
            set_a = set(int(v) for v in a.owned_token_ids)
            p0 = 0.5 * (np.asarray(a.start_ras, dtype=float) + np.asarray(a.end_ras, dtype=float))
            d0 = self._normalize(np.asarray(a.direction_ras, dtype=float))
            for b in hypotheses[i + 1 :]:
                set_b = set(int(v) for v in b.owned_token_ids)
                shared_ids = sorted(set_a & set_b)
                shared_mass = float(sum(token_weight.get(token_id, 0.0) for token_id in shared_ids))
                p1 = 0.5 * (np.asarray(b.start_ras, dtype=float) + np.asarray(b.end_ras, dtype=float))
                d1 = self._normalize(np.asarray(b.direction_ras, dtype=float))
                angle = self._angle_deg(d0, d1)
                line_dist = self._line_to_line_distance(p0, d0, p1, d1)
                overlap = self._segment_overlap_fraction(a.start_ras, a.end_ras, b.start_ras, b.end_ras, d1)
                redundant = angle <= 10.0 and line_dist <= 2.5 and overlap >= 0.3
                if shared_mass <= 1e-6 and not redundant:
                    continue
                penalty = (shared_weight * shared_mass) + (redundancy_weight * overlap if redundant else 0.0)
                out.append(
                    {
                        "a": int(a.hyp_id),
                        "b": int(b.hyp_id),
                        "shared_token_ids": shared_ids,
                        "shared_mass": float(shared_mass),
                        "angle_deg": float(angle),
                        "line_distance_mm": float(line_dist),
                        "segment_overlap_fraction": float(overlap),
                        "penalty": float(penalty),
                    }
                )
        return out

    def _trajectory_from_hypothesis(self, hyp: _Hypothesis) -> dict[str, Any]:
        return {
            "name": f"H{int(hyp.hyp_id):03d}",
            "start_ras": [float(v) for v in hyp.start_ras],
            "end_ras": [float(v) for v in hyp.end_ras],
            "length_mm": float(np.linalg.norm(np.asarray(hyp.end_ras, dtype=float) - np.asarray(hyp.start_ras, dtype=float))),
            "confidence": float(min(1.0, max(0.0, hyp.local_score / 25.0))),
            "support_count": int(len(hyp.owned_token_ids)),
            "params": {
                "local_score": float(hyp.local_score),
                "support_mass": float(hyp.support_mass),
                "covered_length_mm": float(hyp.covered_length_mm),
                "depth_span_mm": float(hyp.depth_span_mm),
                "internal_gap_mm": float(hyp.internal_gap_mm),
                "width_mm": float(hyp.width_mm),
                "in_head_fraction": float(hyp.in_head_fraction),
                "seed_kind": str(hyp.seed_kind),
                "seed_blob_ids": list(int(v) for v in hyp.seed_blob_ids),
            },
        }

    def _trajectory_from_selected(self, hyp: _SelectedHypothesis) -> dict[str, Any]:
        start = np.asarray(hyp.refined_start_ras if hyp.refinement_success and hyp.refined_start_ras is not None else hyp.start_ras, dtype=float)
        end = np.asarray(hyp.refined_end_ras if hyp.refinement_success and hyp.refined_end_ras is not None else hyp.end_ras, dtype=float)
        return {
            "name": f"D{int(hyp.hyp_id):03d}",
            "start_ras": [float(v) for v in start],
            "end_ras": [float(v) for v in end],
            "length_mm": float(np.linalg.norm(end - start)),
            "confidence": float(min(1.0, max(0.0, hyp.global_score / 25.0))),
            "support_count": int(len(hyp.owned_token_ids)),
            "params": {
                "local_score": float(hyp.local_score),
                "global_score": float(hyp.global_score),
                "late_model_prior_score": float(hyp.late_model_prior_score),
                "best_model_id": str(hyp.best_model_id),
                "refinement_success": int(bool(hyp.refinement_success)),
                "refinement_reason": str(hyp.refinement_reason),
                **dict(hyp.extras),
            },
        }

    def _record_support_diagnostics(self, diagnostics, support: dict[str, Any]) -> None:
        diagnostics.set_count("candidate_points_total", int(np.asarray(support.get("candidate_points_ras"), dtype=float).reshape(-1, 3).shape[0]))
        diagnostics.set_count("candidate_points_after_depth", int(support.get("candidate_points_after_mask", support.get("preview", {}).get("depth_kept_count") or 0)))
        diagnostics.set_count("blob_count_total", int(support.get("blob_result", {}).get("blob_count_total") or 0))
        token_summary = dict(support.get("token_summary") or {})
        for key, value in token_summary.items():
            diagnostics.set_count(str(key), int(value))
        diagnostics.set_extra("chosen_threshold_hu", float(support.get("threshold_hu") or 0.0))
        diagnostics.set_extra("threshold_trials", list(support.get("preview_stats") or []))
        diagnostics.set_extra("token_summary", token_summary)
        diagnostics.set_extra("support_pool_name", str(support.get("pool_name") or "proposal"))
        diagnostics.set_extra("proposal_support_mode", str(support.get("support_mode") or "depth_mask"))
        if support.get("support_mode_stats") is not None:
            diagnostics.set_extra("proposal_support_mode_stats", dict(support.get("support_mode_stats") or {}))
        if support.get("support_mode_counts") is not None:
            diagnostics.set_extra("proposal_support_mode_counts", list(support.get("support_mode_counts") or []))

    def _record_refinement_support_diagnostics(self, diagnostics, support: dict[str, Any]) -> None:
        diagnostics.set_extra("refinement_support_pool_name", str(support.get("pool_name") or "refinement"))
        diagnostics.set_extra("refinement_chosen_threshold_hu", float(support.get("threshold_hu") or 0.0))
        diagnostics.set_extra("refinement_threshold_trials", list(support.get("preview_stats") or []))
        diagnostics.set_extra("refinement_candidate_points_after_depth", int(support.get("preview", {}).get("depth_kept_count") or 0))
        diagnostics.set_extra("refinement_blob_count_total", int(support.get("blob_result", {}).get("blob_count_total") or 0))
        diagnostics.set_extra("refinement_token_summary", dict(support.get("token_summary") or {}))

    def _attach_support_artifact(self, result: DetectionResult, ctx: DetectionContext, support: dict[str, Any]) -> None:
        writer = self.get_artifact_writer(ctx, result)
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
            for token in list(support.get("tokens") or [])
        ]
        token_path = writer.write_csv_rows(
            "de_novo_tokens.csv",
            ["token_id", "blob_id", "blob_kind", "x", "y", "z", "weight", "purity", "seedable", "depth_mm"],
            token_rows,
        )
        add_artifact(result["artifacts"], kind="token_csv", path=token_path, description="De novo support tokens", stage="support_pool")
        result["artifacts"].extend(
            write_standard_artifacts(
                writer,
                result,
                blobs=[],
                pipeline_payload={
                    "pipeline_id": self.pipeline_id,
                    "pipeline_version": self.pipeline_version,
                    "support": {
                        "threshold_hu": float(support.get("threshold_hu") or 0.0),
                        "preview_stats": list(support.get("preview_stats") or []),
                        "token_summary": dict(support.get("token_summary") or {}),
                    },
                },
            )
        )

    @staticmethod
    def _hypothesis_json(h: _Hypothesis) -> dict[str, Any]:
        return {
            "hyp_id": int(h.hyp_id),
            "seed_kind": str(h.seed_kind),
            "seed_blob_ids": [int(v) for v in h.seed_blob_ids],
            "start_ras": [float(v) for v in h.start_ras],
            "end_ras": [float(v) for v in h.end_ras],
            "local_score": float(h.local_score),
            "support_mass": float(h.support_mass),
            "covered_length_mm": float(h.covered_length_mm),
            "depth_span_mm": float(h.depth_span_mm),
            "internal_gap_mm": float(h.internal_gap_mm),
            "width_mm": float(h.width_mm),
            "in_head_fraction": float(h.in_head_fraction),
            "owned_token_ids": [int(v) for v in h.owned_token_ids],
            "token_blob_ids": [int(v) for v in h.token_blob_ids],
            "extras": dict(h.extras),
        }

    @staticmethod
    def _selected_json(h: _SelectedHypothesis) -> dict[str, Any]:
        return {
            "hyp_id": int(h.hyp_id),
            "start_ras": [float(v) for v in h.start_ras],
            "end_ras": [float(v) for v in h.end_ras],
            "refined_start_ras": None if h.refined_start_ras is None else [float(v) for v in h.refined_start_ras],
            "refined_end_ras": None if h.refined_end_ras is None else [float(v) for v in h.refined_end_ras],
            "local_score": float(h.local_score),
            "global_score": float(h.global_score),
            "late_model_prior_score": float(h.late_model_prior_score),
            "best_model_id": str(h.best_model_id),
            "refinement_success": int(bool(h.refinement_success)),
            "refinement_reason": str(h.refinement_reason),
            "owned_token_ids": [int(v) for v in h.owned_token_ids],
            "extras": dict(h.extras),
        }

    @staticmethod
    def _normalize(vec: np.ndarray | list[float] | tuple[float, float, float]) -> np.ndarray:
        arr = np.asarray(vec, dtype=float).reshape(3)
        norm = float(np.linalg.norm(arr))
        if norm <= 1e-9:
            return np.asarray([0.0, 0.0, 1.0], dtype=float)
        return arr / norm

    @staticmethod
    def _pairwise_distances(points: np.ndarray) -> np.ndarray:
        diff = points[:, None, :] - points[None, :, :]
        return np.linalg.norm(diff, axis=2)

    @staticmethod
    def _line_distances(points: np.ndarray, p0: np.ndarray, direction_unit: np.ndarray) -> np.ndarray:
        rel = points - p0.reshape(1, 3)
        t = rel @ direction_unit.reshape(3)
        closest = p0.reshape(1, 3) + np.outer(t, direction_unit.reshape(3))
        return np.linalg.norm(points - closest, axis=1)

    @staticmethod
    def _weighted_pca_line(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w = np.asarray(weights, dtype=float).reshape(-1)
        w = np.maximum(w, 0.0)
        s = float(np.sum(w))
        if s <= 1e-9:
            raise ValueError("zero weights")
        wn = w / s
        center = np.sum(points * wn[:, None], axis=0)
        x = points - center.reshape(1, 3)
        cov = (x.T * wn.reshape(1, -1)) @ x
        evals, evecs = np.linalg.eigh(cov)
        axis = evecs[:, int(np.argmax(evals))]
        axis = DeNovoHypothesisSelectV1Pipeline._normalize(axis)
        return center.astype(float), axis.astype(float)

    @staticmethod
    def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
        aa = DeNovoHypothesisSelectV1Pipeline._normalize(a)
        bb = DeNovoHypothesisSelectV1Pipeline._normalize(b)
        c = float(np.clip(abs(float(np.dot(aa, bb))), 0.0, 1.0))
        return float(np.degrees(np.arccos(c)))

    @staticmethod
    def _line_to_line_distance(p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray) -> float:
        u = DeNovoHypothesisSelectV1Pipeline._normalize(d1)
        v = DeNovoHypothesisSelectV1Pipeline._normalize(d2)
        w0 = np.asarray(p1, dtype=float).reshape(3) - np.asarray(p2, dtype=float).reshape(3)
        c = np.cross(u, v)
        cn = float(np.linalg.norm(c))
        if cn <= 1e-6:
            return float(np.linalg.norm(np.cross(w0, u)))
        return float(abs(np.dot(w0, c)) / cn)

    @staticmethod
    def _segment_overlap_fraction(start_a, end_a, start_b, end_b, axis_ref: np.ndarray | None = None) -> float:
        a0 = np.asarray(start_a, dtype=float)
        a1 = np.asarray(end_a, dtype=float)
        b0 = np.asarray(start_b, dtype=float)
        b1 = np.asarray(end_b, dtype=float)
        if axis_ref is None:
            axis_ref = DeNovoHypothesisSelectV1Pipeline._normalize(b1 - b0)
        else:
            axis_ref = DeNovoHypothesisSelectV1Pipeline._normalize(axis_ref)
        origin = 0.5 * (b0 + b1)
        a_proj = sorted([float(np.dot(a0 - origin, axis_ref)), float(np.dot(a1 - origin, axis_ref))])
        b_proj = sorted([float(np.dot(b0 - origin, axis_ref)), float(np.dot(b1 - origin, axis_ref))])
        overlap = max(0.0, min(a_proj[1], b_proj[1]) - max(a_proj[0], b_proj[0]))
        denom = max(1e-6, min(a_proj[1] - a_proj[0], b_proj[1] - b_proj[0]))
        return float(overlap / denom)

    @staticmethod
    def _canonical_direction(direction: np.ndarray) -> np.ndarray:
        d = DeNovoHypothesisSelectV1Pipeline._normalize(direction)
        lead = int(np.argmax(np.abs(d)))
        if float(d[lead]) < 0.0:
            d = -d
        return d

    @staticmethod
    def _hough_key(direction: np.ndarray, offset: np.ndarray, *, dir_bin_deg: float, offset_bin_mm: float) -> tuple[Any, ...]:
        d = DeNovoHypothesisSelectV1Pipeline._canonical_direction(direction)
        denom = max(math.sin(math.radians(max(dir_bin_deg, 1.0))), 1e-3)
        return (
            int(round(float(d[0]) / denom)),
            int(round(float(d[1]) / denom)),
            int(round(float(d[2]) / denom)),
            int(round(float(offset[0]) / offset_bin_mm)),
            int(round(float(offset[1]) / offset_bin_mm)),
            int(round(float(offset[2]) / offset_bin_mm)),
        )

    @staticmethod
    def _direction_key(direction: np.ndarray, *, dir_bin_deg: float) -> tuple[int, int, int]:
        d = DeNovoHypothesisSelectV1Pipeline._canonical_direction(direction)
        denom = max(math.sin(math.radians(max(dir_bin_deg, 1.0))), 1e-3)
        return (
            int(round(float(d[0]) / denom)),
            int(round(float(d[1]) / denom)),
            int(round(float(d[2]) / denom)),
        )

    @staticmethod
    def _orthonormal_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = DeNovoHypothesisSelectV1Pipeline._canonical_direction(direction)
        ref = np.asarray([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(d, ref))) > 0.9:
            ref = np.asarray([0.0, 1.0, 0.0], dtype=float)
        u = np.cross(d, ref)
        u = DeNovoHypothesisSelectV1Pipeline._normalize(u)
        v = np.cross(d, u)
        v = DeNovoHypothesisSelectV1Pipeline._normalize(v)
        return u.astype(float), v.astype(float)

    @staticmethod
    def _seed_key(point: np.ndarray, direction: np.ndarray) -> tuple[Any, ...]:
        p = np.asarray(point, dtype=float).reshape(3)
        d = DeNovoHypothesisSelectV1Pipeline._normalize(direction)
        if d[2] < 0.0:
            d = -d
        return (
            round(float(p[0]), 1),
            round(float(p[1]), 1),
            round(float(p[2]), 1),
            round(float(d[0]), 2),
            round(float(d[1]), 2),
            round(float(d[2]), 2),
        )

    @staticmethod
    def _segment_from_run(center: np.ndarray, axis: np.ndarray, run: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        start = np.asarray(center, dtype=float) + np.asarray(axis, dtype=float) * float(run["t_min_mm"])
        end = np.asarray(center, dtype=float) + np.asarray(axis, dtype=float) * float(run["t_max_mm"])
        return start.astype(float), end.astype(float)

    @staticmethod
    def _ras_to_lps(points_ras: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_ras, dtype=float)
        if pts.ndim == 1:
            out = pts.reshape(3).copy()
            out[0] *= -1.0
            out[1] *= -1.0
            return out
        out = pts.reshape(-1, 3).copy()
        out[:, 0] *= -1.0
        out[:, 1] *= -1.0
        return out

    @staticmethod
    def _lps_to_ras(point_lps: np.ndarray) -> np.ndarray:
        p = np.asarray(point_lps, dtype=float).reshape(3)
        return np.asarray([-p[0], -p[1], p[2]], dtype=float)

    @staticmethod
    def _ijk_kji_to_ras(ctx: DetectionContext, ijk_kji: np.ndarray) -> np.ndarray:
        arr = np.asarray(ctx["ijk_kji_to_ras_fn"](ijk_kji), dtype=float).reshape(-1, 3)
        return arr.astype(float)

    @staticmethod
    def _depth_at_ras(point_ras: np.ndarray, depth_map_kji: np.ndarray | None, support: dict[str, Any]) -> float | None:
        if depth_map_kji is None:
            return None
        ras_to_ijk_fn = support.get("ras_to_ijk_fn") if isinstance(support, dict) else None
        if ras_to_ijk_fn is None:
            return None
        ijk = ras_to_ijk_fn(point_ras)
        i = int(round(float(ijk[0])))
        j = int(round(float(ijk[1])))
        k = int(round(float(ijk[2])))
        if k < 0 or j < 0 or i < 0 or k >= depth_map_kji.shape[0] or j >= depth_map_kji.shape[1] or i >= depth_map_kji.shape[2]:
            return None
        val = float(depth_map_kji[k, j, i])
        return val if math.isfinite(val) else None

    @staticmethod
    def _orient_shallow_to_deep(start: np.ndarray, end: np.ndarray, depth_map_kji: np.ndarray | None, _candidate_mask_kji: np.ndarray | None, support: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        ras_to_ijk_fn = support.get("ras_to_ijk_fn")
        if ras_to_ijk_fn is None:
            ras_to_ijk_fn = support.get("_ras_to_ijk_fn")
        if ras_to_ijk_fn is None:
            ras_to_ijk_fn = support.get("ctx_ras_to_ijk_fn")
        if ras_to_ijk_fn is None:
            ras_to_ijk_fn = support.get("ras_to_ijk")
        if ras_to_ijk_fn is None:
            return start, end
        d0 = DeNovoHypothesisSelectV1Pipeline._depth_at_ras_with_fn(start, depth_map_kji, ras_to_ijk_fn)
        d1 = DeNovoHypothesisSelectV1Pipeline._depth_at_ras_with_fn(end, depth_map_kji, ras_to_ijk_fn)
        if d0 is not None and d1 is not None and abs(d0 - d1) > 1e-3:
            return (start, end) if d0 <= d1 else (end, start)
        return start, end

    @staticmethod
    def _depth_at_ras_with_fn(point_ras: np.ndarray, depth_map_kji: np.ndarray | None, ras_to_ijk_fn) -> float | None:
        if depth_map_kji is None or ras_to_ijk_fn is None:
            return None
        ijk = ras_to_ijk_fn(point_ras)
        i = int(round(float(ijk[0])))
        j = int(round(float(ijk[1])))
        k = int(round(float(ijk[2])))
        if k < 0 or j < 0 or i < 0 or k >= depth_map_kji.shape[0] or j >= depth_map_kji.shape[1] or i >= depth_map_kji.shape[2]:
            return None
        val = float(depth_map_kji[k, j, i])
        return val if math.isfinite(val) else None

    @staticmethod
    def _segment_inside_fraction(start_ras: np.ndarray, end_ras: np.ndarray, mask_kji: np.ndarray | None, support: dict[str, Any], step_mm: float = 1.0) -> float:
        ras_to_ijk_fn = support.get("ras_to_ijk_fn") if isinstance(support, dict) else None
        if mask_kji is None or ras_to_ijk_fn is None:
            return 1.0
        seg = np.asarray(end_ras, dtype=float) - np.asarray(start_ras, dtype=float)
        length = float(np.linalg.norm(seg))
        if length <= 1e-9:
            return 0.0
        n = max(2, int(math.ceil(length / max(1e-3, float(step_mm)))) + 1)
        ts = np.linspace(0.0, 1.0, n)
        dims = mask_kji.shape
        inside = 0
        for t in ts:
            p = np.asarray(start_ras, dtype=float) + t * seg
            ijk = ras_to_ijk_fn(p)
            i = int(round(float(ijk[0])))
            j = int(round(float(ijk[1])))
            k = int(round(float(ijk[2])))
            if 0 <= k < dims[0] and 0 <= j < dims[1] and 0 <= i < dims[2] and bool(mask_kji[k, j, i]):
                inside += 1
        return float(inside) / float(max(1, n))
