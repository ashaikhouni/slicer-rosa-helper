"""Shank-only voxel pipeline optimized for shaft direction and endpoints.

This pipeline intentionally does not attempt contact recovery. It over-generates
voxel/RANSAC line proposals from the existing shank_core detector, then rescoring,
refits, deduplicates, and endpoint-corrects them against gated metal support voxels.

Design intent:
- maximize shank recovery
- minimize duplicate/extra shanks
- optimize line direction + deep endpoint first, shallow start second
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from shank_core.pipeline import run_detection

from ..artifacts import add_artifact, write_standard_artifacts
from ..contracts import DetectionContext, DetectionResult
from .base import BaseDetectionPipeline


def _normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-9:
        return np.asarray([0.0, 0.0, 1.0], dtype=float)
    return arr / norm


def _depth_at_ras_mm(point_ras: np.ndarray, head_depth_kji: np.ndarray | None, ras_to_ijk_fn) -> float | None:
    if head_depth_kji is None or ras_to_ijk_fn is None:
        return None
    ijk = ras_to_ijk_fn(point_ras)
    i = int(round(float(ijk[0])))
    j = int(round(float(ijk[1])))
    k = int(round(float(ijk[2])))
    if k < 0 or j < 0 or i < 0:
        return None
    if k >= head_depth_kji.shape[0] or j >= head_depth_kji.shape[1] or i >= head_depth_kji.shape[2]:
        return None
    val = float(head_depth_kji[k, j, i])
    if not np.isfinite(val):
        return None
    return val


def _line_distances(points: np.ndarray, p0: np.ndarray, direction_unit: np.ndarray) -> np.ndarray:
    rel = points - p0.reshape(1, 3)
    t = rel @ direction_unit.reshape(3)
    closest = p0.reshape(1, 3) + np.outer(t, direction_unit.reshape(3))
    return np.linalg.norm(points - closest, axis=1)


def _distance_to_segment_mask(points: np.ndarray, start: np.ndarray, end: np.ndarray, radius_mm: float) -> np.ndarray:
    seg = end - start
    seg_len2 = float(np.dot(seg, seg))
    if seg_len2 <= 1e-9:
        d = np.linalg.norm(points - start.reshape(1, 3), axis=1)
        return d <= float(radius_mm)
    rel = points - start.reshape(1, 3)
    t = (rel @ seg) / seg_len2
    t = np.clip(t, 0.0, 1.0)
    closest = start.reshape(1, 3) + np.outer(t, seg)
    d = np.linalg.norm(points - closest, axis=1)
    return d <= float(radius_mm)


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
    axis = _normalize(evecs[:, int(np.argmax(evals))])
    return center.astype(float), axis.astype(float)


def _segment_inside_fraction(start_ras: np.ndarray, end_ras: np.ndarray, mask_kji: np.ndarray | None, ras_to_ijk_fn, step_mm: float = 1.0) -> float:
    if mask_kji is None or ras_to_ijk_fn is None:
        return 1.0
    seg = end_ras - start_ras
    length = float(np.linalg.norm(seg))
    if length <= 1e-9:
        return 0.0
    n = max(2, int(math.ceil(length / max(1e-3, float(step_mm)))) + 1)
    ts = np.linspace(0.0, 1.0, n)
    dims = mask_kji.shape
    inside = 0
    for t in ts:
        p = start_ras + t * seg
        ijk = ras_to_ijk_fn(p)
        i = int(round(float(ijk[0])))
        j = int(round(float(ijk[1])))
        k = int(round(float(ijk[2])))
        if 0 <= k < dims[0] and 0 <= j < dims[1] and 0 <= i < dims[2] and bool(mask_kji[k, j, i]):
            inside += 1
    return float(inside) / float(max(1, n))


def _projected_endpoints(points: np.ndarray, center: np.ndarray, axis: np.ndarray, lo_q: float, hi_q: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    proj = (points - center.reshape(1, 3)) @ axis.reshape(3)
    lo = float(np.quantile(proj, lo_q))
    hi = float(np.quantile(proj, hi_q))
    start = center + axis * lo
    end = center + axis * hi
    return start.astype(float), end.astype(float), proj


def _orient_shallow_to_deep(start: np.ndarray, end: np.ndarray, depth_kji: np.ndarray | None, ras_to_ijk_fn, center_ras: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d0 = _depth_at_ras_mm(start, depth_kji, ras_to_ijk_fn)
    d1 = _depth_at_ras_mm(end, depth_kji, ras_to_ijk_fn)
    if d0 is not None and d1 is not None and abs(float(d0) - float(d1)) > 1e-3:
        return (start, end) if float(d0) <= float(d1) else (end, start)
    c0 = float(np.linalg.norm(start - center_ras))
    c1 = float(np.linalg.norm(end - center_ras))
    return (start, end) if c0 >= c1 else (end, start)


def _extend_to_surface(start: np.ndarray, axis: np.ndarray, gating_mask_kji: np.ndarray | None, ras_to_ijk_fn, max_backtrack_mm: float = 20.0, step_mm: float = 0.5) -> np.ndarray:
    if gating_mask_kji is None or ras_to_ijk_fn is None:
        return start
    dims = gating_mask_kji.shape
    best = np.asarray(start, dtype=float)
    last_inside = np.asarray(start, dtype=float)
    samples = max(1, int(math.ceil(max_backtrack_mm / max(step_mm, 1e-3))))
    for idx in range(1, samples + 1):
        p = start - axis * (idx * step_mm)
        ijk = ras_to_ijk_fn(p)
        i = int(round(float(ijk[0])))
        j = int(round(float(ijk[1])))
        k = int(round(float(ijk[2])))
        inside = 0 <= k < dims[0] and 0 <= j < dims[1] and 0 <= i < dims[2] and bool(gating_mask_kji[k, j, i])
        if inside:
            last_inside = p
            best = p
        else:
            break
    return np.asarray(best if np.any(best != start) else last_inside, dtype=float)


def _line_distance(p0: np.ndarray, d0: np.ndarray, p1: np.ndarray, d1: np.ndarray) -> float:
    u = _normalize(d0)
    v = _normalize(d1)
    w0 = p0 - p1
    c = np.cross(u, v)
    cn = float(np.linalg.norm(c))
    if cn <= 1e-6:
        return float(np.linalg.norm(np.cross(w0, u)))
    return float(abs(np.dot(w0, c)) / cn)


def _midpoint(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    return 0.5 * (start + end)


class ShankAxisV1Pipeline(BaseDetectionPipeline):
    """Shank-focused voxel proposal pipeline with geometry-first selection."""

    pipeline_id = "shank_axis_v1"
    display_name = "Shank Axis v1"
    pipeline_version = "1.0.0"

    def _proposal_config(self, cfg: dict[str, Any]) -> dict[str, Any]:
        proposal = dict(cfg)
        proposal["max_lines"] = int(cfg.get("proposal_max_lines", max(int(cfg.get("max_lines", 30)) * 4, 40)))
        proposal["min_inliers"] = int(cfg.get("proposal_min_inliers", min(int(cfg.get("min_inliers", 6)), 6)))
        proposal["candidate_mode"] = "voxel"
        proposal["min_model_score"] = float(cfg["min_model_score"]) if cfg.get("min_model_score") is not None else None
        return proposal

    def _refine_lines(self, raw: dict[str, Any], ctx: DetectionContext, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        support_points_raw = raw.get("in_mask_points_ras")
        support_points = np.asarray(support_points_raw if support_points_raw is not None else [], dtype=float).reshape(-1, 3)
        if support_points.size == 0:
            return []
        center_ras_raw = ctx.get("center_ras")
        center_ras = np.asarray(center_ras_raw if center_ras_raw is not None else [0.0, 0.0, 0.0], dtype=float)
        head_depth = raw.get("head_distance_map_kji")
        gating_mask = raw.get("gating_mask_kji", raw.get("head_mask_kji"))
        ras_to_ijk_fn = ctx.get("ras_to_ijk_fn")
        radius_mm = float(cfg.get("selection_support_radius_mm", max(1.2, float(cfg.get("inlier_radius_mm", 1.2)) * 1.25)))
        lo_q = float(cfg.get("endpoint_lo_quantile", 0.05))
        hi_q = float(cfg.get("endpoint_hi_quantile", 0.95))
        refined: list[dict[str, Any]] = []

        for idx, line in enumerate(list(raw.get("lines") or []), start=1):
            start = np.asarray(line.get("start_ras", [0.0, 0.0, 0.0]), dtype=float)
            end = np.asarray(line.get("end_ras", [0.0, 0.0, 0.0]), dtype=float)
            support_mask = _distance_to_segment_mask(support_points, start, end, radius_mm=radius_mm)
            pts = support_points[support_mask]
            if pts.shape[0] < 3:
                continue
            raw_axis = _normalize(end - start)
            raw_mid = 0.5 * (start + end)
            fit_start, fit_end, proj = _projected_endpoints(pts, raw_mid, raw_axis, lo_q=lo_q, hi_q=hi_q)
            fit_start, fit_end = _orient_shallow_to_deep(fit_start, fit_end, head_depth, ras_to_ijk_fn, center_ras)
            axis = _normalize(fit_end - fit_start)
            start_extend_mm = float(cfg.get("start_extend_mm", 0.0))
            if start_extend_mm > 0.0:
                fit_start = _extend_to_surface(fit_start, axis, gating_mask, ras_to_ijk_fn, max_backtrack_mm=start_extend_mm)
                fit_start, fit_end = _orient_shallow_to_deep(fit_start, fit_end, head_depth, ras_to_ijk_fn, center_ras)

            rms = float(np.sqrt(np.mean(_line_distances(pts, raw_mid, raw_axis) ** 2))) if pts.shape[0] else 0.0
            inside_fraction = _segment_inside_fraction(fit_start, fit_end, gating_mask, ras_to_ijk_fn, step_mm=1.0)
            entry_depth = _depth_at_ras_mm(fit_start, head_depth, ras_to_ijk_fn)
            target_depth = _depth_at_ras_mm(fit_end, head_depth, ras_to_ijk_fn)
            depth_span = 0.0
            if entry_depth is not None and target_depth is not None:
                depth_span = float(target_depth - entry_depth)
            line_score = (
                2.5 * inside_fraction
                + 0.045 * float(pts.shape[0])
                + 0.02 * float(np.linalg.norm(fit_end - fit_start))
                + 0.01 * max(0.0, depth_span)
                - 0.8 * rms
            )
            if entry_depth is not None and entry_depth > float(cfg.get("min_metal_depth_mm", 5.0)) + float(cfg.get("start_zone_window_mm", 10.0)) + 4.0:
                line_score -= 2.0

            refined.append(
                {
                    "name": str(line.get("name") or f"S{idx:02d}"),
                    "start_ras": [float(v) for v in fit_start],
                    "end_ras": [float(v) for v in fit_end],
                    "length_mm": float(np.linalg.norm(fit_end - fit_start)),
                    "inlier_count": int(pts.shape[0]),
                    "support_weight": float(pts.shape[0]),
                    "rms_mm": float(rms),
                    "inside_fraction": float(inside_fraction),
                    "depth_span_mm": float(depth_span),
                    "entry_depth_mm": None if entry_depth is None else float(entry_depth),
                    "target_depth_mm": None if target_depth is None else float(target_depth),
                    "selection_score": float(line_score),
                }
            )
        return refined

    def _select_lines(self, lines: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
        angle_thresh = float(cfg.get("selection_nms_angle_deg", 12.0))
        line_thresh = float(cfg.get("selection_nms_line_distance_mm", 3.0))
        mid_thresh = float(cfg.get("selection_nms_midpoint_mm", 8.0))
        min_support = float(cfg.get("selection_min_support_weight", 8.0))
        min_inside = float(cfg.get("selection_min_inside_fraction", 0.55))
        max_length = float(cfg.get("selection_max_length_mm", 95.0))
        min_length = float(cfg.get("selection_min_length_mm", max(15.0, float(cfg.get("min_length_mm", 20.0)) * 0.75)))
        min_depth_span = float(cfg.get("selection_min_depth_span_mm", 10.0))
        target_count = cfg.get("selection_target_count")
        try:
            target_count = None if target_count in (None, "", 0) else int(target_count)
        except Exception:
            target_count = None

        ordered = sorted(
            lines,
            key=lambda ln: (
                float(ln.get("selection_score", 0.0)),
                float(ln.get("inside_fraction", 0.0)),
                float(ln.get("support_weight", 0.0)),
                float(ln.get("depth_span_mm", 0.0)),
                -float(ln.get("rms_mm", 999.0)),
            ),
            reverse=True,
        )

        kept: list[dict[str, Any]] = []
        for line in ordered:
            if float(line.get("support_weight", 0.0)) < min_support:
                continue
            if float(line.get("inside_fraction", 0.0)) < min_inside:
                continue
            if float(line.get("length_mm", 0.0)) < min_length:
                continue
            if float(line.get("length_mm", 0.0)) > max_length:
                continue
            if float(line.get("depth_span_mm", 0.0)) < min_depth_span:
                continue
            s0 = np.asarray(line.get("start_ras", [0.0, 0.0, 0.0]), dtype=float)
            e0 = np.asarray(line.get("end_ras", [0.0, 0.0, 0.0]), dtype=float)
            d0 = _normalize(e0 - s0)
            m0 = _midpoint(s0, e0)
            duplicate = False
            for kept_line in kept:
                s1 = np.asarray(kept_line.get("start_ras", [0.0, 0.0, 0.0]), dtype=float)
                e1 = np.asarray(kept_line.get("end_ras", [0.0, 0.0, 0.0]), dtype=float)
                d1 = _normalize(e1 - s1)
                m1 = _midpoint(s1, e1)
                angle = float(np.degrees(np.arccos(np.clip(abs(float(np.dot(d0, d1))), 0.0, 1.0))))
                line_dist = _line_distance(m0, d0, m1, d1)
                mid_dist = float(np.linalg.norm(m0 - m1))
                if angle <= angle_thresh and line_dist <= line_thresh and mid_dist <= mid_thresh:
                    duplicate = True
                    break
            if duplicate:
                continue
            kept.append(line)
            if target_count is not None and len(kept) >= target_count:
                break

        return kept

    def run(self, ctx: DetectionContext) -> DetectionResult:
        t_start = time.perf_counter()
        result = self.make_result(ctx)
        diagnostics = self.diagnostics(result)
        cfg = self._config(ctx)

        try:
            raw = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="proposal_detection",
                fn=lambda: run_detection(
                    arr_kji=np.asarray(ctx["arr_kji"]),
                    spacing_xyz=tuple(ctx.get("spacing_xyz") or (1.0, 1.0, 1.0)),
                    threshold=float(cfg.get("threshold", 1800.0)),
                    ijk_kji_to_ras_fn=ctx["ijk_kji_to_ras_fn"],
                    ras_to_ijk_fn=ctx["ras_to_ijk_fn"],
                    center_ras=(ctx.get("center_ras") if ctx.get("center_ras") is not None else [0.0, 0.0, 0.0]),
                    max_points=int(self._proposal_config(cfg)["max_points"] if "max_points" in self._proposal_config(cfg) else cfg.get("max_points", 300000)),
                    max_lines=int(self._proposal_config(cfg)["max_lines"]),
                    inlier_radius_mm=float(cfg.get("inlier_radius_mm", 1.2)),
                    min_length_mm=float(cfg.get("min_length_mm", 20.0)),
                    min_inliers=int(self._proposal_config(cfg)["min_inliers"]),
                    ransac_iterations=int(cfg.get("ransac_iterations", 240)),
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
                    min_metal_depth_mm=float(cfg.get("min_metal_depth_mm", 5.0)),
                    max_metal_depth_mm=float(cfg.get("max_metal_depth_mm", 220.0)),
                    start_zone_window_mm=float(cfg.get("start_zone_window_mm", 10.0)),
                    candidate_mode="voxel",
                    enable_rescue_pass=bool(cfg.get("enable_rescue_pass", True)),
                    rescue_min_inliers_scale=float(cfg.get("rescue_min_inliers_scale", 0.6)),
                    rescue_max_lines=int(cfg.get("rescue_max_lines", 6)),
                    models_by_id=((ctx.get("extras") or {}).get("models_by_id") if isinstance(ctx.get("extras"), dict) else None),
                    min_model_score=cfg.get("min_model_score"),
                    include_debug_masks=False,
                ),
            )

            refined = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="geometry_refine",
                fn=lambda: self._refine_lines(raw, ctx, cfg),
            )
            selected = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="selection",
                fn=lambda: self._select_lines(refined, cfg),
            )

            trajectories = []
            for idx, line in enumerate(selected, start=1):
                trajectories.append(
                    {
                        "name": str(line.get("name") or f"S{idx:02d}"),
                        "start_ras": [float(v) for v in line.get("start_ras", [0.0, 0.0, 0.0])],
                        "end_ras": [float(v) for v in line.get("end_ras", [0.0, 0.0, 0.0])],
                        "length_mm": float(line.get("length_mm", 0.0)),
                        "confidence": float(min(1.0, max(0.0, float(line.get("selection_score", 0.0)) / 10.0))),
                        "support_count": int(line.get("inlier_count", 0)),
                        "params": {
                            "rms_mm": float(line.get("rms_mm", 0.0)),
                            "inside_fraction": float(line.get("inside_fraction", 0.0)),
                            "depth_span_mm": float(line.get("depth_span_mm", 0.0)),
                            "entry_depth_mm": line.get("entry_depth_mm"),
                            "target_depth_mm": line.get("target_depth_mm"),
                            "selection_score": float(line.get("selection_score", 0.0)),
                        },
                    }
                )
            result["trajectories"] = trajectories
            result["contacts"] = []
            result["warnings"].append("contact_detection_not_implemented")

            diagnostics.set_count("candidate_points_total", int(raw.get("candidate_points_total", 0)))
            diagnostics.set_count("candidate_points_after_depth", int(raw.get("candidate_points_after_depth", 0)))
            diagnostics.set_count("proposal_lines_total", int(len(list(raw.get("lines") or []))))
            diagnostics.set_count("geometry_refined_lines", int(len(refined)))
            diagnostics.set_count("final_lines_kept", int(len(selected)))
            diagnostics.set_extra("selection_target_count", cfg.get("selection_target_count"))
            diagnostics.set_extra("proposal_min_inliers", int(self._proposal_config(cfg)["min_inliers"]))
            diagnostics.set_extra("proposal_max_lines", int(self._proposal_config(cfg)["max_lines"]))
            diagnostics.note("shank_axis_v1 optimizes shaft geometry only")

            writer = self.get_artifact_writer(ctx, result)
            line_rows = [
                [
                    str(line.get("name", "")),
                    float(line.get("selection_score", 0.0)),
                    float(line.get("length_mm", 0.0)),
                    float(line.get("inside_fraction", 0.0)),
                    float(line.get("rms_mm", 0.0)),
                ]
                for line in refined
            ]
            sel_path = writer.write_csv_rows(
                "shank_axis_candidates.csv",
                ["name", "selection_score", "length_mm", "inside_fraction", "rms_mm"],
                line_rows,
            )
            add_artifact(
                result["artifacts"],
                kind="candidate_csv",
                path=sel_path,
                description="Refined shank-axis candidates before final selection",
                stage="selection",
            )
            result["artifacts"].extend(
                write_standard_artifacts(
                    writer,
                    result,
                    blobs=[],
                    pipeline_payload={
                        "pipeline_id": self.pipeline_id,
                        "pipeline_version": self.pipeline_version,
                        "proposal_summary": {
                            "raw_line_count": int(len(list(raw.get("lines") or []))),
                            "refined_line_count": int(len(refined)),
                            "selected_line_count": int(len(selected)),
                        },
                    },
                )
            )
        except Exception as exc:
            self.fail(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage=str(getattr(exc, "stage", "pipeline")),
                exc=exc,
            )

        return self.finalize(result, diagnostics, t_start)
