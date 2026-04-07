"""Corridor-clustering shank detector from gated metal voxels.

This pipeline builds local orientation estimates from gated metal voxels, turns
high-anisotropy support points into corridor proposals, then clusters similar
proposals into one fitted shaft per geometric corridor.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from shank_core.masking import build_preview_masks

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
    return val if np.isfinite(val) else None


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(diff, axis=2)


def _weighted_pca_line(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    order = np.argsort(evals)
    evals = np.asarray(evals[order], dtype=float)
    axis = _normalize(evecs[:, order[-1]])
    return center.astype(float), axis.astype(float), evals.astype(float)


def _line_distances(points: np.ndarray, p0: np.ndarray, direction_unit: np.ndarray) -> np.ndarray:
    rel = points - p0.reshape(1, 3)
    t = rel @ direction_unit.reshape(3)
    closest = p0.reshape(1, 3) + np.outer(t, direction_unit.reshape(3))
    return np.linalg.norm(points - closest, axis=1)


def _projected_endpoints(points: np.ndarray, center: np.ndarray, axis: np.ndarray, lo_q: float, hi_q: float) -> tuple[np.ndarray, np.ndarray]:
    proj = (points - center.reshape(1, 3)) @ axis.reshape(3)
    lo = float(np.quantile(proj, lo_q))
    hi = float(np.quantile(proj, hi_q))
    start = center + axis * lo
    end = center + axis * hi
    return start.astype(float), end.astype(float)


def _orient_shallow_to_deep(start: np.ndarray, end: np.ndarray, depth_kji: np.ndarray | None, ras_to_ijk_fn, center_ras: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d0 = _depth_at_ras_mm(start, depth_kji, ras_to_ijk_fn)
    d1 = _depth_at_ras_mm(end, depth_kji, ras_to_ijk_fn)
    if d0 is not None and d1 is not None and abs(float(d0) - float(d1)) > 1e-3:
        return (start, end) if float(d0) <= float(d1) else (end, start)
    c0 = float(np.linalg.norm(start - center_ras))
    c1 = float(np.linalg.norm(end - center_ras))
    return (start, end) if c0 >= c1 else (end, start)


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


def _line_distance(p0: np.ndarray, d0: np.ndarray, p1: np.ndarray, d1: np.ndarray) -> float:
    u = _normalize(d0)
    v = _normalize(d1)
    w0 = p0 - p1
    c = np.cross(u, v)
    cn = float(np.linalg.norm(c))
    if cn <= 1e-6:
        return float(np.linalg.norm(np.cross(w0, u)))
    return float(abs(np.dot(w0, c)) / cn)


class ShankClusterV1Pipeline(BaseDetectionPipeline):
    pipeline_id = "shank_cluster_v1"
    display_name = "Shank Cluster v1"
    pipeline_version = "1.0.0"

    def _build_support(self, ctx: DetectionContext, cfg: dict[str, Any]) -> dict[str, Any]:
        arr_kji = np.asarray(ctx["arr_kji"])
        spacing_xyz = tuple(ctx.get("spacing_xyz") or (1.0, 1.0, 1.0))
        preview = build_preview_masks(
            arr_kji=arr_kji,
            spacing_xyz=spacing_xyz,
            threshold=float(cfg.get("threshold", 1800.0)),
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
            include_debug_masks=False,
        )
        ijk_kji = np.asarray(preview.get("in_mask_ijk_kji") if preview.get("in_mask_ijk_kji") is not None else [], dtype=float).reshape(-1, 3)
        pts_ras = np.asarray(ctx["ijk_kji_to_ras_fn"](ijk_kji), dtype=float).reshape(-1, 3) if ijk_kji.size else np.zeros((0, 3), dtype=float)
        max_points = int(cfg.get("max_points", 5000))
        if pts_ras.shape[0] > max_points:
            order = np.lexsort((pts_ras[:, 2], pts_ras[:, 1], pts_ras[:, 0]))
            stride = max(1, int(math.ceil(float(len(order)) / float(max_points))))
            keep = order[::stride][:max_points]
            ijk_kji = ijk_kji[keep]
            pts_ras = pts_ras[keep]
        return {
            "preview": preview,
            "ijk_kji": ijk_kji,
            "points_ras": pts_ras,
            "depth_map_kji": preview.get("head_distance_map_kji"),
            "gating_mask_kji": preview.get("gating_mask_kji", preview.get("head_mask_kji")),
        }

    def _local_orientation(self, points: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = points.shape[0]
        if n == 0:
            return np.zeros((0, 3)), np.zeros((0,)), np.zeros((0,), dtype=int)
        radius = float(cfg.get("seed_neighbor_radius_mm", 4.5))
        min_neighbors = int(cfg.get("seed_min_neighbors", 6))
        dist = _pairwise_distances(points)
        axes = np.zeros((n, 3), dtype=float)
        anis = np.zeros((n,), dtype=float)
        counts = np.zeros((n,), dtype=int)
        for i in range(n):
            nbr = np.where(dist[i] <= radius)[0]
            counts[i] = int(nbr.size)
            if nbr.size < min_neighbors:
                continue
            _, axis, evals = _weighted_pca_line(points[nbr], np.ones((nbr.size,), dtype=float))
            denom = float(max(1e-6, evals[-1]))
            anisotropy = float(max(0.0, 1.0 - (evals[0] + evals[1]) / (2.0 * denom)))
            axes[i] = axis
            anis[i] = anisotropy
        return axes, anis, counts

    def _seed_candidates(self, points: np.ndarray, axes: np.ndarray, anis: np.ndarray, counts: np.ndarray, support: dict[str, Any], ctx: DetectionContext, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        depth_map = support.get("depth_map_kji")
        ras_to_ijk_fn = ctx.get("ras_to_ijk_fn")
        seeds = []
        for i in range(points.shape[0]):
            if counts[i] < int(cfg.get("seed_min_neighbors", 6)):
                continue
            if float(anis[i]) < float(cfg.get("seed_min_anisotropy", 0.65)):
                continue
            depth = _depth_at_ras_mm(points[i], depth_map, ras_to_ijk_fn)
            depth_boost = 1.0 + 0.01 * float(depth if depth is not None else 0.0)
            score = float(anis[i]) * float(counts[i]) * depth_boost
            seeds.append({"seed_index": int(i), "point": points[i], "axis": axes[i], "score": score})
        seeds.sort(key=lambda s: float(s["score"]), reverse=True)
        kept: list[dict[str, Any]] = []
        for seed in seeds:
            duplicate = False
            for prev in kept:
                angle = float(np.degrees(np.arccos(np.clip(abs(float(np.dot(_normalize(seed["axis"]), _normalize(prev["axis"])))), 0.0, 1.0))))
                line_dist = _line_distance(np.asarray(seed["point"], dtype=float), np.asarray(seed["axis"], dtype=float), np.asarray(prev["point"], dtype=float), np.asarray(prev["axis"], dtype=float))
                if angle <= float(cfg.get("seed_nms_angle_deg", 10.0)) and line_dist <= float(cfg.get("seed_nms_line_distance_mm", 2.5)):
                    duplicate = True
                    break
            if not duplicate:
                kept.append(seed)
            if len(kept) >= int(cfg.get("max_seed_count", 80)):
                break
        return kept

    def _proposal_from_seed(self, seed: dict[str, Any], points: np.ndarray, axes: np.ndarray, anis: np.ndarray, support: dict[str, Any], ctx: DetectionContext, cfg: dict[str, Any]) -> dict[str, Any] | None:
        center = np.asarray(seed["point"], dtype=float)
        axis = _normalize(np.asarray(seed["axis"], dtype=float))
        corridor_radius = float(cfg.get("corridor_radius_mm", 1.8))
        min_axis_agree = float(cfg.get("corridor_min_axis_agree", 0.7))
        dist = _line_distances(points, center, axis)
        axis_dot = np.sum(axes * axis.reshape(1, 3), axis=1)
        axis_ok = (np.abs(axis_dot) >= min_axis_agree) | (anis <= 1e-6)
        chosen = (dist <= corridor_radius) & axis_ok
        min_support = int(cfg.get("corridor_min_support_points", max(6, int(cfg.get("min_inliers", 6)))))
        if int(np.count_nonzero(chosen)) < min_support:
            return None
        chosen_points = points[chosen]
        weights = np.maximum(0.1, anis[chosen]) * np.exp(-0.5 * (dist[chosen] / max(corridor_radius, 1e-3)) ** 2)
        center, axis, _evals = _weighted_pca_line(chosen_points, weights)
        start, end = _projected_endpoints(
            chosen_points,
            center,
            axis,
            float(cfg.get("endpoint_lo_quantile", 0.05)),
            float(cfg.get("endpoint_hi_quantile", 0.95)),
        )
        center_ras = np.asarray(ctx.get("center_ras") if ctx.get("center_ras") is not None else [0.0, 0.0, 0.0], dtype=float)
        start, end = _orient_shallow_to_deep(start, end, support.get("depth_map_kji"), ctx.get("ras_to_ijk_fn"), center_ras)
        axis = _normalize(end - start)
        rms = float(np.sqrt(np.mean(_line_distances(chosen_points, center, axis) ** 2))) if chosen_points.shape[0] else 0.0
        inside_fraction = _segment_inside_fraction(start, end, support.get("gating_mask_kji"), ctx.get("ras_to_ijk_fn"), step_mm=1.0)
        entry_depth = _depth_at_ras_mm(start, support.get("depth_map_kji"), ctx.get("ras_to_ijk_fn"))
        target_depth = _depth_at_ras_mm(end, support.get("depth_map_kji"), ctx.get("ras_to_ijk_fn"))
        depth_span = 0.0 if entry_depth is None or target_depth is None else float(target_depth - entry_depth)
        length_mm = float(np.linalg.norm(end - start))
        if length_mm < float(cfg.get("min_length_mm", 20.0)):
            return None
        if depth_span < float(cfg.get("corridor_min_depth_span_mm", 10.0)):
            return None
        score = (0.08 * chosen_points.shape[0]) + (0.03 * length_mm) + (0.03 * depth_span) + (2.0 * inside_fraction) - rms
        return {
            "name": f"C{int(seed['seed_index']):03d}",
            "start_ras": [float(v) for v in start],
            "end_ras": [float(v) for v in end],
            "length_mm": length_mm,
            "support_weight": float(chosen_points.shape[0]),
            "inlier_count": int(chosen_points.shape[0]),
            "inside_fraction": float(inside_fraction),
            "depth_span_mm": float(depth_span),
            "entry_depth_mm": None if entry_depth is None else float(entry_depth),
            "target_depth_mm": None if target_depth is None else float(target_depth),
            "rms_mm": float(rms),
            "selection_score": float(score),
            "assigned_global_indices": np.where(chosen)[0].tolist(),
        }

    def _cluster_proposals(self, proposals: list[dict[str, Any]], points: np.ndarray, support: dict[str, Any], ctx: DetectionContext, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        ordered = sorted(proposals, key=lambda ln: float(ln.get("selection_score", 0.0)), reverse=True)
        groups: list[list[dict[str, Any]]] = []
        for prop in ordered:
            s0 = np.asarray(prop["start_ras"], dtype=float)
            e0 = np.asarray(prop["end_ras"], dtype=float)
            d0 = _normalize(e0 - s0)
            mid0 = 0.5 * (s0 + e0)
            placed = False
            for grp in groups:
                ref = grp[0]
                s1 = np.asarray(ref["start_ras"], dtype=float)
                e1 = np.asarray(ref["end_ras"], dtype=float)
                d1 = _normalize(e1 - s1)
                mid1 = 0.5 * (s1 + e1)
                angle = float(np.degrees(np.arccos(np.clip(abs(float(np.dot(d0, d1))), 0.0, 1.0))))
                line_dist = _line_distance(mid0, d0, mid1, d1)
                mid_dist = float(np.linalg.norm(mid0 - mid1))
                if angle <= float(cfg.get("cluster_angle_deg", 10.0)) and line_dist <= float(cfg.get("cluster_line_distance_mm", 3.0)) and mid_dist <= float(cfg.get("cluster_midpoint_distance_mm", 12.0)):
                    grp.append(prop)
                    placed = True
                    break
            if not placed:
                groups.append([prop])

        merged: list[dict[str, Any]] = []
        for grp in groups:
            idx = sorted({int(i) for item in grp for i in list(item.get("assigned_global_indices") or [])})
            if len(idx) < int(cfg.get("cluster_min_support_points", max(6, int(cfg.get("min_inliers", 6))))):
                continue
            chosen_points = points[np.asarray(idx, dtype=int)]
            weights = np.ones((chosen_points.shape[0],), dtype=float)
            center, axis, _ = _weighted_pca_line(chosen_points, weights)
            start, end = _projected_endpoints(chosen_points, center, axis, float(cfg.get("endpoint_lo_quantile", 0.05)), float(cfg.get("endpoint_hi_quantile", 0.95)))
            center_ras = np.asarray(ctx.get("center_ras") if ctx.get("center_ras") is not None else [0.0, 0.0, 0.0], dtype=float)
            start, end = _orient_shallow_to_deep(start, end, support.get("depth_map_kji"), ctx.get("ras_to_ijk_fn"), center_ras)
            axis = _normalize(end - start)
            rms = float(np.sqrt(np.mean(_line_distances(chosen_points, center, axis) ** 2)))
            inside_fraction = _segment_inside_fraction(start, end, support.get("gating_mask_kji"), ctx.get("ras_to_ijk_fn"), step_mm=1.0)
            entry_depth = _depth_at_ras_mm(start, support.get("depth_map_kji"), ctx.get("ras_to_ijk_fn"))
            target_depth = _depth_at_ras_mm(end, support.get("depth_map_kji"), ctx.get("ras_to_ijk_fn"))
            depth_span = 0.0 if entry_depth is None or target_depth is None else float(target_depth - entry_depth)
            length_mm = float(np.linalg.norm(end - start))
            score = (0.08 * chosen_points.shape[0]) + (0.03 * length_mm) + (0.03 * depth_span) + (2.0 * inside_fraction) - rms
            merged.append({
                "name": str(grp[0].get("name")),
                "start_ras": [float(v) for v in start],
                "end_ras": [float(v) for v in end],
                "length_mm": length_mm,
                "support_weight": float(chosen_points.shape[0]),
                "inlier_count": int(chosen_points.shape[0]),
                "inside_fraction": float(inside_fraction),
                "depth_span_mm": float(depth_span),
                "entry_depth_mm": None if entry_depth is None else float(entry_depth),
                "target_depth_mm": None if target_depth is None else float(target_depth),
                "rms_mm": float(rms),
                "selection_score": float(score),
                "assigned_global_indices": idx,
            })
        return merged

    def _select_final(self, lines: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
        ordered = sorted(lines, key=lambda ln: float(ln.get("selection_score", 0.0)), reverse=True)
        kept = []
        target_count = cfg.get("selection_target_count")
        try:
            target_count = None if target_count in (None, "", 0) else int(target_count)
        except Exception:
            target_count = None
        for line in ordered:
            s0 = np.asarray(line["start_ras"], dtype=float)
            e0 = np.asarray(line["end_ras"], dtype=float)
            d0 = _normalize(e0 - s0)
            duplicate = False
            for prev in kept:
                s1 = np.asarray(prev["start_ras"], dtype=float)
                e1 = np.asarray(prev["end_ras"], dtype=float)
                d1 = _normalize(e1 - s1)
                angle = float(np.degrees(np.arccos(np.clip(abs(float(np.dot(d0, d1))), 0.0, 1.0))))
                line_dist = _line_distance(0.5 * (s0 + e0), d0, 0.5 * (s1 + e1), d1)
                if angle <= float(cfg.get("selection_nms_angle_deg", 8.0)) and line_dist <= float(cfg.get("selection_nms_line_distance_mm", 2.5)):
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
            if "arr_kji" not in ctx or "ijk_kji_to_ras_fn" not in ctx:
                result["trajectories"] = []
                result["contacts"] = []
                result["warnings"].append("shank_cluster_v1 missing volume context; returning empty result")
                diagnostics.note("shank_cluster_v1 requires arr_kji and ijk_kji_to_ras_fn for detection")
                return self.finalize(result, diagnostics, t_start)

            support = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="support_build", fn=lambda: self._build_support(ctx, cfg))
            points = np.asarray(support.get("points_ras") if support.get("points_ras") is not None else [], dtype=float).reshape(-1, 3)
            if points.shape[0] == 0:
                result["trajectories"] = []
                result["contacts"] = []
                return self.finalize(result, diagnostics, t_start)

            axes, anis, counts = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="local_orientation", fn=lambda: self._local_orientation(points, cfg))
            seeds = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="seed_generation", fn=lambda: self._seed_candidates(points, axes, anis, counts, support, ctx, cfg))
            proposals = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="corridor_proposals",
                fn=lambda: [p for p in (self._proposal_from_seed(seed, points, axes, anis, support, ctx, cfg) for seed in seeds) if p is not None],
            )
            clustered = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="proposal_clustering", fn=lambda: self._cluster_proposals(proposals, points, support, ctx, cfg))
            selected = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="selection", fn=lambda: self._select_final(clustered, cfg))

            result["trajectories"] = [
                {
                    "name": str(line.get("name") or f"C{idx:02d}"),
                    "start_ras": list(line.get("start_ras", [0.0, 0.0, 0.0])),
                    "end_ras": list(line.get("end_ras", [0.0, 0.0, 0.0])),
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
                for idx, line in enumerate(selected, start=1)
            ]
            result["contacts"] = []
            result["warnings"].append("contact_detection_not_implemented")

            diagnostics.set_count("candidate_points_total", int(points.shape[0]))
            diagnostics.set_count("seed_count", int(len(seeds)))
            diagnostics.set_count("proposal_count", int(len(proposals)))
            diagnostics.set_count("clustered_count", int(len(clustered)))
            diagnostics.set_count("final_lines_kept", int(len(selected)))
            diagnostics.note("shank_cluster_v1 clusters seed corridors before final shaft fitting")

            writer = self.get_artifact_writer(ctx, result)
            seed_rows = [[int(seed["seed_index"]), float(seed["score"])] for seed in seeds]
            seed_path = writer.write_csv_rows("shank_cluster_seeds.csv", ["seed_index", "score"], seed_rows)
            add_artifact(result["artifacts"], kind="seed_csv", path=seed_path, description="Corridor clustering seeds", stage="seed_generation")
            result["artifacts"].extend(
                write_standard_artifacts(
                    writer,
                    result,
                    blobs=[],
                    pipeline_payload={
                        "pipeline_id": self.pipeline_id,
                        "pipeline_version": self.pipeline_version,
                        "counts": {
                            "points": int(points.shape[0]),
                            "seeds": int(len(seeds)),
                            "proposals": int(len(proposals)),
                            "clustered": int(len(clustered)),
                            "selected": int(len(selected)),
                        },
                    },
                )
            )
        except Exception as exc:
            self.fail(ctx=ctx, result=result, diagnostics=diagnostics, stage=str(getattr(exc, "stage", "pipeline")), exc=exc)

        return self.finalize(result, diagnostics, t_start)
