"""Segment-stitching shank detector from gated metal support.

The pipeline works at an intermediate scale:
- reduce gated metal voxels into support points
- extract short local line segments by neighborhood PCA
- connect compatible segments into chains
- fit one shaft per chain

This targets the observed failure mode where support exists near the true shank
but current pipelines fail to assemble it into a coherent global trajectory.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
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


def _line_distances(points: np.ndarray, p0: np.ndarray, direction_unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rel = points - p0.reshape(1, 3)
    t = rel @ direction_unit.reshape(3)
    closest = p0.reshape(1, 3) + np.outer(t, direction_unit.reshape(3))
    return np.linalg.norm(points - closest, axis=1), t


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


class ShankStitchV1Pipeline(BaseDetectionPipeline):
    pipeline_id = "shank_stitch_v1"
    display_name = "Shank Stitch v1"
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
        return {
            "preview": preview,
            "ijk_kji": ijk_kji,
            "points_ras": pts_ras,
            "depth_map_kji": preview.get("head_distance_map_kji"),
            "gating_mask_kji": preview.get("gating_mask_kji", preview.get("head_mask_kji")),
        }

    def _reduce_support(self, points_ras: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
        if points_ras.shape[0] == 0:
            return {"points_ras": np.zeros((0, 3), dtype=float), "weights": np.zeros((0,), dtype=float), "members": []}
        grid_mm = float(cfg.get("support_grid_mm", 1.5))
        keys = np.floor(points_ras / max(grid_mm, 1e-3)).astype(int)
        buckets: dict[tuple[int, int, int], list[int]] = {}
        for idx, key in enumerate(keys):
            buckets.setdefault((int(key[0]), int(key[1]), int(key[2])), []).append(int(idx))
        pts = []
        weights = []
        members = []
        for key in sorted(buckets.keys()):
            idxs = buckets[key]
            pts.append(np.mean(points_ras[np.asarray(idxs, dtype=int)], axis=0))
            weights.append(float(len(idxs)))
            members.append(list(idxs))
        return {
            "points_ras": np.asarray(pts, dtype=float).reshape(-1, 3),
            "weights": np.asarray(weights, dtype=float).reshape(-1),
            "members": members,
        }

    def _extract_segments(self, points: np.ndarray, weights: np.ndarray, depth_map_kji: np.ndarray | None, ras_to_ijk_fn, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        n = points.shape[0]
        if n == 0:
            return []
        radius = float(cfg.get("segment_radius_mm", 4.5))
        min_neighbors = int(cfg.get("segment_min_neighbors", 5))
        min_anis = float(cfg.get("segment_min_anisotropy", 0.60))
        dist = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        segs: list[dict[str, Any]] = []
        for i in range(n):
            nbr = np.where(dist[i] <= radius)[0]
            if nbr.size < min_neighbors:
                continue
            ctr, axis, evals = _weighted_pca_line(points[nbr], weights[nbr])
            denom = float(max(1e-6, evals[-1]))
            anis = float(max(0.0, 1.0 - (evals[0] + evals[1]) / (2.0 * denom)))
            if anis < min_anis:
                continue
            proj = (points[nbr] - ctr.reshape(1, 3)) @ axis.reshape(3)
            span = float(np.max(proj) - np.min(proj))
            if span < float(cfg.get("segment_min_span_mm", 6.0)):
                continue
            depth = _depth_at_ras_mm(ctr, depth_map_kji, ras_to_ijk_fn)
            segs.append(
                {
                    "segment_id": len(segs),
                    "center": ctr,
                    "axis": axis,
                    "anisotropy": anis,
                    "span_mm": span,
                    "mass": float(np.sum(weights[nbr])),
                    "support_indices": [int(v) for v in nbr.tolist()],
                    "depth_mm": float(depth if depth is not None else 0.0),
                }
            )
        # NMS to reduce redundant local segments.
        kept: list[dict[str, Any]] = []
        ordered = sorted(segs, key=lambda s: (float(s["mass"]) * float(s["anisotropy"]) * float(s["span_mm"])), reverse=True)
        for seg in ordered:
            dup = False
            for prev in kept:
                angle = float(np.degrees(np.arccos(np.clip(abs(float(np.dot(_normalize(seg["axis"]), _normalize(prev["axis"])))), 0.0, 1.0))))
                line_dist = _line_distance(np.asarray(seg["center"], dtype=float), np.asarray(seg["axis"], dtype=float), np.asarray(prev["center"], dtype=float), np.asarray(prev["axis"], dtype=float))
                if angle <= float(cfg.get("segment_nms_angle_deg", 10.0)) and line_dist <= float(cfg.get("segment_nms_line_distance_mm", 2.0)):
                    dup = True
                    break
            if not dup:
                kept.append(seg)
            if len(kept) >= int(cfg.get("max_segments", 300)):
                break
        for idx, seg in enumerate(kept):
            seg["segment_id"] = int(idx)
        return kept

    def _build_segment_graph(self, segments: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[int, list[tuple[int, float]]]:
        adjacency: dict[int, list[tuple[int, float]]] = defaultdict(list)
        if not segments:
            return {}
        max_gap = float(cfg.get("stitch_max_gap_mm", 12.0))
        max_line_dist = float(cfg.get("stitch_max_line_distance_mm", 3.0))
        max_angle = float(cfg.get("stitch_max_angle_deg", 15.0))
        min_depth_step = float(cfg.get("stitch_min_depth_step_mm", -2.0))
        for i, seg_i in enumerate(segments):
            for j, seg_j in enumerate(segments):
                if i == j:
                    continue
                if float(seg_j["depth_mm"]) < float(seg_i["depth_mm"]) + min_depth_step:
                    continue
                axis_i = _normalize(np.asarray(seg_i["axis"], dtype=float))
                axis_j = _normalize(np.asarray(seg_j["axis"], dtype=float))
                angle = float(np.degrees(np.arccos(np.clip(abs(float(np.dot(axis_i, axis_j))), 0.0, 1.0))))
                if angle > max_angle:
                    continue
                vec = np.asarray(seg_j["center"], dtype=float) - np.asarray(seg_i["center"], dtype=float)
                gap = float(np.linalg.norm(vec))
                if gap > max_gap:
                    continue
                line_dist = _line_distance(np.asarray(seg_i["center"], dtype=float), axis_i, np.asarray(seg_j["center"], dtype=float), axis_j)
                if line_dist > max_line_dist:
                    continue
                along = float(np.dot(_normalize(vec), axis_i))
                if abs(along) < float(cfg.get("stitch_min_forward_agree", 0.55)):
                    continue
                score = float(seg_j["mass"]) + float(seg_j["span_mm"]) - 0.4 * angle - 0.5 * gap - 0.7 * line_dist
                adjacency[int(seg_i["segment_id"])].append((int(seg_j["segment_id"]), score))
        return dict(adjacency)

    def _best_chain(self, segments: list[dict[str, Any]], adjacency: dict[int, list[tuple[int, float]]], available: set[int]) -> list[int]:
        if not available:
            return []
        seg_by_id = {int(seg["segment_id"]): seg for seg in segments}
        order = sorted(list(available), key=lambda sid: float(seg_by_id[sid]["depth_mm"]))
        best_score: dict[int, float] = {}
        parent: dict[int, int] = {}
        for sid in order:
            base = float(seg_by_id[sid]["mass"]) + 0.5 * float(seg_by_id[sid]["span_mm"]) + 2.0 * float(seg_by_id[sid]["anisotropy"])
            best_score[sid] = max(best_score.get(sid, -float("inf")), base)
            for nbr, edge_score in adjacency.get(sid, []):
                if nbr not in available:
                    continue
                cand = float(best_score[sid]) + float(edge_score)
                if cand > best_score.get(nbr, -float("inf")):
                    best_score[nbr] = cand
                    parent[nbr] = sid
        if not best_score:
            return []
        end = max(best_score.items(), key=lambda kv: float(kv[1]))[0]
        chain = [int(end)]
        cur = int(end)
        seen = {cur}
        while cur in parent and parent[cur] not in seen:
            cur = int(parent[cur])
            chain.append(cur)
            seen.add(cur)
        chain.reverse()
        return chain

    def _fit_chain(self, chain_ids: list[int], segments: list[dict[str, Any]], points: np.ndarray, weights: np.ndarray, support: dict[str, Any], ctx: DetectionContext, cfg: dict[str, Any]) -> dict[str, Any] | None:
        if not chain_ids:
            return None
        seg_by_id = {int(seg["segment_id"]): seg for seg in segments}
        support_idx = sorted({int(i) for sid in chain_ids for i in list(seg_by_id[int(sid)]["support_indices"])})
        if len(support_idx) < int(cfg.get("min_inliers", 6)):
            return None
        fit_points = points[np.asarray(support_idx, dtype=int)]
        fit_weights = weights[np.asarray(support_idx, dtype=int)]
        ctr, axis, _ = _weighted_pca_line(fit_points, fit_weights)
        start, end = _projected_endpoints(fit_points, ctr, axis, float(cfg.get("endpoint_lo_quantile", 0.05)), float(cfg.get("endpoint_hi_quantile", 0.95)))
        center_ras = np.asarray(ctx.get("center_ras") if ctx.get("center_ras") is not None else [0.0, 0.0, 0.0], dtype=float)
        start, end = _orient_shallow_to_deep(start, end, support.get("depth_map_kji"), ctx.get("ras_to_ijk_fn"), center_ras)
        axis = _normalize(end - start)
        residuals, proj = _line_distances(fit_points, ctr, axis)
        rms = float(np.sqrt(np.average(residuals ** 2, weights=np.maximum(fit_weights, 1e-6))))
        length_mm = float(np.linalg.norm(end - start))
        if length_mm < float(cfg.get("min_length_mm", 20.0)):
            return None
        entry_depth = _depth_at_ras_mm(start, support.get("depth_map_kji"), ctx.get("ras_to_ijk_fn"))
        target_depth = _depth_at_ras_mm(end, support.get("depth_map_kji"), ctx.get("ras_to_ijk_fn"))
        depth_span = 0.0 if entry_depth is None or target_depth is None else float(target_depth - entry_depth)
        if depth_span < float(cfg.get("min_depth_span_mm", 10.0)):
            return None
        inside_fraction = _segment_inside_fraction(start, end, support.get("gating_mask_kji"), ctx.get("ras_to_ijk_fn"), step_mm=1.0)
        bins = max(3, int(math.ceil(length_mm / max(float(cfg.get("coverage_bin_mm", 6.0)), 1e-3))))
        hist, _ = np.histogram(np.clip(proj, 0.0, length_mm), bins=bins, range=(0.0, max(1e-6, length_mm)))
        coverage_ratio = float(np.count_nonzero(hist > 0)) / float(max(1, bins))
        full = np.concatenate([[0.0], np.sort(np.clip(proj, 0.0, length_mm)), [length_mm]])
        largest_gap = float(np.max(np.diff(full))) if full.size >= 2 else float("inf")
        score = (1.0 * float(np.sum(fit_weights))) + (0.04 * length_mm) + (2.0 * coverage_ratio) + (2.0 * inside_fraction) + (0.02 * depth_span) - (1.5 * rms) - (0.05 * largest_gap)
        return {
            "name": f"S{chain_ids[0]:03d}",
            "start_ras": [float(v) for v in start],
            "end_ras": [float(v) for v in end],
            "length_mm": float(length_mm),
            "support_weight": float(np.sum(fit_weights)),
            "inlier_count": int(len(support_idx)),
            "inside_fraction": float(inside_fraction),
            "depth_span_mm": float(depth_span),
            "entry_depth_mm": None if entry_depth is None else float(entry_depth),
            "target_depth_mm": None if target_depth is None else float(target_depth),
            "rms_mm": float(rms),
            "coverage_ratio": float(coverage_ratio),
            "largest_gap_mm": float(largest_gap),
            "selection_score": float(score),
            "chain_segment_ids": [int(v) for v in chain_ids],
            "assigned_support_indices": support_idx,
        }

    def _extract_chains(self, segments: list[dict[str, Any]], adjacency: dict[int, list[tuple[int, float]]], points: np.ndarray, weights: np.ndarray, support: dict[str, Any], ctx: DetectionContext, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        available = {int(seg["segment_id"]) for seg in segments}
        lines: list[dict[str, Any]] = []
        max_lines = int(cfg.get("max_lines", 30))
        min_chain_len = int(cfg.get("chain_min_segments", 2))
        for _ in range(max_lines):
            chain = self._best_chain(segments, adjacency, available)
            if len(chain) < min_chain_len:
                break
            fitted = self._fit_chain(chain, segments, points, weights, support, ctx, cfg)
            if fitted is None:
                for sid in chain:
                    available.discard(int(sid))
                continue
            if float(fitted["selection_score"]) < float(cfg.get("min_selection_score", 8.0)):
                break
            lines.append(fitted)
            for sid in chain:
                available.discard(int(sid))
        return lines

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
            dup = False
            for prev in kept:
                s1 = np.asarray(prev["start_ras"], dtype=float)
                e1 = np.asarray(prev["end_ras"], dtype=float)
                d1 = _normalize(e1 - s1)
                angle = float(np.degrees(np.arccos(np.clip(abs(float(np.dot(d0, d1))), 0.0, 1.0))))
                line_dist = _line_distance(0.5 * (s0 + e0), d0, 0.5 * (s1 + e1), d1)
                if angle <= float(cfg.get("selection_nms_angle_deg", 8.0)) and line_dist <= float(cfg.get("selection_nms_line_distance_mm", 2.5)):
                    dup = True
                    break
            if dup:
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
                result["warnings"].append("shank_stitch_v1 missing volume context; returning empty result")
                diagnostics.note("shank_stitch_v1 requires arr_kji and ijk_kji_to_ras_fn for detection")
                return self.finalize(result, diagnostics, t_start)

            support = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="support_build", fn=lambda: self._build_support(ctx, cfg))
            raw_points = np.asarray(support.get("points_ras") if support.get("points_ras") is not None else [], dtype=float).reshape(-1, 3)
            if raw_points.shape[0] == 0:
                result["trajectories"] = []
                result["contacts"] = []
                return self.finalize(result, diagnostics, t_start)

            reduced = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="support_reduce", fn=lambda: self._reduce_support(raw_points, cfg))
            points = np.asarray(reduced["points_ras"], dtype=float).reshape(-1, 3)
            weights = np.asarray(reduced["weights"], dtype=float).reshape(-1)
            segments = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="segment_extract",
                fn=lambda: self._extract_segments(points, weights, support.get("depth_map_kji"), ctx.get("ras_to_ijk_fn"), cfg),
            )
            adjacency = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="segment_graph", fn=lambda: self._build_segment_graph(segments, cfg))
            chains = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="segment_stitch",
                fn=lambda: self._extract_chains(segments, adjacency, points, weights, support, ctx, cfg),
            )
            selected = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="selection", fn=lambda: self._select_final(chains, cfg))

            result["trajectories"] = [
                {
                    "name": str(line.get("name") or f"S{idx:02d}"),
                    "start_ras": list(line.get("start_ras", [0.0, 0.0, 0.0])),
                    "end_ras": list(line.get("end_ras", [0.0, 0.0, 0.0])),
                    "length_mm": float(line.get("length_mm", 0.0)),
                    "confidence": float(min(1.0, max(0.0, float(line.get("selection_score", 0.0)) / 12.0))),
                    "support_count": int(line.get("inlier_count", 0)),
                    "params": {
                        "rms_mm": float(line.get("rms_mm", 0.0)),
                        "inside_fraction": float(line.get("inside_fraction", 0.0)),
                        "depth_span_mm": float(line.get("depth_span_mm", 0.0)),
                        "entry_depth_mm": line.get("entry_depth_mm"),
                        "target_depth_mm": line.get("target_depth_mm"),
                        "coverage_ratio": float(line.get("coverage_ratio", 0.0)),
                        "largest_gap_mm": float(line.get("largest_gap_mm", 0.0)),
                        "selection_score": float(line.get("selection_score", 0.0)),
                    },
                }
                for idx, line in enumerate(selected, start=1)
            ]
            result["contacts"] = []
            result["warnings"].append("contact_detection_not_implemented")

            diagnostics.set_count("candidate_points_total", int(raw_points.shape[0]))
            diagnostics.set_count("support_points_total", int(points.shape[0]))
            diagnostics.set_count("segment_count", int(len(segments)))
            diagnostics.set_count("chain_count", int(len(chains)))
            diagnostics.set_count("final_lines_kept", int(len(selected)))
            diagnostics.note("shank_stitch_v1 stitches local shaft segments into global chains")

            writer = self.get_artifact_writer(ctx, result)
            seg_rows = [
                [
                    int(seg["segment_id"]),
                    float(seg["center"][0]),
                    float(seg["center"][1]),
                    float(seg["center"][2]),
                    float(seg["axis"][0]),
                    float(seg["axis"][1]),
                    float(seg["axis"][2]),
                    float(seg["anisotropy"]),
                    float(seg["span_mm"]),
                    float(seg["mass"]),
                ]
                for seg in segments
            ]
            seg_path = writer.write_csv_rows(
                "shank_segments.csv",
                ["segment_id", "center_x", "center_y", "center_z", "axis_x", "axis_y", "axis_z", "anisotropy", "span_mm", "mass"],
                seg_rows,
            )
            add_artifact(result["artifacts"], kind="segment_csv", path=seg_path, description="Local segment primitives", stage="segment_extract")
            result["artifacts"].extend(
                write_standard_artifacts(
                    writer,
                    result,
                    blobs=[],
                    pipeline_payload={
                        "pipeline_id": self.pipeline_id,
                        "pipeline_version": self.pipeline_version,
                        "counts": {
                            "raw_points": int(raw_points.shape[0]),
                            "support_points": int(points.shape[0]),
                            "segments": int(len(segments)),
                            "chains": int(len(chains)),
                            "selected": int(len(selected)),
                        },
                    },
                )
            )
        except Exception as exc:
            self.fail(ctx=ctx, result=result, diagnostics=diagnostics, stage=str(getattr(exc, "stage", "pipeline")), exc=exc)
        return self.finalize(result, diagnostics, t_start)
