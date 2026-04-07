"""Global line-hypothesis shank detector from gated metal voxels.

This pipeline builds a reduced support cloud from gated metal voxels, generates
deterministic global line hypotheses, scores them against the entire support set,
then greedily peels explained supports. It optimizes shaft geometry directly:
direction, deep endpoint, shallow endpoint, span, and support coverage.
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


def _line_distances(points: np.ndarray, p0: np.ndarray, direction_unit: np.ndarray) -> np.ndarray:
    rel = points - p0.reshape(1, 3)
    t = rel @ direction_unit.reshape(3)
    closest = p0.reshape(1, 3) + np.outer(t, direction_unit.reshape(3))
    return np.linalg.norm(points - closest, axis=1), t


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


class ShankHypothesisV1Pipeline(BaseDetectionPipeline):
    pipeline_id = "shank_hypothesis_v1"
    display_name = "Shank Hypothesis v1"
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

    def _downsample_support(self, ijk_kji: np.ndarray, points_ras: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
        if points_ras.shape[0] == 0:
            return {"points_ras": np.zeros((0, 3), dtype=float), "weights": np.zeros((0,), dtype=float), "members": []}
        step_mm = float(cfg.get("support_grid_mm", 1.5))
        keys = np.floor(points_ras / max(step_mm, 1e-3)).astype(int)
        buckets: dict[tuple[int, int, int], list[int]] = {}
        for idx, key in enumerate(keys):
            buckets.setdefault((int(key[0]), int(key[1]), int(key[2])), []).append(int(idx))
        pts = []
        w = []
        members = []
        for key in sorted(buckets.keys()):
            idxs = buckets[key]
            pts.append(np.mean(points_ras[np.asarray(idxs, dtype=int)], axis=0))
            w.append(float(len(idxs)))
            members.append(list(idxs))
        return {
            "points_ras": np.asarray(pts, dtype=float).reshape(-1, 3),
            "weights": np.asarray(w, dtype=float).reshape(-1),
            "members": members,
        }

    def _local_axes(self, points: np.ndarray, weights: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        n = points.shape[0]
        if n == 0:
            return np.zeros((0, 3)), np.zeros((0,))
        radius = float(cfg.get("support_neighbor_radius_mm", 5.0))
        dist = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        axes = np.zeros((n, 3), dtype=float)
        anis = np.zeros((n,), dtype=float)
        min_neighbors = int(cfg.get("support_min_neighbors", 5))
        for i in range(n):
            nbr = np.where(dist[i] <= radius)[0]
            if nbr.size < min_neighbors:
                continue
            _, axis, evals = _weighted_pca_line(points[nbr], weights[nbr])
            denom = float(max(1e-6, evals[-1]))
            anisotropy = float(max(0.0, 1.0 - (evals[0] + evals[1]) / (2.0 * denom)))
            axes[i] = axis
            anis[i] = anisotropy
        return axes, anis

    def _candidate_indices(self, points: np.ndarray, weights: np.ndarray, depth_map_kji: np.ndarray | None, ras_to_ijk_fn, cfg: dict[str, Any]) -> np.ndarray:
        if points.shape[0] == 0:
            return np.zeros((0,), dtype=int)
        depths = np.asarray([_depth_at_ras_mm(p, depth_map_kji, ras_to_ijk_fn) or 0.0 for p in points], dtype=float)
        shallow_q = float(cfg.get("hypothesis_shallow_quantile", 0.35))
        deep_q = float(cfg.get("hypothesis_deep_quantile", 0.35))
        shallow_cut = float(np.quantile(depths, shallow_q))
        deep_cut = float(np.quantile(depths, 1.0 - deep_q))
        top_w = max(1.0, float(np.quantile(weights, 0.5)))
        mask = (depths <= shallow_cut) | (depths >= deep_cut) | (weights >= top_w)
        idx = np.where(mask)[0]
        max_candidates = int(cfg.get("max_hypothesis_points", 120))
        if idx.size > max_candidates:
            order = np.argsort(weights[idx])[::-1]
            idx = idx[order[:max_candidates]]
        return np.asarray(sorted(set(int(v) for v in idx.tolist())), dtype=int)

    def _generate_hypotheses(self, points: np.ndarray, weights: np.ndarray, axes: np.ndarray, anis: np.ndarray, depth_map_kji: np.ndarray | None, ras_to_ijk_fn, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        if points.shape[0] == 0:
            return []
        cand = self._candidate_indices(points, weights, depth_map_kji, ras_to_ijk_fn, cfg)
        if cand.size == 0:
            return []
        hypotheses: list[dict[str, Any]] = []
        for i in cand.tolist():
            if float(anis[i]) >= float(cfg.get("axis_seed_min_anisotropy", 0.55)):
                hypotheses.append({"point": points[i], "axis": _normalize(axes[i]), "seed_type": "axis", "seed_indices": (int(i),)})
        depths = np.asarray([_depth_at_ras_mm(points[i], depth_map_kji, ras_to_ijk_fn) or 0.0 for i in cand.tolist()], dtype=float)
        shallow = [int(cand[i]) for i in range(len(cand)) if depths[i] <= float(np.quantile(depths, 0.4))]
        deep = [int(cand[i]) for i in range(len(cand)) if depths[i] >= float(np.quantile(depths, 0.6))]
        if not shallow:
            shallow = [int(v) for v in cand[: min(len(cand), 30)].tolist()]
        if not deep:
            deep = [int(v) for v in cand[-min(len(cand), 30) :].tolist()]
        pair_cap = int(cfg.get("max_pair_hypotheses", 300))
        count = 0
        for i in shallow:
            for j in deep:
                if i == j:
                    continue
                vec = points[j] - points[i]
                dist = float(np.linalg.norm(vec))
                if dist < float(cfg.get("pair_min_span_mm", 20.0)):
                    continue
                hypotheses.append({"point": points[i], "axis": _normalize(vec), "seed_type": "pair", "seed_indices": (int(i), int(j))})
                count += 1
                if count >= pair_cap:
                    break
            if count >= pair_cap:
                break
        # NMS
        kept: list[dict[str, Any]] = []
        for hyp in hypotheses:
            dup = False
            for prev in kept:
                angle = float(np.degrees(np.arccos(np.clip(abs(float(np.dot(_normalize(hyp["axis"]), _normalize(prev["axis"])))), 0.0, 1.0))))
                line_dist = _line_distance(np.asarray(hyp["point"], dtype=float), np.asarray(hyp["axis"], dtype=float), np.asarray(prev["point"], dtype=float), np.asarray(prev["axis"], dtype=float))
                if angle <= float(cfg.get("hypothesis_nms_angle_deg", 8.0)) and line_dist <= float(cfg.get("hypothesis_nms_line_distance_mm", 2.5)):
                    dup = True
                    break
            if not dup:
                kept.append(hyp)
            if len(kept) >= int(cfg.get("max_hypotheses", 160)):
                break
        return kept

    def _score_hypothesis(self, hyp: dict[str, Any], points: np.ndarray, weights: np.ndarray, active: np.ndarray, support: dict[str, Any], ctx: DetectionContext, cfg: dict[str, Any]) -> dict[str, Any] | None:
        active_idx = np.where(active)[0]
        if active_idx.size == 0:
            return None
        pts = points[active_idx]
        w = weights[active_idx]
        p0 = np.asarray(hyp["point"], dtype=float)
        axis = _normalize(np.asarray(hyp["axis"], dtype=float))
        d, t = _line_distances(pts, p0, axis)
        radius = float(cfg.get("fit_radius_mm", 1.6))
        inlier = d <= radius
        if int(np.count_nonzero(inlier)) < int(cfg.get("min_inliers", 6)):
            return None
        inlier_pts = pts[inlier]
        inlier_w = w[inlier] * np.exp(-0.5 * (d[inlier] / max(radius, 1e-3)) ** 2)
        center, axis, _ = _weighted_pca_line(inlier_pts, inlier_w)
        start, end, proj = _projected_endpoints(
            inlier_pts,
            center,
            axis,
            float(cfg.get("endpoint_lo_quantile", 0.03)),
            float(cfg.get("endpoint_hi_quantile", 0.97)),
        )
        center_ras = np.asarray(ctx.get("center_ras") if ctx.get("center_ras") is not None else [0.0, 0.0, 0.0], dtype=float)
        start, end = _orient_shallow_to_deep(start, end, support.get("depth_map_kji"), ctx.get("ras_to_ijk_fn"), center_ras)
        axis = _normalize(end - start)
        rms = float(np.sqrt(np.average(_line_distances(inlier_pts, center, axis)[0] ** 2, weights=np.maximum(inlier_w, 1e-6))))
        length_mm = float(np.linalg.norm(end - start))
        if length_mm < float(cfg.get("min_length_mm", 20.0)):
            return None
        inside_fraction = _segment_inside_fraction(start, end, support.get("gating_mask_kji"), ctx.get("ras_to_ijk_fn"), step_mm=1.0)
        entry_depth = _depth_at_ras_mm(start, support.get("depth_map_kji"), ctx.get("ras_to_ijk_fn"))
        target_depth = _depth_at_ras_mm(end, support.get("depth_map_kji"), ctx.get("ras_to_ijk_fn"))
        depth_span = 0.0 if entry_depth is None or target_depth is None else float(target_depth - entry_depth)
        if depth_span < float(cfg.get("min_depth_span_mm", 10.0)):
            return None
        proj_line = (inlier_pts - center.reshape(1, 3)) @ axis.reshape(3)
        bins = max(3, int(math.ceil(length_mm / max(float(cfg.get("coverage_bin_mm", 6.0)), 1e-3))))
        hist, _ = np.histogram(proj_line, bins=bins)
        occupied = int(np.count_nonzero(hist > 0))
        coverage = float(occupied) / float(max(1, bins))
        support_mass = float(np.sum(inlier_w))
        score = (1.2 * support_mass) + (0.04 * length_mm) + (1.5 * coverage) + (2.0 * inside_fraction) + (0.02 * depth_span) - (1.5 * rms)
        if entry_depth is not None:
            start_window = float(cfg.get("start_zone_window_mm", 10.0))
            ideal = float(cfg.get("min_metal_depth_mm", 5.0))
            score -= 0.1 * max(0.0, float(entry_depth) - (ideal + start_window))
        return {
            "name": f"H{len(active_idx):03d}",
            "start_ras": [float(v) for v in start],
            "end_ras": [float(v) for v in end],
            "length_mm": float(length_mm),
            "support_weight": float(support_mass),
            "inlier_count": int(np.count_nonzero(inlier)),
            "inside_fraction": float(inside_fraction),
            "depth_span_mm": float(depth_span),
            "entry_depth_mm": None if entry_depth is None else float(entry_depth),
            "target_depth_mm": None if target_depth is None else float(target_depth),
            "rms_mm": float(rms),
            "coverage_ratio": float(coverage),
            "selection_score": float(score),
            "assigned_global_indices": active_idx[inlier].tolist(),
        }

    def _extract_lines(self, hypotheses: list[dict[str, Any]], points: np.ndarray, weights: np.ndarray, support: dict[str, Any], ctx: DetectionContext, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        active = np.ones((points.shape[0],), dtype=bool)
        lines: list[dict[str, Any]] = []
        max_lines = int(cfg.get("max_lines", 30))
        for _ in range(max_lines):
            best = None
            best_score = -float("inf")
            for hyp in hypotheses:
                scored = self._score_hypothesis(hyp, points, weights, active, support, ctx, cfg)
                if scored is None:
                    continue
                score = float(scored.get("selection_score", -float("inf")))
                if score > best_score:
                    best_score = score
                    best = scored
            if best is None:
                break
            if float(best.get("selection_score", 0.0)) < float(cfg.get("min_selection_score", 8.0)):
                break
            lines.append(best)
            remove_idx = np.asarray(best.get("assigned_global_indices") or [], dtype=int).reshape(-1)
            if remove_idx.size:
                active[remove_idx] = False
            if int(np.count_nonzero(active)) < int(cfg.get("min_inliers", 6)):
                break
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
                result["warnings"].append("shank_hypothesis_v1 missing volume context; returning empty result")
                diagnostics.note("shank_hypothesis_v1 requires arr_kji and ijk_kji_to_ras_fn for detection")
                return self.finalize(result, diagnostics, t_start)

            support = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="support_build", fn=lambda: self._build_support(ctx, cfg))
            raw_points = np.asarray(support.get("points_ras") if support.get("points_ras") is not None else [], dtype=float).reshape(-1, 3)
            raw_ijk = np.asarray(support.get("ijk_kji") if support.get("ijk_kji") is not None else [], dtype=float).reshape(-1, 3)
            if raw_points.shape[0] == 0:
                result["trajectories"] = []
                result["contacts"] = []
                return self.finalize(result, diagnostics, t_start)

            reduced = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="support_reduce", fn=lambda: self._downsample_support(raw_ijk, raw_points, cfg))
            points = np.asarray(reduced["points_ras"], dtype=float).reshape(-1, 3)
            weights = np.asarray(reduced["weights"], dtype=float).reshape(-1)
            axes, anis = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="local_axes", fn=lambda: self._local_axes(points, weights, cfg))
            hypotheses = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="hypothesis_generation",
                fn=lambda: self._generate_hypotheses(points, weights, axes, anis, support.get("depth_map_kji"), ctx.get("ras_to_ijk_fn"), cfg),
            )
            lines = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="hypothesis_selection",
                fn=lambda: self._extract_lines(hypotheses, points, weights, support, ctx, cfg),
            )
            selected = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="selection", fn=lambda: self._select_final(lines, cfg))

            result["trajectories"] = [
                {
                    "name": str(line.get("name") or f"H{idx:02d}"),
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
                        "selection_score": float(line.get("selection_score", 0.0)),
                    },
                }
                for idx, line in enumerate(selected, start=1)
            ]
            result["contacts"] = []
            result["warnings"].append("contact_detection_not_implemented")

            diagnostics.set_count("candidate_points_total", int(raw_points.shape[0]))
            diagnostics.set_count("support_points_total", int(points.shape[0]))
            diagnostics.set_count("hypothesis_count", int(len(hypotheses)))
            diagnostics.set_count("proposal_count", int(len(lines)))
            diagnostics.set_count("final_lines_kept", int(len(selected)))
            diagnostics.note("shank_hypothesis_v1 uses global line hypotheses over reduced support points")

            writer = self.get_artifact_writer(ctx, result)
            hyp_rows = [[idx, str(h["seed_type"]), *[float(v) for v in h["point"]], *[float(v) for v in h["axis"]]] for idx, h in enumerate(hypotheses, start=1)]
            hyp_path = writer.write_csv_rows("shank_hypotheses.csv", ["hypothesis_index", "seed_type", "point_x", "point_y", "point_z", "axis_x", "axis_y", "axis_z"], hyp_rows)
            add_artifact(result["artifacts"], kind="hypothesis_csv", path=hyp_path, description="Global line hypotheses", stage="hypothesis_generation")
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
                            "hypotheses": int(len(hypotheses)),
                            "proposals": int(len(lines)),
                            "selected": int(len(selected)),
                        },
                    },
                )
            )
        except Exception as exc:
            self.fail(ctx=ctx, result=result, diagnostics=diagnostics, stage=str(getattr(exc, "stage", "pipeline")), exc=exc)
        return self.finalize(result, diagnostics, t_start)
