"""Graph-based shank detector from gated metal voxels.

This pipeline builds a sparse local graph over gated metal voxels using
orientation-consistent edges, extracts long path-like supports from each graph
component, and fits one shaft per recovered path.
"""

from __future__ import annotations

import heapq
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


def _projected_endpoints(points: np.ndarray, center: np.ndarray, axis: np.ndarray, lo_q: float, hi_q: float) -> tuple[np.ndarray, np.ndarray]:
    proj = (points - center.reshape(1, 3)) @ axis.reshape(3)
    lo = float(np.quantile(proj, lo_q))
    hi = float(np.quantile(proj, hi_q))
    start = center + axis * lo
    end = center + axis * hi
    return start.astype(float), end.astype(float)


def _line_distances(points: np.ndarray, p0: np.ndarray, direction_unit: np.ndarray) -> np.ndarray:
    rel = points - p0.reshape(1, 3)
    t = rel @ direction_unit.reshape(3)
    closest = p0.reshape(1, 3) + np.outer(t, direction_unit.reshape(3))
    return np.linalg.norm(points - closest, axis=1)


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


def _neighbor_offsets(radius_mm: float, spacing_xyz: tuple[float, float, float]) -> list[tuple[int, int, int]]:
    sx, sy, sz = [max(1e-3, float(v)) for v in spacing_xyz]
    ri = int(math.ceil(radius_mm / sx))
    rj = int(math.ceil(radius_mm / sy))
    rk = int(math.ceil(radius_mm / sz))
    offsets = []
    for dk in range(-rk, rk + 1):
        for dj in range(-rj, rj + 1):
            for di in range(-ri, ri + 1):
                if dk == 0 and dj == 0 and di == 0:
                    continue
                dist = math.sqrt((di * sx) ** 2 + (dj * sy) ** 2 + (dk * sz) ** 2)
                if dist <= radius_mm + 1e-6:
                    offsets.append((dk, dj, di))
    return offsets


def _dijkstra(start: int, adjacency: dict[int, list[tuple[int, float]]], allowed: set[int]) -> tuple[dict[int, float], dict[int, int]]:
    dist = {start: 0.0}
    parent: dict[int, int] = {}
    heap: list[tuple[float, int]] = [(0.0, start)]
    while heap:
        cur_d, node = heapq.heappop(heap)
        if cur_d > dist.get(node, float("inf")) + 1e-9:
            continue
        for nbr, weight in adjacency.get(node, []):
            if nbr not in allowed:
                continue
            nd = cur_d + float(weight)
            if nd + 1e-9 < dist.get(nbr, float("inf")):
                dist[nbr] = nd
                parent[nbr] = node
                heapq.heappush(heap, (nd, nbr))
    return dist, parent


def _reconstruct_path(parent: dict[int, int], start: int, end: int) -> list[int]:
    path = [end]
    cur = end
    while cur != start and cur in parent:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path if path and path[0] == start else [start]


class ShankGraphV1Pipeline(BaseDetectionPipeline):
    pipeline_id = "shank_graph_v1"
    display_name = "Shank Graph v1"
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
            "spacing_xyz": spacing_xyz,
        }

    def _local_orientation(self, ijk_kji: np.ndarray, points_ras: np.ndarray, spacing_xyz: tuple[float, float, float], cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
        n = points_ras.shape[0]
        if n == 0:
            return np.zeros((0, 3)), np.zeros((0,)), []
        radius_mm = float(cfg.get("graph_neighbor_radius_mm", 4.0))
        min_neighbors = int(cfg.get("graph_min_neighbors", 6))
        offsets = _neighbor_offsets(radius_mm, spacing_xyz)
        occupied = {tuple(int(round(float(v))) for v in ijk_kji[idx]): idx for idx in range(n)}
        axes = np.zeros((n, 3), dtype=float)
        anis = np.zeros((n,), dtype=float)
        neighborhoods: list[list[int]] = [[] for _ in range(n)]
        for idx in range(n):
            k, j, i = [int(round(float(v))) for v in ijk_kji[idx]]
            nbrs = [idx]
            for dk, dj, di in offsets:
                other = occupied.get((k + dk, j + dj, i + di))
                if other is not None:
                    nbrs.append(int(other))
            nbrs = sorted(set(nbrs))
            neighborhoods[idx] = nbrs
            if len(nbrs) < min_neighbors:
                continue
            _, axis, evals = _weighted_pca_line(points_ras[np.asarray(nbrs, dtype=int)], np.ones((len(nbrs),), dtype=float))
            denom = float(max(1e-6, evals[-1]))
            anisotropy = float(max(0.0, 1.0 - (evals[0] + evals[1]) / (2.0 * denom)))
            axes[idx] = axis
            anis[idx] = anisotropy
        return axes, anis, neighborhoods

    def _build_graph(self, ijk_kji: np.ndarray, points_ras: np.ndarray, axes: np.ndarray, anis: np.ndarray, spacing_xyz: tuple[float, float, float], cfg: dict[str, Any]) -> dict[int, list[tuple[int, float]]]:
        n = points_ras.shape[0]
        adjacency: dict[int, list[tuple[int, float]]] = defaultdict(list)
        if n == 0:
            return {}
        radius_mm = float(cfg.get("edge_radius_mm", 2.0))
        offsets = _neighbor_offsets(radius_mm, spacing_xyz)
        occupied = {tuple(int(round(float(v))) for v in ijk_kji[idx]): idx for idx in range(n)}
        min_anis = float(cfg.get("edge_min_anisotropy", 0.55))
        axis_agree = float(cfg.get("edge_min_axis_agree", 0.7))
        vector_agree = float(cfg.get("edge_min_vector_agree", 0.75))
        max_degree = int(cfg.get("edge_max_degree", 3))
        provisional: dict[int, list[tuple[float, int, float]]] = defaultdict(list)
        for idx in range(n):
            if float(anis[idx]) < min_anis:
                continue
            k, j, i = [int(round(float(v))) for v in ijk_kji[idx]]
            for dk, dj, di in offsets:
                other = occupied.get((k + dk, j + dj, i + di))
                if other is None or other <= idx:
                    continue
                if float(anis[other]) < min_anis:
                    continue
                axis_i = _normalize(axes[idx])
                axis_j = _normalize(axes[other])
                axis_dot = abs(float(np.dot(axis_i, axis_j)))
                if axis_dot < axis_agree:
                    continue
                vec = points_ras[other] - points_ras[idx]
                dist = float(np.linalg.norm(vec))
                if dist <= 1e-6:
                    continue
                vec_u = vec / dist
                if abs(float(np.dot(axis_i, vec_u))) < vector_agree:
                    continue
                if abs(float(np.dot(axis_j, vec_u))) < vector_agree:
                    continue
                score = (0.5 * (float(anis[idx]) + float(anis[other]))) + axis_dot - 0.1 * dist
                provisional[idx].append((score, int(other), dist))
                provisional[other].append((score, int(idx), dist))
        for idx, edges in provisional.items():
            edges.sort(key=lambda item: float(item[0]), reverse=True)
            for _score, other, dist in edges[:max_degree]:
                adjacency[idx].append((int(other), float(dist)))
        # symmetrize
        out: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for idx, edges in adjacency.items():
            for other, dist in edges:
                out[idx].append((other, dist))
                if not any(n == idx for n, _ in out[other]):
                    out[other].append((idx, dist))
        return dict(out)

    def _components(self, adjacency: dict[int, list[tuple[int, float]]], nodes: set[int]) -> list[set[int]]:
        comps: list[set[int]] = []
        seen: set[int] = set()
        for start in sorted(nodes):
            if start in seen:
                continue
            stack = [start]
            comp: set[int] = set()
            seen.add(start)
            while stack:
                node = stack.pop()
                comp.add(node)
                for nbr, _dist in adjacency.get(node, []):
                    if nbr in nodes and nbr not in seen:
                        seen.add(nbr)
                        stack.append(nbr)
            comps.append(comp)
        return comps

    def _extract_paths(self, adjacency: dict[int, list[tuple[int, float]]], nodes: set[int], points_ras: np.ndarray, support: dict[str, Any], ctx: DetectionContext, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        min_path_nodes = int(cfg.get("graph_min_path_nodes", max(6, int(cfg.get("min_inliers", 6)))))
        remaining = set(nodes)
        while len(remaining) >= min_path_nodes:
            comps = self._components(adjacency, remaining)
            if not comps:
                break
            progress = False
            for comp in comps:
                if len(comp) < min_path_nodes:
                    continue
                start = next(iter(comp))
                dist0, _ = _dijkstra(start, adjacency, comp)
                if not dist0:
                    continue
                u = max(dist0.items(), key=lambda kv: float(kv[1]))[0]
                dist1, parent = _dijkstra(u, adjacency, comp)
                if not dist1:
                    continue
                v, path_len = max(dist1.items(), key=lambda kv: float(kv[1]))
                path = _reconstruct_path(parent, u, v)
                if len(path) < min_path_nodes:
                    continue
                path_points = points_ras[np.asarray(path, dtype=int)]
                center, axis, _ = _weighted_pca_line(path_points, np.ones((len(path),), dtype=float))
                corridor_radius = float(cfg.get("path_expand_radius_mm", 1.5))
                d = _line_distances(points_ras[np.asarray(sorted(comp), dtype=int)], center, axis)
                comp_idx = np.asarray(sorted(comp), dtype=int)
                expanded = comp_idx[d <= corridor_radius]
                if expanded.size < min_path_nodes:
                    expanded = np.asarray(path, dtype=int)
                fit_points = points_ras[expanded]
                center, axis, _ = _weighted_pca_line(fit_points, np.ones((fit_points.shape[0],), dtype=float))
                start_ras, end_ras = _projected_endpoints(fit_points, center, axis, float(cfg.get("endpoint_lo_quantile", 0.05)), float(cfg.get("endpoint_hi_quantile", 0.95)))
                center_ras = np.asarray(ctx.get("center_ras") if ctx.get("center_ras") is not None else [0.0, 0.0, 0.0], dtype=float)
                start_ras, end_ras = _orient_shallow_to_deep(start_ras, end_ras, support.get("depth_map_kji"), ctx.get("ras_to_ijk_fn"), center_ras)
                axis = _normalize(end_ras - start_ras)
                rms = float(np.sqrt(np.mean(_line_distances(fit_points, center, axis) ** 2))) if fit_points.shape[0] else 0.0
                inside_fraction = _segment_inside_fraction(start_ras, end_ras, support.get("gating_mask_kji"), ctx.get("ras_to_ijk_fn"), step_mm=1.0)
                entry_depth = _depth_at_ras_mm(start_ras, support.get("depth_map_kji"), ctx.get("ras_to_ijk_fn"))
                target_depth = _depth_at_ras_mm(end_ras, support.get("depth_map_kji"), ctx.get("ras_to_ijk_fn"))
                depth_span = 0.0 if entry_depth is None or target_depth is None else float(target_depth - entry_depth)
                length_mm = float(np.linalg.norm(end_ras - start_ras))
                if length_mm < float(cfg.get("min_length_mm", 20.0)) or depth_span < float(cfg.get("graph_min_depth_span_mm", 10.0)):
                    remaining.difference_update(set(int(v) for v in expanded.tolist()))
                    progress = True
                    continue
                score = (0.1 * fit_points.shape[0]) + (0.03 * length_mm) + (0.03 * depth_span) + (2.0 * inside_fraction) - rms
                results.append({
                    "name": f"P{len(results)+1:03d}",
                    "start_ras": [float(v) for v in start_ras],
                    "end_ras": [float(v) for v in end_ras],
                    "length_mm": length_mm,
                    "support_weight": float(fit_points.shape[0]),
                    "inlier_count": int(fit_points.shape[0]),
                    "inside_fraction": float(inside_fraction),
                    "depth_span_mm": float(depth_span),
                    "entry_depth_mm": None if entry_depth is None else float(entry_depth),
                    "target_depth_mm": None if target_depth is None else float(target_depth),
                    "rms_mm": float(rms),
                    "selection_score": float(score),
                    "assigned_global_indices": [int(v) for v in expanded.tolist()],
                    "graph_path_length_mm": float(path_len),
                })
                remaining.difference_update(set(int(v) for v in expanded.tolist()))
                progress = True
            if not progress:
                break
        return results

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
                result["warnings"].append("shank_graph_v1 missing volume context; returning empty result")
                diagnostics.note("shank_graph_v1 requires arr_kji and ijk_kji_to_ras_fn for detection")
                return self.finalize(result, diagnostics, t_start)

            support = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="support_build", fn=lambda: self._build_support(ctx, cfg))
            points_ras = np.asarray(support.get("points_ras") if support.get("points_ras") is not None else [], dtype=float).reshape(-1, 3)
            ijk_kji = np.asarray(support.get("ijk_kji") if support.get("ijk_kji") is not None else [], dtype=float).reshape(-1, 3)
            if points_ras.shape[0] == 0:
                result["trajectories"] = []
                result["contacts"] = []
                return self.finalize(result, diagnostics, t_start)

            axes, anis, neighborhoods = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="local_orientation",
                fn=lambda: self._local_orientation(ijk_kji, points_ras, support["spacing_xyz"], cfg),
            )
            adjacency = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="graph_build",
                fn=lambda: self._build_graph(ijk_kji, points_ras, axes, anis, support["spacing_xyz"], cfg),
            )
            nodes = set(int(i) for i in range(points_ras.shape[0]) if int(len(adjacency.get(int(i), []))) > 0)
            proposals = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="path_extraction",
                fn=lambda: self._extract_paths(adjacency, nodes, points_ras, support, ctx, cfg),
            )
            selected = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="selection", fn=lambda: self._select_final(proposals, cfg))

            result["trajectories"] = [
                {
                    "name": str(line.get("name") or f"P{idx:02d}"),
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
                        "graph_path_length_mm": float(line.get("graph_path_length_mm", 0.0)),
                    },
                }
                for idx, line in enumerate(selected, start=1)
            ]
            result["contacts"] = []
            result["warnings"].append("contact_detection_not_implemented")

            diagnostics.set_count("candidate_points_total", int(points_ras.shape[0]))
            diagnostics.set_count("graph_node_count", int(len(nodes)))
            diagnostics.set_count("proposal_count", int(len(proposals)))
            diagnostics.set_count("final_lines_kept", int(len(selected)))
            diagnostics.note("shank_graph_v1 extracts long graph paths from orientation-consistent voxel neighborhoods")

            writer = self.get_artifact_writer(ctx, result)
            path_rows = [[int(i), float(anis[i]), int(len(adjacency.get(i, [])))] for i in range(points_ras.shape[0])]
            path_path = writer.write_csv_rows("shank_graph_nodes.csv", ["node_index", "anisotropy", "degree"], path_rows)
            add_artifact(result["artifacts"], kind="graph_csv", path=path_path, description="Graph node summary", stage="graph_build")
            result["artifacts"].extend(
                write_standard_artifacts(
                    writer,
                    result,
                    blobs=[],
                    pipeline_payload={
                        "pipeline_id": self.pipeline_id,
                        "pipeline_version": self.pipeline_version,
                        "counts": {
                            "points": int(points_ras.shape[0]),
                            "nodes": int(len(nodes)),
                            "proposals": int(len(proposals)),
                            "selected": int(len(selected)),
                        },
                    },
                )
            )
        except Exception as exc:
            self.fail(ctx=ctx, result=result, diagnostics=diagnostics, stage=str(getattr(exc, "stage", "pipeline")), exc=exc)

        return self.finalize(result, diagnostics, t_start)
