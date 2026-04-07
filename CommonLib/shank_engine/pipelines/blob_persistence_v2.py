"""Adaptive persistence-aware shank detector.

This pipeline uses multi-threshold blob lineage roles to separate stable shank
cores from weaker late-emergent evidence. Stable core lineages seed the first
pass. If those core-supported lines do not explain enough weighted support, a
second pass adds semi-stable residual lineages as rescue candidates.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from shank_core.masking import build_preview_masks

from ..artifacts import add_artifact, write_standard_artifacts
from ..contracts import DetectionContext, DetectionResult
from ..lineage_tracking import build_lineages, extract_threshold_levels, score_lineage_roles, summarize_lineages
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


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    vals = np.asarray(values, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if vals.size == 0:
        return 0.0
    if vals.size == 1:
        return float(vals[0])
    order = np.argsort(vals)
    vals = vals[order]
    w = np.maximum(w[order], 1e-9)
    cdf = np.cumsum(w)
    cdf /= float(cdf[-1])
    return float(np.interp(float(q), cdf, vals))


def _dense_terminal_endpoints(
    points: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    weights: np.ndarray,
    *,
    window_mm: float,
    search_fraction: float,
    fallback_lo_q: float,
    fallback_hi_q: float,
) -> tuple[np.ndarray, np.ndarray]:
    proj = np.asarray((points - center.reshape(1, 3)) @ axis.reshape(3), dtype=float).reshape(-1)
    w = np.maximum(np.asarray(weights, dtype=float).reshape(-1), 1e-9)
    if proj.size < 3:
        return _projected_endpoints(points, center, axis, fallback_lo_q, fallback_hi_q)

    lo_all = float(np.min(proj))
    hi_all = float(np.max(proj))
    span = float(hi_all - lo_all)
    if span <= 1e-6:
        return _projected_endpoints(points, center, axis, fallback_lo_q, fallback_hi_q)

    search_span = max(float(window_mm), float(span * search_fraction))
    shallow_mask = proj <= (lo_all + search_span)
    deep_mask = proj >= (hi_all - search_span)
    if int(np.count_nonzero(shallow_mask)) < 2 or int(np.count_nonzero(deep_mask)) < 2:
        return _projected_endpoints(points, center, axis, fallback_lo_q, fallback_hi_q)

    def _best_center(mask: np.ndarray) -> float:
        idx = np.where(mask)[0]
        best_t = float(proj[idx[0]])
        best_mass = -1.0
        half = 0.5 * float(window_mm)
        for i in idx.tolist():
            t0 = float(proj[i])
            local = np.abs(proj - t0) <= half
            mass = float(np.sum(w[local]))
            if mass > best_mass:
                best_mass = mass
                best_t = t0
        return best_t

    half = 0.5 * float(window_mm)
    shallow_center = _best_center(shallow_mask)
    deep_center = _best_center(deep_mask)
    shallow_local = np.abs(proj - shallow_center) <= half
    deep_local = np.abs(proj - deep_center) <= half
    if int(np.count_nonzero(shallow_local)) < 2 or int(np.count_nonzero(deep_local)) < 2:
        return _projected_endpoints(points, center, axis, fallback_lo_q, fallback_hi_q)

    start_t = _weighted_quantile(
        proj[shallow_local],
        w[shallow_local],
        float(0.35),
    )
    end_t = _weighted_quantile(
        proj[deep_local],
        w[deep_local],
        float(0.65),
    )
    if end_t <= start_t:
        return _projected_endpoints(points, center, axis, fallback_lo_q, fallback_hi_q)
    start = center + axis * float(start_t)
    end = center + axis * float(end_t)
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


def _blob_voxel_points_ras(
    *,
    levels: list[dict[str, Any]],
    node: dict[str, Any],
    ijk_kji_to_ras_fn,
    cache: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    level_index = int(node.get("level_index", 0))
    blob = dict(node.get("blob") or {})
    blob_id = int(blob.get("blob_id", 0))
    key = (level_index, blob_id)
    if key in cache:
        return cache[key]
    labels = np.asarray(levels[level_index]["labels_kji"], dtype=np.int32)
    pts_kji = np.argwhere(labels == blob_id)
    if pts_kji.size == 0:
        pts_ras = np.asarray(blob.get("centroid_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(1, 3)
    else:
        if pts_kji.shape[0] > 1200:
            stride = int(max(1, math.ceil(float(pts_kji.shape[0]) / 1200.0)))
            pts_kji = pts_kji[::stride]
        pts_ras = np.asarray(ijk_kji_to_ras_fn(pts_kji), dtype=float).reshape(-1, 3)
    cache[key] = pts_ras
    return pts_ras


def _axial_support_points(
    *,
    points_ras: np.ndarray,
    centroid_ras: np.ndarray,
    axis_ras: np.ndarray,
    blob_length_mm: float,
    spacing_mm: float,
    window_mm: float,
    min_length_mm: float,
    max_samples: int,
) -> list[np.ndarray]:
    pts = np.asarray(points_ras, dtype=float).reshape(-1, 3)
    if pts.shape[0] == 0:
        return [np.asarray(centroid_ras, dtype=float).reshape(3)]
    axis = _normalize(axis_ras)
    center = np.asarray(centroid_ras, dtype=float).reshape(3)
    length_mm = float(blob_length_mm)
    if pts.shape[0] < 4 or length_mm < float(min_length_mm):
        return [center]

    proj = np.asarray((pts - center.reshape(1, 3)) @ axis.reshape(3), dtype=float).reshape(-1)
    lo = float(np.min(proj))
    hi = float(np.max(proj))
    span = float(hi - lo)
    if span < float(min_length_mm):
        return [center]

    sample_count = int(max(2, min(int(math.ceil(span / max(spacing_mm, 1e-3))) + 1, int(max_samples))))
    centers_t = np.linspace(lo, hi, sample_count)
    half = 0.5 * float(window_mm)
    out: list[np.ndarray] = []
    for t0 in centers_t.tolist():
        local = np.abs(proj - float(t0)) <= half
        if int(np.count_nonzero(local)) <= 0:
            continue
        local_center = np.mean(pts[local], axis=0)
        if out and float(np.linalg.norm(local_center - out[-1])) < 0.75:
            continue
        out.append(np.asarray(local_center, dtype=float).reshape(3))
    return out or [center]




def _serialize_seed(seed: dict[str, Any]) -> dict[str, Any]:
    return {
        "point_ras": [float(v) for v in np.asarray(seed.get("point", [0.0, 0.0, 0.0]), dtype=float).reshape(3)],
        "axis_ras": [float(v) for v in _normalize(np.asarray(seed.get("axis", [0.0, 0.0, 1.0]), dtype=float).reshape(3))],
        "seed_score": float(seed.get("seed_score", 0.0)),
        "seed_role": str(seed.get("seed_role") or "unknown"),
        "lineage_id": int(seed.get("lineage_id", -1)),
    }


def _serialize_support(support: dict[str, Any]) -> dict[str, Any]:
    return {
        "point_ras": [float(v) for v in np.asarray(support.get("point_ras", [0.0, 0.0, 0.0]), dtype=float).reshape(3)],
        "axis_ras": [float(v) for v in _normalize(np.asarray(support.get("axis_ras", [0.0, 0.0, 1.0]), dtype=float).reshape(3))],
        "support_weight": float(support.get("support_weight", 0.0)),
        "lineage_id": int(support.get("lineage_id", -1)),
        "seed_role": str(support.get("seed_role") or support.get("role") or "unknown"),
        "p_core": float(support.get("p_core", 0.0)),
        "p_contact": float(support.get("p_contact", 0.0)),
        "p_junction": float(support.get("p_junction", 0.0)),
        "threshold_hu": float(support.get("threshold_hu", 0.0)),
        "length_mm": float(support.get("length_mm", 0.0)),
        "diameter_mm": float(support.get("diameter_mm", 0.0)),
    }





def _projection_gap_split(
    points: np.ndarray,
    weights: np.ndarray,
    *,
    center: np.ndarray,
    axis: np.ndarray,
    max_gap_mm: float,
    min_cluster_points: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if pts.shape[0] < max(2 * int(min_cluster_points), 6):
        return None
    proj = np.asarray((pts - center.reshape(1, 3)) @ axis.reshape(3), dtype=float).reshape(-1)
    order = np.argsort(proj)
    proj_ord = proj[order]
    gaps = np.diff(proj_ord)
    if gaps.size == 0:
        return None
    split_idx = int(np.argmax(gaps))
    max_gap = float(gaps[split_idx])
    if max_gap < float(max_gap_mm):
        return None
    left = order[: split_idx + 1]
    right = order[split_idx + 1 :]
    if left.size < int(min_cluster_points) or right.size < int(min_cluster_points):
        return None
    left_mass = float(np.sum(w[left]))
    right_mass = float(np.sum(w[right]))
    keep = left if left_mass >= right_mass else right
    return pts[keep], w[keep]


class BlobPersistenceV2Pipeline(BaseDetectionPipeline):
    pipeline_id = "blob_persistence_v2"
    display_name = "Blob Persistence v2"
    pipeline_version = "2.0.0"

    def _build_gating(self, ctx: DetectionContext, cfg: dict[str, Any]) -> dict[str, Any]:
        arr_kji = np.asarray(ctx["arr_kji"])
        spacing_xyz = tuple(ctx.get("spacing_xyz") or (1.0, 1.0, 1.0))
        thresholds = [float(v) for v in list(cfg.get("threshold_schedule_hu", [2600.0, 2200.0, 1800.0, 1500.0, 1200.0]))]
        preview = build_preview_masks(
            arr_kji=arr_kji,
            spacing_xyz=spacing_xyz,
            threshold=float(thresholds[min(len(thresholds) // 2, len(thresholds) - 1)]),
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
        return {
            "gating_mask_kji": np.asarray(preview.get("gating_mask_kji"), dtype=bool),
            "depth_map_kji": np.asarray(preview.get("head_distance_map_kji"), dtype=np.float32) if preview.get("head_distance_map_kji") is not None else None,
        }

    def _extract_levels(self, ctx: DetectionContext, gating: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
        return extract_threshold_levels(
            arr_kji=np.asarray(ctx["arr_kji"]),
            gating_mask_kji=np.asarray(gating["gating_mask_kji"], dtype=bool),
            depth_map_kji=gating.get("depth_map_kji"),
            thresholds_hu=[float(v) for v in list(cfg.get("threshold_schedule_hu", [2600.0, 2200.0, 1800.0, 1500.0, 1200.0]))],
            ijk_kji_to_ras_fn=ctx["ijk_kji_to_ras_fn"],
        )

    def _node_observations(self, lineage: dict[str, Any]) -> list[dict[str, Any]]:
        # Keep the full descendant chain. The earlier summary-node reduction was
        # too lossy: it removed span information that is needed for both first
        # pass fitting and endpoint estimation.
        return list(lineage.get("nodes") or [])

    def _build_supports(self, lineages: list[dict[str, Any]], levels: list[dict[str, Any]], ctx: DetectionContext, cfg: dict[str, Any]) -> dict[str, Any]:
        summaries = summarize_lineages(lineages, total_levels=max(1, len({int(node['level_index']) for lineage in lineages for node in lineage.get('nodes') or []})))
        by_id = {int(row["lineage_id"]): row for row in summaries}
        core_supports: list[dict[str, Any]] = []
        rescue_supports: list[dict[str, Any]] = []
        core_seeds: list[dict[str, Any]] = []
        rescue_seeds: list[dict[str, Any]] = []
        lineage_rows: list[dict[str, Any]] = []
        voxel_cache: dict[tuple[int, int], np.ndarray] = {}

        min_core_prob = float(cfg.get("core_min_probability", 0.24))
        max_core_junction = float(cfg.get("core_max_junction_probability", 0.58))
        rescue_min_prob = float(cfg.get("rescue_min_core_probability", 0.10))
        rescue_max_junction = float(cfg.get("rescue_max_junction_probability", 0.70))
        min_support_length = float(cfg.get("min_support_length_mm", 3.0))
        axial_sample_min_length = float(cfg.get("axial_sample_min_length_mm", 8.0))
        axial_sample_spacing = float(cfg.get("axial_sample_spacing_mm", 4.0))
        axial_sample_window = float(cfg.get("axial_sample_window_mm", 4.5))
        axial_sample_max = int(cfg.get("axial_sample_max_per_blob", 8))

        for lineage in lineages:
            lineage_id = int(lineage["lineage_id"])
            summary = by_id.get(lineage_id, {})
            p_core = float(summary.get("p_core", 0.0))
            p_contact = float(summary.get("p_contact", 0.0))
            p_junction = float(summary.get("p_junction", 1.0))
            top_length = float(summary.get("top_length_mm", 0.0))
            last_length = float(summary.get("last_length_mm", 0.0))
            support_priority = float(summary.get("support_priority", 0.0))
            max_length = max(top_length, last_length)
            elongated = max_length >= min_support_length
            if not elongated:
                continue

            is_core = p_core >= min_core_prob and p_junction <= max_core_junction
            is_rescue = (not is_core) and p_core >= rescue_min_prob and p_junction <= rescue_max_junction and p_contact < 0.70
            if not is_core and not is_rescue:
                continue

            obs_nodes = self._node_observations(lineage)
            if not obs_nodes:
                continue
            base_weight = max(0.25, support_priority) * max(0.15, 1.0 - 0.65 * p_junction)
            node_bias = []
            for idx, node in enumerate(obs_nodes):
                blob = dict(node.get("blob") or {})
                node_bias.append(max(0.25, float(blob.get("length_mm") or 0.0)))
            node_bias_arr = np.asarray(node_bias, dtype=float)
            node_bias_arr = node_bias_arr / float(np.sum(node_bias_arr))

            target_supports = core_supports if is_core else rescue_supports
            target_seeds = core_seeds if is_core else rescue_seeds
            role_name = "core" if is_core else "rescue"
            rep_axis = _normalize(np.asarray(summary.get("top_axis_ras") or [0.0, 0.0, 1.0], dtype=float))
            rep_point = np.asarray(summary.get("top_centroid_ras") or [0.0, 0.0, 0.0], dtype=float)
            seed_score = base_weight * (1.25 if is_core else 0.8) * max(1.0, top_length)
            target_seeds.append(
                {
                    "point": rep_point,
                    "axis": rep_axis,
                    "score": float(seed_score),
                    "lineage_id": lineage_id,
                    "seed_role": role_name,
                }
            )
            last_blob = dict(lineage.get("last_blob") or {})
            last_point = np.asarray(last_blob.get("centroid_ras") or rep_point, dtype=float)
            last_axis = _normalize(np.asarray(last_blob.get("pca_axis_ras") or rep_axis, dtype=float))
            if float(np.linalg.norm(last_point - rep_point)) >= 4.0:
                target_seeds.append(
                    {
                        "point": last_point,
                        "axis": last_axis,
                        "score": float(seed_score * 0.9),
                        "lineage_id": lineage_id,
                        "seed_role": role_name,
                    }
                )

            for obs_idx, node in enumerate(obs_nodes):
                blob = dict(node.get("blob") or {})
                node_weight = float(base_weight * node_bias_arr[obs_idx])
                axis_ras = _normalize(np.asarray(blob.get("pca_axis_ras") or rep_axis, dtype=float))
                centroid_ras = np.asarray(blob.get("centroid_ras") or [0.0, 0.0, 0.0], dtype=float)
                pts_ras = _blob_voxel_points_ras(
                    levels=levels,
                    node=node,
                    ijk_kji_to_ras_fn=ctx["ijk_kji_to_ras_fn"],
                    cache=voxel_cache,
                )
                sample_points = _axial_support_points(
                    points_ras=pts_ras,
                    centroid_ras=centroid_ras,
                    axis_ras=axis_ras,
                    blob_length_mm=float(blob.get("length_mm") or 0.0),
                    spacing_mm=axial_sample_spacing,
                    window_mm=axial_sample_window,
                    min_length_mm=axial_sample_min_length,
                    max_samples=axial_sample_max,
                )
                sample_weight = float(node_weight) / float(max(1, len(sample_points)))
                for sample_idx, sample_point in enumerate(sample_points):
                    target_supports.append(
                        {
                            "point_ras": np.asarray(sample_point, dtype=float).reshape(3),
                            "axis_ras": axis_ras,
                            "support_weight": float(sample_weight),
                            "lineage_id": lineage_id,
                            "role": role_name,
                            "p_core": float(p_core),
                            "p_contact": float(p_contact),
                            "p_junction": float(p_junction),
                            "threshold_hu": float(node.get("threshold_hu") or 0.0),
                            "length_mm": float(blob.get("length_mm") or 0.0),
                            "diameter_mm": float(blob.get("diameter_mm") or 0.0),
                            "voxel_count": int(blob.get("voxel_count") or 0),
                            "support_kind": "axial_sample" if len(sample_points) > 1 else "blob_centroid",
                            "sample_index": int(sample_idx),
                            "sample_count": int(len(sample_points)),
                        }
                    )
            lineage_rows.append(
                {
                    "lineage_id": lineage_id,
                    "role": role_name,
                    "p_core": p_core,
                    "p_contact": p_contact,
                    "p_junction": p_junction,
                    "support_priority": support_priority,
                    "observation_count": int(len(obs_nodes)),
                    "top_length_mm": top_length,
                    "last_length_mm": last_length,
                    "merge_count": int(summary.get("merge_count", 0)),
                    "split_count": int(summary.get("split_count", 0)),
                    "support_count": int(sum(1 for s in target_supports if int(s["lineage_id"]) == lineage_id)),
                }
            )
        return {
            "summaries": summaries,
            "core_supports": core_supports,
            "rescue_supports": rescue_supports,
            "core_seeds": core_seeds,
            "rescue_seeds": rescue_seeds,
            "lineage_rows": lineage_rows,
        }

    def _nms_seeds(self, seeds: list[dict[str, Any]], cfg: dict[str, Any], limit_key: str) -> list[dict[str, Any]]:
        ordered = sorted(seeds, key=lambda s: float(s.get("score", 0.0)), reverse=True)
        kept: list[dict[str, Any]] = []
        angle_thr = float(cfg.get("seed_nms_angle_deg", 8.0))
        dist_thr = float(cfg.get("seed_nms_line_distance_mm", 2.5))
        limit = int(cfg.get(limit_key, int(cfg.get("max_lines", 30)) * 4))
        for seed in ordered:
            dup = False
            for prev in kept:
                angle = float(np.degrees(np.arccos(np.clip(abs(float(np.dot(_normalize(seed["axis"]), _normalize(prev["axis"])))), 0.0, 1.0))))
                line_dist = _line_distance(np.asarray(seed["point"], dtype=float), np.asarray(seed["axis"], dtype=float), np.asarray(prev["point"], dtype=float), np.asarray(prev["axis"], dtype=float))
                if angle <= angle_thr and line_dist <= dist_thr:
                    dup = True
                    break
            if not dup:
                kept.append(seed)
            if len(kept) >= limit:
                break
        return kept

    def _fit_lines(self, seeds: list[dict[str, Any]], supports: list[dict[str, Any]], gating: dict[str, Any], ctx: DetectionContext, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        if not seeds or not supports:
            return []
        points = np.asarray([s["point_ras"] for s in supports], dtype=float).reshape(-1, 3)
        axes = np.asarray([s["axis_ras"] for s in supports], dtype=float).reshape(-1, 3)
        weights = np.asarray([float(s["support_weight"]) for s in supports], dtype=float).reshape(-1)
        junction_penalty = np.asarray([float(s.get("p_junction", 0.0)) for s in supports], dtype=float).reshape(-1)
        lines: list[dict[str, Any]] = []
        max_lines = int(cfg.get("max_lines", 30))
        radius = float(cfg.get("support_fit_radius_mm", 2.6))
        min_axis_agree = float(cfg.get("support_min_axis_agree", 0.55))
        min_support_weight = float(cfg.get("min_support_mass", 2.0))
        min_line_supports = int(cfg.get("min_inliers", 6))
        proposal_limit = int(cfg.get("proposal_seed_limit", max_lines * 6))
        for seed_idx, seed in enumerate(seeds[:proposal_limit]):
            pts = points
            pts_axes = axes
            pts_w = weights
            pts_j = junction_penalty
            active_idx = np.arange(points.shape[0], dtype=int)
            p0 = np.asarray(seed["point"], dtype=float)
            axis = _normalize(np.asarray(seed["axis"], dtype=float))
            dist, _ = _line_distances(pts, p0, axis)
            seed_lineage = int(seed.get("lineage_id", -1))
            same_lineage = np.asarray(
                [int(s.get("lineage_id", -1)) == seed_lineage for s in supports],
                dtype=bool,
            )
            axis_ok = (np.abs(np.sum(pts_axes * axis.reshape(1, 3), axis=1)) >= min_axis_agree) | same_lineage
            inlier = (dist <= radius) & axis_ok
            if int(np.count_nonzero(inlier)) < min_line_supports:
                continue
            fit_points = pts[inlier]
            fit_w = pts_w[inlier] * np.maximum(0.15, 1.0 - 0.5 * pts_j[inlier])
            same_lineage_inlier = same_lineage[inlier]
            fit_w = fit_w * np.where(same_lineage_inlier, 1.35, 1.0)
            if float(np.sum(fit_w)) < min_support_weight:
                continue
            ctr, axis, _ = _weighted_pca_line(fit_points, fit_w)
            split_keep = _projection_gap_split(
                fit_points,
                fit_w,
                center=ctr,
                axis=axis,
                max_gap_mm=float(cfg.get("fit_max_projection_gap_mm", 10.0)),
                min_cluster_points=int(cfg.get("fit_gap_split_min_cluster_points", max(3, min_line_supports // 2))),
            )
            if split_keep is not None:
                fit_points, fit_w = split_keep
                if fit_points.shape[0] < min_line_supports or float(np.sum(fit_w)) < min_support_weight:
                    continue
                ctr, axis, _ = _weighted_pca_line(fit_points, fit_w)
            start, end = _dense_terminal_endpoints(
                fit_points,
                ctr,
                axis,
                fit_w,
                window_mm=float(cfg.get("endpoint_density_window_mm", 6.0)),
                search_fraction=float(cfg.get("endpoint_search_fraction", 0.30)),
                fallback_lo_q=float(cfg.get("endpoint_lo_quantile", 0.08)),
                fallback_hi_q=float(cfg.get("endpoint_hi_quantile", 0.92)),
            )
            center_ras = np.asarray(ctx.get("center_ras") if ctx.get("center_ras") is not None else [0.0, 0.0, 0.0], dtype=float)
            start, end = _orient_shallow_to_deep(start, end, gating.get("depth_map_kji"), ctx.get("ras_to_ijk_fn"), center_ras)
            axis = _normalize(end - start)
            residuals, proj = _line_distances(fit_points, ctr, axis)
            rms = float(np.sqrt(np.average(residuals ** 2, weights=np.maximum(fit_w, 1e-6))))
            length_mm = float(np.linalg.norm(end - start))
            if length_mm < float(cfg.get("min_length_mm", 20.0)):
                continue
            entry_depth = _depth_at_ras_mm(start, gating.get("depth_map_kji"), ctx.get("ras_to_ijk_fn"))
            target_depth = _depth_at_ras_mm(end, gating.get("depth_map_kji"), ctx.get("ras_to_ijk_fn"))
            depth_span = 0.0 if entry_depth is None or target_depth is None else float(target_depth - entry_depth)
            if depth_span < float(cfg.get("min_depth_span_mm", 10.0)):
                continue
            inside_fraction = _segment_inside_fraction(start, end, gating.get("gating_mask_kji"), ctx.get("ras_to_ijk_fn"), step_mm=1.0)
            span = float(np.max(proj) - np.min(proj)) if proj.size else 0.0
            score = (
                float(np.sum(fit_w))
                + 0.04 * length_mm
                + 0.03 * depth_span
                + 0.80 * inside_fraction
                + 0.02 * span
                - 1.35 * rms
            )
            lines.append(
                {
                    "name": f"P{seed_idx+1:03d}",
                    "start_ras": [float(v) for v in start],
                    "end_ras": [float(v) for v in end],
                    "length_mm": float(length_mm),
                    "support_weight": float(np.sum(fit_w)),
                    "inlier_count": int(np.count_nonzero(inlier)),
                    "inside_fraction": float(inside_fraction),
                    "depth_span_mm": float(depth_span),
                    "entry_depth_mm": None if entry_depth is None else float(entry_depth),
                    "target_depth_mm": None if target_depth is None else float(target_depth),
                    "rms_mm": float(rms),
                    "selection_score": float(score),
                    "assigned_global_indices": active_idx[inlier].tolist(),
                    "seed_role": str(seed.get("seed_role") or "unknown"),
                }
            )
        return lines

    def _coverage_ratio(self, lines: list[dict[str, Any]], supports: list[dict[str, Any]]) -> float:
        if not supports:
            return 1.0
        total = float(sum(float(s.get("support_weight", 0.0)) for s in supports))
        if total <= 1e-9:
            return 1.0
        used = set()
        covered = 0.0
        for line in lines:
            for idx in list(line.get("assigned_global_indices") or []):
                idx = int(idx)
                if idx in used or idx < 0 or idx >= len(supports):
                    continue
                used.add(idx)
                covered += float(supports[idx].get("support_weight", 0.0))
        return float(covered / total)

    def _residual_supports(self, lines: list[dict[str, Any]], supports: list[dict[str, Any]]) -> list[dict[str, Any]]:
        used = set(int(idx) for line in lines for idx in list(line.get("assigned_global_indices") or []))
        return [s for idx, s in enumerate(supports) if idx not in used]

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
            idx0 = set(int(i) for i in list(line.get("assigned_global_indices") or []))
            dup = False
            for prev in kept:
                s1 = np.asarray(prev["start_ras"], dtype=float)
                e1 = np.asarray(prev["end_ras"], dtype=float)
                d1 = _normalize(e1 - s1)
                idx1 = set(int(i) for i in list(prev.get("assigned_global_indices") or []))
                angle = float(np.degrees(np.arccos(np.clip(abs(float(np.dot(d0, d1))), 0.0, 1.0))))
                line_dist = _line_distance(0.5 * (s0 + e0), d0, 0.5 * (s1 + e1), d1)
                overlap = 0.0
                if idx0 and idx1:
                    inter = len(idx0.intersection(idx1))
                    overlap = float(inter) / float(max(1, min(len(idx0), len(idx1))))
                if (
                    (angle <= float(cfg.get("selection_nms_angle_deg", 8.0)) and line_dist <= float(cfg.get("selection_nms_line_distance_mm", 2.5)))
                    or overlap >= float(cfg.get("selection_nms_support_overlap", 0.45))
                ):
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
                result["warnings"].append("blob_persistence_v2 missing volume context; returning empty result")
                diagnostics.note("blob_persistence_v2 requires arr_kji and ijk_kji_to_ras_fn for detection")
                return self.finalize(result, diagnostics, t_start)

            gating = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="gating", fn=lambda: self._build_gating(ctx, cfg))
            levels = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="multi_threshold_blobs", fn=lambda: self._extract_levels(ctx, gating, cfg))
            lineages = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="lineage_tracking", fn=lambda: build_lineages(levels))
            support_state = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="lineage_support",
                fn=lambda: self._build_supports(lineages, levels, ctx, cfg),
            )

            core_seeds = self._nms_seeds(list(support_state["core_seeds"]), cfg, "max_core_seeds")
            rescue_seeds_all = self._nms_seeds(list(support_state["rescue_seeds"]), cfg, "max_rescue_seeds")
            core_supports = list(support_state["core_supports"])
            rescue_supports = list(support_state["rescue_supports"])

            combined_supports = core_supports + rescue_supports
            lines_first = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="core_fit",
                fn=lambda: self._fit_lines(core_seeds, combined_supports, gating, ctx, cfg),
            )
            coverage_ratio = self._coverage_ratio(lines_first, combined_supports)
            rescue_trigger = bool(
                rescue_supports and (
                    coverage_ratio < float(cfg.get("coverage_min_ratio_for_no_rescue", 0.72))
                    or len(lines_first) < int(cfg.get("rescue_min_first_pass_lines", 6))
                )
            )
            diagnostics.set_count("core_fit_lines", int(len(lines_first)))
            diagnostics.set_extra("coverage_ratio", float(coverage_ratio))
            diagnostics.set_count("bead_rescue_triggered", 1 if rescue_trigger else 0)

            lines_second: list[dict[str, Any]] = []
            if rescue_trigger:
                residual_supports = self._residual_supports(lines_first, combined_supports)
                residual_lineage_ids = {int(s.get("lineage_id", -1)) for s in residual_supports}
                rescue_seeds = [seed for seed in rescue_seeds_all if int(seed.get("lineage_id", -1)) in residual_lineage_ids]
                lines_second = self.run_stage(
                    ctx=ctx,
                    result=result,
                    diagnostics=diagnostics,
                    stage_name="rescue_fit",
                    fn=lambda: self._fit_lines(rescue_seeds, residual_supports, gating, ctx, cfg),
                )
            all_lines = lines_first + lines_second
            selected = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="selection", fn=lambda: self._select_final(all_lines, cfg))
            if bool(cfg.get("return_candidate_lines", False)) or bool(cfg.get("return_proposal_debug", False)):
                result["meta"].setdefault("extras", {})
            if bool(cfg.get("return_candidate_lines", False)):
                result["meta"]["extras"]["candidate_lines"] = list(all_lines)
            if bool(cfg.get("return_proposal_debug", False)):
                result["meta"]["extras"]["proposal_debug"] = {
                    "core_seeds": [_serialize_seed(seed) for seed in core_seeds],
                    "rescue_seeds": [_serialize_seed(seed) for seed in rescue_seeds_all],
                    "core_supports": [_serialize_support(support) for support in core_supports],
                    "rescue_supports": [_serialize_support(support) for support in rescue_supports],
                    "lines_first": list(lines_first),
                    "lines_second": list(lines_second),
                    "coverage_ratio": float(coverage_ratio),
                    "rescue_trigger": bool(rescue_trigger),
                }

            result["trajectories"] = [
                {
                    "name": str(line.get("name") or f"P{idx:02d}"),
                    "start_ras": list(line.get("start_ras", [0.0, 0.0, 0.0])),
                    "end_ras": list(line.get("end_ras", [0.0, 0.0, 0.0])),
                    "length_mm": float(line.get("length_mm", 0.0)),
                    "confidence": float(min(1.0, max(0.0, float(line.get("selection_score", 0.0)) / 14.0))),
                    "support_count": int(line.get("inlier_count", 0)),
                    "params": {
                        "rms_mm": float(line.get("rms_mm", 0.0)),
                        "inside_fraction": float(line.get("inside_fraction", 0.0)),
                        "depth_span_mm": float(line.get("depth_span_mm", 0.0)),
                        "entry_depth_mm": line.get("entry_depth_mm"),
                        "target_depth_mm": line.get("target_depth_mm"),
                        "selection_score": float(line.get("selection_score", 0.0)),
                        "seed_role": str(line.get("seed_role") or "unknown"),
                    },
                }
                for idx, line in enumerate(selected, start=1)
            ]
            result["contacts"] = []
            result["warnings"].append("contact_detection_not_implemented")

            diagnostics.set_count("threshold_levels", int(len(levels)))
            diagnostics.set_count("lineage_count_total", int(len(lineages)))
            diagnostics.set_count("lineage_count_core", int(len({int(s['lineage_id']) for s in core_supports})))
            diagnostics.set_count("lineage_count_rescue", int(len({int(s['lineage_id']) for s in rescue_supports})))
            diagnostics.set_count("core_support_count", int(len(core_supports)))
            diagnostics.set_count("rescue_support_count", int(len(rescue_supports)))
            diagnostics.set_count("core_seed_count", int(len(core_seeds)))
            diagnostics.set_count("rescue_seed_count", int(len(rescue_seeds_all)))
            diagnostics.set_count("proposal_count_first", int(len(lines_first)))
            diagnostics.set_count("proposal_count_second", int(len(lines_second)))
            diagnostics.set_count("final_lines_kept", int(len(selected)))
            diagnostics.note("blob_persistence_v2 seeds shafts from stable lineage cores and uses semi-stable residual rescue")

            writer = self.get_artifact_writer(ctx, result)
            lineage_path = writer.write_csv_rows(
                "lineage_supports.csv",
                [
                    "lineage_id",
                    "role",
                    "p_core",
                    "p_contact",
                    "p_junction",
                    "support_priority",
                    "observation_count",
                    "top_length_mm",
                    "last_length_mm",
                    "merge_count",
                    "split_count",
                ],
                [
                    [
                        int(row["lineage_id"]),
                        str(row["role"]),
                        float(row["p_core"]),
                        float(row["p_contact"]),
                        float(row["p_junction"]),
                        float(row["support_priority"]),
                        int(row["observation_count"]),
                        float(row["top_length_mm"]),
                        float(row["last_length_mm"]),
                        int(row["merge_count"]),
                        int(row["split_count"]),
                    ]
                    for row in support_state["lineage_rows"]
                ],
            )
            add_artifact(result["artifacts"], kind="lineage_csv", path=lineage_path, description="Lineage role support rows", stage="lineage_support")
            result["artifacts"].extend(
                write_standard_artifacts(
                    writer,
                    result,
                    blobs=[],
                    pipeline_payload={
                        "pipeline_id": self.pipeline_id,
                        "pipeline_version": self.pipeline_version,
                        "counts": {
                            "levels": int(len(levels)),
                            "lineages": int(len(lineages)),
                            "core_supports": int(len(core_supports)),
                            "rescue_supports": int(len(rescue_supports)),
                            "core_seeds": int(len(core_seeds)),
                            "rescue_seeds": int(len(rescue_seeds_all)),
                            "first_pass": int(len(lines_first)),
                            "second_pass": int(len(lines_second)),
                            "selected": int(len(selected)),
                        },
                    },
                )
            )
        except Exception as exc:
            self.fail(ctx=ctx, result=result, diagnostics=diagnostics, stage=str(getattr(exc, "stage", "pipeline")), exc=exc)
        return self.finalize(result, diagnostics, t_start)
