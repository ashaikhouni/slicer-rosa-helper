"""Blob-consensus shank detector.

This pipeline treats multi-threshold raw connected components as three evidence
types:
- compact bead-like blobs
- single-line elongated segments
- merged/junction-like elongated blobs

Shank hypotheses are generated from compact-blob consensus first, then
reinforced or rescued with segment evidence. Contacts are intentionally left
unimplemented in this pass.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from rosa_core.electrode_models import load_electrode_library
from shank_core.masking import build_preview_masks

from ..artifacts import add_artifact, write_standard_artifacts
from ..contracts import DetectionContext, DetectionResult
from ..lineage_tracking import build_lineages, extract_threshold_levels
from .base import BaseDetectionPipeline


@dataclass(frozen=True)
class _ElectrodePrior:
    model_id: str
    pitch_mm: float
    diameter_mm: float
    total_length_mm: float
    contact_count: int


@dataclass(frozen=True)
class _BlobPrimitive:
    key: str
    lineage_id: int
    point_ras: tuple[float, float, float]
    axis_ras: tuple[float, float, float]
    score: float
    length_mm: float
    diameter_mm: float
    threshold_hu: float
    depth_mm: float
    voxel_count: int
    kind: str
    meta: dict[str, Any]


@dataclass(frozen=True)
class _SegmentPrimitive:
    key: str
    lineage_id: int
    point_ras: tuple[float, float, float]
    axis_ras: tuple[float, float, float]
    score: float
    length_mm: float
    diameter_mm: float
    threshold_hu: float
    depth_mm: float
    source_kind: str
    meta: dict[str, Any]


@dataclass(frozen=True)
class _Candidate:
    candidate_id: str
    source: str
    start_ras: tuple[float, float, float]
    end_ras: tuple[float, float, float]
    direction_ras: tuple[float, float, float]
    score: float
    support_mass: float
    pitch_score: float
    skull_score: float
    span_mm: float
    rms_mm: float
    support_blob_keys: tuple[str, ...]
    seed_keys: tuple[str, ...]
    best_model_id: str | None
    meta: dict[str, Any]


def _normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-9:
        return np.asarray([0.0, 0.0, 1.0], dtype=float)
    return arr / norm


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    aa = _normalize(np.asarray(a, dtype=float))
    bb = _normalize(np.asarray(b, dtype=float))
    dot = float(abs(np.dot(aa, bb)))
    dot = min(1.0, max(-1.0, dot))
    return float(np.degrees(np.arccos(dot)))


def _line_distance(p0: np.ndarray, d0: np.ndarray, p1: np.ndarray, d1: np.ndarray) -> float:
    u = _normalize(d0)
    v = _normalize(d1)
    w0 = p0 - p1
    c = np.cross(u, v)
    cn = float(np.linalg.norm(c))
    if cn <= 1e-6:
        return float(np.linalg.norm(np.cross(w0, u)))
    return float(abs(np.dot(w0, c)) / cn)


def _line_distances(points: np.ndarray, p0: np.ndarray, direction_unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rel = points - p0.reshape(1, 3)
    t = rel @ direction_unit.reshape(3)
    closest = p0.reshape(1, 3) + np.outer(t, direction_unit.reshape(3))
    return np.linalg.norm(points - closest, axis=1), t


def _sample_points(points: np.ndarray, *, max_points: int) -> np.ndarray:
    pts = np.asarray(points)
    if pts.shape[0] <= int(max_points):
        return pts
    idx = np.linspace(0, pts.shape[0] - 1, int(max_points), dtype=int)
    return pts[idx]


def _weighted_pca_line(points: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if pts.shape[0] < 2:
        raise ValueError("need at least 2 points")
    if weights is None:
        wn = np.ones((pts.shape[0],), dtype=float) / float(pts.shape[0])
    else:
        w = np.maximum(np.asarray(weights, dtype=float).reshape(-1), 0.0)
        s = float(np.sum(w))
        if s <= 1e-9:
            raise ValueError("zero weights")
        wn = w / s
    center = np.sum(pts * wn[:, None], axis=0)
    x = pts - center.reshape(1, 3)
    cov = (x.T * wn.reshape(1, -1)) @ x
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)
    evals = np.asarray(evals[order], dtype=float)
    axis = _normalize(evecs[:, order[-1]])
    return center.astype(float), axis.astype(float), evals.astype(float)


def _fit_line_stats(points: np.ndarray) -> dict[str, Any]:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if pts.shape[0] <= 0:
        raise ValueError("need at least 1 point")
    if pts.shape[0] == 1:
        center = np.asarray(pts[0], dtype=float)
        return {
            "center_ras": center.astype(float),
            "axis_ras": np.asarray([0.0, 0.0, 1.0], dtype=float),
            "evals": np.asarray([0.0, 0.0, 0.0], dtype=float),
            "length_mm": 0.0,
            "diameter_mm": 0.0,
            "residual_rms_mm": 0.0,
            "aspect_ratio": 0.0,
        }
    center, axis, evals = _weighted_pca_line(pts)
    dists, proj = _line_distances(pts, center, axis)
    length = float(np.max(proj) - np.min(proj)) if proj.size else 0.0
    residual = float(np.sqrt(np.mean(np.square(dists)))) if dists.size else 0.0
    diameter = float(2.0 * np.sqrt(max(1e-9, float(np.mean(evals[:2])))))
    aspect = float(length / max(diameter, 1e-6))
    return {
        "center_ras": center.astype(float),
        "axis_ras": axis.astype(float),
        "evals": evals.astype(float),
        "length_mm": float(length),
        "diameter_mm": float(diameter),
        "residual_rms_mm": float(residual),
        "aspect_ratio": float(aspect),
    }


def _dilate_one_voxel(mask: np.ndarray) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    out = np.zeros_like(src, dtype=bool)
    if src.size == 0:
        return out
    for dk in (-1, 0, 1):
        ks = slice(max(0, -dk), min(src.shape[0], src.shape[0] - dk))
        kd = slice(max(0, dk), min(src.shape[0], src.shape[0] + dk))
        for dj in (-1, 0, 1):
            js = slice(max(0, -dj), min(src.shape[1], src.shape[1] - dj))
            jd = slice(max(0, dj), min(src.shape[1], src.shape[1] + dj))
            for di in (-1, 0, 1):
                is_ = slice(max(0, -di), min(src.shape[2], src.shape[2] - di))
                id_ = slice(max(0, di), min(src.shape[2], src.shape[2] + di))
                out[kd, jd, id_] |= src[ks, js, is_]
    return out


def _bbox_points_from_mask(mask: np.ndarray, slc: tuple[slice, slice, slice]) -> np.ndarray:
    idx = np.argwhere(np.asarray(mask[slc], dtype=bool))
    if idx.size <= 0:
        return np.zeros((0, 3), dtype=int)
    offs = np.asarray([slc[0].start or 0, slc[1].start or 0, slc[2].start or 0], dtype=int).reshape(1, 3)
    return idx.astype(int) + offs


def _grown_features(raw_points_kji: np.ndarray, raw_axis_ras: np.ndarray, grown_mask: np.ndarray, ijk_kji_to_ras_fn) -> dict[str, Any]:
    pts = np.asarray(raw_points_kji, dtype=int).reshape(-1, 3)
    k0, j0, i0 = np.maximum(np.min(pts, axis=0) - 1, 0)
    k1, j1, i1 = np.min([np.max(pts, axis=0) + 2, np.asarray(grown_mask.shape)], axis=0)
    slc = (slice(int(k0), int(k1)), slice(int(j0), int(j1)), slice(int(i0), int(i1)))
    grown_points_kji = _bbox_points_from_mask(grown_mask, slc)
    if grown_points_kji.size <= 0:
        return {"grown_voxel_count": 0, "grown_support_ratio": 0.0, "grown_axis_stability_deg": 180.0}
    grown_points_ras = np.asarray(ijk_kji_to_ras_fn(grown_points_kji), dtype=float).reshape(-1, 3)
    stats = _fit_line_stats(_sample_points(grown_points_ras, max_points=2500))
    return {
        "grown_voxel_count": int(grown_points_kji.shape[0]),
        "grown_support_ratio": float(grown_points_kji.shape[0]) / float(max(1, pts.shape[0])),
        "grown_axis_stability_deg": float(_angle_deg(np.asarray(raw_axis_ras, dtype=float), np.asarray(stats["axis_ras"], dtype=float))),
    }


def _depth_at_ras_mm(point_ras: np.ndarray, head_depth_kji: np.ndarray | None, ras_to_ijk_fn) -> float | None:
    if head_depth_kji is None or ras_to_ijk_fn is None:
        return None
    ijk = ras_to_ijk_fn(point_ras)
    i = int(round(float(ijk[0])))
    j = int(round(float(ijk[1])))
    k = int(round(float(ijk[2])))
    if k < 0 or j < 0 or i < 0 or k >= head_depth_kji.shape[0] or j >= head_depth_kji.shape[1] or i >= head_depth_kji.shape[2]:
        return None
    val = float(head_depth_kji[k, j, i])
    return val if np.isfinite(val) else None


def _projected_endpoints(points: np.ndarray, center: np.ndarray, axis: np.ndarray, lo_q: float, hi_q: float) -> tuple[np.ndarray, np.ndarray]:
    proj = (points - center.reshape(1, 3)) @ axis.reshape(3)
    lo = float(np.quantile(proj, lo_q))
    hi = float(np.quantile(proj, hi_q))
    return (center + axis * lo).astype(float), (center + axis * hi).astype(float)


def _orient_shallow_to_deep(start: np.ndarray, end: np.ndarray, depth_kji: np.ndarray | None, ras_to_ijk_fn, center_ras: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d0 = _depth_at_ras_mm(start, depth_kji, ras_to_ijk_fn)
    d1 = _depth_at_ras_mm(end, depth_kji, ras_to_ijk_fn)
    if d0 is not None and d1 is not None and abs(float(d0) - float(d1)) > 1e-3:
        return (start, end) if float(d0) <= float(d1) else (end, start)
    c0 = float(np.linalg.norm(start - center_ras))
    c1 = float(np.linalg.norm(end - center_ras))
    return (start, end) if c0 >= c1 else (end, start)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _center_pitch_from_model(model: dict[str, Any]) -> float | None:
    c2c = model.get("center_to_center_separation_mm")
    if c2c is not None:
        val = _safe_float(c2c, 0.0)
        if val > 0.0:
            return val
    offsets = list(model.get("contact_center_offsets_from_tip_mm") or [])
    if len(offsets) >= 2:
        diffs = np.diff(np.asarray(offsets, dtype=float))
        if diffs.size:
            return float(np.median(diffs))
    return None


def _load_family_priors() -> list[_ElectrodePrior]:
    library = load_electrode_library()
    priors: list[_ElectrodePrior] = []
    for model in list(library.get("models") or []):
        pitch = _center_pitch_from_model(model)
        if pitch is None or pitch <= 0.0:
            continue
        priors.append(
            _ElectrodePrior(
                model_id=str(model.get("id") or "unknown"),
                pitch_mm=float(pitch),
                diameter_mm=float(_safe_float(model.get("diameter_mm"), 0.8)),
                total_length_mm=float(_safe_float(model.get("total_exploration_length_mm"), 0.0)),
                contact_count=int(model.get("contact_count") or 0),
            )
        )
    return priors


def _pitch_compatibility(projections_mm: np.ndarray, diameters_mm: np.ndarray, priors: list[_ElectrodePrior]) -> tuple[float, str | None]:
    proj = np.sort(np.asarray(projections_mm, dtype=float).reshape(-1))
    if proj.size < 2 or not priors:
        return 0.0, None
    raw_gaps = np.diff(proj)
    gaps = raw_gaps[np.logical_and(raw_gaps >= 1.0, raw_gaps <= 8.0)]
    if gaps.size <= 0:
        return 0.0, None
    median_gap = float(np.median(gaps))
    span = float(np.max(proj) - np.min(proj))
    obs_count = int(proj.size)
    mean_diam = float(np.mean(np.asarray(diameters_mm, dtype=float).reshape(-1))) if np.asarray(diameters_mm).size else 0.8
    best_score = 0.0
    best_id: str | None = None
    for prior in priors:
        gap_score = 1.0 - _clamp01(abs(median_gap - prior.pitch_mm) / max(1.0, 0.6 * prior.pitch_mm))
        span_score = 1.0 - _clamp01(abs(span - prior.total_length_mm) / max(8.0, 0.8 * max(prior.total_length_mm, 1.0)))
        diam_score = 1.0 - _clamp01(abs(mean_diam - prior.diameter_mm) / 2.5)
        count_score = _clamp01(float(obs_count) / float(max(2, min(prior.contact_count, 6))))
        score = 0.45 * gap_score + 0.25 * span_score + 0.15 * diam_score + 0.15 * count_score
        if score > best_score:
            best_score = float(score)
            best_id = str(prior.model_id)
    return float(best_score), best_id


def _projection_runs(sorted_proj: np.ndarray, *, max_gap_mm: float) -> list[tuple[int, int]]:
    proj = np.asarray(sorted_proj, dtype=float).reshape(-1)
    if proj.size <= 0:
        return []
    if proj.size == 1:
        return [(0, 1)]
    runs: list[tuple[int, int]] = []
    start = 0
    for idx in range(1, proj.size):
        if float(proj[idx] - proj[idx - 1]) > float(max_gap_mm):
            runs.append((start, idx))
            start = idx
    runs.append((start, proj.size))
    return runs


def _gap_regularity_score(projections_mm: np.ndarray) -> float:
    proj = np.sort(np.asarray(projections_mm, dtype=float).reshape(-1))
    if proj.size < 3:
        return 0.5
    gaps = np.diff(proj)
    gaps = gaps[np.logical_and(gaps >= 1.0, gaps <= 8.0)]
    if gaps.size <= 0:
        return 0.0
    median_gap = float(np.median(gaps))
    if median_gap <= 1e-6:
        return 0.0
    mad = float(np.median(np.abs(gaps - median_gap)))
    return 1.0 - _clamp01(mad / max(0.8, 0.45 * median_gap))


def _classify_component(blob: dict[str, Any], fit_stats: dict[str, Any], grown_stats: dict[str, Any]) -> tuple[str, float]:
    vox = int(blob.get("voxel_count") or 0)
    length = float(fit_stats["length_mm"])
    diameter = float(fit_stats["diameter_mm"])
    residual = float(fit_stats["residual_rms_mm"])
    aspect = float(fit_stats["aspect_ratio"])
    depth = float(blob.get("depth_mean") or 0.0)
    grown_ratio = float(grown_stats.get("grown_support_ratio", 0.0))
    grown_axis = float(grown_stats.get("grown_axis_stability_deg", 180.0))

    tiny_blob_like = vox <= 4 and length <= 4.5 and diameter <= 2.5 and residual <= 1.6
    blob_like = (
        vox >= 1
        and length <= 11.0
        and diameter <= 5.5
        and aspect <= 4.5
        and residual <= 2.6
    ) or tiny_blob_like
    line_like = (
        vox >= 10
        and length >= 6.0
        and aspect >= 3.0
        and residual <= max(1.6, 0.70 * max(diameter, 0.8))
        and diameter <= 3.8
        and grown_axis <= 24.0
    )
    merged_like = (
        vox >= 8
        and (
            (length >= 8.0 and not line_like)
            or diameter >= 4.5
            or residual >= max(2.2, 0.95 * max(diameter, 1.0))
            or (grown_ratio >= 5.0 and grown_axis >= 18.0)
        )
    )
    if blob_like:
        score = (
            0.30 * (1.0 - _clamp01(length / 11.0))
            + 0.20 * (1.0 - _clamp01(diameter / 5.5))
            + 0.20 * (1.0 - _clamp01(residual / 2.6))
            + 0.15 * (1.0 - _clamp01(max(0.0, aspect - 1.0) / 3.5))
            + 0.15 * _clamp01(depth / 10.0)
        )
        return "blob", float(score)
    if line_like:
        score = (
            0.35 * _clamp01((aspect - 3.0) / 8.0)
            + 0.25 * _clamp01((length - 6.0) / 18.0)
            + 0.20 * (1.0 - _clamp01(residual / 2.0))
            + 0.10 * (1.0 - _clamp01(abs(diameter - 1.2) / 2.5))
            + 0.10 * (1.0 - _clamp01(grown_axis / 24.0))
        )
        return "segment", float(score)
    if merged_like:
        score = (
            0.30 * _clamp01(length / 25.0)
            + 0.25 * _clamp01(diameter / 6.0)
            + 0.20 * _clamp01(residual / 3.0)
            + 0.15 * _clamp01(grown_ratio / 8.0)
            + 0.10 * _clamp01(grown_axis / 45.0)
        )
        return "merged_segment", float(score)
    return "junk", 0.0


def _decompose_merged_blob(points_ras: np.ndarray, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    pts = np.asarray(points_ras, dtype=float).reshape(-1, 3)
    if pts.shape[0] < int(cfg.get("decomp_min_points", 20)):
        return []
    remaining = pts.copy()
    original_count = int(pts.shape[0])
    out: list[dict[str, Any]] = []
    max_lines = int(cfg.get("decomp_max_lines", 3))
    radius_mm = float(cfg.get("decomp_line_radius_mm", 2.0))
    min_frac = float(cfg.get("decomp_min_fraction_per_line", 0.18))
    min_pts = int(cfg.get("decomp_min_points_per_line", 12))
    for _ in range(max_lines):
        if remaining.shape[0] < min_pts:
            break
        stats0 = _fit_line_stats(_sample_points(remaining, max_points=int(cfg.get("decomp_fit_sample_points", 2500))))
        dists, _ = _line_distances(remaining, np.asarray(stats0["center_ras"], dtype=float), np.asarray(stats0["axis_ras"], dtype=float))
        inlier_mask = dists <= radius_mm
        if int(np.count_nonzero(inlier_mask)) < min_pts:
            break
        inliers = remaining[inlier_mask]
        if float(inliers.shape[0]) / float(max(1, original_count)) < min_frac:
            break
        stats = _fit_line_stats(_sample_points(inliers, max_points=int(cfg.get("decomp_fit_sample_points", 2500))))
        if float(stats["residual_rms_mm"]) > float(cfg.get("decomp_max_residual_mm", 1.8)):
            break
        out.append(
            {
                "center_ras": np.asarray(stats["center_ras"], dtype=float),
                "axis_ras": np.asarray(stats["axis_ras"], dtype=float),
                "length_mm": float(stats["length_mm"]),
                "diameter_mm": float(stats["diameter_mm"]),
                "residual_rms_mm": float(stats["residual_rms_mm"]),
                "explained_fraction": float(inliers.shape[0]) / float(max(1, original_count)),
            }
        )
        remaining = remaining[np.logical_not(inlier_mask)]
    return out


class BlobConsensusV1Pipeline(BaseDetectionPipeline):
    pipeline_id = "blob_consensus_v1"
    display_name = "Blob Consensus v1"
    pipeline_version = "1.0.0"

    def _build_gating(self, ctx: DetectionContext, cfg: dict[str, Any]) -> dict[str, Any]:
        arr_kji = np.asarray(ctx["arr_kji"])
        spacing_xyz = tuple(ctx.get("spacing_xyz") or (1.0, 1.0, 1.0))
        threshold = float(max(list(cfg.get("threshold_schedule_hu", [2600.0, 2200.0, 1800.0, 1500.0, 1200.0]))))
        preview = build_preview_masks(
            arr_kji=arr_kji,
            spacing_xyz=spacing_xyz,
            threshold=threshold,
            use_head_mask=bool(cfg.get("use_head_mask", True)),
            build_head_mask=bool(cfg.get("build_head_mask", True)),
            head_mask_threshold_hu=float(cfg.get("head_mask_threshold_hu", -500.0)),
            head_mask_method=str(cfg.get("head_mask_method", "outside_air")),
            head_gate_erode_vox=int(cfg.get("head_gate_erode_vox", 1)),
            head_gate_dilate_vox=int(cfg.get("head_gate_dilate_vox", 1)),
            head_gate_margin_mm=float(cfg.get("head_gate_margin_mm", 0.0)),
            min_metal_depth_mm=float(cfg.get("min_metal_depth_mm", 5.0)),
            max_metal_depth_mm=float(cfg.get("max_metal_depth_mm", 220.0)),
            include_debug_masks=False,
        )
        return {
            "gating_mask_kji": np.asarray(preview.get("gating_mask_kji"), dtype=bool),
            "head_distance_map_kji": np.asarray(preview.get("head_distance_map_kji"), dtype=np.float32) if preview.get("head_distance_map_kji") is not None else None,
            "metal_in_head_mask_kji": np.asarray(preview.get("metal_in_head_mask_kji"), dtype=bool) if preview.get("metal_in_head_mask_kji") is not None else None,
        }

    def _node_lineage_map(self, lineages: list[dict[str, Any]]) -> dict[tuple[int, int], int]:
        out: dict[tuple[int, int], int] = {}
        for lineage in lineages:
            lineage_id = int(lineage.get("lineage_id") or 0)
            for node in list(lineage.get("nodes") or []):
                level_index = int(node.get("level_index") or 0)
                blob_id = int((node.get("blob") or {}).get("blob_id") or 0)
                out[(level_index, blob_id)] = lineage_id
        return out

    def _extract_primitives(self, ctx: DetectionContext, gating: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
        thresholds = [float(v) for v in list(cfg.get("threshold_schedule_hu", [2600.0, 2200.0, 1800.0, 1500.0, 1200.0]))]
        levels = extract_threshold_levels(
            arr_kji=np.asarray(ctx["arr_kji"], dtype=float),
            gating_mask_kji=np.asarray(gating.get("gating_mask_kji"), dtype=bool),
            depth_map_kji=gating.get("head_distance_map_kji"),
            thresholds_hu=thresholds,
            ijk_kji_to_ras_fn=ctx["ijk_kji_to_ras_fn"],
        )
        lineages = build_lineages(levels)
        lineage_map = self._node_lineage_map(lineages)
        blobs: list[_BlobPrimitive] = []
        segments: list[_SegmentPrimitive] = []
        merged_rows: list[dict[str, Any]] = []
        for level_index, level in enumerate(levels):
            threshold_hu = float(level["threshold_hu"])
            labels = np.asarray(level["labels_kji"], dtype=np.int32)
            raw_mask = labels > 0
            grown_mask = _dilate_one_voxel(raw_mask)
            for blob in list(level.get("blobs") or []):
                blob_id = int(blob.get("blob_id") or 0)
                lineage_id = int(lineage_map.get((int(level_index), blob_id), 0))
                pts_kji = np.argwhere(labels == blob_id)
                if pts_kji.size <= 0:
                    continue
                pts_ras = np.asarray(ctx["ijk_kji_to_ras_fn"](_sample_points(pts_kji, max_points=int(cfg.get("component_fit_sample_points", 3000)))), dtype=float).reshape(-1, 3)
                fit_stats = _fit_line_stats(pts_ras)
                grown_stats = _grown_features(pts_kji, np.asarray(fit_stats["axis_ras"], dtype=float), grown_mask, ctx["ijk_kji_to_ras_fn"])
                kind, score = _classify_component(blob, fit_stats, grown_stats)
                key = f"L{level_index}_B{blob_id}"
                if kind == "blob":
                    blobs.append(
                        _BlobPrimitive(
                            key=key,
                            lineage_id=lineage_id,
                            point_ras=tuple(float(v) for v in np.asarray(fit_stats["center_ras"], dtype=float)),
                            axis_ras=tuple(float(v) for v in np.asarray(fit_stats["axis_ras"], dtype=float)),
                            score=float(score),
                            length_mm=float(fit_stats["length_mm"]),
                            diameter_mm=float(fit_stats["diameter_mm"]),
                            threshold_hu=float(threshold_hu),
                            depth_mm=float(blob.get("depth_mean") or 0.0),
                            voxel_count=int(blob.get("voxel_count") or 0),
                            kind=kind,
                            meta={"aspect_ratio": float(fit_stats["aspect_ratio"]), "residual_rms_mm": float(fit_stats["residual_rms_mm"])},
                        )
                    )
                elif kind == "segment":
                    segments.append(
                        _SegmentPrimitive(
                            key=key,
                            lineage_id=lineage_id,
                            point_ras=tuple(float(v) for v in np.asarray(fit_stats["center_ras"], dtype=float)),
                            axis_ras=tuple(float(v) for v in np.asarray(fit_stats["axis_ras"], dtype=float)),
                            score=float(score),
                            length_mm=float(fit_stats["length_mm"]),
                            diameter_mm=float(fit_stats["diameter_mm"]),
                            threshold_hu=float(threshold_hu),
                            depth_mm=float(blob.get("depth_mean") or 0.0),
                            source_kind="segment",
                            meta={"aspect_ratio": float(fit_stats["aspect_ratio"]), "residual_rms_mm": float(fit_stats["residual_rms_mm"])},
                        )
                    )
                elif kind == "merged_segment":
                    full_pts_ras = np.asarray(ctx["ijk_kji_to_ras_fn"](_sample_points(pts_kji, max_points=int(cfg.get("decomp_max_points", 5000)))), dtype=float).reshape(-1, 3)
                    sub_lines = _decompose_merged_blob(full_pts_ras, cfg)
                    merged_rows.append({"key": key, "sub_line_count": len(sub_lines), "score": float(score)})
                    for sub_idx, sub in enumerate(sub_lines, start=1):
                        segments.append(
                            _SegmentPrimitive(
                                key=f"{key}_S{sub_idx}",
                                lineage_id=lineage_id,
                                point_ras=tuple(float(v) for v in np.asarray(sub["center_ras"], dtype=float)),
                                axis_ras=tuple(float(v) for v in np.asarray(sub["axis_ras"], dtype=float)),
                                score=float(min(1.0, 0.6 * score + 0.4 * float(sub["explained_fraction"]))),
                                length_mm=float(sub["length_mm"]),
                                diameter_mm=float(sub["diameter_mm"]),
                                threshold_hu=float(threshold_hu),
                                depth_mm=float(blob.get("depth_mean") or 0.0),
                                source_kind="merged_subline",
                                meta={"residual_rms_mm": float(sub["residual_rms_mm"]), "explained_fraction": float(sub["explained_fraction"])},
                            )
                        )
        dedup_blobs: list[_BlobPrimitive] = []
        for rec in sorted(blobs, key=lambda x: (-float(x.score), float(x.threshold_hu), x.lineage_id, x.key)):
            pt = np.asarray(rec.point_ras, dtype=float)
            if any(float(np.linalg.norm(pt - np.asarray(k.point_ras, dtype=float))) <= float(cfg.get("blob_dedup_radius_mm", 2.5)) for k in dedup_blobs):
                continue
            dedup_blobs.append(rec)
        return {"levels": levels, "lineages": lineages, "blobs": dedup_blobs, "segments": segments, "merged_rows": merged_rows}

    def _build_blob_candidates(self, blobs: list[_BlobPrimitive], priors: list[_ElectrodePrior], gating: dict[str, Any], ctx: DetectionContext, cfg: dict[str, Any]) -> tuple[list[_Candidate], list[dict[str, Any]]]:
        if len(blobs) < 2:
            return [], []
        points = np.asarray([b.point_ras for b in blobs], dtype=float).reshape(-1, 3)
        weights = np.asarray([float(b.score) for b in blobs], dtype=float)
        diameters = np.asarray([float(b.diameter_mm) for b in blobs], dtype=float)
        pair_min = float(cfg.get("pair_min_distance_mm", 3.0))
        pair_max = float(cfg.get("pair_max_distance_mm", 90.0))
        support_radius = float(cfg.get("support_radius_mm", 2.8))
        support_span_pad = float(cfg.get("support_span_pad_mm", 5.0))
        min_support_blobs = int(cfg.get("min_support_blobs", 4))
        max_neighbors = int(cfg.get("max_blob_neighbors", 10))
        max_gap_mm = float(cfg.get("max_chain_gap_mm", 10.0))
        min_chain_span_mm = float(cfg.get("min_chain_span_mm", 10.0))
        min_chain_regularity = float(cfg.get("min_chain_regularity", 0.20))
        candidates: list[_Candidate] = []
        support_rows: list[dict[str, Any]] = []
        seen_support_sets: set[tuple[str, ...]] = set()
        for i, rec_i in enumerate(blobs):
            pi = points[i]
            dists = np.linalg.norm(points - pi.reshape(1, 3), axis=1)
            neighbor_idx = np.argsort(dists)
            used = 0
            for j in neighbor_idx.tolist():
                if j <= i:
                    continue
                if used >= max_neighbors:
                    break
                dist = float(dists[j])
                if dist < pair_min or dist > pair_max:
                    continue
                used += 1
                axis = _normalize(points[j] - pi)
                center = 0.5 * (pi + points[j])
                line_dist, proj = _line_distances(points, center, axis)
                lo = float(min(proj[i], proj[j]) - support_span_pad)
                hi = float(max(proj[i], proj[j]) + support_span_pad)
                support_mask = np.logical_and.reduce((line_dist <= support_radius, proj >= lo, proj <= hi))
                support_idx = np.where(support_mask)[0]
                if support_idx.size < min_support_blobs:
                    continue
                sort_order = np.argsort(proj[support_idx])
                sorted_idx = support_idx[sort_order]
                sorted_proj = proj[sorted_idx]
                runs = _projection_runs(sorted_proj, max_gap_mm=max_gap_mm)
                for run_start, run_end in runs:
                    run_idx = sorted_idx[run_start:run_end]
                    if run_idx.size < min_support_blobs:
                        continue
                    if i not in run_idx or j not in run_idx:
                        continue
                    run_proj = proj[run_idx]
                    run_span_mm = float(np.max(run_proj) - np.min(run_proj)) if run_proj.size else 0.0
                    if run_span_mm < min_chain_span_mm:
                        continue
                    regularity = _gap_regularity_score(run_proj)
                    if regularity < min_chain_regularity:
                        continue
                    pitch_score, best_model_id = _pitch_compatibility(run_proj, diameters[run_idx], priors)
                    if pitch_score < float(cfg.get("min_pitch_score", 0.15)):
                        continue
                    support_keys = tuple(blobs[k].key for k in run_idx.tolist())
                    if support_keys in seen_support_sets:
                        continue
                    support_points = points[run_idx]
                    support_weights = weights[run_idx]
                    try:
                        fit_center, fit_axis, _ = _weighted_pca_line(support_points, support_weights)
                    except Exception:
                        continue
                    dfit, fit_proj = _line_distances(support_points, fit_center, fit_axis)
                    rms = float(np.sqrt(np.average(np.square(dfit), weights=np.maximum(support_weights, 1e-6)))) if dfit.size else 0.0
                    if rms > float(cfg.get("max_rms_mm", 2.0)):
                        continue
                    start, end = _projected_endpoints(support_points, fit_center, fit_axis, 0.10, 0.90)
                    start, end = _orient_shallow_to_deep(start, end, gating.get("head_distance_map_kji"), ctx.get("ras_to_ijk_fn"), np.asarray(ctx.get("center_ras") or [0.0, 0.0, 0.0], dtype=float))
                    span_mm = float(np.linalg.norm(end - start))
                    if span_mm < float(cfg.get("min_candidate_span_mm", 10.0)):
                        continue
                    d0 = _depth_at_ras_mm(np.asarray(start, dtype=float), gating.get("head_distance_map_kji"), ctx.get("ras_to_ijk_fn"))
                    d1 = _depth_at_ras_mm(np.asarray(end, dtype=float), gating.get("head_distance_map_kji"), ctx.get("ras_to_ijk_fn"))
                    entry_depth = min(v for v in [d0, d1] if v is not None) if any(v is not None for v in [d0, d1]) else None
                    skull_score = 1.0 - _clamp01((float(entry_depth) if entry_depth is not None else 20.0) / float(cfg.get("entry_depth_good_mm", 12.0)))
                    support_mass = float(np.sum(support_weights))
                    count_score = _clamp01((float(run_idx.size) - 3.0) / 5.0)
                    span_score = _clamp01((span_mm - 10.0) / 25.0)
                    score = 0.30 * support_mass + 0.14 * count_score + 0.14 * span_score + 0.22 * pitch_score + 0.10 * regularity + 0.10 * skull_score
                    cand_id = f"C{len(candidates)+1:04d}"
                    seen_support_sets.add(support_keys)
                    candidates.append(
                        _Candidate(
                            candidate_id=cand_id,
                            source="blob_chain",
                            start_ras=tuple(float(v) for v in start),
                            end_ras=tuple(float(v) for v in end),
                            direction_ras=tuple(float(v) for v in _normalize(np.asarray(end) - np.asarray(start))),
                            score=float(score),
                            support_mass=float(support_mass),
                            pitch_score=float(pitch_score),
                            skull_score=float(skull_score),
                            span_mm=float(span_mm),
                            rms_mm=float(rms),
                            support_blob_keys=support_keys,
                            seed_keys=(rec_i.key, blobs[j].key),
                            best_model_id=best_model_id,
                            meta={
                                "support_blob_count": int(run_idx.size),
                                "run_span_mm": float(run_span_mm),
                                "entry_depth_mm": None if entry_depth is None else float(entry_depth),
                                "regularity_score": float(regularity),
                            },
                        )
                    )
                    for local_idx, k in enumerate(run_idx.tolist()):
                        support_rows.append(
                            {
                                "candidate_id": cand_id,
                                "blob_key": blobs[k].key,
                                "blob_score": float(blobs[k].score),
                                "line_distance_mm": float(line_dist[k]),
                                "projection_mm": float(run_proj[local_idx]),
                            }
                        )
        return candidates, support_rows

    def _consensus_select(self, candidates: list[_Candidate], blobs: list[_BlobPrimitive], cfg: dict[str, Any]) -> tuple[list[_Candidate], list[dict[str, Any]]]:
        blob_map = {b.key: b for b in blobs}
        ownership: dict[str, list[_Candidate]] = {}
        for cand in candidates:
            for key in cand.support_blob_keys:
                ownership.setdefault(key, []).append(cand)
        assignment_rows: list[dict[str, Any]] = []
        assigned_support: dict[str, list[_BlobPrimitive]] = {}
        for key, cand_list in ownership.items():
            ordered = sorted(cand_list, key=lambda c: float(c.score), reverse=True)
            best = ordered[0]
            second = ordered[1] if len(ordered) > 1 else None
            resolved = int(second is not None and len(best.support_blob_keys) > len(second.support_blob_keys))
            assignment_rows.append(
                {
                    "blob_key": key,
                    "assigned_candidate_id": best.candidate_id,
                    "ambiguous": int(len(ordered) > 1),
                    "resolved_by_extra_support": resolved,
                }
            )
            assigned_support.setdefault(best.candidate_id, []).append(blob_map[key])

        rescored: list[_Candidate] = []
        for cand in candidates:
            supports = assigned_support.get(cand.candidate_id, [])
            if len(supports) < int(cfg.get("min_consensus_blobs", 4)):
                continue
            pts = np.asarray([b.point_ras for b in supports], dtype=float).reshape(-1, 3)
            w = np.asarray([float(b.score) for b in supports], dtype=float)
            center, axis, _ = _weighted_pca_line(pts, w)
            dists, proj = _line_distances(pts, center, axis)
            rms = float(np.sqrt(np.average(np.square(dists), weights=np.maximum(w, 1e-6)))) if dists.size else 0.0
            if rms > float(cfg.get("max_rms_mm", 2.0)):
                continue
            start, end = _projected_endpoints(pts, center, axis, 0.05, 0.95)
            span_mm = float(np.linalg.norm(end - start))
            if span_mm < float(cfg.get("min_consensus_span_mm", 12.0)):
                continue
            assigned_fraction = float(len(supports)) / float(max(1, len(cand.support_blob_keys)))
            if assigned_fraction < float(cfg.get("min_assigned_fraction", 0.55)):
                continue
            regularity = _gap_regularity_score(proj)
            if regularity < float(cfg.get("min_assigned_regularity", 0.15)):
                continue
            support_mass = float(np.sum(w))
            score = (
                0.42 * float(cand.score)
                + 0.22 * support_mass
                + 0.10 * _clamp01((span_mm - 12.0) / 25.0)
                + 0.08 * regularity
                + 0.08 * assigned_fraction
                + 0.05 * float(cand.pitch_score)
                + 0.05 * float(cand.skull_score)
                - 0.08 * _clamp01(rms / 2.5)
            )
            rescored.append(
                _Candidate(
                    candidate_id=cand.candidate_id,
                    source=cand.source,
                    start_ras=tuple(float(v) for v in start),
                    end_ras=tuple(float(v) for v in end),
                    direction_ras=tuple(float(v) for v in _normalize(np.asarray(end) - np.asarray(start))),
                    score=float(score),
                    support_mass=float(support_mass),
                    pitch_score=float(cand.pitch_score),
                    skull_score=float(cand.skull_score),
                    span_mm=float(span_mm),
                    rms_mm=float(rms),
                    support_blob_keys=tuple(s.key for s in supports),
                    seed_keys=tuple(cand.seed_keys),
                    best_model_id=cand.best_model_id,
                    meta={**cand.meta, "assigned_blob_count": int(len(supports)), "assigned_fraction": float(assigned_fraction), "regularity_score": float(regularity)},
                )
            )
        accepted: list[_Candidate] = []
        target_count = cfg.get("selection_target_count")
        target_count = int(target_count) if target_count is not None else None
        for cand in sorted(rescored, key=lambda c: float(c.score), reverse=True):
            dup = False
            cand_support = set(cand.support_blob_keys)
            for prev in accepted:
                prev_support = set(prev.support_blob_keys)
                overlap = float(len(cand_support.intersection(prev_support))) / float(max(1, min(len(cand_support), len(prev_support)))) if cand_support and prev_support else 0.0
                angle = _angle_deg(np.asarray(cand.direction_ras, dtype=float), np.asarray(prev.direction_ras, dtype=float))
                line_dist = _line_distance(np.asarray(cand.start_ras, dtype=float), np.asarray(cand.direction_ras, dtype=float), np.asarray(prev.start_ras, dtype=float), np.asarray(prev.direction_ras, dtype=float))
                if overlap >= float(cfg.get("selection_overlap_drop", 0.35)) or (angle <= float(cfg.get("selection_nms_angle_deg", 8.0)) and line_dist <= float(cfg.get("selection_nms_line_distance_mm", 2.5))):
                    dup = True
                    break
            if dup:
                continue
            accepted.append(cand)
            if target_count is not None and len(accepted) >= target_count:
                break
            if target_count is None and len(accepted) >= int(cfg.get("max_lines", 30)):
                break
        return accepted, assignment_rows

    def _reinforce_and_rescue(self, accepted: list[_Candidate], segments: list[_SegmentPrimitive], priors: list[_ElectrodePrior], gating: dict[str, Any], ctx: DetectionContext, cfg: dict[str, Any]) -> tuple[list[_Candidate], list[_Candidate], list[dict[str, Any]]]:
        reinforced: list[_Candidate] = []
        absorption_rows: list[dict[str, Any]] = []
        used_segment_keys: set[str] = set()
        center_ras = np.asarray(ctx.get("center_ras") or [0.0, 0.0, 0.0], dtype=float)
        for cand in accepted:
            hp0 = np.asarray(cand.start_ras, dtype=float)
            hd = np.asarray(cand.direction_ras, dtype=float)
            seg_pts = []
            seg_w = []
            local_used: list[str] = []
            for seg in segments:
                center = np.asarray(seg.point_ras, dtype=float)
                axis = np.asarray(seg.axis_ras, dtype=float)
                dist = _line_distance(center, axis, hp0, hd)
                ang = _angle_deg(axis, hd)
                absorbed = dist <= float(cfg.get("segment_absorb_distance_mm", 3.2)) and ang <= float(cfg.get("segment_absorb_angle_deg", 16.0))
                absorption_rows.append(
                    {
                        "segment_key": seg.key,
                        "status": "absorbed" if absorbed else "residual",
                        "hypothesis_id": cand.candidate_id if absorbed else "",
                        "line_distance_mm": float(dist),
                        "angle_deg": float(ang),
                        "source_kind": seg.source_kind,
                    }
                )
                if absorbed:
                    seg_pts.append(center)
                    seg_w.append(max(0.35, float(seg.score)))
                    local_used.append(seg.key)
                    used_segment_keys.add(seg.key)
            if seg_pts:
                blob_pts = [np.asarray(hp0, dtype=float), np.asarray(cand.end_ras, dtype=float)]
                blob_w = [max(1.0, cand.support_mass * 0.2), max(1.0, cand.support_mass * 0.2)]
                all_pts = np.vstack([np.asarray(blob_pts, dtype=float), np.asarray(seg_pts, dtype=float)])
                all_w = np.asarray(blob_w + seg_w, dtype=float)
                center, axis, _ = _weighted_pca_line(all_pts, all_w)
                start, end = _projected_endpoints(all_pts, center, axis, 0.05, 0.95)
                start, end = _orient_shallow_to_deep(start, end, gating.get("head_distance_map_kji"), ctx.get("ras_to_ijk_fn"), center_ras)
                span_mm = float(np.linalg.norm(end - start))
                reinforced.append(
                    _Candidate(
                        candidate_id=cand.candidate_id,
                        source="blob_consensus_reinforced",
                        start_ras=tuple(float(v) for v in start),
                        end_ras=tuple(float(v) for v in end),
                        direction_ras=tuple(float(v) for v in _normalize(np.asarray(end) - np.asarray(start))),
                        score=float(cand.score + 0.20 * len(local_used)),
                        support_mass=float(cand.support_mass + 0.5 * len(local_used)),
                        pitch_score=float(cand.pitch_score),
                        skull_score=float(cand.skull_score),
                        span_mm=float(span_mm),
                        rms_mm=float(cand.rms_mm),
                        support_blob_keys=cand.support_blob_keys,
                        seed_keys=cand.seed_keys + tuple(local_used[:3]),
                        best_model_id=cand.best_model_id,
                        meta={**cand.meta, "segment_reinforcement_count": int(len(local_used))},
                    )
                )
            else:
                reinforced.append(cand)

        rescue: list[_Candidate] = []
        residual_segments = [seg for seg in segments if seg.key not in used_segment_keys]
        for seg in residual_segments:
            if float(seg.score) < float(cfg.get("segment_rescue_min_score", 0.60)):
                continue
            if float(seg.length_mm) < float(cfg.get("segment_rescue_min_length_mm", 12.0)):
                continue
            center = np.asarray(seg.point_ras, dtype=float)
            axis = np.asarray(seg.axis_ras, dtype=float)
            start = center - 0.5 * float(seg.length_mm) * _normalize(axis)
            end = center + 0.5 * float(seg.length_mm) * _normalize(axis)
            start, end = _orient_shallow_to_deep(start, end, gating.get("head_distance_map_kji"), ctx.get("ras_to_ijk_fn"), center_ras)
            pitch_score, best_model_id = _pitch_compatibility(np.asarray([-0.5 * seg.length_mm, 0.5 * seg.length_mm], dtype=float), np.asarray([seg.diameter_mm, seg.diameter_mm], dtype=float), priors)
            d0 = _depth_at_ras_mm(np.asarray(start, dtype=float), gating.get("head_distance_map_kji"), ctx.get("ras_to_ijk_fn"))
            d1 = _depth_at_ras_mm(np.asarray(end, dtype=float), gating.get("head_distance_map_kji"), ctx.get("ras_to_ijk_fn"))
            entry_depth = min(v for v in [d0, d1] if v is not None) if any(v is not None for v in [d0, d1]) else None
            skull_score = 1.0 - _clamp01((float(entry_depth) if entry_depth is not None else 20.0) / float(cfg.get("entry_depth_good_mm", 12.0)))
            rescue.append(
                _Candidate(
                    candidate_id=f"R{len(rescue)+1:04d}",
                    source="segment_rescue",
                    start_ras=tuple(float(v) for v in start),
                    end_ras=tuple(float(v) for v in end),
                    direction_ras=tuple(float(v) for v in _normalize(np.asarray(end) - np.asarray(start))),
                    score=float(0.50 * seg.score + 0.25 * pitch_score + 0.25 * skull_score),
                    support_mass=float(seg.score),
                    pitch_score=float(pitch_score),
                    skull_score=float(skull_score),
                    span_mm=float(seg.length_mm),
                    rms_mm=float(seg.meta.get("residual_rms_mm", 0.0)),
                    support_blob_keys=(seg.key,),
                    seed_keys=(seg.key,),
                    best_model_id=best_model_id,
                    meta={"source_kind": seg.source_kind},
                )
            )

        final_candidates: list[_Candidate] = []
        for cand in sorted(reinforced + rescue, key=lambda c: float(c.score), reverse=True):
            dup = False
            for prev in final_candidates:
                angle = _angle_deg(np.asarray(cand.direction_ras, dtype=float), np.asarray(prev.direction_ras, dtype=float))
                line_dist = _line_distance(np.asarray(cand.start_ras, dtype=float), np.asarray(cand.direction_ras, dtype=float), np.asarray(prev.start_ras, dtype=float), np.asarray(prev.direction_ras, dtype=float))
                if angle <= float(cfg.get("final_nms_angle_deg", 8.0)) and line_dist <= float(cfg.get("final_nms_line_distance_mm", 3.0)):
                    dup = True
                    break
            if dup:
                continue
            final_candidates.append(cand)
            if len(final_candidates) >= int(cfg.get("max_lines", 30)):
                break
        return final_candidates, rescue, absorption_rows

    def run(self, ctx: DetectionContext) -> DetectionResult:
        t_start = time.perf_counter()
        result = self.make_result(ctx)
        diagnostics = self.diagnostics(result)
        cfg = self._config(ctx)
        try:
            cfg.setdefault("_electrode_family_priors", _load_family_priors())
            if "arr_kji" not in ctx or "ijk_kji_to_ras_fn" not in ctx:
                result["warnings"].append("blob_consensus_v1 missing volume context; returning empty result")
                return self.finalize(result, diagnostics, t_start)

            gating = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="gating", fn=lambda: self._build_gating(ctx, cfg))
            primitives = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="primitive_extraction", fn=lambda: self._extract_primitives(ctx, gating, cfg))
            priors = list(cfg.get("_electrode_family_priors") or [])
            blob_candidates, support_rows = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="blob_consensus_proposal", fn=lambda: self._build_blob_candidates(primitives["blobs"], priors, gating, ctx, cfg))
            accepted, assignment_rows = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="blob_consensus_selection", fn=lambda: self._consensus_select(blob_candidates, primitives["blobs"], cfg))
            final_lines, rescue_lines, absorption_rows = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="segment_reinforcement", fn=lambda: self._reinforce_and_rescue(accepted, primitives["segments"], priors, gating, ctx, cfg))

            result["trajectories"] = [
                {
                    "name": str(line.meta.get("name") or f"P{idx:02d}"),
                    "start_ras": list(line.start_ras),
                    "end_ras": list(line.end_ras),
                    "length_mm": float(np.linalg.norm(np.asarray(line.end_ras, dtype=float) - np.asarray(line.start_ras, dtype=float))),
                    "confidence": float(min(1.0, max(0.0, float(line.score) / 10.0))),
                    "support_count": int(len(line.support_blob_keys)),
                    "params": {
                        "selection_score": float(line.score),
                        "support_mass": float(line.support_mass),
                        "pitch_score": float(line.pitch_score),
                        "skull_score": float(line.skull_score),
                        "span_mm": float(line.span_mm),
                        "rms_mm": float(line.rms_mm),
                        "best_model_id": line.best_model_id,
                        "source": line.source,
                    },
                }
                for idx, line in enumerate(final_lines, start=1)
            ]
            result["contacts"] = []
            result["warnings"].append("contact_detection_not_implemented")

            diagnostics.set_count("threshold_levels", int(len(cfg.get("threshold_schedule_hu", [2600.0, 2200.0, 1800.0, 1500.0, 1200.0]))))
            diagnostics.set_count("primitive_blob_count", int(len(primitives["blobs"])))
            diagnostics.set_count("primitive_segment_count", int(sum(1 for s in primitives["segments"] if s.source_kind == "segment")))
            diagnostics.set_count("primitive_merged_subline_count", int(sum(1 for s in primitives["segments"] if s.source_kind == "merged_subline")))
            diagnostics.set_count("candidate_blob_consensus", int(len(blob_candidates)))
            diagnostics.set_count("accepted_blob_consensus", int(len(accepted)))
            diagnostics.set_count("segment_rescue_count", int(len(rescue_lines)))
            diagnostics.set_count("final_lines_kept", int(len(final_lines)))
            diagnostics.set_extra(
                "primitive_counts",
                {
                    "blobs": len(primitives["blobs"]),
                    "segments": int(sum(1 for s in primitives["segments"] if s.source_kind == "segment")),
                    "merged_sublines": int(sum(1 for s in primitives["segments"] if s.source_kind == "merged_subline")),
                    "merged_components": len(primitives["merged_rows"]),
                },
            )
            diagnostics.set_extra("electrode_prior_count", int(len(priors)))
            diagnostics.set_extra(
                "consensus_counts",
                {
                    "support_assignments": len(assignment_rows),
                    "ambiguous_blob_count": int(sum(int(r["ambiguous"]) for r in assignment_rows)),
                    "resolved_by_extra_support": int(sum(int(r["resolved_by_extra_support"]) for r in assignment_rows)),
                },
            )
            diagnostics.note("blob_consensus_v1 seeds from compact-blob consensus, then reinforces or rescues with segment evidence")

            writer = self.get_artifact_writer(ctx, result)
            primitive_path = writer.write_csv_rows(
                "blob_consensus_primitives.csv",
                ["key", "kind", "lineage_id", "x", "y", "z", "axis_x", "axis_y", "axis_z", "score", "length_mm", "diameter_mm", "threshold_hu", "depth_mm"],
                [
                    [
                        rec.key,
                        rec.kind,
                        int(rec.lineage_id),
                        float(rec.point_ras[0]),
                        float(rec.point_ras[1]),
                        float(rec.point_ras[2]),
                        float(rec.axis_ras[0]),
                        float(rec.axis_ras[1]),
                        float(rec.axis_ras[2]),
                        float(rec.score),
                        float(rec.length_mm),
                        float(rec.diameter_mm),
                        float(rec.threshold_hu),
                        float(rec.depth_mm),
                    ]
                    for rec in primitives["blobs"]
                ]
                + [
                    [
                        rec.key,
                        rec.source_kind,
                        int(rec.lineage_id),
                        float(rec.point_ras[0]),
                        float(rec.point_ras[1]),
                        float(rec.point_ras[2]),
                        float(rec.axis_ras[0]),
                        float(rec.axis_ras[1]),
                        float(rec.axis_ras[2]),
                        float(rec.score),
                        float(rec.length_mm),
                        float(rec.diameter_mm),
                        float(rec.threshold_hu),
                        float(rec.depth_mm),
                    ]
                    for rec in primitives["segments"]
                ],
            )
            add_artifact(result["artifacts"], kind="primitives_csv", path=primitive_path, description="blob-consensus primitives", stage="primitive_extraction")
            candidate_path = writer.write_csv_rows(
                "blob_consensus_candidates.csv",
                ["candidate_id", "source", "score", "support_mass", "pitch_score", "skull_score", "span_mm", "rms_mm", "support_blob_count", "best_model_id"],
                [
                    [c.candidate_id, c.source, float(c.score), float(c.support_mass), float(c.pitch_score), float(c.skull_score), float(c.span_mm), float(c.rms_mm), int(len(c.support_blob_keys)), c.best_model_id or ""]
                    for c in blob_candidates
                ],
            )
            add_artifact(result["artifacts"], kind="candidates_csv", path=candidate_path, description="blob-consensus candidate lines", stage="blob_consensus_proposal")
            assign_path = writer.write_csv_rows(
                "blob_consensus_assignments.csv",
                ["blob_key", "assigned_candidate_id", "ambiguous", "resolved_by_extra_support"],
                [[r["blob_key"], r["assigned_candidate_id"], int(r["ambiguous"]), int(r["resolved_by_extra_support"])] for r in assignment_rows],
            )
            add_artifact(result["artifacts"], kind="assignments_csv", path=assign_path, description="blob ownership assignments", stage="blob_consensus_selection")
            absorb_path = writer.write_csv_rows(
                "blob_consensus_segment_absorption.csv",
                ["segment_key", "status", "hypothesis_id", "line_distance_mm", "angle_deg", "source_kind"],
                [[r["segment_key"], r["status"], r["hypothesis_id"], float(r["line_distance_mm"]), float(r["angle_deg"]), r["source_kind"]] for r in absorption_rows],
            )
            add_artifact(result["artifacts"], kind="segment_absorption_csv", path=absorb_path, description="segment absorption/residual table", stage="segment_reinforcement")

            write_standard_artifacts(
                writer,
                result,
                pipeline_payload={
                    "primitive_counts": result["diagnostics"].get("extras", {}).get("primitive_counts", {}),
                    "candidate_count": len(blob_candidates),
                    "accepted_candidate_count": len(accepted),
                    "final_line_count": len(final_lines),
                },
            )
            return self.finalize(result, diagnostics, t_start)
        except Exception as exc:
            self.fail(ctx=ctx, result=result, diagnostics=diagnostics, stage="pipeline", exc=exc)
            return self.finalize(result, diagnostics, t_start)
