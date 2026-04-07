#!/usr/bin/env python3
"""Analyze blob/segment consensus evidence for SEEG shank detection.

This is an analysis-only tool. It does not implement a detector. It builds
multi-threshold connected components, classifies them into compact blobs,
line-like segments, merged multi-line candidates, and junk, then studies a
blob-consensus voting scheme plus segment absorption/rescue behavior against
cached ground-truth shanks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIB_DIR = os.path.join(REPO_ROOT, "CommonLib")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

from rosa_core.electrode_models import load_electrode_library  # noqa: E402
from shank_core.masking import build_preview_masks  # noqa: E402
from shank_engine.lineage_tracking import build_lineages, extract_threshold_levels, summarize_lineages  # noqa: E402
from tools.eval_seeg_localization import (  # noqa: E402
    GroundTruthShank,
    PredictedShank,
    _point_to_segment_distance,
    _unit,
    build_detection_context,
    compare_shanks,
    iter_subject_rows,
    load_ground_truth_shanks,
)


@dataclass(frozen=True)
class ElectrodePrior:
    model_id: str
    pitch_mm: float
    diameter_mm: float
    total_length_mm: float
    contact_count: int


@dataclass(frozen=True)
class PrimitiveRecord:
    subject_id: str
    threshold_hu: float
    level_index: int
    blob_id: int
    lineage_id: int
    kind: str
    score: float
    point_ras: tuple[float, float, float]
    axis_ras: tuple[float, float, float]
    length_mm: float
    diameter_mm: float
    residual_rms_mm: float
    depth_mean_mm: float
    grown_support_ratio: float
    grown_axis_stability_deg: float
    voxel_count: int
    meta: dict[str, Any]


@dataclass(frozen=True)
class HypothesisRecord:
    subject_id: str
    hypothesis_id: str
    source: str
    start_ras: tuple[float, float, float]
    end_ras: tuple[float, float, float]
    direction_ras: tuple[float, float, float]
    score: float
    support_mass: float
    pitch_score: float
    skull_score: float
    support_blob_ids: tuple[str, ...]
    seed_blob_ids: tuple[str, ...]
    best_model_id: str | None
    meta: dict[str, Any]


def _write_tsv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def _normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(3)
    n = float(np.linalg.norm(arr))
    if n <= 1e-9:
        return np.asarray([0.0, 0.0, 1.0], dtype=float)
    return arr / n


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


def _fit_line_stats(points_ras: np.ndarray) -> dict[str, Any]:
    pts = np.asarray(points_ras, dtype=float).reshape(-1, 3)
    if pts.shape[0] == 0:
        raise ValueError("need at least 1 point")
    if pts.shape[0] == 1:
        center = pts[0].astype(float)
        return {
            "center_ras": center,
            "axis_ras": np.asarray([0.0, 0.0, 1.0], dtype=float),
            "evals": np.asarray([0.0, 0.0, 0.0], dtype=float),
            "length_mm": 0.0,
            "diameter_mm": 0.0,
            "residual_rms_mm": 0.0,
            "aspect_ratio": 0.0,
            "proj_min": 0.0,
            "proj_max": 0.0,
        }
    center, axis, evals = _weighted_pca_line(pts)
    dists, t = _line_distances(pts, center, axis)
    length_mm = float(np.max(t) - np.min(t)) if pts.shape[0] >= 2 else 0.0
    diameter_mm = float(2.0 * math.sqrt(max(1e-9, float(np.mean(np.maximum(evals[:2], 0.0))))))
    rms = float(np.sqrt(np.mean(np.square(dists)))) if dists.size else 0.0
    aspect = float(length_mm / max(diameter_mm, 0.25))
    return {
        "center_ras": center.astype(float),
        "axis_ras": axis.astype(float),
        "evals": evals.astype(float),
        "length_mm": float(length_mm),
        "diameter_mm": float(diameter_mm),
        "residual_rms_mm": float(rms),
        "aspect_ratio": float(aspect),
        "proj_min": float(np.min(t)) if t.size else 0.0,
        "proj_max": float(np.max(t)) if t.size else 0.0,
    }


def _sample_points(points: np.ndarray, max_points: int = 4000) -> np.ndarray:
    pts = np.asarray(points)
    if pts.shape[0] <= max_points:
        return pts
    stride = int(max(1, math.ceil(float(pts.shape[0]) / float(max_points))))
    return pts[::stride]


def _dilate_one_voxel(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    padded = np.pad(arr, 1, mode="constant", constant_values=False)
    out = np.zeros_like(arr, dtype=bool)
    for dk in range(3):
        for dj in range(3):
            for di in range(3):
                out |= padded[dk:dk + arr.shape[0], dj:dj + arr.shape[1], di:di + arr.shape[2]]
    return out


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


def _orient_shallow_to_deep(start: np.ndarray, end: np.ndarray, depth_kji: np.ndarray | None, ras_to_ijk_fn, center_ras: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d0 = _depth_at_ras_mm(start, depth_kji, ras_to_ijk_fn)
    d1 = _depth_at_ras_mm(end, depth_kji, ras_to_ijk_fn)
    if d0 is not None and d1 is not None and abs(float(d0) - float(d1)) > 1e-3:
        return (start, end) if float(d0) <= float(d1) else (end, start)
    c0 = float(np.linalg.norm(start - center_ras))
    c1 = float(np.linalg.norm(end - center_ras))
    return (start, end) if c0 >= c1 else (end, start)


def _center_pitch_from_model(model: dict[str, Any]) -> float | None:
    c2c = model.get("center_to_center_separation_mm")
    if c2c is not None:
        val = float(c2c)
        if np.isfinite(val) and val > 0.0:
            return float(val)
    offsets = list(model.get("contact_center_offsets_from_tip_mm") or [])
    if len(offsets) >= 2:
        diffs = np.diff(np.asarray(offsets, dtype=float))
        if diffs.size:
            return float(np.median(diffs))
    return None


def _load_family_priors() -> list[ElectrodePrior]:
    library = load_electrode_library()
    priors: list[ElectrodePrior] = []
    for model in list(library.get("models") or []):
        pitch = _center_pitch_from_model(model)
        if pitch is None or pitch <= 0.0:
            continue
        priors.append(
            ElectrodePrior(
                model_id=str(model.get("id") or "unknown"),
                pitch_mm=float(pitch),
                diameter_mm=float(model.get("diameter_mm") or 0.8),
                total_length_mm=float(model.get("total_exploration_length_mm") or 0.0),
                contact_count=int(model.get("contact_count") or 0),
            )
        )
    return priors


def _pitch_compatibility(projections_mm: np.ndarray, diameters_mm: np.ndarray, priors: list[ElectrodePrior]) -> tuple[float, str | None]:
    proj = np.sort(np.asarray(projections_mm, dtype=float).reshape(-1))
    if proj.size < 2 or not priors:
        return 0.0, None
    raw_gaps = np.diff(proj)
    gaps = raw_gaps[np.logical_and(raw_gaps >= 1.0, raw_gaps <= 8.0)]
    if gaps.size == 0:
        return 0.0, None
    median_gap = float(np.median(gaps))
    span = float(np.max(proj) - np.min(proj))
    obs_count = int(proj.size)
    mean_diam = float(np.mean(np.asarray(diameters_mm, dtype=float).reshape(-1))) if np.asarray(diameters_mm).size else 0.8
    best_score = 0.0
    best_id = None
    for prior in priors:
        gap_score = 1.0 - _clamp01(abs(median_gap - prior.pitch_mm) / max(1.0, 0.6 * prior.pitch_mm))
        span_score = 1.0 - _clamp01(abs(span - prior.total_length_mm) / max(8.0, 0.8 * prior.total_length_mm))
        diam_score = 1.0 - _clamp01(abs(mean_diam - prior.diameter_mm) / 2.5)
        count_score = _clamp01(float(obs_count) / float(max(2, min(prior.contact_count, 6))))
        score = 0.45 * gap_score + 0.25 * span_score + 0.15 * diam_score + 0.15 * count_score
        if score > best_score:
            best_score = float(score)
            best_id = str(prior.model_id)
    return float(best_score), best_id


def _node_map_from_lineages(lineages: list[dict[str, Any]]) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for lineage in lineages:
        lineage_id = int(lineage["lineage_id"])
        for node in list(lineage.get("nodes") or []):
            level_index = int(node.get("level_index", 0))
            blob_id = int((node.get("blob") or {}).get("blob_id", 0))
            out[(level_index, blob_id)] = lineage_id
    return out


def _bbox_points_from_mask(mask: np.ndarray, slc: tuple[slice, slice, slice]) -> np.ndarray:
    sub = np.asarray(mask[slc], dtype=bool)
    coords = np.argwhere(sub)
    if coords.size == 0:
        return coords.reshape(-1, 3)
    offs = np.asarray([slc[0].start or 0, slc[1].start or 0, slc[2].start or 0], dtype=int)
    return (coords + offs.reshape(1, 3)).astype(int)


def _grown_features(raw_points_kji: np.ndarray, raw_axis_ras: np.ndarray, raw_length_mm: float, grown_mask: np.ndarray, ijk_kji_to_ras_fn) -> dict[str, Any]:
    pts = np.asarray(raw_points_kji, dtype=int).reshape(-1, 3)
    k0, j0, i0 = np.maximum(np.min(pts, axis=0) - 1, 0)
    k1, j1, i1 = np.min([np.max(pts, axis=0) + 2, np.asarray(grown_mask.shape)], axis=0)
    slc = (slice(int(k0), int(k1)), slice(int(j0), int(j1)), slice(int(i0), int(i1)))
    grown_points_kji = _bbox_points_from_mask(grown_mask, slc)
    if grown_points_kji.size == 0:
        return {"grown_voxel_count": 0, "grown_support_ratio": 0.0, "grown_axis_stability_deg": 180.0, "grown_length_mm": 0.0}
    grown_points_ras = np.asarray(ijk_kji_to_ras_fn(grown_points_kji), dtype=float).reshape(-1, 3)
    stats = _fit_line_stats(_sample_points(grown_points_ras, max_points=2500))
    return {
        "grown_voxel_count": int(grown_points_kji.shape[0]),
        "grown_support_ratio": float(grown_points_kji.shape[0]) / float(max(1, pts.shape[0])),
        "grown_axis_stability_deg": float(_angle_deg(np.asarray(raw_axis_ras, dtype=float), np.asarray(stats["axis_ras"], dtype=float))),
        "grown_length_mm": float(stats["length_mm"]),
    }


def _classify_component(blob: dict[str, Any], fit_stats: dict[str, Any], grown_stats: dict[str, Any]) -> tuple[str, float]:
    vox = int(blob.get("voxel_count") or 0)
    length = float(fit_stats["length_mm"])
    diameter = float(fit_stats["diameter_mm"])
    residual = float(fit_stats["residual_rms_mm"])
    aspect = float(fit_stats["aspect_ratio"])
    depth = float(blob.get("depth_mean") or 0.0)
    grown_ratio = float(grown_stats.get("grown_support_ratio", 0.0))
    grown_axis = float(grown_stats.get("grown_axis_stability_deg", 180.0))

    # Evaluate compact blobs first. Tiny or compact components should not be
    # swallowed by the merged-segment bucket simply because a 1-voxel grow adds
    # local support around them.
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
    return "junk", float(0.0)


def _decompose_merged_blob(points_ras: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    pts = np.asarray(points_ras, dtype=float).reshape(-1, 3)
    if pts.shape[0] < int(cfg.get("decomp_min_points", 20)):
        return {"sub_lines": [], "explained_fraction": 0.0, "remaining_fraction": 1.0}
    remaining = pts.copy()
    original_count = int(pts.shape[0])
    sub_lines: list[dict[str, Any]] = []
    max_lines = int(cfg.get("decomp_max_lines", 4))
    radius_mm = float(cfg.get("decomp_line_radius_mm", 2.0))
    min_frac = float(cfg.get("decomp_min_fraction_per_line", 0.15))
    min_pts = int(cfg.get("decomp_min_points_per_line", 12))

    for _ in range(max_lines):
        if remaining.shape[0] < min_pts:
            break
        sample = _sample_points(remaining, max_points=int(cfg.get("decomp_fit_sample_points", 2500)))
        try:
            stats0 = _fit_line_stats(sample)
        except Exception:
            break
        dists, _t = _line_distances(remaining, np.asarray(stats0["center_ras"], dtype=float), np.asarray(stats0["axis_ras"], dtype=float))
        inlier_mask = dists <= radius_mm
        if int(np.count_nonzero(inlier_mask)) < min_pts:
            break
        inliers = remaining[inlier_mask]
        if float(inliers.shape[0]) / float(max(1, original_count)) < min_frac:
            break
        stats = _fit_line_stats(_sample_points(inliers, max_points=int(cfg.get("decomp_fit_sample_points", 2500))))
        dists_refined, _ = _line_distances(inliers, np.asarray(stats["center_ras"], dtype=float), np.asarray(stats["axis_ras"], dtype=float))
        residual = float(np.sqrt(np.mean(np.square(dists_refined)))) if dists_refined.size else float("inf")
        if residual > float(cfg.get("decomp_max_residual_mm", 1.8)):
            break
        start, end = _projected_endpoints(inliers, np.asarray(stats["center_ras"], dtype=float), np.asarray(stats["axis_ras"], dtype=float), 0.05, 0.95)
        sub_lines.append(
            {
                "center_ras": np.asarray(stats["center_ras"], dtype=float),
                "axis_ras": np.asarray(stats["axis_ras"], dtype=float),
                "length_mm": float(stats["length_mm"]),
                "diameter_mm": float(stats["diameter_mm"]),
                "residual_rms_mm": float(residual),
                "point_count": int(inliers.shape[0]),
                "explained_fraction": float(inliers.shape[0]) / float(max(1, original_count)),
                "start_ras": start.astype(float),
                "end_ras": end.astype(float),
            }
        )
        remaining = remaining[~inlier_mask]
        if remaining.shape[0] < min_pts:
            break
    explained_count = int(sum(int(line["point_count"]) for line in sub_lines))
    return {
        "sub_lines": sub_lines,
        "explained_fraction": float(explained_count) / float(max(1, original_count)),
        "remaining_fraction": float(max(0, original_count - explained_count)) / float(max(1, original_count)),
    }


def _pred_from_line(name: str, start_ras: np.ndarray, end_ras: np.ndarray, support_count: int = 0, confidence: float = 0.0) -> PredictedShank:
    direction = _unit(np.asarray(end_ras, dtype=float) - np.asarray(start_ras, dtype=float))
    return PredictedShank(
        name=str(name),
        start_ras=tuple(float(v) for v in np.asarray(start_ras, dtype=float)),
        end_ras=tuple(float(v) for v in np.asarray(end_ras, dtype=float)),
        direction_ras=tuple(float(v) for v in direction),
        length_mm=float(np.linalg.norm(np.asarray(end_ras, dtype=float) - np.asarray(start_ras, dtype=float))),
        support_count=int(support_count),
        confidence=float(confidence),
    )


def _primitive_score_row(rec: PrimitiveRecord) -> dict[str, Any]:
    return {
        "subject_id": rec.subject_id,
        "threshold_hu": f"{rec.threshold_hu:.2f}",
        "level_index": int(rec.level_index),
        "blob_id": int(rec.blob_id),
        "lineage_id": int(rec.lineage_id),
        "kind": rec.kind,
        "score": f"{rec.score:.4f}",
        "x": f"{rec.point_ras[0]:.4f}",
        "y": f"{rec.point_ras[1]:.4f}",
        "z": f"{rec.point_ras[2]:.4f}",
        "axis_x": f"{rec.axis_ras[0]:.6f}",
        "axis_y": f"{rec.axis_ras[1]:.6f}",
        "axis_z": f"{rec.axis_ras[2]:.6f}",
        "length_mm": f"{rec.length_mm:.4f}",
        "diameter_mm": f"{rec.diameter_mm:.4f}",
        "residual_rms_mm": f"{rec.residual_rms_mm:.4f}",
        "depth_mean_mm": f"{rec.depth_mean_mm:.4f}",
        "grown_support_ratio": f"{rec.grown_support_ratio:.4f}",
        "grown_axis_stability_deg": f"{rec.grown_axis_stability_deg:.4f}",
        "voxel_count": int(rec.voxel_count),
        "meta_json": json.dumps(rec.meta, sort_keys=True),
    }


def _hypothesis_row(h: HypothesisRecord) -> dict[str, Any]:
    support_only_score = float(h.meta.get("score_support_only", 0.0))
    score_no_pitch = float(h.meta.get("score_no_pitch", h.score))
    score_no_skull = float(h.meta.get("score_no_skull", h.score))
    return {
        "subject_id": h.subject_id,
        "hypothesis_id": h.hypothesis_id,
        "source": h.source,
        "start_x": f"{h.start_ras[0]:.4f}",
        "start_y": f"{h.start_ras[1]:.4f}",
        "start_z": f"{h.start_ras[2]:.4f}",
        "end_x": f"{h.end_ras[0]:.4f}",
        "end_y": f"{h.end_ras[1]:.4f}",
        "end_z": f"{h.end_ras[2]:.4f}",
        "dir_x": f"{h.direction_ras[0]:.6f}",
        "dir_y": f"{h.direction_ras[1]:.6f}",
        "dir_z": f"{h.direction_ras[2]:.6f}",
        "score": f"{h.score:.4f}",
        "score_support_only": f"{support_only_score:.4f}",
        "score_no_pitch": f"{score_no_pitch:.4f}",
        "score_no_skull": f"{score_no_skull:.4f}",
        "support_mass": f"{h.support_mass:.4f}",
        "pitch_score": f"{h.pitch_score:.4f}",
        "skull_score": f"{h.skull_score:.4f}",
        "support_blob_ids": json.dumps(list(h.support_blob_ids)),
        "seed_blob_ids": json.dumps(list(h.seed_blob_ids)),
        "best_model_id": h.best_model_id or "",
        "meta_json": json.dumps(h.meta, sort_keys=True),
    }


def _make_component_primitives(
    *,
    subject_id: str,
    levels: list[dict[str, Any]],
    lineages: list[dict[str, Any]],
    gt_shanks: list[GroundTruthShank],
    ctx: dict[str, Any],
    gating: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[list[PrimitiveRecord], list[dict[str, Any]], list[dict[str, Any]]]:
    node_map = _node_map_from_lineages(lineages)
    components: list[PrimitiveRecord] = []
    decomposition_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    for level_index, level in enumerate(levels):
        threshold_hu = float(level["threshold_hu"])
        raw_mask = np.asarray(level["labels_kji"], dtype=np.int32) > 0
        grown_mask = _dilate_one_voxel(raw_mask)
        blobs = list(level.get("blobs") or [])
        labels = np.asarray(level["labels_kji"], dtype=np.int32)
        for blob in blobs:
            blob_id = int(blob["blob_id"])
            lineage_id = int(node_map.get((int(level_index), blob_id), 0))
            pts_kji = np.argwhere(labels == blob_id)
            if pts_kji.size == 0:
                continue
            pts_ras = np.asarray(ctx["ijk_kji_to_ras_fn"](_sample_points(pts_kji, max_points=int(cfg.get("component_fit_sample_points", 3000)))), dtype=float).reshape(-1, 3)
            fit_stats = _fit_line_stats(pts_ras)
            grown_stats = _grown_features(pts_kji, np.asarray(fit_stats["axis_ras"], dtype=float), float(fit_stats["length_mm"]), grown_mask, ctx["ijk_kji_to_ras_fn"])
            kind, score = _classify_component(blob, fit_stats, grown_stats)
            rec = PrimitiveRecord(
                subject_id=subject_id,
                threshold_hu=threshold_hu,
                level_index=int(level_index),
                blob_id=blob_id,
                lineage_id=lineage_id,
                kind=str(kind),
                score=float(score),
                point_ras=tuple(float(v) for v in np.asarray(fit_stats["center_ras"], dtype=float)),
                axis_ras=tuple(float(v) for v in np.asarray(fit_stats["axis_ras"], dtype=float)),
                length_mm=float(fit_stats["length_mm"]),
                diameter_mm=float(fit_stats["diameter_mm"]),
                residual_rms_mm=float(fit_stats["residual_rms_mm"]),
                depth_mean_mm=float(blob.get("depth_mean") or 0.0),
                grown_support_ratio=float(grown_stats["grown_support_ratio"]),
                grown_axis_stability_deg=float(grown_stats["grown_axis_stability_deg"]),
                voxel_count=int(blob.get("voxel_count") or 0),
                meta={
                    "aspect_ratio": float(fit_stats["aspect_ratio"]),
                    "hu_q95": float(blob.get("hu_q95") or blob.get("hu_max") or 0.0),
                    "topology_source": "raw_component",
                    "grown_voxel_count": int(grown_stats["grown_voxel_count"]),
                    "grown_length_mm": float(grown_stats["grown_length_mm"]),
                },
            )
            components.append(rec)
            if rec.kind == "segment":
                segment_rows.append(
                    {
                        "subject_id": subject_id,
                        "source_kind": "segment",
                        "threshold_hu": f"{threshold_hu:.2f}",
                        "blob_key": f"L{level_index}_B{blob_id}",
                        "lineage_id": int(lineage_id),
                        "x": f"{rec.point_ras[0]:.4f}",
                        "y": f"{rec.point_ras[1]:.4f}",
                        "z": f"{rec.point_ras[2]:.4f}",
                        "axis_x": f"{rec.axis_ras[0]:.6f}",
                        "axis_y": f"{rec.axis_ras[1]:.6f}",
                        "axis_z": f"{rec.axis_ras[2]:.6f}",
                        "length_mm": f"{rec.length_mm:.4f}",
                        "diameter_mm": f"{rec.diameter_mm:.4f}",
                        "score": f"{rec.score:.4f}",
                    }
                )
            elif rec.kind == "merged_segment":
                full_pts_ras = np.asarray(ctx["ijk_kji_to_ras_fn"](_sample_points(pts_kji, max_points=int(cfg.get("decomp_max_points", 5000)))), dtype=float).reshape(-1, 3)
                decomp = _decompose_merged_blob(full_pts_ras, cfg)
                align_ids: list[str] = []
                for sub_idx, sub in enumerate(list(decomp["sub_lines"])):
                    sub_center = np.asarray(sub["center_ras"], dtype=float)
                    sub_axis = np.asarray(sub["axis_ras"], dtype=float)
                    matched_gt: list[str] = []
                    for gt in gt_shanks:
                        gt_start = np.asarray(gt.start_ras, dtype=float)
                        gt_end = np.asarray(gt.end_ras, dtype=float)
                        dist = _point_to_segment_distance(sub_center, gt_start, gt_end)
                        ang = _angle_deg(sub_axis, np.asarray(gt.direction_ras, dtype=float))
                        if dist <= float(cfg.get("decomp_gt_match_radius_mm", 3.5)) and ang <= float(cfg.get("decomp_gt_match_angle_deg", 18.0)):
                            matched_gt.append(str(gt.shank))
                    segment_rows.append(
                        {
                            "subject_id": subject_id,
                            "source_kind": "merged_subline",
                            "threshold_hu": f"{threshold_hu:.2f}",
                            "blob_key": f"L{level_index}_B{blob_id}_S{sub_idx+1}",
                            "lineage_id": int(lineage_id),
                            "x": f"{float(np.asarray(sub['center_ras'])[0]):.4f}",
                            "y": f"{float(np.asarray(sub['center_ras'])[1]):.4f}",
                            "z": f"{float(np.asarray(sub['center_ras'])[2]):.4f}",
                            "axis_x": f"{float(np.asarray(sub['axis_ras'])[0]):.6f}",
                            "axis_y": f"{float(np.asarray(sub['axis_ras'])[1]):.6f}",
                            "axis_z": f"{float(np.asarray(sub['axis_ras'])[2]):.6f}",
                            "length_mm": f"{float(sub['length_mm']):.4f}",
                            "diameter_mm": f"{float(sub['diameter_mm']):.4f}",
                            "score": f"{float(sub['explained_fraction']):.4f}",
                        }
                    )
                    align_ids.extend(matched_gt)
                decomposition_rows.append(
                    {
                        "subject_id": subject_id,
                        "threshold_hu": f"{threshold_hu:.2f}",
                        "level_index": int(level_index),
                        "blob_id": int(blob_id),
                        "lineage_id": int(lineage_id),
                        "sub_line_count": int(len(decomp["sub_lines"])),
                        "explained_fraction": f"{float(decomp['explained_fraction']):.4f}",
                        "remaining_fraction": f"{float(decomp['remaining_fraction']):.4f}",
                        "length_mm": f"{rec.length_mm:.4f}",
                        "diameter_mm": f"{rec.diameter_mm:.4f}",
                        "residual_rms_mm": f"{rec.residual_rms_mm:.4f}",
                        "aligned_gt_count": int(len(set(align_ids))),
                        "aligned_gt_mode": "multiple" if len(set(align_ids)) > 1 else ("single" if len(set(align_ids)) == 1 else "none"),
                        "aligned_gt_shanks_json": json.dumps(sorted(set(align_ids))),
                        "sub_lines_json": json.dumps(
                            [
                                {
                                    "axis_ras": [float(v) for v in np.asarray(s["axis_ras"], dtype=float)],
                                    "length_mm": float(s["length_mm"]),
                                    "diameter_mm": float(s["diameter_mm"]),
                                    "residual_rms_mm": float(s["residual_rms_mm"]),
                                    "explained_fraction": float(s["explained_fraction"]),
                                }
                                for s in list(decomp["sub_lines"])
                            ],
                            sort_keys=True,
                        ),
                    }
                )
    return components, decomposition_rows, segment_rows


def _dedup_blob_primitives(components: list[PrimitiveRecord], radius_mm: float) -> list[PrimitiveRecord]:
    kept: list[PrimitiveRecord] = []
    blobs = [c for c in components if c.kind == "blob"]
    for rec in sorted(blobs, key=lambda x: (-float(x.score), float(x.threshold_hu), x.lineage_id, x.blob_id)):
        pt = np.asarray(rec.point_ras, dtype=float)
        if any(float(np.linalg.norm(pt - np.asarray(k.point_ras, dtype=float))) <= float(radius_mm) for k in kept):
            continue
        kept.append(rec)
    return kept


def _build_blob_hypotheses(
    *,
    subject_id: str,
    blob_primitives: list[PrimitiveRecord],
    gt_shanks: list[GroundTruthShank],
    priors: list[ElectrodePrior],
    gating: dict[str, Any],
    ctx: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[list[HypothesisRecord], list[dict[str, Any]], list[HypothesisRecord], list[dict[str, Any]]]:
    if len(blob_primitives) < 2:
        return [], [], [], []
    points = np.asarray([b.point_ras for b in blob_primitives], dtype=float).reshape(-1, 3)
    scores = np.asarray([float(b.score) for b in blob_primitives], dtype=float).reshape(-1)
    diameters = np.asarray([float(b.diameter_mm) for b in blob_primitives], dtype=float).reshape(-1)
    max_neighbors = int(cfg.get("max_blob_neighbors", 12))
    pair_min = float(cfg.get("hypothesis_min_pair_distance_mm", 8.0))
    pair_max = float(cfg.get("hypothesis_max_pair_distance_mm", 90.0))
    support_radius = float(cfg.get("hypothesis_support_radius_mm", 2.8))
    support_span_pad = float(cfg.get("hypothesis_support_span_pad_mm", 5.0))
    min_support_blobs = int(cfg.get("hypothesis_min_support_blobs", 3))

    raw_hyps: list[HypothesisRecord] = []
    support_rows: list[dict[str, Any]] = []
    for i, rec_i in enumerate(blob_primitives):
        pi = points[i]
        dists = np.linalg.norm(points - pi.reshape(1, 3), axis=1)
        neighbor_idx = np.argsort(dists)
        count = 0
        for j in neighbor_idx.tolist():
            if j <= i:
                continue
            if count >= max_neighbors:
                break
            dist = float(dists[j])
            if dist < pair_min or dist > pair_max:
                continue
            count += 1
            pj = points[j]
            axis = _normalize(pj - pi)
            center = 0.5 * (pi + pj)
            line_dist, proj = _line_distances(points, center, axis)
            lo = float(min(proj[i], proj[j]) - support_span_pad)
            hi = float(max(proj[i], proj[j]) + support_span_pad)
            support_mask = np.logical_and.reduce((line_dist <= support_radius, proj >= lo, proj <= hi))
            support_idx = np.where(support_mask)[0]
            if support_idx.size < min_support_blobs:
                continue
            pitch_score, best_model_id = _pitch_compatibility(proj[support_idx], diameters[support_idx], priors)
            if pitch_score < float(cfg.get("min_pitch_score", 0.15)):
                continue
            weighted_pts = points[support_idx]
            weighted_w = scores[support_idx]
            try:
                fit_center, fit_axis, _ = _weighted_pca_line(weighted_pts, weighted_w)
            except Exception:
                continue
            start, end = _projected_endpoints(weighted_pts, fit_center, fit_axis, 0.10, 0.90)
            start, end = _orient_shallow_to_deep(start, end, gating.get("head_distance_map_kji"), ctx.get("ras_to_ijk_fn"), np.asarray(ctx.get("center_ras") or [0.0, 0.0, 0.0], dtype=float))
            d0 = _depth_at_ras_mm(np.asarray(start, dtype=float), gating.get("head_distance_map_kji"), ctx.get("ras_to_ijk_fn"))
            d1 = _depth_at_ras_mm(np.asarray(end, dtype=float), gating.get("head_distance_map_kji"), ctx.get("ras_to_ijk_fn"))
            entry_depth = min(v for v in [d0, d1] if v is not None) if any(v is not None for v in [d0, d1]) else None
            skull_score = 1.0 - _clamp01((float(entry_depth) if entry_depth is not None else 20.0) / float(cfg.get("entry_depth_good_mm", 12.0)))
            support_mass = float(np.sum(scores[support_idx]))
            support_only_score = 0.52 * support_mass
            score_no_pitch = support_only_score + 0.20 * skull_score
            score_no_skull = support_only_score + 0.28 * pitch_score
            score = support_only_score + 0.28 * pitch_score + 0.20 * skull_score
            hyp_id = f"H{len(raw_hyps)+1:04d}"
            raw_hyps.append(
                HypothesisRecord(
                    subject_id=subject_id,
                    hypothesis_id=hyp_id,
                    source="blob_pair",
                    start_ras=tuple(float(v) for v in start),
                    end_ras=tuple(float(v) for v in end),
                    direction_ras=tuple(float(v) for v in _normalize(np.asarray(end) - np.asarray(start))),
                    score=float(score),
                    support_mass=float(support_mass),
                    pitch_score=float(pitch_score),
                    skull_score=float(skull_score),
                    support_blob_ids=tuple(f"L{blob_primitives[k].level_index}_B{blob_primitives[k].blob_id}" for k in support_idx.tolist()),
                    seed_blob_ids=(f"L{rec_i.level_index}_B{rec_i.blob_id}", f"L{blob_primitives[j].level_index}_B{blob_primitives[j].blob_id}"),
                    best_model_id=best_model_id,
                    meta={
                        "entry_depth_mm": None if entry_depth is None else float(entry_depth),
                        "seed_distance_mm": float(dist),
                        "support_blob_count": int(support_idx.size),
                        "score_support_only": float(support_only_score),
                        "score_no_pitch": float(score_no_pitch),
                        "score_no_skull": float(score_no_skull),
                    },
                )
            )
            for k in support_idx.tolist():
                support_rows.append(
                    {
                        "subject_id": subject_id,
                        "hypothesis_id": hyp_id,
                        "blob_key": f"L{blob_primitives[k].level_index}_B{blob_primitives[k].blob_id}",
                        "blob_lineage_id": int(blob_primitives[k].lineage_id),
                        "blob_score": f"{float(blob_primitives[k].score):.4f}",
                        "support_weight": f"{float(scores[k]):.4f}",
                        "line_distance_mm": f"{float(line_dist[k]):.4f}",
                        "projection_mm": f"{float(proj[k]):.4f}",
                    }
                )

    accepted: list[HypothesisRecord] = []
    assigned_rows: list[dict[str, Any]] = []
    blob_to_hyps: dict[str, list[tuple[HypothesisRecord, float]]] = {}
    for hyp in raw_hyps:
        hyp_pred = _pred_from_line(hyp.hypothesis_id, np.asarray(hyp.start_ras), np.asarray(hyp.end_ras), support_count=len(hyp.support_blob_ids), confidence=hyp.score)
        for blob_key in hyp.support_blob_ids:
            blob_to_hyps.setdefault(blob_key, []).append((hyp, float(hyp.score)))
    for blob_key, pairs in blob_to_hyps.items():
        pairs = sorted(pairs, key=lambda item: item[1], reverse=True)
        best = pairs[0][0]
        second = pairs[1][0] if len(pairs) > 1 else None
        resolved_by_extra = 0
        if second is not None and len(best.support_blob_ids) > len(second.support_blob_ids):
            resolved_by_extra = 1
        assigned_rows.append(
            {
                "subject_id": subject_id,
                "blob_key": blob_key,
                "assigned_hypothesis_id": best.hypothesis_id,
                "candidate_hypothesis_ids": json.dumps([h.hypothesis_id for h, _ in pairs]),
                "ambiguous": int(len(pairs) > 1),
                "resolved_by_extra_support": int(resolved_by_extra),
            }
        )
    for hyp in sorted(raw_hyps, key=lambda h: float(h.score), reverse=True):
        overlap = 0.0
        hyp_support = set(hyp.support_blob_ids)
        dup = False
        for prev in accepted:
            prev_support = set(prev.support_blob_ids)
            if hyp_support and prev_support:
                overlap = float(len(hyp_support.intersection(prev_support))) / float(max(1, min(len(hyp_support), len(prev_support))))
            angle = _angle_deg(np.asarray(hyp.direction_ras, dtype=float), np.asarray(prev.direction_ras, dtype=float))
            line_dist = _line_distance(np.asarray(hyp.start_ras, dtype=float), np.asarray(hyp.direction_ras, dtype=float), np.asarray(prev.start_ras, dtype=float), np.asarray(prev.direction_ras, dtype=float))
            if overlap >= float(cfg.get("consensus_overlap_drop", 0.55)) or (angle <= float(cfg.get("consensus_angle_deg", 8.0)) and line_dist <= float(cfg.get("consensus_line_distance_mm", 3.0))):
                dup = True
                break
        if dup:
            continue
        accepted.append(hyp)
        if len(accepted) >= int(cfg.get("consensus_max_lines", 30)):
            break
    return raw_hyps, support_rows, accepted, assigned_rows


def _segment_absorption(
    *,
    subject_id: str,
    segment_rows: list[dict[str, Any]],
    accepted_hypotheses: list[HypothesisRecord],
    priors: list[ElectrodePrior],
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[HypothesisRecord]]:
    absorption_rows: list[dict[str, Any]] = []
    rescue_hyps: list[HypothesisRecord] = []
    for row in segment_rows:
        center = np.asarray([float(row["x"]), float(row["y"]), float(row["z"])], dtype=float)
        axis = np.asarray([float(row["axis_x"]), float(row["axis_y"]), float(row["axis_z"])], dtype=float)
        length = float(row["length_mm"])
        diameter = float(row["diameter_mm"])
        best_match = None
        best_score = -1.0
        for hyp in accepted_hypotheses:
            hp0 = np.asarray(hyp.start_ras, dtype=float)
            hd = np.asarray(hyp.direction_ras, dtype=float)
            dist = _line_distance(center, axis, hp0, hd)
            ang = _angle_deg(axis, hd)
            if dist > float(cfg.get("segment_absorb_distance_mm", 3.2)) or ang > float(cfg.get("segment_absorb_angle_deg", 16.0)):
                continue
            score = 1.0 - _clamp01(dist / float(cfg.get("segment_absorb_distance_mm", 3.2))) + 1.0 - _clamp01(ang / float(cfg.get("segment_absorb_angle_deg", 16.0)))
            if score > best_score:
                best_score = float(score)
                best_match = hyp
        absorbed = int(best_match is not None)
        row_out = {
            "subject_id": subject_id,
            "source_kind": row["source_kind"],
            "blob_key": row["blob_key"],
            "absorbed": int(absorbed),
            "absorbed_hypothesis_id": best_match.hypothesis_id if best_match is not None else "",
            "line_distance_mm": "" if best_match is None else f"{_line_distance(center, axis, np.asarray(best_match.start_ras, dtype=float), np.asarray(best_match.direction_ras, dtype=float)):.4f}",
            "angle_deg": "" if best_match is None else f"{_angle_deg(axis, np.asarray(best_match.direction_ras, dtype=float)):.4f}",
            "length_mm": f"{length:.4f}",
            "diameter_mm": f"{diameter:.4f}",
            "threshold_hu": row["threshold_hu"],
        }
        absorption_rows.append(row_out)
        if best_match is None:
            start = center - 0.5 * length * _normalize(axis)
            end = center + 0.5 * length * _normalize(axis)
            pitch_score, best_model_id = _pitch_compatibility(np.asarray([-0.5 * length, 0.5 * length], dtype=float), np.asarray([diameter, diameter], dtype=float), priors)
            rescue_hyps.append(
                HypothesisRecord(
                    subject_id=subject_id,
                    hypothesis_id=f"R{len(rescue_hyps)+1:04d}",
                    source="segment_rescue",
                    start_ras=tuple(float(v) for v in start),
                    end_ras=tuple(float(v) for v in end),
                    direction_ras=tuple(float(v) for v in _normalize(axis)),
                    score=float(0.35 + 0.45 * pitch_score),
                    support_mass=1.0,
                    pitch_score=float(pitch_score),
                    skull_score=0.0,
                    support_blob_ids=(str(row["blob_key"]),),
                    seed_blob_ids=(str(row["blob_key"]),),
                    best_model_id=best_model_id,
                    meta={"source_kind": row["source_kind"]},
                )
            )
    return absorption_rows, rescue_hyps


def _coverage_reason_rows(
    *,
    subject_id: str,
    gt_shanks: list[GroundTruthShank],
    blob_primitives: list[PrimitiveRecord],
    segment_rows: list[dict[str, Any]],
    components: list[PrimitiveRecord],
    raw_hypotheses: list[HypothesisRecord],
    accepted_hypotheses: list[HypothesisRecord],
    rescue_hypotheses: list[HypothesisRecord],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    blob_points = np.asarray([b.point_ras for b in blob_primitives], dtype=float).reshape(-1, 3) if blob_primitives else np.zeros((0, 3), dtype=float)
    segment_centers = []
    segment_axes = []
    for row in segment_rows:
        segment_centers.append([float(row["x"]), float(row["y"]), float(row["z"])])
        segment_axes.append([float(row["axis_x"]), float(row["axis_y"]), float(row["axis_z"])])
    segment_centers_arr = np.asarray(segment_centers, dtype=float).reshape(-1, 3) if segment_centers else np.zeros((0, 3), dtype=float)
    segment_axes_arr = np.asarray(segment_axes, dtype=float).reshape(-1, 3) if segment_axes else np.zeros((0, 3), dtype=float)
    merged_components = [c for c in components if c.kind == "merged_segment"]
    merged_points = np.asarray([c.point_ras for c in merged_components], dtype=float).reshape(-1, 3) if merged_components else np.zeros((0, 3), dtype=float)
    for gt in gt_shanks:
        gt_start = np.asarray(gt.start_ras, dtype=float)
        gt_end = np.asarray(gt.end_ras, dtype=float)
        gt_dir = _unit(np.asarray(gt.direction_ras, dtype=float))
        blob_support = []
        for idx, blob in enumerate(blob_primitives):
            dist = _point_to_segment_distance(blob_points[idx], gt_start, gt_end)
            if dist <= float(cfg.get("gt_blob_support_radius_mm", 3.0)):
                blob_support.append(blob)
        segment_support_idx = []
        for idx in range(segment_centers_arr.shape[0]):
            dist = _point_to_segment_distance(segment_centers_arr[idx], gt_start, gt_end)
            ang = _angle_deg(segment_axes_arr[idx], gt_dir)
            if dist <= float(cfg.get("gt_segment_support_radius_mm", 3.5)) and ang <= float(cfg.get("gt_segment_support_angle_deg", 18.0)):
                segment_support_idx.append(idx)
        junction_near_count = 0
        for idx in range(merged_points.shape[0]):
            dist = _point_to_segment_distance(merged_points[idx], gt_start, gt_end)
            if dist <= float(cfg.get("gt_junction_support_radius_mm", 4.0)):
                junction_near_count += 1

        gt_len = float(np.linalg.norm(gt_end - gt_start))
        shallow_blob_support = 0
        deep_blob_support = 0
        if gt_len > 1e-6:
            for blob in blob_support:
                point = np.asarray(blob.point_ras, dtype=float)
                t = float(np.dot(point - gt_start, gt_dir))
                frac = t / gt_len
                if frac <= 0.25:
                    shallow_blob_support += 1
                if frac >= 0.75:
                    deep_blob_support += 1

        gt_contact_residuals = [
            _point_to_segment_distance(np.asarray(contact, dtype=float), gt_start, gt_end)
            for contact in gt.contacts_ras
        ]
        gt_line_residual_mm = float(np.mean(gt_contact_residuals)) if gt_contact_residuals else 0.0

        def _loose_match_exists(hyps: list[HypothesisRecord]) -> list[tuple[HypothesisRecord, Any]]:
            hits = []
            for hyp in hyps:
                pm = compare_shanks(gt, _pred_from_line(hyp.hypothesis_id, np.asarray(hyp.start_ras), np.asarray(hyp.end_ras), len(hyp.support_blob_ids), hyp.score))
                if pm.end_error_mm <= float(cfg.get("loose_end_mm", 15.0)) and pm.start_error_mm <= float(cfg.get("loose_start_mm", 20.0)) and pm.angle_deg <= float(cfg.get("loose_angle_deg", 10.0)):
                    hits.append((hyp, pm))
            return hits

        def _best_rank(hits: list[tuple[HypothesisRecord, Any]], key: str) -> int:
            if not hits:
                return 0
            def _score_for_rank(hyp: HypothesisRecord) -> float:
                if key == "score":
                    return float(hyp.score)
                return float(hyp.meta.get(key, 0.0))
            ordered = sorted(raw_hypotheses, key=_score_for_rank, reverse=True)
            index = {hyp.hypothesis_id: idx + 1 for idx, hyp in enumerate(ordered)}
            return min(index.get(hyp.hypothesis_id, 10 ** 9) for hyp, _pm in hits)

        raw_hits = _loose_match_exists(raw_hypotheses)
        accepted_hits = _loose_match_exists(accepted_hypotheses)
        rescue_hits = _loose_match_exists(rescue_hypotheses)
        rank_with_priors = _best_rank(raw_hits, "score")
        rank_support_only = _best_rank(raw_hits, "score_support_only")
        rank_no_pitch = _best_rank(raw_hits, "score_no_pitch")
        rank_no_skull = _best_rank(raw_hits, "score_no_skull")
        if not blob_support:
            reason = "no_blob_support"
        elif not raw_hits:
            reason = "blob_support_but_no_plausible_hypothesis"
        elif raw_hits and not accepted_hits and rescue_hits:
            reason = "needs_segment_rescue"
        elif raw_hits and not accepted_hits:
            reason = "hypothesis_exists_but_loses_consensus"
        else:
            best_pm = sorted(accepted_hits or rescue_hits, key=lambda pair: pair[1].score)[0][1]
            if best_pm.end_error_mm > float(cfg.get("strict_end_mm", 4.0)) or best_pm.start_error_mm > float(cfg.get("strict_start_mm", 15.0)):
                reason = "endpoint_skull_prior_failure"
            else:
                reason = "strict_match_present"
        if blob_support and segment_support_idx:
            support_pattern = "beads_on_string"
        elif blob_support:
            support_pattern = "beads_only"
        elif segment_support_idx:
            support_pattern = "string_only"
        else:
            support_pattern = "none"
        rows.append(
            {
                "subject_id": subject_id,
                "gt_shank": gt.shank,
                "blob_support_count": int(len(blob_support)),
                "segment_support_count": int(len(segment_support_idx)),
                "junction_near_count": int(junction_near_count),
                "support_pattern": str(support_pattern),
                "shallow_blob_support_count": int(shallow_blob_support),
                "deep_blob_support_count": int(deep_blob_support),
                "weak_shallow_entry_support": int(shallow_blob_support == 0),
                "gt_line_residual_mm": f"{gt_line_residual_mm:.4f}",
                "slight_bend_hint": int(gt_line_residual_mm >= float(cfg.get("gt_bend_residual_mm", 1.5))),
                "raw_hypothesis_hits": int(len(raw_hits)),
                "accepted_hypothesis_hits": int(len(accepted_hits)),
                "rescue_hypothesis_hits": int(len(rescue_hits)),
                "best_raw_rank_with_priors": int(rank_with_priors),
                "best_raw_rank_support_only": int(rank_support_only),
                "best_raw_rank_no_pitch": int(rank_no_pitch),
                "best_raw_rank_no_skull": int(rank_no_skull),
                "failure_reason": str(reason),
            }
        )
    return rows


def _analyze_subject(row: dict[str, str], args: argparse.Namespace, priors: list[ElectrodePrior]) -> dict[str, Any]:
    subject_id = str(row["subject_id"])
    cfg = {
        "threshold_schedule_hu": [float(v) for v in args.thresholds_hu],
        "component_fit_sample_points": int(args.component_fit_sample_points),
        "decomp_max_points": int(args.decomp_max_points),
        "decomp_fit_sample_points": int(args.decomp_fit_sample_points),
        "decomp_min_points": int(args.decomp_min_points),
        "decomp_min_points_per_line": int(args.decomp_min_points_per_line),
        "decomp_min_fraction_per_line": float(args.decomp_min_fraction_per_line),
        "decomp_max_lines": int(args.decomp_max_lines),
        "decomp_line_radius_mm": float(args.decomp_line_radius_mm),
        "decomp_max_residual_mm": float(args.decomp_max_residual_mm),
        "max_blob_neighbors": int(args.max_blob_neighbors),
        "hypothesis_min_pair_distance_mm": float(args.hypothesis_min_pair_distance_mm),
        "hypothesis_max_pair_distance_mm": float(args.hypothesis_max_pair_distance_mm),
        "hypothesis_support_radius_mm": float(args.hypothesis_support_radius_mm),
        "hypothesis_support_span_pad_mm": float(args.hypothesis_support_span_pad_mm),
        "hypothesis_min_support_blobs": int(args.hypothesis_min_support_blobs),
        "consensus_overlap_drop": float(args.consensus_overlap_drop),
        "consensus_angle_deg": float(args.consensus_angle_deg),
        "consensus_line_distance_mm": float(args.consensus_line_distance_mm),
        "consensus_max_lines": int(args.consensus_max_lines),
        "segment_absorb_distance_mm": float(args.segment_absorb_distance_mm),
        "segment_absorb_angle_deg": float(args.segment_absorb_angle_deg),
        "gt_blob_support_radius_mm": float(args.gt_blob_support_radius_mm),
        "loose_end_mm": float(args.loose_end_mm),
        "loose_start_mm": float(args.loose_start_mm),
        "loose_angle_deg": float(args.loose_angle_deg),
        "strict_end_mm": float(args.strict_end_mm),
        "strict_start_mm": float(args.strict_start_mm),
        "entry_depth_good_mm": float(args.entry_depth_good_mm),
        "min_pitch_score": float(args.min_pitch_score),
    }
    config = {
        "threshold": float(max(args.thresholds_hu)),
        "use_head_mask": True,
        "build_head_mask": True,
        "head_mask_threshold_hu": float(args.head_mask_threshold_hu),
        "head_mask_method": str(args.head_mask_method),
        "head_gate_erode_vox": int(args.head_gate_erode_vox),
        "head_gate_dilate_vox": int(args.head_gate_dilate_vox),
        "head_gate_margin_mm": float(args.head_gate_margin_mm),
        "min_metal_depth_mm": float(args.min_metal_depth_mm),
        "max_metal_depth_mm": float(args.max_metal_depth_mm),
    }
    ctx, _img = build_detection_context(row["ct_path"], run_id=f"blob_segment_consensus_{subject_id}", config=config, extras={})
    gating = build_preview_masks(
        ctx["arr_kji"],
        ctx["spacing_xyz"],
        threshold=float(max(args.thresholds_hu)),
        use_head_mask=True,
        build_head_mask=True,
        head_mask_threshold_hu=float(args.head_mask_threshold_hu),
        head_mask_method=str(args.head_mask_method),
        head_gate_erode_vox=int(args.head_gate_erode_vox),
        head_gate_dilate_vox=int(args.head_gate_dilate_vox),
        head_gate_margin_mm=float(args.head_gate_margin_mm),
        min_metal_depth_mm=float(args.min_metal_depth_mm),
        max_metal_depth_mm=float(args.max_metal_depth_mm),
        include_debug_masks=False,
    )
    gating_mask = np.asarray(gating.get("gating_mask_kji"), dtype=bool)
    if gating_mask.size == 0 or not np.any(gating_mask):
        gating_mask = np.ones_like(ctx["arr_kji"], dtype=bool)
    levels = extract_threshold_levels(
        arr_kji=np.asarray(ctx["arr_kji"], dtype=float),
        gating_mask_kji=gating_mask,
        depth_map_kji=gating.get("head_distance_map_kji"),
        thresholds_hu=list(args.thresholds_hu),
        ijk_kji_to_ras_fn=ctx["ijk_kji_to_ras_fn"],
    )
    lineages = build_lineages(levels)
    _summaries = summarize_lineages(lineages, total_levels=len(levels))
    gt_shanks = load_ground_truth_shanks(row["labels_path"], row.get("shanks_path"))
    components, decomposition_rows, segment_rows = _make_component_primitives(
        subject_id=subject_id,
        levels=levels,
        lineages=lineages,
        gt_shanks=gt_shanks,
        ctx=ctx,
        gating=gating,
        cfg=cfg,
    )
    blob_primitives = _dedup_blob_primitives(components, radius_mm=float(args.blob_dedup_radius_mm))
    raw_hyps, support_rows, accepted_hyps, assigned_rows = _build_blob_hypotheses(subject_id=subject_id, blob_primitives=blob_primitives, gt_shanks=gt_shanks, priors=priors, gating=gating, ctx=ctx, cfg=cfg)
    absorption_rows, rescue_hyps = _segment_absorption(subject_id=subject_id, segment_rows=segment_rows, accepted_hypotheses=accepted_hyps, priors=priors, cfg=cfg)
    coverage_rows = _coverage_reason_rows(
        subject_id=subject_id,
        gt_shanks=gt_shanks,
        blob_primitives=blob_primitives,
        segment_rows=segment_rows,
        components=components,
        raw_hypotheses=raw_hyps,
        accepted_hypotheses=accepted_hyps,
        rescue_hypotheses=rescue_hyps,
        cfg=cfg,
    )
    merged_blob_count = sum(1 for c in components if c.kind == "merged_segment")
    merged_fully_explained = sum(1 for row in decomposition_rows if float(row["explained_fraction"]) >= 0.80)
    absorbed_count = sum(int(r["absorbed"]) for r in absorption_rows)
    rescue_hits = 0
    for gt in gt_shanks:
        for hyp in rescue_hyps:
            pm = compare_shanks(gt, _pred_from_line(hyp.hypothesis_id, np.asarray(hyp.start_ras), np.asarray(hyp.end_ras), len(hyp.support_blob_ids), hyp.score))
            if pm.end_error_mm <= float(args.loose_end_mm) and pm.start_error_mm <= float(args.loose_start_mm) and pm.angle_deg <= float(args.loose_angle_deg):
                rescue_hits += 1
                break
    subject_summary = {
        "subject_id": subject_id,
        "gt_shanks": int(len(gt_shanks)),
        "component_total": int(len(components)),
        "blob_total": int(sum(1 for c in components if c.kind == "blob")),
        "segment_total": int(sum(1 for c in components if c.kind == "segment")),
        "merged_segment_total": int(merged_blob_count),
        "junk_total": int(sum(1 for c in components if c.kind == "junk")),
        "dedup_blob_total": int(len(blob_primitives)),
        "raw_hypothesis_total": int(len(raw_hyps)),
        "accepted_hypothesis_total": int(len(accepted_hyps)),
        "rescue_hypothesis_total": int(len(rescue_hyps)),
        "blob_support_gt_count": int(sum(1 for r in coverage_rows if r["blob_support_count"] > 0)),
        "beads_only_gt_count": int(sum(1 for r in coverage_rows if r["support_pattern"] == "beads_only")),
        "string_only_gt_count": int(sum(1 for r in coverage_rows if r["support_pattern"] == "string_only")),
        "beads_on_string_gt_count": int(sum(1 for r in coverage_rows if r["support_pattern"] == "beads_on_string")),
        "junction_near_gt_count": int(sum(1 for r in coverage_rows if int(r["junction_near_count"]) > 0)),
        "weak_shallow_entry_gt_count": int(sum(1 for r in coverage_rows if int(r["weak_shallow_entry_support"]) > 0)),
        "accepted_blob_hypothesis_gt_count": int(sum(1 for r in coverage_rows if r["accepted_hypothesis_hits"] > 0)),
        "rescue_gt_count": int(sum(1 for r in coverage_rows if r["rescue_hypothesis_hits"] > 0)),
        "raw_rank_improved_vs_support_only": int(sum(1 for r in coverage_rows if int(r["best_raw_rank_with_priors"]) > 0 and int(r["best_raw_rank_with_priors"]) < int(r["best_raw_rank_support_only"] or 0))),
        "raw_rank_improved_vs_no_pitch": int(sum(1 for r in coverage_rows if int(r["best_raw_rank_with_priors"]) > 0 and int(r["best_raw_rank_with_priors"]) < int(r["best_raw_rank_no_pitch"] or 0))),
        "raw_rank_improved_vs_no_skull": int(sum(1 for r in coverage_rows if int(r["best_raw_rank_with_priors"]) > 0 and int(r["best_raw_rank_with_priors"]) < int(r["best_raw_rank_no_skull"] or 0))),
        "segment_absorbed_fraction": 0.0 if not absorption_rows else float(absorbed_count) / float(len(absorption_rows)),
        "merged_fully_explained_fraction": 0.0 if not decomposition_rows else float(merged_fully_explained) / float(len(decomposition_rows)),
        "ambiguous_blob_count": int(sum(int(r["ambiguous"]) for r in assigned_rows)),
        "ambiguity_resolved_by_extra_support": int(sum(int(r["resolved_by_extra_support"]) for r in assigned_rows)),
    }
    return {
        "subject_summary": subject_summary,
        "primitive_rows": [_primitive_score_row(c) for c in components],
        "decomposition_rows": decomposition_rows,
        "hypothesis_rows": [_hypothesis_row(h) for h in raw_hyps],
        "accepted_hypothesis_rows": [_hypothesis_row(h) for h in accepted_hyps],
        "support_rows": support_rows,
        "assignment_rows": assigned_rows,
        "segment_absorption_rows": absorption_rows,
        "rescue_hypothesis_rows": [_hypothesis_row(h) for h in rescue_hyps],
        "coverage_rows": coverage_rows,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze blob/segment consensus evidence for SEEG CTs")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--subjects", default="T1,T2,T3,T25")
    p.add_argument("--thresholds-hu", default="2600,2200,1800,1500,1200")
    p.add_argument("--head-mask-method", choices=["outside_air", "not_air_lcc"], default="outside_air")
    p.add_argument("--head-mask-threshold-hu", type=float, default=-500.0)
    p.add_argument("--head-gate-erode-vox", type=int, default=1)
    p.add_argument("--head-gate-dilate-vox", type=int, default=1)
    p.add_argument("--head-gate-margin-mm", type=float, default=0.0)
    p.add_argument("--min-metal-depth-mm", type=float, default=5.0)
    p.add_argument("--max-metal-depth-mm", type=float, default=220.0)
    p.add_argument("--component-fit-sample-points", type=int, default=3000)
    p.add_argument("--blob-dedup-radius-mm", type=float, default=2.5)
    p.add_argument("--decomp-max-points", type=int, default=5000)
    p.add_argument("--decomp-fit-sample-points", type=int, default=2500)
    p.add_argument("--decomp-min-points", type=int, default=20)
    p.add_argument("--decomp-min-points-per-line", type=int, default=12)
    p.add_argument("--decomp-min-fraction-per-line", type=float, default=0.15)
    p.add_argument("--decomp-max-lines", type=int, default=4)
    p.add_argument("--decomp-line-radius-mm", type=float, default=2.0)
    p.add_argument("--decomp-max-residual-mm", type=float, default=1.8)
    p.add_argument("--max-blob-neighbors", type=int, default=12)
    p.add_argument("--hypothesis-min-pair-distance-mm", type=float, default=8.0)
    p.add_argument("--hypothesis-max-pair-distance-mm", type=float, default=90.0)
    p.add_argument("--hypothesis-support-radius-mm", type=float, default=2.8)
    p.add_argument("--hypothesis-support-span-pad-mm", type=float, default=5.0)
    p.add_argument("--hypothesis-min-support-blobs", type=int, default=3)
    p.add_argument("--consensus-overlap-drop", type=float, default=0.55)
    p.add_argument("--consensus-angle-deg", type=float, default=8.0)
    p.add_argument("--consensus-line-distance-mm", type=float, default=3.0)
    p.add_argument("--consensus-max-lines", type=int, default=30)
    p.add_argument("--segment-absorb-distance-mm", type=float, default=3.2)
    p.add_argument("--segment-absorb-angle-deg", type=float, default=16.0)
    p.add_argument("--gt-blob-support-radius-mm", type=float, default=3.0)
    p.add_argument("--min-pitch-score", type=float, default=0.15)
    p.add_argument("--entry-depth-good-mm", type=float, default=12.0)
    p.add_argument("--loose-end-mm", type=float, default=15.0)
    p.add_argument("--loose-start-mm", type=float, default=20.0)
    p.add_argument("--loose-angle-deg", type=float, default=10.0)
    p.add_argument("--strict-end-mm", type=float, default=4.0)
    p.add_argument("--strict-start-mm", type=float, default=15.0)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    args.thresholds_hu = [float(v.strip()) for v in str(args.thresholds_hu).split(",") if v.strip()]
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    subject_filter = {s.strip() for s in str(args.subjects).split(",") if s.strip()} or None
    subject_rows = iter_subject_rows(dataset_root, subject_filter)
    if not subject_rows:
        raise SystemExit("No subjects matched")
    priors = _load_family_priors()

    all_subject_rows: list[dict[str, Any]] = []
    all_primitives: list[dict[str, Any]] = []
    all_decomp: list[dict[str, Any]] = []
    all_hyps: list[dict[str, Any]] = []
    all_accepted: list[dict[str, Any]] = []
    all_support_rows: list[dict[str, Any]] = []
    all_assignment_rows: list[dict[str, Any]] = []
    all_absorption: list[dict[str, Any]] = []
    all_rescue: list[dict[str, Any]] = []
    all_coverage: list[dict[str, Any]] = []

    for row in subject_rows:
        subject_id = str(row["subject_id"])
        data = _analyze_subject(row, args, priors)
        subject_dir = out_dir / subject_id
        subject_dir.mkdir(parents=True, exist_ok=True)

        summary = dict(data["subject_summary"])
        all_subject_rows.append(summary)
        all_primitives.extend(list(data["primitive_rows"]))
        all_decomp.extend(list(data["decomposition_rows"]))
        all_hyps.extend(list(data["hypothesis_rows"]))
        all_accepted.extend(list(data["accepted_hypothesis_rows"]))
        all_support_rows.extend(list(data["support_rows"]))
        all_assignment_rows.extend(list(data["assignment_rows"]))
        all_absorption.extend(list(data["segment_absorption_rows"]))
        all_rescue.extend(list(data["rescue_hypothesis_rows"]))
        all_coverage.extend(list(data["coverage_rows"]))

        with open(subject_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True, default=_json_default)
        if data["primitive_rows"]:
            _write_tsv(subject_dir / "primitive_table.tsv", list(data["primitive_rows"]), list(data["primitive_rows"][0].keys()))
        if data["decomposition_rows"]:
            _write_tsv(subject_dir / "merged_segment_decomposition.tsv", list(data["decomposition_rows"]), list(data["decomposition_rows"][0].keys()))
        if data["hypothesis_rows"]:
            _write_tsv(subject_dir / "blob_consensus_hypotheses.tsv", list(data["hypothesis_rows"]), list(data["hypothesis_rows"][0].keys()))
        if data["support_rows"]:
            _write_tsv(subject_dir / "blob_hypothesis_support.tsv", list(data["support_rows"]), list(data["support_rows"][0].keys()))
        if data["assignment_rows"]:
            _write_tsv(subject_dir / "blob_hypothesis_assignments.tsv", list(data["assignment_rows"]), list(data["assignment_rows"][0].keys()))
        if data["segment_absorption_rows"]:
            _write_tsv(subject_dir / "segment_absorption.tsv", list(data["segment_absorption_rows"]), list(data["segment_absorption_rows"][0].keys()))
        if data["rescue_hypothesis_rows"]:
            _write_tsv(subject_dir / "segment_rescue_hypotheses.tsv", list(data["rescue_hypothesis_rows"]), list(data["rescue_hypothesis_rows"][0].keys()))
        if data["coverage_rows"]:
            _write_tsv(subject_dir / "gt_coverage.tsv", list(data["coverage_rows"]), list(data["coverage_rows"][0].keys()))
        print(f"[consensus] {subject_id}: components={summary['component_total']} blobs={summary['blob_total']} segments={summary['segment_total']} merged={summary['merged_segment_total']} hyps={summary['accepted_hypothesis_total']} rescue={summary['rescue_hypothesis_total']}")

    if all_subject_rows:
        _write_tsv(out_dir / "subject_summary.tsv", all_subject_rows, list(all_subject_rows[0].keys()))
    if all_primitives:
        _write_tsv(out_dir / "primitive_table.tsv", all_primitives, list(all_primitives[0].keys()))
    if all_decomp:
        _write_tsv(out_dir / "merged_segment_decomposition.tsv", all_decomp, list(all_decomp[0].keys()))
    if all_hyps:
        _write_tsv(out_dir / "blob_consensus_hypotheses.tsv", all_hyps, list(all_hyps[0].keys()))
    if all_accepted:
        _write_tsv(out_dir / "blob_consensus_hypotheses_accepted.tsv", all_accepted, list(all_accepted[0].keys()))
    if all_support_rows:
        _write_tsv(out_dir / "blob_hypothesis_support.tsv", all_support_rows, list(all_support_rows[0].keys()))
    if all_assignment_rows:
        _write_tsv(out_dir / "blob_hypothesis_assignments.tsv", all_assignment_rows, list(all_assignment_rows[0].keys()))
    if all_absorption:
        _write_tsv(out_dir / "segment_absorption.tsv", all_absorption, list(all_absorption[0].keys()))
    if all_rescue:
        _write_tsv(out_dir / "segment_rescue_hypotheses.tsv", all_rescue, list(all_rescue[0].keys()))
    if all_coverage:
        _write_tsv(out_dir / "gt_coverage.tsv", all_coverage, list(all_coverage[0].keys()))

    reason_counts: dict[str, int] = {}
    for row in all_coverage:
        reason = str(row["failure_reason"])
        reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
    cohort_summary = {
        "subject_count": len(all_subject_rows),
        "subjects": [str(r["subject_id"]) for r in all_subject_rows],
        "gt_total": int(sum(int(r["gt_shanks"]) for r in all_subject_rows)),
        "blob_support_gt_count": int(sum(int(r["blob_support_gt_count"]) for r in all_subject_rows)),
        "beads_only_gt_count": int(sum(int(r["beads_only_gt_count"]) for r in all_subject_rows)),
        "string_only_gt_count": int(sum(int(r["string_only_gt_count"]) for r in all_subject_rows)),
        "beads_on_string_gt_count": int(sum(int(r["beads_on_string_gt_count"]) for r in all_subject_rows)),
        "junction_near_gt_count": int(sum(int(r["junction_near_gt_count"]) for r in all_subject_rows)),
        "weak_shallow_entry_gt_count": int(sum(int(r["weak_shallow_entry_gt_count"]) for r in all_subject_rows)),
        "accepted_blob_hypothesis_gt_count": int(sum(int(r["accepted_blob_hypothesis_gt_count"]) for r in all_subject_rows)),
        "rescue_gt_count": int(sum(int(r["rescue_gt_count"]) for r in all_subject_rows)),
        "raw_rank_improved_vs_support_only": int(sum(int(r["raw_rank_improved_vs_support_only"]) for r in all_subject_rows)),
        "raw_rank_improved_vs_no_pitch": int(sum(int(r["raw_rank_improved_vs_no_pitch"]) for r in all_subject_rows)),
        "raw_rank_improved_vs_no_skull": int(sum(int(r["raw_rank_improved_vs_no_skull"]) for r in all_subject_rows)),
        "reason_counts": reason_counts,
        "merged_fully_explained_fraction_mean": float(np.mean([float(r["merged_fully_explained_fraction"]) for r in all_subject_rows])) if all_subject_rows else 0.0,
        "segment_absorbed_fraction_mean": float(np.mean([float(r["segment_absorbed_fraction"]) for r in all_subject_rows])) if all_subject_rows else 0.0,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(cohort_summary, f, indent=2, sort_keys=True, default=_json_default)
    print(f"[consensus] wrote {out_dir}")


if __name__ == "__main__":
    main()
