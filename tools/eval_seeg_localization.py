#!/usr/bin/env python3
"""Evaluate shank-detection pipelines against the SEEG localization dataset.

This tool stays in the pure-Python engine layer. It loads registered CT volumes,
runs one detection pipeline per subject, derives ground-truth shank centerlines from
contact label tables, and scores predicted shanks with one-to-one assignment.
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
try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIB_DIR = os.path.join(REPO_ROOT, "CommonLib")
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

from rosa_core.electrode_models import load_electrode_library, model_map  # noqa: E402
from shank_core.io import (  # noqa: E402
    image_ijk_ras_matrices,
    kji_to_ras_points_matrix,
    ras_to_ijk_float_matrix,
    read_volume,
)
from shank_engine import PipelineRegistry, VolumeRef, register_builtin_pipelines  # noqa: E402


@dataclass(frozen=True)
class GroundTruthShank:
    subject_id: str
    shank: str
    contact_count: int
    start_ras: tuple[float, float, float]
    end_ras: tuple[float, float, float]
    direction_ras: tuple[float, float, float]
    span_mm: float
    contacts_ras: tuple[tuple[float, float, float], ...]
    contact_indices: tuple[int, ...]


@dataclass(frozen=True)
class PredictedShank:
    name: str
    start_ras: tuple[float, float, float]
    end_ras: tuple[float, float, float]
    direction_ras: tuple[float, float, float]
    length_mm: float
    support_count: int
    confidence: float


@dataclass(frozen=True)
class PairMetric:
    start_error_mm: float
    end_error_mm: float
    angle_deg: float
    midpoint_dist_mm: float
    mean_contact_dist_mm: float
    max_contact_dist_mm: float
    score: float


def _read_tsv_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _write_tsv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _as_float3(row: dict[str, str], keys: tuple[str, str, str] = ("x", "y", "z")) -> np.ndarray:
    return np.array([float(row[keys[0]]), float(row[keys[1]]), float(row[keys[2]])], dtype=float)


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return vec / norm


def _fit_line_to_contacts(points_ras: np.ndarray, contact_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    center = points_ras.mean(axis=0)
    centered = points_ras - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = _unit(vh[0])
    proj = centered @ direction
    if len(points_ras) > 1:
        corr = np.corrcoef(proj, contact_indices.astype(float))[0, 1]
        if np.isfinite(corr) and corr < 0.0:
            direction = -direction
            proj = -proj
    start = center + direction * float(np.min(proj))
    end = center + direction * float(np.max(proj))
    span = float(np.max(proj) - np.min(proj))
    return start, end, span


def load_ground_truth_shanks(labels_path: str | Path, shanks_path: str | Path | None = None) -> list[GroundTruthShank]:
    label_rows = _read_tsv_rows(labels_path)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in label_rows:
        grouped.setdefault(str(row["shank"]), []).append(row)

    cached_rows = _read_tsv_rows(shanks_path) if shanks_path and Path(shanks_path).exists() else []
    cached_by_shank = {str(row["shank"]): row for row in cached_rows}

    shanks: list[GroundTruthShank] = []
    for shank_name, shank_rows in sorted(grouped.items()):
        shank_rows = sorted(shank_rows, key=lambda r: int(r["contact_index"]))
        pts = np.asarray([_as_float3(r) for r in shank_rows], dtype=float)
        idx = np.asarray([int(r["contact_index"]) for r in shank_rows], dtype=int)
        cached = cached_by_shank.get(shank_name)
        if cached:
            start = np.asarray([float(cached["start_x"]), float(cached["start_y"]), float(cached["start_z"])], dtype=float)
            end = np.asarray([float(cached["end_x"]), float(cached["end_y"]), float(cached["end_z"])], dtype=float)
            direction = np.asarray([float(cached["dir_x"]), float(cached["dir_y"]), float(cached["dir_z"])], dtype=float)
            span = float(cached["span_mm"])
        else:
            start, end, span = _fit_line_to_contacts(pts, idx)
            direction = _unit(end - start)
        shanks.append(
            GroundTruthShank(
                subject_id=str(shank_rows[0]["subject_id"]),
                shank=str(shank_name),
                contact_count=len(shank_rows),
                start_ras=tuple(float(v) for v in start),
                end_ras=tuple(float(v) for v in end),
                direction_ras=tuple(float(v) for v in _unit(np.asarray(direction, dtype=float))),
                span_mm=float(span),
                contacts_ras=tuple(tuple(float(v) for v in p) for p in pts),
                contact_indices=tuple(int(v) for v in idx),
            )
        )
    return shanks


def load_guided_ground_truth_shanks(labels_path: str | Path, guided_shanks_path: str | Path) -> list[GroundTruthShank]:
    label_rows = _read_tsv_rows(labels_path)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in label_rows:
        grouped.setdefault(str(row["shank"]), []).append(row)

    guided_rows = _read_tsv_rows(guided_shanks_path)
    shanks: list[GroundTruthShank] = []
    for row in guided_rows:
        shank_name = str(row["shank"])
        shank_rows = sorted(grouped.get(shank_name, []), key=lambda r: int(r["contact_index"])) if shank_name in grouped else []
        contact_count = int(row.get("contact_count") or len(shank_rows) or 0)
        if contact_count <= 0:
            continue
        start = np.asarray([float(row["start_x"]), float(row["start_y"]), float(row["start_z"])], dtype=float)
        end = np.asarray([float(row["end_x"]), float(row["end_y"]), float(row["end_z"])], dtype=float)
        direction = np.asarray([float(row.get("dir_x") or 0.0), float(row.get("dir_y") or 0.0), float(row.get("dir_z") or 0.0)], dtype=float)
        if float(np.linalg.norm(direction)) <= 1e-8:
            direction = _unit(end - start)
        else:
            direction = _unit(direction)
        span = float(row.get("span_mm") or float(np.linalg.norm(end - start)))
        if shank_rows:
            contact_indices = tuple(int(r["contact_index"]) for r in shank_rows)
            ts = np.linspace(0.0, 1.0, contact_count)
        else:
            contact_indices = tuple(range(1, contact_count + 1))
            ts = np.linspace(0.0, 1.0, contact_count)
        contacts = tuple(tuple(float(v) for v in (start + t * (end - start))) for t in ts)
        subject_id = str(row.get("subject_id") or (shank_rows[0]["subject_id"] if shank_rows else ""))
        shanks.append(
            GroundTruthShank(
                subject_id=subject_id,
                shank=shank_name,
                contact_count=contact_count,
                start_ras=tuple(float(v) for v in start),
                end_ras=tuple(float(v) for v in end),
                direction_ras=tuple(float(v) for v in direction),
                span_mm=float(span),
                contacts_ras=contacts,
                contact_indices=contact_indices,
            )
        )
    return sorted(shanks, key=lambda item: item.shank)


def _guided_gt_path_for_row(row: dict[str, str], guided_gt_root: str | Path | None) -> Path | None:
    if guided_gt_root in (None, ""):
        return None
    root = Path(guided_gt_root).expanduser().resolve()
    subject_id = str(row["subject_id"])
    path = root / subject_id / f"{subject_id}_shanks_guided.tsv"
    return path if path.exists() else None


def load_reference_ground_truth_shanks(row: dict[str, str], guided_gt_root: str | Path | None = None) -> tuple[list[GroundTruthShank], str]:
    guided_path = _guided_gt_path_for_row(row, guided_gt_root)
    if guided_path is not None:
        return load_guided_ground_truth_shanks(row["labels_path"], guided_path), str(guided_path)
    return load_ground_truth_shanks(row["labels_path"], row.get("shanks_path")), str(row.get("shanks_path") or row["labels_path"])


def _trajectory_direction(start_ras: list[float] | tuple[float, float, float], end_ras: list[float] | tuple[float, float, float]) -> np.ndarray:
    return _unit(np.asarray(end_ras, dtype=float) - np.asarray(start_ras, dtype=float))


def predicted_shanks_from_result(result: dict[str, Any]) -> list[PredictedShank]:
    out: list[PredictedShank] = []
    for idx, traj in enumerate(list(result.get("trajectories") or []), start=1):
        start = tuple(float(v) for v in list(traj.get("start_ras") or [0.0, 0.0, 0.0])[:3])
        end = tuple(float(v) for v in list(traj.get("end_ras") or [0.0, 0.0, 0.0])[:3])
        out.append(
            PredictedShank(
                name=str(traj.get("name") or f"P{idx:02d}"),
                start_ras=start,
                end_ras=end,
                direction_ras=tuple(float(v) for v in _trajectory_direction(start, end)),
                length_mm=float(traj.get("length_mm", 0.0)),
                support_count=int(traj.get("support_count", 0)),
                confidence=float(traj.get("confidence", 0.0)),
            )
        )
    return out


def _point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    seg = end - start
    denom = float(np.dot(seg, seg))
    if denom <= 1e-8:
        return float(np.linalg.norm(point - start))
    t = float(np.dot(point - start, seg) / denom)
    t = min(1.0, max(0.0, t))
    closest = start + t * seg
    return float(np.linalg.norm(point - closest))


def compare_shanks(gt: GroundTruthShank, pred: PredictedShank) -> PairMetric:
    gt_contacts = np.asarray(gt.contacts_ras, dtype=float)
    gt_start = np.asarray(gt.start_ras, dtype=float)
    gt_end = np.asarray(gt.end_ras, dtype=float)
    pred_start = np.asarray(pred.start_ras, dtype=float)
    pred_end = np.asarray(pred.end_ras, dtype=float)

    dot_signed = float(np.dot(np.asarray(gt.direction_ras, dtype=float), np.asarray(pred.direction_ras, dtype=float)))
    if dot_signed < 0.0:
        pred_start, pred_end = pred_end, pred_start

    dists = np.asarray([_point_to_segment_distance(p, pred_start, pred_end) for p in gt_contacts], dtype=float)
    gt_mid = 0.5 * (gt_start + gt_end)
    pred_mid = 0.5 * (pred_start + pred_end)
    midpoint_dist = float(np.linalg.norm(gt_mid - pred_mid))
    dot = float(abs(np.dot(np.asarray(gt.direction_ras, dtype=float), _unit(pred_end - pred_start))))
    dot = min(1.0, max(-1.0, dot))
    angle_deg = float(np.degrees(np.arccos(dot)))
    start_error_mm = float(np.linalg.norm(gt_start - pred_start))
    end_error_mm = float(np.linalg.norm(gt_end - pred_end))
    mean_dist = float(np.mean(dists)) if dists.size else float("inf")
    max_dist = float(np.max(dists)) if dists.size else float("inf")
    score = end_error_mm + 0.35 * start_error_mm + 0.1 * angle_deg + 0.05 * midpoint_dist
    return PairMetric(
        start_error_mm=start_error_mm,
        end_error_mm=end_error_mm,
        angle_deg=angle_deg,
        midpoint_dist_mm=midpoint_dist,
        mean_contact_dist_mm=mean_dist,
        max_contact_dist_mm=max_dist,
        score=score,
    )


def _assign_pairs(cost: np.ndarray) -> tuple[list[int], list[int]]:
    """Return one-to-one assignment using SciPy when available, else greedy fallback."""
    if linear_sum_assignment is not None:
        rows, cols = linear_sum_assignment(cost)
        return rows.tolist(), cols.tolist()

    remaining = [((int(i), int(j)), float(cost[i, j])) for i in range(cost.shape[0]) for j in range(cost.shape[1])]
    remaining.sort(key=lambda item: item[1])
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    row_ind: list[int] = []
    col_ind: list[int] = []
    for (i, j), _score in remaining:
        if i in used_rows or j in used_cols:
            continue
        used_rows.add(i)
        used_cols.add(j)
        row_ind.append(i)
        col_ind.append(j)
    return row_ind, col_ind


def match_shanks(
    gt_shanks: list[GroundTruthShank],
    pred_shanks: list[PredictedShank],
    *,
    match_distance_mm: float,
    match_angle_deg: float,
    match_start_mm: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not gt_shanks and not pred_shanks:
        return [], {"matched": 0, "false_negative": 0, "false_positive": 0}

    if not gt_shanks:
        rows = [
            {
                "subject_id": "",
                "gt_shank": "",
                "pred_shank": p.name,
                "matched": 0,
                "reason": "no_ground_truth",
                "mean_contact_dist_mm": "",
                "max_contact_dist_mm": "",
                "angle_deg": "",
                "midpoint_dist_mm": "",
            }
            for p in pred_shanks
        ]
        return rows, {"matched": 0, "false_negative": 0, "false_positive": len(pred_shanks)}

    if not pred_shanks:
        rows = [
            {
                "subject_id": gt.subject_id,
                "gt_shank": gt.shank,
                "pred_shank": "",
                "matched": 0,
                "reason": "missed",
                "mean_contact_dist_mm": "",
                "max_contact_dist_mm": "",
                "angle_deg": "",
                "midpoint_dist_mm": "",
            }
            for gt in gt_shanks
        ]
        return rows, {"matched": 0, "false_negative": len(gt_shanks), "false_positive": 0}

    cost = np.zeros((len(gt_shanks), len(pred_shanks)), dtype=float)
    metrics: dict[tuple[int, int], PairMetric] = {}
    for i, gt in enumerate(gt_shanks):
        for j, pred in enumerate(pred_shanks):
            pm = compare_shanks(gt, pred)
            metrics[(i, j)] = pm
            cost[i, j] = pm.score

    row_ind, col_ind = _assign_pairs(cost)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    rows: list[dict[str, Any]] = []
    matched_count = 0

    for i, j in zip(list(row_ind), list(col_ind)):
        gt = gt_shanks[i]
        pred = pred_shanks[j]
        pm = metrics[(i, j)]
        is_match = pm.end_error_mm <= match_distance_mm and pm.start_error_mm <= match_start_mm and pm.angle_deg <= match_angle_deg
        rows.append(
            {
                "subject_id": gt.subject_id,
                "gt_shank": gt.shank,
                "pred_shank": pred.name,
                "matched": int(is_match),
                "reason": "matched" if is_match else "assignment_threshold_fail",
                "start_error_mm": f"{pm.start_error_mm:.4f}",
                "end_error_mm": f"{pm.end_error_mm:.4f}",
                "angle_deg": f"{pm.angle_deg:.4f}",
                "midpoint_dist_mm": f"{pm.midpoint_dist_mm:.4f}",
                "mean_contact_dist_mm": f"{pm.mean_contact_dist_mm:.4f}",
                "max_contact_dist_mm": f"{pm.max_contact_dist_mm:.4f}",
                "pred_support_count": pred.support_count,
                "pred_confidence": f"{pred.confidence:.4f}",
            }
        )
        if is_match:
            matched_gt.add(i)
            matched_pred.add(j)
            matched_count += 1

    for i, gt in enumerate(gt_shanks):
        if i not in matched_gt:
            rows.append(
                {
                    "subject_id": gt.subject_id,
                    "gt_shank": gt.shank,
                    "pred_shank": "",
                    "matched": 0,
                    "reason": "missed",
                    "start_error_mm": "",
                    "end_error_mm": "",
                    "angle_deg": "",
                    "midpoint_dist_mm": "",
                    "mean_contact_dist_mm": "",
                    "max_contact_dist_mm": "",
                    "pred_support_count": "",
                    "pred_confidence": "",
                }
            )

    for j, pred in enumerate(pred_shanks):
        if j not in matched_pred:
            rows.append(
                {
                    "subject_id": gt_shanks[0].subject_id,
                    "gt_shank": "",
                    "pred_shank": pred.name,
                    "matched": 0,
                    "reason": "false_positive",
                    "start_error_mm": "",
                    "end_error_mm": "",
                    "angle_deg": "",
                    "midpoint_dist_mm": "",
                    "mean_contact_dist_mm": "",
                    "max_contact_dist_mm": "",
                    "pred_support_count": pred.support_count,
                    "pred_confidence": f"{pred.confidence:.4f}",
                }
            )

    summary = {
        "matched": matched_count,
        "false_negative": len(gt_shanks) - matched_count,
        "false_positive": len(pred_shanks) - matched_count,
    }
    return rows, summary


def build_detection_context(ct_path: str, run_id: str, config: dict[str, Any], extras: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    img, arr_kji, spacing_xyz = read_volume(ct_path)
    ijk_to_ras, ras_to_ijk = image_ijk_ras_matrices(img)
    size_xyz = img.GetSize()
    center_ijk = [0.5 * (float(size_xyz[0]) - 1.0), 0.5 * (float(size_xyz[1]) - 1.0), 0.5 * (float(size_xyz[2]) - 1.0)]
    center_ras = kji_to_ras_points_matrix([[center_ijk[2], center_ijk[1], center_ijk[0]]], ijk_to_ras)[0].tolist()
    ctx = {
        "run_id": run_id,
        "ct": VolumeRef(volume_id=Path(ct_path).stem, path=str(ct_path), spacing_xyz=tuple(float(v) for v in spacing_xyz)),
        "arr_kji": arr_kji,
        "spacing_xyz": tuple(float(v) for v in spacing_xyz),
        "ijk_kji_to_ras_fn": lambda ijk_kji: kji_to_ras_points_matrix(ijk_kji, ijk_to_ras),
        "ras_to_ijk_fn": lambda ras_xyz: ras_to_ijk_float_matrix(ras_xyz, ras_to_ijk),
        "center_ras": center_ras,
        "config": config,
        "extras": extras,
    }
    return ctx, img


def default_detection_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "threshold": float(args.metal_threshold_hu),
        "max_points": int(args.max_points),
        "max_lines": int(args.max_lines),
        "selection_target_count": (None if args.selection_target_count is None else int(args.selection_target_count)),
        "inlier_radius_mm": float(args.inlier_radius_mm),
        "min_length_mm": float(args.min_length_mm),
        "min_inliers": int(args.min_inliers),
        "ransac_iterations": int(args.ransac_iterations),
        "use_head_mask": bool(args.use_head_mask),
        "build_head_mask": bool(args.build_head_mask),
        "head_mask_threshold_hu": float(args.head_mask_threshold_hu),
        "head_mask_aggressive_cleanup": bool(args.head_mask_aggressive_cleanup),
        "head_mask_close_mm": float(args.head_mask_close_mm),
        "head_mask_method": str(args.head_mask_method),
        "head_gate_erode_vox": int(args.head_gate_erode_vox),
        "head_gate_dilate_vox": int(args.head_gate_dilate_vox),
        "head_gate_margin_mm": float(args.head_gate_margin_mm),
        "min_metal_depth_mm": float(args.min_metal_depth_mm),
        "max_metal_depth_mm": float(args.max_metal_depth_mm),
        "min_blob_voxels": int(args.min_blob_voxels),
        "max_blob_voxels": int(args.max_blob_voxels),
        "min_blob_peak_hu": float(args.min_blob_peak_hu) if args.min_blob_peak_hu is not None else None,
        "use_distance_mask_for_blob_candidates": bool(args.use_distance_mask_for_blob_candidates),
        "enable_rescue_pass": bool(args.enable_rescue_pass),
        "rescue_min_inliers_scale": float(args.rescue_min_inliers_scale),
        "rescue_max_lines": int(args.rescue_max_lines),
        "min_model_score": float(args.min_model_score) if bool(args.use_model_score) else None,
        "debug_masks": False,
        "match_start_mm": float(args.match_start_mm),
    }


def iter_subject_rows(dataset_root: Path, subjects_filter: set[str] | None) -> list[dict[str, str]]:
    manifest_path = dataset_root / "contact_label_dataset" / "subjects.tsv"
    rows = _read_tsv_rows(manifest_path)
    out = []
    for row in rows:
        sid = str(row["subject_id"])
        if subjects_filter and sid not in subjects_filter:
            continue
        out.append(row)
    return out


def evaluate_subject(
    row: dict[str, str],
    *,
    pipeline_key: str,
    registry: PipelineRegistry,
    config: dict[str, Any],
    extras: dict[str, Any],
    match_distance_mm: float,
    match_angle_deg: float,
    guided_gt_root: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    subject_id = str(row["subject_id"])
    gt_shanks, gt_source_path = load_reference_ground_truth_shanks(row, guided_gt_root)
    ctx, _img = build_detection_context(row["ct_path"], run_id=f"eval_{subject_id}_{pipeline_key}", config=config, extras=dict(extras))
    result = registry.run(pipeline_key, ctx)
    if str(result.get("status", "ok")).lower() == "error":
        err = dict(result.get("error") or {})
        raise RuntimeError(f"{subject_id}: {err.get('message', 'Detection failed')} (stage={err.get('stage', 'pipeline')})")
    pred_shanks = predicted_shanks_from_result(result)
    assignments, match_summary = match_shanks(
        gt_shanks,
        pred_shanks,
        match_distance_mm=match_distance_mm,
        match_angle_deg=match_angle_deg,
        match_start_mm=float(config.get("match_start_mm", 15.0)),
    )
    diagnostics = dict(result.get("diagnostics") or {})
    counts = dict(diagnostics.get("counts") or {})
    timing = dict(diagnostics.get("timing") or {})
    matched_rows = [r for r in assignments if int(r.get("matched", 0)) == 1]
    mean_end_error = float(np.mean([float(r["end_error_mm"]) for r in matched_rows])) if matched_rows else float("nan")
    mean_start_error = float(np.mean([float(r["start_error_mm"]) for r in matched_rows])) if matched_rows else float("nan")
    mean_match_angle = float(np.mean([float(r["angle_deg"]) for r in matched_rows])) if matched_rows else float("nan")
    subject_summary = {
        "subject_id": subject_id,
        "pipeline_key": pipeline_key,
        "ct_path": row["ct_path"],
        "labels_path": row["labels_path"],
        "gt_source_path": gt_source_path,
        "gt_shanks": len(gt_shanks),
        "pred_shanks": len(pred_shanks),
        "matched_shanks": match_summary["matched"],
        "false_negative": match_summary["false_negative"],
        "false_positive": match_summary["false_positive"],
        "recall": 0.0 if not gt_shanks else float(match_summary["matched"]) / float(len(gt_shanks)),
        "precision": 0.0 if not pred_shanks else float(match_summary["matched"]) / float(len(pred_shanks)),
        "mean_match_end_error_mm": "" if math.isnan(mean_end_error) else f"{mean_end_error:.4f}",
        "mean_match_start_error_mm": "" if math.isnan(mean_start_error) else f"{mean_start_error:.4f}",
        "mean_match_angle_deg": "" if math.isnan(mean_match_angle) else f"{mean_match_angle:.4f}",
        "candidate_points_total": int(counts.get("candidate_points_total", 0)),
        "candidate_points_after_depth": int(counts.get("candidate_points_after_depth", 0)),
        "blob_count_total": int(counts.get("blob_count_total", 0)),
        "blob_count_kept": int(counts.get("blob_count_kept", 0)),
        "fit1_lines_proposed": int(counts.get("fit1_lines_proposed", 0)),
        "fit2_lines_kept": int(counts.get("fit2_lines_kept", 0)),
        "rescue_lines_kept": int(counts.get("rescue_lines_kept", 0)),
        "runtime_total_ms": f"{float(timing.get('total_ms', 0.0)):.2f}",
    }
    artifact_payload = {
        "subject_summary": subject_summary,
        "assignments": assignments,
        "result": result,
        "ground_truth_shanks": [gt.__dict__ for gt in gt_shanks],
        "predicted_shanks": [pred.__dict__ for pred in pred_shanks],
    }
    return subject_summary, assignments, artifact_payload


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate SEEG shank detection against the localization dataset")
    p.add_argument("--dataset-root", required=True, help="Path to seeg_localization root")
    p.add_argument("--out-dir", required=True, help="Directory for evaluation outputs")
    p.add_argument("--pipeline-key", default="contact_pitch_v1", help="Detection pipeline key")
    p.add_argument("--subjects", default="", help="Comma-separated subject ids to evaluate (default: all in manifest)")
    p.add_argument("--match-distance-mm", type=float, default=4.0, help="Maximum deep-end error for a match.")
    p.add_argument("--match-start-mm", type=float, default=15.0, help="Maximum shallow-end/start error for a match.")
    p.add_argument("--match-angle-deg", type=float, default=25.0, help="Maximum direction error for a match.")
    p.add_argument("--metal-threshold-hu", type=float, default=1800.0)
    p.add_argument("--use-head-mask", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--build-head-mask", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--head-mask-threshold-hu", type=float, default=-500.0)
    p.add_argument("--head-mask-close-mm", type=float, default=2.0)
    p.add_argument("--head-mask-method", choices=["outside_air", "not_air_lcc"], default="outside_air")
    p.add_argument("--head-mask-aggressive-cleanup", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--head-gate-erode-vox", type=int, default=1)
    p.add_argument("--head-gate-dilate-vox", type=int, default=1)
    p.add_argument("--head-gate-margin-mm", type=float, default=0.0)
    p.add_argument("--min-metal-depth-mm", type=float, default=5.0)
    p.add_argument("--max-metal-depth-mm", type=float, default=220.0)
    p.add_argument("--max-points", type=int, default=300000)
    p.add_argument("--max-lines", type=int, default=30)
    p.add_argument("--selection-target-count", type=int, default=None, help="Optional exact number of shanks to keep in final selection.")
    p.add_argument("--inlier-radius-mm", type=float, default=1.2)
    p.add_argument("--min-length-mm", type=float, default=20.0)
    p.add_argument("--min-inliers", type=int, default=250)
    p.add_argument("--ransac-iterations", type=int, default=240)
    p.add_argument("--min-blob-voxels", type=int, default=2)
    p.add_argument("--max-blob-voxels", type=int, default=1200)
    p.add_argument("--min-blob-peak-hu", type=float, default=None)
    p.add_argument("--use-distance-mask-for-blob-candidates", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--enable-rescue-pass", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--rescue-min-inliers-scale", type=float, default=0.6)
    p.add_argument("--rescue-max-lines", type=int, default=6)
    p.add_argument("--use-model-score", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--min-model-score", type=float, default=0.10)
    p.add_argument("--electrode-library", default=None)
    p.add_argument("--guided-gt-root", default=None, help="Optional root containing <subject>/<subject>_shanks_guided.tsv to use as GT")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects_filter = {s.strip() for s in str(args.subjects).split(",") if s.strip()} or None
    registry = PipelineRegistry()
    register_builtin_pipelines(registry)
    config = default_detection_config(args)
    extras: dict[str, Any] = {}
    if bool(args.use_model_score):
        extras["models_by_id"] = model_map(load_electrode_library(args.electrode_library))

    subject_rows = iter_subject_rows(dataset_root, subjects_filter)
    summaries: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []

    for row in subject_rows:
        subject_id = str(row["subject_id"])
        print(f"[eval] {subject_id} pipeline={args.pipeline_key}")
        summary, assignments, artifact_payload = evaluate_subject(
            row,
            pipeline_key=str(args.pipeline_key),
            registry=registry,
            config=config,
            extras=extras,
            match_distance_mm=float(args.match_distance_mm),
            match_angle_deg=float(args.match_angle_deg),
            guided_gt_root=args.guided_gt_root,
        )
        summaries.append(summary)
        assignment_rows.extend(assignments)
        with open(out_dir / f"{subject_id}_details.json", "w", encoding="utf-8") as f:
            json.dump(artifact_payload, f, indent=2)

    summary_fields = [
        "subject_id",
        "pipeline_key",
        "gt_shanks",
        "pred_shanks",
        "matched_shanks",
        "false_negative",
        "false_positive",
        "recall",
        "precision",
        "mean_match_end_error_mm",
        "mean_match_start_error_mm",
        "mean_match_angle_deg",
        "candidate_points_total",
        "candidate_points_after_depth",
        "blob_count_total",
        "blob_count_kept",
        "fit1_lines_proposed",
        "fit2_lines_kept",
        "rescue_lines_kept",
        "runtime_total_ms",
        "ct_path",
        "labels_path",
        "gt_source_path",
    ]
    _write_tsv(out_dir / "subject_metrics.tsv", summaries, summary_fields)

    assignment_fields = [
        "subject_id",
        "gt_shank",
        "pred_shank",
        "matched",
        "reason",
        "start_error_mm",
        "end_error_mm",
        "angle_deg",
        "midpoint_dist_mm",
        "mean_contact_dist_mm",
        "max_contact_dist_mm",
        "pred_support_count",
        "pred_confidence",
    ]
    _write_tsv(out_dir / "shank_assignments.tsv", assignment_rows, assignment_fields)

    cohort = {
        "pipeline_key": str(args.pipeline_key),
        "subject_count": len(summaries),
        "gt_shanks": int(sum(int(r["gt_shanks"]) for r in summaries)),
        "pred_shanks": int(sum(int(r["pred_shanks"]) for r in summaries)),
        "matched_shanks": int(sum(int(r["matched_shanks"]) for r in summaries)),
    }
    cohort["recall"] = 0.0 if cohort["gt_shanks"] <= 0 else float(cohort["matched_shanks"]) / float(cohort["gt_shanks"])
    cohort["precision"] = 0.0 if cohort["pred_shanks"] <= 0 else float(cohort["matched_shanks"]) / float(cohort["pred_shanks"])
    with open(out_dir / "cohort_summary.json", "w", encoding="utf-8") as f:
        json.dump(cohort, f, indent=2)

    print(
        "[eval] "
        f"subjects={cohort['subject_count']} gt={cohort['gt_shanks']} pred={cohort['pred_shanks']} "
        f"matched={cohort['matched_shanks']} recall={cohort['recall']:.3f} precision={cohort['precision']:.3f}"
    )
    print(f"[eval] wrote outputs under {out_dir}")


if __name__ == "__main__":
    main()
