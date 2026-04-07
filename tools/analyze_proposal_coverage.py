#!/usr/bin/env python3
"""Analyze where blob_persistence_v2 loses shanks in the proposal stage."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from eval_seeg_localization import (  # type: ignore
    PipelineRegistry,
    build_detection_context,
    compare_shanks,
    default_detection_config,
    iter_subject_rows,
    load_ground_truth_shanks,
    match_shanks,
    register_builtin_pipelines,
)


def _unit(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(3)
    n = float(np.linalg.norm(arr))
    if n <= 1e-8:
        return np.asarray([0.0, 0.0, 1.0], dtype=float)
    return arr / n


def _line_distance(p0: np.ndarray, d0: np.ndarray, p1: np.ndarray, d1: np.ndarray) -> float:
    u = _unit(d0)
    v = _unit(d1)
    w0 = p0 - p1
    c = np.cross(u, v)
    cn = float(np.linalg.norm(c))
    if cn <= 1e-6:
        return float(np.linalg.norm(np.cross(w0, u)))
    return float(abs(np.dot(w0, c)) / cn)


def _point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    seg = end - start
    denom = float(np.dot(seg, seg))
    if denom <= 1e-8:
        return float(np.linalg.norm(point - start))
    t = float(np.dot(point - start, seg) / denom)
    t = min(1.0, max(0.0, t))
    closest = start + t * seg
    return float(np.linalg.norm(point - closest))


def _write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _strict_match_exists(gt, candidate_lines: list[dict[str, Any]], *, match_end_mm: float, match_start_mm: float, match_angle_deg: float) -> bool:
    for cand in candidate_lines:
        pm = compare_shanks(gt, _candidate_to_pred(cand))
        if pm.end_error_mm <= match_end_mm and pm.start_error_mm <= match_start_mm and pm.angle_deg <= match_angle_deg:
            return True
    return False


def _candidate_to_pred(candidate: dict[str, Any]):
    from eval_candidate_ranker import candidate_to_predicted_shank  # type: ignore
    return candidate_to_predicted_shank(candidate)


def _seed_near_gt(gt, seed: dict[str, Any], *, max_line_distance_mm: float, max_angle_deg: float) -> bool:
    gt_start = np.asarray(gt.start_ras, dtype=float)
    gt_end = np.asarray(gt.end_ras, dtype=float)
    gt_dir = _unit(gt_end - gt_start)
    seed_point = np.asarray(seed.get("point_ras") or [0.0, 0.0, 0.0], dtype=float)
    seed_axis = _unit(np.asarray(seed.get("axis_ras") or [0.0, 0.0, 1.0], dtype=float))
    angle = float(np.degrees(np.arccos(np.clip(abs(float(np.dot(gt_dir, seed_axis))), 0.0, 1.0))))
    line_dist = _line_distance(seed_point, seed_axis, 0.5 * (gt_start + gt_end), gt_dir)
    return angle <= max_angle_deg and line_dist <= max_line_distance_mm


def _support_near_gt(gt, support: dict[str, Any], *, max_point_distance_mm: float, max_angle_deg: float) -> bool:
    gt_start = np.asarray(gt.start_ras, dtype=float)
    gt_end = np.asarray(gt.end_ras, dtype=float)
    gt_dir = _unit(gt_end - gt_start)
    pt = np.asarray(support.get("point_ras") or [0.0, 0.0, 0.0], dtype=float)
    axis = _unit(np.asarray(support.get("axis_ras") or [0.0, 0.0, 1.0], dtype=float))
    angle = float(np.degrees(np.arccos(np.clip(abs(float(np.dot(gt_dir, axis))), 0.0, 1.0))))
    dist = _point_to_segment_distance(pt, gt_start, gt_end)
    return angle <= max_angle_deg and dist <= max_point_distance_mm


def _best_candidate_metrics(gt, candidate_lines: list[dict[str, Any]]) -> dict[str, float]:
    best = None
    for cand in candidate_lines:
        pm = compare_shanks(gt, _candidate_to_pred(cand))
        if best is None or pm.score < best.score:
            best = pm
    if best is None:
        return {"best_end_error_mm": float("inf"), "best_start_error_mm": float("inf"), "best_angle_deg": float("inf")}
    return {
        "best_end_error_mm": float(best.end_error_mm),
        "best_start_error_mm": float(best.start_error_mm),
        "best_angle_deg": float(best.angle_deg),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze proposal-stage coverage for blob_persistence_v2")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--subjects", default="T1,T2,T3,T25")
    p.add_argument("--pipeline-key", default="blob_persistence_v2")
    p.add_argument("--metal-threshold-hu", type=float, default=1800.0)
    p.add_argument("--min-metal-depth-mm", type=float, default=5.0)
    p.add_argument("--max-metal-depth-mm", type=float, default=220.0)
    p.add_argument("--min-inliers", type=int, default=6)
    p.add_argument("--proposal-seed-limit", type=int, default=180)
    p.add_argument("--support-distance-mm", type=float, default=3.0)
    p.add_argument("--support-angle-deg", type=float, default=15.0)
    p.add_argument("--seed-line-distance-mm", type=float, default=4.0)
    p.add_argument("--seed-angle-deg", type=float, default=12.0)
    p.add_argument("--loose-end-mm", type=float, default=15.0)
    p.add_argument("--loose-start-mm", type=float, default=20.0)
    p.add_argument("--loose-angle-deg", type=float, default=10.0)
    p.add_argument("--strict-end-mm", type=float, default=4.0)
    p.add_argument("--strict-start-mm", type=float, default=15.0)
    p.add_argument("--strict-angle-deg", type=float, default=25.0)
    p.add_argument("--head-mask-method", choices=["outside_air", "not_air_lcc"], default="outside_air")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    subjects = {s.strip() for s in str(args.subjects).split(",") if s.strip()}
    rows = iter_subject_rows(dataset_root, subjects)
    if not rows:
        raise SystemExit("No subjects matched the requested filter")

    registry = PipelineRegistry()
    register_builtin_pipelines(registry)
    if not hasattr(args, "selection_target_count"):
        args.selection_target_count = None
    args.use_head_mask = True
    args.build_head_mask = True
    args.head_mask_threshold_hu = -500.0
    args.head_mask_close_mm = 2.0
    args.head_mask_aggressive_cleanup = True
    args.head_gate_erode_vox = 1
    args.head_gate_dilate_vox = 1
    args.head_gate_margin_mm = 0.0
    args.max_points = 300000
    args.max_lines = 40
    args.inlier_radius_mm = 1.2
    args.min_length_mm = 20.0
    args.ransac_iterations = 240
    args.min_blob_voxels = 2
    args.max_blob_voxels = 1200
    args.min_blob_peak_hu = None
    args.use_distance_mask_for_blob_candidates = True
    args.enable_rescue_pass = True
    args.rescue_min_inliers_scale = 0.6
    args.rescue_max_lines = 6
    args.use_model_score = False
    args.min_model_score = 0.1
    args.electrode_library = None
    args.match_start_mm = float(args.strict_start_mm)

    config = default_detection_config(args)
    config["proposal_seed_limit"] = int(args.proposal_seed_limit)
    config["return_candidate_lines"] = True
    config["return_proposal_debug"] = True

    per_gt_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []

    for row in rows:
        sid = str(row["subject_id"])
        gt_shanks = load_ground_truth_shanks(row["labels_path"], row.get("shanks_path"))
        ctx, _ = build_detection_context(row["ct_path"], run_id=f"proposal_{sid}", config=config, extras={"electrode_library": None})
        result = registry.run(str(args.pipeline_key), ctx)
        if str(result.get("status", "ok")).lower() == "error":
            err = dict(result.get("error") or {})
            raise RuntimeError(f"{sid}: {err.get('message', 'Detection failed')} (stage={err.get('stage', 'pipeline')})")

        extras = dict((result.get("meta") or {}).get("extras") or {})
        proposal_debug = dict(extras.get("proposal_debug") or {})
        candidate_lines = list(extras.get("candidate_lines") or [])
        core_seeds = list(proposal_debug.get("core_seeds") or [])
        rescue_seeds = list(proposal_debug.get("rescue_seeds") or [])
        core_supports = list(proposal_debug.get("core_supports") or [])
        rescue_supports = list(proposal_debug.get("rescue_supports") or [])
        all_supports = core_supports + rescue_supports
        all_seeds = core_seeds + rescue_seeds

        strict_preds = [_candidate_to_pred(c) for c in candidate_lines]
        _, strict_summary = match_shanks(
            gt_shanks,
            strict_preds,
            match_distance_mm=float(args.strict_end_mm),
            match_start_mm=float(args.strict_start_mm),
            match_angle_deg=float(args.strict_angle_deg),
        )

        subject_rows.append({
            "subject_id": sid,
            "gt_count": len(gt_shanks),
            "support_count": len(all_supports),
            "seed_count": len(all_seeds),
            "candidate_count": len(candidate_lines),
            "strict_match_count": int(strict_summary["matched"]),
            "coverage_ratio": float(proposal_debug.get("coverage_ratio", 0.0)),
            "rescue_trigger": int(bool(proposal_debug.get("rescue_trigger", False))),
        })

        for gt in gt_shanks:
            support_hits = [s for s in all_supports if _support_near_gt(gt, s, max_point_distance_mm=float(args.support_distance_mm), max_angle_deg=float(args.support_angle_deg))]
            seed_hits = [s for s in all_seeds if _seed_near_gt(gt, s, max_line_distance_mm=float(args.seed_line_distance_mm), max_angle_deg=float(args.seed_angle_deg))]
            loose_hits = []
            strict_hits = []
            for cand in candidate_lines:
                pm = compare_shanks(gt, _candidate_to_pred(cand))
                if pm.end_error_mm <= float(args.loose_end_mm) and pm.start_error_mm <= float(args.loose_start_mm) and pm.angle_deg <= float(args.loose_angle_deg):
                    loose_hits.append(cand)
                if pm.end_error_mm <= float(args.strict_end_mm) and pm.start_error_mm <= float(args.strict_start_mm) and pm.angle_deg <= float(args.strict_angle_deg):
                    strict_hits.append(cand)
            best = _best_candidate_metrics(gt, candidate_lines)
            if strict_hits:
                failure_stage = "strict_match_present"
            elif loose_hits:
                failure_stage = "endpoint_or_selection_error"
            elif seed_hits:
                failure_stage = "fit_failed_from_seed"
            elif support_hits:
                failure_stage = "seed_missing"
            else:
                failure_stage = "support_missing"
            per_gt_rows.append({
                "subject_id": sid,
                "gt_shank": gt.shank,
                "support_hit_count": len(support_hits),
                "support_hit_weight": f"{sum(float(s.get('support_weight', 0.0)) for s in support_hits):.4f}",
                "seed_hit_count": len(seed_hits),
                "loose_candidate_count": len(loose_hits),
                "strict_candidate_count": len(strict_hits),
                "best_end_error_mm": f"{best['best_end_error_mm']:.4f}" if math.isfinite(best['best_end_error_mm']) else "",
                "best_start_error_mm": f"{best['best_start_error_mm']:.4f}" if math.isfinite(best['best_start_error_mm']) else "",
                "best_angle_deg": f"{best['best_angle_deg']:.4f}" if math.isfinite(best['best_angle_deg']) else "",
                "failure_stage": failure_stage,
            })

    _write_tsv(out_dir / "subject_summary.tsv", subject_rows, [
        "subject_id", "gt_count", "support_count", "seed_count", "candidate_count", "strict_match_count", "coverage_ratio", "rescue_trigger"
    ])
    _write_tsv(out_dir / "gt_stage_summary.tsv", per_gt_rows, [
        "subject_id", "gt_shank", "support_hit_count", "support_hit_weight", "seed_hit_count", "loose_candidate_count", "strict_candidate_count", "best_end_error_mm", "best_start_error_mm", "best_angle_deg", "failure_stage"
    ])

    failure_counts: dict[str, int] = {}
    for row in per_gt_rows:
        failure_counts[str(row["failure_stage"])] = failure_counts.get(str(row["failure_stage"]), 0) + 1
    summary = {
        "subjects": sorted(subjects),
        "subject_count": len(subject_rows),
        "gt_total": len(per_gt_rows),
        "failure_stage_counts": failure_counts,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
