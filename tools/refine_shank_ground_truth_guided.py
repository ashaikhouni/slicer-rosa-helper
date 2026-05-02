#!/usr/bin/env python3
"""Refine cached shank ground truth with the guided-fit contact fitter.

This script treats the cached shank table as planned trajectories, extracts CT
candidate points from each subject volume, then reuses the same
`fit_electrode_axis_and_tip` logic used by the Postop CT Localization guided
fit workflow to snap each shank toward the actual metal trajectory.

Outputs are written separately from the original dataset so the manual labels
remain untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIB_DIR = os.path.join(REPO_ROOT, "CommonLib")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

from rosa_core.contact_fit import (  # noqa: E402
    fit_electrode_axis_and_tip,
    refine_fit_batch_with_global_coordinate_ascent,
)
from rosa_core.electrode_models import load_electrode_library  # noqa: E402
from shank_core.io import image_ijk_ras_matrices, kji_to_ras_points_matrix, ras_to_ijk_float_matrix, read_volume  # noqa: E402
from shank_core.masking import build_preview_masks  # noqa: E402


def read_tsv_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv_rows(path: str | Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _unit(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(3)
    n = float(np.linalg.norm(arr))
    if n <= 1e-8:
        return np.asarray([0.0, 0.0, 1.0], dtype=float)
    return arr / n


def _ras_to_lps(point_ras: np.ndarray | list[float]) -> np.ndarray:
    p = np.asarray(point_ras, dtype=float).reshape(3)
    return np.asarray([-p[0], -p[1], p[2]], dtype=float)


def _lps_to_ras(point_lps: np.ndarray | list[float]) -> np.ndarray:
    p = np.asarray(point_lps, dtype=float).reshape(3)
    return np.asarray([-p[0], -p[1], p[2]], dtype=float)


def _center_pitch_from_model(model: dict[str, Any]) -> float | None:
    c2c = model.get("center_to_center_separation_mm")
    if c2c is not None:
        try:
            val = float(c2c)
        except Exception:
            val = 0.0
        if math.isfinite(val) and val > 0.0:
            return val
    offsets = list(model.get("contact_center_offsets_from_tip_mm") or [])
    if len(offsets) >= 2:
        diffs = np.diff(np.asarray(offsets, dtype=float))
        if diffs.size:
            return float(np.median(diffs))
    return None


def choose_model_for_shank(models: list[dict[str, Any]], contact_count: int, mean_intercontact_mm: float, span_mm: float) -> dict[str, Any] | None:
    best = None
    best_score = None
    for model in models:
        model_count = int(model.get("contact_count") or 0)
        if model_count != int(contact_count):
            continue
        pitch = _center_pitch_from_model(model)
        if pitch is None:
            continue
        model_span = float(model.get("total_exploration_length_mm") or 0.0)
        pitch_delta = abs(float(mean_intercontact_mm) - float(pitch))
        span_delta = abs(float(span_mm) - float(model_span))
        score = pitch_delta + 0.2 * span_delta
        if best_score is None or score < best_score:
            best = model
            best_score = score
    if best is not None:
        return best

    # Fallback: nearest contact count, then pitch/span.
    for model in models:
        pitch = _center_pitch_from_model(model)
        if pitch is None:
            continue
        count_delta = abs(int(model.get("contact_count") or 0) - int(contact_count))
        model_span = float(model.get("total_exploration_length_mm") or 0.0)
        score = 10.0 * count_delta + abs(float(mean_intercontact_mm) - float(pitch)) + 0.2 * abs(float(span_mm) - float(model_span))
        if best_score is None or score < best_score:
            best = model
            best_score = score
    return best


def _extract_threshold_candidates_lps(
    arr_kji: np.ndarray,
    ijk_to_ras: np.ndarray,
    threshold: float,
    candidate_mask_kji: np.ndarray | None = None,
) -> np.ndarray:
    if candidate_mask_kji is not None:
        idx = np.argwhere(np.asarray(candidate_mask_kji, dtype=bool))
    else:
        idx = np.argwhere(np.asarray(arr_kji, dtype=float) >= float(threshold))
    if idx.size == 0:
        return np.empty((0, 3), dtype=float)
    ras = np.asarray(kji_to_ras_points_matrix(idx.astype(float), ijk_to_ras), dtype=float).reshape(-1, 3)
    lps = ras.copy()
    lps[:, 0] *= -1.0
    lps[:, 1] *= -1.0
    return lps


def _adaptive_subject_candidates(
    *,
    arr_kji: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    ijk_to_ras: np.ndarray,
    start_threshold_hu: float,
    min_threshold_hu: float,
    threshold_step_hu: float,
    target_candidate_points: int,
    head_mask_method: str,
    head_mask_threshold_hu: float,
    min_metal_depth_mm: float,
    max_metal_depth_mm: float,
) -> tuple[np.ndarray, np.ndarray | None, float, dict[str, object]]:
    threshold = float(start_threshold_hu)
    min_threshold = float(min_threshold_hu)
    step = max(1.0, float(threshold_step_hu))
    best_preview: dict[str, object] | None = None
    best_threshold = float(threshold)
    best_count = -1
    while True:
        preview = build_preview_masks(
            arr_kji=np.asarray(arr_kji, dtype=float),
            spacing_xyz=tuple(spacing_xyz),
            threshold=float(threshold),
            use_head_mask=True,
            build_head_mask=True,
            head_mask_threshold_hu=float(head_mask_threshold_hu),
            head_mask_method=str(head_mask_method),
            head_gate_erode_vox=1,
            head_gate_dilate_vox=1,
            head_gate_margin_mm=0.0,
            min_metal_depth_mm=float(min_metal_depth_mm),
            max_metal_depth_mm=float(max_metal_depth_mm),
            include_debug_masks=False,
        )
        count = int(preview.get("depth_kept_count") or 0)
        if count > best_count:
            best_preview = preview
            best_threshold = float(threshold)
            best_count = count
        if count >= int(target_candidate_points) or threshold <= min_threshold + 1e-6:
            break
        threshold = max(min_threshold, threshold - step)
    if best_preview is None:
        best_preview = {}
    depth_map_kji = (
        np.asarray(best_preview.get("head_distance_map_kji"), dtype=np.float32)
        if best_preview.get("head_distance_map_kji") is not None
        else None
    )
    candidate_points_lps = _extract_threshold_candidates_lps(
        arr_kji,
        ijk_to_ras,
        threshold=float(best_threshold),
        candidate_mask_kji=np.asarray(best_preview.get("metal_depth_pass_mask_kji"), dtype=bool),
    )
    return candidate_points_lps, depth_map_kji, float(best_threshold), best_preview


def _adaptive_roi_sequence(base_roi_mm: float) -> list[float]:
    values = [float(base_roi_mm), 8.0, 10.0]
    out: list[float] = []
    for value in values:
        if any(abs(value - existing) < 1e-6 for existing in out):
            continue
        out.append(value)
    return out


def _fit_with_adaptive_roi(
    *,
    candidate_points_lps: np.ndarray,
    planned_entry_lps: np.ndarray,
    planned_target_lps: np.ndarray,
    contact_offsets_mm: list[float] | np.ndarray | None,
    tip_at: str,
    base_roi_radius_mm: float,
    max_angle_deg: float,
    max_depth_shift_mm: float,
    fit_mode: str,
) -> dict[str, object]:
    fit: dict[str, object] | None = None
    for roi_mm in _adaptive_roi_sequence(float(base_roi_radius_mm)):
        fit = fit_electrode_axis_and_tip(
            candidate_points_lps=candidate_points_lps,
            planned_entry_lps=planned_entry_lps,
            planned_target_lps=planned_target_lps,
            contact_offsets_mm=contact_offsets_mm,
            tip_at=tip_at,
            roi_radius_mm=float(roi_mm),
            max_angle_deg=float(max_angle_deg),
            max_depth_shift_mm=float(max_depth_shift_mm),
            fit_mode=str(fit_mode),
        )
        fit["roi_radius_mm_attempted"] = float(roi_mm)
        if bool(fit.get("success")):
            break
        reason = str(fit.get("reason") or "").lower()
        sparse_failure = (
            "too few" in reason
            or "no candidate" in reason
            or "no compact clusters" in reason
            or int(fit.get("points_in_roi") or 0) < 300
        )
        if not sparse_failure:
            break
    return fit if fit is not None else {"success": False, "reason": "fit_not_attempted"}


def _depth_at_ras(point_ras: np.ndarray, depth_map_kji: np.ndarray | None, ras_to_ijk) -> float | None:
    if depth_map_kji is None:
        return None
    ijk = ras_to_ijk(point_ras)
    i = int(round(float(ijk[0])))
    j = int(round(float(ijk[1])))
    k = int(round(float(ijk[2])))
    if k < 0 or j < 0 or i < 0 or k >= depth_map_kji.shape[0] or j >= depth_map_kji.shape[1] or i >= depth_map_kji.shape[2]:
        return None
    value = float(depth_map_kji[k, j, i])
    return value if math.isfinite(value) else None


def orient_entry_target(start_ras: np.ndarray, end_ras: np.ndarray, depth_map_kji: np.ndarray | None, ras_to_ijk) -> tuple[np.ndarray, np.ndarray]:
    d0 = _depth_at_ras(start_ras, depth_map_kji, ras_to_ijk)
    d1 = _depth_at_ras(end_ras, depth_map_kji, ras_to_ijk)
    if d0 is not None and d1 is not None and abs(d0 - d1) > 1e-3:
        return (start_ras, end_ras) if d0 <= d1 else (end_ras, start_ras)
    return start_ras, end_ras


def refine_subject_shanks(
    *,
    subject_row: dict[str, str],
    shank_rows: list[dict[str, str]],
    models: list[dict[str, Any]],
    threshold_hu: float,
    min_threshold_hu: float,
    threshold_step_hu: float,
    target_candidate_points: int,
    roi_radius_mm: float,
    max_angle_deg: float,
    max_depth_shift_mm: float,
    fit_mode: str,
    head_mask_method: str,
    head_mask_threshold_hu: float,
    min_metal_depth_mm: float,
    max_metal_depth_mm: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    subject_id = str(subject_row["subject_id"])
    img, arr_kji, spacing_xyz = read_volume(subject_row["ct_path"])
    ijk_to_ras, ras_to_ijk = image_ijk_ras_matrices(img)
    ras_to_ijk_fn = lambda ras_xyz: ras_to_ijk_float_matrix(ras_xyz, ras_to_ijk)

    candidate_points_lps, depth_map_kji, used_threshold_hu, preview = _adaptive_subject_candidates(
        arr_kji=np.asarray(arr_kji, dtype=float),
        spacing_xyz=tuple(spacing_xyz),
        ijk_to_ras=ijk_to_ras,
        start_threshold_hu=float(threshold_hu),
        min_threshold_hu=float(min_threshold_hu),
        threshold_step_hu=float(threshold_step_hu),
        target_candidate_points=int(target_candidate_points),
        head_mask_method=str(head_mask_method),
        head_mask_threshold_hu=float(head_mask_threshold_hu),
        min_metal_depth_mm=float(min_metal_depth_mm),
        max_metal_depth_mm=float(max_metal_depth_mm),
    )

    refined_rows: list[dict[str, object]] = []
    metrics_rows: list[dict[str, object]] = []
    success_count = 0
    fit_results_by_name: dict[str, dict[str, object]] = {}
    offsets_by_name: dict[str, list[float]] = {}
    prepared_rows: list[tuple[dict[str, str], dict[str, Any] | None, np.ndarray, np.ndarray]] = []
    effective_fit_mode = "em_v1" if str(fit_mode) == "em_v2" else str(fit_mode)

    for row in shank_rows:
        start_ras = np.asarray([float(row["start_x"]), float(row["start_y"]), float(row["start_z"])], dtype=float)
        end_ras = np.asarray([float(row["end_x"]), float(row["end_y"]), float(row["end_z"])], dtype=float)
        entry_ras, target_ras = orient_entry_target(start_ras, end_ras, depth_map_kji, ras_to_ijk_fn)
        model = choose_model_for_shank(
            models,
            contact_count=int(row.get("contact_count") or 0),
            mean_intercontact_mm=float(row.get("mean_intercontact_mm") or 0.0),
            span_mm=float(row.get("span_mm") or np.linalg.norm(target_ras - entry_ras)),
        )
        offsets = list(model.get("contact_center_offsets_from_tip_mm") or []) if model is not None else []
        prepared_rows.append((row, model, entry_ras, target_ras))
        offsets_by_name[row["shank"]] = offsets
        fit = _fit_with_adaptive_roi(
            candidate_points_lps=candidate_points_lps,
            planned_entry_lps=_ras_to_lps(entry_ras),
            planned_target_lps=_ras_to_lps(target_ras),
            contact_offsets_mm=offsets,
            tip_at="target",
            base_roi_radius_mm=float(roi_radius_mm),
            max_angle_deg=float(max_angle_deg),
            max_depth_shift_mm=float(max_depth_shift_mm),
            fit_mode=str(effective_fit_mode),
        )
        fit_results_by_name[row["shank"]] = fit

    if str(fit_mode) == "em_v2":
        fit_results_by_name = refine_fit_batch_with_global_coordinate_ascent(
            fit_results_by_name,
            candidate_points_lps=candidate_points_lps,
            contact_offsets_by_name=offsets_by_name,
        )
        for fit in fit_results_by_name.values():
            if bool(fit.get("success")):
                fit["fit_mode_used"] = "em_v2"
                fit["fit_mode_requested"] = "em_v2"

    for row, model, entry_ras, target_ras in prepared_rows:
        fit = fit_results_by_name[row["shank"]]

        refined = dict(row)
        refined["subject_id"] = subject_id
        refined["refit_threshold_hu"] = f"{float(used_threshold_hu):.1f}"
        refined["refit_model_id"] = str(model.get("id") or "") if model is not None else ""
        refined["refit_status"] = "failed"
        refined["refit_reason"] = str(fit.get("reason") or "")

        metric_row: dict[str, object] = {
            "subject_id": subject_id,
            "shank": row["shank"],
            "success": int(bool(fit.get("success"))),
            "model_id": str(model.get("id") or "") if model is not None else "",
            "points_in_roi": int(fit.get("points_in_roi") or 0),
            "slab_centroids": int(fit.get("slab_centroids") or 0),
            "slab_inliers": int(fit.get("slab_inliers") or 0),
            "angle_deg": f"{float(fit.get('angle_deg') or 0.0):.4f}",
            "tip_shift_mm": f"{float(fit.get('tip_shift_mm') or 0.0):.4f}",
            "lateral_shift_mm": f"{float(fit.get('lateral_shift_mm') or 0.0):.4f}",
            "residual_mm": f"{float(fit.get('residual_mm') or 0.0):.4f}",
            "roi_radius_mm_attempted": "" if fit.get("roi_radius_mm_attempted") in (None, "") else f"{float(fit.get('roi_radius_mm_attempted')):.2f}",
            "one_d_residual_mm": "" if not math.isfinite(float(fit.get("one_d_residual_mm") or float("nan"))) else f"{float(fit.get('one_d_residual_mm')):.4f}",
            "reason": str(fit.get("reason") or ""),
            "fit_mode_used": str(fit.get("fit_mode_used") or effective_fit_mode),
            "deep_anchor_source": str(fit.get("deep_anchor_source") or ""),
            "terminal_blob_source": str(fit.get("terminal_blob_source") or ""),
            "local_terminal_cluster_count": int(fit.get("local_terminal_cluster_count") or 0),
            "terminal_blob_selected_cluster_id": str(fit.get("terminal_blob_selected_cluster_id") or ""),
            "terminal_blob_selected_anchor_t_mm": "" if fit.get("terminal_blob_selected_anchor_t_mm") in (None, "") else f"{float(fit.get('terminal_blob_selected_anchor_t_mm')):.4f}",
            "terminal_blob_candidate_count": int(fit.get("terminal_blob_candidate_count") or 0),
            "local_terminal_morphology": str(fit.get("local_terminal_morphology") or ""),
            "terminal_anchor_mode": str(fit.get("terminal_anchor_mode") or ""),
            "terminal_support_run_count": int(fit.get("terminal_support_run_count") or 0),
            "terminal_support_longest_run_index": "" if fit.get("terminal_support_longest_run_index") in (None, "") else str(fit.get("terminal_support_longest_run_index")),
            "terminal_support_selected_run_index": "" if fit.get("terminal_support_selected_run_index") in (None, "") else str(fit.get("terminal_support_selected_run_index")),
            "terminal_support_runs_json": str(fit.get("terminal_support_runs_json") or ""),
            "configured_gap_priors_mm": str(fit.get("configured_gap_priors_mm") or ""),
            "axis_support_selected_run_json": str(fit.get("axis_support_selected_run_json") or ""),
            "axis_support_runs_json": str(fit.get("axis_support_runs_json") or ""),
            "proximal_t_override_mm": "" if fit.get("proximal_t_override_mm") in (None, "") else f"{float(fit.get('proximal_t_override_mm')):.4f}",
            "fitted_span_mm": "" if fit.get("fitted_span_mm") in (None, "") else f"{float(fit.get('fitted_span_mm')):.4f}",
            "global_coordinate_ascent_score": "" if fit.get("global_coordinate_ascent_score") in (None, "") else f"{float(fit.get('global_coordinate_ascent_score')):.4f}",
            "terminal_blob_diagnostics_json": str(fit.get("terminal_blob_diagnostics_json") or ""),
        }

        if fit.get("success"):
            fitted_entry_ras = _lps_to_ras(np.asarray(fit["entry_lps"], dtype=float))
            fitted_target_ras = _lps_to_ras(np.asarray(fit["target_lps"], dtype=float))
            direction = _unit(fitted_target_ras - fitted_entry_ras)
            span_mm = float(np.linalg.norm(fitted_target_ras - fitted_entry_ras))
            refined.update(
                {
                    "start_x": f"{float(fitted_entry_ras[0]):.6f}",
                    "start_y": f"{float(fitted_entry_ras[1]):.6f}",
                    "start_z": f"{float(fitted_entry_ras[2]):.6f}",
                    "end_x": f"{float(fitted_target_ras[0]):.6f}",
                    "end_y": f"{float(fitted_target_ras[1]):.6f}",
                    "end_z": f"{float(fitted_target_ras[2]):.6f}",
                    "dir_x": f"{float(direction[0]):.6f}",
                    "dir_y": f"{float(direction[1]):.6f}",
                    "dir_z": f"{float(direction[2]):.6f}",
                    "span_mm": f"{float(span_mm):.6f}",
                    "fit_method": "guided_fit_from_cached_shank",
                    "refit_status": "ok",
                    "refit_reason": "",
                }
            )
            success_count += 1
        refined_rows.append(refined)
        metrics_rows.append(metric_row)

    summary = {
        "subject_id": subject_id,
        "gt_shanks": len(shank_rows),
        "refit_success": int(success_count),
        "refit_failed": int(len(shank_rows) - success_count),
        "candidate_points": int(candidate_points_lps.shape[0]),
        "threshold_hu": float(used_threshold_hu),
    }
    return refined_rows, metrics_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine cached shank TSVs with guided-fit snapping on postop CT")
    parser.add_argument("--dataset-root", required=True, help="Path to seeg_localization root")
    parser.add_argument("--out-dir", required=True, help="Output directory for refined shank TSVs and metrics")
    parser.add_argument("--subjects", default="", help="Comma-separated subject IDs; empty means all")
    parser.add_argument("--shanks-path", default="", help="Override shank TSV path; default uses contact_label_dataset/all_shanks.tsv")
    parser.add_argument("--threshold-hu", type=float, default=600.0)
    parser.add_argument("--min-threshold-hu", type=float, default=500.0)
    parser.add_argument("--threshold-step-hu", type=float, default=50.0)
    parser.add_argument("--target-candidate-points", type=int, default=300000)
    parser.add_argument("--roi-radius-mm", type=float, default=5.0)
    parser.add_argument("--max-angle-deg", type=float, default=12.0)
    parser.add_argument("--max-depth-shift-mm", type=float, default=2.0)
    parser.add_argument("--fit-mode", default="deep_anchor_v2", choices=["deep_anchor_v2", "slab_v1", "em_v1", "em_v2"])
    parser.add_argument("--head-mask-method", choices=["outside_air", "not_air_lcc"], default="outside_air")
    parser.add_argument("--head-mask-threshold-hu", type=float, default=-500.0)
    parser.add_argument("--min-metal-depth-mm", type=float, default=5.0)
    parser.add_argument("--max-metal-depth-mm", type=float, default=220.0)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    manifest_path = dataset_root / "contact_label_dataset" / "subjects.tsv"
    shanks_path = Path(args.shanks_path).expanduser().resolve() if args.shanks_path else dataset_root / "contact_label_dataset" / "all_shanks.tsv"

    subject_rows = read_tsv_rows(manifest_path)
    subjects_filter = {s.strip() for s in str(args.subjects).split(",") if s.strip()}
    if subjects_filter:
        subject_rows = [row for row in subject_rows if str(row["subject_id"]) in subjects_filter]
    if not subject_rows:
        raise SystemExit("No subjects matched")

    shank_rows_all = read_tsv_rows(shanks_path)
    shanks_by_subject: dict[str, list[dict[str, str]]] = {}
    for row in shank_rows_all:
        shanks_by_subject.setdefault(str(row["subject_id"]), []).append(row)

    library = load_electrode_library()
    models = list(library.get("models") or [])

    all_refined: list[dict[str, object]] = []
    all_metrics: list[dict[str, object]] = []
    subject_summary: list[dict[str, object]] = []
    for subject_row in subject_rows:
        subject_id = str(subject_row["subject_id"])
        shank_rows = shanks_by_subject.get(subject_id, [])
        if not shank_rows:
            continue
        refined_rows, metrics_rows, summary = refine_subject_shanks(
            subject_row=subject_row,
            shank_rows=shank_rows,
            models=models,
            threshold_hu=float(args.threshold_hu),
            min_threshold_hu=float(args.min_threshold_hu),
            threshold_step_hu=float(args.threshold_step_hu),
            target_candidate_points=int(args.target_candidate_points),
            roi_radius_mm=float(args.roi_radius_mm),
            max_angle_deg=float(args.max_angle_deg),
            max_depth_shift_mm=float(args.max_depth_shift_mm),
            fit_mode=str(args.fit_mode),
            head_mask_method=str(args.head_mask_method),
            head_mask_threshold_hu=float(args.head_mask_threshold_hu),
            min_metal_depth_mm=float(args.min_metal_depth_mm),
            max_metal_depth_mm=float(args.max_metal_depth_mm),
        )
        all_refined.extend(refined_rows)
        all_metrics.extend(metrics_rows)
        subject_summary.append(summary)
        subject_dir = out_dir / subject_id
        subject_dir.mkdir(parents=True, exist_ok=True)
        write_tsv_rows(subject_dir / f"{subject_id}_shanks_guided.tsv", refined_rows, list(refined_rows[0].keys()))
        write_tsv_rows(subject_dir / f"{subject_id}_guided_fit_metrics.tsv", metrics_rows, list(metrics_rows[0].keys()))
        print(f"[guided-shanks] {subject_id}: refined={summary['refit_success']}/{summary['gt_shanks']} candidates={summary['candidate_points']}")

    if all_refined:
        write_tsv_rows(out_dir / "all_shanks_guided.tsv", all_refined, list(all_refined[0].keys()))
    if all_metrics:
        write_tsv_rows(out_dir / "guided_fit_metrics.tsv", all_metrics, list(all_metrics[0].keys()))
    if subject_summary:
        write_tsv_rows(out_dir / "subject_summary.tsv", subject_summary, list(subject_summary[0].keys()))
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "subject_count": len(subject_summary),
                    "subjects": [str(row["subject_id"]) for row in subject_summary],
                    "gt_total": int(sum(int(row["gt_shanks"]) for row in subject_summary)),
                    "refit_success_total": int(sum(int(row["refit_success"]) for row in subject_summary)),
                    "refit_failed_total": int(sum(int(row["refit_failed"]) for row in subject_summary)),
                },
                f,
                indent=2,
                sort_keys=True,
            )
    print(f"[guided-shanks] wrote {out_dir}")


if __name__ == "__main__":
    main()
