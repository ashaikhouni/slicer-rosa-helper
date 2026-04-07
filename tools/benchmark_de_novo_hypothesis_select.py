#!/usr/bin/env python3
"""Benchmark the de novo hypothesis-select pipeline on the SEEG localization corpus."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIB_DIR = os.path.join(REPO_ROOT, "CommonLib")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

from eval_seeg_localization import (  # noqa: E402
    _write_tsv,
    build_detection_context,
    iter_subject_rows,
    load_reference_ground_truth_shanks,
    match_shanks,
    predicted_shanks_from_result,
)
from rosa_core.electrode_models import load_electrode_library, model_map  # noqa: E402
from shank_engine import PipelineRegistry, register_builtin_pipelines  # noqa: E402


DEFAULT_STAGES = ["support_pool", "hypothesis_generation", "global_selection", "late_model_priors"]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark de novo hypothesis-select shank detection")
    p.add_argument("--dataset-root", required=True, help="Path to seeg_localization root")
    p.add_argument("--out-dir", required=True, help="Directory for benchmark outputs")
    p.add_argument("--subjects", default="", help="Comma-separated subject ids to evaluate")
    p.add_argument("--pipeline-key", default="de_novo_hypothesis_select_v1")
    p.add_argument("--stages", default=",".join(DEFAULT_STAGES), help="Comma-separated ablation stages")
    p.add_argument("--match-distance-mm", type=float, default=4.0)
    p.add_argument("--match-start-mm", type=float, default=15.0)
    p.add_argument("--match-angle-deg", type=float, default=25.0)
    p.add_argument("--use-model-priors", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--electrode-library", default=None)
    p.add_argument("--head-mask-method", choices=["outside_air", "not_air_lcc"], default="outside_air")
    p.add_argument("--head-mask-threshold-hu", type=float, default=-500.0)
    p.add_argument("--min-metal-depth-mm", type=float, default=5.0)
    p.add_argument("--max-metal-depth-mm", type=float, default=220.0)
    p.add_argument("--target-candidate-points", type=int, default=300000)
    p.add_argument("--proposal-thresholds", default="1500", help="Comma-separated proposal thresholds in HU")
    p.add_argument("--proposal-support-mode", choices=["depth_mask", "gaussian_residual", "ellipsoid_core", "union_depth_and_gaussian", "objectness_ridge"], default="depth_mask")
    p.add_argument("--proposal-min-metal-depth-mm", type=float, default=10.0)
    p.add_argument("--proposal-target-candidate-points", type=int, default=0)
    p.add_argument("--proposal-ellipsoid-candidate-mode", choices=["metal_in_head", "depth_mask"], default="metal_in_head")
    p.add_argument("--guided-fit-mode", default="deep_anchor_v2")
    p.add_argument("--guided-roi-radius-mm", type=float, default=5.0)
    p.add_argument("--guided-max-angle-deg", type=float, default=12.0)
    p.add_argument("--guided-max-depth-shift-mm", type=float, default=6.0)
    p.add_argument("--guided-gt-root", default=None, help="Optional root containing <subject>/<subject>_shanks_guided.tsv to use as GT")
    return p


def _parse_float_list(value: str, default: list[float]) -> list[float]:
    items = [s.strip() for s in str(value).split(",") if s.strip()]
    if not items:
        return list(default)
    return [float(x) for x in items]


def default_config(args: argparse.Namespace, stage: str) -> dict[str, Any]:
    return {
        "threshold_schedule_hu": [600.0, 550.0, 500.0],
        "target_candidate_points": int(args.target_candidate_points),
        "proposal_threshold_schedule_hu": _parse_float_list(args.proposal_thresholds, [1500.0]),
        "proposal_target_candidate_points": int(args.proposal_target_candidate_points),
        "proposal_min_metal_depth_mm": float(args.proposal_min_metal_depth_mm),
        "proposal_max_metal_depth_mm": float(args.max_metal_depth_mm),
        "proposal_support_mode": str(args.proposal_support_mode),
        "proposal_ellipsoid_candidate_mode": str(args.proposal_ellipsoid_candidate_mode),
        "refinement_threshold_schedule_hu": [600.0, 550.0, 500.0],
        "refinement_target_candidate_points": int(args.target_candidate_points),
        "refinement_min_metal_depth_mm": float(args.min_metal_depth_mm),
        "refinement_max_metal_depth_mm": float(args.max_metal_depth_mm),
        "use_head_mask": True,
        "build_head_mask": True,
        "head_mask_threshold_hu": float(args.head_mask_threshold_hu),
        "head_mask_method": str(args.head_mask_method),
        "min_metal_depth_mm": float(args.min_metal_depth_mm),
        "max_metal_depth_mm": float(args.max_metal_depth_mm),
        "guided_fit_mode": str(args.guided_fit_mode),
        "guided_roi_radius_mm": float(args.guided_roi_radius_mm),
        "guided_max_angle_deg": float(args.guided_max_angle_deg),
        "guided_max_depth_shift_mm": float(args.guided_max_depth_shift_mm),
        "ablation_stage": str(stage),
        "match_start_mm": float(args.match_start_mm),
    }


def summarize_subject(*, row: dict[str, str], stage: str, pipeline_key: str, result: dict[str, Any], assignments: list[dict[str, Any]], match_summary: dict[str, Any], gt_count: int, gt_source_path: str) -> dict[str, Any]:
    diagnostics = dict(result.get("diagnostics") or {})
    counts = dict(diagnostics.get("counts") or {})
    extras = dict(diagnostics.get("extras") or {})
    matched_rows = [r for r in assignments if int(r.get("matched", 0)) == 1]
    mean_end_error = ""
    mean_start_error = ""
    mean_angle = ""
    if matched_rows:
        mean_end_error = f"{sum(float(r['end_error_mm']) for r in matched_rows) / float(len(matched_rows)):.4f}"
        mean_start_error = f"{sum(float(r['start_error_mm']) for r in matched_rows) / float(len(matched_rows)):.4f}"
        mean_angle = f"{sum(float(r['angle_deg']) for r in matched_rows) / float(len(matched_rows)):.4f}"
    pred_count = len(list(result.get("trajectories") or []))
    return {
        "subject_id": str(row["subject_id"]),
        "stage": str(stage),
        "pipeline_key": str(pipeline_key),
        "gt_shanks": int(gt_count),
        "pred_shanks": int(pred_count),
        "matched_shanks": int(match_summary["matched"]),
        "false_negative": int(match_summary["false_negative"]),
        "false_positive": int(match_summary["false_positive"]),
        "recall": 0.0 if gt_count <= 0 else float(match_summary["matched"]) / float(gt_count),
        "precision": 0.0 if pred_count <= 0 else float(match_summary["matched"]) / float(pred_count),
        "exact_count": int(pred_count == gt_count),
        "mean_match_end_error_mm": mean_end_error,
        "mean_match_start_error_mm": mean_start_error,
        "mean_match_angle_deg": mean_angle,
        "candidate_points_total": int(counts.get("candidate_points_total", 0)),
        "candidate_points_after_depth": int(counts.get("candidate_points_after_depth", 0)),
        "blob_count_total": int(counts.get("blob_count_total", 0)),
        "token_count_total": int(counts.get("token_count_total", 0)),
        "linelet_count": int(counts.get("linelet_count", 0)),
        "seed_count": int(counts.get("seed_count", 0)),
        "proposal_count_before_nms": int(counts.get("proposal_count_before_nms", 0)),
        "hypothesis_count_generated": int(counts.get("hypothesis_count_generated", 0)),
        "selected_count": int(counts.get("selected_count", 0)),
        "conflict_edge_count": int(counts.get("conflict_edge_count", 0)),
        "chosen_threshold_hu": "" if extras.get("chosen_threshold_hu") in (None, "") else f"{float(extras.get('chosen_threshold_hu')):.1f}",
        "runtime_total_ms": f"{float(dict(diagnostics.get('timing') or {}).get('total_ms', 0.0)):.2f}",
        "ct_path": str(row["ct_path"]),
        "labels_path": str(row["labels_path"]),
        "gt_source_path": str(gt_source_path),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stages = [s.strip() for s in str(args.stages).split(",") if s.strip()] or list(DEFAULT_STAGES)
    subjects_filter = {s.strip() for s in str(args.subjects).split(",") if s.strip()} or None
    registry = PipelineRegistry()
    register_builtin_pipelines(registry)
    extras: dict[str, Any] = {}
    if bool(args.use_model_priors):
        extras["models_by_id"] = model_map(load_electrode_library(args.electrode_library))

    subject_rows = iter_subject_rows(dataset_root, subjects_filter)
    subject_metrics: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []

    for stage in stages:
        stage_dir = out_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_subject_metrics: list[dict[str, Any]] = []
        stage_assignment_rows: list[dict[str, Any]] = []
        for row in subject_rows:
            subject_id = str(row["subject_id"])
            print(f"[benchmark] stage={stage} subject={subject_id}")
            gt_shanks, gt_source_path = load_reference_ground_truth_shanks(row, args.guided_gt_root)
            config = default_config(args, stage)
            ctx, _img = build_detection_context(row["ct_path"], run_id=f"bench_{subject_id}_{stage}", config=config, extras=dict(extras))
            result = registry.run(str(args.pipeline_key), ctx)
            if str(result.get("status", "ok")).lower() == "error":
                err = dict(result.get("error") or {})
                raise RuntimeError(f"{subject_id}:{stage}: {err.get('message', 'Detection failed')} (stage={err.get('stage', 'pipeline')})")
            pred_shanks = predicted_shanks_from_result(result)
            assignments, match_summary = match_shanks(
                gt_shanks,
                pred_shanks,
                match_distance_mm=float(args.match_distance_mm),
                match_angle_deg=float(args.match_angle_deg),
                match_start_mm=float(args.match_start_mm),
            )
            for assignment in assignments:
                assignment["stage"] = str(stage)
            subject_summary = summarize_subject(
                row=row,
                stage=stage,
                pipeline_key=str(args.pipeline_key),
                result=result,
                assignments=assignments,
                match_summary=match_summary,
                gt_count=len(gt_shanks),
                gt_source_path=gt_source_path,
            )
            stage_subject_metrics.append(subject_summary)
            stage_assignment_rows.extend(assignments)
            with open(stage_dir / f"{subject_id}_details.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "subject_summary": subject_summary,
                        "assignments": assignments,
                        "result": result,
                        "ground_truth_shanks": [gt.__dict__ for gt in gt_shanks],
                        "predicted_shanks": [pred.__dict__ for pred in pred_shanks],
                    },
                    f,
                    indent=2,
                )
        subject_metrics.extend(stage_subject_metrics)
        assignment_rows.extend(stage_assignment_rows)
        gt_total = int(sum(int(r["gt_shanks"]) for r in stage_subject_metrics))
        pred_total = int(sum(int(r["pred_shanks"]) for r in stage_subject_metrics))
        matched_total = int(sum(int(r["matched_shanks"]) for r in stage_subject_metrics))
        exact_count_subjects = int(sum(int(r["exact_count"]) for r in stage_subject_metrics))
        cohort_rows.append(
            {
                "stage": str(stage),
                "subject_count": int(len(stage_subject_metrics)),
                "gt_shanks": gt_total,
                "pred_shanks": pred_total,
                "matched_shanks": matched_total,
                "recall": 0.0 if gt_total <= 0 else float(matched_total) / float(gt_total),
                "precision": 0.0 if pred_total <= 0 else float(matched_total) / float(pred_total),
                "exact_count_subjects": exact_count_subjects,
                "exact_count_rate": 0.0 if not stage_subject_metrics else float(exact_count_subjects) / float(len(stage_subject_metrics)),
            }
        )
        subject_fields = [
            "subject_id",
            "stage",
            "pipeline_key",
            "gt_shanks",
            "pred_shanks",
            "matched_shanks",
            "false_negative",
            "false_positive",
            "recall",
            "precision",
            "exact_count",
            "mean_match_end_error_mm",
            "mean_match_start_error_mm",
            "mean_match_angle_deg",
            "candidate_points_total",
            "candidate_points_after_depth",
            "blob_count_total",
            "token_count_total",
            "linelet_count",
            "seed_count",
            "proposal_count_before_nms",
            "hypothesis_count_generated",
            "selected_count",
            "conflict_edge_count",
            "chosen_threshold_hu",
            "runtime_total_ms",
            "ct_path",
            "labels_path",
            "gt_source_path",
        ]
        assignment_fields = [
            "subject_id",
            "stage",
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
        _write_tsv(stage_dir / "subject_metrics.tsv", stage_subject_metrics, subject_fields)
        _write_tsv(stage_dir / "shank_assignments.tsv", stage_assignment_rows, assignment_fields)

    if subject_metrics:
        _write_tsv(out_dir / "subject_metrics.tsv", subject_metrics, subject_fields)
    if assignment_rows:
        _write_tsv(out_dir / "shank_assignments.tsv", assignment_rows, assignment_fields)
    if cohort_rows:
        _write_tsv(out_dir / "cohort_metrics.tsv", cohort_rows, list(cohort_rows[0].keys()))
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump({"pipeline_key": str(args.pipeline_key), "stages": cohort_rows}, f, indent=2)
    print(f"[benchmark] wrote {out_dir}")


if __name__ == "__main__":
    main()
