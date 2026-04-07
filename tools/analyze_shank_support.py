#!/usr/bin/env python3
"""Diagnose why gated CT support fails to recover ground-truth SEEG shanks.

For each cached ground-truth shank, this tool measures support in two masks:
- permissive in-head metal (`metal_in_head`)
- depth-gated metal (`metal_depth_pass`)

The output is diagnostic rather than evaluative. It answers:
- is support near the true shank present at all?
- did depth gating remove it?
- is support fragmented along the shaft?
- is the shallow or deep endpoint unsupported?
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIB_DIR = os.path.join(REPO_ROOT, "CommonLib")
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

from shank_core.io import image_ijk_ras_matrices, kji_to_ras_points_matrix, ras_to_ijk_float_matrix, read_volume  # noqa: E402
from shank_core.masking import build_preview_masks  # noqa: E402


def _read_tsv_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _write_tsv_rows(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return vec / norm


def _load_subject_manifest(dataset_root: Path, subject_id: str) -> dict[str, str]:
    rows = _read_tsv_rows(dataset_root / "contact_label_dataset" / "subjects.tsv")
    for row in rows:
        if str(row["subject_id"]) == str(subject_id):
            return row
    raise KeyError(f"subject not found in manifest: {subject_id}")


def _load_gt_shanks(shanks_path: str | Path) -> list[dict[str, Any]]:
    rows = _read_tsv_rows(shanks_path)
    out = []
    for row in rows:
        out.append(
            {
                "subject_id": str(row["subject_id"]),
                "shank": str(row["shank"]),
                "contact_count": int(row["contact_count"]),
                "start_ras": np.array([float(row["start_x"]), float(row["start_y"]), float(row["start_z"])], dtype=float),
                "end_ras": np.array([float(row["end_x"]), float(row["end_y"]), float(row["end_z"])], dtype=float),
                "dir_ras": _unit(np.array([float(row["dir_x"]), float(row["dir_y"]), float(row["dir_z"])], dtype=float)),
                "span_mm": float(row["span_mm"]),
            }
        )
    return out


def _mask_points_ras(mask_kji: np.ndarray | None, ijk_to_ras: np.ndarray) -> np.ndarray:
    if mask_kji is None:
        return np.zeros((0, 3), dtype=float)
    idx = np.argwhere(np.asarray(mask_kji, dtype=bool))
    if idx.size == 0:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(kji_to_ras_points_matrix(idx.astype(float), ijk_to_ras), dtype=float).reshape(-1, 3)


def _line_support_metrics(
    points_ras: np.ndarray,
    start_ras: np.ndarray,
    end_ras: np.ndarray,
    tube_radius_mm: float,
    endpoint_radius_mm: float,
    bins: int,
    extension_mm: float,
) -> dict[str, Any]:
    if points_ras.shape[0] == 0:
        return {
            "tube_count": 0,
            "coverage_ratio": 0.0,
            "largest_gap_mm": float("inf"),
            "start_near_count": 0,
            "end_near_count": 0,
            "residual_mean_mm": float("inf"),
            "residual_p95_mm": float("inf"),
        }
    axis = _unit(end_ras - start_ras)
    rel = points_ras - start_ras.reshape(1, 3)
    proj = rel @ axis.reshape(3)
    span = float(np.linalg.norm(end_ras - start_ras))
    min_t = -float(extension_mm)
    max_t = span + float(extension_mm)
    closest = start_ras.reshape(1, 3) + np.outer(proj, axis.reshape(3))
    resid = np.linalg.norm(points_ras - closest, axis=1)
    on_axis = (proj >= min_t) & (proj <= max_t)
    in_tube = on_axis & (resid <= float(tube_radius_mm))
    tube_proj = proj[in_tube]
    tube_resid = resid[in_tube]
    if tube_proj.size:
        hist, _ = np.histogram(tube_proj, bins=max(1, int(bins)), range=(0.0, max(1e-6, span)))
        coverage_ratio = float(np.count_nonzero(hist > 0)) / float(max(1, len(hist)))
        ordered = np.sort(np.clip(tube_proj, 0.0, span))
        full = np.concatenate([[0.0], ordered, [span]])
        largest_gap = float(np.max(np.diff(full))) if full.size >= 2 else float("inf")
        residual_mean = float(np.mean(tube_resid))
        residual_p95 = float(np.quantile(tube_resid, 0.95))
    else:
        coverage_ratio = 0.0
        largest_gap = float("inf")
        residual_mean = float("inf")
        residual_p95 = float("inf")
    start_near = int(np.count_nonzero(np.linalg.norm(points_ras - start_ras.reshape(1, 3), axis=1) <= float(endpoint_radius_mm)))
    end_near = int(np.count_nonzero(np.linalg.norm(points_ras - end_ras.reshape(1, 3), axis=1) <= float(endpoint_radius_mm)))
    return {
        "tube_count": int(np.count_nonzero(in_tube)),
        "coverage_ratio": coverage_ratio,
        "largest_gap_mm": largest_gap,
        "start_near_count": start_near,
        "end_near_count": end_near,
        "residual_mean_mm": residual_mean,
        "residual_p95_mm": residual_p95,
    }


def _classify_failure(metal_in_head: dict[str, Any], depth_pass: dict[str, Any], cfg: dict[str, Any]) -> str:
    min_support = int(cfg["min_support_points"])
    if int(metal_in_head["tube_count"]) < min_support:
        return "insufficient_in_head_support"
    if int(depth_pass["tube_count"]) < min_support and int(metal_in_head["tube_count"]) >= min_support:
        return "depth_gate_removed_support"
    if int(depth_pass["end_near_count"]) == 0:
        return "deep_endpoint_missing"
    if int(depth_pass["start_near_count"]) == 0:
        return "shallow_endpoint_missing"
    if float(depth_pass["coverage_ratio"]) < float(cfg["min_coverage_ratio"]):
        return "fragmented_axis_support"
    if float(depth_pass["largest_gap_mm"]) > float(cfg["max_gap_mm"]):
        return "large_axis_gap"
    return "support_present_fitting_failed"


def analyze_subject(subject_id: str, dataset_root: Path, cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = _load_subject_manifest(dataset_root, subject_id)
    ct_path = str(manifest["ct_path"])
    shanks_path = str(manifest["shanks_path"])
    gt_shanks = _load_gt_shanks(shanks_path)

    image, arr_kji, spacing_xyz = read_volume(ct_path)
    ijk_to_ras, ras_to_ijk = image_ijk_ras_matrices(image)
    preview = build_preview_masks(
        arr_kji=np.asarray(arr_kji),
        spacing_xyz=tuple(spacing_xyz),
        threshold=float(cfg["metal_threshold_hu"]),
        use_head_mask=True,
        build_head_mask=True,
        head_mask_threshold_hu=float(cfg["head_threshold_hu"]),
        head_mask_aggressive_cleanup=True,
        head_mask_close_mm=2.0,
        head_mask_method=str(cfg["head_mask_method"]),
        head_mask_metal_dilate_mm=1.0,
        head_gate_erode_vox=1,
        head_gate_dilate_vox=1,
        head_gate_margin_mm=0.0,
        min_metal_depth_mm=float(cfg["min_metal_depth_mm"]),
        max_metal_depth_mm=float(cfg["max_metal_depth_mm"]),
        include_debug_masks=False,
    )

    pts_in_head = _mask_points_ras(preview.get("metal_in_head_mask_kji"), ijk_to_ras)
    pts_depth = _mask_points_ras(preview.get("metal_depth_pass_mask_kji"), ijk_to_ras)

    rows: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    for shank in gt_shanks:
        metrics_head = _line_support_metrics(
            pts_in_head,
            shank["start_ras"],
            shank["end_ras"],
            float(cfg["tube_radius_mm"]),
            float(cfg["endpoint_radius_mm"]),
            int(cfg["coverage_bins"]),
            float(cfg["projection_extension_mm"]),
        )
        metrics_depth = _line_support_metrics(
            pts_depth,
            shank["start_ras"],
            shank["end_ras"],
            float(cfg["tube_radius_mm"]),
            float(cfg["endpoint_radius_mm"]),
            int(cfg["coverage_bins"]),
            float(cfg["projection_extension_mm"]),
        )
        reason = _classify_failure(metrics_head, metrics_depth, cfg)
        reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
        row = {
            "subject_id": subject_id,
            "shank": shank["shank"],
            "span_mm": f"{float(shank['span_mm']):.3f}",
            "metal_in_head_tube_count": int(metrics_head["tube_count"]),
            "metal_in_head_coverage_ratio": f"{float(metrics_head['coverage_ratio']):.4f}",
            "metal_in_head_largest_gap_mm": "" if not np.isfinite(metrics_head["largest_gap_mm"]) else f"{float(metrics_head['largest_gap_mm']):.3f}",
            "metal_in_head_start_near_count": int(metrics_head["start_near_count"]),
            "metal_in_head_end_near_count": int(metrics_head["end_near_count"]),
            "depth_pass_tube_count": int(metrics_depth["tube_count"]),
            "depth_pass_coverage_ratio": f"{float(metrics_depth['coverage_ratio']):.4f}",
            "depth_pass_largest_gap_mm": "" if not np.isfinite(metrics_depth["largest_gap_mm"]) else f"{float(metrics_depth['largest_gap_mm']):.3f}",
            "depth_pass_start_near_count": int(metrics_depth["start_near_count"]),
            "depth_pass_end_near_count": int(metrics_depth["end_near_count"]),
            "depth_pass_residual_mean_mm": "" if not np.isfinite(metrics_depth["residual_mean_mm"]) else f"{float(metrics_depth['residual_mean_mm']):.3f}",
            "depth_pass_residual_p95_mm": "" if not np.isfinite(metrics_depth["residual_p95_mm"]) else f"{float(metrics_depth['residual_p95_mm']):.3f}",
            "failure_reason": reason,
        }
        rows.append(row)

    summary = {
        "subject_id": subject_id,
        "ct_path": ct_path,
        "shanks_path": shanks_path,
        "n_shanks": len(gt_shanks),
        "mask_counts": {
            "metal_in_head_count": int(preview.get("metal_in_head_count", 0)),
            "depth_kept_count": int(preview.get("depth_kept_count", 0)),
        },
        "reason_counts": reason_counts,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose GT shank support coverage in gated CT masks")
    parser.add_argument("--dataset-root", required=True, help="Path to seeg_localization root")
    parser.add_argument("--subjects", required=True, help="Comma-separated subject ids")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--metal-threshold-hu", type=float, default=1800.0)
    parser.add_argument("--head-threshold-hu", type=float, default=-500.0)
    parser.add_argument("--head-mask-method", default="outside_air")
    parser.add_argument("--min-metal-depth-mm", type=float, default=5.0)
    parser.add_argument("--max-metal-depth-mm", type=float, default=220.0)
    parser.add_argument("--tube-radius-mm", type=float, default=1.5)
    parser.add_argument("--endpoint-radius-mm", type=float, default=3.0)
    parser.add_argument("--coverage-bins", type=int, default=8)
    parser.add_argument("--projection-extension-mm", type=float, default=2.0)
    parser.add_argument("--min-support-points", type=int, default=10)
    parser.add_argument("--min-coverage-ratio", type=float, default=0.35)
    parser.add_argument("--max-gap-mm", type=float, default=12.0)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "metal_threshold_hu": float(args.metal_threshold_hu),
        "head_threshold_hu": float(args.head_threshold_hu),
        "head_mask_method": str(args.head_mask_method),
        "min_metal_depth_mm": float(args.min_metal_depth_mm),
        "max_metal_depth_mm": float(args.max_metal_depth_mm),
        "tube_radius_mm": float(args.tube_radius_mm),
        "endpoint_radius_mm": float(args.endpoint_radius_mm),
        "coverage_bins": int(args.coverage_bins),
        "projection_extension_mm": float(args.projection_extension_mm),
        "min_support_points": int(args.min_support_points),
        "min_coverage_ratio": float(args.min_coverage_ratio),
        "max_gap_mm": float(args.max_gap_mm),
    }

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for subject_id in [s.strip() for s in str(args.subjects).split(",") if s.strip()]:
        rows, summary = analyze_subject(subject_id, dataset_root, cfg)
        all_rows.extend(rows)
        summaries.append(summary)
        subject_path = out_dir / f"{subject_id}_support.tsv"
        if rows:
            _write_tsv_rows(subject_path, rows, list(rows[0].keys()))
        print(f"[support] {subject_id}: {summary['reason_counts']}")

    if all_rows:
        _write_tsv_rows(out_dir / "all_support.tsv", all_rows, list(all_rows[0].keys()))
    (out_dir / "summary.json").write_text(json.dumps({"config": cfg, "subjects": summaries}, indent=2), encoding="utf-8")
    print(f"[support] wrote {out_dir}")


if __name__ == "__main__":
    main()
