#!/usr/bin/env python3
"""Focused headless debugger for one guided-fit shank."""

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
    _cluster_points_radius,
    _summarize_clusters,
    fit_electrode_axis_and_tip,
)
from rosa_core.electrode_models import load_electrode_library  # noqa: E402
from shank_core.io import image_ijk_ras_matrices, kji_to_ras_points_matrix, ras_to_ijk_float_matrix, read_volume  # noqa: E402
from shank_core.masking import build_preview_masks  # noqa: E402
from tools.refine_shank_ground_truth_guided import (  # noqa: E402
    _choose_model,
    _lps_to_ras,
    _orient_entry_target,
    _ras_to_lps,
    read_tsv_rows,
)


def _extract_threshold_candidates_lps(arr_kji: np.ndarray, ijk_to_ras: np.ndarray, threshold: float) -> np.ndarray:
    idx = np.argwhere(np.asarray(arr_kji, dtype=float) >= float(threshold))
    if idx.size == 0:
        return np.empty((0, 3), dtype=float)
    ras = np.asarray(kji_to_ras_points_matrix(idx.astype(float), ijk_to_ras), dtype=float).reshape(-1, 3)
    lps = ras.copy()
    lps[:, 0] *= -1.0
    lps[:, 1] *= -1.0
    return lps


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--shank", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fit-mode", default="deep_anchor_v2", choices=["slab_v1", "deep_anchor_v2", "em_v1", "em_v2"])
    parser.add_argument("--threshold-hu", type=float, default=1800.0)
    parser.add_argument("--roi-radius-mm", type=float, default=5.0)
    parser.add_argument("--max-angle-deg", type=float, default=12.0)
    parser.add_argument("--max-depth-shift-mm", type=float, default=2.0)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    subject_rows = {r["subject_id"]: r for r in read_tsv_rows(dataset_root / "contact_label_dataset" / "subjects.tsv")}
    shank_rows = [
        r for r in read_tsv_rows(dataset_root / "contact_label_dataset" / "all_shanks.tsv")
        if r["subject_id"] == args.subject and r["shank"] == args.shank
    ]
    if not shank_rows:
        raise SystemExit(f"Missing shank {args.subject}/{args.shank}")
    row = shank_rows[0]
    subject_row = subject_rows[args.subject]

    img, arr_kji, spacing_xyz = read_volume(subject_row["ct_path"])
    ijk_to_ras, ras_to_ijk = image_ijk_ras_matrices(img)
    ras_to_ijk_fn = lambda ras_xyz: ras_to_ijk_float_matrix(ras_xyz, ras_to_ijk)
    preview = build_preview_masks(
        arr_kji=np.asarray(arr_kji, dtype=float),
        spacing_xyz=tuple(spacing_xyz),
        threshold=float(args.threshold_hu),
        use_head_mask=True,
        build_head_mask=True,
        head_mask_threshold_hu=150.0,
        head_mask_method="otsu",
        head_gate_erode_vox=1,
        head_gate_dilate_vox=1,
        head_gate_margin_mm=0.0,
        min_metal_depth_mm=0.0,
        max_metal_depth_mm=12.0,
        include_debug_masks=False,
    )
    depth_map_kji = np.asarray(preview.get("head_distance_map_kji"), dtype=np.float32) if preview.get("head_distance_map_kji") is not None else None

    start_ras = np.asarray([float(row["start_x"]), float(row["start_y"]), float(row["start_z"])], dtype=float)
    end_ras = np.asarray([float(row["end_x"]), float(row["end_y"]), float(row["end_z"])], dtype=float)
    entry_ras, target_ras = _orient_entry_target(start_ras, end_ras, depth_map_kji, ras_to_ijk_fn)

    models = list((load_electrode_library().get("models") or []))
    model = _choose_model(
        models,
        contact_count=int(row.get("contact_count") or 0),
        mean_intercontact_mm=float(row.get("mean_intercontact_mm") or 0.0),
        span_mm=float(row.get("span_mm") or np.linalg.norm(target_ras - entry_ras)),
    )
    offsets = list(model.get("contact_center_offsets_from_tip_mm") or []) if model is not None else []

    candidate_points_lps = _extract_threshold_candidates_lps(arr_kji, ijk_to_ras, threshold=float(args.threshold_hu))
    fit = fit_electrode_axis_and_tip(
        candidate_points_lps=candidate_points_lps,
        planned_entry_lps=_ras_to_lps(entry_ras),
        planned_target_lps=_ras_to_lps(target_ras),
        contact_offsets_mm=offsets,
        tip_at="target",
        roi_radius_mm=float(args.roi_radius_mm),
        max_angle_deg=float(args.max_angle_deg),
        max_depth_shift_mm=float(args.max_depth_shift_mm),
        fit_mode=str(args.fit_mode),
    )

    roi_pts = candidate_points_lps
    if fit.get("success"):
        center = np.asarray(fit["center_lps"], dtype=float)
        axis = np.asarray(fit["deep_axis_lps"], dtype=float)
    else:
        center = _ras_to_lps(0.5 * (entry_ras + target_ras))
        axis = _ras_to_lps(target_ras) - _ras_to_lps(entry_ras)
    clusters = _cluster_points_radius(np.asarray(candidate_points_lps, dtype=float), radius_mm=1.6, min_cluster_size=6)
    summaries = _summarize_clusters(np.asarray(candidate_points_lps, dtype=float), clusters, center, axis / np.linalg.norm(axis))

    out_dir = Path(args.out_dir)
    _write_json(
        out_dir / f"{args.subject}_{args.shank}_{args.fit_mode}_fit.json",
        {
            "subject": args.subject,
            "shank": args.shank,
            "fit_mode": args.fit_mode,
            "planned_entry_ras": entry_ras.tolist(),
            "planned_target_ras": target_ras.tolist(),
            "fit": fit,
            "model_id": "" if model is None else model.get("id"),
        },
    )
    _write_json(
        out_dir / f"{args.subject}_{args.shank}_{args.fit_mode}_clusters.json",
        [
            {
                "cluster_id": int(item["cluster_id"]),
                "count": int(item["count"]),
                "span_mm": float(item["span_mm"]),
                "radial_rms_mm": float(item["radial_rms_mm"]),
                "t_planned_mm": float(item["t_planned_mm"]),
                "center_lps": np.asarray(item["center"], dtype=float).tolist(),
            }
            for item in summaries
        ],
    )


if __name__ == "__main__":
    main()
