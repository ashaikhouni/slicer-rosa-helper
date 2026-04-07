#!/usr/bin/env python3
"""Analyze multi-threshold blob lineages and role scores for SEEG CTs."""

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

from shank_core.io import image_ijk_ras_matrices, read_volume  # noqa: E402
from shank_core.masking import build_preview_masks  # noqa: E402
from shank_engine.lineage_tracking import extract_threshold_levels, build_lineages, summarize_lineages  # noqa: E402


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


def _load_subject_manifest(dataset_root: Path, subject_id: str) -> dict[str, str]:
    rows = _read_tsv_rows(dataset_root / "contact_label_dataset" / "subjects.tsv")
    for row in rows:
        if str(row["subject_id"]) == str(subject_id):
            return row
    raise KeyError(f"subject not found: {subject_id}")


def analyze_subject(subject_id: str, dataset_root: Path, thresholds: list[float], cfg: dict[str, Any]) -> dict[str, Any]:
    manifest = _load_subject_manifest(dataset_root, subject_id)
    ct_path = str(manifest["ct_path"])

    image, arr_kji, spacing_xyz = read_volume(ct_path)
    _ijk_to_ras, _ras_to_ijk = image_ijk_ras_matrices(image)
    preview = build_preview_masks(
        arr_kji=np.asarray(arr_kji),
        spacing_xyz=tuple(spacing_xyz),
        threshold=float(thresholds[len(thresholds) // 2]),
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
    levels = extract_threshold_levels(
        arr_kji=np.asarray(arr_kji),
        gating_mask_kji=np.asarray(preview.get("gating_mask_kji"), dtype=bool),
        depth_map_kji=np.asarray(preview.get("head_distance_map_kji"), dtype=np.float32) if preview.get("head_distance_map_kji") is not None else None,
        thresholds_hu=thresholds,
        ijk_kji_to_ras_fn=lambda pts: __import__("shank_core.io", fromlist=["kji_to_ras_points_matrix"]).kji_to_ras_points_matrix(pts, _ijk_to_ras),
    )
    lineages = build_lineages(levels)
    summaries = summarize_lineages(lineages, total_levels=len(levels))
    level_summaries = [{"threshold_hu": float(level["threshold_hu"]), "blob_count": int(len(level["blobs"]))} for level in levels]
    return {
        "subject_id": subject_id,
        "ct_path": ct_path,
        "thresholds_hu": [float(v) for v in thresholds],
        "level_summaries": level_summaries,
        "lineages": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze multi-threshold blob persistence for SEEG CTs")
    parser.add_argument("--dataset-root", required=True, help="Path to seeg_localization root")
    parser.add_argument("--subjects", required=True, help="Comma-separated subject ids")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--thresholds-hu", default="2600,2200,1800,1500,1200", help="Descending comma-separated thresholds")
    parser.add_argument("--head-threshold-hu", type=float, default=-500.0)
    parser.add_argument("--head-mask-method", default="outside_air")
    parser.add_argument("--min-metal-depth-mm", type=float, default=5.0)
    parser.add_argument("--max-metal-depth-mm", type=float, default=220.0)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = sorted([float(v.strip()) for v in str(args.thresholds_hu).split(",") if v.strip()], reverse=True)
    cfg = {
        "head_threshold_hu": float(args.head_threshold_hu),
        "head_mask_method": str(args.head_mask_method),
        "min_metal_depth_mm": float(args.min_metal_depth_mm),
        "max_metal_depth_mm": float(args.max_metal_depth_mm),
    }

    blob_count_rows = []
    for subject_id in [s.strip() for s in str(args.subjects).split(",") if s.strip()]:
        data = analyze_subject(subject_id, dataset_root, thresholds, cfg)
        subj_dir = out_dir / subject_id
        subj_dir.mkdir(parents=True, exist_ok=True)
        (subj_dir / "persistence.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        if data["lineages"]:
            _write_tsv(subj_dir / "lineages.tsv", data["lineages"], list(data["lineages"][0].keys()))
        for level in data["level_summaries"]:
            blob_count_rows.append({"subject_id": subject_id, "threshold_hu": float(level["threshold_hu"]), "blob_count": int(level["blob_count"])})
        top_core = sorted(data["lineages"], key=lambda row: float(row["support_priority"]), reverse=True)[:5]
        print(
            f"[persistence] {subject_id}: levels={[x['blob_count'] for x in data['level_summaries']]}, "
            f"top_core={[{'id': int(x['lineage_id']), 'p_core': round(float(x['p_core']), 3), 'p_junction': round(float(x['p_junction']), 3)} for x in top_core]}"
        )
    if blob_count_rows:
        _write_tsv(out_dir / "blob_counts.tsv", blob_count_rows, list(blob_count_rows[0].keys()))
    print(f"[persistence] wrote {out_dir}")


if __name__ == "__main__":
    main()
