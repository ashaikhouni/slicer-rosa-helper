#!/usr/bin/env python3
"""CLI utilities for ShankDetect core processing.

Phase 1 focuses on mask-iteration speed outside Slicer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIB_DIR = os.path.join(REPO_ROOT, "ShankDetect", "Lib")
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)
ROSA_LIB_DIR = os.path.join(REPO_ROOT, "RosaHelper", "Lib")
if ROSA_LIB_DIR not in sys.path:
    sys.path.insert(0, ROSA_LIB_DIR)

from shank_core.io import (  # noqa: E402
    image_ijk_ras_matrices,
    kji_to_ras_points_matrix,
    kji_to_ras_points,
    ras_to_ijk_float_matrix,
    read_volume,
    write_mask_like,
    write_points_csv,
)
from shank_core.masking import build_preview_masks  # noqa: E402
from rosa_core.electrode_models import load_electrode_library, model_map  # noqa: E402


def _add_common_mask_args(parser):
    bool_action = argparse.BooleanOptionalAction
    parser.add_argument("--ct", required=True, help="Input CT NIfTI path")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--metal-threshold-hu", type=float, default=1800.0)
    parser.add_argument("--use-head-mask", action=bool_action, default=True)
    parser.add_argument("--build-head-mask", action=bool_action, default=True)
    parser.add_argument("--head-mask-threshold-hu", type=float, default=-500.0)
    parser.add_argument("--head-mask-close-mm", type=float, default=2.0)
    parser.add_argument(
        "--head-mask-method",
        choices=["legacy", "tissue_cut", "tissue_cut_noclose", "outside_air"],
        default="outside_air",
        help="Head mask construction method (tissue_cut_noclose skips closing for speed comparison).",
    )
    parser.add_argument(
        "--head-mask-metal-dilate-mm",
        type=float,
        default=1.0,
        help="Only for tissue_cut: dilation radius to suppress metal bridges before closing/fill.",
    )
    parser.add_argument("--min-metal-depth-mm", type=float, default=5.0)
    parser.add_argument("--max-metal-depth-mm", type=float, default=220.0)
    parser.add_argument("--head-mask-aggressive-cleanup", action=bool_action, default=True)

def _add_detection_args(parser):
    parser.add_argument("--max-points", type=int, default=300000)
    parser.add_argument("--max-lines", type=int, default=30)
    parser.add_argument("--inlier-radius-mm", type=float, default=1.2)
    parser.add_argument("--min-length-mm", type=float, default=20.0)
    parser.add_argument("--min-inliers", type=int, default=250)
    parser.add_argument("--ransac-iterations", type=int, default=240)
    bool_action = argparse.BooleanOptionalAction
    parser.add_argument("--use-model-score", action=bool_action, default=True)
    parser.add_argument("--min-model-score", type=float, default=0.10)
    parser.add_argument(
        "--electrode-library",
        default=None,
        help="Optional path to electrode library JSON (defaults to bundled DIXI D08 library).",
    )


def cmd_preview_masks(args):
    os.makedirs(args.out_dir, exist_ok=True)

    img, arr_kji, spacing_xyz = read_volume(args.ct)
    result = build_preview_masks(
        arr_kji=arr_kji,
        spacing_xyz=spacing_xyz,
        threshold=args.metal_threshold_hu,
        use_head_mask=args.use_head_mask,
        build_head_mask=args.build_head_mask,
        head_mask_threshold_hu=args.head_mask_threshold_hu,
        head_mask_aggressive_cleanup=args.head_mask_aggressive_cleanup,
        head_mask_close_mm=args.head_mask_close_mm,
        head_mask_method=args.head_mask_method,
        head_mask_metal_dilate_mm=args.head_mask_metal_dilate_mm,
        min_metal_depth_mm=args.min_metal_depth_mm,
        max_metal_depth_mm=args.max_metal_depth_mm,
    )

    masks_dir = os.path.join(args.out_dir, "masks")
    points_dir = os.path.join(args.out_dir, "points")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(points_dir, exist_ok=True)

    write_mask_like(img, result["metal_mask_kji"], os.path.join(masks_dir, "metal_mask.nii.gz"))
    gating = result.get("gating_mask_kji", result.get("head_mask_kji"))
    if gating is not None:
        write_mask_like(img, gating, os.path.join(masks_dir, "gating_mask.nii.gz"))
    ijk_kji = result.get("in_mask_ijk_kji")
    if ijk_kji is None:
        ijk_kji = []
    ras = kji_to_ras_points(img, ijk_kji)
    write_points_csv(
        os.path.join(points_dir, "in_mask_points_kji.csv"),
        [[p[0], p[1], p[2]] for p in ijk_kji],
        columns=("k", "j", "i"),
    )
    write_points_csv(os.path.join(points_dir, "in_mask_points_ras.csv"), ras, columns=("x", "y", "z"))

    summary = {
        "ct": os.path.abspath(args.ct),
        "candidate_count": int(result.get("candidate_count", 0)),
        "in_mask_count": int(result.get("head_mask_kept_count", 0)),
        "metal_in_head_count": int(result.get("metal_in_head_count", 0)),
        "depth_kept_count": int(result.get("depth_kept_count", 0)),
        "gating_mask_type": str(result.get("gating_mask_type", "none")),
        "inside_method": str(result.get("inside_method", "none")),
        "profile_ms": result.get("profile_ms", {}),
        "profile_flags": result.get("profile_flags", {}),
        "head_mask_method": args.head_mask_method,
        "head_mask_metal_dilate_mm": float(args.head_mask_metal_dilate_mm),
        "outputs": {
            "metal_mask": os.path.join(masks_dir, "metal_mask.nii.gz"),
            "gating_mask": os.path.join(masks_dir, "gating_mask.nii.gz"),
            "in_mask_points_kji": os.path.join(points_dir, "in_mask_points_kji.csv"),
            "in_mask_points_ras": os.path.join(points_dir, "in_mask_points_ras.csv"),
        },
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        "[preview-masks] "
        f"candidates={summary['candidate_count']} in-head={summary['metal_in_head_count']} "
        f"in-mask={summary['in_mask_count']} mask={summary['gating_mask_type']} "
        f"inside_method={summary['inside_method']}"
    )
    prof = summary.get("profile_ms", {}) or {}
    if prof:
        print(
            "[preview-masks] profile(ms) "
            f"total={float(prof.get('total', 0.0)):.1f} threshold={float(prof.get('threshold', 0.0)):.1f} "
            f"head_mask={float(prof.get('head_mask', 0.0)):.1f} distance={float(prof.get('distance_map', 0.0)):.1f} "
            f"enum={float(prof.get('candidate_enum', 0.0)):.1f} gate={float(prof.get('head_gate', 0.0)):.1f} "
            f"depth={float(prof.get('depth_gate', 0.0)):.1f}"
        )
    flags = summary.get("profile_flags", {}) or {}
    if flags:
        print(
            "[preview-masks] cache "
            f"gating={bool(flags.get('used_precomputed_gating_mask', False))} "
            f"distance={bool(flags.get('used_precomputed_distance_map', False))}"
        )
    print(f"[preview-masks] wrote outputs under {args.out_dir}")


def cmd_detect(args):
    os.makedirs(args.out_dir, exist_ok=True)
    from shank_core.pipeline import run_detection  # local import: only needed for detect command

    img, arr_kji, spacing_xyz = read_volume(args.ct)
    ijk_to_ras, ras_to_ijk = image_ijk_ras_matrices(img)
    size_xyz = img.GetSize()
    center_ijk = [0.5 * (float(size_xyz[0]) - 1.0), 0.5 * (float(size_xyz[1]) - 1.0), 0.5 * (float(size_xyz[2]) - 1.0)]
    center_ras = kji_to_ras_points_matrix([[center_ijk[2], center_ijk[1], center_ijk[0]]], ijk_to_ras)[0].tolist()

    models_by_id = None
    if bool(args.use_model_score):
        lib = load_electrode_library(args.electrode_library)
        models_by_id = model_map(lib)

    result = run_detection(
        arr_kji=arr_kji,
        spacing_xyz=spacing_xyz,
        threshold=args.metal_threshold_hu,
        ijk_kji_to_ras_fn=lambda ijk_kji: kji_to_ras_points_matrix(ijk_kji, ijk_to_ras),
        ras_to_ijk_fn=lambda ras_xyz: ras_to_ijk_float_matrix(ras_xyz, ras_to_ijk),
        center_ras=center_ras,
        max_points=args.max_points,
        max_lines=args.max_lines,
        inlier_radius_mm=args.inlier_radius_mm,
        min_length_mm=args.min_length_mm,
        min_inliers=args.min_inliers,
        ransac_iterations=args.ransac_iterations,
        use_head_mask=args.use_head_mask,
        build_head_mask=args.build_head_mask,
        head_mask_threshold_hu=args.head_mask_threshold_hu,
        head_mask_aggressive_cleanup=args.head_mask_aggressive_cleanup,
        head_mask_close_mm=args.head_mask_close_mm,
        head_mask_method=args.head_mask_method,
        head_mask_metal_dilate_mm=args.head_mask_metal_dilate_mm,
        min_metal_depth_mm=args.min_metal_depth_mm,
        max_metal_depth_mm=args.max_metal_depth_mm,
        models_by_id=models_by_id,
        min_model_score=args.min_model_score if bool(args.use_model_score) else None,
    )

    masks_dir = os.path.join(args.out_dir, "masks")
    lines_dir = os.path.join(args.out_dir, "lines")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(lines_dir, exist_ok=True)

    write_mask_like(img, result["metal_mask_kji"], os.path.join(masks_dir, "metal_mask.nii.gz"))
    gating = result.get("gating_mask_kji", result.get("head_mask_kji"))
    if gating is not None:
        write_mask_like(img, gating, os.path.join(masks_dir, "gating_mask.nii.gz"))

    lines = result.get("lines", [])
    lines_json = os.path.join(lines_dir, "trajectories.json")
    with open(lines_json, "w", encoding="utf-8") as f:
        json.dump(lines, f, indent=2)

    lines_csv = os.path.join(lines_dir, "trajectories.csv")
    with open(lines_csv, "w", encoding="utf-8") as f:
        f.write("name,start_x,start_y,start_z,end_x,end_y,end_z,length_mm,inlier_count,rms_mm,inside_fraction\n")
        for idx, line in enumerate(lines, start=1):
            name = f"T{idx:02d}"
            s = line.get("start_ras", [0.0, 0.0, 0.0])
            e = line.get("end_ras", [0.0, 0.0, 0.0])
            f.write(
                f"{name},{float(s[0]):.6f},{float(s[1]):.6f},{float(s[2]):.6f},"
                f"{float(e[0]):.6f},{float(e[1]):.6f},{float(e[2]):.6f},"
                f"{float(line.get('length_mm', 0.0)):.6f},{int(line.get('inlier_count', 0))},"
                f"{float(line.get('rms_mm', 0.0)):.6f},{float(line.get('inside_fraction', 0.0)):.6f}\n"
            )

    summary = {
        "ct": os.path.abspath(args.ct),
        "candidate_count": int(result.get("candidate_count", 0)),
        "in_mask_count": int(result.get("head_mask_kept_count", 0)),
        "line_count": len(lines),
        "use_model_score": bool(args.use_model_score),
        "min_model_score": float(args.min_model_score),
        "head_mask_method": args.head_mask_method,
        "head_mask_metal_dilate_mm": float(args.head_mask_metal_dilate_mm),
        "profile_ms": result.get("profile_ms", {}),
        "profile_flags": result.get("profile_flags", {}),
        "outputs": {
            "metal_mask": os.path.join(masks_dir, "metal_mask.nii.gz"),
            "gating_mask": os.path.join(masks_dir, "gating_mask.nii.gz"),
            "trajectories_json": lines_json,
            "trajectories_csv": lines_csv,
        },
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        "[detect] "
        f"candidates={summary['candidate_count']} in-mask={summary['in_mask_count']} "
        f"lines={summary['line_count']}"
    )
    prof = summary.get("profile_ms", {}) or {}
    if prof:
        p_prev = prof.get("preview", {}) if isinstance(prof.get("preview"), dict) else {}
        p_det = prof.get("detect", {}) if isinstance(prof.get("detect"), dict) else {}
        print(
            "[detect] profile(ms) "
            f"total={float(prof.get('total', 0.0)):.1f} "
            f"preview_stage={float(prof.get('preview_stage', 0.0)):.1f} "
            f"detect_stage={float(prof.get('detect_stage', 0.0)):.1f}"
        )
        print(
            "[detect] profile detail(ms) "
            f"head_mask={float(p_prev.get('head_mask', 0.0)):.1f} "
            f"distance={float(p_prev.get('distance_map', 0.0)):.1f} "
            f"first_pass={float(p_det.get('first_pass', 0.0)):.1f} "
            f"refine={float(p_det.get('refine', 0.0)):.1f}"
        )
    flags = summary.get("profile_flags", {}) or {}
    if flags:
        print(
            "[detect] cache "
            f"gating={bool(flags.get('used_precomputed_gating_mask', False))} "
            f"distance={bool(flags.get('used_precomputed_distance_map', False))} "
            f"head_cache_hit={bool(flags.get('head_cache_hit', False))}"
        )
    print(f"[detect] wrote outputs under {args.out_dir}")


def main():
    parser = argparse.ArgumentParser(description="ShankDetect CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prev = sub.add_parser("preview-masks", help="Build and export raw masks from CT")
    _add_common_mask_args(p_prev)
    p_prev.set_defaults(func=cmd_preview_masks)

    p_detect = sub.add_parser("detect", help="Run full trajectory detection from CT")
    _add_common_mask_args(p_detect)
    _add_detection_args(p_detect)
    p_detect.set_defaults(func=cmd_detect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
