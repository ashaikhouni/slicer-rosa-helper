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

from shank_core.io import (  # noqa: E402
    kji_to_ras_points,
    read_volume,
    write_mask_like,
    write_points_csv,
)
from shank_core.masking import build_preview_masks  # noqa: E402


def _add_common_mask_args(parser):
    bool_action = argparse.BooleanOptionalAction
    parser.add_argument("--ct", required=True, help="Input CT NIfTI path")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--metal-threshold-hu", type=float, default=1800.0)
    parser.add_argument("--use-head-mask", action=bool_action, default=True)
    parser.add_argument("--build-head-mask", action=bool_action, default=True)
    parser.add_argument("--head-mask-threshold-hu", type=float, default=-300.0)
    parser.add_argument("--head-mask-close-mm", type=float, default=2.0)
    parser.add_argument("--min-metal-depth-mm", type=float, default=0.0)
    parser.add_argument("--max-metal-depth-mm", type=float, default=200.0)
    parser.add_argument("--head-mask-aggressive-cleanup", action=bool_action, default=True)
    parser.add_argument(
        "--debug-soft-outputs",
        action=bool_action,
        default=False,
        help="Reserved debug switch (no extra outputs in head-distance mode).",
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
        min_metal_depth_mm=args.min_metal_depth_mm,
        max_metal_depth_mm=args.max_metal_depth_mm,
        debug_soft_outputs=args.debug_soft_outputs,
    )

    masks_dir = os.path.join(args.out_dir, "masks")
    points_dir = os.path.join(args.out_dir, "points")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(points_dir, exist_ok=True)

    write_mask_like(img, result["metal_mask_kji"], os.path.join(masks_dir, "metal_mask.nii.gz"))
    if result.get("head_mask_kji") is not None:
        write_mask_like(img, result["head_mask_kji"], os.path.join(masks_dir, "gating_mask.nii.gz"))
    debug_written = []
    debug_masks = result.get("debug_masks_kji") or {}
    for key, arr in debug_masks.items():
        out_name = f"{key}.nii.gz"
        out_path = os.path.join(masks_dir, out_name)
        write_mask_like(img, arr, out_path)
        debug_written.append(out_path)

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
        "outputs": {
            "metal_mask": os.path.join(masks_dir, "metal_mask.nii.gz"),
            "gating_mask": os.path.join(masks_dir, "gating_mask.nii.gz"),
            "in_mask_points_kji": os.path.join(points_dir, "in_mask_points_kji.csv"),
            "in_mask_points_ras": os.path.join(points_dir, "in_mask_points_ras.csv"),
        },
    }
    if debug_written:
        summary["outputs"]["debug_masks"] = debug_written
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        "[preview-masks] "
        f"candidates={summary['candidate_count']} in-head={summary['metal_in_head_count']} "
        f"in-mask={summary['in_mask_count']} mask={summary['gating_mask_type']} "
        f"inside_method={summary['inside_method']}"
    )
    print(f"[preview-masks] wrote outputs under {args.out_dir}")


def main():
    parser = argparse.ArgumentParser(description="ShankDetect CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prev = sub.add_parser("preview-masks", help="Build and export raw masks from CT")
    _add_common_mask_args(p_prev)
    p_prev.set_defaults(func=cmd_preview_masks)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
