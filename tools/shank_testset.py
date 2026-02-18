#!/usr/bin/env python3
"""Synthetic testset generator and smoke checks for shank mask logic.

Run with Slicer Python for guaranteed deps:
  /Applications/Slicer.app/Contents/MacOS/Slicer --no-splash --testing \
    --python-script tools/shank_testset.py -- make-synthetic --out-dir /tmp/shank_testset
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import SimpleITK as sitk

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIB_DIR = os.path.join(REPO_ROOT, "ShankDetect", "Lib")
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

from shank_core.masking import build_preview_masks  # noqa: E402


def _write_nifti(arr_kji, spacing_xyz, out_path):
    img = sitk.GetImageFromArray(np.asarray(arr_kji, dtype=np.int16))
    img.SetSpacing(tuple(float(v) for v in spacing_xyz))
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1.0, 0.0, 0.0,
                      0.0, 1.0, 0.0,
                      0.0, 0.0, 1.0))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sitk.WriteImage(img, out_path)


def _paint_sphere(arr, center_kji, radius_vox, value):
    k0, j0, i0 = [int(v) for v in center_kji]
    rk = int(np.ceil(radius_vox))
    for k in range(max(0, k0 - rk), min(arr.shape[0], k0 + rk + 1)):
        for j in range(max(0, j0 - rk), min(arr.shape[1], j0 + rk + 1)):
            for i in range(max(0, i0 - rk), min(arr.shape[2], i0 + rk + 1)):
                if ((k - k0) ** 2 + (j - j0) ** 2 + (i - i0) ** 2) <= (radius_vox ** 2):
                    arr[k, j, i] = value


def _paint_line_cylinder(arr, p0_kji, p1_kji, radius_vox, value):
    p0 = np.asarray(p0_kji, dtype=float)
    p1 = np.asarray(p1_kji, dtype=float)
    v = p1 - p0
    n = float(np.linalg.norm(v))
    if n <= 1e-6:
        return
    u = v / n
    steps = int(np.ceil(n * 2.0)) + 1
    for t in np.linspace(0.0, n, steps):
        c = p0 + u * t
        _paint_sphere(arr, c, radius_vox, value)


def make_synthetic_case(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    spacing_xyz = (0.8, 0.8, 0.8)
    shape_kji = (160, 160, 160)
    arr = np.full(shape_kji, -1000, dtype=np.int16)

    center = np.array([80.0, 80.0, 80.0], dtype=float)  # k,j,i
    zz, yy, xx = np.indices(shape_kji)
    rr = np.sqrt((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2)

    # Soft tissue core and skull shell.
    arr[rr <= 60.0] = 40
    arr[(rr > 60.0) & (rr <= 65.0)] = 1100

    # One intracranial electrode-like shank.
    p_entry = np.array([88.0, 40.0, 40.0])
    p_target = np.array([76.0, 110.0, 118.0])
    _paint_line_cylinder(arr, p_entry, p_target, radius_vox=1.2, value=2800)

    # Add contact-like clusters along the shaft.
    for t in np.linspace(0.05, 0.95, 12):
        pc = p_entry * (1.0 - t) + p_target * t
        _paint_sphere(arr, pc, radius_vox=1.3, value=3000)

    # Add extracranial metal nuisance near scalp.
    _paint_line_cylinder(arr, np.array([30.0, 20.0, 20.0]), np.array([30.0, 140.0, 25.0]), radius_vox=1.5, value=2800)

    ct_path = os.path.join(out_dir, "ct_synthetic.nii.gz")
    _write_nifti(arr, spacing_xyz, ct_path)

    meta = {
        "ct": ct_path,
        "spacing_xyz": spacing_xyz,
        "shape_kji": list(shape_kji),
        "planned_shank": {
            "entry_kji": p_entry.tolist(),
            "target_kji": p_target.tolist(),
        },
        "expected": {
            "auto_min_in_mask_count": 100,
            "soft_min_in_mask_count": 50,
        },
    }
    with open(os.path.join(out_dir, "expected.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[testset] wrote {ct_path}")
    print(f"[testset] wrote {os.path.join(out_dir, 'expected.json')}")


def _run_preview(ct_path, min_depth_mm):
    img = sitk.ReadImage(ct_path)
    arr = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    return build_preview_masks(
        arr_kji=arr,
        spacing_xyz=spacing,
        threshold=1800.0,
        use_head_mask=True,
        build_head_mask=True,
        head_mask_threshold_hu=-300.0,
        head_mask_aggressive_cleanup=True,
        head_mask_close_mm=2.0,
        min_metal_depth_mm=float(min_depth_mm),
        max_metal_depth_mm=80.0,
    )


def check_case(case_dir):
    with open(os.path.join(case_dir, "expected.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    ct_path = meta["ct"]
    expected = meta.get("expected", {})

    auto = _run_preview(ct_path, 0.0)
    soft = _run_preview(ct_path, 1.0)

    auto_count = int(auto.get("head_mask_kept_count", 0))
    soft_count = int(soft.get("head_mask_kept_count", 0))

    print(
        f"[check] depth>=0mm: candidates={auto.get('candidate_count')} in-mask={auto_count} "
        f"inside_method={auto.get('inside_method')}"
    )
    print(
        f"[check] depth>=1mm: candidates={soft.get('candidate_count')} in-mask={soft_count} "
        f"inside_method={soft.get('inside_method')}"
    )

    ok = True
    if auto_count < int(expected.get("auto_min_in_mask_count", 0)):
        print("[check] FAIL: auto in-mask below expected")
        ok = False
    if soft_count < int(expected.get("soft_min_in_mask_count", 0)):
        print("[check] FAIL: soft in-mask below expected")
        ok = False

    if ok:
        print("[check] PASS")
        return 0
    return 2


def main():
    parser = argparse.ArgumentParser(description="Shank synthetic testset tools")
    sub = parser.add_subparsers(dest="command", required=True)

    p_make = sub.add_parser("make-synthetic", help="Generate synthetic CT test case")
    p_make.add_argument("--out-dir", required=True)

    p_check = sub.add_parser("check", help="Run preview-mask checks on synthetic case")
    p_check.add_argument("--case-dir", required=True)

    args = parser.parse_args()
    if args.command == "make-synthetic":
        make_synthetic_case(args.out_dir)
        return
    if args.command == "check":
        rc = check_case(args.case_dir)
        sys.exit(rc)


if __name__ == "__main__":
    main()
