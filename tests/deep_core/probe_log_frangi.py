"""Probe E: Laplacian-of-Gaussian as a sharpness gate.

Hypothesis (Ammar): contacts and bolts are SHARP bright peaks (appear dark
in the Laplacian, i.e. strongly negative LoG response). Halo and broad
skull are smooth — weak LoG. So thresholding LoG (or multiplying Frangi
by |LoG|) may separate real contacts from halo on T2 where HU-based
filters fail.

Tests on T2 (failing subject) and T22 (working subject):

  1. Raw LoG distribution in GT tubes vs outside.
  2. Pass rate of voxels at LoG threshold sweep (on Frangi-positive cloud).
  3. Pass rate of voxels under a combined score = Frangi * |LoG| threshold.
  4. Compare: retention of in-tube voxels vs out-of-tube.

Usage:
  /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
    tests/deep_core/probe_log_frangi.py [T22|T2]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))
sys.path.insert(0, str(REPO_ROOT / "PostopCTLocalization"))
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "deep_core"))

from probe_contact_recovery import (  # noqa: E402
    DATASET_ROOT, load_gt, _line_axis_len, FRANGI_S1_THR,
)
from probe_detector_v4 import frangi_single  # noqa: E402


def log_sigma(img, sigma_mm):
    import SimpleITK as sitk
    log = sitk.LaplacianRecursiveGaussian(img, sigma=float(sigma_mm))
    return sitk.GetArrayFromImage(log).astype(np.float32)


def describe(arr, label):
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        print(f"  {label}: (empty)")
        return
    print(
        f"  {label:22s} n={arr.size:>6d}  "
        f"min={arr.min():>8.1f}  p5={np.percentile(arr,5):>8.1f}  "
        f"p25={np.percentile(arr,25):>8.1f}  "
        f"med={np.median(arr):>8.1f}  "
        f"p75={np.percentile(arr,75):>8.1f}  "
        f"p95={np.percentile(arr,95):>8.1f}  "
        f"max={arr.max():>8.1f}"
    )


def run(subject_id):
    import SimpleITK as sitk
    from shank_core.io import image_ijk_ras_matrices

    ct_path = DATASET_ROOT / "post_registered_ct" / f"{subject_id}_post_registered.nii.gz"
    print(f"# subject={subject_id}  ct={ct_path.name}")
    img = sitk.ReadImage(str(ct_path))

    t0 = time.time()
    frangi_s1 = frangi_single(img, sigma=1.0)
    print(f"# frangi sigma=1: max={frangi_s1.max():.1f} ({time.time()-t0:.1f}s)")

    # Compute LoG at a few scales
    log_scales = [0.5, 1.0, 1.5, 2.0]
    logs = {}
    for s in log_scales:
        t0 = time.time()
        l = log_sigma(img, s)
        logs[s] = l
        print(f"# log sigma={s}: min={l.min():.1f} max={l.max():.1f} "
              f"p1={np.percentile(l,1):.1f} p99={np.percentile(l,99):.1f} "
              f"({time.time()-t0:.1f}s)")

    # Pick Frangi-positive cloud
    frangi_mask = frangi_s1 >= FRANGI_S1_THR
    kk, jj, ii = np.where(frangi_mask)
    n0 = int(kk.size)
    print(f"# frangi>=10 cloud: {n0} voxels")

    # Build GT tube membership
    ijk_to_ras_mat, _ = image_ijk_ras_matrices(img)
    ijk_to_ras_mat = np.asarray(ijk_to_ras_mat, dtype=float)
    ijk = np.stack([ii, jj, kk], axis=1).astype(np.float64)
    h = np.concatenate([ijk, np.ones((ijk.shape[0], 1))], axis=1)
    pts_ras = (ijk_to_ras_mat @ h.T).T[:, :3]

    gt = load_gt(subject_id)
    any_tube = np.zeros(n0, dtype=bool)
    for (name, g_s, g_e, L) in gt:
        axis, gL = _line_axis_len(g_s, g_e)
        diffs = pts_ras - g_s[None, :]
        proj = diffs @ axis
        perp = diffs - np.outer(proj, axis)
        dist = np.linalg.norm(perp, axis=1)
        any_tube |= (proj >= 0.0) & (proj <= gL) & (dist <= 2.0)
    n_tube = int(any_tube.sum())
    n_out = n0 - n_tube
    print(f"# in GT tubes: {n_tube}   out of tubes: {n_out}")

    # --- LoG distribution in tubes vs out ---
    print("\n## Raw LoG distribution (Frangi-positive voxels only)")
    for s in log_scales:
        l = logs[s]
        v_in = l[kk[any_tube], jj[any_tube], ii[any_tube]]
        v_out = l[kk[~any_tube], jj[~any_tube], ii[~any_tube]]
        print(f"# log sigma={s}:")
        describe(v_in, "in tube")
        describe(v_out, "out of tube")

    # --- Threshold sweep on LoG alone ---
    print("\n## Threshold sweep on LoG: keep voxels where LoG <= -T")
    print(f"{'scale':>5s} {'T':>6s}  {'kept':>7s}  "
          f"{'pass_in':>7s}  {'pass_out':>8s}  "
          f"{'GT_kept':>7s}  {'FP_frac':>7s}")
    for s in log_scales:
        l = logs[s]
        v_all = l[kk, jj, ii]
        for T in [10, 30, 100, 300, 1000, 3000]:
            keep = v_all <= -float(T)
            kept = int(keep.sum())
            if kept == 0:
                continue
            n_in = int((keep & any_tube).sum())
            n_out_p = int((keep & ~any_tube).sum())
            pass_in = 100.0 * n_in / max(n_tube, 1)
            pass_out = 100.0 * n_out_p / max(n_out, 1)
            fp_frac = 100.0 * n_out_p / kept
            print(f"{s:>5.1f} {T:>6d}  {kept:>7d}  "
                  f"{pass_in:>6.1f}%  {pass_out:>7.1f}%  "
                  f"{n_in:>7d}  {fp_frac:>6.1f}%")

    # --- Combined score: Frangi * |LoG| (clip LoG to negatives) ---
    print("\n## Combined score: Frangi * max(-LoG, 0)  (only bright-peak LoG)")
    print(f"{'scale':>5s} {'score_T':>7s}  {'kept':>7s}  "
          f"{'pass_in':>7s}  {'pass_out':>8s}  "
          f"{'GT_kept':>7s}  {'FP_frac':>7s}")
    frangi_at_cloud = frangi_s1[kk, jj, ii]
    for s in log_scales:
        l = logs[s]
        log_at_cloud = l[kk, jj, ii]
        # Clip LoG to bright-peak component only (negative values)
        bright = np.maximum(-log_at_cloud, 0.0)
        score = frangi_at_cloud * bright
        # Sweep thresholds as quantiles for stability
        for T in [1e2, 1e3, 1e4, 1e5, 1e6]:
            keep = score >= T
            kept = int(keep.sum())
            if kept == 0:
                continue
            n_in = int((keep & any_tube).sum())
            n_out_p = int((keep & ~any_tube).sum())
            pass_in = 100.0 * n_in / max(n_tube, 1)
            pass_out = 100.0 * n_out_p / max(n_out, 1)
            fp_frac = 100.0 * n_out_p / kept
            print(f"{s:>5.1f} {T:>7.0f}  {kept:>7d}  "
                  f"{pass_in:>6.1f}%  {pass_out:>7.1f}%  "
                  f"{n_in:>7d}  {fp_frac:>6.1f}%")

    # --- Save NIFTI outputs for a couple of chosen operating points ---
    import SimpleITK as sitk
    best_s = 1.0
    l = logs[best_s]
    # Threshold-only at T=100
    keep_log = np.zeros_like(l, dtype=np.uint8)
    mask = (l <= -100.0) & frangi_mask
    keep_log[mask] = 1
    out_img = sitk.GetImageFromArray(keep_log)
    out_img.CopyInformation(img)
    sitk.WriteImage(out_img, f"/tmp/log_thr_{subject_id}_s1_T100.nii.gz")

    # Combined score at sigma=1, T=1e4
    log_at = l[kk, jj, ii]
    frangi_at = frangi_s1[kk, jj, ii]
    score = frangi_at * np.maximum(-log_at, 0.0)
    keep_combo = score >= 1e4
    combo_arr = np.zeros_like(l, dtype=np.uint8)
    combo_arr[kk[keep_combo], jj[keep_combo], ii[keep_combo]] = 1
    out_img = sitk.GetImageFromArray(combo_arr)
    out_img.CopyInformation(img)
    sitk.WriteImage(out_img, f"/tmp/log_x_frangi_{subject_id}_s1.nii.gz")
    print(f"\n# wrote /tmp/log_thr_{subject_id}_s1_T100.nii.gz and "
          f"/tmp/log_x_frangi_{subject_id}_s1.nii.gz")


if __name__ == "__main__":
    subj = sys.argv[1] if len(sys.argv) > 1 else "T2"
    run(subj)
