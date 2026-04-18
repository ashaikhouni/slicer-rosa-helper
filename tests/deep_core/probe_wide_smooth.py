"""Probe D: can we salvage the soft-tissue filter on clipped CTs by
pre-smoothing the raw CT with a wide kernel (removing contact halos)?

Two variants:
  1. Wide Gaussian smooth (sigma in {3, 8, 15, 25} mm). Test: smoothed[v]
     in [-50, 80]. The larger sigma averages halos into surrounding brain.
  2. Wide-neighborhood quantile / mode: for each candidate, take the 5/25
     percentile of a LARGE-radius sphere (15 mm). Brain is the majority in
     a 15-mm sphere even when contacts and halos dominate within 5 mm.

Metric per subject: for the Frangi sigma=1 >= 10 cloud, what fraction of
in-GT-tube voxels pass vs out-of-tube voxels pass?  Better than the 5-mm-
median baseline means the filter retains contacts while still rejecting
bone/air.

Usage:
  /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
    tests/deep_core/probe_wide_smooth.py [T22|T2]
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
    DATASET_ROOT, load_gt, sphere_offsets, neighborhood_median,
    _line_axis_len, FRANGI_S1_THR, SOFT_HU_LO, SOFT_HU_HI,
)
from probe_detector_v4 import frangi_single  # noqa: E402


def gaussian_smooth(img, sigma_mm):
    import SimpleITK as sitk
    sm = sitk.SmoothingRecursiveGaussian(img, sigma=float(sigma_mm))
    return sitk.GetArrayFromImage(sm).astype(np.float32)


def neighborhood_quantile(raw_ct, kk, jj, ii, offs, q, chunk=2500):
    """Like neighborhood_median but returns the q-quantile (0-100)."""
    dk = np.array([o[0] for o in offs], dtype=np.int32)
    dj = np.array([o[1] for o in offs], dtype=np.int32)
    di = np.array([o[2] for o in offs], dtype=np.int32)
    K, J, I = raw_ct.shape
    N = kk.shape[0]
    out = np.empty(N, dtype=np.float32)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        kc = kk[start:end][None, :] + dk[:, None]
        jc = jj[start:end][None, :] + dj[:, None]
        ic = ii[start:end][None, :] + di[:, None]
        valid = (
            (kc >= 0) & (kc < K) & (jc >= 0) & (jc < J) & (ic >= 0) & (ic < I)
        )
        kcC = np.clip(kc, 0, K - 1)
        jcC = np.clip(jc, 0, J - 1)
        icC = np.clip(ic, 0, I - 1)
        vals = raw_ct[kcC, jcC, icC].astype(np.float32)
        vals = np.where(valid, vals, np.nan)
        out[start:end] = np.nanpercentile(vals, q, axis=0)
    return out


def run(subject_id):
    import SimpleITK as sitk
    from shank_core.io import image_ijk_ras_matrices

    ct_path = DATASET_ROOT / "post_registered_ct" / f"{subject_id}_post_registered.nii.gz"
    print(f"# subject={subject_id}  ct={ct_path.name}")
    img = sitk.ReadImage(str(ct_path))
    spacing = img.GetSpacing()
    raw_ct = sitk.GetArrayFromImage(img).astype(np.float32)

    t0 = time.time()
    frangi_s1 = frangi_single(img, sigma=1.0)
    print(f"# frangi sigma=1: max={frangi_s1.max():.1f} ({time.time()-t0:.1f}s)")

    frangi_mask = frangi_s1 >= FRANGI_S1_THR
    kk, jj, ii = np.where(frangi_mask)
    n0 = int(kk.size)
    print(f"# frangi>=10 cloud: {n0} voxels")

    ijk_to_ras_mat, _ = image_ijk_ras_matrices(img)
    ijk_to_ras_mat = np.asarray(ijk_to_ras_mat, dtype=float)

    def kji_to_ras(kji_arr):
        ijk = np.stack([kji_arr[:, 2], kji_arr[:, 1], kji_arr[:, 0]], axis=1)
        h = np.concatenate([ijk, np.ones((ijk.shape[0], 1))], axis=1)
        return (ijk_to_ras_mat @ h.T).T[:, :3]

    pts_kji = np.stack([kk, jj, ii], axis=1).astype(np.float64)
    pts_ras = kji_to_ras(pts_kji)

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

    def report(tag, values):
        # Candidate-value array `values` should be size n0 aligned with
        # frangi_mask voxels. Pass rate = fraction in [-50, 80].
        passed = (values >= SOFT_HU_LO) & (values <= SOFT_HU_HI)
        n_in = int((passed & any_tube).sum())
        n_out_p = int((passed & ~any_tube).sum())
        pass_in = 100.0 * n_in / max(n_tube, 1)
        pass_out = 100.0 * n_out_p / max(n_total_out := (n0 - n_tube), 1)
        specificity = 1.0 - (n_out_p / max(n_total_out, 1))
        total_kept = n_in + n_out_p
        fp_frac = 100.0 * n_out_p / max(total_kept, 1)
        print(
            f"  {tag:35s}  pass_in={pass_in:5.1f}%  "
            f"pass_out={pass_out:5.1f}%  "
            f"kept={total_kept:>6d}  GT_in_tube_kept={n_in:>5d}  "
            f"FP_frac={fp_frac:5.1f}%"
        )

    # --- Baseline: 5mm median (original Probe B filter) ---
    offs5 = sphere_offsets(spacing, 5.0)
    t0 = time.time()
    med5 = neighborhood_median(raw_ct, kk, jj, ii, offs5)
    print(f"\n# baseline 5mm median ({time.time()-t0:.1f}s):")
    report("baseline 5mm median", med5)

    # --- Wide Gaussian smooth ---
    for sig in [3.0, 8.0, 15.0, 25.0]:
        t0 = time.time()
        sm = gaussian_smooth(img, sig)
        print(f"\n# gaussian sigma={sig}mm  ({time.time()-t0:.1f}s)  "
              f"min={sm.min():.0f} max={sm.max():.0f}")
        vals = sm[kk, jj, ii]
        report(f"gauss sigma={sig}mm", vals)

    # --- Wide-sphere quantiles ---
    for radius, q in [(5.0, 25.0), (10.0, 25.0), (15.0, 25.0),
                       (10.0, 5.0), (15.0, 5.0)]:
        offs = sphere_offsets(spacing, radius)
        t0 = time.time()
        vals = neighborhood_quantile(raw_ct, kk, jj, ii, offs, q)
        print(f"\n# quantile p{int(q)} radius={radius}mm  "
              f"offsets={len(offs)}  ({time.time()-t0:.1f}s)")
        report(f"p{int(q)} radius={radius}mm", vals)


if __name__ == "__main__":
    subj = sys.argv[1] if len(sys.argv) > 1 else "T2"
    run(subj)
