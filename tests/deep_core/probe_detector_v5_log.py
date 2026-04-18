"""Probe v5-LoG: drop-in replacement of the v4 cloud with a LoG-gated cloud.

Everything else (iterative RANSAC, dedup, periodicity confirmation, GT
matching) is identical to v4. Only the cloud changes:

  v4 cloud: Frangi sigma=1 >= 10  INTERSECT  (hull + head_distance >= 10mm)
  v5 cloud: LoG sigma=1 <= -T     INTERSECT  hull
            [hull only -- not head_distance, so bolts aren't excluded]

Measures whether LoG alone (scanner-universal signature) can replace the
HU-based intracranial distance gate.

Usage:
  /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
    tests/deep_core/probe_detector_v5_log.py [T22|T2] [LOG_THR]
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

from probe_detector_v4 import (  # noqa: E402
    DATASET_ROOT, INTRACRANIAL_MIN_DISTANCE_MM,
    RANSAC_MIN_INLIERS, RANSAC_MIN_SPAN_MM, RANSAC_MAX_SPAN_MM,
    RANSAC_TOL_MM, RANSAC_ITER, RANSAC_EXCLUSION_MM,
    PITCH_MIN_MM, PITCH_MAX_MM, PITCH_MIN_SNR, PITCH_MIN_TRACK_LEN_MM,
    LONG_TRACK_BYPASS_MM,
    frangi_single, ransac_one_line, dedup_tracks,
    periodicity_along_polyline, save_label_volume,
    load_gt, match_tracks_to_gt,
)


def build_hull_only(img):
    """Hull mask only (no head_distance gate). So bolts at skull aren't
    excluded by construction."""
    import SimpleITK as sitk
    thr = sitk.BinaryThreshold(img, lowerThreshold=-500.0, upperThreshold=1e9,
                                insideValue=1, outsideValue=0)
    cc = sitk.RelabelComponent(sitk.ConnectedComponent(thr), sortByObjectSize=True)
    largest = sitk.Equal(cc, 1)
    closed = sitk.BinaryMorphologicalClosing(largest, kernelRadius=(3, 3, 3))
    hull = sitk.BinaryFillhole(closed)
    return sitk.GetArrayFromImage(hull).astype(bool)


def log_sigma(img, sigma_mm):
    import SimpleITK as sitk
    log = sitk.LaplacianRecursiveGaussian(img, sigma=float(sigma_mm))
    return sitk.GetArrayFromImage(log).astype(np.float32)


def iterative_ransac_on(arr_values, arr_mask):
    """Run v4-style iterative RANSAC with 'weight' given by the magnitude
    of the passed values.  arr_values carries the score (higher = stronger
    tube); arr_mask defines the candidate voxels."""
    kk, jj, ii = np.where(arr_mask)
    pts = np.stack([kk.astype(float), jj.astype(float), ii.astype(float)], axis=1)
    weights = arr_values[arr_mask].astype(float)
    # Guard against zero or negative weights
    weights = np.clip(weights, 1e-3, None)
    print(f"  cloud voxels: {pts.shape[0]}  "
          f"weight_max={weights.max():.1f}  mean={weights.mean():.1f}")
    rng = np.random.default_rng(0)
    lines = []
    active = np.ones(pts.shape[0], dtype=bool)
    for iteration in range(60):
        if int(active.sum()) < RANSAC_MIN_INLIERS:
            break
        active_pts = pts[active]
        active_w = weights[active]
        result = ransac_one_line(
            active_pts, active_w, tol_mm=RANSAC_TOL_MM, n_iter=RANSAC_ITER,
            min_inliers=RANSAC_MIN_INLIERS, min_span=RANSAC_MIN_SPAN_MM, rng=rng,
        )
        if result is None:
            break
        step = 0.5
        n_steps = int(result["span_mm"] / step) + 1
        t = np.linspace(result["span_lo"], result["span_hi"], n_steps)
        polyline = result["center"] + t[:, None] * result["axis"]
        lines.append({
            "polyline_kji": polyline,
            "length_mm": float(result["span_mm"]),
            "n_inliers": int(result["n_inliers"]),
            "axis": result["axis"],
            "center": result["center"],
        })
        diffs = active_pts - result["center"]
        proj = diffs @ result["axis"]
        perp = diffs - np.outer(proj, result["axis"])
        dists = np.linalg.norm(perp, axis=1)
        to_remove = (
            (dists < RANSAC_EXCLUSION_MM)
            & (proj >= result["span_lo"] - 2.0)
            & (proj <= result["span_hi"] + 2.0)
        )
        active_idx = np.where(active)[0]
        active[active_idx[to_remove]] = False
    return lines


def run(subject_id, log_thr):
    import SimpleITK as sitk
    from shank_core.io import image_ijk_ras_matrices

    ct_path = DATASET_ROOT / "post_registered_ct" / f"{subject_id}_post_registered.nii.gz"
    print(f"# subject={subject_id}  log_thr={log_thr}")
    img = sitk.ReadImage(str(ct_path))

    t0 = time.time()
    hull = build_hull_only(img)
    print(f"# hull mask (no head_distance gate): {int(hull.sum())} voxels "
          f"({time.time()-t0:.1f}s)")

    t0 = time.time()
    frangi_s1 = frangi_single(img, sigma=1.0)
    print(f"# frangi sigma=1: max={frangi_s1.max():.1f} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    log_s1 = log_sigma(img, sigma_mm=1.0)
    print(f"# log sigma=1: min={log_s1.min():.1f} p1={np.percentile(log_s1, 1):.1f} "
          f"({time.time()-t0:.1f}s)")

    # Cloud: LoG strong AND inside hull
    cloud_mask = (log_s1 <= -float(log_thr)) & hull

    # Weight voxels by Frangi response (for RANSAC sampling bias)
    weights = np.where(cloud_mask, frangi_s1, 0.0)

    t0 = time.time()
    tracks = iterative_ransac_on(weights, cloud_mask)
    n_pre = len(tracks)
    tracks = dedup_tracks(tracks)
    print(f"# RANSAC: {n_pre} -> {len(tracks)} after dedup "
          f"({time.time()-t0:.1f}s)")

    ijk_to_ras_mat, _ = image_ijk_ras_matrices(img)
    ijk_to_ras_mat = np.asarray(ijk_to_ras_mat, dtype=float)

    def kji_to_ras(kji):
        ijk = np.stack([kji[:, 2], kji[:, 1], kji[:, 0]], axis=1)
        h = np.concatenate([ijk, np.ones((ijk.shape[0], 1))], axis=1)
        return (ijk_to_ras_mat @ h.T).T[:, :3]

    tracks_ras = []
    for t in tracks:
        pts_ras = kji_to_ras(t["polyline_kji"])
        tracks_ras.append({
            "polyline_ras": pts_ras, "start_ras": pts_ras[0], "end_ras": pts_ras[-1],
            "length_mm": t["length_mm"], "n_inliers": t["n_inliers"],
            "polyline_kji": t["polyline_kji"],
        })

    # We still need periodicity confirmation; pass the cloud mask as
    # "intracranial" for the purposes of in-brain-only sampling.
    accepted = []
    for t, ts in zip(tracks, tracks_ras):
        pitch, snr = periodicity_along_polyline(frangi_s1, t["polyline_kji"], cloud_mask)
        periodic_ok = (
            ts["length_mm"] >= PITCH_MIN_TRACK_LEN_MM
            and np.isfinite(pitch)
            and PITCH_MIN_MM <= pitch <= PITCH_MAX_MM
            and snr >= PITCH_MIN_SNR
        )
        passes = periodic_ok or ts["length_mm"] >= LONG_TRACK_BYPASS_MM
        ts["pitch_mm"] = pitch; ts["snr"] = snr; ts["passes"] = passes
        if passes:
            accepted.append(ts)

    print(f"\n# total tracks={len(tracks)}  accepted={len(accepted)}")
    gt = load_gt(subject_id)
    matches = match_tracks_to_gt(accepted, gt)
    n_matched = sum(1 for m in matches if m["matched"])
    fp = len(accepted) - n_matched
    print(f"# matched={n_matched}/{len(gt)}  FP={fp}")

    for m in matches:
        print(
            f"  {m['gt_name']:10s} {'YES' if m['matched'] else 'no':>5s}  "
            f"ang={m.get('angle_deg', float('nan')):.2f}deg  "
            f"mid_d={m.get('mid_d_mm', float('nan')):.2f}mm"
        )

    save_label_volume(accepted, frangi_s1.shape, img,
                      f"/tmp/detected_{subject_id}_v5log_T{int(log_thr)}.nii.gz")
    print(f"\n# wrote /tmp/detected_{subject_id}_v5log_T{int(log_thr)}.nii.gz")


if __name__ == "__main__":
    subj = sys.argv[1] if len(sys.argv) > 1 else "T22"
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
    run(subj, thr)
