"""Sample Frangi σ=1 along raw walker lines and compare the distribution
for true positives vs false positives.

For each subject:
  1. Run raw walker (no guards).
  2. For each walker line, sample Frangi at 0.5 mm steps along the axis
     from start to end. Compute mean, median, max, and fraction of
     samples above Frangi ≥ 30 (the stage-2 threshold).
  3. Classify each line as TP (matches a CLEAN GT within 5°/3mm) or FP.
  4. Report the distributions.

If Frangi is a useful FP filter we'd expect TP lines to saturate the
response (high mean, near-100% above threshold) and FP lines to sit
well below.

Run
---
    /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_walker_frangi_fp.py [--strategy dixi|dixi_all]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper")
sys.path.insert(0, str(ROOT / "CommonLib"))
sys.path.insert(0, str(ROOT / "PostopCTLocalization"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import SimpleITK as sitk

from postop_ct_localization import contact_pitch_v1_fit as cpfit
from postop_ct_localization import guided_fit_engine as gfe
from eval_seeg_localization import (
    iter_subject_rows, image_ijk_ras_matrices,
    load_reference_ground_truth_shanks,
)

DATASET_ROOT = Path("/Users/ammar/Dropbox/thalamus_subjects/seeg_localization")
EXCLUDE = {"T19", "T21"}
REDONE_GT_CSVS = {
    "T4": DATASET_ROOT / "contact_label_dataset" / "rosa_helper_import"
          / "T4" / "ROSA_Contacts_T4_final_trajectory_points.csv",
}

CLEAN_MIN_INLIERS = 5
CLEAN_MAX_LATERAL_MM = 2.0
CLEAN_MAX_ANGLE_DEG = 5.0

MATCH_ANGLE_DEG = 5.0
MATCH_MID_MM = 3.0


def _load_redone_csv(csv_path, subject_id):
    by = defaultdict(dict)
    with Path(csv_path).open() as f:
        for r in csv.DictReader(f):
            by[r["trajectory"]][r["point_type"]] = (
                float(r["x_world_ras"]),
                float(r["y_world_ras"]),
                float(r["z_world_ras"]),
            )
    out = []
    for shank, ends in sorted(by.items()):
        if "entry" in ends and "target" in ends:
            out.append(type("GT", (), dict(
                subject_id=subject_id, shank=shank,
                start_ras=ends["entry"], end_ras=ends["target"],
            )))
    return out


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def _axis_mid(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return _unit(b - a), 0.5 * (a + b)


def _angle_deg(u, v):
    return float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(u, v)))))))


def _perp_mid(g_mid, t_mid, t_axis):
    d = g_mid - t_mid
    p = d - (d @ t_axis) * t_axis
    return float(np.linalg.norm(p))


def _sample_axis(frangi_kji, ras_to_ijk_mat, start, end, step_mm=0.5):
    K, J, I = frangi_kji.shape
    s = np.asarray(start, dtype=float); e = np.asarray(end, dtype=float)
    d = e - s
    L = float(np.linalg.norm(d))
    if L < 1e-3:
        return np.array([])
    u = d / L
    n = max(2, int(L / step_mm) + 1)
    vals = np.zeros(n, dtype=np.float32)
    for idx in range(n):
        p = s + (idx * step_mm if idx < n - 1 else L) * u
        h = np.array([p[0], p[1], p[2], 1.0])
        ijk = (ras_to_ijk_mat @ h)[:3]
        kc = int(np.clip(round(ijk[2]), 0, K - 1))
        jc = int(np.clip(round(ijk[1]), 0, J - 1))
        ic = int(np.clip(round(ijk[0]), 0, I - 1))
        vals[idx] = float(frangi_kji[kc, jc, ic])
    return vals


def _is_clean(snap):
    if not snap.get("success"):
        return False
    return (
        int(snap.get("n_inliers", 0)) >= CLEAN_MIN_INLIERS
        and bool(snap.get("tight_refit", False))
        and float(snap.get("lateral_shift_mm", 1e9)) <= CLEAN_MAX_LATERAL_MM
        and float(snap.get("angle_deg", 1e9)) <= CLEAN_MAX_ANGLE_DEG
    )


def _raw_walker(pts_c, amps_c, pitches_mm):
    if pts_c.shape[0] < 2:
        return []
    dist = np.sqrt(np.sum(
        (pts_c[:, None, :] - pts_c[None, :, :]) ** 2, axis=2,
    ))
    tol = cpfit.PITCH_TOL_MM
    hyps = []
    for pitch in pitches_mm:
        pair_mask = np.zeros_like(dist, dtype=bool)
        for mult in (1, 2, 3):
            pair_mask |= ((dist >= mult * pitch - tol)
                           & (dist <= mult * pitch + tol))
        iu, ju = np.where(np.triu(pair_mask, k=1))
        for pi, pj in zip(iu, ju):
            h = cpfit._walk_line(int(pi), int(pj), pts_c, amps_c,
                                   pitch_mm=float(pitch))
            if h is not None:
                hyps.append(h)
    hyps.sort(key=lambda h: -h["n_blobs"])
    return cpfit._dedup_stage1_lines(hyps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="dixi",
                      choices=["dixi", "dixi_mm", "dixi_all", "pmt", "mixed"])
    args = ap.parse_args()
    pitches_mm = cpfit.resolve_pitches_for_strategy(args.strategy)
    print(
        f"Frangi-along-walker-line probe  strategy={args.strategy}  "
        f"pitches={tuple(round(p,2) for p in pitches_mm)}"
    )

    rows = [r for r in iter_subject_rows(DATASET_ROOT, None)
             if str(r["subject_id"]) not in EXCLUDE]
    print(f"Processing {len(rows)} subjects")

    tp_records = []
    fp_records = []
    for row in rows:
        sid = row["subject_id"]
        img = sitk.ReadImage(row["ct_path"])
        img_c = sitk.Clamp(img, lowerBound=-1024.0, upperBound=cpfit.HU_CLIP_MAX)
        ijk2ras, ras2ijk = image_ijk_ras_matrices(img_c)
        ijk2ras = np.asarray(ijk2ras, dtype=float)
        ras2ijk = np.asarray(ras2ijk, dtype=float)
        features = gfe.compute_features(img_c, ijk2ras)
        log1 = features["log"]
        frangi = cpfit.frangi_single(img_c, sigma=cpfit.FRANGI_STAGE1_SIGMA)
        kji_to_ras = cpfit._kji_to_ras_fn_from_matrix(ijk2ras)

        if sid in REDONE_GT_CSVS and REDONE_GT_CSVS[sid].exists():
            gt_shanks = _load_redone_csv(REDONE_GT_CSVS[sid], sid)
        else:
            gt_shanks, _ = load_reference_ground_truth_shanks(row, None)

        snaps = {}
        for gi, gt in enumerate(gt_shanks):
            snap = gfe.fit_trajectory(
                np.asarray(gt.start_ras), np.asarray(gt.end_ras),
                features, ijk2ras, ras2ijk,
            )
            if _is_clean(snap):
                snaps[gi] = (
                    np.asarray(snap["start_ras"], dtype=float),
                    np.asarray(snap["end_ras"], dtype=float),
                )

        blobs = cpfit.extract_blobs(log1, threshold=cpfit.LOG_BLOB_THRESHOLD)
        pts_ras = np.array([kji_to_ras(b["kji"]) for b in blobs])
        amps = np.array([b["amp"] for b in blobs], dtype=float)
        nvox = np.array([b["n_vox"] for b in blobs], dtype=int)
        mask = nvox <= cpfit.LOG_BLOB_MAX_VOXELS
        pts_c = pts_ras[mask]
        amps_c = amps[mask]

        walker_lines = _raw_walker(pts_c, amps_c, pitches_mm)

        for l in walker_lines:
            vals = _sample_axis(frangi, ras2ijk,
                                 l["start_ras"], l["end_ras"], step_mm=0.5)
            if vals.size == 0:
                continue
            rec = dict(
                subject=sid,
                n_blobs=int(l["n_blobs"]),
                span_mm=float(l.get("span_mm", 0.0)),
                amp_sum=float(l.get("amp_sum", 0.0)),
                fr_mean=float(vals.mean()),
                fr_median=float(np.median(vals)),
                fr_max=float(vals.max()),
                fr_min=float(vals.min()),
                fr_p10=float(np.percentile(vals, 10)),
                fr_p25=float(np.percentile(vals, 25)),
                fr_frac_30=float((vals >= 30).mean()),
                n_samples=int(vals.size),
            )
            # Classify
            t_axis, t_mid = _axis_mid(l["start_ras"], l["end_ras"])
            is_tp = False
            for gi, (gs, ge) in snaps.items():
                g_axis, g_mid = _axis_mid(gs, ge)
                if (_angle_deg(g_axis, t_axis) <= MATCH_ANGLE_DEG
                        and _perp_mid(g_mid, t_mid, t_axis) <= MATCH_MID_MM):
                    is_tp = True
                    break
            if is_tp:
                tp_records.append(rec)
            else:
                fp_records.append(rec)

    print(f"\nTotal lines: TP={len(tp_records)}  FP={len(fp_records)}")

    def _qstat(arr, label):
        if not arr:
            print(f"  {label}: (n=0)")
            return
        a = np.asarray(arr)
        print(
            f"  {label:12s} p10={np.percentile(a,10):7.2f}  "
            f"p25={np.percentile(a,25):7.2f}  "
            f"p50={np.percentile(a,50):7.2f}  "
            f"p75={np.percentile(a,75):7.2f}  "
            f"p90={np.percentile(a,90):7.2f}  "
            f"min={a.min():7.2f}  max={a.max():7.2f}"
        )

    print(f"\nTP Frangi stats (n={len(tp_records)}):")
    _qstat([r["fr_mean"] for r in tp_records], "fr_mean")
    _qstat([r["fr_median"] for r in tp_records], "fr_median")
    _qstat([r["fr_max"] for r in tp_records], "fr_max")
    _qstat([r["fr_p25"] for r in tp_records], "fr_p25")
    _qstat([r["fr_frac_30"] for r in tp_records], "fr_frac≥30")

    print(f"\nFP Frangi stats (n={len(fp_records)}):")
    _qstat([r["fr_mean"] for r in fp_records], "fr_mean")
    _qstat([r["fr_median"] for r in fp_records], "fr_median")
    _qstat([r["fr_max"] for r in fp_records], "fr_max")
    _qstat([r["fr_p25"] for r in fp_records], "fr_p25")
    _qstat([r["fr_frac_30"] for r in fp_records], "fr_frac≥30")

    # Discrimination: what threshold on each feature kills most FP while
    # keeping most TP?
    print(f"\n{'='*80}\nDiscrimination — TP retention / FP rejection @ thresholds\n{'='*80}")
    for feat in ["fr_mean", "fr_median", "fr_p25", "fr_frac_30"]:
        tp_vals = np.array([r[feat] for r in tp_records])
        fp_vals = np.array([r[feat] for r in fp_records])
        # For frac, threshold should be a fraction. Reuse sweep.
        if feat == "fr_frac_30":
            thresholds = [0.30, 0.50, 0.70, 0.80, 0.90, 0.95]
        else:
            thresholds = [5, 10, 20, 30, 50, 100, 150, 200]
        print(f"\n  {feat}:")
        print(f"    {'thr':>7s}  {'TP kept':>8s}  {'FP rejected':>12s}")
        for thr in thresholds:
            tp_keep = float((tp_vals >= thr).mean())
            fp_reject = float((fp_vals < thr).mean())
            print(f"    {thr:>7}  {100*tp_keep:>7.1f}%  {100*fp_reject:>11.1f}%")


if __name__ == "__main__":
    main()
