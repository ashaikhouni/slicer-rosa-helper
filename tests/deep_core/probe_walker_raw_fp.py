"""Count raw walker false positives with no post-walker guards.

The production stage-1 pipeline chains: walker → dedup → amp_sum gate →
arbitrate → extend → second-pass → dedup → deep-tip prior → bolt-anchor
→ length → air/bone → deep-tip nodata. Each layer rejects a share of
walker hypotheses. This probe asks: if we kept only the walker's
intrinsic _walk_line filters (pitch bands, AX_TOL_MM, PERP_TOL_MM,
MIN_BLOBS_PER_LINE, MIN/MAX_LINE_SPAN_MM, MAX_INLIER_GAP_MM) and a
minimal dedup (to avoid double-counting the same physical line), how
many FPs slip through?

Matching: for each subject with CLEAN snapped GT axes (guided-fit
succeeds with tight thresholds — the ones we trust), count walker
lines that:
  - match some GT shank within 5° / 3 mm  → TP (relaxed a bit vs 3°/2mm
    since walker PCA axes before arbitrate can drift 1-2°)
  - don't match any GT shank              → FP
Report totals per subject and across dataset.

Run
---
    /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_walker_raw_fp.py [--strategy dixi|dixi_all]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path("/Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper")
sys.path.insert(0, str(ROOT / "CommonLib"))
sys.path.insert(0, str(ROOT / "PostopCTLocalization"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import SimpleITK as sitk

from rosa_detect import contact_pitch_v1_fit as cpfit
from rosa_detect import guided_fit_engine as gfe
from eval_seeg_localization import (
    iter_subject_rows, image_ijk_ras_matrices,
    load_reference_ground_truth_shanks,
)

DATASET_ROOT = Path("/Users/ammar/Dropbox/thalamus_subjects/seeg_localization")
EXCLUDE = {"T19", "T21"}  # unreliable manual GT per memory

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


def _is_clean(snap):
    if not snap.get("success"):
        return False
    return (
        int(snap.get("n_inliers", 0)) >= CLEAN_MIN_INLIERS
        and bool(snap.get("tight_refit", False))
        and float(snap.get("lateral_shift_mm", 1e9)) <= CLEAN_MAX_LATERAL_MM
        and float(snap.get("angle_deg", 1e9)) <= CLEAN_MAX_ANGLE_DEG
    )


def _raw_walker(log_arr, pts_c, amps_c, pitches_mm):
    """Seed-pair walker + minimal dedup. No amp_sum gate, no arbitration,
    no extend, no second pass, no deep-tip prior.
    """
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
            lo = mult * pitch - tol
            hi = mult * pitch + tol
            pair_mask |= (dist >= lo) & (dist <= hi)
        iu, ju = np.where(np.triu(pair_mask, k=1))
        for pi, pj in zip(iu, ju):
            h = cpfit._walk_line(int(pi), int(pj), pts_c, amps_c,
                                   pitch_mm=float(pitch))
            if h is not None:
                h["seed_pitch_mm"] = float(pitch)
                hyps.append(h)
    hyps.sort(key=lambda h: -h["n_blobs"])
    lines = cpfit._dedup_stage1_lines(hyps)
    return lines


def _match_count(walker_lines, gt_shanks, snaps):
    """Return (tp, fp, fn, matched_lines_set, matched_gt_set)."""
    matched_gt = set()
    matched_line = set()
    for gi, gt in enumerate(gt_shanks):
        if gi not in snaps:
            continue
        s_start, s_end = snaps[gi]
        g_axis, g_mid = _axis_mid(s_start, s_end)
        for li, l in enumerate(walker_lines):
            t_axis, t_mid = _axis_mid(l["start_ras"], l["end_ras"])
            if (_angle_deg(g_axis, t_axis) <= MATCH_ANGLE_DEG
                    and _perp_mid(g_mid, t_mid, t_axis) <= MATCH_MID_MM):
                matched_gt.add(gi)
                matched_line.add(li)
    tp = len(matched_line)
    fp = len(walker_lines) - tp
    fn = len([i for i in range(len(gt_shanks)) if i in snaps]) - len(matched_gt)
    return tp, fp, fn, matched_line, matched_gt


def _run_subject(row, pitches_mm, gt_src_label):
    sid = row["subject_id"]
    ct_path = row["ct_path"]
    if sid in REDONE_GT_CSVS and REDONE_GT_CSVS[sid].exists():
        gt_shanks = _load_redone_csv(REDONE_GT_CSVS[sid], sid)
        gt_src = "redone"
    else:
        gt_shanks, _ = load_reference_ground_truth_shanks(row, None)
        gt_src = "standard"
    if not gt_shanks:
        return dict(subject=sid, error="no GT")

    img = sitk.ReadImage(ct_path)
    img_c = sitk.Clamp(img, lowerBound=-1024.0, upperBound=cpfit.HU_CLIP_MAX)
    ijk2ras, ras2ijk = image_ijk_ras_matrices(img_c)
    ijk2ras = np.asarray(ijk2ras, dtype=float)
    ras2ijk = np.asarray(ras2ijk, dtype=float)
    features = gfe.compute_features(img_c, ijk2ras)
    log1 = features["log"]
    kji_to_ras = cpfit._kji_to_ras_fn_from_matrix(ijk2ras)

    # Snap each GT and keep CLEAN ones for matching.
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
    n_clean = len(snaps)

    # Raw walker output.
    blobs = cpfit.extract_blobs(log1, threshold=cpfit.LOG_BLOB_THRESHOLD)
    pts_ras = np.array([kji_to_ras(b["kji"]) for b in blobs])
    amps = np.array([b["amp"] for b in blobs], dtype=float)
    nvox = np.array([b["n_vox"] for b in blobs], dtype=int)
    mask = nvox <= cpfit.LOG_BLOB_MAX_VOXELS
    pts_c = pts_ras[mask]
    amps_c = amps[mask]

    walker_lines = _raw_walker(log1, pts_c, amps_c, pitches_mm)
    tp, fp, fn, matched_line, matched_gt = _match_count(walker_lines, gt_shanks, snaps)

    # Also compute what fraction of walker lines pass each downstream gate.
    # Just the amp_sum gate for quick context.
    n_pass_amp = sum(1 for l in walker_lines
                       if l.get("amp_sum", 0.0) >= cpfit.AMP_SUM_MIN)

    return dict(
        subject=sid, gt_src=gt_src, n_gt=len(gt_shanks), n_clean=n_clean,
        n_walker=len(walker_lines), tp=tp, fp=fp, fn=fn,
        n_pass_amp=n_pass_amp,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="dixi",
                      choices=["dixi", "dixi_mm", "dixi_all", "pmt", "mixed"])
    ap.add_argument("subjects", nargs="*")
    args = ap.parse_args()
    subjects_filter = set(args.subjects) if args.subjects else None

    pitches_mm = cpfit.resolve_pitches_for_strategy(args.strategy)
    print(
        f"Raw walker probe — strategy={args.strategy} "
        f"pitches={tuple(round(p, 2) for p in pitches_mm)}  "
        f"match window: angle ≤ {MATCH_ANGLE_DEG}°, mid ≤ {MATCH_MID_MM} mm"
    )

    rows = iter_subject_rows(DATASET_ROOT, subjects_filter)
    rows = [r for r in rows if str(r["subject_id"]) not in EXCLUDE]
    print(f"Processing {len(rows)} subjects")

    results = []
    for row in rows:
        sid = row["subject_id"]
        t0 = time.time()
        try:
            res = _run_subject(row, pitches_mm, args.strategy)
        except Exception as exc:
            res = dict(subject=sid, error=repr(exc))
        dt = time.time() - t0
        if "error" in res:
            print(f"  {sid}: ERROR {res['error']}  ({dt:.1f}s)")
        else:
            print(
                f"  {sid} [{res['gt_src']}]: walker={res['n_walker']:>4d}  "
                f"TP={res['tp']:>3d}/{res['n_clean']:<3d}  "
                f"FP={res['fp']:>4d}  FN={res['fn']:>2d}  "
                f"pass_amp_gate={res['n_pass_amp']:>4d}  ({dt:.1f}s)"
            )
        results.append(res)

    print(f"\n{'='*110}\nSUMMARY ({args.strategy})\n{'='*110}")
    print(
        f"{'subject':12s} {'n_walker':>8s} {'TP':>4s} {'CLEAN':>6s} "
        f"{'FP':>5s} {'FN':>3s} {'recall':>7s} {'precision':>10s}"
    )
    tot_walker = tot_tp = tot_clean = tot_fp = tot_fn = tot_pass_amp = 0
    for res in results:
        if "error" in res:
            print(f"{res['subject']:12s}  ERROR")
            continue
        n_walker = res["n_walker"]; tp = res["tp"]; fp = res["fp"]
        fn = res["fn"]; nc = res["n_clean"]
        recall = (100.0 * tp / nc) if nc else 0.0
        precision = (100.0 * tp / n_walker) if n_walker else 0.0
        tot_walker += n_walker; tot_tp += tp; tot_clean += nc
        tot_fp += fp; tot_fn += fn; tot_pass_amp += res["n_pass_amp"]
        print(
            f"{res['subject']:12s} {n_walker:>8d} {tp:>4d} {nc:>6d} "
            f"{fp:>5d} {fn:>3d} {recall:>6.1f}% {precision:>9.1f}%"
        )
    rec = (100.0 * tot_tp / tot_clean) if tot_clean else 0.0
    prec = (100.0 * tot_tp / tot_walker) if tot_walker else 0.0
    fp_per_subject = tot_fp / max(1, len(results))
    print(
        f"\n  DATASET TOTAL:  walker={tot_walker}  TP={tot_tp}/{tot_clean}  "
        f"FP={tot_fp}  FN={tot_fn}"
    )
    print(
        f"  recall={rec:.1f}%  precision={prec:.1f}%  "
        f"mean FP/subject={fp_per_subject:.1f}"
    )
    print(
        f"\n  Downstream filter preview: of {tot_walker} walker lines, "
        f"{tot_pass_amp} pass amp_sum ≥ {cpfit.AMP_SUM_MIN:.0f} gate "
        f"({100*tot_pass_amp/max(1,tot_walker):.1f}%). "
        f"Production adds arbitration, extend, deep-tip prior, bolt anchor, "
        f"length and air/bone filters on top."
    )


if __name__ == "__main__":
    main()
