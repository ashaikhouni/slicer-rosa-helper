"""Evaluate the stage-1 walker against trustworthy snapped-GT axes.

For every subject (excluding T17, T22, T19, T21):
  1. Load GT (redone CSV for T4; standard manifest otherwise).
  2. Guided-fit each GT shank — keep only CLEAN snaps (n_inliers ≥ 5,
     tight_refit=True, lateral ≤ 2 mm, angle ≤ 5°).
  3. Run stage 1 (blob-pitch walker) on the CT.
  4. For each CLEAN GT: find the best-matching stage-1 line and flag
     whether it meets a TIGHT match window (3° / 2 mm midpoint).
  5. Per-subject detection rate; per-missed-shank, dump the LoG blob
     statistics along the snapped axis.

Output: per-subject table + list of missed CLEAN shanks + blob-stat
breakdown of detected vs missed.

Run
---
    /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_stage1_vs_snapped_gt.py
"""
from __future__ import annotations

import csv
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
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
EXCLUDE = {"T19", "T21", "T17"}  # GT unreliable per prior probes; keep T22 for reference

REDONE_GT_CSVS = {
    "T4": DATASET_ROOT / "contact_label_dataset" / "rosa_helper_import"
          / "T4" / "ROSA_Contacts_T4_final_trajectory_points.csv",
}

# Snap → CLEAN thresholds (same as the snap probe)
CLEAN_MIN_INLIERS = 5
CLEAN_MAX_LATERAL_MM = 2.0
CLEAN_MAX_ANGLE_DEG = 5.0

# Stage-1-vs-snapped-GT match window (tight — snapped axes are CT-accurate)
MATCH_ANGLE_DEG = 3.0
MATCH_MID_MM = 2.0

# Blob-stat probe
TUBE_RADIUS_MM = 2.0
STRONG_AMP = 500.0


@dataclass
class _RedoneShank:
    subject_id: str
    shank: str
    start_ras: tuple
    end_ras: tuple


def _load_redone_csv(csv_path, subject_id):
    by = defaultdict(dict)
    with Path(csv_path).open() as f:
        for row in csv.DictReader(f):
            by[row["trajectory"]][row["point_type"]] = (
                float(row["x_world_ras"]),
                float(row["y_world_ras"]),
                float(row["z_world_ras"]),
            )
    out = []
    for shank, ends in sorted(by.items()):
        if "entry" in ends and "target" in ends:
            out.append(_RedoneShank(subject_id, shank,
                                     ends["entry"], ends["target"]))
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


def _blobs_in_tube(pts_ras, amps, start, end, radius_mm):
    start = np.asarray(start, dtype=float); end = np.asarray(end, dtype=float)
    u = end - start
    L = float(np.linalg.norm(u))
    if L < 1e-3:
        return np.array([]), np.array([])
    u /= L
    v = pts_ras - start[None, :]
    axial = v @ u
    perp = v - axial[:, None] * u[None, :]
    perp_d = np.linalg.norm(perp, axis=1)
    keep = (axial >= -2.0) & (axial <= L + 2.0) & (perp_d <= radius_mm)
    ax = axial[keep]; am = amps[keep]
    order = np.argsort(ax)
    return ax[order], am[order]


def _is_clean(snap):
    if not snap.get("success"):
        return False
    return (
        int(snap.get("n_inliers", 0)) >= CLEAN_MIN_INLIERS
        and bool(snap.get("tight_refit", False))
        and float(snap.get("lateral_shift_mm", 1e9)) <= CLEAN_MAX_LATERAL_MM
        and float(snap.get("angle_deg", 1e9)) <= CLEAN_MAX_ANGLE_DEG
    )


def _best_stage1_match(trajs, g_axis, g_mid):
    best = None
    for ti, t in enumerate(trajs):
        t_axis, t_mid = _axis_mid(t["start_ras"], t["end_ras"])
        ang = _angle_deg(g_axis, t_axis)
        mid = _perp_mid(g_mid, t_mid, t_axis)
        score = ang + mid
        if best is None or score < best["score"]:
            best = dict(
                ti=ti, ang=ang, mid=mid, score=score,
                n_blobs=int(t.get("n_blobs", t.get("n_inliers", 0))),
                span=float(t.get("span_mm", 0.0)),
            )
    return best


def _run_subject(row):
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
    pts_ras = features["blob_pts_ras"]
    amps = features["blob_amps"]
    log1 = features["log"]
    dist_arr = features["head_distance"]
    kji_to_ras = cpfit._kji_to_ras_fn_from_matrix(ijk2ras)

    # Collect CLEAN snapped axes.
    clean = []
    for gt in gt_shanks:
        snap = gfe.fit_trajectory(
            np.asarray(gt.start_ras), np.asarray(gt.end_ras),
            features, ijk2ras, ras2ijk,
        )
        if _is_clean(snap):
            clean.append(dict(
                shank=gt.shank,
                start=np.asarray(snap["start_ras"], dtype=float),
                end=np.asarray(snap["end_ras"], dtype=float),
            ))

    # Stage 1 walker — dixi pitch (production default).
    auto_pitches = cpfit.resolve_pitches_for_strategy("dixi")
    stage1_lines, pts_blobs = cpfit.run_stage1(
        log1, kji_to_ras, dist_arr, ras2ijk,
        pitches_mm=auto_pitches,
    )

    # Match each clean GT.
    result_shanks = []
    for c in clean:
        g_axis, g_mid = _axis_mid(c["start"], c["end"])
        best = _best_stage1_match(stage1_lines, g_axis, g_mid)
        matched = (
            best is not None
            and best["ang"] <= MATCH_ANGLE_DEG
            and best["mid"] <= MATCH_MID_MM
        )
        # Blob stats along snapped axis.
        depths, ain = _blobs_in_tube(pts_ras, amps,
                                      c["start"], c["end"], TUBE_RADIUS_MM)
        n_strong = int((ain >= STRONG_AMP).sum())
        amp_p50 = float(np.percentile(ain, 50)) if ain.size else float("nan")
        amp_p10 = float(np.percentile(ain, 10)) if ain.size else float("nan")
        axis_len = float(np.linalg.norm(c["end"] - c["start"]))
        rec = dict(
            subject=sid, shank=c["shank"], matched=matched,
            best=best,
            n_blobs=int(ain.size), n_strong=n_strong,
            amp_p50=amp_p50, amp_p10=amp_p10,
            length=axis_len,
        )
        result_shanks.append(rec)

    return dict(
        subject=sid, gt_src=gt_src, n_clean=len(clean),
        n_gt=len(gt_shanks), shanks=result_shanks,
        n_stage1=len(stage1_lines),
        auto_pitches=auto_pitches,
    )


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    filt = set(args) if args else None
    rows = iter_subject_rows(DATASET_ROOT, filt)
    rows = [r for r in rows if str(r["subject_id"]) not in EXCLUDE]
    print(f"Processing {len(rows)} subjects (EXCLUDE={sorted(EXCLUDE)})")
    print(f"Match window: angle ≤ {MATCH_ANGLE_DEG}°, mid ≤ {MATCH_MID_MM} mm")

    results = []
    for row in rows:
        sid = row["subject_id"]
        t0 = time.time()
        try:
            res = _run_subject(row)
        except Exception as exc:
            res = dict(subject=sid, error=repr(exc))
        dt = time.time() - t0
        if "error" in res:
            print(f"  {sid}: ERROR {res['error']}  ({dt:.1f}s)")
        else:
            m = sum(1 for s in res["shanks"] if s["matched"])
            n = len(res["shanks"])
            pitches = res.get("auto_pitches", ())
            pstr = "[" + ",".join(f"{p:.2f}" for p in pitches) + "]"
            print(
                f"  {sid} [{res['gt_src']}]: "
                f"stage1={res['n_stage1']} lines  |  "
                f"CLEAN shanks {n}/{res['n_gt']}  |  "
                f"detected {m}/{n}  pitch={pstr}  "
                f"({dt:.1f}s)"
            )
        results.append(res)

    # ---- Per-subject summary -----------------
    print(f"\n{'='*110}\nPER-SUBJECT SUMMARY\n{'='*110}")
    print(
        f"{'subject':12s} {'gt_src':10s} {'n_clean':>7s} "
        f"{'n_match':>7s} {'n_miss':>6s} {'rate':>6s} {'stage1':>6s}"
    )
    total_clean = 0; total_match = 0
    for res in results:
        if "error" in res:
            print(f"{res['subject']:12s}  ERROR {res['error']}")
            continue
        n = len(res["shanks"]); m = sum(1 for s in res["shanks"] if s["matched"])
        total_clean += n; total_match += m
        rate = (100.0 * m / n) if n else 0.0
        print(
            f"{res['subject']:12s} {res['gt_src']:10s} "
            f"{n:>7d} {m:>7d} {n - m:>6d} "
            f"{rate:>5.1f}% {res['n_stage1']:>6d}"
        )
    rate = (100.0 * total_match / total_clean) if total_clean else 0.0
    print(
        f"\n  DATASET TOTAL: {total_match}/{total_clean} "
        f"CLEAN shanks detected = {rate:.1f}%"
    )

    # ---- Missed CLEAN shanks -----------------
    print(f"\n{'='*110}\nMISSED CLEAN SHANKS (best stage-1 match above tight window)\n{'='*110}")
    print(
        f"{'subject':10s} {'shank':10s} {'len':>5s} "
        f"{'n_blob':>6s} {'n_str':>5s} {'amp_p50':>7s} "
        f"{'best_ang':>8s} {'best_mid':>8s} {'best_n':>6s}"
    )
    missed = []
    for res in results:
        if "shanks" not in res:
            continue
        for s in res["shanks"]:
            if s["matched"]:
                continue
            missed.append(s)
            b = s["best"] or {}
            print(
                f"{s['subject']:10s} {s['shank']:10s} {s['length']:5.1f} "
                f"{s['n_blobs']:>6d} {s['n_strong']:>5d} {s['amp_p50']:>7.1f} "
                f"{b.get('ang', float('nan')):>8.2f} "
                f"{b.get('mid', float('nan')):>8.2f} "
                f"{int(b.get('n_blobs', 0)):>6d}"
            )
    print(f"\n  Total missed: {len(missed)}")

    # ---- Detected vs missed blob-stat distribution ----
    print(
        f"\n{'='*110}\nBlob-stat distribution along snapped axis "
        f"(tube r={TUBE_RADIUS_MM}mm)\n{'='*110}"
    )
    det = [s for res in results if "shanks" in res
            for s in res["shanks"] if s["matched"]]
    mis = [s for res in results if "shanks" in res
            for s in res["shanks"] if not s["matched"]]

    def _qs(arr, label):
        if not arr:
            print(f"  {label}: (n=0)")
            return
        print(
            f"  {label} (n={len(arr)}): "
            f"p10={np.percentile(arr, 10):.1f} "
            f"p50={np.percentile(arr, 50):.1f} "
            f"p90={np.percentile(arr, 90):.1f} "
            f"min={min(arr):.1f} max={max(arr):.1f}"
        )

    print("\n  DETECTED shanks:")
    _qs([s["n_strong"] for s in det], "n_strong    ")
    _qs([s["amp_p50"] for s in det if not np.isnan(s["amp_p50"])], "amp_p50     ")
    _qs([s["length"] for s in det], "axis length ")
    print("\n  MISSED shanks:")
    _qs([s["n_strong"] for s in mis], "n_strong    ")
    _qs([s["amp_p50"] for s in mis if not np.isnan(s["amp_p50"])], "amp_p50     ")
    _qs([s["length"] for s in mis], "axis length ")


if __name__ == "__main__":
    main()
