"""Snap GT axes via guided-fit, then run the stage-1 LoG-blob probe on
the snapped axes across all dataset subjects.

For each subject:
  1. Load GT shanks via eval_seeg_localization.load_reference_ground_truth_shanks.
  2. Compute LoG/hull/bolts once via guided_fit_engine.compute_features.
  3. For each GT shank, call guided_fit_engine.fit_trajectory to snap
     the rough GT axis to the imaged metal. Classify the result as:
       - CLEAN   — success, n_inliers ≥ 5, tight_refit=True,
                   lateral_shift ≤ 2.0, angle ≤ 5.0
       - DEGRADED — success but some criterion missed (still a usable
                   axis, but GT was inaccurate)
       - FAIL    — success=False → GT cannot be trusted; flag for
                   manual redo
  4. For the CLEAN and DEGRADED snaps, extract LoG blobs inside a
     2 mm tube around the snapped axis and report per-shank stats
     (n strong, amp_p50, spacing std/cv).

Outputs three tables:
  - Per-subject summary (counts of CLEAN / DEGRADED / FAIL).
  - All FAILS — shanks that need manual GT redo.
  - Wire-class candidates among CLEAN shanks — shanks where the
    snapped axis has unusually few strong blobs or weak amplitudes
    (the pattern originally hypothesized for T4 LSFG but which the
    T4 probe empirically rejected).

Run
---
    /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_snapped_gt_blob_stats.py [subject ...]

If subjects are provided as args, only those are processed. Otherwise
iterates the full manifest (T19 and T21 are skipped; they have
unreliable manual GT per `reference_seeg_dataset` memory).
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

from rosa_detect import contact_pitch_v1_fit as cpfit
from rosa_detect import guided_fit_engine as gfe
from eval_seeg_localization import (
    iter_subject_rows, image_ijk_ras_matrices,
    load_reference_ground_truth_shanks,
)

DATASET_ROOT = Path("/Users/ammar/Dropbox/thalamus_subjects/seeg_localization")
EXCLUDE = {"T19", "T21"}  # unreliable manual GT per memory

# Subjects with a redone (CT-accurate) GT in the rosa_helper_import folder.
# Key = subject_id, value = CSV path with columns:
#   trajectory, point_type (entry|target), x_world_ras, y_world_ras, z_world_ras
REDONE_GT_CSVS = {
    "T4": DATASET_ROOT / "contact_label_dataset" / "rosa_helper_import"
          / "T4" / "ROSA_Contacts_T4_final_trajectory_points.csv",
}


@dataclass
class _RedoneShank:
    subject_id: str
    shank: str
    start_ras: tuple
    end_ras: tuple


def _load_redone_csv(csv_path, subject_id):
    by_shank = defaultdict(dict)
    with Path(csv_path).open() as f:
        for row in csv.DictReader(f):
            pt = (
                float(row["x_world_ras"]),
                float(row["y_world_ras"]),
                float(row["z_world_ras"]),
            )
            by_shank[row["trajectory"]][row["point_type"]] = pt
    out = []
    for shank, ends in sorted(by_shank.items()):
        if "entry" not in ends or "target" not in ends:
            continue
        out.append(_RedoneShank(
            subject_id=subject_id, shank=shank,
            start_ras=ends["entry"], end_ras=ends["target"],
        ))
    return out

# Snap quality thresholds
CLEAN_MIN_INLIERS = 5
CLEAN_MAX_LATERAL_MM = 2.0
CLEAN_MAX_ANGLE_DEG = 5.0

# Blob probe thresholds
TUBE_RADIUS_MM = 2.0
STRONG_AMP = 500.0


def _axis(a, b):
    d = b - a
    L = float(np.linalg.norm(d))
    return d / L, L


def _blobs_in_tube(pts_ras, amps, start, end, radius_mm):
    u, L = _axis(np.asarray(start), np.asarray(end))
    v = pts_ras - np.asarray(start)[None, :]
    axial = v @ u
    perp = v - axial[:, None] * u[None, :]
    perp_d = np.linalg.norm(perp, axis=1)
    keep = (axial >= -2.0) & (axial <= L + 2.0) & (perp_d <= radius_mm)
    ax = axial[keep]; am = amps[keep]
    order = np.argsort(ax)
    return ax[order], am[order]


def _spacing(depths, amps, strong_min):
    strong = depths[amps >= strong_min]
    if strong.size < 2:
        return None
    dedup = [float(strong[0])]
    for d in strong[1:]:
        if abs(float(d) - dedup[-1]) > 1.2:
            dedup.append(float(d))
    if len(dedup) < 2:
        return None
    sp = np.diff(np.array(dedup))
    return dict(
        n=int((amps >= strong_min).sum()),
        n_dedup=len(dedup),
        med=float(np.median(sp)),
        mean=float(np.mean(sp)),
        std=float(np.std(sp)),
        cv=float(np.std(sp) / np.mean(sp)) if np.mean(sp) > 0 else float("nan"),
    )


def _classify(snap):
    """CLEAN / DEGRADED / FAIL."""
    if not snap.get("success"):
        return "FAIL", snap.get("reason", "unknown")
    n_in = int(snap.get("n_inliers", 0))
    lat = float(snap.get("lateral_shift_mm", 0.0))
    ang = float(snap.get("angle_deg", 0.0))
    tight = bool(snap.get("tight_refit", False))
    if (n_in >= CLEAN_MIN_INLIERS and tight
            and lat <= CLEAN_MAX_LATERAL_MM
            and ang <= CLEAN_MAX_ANGLE_DEG):
        return "CLEAN", None
    reasons = []
    if n_in < CLEAN_MIN_INLIERS:
        reasons.append(f"n_in={n_in}")
    if not tight:
        reasons.append("not_tight")
    if lat > CLEAN_MAX_LATERAL_MM:
        reasons.append(f"lat={lat:.2f}")
    if ang > CLEAN_MAX_ANGLE_DEG:
        reasons.append(f"ang={ang:.2f}")
    return "DEGRADED", ",".join(reasons)


def _run_subject(row):
    sid = row["subject_id"]
    ct_path = row["ct_path"]
    gt_source = "standard"
    if sid in REDONE_GT_CSVS and REDONE_GT_CSVS[sid].exists():
        try:
            gt_shanks = _load_redone_csv(REDONE_GT_CSVS[sid], sid)
            gt_source = f"redone ({REDONE_GT_CSVS[sid].name})"
        except Exception as exc:
            return dict(subject=sid, error=f"redone GT load: {exc!r}")
    else:
        try:
            gt_shanks, _gt_src = load_reference_ground_truth_shanks(row, None)
        except Exception as exc:
            return dict(subject=sid, error=f"GT load: {exc!r}")
    if not gt_shanks:
        return dict(subject=sid, error="no GT shanks")
    print(f"    [{sid}] GT source: {gt_source} ({len(gt_shanks)} shanks)")

    try:
        img = sitk.ReadImage(ct_path)
        img_clamp = sitk.Clamp(
            img, lowerBound=-1024.0, upperBound=cpfit.HU_CLIP_MAX,
        )
        ijk_to_ras_mat, ras_to_ijk_mat = image_ijk_ras_matrices(img_clamp)
        ijk_to_ras_mat = np.asarray(ijk_to_ras_mat, dtype=float)
        ras_to_ijk_mat = np.asarray(ras_to_ijk_mat, dtype=float)
        features = gfe.compute_features(img_clamp, ijk_to_ras_mat)
    except Exception as exc:
        return dict(subject=sid, error=f"preprocess: {exc!r}")

    pts_ras = features["blob_pts_ras"]
    amps = features["blob_amps"]

    per_shank = []
    for gt in gt_shanks:
        rec = dict(
            subject=sid,
            shank=gt.shank,
            gt_start=np.asarray(gt.start_ras, dtype=float),
            gt_end=np.asarray(gt.end_ras, dtype=float),
        )
        try:
            snap = gfe.fit_trajectory(
                np.asarray(gt.start_ras), np.asarray(gt.end_ras),
                features, ijk_to_ras_mat, ras_to_ijk_mat,
            )
        except Exception as exc:
            snap = {"success": False, "reason": f"fit raised {exc!r}"}
        cls, reason = _classify(snap)
        rec["class"] = cls
        rec["reason"] = reason
        rec["snap_angle"] = float(snap.get("angle_deg", np.nan))
        rec["snap_lateral"] = float(snap.get("lateral_shift_mm", np.nan))
        rec["snap_n_in"] = int(snap.get("n_inliers", 0))
        rec["snap_tight"] = bool(snap.get("tight_refit", False))
        rec["gt_source"] = gt_source

        if cls in ("CLEAN", "DEGRADED"):
            start = np.asarray(snap["start_ras"], dtype=float)
            end = np.asarray(snap["end_ras"], dtype=float)
            rec["snap_start"] = start
            rec["snap_end"] = end
            rec["snap_length"] = float(np.linalg.norm(end - start))
            depths, ain = _blobs_in_tube(pts_ras, amps, start, end, TUBE_RADIUS_MM)
            rec["n_blobs_tube"] = int(depths.size)
            rec["amp_p50"] = float(np.percentile(ain, 50)) if ain.size else float("nan")
            rec["amp_p90"] = float(np.percentile(ain, 90)) if ain.size else float("nan")
            rec["n_strong"] = int((ain >= STRONG_AMP).sum())
            st = _spacing(depths, ain, STRONG_AMP)
            rec["sp_med"] = st["med"] if st else float("nan")
            rec["sp_mean"] = st["mean"] if st else float("nan")
            rec["sp_std"] = st["std"] if st else float("nan")
            rec["sp_cv"] = st["cv"] if st else float("nan")
        per_shank.append(rec)
    return dict(subject=sid, shanks=per_shank, n_gt=len(gt_shanks))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    subjects_filter = set(args) if args else None
    rows = iter_subject_rows(DATASET_ROOT, subjects_filter)
    rows = [r for r in rows if str(r["subject_id"]) not in EXCLUDE]
    print(f"Processing {len(rows)} subjects (excluding {sorted(EXCLUDE)})")

    all_results = []
    for row in rows:
        sid = row["subject_id"]
        t0 = time.time()
        res = _run_subject(row)
        dt = time.time() - t0
        if "error" in res:
            print(f"  {sid}: ERROR — {res['error']}  ({dt:.1f}s)")
        else:
            cls_counts = {"CLEAN": 0, "DEGRADED": 0, "FAIL": 0}
            for sh in res["shanks"]:
                cls_counts[sh["class"]] += 1
            print(
                f"  {sid}: {res['n_gt']:>2d} shanks — "
                f"CLEAN {cls_counts['CLEAN']:>2d} / "
                f"DEGRADED {cls_counts['DEGRADED']:>2d} / "
                f"FAIL {cls_counts['FAIL']:>2d}   ({dt:.1f}s)"
            )
        all_results.append(res)

    # ---- Per-subject summary table -----------------------
    print(f"\n{'='*110}\nPER-SUBJECT SUMMARY\n{'='*110}")
    print(f"{'subject':12s} {'n_gt':>4s} {'CLEAN':>6s} {'DEGRD':>6s} {'FAIL':>5s} "
          f"{'frac_usable':>12s}  flag")
    bad_subjects = []
    for res in all_results:
        if "error" in res:
            print(f"{res['subject']:12s}  ERROR: {res['error']}")
            bad_subjects.append((res["subject"], "error"))
            continue
        shs = res["shanks"]
        c = sum(1 for s in shs if s["class"] == "CLEAN")
        d = sum(1 for s in shs if s["class"] == "DEGRADED")
        f = sum(1 for s in shs if s["class"] == "FAIL")
        n = len(shs)
        frac = (c + d) / max(1, n)
        flag = ""
        if f > 0: flag += "HAS_FAILS "
        if frac < 0.80: flag += "LOW_USABLE "
        print(
            f"{res['subject']:12s} {n:>4d} {c:>6d} {d:>6d} {f:>5d} "
            f"{100*frac:>11.1f}%  {flag}"
        )
        if flag:
            bad_subjects.append((res["subject"], flag.strip()))

    # ---- FAIL listing --------------------------------
    print(f"\n{'='*110}\nFAILED GT SNAPS (candidates for manual redo)\n{'='*110}")
    print(f"{'subject':12s} {'shank':10s} {'reason':70s}")
    total_fails = 0
    for res in all_results:
        if "shanks" not in res:
            continue
        for sh in res["shanks"]:
            if sh["class"] != "FAIL":
                continue
            total_fails += 1
            reason = sh.get("reason") or "(no reason)"
            print(f"{res['subject']:12s} {sh['shank']:10s} {reason[:70]}")
    print(f"\n  Total FAILs across dataset: {total_fails}")

    # ---- DEGRADED listing (usable-ish but GT was off) ----------
    print(f"\n{'='*110}\nDEGRADED SNAPS (GT was off; snap recovered)\n{'='*110}")
    print(
        f"{'subject':12s} {'shank':10s} {'n_in':>4s} {'lat_mm':>7s} "
        f"{'ang':>6s} {'tight':>5s}  {'reason':40s}"
    )
    n_degraded = 0
    for res in all_results:
        if "shanks" not in res:
            continue
        for sh in res["shanks"]:
            if sh["class"] != "DEGRADED":
                continue
            n_degraded += 1
            print(
                f"{res['subject']:12s} {sh['shank']:10s} "
                f"{sh['snap_n_in']:>4d} {sh['snap_lateral']:>7.2f} "
                f"{sh['snap_angle']:>6.2f} {str(sh['snap_tight']):>5s}  "
                f"{str(sh.get('reason') or '')[:40]}"
            )
    print(f"\n  Total DEGRADED: {n_degraded}")

    # ---- CLEAN blob-stat quantiles across the dataset ---------
    print(
        f"\n{'='*110}\nCLEAN-shank LoG-blob statistics (snapped axis, "
        f"tube r={TUBE_RADIUS_MM}mm)\n{'='*110}"
    )
    clean_rows = []
    for res in all_results:
        if "shanks" not in res:
            continue
        for sh in res["shanks"]:
            if sh["class"] == "CLEAN":
                clean_rows.append(sh)
    print(f"  n = {len(clean_rows)} clean shanks")
    if clean_rows:
        ns = np.array([sh["n_strong"] for sh in clean_rows], dtype=float)
        p50 = np.array([sh["amp_p50"] for sh in clean_rows], dtype=float)
        cv = np.array([sh["sp_cv"] for sh in clean_rows], dtype=float)
        cv_valid = cv[~np.isnan(cv)]
        print(
            f"  n_strong: min={int(np.min(ns))} p10={int(np.percentile(ns, 10))} "
            f"p50={int(np.percentile(ns, 50))} p90={int(np.percentile(ns, 90))} "
            f"max={int(np.max(ns))}"
        )
        print(
            f"  amp_p50: p10={np.percentile(p50, 10):.1f} "
            f"p50={np.percentile(p50, 50):.1f} p90={np.percentile(p50, 90):.1f}"
        )
        if cv_valid.size > 0:
            print(
                f"  spacing cv: p10={np.percentile(cv_valid, 10):.2f} "
                f"p50={np.percentile(cv_valid, 50):.2f} "
                f"p90={np.percentile(cv_valid, 90):.2f}"
            )

        # Wire-class candidates among CLEAN — few strong blobs OR weak p50.
        print(
            f"\n  Wire-class candidates among CLEAN shanks "
            f"(n_strong ≤ 4 OR amp_p50 < 400):"
        )
        print(
            f"  {'subject':12s} {'shank':10s} {'n_str':>5s} {'amp_p50':>8s} "
            f"{'sp_cv':>6s} {'lat_mm':>7s}"
        )
        n_wire = 0
        for sh in clean_rows:
            if sh["n_strong"] <= 4 or (not np.isnan(sh["amp_p50"]) and sh["amp_p50"] < 400):
                n_wire += 1
                print(
                    f"  {sh['subject']:12s} {sh['shank']:10s} "
                    f"{sh['n_strong']:>5d} "
                    f"{sh['amp_p50']:>8.1f} "
                    f"{sh['sp_cv']:>6.2f} "
                    f"{sh['snap_lateral']:>7.2f}"
                )
        print(f"  Total wire-class-ish (snap-CLEAN): {n_wire} / {len(clean_rows)}")

    # ---- Roll-up for user -----------------------------
    print(f"\n{'='*110}\nSUBJECTS RECOMMENDED FOR MANUAL GT REVIEW\n{'='*110}")
    if bad_subjects:
        for sid, flag in bad_subjects:
            print(f"  {sid}: {flag}")
    else:
        print("  (none — every subject has ≥80% usable GT)")


if __name__ == "__main__":
    main()
