"""Per-inlier amplitude distribution probe for matched lines.

The LoG-threshold sweep shows recall holds at 295/295 even at thr=600,
but my "Pt/Ir contacts hit 1500-3000" claim wasn't grounded in this
dataset's measurements. To ground the threshold choice, dump the
PER-INLIER |LoG| amplitudes for every matched line: each inlier is one
detected contact, so the array tells us how strong the weakest contact
on a real shank actually is. The minimum per-inlier amp across all
matched lines is the empirical lower bound for safe LOG_BLOB_THRESHOLD.

Output:
  - Across-dataset distribution: p1/p5/p10/p50 of per-inlier |LoG| on
    matched lines, plus the overall minimum
  - Per-subject worst-shank minimum
  - The 30 weakest individual inliers across the whole dataset

Run:
    cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper && \\
        /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_inlier_amp_distribution.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper")
sys.path.insert(0, str(ROOT / "CommonLib"))
sys.path.insert(0, str(ROOT / "PostopCTLocalization"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np

from rosa_detect.service import run_contact_pitch_v1
from eval_seeg_localization import (
    build_detection_context,
    iter_subject_rows,
    load_reference_ground_truth_shanks,
)


DATASET_ROOT = Path("/Users/ammar/Dropbox/thalamus_subjects/seeg_localization")
EXCLUDE = {"T17", "T19", "T21"}

MATCH_ANGLE_DEG = 10.0
MATCH_MID_MM = 8.0


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def _greedy_match(gt_shanks, trajs):
    pairs = []
    for gi, g in enumerate(gt_shanks):
        g_s = np.asarray(g.start_ras, dtype=float)
        g_e = np.asarray(g.end_ras, dtype=float)
        g_axis = _unit(g_e - g_s)
        g_mid = 0.5 * (g_s + g_e)
        for ti, t in enumerate(trajs):
            t_s = np.asarray(t["start_ras"], dtype=float)
            t_e = np.asarray(t["end_ras"], dtype=float)
            t_axis = _unit(t_e - t_s)
            ang = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(g_axis, t_axis)))))))
            t_mid = 0.5 * (t_s + t_e)
            d = g_mid - t_mid
            p = d - (d @ t_axis) * t_axis
            mid_d = float(np.linalg.norm(p))
            if ang <= MATCH_ANGLE_DEG and mid_d <= MATCH_MID_MM:
                pairs.append((ang + mid_d, gi, ti, ang, mid_d))
    pairs.sort(key=lambda p: p[0])
    used_g, used_t = set(), set()
    matched_ti = {}
    for _s, gi, ti, ang, mid_d in pairs:
        if gi in used_g or ti in used_t:
            continue
        used_g.add(gi); used_t.add(ti)
        matched_ti[ti] = (str(gt_shanks[gi].shank), ang, mid_d)
    return matched_ti


def main():
    rows = iter_subject_rows(DATASET_ROOT, None)
    rows = [r for r in rows if str(r["subject_id"]) not in EXCLUDE]
    rows.sort(key=lambda r: str(r["subject_id"]))

    # Per-inlier amps from every matched line, plus per-line worst-min.
    all_inlier_amps: list[float] = []
    weakest_inliers: list[tuple[float, str, str, int, int]] = []
    # (amp, subject, shank, n_inliers_in_line, mean_amp_in_line)
    per_subject_worst: dict[str, tuple[float, str]] = {}

    print(f"running pipeline on {len(rows)} subjects (auto strategy)\n")
    for row in rows:
        sid = str(row["subject_id"])
        gt, _ = load_reference_ground_truth_shanks(row)
        ctx, _ = build_detection_context(
            row["ct_path"], run_id=f"probe_amp_{sid}", config={}, extras={},
        )
        ctx["contact_pitch_v1_pitch_strategy"] = "auto"
        result = run_contact_pitch_v1(ctx)
        if str(result.get("status", "ok")).lower() == "error":
            print(f"  {sid}: ERROR")
            continue
        trajs = list(result.get("trajectories") or [])
        matched = _greedy_match(gt, trajs)
        worst_amp = float("inf")
        worst_shank = ""
        for ti, t in enumerate(trajs):
            if ti not in matched:
                continue
            shank = matched[ti][0]
            amps = t.get("inlier_amps")
            if amps is None:
                continue
            amps = np.asarray(amps, dtype=float)
            if amps.size == 0:
                continue
            mean = float(amps.mean())
            for a in amps:
                all_inlier_amps.append(float(a))
                weakest_inliers.append((float(a), sid, shank, int(amps.size), mean))
            line_min = float(amps.min())
            if line_min < worst_amp:
                worst_amp = line_min
                worst_shank = shank
        per_subject_worst[sid] = (worst_amp, worst_shank)
        print(f"  {sid}: {len(matched)}/{len(gt)} matched, worst inlier amp = {worst_amp:.1f} on {worst_shank}")

    arr = np.asarray(all_inlier_amps, dtype=float)
    print(f"\n=== per-inlier amplitude distribution across {arr.size} matched-line inliers ===")
    print(f"  min      = {arr.min():.1f}")
    print(f"  p1       = {np.percentile(arr, 1):.1f}")
    print(f"  p5       = {np.percentile(arr, 5):.1f}")
    print(f"  p10      = {np.percentile(arr, 10):.1f}")
    print(f"  p25      = {np.percentile(arr, 25):.1f}")
    print(f"  p50      = {np.percentile(arr, 50):.1f}")
    print(f"  p75      = {np.percentile(arr, 75):.1f}")
    print(f"  p90      = {np.percentile(arr, 90):.1f}")
    print(f"  p95      = {np.percentile(arr, 95):.1f}")
    print(f"  p99      = {np.percentile(arr, 99):.1f}")
    print(f"  max      = {arr.max():.1f}")

    print(f"\n=== per-subject worst-line worst-inlier amp ===")
    for sid in sorted(per_subject_worst):
        amp, shank = per_subject_worst[sid]
        print(f"  {sid}: {amp:>7.1f} on {shank}")

    print(f"\n=== 30 weakest individual inliers across the dataset ===")
    weakest_inliers.sort(key=lambda x: x[0])
    print(f"{'amp':>8s}  {'subj':>4s}  {'shank':>10s}  {'line_n':>6s}  {'line_mean':>9s}")
    for amp, sid, shank, n, mean in weakest_inliers[:30]:
        print(f"{amp:>8.1f}  {sid:>4s}  {shank:>10s}  {n:>6d}  {mean:>9.1f}")

    # Threshold-survival counts: how many inliers would each candidate
    # threshold drop below?
    print(f"\n=== inlier-survival counts at candidate LOG_BLOB_THRESHOLD values ===")
    print(f"{'threshold':>10s}  {'inliers >=':>11s}  {'dropped':>8s}  {'frac':>7s}")
    for thr in [300, 350, 400, 450, 500, 550, 600, 650, 700]:
        kept = int((arr >= thr).sum())
        dropped = int((arr < thr).sum())
        print(f"{thr:>10d}  {kept:>11d}  {dropped:>8d}  {dropped/arr.size:>7.2%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
