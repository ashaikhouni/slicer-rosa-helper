"""Disambiguate the T4 threshold-500 'miss' of LSFG against Ammar's
redone T4 GT (trajectory endpoints, one entry + target per shank).

Question
--------
At threshold 300, T4 matches 14/16 on the OLD nominal-contact-axis GT.
At threshold 500, T4 drops to 13/16 (also loses LSFG). Was that a real
regression, or a GT-axis-noise artifact? Ammar redid the T4 GT
trajectory endpoints; compare both thresholds against the new GT and
report per-shank angle + midpoint error.

Run
---
    /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_t4_threshold_vs_new_gt.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper")
sys.path.insert(0, str(ROOT / "CommonLib"))
sys.path.insert(0, str(ROOT / "PostopCTLocalization"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np

from rosa_detect.service import run_contact_pitch_v1
from rosa_detect import contact_pitch_v1_fit as cpfit
from eval_seeg_localization import build_detection_context, iter_subject_rows

DATASET_ROOT = Path(
    "/Users/ammar/Dropbox/thalamus_subjects/seeg_localization"
)
NEW_GT_CSV = (
    DATASET_ROOT / "contact_label_dataset" / "rosa_helper_import" / "T4"
    / "ROSA_Contacts_T4_final_trajectory_points.csv"
)

THRESHOLDS = [300.0, 500.0]
MATCH_ANGLE_DEG = 10.0
MATCH_MID_MM = 8.0


def _load_new_gt() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {shank: (entry_ras, target_ras)}."""
    by_shank: dict[str, dict] = defaultdict(dict)
    with NEW_GT_CSV.open() as f:
        rd = csv.DictReader(f)
        for row in rd:
            shank = row["trajectory"]
            pt = np.array(
                [float(row["x_world_ras"]),
                 float(row["y_world_ras"]),
                 float(row["z_world_ras"])],
                dtype=float,
            )
            by_shank[shank][row["point_type"]] = pt
    return {
        s: (d["entry"], d["target"])
        for s, d in by_shank.items()
        if "entry" in d and "target" in d
    }


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def _axis_midpoint(a: np.ndarray, b: np.ndarray):
    axis = _unit(b - a)
    mid = 0.5 * (a + b)
    return axis, mid


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(u, v)))))))


def _perp_mid_distance(g_mid, t_mid, t_axis) -> float:
    v = g_mid - t_mid
    perp = v - (v @ t_axis) * t_axis
    return float(np.linalg.norm(perp))


def _run_t4(threshold: float):
    # Temporarily set threshold without mutating source.
    orig = cpfit.LOG_BLOB_THRESHOLD
    cpfit.LOG_BLOB_THRESHOLD = float(threshold)
    try:
        rows = iter_subject_rows(DATASET_ROOT, {"T4"})
        assert rows, "T4 not in manifest"
        row = rows[0]
        ctx, _ = build_detection_context(
            row["ct_path"], run_id=f"probe_t4_thr{int(threshold)}",
            config={}, extras={},
        )
        ctx["contact_pitch_v1_pitch_strategy"] = "dixi"
        result = run_contact_pitch_v1(ctx)
        return list(result.get("trajectories") or [])
    finally:
        cpfit.LOG_BLOB_THRESHOLD = orig


def _match_report(trajs, gts, label: str):
    print(f"\n{'=' * 90}")
    print(f"THRESHOLD {label}")
    print("=" * 90)

    # Per-GT: best-matching detected trajectory.
    print(
        f"\nPer-GT-shank best-match report "
        f"(match window: angle <= {MATCH_ANGLE_DEG}, mid <= {MATCH_MID_MM}):"
    )
    print(
        f"{'shank':8s} {'best_t':>7s}  {'angle':>7s} {'mid_d':>7s}  "
        f"{'MATCH':>6s}  {'n_inl':>5s} {'span':>6s} {'len':>6s}  {'src':>6s}"
    )
    matched = 0
    near_miss = []
    for shank, (g_s, g_e) in sorted(gts.items()):
        g_axis, g_mid = _axis_midpoint(g_s, g_e)
        best = None
        for ti, t in enumerate(trajs):
            t_axis, t_mid = _axis_midpoint(
                np.asarray(t["start_ras"]), np.asarray(t["end_ras"]),
            )
            ang = _angle_deg(g_axis, t_axis)
            mid_d = _perp_mid_distance(g_mid, t_mid, t_axis)
            # Pick the single best (lowest angle + midpoint) trajectory.
            score = ang + mid_d
            if best is None or score < best["score"]:
                best = {
                    "ti": ti, "ang": ang, "mid_d": mid_d, "score": score,
                    "n_inl": int(t.get("n_inliers", 0)),
                    "span": float(t.get("contact_span_mm", 0.0)),
                    "length": float(t.get("length_mm", 0.0)),
                }
        if best is None:
            print(f"{shank:8s} {'-':>7s}  {'-':>7s} {'-':>7s}  {'NO TRAJ':>6s}")
            continue
        match = best["ang"] <= MATCH_ANGLE_DEG and best["mid_d"] <= MATCH_MID_MM
        if match:
            matched += 1
        elif best["ang"] <= MATCH_ANGLE_DEG + 5 and best["mid_d"] <= MATCH_MID_MM + 5:
            near_miss.append((shank, best))
        mark = "MATCH" if match else (
            "near" if best["ang"] <= 15 and best["mid_d"] <= 13 else "miss"
        )
        src = str(trajs[best["ti"]].get("source") or "?")
        print(
            f"{shank:8s} t{best['ti']:>5d}   "
            f"{best['ang']:>6.2f}  {best['mid_d']:>6.2f}   "
            f"{mark:>6s}  {best['n_inl']:>5d} {best['span']:>6.1f} "
            f"{best['length']:>6.1f}  {src:>6s}"
        )
    print(f"\nTotal matched (greedy best per GT): {matched}/{len(gts)}")
    if near_miss:
        print("\nNEAR-MISSES (could be real detection failing noisy match window):")
        for shank, b in near_miss:
            print(
                f"  {shank:8s}: ang={b['ang']:.2f}° (window {MATCH_ANGLE_DEG})  "
                f"mid_d={b['mid_d']:.2f}mm (window {MATCH_MID_MM})"
            )

    # Orphan detections — emitted but no GT within match window.
    print(f"\nEmitted trajectories: {len(trajs)}")
    orphans = []
    for ti, t in enumerate(trajs):
        t_axis, t_mid = _axis_midpoint(
            np.asarray(t["start_ras"]), np.asarray(t["end_ras"]),
        )
        best_g = None
        for shank, (g_s, g_e) in gts.items():
            g_axis, g_mid = _axis_midpoint(g_s, g_e)
            ang = _angle_deg(g_axis, t_axis)
            mid_d = _perp_mid_distance(g_mid, t_mid, t_axis)
            score = ang + mid_d
            if best_g is None or score < best_g["score"]:
                best_g = {
                    "shank": shank, "ang": ang, "mid_d": mid_d, "score": score,
                }
        if best_g["ang"] > MATCH_ANGLE_DEG or best_g["mid_d"] > MATCH_MID_MM:
            orphans.append((ti, best_g, t))
    print(f"Orphan detections (no GT within window): {len(orphans)}")
    for ti, b, t in orphans:
        print(
            f"  t{ti:>2d}: closest GT={b['shank']} "
            f"(ang={b['ang']:.2f}° mid_d={b['mid_d']:.2f}mm)  "
            f"n_inl={int(t.get('n_inliers', 0))}  "
            f"span={float(t.get('contact_span_mm', 0)):.1f}  "
            f"len={float(t.get('length_mm', 0)):.1f}"
        )


def main():
    gts = _load_new_gt()
    print(f"Loaded NEW GT: {len(gts)} shanks — {sorted(gts.keys())}")

    for thr in THRESHOLDS:
        trajs = _run_t4(thr)
        _match_report(trajs, gts, f"{int(thr)} (new GT)")


if __name__ == "__main__":
    main()
