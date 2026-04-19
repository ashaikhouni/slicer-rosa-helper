"""Run contact_pitch_v1 on the AMC099 CT and compare to a ROSA GT CSV.

Standalone probe — not part of unittest. Use to diagnose the
"3 missing trajectories + slow" report on AMC099.

Usage:
    cd slicer-rosa-helper
    /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \
        tests/deep_core/probe_amc099.py [--pitch-strategy auto|dixi|pmt|mixed]

The CT path and GT CSV are hard-coded from the user's memory:
    CT : /Users/ammar/Dropbox/MRI-Pipeline/inbox/brunner/FREESURFER/AMC099/NIfTI/CT/CT.nii.gz
    GT : /Users/ammar/Dropbox/tmp/ROSA_Contacts_final_trajectory_points.csv

The CSV has columns ``trajectory, point_type, x_frame_ras, y_frame_ras, ...``
with one row for ``entry`` and one for ``target`` per trajectory.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))
sys.path.insert(0, str(REPO_ROOT / "PostopCTLocalization"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import numpy as np  # noqa: E402

from eval_seeg_localization import build_detection_context  # noqa: E402
from shank_engine import PipelineRegistry, register_builtin_pipelines  # noqa: E402


DEFAULT_CT = "/Users/ammar/Dropbox/MRI-Pipeline/inbox/brunner/FREESURFER/AMC099/NIfTI/CT/CT.nii.gz"
DEFAULT_GT = "/Users/ammar/Dropbox/tmp/ROSA_Contacts_final_trajectory_points.csv"


def load_rosa_gt_csv(csv_path: str) -> list[dict]:
    """Parse the ROSA trajectory_points CSV into a list of dicts with
    ``name, entry_ras, target_ras``. One dict per trajectory name."""
    entries: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    with open(csv_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = row["trajectory"]
            pt = np.array([
                float(row["x_world_ras"]),
                float(row["y_world_ras"]),
                float(row["z_world_ras"]),
            ], dtype=float)
            if row["point_type"] == "entry":
                entries[name] = pt
            elif row["point_type"] == "target":
                targets[name] = pt
    common = sorted(set(entries).intersection(set(targets)))
    out = []
    for name in common:
        out.append({
            "name": name,
            "entry_ras": entries[name],
            "target_ras": targets[name],
        })
    return out


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def match_gt_to_trajs(
    gt: list[dict],
    trajs: list[dict],
    *,
    max_angle_deg: float = 10.0,
    max_mid_mm: float = 8.0,
) -> tuple[set[int], set[int], list[tuple[int, int, float, float]]]:
    """Greedy 1-to-1 axis-midpoint match, same as regression test."""
    pairs = []
    for gi, g in enumerate(gt):
        g_s = g["entry_ras"]
        g_e = g["target_ras"]
        g_axis = _unit(g_e - g_s)
        g_mid = 0.5 * (g_s + g_e)
        for ti, t in enumerate(trajs):
            t_s = np.asarray(t["start_ras"], dtype=float)
            t_e = np.asarray(t["end_ras"], dtype=float)
            t_axis = _unit(t_e - t_s)
            ang = float(np.degrees(np.arccos(
                min(1.0, abs(float(np.dot(g_axis, t_axis)))))))
            t_mid = 0.5 * (t_s + t_e)
            v = g_mid - t_mid
            perp = v - float(np.dot(v, t_axis)) * t_axis
            mid_d = float(np.linalg.norm(perp))
            if ang <= max_angle_deg and mid_d <= max_mid_mm:
                pairs.append((ang + mid_d, gi, ti, ang, mid_d))
    pairs.sort(key=lambda p: p[0])
    used_gt: set[int] = set()
    used_tr: set[int] = set()
    details: list[tuple[int, int, float, float]] = []
    for _s, gi, ti, ang, mid in pairs:
        if gi in used_gt or ti in used_tr:
            continue
        used_gt.add(gi)
        used_tr.add(ti)
        details.append((gi, ti, ang, mid))
    return used_gt, used_tr, details


def _nearest_distance_to_trajs(
    gt_row: dict, trajs: list[dict]
) -> tuple[float, float, int]:
    """For a missed GT, report best (angle, midpoint-perp) to any traj."""
    g_s = gt_row["entry_ras"]
    g_e = gt_row["target_ras"]
    g_axis = _unit(g_e - g_s)
    g_mid = 0.5 * (g_s + g_e)
    best = (999.0, 999.0, -1)
    for ti, t in enumerate(trajs):
        t_s = np.asarray(t["start_ras"], dtype=float)
        t_e = np.asarray(t["end_ras"], dtype=float)
        t_axis = _unit(t_e - t_s)
        ang = float(np.degrees(np.arccos(
            min(1.0, abs(float(np.dot(g_axis, t_axis)))))))
        t_mid = 0.5 * (t_s + t_e)
        v = g_mid - t_mid
        perp = v - float(np.dot(v, t_axis)) * t_axis
        mid_d = float(np.linalg.norm(perp))
        score = ang + mid_d
        if score < (best[0] + best[1]):
            best = (ang, mid_d, ti)
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct", default=DEFAULT_CT)
    parser.add_argument("--gt", default=DEFAULT_GT)
    parser.add_argument(
        "--pitch-strategy",
        default="dixi",
        choices=["dixi", "pmt_35", "pmt", "mixed", "auto"],
    )
    args = parser.parse_args()

    print(f"CT : {args.ct}")
    print(f"GT : {args.gt}")
    print(f"strategy: {args.pitch_strategy}")

    gt_rows = load_rosa_gt_csv(args.gt)
    print(f"GT trajectories: {len(gt_rows)} ({[r['name'] for r in gt_rows]})")

    registry = PipelineRegistry()
    register_builtin_pipelines(registry)

    t0 = time.perf_counter()
    ctx, img = build_detection_context(
        args.ct, run_id=f"amc099_{args.pitch_strategy}",
        config={}, extras={},
    )
    ctx["contact_pitch_v1_pitch_strategy"] = args.pitch_strategy
    ctx["logger"] = lambda msg: print(f"[fit] {msg}")
    t_ctx = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = registry.run("contact_pitch_v1", ctx)
    t_run = time.perf_counter() - t0

    print(f"\nload+context: {t_ctx:.2f} s")
    print(f"pipeline run: {t_run:.2f} s")

    if result.get("status") != "ok":
        print("PIPELINE FAILED")
        print(result.get("error"))
        return 1

    trajs = list(result.get("trajectories") or [])
    print(f"detected trajectories: {len(trajs)}")
    for ti, t in enumerate(trajs):
        s = np.asarray(t["start_ras"], dtype=float)
        e = np.asarray(t["end_ras"], dtype=float)
        length = float(np.linalg.norm(e - s))
        src = t.get("source", "?")
        print(
            f"  [{ti:02d}] src={src:<10} len={length:5.1f} mm  "
            f"start={s.round(1).tolist()}  end={e.round(1).tolist()}"
        )

    used_gt, used_tr, details = match_gt_to_trajs(gt_rows, trajs)
    n_matched = len(used_gt)
    n_fp = len(trajs) - n_matched

    print(f"\nmatched {n_matched}/{len(gt_rows)}, FP = {n_fp}")

    missed = [i for i in range(len(gt_rows)) if i not in used_gt]
    if missed:
        print("\nMISSED GT TRAJECTORIES:")
        for gi in missed:
            g = gt_rows[gi]
            length = float(np.linalg.norm(g["target_ras"] - g["entry_ras"]))
            ang, mid, ti = _nearest_distance_to_trajs(g, trajs)
            print(
                f"  {g['name']:<6} len={length:5.1f} mm  "
                f"entry={g['entry_ras'].round(1).tolist()}  "
                f"target={g['target_ras'].round(1).tolist()}  "
                f"nearest det [{ti}] angle={ang:.1f}°, mid_perp={mid:.1f} mm"
            )

    used_tr_set = set(used_tr)
    fps = [ti for ti in range(len(trajs)) if ti not in used_tr_set]
    if fps:
        print(f"\nFALSE POSITIVE detections ({len(fps)}):")
        for ti in fps:
            t = trajs[ti]
            s = np.asarray(t["start_ras"], dtype=float)
            e = np.asarray(t["end_ras"], dtype=float)
            length = float(np.linalg.norm(e - s))
            src = t.get("source", "?")
            print(f"  [{ti:02d}] src={src:<10} len={length:5.1f} mm  "
                  f"start={s.round(1).tolist()}  end={e.round(1).tolist()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
