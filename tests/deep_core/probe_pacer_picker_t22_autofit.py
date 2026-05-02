"""Run the full Auto Fit pipeline on T22 (same code path as Slicer's
Auto Fit), then compare each detected trajectory's `suggested_model_id`
against the per-trajectory GT from
`tests/data/T22/T22_aligned_world_coords.txt`.

Closes the headless-vs-Slicer gap. The dataset gate
(`test_pipeline_dataset_contact_pitch_v1.py`) only checks detection
counts (recall + FP cap), not picker accuracy. This probe surfaces
exactly which model the picker chose on each Auto-Fit-detected
trajectory and matches them to GT.

Run:
    cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper && \\
      /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
      tests/deep_core/probe_pacer_picker_t22_autofit.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "CommonLib"))
sys.path.insert(0, str(ROOT / "PostopCTLocalization"))
sys.path.insert(0, str(ROOT / "tools"))

from shank_engine import PipelineRegistry, register_builtin_pipelines
from eval_seeg_localization import (  # noqa: E402
    build_detection_context,
    iter_subject_rows,
)


DATASET_ROOT = Path("/Users/ammar/Dropbox/thalamus_subjects/seeg_localization")
T22_DIR = ROOT / "tests" / "data" / "T22"
GT_FILE = T22_DIR / "T22_aligned_world_coords.txt"

MATCH_ANGLE_DEG = 10.0
MATCH_MID_MM = 8.0


def load_gt_trajectories(path: Path):
    """Return list of (name, model_id, n_contacts, start_ras, end_ras)
    where start = most-superficial contact, end = deepest contact."""
    by_traj = defaultdict(list)
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split(",")
            if len(cols) < 13:
                continue
            traj = cols[0]
            idx = int(cols[2])
            ras = [float(cols[6]), float(cols[7]), float(cols[8])]
            model = cols[12]
            by_traj[traj].append((idx, ras, model))
    out = []
    for name, rows in sorted(by_traj.items()):
        rows.sort()
        models = {m for _, _, m in rows}
        if len(models) != 1:
            continue
        deepest = np.array(rows[0][1], dtype=float)        # contact 1 (tip)
        shallowest = np.array(rows[-1][1], dtype=float)    # contact N (entry)
        out.append({
            "name": name,
            "model_id": models.pop(),
            "n_contacts": len(rows),
            "start_ras": shallowest,   # entry side
            "end_ras": deepest,        # tip side
        })
    return out


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def match_trajectories(gt_list, trajs):
    """Greedy 1-to-1 match by angle + midpoint perpendicular distance.
    Returns list of (gt_idx, traj_idx) pairs."""
    pairs = []
    for gi, g in enumerate(gt_list):
        g_axis = _unit(g["end_ras"] - g["start_ras"])
        g_mid = 0.5 * (g["start_ras"] + g["end_ras"])
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
            if ang <= MATCH_ANGLE_DEG and mid_d <= MATCH_MID_MM:
                pairs.append((ang + mid_d, gi, ti))
    pairs.sort(key=lambda p: p[0])
    used_gt = set(); used_tr = set()
    matched = []
    for _s, gi, ti in pairs:
        if gi in used_gt or ti in used_tr:
            continue
        used_gt.add(gi); used_tr.add(ti)
        matched.append((gi, ti))
    return matched


def main() -> int:
    print("Loading T22 GT…")
    gt = load_gt_trajectories(GT_FILE)
    print(f"  {len(gt)} GT trajectories: " + ", ".join(
        f"{g['name']}={g['model_id']}({g['n_contacts']})" for g in gt
    ))
    print("\nRunning Auto Fit pipeline (contact_pitch_v1) on T22…")
    rows = iter_subject_rows(DATASET_ROOT, {"T22"})
    if not rows:
        print("T22 not in dataset", file=sys.stderr)
        return 1
    registry = PipelineRegistry()
    register_builtin_pipelines(registry)
    ctx, _ = build_detection_context(
        rows[0]["ct_path"], run_id="contact_pitch_T22_probe", config={}, extras={},
    )
    ctx["contact_pitch_v1_pitch_strategy"] = "dixi"
    result = registry.run("contact_pitch_v1", ctx)
    if result.get("status") != "ok":
        print(f"Pipeline failed: {result.get('error')}", file=sys.stderr)
        return 1
    trajs = list(result.get("trajectories") or [])
    print(f"  {len(trajs)} detected trajectories")

    matched = match_trajectories(gt, trajs)
    print(f"\nMatched {len(matched)}/{len(gt)} trajectories to GT")

    print("\n--- per-trajectory comparison ---")
    print(f"{'gt':<5} {'truth':<11} {'picked':<13} {'method':<18} "
          f"{'n':>3} {'span_mm':>8} {'intra_mm':>9}  match")
    n_picker_correct = 0
    rows_for_misses = []
    for gi, ti in sorted(matched, key=lambda p: gt[p[0]]["name"]):
        g = gt[gi]
        t = trajs[ti]
        truth = g["model_id"]
        picked = str(t.get("suggested_model_id") or "")
        method = str(t.get("suggested_model_method") or "")
        n_obs = int(t.get("n_inliers") or 0)
        span = float(t.get("contact_span_mm") or 0.0)
        intra = float(t.get("intracranial_length_mm") or 0.0)
        is_match = (picked == truth)
        if is_match:
            n_picker_correct += 1
        else:
            rows_for_misses.append((g["name"], truth, picked, t))
        marker = "OK " if is_match else "MISS"
        print(f"{g['name']:<5} {truth:<11} {picked:<13} {method:<18} "
              f"{n_obs:>3} {span:>8.2f} {intra:>9.2f}  {marker}")

    n_unmatched_gt = len(gt) - len(matched)
    print(f"\nPicker accuracy: {n_picker_correct}/{len(matched)} on matched "
          f"({n_unmatched_gt} GT shanks unmatched by detector)")

    return 0 if n_picker_correct == len(matched) else 2


if __name__ == "__main__":
    sys.exit(main())
