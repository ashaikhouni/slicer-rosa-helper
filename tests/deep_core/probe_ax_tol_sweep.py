"""AX_TOL_MM sweep across the full dataset.

AX_TOL_MM is the per-step axial deviation allowed for an inlier
relative to k * pitch_seed during the walker chain extension. Same
error budget as PITCH_TOL_MM (per-peak axial position uncertainty)
but applied to each inlier instead of the seed pair.

Math (Box r=1 + sub-voxel quadratic refinement, 1 mm canonical):
  per-peak axial position error (1 sigma):
    sub-voxel residual          ~0.10 mm
    manufacturing               ~0.05 mm
    registration                ~0.10 mm
    combined                    ~0.15 mm

  2 sigma (95%)  =  0.30 mm
  2.5 sigma      =  0.38 mm
  3 sigma        =  0.45 mm

The current AX_TOL_MM = 0.7 was Ball r=2 era empirical. Math says
0.3-0.4 mm should suffice for Box r=1.

PERP_TOL_MM (1.5 mm) is dominated by real shank curvature/wobble,
not detection error, so it is NOT in scope for this sweep.

Sweep:
  0.7 (current), 0.5, 0.4, 0.3

Run:
    cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper && \\
        /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_ax_tol_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper")
sys.path.insert(0, str(ROOT / "CommonLib"))
sys.path.insert(0, str(ROOT / "PostopCTLocalization"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np

from shank_engine import PipelineRegistry, register_builtin_pipelines
from postop_ct_localization import contact_pitch_v1_fit as cpfit
from eval_seeg_localization import (
    build_detection_context,
    iter_subject_rows,
    load_reference_ground_truth_shanks,
)


DATASET_ROOT = Path("/Users/ammar/Dropbox/thalamus_subjects/seeg_localization")
EXCLUDE = {"T17", "T19", "T21"}

TOLERANCES = [0.7, 0.5, 0.4, 0.3]
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
    registry = PipelineRegistry()
    register_builtin_pipelines(registry)

    rows = iter_subject_rows(DATASET_ROOT, None)
    rows = [r for r in rows if str(r["subject_id"]) not in EXCLUDE]
    rows.sort(key=lambda r: str(r["subject_id"]))

    orig_tol = cpfit.AX_TOL_MM

    summary = []
    per_subject_lost = {}

    for tol in TOLERANCES:
        print(f"\n=== AX_TOL_MM = {tol} ===")
        cpfit.AX_TOL_MM = tol
        n_match_all = n_orph_all = n_emit_all = 0
        n_high_orph = n_med_orph = 0
        per_subject_lost[tol] = {}
        for row in rows:
            sid = str(row["subject_id"])
            gt, _ = load_reference_ground_truth_shanks(row)
            ctx, _ = build_detection_context(
                row["ct_path"], run_id=f"probe_ax{tol}_{sid}", config={}, extras={},
            )
            ctx["contact_pitch_v1_pitch_strategy"] = "auto"
            result = registry.run("contact_pitch_v1", ctx)
            if str(result.get("status", "ok")).lower() == "error":
                print(f"  {sid}: ERROR")
                continue
            trajs = list(result.get("trajectories") or [])
            matched = _greedy_match(gt, trajs)
            high_o = sum(1 for ti, t in enumerate(trajs)
                         if ti not in matched and str(t.get("confidence_label","?")) == "high")
            med_o = sum(1 for ti, t in enumerate(trajs)
                        if ti not in matched and str(t.get("confidence_label","?")) == "medium")
            n_match_all += len(matched)
            n_orph_all += len(trajs) - len(matched)
            n_emit_all += len(trajs)
            n_high_orph += high_o
            n_med_orph += med_o
            print(f"  {sid}: {len(matched)}/{len(gt)} matched, {len(trajs)-len(matched)} orphans (high={high_o}, medium={med_o})")
            if len(matched) < len(gt):
                per_subject_lost[tol][sid] = sorted(
                    {str(g.shank) for g in gt} - {info[0] for info in matched.values()}
                )
        summary.append({"tol": tol, "matched": n_match_all, "emit": n_emit_all,
                        "orph": n_orph_all, "high_o": n_high_orph, "med_o": n_med_orph})

    cpfit.AX_TOL_MM = orig_tol

    print(f"\n=== SUMMARY ===")
    print(f"{'tol':>4s} {'matched':>8s} {'emit':>5s} {'orph':>5s} {'high_o':>7s} {'med_o':>6s}")
    for s in summary:
        print(f"{s['tol']:>4.1f} {s['matched']:>8d} {s['emit']:>5d} {s['orph']:>5d} {s['high_o']:>7d} {s['med_o']:>6d}")

    print("\n=== shanks lost vs production (tol=0.7) ===")
    for tol in TOLERANCES[1:]:
        lost = per_subject_lost.get(tol, {})
        if not lost:
            print(f"  tol={tol}: no shanks lost")
        else:
            total = sum(len(v) for v in lost.values())
            print(f"  tol={tol}: {total} shanks lost")
            for sid, shanks in sorted(lost.items()):
                print(f"    {sid}: {shanks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
