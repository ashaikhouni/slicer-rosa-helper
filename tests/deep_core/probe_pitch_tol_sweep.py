"""PITCH_TOL_MM sweep across the full dataset.

Math (kernel-aware error budget for inter-contact pitch detection at
1 mm canonical spacing with Box r=1 + sub-voxel quadratic refinement):

  per-peak position error (1sigma):
    sub-voxel residual  ~0.10 mm
    manufacturing       ~0.05 mm
    registration        ~0.10 mm
    combined            ~0.15 mm

  inter-contact distance error (1sigma):
    sigma_d = sqrt(2) * 0.15  ~0.21 mm

  tolerance windows:
    2.0 sigma (95%)    0.42 mm
    2.5 sigma (99%)    0.53 mm
    3.0 sigma (99.7%)  0.63 mm

The current PITCH_TOL_MM = 0.5 was empirical (Ball r=2 era — kernel
snap on (+-1, +-1, +-2) family added ~0.5-1 mm of position error).
Box r=1 doesn't kernel-snap at 3.5 mm pitch, so the math says
0.3-0.4 mm should be sufficient.

Sweep:
  - 0.5 (current)
  - 0.4 (2 sigma)
  - 0.3 (~1.5 sigma)
  - 0.2 (~1 sigma; aggressive)

Report: matched count, total emissions, orphan total/medium/high.

Run:
    cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper && \\
        /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_pitch_tol_sweep.py
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

TOLERANCES = [0.5, 0.4, 0.3, 0.2]
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


def _summarize(trajs, gt):
    matched = _greedy_match(gt, trajs)
    bands_o = {"high":0,"medium":0,"low":0}
    for ti, t in enumerate(trajs):
        if ti not in matched:
            cl = str(t.get("confidence_label","?"))
            bands_o[cl] = bands_o.get(cl,0)+1
    return matched, bands_o


def main():
    registry = PipelineRegistry()
    register_builtin_pipelines(registry)

    rows = iter_subject_rows(DATASET_ROOT, None)
    rows = [r for r in rows if str(r["subject_id"]) not in EXCLUDE]
    rows.sort(key=lambda r: str(r["subject_id"]))

    orig_tol = cpfit.PITCH_TOL_MM

    summary_rows = []
    per_subject_lost = {}

    for tol in TOLERANCES:
        print(f"\n{'='*78}\nPITCH_TOL_MM = {tol}\n{'='*78}")
        cpfit.PITCH_TOL_MM = tol

        n_match_all = 0
        n_orph_all = 0
        n_emit_all = 0
        n_high_orph = 0
        n_med_orph = 0
        per_subject_lost[tol] = {}

        for row in rows:
            sid = str(row["subject_id"])
            gt, _ = load_reference_ground_truth_shanks(row)
            ctx, _ = build_detection_context(
                row["ct_path"],
                run_id=f"probe_tol{tol}_{sid}",
                config={},
                extras={},
            )
            ctx["contact_pitch_v1_pitch_strategy"] = "auto"
            result = registry.run("contact_pitch_v1", ctx)
            if str(result.get("status", "ok")).lower() == "error":
                print(f"  {sid}: ERROR")
                continue
            trajs = list(result.get("trajectories") or [])
            matched, bands_o = _summarize(trajs, gt)
            n_emit = len(trajs)
            n_match = len(matched)
            n_orph = n_emit - n_match
            n_match_all += n_match
            n_orph_all += n_orph
            n_emit_all += n_emit
            n_high_orph += bands_o["high"]
            n_med_orph += bands_o["medium"]
            print(f"  {sid}: {n_match}/{len(gt)} matched, {n_orph} orphans (high={bands_o['high']}, medium={bands_o['medium']})")
            if n_match < len(gt):
                lost = sorted({str(g.shank) for g in gt} - {info[0] for info in matched.values()})
                per_subject_lost[tol][sid] = lost

        summary_rows.append({
            "tol": tol,
            "matched": n_match_all,
            "emit": n_emit_all,
            "orph": n_orph_all,
            "high_orph": n_high_orph,
            "med_orph": n_med_orph,
        })

    cpfit.PITCH_TOL_MM = orig_tol

    print(f"\n{'='*78}\nSUMMARY\n{'='*78}")
    print(f"{'tol':>4s} {'matched':>8s} {'emit':>5s} {'orph':>5s} {'high_o':>7s} {'med_o':>6s}")
    for s in summary_rows:
        print(f"{s['tol']:>4.1f} {s['matched']:>8d} {s['emit']:>5d} {s['orph']:>5d} {s['high_orph']:>7d} {s['med_orph']:>6d}")

    print("\n=== shanks lost vs production (tol=0.5) ===")
    for tol in TOLERANCES[1:]:
        lost = per_subject_lost.get(tol, {})
        if not lost:
            print(f"  tol={tol}: no shanks lost")
        else:
            total_lost = sum(len(v) for v in lost.values())
            print(f"  tol={tol}: {total_lost} shanks lost")
            for sid, shanks in sorted(lost.items()):
                print(f"    {sid}: {shanks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
