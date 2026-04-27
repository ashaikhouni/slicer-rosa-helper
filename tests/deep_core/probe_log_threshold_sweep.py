"""LOG_BLOB_THRESHOLD sweep across the full dataset.

Hypothesis from the orphan gate-margin probe: ``LOG_BLOB_THRESHOLD =
300`` was calibrated when the LoG-blob extractor used SITK Ball r=2
(81-voxel kernel). Box r=1 (27 voxels, 26-connectivity) keeps more
local minima alive, including weak peaks just above the 300 floor.
Those weak peaks form spurious candidate chains and surface as
medium-band orphans (62 of 86 hit the n_inliers floor of 5).

Sweep:
  - 300 (current production)
  - 400
  - 500
  - 600

For each threshold, run the full pipeline on every dataset subject and
report:
  - matched count (recall)
  - total emissions
  - orphan total / medium / high
  - bolt-source breakdown of orphans

Goal: find the highest threshold that preserves 295/295 recall while
collapsing the FP tail.

Run:
    cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper && \\
        /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_log_threshold_sweep.py
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

THRESHOLDS = [300.0, 400.0, 500.0, 600.0]
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
    bands_m = {"high":0,"medium":0,"low":0}
    bands_o = {"high":0,"medium":0,"low":0}
    bolts_o: dict[str, int] = {}
    for ti, t in enumerate(trajs):
        cl = str(t.get("confidence_label","?"))
        if ti in matched:
            bands_m[cl] = bands_m.get(cl,0)+1
        else:
            bands_o[cl] = bands_o.get(cl,0)+1
            bs = str(t.get("bolt_source","?"))
            bolts_o[bs] = bolts_o.get(bs,0)+1
    return matched, bands_m, bands_o, bolts_o


def main():
    registry = PipelineRegistry()
    register_builtin_pipelines(registry)

    rows = iter_subject_rows(DATASET_ROOT, None)
    rows = [r for r in rows if str(r["subject_id"]) not in EXCLUDE]
    rows.sort(key=lambda r: str(r["subject_id"]))

    orig_thr = cpfit.LOG_BLOB_THRESHOLD
    orig_axis_thr = cpfit.AXIS_REFINE_MIN_ABS

    summary_rows = []
    per_subject_lost = {}  # threshold -> {subject: lost_shanks}

    for thr in THRESHOLDS:
        print(f"\n{'='*78}\nthreshold = {thr}\n{'='*78}")
        cpfit.LOG_BLOB_THRESHOLD = thr
        # Keep deep-end refinement consistent with detection threshold —
        # otherwise refinement would chase peaks the detector now ignores.
        cpfit.AXIS_REFINE_MIN_ABS = thr

        n_match_all = 0
        n_orph_all = 0
        n_emit_all = 0
        n_high_orph = 0
        n_med_orph = 0
        bolts_total: dict[str, int] = {}
        per_subject_lost[thr] = {}

        for row in rows:
            sid = str(row["subject_id"])
            gt, _ = load_reference_ground_truth_shanks(row)
            ctx, _ = build_detection_context(
                row["ct_path"],
                run_id=f"probe_thr{int(thr)}_{sid}",
                config={},
                extras={},
            )
            ctx["contact_pitch_v1_pitch_strategy"] = "auto"
            result = registry.run("contact_pitch_v1", ctx)
            if str(result.get("status", "ok")).lower() == "error":
                err = dict(result.get("error") or {})
                print(f"  {sid}: ERROR {err.get('message')}")
                continue
            trajs = list(result.get("trajectories") or [])
            matched, bands_m, bands_o, bolts_o = _summarize(trajs, gt)
            n_emit = len(trajs)
            n_match = len(matched)
            n_orph = n_emit - n_match
            n_match_all += n_match
            n_orph_all += n_orph
            n_emit_all += n_emit
            n_high_orph += bands_o["high"]
            n_med_orph += bands_o["medium"]
            for k, v in bolts_o.items():
                bolts_total[k] = bolts_total.get(k, 0) + v
            print(f"  {sid}: {n_match}/{len(gt)} matched, {n_orph} orphans (high={bands_o['high']}, medium={bands_o['medium']})")
            if n_match < len(gt):
                lost = sorted({str(g.shank) for g in gt} - {info[0] for info in matched.values()})
                per_subject_lost[thr][sid] = lost

        summary_rows.append({
            "thr": thr,
            "matched": n_match_all,
            "emit": n_emit_all,
            "orph": n_orph_all,
            "high_orph": n_high_orph,
            "med_orph": n_med_orph,
            "bolts": dict(bolts_total),
        })

    cpfit.LOG_BLOB_THRESHOLD = orig_thr
    cpfit.AXIS_REFINE_MIN_ABS = orig_axis_thr

    print(f"\n{'='*78}\nSUMMARY\n{'='*78}")
    print(f"{'thr':>4s} {'matched':>8s} {'emit':>5s} {'orph':>5s} {'high_o':>7s} {'med_o':>6s}  bolt orphan mix")
    for s in summary_rows:
        print(f"{int(s['thr']):>4d} {s['matched']:>8d} {s['emit']:>5d} {s['orph']:>5d} {s['high_orph']:>7d} {s['med_orph']:>6d}  {dict(sorted(s['bolts'].items()))}")

    print("\n=== shanks lost vs production (thr=300) ===")
    for thr in THRESHOLDS[1:]:
        lost = per_subject_lost.get(thr, {})
        if not lost:
            print(f"  thr={int(thr)}: no shanks lost")
        else:
            total_lost = sum(len(v) for v in lost.values())
            print(f"  thr={int(thr)}: {total_lost} shanks lost")
            for sid, shanks in sorted(lost.items()):
                print(f"    {sid}: {shanks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
