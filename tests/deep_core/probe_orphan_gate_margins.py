"""Orphan gate-margin probe.

For each emitted trajectory (matched + orphan), compute the "margin"
above each hard gate floor in the pipeline. Margins are normalized:
0 = at the gate, 1 = just above the gate by one unit. The goal is to
identify which gate(s) FP orphans barely pass, so we can decide
whether tightening that gate would lose any matches.

Hypothesis: many of the medium-band orphans are in the dataset because
gates calibrated for the older Ball r=2 kernel are too lax for the
newer Box r=1 (which finds more candidate blobs and thus more
candidate chains). The hard gates that were lowered during the
recall-first phase to recover specific subjects (T4 RPOG, T5 LIPR,
etc.) may now be over-permissive.

Output: per-orphan gate-margin breakdown plus aggregated p10/p50/p90
percentiles for matched vs orphan distributions of each margin. The
gate(s) where the matched p10 is well above the orphan p90 are
candidates for tightening.

Run:
    cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper && \\
        /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_orphan_gate_margins.py
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


def _gate_margins(rec):
    """Return a dict of margin-above-gate-floor for each gate. Negative
    or zero means the trajectory barely passes / is at the gate.
    Higher = comfortably above the gate.
    """
    n = int(rec.get("n_inliers", 0))
    span = float(rec.get("contact_span_mm", 0.0))
    length = float(rec.get("length_mm", 0.0))
    dist_max = float(rec.get("dist_max_mm", 0.0))
    dist_mean = float(rec.get("dist_mean_mm", 0.0))
    amp = float(rec.get("amp_sum", 0.0))
    frangi = float(rec.get("frangi_median_mm", 0.0))
    pitch = float(rec.get("original_median_pitch_mm", 0.0))

    # Closest library-pitch deviation.
    if pitch > 0:
        pitch_dev = min(abs(pitch - lp) for lp in cpfit.LIBRARY_PITCHES_MM)
    else:
        pitch_dev = float("nan")

    # Determine which deep_tip floor applies (long vs short).
    if pitch > 0 and pitch <= cpfit.DEEP_TIP_SHORT_MAX_AVG_PITCH_MM:
        deep_tip_floor = cpfit.DEEP_TIP_MIN_SHORT_MM
    else:
        deep_tip_floor = cpfit.DEEP_TIP_MIN_MM

    return {
        "n_inliers__floor5": n - cpfit.MIN_BLOBS_PER_LINE,
        "contact_span_mm": span - cpfit.MIN_LINE_SPAN_MM,
        "length_mm": length - cpfit.MIN_POST_ANCHOR_LEN_MM,
        "dist_max_mm__deep_tip": dist_max - deep_tip_floor,
        "frangi_median_mm__floor30": frangi - cpfit.FRANGI_LINE_MIN_MEDIAN,
        "pitch_dev_mm__neg_lower_better": -pitch_dev,  # smaller dev = better
        "amp_sum_per_inlier": amp / max(1, n),  # per-contact amp; not gated but informative
    }


def _summarize_distributions(matched_recs, orphan_recs):
    keys = list(_gate_margins(matched_recs[0]).keys()) if matched_recs else []
    print(f"\n{'gate margin':<40s}  {'matched p10/p50/p90':>30s}    {'orphan p10/p50/p90':>30s}")
    print("-" * 110)
    for k in keys:
        m = np.array([float(_gate_margins(r)[k]) for r in matched_recs], dtype=float)
        o = np.array([float(_gate_margins(r)[k]) for r in orphan_recs], dtype=float)
        m = m[~np.isnan(m)]
        o = o[~np.isnan(o)]
        if m.size == 0 or o.size == 0:
            continue
        mp = (np.percentile(m, 10), np.percentile(m, 50), np.percentile(m, 90))
        op = (np.percentile(o, 10), np.percentile(o, 50), np.percentile(o, 90))
        # Annotate where matched p10 is above orphan p90 (a clean separation).
        sep = "  *" if mp[0] > op[2] else ""
        print(f"{k:<40s}  {mp[0]:>9.2f} {mp[1]:>9.2f} {mp[2]:>9.2f}    {op[0]:>9.2f} {op[1]:>9.2f} {op[2]:>9.2f}{sep}")


def main():
    registry = PipelineRegistry()
    register_builtin_pipelines(registry)

    rows = iter_subject_rows(DATASET_ROOT, None)
    rows = [r for r in rows if str(r["subject_id"]) not in EXCLUDE]
    rows.sort(key=lambda r: str(r["subject_id"]))

    matched_all: list[dict] = []
    orphan_all: list[dict] = []
    print(f"running pipeline on {len(rows)} subjects (auto strategy)")
    for row in rows:
        subject_id = str(row["subject_id"])
        gt, _ = load_reference_ground_truth_shanks(row)
        ctx, _ = build_detection_context(
            row["ct_path"],
            run_id=f"probe_gate_{subject_id}",
            config={},
            extras={},
        )
        ctx["contact_pitch_v1_pitch_strategy"] = "auto"
        result = registry.run("contact_pitch_v1", ctx)
        if str(result.get("status", "ok")).lower() == "error":
            err = dict(result.get("error") or {})
            print(f"  {subject_id}: ERROR {err.get('message')}")
            continue
        trajs = list(result.get("trajectories") or [])
        matched = _greedy_match(gt, trajs)
        for ti, t in enumerate(trajs):
            t["__subject"] = subject_id
            if ti in matched:
                t["__shank"] = matched[ti][0]
                matched_all.append(t)
            else:
                orphan_all.append(t)
        n_o = len(trajs) - len(matched)
        print(f"  {subject_id}: {len(matched)}/{len(gt)} matched, {n_o} orphans")

    print(f"\ntotals: matched={len(matched_all)}, orphans={len(orphan_all)}")

    print("\n=== gate-margin distribution (* = matched p10 > orphan p90, clean separation) ===")
    _summarize_distributions(matched_all, orphan_all)

    # Filter to medium-band orphans specifically (the FP tail).
    medium_orphans = [r for r in orphan_all if str(r.get("confidence_label","?")) == "medium"]
    print(f"\n=== gate margins for matched ({len(matched_all)}) vs MEDIUM-band orphans ({len(medium_orphans)}) ===")
    _summarize_distributions(matched_all, medium_orphans)

    # Per-medium-orphan dump showing which gate(s) they barely pass.
    print("\n=== medium-band orphans, sorted by minimum-gate-margin ascending ===")
    print(f"  shows the 'tightest' gate per orphan — which floor it sits closest to")
    rows_dump = []
    for r in medium_orphans:
        m = _gate_margins(r)
        # Identify the tightest passing gate (smallest non-negative margin
        # among the genuinely gated ones).
        gates_only = {k: v for k, v in m.items() if k in {
            "n_inliers__floor5",
            "contact_span_mm",
            "length_mm",
            "dist_max_mm__deep_tip",
            "frangi_median_mm__floor30",
        }}
        tightest_k = min(gates_only, key=lambda k: gates_only[k])
        tightest_v = gates_only[tightest_k]
        rows_dump.append((tightest_v, tightest_k, r))
    rows_dump.sort(key=lambda x: x[0])
    for tv, tk, r in rows_dump:
        print(
            f"  {str(r['__subject']):>3s}  conf={float(r.get('confidence',0.0)):.3f}  "
            f"bolt={str(r.get('bolt_source','?')):>10s}  "
            f"tightest gate: {tk:<36s} margin={tv:>5.2f}  "
            f"n={int(r.get('n_inliers',0)):>2d} amp={float(r.get('amp_sum',0.0)):>6.0f} "
            f"span={float(r.get('contact_span_mm',0.0)):>5.1f} "
            f"frangi={float(r.get('frangi_median_mm',0.0)):>5.0f} "
            f"pitch={float(r.get('original_median_pitch_mm',0.0)):>4.2f}"
        )

    # Tight-gate frequency — which gate is the bottleneck for medium orphans?
    from collections import Counter
    counter = Counter(tk for _, tk, _ in rows_dump)
    print("\n=== medium-band orphan tightest-gate frequency ===")
    for k, n in counter.most_common():
        print(f"  {k:<40s}  {n:>3d} orphans")

    return 0


if __name__ == "__main__":
    sys.exit(main())
