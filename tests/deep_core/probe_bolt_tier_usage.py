"""Bolt-anchor tier usage probe.

Concept-swap #2 proposes collapsing the 3-tier bolt cascade
(LoG@800 → HU@2000 → axis-synth) into a single metal-mass-along-axis
pass. Before refactoring, count how many MATCHED (real) shanks each
tier carries on the current dataset. The refactor must preserve all
of these.

Per subject:
  - n trajectories per `bolt_source` ∈ {log, hu_rescue, axis_synth, none}
  - matched vs orphan within each tier
  - confidence band within each tier

Aggregate: which tiers are load-bearing for real shanks?

Run:
    cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper && \\
        /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_bolt_tier_usage.py
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper")
sys.path.insert(0, str(ROOT / "CommonLib"))
sys.path.insert(0, str(ROOT / "PostopCTLocalization"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np

from shank_engine import PipelineRegistry, register_builtin_pipelines
from eval_seeg_localization import (
    build_detection_context, iter_subject_rows,
    load_reference_ground_truth_shanks,
)


DATASET_ROOT = Path("/Users/ammar/Dropbox/thalamus_subjects/seeg_localization")
EXCLUDE = {"T17", "T19", "T21"}
TIERS = ("log", "hu_rescue", "axis_synth", "none")


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def _greedy_match(gt_shanks, trajs):
    pairs = []
    for gi, g in enumerate(gt_shanks):
        gs, ge = np.asarray(g.start_ras, dtype=float), np.asarray(g.end_ras, dtype=float)
        ga = _unit(ge - gs); gm = 0.5 * (gs + ge)
        for ti, t in enumerate(trajs):
            ts, te = np.asarray(t["start_ras"], dtype=float), np.asarray(t["end_ras"], dtype=float)
            ta = _unit(te - ts); tm = 0.5 * (ts + te)
            ang = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(ga, ta)))))))
            d = gm - tm; mid = float(np.linalg.norm(d - (d @ ta) * ta))
            if ang <= 10 and mid <= 8:
                pairs.append((ang + mid, gi, ti))
    pairs.sort(); used_g, used_t = set(), set(); m = {}
    for _s, gi, ti in pairs:
        if gi in used_g or ti in used_t: continue
        used_g.add(gi); used_t.add(ti); m[ti] = str(gt_shanks[gi].shank)
    return m


def main():
    registry = PipelineRegistry()
    register_builtin_pipelines(registry)
    rows = iter_subject_rows(DATASET_ROOT, None)
    rows = [r for r in rows if str(r["subject_id"]) not in EXCLUDE]
    rows.sort(key=lambda r: int(str(r["subject_id"]).lstrip("T")))

    matched_by_tier = Counter()
    orphan_by_tier = Counter()
    matched_band_by_tier = defaultdict(Counter)
    orphan_band_by_tier = defaultdict(Counter)
    matched_subjects_by_tier = defaultdict(list)  # for hu_rescue + axis_synth: which shanks

    for row in rows:
        sid = str(row["subject_id"])
        gt, _ = load_reference_ground_truth_shanks(row)
        ctx, _img = build_detection_context(
            row["ct_path"], run_id=f"probe_tier_{sid}", config={}, extras={},
        )
        ctx["contact_pitch_v1_pitch_strategy"] = "auto"
        result = registry.run("contact_pitch_v1", ctx)
        trajs = list(result.get("trajectories") or [])
        matched = _greedy_match(gt, trajs)
        for ti, t in enumerate(trajs):
            tier = str(t.get("bolt_source", "?"))
            band = str(t.get("confidence_label", "?"))
            if ti in matched:
                matched_by_tier[tier] += 1
                matched_band_by_tier[tier][band] += 1
                if tier in ("hu_rescue", "axis_synth"):
                    matched_subjects_by_tier[tier].append(
                        (sid, matched[ti], int(t.get("n_inliers", 0)),
                         float(t.get("dist_max_mm", 0.0)),
                         float(t.get("original_median_pitch_mm", 0.0)),
                         float(t.get("confidence", 0.0)))
                    )
            else:
                orphan_by_tier[tier] += 1
                orphan_band_by_tier[tier][band] += 1

    print("\n=== matched trajectory bolt-source tier usage ===")
    print(f"{'tier':>11s}  {'count':>5s}   {'high':>5s} {'medium':>6s} {'low':>4s}")
    tot_matched = 0
    for tier in TIERS:
        n = matched_by_tier[tier]
        h = matched_band_by_tier[tier]["high"]
        m = matched_band_by_tier[tier]["medium"]
        l = matched_band_by_tier[tier]["low"]
        tot_matched += n
        print(f"{tier:>11s}  {n:>5d}   {h:>5d} {m:>6d} {l:>4d}")
    print(f"{'TOTAL':>11s}  {tot_matched:>5d}")

    print("\n=== orphan trajectory bolt-source tier usage ===")
    print(f"{'tier':>11s}  {'count':>5s}   {'high':>5s} {'medium':>6s} {'low':>4s}")
    tot_orphan = 0
    for tier in TIERS:
        n = orphan_by_tier[tier]
        h = orphan_band_by_tier[tier]["high"]
        m = orphan_band_by_tier[tier]["medium"]
        l = orphan_band_by_tier[tier]["low"]
        tot_orphan += n
        print(f"{tier:>11s}  {n:>5d}   {h:>5d} {m:>6d} {l:>4d}")
    print(f"{'TOTAL':>11s}  {tot_orphan:>5d}")

    # Detail: which real shanks NEED tier 2 / tier 3?
    for tier in ("hu_rescue", "axis_synth"):
        recs = matched_subjects_by_tier[tier]
        if not recs: continue
        print(f"\n=== matched shanks rescued by tier '{tier}' ({len(recs)}) ===")
        print(f"{'subj':>4s}  {'shank':>10s}  {'n':>3s}  {'dmax':>5s}  {'pitch':>5s}  {'conf':>5s}")
        for sid, shank, n, dmax, pitch, conf in recs:
            print(f"{sid:>4s}  {shank:>10s}  {n:>3d}  {dmax:>5.1f}  {pitch:>5.2f}  {conf:>5.2f}")


if __name__ == "__main__":
    main()
