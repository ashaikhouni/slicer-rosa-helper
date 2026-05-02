"""Inspect the post-Swap-#2 T3 high-band orphan.

Pipeline 1.0.19 emits a new high-band orphan on T3 (n=12 inliers,
pitch=3.61 mm, frac_strong=0.85, conf=0.93) that the pre-Swap-#2
cascade rejected. The metrics look SEEG-class — needs to be classified
as either:
  (a) a near-duplicate of an existing matched T3 emission that survived
      ``_dedup_trajectories`` (algorithm bug; fix dedup),
  (b) a real shank missing from the GT (annotation gap; flag for redo),
  (c) a genuine FP.

For each emitted T3 trajectory and each GT shank, this probe dumps:
  - geometry (start/end RAS, axis, midpoint)
  - bolt_id (which CC anchored)
  - score_components
  - per-other-emission angular + lateral closest-approach distances
  - per-GT angular + midpoint-perpendicular distances (the same metrics
    used by the greedy match window)

A high-band emission whose angular + perp distance to the nearest
GT shank exceeds the match window (10°/8 mm) is, by definition, NOT
in GT. If it ALSO sits parallel/close to a matched emission, that's
case (a). If it's at a unique location, that's case (b) or (c).

Usage:
    cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper && \\
        /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_t3_high_orphan.py
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
    build_detection_context, iter_subject_rows,
    load_reference_ground_truth_shanks,
)

DATASET_ROOT = Path("/Users/ammar/Dropbox/thalamus_subjects/seeg_localization")
SUBJECT = sys.argv[1] if len(sys.argv) > 1 else "T3"


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def _greedy_match(gt_shanks, trajs):
    pairs = []
    for gi, g in enumerate(gt_shanks):
        gs = np.asarray(g.start_ras, dtype=float)
        ge = np.asarray(g.end_ras, dtype=float)
        ga = _unit(ge - gs)
        gm = 0.5 * (gs + ge)
        for ti, t in enumerate(trajs):
            ts = np.asarray(t["start_ras"], dtype=float)
            te = np.asarray(t["end_ras"], dtype=float)
            ta = _unit(te - ts)
            tm = 0.5 * (ts + te)
            ang = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(ga, ta)))))))
            d = gm - tm
            mid = float(np.linalg.norm(d - (d @ ta) * ta))
            if ang <= 10 and mid <= 8:
                pairs.append((ang + mid, gi, ti))
    pairs.sort()
    used_g, used_t = set(), set()
    m = {}
    for _s, gi, ti in pairs:
        if gi in used_g or ti in used_t:
            continue
        used_g.add(gi)
        used_t.add(ti)
        m[ti] = str(gt_shanks[gi].shank)
    return m


def _ang_mid_to(t1, t2):
    s1 = np.asarray(t1["start_ras"], dtype=float)
    e1 = np.asarray(t1["end_ras"], dtype=float)
    s2 = np.asarray(t2["start_ras"], dtype=float)
    e2 = np.asarray(t2["end_ras"], dtype=float)
    a1 = _unit(e1 - s1)
    a2 = _unit(e2 - s2)
    m1 = 0.5 * (s1 + e1)
    m2 = 0.5 * (s2 + e2)
    cos_ang = max(-1.0, min(1.0, abs(float(np.dot(a1, a2)))))
    ang = float(np.degrees(np.arccos(cos_ang)))
    d = m1 - m2
    perp = float(np.linalg.norm(d - (d @ a2) * a2))
    return ang, perp


def _ang_perp_to_gt(t, g):
    ts = np.asarray(t["start_ras"], dtype=float)
    te = np.asarray(t["end_ras"], dtype=float)
    ta = _unit(te - ts)
    tm = 0.5 * (ts + te)
    gs = np.asarray(g.start_ras, dtype=float)
    ge = np.asarray(g.end_ras, dtype=float)
    ga = _unit(ge - gs)
    gm = 0.5 * (gs + ge)
    cos_ang = max(-1.0, min(1.0, abs(float(np.dot(ta, ga)))))
    ang = float(np.degrees(np.arccos(cos_ang)))
    d = gm - tm
    perp = float(np.linalg.norm(d - (d @ ta) * ta))
    return ang, perp


def main():
    rows = [
        r for r in iter_subject_rows(DATASET_ROOT, {SUBJECT})
        if str(r["subject_id"]) == SUBJECT
    ]
    if not rows:
        print(f"ERROR: subject {SUBJECT} not found")
        return 1
    row = rows[0]
    gt, _ = load_reference_ground_truth_shanks(row)

    ctx, _ = build_detection_context(
        row["ct_path"], run_id=f"probe_t3_high_orphan",
        config={}, extras={},
    )
    ctx["contact_pitch_v1_pitch_strategy"] = "auto"
    result = run_contact_pitch_v1(ctx)
    trajs = list(result.get("trajectories") or [])
    matched = _greedy_match(gt, trajs)

    # Find the high-band orphan(s).
    high_orphans = [
        ti for ti, t in enumerate(trajs)
        if ti not in matched and str(t.get("confidence_label")) == "high"
    ]
    print(f"\n{SUBJECT}: {len(matched)}/{len(gt)} matched, "
          f"{len(trajs) - len(matched)} orphans ({len(high_orphans)} high-band)")
    if not high_orphans:
        print("No high-band orphan to inspect.")
        return 0

    for hi in high_orphans:
        target = trajs[hi]
        s = np.asarray(target["start_ras"], dtype=float)
        e = np.asarray(target["end_ras"], dtype=float)
        a = _unit(e - s)
        m = 0.5 * (s + e)

        print(f"\n=== HIGH-BAND ORPHAN  ti={hi} ===")
        print(f"  start_ras = ({s[0]:+7.2f}, {s[1]:+7.2f}, {s[2]:+7.2f})")
        print(f"  end_ras   = ({e[0]:+7.2f}, {e[1]:+7.2f}, {e[2]:+7.2f})")
        print(f"  midpoint  = ({m[0]:+7.2f}, {m[1]:+7.2f}, {m[2]:+7.2f})")
        print(f"  axis      = ({a[0]:+.3f}, {a[1]:+.3f}, {a[2]:+.3f})")
        print(f"  length    = {float(target.get('length_mm', 0.0)):.1f} mm")
        print(f"  bolt_src  = {target.get('bolt_source')}")
        print(f"  bolt_id   = {target.get('bolt_id')}")
        print(f"  bolt_n_vox= {target.get('bolt_n_vox')}")
        print(f"  n_inliers = {target.get('n_inliers')}")
        print(f"  pitch     = {target.get('original_median_pitch_mm', 0.0):.2f} mm")
        print(f"  frac_strg = {target.get('frac_strong_metal', 0.0):.2f}")
        print(f"  conf      = {target.get('confidence', 0.0):.2f} "
              f"({target.get('confidence_label')})")
        sc = target.get("score_components") or {}
        for k in sorted(sc):
            print(f"    score.{k:>17s} = {sc[k]:.2f}")

        # Closest other emission (any).
        print(f"\n  --- closest other emissions (any kind) ---")
        peers = []
        for ti, t in enumerate(trajs):
            if ti == hi:
                continue
            ang, perp = _ang_mid_to(target, t)
            kind = "matched" if ti in matched else "orphan"
            peers.append((ang + perp, ang, perp, ti, kind, t))
        peers.sort()
        for total, ang, perp, ti, kind, t in peers[:5]:
            label = matched.get(ti, "")
            bs = t.get("bolt_source", "?")
            bid = t.get("bolt_id", -1)
            print(
                f"    ti={ti:>2d} {kind:>7s} {label:>5s} "
                f"bolt={bs:>11s} bolt_id={bid:>3d}  "
                f"ang={ang:>5.2f}°  perp={perp:>5.2f} mm"
            )
            if ang < 5.0 and perp < 8.0:
                print(f"      ^^ NEAR DUPLICATE — possible dedup leak")

        # Closest GT shank.
        print(f"\n  --- closest GT shanks ---")
        gt_dists = []
        for gi, g in enumerate(gt):
            ang, perp = _ang_perp_to_gt(target, g)
            gt_dists.append((ang + perp, ang, perp, gi, g))
        gt_dists.sort()
        for total, ang, perp, gi, g in gt_dists[:5]:
            within_window = ang <= 10.0 and perp <= 8.0
            tag = " (WITHIN MATCH WINDOW)" if within_window else ""
            print(
                f"    gt[{gi:>2d}] {str(g.shank):>6s}  "
                f"ang={ang:>5.2f}°  perp={perp:>5.2f} mm{tag}"
            )

        # Whether the matched trajectory at this gt was bound to a different
        # emission (so the orphan loses out in greedy match by tie-break).
        for total, ang, perp, gi, g in gt_dists[:3]:
            within_window = ang <= 10.0 and perp <= 8.0
            if not within_window:
                continue
            # Was this gi already matched, and to whom?
            for ti, name in matched.items():
                if name == str(g.shank):
                    print(f"\n    GT[{gi}] {g.shank} is already matched "
                          f"to ti={ti}; the orphan is the loser of greedy "
                          f"match tie-break.")
                    other_ang, other_perp = _ang_perp_to_gt(trajs[ti], g)
                    print(f"      winner ti={ti}: ang={other_ang:.2f}° perp={other_perp:.2f} mm")
                    print(f"      orphan ti={hi}: ang={ang:.2f}° perp={perp:.2f} mm")
                    break

    # Show ALL emissions with bolt_id for cross-comparison.
    print(f"\n=== ALL emissions on {SUBJECT} (sorted by bolt_id) ===")
    rows_all = []
    for ti, t in enumerate(trajs):
        rows_all.append({
            "ti": ti,
            "kind": "matched" if ti in matched else "orphan",
            "label": matched.get(ti, ""),
            "bolt_src": str(t.get("bolt_source", "?")),
            "bolt_id": int(t.get("bolt_id", -1)),
            "n": int(t.get("n_inliers", 0)),
            "conf": float(t.get("confidence", 0.0)),
            "band": str(t.get("confidence_label", "?")),
        })
    rows_all.sort(key=lambda r: (r["bolt_id"], r["ti"]))
    print(f"{'ti':>3s} {'kind':>7s} {'label':>5s}  "
          f"{'bolt':>11s} {'b.id':>4s}  {'n':>3s} {'band':>6s} {'conf':>5s}")
    for r in rows_all:
        print(
            f"{r['ti']:>3d} {r['kind']:>7s} {r['label']:>5s}  "
            f"{r['bolt_src']:>11s} {r['bolt_id']:>4d}  "
            f"{r['n']:>3d} {r['band']:>6s} {r['conf']:>5.2f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
