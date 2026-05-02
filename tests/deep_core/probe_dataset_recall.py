"""Quick dataset recall + orphan-band summary.

Runs the contact_pitch_v1 pipeline on every dataset subject (excluding
T17/T19/T21) and prints:
  - per-subject matched/total + orphan count
  - dataset totals (matched recall, orphan band breakdown)
  - bolt_source distribution among matched + orphan
  - per-must-preserve shank confirmation

Used to verify that pipeline refactors keep the regression promise:
295/295 matched, ≤4 orphans (no high-band orphans).

Usage:
    cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper && \\
        /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_dataset_recall.py
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

from rosa_detect.service import run_contact_pitch_v1
from eval_seeg_localization import (
    build_detection_context, iter_subject_rows,
    load_reference_ground_truth_shanks,
)

DATASET_ROOT = Path("/Users/ammar/Dropbox/thalamus_subjects/seeg_localization")
EXCLUDE = {"T17", "T19", "T21"}

MUST_PRESERVE = {
    ("T4", "RSFG"),
    ("T3", "LAI"),
    ("T4", "LCMN"),
    ("T4", "LSFG"),
}


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


def main():
    rows = iter_subject_rows(DATASET_ROOT, None)
    rows = [r for r in rows if str(r["subject_id"]) not in EXCLUDE]
    rows.sort(key=lambda r: int(str(r["subject_id"]).lstrip("T")))

    matched_by_band = Counter()
    orphan_by_band = Counter()
    matched_by_source = Counter()
    orphan_by_source = Counter()
    must_preserve_seen = []
    notable_orphans = []
    total_gt = 0
    total_emit = 0

    for row in rows:
        sid = str(row["subject_id"])
        gt, _ = load_reference_ground_truth_shanks(row)
        ctx, _ = build_detection_context(
            row["ct_path"], run_id=f"probe_recall_{sid}",
            config={}, extras={},
        )
        ctx["contact_pitch_v1_pitch_strategy"] = "auto"
        result = run_contact_pitch_v1(ctx)
        trajs = list(result.get("trajectories") or [])
        matched = _greedy_match(gt, trajs)
        total_gt += len(gt)
        total_emit += len(trajs)
        for ti, t in enumerate(trajs):
            band = str(t.get("confidence_label", "?"))
            src = str(t.get("bolt_source", "?"))
            name = matched.get(ti, "")
            if ti in matched:
                matched_by_band[band] += 1
                matched_by_source[src] += 1
                if (sid, name) in MUST_PRESERVE:
                    must_preserve_seen.append({
                        "subject": sid, "name": name,
                        "bolt_source": src, "band": band,
                        "n_inliers": int(t.get("n_inliers", 0)),
                        "conf": float(t.get("confidence", 0.0)),
                    })
            else:
                orphan_by_band[band] += 1
                orphan_by_source[src] += 1
                if band in ("high", "medium"):
                    s = list(t.get("start_ras") or [])
                    e = list(t.get("end_ras") or [])
                    notable_orphans.append({
                        "subject": sid, "ti": ti, "band": band,
                        "bolt_source": src,
                        "n": int(t.get("n_inliers", 0)),
                        "pitch": float(t.get("original_median_pitch_mm", 0.0) or 0.0),
                        "conf": float(t.get("confidence", 0.0)),
                        "frac_strong": float(t.get("frac_strong_metal", 0.0) or 0.0),
                        "dist_max": float(t.get("dist_max_mm", 0.0) or 0.0),
                        "length_mm": float(t.get("length_mm", 0.0) or 0.0),
                        "start": s, "end": e,
                    })
        n_orph = len(trajs) - len(matched)
        print(f"  {sid:>4s}: {len(matched)}/{len(gt)} matched, {n_orph} orphans")

    print(f"\n=== dataset totals ===")
    print(f"GT shanks:        {total_gt}")
    print(f"Matched:          {sum(matched_by_band.values())} "
          f"({100.0*sum(matched_by_band.values())/max(1,total_gt):.1f}%)")
    print(f"Orphans:          {sum(orphan_by_band.values())}")
    print(f"Total emissions:  {total_emit}")

    print(f"\n=== matched score-band breakdown ===")
    for band in ("high", "medium", "low"):
        print(f"  {band:>6s}: {matched_by_band[band]:>4d}")

    print(f"\n=== orphan score-band breakdown ===")
    for band in ("high", "medium", "low"):
        print(f"  {band:>6s}: {orphan_by_band[band]:>4d}")

    print(f"\n=== matched bolt_source breakdown ===")
    for src in sorted(matched_by_source.keys() | orphan_by_source.keys()):
        print(f"  {src:>13s}: matched={matched_by_source[src]:>4d}  "
              f"orphan={orphan_by_source[src]:>3d}")

    print(f"\n=== orphan detail (medium + high band) ===")
    if not notable_orphans:
        print("  none")
    else:
        notable_orphans.sort(key=lambda r: (r["band"] != "high", -r["conf"]))
        print(f"{'subj':>4s} {'band':>6s} {'bolt':>11s}  "
              f"{'n':>3s}  {'pitch':>5s}  {'frac_str':>8s}  "
              f"{'dmax':>5s}  {'len':>5s}  {'conf':>5s}")
        for r in notable_orphans:
            print(
                f"{r['subject']:>4s} {r['band']:>6s} {r['bolt_source']:>11s}  "
                f"{r['n']:>3d}  {r['pitch']:>5.2f}  {r['frac_strong']:>8.2f}  "
                f"{r['dist_max']:>5.1f}  {r['length_mm']:>5.1f}  {r['conf']:>5.2f}"
            )

    print(f"\n=== must-preserve shanks (4) ===")
    seen_keys = {(r["subject"], r["name"]) for r in must_preserve_seen}
    for key in sorted(MUST_PRESERVE):
        if key in seen_keys:
            r = next(r for r in must_preserve_seen if (r["subject"], r["name"]) == key)
            print(f"  {key[0]:>3s} {key[1]:>5s}: bolt={r['bolt_source']:>11s}  "
                  f"band={r['band']:>6s}  n={r['n_inliers']:>2d}  "
                  f"conf={r['conf']:.2f}")
        else:
            print(f"  {key[0]:>3s} {key[1]:>5s}: NOT FOUND IN MATCHED")

    return 0


if __name__ == "__main__":
    sys.exit(main())
