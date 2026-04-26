"""Effect-of-gates probe — does DEEP_TIP_* still drop anything today?

Concept-swap #5 retires DEEP_TIP_MIN_MM, DEEP_TIP_MIN_SHORT_MM,
DEEP_TIP_SHORT_MAX_AVG_PITCH_MM, MIN_INLIER_DIST_MEAN_MM. Before
designing a replacement, find out if those gates are doing any
work in the current dataset:

  1. Run the pipeline normally (baseline).
  2. Run with the gates effectively disabled (set thresholds to 0).
  3. Compare emitted trajectory counts and matched/orphan splits.

If the disabled run produces the same recall and a similar orphan
count, the gates are vestigial — retire them as dead code, score
framework backstops them. If the disabled run blows up orphan
count, we need a real replacement.

Run
---
    cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper && \\
        /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_deep_tip_gate_effect.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path("/Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper")
sys.path.insert(0, str(ROOT / "CommonLib"))
sys.path.insert(0, str(ROOT / "PostopCTLocalization"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np

from postop_ct_localization import contact_pitch_v1_fit as cpfit
from eval_seeg_localization import (
    iter_subject_rows,
    load_reference_ground_truth_shanks,
    build_detection_context,
)
from shank_engine import PipelineRegistry, register_builtin_pipelines

DATASET_ROOT = Path(
    os.environ.get(
        "ROSA_SEEG_DATASET",
        "/Users/ammar/Dropbox/thalamus_subjects/seeg_localization",
    )
)
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
            mid_d = float(np.linalg.norm(d - (d @ t_axis) * t_axis))
            if ang <= MATCH_ANGLE_DEG and mid_d <= MATCH_MID_MM:
                pairs.append((ang + mid_d, gi, ti))
    pairs.sort(key=lambda p: p[0])
    used_g, used_t = set(), set()
    matched = set()
    for _s, gi, ti in pairs:
        if gi in used_g or ti in used_t:
            continue
        used_g.add(gi)
        used_t.add(ti)
        matched.add(ti)
    return matched


def _run_dataset(label, *, return_per_traj=False):
    registry = PipelineRegistry()
    register_builtin_pipelines(registry)
    rows = iter_subject_rows(DATASET_ROOT, None)
    rows = [r for r in rows if str(r["subject_id"]) not in EXCLUDE]
    rows.sort(key=lambda r: str(r["subject_id"]))

    print(f"\n=== {label} ===")
    totals = {"matched": 0, "orphans": 0, "emitted": 0, "gt": 0}
    rows_per_subject = []
    per_traj = []  # (subject_id, label_str, confidence, confidence_label)
    for row in rows:
        sid = str(row["subject_id"])
        gt, _ = load_reference_ground_truth_shanks(row)
        ctx, _ = build_detection_context(
            row["ct_path"], run_id=f"probe_{sid}_{label}",
            config={}, extras={},
        )
        ctx["contact_pitch_v1_pitch_strategy"] = "auto"
        result = registry.run("contact_pitch_v1", ctx)
        trajs = list(result.get("trajectories") or [])
        matched = _greedy_match(gt, trajs)
        n_match = len(matched)
        n_orph = len(trajs) - n_match
        rows_per_subject.append((sid, len(gt), n_match, n_orph, len(trajs)))
        totals["gt"] += len(gt)
        totals["matched"] += n_match
        totals["orphans"] += n_orph
        totals["emitted"] += len(trajs)
        if return_per_traj:
            for ti, t in enumerate(trajs):
                per_traj.append((
                    sid,
                    "matched" if ti in matched else "orphan",
                    float(t.get("confidence") or float("nan")),
                    str(t.get("confidence_label") or ""),
                    float(t.get("dist_max_mm") or 0.0),
                    int(t.get("n_inliers") or 0),
                ))

    print(f"{'subject':>8s} {'gt':>4s} {'matched':>8s} {'orph':>5s} {'emit':>5s}")
    for sid, n_gt, n_m, n_o, n_e in rows_per_subject:
        print(f"{sid:>8s} {n_gt:>4d} {n_m:>8d} {n_o:>5d} {n_e:>5d}")
    print(
        f"\nTotals: matched={totals['matched']}/{totals['gt']} "
        f"({totals['matched']/max(1,totals['gt'])*100:.1f}% recall) "
        f"orphans={totals['orphans']} emitted={totals['emitted']}"
    )
    if return_per_traj:
        return rows_per_subject, totals, per_traj
    return rows_per_subject, totals


def main():
    print("Baseline (gates active)")
    print("DEEP_TIP_MIN_MM        =", cpfit.DEEP_TIP_MIN_MM)
    print("DEEP_TIP_MIN_SHORT_MM  =", cpfit.DEEP_TIP_MIN_SHORT_MM)
    print("DEEP_TIP_SHORT_MAX_AVG_PITCH_MM =", cpfit.DEEP_TIP_SHORT_MAX_AVG_PITCH_MM)
    print("MIN_INLIER_DIST_MEAN_MM =", cpfit.MIN_INLIER_DIST_MEAN_MM)
    base_rows, base_totals, base_per_traj = _run_dataset("baseline", return_per_traj=True)

    # Disable the deep-tip gates by setting their thresholds to 0.
    saved = {
        "DEEP_TIP_MIN_MM": cpfit.DEEP_TIP_MIN_MM,
        "DEEP_TIP_MIN_SHORT_MM": cpfit.DEEP_TIP_MIN_SHORT_MM,
        "MIN_INLIER_DIST_MEAN_MM": cpfit.MIN_INLIER_DIST_MEAN_MM,
    }
    try:
        # 1 mm is effectively "no gate" — no real shank has dist_max < 1 mm.
        # Using 1 instead of 0 because the score framework has a
        # ``dist_max / DEEP_TIP_MIN_MM`` term that would crash on zero.
        # Score saturation point is restored to 30 inside the call so the
        # depth_score behaves identically to baseline; only the gate is off.
        cpfit.DEEP_TIP_MIN_MM = 1.0
        cpfit.DEEP_TIP_MIN_SHORT_MM = 1.0
        cpfit.MIN_INLIER_DIST_MEAN_MM = 0.0
        no_rows, no_totals, no_per_traj = _run_dataset(
            "deep-tip + intracranial-mean gates DISABLED",
            return_per_traj=True,
        )
    finally:
        for k, v in saved.items():
            setattr(cpfit, k, v)

    # Per-subject delta.
    print("\n=== per-subject delta (baseline -> disabled) ===")
    print(f"{'subject':>8s} {'gt':>4s}  {'M base':>7s} {'M no':>5s}  {'O base':>7s} {'O no':>5s}  {'E base':>7s} {'E no':>5s}")
    base_map = {r[0]: r for r in base_rows}
    for sid, n_gt, n_m, n_o, n_e in no_rows:
        b = base_map[sid]
        print(
            f"{sid:>8s} {n_gt:>4d}  {b[2]:>7d} {n_m:>5d}  "
            f"{b[3]:>7d} {n_o:>5d}  {b[4]:>7d} {n_e:>5d}"
        )

    print("\n=== summary ===")
    print(f"baseline:  matched={base_totals['matched']}/{base_totals['gt']} "
          f"orphans={base_totals['orphans']} emitted={base_totals['emitted']}")
    print(f"disabled:  matched={no_totals['matched']}/{no_totals['gt']} "
          f"orphans={no_totals['orphans']} emitted={no_totals['emitted']}")
    print(f"delta: matched={no_totals['matched']-base_totals['matched']:+d} "
          f"orphans={no_totals['orphans']-base_totals['orphans']:+d} "
          f"emitted={no_totals['emitted']-base_totals['emitted']:+d}")

    # Score-band breakdown of all emissions in each run.
    def _band_counts(per_traj):
        out = {"matched": {"high": 0, "medium": 0, "low": 0},
                "orphan":  {"high": 0, "medium": 0, "low": 0}}
        for sid, lab, conf, conf_lab, dmax, n in per_traj:
            band = conf_lab if conf_lab in ("high", "medium", "low") else "low"
            out[lab][band] += 1
        return out
    print("\n=== confidence band by label (baseline) ===")
    bb = _band_counts(base_per_traj)
    print(f"  matched: high={bb['matched']['high']} medium={bb['matched']['medium']} low={bb['matched']['low']}")
    print(f"  orphan:  high={bb['orphan']['high']} medium={bb['orphan']['medium']} low={bb['orphan']['low']}")
    print("=== confidence band by label (gates DISABLED) ===")
    nb = _band_counts(no_per_traj)
    print(f"  matched: high={nb['matched']['high']} medium={nb['matched']['medium']} low={nb['matched']['low']}")
    print(f"  orphan:  high={nb['orphan']['high']} medium={nb['orphan']['medium']} low={nb['orphan']['low']}")
    print(f"=== delta orphan band: high={nb['orphan']['high']-bb['orphan']['high']:+d} "
          f"medium={nb['orphan']['medium']-bb['orphan']['medium']:+d} "
          f"low={nb['orphan']['low']-bb['orphan']['low']:+d}")

    # The new orphans — lines the gate would have rejected in baseline.
    # Need to identify them: match by (subject, ti) is ambiguous because
    # ti changes when more lines are emitted. Use confidence + dist_max
    # as a coarse signature: anything in the "disabled" run with
    # dist_max < 30 (which the baseline gate would have killed) is a
    # newly-emitted line.
    newly_emitted_lo_dist = [
        t for t in no_per_traj if t[1] == "orphan" and t[4] < 30.0
    ]
    print(f"\n=== newly-emitted orphans with dist_max < 30 mm "
          f"(would have been gated): {len(newly_emitted_lo_dist)} ===")
    band_counts = {"high": 0, "medium": 0, "low": 0}
    for sid, lab, conf, conf_lab, dmax, n in newly_emitted_lo_dist:
        band_counts[conf_lab if conf_lab in band_counts else "low"] += 1
    print(f"  band: high={band_counts['high']} medium={band_counts['medium']} low={band_counts['low']}")
    print("\nworst 20 (lowest confidence among newly-emitted shallow orphans):")
    newly_emitted_lo_dist.sort(key=lambda t: t[2])
    for sid, lab, conf, conf_lab, dmax, n in newly_emitted_lo_dist[:20]:
        print(
            f"  {sid:>4s}  conf={conf:.3f}({conf_lab:>6s})  "
            f"dist_max={dmax:>5.1f}  n_inliers={n:>2d}"
        )
    print("\ntop 20 (highest confidence among newly-emitted shallow orphans):")
    newly_emitted_lo_dist.sort(key=lambda t: -t[2])
    for sid, lab, conf, conf_lab, dmax, n in newly_emitted_lo_dist[:20]:
        print(
            f"  {sid:>4s}  conf={conf:.3f}({conf_lab:>6s})  "
            f"dist_max={dmax:>5.1f}  n_inliers={n:>2d}"
        )


if __name__ == "__main__":
    main()
