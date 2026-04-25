"""Score-framework feature probe.

Run the contact_pitch_v1 pipeline across the full SEEG dataset (T17/T19/T21
excluded as unreliable GT). Match each emitted trajectory to GT with the
standard 10 deg / 8 mm window via greedy 1:1, then dump per-trajectory
features to a TSV. The score framework reads this TSV to set initial
weights and a single EMIT_THRESHOLD that separates real matches from
auto orphans with margin.

Run
---
    cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper && \\
        /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_score_features.py

Output
------
    /tmp/probe_score_features.tsv  (one row per emitted trajectory)
    plus a stdout summary of joint distributions and naive cutoff sweeps.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path("/Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper")
sys.path.insert(0, str(ROOT / "CommonLib"))
sys.path.insert(0, str(ROOT / "PostopCTLocalization"))
sys.path.insert(0, str(ROOT / "tools"))

import os
import numpy as np

from shank_engine import PipelineRegistry, register_builtin_pipelines
from postop_ct_localization import contact_pitch_v1_fit as cpfit
from eval_seeg_localization import (
    build_detection_context,
    iter_subject_rows,
    load_reference_ground_truth_shanks,
)

DATASET_ROOT = Path(
    os.environ.get(
        "ROSA_SEEG_DATASET",
        "/Users/ammar/Dropbox/thalamus_subjects/seeg_localization",
    )
)
EXCLUDE = {"T17", "T19", "T21"}

MATCH_ANGLE_DEG = 10.0
MATCH_MID_MM = 8.0

OUT_TSV = Path("/tmp/probe_score_features.tsv")

# Mirror the detector's library so the probe's pitch_dev_mm matches what
# the in-detector score sees.
LIBRARY_PITCHES_MM = cpfit.LIBRARY_PITCHES_MM


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def _angle_deg(u, v):
    return float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(u, v)))))))


def _perp_mid(g_mid, t_mid, t_axis):
    d = g_mid - t_mid
    p = d - (d @ t_axis) * t_axis
    return float(np.linalg.norm(p))


def _greedy_match(gt_shanks, trajs):
    """Return (matched_traj_idx_set, traj_idx -> shank_label)."""
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
            ang = _angle_deg(g_axis, t_axis)
            t_mid = 0.5 * (t_s + t_e)
            mid_d = _perp_mid(g_mid, t_mid, t_axis)
            if ang <= MATCH_ANGLE_DEG and mid_d <= MATCH_MID_MM:
                pairs.append((ang + mid_d, gi, ti, ang, mid_d))
    pairs.sort(key=lambda p: p[0])
    used_g, used_t = set(), set()
    matched_ti = {}
    for _s, gi, ti, ang, mid_d in pairs:
        if gi in used_g or ti in used_t:
            continue
        used_g.add(gi)
        used_t.add(ti)
        matched_ti[ti] = (str(gt_shanks[gi].shank), ang, mid_d)
    return matched_ti


def _nearest_library_pitch(p):
    if not p or p != p:  # nan / None / 0
        return float("nan")
    return float(min(abs(p - x) for x in LIBRARY_PITCHES_MM))


def _features_for(rec):
    amp_sum = float(rec.get("amp_sum") or 0.0)
    n_inliers = int(rec.get("n_inliers") or 0)
    contact_span_mm = float(rec.get("contact_span_mm") or rec.get("original_span_mm") or 0.0)
    length_mm = float(rec.get("length_mm") or 0.0)
    frangi_median_mm = float(rec.get("frangi_median_mm") or 0.0)
    pitch = float(rec.get("original_median_pitch_mm") or 0.0)
    pitch_dev_mm = _nearest_library_pitch(pitch)
    dist_max_mm = float(rec.get("dist_max_mm") or 0.0)
    dist_mean_mm = float(rec.get("dist_mean_mm") or 0.0)
    bolt_source = str(rec.get("bolt_source") or "log")
    source = str(rec.get("source") or "?")
    return {
        "amp_sum": amp_sum,
        "n_inliers": n_inliers,
        "contact_span_mm": contact_span_mm,
        "length_mm": length_mm,
        "frangi_median_mm": frangi_median_mm,
        "original_median_pitch_mm": pitch,
        "pitch_dev_mm": pitch_dev_mm,
        "dist_max_mm": dist_max_mm,
        "dist_mean_mm": dist_mean_mm,
        "bolt_source": bolt_source,
        "source": source,
        "score_in_detector": float(rec.get("confidence") or float("nan")),
        "confidence_in_detector": str(rec.get("confidence_label") or ""),
    }


def main():
    registry = PipelineRegistry()
    register_builtin_pipelines(registry)

    rows = iter_subject_rows(DATASET_ROOT, None)
    rows = [r for r in rows if str(r["subject_id"]) not in EXCLUDE]
    rows.sort(key=lambda r: str(r["subject_id"]))

    fieldnames = [
        "subject_id", "label", "matched_shank", "match_angle", "match_mid_mm",
        "source", "bolt_source",
        "amp_sum", "n_inliers", "contact_span_mm", "length_mm",
        "frangi_median_mm", "original_median_pitch_mm", "pitch_dev_mm",
        "dist_max_mm", "dist_mean_mm",
        "score_in_detector", "confidence_in_detector",
        "trajectory_idx",
    ]
    out_rows = []
    n_matched_total = 0
    n_orphan_total = 0
    n_gt_total = 0
    print(f"running pipeline on {len(rows)} subjects (auto strategy)")
    for row in rows:
        subject_id = str(row["subject_id"])
        gt, _ = load_reference_ground_truth_shanks(row)
        n_gt_total += len(gt)
        ctx, _ = build_detection_context(
            row["ct_path"],
            run_id=f"probe_score_{subject_id}",
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
        n_matched = len(matched)
        n_orphan = len(trajs) - n_matched
        n_matched_total += n_matched
        n_orphan_total += n_orphan
        print(
            f"  {subject_id}: {n_matched}/{len(gt)} matched, {n_orphan} orphans, "
            f"{len(trajs)} emitted"
        )
        for ti, t in enumerate(trajs):
            feats = _features_for(t)
            label = "matched" if ti in matched else "orphan"
            shank, ang, mid_d = matched.get(ti, ("", float("nan"), float("nan")))
            out_rows.append({
                "subject_id": subject_id,
                "label": label,
                "matched_shank": shank,
                "match_angle": f"{ang:.3f}" if ang == ang else "",
                "match_mid_mm": f"{mid_d:.3f}" if mid_d == mid_d else "",
                "trajectory_idx": ti,
                **feats,
            })

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {len(out_rows)} rows to {OUT_TSV}")
    print(
        f"totals: gt={n_gt_total} matched={n_matched_total} "
        f"orphans={n_orphan_total} recall={n_matched_total/max(1, n_gt_total):.4f}"
    )

    # Joint distribution summary.
    matched_rows = [r for r in out_rows if r["label"] == "matched"]
    orphan_rows = [r for r in out_rows if r["label"] == "orphan"]
    print(f"\n=== matched n={len(matched_rows)}  orphan n={len(orphan_rows)} ===")
    print(f"{'feature':>22s}  {'matched (p10/p50/p90)':>34s}  {'orphan (p10/p50/p90)':>34s}")
    for k in ["amp_sum", "n_inliers", "contact_span_mm", "length_mm",
              "frangi_median_mm", "original_median_pitch_mm", "pitch_dev_mm",
              "dist_max_mm", "dist_mean_mm"]:
        m = np.array([float(r[k]) for r in matched_rows], dtype=float)
        o = np.array([float(r[k]) for r in orphan_rows], dtype=float)

        def _pct(a):
            if a.size == 0:
                return ("nan", "nan", "nan")
            a2 = a[~np.isnan(a)]
            if a2.size == 0:
                return ("nan", "nan", "nan")
            return tuple(f"{np.percentile(a2, p):8.2f}" for p in (10, 50, 90))

        mp = _pct(m); op = _pct(o)
        print(f"{k:>22s}  {mp[0]} {mp[1]} {mp[2]}   {op[0]} {op[1]} {op[2]}")

    # Bolt-source breakdown.
    print("\n=== bolt_source x label ===")
    keys = ["log", "hu_rescue", "axis_synth", "none"]
    print(f"{'bolt_source':>14s}  {'matched':>8s}  {'orphan':>8s}")
    for k in keys:
        nm = sum(1 for r in matched_rows if r["bolt_source"] == k)
        no = sum(1 for r in orphan_rows if r["bolt_source"] == k)
        print(f"{k:>14s}  {nm:>8d}  {no:>8d}")
    other_m = sum(1 for r in matched_rows if r["bolt_source"] not in keys)
    other_o = sum(1 for r in orphan_rows if r["bolt_source"] not in keys)
    if other_m or other_o:
        print(f"{'(other)':>14s}  {other_m:>8d}  {other_o:>8d}")

    # Score-cutoff sweep with the proposed weights from the handoff.
    weights = {
        "amp": 1.0, "n": 1.0, "frangi": 1.0, "pitch": 1.0,
        "span": 1.0, "length": 1.0, "depth": 0.5,
        "intracranial": 0.5, "bolt": 1.0,
    }

    def _piecewise_band(x, lo, hi):
        # 1.0 inside [lo, hi]; linear falloff to 0 outside the band over half the band width.
        w = max(1.0, (hi - lo) / 4.0)
        if x < lo - w or x > hi + w:
            return 0.0
        if x < lo:
            return float((x - (lo - w)) / w)
        if x > hi:
            return float(((hi + w) - x) / w)
        return 1.0

    def _bolt_score(src):
        return {"log": 1.0, "hu_rescue": 0.7, "axis_synth": 0.4, "none": 0.1}.get(src, 0.5)

    def _score(r):
        amp_s = min(1.0, float(r["amp_sum"]) / 5000.0)
        n_s = max(0.0, min(1.0, (float(r["n_inliers"]) - 5) / 10.0))
        f_s = min(1.0, float(r["frangi_median_mm"]) / 30.0)
        pd = float(r["pitch_dev_mm"])
        if pd != pd:  # nan
            pitch_s = 0.0
        else:
            pitch_s = max(0.0, 1.0 - pd / 0.5)
        span_s = _piecewise_band(float(r["contact_span_mm"]), 12.0, 90.0)
        length_s = _piecewise_band(float(r["length_mm"]), 30.0, 140.0)
        depth_s = min(1.0, float(r["dist_max_mm"]) / 30.0)
        intra_s = min(1.0, float(r["dist_mean_mm"]) / 10.0)
        bolt_s = _bolt_score(r["bolt_source"])
        total = (
            weights["amp"] * amp_s
            + weights["n"] * n_s
            + weights["frangi"] * f_s
            + weights["pitch"] * pitch_s
            + weights["span"] * span_s
            + weights["length"] * length_s
            + weights["depth"] * depth_s
            + weights["intracranial"] * intra_s
            + weights["bolt"] * bolt_s
        )
        return total / sum(weights.values())  # normalize to [0, 1]-ish

    # Recompute scores in-probe (sanity check that detector & probe agree).
    matched_scores = np.array([_score(r) for r in matched_rows], dtype=float)
    orphan_scores = np.array([_score(r) for r in orphan_rows], dtype=float)
    detector_matched = np.array(
        [float(r["score_in_detector"]) for r in matched_rows], dtype=float
    )
    detector_orphan = np.array(
        [float(r["score_in_detector"]) for r in orphan_rows], dtype=float
    )

    print("\n=== detector-attached score percentiles (read from rec.score) ===")
    if detector_matched.size:
        print(
            f"  matched p10/p50/p90 = "
            f"{np.percentile(detector_matched, 10):.3f} / "
            f"{np.percentile(detector_matched, 50):.3f} / "
            f"{np.percentile(detector_matched, 90):.3f}"
        )
    if detector_orphan.size:
        print(
            f"  orphan  p10/p50/p90 = "
            f"{np.percentile(detector_orphan, 10):.3f} / "
            f"{np.percentile(detector_orphan, 50):.3f} / "
            f"{np.percentile(detector_orphan, 90):.3f}"
        )

    print("\n=== confidence label distribution (in-detector) ===")
    for lab in ("high", "medium", "low"):
        nm = sum(1 for r in matched_rows if r["confidence_in_detector"] == lab)
        no = sum(1 for r in orphan_rows if r["confidence_in_detector"] == lab)
        print(f"  {lab:>6s}  matched={nm:>4d}  orphan={no:>4d}")

    print("\n=== total-score percentiles (normalized) ===")
    print(
        f"  matched p10/p50/p90 = {np.percentile(matched_scores, 10):.3f} "
        f"/ {np.percentile(matched_scores, 50):.3f} "
        f"/ {np.percentile(matched_scores, 90):.3f}"
    )
    if orphan_scores.size:
        print(
            f"  orphan  p10/p50/p90 = {np.percentile(orphan_scores, 10):.3f} "
            f"/ {np.percentile(orphan_scores, 50):.3f} "
            f"/ {np.percentile(orphan_scores, 90):.3f}"
        )
    print("\n=== cutoff sweep ===")
    print(f"{'cutoff':>8s} {'kept_match':>10s} {'kept_orphan':>11s} {'lost_match':>10s} {'recall':>7s}")
    for cutoff in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        km = int((matched_scores >= cutoff).sum())
        ko = int((orphan_scores >= cutoff).sum()) if orphan_scores.size else 0
        lost = len(matched_scores) - km
        print(
            f"{cutoff:>8.2f} {km:>10d} {ko:>11d} {lost:>10d} "
            f"{km/max(1, n_gt_total):>7.4f}"
        )

    # Per-subject orphan dump (helpful for inspecting T1's orphans).
    print("\n=== orphans by subject (in-detector score) ===")
    by_subj = {}
    for r in orphan_rows:
        s = float(r["score_in_detector"])
        by_subj.setdefault(r["subject_id"], []).append((s, r))
    for sid in sorted(by_subj):
        items = sorted(by_subj[sid], key=lambda p: -p[0])
        print(f"\n  {sid}: {len(items)} orphans")
        for s, r in items:
            print(
                f"    score={s:.3f}({r['confidence_in_detector']:>6s})  "
                f"src={r['source']:>6s} "
                f"bolt={r['bolt_source']:>10s}  "
                f"n={int(r['n_inliers']):>2d} amp={float(r['amp_sum']):>7.0f} "
                f"span={float(r['contact_span_mm']):>5.1f} "
                f"len={float(r['length_mm']):>5.1f} "
                f"frangi={float(r['frangi_median_mm']):>6.2f} "
                f"pitch={float(r['original_median_pitch_mm']):>4.2f} "
                f"pd={float(r['pitch_dev_mm']):>4.2f} "
                f"d_max={float(r['dist_max_mm']):>5.1f} "
                f"d_mean={float(r['dist_mean_mm']):>4.1f}"
            )


if __name__ == "__main__":
    main()
