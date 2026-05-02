"""Metal-continuity probe.

For every emitted trajectory on a subject, compute a 1D metal-evidence
profile along the full bolt-tip → deep-tip line at 0.5 mm steps:

    evidence(s) = max(|LoG(s)| / 800, (HU(s) - 0) / 2000)

Then characterize:
  - frac_with_evidence:    fraction of samples with evidence >= 0.15
                            (above insulation level for SEEG wires)
  - longest_gap_mm:         max contiguous distance with evidence < 0.15
  - frac_strong:            fraction with evidence >= 1.0 (contact-class)

Hypothesis: real shanks have continuous wire/insulation/contacts from
bolt to deep tip → frac_with_evidence > 0.7, longest_gap_mm < 8.
Orphans with synthesized extensions or cross-shank chains have one
short cluster + long gap → frac_with_evidence < 0.4, longest_gap_mm >
20 mm.

Run on T1 first to validate the screenshot diagnosis.

Usage:
    cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper && \\
        /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_metal_continuity.py [SUBJECT]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper")
sys.path.insert(0, str(ROOT / "CommonLib"))
sys.path.insert(0, str(ROOT / "PostopCTLocalization"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import map_coordinates

from rosa_detect import contact_pitch_v1_fit as cpfit
from rosa_detect.service import run_contact_pitch_v1
from shank_core.io import image_ijk_ras_matrices
from eval_seeg_localization import (
    build_detection_context, iter_subject_rows,
    load_reference_ground_truth_shanks,
)

DATASET_ROOT = Path("/Users/ammar/Dropbox/thalamus_subjects/seeg_localization")

SAMPLE_STEP_MM = 0.5
LOG_NORM = 800.0
HU_NORM = 2000.0
EVIDENCE_THR = 0.15
GAP_THR_MM = 8.0


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


def _canonical_log_and_ct(raw_img):
    spacing = raw_img.GetSpacing()
    if min(float(s) for s in spacing) < cpfit.CANONICAL_SPACING_MM * 0.95:
        size_in = raw_img.GetSize()
        target = (cpfit.CANONICAL_SPACING_MM,)*3
        target_size = [
            max(1, int(round(size_in[i] * float(spacing[i]) / cpfit.CANONICAL_SPACING_MM)))
            for i in range(3)
        ]
        rs = sitk.ResampleImageFilter()
        rs.SetOutputSpacing(target); rs.SetSize(target_size)
        rs.SetOutputOrigin(raw_img.GetOrigin())
        rs.SetOutputDirection(raw_img.GetDirection())
        rs.SetInterpolator(sitk.sitkLinear)
        rs.SetDefaultPixelValue(-1024)
        img = rs.Execute(raw_img)
        if getattr(cpfit, "RAW_RESAMPLE_GAUSSIAN_SIGMA_MM", 0.0) > 0:
            img = sitk.SmoothingRecursiveGaussian(img, cpfit.RAW_RESAMPLE_GAUSSIAN_SIGMA_MM)
    else:
        img = raw_img
    img_clamped = sitk.Clamp(img, lowerBound=-1024.0, upperBound=cpfit.HU_CLIP_MAX)
    log_sitk = sitk.LaplacianRecursiveGaussian(
        sitk.Cast(img_clamped, sitk.sitkFloat32), sigma=cpfit.LOG_SIGMA_MM,
    )
    log_arr = sitk.GetArrayFromImage(log_sitk)
    ct_arr = sitk.GetArrayFromImage(img).astype(np.float32)
    _, ras_to_ijk = image_ijk_ras_matrices(img)
    return log_arr, ct_arr, np.asarray(ras_to_ijk, dtype=float)


def _sample_along(arr, ras_to_ijk, start, end, step_mm=SAMPLE_STEP_MM):
    s = np.asarray(start, dtype=float); e = np.asarray(end, dtype=float)
    L = float(np.linalg.norm(e - s))
    n = max(8, int(round(L / step_mm)) + 1)
    t = np.linspace(0.0, 1.0, n)
    pts = s[None, :] + t[:, None] * (e - s)[None, :]
    h = np.concatenate([pts, np.ones((n, 1))], axis=1)
    ijk = (ras_to_ijk @ h.T).T[:, :3]
    kji = ijk[:, [2, 1, 0]]
    samples = map_coordinates(arr, kji.T, order=1, mode='nearest')
    return samples.astype(float), L, n


def _continuity(log_samples, ct_samples, step_mm=SAMPLE_STEP_MM):
    """Return (frac_with_evidence, longest_gap_mm, frac_strong, mean_evidence)."""
    log_norm = np.abs(log_samples) / LOG_NORM
    hu_norm = np.maximum(0, ct_samples) / HU_NORM
    evidence = np.maximum(log_norm, hu_norm)
    has_ev = evidence >= EVIDENCE_THR
    frac_with_ev = float(has_ev.mean())
    # Longest gap (consecutive False) in mm.
    longest_gap = 0
    cur = 0
    for v in has_ev:
        if not v:
            cur += 1
            longest_gap = max(longest_gap, cur)
        else:
            cur = 0
    longest_gap_mm = longest_gap * step_mm
    frac_strong = float((evidence >= 1.0).mean())
    mean_ev = float(evidence.mean())
    return frac_with_ev, longest_gap_mm, frac_strong, mean_ev


def _run_dataset(rows):
    matched_all = []; orphan_all = []
    for row in rows:
        sid = str(row["subject_id"])
        gt, _ = load_reference_ground_truth_shanks(row)
        ctx, raw_img = build_detection_context(
            row["ct_path"], run_id=f"probe_cont_{sid}", config={}, extras={},
        )
        ctx["contact_pitch_v1_pitch_strategy"] = "auto"
        result = run_contact_pitch_v1(ctx)
        trajs = list(result.get("trajectories") or [])
        matched = _greedy_match(gt, trajs)
        log_arr, ct_arr, ras_to_ijk = _canonical_log_and_ct(raw_img)
        for ti, t in enumerate(trajs):
            s = np.asarray(t["start_ras"], dtype=float)
            e = np.asarray(t["end_ras"], dtype=float)
            log_samples, _, _ = _sample_along(log_arr, ras_to_ijk, s, e)
            ct_samples, _, _ = _sample_along(ct_arr, ras_to_ijk, s, e)
            frac_ev, gap_mm, frac_strong, mean_ev = _continuity(log_samples, ct_samples)
            rec = {
                "subject": sid, "ti": ti,
                "frac_ev": frac_ev, "gap_mm": gap_mm,
                "frac_strong": frac_strong, "mean_ev": mean_ev,
                "bolt_source": str(t.get("bolt_source", "?")),
                "n_inliers": int(t.get("n_inliers", 0)),
                "L": float(np.linalg.norm(e - s)),
                "contact_span": float(t.get("contact_span_mm", 0.0)),
                "conf": float(t.get("confidence", 0.0)),
                "conf_label": str(t.get("confidence_label", "?")),
                "pitch": float(t.get("original_median_pitch_mm", 0.0)),
            }
            if ti in matched:
                rec["name"] = matched[ti]; matched_all.append(rec)
            else:
                orphan_all.append(rec)
        n_o = len(trajs) - len(matched)
        print(f"  {sid:>4s}: {len(matched)}/{len(gt)} matched, {n_o} orphans")

    print(f"\n=== summary distributions ({len(matched_all)} matched, {len(orphan_all)} orphans) ===")
    for key in ("frac_ev", "gap_mm", "frac_strong", "mean_ev"):
        for kind, recs in (("matched", matched_all), ("orphan", orphan_all)):
            vals = np.array([r[key] for r in recs])
            print(f"  {key:<12s}  {kind:>8s}  n={len(vals):>4d}  "
                  f"p1={np.percentile(vals,1):>6.2f}  p10={np.percentile(vals,10):>6.2f}  "
                  f"p50={np.percentile(vals,50):>6.2f}  p90={np.percentile(vals,90):>6.2f}  "
                  f"p99={np.percentile(vals,99):>6.2f}")

    print("\n=== orphan detail (sorted by gap_mm desc) ===")
    print(f"{'subj':>4s} {'L':>5s}  {'frac_ev':>7s} {'gap':>5s}  {'fr_strong':>9s}  "
          f"{'bolt':>11s}  {'n':>2s}  {'cspan':>5s}  {'conf':>5s}  {'band':>6s}  {'pitch':>5s}")
    orphan_all.sort(key=lambda r: -r["gap_mm"])
    for r in orphan_all:
        flag = " GAP" if r["gap_mm"] > 8 else ("" if r["frac_strong"] > 0.1 else " WEAK")
        print(
            f"{r['subject']:>4s} {r['L']:>5.1f}  {r['frac_ev']:>7.2f} {r['gap_mm']:>5.1f}  "
            f"{r['frac_strong']:>9.2f}  {r['bolt_source']:>11s}  {r['n_inliers']:>2d}  "
            f"{r['contact_span']:>5.1f}  {r['conf']:>5.2f}  {r['conf_label']:>6s}  "
            f"{r['pitch']:>5.2f}{flag}"
        )

    # How many orphans would a `frac_strong < 0.1` gate drop?
    drop_count = sum(1 for r in orphan_all if r["frac_strong"] < 0.1)
    matched_lost = sum(1 for r in matched_all if r["frac_strong"] < 0.1)
    print(f"\nSimulated gate `frac_strong >= 0.1`: drops {drop_count}/{len(orphan_all)} orphans, "
          f"loses {matched_lost}/{len(matched_all)} matched")
    drop_count_g = sum(1 for r in orphan_all if r["gap_mm"] > 8)
    matched_lost_g = sum(1 for r in matched_all if r["gap_mm"] > 8)
    print(f"Simulated gate `gap_mm <= 8`:        drops {drop_count_g}/{len(orphan_all)} orphans, "
          f"loses {matched_lost_g}/{len(matched_all)} matched")

    # Combined gate
    drop_count_c = sum(1 for r in orphan_all if r["frac_strong"] < 0.1 or r["gap_mm"] > 8)
    matched_lost_c = sum(1 for r in matched_all if r["frac_strong"] < 0.1 or r["gap_mm"] > 8)
    print(f"Combined `frac_strong>=0.1 AND gap_mm<=8`: drops {drop_count_c}/{len(orphan_all)} orphans, "
          f"loses {matched_lost_c}/{len(matched_all)} matched")
    if matched_lost_c:
        print("\n  matched losses under combined gate:")
        for r in matched_all:
            if r["frac_strong"] < 0.1 or r["gap_mm"] > 8:
                print(f"    {r['subject']:>4s} {r.get('name','?'):>6s} "
                      f"frac_strong={r['frac_strong']:.2f} gap={r['gap_mm']:.1f}")
    return 0


def main():
    subject_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if subject_arg == "ALL":
        subjects_filter = None
    elif subject_arg:
        subjects_filter = {subject_arg}
    else:
        subjects_filter = {"T1"}
    rows = iter_subject_rows(DATASET_ROOT, subjects_filter)
    rows = [r for r in rows if str(r["subject_id"]) not in {"T17", "T19", "T21"}]
    rows.sort(key=lambda r: int(str(r["subject_id"]).lstrip("T")))
    if not rows:
        print(f"ERROR: no subjects found"); return 1
    if len(rows) > 1:
        return _run_dataset(rows)
    row = rows[0]
    gt, _ = load_reference_ground_truth_shanks(row)

    ctx, raw_img = build_detection_context(
        row["ct_path"], run_id=f"probe_cont_{subject}", config={}, extras={},
    )
    ctx["contact_pitch_v1_pitch_strategy"] = "auto"
    result = run_contact_pitch_v1(ctx)
    trajs = list(result.get("trajectories") or [])
    matched = _greedy_match(gt, trajs)
    log_arr, ct_arr, ras_to_ijk = _canonical_log_and_ct(raw_img)

    print(f"{subject}: {len(trajs)} emissions ({len(matched)} matched, "
          f"{len(trajs) - len(matched)} orphans)")
    print(f"thresholds: evidence >= {EVIDENCE_THR}, gap > {GAP_THR_MM} mm flagged\n")

    rows_out = []
    for ti, t in enumerate(trajs):
        s = np.asarray(t["start_ras"], dtype=float)
        e = np.asarray(t["end_ras"], dtype=float)
        L = float(np.linalg.norm(e - s))
        log_samples, _, n = _sample_along(log_arr, ras_to_ijk, s, e)
        ct_samples, _, _ = _sample_along(ct_arr, ras_to_ijk, s, e)
        frac_ev, gap_mm, frac_strong, mean_ev = _continuity(log_samples, ct_samples)
        kind = "matched" if ti in matched else "orphan"
        name = matched.get(ti, "")
        rows_out.append({
            "ti": ti, "kind": kind, "name": name,
            "L": L, "frac_ev": frac_ev, "gap_mm": gap_mm,
            "frac_strong": frac_strong, "mean_ev": mean_ev,
            "bolt_source": str(t.get("bolt_source", "?")),
            "n_inliers": int(t.get("n_inliers", 0)),
            "contact_span": float(t.get("contact_span_mm", 0.0)),
            "conf": float(t.get("confidence", 0.0)),
            "conf_label": str(t.get("confidence_label", "?")),
            "pitch": float(t.get("original_median_pitch_mm", 0.0)),
        })

    print(f"{'idx':>3s} {'kind':>7s} {'name':>5s}  {'L':>5s}  {'frac_ev':>7s}  "
          f"{'gap_mm':>6s}  {'frac_strong':>11s}  {'bolt':>11s}  {'n':>2s}  "
          f"{'contact':>7s}  {'conf':>5s}  flag")
    rows_out.sort(key=lambda r: (r["kind"] != "orphan", -r["gap_mm"]))
    for r in rows_out:
        flag = ""
        if r["gap_mm"] > GAP_THR_MM:
            flag = f" GAP-{r['gap_mm']:.0f}mm"
        if r["frac_ev"] < 0.5:
            flag += " low-ev"
        print(
            f"{r['ti']:>3d} {r['kind']:>7s} {r['name']:>5s}  "
            f"{r['L']:>5.1f}  {r['frac_ev']:>7.2f}  "
            f"{r['gap_mm']:>6.1f}  {r['frac_strong']:>11.2f}  "
            f"{r['bolt_source']:>11s}  {r['n_inliers']:>2d}  "
            f"{r['contact_span']:>7.1f}  {r['conf']:>5.2f}{flag}"
        )

    # Summary stats: matched vs orphan
    print("\n=== summary distributions ===")
    m_recs = [r for r in rows_out if r["kind"] == "matched"]
    o_recs = [r for r in rows_out if r["kind"] == "orphan"]
    for key in ("frac_ev", "gap_mm", "frac_strong", "mean_ev"):
        for kind, recs in (("matched", m_recs), ("orphan", o_recs)):
            if not recs: continue
            vals = np.array([r[key] for r in recs])
            print(f"  {key:<12s}  {kind:>8s}  n={len(vals):>3d}  "
                  f"p10={np.percentile(vals,10):>6.2f}  "
                  f"p50={np.percentile(vals,50):>6.2f}  "
                  f"p90={np.percentile(vals,90):>6.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
