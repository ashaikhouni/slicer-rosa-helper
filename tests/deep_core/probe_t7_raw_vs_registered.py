"""T7 raw-vs-registered detection-delta probe.

Background
----------
The dataset's registered T7 (1mm iso) emits a clean 14/14 with 0 orphans.
The raw post_ct.nii.gz (0.415 mm anisotropic) — after the canonical-1mm
resample at pipeline entry — emits 21 trajectories (14 matched + 7
orphans). The hypothesis from the prior session memory is that the
canonical resample's linear interpolation does less anti-aliasing than
the dataset's original registration (probably B-spline), so the raw
input keeps high-frequency aliasing artifacts that LoG sees as extra
"contacts".

This probe runs the full pipeline on:
  1. registered T7 (no resample needed)
  2. raw T7 with sitkLinear resample (current production)
  3. raw T7 with sitkBSpline resample
  4. raw T7 with sitkLinear + post-resample Gaussian smoothing

For each run we report: total emissions, matched / orphan split, and the
orphan dump (score, bolt_source, n_inliers, amp). The goal is to
identify the cleanest interp / smoothing choice that converges raw
output to the registered output.

Run:
    cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper && \\
        /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_t7_raw_vs_registered.py
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

from rosa_detect import contact_pitch_v1_fit as cpfit
from shank_core.io import image_ijk_ras_matrices
from eval_seeg_localization import (
    iter_subject_rows,
    load_reference_ground_truth_shanks,
)


DATASET_ROOT = Path("/Users/ammar/Dropbox/thalamus_subjects/seeg_localization")
RAW_CT = Path("/Users/ammar/Dropbox/thalamus_subjects/T7/Post_ct/post_ct.nii.gz")

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


def _run(img: sitk.Image, label: str, gt) -> tuple[list, dict]:
    """Run the full pipeline; return trajectories + summary dict."""
    ijk_to_ras_mat, ras_to_ijk_mat = image_ijk_ras_matrices(img)
    trajs = cpfit.run_two_stage_detection(
        img, ijk_to_ras_mat, ras_to_ijk_mat,
        return_features=False,
        pitch_strategy="auto",
    )
    matched = _greedy_match(gt, trajs)
    n_matched = len(matched)
    n_orphan = len(trajs) - n_matched

    # Bolt-source breakdown.
    bolt_breakdown_m: dict[str, int] = {}
    bolt_breakdown_o: dict[str, int] = {}
    label_count_m: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    label_count_o: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    orphans = []
    for ti, t in enumerate(trajs):
        bs = str(t.get("bolt_source", "?"))
        cl = str(t.get("confidence_label", "?"))
        if ti in matched:
            bolt_breakdown_m[bs] = bolt_breakdown_m.get(bs, 0) + 1
            label_count_m[cl] = label_count_m.get(cl, 0) + 1
        else:
            bolt_breakdown_o[bs] = bolt_breakdown_o.get(bs, 0) + 1
            label_count_o[cl] = label_count_o.get(cl, 0) + 1
            orphans.append(t)

    summary = {
        "label": label,
        "n_total": len(trajs),
        "n_matched": n_matched,
        "n_orphan": n_orphan,
        "matched_bolt": bolt_breakdown_m,
        "orphan_bolt": bolt_breakdown_o,
        "matched_band": label_count_m,
        "orphan_band": label_count_o,
        "orphans": orphans,
    }
    return trajs, summary


def _print_summary(s: dict):
    print(f"\n=== {s['label']} ===")
    print(f"  emissions: {s['n_total']} ({s['n_matched']} matched, {s['n_orphan']} orphan)")
    print(f"  matched bolt: {dict(sorted(s['matched_bolt'].items()))}")
    print(f"  orphan  bolt: {dict(sorted(s['orphan_bolt'].items()))}")
    print(f"  matched band: high={s['matched_band']['high']}  medium={s['matched_band']['medium']}  low={s['matched_band']['low']}")
    print(f"  orphan  band: high={s['orphan_band']['high']}  medium={s['orphan_band']['medium']}  low={s['orphan_band']['low']}")
    if s["orphans"]:
        print(f"  orphan dump (score, bolt, n, amp, frangi, pitch, pd, len, span):")
        for o in sorted(s["orphans"], key=lambda r: -float(r.get("confidence", 0.0))):
            print(
                f"    score={float(o.get('confidence', 0.0)):.3f}({str(o.get('confidence_label','?'))[:6]:>6s})  "
                f"bolt={str(o.get('bolt_source','?')):>10s}  "
                f"n={int(o.get('n_inliers',0)):>2d} "
                f"amp={float(o.get('amp_sum',0.0)):>7.0f} "
                f"frangi={float(o.get('frangi_median_mm',0.0)):>6.1f} "
                f"pitch={float(o.get('original_median_pitch_mm',0.0)):>4.2f} "
                f"pd={(min(abs(float(o.get('original_median_pitch_mm',0.0)) - p) for p in cpfit.LIBRARY_PITCHES_MM) if float(o.get('original_median_pitch_mm',0.0))>0 else float('nan')):>4.2f} "
                f"len={float(o.get('length_mm',0.0)):>5.1f} "
                f"span={float(o.get('contact_span_mm',0.0)):>5.1f}"
            )


def _resample_to_canonical(img: sitk.Image, interp, sigma_mm: float = 0.0) -> sitk.Image:
    """Custom canonical-1mm resample with selectable interpolator and
    optional pre/post Gaussian smoothing.

    sigma_mm > 0 applies an isotropic Gaussian smoothing AFTER resampling
    (in mm units of the canonical grid).
    """
    spacing = img.GetSpacing()
    if min(float(s) for s in spacing) >= cpfit.CANONICAL_SPACING_MM * 0.95:
        return img
    size_in = img.GetSize()
    target_spacing = (cpfit.CANONICAL_SPACING_MM,) * 3
    target_size = [
        max(1, int(round(size_in[i] * float(spacing[i]) / cpfit.CANONICAL_SPACING_MM)))
        for i in range(3)
    ]
    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing(target_spacing)
    rs.SetSize(target_size)
    rs.SetOutputOrigin(img.GetOrigin())
    rs.SetOutputDirection(img.GetDirection())
    rs.SetInterpolator(interp)
    rs.SetDefaultPixelValue(-1024)
    out = rs.Execute(img)
    if sigma_mm > 0:
        out = sitk.SmoothingRecursiveGaussian(out, sigma_mm)
    return out


def main():
    rows = iter_subject_rows(DATASET_ROOT, {"T7"})
    if not rows:
        print("ERROR: T7 not in dataset manifest")
        return 1
    row = rows[0]
    gt, _ = load_reference_ground_truth_shanks(row)
    print(f"T7: {len(gt)} GT shanks")

    registered_path = row["ct_path"]
    print(f"\nregistered: {registered_path}")
    print(f"raw:        {RAW_CT}")

    # 1. Registered (no resample needed; 1mm iso already)
    img_reg = sitk.ReadImage(str(registered_path))
    print(f"  registered spacing: {img_reg.GetSpacing()}, size: {img_reg.GetSize()}")
    _trajs_r, sum_reg = _run(img_reg, "REGISTERED (production path)", gt)
    _print_summary(sum_reg)

    # 2. Raw with the production sitkLinear canonical resample
    img_raw = sitk.ReadImage(str(RAW_CT))
    print(f"\n  raw spacing: {img_raw.GetSpacing()}, size: {img_raw.GetSize()}")
    img_raw_lin = _resample_to_canonical(img_raw, sitk.sitkLinear)
    print(f"  raw → linear resample → spacing: {img_raw_lin.GetSpacing()}, size: {img_raw_lin.GetSize()}")
    # Skip the inner resample — we already did it; pass through canonical-spaced image.
    _trajs_l, sum_lin = _run(img_raw_lin, "RAW + sitkLinear (current production)", gt)
    _print_summary(sum_lin)

    # 3. Raw with sitkBSpline canonical resample
    img_raw_bsp = _resample_to_canonical(img_raw, sitk.sitkBSpline)
    _trajs_b, sum_bsp = _run(img_raw_bsp, "RAW + sitkBSpline", gt)
    _print_summary(sum_bsp)

    # 4. Raw with sitkLinear + post-resample Gaussian σ=0.5 mm
    img_raw_g05 = _resample_to_canonical(img_raw, sitk.sitkLinear, sigma_mm=0.5)
    _trajs_g05, sum_g05 = _run(img_raw_g05, "RAW + sitkLinear + Gaussian σ=0.5 mm", gt)
    _print_summary(sum_g05)

    # 5. Raw with sitkLinear + post-resample Gaussian σ=0.7 mm
    img_raw_g07 = _resample_to_canonical(img_raw, sitk.sitkLinear, sigma_mm=0.7)
    _trajs_g07, sum_g07 = _run(img_raw_g07, "RAW + sitkLinear + Gaussian σ=0.7 mm", gt)
    _print_summary(sum_g07)

    print("\n=== summary table ===")
    print(f"{'config':<46s} {'emissions':>9s} {'matched':>8s} {'orphan':>7s}")
    for s in (sum_reg, sum_lin, sum_bsp, sum_g05, sum_g07):
        print(f"{s['label']:<46s} {s['n_total']:>9d} {s['n_matched']:>8d} {s['n_orphan']:>7d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
