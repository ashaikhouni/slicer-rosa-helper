"""S56-raw vs T23-registered detection-delta probe.

Background
----------
S56 is the patient whose post-op CT is the source of T23 in the
SEEG dataset (T23 = S56 rigidly registered to MRI + resampled to
1mm iso). Same physical implantation; the detector should emit the
same shank set on both inputs. The user reports S56 raw emits FEWER
trajectories than T23 registered. This probe measures the gap and
sweeps interpolator + post-resample smoothing variants on the raw
input to find a configuration that converges S56-raw onto
T23-registered.

Mirrors `probe_t7_raw_vs_registered.py`, with one twist: T23 GT lives
in MRI/RAS space, so it's only valid for matching against the T23
run. The S56 run reports emissions counts and bolt-source breakdown
only (no GT match, since the CT-to-MRI registration matrix isn't
applied to S56's emissions here).

Run:
    cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper && \\
        /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_s56_raw_vs_t23_registered.py
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

from postop_ct_localization import contact_pitch_v1_fit as cpfit
from shank_core.io import image_ijk_ras_matrices
from eval_seeg_localization import (
    iter_subject_rows,
    load_reference_ground_truth_shanks,
)


DATASET_ROOT = Path("/Users/ammar/Dropbox/thalamus_subjects/seeg_localization")
RAW_CT = Path("/Users/ammar/Documents/Data/imaging/S56/Post_CT/S56_CT.nii.gz")

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
    matched_gi = {}
    for _s, gi, ti, ang, mid_d in pairs:
        if gi in used_g or ti in used_t:
            continue
        used_g.add(gi); used_t.add(ti)
        matched_ti[ti] = (str(gt_shanks[gi].shank), ang, mid_d)
        matched_gi[gi] = (ti, ang, mid_d)
    return matched_ti, matched_gi


def _run(img: sitk.Image, label: str, gt=None) -> tuple[list, dict]:
    ijk_to_ras_mat, ras_to_ijk_mat = image_ijk_ras_matrices(img)
    trajs = cpfit.run_two_stage_detection(
        img, ijk_to_ras_mat, ras_to_ijk_mat,
        return_features=False,
        pitch_strategy="auto",
    )
    if gt is None:
        matched_ti, matched_gi = {}, {}
    else:
        matched_ti, matched_gi = _greedy_match(gt, trajs)
    n_matched = len(matched_ti)
    n_orphan = len(trajs) - n_matched

    bolt_breakdown_m: dict[str, int] = {}
    bolt_breakdown_o: dict[str, int] = {}
    label_count_all: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    orphans = []
    matched = []
    for ti, t in enumerate(trajs):
        bs = str(t.get("bolt_source", "?"))
        cl = str(t.get("confidence_label", "?"))
        label_count_all[cl] = label_count_all.get(cl, 0) + 1
        if ti in matched_ti:
            bolt_breakdown_m[bs] = bolt_breakdown_m.get(bs, 0) + 1
            matched.append((matched_ti[ti][0], t))
        else:
            bolt_breakdown_o[bs] = bolt_breakdown_o.get(bs, 0) + 1
            orphans.append(t)

    missing_gt = []
    if gt is not None:
        for gi, g in enumerate(gt):
            if gi not in matched_gi:
                missing_gt.append(str(g.shank))

    return trajs, {
        "label": label,
        "n_total": len(trajs),
        "n_matched": n_matched,
        "n_orphan": n_orphan,
        "matched_bolt": bolt_breakdown_m,
        "orphan_bolt": bolt_breakdown_o,
        "all_band": label_count_all,
        "matched": matched,
        "orphans": orphans,
        "missing_gt": missing_gt,
    }


def _print_summary(s: dict, with_gt: bool):
    print(f"\n=== {s['label']} ===")
    if with_gt:
        print(f"  emissions: {s['n_total']} ({s['n_matched']} matched, {s['n_orphan']} orphan)")
        if s["missing_gt"]:
            print(f"  GT shanks NOT matched: {s['missing_gt']}")
    else:
        print(f"  emissions: {s['n_total']}")
    print(f"  band: high={s['all_band']['high']}  medium={s['all_band']['medium']}  low={s['all_band']['low']}")
    if s["matched_bolt"]:
        print(f"  matched bolt: {dict(sorted(s['matched_bolt'].items()))}")
    if s["orphan_bolt"]:
        print(f"  orphan  bolt: {dict(sorted(s['orphan_bolt'].items()))}")


def _resample_to_canonical(img: sitk.Image, interp, sigma_mm: float = 0.0) -> sitk.Image:
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
    rows = iter_subject_rows(DATASET_ROOT, {"T23"})
    if not rows:
        print("ERROR: T23 not in dataset manifest")
        return 1
    row = rows[0]
    gt, _ = load_reference_ground_truth_shanks(row)
    print(f"T23 GT: {len(gt)} shanks ({[g.shank for g in gt]})")

    registered_path = row["ct_path"]
    print(f"\nregistered (T23): {registered_path}")
    print(f"raw        (S56): {RAW_CT}")

    img_reg = sitk.ReadImage(str(registered_path))
    print(f"  T23 registered: spacing={img_reg.GetSpacing()}, size={img_reg.GetSize()}")
    _, sum_reg = _run(img_reg, "T23 REGISTERED (production path)", gt)
    _print_summary(sum_reg, with_gt=True)

    img_raw = sitk.ReadImage(str(RAW_CT))
    print(f"\n  S56 raw: spacing={img_raw.GetSpacing()}, size={img_raw.GetSize()}")
    img_raw_lin = _resample_to_canonical(img_raw, sitk.sitkLinear)
    print(f"  S56 raw -> sitkLinear -> spacing={img_raw_lin.GetSpacing()}, size={img_raw_lin.GetSize()}")
    _, sum_lin = _run(img_raw_lin, "S56 RAW + sitkLinear (current production)")
    _print_summary(sum_lin, with_gt=False)

    img_raw_bsp = _resample_to_canonical(img_raw, sitk.sitkBSpline)
    _, sum_bsp = _run(img_raw_bsp, "S56 RAW + sitkBSpline")
    _print_summary(sum_bsp, with_gt=False)

    img_raw_g05 = _resample_to_canonical(img_raw, sitk.sitkLinear, sigma_mm=0.5)
    _, sum_g05 = _run(img_raw_g05, "S56 RAW + sitkLinear + Gaussian σ=0.5 mm")
    _print_summary(sum_g05, with_gt=False)

    img_raw_g07 = _resample_to_canonical(img_raw, sitk.sitkLinear, sigma_mm=0.7)
    _, sum_g07 = _run(img_raw_g07, "S56 RAW + sitkLinear + Gaussian σ=0.7 mm")
    _print_summary(sum_g07, with_gt=False)

    print("\n=== summary table ===")
    print(f"{'config':<48s} {'emissions':>9s} {'high':>5s} {'med':>5s} {'low':>5s}")
    for s in (sum_reg, sum_lin, sum_bsp, sum_g05, sum_g07):
        b = s["all_band"]
        print(
            f"{s['label']:<48s} {s['n_total']:>9d} "
            f"{b['high']:>5d} {b['medium']:>5d} {b['low']:>5d}"
        )

    print("\n=== per-trajectory dump (sorted by midpoint Z, descending) ===")

    def _dump(label, trajs):
        print(f"\n-- {label} --")
        rows = []
        for t in trajs:
            s = np.asarray(t["start_ras"], dtype=float)
            e = np.asarray(t["end_ras"], dtype=float)
            mid = 0.5 * (s + e)
            length = float(np.linalg.norm(e - s))
            rows.append((mid[2], s, e, mid, length, t))
        rows.sort(key=lambda r: -r[0])
        for _, s, e, mid, length, t in rows:
            print(
                f"   conf={float(t.get('confidence', 0)):.2f}({str(t.get('confidence_label','?'))[:6]:>6s})  "
                f"bolt={str(t.get('bolt_source','?')):>10s}  "
                f"n={int(t.get('n_inliers',0)):>2d}  "
                f"pitch={float(t.get('original_median_pitch_mm',0)):.2f}  "
                f"len={length:.1f}  "
                f"mid=({mid[0]:+.1f},{mid[1]:+.1f},{mid[2]:+.1f})"
            )

    _dump(sum_reg["label"], [t for _, t in sum_reg["matched"]] + sum_reg["orphans"])
    _dump(sum_lin["label"], sum_lin["orphans"])  # everything is "orphan" since no GT
    _dump(sum_bsp["label"], sum_bsp["orphans"])
    _dump(sum_g05["label"], sum_g05["orphans"])
    _dump(sum_g07["label"], sum_g07["orphans"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
