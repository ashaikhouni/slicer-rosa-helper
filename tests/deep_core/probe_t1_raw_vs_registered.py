"""T1 raw-vs-registered detection-delta probe with SITK CT-CT rigid registration.

Background
----------
Unlike T7 (whose raw and registered CTs share physical space — the raw
just had higher-resolution voxels), T1's raw post-op CT lives in a
completely different RAS coordinate frame from the registered version
(z origin differs by 700+ mm; in-plane rotation ~5°). The dataset's
registration pipeline mapped raw post-op CT → reference (patient T1
MRI) space; trajectories detected on the raw image are in scanner RAS
and don't directly compare against the registered-RAS GT.

This probe:
  1. Reads both images.
  2. Runs SITK rigid CT-CT registration (Mattes MI) to recover the
     raw → registered transform.
  3. Runs the contact_pitch_v1 pipeline on each.
  4. Applies the registration transform to raw trajectory endpoints
     so they land in registered-RAS.
  5. Runs the same greedy GT match on both sets.

Output: per-config emissions count, GT match recall, FP count, and
high-band counts. Confirms whether raw + new sigma=0.7 smoothing
detects the same 12 GT shanks as registered.

Run:
    cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper && \\
        /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_t1_raw_vs_registered.py
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
REGISTERED_CT = Path("/Users/ammar/Dropbox/thalamus_subjects/seeg_localization/contact_label_dataset/ct/T1_ct.nii.gz")
RAW_CT = Path("/Users/ammar/Dropbox/thalamus_subjects/T1/post_ct/postop_CT.nii.gz")

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


def _register_ct_to_ct(fixed: sitk.Image, moving: sitk.Image) -> sitk.Transform:
    """Rigid CT-CT registration. Returns transform mapping moving -> fixed
    in physical / RAS space.

    Note: SITK's convention is that the returned transform takes points
    in FIXED space and finds where they came from in MOVING space (used
    by Resample). To map a moving-space POINT into fixed space, we need
    the INVERSE of this transform.
    """
    # Convert to LPS-cast float32 to ensure consistent metric eval.
    fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)

    init_tx = sitk.CenteredTransformInitializer(
        fixed_f, moving_f, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.05, seed=42)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=200,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10,
    )
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    R.SetInitialTransform(init_tx, inPlace=False)

    print("  registering CT-CT (Mattes MI, 3 levels)...")
    tx = R.Execute(fixed_f, moving_f)
    print(f"  final metric value: {R.GetMetricValue():.4f}")
    print(f"  iterations: {R.GetOptimizerIteration()}")
    print(f"  stop condition: {R.GetOptimizerStopConditionDescription()}")
    return tx


def _run_pipeline(img: sitk.Image) -> list:
    ijk_to_ras_mat, ras_to_ijk_mat = image_ijk_ras_matrices(img)
    return cpfit.run_two_stage_detection(
        img, ijk_to_ras_mat, ras_to_ijk_mat,
        return_features=False,
        pitch_strategy="auto",
    )


def _summarise(label: str, trajs: list, gt) -> None:
    matched = _greedy_match(gt, trajs)
    bands_m = {"high":0,"medium":0,"low":0}
    bands_o = {"high":0,"medium":0,"low":0}
    bolts_m: dict[str,int] = {}
    bolts_o: dict[str,int] = {}
    for ti, t in enumerate(trajs):
        cl = str(t.get("confidence_label","?"))
        bs = str(t.get("bolt_source","?"))
        if ti in matched:
            bands_m[cl] = bands_m.get(cl,0)+1
            bolts_m[bs] = bolts_m.get(bs,0)+1
        else:
            bands_o[cl] = bands_o.get(cl,0)+1
            bolts_o[bs] = bolts_o.get(bs,0)+1
    print(f"\n=== {label} ===")
    print(f"  GT={len(gt)}  emissions={len(trajs)}  matched={len(matched)}/{len(gt)}  orphans={len(trajs)-len(matched)}")
    print(f"  matched bands: {bands_m}    bolt: {dict(sorted(bolts_m.items()))}")
    print(f"  orphan  bands: {bands_o}    bolt: {dict(sorted(bolts_o.items()))}")
    if matched:
        gt_names = {str(g.shank) for g in gt}
        matched_names = {info[0] for info in matched.values()}
        unmatched = sorted(gt_names - matched_names)
        if unmatched:
            print(f"  unmatched GT shanks: {unmatched}")


def main():
    rows = iter_subject_rows(DATASET_ROOT, {"T1"})
    if not rows:
        print("ERROR: T1 not in dataset manifest")
        return 1
    row = rows[0]
    gt, _ = load_reference_ground_truth_shanks(row)
    print(f"T1: {len(gt)} GT shanks (in registered RAS)")

    img_reg = sitk.ReadImage(str(REGISTERED_CT))
    img_raw = sitk.ReadImage(str(RAW_CT))
    print(f"\nregistered: spacing={tuple(round(s,3) for s in img_reg.GetSpacing())}, size={img_reg.GetSize()}")
    print(f"raw:        spacing={tuple(round(s,3) for s in img_raw.GetSpacing())}, size={img_raw.GetSize()}")

    # 1. Run pipeline on registered (no extra transform needed).
    trajs_reg = _run_pipeline(img_reg)
    _summarise("REGISTERED (production)", trajs_reg, gt)

    # 2. Register raw -> registered (CT-CT rigid).
    tx_fixed_to_moving = _register_ct_to_ct(fixed=img_reg, moving=img_raw)
    tx_moving_to_fixed = tx_fixed_to_moving.GetInverse()

    def _ras_raw_to_reg(p_ras):
        p_lps = np.array([-p_ras[0], -p_ras[1], p_ras[2]])
        p_lps_reg = tx_moving_to_fixed.TransformPoint(tuple(p_lps.tolist()))
        return np.array([-p_lps_reg[0], -p_lps_reg[1], p_lps_reg[2]])

    # 3. Sweep σ values on raw input.
    orig_sigma = cpfit.RAW_RESAMPLE_GAUSSIAN_SIGMA_MM
    sigma_results = []
    for sigma in [0.0, 0.3, 0.5, 0.7]:
        cpfit.RAW_RESAMPLE_GAUSSIAN_SIGMA_MM = sigma
        trajs_raw = _run_pipeline(img_raw)
        trajs_raw_in_reg = []
        for t in trajs_raw:
            new_t = dict(t)
            new_t["start_ras"] = _ras_raw_to_reg(np.asarray(t["start_ras"], dtype=float))
            new_t["end_ras"] = _ras_raw_to_reg(np.asarray(t["end_ras"], dtype=float))
            trajs_raw_in_reg.append(new_t)
        _summarise(f"RAW (sigma={sigma}) -> registered RAS", trajs_raw_in_reg, gt)
        matched = _greedy_match(gt, trajs_raw_in_reg)
        sigma_results.append((sigma, len(trajs_raw), len(matched), len(trajs_raw)-len(matched)))
    cpfit.RAW_RESAMPLE_GAUSSIAN_SIGMA_MM = orig_sigma

    # 4. Summary table comparing registered vs all sigma sweeps.
    print("\n=== sigma sweep summary ===")
    matched_reg = _greedy_match(gt, trajs_reg)
    print(f"{'config':<40s} {'emit':>5s} {'matched':>8s} {'orphan':>7s}")
    print(f"{'REGISTERED (no smoothing fires)':<40s} {len(trajs_reg):>5d} {len(matched_reg):>8d} {len(trajs_reg)-len(matched_reg):>7d}")
    for sigma, n_emit, n_match, n_orph in sigma_results:
        print(f"{f'RAW + sigma={sigma}':<40s} {n_emit:>5d} {n_match:>8d} {n_orph:>7d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
