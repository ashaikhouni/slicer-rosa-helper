"""Probe: gradient-magnitude filter as a bolt/skull anatomical detector.

For a given subject:
  1. Apply GradientMagnitudeRecursiveGaussianImageFilter (sigma=1) to the CT.
  2. Save the GM volume to /tmp/gm_{subject}.nii.gz for Slicer inspection.
  3. Report CT and GM intensity stats (to confirm whether T2 clipping damages
     the GM signal the way it damages HU).
  4. For T22 (which has GT shanks), for each GT shank: extend the shallow
     endpoint outward along -direction, sample CT and GM along a line from
     -30mm to +30mm relative to the shallow start, and print the profile so
     we can see if a bolt-through-skull event is detectable as a GM peak.

Usage:
  /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
    tests/deep_core/probe_gm_bolt_entry.py T22
  /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
    tests/deep_core/probe_gm_bolt_entry.py T2
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))
sys.path.insert(0, str(REPO_ROOT / "PostopCTLocalization"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

DATASET_ROOT = Path(
    os.environ.get(
        "ROSA_SEEG_DATASET",
        "/Users/ammar/Dropbox/thalamus_subjects/seeg_localization",
    )
)


def _unit(v):
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def _ras_to_kji(pt_ras, ras_to_ijk_mat):
    h = np.array([pt_ras[0], pt_ras[1], pt_ras[2], 1.0])
    ijk = ras_to_ijk_mat @ h
    return np.array([ijk[2], ijk[1], ijk[0]], dtype=float)


def _sample(arr, kji_float):
    k, j, i = kji_float
    ki, ji, ii = int(round(k)), int(round(j)), int(round(i))
    s = arr.shape
    if 0 <= ki < s[0] and 0 <= ji < s[1] and 0 <= ii < s[2]:
        return float(arr[ki, ji, ii])
    return float("nan")


def run(subject_id):
    import SimpleITK as sitk
    from shank_core.io import image_ijk_ras_matrices
    from eval_seeg_localization import iter_subject_rows, load_ground_truth_shanks

    rows = iter_subject_rows(DATASET_ROOT, {subject_id})
    if not rows:
        raise SystemExit(f"subject {subject_id} not in manifest")
    row = rows[0]
    ct_path = row["ct_path"]
    print(f"# subject={subject_id}")
    print(f"# ct={ct_path}")

    img = sitk.ReadImage(ct_path)
    ct_arr_kji = sitk.GetArrayFromImage(img).astype(np.float32)

    print(
        f"# CT shape={ct_arr_kji.shape} "
        f"HU min={ct_arr_kji.min():.0f} max={ct_arr_kji.max():.0f} "
        f"p99={np.percentile(ct_arr_kji, 99):.0f} "
        f"p999={np.percentile(ct_arr_kji, 99.9):.0f}"
    )
    sat_count = int((ct_arr_kji >= ct_arr_kji.max() - 1).sum())
    print(f"# voxels at max HU: {sat_count} (saturation signature if many)")

    gm_filter = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
    gm_filter.SetSigma(1.0)
    gm_filter.SetNormalizeAcrossScale(False)
    gm_img = gm_filter.Execute(img)
    gm_arr_kji = sitk.GetArrayFromImage(gm_img).astype(np.float32)

    print(
        f"# GM min={gm_arr_kji.min():.1f} max={gm_arr_kji.max():.1f} "
        f"mean={gm_arr_kji.mean():.1f} "
        f"p50={np.percentile(gm_arr_kji, 50):.1f} "
        f"p95={np.percentile(gm_arr_kji, 95):.1f} "
        f"p999={np.percentile(gm_arr_kji, 99.9):.1f}"
    )

    out_gm = f"/tmp/gm_{subject_id}.nii.gz"
    sitk.WriteImage(gm_img, out_gm)
    print(f"# wrote {out_gm}")

    shanks_path = row.get("shanks_path")
    labels_path = row["labels_path"]
    if not labels_path or not Path(labels_path).exists():
        print("# no GT labels available; skipping shank probing")
        return

    gt_shanks = load_ground_truth_shanks(labels_path, shanks_path)
    if not gt_shanks:
        print("# no GT shanks parsed")
        return

    ijk_to_ras_mat, ras_to_ijk_mat = image_ijk_ras_matrices(img)
    ijk_to_ras_mat = np.asarray(ijk_to_ras_mat, dtype=float)
    ras_to_ijk_mat = np.asarray(ras_to_ijk_mat, dtype=float)

    print(f"# GT shanks = {len(gt_shanks)}")
    print(
        "# sample step along shank extension: ct_hu / gm  at offsets -30..+30mm\n"
        "# start = GT shallow end (contact 1 extrapolation). "
        "outward = -direction (from deep to shallow end, extended)."
    )

    offsets = np.linspace(-30.0, 30.0, 61)

    for gt in gt_shanks:
        start = np.asarray(gt.start_ras, dtype=float)
        direction = _unit(np.asarray(gt.direction_ras, dtype=float))

        ct_profile = []
        gm_profile = []
        for off in offsets:
            pt = start + off * (-direction)
            kji = _ras_to_kji(pt, ras_to_ijk_mat)
            ct_profile.append(_sample(ct_arr_kji, kji))
            gm_profile.append(_sample(gm_arr_kji, kji))
        ct_profile = np.asarray(ct_profile, dtype=float)
        gm_profile = np.asarray(gm_profile, dtype=float)

        ct_peak_idx = int(np.nanargmax(ct_profile))
        gm_peak_idx = int(np.nanargmax(gm_profile))
        gm_peak_val = float(gm_profile[gm_peak_idx])
        gm_peak_off = float(offsets[gm_peak_idx])

        print(
            f"\n## {gt.shank}  span={gt.span_mm:.1f}mm  n={gt.contact_count}"
        )
        print(
            f"  CT max along extension: off={offsets[ct_peak_idx]:+.1f}mm "
            f"HU={ct_profile[ct_peak_idx]:.0f}"
        )
        print(
            f"  GM max along extension: off={gm_peak_off:+.1f}mm "
            f"val={gm_peak_val:.1f}"
        )

        print("  offset_mm | CT_HU | GM")
        for off, ct_v, gm_v in zip(offsets[::5], ct_profile[::5], gm_profile[::5]):
            print(f"    {off:+6.1f} | {ct_v:6.0f} | {gm_v:7.1f}")


if __name__ == "__main__":
    subj = sys.argv[1] if len(sys.argv) > 1 else "T22"
    run(subj)
