"""Probe: does the gradient-magnitude volume peak at ROSA entry points?

Uses the high-quality T22 GT at
  contact_label_dataset/rosa_helper_import/T22/ROSA_Contacts_final_trajectory_points.csv
which has explicit (entry, target) RAS pairs for each trajectory.

For each trajectory:
  - Sample CT HU and GM along the entry->target direction from
    entry - 15 mm (outside the skull) to entry + 60 mm (deep into brain).
  - Report:
      * GM at entry point
      * GM peak in a +/- 5 mm window around entry
      * GM at target (deepest contact, inside brain)
      * whether the entry-region GM peak is distinguishable from brain
        background

Also dumps a small label mask marking entry points so we can visually
verify in Slicer.

Usage:
  /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
    tests/deep_core/probe_gm_entry_points.py
"""
from __future__ import annotations

import csv
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
SUBJECT = "T22"
CT_PATH = DATASET_ROOT / "post_registered_ct" / f"{SUBJECT}_post_registered.nii.gz"
GT_CSV = (
    DATASET_ROOT
    / "contact_label_dataset"
    / "rosa_helper_import"
    / SUBJECT
    / "ROSA_Contacts_final_trajectory_points.csv"
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


def load_trajectories(csv_path):
    traj = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["trajectory"]
            kind = row["point_type"]
            pt = np.array(
                [float(row["x_world_ras"]), float(row["y_world_ras"]), float(row["z_world_ras"])],
                dtype=float,
            )
            traj.setdefault(name, {})[kind] = pt
    return traj


def main():
    import SimpleITK as sitk
    from shank_core.io import image_ijk_ras_matrices

    print(f"# subject={SUBJECT}")
    print(f"# ct={CT_PATH}")
    print(f"# gt={GT_CSV}")

    img = sitk.ReadImage(str(CT_PATH))
    ct_arr = sitk.GetArrayFromImage(img).astype(np.float32)

    gm_filter = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
    gm_filter.SetSigma(1.0)
    gm_filter.SetNormalizeAcrossScale(False)
    gm_img = gm_filter.Execute(img)
    gm_arr = sitk.GetArrayFromImage(gm_img).astype(np.float32)

    ijk_to_ras_mat, ras_to_ijk_mat = image_ijk_ras_matrices(img)
    ras_to_ijk_mat = np.asarray(ras_to_ijk_mat, dtype=float)

    print(
        f"# CT HU min={ct_arr.min():.0f} max={ct_arr.max():.0f} "
        f"p999={np.percentile(ct_arr, 99.9):.0f}"
    )
    print(
        f"# GM mean={gm_arr.mean():.1f} p50={np.percentile(gm_arr, 50):.1f} "
        f"p95={np.percentile(gm_arr, 95):.1f} "
        f"p99={np.percentile(gm_arr, 99):.1f} max={gm_arr.max():.1f}"
    )

    traj = load_trajectories(GT_CSV)
    print(f"# trajectories = {len(traj)}")

    offsets = np.linspace(-15.0, 60.0, 151)  # 0.5 mm step

    # Table header
    print(
        "\n# per-trajectory: "
        "gm_entry = GM value AT entry point, "
        "gm_peak_window = max GM in +/- 5mm around entry, "
        "gm_target = GM at deepest contact (should be small)"
    )
    print(
        f"{'shank':8s} {'gm_entry':>9s} {'gm_peak':>9s} {'peak_off':>9s} "
        f"{'hu_entry':>9s} {'hu_peak':>9s} {'gm_target':>10s} {'hu_target':>10s}"
    )

    label_arr = np.zeros(ct_arr.shape, dtype=np.uint16)

    for tname in sorted(traj.keys()):
        pair = traj[tname]
        if "entry" not in pair or "target" not in pair:
            continue
        entry = pair["entry"]
        target = pair["target"]
        direction = _unit(target - entry)  # inward

        # sample profile
        ct_prof = []
        gm_prof = []
        for off in offsets:
            pt = entry + off * direction
            kji = _ras_to_kji(pt, ras_to_ijk_mat)
            ct_prof.append(_sample(ct_arr, kji))
            gm_prof.append(_sample(gm_arr, kji))
        ct_prof = np.asarray(ct_prof, dtype=float)
        gm_prof = np.asarray(gm_prof, dtype=float)

        # entry = offset 0. window +-5mm.
        entry_idx = int(np.argmin(np.abs(offsets - 0.0)))
        w_mask = np.abs(offsets) <= 5.0
        gm_entry = gm_prof[entry_idx]
        hu_entry = ct_prof[entry_idx]
        idx_in_w = np.argmax(gm_prof[w_mask])
        off_in_w = offsets[w_mask][idx_in_w]
        gm_peak_w = gm_prof[w_mask][idx_in_w]
        hu_peak_w = ct_prof[w_mask][idx_in_w]

        # Target: sample at target position directly.
        kji_t = _ras_to_kji(target, ras_to_ijk_mat)
        gm_target = _sample(gm_arr, kji_t)
        hu_target = _sample(ct_arr, kji_t)

        print(
            f"{tname:8s} {gm_entry:9.1f} {gm_peak_w:9.1f} {off_in_w:+9.1f} "
            f"{hu_entry:9.0f} {hu_peak_w:9.0f} {gm_target:10.1f} {hu_target:10.0f}"
        )

        # mark entry and target in the label mask
        for label_val, pt in ((1, entry), (2, target)):
            kji = _ras_to_kji(pt, ras_to_ijk_mat)
            ki, ji, ii = int(round(kji[0])), int(round(kji[1])), int(round(kji[2]))
            if (
                0 <= ki < ct_arr.shape[0]
                and 0 <= ji < ct_arr.shape[1]
                and 0 <= ii < ct_arr.shape[2]
            ):
                # 3mm radius sphere-ish (just nearest voxels)
                for dk in range(-2, 3):
                    for dj in range(-2, 3):
                        for di in range(-2, 3):
                            if dk * dk + dj * dj + di * di <= 4:
                                kk, jj, iii = ki + dk, ji + dj, ii + di
                                if (
                                    0 <= kk < ct_arr.shape[0]
                                    and 0 <= jj < ct_arr.shape[1]
                                    and 0 <= iii < ct_arr.shape[2]
                                ):
                                    label_arr[kk, jj, iii] = label_val

    # Save marked entry/target points
    out_label = f"/tmp/entry_points_{SUBJECT}.nii.gz"
    out_img = sitk.GetImageFromArray(label_arr)
    out_img.CopyInformation(img)
    sitk.WriteImage(out_img, out_label)
    print(f"\n# wrote {out_label} (label 1=entry, 2=target)")
    print(f"# GM volume at /tmp/gm_{SUBJECT}.nii.gz (from previous probe)")

    # Quick distribution probe: sample random points inside a central cube
    # to characterize brain-background GM.
    np.random.seed(0)
    cz, cy, cx = (s // 2 for s in ct_arr.shape)
    r = 30
    sample = gm_arr[cz - r : cz + r, cy - r : cy + r, cx - r : cx + r]
    print(
        f"\n# central cube (~brain) GM: "
        f"mean={sample.mean():.1f} p50={np.percentile(sample, 50):.1f} "
        f"p95={np.percentile(sample, 95):.1f} p99={np.percentile(sample, 99):.1f}"
    )


if __name__ == "__main__":
    main()
