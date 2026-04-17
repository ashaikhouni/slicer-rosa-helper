"""Probe: multi-scale Frangi / Sato tube-enhancement on raw CT.

SimpleITK 2.5 only exposes `ObjectnessMeasure` (single-scale). We control
scale by pre-smoothing with a recursive Gaussian. Multi-scale is the max
across pre-smoothed volumes.

For each subject:
  1. Build a head mask (HU >= -500, largest CC, close+fill).
  2. For each sigma in [1.0, 2.0, 3.0]:
       smoothed = GaussianSmooth(CT, sigma)
       obj = ObjectnessMeasure(smoothed, objectDimension=1 (tube),
                               brightObject=True)
     Take max across sigmas -> tube response volume.
  3. Save raw and head-masked tube response.
  4. For T22: report tube response AT each ROSA entry point (where the
     bolt pierces the skull) and AT each target (deepest contact).

Usage:
  /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
    tests/deep_core/probe_frangi.py [T22|T2]
"""
from __future__ import annotations

import csv
import os
import sys
import time
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


def head_mask(img):
    """HU >= -500 -> largest CC -> close -> fill holes."""
    import SimpleITK as sitk
    thr = sitk.BinaryThreshold(img, lowerThreshold=-500.0, upperThreshold=1e9,
                                insideValue=1, outsideValue=0)
    cc = sitk.RelabelComponent(sitk.ConnectedComponent(thr), sortByObjectSize=True)
    largest = sitk.Equal(cc, 1)
    closed = sitk.BinaryMorphologicalClosing(largest, kernelRadius=(3, 3, 3))
    filled = sitk.BinaryFillhole(closed)
    return filled


def tube_response_multiscale(img, sigmas=(1.0, 2.0, 3.0)):
    import SimpleITK as sitk
    max_arr = None
    for s in sigmas:
        t0 = time.time()
        smoothed = sitk.SmoothingRecursiveGaussian(img, sigma=s)
        t1 = time.time()
        obj = sitk.ObjectnessMeasure(
            smoothed,
            alpha=0.5, beta=0.5, gamma=5.0,
            scaleObjectnessMeasure=True,
            objectDimension=1,
            brightObject=True,
        )
        t2 = time.time()
        a = sitk.GetArrayFromImage(obj)
        if max_arr is None:
            max_arr = a
        else:
            max_arr = np.maximum(max_arr, a)
        print(
            f"  sigma={s}: smooth={t1-t0:.1f}s obj={t2-t1:.1f}s "
            f"max={a.max():.3f} p99={np.percentile(a, 99):.3f}"
        )
    out = sitk.GetImageFromArray(max_arr)
    out.CopyInformation(img)
    return out


def load_rosa_entries(subject_id):
    csv_path = (
        DATASET_ROOT / "contact_label_dataset" / "rosa_helper_import"
        / subject_id / "ROSA_Contacts_final_trajectory_points.csv"
    )
    if not csv_path.exists():
        return None
    traj = {}
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            name = row["trajectory"]
            pt = np.array(
                [float(row["x_world_ras"]), float(row["y_world_ras"]), float(row["z_world_ras"])],
                dtype=float,
            )
            traj.setdefault(name, {})[row["point_type"]] = pt
    return traj


def run(subject_id):
    import SimpleITK as sitk
    from shank_core.io import image_ijk_ras_matrices

    ct_path = DATASET_ROOT / "post_registered_ct" / f"{subject_id}_post_registered.nii.gz"
    print(f"# subject={subject_id}  ct={ct_path}")

    img = sitk.ReadImage(str(ct_path))
    arr = sitk.GetArrayFromImage(img)
    print(f"# shape={arr.shape}  HU min={arr.min():.0f} max={arr.max():.0f}")

    t0 = time.time()
    print("# computing head mask")
    hm = head_mask(img)
    hm_arr = sitk.GetArrayFromImage(hm).astype(bool)
    print(f"  head mask voxels: {int(hm_arr.sum())}  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    print("# computing multi-scale tube response")
    tube = tube_response_multiscale(img, sigmas=(1.0, 2.0, 3.0))
    tube_arr = sitk.GetArrayFromImage(tube)
    print(f"  total tube-response time: {time.time()-t0:.1f}s")
    print(
        f"  tube raw: max={tube_arr.max():.3f} "
        f"p95={np.percentile(tube_arr, 95):.3f} "
        f"p99={np.percentile(tube_arr, 99):.3f} "
        f"p999={np.percentile(tube_arr, 99.9):.3f}"
    )

    tube_masked = tube_arr.copy()
    tube_masked[~hm_arr] = 0.0
    print(
        f"  tube masked: max={tube_masked.max():.3f} "
        f"p95={np.percentile(tube_masked, 95):.3f} "
        f"p99={np.percentile(tube_masked, 99):.3f} "
        f"p999={np.percentile(tube_masked, 99.9):.3f}"
    )

    out_raw = f"/tmp/frangi_{subject_id}_raw.nii.gz"
    out_masked = f"/tmp/frangi_{subject_id}_masked.nii.gz"
    out_headmask = f"/tmp/headmask_{subject_id}.nii.gz"
    sitk.WriteImage(tube, out_raw)
    masked_img = sitk.GetImageFromArray(tube_masked)
    masked_img.CopyInformation(img)
    sitk.WriteImage(masked_img, out_masked)
    sitk.WriteImage(sitk.Cast(hm, sitk.sitkUInt8), out_headmask)
    print(f"# wrote {out_raw}")
    print(f"# wrote {out_masked}")
    print(f"# wrote {out_headmask}")

    # Sample at ROSA entry and target points if available
    traj = load_rosa_entries(subject_id)
    if traj:
        _, ras_to_ijk_mat = image_ijk_ras_matrices(img)
        ras_to_ijk_mat = np.asarray(ras_to_ijk_mat, dtype=float)

        print(f"\n# tube response at ROSA GT points (T22)")
        print(f"{'traj':8s} {'frangi_entry':>14s} {'frangi_target':>14s}")
        for name in sorted(traj.keys()):
            pair = traj[name]
            if "entry" not in pair or "target" not in pair:
                continue
            kji_e = _ras_to_kji(pair["entry"], ras_to_ijk_mat)
            kji_t = _ras_to_kji(pair["target"], ras_to_ijk_mat)
            fe = _sample(tube_arr, kji_e)
            ft = _sample(tube_arr, kji_t)
            print(f"{name:8s} {fe:14.3f} {ft:14.3f}")


if __name__ == "__main__":
    subj = sys.argv[1] if len(sys.argv) > 1 else "T22"
    run(subj)
