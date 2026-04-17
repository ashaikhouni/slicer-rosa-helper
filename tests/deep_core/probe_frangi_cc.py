"""Probe: Frangi tube response (sigma=2,3) -> threshold -> CC -> linearity
filter -> label map of electrode candidates.

Usage:
  /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
    tests/deep_core/probe_frangi_cc.py [T22|T2] [frangi_threshold]
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


def head_mask(img):
    import SimpleITK as sitk
    thr = sitk.BinaryThreshold(img, lowerThreshold=-500.0, upperThreshold=1e9,
                                insideValue=1, outsideValue=0)
    cc = sitk.RelabelComponent(sitk.ConnectedComponent(thr), sortByObjectSize=True)
    largest = sitk.Equal(cc, 1)
    closed = sitk.BinaryMorphologicalClosing(largest, kernelRadius=(3, 3, 3))
    return sitk.BinaryFillhole(closed)


def tube_response(img, sigmas):
    import SimpleITK as sitk
    max_arr = None
    for s in sigmas:
        smoothed = sitk.SmoothingRecursiveGaussian(img, sigma=s)
        obj = sitk.ObjectnessMeasure(
            smoothed, alpha=0.5, beta=0.5, gamma=5.0,
            scaleObjectnessMeasure=True,
            objectDimension=1, brightObject=True,
        )
        a = sitk.GetArrayFromImage(obj)
        max_arr = a if max_arr is None else np.maximum(max_arr, a)
    out = sitk.GetImageFromArray(max_arr)
    out.CopyInformation(img)
    return out, max_arr


def analyze_ccs(labeled_arr, spacing_xyz):
    """For each CC: voxel count, span_mm along PC1, linearity (lam2/lam1).

    spacing_xyz is (x, y, z). Our arrays are KJI which maps to ZYX, so when
    converting voxel indices to mm we use (z=spacing[2], y=spacing[1], x=spacing[0]).
    """
    n = int(labeled_arr.max())
    results = []  # list of (label, n_vox, span_mm, ratio, center_ras_ijk)
    sx, sy, sz = float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])
    for lab in range(1, n + 1):
        kk, jj, ii = np.where(labeled_arr == lab)
        if kk.size < 4:
            continue
        pts = np.stack([ii * sx, jj * sy, kk * sz], axis=1).astype(np.float64)
        c = pts.mean(axis=0)
        X = pts - c
        cov = X.T @ X / max(1, pts.shape[0] - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]
        l1, l2, l3 = float(eigvals[0]), float(eigvals[1]), float(eigvals[2])
        # project onto PC1 to get span
        pc1 = eigvecs[:, 0]
        proj = X @ pc1
        span_mm = float(proj.max() - proj.min())
        ratio = l2 / l1 if l1 > 1e-12 else 1.0
        results.append(
            dict(
                label=lab,
                n_vox=int(pts.shape[0]),
                span_mm=span_mm,
                ratio=ratio,
                l1=l1, l2=l2, l3=l3,
                center_xyz=c,
                axis=pc1,
            )
        )
    return results


def run(subject_id, frangi_thr):
    import SimpleITK as sitk

    ct_path = DATASET_ROOT / "post_registered_ct" / f"{subject_id}_post_registered.nii.gz"
    print(f"# subject={subject_id}  threshold={frangi_thr}")
    print(f"# ct={ct_path}")

    img = sitk.ReadImage(str(ct_path))
    spacing = img.GetSpacing()

    t0 = time.time()
    hm = head_mask(img)
    hm_arr = sitk.GetArrayFromImage(hm).astype(bool)
    print(f"# head mask: {int(hm_arr.sum())} voxels ({time.time()-t0:.1f}s)")

    t0 = time.time()
    tube_img, tube_arr = tube_response(img, sigmas=(2.0, 3.0))
    print(
        f"# frangi sigma=[2,3]: {time.time()-t0:.1f}s "
        f"max={tube_arr.max():.2f} p99={np.percentile(tube_arr, 99):.2f} "
        f"p999={np.percentile(tube_arr, 99.9):.2f}"
    )

    # Apply head mask and threshold
    bin_arr = (tube_arr >= float(frangi_thr)) & hm_arr
    print(f"# voxels above threshold {frangi_thr} inside head mask: {int(bin_arr.sum())}")

    # CC on threshold mask
    bin_img = sitk.GetImageFromArray(bin_arr.astype(np.uint8))
    bin_img.CopyInformation(img)
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)
    cc_img = cc_filter.Execute(bin_img)
    cc_img = sitk.RelabelComponent(cc_img, sortByObjectSize=True)
    cc_arr = sitk.GetArrayFromImage(cc_img)
    n_cc = int(cc_arr.max())
    print(f"# connected components: {n_cc}")

    # Analyze CCs
    infos = analyze_ccs(cc_arr, spacing)
    infos.sort(key=lambda d: -d["span_mm"])

    print("\n# top 30 CCs by span (span_mm, ratio=lam2/lam1, n_vox)")
    print(f"{'label':>6s} {'n_vox':>7s} {'span_mm':>8s} {'ratio':>7s}  kept?")
    kept_labels = []
    # Filter: electrode-like = span >= 15mm AND ratio <= 0.1 (linear)
    SPAN_MIN = 15.0
    RATIO_MAX = 0.15
    for d in infos[:30]:
        keep = (d["span_mm"] >= SPAN_MIN) and (d["ratio"] <= RATIO_MAX)
        mark = "YES" if keep else "no"
        print(
            f"{d['label']:>6d} {d['n_vox']:>7d} "
            f"{d['span_mm']:>8.1f} {d['ratio']:>7.3f}  {mark}"
        )
        if keep:
            kept_labels.append(d["label"])

    # Also collect any kept ones beyond the printed 30
    for d in infos[30:]:
        if (d["span_mm"] >= SPAN_MIN) and (d["ratio"] <= RATIO_MAX):
            kept_labels.append(d["label"])

    print(f"\n# kept (electrode candidates): {len(kept_labels)}")

    # Paint kept CCs with sequential labels in a new volume
    out = np.zeros_like(cc_arr, dtype=np.uint16)
    for i, lab in enumerate(kept_labels, start=1):
        out[cc_arr == lab] = i
    out_img = sitk.GetImageFromArray(out)
    out_img.CopyInformation(img)
    out_path = f"/tmp/electrodes_{subject_id}_t{int(frangi_thr)}.nii.gz"
    sitk.WriteImage(out_img, out_path)
    print(f"# wrote {out_path}")

    # Also save full CC label map for comparison
    cc_path = f"/tmp/all_ccs_{subject_id}_t{int(frangi_thr)}.nii.gz"
    sitk.WriteImage(sitk.Cast(cc_img, sitk.sitkUInt16), cc_path)
    print(f"# wrote {cc_path}")

    # Save the frangi[sigma=2,3] volume too
    frangi_path = f"/tmp/frangi23_{subject_id}.nii.gz"
    sitk.WriteImage(tube_img, frangi_path)
    print(f"# wrote {frangi_path}")


if __name__ == "__main__":
    subj = sys.argv[1] if len(sys.argv) > 1 else "T22"
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    run(subj, thr)
