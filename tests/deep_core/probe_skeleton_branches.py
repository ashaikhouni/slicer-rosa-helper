"""Probe: find electrodes as long thin skeleton branches attached to skull.

Pipeline (no HU cleverness needed):
  1. Threshold CT at HU >= thr (default 1500) -> captures skull + bolts +
     electrode contacts as a single blob.
  2. Morphological close with radius 1 to bridge inter-contact gaps.
  3. Largest connected component.
  4. 3D binary thinning (skeletonization).
  5. Classify skeleton voxels by 26-neighbor count:
       1 -> endpoint
       2 -> line
       >=3 -> junction
  6. Extract arcs (connected components of line voxels). Each bolt+electrode
     appears as a long 1D arc bounded by a junction (skull) on one side and
     an endpoint (electrode tip) on the other.
  7. Filter arcs by mm length, report histogram, save label volume for
     Slicer inspection.

Usage:
  /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
    tests/deep_core/probe_skeleton_branches.py [T22|T2] [threshold_hu]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))
sys.path.insert(0, str(REPO_ROOT / "PostopCTLocalization"))

DATASET_ROOT = Path(
    os.environ.get(
        "ROSA_SEEG_DATASET",
        "/Users/ammar/Dropbox/thalamus_subjects/seeg_localization",
    )
)


def largest_connected_component(binary_img):
    import SimpleITK as sitk
    cc = sitk.ConnectedComponent(binary_img)
    relabel = sitk.RelabelComponent(cc, sortByObjectSize=True)
    return sitk.Equal(relabel, 1)


def neighbor_count_26(skel_arr):
    """26-neighbor count at each voxel; 0 outside skeleton."""
    padded = np.pad(skel_arr.astype(np.int8), 1)
    total = np.zeros_like(skel_arr, dtype=np.int16)
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                shifted = padded[
                    1 + dz : padded.shape[0] - 1 + dz,
                    1 + dy : padded.shape[1] - 1 + dy,
                    1 + dx : padded.shape[2] - 1 + dx,
                ]
                total += shifted
    return total * skel_arr.astype(np.int16)


def run(subject_id, threshold_hu):
    import SimpleITK as sitk

    ct_path = DATASET_ROOT / "post_registered_ct" / f"{subject_id}_post_registered.nii.gz"
    print(f"# subject={subject_id}  threshold_hu={threshold_hu}")
    print(f"# ct={ct_path}")

    img = sitk.ReadImage(str(ct_path))
    spacing = np.asarray(img.GetSpacing(), dtype=float)  # (x, y, z)
    mean_spacing = float(spacing.mean())
    print(f"# spacing xyz = {spacing.tolist()}  mean = {mean_spacing:.3f} mm")

    ct_arr = sitk.GetArrayFromImage(img)
    print(f"# CT shape={ct_arr.shape}  HU min={ct_arr.min()} max={ct_arr.max()}")

    # Step 1: threshold
    thr_img = sitk.BinaryThreshold(
        img, lowerThreshold=float(threshold_hu), upperThreshold=1e9,
        insideValue=1, outsideValue=0,
    )
    thr_arr = sitk.GetArrayFromImage(thr_img)
    print(f"# thresholded voxels (>= {threshold_hu}): {int(thr_arr.sum())}")

    # Step 2: binary close
    closed = sitk.BinaryMorphologicalClosing(thr_img, kernelRadius=(1, 1, 1))

    # Step 3: largest CC
    largest = largest_connected_component(closed)
    largest_arr = sitk.GetArrayFromImage(largest)
    print(f"# largest CC voxels: {int(largest_arr.sum())}")

    # Step 4: thinning (binary skeleton)
    skel_img = sitk.BinaryThinning(largest)
    skel_arr = sitk.GetArrayFromImage(skel_img).astype(np.uint8)
    print(f"# skeleton voxels: {int(skel_arr.sum())}")

    # Step 5: neighbor count classification
    nc = neighbor_count_26(skel_arr)
    endpoint = (nc == 1) & (skel_arr == 1)
    line = (nc == 2) & (skel_arr == 1)
    junction = (nc >= 3) & (skel_arr == 1)
    print(
        f"# endpoints={int(endpoint.sum())} "
        f"line_voxels={int(line.sum())} "
        f"junctions={int(junction.sum())}"
    )

    # Step 6: arcs = CCs of line voxels
    line_img = sitk.GetImageFromArray(line.astype(np.uint8))
    line_img.CopyInformation(img)
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)
    line_cc = cc_filter.Execute(line_img)
    line_cc = sitk.RelabelComponent(line_cc, sortByObjectSize=True)
    line_cc_arr = sitk.GetArrayFromImage(line_cc)

    n_arcs = int(line_cc_arr.max())
    print(f"# arcs (line CCs) = {n_arcs}")

    # Arc length in mm (voxel count * mean spacing -- coarse but fine for filter)
    # For each arc: count, touches_junction, touches_endpoint
    # Dilate arc labels by 1 to check adjacency.
    arc_lens = []
    arc_touches_j = []
    arc_touches_e = []
    # Pre-pad junction and endpoint masks for adjacency test.
    j_pad = np.pad(junction, 1)
    e_pad = np.pad(endpoint, 1)

    labels_interest = list(range(1, min(n_arcs, 400) + 1))
    for lab in labels_interest:
        mask = (line_cc_arr == lab)
        count = int(mask.sum())
        length_mm = count * mean_spacing
        arc_lens.append(length_mm)

        # Adjacency: dilate mask by 1, see if overlaps junction / endpoint
        mp = np.pad(mask, 1)
        dilated = np.zeros_like(mp)
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    dilated |= np.roll(np.roll(np.roll(mp, dz, 0), dy, 1), dx, 2)
        dilated = dilated[1:-1, 1:-1, 1:-1]
        touches_j = bool((dilated & junction).any())
        touches_e = bool((dilated & endpoint).any())
        arc_touches_j.append(touches_j)
        arc_touches_e.append(touches_e)

    arc_lens = np.asarray(arc_lens, dtype=float)

    # Histogram summary
    bins = [0, 5, 10, 15, 20, 30, 40, 60, 80, 120, 9999]
    hist, edges = np.histogram(arc_lens, bins=bins)
    print("\n# arc length histogram (mm)")
    for lo, hi, c in zip(edges[:-1], edges[1:], hist):
        print(f"  {lo:5.0f}-{hi:<5.0f}  n={c}")

    # Long arcs (candidate bolt+electrodes)
    print("\n# arcs >= 20 mm  [length_mm  touches_junction  touches_endpoint]")
    for lab, length, tj, te in sorted(
        zip(labels_interest, arc_lens, arc_touches_j, arc_touches_e),
        key=lambda x: -x[1],
    ):
        if length < 20:
            break
        print(f"  arc {lab:4d}  len={length:6.1f}  j={tj}  e={te}")

    # Save label volume: arcs >= 30mm painted, junctions painted with high label.
    out = np.zeros_like(line_cc_arr, dtype=np.uint16)
    long_count = 0
    for lab, length in zip(labels_interest, arc_lens):
        if length >= 30.0:
            long_count += 1
            out[line_cc_arr == lab] = long_count  # sequential label 1..N
    out[junction] = 1000
    out[endpoint] = 2000

    out_img = sitk.GetImageFromArray(out)
    out_img.CopyInformation(img)
    out_path = f"/tmp/skel_{subject_id}_thr{threshold_hu}.nii.gz"
    sitk.WriteImage(out_img, out_path)
    print(f"\n# long arcs (>=30mm) painted with labels 1..{long_count}")
    print(f"# junctions=label 1000, endpoints=label 2000")
    print(f"# wrote {out_path}")

    # Also save the largest-CC mask so we can eyeball input quality.
    mask_path = f"/tmp/mask_{subject_id}_thr{threshold_hu}.nii.gz"
    sitk.WriteImage(sitk.Cast(largest, sitk.sitkUInt8), mask_path)
    print(f"# wrote {mask_path}")


if __name__ == "__main__":
    subj = sys.argv[1] if len(sys.argv) > 1 else "T22"
    thr = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    run(subj, thr)
