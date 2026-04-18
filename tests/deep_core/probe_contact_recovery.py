"""Probe B: brain-only contact recovery with soft-tissue neighborhood filter.

Goal: measure whether Frangi sigma=1 + a soft-tissue neighborhood filter
(no RANSAC, no heavy masking) gives a usable voxel cloud that covers
GT shanks and has limited false-positive spread.

Method:
  1. Frangi sigma=1 on raw CT, threshold >= 10.
  2. Soft-tissue filter: keep voxels whose 5 mm-radius raw-CT neighborhood
     has median HU in [-50, 80].
  3. Run twice: with and without an intracranial mask (head_distance >= 10
     mm), so we can see if soft-tissue filter alone is enough.
  4. Per GT shank: voxels per mm along GT axis within 2 mm-radius tube,
     plus longest axis gap where no inliers are found.
  5. Globally: total kept voxels, voxels NOT inside any GT tube, FP fraction.

Usage:
  /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
    tests/deep_core/probe_contact_recovery.py [T22|T2]
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

FRANGI_S1_THR = 10.0
SOFT_HU_LO = -50.0
SOFT_HU_HI = 80.0
SOFT_RADIUS_MM = 5.0

INTRACRANIAL_MIN_DISTANCE_MM = 10.0

GT_TUBE_RADIUS_MM = 2.0
GT_BIN_STEP_MM = 1.0  # bins for density / gap analysis


def _unit(v):
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def build_intracranial(img):
    import SimpleITK as sitk
    thr = sitk.BinaryThreshold(img, lowerThreshold=-500.0, upperThreshold=1e9,
                                insideValue=1, outsideValue=0)
    cc = sitk.RelabelComponent(sitk.ConnectedComponent(thr), sortByObjectSize=True)
    largest = sitk.Equal(cc, 1)
    closed = sitk.BinaryMorphologicalClosing(largest, kernelRadius=(3, 3, 3))
    hull = sitk.BinaryFillhole(closed)
    dist = sitk.SignedMaurerDistanceMap(
        hull, insideIsPositive=True, squaredDistance=False, useImageSpacing=True,
    )
    dist_arr = sitk.GetArrayFromImage(dist).astype(np.float32)
    return dist_arr >= INTRACRANIAL_MIN_DISTANCE_MM


def frangi_sigma(img, sigma):
    import SimpleITK as sitk
    sm = sitk.SmoothingRecursiveGaussian(img, sigma=float(sigma))
    ob = sitk.ObjectnessMeasure(sm, objectDimension=1, brightObject=True)
    return sitk.GetArrayFromImage(ob).astype(np.float32)


def sphere_offsets(spacing_xyz, radius_mm):
    """Return list of (dk, dj, di) offsets within radius_mm in world space.
    Array is KJI; voxel spacing is (sx, sy, sz) with x<->i, y<->j, z<->k.
    """
    sx, sy, sz = float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])
    rk = int(np.ceil(radius_mm / sz))
    rj = int(np.ceil(radius_mm / sy))
    ri = int(np.ceil(radius_mm / sx))
    offs = []
    for dk in range(-rk, rk + 1):
        for dj in range(-rj, rj + 1):
            for di in range(-ri, ri + 1):
                d2 = (dk * sz) ** 2 + (dj * sy) ** 2 + (di * sx) ** 2
                if d2 <= radius_mm ** 2:
                    offs.append((dk, dj, di))
    return offs


def neighborhood_median(raw_ct, kk, jj, ii, offs, chunk=4000):
    """For each candidate voxel (kk[n], jj[n], ii[n]), compute median raw-CT
    value over the sphere neighborhood.
    """
    K, J, I = raw_ct.shape
    dk = np.array([o[0] for o in offs], dtype=np.int32)
    dj = np.array([o[1] for o in offs], dtype=np.int32)
    di = np.array([o[2] for o in offs], dtype=np.int32)
    N = kk.shape[0]
    out = np.empty(N, dtype=np.float32)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        kc = kk[start:end][None, :] + dk[:, None]     # (M, chunk)
        jc = jj[start:end][None, :] + dj[:, None]
        ic = ii[start:end][None, :] + di[:, None]
        valid = (
            (kc >= 0) & (kc < K) & (jc >= 0) & (jc < J) & (ic >= 0) & (ic < I)
        )
        # Clamp to avoid index error; we will mask by 'valid' for median.
        kcC = np.clip(kc, 0, K - 1)
        jcC = np.clip(jc, 0, J - 1)
        icC = np.clip(ic, 0, I - 1)
        vals = raw_ct[kcC, jcC, icC].astype(np.float32)
        # Mark invalid as NaN so nanmedian ignores them.
        vals = np.where(valid, vals, np.nan)
        out[start:end] = np.nanmedian(vals, axis=0)
    return out


def load_gt(subject_id):
    csv_path = (
        DATASET_ROOT / "contact_label_dataset" / "rosa_helper_import"
        / subject_id / "ROSA_Contacts_final_trajectory_points.csv"
    )
    if csv_path.exists():
        traj = {}
        with open(csv_path) as f:
            r = csv.DictReader(f)
            for row in r:
                name = row["trajectory"]
                pt = np.array(
                    [float(row["x_world_ras"]), float(row["y_world_ras"]),
                     float(row["z_world_ras"])], dtype=float,
                )
                traj.setdefault(name, {})[row["point_type"]] = pt
        out = []
        for name, pair in traj.items():
            if "entry" in pair and "target" in pair:
                L = float(np.linalg.norm(pair["target"] - pair["entry"]))
                out.append((name, pair["entry"], pair["target"], L))
        return out
    from eval_seeg_localization import iter_subject_rows, load_ground_truth_shanks
    rows = iter_subject_rows(DATASET_ROOT, {subject_id})
    if not rows:
        return []
    row = rows[0]
    shanks = load_ground_truth_shanks(row["labels_path"], row.get("shanks_path"))
    return [
        (s.shank, np.asarray(s.start_ras, dtype=float),
         np.asarray(s.end_ras, dtype=float), float(s.span_mm))
        for s in shanks
    ]


def per_gt_coverage(points_ras, gt, tube_r_mm, bin_mm):
    """For each GT shank, report inlier density/mm and longest axial gap
    (mm) along the GT axis within a tube of radius tube_r_mm.

    Returns list of dicts with (name, L, n_tube, density_per_mm,
    longest_gap_mm, tube_mask) where tube_mask[i] is True if points_ras[i]
    lies in this shank's tube.
    """
    if points_ras.shape[0] == 0:
        return [dict(name=name, L=L, n_tube=0, density=0.0,
                     longest_gap=L, tube_mask=np.zeros(0, dtype=bool))
                for (name, _, _, L) in gt]
    results = []
    for (name, g_s, g_e, L) in gt:
        axis, gL = _line_axis_len(g_s, g_e)
        diffs = points_ras - g_s[None, :]
        proj = diffs @ axis
        perp = diffs - np.outer(proj, axis)
        dist = np.linalg.norm(perp, axis=1)
        in_axis = (proj >= 0.0) & (proj <= gL)
        in_tube = in_axis & (dist <= tube_r_mm)
        n_tube = int(in_tube.sum())
        density = n_tube / max(gL, 1e-6)
        # Axial-gap analysis
        if n_tube == 0:
            longest = float(gL)
        else:
            in_proj = proj[in_tube]
            n_bins = int(np.ceil(gL / bin_mm))
            occupied = np.zeros(n_bins, dtype=bool)
            bin_idx = np.clip((in_proj / bin_mm).astype(int), 0, n_bins - 1)
            occupied[bin_idx] = True
            # longest consecutive run of empty bins
            longest_empty = 0
            cur = 0
            for val in occupied:
                if not val:
                    cur += 1
                    if cur > longest_empty:
                        longest_empty = cur
                else:
                    cur = 0
            longest = float(longest_empty * bin_mm)
        results.append(dict(name=name, L=L, n_tube=n_tube,
                            density=density, longest_gap=longest,
                            tube_mask=in_tube))
    return results


def _line_axis_len(start, end):
    d = end - start
    n = float(np.linalg.norm(d))
    return (d / n) if n > 1e-9 else np.array([0.0, 0.0, 1.0]), n


def run(subject_id):
    import SimpleITK as sitk
    from shank_core.io import image_ijk_ras_matrices

    ct_path = DATASET_ROOT / "post_registered_ct" / f"{subject_id}_post_registered.nii.gz"
    print(f"# subject={subject_id}  ct={ct_path.name}")
    img = sitk.ReadImage(str(ct_path))
    spacing = img.GetSpacing()
    raw_ct = sitk.GetArrayFromImage(img).astype(np.float32)
    ijk_to_ras_mat, _ = image_ijk_ras_matrices(img)
    ijk_to_ras_mat = np.asarray(ijk_to_ras_mat, dtype=float)

    t0 = time.time()
    intracranial = build_intracranial(img)
    print(f"# intracranial mask: {int(intracranial.sum())} voxels "
          f"({time.time()-t0:.1f}s)")

    t0 = time.time()
    frangi_s1 = frangi_sigma(img, sigma=1.0)
    print(f"# frangi sigma=1: {time.time()-t0:.1f}s "
          f"max={frangi_s1.max():.1f}")

    # Stage 1: Frangi cloud
    frangi_mask = frangi_s1 >= FRANGI_S1_THR
    kk0, jj0, ii0 = np.where(frangi_mask)
    n0 = int(kk0.size)
    print(f"# frangi>={FRANGI_S1_THR}: {n0} voxels")

    # Stage 2: soft-tissue neighborhood filter
    t0 = time.time()
    offs = sphere_offsets(spacing, SOFT_RADIUS_MM)
    print(f"# soft-tissue sphere: {len(offs)} voxels (radius {SOFT_RADIUS_MM} mm)")
    med = neighborhood_median(raw_ct, kk0, jj0, ii0, offs)
    soft_ok = (med >= SOFT_HU_LO) & (med <= SOFT_HU_HI)
    print(f"# soft-tissue pass ({SOFT_HU_LO}..{SOFT_HU_HI} HU): "
          f"{int(soft_ok.sum())}/{n0} ({time.time()-t0:.1f}s)")

    # Convert candidate voxels to RAS
    def kji_to_ras(kji_arr):
        ijk = np.stack([kji_arr[:, 2], kji_arr[:, 1], kji_arr[:, 0]], axis=1)
        h = np.concatenate([ijk, np.ones((ijk.shape[0], 1))], axis=1)
        return (ijk_to_ras_mat @ h.T).T[:, :3]

    gt = load_gt(subject_id)
    print(f"# GT trajectories: {len(gt)}")

    configs = [("without_mask", soft_ok), ("with_mask", soft_ok & intracranial[kk0, jj0, ii0])]

    for label, keep in configs:
        kk = kk0[keep]; jj = jj0[keep]; ii = ii0[keep]
        n_keep = int(kk.size)
        pts_kji = np.stack([kk, jj, ii], axis=1).astype(np.float64)
        pts_ras = kji_to_ras(pts_kji) if n_keep > 0 else np.zeros((0, 3))

        print(f"\n## config={label}  kept voxels: {n_keep}")
        cov = per_gt_coverage(pts_ras, gt, GT_TUBE_RADIUS_MM, GT_BIN_STEP_MM)
        # Aggregate
        total_in_tube = 0
        any_tube = np.zeros(n_keep, dtype=bool)
        for c in cov:
            any_tube = any_tube | c["tube_mask"]
            total_in_tube += c["n_tube"]  # this may double-count shared voxels
        n_any_tube = int(any_tube.sum())
        n_fp = n_keep - n_any_tube
        fp_frac = (n_fp / n_keep) if n_keep else 0.0
        # Worst-shank stats
        if cov:
            min_density = min(c["density"] for c in cov)
            median_density = float(np.median([c["density"] for c in cov]))
            worst_gap = max(c["longest_gap"] for c in cov)
        else:
            min_density = median_density = worst_gap = 0.0
        print(
            f"# totals: kept={n_keep}  in_any_tube={n_any_tube}  "
            f"fp={n_fp} ({fp_frac*100:.1f}%)"
        )
        print(
            f"# GT coverage: median density={median_density:.2f} vox/mm  "
            f"worst shank density={min_density:.2f} vox/mm  "
            f"worst longest-gap={worst_gap:.1f} mm"
        )
        print(f"{'shank':10s} {'L_mm':>6s} {'n_tube':>7s} {'vox/mm':>7s} {'gap_mm':>7s}")
        for c in cov:
            print(
                f"{c['name']:10s} {c['L']:6.1f} {c['n_tube']:>7d} "
                f"{c['density']:7.2f} {c['longest_gap']:7.1f}"
            )

        # Save NIFTI cloud for Slicer
        out = np.zeros_like(frangi_s1, dtype=np.uint8)
        if n_keep > 0:
            out[kk, jj, ii] = 1
        out_img = sitk.GetImageFromArray(out)
        out_img.CopyInformation(img)
        suffix = label.replace("_", "")  # withoutmask / withmask
        out_path = f"/tmp/contact_cloud_{subject_id}_{suffix}.nii.gz"
        sitk.WriteImage(out_img, out_path)
        print(f"# wrote {out_path}")


if __name__ == "__main__":
    subj = sys.argv[1] if len(sys.argv) > 1 else "T22"
    run(subj)
