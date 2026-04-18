"""Probe H: two-stage detector = blob-pitch (primary) + Frangi-tube fallback.

Stage 1: blob-pitch v2 catches shanks with resolvable contacts at Dixi
pitch. High precision, fast.

Stage 2: for shanks like T2 RSAN that appear as continuous dark tubes
without contact resolution, run Frangi sigma=1 RANSAC (v4-style) on the
voxel cloud MINUS a 3 mm exclusion zone around stage-1 lines. This keeps
the cloud small, avoids re-fitting already-detected shanks, and catches
pitch-less shanks.

Usage:
  /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
    tests/deep_core/probe_two_stage.py [T22|T2]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))
sys.path.insert(0, str(REPO_ROOT / "PostopCTLocalization"))
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "deep_core"))

import probe_blob_pitch as bp  # noqa: E402
from probe_detector_v4 import (  # noqa: E402
    DATASET_ROOT, FRANGI_S1_THR, RANSAC_MIN_INLIERS, RANSAC_MIN_SPAN_MM,
    RANSAC_MAX_SPAN_MM, RANSAC_TOL_MM, RANSAC_ITER, RANSAC_EXCLUSION_MM,
    PITCH_MIN_MM, PITCH_MAX_MM, PITCH_MIN_SNR, PITCH_MIN_TRACK_LEN_MM,
    LONG_TRACK_BYPASS_MM,
    build_masks, frangi_single, ransac_one_line, dedup_tracks,
    periodicity_along_polyline,
    load_gt, match_tracks_to_gt, _unit,
)

FALLBACK_EXCLUSION_MM = 3.0
# Stage-2 shank-reach prior: real shanks enter through the skull so their
# shallowest CC voxel sits near the intracranial boundary (dist ~10 mm from
# hull surface) AND their deep tip reaches well into the brain. Sulcal /
# vessel tubular FPs sit right on the cortex and never get deep. Require:
#   dist_min ≤ HULL_ENDPOINT_MAX_MM (one end near hull)
#   dist_max ≥ DEEP_TIP_MIN_MM     (other end deep in brain)
HULL_ENDPOINT_MAX_MM = 15.0
DEEP_TIP_MIN_MM = 25.0


def hull_signed_distance(img, hull_arr):
    """Signed distance in mm from the hull surface (inside positive).
    Matches the head-distance field used by build_masks internally."""
    import SimpleITK as sitk
    hull_sitk = sitk.GetImageFromArray(hull_arr.astype(np.uint8))
    hull_sitk.CopyInformation(img)
    dist = sitk.SignedMaurerDistanceMap(
        hull_sitk, insideIsPositive=True, squaredDistance=False,
        useImageSpacing=True,
    )
    return sitk.GetArrayFromImage(dist).astype(np.float32)


def run_stage1(img, log1, dist_arr=None, ras_to_ijk_mat=None,
               deep_tip_min_mm=DEEP_TIP_MIN_MM):
    """Run blob-pitch v2 on LoG sigma=1. Returns (lines, blobs_kji).

    If dist_arr + ras_to_ijk_mat are provided, apply the deep-tip prior:
    drop lines whose deepest inlier blob has head_distance < deep_tip_min_mm.
    Real shanks' distal contacts sit deep in brain; skull-/bone-assembled
    spurious lines have all inliers at head_distance ≤ 0.
    """
    blobs = bp.extract_blobs(log1, img, threshold=300.0)
    to_ras = bp.kji_to_ras_fn(img)
    pts_ras = np.array([to_ras(b["kji"]) for b in blobs])
    amps = np.array([b["amp"] for b in blobs], dtype=float)
    n_vox = np.array([b["n_vox"] for b in blobs], dtype=int)
    contact_mask = n_vox <= 500
    pts_c = pts_ras[contact_mask]
    amps_c = amps[contact_mask]

    # Pitch-compatible pairs
    dist = np.sqrt(np.sum((pts_c[:, None, :] - pts_c[None, :, :]) ** 2, axis=2))
    pair_mask = np.zeros_like(dist, dtype=bool)
    for mult in (1, 2, 3):
        lo = mult * bp.PITCH_MM - bp.PITCH_TOL_MM
        hi = mult * bp.PITCH_MM + bp.PITCH_TOL_MM
        pair_mask |= (dist >= lo) & (dist <= hi)
    iu, ju = np.where(np.triu(pair_mask, k=1))

    hyps = []
    for pi, pj in zip(iu, ju):
        h = bp.walk_line(int(pi), int(pj), pts_c, amps_c)
        if h is not None:
            hyps.append(h)
    hyps.sort(key=lambda h: -h["n_blobs"])
    lines = bp.dedup_lines(hyps)
    lines = [l for l in lines if l.get("amp_sum", 0.0) >= 6000.0]

    if dist_arr is not None and ras_to_ijk_mat is not None:
        K, J, I = dist_arr.shape
        kept = []
        for l in lines:
            inlier_ras = pts_c[l["inlier_idx"]]
            h = np.concatenate([inlier_ras, np.ones((inlier_ras.shape[0], 1))],
                               axis=1)
            ijk = (ras_to_ijk_mat @ h.T).T[:, :3]
            ii = np.clip(np.round(ijk[:, 0]).astype(int), 0, I - 1)
            jj = np.clip(np.round(ijk[:, 1]).astype(int), 0, J - 1)
            kk = np.clip(np.round(ijk[:, 2]).astype(int), 0, K - 1)
            inlier_dists = dist_arr[kk, jj, ii]
            d_max = float(inlier_dists.max())
            d_min = float(inlier_dists.min())
            l["dist_min_mm"] = d_min
            l["dist_max_mm"] = d_max
            if d_max < deep_tip_min_mm:
                continue
            kept.append(l)
        lines = kept
    return lines, pts_c


def compute_exclusion_mask(cloud_mask_shape, lines_ras, ijk_to_ras_mat,
                           ras_to_ijk_mat, radius_mm=FALLBACK_EXCLUSION_MM,
                           step_mm=1.0):
    """Build a boolean mask marking voxels within `radius_mm` of any
    stage-1 line. Used to exclude those voxels from the fallback cloud.
    """
    excl = np.zeros(cloud_mask_shape, dtype=bool)
    K, J, I = cloud_mask_shape
    for line in lines_ras:
        start = line["start_ras"]
        end = line["end_ras"]
        axis = end - start
        L = float(np.linalg.norm(axis))
        if L < 1e-3:
            continue
        axis = axis / L
        n_steps = int(L / step_mm) + 1
        for k_step in range(n_steps):
            p_ras = start + k_step * step_mm * axis
            # Stamp a sphere of `radius_mm` in voxel space around p_ras
            h = np.array([p_ras[0], p_ras[1], p_ras[2], 1.0])
            ijk = (ras_to_ijk_mat @ h)[:3]
            k, j, i = int(round(ijk[2])), int(round(ijk[1])), int(round(ijk[0]))
            r = int(np.ceil(radius_mm / 0.5))  # assume ~0.5 mm spacing
            k0, k1 = max(0, k - r), min(K, k + r + 1)
            j0, j1 = max(0, j - r), min(J, j + r + 1)
            i0, i1 = max(0, i - r), min(I, i + r + 1)
            excl[k0:k1, j0:j1, i0:i1] = True
    return excl


def run_stage2(frangi_s1, intracranial_mask, exclusion_mask, spacing_xyz,
               dist_arr=None, hull_endpoint_max_mm=HULL_ENDPOINT_MAX_MM,
               deep_tip_min_mm=DEEP_TIP_MIN_MM, frangi_thr=30.0):
    """Fast shaft detector: CC + PCA on residual Frangi cloud. Instead of
    RANSAC, we rely on the fact that a continuous dark shaft (e.g. RSAN)
    with contacts invisible on CT shows up as a single elongated CC in the
    Frangi tube response; PCA gives the axis directly. Uses a higher
    Frangi threshold than stage 1 (30 vs 10) to get thinner CCs and avoid
    merging with adjacent bone structures.
    """
    import SimpleITK as sitk
    cloud_mask = (frangi_s1 >= frangi_thr) & intracranial_mask & ~exclusion_mask
    bin_img = sitk.GetImageFromArray(cloud_mask.astype(np.uint8))
    cc = sitk.ConnectedComponentImageFilter()
    cc.SetFullyConnected(True)
    cc_img = cc.Execute(bin_img)
    cc_arr = sitk.GetArrayFromImage(cc_img)
    n_cc = int(cc_arr.max())
    print(f"  stage-2 residual: {int(cloud_mask.sum())} voxels, {n_cc} CCs")

    sx, sy, sz = float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])
    lines = []
    # Vectorized CC iteration via np.unique / argsort
    flat = cc_arr.ravel()
    order = np.argsort(flat)
    flat_sorted = flat[order]
    changes = np.where(np.diff(flat_sorted) != 0)[0] + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [flat_sorted.size]])
    for s, e in zip(starts, ends):
        lab = flat_sorted[s]
        if lab == 0:
            continue
        idxs = order[s:e]
        n_vox = idxs.size
        # Shank CCs: big enough to be a tube, generous upper bound so we
        # don't lose elongated shanks that include a bolt or bone bit.
        if n_vox < 30 or n_vox > 20000:
            continue
        kk, jj, ii = np.unravel_index(idxs, cc_arr.shape)
        # PCA in world-mm
        pts_mm = np.stack([ii * sx, jj * sy, kk * sz], axis=1).astype(np.float64)
        c = pts_mm.mean(axis=0)
        X = pts_mm - c
        cov = X.T @ X / max(1, n_vox - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]
        l1, l2 = float(eigvals[0]), float(eigvals[1])
        pc1 = eigvecs[:, 0]
        proj = X @ pc1
        lo, hi = float(proj.min()), float(proj.max())
        span = hi - lo
        if span < 20.0 or span > 85.0:
            continue
        # Tubularity via span / perp-RMS -- handles curved tubes where the
        # eigenvalue aspect ratio collapses.
        perp = X - np.outer(proj, pc1)
        perp_rms = float(np.sqrt(np.mean(np.sum(perp * perp, axis=1))))
        if perp_rms > 3.0:
            continue
        aspect_geom = span / max(perp_rms, 1e-3)
        if aspect_geom < 5.0:
            continue
        if dist_arr is not None:
            cc_dist_min = float(dist_arr[kk, jj, ii].min())
            cc_dist_max = float(dist_arr[kk, jj, ii].max())
            if cc_dist_min > hull_endpoint_max_mm:
                continue
            if cc_dist_max < deep_tip_min_mm:
                continue
        else:
            cc_dist_min = float("nan")
            cc_dist_max = float("nan")
        aspect = l1 / max(l2, 1e-6)
        # Build polyline in kji space
        kji_pts = np.stack([kk, jj, ii], axis=1).astype(np.float64)
        kji_c = kji_pts.mean(axis=0)
        X_kji = kji_pts - kji_c
        _, _, Vt = np.linalg.svd(X_kji, full_matrices=False)
        kji_axis = _unit(Vt[0])
        kji_proj = X_kji @ kji_axis
        kji_lo, kji_hi = float(kji_proj.min()), float(kji_proj.max())
        step = 0.5
        n_steps = int(span / step) + 1
        t = np.linspace(kji_lo, kji_hi, n_steps)
        polyline = kji_c + t[:, None] * kji_axis
        lines.append({
            "polyline_kji": polyline,
            "length_mm": span,
            "n_inliers": n_vox,
            "axis": kji_axis, "center": kji_c,
            "aspect": aspect,
            "dist_min_mm": cc_dist_min,
            "dist_max_mm": cc_dist_max,
        })
    return lines, cc_arr


def describe_ccs_near_axis(cc_arr, gt_entry_ras, gt_target_ras,
                            ijk_to_ras_mat, spacing_xyz, name):
    sx, sy, sz = float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])
    axis = gt_target_ras - gt_entry_ras
    L = float(np.linalg.norm(axis))
    axis = axis / max(L, 1e-6)
    kk, jj, ii = np.where(cc_arr > 0)
    if kk.size == 0:
        print(f"  {name}: no CCs in residual")
        return
    ijk = np.stack([ii, jj, kk], axis=1).astype(np.float64)
    h = np.concatenate([ijk, np.ones((ijk.shape[0], 1))], axis=1)
    pts_ras = (ijk_to_ras_mat @ h.T).T[:, :3]
    diffs = pts_ras - gt_entry_ras[None, :]
    proj = diffs @ axis
    perp = diffs - np.outer(proj, axis)
    perp_d = np.linalg.norm(perp, axis=1)
    in_tube = (proj >= -3.0) & (proj <= L + 1.0) & (perp_d <= 2.0)
    tube_labels = cc_arr[kk[in_tube], jj[in_tube], ii[in_tube]]
    if tube_labels.size == 0:
        print(f"  {name}: no voxels near axis in stage-2 residual")
        return
    unique, counts = np.unique(tube_labels, return_counts=True)
    order = np.argsort(-counts)
    print(f"  {name}: CCs touching GT tube:")
    for lab, cnt in zip(unique[order][:5], counts[order][:5]):
        mask = cc_arr == lab
        kk2, jj2, ii2 = np.where(mask)
        n = kk2.size
        pts_mm = np.stack([ii2 * sx, jj2 * sy, kk2 * sz], axis=1).astype(np.float64)
        cc_center = pts_mm.mean(axis=0)
        X = pts_mm - cc_center
        cov = X.T @ X / max(1, n - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = eigvals[::-1]
        aspect = float(eigvals[0] / max(eigvals[1], 1e-6))
        pc1 = eigvecs[:, -1]
        span = float((X @ pc1).max() - (X @ pc1).min())
        print(f"    label={lab} n_vox={n} aspect={aspect:.1f} span={span:.1f}mm "
              f"(GT-tube hits={cnt})")


def run(subject_id):
    import SimpleITK as sitk
    from shank_core.io import image_ijk_ras_matrices

    ct_path = DATASET_ROOT / "post_registered_ct" / f"{subject_id}_post_registered.nii.gz"
    print(f"# subject={subject_id}")
    img = sitk.ReadImage(str(ct_path))

    t0 = time.time()
    hull, intracranial = build_masks(img)
    dist_arr = hull_signed_distance(img, hull)
    log1 = bp.log_sigma(img, sigma_mm=1.0)
    frangi_s1 = frangi_single(img, sigma=1.0)
    ijk_to_ras_mat, ras_to_ijk_mat = image_ijk_ras_matrices(img)
    ijk_to_ras_mat = np.asarray(ijk_to_ras_mat, dtype=float)
    ras_to_ijk_mat = np.asarray(ras_to_ijk_mat, dtype=float)
    print(f"# preprocessing ({time.time()-t0:.1f}s)")

    # --- Stage 1: blob-pitch ---
    t0 = time.time()
    stage1_lines, pts_blobs = run_stage1(
        img, log1, dist_arr=dist_arr, ras_to_ijk_mat=ras_to_ijk_mat,
    )
    print(f"# stage 1 blob-pitch lines: {len(stage1_lines)}  "
          f"({time.time()-t0:.1f}s)")

    # --- Stage 2: Frangi fallback on exclusion-zoned cloud ---
    t0 = time.time()
    excl = compute_exclusion_mask(
        frangi_s1.shape, stage1_lines, ijk_to_ras_mat, ras_to_ijk_mat,
    )
    print(f"# exclusion zone: {int(excl.sum())} voxels "
          f"({time.time()-t0:.1f}s)")
    t0 = time.time()
    stage2_lines, stage2_cc_arr = run_stage2(
        frangi_s1, intracranial, excl, img.GetSpacing(),
        dist_arr=dist_arr,
    )
    print(f"# stage 2 Frangi lines: {len(stage2_lines)}  "
          f"({time.time()-t0:.1f}s)")

    # Convert stage-2 to same start_ras/end_ras format for matching
    def kji_to_ras_arr(kji):
        ijk = np.stack([kji[:, 2], kji[:, 1], kji[:, 0]], axis=1)
        h = np.concatenate([ijk, np.ones((ijk.shape[0], 1))], axis=1)
        return (ijk_to_ras_mat @ h.T).T[:, :3]

    stage2_ras = []
    for l in stage2_lines:
        pts_ras = kji_to_ras_arr(l["polyline_kji"])
        stage2_ras.append(dict(
            start_ras=pts_ras[0], end_ras=pts_ras[-1],
            length_mm=l["length_mm"],
            n_inliers=l["n_inliers"],
            polyline_kji=l["polyline_kji"], polyline_ras=pts_ras,
            dist_min_mm=l.get("dist_min_mm", float("nan")),
            dist_max_mm=l.get("dist_max_mm", float("nan")),
        ))

    # Stage-2 is specifically for pitch-less shanks, so we do NOT apply a
    # periodicity gate. Accept all Frangi-CC hypotheses that survived the
    # span/aspect filters in run_stage2.
    stage2_accepted = stage2_ras
    print(f"# stage 2 accepted (no periodicity gate): {len(stage2_accepted)}")

    # Diagnostic: which CCs are near RSAN's GT axis?
    gt_full = load_gt(subject_id)
    for (name, g_s, g_e, L) in gt_full:
        if name in ("RSAN", "LSAN"):
            describe_ccs_near_axis(stage2_cc_arr, g_s, g_e,
                                    ijk_to_ras_mat, img.GetSpacing(), name)

    # --- Combine + evaluate ---
    combined = []
    for s in stage1_lines:
        combined.append(dict(start_ras=s["start_ras"], end_ras=s["end_ras"],
                             source="stage1",
                             dist_min_mm=s.get("dist_min_mm", float("nan")),
                             dist_max_mm=s.get("dist_max_mm", float("nan"))))
    for s in stage2_accepted:
        combined.append(dict(start_ras=s["start_ras"], end_ras=s["end_ras"],
                             source="stage2",
                             dist_min_mm=s.get("dist_min_mm", float("nan")),
                             dist_max_mm=s.get("dist_max_mm", float("nan"))))

    gt = load_gt(subject_id)
    matches = match_tracks_to_gt(combined, gt)
    n_matched = sum(1 for m in matches if m["matched"])
    fp = len(combined) - n_matched
    print(f"\n# combined total = {len(combined)}  "
          f"matched = {n_matched}/{len(gt)}  FP = {fp}")

    # per-shank report with which stage caught it
    for m in matches:
        src = "-"
        if m["matched"] and 0 <= m.get("track_index", -1) < len(combined):
            src = combined[m["track_index"]]["source"]
        print(
            f"  {m['gt_name']:10s} {'YES' if m['matched'] else 'no':>5s}  "
            f"src={src:7s}  ang={m.get('angle_deg', float('nan')):.2f}°  "
            f"mid_d={m.get('mid_d_mm', float('nan')):.2f}mm"
        )

    # Breakdown
    stage1_matched = sum(
        1 for m in matches
        if m["matched"] and combined[m["track_index"]]["source"] == "stage1"
    )
    stage2_matched = sum(
        1 for m in matches
        if m["matched"] and combined[m["track_index"]]["source"] == "stage2"
    )
    stage1_fp = sum(
        1 for i, c in enumerate(combined)
        if c["source"] == "stage1"
        and i not in {mm["track_index"] for mm in matches if mm["matched"]}
    )
    stage2_fp = sum(
        1 for i, c in enumerate(combined)
        if c["source"] == "stage2"
        and i not in {mm["track_index"] for mm in matches if mm["matched"]}
    )
    print(f"\n# breakdown:  stage1 {stage1_matched} matched + {stage1_fp} FP"
          f"  |  stage2 {stage2_matched} matched + {stage2_fp} FP")

    matched_idx = {mm["track_index"] for mm in matches if mm["matched"]}
    print("\n# stage1 hull-dist (min_mm -> max_mm):")
    for i, c in enumerate(combined):
        if c["source"] != "stage1":
            continue
        tp = "TP" if i in matched_idx else "FP"
        dmin = c.get("dist_min_mm", float("nan"))
        dmax = c.get("dist_max_mm", float("nan"))
        print(f"  {tp}  dist=[{dmin:5.1f},{dmax:5.1f}] mm")
    print("\n# stage2 hull-dist (min_mm -> max_mm):")
    for i, c in enumerate(combined):
        if c["source"] != "stage2":
            continue
        tp = "TP" if i in matched_idx else "FP"
        dmin = c.get("dist_min_mm", float("nan"))
        dmax = c.get("dist_max_mm", float("nan"))
        print(f"  {tp}  dist=[{dmin:5.1f},{dmax:5.1f}] mm")


if __name__ == "__main__":
    subj = sys.argv[1] if len(sys.argv) > 1 else "T2"
    run(subj)
