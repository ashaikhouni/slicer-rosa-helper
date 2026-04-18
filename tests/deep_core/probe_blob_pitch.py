"""Probe G: blob-space detector with vendor-pitch prior.

Replace voxel-cloud RANSAC with blob-cloud RANSAC:

  1. LoG sigma=1 on the raw CT; intracranial region is skipped (no mask).
  2. Threshold LoG <= T (strong dark blobs) and connected-component to get
     discrete blob candidates. One RAS point per blob (weighted centroid).
     Typically ~300-800 blobs per subject -- dramatically smaller search
     space than voxels.
  3. Pitch-seeded line growth: for each blob pair within Dixi pitch
     [3.2, 3.8] mm, define an axis and walk both directions at 3.5 mm
     steps; at each expected contact location, accept a blob if perp
     < 1.0 mm and axial residual < 0.5 mm. Record the run.
  4. Filter: keep lines with >= 6 aligned blobs (>=15 mm span).
  5. Dedup.
  6. Evaluate: GT match (angle <=10, mid_d <=8).

Usage:
  /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
    tests/deep_core/probe_blob_pitch.py [T22|T2] [LOG_THR]
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

from probe_detector_v4 import (  # noqa: E402
    DATASET_ROOT, MATCH_ANGLE_DEG, MATCH_MIDPOINT_MM,
    load_gt, match_tracks_to_gt, _unit,
)

# Vendor prior (Dixi Microdeep)
PITCH_MM = 3.5
PITCH_TOL_MM = 0.5       # accept slight pitch variation (calibration)
PERP_TOL_MM = 1.5        # contacts can sit 1-1.5 mm off the RANSAC axis
AX_TOL_MM = 0.7          # axial residual vs k*pitch
MAX_K_STEPS = 20         # up to 40 contacts
MIN_BLOBS_PER_LINE = 6

DEDUP_ANGLE_DEG = 3.0
DEDUP_PERP_MM = 2.0
DEDUP_OVERLAP_FRAC = 0.3


def log_sigma(img, sigma_mm):
    import SimpleITK as sitk
    log = sitk.LaplacianRecursiveGaussian(img, sigma=float(sigma_mm))
    return sitk.GetArrayFromImage(log).astype(np.float32)


def extract_blobs(log_arr, img, threshold):
    """Regional-minima blob extraction. Each contact (local LoG minimum)
    becomes one blob, regardless of whether its LoG well is connected to
    neighbors through the skull. Uses SITK grayscale erode to find local
    minima within a ~1 mm radius, then thresholds by absolute LoG value.
    """
    import SimpleITK as sitk
    # Grayscale erode = local MIN over neighborhood
    erode = sitk.GrayscaleErode(
        sitk.GetImageFromArray(log_arr),
        kernelRadius=[2, 2, 2],  # ~1 mm at 0.5 mm spacing
    )
    eroded = sitk.GetArrayFromImage(erode).astype(np.float32)
    # A voxel is a regional min if value equals local min
    is_local_min = (log_arr <= eroded + 1e-4)
    strong = is_local_min & (log_arr <= -abs(threshold))
    kk, jj, ii = np.where(strong)
    blobs = []
    for k, j, i in zip(kk, jj, ii):
        val = float(log_arr[k, j, i])
        blobs.append(dict(
            kji=np.array([float(k), float(j), float(i)]),
            amp=-val, n_vox=1,  # 1 since we represent each min as a point
        ))
    return blobs


def kji_to_ras_fn(img):
    from shank_core.io import image_ijk_ras_matrices
    m, _ = image_ijk_ras_matrices(img)
    m = np.asarray(m, dtype=float)
    def _f(kji):
        if kji.ndim == 1:
            i, j, k = float(kji[2]), float(kji[1]), float(kji[0])
            return (m @ np.array([i, j, k, 1.0]))[:3]
        ijk = np.stack([kji[:, 2], kji[:, 1], kji[:, 0]], axis=1)
        h = np.concatenate([ijk, np.ones((ijk.shape[0], 1))], axis=1)
        return (m @ h.T).T[:, :3]
    return _f


def _walk_with_pitch(axis, anchor, pts, amps, pitch, perp_tol, ax_tol, max_k):
    """Walk at a fixed pitch from anchor along axis, collecting inlier
    blobs. Returns {"inliers": set, "pitch": float, "n_inliers": int}."""
    diffs = pts - anchor
    proj = diffs @ axis
    perp = diffs - np.outer(proj, axis)
    perp_d = np.linalg.norm(perp, axis=1)
    within_perp = perp_d <= perp_tol
    inliers = set()
    for k in range(-max_k, max_k + 1):
        target = k * pitch
        ax_resid = np.abs(proj - target)
        cand = np.where(within_perp & (ax_resid <= ax_tol))[0]
        if cand.size == 0:
            continue
        best_blob = cand[np.argmax(amps[cand])]
        inliers.add(int(best_blob))
    if not inliers:
        return None
    return dict(inliers=inliers, pitch=pitch, n_inliers=len(inliers))


def walk_line(seed_idx, neighbor_idx, pts, amps,
              pitch=PITCH_MM, perp_tol=PERP_TOL_MM, ax_tol=AX_TOL_MM,
              max_k=MAX_K_STEPS):
    """Given a seed pair, walk both directions and accumulate aligned
    blobs. Multi-pitch seed: try several candidate pitches around the seed
    estimate (covers per-shank pitch variation + sub-voxel detection
    noise) and keep the walk with most inliers.
    """
    p0 = pts[seed_idx]
    p1 = pts[neighbor_idx]
    seed_d = float(np.linalg.norm(p1 - p0))
    k_seed = max(1, int(round(seed_d / pitch)))
    pitch_seed = seed_d / k_seed
    if not (PITCH_MM - PITCH_TOL_MM <= pitch_seed <= PITCH_MM + PITCH_TOL_MM):
        return None
    axis = (p1 - p0) / seed_d
    # Multi-pitch candidates: seed estimate + small perturbations. Handles
    # shanks whose real contact spacing drifts across the line.
    best = None
    for dp in (-0.2, -0.1, 0.0, 0.1, 0.2):
        pitch_try = pitch_seed + dp
        if not (PITCH_MM - PITCH_TOL_MM <= pitch_try <= PITCH_MM + PITCH_TOL_MM):
            continue
        r = _walk_with_pitch(axis, p0, pts, amps, pitch_try, perp_tol, ax_tol, max_k)
        if r is None:
            continue
        if best is None or r["n_inliers"] > best["n_inliers"]:
            best = r
    if best is None:
        return None
    inliers = list(best["inliers"])
    if len(inliers) < MIN_BLOBS_PER_LINE:
        return None
    # PCA-refine axis on inlier blobs
    inlier_pts = pts[inliers]
    c = inlier_pts.mean(axis=0)
    X = inlier_pts - c
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    axis_ref = _unit(Vt[0])
    proj_ref = X @ axis_ref
    lo, hi = float(proj_ref.min()), float(proj_ref.max())
    span = hi - lo
    if span < 15.0 or span > 90.0:
        return None
    # Measure actual pitch consistency on inliers -- real shanks have low
    # std in consecutive-gap distribution; FPs have erratic gaps
    sorted_proj = np.sort(proj_ref)
    gaps = np.diff(sorted_proj)
    # Use only gaps that look like single-pitch hits (k=1 spacing)
    single_gaps = gaps[(gaps >= PITCH_MM - PITCH_TOL_MM)
                        & (gaps <= PITCH_MM + PITCH_TOL_MM)]
    pitch_std = float(np.std(single_gaps)) if single_gaps.size >= 2 else 99.0
    return dict(
        axis=axis_ref, center=c, inlier_idx=inliers,
        span_mm=span, span_lo=lo, span_hi=hi,
        start_ras=c + lo * axis_ref, end_ras=c + hi * axis_ref,
        n_blobs=len(inliers),
        pitch_std=pitch_std,
        amp_sum=float(np.sum([amps[i] for i in inliers])),
    )


def dedup_lines(lines):
    if len(lines) < 2:
        return lines
    keep = [True] * len(lines)
    for i in range(len(lines)):
        if not keep[i]:
            continue
        a = lines[i]
        for j in range(i + 1, len(lines)):
            if not keep[j]:
                continue
            b = lines[j]
            ang = float(np.degrees(np.arccos(
                np.clip(abs(np.dot(a["axis"], b["axis"])), 0, 1))))
            if ang > DEDUP_ANGLE_DEG:
                continue
            dv = b["center"] - a["center"]
            par = dv @ a["axis"]
            perp = dv - par * a["axis"]
            perp_d = float(np.linalg.norm(perp))
            if perp_d > DEDUP_PERP_MM:
                continue
            b_lo = par + b["span_lo"]; b_hi = par + b["span_hi"]
            overlap = max(0.0, min(a["span_hi"], b_hi) - max(a["span_lo"], b_lo))
            shorter = min(a["span_hi"] - a["span_lo"], b["span_hi"] - b["span_lo"])
            if shorter > 1e-6 and overlap / shorter >= DEDUP_OVERLAP_FRAC:
                # keep the one with more inliers
                if a["n_blobs"] >= b["n_blobs"]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break
    return [lines[i] for i in range(len(lines)) if keep[i]]


def run(subject_id, log_thr):
    import SimpleITK as sitk

    ct_path = DATASET_ROOT / "post_registered_ct" / f"{subject_id}_post_registered.nii.gz"
    print(f"# subject={subject_id}  log_thr={log_thr}")
    img = sitk.ReadImage(str(ct_path))

    t0 = time.time()
    log1 = log_sigma(img, sigma_mm=1.0)
    print(f"# LoG sigma=1 min={log1.min():.1f}  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    blobs = extract_blobs(log1, img, threshold=log_thr)
    print(f"# blobs: {len(blobs)}  ({time.time()-t0:.1f}s)")

    to_ras = kji_to_ras_fn(img)
    pts_ras = np.array([to_ras(b["kji"]) for b in blobs])
    amps = np.array([b["amp"] for b in blobs], dtype=float)
    n_vox = np.array([b["n_vox"] for b in blobs], dtype=int)

    # Blob-size histogram
    if len(blobs) > 0:
        pc = np.percentile(n_vox, [50, 75, 90, 95, 99])
        print(f"# blob size p50/75/90/95/99: {pc}  max={n_vox.max()}")

    # Filter huge blobs (likely bolts or skull bits, not contacts)
    # Contacts on Dixi: ~1-2 mm, so < ~50 voxels usually. Keep permissive
    # upper bound so we don't cut elongated-but-still-informative blobs.
    contact_mask = n_vox <= 500
    pts_c = pts_ras[contact_mask]
    amps_c = amps[contact_mask]
    print(f"# contact-sized blobs (<=500 vox): {pts_c.shape[0]}")

    # Find pitch-compatible pairs: distance in {1,2,3} * pitch (skip up to
    # 2 missed contacts between endpoints). This adds robustness when some
    # contacts fall below the LoG threshold.
    N = pts_c.shape[0]
    t0 = time.time()
    dist2 = np.sum((pts_c[:, None, :] - pts_c[None, :, :]) ** 2, axis=2)
    dist = np.sqrt(dist2)
    pair_mask = np.zeros_like(dist, dtype=bool)
    for mult in (1, 2, 3):
        lo = mult * PITCH_MM - PITCH_TOL_MM
        hi = mult * PITCH_MM + PITCH_TOL_MM
        pair_mask |= (dist >= lo) & (dist <= hi)
    iu, ju = np.where(np.triu(pair_mask, k=1))
    n_pairs = iu.size
    print(f"# pairs at k*pitch (k=1,2,3): {n_pairs}  "
          f"({time.time()-t0:.1f}s)")

    # Walk each pair
    t0 = time.time()
    hypotheses = []
    for pi, pj in zip(iu, ju):
        h = walk_line(int(pi), int(pj), pts_c, amps_c)
        if h is not None:
            hypotheses.append(h)
    print(f"# hypotheses (>= {MIN_BLOBS_PER_LINE} aligned blobs): "
          f"{len(hypotheses)}  ({time.time()-t0:.1f}s)")

    # Sort by n_blobs descending (strongest lines first) for stable dedup
    hypotheses.sort(key=lambda h: -h["n_blobs"])
    lines = dedup_lines(hypotheses)
    print(f"# after dedup: {len(lines)}")

    # FP gate: real shanks have high total blob amplitude (strong LoG
    # minima); skull/bone spurious lines assemble from weak blobs.
    AMP_SUM_MIN = 6000.0
    lines = [l for l in lines if l.get("amp_sum", 0.0) >= AMP_SUM_MIN]
    print(f"# after amp_sum >= {AMP_SUM_MIN:.0f}: {len(lines)}")

    # Build candidates for matching (start/end RAS)
    cand = [
        dict(start_ras=l["start_ras"], end_ras=l["end_ras"])
        for l in lines
    ]
    gt = load_gt(subject_id)
    matches = match_tracks_to_gt(cand, gt)
    n_matched = sum(1 for m in matches if m["matched"])
    fp = len(cand) - n_matched
    print(f"\n# matched={n_matched}/{len(gt)}  FP={fp}")

    # Focused diagnostic: for each missed shank, find the best blob seed
    # pair along the GT axis and simulate the walk.
    print(f"\n## Failure diagnostic for missed shanks (oracle seed)")
    for m in matches:
        if m["matched"]:
            continue
        name = m["gt_name"]
        g = next(x for x in gt if x[0] == name)
        _, g_s, g_e, L = g
        axis_g = (g_e - g_s) / max(float(np.linalg.norm(g_e - g_s)), 1e-6)
        diffs = pts_c - g_s[None, :]
        proj_g = diffs @ axis_g
        perp_g = diffs - np.outer(proj_g, axis_g)
        perp_d = np.linalg.norm(perp_g, axis=1)
        in_seg = (proj_g >= -3.0) & (proj_g <= L + 1.0) & (perp_d <= 2.0)
        near_idx = np.where(in_seg)[0]
        if near_idx.size < 2:
            print(f"  {name}: only {near_idx.size} blobs near axis, cannot seed")
            continue
        # Enumerate pairs of near-axis blobs at k*pitch distances
        best_hyp = None
        best_seed_info = None
        for ii in range(near_idx.size):
            for jj in range(ii + 1, near_idx.size):
                bi, bj = int(near_idx[ii]), int(near_idx[jj])
                d = float(np.linalg.norm(pts_c[bj] - pts_c[bi]))
                # Check if distance is near k*pitch for any k
                is_pitchy = any(
                    abs(d - k * PITCH_MM) <= PITCH_TOL_MM for k in (1, 2, 3)
                )
                if not is_pitchy:
                    continue
                h = walk_line(bi, bj, pts_c, amps_c)
                if h is None:
                    continue
                if best_hyp is None or h["n_blobs"] > best_hyp["n_blobs"]:
                    best_hyp = h
                    best_seed_info = (bi, bj, d)
        if best_hyp is None:
            # Which pairs were pitchy?  For each, why did the walk fail?
            print(f"  {name}: walker returned None for all pitchy pairs. "
                  f"Diagnosing first 3:")
            n_shown = 0
            for ii in range(near_idx.size):
                for jj in range(ii + 1, near_idx.size):
                    bi, bj = int(near_idx[ii]), int(near_idx[jj])
                    d = float(np.linalg.norm(pts_c[bj] - pts_c[bi]))
                    if not any(abs(d - k * PITCH_MM) <= PITCH_TOL_MM
                                for k in (1, 2, 3)):
                        continue
                    # Simulate walk with GT axis to see theoretical max
                    axis_g_local = (pts_c[bj] - pts_c[bi]) / max(d, 1e-6)
                    # axis angle to GT
                    ang_to_gt = float(np.degrees(np.arccos(np.clip(
                        abs(np.dot(axis_g_local, axis_g)), 0, 1))))
                    # Try walk with relaxed perp_tol to see how many blobs
                    # hit the seed axis
                    for pt_try in (1.0, 1.5, 2.0):
                        r = walk_line(bi, bj, pts_c, amps_c, perp_tol=pt_try)
                        nb = r["n_blobs"] if r is not None else 0
                        if nb > 0:
                            print(f"    seed {ii}->{jj} d={d:.2f} "
                                  f"ang_to_gt={ang_to_gt:.2f}° "
                                  f"perp_tol={pt_try} -> {nb} blobs")
                            break
                    else:
                        print(f"    seed {ii}->{jj} d={d:.2f} "
                              f"ang_to_gt={ang_to_gt:.2f}° -- 0 walks found")
                    n_shown += 1
                    if n_shown >= 3:
                        break
                if n_shown >= 3:
                    break
        else:
            bi, bj, d = best_seed_info
            test_axis = best_hyp["axis"]
            ang = float(np.degrees(np.arccos(
                np.clip(abs(np.dot(axis_g, test_axis)), 0, 1))))
            print(f"  {name}: best oracle hyp has {best_hyp['n_blobs']} blobs, "
                  f"span {best_hyp['span_mm']:.1f} mm, seed d={d:.2f} mm, "
                  f"axis err {ang:.2f}° -- "
                  f"{'should match (likely deduped away?)' if best_hyp['n_blobs'] >= MIN_BLOBS_PER_LINE else 'below min_blobs'}")

    # Per-GT diagnostic: count blobs near each GT axis (within 2 mm perp,
    # within the shank segment) and spacing between consecutive such blobs.
    print(f"\n## Per-GT blob availability near true axis")
    print(f"{'shank':10s} {'L':>5s} {'n_near':>6s} {'gaps(mm)':>25s}")
    for (name, g_s, g_e, L) in gt:
        axis_g = (g_e - g_s) / max(float(np.linalg.norm(g_e - g_s)), 1e-6)
        diffs = pts_c - g_s[None, :]
        proj = diffs @ axis_g
        perp = diffs - np.outer(proj, axis_g)
        perp_d = np.linalg.norm(perp, axis=1)
        in_seg = (proj >= -3.0) & (proj <= L + 1.0) & (perp_d <= 2.0)
        near_idx = np.where(in_seg)[0]
        near_proj = np.sort(proj[near_idx])
        gaps = np.diff(near_proj) if near_proj.size >= 2 else np.array([])
        gap_str = " ".join(f"{g:.2f}" for g in gaps[:10])
        print(f"{name:10s} {L:5.1f} {near_idx.size:>6d}   {gap_str}")
    for m in matches:
        print(
            f"  {m['gt_name']:10s} {'YES' if m['matched'] else 'no':>5s}  "
            f"ang={m.get('angle_deg', float('nan')):.2f}deg  "
            f"mid_d={m.get('mid_d_mm', float('nan')):.2f}mm"
        )

    # Per-line summary with match flag
    matched_ti = {mm["track_index"] for mm in matches if mm["matched"]}
    print(f"\n{'#':>3s} {'n_blobs':>7s} {'span':>6s} {'p_std':>6s} "
          f"{'amp_sum':>8s} {'match':>6s}")
    for i, l in enumerate(lines):
        is_match = i in matched_ti
        print(
            f"{i+1:>3d} {l['n_blobs']:>7d} {l['span_mm']:>6.1f} "
            f"{l.get('pitch_std', 0.0):>6.3f} "
            f"{l.get('amp_sum', 0.0):>8.0f} "
            f"{'YES' if is_match else '-':>6s}"
        )

    # Write NIFTI of accepted blobs (color by line index)
    label = np.zeros_like(log1, dtype=np.uint16)
    for li, l in enumerate(lines, start=1):
        for bi in l["inlier_idx"]:
            # blob -> original index in blobs[]
            orig_idx = np.where(contact_mask)[0][bi]
            kji = blobs[orig_idx]["kji"]
            k, j, i_ = int(round(kji[0])), int(round(kji[1])), int(round(kji[2]))
            if 0 <= k < label.shape[0] and 0 <= j < label.shape[1] and 0 <= i_ < label.shape[2]:
                for dk in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        for di in (-1, 0, 1):
                            kk, jj, ii = k + dk, j + dj, i_ + di
                            if (0 <= kk < label.shape[0] and 0 <= jj < label.shape[1]
                                    and 0 <= ii < label.shape[2]):
                                label[kk, jj, ii] = li
    out_img = sitk.GetImageFromArray(label)
    out_img.CopyInformation(img)
    out_path = f"/tmp/blob_lines_{subject_id}_T{int(log_thr)}.nii.gz"
    sitk.WriteImage(out_img, out_path)
    print(f"\n# wrote {out_path}")


if __name__ == "__main__":
    subj = sys.argv[1] if len(sys.argv) > 1 else "T2"
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 200.0
    run(subj, thr)
