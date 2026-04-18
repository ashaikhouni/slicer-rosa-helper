"""Probe C: strict periodicity gate as replacement for the length bypass.

v4 accepts a track if (periodic pitch check passes) OR (track length >= 40 mm).
The length bypass is the suspected source of many FPs (skull ridges, straight
bone fragments). Here we measure whether a stricter periodicity test --
windowed local FFT, require a contiguous periodic segment >= 20 mm, no
length bypass -- retains all real electrodes while cutting FPs.

Runs the v4 RANSAC stage unchanged, then re-classifies each candidate with
the new acceptance rule and reports recall + FP count.

Usage:
  /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
    tests/deep_core/probe_periodicity_gate.py [T22|T2]
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

# Reuse v4's RANSAC + helpers.
from probe_detector_v4 import (  # noqa: E402
    DATASET_ROOT, FRANGI_S1_THR, PITCH_STEP_MM, PITCH_BAND_LO, PITCH_BAND_HI,
    PITCH_MIN_MM, PITCH_MAX_MM, PITCH_MIN_SNR,
    MATCH_ANGLE_DEG, MATCH_MIDPOINT_MM,
    build_masks, frangi_single, iterative_ransac, dedup_tracks,
    _orthonormal_basis, _sample_trilinear, _sample_mask, _unit,
    load_gt, match_tracks_to_gt, save_label_volume,
)

# ---- New gate params ----
WINDOW_LEN_MM = 20.0     # sliding FFT window
WINDOW_STEP_MM = 1.25    # 5 samples at 0.25 mm step
PERIODIC_SEGMENT_MIN_MM = 20.0


def _periodic_mask_along_line(
    frangi_s1, polyline_kji, intracranial_mask, pitch_step=PITCH_STEP_MM,
    window_len_mm=WINDOW_LEN_MM, window_step_mm=WINDOW_STEP_MM,
):
    """For a RANSAC line, return (axis position mm, is_periodic array)
    covering the full line length.  Each output sample represents one
    windowed FFT evaluation.
    """
    if polyline_kji.shape[0] < 3:
        return np.array([]), np.array([], dtype=bool)
    center = polyline_kji.mean(axis=0)
    X = polyline_kji - center
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    axis = _unit(Vt[0])
    proj = X @ axis
    lo, hi = float(proj.min()), float(proj.max())
    total_len = hi - lo
    if total_len < window_len_mm:
        return np.array([]), np.array([], dtype=bool)

    # Dense 1D signal along line
    n_steps = int(total_len / pitch_step) + 1
    u, v = _orthonormal_basis(axis)
    vals = np.empty(n_steps, dtype=float)
    for i in range(n_steps):
        c = center + (lo + i * pitch_step) * axis
        # No intracranial mask gating -- we want to see periodicity
        # wherever it occurs along the line, not only in a pre-masked region.
        samples = [_sample_trilinear(frangi_s1, c)]
        for k in range(8):
            ang = 2.0 * np.pi * k / 8
            off = 1.0 * (np.cos(ang) * u + np.sin(ang) * v)
            samples.append(_sample_trilinear(frangi_s1, c + off))
        s = np.asarray(samples, dtype=float)
        s = s[np.isfinite(s)]
        vals[i] = float(s.max()) if s.size else np.nan

    # Sliding-window FFT
    W = int(round(window_len_mm / pitch_step))
    step = max(1, int(round(window_step_mm / pitch_step)))
    positions = []
    periodic = []
    freqs = np.fft.rfftfreq(W, d=pitch_step)
    band_mask = (freqs > PITCH_BAND_LO) & (freqs < PITCH_BAND_HI)
    for start in range(0, n_steps - W + 1, step):
        w = vals[start:start + W]
        fin = np.isfinite(w)
        if fin.sum() < W * 0.8:
            positions.append(lo + (start + W // 2) * pitch_step)
            periodic.append(False)
            continue
        w = np.where(fin, w, np.nanmean(w[fin]))
        w = w - w.mean()
        spec = np.fft.rfft(w)
        power = (np.abs(spec) ** 2) / W
        if not band_mask.any():
            positions.append(lo + (start + W // 2) * pitch_step)
            periodic.append(False)
            continue
        band = power[band_mask]
        bfreq = freqs[band_mask]
        pi = int(np.argmax(band))
        peak_pitch = 1.0 / float(bfreq[pi]) if bfreq[pi] > 0 else float("nan")
        peak_power = float(band[pi])
        excl = np.zeros_like(band, dtype=bool)
        excl[max(0, pi - 3): pi + 4] = True
        bg = band[~excl]
        bg_power = float(bg.mean()) if bg.size else 0.0
        snr = peak_power / bg_power if bg_power > 0 else float("inf")
        ok = (
            np.isfinite(peak_pitch)
            and PITCH_MIN_MM <= peak_pitch <= PITCH_MAX_MM
            and snr >= PITCH_MIN_SNR
        )
        positions.append(lo + (start + W // 2) * pitch_step)
        periodic.append(bool(ok))
    return np.asarray(positions), np.asarray(periodic, dtype=bool)


def _longest_periodic_run_mm(is_periodic, window_step_mm=WINDOW_STEP_MM):
    if is_periodic.size == 0:
        return 0.0
    best = cur = 0
    for v in is_periodic:
        if v:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    # A run of K windows covers (K-1)*step + window_len. But for comparison
    # stability, report the axis span from first to last periodic position
    # in the longest run.
    # Find longest run's indices
    if best == 0:
        return 0.0
    run_end = run_start = 0
    cur_start = 0
    cur = 0
    for i, v in enumerate(is_periodic):
        if v:
            if cur == 0:
                cur_start = i
            cur += 1
            if cur > run_end - run_start + 1:
                run_start = cur_start
                run_end = i
        else:
            cur = 0
    # Span = (run_end - run_start) * window_step + window_len
    return float((run_end - run_start) * window_step_mm + WINDOW_LEN_MM)


def run(subject_id):
    import SimpleITK as sitk
    from shank_core.io import image_ijk_ras_matrices

    ct_path = DATASET_ROOT / "post_registered_ct" / f"{subject_id}_post_registered.nii.gz"
    print(f"# subject={subject_id}  ct={ct_path.name}")
    img = sitk.ReadImage(str(ct_path))

    t0 = time.time()
    hull, intracranial = build_masks(img)
    print(f"# masks: intracranial={int(intracranial.sum())} "
          f"({time.time()-t0:.1f}s)")
    t0 = time.time()
    frangi_s1 = frangi_single(img, sigma=1.0)
    print(f"# frangi sigma=1: max={frangi_s1.max():.1f} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    tracks = iterative_ransac(frangi_s1, intracranial)
    n_pre = len(tracks)
    tracks = dedup_tracks(tracks)
    print(f"# RANSAC: {n_pre} -> {len(tracks)} after dedup ({time.time()-t0:.1f}s)")

    ijk_to_ras_mat, _ = image_ijk_ras_matrices(img)
    ijk_to_ras_mat = np.asarray(ijk_to_ras_mat, dtype=float)

    def kji_to_ras(kji):
        ijk = np.stack([kji[:, 2], kji[:, 1], kji[:, 0]], axis=1)
        h = np.concatenate([ijk, np.ones((ijk.shape[0], 1))], axis=1)
        return (ijk_to_ras_mat @ h.T).T[:, :3]

    # Classify each track
    rows = []
    for ti, t in enumerate(tracks):
        pts_ras = kji_to_ras(t["polyline_kji"])
        # Windowed periodicity
        _, is_per = _periodic_mask_along_line(
            frangi_s1, t["polyline_kji"], intracranial
        )
        seg_mm = _longest_periodic_run_mm(is_per)
        accepted_strict = seg_mm >= PERIODIC_SEGMENT_MIN_MM
        rows.append(dict(
            index=ti, length_mm=float(t["length_mm"]),
            n_inliers=int(t["n_inliers"]),
            periodic_segment_mm=seg_mm,
            accepted=accepted_strict,
            polyline_kji=t["polyline_kji"],
            start_ras=pts_ras[0], end_ras=pts_ras[-1],
        ))

    gt = load_gt(subject_id)
    accepted = [r for r in rows if r["accepted"]]
    print(f"\n# tracks={len(rows)}  accepted(strict periodicity >= "
          f"{PERIODIC_SEGMENT_MIN_MM:.0f}mm)={len(accepted)}  GT={len(gt)}")

    # Build the (start,end,axis) format match_tracks_to_gt wants
    cand_for_match = [
        dict(start_ras=r["start_ras"], end_ras=r["end_ras"])
        for r in accepted
    ]
    matches = match_tracks_to_gt(cand_for_match, gt)
    n_matched = sum(1 for m in matches if m["matched"])
    fp = len(accepted) - n_matched
    print(f"# STRICT: matched={n_matched}/{len(gt)}  FP={fp}")

    # Also for comparison: v4's old rule (length >= 40 bypass OR periodic pass)
    # Note: I already don't have the v4 periodicity value here, but we can
    # approximate with "length >= 40" alone to isolate the bypass contribution.
    v4_bypass = [r for r in rows if r["length_mm"] >= 40.0]
    v4_b_match = match_tracks_to_gt(
        [dict(start_ras=r["start_ras"], end_ras=r["end_ras"]) for r in v4_bypass],
        gt,
    )
    n_v4b = sum(1 for m in v4_b_match if m["matched"])
    print(f"# for reference, 'length>=40 alone' path: matched={n_v4b}/{len(gt)} "
          f"FP={len(v4_bypass) - n_v4b}")

    # Per-track table (truncated)
    print(
        f"\n{'#':>3s} {'len':>6s} {'n_in':>5s} {'per_seg':>8s} {'strict':>7s}"
    )
    rows.sort(key=lambda r: -r["periodic_segment_mm"])
    for r in rows:
        print(
            f"{r['index']+1:>3d} {r['length_mm']:>6.1f} {r['n_inliers']:>5d} "
            f"{r['periodic_segment_mm']:>8.1f} "
            f"{'YES' if r['accepted'] else 'no':>7s}"
        )

    print(f"\n{'shank':10s} {'matched':>8s} {'ang':>6s} {'mid_d':>6s}")
    for m in matches:
        print(
            f"{m['gt_name']:10s} {'YES' if m['matched'] else 'no':>8s} "
            f"{m.get('angle_deg', float('nan')):6.2f} "
            f"{m.get('mid_d_mm', float('nan')):6.2f}"
        )


if __name__ == "__main__":
    subj = sys.argv[1] if len(sys.argv) > 1 else "T22"
    run(subj)
