"""Probe F: oracle test for Approach A -- given the correct GT axis for
each shank, can we recover contacts from the LoG 1D profile?

This answers: is the LPRG miss a signal problem or an axis-finding problem?

Method:
  1. Compute LoG sigma=1 (and sigma=0.5) on the raw CT.
  2. For each GT shank, sample LoG along the GT axis with a small
     cylindrical max reducer (take min across 8 surrounding points at 0.5 mm
     radius -- contacts are dark in LoG, so min = strongest contact signal).
  3. scipy.find_peaks on -profile (minima in LoG = contacts) with
     min_distance = 3 mm and prominence thresholds.
  4. Report per-shank: peak count, median pitch, pitch std, peak amplitudes.
  5. Flag shanks where peak pitch is inconsistent with vendor spec
     (Dixi: 3.5 mm).

Usage:
  /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
    tests/deep_core/probe_log_oracle.py [T22|T2]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np


def find_peaks(signal, distance=1, prominence=0.0):
    """Minimal find_peaks: returns indices of local maxima with min-distance
    and prominence constraints. Enough for our 1D profile analysis."""
    signal = np.asarray(signal, dtype=float)
    n = signal.size
    # local maxima candidates
    cand = []
    for i in range(1, n - 1):
        if signal[i] > signal[i - 1] and signal[i] >= signal[i + 1]:
            # prominence: signal[i] minus the max of the valleys to either
            # side (walk outward until we find a value >= signal[i]).
            # Simple approximation: min within +-distance neighborhood.
            lo = max(0, i - distance)
            hi = min(n, i + distance + 1)
            local_min = float(np.min(signal[lo:hi]))
            prom = signal[i] - local_min
            if prom >= prominence:
                cand.append((i, signal[i], prom))
    if not cand:
        return np.array([], dtype=int), {}
    # enforce min-distance: sort by height desc, greedily keep
    cand.sort(key=lambda t: -t[1])
    kept = []
    kept_idx = []
    for i, h, pr in cand:
        if all(abs(i - j) >= distance for j in kept_idx):
            kept_idx.append(i)
            kept.append((i, h, pr))
    kept.sort(key=lambda t: t[0])
    return np.array([t[0] for t in kept], dtype=int), {}

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))
sys.path.insert(0, str(REPO_ROOT / "PostopCTLocalization"))
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "deep_core"))

from probe_detector_v4 import (  # noqa: E402
    DATASET_ROOT, _orthonormal_basis, _sample_trilinear, _unit, load_gt,
)

STEP_MM = 0.25
RING_RADIUS_MM = 0.5
RING_N = 8

# Vendor prior: Dixi Microdeep
EXPECTED_PITCH_MM = 3.5
PITCH_TOLERANCE_MM = 0.4


def log_sigma(img, sigma_mm):
    import SimpleITK as sitk
    log = sitk.LaplacianRecursiveGaussian(img, sigma=float(sigma_mm))
    return sitk.GetArrayFromImage(log).astype(np.float32)


def sample_log_profile(log_arr, ras_to_ijk, start_ras, end_ras, step_mm):
    """Sample LoG along a line from start_ras to end_ras. At each axial
    position, take the MIN over a small ring (contacts are negative in LoG,
    so min = strongest contact signal; accounts for axis offset up to
    ring_radius).
    """
    d = end_ras - start_ras
    L = float(np.linalg.norm(d))
    axis = d / L
    u, v = _orthonormal_basis(axis)
    n_steps = int(L / step_mm) + 1
    positions = np.arange(n_steps) * step_mm
    profile = np.empty(n_steps, dtype=float)
    for i, s in enumerate(positions):
        c_ras = start_ras + s * axis
        c_ijk = ras_to_ijk @ np.array([c_ras[0], c_ras[1], c_ras[2], 1.0])
        c_kji = np.array([c_ijk[2], c_ijk[1], c_ijk[0]])
        # Center + ring of RING_N points
        samples = [_sample_trilinear(log_arr, c_kji)]
        for k in range(RING_N):
            ang = 2.0 * np.pi * k / RING_N
            off_ras = RING_RADIUS_MM * (np.cos(ang) * u + np.sin(ang) * v)
            p_ras = c_ras + off_ras
            p_ijk = ras_to_ijk @ np.array([p_ras[0], p_ras[1], p_ras[2], 1.0])
            p_kji = np.array([p_ijk[2], p_ijk[1], p_ijk[0]])
            samples.append(_sample_trilinear(log_arr, p_kji))
        s = np.asarray(samples, dtype=float)
        s = s[np.isfinite(s)]
        profile[i] = float(s.min()) if s.size else np.nan
    return positions, profile


def detect_contacts(profile, step_mm, min_distance_mm=3.0, prominence=50.0):
    """Find negative peaks (contact minima) in LoG profile."""
    finite = np.isfinite(profile)
    if finite.sum() < 5:
        return np.array([], dtype=int), np.array([])
    filled = np.where(finite, profile, np.nanmax(profile[finite]))
    inv = -filled
    min_dist = max(1, int(min_distance_mm / step_mm))
    peaks, _ = find_peaks(inv, distance=min_dist, prominence=prominence)
    amps = profile[peaks]
    return peaks, amps


def analyze(subject_id):
    import SimpleITK as sitk
    from shank_core.io import image_ijk_ras_matrices

    ct_path = DATASET_ROOT / "post_registered_ct" / f"{subject_id}_post_registered.nii.gz"
    print(f"# subject={subject_id}")
    img = sitk.ReadImage(str(ct_path))

    t0 = time.time()
    log1 = log_sigma(img, sigma_mm=1.0)
    log05 = log_sigma(img, sigma_mm=0.5)
    print(f"# LoG sigma=1 min={log1.min():.1f}  "
          f"LoG sigma=0.5 min={log05.min():.1f}  ({time.time()-t0:.1f}s)")

    _, ras_to_ijk = image_ijk_ras_matrices(img)
    ras_to_ijk = np.asarray(ras_to_ijk, dtype=float)

    gt = load_gt(subject_id)
    print(f"# GT shanks: {len(gt)}")
    print(
        f"\n{'shank':10s} {'L_mm':>6s} "
        f"{'s1_peaks':>9s} {'s1_pitch':>9s} {'s1_amp_p50':>11s} "
        f"{'s05_peaks':>10s} {'s05_pitch':>10s} {'s05_amp_p50':>12s} "
        f"{'dixi?':>6s}"
    )

    for (name, g_s, g_e, L) in gt:
        # Exclude proximal 5 mm (bolt area) from pitch estimation
        axis = (g_e - g_s) / max(float(np.linalg.norm(g_e - g_s)), 1e-6)
        inner_start = g_s + 5.0 * axis
        pos1, prof1 = sample_log_profile(log1, ras_to_ijk, inner_start, g_e, STEP_MM)
        pos05, prof05 = sample_log_profile(log05, ras_to_ijk, inner_start, g_e, STEP_MM)

        peaks1, amps1 = detect_contacts(prof1, STEP_MM, prominence=50.0)
        peaks05, amps05 = detect_contacts(prof05, STEP_MM, prominence=100.0)

        def pitch_stats(peaks):
            if len(peaks) < 2:
                return float("nan"), float("nan")
            diffs = np.diff(peaks) * STEP_MM
            return float(np.median(diffs)), float(np.std(diffs))

        med1, std1 = pitch_stats(peaks1)
        med05, std05 = pitch_stats(peaks05)

        amp1_p50 = float(np.median(amps1)) if amps1.size else float("nan")
        amp05_p50 = float(np.median(amps05)) if amps05.size else float("nan")

        dixi_ok1 = (
            np.isfinite(med1)
            and abs(med1 - EXPECTED_PITCH_MM) < PITCH_TOLERANCE_MM
            and len(peaks1) >= 6
        )
        dixi_ok05 = (
            np.isfinite(med05)
            and abs(med05 - EXPECTED_PITCH_MM) < PITCH_TOLERANCE_MM
            and len(peaks05) >= 6
        )
        dixi_tag = "YES" if (dixi_ok1 or dixi_ok05) else "no"

        print(
            f"{name:10s} {L:6.1f} "
            f"{len(peaks1):>9d} {med1:>9.2f} {amp1_p50:>11.1f} "
            f"{len(peaks05):>10d} {med05:>10.2f} {amp05_p50:>12.1f} "
            f"{dixi_tag:>6s}"
        )

    # Also report LPRG's full profile for T2 specifically.
    if subject_id == "T2":
        for (name, g_s, g_e, L) in gt:
            if name != "LPRG":
                continue
            axis = (g_e - g_s) / max(float(np.linalg.norm(g_e - g_s)), 1e-6)
            inner_start = g_s + 5.0 * axis
            pos1, prof1 = sample_log_profile(
                log1, ras_to_ijk, inner_start, g_e, STEP_MM
            )
            print(f"\n# LPRG LoG sigma=1 profile (every 1 mm):")
            for i in range(0, len(pos1), int(1.0 / STEP_MM)):
                p = prof1[i]
                bar = "#" * max(0, int(abs(p) / 30)) if np.isfinite(p) else ""
                print(f"  s={pos1[i]:5.1f}mm  log={p:8.1f}  {bar}")


if __name__ == "__main__":
    subj = sys.argv[1] if len(sys.argv) > 1 else "T2"
    analyze(subj)
