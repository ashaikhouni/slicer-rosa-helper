"""Centerline snap to LoG-bright centroids (Stage B refinement).

Two variants:

* ``snap_centerline_to_centroid`` — the v2 production snap. For each arc
  position, sample a perpendicular disk; weight each voxel by ``-LoG`` above
  ``log_threshold``; shift the arc to the weighted centroid. Smooth the
  resulting polyline with a uniform filter. Recovers placements where the
  polynomial axis is 1-2 mm off the actual electrode axis.

* ``snap_centerline_owned`` — same logic, but at each arc step discards disk
  voxels closer to a neighbor's centerline than ours. Prevents drift toward
  passing shanks (the T18/X03 motivating case in the notebook). Used by the
  two-pass runner (``run_two_pass``).
"""
from __future__ import annotations

import numpy as np

from .constants import (
    SNAP_LOG_THRESHOLD,
    SNAP_RADIUS_MM,
    SNAP_SMOOTH_WINDOW,
    SNAP_STEP_MM,
)
from .polyline import min_dist_pts_to_polyline, polyline_pos_tan, polyline_segments, ortho_uv


def snap_centerline_to_centroid(
    centerline: np.ndarray, log_arr_kji, r2i,
    *, snap_radius_mm: float = SNAP_RADIUS_MM,
    step_mm: float = SNAP_STEP_MM,
    log_threshold: float = SNAP_LOG_THRESHOLD,
    n_radii: int = 4, n_angles: int = 16,
    smooth_window: int = SNAP_SMOOTH_WINDOW,
) -> np.ndarray:
    """Recenter ``centerline`` arc-by-arc on the local LoG-bright centroid.

    LoG σ=1 is a calibrated metal-bright detector — its threshold (default
    ``LOG_BLOB_THRESHOLD = 500`` per stage 1) is invariant to subject-level
    CT acquisition / windowing. Raw-HU snap admits between-contact wire
    voxels (HU 500-1000) on borderline cases (T4/RHH).
    """
    from ..volume_sampling import sample_trilinear_batch
    from scipy.ndimage import uniform_filter1d

    starts, dirs, lens, cum_start = polyline_segments(centerline)
    total_arc = float(cum_start[-1] + lens[-1])
    arcs = np.arange(0.0, total_arc + 0.5 * step_mm, step_mm)
    snapped = np.zeros((len(arcs), 3), dtype=float)

    n_per_disk = n_radii * n_angles
    off_u = np.zeros(n_per_disk, dtype=float)
    off_v = np.zeros(n_per_disk, dtype=float)
    idx = 0
    for r_i in range(1, n_radii + 1):
        rr = snap_radius_mm * r_i / n_radii
        for a_i in range(n_angles):
            ang = 2.0 * np.pi * a_i / n_angles
            off_u[idx] = rr * np.cos(ang)
            off_v[idx] = rr * np.sin(ang)
            idx += 1

    for ai, t in enumerate(arcs):
        center, tangent = polyline_pos_tan(centerline, float(t))
        u, v = ortho_uv(tangent)
        pts = (center[None, :]
               + off_u[:, None] * u[None, :]
               + off_v[:, None] * v[None, :])
        log_vals = sample_trilinear_batch(log_arr_kji, r2i, pts)
        sig = -log_vals
        valid = np.isfinite(sig) & (sig > log_threshold)
        if np.any(valid):
            w = sig[valid] - log_threshold
            mu = float((w * off_u[valid]).sum() / w.sum())
            mv = float((w * off_v[valid]).sum() / w.sum())
            snapped[ai] = center + mu * u + mv * v
        else:
            snapped[ai] = center
    if smooth_window > 1:
        snapped = uniform_filter1d(snapped, size=smooth_window, axis=0, mode="nearest")
    return snapped


def snap_centerline_owned(
    centerline, log_arr_kji, r2i,
    *, others,
    snap_radius_mm: float = 4.0,
    step_mm: float = 0.5,
    log_threshold: float = 500.0,
    n_radii: int = 4, n_angles: int = 16,
    smooth_window: int = 5,
) -> np.ndarray:
    """Ownership-aware variant of ``snap_centerline_to_centroid``.

    Same centroid-of-bright-LoG logic, but at each arc step we discard disk
    voxels that sit closer to a neighbor's centerline than ours. Defaults
    match the notebook's exploration values (radius 4 mm, n_radii 4, n_angles 16
    — wider than the production snap because cross-shank ownership requires
    enough voxels in the disk to remain after masking).
    """
    from ..volume_sampling import sample_trilinear_batch
    from scipy.ndimage import uniform_filter1d

    cl = np.asarray(centerline, dtype=float)
    diffs = np.diff(cl, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(cum[-1])
    arcs = np.arange(0.0, total + 0.5 * step_mm, step_mm)
    snapped = np.zeros((len(arcs), 3), dtype=float)

    n_per_disk = n_radii * n_angles
    off_u = np.zeros(n_per_disk, dtype=float)
    off_v = np.zeros(n_per_disk, dtype=float)
    idx = 0
    for r_i in range(1, n_radii + 1):
        rr = snap_radius_mm * r_i / n_radii
        for a_i in range(n_angles):
            ang = 2.0 * np.pi * a_i / n_angles
            off_u[idx] = rr * np.cos(ang)
            off_v[idx] = rr * np.sin(ang)
            idx += 1
    dist_self = np.sqrt(off_u ** 2 + off_v ** 2)
    others_arr = [np.asarray(o, dtype=float) for o in others if o is not None and len(o) >= 2]

    for ai, t in enumerate(arcs):
        i = int(np.searchsorted(cum, t, side="right") - 1)
        i = max(0, min(i, len(diffs) - 1))
        t_frac = (t - cum[i]) / max(seg_lens[i], 1e-9)
        center = cl[i] + t_frac * diffs[i]
        tangent = diffs[i] / max(seg_lens[i], 1e-9)
        u, v = ortho_uv(tangent)
        pts = center[None, :] + off_u[:, None] * u[None, :] + off_v[:, None] * v[None, :]

        dist_other = np.full(n_per_disk, np.inf, dtype=float)
        for ocl in others_arr:
            dist_other = np.minimum(dist_other, min_dist_pts_to_polyline(pts, ocl))
        owned = dist_self <= dist_other

        log_vals = sample_trilinear_batch(log_arr_kji, r2i, pts)
        sig = -log_vals
        valid = np.isfinite(sig) & (sig > log_threshold) & owned
        if np.any(valid):
            w = sig[valid] - log_threshold
            mu = float((w * off_u[valid]).sum() / w.sum())
            mv = float((w * off_v[valid]).sum() / w.sum())
            snapped[ai] = center + mu * u + mv * v
        else:
            snapped[ai] = center
    if smooth_window > 1:
        snapped = uniform_filter1d(snapped, size=smooth_window, axis=0, mode="nearest")
    return snapped


__all__ = ["snap_centerline_owned", "snap_centerline_to_centroid"]
