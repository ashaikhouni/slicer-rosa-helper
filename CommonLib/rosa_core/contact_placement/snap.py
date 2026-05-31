"""Centerline snap to LoG-bright centroids (Stage B refinement).

``snap_centerline_to_centroid`` — the v2 production snap. For each arc
position, sample a perpendicular disk; weight each voxel by ``-LoG`` above
``log_threshold``; shift the arc to the weighted centroid. Smooth the resulting
polyline with a uniform filter. Recovers placements where the polynomial axis is
1-2 mm off the actual electrode axis.

(The cross-shank ownership variants — ``snap_centerline_owned`` /
``compute_voxel_ownership`` / ``snap_centerline_voxel_owned`` — were the
``run_two_pass`` cross-shank mechanism; both the two_pass engine and these were
retired with the placement consolidation, the snap-flow's
``arbitrate_shared_peaks`` having subsumed them.)
"""
from __future__ import annotations

import numpy as np

from .constants import (
    SNAP_LOG_THRESHOLD,
    SNAP_RADIUS_MM,
    SNAP_SMOOTH_WINDOW,
    SNAP_STEP_MM,
)
from .polyline import polyline_pos_tan, polyline_segments, ortho_uv


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


__all__ = [
    "snap_centerline_to_centroid",
]
