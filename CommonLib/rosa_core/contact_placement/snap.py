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
    OWN_CONTEST_MARGIN_MM,
    OWN_MIN_VOXELS_PER_ARC,
    OWN_R_MM,
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


def compute_voxel_ownership(
    centerlines, log_arr_kji, ras_to_ijk_mat, *,
    r_own_mm: float = OWN_R_MM,
    contest_margin_mm: float = OWN_CONTEST_MARGIN_MM,
    log_threshold: float = SNAP_LOG_THRESHOLD,
):
    """Per-voxel closest-seed classification on the LoG-bright voxels.

    For every voxel with ``-LoG > log_threshold``, find its two closest
    seed-line perp distances. Return a kji-shape ``int16`` label volume:

    * ``0``  — background (no bright voxel here, or closest seed > ``r_own_mm``)
    * ``-1`` — contested (the two closest seeds are within ``contest_margin_mm``)
    * ``i+1`` — uniquely owned by seed index ``i`` (``i`` is the position in
      ``centerlines``).

    The output is consumed by ``snap_centerline_voxel_owned`` (and Stage C /
    Stage D, when soft ownership propagates downstream).
    """
    log_arr_kji = np.asarray(log_arr_kji)
    own = np.zeros(log_arr_kji.shape, dtype=np.int16)
    bright = (-log_arr_kji) > log_threshold
    n_bright = int(bright.sum())
    if n_bright == 0 or len(centerlines) == 0:
        return own
    k_idx, j_idx, i_idx = np.where(bright)
    ones = np.ones(n_bright, dtype=np.float32)
    ijk_h = np.stack([i_idx, j_idx, k_idx, ones], axis=1).astype(np.float32)
    # ras_to_ijk_mat is RAS->IJK; invert for IJK->RAS.
    i2r = np.linalg.inv(np.asarray(ras_to_ijk_mat, dtype=float))
    bright_ras = (i2r @ ijk_h.T).T[:, :3]
    dists = np.full((n_bright, len(centerlines)), np.inf, dtype=np.float32)
    for ci, cl in enumerate(centerlines):
        cl = np.asarray(cl, dtype=float)
        if len(cl) < 2:
            continue
        dists[:, ci] = min_dist_pts_to_polyline(bright_ras, cl).astype(np.float32)
    # Two smallest distances per voxel — partial sort is faster than full sort.
    if dists.shape[1] >= 2:
        part = np.argpartition(dists, 1, axis=1)[:, :2]
        d_part = np.take_along_axis(dists, part, axis=1)
        order = np.argsort(d_part, axis=1)
        s1_idx = np.take_along_axis(part, order, axis=1)[:, 0]
        s2_idx = np.take_along_axis(part, order, axis=1)[:, 1]
        s1_d = np.take_along_axis(d_part, order, axis=1)[:, 0]
        s2_d = np.take_along_axis(d_part, order, axis=1)[:, 1]
    else:
        s1_idx = np.zeros(n_bright, dtype=np.int64)
        s1_d = dists[:, 0]
        s2_idx = s1_idx
        s2_d = np.full_like(s1_d, np.inf)
    in_reach = s1_d <= r_own_mm
    contested = in_reach & (s2_d <= r_own_mm) & ((s2_d - s1_d) < contest_margin_mm)
    unique = in_reach & ~contested
    labels = np.zeros(n_bright, dtype=np.int16)
    labels[unique] = s1_idx[unique].astype(np.int16) + 1
    labels[contested] = -1
    own[k_idx, j_idx, i_idx] = labels
    return own


def snap_centerline_voxel_owned(
    centerline, log_arr_kji, r2i,
    *, voxel_ownership, seed_idx,
    snap_radius_mm: float = SNAP_RADIUS_MM,
    step_mm: float = SNAP_STEP_MM,
    log_threshold: float = SNAP_LOG_THRESHOLD,
    n_radii: int = 4, n_angles: int = 16,
    smooth_window: int = SNAP_SMOOTH_WINDOW,
    min_owned_voxels: int = OWN_MIN_VOXELS_PER_ARC,
) -> np.ndarray:
    """LoG-centroid snap that only sums voxels uniquely owned by ``seed_idx``.

    Mirrors ``snap_centerline_to_centroid`` arc-by-arc, but at each step
    masks the disk samples to the voxels whose ``voxel_ownership`` label
    is ``seed_idx + 1`` (uniquely owned). If fewer than ``min_owned_voxels``
    survive, that arc is marked invalid and the position is linearly
    interpolated from the neighboring valid arcs after the per-arc loop.

    This is the production form of ``snap_with_ownership`` from
    ``notebooks/s54_blob_ownership_prototype.ipynb``.
    """
    from ..volume_sampling import sample_trilinear_batch
    from scipy.ndimage import uniform_filter1d

    voxel_ownership = np.asarray(voxel_ownership)
    target_label = int(seed_idx) + 1
    starts, dirs, lens, cum_start = polyline_segments(centerline)
    total_arc = float(cum_start[-1] + lens[-1])
    arcs = np.arange(0.0, total_arc + 0.5 * step_mm, step_mm)
    snapped = np.zeros((len(arcs), 3), dtype=float)
    valid_arc = np.zeros(len(arcs), dtype=bool)

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

    kshape, jshape, ishape = voxel_ownership.shape
    r2i_mat = np.asarray(r2i, dtype=float)

    for ai, t in enumerate(arcs):
        center, tangent = polyline_pos_tan(centerline, float(t))
        u, v = ortho_uv(tangent)
        pts = (center[None, :]
               + off_u[:, None] * u[None, :]
               + off_v[:, None] * v[None, :])
        # RAS -> IJK lookup for the ownership label.
        ones = np.ones((n_per_disk, 1), dtype=float)
        ijk = (r2i_mat @ np.hstack([pts, ones]).T).T[:, :3]
        ijk_round = np.round(ijk).astype(int)
        i_v = ijk_round[:, 0]; j_v = ijk_round[:, 1]; k_v = ijk_round[:, 2]
        in_bounds = (
            (k_v >= 0) & (k_v < kshape) &
            (j_v >= 0) & (j_v < jshape) &
            (i_v >= 0) & (i_v < ishape)
        )
        own_at_sample = np.zeros(n_per_disk, dtype=np.int16)
        if in_bounds.any():
            own_at_sample[in_bounds] = voxel_ownership[
                k_v[in_bounds], j_v[in_bounds], i_v[in_bounds]
            ]
        log_vals = sample_trilinear_batch(log_arr_kji, r2i_mat, pts)
        sig = -log_vals
        usable = (
            np.isfinite(sig) & (sig > log_threshold) & (own_at_sample == target_label)
        )
        if usable.sum() >= min_owned_voxels:
            w = sig[usable] - log_threshold
            mu = float((w * off_u[usable]).sum() / w.sum())
            mv = float((w * off_v[usable]).sum() / w.sum())
            snapped[ai] = center + mu * u + mv * v
            valid_arc[ai] = True
        else:
            snapped[ai] = center
    # Linearly interpolate invalid arcs from neighboring valid arcs along
    # the centerline. Preserves curvature in trusted segments and fills
    # contested gaps with a smooth bridge.
    if valid_arc.any() and (~valid_arc).any():
        good = np.where(valid_arc)[0]
        for axis in range(3):
            snapped[:, axis] = np.interp(
                np.arange(len(arcs)), good, snapped[good, axis]
            )
    elif not valid_arc.any():
        # No trusted arcs — fall through to the seed centerline. Sampling
        # ``polyline_pos_tan`` over the full ``arcs`` grid reproduces it.
        for ai, t in enumerate(arcs):
            snapped[ai] = polyline_pos_tan(centerline, float(t))[0]
    if smooth_window > 1:
        snapped = uniform_filter1d(snapped, size=smooth_window, axis=0, mode="nearest")
    return snapped


__all__ = [
    "compute_voxel_ownership",
    "snap_centerline_owned",
    "snap_centerline_to_centroid",
    "snap_centerline_voxel_owned",
]
