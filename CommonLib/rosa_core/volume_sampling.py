"""Shared RAS-to-voxel sampling primitives.

Auto Fit, Guided Fit, and peak-driven contact placement each used to
carry their own copies of "transform a RAS point through a 4x4
ras-to-ijk matrix and read the array at the result." Those copies
diverged on interpolation (nearest vs trilinear) and on bounds-clipping
behavior, but the matrix math was identical.

Helpers here preserve the exact arithmetic of the original sites so
swapping callers over does not change CLI/Slicer parity. The two
relevant volume conventions are kept explicit:

- IJK matrix (4x4) maps RAS -> (i, j, k) in physical-axis order.
- ``arr_kji`` is the numpy view of the same volume, indexed [k, j, i].
"""

from __future__ import annotations

import numpy as np


def ras_to_ijk_pt(ras_to_ijk_mat, ras_xyz):
    """Transform a single RAS point through a 4x4 ras-to-ijk matrix.

    Returns ``(i, j, k)`` floats — physical-axis order — matching the
    original ``(ras_to_ijk_mat @ h)[:3]`` slice used at every site.
    """
    h = np.array(
        [float(ras_xyz[0]), float(ras_xyz[1]), float(ras_xyz[2]), 1.0]
    )
    ijk = (ras_to_ijk_mat @ h)[:3]
    return float(ijk[0]), float(ijk[1]), float(ijk[2])


def clip_to_voxel(shape_kji, i, j, k):
    """Round and clip ``(i, j, k)`` floats to the in-bounds nearest voxel.

    Returns ``(k_int, j_int, i_int)`` so callers can index ``arr_kji``
    directly: ``arr_kji[kc, jc, ic]``.
    """
    K, J, I = shape_kji
    ic = int(np.clip(round(i), 0, I - 1))
    jc = int(np.clip(round(j), 0, J - 1))
    kc = int(np.clip(round(k), 0, K - 1))
    return kc, jc, ic


def sample_nearest_at_ras(arr_kji, ras_to_ijk_mat, ras_xyz):
    """Nearest-voxel array lookup at a RAS point (clipped to bounds)."""
    i, j, k = ras_to_ijk_pt(ras_to_ijk_mat, ras_xyz)
    kc, jc, ic = clip_to_voxel(arr_kji.shape, i, j, k)
    return float(arr_kji[kc, jc, ic])


def sample_trilinear_at_ras(arr_kji, ras_to_ijk_mat, ras_xyz):
    """Trilinear array lookup at a RAS point.

    Returns NaN if the RAS point lies outside the valid trilinear range
    (one voxel margin on each side). Matches the prior
    ``contact_peak_fit._sample_trilinear`` semantics exactly.
    """
    i, j, k = ras_to_ijk_pt(ras_to_ijk_mat, ras_xyz)
    s = arr_kji.shape
    if not (0 <= k < s[0] - 1 and 0 <= j < s[1] - 1 and 0 <= i < s[2] - 1):
        return float("nan")
    k0, j0, i0 = int(k), int(j), int(i)
    dk, dj, di = k - k0, j - j0, i - i0
    v000 = arr_kji[k0, j0, i0];     v001 = arr_kji[k0, j0, i0 + 1]
    v010 = arr_kji[k0, j0 + 1, i0]; v011 = arr_kji[k0, j0 + 1, i0 + 1]
    v100 = arr_kji[k0 + 1, j0, i0]; v101 = arr_kji[k0 + 1, j0, i0 + 1]
    v110 = arr_kji[k0 + 1, j0 + 1, i0]
    v111 = arr_kji[k0 + 1, j0 + 1, i0 + 1]
    c00 = v000 * (1 - di) + v001 * di
    c01 = v010 * (1 - di) + v011 * di
    c10 = v100 * (1 - di) + v101 * di
    c11 = v110 * (1 - di) + v111 * di
    c0 = c00 * (1 - dj) + c01 * dj
    c1 = c10 * (1 - dj) + c11 * dj
    return float(c0 * (1 - dk) + c1 * dk)


def sample_trilinear_batch(arr_kji, ras_to_ijk_mat, ras_points):
    """Vectorized trilinear lookup for a batch of RAS points.

    Args:
        arr_kji: numpy array indexed [k, j, i].
        ras_to_ijk_mat: 4x4 RAS→IJK matrix (i, j, k order in output).
        ras_points: (N, 3) numpy array of RAS coordinates.

    Returns:
        (N,) numpy array of float values; NaN where the point is outside
        the valid trilinear range (one voxel margin), matching the
        single-point ``sample_trilinear_at_ras`` semantics exactly.

    Replaces Python loops over ``sample_trilinear_at_ras`` calls. ~10×
    faster on disk-sampling hot paths (`sample_disk_along_polyline`,
    `_snap_centerline_to_centroid`, `refine_axis_via_centroid`).
    """
    pts = np.asarray(ras_points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    N = pts.shape[0]
    if N == 0:
        return np.zeros(0, dtype=float)
    # RAS → IJK as one matmul.
    homog = np.empty((N, 4), dtype=float)
    homog[:, :3] = pts
    homog[:, 3] = 1.0
    ijk = homog @ ras_to_ijk_mat.T  # (N, 4); take :3
    i = ijk[:, 0]; j = ijk[:, 1]; k = ijk[:, 2]
    s = arr_kji.shape
    K, J, I = s[0], s[1], s[2]
    in_bounds = (
        (k >= 0) & (k < K - 1)
        & (j >= 0) & (j < J - 1)
        & (i >= 0) & (i < I - 1)
    )
    out = np.full(N, np.nan, dtype=float)
    if not np.any(in_bounds):
        return out
    # Floor-to-int corner indices and fractional parts (only for in-bounds).
    ib = np.where(in_bounds)[0]
    ki = k[ib]; ji = j[ib]; ii = i[ib]
    k0 = ki.astype(np.int64); j0 = ji.astype(np.int64); i0 = ii.astype(np.int64)
    # Match the single-point function's float32-when-array-is-float32
    # promotion: numpy scalar*python-float keeps scalar dtype, but
    # array*python-float follows weak typing too — except when the
    # weight comes from a float64 numpy array (our `di` here). Cast
    # weights to the array's dtype so the trilinear math stays in the
    # volume's precision exactly as the scalar version does.
    arr_dtype = arr_kji.dtype
    dk = (ki - k0).astype(arr_dtype, copy=False)
    dj = (ji - j0).astype(arr_dtype, copy=False)
    di = (ii - i0).astype(arr_dtype, copy=False)
    # Gather the 8 corner values via fancy indexing.
    v000 = arr_kji[k0,     j0,     i0]
    v001 = arr_kji[k0,     j0,     i0 + 1]
    v010 = arr_kji[k0,     j0 + 1, i0]
    v011 = arr_kji[k0,     j0 + 1, i0 + 1]
    v100 = arr_kji[k0 + 1, j0,     i0]
    v101 = arr_kji[k0 + 1, j0,     i0 + 1]
    v110 = arr_kji[k0 + 1, j0 + 1, i0]
    v111 = arr_kji[k0 + 1, j0 + 1, i0 + 1]
    one = arr_dtype.type(1.0)
    one_di = one - di
    c00 = v000 * one_di + v001 * di
    c01 = v010 * one_di + v011 * di
    c10 = v100 * one_di + v101 * di
    c11 = v110 * one_di + v111 * di
    one_dj = one - dj
    c0 = c00 * one_dj + c01 * dj
    c1 = c10 * one_dj + c11 * dj
    out[ib] = c0 * (one - dk) + c1 * dk
    return out


def iter_axis_points(start_ras, end_ras, step_mm):
    """Yield ``(t_mm, point_ras)`` evenly along [start_ras, end_ras].

    Sampling matches the prior inline loops in ``contact_pitch_v1_fit``:

    - ``n = max(2, int(L / step_mm) + 1)`` points.
    - ``t = idx * step_mm`` for ``idx < n - 1``, then ``t = L`` for the
      last point. So both endpoints are always sampled.
    - Caller is responsible for handling ``L < step_mm`` before
      iterating; this helper still yields two points (start, end) but
      with ``t == 0`` and ``t == L`` so callers that want to bail can
      check ``L`` themselves.
    """
    s = np.asarray(start_ras, dtype=float)
    e = np.asarray(end_ras, dtype=float)
    d = e - s
    L = float(np.linalg.norm(d))
    if L <= 0.0:
        yield 0.0, s
        return
    u = d / L
    n = max(2, int(L / step_mm) + 1)
    for idx in range(n):
        t = idx * step_mm if idx < n - 1 else L
        yield t, s + t * u
