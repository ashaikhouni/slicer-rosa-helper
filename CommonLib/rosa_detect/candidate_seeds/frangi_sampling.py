"""Frangi-along-line + median-inlier-pitch helpers.

Used by v1's stage1 walker (FRANGI_LINE_MIN_MEDIAN gate) and by the
walker line refit (median pitch is the robust pitch estimate).
"""
from __future__ import annotations

import numpy as np

from rosa_core.volume_sampling import iter_axis_points, sample_nearest_at_ras

from .constants import ALONG_AXIS_STEP_MM


def frangi_along_line_stats(start_ras, end_ras, frangi_arr, ras_to_ijk_mat,
                             step_mm: float = ALONG_AXIS_STEP_MM) -> tuple[float, float]:
    """Sample Frangi σ=1 at ``step_mm`` intervals along ``[start, end]``
    and return ``(mean, median)``.

    Samples the nearest-voxel value (no interpolation). Returns
    ``(0.0, 0.0)`` when the segment is too short or both endpoints fall
    outside the volume.
    """
    L = float(np.linalg.norm(np.asarray(end_ras) - np.asarray(start_ras)))
    if L < step_mm:
        return 0.0, 0.0
    vals = np.array(
        [
            sample_nearest_at_ras(frangi_arr, ras_to_ijk_mat, p)
            for _t, p in iter_axis_points(start_ras, end_ras, step_mm)
        ],
        dtype=np.float32,
    )
    return float(vals.mean()), float(np.median(vals))


def median_inlier_pitch(pts, axis) -> float:
    """Median consecutive spacing of ``pts`` projected onto ``axis``.

    Robust to a single far-away outlier that would skew mean-based
    avg_pitch (``original_span_mm / (n-1)``). When the walker absorbs a
    spurious blob at one end, the mean pitch inflates past the
    ``looks_like_seeg`` threshold even though most of the inliers sit on
    a regular 3.5 mm chain; median collapses back to the true pitch.
    """
    pts = np.asarray(pts, dtype=float)
    axis = np.asarray(axis, dtype=float)
    if pts.shape[0] < 2:
        return float("inf")
    c = pts.mean(axis=0)
    proj = np.sort((pts - c) @ axis)
    diffs = np.diff(proj)
    if diffs.size == 0:
        return float("inf")
    return float(np.median(diffs))


__all__ = ["frangi_along_line_stats", "median_inlier_pitch"]
