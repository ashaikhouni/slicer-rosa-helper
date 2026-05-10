"""Pure geometry helpers for the candidate-seed pipeline.

Lifted verbatim from ``contact_pitch_v1_fit.py`` (Session 4 Phase B
stage extraction). Public-named (no leading underscore); cpfit
re-exports under the legacy ``_*`` names for back-compat with probes
and tests.
"""
from __future__ import annotations

import numpy as np

from rosa_core.volume_sampling import sample_nearest_at_ras


def unit(v) -> np.ndarray:
    """Unit vector with safe fallback for zero / near-zero inputs."""
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def sample_dist_at_ras(dist_arr, ras_to_ijk_mat, ras_xyz):
    """Look up head_distance at a RAS point (nearest voxel)."""
    return sample_nearest_at_ras(dist_arr, ras_to_ijk_mat, ras_xyz)


def orient_shallow_to_deep(start_ras, end_ras, dist_arr, ras_to_ijk_mat):
    """Return ``(shallow_ras, deep_ras)`` so the shallower end (smaller
    head_distance = closer to hull surface) comes first.

    Disambiguates PCA axis direction so downstream visualization can
    color shallow vs deep consistently.
    """
    d_start = sample_dist_at_ras(dist_arr, ras_to_ijk_mat, start_ras)
    d_end = sample_dist_at_ras(dist_arr, ras_to_ijk_mat, end_ras)
    if d_start <= d_end:
        return np.asarray(start_ras, dtype=float), np.asarray(end_ras, dtype=float)
    return np.asarray(end_ras, dtype=float), np.asarray(start_ras, dtype=float)


def min_perp_to_other_segments(p, segs, skip_idx) -> float:
    """Minimum perpendicular distance from point ``p`` to any other
    segment in ``segs`` (skipping the one at ``skip_idx``).

    Uses segment-to-point distance (clamped along-projection), not
    infinite line — so crossing shanks compare only where they
    actually live.
    """
    best = float("inf")
    for i, seg in enumerate(segs):
        if i == skip_idx:
            continue
        v = p - seg["s"]
        along = float(v @ seg["a"])
        along_c = max(0.0, min(seg["L"], along))
        proj = seg["s"] + along_c * seg["a"]
        d = float(np.linalg.norm(p - proj))
        if d < best:
            best = d
    return best


def kji_to_ras_fn_from_matrix(ijk_to_ras_mat):
    """Build a closure converting voxel KJI coords to RAS via the
    supplied IJK→RAS 4×4. Accepts a single (3,) point or a (N, 3) batch.
    """
    m = np.asarray(ijk_to_ras_mat, dtype=float)

    def _fn(kji):
        if kji.ndim == 1:
            i, j, k = float(kji[2]), float(kji[1]), float(kji[0])
            return (m @ np.array([i, j, k, 1.0]))[:3]
        ijk = np.stack([kji[:, 2], kji[:, 1], kji[:, 0]], axis=1)
        h = np.concatenate([ijk, np.ones((ijk.shape[0], 1))], axis=1)
        return (m @ h.T).T[:, :3]
    return _fn


__all__ = [
    "unit",
    "sample_dist_at_ras",
    "orient_shallow_to_deep",
    "min_perp_to_other_segments",
    "kji_to_ras_fn_from_matrix",
]
