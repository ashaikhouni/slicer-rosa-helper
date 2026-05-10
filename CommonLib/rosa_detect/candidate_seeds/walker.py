"""Blob-pitch walker — chains LoG blobs into candidate shank lines.

Three functions:

* :func:`walk_with_pitch_precomputed` — vectorized pitch-matching step.
* :func:`walk_line` — runs the walker for a single seed pair across 5
  pitch perturbations, picks the best, refits via PCA, validates span /
  contiguity. Returns a line dict or None.
* :func:`refit_line_from_inliers` — recompute axis / span / endpoints
  from the current inlier set after arbitration / extension.

Strategy-scoped constants (``MIN_LINE_SPAN_MM``, ``MAX_LINE_SPAN_MM``,
``MAX_INLIER_GAP_MM``) are looked up via cpfit at call time so the
``StrategyBoundsScope`` context manager keeps working without these
moved functions needing to know about its mutation. Cleanup target for
Move 7: pass bounds explicitly instead of via cpfit module mutation.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..primitives.geometry import unit
from .constants import (
    AX_TOL_MM,
    MAX_K_STEPS,
    MIN_BLOBS_PER_LINE,
    PERP_TOL_MM,
    PITCH_MM,
    PITCH_TOL_MM,
)


def walk_with_pitch_precomputed(proj, within_perp, amps, pitch, ax_tol, max_k):
    """Pitch-matching step given pre-computed per-blob axis projection
    and perp-tolerance mask (:func:`walk_line` computes these once per
    seed pair and reuses across the 5 pitch perturbations — the axis
    and perp mask don't change, only the per-k targets do).

    Vectorized: each blob's natural slot is ``k = round(proj / pitch)``.
    A blob is accepted when its perp tolerance holds, ``|k| ≤ max_k``
    and ``|proj − k·pitch| ≤ ax_tol``. For each surviving k slot, keep
    the single blob with the highest amplitude.
    """
    k_nearest = np.rint(proj / pitch).astype(np.int64)
    target = k_nearest * pitch
    ax_resid = np.abs(proj - target)
    valid = within_perp & (ax_resid <= ax_tol) & (np.abs(k_nearest) <= max_k)
    if not np.any(valid):
        return None
    idx_valid = np.where(valid)[0]
    k_valid = k_nearest[idx_valid]
    amps_valid = amps[idx_valid]
    order = np.argsort(k_valid, kind="stable")
    sorted_k = k_valid[order]
    sorted_idx = idx_valid[order]
    sorted_amps = amps_valid[order]
    change = np.where(np.diff(sorted_k) != 0)[0] + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [sorted_k.size]])
    inliers = set()
    for s, e in zip(starts, ends):
        local_best = int(np.argmax(sorted_amps[s:e]))
        inliers.add(int(sorted_idx[s + local_best]))
    return dict(inliers=inliers, pitch=pitch, n_inliers=len(inliers))


def walk_line(seed_idx, neighbor_idx, pts, amps, pitch_mm: float = PITCH_MM):
    """Seed-pair walk: chain blobs along one pitch hypothesis.

    Strategy-scoped span / gap bounds (``MIN_LINE_SPAN_MM`` /
    ``MAX_LINE_SPAN_MM`` / ``MAX_INLIER_GAP_MM``) are looked up via
    cpfit at call time so this function respects ``StrategyBoundsScope``.
    """
    from .. import contact_pitch_v1_fit as _cpfit
    MIN_LINE_SPAN_MM = _cpfit.MIN_LINE_SPAN_MM
    MAX_LINE_SPAN_MM = _cpfit.MAX_LINE_SPAN_MM
    MAX_INLIER_GAP_MM = _cpfit.MAX_INLIER_GAP_MM

    p0 = pts[seed_idx]
    p1 = pts[neighbor_idx]
    seed_d = float(np.linalg.norm(p1 - p0))
    k_seed = max(1, int(round(seed_d / pitch_mm)))
    pitch_seed = seed_d / k_seed
    if not (pitch_mm - PITCH_TOL_MM <= pitch_seed <= pitch_mm + PITCH_TOL_MM):
        return None
    axis = (p1 - p0) / seed_d
    diffs = pts - p0
    proj = diffs @ axis
    d2 = np.einsum("ij,ij->i", diffs, diffs)
    perp_sq = d2 - proj * proj
    within_perp = perp_sq <= (PERP_TOL_MM * PERP_TOL_MM)
    best = None
    for dp in (-0.2, -0.1, 0.0, 0.1, 0.2):
        pitch_try = pitch_seed + dp
        if not (pitch_mm - PITCH_TOL_MM <= pitch_try <= pitch_mm + PITCH_TOL_MM):
            continue
        r = walk_with_pitch_precomputed(
            proj, within_perp, amps, pitch_try, AX_TOL_MM, MAX_K_STEPS,
        )
        if r is None:
            continue
        if best is None or r["n_inliers"] > best["n_inliers"]:
            best = r
    if best is None:
        return None
    inliers = list(best["inliers"])
    if len(inliers) < MIN_BLOBS_PER_LINE:
        return None
    inlier_pts = pts[inliers]
    c = inlier_pts.mean(axis=0)
    X = inlier_pts - c
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    axis_ref = unit(Vt[0])
    proj_ref = X @ axis_ref

    sort_idx = np.argsort(proj_ref)
    sorted_proj = proj_ref[sort_idx].tolist()
    sorted_inliers = [inliers[i] for i in sort_idx]
    while len(sorted_inliers) > MIN_BLOBS_PER_LINE:
        front_gap = sorted_proj[1] - sorted_proj[0]
        back_gap = sorted_proj[-1] - sorted_proj[-2]
        if max(front_gap, back_gap) <= MAX_INLIER_GAP_MM:
            break
        if front_gap >= back_gap:
            sorted_proj.pop(0)
            sorted_inliers.pop(0)
        else:
            sorted_proj.pop()
            sorted_inliers.pop()
    if len(sorted_inliers) < MIN_BLOBS_PER_LINE:
        return None
    gaps_after = np.diff(sorted_proj)
    if gaps_after.size > 0 and float(gaps_after.max()) > MAX_INLIER_GAP_MM:
        return None
    inliers = sorted_inliers
    inlier_pts = pts[inliers]
    c = inlier_pts.mean(axis=0)
    X = inlier_pts - c
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    axis_ref = unit(Vt[0])
    proj_ref = X @ axis_ref
    lo, hi = float(proj_ref.min()), float(proj_ref.max())
    span = hi - lo
    if span < MIN_LINE_SPAN_MM or span > MAX_LINE_SPAN_MM:
        return None
    return dict(
        axis=axis_ref, center=c, inlier_idx=inliers,
        span_mm=span, span_lo=lo, span_hi=hi,
        start_ras=c + lo * axis_ref, end_ras=c + hi * axis_ref,
        n_blobs=len(inliers),
        amp_sum=float(np.sum([amps[i] for i in inliers])),
    )


def refit_line_from_inliers(line, pts_c, amps_c, min_blobs: int | None = None):
    """Recompute axis / span / endpoints / amp_sum from the current
    ``inlier_idx`` list. Mutates ``line`` and returns it; returns None
    if the line has too few inliers or too short a span after refit.

    ``min_blobs`` defaults to cpfit's current ``MIN_BLOBS_PER_LINE``
    (resolved at call time so :class:`StrategyBoundsScope` swaps are
    respected).
    """
    from .. import contact_pitch_v1_fit as _cpfit
    MIN_LINE_SPAN_MM = _cpfit.MIN_LINE_SPAN_MM
    MAX_LINE_SPAN_MM = _cpfit.MAX_LINE_SPAN_MM

    if min_blobs is None:
        min_blobs = _cpfit.MIN_BLOBS_PER_LINE

    inliers = list(line["inlier_idx"])
    if len(inliers) < min_blobs:
        return None
    inlier_pts = pts_c[inliers]
    c = inlier_pts.mean(axis=0)
    X = inlier_pts - c
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    axis_ref = unit(Vt[0])
    proj_ref = X @ axis_ref
    lo = float(proj_ref.min())
    hi = float(proj_ref.max())
    span = hi - lo
    if span < MIN_LINE_SPAN_MM or span > MAX_LINE_SPAN_MM:
        return None
    line["axis"] = axis_ref
    line["center"] = c
    line["span_lo"] = lo
    line["span_hi"] = hi
    line["span_mm"] = span
    line["start_ras"] = c + lo * axis_ref
    line["end_ras"] = c + hi * axis_ref
    line["n_blobs"] = len(inliers)
    line["inlier_idx"] = inliers
    line["amp_sum"] = float(np.sum(amps_c[inliers]))
    return line


__all__ = [
    "walk_with_pitch_precomputed",
    "walk_line",
    "refit_line_from_inliers",
]
