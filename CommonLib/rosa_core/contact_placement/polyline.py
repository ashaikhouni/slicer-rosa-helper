"""Polyline geometry helpers used across the staged placement stages.

Lifted verbatim (logic-preserving) from ``contact_placement_v2.py`` plus the
notebook's added arc/perp variants. All functions are stateless and operate
on ``np.ndarray`` polylines of shape ``(K, 3)``; they don't depend on
SimpleITK or features.
"""
from __future__ import annotations

import numpy as np


def polyline_segments(polyline: np.ndarray):
    """Decompose a polyline into per-segment (start, unit_dir, length, cum_start).

    Returns 4-tuple of arrays. Filters out zero-length segments. Raises
    ``ValueError`` for non-(K, 3) inputs or all-zero polylines.
    """
    P = np.asarray(polyline, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] < 2:
        raise ValueError("polyline must be (K,3) with K>=2")
    diffs = np.diff(P, axis=0)
    lens = np.linalg.norm(diffs, axis=1)
    keep = lens > 1e-9
    if not keep.any():
        raise ValueError("polyline has zero arc length")
    starts = P[:-1][keep]
    diffs = diffs[keep]
    lens = lens[keep]
    dirs = diffs / lens[:, None]
    cum_start = np.concatenate([[0.0], np.cumsum(lens[:-1])])
    return starts, dirs, lens, cum_start


def polyline_pos_at_arc(polyline: np.ndarray, arc_mm: float) -> np.ndarray:
    """Position on the polyline at the given arc length (clamped to ends)."""
    starts, dirs, lens, cum_start = polyline_segments(polyline)
    total = float(cum_start[-1] + lens[-1])
    if arc_mm <= 0.0:
        return starts[0].copy()
    if arc_mm >= total:
        return starts[-1] + lens[-1] * dirs[-1]
    i = int(np.searchsorted(cum_start + lens, arc_mm, side="right"))
    i = min(i, len(starts) - 1)
    t = arc_mm - cum_start[i]
    return starts[i] + t * dirs[i]


def polyline_pos_tan(polyline: np.ndarray, arc_mm: float):
    """Position + unit tangent on the polyline at the given arc."""
    starts, dirs, lens, cum_start = polyline_segments(polyline)
    total = float(cum_start[-1] + lens[-1])
    if arc_mm <= 0.0:
        return starts[0].copy(), dirs[0]
    if arc_mm >= total:
        return starts[-1] + lens[-1] * dirs[-1], dirs[-1]
    i = int(np.searchsorted(cum_start + lens, arc_mm, side="right"))
    i = min(i, len(starts) - 1)
    t = arc_mm - cum_start[i]
    return starts[i] + t * dirs[i], dirs[i]


def polyline_at_arc(polyline: np.ndarray, arc: float) -> np.ndarray:
    """Vectorized wrapper used by ``stage_e_place``: interpolate a single arc.

    Equivalent to ``polyline_pos_at_arc`` but takes arrays of (K, 3) directly
    without computing segment metadata twice — kept separate to match the
    notebook's call signature exactly.
    """
    diffs = np.diff(polyline, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    arc = float(np.clip(arc, 0.0, cum[-1]))
    i = int(np.searchsorted(cum, arc, side="right") - 1)
    i = min(max(i, 0), len(diffs) - 1)
    t = (arc - cum[i]) / max(seg_lens[i], 1e-9)
    return polyline[i] + t * diffs[i]


def project_to_polyline_arc(polyline: np.ndarray, point_ras: np.ndarray) -> float:
    """Arc-length of the closest polyline point to ``point_ras``.

    Per-segment closed-form projection, then min across segments. Used by
    ``stage_a_anchor`` to convert a seed-line bolt position to an arc on the
    refined centerline, and by ``score_cc_overlap`` to project bolt CCs.
    """
    starts, dirs, lens, cum_start = polyline_segments(polyline)
    pt = np.asarray(point_ras, dtype=float)
    best_d = np.inf
    best_arc = 0.0
    for i in range(len(starts)):
        a = starts[i]; L = lens[i]; u = dirs[i]
        t = float(np.clip((pt - a) @ u, 0.0, L))
        proj = a + t * u
        d = float(np.linalg.norm(pt - proj))
        if d < best_d:
            best_d = d
            best_arc = float(cum_start[i] + t)
    return best_arc


def project_to_polyline(pt: np.ndarray, polyline: np.ndarray) -> tuple[float, float]:
    """Return ``(arc_along_polyline, perp_distance)`` for the closest point.

    Vectorized over polyline segments — faster than ``project_to_polyline_arc``
    when both arc and perp are needed (used by ``score_cc_overlap``).
    """
    p = polyline.astype(float)
    if len(p) < 2:
        return 0.0, float(np.linalg.norm(pt - p[0])) if len(p) else float("inf")
    a = p[:-1]
    b = p[1:]
    ab = b - a
    ab_len2 = np.einsum("ij,ij->i", ab, ab)
    ap = pt - a
    t = np.einsum("ij,ij->i", ap, ab) / np.maximum(ab_len2, 1e-12)
    t = np.clip(t, 0.0, 1.0)
    closest = a + t[:, None] * ab
    diffs = pt - closest
    dists = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))
    seg_idx = int(np.argmin(dists))
    perp = float(dists[seg_idx])
    seg_lens = np.sqrt(ab_len2)
    cum_start = np.concatenate([[0.0], np.cumsum(seg_lens[:-1])])
    arc = float(cum_start[seg_idx] + t[seg_idx] * seg_lens[seg_idx])
    return arc, perp


def straight_centerline(start: np.ndarray, end: np.ndarray, n_points: int = 64) -> np.ndarray:
    """Discretize a straight start→end segment into ``n_points``."""
    s = np.asarray(start, dtype=float)
    e = np.asarray(end, dtype=float)
    ts = np.linspace(0.0, 1.0, n_points)
    return np.array([s + t * (e - s) for t in ts])


def extend_centerline_tail(centerline: np.ndarray, extra_mm: float) -> np.ndarray:
    """Extend the centerline past its deep endpoint by ``extra_mm``.

    Lets the walker sample signal slightly past the auto-fit axis tip — the
    matched filter can then evaluate model tip positions that fall just past
    the polynomial endpoint. Returns the input unchanged if ``extra_mm <= 0``.
    """
    if extra_mm <= 1e-6:
        return centerline
    cl = np.asarray(centerline, dtype=float)
    tail_dir = cl[-1] - cl[-2]
    tail_len = float(np.linalg.norm(tail_dir))
    if tail_len < 1e-9:
        return cl
    tail_unit = tail_dir / tail_len
    new_tip = cl[-1] + tail_unit * float(extra_mm)
    return np.vstack([cl, new_tip[None, :]])


def min_dist_pts_to_polyline(pts: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Min distance from each point in ``pts`` ``(n, 3)`` to any polyline segment.

    Vectorized closed-form. Returns ``(n,)`` array of distances; ``inf`` when
    the polyline has < 2 points. (General polyline primitive — was the
    cross-shank ownership mask's distance test before that path was retired.)
    """
    if len(polyline) < 2:
        return np.full(pts.shape[0], np.inf, dtype=float)
    p = polyline.astype(float)
    a = p[:-1]                        # (m-1, 3) segment starts
    b = p[1:]                         # (m-1, 3) segment ends
    ab = b - a
    ab_len2 = np.einsum("ij,ij->i", ab, ab)
    ap = pts[:, None, :] - a[None, :, :]
    t = np.einsum("nmi,mi->nm", ap, ab) / np.maximum(ab_len2[None, :], 1e-12)
    t = np.clip(t, 0.0, 1.0)
    closest = a[None, :, :] + t[..., None] * ab[None, :, :]
    diffs = pts[:, None, :] - closest
    dists = np.sqrt(np.einsum("nmi,nmi->nm", diffs, diffs))
    return dists.min(axis=1)


def ortho_uv(tangent: np.ndarray):
    """Two unit vectors perpendicular to ``tangent``.

    Uses the canonical "least-aligned axis" trick to avoid degeneracy when
    ``tangent`` is near-parallel to the world x-axis. Used by the walker
    and snap modules to build per-arc disk sampling bases.
    """
    any_v = np.array([1.0, 0.0, 0.0]) if abs(tangent[0]) <= 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(tangent, any_v); u /= np.linalg.norm(u)
    v = np.cross(tangent, u);     v /= np.linalg.norm(v)
    return u, v


__all__ = [
    "extend_centerline_tail",
    "min_dist_pts_to_polyline",
    "ortho_uv",
    "polyline_at_arc",
    "polyline_pos_at_arc",
    "polyline_pos_tan",
    "polyline_segments",
    "project_to_polyline",
    "project_to_polyline_arc",
    "straight_centerline",
]
