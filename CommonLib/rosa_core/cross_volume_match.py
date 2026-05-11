"""Cross-volume trajectory matching from line geometry alone.

Use case: same patient, two coordinate frames. A ``.ros`` file carries
surgeon-planned trajectories in the ROSA reference frame; a separate CT
(post-registered to a different MRI atlas, say) carries the same physical
electrodes but in a different RAS. We want to map detector emissions on
the second CT back to the planned names — *without* requiring the ROSA
reference volume on disk and *without* image-to-image registration.

The trajectory line set IS the registration. Same patient → same bolts
on the skull → same line bundle, modulo a rigid transform.

Why we score on lines, not endpoints
------------------------------------
The ``.ros`` file's start/end are the surgeon's planned entry intent +
target intent. The detector's start/end are the bolt-CC anchor + the
deepest visible contact. For the SAME physical electrode these can drift
several mm at each end. What is stable across both sources:

* the unit axis direction (modulo a global sign flip per electrode),
* the infinite line the segment lies on,
* the perpendicular distance between two infinite lines.

So the RANSAC primitive is line geometry, not point sets:

* Step A — solve R via orthogonal Procrustes on direction-vector
  triplets, with a per-axis sign sweep (``±d`` per electrode handles the
  entry→target vs target→entry ambiguity).
* Step B — solve t from midpoint centroids of the inlier triplet:
  ``t = mean(ros_mid) − R @ mean(det_mid)``.
* Step C — score every (det, ros) pair under (R, t) by axis angle AND
  perpendicular line-to-line distance. Both are infinite-line
  properties — endpoint drift doesn't enter.

Validated on s57.ros + T24 CT (same patient, two registrations): 16/16
det lines inlier in best hypothesis; 16/17 ROS named by greedy matcher;
all matched pairs ≤ 11° axis-angle and ≤ 6 mm perp.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations, product
from typing import Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class TrajectoryLine:
    """Infinite-line representation of a trajectory: midpoint + unit axis.

    The sign of ``direction`` is arbitrary — the matcher handles per-line
    flips internally so callers don't have to standardize entry→target
    vs target→entry conventions.
    """

    name: str
    mid: np.ndarray   # (3,) float
    direction: np.ndarray  # (3,) unit float


@dataclass
class CrossVolumeMatch:
    """Result of :func:`cross_volume_match`.

    ``transform_4x4`` maps a point from the *detector* frame to the *ROS*
    frame: ``p_ros = transform @ [p_det; 1]``. Identity if matching
    failed (``ransac_inliers < 3``).

    ``pairs`` is one entry per ROS trajectory: ``(ros_name, det_name,
    angle_deg, perp_mm)``. ``det_name`` is ``None`` for ROS lines that
    found no detector counterpart within the match tolerances.
    """

    transform_4x4: np.ndarray
    ransac_inliers: int
    refined_inliers: int
    pairs: list[tuple[str, str | None, float | None, float | None]]
    diagnostics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------
# Pure-geometry helpers (importable for tests / probes)
# ---------------------------------------------------------------------


def trajectories_to_lines(
    trajs: Iterable[dict | "TrajectoryLine"],
) -> list[TrajectoryLine]:
    """Coerce a heterogeneous trajectory list to ``list[TrajectoryLine]``.

    Accepts either ``TrajectoryLine`` (returned unchanged) or dicts with
    ``name`` + ``start`` (or ``start_ras``) + ``end`` (or ``end_ras``).
    Drops zero-length entries silently.
    """
    out: list[TrajectoryLine] = []
    for t in trajs:
        if isinstance(t, TrajectoryLine):
            out.append(t)
            continue
        s = np.asarray(t.get("start", t.get("start_ras")), dtype=float)
        e = np.asarray(t.get("end", t.get("end_ras")), dtype=float)
        v = e - s
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            continue
        out.append(TrajectoryLine(
            name=str(t.get("name") or ""),
            mid=0.5 * (s + e),
            direction=v / n,
        ))
    return out


def orthogonal_procrustes_R(src_dirs: np.ndarray, dst_dirs: np.ndarray) -> np.ndarray:
    """Best rotation that maps src direction unit vectors → dst direction
    unit vectors (no translation, no scale, no reflection). Both arrays
    are (N, 3). Direction vectors are anchored at the origin so the
    centroid step from full Procrustes is dropped.
    """
    H = src_dirs.T @ dst_dirs
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R


def line_perp_distance(
    p_a: np.ndarray, d_a: np.ndarray,
    p_b: np.ndarray, d_b: np.ndarray,
) -> float:
    """Perpendicular distance between two infinite 3-D lines.

    Each line: anchor + s · unit_dir. Closest-approach distance is
    ``|(p_b − p_a) · n_hat|`` where ``n_hat = (d_a × d_b) / ||d_a × d_b||``.
    For (near-)parallel lines the cross product collapses; fall back to
    the point-to-line distance from ``p_b`` onto line A.
    """
    delta = p_b - p_a
    n = np.cross(d_a, d_b)
    n_norm = float(np.linalg.norm(n))
    if n_norm < 1e-6:
        return float(np.linalg.norm(delta - (delta @ d_a) * d_a))
    return float(abs(delta @ (n / n_norm)))


# ---------------------------------------------------------------------
# RANSAC + refinement
# ---------------------------------------------------------------------


def _ransac_align_lines(
    ros_lines: Sequence[TrajectoryLine],
    det_lines: Sequence[TrajectoryLine],
    *,
    angle_tol_deg: float,
    perp_tol_mm: float,
    n_iter: int,
    seed: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """RANSAC on LINES — endpoint-drift-immune.

    Per iteration: pick 3 random det lines + 3 random ros lines. For
    each of 3! orderings × 2³ axis-sign combos on the det axes, fit R
    via orthogonal Procrustes on the direction triplets, t from midpoint
    centroids. Apply (R, t) to every det line; count inliers as
    (det, ros) pairs whose axis angle ≤ tol AND perpendicular line-to-
    line distance ≤ tol.

    Returns ``(M_4x4, inlier_pairs)``. ``M_4x4`` is identity and
    ``inlier_pairs`` empty when fewer than 3 lines exist on either side.
    """
    rng = np.random.default_rng(seed)
    n_d, n_r = len(det_lines), len(ros_lines)
    if n_d < 3 or n_r < 3:
        return np.eye(4), []

    det_dirs = np.asarray([l.direction for l in det_lines], dtype=float)
    det_mids = np.asarray([l.mid for l in det_lines], dtype=float)
    ros_dirs = np.asarray([l.direction for l in ros_lines], dtype=float)
    ros_mids = np.asarray([l.mid for l in ros_lines], dtype=float)

    cos_tol = float(np.cos(np.radians(angle_tol_deg)))
    sign_combos = list(product([+1.0, -1.0], repeat=3))

    best_M = np.eye(4)
    best_pairs: list[tuple[int, int]] = []
    best_n = 0

    for _ in range(n_iter):
        d_idx = rng.choice(n_d, size=3, replace=False)
        r_idx = rng.choice(n_r, size=3, replace=False)
        d_dirs_s = det_dirs[d_idx]
        d_mids_s = det_mids[d_idx]

        for perm in permutations(range(3)):
            r_pick = r_idx[list(perm)]
            r_dirs_s = ros_dirs[r_pick]
            r_mids_s = ros_mids[r_pick]

            for signs in sign_combos:
                d_signed = d_dirs_s * np.asarray(signs)[:, None]
                try:
                    R = orthogonal_procrustes_R(d_signed, r_dirs_s)
                except np.linalg.LinAlgError:
                    continue
                t = r_mids_s.mean(axis=0) - R @ d_mids_s.mean(axis=0)

                det_dirs_t = det_dirs @ R.T
                det_mids_t = det_mids @ R.T + t

                pairs: list[tuple[int, int]] = []
                for di in range(n_d):
                    da = det_dirs_t[di]
                    pa = det_mids_t[di]
                    best_pd = np.inf
                    best_ri = -1
                    for ri in range(n_r):
                        if abs(float(da @ ros_dirs[ri])) < cos_tol:
                            continue
                        pd = line_perp_distance(pa, da, ros_mids[ri], ros_dirs[ri])
                        if pd < best_pd:
                            best_pd = pd
                            best_ri = ri
                    if best_ri >= 0 and best_pd <= perp_tol_mm:
                        pairs.append((di, best_ri))

                if len(pairs) > best_n:
                    M = np.eye(4)
                    M[:3, :3] = R
                    M[:3, 3] = t
                    best_M = M
                    best_pairs = pairs
                    best_n = len(pairs)

    return best_M, best_pairs


def _refine_from_pairs(
    det_lines: Sequence[TrajectoryLine],
    ros_lines: Sequence[TrajectoryLine],
    pairs: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Refit (R, t) from inlier pairs.

    R from direction Procrustes on inlier axes (sign-aligned per pair to
    the ros axis via dot-product). t from inlier midpoint centroids.
    Returns identity for empty pair lists.
    """
    if not pairs:
        return np.eye(4)
    d_dirs = np.asarray([det_lines[di].direction for di, _ in pairs], dtype=float)
    r_dirs = np.asarray([ros_lines[ri].direction for _, ri in pairs], dtype=float)
    d_mids = np.asarray([det_lines[di].mid for di, _ in pairs], dtype=float)
    r_mids = np.asarray([ros_lines[ri].mid for _, ri in pairs], dtype=float)
    dots = np.einsum("ij,ij->i", d_dirs, r_dirs)
    signs = np.where(dots < 0, -1.0, 1.0)
    d_dirs = d_dirs * signs[:, None]
    R = orthogonal_procrustes_R(d_dirs, r_dirs)
    t = r_mids.mean(axis=0) - R @ d_mids.mean(axis=0)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


# ---------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------


def cross_volume_match(
    ros_trajs: Iterable[dict | TrajectoryLine],
    det_trajs: Iterable[dict | TrajectoryLine],
    *,
    angle_tol_deg: float = 15.0,
    ransac_perp_tol_mm: float = 8.0,
    match_perp_tol_mm: float = 12.0,
    n_iter: int = 2000,
    seed: int = 42,
) -> CrossVolumeMatch:
    """Match detector trajectories against ROS-planned trajectories
    purely from line geometry.

    Parameters
    ----------
    ros_trajs, det_trajs
        Iterables of either ``TrajectoryLine`` or dicts with ``name`` +
        ``start``/``end`` (or ``start_ras``/``end_ras``) keys. ROS lines
        are the ones whose names you want propagated.
    angle_tol_deg
        Axis-angle tolerance for both RANSAC inlier counting and the
        final greedy assignment (default 15°).
    ransac_perp_tol_mm
        Perpendicular line-to-line distance tolerance during RANSAC
        inlier counting (default 8 mm — tight enough to reject random
        triplet hypotheses).
    match_perp_tol_mm
        Perpendicular tolerance for the final greedy ROS↔det
        assignment (default 12 mm — slightly looser than RANSAC since
        the refined transform is generally good and this just allows a
        few pairs near the edge to find each other).
    n_iter, seed
        RANSAC iteration budget + RNG seed.

    Returns
    -------
    CrossVolumeMatch
        Recovered transform + per-ROS-line match (or ``None`` for ROS
        lines without a detector counterpart). When fewer than 3 lines
        exist on either side the result is identity-transform / empty.
    """
    ros_lines = trajectories_to_lines(ros_trajs)
    det_lines = trajectories_to_lines(det_trajs)

    M, ransac_pairs = _ransac_align_lines(
        ros_lines, det_lines,
        angle_tol_deg=angle_tol_deg,
        perp_tol_mm=ransac_perp_tol_mm,
        n_iter=n_iter,
        seed=seed,
    )
    if len(ransac_pairs) < 3:
        return CrossVolumeMatch(
            transform_4x4=np.eye(4),
            ransac_inliers=len(ransac_pairs),
            refined_inliers=0,
            pairs=[(r.name, None, None, None) for r in ros_lines],
            diagnostics={
                "n_ros": len(ros_lines), "n_det": len(det_lines),
                "n_iter": int(n_iter),
                "reason": "ransac_under_min_inliers" if len(ros_lines) >= 3
                          and len(det_lines) >= 3 else "too_few_lines",
            },
        )

    M = _refine_from_pairs(det_lines, ros_lines, ransac_pairs)

    # Push every det line through the refined transform and count
    # inliers under the same RANSAC tolerance (diagnostic only).
    R, t = M[:3, :3], M[:3, 3]
    cos_tol = float(np.cos(np.radians(angle_tol_deg)))
    refined_inliers = 0
    for d in det_lines:
        da = R @ d.direction
        pa = R @ d.mid + t
        best_pd = np.inf
        for r in ros_lines:
            if abs(float(da @ r.direction)) < cos_tol:
                continue
            pd = line_perp_distance(pa, da, r.mid, r.direction)
            best_pd = min(best_pd, pd)
        if best_pd <= ransac_perp_tol_mm:
            refined_inliers += 1

    # Greedy ROS-major assignment: each ROS line takes its best unused
    # det neighbor under the (looser) match tolerances.
    used = set()
    pairs_out: list[tuple[str, str | None, float | None, float | None]] = []
    det_dirs_t = np.asarray([R @ d.direction for d in det_lines])
    det_mids_t = np.asarray([R @ d.mid + t for d in det_lines])
    for r in ros_lines:
        best = None
        for di, d in enumerate(det_lines):
            if di in used:
                continue
            cos_ang = abs(float(r.direction @ det_dirs_t[di]))
            if cos_ang < cos_tol:
                continue
            pd = line_perp_distance(r.mid, r.direction,
                                     det_mids_t[di], det_dirs_t[di])
            if pd > match_perp_tol_mm:
                continue
            score = (1.0 - cos_ang) * 30.0 + pd
            if best is None or score < best[0]:
                best = (score, di, np.degrees(np.arccos(min(cos_ang, 1.0))), pd)
        if best is None:
            pairs_out.append((r.name, None, None, None))
        else:
            _, di, ang, pd = best
            used.add(di)
            pairs_out.append((r.name, det_lines[di].name, float(ang), float(pd)))

    return CrossVolumeMatch(
        transform_4x4=M,
        ransac_inliers=len(ransac_pairs),
        refined_inliers=int(refined_inliers),
        pairs=pairs_out,
        diagnostics={
            "n_ros": len(ros_lines),
            "n_det": len(det_lines),
            "n_iter": int(n_iter),
            "angle_tol_deg": float(angle_tol_deg),
            "ransac_perp_tol_mm": float(ransac_perp_tol_mm),
            "match_perp_tol_mm": float(match_perp_tol_mm),
        },
    )


__all__ = [
    "TrajectoryLine",
    "CrossVolumeMatch",
    "cross_volume_match",
    "trajectories_to_lines",
    "orthogonal_procrustes_R",
    "line_perp_distance",
]
