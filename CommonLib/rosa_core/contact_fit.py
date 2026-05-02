"""Postop CT candidate fitting helpers for electrode axis/depth refinement.

V2 strategy:
- finite cylindrical ROI around planned segment
- slab centroids sampled along planned depth axis
- RANSAC line fit on slab centroids (robust to nearby electrodes/outliers)
- depth anchor from deepest valid inlier centroid near planned target
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math

import numpy as np

from .types import ContactFitResult, Point3D


@dataclass(frozen=True)
class GuidedFitEMConfig:
    roi_radius_mm: float = 5.0
    cluster_radius_mm: float = 1.6
    min_cluster_size: int = 6
    assignment_distance_mm: float = 1.6
    assignment_margin_mm: float = 6.0
    support_radial_limit_mm: float = 1.6
    endpoint_radial_limit_mm: float = 1.2
    max_gap_mm: float = 4.5
    axis_em_iters: int = 5
    min_clusters_for_fit: int = 3


def unit(v: np.ndarray | list[float]) -> np.ndarray:
    """Return normalized 3D vector."""
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n <= 1e-9:
        raise ValueError("Zero-length vector")
    return v / n


def angle_deg(u: np.ndarray | list[float], v: np.ndarray | list[float]) -> float:
    """Return unsigned angle in degrees between two 3D vectors."""
    u = unit(u)
    v = unit(v)
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(math.degrees(math.acos(dot)))


def point_line_distance(
    points: np.ndarray | list[list[float]],
    line_point: np.ndarray | list[float],
    line_dir: np.ndarray | list[float],
) -> np.ndarray:
    """Return perpendicular distances from Nx3 points to an infinite 3D line."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    p0 = np.asarray(line_point, dtype=float)
    d = unit(line_dir)
    rel = pts - p0
    proj = rel @ d
    closest = p0 + np.outer(proj, d)
    return np.linalg.norm(pts - closest, axis=1)


def filter_points_in_segment_cylinder(
    points: np.ndarray | list[list[float]],
    seg_start: np.ndarray | list[float],
    seg_end: np.ndarray | list[float],
    radius_mm: float,
    margin_mm: float = 5.0,
) -> np.ndarray:
    """Keep points near a finite line segment (with along-axis margin)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if pts.size == 0:
        return pts.reshape(0, 3)

    p0 = np.asarray(seg_start, dtype=float)
    p1 = np.asarray(seg_end, dtype=float)
    axis = unit(p1 - p0)
    length = float(np.linalg.norm(p1 - p0))
    rel = pts - p0
    t = rel @ axis
    closest = p0 + np.outer(t, axis)
    d = np.linalg.norm(pts - closest, axis=1)
    m = float(margin_mm)
    keep = (d <= float(radius_mm)) & (t >= -m) & (t <= length + m)
    return pts[keep]


def fit_axis_pca(
    points: np.ndarray | list[list[float]],
    weights: np.ndarray | list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit principal axis by PCA and return ``(center, axis_unit)``.

    With ``weights=None``: unweighted covariance + ``eigh``.
    With weights: amplitude-weighted centroid + SVD on the weighted
    centered matrix (matches the prior ``guided_fit_engine._pca_axis``).
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if pts.shape[0] < 3:
        raise ValueError("At least 3 points are required for axis fitting")
    if weights is None:
        center = np.mean(pts, axis=0)
        centered = pts - center
        cov = centered.T @ centered / max(pts.shape[0] - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, int(np.argmax(eigvals))]
        return center, unit(axis)
    w = np.asarray(weights, dtype=float).reshape(-1)
    w = w / float(w.sum() or 1.0)
    center = np.sum(pts * w[:, None], axis=0)
    centered = pts - center
    weighted = centered * w[:, None]
    _U, _S, Vt = np.linalg.svd(weighted, full_matrices=False)
    return center, unit(Vt[0])


def build_slab_centroids(
    points: np.ndarray | list[list[float]],
    origin: np.ndarray | list[float],
    axis: np.ndarray | list[float],
    t_min: float,
    t_max: float,
    step_mm: float = 1.0,
    slab_half_thickness_mm: float = 0.9,
    min_points_per_slab: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample centroids of threshold points in depth slabs along a reference axis."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if pts.size == 0:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)

    origin = np.asarray(origin, dtype=float)
    axis = unit(axis)
    t = (pts - origin) @ axis
    centers = np.arange(float(t_min), float(t_max) + float(step_mm) * 0.5, float(step_mm))
    centroids = []
    centroid_t = []
    half = float(slab_half_thickness_mm)

    for tc in centers:
        mask = np.abs(t - tc) <= half
        n = int(np.count_nonzero(mask))
        if n < int(min_points_per_slab):
            continue
        c = np.mean(pts[mask], axis=0)
        centroids.append(c)
        centroid_t.append(float(np.dot(c - origin, axis)))

    if not centroids:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)
    return np.asarray(centroids, dtype=float), np.asarray(centroid_t, dtype=float)


def ransac_fit_line(
    points: np.ndarray | list[list[float]],
    max_iterations: int = 200,
    inlier_threshold_mm: float = 0.9,
    min_inliers: int = 6,
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit a 3D line with RANSAC, then refine with PCA on inliers."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    n = pts.shape[0]
    if n < 2:
        raise ValueError("At least 2 points are required for RANSAC line fit")

    rng = np.random.default_rng(int(rng_seed))
    best_mask = None
    best_count = -1
    best_score = float("inf")
    thr = float(inlier_threshold_mm)

    for _ in range(int(max_iterations)):
        i, j = rng.choice(n, size=2, replace=False)
        p0 = pts[i]
        d = pts[j] - p0
        dn = float(np.linalg.norm(d))
        if dn <= 1e-9:
            continue
        d /= dn
        dist = point_line_distance(pts, p0, d)
        mask = dist <= thr
        count = int(np.count_nonzero(mask))
        if count <= 0:
            continue
        score = float(np.mean(dist[mask]))
        if (count > best_count) or (count == best_count and score < best_score):
            best_count = count
            best_score = score
            best_mask = mask

    if best_mask is None or best_count < int(min_inliers):
        # Fallback to PCA on all points when RANSAC cannot find a stable consensus.
        center, axis = fit_axis_pca(pts)
        return center, axis, np.ones((n,), dtype=bool), 0.0

    inliers = pts[best_mask]
    center, axis = fit_axis_pca(inliers)
    dist = point_line_distance(inliers, center, axis)
    rms = float(np.sqrt(np.mean(dist**2))) if dist.size else 0.0
    return center, axis, best_mask, rms


def _planned_tip_and_axis(
    entry: np.ndarray | Point3D,
    target: np.ndarray | Point3D,
    tip_at: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return planned tip point and axis direction in LPS."""
    entry = np.asarray(entry, dtype=float)
    target = np.asarray(target, dtype=float)
    tip_at = (tip_at or "target").lower()
    if tip_at == "entry":
        tip = entry
        axis = unit(target - entry)
    else:
        tip = target
        axis = unit(entry - target)
    return tip, axis


def _cluster_points_radius(
    points: np.ndarray | list[list[float]],
    radius_mm: float,
    min_cluster_size: int = 6,
) -> list[np.ndarray]:
    """Group nearby candidate points into compact local clusters."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    n = int(pts.shape[0])
    if n == 0:
        return []
    radius2 = float(radius_mm) ** 2
    visited = np.zeros((n,), dtype=bool)
    clusters: list[np.ndarray] = []
    for seed in range(n):
        if visited[seed]:
            continue
        queue = [seed]
        visited[seed] = True
        members = []
        while queue:
            idx = queue.pop()
            members.append(idx)
            diff = pts - pts[idx]
            nbrs = np.where(np.sum(diff * diff, axis=1) <= radius2)[0]
            for nbr in nbrs.tolist():
                if not visited[nbr]:
                    visited[nbr] = True
                    queue.append(nbr)
        if len(members) >= int(min_cluster_size):
            clusters.append(np.asarray(members, dtype=int))
    return clusters


def _contact_pitch_from_offsets(contact_offsets_mm: list[float] | np.ndarray | None) -> float | None:
    """Estimate center-to-center pitch from ordered contact offsets."""
    offs = np.asarray(contact_offsets_mm if contact_offsets_mm is not None else [], dtype=float).reshape(-1)
    if offs.size < 2:
        return None
    diffs = np.diff(np.sort(offs))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return None
    return float(np.median(diffs))


def _configured_gap_priors_mm(contact_offsets_mm: list[float] | np.ndarray | None) -> list[float]:
    """Return larger configured inter-contact gaps implied by electrode offsets."""
    offs = np.asarray(contact_offsets_mm if contact_offsets_mm is not None else [], dtype=float).reshape(-1)
    if offs.size < 2:
        return []
    diffs = np.diff(np.sort(offs))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return []
    base_pitch = float(np.median(diffs))
    configured = sorted({float(d) for d in diffs if float(d) > base_pitch + 2.0})
    return configured


def _weighted_fit_axis_pca(
    points: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit principal axis with non-negative per-point weights."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if pts.shape[0] < 2:
        raise ValueError("At least 2 points are required for weighted axis fitting")
    w = np.clip(w, 0.0, None)
    if float(np.sum(w)) <= 1e-9:
        return fit_axis_pca(pts)
    center = np.average(pts, axis=0, weights=w)
    centered = pts - center
    cov = (centered * w[:, None]).T @ centered / max(float(np.sum(w)), 1.0)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    return center, unit(axis)


def _summarize_clusters(
    roi_pts: np.ndarray,
    clusters: list[np.ndarray],
    origin: np.ndarray,
    planned_deep_axis: np.ndarray,
) -> list[dict[str, object]]:
    """Compute geometry descriptors for local point clusters."""
    summaries: list[dict[str, object]] = []
    for cluster_id, members in enumerate(clusters):
        pts = roi_pts[members]
        center, axis = fit_axis_pca(pts)
        rel = pts - center
        t = rel @ axis
        radial = point_line_distance(pts, center, axis)
        eigen_span = float(np.max(t) - np.min(t)) if t.size else 0.0
        t_planned = float(np.dot(center - origin, planned_deep_axis))
        summaries.append(
            {
                "cluster_id": int(cluster_id),
                "center": center,
                "axis": axis,
                "count": int(pts.shape[0]),
                "span_mm": float(eigen_span),
                "radial_rms_mm": float(np.sqrt(np.mean(radial**2))) if radial.size else 0.0,
                "t_planned_mm": t_planned,
                "points": pts,
            }
        )
    return summaries


def _terminal_contiguous_chain(
    clusters: list[dict[str, object]],
    *,
    line_point: np.ndarray,
    line_axis: np.ndarray,
    max_gap_mm: float,
) -> list[dict[str, object]]:
    """Return deepest contiguous terminal cluster run along `line_axis`.

    Any gap larger than `max_gap_mm` breaks continuity. This prevents the deep
    anchor from jumping across a likely unrelated passing electrode.
    """
    if not clusters:
        return []
    axis = unit(line_axis)
    ordered = []
    for item in clusters:
        center = np.asarray(item["center"], dtype=float)
        ordered.append((float(np.dot(center - line_point, axis)), item))
    ordered.sort(key=lambda pair: pair[0])
    start_index = len(ordered) - 1
    if len(ordered) >= 2 and (ordered[-1][0] - ordered[-2][0]) > float(max_gap_mm):
        # Ignore an isolated deepest cluster; it is more likely a passing electrode
        # than the real terminal contact chain.
        start_index = len(ordered) - 2
    chain = [ordered[start_index][1]]
    prev_t = ordered[start_index][0]
    for t_val, item in reversed(ordered[:start_index]):
        if (prev_t - float(t_val)) > float(max_gap_mm):
            break
        chain.append(item)
        prev_t = float(t_val)
    chain.reverse()
    return chain


def _deepest_tight_terminal_centroid_t(
    clusters: list[dict[str, object]],
    *,
    line_point: np.ndarray,
    line_axis: np.ndarray,
    radial_limit_mm: float,
    max_gap_mm: float,
    min_chain_clusters: int = 3,
) -> tuple[float | None, int]:
    """Return deepest centroid t from the best tight contiguous run near the axis."""
    if not clusters:
        return None, 0
    axis = unit(line_axis)
    qualified: list[tuple[float, dict[str, object]]] = []
    for item in clusters:
        center = np.asarray(item["center"], dtype=float)
        radial = float(point_line_distance(np.asarray([center]), line_point, axis)[0])
        if radial > float(radial_limit_mm):
            continue
        qualified.append((float(np.dot(center - line_point, axis)), item))
    if not qualified:
        return None, 0
    qualified.sort(key=lambda pair: pair[0])
    runs: list[list[tuple[float, dict[str, object]]]] = []
    current = [qualified[0]]
    for t_val, item in qualified[1:]:
        if (float(t_val) - float(current[-1][0])) <= float(max_gap_mm):
            current.append((t_val, item))
        else:
            runs.append(current)
            current = [(t_val, item)]
    runs.append(current)

    eligible = [run for run in runs if len(run) >= int(min_chain_clusters)]
    if not eligible:
        return None, 0
    best_run = max(eligible, key=lambda run: (run[-1][0], len(run)))
    return float(best_run[-1][0]), int(len(best_run))


def _deepest_directional_terminal_centroid_t(
    clusters: list[dict[str, object]],
    *,
    line_point: np.ndarray,
    line_axis: np.ndarray,
    radial_limit_mm: float,
    max_gap_mm: float,
    direction_cos_min: float = 0.92,
    min_chain_clusters: int = 3,
) -> tuple[float | None, int]:
    """Return deepest centroid t from a tight terminal run with local direction consistency."""
    if not clusters:
        return None, 0
    axis = unit(line_axis)
    qualified: list[tuple[float, np.ndarray, dict[str, object]]] = []
    for item in clusters:
        center = np.asarray(item["center"], dtype=float)
        radial = float(point_line_distance(np.asarray([center]), line_point, axis)[0])
        if radial > float(radial_limit_mm):
            continue
        qualified.append((float(np.dot(center - line_point, axis)), center, item))
    if len(qualified) < int(min_chain_clusters):
        return None, 0
    qualified.sort(key=lambda triple: triple[0])

    accepted = [qualified[0]]
    for candidate in qualified[1:]:
        cand_t, cand_center, _ = candidate
        prev_t, prev_center, _ = accepted[-1]
        if (float(cand_t) - float(prev_t)) > float(max_gap_mm):
            accepted = [candidate]
            continue
        if len(accepted) >= 2:
            prev2_t, prev2_center, _ = accepted[-2]
            local_dir = unit(prev_center - prev2_center)
            step_dir = unit(cand_center - prev_center)
            if float(abs(np.dot(local_dir, step_dir))) < float(direction_cos_min):
                break
        accepted.append(candidate)
    if len(accepted) < int(min_chain_clusters):
        return None, 0
    return float(accepted[-1][0]), int(len(accepted))


def _local_terminal_compact_clusters(
    roi_pts: np.ndarray,
    *,
    line_point: np.ndarray,
    line_axis: np.ndarray,
    planned_target_t_mm: float,
    back_span_mm: float = 16.0,
    forward_span_mm: float = 3.0,
    radial_limit_mm: float = 1.25,
    cluster_radius_mm: float = 1.35,
    min_cluster_size: int = 4,
) -> list[dict[str, object]]:
    """Build compact local clusters in a narrow terminal window near the deep end."""
    pts = np.asarray(roi_pts, dtype=float).reshape(-1, 3)
    if pts.size == 0:
        return []
    axis = unit(line_axis)
    t = (pts - line_point) @ axis
    radial = point_line_distance(pts, line_point, axis)
    keep = (
        (t >= float(planned_target_t_mm) - float(back_span_mm))
        & (t <= float(planned_target_t_mm) + float(forward_span_mm))
        & (radial <= float(radial_limit_mm))
    )
    terminal_pts = pts[keep]
    if terminal_pts.shape[0] == 0:
        return []
    clusters = _cluster_points_radius(
        terminal_pts,
        radius_mm=float(cluster_radius_mm),
        min_cluster_size=int(min_cluster_size),
    )
    if not clusters:
        return []
    summaries = _summarize_clusters(terminal_pts, clusters, line_point, axis)
    return [
        summary
        for summary in summaries
        if float(summary["radial_rms_mm"]) <= 1.0
    ]


def _terminal_blob_axis_stats(
    cluster: dict[str, object],
    *,
    line_point: np.ndarray,
    line_axis: np.ndarray,
    distal_quantile: float = 0.9,
) -> dict[str, float]:
    """Return center/distal terminal projections for one cluster."""
    axis = unit(line_axis)
    center = np.asarray(cluster["center"], dtype=float)
    radial = float(point_line_distance(np.asarray([center]), line_point, axis)[0])
    center_t = float(np.dot(center - line_point, axis))
    pts = np.asarray(cluster.get("points"), dtype=float).reshape(-1, 3)
    if pts.size:
        t_vals = (pts - line_point) @ axis
        distal_t = float(np.quantile(t_vals, float(distal_quantile)))
        max_t = float(np.max(t_vals))
    else:
        distal_t = center_t
        max_t = center_t
    return {
        "center_t_mm": center_t,
        "distal_t_mm": distal_t,
        "max_t_mm": max_t,
        "radial_mm": radial,
    }


def _cluster_distal_extent_t(
    cluster_summary: dict[str, object],
    *,
    line_point: np.ndarray,
    line_axis: np.ndarray,
    distal_quantile: float = 0.98,
) -> float | None:
    """Return distal projected extent of one cluster along the fitted axis."""
    pts = np.asarray(cluster_summary.get("points"), dtype=float).reshape(-1, 3)
    if pts.size == 0:
        return None
    axis = unit(line_axis)
    t_vals = (pts - line_point) @ axis
    if t_vals.size == 0:
        return None
    return float(np.quantile(t_vals, float(distal_quantile)))


def _score_terminal_blob_candidates(
    clusters: list[dict[str, object]],
    *,
    line_point: np.ndarray,
    line_axis: np.ndarray,
    planned_target_t_mm: float,
    max_gap_mm: float,
    max_depth_shift_mm: float,
) -> dict[str, object]:
    """Score local terminal blobs and select the best deep anchor candidate."""
    if not clusters:
        return {"selected_cluster_id": None, "selected_anchor_t_mm": None, "diagnostics": []}
    ordered = sorted(clusters, key=lambda item: float(item["t_planned_mm"]))
    diagnostics: list[dict[str, object]] = []
    prev_center_t = None
    best_score = None
    best_cluster_id = None
    best_anchor_t = None
    for cluster in ordered:
        stats = _terminal_blob_axis_stats(cluster, line_point=line_point, line_axis=line_axis)
        center_t = float(stats["center_t_mm"])
        distal_t = float(stats["distal_t_mm"])
        radial = float(stats["radial_mm"])
        span_mm = float(cluster.get("span_mm", 0.0) or 0.0)
        count = int(cluster.get("count", 0) or 0)
        gap_from_prev = None if prev_center_t is None else float(center_t - prev_center_t)
        prev_center_t = center_t

        continuity_score = 0.0
        continuity_ok = True
        if gap_from_prev is not None:
            if gap_from_prev <= float(max_gap_mm) + 1.25:
                continuity_score = 1.0 - 0.20 * abs(gap_from_prev - 3.5)
            else:
                continuity_ok = False
                continuity_score = -1.5 * (gap_from_prev - (float(max_gap_mm) + 1.25))

        over_depth_penalty = 1.15 * max(0.0, distal_t - float(planned_target_t_mm))
        under_depth_penalty = 0.35 * max(0.0, float(planned_target_t_mm) - distal_t)
        shape_penalty = 0.35 * max(0.0, span_mm - 2.5)
        radial_penalty = 2.0 * radial
        count_bonus = 0.04 * min(count, 20)
        score = distal_t - over_depth_penalty - under_depth_penalty - shape_penalty - radial_penalty + continuity_score + count_bonus

        eligible = True
        if gap_from_prev is not None and gap_from_prev > float(max_gap_mm) + 1.25:
            eligible = False
        if (distal_t < float(planned_target_t_mm) - 8.0) and (gap_from_prev is None or gap_from_prev > float(max_gap_mm) + 1.25):
            eligible = False
        if distal_t > float(planned_target_t_mm) + float(max_depth_shift_mm) + 2.5:
            eligible = False

        diagnostics.append(
            {
                "cluster_id": int(cluster.get("cluster_id", -1)),
                "center_t_mm": round(center_t, 4),
                "distal_t_mm": round(distal_t, 4),
                "max_t_mm": round(float(stats["max_t_mm"]), 4),
                "radial_mm": round(radial, 4),
                "count": count,
                "span_mm": round(span_mm, 4),
                "gap_from_prev_mm": None if gap_from_prev is None else round(gap_from_prev, 4),
                "continuity_ok": bool(continuity_ok),
                "score": round(float(score), 4),
                "eligible": bool(eligible),
                "selected": False,
            }
        )
        if not eligible:
            continue
        if best_score is None or float(score) > float(best_score):
            best_score = float(score)
            best_cluster_id = int(cluster.get("cluster_id", -1))
            best_anchor_t = float(distal_t)

    for diag in diagnostics:
        if best_cluster_id is not None and int(diag["cluster_id"]) == int(best_cluster_id):
            diag["selected"] = True
    return {
        "selected_cluster_id": best_cluster_id,
        "selected_anchor_t_mm": best_anchor_t,
        "diagnostics": diagnostics,
    }


def _terminal_extent_profile_t(
    roi_pts: np.ndarray,
    *,
    line_point: np.ndarray,
    line_axis: np.ndarray,
    start_t_mm: float,
    end_t_mm: float,
    radial_limit_mm: float = 0.75,
    bin_mm: float = 0.5,
    min_points_per_bin: int = 3,
    max_empty_gap_bins: int = 2,
) -> float | None:
    """Return distal occupied extent in a tight tube along the fitted axis."""
    pts = np.asarray(roi_pts, dtype=float).reshape(-1, 3)
    if pts.size == 0:
        return None
    axis = unit(line_axis)
    t = (pts - line_point) @ axis
    radial = point_line_distance(pts, line_point, axis)
    keep = (t >= float(start_t_mm)) & (t <= float(end_t_mm)) & (radial <= float(radial_limit_mm))
    t_keep = t[keep]
    if t_keep.size == 0:
        return None
    bins = np.arange(float(start_t_mm), float(end_t_mm) + float(bin_mm) * 1.01, float(bin_mm))
    if bins.size < 2:
        return None
    counts, edges = np.histogram(t_keep, bins=bins)
    occupied = counts >= int(min_points_per_bin)
    if not np.any(occupied):
        return None
    last_occ_idx = int(np.max(np.where(occupied)[0]))
    empty_run = 0
    distal_idx = last_occ_idx
    for idx in range(last_occ_idx, -1, -1):
        if occupied[idx]:
            distal_idx = idx
            empty_run = 0
        else:
            empty_run += 1
            if empty_run > int(max_empty_gap_bins):
                break
    return float(edges[distal_idx + 1])


def _endpoint_from_forward_void(
    roi_pts: np.ndarray,
    *,
    line_point: np.ndarray,
    line_axis: np.ndarray,
    seed_t_mm: float,
    search_end_t_mm: float,
    radial_limit_mm: float = 0.8,
    bin_mm: float = 0.5,
    min_points_per_bin: int = 2,
    void_mm: float = 5.0,
    max_gap_mm: float = 4.5,
    allowed_gap_mm: list[float] | None = None,
) -> float | None:
    """Extend a terminal candidate until the last occupied bin before a forward void."""
    diag = _support_run_diagnostics(
        roi_pts,
        line_point=line_point,
        line_axis=line_axis,
        seed_t_mm=seed_t_mm,
        search_end_t_mm=search_end_t_mm,
        radial_limit_mm=radial_limit_mm,
        bin_mm=bin_mm,
        min_points_per_bin=min_points_per_bin,
        void_mm=void_mm,
        max_gap_mm=max_gap_mm,
        allowed_gap_mm=allowed_gap_mm,
    )
    return diag["selected_endpoint_t_mm"]


def _support_run_diagnostics(
    roi_pts: np.ndarray,
    *,
    line_point: np.ndarray,
    line_axis: np.ndarray,
    seed_t_mm: float,
    search_end_t_mm: float,
    radial_limit_mm: float = 0.8,
    bin_mm: float = 0.5,
    min_points_per_bin: int = 2,
    void_mm: float = 5.0,
    max_gap_mm: float = 4.5,
    allowed_gap_mm: list[float] | None = None,
) -> dict[str, object]:
    """Return occupied support runs and selected endpoint diagnostics in a tight tube."""
    pts = np.asarray(roi_pts, dtype=float).reshape(-1, 3)
    if pts.size == 0:
        return {"selected_endpoint_t_mm": None, "runs": [], "longest_run_index": None, "selected_run_index": None}
    axis = unit(line_axis)
    t = (pts - line_point) @ axis
    radial = point_line_distance(pts, line_point, axis)
    start_t = float(seed_t_mm) - float(bin_mm)
    end_t = max(float(search_end_t_mm), start_t + float(bin_mm))
    keep = (t >= start_t) & (t <= end_t) & (radial <= float(radial_limit_mm))
    t_keep = t[keep]
    if t_keep.size == 0:
        return {"selected_endpoint_t_mm": None, "runs": [], "longest_run_index": None, "selected_run_index": None}
    bins = np.arange(start_t, end_t + float(bin_mm) * 1.01, float(bin_mm))
    if bins.size < 2:
        return {"selected_endpoint_t_mm": None, "runs": [], "longest_run_index": None, "selected_run_index": None}
    counts, edges = np.histogram(t_keep, bins=bins)
    occupied = counts >= int(min_points_per_bin)
    if not np.any(occupied):
        return {"selected_endpoint_t_mm": None, "runs": [], "longest_run_index": None, "selected_run_index": None}
    void_bins = max(1, int(round(float(void_mm) / float(bin_mm))))
    max_gap_bins = max(1, int(round(float(max_gap_mm) / float(bin_mm))))
    allowed_gap_bins = [max(1, int(round(float(g) / float(bin_mm)))) for g in (allowed_gap_mm or []) if float(g) > 0.0]
    allowed_gap_tol_bins = max(1, int(round(1.25 / float(bin_mm))))
    occupied_idx = np.where(occupied)[0]
    seed_idx = int(np.searchsorted(edges[1:], float(seed_t_mm), side="left"))
    valid_idx = [idx for idx in occupied_idx.tolist() if idx >= seed_idx]
    if not valid_idx:
        runs = []
        return {"selected_endpoint_t_mm": None, "runs": runs, "longest_run_index": None, "selected_run_index": None}
    runs_idx: list[list[int]] = []
    current_run = [valid_idx[0]]
    for idx in valid_idx[1:]:
        gap_bins = idx - current_run[-1]
        allowed_large_gap = any(abs(int(gap_bins) - int(cfg)) <= int(allowed_gap_tol_bins) for cfg in allowed_gap_bins)
        if int(gap_bins) <= int(max_gap_bins) or allowed_large_gap:
            current_run.append(idx)
        else:
            runs_idx.append(current_run)
            current_run = [idx]
    runs_idx.append(current_run)
    runs: list[dict[str, object]] = []
    for run in runs_idx:
        start_idx = int(run[0])
        end_idx = int(run[-1])
        run_start_t = float(edges[start_idx])
        run_end_t = float(edges[end_idx + 1])
        gap_from_seed = max(0.0, run_start_t - float(seed_t_mm))
        reason = "candidate"
        if run_end_t < float(seed_t_mm):
            reason = "before_seed"
        elif gap_from_seed > float(max_gap_mm):
            allowed_seed_gap = any(abs(float(gap_from_seed) - float(cfg)) <= 1.25 for cfg in (allowed_gap_mm or []))
            if not allowed_seed_gap:
                reason = "gap_from_seed"
            else:
                reason = "configured_gap"
        runs.append(
            {
                "start_t_mm": round(run_start_t, 4),
                "end_t_mm": round(run_end_t, 4),
                "length_mm": round(run_end_t - run_start_t, 4),
                "bin_count": int(len(run)),
                "gap_from_seed_mm": round(gap_from_seed, 4),
                "reason": reason,
            }
        )
    longest_run_index = int(max(range(len(runs)), key=lambda i: float(runs[i]["length_mm"]))) if runs else None
    contiguous = [valid_idx[0]]
    prev_idx = valid_idx[0]
    for idx in valid_idx[1:]:
        gap_bins = idx - prev_idx
        allowed_large_gap = any(abs(int(gap_bins) - int(cfg)) <= int(allowed_gap_tol_bins) for cfg in allowed_gap_bins)
        if int(gap_bins) > int(max_gap_bins) and not allowed_large_gap:
            break
        contiguous.append(idx)
        prev_idx = idx
    selected_run_index = 0 if runs else None
    for idx in contiguous[::-1]:
        forward = occupied[idx + 1 : idx + 1 + void_bins]
        if forward.size == 0 or not np.any(forward):
            return {
                "selected_endpoint_t_mm": float(edges[idx + 1]),
                "runs": runs,
                "longest_run_index": longest_run_index,
                "selected_run_index": selected_run_index,
            }
    return {
        "selected_endpoint_t_mm": float(edges[contiguous[-1] + 1]),
        "runs": runs,
        "longest_run_index": longest_run_index,
        "selected_run_index": selected_run_index,
    }


def _fit_single_cluster_string_axis(
    cluster_summary: dict[str, object],
    *,
    planned_deep_axis: np.ndarray,
    min_span_mm: float = 18.0,
    max_radial_rms_mm: float = 1.1,
    interior_trim_frac: float = 0.12,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return `(center, axis)` for one dominant elongated cluster if it looks line-like."""
    span_mm = float(cluster_summary.get("span_mm", 0.0) or 0.0)
    radial_rms_mm = float(cluster_summary.get("radial_rms_mm", 0.0) or 0.0)
    if span_mm < float(min_span_mm):
        return None
    if radial_rms_mm > float(max_radial_rms_mm):
        return None
    pts = np.asarray(cluster_summary.get("points"), dtype=float).reshape(-1, 3)
    if pts.shape[0] < 8:
        return None
    center0, axis0 = fit_axis_pca(pts)
    if float(np.dot(axis0, planned_deep_axis)) < 0.0:
        axis0 = -axis0
    t = (pts - center0) @ axis0
    q_lo = float(np.quantile(t, float(interior_trim_frac)))
    q_hi = float(np.quantile(t, 1.0 - float(interior_trim_frac)))
    interior = pts[(t >= q_lo) & (t <= q_hi)]
    if interior.shape[0] >= 8:
        center1, axis1 = fit_axis_pca(interior)
        if float(np.dot(axis1, planned_deep_axis)) < 0.0:
            axis1 = -axis1
        return center1, axis1
    return center0, axis0


def _fit_axis_from_interior_cluster_centers(
    clusters: list[dict[str, object]],
    *,
    planned_deep_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit axis from interior cluster centers, trimming terminal-most support."""
    centers = np.asarray([np.asarray(item["center"], dtype=float) for item in clusters], dtype=float)
    weights = np.asarray([math.sqrt(max(1.0, float(item.get("count", 1.0)))) for item in clusters], dtype=float)
    center0, axis0 = _weighted_fit_axis_pca(centers, weights)
    if float(np.dot(axis0, planned_deep_axis)) < 0.0:
        axis0 = -axis0
    t = (centers - center0) @ axis0
    order = np.argsort(t)
    n = len(order)
    if n >= 5:
        keep = order[1:-1]
    elif n >= 3:
        keep = order[0:-1]
    else:
        keep = order
    interior_centers = centers[keep]
    interior_weights = weights[keep]
    center1, axis1 = _weighted_fit_axis_pca(interior_centers, interior_weights)
    if float(np.dot(axis1, planned_deep_axis)) < 0.0:
        axis1 = -axis1
    return center1, axis1


def _cluster_axis_interval(
    cluster_summary: dict[str, object],
    *,
    line_point: np.ndarray,
    line_axis: np.ndarray,
    radial_limit_mm: float,
    low_q: float = 0.02,
    high_q: float = 0.98,
) -> dict[str, float] | None:
    """Return robust interval of cluster support along a fixed axis inside a tight tube."""
    pts = np.asarray(cluster_summary.get("points"), dtype=float).reshape(-1, 3)
    if pts.shape[0] == 0:
        return None
    axis = unit(line_axis)
    radial = point_line_distance(pts, line_point, axis)
    keep = radial <= float(radial_limit_mm)
    if not np.any(keep):
        return None
    pts_keep = pts[keep]
    t = (pts_keep - line_point) @ axis
    if t.size == 0:
        return None
    t_min = float(np.quantile(t, float(low_q)))
    t_max = float(np.quantile(t, float(high_q)))
    center = np.asarray(cluster_summary["center"], dtype=float)
    center_t = float(np.dot(center - line_point, axis))
    center_radial = float(point_line_distance(np.asarray([center]), line_point, axis)[0])
    cluster_axis = unit(np.asarray(cluster_summary.get("axis"), dtype=float))
    axis_cos = float(abs(np.dot(cluster_axis, axis))) if np.isfinite(cluster_axis).all() else 1.0
    return {
        "t_min_mm": t_min,
        "t_max_mm": t_max,
        "center_t_mm": center_t,
        "center_radial_mm": center_radial,
        "axis_cos": axis_cos,
        "point_count": int(pts_keep.shape[0]),
    }


def _axis_support_run_from_clusters(
    cluster_summaries: list[dict[str, object]],
    *,
    core_cluster_ids: set[int],
    line_point: np.ndarray,
    line_axis: np.ndarray,
    planned_entry_t_mm: float | None = None,
    planned_target_t_mm: float | None = None,
    radial_limit_mm: float,
    max_gap_mm: float,
    allowed_gap_mm: list[float] | None = None,
    elongated_direction_cos_min: float = 0.78,
) -> dict[str, object]:
    """Select a contiguous same-axis support run after the axis is already fixed."""
    objects: list[dict[str, object]] = []
    allowed = [float(x) for x in (allowed_gap_mm or []) if float(x) > 0.0]
    for summary in cluster_summaries:
        interval = _cluster_axis_interval(
            summary,
            line_point=line_point,
            line_axis=line_axis,
            radial_limit_mm=radial_limit_mm,
        )
        reason = "compatible"
        compatible = interval is not None
        if interval is None:
            reason = "off_axis"
        else:
            span_mm = float(summary.get("span_mm", 0.0) or 0.0)
            if span_mm >= 5.0 and float(interval["axis_cos"]) < float(elongated_direction_cos_min):
                compatible = False
                reason = "direction"
        item = {
            "cluster_id": int(summary.get("cluster_id", -1)),
            "span_mm": float(summary.get("span_mm", 0.0) or 0.0),
            "count": int(summary.get("count", 0) or 0),
            "is_core": int(summary.get("cluster_id", -1)) in core_cluster_ids,
            "compatible": bool(compatible),
            "reason": reason,
        }
        if interval is not None:
            item.update(interval)
            item["tube_fraction"] = float(interval["point_count"]) / max(1.0, float(item["count"]))
        objects.append(item)

    compatible = [item for item in objects if bool(item["compatible"])]
    if not compatible:
        return {"selected_run": None, "runs": [], "objects": objects}
    compatible.sort(key=lambda item: (float(item["t_min_mm"]), float(item["t_max_mm"])))
    runs: list[list[dict[str, object]]] = []
    current = [compatible[0]]
    current_end = float(compatible[0]["t_max_mm"])
    for item in compatible[1:]:
        gap = float(item["t_min_mm"]) - float(current_end)
        allowed_large_gap = any(abs(float(gap) - cfg) <= 1.25 for cfg in allowed)
        if gap <= float(max_gap_mm) or allowed_large_gap:
            current.append(item)
            current_end = max(current_end, float(item["t_max_mm"]))
        else:
            runs.append(current)
            current = [item]
            current_end = float(item["t_max_mm"])
    runs.append(current)

    radial_sigma_mm = max(float(radial_limit_mm) * 0.75, 0.5)
    scored_runs: list[dict[str, object]] = []
    scored_run_items: list[tuple[list[dict[str, object]], dict[str, object]]] = []
    for idx, run in enumerate(runs):
        t_min = min(float(item["t_min_mm"]) for item in run)
        t_max = max(float(item["t_max_mm"]) for item in run)
        core_hits = sum(1 for item in run if bool(item["is_core"]))
        support_mass = 0.0
        core_support_mass = 0.0
        internal_gap_mm = 0.0
        prev_end = None
        for item in run:
            weight = math.sqrt(max(1.0, float(item.get("point_count", item.get("count", 1)) or 1.0)))
            radial_term = math.exp(-((float(item["center_radial_mm"]) / radial_sigma_mm) ** 2))
            tube_fraction = float(item.get("tube_fraction", 1.0) or 0.0)
            support_mass += float(weight * radial_term * tube_fraction)
            if bool(item["is_core"]):
                core_support_mass += float(weight * radial_term * tube_fraction)
            if prev_end is not None:
                internal_gap_mm += max(0.0, float(item["t_min_mm"]) - float(prev_end))
            prev_end = float(item["t_max_mm"])
        entry_penalty_mm = 0.0 if planned_entry_t_mm is None else abs(float(t_min) - float(planned_entry_t_mm))
        target_penalty_mm = 0.0 if planned_target_t_mm is None else abs(float(t_max) - float(planned_target_t_mm))
        proposal_penalty_mm = float(entry_penalty_mm + target_penalty_mm)
        run_score = (
            float(support_mass)
            + 0.35 * float(t_max - t_min)
            + 0.50 * float(core_support_mass)
            - 0.60 * float(internal_gap_mm)
            - 0.10 * float(entry_penalty_mm)
            - 0.45 * float(target_penalty_mm)
        )
        scored = {
            "run_index": int(idx),
            "cluster_ids": [int(item["cluster_id"]) for item in run],
            "t_min_mm": round(t_min, 4),
            "t_max_mm": round(t_max, 4),
            "length_mm": round(t_max - t_min, 4),
            "core_hits": int(core_hits),
            "cluster_count": int(len(run)),
            "support_mass": round(float(support_mass), 4),
            "core_support_mass": round(float(core_support_mass), 4),
            "internal_gap_mm": round(float(internal_gap_mm), 4),
            "entry_penalty_mm": round(float(entry_penalty_mm), 4),
            "target_penalty_mm": round(float(target_penalty_mm), 4),
            "proposal_penalty_mm": round(float(proposal_penalty_mm), 4),
            "score": round(float(run_score), 4),
            "allowed": bool(core_hits > 0),
        }
        scored_runs.append(scored)
        scored_run_items.append((run, scored))

    eligible = [(run, scored) for run, scored in scored_run_items if int(scored["core_hits"]) > 0]
    if not eligible:
        return {"selected_run": None, "runs": scored_runs, "objects": objects}
    selected, selected_scored = max(
        eligible,
        key=lambda pair: (
            float(pair[1]["score"]),
            int(pair[1]["core_hits"]),
            float(pair[1]["t_max_mm"]),
        ),
    )
    core_tube_fractions = [float(item.get("tube_fraction", 1.0) or 0.0) for item in selected if bool(item["is_core"])]
    tail_fraction_threshold = 0.5 if not core_tube_fractions else float(max(0.45, 0.7 * float(np.median(core_tube_fractions))))
    tail_items = [
        item for item in selected
        if bool(item["is_core"]) or float(item.get("tube_fraction", 0.0) or 0.0) >= tail_fraction_threshold
    ]
    if not tail_items:
        tail_items = list(selected)
    selected_run = {
        "cluster_ids": [int(item["cluster_id"]) for item in selected],
        "t_min_mm": float(min(float(item["t_min_mm"]) for item in selected)),
        "t_max_mm": float(max(float(item["t_max_mm"]) for item in selected)),
        "effective_t_max_mm": float(max(float(item["t_max_mm"]) for item in tail_items)),
        "core_hits": int(sum(1 for item in selected if bool(item["is_core"]))),
        "cluster_count": int(len(selected)),
        "score": float(selected_scored["score"]),
        "support_mass": float(selected_scored["support_mass"]),
        "core_support_mass": float(selected_scored["core_support_mass"]),
        "internal_gap_mm": float(selected_scored["internal_gap_mm"]),
        "tail_fraction_threshold": float(tail_fraction_threshold),
        "entry_penalty_mm": float(selected_scored["entry_penalty_mm"]),
        "target_penalty_mm": float(selected_scored["target_penalty_mm"]),
        "proposal_penalty_mm": float(selected_scored["proposal_penalty_mm"]),
    }
    return {"selected_run": selected_run, "runs": scored_runs, "objects": objects}


def _weighted_rms_to_line_through_target(
    points: np.ndarray,
    weights: np.ndarray,
    target: np.ndarray,
    axis: np.ndarray,
) -> float:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if pts.shape[0] == 0:
        return float("inf")
    radial = point_line_distance(pts, np.asarray(target, dtype=float), np.asarray(axis, dtype=float))
    w = np.clip(w, 0.0, None)
    if float(np.sum(w)) <= 1e-9:
        return float(np.sqrt(np.mean(radial**2)))
    return float(np.sqrt(np.sum(w * (radial**2)) / float(np.sum(w))))


def _cleanup_fit_with_fixed_target_and_length(
    fit: ContactFitResult,
    *,
    cluster_summaries: list[dict[str, object]],
    axis_support: dict[str, object],
    selected_support_run: dict[str, object] | None,
    line_point: np.ndarray,
    line_axis: np.ndarray,
    roi_pts: np.ndarray,
    planned_entry_lps: np.ndarray,
    planned_target_lps: np.ndarray,
    contact_offsets_mm: list[float] | np.ndarray | None,
) -> ContactFitResult:
    """Refine the start point with fixed deep endpoint and fixed span.

    The cleanup uses only trusted support from the selected run, favors the
    interior part of that run, and keeps the current deep endpoint unchanged.
    """
    if not bool(fit.get("success")) or selected_support_run is None:
        return fit

    target = np.asarray(fit["target_lps"], dtype=float)
    entry = np.asarray(fit["entry_lps"], dtype=float)
    current_axis = unit(target - entry)
    length = float(np.linalg.norm(target - entry))
    if length <= 1e-6:
        return fit

    selected_ids = {int(x) for x in (selected_support_run.get("cluster_ids") or [])}
    if not selected_ids:
        return fit
    object_by_id = {
        int(item.get("cluster_id", -1)): item
        for item in (axis_support.get("objects") or [])
    }
    run_t_min = float(selected_support_run.get("t_min_mm", 0.0) or 0.0)
    run_t_max = float(selected_support_run.get("t_max_mm", run_t_min) or run_t_min)
    run_length = max(0.0, run_t_max - run_t_min)
    interior_margin_mm = min(10.0, 0.15 * run_length)
    tail_fraction_threshold = float(selected_support_run.get("tail_fraction_threshold", 0.0) or 0.0)

    support_points: list[np.ndarray] = []
    support_weights: list[float] = []
    fallback_points: list[np.ndarray] = []
    fallback_weights: list[float] = []
    for summary in cluster_summaries:
        cluster_id = int(summary.get("cluster_id", -1))
        if cluster_id not in selected_ids:
            continue
        obj = object_by_id.get(cluster_id)
        if obj is None or not bool(obj.get("compatible", False)):
            continue
        center = np.asarray(summary["center"], dtype=float)
        t_center = float(np.dot(center - np.asarray(line_point, dtype=float), np.asarray(line_axis, dtype=float)))
        tube_fraction = float(obj.get("tube_fraction", 1.0) or 0.0)
        is_core = bool(obj.get("is_core", False))
        weight = max(0.15, tube_fraction) * (1.5 if is_core else 1.0)
        fallback_points.append(center)
        fallback_weights.append(weight)
        if (not is_core) and tube_fraction < tail_fraction_threshold:
            continue
        if len(selected_ids) >= 4 and t_center < run_t_min + interior_margin_mm:
            continue
        support_points.append(center)
        support_weights.append(weight)
    if len(support_points) < 3:
        support_points = fallback_points
        support_weights = fallback_weights
    if len(support_points) < 3:
        return fit

    current_segment_pts = filter_points_in_segment_cylinder(
        roi_pts,
        entry,
        target,
        radius_mm=2.2,
        margin_mm=0.0,
    )
    slab_points, slab_t = build_slab_centroids(
        current_segment_pts,
        origin=target,
        axis=-current_axis,
        t_min=0.0,
        t_max=length,
        step_mm=2.0,
        slab_half_thickness_mm=1.0,
        min_points_per_slab=4,
    )
    if slab_points.shape[0] >= 4:
        before_rms = _weighted_rms_to_line_through_target(
            slab_points,
            np.ones((slab_points.shape[0],), dtype=float),
            target,
            current_axis,
        )
        rel = slab_points - target[None, :]
        cov = rel.T @ rel / max(float(slab_points.shape[0]), 1.0)
        eigvals, eigvecs = np.linalg.eigh(cov)
        refined_axis = unit(eigvecs[:, int(np.argmax(eigvals))])
        if float(np.dot(refined_axis, current_axis)) < 0.0:
            refined_axis = -refined_axis
        after_rms = _weighted_rms_to_line_through_target(
            slab_points,
            np.ones((slab_points.shape[0],), dtype=float),
            target,
            refined_axis,
        )
        if math.isfinite(after_rms) and after_rms < before_rms - 1e-4:
            pts = slab_points
            weights = np.ones((slab_points.shape[0],), dtype=float)
            cleanup_source = "slab_centroids"
        else:
            pts = np.empty((0, 3), dtype=float)
            weights = np.empty((0,), dtype=float)
            cleanup_source = "slab_centroids"
    else:
        pts = np.empty((0, 3), dtype=float)
        weights = np.empty((0,), dtype=float)
        cleanup_source = "cluster_centers"

    if pts.shape[0] < 3:
        pts = np.asarray(support_points, dtype=float)
        weights = np.asarray(support_weights, dtype=float)
        cleanup_source = "cluster_centers"
    before_rms = _weighted_rms_to_line_through_target(pts, weights, target, current_axis)

    rel = pts - target[None, :]
    w = np.clip(weights, 0.0, None)
    cov = (rel * w[:, None]).T @ rel / max(float(np.sum(w)), 1.0)
    eigvals, eigvecs = np.linalg.eigh(cov)
    refined_axis = unit(eigvecs[:, int(np.argmax(eigvals))])
    if float(np.dot(refined_axis, current_axis)) < 0.0:
        refined_axis = -refined_axis
    after_rms = _weighted_rms_to_line_through_target(pts, weights, target, refined_axis)
    if not math.isfinite(after_rms) or after_rms >= before_rms - 1e-4:
        return fit

    new_entry = target - refined_axis * length
    new_center = 0.5 * (new_entry + target)
    planned_entry = np.asarray(planned_entry_lps, dtype=float)
    planned_target = np.asarray(planned_target_lps, dtype=float)
    planned_axis = unit(planned_target - planned_entry)
    ang = angle_deg(planned_axis, refined_axis)

    offs = np.asarray(contact_offsets_mm if contact_offsets_mm is not None else [], dtype=float).reshape(-1)
    if offs.size > 0:
        pred_centers = target[None, :] - np.outer(offs, refined_axis)
        diff = pred_centers[:, None, :] - roi_pts[None, :, :]
        d3 = np.linalg.norm(diff, axis=2)
        residual_3d = float(np.mean(np.min(d3, axis=1)))
        cand_t = (roi_pts - target) @ refined_axis
        pred_t = (pred_centers - target) @ refined_axis
        nearest_1d = np.min(np.abs(cand_t[None, :] - pred_t[:, None]), axis=1)
        one_d_residual = float(np.mean(nearest_1d))
    else:
        residual_3d = float(fit.get("residual_mm", 0.0) or 0.0)
        one_d_residual = float(fit.get("one_d_residual_mm", float("nan")))

    delta_target = target - planned_target
    target_shift_mm = float(np.dot(delta_target, refined_axis))
    lateral_vec = delta_target - np.dot(delta_target, refined_axis) * refined_axis
    lateral_shift_mm = float(np.linalg.norm(lateral_vec))

    new_fit = dict(fit)
    new_fit["entry_lps"] = new_entry.tolist()
    new_fit["target_lps"] = target.tolist()
    new_fit["axis_lps"] = (-refined_axis).tolist()
    new_fit["deep_axis_lps"] = refined_axis.tolist()
    new_fit["center_lps"] = new_center.tolist()
    new_fit["angle_deg"] = float(ang)
    new_fit["residual_mm"] = float(residual_3d)
    new_fit["one_d_residual_mm"] = float(one_d_residual)
    new_fit["tip_shift_mm"] = float(target_shift_mm)
    new_fit["lateral_shift_mm"] = float(lateral_shift_mm)
    new_fit["deep_t_raw_mm"] = float(np.dot(target - new_center, refined_axis))
    new_fit["deep_t_clamped_mm"] = float(np.dot(target - new_center, refined_axis))
    new_fit["planned_target_t_mm"] = float(np.dot(planned_target - new_center, refined_axis))
    new_fit["fitted_span_mm"] = float(length)
    new_fit["start_cleanup_applied"] = 1
    new_fit["start_cleanup_rms_before_mm"] = float(before_rms)
    new_fit["start_cleanup_rms_after_mm"] = float(after_rms)
    new_fit["start_cleanup_angle_delta_deg"] = float(angle_deg(current_axis, refined_axis))
    new_fit["start_cleanup_support_count"] = int(len(pts))
    new_fit["start_cleanup_source"] = str(cleanup_source)
    return new_fit


def _diagnose_directional_terminal_centroids(
    clusters: list[dict[str, object]],
    *,
    line_point: np.ndarray,
    line_axis: np.ndarray,
    radial_limit_mm: float,
    max_gap_mm: float,
    direction_cos_min: float = 0.92,
    min_chain_clusters: int = 3,
) -> dict[str, object]:
    """Return terminal-chain decision diagnostics for compact clusters."""
    axis = unit(line_axis)
    diagnostics: list[dict[str, object]] = []
    qualified: list[tuple[float, np.ndarray, dict[str, object]]] = []
    for item in clusters:
        center = np.asarray(item["center"], dtype=float)
        radial = float(point_line_distance(np.asarray([center]), line_point, axis)[0])
        t_val = float(np.dot(center - line_point, axis))
        diag = {
            "cluster_id": int(item.get("cluster_id", -1)),
            "t_mm": round(t_val, 4),
            "radial_mm": round(radial, 4),
            "count": int(item.get("count", 0)),
            "span_mm": round(float(item.get("span_mm", 0.0) or 0.0), 4),
            "qualified_radial": bool(radial <= float(radial_limit_mm)),
            "accepted": False,
            "reason": "radial",
        }
        diagnostics.append(diag)
        if radial > float(radial_limit_mm):
            continue
        qualified.append((t_val, center, item))
    if len(qualified) < int(min_chain_clusters):
        for diag in diagnostics:
            if bool(diag["qualified_radial"]):
                diag["reason"] = "insufficient_qualified"
        return {
            "selected_t_mm": None,
            "selected_cluster_id": None,
            "chain_count": 0,
            "diagnostics": diagnostics,
        }
    qualified.sort(key=lambda triple: triple[0])
    accepted = [qualified[0]]
    accepted_ids = {int(qualified[0][2].get("cluster_id", -1))}
    rejected_reason_by_id: dict[int, str] = {
        int(item.get("cluster_id", -1)): "truncated" for _, _, item in qualified[1:]
    }
    for candidate in qualified[1:]:
        cand_t, cand_center, cand_item = candidate
        cand_id = int(cand_item.get("cluster_id", -1))
        prev_t, prev_center, _ = accepted[-1]
        gap = float(cand_t) - float(prev_t)
        if gap > float(max_gap_mm):
            accepted = [candidate]
            accepted_ids = {cand_id}
            rejected_reason_by_id[cand_id] = "restart_after_gap"
            continue
        if len(accepted) >= 2:
            _, prev2_center, _ = accepted[-2]
            local_dir = unit(prev_center - prev2_center)
            step_dir = unit(cand_center - prev_center)
            if float(abs(np.dot(local_dir, step_dir))) < float(direction_cos_min):
                rejected_reason_by_id[cand_id] = "direction"
                break
        accepted.append(candidate)
        accepted_ids.add(cand_id)
        rejected_reason_by_id[cand_id] = "accepted"
    selected_t_mm = None
    selected_cluster_id = None
    if len(accepted) >= int(min_chain_clusters):
        selected_t_mm = float(accepted[-1][0])
        selected_cluster_id = int(accepted[-1][2].get("cluster_id", -1))
    else:
        accepted_ids = set()
    for diag in diagnostics:
        cluster_id = int(diag["cluster_id"])
        if not bool(diag["qualified_radial"]):
            continue
        diag["accepted"] = bool(cluster_id in accepted_ids)
        diag["reason"] = str(rejected_reason_by_id.get(cluster_id, "qualified"))
    return {
        "selected_t_mm": selected_t_mm,
        "selected_cluster_id": selected_cluster_id,
        "chain_count": int(len(accepted) if selected_t_mm is not None else 0),
        "diagnostics": diagnostics,
    }


def build_compact_centroid_observations(
    candidate_points_lps: np.ndarray | list[list[float]],
    cluster_radius_mm: float = 1.6,
    min_cluster_size: int = 6,
) -> list[dict[str, object]]:
    """Return compact centroid observations from candidate CT points."""
    pts = np.asarray(candidate_points_lps, dtype=float).reshape(-1, 3)
    clusters = _cluster_points_radius(pts, radius_mm=cluster_radius_mm, min_cluster_size=min_cluster_size)
    observations: list[dict[str, object]] = []
    for cluster_id, members in enumerate(clusters):
        cluster_pts = pts[members]
        center, axis = fit_axis_pca(cluster_pts)
        radial = point_line_distance(cluster_pts, center, axis)
        rel = cluster_pts - center
        t = rel @ axis
        observations.append(
            {
                "cluster_id": int(cluster_id),
                "center": center,
                "count": int(cluster_pts.shape[0]),
                "span_mm": float(np.max(t) - np.min(t)) if t.size else 0.0,
                "radial_rms_mm": float(np.sqrt(np.mean(radial**2))) if radial.size else 0.0,
            }
        )
    return observations


def _observation_segment_score(
    obs: dict[str, object],
    fit: ContactFitResult,
    *,
    radial_sigma_mm: float,
    projection_margin_mm: float,
) -> float:
    """Compatibility score of one compact observation with one fitted segment."""
    if not bool(fit.get("success")):
        return 0.0
    center = np.asarray(obs["center"], dtype=float)
    entry = np.asarray(fit["entry_lps"], dtype=float)
    target = np.asarray(fit["target_lps"], dtype=float)
    axis = unit(np.asarray(fit["deep_axis_lps"], dtype=float))
    length = float(np.linalg.norm(target - entry))
    proj = float(np.dot(center - entry, axis))
    if proj < -float(projection_margin_mm) or proj > length + float(projection_margin_mm):
        return 0.0
    radial = float(point_line_distance(np.asarray([center]), entry, axis)[0])
    count = math.sqrt(max(1.0, float(obs.get("count", 1.0) or 1.0)))
    return float(count * math.exp(-((radial / max(float(radial_sigma_mm), 1e-3)) ** 2)))


def _global_fit_score(
    fit_results_by_name: dict[str, ContactFitResult],
    observations: list[dict[str, object]],
    *,
    local_fit_prior_by_name: dict[str, ContactFitResult] | None = None,
    radial_sigma_mm: float = 1.25,
    projection_margin_mm: float = 6.0,
    deviation_penalty: float = 0.08,
) -> float:
    """Score a whole set of fitted segments jointly."""
    total = 0.0
    for obs in observations:
        best = 0.0
        for fit in fit_results_by_name.values():
            best = max(best, _observation_segment_score(obs, fit, radial_sigma_mm=radial_sigma_mm, projection_margin_mm=projection_margin_mm))
        total += float(best)
    if local_fit_prior_by_name:
        for name, fit in fit_results_by_name.items():
            base = local_fit_prior_by_name.get(name)
            if base is None or not bool(fit.get("success")) or not bool(base.get("success")):
                continue
            entry = np.asarray(fit["entry_lps"], dtype=float)
            target = np.asarray(fit["target_lps"], dtype=float)
            base_entry = np.asarray(base["entry_lps"], dtype=float)
            base_target = np.asarray(base["target_lps"], dtype=float)
            total -= float(deviation_penalty) * (
                float(np.linalg.norm(entry - base_entry)) + float(np.linalg.norm(target - base_target))
            )
    return float(total)


def _candidate_support_runs_for_fit(
    observations: list[dict[str, object]],
    fit: ContactFitResult,
    *,
    contact_offsets_mm: list[float] | np.ndarray | None = None,
    radial_limit_mm: float = 1.35,
    projection_margin_mm: float = 6.0,
    default_gap_mm: float = 4.5,
) -> list[dict[str, object]]:
    """Build candidate contiguous support runs along a fixed fitted axis."""
    if not bool(fit.get("success")):
        return []
    entry = np.asarray(fit["entry_lps"], dtype=float)
    axis = unit(np.asarray(fit["deep_axis_lps"], dtype=float))
    expected_gap = _contact_pitch_from_offsets(contact_offsets_mm)
    max_gap_mm = float(default_gap_mm if expected_gap is None else max(3.0, min(5.0, float(expected_gap) + 0.5)))
    allowed_gap_mm = _configured_gap_priors_mm(contact_offsets_mm)

    items: list[dict[str, object]] = []
    for obs in observations:
        center = np.asarray(obs["center"], dtype=float)
        proj = float(np.dot(center - entry, axis))
        radial = float(point_line_distance(np.asarray([center]), entry, axis)[0])
        if radial > float(radial_limit_mm):
            continue
        if proj < -float(projection_margin_mm):
            continue
        half_span = 0.5 * min(float(obs.get("span_mm", 0.0) or 0.0), 3.0)
        items.append(
            {
                "cluster_id": int(obs["cluster_id"]),
                "t_min_mm": float(proj - half_span),
                "t_max_mm": float(proj + half_span),
                "proj_mm": float(proj),
                "count": int(obs.get("count", 0) or 0),
            }
        )
    if not items:
        return []
    items.sort(key=lambda item: (float(item["t_min_mm"]), float(item["t_max_mm"])))
    runs: list[list[dict[str, object]]] = []
    current = [items[0]]
    current_end = float(items[0]["t_max_mm"])
    for item in items[1:]:
        gap = float(item["t_min_mm"]) - float(current_end)
        allowed_large_gap = any(abs(float(gap) - float(cfg)) <= 1.25 for cfg in allowed_gap_mm)
        if gap <= float(max_gap_mm) or allowed_large_gap:
            current.append(item)
            current_end = max(current_end, float(item["t_max_mm"]))
        else:
            runs.append(current)
            current = [item]
            current_end = float(item["t_max_mm"])
    runs.append(current)
    out: list[dict[str, object]] = []
    for idx, run in enumerate(runs):
        out.append(
            {
                "run_index": int(idx),
                "cluster_ids": [int(item["cluster_id"]) for item in run],
                "t_min_mm": float(min(float(item["t_min_mm"]) for item in run)),
                "t_max_mm": float(max(float(item["t_max_mm"]) for item in run)),
                "cluster_count": int(len(run)),
            }
        )
    return out


def _with_segment_run(
    fit: ContactFitResult,
    run: dict[str, object],
) -> ContactFitResult:
    """Return a copy of `fit` with endpoints replaced by a fixed-axis support run."""
    new_fit = dict(fit)
    center = np.asarray(fit["center_lps"], dtype=float)
    axis = unit(np.asarray(fit["deep_axis_lps"], dtype=float))
    entry = center + axis * float(run["t_min_mm"])
    target = center + axis * float(run["t_max_mm"])
    base_target = np.asarray(fit["target_lps"], dtype=float)
    delta_target = target - base_target
    lateral_vec = delta_target - np.dot(delta_target, axis) * axis
    new_fit["entry_lps"] = entry.tolist()
    new_fit["target_lps"] = target.tolist()
    new_fit["center_lps"] = (0.5 * (entry + target)).tolist()
    new_fit["tip_shift_mm"] = float(np.dot(delta_target, axis))
    new_fit["lateral_shift_mm"] = float(np.linalg.norm(lateral_vec))
    new_fit["deep_anchor_source"] = "global_coordinate_ascent"
    new_fit["axis_support_selected_run_json"] = json.dumps(run, separators=(",", ":"))
    new_fit["terminal_anchor_mode"] = "coordinate_ascent_run"
    return new_fit


def refine_fit_batch_with_global_coordinate_ascent(
    fit_results_by_name: dict[str, ContactFitResult],
    *,
    candidate_points_lps: np.ndarray | list[list[float]],
    contact_offsets_by_name: dict[str, list[float] | np.ndarray | None] | None = None,
    radial_limit_mm: float = 1.35,
    projection_margin_mm: float = 6.0,
    max_passes: int = 2,
) -> dict[str, ContactFitResult]:
    """Coordinate-ascent refinement: local fits first, then global run selection."""
    observations = build_compact_centroid_observations(candidate_points_lps)
    if not observations:
        return dict(fit_results_by_name)
    refined = {name: dict(fit) for name, fit in fit_results_by_name.items()}
    local_reference = {name: dict(fit) for name, fit in fit_results_by_name.items()}
    current_score = _global_fit_score(refined, observations, local_fit_prior_by_name=local_reference, projection_margin_mm=projection_margin_mm)
    for _ in range(int(max_passes)):
        improved = False
        for name, fit in list(refined.items()):
            if not bool(fit.get("success")):
                continue
            runs = _candidate_support_runs_for_fit(
                observations,
                fit,
                contact_offsets_mm=None if contact_offsets_by_name is None else contact_offsets_by_name.get(name),
                radial_limit_mm=radial_limit_mm,
                projection_margin_mm=projection_margin_mm,
            )
            if not runs:
                continue
            candidates = [dict(fit)] + [_with_segment_run(fit, run) for run in runs]
            best_fit = dict(fit)
            best_score = current_score
            for candidate in candidates[1:]:
                trial = dict(refined)
                trial[name] = candidate
                trial_score = _global_fit_score(
                    trial,
                    observations,
                    local_fit_prior_by_name=local_reference,
                    projection_margin_mm=projection_margin_mm,
                )
                if trial_score > best_score + 1e-6:
                    best_score = trial_score
                    best_fit = candidate
            if best_fit is not fit and best_score > current_score + 1e-6:
                refined[name] = best_fit
                current_score = best_score
                improved = True
                refined[name]["global_coordinate_ascent_score"] = float(best_score)
        if not improved:
            break
    return refined


def refine_fit_batch_with_exclusive_terminal_assignment(
    fit_results_by_name: dict[str, ContactFitResult],
    *,
    candidate_points_lps: np.ndarray | list[list[float]],
    contact_offsets_by_name: dict[str, list[float] | np.ndarray | None] | None = None,
    max_line_distance_mm: float = 0.9,
    ambiguity_margin_mm: float = 0.25,
    projection_margin_mm: float = 5.0,
) -> dict[str, ContactFitResult]:
    """Refine successful fits with shared terminal eligibility and clear-dominance veto.

    Interior support can be ambiguous across nearby trajectories, but terminal
    clipping is especially sensitive to over-hard assignment. This pass keeps
    terminal centroids eligible for multiple trajectories unless another
    trajectory clearly dominates them.
    """
    observations = build_compact_centroid_observations(candidate_points_lps)
    if not observations:
        return dict(fit_results_by_name)

    fit_names = [name for name, fit in fit_results_by_name.items() if bool(fit.get("success"))]
    if len(fit_names) < 2:
        return dict(fit_results_by_name)

    compatible_by_name: dict[str, list[dict[str, object]]] = {name: [] for name in fit_names}
    pitch_by_name: dict[str, float | None] = {}
    gap_limit_by_name: dict[str, float] = {}
    for name in fit_names:
        contact_offsets_mm = None if contact_offsets_by_name is None else contact_offsets_by_name.get(name)
        expected_pitch_mm = _contact_pitch_from_offsets(contact_offsets_mm)
        pitch_by_name[name] = expected_pitch_mm
        gap_limit_by_name[name] = max(3.0, min(5.0, float(expected_pitch_mm) + 0.5)) if expected_pitch_mm is not None else 4.5

    for obs in observations:
        center = np.asarray(obs["center"], dtype=float)
        for name in fit_names:
            fit = fit_results_by_name[name]
            entry = np.asarray(fit["entry_lps"], dtype=float)
            target = np.asarray(fit["target_lps"], dtype=float)
            deep_axis = unit(np.asarray(fit["deep_axis_lps"], dtype=float))
            length = float(np.linalg.norm(target - entry))
            proj = float(np.dot(center - entry, deep_axis))
            if proj < -float(projection_margin_mm) or proj > length + float(projection_margin_mm):
                continue
            line_dist = float(point_line_distance(np.asarray([center]), target, deep_axis)[0])
            if line_dist > float(max_line_distance_mm):
                continue
            compatible_by_name[name].append(
                {
                    "obs": obs,
                    "center": center,
                    "proj": proj,
                    "line_dist": line_dist,
                }
            )

    score_by_obs_and_name: dict[int, dict[str, float]] = {}
    for name in fit_names:
        items = sorted(compatible_by_name[name], key=lambda item: float(item["proj"]))
        expected_pitch_mm = pitch_by_name[name]
        gap_limit_mm = gap_limit_by_name[name]
        for idx, item in enumerate(items):
            line_dist = float(item["line_dist"])
            proj = float(item["proj"])
            prox_score = math.exp(-((line_dist / max(float(max_line_distance_mm), 1e-3)) ** 2))
            gap_score = 0.5
            dir_score = 0.5
            if idx >= 1:
                prev = items[idx - 1]
                gap = max(0.0, proj - float(prev["proj"]))
                if gap <= float(gap_limit_mm):
                    if expected_pitch_mm is not None:
                        sigma_gap = max(0.75, 0.35 * float(expected_pitch_mm))
                        gap_score = math.exp(-(((gap - float(expected_pitch_mm)) / sigma_gap) ** 2))
                    else:
                        gap_score = math.exp(-((gap / max(float(gap_limit_mm), 1e-3)) ** 2))
                else:
                    gap_score = 0.0
            if idx >= 2:
                prev = items[idx - 1]
                prev2 = items[idx - 2]
                step_prev = np.asarray(prev["center"], dtype=float) - np.asarray(prev2["center"], dtype=float)
                step_curr = np.asarray(item["center"], dtype=float) - np.asarray(prev["center"], dtype=float)
                if float(np.linalg.norm(step_prev)) > 1e-6 and float(np.linalg.norm(step_curr)) > 1e-6:
                    dir_score = max(0.0, float(np.dot(unit(step_prev), unit(step_curr))))
            total_score = (2.0 * prox_score) + (1.25 * gap_score) + (0.75 * dir_score)
            score_by_obs_and_name.setdefault(int(item["obs"]["cluster_id"]), {})[name] = float(total_score)

    dominance_margin = 0.9
    assigned_by_name: dict[str, list[dict[str, object]]] = {name: [] for name in fit_names}
    veto_count_by_name: dict[str, int] = {name: 0 for name in fit_names}
    for name in fit_names:
        for item in compatible_by_name[name]:
            cluster_id = int(item["obs"]["cluster_id"])
            scores = score_by_obs_and_name.get(cluster_id, {})
            my_score = float(scores.get(name, 0.0))
            if scores:
                best_name, best_score = max(scores.items(), key=lambda kv: float(kv[1]))
                if best_name != name and (float(best_score) - float(my_score)) > float(dominance_margin):
                    veto_count_by_name[name] += 1
                    continue
            assigned_by_name[name].append(item["obs"])

    refined = dict(fit_results_by_name)
    for name in fit_names:
        fit = dict(fit_results_by_name[name])
        assigned = assigned_by_name.get(name) or []
        if len(assigned) < 3:
            fit["exclusive_terminal_assignment_count"] = int(len(assigned))
            refined[name] = fit
            continue
        center = np.asarray(fit["center_lps"], dtype=float)
        deep_axis = unit(np.asarray(fit["deep_axis_lps"], dtype=float))
        target = np.asarray(fit["target_lps"], dtype=float)
        entry = np.asarray(fit["entry_lps"], dtype=float)
        planned_length = float(np.linalg.norm(target - entry))
        contact_offsets_mm = None if contact_offsets_by_name is None else contact_offsets_by_name.get(name)
        expected_pitch_mm = pitch_by_name.get(name)
        gap_limit_mm = gap_limit_by_name.get(name, 4.5)
        t_mm, chain_count = _deepest_directional_terminal_centroid_t(
            assigned,
            line_point=center,
            line_axis=deep_axis,
            radial_limit_mm=float(max_line_distance_mm),
            max_gap_mm=gap_limit_mm,
            min_chain_clusters=3,
        )
        fit["exclusive_terminal_assignment_count"] = int(len(assigned))
        fit["exclusive_terminal_chain_count"] = int(chain_count)
        fit["exclusive_terminal_veto_count"] = int(veto_count_by_name.get(name) or 0)
        fit["exclusive_assignment_mode"] = "shared_terminal_with_veto"
        if t_mm is None:
            refined[name] = fit
            continue
        new_target = center + deep_axis * float(t_mm)
        new_entry = new_target - deep_axis * planned_length
        delta_target = new_target - target
        fit["entry_lps"] = new_entry.tolist()
        fit["target_lps"] = new_target.tolist()
        fit["tip_shift_mm"] = float(np.dot(delta_target, deep_axis))
        lateral_vec = delta_target - np.dot(delta_target, deep_axis) * deep_axis
        fit["lateral_shift_mm"] = float(np.linalg.norm(lateral_vec))
        fit["deep_anchor_source"] = "exclusive_terminal_assignment"
        refined[name] = fit
    return refined


def _build_fit_result(
    *,
    roi_pts: np.ndarray,
    planned_entry_lps: np.ndarray,
    planned_target_lps: np.ndarray,
    contact_offsets_mm: list[float] | np.ndarray | None,
    tip_at: str,
    center: np.ndarray,
    fit_deep_axis: np.ndarray,
    planned_length: float,
    max_depth_shift_mm: float,
    angle_limit_deg: float,
    fit_mode_used: str,
    extra_metrics: dict[str, object] | None = None,
    deep_anchor_t_override_mm: float | None = None,
    deep_anchor_source_override: str | None = None,
    proximal_anchor_t_override_mm: float | None = None,
) -> ContactFitResult:
    """Build final fit result once axis and deep anchor model are determined."""
    entry = np.asarray(planned_entry_lps, dtype=float)
    target = np.asarray(planned_target_lps, dtype=float)
    planned_tip, planned_axis = _planned_tip_and_axis(entry, target, tip_at=tip_at)
    planned_deep_axis = unit(target - entry)
    if float(np.dot(fit_deep_axis, planned_deep_axis)) < 0.0:
        fit_deep_axis = -fit_deep_axis
    fit_super_axis = -fit_deep_axis
    ang = angle_deg(planned_deep_axis, fit_deep_axis)
    if ang > float(angle_limit_deg):
        result: ContactFitResult = {
            "success": False,
            "reason": f"Angle deviation {ang:.2f} deg exceeds max {angle_limit_deg:.2f}",
            "angle_deg": float(ang),
            "points_in_roi": int(roi_pts.shape[0]),
            "fit_mode_used": str(fit_mode_used),
        }
        if extra_metrics:
            result.update(extra_metrics)
        return result

    projected_t = (roi_pts - center) @ fit_deep_axis
    planned_target_t = float(np.dot(target - center, fit_deep_axis))
    deep_min = planned_target_t - float(max_depth_shift_mm)
    deep_max = planned_target_t + float(max_depth_shift_mm)
    near_target = (projected_t >= deep_min) & (projected_t <= deep_max)
    if deep_anchor_t_override_mm is not None:
        deep_t_raw = float(deep_anchor_t_override_mm)
        deep_anchor_source = str(deep_anchor_source_override or "override")
    elif np.any(near_target):
        deep_t_raw = float(np.max(projected_t[near_target]))
        deep_anchor_source = "roi_near_target"
    else:
        deep_t_raw = float(np.quantile(projected_t, 0.95))
        deep_anchor_source = "roi_quantile_fallback"
    # For CT-refined fitting we use max_depth_shift_mm as a search prior, not a hard
    # endpoint clamp. The selected support should be allowed to win if it is farther
    # than the original proposal expected.
    deep_t = float(deep_t_raw)

    fitted_target = center + fit_deep_axis * deep_t
    if proximal_anchor_t_override_mm is not None:
        fitted_entry = center + fit_deep_axis * float(proximal_anchor_t_override_mm)
        fitted_span_mm = float(np.linalg.norm(fitted_target - fitted_entry))
    else:
        fitted_entry = fitted_target - fit_deep_axis * planned_length
        fitted_span_mm = float(planned_length)

    offs = np.asarray(contact_offsets_mm if contact_offsets_mm is not None else [], dtype=float).reshape(-1)
    tip_at_norm = (tip_at or "target").lower()
    if tip_at_norm == "entry":
        fitted_tip = fitted_entry
        offsets_axis = fit_deep_axis
    else:
        fitted_tip = fitted_target
        offsets_axis = fit_super_axis
    if offs.size > 0:
        pred_centers = fitted_tip[None, :] + np.outer(offs, offsets_axis)
        diff = pred_centers[:, None, :] - roi_pts[None, :, :]
        d3 = np.linalg.norm(diff, axis=2)
        residual_3d = float(np.mean(np.min(d3, axis=1)))
        cand_t = (roi_pts - fitted_target) @ fit_deep_axis
        pred_t = (pred_centers - fitted_target) @ fit_deep_axis
        nearest_1d = np.min(np.abs(cand_t[None, :] - pred_t[:, None]), axis=1)
        one_d_residual = float(np.mean(nearest_1d))
    else:
        residual_3d = 0.0
        one_d_residual = float("nan")

    target_shift_mm = float(np.dot(fitted_target - target, fit_deep_axis))
    delta_target = fitted_target - target
    lateral_vec = delta_target - np.dot(delta_target, fit_deep_axis) * fit_deep_axis
    lateral_shift_mm = float(np.linalg.norm(lateral_vec))

    result: ContactFitResult = {
        "success": True,
        "entry_lps": fitted_entry.tolist(),
        "target_lps": fitted_target.tolist(),
        "tip_shift_mm": target_shift_mm,
        "lateral_shift_mm": lateral_shift_mm,
        "angle_deg": float(ang),
        "residual_mm": residual_3d,
        "one_d_residual_mm": float(one_d_residual),
        "points_in_roi": int(roi_pts.shape[0]),
        "fit_mode_used": str(fit_mode_used),
        "axis_lps": fit_super_axis.tolist(),
        "deep_axis_lps": fit_deep_axis.tolist(),
        "center_lps": center.tolist(),
        "deep_t_raw_mm": deep_t_raw,
        "deep_t_clamped_mm": deep_t,
        "planned_target_t_mm": planned_target_t,
        "proximal_t_override_mm": "" if proximal_anchor_t_override_mm is None else float(proximal_anchor_t_override_mm),
        "fitted_span_mm": float(fitted_span_mm),
        "planned_tip_lps": planned_tip.tolist(),
        "planned_tip_axis_lps": planned_axis.tolist(),
        "deep_anchor_source": deep_anchor_source,
    }
    if extra_metrics:
        result.update(extra_metrics)
    return result


def _fit_electrode_axis_and_tip_em_v1(
    candidate_points_lps: np.ndarray | list[list[float]],
    planned_entry_lps: Point3D,
    planned_target_lps: Point3D,
    contact_offsets_mm: list[float] | np.ndarray | None = None,
    tip_at: str = "target",
    roi_radius_mm: float = 5.0,
    max_angle_deg: float = 12.0,
    max_depth_shift_mm: float = 2.0,
) -> ContactFitResult:
    """EM-style local registration of one proposed trajectory to nearby CT support objects."""
    pts = np.asarray(candidate_points_lps, dtype=float).reshape(-1, 3)
    if pts.size == 0:
        return {"success": False, "reason": "No candidate points", "fit_mode_used": "em_v1"}

    entry = np.asarray(planned_entry_lps, dtype=float)
    target = np.asarray(planned_target_lps, dtype=float)
    planned_length = float(np.linalg.norm(entry - target))
    if planned_length <= 1e-6:
        return {"success": False, "reason": "Zero-length planned trajectory", "fit_mode_used": "em_v1"}

    cfg = GuidedFitEMConfig(roi_radius_mm=max(float(roi_radius_mm), 5.0))
    planned_deep_axis = unit(target - entry)
    planned_center = 0.5 * (entry + target)
    configured_gap_mm = _configured_gap_priors_mm(contact_offsets_mm)

    roi_pts = filter_points_in_segment_cylinder(
        pts,
        seg_start=entry,
        seg_end=target,
        radius_mm=float(cfg.roi_radius_mm),
        margin_mm=float(max_depth_shift_mm) + float(cfg.assignment_margin_mm),
    )
    if roi_pts.shape[0] < 24:
        return {
            "success": False,
            "reason": "Too few CT points in ROI",
            "points_in_roi": int(roi_pts.shape[0]),
            "fit_mode_used": "em_v1",
        }

    clusters = _cluster_points_radius(
        roi_pts,
        radius_mm=float(cfg.cluster_radius_mm),
        min_cluster_size=int(cfg.min_cluster_size),
    )
    cluster_summaries = _summarize_clusters(roi_pts, clusters, planned_center, planned_deep_axis)
    if not cluster_summaries:
        return {
            "success": False,
            "reason": "No support objects in ROI",
            "points_in_roi": int(roi_pts.shape[0]),
            "fit_mode_used": "em_v1",
        }

    if len(cluster_summaries) == 1:
        string_fit = _fit_single_cluster_string_axis(
            cluster_summaries[0],
            planned_deep_axis=planned_deep_axis,
            min_span_mm=12.0,
            max_radial_rms_mm=1.4,
        )
        if string_fit is not None:
            center, axis = string_fit
            selected_support_run = {
                "cluster_ids": [int(cluster_summaries[0]["cluster_id"])],
                "t_min_mm": float(_cluster_axis_interval(cluster_summaries[0], line_point=center, line_axis=axis, radial_limit_mm=float(cfg.support_radial_limit_mm))["t_min_mm"]),
                "t_max_mm": float(_cluster_axis_interval(cluster_summaries[0], line_point=center, line_axis=axis, radial_limit_mm=float(cfg.support_radial_limit_mm))["t_max_mm"]),
                "core_hits": 1,
                "cluster_count": 1,
            }
            result = _build_fit_result(
                roi_pts=roi_pts,
                planned_entry_lps=entry,
                planned_target_lps=target,
                contact_offsets_mm=contact_offsets_mm,
                tip_at=tip_at,
                center=center,
                fit_deep_axis=axis,
                planned_length=planned_length,
                max_depth_shift_mm=max_depth_shift_mm,
                angle_limit_deg=max_angle_deg,
                fit_mode_used="em_v1",
                deep_anchor_t_override_mm=float(selected_support_run["t_max_mm"]),
                deep_anchor_source_override="em_support_run",
                proximal_anchor_t_override_mm=float(selected_support_run["t_min_mm"]),
                extra_metrics={
                    "cluster_count": 1,
                    "em_axis_iterations": 1,
                    "em_axis_score_history_json": "[1.0]",
                    "em_assigned_cluster_ids_json": json.dumps([int(cluster_summaries[0]["cluster_id"])]),
                    "axis_support_selected_run_json": json.dumps(selected_support_run, separators=(",", ":")),
                    "axis_support_runs_json": json.dumps([selected_support_run], separators=(",", ":")),
                    "configured_gap_priors_mm": json.dumps(configured_gap_mm, separators=(",", ":")),
                    "terminal_blob_source": "em_support_objects",
                    "terminal_anchor_mode": "support_run_interval",
                    "local_terminal_morphology": "em_support_run",
                    "em_config_json": json.dumps(asdict(cfg), separators=(",", ":")),
                },
            )
            result["fit_mode_requested"] = "em_v1"
            return result

    center = planned_center.copy()
    axis = planned_deep_axis.copy()
    score_history: list[float] = []
    assigned_ids: set[int] = set()
    selected_support_run: dict[str, object] | None = None
    axis_support_runs: list[dict[str, object]] = []

    for _ in range(int(cfg.axis_em_iters)):
        t_centers = np.asarray(
            [float(np.dot(np.asarray(item["center"], dtype=float) - center, axis)) for item in cluster_summaries],
            dtype=float,
        )
        radial = point_line_distance(
            np.asarray([np.asarray(item["center"], dtype=float) for item in cluster_summaries], dtype=float),
            center,
            axis,
        )
        within_t = np.abs(t_centers - float(np.dot(target - center, axis))) <= (planned_length * 0.75 + float(cfg.assignment_margin_mm))
        compatible = (radial <= float(cfg.support_radial_limit_mm) * 1.5) & within_t
        assigned = [item for item, keep in zip(cluster_summaries, compatible.tolist()) if keep]
        if len(assigned) < int(cfg.min_clusters_for_fit):
            break

        compact_assigned = [
            item for item in assigned
            if float(item.get("span_mm", 0.0) or 0.0) <= 4.5 and float(item.get("radial_rms_mm", 0.0) or 0.0) <= 1.2
        ]
        if len(compact_assigned) >= int(cfg.min_clusters_for_fit):
            center, axis = _fit_axis_from_interior_cluster_centers(
                compact_assigned,
                planned_deep_axis=planned_deep_axis,
            )
            assigned_ids = {int(item["cluster_id"]) for item in compact_assigned}
        else:
            dominant = max(assigned, key=lambda item: float(item.get("count", 0) or 0.0))
            string_fit = _fit_single_cluster_string_axis(
                dominant,
                planned_deep_axis=planned_deep_axis,
                min_span_mm=12.0,
                max_radial_rms_mm=1.4,
            )
            if string_fit is None:
                assigned_centers = np.asarray([np.asarray(item["center"], dtype=float) for item in assigned], dtype=float)
                weights = np.asarray([math.sqrt(max(1.0, float(item.get("count", 1.0)))) for item in assigned], dtype=float)
                center, axis = _weighted_fit_axis_pca(assigned_centers, weights)
                if float(np.dot(axis, planned_deep_axis)) < 0.0:
                    axis = -axis
            else:
                center, axis = string_fit
            assigned_ids = {int(item["cluster_id"]) for item in assigned}

        axis_support = _axis_support_run_from_clusters(
            cluster_summaries,
            core_cluster_ids=assigned_ids,
            line_point=center,
            line_axis=axis,
            planned_entry_t_mm=float(np.dot(entry - center, axis)),
            planned_target_t_mm=float(np.dot(target - center, axis)),
            radial_limit_mm=float(cfg.support_radial_limit_mm),
            max_gap_mm=float(cfg.max_gap_mm),
            allowed_gap_mm=configured_gap_mm,
            elongated_direction_cos_min=0.76,
        )
        selected_support_run = axis_support["selected_run"]
        axis_support_runs = axis_support["runs"]
        if selected_support_run is None:
            axis_support = _axis_support_run_from_clusters(
                cluster_summaries,
                core_cluster_ids=assigned_ids,
                line_point=center,
                line_axis=axis,
                planned_entry_t_mm=float(np.dot(entry - center, axis)),
                planned_target_t_mm=float(np.dot(target - center, axis)),
                radial_limit_mm=float(cfg.support_radial_limit_mm) * 1.35,
                max_gap_mm=float(cfg.max_gap_mm),
                allowed_gap_mm=configured_gap_mm,
                elongated_direction_cos_min=0.7,
            )
            selected_support_run = axis_support["selected_run"]
            axis_support_runs = axis_support["runs"]
        if selected_support_run is None:
            break
        score_history.append(float(selected_support_run.get("score", 0.0)))

    if selected_support_run is None:
        axis_support = _axis_support_run_from_clusters(
            cluster_summaries,
            core_cluster_ids={int(item["cluster_id"]) for item in cluster_summaries},
            line_point=center,
            line_axis=axis,
            planned_entry_t_mm=float(np.dot(entry - center, axis)),
            planned_target_t_mm=float(np.dot(target - center, axis)),
            radial_limit_mm=float(cfg.support_radial_limit_mm) * 1.5,
            max_gap_mm=float(cfg.max_gap_mm),
            allowed_gap_mm=configured_gap_mm,
            elongated_direction_cos_min=0.68,
        )
        selected_support_run = axis_support["selected_run"]
        axis_support_runs = axis_support["runs"]
    if selected_support_run is None:
        fallback = fit_electrode_axis_and_tip(
            candidate_points_lps=pts,
            planned_entry_lps=planned_entry_lps,
            planned_target_lps=planned_target_lps,
            contact_offsets_mm=contact_offsets_mm,
            tip_at=tip_at,
            roi_radius_mm=roi_radius_mm,
            max_angle_deg=max_angle_deg,
            max_depth_shift_mm=max_depth_shift_mm,
            fit_mode="slab_v1",
        )
        fallback["fit_mode_used"] = "em_v1_fallback"
        fallback["fit_mode_requested"] = "em_v1"
        return fallback

    if int(selected_support_run.get("cluster_count", 0) or 0) <= 1 and len(cluster_summaries) >= 3:
        alt_support = _axis_support_run_from_clusters(
            cluster_summaries,
            core_cluster_ids={int(item["cluster_id"]) for item in cluster_summaries},
            line_point=center,
            line_axis=axis,
            planned_entry_t_mm=float(np.dot(entry - center, axis)),
            planned_target_t_mm=float(np.dot(target - center, axis)),
            radial_limit_mm=float(cfg.support_radial_limit_mm) * 2.0,
            max_gap_mm=float(cfg.max_gap_mm) + 1.0,
            allowed_gap_mm=configured_gap_mm,
            elongated_direction_cos_min=0.65,
        )
        if alt_support["selected_run"] is not None and float(alt_support["selected_run"].get("score", float("-inf")) or float("-inf")) > float(selected_support_run.get("score", float("-inf")) or float("-inf")):
            selected_support_run = alt_support["selected_run"]
            axis_support_runs = alt_support["runs"]

    selected_ids = {int(x) for x in selected_support_run["cluster_ids"]}
    selected_clusters = [item for item in cluster_summaries if int(item["cluster_id"]) in selected_ids]
    if len(selected_clusters) >= 2:
        selected_centers = np.asarray([np.asarray(item["center"], dtype=float) for item in selected_clusters], dtype=float)
        selected_weights = np.asarray([math.sqrt(max(1.0, float(item.get("count", 1.0)))) for item in selected_clusters], dtype=float)
        center, axis = _weighted_fit_axis_pca(selected_centers, selected_weights)
        if float(np.dot(axis, planned_deep_axis)) < 0.0:
            axis = -axis
        axis_support = _axis_support_run_from_clusters(
            cluster_summaries,
            core_cluster_ids=selected_ids,
            line_point=center,
            line_axis=axis,
            planned_entry_t_mm=float(np.dot(entry - center, axis)),
            planned_target_t_mm=float(np.dot(target - center, axis)),
            radial_limit_mm=float(cfg.support_radial_limit_mm),
            max_gap_mm=float(cfg.max_gap_mm),
            allowed_gap_mm=configured_gap_mm,
            elongated_direction_cos_min=0.72,
        )
        if axis_support["selected_run"] is not None:
            selected_support_run = axis_support["selected_run"]
            axis_support_runs = axis_support["runs"]

    distal_t_mm = float(selected_support_run["t_max_mm"])
    proximal_t_mm = float(selected_support_run["t_min_mm"])
    result = _build_fit_result(
        roi_pts=roi_pts,
        planned_entry_lps=entry,
        planned_target_lps=target,
        contact_offsets_mm=contact_offsets_mm,
        tip_at=tip_at,
        center=center,
        fit_deep_axis=axis,
        planned_length=planned_length,
        max_depth_shift_mm=max_depth_shift_mm,
        angle_limit_deg=max_angle_deg,
        fit_mode_used="em_v1",
        deep_anchor_t_override_mm=distal_t_mm,
        deep_anchor_source_override="em_support_run",
        proximal_anchor_t_override_mm=proximal_t_mm,
        extra_metrics={
            "cluster_count": int(len(clusters)),
            "em_axis_iterations": int(max(1, len(score_history))),
            "em_axis_score_history_json": json.dumps([round(float(x), 4) for x in score_history], separators=(",", ":")),
            "em_assigned_cluster_ids_json": json.dumps(sorted(int(x) for x in assigned_ids), separators=(",", ":")),
            "axis_support_selected_run_json": json.dumps(selected_support_run, separators=(",", ":")),
            "axis_support_runs_json": json.dumps(axis_support_runs, separators=(",", ":")),
            "configured_gap_priors_mm": json.dumps(configured_gap_mm, separators=(",", ":")),
            "terminal_blob_source": "em_support_objects",
            "terminal_anchor_mode": "support_run_interval",
            "local_terminal_morphology": "em_support_run",
            "em_config_json": json.dumps(asdict(cfg), separators=(",", ":")),
        },
    )
    result["fit_mode_requested"] = "em_v1"
    return result


def _fit_electrode_axis_and_tip_deep_anchored(
    candidate_points_lps: np.ndarray | list[list[float]],
    planned_entry_lps: Point3D,
    planned_target_lps: Point3D,
    contact_offsets_mm: list[float] | np.ndarray | None = None,
    tip_at: str = "target",
    roi_radius_mm: float = 5.0,
    max_angle_deg: float = 12.0,
    max_depth_shift_mm: float = 2.0,
) -> ContactFitResult:
    """Deep-end anchored guided fit that treats the planned line as a search prior."""
    pts = np.asarray(candidate_points_lps, dtype=float).reshape(-1, 3)
    if pts.size == 0:
        return {"success": False, "reason": "No candidate points", "fit_mode_used": "deep_anchor_v2"}

    entry = np.asarray(planned_entry_lps, dtype=float)
    target = np.asarray(planned_target_lps, dtype=float)
    planned_length = float(np.linalg.norm(entry - target))
    if planned_length <= 1e-6:
        return {"success": False, "reason": "Zero-length planned trajectory", "fit_mode_used": "deep_anchor_v2"}
    planned_deep_axis = unit(target - entry)
    configured_gap_mm = _configured_gap_priors_mm(contact_offsets_mm)

    roi_pts = filter_points_in_segment_cylinder(
        pts,
        seg_start=entry,
        seg_end=target,
        radius_mm=float(max(roi_radius_mm, 5.0)),
        margin_mm=float(max_depth_shift_mm) + 5.0,
    )
    if roi_pts.shape[0] < 24:
        return {
            "success": False,
            "reason": "Too few CT points in ROI",
            "points_in_roi": int(roi_pts.shape[0]),
            "fit_mode_used": "deep_anchor_v2",
        }

    clusters = _cluster_points_radius(roi_pts, radius_mm=1.6, min_cluster_size=6)
    cluster_summaries = _summarize_clusters(roi_pts, clusters, entry, planned_deep_axis)
    compact_clusters = [
        summary for summary in cluster_summaries
        if float(summary["span_mm"]) <= 4.5 and float(summary["radial_rms_mm"]) <= 1.2
    ]
    if len(clusters) == 1 and len(cluster_summaries) == 1 and len(compact_clusters) == 0:
        string_fit = _fit_single_cluster_string_axis(
            cluster_summaries[0],
            planned_deep_axis=planned_deep_axis,
            min_span_mm=18.0,
            max_radial_rms_mm=1.1,
        )
        if string_fit is not None:
            center, fit_axis = string_fit
            planned_target_t_mm = float(np.dot(target - center, fit_axis))
            axis_support = _axis_support_run_from_clusters(
                cluster_summaries,
                core_cluster_ids={int(cluster_summaries[0]["cluster_id"])},
                line_point=center,
                line_axis=fit_axis,
                planned_entry_t_mm=float(np.dot(entry - center, fit_axis)),
                planned_target_t_mm=float(np.dot(target - center, fit_axis)),
                radial_limit_mm=1.2,
                max_gap_mm=4.5,
                allowed_gap_mm=configured_gap_mm,
                elongated_direction_cos_min=0.72,
            )
            selected_support_run = axis_support["selected_run"]
            extent_t_mm = None if selected_support_run is None else float(selected_support_run["t_max_mm"])
            support_run_diag = _support_run_diagnostics(
                roi_pts,
                line_point=center,
                line_axis=fit_axis,
                seed_t_mm=float(extent_t_mm if extent_t_mm is not None else (planned_target_t_mm - 2.0)),
                search_end_t_mm=planned_target_t_mm + 3.0,
                radial_limit_mm=1.0,
                bin_mm=0.5,
                min_points_per_bin=3,
                void_mm=5.0,
                max_gap_mm=4.5,
                allowed_gap_mm=configured_gap_mm,
            )
            diag_extent_t_mm = support_run_diag["selected_endpoint_t_mm"]
            if diag_extent_t_mm is not None and (extent_t_mm is None or float(diag_extent_t_mm) > float(extent_t_mm)):
                extent_t_mm = float(diag_extent_t_mm)
            if extent_t_mm is None:
                extent_t_mm = _cluster_distal_extent_t(
                    cluster_summaries[0],
                    line_point=center,
                    line_axis=fit_axis,
                    distal_quantile=0.98,
                )
            result = _build_fit_result(
                roi_pts=roi_pts,
                planned_entry_lps=entry,
                planned_target_lps=target,
                contact_offsets_mm=contact_offsets_mm,
                tip_at=tip_at,
                center=center,
                fit_deep_axis=fit_axis,
                planned_length=planned_length,
                max_depth_shift_mm=max_depth_shift_mm,
                angle_limit_deg=max_angle_deg,
                fit_mode_used="deep_anchor_v2",
                deep_anchor_t_override_mm=extent_t_mm,
                deep_anchor_source_override="string_terminal_extent" if extent_t_mm is not None else None,
                proximal_anchor_t_override_mm=None if selected_support_run is None else float(selected_support_run["t_min_mm"]),
                extra_metrics={
                    "cluster_count": int(len(clusters)),
                    "compact_cluster_count": int(len(compact_clusters)),
                    "string_cluster_span_mm": float(cluster_summaries[0]["span_mm"]),
                    "string_cluster_radial_rms_mm": float(cluster_summaries[0]["radial_rms_mm"]),
                    "terminal_blob_source": "single_string_cluster",
                    "terminal_anchor_mode": "tight_extent",
                    "local_terminal_morphology": "string_like",
                    "support_run_span_mm": "" if selected_support_run is None else round(float(selected_support_run["t_max_mm"]) - float(selected_support_run["t_min_mm"]), 4),
                    "local_terminal_cluster_count": 1,
                    "terminal_blob_candidate_count": 1,
                    "terminal_blob_selected_cluster_id": int(cluster_summaries[0]["cluster_id"]),
                    "terminal_blob_selected_anchor_t_mm": "" if extent_t_mm is None else float(extent_t_mm),
                    "configured_gap_priors_mm": json.dumps(configured_gap_mm, separators=(",", ":")),
                    "axis_support_selected_run_json": "null" if selected_support_run is None else json.dumps(selected_support_run, separators=(",", ":")),
                    "axis_support_runs_json": json.dumps(axis_support["runs"], separators=(",", ":")),
                    "terminal_support_run_count": int(len(support_run_diag["runs"])),
                    "terminal_support_longest_run_index": "" if support_run_diag["longest_run_index"] is None else int(support_run_diag["longest_run_index"]),
                    "terminal_support_selected_run_index": "" if support_run_diag["selected_run_index"] is None else int(support_run_diag["selected_run_index"]),
                    "terminal_support_runs_json": json.dumps(support_run_diag["runs"], separators=(",", ":")),
                    "terminal_blob_diagnostics_json": json.dumps(
                        [
                            {
                                "cluster_id": int(cluster_summaries[0]["cluster_id"]),
                                "center_t_mm": round(float(np.dot(np.asarray(cluster_summaries[0]["center"], dtype=float) - center, fit_axis)), 4),
                                "distal_t_mm": "" if extent_t_mm is None else round(float(extent_t_mm), 4),
                                "span_mm": round(float(cluster_summaries[0]["span_mm"]), 4),
                                "radial_rms_mm": round(float(cluster_summaries[0]["radial_rms_mm"]), 4),
                                "selected": bool(extent_t_mm is not None),
                            }
                        ],
                        separators=(",", ":"),
                    ),
                },
            )
            result["fit_mode_requested"] = "deep_anchor_v2"
            return result
    if len(clusters) < 3:
        fallback = fit_electrode_axis_and_tip(
            candidate_points_lps=pts,
            planned_entry_lps=planned_entry_lps,
            planned_target_lps=planned_target_lps,
            contact_offsets_mm=contact_offsets_mm,
            tip_at=tip_at,
            roi_radius_mm=roi_radius_mm,
            max_angle_deg=max_angle_deg,
            max_depth_shift_mm=max_depth_shift_mm,
            fit_mode="slab_v1",
        )
        fallback["fit_mode_used"] = fallback.get("fit_mode_used", "slab_v1_fallback")
        fallback["fit_mode_requested"] = "deep_anchor_v2"
        fallback["cluster_count"] = int(len(clusters))
        return fallback
    if len(compact_clusters) < 3:
        fallback = fit_electrode_axis_and_tip(
            candidate_points_lps=pts,
            planned_entry_lps=planned_entry_lps,
            planned_target_lps=planned_target_lps,
            contact_offsets_mm=contact_offsets_mm,
            tip_at=tip_at,
            roi_radius_mm=roi_radius_mm,
            max_angle_deg=max_angle_deg,
            max_depth_shift_mm=max_depth_shift_mm,
            fit_mode="slab_v1",
        )
        fallback["fit_mode_used"] = fallback.get("fit_mode_used", "slab_v1_fallback")
        fallback["fit_mode_requested"] = "deep_anchor_v2"
        fallback["cluster_count"] = int(len(clusters))
        fallback["compact_cluster_count"] = int(len(compact_clusters))
        return fallback

    compact_clusters = sorted(compact_clusters, key=lambda item: float(item["t_planned_mm"]))
    deepest = compact_clusters[-1]

    best = None
    for anchor in compact_clusters[-min(3, len(compact_clusters)):]:
        anchor_center = np.asarray(anchor["center"], dtype=float)
        for other in compact_clusters:
            if int(other["cluster_id"]) == int(anchor["cluster_id"]):
                continue
            delta = np.asarray(other["center"], dtype=float) - anchor_center
            if float(np.linalg.norm(delta)) < 8.0:
                continue
            axis = unit(delta)
            if float(np.dot(axis, planned_deep_axis)) < 0.0:
                axis = -axis
            ang = angle_deg(axis, planned_deep_axis)
            if ang > float(max_angle_deg):
                continue
            centers = np.asarray([np.asarray(item["center"], dtype=float) for item in compact_clusters], dtype=float)
            d = point_line_distance(centers, anchor_center, axis)
            mask = d <= 1.6
            if int(np.count_nonzero(mask)) < 3:
                continue
            inliers = [compact_clusters[idx] for idx, keep in enumerate(mask.tolist()) if keep]
            tvals = np.asarray([(np.asarray(item["center"], dtype=float) - anchor_center) @ axis for item in inliers], dtype=float)
            span = float(np.max(tvals) - np.min(tvals)) if tvals.size else 0.0
            if span < 12.0:
                continue
            weights = np.asarray([math.sqrt(max(1.0, float(item["count"]))) for item in inliers], dtype=float)
            score = float(np.sum(weights)) + 0.08 * span - 2.0 * float(np.mean(d[mask]))
            if best is None or score > float(best["score"]):
                best = {
                    "anchor_center": anchor_center,
                    "axis": axis,
                    "inliers": inliers,
                    "score": score,
                    "span": span,
                }

    if best is None:
        fallback = fit_electrode_axis_and_tip(
            candidate_points_lps=pts,
            planned_entry_lps=planned_entry_lps,
            planned_target_lps=planned_target_lps,
            contact_offsets_mm=contact_offsets_mm,
            tip_at=tip_at,
            roi_radius_mm=roi_radius_mm,
            max_angle_deg=max_angle_deg,
            max_depth_shift_mm=max_depth_shift_mm,
            fit_mode="slab_v1",
        )
        fallback["fit_mode_used"] = fallback.get("fit_mode_used", "slab_v1_fallback")
        fallback["fit_mode_requested"] = "deep_anchor_v2"
        fallback["cluster_count"] = int(len(clusters))
        fallback["compact_cluster_count"] = int(len(compact_clusters))
        return fallback

    center, fit_axis = _fit_axis_from_interior_cluster_centers(
        best["inliers"],
        planned_deep_axis=planned_deep_axis,
    )
    planned_target_t_mm = float(np.dot(target - center, fit_axis))
    axis_support = _axis_support_run_from_clusters(
        cluster_summaries,
        core_cluster_ids={int(item["cluster_id"]) for item in best["inliers"]},
        line_point=center,
        line_axis=fit_axis,
        planned_entry_t_mm=float(np.dot(entry - center, fit_axis)),
        planned_target_t_mm=float(np.dot(target - center, fit_axis)),
        radial_limit_mm=1.25,
        max_gap_mm=4.0,
        allowed_gap_mm=configured_gap_mm,
        elongated_direction_cos_min=0.8,
    )
    selected_support_run = axis_support["selected_run"]

    expected_pitch_mm = _contact_pitch_from_offsets(contact_offsets_mm)
    terminal_gap_limit_mm = max(3.0, min(5.0, float(expected_pitch_mm) + 0.5)) if expected_pitch_mm is not None else 4.5
    terminal_chain = _terminal_contiguous_chain(
        best["inliers"],
        line_point=center,
        line_axis=fit_axis,
        max_gap_mm=terminal_gap_limit_mm,
    )
    tip_offset_mm = 0.0
    tight_terminal_t_mm = None
    tight_terminal_chain_count = 0
    generic_terminal_gap_mm = 4.0
    local_terminal_clusters = _local_terminal_compact_clusters(
        roi_pts,
        line_point=center,
        line_axis=fit_axis,
        planned_target_t_mm=planned_target_t_mm,
        back_span_mm=16.0,
        forward_span_mm=3.0,
        radial_limit_mm=1.25,
        cluster_radius_mm=1.35,
        min_cluster_size=4,
    )
    terminal_blob_source = "local_terminal_blob_window"
    support_run_diag = None
    terminal_blob_diag = _score_terminal_blob_candidates(
        local_terminal_clusters if local_terminal_clusters else compact_clusters,
        line_point=center,
        line_axis=fit_axis,
        planned_target_t_mm=planned_target_t_mm,
        max_gap_mm=generic_terminal_gap_mm,
        max_depth_shift_mm=max_depth_shift_mm,
    )
    terminal_blob_t_mm = terminal_blob_diag["selected_anchor_t_mm"]
    terminal_blob_chain_count = 1 if terminal_blob_t_mm is not None else 0
    terminal_anchor_mode = "blob_distal"
    local_terminal_morphology = "contact_like"
    if not local_terminal_clusters:
        terminal_blob_source = "global_compact_clusters"
    elif terminal_blob_t_mm is None and local_terminal_clusters:
        ordered_local = sorted(local_terminal_clusters, key=lambda item: float(item["t_planned_mm"]))
        if len(ordered_local) >= 2:
            deepest_local = ordered_local[-1]
            prev_local = ordered_local[-2]
            deepest_t = float(deepest_local["t_planned_mm"])
            prev_t = float(prev_local["t_planned_mm"])
            deepest_radial = float(
                point_line_distance(np.asarray([np.asarray(deepest_local["center"], dtype=float)]), center, fit_axis)[0]
            )
            tail_gap = deepest_t - prev_t
            if (
                tail_gap <= 5.25
                and deepest_radial <= 0.35
                and abs(deepest_t - planned_target_t_mm) <= float(max_depth_shift_mm) + 1.5
            ):
                anchor_stats = _terminal_blob_axis_stats(deepest_local, line_point=center, line_axis=fit_axis)
                terminal_blob_t_mm = float(anchor_stats["distal_t_mm"])
                terminal_blob_chain_count = 2
                terminal_blob_source = "local_terminal_tail_fallback"
                terminal_blob_diag["selected_anchor_t_mm"] = float(anchor_stats["distal_t_mm"])
                terminal_blob_diag["selected_cluster_id"] = int(deepest_local["cluster_id"])
    selected_cluster = None
    if terminal_blob_diag.get("selected_cluster_id") is not None:
        selected_id = int(terminal_blob_diag["selected_cluster_id"])
        search_clusters = local_terminal_clusters if local_terminal_clusters else compact_clusters
        for item in search_clusters:
            if int(item.get("cluster_id", -1)) == selected_id:
                selected_cluster = item
                break
    if selected_cluster is not None:
        selected_span_mm = float(selected_cluster.get("span_mm", 0.0) or 0.0)
        if selected_span_mm >= 4.5:
            local_terminal_morphology = "string_like"
            support_run_diag = _support_run_diagnostics(
                roi_pts,
                line_point=center,
                line_axis=fit_axis,
                seed_t_mm=float(terminal_blob_t_mm or planned_target_t_mm),
                search_end_t_mm=planned_target_t_mm + 3.0,
                radial_limit_mm=0.8,
                bin_mm=0.5,
                min_points_per_bin=3,
                void_mm=5.0,
                max_gap_mm=4.5,
                allowed_gap_mm=configured_gap_mm,
            )
            extent_t_mm = support_run_diag["selected_endpoint_t_mm"]
            if extent_t_mm is not None:
                terminal_blob_t_mm = float(extent_t_mm)
                terminal_anchor_mode = "tight_extent"
    if terminal_blob_t_mm is not None:
        support_run_diag = _support_run_diagnostics(
            roi_pts,
            line_point=center,
            line_axis=fit_axis,
            seed_t_mm=float(terminal_blob_t_mm),
            search_end_t_mm=planned_target_t_mm + float(max_depth_shift_mm) + 3.0,
            radial_limit_mm=0.9 if local_terminal_morphology == "string_like" else 0.8,
            bin_mm=0.5,
            min_points_per_bin=2,
            void_mm=5.0,
            max_gap_mm=4.5,
            allowed_gap_mm=configured_gap_mm,
        )
        void_extend_t_mm = support_run_diag["selected_endpoint_t_mm"]
        if void_extend_t_mm is not None and float(void_extend_t_mm) > float(terminal_blob_t_mm):
            terminal_blob_t_mm = float(void_extend_t_mm)
            terminal_anchor_mode = "void_extent"
    if terminal_blob_t_mm is None and selected_support_run is not None:
        support_run_t_mm = float(selected_support_run["t_max_mm"])
        terminal_blob_t_mm = support_run_t_mm
        terminal_anchor_mode = "axis_support_run"
        terminal_blob_source = "fixed_axis_support_run"
    if terminal_blob_t_mm is not None:
        deep_anchor_t_override_mm = float(terminal_blob_t_mm)
        deep_anchor_source_override = "axis_support_run" if terminal_anchor_mode == "axis_support_run" else "terminal_blob_centroid"
        tight_terminal_t_mm = float(terminal_blob_t_mm)
        tight_terminal_chain_count = int(terminal_blob_chain_count)
    elif terminal_chain:
        terminal_centers = np.asarray([np.asarray(item["center"], dtype=float) for item in terminal_chain], dtype=float)
        terminal_t = (terminal_centers - center) @ fit_axis
        deep_anchor_t_override_mm = float(np.max(terminal_t))
        deep_anchor_source_override = "terminal_bead_chain"
    else:
        deep_anchor_t_override_mm = None
        deep_anchor_source_override = None

    if selected_support_run is not None:
        run_terminal_t_mm = float(selected_support_run.get("effective_t_max_mm", selected_support_run["t_max_mm"]))
        if terminal_blob_t_mm is None or run_terminal_t_mm > float(terminal_blob_t_mm) + 0.25:
            deep_anchor_t_override_mm = run_terminal_t_mm
            deep_anchor_source_override = "axis_support_run"
            terminal_anchor_mode = "support_run_interval"
            tight_terminal_t_mm = float(deep_anchor_t_override_mm)
    proximal_anchor_t_override_mm = None

    result = _build_fit_result(
        roi_pts=roi_pts,
        planned_entry_lps=entry,
        planned_target_lps=target,
        contact_offsets_mm=contact_offsets_mm,
        tip_at=tip_at,
        center=center,
        fit_deep_axis=fit_axis,
        planned_length=planned_length,
        max_depth_shift_mm=max_depth_shift_mm,
        angle_limit_deg=max_angle_deg,
        fit_mode_used="deep_anchor_v2",
        deep_anchor_t_override_mm=deep_anchor_t_override_mm,
        deep_anchor_source_override=deep_anchor_source_override,
        proximal_anchor_t_override_mm=proximal_anchor_t_override_mm,
        extra_metrics={
            "cluster_count": int(len(clusters)),
            "compact_cluster_count": int(len(compact_clusters)),
            "bead_cluster_inliers": int(len(best["inliers"])),
            "bead_chain_span_mm": float(best["span"]),
            "bead_chain_score": float(best["score"]),
            "deep_anchor_cluster_t_mm": float(deepest["t_planned_mm"]),
            "deep_terminal_chain_count": int(len(terminal_chain)),
            "deep_terminal_gap_limit_mm": float(terminal_gap_limit_mm),
            "deep_terminal_tip_offset_mm": float(tip_offset_mm if terminal_chain else 0.0),
            "deep_terminal_centroid_t_mm": "" if tight_terminal_t_mm is None else float(tight_terminal_t_mm),
            "deep_terminal_centroid_chain_count": int(tight_terminal_chain_count),
            "terminal_blob_gap_limit_mm": float(generic_terminal_gap_mm),
            "local_terminal_cluster_count": int(len(local_terminal_clusters)),
            "local_terminal_window_target_t_mm": float(planned_target_t_mm),
            "local_terminal_morphology": str(local_terminal_morphology),
            "terminal_anchor_mode": str(terminal_anchor_mode),
            "terminal_blob_selected_cluster_id": int(terminal_blob_diag["selected_cluster_id"]) if terminal_blob_diag["selected_cluster_id"] is not None else "",
            "terminal_blob_selected_anchor_t_mm": "" if terminal_blob_diag.get("selected_anchor_t_mm") is None else float(terminal_blob_diag["selected_anchor_t_mm"]),
            "terminal_blob_candidate_count": int(len(terminal_blob_diag["diagnostics"])),
            "terminal_blob_source": str(terminal_blob_source),
            "configured_gap_priors_mm": json.dumps(configured_gap_mm, separators=(",", ":")),
            "axis_support_selected_run_json": "null" if selected_support_run is None else json.dumps(selected_support_run, separators=(",", ":")),
            "axis_support_runs_json": json.dumps(axis_support["runs"], separators=(",", ":")),
            "terminal_blob_diagnostics_json": json.dumps(terminal_blob_diag["diagnostics"], separators=(",", ":")),
            "terminal_support_run_count": 0 if support_run_diag is None else int(len(support_run_diag["runs"])),
            "terminal_support_longest_run_index": "" if support_run_diag is None or support_run_diag["longest_run_index"] is None else int(support_run_diag["longest_run_index"]),
            "terminal_support_selected_run_index": "" if support_run_diag is None or support_run_diag["selected_run_index"] is None else int(support_run_diag["selected_run_index"]),
            "terminal_support_runs_json": "[]" if support_run_diag is None else json.dumps(support_run_diag["runs"], separators=(",", ":")),
        },
    )
    result = _cleanup_fit_with_fixed_target_and_length(
        result,
        cluster_summaries=cluster_summaries,
        axis_support=axis_support,
        selected_support_run=selected_support_run,
        line_point=center,
        line_axis=fit_axis,
        roi_pts=roi_pts,
        planned_entry_lps=entry,
        planned_target_lps=target,
        contact_offsets_mm=contact_offsets_mm,
    )
    result["fit_mode_requested"] = "deep_anchor_v2"
    return result


def fit_electrode_axis_and_tip(
    candidate_points_lps: np.ndarray | list[list[float]],
    planned_entry_lps: Point3D,
    planned_target_lps: Point3D,
    contact_offsets_mm: list[float] | np.ndarray | None = None,
    tip_at: str = "target",
    roi_radius_mm: float = 3.0,
    max_angle_deg: float = 12.0,
    max_depth_shift_mm: float = 20.0,
    fit_mode: str = "slab_v1",
) -> ContactFitResult:
    """Fit observed axis/depth from CT candidates near a planned trajectory.

    Returns a dictionary with `success`, fitted `entry_lps`/`target_lps`, and metrics.
    """
    fit_mode_norm = str(fit_mode or "slab_v1").strip().lower()
    if fit_mode_norm in {"em_v1", "guided_fit_em_v1"}:
        return _fit_electrode_axis_and_tip_em_v1(
            candidate_points_lps=candidate_points_lps,
            planned_entry_lps=planned_entry_lps,
            planned_target_lps=planned_target_lps,
            contact_offsets_mm=contact_offsets_mm,
            tip_at=tip_at,
            roi_radius_mm=roi_radius_mm,
            max_angle_deg=max_angle_deg,
            max_depth_shift_mm=max_depth_shift_mm,
        )
    if fit_mode_norm in {"deep_anchor_v2", "guided_fit_v2"}:
        return _fit_electrode_axis_and_tip_deep_anchored(
            candidate_points_lps=candidate_points_lps,
            planned_entry_lps=planned_entry_lps,
            planned_target_lps=planned_target_lps,
            contact_offsets_mm=contact_offsets_mm,
            tip_at=tip_at,
            roi_radius_mm=roi_radius_mm,
            max_angle_deg=max_angle_deg,
            max_depth_shift_mm=max_depth_shift_mm,
        )
    pts = np.asarray(candidate_points_lps, dtype=float).reshape(-1, 3)
    if pts.size == 0:
        return {"success": False, "reason": "No candidate points", "fit_mode_used": "slab_v1"}

    entry = np.asarray(planned_entry_lps, dtype=float)
    target = np.asarray(planned_target_lps, dtype=float)
    planned_length = float(np.linalg.norm(entry - target))
    if planned_length <= 1e-6:
        return {"success": False, "reason": "Zero-length planned trajectory", "fit_mode_used": "slab_v1"}

    planned_tip, planned_axis = _planned_tip_and_axis(entry, target, tip_at=tip_at)
    planned_deep_axis = unit(target - entry)

    segment_margin = float(max_depth_shift_mm) + 5.0
    slab_t_min = -float(max_depth_shift_mm)
    slab_t_max = planned_length + float(max_depth_shift_mm)

    attempt_radii = [float(roi_radius_mm)]
    fallback_roi_radius_mm = 5.0
    if float(roi_radius_mm) + 1e-6 < fallback_roi_radius_mm:
        attempt_radii.append(fallback_roi_radius_mm)

    fit_inputs = None
    last_failure = None
    for attempt_index, attempt_roi_radius_mm in enumerate(attempt_radii, start=1):
        roi_pts = filter_points_in_segment_cylinder(
            pts,
            seg_start=entry,
            seg_end=target,
            radius_mm=attempt_roi_radius_mm,
            margin_mm=segment_margin,
        )
        if roi_pts.shape[0] < 24:
            last_failure = {
                "success": False,
                "reason": "Too few CT points in ROI",
                "points_in_roi": int(roi_pts.shape[0]),
                "roi_radius_mm_used": float(attempt_roi_radius_mm),
                "fit_attempt_index": int(attempt_index),
                "fit_mode_used": "slab_v1",
            }
            continue

        slab_centroids, slab_t = build_slab_centroids(
            roi_pts,
            origin=entry,
            axis=planned_deep_axis,
            t_min=slab_t_min,
            t_max=slab_t_max,
            step_mm=1.0,
            slab_half_thickness_mm=0.9,
            min_points_per_slab=8,
        )
        if slab_centroids.shape[0] < 8:
            last_failure = {
                "success": False,
                "reason": "Too few slab centroids for robust line fit",
                "points_in_roi": int(roi_pts.shape[0]),
                "slab_centroids": int(slab_centroids.shape[0]),
                "roi_radius_mm_used": float(attempt_roi_radius_mm),
                "fit_attempt_index": int(attempt_index),
                "fit_mode_used": "slab_v1",
            }
            continue

        fit_inputs = {
            "roi_pts": roi_pts,
            "slab_centroids": slab_centroids,
            "slab_t": slab_t,
            "roi_radius_mm_used": float(attempt_roi_radius_mm),
            "fit_attempt_index": int(attempt_index),
        }
        break

    if fit_inputs is None:
        return last_failure or {"success": False, "reason": "Too few slab centroids for robust line fit"}

    roi_pts = fit_inputs["roi_pts"]
    slab_centroids = fit_inputs["slab_centroids"]
    slab_t = fit_inputs["slab_t"]
    roi_radius_mm_used = float(fit_inputs["roi_radius_mm_used"])
    fit_attempt_index = int(fit_inputs["fit_attempt_index"])

    center, fit_axis, inlier_mask, ransac_rms = ransac_fit_line(
        slab_centroids,
        max_iterations=220,
        inlier_threshold_mm=min(1.2, max(0.45, float(roi_radius_mm) * 0.55)),
        min_inliers=max(6, int(0.45 * slab_centroids.shape[0])),
        rng_seed=0,
    )
    if float(np.dot(fit_axis, planned_deep_axis)) < 0.0:
        fit_axis = -fit_axis
    fit_deep_axis = fit_axis
    fit_super_axis = -fit_deep_axis

    ang = angle_deg(planned_deep_axis, fit_deep_axis)
    if ang > float(max_angle_deg):
        return {
            "success": False,
            "reason": f"Angle deviation {ang:.2f} deg exceeds max {max_angle_deg:.2f}",
            "angle_deg": float(ang),
            "points_in_roi": int(roi_pts.shape[0]),
            "slab_centroids": int(slab_centroids.shape[0]),
            "fit_mode_used": "slab_v1",
        }

    inlier_centroids = slab_centroids[inlier_mask] if inlier_mask is not None else slab_centroids
    if inlier_centroids.shape[0] < 3:
        inlier_centroids = slab_centroids

    # Deep anchor: deepest inlier centroid near planned target depth.
    t_in = (inlier_centroids - center) @ fit_deep_axis
    planned_target_t = float(np.dot(target - center, fit_deep_axis))
    deep_min = planned_target_t - float(max_depth_shift_mm)
    deep_max = planned_target_t + float(max_depth_shift_mm)
    near_target = (t_in >= deep_min) & (t_in <= deep_max)
    if np.any(near_target):
        deep_t_raw = float(np.max(t_in[near_target]))
    else:
        deep_t_raw = float(np.quantile(t_in, 0.95))
    deep_t = float(np.clip(deep_t_raw, deep_min, deep_max))

    fitted_target = center + fit_deep_axis * deep_t
    fitted_entry = fitted_target - fit_deep_axis * planned_length

    offs = np.asarray(contact_offsets_mm if contact_offsets_mm is not None else [], dtype=float).reshape(-1)
    tip_at_norm = (tip_at or "target").lower()
    if tip_at_norm == "entry":
        fitted_tip = fitted_entry
        offsets_axis = fit_deep_axis
    else:
        fitted_tip = fitted_target
        offsets_axis = fit_super_axis
    if offs.size > 0:
        pred_centers = fitted_tip[None, :] + np.outer(offs, offsets_axis)

        # Residual in 3D from predicted contacts to ROI cloud.
        diff = pred_centers[:, None, :] - roi_pts[None, :, :]
        d3 = np.linalg.norm(diff, axis=2)
        residual_3d = float(np.mean(np.min(d3, axis=1)))

        # Residual in 1D along fitted deep axis.
        cand_t = (roi_pts - fitted_target) @ fit_deep_axis
        pred_t = (pred_centers - fitted_target) @ fit_deep_axis
        nearest_1d = np.min(np.abs(cand_t[None, :] - pred_t[:, None]), axis=1)
        one_d_residual = float(np.mean(nearest_1d))
    else:
        # Model-free trajectory fit mode: use robust line-fit residuals only.
        residual_3d = float(ransac_rms)
        one_d_residual = float("nan")

    target_shift_mm = float(np.dot(fitted_target - target, fit_deep_axis))
    delta_target = fitted_target - target
    lateral_vec = delta_target - np.dot(delta_target, fit_deep_axis) * fit_deep_axis
    lateral_shift_mm = float(np.linalg.norm(lateral_vec))

    return {
        "success": True,
        "entry_lps": fitted_entry.tolist(),
        "target_lps": fitted_target.tolist(),
        "tip_shift_mm": target_shift_mm,
        "lateral_shift_mm": lateral_shift_mm,
        "angle_deg": float(ang),
        "residual_mm": residual_3d,
        "one_d_residual_mm": float(one_d_residual),
        "points_in_roi": int(roi_pts.shape[0]),
        "fit_mode_used": "slab_v1",
        "slab_centroids": int(slab_centroids.shape[0]),
        "slab_inliers": int(inlier_centroids.shape[0]),
        "ransac_rms_mm": float(ransac_rms),
        "roi_radius_mm_used": float(roi_radius_mm_used),
        "fit_attempt_index": int(fit_attempt_index),
        "axis_lps": fit_super_axis.tolist(),
        "deep_axis_lps": fit_deep_axis.tolist(),
        "center_lps": center.tolist(),
        "deep_t_raw_mm": deep_t_raw,
        "deep_t_clamped_mm": deep_t,
        "planned_target_t_mm": planned_target_t,
        "planned_tip_lps": planned_tip.tolist(),
        "planned_tip_axis_lps": planned_axis.tolist(),
        "slab_t_min_mm": slab_t_min,
        "slab_t_max_mm": slab_t_max,
        "slab_t_used_count": int(slab_t.shape[0]),
    }
