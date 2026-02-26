"""Postop CT candidate fitting helpers for electrode axis/depth refinement.

V2 strategy:
- finite cylindrical ROI around planned segment
- slab centroids sampled along planned depth axis
- RANSAC line fit on slab centroids (robust to nearby electrodes/outliers)
- depth anchor from deepest valid inlier centroid near planned target
"""

import math

import numpy as np


def unit(v):
    """Return normalized 3D vector."""
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n <= 1e-9:
        raise ValueError("Zero-length vector")
    return v / n


def angle_deg(u, v):
    """Return unsigned angle in degrees between two 3D vectors."""
    u = unit(u)
    v = unit(v)
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(math.degrees(math.acos(dot)))


def point_line_distance(points, line_point, line_dir):
    """Return perpendicular distances from Nx3 points to an infinite 3D line."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    p0 = np.asarray(line_point, dtype=float)
    d = unit(line_dir)
    rel = pts - p0
    proj = rel @ d
    closest = p0 + np.outer(proj, d)
    return np.linalg.norm(pts - closest, axis=1)


def filter_points_in_segment_cylinder(points, seg_start, seg_end, radius_mm, margin_mm=5.0):
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


def fit_axis_pca(points):
    """Fit principal axis by PCA and return `(center, axis_unit)`."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if pts.shape[0] < 3:
        raise ValueError("At least 3 points are required for axis fitting")
    center = np.mean(pts, axis=0)
    centered = pts - center
    cov = centered.T @ centered / max(pts.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    return center, unit(axis)


def build_slab_centroids(
    points,
    origin,
    axis,
    t_min,
    t_max,
    step_mm=1.0,
    slab_half_thickness_mm=0.9,
    min_points_per_slab=8,
):
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
    points,
    max_iterations=200,
    inlier_threshold_mm=0.9,
    min_inliers=6,
    rng_seed=0,
):
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


def _planned_tip_and_axis(entry, target, tip_at):
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


def fit_electrode_axis_and_tip(
    candidate_points_lps,
    planned_entry_lps,
    planned_target_lps,
    contact_offsets_mm,
    tip_at="target",
    roi_radius_mm=3.0,
    max_angle_deg=12.0,
    max_depth_shift_mm=20.0,
):
    """Fit observed axis/depth from CT candidates near a planned trajectory.

    Returns a dictionary with `success`, fitted `entry_lps`/`target_lps`, and metrics.
    """
    pts = np.asarray(candidate_points_lps, dtype=float).reshape(-1, 3)
    if pts.size == 0:
        return {"success": False, "reason": "No candidate points"}

    entry = np.asarray(planned_entry_lps, dtype=float)
    target = np.asarray(planned_target_lps, dtype=float)
    planned_length = float(np.linalg.norm(entry - target))
    if planned_length <= 1e-6:
        return {"success": False, "reason": "Zero-length planned trajectory"}

    planned_tip, planned_axis = _planned_tip_and_axis(entry, target, tip_at=tip_at)
    planned_deep_axis = unit(target - entry)

    segment_margin = float(max_depth_shift_mm) + 5.0
    roi_pts = filter_points_in_segment_cylinder(
        pts,
        seg_start=entry,
        seg_end=target,
        radius_mm=roi_radius_mm,
        margin_mm=segment_margin,
    )
    if roi_pts.shape[0] < 24:
        return {"success": False, "reason": "Too few CT points in ROI", "points_in_roi": int(roi_pts.shape[0])}

    slab_t_min = -float(max_depth_shift_mm)
    slab_t_max = planned_length + float(max_depth_shift_mm)
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
        return {
            "success": False,
            "reason": "Too few slab centroids for robust line fit",
            "points_in_roi": int(roi_pts.shape[0]),
            "slab_centroids": int(slab_centroids.shape[0]),
        }

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

    offs = np.asarray(contact_offsets_mm, dtype=float).reshape(-1)
    if offs.size == 0:
        return {"success": False, "reason": "No model offsets provided"}
    tip_at_norm = (tip_at or "target").lower()
    if tip_at_norm == "entry":
        fitted_tip = fitted_entry
        offsets_axis = fit_deep_axis
    else:
        fitted_tip = fitted_target
        offsets_axis = fit_super_axis
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
        "slab_centroids": int(slab_centroids.shape[0]),
        "slab_inliers": int(inlier_centroids.shape[0]),
        "ransac_rms_mm": float(ransac_rms),
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
