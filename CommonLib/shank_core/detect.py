"""Core trajectory detection logic (Slicer-independent).

This module operates on:
- threshold/gating outputs from ``masking.build_preview_masks``
- RAS/IJK conversion callables supplied by the caller

Pipeline stages:
1) first-pass RANSAC line proposals
2) segment clamping to gating mask
3) exclusive reassignment refinement
4) optional model-template scoring and gap rejection
"""

from __future__ import annotations

import math
import time

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None
try:
    import SimpleITK as sitk
except ImportError:  # pragma: no cover
    sitk = None

from .blob_candidates import build_blob_labelmap, extract_blob_candidates, filter_blob_candidates


def _require_numpy():
    if np is None:
        raise RuntimeError("numpy is required for detection")


def _build_head_depth_map_kji(mask_kji, spacing_xyz):
    """Signed distance map (mm) from outer head surface; inside values are positive."""
    if sitk is None or mask_kji is None:
        return None
    mask = np.asarray(mask_kji, dtype=np.uint8)
    if mask.size == 0 or int(np.count_nonzero(mask)) == 0:
        return None
    img = sitk.GetImageFromArray(mask)
    img.SetSpacing((float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])))
    dist_img = sitk.SignedMaurerDistanceMap(
        img,
        insideIsPositive=True,
        squaredDistance=False,
        useImageSpacing=True,
    )
    return sitk.GetArrayFromImage(dist_img).astype(np.float32)


def _depth_at_ras_mm(point_ras, head_depth_kji, ras_to_ijk_fn):
    """Head-depth at one RAS point, or None if unavailable/outside volume."""
    if head_depth_kji is None:
        return None
    ijk = ras_to_ijk_fn(point_ras)
    i = int(round(float(ijk[0])))
    j = int(round(float(ijk[1])))
    k = int(round(float(ijk[2])))
    if (
        k < 0
        or j < 0
        or i < 0
        or k >= head_depth_kji.shape[0]
        or j >= head_depth_kji.shape[1]
        or i >= head_depth_kji.shape[2]
    ):
        return None
    val = float(head_depth_kji[k, j, i])
    if not np.isfinite(val):
        return None
    return val


def _orient_segment_entry_target(start_ras, end_ras, center_ras, head_depth_kji, ras_to_ijk_fn):
    """Choose entry/target endpoints.

    Preferred rule:
    - entry is endpoint with smaller head-depth (closer to surface).
    Fallback:
    - entry is endpoint farther from center_ras.
    """
    p0 = np.asarray(start_ras, dtype=float)
    p1 = np.asarray(end_ras, dtype=float)

    d0 = _depth_at_ras_mm(p0, head_depth_kji=head_depth_kji, ras_to_ijk_fn=ras_to_ijk_fn)
    d1 = _depth_at_ras_mm(p1, head_depth_kji=head_depth_kji, ras_to_ijk_fn=ras_to_ijk_fn)
    if d0 is not None and d1 is not None and abs(float(d0) - float(d1)) > 1e-3:
        if float(d0) <= float(d1):
            return p0, p1
        return p1, p0

    c0 = float(np.linalg.norm(p0 - center_ras))
    c1 = float(np.linalg.norm(p1 - center_ras))
    if c0 < c1:
        return p1, p0
    return p0, p1


def _line_distances(points, p0, direction_unit):
    """Distance from points to an infinite 3D line."""
    rel = points - p0
    t = rel @ direction_unit
    closest = p0 + np.outer(t, direction_unit)
    return np.linalg.norm(points - closest, axis=1)


def _fit_axis_pca(points):
    """Principal-axis fit for a point cloud."""
    center = np.mean(points, axis=0)
    x = points - center
    cov = x.T @ x / max(points.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    n = np.linalg.norm(axis)
    if n <= 1e-9:
        raise ValueError("failed to estimate principal axis")
    return center, axis / n


def _ransac_fit_line(points, distance_threshold_mm=1.2, min_inliers=250, iterations=240):
    """Robust line fit from 3D points using pair-sampling RANSAC."""
    n = int(points.shape[0])
    if n < 2:
        return None

    rng = np.random.default_rng(0)
    best_mask = None
    best_count = -1
    best_score = float("inf")
    max_iterations = max(1, int(iterations))
    target_iterations = max_iterations
    confidence = 0.999
    min_iterations_floor = min(max_iterations, 80)

    for it in range(max_iterations):
        i, j = rng.choice(n, size=2, replace=False)
        p0 = points[i]
        d = points[j] - p0
        dn = float(np.linalg.norm(d))
        if dn <= 1e-9:
            continue
        d = d / dn
        dist = _line_distances(points, p0, d)
        mask = dist <= float(distance_threshold_mm)
        count = int(np.count_nonzero(mask))
        if count <= 0:
            continue
        improved = False
        if count > best_count:
            score = float(np.mean(dist[mask]))
            best_mask = mask
            best_count = count
            best_score = score
            improved = True
        elif count == best_count and best_count > 0:
            score = float(np.mean(dist[mask]))
            if score < best_score:
                best_mask = mask
                best_score = score
                improved = True

        if improved:
            # Adaptive RANSAC: stop once current inlier ratio implies enough hypotheses.
            if best_count >= int(min_inliers):
                w = max(1e-6, min(0.999999, float(best_count) / float(max(1, n))))
                success_prob = max(1e-12, min(1.0 - 1e-12, w * w))
                denom = math.log(max(1e-12, 1.0 - success_prob))
                if denom < -1e-12:
                    required = int(math.ceil(math.log(1.0 - confidence) / denom))
                    target_iterations = max(min_iterations_floor, min(target_iterations, required))

        if best_count >= int(min_inliers) and (it + 1) >= target_iterations:
            break

    if best_mask is None or best_count < int(min_inliers):
        return None

    inliers = points[best_mask]
    center, axis = _fit_axis_pca(inliers)
    dist = _line_distances(inliers, center, axis)
    rms = float(np.sqrt(np.mean(dist**2))) if dist.size else 0.0
    return {
        "inlier_mask": best_mask,
        "center": center,
        "axis": axis,
        "inlier_count": int(best_count),
        "rms_mm": rms,
    }


def _distance_to_segment_mask(points, start, end, radius_mm):
    """Boolean mask for points within radius of a finite segment."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    seg = end - start
    seg_len2 = float(np.dot(seg, seg))
    if seg_len2 <= 1e-9:
        d = np.linalg.norm(points - start.reshape(1, 3), axis=1)
        return d <= float(radius_mm)

    rel = points - start.reshape(1, 3)
    t = (rel @ seg) / seg_len2
    t = np.clip(t, 0.0, 1.0)
    closest = start.reshape(1, 3) + np.outer(t, seg)
    d = np.linalg.norm(points - closest, axis=1)
    return d <= float(radius_mm)


def _distance_to_line_with_extension(points, start, end, extension_mm=8.0):
    """Distance to a line axis, valid only within finite length + extension."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    seg = end - start
    length = float(np.linalg.norm(seg))
    if length <= 1e-9:
        d = np.linalg.norm(points - start.reshape(1, 3), axis=1)
        return d, np.zeros((points.shape[0],), dtype=bool)
    u = seg / length
    rel = points - start.reshape(1, 3)
    t = rel @ u
    closest = start.reshape(1, 3) + np.outer(t, u)
    d = np.linalg.norm(points - closest, axis=1)
    ext = float(max(0.0, extension_mm))
    valid = (t >= -ext) & (t <= (length + ext))
    return d, valid


def _normalize_candidate_mode(value):
    mode = str(value or "voxel").strip().lower()
    if mode not in ("voxel", "blob_centroid"):
        return "voxel"
    return mode


def _line_support_mask(points, line, radius_mm):
    return _distance_to_segment_mask(
        points,
        start=line["start_ras"],
        end=line["end_ras"],
        radius_mm=radius_mm,
    )


def _assigned_mask_for_lines(points, lines, radius_mm):
    if points is None or points.shape[0] == 0 or not lines:
        return np.zeros((0,), dtype=bool) if points is None else np.zeros((points.shape[0],), dtype=bool)
    assigned = np.zeros((points.shape[0],), dtype=bool)
    for line in lines:
        assigned |= _line_support_mask(points, line, radius_mm=radius_mm)
    return assigned


def _line_quality_tuple(line):
    """Quality ordering used for line conflict suppression."""
    model_score = line.get("best_model_score", None)
    has_model = 1 if model_score is not None else 0
    model_val = float(model_score) if model_score is not None else -1e9
    return (
        has_model,
        model_val,
        float(line.get("inside_fraction", 0.0)),
        float(line.get("support_weight", 0.0)),
        float(line.get("inlier_count", 0.0)),
        float(line.get("depth_span_mm", 0.0)),
        -float(line.get("rms_mm", 999.0)),
    )


def _suppress_conflicting_lines_by_support(lines, support_points, support_radius_mm=1.2, overlap_threshold=0.60):
    """Suppress lines that reuse most support points from a better line."""
    if not lines or support_points is None or support_points.shape[0] == 0:
        return list(lines), 0

    ordered = sorted(lines, key=_line_quality_tuple, reverse=True)
    kept = []
    kept_masks = []
    rejected = 0
    for cand in ordered:
        c_mask = _distance_to_segment_mask(
            support_points,
            start=cand["start_ras"],
            end=cand["end_ras"],
            radius_mm=support_radius_mm,
        )
        c_count = int(np.count_nonzero(c_mask))
        if c_count <= 0:
            continue

        conflict = False
        for k_mask in kept_masks:
            k_count = int(np.count_nonzero(k_mask))
            inter = int(np.count_nonzero(np.logical_and(c_mask, k_mask)))
            denom = max(1, min(c_count, k_count))
            if (float(inter) / float(denom)) >= float(overlap_threshold):
                conflict = True
                break
        if conflict:
            rejected += 1
            continue
        kept.append(cand)
        kept_masks.append(c_mask)

    return kept, rejected


def _segment_inside_mask_fraction(start_ras, end_ras, mask_kji, ras_to_ijk_fn, step_mm=1.0):
    """Fraction of sampled segment points that land inside mask."""
    start = np.asarray(start_ras, dtype=float)
    end = np.asarray(end_ras, dtype=float)
    seg = end - start
    length = float(np.linalg.norm(seg))
    if length <= 1e-9:
        return 0.0
    n = max(2, int(math.ceil(length / max(1e-3, float(step_mm)))) + 1)
    ts = np.linspace(0.0, 1.0, n)

    dims = mask_kji.shape
    inside = 0
    total = 0
    for t in ts:
        p = start + t * seg
        ijk = ras_to_ijk_fn(p)
        i = int(round(float(ijk[0])))
        j = int(round(float(ijk[1])))
        k = int(round(float(ijk[2])))
        total += 1
        if 0 <= k < dims[0] and 0 <= j < dims[1] and 0 <= i < dims[2]:
            if bool(mask_kji[k, j, i]):
                inside += 1
    return float(inside) / float(max(1, total))


def _clamp_segment_to_mask(start_ras, end_ras, mask_kji, ras_to_ijk_fn, step_mm=0.5):
    """Clip segment to first/last in-mask samples."""
    start = np.asarray(start_ras, dtype=float)
    end = np.asarray(end_ras, dtype=float)
    seg = end - start
    length = float(np.linalg.norm(seg))
    if length <= 1e-9:
        return None
    n = max(2, int(math.ceil(length / max(1e-3, float(step_mm)))) + 1)
    ts = np.linspace(0.0, 1.0, n)
    pts = start.reshape(1, 3) + np.outer(ts, seg)

    dims = mask_kji.shape
    inside = np.zeros((n,), dtype=bool)
    for idx in range(n):
        ijk = ras_to_ijk_fn(pts[idx])
        i = int(round(float(ijk[0])))
        j = int(round(float(ijk[1])))
        k = int(round(float(ijk[2])))
        if 0 <= k < dims[0] and 0 <= j < dims[1] and 0 <= i < dims[2]:
            inside[idx] = bool(mask_kji[k, j, i])
    idxs = np.where(inside)[0]
    if idxs.size < 2:
        return None
    p0 = pts[int(idxs[0])]
    p1 = pts[int(idxs[-1])]
    if float(np.linalg.norm(p1 - p0)) <= 1e-3:
        return None
    return [float(p0[0]), float(p0[1]), float(p0[2])], [float(p1[0]), float(p1[1]), float(p1[2])]


def _sample_local_max_hu(arr_kji, ijk_float, radius_vox_ijk):
    """Local max HU around one continuous IJK point."""
    i0 = int(round(float(ijk_float[0])))
    j0 = int(round(float(ijk_float[1])))
    k0 = int(round(float(ijk_float[2])))
    ri, rj, rk = radius_vox_ijk
    i1 = max(0, i0 - ri)
    i2 = min(arr_kji.shape[2] - 1, i0 + ri)
    j1 = max(0, j0 - rj)
    j2 = min(arr_kji.shape[1] - 1, j0 + rj)
    k1 = max(0, k0 - rk)
    k2 = min(arr_kji.shape[0] - 1, k0 + rk)
    if i1 > i2 or j1 > j2 or k1 > k2:
        return float("-inf")
    patch = arr_kji[k1 : k2 + 1, j1 : j2 + 1, i1 : i2 + 1]
    return float(np.max(patch)) if patch.size else float("-inf")


def _sample_axis_profile_hu(arr_kji, tip_ras, entry_ras, spacing_xyz, ras_to_ijk_fn, step_mm=0.5, radial_sample_mm=0.8):
    """Sample HU profile along one candidate trajectory axis."""
    tip = np.asarray(tip_ras, dtype=float)
    entry = np.asarray(entry_ras, dtype=float)
    axis = entry - tip
    length = float(np.linalg.norm(axis))
    if length <= 1e-6:
        return np.empty((0,), dtype=float), np.empty((0,), dtype=float)
    axis = axis / length

    ri = max(0, int(round(float(radial_sample_mm) / max(1e-6, float(spacing_xyz[0])))))
    rj = max(0, int(round(float(radial_sample_mm) / max(1e-6, float(spacing_xyz[1])))))
    rk = max(0, int(round(float(radial_sample_mm) / max(1e-6, float(spacing_xyz[2])))))
    radius_vox_ijk = (ri, rj, rk)

    n = max(2, int(math.floor(length / max(1e-6, float(step_mm)))) + 1)
    t = np.linspace(0.0, length, n)
    vals = np.zeros((n,), dtype=float)
    for idx, dist in enumerate(t):
        p = tip + dist * axis
        ijk = ras_to_ijk_fn(p)
        vals[idx] = _sample_local_max_hu(arr_kji, ijk, radius_vox_ijk)
    return t, vals


def _score_model_template_on_profile(t_mm, hu_vals, model, line_length_mm):
    """Template-match one electrode model against HU profile."""
    if t_mm.size < 8:
        return {"score": float("-inf"), "shift_mm": 0.0}
    offsets = [float(x) for x in model.get("contact_center_offsets_from_tip_mm", [])]
    if not offsets:
        return {"score": float("-inf"), "shift_mm": 0.0}

    mean = float(np.mean(hu_vals))
    std = float(np.std(hu_vals))
    if std <= 1e-6:
        return {"score": float("-inf"), "shift_mm": 0.0}
    z = (hu_vals - mean) / std
    t_max = float(np.max(t_mm))

    shifts = np.arange(-6.0, 6.01, 0.5)
    best_score = float("-inf")
    best_shift = 0.0
    for shift in shifts:
        centers = [o + float(shift) for o in offsets]
        centers = [c for c in centers if 0.0 <= c <= t_max]
        if len(centers) < max(3, int(0.4 * len(offsets))):
            continue
        contact_vals = np.interp(centers, t_mm, z)

        gap_vals = []
        for i in range(len(centers) - 1):
            gap_vals.append(0.5 * (centers[i] + centers[i + 1]))
        if gap_vals:
            gap_vals = np.interp(np.asarray(gap_vals, dtype=float), t_mm, z)
            gap_mean = float(np.mean(gap_vals))
        else:
            gap_mean = 0.0
        contact_mean = float(np.mean(contact_vals))

        score = contact_mean - gap_mean
        model_len = float(model.get("total_exploration_length_mm", line_length_mm))
        length_mismatch = abs(float(line_length_mm) - model_len)
        score -= max(0.0, length_mismatch - 5.0) * 0.05

        if score > best_score:
            best_score = score
            best_shift = float(shift)
    return {"score": best_score, "shift_mm": best_shift}


def _select_best_model_for_line(arr_kji, spacing_xyz, ras_to_ijk_fn, line, models_by_id):
    """Return best-scoring model (if any) for one line candidate."""
    t_mm, hu_vals = _sample_axis_profile_hu(
        arr_kji=arr_kji,
        tip_ras=line["end_ras"],
        entry_ras=line["start_ras"],
        spacing_xyz=spacing_xyz,
        ras_to_ijk_fn=ras_to_ijk_fn,
        step_mm=0.5,
        radial_sample_mm=0.8,
    )
    if t_mm.size < 8:
        return {"best_model_id": "", "best_model_score": None, "best_model_shift_mm": None}

    best_id = ""
    best_score = float("-inf")
    best_shift = 0.0
    for model_id in sorted(models_by_id.keys()):
        score = _score_model_template_on_profile(
            t_mm=t_mm,
            hu_vals=hu_vals,
            model=models_by_id[model_id],
            line_length_mm=float(line["length_mm"]),
        )
        if score["score"] > best_score:
            best_score = float(score["score"])
            best_shift = float(score["shift_mm"])
            best_id = model_id
    if not best_id:
        return {"best_model_id": "", "best_model_score": None, "best_model_shift_mm": None}
    return {
        "best_model_id": best_id,
        "best_model_score": float(best_score),
        "best_model_shift_mm": float(best_shift),
    }


def _model_max_center_gap_mm(model):
    """Maximum allowed center-to-center gap from one model definition."""
    offsets = [float(x) for x in model.get("contact_center_offsets_from_tip_mm", [])]
    if len(offsets) < 2:
        return 0.0
    return float(max(offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1)))


def _max_empty_gap_mm_from_line_inliers(t_vals, lo, hi, bin_mm=0.5):
    """Largest empty interval along a line, from projected inlier positions."""
    span = float(hi - lo)
    if span <= max(1e-6, float(bin_mm)):
        return 0.0
    t = np.asarray(t_vals, dtype=float)
    t = t[np.isfinite(t)]
    t = t[(t >= float(lo)) & (t <= float(hi))]
    if t.size < 2:
        return span

    n_bins = max(2, int(math.ceil(span / max(1e-6, float(bin_mm)))))
    occ = np.zeros((n_bins,), dtype=np.uint8)
    u = t - float(lo)
    idx = np.floor(u / max(1e-6, float(bin_mm))).astype(int)
    idx = np.clip(idx, 0, n_bins - 1)
    occ[idx] = 1

    occ_dil = occ.copy()
    occ_dil[1:] = np.maximum(occ_dil[1:], occ[:-1])
    occ_dil[:-1] = np.maximum(occ_dil[:-1], occ[1:])
    occ = occ_dil
    occ_idx = np.where(occ > 0)[0]
    if occ_idx.size < 2:
        return span
    gaps_bins = np.diff(occ_idx) - 1
    max_gap_bins = int(np.max(gaps_bins)) if gaps_bins.size else 0
    return float(max(0, max_gap_bins)) * float(bin_mm)


def _refine_lines_exclusive(
    arr_kji,
    spacing_xyz,
    ras_to_ijk_fn,
    center_ras,
    head_depth_kji,
    support_points,
    raw_lines,
    inlier_radius_mm,
    min_length_mm,
    min_inliers,
    gating_mask_kji,
    models_by_id=None,
    min_model_score=None,
    min_metal_depth_mm=5.0,
    start_zone_window_mm=10.0,
    support_weights=None,
    apply_start_zone_prior=True,
):
    """Second-pass refinement where each support point belongs to at most one line."""
    if not raw_lines or support_points is None or support_points.shape[0] == 0:
        empty_assigned = np.zeros((0,), dtype=bool)
        return [], {
            "gap_reject_count": 0,
            "duplicate_reject_count": 0,
            "start_zone_reject_count": 0,
            "length_reject_count": 0,
            "inlier_reject_count": 0,
            "assigned_mask": empty_assigned,
        }

    # Exclusive assignment stage: build one winner line index per point.
    n = support_points.shape[0]
    m = len(raw_lines)
    if support_weights is None or len(support_weights) != n:
        support_weights = np.ones((n,), dtype=float)
    else:
        support_weights = np.asarray(support_weights, dtype=float).reshape(-1)
        support_weights = np.where(np.isfinite(support_weights), support_weights, 1.0)
        support_weights = np.maximum(support_weights, 0.0)
    dist_mat = np.full((n, m), np.inf, dtype=float)
    for li, line in enumerate(raw_lines):
        d, valid = _distance_to_line_with_extension(
            support_points,
            start=line["start_ras"],
            end=line["end_ras"],
            extension_mm=8.0,
        )
        d[~valid] = np.inf
        dist_mat[:, li] = d

    assign_radius_mm = max(1.5, float(inlier_radius_mm) * 1.25)
    best_idx = np.argmin(dist_mat, axis=1)
    best_dist = dist_mat[np.arange(n), best_idx]
    assigned = np.full((n,), -1, dtype=int)
    assigned[best_dist <= assign_radius_mm] = best_idx[best_dist <= assign_radius_mm]

    # Re-fit each assigned cluster and enforce geometric/model plausibility.
    kept = []
    gap_reject_count = 0
    start_zone_reject_count = 0
    length_reject_count = 0
    inlier_reject_count = 0

    for li in range(m):
        pts = support_points[assigned == li]
        if pts.shape[0] < int(min_inliers):
            inlier_reject_count += 1
            continue
        center, axis = _fit_axis_pca(pts)
        t = (pts - center.reshape(1, 3)) @ axis
        if t.size < 2:
            continue
        lo = float(np.percentile(t, 2.0))
        hi = float(np.percentile(t, 98.0))
        length = float(hi - lo)
        if length < float(min_length_mm):
            length_reject_count += 1
            continue

        p0 = center + axis * lo
        p1 = center + axis * hi
        entry_ras, target_ras = _orient_segment_entry_target(
            start_ras=p0,
            end_ras=p1,
            center_ras=center_ras,
            head_depth_kji=head_depth_kji,
            ras_to_ijk_fn=ras_to_ijk_fn,
        )

        inside_fraction = 1.0
        if gating_mask_kji is not None:
            inside_fraction = _segment_inside_mask_fraction(
                start_ras=entry_ras,
                end_ras=target_ras,
                mask_kji=gating_mask_kji,
                ras_to_ijk_fn=ras_to_ijk_fn,
                step_mm=1.0,
            )
            clamped = _clamp_segment_to_mask(
                start_ras=entry_ras,
                end_ras=target_ras,
                mask_kji=gating_mask_kji,
                ras_to_ijk_fn=ras_to_ijk_fn,
                step_mm=0.5,
            )
            if clamped is None:
                continue
            entry_ras, target_ras = clamped
            entry_ras, target_ras = _orient_segment_entry_target(
                start_ras=entry_ras,
                end_ras=target_ras,
                center_ras=center_ras,
                head_depth_kji=head_depth_kji,
                ras_to_ijk_fn=ras_to_ijk_fn,
            )
            length = float(np.linalg.norm(np.asarray(entry_ras) - np.asarray(target_ras)))
            if length < float(min_length_mm):
                length_reject_count += 1
                continue

        rms = float(np.sqrt(np.mean(_line_distances(pts, center, axis) ** 2))) if pts.shape[0] else 0.0
        support_weight = float(np.sum(support_weights[assigned == li]))
        depth_span = 0.0
        if head_depth_kji is not None and pts.shape[0] > 0:
            dvals = []
            for point in pts:
                d = _depth_at_ras_mm(point, head_depth_kji=head_depth_kji, ras_to_ijk_fn=ras_to_ijk_fn)
                if d is not None:
                    dvals.append(float(d))
            if dvals:
                depth_span = float(max(dvals) - min(dvals))
        line = {
            "start_ras": [float(entry_ras[0]), float(entry_ras[1]), float(entry_ras[2])],
            "end_ras": [float(target_ras[0]), float(target_ras[1]), float(target_ras[2])],
            "length_mm": float(length),
            "inlier_count": int(pts.shape[0]),
            "support_weight": support_weight,
            "rms_mm": float(rms),
            "inside_fraction": float(inside_fraction),
            "depth_span_mm": float(depth_span),
        }

        # Require at least one support point in the shallow start-depth zone:
        # [0, min_metal_depth_mm + start_zone_window_mm].
        # Also allow entry-point depth to satisfy this rule; blob-centroid mode can
        # have sparse support near entry even when the fitted entry is valid.
        if bool(apply_start_zone_prior) and head_depth_kji is not None and pts.shape[0] > 0:
            depth_vals = []
            for point in pts:
                d = _depth_at_ras_mm(point, head_depth_kji=head_depth_kji, ras_to_ijk_fn=ras_to_ijk_fn)
                if d is not None:
                    depth_vals.append(float(d))
            if depth_vals:
                start_hi = float(max(0.0, min_metal_depth_mm) + max(0.0, start_zone_window_mm))
                has_start_zone_point = any((dv >= 0.0) and (dv <= start_hi) for dv in depth_vals)
                if not has_start_zone_point:
                    entry_depth = _depth_at_ras_mm(
                        np.asarray(entry_ras, dtype=float),
                        head_depth_kji=head_depth_kji,
                        ras_to_ijk_fn=ras_to_ijk_fn,
                    )
                    if entry_depth is not None:
                        has_start_zone_point = (float(entry_depth) >= 0.0) and (float(entry_depth) <= float(start_hi + 2.0))
                if not has_start_zone_point:
                    start_zone_reject_count += 1
                    continue

        if models_by_id:
            model_fit = _select_best_model_for_line(
                arr_kji=arr_kji,
                spacing_xyz=spacing_xyz,
                ras_to_ijk_fn=ras_to_ijk_fn,
                line=line,
                models_by_id=models_by_id,
            )
            line.update(model_fit)
            best_model_id = str(model_fit.get("best_model_id") or "")
            if min_model_score is not None and model_fit.get("best_model_score") is not None:
                if float(model_fit["best_model_score"]) < float(min_model_score):
                    continue

            observed_gap = _max_empty_gap_mm_from_line_inliers(t_vals=t, lo=lo, hi=hi, bin_mm=0.5)
            if best_model_id and best_model_id in models_by_id:
                max_model_gap = _model_max_center_gap_mm(models_by_id[best_model_id])
                allowed_gap = float(max_model_gap) + 3.0
            else:
                allowed_gap = 14.0
            line["max_observed_gap_mm"] = float(observed_gap)
            line["max_allowed_gap_mm"] = float(allowed_gap)
            if observed_gap > allowed_gap:
                gap_reject_count += 1
                continue

        kept.append(line)

    kept, duplicate_reject_count = _suppress_conflicting_lines_by_support(
        lines=kept,
        support_points=support_points,
        support_radius_mm=max(1.0, float(inlier_radius_mm)),
        overlap_threshold=0.60,
    )
    assigned_mask = _assigned_mask_for_lines(
        points=support_points,
        lines=kept,
        radius_mm=max(1.0, float(inlier_radius_mm)),
    )
    return kept, {
        "gap_reject_count": int(gap_reject_count),
        "start_zone_reject_count": int(start_zone_reject_count),
        "duplicate_reject_count": int(duplicate_reject_count),
        "length_reject_count": int(length_reject_count),
        "inlier_reject_count": int(inlier_reject_count),
        "assigned_mask": assigned_mask,
    }


def _fit_line_proposals(
    points,
    center_ras,
    head_depth_kji,
    ras_to_ijk_fn,
    gating_mask_kji,
    inlier_radius_mm,
    min_length_mm,
    min_inliers,
    ransac_iterations,
    max_lines,
):
    """First-pass RANSAC proposals over support points."""
    remaining = np.asarray(points, dtype=float).reshape(-1, 3)
    lines = []
    length_reject_count = 0
    for _ in range(int(max_lines)):
        if remaining.shape[0] < int(min_inliers):
            break
        fit = _ransac_fit_line(
            remaining,
            distance_threshold_mm=inlier_radius_mm,
            min_inliers=min_inliers,
            iterations=ransac_iterations,
        )
        if fit is None:
            break

        mask = fit["inlier_mask"]
        inliers = remaining[mask]
        center = fit["center"]
        axis = fit["axis"]
        t = (inliers - center.reshape(1, 3)) @ axis
        if t.size < 2:
            remaining = remaining[~mask]
            continue
        lo = float(np.percentile(t, 2.0))
        hi = float(np.percentile(t, 98.0))
        length = float(hi - lo)
        if length < float(min_length_mm):
            length_reject_count += 1
            remaining = remaining[~mask]
            continue

        p_start = center + axis * lo
        p_end = center + axis * hi
        entry_ras, target_ras = _orient_segment_entry_target(
            start_ras=p_start,
            end_ras=p_end,
            center_ras=center_ras,
            head_depth_kji=head_depth_kji,
            ras_to_ijk_fn=ras_to_ijk_fn,
        )
        inside_fraction = 1.0
        if gating_mask_kji is not None:
            inside_fraction = _segment_inside_mask_fraction(
                start_ras=entry_ras,
                end_ras=target_ras,
                mask_kji=gating_mask_kji,
                ras_to_ijk_fn=ras_to_ijk_fn,
                step_mm=1.0,
            )
            clamped = _clamp_segment_to_mask(
                start_ras=entry_ras,
                end_ras=target_ras,
                mask_kji=gating_mask_kji,
                ras_to_ijk_fn=ras_to_ijk_fn,
                step_mm=0.5,
            )
            if clamped is None:
                remaining = remaining[~mask]
                continue
            entry_ras, target_ras = clamped
            entry_ras, target_ras = _orient_segment_entry_target(
                start_ras=entry_ras,
                end_ras=target_ras,
                center_ras=center_ras,
                head_depth_kji=head_depth_kji,
                ras_to_ijk_fn=ras_to_ijk_fn,
            )
            length = float(np.linalg.norm(np.asarray(entry_ras) - np.asarray(target_ras)))
            if length < float(min_length_mm):
                length_reject_count += 1
                remaining = remaining[~mask]
                continue

        lines.append(
            {
                "start_ras": [float(entry_ras[0]), float(entry_ras[1]), float(entry_ras[2])],
                "end_ras": [float(target_ras[0]), float(target_ras[1]), float(target_ras[2])],
                "length_mm": float(length),
                "inlier_count": int(fit["inlier_count"]),
                "support_weight": float(fit["inlier_count"]),
                "rms_mm": float(fit["rms_mm"]),
                "inside_fraction": float(inside_fraction),
                "depth_span_mm": 0.0,
            }
        )
        remaining = remaining[~mask]
    return lines, {"length_reject_count": int(length_reject_count)}


def detect_from_preview(
    arr_kji,
    spacing_xyz,
    preview,
    ijk_kji_to_ras_fn,
    ras_to_ijk_fn,
    center_ras,
    max_points=300000,
    max_lines=30,
    inlier_radius_mm=1.2,
    min_length_mm=20.0,
    min_inliers=250,
    ransac_iterations=240,
    exclude_segments=None,
    exclude_radius_mm=2.0,
    models_by_id=None,
    min_model_score=None,
    min_metal_depth_mm=5.0,
    start_zone_window_mm=10.0,
    candidate_mode="voxel",
    min_blob_voxels=2,
    max_blob_voxels=1200,
    min_blob_peak_hu=None,
    max_blob_elongation=None,
    enable_rescue_pass=True,
    rescue_min_inliers_scale=0.6,
    rescue_max_lines=6,
    apply_start_zone_prior=None,
):
    """Detect trajectory lines using a precomputed preview mask result."""
    _require_numpy()
    t0 = time.perf_counter()
    t_setup0 = time.perf_counter()
    mode = _normalize_candidate_mode(candidate_mode)
    if apply_start_zone_prior is None:
        apply_start_zone_prior = (mode == "voxel")
    blob_ms = 0.0
    effective_min_inliers = int(min_inliers)
    effective_inlier_radius_mm = float(inlier_radius_mm)

    candidate_count = int(preview.get("candidate_count", 0))
    empty_ras = np.empty((0, 3), dtype=float)
    default_profile = {
        "setup": 0.0,
        "subsample": 0.0,
        "depth_map": 0.0,
        "ras_convert": 0.0,
        "blob_stage": 0.0,
        "exclude": 0.0,
        "first_pass": 0.0,
        "refine": 0.0,
        "rescue": 0.0,
        "total": 0.0,
    }
    if candidate_count == 0:
        return {
            "head_mask_kept_count": 0,
            "inside_method": "none",
            "metal_in_head_count": 0,
            "depth_kept_count": 0,
            "gap_reject_count": 0,
            "start_zone_reject_count": 0,
            "duplicate_reject_count": 0,
            "length_reject_count": 0,
            "inlier_reject_count": 0,
            "candidate_points_total": 0,
            "candidate_points_after_mask": 0,
            "candidate_points_after_depth": 0,
            "effective_min_inliers": int(min_inliers),
            "effective_inlier_radius_mm": float(inlier_radius_mm),
            "fit1_lines_proposed": 0,
            "fit2_lines_kept": 0,
            "rescue_lines_kept": 0,
            "final_lines_kept": 0,
            "assigned_points_after_refine": 0,
            "unassigned_points_after_refine": 0,
            "rescued_points": 0,
            "final_unassigned_points": 0,
            "blob_count_total": 0,
            "blob_count_kept": 0,
            "blob_reject_small": 0,
            "blob_reject_large": 0,
            "blob_reject_intensity": 0,
            "blob_reject_shape": 0,
            "blob_centroids_all_ras": empty_ras,
            "blob_centroids_kept_ras": empty_ras,
            "blob_centroids_rejected_ras": empty_ras,
            "blob_labelmap_kji": np.zeros((0,), dtype=np.uint16),
            "metal_depth_all_mm": np.empty((0,), dtype=float),
            "metal_depth_values_mm": np.empty((0,), dtype=float),
            "in_mask_depth_values_mm": np.empty((0,), dtype=float),
            "in_mask_points_ras": empty_ras,
            "profile_ms": default_profile,
            "lines": [],
        }

    gating_mask_kji = preview.get("gating_mask_kji")
    if gating_mask_kji is None:
        gating_mask_kji = preview.get("head_mask_kji")
    inside_method = str(preview.get("inside_method", "none"))
    gating_mask_type = str(preview.get("gating_mask_type", "none"))

    ijk_all = np.asarray(preview.get("in_mask_ijk_kji"), dtype=int).reshape(-1, 3)
    head_mask_kept_count_full = int(ijk_all.shape[0])
    metal_in_head_count = int(preview.get("metal_in_head_count", head_mask_kept_count_full))
    depth_kept_count = int(preview.get("depth_kept_count", head_mask_kept_count_full))
    metal_depth_all_mm = np.asarray(preview.get("metal_depth_all_mm"), dtype=float).reshape(-1)
    metal_depth_values_mm = np.asarray(preview.get("metal_depth_values_mm"), dtype=float).reshape(-1)
    sampled_depth_values_mm = metal_depth_values_mm

    t_subsample0 = time.perf_counter()
    if head_mask_kept_count_full > int(max_points):
        rng = np.random.default_rng(0)
        pick = rng.choice(head_mask_kept_count_full, size=int(max_points), replace=False)
        ijk_kji = ijk_all[pick]
        if metal_depth_values_mm.size == head_mask_kept_count_full:
            sampled_depth_values_mm = metal_depth_values_mm[pick]
        else:
            sampled_depth_values_mm = np.empty((0,), dtype=float)
    else:
        ijk_kji = ijk_all
        if metal_depth_values_mm.size != head_mask_kept_count_full:
            sampled_depth_values_mm = np.empty((0,), dtype=float)
    subsample_ms = (time.perf_counter() - t_subsample0) * 1000.0

    t_depthmap0 = time.perf_counter()
    head_depth_kji = preview.get("head_distance_map_kji")
    if head_depth_kji is not None:
        head_depth_kji = np.asarray(head_depth_kji, dtype=np.float32)
    else:
        head_depth_kji = _build_head_depth_map_kji(gating_mask_kji, spacing_xyz=spacing_xyz)
    if sampled_depth_values_mm.size == 0 and head_depth_kji is not None and ijk_kji.shape[0] > 0:
        sampled_depth_values_mm = np.asarray(
            head_depth_kji[ijk_kji[:, 0], ijk_kji[:, 1], ijk_kji[:, 2]],
            dtype=float,
        )
    depth_map_ms = (time.perf_counter() - t_depthmap0) * 1000.0

    t_ras0 = time.perf_counter()
    points = np.asarray(ijk_kji_to_ras_fn(ijk_kji), dtype=float).reshape(-1, 3)
    point_weights = np.ones((points.shape[0],), dtype=float)
    blob_count_total = 0
    blob_count_kept = 0
    blob_reject_small = 0
    blob_reject_large = 0
    blob_reject_intensity = 0
    blob_reject_shape = 0
    blob_centroids_all_ras = empty_ras
    blob_centroids_kept_ras = empty_ras
    blob_centroids_rejected_ras = empty_ras
    blob_labelmap_kji = np.zeros((0,), dtype=np.uint16)
    if mode == "blob_centroid":
        t_blob0 = time.perf_counter()
        metal_blob_mask_kji = preview.get("metal_in_gate_mask_kji")
        if metal_blob_mask_kji is None:
            metal_blob_mask_kji = preview.get("metal_depth_pass_mask_kji")
        metal_mask_kji = np.asarray(preview.get("metal_mask_kji"), dtype=bool)
        if gating_mask_kji is not None:
            debug_blob_mask_kji = np.logical_and(metal_mask_kji, np.asarray(gating_mask_kji, dtype=bool))
        else:
            debug_blob_mask_kji = metal_mask_kji
        # Debug centroids/labels are extracted from all in-head metal voxels.
        # Fitting still uses depth-pass metal blobs below.
        # Use face-connectivity (6-neighbor) so contacts that only touch
        # diagonally across slices do not collapse into a single 3D blob.
        blob_debug = extract_blob_candidates(
            metal_mask_kji=debug_blob_mask_kji,
            arr_kji=arr_kji,
            depth_map_kji=head_depth_kji,
            ijk_kji_to_ras_fn=ijk_kji_to_ras_fn,
            fully_connected=False,
        )
        blob_raw = extract_blob_candidates(
            metal_mask_kji=metal_blob_mask_kji,
            arr_kji=arr_kji,
            depth_map_kji=head_depth_kji,
            ijk_kji_to_ras_fn=ijk_kji_to_ras_fn,
            fully_connected=False,
        )
        blob_filtered = filter_blob_candidates(
            blob_result=blob_raw,
            min_blob_voxels=int(min_blob_voxels),
            max_blob_voxels=int(max_blob_voxels),
            min_blob_peak_hu=min_blob_peak_hu,
            max_blob_elongation=max_blob_elongation,
        )
        blob_count_total = int(blob_raw.get("blob_count_total", 0))
        blob_count_kept = int(blob_filtered.get("blob_count_kept", 0))
        blob_reject_small = int(blob_filtered.get("blob_reject_small", 0))
        blob_reject_large = int(blob_filtered.get("blob_reject_large", 0))
        blob_reject_intensity = int(blob_filtered.get("blob_reject_intensity", 0))
        blob_reject_shape = int(blob_filtered.get("blob_reject_shape", 0))
        blob_centroids_all_ras = np.asarray(blob_debug.get("blobs", []), dtype=object)
        if blob_centroids_all_ras.size:
            blob_centroids_all_ras = np.asarray(
                [b.get("centroid_ras") for b in blob_debug.get("blobs", []) if b.get("centroid_ras") is not None],
                dtype=float,
            ).reshape(-1, 3)
        else:
            blob_centroids_all_ras = empty_ras
        blob_centroids_kept_ras = np.asarray(blob_filtered.get("blob_centroids_kept_ras", empty_ras), dtype=float).reshape(-1, 3)
        blob_centroids_rejected_ras = np.asarray(blob_filtered.get("blob_centroids_rejected_ras", empty_ras), dtype=float).reshape(-1, 3)
        blob_labelmap_kji = build_blob_labelmap(blob_debug.get("labels_kji"), keep_blob_ids=None)
        points = np.asarray(blob_filtered.get("candidate_points_ras", empty_ras), dtype=float).reshape(-1, 3)
        point_weights = np.asarray(blob_filtered.get("candidate_weights", np.ones((points.shape[0],), dtype=float)), dtype=float)
        sampled_depth_values_mm = np.empty((0,), dtype=float)
        # Blob mode uses depth as a soft feature/weighting signal, not a hard
        # pre-filter. Report depth-kept count as mask-kept for diagnostics.
        depth_kept_count = int(metal_in_head_count)
        blob_ms = (time.perf_counter() - t_blob0) * 1000.0
    in_mask_points_ras = points.copy()
    ras_convert_ms = (time.perf_counter() - t_ras0) * 1000.0
    head_mask_kept_count = int(points.shape[0])
    if mode == "blob_centroid":
        n_support = int(points.shape[0])
        if n_support > 0:
            avg_support_per_line = float(n_support) / float(max(1, int(max_lines)))
            auto_min = int(round(0.60 * avg_support_per_line))
            auto_min = max(3, min(18, auto_min))
            effective_min_inliers = min(int(min_inliers), auto_min, max(3, n_support - 1))
            effective_min_inliers = max(3, effective_min_inliers)
            # Blob-centroid support is sparser/noisier than voxel clouds.
            effective_inlier_radius_mm = max(float(inlier_radius_mm), 2.0)
        else:
            effective_min_inliers = int(min_inliers)
    setup_ms = (time.perf_counter() - t_setup0) * 1000.0

    if head_mask_kept_count == 0:
        return {
            "head_mask_kept_count": 0,
            "gating_mask_type": gating_mask_type,
            "inside_method": inside_method,
            "metal_in_head_count": metal_in_head_count,
            "depth_kept_count": depth_kept_count,
            "gap_reject_count": 0,
            "start_zone_reject_count": 0,
            "duplicate_reject_count": 0,
            "length_reject_count": 0,
            "inlier_reject_count": 0,
            "candidate_points_total": int(candidate_count),
            "candidate_points_after_mask": int(metal_in_head_count),
            "candidate_points_after_depth": int(depth_kept_count),
            "effective_min_inliers": int(effective_min_inliers),
            "effective_inlier_radius_mm": float(effective_inlier_radius_mm),
            "fit1_lines_proposed": 0,
            "fit2_lines_kept": 0,
            "rescue_lines_kept": 0,
            "final_lines_kept": 0,
            "assigned_points_after_refine": 0,
            "unassigned_points_after_refine": 0,
            "rescued_points": 0,
            "final_unassigned_points": 0,
            "blob_count_total": int(blob_count_total),
            "blob_count_kept": int(blob_count_kept),
            "blob_reject_small": int(blob_reject_small),
            "blob_reject_large": int(blob_reject_large),
            "blob_reject_intensity": int(blob_reject_intensity),
            "blob_reject_shape": int(blob_reject_shape),
            "blob_centroids_all_ras": blob_centroids_all_ras,
            "blob_centroids_kept_ras": blob_centroids_kept_ras,
            "blob_centroids_rejected_ras": blob_centroids_rejected_ras,
            "blob_labelmap_kji": blob_labelmap_kji,
            "metal_depth_all_mm": metal_depth_all_mm,
            "metal_depth_values_mm": metal_depth_values_mm,
            "in_mask_depth_values_mm": sampled_depth_values_mm,
            "in_mask_points_ras": in_mask_points_ras,
            "profile_ms": {
                "setup": float(setup_ms),
                "subsample": float(subsample_ms),
                "depth_map": float(depth_map_ms),
                "ras_convert": float(ras_convert_ms),
                "blob_stage": float(blob_ms),
                "exclude": 0.0,
                "first_pass": 0.0,
                "refine": 0.0,
                "rescue": 0.0,
                "total": float((time.perf_counter() - t0) * 1000.0),
            },
            "lines": [],
        }

    # First pass: RANSAC over remaining points, removing inliers after each accepted line.
    remaining = points
    t_exclude0 = time.perf_counter()
    if exclude_segments:
        keep = np.ones((remaining.shape[0],), dtype=bool)
        for seg in exclude_segments:
            seg_mask = _distance_to_segment_mask(
                remaining,
                seg["start_ras"],
                seg["end_ras"],
                radius_mm=exclude_radius_mm,
            )
            keep &= ~seg_mask
        remaining = remaining[keep]
        point_weights = point_weights[keep]
    exclude_ms = (time.perf_counter() - t_exclude0) * 1000.0

    support_points = remaining.copy()
    support_weights = point_weights.copy()
    t_firstpass0 = time.perf_counter()
    lines, first_pass_stats = _fit_line_proposals(
        points=remaining,
        center_ras=np.asarray(center_ras, dtype=float),
        head_depth_kji=head_depth_kji,
        ras_to_ijk_fn=ras_to_ijk_fn,
        gating_mask_kji=gating_mask_kji,
        inlier_radius_mm=float(effective_inlier_radius_mm),
        min_length_mm=float(min_length_mm),
        min_inliers=int(effective_min_inliers),
        ransac_iterations=int(ransac_iterations),
        max_lines=int(max_lines),
    )
    first_pass_ms = (time.perf_counter() - t_firstpass0) * 1000.0
    fit1_lines_proposed = int(len(lines))

    # Second pass: exclusive reassignment to reduce duplicated/overlapping shanks.
    t_refine0 = time.perf_counter()
    lines_refined, refine_stats = _refine_lines_exclusive(
        arr_kji=arr_kji,
        spacing_xyz=spacing_xyz,
        ras_to_ijk_fn=ras_to_ijk_fn,
        center_ras=np.asarray(center_ras, dtype=float),
        head_depth_kji=head_depth_kji,
        support_points=support_points,
        raw_lines=lines,
        inlier_radius_mm=float(effective_inlier_radius_mm),
        min_length_mm=float(min_length_mm),
        min_inliers=int(effective_min_inliers),
        gating_mask_kji=gating_mask_kji,
        models_by_id=models_by_id,
        min_model_score=min_model_score,
        min_metal_depth_mm=float(min_metal_depth_mm),
        start_zone_window_mm=float(start_zone_window_mm),
        support_weights=support_weights,
        apply_start_zone_prior=bool(apply_start_zone_prior),
    )
    refine_ms = (time.perf_counter() - t_refine0) * 1000.0
    gap_reject_count = int(refine_stats.get("gap_reject_count", 0))
    start_zone_reject_count = int(refine_stats.get("start_zone_reject_count", 0))
    duplicate_reject_count = int(refine_stats.get("duplicate_reject_count", 0))
    length_reject_count = int(first_pass_stats.get("length_reject_count", 0)) + int(refine_stats.get("length_reject_count", 0))
    inlier_reject_count = int(refine_stats.get("inlier_reject_count", 0))
    assigned_mask = np.asarray(
        refine_stats.get("assigned_mask", np.zeros((support_points.shape[0],), dtype=bool)),
        dtype=bool,
    ).reshape(-1)
    if assigned_mask.size != support_points.shape[0]:
        assigned_mask = _assigned_mask_for_lines(
            points=support_points,
            lines=lines_refined,
            radius_mm=max(1.0, float(inlier_radius_mm)),
        )

    fit2_lines_kept = int(len(lines_refined))
    assigned_points_after_refine = int(np.count_nonzero(assigned_mask))
    unassigned_points_after_refine = int(max(0, support_points.shape[0] - assigned_points_after_refine))

    # Rescue pass over still-unassigned support points.
    rescue_lines = []
    rescued_points = 0
    rescue_ms = 0.0
    if bool(enable_rescue_pass) and unassigned_points_after_refine > 0:
        t_rescue0 = time.perf_counter()
        rescue_points = support_points[~assigned_mask]
        rescue_weights = support_weights[~assigned_mask]
        rescue_min_inliers = max(
            5,
            int(round(float(effective_min_inliers) * float(max(0.1, rescue_min_inliers_scale)))),
        )
        rescue_raw, rescue_first_stats = _fit_line_proposals(
            points=rescue_points,
            center_ras=np.asarray(center_ras, dtype=float),
            head_depth_kji=head_depth_kji,
            ras_to_ijk_fn=ras_to_ijk_fn,
            gating_mask_kji=gating_mask_kji,
            inlier_radius_mm=float(effective_inlier_radius_mm),
            min_length_mm=float(min_length_mm),
            min_inliers=int(rescue_min_inliers),
            ransac_iterations=int(ransac_iterations),
            max_lines=int(max(0, rescue_max_lines)),
        )
        length_reject_count += int(rescue_first_stats.get("length_reject_count", 0))
        rescue_lines, rescue_refine_stats = _refine_lines_exclusive(
            arr_kji=arr_kji,
            spacing_xyz=spacing_xyz,
            ras_to_ijk_fn=ras_to_ijk_fn,
            center_ras=np.asarray(center_ras, dtype=float),
            head_depth_kji=head_depth_kji,
            support_points=rescue_points,
            raw_lines=rescue_raw,
            inlier_radius_mm=float(effective_inlier_radius_mm),
            min_length_mm=float(min_length_mm),
            min_inliers=int(rescue_min_inliers),
            gating_mask_kji=gating_mask_kji,
            models_by_id=models_by_id,
            min_model_score=min_model_score,
            min_metal_depth_mm=float(min_metal_depth_mm),
            start_zone_window_mm=float(start_zone_window_mm),
            support_weights=rescue_weights,
            apply_start_zone_prior=bool(apply_start_zone_prior),
        )
        gap_reject_count += int(rescue_refine_stats.get("gap_reject_count", 0))
        start_zone_reject_count += int(rescue_refine_stats.get("start_zone_reject_count", 0))
        duplicate_reject_count += int(rescue_refine_stats.get("duplicate_reject_count", 0))
        length_reject_count += int(rescue_refine_stats.get("length_reject_count", 0))
        inlier_reject_count += int(rescue_refine_stats.get("inlier_reject_count", 0))
        rescue_assigned_mask = np.asarray(
            rescue_refine_stats.get("assigned_mask", np.zeros((rescue_points.shape[0],), dtype=bool)),
            dtype=bool,
        ).reshape(-1)
        rescued_points = int(np.count_nonzero(rescue_assigned_mask))
        rescue_ms = (time.perf_counter() - t_rescue0) * 1000.0

    all_lines = list(lines_refined) + list(rescue_lines)
    all_lines, extra_dup = _suppress_conflicting_lines_by_support(
        lines=all_lines,
        support_points=support_points,
        support_radius_mm=max(1.0, float(inlier_radius_mm)),
        overlap_threshold=0.60,
    )
    duplicate_reject_count += int(extra_dup)
    lines = sorted(
        all_lines,
        key=lambda line: (
            float(line.get("best_model_score", 0.0)) if line.get("best_model_score") is not None else -1e9,
            float(line.get("inside_fraction", 0.0)),
            float(line.get("support_weight", 0.0)),
            float(line.get("inlier_count", 0)),
            float(line.get("depth_span_mm", 0.0)),
            -float(line.get("rms_mm", 999.0)),
        ),
        reverse=True,
    )
    final_assigned_mask = _assigned_mask_for_lines(
        points=support_points,
        lines=lines,
        radius_mm=max(1.0, float(inlier_radius_mm)),
    )
    final_unassigned_points = int(max(0, support_points.shape[0] - int(np.count_nonzero(final_assigned_mask))))

    return {
        "head_mask_kept_count": head_mask_kept_count,
        "gating_mask_type": gating_mask_type,
        "inside_method": inside_method,
        "metal_in_head_count": metal_in_head_count,
        "depth_kept_count": depth_kept_count,
        "candidate_points_total": int(candidate_count),
        "candidate_points_after_mask": int(metal_in_head_count),
        "candidate_points_after_depth": int(depth_kept_count),
        "effective_min_inliers": int(effective_min_inliers),
        "effective_inlier_radius_mm": float(effective_inlier_radius_mm),
        "fit1_lines_proposed": int(fit1_lines_proposed),
        "fit2_lines_kept": int(fit2_lines_kept),
        "rescue_lines_kept": int(len(rescue_lines)),
        "final_lines_kept": int(len(lines)),
        "assigned_points_after_refine": int(assigned_points_after_refine),
        "unassigned_points_after_refine": int(unassigned_points_after_refine),
        "rescued_points": int(rescued_points),
        "final_unassigned_points": int(final_unassigned_points),
        "gap_reject_count": int(gap_reject_count),
        "start_zone_reject_count": int(start_zone_reject_count),
        "duplicate_reject_count": int(duplicate_reject_count),
        "length_reject_count": int(length_reject_count),
        "inlier_reject_count": int(inlier_reject_count),
        "blob_count_total": int(blob_count_total),
        "blob_count_kept": int(blob_count_kept),
        "blob_reject_small": int(blob_reject_small),
        "blob_reject_large": int(blob_reject_large),
        "blob_reject_intensity": int(blob_reject_intensity),
        "blob_reject_shape": int(blob_reject_shape),
        "blob_centroids_all_ras": blob_centroids_all_ras,
        "blob_centroids_kept_ras": blob_centroids_kept_ras,
        "blob_centroids_rejected_ras": blob_centroids_rejected_ras,
        "blob_labelmap_kji": blob_labelmap_kji,
        "metal_depth_all_mm": metal_depth_all_mm,
        "metal_depth_values_mm": metal_depth_values_mm,
        "in_mask_depth_values_mm": sampled_depth_values_mm,
        "in_mask_points_ras": in_mask_points_ras,
        "profile_ms": {
            "setup": float(setup_ms),
            "subsample": float(subsample_ms),
            "depth_map": float(depth_map_ms),
            "ras_convert": float(ras_convert_ms),
            "blob_stage": float(blob_ms),
            "exclude": float(exclude_ms),
            "first_pass": float(first_pass_ms),
            "refine": float(refine_ms),
            "rescue": float(rescue_ms),
            "total": float((time.perf_counter() - t0) * 1000.0),
        },
        "lines": lines,
    }
