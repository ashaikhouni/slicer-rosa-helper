"""LoG-based guided-fit engine.

Given a *planned* trajectory (entry → target, RAS), snap it to the
actual imaged shank using the same scanner-agnostic LoG σ=1 signal
that powers the Auto Fit (``contact_pitch_v1``) detector. Replaces
the legacy HU-threshold / ROI-sweep guided-fit pipeline.

Per trajectory:
  1. Collect LoG regional-minima blobs (pre-computed once per volume).
  2. Keep blobs inside a cylindrical ROI around the planned axis
     (``perp ≤ roi_radius_mm``, ``along`` within the planned segment
     ± a short end-pad).
  3. PCA through the cylinder inliers → corrected axis.
  4. Reject if the axis tilts more than ``max_angle_deg`` off the
     planned axis or if the corrected midpoint shifts more than
     ``max_lateral_shift_mm`` off the planned midpoint.
  5. Endpoints come from the shallowest/deepest inlier projections
     on the corrected axis; the deep tip is then pushed outward by
     axis-directed LoG sampling (same rule as contact_pitch_v1 step 6).

Returns a dict with ``success`` + RAS endpoints + diagnostics.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import contact_pitch_v1_fit as cpfit


DEFAULT_ROI_RADIUS_MM = 5.0
DEFAULT_MAX_ANGLE_DEG = 12.0
DEFAULT_MAX_LATERAL_SHIFT_MM = 6.0
DEFAULT_MIN_INLIERS = 4
# End-pad along the planned axis: keep blobs up to this far past the
# planned entry/target so the detected extent isn't clipped by a
# slightly short plan.
AXIS_END_PAD_MM = 8.0
# Guided fit trusts the planned/seeded end more than Auto Fit does,
# so the deep-end refinement is bounded to a small extension. This
# still rescues merged-contact shafts (T2 RAI-style) without letting
# the walker thread into brain-tissue LoG peaks 30+ mm past the
# real tip (seen on T2 LAI with the default 40 mm bound).
DEEP_REFINE_MAX_EXTEND_MM = 5.0


def _unit(v):
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def compute_features(img, ras_to_ijk_mat):
    """One-time preprocessing: hull, head-distance, and LoG σ=1.

    The LoG volume is what the fit reads; hull / head-distance are
    kept for callers that want to gate by intracranial depth.
    Matches ``contact_pitch_v1`` exactly so the two paths can share a
    single preprocessing pass when invoked together.
    """
    hull_arr, intracranial, dist_arr = cpfit.build_masks(img)
    log1 = cpfit.log_sigma(img, sigma_mm=cpfit.LOG_SIGMA_MM)
    return {
        "log": log1,
        "hull": hull_arr,
        "intracranial": intracranial,
        "head_distance": dist_arr,
    }


def _extract_blob_cloud_ras(log_arr, ijk_to_ras_mat,
                              threshold=cpfit.LOG_BLOB_THRESHOLD):
    """Return the RAS centroids and amplitudes of all LoG regional
    minima strong enough to be contact candidates.
    """
    blobs = cpfit.extract_blobs(log_arr, threshold=threshold)
    if not blobs:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)
    kji = np.array([b["kji"] for b in blobs], dtype=float)
    amps = np.array([b["amp"] for b in blobs], dtype=float)
    # kji → ras: pack as (i, j, k, 1) to multiply by the 4×4 matrix.
    ij_k = np.stack([kji[:, 2], kji[:, 1], kji[:, 0]], axis=1)
    h = np.concatenate([ij_k, np.ones((ij_k.shape[0], 1))], axis=1)
    ras = (ijk_to_ras_mat @ h.T).T[:, :3]
    return ras, amps


def fit_trajectory(planned_start_ras, planned_end_ras, features,
                    ijk_to_ras_mat, ras_to_ijk_mat,
                    roi_radius_mm=DEFAULT_ROI_RADIUS_MM,
                    max_angle_deg=DEFAULT_MAX_ANGLE_DEG,
                    max_lateral_shift_mm=DEFAULT_MAX_LATERAL_SHIFT_MM,
                    min_inliers=DEFAULT_MIN_INLIERS,
                    log_threshold=cpfit.LOG_BLOB_THRESHOLD):
    """Snap one planned trajectory to the imaged shank.

    Returns a dict with ``success`` and — on success — ``start_ras``,
    ``end_ras``, plus per-fit diagnostics. On failure ``success=False``
    and ``reason`` explains why.
    """
    planned_start = np.asarray(planned_start_ras, dtype=float).reshape(3)
    planned_end = np.asarray(planned_end_ras, dtype=float).reshape(3)
    planned_vec = planned_end - planned_start
    planned_length = float(np.linalg.norm(planned_vec))
    if planned_length < 1e-3:
        return {"success": False, "reason": "planned trajectory has zero length"}
    planned_axis = planned_vec / planned_length

    pts_ras, amps = _extract_blob_cloud_ras(
        features["log"], np.asarray(ijk_to_ras_mat, dtype=float),
        threshold=log_threshold,
    )
    if pts_ras.shape[0] == 0:
        return {"success": False, "reason": "no LoG blobs in volume"}

    # Cylinder filter along the planned axis: perp ≤ roi_radius_mm,
    # along within [-pad, length + pad] from the planned start.
    diffs = pts_ras - planned_start
    along = diffs @ planned_axis
    perp = diffs - np.outer(along, planned_axis)
    perp_d = np.linalg.norm(perp, axis=1)
    in_roi = (
        (perp_d <= float(roi_radius_mm))
        & (along >= -AXIS_END_PAD_MM)
        & (along <= planned_length + AXIS_END_PAD_MM)
    )
    inlier_pts = pts_ras[in_roi]
    inlier_amps = amps[in_roi]
    n_inliers = int(inlier_pts.shape[0])
    if n_inliers < int(min_inliers):
        return {
            "success": False,
            "reason": f"too few LoG blobs in ROI ({n_inliers} < {int(min_inliers)})",
            "n_inliers": n_inliers,
        }

    # PCA through inliers → corrected axis. Weight each blob by its
    # amplitude so bright contacts dominate the fit.
    w = inlier_amps / float(inlier_amps.sum() or 1.0)
    centroid = np.sum(inlier_pts * w[:, None], axis=0)
    X = inlier_pts - centroid
    Xw = X * w[:, None]
    _U, _S, Vt = np.linalg.svd(Xw, full_matrices=False)
    fit_axis = _unit(Vt[0])
    # Keep the fitted axis pointing roughly the same direction as the
    # planned axis so shallow/deep endpoints stay consistent.
    if float(np.dot(fit_axis, planned_axis)) < 0:
        fit_axis = -fit_axis
    cos = float(np.clip(abs(np.dot(fit_axis, planned_axis)), 0.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cos)))
    if angle_deg > float(max_angle_deg):
        return {
            "success": False,
            "reason": f"axis tilt {angle_deg:.1f}° > {float(max_angle_deg):.1f}°",
            "n_inliers": n_inliers,
            "angle_deg": angle_deg,
        }

    # Lateral shift of the fit midpoint vs the planned midpoint,
    # measured perpendicular to the planned axis.
    planned_mid = 0.5 * (planned_start + planned_end)
    mid_offset = centroid - planned_mid
    along_mid = float(np.dot(mid_offset, planned_axis))
    lat_offset = mid_offset - along_mid * planned_axis
    lateral_shift_mm = float(np.linalg.norm(lat_offset))
    if lateral_shift_mm > float(max_lateral_shift_mm):
        return {
            "success": False,
            "reason": (
                f"midpoint lateral shift {lateral_shift_mm:.2f} mm > "
                f"{float(max_lateral_shift_mm):.2f} mm"
            ),
            "n_inliers": n_inliers,
            "angle_deg": angle_deg,
            "lateral_shift_mm": lateral_shift_mm,
        }

    # Endpoints: shallowest/deepest inlier projections on the fit axis.
    proj_fit = (inlier_pts - centroid) @ fit_axis
    shallow_along = float(proj_fit.min())
    deep_along = float(proj_fit.max())
    shallow_ras = centroid + shallow_along * fit_axis
    deep_ras = centroid + deep_along * fit_axis

    # Orient shallow → deep: shallow end is the one closer to the
    # planned entry (which is typically outside the brain).
    d_shallow = float(np.linalg.norm(shallow_ras - planned_start))
    d_deep = float(np.linalg.norm(deep_ras - planned_start))
    if d_shallow > d_deep:
        shallow_ras, deep_ras = deep_ras, shallow_ras

    # Axis-directed deep-end refinement: walk outward from the deep
    # tip along the fit axis, extending as long as LoG stays strong.
    # Bounded to ``DEEP_REFINE_MAX_EXTEND_MM`` — guided fit has a
    # strong axis + endpoint prior so the refinement only needs to
    # rescue merged-contact shafts, not re-detect the tip from scratch.
    rec = {"start_ras": shallow_ras, "end_ras": deep_ras}
    try:
        refined_end = cpfit._refine_deep_end_via_axis_log(
            rec, features["log"], np.asarray(ras_to_ijk_mat, dtype=float),
            max_extend_mm=DEEP_REFINE_MAX_EXTEND_MM,
        )
        if refined_end is not None:
            deep_ras = np.asarray(refined_end, dtype=float)
    except Exception:
        # Refinement is best-effort — a failure just means we keep
        # the PCA-derived deep endpoint.
        pass

    fit_length = float(np.linalg.norm(deep_ras - shallow_ras))

    return {
        "success": True,
        "start_ras": [float(v) for v in shallow_ras],
        "end_ras": [float(v) for v in deep_ras],
        "axis_ras": [float(v) for v in fit_axis],
        "n_inliers": n_inliers,
        "angle_deg": angle_deg,
        "lateral_shift_mm": lateral_shift_mm,
        "length_mm": fit_length,
        "roi_radius_mm": float(roi_radius_mm),
    }
