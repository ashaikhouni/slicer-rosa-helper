"""LoG-based guided-fit engine.

Given a planned/seeded trajectory (entry → target, RAS), snap it to
the actual imaged shank using the same scanner-agnostic LoG σ=1 signal
and bolt-anchor machinery that powers Auto Fit (``contact_pitch_v1``).

The aim is to produce the same output shape as Auto Fit — bolt tip,
skull entry, deep tip — so downstream modules can't tell whether a
trajectory came from Auto Fit or Guided Fit.

Per trajectory:
  1. Wide cylinder around the seed axis collects LoG regional-minima
     blobs (``roi_radius_mm`` perp, ``along`` within the seed segment
     ± end-pad).
  2. Amplitude-weighted PCA on the wide set → rough axis.
  3. Tight cylinder (``TIGHT_PERP_TOL_MM``) around the rough axis
     rejects cross-shank contamination; re-fit PCA on the survivors
     → final axis. This mirrors the "wide-then-refine" pattern that
     contact_pitch_v1's arbitrator uses.
  4. Gate on axis tilt + midpoint lateral shift.
  5. Bolt anchor (``anchor_trajectory_to_bolt``): finds a hull-touching
     LoG-bright CC along the fitted axis and produces both
     ``bolt_tip_ras`` (outermost bolt voxel, shallow-side) and
     ``skull_entry_ras`` (innermost bolt voxel in the skull/dura band).
  6. Endpoints: shallowest/deepest inlier projections on the fit
     axis, then the deep tip is refined via axis-directed LoG sampling
     (bounded to ``DEEP_REFINE_MAX_EXTEND_MM`` — we have a strong
     axis prior).

Fails gracefully: if no bolt anchors within tolerances, returns the
PCA-based endpoints only and flags ``bolt_anchored=False``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import contact_pitch_v1_fit as cpfit


DEFAULT_ROI_RADIUS_MM = 5.0
DEFAULT_MAX_ANGLE_DEG = 12.0
DEFAULT_MAX_LATERAL_SHIFT_MM = 6.0
DEFAULT_MIN_INLIERS = 4

# End-pad along the seed axis: accept blobs up to this far past the
# planned entry/target so the detected extent isn't clipped by a
# slightly short seed.
AXIS_END_PAD_MM = 8.0

# Tight perp tolerance for the PCA re-fit after an initial wide-axis
# pass. Matches contact_pitch_v1's walker pitch tolerance so the
# inlier set mirrors what Auto Fit would accept.
TIGHT_PERP_TOL_MM = 1.5

# Guided fit trusts the seed endpoints more than Auto Fit does, so
# the deep-end refinement is bounded to a small extension. This still
# rescues merged-contact shafts (T2 RAI-style) without letting the
# walker thread into brain-tissue LoG peaks 30+ mm past the real tip.
DEEP_REFINE_MAX_EXTEND_MM = 5.0


def _unit(v):
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def compute_features(img, ijk_to_ras_mat, spacing_xyz=None):
    """One-time preprocessing per volume. Computes the full set of
    arrays Auto Fit uses — hull, head distance, LoG σ=1, blob cloud,
    and bolt candidates — so a batch of seeds share one preprocessing
    pass. ``ijk_to_ras_mat`` is needed here because bolt extraction
    has to emit RAS coordinates for the downstream anchor call.
    """
    ijk_to_ras_mat = np.asarray(ijk_to_ras_mat, dtype=float)
    hull_arr, intracranial, dist_arr = cpfit.build_masks(img)
    log1 = cpfit.log_sigma(img, sigma_mm=cpfit.LOG_SIGMA_MM)

    pts_ras, amps = _extract_blob_cloud_ras(log1, ijk_to_ras_mat)

    spacing = spacing_xyz
    if spacing is None:
        try:
            spacing = tuple(float(v) for v in img.GetSpacing())
        except Exception:
            spacing = (1.0, 1.0, 1.0)

    bolts, bolt_mask = cpfit.extract_bolt_candidates(
        log1, dist_arr, ijk_to_ras_mat, spacing,
    )

    return {
        "log": log1,
        "hull": hull_arr,
        "intracranial": intracranial,
        "head_distance": dist_arr,
        "blob_pts_ras": pts_ras,
        "blob_amps": amps,
        "bolts": bolts,
        "bolt_mask": bolt_mask,
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
    # kji → ras via the 4×4 ijk-to-ras matrix.
    ij_k = np.stack([kji[:, 2], kji[:, 1], kji[:, 0]], axis=1)
    h = np.concatenate([ij_k, np.ones((ij_k.shape[0], 1))], axis=1)
    ras = (ijk_to_ras_mat @ h.T).T[:, :3]
    return ras, amps


def _pca_axis(points, weights):
    """Amplitude-weighted PCA principal axis of an Nx3 RAS point cloud.
    Returns (centroid, axis_unit).
    """
    w = weights / float(weights.sum() or 1.0)
    centroid = np.sum(points * w[:, None], axis=0)
    X = points - centroid
    Xw = X * w[:, None]
    _U, _S, Vt = np.linalg.svd(Xw, full_matrices=False)
    return centroid, _unit(Vt[0])


def fit_trajectory(planned_start_ras, planned_end_ras, features,
                    ijk_to_ras_mat, ras_to_ijk_mat,
                    roi_radius_mm=DEFAULT_ROI_RADIUS_MM,
                    max_angle_deg=DEFAULT_MAX_ANGLE_DEG,
                    max_lateral_shift_mm=DEFAULT_MAX_LATERAL_SHIFT_MM,
                    min_inliers=DEFAULT_MIN_INLIERS):
    """Snap one seeded trajectory to the imaged shank.

    Returns a dict with ``success`` and — on success — ``start_ras``
    (bolt tip / outer end; falls back to the PCA shallow endpoint if
    no bolt is found), ``end_ras`` (deep tip, refined), optionally
    ``skull_entry_ras``, ``bolt_tip_ras``, plus per-fit diagnostics.
    """
    planned_start = np.asarray(planned_start_ras, dtype=float).reshape(3)
    planned_end = np.asarray(planned_end_ras, dtype=float).reshape(3)
    planned_vec = planned_end - planned_start
    planned_length = float(np.linalg.norm(planned_vec))
    if planned_length < 1e-3:
        return {"success": False, "reason": "planned trajectory has zero length"}
    planned_axis = planned_vec / planned_length

    pts_ras = features.get("blob_pts_ras")
    amps = features.get("blob_amps")
    if pts_ras is None or amps is None:
        # Lazy extraction if compute_features was skipped
        pts_ras, amps = _extract_blob_cloud_ras(
            features["log"], np.asarray(ijk_to_ras_mat, dtype=float),
        )
    if pts_ras.shape[0] == 0:
        return {"success": False, "reason": "no LoG blobs in volume"}

    # ---- Pass 1: wide cylinder around the seed axis ------------------
    diffs = pts_ras - planned_start
    along = diffs @ planned_axis
    perp = diffs - np.outer(along, planned_axis)
    perp_d = np.linalg.norm(perp, axis=1)
    in_roi = (
        (perp_d <= float(roi_radius_mm))
        & (along >= -AXIS_END_PAD_MM)
        & (along <= planned_length + AXIS_END_PAD_MM)
    )
    wide_pts = pts_ras[in_roi]
    wide_amps = amps[in_roi]
    n_wide = int(wide_pts.shape[0])
    if n_wide < int(min_inliers):
        return {
            "success": False,
            "reason": f"too few LoG blobs in ROI ({n_wide} < {int(min_inliers)})",
            "n_inliers": n_wide,
        }

    centroid_rough, axis_rough = _pca_axis(wide_pts, wide_amps)
    if float(np.dot(axis_rough, planned_axis)) < 0:
        axis_rough = -axis_rough

    # ---- Pass 2: tight cylinder around the rough axis ----------------
    # This drops cross-shank contacts that drifted into the wide ROI
    # and locks PCA onto the MAIN LINE of contacts belonging to the
    # seeded shank.
    diffs_rough = wide_pts - centroid_rough
    along_rough = diffs_rough @ axis_rough
    perp_rough = diffs_rough - np.outer(along_rough, axis_rough)
    perp_rough_d = np.linalg.norm(perp_rough, axis=1)
    tight_mask = perp_rough_d <= TIGHT_PERP_TOL_MM
    tight_pts = wide_pts[tight_mask]
    tight_amps = wide_amps[tight_mask]
    n_tight = int(tight_pts.shape[0])
    if n_tight < int(min_inliers):
        # Not enough contacts sit on the main line; fall back to the
        # wide set but flag that the refit was degraded.
        tight_pts = wide_pts
        tight_amps = wide_amps
        n_tight = n_wide
        tight_pass = False
    else:
        tight_pass = True

    centroid, fit_axis = _pca_axis(tight_pts, tight_amps)
    if float(np.dot(fit_axis, planned_axis)) < 0:
        fit_axis = -fit_axis

    cos = float(np.clip(abs(np.dot(fit_axis, planned_axis)), 0.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cos)))
    if angle_deg > float(max_angle_deg):
        return {
            "success": False,
            "reason": f"axis tilt {angle_deg:.1f}° > {float(max_angle_deg):.1f}°",
            "n_inliers": n_tight,
            "angle_deg": angle_deg,
        }

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
            "n_inliers": n_tight,
            "angle_deg": angle_deg,
            "lateral_shift_mm": lateral_shift_mm,
        }

    # ---- Endpoints from inlier projections + axis-LoG refinement ----
    proj_fit = (tight_pts - centroid) @ fit_axis
    shallow_along = float(proj_fit.min())
    deep_along = float(proj_fit.max())
    shallow_ras = centroid + shallow_along * fit_axis
    deep_ras = centroid + deep_along * fit_axis

    # Orient shallow → deep based on proximity to the planned start.
    if (np.linalg.norm(shallow_ras - planned_start)
            > np.linalg.norm(deep_ras - planned_start)):
        shallow_ras, deep_ras = deep_ras, shallow_ras
        fit_axis = -fit_axis

    rec = {"start_ras": shallow_ras, "end_ras": deep_ras}
    try:
        refined_end = cpfit._refine_deep_end_via_axis_log(
            rec, features["log"], np.asarray(ras_to_ijk_mat, dtype=float),
            max_extend_mm=DEEP_REFINE_MAX_EXTEND_MM,
        )
        if refined_end is not None:
            deep_ras = np.asarray(refined_end, dtype=float)
    except Exception:
        pass

    # ---- Bolt anchor: produces bolt_tip_ras + skull_entry_ras -------
    bolt_tip_ras = None
    skull_entry_ras = None
    bolt_n_vox = 0
    bolts = features.get("bolts") or []
    if bolts:
        try:
            new_start, skull_entry, bolt = cpfit.anchor_trajectory_to_bolt(
                shallow_ras, deep_ras, bolts,
            )
            if new_start is not None:
                bolt_tip_ras = np.asarray(new_start, dtype=float)
                if skull_entry is not None:
                    skull_entry_ras = np.asarray(skull_entry, dtype=float)
                bolt_n_vox = int(bolt.get("n_vox", 0))
        except Exception:
            pass

    # Prefer the bolt_tip as the outer endpoint when available — that
    # matches Auto Fit's convention and lets downstream code compute
    # intracranial length as |skull_entry → deep|.
    start_out = bolt_tip_ras if bolt_tip_ras is not None else shallow_ras

    fit_length = float(np.linalg.norm(deep_ras - start_out))

    result = {
        "success": True,
        "start_ras": [float(v) for v in start_out],
        "end_ras": [float(v) for v in deep_ras],
        "axis_ras": [float(v) for v in fit_axis],
        "n_inliers": n_tight,
        "n_wide_inliers": n_wide,
        "tight_refit": tight_pass,
        "angle_deg": angle_deg,
        "lateral_shift_mm": lateral_shift_mm,
        "length_mm": fit_length,
        "roi_radius_mm": float(roi_radius_mm),
        "bolt_anchored": bolt_tip_ras is not None,
        "bolt_n_vox": bolt_n_vox,
    }
    if skull_entry_ras is not None:
        result["skull_entry_ras"] = [float(v) for v in skull_entry_ras]
    if bolt_tip_ras is not None:
        result["bolt_tip_ras"] = [float(v) for v in bolt_tip_ras]
    return result
