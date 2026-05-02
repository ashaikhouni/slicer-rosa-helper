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


def compute_features(img, ijk_to_ras_mat, ras_to_ijk_mat=None, spacing_xyz=None):
    """One-time preprocessing per volume. Runs the SAME pipeline-entry
    canonicalization (resample-to-1mm + anisotropic anti-alias +
    HU clamp) Auto Fit uses, then computes the same feature set —
    hull, head distance, LoG σ=1, blob cloud, Frangi σ=1, CT array,
    and bolt candidates from the unified metal-evidence pool — so a
    batch of seeds share one preprocessing pass and so the same
    scoring rubric (frangi, frac_strong_metal, bolt_source) can be
    applied to guided-fit results.

    Because canonicalization may resample the volume, the canonical
    img + IJK↔RAS matrices are stamped into the returned dict under
    keys ``img`` / ``ijk_to_ras_mat`` / ``ras_to_ijk_mat``. Callers
    MUST use these for any subsequent ``fit_trajectory`` call so the
    canonical grid is consistent with the feature kernels — passing
    the original (pre-resample) matrices would compute trajectories
    on a grid that doesn't match where the LoG / Frangi peaks live.
    """
    import SimpleITK as sitk
    img, ijk_to_ras_mat, ras_to_ijk_mat = cpfit.prepare_volume(
        img, ijk_to_ras_mat, ras_to_ijk_mat,
    )
    hull_arr, intracranial, dist_arr = cpfit.build_masks(img)
    log1 = cpfit.log_sigma(img, sigma_mm=cpfit.LOG_SIGMA_MM)
    frangi_s1 = cpfit.frangi_single(img, sigma=cpfit.FRANGI_STAGE1_SIGMA)
    ct_arr_kji = sitk.GetArrayFromImage(img).astype(np.float32)

    pts_ras, amps = cpfit.extract_blob_cloud_ras(log1, ijk_to_ras_mat)

    # ``spacing_xyz`` overrides the (now-canonical) image spacing only
    # when the caller has a reason to lie about it. Default: trust the
    # canonical grid — that's the point of prepare_volume.
    spacing = spacing_xyz
    if spacing is None:
        try:
            spacing = tuple(float(v) for v in img.GetSpacing())
        except Exception:
            spacing = (1.0, 1.0, 1.0)

    # Unified metal-evidence bolt extraction — same path as Auto Fit's
    # ``run_two_stage_detection``. Picks up bolts that LoG alone misses
    # (HU-saturated metal CCs).
    metal_evidence_vol = cpfit.compute_metal_evidence_volume(log1, ct_arr_kji)
    bolts, bolt_mask = cpfit.extract_bolt_candidates(
        log1, dist_arr, ijk_to_ras_mat, spacing,
        ct_arr=metal_evidence_vol,
        hu_threshold=cpfit.METAL_BOLT_THRESHOLD,
        hull_proximity_mm=cpfit.BOLT_HULL_PROXIMITY_MM,
    )

    return {
        # Canonical-grid img + matrices, updated by prepare_volume.
        # Callers pass these to fit_trajectory.
        "img": img,
        "ijk_to_ras_mat": ijk_to_ras_mat,
        "ras_to_ijk_mat": ras_to_ijk_mat,
        "log": log1,
        "frangi": frangi_s1,
        "ct_arr_kji": ct_arr_kji,
        "hull": hull_arr,
        "intracranial": intracranial,
        "head_distance": dist_arr,
        "blob_pts_ras": pts_ras,
        "blob_amps": amps,
        "bolts": bolts,
        "bolt_mask": bolt_mask,
    }


def _pca_axis(points, weights):
    """Amplitude-weighted PCA principal axis of an Nx3 RAS point cloud.
    Returns (centroid, axis_unit). Thin wrapper around the canonical
    ``rosa_core.contact_fit.fit_axis_pca`` so Auto Fit, Guided Fit, and
    callers in tools/tests share one implementation.
    """
    from rosa_core.contact_fit import fit_axis_pca

    return fit_axis_pca(points, weights=weights)


def match_seed_to_auto_traj(planned_start_ras, planned_end_ras, auto_trajs,
                              max_angle_deg=DEFAULT_MAX_ANGLE_DEG,
                              max_lateral_shift_mm=DEFAULT_MAX_LATERAL_SHIFT_MM):
    """Match a seed against an existing list of Auto Fit trajectories.

    For Phase 2 of Guided Fit ↔ Auto Fit unification: if Auto Fit has
    already detected a shank near the seed, Guided Fit should inherit
    that result (full walker validation + post-anchor scoring) rather
    than re-derive a parallel PCA fit. Selection is closest by
    ``angle + perpendicular-midpoint distance``, with auto-fit
    ``confidence`` as tie-break.

    ``auto_trajs`` is the list of trajectory dicts produced by
    ``rosa_scene.TrajectorySceneService.collect_working_trajectory_rows``
    or ``logic.collect_trajectories_by_source("auto_fit", ...)``.
    Both shapes carry the explicit-frame ``start_ras`` / ``end_ras``
    keys; the legacy ``start`` / ``end`` keys are LPS and are NOT
    accepted here — silently falling back to them previously produced
    a sign-flipped X/Y comparison against the planned-RAS seed.

    Returns a fit-shaped dict (same keys as ``fit_trajectory``'s
    success branch) or ``None`` when no auto trajectory satisfies the
    tolerances.
    """
    if not auto_trajs:
        return None
    planned_start = np.asarray(planned_start_ras, dtype=float).reshape(3)
    planned_end = np.asarray(planned_end_ras, dtype=float).reshape(3)
    seed_axis = _unit(planned_end - planned_start)
    seed_mid = 0.5 * (planned_start + planned_end)

    best = None  # (score=ang+mid_d, traj, ang, mid_d, conf)
    for tr in auto_trajs:
        ts_raw = tr.get("start_ras")
        te_raw = tr.get("end_ras")
        if ts_raw is None or te_raw is None:
            continue
        ts = np.asarray(ts_raw, dtype=float).reshape(3)
        te = np.asarray(te_raw, dtype=float).reshape(3)
        tr_axis = _unit(te - ts)
        cos_a = float(np.clip(abs(np.dot(seed_axis, tr_axis)), 0.0, 1.0))
        ang = float(np.degrees(np.arccos(cos_a)))
        if ang > float(max_angle_deg):
            continue
        tr_mid = 0.5 * (ts + te)
        d = seed_mid - tr_mid
        perp = d - float(np.dot(d, tr_axis)) * tr_axis
        mid_d = float(np.linalg.norm(perp))
        if mid_d > float(max_lateral_shift_mm):
            continue
        score = ang + mid_d
        try:
            conf = float(tr.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        if best is None:
            best = (score, tr, ang, mid_d, conf)
            continue
        # Closest wins; ties (within 0.5 mm+deg) broken by higher confidence.
        if score < best[0] - 0.5:
            best = (score, tr, ang, mid_d, conf)
        elif abs(score - best[0]) <= 0.5 and conf > best[4]:
            best = (score, tr, ang, mid_d, conf)

    if best is None:
        return None
    score, tr, ang, mid_d, conf = best
    ts = np.asarray(tr.get("start_ras"), dtype=float).reshape(3)
    te = np.asarray(tr.get("end_ras"), dtype=float).reshape(3)
    bolt_src = str(tr.get("bolt_source") or "")
    # Preserve the auto trajectory's start/end orientation. Auto Fit
    # established it via bidirectional bolt anchor (start = bolt-side,
    # end = deep tip); seed direction may disagree but the auto-derived
    # orientation is authoritative because it's grounded in the imaged
    # metal CC, not the planned axis.
    return {
        "success": True,
        "start_ras": [float(v) for v in ts],
        "end_ras": [float(v) for v in te],
        "axis_ras": [float(v) for v in _unit(te - ts)],
        "n_inliers": int(tr.get("n_inliers", 0) or 0),
        "n_wide_inliers": int(tr.get("n_inliers", 0) or 0),
        "tight_refit": True,
        "angle_deg": ang,
        "lateral_shift_mm": mid_d,
        "length_mm": float(np.linalg.norm(te - ts)),
        "intracranial_length_mm": float(
            tr.get("intracranial_length_mm") or np.linalg.norm(te - ts)
        ),
        "roi_radius_mm": 0.0,
        "bolt_anchored": (bolt_src == "metal"),
        "bolt_n_vox": int(tr.get("bolt_n_vox", 0) or 0),
        "bolt_source": bolt_src,
        "confidence": conf,
        "confidence_label": str(tr.get("confidence_label") or ""),
        "frangi_mean_mm": float(tr.get("frangi_mean_mm") or 0.0),
        "frangi_median_mm": float(tr.get("frangi_median_mm") or 0.0),
        "frac_strong_metal": float(tr.get("frac_strong_metal") or 0.0),
        "original_median_pitch_mm": float(tr.get("original_median_pitch_mm") or 0.0),
        "contact_span_mm": float(tr.get("contact_span_mm") or 0.0),
        "matched_auto_source": True,
        "matched_auto_name": str(tr.get("name") or ""),
    }


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
    # Structured warnings: any score-affecting fallback (failed deep-end
    # refinement, bolt anchor, dist sampling, Frangi, frac-strong-metal)
    # appends a one-line reason here. The caller surfaces these via
    # ``self.log`` so a silent fallback can never mask a regression.
    warnings: list[str] = []
    planned_axis = planned_vec / planned_length

    pts_ras = features.get("blob_pts_ras")
    amps = features.get("blob_amps")
    if pts_ras is None or amps is None:
        # Lazy extraction if compute_features was skipped
        pts_ras, amps = cpfit.extract_blob_cloud_ras(
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
    except Exception as exc:
        warnings.append(f"deep-end LoG refinement skipped: {exc}")

    # ---- Bolt anchor: produces bolt_tip_ras + skull_entry_ras -------
    # Try both orientations (shallow→deep and deep→shallow) and keep
    # whichever has more bolt-tube voxels — mirrors the auto-fit
    # ``_anchor_or_reject`` strategy. Necessary because seed sources
    # don't all use the same start=bolt convention; an upstream-flipped
    # seed would otherwise force the anchor to look for the bolt at
    # the deep-tip end and silently miss it.
    bolt_tip_ras = None
    skull_entry_ras = None
    bolt_n_vox = 0
    bolts = features.get("bolts") or []
    if bolts:
        try:
            fwd = cpfit.anchor_trajectory_to_bolt(shallow_ras, deep_ras, bolts)
            bwd = cpfit.anchor_trajectory_to_bolt(deep_ras, shallow_ras, bolts)
            fwd_n = int(fwd[2].get("tube_n_vox", 0)) if fwd[2] is not None else 0
            bwd_n = int(bwd[2].get("tube_n_vox", 0)) if bwd[2] is not None else 0
            if bwd_n > fwd_n:
                # The fit's PCA orientation was inverted relative to the
                # imaged shank's bolt→tip direction. Swap before storing
                # the result so start_ras / end_ras carry the right
                # bolt-side / deep-tip semantics.
                shallow_ras, deep_ras = deep_ras, shallow_ras
                fit_axis = -fit_axis
                new_start, skull_entry, bolt = bwd
            else:
                new_start, skull_entry, bolt = fwd
            if new_start is not None:
                bolt_tip_ras = np.asarray(new_start, dtype=float)
                if skull_entry is not None:
                    skull_entry_ras = np.asarray(skull_entry, dtype=float)
                bolt_n_vox = int(bolt.get("n_vox", 0))
        except Exception as exc:
            warnings.append(f"bolt anchor skipped: {exc}")

    # Prefer the bolt_tip as the outer endpoint when available — that
    # matches Auto Fit's convention and lets downstream code compute
    # intracranial length as |skull_entry → deep|.
    start_out = bolt_tip_ras if bolt_tip_ras is not None else shallow_ras

    fit_length = float(np.linalg.norm(deep_ras - start_out))

    # Score the guided-fit result with the same rubric Auto Fit uses
    # so downstream UI (confidence filter, mark/remove, Trajectory
    # Set table) treats the two sources interchangeably.
    bolt_source = "metal" if bolt_tip_ras is not None else "none"
    intracranial_endpoint = (
        skull_entry_ras if skull_entry_ras is not None else start_out
    )
    intra_length = float(np.linalg.norm(deep_ras - np.asarray(intracranial_endpoint, dtype=float)))
    # Pre-anchor span and amp_sum: derived from the tight-refit inlier set.
    proj_centered = (tight_pts - centroid) @ fit_axis
    contact_span_mm = float(proj_centered.max() - proj_centered.min())
    amp_sum = float(np.sum(tight_amps))
    # dist_min/max along the line in the head-distance map.
    dist_arr = features.get("head_distance")
    ras_to_ijk = np.asarray(ras_to_ijk_mat, dtype=float)
    if dist_arr is not None:
        try:
            shallow_d = cpfit._sample_dist_at_ras(dist_arr, ras_to_ijk, intracranial_endpoint)
            deep_d = cpfit._sample_dist_at_ras(dist_arr, ras_to_ijk, deep_ras)
            dist_min_mm = float(min(shallow_d, deep_d))
            dist_max_mm = float(max(shallow_d, deep_d))
            dist_mean_mm = float(0.5 * (shallow_d + deep_d))
        except Exception as exc:
            warnings.append(f"dist sampling failed, using NaN: {exc}")
            dist_min_mm = dist_max_mm = dist_mean_mm = float("nan")
    else:
        dist_min_mm = dist_max_mm = dist_mean_mm = float("nan")
    # Frangi tubularity along the post-anchor axis.
    frangi_arr = features.get("frangi")
    frangi_mean_mm = frangi_median_mm = 0.0
    if frangi_arr is not None:
        try:
            f_mean, f_med = cpfit._frangi_along_line_stats(
                start_out, deep_ras, frangi_arr, ras_to_ijk,
            )
            frangi_mean_mm = float(f_mean)
            frangi_median_mm = float(f_med)
        except Exception as exc:
            warnings.append(f"Frangi along-line skipped, using 0: {exc}")
    # Metal-continuity: fraction of axis samples saturating the unified
    # metal-evidence threshold.
    ct_arr_kji = features.get("ct_arr_kji")
    if ct_arr_kji is not None:
        try:
            frac_strong = cpfit._frac_strong_metal_along_line(
                start_out, deep_ras,
                features.get("log"), ct_arr_kji, ras_to_ijk,
            )
        except Exception as exc:
            warnings.append(f"frac_strong_metal skipped, using 0: {exc}")
            frac_strong = 0.0
    else:
        frac_strong = 0.0
    # Median NN spacing among tight inliers — proxy for contact pitch.
    pitch_mm = 0.0
    if tight_pts.shape[0] >= 2:
        sorted_along = np.sort(proj_centered)
        diffs = np.diff(sorted_along)
        if diffs.size > 0:
            pitch_mm = float(np.median(diffs))
    score_rec = {
        "amp_sum": amp_sum,
        "n_inliers": int(n_tight),
        "frangi_median_mm": frangi_median_mm,
        "frac_strong_metal": float(frac_strong),
        "original_median_pitch_mm": pitch_mm,
        "contact_span_mm": contact_span_mm,
        "length_mm": fit_length,
        "dist_min_mm": dist_min_mm,
        "dist_max_mm": dist_max_mm,
        "dist_mean_mm": dist_mean_mm,
        "bolt_source": bolt_source,
    }
    confidence, confidence_label, score_components = cpfit._compute_trajectory_score(score_rec)

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
        "intracranial_length_mm": intra_length,
        "roi_radius_mm": float(roi_radius_mm),
        "bolt_anchored": bolt_tip_ras is not None,
        "bolt_n_vox": bolt_n_vox,
        # Auto-Fit-equivalent score fields (consumed by Slicer UI's
        # confidence filter and Rosa.* attribute stampers).
        "bolt_source": bolt_source,
        "confidence": float(confidence),
        "confidence_label": str(confidence_label),
        "frangi_mean_mm": frangi_mean_mm,
        "frangi_median_mm": frangi_median_mm,
        "frac_strong_metal": float(frac_strong),
        "original_median_pitch_mm": pitch_mm,
        "contact_span_mm": contact_span_mm,
        "dist_min_mm": dist_min_mm,
        "dist_max_mm": dist_max_mm,
        "dist_mean_mm": dist_mean_mm,
    }
    if skull_entry_ras is not None:
        result["skull_entry_ras"] = [float(v) for v in skull_entry_ras]
    if bolt_tip_ras is not None:
        result["bolt_tip_ras"] = [float(v) for v in bolt_tip_ras]
    if warnings:
        result["warnings"] = list(warnings)
    return result
