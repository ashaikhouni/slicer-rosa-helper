"""Guided-fit engine — snap seeded trajectories to metal.

Given planned/seeded trajectories (entry → target, RAS), snap each to the
actual imaged shank. Used by the Slicer Guided Fit module and the CLI
``rosa-agent detect --seeds``.

The snap itself (``fit_trajectory``) runs the CANONICAL snap-flow —
``run_seeded_fit`` (``snap_via_signal_walk`` on-axis contact-peak walk +
``arbitrate_shared_peaks``) — the SAME engine the ``fit-rosa`` CLI and
``place_seeg`` use, so a seed snaps identically in Guided Fit, the CLI, and
contact placement (one snapper everywhere). Entry = shallowest detected
contact, tip = deepest; NO bolt walk (matching ``fit-rosa`` — a start beyond
the bolt overshoots, and the downstream model fit owns contact placement).
This replaced the old PCA-of-blobs-in-a-cylinder fit + bolt anchor, which
had diverged from the snap-flow.

``fit_seeds_against_auto`` is the per-seed entry point: it first tries to
match each seed to an existing Auto Fit emission (``match_seed_to_auto_traj``;
inherits the detector's walker-validated geometry) and otherwise snaps the
seed with ``fit_trajectory``. ``compute_features`` does the one-time
per-volume preprocessing (canonicalize + LoG/Frangi + blob cloud + bolts +
shared-selector intracranial mask) both paths share.

Results carry Auto-Fit-equivalent score fields (confidence / label /
bolt_source) via ``compute_trajectory_score`` so downstream UI treats Guided
Fit and Auto Fit trajectories interchangeably.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .candidate_seeds.blob_extraction import extract_blob_cloud_ras
from .candidate_seeds.confidence_score import compute_trajectory_score
from .candidate_seeds.frangi_sampling import frangi_along_line_stats
from .candidate_seeds.metal_evidence import (
    compute_metal_evidence_volume,
    frac_strong_metal_along_line,
)
from .primitives.bolt_anchor import (
    BOLT_HULL_PROXIMITY_MM,
    METAL_BOLT_THRESHOLD,
    extract_bolt_candidates,
)
from .primitives.geometry import sample_dist_at_ras
from .primitives.preprocessing import (
    FRANGI_STAGE1_SIGMA,
    LOG_SIGMA_MM,
    build_masks,
    frangi_single,
    log_sigma,
    prepare_volume,
)


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

# LoG-neg metal threshold for the canonical snap walker (run_seeded_fit) —
# notebook default; matches fit-rosa's LOG_NEG_THRESHOLD and seeded_fit's
# _build_starter cell-1, so the guided snap behaves identically to the CLI.
LOG_NEG_THRESHOLD = 300.0


def _unit(v):
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def compute_features(img, ijk_to_ras_mat, ras_to_ijk_mat=None, spacing_xyz=None,
                     *, mask_backend="auto", brain_mask=None, synthstrip_path=None,
                     compute_intracranial=True, log=None):
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

    Brain mask: the ``intracranial`` mask comes from the shared backend
    selector — ``mask_backend="auto"`` = SynthStrip-if-available →
    LoG-watershed; ``brain_mask`` (array/SITK image) overrides. It is the
    expensive part of this function. It is consumed ONLY by the matched-filter
    PICK (``place_seeg`` / ``fit-rosa`` proximal ``mf_anchor``), NOT by the snap
    or by guided-fit scoring (which use the cheap ``build_masks`` head-distance).
    So snap-only callers — Guided Fit and ``detect --seeds`` — pass
    ``compute_intracranial=False`` to skip the SynthStrip/watershed build
    entirely; ``intracranial`` is then ``None``. ``build_masks`` always provides
    the hull + head-distance arrays (skull-synth / bolt extent / scoring depth).
    """
    import SimpleITK as sitk
    from .services.mask_backend import compute_intracranial_mask
    img, ijk_to_ras_mat, ras_to_ijk_mat = prepare_volume(
        img, ijk_to_ras_mat, ras_to_ijk_mat,
    )
    hull_arr, _intracranial_legacy, dist_arr = build_masks(img)
    if compute_intracranial:
        intracranial, _mask_backend_used = compute_intracranial_mask(
            img, backend=mask_backend, brain_mask=brain_mask,
            synthstrip_path=synthstrip_path, log=log,
        )
    else:
        # Snap-only caller: the intracranial mask is never read (the snap is
        # mask-free; scoring uses the cheap build_masks head-distance), so we
        # skip the expensive SynthStrip/watershed build.
        intracranial = None
    log1 = log_sigma(img, sigma_mm=LOG_SIGMA_MM)
    frangi_s1 = frangi_single(img, sigma=FRANGI_STAGE1_SIGMA)
    ct_arr_kji = sitk.GetArrayFromImage(img).astype(np.float32)

    pts_ras, amps = extract_blob_cloud_ras(log1, ijk_to_ras_mat)

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
    metal_evidence_vol = compute_metal_evidence_volume(log1, ct_arr_kji)
    bolts, bolt_mask = extract_bolt_candidates(
        log1, dist_arr, ijk_to_ras_mat, spacing,
        ct_arr=metal_evidence_vol,
        hu_threshold=METAL_BOLT_THRESHOLD,
        hull_proximity_mm=BOLT_HULL_PROXIMITY_MM,
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
    than re-snap it independently. Selection is closest by
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
    """Snap one seeded trajectory to the imaged shank via the CANONICAL
    snap-flow (``run_seeded_fit`` -> ``snap_via_signal_walk``).

    This is the SAME engine the ``fit-rosa`` CLI and ``place_seeg`` use, so
    a seed snaps identically in Guided Fit, the CLI, and contact placement
    (one snapper everywhere). The on-axis contact-peak walk replaces the
    old PCA-of-blobs-in-a-cylinder fit.

    Returns a dict with ``success`` and — on success — ``start_ras`` (the
    shallowest detected contact), ``end_ras`` (the deepest detected
    contact), ``axis_ras``, plus the Auto-Fit-equivalent score fields.

    ``run_seeded_fit`` deliberately stops at the shallowest contact rather
    than walking out to the bolt edge (a start *beyond* the bolt overshoots;
    near-bolt contacts are unresolvable and the downstream model fit owns
    contact placement). So Guided Fit no longer emits ``skull_entry_ras`` /
    ``bolt_tip_ras`` and ``bolt_anchored`` is always False — matching
    ``fit-rosa``. ``roi_radius_mm`` is accepted for API stability but unused
    (the snap walks the on-axis profile, not a cylinder of blobs).
    """
    planned_start = np.asarray(planned_start_ras, dtype=float).reshape(3)
    planned_end = np.asarray(planned_end_ras, dtype=float).reshape(3)
    planned_vec = planned_end - planned_start
    planned_length = float(np.linalg.norm(planned_vec))
    if planned_length < 1e-3:
        return {"success": False, "reason": "planned trajectory has zero length"}
    # Structured warnings: any score-affecting fallback (amp / dist
    # sampling, Frangi, frac-strong-metal) appends a one-line reason here.
    # The caller surfaces these via ``self.log`` so a silent fallback can
    # never mask a regression.
    warnings: list[str] = []
    planned_axis = planned_vec / planned_length

    log_arr = features.get("log")
    if log_arr is None:
        return {"success": False, "reason": "features missing 'log' volume"}
    ras_to_ijk = np.asarray(ras_to_ijk_mat, dtype=float)
    log_neg = np.clip(-np.asarray(log_arr), 0.0, None).astype("float32", copy=False)

    # ---- Canonical snap (run_seeded_fit): on-axis contact-peak walk -----
    # Single source of truth for "snap a seed to metal" — the SAME engine
    # the fit-rosa CLI and place_seeg use. Entry = shallowest detected
    # contact, tip = deepest. No bolt walk. Lazy import avoids a load-time
    # rosa_core <-> rosa_detect cycle.
    from rosa_core.seeded_fit import run_seeded_fit
    chains = run_seeded_fit(
        [{"name": "seed", "start": planned_start, "end": planned_end}],
        signal_vol=log_neg, threshold=LOG_NEG_THRESHOLD,
        ras_to_ijk=ras_to_ijk, bolt_signal_vol=features.get("metal_evidence"),
    )
    chain = chains[0] if chains else None
    if chain is None:
        return {
            "success": False,
            "reason": "snap found no contact chain (off-metal seed / too few peaks)",
            "n_inliers": 0,
        }

    fit_axis = _unit(chain["axis"])
    if float(np.dot(fit_axis, planned_axis)) < 0:
        fit_axis = -fit_axis
    shallow_ras = np.asarray(chain["entry_ras"], dtype=float)
    deep_ras = np.asarray(chain["tip_ras"], dtype=float)
    tight_pts = np.asarray(chain.get("kept_pts"), dtype=float).reshape(-1, 3)
    n_tight = int(tight_pts.shape[0])
    n_wide = n_tight
    tight_pass = True
    if n_tight < int(min_inliers):
        return {
            "success": False,
            "reason": f"snap chain too short ({n_tight} < {int(min_inliers)} contacts)",
            "n_inliers": n_tight,
        }
    centroid = tight_pts.mean(axis=0)

    # ---- Sanity gates vs the seed (flag a snap that drifted off-shank) --
    cos = float(np.clip(abs(np.dot(fit_axis, planned_axis)), 0.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cos)))
    if angle_deg > float(max_angle_deg):
        return {
            "success": False,
            "reason": f"axis tilt {angle_deg:.1f} deg > {float(max_angle_deg):.1f} deg",
            "n_inliers": n_tight,
            "angle_deg": angle_deg,
        }

    planned_mid = 0.5 * (planned_start + planned_end)
    mid_offset = centroid - planned_mid
    along_mid = float(np.dot(mid_offset, fit_axis))
    lat_offset = mid_offset - along_mid * fit_axis
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

    # ---- Amplitudes at the snapped contacts (for the score's amp term):
    # sample the LoG-neg metal signal directly at each kept contact.
    from rosa_core.volume_sampling import sample_trilinear_batch
    try:
        tight_amps = np.nan_to_num(
            np.asarray(sample_trilinear_batch(log_neg, ras_to_ijk, tight_pts),
                       dtype=float),
            nan=0.0,
        )
    except Exception as exc:
        warnings.append(f"amp sampling failed, using zeros: {exc}")
        tight_amps = np.zeros(n_tight, dtype=float)

    # Entry = shallowest contact; NO bolt anchor. run_seeded_fit stops at
    # the shallowest contact and never walks out to the bolt edge, so
    # Guided Fit no longer emits skull_entry_ras / bolt_tip_ras and
    # bolt_anchored is always False — identical to fit-rosa / place_seeg.
    skull_entry_ras = None
    bolt_tip_ras = None
    bolt_n_vox = 0
    start_out = shallow_ras
    fit_length = float(np.linalg.norm(deep_ras - start_out))

    # Score the guided-fit result with the same rubric Auto Fit uses
    # so downstream UI (confidence filter, mark/remove, Trajectory
    # Set table) treats the two sources interchangeably. A successful
    # snap chain sits on metal -> bolt_source "metal".
    bolt_source = "metal"
    intracranial_endpoint = start_out
    intra_length = float(np.linalg.norm(deep_ras - np.asarray(intracranial_endpoint, dtype=float)))
    # Contact span and amp_sum: derived from the snapped contact set.
    proj_centered = (tight_pts - centroid) @ fit_axis
    contact_span_mm = float(proj_centered.max() - proj_centered.min())
    amp_sum = float(np.sum(tight_amps))
    # dist_min/max along the line in the head-distance map.
    dist_arr = features.get("head_distance")
    ras_to_ijk = np.asarray(ras_to_ijk_mat, dtype=float)
    if dist_arr is not None:
        try:
            shallow_d = sample_dist_at_ras(dist_arr, ras_to_ijk, intracranial_endpoint)
            deep_d = sample_dist_at_ras(dist_arr, ras_to_ijk, deep_ras)
            dist_min_mm = float(min(shallow_d, deep_d))
            dist_max_mm = float(max(shallow_d, deep_d))
            dist_mean_mm = float(0.5 * (shallow_d + deep_d))
        except Exception as exc:
            warnings.append(f"dist sampling failed, using NaN: {exc}")
            dist_min_mm = dist_max_mm = dist_mean_mm = float("nan")
    else:
        dist_min_mm = dist_max_mm = dist_mean_mm = float("nan")
    # Frangi tubularity along the snapped axis.
    frangi_arr = features.get("frangi")
    frangi_mean_mm = frangi_median_mm = 0.0
    if frangi_arr is not None:
        try:
            f_mean, f_med = frangi_along_line_stats(
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
            frac_strong = frac_strong_metal_along_line(
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
    confidence, confidence_label, score_components = compute_trajectory_score(score_rec)

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


# ---------------------------------------------------------------------
# Unified seed-fitting entry point — used by BOTH:
#   * Slicer Guided Fit module (PostopCTLocalization/guided_fit.py)
#   * CLI `rosa-agent detect --seeds` (cli/rosa_agent/commands/detect.py)
#
# Single source of truth so both surfaces have identical fit behavior.
# ---------------------------------------------------------------------


def fit_seeds_against_auto(
    seeds,
    features,
    ijk_to_ras_mat,
    ras_to_ijk_mat,
    *,
    auto_trajs=None,
    auto_run_if_missing=True,
    roi_radius_mm=DEFAULT_ROI_RADIUS_MM,
    max_angle_deg=DEFAULT_MAX_ANGLE_DEG,
    max_lateral_shift_mm=DEFAULT_MAX_LATERAL_SHIFT_MM,
    min_inliers=DEFAULT_MIN_INLIERS,
    progress_log=None,
):
    """Per-seed guided fit, match-against-auto then canonical-snap fallback.

    For each seed:
      1. If an auto trajectory is in tolerance (`max_angle_deg` /
         `max_lateral_shift_mm`), inherit its fit verbatim. Each auto
         trajectory can be claimed by at most one seed.
      2. Otherwise, snap the seed with ``fit_trajectory`` (the canonical
         snap-flow — ``run_seeded_fit`` / ``snap_via_signal_walk``).

    The auto trajectory pool comes from:
      * ``auto_trajs`` if explicitly supplied (Slicer passes its scene
        cache; pass ``[]`` to disable match-against-auto entirely).
      * ``run_contact_pitch_v1_with_features`` run internally on the
        canonical CT in ``features["img"]`` when ``auto_trajs is None``
        and ``auto_run_if_missing`` is True. This is what makes a fresh
        Slicer "Fit All" or CLI ``rosa-agent detect --seeds`` invocation
        produce auto-aware results without the caller having to remember
        to run Auto Fit first.

    Args:
        seeds: list of dicts with keys ``name``, ``start_ras``, ``end_ras``.
        features: dict from ``compute_features``. Must include ``img``,
            ``ijk_to_ras_mat``, ``ras_to_ijk_mat`` if ``auto_run_if_missing``.
        ijk_to_ras_mat / ras_to_ijk_mat: canonical-grid matrices (must
            match those returned by ``compute_features``).
        auto_trajs: pre-computed auto-fit trajectories (list of dicts with
            ``start_ras`` / ``end_ras``), or ``None`` to auto-compute.
        auto_run_if_missing: when ``auto_trajs is None``, whether to run
            auto fit. Set False to disable match-against-auto entirely.
        roi_radius_mm / max_angle_deg / max_lateral_shift_mm /
            min_inliers: forwarded to ``fit_trajectory`` and used as
            tolerances for ``match_seed_to_auto_traj``.
        progress_log: callable taking one string; receives status lines
            (auto-fit start/end, per-seed match/fail). Defaults to no-op.

    Returns:
        list of fit-result dicts in the same order as ``seeds``. Each
        fit dict has the keys returned by ``match_seed_to_auto_traj``
        or ``fit_trajectory`` (``success``, ``start_ras`` / ``end_ras``
        on success, ``reason`` on failure), plus:
          * ``name``: the seed's name
          * ``matched_path``: ``'auto'`` / ``'snap'`` / ``'failed'``
    """
    log = progress_log if callable(progress_log) else (lambda _msg: None)

    # Resolve the auto trajectory pool.
    if auto_trajs is None:
        if auto_run_if_missing:
            from .service import run_contact_pitch_v1_with_features
            log("[guided] no auto trajectories supplied; running auto fit")
            try:
                ctx = {
                    "img": features.get("img"),
                    "ijk_to_ras_4x4": np.asarray(ijk_to_ras_mat, dtype=float),
                    "ras_to_ijk_4x4": np.asarray(ras_to_ijk_mat, dtype=float),
                }
                auto_result, _features = run_contact_pitch_v1_with_features(ctx)
                auto_trajs = list(auto_result.get("trajectories") or [])
                log(f"[guided] auto fit produced {len(auto_trajs)} trajectories")
            except Exception as exc:
                log(f"[guided] auto fit failed ({exc}); falling back to snap-only")
                auto_trajs = []
        else:
            auto_trajs = []
    else:
        auto_trajs = list(auto_trajs)

    remaining = list(auto_trajs)
    out: list[dict] = []
    for seed in seeds:
        name = str(seed.get("name") or f"seed_{len(out) + 1}")
        ss = seed["start_ras"]
        se = seed["end_ras"]
        fit = None

        if remaining:
            try:
                fit = match_seed_to_auto_traj(
                    planned_start_ras=ss, planned_end_ras=se,
                    auto_trajs=remaining,
                    max_angle_deg=max_angle_deg,
                    max_lateral_shift_mm=max_lateral_shift_mm,
                )
            except Exception as exc:
                log(f"[guided] {name}: match-auto crashed ({exc})")
                fit = None
            if fit is not None:
                # Remove the claimed auto traj from the pool so other
                # seeds don't double-claim it.
                claimed_start = list(fit.get("start_ras") or [])
                claimed_end = list(fit.get("end_ras") or [])
                consumed = False
                next_remaining = []
                for t in remaining:
                    if (not consumed
                            and list(t.get("start_ras") or []) == claimed_start
                            and list(t.get("end_ras") or []) == claimed_end):
                        consumed = True
                        continue
                    next_remaining.append(t)
                remaining = next_remaining
                fit = dict(fit)
                fit["matched_path"] = "auto"

        if fit is None:
            try:
                fit = fit_trajectory(
                    planned_start_ras=ss, planned_end_ras=se,
                    features=features,
                    ijk_to_ras_mat=ijk_to_ras_mat,
                    ras_to_ijk_mat=ras_to_ijk_mat,
                    roi_radius_mm=roi_radius_mm,
                    max_angle_deg=max_angle_deg,
                    max_lateral_shift_mm=max_lateral_shift_mm,
                    min_inliers=min_inliers,
                )
                fit = dict(fit)
                fit["matched_path"] = "snap" if fit.get("success") else "failed"
            except Exception as exc:
                log(f"[guided] {name}: fit_trajectory crashed ({exc})")
                fit = {"success": False, "reason": f"crash: {exc}", "matched_path": "failed"}

        fit["name"] = name
        out.append(fit)

    return out
