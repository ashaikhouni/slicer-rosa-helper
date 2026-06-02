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

``fit_seeds_against_auto`` is the batch entry point: it snaps all the user's
seeds in one ``run_seeded_fit`` call (so ``arbitrate_shared_peaks`` resolves
cross-shank conflicts between them) and scores each; ``fit_trajectory`` is the
single-seed form. (An earlier match-against-auto step was removed — with the
canonical snapper, snapping the provided seed directly beats running a full
auto-detection just to match it.) ``compute_features`` does the one-time
per-volume preprocessing (canonicalize + LoG/Frangi + blob cloud + bolts +
optional intracranial mask) they share.

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


def _fit_dict_from_chain(chain, planned_start_ras, planned_end_ras, *,
                         features, ras_to_ijk, log_neg,
                         roi_radius_mm=DEFAULT_ROI_RADIUS_MM,
                         max_angle_deg=DEFAULT_MAX_ANGLE_DEG,
                         max_lateral_shift_mm=DEFAULT_MAX_LATERAL_SHIFT_MM,
                         min_inliers=DEFAULT_MIN_INLIERS):
    """Score one snapped chain into a guided-fit result dict.

    Shared by ``fit_trajectory`` (single seed) and ``fit_seeds_against_auto``
    (batch) so a chain is scored identically however it was snapped. ``chain``
    is one ``run_seeded_fit`` result (``{axis, entry_ras, tip_ras, kept_pts}``)
    or ``None`` when the snap failed. ``planned_start_ras`` / ``planned_end_ras``
    are the seed endpoints (RAS) — used to orient the axis and to gate the snap
    against the seed (axis tilt + midpoint lateral shift). ``log_neg`` is the
    metal signal sampled for the contact-amplitude score term.

    Returns ``{"success": True, "start_ras" (shallowest contact), "end_ras"
    (deep tip), "axis_ras", + Auto-Fit-equivalent score fields}`` or
    ``{"success": False, "reason": ...}``. ``bolt_anchored`` is always False and
    ``skull_entry_ras`` / ``bolt_tip_ras`` are never emitted (snap-flow
    convention: entry is the shallowest contact, no bolt walk).
    """
    planned_start = np.asarray(planned_start_ras, dtype=float).reshape(3)
    planned_end = np.asarray(planned_end_ras, dtype=float).reshape(3)
    planned_vec = planned_end - planned_start
    planned_length = float(np.linalg.norm(planned_vec))
    if planned_length < 1e-3:
        return {"success": False, "reason": "planned trajectory has zero length"}
    planned_axis = planned_vec / planned_length
    if chain is None:
        return {
            "success": False,
            "reason": "snap found no contact chain (off-metal seed / too few peaks)",
            "n_inliers": 0,
        }
    ras_to_ijk = np.asarray(ras_to_ijk, dtype=float)
    # Structured warnings: any score-affecting fallback (amp / dist sampling,
    # Frangi, frac-strong-metal) appends a one-line reason here so a silent
    # fallback can never mask a regression.
    warnings: list[str] = []

    fit_axis = _unit(chain["axis"])
    if float(np.dot(fit_axis, planned_axis)) < 0:
        fit_axis = -fit_axis
    shallow_ras = np.asarray(chain["entry_ras"], dtype=float)
    deep_ras = np.asarray(chain["tip_ras"], dtype=float)
    tight_pts = np.asarray(chain.get("kept_pts"), dtype=float).reshape(-1, 3)
    n_tight = int(tight_pts.shape[0])
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

    # Entry = shallowest contact; NO bolt anchor (snap-flow convention).
    start_out = shallow_ras
    fit_length = float(np.linalg.norm(deep_ras - start_out))
    bolt_source = "metal"   # a successful snap chain sits on metal

    # ---- Score with the same rubric Auto Fit uses (compute_trajectory_score).
    proj_centered = (tight_pts - centroid) @ fit_axis
    contact_span_mm = float(proj_centered.max() - proj_centered.min())
    amp_sum = float(np.sum(tight_amps))
    dist_arr = features.get("head_distance")
    if dist_arr is not None:
        try:
            shallow_d = sample_dist_at_ras(dist_arr, ras_to_ijk, start_out)
            deep_d = sample_dist_at_ras(dist_arr, ras_to_ijk, deep_ras)
            dist_min_mm = float(min(shallow_d, deep_d))
            dist_max_mm = float(max(shallow_d, deep_d))
            dist_mean_mm = float(0.5 * (shallow_d + deep_d))
        except Exception as exc:
            warnings.append(f"dist sampling failed, using NaN: {exc}")
            dist_min_mm = dist_max_mm = dist_mean_mm = float("nan")
    else:
        dist_min_mm = dist_max_mm = dist_mean_mm = float("nan")
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
    ct_arr_kji = features.get("ct_arr_kji")
    if ct_arr_kji is not None:
        try:
            frac_strong = frac_strong_metal_along_line(
                start_out, deep_ras, features.get("log"), ct_arr_kji, ras_to_ijk,
            )
        except Exception as exc:
            warnings.append(f"frac_strong_metal skipped, using 0: {exc}")
            frac_strong = 0.0
    else:
        frac_strong = 0.0
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
    confidence, confidence_label, _score_components = compute_trajectory_score(score_rec)

    result = {
        "success": True,
        "start_ras": [float(v) for v in start_out],
        "end_ras": [float(v) for v in deep_ras],
        "axis_ras": [float(v) for v in fit_axis],
        "n_inliers": n_tight,
        "n_wide_inliers": n_tight,
        "tight_refit": True,
        "angle_deg": angle_deg,
        "lateral_shift_mm": lateral_shift_mm,
        "length_mm": fit_length,
        "intracranial_length_mm": fit_length,
        "roi_radius_mm": float(roi_radius_mm),
        "bolt_anchored": False,
        "bolt_n_vox": 0,
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
    if warnings:
        result["warnings"] = list(warnings)
    return result


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
    log_arr = features.get("log")
    if log_arr is None:
        return {"success": False, "reason": "features missing 'log' volume"}
    ras_to_ijk = np.asarray(ras_to_ijk_mat, dtype=float)
    log_neg = np.clip(-np.asarray(log_arr), 0.0, None).astype("float32", copy=False)
    # Canonical snap (run_seeded_fit): on-axis contact-peak walk — the SAME
    # engine the fit-rosa CLI and place_seeg use. Lazy import avoids a
    # load-time rosa_core <-> rosa_detect cycle. For a BATCH of seeds prefer
    # fit_seeds_against_auto: it snaps them in one run_seeded_fit call so
    # arbitrate_shared_peaks resolves cross-shank conflicts between them.
    from rosa_core.seeded_fit import run_seeded_fit
    chains = run_seeded_fit(
        [{"name": "seed",
          "start": np.asarray(planned_start_ras, dtype=float),
          "end": np.asarray(planned_end_ras, dtype=float)}],
        signal_vol=log_neg, threshold=LOG_NEG_THRESHOLD,
        ras_to_ijk=ras_to_ijk, bolt_signal_vol=features.get("metal_evidence"),
    )
    return _fit_dict_from_chain(
        chains[0] if chains else None, planned_start_ras, planned_end_ras,
        features=features, ras_to_ijk=ras_to_ijk, log_neg=log_neg,
        roi_radius_mm=roi_radius_mm, max_angle_deg=max_angle_deg,
        max_lateral_shift_mm=max_lateral_shift_mm, min_inliers=min_inliers,
    )


# ---------------------------------------------------------------------
# Batch seed-fitting entry point — used by BOTH:
#   * Slicer Guided Fit module (PostopCTLocalization/guided_fit.py)
#   * CLI `rosa-agent detect --seeds` (cli/rosa_agent/commands/detect.py)
#
# Single source of truth so both surfaces snap identically.
# ---------------------------------------------------------------------


def fit_seeds_against_auto(
    seeds,
    features,
    ijk_to_ras_mat,
    ras_to_ijk_mat,
    *,
    max_angle_deg=DEFAULT_MAX_ANGLE_DEG,
    max_lateral_shift_mm=DEFAULT_MAX_LATERAL_SHIFT_MM,
    min_inliers=DEFAULT_MIN_INLIERS,
    progress_log=None,
):
    """Batch-snap a list of seeds to metal via the canonical snap-flow.

    All seeds are snapped in ONE ``run_seeded_fit`` call, so
    ``arbitrate_shared_peaks`` resolves cross-shank contention between them
    (close pairs can't both claim the same contacts); each chain is then scored
    with :func:`_fit_dict_from_chain`.

    The historical match-against-auto step (run a full auto-detection, match
    each seed to the nearest emission, inherit its geometry) was REMOVED. It
    existed only to dodge the old weak PCA fit. Now that the snapper IS the
    canonical ``run_seeded_fit``, snapping the provided seed directly is just as
    accurate, more seed-faithful (no greedy mis-match to the wrong emission),
    and far cheaper (no auto-detection run purely to match). The function name
    is retained so the Slicer + CLI callers don't change.

    Args:
        seeds: list of dicts with keys ``name``, ``start_ras``, ``end_ras``.
        features: dict from ``compute_features`` (needs ``log``; ``head_distance``
            / ``frangi`` / ``ct_arr_kji`` used for scoring when present).
        ijk_to_ras_mat / ras_to_ijk_mat: canonical-grid matrices (must match
            those returned by ``compute_features``).
        max_angle_deg / max_lateral_shift_mm / min_inliers: per-seed snap-vs-seed
            sanity gates — a snap that drifts past these is reported as a miss.
        progress_log: callable taking one string; receives status lines.

    Returns:
        list of fit-result dicts in the same order as ``seeds``. Each has the
        keys :func:`_fit_dict_from_chain` returns, plus ``name`` and
        ``matched_path`` (``'snap'`` on success / ``'failed'`` otherwise).
    """
    log = progress_log if callable(progress_log) else (lambda _msg: None)

    specs = [
        {
            "name": str(seed.get("name") or f"seed_{i + 1}"),
            "start": np.asarray(seed["start_ras"], dtype=float),
            "end": np.asarray(seed["end_ras"], dtype=float),
        }
        for i, seed in enumerate(seeds)
    ]

    log_arr = features.get("log")
    if log_arr is None:
        return [
            {"success": False, "reason": "features missing 'log' volume",
             "name": s["name"], "matched_path": "failed"}
            for s in specs
        ]
    ras_to_ijk = np.asarray(ras_to_ijk_mat, dtype=float)
    log_neg = np.clip(-np.asarray(log_arr), 0.0, None).astype("float32", copy=False)

    # ONE batch snap across all seeds -> arbitrate_shared_peaks resolves
    # cross-shank conflicts between them (lazy import avoids a load-time cycle).
    from rosa_core.seeded_fit import run_seeded_fit
    chains = run_seeded_fit(
        [{"name": s["name"], "start": s["start"], "end": s["end"]} for s in specs],
        signal_vol=log_neg, threshold=LOG_NEG_THRESHOLD,
        ras_to_ijk=ras_to_ijk, bolt_signal_vol=features.get("metal_evidence"),
        log=log,
    )

    out: list[dict] = []
    n_ok = 0
    for spec, chain in zip(specs, chains):
        try:
            fit = _fit_dict_from_chain(
                chain, spec["start"], spec["end"],
                features=features, ras_to_ijk=ras_to_ijk, log_neg=log_neg,
                max_angle_deg=max_angle_deg,
                max_lateral_shift_mm=max_lateral_shift_mm, min_inliers=min_inliers,
            )
        except Exception as exc:
            log(f"[guided] {spec['name']}: scoring crashed ({exc})")
            fit = {"success": False, "reason": f"crash: {exc}"}
        fit = dict(fit)
        fit["name"] = spec["name"]
        ok = bool(fit.get("success"))
        fit["matched_path"] = "snap" if ok else "failed"
        n_ok += int(ok)
        if not ok:
            log(f"[guided] {spec['name']}: {fit.get('reason', 'snap failed')}")
        out.append(fit)
    log(f"[guided] batch-snapped {n_ok}/{len(specs)} seed(s)")
    return out
