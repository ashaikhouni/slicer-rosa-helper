"""Contact-pitch v1 detector: LoG-blob + library-pitch SEEG shank detection.

Pipeline:
  * preprocessing: hull mask, intracranial mask, hull signed-distance,
    LoG sigma=1, Frangi sigma=1.
  * blob-pitch walker: LoG regional-minima blobs + library-pitch
    Hough-style walk, amp_sum gate, deep-tip prior, Frangi-along-axis
    gate.
  * bolt anchoring (LoG primary, HU rescue, axis-to-skull synth, or no-
    anchor recall-first emission with confidence-score downweighting).
  * post-anchor dedup, deep-end refinement, crossing-tip retreat.
  * physical-evidence confidence score per emission.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

# Electrode-model picker — single source of truth for Auto Fit, Guided Fit,
# Manual Fit, and Contacts & Trajectory View. Re-exported here for
# backwards-compat with callers that still import from this module
# (`tests/deep_core/test_walker_signature_classifier.py` and others).
from rosa_core.electrode_classifier import (  # noqa: F401
    PITCH_STRATEGY_PITCHES_MM,
    PITCH_STRATEGY_VENDORS,
    VENDOR_ID_PREFIXES,
    classify_by_count_and_span,
    classify_by_walker_signature,
    classify_electrode_model,
    classify_pacer_template,
    filter_models_for_strategy,
    suggest_shortest_covering_model,
)
from rosa_core.volume_sampling import (
    clip_to_voxel,
    iter_axis_points,
    ras_to_ijk_pt,
    sample_nearest_at_ras,
)


# ---- Config (match probe_two_stage.py + probe_blob_pitch.py) ----------

# Preprocessing + bolt-anchor primitives live under the strategy-
# agnostic rosa_detect.primitives package. Re-exported below for
# backwards compat with the many test/probe files that read symbols
# via ``cpfit.<symbol>``.
from .primitives.preprocessing import (  # noqa: F401
    CANONICAL_SPACING_MM,
    FRANGI_STAGE1_SIGMA,
    HU_CLIP_MAX,
    INTRACRANIAL_MIN_DISTANCE_MM,
    LOG_SIGMA_MM,
    build_masks,
    frangi_single,
    log_sigma,
    prepare_volume,
)
from .primitives.bolt_anchor import (  # noqa: F401
    BOLT_BASE_MAX_DIST_MM,
    BOLT_HULL_PROXIMITY_MM,
    BOLT_MAX_INWARD_ALONG_MM,
    BOLT_MIN_TUBE_VOXELS,
    BOLT_MIN_VOXELS,
    BOLT_SEARCH_OUTWARD_MM,
    BOLT_SHALLOW_HULL_PROX_MM,
    BOLT_TUBE_RADIUS_MM,
    HU_BOLT_NORMALIZER,
    LOG_BOLT_NORMALIZER,
    METAL_BOLT_THRESHOLD,
    anchor_trajectory_to_bolt,
    extract_bolt_candidates,
)
# Generic geometry helpers — shared across v1 (candidate seeds) and the
# placer (rosa_core.contact_placement). cpfit re-exports under the
# legacy ``_*`` names for back-compat with probes and tests.
from .primitives.geometry import (  # noqa: F401
    unit as _unit,
    sample_dist_at_ras as _sample_dist_at_ras,
    orient_shallow_to_deep as _orient_shallow_to_deep,
    min_perp_to_other_segments as _min_perp_to_other_segments,
    kji_to_ras_fn_from_matrix as _kji_to_ras_fn_from_matrix,
)
# All tunable knobs for v1 detection live in
# ``rosa_detect.candidate_seeds.constants``. Re-exported here so legacy
# callers (probes, tests) that read symbols via ``cpfit.<NAME>`` still
# work. Calibration rationale moved to constants.py docstrings.
from .candidate_seeds.constants import (  # noqa: F401
    ANCHOR_TOTAL_OVERSHOOT_MM,
    AX_TOL_MM,
    AXIS_REFINE_MAX_MM,
    AXIS_REFINE_MIN_ABS,
    AXIS_REFINE_MISS_MM,
    AXIS_REFINE_STEP_MM,
    AXIS_SKULL_SYNTH_BOLT_PROTRUDE_MM,
    AXIS_SKULL_SYNTH_MAX_OUTWARD_MM,
    AXIS_SKULL_SYNTH_STEP_MM,
    BOLT_PROTRUSION_MIN_MM,
    CROSSING_RETREAT_STEP_MM,
    CROSSING_TIP_CLEARANCE_MM,
    DEEP_END_MARGIN_PAST_LAST_CONTACT_MM,
    DEEP_TIP_MIN_MM,
    DEEP_TIP_MIN_SHORT_MM,
    DEEP_TIP_SHORT_MAX_AVG_PITCH_MM,
    FRANGI_LINE_MIN_MEDIAN,
    LOG_BLOB_MAX_VOXELS,
    LOG_BLOB_THRESHOLD,
    MAX_K_STEPS,
    MIN_BLOBS_PER_LINE,
    MIN_BLOBS_POST_ARBITRATION,
    PERP_TOL_MM,
    PITCH_AUTO_MAX_MM,
    PITCH_AUTO_MAX_PEAKS,
    PITCH_AUTO_MIN_MM,
    PITCH_AUTO_PEAK_EXCLUSION_MM,
    PITCH_AUTO_SECONDARY_FRAC,
    PITCH_MM,
    PITCH_SNAP_MM,
    PITCH_TOL_MM,
    POST_ANCHOR_DEDUP_ANG_DEG,
    POST_ANCHOR_DEDUP_PERP_MM,
    SCORE_AMP_SAT,
    SCORE_BOLT_VALUES,
    SCORE_DEPTH_SAT_MM,
    SCORE_HIGH_THRESHOLD,
    SCORE_INTRACRANIAL_SAT_MM,
    SCORE_LENGTH_SHOULDER_MM,
    SCORE_MEDIUM_THRESHOLD,
    SCORE_METAL_CONTINUITY_SAT,
    SCORE_N_INLIERS_OVER_SLACK,
    SCORE_N_INLIERS_SLOPE,
    SCORE_PITCH_TOL_MM,
    SCORE_SPAN_SHOULDER_MM,
    SCORE_WEIGHTS,
    SEEG_VENDORS,
    STAGE1_DEDUP_ANGLE_DEG,
    STAGE1_DEDUP_OVERLAP_FRAC,
    STAGE1_DEDUP_PERP_MM,
    WALKER_GAP_SLACK_MM,
    WALKER_SPAN_OVER_SLACK_MM,
    WALKER_SPAN_UNDER_SLACK_MM,
    WIRE_CLASS_MIN_DEPTH_MM,
    WIRE_CLASS_MIN_ELONGATION,
    WIRE_CLASS_MIN_SPAN_MM,
    WIRE_CLASS_MIN_VOXELS,
    _BUNDLED_LIBRARY_BOUNDS_FALLBACK,
)

# ---- Library-derived bounds ------------------------------------------
#
# A real electrode's contact count, span, and within-electrode pitch
# are answered by the bundled electrode-model library
# (CommonLib/resources/electrodes/electrode_models.json) — extracting
# the bounds from that library at module load is the principled way
# to ask "is this trajectory shape consistent with a real electrode?"
# instead of carrying hardcoded snapshots that go stale every time the
# library changes (the file's history shows AMC099 L_5, subject-137
# L_3, T21 L_8/L_9/L_13 each prompting a one-off bump).
#
# Slack constants below stay separate, named, and small. Each one
# answers one physical question (sub-voxel walker drift, bolt
# voxel pull-on, family variants the library doesn't yet enumerate)
# rather than "make subject X pass."

_BUNDLED_LIBRARY_BOUNDS_FALLBACK = {
    # Snapshot of the in-tree library used when rosa_core is
    # unavailable (stripped install, Slicer-less environments).
    "min_contacts": 5,
    "max_contacts": 18,
    "min_contact_span_mm": 14.0,
    "max_contact_span_mm": 78.5,
    "max_within_electrode_pitch_mm": 13.0,
    "regular_pitches_mm": (3.5, 3.9, 3.97, 4.43, 4.8, 6.1),
}


SEEG_VENDORS = ("Dixi", "PMT", "AdTech")


# Library bounds + strategy-scoped walker constants live in
# candidate_seeds.pitch_library. Re-exported here for back-compat.
from .candidate_seeds.pitch_library import (  # noqa: F401
    LIBRARY_BOUNDS as _LIBRARY_BOUNDS,
    compute_library_bounds as _compute_library_bounds,
    library_bounds_for_strategy,
    model_vendor as _model_vendor,
    strategy_global_overrides as _strategy_global_overrides,
    StrategyBoundsScope as _StrategyBoundsScope,
    with_strategy_bounds as _with_strategy_bounds,
)

# Walker endpoint + chaining slacks. Each one answers "what physical
# effect loosens this bound past the strict library value?" — never
# "which subject passes after this number?".
WALKER_SPAN_UNDER_SLACK_MM = 2.0   # walker can miss endpoint contacts
                                    # under sub-voxel drift / partial
                                    # volume bias (~0.5 mm × 2 endpoints
                                    # × small pitch).
WALKER_SPAN_OVER_SLACK_MM = 11.5   # walker can chain a few bolt voxels
                                    # past the shallowest real contact
                                    # when bolt LoG response sits in the
                                    # contact-amplitude band; tighter
                                    # values cut real shanks short.
WALKER_GAP_SLACK_MM = 9.0          # 2 consecutive missed contacts at
                                    # the smallest library pitch
                                    # (3.5 mm) plus walker drift; real
                                    # shanks rarely lose >2 in a row.

# Geometric chain-formation floor — independent of library bounds.
# Set below the smallest library model (DIXI-5AM, PMT-8) so a real
# shank with 1-2 under-resolved contacts (low HU, partial-volume,
# or motion) still forms a chain and reaches the matched-filter
# picker. Library bounds still constrain the model picker downstream;
# they shouldn't double as the chain gate.
MIN_BLOBS_PER_LINE = 3
MIN_LINE_SPAN_MM = (
    _LIBRARY_BOUNDS["min_contact_span_mm"] - WALKER_SPAN_UNDER_SLACK_MM
)
MAX_LINE_SPAN_MM = (
    _LIBRARY_BOUNDS["max_contact_span_mm"] + WALKER_SPAN_OVER_SLACK_MM
)
MAX_INLIER_GAP_MM = (
    _LIBRARY_BOUNDS["max_within_electrode_pitch_mm"] + WALKER_GAP_SLACK_MM
)

STAGE1_DEDUP_ANGLE_DEG = 3.0
STAGE1_DEDUP_PERP_MM = 2.0
STAGE1_DEDUP_OVERLAP_FRAC = 0.3

DEEP_TIP_MIN_MM = 30.0          # strict floor for long lines (where
                                # sinus / skull-base tube FPs hide).
DEEP_TIP_MIN_SHORT_MM = 15.0    # short-line relaxation: superficial
                                # top-of-skull depths (T21 L_8/L_9/L_13)
                                # only reach ~15-20 mm intracranial.
DEEP_TIP_SHORT_MAX_AVG_PITCH_MM = 7.0
                                # Deep-tip prior discriminator: walker
                                # lines whose pre-extend inter-contact
                                # gap averages ≤ this get the relaxed
                                # DEEP_TIP_MIN_SHORT_MM=15 floor; any
                                # wider avg pitch means "not a real
                                # SEEG chain" → strict DEEP_TIP_MIN_MM
                                # floor. 7 mm covers Dixi (3.5),
                                # PMT 16B/C (3.97 / 4.43), and
                                # over-extension slack up to 2× nominal
                                # pitch. Cross-shank bridges + sinus
                                # FPs land well above 7 mm avg.
# Post-anchor length bounds. Anchored length = bolt-tip → deep contact
# = library contact span + bolt protrusion (and, for thin-wire PMT,
# the deep wire-segment gap that the contact span doesn't include).
# Both bounds are derived from the library at module load — adding a
# longer or shorter electrode model reshapes them automatically.
BOLT_PROTRUSION_MIN_MM = 16.0      # Short PMT bolts protrude ~12 mm
                                    # past the skull plus ~4 mm of
                                    # shallow electrode tail before the
                                    # first contact.
ANCHOR_TOTAL_OVERSHOOT_MM = 61.5    # Long-bolt + thin-wire-PMT slack
                                    # past the deepest library contact
                                    # span. Long DIXI bolts protrude
                                    # ~50 mm; the thin-wire PMT family
                                    # adds an unmodelled ~10 mm wire
                                    # segment beyond the deepest
                                    # contact (subject-137 L_3 measured
                                    # 84 mm contact span + 48 mm bolt
                                    # + wire = 132 mm anchored). This
                                    # collapses to ~50 mm once a thin-
                                    # wire PMT model ships in
                                    # electrode_models.json.

MIN_POST_ANCHOR_LEN_MM = (
    _LIBRARY_BOUNDS["min_contact_span_mm"] + BOLT_PROTRUSION_MIN_MM
)
MAX_POST_ANCHOR_LEN_MM = (
    _LIBRARY_BOUNDS["max_contact_span_mm"] + ANCHOR_TOTAL_OVERSHOOT_MM
)

# Post-anchor dedup: same-bolt + same-physical-line duplicates (multi-
# pitch walker passes that found disjoint blob ranges along one
# electrode). Two trajectories sharing a bolt CC are the same physical
# shank when their axes are nearly parallel AND their midpoints sit
# on the same line. Per-axis variation (1-3°) over a 50-80 mm reach
# would shift start_ras / end_ras by 5-15 mm, but the midpoint-to-
# axis perpendicular stays small. Distinct shanks merged into one
# bolt CC (T1, T22) have either non-parallel axes or non-coincident
# midpoints, so they survive.
POST_ANCHOR_DEDUP_PERP_MM = 3.0
POST_ANCHOR_DEDUP_ANG_DEG = 8.0  # 5° was too tight: auto-pitch
                                  # half-aliased walker passes (every-
                                  # other-contact at 6.7 mm vs 3.5)
                                  # produce axes 5-6° from the full-
                                  # pitch pass. 8° catches them; the
                                  # midpoint-perp 3 mm gate still
                                  # preserves distinct adjacent shanks
                                  # whose axes happen to be parallel
                                  # (T1, T22 mega-CC cases — those
                                  # have midpoint perp >> 3 mm).

# LOG_BOLT_NORMALIZER, HU_BOLT_NORMALIZER, METAL_BOLT_THRESHOLD moved to
# rosa_detect.primitives.bolt_anchor and re-exported above.

AXIS_SKULL_SYNTH_STEP_MM = 0.5      # Synth fallback (only path remaining
AXIS_SKULL_SYNTH_MAX_OUTWARD_MM = 80.0
AXIS_SKULL_SYNTH_BOLT_PROTRUDE_MM = 15.0
                                    # when the unified bolt CC pass finds
                                    # no anchor): walk outward along the
                                    # walker axis until it crosses the
                                    # hull, place a synthetic bolt-tip
                                    # PROTRUDE_MM further out. Recovers
                                    # T4-class subjects whose bolts sit
                                    # outside the CT acquisition window.

# BOLT_MIN_VOXELS through BOLT_BASE_MAX_DIST_MM moved to
# rosa_detect.primitives.bolt_anchor and re-exported above.


# ---- Pitch strategy + auto-detection ---------------------------------

# Candidate electrode-pitch set per UI strategy. The walker runs once
# per pitch in this set; hypotheses across pitches are unioned before
# dedup/arbitration so multi-family cases (e.g. Dixi + PMT on the
# same scan) get both families detected without the user picking a
# single pitch.
#
# For the "auto" strategy, pitches are estimated at runtime from the
# intracranial blob cloud's mutual-NN distance distribution (see
# ``detect_pitch_from_intracranial_blobs``). On a clean Dixi case the
# auto detector returns ≈ 3.3 mm; the surrounding ±0.5 mm tolerance in
# the walker absorbs the sub-bin localization bias.
# PITCH_STRATEGY_PITCHES_MM and PITCH_STRATEGY_VENDORS are now defined in
# `rosa_core.electrode_classifier` and re-exported via the import block at
# the top of this module.


# Pitch auto-detection + library snap live in candidate_seeds.pitch_resolution.
# Re-exported for back-compat.
from .candidate_seeds.pitch_library import LIBRARY_PITCHES_MM  # noqa: F401
from .candidate_seeds.pitch_resolution import (  # noqa: F401
    detect_pitch_from_intracranial_blobs,
    resolve_pitches_for_strategy,
    snap_to_library_pitch as _snap_to_library_pitch,
)
# Walker (blob-pitch chaining) + stage-1 runner. Re-exported under the
# legacy ``_*`` / no-prefix names so probes, tests, guided_fit_engine
# (cpfit._walk_line, cpfit.run_stage1, etc.) keep working.
from .candidate_seeds.walker import (  # noqa: F401
    refit_line_from_inliers as _refit_line_from_inliers,
    walk_line as _walk_line,
    walk_with_pitch_precomputed as _walk_with_pitch_precomputed,
)
from .candidate_seeds.stage1_runner import (  # noqa: F401
    arbitrate_blob_ownership as _arbitrate_blob_ownership,
    dedup_stage1_lines as _dedup_stage1_lines,
    extend_deep_end as _extend_deep_end,
    run_stage1,
    second_pass_orphan_walker as _second_pass_orphan_walker,
)
# Stage-2 trajectory refinement: crossing-tip retreat, deep-end LoG
# refinement, axis-to-skull synth anchor, cross-stage dedup. Re-exported
# under legacy ``_*`` names so the orchestrator + probes keep working.
from .candidate_seeds.crossing_tips import (  # noqa: F401
    retreat_crossing_tips as _retreat_crossing_tips,
)
from .candidate_seeds.deep_end_refine import (  # noqa: F401
    clip_deep_end_to_inliers as _clip_deep_end_to_inliers,
    refine_deep_end_via_axis_log as _refine_deep_end_via_axis_log,
)
from .candidate_seeds.synth_anchor import (  # noqa: F401
    axis_to_skull_synth as _axis_to_skull_synth,
)
from .candidate_seeds.dedup import (  # noqa: F401
    dedup_trajectories as _dedup_trajectories,
)
# Confidence score (continuous physical-evidence score in [0, 1] +
# high/medium/low banding). Re-exported under legacy ``_*`` names so
# the orchestrator + emit gates keep working.
from .candidate_seeds.confidence_score import (  # noqa: F401
    bolt_source_score as _bolt_source_score,
    compute_trajectory_score as _compute_trajectory_score,
    trapezoid_score as _trapezoid_score,
)
# Two-stage detection orchestrator (top-level entry point of the v1
# detector). The function is `@with_strategy_bounds`-decorated inside
# the new module, so the bound version flows through transparently.
from .candidate_seeds.orchestrator import (  # noqa: F401
    run_two_stage_detection,
)


# ---- Stage 1: blob-pitch ---------------------------------------------

# Blob extraction (LoG regional minima) lives in candidate_seeds.blob_extraction.
# Re-exported below so legacy callers (probes, tests, guided_fit_engine)
# that read symbols via ``cpfit.<name>`` keep working.
from .candidate_seeds.blob_extraction import (  # noqa: F401
    extract_blobs,
    extract_blob_cloud_ras,
)
from .candidate_seeds.constants import LOG_BLOB_SUBVOXEL_DEFAULT  # noqa: F401
# Frangi sampling, metal evidence, and median inlier pitch live in
# candidate_seeds.frangi_sampling and candidate_seeds.metal_evidence.
# Re-exported below for back-compat.
from .candidate_seeds.frangi_sampling import (  # noqa: F401
    frangi_along_line_stats as _frangi_along_line_stats,
    median_inlier_pitch as _median_inlier_pitch,
)
from .candidate_seeds.metal_evidence import (  # noqa: F401
    compute_metal_evidence_volume,
    frac_strong_metal_along_line as _frac_strong_metal_along_line,
)




# ---- Stage-2 trajectory refinement ------------------------------------
#
# Crossing-tip retreat, deep-end LoG refinement, axis-to-skull synth
# anchor, and cross-stage dedup live in candidate_seeds.* (see the
# re-export block above). Stage-2 calibration constants
# (AXIS_REFINE_*, DEEP_END_MARGIN_PAST_LAST_CONTACT_MM, CROSSING_*,
# AXIS_SKULL_SYNTH_*, POST_ANCHOR_DEDUP_*) live in
# candidate_seeds.constants and are re-imported at the top of this
# module so legacy callers (probes, tests) that read
# ``cpfit.<NAME>`` keep working.


# ---- Post-detection electrode classification -------------------------
#
# The constants (VENDOR_ID_PREFIXES, PITCH_STRATEGY_*), helper
# (_vendor_prefixes, _model_pitch_median_mm), library filter
# (filter_models_for_strategy), and scoring functions
# (suggest_shortest_covering_model, classify_by_walker_signature,
# classify_by_count_and_span, classify_pacer_template, and the unified
# classify_electrode_model dispatcher) live in
# `rosa_core.electrode_classifier`. Re-exported via the import block at
# the top of this module so existing callers (probes, tests) keep
# working unchanged.
#
# `refine_signature_via_axis_peaks` below stays here because it operates
# on a feature volume passed in by the detection pipeline and is not
# part of the picker's public surface.


def refine_signature_via_axis_peaks(rec, log_arr, ras_to_ijk_mat,
                                     step_mm=0.25,
                                     disk_radius_mm=2.0,
                                     n_radii=4,
                                     n_angles=8,
                                     min_amplitude=200.0,
                                     min_separation_mm=2.0,
                                     min_peaks_required=4,
                                     shallow_pad_mm=1.5,
                                     deep_pad_mm=3.0):
    """Re-derive ``(n_inliers, median_pitch, contact_span)`` for one
    trajectory by 1-D peak picking on the LoG profile sampled along
    its intracranial axis. Returns ``None`` if the axis yields fewer
    than ``min_peaks_required`` peaks (caller should keep walker
    stats in that case).

    The walker's NN-spacing pitch is biased on anisotropic CTs (e.g.
    S56's auto-detect locked to 3.14 mm instead of 3.5 mm — sub-voxel
    aliasing of blob centroids on the X/Y-downsampled grid). Peak
    detection along the FIT axis samples the LoG at 0.25 mm steps with
    trilinear interpolation, recovering sub-voxel peak positions and
    thus the true contact pitch.

    The ``shallow_pad_mm`` / ``deep_pad_mm`` extend the sampling range
    slightly past the skull entry and deep tip — small headroom catches
    the first / last contact when the bolt anchor or deep-end refine
    placed the endpoint just past the contact.

    Returns dict with keys ``n_peaks``, ``median_pitch_mm``,
    ``peak_span_mm``, ``peak_arc_mm`` (list, debug) or ``None``.
    """
    from rosa_core.contact_peak_fit import sample_axis_profile, detect_peaks_1d
    entry = np.asarray(
        rec.get("skull_entry_ras", rec.get("start_ras")), dtype=float,
    ).reshape(3)
    end = np.asarray(rec.get("end_ras"), dtype=float).reshape(3)
    axis_vec = end - entry
    L = float(np.linalg.norm(axis_vec))
    if L < 5.0:
        return None
    axis_unit = axis_vec / L
    sample_start = entry - float(shallow_pad_mm) * axis_unit
    sample_end = end + float(deep_pad_mm) * axis_unit
    try:
        arc_mm, profile = sample_axis_profile(
            volume_kji=log_arr,
            ras_to_ijk_mat=np.asarray(ras_to_ijk_mat, dtype=float),
            start_ras=sample_start, end_ras=sample_end,
            step_mm=step_mm, disk_radius_mm=disk_radius_mm,
            n_radii=n_radii, n_angles=n_angles, reducer="min",
        )
    except Exception:
        return None
    peaks_arc = detect_peaks_1d(
        profile, step_mm=step_mm, polarity="min",
        min_amplitude=min_amplitude,
        min_separation_mm=min_separation_mm,
    )
    if len(peaks_arc) < int(min_peaks_required):
        return None
    arr = np.asarray(sorted(peaks_arc), dtype=float)
    diffs = np.diff(arr)
    if diffs.size == 0:
        return None
    median_pitch = float(np.median(diffs))
    peak_span = float(arr[-1] - arr[0])
    return {
        "n_peaks": int(arr.size),
        "median_pitch_mm": median_pitch,
        "peak_span_mm": peak_span,
        "peak_arc_mm": [float(v) for v in arr.tolist()],
    }


