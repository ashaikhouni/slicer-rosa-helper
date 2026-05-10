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


# ---- Score framework v1 ----------------------------------------------
#
# Each emitted trajectory carries a continuous physical-evidence score
# in [0, 1] assembled from the same measurements the hard gates already
# apply: amp_sum, n_inliers, frangi shaft response, on-library pitch,
# pre-anchor contact span, post-anchor length, intracranial depth, and
# bolt-anchor source. Each component saturates at the corresponding
# gate threshold (`measure ≥ threshold` → 1.0), so a clean SEEG line
# scores near 1.0 on every term.
#
# v1 keeps the existing emit-time gates as fallbacks. The score is
# attached as metadata (`score`, `confidence`, `score_components`) on
# every survivor so downstream code can rank, filter, or surface the
# weakest emissions for review without changing detection behaviour.
SCORE_WEIGHTS = {
    "amp": 1.0,
    "n_inliers": 1.0,
    "frangi": 1.0,
    "pitch": 1.0,
    "span": 1.0,
    "length": 1.0,
    "depth": 1.0,         # was 0.5; SEEG is a depth-electrode technique by
                           # definition, so depth is a load-bearing signal,
                           # not a soft prior.
    "intracranial": 0.5,
    "bolt": 1.0,
    "metal_continuity": 2.0,  # frac_strong-along-axis: real shanks have
                               # discrete contact-saturating peaks all
                               # along the line (matched p10=0.27,
                               # p50=0.65); cross-shank chains and
                               # synth-extended FPs have frac near 0
                               # (orphan p50=0.01). Weight 2.0 pushes
                               # zero-saturation orphans into LOW band.
}
SCORE_METAL_CONTINUITY_SAT = 0.10  # frac_strong saturation: matched p1
                                    # ≈ 0.16, so 0.10 gives nearly all
                                    # matched full credit while penalizing
                                    # orphans clustered at 0.00-0.05.
SCORE_HIGH_THRESHOLD = 0.80
SCORE_MEDIUM_THRESHOLD = 0.50
SCORE_PITCH_TOL_MM = 0.25      # falloff width around library pitch.
                                # Calibrated against dataset matched
                                # distribution: matched p90 = 0.18 mm
                                # dev (well within tolerance), orphan
                                # p50 = 0.35 mm dev (now scores 0).
                                # Math: per-peak position σ ≈ 0.15 mm,
                                # median pitch σ ≈ 0.07 mm for N=15
                                # contacts, so 0.25 mm covers 3.5-σ of
                                # measurement noise. Earlier value
                                # 1.0 mm was a generous shoulder
                                # carried over from the Ball r=2 era.
SCORE_SPAN_SHOULDER_MM = 6.0   # linear falloff outside [12, 90]
SCORE_LENGTH_SHOULDER_MM = 10.0  # linear falloff outside [30, 140]
SCORE_AMP_SAT = 5000.0
SCORE_N_INLIERS_SLOPE = 10.0   # n_inliers = MIN_BLOBS_PER_LINE → 0,
                                # n_inliers = MIN + slope → 1.0
SCORE_N_INLIERS_OVER_SLACK = 12.0  # falloff width above the library
                                    # contact-count maximum. n equal to
                                    # the max → 1.0; n at max + slack →
                                    # 0.0. Catches walker chains that
                                    # ran into a continuous metal
                                    # structure (e.g. the bolt itself
                                    # at thread-pitch aliasing).
SCORE_DEPTH_SAT_MM = 30.0      # dist_max at which the depth term
                                # saturates at 1.0. Independent of
                                # ``DEEP_TIP_MIN_MM`` so the gate can
                                # be tuned / retired without changing
                                # the score's depth behaviour.
SCORE_INTRACRANIAL_SAT_MM = 10.0
SCORE_BOLT_VALUES = {
    "metal": 1.0,        # unified bolt CC (replaces "log" + "hu_rescue")
    "metal_cc": 0.7,     # wire-class: bolt CC extends into brain as
                          # continuous metal; walker found no contact-pitch
                          # line (saturated/merged contacts), but the CC
                          # itself defines the shank axis. Lower than
                          # "metal" because no contact-pitch validation.
    "synthesized": 0.4,  # axis-to-skull synth fallback
    "none": 0.1,         # no anchor and synth couldn't reach hull
}

# Wire-class extension: when a bolt CC is unmatched by any walker line
# AND its connected metal extends into the brain, the CC IS the shank
# (saturated contacts merging). Emit a PCA-fit trajectory through the
# bolt+wire CC. Catches AMC099 L_4 and similar short / lateral / strong-
# metal cases.
WIRE_CLASS_MIN_DEPTH_MM = 15.0       # bolt CC's deepest voxel must sit
                                       # ≥ 15 mm inside the head — true bolt
                                       # screws cap at ~10 mm. The signal is
                                       # "metal continues past the bolt".
WIRE_CLASS_MIN_SPAN_MM = 15.0         # PCA major-axis projection range.
                                       # Below 15 mm overlaps screw-only
                                       # CCs and short surgical clips.
WIRE_CLASS_MIN_VOXELS = 50            # excludes tiny clips and CC noise.
WIRE_CLASS_MIN_ELONGATION = 0.65      # major eigenvalue / sum-of-eigenvalues.
                                       # Cylindrical bolts/wires sit > 0.7;
                                       # blob-shaped clips < 0.5.


def _trapezoid_score(value, lo, hi, shoulder_mm):
    """1.0 inside [lo, hi]; linear falloff to 0 over `shoulder_mm`."""
    if value < lo - shoulder_mm or value > hi + shoulder_mm:
        return 0.0
    if value < lo:
        return float((value - (lo - shoulder_mm)) / shoulder_mm)
    if value > hi:
        return float(((hi + shoulder_mm) - value) / shoulder_mm)
    return 1.0


def _bolt_source_score(src):
    return SCORE_BOLT_VALUES.get(str(src), 0.5)


def _compute_trajectory_score(rec):
    """Return (score, confidence, components) for one trajectory record."""
    components = {}
    is_wire_class = bool(rec.get("wire_class"))

    # ``amp_sum`` and ``n_inliers`` are walker-only signals. Wire-class
    # trajectories come from a bolt-CC PCA fit and have neither — they
    # would force-zero those components and drag the score artificially
    # low. Skip them and let the remaining components (frangi, span,
    # length, depth, intracranial, bolt, metal_continuity) carry the
    # signal.
    if "amp_sum" in rec and not is_wire_class:
        components["amp"] = (
            min(1.0, max(0.0, float(rec["amp_sum"]) / SCORE_AMP_SAT)),
            SCORE_WEIGHTS["amp"],
        )

    if not is_wire_class:
        n = int(rec.get("n_inliers", 0))
        # Lower side: linear ramp from MIN_BLOBS_PER_LINE up by SCORE_N_INLIERS_SLOPE.
        # Upper side: 1.0 up to the library's max contact count, then linear
        # falloff over SCORE_N_INLIERS_OVER_SLACK. n far above the library
        # max means the walker chained a continuous metal structure (the
        # bolt itself, an insulated wire shaft) instead of discrete contacts.
        lib_max = int(_LIBRARY_BOUNDS["max_contacts"])
        if n <= MIN_BLOBS_PER_LINE:
            n_score = 0.0
        elif n <= MIN_BLOBS_PER_LINE + SCORE_N_INLIERS_SLOPE:
            n_score = (n - MIN_BLOBS_PER_LINE) / SCORE_N_INLIERS_SLOPE
        elif n <= lib_max:
            n_score = 1.0
        else:
            n_score = max(0.0, 1.0 - (n - lib_max) / SCORE_N_INLIERS_OVER_SLACK)
        components["n_inliers"] = (n_score, SCORE_WEIGHTS["n_inliers"])

    if "frangi_median_mm" in rec:
        components["frangi"] = (
            min(1.0, max(0.0, float(rec["frangi_median_mm"]) / FRANGI_LINE_MIN_MEDIAN)),
            SCORE_WEIGHTS["frangi"],
        )

    if "frac_strong_metal" in rec:
        components["metal_continuity"] = (
            min(1.0, max(0.0, float(rec["frac_strong_metal"]) / SCORE_METAL_CONTINUITY_SAT)),
            SCORE_WEIGHTS["metal_continuity"],
        )

    pitch = rec.get("original_median_pitch_mm")
    if pitch is not None and float(pitch) > 0.0:
        dev = min(abs(float(pitch) - lib) for lib in LIBRARY_PITCHES_MM)
        components["pitch"] = (
            min(1.0, max(0.0, 1.0 - dev / SCORE_PITCH_TOL_MM)),
            SCORE_WEIGHTS["pitch"],
        )

    if "contact_span_mm" in rec:
        components["span"] = (
            _trapezoid_score(
                float(rec["contact_span_mm"]),
                MIN_LINE_SPAN_MM, MAX_LINE_SPAN_MM,
                SCORE_SPAN_SHOULDER_MM,
            ),
            SCORE_WEIGHTS["span"],
        )

    length = float(rec.get("length_mm", 0.0))
    if length > 0:
        components["length"] = (
            _trapezoid_score(
                length,
                MIN_POST_ANCHOR_LEN_MM, MAX_POST_ANCHOR_LEN_MM,
                SCORE_LENGTH_SHOULDER_MM,
            ),
            SCORE_WEIGHTS["length"],
        )

    dist_max = float(rec.get("dist_max_mm", 0.0))
    components["depth"] = (
        min(1.0, max(0.0, dist_max / SCORE_DEPTH_SAT_MM)),
        SCORE_WEIGHTS["depth"],
    )

    dist_mean = rec.get("dist_mean_mm")
    if dist_mean is not None and float(dist_mean) == float(dist_mean):
        components["intracranial"] = (
            min(1.0, max(0.0, float(dist_mean) / SCORE_INTRACRANIAL_SAT_MM)),
            SCORE_WEIGHTS["intracranial"],
        )

    bolt_src = str(rec.get("bolt_source", "metal"))
    components["bolt"] = (
        _bolt_source_score(bolt_src),
        SCORE_WEIGHTS["bolt"],
    )

    weighted = sum(v * w for v, w in components.values())
    total_w = sum(w for _, w in components.values())
    score = weighted / total_w if total_w > 0 else 0.0

    # Confidence policy: high band is reserved for trajectories with BOTH
    # contact-pitch validation AND a real metal bolt CC (bolt_source ==
    # "metal"). Anything missing one or the other caps at medium:
    #
    #   pitch + metal bolt    → high allowed
    #   pitch + synthesized   → cap medium  (CT didn't capture the bolt;
    #                                         the synth fallback is a
    #                                         best-guess on degraded input)
    #   pitch + no anchor     → cap medium  (bolt_source == "none")
    #   bolt CC + no pitch    → cap medium  (wire_class; metal_cc bolt)
    #
    # Wire-class records carry bolt_source == "metal_cc" by construction,
    # so the single ``bolt_src != "metal"`` test covers them too.
    if bolt_src != "metal" and score >= SCORE_HIGH_THRESHOLD:
        score = SCORE_HIGH_THRESHOLD - 0.01

    if score >= SCORE_HIGH_THRESHOLD:
        label = "high"
    elif score >= SCORE_MEDIUM_THRESHOLD:
        label = "medium"
    else:
        label = "low"

    return score, label, {k: float(v) for k, (v, _) in components.items()}


# ---- Orchestration ----------------------------------------------------

@_with_strategy_bounds
def run_two_stage_detection(img, ijk_to_ras_mat, ras_to_ijk_mat,
                             return_features=False, progress_logger=None,
                             suggestion_vendors=None,
                             pitch_strategy=None,
                             pitches_mm=None):
    """Run the full SEEG shank detector on a SITK image.

    Args:
        img: SimpleITK image (raw CT).
        ijk_to_ras_mat: 4x4 numpy matrix.
        ras_to_ijk_mat: 4x4 numpy matrix.
        return_features: if True, return (trajectories, feature_arrays)
            where feature_arrays is a dict with the LoG, Frangi, hull
            head-distance, intracranial and hull arrays (KJI-order).
        progress_logger: optional callable(message: str) invoked at each
            major checkpoint. The Slicer widget passes a callback that
            updates the status panel and runs `app.processEvents()` so
            the UI doesn't appear hung during the ~10–20 s detection.

    Returns:
        list[dict] or (list[dict], dict): trajectories list (always) and
        optionally a feature_arrays dict for debugging / visualization.
    """
    def _log(msg):
        if progress_logger is not None:
            try:
                progress_logger(msg)
            except Exception:
                pass

    import SimpleITK as sitk
    # Canonicalize spacing + anti-alias + HU clamp. Shared with
    # ``guided_fit_engine.compute_features`` so both paths see the
    # identical preprocessed volume — any drift here is a P0 parity
    # bug per ``feedback_cli_slicer_parity.md``.
    img, ijk_to_ras_mat, ras_to_ijk_mat = prepare_volume(
        img, ijk_to_ras_mat, ras_to_ijk_mat,
    )
    _log("preprocessing: hull, head-distance, intracranial mask…")
    ct_arr_kji = sitk.GetArrayFromImage(img).astype(np.float32)
    # Input fingerprint — lets us compare Slicer vs CLI runs byte-for-byte.
    # If Slicer returns a different trajectory count, the most common
    # causes are (a) HU rescaling (NIfTI scl_slope/scl_inter applied
    # differently) and (b) IJK→RAS matrix mismatch; this trace exposes both.
    try:
        _sp = img.GetSpacing()
        _dg = [ijk_to_ras_mat[i, i] for i in range(3)]
        _org = [ijk_to_ras_mat[i, 3] for i in range(3)]
        _log(
            f"input fingerprint: shape={ct_arr_kji.shape} "
            f"HU[min/mean/max]={ct_arr_kji.min():.1f}/"
            f"{ct_arr_kji.mean():.1f}/{ct_arr_kji.max():.1f} "
            f"spacing={tuple(f'{s:.4f}' for s in _sp)} "
            f"ijk2ras_diag={tuple(f'{d:+.4f}' for d in _dg)} "
            f"origin={tuple(f'{o:+.2f}' for o in _org)}"
        )
    except Exception:
        pass
    hull, intracranial, dist_arr = build_masks(img)
    _log("preprocessing: LoG σ=1…")
    log1 = log_sigma(img, sigma_mm=LOG_SIGMA_MM)
    _log("preprocessing: Frangi σ=1…")
    frangi_s1 = frangi_single(img, sigma=FRANGI_STAGE1_SIGMA)
    kji_to_ras = _kji_to_ras_fn_from_matrix(ijk_to_ras_mat)

    # Resolve walker pitches from the caller's strategy. Explicit
    # ``pitches_mm`` override takes precedence (used by unit tests and
    # power users). Otherwise fall back to strategy lookup — "auto"
    # auto-detects pitch from the intracranial blob cloud here so
    # stage-1 sees the right pitches from its first pass.
    if pitches_mm is not None and len(tuple(pitches_mm)) > 0:
        resolved_pitches = tuple(float(p) for p in pitches_mm)
    elif pitch_strategy is not None:
        strat_key = str(pitch_strategy).lower()
        if strat_key == "auto":
            _log("auto-detect pitch: extracting blobs…")
            _blobs_preview = extract_blobs(log1, threshold=LOG_BLOB_THRESHOLD)
            _pts_preview = (
                np.array([kji_to_ras(b["kji"]) for b in _blobs_preview])
                if _blobs_preview
                else np.empty((0, 3), dtype=float)
            )
            if _pts_preview.shape[0] > 0:
                _n_vox_preview = np.array(
                    [b["n_vox"] for b in _blobs_preview], dtype=int,
                )
                _pts_c_preview = _pts_preview[_n_vox_preview <= LOG_BLOB_MAX_VOXELS]
            else:
                _pts_c_preview = _pts_preview
            _raw_detected = detect_pitch_from_intracranial_blobs(
                _pts_c_preview, dist_arr, ras_to_ijk_mat,
            ) if _pts_c_preview.shape[0] > 0 else []
            resolved_pitches = resolve_pitches_for_strategy(
                "auto",
                pts_c=_pts_c_preview,
                dist_arr=dist_arr,
                ras_to_ijk_mat=ras_to_ijk_mat,
            )
            if _raw_detected:
                _log(
                    f"auto-detect pitch: raw={[f'{p:.2f}' for p in _raw_detected]} "
                    f"→ snapped={[f'{p:.2f}' for p in resolved_pitches]} mm"
                )
            else:
                _log(
                    f"auto-detect pitch: using {[f'{p:.2f}' for p in resolved_pitches]} mm (fallback)"
                )
        else:
            resolved_pitches = resolve_pitches_for_strategy(strat_key)
    else:
        resolved_pitches = (PITCH_MM,)

    _log(
        f"stage 1: blob-pitch walker — pitches={[f'{p:.2f}' for p in resolved_pitches]} mm"
    )
    stage1_lines, pts_blobs = run_stage1(
        log1, kji_to_ras, dist_arr, ras_to_ijk_mat,
        pitches_mm=resolved_pitches,
        frangi_arr=frangi_s1,
    )
    _log(f"stage 1: {len(stage1_lines)} candidate lines after walk + arbitrate + extend")
    # Attach inlier RAS coords AND LoG amplitudes to each stage-1 line
    # so post-anchor refinement can clip the deep end to the last
    # STRONG real contact (weak/noisy blobs added by extension don't
    # count as legit deep endpoints).
    import numpy as _np
    # Re-derive LoG amplitudes at each contact-sized blob position —
    # pts_blobs is already the contact-filtered cloud, so indexing
    # matches line["inlier_idx"]. If this fails (shape mismatch, etc.)
    # the deep-end strong-contact clipping silently degrades to "no
    # amplitude data" — log it loudly so a regression can't hide here.
    try:
        K, J, I = log1.shape
        h_all = _np.concatenate([pts_blobs, _np.ones((pts_blobs.shape[0], 1))], axis=1)
        ijk_all = (ras_to_ijk_mat @ h_all.T).T[:, :3]
        ii = _np.clip(_np.round(ijk_all[:, 0]).astype(int), 0, I - 1)
        jj = _np.clip(_np.round(ijk_all[:, 1]).astype(int), 0, J - 1)
        kk = _np.clip(_np.round(ijk_all[:, 2]).astype(int), 0, K - 1)
        blob_amps = _np.abs(log1[kk, jj, ii]).astype(_np.float32)
    except Exception as exc:
        _log(
            f"warning: blob_amps re-derivation failed ({exc}); "
            f"deep-end strong-contact clipping disabled for all stage-1 lines"
        )
        blob_amps = None
    for l in stage1_lines:
        try:
            l["inlier_ras"] = _np.asarray(pts_blobs[l["inlier_idx"]], dtype=float)
            if blob_amps is not None:
                l["inlier_amps"] = _np.asarray(blob_amps[l["inlier_idx"]], dtype=float)
        except Exception as exc:
            _log(
                f"warning: line inlier_idx → blob lookup failed "
                f"({exc}); inlier_ras / inlier_amps cleared for one line"
            )
            l["inlier_ras"] = None
            l["inlier_amps"] = None

    _log("bolt extraction (unified metal-evidence)…")
    metal_evidence_vol = compute_metal_evidence_volume(log1, ct_arr_kji)
    bolts, bolt_mask = extract_bolt_candidates(
        log1, dist_arr, ijk_to_ras_mat, img.GetSpacing(),
        ct_arr=metal_evidence_vol, hu_threshold=METAL_BOLT_THRESHOLD,
        hull_proximity_mm=BOLT_HULL_PROXIMITY_MM,
    )
    _log(f"bolt extraction: {len(bolts)} bolt candidates "
         f"(metal_evidence ≥ {METAL_BOLT_THRESHOLD:.2f}, "
         f"hull_prox ≤ {BOLT_HULL_PROXIMITY_MM:.1f} mm)")

    def _assemble(l):
        rec = dict(
            start_ras=np.asarray(l["start_ras"], dtype=float),
            end_ras=np.asarray(l["end_ras"], dtype=float),
            shallow_endpoint_name="start",
            deep_endpoint_name="end",
            length_mm=float(l.get("span_mm", l.get("length_mm", 0.0))),
            n_inliers=int(l.get("n_blobs", l.get("n_inliers", 0))),
            dist_min_mm=float(l.get("dist_min_mm", float("nan"))),
            dist_max_mm=float(l.get("dist_max_mm", float("nan"))),
            dist_mean_mm=float(l.get("dist_mean_mm", float("nan"))),
            amp_sum=float(l.get("amp_sum", 0.0)),
        )
        # Carry the walker's blob-inlier set through anchoring so
        # ``_dedup_trajectories`` can apply the inlier-subset rule
        # (drop only when no blob is orphaned).
        inlier_idx = l.get("inlier_idx")
        if inlier_idx is not None:
            rec["inlier_idx"] = list(int(b) for b in inlier_idx)
        # Preserve the pre-anchor inlier span — the actual distance
        # between the shallowest and deepest detected contacts. The
        # post-anchor ``length_mm`` overwrites this with the
        # bolt-tip → deep-tip length, so downstream classifiers
        # need this field to see the true contact span.
        rec["contact_span_mm"] = float(l.get("span_mm", 0.0))
        rec["original_span_mm"] = float(
            l.get("original_span_mm", l.get("span_mm", 0.0))
        )
        rec["original_median_pitch_mm"] = float(l.get(
            "original_median_pitch_mm",
            (float(l.get("original_span_mm", l.get("span_mm", 0.0)))
             / max(1, int(l.get("n_blobs", 2)) - 1)),
        ))
        if "frangi_mean_mm" in l:
            rec["frangi_mean_mm"] = float(l["frangi_mean_mm"])
        if "frangi_median_mm" in l:
            rec["frangi_median_mm"] = float(l["frangi_median_mm"])
        inlier_ras = l.get("inlier_ras")
        if inlier_ras is not None:
            rec["inlier_ras"] = np.asarray(inlier_ras, dtype=float)
        inlier_amps = l.get("inlier_amps")
        if inlier_amps is not None:
            rec["inlier_amps"] = np.asarray(inlier_amps, dtype=float)
        return rec

    def _is_genuine_seeg_line(rec):
        """Strong-SEEG-chain gate for the synth fallback.

        A real SEEG electrode's walker pre-extension line has Dixi/PMT
        pitch (3-5 mm). Uses the pre-extend MEDIAN pitch — robust to
        one walker-absorbed outlier that would skew a mean-based
        statistic past the 7 mm cap even when most inliers sit on a
        regular chain. Falls back to min(span_pre, span_post)/(n-1)
        when median isn't available.
        """
        n = int(rec.get("n_inliers", 0))
        if n < MIN_BLOBS_PER_LINE:
            return False
        dist_max = float(rec.get("dist_max_mm", 0.0))
        if dist_max < DEEP_TIP_MIN_MM:
            return False
        span_post = float(rec.get("contact_span_mm", rec.get("length_mm", 0.0)))
        span_pre = float(rec.get("original_span_mm", span_post))
        span_for_pitch = min(span_pre, span_post) if span_pre > 0 else span_post
        fallback_avg = span_for_pitch / (n - 1) if n > 1 else float("inf")
        median_pitch = float(rec.get("original_median_pitch_mm", fallback_avg))
        if median_pitch > DEEP_TIP_SHORT_MAX_AVG_PITCH_MM:
            return False
        return True

    # Anchor each candidate to the unified metal-evidence bolt CC pool
    # BEFORE dedup. Length and Frangi-tubularity filters catch the rare
    # walker false positives that survive the anchor step.
    def _anchor_or_reject(rec):
        # ``_orient_shallow_to_deep`` upstream uses hull head-distance
        # to pick which endpoint is the shallow one, but that's
        # ambiguous for trajectories whose deep tip sits in a deep
        # sulcus as close to its local hull surface as the bolt side
        # (T22 LGR: orbital-floor tip is ~10 mm from hull, skull-top
        # bolt is ~15 mm from hull — orientation flipped). Let the
        # bolt CC decide by trying both orientations and keeping the
        # one whose bolt anchor has more tube voxels.
        fwd = anchor_trajectory_to_bolt(
            rec["start_ras"], rec["end_ras"], bolts,
        )
        bwd = anchor_trajectory_to_bolt(
            rec["end_ras"], rec["start_ras"], bolts,
        )
        fwd_n = int(fwd[2].get("tube_n_vox", 0)) if fwd[2] is not None else 0
        bwd_n = int(bwd[2].get("tube_n_vox", 0)) if bwd[2] is not None else 0

        if bwd_n > fwd_n:
            # Orientation was wrong; flip before writing results back.
            rec["start_ras"], rec["end_ras"] = (
                np.asarray(rec["end_ras"], dtype=float),
                np.asarray(rec["start_ras"], dtype=float),
            )
            new_start, skull_entry, bolt = bwd
        else:
            new_start, skull_entry, bolt = fwd
        # Synth fallback: when the unified bolt CC pass found no anchor
        # (T4-class shanks whose bolts sit outside the CT acquisition
        # window) AND the walker line looks unambiguously like a real
        # SEEG chain, synthesize a skull_entry + bolt_tip by walking the
        # walker axis outward until it crosses the hull surface.
        bolt_from_synth = None
        if new_start is None and _is_genuine_seeg_line(rec):
            s0, e0 = _orient_shallow_to_deep(
                rec["start_ras"], rec["end_ras"],
                dist_arr, ras_to_ijk_mat,
            )
            synth_skull, synth_tip = _axis_to_skull_synth(
                s0, e0, dist_arr, ras_to_ijk_mat,
            )
            if synth_skull is not None:
                rec["start_ras"] = np.asarray(s0, dtype=float)
                rec["end_ras"] = np.asarray(e0, dtype=float)
                new_start = synth_tip
                skull_entry = synth_skull
                bolt_from_synth = {"n_vox": 0, "dist_min_mm": float("nan"),
                                    "id": -1}
        if new_start is None:
            # Recall-first emission: no metal-evidence bolt CC AND
            # axis-to-skull synth never crossed the hull (e.g. T4 RPOG —
            # bolt sits outside the CT acquisition window AND the axis
            # never reaches the hull). Emit the walker's line as-is.
            # Confidence scoring will downweight these no-anchor
            # emissions.
            rec["bolt_source"] = "none"
            bolt = {"n_vox": 0, "dist_min_mm": float("nan"), "id": -1}
        else:
            if bolt_from_synth is not None:
                bolt = bolt_from_synth
                rec["bolt_source"] = "synthesized"
            else:
                rec["bolt_source"] = "metal"
            rec["start_ras"] = np.asarray(new_start, dtype=float)
            if skull_entry is not None:
                rec["skull_entry_ras"] = np.asarray(skull_entry, dtype=float)
        rec["length_mm"] = float(np.linalg.norm(rec["end_ras"] - rec["start_ras"]))
        # Length sanity: real SEEG total length (bolt + shank) is bounded.
        if (rec["length_mm"] < MIN_POST_ANCHOR_LEN_MM
                or rec["length_mm"] > MAX_POST_ANCHOR_LEN_MM):
            return None
        # Re-validate Frangi tubular evidence on the FULL extended axis
        # (bolt-tip → deep-tip), not just the contact span. Stage 1's
        # Frangi gate only saw the contact span; the bolt anchor extends
        # ``start_ras`` outward by 15-60 mm without checking what's in
        # between. Real shanks have continuous wire+bolt support and
        # keep frangi_median ≥ 30 even on the extended line. Synthesized
        # FPs extend through brain/air with no metal support and drop
        # out here. Overwrite the score's ``frangi_median_mm`` so it
        # sees the true post-anchor value.
        new_fmean, new_fmed = _frangi_along_line_stats(
            rec["start_ras"], rec["end_ras"], frangi_s1, ras_to_ijk_mat,
        )
        rec["frangi_mean_mm"] = float(new_fmean)
        rec["frangi_median_mm"] = float(new_fmed)
        # Metal-continuity score feature: fraction of full-axis samples
        # whose unified metal evidence saturates (|LoG|≥LOG_BOLT_NORMALIZER
        # OR HU≥HU_BOLT_NORMALIZER). Real shanks have many discrete contact
        # peaks along the axis; cross-shank bone-assembled chains have a
        # few saturating spots clustered at one end with empty middle.
        # Computed BEFORE the Frangi gate so the gate can defer to it on
        # thin-wire shanks (long wire between bolt and contact array
        # contributes no Frangi response, dragging median below 30 even
        # though both ends have strong metal evidence).
        rec["frac_strong_metal"] = _frac_strong_metal_along_line(
            rec["start_ras"], rec["end_ras"],
            log1, ct_arr_kji, ras_to_ijk_mat,
        )
        # Frangi gate fires ONLY when the bolt anchor failed
        # (synthesized / none). When ``bolt_source == "metal"`` the
        # anchor latched onto a real bolt CC and that's already proof
        # the line is real — even if the segment between bolt and
        # contacts is a wire too thin to register as tubular (e.g.
        # ct88 L_37: 50 mm wire gap, frangi_median=8.6, but bolt
        # anchored at 122 tube voxels). The score's Frangi component
        # penalizes weak Frangi proportionally; no need for a hard
        # cliff. Synthesized / none lines still get the gate because
        # the synth fallback can extend through brain/air with no
        # metal evidence on a cross-shank trajectory.
        if (rec["bolt_source"] != "metal"
                and new_fmed < FRANGI_LINE_MIN_MEDIAN):
            return None
        rec["bolt_n_vox"] = int(bolt["n_vox"])
        rec["bolt_dist_min_mm"] = float(bolt["dist_min_mm"])
        rec["bolt_id"] = int(bolt.get("id", -1))
        return rec

    _log("anchoring + length filters…")
    anchored: list[dict[str, Any]] = []
    for l in stage1_lines:
        rec = _anchor_or_reject(_assemble(l))
        if rec is not None:
            anchored.append(rec)
    _log(f"anchoring: {len(anchored)} survived")

    # Wire-class extension: emit trajectories from unmatched bolt CCs whose
    # connected metal extends into the brain. Catches shanks where contacts
    # saturate into a continuous wire (clinical CTs, lateral / short
    # entries — e.g. AMC099 L_4) that the contact-pitch walker can't
    # resolve into discrete LoG peaks.
    used_bolt_ids = {
        int(rec.get("bolt_id", -1))
        for rec in anchored
        if int(rec.get("bolt_id", -1)) >= 0
    }
    wire_recs: list[dict[str, Any]] = []
    for bolt in bolts:
        if int(bolt["id"]) in used_bolt_ids:
            continue
        if int(bolt["n_vox"]) < WIRE_CLASS_MIN_VOXELS:
            continue
        if float(bolt["dist_max_mm"]) < WIRE_CLASS_MIN_DEPTH_MM:
            continue
        pts = np.asarray(bolt["pts_ras"], dtype=float)
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        cov = (centered.T @ centered) / max(1, len(pts) - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        total_var = float(eigvals.sum())
        if total_var <= 1e-9:
            continue
        elongation = float(eigvals[-1]) / total_var
        if elongation < WIRE_CLASS_MIN_ELONGATION:
            continue
        axis_dir = eigvecs[:, -1]
        proj = centered @ axis_dir
        span = float(proj.max() - proj.min())
        if span < WIRE_CLASS_MIN_SPAN_MM:
            continue
        p_min = centroid + float(proj.min()) * axis_dir
        p_max = centroid + float(proj.max()) * axis_dir
        s_ras, e_ras = _orient_shallow_to_deep(
            p_min, p_max, dist_arr, ras_to_ijk_mat,
        )
        rec = dict(
            start_ras=np.asarray(s_ras, dtype=float),
            end_ras=np.asarray(e_ras, dtype=float),
            shallow_endpoint_name="start",
            deep_endpoint_name="end",
            length_mm=float(np.linalg.norm(e_ras - s_ras)),
            n_inliers=0,
            contact_span_mm=span,
            original_span_mm=span,
            original_median_pitch_mm=0.0,
            dist_min_mm=float(bolt["dist_min_mm"]),
            dist_max_mm=float(bolt["dist_max_mm"]),
            dist_mean_mm=float(np.mean(bolt["pts_dist"])),
            bolt_source="metal_cc",
            bolt_n_vox=int(bolt["n_vox"]),
            bolt_dist_min_mm=float(bolt["dist_min_mm"]),
            bolt_id=int(bolt["id"]),
            wire_class=True,
            inlier_idx=[],
        )
        new_fmean, new_fmed = _frangi_along_line_stats(
            rec["start_ras"], rec["end_ras"], frangi_s1, ras_to_ijk_mat,
        )
        rec["frangi_mean_mm"] = float(new_fmean)
        rec["frangi_median_mm"] = float(new_fmed)
        rec["frac_strong_metal"] = _frac_strong_metal_along_line(
            rec["start_ras"], rec["end_ras"],
            log1, ct_arr_kji, ras_to_ijk_mat,
        )
        rec["skull_entry_ras"] = np.asarray(s_ras, dtype=float)
        rec["intracranial_length_mm"] = float(np.linalg.norm(e_ras - s_ras))
        wire_recs.append(rec)
    if wire_recs:
        _log(f"wire-class: emitting {len(wire_recs)} from unmatched bolt CCs")
        anchored.extend(wire_recs)

    # Dedup keeps the longer line of each cluster.
    anchored.sort(key=lambda rec: -float(rec.get("length_mm", 0.0)))
    anchored = _dedup_trajectories(anchored)
    _log(f"final dedup: {len(anchored)} trajectories")

    # Axis-directed deep-end refinement. The 3D regional-minima blob
    # extractor misses contacts when the per-contact LoG wells merge
    # into one continuous CC (seen on T2 X06 / RAI, where the deep 3–4
    # contacts sit inside a single long bright shaft and don't produce
    # distinct 3D minima). Sample the LoG profile 1-dimensionally along
    # the trajectory axis and push ``end_ras`` out to the last real
    # contact peak.
    for rec in anchored:
        new_end = _refine_deep_end_via_axis_log(
            rec, log1, ras_to_ijk_mat,
        )
        if new_end is not None:
            rec["end_ras"] = new_end
        # Hard cap: end must sit within DEEP_END_MARGIN_PAST_LAST_CONTACT_MM
        # of the deepest walker inlier. No SEEG electrode has a long gap
        # past its last contact; anything further is over-reach.
        clipped = _clip_deep_end_to_inliers(rec)
        if clipped is not None:
            rec["end_ras"] = clipped
        if new_end is not None or clipped is not None:
            rec["length_mm"] = float(
                np.linalg.norm(
                    np.asarray(rec["end_ras"]) - np.asarray(rec["start_ras"])
                )
            )

    # Crossing-tip retreat: after all trajectories have been extended,
    # pull back any tip that lives inside another's contact-acceptance
    # tube. Runs only at the final stage so every axis has settled
    # before we decide which tip is the intruder. Passing the LoG
    # volume lets the retreat additionally snap the pulled-back tip to
    # the deep edge of the last real contact instead of floating in the
    # gap between contacts.
    _log("crossing-tip retreat…")
    _retreat_crossing_tips(
        anchored,
        log_arr=log1,
        ras_to_ijk_mat=ras_to_ijk_mat,
        logger=_log,
    )

    # Intracranial-only length (skull entry → deep tip). The existing
    # ``length_mm`` is bolt-tip → deep-tip and includes ~15–25 mm of bolt
    # protrusion outside the skull; downstream displays/clinical reporting
    # want the part that actually sits inside the brain.
    for rec in anchored:
        entry = np.asarray(
            rec.get("skull_entry_ras", rec.get("start_ras")),
            dtype=float,
        )
        end = np.asarray(rec["end_ras"], dtype=float)
        rec["intracranial_length_mm"] = float(np.linalg.norm(end - entry))

    # Suggested electrode model per stage-1 trajectory. Uses the
    # pre-anchor contact span + inlier count against the library,
    # filtered by the caller's ``suggestion_vendors`` selection (or
    # all known vendors when not specified). Advisory only — downstream
    # modules such as Contacts & Trajectory View do the actual contact
    # fitting and the user can override this suggestion.
    if suggestion_vendors is not None:
        vendors_for_suggest = tuple(suggestion_vendors)
    elif pitch_strategy is not None:
        strat_key = str(pitch_strategy).lower()
        vendors_for_suggest = PITCH_STRATEGY_VENDORS.get(
            strat_key, tuple(VENDOR_ID_PREFIXES.keys()),
        )
    else:
        vendors_for_suggest = tuple(VENDOR_ID_PREFIXES.keys())
    if not vendors_for_suggest:
        _log("no vendors selected; skipping electrode suggestions")
        _models = []
    else:
        try:
            from rosa_core.electrode_models import load_electrode_library
            _library = load_electrode_library()
            _models = list(_library.get("models") or [])
        except Exception as exc:
            _log(f"electrode library load failed ({exc}); no suggestions emitted")
            _models = []
    # Strategy-aware library filter: when the user picks a specific
    # pitch strategy (e.g. "Dixi AM (3.5 mm)"), suggestions should
    # come only from that strategy's library family — vendor + pitch
    # set jointly. The vendor-only filter inside the classifiers is
    # too loose to distinguish DIXI-AM from DIXI-MM (both pass "Dixi"
    # prefix but ride different pitch families).
    _models_filtered = filter_models_for_strategy(
        _models, pitch_strategy if _models else None,
    )
    if _models and not _models_filtered:
        _log(
            f"library filter for strategy '{pitch_strategy}' "
            f"eliminated all models; keeping vendor-prefix subset "
            f"of {len(_models)} for fallback"
        )
        _models_filtered = _models  # safety: don't strand suggestions
    if _models_filtered:
        n_suggested = 0
        for rec in anchored:
            intra = float(rec.get("intracranial_length_mm") or 0.0)
            if intra < 5.0:
                continue
            # Unified picker (`classify_electrode_model`): preferred
            # path is PaCER template-correlation against the canonical-
            # resampled CT volume; falls back to walker-signature joint
            # scoring (when CT path returns no candidate), then to
            # length-only with dura tolerance. Same picker is called
            # from Manual Fit / Guided Fit / Contacts & Trajectory View.
            pitch_obs = float(rec.get("original_median_pitch_mm") or 0.0)
            n_obs = int(rec.get("n_inliers") or 0)
            span_obs = float(rec.get("contact_span_mm") or 0.0)
            start_ras = rec.get("skull_entry_ras", rec.get("start_ras"))
            end_ras = rec.get("end_ras")
            walker_sig = (
                (n_obs, pitch_obs, span_obs)
                if (n_obs > 0 and pitch_obs > 0.0) else None
            )
            best = classify_electrode_model(
                start_ras=start_ras, end_ras=end_ras,
                models=_models_filtered,
                vendors=vendors_for_suggest,
                ct_volume_kji=ct_arr_kji,
                ras_to_ijk_mat=ras_to_ijk_mat,
                walker_signature=walker_sig,
                intracranial_length_mm=intra,
            )
            if best is None:
                continue
            rec["suggested_model_id"] = str(best["model_id"])
            rec["suggested_model_method"] = str(best.get("method") or "")
            rec["suggested_model_score"] = float(best.get("score") or 0.0)
            # Method-specific diagnostics — kept on rec for QC + logged.
            if best.get("method") == "pacer_template":
                rec["suggested_tip_arc_mm"] = float(best.get("tip_arc_mm") or 0.0)
                rec["suggested_coverage"] = float(best.get("coverage") or 0.0)
                rec["suggested_runner_up_id"] = str(best.get("runner_up_id") or "")
                rec["suggested_margin"] = float(best.get("margin") or 0.0)
                # PaCER also returns expected per-contact RAS positions —
                # downstream contact-generation can reuse these without
                # recomputing from scratch.
                rec["suggested_contacts_ras"] = list(best.get("contacts_ras") or [])
            elif best.get("method") == "walker_signature":
                rec["suggested_model_length_mm"] = float(best.get("model_total_mm") or 0.0)
                rec["suggested_model_gap_mm"] = float(best.get("length_err_mm") or 0.0)
                rec["suggested_model_pitch_err_mm"] = float(best.get("pitch_err_mm") or 0.0)
                rec["suggested_model_count_err"] = int(best.get("count_err") or 0)
                rec["suggested_model_span_err_mm"] = float(best.get("span_err_mm") or 0.0)
            elif best.get("method") == "shortest_covering":
                rec["suggested_model_length_mm"] = float(best.get("model_length_mm") or 0.0)
                rec["suggested_model_gap_mm"] = float(best.get("gap_mm") or 0.0)
            n_suggested += 1
        _log(
            f"suggested electrodes: {n_suggested} trajectories "
            f"(vendors={'+'.join(vendors_for_suggest)})"
        )

    # Confidence score (v1). Each survivor gets a continuous physical-
    # evidence score in [0, 1] plus a coarse confidence label.
    # ``confidence`` is the numeric score (canonical engine schema
    # expects a float there); ``confidence_label`` carries the band.
    for rec in anchored:
        score, label, components = _compute_trajectory_score(rec)
        rec["confidence"] = score
        rec["confidence_label"] = label
        rec["score_components"] = components

    # Convert to JSON-safe dicts (tuples of floats).
    trajectories: list[dict[str, Any]] = []
    for rec in anchored:
        out = dict(rec)
        out["start_ras"] = [float(x) for x in rec["start_ras"]]
        out["end_ras"] = [float(x) for x in rec["end_ras"]]
        if "skull_entry_ras" in rec:
            out["skull_entry_ras"] = [float(x) for x in rec["skull_entry_ras"]]
        trajectories.append(out)

    # Per-trajectory fingerprint — compact trace makes it easy to diff
    # Slicer-run results against a CLI run to spot which specific
    # trajectory disappeared / shifted when subject-level totals don't match.
    try:
        _log(f"trajectory summary ({len(trajectories)} kept):")
        for _i, _t in enumerate(trajectories):
            _se = _t.get("skull_entry_ras") or _t.get("start_ras") or [0.0, 0.0, 0.0]
            _en = _t.get("end_ras") or [0.0, 0.0, 0.0]
            _bolt = str(_t.get("bolt_source") or "?")
            _n = int(_t.get("n_inliers") or 0)
            _sp = float(_t.get("contact_span_mm") or 0.0)
            _sc = float(_t.get("confidence") or 0.0)
            _conf = str(_t.get("confidence_label") or "?")
            _log(
                f"  [{_i:02d}] bolt={_bolt} n={_n} span={_sp:.1f}mm "
                f"score={_sc:.2f}({_conf}) "
                f"skull_entry=({_se[0]:+.1f},{_se[1]:+.1f},{_se[2]:+.1f}) "
                f"deep_tip=({_en[0]:+.1f},{_en[1]:+.1f},{_en[2]:+.1f})"
            )
    except Exception:
        pass

    if return_features:
        features = {
            "log_sigma1": log1,
            "frangi_sigma1": frangi_s1,
            "head_distance": dist_arr,
            "intracranial": intracranial.astype(np.uint8),
            "hull": hull.astype(np.uint8),
            "bolt_mask": bolt_mask,
            # IJK->RAS matrix for the grid the feature arrays live on.
            # Differs from the input volume's matrix when canonical-1mm
            # resampling fired (raw sub-mm input). Slicer must use this
            # to position the feature volumes correctly in the scene.
            "ijk_to_ras_mat": np.asarray(ijk_to_ras_mat, dtype=float),
        }
        return trajectories, features
    return trajectories


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


