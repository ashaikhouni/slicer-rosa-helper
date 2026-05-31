"""rosa_core.contact_placement — staged contact placement.

Public surface:

* Staged pipeline (``place_seed``, ``PlacementCtx``, stage functions A-F) — the
  canonical placement path used by ``placement_modes.place_seeg``.
* Bolt-end / metal-mass landmark estimators (``estimate_bolt_end_from_metal_mass``,
  ``entry_arc_from_metal_mass``, ``refine_axis_via_centroid``,
  ``sample_disk_along_polyline``, ``median_library_pitch_mm``) in
  ``contact_placement.bolt_end`` — used by ``stage_a_anchor``.

(The pre-staged v2 ``contact_placement_legacy`` pipeline was retired 2026-05-30;
its only live survivors are the bolt-end estimators above.)
"""
from __future__ import annotations

# Bolt-end / metal-mass landmark estimators.
from .bolt_end import (
    entry_arc_from_metal_mass,
    estimate_bolt_end_from_metal_mass,
    median_library_pitch_mm,
    refine_axis_via_centroid,
    sample_disk_along_polyline,
)

# New staged pipeline.
from .constants import (
    BOLT_ONLY_PENALTY_MAX,
    BOLT_ONLY_PENALTY_THRESHOLD,
    CC_HU_THRESHOLD,
    CC_OVERLAP_MAX_ARC_PAST_BOLT_MM,
    CC_OVERLAP_MAX_PERP_MM,
    CC_OVERLAP_PERP_SCALE_MM,
    CC_ROI_HALF_MM,
    COMPOUND_BANDS,
    COMPOUND_WEIGHTS,
    DEGENERATE_CONTACT_ZONE_MM,
    LOG_TOTAL_THRESHOLD,
    MAX_SLOT_CC_VOLUME_P90_MM3,
    MIN_CORR_FOR_REAL_SHANK,
    MIN_SLOT_HU_MEAN,
    SEEDER_LABEL_TO_SCORE,
    SNAP_LOG_THRESHOLD,
    SNAP_RADIUS_MM,
    SNAP_SMOOTH_WINDOW,
    SNAP_STEP_MM,
    WALK_AGGREGATOR,
    WALK_DISK_RADIUS_MM,
    WALK_FIRST_CONTACT_MIN_MM,
    WALK_HU_MIN,
    WALK_N_ANGLES,
    WALK_N_RADII,
    WALK_STEP_MM,
    WALK_TIP_PAD_MM,
)
from .context import PlacementCtx
from .polyline import (
    extend_centerline_tail,
    min_dist_pts_to_polyline,
    polyline_at_arc,
    polyline_pos_at_arc,
    polyline_pos_tan,
    polyline_segments,
    project_to_polyline,
    project_to_polyline_arc,
    straight_centerline,
)
from .snap import snap_centerline_owned, snap_centerline_to_centroid
from .cc_volume import slot_cc_volume_mm3
from .stage_a_anchor import anchor_bolt_less, anchor_metal, stage_anchor
from .stage_b_refine import refine_log_snap, refine_noop
from .stage_c_sample import (
    aggregate_disk,
    sample_hu_max,
    sample_neg_log_max,
    walk_centerline,
)
from .stage_d_pick import (
    per_model_corrs,
    pick_model,
)
from .stage_e_place import place_at_match
from .stage_f_score import (
    score_cc_overlap,
    score_compound,
    score_simple,
    tube_like_frac,
)
from .compose import place_seed, place_v3
from .postpass_fft import apply_subject_fft_normalization
from .snap_adapter import snap_chain_to_ctx, snap_fit_to_ctxs

__all__ = [
    # Bolt-end / metal-mass landmark estimation (rosa_core.contact_placement.bolt_end).
    "entry_arc_from_metal_mass",
    "estimate_bolt_end_from_metal_mass",
    "median_library_pitch_mm",
    "refine_axis_via_centroid",
    "sample_disk_along_polyline",
    # Constants.
    "BOLT_ONLY_PENALTY_MAX",
    "BOLT_ONLY_PENALTY_THRESHOLD",
    "CC_HU_THRESHOLD",
    "CC_OVERLAP_MAX_ARC_PAST_BOLT_MM",
    "CC_OVERLAP_MAX_PERP_MM",
    "CC_OVERLAP_PERP_SCALE_MM",
    "CC_ROI_HALF_MM",
    "COMPOUND_BANDS",
    "COMPOUND_WEIGHTS",
    "DEGENERATE_CONTACT_ZONE_MM",
    "LOG_TOTAL_THRESHOLD",
    "MAX_SLOT_CC_VOLUME_P90_MM3",
    "MIN_CORR_FOR_REAL_SHANK",
    "MIN_SLOT_HU_MEAN",
    "SEEDER_LABEL_TO_SCORE",
    "SNAP_LOG_THRESHOLD",
    "SNAP_RADIUS_MM",
    "SNAP_SMOOTH_WINDOW",
    "SNAP_STEP_MM",
    "WALK_AGGREGATOR",
    "WALK_DISK_RADIUS_MM",
    "WALK_FIRST_CONTACT_MIN_MM",
    "WALK_HU_MIN",
    "WALK_N_ANGLES",
    "WALK_N_RADII",
    "WALK_STEP_MM",
    "WALK_TIP_PAD_MM",
    # Context.
    "PlacementCtx",
    # Geometry helpers.
    "extend_centerline_tail",
    "min_dist_pts_to_polyline",
    "polyline_at_arc",
    "polyline_pos_at_arc",
    "polyline_pos_tan",
    "polyline_segments",
    "project_to_polyline",
    "project_to_polyline_arc",
    "straight_centerline",
    "snap_centerline_owned",
    "snap_centerline_to_centroid",
    "slot_cc_volume_mm3",
    # Stages.
    "anchor_bolt_less",
    "anchor_metal",
    "stage_anchor",
    "refine_log_snap",
    "refine_noop",
    "aggregate_disk",
    "sample_hu_max",
    "sample_neg_log_max",
    "walk_centerline",
    "per_model_corrs",
    "pick_model",
    "place_at_match",
    "score_cc_overlap",
    "score_compound",
    "score_simple",
    "tube_like_frac",
    # Composers.
    "place_seed",
    "place_v3",
    "apply_subject_fft_normalization",
    "snap_chain_to_ctx",
    "snap_fit_to_ctxs",
]
