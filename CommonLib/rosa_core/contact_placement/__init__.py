"""rosa_core.contact_placement — staged contact placement.

Public surface:

* New staged pipeline (``place_seed``, ``PlacementCtx``, stage functions A-F)
  lifted from ``slicer-rosa-helper/notebooks/v1_seeds_v2_placement_qc.ipynb``.
* Legacy production helpers (``place_contacts_for_axis``,
  ``place_contacts_for_trajectories``, ``estimate_bolt_end_from_metal_mass``,
  ``sample_disk_along_polyline``, etc.) re-exported from the renamed
  ``contact_placement_legacy`` module so existing Slicer Auto-Fit, CLI, and
  test imports keep working without code changes.

Legacy re-exports will be removed in Session 4 once Slicer Auto-Fit migrates
to the staged pipeline via ``rosa_core.placement_modes.place_seeg``.
"""
from __future__ import annotations

# Legacy re-exports (production Auto-Fit relies on these).
from ..contact_placement_legacy import (
    ContactPlacementBatch,
    ContactPlacementConfig,
    ContactPlacementResult,
    assign_axis_owners,
    entry_arc_from_metal_mass,
    estimate_bolt_end_from_metal_mass,
    median_library_pitch_mm,
    place_contacts_for_axis,
    place_contacts_for_trajectories,
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
    pick_extent_aware,
    pick_matched_filter,
)
from .stage_e_place import place_at_match
from .stage_f_score import (
    score_cc_overlap,
    score_compound,
    score_simple,
    tube_like_frac,
)
from .compose import place_seed, place_v3
from .two_pass import run_two_pass
from .postpass_fft import apply_subject_fft_normalization

__all__ = [
    # Legacy production API.
    "ContactPlacementBatch",
    "ContactPlacementConfig",
    "ContactPlacementResult",
    "assign_axis_owners",
    "entry_arc_from_metal_mass",
    "estimate_bolt_end_from_metal_mass",
    "median_library_pitch_mm",
    "place_contacts_for_axis",
    "place_contacts_for_trajectories",
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
    "pick_extent_aware",
    "pick_matched_filter",
    "place_at_match",
    "score_cc_overlap",
    "score_compound",
    "score_simple",
    "tube_like_frac",
    # Composers.
    "place_seed",
    "place_v3",
    "run_two_pass",
    "apply_subject_fft_normalization",
]


# Fallback: any private legacy symbol the package doesn't explicitly export
# is forwarded to ``contact_placement_legacy``. This keeps the existing
# ``tests/rosa_core/test_contact_placement.py`` working without enumerating
# every private helper. Removed in Session 4 with ``contact_placement_legacy``.
def __getattr__(name: str):
    from .. import contact_placement_legacy
    if hasattr(contact_placement_legacy, name):
        value = getattr(contact_placement_legacy, name)
        globals()[name] = value  # cache so future lookups skip __getattr__
        return value
    raise AttributeError(f"module 'rosa_core.contact_placement' has no attribute {name!r}")
