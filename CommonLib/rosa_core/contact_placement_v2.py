"""Backward-compatibility shim for ``contact_placement_v2``.

All algorithmic content has moved to the ``rosa_core.contact_placement``
package as part of the 2026-05-09 staged-pipeline refactor (see
``handoff_v3_production_lift_2026-05-09.md``). This file is a thin re-export
layer kept so existing callers keep working through the migration:

* ``rosa_core.unified_detect`` (will be deleted in Session 4)
* ``rosa_core.emission_qc`` (will migrate in Session 4)
* ``tests/rosa_core/test_contact_placement_v2.py`` (will be migrated /
  superseded in Session 4)
* External notebook code that has been updated to use the new package
  directly is unaffected.

To migrate caller code, change:

    from rosa_core.contact_placement_v2 import (
        _snap_centerline_to_centroid, _extend_centerline_tail, ...
    )

to:

    from rosa_core.contact_placement import (
        snap_centerline_to_centroid, extend_centerline_tail, ...
    )

Note the leading-underscore drop — the helpers are public surface in the
new package.

The remaining live function is ``place_contacts_for_seed_v2`` itself, which
``unified_detect`` still uses; once unified_detect is deleted, this whole
file goes too.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .contact_placement import (
    CC_HU_THRESHOLD,
    CC_ROI_HALF_MM,
    DEGENERATE_CONTACT_ZONE_MM,
    MAX_SLOT_CC_VOLUME_P90_MM3,
    MIN_CORR_FOR_REAL_SHANK,
    MIN_SLOT_HU_MEAN,
    SNAP_LOG_THRESHOLD,
    SNAP_RADIUS_MM,
    SNAP_SMOOTH_WINDOW,
    SNAP_STEP_MM,
    WALK_DISK_RADIUS_MM,
    WALK_FIRST_CONTACT_MIN_MM,
    WALK_HU_MIN,
    WALK_N_ANGLES,
    WALK_N_RADII,
    WALK_STEP_MM,
    WALK_TIP_PAD_MM,
    extend_centerline_tail as _extend_centerline_tail,
    polyline_at_arc as _polyline_pos_at_arc,
    polyline_pos_tan as _polyline_pos_tan,
    polyline_segments as _polyline_segments,
    project_to_polyline_arc as _project_to_polyline_arc,
    slot_cc_volume_mm3 as _slot_cc_volume_mm3,
    snap_centerline_to_centroid as _snap_centerline_to_centroid,
    straight_centerline as _straight_centerline,
)
from .matched_filter import (
    SIGMA_CONTACT_MM_DEFAULT,
    MatchedFilterResult,
    matched_filter_pick,
)


# Underscore-prefixed alias for the legacy "_ortho_uv" name.
def _ortho_uv(tangent: np.ndarray):
    """Alias for the new public ``ortho_uv`` (kept for legacy import paths)."""
    from .contact_placement.polyline import ortho_uv
    return ortho_uv(tangent)


# ---------------------------------------------------------------------
# Result dataclass — unchanged
# ---------------------------------------------------------------------


@dataclass
class PlacementV2Result:
    """Result of ``place_contacts_for_seed_v2``."""

    success: bool
    model_id: str | None
    placed_ras: list[list[float]]
    centerline_ras: list[list[float]] | None
    corr_score: float
    bolt_end_arc_mm: float
    bolt_source: str   # "metal" | "synthesized" | "bolt_less" | "none"
    n_placed: int
    rejected_reason: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Live function — kept until unified_detect is deleted in Session 4
# ---------------------------------------------------------------------


def place_contacts_for_seed_v2(
    seed_start_ras,
    seed_end_ras,
    *,
    features: dict,
    library_models: Sequence[dict],
    sigma_contact_mm: float = SIGMA_CONTACT_MM_DEFAULT,
    min_corr_for_real_shank: float = MIN_CORR_FOR_REAL_SHANK,
    min_slot_hu_mean: float | None = None,
    max_slot_cc_volume_p90_mm3: float | None = None,
    cc_hu_threshold: float = CC_HU_THRESHOLD,
    cc_roi_half_mm: float = CC_ROI_HALF_MM,
    add_valley_anti_template: bool = False,
    valley_anti_alpha: float = 1.0,
    walk_step_mm: float = WALK_STEP_MM,
    walk_disk_radius_mm: float = WALK_DISK_RADIUS_MM,
    first_contact_min_mm: float = WALK_FIRST_CONTACT_MIN_MM,
    walk_tip_pad_mm: float = WALK_TIP_PAD_MM,
    walk_hu_min: float = WALK_HU_MIN,
) -> PlacementV2Result:
    """Place contacts on a seed trajectory using the matched-filter pipeline.

    See module docstring — this function is being phased out in favor of
    ``rosa_core.contact_placement.place_seed`` (the staged equivalent).
    Kept identical to the pre-refactor implementation so unified_detect's
    behavior is unchanged through the migration.
    """
    from .contact_placement import (
        estimate_bolt_end_from_metal_mass,
        sample_disk_along_polyline,
    )

    s = np.asarray(seed_start_ras, dtype=float)
    e = np.asarray(seed_end_ras, dtype=float)
    seed_len = float(np.linalg.norm(e - s))
    if seed_len < 1e-3:
        return PlacementV2Result(
            success=False, model_id=None, placed_ras=[], centerline_ras=None,
            corr_score=0.0, bolt_end_arc_mm=0.0, bolt_source="none",
            n_placed=0, rejected_reason="seed_zero_length",
        )

    ct_arr = features["ct_arr_kji"]
    log_arr = features.get("log")
    r2i = np.asarray(features["ras_to_ijk_mat"], dtype=float)

    def _try_bolt_end(seed_start, seed_end):
        try:
            be = estimate_bolt_end_from_metal_mass(
                seed_start, seed_end, features=features,
                library_models=library_models,
            )
        except Exception:
            return None, None
        be_arc = be.get("bolt_end_arc_mm")
        cp = be.get("centerline")
        if be_arc is None or cp is None:
            return None, None
        cp_total = float(np.linalg.norm(np.diff(cp, axis=0), axis=1).sum())
        if cp_total - float(be_arc) < DEGENERATE_CONTACT_ZONE_MM:
            return None, None
        return be_arc, cp

    bolt_end_straight, cp = _try_bolt_end(s, e)
    if bolt_end_straight is None:
        be_rev, cp_rev = _try_bolt_end(e, s)
        if be_rev is not None:
            s, e = e, s
            seed_len = float(np.linalg.norm(e - s))
            bolt_end_straight, cp = be_rev, cp_rev

    use_fallback = bolt_end_straight is None or cp is None

    if use_fallback:
        centerline = _straight_centerline(s, e)
        bolt_source = "bolt_less"
        bolt_end_cl_arc = 0.0
        max_extend = 0.0
    else:
        centerline_poly = np.asarray(cp, dtype=float)
        if log_arr is not None:
            centerline = _snap_centerline_to_centroid(centerline_poly, log_arr, r2i)
        else:
            centerline = centerline_poly
        u_str = (e - s) / seed_len
        bolt_pt = s + float(bolt_end_straight) * u_str
        bolt_end_cl_arc = _project_to_polyline_arc(centerline, bolt_pt)
        bolt_source = "metal"
        max_extend = walk_tip_pad_mm

    cl_max = float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum())

    centerline_for_walker = _extend_centerline_tail(centerline, max_extend)
    arcs, max_b, _total_b = sample_disk_along_polyline(
        ct_arr, r2i, centerline_for_walker,
        polarity="positive", step_mm=walk_step_mm,
        disk_radius_mm=walk_disk_radius_mm,
        n_radii=WALK_N_RADII, n_angles=WALK_N_ANGLES,
        total_threshold=walk_hu_min,
    )
    signal = max_b

    match: MatchedFilterResult = matched_filter_pick(
        arcs, signal, library_models,
        bolt_end_arc=bolt_end_cl_arc,
        first_contact_min_mm=first_contact_min_mm,
        profile_end_arc=cl_max,
        max_extend_tip_mm=max_extend,
        sigma_contact_mm=sigma_contact_mm,
        add_valley_anti_template=add_valley_anti_template,
        valley_anti_alpha=valley_anti_alpha,
    )

    if match.best_model_id is None:
        return PlacementV2Result(
            success=False, model_id=None, placed_ras=[], centerline_ras=centerline.tolist(),
            corr_score=0.0, bolt_end_arc_mm=bolt_end_cl_arc, bolt_source=bolt_source,
            n_placed=0, rejected_reason="no_model",
            diagnostics={"cl_max_mm": cl_max, "in_zone_p75": float(np.percentile(signal, 75))},
        )

    if match.corr < float(min_corr_for_real_shank):
        return PlacementV2Result(
            success=False, model_id=match.best_model_id, placed_ras=[],
            centerline_ras=centerline.tolist(),
            corr_score=match.corr, bolt_end_arc_mm=bolt_end_cl_arc,
            bolt_source=bolt_source, n_placed=match.n_slots,
            rejected_reason=f"corr_below_threshold({match.corr:.3f}<{min_corr_for_real_shank:.3f})",
        )

    placed = [_polyline_pos_at_arc(centerline, float(a)).tolist()
              for a in match.slot_arcs]

    if min_slot_hu_mean is not None:
        from .volume_sampling import sample_trilinear_at_ras
        slot_hus = []
        for pt in placed:
            hu = float(sample_trilinear_at_ras(ct_arr, r2i, np.asarray(pt, dtype=float)))
            if np.isfinite(hu): slot_hus.append(hu)
        slot_hu_mean = float(np.mean(slot_hus)) if slot_hus else 0.0
        if slot_hu_mean < float(min_slot_hu_mean):
            return PlacementV2Result(
                success=False, model_id=match.best_model_id, placed_ras=[],
                centerline_ras=centerline.tolist(),
                corr_score=match.corr, bolt_end_arc_mm=bolt_end_cl_arc,
                bolt_source=bolt_source, n_placed=match.n_slots,
                rejected_reason=(
                    f"slot_hu_mean_below_threshold"
                    f"({slot_hu_mean:.0f}<{min_slot_hu_mean:.0f})"
                ),
                diagnostics={"slot_hu_mean": slot_hu_mean},
            )

    if max_slot_cc_volume_p90_mm3 is not None and placed:
        spacing_xyz = features["img"].GetSpacing()
        slot_volumes = [
            _slot_cc_volume_mm3(
                ct_arr, r2i, np.asarray(pt, dtype=float), spacing_xyz,
                hu_threshold=float(cc_hu_threshold),
                roi_half_mm=float(cc_roi_half_mm),
            )
            for pt in placed
        ]
        slot_cc_p90 = float(np.percentile(np.asarray(slot_volumes, dtype=float), 90))
        if slot_cc_p90 > float(max_slot_cc_volume_p90_mm3):
            return PlacementV2Result(
                success=False, model_id=match.best_model_id, placed_ras=[],
                centerline_ras=centerline.tolist(),
                corr_score=match.corr, bolt_end_arc_mm=bolt_end_cl_arc,
                bolt_source=bolt_source, n_placed=match.n_slots,
                rejected_reason=(
                    f"slot_cc_volume_p90_above_threshold"
                    f"({slot_cc_p90:.1f}>{max_slot_cc_volume_p90_mm3:.1f})"
                ),
                diagnostics={"slot_cc_volume_p90_mm3": slot_cc_p90},
            )

    return PlacementV2Result(
        success=True,
        model_id=match.best_model_id,
        placed_ras=placed,
        centerline_ras=centerline.tolist(),
        corr_score=match.corr,
        bolt_end_arc_mm=bolt_end_cl_arc,
        bolt_source=bolt_source,
        n_placed=match.n_slots,
        rejected_reason="",
        diagnostics={
            "cl_max_mm": cl_max,
            "tip_arc_mm": float(match.tip_arc) if match.tip_arc is not None else float("nan"),
            "n_covered": match.n_covered,
        },
    )


__all__ = [
    # Constants — re-exported from new package.
    "CC_HU_THRESHOLD",
    "CC_ROI_HALF_MM",
    "DEGENERATE_CONTACT_ZONE_MM",
    "MAX_SLOT_CC_VOLUME_P90_MM3",
    "MIN_CORR_FOR_REAL_SHANK",
    "MIN_SLOT_HU_MEAN",
    "SNAP_LOG_THRESHOLD",
    "SNAP_RADIUS_MM",
    "SNAP_SMOOTH_WINDOW",
    "SNAP_STEP_MM",
    "WALK_DISK_RADIUS_MM",
    "WALK_FIRST_CONTACT_MIN_MM",
    "WALK_HU_MIN",
    "WALK_N_ANGLES",
    "WALK_N_RADII",
    "WALK_STEP_MM",
    "WALK_TIP_PAD_MM",
    # Helpers (legacy underscore-prefixed names).
    "_extend_centerline_tail",
    "_ortho_uv",
    "_polyline_pos_at_arc",
    "_polyline_pos_tan",
    "_polyline_segments",
    "_project_to_polyline_arc",
    "_slot_cc_volume_mm3",
    "_snap_centerline_to_centroid",
    "_straight_centerline",
    # Live API.
    "PlacementV2Result",
    "place_contacts_for_seed_v2",
]
