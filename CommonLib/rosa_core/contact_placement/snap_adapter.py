"""Bridge the fit-rosa snap-flow into a scored-ready ``PlacementCtx``.

:func:`snap_chain_to_ctx` takes ONE ``seeded_fit`` chain (the on-axis
``snap_via_signal_walk`` + ``arbitrate_shared_peaks`` output) plus its
``TrajectoryLandmarks`` and produces a ``PlacementCtx`` carrying every field
Stage-F scoring and the public ``PlacedTrajectory`` contract need — running the
SAME ``tube-max profile -> mf_anchor -> pick_electrode_model(chain=...) ->
place_contacts_from_tip`` sequence the headless ``fit-rosa`` CLI runs
(``cli/rosa_agent/commands/fit_rosa.py``). So the Slicer / ``place_seeg``
snap-flow and the CLI place contacts identically (CLI<->Slicer parity), and the
covering-floor finally activates in the staged pipeline because the pick now
receives a real ``chain`` instead of ``chain=None``.

This is the irreducible new component of the contact-placement consolidation:
the snap produces ``FittedChain`` + ``TrajectoryLandmarks``; the staged scorers
consume a ``PlacementCtx``. This module is that adapter. It is UNWIRED in this
PR — ``placement_modes`` is routed onto it behind a flag in a follow-up.

Placement note: the contacts come from ``place_contacts_from_tip`` (offsets from
the fitted tip along the axis), NOT ``stage_e.place_at_match`` — ``polyline_at_arc``
clamps to ``[0, len]``, so projecting the matcher's slot arcs onto the 2-point
straight centerline would pile bolt-zone / tip-extension slots at the endpoints.
``place_contacts_from_tip`` extrapolates correctly and is the fit-rosa-validated
placement.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..centerline import orthonormal_basis_for_axis, unit
from ..matched_filter import matched_filter_pick
from ..model_pick import pick_electrode_model
from ..volume_sampling import sample_trilinear_batch
from .context import PlacementCtx
from .stage_d_pick import _result_for

# Tube-max profile geometry — identical to fit-rosa's ``_sample_log_neg_tube``
# so the matched_filter / no_metal_rerank thresholds behave as validated in
# notebooks/seeded_fit. (Temporarily duplicated from
# ``cli/rosa_agent/commands/fit_rosa.py``; the CLI dedups onto this helper in a
# follow-up once the adapter is proven.)
_MF_TUBE_RADIUS_MM = 4.0
_MF_STEP_MM = 0.3
_MF_PAD_MM = 2.0

# Tip-search slack for the matcher — fit-rosa's validated values (symmetric
# ±~2 mm absorbs the snap-tip vs physical-tip offset).
_MAX_EXTEND_TIP_MM = 2.0
_MAX_TIP_SHORT_MM = 2.5


def _sample_log_neg_tube(log_neg, ras_to_ijk, entry_ras, axis_unit, length_mm):
    """Tube-max LoG-neg profile along ``(entry_ras, axis_unit)`` from ``-pad``
    to ``length + pad`` in ``_MF_STEP_MM`` steps. Same shape as the snap-side
    profile, so the matched-filter coverage / no_metal_rerank thresholds behave
    as validated. Returns ``(arcs, sig)``."""
    perp1, perp2 = orthonormal_basis_for_axis(axis_unit)
    n = int(2 * _MF_TUBE_RADIUS_MM / 0.5) + 1
    ua = np.linspace(-_MF_TUBE_RADIUS_MM, _MF_TUBE_RADIUS_MM, n)
    uu, vv = np.meshgrid(ua, ua, indexing="ij")
    disk = (uu ** 2 + vv ** 2) <= _MF_TUBE_RADIUS_MM ** 2
    du, dv = uu[disk], vv[disk]

    arcs = np.arange(-_MF_PAD_MM, length_mm + _MF_PAD_MM, _MF_STEP_MM)
    sig = np.zeros(len(arcs), dtype=float)
    log_neg_f32 = np.asarray(log_neg).astype("float32", copy=False)
    for i, t in enumerate(arcs):
        c = entry_ras + t * axis_unit
        pts = c[None] + du[:, None] * perp1[None] + dv[:, None] * perp2[None]
        v = sample_trilinear_batch(log_neg_f32, ras_to_ijk, pts)
        finite = v[np.isfinite(v)] if v.size else v
        sig[i] = float(np.max(finite)) if finite.size else 0.0
    return arcs, sig


def _intra_mask_anchor(intracranial_mask, ras_to_ijk, entry_ras, axis_unit,
                       length_mm):
    """Arc (mm) along ``(entry_ras, axis_unit)`` where the trajectory first
    crosses into the intracranial mask. 0.0 if already in-brain at entry (common
    — the snap's ``entry_ras`` is the most-proximal detected contact, which often
    sits in cortex)."""
    mask = np.asarray(intracranial_mask).astype("float32", copy=False)
    ts = np.arange(0.0, length_mm + 1.0, 0.5)
    inb = np.zeros(len(ts), dtype=float)
    for i, t in enumerate(ts):
        v = sample_trilinear_batch(mask, ras_to_ijk, (entry_ras + t * axis_unit)[None])
        inb[i] = float(v[0]) if v.size and np.isfinite(v[0]) else 0.0
    inside = np.where(inb > 0.5)[0]
    return float(ts[inside[0]]) if inside.size else 0.0


def snap_chain_to_ctx(
    chain: dict,
    *,
    features: dict,
    library_models: list[dict],
    landmarks: Any = None,
    bolts: list[dict] | None = None,
    forced_model_id: str | None = None,
    seeder_label: str = "",
    seeder_confidence: float = 0.0,
    seeder_model: str | None = None,
) -> PlacementCtx:
    """Build a scored-ready ``PlacementCtx`` from one snapped chain.

    Args:
        chain: a ``seeded_fit`` chain dict — ``{axis, entry_ras, tip_ras,
            kept_pts, ...}`` (``FittedChain.as_dict()``). ``kept_pts``/
            ``entry_ras``/``axis`` drive the covering-floor in the pick.
        features: ``compute_features`` dict — needs ``log`` (raw LoG; the tube
            samples ``clip(-log, 0)``), ``ras_to_ijk_mat``, ``intracranial_mask``.
        library_models: strategy-filtered candidate models (the same set the
            staged pipeline passes as ``ctx.library_models``).
        landmarks: optional ``TrajectoryLandmarks`` — its ``bone_arc_mm`` joins
            the intracranial-mask entry in ``mf_anchor = min(mask, bone)``.
        bolts / seeder_*: threaded onto the ctx for the compound score.
        forced_model_id: when set (mode 4 vouched model), score and place exactly
            that model — the covering-floor is skipped (forced means forced).

    Returns a ``PlacementCtx`` with ``centerline`` (2-point straight axis),
    ``walk_arcs``/``walk_signal`` (tube-max profile), ``bolt_end_arc``
    (``mf_anchor``), ``bolt_source="metal"``, ``match`` (the picked model's
    ``MatchedFilterResult``), and ``placed_ras`` (contacts from the fitted tip).
    Ready for ``score_simple`` -> ``score_cc_overlap`` -> ``score_compound``.
    """
    axis_unit = unit(np.asarray(chain["axis"], dtype=float))
    entry_ras = np.asarray(chain["entry_ras"], dtype=float)
    tip_ras = np.asarray(chain["tip_ras"], dtype=float)
    mf_length = float(np.linalg.norm(tip_ras - entry_ras))

    ras_to_ijk = np.asarray(features["ras_to_ijk_mat"], dtype=float)
    log_neg = np.clip(-np.asarray(features["log"]), 0.0, None)
    mf_arcs, mf_sig = _sample_log_neg_tube(
        log_neg, ras_to_ijk, entry_ras, axis_unit, mf_length,
    )

    # Proximal matcher anchor = min(intracranial-mask entry, bone inner edge).
    # The CT brain mask under-segments at the skull base, so its entry can sit
    # deeper than the most-superficial real contacts; the bone landmark catches
    # the transition where the mask fails. Take whichever is more proximal.
    mask_anchor = _intra_mask_anchor(
        features["intracranial_mask"], ras_to_ijk, entry_ras, axis_unit, mf_length,
    )
    bone_anchor = getattr(landmarks, "bone_arc_mm", None) if landmarks is not None else None
    mf_anchor = (float(min(mask_anchor, bone_anchor))
                 if bone_anchor is not None else float(mask_anchor))

    models_dict = {str(m.get("id") or ""): m for m in library_models if isinstance(m, dict)}

    if forced_model_id is not None and forced_model_id in models_dict:
        # Vouched model: score exactly it (no picker, no covering-floor).
        predicted_id = forced_model_id
        mf_res = matched_filter_pick(
            mf_arcs, mf_sig, [models_dict[forced_model_id]],
            bolt_end_arc=mf_anchor, profile_end_arc=mf_length,
            max_extend_tip_mm=_MAX_EXTEND_TIP_MM, max_tip_short_mm=_MAX_TIP_SHORT_MM,
        )
    else:
        predicted_id, _branch, mf_res, _diag = pick_electrode_model(
            mf_arcs, mf_sig, library_models, models_dict,
            bolt_end_arc=mf_anchor, profile_end_arc=mf_length, chain=chain,
            max_extend_tip_mm=_MAX_EXTEND_TIP_MM, max_tip_short_mm=_MAX_TIP_SHORT_MM,
        )

    match = _result_for(mf_res, predicted_id)

    placed_ras: list[np.ndarray] = []
    if predicted_id is not None and predicted_id in models_dict:
        from ..electrode_classifier import place_contacts_from_tip
        placed = place_contacts_from_tip(tip_ras, axis_unit, models_dict[predicted_id])
        placed_ras = [np.asarray(c, dtype=float) for c in np.asarray(placed)]

    return PlacementCtx(
        seed_start=entry_ras,
        seed_end=tip_ras,
        features=features,
        library_models=library_models,
        bolts=bolts,
        seeder_label=str(seeder_label or ""),
        seeder_confidence=float(seeder_confidence),
        seeder_model=seeder_model,
        centerline=np.vstack([entry_ras, tip_ras]),
        bolt_end_arc=mf_anchor,
        bolt_source="metal",
        walk_arcs=mf_arcs,
        walk_signal=mf_sig,
        signal_kind="neg_log_max",
        match=match,
        placed_ras=placed_ras,
    )


__all__ = ["snap_chain_to_ctx"]
