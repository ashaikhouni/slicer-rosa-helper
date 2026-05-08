"""Unified trajectory detection + contact placement (M8 composition).

End-to-end unseeded pipeline composing existing primitives:

  Stage A (caller-supplied): canonical-grid features dict from
      ``rosa_detect.guided_fit_engine.compute_features``.

  Stage B (this module): permissive candidate generation +
      bolt-anchor refinement + synth fallback + bolt-anchor rejection.
        - ``f1.run_stage1(frangi_arr=None)`` — keeps low-amplitude
          shanks the post-anchor Frangi gate would otherwise drop
          (e.g. AMC135 / MTG-amygdala, hetero-mid-M).
        - ``extract_bolt_candidates`` in **HU-mode** on the unified
          metal-evidence volume (matches v1's call; the LoG-mode
          default misses bolts where saturated CT clipping suppresses
          LoG response — exactly the T4 / LCMN, T4 / RHH cases).
        - For each chain: ``anchor_trajectory_to_bolt`` in both
          orientations; if neither anchors and the chain looks like a
          genuine SEEG line, try ``_axis_to_skull_synth`` as a
          second-chance fallback.
        - Drop chains that neither anchored nor synthesized — that's
          the dominant FP filter (M5 → M2 inflates orphans 7 → 53).

  Stage C (this module): ``place_contacts_for_seed_v2`` on the
      refined seed; reject when matched-filter corr is below
      ``min_corr_for_real_shank``. Optional HU floor for cleanup.

  Stage D (this module): post-placement axis dedup at
      4 mm / 12° (matches the validated probe defaults).

Validated 2026-05-09 on 7-subject benchmark (5 AMC + T22 + T4):

  v1 production + FOV fix     79/82 / ~8 orphans
  M8 unified (this module)    80/82 /   6 orphans

See ``project_unified_pipeline_m8_2026-05-09.md`` for the full
analysis and the architectural ceiling (RHH + LPT only).

This module is **additive** — does not modify
``rosa_detect.contact_pitch_v1_fit`` or any v1 primitives. Callers
opt in by importing ``detect_and_place_unified`` instead of
``run_contact_pitch_v1_with_features``.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


# ---------------------------------------------------------------------
# Defaults — physical, not subject-tuned.
# ---------------------------------------------------------------------

DEDUP_PERP_MM = 4.0
DEDUP_ANGLE_DEG = 12.0
PRE_DEDUP_PERP_MM = 2.0     # tight pre-dedup so two real shanks <2mm apart
PRE_DEDUP_ANGLE_DEG = 6.0   # don't get merged before placement.

# Stage 1 candidate floor. v1's default is _LIBRARY_BOUNDS["min_contacts"]=5,
# calibrated when the validator was bolt-anchor + post-anchor Frangi-median.
# In the unified pipeline the validator is matched-filter NCC ≥ 0.35 +
# HU=1500 floor, so the blob-count gate is duplicate work — it costs
# AMC137/LPT (4-blob walker chain that _extend_deep_end completes to 5
# blobs spanning ~22 mm). 4 is the sweet spot:
#   MIN_BLOBS=5 → 80/82 / 6 orph
#   MIN_BLOBS=4 → 81/82 / 7 orph (recovers LPT, +3 orph on AMC91)
#   MIN_BLOBS=3 → 81/82 / 10 orph (no extra TPs, just over-permissive)
# See project_unified_pipeline_m9_2026-05-08.md.
MIN_BLOBS_PER_LINE_UNIFIED = 4


@dataclass
class UnifiedTrajectory:
    """One emitted trajectory + its placed contacts.

    ``corr_score`` is the matched-filter Pearson correlation against
    the winning library template (``rosa_core.matched_filter``).
    ``bolt_source`` matches the v2 placement convention:
    ``"metal"`` / ``"synthesized"`` / ``"bolt_less"`` / ``"none"``.

    ``anchored`` / ``synthed`` flag which stage-B refinement path the
    chain took (``anchored`` from ``anchor_trajectory_to_bolt``,
    ``synthed`` from ``_axis_to_skull_synth``). Useful for downstream
    confidence signaling.
    """
    start_ras: list[float]
    end_ras: list[float]
    model_id: str | None
    corr_score: float
    placed_ras: list[list[float]]
    centerline_ras: list[list[float]] | None
    bolt_source: str
    n_placed: int
    anchored: bool
    synthed: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _is_genuine_seeg_chain(chain, *, min_blobs, min_dist_max, max_pitch):
    """Mirrors v1's ``_is_genuine_seeg_line`` gate. Only allow synth
    fallback for strong stage1 chains so we don't synthesize axes from
    cross-shank / bone-feature chains.
    """
    n = int(chain.get("n_inliers", chain.get("n_blobs", 0)))
    if n < min_blobs:
        return False
    if float(chain.get("dist_max_mm", 0.0)) < min_dist_max:
        return False
    span_post = float(chain.get("contact_span_mm", chain.get("length_mm", 0.0)))
    span_pre = float(chain.get("original_span_mm", span_post))
    span_for_pitch = min(span_pre, span_post) if span_pre > 0 else span_post
    fallback_avg = span_for_pitch / (n - 1) if n > 1 else float("inf")
    median_pitch = float(chain.get("original_median_pitch_mm", fallback_avg))
    if median_pitch > max_pitch:
        return False
    return True


def _refine_axis_via_bolt(start_ras, end_ras, bolts, anchor_fn):
    """Try ``anchor_trajectory_to_bolt`` in both orientations and keep
    the one with more bolt-tube voxels. Returns
    (new_start, new_end, anchored). Original (start, end) on miss.
    """
    fwd = anchor_fn(start_ras, end_ras, bolts)
    bwd = anchor_fn(end_ras, start_ras, bolts)
    fwd_n = int(fwd[2].get("tube_n_vox", 0)) if fwd[2] is not None else 0
    bwd_n = int(bwd[2].get("tube_n_vox", 0)) if bwd[2] is not None else 0
    if bwd_n > fwd_n and bwd[0] is not None:
        return np.asarray(bwd[0], float), np.asarray(start_ras, float), True
    if fwd[0] is not None:
        return np.asarray(fwd[0], float), np.asarray(end_ras, float), True
    return np.asarray(start_ras, float), np.asarray(end_ras, float), False


def _synth_fallback(chain, dist_arr, ras_to_ijk_mat,
                     orient_fn, synth_fn,
                     min_blobs, min_dist_max, max_pitch):
    """v1's synth fallback: orient chain shallow→deep via head-distance,
    walk outward until the axis crosses the hull, return synth_tip as
    the new shallow end. Only fires for genuine SEEG chains.

    Returns (start, end) or (None, None).
    """
    if not _is_genuine_seeg_chain(chain,
                                    min_blobs=min_blobs,
                                    min_dist_max=min_dist_max,
                                    max_pitch=max_pitch):
        return None, None
    s0, e0 = orient_fn(
        np.asarray(chain["start_ras"], float),
        np.asarray(chain["end_ras"], float),
        dist_arr, ras_to_ijk_mat,
    )
    skull, tip = synth_fn(s0, e0, dist_arr, ras_to_ijk_mat)
    if skull is None:
        return None, None
    return np.asarray(tip, float), np.asarray(e0, float)


def _axis_dup(s1, e1, s2, e2, perp_mm, angle_deg):
    """Two axes are duplicates if their direction angle ≤ ``angle_deg``
    AND the lateral midpoint drift along the shared direction
    ≤ ``perp_mm``. Direction-free (uses ``|cos|``).
    """
    d1 = e1 - s1; L1 = float(np.linalg.norm(d1))
    d2 = e2 - s2; L2 = float(np.linalg.norm(d2))
    if L1 < 1e-6 or L2 < 1e-6:
        return False
    u1 = d1 / L1; u2 = d2 / L2
    cosang = float(np.clip(abs(u1 @ u2), 0.0, 1.0))
    ang = float(np.degrees(np.arccos(cosang)))
    if ang > angle_deg:
        return False
    mid = 0.5 * (s2 + e2)
    ap = mid - s1
    perp = float(np.linalg.norm(ap - (ap @ u1) * u1))
    return perp <= perp_mm


def detect_and_place_unified(
    features: dict,
    library_models: Sequence[dict],
    *,
    pitch_strategy_pitches_mm: Sequence[float] | None = None,
    min_corr: float | None = None,
    min_slot_hu_mean: float | None = None,
    max_slot_cc_volume_p90_mm3: float | None = None,
    dedup_perp_mm: float = DEDUP_PERP_MM,
    dedup_angle_deg: float = DEDUP_ANGLE_DEG,
    enable_synth_fallback: bool = True,
    min_blobs_per_line: int = MIN_BLOBS_PER_LINE_UNIFIED,
) -> list[UnifiedTrajectory]:
    """Unseeded trajectory detection + contact placement.

    Args:
        features: canonical-grid features dict from
            ``rosa_detect.guided_fit_engine.compute_features``. Must
            contain ``log`` (LoG σ=1), ``ct_arr_kji``, ``head_distance``
            (SDF), ``ijk_to_ras_mat``, ``ras_to_ijk_mat``.
        library_models: electrode library entries (output of
            ``rosa_core.electrode_classifier.filter_models_for_strategy``
            or the full library).
        pitch_strategy_pitches_mm: pitches passed to ``run_stage1``.
            ``None`` → derive from ``library_models`` (set of unique
            ``pitch_mm``); empty fallback to ``[3.5]``.
        min_corr: matched-filter corr threshold. ``None`` → default
            (``MIN_CORR_FOR_REAL_SHANK = 0.35``).
        min_slot_hu_mean: per-slot HU floor cleanup. ``None`` →
            default (``MIN_SLOT_HU_MEAN = 1500``). Set to
            ``False``-ish via ``0.0`` or pass an explicit small value
            to relax.
        max_slot_cc_volume_p90_mm3: per-slot CC volume cap. ``None``
            (default) is intentional — the cap kills T-series DIXI
            shanks structurally; opt-in only when callers know they
            see clinical-clip FPs.
        dedup_perp_mm / dedup_angle_deg: post-placement axis dedup
            tolerances. Tightens to ``PRE_DEDUP_*`` for the pre-place
            grouping stage.
        enable_synth_fallback: include v1's ``_axis_to_skull_synth``
            second-chance branch when ``anchor_trajectory_to_bolt``
            misses. Costs nothing on the 7-subject benchmark (M7 ≡ M5)
            but kept on by default for distributions where stage-1
            chains genuinely lack bolt CCs.
        min_blobs_per_line: stage-1 walker chain floor. Default 4 (M9).
            v1's default 5 was calibrated for the bolt-anchor+Frangi
            validator; in the unified pipeline the matched-filter NCC
            is the validator and 5 is over-strict. Lowering to 4
            recovers AMC137/LPT for +1 orphan.

    Returns:
        list of ``UnifiedTrajectory`` ordered by descending corr_score
        (highest-confidence first, matches the dedup keep order).
    """
    # Lazy imports — keep ``import rosa_core`` cheap.
    from rosa_detect import contact_pitch_v1_fit as f1
    from rosa_detect.primitives.bolt_anchor import (
        extract_bolt_candidates,
        anchor_trajectory_to_bolt,
        METAL_BOLT_THRESHOLD,
        BOLT_HULL_PROXIMITY_MM,
    )
    from .contact_placement_v2 import (
        place_contacts_for_seed_v2,
        MIN_CORR_FOR_REAL_SHANK,
        MIN_SLOT_HU_MEAN,
    )

    if min_corr is None:
        min_corr = MIN_CORR_FOR_REAL_SHANK
    if min_slot_hu_mean is None:
        min_slot_hu_mean = MIN_SLOT_HU_MEAN

    log_arr = features["log"]
    ct_arr_kji = features["ct_arr_kji"]
    dist_arr = features["head_distance"]
    i2r = np.asarray(features["ijk_to_ras_mat"], dtype=float)
    r2i = np.asarray(features["ras_to_ijk_mat"], dtype=float)
    spacing = features.get("spacing_xyz")
    if spacing is None:
        # SimpleITK image carries spacing. Fall back if not exposed.
        img = features.get("img")
        spacing = img.GetSpacing() if img is not None else (1.0, 1.0, 1.0)

    # Stage B.1 — bolt extraction in HU-mode (matches v1's call).
    metal_evidence = f1.compute_metal_evidence_volume(log_arr, ct_arr_kji)
    bolts, _ = extract_bolt_candidates(
        log_arr, dist_arr, i2r, spacing,
        ras_to_ijk_mat=r2i,
        ct_arr=metal_evidence, hu_threshold=METAL_BOLT_THRESHOLD,
        hull_proximity_mm=BOLT_HULL_PROXIMITY_MM,
    )

    # Stage B.2 — permissive stage1.
    def kji_to_ras(kji):
        k, j, i = int(kji[0]), int(kji[1]), int(kji[2])
        return (i2r @ np.array([i, j, k, 1.0]))[:3]

    if pitch_strategy_pitches_mm is None:
        pitches = sorted(
            {float(m["pitch_mm"]) for m in library_models if m.get("pitch_mm")}
        )
        if not pitches:
            pitches = [3.5]
    else:
        pitches = list(pitch_strategy_pitches_mm)

    # Run stage 1 with the unified pipeline's MIN_BLOBS floor. v1 reads
    # ``MIN_BLOBS_PER_LINE`` from the module global at call time
    # (multiple internal call sites), so the cleanest override is a
    # try/finally swap on the module attribute. This is single-threaded
    # per detection call by construction (the Slicer scene + CLI are
    # both single-threaded callers), so there's no concurrency hazard.
    saved_min_blobs = f1.MIN_BLOBS_PER_LINE
    f1.MIN_BLOBS_PER_LINE = int(min_blobs_per_line)
    try:
        chains, _pts = f1.run_stage1(
            log_arr, kji_to_ras, dist_arr, r2i,
            pitches_mm=pitches, frangi_arr=None,
        )
    finally:
        f1.MIN_BLOBS_PER_LINE = saved_min_blobs

    # Stage B.3 — refine axis (anchor / synth) and reject when neither.
    cands: list[dict] = []
    for c in chains:
        s = np.asarray(c["start_ras"], float)
        e = np.asarray(c["end_ras"], float)
        s, e, anchored = _refine_axis_via_bolt(s, e, bolts, anchor_trajectory_to_bolt)
        synthed = False
        if not anchored and enable_synth_fallback:
            s_syn, e_syn = _synth_fallback(
                c, dist_arr, r2i,
                f1._orient_shallow_to_deep, f1._axis_to_skull_synth,
                f1.MIN_BLOBS_PER_LINE,
                f1.DEEP_TIP_MIN_MM,
                f1.DEEP_TIP_SHORT_MAX_AVG_PITCH_MM,
            )
            if s_syn is not None:
                s, e, synthed = s_syn, e_syn, True
        if not (anchored or synthed):
            continue
        cands.append({"start_ras": s, "end_ras": e,
                       "anchored": anchored, "synthed": synthed})

    # Pre-dedup at tight tolerance: group close candidates so we try
    # each group member in corr order and keep the first that places
    # successfully. Mirrors the validated probe behavior.
    groups: list[list[dict]] = []
    for cand in cands:
        s = cand["start_ras"]; e = cand["end_ras"]
        placed = False
        for grp in groups:
            g0 = grp[0]
            if _axis_dup(g0["start_ras"], g0["end_ras"], s, e,
                          PRE_DEDUP_PERP_MM, PRE_DEDUP_ANGLE_DEG):
                grp.append(cand); placed = True; break
        if not placed:
            groups.append([cand])

    # Stage C — place + score per group; first success wins.
    raw_results: list[dict] = []
    for grp in groups:
        for cand in grp:
            try:
                res = place_contacts_for_seed_v2(
                    cand["start_ras"], cand["end_ras"],
                    features=features, library_models=library_models,
                    min_corr_for_real_shank=min_corr,
                    min_slot_hu_mean=min_slot_hu_mean,
                    max_slot_cc_volume_p90_mm3=max_slot_cc_volume_p90_mm3,
                )
            except Exception as exc:
                warnings.warn(
                    f"unified_detect: place_contacts_for_seed_v2 raised "
                    f"({exc}); skipping candidate "
                    f"{cand['start_ras']!r} → {cand['end_ras']!r}",
                    stacklevel=2,
                )
                continue
            if res.success:
                raw_results.append({
                    "res": res,
                    "anchored": cand["anchored"],
                    "synthed": cand["synthed"],
                })
                break  # first success per group wins; remaining members are duplicates.

    # Stage D — post-placement dedup; sort by corr (best first).
    raw_results.sort(key=lambda r: -float(r["res"].corr_score))
    kept: list[UnifiedTrajectory] = []
    for entry in raw_results:
        res = entry["res"]
        cl = np.asarray(res.centerline_ras, float)
        if cl.shape[0] < 2:
            continue
        s_new = cl[0]; e_new = cl[-1]
        is_dup = False
        for k in kept:
            ks = np.asarray(k.centerline_ras, float)[0]
            ke = np.asarray(k.centerline_ras, float)[-1]
            if _axis_dup(ks, ke, s_new, e_new, dedup_perp_mm, dedup_angle_deg):
                is_dup = True; break
        if is_dup:
            continue
        kept.append(UnifiedTrajectory(
            start_ras=[float(x) for x in s_new],
            end_ras=[float(x) for x in e_new],
            model_id=res.model_id,
            corr_score=float(res.corr_score),
            placed_ras=[[float(x) for x in pt] for pt in (res.placed_ras or [])],
            centerline_ras=[[float(x) for x in pt] for pt in cl],
            bolt_source=str(res.bolt_source),
            n_placed=int(res.n_placed),
            anchored=bool(entry["anchored"]),
            synthed=bool(entry["synthed"]),
        ))

    return kept
