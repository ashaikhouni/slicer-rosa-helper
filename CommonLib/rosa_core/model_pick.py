"""Shared electrode-model pick orchestration — the single source of truth.

Given a centerline signal, the bolt anchor, and the snapped contact chain, decide
which library model the electrode is. Both the headless CLI (``rosa_agent
fit-rosa``) and the Slicer-side placement call :func:`pick_electrode_model`, so
the matcher behaviour can never diverge between them (CLI<->Slicer parity).

Composition (each step lives in :mod:`rosa_core.matched_filter`):

    matched_filter_pick           -- comb-template library scoring
      -> extent_aware_pick        -- coverage-aware re-rank on near-ties
      -> family routing           -- cluster (CM/BM) trust extent-aware;
                                     uniform (AM/PMT) go through the verifier
      -> no_metal_rerank          -- downgrade a model with dead in-brain slots
      -> covering-floor           -- never under-cover a resolved proximal
                                     contact (bump UP to the smallest covering
                                     same-family model). AM/PMT only.

Relocated 2026-05-29 out of ``cli/rosa_agent/commands/fit_rosa.py`` so the
algorithm lives in the shared core, not the CLI layer.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .matched_filter import matched_filter_pick, extent_aware_pick, no_metal_rerank

# Covering-floor tuning (validated on DIXI T1-T25 + o-arm JR + AMC PMT).
_AM_FLOOR_PROMINENCE = 0.15   # min (peak-valley)/peak for a "resolved" bump
_AM_FLOOR_JUMP_PITCH = 1.3    # gap (in pitch units) that breaks array contiguity


def model_family(model_dict) -> str:
    """Coarse family for routing: 'cluster' (CM-style, non-uniform pitch),
    'mm' (DIXI MM family), or 'uniform' (everything else — AM / PMT). Used to
    skip the no_metal_rerank verifier on CM models where inter-cluster wire gaps
    would trigger false dead-slot flags, and to gate the covering-floor."""
    if not model_dict:
        return "?"
    m_id = str(model_dict.get("id", ""))
    if "MM" in m_id:
        return "mm"
    offs = sorted(model_dict.get("contact_center_offsets_from_tip_mm") or [])
    if len(offs) < 3:
        return "?"
    diffs = np.diff(offs)
    cv = float(np.std(diffs) / max(np.mean(diffs), 1e-6))
    return "cluster" if cv > 0.15 else "uniform"


def _count_resolved_proximal(chain, mf_arcs, mf_sig, anchor,
                             prom_thr: float = _AM_FLOOR_PROMINENCE,
                             jump_pitch: float = _AM_FLOOR_JUMP_PITCH) -> int:
    """Number of *resolved* contacts in the contiguous array — snap peaks that
    (a) lie distal of the bolt anchor, (b) show real local modulation (a bump
    with prominence >= ``prom_thr`` on ``mf_sig``, not flat bolt), and (c) are
    contiguous with the deep tip (no gap > ``jump_pitch`` x pitch).

    The contiguity walk (from the deepest peak proximally, stopping at the first
    jump) is essential: the bolt anchor is sometimes a few mm proximal of the
    first real contact, leaving a gap in which an isolated bright lead/bolt peak
    can sit. That peak passes (a) and (b) but is separated from the array body by
    a jump, so without (c) it would inflate the count and over-bump the model
    (observed on AMC PMT shanks). Counting only the contiguous run from the tip
    credits exactly the contacts the shank resolves as a coherent array."""
    from .centerline import unit
    pts = np.asarray(chain.get("kept_pts", []), dtype=float)
    if len(pts) < 2:
        return int(len(pts))
    entry = np.asarray(chain["entry_ras"], dtype=float)
    axis = unit(np.asarray(chain["axis"], dtype=float))
    snap_arcs = np.sort((pts - entry) @ axis)
    pitch = float(np.median(np.diff(snap_arcs)))
    if not np.isfinite(pitch) or pitch <= 0:
        return int(len(pts))
    arcs = np.asarray(mf_arcs, dtype=float)
    sig = np.asarray(mf_sig, dtype=float)
    step = float(np.median(np.diff(arcs))) if len(arcs) > 1 else 0.3
    half = max(1, int(round(pitch / step)))

    def _prominent(a: float) -> bool:
        i = int(np.argmin(np.abs(arcs - a)))
        lo, hi = max(0, i - half), min(len(sig), i + half + 1)
        if hi - lo < 3:
            return True
        pk = float(sig[i])
        valley = max(float(sig[lo:i + 1].min()), float(sig[i:hi].min()))
        return pk > 0 and (pk - valley) / pk >= prom_thr

    # Prominent peaks distal of the anchor, sorted shallow -> deep.
    resolved = [a for a in snap_arcs if a >= float(anchor) and _prominent(a)]
    if not resolved:
        return 0
    # Contiguous run anchored at the deep tip: walk proximally while the gap to
    # the next-shallower resolved peak stays within the jump gate.
    run = 1
    for j in range(len(resolved) - 1, 0, -1):
        if resolved[j] - resolved[j - 1] <= jump_pitch * pitch:
            run += 1
        else:
            break
    return run


def _smallest_covering(current_id, n_resolved, models_dict, mf_res=None) -> str:
    """Smallest model of the SAME family (same vendor prefix + ``type`` as
    ``current_id``) whose contact_count >= ``n_resolved``. Caps at the largest
    family model when none covers; returns ``current_id`` if no ladder exists.

    When several models share the covering contact count (e.g. PMT-16A/B/C —
    same 16 contacts, different spacing), prefer the variant the matched filter
    scored highest (so the floor only fixes the COUNT and defers the spacing
    choice to the matcher), falling back to the first by id."""
    vendor = str(current_id).split("-", 1)[0]
    ctype = models_dict[current_id].get("type")
    ladder = sorted(
        ((mid, int(m.get("contact_count") or 0)) for mid, m in models_dict.items()
         if m.get("type") == ctype and str(mid).split("-", 1)[0] == vendor),
        key=lambda kv: kv[1],
    )
    if not ladder:
        return current_id
    covering = [(mid, c) for mid, c in ladder if c >= n_resolved]
    if not covering:
        return ladder[-1][0]
    target_count = covering[0][1]
    variants = [mid for mid, c in covering if c == target_count]
    if len(variants) == 1 or mf_res is None or not getattr(mf_res, "per_model", None):
        return variants[0]
    corr = {r.best_model_id: r.corr for r in mf_res.per_model if r.best_model_id}
    return max(variants, key=lambda v: corr.get(v, float("-inf")))


def pick_electrode_model(
    mf_arcs, mf_sig, candidate_models, models_dict, *,
    bolt_end_arc: float, profile_end_arc: float, chain,
    max_extend_tip_mm: float = 2.0, max_tip_short_mm: float = 2.5,
) -> tuple[str | None, str, Any, dict[str, Any]]:
    """Pick the electrode model from the matched-filter signal.

    Args:
        mf_arcs, mf_sig: centerline arc positions + per-arc tube-max signal.
        candidate_models: library models to score (already strategy-filtered).
        models_dict: id -> model dict for the full library (for family / count).
        bolt_end_arc: bolt->contact transition arc (matcher proximal anchor).
        profile_end_arc: centerline tip arc (matcher distal reference).
        chain: the snapped chain dict (kept_pts/entry_ras/axis) — drives the
            covering-floor resolved-contact count. Pass ``None`` to disable the
            covering-floor (e.g. callers that have no detected-peak chain at
            pick time); the rest of the pick is unaffected.
        max_extend_tip_mm / max_tip_short_mm: matcher tip-search slack.

    Returns ``(predicted_id, branch, mf_res, diag)``. ``predicted_id`` is None on
    no-pick; ``branch`` is one of matched_filter_cluster / no_pick /
    matched_filter_verified / matched_filter_downgraded / matched_filter_covering_floor;
    ``diag`` carries matched_filter_corr / matched_filter_best / extent_aware_pick
    and (when the floor fires) covering_floor_from / covering_floor_n_resolved.
    """
    mf_res = matched_filter_pick(
        mf_arcs, mf_sig, candidate_models,
        bolt_end_arc=bolt_end_arc, profile_end_arc=profile_end_arc,
        max_extend_tip_mm=max_extend_tip_mm, max_tip_short_mm=max_tip_short_mm,
    )
    pred_ea = extent_aware_pick(mf_res)
    # Family routing: cluster (CM) trusts extent-aware; AM/MM go through the
    # dead-slot verifier (CM templates have legitimate inter-cluster gaps the
    # verifier would wrongly flag).
    ea_model = models_dict.get(pred_ea) if pred_ea else None
    family = model_family(ea_model)
    if family == "cluster":
        predicted_id = pred_ea
        branch = "matched_filter_cluster"
    elif pred_ea is None:
        predicted_id = None
        branch = "no_pick"
    else:
        predicted_id = no_metal_rerank(
            mf_res, mf_arcs, mf_sig, anchor_arc=bolt_end_arc,
        )
        branch = ("matched_filter_verified"
                  if predicted_id == pred_ea
                  else "matched_filter_downgraded")
    diag: dict[str, Any] = {
        "matched_filter_corr": float(mf_res.corr) if mf_res.best_model_id else None,
        "matched_filter_best": mf_res.best_model_id,
        "extent_aware_pick": pred_ea,
    }

    # Covering-floor: a model cannot have fewer contacts than the shank visibly
    # resolves. If an AM/PMT pick under-covers the resolved count, bump UP to the
    # smallest same-family model that covers it (over-extension into the bolt is
    # acceptable; orphaning a resolved contact is not). CM/BM/MM never resized.
    if (chain is not None and predicted_id is not None
            and model_family(models_dict[predicted_id]) == "uniform"):
        n_resolved = _count_resolved_proximal(chain, mf_arcs, mf_sig, bolt_end_arc)
        if int(models_dict[predicted_id].get("contact_count") or 0) < n_resolved:
            bumped = _smallest_covering(predicted_id, n_resolved, models_dict, mf_res)
            if bumped != predicted_id:
                diag["covering_floor_from"] = predicted_id
                diag["covering_floor_n_resolved"] = int(n_resolved)
                predicted_id = bumped
                branch = "matched_filter_covering_floor"
    return predicted_id, branch, mf_res, diag


__all__ = ["pick_electrode_model", "model_family"]
