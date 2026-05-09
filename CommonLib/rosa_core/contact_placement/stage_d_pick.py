"""Stage D — pick library electrode model.

Two-step composition:

* ``pick_matched_filter`` — Pearson NCC against the library comb-template.
  Returns the full ``MatchedFilterResult`` (winning model_id, n_slots,
  n_covered, slot arcs, corr).
* ``pick_extent_aware`` — re-rank by ``corr × √(n_covered / max_n_covered)``.
  Pearson NCC normalizes by ``||t||·||s||``; since ``||t|| ∝ √n_slots``, short
  templates get an unfair denominator boost when aligned with a clean subset
  of a longer signal (T18/X11: 5AM beats 15AM by 0.04 corr despite 15 visible
  peaks). Multiplying by ``√(n_cov / max_n_cov)`` re-normalizes — equivalent
  to using the longest template's denominator for everyone.

  Margin defer: when matched-filter raw ``top1 − top2 > PICK_OVERRIDE_MARGIN``,
  trust the matched filter — only re-rank ties.

Validated 2026-05-09 across 7 subjects (79 GT shanks): plain matched filter
78.5%/82.3% (HU/LoG); +dn 79.7%/83.5%. Replaces the prior peak-count re-rank
which regressed T18 HU 12→11.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..matched_filter import matched_filter_pick
from .constants import PICK_OVERRIDE_MARGIN, WALK_TIP_PAD_MM
from .context import PlacementCtx


def per_model_corrs(ctx: PlacementCtx) -> list[tuple]:
    """Score every library model against this ctx's signal.

    Returns list of ``(model_id, n_slots, n_covered, corr)`` sorted desc by corr.
    Used by ``pick_extent_aware`` for the denominator correction and by
    ``score_simple`` for the model-corr uniformity / margin features.
    """
    if ctx.walk_arcs is None or ctx.walk_signal is None:
        return []
    cl = np.asarray(ctx.centerline, dtype=float)
    cl_max = float(np.linalg.norm(np.diff(cl, axis=0), axis=1).sum())
    max_extend = WALK_TIP_PAD_MM if ctx.bolt_source == "metal" else 0.0
    out = []
    for m in ctx.library_models:
        try:
            r = matched_filter_pick(
                ctx.walk_arcs, ctx.walk_signal, [m],
                bolt_end_arc=ctx.bolt_end_arc,
                profile_end_arc=cl_max,
                max_extend_tip_mm=max_extend,
            )
            out.append((str(m.get("id") or ""), int(r.n_slots), int(r.n_covered), float(r.corr)))
        except Exception:
            continue
    out.sort(key=lambda t: -t[3])
    return out


def pick_matched_filter(ctx: PlacementCtx) -> PlacementCtx:
    """Run matched-filter pick across the full library, store ``MatchedFilterResult``."""
    cl = np.asarray(ctx.centerline, dtype=float)
    cl_max = float(np.linalg.norm(np.diff(cl, axis=0), axis=1).sum())
    max_extend = WALK_TIP_PAD_MM if ctx.bolt_source == "metal" else 0.0
    res = matched_filter_pick(
        ctx.walk_arcs, ctx.walk_signal, ctx.library_models,
        bolt_end_arc=ctx.bolt_end_arc,
        profile_end_arc=cl_max,
        max_extend_tip_mm=max_extend,
    )
    return replace(ctx, match=res)


def pick_extent_aware(ctx: PlacementCtx) -> PlacementCtx:
    """Re-rank by ``corr × √(n_covered / max_n_covered)`` (denominator correction).

    Margin defer: only re-rank when raw ``top1 − top2 < PICK_OVERRIDE_MARGIN``.
    If the preferred winner differs, re-run ``matched_filter_pick`` against
    just that model so its ``slot_arcs`` (used by Stage E) align with the
    better template.
    """
    if ctx.match is None or ctx.walk_arcs is None or ctx.centerline is None:
        return ctx
    pmc = per_model_corrs(ctx)
    if len(pmc) < 2:
        return ctx

    if pmc[0][3] - pmc[1][3] > PICK_OVERRIDE_MARGIN:
        return ctx

    max_cov = max(t[2] for t in pmc)
    if max_cov == 0:
        return ctx
    weighted = [(t[0], t[3] * float(np.sqrt(t[2] / max_cov))) for t in pmc]
    weighted.sort(key=lambda x: -x[1])
    preferred_id = weighted[0][0]
    if preferred_id == ctx.match.best_model_id:
        return ctx

    lookup = {str(m.get("id") or ""): m for m in ctx.library_models}
    preferred_model = lookup.get(preferred_id)
    if preferred_model is None:
        return ctx
    cl = np.asarray(ctx.centerline, dtype=float)
    cl_total = float(np.linalg.norm(np.diff(cl, axis=0), axis=1).sum())
    max_extend = WALK_TIP_PAD_MM if ctx.bolt_source == "metal" else 0.0
    new_match = matched_filter_pick(
        ctx.walk_arcs, ctx.walk_signal, [preferred_model],
        bolt_end_arc=ctx.bolt_end_arc,
        profile_end_arc=cl_total,
        max_extend_tip_mm=max_extend,
    )
    return replace(ctx, match=new_match)


__all__ = ["per_model_corrs", "pick_extent_aware", "pick_matched_filter"]
