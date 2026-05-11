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

from ..matched_filter import MatchedFilterResult, matched_filter_pick
from .constants import PICK_OVERRIDE_MARGIN, WALK_TIP_PAD_MM
from .context import PlacementCtx


def per_model_corrs(ctx: PlacementCtx) -> list[tuple]:
    """Score every library model against this ctx's signal.

    Returns list of ``(model_id, n_slots, n_covered, corr)`` sorted desc by corr.
    Used by ``pick_extent_aware`` for the denominator correction and by
    ``score_simple`` for the model-corr uniformity / margin features.

    Reads from ``ctx.match.per_model`` when available — populated by
    ``pick_matched_filter`` in a single library pass — so the ``stage_d``
    pipeline doesn't redo per-model scoring. Falls back to a fresh
    library scan when ``ctx.match`` is None or carries no per-model
    data (older callers that bypass pick_matched_filter).
    """
    if ctx.walk_arcs is None or ctx.walk_signal is None:
        return []
    if ctx.match is not None and ctx.match.per_model is not None:
        return [
            (str(r.best_model_id or ""), int(r.n_slots), int(r.n_covered), float(r.corr))
            for r in ctx.match.per_model
        ]
    # Fallback path — preserves the old shape when callers run
    # per_model_corrs without a prior pick_matched_filter.
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
    When the preferred winner differs from the matched-filter pick, swap
    in the preferred model's ``MatchedFilterResult`` from the cached
    ``ctx.match.per_model`` list — same data ``matched_filter_pick``
    already produced in the single library pass, no redundant rescoring.
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

    # Pull the preferred model's already-computed result out of the
    # per-model cache. Carry the per_model list forward so the picked
    # ctx.match still exposes the full ranking to downstream stages
    # (score_simple reads it for the model-corr-uniformity feature).
    if ctx.match.per_model is not None:
        preferred = next(
            (r for r in ctx.match.per_model if r.best_model_id == preferred_id),
            None,
        )
        if preferred is not None:
            new_match = MatchedFilterResult(
                best_model_id=preferred.best_model_id,
                tip_arc=preferred.tip_arc,
                slot_arcs=preferred.slot_arcs,
                n_slots=preferred.n_slots,
                n_covered=preferred.n_covered,
                corr=preferred.corr,
                per_model=ctx.match.per_model,
            )
            return replace(ctx, match=new_match)
    # Fallback (no per_model cache available — older caller path).
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
