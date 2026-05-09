"""Stage E — place contacts on centerline.

Project picked slot arcs onto the centerline → RAS. Trivial; it's its own
module so the stage swap point is obvious at the call site.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .context import PlacementCtx
from .polyline import polyline_at_arc


def place_at_match(ctx: PlacementCtx) -> PlacementCtx:
    """Project Stage D's ``match.slot_arcs`` onto ``centerline`` → ``placed_ras``.

    No-op if the matched filter rejected (no slot_arcs). Returns the ctx
    unchanged so the score stages still see ``placed_ras = []``.
    """
    if ctx.match is None or ctx.match.slot_arcs is None or ctx.match.best_model_id is None:
        return ctx
    cl = np.asarray(ctx.centerline, dtype=float)
    placed = [polyline_at_arc(cl, float(a)) for a in ctx.match.slot_arcs]
    return replace(ctx, placed_ras=placed)


__all__ = ["place_at_match"]
