"""Stage A — anchor.

Two distinct walker concepts in this codebase (don't conflate):

* The **bolt-end walker** (``estimate_bolt_end_from_metal_mass``) walks the seed
  looking for where the metal-mass tail drops. **Only runs for metal anchors.**
  ``anchor_bolt_less`` skips it entirely — there's no bolt to walk to.
* The **disk-stat sampler** (Stage C) walks the centerline emitting a 1D HU/LoG
  signal for the matched filter. Runs for both anchor types — it's not finding
  a bolt, it's collecting the placement signal.

``anchor_metal`` runs the bolt-end walker + the degenerate-contact-zone
reject. ``anchor_bolt_less`` is the explicit straight-seed fallback: the
entire emitter seed becomes the centerline, ``bolt_end_arc=0.0``,
``max_extend=0.0``. The matched filter scores across the whole centerline.

**No reverse retry** — seeders emit canonical entry→target direction;
manual-mode flips are the caller's problem to fix upstream.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .constants import DEGENERATE_CONTACT_ZONE_MM
from .context import PlacementCtx


def anchor_metal(ctx: PlacementCtx) -> PlacementCtx | None:
    """Try to anchor via the metal-mass walker. Returns ``None`` on failure.

    Failure modes:
      * The walker itself raises (sometimes happens on degenerate seeds).
      * ``bolt_end_arc_mm`` or ``centerline`` is missing from the result.
      * Centerline length minus bolt_end is below ``DEGENERATE_CONTACT_ZONE_MM``
        (no contact zone left after the bolt — common for cropped-bolt shanks
        like AMC137 LI/LPT/RI/RU).
    """
    from .bolt_end import estimate_bolt_end_from_metal_mass
    try:
        be = estimate_bolt_end_from_metal_mass(
            ctx.seed_start, ctx.seed_end,
            features=ctx.features, library_models=ctx.library_models,
        )
    except Exception:
        return None
    be_arc = be.get("bolt_end_arc_mm")
    cp = be.get("centerline")
    if be_arc is None or cp is None:
        return None
    cp = np.asarray(cp, dtype=float)
    cp_total = float(np.linalg.norm(np.diff(cp, axis=0), axis=1).sum())
    if cp_total - float(be_arc) < DEGENERATE_CONTACT_ZONE_MM:
        return None
    return replace(ctx, centerline=cp, bolt_end_arc=float(be_arc), bolt_source="metal")


def anchor_bolt_less(ctx: PlacementCtx) -> PlacementCtx:
    """Straight-seed fallback: 2-point centerline, no bolt zone, no extension."""
    cl = np.vstack([ctx.seed_start, ctx.seed_end])
    return replace(ctx, centerline=cl, bolt_end_arc=0.0, bolt_source="bolt_less")


def stage_anchor(ctx: PlacementCtx) -> PlacementCtx:
    """Compose: try metal, fall back to bolt-less."""
    return anchor_metal(ctx) or anchor_bolt_less(ctx)


__all__ = ["anchor_bolt_less", "anchor_metal", "stage_anchor"]
