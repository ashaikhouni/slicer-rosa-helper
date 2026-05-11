"""Stage B — refine centerline.

For metal anchors: snap polynomial centerline to the LoG-centroid track
(recovers axes 1–2 mm off the actual electrode). For bolt-less anchors: no-op
(no reliable basis to refine a 2-point straight seed without contact peaks).

When ``ctx.other_centerlines`` is set (Pass 2 of ``run_two_pass``), the snap
masks LoG voxels closer to a neighboring centerline before computing the
centroid — prevents drift toward passing shanks.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .context import PlacementCtx
from .snap import snap_centerline_owned, snap_centerline_to_centroid


def refine_log_snap(ctx: PlacementCtx) -> PlacementCtx:
    """Snap centerline to LoG centroid; no-op for bolt-less anchors."""
    if ctx.bolt_source != "metal":
        return ctx
    log_arr = ctx.features.get("log")
    if log_arr is None:
        return ctx
    r2i = np.asarray(ctx.features["ras_to_ijk_mat"], dtype=float)
    if not ctx.other_centerlines:
        snapped = snap_centerline_to_centroid(ctx.centerline, log_arr, r2i)
        return replace(ctx, centerline=np.asarray(snapped, dtype=float))
    snapped = snap_centerline_owned(
        ctx.centerline, log_arr, r2i, others=ctx.other_centerlines,
    )
    return replace(ctx, centerline=np.asarray(snapped, dtype=float))


def refine_noop(ctx: PlacementCtx) -> PlacementCtx:
    """Identity refine — useful for ablation studies and tests."""
    return ctx


__all__ = ["refine_log_snap", "refine_noop"]
