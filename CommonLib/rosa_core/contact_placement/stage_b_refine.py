"""Stage B — refine centerline.

For metal anchors: snap polynomial centerline to the LoG-centroid track
(recovers axes 1–2 mm off the actual electrode). For bolt-less anchors: no-op
(no reliable basis to refine a 2-point straight seed without contact peaks).

Three snap variants, picked in priority order:

1. ``ctx.voxel_ownership`` set → ``snap_centerline_voxel_owned``: uses a
   precomputed per-voxel ownership label volume. Voxels not uniquely owned
   by this ctx's seed are masked out; arcs with too few owned voxels are
   linearly interpolated. This is the cross-shank-aware path used when
   ``run_two_pass`` is called with ``use_voxel_ownership=True``.
2. ``ctx.other_centerlines`` set → ``snap_centerline_owned``: per-disk
   nearest-line ownership (the legacy Pass-2-aware variant).
3. otherwise → ``snap_centerline_to_centroid``: production LoG-centroid snap.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .context import PlacementCtx
from .snap import (
    snap_centerline_owned,
    snap_centerline_to_centroid,
    snap_centerline_voxel_owned,
)


def refine_log_snap(ctx: PlacementCtx) -> PlacementCtx:
    """Snap centerline to LoG centroid; no-op for bolt-less anchors."""
    if ctx.bolt_source != "metal":
        return ctx
    log_arr = ctx.features.get("log")
    if log_arr is None:
        return ctx
    r2i = np.asarray(ctx.features["ras_to_ijk_mat"], dtype=float)
    if ctx.voxel_ownership is not None and ctx.voxel_seed_idx is not None:
        snapped = snap_centerline_voxel_owned(
            ctx.centerline, log_arr, r2i,
            voxel_ownership=ctx.voxel_ownership,
            seed_idx=int(ctx.voxel_seed_idx),
        )
        return replace(ctx, centerline=np.asarray(snapped, dtype=float))
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
