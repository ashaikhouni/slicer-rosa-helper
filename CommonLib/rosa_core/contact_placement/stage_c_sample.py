"""Stage C — sample disk-stat signal along the centerline.

Two interchangeable implementations:

* ``sample_hu_max`` — positive polarity on raw HU
* ``sample_neg_log_max`` — negative polarity on LoG σ=1

Both wrap ``walk_centerline`` and emit per-arc ``WALK_AGGREGATOR``-statistic
of disk samples.

Cross-shank ownership is honored when set (priority order):

1. ``ctx.voxel_ownership`` (precomputed per-voxel label volume) — the
   ``use_voxel_ownership=True`` path. Each disk sample is masked in only if
   its voxel is uniquely owned by this seed (``label == seed_idx + 1``).
2. ``ctx.other_centerlines`` (per-disk nearest-line) — the legacy path.
   Each sample is owned by whichever centerline it's closest to.

Both routes prevent the walker from inflating the per-arc signal with a
neighbor's metal voxels. Without either, no masking is applied.

Disk geometry: 1 + ``WALK_N_RADII`` × ``WALK_N_ANGLES`` = 37 samples per disk
(more stable than ``sample_disk_along_polyline``'s 17-sample default).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .constants import (
    LOG_TOTAL_THRESHOLD,
    WALK_AGGREGATOR,
    WALK_DISK_RADIUS_MM,
    WALK_HU_MIN,
    WALK_N_ANGLES,
    WALK_N_RADII,
    WALK_STEP_MM,
    WALK_TIP_PAD_MM,
)
from .context import PlacementCtx
from .polyline import extend_centerline_tail, min_dist_pts_to_polyline, ortho_uv


def aggregate_disk(samples: np.ndarray, mask: np.ndarray, kind: str) -> float:
    """Aggregate disk samples by ``kind`` over the masked-finite values.

    Supports "max" / "p90" / "p75" / "median". Unknown kinds fall back to "max".
    Returns 0.0 if no finite-valid samples remain (e.g. all owned by neighbor).
    """
    finite = np.isfinite(samples) & mask
    if not finite.any():
        return 0.0
    s = samples[finite]
    if kind == "max":
        return float(s.max())
    if kind == "p90":
        return float(np.percentile(s, 90))
    if kind == "p75":
        return float(np.percentile(s, 75))
    if kind == "median":
        return float(np.median(s))
    return float(s.max())


def walk_centerline(
    ctx: PlacementCtx, *, volume, polarity: str,
    total_threshold: float = WALK_HU_MIN,
):
    """Walk the centerline emitting per-arc disk stats.

    Args:
        volume: numpy array (KJI) to sample. ``ct_arr_kji`` for HU mode,
            ``log`` for −LoG mode.
        polarity: ``"positive"`` or ``"negative"``. Negative means flip sign
            after sampling (LoG response is negative at metal-bright minima).
        total_threshold: legacy threshold passed for parity with
            ``sample_disk_along_polyline``; the staged walker doesn't gate on
            it (the matched filter handles below-threshold suppression).

    Returns ``(arcs, signal)`` — same shape as ``sample_disk_along_polyline``.
    Centerline is extended by ``WALK_TIP_PAD_MM`` only when ``bolt_source ==
    "metal"`` (bolt_less seeds use the seed line as the contact zone — no
    extension).
    """
    from ..volume_sampling import sample_trilinear_batch
    r2i = np.asarray(ctx.features["ras_to_ijk_mat"], dtype=float)
    cl = np.asarray(ctx.centerline, dtype=float)
    max_extend = WALK_TIP_PAD_MM if ctx.bolt_source == "metal" else 0.0
    cl_walk = extend_centerline_tail(cl, max_extend) if max_extend > 0 else cl

    diffs = np.diff(cl_walk, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(cum[-1])
    arcs = np.arange(0.0, total + 0.5 * WALK_STEP_MM, WALK_STEP_MM)
    out = np.zeros(len(arcs), dtype=float)

    n_per = 1 + WALK_N_RADII * WALK_N_ANGLES
    off_u = np.zeros(n_per, dtype=float)
    off_v = np.zeros(n_per, dtype=float)
    idx = 1
    for r_idx in range(1, WALK_N_RADII + 1):
        rr = WALK_DISK_RADIUS_MM * r_idx / WALK_N_RADII
        for a_idx in range(WALK_N_ANGLES):
            ang = 2.0 * np.pi * a_idx / WALK_N_ANGLES
            off_u[idx] = rr * np.cos(ang)
            off_v[idx] = rr * np.sin(ang)
            idx += 1

    others = [np.asarray(ocl, dtype=float) for ocl in (ctx.other_centerlines or [])
              if ocl is not None and len(ocl) >= 2]
    dist_self = np.sqrt(off_u ** 2 + off_v ** 2)

    own_vol = ctx.voxel_ownership
    own_idx = ctx.voxel_seed_idx
    use_vox = own_vol is not None and own_idx is not None
    target_label = (int(own_idx) + 1) if use_vox else 0
    if use_vox:
        own_kshape, own_jshape, own_ishape = own_vol.shape

    for ai, t in enumerate(arcs):
        i = int(np.searchsorted(cum, t, side="right") - 1)
        i = max(0, min(i, len(diffs) - 1))
        t_frac = (t - cum[i]) / max(seg_lens[i], 1e-9)
        center = cl_walk[i] + t_frac * diffs[i]
        tangent = diffs[i] / max(seg_lens[i], 1e-9)
        u, v = ortho_uv(tangent)
        pts = center[None, :] + off_u[:, None] * u[None, :] + off_v[:, None] * v[None, :]

        if use_vox:
            # Per-voxel ownership lookup: own only the voxels this seed
            # uniquely claimed. Unowned (background or contested) voxels
            # are masked out, same effect as Pass-2's per-disk rule but
            # consistent with Stage B's earlier ownership decision.
            ones = np.ones((n_per, 1), dtype=float)
            ijk = (r2i @ np.hstack([pts, ones]).T).T[:, :3]
            ijk_round = np.round(ijk).astype(int)
            i_v, j_v, k_v = ijk_round[:, 0], ijk_round[:, 1], ijk_round[:, 2]
            in_bounds = (
                (k_v >= 0) & (k_v < own_kshape) &
                (j_v >= 0) & (j_v < own_jshape) &
                (i_v >= 0) & (i_v < own_ishape)
            )
            owned = np.zeros(n_per, dtype=bool)
            if in_bounds.any():
                labels_at = own_vol[k_v[in_bounds], j_v[in_bounds], i_v[in_bounds]]
                owned[in_bounds] = (labels_at == target_label)
        elif others:
            dist_other = np.full(n_per, np.inf, dtype=float)
            for ocl in others:
                dist_other = np.minimum(dist_other, min_dist_pts_to_polyline(pts, ocl))
            owned = dist_self <= dist_other
        else:
            owned = np.ones(n_per, dtype=bool)

        samples = sample_trilinear_batch(volume, r2i, pts)
        if polarity == "negative":
            samples = -samples
        out[ai] = aggregate_disk(samples, owned, WALK_AGGREGATOR)

    return arcs, out


def sample_hu_max(ctx: PlacementCtx) -> PlacementCtx:
    """Positive-polarity disk walker on raw HU."""
    arcs, sig = walk_centerline(
        ctx, volume=ctx.features["ct_arr_kji"],
        polarity="positive", total_threshold=WALK_HU_MIN,
    )
    return replace(ctx, walk_arcs=arcs, walk_signal=sig, signal_kind="hu_max")


def sample_neg_log_max(ctx: PlacementCtx) -> PlacementCtx:
    """Negative-polarity disk walker on LoG σ=1 (preferred — see notebook)."""
    arcs, sig = walk_centerline(
        ctx, volume=ctx.features["log"],
        polarity="negative", total_threshold=LOG_TOTAL_THRESHOLD,
    )
    return replace(ctx, walk_arcs=arcs, walk_signal=sig, signal_kind="neg_log_max")


__all__ = [
    "aggregate_disk",
    "sample_hu_max",
    "sample_neg_log_max",
    "walk_centerline",
]
