"""Cross-shank-aware two-pass placement orchestrator.

Pass 1: anchor + refine all seeds, collect refined centerlines.
Pass 2: per-seed sample + pick + place + score with ``other_centerlines`` set
        so the walker / refine masks voxels owned by neighboring shanks.

A pass-1.5 ownership-aware re-refine was tried 2026-05-09 — helped T18
(HU +1) but regressed AMC135/T1/T2/T3/T4 (cumulative −7 HU). Reverted; do
not re-introduce.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Iterable

import numpy as np

from .context import PlacementCtx
from .snap import compute_voxel_ownership
from .stage_a_anchor import stage_anchor
from .stage_b_refine import refine_log_snap
from .stage_c_sample import sample_hu_max
from .stage_d_pick import pick_model
from .stage_e_place import place_at_match
from .stage_f_score import score_cc_overlap, score_compound, score_simple


def _seed_to_ctx(
    seed: dict, *,
    features: dict, library_models: list[dict],
    bolts: list[dict] | None,
) -> PlacementCtx:
    """Build a fresh PlacementCtx from a seed dict.

    Seed dict shape: ``{"start_ras": (3,), "end_ras": (3,), [confidence],
    [confidence_label], [electrode_model]}``. The optional fields populate
    the seeder-side metadata used by the compound score.
    """
    return PlacementCtx(
        seed_start=np.asarray(seed["start_ras"], dtype=float),
        seed_end=np.asarray(seed["end_ras"], dtype=float),
        features=features,
        library_models=library_models,
        bolts=bolts,
        seeder_confidence=float(seed.get("confidence") or 0.0),
        seeder_label=str(seed.get("confidence_label") or ""),
        seeder_model=seed.get("electrode_model"),
    )


def run_two_pass(
    seeds: Iterable[dict], *,
    features: dict,
    library_models: list[dict],
    bolts: list[dict] | None = None,
    refine_fn: Callable[[PlacementCtx], PlacementCtx] = refine_log_snap,
    sample_fn: Callable[[PlacementCtx], PlacementCtx] = sample_hu_max,
    use_voxel_ownership: bool = False,
) -> list[PlacementCtx]:
    """Two-pass placement: anchor+refine all seeds, then sample+place each
    with cross-shank ownership masks.

    Returns a list of fully-scored ``PlacementCtx`` objects, one per input
    seed, in the original input order. ``other_centerlines`` for each ctx
    excludes its own refined centerline.

    When ``use_voxel_ownership=True``, the orchestrator splits Pass 1 into
    Pass 1a (anchor every seed) and Pass 1b (refine each seed) and inserts a
    per-voxel ownership computation between them. Each ctx then carries the
    shared ownership volume and its own seed index, so Stage B's refine can
    mask LoG voxels owned by a neighbor before snapping. This prevents the
    "snap drifts onto a passing shank's blobs" failure mode (S54 LSFG
    motivating case 2026-05-11). The pre-refine anchored lines are used as
    the ownership boundary — trustworthy axes that haven't drifted yet.
    Default is False (no behavior change vs. pre-2026-05-11).

    The post-pass ``apply_subject_fft_normalization`` is NOT called here —
    callers (``placement_modes``) decide whether subject normalization is
    appropriate (yes for mode-1 batch runs, no for mode-4 single-seed).
    """
    seeds_list = list(seeds)

    ownership_vol = None
    if use_voxel_ownership:
        # Pass 1a — anchor every seed (no refine yet).
        anchored: list[PlacementCtx] = []
        for s in seeds_list:
            ctx = _seed_to_ctx(s, features=features, library_models=library_models, bolts=bolts)
            ctx = stage_anchor(ctx)
            anchored.append(ctx)
        # Build the per-voxel ownership volume from the anchored centerlines.
        # Skipping entries whose anchor produced no centerline preserves
        # index alignment because the bolt-less fallback always sets one.
        anchor_lines = [
            np.asarray(c.centerline, dtype=float) for c in anchored
            if c.centerline is not None
        ]
        log_arr = features.get("log")
        r2i = features.get("ras_to_ijk_mat")
        if log_arr is None or r2i is None or not anchor_lines:
            ownership_vol = None
        else:
            ownership_vol = compute_voxel_ownership(
                anchor_lines, log_arr, np.asarray(r2i, dtype=float),
            )
        # Pass 1b — refine each seed with the ownership volume in ctx.
        # The ownership volume + seed index stay on the ctx for Pass 2 so
        # Stages C/D/E/F consume the SAME voxel labels Stage B used (no
        # redundant per-disk nearest-line recomputation, and consistent
        # ownership decisions across stages).
        pass1: list[PlacementCtx] = []
        for ei, base in enumerate(anchored):
            if ownership_vol is not None:
                ctx = replace(
                    base,
                    voxel_ownership=ownership_vol,
                    voxel_seed_idx=ei,
                )
            else:
                ctx = base
            ctx = refine_fn(ctx)
            pass1.append(ctx)
    else:
        pass1 = []
        for s in seeds_list:
            ctx = _seed_to_ctx(s, features=features, library_models=library_models, bolts=bolts)
            ctx = stage_anchor(ctx)
            ctx = refine_fn(ctx)
            pass1.append(ctx)

    all_centerlines = [
        np.asarray(c.centerline, float) for c in pass1 if c.centerline is not None
    ]

    # If voxel ownership was used in Pass 1, REBUILD it from the refined
    # centerlines so Stage C/D/E/F see ownership consistent with the line
    # they're sampling. The Pass-1 volume was built from anchored (pre-refine)
    # lines and would mis-label voxels along the moved centerlines.
    if use_voxel_ownership and all_centerlines:
        log_arr = features.get("log")
        r2i = features.get("ras_to_ijk_mat")
        if log_arr is not None and r2i is not None:
            ownership_vol = compute_voxel_ownership(
                all_centerlines, log_arr, np.asarray(r2i, dtype=float),
            )

    pass2: list[PlacementCtx] = []
    for ei, base in enumerate(pass1):
        # Skip our own centerline. Indexing into all_centerlines requires
        # accounting for entries whose centerline was None — but every base
        # in pass1 produced a centerline (anchor falls through to bolt_less
        # which always produces a 2-point line), so the indices align.
        #
        # Ownership for Stages C/D/E/F:
        #   * voxel-ownership path (use_voxel_ownership=True): the rebuilt
        #     post-refine ownership volume goes onto ctx; Stage C's walker
        #     consumes it directly (per-voxel labels — consistent with
        #     Stage B's earlier mask, computed against the right geometry).
        #   * legacy path: per-disk nearest-line via ``other_centerlines``.
        others = [cl for j, cl in enumerate(all_centerlines) if j != ei]
        if use_voxel_ownership and ownership_vol is not None:
            ctx = replace(
                base,
                other_centerlines=others,
                voxel_ownership=ownership_vol,
                voxel_seed_idx=ei,
            )
        else:
            ctx = replace(base, other_centerlines=others)
        ctx = sample_fn(ctx)
        ctx = pick_model(ctx)
        ctx = place_at_match(ctx)
        ctx = score_simple(ctx)
        ctx = score_cc_overlap(ctx)
        ctx = score_compound(ctx)
        pass2.append(ctx)

    return pass2


__all__ = ["run_two_pass"]
