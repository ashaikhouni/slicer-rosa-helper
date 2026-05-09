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
from .stage_a_anchor import stage_anchor
from .stage_b_refine import refine_log_snap
from .stage_c_sample import sample_hu_max
from .stage_d_pick import pick_extent_aware, pick_matched_filter
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
) -> list[PlacementCtx]:
    """Two-pass placement: anchor+refine all seeds, then sample+place each
    with cross-shank ownership masks.

    Returns a list of fully-scored ``PlacementCtx`` objects, one per input
    seed, in the original input order. ``other_centerlines`` for each ctx
    excludes its own refined centerline.

    The post-pass ``apply_subject_fft_normalization`` is NOT called here —
    callers (``placement_modes``) decide whether subject normalization is
    appropriate (yes for mode-1 batch runs, no for mode-4 single-seed).
    """
    seeds_list = list(seeds)

    pass1: list[PlacementCtx] = []
    for s in seeds_list:
        ctx = _seed_to_ctx(s, features=features, library_models=library_models, bolts=bolts)
        ctx = stage_anchor(ctx)
        ctx = refine_fn(ctx)
        pass1.append(ctx)

    all_centerlines = [
        np.asarray(c.centerline, float) for c in pass1 if c.centerline is not None
    ]

    pass2: list[PlacementCtx] = []
    for ei, base in enumerate(pass1):
        # Skip our own centerline. Indexing into all_centerlines requires
        # accounting for entries whose centerline was None — but every base
        # in pass1 produced a centerline (anchor falls through to bolt_less
        # which always produces a 2-point line), so the indices align.
        others = [cl for j, cl in enumerate(all_centerlines) if j != ei]
        ctx = replace(base, other_centerlines=others)
        ctx = sample_fn(ctx)
        ctx = pick_matched_filter(ctx)
        ctx = pick_extent_aware(ctx)
        ctx = place_at_match(ctx)
        ctx = score_simple(ctx)
        ctx = score_cc_overlap(ctx)
        ctx = score_compound(ctx)
        pass2.append(ctx)

    return pass2


__all__ = ["run_two_pass"]
