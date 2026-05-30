"""Single-seed staged placement composer.

``place_seed`` runs Stages A → F.3 once for one seed pair. For multi-seed
two-pass cross-shank-aware placement, use ``run_two_pass``. For the user-facing
mode dispatcher (CT-only, count-only, etc.), use ``rosa_core.placement_modes``.

``place_v3`` is a deprecated alias kept until Slicer Auto-Fit migrates in
Session 4.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .context import PlacementCtx
from .stage_a_anchor import stage_anchor
from .stage_b_refine import refine_log_snap
from .stage_c_sample import sample_hu_max
from .stage_d_pick import pick_model
from .stage_e_place import place_at_match
from .stage_f_score import score_cc_overlap, score_compound, score_simple


def place_seed(
    seed_start, seed_end, *,
    features: dict,
    library_models: list[dict],
    bolts: list[dict] | None = None,
    other_centerlines: list[np.ndarray] | None = None,
    seeder_confidence: float = 0.0,
    seeder_label: str = "",
    seeder_model: str | None = None,
    refine_fn: Callable[[PlacementCtx], PlacementCtx] = refine_log_snap,
    sample_fn: Callable[[PlacementCtx], PlacementCtx] = sample_hu_max,
) -> PlacementCtx:
    """Run the full A→F.3 staged pipeline for one seed.

    Stage swap points exposed as kwargs:
      * ``refine_fn`` — default ``refine_log_snap``; pass ``refine_noop`` to skip.
      * ``sample_fn`` — default ``sample_hu_max``; pass ``sample_neg_log_max``
        for the LoG-side compound (notebook-validated to dominate HU).

    No bidirectional retry — seeders emit canonical entry→target. Caller is
    responsible for fixing reversed seeds upstream.
    """
    ctx = PlacementCtx(
        seed_start=np.asarray(seed_start, dtype=float),
        seed_end=np.asarray(seed_end, dtype=float),
        features=features,
        library_models=library_models,
        bolts=bolts,
        other_centerlines=other_centerlines,
        seeder_confidence=float(seeder_confidence),
        seeder_label=str(seeder_label or ""),
        seeder_model=seeder_model,
    )
    ctx = stage_anchor(ctx)
    ctx = refine_fn(ctx)
    ctx = sample_fn(ctx)
    ctx = pick_model(ctx)
    ctx = place_at_match(ctx)
    ctx = score_simple(ctx)
    ctx = score_cc_overlap(ctx)
    ctx = score_compound(ctx)
    return ctx


# Deprecated alias — remove in Session 4.
place_v3 = place_seed


__all__ = ["place_seed", "place_v3"]
