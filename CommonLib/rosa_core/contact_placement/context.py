"""``PlacementCtx`` — the single state object threaded through every stage.

Each stage is a function ``(PlacementCtx) -> PlacementCtx``. The dataclass is
**mutable in spirit**: stages return a new instance via ``dataclasses.replace``
so the inputs are immutable to each stage but the field-by-field write order
across the pipeline is clear at the call site.

Don't add behavior here — this is data, not logic. Score components are
computed in ``stage_f_score``; matched-filter results live in their own type.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..matched_filter import MatchedFilterResult


@dataclass
class PlacementCtx:
    """Per-emission state across the staged pipeline.

    Lifecycle (which stage writes which field):

    * Init (``compose.place_seed``): ``seed_start``, ``seed_end``, ``features``,
      ``library_models``, ``bolts``, ``seeder_*``, optional ``other_centerlines``.
    * Stage A (``stage_a_anchor``): ``centerline``, ``bolt_end_arc``, ``bolt_source``.
    * Stage B (``stage_b_refine``): refined ``centerline``.
    * Stage C (``stage_c_sample``): ``walk_arcs``, ``walk_signal``, ``signal_kind``.
    * Stage D (``stage_d_pick``): ``match``.
    * Stage E (``stage_e_place``): ``placed_ras``.
    * Stage F (``stage_f_score``): ``score_components`` (the dict that holds
      every per-emission diagnostic + the final ``compound_score`` and ``band``).
    """

    seed_start: np.ndarray
    seed_end: np.ndarray
    features: dict
    library_models: list[dict]

    # Optional global bolt CC list (subject-level, from the stage-1
    # ``extract_bolt_candidates`` pass). Independent of the per-emission
    # metal-mass walker that produces ``bolt_end_arc`` — used as a
    # confidence-only "reliable bolt present?" signal in ``score_cc_overlap``.
    bolts: list[dict] | None = None

    # Other emissions' centerlines (post-anchor + post-refine) for the
    # cross-shank ownership mask in Stage C. When set, each disk voxel
    # is owned by the centerline it's closest to; voxels owned by another
    # shank are zeroed before aggregating per-arc walker stats. Populated
    # by ``run_two_pass`` between pass 1 and pass 2.
    other_centerlines: list[np.ndarray] | None = None

    # Precomputed per-voxel ownership label volume (kji int16) and this
    # ctx's seed index within it. When both are set, Stage B uses
    # ``snap_centerline_voxel_owned`` (per-voxel masked snap with arc
    # interpolation across contested regions) instead of the production
    # LoG-centroid snap. Populated by ``run_two_pass`` when called with
    # ``use_voxel_ownership=True``.
    voxel_ownership: np.ndarray | None = None
    voxel_seed_idx: int | None = None

    # Seeder-side metadata (carried through from a v1 trajectory dict so
    # the compound score can fuse seeder confidence with the placement-side
    # signals — same direction as the user's 5-mode input API).
    seeder_confidence: float = 0.0
    seeder_label: str = ""                 # "high" | "medium" | "low" | ""
    seeder_model: str | None = None

    # Stage A / B writes:
    centerline: np.ndarray | None = None
    bolt_end_arc: float = 0.0
    bolt_source: str = "unknown"           # "metal" | "bolt_less" — walker outcome only

    # Stage C writes:
    walk_arcs: np.ndarray | None = None
    walk_signal: np.ndarray | None = None
    signal_kind: str = ""                  # "hu_max" | "neg_log_max" | ...

    # Stage D writes:
    match: MatchedFilterResult | None = None

    # Stage E writes:
    placed_ras: list[np.ndarray] = field(default_factory=list)

    # Stage F writes (and is the final user-facing summary):
    score_components: dict[str, Any] = field(default_factory=dict)


__all__ = ["PlacementCtx"]
