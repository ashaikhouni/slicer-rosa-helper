"""Public 5-mode dispatcher for SEEG contact placement.

This module is the single user-facing entry point shared by the Slicer Auto-Fit
panel (Session 4) and the ``rosa-agent place`` CLI (Session 3). Mode is implied
by which optional fields are passed:

==============  =============  =================  ================================================
seeds            expected       n_expected         mode
==============  =============  =================  ================================================
None             None           None               1 — auto (CT only)
None             None           int                2 — count constraint
None             list           None               3 — names + types known, find them
list w/ model_id None           None               4 — placement only (user vouched)
list w/o model_id None          None               5 — seeds + library match
==============  =============  =================  ================================================

**Session 1 implements only mode 4.** Modes 1, 2, 3, 5 require candidate-seed
extraction from ``rosa_detect.contact_pitch_v1_fit`` (Session 2) and raise
``NotImplementedError`` for now.

The result is always a ``PlacementBatch`` carrying ``PlacedTrajectory``
records (band-classified, score-componented) plus optional QC directory
output (Session 3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import numpy as np

from .contact_placement import (
    PlacementCtx,
    apply_subject_fft_normalization,
    place_seed,
)


# ---------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------


@dataclass
class Seed:
    """Per-trajectory input. Endpoints in RAS millimetres.

    ``model_id`` distinguishes mode 4 (caller vouches for the electrode
    model — placement only, no library search) from mode 5 (caller knows
    where the shank is but wants library matching).
    """

    name: str
    start_ras: np.ndarray
    end_ras: np.ndarray
    model_id: str | None = None

    def __post_init__(self):
        self.start_ras = np.asarray(self.start_ras, dtype=float)
        self.end_ras = np.asarray(self.end_ras, dtype=float)


@dataclass
class PlacedTrajectory:
    """Fully-scored emission. Direct dataclass mirror of
    ``PlacementCtx.score_components`` plus the geometry the caller cares about.

    Slicer adapters convert this to MRML nodes; CLI adapters convert to TSV
    rows; QC writers serialize ``score_components`` for diagnostics.
    """

    name: str
    start_ras: np.ndarray
    end_ras: np.ndarray
    centerline_ras: np.ndarray | None
    contacts_ras: list[np.ndarray]
    model_id: str | None
    compound_score: float
    band: Literal["high", "medium", "low"]
    bolt_source: str
    bolt_end_arc_mm: float
    score_components: dict
    diagnostics: dict = field(default_factory=dict)


@dataclass
class PlacementBatch:
    """Result of ``place_seeg``.

    ``trajectories`` is in input-seed order (or candidate-emission order for
    auto modes). ``qc_dir`` is set when the caller passes ``output_dir``;
    Session 3 will populate it via ``rosa_core.qc_output.write_qc_directory``.
    """

    trajectories: list[PlacedTrajectory]
    qc_dir: Path | None = None
    diagnostics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------
# Internal: ctx → PlacedTrajectory adapter
# ---------------------------------------------------------------------


def _ctx_to_placed(name: str, ctx: PlacementCtx) -> PlacedTrajectory:
    """Convert a fully-scored ``PlacementCtx`` to a public ``PlacedTrajectory``.

    Pulls fields from ``ctx.score_components`` (the dict score stages write
    to) — extracted as a function rather than inline so the adapter is
    testable independently of running the whole pipeline.
    """
    sc = ctx.score_components
    return PlacedTrajectory(
        name=name,
        start_ras=np.asarray(ctx.seed_start, dtype=float),
        end_ras=np.asarray(ctx.seed_end, dtype=float),
        centerline_ras=(
            np.asarray(ctx.centerline, dtype=float)
            if ctx.centerline is not None else None
        ),
        contacts_ras=[np.asarray(p, dtype=float) for p in (ctx.placed_ras or [])],
        model_id=sc.get("model_id"),
        compound_score=float(sc.get("compound_score", 0.0)),
        band=sc.get("band", "low"),
        bolt_source=ctx.bolt_source,
        bolt_end_arc_mm=float(ctx.bolt_end_arc),
        score_components=dict(sc),
        diagnostics={
            "signal_kind": ctx.signal_kind,
            "n_slots": int(sc.get("n_slots", 0)),
            "n_covered": int(sc.get("n_covered", 0)),
            "fft_reliable": bool(sc.get("fft_reliable", False)),
        },
    )


# ---------------------------------------------------------------------
# Mode 4: placement only (user-supplied seeds + model_id)
# ---------------------------------------------------------------------


def _filter_library_to_model(library_models: list[dict], model_id: str | None) -> list[dict]:
    """When the caller passed ``model_id``, restrict the library to just that
    model so the matched filter has nothing else to consider.

    When ``model_id`` is None (mode 5), pass the full library through.
    """
    if model_id is None:
        return library_models
    out = [m for m in library_models if str(m.get("id") or "") == str(model_id)]
    if not out:
        # User asked for an unknown model — fall through to full library
        # rather than raising. Caller sees the actual pick in score_components.
        return library_models
    return out


def _place_mode_4(
    seeds: list[Seed], *,
    features: dict,
    library_models: list[dict],
    bolts: list[dict] | None,
    sample_fn: Callable[[PlacementCtx], PlacementCtx],
) -> list[tuple[str, PlacementCtx]]:
    """Mode 4 inner: per-seed staged placement with the user-vouched model.

    Returns list of ``(name, ctx)`` pairs in input seed order. Caller wraps
    in ``PlacedTrajectory`` via ``_ctx_to_placed``.

    No two-pass cross-shank ownership — when the caller has vouched for
    every seed, it's their responsibility to deconflict. (Two-pass is opt-in
    via mode 1/2 batch flows in Session 2.)
    """
    out: list[tuple[str, PlacementCtx]] = []
    for s in seeds:
        models = _filter_library_to_model(library_models, s.model_id)
        ctx = place_seed(
            s.start_ras, s.end_ras,
            features=features,
            library_models=models,
            bolts=bolts,
            sample_fn=sample_fn,
        )
        out.append((s.name, ctx))
    return out


# ---------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------


def place_seeg(
    ct,                                            # path | sitk.Image | (numpy_kji, ijk_to_ras)
    *,
    seeds: list[Seed] | None = None,
    expected: list[tuple[str, str]] | None = None,
    n_expected: int | None = None,
    library: str | list[dict] | None = None,
    output_dir: Path | str | None = None,
    sample_fn: Callable[[PlacementCtx], PlacementCtx] | None = None,
    apply_subject_fft_norm: bool = False,
    features: dict | None = None,
    bolts: list[dict] | None = None,
) -> PlacementBatch:
    """Single user-facing entry — see module docstring for the 5-mode table.

    Args:
        ct: CT volume. Currently the implementation requires a precomputed
            ``features`` dict; CT-from-path / CT-from-sitk loaders land in
            Session 2 alongside the mode-1 candidate-seed generator.
        seeds: per-trajectory inputs (modes 4 and 5).
        expected, n_expected: mode 2/3 inputs (NotImplementedError until
            Session 2).
        library: pitch-strategy key ("dixi", "pmt_35", ...), explicit model
            list, or None for full library.
        output_dir: when set, Session 3's ``qc_output.write_qc_directory``
            will populate it. Currently stored on the result but no files
            are written yet.
        sample_fn: stage-C swap. Default uses ``sample_neg_log_max`` (LoG-
            side dominates per the 2026-05-09 11-subject sweep).
        apply_subject_fft_norm: enable per-subject FFT p75 normalization.
            Off by default for mode 4 (single-seed placement); on by default
            for mode 1/2 batch flows in Session 2.
        features, bolts: precomputed feature dict and bolt CC list. Mode-4
            callers can pass them directly to skip CT loading. The loader-
            built ``features`` dict comes from
            ``rosa_detect.guided_fit_engine.compute_features``.

    Raises:
        NotImplementedError: modes 1, 2, 3, 5 (Session 2 work).
        ValueError: incompatible mode dispatch (e.g. seeds + expected both set).
    """
    # Mode dispatch.
    has_seeds = seeds is not None and len(seeds) > 0
    has_expected = expected is not None
    has_n_expected = n_expected is not None

    if has_expected and (has_seeds or has_n_expected):
        raise ValueError("expected= cannot be combined with seeds= or n_expected=")
    if has_n_expected and has_seeds:
        raise ValueError("n_expected= cannot be combined with seeds=")

    if has_seeds:
        seeds_have_models = all(s.model_id is not None for s in seeds)
        if seeds_have_models:
            mode = 4
        else:
            mode = 5
    elif has_n_expected:
        mode = 2
    elif has_expected:
        mode = 3
    else:
        mode = 1

    if mode != 4:
        raise NotImplementedError(
            f"place_seeg mode {mode} is implemented in Session 2 (this is "
            f"Session 1 — only mode 4 / seeded placement is wired)"
        )

    # Mode 4 implementation.
    if features is None:
        raise NotImplementedError(
            "Session 1 mode-4 requires a precomputed `features` dict from "
            "`rosa_detect.guided_fit_engine.compute_features`. CT loaders "
            "land in Session 2."
        )

    library_models = _resolve_library(library)
    sample_fn = sample_fn or _default_sample_fn()

    pairs = _place_mode_4(
        list(seeds), features=features, library_models=library_models,
        bolts=bolts, sample_fn=sample_fn,
    )

    if apply_subject_fft_norm:
        ctxs = [c for _, c in pairs]
        ctxs = apply_subject_fft_normalization(ctxs)
        pairs = [(name, c) for (name, _), c in zip(pairs, ctxs)]

    placed = [_ctx_to_placed(name, ctx) for name, ctx in pairs]

    return PlacementBatch(
        trajectories=placed,
        qc_dir=Path(output_dir) if output_dir is not None else None,
        diagnostics={
            "mode": mode,
            "n_seeds": len(seeds),
            "n_library_models": len(library_models),
            "subject_fft_normalized": apply_subject_fft_norm,
        },
    )


# ---------------------------------------------------------------------
# Helpers (private)
# ---------------------------------------------------------------------


def _resolve_library(library: str | list[dict] | None) -> list[dict]:
    """Resolve ``library`` arg to a concrete model list.

    Strings are pitch-strategy keys (dispatched through
    ``electrode_classifier.filter_models_for_strategy``); lists are taken
    verbatim; None loads the full bundled library.
    """
    if isinstance(library, list):
        return library
    from .electrode_models import load_electrode_library
    full = load_electrode_library()["models"]
    if library is None:
        return full
    if isinstance(library, str):
        from .electrode_classifier import filter_models_for_strategy
        return filter_models_for_strategy(full, library)
    raise TypeError(f"library must be str | list | None, got {type(library).__name__}")


def _default_sample_fn():
    """Default to LoG-side sampler (notebook-validated to dominate HU)."""
    from .contact_placement import sample_neg_log_max
    return sample_neg_log_max


__all__ = [
    "PlacedTrajectory",
    "PlacementBatch",
    "Seed",
    "place_seeg",
]
