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

All five modes are live. Modes 1/2/3/5 trigger candidate-seed
extraction via ``rosa_detect.candidate_seeds.orchestrator``;
mode 4 takes the user's seeds + vouched ``model_id`` and snaps them
to whatever the auto-detect emitted (then re-runs placement against
the vouched template).

The result is always a ``PlacementBatch`` carrying ``PlacedTrajectory``
records (band-classified, score-componented) plus optional QC
directory output (``rosa_core.qc_output.write_qc_directory``).
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
    run_two_pass,
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

    ``seeder_*`` fields carry optional per-seed metadata that the placer's
    compound score reads (s_seeder weight = 0.10). ``generate_candidate_seeds``
    populates them from v1's stage1 confidence; user-supplied seeds (mode 4/5)
    typically leave them at defaults (neutral 0.5 in s_seeder).
    """

    name: str
    start_ras: np.ndarray
    end_ras: np.ndarray
    model_id: str | None = None
    seeder_label: str = ""               # "high" | "medium" | "low" | ""
    seeder_confidence: float = 0.0
    seeder_model: str | None = None

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
    ``rosa_core.qc_output.write_qc_directory`` is the writer that
    populates it.

    ``features`` / ``bolts`` are the precomputed inputs the placer used
    internally — exposed on the batch so downstream callers (figure
    renderers, QC writers, post-pass analyses) don't have to reload the
    CT just to get the LoG / Frangi / hull-distance arrays back. The
    feature dict can be 50-100 MB so callers that don't need it should
    not hold a reference to the batch indefinitely; the placer itself
    drops its references after building the batch so the only lingering
    holders are whatever the caller does with the returned object.
    """

    trajectories: list[PlacedTrajectory]
    qc_dir: Path | None = None
    diagnostics: dict = field(default_factory=dict)
    features: dict | None = None
    bolts: list[dict] | None = None


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

    Raises ValueError when ``model_id`` is not in ``library_models`` —
    a vouched model that isn't available is a real input error, not a
    silent fallback. Caller should catch this and surface it per-seed
    rather than scoring against an unintended template.
    """
    if model_id is None:
        return library_models
    out = [m for m in library_models if str(m.get("id") or "") == str(model_id)]
    if not out:
        available = sorted({str(m.get("id") or "") for m in library_models})
        raise ValueError(
            f"vouched model_id {model_id!r} is not in the active library "
            f"({len(library_models)} models: {', '.join(available[:10])}"
            f"{'…' if len(available) > 10 else ''}). "
            f"Pass a different `library=` strategy or correct the seed."
        )
    return out


def _place_mode_4(
    seeds: list[Seed], *,
    features: dict,
    bolts: list[dict] | None,
    library_models: list[dict],
    pitch_strategy: str | None,
    sample_fn: Callable[[PlacementCtx], PlacementCtx],
    progress_logger,
    angle_tol_deg: float,
    perp_tol_mm: float,
) -> tuple[list[tuple[str, PlacementCtx]], dict]:
    """Mode 4 inner: snap to v1 emissions then force the user's model.

    Per user feedback (2026-05-09): a user-clicked seed in mode 4 might
    be reversed, off-axis, or shorter than the bolt-to-tip span — but
    the user has additionally vouched for the electrode model. The snap
    inherits v1's bolt-anchored geometry (canonical direction, full
    contact zone), then we re-place with the library filtered to the
    vouched model.

    Algorithm:
      1. Run candidate generation + two-pass placement (same as mode 5).
      2. Snap each user seed to its closest v1 candidate (axis tolerance).
      3. For each snapped pair: if the candidate's matched-filter pick
         already equals the vouched model_id, return that ctx as-is
         (already correctly scored). Otherwise re-run ``place_seed`` on
         the candidate's bolt-anchored geometry (``ctx.seed_start``,
         ``ctx.seed_end``) with library filtered to ``seed.model_id`` —
         constraints the matched filter to the vouched template.
      4. For seeds with no snap (v1 missed): fall back to per-seed
         ``place_seed`` on the user's raw seed with library filtered.
         The seed may still place even if v1 missed — caller said "this
         is where the shank is".

    Diagnostics surface match counts + per-seed mode (snap vs fallback)
    so callers see when the snap path triggered.
    """
    auto_pairs = _run_auto(
        features=features, bolts=bolts, library_models=library_models,
        pitch_strategy=pitch_strategy, sample_fn=sample_fn,
        progress_logger=progress_logger,
    )
    matches = _greedy_axis_match(
        seeds, auto_pairs,
        angle_tol_deg=angle_tol_deg, perp_tol_mm=perp_tol_mm,
    )
    seed_to_cand = dict(matches)  # seed_idx → candidate_idx

    out: list[tuple[str, PlacementCtx]] = []
    per_seed_outcome: list[dict] = []
    n_snapped_passthrough = 0
    n_snapped_re_placed = 0
    n_fallback = 0
    for si, s in enumerate(seeds):
        cand_idx = seed_to_cand.get(si)
        if cand_idx is not None:
            _name, cand_ctx = auto_pairs[cand_idx]
            picked = cand_ctx.score_components.get("model_id")
            if picked == s.model_id:
                ctx = cand_ctx
                outcome = "snap_passthrough"
                n_snapped_passthrough += 1
            else:
                # Re-run with library filtered to vouched model.
                ctx = place_seed(
                    cand_ctx.seed_start, cand_ctx.seed_end,
                    features=features,
                    library_models=_filter_library_to_model(library_models, s.model_id),
                    bolts=bolts,
                    sample_fn=sample_fn,
                    seeder_label=s.seeder_label,
                    seeder_confidence=s.seeder_confidence,
                    seeder_model=s.seeder_model,
                )
                outcome = "snap_re_placed"
                n_snapped_re_placed += 1
        else:
            ctx = place_seed(
                s.start_ras, s.end_ras,
                features=features,
                library_models=_filter_library_to_model(library_models, s.model_id),
                bolts=bolts,
                sample_fn=sample_fn,
                seeder_label=s.seeder_label,
                seeder_confidence=s.seeder_confidence,
                seeder_model=s.seeder_model,
            )
            outcome = "fallback_per_seed"
            n_fallback += 1
        out.append((s.name, ctx))
        per_seed_outcome.append({"name": s.name, "outcome": outcome})

    diag = {
        "n_input_seeds": len(seeds),
        "n_candidates_generated": len(auto_pairs),
        "n_snapped_passthrough": n_snapped_passthrough,
        "n_snapped_re_placed": n_snapped_re_placed,
        "n_fallback_per_seed": n_fallback,
        "snap_angle_tol_deg": angle_tol_deg,
        "snap_perp_tol_mm": perp_tol_mm,
        "per_seed_outcome": per_seed_outcome,
    }
    return out, diag


def _place_mode_5(
    seeds: list[Seed], *,
    features: dict,
    bolts: list[dict] | None,
    library_models: list[dict],
    pitch_strategy: str | None,
    sample_fn: Callable[[PlacementCtx], PlacementCtx],
    progress_logger,
    angle_tol_deg: float,
    perp_tol_mm: float,
) -> tuple[list[tuple[str, PlacementCtx]], dict]:
    """Mode 5 inner: snap user seeds to mode-1 candidate emissions.

    Algorithm (per user 2026-05-09 + 2026-05-10 follow-up):
      1. Run candidate generation + two-pass placement.
      2. Snap each user seed to its closest v1 candidate within tolerance.
      3. Per snapped pair → use the candidate's already-placed ctx as-is.
      4. Per unsnapped seed (no v1 candidate within tolerance) → fall
         back to per-seed ``place_seed`` on the user's raw seed with the
         full library. The user said "place contacts on these seeds";
         if v1 missed (planned-vs-actual drift on imported ROSA, etc.),
         we still try.

    Inheriting v1's bolt-anchored geometry on the snap path matches
    notebook parity (T18 13/13 picks). The fallback path matches mode 4's
    behavior — never silently drop a user seed.

    Returns ``([(name, ctx)], diag_dict)``. The diag reports the snap
    counts + the fallback count so callers can see the breakdown.
    """
    auto_pairs = _run_auto(
        features=features, bolts=bolts, library_models=library_models,
        pitch_strategy=pitch_strategy, sample_fn=sample_fn,
        progress_logger=progress_logger,
    )
    n_candidates = len(auto_pairs)
    matches = _greedy_axis_match(
        seeds, auto_pairs,
        angle_tol_deg=angle_tol_deg, perp_tol_mm=perp_tol_mm,
    )
    seed_to_cand = dict(matches)  # seed_idx → candidate_idx
    matched_cand_indices = {ci for _si, ci in matches}

    out: list[tuple[str, PlacementCtx]] = []
    n_snapped = 0
    n_fallback = 0
    per_seed_outcome: list[dict] = []
    for si, s in enumerate(seeds):
        cand_idx = seed_to_cand.get(si)
        if cand_idx is not None:
            _cand_name, ctx = auto_pairs[cand_idx]
            outcome = "snap"
            n_snapped += 1
        else:
            ctx = place_seed(
                s.start_ras, s.end_ras,
                features=features,
                library_models=library_models,
                bolts=bolts,
                sample_fn=sample_fn,
                seeder_label=s.seeder_label,
                seeder_confidence=s.seeder_confidence,
                seeder_model=s.seeder_model,
            )
            outcome = "fallback_per_seed"
            n_fallback += 1
        out.append((s.name, ctx))
        per_seed_outcome.append({"name": s.name, "outcome": outcome})

    diag = {
        "n_input_seeds": len(seeds),
        "n_candidates_generated": n_candidates,
        "n_seeds_snapped": n_snapped,
        "n_seeds_fallback": n_fallback,
        "n_candidates_dropped": n_candidates - len(matched_cand_indices),
        "snap_angle_tol_deg": angle_tol_deg,
        "snap_perp_tol_mm": perp_tol_mm,
        "per_seed_outcome": per_seed_outcome,
    }
    return out, diag


def _greedy_axis_match(
    seeds: list[Seed],
    candidates: list[tuple[str, PlacementCtx]],
    *,
    angle_tol_deg: float,
    perp_tol_mm: float,
) -> list[tuple[int, int]]:
    """Greedy match: each seed claims the closest candidate within tolerance.

    Returns list of ``(seed_idx, candidate_idx)`` pairs in cost-ascending
    order. Reciprocal: each seed and each candidate appear at most once.
    Same algorithm the integration tests use for GT axis matching.
    """
    if not seeds or not candidates:
        return []

    pairs = []
    for si, s in enumerate(seeds):
        ax_s = _unit_axis(s.end_ras - s.start_ras)
        mid_s = 0.5 * (s.start_ras + s.end_ras)
        for ci, (_name, ctx) in enumerate(candidates):
            ax_c = _unit_axis(ctx.seed_end - ctx.seed_start)
            mid_c = 0.5 * (ctx.seed_start + ctx.seed_end)
            ang = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(ax_s, ax_c)))))))
            v = mid_s - mid_c
            perp = v - float(np.dot(v, ax_c)) * ax_c
            d_perp = float(np.linalg.norm(perp))
            if ang <= angle_tol_deg and d_perp <= perp_tol_mm:
                pairs.append((ang + d_perp, si, ci))

    pairs.sort(key=lambda p: p[0])
    used_s, used_c = set(), set()
    out = []
    for _cost, si, ci in pairs:
        if si in used_s or ci in used_c:
            continue
        used_s.add(si); used_c.add(ci)
        out.append((si, ci))
    return out


def _unit_axis(v: np.ndarray) -> np.ndarray:
    """Unit vector with safe fallback for zero vectors."""
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def _place_two_pass(
    seeds: list[Seed], *,
    features: dict,
    library_models: list[dict],
    bolts: list[dict] | None,
    sample_fn: Callable[[PlacementCtx], PlacementCtx],
) -> list[tuple[str, PlacementCtx]]:
    """Run cross-shank-aware two-pass placement on a seed batch.

    Used by modes 1, 2, 3 — anywhere the seeds came from the same auto-detect
    pass, since two-pass ownership masking helps with passing-shank artifacts
    (memory: project_v3_staged_scoring_2026-05-09 cross-shank notes).
    """
    seed_dicts = [
        {
            "start_ras": s.start_ras,
            "end_ras":   s.end_ras,
            "confidence":       s.seeder_confidence,
            "confidence_label": s.seeder_label,
            "electrode_model":  s.seeder_model,
        }
        for s in seeds
    ]
    ctxs = run_two_pass(
        seed_dicts,
        features=features,
        library_models=library_models,
        bolts=bolts,
        sample_fn=sample_fn,
    )
    return [(s.name, c) for s, c in zip(seeds, ctxs)]


# ---------------------------------------------------------------------
# Mode-3 (named, no seed) — assignment by best matched-filter pick
# ---------------------------------------------------------------------


def _assign_by_expected(
    pairs: list[tuple[str, PlacementCtx]],
    expected: list[tuple[str, str]],
) -> tuple[list[tuple[str, PlacementCtx]], dict]:
    """Greedy assignment: each ``(name, model_id)`` in ``expected`` claims
    the unused candidate with highest compound score whose pick matches.

    Returns ``([(name, ctx)], diag)``:
      * The list pairs each requested name with the chosen candidate (or is
        omitted when no candidates remain).
      * ``diag`` carries the duplicate-model warning and per-name match
        outcome for QC inspection.

    **Duplicate-model limitation** (per user 2026-05-09): if two entries in
    ``expected`` share the same ``model_id`` (e.g. ``L1=DIXI-15CM`` and
    ``L2=DIXI-15CM``), there's no signal in the CT alone that distinguishes
    which physical shank goes to which name. The greedy pass assigns the
    highest-scoring matching candidate to the first entry, second-highest
    to the next, etc. — but the assignment among same-model entries is
    arbitrary. Caller should disambiguate via mode 4 (seed + model) or
    mode 5 (seed) when shank-to-name correspondence matters. The diagnostic
    surfaces a warning when duplicates are present.
    """
    from collections import Counter

    by_score = sorted(
        enumerate(pairs),
        key=lambda ip: -float(ip[1][1].score_components.get("compound_score", 0.0)),
    )

    model_counts = Counter(m for _, m in expected)
    duplicate_models = {m: c for m, c in model_counts.items() if c > 1}

    assigned_indices: set[int] = set()
    out: list[tuple[str, PlacementCtx]] = []
    per_name_outcome: list[dict] = []
    for name, want_model in expected:
        chosen_idx = None
        outcome = "matched_by_model"
        for idx, (_, ctx) in by_score:
            if idx in assigned_indices:
                continue
            picked = ctx.score_components.get("model_id")
            if picked == want_model:
                chosen_idx = idx
                break
        if chosen_idx is None:
            # No candidate picked this model — fall back to highest-score
            # unassigned candidate (caller sees ``model_id`` mismatch in
            # ``score_components``).
            outcome = "fallback_no_model_match"
            for idx, _ in by_score:
                if idx not in assigned_indices:
                    chosen_idx = idx
                    break
        if chosen_idx is None:
            per_name_outcome.append(
                {"name": name, "want_model": want_model, "outcome": "no_candidate_left"},
            )
            continue
        assigned_indices.add(chosen_idx)
        _, ctx = pairs[chosen_idx]
        out.append((name, ctx))
        per_name_outcome.append(
            {"name": name, "want_model": want_model, "outcome": outcome},
        )

    diag = {
        "n_expected": len(expected),
        "n_emitted": len(out),
        "duplicate_models": duplicate_models,
        "per_name_outcome": per_name_outcome,
    }
    if duplicate_models:
        import warnings as _warnings
        _warnings.warn(
            f"place_seeg mode 3: duplicate model_ids in expected — assignment "
            f"among same-model entries is arbitrary (CT alone cannot "
            f"distinguish identical-model shanks). Affected models: "
            f"{duplicate_models}. Use mode 4/5 with seeds when shank-to-name "
            f"correspondence matters.",
            stacklevel=3,
        )
    return out, diag


# ---------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------


def place_seeg(
    ct,                                            # path | sitk.Image | None (when features= passed)
    *,
    seeds: list[Seed] | None = None,
    expected: list[tuple[str, str]] | None = None,
    n_expected: int | None = None,
    library: str | list[dict] | None = None,
    pitch_strategy: str | None = None,
    output_dir: Path | str | None = None,
    sample_fn: Callable[[PlacementCtx], PlacementCtx] | None = None,
    apply_subject_fft_norm: bool | None = None,
    features: dict | None = None,
    bolts: list[dict] | None = None,
    band_floor: str | None = None,
    snap_angle_tol_deg: float = 12.0,
    snap_perp_tol_mm: float = 8.0,
    progress_logger=None,
) -> PlacementBatch:
    """Single user-facing entry — see module docstring for the 5-mode table.

    Args:
        ct: CT volume — path, ``SimpleITK.Image``, or ``None`` when
            ``features`` is precomputed. The loader is invoked lazily; passing
            both ``ct`` and ``features`` skips loading.
        seeds: per-trajectory inputs (modes 4 and 5).
        expected: mode-3 input — list of ``(name, model_id)`` tuples to find.
        n_expected: mode-2 input — expected total shank count.
        library: pitch-strategy key (``"dixi"``, ``"pmt_35"``, ...), explicit
            model list, or ``None`` for the full bundled library.
        pitch_strategy: override for the candidate-seed generator's pitch
            strategy (modes 1/2/3). Defaults to ``library`` when it's a
            string, else ``None`` (auto).
        output_dir: when set, Session 3's ``qc_output.write_qc_directory``
            will populate it. Currently stored on the result; no files are
            written yet.
        sample_fn: stage-C swap. Default ``sample_neg_log_max`` (LoG-side
            dominates per the 2026-05-09 11-subject sweep).
        apply_subject_fft_norm: enable per-subject FFT p75 normalization.
            Defaults: ``False`` for modes 4/5 (per-seed); ``True`` for modes
            1/2/3 (batch). Pass explicit ``True``/``False`` to override.
        features, bolts: precomputed inputs from
            ``rosa_core.volume_loader.load_features_and_bolts`` — pass them
            to skip CT loading.
        band_floor: optional ``"high"`` / ``"medium"`` / ``"low"`` filter
            applied to the output (mode 1 default: ``"medium"``; modes 2/3
            ignore — they have an explicit selection rule; modes 4/5 keep
            everything by default).
        progress_logger: callable forwarded to the candidate-seed generator
            in modes 1/2/3.

    Raises:
        ValueError: incompatible mode dispatch (e.g. ``seeds=`` + ``expected=``).
        TypeError: ``library`` of unsupported type.
        FileNotFoundError: ``ct`` is a non-existent path.
    """
    mode = _dispatch_mode(seeds, expected, n_expected)

    library_models = _resolve_library(library)
    if pitch_strategy is None and isinstance(library, str):
        pitch_strategy = library
    sample_fn = sample_fn or _default_sample_fn()
    if apply_subject_fft_norm is None:
        # Modes 1/2/3/5 generate candidates internally → subject FFT
        # normalization is appropriate (the candidate batch IS the subject's
        # full shank set). Mode 4 also runs candidate generation under the
        # hood (snap-then-force-model), so it benefits too.
        apply_subject_fft_norm = mode in (1, 2, 3, 4, 5)

    if features is None or bolts is None:
        if ct is None:
            raise ValueError(
                "place_seeg needs either a CT (path | SimpleITK.Image) or both "
                "`features` and `bolts` precomputed."
            )
        from .volume_loader import load_features_and_bolts
        features, bolts = load_features_and_bolts(ct)

    if mode == 1:
        pairs = _run_auto(
            features=features, bolts=bolts, library_models=library_models,
            pitch_strategy=pitch_strategy, sample_fn=sample_fn,
            progress_logger=progress_logger,
        )
        if apply_subject_fft_norm:
            pairs = _apply_fft_norm(pairs)
        if band_floor is None:
            band_floor = "medium"
        pairs = _filter_by_band(pairs, band_floor)
    elif mode == 2:
        pairs = _run_auto(
            features=features, bolts=bolts, library_models=library_models,
            pitch_strategy=pitch_strategy, sample_fn=sample_fn,
            progress_logger=progress_logger,
        )
        if apply_subject_fft_norm:
            pairs = _apply_fft_norm(pairs)
        pairs = sorted(
            pairs,
            key=lambda p: -float(p[1].score_components.get("compound_score", 0.0)),
        )[:int(n_expected)]
    elif mode == 3:
        pairs = _run_auto(
            features=features, bolts=bolts, library_models=library_models,
            pitch_strategy=pitch_strategy, sample_fn=sample_fn,
            progress_logger=progress_logger,
        )
        if apply_subject_fft_norm:
            pairs = _apply_fft_norm(pairs)
        pairs, mode3_diag = _assign_by_expected(pairs, list(expected))
    elif mode == 4:
        pairs, mode4_diag = _place_mode_4(
            list(seeds),
            features=features, bolts=bolts,
            library_models=library_models,
            pitch_strategy=pitch_strategy,
            sample_fn=sample_fn, progress_logger=progress_logger,
            angle_tol_deg=snap_angle_tol_deg,
            perp_tol_mm=snap_perp_tol_mm,
        )
        if apply_subject_fft_norm:
            pairs = _apply_fft_norm(pairs)
    else:  # mode == 5
        # Run candidates first (apply_subject_fft_norm applies internally),
        # then snap user seeds to closest mode-1 emission. Drop on both sides
        # of the snap (unmatched seeds + unmatched candidates).
        pairs, mode5_diag = _place_mode_5(
            list(seeds),
            features=features, bolts=bolts,
            library_models=library_models,
            pitch_strategy=pitch_strategy,
            sample_fn=sample_fn, progress_logger=progress_logger,
            angle_tol_deg=snap_angle_tol_deg,
            perp_tol_mm=snap_perp_tol_mm,
        )
        if apply_subject_fft_norm:
            pairs = _apply_fft_norm(pairs)

    placed = [_ctx_to_placed(name, ctx) for name, ctx in pairs]

    diagnostics = {
        "mode": mode,
        "n_input_seeds": len(seeds) if seeds else 0,
        "n_emitted": len(placed),
        "n_library_models": len(library_models),
        "subject_fft_normalized": apply_subject_fft_norm,
        "band_floor": band_floor,
    }
    if mode == 3:
        diagnostics["mode3"] = mode3_diag
    elif mode == 4:
        diagnostics["mode4"] = mode4_diag
    elif mode == 5:
        diagnostics["mode5"] = mode5_diag
    return PlacementBatch(
        trajectories=placed,
        qc_dir=Path(output_dir) if output_dir is not None else None,
        diagnostics=diagnostics,
        features=features,
        bolts=bolts,
    )


# ---------------------------------------------------------------------
# Dispatch + flow helpers
# ---------------------------------------------------------------------


def _dispatch_mode(
    seeds: list[Seed] | None,
    expected: list[tuple[str, str]] | None,
    n_expected: int | None,
) -> int:
    """Pick the mode from the optional-arg combination. Validates exclusivity
    AND emptiness — empty seed/expected lists or non-positive n_expected
    are treated as input errors, not silently demoted to mode 1.

    Without these checks, a caller that meant "place these specific seeds"
    would silently fall through to full auto detection when their seed
    file was empty / the expected list was filtered to nothing — masking
    the upstream bug.
    """
    if seeds is not None and len(seeds) == 0:
        raise ValueError(
            "seeds= was passed but is empty. Pass at least one Seed for "
            "mode 4/5, or pass seeds=None for auto detection."
        )
    if expected is not None and len(expected) == 0:
        raise ValueError(
            "expected= was passed but is empty. Pass at least one "
            "(name, model_id) pair for mode 3, or pass expected=None."
        )
    if n_expected is not None and int(n_expected) <= 0:
        raise ValueError(
            f"n_expected= must be a positive integer, got {n_expected!r}."
        )

    has_seeds = seeds is not None
    has_expected = expected is not None
    has_n_expected = n_expected is not None

    if has_expected and (has_seeds or has_n_expected):
        raise ValueError("expected= cannot be combined with seeds= or n_expected=")
    if has_n_expected and has_seeds:
        raise ValueError("n_expected= cannot be combined with seeds=")

    if has_seeds:
        return 4 if all(s.model_id is not None for s in seeds) else 5
    if has_n_expected:
        return 2
    if has_expected:
        return 3
    return 1


def _run_auto(
    *,
    features: dict,
    bolts: list[dict] | None,
    library_models: list[dict],
    pitch_strategy: str | None,
    sample_fn: Callable[[PlacementCtx], PlacementCtx],
    progress_logger,
) -> list[tuple[str, PlacementCtx]]:
    """Modes 1/2/3 inner: candidate generation → two-pass placement.

    The two-pass placement applies cross-shank ownership masking — relevant
    for batch auto-detect runs where multiple emissions can share voxels.
    """
    from rosa_detect.candidate_seeds import generate_candidate_seeds
    seeds = generate_candidate_seeds(
        features, bolts=bolts, pitch_strategy=pitch_strategy,
        progress_logger=progress_logger,
    )
    if not seeds:
        return []
    return _place_two_pass(
        seeds, features=features, library_models=library_models,
        bolts=bolts, sample_fn=sample_fn,
    )


def _apply_fft_norm(
    pairs: list[tuple[str, PlacementCtx]],
) -> list[tuple[str, PlacementCtx]]:
    """Re-normalize each ctx's FFT score against the subject's reliable p75."""
    if not pairs:
        return pairs
    ctxs = [c for _, c in pairs]
    ctxs = apply_subject_fft_normalization(ctxs)
    return [(name, c) for (name, _), c in zip(pairs, ctxs)]


def _filter_by_band(
    pairs: list[tuple[str, PlacementCtx]],
    band_floor: str,
) -> list[tuple[str, PlacementCtx]]:
    """Keep emissions whose band is at least ``band_floor``."""
    rank = {"low": 0, "medium": 1, "high": 2}
    threshold = rank.get(band_floor, 1)
    return [
        (n, c) for n, c in pairs
        if rank.get(c.score_components.get("band", "low"), 0) >= threshold
    ]


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
