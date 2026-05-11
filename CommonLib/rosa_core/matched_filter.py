"""Matched-filter library scoring for SEEG contact placement.

Pearson cross-correlation between a 1D signal sampled along a centerline
arc and a sum-of-Gaussians template at the candidate model's slot
positions. This replaces the heuristic ``template_match_signal`` /
``template_match_peaks`` matchers (slot floor, baseline percentile,
``_better`` tie-break, etc.) with a single physical knob:
``sigma_contact_mm`` ≈ contact half-length (~1.0 mm: FWHM=2.355σ ≈
2.4 mm = one full contact along the axis).

Validated on AMC88+T22 (matched_filter_amc88_t22.ipynb): 17/17 GOOD
under the strict on-metal metric. Full-dataset (6 subjects, 62 placeable
shanks): 60-61/62 depending on signal kind; ties or beats the heavily
tuned heuristic (61/62 with deepest-peak under-reach penalty).

Bolt-less mode: when there is no real bolt CC and ``bolt_end_arc=0``,
pass ``max_tip_override`` to clamp the deepest model slot to the actual
metal extent (e.g. deepest detected signal peak). This prevents the
matcher from over-extending past the electrode tip onto bone.

The scoring formula:

    template(arc) = Σᵢ exp(-½ ((arc − slot_i) / σ_contact)²)
        for slot_i in contact zone (slot_i ≥ cutoff)
    score = Pearson(signal[in_zone], template[in_zone])

Pearson normalizes for shank-to-shank brightness automatically; the
contact-zone restriction makes bolt-zone slots a "free pass". Tie-break
on equal correlation: shorter ``n_slots`` wins (avoids longer models
quietly winning when their bolt-zone slots are zeroed).

Boundary-clean of Slicer / VTK / Qt. Caller is responsible for
sampling the disk-stat signal along the centerline; this module only
sees the 1D ``(arcs, signal)`` array and the library models.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# Default matches the validated dataset config (single physical knob).
SIGMA_CONTACT_MM_DEFAULT = 1.0


@dataclass
class MatchedFilterResult:
    """Result of a matched-filter library pick.

    ``best_model_id`` is None when no candidate scored above the empty
    sentinel (e.g. zero in-zone signal, or every candidate violated the
    bolt-zone fraction cap).

    ``per_model`` carries the per-model best (one entry per library
    model that passed the geometric gates and produced a finite
    correlation). Sorted by corr descending. Stage-D's
    ``pick_extent_aware`` re-rank reads from here instead of re-calling
    ``matched_filter_pick`` per model — same data, single pass over the
    tip grid. Each entry includes the model's own ``tip_arc`` and
    ``slot_arcs`` so a preferred-model switch can use them directly
    without another matched_filter_pick call.
    """

    best_model_id: str | None
    tip_arc: float | None
    slot_arcs: np.ndarray | None
    n_slots: int
    n_covered: int
    corr: float
    per_model: list["MatchedFilterResult"] | None = None


_EMPTY = MatchedFilterResult(
    best_model_id=None, tip_arc=None, slot_arcs=None,
    n_slots=0, n_covered=0, corr=0.0,
    per_model=None,
)


def matched_filter_pick(
    arcs: np.ndarray,
    signal: np.ndarray,
    library_models: Sequence[dict],
    *,
    bolt_end_arc: float,
    first_contact_min_mm: float = 1.0,
    profile_end_arc: float,
    max_extend_tip_mm: float = 3.0,
    tip_grid_step_mm: float = 0.25,
    sigma_contact_mm: float = SIGMA_CONTACT_MM_DEFAULT,
    max_bolt_frac: float = 0.5,
    max_tip_short_mm: float | None = 1.0,
    metal_extent_threshold_frac: float = 0.5,
    max_tip_override: float | None = None,
    tie_eps: float = 1e-6,
    add_valley_anti_template: bool = False,
    valley_anti_alpha: float = 1.0,
) -> MatchedFilterResult:
    """Pick the best library model via Pearson cross-correlation.

    Args:
        arcs: 1D arc-length sample positions along the centerline (mm).
        signal: per-arc disk-stat signal (e.g. max HU in a perpendicular
            disk). ``arcs`` and ``signal`` must have the same length.
        library_models: candidate electrode models. Each must expose
            ``id`` (str) and ``contact_center_offsets_from_tip_mm`` (sequence
            of contact offsets from the deep tip; offset 0 = tip).
        bolt_end_arc: arc of the bolt → contact transition. Slots before
            ``bolt_end_arc + first_contact_min_mm`` are bolt-zone "free
            pass" (don't enter the template).
        first_contact_min_mm: gap between bolt end and first contact.
        profile_end_arc: arc of the centerline tip (deep end).
        max_extend_tip_mm: how far past ``profile_end_arc`` the model's
            tip is allowed (slack for axis under-reach).
        tip_grid_step_mm: tip-position search granularity.
        sigma_contact_mm: Gaussian σ for the comb template (≈ contact
            half-length).
        max_bolt_frac: reject candidates with more than this fraction of
            slots in the bolt zone.
        max_tip_short_mm: reject any (model, tip-position) candidate whose
            deepest slot is more than this many mm proximal to
            ``profile_end_arc``. The deep tip is the one anchor that
            survives across user conventions (start_ras can be the bolt
            outer, the intracranial entry, or the contact-array start
            depending on the planner; end_ras consistently marks the
            implant target). Default 1.0 mm. ``None`` disables the gate
            (equivalent to the pre-2026-05 behaviour).
        metal_extent_threshold_frac: fraction of the signal's 90th-
            percentile used to define the "metal extent" along the
            centerline — the most-shallow arc where the centerline-
            sampled signal first crosses above this threshold. Slots
            that would fall proximal to the metal extent are rejected
            (would place electrode contacts into air past the back of
            the bolt). 0.5 is conservative; raise toward 1.0 to insist
            on saturated metal at every slot. Auto-disabled when
            ``max_tip_short_mm`` is None.
        max_tip_override: hard cap on the model's deepest slot arc (used
            in bolt-less mode to clamp to the actual metal extent).
        tie_eps: correlation tolerance for the hierarchical tie-break.
            When two candidates' corrs are within this band, fall back
            to (a) more n_covered slots wins, then (b) smaller n_slots
            wins. (a) prevents the matcher from picking too-short models
            on full-contact electrodes (e.g. DIXI-8AM beating DIXI-12AM
            on a 12-contact shank because both score identical Pearson
            NCC against a 12-peak signal); (b) blocks longer models
            quietly covering the same contacts plus bolt-zone phantoms.
        add_valley_anti_template: if True, subtract Gaussians at the
            geometric midpoints between consecutive in-contact slots
            from the template. Real shanks have HU peaks at slots and
            HU valleys at midpoints (wire only between contacts) — the
            anti-template rewards both alignments. Bone / surgical-clip
            chains with uniform-high HU get correlation crushed because
            high HU at midpoints fights the negative anti-template
            Gaussians. The anti-template's geometric midpoints follow
            each library model's actual contact pattern, so DIXI CM/BM
            electrodes get midpoints inside their 9-13mm cluster gaps
            (where wire-only HU separates from bone HU).
        valley_anti_alpha: weight of the anti-template (1.0 = same
            magnitude as the positive Gaussians).

    Returns:
        A ``MatchedFilterResult``. ``best_model_id is None`` indicates
        no model scored.
    """
    arcs = np.asarray(arcs, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if arcs.shape != signal.shape:
        raise ValueError(f"arcs/signal shape mismatch: {arcs.shape} vs {signal.shape}")
    if signal.size == 0 or not library_models:
        return _EMPTY
    cutoff = float(bolt_end_arc) + float(first_contact_min_mm)
    in_zone = (arcs >= cutoff) & (arcs <= float(profile_end_arc) + float(max_extend_tip_mm))
    if not in_zone.any():
        return _EMPTY
    arcs_in = arcs[in_zone]
    sig_in = signal[in_zone]
    sig_zm = sig_in - sig_in.mean()
    sig_norm = float(np.sqrt((sig_zm * sig_zm).sum()))
    if sig_norm < 1e-9:
        return _EMPTY
    inv_2sig2 = 1.0 / (2.0 * float(sigma_contact_mm) * float(sigma_contact_mm))

    # Metal extent along the centerline: the most-shallow arc where the
    # signal first crosses above (metal_extent_threshold_frac × p90).
    # Used to bound the proximal-most slot of any candidate template —
    # contacts can extend back into the bolt zone, but only as far as
    # the visible metal CC actually goes (no contacts in air past the
    # back of the bolt). Auto-disabled when the tip-coverage gate is
    # disabled — the two checks together define the "valid metal arc"
    # that contacts must inhabit.
    if max_tip_short_mm is not None and max_tip_override is None:
        finite_full = signal[np.isfinite(signal)]
        if finite_full.size:
            metal_threshold = (
                float(metal_extent_threshold_frac) * float(np.percentile(finite_full, 90))
            )
            is_metal = signal >= metal_threshold
            if is_metal.any():
                metal_start_arc = float(arcs[int(np.argmax(is_metal))])
            else:
                metal_start_arc = float("-inf")
        else:
            metal_start_arc = float("-inf")
    else:
        metal_start_arc = float("-inf")

    # Hierarchical tie-break: corr (primary) → n_covered (secondary,
    # prefer the model that explains MORE contacts when corrs tie) →
    # n_slots (tertiary, smallest electrode of equally-covering ones to
    # block excessive bolt-zone extension).
    #
    # The previous "shorter n_slots wins" rule was the ONLY tie-break
    # and caused the matcher to silently prefer DIXI-8AM over DIXI-12AM
    # on a true 12-contact electrode (both score the same Pearson NCC
    # because adding a 9th-12th aligned slot leaves the normalized
    # comb-vs-comb correlation unchanged). For a real 12-contact shank
    # we want the 12-contact model picked. Coverage as the secondary
    # criterion fixes this without regressing the original purpose:
    # when two models have equal n_covered, the smaller n_slots still
    # wins (which is what blocks an 18-contact model from quietly
    # covering 12 real contacts + 6 bolt-zone phantoms).
    best = {"corr": -np.inf, "n_covered": -1, "n_slots": float("inf")}
    per_model_best: dict[str, dict] = {}
    for m in library_models:
        offs = np.asarray(m["contact_center_offsets_from_tip_mm"], dtype=float)
        n_slots = len(offs)
        if n_slots < 2:
            continue
        offs_min = float(offs.min())
        min_tip = cutoff + offs_min
        max_tip = float(profile_end_arc) + float(max_extend_tip_mm) + offs_min
        if max_tip_override is not None:
            max_tip = min(max_tip, float(max_tip_override) + offs_min)
        if min_tip > max_tip:
            continue
        model_id = str(m.get("id") or "")
        for tip in np.arange(min_tip, max_tip + 1e-6, tip_grid_step_mm):
            slot_arcs = tip - offs
            in_contact = slot_arcs >= cutoff
            n_covered = int(in_contact.sum())
            if n_covered == 0:
                continue
            if (n_slots - n_covered) / n_slots > float(max_bolt_frac):
                continue
            # Tip-coverage gate (see ``max_tip_short_mm`` docstring).
            # The deepest slot must reach within ``max_tip_short_mm`` of
            # ``profile_end_arc``; templates that leave the deep half of
            # the trajectory unexplained are rejected here even if
            # they correlate well over the bolt-zone signal.
            #
            # Metal-extent gate: the proximal-most slot must not fall
            # before ``metal_start_arc`` — placing contacts in air past
            # the back of the bolt is physically impossible. This is
            # what stops the matcher from picking a too-long electrode
            # (e.g. DIXI-18CM on a short shank) just because its deepest
            # slot reaches the tip — its proximal slots would have to
            # sit in air.
            #
            # Skip both gates when the caller has supplied
            # ``max_tip_override`` — that's an explicit "clamp deepest
            # slot to X" instruction, conflicting with "deepest slot
            # must reach the unclamped profile_end".
            if max_tip_short_mm is not None and max_tip_override is None:
                tip_short = float(profile_end_arc) - float(slot_arcs.max())
                if tip_short > float(max_tip_short_mm):
                    continue
                if float(slot_arcs.min()) < metal_start_arc:
                    continue
            slots_cz = slot_arcs[in_contact]
            d = arcs_in[:, None] - slots_cz[None, :]
            template = np.exp(-(d * d) * inv_2sig2).sum(axis=1)
            if add_valley_anti_template and slots_cz.size >= 2:
                # Geometric midpoints between consecutive in-contact slots.
                # For DIXI CM/BM models these midpoints fall inside the
                # 9-13mm cluster gaps; for uniform-pitch PMT models they
                # sit at half-pitch between every pair.
                sorted_slots = np.sort(slots_cz)
                midpoints = 0.5 * (sorted_slots[:-1] + sorted_slots[1:])
                d_anti = arcs_in[:, None] - midpoints[None, :]
                anti = np.exp(-(d_anti * d_anti) * inv_2sig2).sum(axis=1)
                template = template - float(valley_anti_alpha) * anti
            t_zm = template - template.mean()
            t_norm = float(np.sqrt((t_zm * t_zm).sum()))
            if t_norm < 1e-9:
                continue
            corr = float((sig_zm * t_zm).sum()) / (sig_norm * t_norm)
            corr_delta = corr - best["corr"]
            if corr_delta > tie_eps:
                takes = True
            elif abs(corr_delta) <= tie_eps:
                # Tie on corr → break by n_covered (more = better),
                # then by n_slots (smaller = better) at equal coverage.
                if n_covered > best["n_covered"]:
                    takes = True
                elif n_covered == best["n_covered"] and n_slots < best["n_slots"]:
                    takes = True
                else:
                    takes = False
            else:
                takes = False
            if takes:
                best = {
                    "model_id": m["id"], "tip_arc": float(tip),
                    "slot_arcs": slot_arcs, "corr": corr,
                    "n_slots": int(n_slots), "n_covered": int(n_covered),
                }
            # Per-model best uses the same hierarchical tie-break as
            # the global best. Stores tip_arc + slot_arcs so a downstream
            # extent-aware re-rank can adopt this model's pick directly
            # without another matched_filter_pick call.
            mb = per_model_best.get(model_id)
            if mb is None or corr > mb["corr"] + tie_eps or (
                abs(corr - mb["corr"]) <= tie_eps
                and (n_covered > mb["n_covered"]
                     or (n_covered == mb["n_covered"] and n_slots < mb["n_slots"]))
            ):
                per_model_best[model_id] = {
                    "model_id": model_id, "tip_arc": float(tip),
                    "slot_arcs": slot_arcs, "corr": corr,
                    "n_slots": int(n_slots), "n_covered": int(n_covered),
                }
    per_model_results = sorted(
        (
            MatchedFilterResult(
                best_model_id=mb["model_id"], tip_arc=mb["tip_arc"],
                slot_arcs=mb["slot_arcs"], n_slots=mb["n_slots"],
                n_covered=mb["n_covered"], corr=mb["corr"], per_model=None,
            )
            for mb in per_model_best.values()
        ),
        key=lambda r: -r.corr,
    ) or None
    if "model_id" not in best:
        return MatchedFilterResult(
            best_model_id=None, tip_arc=None, slot_arcs=None,
            n_slots=0, n_covered=0, corr=0.0,
            per_model=per_model_results,
        )
    return MatchedFilterResult(
        best_model_id=best["model_id"],
        tip_arc=best["tip_arc"],
        slot_arcs=best["slot_arcs"],
        n_slots=best["n_slots"],
        n_covered=best["n_covered"],
        corr=best["corr"],
        per_model=per_model_results,
    )


__all__ = [
    "MatchedFilterResult",
    "SIGMA_CONTACT_MM_DEFAULT",
    "matched_filter_pick",
]
