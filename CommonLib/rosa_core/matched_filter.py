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
    """

    best_model_id: str | None
    tip_arc: float | None
    slot_arcs: np.ndarray | None
    n_slots: int
    n_covered: int
    corr: float


_EMPTY = MatchedFilterResult(
    best_model_id=None, tip_arc=None, slot_arcs=None,
    n_slots=0, n_covered=0, corr=0.0,
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
        max_tip_override: hard cap on the model's deepest slot arc (used
            in bolt-less mode to clamp to the actual metal extent).
        tie_eps: correlation tolerance for the shorter-wins-on-tie rule.
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

    best = {"corr": -np.inf, "n_slots": float("inf")}
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
        for tip in np.arange(min_tip, max_tip + 1e-6, tip_grid_step_mm):
            slot_arcs = tip - offs
            in_contact = slot_arcs >= cutoff
            n_covered = int(in_contact.sum())
            if n_covered == 0:
                continue
            if (n_slots - n_covered) / n_slots > float(max_bolt_frac):
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
            if corr > best["corr"] + tie_eps:
                takes = True
            elif abs(corr - best["corr"]) <= tie_eps and n_slots < best["n_slots"]:
                takes = True
            else:
                takes = False
            if takes:
                best = {
                    "model_id": m["id"], "tip_arc": float(tip),
                    "slot_arcs": slot_arcs, "corr": corr,
                    "n_slots": int(n_slots), "n_covered": int(n_covered),
                }
    if "model_id" not in best:
        return _EMPTY
    return MatchedFilterResult(
        best_model_id=best["model_id"],
        tip_arc=best["tip_arc"],
        slot_arcs=best["slot_arcs"],
        n_slots=best["n_slots"],
        n_covered=best["n_covered"],
        corr=best["corr"],
    )


__all__ = [
    "MatchedFilterResult",
    "SIGMA_CONTACT_MM_DEFAULT",
    "matched_filter_pick",
]
