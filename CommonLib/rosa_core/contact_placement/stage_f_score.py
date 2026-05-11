"""Stage F — score: simple components, CC overlap, compound + band.

Three sub-stages (called in order by the composer):

* ``score_simple`` — per-emission diagnostic dict: ``corr``, ``n_covered_frac``,
  ``bolt_zone_frac``, ``placed_hu_*``, model uniformity & margin, zone CV,
  per-segment FFT (CM/BM-aware), tube-likeness.
* ``score_cc_overlap`` — independent confidence signal: nearest global bolt CC
  centroid distance to the centerline (cropped-bolt-aware via centerline
  projection rather than seed-line projection).
* ``score_compound`` — fuse 7 components into a composite + 3-tier band
  ``{high, medium, low}``. Bolt-only-fake penalty subtracted post-sum.

The compound score IS the trajectory validator (per the user's 2026-05-09
unification: "trajectory generation becomes redundant"). Callers should
filter on ``score_components["band"]``, not on the seeder's ``confidence_label``.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..volume_sampling import sample_trilinear_at_ras
from .constants import (
    BOLT_ONLY_PENALTY_MAX,
    BOLT_ONLY_PENALTY_THRESHOLD,
    CC_OVERLAP_MAX_ARC_PAST_BOLT_MM,
    CC_OVERLAP_MAX_PERP_MM,
    CC_OVERLAP_PERP_SCALE_MM,
    COMPOUND_BANDS,
    COMPOUND_WEIGHTS,
    SEEDER_LABEL_TO_SCORE,
    WALK_STEP_MM,
)
from .context import PlacementCtx
from .polyline import project_to_polyline
from .stage_d_pick import per_model_corrs


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _model_uniform_pitch(model) -> tuple[float | None, bool]:
    """Return ``(pitch_mm, uniform)`` for a library model.

    Uniform = all consecutive contact spacings within 0.1 mm. Non-uniform
    models (CM/BM) get FFT'd per cluster instead.

    Accepts either a plain library dict or a
    ``rosa_core.matched_filter.NormalizedLibraryModel``.
    """
    from ..matched_filter import NormalizedLibraryModel
    if isinstance(model, NormalizedLibraryModel):
        offs = model.offsets_np
    else:
        offs = model.get("contact_center_offsets_from_tip_mm")
    if offs is None or len(offs) < 2:
        return None, False
    diffs = np.diff(np.sort(np.asarray(offs, dtype=float)))
    pitch = float(np.mean(diffs))
    return pitch, bool(np.all(np.abs(diffs - pitch) < 0.1))


def _zone_modulation(
    arcs: np.ndarray, sig: np.ndarray, bolt_end: float, cl_max: float,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Restrict signal to ``[bolt_end, cl_max]`` and compute (cv, ptp_mod, z_arcs, z_sig)."""
    mask = (arcs >= bolt_end) & (arcs <= cl_max)
    z_arcs = arcs[mask]
    z_sig = sig[mask]
    if z_sig.size < 4:
        return 0.0, 0.0, z_arcs, z_sig
    mean_s = float(np.mean(z_sig))
    if mean_s <= 1e-6:
        return 0.0, 0.0, z_arcs, z_sig
    cv = float(np.std(z_sig) / mean_s)
    ptp = float((np.max(z_sig) - np.min(z_sig)) / mean_s)
    return cv, ptp, z_arcs, z_sig


def _pitch_power_frac(z_arcs: np.ndarray, z_sig: np.ndarray, pitch_mm: float) -> float:
    """FFT power in a 3-bin band around f=1/pitch divided by total spectral power.

    Detrend + Hann window first; exclude DC from the total. Single-segment
    helper called by the per-segment wrapper for multi-cluster electrodes.
    """
    if z_sig.size < 16 or pitch_mm <= 0:
        return 0.0
    from scipy.signal import detrend
    step = float(z_arcs[1] - z_arcs[0]) if z_arcs.size >= 2 else WALK_STEP_MM
    sig = detrend(z_sig) * np.hanning(z_sig.size)
    spec = np.abs(np.fft.rfft(sig)) ** 2
    freqs = np.fft.rfftfreq(z_sig.size, d=step)
    target = 1.0 / pitch_mm
    if freqs[-1] < target:
        return 0.0  # signal too short to resolve the pitch frequency
    idx = int(np.argmin(np.abs(freqs - target)))
    band = float(spec[max(0, idx - 1):idx + 2].sum())
    total = float(spec[1:].sum())
    return band / total if total > 0 else 0.0


def _pitch_power_frac_per_segment(
    z_arcs: np.ndarray, z_sig: np.ndarray,
    slot_arcs: np.ndarray, pitch_mm: float, *,
    gap_thresh_mm: float = 5.0, min_slots: int = 4,
) -> tuple[float, int, int]:
    """Per-cluster FFT for CM/BM (multi-segment) electrodes.

    Splits ``slot_arcs`` by gaps > ``gap_thresh_mm`` into clusters. For each
    cluster with ≥ ``min_slots`` contacts AND ≥ 16 walker samples, computes
    pitch FFT power frac and averages across reliable clusters.

    Uniform-pitch electrodes have a single cluster → reduces to single FFT.

    Returns ``(mean_power, n_reliable_segments, n_segments_total)``.
    """
    if len(slot_arcs) < 2 or pitch_mm <= 0:
        return 0.0, 0, 0
    slots = np.sort(np.asarray(slot_arcs, dtype=float))
    diffs = np.diff(slots)
    cluster_breaks = np.where(diffs > gap_thresh_mm)[0]
    cluster_starts = np.concatenate([[0], cluster_breaks + 1])
    cluster_ends = np.concatenate([cluster_breaks, [len(slots) - 1]])
    n_segments = len(cluster_starts)

    powers = []
    for cs, ce in zip(cluster_starts, cluster_ends):
        n_in_cluster = int(ce - cs + 1)
        if n_in_cluster < min_slots:
            continue
        first = float(slots[cs]) - 0.5 * pitch_mm
        last = float(slots[ce]) + 0.5 * pitch_mm
        mask = (z_arcs >= first) & (z_arcs <= last)
        seg_arcs = z_arcs[mask]
        seg_sig = z_sig[mask]
        if seg_sig.size < 16:
            continue
        powers.append(_pitch_power_frac(seg_arcs, seg_sig, pitch_mm))

    if not powers:
        return 0.0, 0, n_segments
    return float(np.mean(powers)), len(powers), n_segments


def tube_like_frac(
    ctx: PlacementCtx, *,
    percentile: float = 90.0,
    tube_radius_mm: float = 1.0,
    perp_max_mm: float = 2.5,
    perp_step_mm: float = 0.5,
    arc_step_mm: float = 1.0,
) -> float:
    """Fraction of high-HU disk voxels (top ``100−percentile`` %) within
    ``tube_radius_mm`` of the centerline.

    Self-calibrating — no fixed HU threshold; the cutoff is the top-decile of
    *this emission's* own disk samples. Cross-scanner HU-shift safe.

    Real shanks: top-HU voxels are the contacts → all near centerline → ~1.0.
    Bone-spike / bolt-only fakes: top-HU voxels are radially scattered → ~0.3-0.5.
    Anchor-independent (works for metal AND bolt_less).
    """
    if ctx.centerline is None:
        return 0.0
    cl = np.asarray(ctx.centerline, dtype=float)
    if len(cl) < 2:
        return 0.0

    r2i = np.asarray(ctx.features["ras_to_ijk_mat"], dtype=float)
    ct_arr = ctx.features["ct_arr_kji"]
    diffs = np.diff(cl, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(cum[-1])
    zone_start = float(ctx.bolt_end_arc)
    zone_end = total
    if zone_end - zone_start < 1.0:
        return 0.0

    offsets = np.arange(-perp_max_mm, perp_max_mm + perp_step_mm, perp_step_mm)
    radials = np.sqrt(offsets[:, None] ** 2 + offsets[None, :] ** 2).ravel()

    from ..volume_sampling import sample_trilinear_batch
    arcs = np.arange(zone_start, zone_end, arc_step_mm)

    all_radials = []
    all_hus = []
    for arc in arcs:
        i = int(np.searchsorted(cum, arc, side="right") - 1)
        i = max(0, min(i, len(diffs) - 1))
        t_frac = (arc - cum[i]) / max(seg_lens[i], 1e-9)
        center = cl[i] + t_frac * diffs[i]
        tangent = diffs[i] / max(seg_lens[i], 1e-9)
        any_v = np.array([1.0, 0, 0]) if abs(tangent[0]) <= 0.9 else np.array([0, 1.0, 0])
        u = np.cross(tangent, any_v); u /= max(np.linalg.norm(u), 1e-9)
        v = np.cross(tangent, u);     v /= max(np.linalg.norm(v), 1e-9)

        pts = (center[None, None, :]
               + offsets[:, None, None] * u[None, None, :]
               + offsets[None, :, None] * v[None, None, :]).reshape(-1, 3)
        samples = sample_trilinear_batch(ct_arr, r2i, pts).ravel()
        finite = np.isfinite(samples)
        if finite.any():
            all_radials.append(radials[finite])
            all_hus.append(samples[finite])

    if not all_hus:
        return 0.0
    hus = np.concatenate(all_hus)
    rs = np.concatenate(all_radials)
    cutoff = float(np.percentile(hus, percentile))
    high = hus >= cutoff
    if not high.any():
        return 0.0
    return float(np.mean(rs[high] <= tube_radius_mm))


# ---------------------------------------------------------------------
# Score stages (composer hooks)
# ---------------------------------------------------------------------


def score_simple(ctx: PlacementCtx) -> PlacementCtx:
    """Per-emission diagnostic dict — corr, FFT, tube-likeness, modulation."""
    cl = np.asarray(ctx.centerline, dtype=float)
    cl_total = (
        float(np.linalg.norm(np.diff(cl, axis=0), axis=1).sum()) if len(cl) >= 2 else 0.0
    )
    bolt_zone_frac = ctx.bolt_end_arc / cl_total if cl_total > 0 else 0.0

    r2i = np.asarray(ctx.features["ras_to_ijk_mat"], dtype=float)
    ct_arr = ctx.features["ct_arr_kji"]
    placed_hu = []
    for p in ctx.placed_ras or []:
        try:
            v = float(sample_trilinear_at_ras(ct_arr, r2i, np.asarray(p, dtype=float)))
            if np.isfinite(v):
                placed_hu.append(v)
        except Exception:
            pass

    pmc = per_model_corrs(ctx)
    if pmc:
        top1 = pmc[0][3]
        top2 = pmc[1][3] if len(pmc) > 1 else 0.0
        rest = [t[3] for t in pmc[1:]]
        uniformity = float(np.mean(rest) / top1) if (rest and top1 > 1e-6) else 0.0
        uniformity = float(np.clip(uniformity, 0.0, 1.0))
        margin = float(top1 - top2)
    else:
        uniformity, margin = 1.0, 0.0

    cv = ptp_mod = 0.0
    pitch_power = 0.0
    pitch_mm = None
    uniform_pitch = False
    fft_n_reliable_segments = 0
    fft_n_total_segments = 0
    if ctx.walk_arcs is not None and ctx.walk_signal is not None:
        cv, ptp_mod, z_arcs, z_sig = _zone_modulation(
            ctx.walk_arcs, ctx.walk_signal, ctx.bolt_end_arc, cl_total,
        )
        m = ctx.match
        target_id = m.best_model_id if m else ""
        from ..matched_filter import NormalizedLibraryModel as _NLM
        picked = next(
            (mm for mm in ctx.library_models
             if (mm.model_id if isinstance(mm, _NLM) else str(mm.get("id") or "")) == target_id),
            None,
        )
        if picked is not None:
            pitch_mm, uniform_pitch = _model_uniform_pitch(picked)
            if (pitch_mm and z_sig.size and m is not None
                and m.slot_arcs is not None and len(m.slot_arcs) >= 2):
                pitch_power, fft_n_reliable_segments, fft_n_total_segments = (
                    _pitch_power_frac_per_segment(z_arcs, z_sig, m.slot_arcs, pitch_mm)
                )

    tube_frac = tube_like_frac(ctx) if ctx.centerline is not None else 0.0

    m = ctx.match
    cps = {
        "corr":                     float(m.corr) if m else 0.0,
        "n_slots":                  int(m.n_slots) if m else 0,
        "n_covered":                int(m.n_covered) if m else 0,
        "n_covered_frac":           float(m.n_covered) / m.n_slots if m and m.n_slots else 0.0,
        "bolt_zone_frac":           float(bolt_zone_frac),
        "bolt_source":              ctx.bolt_source,
        "placed_hu_mean":           float(np.mean(placed_hu)) if placed_hu else float("nan"),
        "placed_hu_min":            float(np.min(placed_hu)) if placed_hu else float("nan"),
        "n_placed":                 len(ctx.placed_ras or []),
        "model_id":                 m.best_model_id if m else None,
        "model_corr_uniformity":    uniformity,
        "model_corr_margin":        margin,
        "zone_cv":                  float(cv),
        "zone_ptp_mod":             float(ptp_mod),
        "pitch_power_frac":         float(pitch_power),
        "fft_n_segments":           int(fft_n_total_segments),
        "fft_n_reliable_segments":  int(fft_n_reliable_segments),
        "pitch_mm":                 float(pitch_mm) if pitch_mm else None,
        "model_uniform_pitch":      bool(uniform_pitch),
        "tube_like_frac":           float(tube_frac),
        "per_model_corr":           pmc,  # stashed for renderer + comparison df
    }
    return replace(ctx, score_components=cps)


def score_cc_overlap(ctx: PlacementCtx) -> PlacementCtx:
    """Project bolt CCs onto the centerline; nearest in-zone CC gives an
    overlap score in ``[0, 1]``.

    Centerline-projection (not seed-line projection) drops correctly to the
    real CC even when the trajectory bends. T18/X03 was the diagnostic case:
    its bolt CC sits at seed-line u=39mm but on the snapped centerline its
    arc-along is ~22mm with perp~4mm.
    """
    cps = dict(ctx.score_components)
    cps.update({
        "cc_match": None, "cc_dist_mm": float("nan"),
        "cc_n_voxels": 0, "cc_overlap_score": 0.0,
    })
    if not ctx.bolts:
        return replace(ctx, score_components=cps)

    cl = (
        np.asarray(ctx.centerline, dtype=float)
        if ctx.centerline is not None
        else np.vstack([ctx.seed_start, ctx.seed_end])
    )
    cl_total = (
        float(np.linalg.norm(np.diff(cl, axis=0), axis=1).sum())
        if len(cl) >= 2 else 0.0
    )
    arc_max = (
        float(ctx.bolt_end_arc) + CC_OVERLAP_MAX_ARC_PAST_BOLT_MM
        if ctx.bolt_source == "metal" else cl_total
    )

    best_id, best_dist, best_n, best_arc = None, float("inf"), 0, float("nan")
    for b in ctx.bolts:
        pts = np.asarray(b.get("pts_ras"), dtype=float)
        if pts.size == 0:
            continue
        center = pts.mean(axis=0)
        arc_along, perp = project_to_polyline(center, cl)
        if arc_along > arc_max:
            continue
        if perp > CC_OVERLAP_MAX_PERP_MM:
            continue
        if perp < best_dist:
            best_dist = perp
            best_id = b.get("id")
            best_n = int(b.get("n_vox") or 0)
            best_arc = arc_along

    if best_id is not None:
        overlap = max(0.0, 1.0 - best_dist / CC_OVERLAP_PERP_SCALE_MM)
        cps.update({
            "cc_match":         best_id,
            "cc_dist_mm":       best_dist,
            "cc_arc_along_mm":  best_arc,
            "cc_n_voxels":      best_n,
            "cc_overlap_score": float(overlap),
        })
    return replace(ctx, score_components=cps)


def score_compound(ctx: PlacementCtx, *, fft_norm: float | None = None) -> PlacementCtx:
    """Compose 7-component weighted sum + ``{high, medium, low}`` band.

    Args:
        fft_norm: optional override for ``s_fft`` (per-subject normalization
            post-pass passes subject-relative value).

    Cropped-bolt fix: when ``bolt_source=="metal"`` but ``cc_overlap_score==0``
    (CC out of FOV), substitute neutral 0.5 instead of 0. Walker presence is
    sufficient bolt evidence.

    Reliability gate: ``fft_n_reliable_segments >= 1`` (per-segment FFT).
    When unreliable (no segment had ≥4 slots and ≥16 walker samples), s_fft
    defaults to neutral 0.5.
    """
    sc = ctx.score_components
    bz = sc.get("bolt_zone_frac", 0.0)
    fft_reliable = int(sc.get("fft_n_reliable_segments", 0)) >= 1

    if not fft_reliable:
        s_fft = 0.5
    elif fft_norm is not None:
        s_fft = float(np.clip(fft_norm, 0.0, 1.0))
    else:
        s_fft = float(np.clip(sc.get("pitch_power_frac", 0.0), 0.0, 1.0))

    raw_cc = float(np.clip(sc.get("cc_overlap_score", 0.0), 0.0, 1.0))
    walker_metal = sc.get("bolt_source") == "metal"
    if walker_metal and raw_cc == 0.0:
        s_cc = 0.5  # cropped or out-of-FOV: walker compensates
    else:
        s_cc = raw_cc

    sub = {
        "s_corr":       float(np.clip(sc.get("corr", 0.0), 0.0, 1.0)),
        "s_fft":        s_fft,
        "s_tube":       float(np.clip(sc.get("tube_like_frac", 0.0), 0.0, 1.0)),
        "s_margin":     float(np.clip(sc.get("model_corr_margin", 0.0) / 0.15, 0.0, 1.0)),
        "s_walker":     1.0 if walker_metal else 0.0,
        "s_cc_overlap": s_cc,
        "s_seeder":     SEEDER_LABEL_TO_SCORE.get(ctx.seeder_label, 0.5),
    }

    if bz > BOLT_ONLY_PENALTY_THRESHOLD and sub["s_fft"] < 0.5:
        bolt_only_penalty = BOLT_ONLY_PENALTY_MAX * min(
            1.0,
            (bz - BOLT_ONLY_PENALTY_THRESHOLD) / (1.0 - BOLT_ONLY_PENALTY_THRESHOLD),
        )
    else:
        bolt_only_penalty = 0.0

    composite = sum(
        COMPOUND_WEIGHTS[k] * sub[f"s_{k}"]
        for k in ("corr", "fft", "tube", "margin", "walker", "cc_overlap", "seeder")
    ) - bolt_only_penalty
    composite = float(np.clip(composite, 0.0, 1.0))
    band = (
        "high"   if composite >= COMPOUND_BANDS["high"]
        else "medium" if composite >= COMPOUND_BANDS["medium"]
        else "low"
    )

    cps = dict(sc)
    cps["compound_score"] = composite
    cps["band"] = band
    cps["fft_reliable"] = fft_reliable
    cps["subscores"] = {**sub, "bolt_only_penalty": float(bolt_only_penalty)}
    return replace(ctx, score_components=cps)


__all__ = [
    "score_cc_overlap",
    "score_compound",
    "score_simple",
    "tube_like_frac",
]
