"""Seeded shank fitting + intracranial landmark estimation.

Given a planned trajectory (start/end RAS) and a per-CT signal volume,
the seeded-fit pipeline:

1. **Walks** a tube along the planned axis sampling the signal's per-disk
   max; calls ``scipy.signal.find_peaks`` to detect contact-spaced bumps.
2. **Trims** outlier peaks (perpendicular outliers + terminal-gap outliers).
3. **Refits** the chain axis via SVD on the surviving peaks (PCA),
   re-orienting from entry toward deep tip.
4. **Arbitrates** peaks shared between neighboring chains (assigns each
   conflicted peak to whichever chain best preserves its inter-peak pitch).
5. **Walks to bolt outer edge** along ``-axis`` while the signal stays
   bright — pushes ``entry_ras`` from the most-superficial contact out to
   the physical bolt edge.

Then ``estimate_landmarks`` computes three independent inner-edge estimates
(bone width, ring metal + d(ring), Frangi) anchored on the bone-width
half-fall landmark, and returns their median as a robust combined
intracranial bolt-edge arc.

Top-level wrapper :func:`run_seeded_fit` orchestrates steps 1-5 over a
list of planned trajectories with shared arbitration; per-trajectory
landmarks are computed by :func:`estimate_landmarks`.

Ported from ``notebooks/seeded_fit/_build_starter.py`` 2026-05-18; same
algorithm as the notebook that validated S57 (16/16) and S54 (13/15
strict, 15/15 relaxed).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .centerline import (
    unit,
    orthonormal_basis_for_axis,
    otsu_threshold,
    collect_intracranial_hu_along_trajectory,
    sample_trajectory_profile,
    PROFILE_STEP_MM,
    PROFILE_PAD_TIP_MM,
)
from .volume_sampling import sample_trilinear_batch


# ---------------------------------------------------------------------
# Seeded-fit constants (per cell-1 + cell-9 of _build_starter.py).
# ---------------------------------------------------------------------

METAL_SNAP_TUBE_RADIUS_MM   = 4.0
METAL_SNAP_STEP_MM          = 0.3
METAL_SNAP_MIN_PITCH_MM     = 2.5
METAL_SNAP_MIN_N_PEAKS      = 4
SNAP_EXTEND_PAST_END_MM     = 15.0
SNAP_EXTEND_BEFORE_START_MM = 0.0
SHARED_PEAK_TOL_MM          = 1.5

FRANGI_WALK_SMOOTH_MM       = 3.0
FRANGI_WALK_MAX_MM          = 30.0
FRANGI_PAD_OUTWARD_MM       = 40.0


# ---------------------------------------------------------------------
# Result containers.
# ---------------------------------------------------------------------


@dataclass
class FittedChain:
    """One snap-fit shank.

    Fields mirror the dict ``_fit_chain_from_peaks`` produced in the
    notebook so callers can dot-access or dict-access either way (see
    :meth:`as_dict`)."""

    name: str
    axis: np.ndarray          # unit (entry -> deep tip)
    centroid: np.ndarray
    kept_pts: np.ndarray      # (N, 3) RAS, sorted shallow→deep
    kept_along: np.ndarray    # arc along the fitted axis, mean-centred
    entry_ras: np.ndarray
    tip_ras: np.ndarray
    extent_mm: float
    median_pitch_mm: float
    pitch_mode_mm: float
    n_peaks: int
    bolt_walked_mm: float = 0.0
    n_trimmed: int = 0
    n_dropped_in_arbitration: int = 0

    def as_dict(self) -> dict[str, Any]:
        return dict(
            name=self.name,
            axis=self.axis,
            centroid=self.centroid,
            kept_pts=self.kept_pts,
            kept_along=self.kept_along,
            entry_ras=self.entry_ras,
            tip_ras=self.tip_ras,
            extent_mm=self.extent_mm,
            median_pitch_mm=self.median_pitch_mm,
            pitch_mode_mm=self.pitch_mode_mm,
            n_peaks=self.n_peaks,
            bolt_walked_mm=self.bolt_walked_mm,
            n_trimmed=self.n_trimmed,
            n_dropped_in_arbitration=self.n_dropped_in_arbitration,
        )


@dataclass
class TrajectoryLandmarks:
    """Intracranial-side bolt/bone-edge landmarks for one fitted shank.

    All ``*_arc_mm`` values are arc distances from ``entry_ras`` along the
    fitted axis (positive = into brain). ``bolt_combined_arc_mm`` is the
    median of the three bolt-end estimates when at least one is defined.

    Use :attr:`win_lo` / :attr:`win_hi` directly as the intracranial window
    bounds the electrode-classifier features computation needs.
    """

    name: str
    entry_arc_mm: float         # always 0.0 by construction
    tip_arc_mm: float           # |tip − entry|
    pre_walk_arc_mm: float      # entry_arc + bolt_walked_mm
    bone_arc_mm: float | None
    bolt_rm_arc_mm: float | None
    bolt_dring_arc_mm: float | None
    bolt_frangi_arc_mm: float | None
    bolt_combined_arc_mm: float | None

    @property
    def win_lo(self) -> float:
        """Lower (bolt-side) bound for picker features — bone inner edge
        if available, otherwise 0 (entry)."""
        return float(self.bone_arc_mm) if self.bone_arc_mm is not None else 0.0

    @property
    def win_hi(self) -> float:
        """Upper (tip-side) bound for picker features — fitted tip arc."""
        return float(self.tip_arc_mm)


# ---------------------------------------------------------------------
# Small helpers.
# ---------------------------------------------------------------------


def _pitch_mode(pitches, bin_mm: float = 0.5) -> float:
    p = np.asarray(pitches, dtype=float)
    if len(p) == 0:
        return float("nan")
    if len(p) <= 2:
        return float(np.median(p))
    binned = np.round(p / bin_mm) * bin_mm
    values, counts = np.unique(binned, return_counts=True)
    if counts.max() >= 2:
        return float(values[np.argmax(counts)])
    return float(np.median(p))


def _pitch_fit_score(chain: dict, peak_idx: int) -> float:
    along = chain["kept_along"]
    pitch_mode = chain.get("pitch_mode_mm", float("nan"))
    if not np.isfinite(pitch_mode) or pitch_mode < 1.0:
        return float("inf")
    n = len(along)
    score = 0.0
    seen = 0
    if peak_idx > 0:
        score += abs((along[peak_idx] - along[peak_idx - 1]) - pitch_mode)
        seen += 1
    if peak_idx < n - 1:
        score += abs((along[peak_idx + 1] - along[peak_idx]) - pitch_mode)
        seen += 1
    return score / seen if seen else float("inf")


def _noise_std(arc, signal, mask) -> float | None:
    sig = np.asarray(signal, dtype=float)
    seg = sig[mask & np.isfinite(sig)]
    if seg.size < 5:
        return None
    return float(np.std(seg))


def _trim_perp_outliers(peak_positions, planned_start, planned_end,
                         *, max_perp_from_plan_mm: float = 3.5,
                         max_perp_from_median_mm: float = 1.5,
                         min_peaks: int = 4) -> np.ndarray:
    """Two-stage perpendicular trim. Stage 1 drops peaks > planned-axis
    distance; stage 2 drops peaks far from the median-perp position of
    the stage-1 survivors. See docstring on the notebook source for full
    rationale."""
    peak_positions = np.asarray(peak_positions, dtype=float)
    if len(peak_positions) < min_peaks:
        return peak_positions
    s = np.asarray(planned_start, dtype=float)
    e = np.asarray(planned_end,   dtype=float)
    axis_u = unit(e - s)
    rel = peak_positions - s
    along = rel @ axis_u
    perp_vec = rel - along[:, None] * axis_u
    perp_from_plan = np.linalg.norm(perp_vec, axis=1)
    keep1 = perp_from_plan <= max_perp_from_plan_mm
    if int(keep1.sum()) < min_peaks:
        return peak_positions
    survivors = peak_positions[keep1]
    perp_vec_s = perp_vec[keep1]
    perp_med = np.median(perp_vec_s, axis=0)
    dev_from_med = np.linalg.norm(perp_vec_s - perp_med, axis=1)
    keep2 = dev_from_med <= max_perp_from_median_mm
    if int(keep2.sum()) < min_peaks:
        return survivors
    return survivors[keep2]


def _trim_outlier_endpoints(peak_positions, oriented_along,
                             *, gap_factor: float = 2.0,
                             min_peaks: int = 4) -> np.ndarray:
    """Drop terminal peaks whose gap to the next-inward peak exceeds
    ``gap_factor`` × median(other pitches). Iterates from both ends."""
    peak_positions = np.asarray(peak_positions, dtype=float)
    if len(peak_positions) < min_peaks + 1:
        return peak_positions
    axis_u = unit(oriented_along)
    along  = (peak_positions - peak_positions.mean(axis=0)) @ axis_u
    order  = np.argsort(along)
    pts    = peak_positions[order]
    along  = along[order]
    while len(pts) >= min_peaks + 1:
        diffs = np.diff(along)
        if len(diffs) < 3:
            break
        ref_tail = float(np.median(diffs[:-1]))
        if np.isfinite(ref_tail) and ref_tail >= 1.0 and diffs[-1] > gap_factor * ref_tail:
            pts = pts[:-1]; along = along[:-1]
            continue
        ref_head = float(np.median(diffs[1:]))
        if np.isfinite(ref_head) and ref_head >= 1.0 and diffs[0] > gap_factor * ref_head:
            pts = pts[1:]; along = along[1:]
            continue
        break
    return pts


def _fit_chain_from_peaks(peak_positions, oriented_along) -> dict[str, Any] | None:
    """Refit an axis via PCA on the surviving peaks; reorient to point
    along ``oriented_along``; return a dict ready to wrap as a
    :class:`FittedChain`."""
    peak_positions = np.asarray(peak_positions, dtype=float)
    if len(peak_positions) < 2:
        return None
    centroid = peak_positions.mean(axis=0)
    rel = peak_positions - centroid
    if len(peak_positions) >= 3:
        _, _, Vt = np.linalg.svd(rel, full_matrices=False)
        new_axis = unit(Vt[0])
    else:
        new_axis = unit(peak_positions[-1] - peak_positions[0])
    if new_axis @ oriented_along < 0:
        new_axis = -new_axis
    rel = peak_positions - centroid
    along = rel @ new_axis
    order = np.argsort(along)
    pts_sorted = peak_positions[order]
    along_sorted = along[order]
    diffs = np.diff(along_sorted)
    return {
        "axis": new_axis,
        "centroid": centroid,
        "kept_pts": pts_sorted,
        "kept_along": along_sorted - along_sorted.mean(),
        "entry_ras": pts_sorted[0],
        "tip_ras": pts_sorted[-1],
        "extent_mm": float(along_sorted[-1] - along_sorted[0]),
        "median_pitch_mm": float(np.median(diffs)) if len(diffs) else float("nan"),
        "pitch_mode_mm": _pitch_mode(diffs),
        "n_peaks": int(len(peak_positions)),
    }


# ---------------------------------------------------------------------
# Step 1+2+3: snap-walk + trim + refit.
# ---------------------------------------------------------------------


def snap_via_signal_walk(
    planned_start, planned_end, *,
    signal_vol: np.ndarray,
    ras_to_ijk: np.ndarray,
    threshold: float,
    tube_radius_mm: float = METAL_SNAP_TUBE_RADIUS_MM,
    step_mm: float = METAL_SNAP_STEP_MM,
    min_pitch_mm: float = METAL_SNAP_MIN_PITCH_MM,
    min_n_peaks: int = METAL_SNAP_MIN_N_PEAKS,
    extend_past_end_mm: float = SNAP_EXTEND_PAST_END_MM,
    extend_before_start_mm: float = SNAP_EXTEND_BEFORE_START_MM,
) -> dict[str, Any] | None:
    """Snap a planned (start, end) RAS segment onto signal peaks.

    Walks a tube of radius ``tube_radius_mm`` along the planned axis
    sampling the per-disk max of ``signal_vol`` (typically the LoG-neg
    array — ``-log_sigma1``); detects peaks via
    ``scipy.signal.find_peaks``; recovers each peak's weighted-centroid
    RAS position via second-pass disk sampling at the peak arc; trims
    perp + endpoint outliers; refits via PCA. Returns ``None`` when fewer
    than ``min_n_peaks`` peaks survive.

    Returns a dict (the legacy "chain" shape) with the same keys as
    ``_fit_chain_from_peaks``. Wrap with :class:`FittedChain` for
    type-safe access.
    """
    from scipy.signal import find_peaks

    s = np.asarray(planned_start, dtype=float)
    e = np.asarray(planned_end,   dtype=float)
    L = float(np.linalg.norm(e - s))
    if L < 5.0:
        return None
    axis_plan = unit(e - s)
    perp1, perp2 = orthonormal_basis_for_axis(axis_plan)
    n_d = int(2 * tube_radius_mm / 0.5) + 1
    u_arr = np.linspace(-tube_radius_mm, tube_radius_mm, n_d)
    UU, VV = np.meshgrid(u_arr, u_arr, indexing="ij")
    disk_mask = (UU ** 2 + VV ** 2) <= tube_radius_mm ** 2
    disk_u = UU[disk_mask]; disk_v = VV[disk_mask]
    signal_f32 = signal_vol.astype(np.float32, copy=False)

    t_lo = -extend_before_start_mm
    t_hi = L + extend_past_end_mm
    n_samples = int((t_hi - t_lo) / step_mm) + 1
    t_arr = np.linspace(t_lo, t_hi, n_samples)
    profile = np.zeros(n_samples, dtype=float)
    for i, t in enumerate(t_arr):
        center = s + t * axis_plan
        pts = (center[None, :]
               + disk_u[:, None] * perp1[None, :]
               + disk_v[:, None] * perp2[None, :])
        vals = sample_trilinear_batch(signal_f32, ras_to_ijk, pts)
        profile[i] = float(np.nanmax(vals)) if vals.size else 0.0

    peaks, _ = find_peaks(profile, height=threshold,
                          distance=int(min_pitch_mm / step_mm))
    if len(peaks) < min_n_peaks:
        return None

    peak_positions = np.empty((len(peaks), 3), dtype=float)
    for k, peak_idx in enumerate(peaks):
        t = float(t_arr[peak_idx])
        center = s + t * axis_plan
        pts = (center[None, :]
               + disk_u[:, None] * perp1[None, :]
               + disk_v[:, None] * perp2[None, :])
        vals = sample_trilinear_batch(signal_f32, ras_to_ijk, pts)
        strong = np.isfinite(vals) & (vals >= threshold)
        if int(strong.sum()) > 0:
            w = vals[strong].astype(float)
            c_u = float((w * disk_u[strong]).sum() / w.sum())
            c_v = float((w * disk_v[strong]).sum() / w.sum())
            peak_positions[k] = center + c_u * perp1 + c_v * perp2
        else:
            peak_positions[k] = center
    n_raw = len(peak_positions)
    peak_positions = _trim_perp_outliers(peak_positions, s, e)
    peak_positions = _trim_outlier_endpoints(peak_positions, axis_plan)
    if len(peak_positions) < min_n_peaks:
        return None
    chain = _fit_chain_from_peaks(peak_positions, axis_plan)
    if chain is not None:
        chain["n_trimmed"] = int(n_raw - len(peak_positions))
    return chain


# ---------------------------------------------------------------------
# Step 4: shared-peak arbitration across chains.
# ---------------------------------------------------------------------


def arbitrate_shared_peaks(
    chains: Sequence[dict[str, Any] | None],
    *,
    tolerance_mm: float = SHARED_PEAK_TOL_MM,
    planned_axes: Sequence[np.ndarray] | None = None,
    min_peaks_after: int = 4,
) -> tuple[list[dict[str, Any] | None], int]:
    """For peaks shared between neighboring chains, keep the assignment
    that best preserves each chain's pitch_mode.

    Returns ``(new_chains, n_conflicts)``."""
    indexed = []
    for ci, chain in enumerate(chains):
        if chain is None:
            continue
        for pi, pos in enumerate(chain["kept_pts"]):
            indexed.append((ci, pi, np.asarray(pos, dtype=float)))
    n = len(indexed)
    if n == 0:
        return list(chains), 0
    used = np.zeros(n, dtype=bool); groups = []
    for i in range(n):
        if used[i]:
            continue
        group = [i]; used[i] = True
        for j in range(i + 1, n):
            if used[j] or indexed[i][0] == indexed[j][0]:
                continue
            if np.linalg.norm(indexed[i][2] - indexed[j][2]) < tolerance_mm:
                group.append(j); used[j] = True
        if len(group) > 1:
            groups.append(group)
    peaks_to_drop: dict[int, set[int]] = {}
    for group in groups:
        scored = []
        for idx in group:
            ci, pi, _ = indexed[idx]
            scored.append((_pitch_fit_score(chains[ci], pi), idx))
        scored.sort()
        for _, loser_idx in scored[1:]:
            ci, pi, _ = indexed[loser_idx]
            peaks_to_drop.setdefault(ci, set()).add(pi)
    new_chains: list[dict[str, Any] | None] = []
    for ci, chain in enumerate(chains):
        if chain is None:
            new_chains.append(None); continue
        drop = peaks_to_drop.get(ci, set())
        if not drop:
            new_chains.append(chain); continue
        keep_mask = np.array([pi not in drop for pi in range(len(chain["kept_pts"]))])
        if int(keep_mask.sum()) < min_peaks_after:
            new_chains.append(None); continue
        new_pts = np.asarray(chain["kept_pts"])[keep_mask]
        oriented = planned_axes[ci] if planned_axes is not None else chain["axis"]
        new_chain = _fit_chain_from_peaks(new_pts, oriented)
        new_chain["n_dropped_in_arbitration"] = int(len(chain["kept_pts"]) - keep_mask.sum())
        new_chains.append(new_chain)
    return new_chains, len(groups)


# ---------------------------------------------------------------------
# Step 5: bolt-outer-edge walker.
# ---------------------------------------------------------------------


def walk_to_bolt_outer_edge(
    entry, axis, *,
    signal_vol: np.ndarray,
    ras_to_ijk: np.ndarray,
    step_mm: float = 0.3,
    max_walk_mm: float = FRANGI_WALK_MAX_MM,
    signal_floor: float = 0.4,
    smooth_window_mm: float = FRANGI_WALK_SMOOTH_MM,
) -> tuple[np.ndarray, float]:
    """Slide ``entry`` along ``-axis`` (toward the skull) while
    ``signal_vol`` stays bright; stop at the bolt's outer edge.

    Recommended ``signal_vol`` is ``metal_evidence`` (=
    ``max(|LoG|, HU/2000)``) — combines HU + LoG so the walker doesn't
    halt at the wire→bolt-body cross-section flare. Returns
    ``(new_entry_ras, walked_mm)``.
    """
    from scipy.ndimage import maximum_filter1d
    entry = np.asarray(entry, dtype=float)
    axis  = unit(axis)
    sig_f32 = signal_vol.astype(np.float32, copy=False)
    n_steps = int(max_walk_mm / step_mm) + 1
    t_arr = -np.arange(n_steps + 1) * step_mm   # t=0 is entry, t<0 walks outward
    pts = entry[None, :] + t_arr[:, None] * axis[None, :]
    vals = sample_trilinear_batch(sig_f32, ras_to_ijk, pts)
    vals = np.where(np.isfinite(vals), vals, 0.0)
    win_n = max(1, int(smooth_window_mm / step_mm))
    vals_smoothed = maximum_filter1d(vals, size=win_n, mode="nearest")
    best_t = 0.0
    for k in range(1, len(vals_smoothed)):
        if vals_smoothed[k] < signal_floor:
            break
        best_t = float(t_arr[k])
    return entry + best_t * axis, abs(best_t)


# ---------------------------------------------------------------------
# Intracranial-edge landmark estimators.
# ---------------------------------------------------------------------


def _bone_inner_edge_half_fall(arc, signal, peak_exclusion_radius_mm: float = 10.0,
                                baseline_low_frac: float = 0.3) -> float | None:
    """Walk from bone-width's global peak in +arc until the signal drops
    to halfway between the peak and a peak-excluded baseline.

    Baseline is the mean of the lowest ``baseline_low_frac`` of samples
    outside ±``peak_exclusion_radius_mm`` around the peak.

    Note (2026-05-18): the peak-anchored heuristic is brittle on
    multi-peak bone profiles (trajectory crossing skull entry + a
    second deeper bone structure like skull base / paranasal sinus
    crossing — RIFG on S54 in canonical-grid features). When the
    underlying CT is on the canonical 1 mm grid the multi-peak issue
    surfaces; on native 0.5 mm grids the notebook's same algorithm
    produces stable single-peak profiles. ``fit-rosa`` skips the
    canonicalization step to match the notebook's behavior on the
    native CT grid.
    """
    sig = np.asarray(signal, dtype=float)
    arc = np.asarray(arc, dtype=float)
    finite = np.isfinite(sig)
    if not finite.any():
        return None
    peak_idx = int(np.nanargmax(sig))
    peak_val = float(sig[peak_idx])
    peak_arc = float(arc[peak_idx])
    near_peak = np.abs(arc - peak_arc) <= float(peak_exclusion_radius_mm)
    baseline_mask = (~near_peak) & finite
    if int(baseline_mask.sum()) < 10:
        baseline_mask = finite
    baseline_seg = np.sort(sig[baseline_mask])
    n_low = max(5, int(baseline_low_frac * len(baseline_seg)))
    baseline = float(np.mean(baseline_seg[:n_low]))
    if peak_val <= baseline:
        return None
    threshold = baseline + 0.5 * (peak_val - baseline)
    n = len(sig)
    last_above = peak_idx
    for i in range(peak_idx + 1, n):
        if not finite[i] or sig[i] <= threshold:
            return float(arc[last_above])
        last_above = i
    return float(arc[last_above])


def _inner_edge_from_baseline(arc, signal, std_mult: float = 3.0,
                              min_region_width_mm: float = 1.5,
                              peak_floor_fraction: float = 0.05,
                              peak_exclusion_radius_mm: float = 10.0,
                              reference_arc: float | None = None,
                              baseline_low_frac: float = 0.3) -> float | None:
    """Walk-from-the-deep-edge inner-edge detector. With ``reference_arc``
    set, the picker is peak-anchored: among connected above-threshold
    regions, choose the one whose local-max arc is closest to the
    reference, return its deep-side edge. Without, pick the widest
    region. See notebook source for full rationale."""
    sig = np.asarray(signal, dtype=float)
    arc = np.asarray(arc, dtype=float)
    finite = np.isfinite(sig)
    if not finite.any() or int(finite.sum()) < 10:
        return None
    peak_idx = int(np.nanargmax(sig))
    peak_arc = float(arc[peak_idx])
    near_peak = np.abs(arc - peak_arc) <= float(peak_exclusion_radius_mm)
    baseline_mask = (~near_peak) & finite
    if int(baseline_mask.sum()) < 10:
        baseline_mask = finite
    baseline_seg = np.sort(sig[baseline_mask])
    n_low = max(5, int(baseline_low_frac * len(baseline_seg)))
    low_seg = baseline_seg[:n_low]
    baseline = float(np.mean(low_seg))
    std      = float(np.std(low_seg))
    peak_val = float(sig[peak_idx])
    threshold_baseline = baseline + std_mult * std
    threshold_peak     = peak_floor_fraction * peak_val if peak_val > 0 else 0.0
    threshold          = max(threshold_baseline, threshold_peak)
    if not np.isfinite(threshold):
        return None
    above = (sig > threshold) & finite
    if not above.any():
        return None
    n = len(above)
    runs: list[tuple[int, int, float]] = []
    in_run = False
    run_start = -1
    for i in range(n):
        if above[i] and not in_run:
            in_run = True; run_start = i
        elif not above[i] and in_run:
            in_run = False
            width_mm = float(arc[i - 1] - arc[run_start])
            runs.append((run_start, i - 1, width_mm))
    if in_run:
        width_mm = float(arc[n - 1] - arc[run_start])
        runs.append((run_start, n - 1, width_mm))
    runs = [r for r in runs if r[2] >= min_region_width_mm]
    if not runs:
        return None
    if reference_arc is None or not np.isfinite(reference_arc):
        chosen = max(runs, key=lambda r: r[2])
    else:
        ref = float(reference_arc)
        from scipy.signal import find_peaks
        peaks_idx, _ = find_peaks(np.where(finite, sig, 0.0), height=threshold)
        peak_to_region: dict[int, tuple[int, int, float]] = {}
        for p in peaks_idx:
            for r in runs:
                if r[0] <= p <= r[1]:
                    peak_to_region.setdefault(int(p), r)
                    break
        if peak_to_region:
            closest_peak = min(peak_to_region.keys(),
                                key=lambda p: abs(arc[p] - ref))
            chosen = peak_to_region[closest_peak]
        else:
            chosen = min(runs, key=lambda r: abs(arc[r[1]] - ref))
    return float(arc[chosen[1]])


def _bolt_end_dring_near_reference(arc, signal, reference_arc: float | None,
                                    fraction_of_overall: float = 0.30) -> float | None:
    """Local-minimum picker on ``d(signal)/d(arc)``: find local minima
    whose magnitude is at least ``fraction_of_overall`` × |overall min|;
    return the one whose arc is closest to ``reference_arc`` (typically
    the bone-width inner-edge landmark)."""
    from scipy.signal import find_peaks
    sig = np.asarray(signal, dtype=float)
    arc = np.asarray(arc, dtype=float)
    grad = np.gradient(sig, arc)
    finite = np.isfinite(grad)
    if not finite.any():
        return None
    grad_safe = np.where(finite, grad, np.inf)
    overall_min = float(grad_safe.min())
    if overall_min >= 0:
        return None
    height_thr = -(fraction_of_overall * overall_min)
    peaks, _ = find_peaks(-grad_safe, height=height_thr)
    if len(peaks) == 0:
        return None
    if reference_arc is None or not np.isfinite(reference_arc):
        return float(arc[int(peaks.max())])
    peak_arcs = arc[peaks]
    closest = int(np.argmin(np.abs(peak_arcs - float(reference_arc))))
    return float(arc[peaks[closest]])


# ---------------------------------------------------------------------
# Per-trajectory landmark orchestrator.
# ---------------------------------------------------------------------


def estimate_landmarks(
    chain: dict[str, Any],
    *,
    name: str = "",
    ct_arr: np.ndarray,
    ras_to_ijk: np.ndarray,
    intracranial_mask_arr: np.ndarray | None = None,
    frangi_arr: np.ndarray | None = None,
    metal_hu_floor: float | None = None,
    profile_step_mm: float = PROFILE_STEP_MM,
    pad_tip_mm: float = PROFILE_PAD_TIP_MM,
    frangi_pad_outward_mm: float = FRANGI_PAD_OUTWARD_MM,
    frangi_walk_smooth_mm: float = FRANGI_WALK_SMOOTH_MM,
) -> TrajectoryLandmarks:
    """Compute intracranial-side landmarks for one fitted chain.

    Three independent inner-edge estimators (bone-width, ring-metal +
    d(ring), Frangi) anchored on the bone-width half-fall position give
    a robust median-combined ``bolt_combined_arc_mm``.

    ``metal_hu_floor`` overrides the per-trajectory Otsu threshold for
    the metal HU floor; when ``None`` (default), a per-trajectory
    Otsu-on-intracranial-HU is computed.
    """
    from scipy.ndimage import maximum_filter1d
    axis_unit = unit(chain["axis"])
    entry = np.asarray(chain["entry_ras"], dtype=float)
    tip   = np.asarray(chain["tip_ras"],   dtype=float)
    entry_to_tip = float(np.linalg.norm(tip - entry))
    walked_mm = float(chain.get("bolt_walked_mm", 0.0))

    if metal_hu_floor is None:
        intra_hu = collect_intracranial_hu_along_trajectory(
            entry, tip, axis_unit,
            ct_vol=ct_arr, ras_to_ijk_mat=ras_to_ijk,
            intracranial_mask_arr=intracranial_mask_arr,
        )
        metal_thr = otsu_threshold(intra_hu)
    else:
        metal_thr = float(metal_hu_floor)

    arc_rel, prof = sample_trajectory_profile(
        entry, tip, axis_unit,
        ct_vol=ct_arr, ras_to_ijk_mat=ras_to_ijk,
        metal_hu=metal_thr,
    )

    # Frangi trace (only computed when frangi_arr is supplied; safe to
    # pass None for callers that don't need the Frangi-side estimate).
    bolt_frangi_arc: float | None = None
    if frangi_arr is not None:
        arc_frangi_start = -frangi_pad_outward_mm
        arc_frangi_end   = entry_to_tip + pad_tip_mm
        n_frangi = int((arc_frangi_end - arc_frangi_start) / profile_step_mm) + 1
        arc_frangi = np.linspace(arc_frangi_start, arc_frangi_end, n_frangi)
        frangi_pts = entry[None, :] + arc_frangi[:, None] * axis_unit[None, :]
        frangi_vals = sample_trilinear_batch(
            frangi_arr.astype(np.float32, copy=False), ras_to_ijk, frangi_pts,
        )
        frangi_vals = np.where(np.isfinite(frangi_vals), frangi_vals, 0.0)
        walk_win_n = max(1, int(frangi_walk_smooth_mm / profile_step_mm))
        frangi_smoothed = maximum_filter1d(frangi_vals, size=walk_win_n, mode="nearest")
        bolt_frangi_arc = _inner_edge_from_baseline(arc_frangi, frangi_smoothed,
                                                    reference_arc=None)
        # Real anchor pass uses the bone landmark we're about to compute;
        # we re-estimate Frangi with reference_arc=bone_arc once that's
        # available, below.

    bone_arc = _bone_inner_edge_half_fall(arc_rel, prof["bone_width_mm"])
    bolt_rm_arc = _inner_edge_from_baseline(
        arc_rel, prof["ring_metal_total"], reference_arc=bone_arc,
    )
    bolt_dring_arc = _bolt_end_dring_near_reference(
        arc_rel, prof["ring_metal_total"], bone_arc,
    )
    if frangi_arr is not None:
        # Re-pick Frangi anchored on bone_arc now that it's available.
        bolt_frangi_arc = _inner_edge_from_baseline(
            arc_frangi, frangi_smoothed, reference_arc=bone_arc,
        )

    candidates = [x for x in (bolt_rm_arc, bolt_dring_arc, bolt_frangi_arc)
                  if x is not None]
    bolt_combined_arc = float(np.median(candidates)) if candidates else None

    return TrajectoryLandmarks(
        name=str(name or chain.get("name", "")),
        entry_arc_mm=0.0,
        tip_arc_mm=entry_to_tip,
        pre_walk_arc_mm=walked_mm,
        bone_arc_mm=bone_arc,
        bolt_rm_arc_mm=bolt_rm_arc,
        bolt_dring_arc_mm=bolt_dring_arc,
        bolt_frangi_arc_mm=bolt_frangi_arc,
        bolt_combined_arc_mm=bolt_combined_arc,
    )


# ---------------------------------------------------------------------
# Top-level seeded-fit orchestrator.
# ---------------------------------------------------------------------


def run_seeded_fit(
    planned_trajectories: Sequence[dict[str, Any]],
    *,
    signal_vol: np.ndarray,
    threshold: float,
    ras_to_ijk: np.ndarray,
    bolt_signal_vol: np.ndarray | None = None,
    bolt_signal_floor: float = 0.4,
    log: Callable[[str], None] | None = None,
    label: str = "fit",
) -> list[dict[str, Any] | None]:
    """Run snap → arbitrate → walk-to-bolt-outer-edge over a list of
    planned trajectories.

    Args:
        planned_trajectories: list of dicts with ``name``, ``start``,
            ``end`` (3-tuples of RAS floats). The ``start`` is the
            shallow / bolt-side end, ``end`` is the deep tip.
        signal_vol: 3-D float array (KJI order, native voxel grid) used
            by the snap walker — typically ``log_neg = -log_sigma1``.
        threshold: per-volume snap threshold; usually
            ``LOG_NEG_THRESHOLD = 300.0`` for raw LoG-neg, or
            ``0.10 · bolt_LoG_p90`` when calibrated from a bolt mask.
        ras_to_ijk: 4×4 RAS→IJK matrix for the volume.
        bolt_signal_vol: 3-D array for the bolt-outer-edge walker.
            Recommended: ``metal_evidence = max(|LoG|/800, HU/2000)``;
            falls back to ``signal_vol`` when None.
        bolt_signal_floor: walk halts when the smoothed bolt-signal
            drops below this; default 0.4 matches the notebook
            metal_evidence calibration.
        log: optional ``log(str)`` callback for progress messages.
        label: tag for the progress message.

    Returns the list of chain dicts (or None for failed shanks).
    """
    log_fn = log if log is not None else (lambda _m: None)
    chains_p1: list[dict[str, Any] | None] = []
    planned_axes: list[np.ndarray] = []
    for traj in planned_trajectories:
        start = np.asarray(traj["start"], dtype=float)
        end   = np.asarray(traj["end"],   dtype=float)
        planned_axes.append(unit(end - start))
        chains_p1.append(snap_via_signal_walk(
            start, end,
            signal_vol=signal_vol, threshold=threshold,
            ras_to_ijk=ras_to_ijk,
        ))
    n_p1 = sum(1 for c in chains_p1 if c is not None)
    chains_arb, n_conflicts = arbitrate_shared_peaks(
        chains_p1, planned_axes=planned_axes,
    )
    n_arb = sum(1 for c in chains_arb if c is not None)
    walker_signal = bolt_signal_vol if bolt_signal_vol is not None else signal_vol
    for r in chains_arb:
        if r is None:
            continue
        new_entry, walked = walk_to_bolt_outer_edge(
            r["entry_ras"], r["axis"],
            signal_vol=walker_signal, ras_to_ijk=ras_to_ijk,
            signal_floor=bolt_signal_floor,
        )
        r["entry_ras"]      = new_entry
        r["bolt_walked_mm"] = walked
    log_fn(
        f"[seeded_fit:{label}] pass-1 {n_p1}/{len(chains_p1)} fit, "
        f"arbitration resolved {n_conflicts} conflicts, post-arb {n_arb}"
    )
    return chains_arb


__all__ = [
    "FittedChain",
    "TrajectoryLandmarks",
    "snap_via_signal_walk",
    "arbitrate_shared_peaks",
    "walk_to_bolt_outer_edge",
    "estimate_landmarks",
    "run_seeded_fit",
    # Constants
    "METAL_SNAP_TUBE_RADIUS_MM",
    "METAL_SNAP_STEP_MM",
    "METAL_SNAP_MIN_PITCH_MM",
    "METAL_SNAP_MIN_N_PEAKS",
    "SNAP_EXTEND_PAST_END_MM",
    "SNAP_EXTEND_BEFORE_START_MM",
    "SHARED_PEAK_TOL_MM",
    "FRANGI_WALK_SMOOTH_MM",
    "FRANGI_WALK_MAX_MM",
    "FRANGI_PAD_OUTWARD_MM",
]
