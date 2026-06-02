"""Seeded shank fitting + intracranial landmark estimation.

Given a planned trajectory (start/end RAS) and a per-CT signal volume
(LoG-neg = ``clip(-LoG, 0)``), the seeded-fit pipeline snaps the plan onto
the real electrode using four concepts — no per-subject thresholds beyond
the contact pitch / metal level the caller already supplies:

1. **On-axis detection** (:func:`snap_via_signal_walk`). A contact is a
   bright LoG-neg blob centred on the electrode axis. Sampling on the
   *centerline* (a single ray, perp = 0) registers this shank's contacts at
   full strength while a crossing neighbour's metal — necessarily off this
   axis — contributes only its decaying Gaussian tail, so it neither
   invents nor suppresses peaks. ``find_peaks`` then detects contacts.

2. **Drag-free localization + axis refinement.** Each contact is localised
   by a weighted centroid in a perpendicular disk, restricted to its
   *contact-grade core* (>= ``CONTACT_GRADE_FRAC`` x the shank's median peak)
   so a thin inter-electrode bridge artifact cannot drag the centroid
   off-axis. A PCA line is refit through the contacts and detection
   repeats until the axis stops rotating.

3. **Leave-one-out + jump-gate arbitration** (:func:`arbitrate_shared_peaks`).
   A contact is reassigned away from a chain only when it is a *tail*
   separated from the body by a JUMP (> ``JUMP_GATE_PITCH`` x pitch — a
   contiguous tip is the shank's own and is never peeled) AND, with itself
   left out, it sits closer to a neighbouring chain's *detected* axis,
   within that neighbour's contact extent, than to the line through the
   chain's remaining contacts. Parameter-free in spirit: the scales are the
   electrode's own pitch and the contact pitch.

4. **Start point.** ``entry_ras`` is the shallowest detected contact, left
   at/inward of the bolt — *not* pushed outward to the bolt edge. (The old
   outward ``walk_to_bolt_outer_edge`` followed the bright-but-thin lead wire
   to the connector and overshot the bolt; a start beyond the bolt is worse
   than one in-bolt, and the downstream model fit owns contact placement.)
   ``walk_to_bolt_outer_edge`` is kept as a utility but no longer applied.

Then :func:`estimate_landmarks` computes three independent inner-edge
estimators (bone width, ring metal + d(ring), Frangi) anchored on the
bone-width half-fall landmark, and returns their median as a robust
combined intracranial bolt-edge arc.

:func:`run_seeded_fit` orchestrates detection -> arbitration -> bolt walk
over a list of planned trajectories.

Validated in ``notebooks/seeded_fit/`` (see ``SNAP_ALGORITHM.md``) on 23
SEEG subjects: tip-endpoint accuracy median 0.46 mm, 303/304 shanks within
2 mm; every cross-shank meeting/crossing resolved.
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
# Axis-refinement loop in snap_via_signal_walk: re-detect on the
# refit axis until it stops rotating (or this many passes).
METAL_SNAP_MAX_AXIS_REFITS  = 3
METAL_SNAP_AXIS_CONVERGE_DEG = 0.3
# A contact cannot be dimmer than this fraction of the array's own median
# peak strength — self-calibrated per shank. Used both to reject weak
# on-axis bone past the tip (tail-trim) and to restrict the localization
# centroid to the contact's bright core (so a thin bridge artifact can't
# drag it off-axis).
METAL_SNAP_CONTACT_GRADE_FRAC = 0.5
# Cross-shank arbitration: a tail contact farther than this multiple of the
# array's own pitch is a "jump" (an over-reach onto bridge/neighbour metal)
# and may be peeled; within it the contact is the shank's own contiguous
# tip and is protected. Stable plateau 1.15-1.30 on the SEEG dataset.
METAL_SNAP_JUMP_GATE_PITCH  = 1.3
# Seed-driven neighbour ownership (run_seeded_fit): a neighbouring planned
# segment constrains a shank's perp-disk localisation ONLY when it passes
# within this distance — i.e. inside ~half the tube radius, where its metal
# genuinely dominates part of the disk and the walk would drift onto it
# (e.g. JK RSPL↔RCUN 1.8 mm, T18 RIF↔RAF 1.3 mm). Beyond it the ownership
# boundary only clips the disk edge and needlessly perturbs the axis refit
# (T6 LPMC 4.9 mm, T11 LAMC 3.2 mm regressed when ownership engaged there), so
# moderately-near shanks are left exactly as the un-owned snap.
NEIGHBOR_OWNERSHIP_MAX_MM   = 2.5

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


def _pca_axis(peak_positions, oriented_along) -> np.ndarray:
    """Principal axis of a point set, oriented to agree with
    ``oriented_along``."""
    pts = np.asarray(peak_positions, dtype=float)
    rel = pts - pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(rel, full_matrices=False)
    ax = unit(Vt[0])
    if ax @ oriented_along < 0:
        ax = -ax
    return ax


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


def _point_segment_dist_batch(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Perpendicular distance from each point (N,3) to the segment a-b (clamped
    to the endpoints). Used by the seed-driven neighbour-ownership filter."""
    ab = b - a
    ab2 = float(ab @ ab)
    if ab2 < 1e-12:
        return np.linalg.norm(pts - a[None, :], axis=1)
    t = np.clip((pts - a[None, :]) @ ab / ab2, 0.0, 1.0)
    proj = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(pts - proj, axis=1)


def _owned_mask(pts: np.ndarray, own_seg, neighbor_segs) -> np.ndarray:
    """True where a point is closer to ``own_seg`` than to EVERY neighbour seg.

    The seeds give us a free ownership partition: a metal voxel belongs to the
    planned trajectory whose segment it sits nearest. Restricting a shank's
    perp-disk localisation to its owned voxels stops the walk drifting onto a
    close neighbour's metal (e.g. JK RSPL passing 1.8 mm from RCUN). ``own_seg``
    / each ``neighbor_segs`` entry is an ``(start_ras, end_ras)`` pair."""
    if not neighbor_segs:
        return np.ones(len(pts), dtype=bool)
    d_own = _point_segment_dist_batch(pts, own_seg[0], own_seg[1])
    keep = np.ones(len(pts), dtype=bool)
    for seg in neighbor_segs:
        keep &= d_own <= _point_segment_dist_batch(pts, seg[0], seg[1])
    return keep


def _centerline_profile(signal_f32, ras_to_ijk, origin, axis, t_arr):
    """LoG-neg sampled ON the axis (a single ray, perp=0) at each arc.

    Concept: a contact is a bright LoG-neg blob centred on the
    electrode axis. Sampling on the centerline registers this shank's
    own contacts at (near) full strength while a neighbouring shank's
    metal — necessarily off this axis — contributes only its Gaussian
    tail, which falls away with perpendicular distance. So the
    centerline profile is naturally specific to this electrode."""
    centers = origin[None, :] + t_arr[:, None] * axis[None, :]
    vals = sample_trilinear_batch(signal_f32, ras_to_ijk, centers)
    return np.where(np.isfinite(vals), vals, 0.0)


def _tube_max_profile(signal_f32, ras_to_ijk, origin, axis, t_arr,
                      disk_u, disk_v, perp1, perp2,
                      own_seg=None, neighbor_segs=None):
    """LoG-neg sampled as the MAX over a perpendicular disk at each arc (vs the
    single-ray :func:`_centerline_profile`).

    Used only as the seed-offset rescue in :func:`snap_via_signal_walk`: it sees
    a contact even when the seed axis misses it laterally, at the cost of the
    cross-shank specificity the centerline profile gives — which is why it is a
    fallback that bootstraps the axis re-centering rather than the primary
    detector. When ``neighbor_segs`` is given, the max is taken only over disk
    voxels owned by ``own_seg`` (closer to it than any neighbour), so the rescue
    can't bootstrap onto a close neighbour's metal."""
    centers = origin[None, :] + t_arr[:, None] * axis[None, :]
    out = np.empty(len(t_arr), dtype=float)
    for j, c in enumerate(centers):
        pts = (c[None, :] + disk_u[:, None] * perp1[None, :]
               + disk_v[:, None] * perp2[None, :])
        vals = sample_trilinear_batch(signal_f32, ras_to_ijk, pts)
        keep = np.isfinite(vals)
        if neighbor_segs:
            keep &= _owned_mask(pts, own_seg, neighbor_segs)
        out[j] = float(vals[keep].max()) if keep.any() else 0.0
    return out


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
    max_axis_refits: int = METAL_SNAP_MAX_AXIS_REFITS,
    neighbor_segments: "list[tuple[np.ndarray, np.ndarray]] | None" = None,
) -> dict[str, Any] | None:
    """Snap a planned (start, end) RAS segment onto contact peaks.

    Two concepts, no tuned thresholds beyond the contact pitch / metal
    threshold the caller already supplies:

    1. **On-axis detection.** Contacts are local maxima of LoG-neg
       sampled on the centerline (``_centerline_profile``). Off-axis
       metal from a crossing electrode contributes only its decaying
       Gaussian tail on this axis, so it neither creates spurious peaks
       nor (via tube-max) suppresses this shank's real peaks.

    2. **Axis refinement.** Each detected contact is localised by a
       weighted centroid in a perpendicular disk, then a PCA line is
       refit through the contacts and detection is repeated. This lets
       the axis track the actual electrode when the seed is slightly
       off, and converges in 1-2 iterations. Iteration stops when the
       axis rotation between passes is negligible.

    Cross-shank conflicts (a contact that physically belongs to a
    neighbouring electrode but happens to fall on this axis — e.g. two
    shanks meeting head-to-head) are resolved downstream in
    :func:`arbitrate_shared_peaks` by closest-axis attribution.

    Returns a dict (the legacy "chain" shape) with the same keys as
    ``_fit_chain_from_peaks``; ``None`` if fewer than ``min_n_peaks``
    contacts are found.
    """
    from scipy.signal import find_peaks

    s = np.asarray(planned_start, dtype=float)
    e = np.asarray(planned_end,   dtype=float)
    L = float(np.linalg.norm(e - s))
    if L < 5.0:
        return None
    signal_f32 = signal_vol.astype(np.float32, copy=False)
    # Seed-driven neighbour ownership: when other planned segments are supplied,
    # every perp-disk localisation below keeps only voxels nearer THIS seed than
    # any neighbour, so the walk can't drift onto a close neighbour's metal.
    own_seg = (s, e)

    n_d = int(2 * tube_radius_mm / 0.5) + 1
    u_arr = np.linspace(-tube_radius_mm, tube_radius_mm, n_d)
    UU, VV = np.meshgrid(u_arr, u_arr, indexing="ij")
    disk_mask = (UU ** 2 + VV ** 2) <= tube_radius_mm ** 2
    disk_u = UU[disk_mask]; disk_v = VV[disk_mask]

    t_lo = -extend_before_start_mm
    t_hi = L + extend_past_end_mm
    n_samples = int((t_hi - t_lo) / step_mm) + 1
    t_arr = np.linspace(t_lo, t_hi, n_samples)
    dist = max(1, int(min_pitch_mm / step_mm))

    axis = unit(e - s)
    origin = s.copy()
    peak_positions: np.ndarray | None = None

    for _it in range(max_axis_refits + 1):
        perp1, perp2 = orthonormal_basis_for_axis(axis)
        profile = _centerline_profile(signal_f32, ras_to_ijk, origin, axis, t_arr)
        peaks, _ = find_peaks(profile, height=threshold, distance=dist)
        if len(peaks) < min_n_peaks:
            # Seed-offset rescue. The bare centerline (perp=0) only sees the
            # contacts when the seed axis already runs through them; if the
            # planned/seed axis is laterally offset from the real electrode (a
            # plan-vs-implant / brain-shift mismatch — common on raw surgical
            # plans), the centerline samples mostly off-metal and finds too few
            # peaks. Re-detect on a tube-MAX profile (brightest LoG-neg in the
            # perp disk at each arc) so off-axis contacts still register; the
            # weighted-centroid localization + axis refit below then pull the
            # axis onto the real electrode, and the next pass reverts to the
            # centerline (now on-target) for cross-shank specificity. This is a
            # pure fallback: when the centerline already finds the contacts
            # (on-axis seeds) it never runs, so on-axis behaviour is unchanged.
            profile = _tube_max_profile(
                signal_f32, ras_to_ijk, origin, axis, t_arr,
                disk_u, disk_v, perp1, perp2,
                own_seg=own_seg, neighbor_segs=neighbor_segments)
            peaks, _ = find_peaks(profile, height=threshold, distance=dist)
            if len(peaks) < min_n_peaks:
                return None
        # Contact-grade tail trim. The bare metal ``threshold`` admits
        # weak on-axis bone (skull base past the array tip, LoG-neg
        # ~300-800) that is not contact metal (contacts saturate, ~1500
        # and consistent within an electrode). Trim peaks from each END
        # of the chain while they are below half the array's own median
        # peak strength — this removes a weak bone tail without a tuned
        # absolute level, and without touching dim *interior* contacts
        # or strong collinear-neighbour metal (which is left for the
        # closest-axis arbitration). Tail-only (not a global filter) so
        # it can't destabilise the axis refit.
        heights = profile[peaks]
        contact_level = METAL_SNAP_CONTACT_GRADE_FRAC * float(np.median(heights))
        lo, hi = 0, len(peaks)
        while hi - lo > min_n_peaks and heights[lo] < contact_level:
            lo += 1
        while hi - lo > min_n_peaks and heights[hi - 1] < contact_level:
            hi -= 1
        peaks = peaks[lo:hi]
        # Localise each contact's true centre (weighted centroid in a
        # perpendicular disk) so the axis can be refit to follow the real
        # electrode. The bare metal ``threshold`` is used here (not the
        # contact-grade level) to keep the axis refit stable on dim distal
        # contacts; the drag-free contact-grade re-localization happens once
        # after convergence (see below).
        positions = np.empty((len(peaks), 3), dtype=float)
        for k, pk in enumerate(peaks):
            center = origin + float(t_arr[pk]) * axis
            pts = (center[None, :]
                   + disk_u[:, None] * perp1[None, :]
                   + disk_v[:, None] * perp2[None, :])
            vals = sample_trilinear_batch(signal_f32, ras_to_ijk, pts)
            strong = np.isfinite(vals) & (vals >= threshold)
            if neighbor_segments:
                strong &= _owned_mask(pts, own_seg, neighbor_segments)
            if int(strong.sum()) > 0:
                w = vals[strong].astype(float)
                c_u = float((w * disk_u[strong]).sum() / w.sum())
                c_v = float((w * disk_v[strong]).sum() / w.sum())
                positions[k] = center + c_u * perp1 + c_v * perp2
            else:
                positions[k] = center
        peak_positions = positions
        # Refit the axis through the localised contacts.
        new_axis = unit(positions[-1] - positions[0]) if len(positions) < 3 else _pca_axis(positions, axis)
        # Re-anchor the sampling origin onto the new line (project the
        # original seed start onto it so arc 0 stays near the bolt).
        rotation = float(np.degrees(np.arccos(np.clip(abs(new_axis @ axis), -1.0, 1.0))))
        centroid = positions.mean(axis=0)
        origin = centroid - new_axis * float((centroid - s) @ new_axis)
        axis = new_axis
        if rotation < METAL_SNAP_AXIS_CONVERGE_DEG:
            break

    # Drag-free re-localization (once, on the converged axis). Refine each
    # contact's centre on its *contact-grade core* (>= contact_level), not
    # the bare metal level: where two electrodes meet, a thin inter-electrode
    # bridge artifact (weak metal) entering the disk would pull the centroid
    # off-axis toward the neighbour — making Pass-2 arbitration peel a real
    # tip. Restricting to the bright core keeps the contact on its own axis.
    perp1, perp2 = orthonormal_basis_for_axis(axis)
    localize_level = max(threshold, contact_level)
    centroid = peak_positions.mean(axis=0)
    for k in range(len(peak_positions)):
        center = centroid + float((peak_positions[k] - centroid) @ axis) * axis
        pts = (center[None, :]
               + disk_u[:, None] * perp1[None, :]
               + disk_v[:, None] * perp2[None, :])
        vals = sample_trilinear_batch(signal_f32, ras_to_ijk, pts)
        strong = np.isfinite(vals) & (vals >= localize_level)
        if neighbor_segments:
            strong &= _owned_mask(pts, own_seg, neighbor_segments)
        if int(strong.sum()) > 0:
            w = vals[strong].astype(float)
            c_u = float((w * disk_u[strong]).sum() / w.sum())
            c_v = float((w * disk_v[strong]).sum() / w.sum())
            peak_positions[k] = center + c_u * perp1 + c_v * perp2

    chain = _fit_chain_from_peaks(peak_positions, axis)
    return chain


# ---------------------------------------------------------------------
# Step 4: shared-peak arbitration across chains.
# ---------------------------------------------------------------------


def _principal_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centroid + unit principal direction (PCA) of a point set. The sign
    of the direction is arbitrary — callers use it only for perpendicular
    distance and signed arc, both sign-consistent within a call."""
    centroid = points.mean(axis=0)
    if len(points) >= 3:
        _, _, Vt = np.linalg.svd(points - centroid, full_matrices=False)
        direction = unit(Vt[0])
    else:
        direction = unit(points[-1] - points[0])
    return centroid, direction


def _perp_and_arc(point: np.ndarray, line_pt: np.ndarray,
                  line_dir: np.ndarray) -> tuple[float, float]:
    """(perpendicular distance, signed arc) of ``point`` w.r.t. the line
    through ``line_pt`` along unit ``line_dir``."""
    rel = point - line_pt
    arc = float(rel @ line_dir)
    return float(np.linalg.norm(rel - arc * line_dir)), arc


def _jump_tail_eligible(arcs: np.ndarray, gate_pitch: float) -> set[int]:
    """Indices of contacts lying BEYOND a ``> gate_pitch x pitch`` jump from
    the chain body — the only contacts eligible for cross-shank peeling.

    A contiguous tail (within ~one pitch of its neighbour) is the shank's
    own tip and must never be peeled, even if it sits near a neighbour's
    axis. Pitch is the chain's own median spacing, so the gate is in pitch
    units (physical), not millimetres. A CM cluster lying past an
    inter-cluster gap becomes 'eligible' but, having no neighbour to be
    reassigned to, is left untouched by the arbitration below."""
    n = len(arcs)
    if n < METAL_SNAP_MIN_N_PEAKS + 1:
        return set()
    gaps = np.diff(arcs)
    pitch = float(np.median(gaps))
    eligible: set[int] = set()
    for gi in np.where(gaps > gate_pitch * pitch)[0]:
        deep = set(range(gi + 1, n))
        if len(deep) <= n - len(deep):       # over-reach is the minority run
            eligible |= deep
        head = set(range(0, gi + 1))
        if len(head) <= n - len(head):
            eligible |= head
    return eligible


def arbitrate_shared_peaks(
    chains: Sequence[dict[str, Any] | None],
    *,
    extent_pad_mm: float = METAL_SNAP_MIN_PITCH_MM,
    jump_gate_pitch: float = METAL_SNAP_JUMP_GATE_PITCH,
    min_peaks_after: int = METAL_SNAP_MIN_N_PEAKS,
) -> tuple[list[dict[str, Any] | None], int]:
    """Reassign contacts a chain wrongly claimed from a neighbour.

    Leave-one-out attribution on each chain's *detected* axis, gated by
    contiguity. A tail contact ``p`` of chain A is dropped only when:

    1. **It is a jump.** ``p`` lies beyond a ``> jump_gate_pitch x pitch``
       gap from A's body (:func:`_jump_tail_eligible`). A contiguous tip is
       A's own and is never peeled — this protects genuine deep tips that
       merely sit near a neighbour (e.g. two tips meeting, or a tip landing
       on a neighbour's body at a T-junction).
    2. **A neighbour explains it better.** With ``p`` left out, the line
       through A's *remaining* contacts is farther from ``p`` than some
       other chain's detected axis is — and ``p`` falls within that
       neighbour's contact extent (it has metal there). Leave-one-out makes
       the self-distance robust: an over-reach contact can't pull A's own
       line onto itself.

    Parameter-free in spirit: ``extent_pad_mm`` is the contact pitch the
    caller already supplies; ``jump_gate_pitch`` is in units of each
    shank's own pitch. Returns ``(new_chains, n_conflicts)``.
    """
    # Each surviving chain's detected line + along-axis contact extent.
    lines: list[tuple[np.ndarray, np.ndarray, float, float] | None] = []
    for ch in chains:
        if ch is None:
            lines.append(None); continue
        pts = np.asarray(ch["kept_pts"], dtype=float)
        centroid, direction = _principal_line(pts)
        arc = (pts - centroid) @ direction
        lines.append((centroid, direction, float(arc.min()), float(arc.max())))

    new_chains: list[dict[str, Any] | None] = []
    n_conflicts = 0
    for ci, chain in enumerate(chains):
        if chain is None:
            new_chains.append(None); continue
        c0, d0, *_ = lines[ci]
        pts = np.asarray(chain["kept_pts"], dtype=float)
        pts = pts[np.argsort((pts - c0) @ d0)]          # shallow -> deep
        keep = list(range(len(pts)))

        changed = True
        while changed and len(keep) > min_peaks_after:
            changed = False
            arcs = (pts[keep] - c0) @ d0
            eligible = {keep[i] for i in _jump_tail_eligible(arcs, jump_gate_pitch)}
            for end in (keep[0], keep[-1]):             # only the two tails
                if end not in eligible:
                    continue
                p = pts[end]
                rest = pts[[i for i in keep if i != end]]
                c_rest, d_rest = _principal_line(rest)
                perp_self, _ = _perp_and_arc(p, c_rest, d_rest)
                best_other = np.inf
                for cj, ln in enumerate(lines):
                    if cj == ci or ln is None:
                        continue
                    cN, dN, arc_lo, arc_hi = ln
                    perp_o, arc_o = _perp_and_arc(p, cN, dN)
                    within = (arc_lo - extent_pad_mm) <= arc_o <= (arc_hi + extent_pad_mm)
                    if within and perp_o < best_other:
                        best_other = perp_o
                if best_other < perp_self:
                    keep.remove(end); n_conflicts += 1; changed = True
                    break

        n_dropped = len(pts) - len(keep)
        kept_pts = pts[keep]
        if len(kept_pts) < min_peaks_after:
            new_chains.append(None); continue
        new_chain = _fit_chain_from_peaks(kept_pts, chain["axis"])
        new_chain["n_dropped_in_arbitration"] = int(n_dropped)
        new_chains.append(new_chain)
    return new_chains, n_conflicts


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
    neighbor_aware: bool = True,
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
    # Seed-driven neighbour ownership: each shank's snap is restricted to metal
    # nearer its own planned segment than any other shank's, so the walk can't
    # drift onto a close neighbour (crossing / parallel pairs). The planned list
    # is itself the ownership partition; this is a no-op for a single trajectory
    # and for well-separated shanks (the boundary is far from any real contact).
    segs = [(np.asarray(t["start"], dtype=float), np.asarray(t["end"], dtype=float))
            for t in planned_trajectories]

    def _close_neighbors(i):
        """Planned segments passing within NEIGHBOR_OWNERSHIP_MAX_MM of shank i
        (None when none / disabled — ownership then a no-op)."""
        if not neighbor_aware or len(segs) <= 1:
            return None
        ai, bi = segs[i]
        ts = np.linspace(0.0, 1.0, 16)
        pts_i = ai[None, :] + ts[:, None] * (bi - ai)[None, :]
        nbrs = [segs[j] for j in range(len(segs)) if j != i
                and float(_point_segment_dist_batch(pts_i, segs[j][0], segs[j][1]).min())
                < NEIGHBOR_OWNERSHIP_MAX_MM]
        return nbrs or None

    # Pass 1: per-shank on-axis detection + drag-free localization + refit.
    chains_p1: list[dict[str, Any] | None] = [
        snap_via_signal_walk(
            segs[i][0], segs[i][1],
            signal_vol=signal_vol, threshold=threshold,
            ras_to_ijk=ras_to_ijk,
            neighbor_segments=_close_neighbors(i),
        )
        for i in range(len(segs))
    ]
    n_p1 = sum(1 for c in chains_p1 if c is not None)
    # Pass 2: leave-one-out + jump-gate arbitration on the detected axes.
    chains_arb, n_conflicts = arbitrate_shared_peaks(chains_p1)
    n_arb = sum(1 for c in chains_arb if c is not None)
    # Start point = the shallowest detected contact (already set as
    # ``entry_ras`` by ``_fit_chain_from_peaks``). We deliberately do NOT
    # push it outward to the bolt edge: detection begins at the seed bolt and
    # walks inward, so the shallowest contact is always at/inward of the bolt
    # — never beyond it. The old ``walk_to_bolt_outer_edge`` followed the
    # bright-but-thin lead wire out to the connector and overshot the bolt by
    # up to ~25 mm on many shanks (a start *beyond* the bolt is worse than one
    # in the bolt, which is acceptable — near-bolt contacts are unresolvable
    # and the downstream model fit owns contact placement). ``bolt_signal_vol``
    # / ``bolt_signal_floor`` are retained for API stability.
    for r in chains_arb:
        if r is not None:
            r["bolt_walked_mm"] = 0.0
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
    "METAL_SNAP_CONTACT_GRADE_FRAC",
    "METAL_SNAP_JUMP_GATE_PITCH",
    "FRANGI_WALK_SMOOTH_MM",
    "FRANGI_WALK_MAX_MM",
    "FRANGI_PAD_OUTWARD_MM",
]
