"""Pitch auto-detection from the intracranial blob cloud + library snap.

When the user picks the "auto" pitch strategy, the walker can't run
without a concrete pitch — this module estimates it from the mutual-NN
distance distribution of the intracranial blob cloud and snaps to the
nearest library pitch.
"""
from __future__ import annotations

import numpy as np

from ..primitives.geometry import sample_dist_at_ras
from ..primitives.preprocessing import INTRACRANIAL_MIN_DISTANCE_MM
from .constants import (
    PITCH_AUTO_MAX_MM,
    PITCH_AUTO_MAX_PEAKS,
    PITCH_AUTO_MIN_MM,
    PITCH_AUTO_PEAK_EXCLUSION_MM,
    PITCH_AUTO_SECONDARY_FRAC,
    PITCH_MM,
    PITCH_SNAP_MM,
)
from .pitch_library import LIBRARY_PITCHES_MM


def detect_pitch_from_intracranial_blobs(
    pts_c, dist_arr, ras_to_ijk_mat,
    min_mm: float = PITCH_AUTO_MIN_MM,
    max_mm: float = PITCH_AUTO_MAX_MM,
    max_peaks: int = PITCH_AUTO_MAX_PEAKS,
    secondary_frac: float = PITCH_AUTO_SECONDARY_FRAC,
    peak_exclusion_mm: float = PITCH_AUTO_PEAK_EXCLUSION_MM,
):
    """Estimate electrode pitch(es) from mutual-nearest-neighbour
    distances of the intracranial blob cloud.

    Mutual-NN pairs (A's 1-NN is B AND B's 1-NN is A) are dominated by
    same-shank adjacent contacts, so their distribution peaks at the
    true electrode pitch. Non-mutual neighbours (cross-shank, bolt→
    contact, noise) don't show up. Empirical centroid tends to sit
    ~0.2 mm low of nominal (partial-volume bias) — within walker
    tolerance.

    On mixed-family subjects the histogram has more than one peak. The
    picker iteratively records peaks (centroid in a 0.6 mm window),
    masks out a ±peak_exclusion window, and continues — up to
    ``max_peaks`` peaks each ≥ ``secondary_frac`` of the dominant.

    Returns list of detected pitches in descending peak-height order,
    or empty when the cloud is too sparse / histogram is flat.
    """
    if pts_c is None or len(pts_c) < 10:
        return []
    pts_arr = np.asarray(pts_c, dtype=float)
    hd = np.array([
        sample_dist_at_ras(dist_arr, ras_to_ijk_mat, p) for p in pts_arr
    ])
    pts_ic = pts_arr[hd >= INTRACRANIAL_MIN_DISTANCE_MM]
    if pts_ic.shape[0] < 10:
        return []
    D = np.sqrt(np.sum((pts_ic[:, None, :] - pts_ic[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(D, np.inf)
    nn_idx = np.argmin(D, axis=1)
    mutual_dists = []
    seen = set()
    for i in range(pts_ic.shape[0]):
        j = int(nn_idx[i])
        if int(nn_idx[j]) == i:
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            mutual_dists.append(float(D[i, j]))
    if len(mutual_dists) < 5:
        return []
    m = np.asarray(mutual_dists)
    m = m[(m >= min_mm) & (m <= max_mm)]
    if m.size < 5:
        return []
    bins = np.arange(min_mm, max_mm + 0.25, 0.25)
    hist, edges = np.histogram(m, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if hist.max() <= 0:
        return []

    centroid_window_mm = 0.6
    work = hist.astype(float).copy()
    dominant_h = float(work.max())
    picks: list[float] = []
    for _ in range(int(max_peaks)):
        peak_h = float(work.max())
        if peak_h <= 0:
            break
        if picks and peak_h < secondary_frac * dominant_h:
            break
        peak_idx = int(np.argmax(work))
        peak_mm = float(centers[peak_idx])
        # Centroid uses the original histogram so an earlier pick's
        # exclusion doesn't shrink the neighbouring peak's centroid
        # window; the exclusion only steers which bin seeds the next.
        window = np.abs(centers - peak_mm) <= centroid_window_mm
        denom = float(np.sum(hist[window]))
        if denom <= 0:
            work[peak_idx] = 0.0
            continue
        centroid = float(np.sum(centers[window] * hist[window]) / denom)
        picks.append(round(centroid, 2))
        work[np.abs(centers - peak_mm) <= peak_exclusion_mm] = 0.0
    return picks


def snap_to_library_pitch(
    pitch_mm: float,
    library: tuple = LIBRARY_PITCHES_MM,
    tol_mm: float = PITCH_SNAP_MM,
) -> float:
    """If ``pitch_mm`` is within ``tol_mm`` of a known library pitch,
    return the nominal library value; otherwise return ``pitch_mm``
    unchanged. Removes the ~0.2 mm low-bias from the mutual-NN centroid.
    """
    pitch_mm = float(pitch_mm)
    best = None
    for lib in library:
        d = abs(pitch_mm - float(lib))
        if d <= tol_mm and (best is None or d < best[0]):
            best = (d, float(lib))
    return best[1] if best is not None else pitch_mm


def resolve_pitches_for_strategy(
    strategy, pts_c=None, dist_arr=None, ras_to_ijk_mat=None,
):
    """Map a UI ``strategy`` string to a concrete tuple of walker
    pitches. Falls back to the Dixi default when auto-detection fails
    or the strategy is unrecognised.
    """
    from rosa_core.electrode_classifier import PITCH_STRATEGY_PITCHES_MM

    key = str(strategy or "dixi").lower()
    if key in PITCH_STRATEGY_PITCHES_MM:
        return PITCH_STRATEGY_PITCHES_MM[key]
    if key == "auto":
        if pts_c is None or dist_arr is None or ras_to_ijk_mat is None:
            return (PITCH_MM,)
        detected = detect_pitch_from_intracranial_blobs(
            pts_c, dist_arr, ras_to_ijk_mat,
        )
        if detected:
            snapped: list[float] = []
            for p in detected:
                snap_p = snap_to_library_pitch(p)
                if snap_p not in snapped:
                    snapped.append(snap_p)
            return tuple(snapped)
        return (PITCH_MM,)
    return (PITCH_MM,)


__all__ = [
    "detect_pitch_from_intracranial_blobs",
    "snap_to_library_pitch",
    "resolve_pitches_for_strategy",
]
