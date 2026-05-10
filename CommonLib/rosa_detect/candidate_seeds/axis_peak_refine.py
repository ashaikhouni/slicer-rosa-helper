"""Post-detection axis-profile peak refinement.

After the walker + bolt anchor produce a tentative trajectory, the
walker's NN-spacing pitch can be biased on anisotropic CTs (S56's
auto-detect locked to 3.14 mm instead of 3.5 mm — sub-voxel aliasing
of blob centroids on the X/Y-downsampled grid). Sampling the LoG
profile 1-dimensionally along the FIT axis at 0.25 mm steps with
trilinear interpolation recovers sub-voxel peak positions and thus
the true contact pitch.

Used by the orchestrator's electrode-model picker as a refinement
pass before classification dispatch.
"""
from __future__ import annotations

import numpy as np


def refine_signature_via_axis_peaks(rec, log_arr, ras_to_ijk_mat,
                                    step_mm=0.25,
                                    disk_radius_mm=2.0,
                                    n_radii=4,
                                    n_angles=8,
                                    min_amplitude=200.0,
                                    min_separation_mm=2.0,
                                    min_peaks_required=4,
                                    shallow_pad_mm=1.5,
                                    deep_pad_mm=3.0):
    """Re-derive ``(n_inliers, median_pitch, contact_span)`` for one
    trajectory by 1-D peak picking on the LoG profile sampled along
    its intracranial axis. Returns ``None`` if the axis yields fewer
    than ``min_peaks_required`` peaks (caller should keep walker
    stats in that case).

    The walker's NN-spacing pitch is biased on anisotropic CTs (e.g.
    S56's auto-detect locked to 3.14 mm instead of 3.5 mm — sub-voxel
    aliasing of blob centroids on the X/Y-downsampled grid). Peak
    detection along the FIT axis samples the LoG at 0.25 mm steps with
    trilinear interpolation, recovering sub-voxel peak positions and
    thus the true contact pitch.

    The ``shallow_pad_mm`` / ``deep_pad_mm`` extend the sampling range
    slightly past the skull entry and deep tip — small headroom catches
    the first / last contact when the bolt anchor or deep-end refine
    placed the endpoint just past the contact.

    Returns dict with keys ``n_peaks``, ``median_pitch_mm``,
    ``peak_span_mm``, ``peak_arc_mm`` (list, debug) or ``None``.
    """
    from rosa_core.contact_peak_fit import sample_axis_profile, detect_peaks_1d
    entry = np.asarray(
        rec.get("skull_entry_ras", rec.get("start_ras")), dtype=float,
    ).reshape(3)
    end = np.asarray(rec.get("end_ras"), dtype=float).reshape(3)
    axis_vec = end - entry
    L = float(np.linalg.norm(axis_vec))
    if L < 5.0:
        return None
    axis_unit = axis_vec / L
    sample_start = entry - float(shallow_pad_mm) * axis_unit
    sample_end = end + float(deep_pad_mm) * axis_unit
    try:
        arc_mm, profile = sample_axis_profile(
            volume_kji=log_arr,
            ras_to_ijk_mat=np.asarray(ras_to_ijk_mat, dtype=float),
            start_ras=sample_start, end_ras=sample_end,
            step_mm=step_mm, disk_radius_mm=disk_radius_mm,
            n_radii=n_radii, n_angles=n_angles, reducer="min",
        )
    except Exception:
        return None
    peaks_arc = detect_peaks_1d(
        profile, step_mm=step_mm, polarity="min",
        min_amplitude=min_amplitude,
        min_separation_mm=min_separation_mm,
    )
    if len(peaks_arc) < int(min_peaks_required):
        return None
    arr = np.asarray(sorted(peaks_arc), dtype=float)
    diffs = np.diff(arr)
    if diffs.size == 0:
        return None
    median_pitch = float(np.median(diffs))
    peak_span = float(arr[-1] - arr[0])
    return {
        "n_peaks": int(arr.size),
        "median_pitch_mm": median_pitch,
        "peak_span_mm": peak_span,
        "peak_arc_mm": [float(v) for v in arr.tolist()],
    }


__all__ = ["refine_signature_via_axis_peaks"]
