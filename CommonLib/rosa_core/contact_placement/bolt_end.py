"""Bolt-end / metal-mass landmark estimation (relocated 2026-05-30 from the
retired contact_placement_legacy.py).

Estimates the bolt -> contact transition arc from total HU mass along a
trajectory (straight axis + polynomial centerline, combined). Used by
contact_placement.stage_a_anchor and re-exported from contact_placement.
Slicer/VTK/Qt-clean.
"""
from __future__ import annotations

import numpy as np

def _polyline_segments(path_ras: np.ndarray):
    """Return per-segment start, direction-unit, length for a polyline.

    ``path_ras`` shape ``(K, 3)`` with ``K >= 2``. Zero-length segments
    are dropped (consecutive duplicate points).
    """
    P = np.asarray(path_ras, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] < 2:
        raise ValueError("path_ras must be (K,3) with K>=2")
    diffs = P[1:] - P[:-1]
    lens = np.linalg.norm(diffs, axis=1)
    keep = lens > 1e-9
    if not keep.any():
        raise ValueError("path_ras has zero arc length")
    starts = P[:-1][keep]
    units = (diffs[keep] / lens[keep, None])
    seglens = lens[keep]
    cum_start = np.concatenate([[0.0], np.cumsum(seglens)[:-1]])
    return starts, units, seglens, cum_start

def median_library_pitch_mm(library_models) -> float | None:
    """Median spacing between consecutive contacts across the strategy's
    models. Used as the optimal envelope-smoothing σ in
    ``entry_arc_from_metal_mass``: a Gaussian of σ = pitch averages
    over exactly one full contact-oscillation period, cancelling each
    shaft contact peak with its adjacent valley while leaving the
    bolt's sustained-high mass untouched.

    Returns None when no library models are provided (caller can fall
    back to a fixed default).
    """
    pitches: list[float] = []
    for m in (library_models or []):
        offsets = np.asarray(
            m.get("contact_center_offsets_from_tip_mm") or [],
            dtype=float,
        )
        if offsets.size >= 2:
            sorted_offsets = np.sort(offsets)
            pitches.extend(np.diff(sorted_offsets).tolist())
    if not pitches:
        return None
    return float(np.median(pitches))

def _brightness(volume_values, polarity: str):
    """Convert raw volume values to "brightness" (positive at metal).

    HU: brightness = HU value (positive for metal).
    LoG sigma=1: brightness = -LoG (LoG is NEGATIVE at metal-bright spots).
    """
    if polarity == "positive":
        return volume_values
    if polarity == "negative":
        return -np.asarray(volume_values, dtype=float)
    raise ValueError(f"unknown polarity {polarity!r}")

def refine_axis_via_centroid(
    volume_arr,
    ras_to_ijk_mat,
    start_ras,
    end_ras,
    *,
    polarity: str = "positive",
    arc_step_mm: float = 0.25,
    cross_radius_mm: float = 6.0,
    cross_pixel_mm: float = 0.5,
    filter_value: float = 500.0,
    weight_offset: float = 1500.0,
    poly_deg: int = 2,
):
    """Polynomial-fit centerline through brightness-mass-weighted
    centroids of perpendicular disks along the straight axis.

    For each arc bin along the straight axis, the per-bin centroid is
    the brightness-weighted center of mass within a radius-
    ``cross_radius_mm`` perpendicular disk. We fit two polynomials —
    du(arc) and dv(arc) of degree ``poly_deg`` — through the valid
    centroid track in the straight-axis perpendicular basis, then
    sample densely.

    Defaults are tuned for HU (positive polarity, filter HU ≥ 500,
    weight = max(0, HU − 1500)). For LoG sigma=1 pass
    ``polarity="negative"`` with ``filter_value=100, weight_offset=200``.

    Returns the refined polyline as Kx3 RAS, or None when fewer than
    ``poly_deg + 1`` cross-sections contain enough metal.
    """
    import numpy as np

    s = np.asarray(start_ras, dtype=float)
    e = np.asarray(end_ras, dtype=float)
    L = float(np.linalg.norm(e - s))
    if L < 5.0:
        return None
    axis = (e - s) / L
    any_v = (
        np.array([1.0, 0, 0])
        if abs(axis[0]) <= 0.9
        else np.array([0, 1.0, 0])
    )
    u_perp = np.cross(axis, any_v); u_perp /= np.linalg.norm(u_perp)
    v_perp = np.cross(axis, u_perp); v_perp /= np.linalg.norm(v_perp)

    grid_axis = np.arange(
        -cross_radius_mm, cross_radius_mm + cross_pixel_mm, cross_pixel_mm,
    )
    DU, DV = np.meshgrid(grid_axis, grid_axis)
    in_disk = DU ** 2 + DV ** 2 <= cross_radius_mm ** 2
    DU = DU[in_disk]; DV = DV[in_disk]

    arcs = np.arange(0, L + 0.001, arc_step_mm)
    n = len(arcs)
    du_track = np.full(n, np.nan, dtype=float)
    dv_track = np.full(n, np.nan, dtype=float)

    from ..volume_sampling import sample_trilinear_batch

    # Pre-compute the disk offsets in 3D as a (M, 3) template that we
    # reuse per arc by adding the arc center.
    M = len(DU)
    disk_offsets = (DU[:, None] * u_perp[None, :]
                    + DV[:, None] * v_perp[None, :])  # (M, 3)

    for i, t in enumerate(arcs):
        center = s + t * axis
        pts = center[None, :] + disk_offsets   # (M, 3)
        raw = sample_trilinear_batch(volume_arr, ras_to_ijk_mat, pts)
        bright = _brightness(raw, polarity)
        mask = np.isfinite(bright) & (bright >= filter_value)
        if int(mask.sum()) < 3:
            continue
        w = np.maximum(bright[mask] - weight_offset, 0.0)
        if w.sum() < 1e-6:
            continue
        du_track[i] = float(np.sum(DU[mask] * w) / np.sum(w))
        dv_track[i] = float(np.sum(DV[mask] * w) / np.sum(w))

    valid = np.isfinite(du_track) & np.isfinite(dv_track)
    if int(valid.sum()) < poly_deg + 1:
        return None

    cu = np.polyfit(arcs[valid], du_track[valid], poly_deg)
    cv = np.polyfit(arcs[valid], dv_track[valid], poly_deg)
    du_smooth = np.polyval(cu, arcs)
    dv_smooth = np.polyval(cv, arcs)

    polyline = np.zeros((n, 3), dtype=float)
    for i, t in enumerate(arcs):
        center = s + t * axis
        polyline[i] = (
            center + du_smooth[i] * u_perp + dv_smooth[i] * v_perp
        )
    return polyline

def sample_disk_along_polyline(
    volume_arr,
    ras_to_ijk_mat,
    polyline,
    *,
    polarity: str = "positive",
    step_mm: float = 0.25,
    disk_radius_mm: float = 1.5,
    n_radii: int = 2,
    n_angles: int = 8,
    total_threshold: float = 1500.0,
):
    """Walk a polyline; per arc bin emit (arc, max_brightness, total_brightness).

    ``max_brightness``: max brightness in the perpendicular disk (the
        same signal ``signal_derived_entry_arc`` consumes for max-disk
        profiles).
    ``total_brightness``: Σ max(0, brightness − total_threshold) over
        disk voxels — the integrated metal mass per disk. Sustained-
        high through bolt + bone (HU) or bolt + wire + contacts (LoG);
        the sharp drop where the integral crosses bolt-end is what
        ``entry_arc_from_metal_mass`` keys on.

    For a STRAIGHT polyline (2-point input ``[start, end]``), this
    walks perpendicular to a fixed axis. For a curved polyline it
    walks perpendicular to the local tangent, allowing curved seeds.
    """
    import numpy as np

    starts, dirs, lens, cum_start = _polyline_segments(polyline)
    total = float(cum_start[-1] + lens[-1])
    arcs = np.arange(0, total + 0.5 * step_mm, step_mm)
    max_b = np.zeros(len(arcs), dtype=float)
    total_b = np.zeros(len(arcs), dtype=float)

    from ..volume_sampling import sample_trilinear_batch

    # Pre-compute disk-offset templates (in u-v perpendicular basis).
    # Each per-arc offset is offset_u * u + offset_v * v in 3D.
    n_per_disk = 1 + n_radii * n_angles
    offset_u = np.zeros(n_per_disk, dtype=float)
    offset_v = np.zeros(n_per_disk, dtype=float)
    idx = 1
    for r_idx in range(1, n_radii + 1):
        rr = disk_radius_mm * r_idx / n_radii
        for a_idx in range(n_angles):
            ang = 2 * np.pi * a_idx / n_angles
            offset_u[idx] = rr * np.cos(ang)
            offset_v[idx] = rr * np.sin(ang)
            idx += 1

    for ai, t in enumerate(arcs):
        center, tangent = _polyline_at_arc_with_tangent(
            starts, dirs, lens, cum_start, float(t),
        )
        any_v = (
            np.array([1.0, 0, 0])
            if abs(tangent[0]) <= 0.9
            else np.array([0, 1.0, 0])
        )
        u = np.cross(tangent, any_v); u /= np.linalg.norm(u)
        v = np.cross(tangent, u);     v /= np.linalg.norm(v)
        # Build (n_per_disk, 3) sample-point batch in one numpy expression.
        pts = (center[None, :]
               + offset_u[:, None] * u[None, :]
               + offset_v[:, None] * v[None, :])
        samples = sample_trilinear_batch(volume_arr, ras_to_ijk_mat, pts)
        s_arr = _brightness(samples, polarity)
        finite = s_arr[np.isfinite(s_arr)]
        max_b[ai] = float(finite.max()) if finite.size else 0.0
        total_b[ai] = float(np.maximum(finite - total_threshold, 0.0).sum())
    return arcs, max_b, total_b

def _polyline_at_arc_with_tangent(starts, dirs, lens, cum_start, arc):
    """Position + tangent at arc-length on the polyline. Helper for
    sample_disk_along_polyline (which needs both)."""
    if arc <= 0:
        return starts[0] + arc * dirs[0], dirs[0]
    total = cum_start[-1] + lens[-1]
    if arc >= total:
        return starts[-1] + (arc - cum_start[-1]) * dirs[-1], dirs[-1]
    idx = int(np.searchsorted(cum_start + lens, arc, side="right"))
    idx = min(idx, len(starts) - 1)
    return starts[idx] + (arc - cum_start[idx]) * dirs[idx], dirs[idx]

def entry_arc_from_metal_mass(
    arcs,
    mass,
    *,
    smooth_sigma_mm: float = 3.5,
    plateau_frac: float = 0.50,
    padding_mm: float = 0.5,
    max_gap_mm: float = 6.0,
    threshold_mode: str = "plateau",
    abs_threshold: float = 500.0,
):
    """Detect bolt-end / entry arc from a metal-mass profile.

    Algorithm:
      1. Gaussian-smooth the mass profile with σ = ``smooth_sigma_mm``.
         When σ matches the library's median contact pitch, this
         averages each shaft contact peak with its adjacent valley,
         pulling the shaft envelope down to ~50% of the bolt's
         sustained-high mass.
      2. Find the global max of the smoothed profile (the bolt's
         bone-collar peak).
      3. Threshold:
            "plateau":  threshold = plateau_frac × peak
            "absolute": threshold = abs_threshold (peak-independent)
      4. Walk forward from the peak. Track the FIRST arc where mass
         drops below threshold. If mass stays below threshold for
         ``max_gap_mm`` of arc, the remembered drop is the bolt-end.
         If mass climbs back above threshold within ``max_gap_mm``,
         the dip was an internal bolt feature (e.g., nut-to-bolt-body
         gap); reset and continue.

    Returns the entry arc (float, mm), or None when no plateau
    detected (mass profile flat, peak below threshold, etc).
    """
    import numpy as np

    arcs = np.asarray(arcs, dtype=float)
    mass = np.asarray(mass, dtype=float)
    if len(arcs) < 5:
        return None
    finite = np.isfinite(mass)
    if not finite.any():
        return None
    step_mm = float(arcs[1] - arcs[0]) if len(arcs) >= 2 else 0.25

    if smooth_sigma_mm > 0:
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(
            np.where(finite, mass, 0.0),
            sigma=max(0.5, smooth_sigma_mm / step_mm),
            mode="nearest",
        )
    else:
        smoothed = np.where(finite, mass, 0.0)

    peak_val = float(smoothed.max())
    if peak_val < 1e-3:
        return None

    if threshold_mode == "absolute":
        threshold = float(abs_threshold)
        if peak_val <= threshold:
            return None
    elif threshold_mode == "plateau":
        threshold = float(plateau_frac) * peak_val
    else:
        raise ValueError(f"unknown threshold_mode {threshold_mode!r}")

    peak_idx = int(np.argmax(smoothed))
    max_below_bins = max(1, int(round(max_gap_mm / step_mm)))

    last_drop = None
    bins_below = 0
    for i in range(peak_idx, len(smoothed)):
        if smoothed[i] >= threshold:
            bins_below = 0
            last_drop = None
        else:
            if last_drop is None:
                last_drop = float(arcs[i])
            bins_below += 1
            if bins_below > max_below_bins:
                return last_drop + float(padding_mm)
    if last_drop is not None:
        return last_drop + float(padding_mm)
    return None

def estimate_bolt_end_from_metal_mass(
    bolt_outer_ras,
    deep_end_ras,
    *,
    features: dict,
    library_models=None,
    cross_radius_mm: float = 6.0,
    signal_disk_radius_mm: float = 1.5,
    plateau_frac: float = 0.50,
    max_gap_mm: float = 6.0,
    padding_mm: float = 0.5,
    threshold_mode: str = "plateau",
    abs_threshold: float = 500.0,
    smooth_sigma_fallback_mm: float = 3.5,
    poly_deg: int = 2,
) -> dict:
    """High-level: bolt-end arc from total HU mass along the trajectory.

    Pipeline (per ``bolt_partition_qc.ipynb``):
      1. Polynomial centerline via ``refine_axis_via_centroid`` on the
         volume's LoG (when available, since LoG sees the wire that
         connects bolt to contacts) or HU otherwise.
      2. Total HU mass per perpendicular disk sampled along BOTH the
         straight axis (``[bolt_outer_ras, deep_end_ras]``) and the
         polynomial centerline. Element-wise max combines the two —
         picks up the bolt regardless of which line is closer to it.
      3. Smooth with σ = library median pitch (or
         ``smooth_sigma_fallback_mm`` if no library) and detect the
         bolt-end via ``entry_arc_from_metal_mass``.

    Args:
        bolt_outer_ras: shallow end of the trajectory axis (RAS, mm).
            The bolt's outer-most tube voxel is a good choice.
        deep_end_ras: deep end. The bolt-side end of the search
            interval — usually the trajectory's deep tip + small pad.
        features: dict from ``rosa_detect.guided_fit_engine.compute_features``.
            Must contain ``ct_arr_kji``, ``ras_to_ijk_mat``; will use
            ``log`` if available for centerline estimation.
        library_models: list of electrode-model dicts; provides the
            median pitch for envelope smoothing. When None, falls back
            to ``smooth_sigma_fallback_mm``.

    Returns:
        dict with:
          "bolt_end_arc_mm": detected arc (mm from bolt_outer_ras) or
              None if not detected.
          "centerline": refined polyline (Kx3 RAS) or None.
          "diagnostics": {
              "arcs_mm", "total_mass_combined", "total_mass_straight",
              "total_mass_polyline", "smooth_sigma_mm", "threshold",
              "peak_mass", "peak_arc_mm",
          }
    """
    import numpy as np

    ct_arr = np.asarray(features["ct_arr_kji"], dtype=np.float32)
    r2i = np.asarray(features["ras_to_ijk_mat"], dtype=float)
    log_arr = features.get("log")

    s = np.asarray(bolt_outer_ras, dtype=float)
    e = np.asarray(deep_end_ras, dtype=float)

    # Centerline. Prefer LoG (sees the wire) when available; HU as
    # fallback.
    if log_arr is not None:
        centerline = refine_axis_via_centroid(
            log_arr, r2i, s, e,
            polarity="negative",
            cross_radius_mm=cross_radius_mm,
            filter_value=100.0, weight_offset=200.0,
            poly_deg=poly_deg,
        )
    else:
        centerline = refine_axis_via_centroid(
            ct_arr, r2i, s, e,
            polarity="positive",
            cross_radius_mm=cross_radius_mm,
            filter_value=500.0, weight_offset=1500.0,
            poly_deg=poly_deg,
        )

    # Total HU mass on STRAIGHT axis.
    straight_polyline = np.stack([s, e])
    arc_s, _max_s, total_s = sample_disk_along_polyline(
        ct_arr, r2i, straight_polyline,
        polarity="positive",
        disk_radius_mm=signal_disk_radius_mm,
        total_threshold=1500.0,
    )

    # Total HU mass on CENTERLINE (when available).
    if centerline is not None:
        arc_r, _max_r, total_r = sample_disk_along_polyline(
            ct_arr, r2i, centerline,
            polarity="positive",
            disk_radius_mm=signal_disk_radius_mm,
            total_threshold=1500.0,
        )
        # Re-grid polyline mass onto straight-axis u-coordinate.
        u_r = _polyline_arc_to_u(centerline, arc_r, s, (e - s) / np.linalg.norm(e - s))
        total_polyline_on_u = np.interp(arc_s, u_r, total_r, left=0.0, right=0.0)
        total_combined = np.maximum(total_s, total_polyline_on_u)
    else:
        total_polyline_on_u = np.zeros_like(arc_s)
        total_combined = total_s

    # σ = library median pitch (envelope filter cancelling shaft
    # oscillations); fallback to fixed value if no library.
    pitch_mm = median_library_pitch_mm(library_models)
    sigma_mm = pitch_mm if pitch_mm else smooth_sigma_fallback_mm

    bolt_end_arc = entry_arc_from_metal_mass(
        arc_s, total_combined,
        smooth_sigma_mm=sigma_mm,
        plateau_frac=plateau_frac,
        padding_mm=padding_mm,
        max_gap_mm=max_gap_mm,
        threshold_mode=threshold_mode,
        abs_threshold=abs_threshold,
    )

    peak_idx = int(np.argmax(total_combined)) if total_combined.size else 0
    peak_mass = float(total_combined[peak_idx]) if total_combined.size else 0.0
    peak_arc = float(arc_s[peak_idx]) if total_combined.size else 0.0
    threshold_used = (
        plateau_frac * peak_mass if threshold_mode == "plateau" else abs_threshold
    )
    return {
        "bolt_end_arc_mm": bolt_end_arc,
        "centerline": centerline,
        "diagnostics": {
            "arcs_mm": arc_s,
            "total_mass_combined": total_combined,
            "total_mass_straight": total_s,
            "total_mass_polyline": total_polyline_on_u,
            "smooth_sigma_mm": sigma_mm,
            "threshold": threshold_used,
            "peak_mass": peak_mass,
            "peak_arc_mm": peak_arc,
        },
    }

def _polyline_arc_to_u(polyline, arcs, slab_origin_ras, slab_axis):
    """Convert polyline arc-length positions to slab u-coordinate
    (projection onto the straight axis from slab_origin_ras).
    """
    starts, dirs, lens, cum_start = _polyline_segments(polyline)
    out = np.zeros(len(arcs), dtype=float)
    for i, t in enumerate(arcs):
        if t <= 0:
            pos = starts[0] + t * dirs[0]
        elif t >= cum_start[-1] + lens[-1]:
            pos = starts[-1] + (t - cum_start[-1]) * dirs[-1]
        else:
            idx = int(np.searchsorted(cum_start + lens, t, side="right"))
            idx = min(idx, len(starts) - 1)
            pos = starts[idx] + (t - cum_start[idx]) * dirs[idx]
        out[i] = float(
            (pos - np.asarray(slab_origin_ras, float))
            @ np.asarray(slab_axis, float)
        )
    return out
