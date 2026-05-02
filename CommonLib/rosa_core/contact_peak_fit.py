"""Peak-driven contact detection from a post-op CT.

Given a shank axis (start, end in RAS) and a 3-D scalar volume (LoG σ=1
by default), this module samples along the axis, picks peaks, and
matches the peak pattern against each candidate electrode model in the
library. It returns the best-fit model plus contact positions anchored
on the detected peaks (not the model's nominal offsets), with model-
nominal fallback for missing peaks.

Companion to ``rosa_core.generate_contacts``, which is the model-driven
path. Peak-driven results reflect real imaged contacts even when the
electrode is slightly curved, a contact has drifted, or the manual
model assignment is wrong.

Probe results (tests/deep_core/probe_contact_peak_filters.py, 2026-04-19):

    # T22 (9 shanks, GT axis)        # T2 (12 shanks, GT axis)
    # LoG σ=1 cyl r=2 mm min over 6-point ring + center:
    # median err 0.90 mm, 50% match rate      0.80 mm, 82% match rate

LoG σ=1 is chosen because Auto Fit already computes it (stashed on
the Slicer scene as ``<CT>_ContactPitch_LoG_sigma1``). The engine accepts
the precomputed volume via ``log_volume_kji``; when absent the caller
computes it with ``sitk.LaplacianRecursiveGaussian(img, sigma=1.0)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .transforms import lps_to_ras_point
from .types import ContactRecord, ElectrodeModel, TrajectoryRecord
from .volume_sampling import sample_trilinear_at_ras


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------

PEAK_STEP_MM = 0.25
PEAK_DISK_RADIUS_MM = 2.0  # disk radius for perp-offset sampling
PEAK_DISK_N_RADII = 4      # concentric rings inside the disk
PEAK_DISK_N_ANGLES = 8     # angular samples per ring
PEAK_MIN_SEPARATION_MM = 2.0
PEAK_AMP_MIN_ABS = 100.0
MODEL_MATCH_TOL_MM = 1.25
FALLBACK_MIN_COVERAGE = 0.5
FALLBACK_MAX_RESIDUAL_MM = 1.5


@dataclass
class PeakFitResult:
    """Return payload of ``detect_contacts_on_axis``.

    - ``model_id``: the best-matching electrode model id (empty string if
      no usable match was found — caller should fall back to the model-
      driven path).
    - ``positions_ras``: list of RAS [x, y, z] per contact, one per slot
      in ``contact_center_offsets_from_tip_mm``. Peaks that matched a
      model slot sit on the detected arc-length; unmatched slots fall
      back to the nominal model offset from the fitted tip.
    - ``peak_detected``: parallel bool list marking which contact
      positions came from real peaks vs. nominal fallback.
    - ``tip_arclen_mm``: arc-length of the fitted electrode tip from
      the axis origin (``start_ras``).
    - ``tip_direction``: forward direction (``target`` side) unit vec.
    - ``n_peaks_found``: how many peaks the 1-D peak picker returned.
    - ``n_model_slots``: model's contact count.
    - ``n_matched``: peaks matched to model slots within tolerance.
    - ``mean_residual_mm``: mean abs residual over matched peaks.
    - ``rejected_reason``: non-empty string when we returned no match
      because the best candidate was too weak.
    """

    model_id: str
    positions_ras: list[list[float]]
    peak_detected: list[bool]
    tip_arclen_mm: float
    tip_direction: list[float]
    n_peaks_found: int
    n_model_slots: int
    n_matched: int
    mean_residual_mm: float
    rejected_reason: str = ""


# ---------------------------------------------------------------------
# Axis-profile sampling
# ---------------------------------------------------------------------

def _unit(v):
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if n <= 1e-9:
        raise ValueError("Zero-length axis vector")
    return v / n


def _orthonormal_basis(direction_unit):
    any_vec = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(direction_unit, any_vec)) > 0.9:
        any_vec = np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(direction_unit, any_vec))
    v = _unit(np.cross(direction_unit, u))
    return u, v


def sample_axis_profile(volume_kji, ras_to_ijk_mat, start_ras, end_ras,
                        step_mm=PEAK_STEP_MM,
                        disk_radius_mm=PEAK_DISK_RADIUS_MM,
                        n_radii=PEAK_DISK_N_RADII,
                        n_angles=PEAK_DISK_N_ANGLES,
                        reducer="min"):
    """Sample a 1-D profile along the axis from ``start_ras`` to
    ``end_ras``. At each axial step, sample a perpendicular disk of
    radius ``disk_radius_mm`` (``n_radii`` concentric rings of
    ``n_angles`` points each, plus the on-axis center) and reduce to
    one value with ``reducer`` (``'min'`` / ``'max'`` / ``'mean'``).

    A disk (not a single ring) absorbs 0.5–1.5 mm of axis drift even
    when the real contact sits between the ring and the axis — a ring
    at r=2 mm alone can miss a contact at r=0.8 mm.

    Returns:
        arc_mm: 1-D array of arc-length values (mm from start_ras).
        profile: 1-D array of sampled values (NaN for out-of-bounds).
    """
    start = np.asarray(start_ras, dtype=float).reshape(3)
    end = np.asarray(end_ras, dtype=float).reshape(3)
    axis = end - start
    L = float(np.linalg.norm(axis))
    if L < 1e-3:
        raise ValueError("Trajectory length must be > 0")
    axis = axis / L
    n_steps = int(L / step_mm) + 1
    arc_mm = np.arange(n_steps, dtype=float) * step_mm

    u_basis, v_basis = _orthonormal_basis(axis)
    disk_offsets: list[np.ndarray] = [np.zeros(3, dtype=float)]
    if disk_radius_mm > 0.0 and n_radii > 0 and n_angles > 0:
        for r_idx in range(1, n_radii + 1):
            r = disk_radius_mm * r_idx / n_radii
            for a_idx in range(n_angles):
                ang = 2.0 * np.pi * a_idx / n_angles
                disk_offsets.append(
                    r * (np.cos(ang) * u_basis + np.sin(ang) * v_basis)
                )

    profile = np.full(n_steps, np.nan, dtype=float)
    for idx in range(n_steps):
        center = start + arc_mm[idx] * axis
        samples = [
            sample_trilinear_at_ras(volume_kji, ras_to_ijk_mat, center + off)
            for off in disk_offsets
        ]
        s = np.asarray(samples, dtype=float)
        s = s[np.isfinite(s)]
        if s.size == 0:
            continue
        if reducer == "max":
            profile[idx] = float(s.max())
        elif reducer == "min":
            profile[idx] = float(s.min())
        else:
            profile[idx] = float(s.mean())
    return arc_mm, profile


# ---------------------------------------------------------------------
# 1-D peak picking
# ---------------------------------------------------------------------

def detect_peaks_1d(profile, step_mm, polarity="min",
                    min_amplitude=PEAK_AMP_MIN_ABS,
                    min_separation_mm=PEAK_MIN_SEPARATION_MM):
    """Return arc-length positions (mm) of peaks in a 1-D profile.

    ``polarity='min'``: local minima whose value is ≤ -min_amplitude
    (LoG signal on bright metal goes strongly negative).
    ``polarity='max'``: local maxima whose value is ≥ +min_amplitude.

    Uses greedy non-max suppression at a ``min_separation_mm`` floor —
    peaks closer than that are collapsed to the strongest one. Returns
    positions sorted by arc-length.
    """
    x = np.asarray(profile, dtype=float)
    n = x.size
    if n < 3:
        return []
    # Fill NaN so we don't crash the comparison; NaN masks turn into
    # non-peaks implicitly because any comparison with NaN is False.
    with np.errstate(invalid="ignore"):
        if polarity == "min":
            is_peak = np.zeros(n, dtype=bool)
            is_peak[1:-1] = (x[1:-1] < x[:-2]) & (x[1:-1] < x[2:])
            strong = x <= -abs(min_amplitude)
            amp = -x  # selection priority: most negative first.
        else:
            is_peak = np.zeros(n, dtype=bool)
            is_peak[1:-1] = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])
            strong = x >= abs(min_amplitude)
            amp = x
    candidate = np.where(is_peak & strong)[0]
    if candidate.size == 0:
        return []
    order = candidate[np.argsort(-amp[candidate])]
    min_gap_steps = max(1, int(round(min_separation_mm / step_mm)))
    kept = []
    for idx in order:
        if all(abs(int(idx) - int(j)) >= min_gap_steps for j in kept):
            kept.append(int(idx))
    kept.sort()
    return [float(i) * step_mm for i in kept]


# ---------------------------------------------------------------------
# Model matching
# ---------------------------------------------------------------------

def _match_peaks_to_offsets(peaks_sorted, offsets_from_tip_mm,
                            axis_len_mm, offset_sign,
                            tol_mm=MODEL_MATCH_TOL_MM):
    """Find the tip arc-length that best aligns ``offsets`` with
    ``peaks_sorted``. Returns (tip_arclen_mm, matches, residuals) where
    ``matches`` is a list of (model_slot_idx, peak_idx_in_sorted_list)
    pairs and residuals is a list of |peak - predicted| per match.

    ``offset_sign`` encodes the axis orientation:
        +1  — tip sits on the ``start`` side; contacts grow toward ``end``.
        -1  — tip sits on the ``end`` side; contacts grow toward ``start``.

    Algorithm: for every (peak_i = slot_j) hypothesis set
    tip = peaks[i] - offset_sign * offsets[j], predict all slots, and
    greedily match to available peaks within ``tol_mm``. Score by
    (matched, mean_residual). Keep the hypothesis with the most
    matches; tie-break on residual.
    """
    peaks = list(peaks_sorted)
    offsets = list(offsets_from_tip_mm)
    n_slots = len(offsets)
    n_peaks = len(peaks)
    if n_slots == 0 or n_peaks == 0:
        return 0.0, [], []

    best_key = (-1, -float("inf"))  # (matched, -mean_res) — maximize.
    best_tip = 0.0
    best_matches: list[tuple[int, int]] = []
    best_residuals: list[float] = []

    for peak_i in range(n_peaks):
        for slot_j in range(n_slots):
            tip_arclen = peaks[peak_i] - offset_sign * offsets[slot_j]
            # Tip may be outside [0, L]; allow ±5 mm headroom so an
            # electrode extending slightly past the sampled axis still
            # produces candidates.
            if tip_arclen < -5.0 or tip_arclen > axis_len_mm + 5.0:
                continue
            predicted = [tip_arclen + offset_sign * off for off in offsets]
            used_peaks: set[int] = set()
            matches: list[tuple[int, int]] = []
            residuals: list[float] = []
            for s_idx, pred in enumerate(predicted):
                best_pi = -1
                best_d: Optional[float] = None
                for p_idx, p in enumerate(peaks):
                    if p_idx in used_peaks:
                        continue
                    d = abs(p - pred)
                    if d <= tol_mm and (best_d is None or d < best_d):
                        best_pi = p_idx
                        best_d = d
                if best_pi >= 0 and best_d is not None:
                    used_peaks.add(best_pi)
                    matches.append((s_idx, best_pi))
                    residuals.append(float(best_d))
            matched = len(matches)
            mean_res = (sum(residuals) / matched) if matched else float("inf")
            key = (matched, -mean_res)
            if key > best_key:
                best_key = key
                best_tip = float(tip_arclen)
                best_matches = matches
                best_residuals = residuals
    return best_tip, best_matches, best_residuals


def _match_tip_direction(peaks_sorted, offsets_from_tip_mm, axis_len_mm,
                          tol_mm=MODEL_MATCH_TOL_MM):
    """Try matching with both axis orientations (tip on the ``start``
    side vs. the ``end`` side). Returns
    (tip_arclen_mm, matches, residuals, tip_at).

    ``tip_at='start'``: tip on axis-start side; contact k at
    arc = tip_arclen + offsets[k].
    ``tip_at='end'``: tip on axis-end side; contact k at
    arc = tip_arclen - offsets[k].

    Peak indices in ``matches`` always refer to positions in the sorted
    ``peaks_sorted`` list so the caller can resolve arc-lengths directly.
    """
    tip_start, matches_start, residuals_start = _match_peaks_to_offsets(
        peaks_sorted, offsets_from_tip_mm, axis_len_mm, +1.0, tol_mm=tol_mm,
    )
    n_start = len(matches_start)
    res_start = (sum(residuals_start) / n_start) if n_start else float("inf")

    tip_end, matches_end, residuals_end = _match_peaks_to_offsets(
        peaks_sorted, offsets_from_tip_mm, axis_len_mm, -1.0, tol_mm=tol_mm,
    )
    n_end = len(matches_end)
    res_end = (sum(residuals_end) / n_end) if n_end else float("inf")

    start_score = (n_start, -res_start)
    end_score = (n_end, -res_end)
    if end_score > start_score:
        return tip_end, matches_end, residuals_end, "end"
    return tip_start, matches_start, residuals_start, "start"


def _orient_axis_shallow_to_deep(start_ras, end_ras, dist_arr=None,
                                  ras_to_ijk_mat=None):
    """Return (shallow_ras, deep_ras). Uses head-distance volume when
    provided; otherwise falls back to identity (treats start as shallow).
    """
    start = np.asarray(start_ras, dtype=float).reshape(3)
    end = np.asarray(end_ras, dtype=float).reshape(3)
    if dist_arr is None or ras_to_ijk_mat is None:
        return start, end
    try:
        ks, js, is_ = _ras_to_kji_pt(ras_to_ijk_mat, start)
        ke, je, ie_ = _ras_to_kji_pt(ras_to_ijk_mat, end)
        d_start = _sample_trilinear(dist_arr, ks, js, is_)
        d_end = _sample_trilinear(dist_arr, ke, je, ie_)
    except Exception:
        return start, end
    if np.isfinite(d_start) and np.isfinite(d_end) and d_start > d_end:
        return end, start
    return start, end


def fit_best_electrode(peaks_arclen_mm,
                       axis_len_mm,
                       models_by_id,
                       candidate_ids=None,
                       tol_mm=MODEL_MATCH_TOL_MM,
                       min_coverage_gate=0.6):
    """Iterate candidate models; return (best_id, fit_info).

    ``fit_info`` is a dict:
      - tip_arclen_mm
      - tip_at: 'end' or 'start' (see ``_match_tip_direction``)
      - matches: list of (slot_idx, peak_idx)
      - residuals: list of float
      - mean_residual_mm, n_matched, n_slots
    """
    if candidate_ids is None:
        candidate_ids = sorted(models_by_id.keys())

    peaks_sorted = sorted(peaks_arclen_mm)

    best_id = ""
    # Scoring: (coverage_bin, n_matched, -mean_res, -n_slots).
    # Coverage bin is n_matched / n_slots rounded to the nearest 5 %.
    # Primary on coverage so a 15-slot model that fills 100 % of its
    # slots beats an 18-slot model that only fills 89 % of its slots
    # with the same 15-ish peaks (AM-18 vs. CM-15 on Dixi group
    # electrodes). Within a coverage bin, prefer the model that
    # explains more detected peaks, then tighter residual, then
    # shortest explaining electrode. Coverage below a hard gate is
    # rejected so scattered 5/18-slot matches can't sneak through.
    best_score = (-1.0, -1, -float("inf"), 0)
    best_info = {
        "tip_arclen_mm": 0.0,
        "tip_at": "end",
        "matches": [],
        "residuals": [],
        "mean_residual_mm": float("inf"),
        "n_matched": 0,
        "n_slots": 0,
    }
    for mid in candidate_ids:
        model = models_by_id.get(mid, {})
        offsets = model.get("contact_center_offsets_from_tip_mm")
        if not offsets:
            continue
        tip_arclen, matches, residuals, tip_at = _match_tip_direction(
            peaks_sorted, offsets, axis_len_mm, tol_mm=tol_mm,
        )
        n_match = len(matches)
        n_slots = len(offsets)
        mean_res = (sum(residuals) / n_match) if n_match else float("inf")
        coverage = n_match / max(1, n_slots)
        if coverage < min_coverage_gate:
            continue
        coverage_bin = round(coverage * 20) / 20.0
        score = (coverage_bin, n_match, -mean_res, -n_slots)
        if score > best_score:
            best_score = score
            best_id = mid
            best_info = {
                "tip_arclen_mm": float(tip_arclen),
                "tip_at": tip_at,
                "matches": matches,
                "residuals": residuals,
                "mean_residual_mm": float(mean_res),
                "n_matched": int(n_match),
                "n_slots": int(n_slots),
            }
    return best_id, best_info


# ---------------------------------------------------------------------
# Vendor filtering
# ---------------------------------------------------------------------

def candidate_ids_for_vendors(models_by_id, vendors=None):
    """Filter models by vendor prefix tokens (e.g., {'DIXI', 'PMT'}).

    Matches against the leading token in ``id`` before the first '-'.
    When vendors is None or empty, returns all model ids sorted.
    """
    if not vendors:
        return sorted(models_by_id.keys())
    vendor_set = {str(v).strip().upper() for v in vendors}
    out = []
    for mid in models_by_id.keys():
        prefix = str(mid).split("-", 1)[0].upper()
        if prefix in vendor_set:
            out.append(mid)
    return sorted(out)


# ---------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------

def detect_contacts_on_axis(start_ras,
                            end_ras,
                            log_volume_kji,
                            ras_to_ijk_mat,
                            models_by_id,
                            candidate_ids=None,
                            *,
                            restrict_to_model_id=None,
                            model_free=False,
                            n_contacts_target=None,
                            dist_arr_kji=None,
                            step_mm=PEAK_STEP_MM,
                            disk_radius_mm=PEAK_DISK_RADIUS_MM,
                            n_radii=PEAK_DISK_N_RADII,
                            n_angles=PEAK_DISK_N_ANGLES,
                            amp_min_abs=PEAK_AMP_MIN_ABS,
                            peak_separation_mm=PEAK_MIN_SEPARATION_MM,
                            tol_mm=MODEL_MATCH_TOL_MM,
                            fallback_min_coverage=FALLBACK_MIN_COVERAGE,
                            fallback_max_residual_mm=FALLBACK_MAX_RESIDUAL_MM):
    """Full peak-driven detection on one trajectory axis.

    Args:
        start_ras, end_ras: axis endpoints in RAS (skull_entry and deep
            tip are preferred; orientation is auto-corrected if
            ``dist_arr_kji`` is provided).
        log_volume_kji: 3-D np.float32 array of LoG σ=1 (KJI order).
        ras_to_ijk_mat: 4x4 RAS→IJK matrix.
        models_by_id: library models keyed by id.
        candidate_ids: optional pre-filtered model ids to consider.
        dist_arr_kji: optional head-distance volume used to orient
            shallow→deep.

    Returns: ``PeakFitResult``. ``model_id=""`` and a non-empty
    ``rejected_reason`` means the caller should fall back to the
    model-driven path.
    """
    shallow, deep = _orient_axis_shallow_to_deep(
        start_ras, end_ras, dist_arr_kji, ras_to_ijk_mat,
    )
    axis = deep - shallow
    axis_len_mm = float(np.linalg.norm(axis))
    if axis_len_mm < 5.0:
        return PeakFitResult(
            model_id="", positions_ras=[], peak_detected=[],
            tip_arclen_mm=0.0, tip_direction=[1.0, 0.0, 0.0],
            n_peaks_found=0, n_model_slots=0, n_matched=0,
            mean_residual_mm=float("inf"),
            rejected_reason="axis_len_too_short",
        )
    axis_unit = axis / axis_len_mm

    arc_mm, profile = sample_axis_profile(
        log_volume_kji, ras_to_ijk_mat, shallow, deep,
        step_mm=step_mm,
        disk_radius_mm=disk_radius_mm, n_radii=n_radii, n_angles=n_angles,
        reducer="min",
    )
    peaks = detect_peaks_1d(
        profile, step_mm, polarity="min",
        min_amplitude=amp_min_abs, min_separation_mm=peak_separation_mm,
    )

    if not peaks:
        return PeakFitResult(
            model_id="", positions_ras=[], peak_detected=[],
            tip_arclen_mm=0.0, tip_direction=axis_unit.tolist(),
            n_peaks_found=0, n_model_slots=0, n_matched=0,
            mean_residual_mm=float("inf"),
            rejected_reason="no_peaks",
        )

    if model_free:
        # Model-free path: emit detected peaks as contacts directly.
        # No slot matching, no library lookup. ``n_contacts_target``
        # caps the count by keeping the strongest |amplitude| peaks
        # (LoG signal at metal goes strongly negative, so |profile|
        # ranks brightness directly). Used by the GT-annotation
        # workflow when the user knows the contact count but doesn't
        # want to bias placement to a model's exact pitch pattern.
        if n_contacts_target is not None and len(peaks) > int(n_contacts_target):
            profile_arr = np.asarray(profile, dtype=float)
            peak_amps = []
            for arc in peaks:
                idx = int(round(arc / step_mm))
                idx = max(0, min(profile_arr.size - 1, idx))
                peak_amps.append(abs(float(profile_arr[idx])))
            order = sorted(
                range(len(peaks)), key=lambda k: -peak_amps[k],
            )[: int(n_contacts_target)]
            kept_arcs = sorted(peaks[k] for k in order)
        else:
            kept_arcs = list(peaks)

        shallow_arr = np.asarray(shallow, dtype=float)
        positions_ras = [
            (shallow_arr + arc * axis_unit).tolist() for arc in kept_arcs
        ]
        return PeakFitResult(
            model_id="manual",
            positions_ras=positions_ras,
            peak_detected=[True] * len(positions_ras),
            tip_arclen_mm=float(kept_arcs[0]) if kept_arcs else 0.0,
            tip_direction=axis_unit.tolist(),
            n_peaks_found=len(peaks),
            n_model_slots=len(positions_ras),
            n_matched=len(positions_ras),
            mean_residual_mm=0.0,
            rejected_reason="",
        )

    if restrict_to_model_id:
        # User-assigned model: match peaks only against this one
        # pattern. Coverage gate drops to 0.3 in this branch because
        # the caller has decided which electrode is there — if it's
        # only half-detected we still want to anchor what we can.
        ids = [str(restrict_to_model_id)]
        min_coverage_gate = 0.3
    elif candidate_ids:
        ids = list(candidate_ids)
        min_coverage_gate = 0.6
    else:
        ids = sorted(models_by_id.keys())
        min_coverage_gate = 0.6
    best_id, info = fit_best_electrode(
        peaks, axis_len_mm, models_by_id, candidate_ids=ids, tol_mm=tol_mm,
        min_coverage_gate=min_coverage_gate,
    )
    if not best_id:
        return PeakFitResult(
            model_id="", positions_ras=[], peak_detected=[],
            tip_arclen_mm=0.0, tip_direction=axis_unit.tolist(),
            n_peaks_found=len(peaks), n_model_slots=0, n_matched=0,
            mean_residual_mm=float("inf"),
            rejected_reason="no_candidate_matched",
        )

    offsets = list(models_by_id[best_id]["contact_center_offsets_from_tip_mm"])
    n_slots = len(offsets)
    n_matched = int(info["n_matched"])
    mean_res = float(info["mean_residual_mm"])
    coverage = n_matched / max(1, n_slots)
    if coverage < fallback_min_coverage or mean_res > fallback_max_residual_mm:
        return PeakFitResult(
            model_id="", positions_ras=[], peak_detected=[],
            tip_arclen_mm=float(info["tip_arclen_mm"]),
            tip_direction=axis_unit.tolist(),
            n_peaks_found=len(peaks), n_model_slots=n_slots,
            n_matched=n_matched, mean_residual_mm=mean_res,
            rejected_reason=(
                f"coverage={coverage:.2f} "
                f"mean_res={mean_res:.2f}mm best={best_id}"
            ),
        )

    # Build contact arc-lengths: matched slots → detected peak,
    # unmatched slots → nominal offset from the fitted tip.
    tip_arclen = float(info["tip_arclen_mm"])
    tip_at = info["tip_at"]
    offset_sign = +1.0 if tip_at == "start" else -1.0
    matched_peak_by_slot = {s: p for s, p in info["matches"]}
    # Match indices refer to the sorted peak list inside
    # ``fit_best_electrode`` — mirror that here.
    peaks_sorted = sorted(peaks)

    contact_arcs: list[float] = []
    detected_flags: list[bool] = []
    for s_idx, off in enumerate(offsets):
        nominal_arc = tip_arclen + offset_sign * off
        if s_idx in matched_peak_by_slot:
            p_idx = matched_peak_by_slot[s_idx]
            detected_arc = float(peaks_sorted[p_idx])
            contact_arcs.append(detected_arc)
            detected_flags.append(True)
        else:
            contact_arcs.append(float(nominal_arc))
            detected_flags.append(False)

    # Convert to RAS points.
    shallow_arr = np.asarray(shallow, dtype=float)
    positions_ras = [
        (shallow_arr + arc * axis_unit).tolist() for arc in contact_arcs
    ]
    return PeakFitResult(
        model_id=best_id,
        positions_ras=positions_ras,
        peak_detected=detected_flags,
        tip_arclen_mm=tip_arclen,
        tip_direction=axis_unit.tolist(),
        n_peaks_found=len(peaks),
        n_model_slots=n_slots,
        n_matched=n_matched,
        mean_residual_mm=mean_res,
        rejected_reason="",
    )


# ---------------------------------------------------------------------
# Bridge to the shared contacts schema
# ---------------------------------------------------------------------

def ras_contacts_to_contact_records(result,
                                    trajectory,
                                    tip_at_for_schema="target"):
    """Convert a ``PeakFitResult`` into the ``ContactRecord`` list used
    by ``rosa_core.generate_contacts``. Positions are flipped RAS→LPS
    via ``lps_to_ras_point`` (the symmetric flip) so they land in the
    same LPS space as model-driven contacts.
    """
    label_prefix = trajectory["name"]
    model_id = result.model_id
    out: list[ContactRecord] = []
    for idx, (ras, detected) in enumerate(
        zip(result.positions_ras, result.peak_detected), start=1,
    ):
        lps = lps_to_ras_point(list(ras))  # symmetric flip RAS↔LPS.
        rec: ContactRecord = {
            "trajectory": label_prefix,
            "model_id": model_id,
            "index": idx,
            "label": f"{label_prefix}{idx}",
            "position_lps": lps,
            "tip_at": tip_at_for_schema,
        }
        # Extra keys (not part of the TypedDict but Slicer consumers
        # tolerate them) carry peak-detected provenance for downstream
        # QC. Casting to plain types for JSON round-trips.
        rec["peak_detected"] = bool(detected)  # type: ignore[typeddict-item]
        out.append(rec)
    return out
