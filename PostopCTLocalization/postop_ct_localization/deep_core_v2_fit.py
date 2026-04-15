"""Bolt-first trajectory emission for deep_core_v2.

Two independent fit paths per bolt candidate — the pipeline calls
``fit_bolt_trajectory`` which dispatches based on
``model_fit.v2_fit_mode``:

- ``"two_threshold"`` (default): walks the axis inward from the bolt
  center through a **loose** metal mask (``mask.contact_probe_hu``,
  default 1000 HU) with a gap tolerance, stopping at the deepest
  sample that's still contiguous with the bolt. Validates
  intracranial span against the electrode library.

- ``"intensity_peaks"``: builds a 1D max-HU profile along the axis
  using a 5 mm-diameter disc perpendicular to the axis, detects peaks
  (local maxima above a fraction of the profile max), and fits each
  library electrode's known contact-spacing pattern to the observed
  peak positions. Best-scoring model determines the deep tip and the
  best_model_id. No thresholding of individual voxels is needed — the
  fit uses the relative peak structure.

Both paths take the bolt's RANSAC axis and center as ground truth (no
PCA refit) and anchor the shallow endpoint to the bolt position via
``model_fit.bolt_endpoint_offset_mm``.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from . import deep_core_axis_reconstruction as axr


# Fixed axial walk extent (mm from bolt center, + is scalp, - is deep).
BOLT_INWARD_WALK_MM = 110.0
BOLT_OUTWARD_WALK_MM = 5.0


# ---------------------------------------------------------------------------
# Axis orientation
# ---------------------------------------------------------------------------

def _orient_bolt_axis_toward_scalp(
    axis_ras: np.ndarray,
    center_ras: np.ndarray,
    head_distance_map_kji: np.ndarray,
    ras_to_ijk_fn,
) -> np.ndarray:
    """Flip the axis so +axis points toward the scalp (lower head_distance)."""
    shape = head_distance_map_kji.shape

    def hd(pt):
        ijk = np.asarray(ras_to_ijk_fn(pt), dtype=float)
        kji = np.rint([ijk[2], ijk[1], ijk[0]]).astype(int)
        if any(kji[ax] < 0 or kji[ax] >= shape[ax] for ax in range(3)):
            return float("nan")
        return float(head_distance_map_kji[kji[0], kji[1], kji[2]])

    hd_plus = hd(center_ras + 20.0 * axis_ras)
    hd_minus = hd(center_ras - 20.0 * axis_ras)
    if np.isfinite(hd_plus) and np.isfinite(hd_minus) and hd_minus < hd_plus:
        return -axis_ras
    return axis_ras


# ---------------------------------------------------------------------------
# Lateral tube sampling helpers
# ---------------------------------------------------------------------------

def _build_disc_offsets(axis: np.ndarray, radius_mm: float, n_samples: int) -> np.ndarray:
    """Return (N, 3) ring offsets perpendicular to ``axis`` at ``radius_mm``."""
    u, v = axr._perp_frame(axis)
    angles = np.linspace(0.0, 2.0 * np.pi, max(3, n_samples), endpoint=False)
    return np.stack(
        [radius_mm * (np.cos(a) * u + np.sin(a) * v) for a in angles],
        axis=0,
    )


def _build_filled_disc_offsets(axis: np.ndarray, radius_mm: float) -> np.ndarray:
    """Return (N, 3) offsets filling a disc of ``radius_mm`` on a ~1mm grid."""
    u, v = axr._perp_frame(axis)
    step = 1.0
    coords = np.arange(-radius_mm, radius_mm + 0.5 * step, step)
    pts = []
    for du in coords:
        for dv in coords:
            if du * du + dv * dv <= radius_mm * radius_mm:
                pts.append(du * u + dv * v)
    if not pts:
        pts.append(np.zeros(3))
    return np.stack(pts, axis=0)


def _max_hu_in_disc(
    center_pt_ras: np.ndarray,
    offsets: np.ndarray,
    arr_kji: np.ndarray,
    ras_to_ijk_fn,
) -> float:
    shape = arr_kji.shape
    best = -np.inf
    for off in offsets:
        s = center_pt_ras + off
        ijk = np.asarray(ras_to_ijk_fn(s), dtype=float)
        kji = np.rint([ijk[2], ijk[1], ijk[0]]).astype(int)
        if any(kji[ax] < 0 or kji[ax] >= shape[ax] for ax in range(3)):
            continue
        hu = float(arr_kji[kji[0], kji[1], kji[2]])
        if hu > best:
            best = hu
    return best


# ---------------------------------------------------------------------------
# Approach A: two-threshold walk
# ---------------------------------------------------------------------------

def _find_deep_tip_by_contiguous_metal(
    center_ras: np.ndarray,
    axis_ras: np.ndarray,
    arr_kji: np.ndarray,
    ras_to_ijk_fn,
    *,
    contact_hu: float,
    tube_radius_mm: float,
    inward_walk_mm: float,
    step_mm: float,
    max_gap_mm: float,
) -> float | None:
    """Walk from the bolt center along ``-axis`` (deep) and return the
    deepest axial ``t`` (negative) that is still connected to the bolt
    via metal samples above ``contact_hu`` within ``tube_radius_mm``
    of the axis, under a ``max_gap_mm`` contiguity rule.

    Returns ``None`` if no metal is contiguous with the bolt.
    """
    offsets = _build_disc_offsets(axis_ras, tube_radius_mm, n_samples=12)
    n_steps = int(round(inward_walk_mm / step_mm))
    deepest_t = None
    gap_run = 0
    max_gap_steps = max(1, int(round(max_gap_mm / step_mm)))
    for i in range(n_steps + 1):
        t = -float(i) * step_mm
        pt = center_ras + t * axis_ras
        max_hu = _max_hu_in_disc(pt, offsets, arr_kji, ras_to_ijk_fn)
        if max_hu >= contact_hu:
            deepest_t = t
            gap_run = 0
        else:
            gap_run += 1
            if gap_run > max_gap_steps:
                break
    return deepest_t


def _fit_two_threshold(
    bolt,
    fit: "axr.AxisFit",
    arr_kji: np.ndarray,
    head_distance_map_kji: np.ndarray,
    ras_to_ijk_fn,
    library_models: list[dict[str, Any]],
    library_span_bounds: tuple[float, float],
    cfg,
) -> dict[str, Any] | None:
    contact_hu = float(getattr(cfg, "v2_contact_hu", 1000.0))
    tube_radius = float(getattr(cfg, "v2_contact_probe_radius_mm", 2.5))
    max_gap_mm = float(getattr(cfg, "v2_contact_max_gap_mm", 8.0))
    step_mm = float(getattr(cfg, "extension_step_mm", 0.5))

    t_deep = _find_deep_tip_by_contiguous_metal(
        fit.center, fit.axis, arr_kji, ras_to_ijk_fn,
        contact_hu=contact_hu,
        tube_radius_mm=tube_radius,
        inward_walk_mm=BOLT_INWARD_WALK_MM,
        step_mm=step_mm,
        max_gap_mm=max_gap_mm,
    )
    if t_deep is None:
        return None

    # Quick brain sanity via lateral HU ring classification between the
    # deep tip and the bolt center. Rejects lines where the walk was
    # entirely in bone (dental metal, frame screws that happened to
    # line up with a bolt).
    ts = np.arange(t_deep, 0.0 + step_mm * 0.5, step_mm)
    classes = axr.classify_tissue_along_axis(
        fit, ts, arr_kji, ras_to_ijk_fn, cfg
    )
    brain_span_mm = axr.intracranial_brain_span_mm(classes, step_mm)
    min_span = float(getattr(cfg, "min_intracranial_span_mm", 15.0))
    if brain_span_mm < min_span:
        return None

    bolt_offset = float(getattr(cfg, "bolt_endpoint_offset_mm", 8.0))
    t_interface = -bolt_offset
    t_deep_intra = float(t_deep)
    if t_interface <= t_deep_intra:
        t_interface = min(0.0, t_deep_intra + 1.0)

    intracranial_span = float(t_interface - t_deep_intra)
    if intracranial_span < min_span:
        return None

    # Library span gate (soft trim).
    lib_min, lib_max = library_span_bounds
    span_tol = float(getattr(cfg, "library_span_tolerance_mm", 5.0))
    if lib_max > 0.0:
        if intracranial_span < lib_min - span_tol:
            return None
        if intracranial_span > lib_max + span_tol:
            t_deep_intra = float(t_interface - (lib_max + span_tol))
            intracranial_span = float(t_interface - t_deep_intra)

    best_model_id, _ = axr.library_span_match(
        intracranial_span, library_models, tolerance_mm=span_tol
    )

    return _emit_trajectory(
        bolt, fit, t_deep_intra, t_interface,
        intracranial_span, brain_span_mm, best_model_id,
        source="bolt_v2_walk",
    )


# ---------------------------------------------------------------------------
# Approach B: intensity peaks + library fit
# ---------------------------------------------------------------------------

def _build_intensity_profile(
    center_ras: np.ndarray,
    axis_ras: np.ndarray,
    arr_kji: np.ndarray,
    ras_to_ijk_fn,
    *,
    disc_radius_mm: float,
    inward_mm: float,
    outward_mm: float,
    step_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_values, profile) where profile[i] = max HU in the disc
    perpendicular to ``axis_ras`` at center + t_values[i] * axis_ras.
    """
    offsets = _build_filled_disc_offsets(axis_ras, disc_radius_mm)
    t_values = np.arange(-inward_mm, outward_mm + 0.5 * step_mm, step_mm)
    profile = np.full(t_values.shape[0], -np.inf, dtype=np.float64)
    for i, t in enumerate(t_values):
        pt = center_ras + float(t) * axis_ras
        profile[i] = _max_hu_in_disc(pt, offsets, arr_kji, ras_to_ijk_fn)
    return t_values, profile


def _find_profile_peaks(
    profile: np.ndarray,
    *,
    min_value: float,
    min_separation_samples: int,
) -> np.ndarray:
    """Return indices of local maxima in ``profile`` with value >=
    ``min_value`` separated by at least ``min_separation_samples``
    samples. Greedy peak picking: take the globally-largest sample,
    mask out its neighborhood, repeat.
    """
    mask = np.isfinite(profile) & (profile >= float(min_value))
    if not np.any(mask):
        return np.zeros(0, dtype=int)
    scratch = profile.copy()
    scratch[~mask] = -np.inf
    picked: list[int] = []
    sep = int(max(1, min_separation_samples))
    while True:
        idx = int(np.argmax(scratch))
        if not np.isfinite(scratch[idx]) or scratch[idx] < min_value:
            break
        picked.append(idx)
        lo = max(0, idx - sep)
        hi = min(scratch.size, idx + sep + 1)
        scratch[lo:hi] = -np.inf
    picked.sort()
    return np.asarray(picked, dtype=int)


def _library_contact_spacings(model: dict[str, Any]) -> np.ndarray | None:
    """Return the ascending contact offsets (mm) from the deepest
    contact for one library model, or ``None`` if the model has no
    usable geometry. Offsets are relative to the deepest contact
    (``offsets[0] == 0``).
    """
    # Preferred schema: per-model ``contact_center_offsets_from_tip_mm``.
    from_tip = model.get("contact_center_offsets_from_tip_mm")
    if from_tip is not None:
        arr = np.asarray(from_tip, dtype=float).reshape(-1)
        if arr.size >= 2:
            arr = np.sort(arr)
            return arr - float(arr[0])
    # Back-compat shapes.
    contacts = model.get("contact_offsets_mm")
    if contacts is not None:
        arr = np.asarray(contacts, dtype=float).reshape(-1)
        if arr.size >= 2:
            arr = np.sort(arr)
            return arr - float(arr[0])
    contact_count = model.get("contact_count") or model.get("n_contacts")
    spacing = (
        model.get("contact_spacing_mm")
        or model.get("inter_contact_distance_mm")
        or model.get("center_to_center_separation_mm")
    )
    if contact_count is not None and spacing is not None:
        n = int(contact_count)
        s = float(spacing)
        if n >= 2 and s > 0.0:
            return np.asarray([i * s for i in range(n)], dtype=float)
    return None


def _fit_library_to_peaks(
    peak_t: np.ndarray,
    library_models: list[dict[str, Any]],
    *,
    tolerance_mm: float,
    max_shallow_t: float = 2.0,
) -> tuple[dict[str, Any] | None, float, float, float] | None:
    """For each library model, try aligning its known contact offsets
    to the observed peak positions. Returns the best ``(model,
    t_deepest, t_shallowest, score)`` or ``None`` if no model covers
    any peaks.

    ``score`` is the absolute number of the model's contacts that
    found a peak within ``tolerance_mm``, minus a small penalty for
    RMS alignment error. Absolute (not fractional) so longer
    electrodes aren't penalized relative to short ones.
    """
    if peak_t.size == 0:
        return None
    peaks = np.sort(peak_t.astype(float))
    best = None
    for model in library_models or []:
        spacings = _library_contact_spacings(model)
        if spacings is None:
            continue
        # Try each observed peak as the candidate deepest-contact
        # anchor. For each anchor, compute the model's expected peak
        # positions (anchor + spacings) and count matches.
        for anchor_t in peaks:
            expected = anchor_t + spacings
            match_count = 0
            sq_err = 0.0
            matched_obs: set[int] = set()
            matched_expected_idx: list[int] = []
            for exp_i, exp in enumerate(expected):
                dists = np.abs(peaks - exp)
                for j in np.argsort(dists):
                    if int(j) in matched_obs:
                        continue
                    if dists[j] > tolerance_mm:
                        break
                    matched_obs.add(int(j))
                    sq_err += float(dists[j]) ** 2
                    match_count += 1
                    matched_expected_idx.append(exp_i)
                    break
            if match_count < 2:
                continue
            rms_err = float(np.sqrt(sq_err / match_count))
            coverage = match_count / float(len(spacings))
            # Absolute-count-first scoring with a small penalty for rms
            # error and a tiny bonus for high coverage (breaks ties in
            # favor of the model that uses all its contacts).
            score = float(match_count) - 0.3 * rms_err + 0.1 * coverage
            t_deepest = anchor_t
            t_shallowest = anchor_t + float(spacings[-1])
            # Shallow end must be inside the bolt region — the first
            # electrode contact sits at or below the skull surface,
            # which along the scalp-oriented axis is at t ~ 0 to
            # -(skull thickness). Any fit whose shallow end extends
            # outside the bolt (``t > max_shallow_t``) is a
            # long-library-matched-to-scattered-noise artifact.
            if t_shallowest > max_shallow_t:
                continue
            # Require the MATCHED peaks to span most of the model's
            # expected extent — otherwise a model whose contacts fall
            # in a narrow subrange of ``peaks`` will out-score a
            # tighter fit by brute match count. ``matched_expected_idx``
            # holds the model indices that found a peak; require
            # first_matched_contact ~= 0 and last_matched_contact ~=
            # n_contacts-1 within a small tolerance.
            if matched_expected_idx:
                first_matched = min(matched_expected_idx)
                last_matched = max(matched_expected_idx)
                if first_matched > 1:
                    continue
                if last_matched < len(spacings) - 2:
                    continue
            if best is None or score > best[3]:
                best = (model, t_deepest, t_shallowest, score,
                        match_count, rms_err, coverage)
    if best is None:
        return None
    model, t_deepest, t_shallowest, score, _, _, _ = best
    return model, t_deepest, t_shallowest, score


def _fit_deepest_peak(
    bolt,
    fit: "axr.AxisFit",
    arr_kji: np.ndarray,
    head_distance_map_kji: np.ndarray,
    ras_to_ijk_fn,
    library_models: list[dict[str, Any]],
    library_span_bounds: tuple[float, float],
    cfg,
) -> dict[str, Any] | None:
    """Approach C: build the intensity profile and use the deepest
    qualifying peak as the deep tip. Shallow end is anchored at
    ``-bolt_endpoint_offset_mm`` exactly like v1+bolts.

    This skips library matching entirely — the library is only used
    to validate the final intracranial span. The premise is that
    finding a single deepest contact is much easier than matching a
    whole contact pattern, and it's all we actually need to define
    the trajectory's two endpoints.
    """
    step_mm = float(getattr(cfg, "v2_profile_step_mm", 0.25))
    disc_radius = float(getattr(cfg, "v2_profile_disc_radius_mm", 2.5))
    peak_hu_floor = float(getattr(cfg, "v2_profile_peak_hu_floor", 800.0))
    peak_rel_frac = float(getattr(cfg, "v2_profile_peak_rel_frac", 0.35))
    max_gap_mm = float(getattr(cfg, "v2_contact_max_gap_mm", 8.0))
    bolt_offset = float(getattr(cfg, "bolt_endpoint_offset_mm", 8.0))

    t_values, profile = _build_intensity_profile(
        fit.center, fit.axis, arr_kji, ras_to_ijk_fn,
        disc_radius_mm=disc_radius,
        inward_mm=BOLT_INWARD_WALK_MM,
        outward_mm=BOLT_OUTWARD_WALK_MM,
        step_mm=step_mm,
    )

    finite = profile[np.isfinite(profile)]
    if finite.size == 0:
        return None
    profile_max = float(np.max(finite))
    min_value = max(peak_hu_floor, peak_rel_frac * profile_max)

    # Walk from the bolt (t ~ 0) inward (decreasing t). Track the
    # deepest sample where profile >= min_value under a gap tolerance:
    # if we haven't seen a qualifying sample in the last ``max_gap_mm``
    # steps, stop walking. This gives us the deep tip as the deepest
    # contact-like peak contiguous (within gap tolerance) with the
    # bolt through the electrode shaft.
    bolt_center_idx = int(np.argmin(np.abs(t_values)))
    max_gap_steps = max(1, int(round(max_gap_mm / step_mm)))
    deepest_idx = None
    gap_run = 0
    for i in range(bolt_center_idx, -1, -1):
        val = profile[i]
        if np.isfinite(val) and val >= min_value:
            deepest_idx = i
            gap_run = 0
        else:
            gap_run += 1
            if gap_run > max_gap_steps:
                break
    if deepest_idx is None:
        return None

    t_deep_intra = float(t_values[deepest_idx])
    t_interface = -bolt_offset
    if t_interface <= t_deep_intra:
        t_interface = min(0.0, t_deep_intra + 1.0)

    intracranial_span = float(t_interface - t_deep_intra)
    min_span = float(getattr(cfg, "min_intracranial_span_mm", 15.0))
    if intracranial_span < min_span:
        return None

    # Brain sanity check.
    ts = np.arange(t_deep_intra, t_interface + step_mm * 0.5, step_mm)
    classes = axr.classify_tissue_along_axis(
        fit, ts, arr_kji, ras_to_ijk_fn, cfg
    )
    brain_span_mm = axr.intracranial_brain_span_mm(classes, step_mm)
    if brain_span_mm < min_span:
        return None

    # Library span gate (soft trim).
    lib_min, lib_max = library_span_bounds
    span_tol = float(getattr(cfg, "library_span_tolerance_mm", 5.0))
    if lib_max > 0.0:
        if intracranial_span < lib_min - span_tol:
            return None
        if intracranial_span > lib_max + span_tol:
            t_deep_intra = float(t_interface - (lib_max + span_tol))
            intracranial_span = float(t_interface - t_deep_intra)

    best_model_id, _ = axr.library_span_match(
        intracranial_span, library_models, tolerance_mm=span_tol
    )

    return _emit_trajectory(
        bolt, fit, t_deep_intra, t_interface,
        intracranial_span, brain_span_mm, best_model_id,
        source="bolt_v2_deepest",
    )


def _fit_intensity_peaks(
    bolt,
    fit: "axr.AxisFit",
    arr_kji: np.ndarray,
    head_distance_map_kji: np.ndarray,
    ras_to_ijk_fn,
    library_models: list[dict[str, Any]],
    library_span_bounds: tuple[float, float],
    cfg,
) -> dict[str, Any] | None:
    step_mm = float(getattr(cfg, "v2_profile_step_mm", 0.25))
    disc_radius = float(getattr(cfg, "v2_profile_disc_radius_mm", 2.5))
    min_peak_sep_mm = float(getattr(cfg, "v2_profile_min_peak_sep_mm", 2.5))
    peak_hu_floor = float(getattr(cfg, "v2_profile_peak_hu_floor", 800.0))
    peak_rel_frac = float(getattr(cfg, "v2_profile_peak_rel_frac", 0.35))
    match_tol_mm = float(getattr(cfg, "v2_profile_peak_match_tol_mm", 1.5))

    t_values, profile = _build_intensity_profile(
        fit.center, fit.axis, arr_kji, ras_to_ijk_fn,
        disc_radius_mm=disc_radius,
        inward_mm=BOLT_INWARD_WALK_MM,
        outward_mm=BOLT_OUTWARD_WALK_MM,
        step_mm=step_mm,
    )

    # Adaptive min peak value: the larger of the absolute floor and a
    # fraction of the profile max. For dim electrodes the floor
    # dominates; for bright ones the relative fraction dominates.
    finite = profile[np.isfinite(profile)]
    if finite.size == 0:
        return None
    profile_max = float(np.max(finite))
    min_peak_value = max(peak_hu_floor, peak_rel_frac * profile_max)
    min_sep = int(round(min_peak_sep_mm / step_mm))
    peak_idx = _find_profile_peaks(profile, min_value=min_peak_value,
                                   min_separation_samples=min_sep)
    if peak_idx.size < 2:
        return None
    peak_t = t_values[peak_idx]

    # Only keep peaks on the inward side of the bolt (strictly negative
    # t — we're looking for contacts, not the bolt metal itself).
    # A small positive margin allows the shallow-most contact to sit
    # just barely inside the bolt span.
    peak_t = peak_t[peak_t <= 0.5]
    if peak_t.size < 2:
        return None

    fit_result = _fit_library_to_peaks(
        peak_t, library_models, tolerance_mm=match_tol_mm
    )
    if fit_result is None:
        return None
    model, t_deepest, t_shallowest, score = fit_result

    # Contact-row span == distance between deepest and shallowest
    # matched contact centers. For Approach B we do NOT apply the v1
    # ``min_intracranial_span_mm`` gate — that was calibrated to the
    # deep_core brain-entry-to-deepest-contact span, which is longer
    # than the contact row itself. The library span gate still rejects
    # nonsense fits.
    intracranial_span = float(t_shallowest - t_deepest)

    # Brain sanity check. We sample from the deepest matched peak up
    # to the bolt, not just the contact row, so the brain-span check
    # catches trajectories that never enter the brain proper.
    sanity_t_lo = min(t_deepest, -20.0)
    sanity_t_hi = max(0.0, t_shallowest)
    ts = np.arange(sanity_t_lo, sanity_t_hi + step_mm * 0.5, step_mm)
    classes = axr.classify_tissue_along_axis(
        fit, ts, arr_kji, ras_to_ijk_fn, cfg
    )
    brain_span_mm = axr.intracranial_brain_span_mm(classes, step_mm)
    min_brain = 8.0
    if brain_span_mm < min_brain:
        return None

    return _emit_trajectory(
        bolt, fit, t_deepest, t_shallowest,
        intracranial_span, brain_span_mm,
        str(model.get("id", "")),
        source="bolt_v2_peaks",
    )


# ---------------------------------------------------------------------------
# Shared trajectory emission
# ---------------------------------------------------------------------------

def _emit_trajectory(
    bolt,
    fit,
    t_deep_intra: float,
    t_interface: float,
    intracranial_span: float,
    brain_span_mm: float,
    best_model_id: str,
    *,
    source: str,
) -> dict[str, Any]:
    t_bolt_outer = float(bolt.span_mm) * 0.5
    start_ras = fit.center + fit.axis * float(t_deep_intra)
    end_ras = fit.center + fit.axis * float(t_interface)
    bolt_ras = fit.center + fit.axis * t_bolt_outer
    return {
        "source": source,
        "start_ras": [float(v) for v in start_ras],
        "end_ras": [float(v) for v in end_ras],
        "bolt_ras": [float(v) for v in bolt_ras],
        "axis_ras": [float(v) for v in fit.axis],
        "bolt_center_ras": [float(v) for v in bolt.center_ras],
        "span_mm": float(intracranial_span),
        "intracranial_span_mm": float(intracranial_span),
        "bolt_extent_mm": float(max(0.0, t_bolt_outer - t_interface)),
        "brain_span_mm": float(brain_span_mm),
        "axis_residual_rms_mm": 0.0,
        "axis_residual_median_mm": 0.0,
        "best_model_id": str(best_model_id or ""),
        "model_fit_passed": True,
        "bolt_seed": True,
        "bolt_span_mm": float(bolt.span_mm),
        "bolt_hd_center_mm": float(bolt.hd_center_mm),
        "bolt_fill_frac": float(bolt.fill_frac),
        "atom_id_list": [],
        "explained_atom_ids": [],
    }


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def fit_bolt_trajectory(
    bolt,
    *,
    arr_kji: np.ndarray,
    head_distance_map_kji: np.ndarray,
    ras_to_ijk_fn,
    library_models: list[dict[str, Any]],
    library_span_bounds: tuple[float, float],
    cfg,
) -> dict[str, Any] | None:
    """Turn one bolt candidate into a trajectory dict (or ``None``).

    Dispatches on ``cfg.v2_fit_mode`` (default ``"two_threshold"``).
    """
    axis = np.asarray(bolt.axis_ras, dtype=float).reshape(3)
    axis_n = float(np.linalg.norm(axis))
    if axis_n < 1e-9:
        return None
    axis = axis / axis_n
    center = np.asarray(bolt.center_ras, dtype=float).reshape(3)
    axis = _orient_bolt_axis_toward_scalp(
        axis, center, head_distance_map_kji, ras_to_ijk_fn
    )

    fit = axr.AxisFit(
        center=center,
        axis=axis,
        residual_rms_mm=0.0,
        residual_median_mm=0.0,
        elongation=1.0,
        t_min=-float(bolt.span_mm) * 0.5,
        t_max=+float(bolt.span_mm) * 0.5,
        point_count=int(bolt.n_inliers),
    )

    mode = str(getattr(cfg, "v2_fit_mode", "deepest_peak")).strip().lower()
    if mode == "intensity_peaks":
        return _fit_intensity_peaks(
            bolt, fit, arr_kji, head_distance_map_kji, ras_to_ijk_fn,
            library_models, library_span_bounds, cfg,
        )
    if mode == "two_threshold":
        return _fit_two_threshold(
            bolt, fit, arr_kji, head_distance_map_kji, ras_to_ijk_fn,
            library_models, library_span_bounds, cfg,
        )
    # Default: Approach C — deepest significant peak + bolt-anchored
    # shallow end, no library matching.
    return _fit_deepest_peak(
        bolt, fit, arr_kji, head_distance_map_kji, ras_to_ijk_fn,
        library_models, library_span_bounds, cfg,
    )


def run_bolt_fit_group(
    bolts,
    *,
    arr_kji: np.ndarray,
    head_distance_map_kji: np.ndarray,
    ras_to_ijk_fn,
    library_models: list[dict[str, Any]],
    cfg,
) -> dict[str, Any]:
    """Fit every bolt candidate independently. Returns the same shape as
    ``run_model_fit_group`` so the v2 pipeline can swap it in without
    touching trajectory conversion downstream.
    """
    lib_bounds = axr.library_span_range(library_models)
    accepted: list[dict[str, Any]] = []
    rejected = 0
    for bolt in (bolts or []):
        traj = fit_bolt_trajectory(
            bolt,
            arr_kji=arr_kji,
            head_distance_map_kji=head_distance_map_kji,
            ras_to_ijk_fn=ras_to_ijk_fn,
            library_models=library_models,
            library_span_bounds=lib_bounds,
            cfg=cfg,
        )
        if traj is None:
            rejected += 1
            continue
        accepted.append(traj)
    return {
        "accepted_proposals": accepted,
        "stats": {
            "input_count": int(len(bolts or [])),
            "accepted_count": int(len(accepted)),
            "rejected": int(rejected),
        },
    }
