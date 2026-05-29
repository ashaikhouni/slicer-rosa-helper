"""Electrode-model classification helpers + bolt-boundary helper.

The canonical per-trajectory electrode-model picker is now the
matched-filter in ``rosa_core.contact_placement.stage_d_pick`` (Pearson
NCC of the centerline disk-stat signal against the library comb
template). The PaCER template-correlation picker (``classify_pacer_template``)
and the ``classify_electrode_model`` cascade that dispatched between it,
walker-signature, and length-only fallbacks were removed 2026-05-11 —
they wrote ``Rosa.BestModelId`` to MRML attributes that no module reads
(``ContactsTrajectoryView`` defaults its dropdowns to empty per
its ``_populate_contact_table`` comment), and CLI placement never
consumed them.

What remains in this module:

* ``filter_models_for_strategy`` — vendor + pitch-set library filter.
  Used by ``stage_d_pick`` and notebook callers.
* ``suggest_shortest_covering_model`` — length-only library lookup;
  kept for the unit-test suite under
  ``tests/deep_core/test_walker_signature_classifier.py`` and for
  callers that need a quick best-fit by length alone.
* ``classify_by_walker_signature`` — joint pitch + count + span +
  length scoring; same scope (tests + length-only callers).
* ``classify_by_count_and_span`` — older fallback variant of the
  walker-signature score.
* ``signal_derived_entry_arc`` — bolt -> electrode transition detector
  used as Tier 1 of the bolt-mass walker's bolt-end cascade
  (``contact_placement_legacy.estimate_bolt_end_from_metal_mass``).
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------
# Strategy library: vendor + pitch-set filters
# ---------------------------------------------------------------------

PITCH_STRATEGY_PITCHES_MM = {
    "dixi":      (3.5,),
    "pmt_35":    (3.5,),
    "pmt":       (3.5, 3.97, 4.43),
    "mixed":     (3.5, 3.97, 4.43),
    "dixi_mm":   (3.9, 4.8, 6.1),
    "dixi_all":  (3.5, 3.9, 4.43, 4.8, 6.1),
    "medtronic": (2.0, 3.0, 7.0),
    "neuropace": (3.5, 10.0),
}

PITCH_STRATEGY_VENDORS = {
    "dixi":      ("Dixi",),
    "pmt_35":    ("PMT",),
    "pmt":       ("PMT",),
    "mixed":     ("Dixi", "PMT"),
    "dixi_mm":   ("Dixi",),
    "dixi_all":  ("Dixi",),
    "medtronic": ("Medtronic",),
    "neuropace": ("NeuroPace",),
    "auto":      ("Dixi", "PMT", "AdTech", "NeuroPace"),
}

VENDOR_ID_PREFIXES = {
    "Dixi":      "DIXI-",
    "PMT":       "PMT-",
    "Medtronic": "Medtronic_",
    "NeuroPace": "NeuroPace_",
}


# (label, key) options shared across UI combos that let the user
# restrict the picker library — Auto Fit / Guided Fit (deep_core_widget),
# Manual Fit, Contacts & Trajectory View. Keep in sync with the strategy
# keys in `PITCH_STRATEGY_PITCHES_MM` / `PITCH_STRATEGY_VENDORS`.
PITCH_STRATEGY_OPTIONS = (
    ("Dixi AM (3.5 mm)",                       "dixi"),
    ("Dixi MM hybrid (3.9 / 4.8 / 6.1 mm)",    "dixi_mm"),
    ("Dixi all (AM + MM hybrid)",              "dixi_all"),
    ("PMT 2102-XX-091 (3.5 mm)",               "pmt_35"),
    ("PMT (3.5 / 3.97 / 4.43 mm)",             "pmt"),
    ("Mixed Dixi + PMT",                       "mixed"),
    ("Medtronic DBS (2 / 3 / 7 mm)",           "medtronic"),
    ("NeuroPace RNS depth (3.5 / 10 mm)",      "neuropace"),
    ("All vendors (no restriction)",           "auto"),
)


def _vendor_prefixes(vendors):
    return tuple(
        VENDOR_ID_PREFIXES[v] for v in (vendors or ()) if v in VENDOR_ID_PREFIXES
    )


def _model_pitch_median_mm(model):
    """Median inter-contact spacing for one electrode model. Equals the
    pitch for uniform models; tracks the dominant pitch on the DIXI-MM
    family which has 3 distinct inter-contact spacings."""
    offsets = model.get("contact_center_offsets_from_tip_mm") or []
    if len(offsets) < 2:
        return 0.0
    diffs = sorted(
        float(offsets[i + 1]) - float(offsets[i])
        for i in range(len(offsets) - 1)
    )
    return float(diffs[len(diffs) // 2])


def filter_models_for_strategy(models, strategy_key,
                               pitch_tolerance_mm=0.25):
    """Restrict the model library to those matching a pitch-strategy
    selection — vendor prefix AND median pitch within `pitch_tolerance_mm`
    of one of the strategy's pitches.

    `strategy_key == "auto"` (or unknown) returns the library unchanged.
    """
    if not strategy_key:
        return list(models)
    key = str(strategy_key).strip().lower()
    if key == "auto":
        return list(models)
    pitches = PITCH_STRATEGY_PITCHES_MM.get(key)
    vendors = PITCH_STRATEGY_VENDORS.get(key)
    if not pitches or not vendors:
        return list(models)
    prefixes = _vendor_prefixes(vendors)
    if not prefixes:
        return list(models)
    out = []
    for m in models:
        mid = str(m.get("id") or "")
        if not mid.startswith(prefixes):
            continue
        m_pitch = _model_pitch_median_mm(m)
        if m_pitch <= 0.0:
            continue
        if not any(abs(m_pitch - float(p)) <= float(pitch_tolerance_mm)
                   for p in pitches):
            continue
        out.append(m)
    return out


# ---------------------------------------------------------------------
# Length-only scoring (dura-tolerant covering)
# ---------------------------------------------------------------------

def suggest_shortest_covering_model(intracranial_length_mm, models,
                                    vendors=("Dixi",),
                                    dura_tolerance_mm=10.0):
    """Pick the shortest model whose total exploration length + dura
    tolerance covers `intracranial_length_mm`.

    `intracranial_length_mm` = `|skull_entry − deep_tip|`. Because
    skull_entry sits inside the skull/dura band rather than at contact 1,
    observed length overstates active electrode length by ~5-10 mm of
    soft-tissue margin. `dura_tolerance_mm` absorbs that offset.

    Returns `{"model_id", "model_length_mm", "gap_mm"}` or `None`.
    """
    prefixes = _vendor_prefixes(vendors)
    if not prefixes:
        return None
    L = float(intracranial_length_mm)
    tol = float(dura_tolerance_mm)
    best = None
    for model in models:
        mid = str(model.get("id") or "")
        if not mid.startswith(prefixes):
            continue
        total = model.get("total_exploration_length_mm")
        if total is None:
            offsets = model.get("contact_center_offsets_from_tip_mm") or []
            if len(offsets) < 2:
                continue
            total = float(offsets[-1]) - float(offsets[0])
        total = float(total)
        if total + tol < L:
            continue
        if best is None or total < best["model_length_mm"]:
            best = {
                "model_id": mid,
                "model_length_mm": total,
                "gap_mm": total - L,
            }
    return best


# ---------------------------------------------------------------------
# Walker-signature joint scoring (legacy)
# ---------------------------------------------------------------------

def classify_by_walker_signature(n_observed, pitch_observed_mm,
                                 contact_span_observed_mm,
                                 intracranial_length_mm, models,
                                 vendors=("Dixi",),
                                 pitch_weight_mm=10.0,
                                 count_weight_mm=3.5,
                                 span_weight=1.0,
                                 length_weight=0.5,
                                 dura_tolerance_mm=10.0,
                                 span_shoulder_mm=2.0):
    """Pick the model best explained by walker stats `(n, pitch, span,
    length)`. Score (mm-equivalent units, lower better):

    ```
    pitch_err * pitch_weight_mm
      + count_err * count_weight_mm
      + max(0, span_err - span_shoulder_mm) * span_weight
      + max(0, length_err - dura_tolerance_mm) * length_weight
    ```

    Pitch dominates (within-vendor discriminator). Wire-class fallback:
    when `pitch_observed_mm <= 0` or `n_observed <= 0`, skips those terms.
    """
    prefixes = _vendor_prefixes(vendors)
    if not prefixes:
        return None
    n_obs = int(n_observed) if n_observed and n_observed > 0 else 0
    pitch_obs = float(pitch_observed_mm) if pitch_observed_mm and pitch_observed_mm > 0 else 0.0
    span_obs = float(contact_span_observed_mm or 0.0)
    length_obs = float(intracranial_length_mm or 0.0)
    best = None
    for model in models:
        mid = str(model.get("id") or "")
        if not mid.startswith(prefixes):
            continue
        offsets = model.get("contact_center_offsets_from_tip_mm") or []
        if len(offsets) < 2:
            continue
        n_model = int(model.get("contact_count") or len(offsets))
        span_model = float(offsets[-1]) - float(offsets[0])
        pitch_model = _model_pitch_median_mm(model)
        total_model = model.get("total_exploration_length_mm")
        if total_model is None:
            total_model = span_model
        total_model = float(total_model)

        pitch_term = 0.0
        if pitch_obs > 0.0 and pitch_model > 0.0:
            pitch_term = float(pitch_weight_mm) * abs(pitch_obs - pitch_model)
        count_term = 0.0
        if n_obs > 0:
            count_term = float(count_weight_mm) * abs(n_obs - n_model)
        span_term = 0.0
        span_err = 0.0
        if span_obs > 0.0:
            span_err = abs(span_obs - span_model)
            span_term = float(span_weight) * max(0.0, span_err - float(span_shoulder_mm))
        length_err = abs(length_obs - total_model) if length_obs > 0.0 else 0.0
        length_term = (
            float(length_weight) * max(0.0, length_err - float(dura_tolerance_mm))
            if length_obs > 0.0 else 0.0
        )
        score = pitch_term + count_term + span_term + length_term
        if best is None or score < best["score"]:
            best = {
                "model_id": mid,
                "score": float(score),
                "model_pitch_mm": float(pitch_model),
                "model_n": int(n_model),
                "model_span_mm": float(span_model),
                "model_total_mm": float(total_model),
                "pitch_err_mm": float(abs(pitch_obs - pitch_model)) if pitch_obs > 0 else float("nan"),
                "count_err": int(abs(n_obs - n_model)) if n_obs > 0 else -1,
                "span_err_mm": float(span_err),
                "length_err_mm": float(length_err),
            }
    return best


def classify_by_count_and_span(n_observed, span_observed_mm,
                               models, vendors=("Dixi",),
                               count_weight_mm=3.5):
    """Older fallback — `|N_model − n_obs| * count_weight_mm + |span_err|`."""
    prefixes = _vendor_prefixes(vendors)
    if not prefixes:
        return None
    best = None
    for model in models:
        mid = str(model.get("id") or "")
        if not mid.startswith(prefixes):
            continue
        offsets = model.get("contact_center_offsets_from_tip_mm") or []
        if len(offsets) < 2:
            continue
        n_model = int(model.get("contact_count") or len(offsets))
        span_model = float(offsets[-1]) - float(offsets[0])
        count_err = abs(int(n_observed) - n_model)
        span_err = abs(float(span_observed_mm) - span_model)
        score = count_err * float(count_weight_mm) + span_err
        if best is None or score < best["score"]:
            best = {
                "model_id": mid,
                "score": float(score),
                "count_err": int(count_err),
                "span_err": float(span_err),
                "n_model": n_model,
                "span_model_mm": span_model,
            }
    return best


# ---------------------------------------------------------------------
# Bolt-boundary helper for the contact-placement bolt-mass walker
# ---------------------------------------------------------------------
#
# ``signal_derived_entry_arc`` detects the bolt -> electrode transition
# from a 1D HU profile sampled along a trajectory. The bolt-mass walker
# (``contact_placement_legacy.estimate_bolt_end_from_metal_mass``) calls
# this as Tier 1 of its bolt-end cascade.
#
# Note: the constants are PaCER-namespaced because they were factored
# out of the older PaCER template-correlation picker (removed 2026-05-11
# — see the module-header comment).

_PACER_BOLT_THRESHOLD_HU = 2000.0  # tighter — metal only, excludes skull bone
# Bolt = sustained AND very-bright metal run. Two-criterion test:
#   - run length >= _PACER_BOLT_MIN_RUN_MM at HU >= _PACER_BOLT_THRESHOLD_HU
#   - peak HU within the run >= _PACER_BOLT_PEAK_HU
# SEEG bolts image with peak HU 2400-3000+ (solid titanium screw).
# Electrode contacts peak at 1800-2200 (smaller, partial-volume diluted).
# A short bolt run (1.5-2 mm) is common from edge averaging; the peak-HU
# check prevents a contact run that briefly crosses 2000 HU from
# being mistaken for a bolt.
_PACER_BOLT_MIN_RUN_MM = 1.5
_PACER_BOLT_PEAK_HU = 2400.0
_PACER_BOLT_GAP_MIN_MM = 1.0       # min dim region between bolt and electrode


def signal_derived_entry_arc(profile_arc_mm, profile_values):
    """Detect the bolt -> electrode boundary by the bolt's sustained-
    bright signature, and return the arc-position where the actual
    electrode region starts.

    Key distinction (per user observation): a single SEEG contact is
    a small bright peak ~2-3 mm wide; a bolt is a large sustained
    bright stretch typically 5-15 mm long. So:

      1. Find runs of metal-bright (HU >= ``_PACER_BOLT_THRESHOLD_HU``).
      2. The first run whose length >= ``_PACER_BOLT_MIN_RUN_MM`` is
         the BOLT (a contact would be too short).
      3. After the bolt, find the next metal-bright run preceded by
         a dim gap >= ``_PACER_BOLT_GAP_MIN_MM``. Its start is the
         entry-side of the actual electrode.

    Returns the entry-arc, or None when no bolt signature is found
    (e.g. axis doesn't include a bolt, contacts touch the bolt with
    no clean gap).
    """
    arr = np.asarray(profile_values, dtype=float)
    metal = (arr >= _PACER_BOLT_THRESHOLD_HU) & np.isfinite(arr)
    if int(metal.sum()) < 2:
        return None
    n = len(arr)
    step_mm = (
        float(profile_arc_mm[1] - profile_arc_mm[0]) if n >= 2 else 0.25
    )
    bolt_min_samples = max(3, int(_PACER_BOLT_MIN_RUN_MM / step_mm))
    gap_min_samples = max(2, int(_PACER_BOLT_GAP_MIN_MM / step_mm))
    bool_arr = metal.astype(np.int8)
    edges = np.diff(np.concatenate([[0], bool_arr, [0]]))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    if len(starts) < 1:
        return None
    # Identify the bolt run by length AND peak HU. A bolt is both
    # sustained (>= _PACER_BOLT_MIN_RUN_MM) and very-bright (peak HU
    # within run >= _PACER_BOLT_PEAK_HU). Contacts above 2000 HU
    # rarely peak above 2400.
    bolt_idx = None
    for i in range(len(starts)):
        if (ends[i] - starts[i]) < bolt_min_samples:
            continue
        run_peak = float(np.nanmax(arr[starts[i]:ends[i]]))
        if run_peak < _PACER_BOLT_PEAK_HU:
            continue
        bolt_idx = i
        break
    if bolt_idx is None:
        return None
    # If a second metal-bright run follows the bolt after a dim gap,
    # return that run's start (the entry of the actual electrode). This
    # is the cleanest signal — the electrode-region bright peaks rise
    # above the bolt threshold and the gap is unambiguous.
    for i in range(bolt_idx, len(starts) - 1):
        gap_samples = starts[i + 1] - ends[i]
        if gap_samples >= gap_min_samples:
            return float(profile_arc_mm[starts[i + 1]])
    # No second bolt-threshold run exists — common when electrode
    # contacts peak below the bolt threshold (e.g. below 2000 HU). The
    # entry of the electrode is then just past the bolt's end. Return
    # ``arc[bolt_end] + gap_min_mm`` so the electrode-side contacts
    # are not muted.
    bolt_end_idx = ends[bolt_idx] - 1  # last sample within bolt run
    if bolt_end_idx >= len(profile_arc_mm) - 1:
        return None
    return float(profile_arc_mm[bolt_end_idx]) + float(_PACER_BOLT_GAP_MIN_MM)


# Backwards-compat alias for older imports.
_signal_derived_entry_arc = signal_derived_entry_arc


# ---------------------------------------------------------------------
# Library-driven pitch-match decision-tree picker (2026-05-18).
#
# Ported from notebooks/seeded_fit/_build_fit_library.py. Strictly beats
# classify_by_walker_signature on the S57+S54 fixture (29/31 strict /
# 31/31 relaxed vs 17/31 strict for classify_by_walker_signature with
# snap features). Uses HARD pitch-family gates (±0.4 mm) to prevent
# AM ↔ MM cross-family confusion, plus a four-branch decision tree:
#
#   1. cluster        → force_len OR ≥2 cluster votes
#   2. pitch_match    → eroded peaks are cleanly separated (low pitch_cv)
#   3. regular_AM     → snap n_peaks ≥ 5 AND pitch_cv < 0.20
#   4. continuous_AM  → fallback by total_length + contact-count floor
#
# Library-driven: classifies depth-vs-cluster via `model_uniform_pitch`
# (any pitch ≥ 7mm or > 2× median → cluster). Any electrode family that
# follows the standard JSON schema slots into the right pitch bucket
# automatically; no vendor/suffix string matching.
# ---------------------------------------------------------------------


# Picker thresholds (per cell-18 of _build_fit_library.py).

CLEAR_PEAK_SIGNAL       = "metal_width_eroded_mm"
CLEAR_PEAK_HEIGHT_FRAC  = 0.20
CLEAR_PEAK_MIN_DIST_MM  = 2.5

LARGE_PITCH_THRESHOLD   = 7.0
CM_BM_MARGIN            = 0.10
STRONG_ENVELOPE_MARGIN  = 0.20
LENGTH_GUARD_MM         = 5.0
IRREGULAR_CV            = 0.30
MIN_CLUSTER_VOTES       = 2

PITCH_CV_THRESHOLD      = 0.20
MIN_PEAKS_FOR_REGULAR   = 5
PEAK_SPAN_TOLERANCE_MM  = 2.0

CLEAN_PITCH_CV_MAX      = 0.25
MIN_CLEAR_FOR_CLEAN     = 6
PITCH_MATCH_TOL_MM      = 0.4


def model_uniform_pitch(model) -> float | None:
    """Median small-pitch for a library model, returning ``None`` for
    cluster models.

    Cluster detection: any inter-contact pitch ≥ 7 mm in absolute terms,
    OR > 2× the median small-pitch, marks the model as a cluster
    (DIXI-15CM / 15BM / 18CM-style). This is library-driven and works
    for any vendor whose JSON entry follows the standard
    ``contact_center_offsets_from_tip_mm`` schema.

    Returns the median small-pitch (≈ 3.5 mm for AM, 3.9/4.8/6.1 mm for
    DIXI-MM variants) when the model is a uniform-pitch depth electrode;
    ``None`` for clusters. Compare against
    :func:`_model_pitch_median_mm` which returns a numeric value even
    for cluster models (its median collapses to the small-pitch since
    cluster models have more 3.5 mm gaps than 13 mm gaps).
    """
    offsets = model.get("contact_center_offsets_from_tip_mm") or []
    if len(offsets) < 2:
        return None
    pitches = [float(offsets[i + 1]) - float(offsets[i])
               for i in range(len(offsets) - 1)]
    small = [p for p in pitches if p < 7.0]
    if not small:
        return None
    s = sorted(small)
    median_p = s[len(s) // 2]
    if any(p >= 7.0 or p > 2.0 * median_p for p in pitches):
        return None
    return float(median_p)


def trajectory_peak_features(
    chain: dict, prof: dict, arc_rel,
    win_lo: float, win_hi: float,
    *,
    signal_name: str = CLEAR_PEAK_SIGNAL,
    profile_step_mm: float = 0.3,
    clear_peak_height_frac: float = CLEAR_PEAK_HEIGHT_FRAC,
    clear_peak_min_dist_mm: float = CLEAR_PEAK_MIN_DIST_MM,
    large_pitch_threshold_mm: float = LARGE_PITCH_THRESHOLD,
) -> dict:
    """Find clear peaks on the eroded centerline (or another peaky
    signal) RESTRICTED to ``[win_lo, win_hi]`` = the intracranial
    window (from the bone-inner landmark to the fitted tip).

    Returns a dict the picker scores against. Snap-side fields
    (``n_peaks``, ``snap_span_mm``) come from the supplied ``chain``
    (its ``kept_pts``, ``axis``, ``entry_ras``); eroded-side fields
    (``n_clear``, ``pitch_cv``, ``peak_span_mm``, ``n_large_pitches``,
    ``max_pitch_mm``, ``pitch_mean_mm``) come from
    ``scipy.signal.find_peaks`` on the windowed signal.

    Without the window restriction, eroded metal_width also peaks
    inside the bolt zone (bolt body survives 1-iter erosion) → peak_span
    balloons, n_clear over-counts.
    """
    import numpy as np
    from scipy.signal import find_peaks
    from .centerline import build_detrended_ratio, unit

    kept = np.asarray(chain.get("kept_pts", []), dtype=float)
    n_snap = int(len(kept))
    snap_span = 0.0
    if n_snap >= 2:
        axis_u = unit(np.asarray(chain["axis"], dtype=float))
        entry_arr = np.asarray(chain["entry_ras"], dtype=float)
        snap_arcs = (kept - entry_arr) @ axis_u
        snap_span = float(snap_arcs.max() - snap_arcs.min())

    if signal_name == "detrended_ratio":
        full_sig = build_detrended_ratio(prof, step_mm=profile_step_mm)
    else:
        full_sig = np.asarray(prof[signal_name], dtype=float)
    full_sig = np.where(np.isfinite(full_sig), full_sig, 0.0)
    arc = np.asarray(arc_rel, dtype=float)

    empty = dict(
        n_peaks=n_snap, n_clear=0,
        pitch_mean_mm=float("nan"), pitch_cv=float("inf"),
        peak_span_mm=0.0, snap_span_mm=snap_span,
        n_large_pitches=0, max_pitch_mm=0.0,
        clear_signal=signal_name,
    )
    in_win = (arc >= win_lo) & (arc <= win_hi)
    if int(in_win.sum()) < 5:
        return empty
    sig = full_sig[in_win]
    arc_w = arc[in_win]
    sig_max = float(sig.max()) if sig.size else 0.0
    if sig_max <= 0:
        return empty
    min_h = clear_peak_height_frac * sig_max
    min_d = max(1, int(clear_peak_min_dist_mm / profile_step_mm))
    peak_idx, _ = find_peaks(sig, height=min_h, distance=min_d)
    if len(peak_idx) < 2:
        empty["n_clear"] = int(len(peak_idx))
        return empty
    peak_arcs = np.sort(arc_w[peak_idx])
    diffs = np.diff(peak_arcs)
    pitch_mean = float(diffs.mean())
    pitch_cv = float(diffs.std() / pitch_mean) if pitch_mean > 0 else float("inf")
    return dict(
        n_peaks=n_snap,
        n_clear=int(len(peak_arcs)),
        pitch_mean_mm=pitch_mean,
        pitch_cv=pitch_cv,
        peak_span_mm=float(peak_arcs[-1] - peak_arcs[0]),
        snap_span_mm=snap_span,
        n_large_pitches=int((diffs > large_pitch_threshold_mm).sum()),
        max_pitch_mm=float(diffs.max()),
        clear_signal=signal_name,
    )


def _pick_by_pitch_match(n_peaks, snap_span_mm, median_pitch_mm,
                          depth_ids, models_dict,
                          pitch_tol_mm: float = PITCH_MATCH_TOL_MM,
                          span_tol_mm: float = PEAK_SPAN_TOLERANCE_MM):
    """Pick the uniform-pitch model whose pitch matches the observed
    median pitch (within ``pitch_tol_mm``) AND whose contact_count ≥
    n_peaks AND model_span ≥ snap_span - span_tol_mm. Sort by
    (pitch_diff, model_span, contact_count)."""
    qualifying = []
    for m_id in depth_ids:
        m = models_dict[m_id]
        m_pitch = model_uniform_pitch(m)
        if m_pitch is None:
            continue
        if abs(m_pitch - median_pitch_mm) > pitch_tol_mm:
            continue
        if int(m.get("contact_count") or 0) < n_peaks:
            continue
        offsets = m["contact_center_offsets_from_tip_mm"]
        model_span = float(offsets[-1] - offsets[0])
        if model_span < snap_span_mm - span_tol_mm:
            continue
        qualifying.append((m_id, abs(m_pitch - median_pitch_mm), model_span,
                           int(m["contact_count"])))
    if not qualifying:
        return None
    qualifying.sort(key=lambda x: (x[1], x[2], x[3]))
    return qualifying[0][0]


def _pick_minimal_extension_am(n_peaks, peak_span_mm, am_ids, models_dict,
                                span_tolerance_mm: float = PEAK_SPAN_TOLERANCE_MM,
                                target_pitch_mm: float | None = None,
                                pitch_tol_mm: float = PITCH_MATCH_TOL_MM):
    """Smallest depth model whose contact_count ≥ n_peaks AND
    model_span ≥ peak_span - span_tolerance_mm. 'Minimal extension' =
    smallest covering model (least overhang past most-superficial peak).

    When ``target_pitch_mm`` is supplied, candidates are filtered to
    those whose model pitch lies within ``pitch_tol_mm`` of it. This
    keeps the fallback paths inside the right pitch family even when
    the eroded signal was too noisy for the pitch_match path to fire."""
    qualifying = []
    for m_id in am_ids:
        m = models_dict[m_id]
        if int(m.get("contact_count") or 0) < n_peaks:
            continue
        if target_pitch_mm is not None:
            m_pitch = model_uniform_pitch(m)
            if m_pitch is None or abs(m_pitch - target_pitch_mm) > pitch_tol_mm:
                continue
        offsets = m["contact_center_offsets_from_tip_mm"]
        model_span = float(offsets[-1] - offsets[0])
        if model_span < peak_span_mm - span_tolerance_mm:
            continue
        qualifying.append((m_id, model_span, int(m["contact_count"])))
    if not qualifying:
        return None
    qualifying.sort(key=lambda x: (x[1], x[2]))
    return qualifying[0][0]


def _pick_minimal_extension_cluster(peak_span_mm, cluster_ids, models_dict,
                                     *, snap_span_mm: float | None = None,
                                     span_tolerance_mm: float = PEAK_SPAN_TOLERANCE_MM):
    """Smallest cluster model whose model_span ≥ observed span. Uses
    the larger of peak_span and snap_span (eroded under-counts at
    cluster boundaries; snap overshoots by 1-2 mm past the deepest
    contact). ``span_tolerance_mm`` slack covers that overshoot so we
    don't reject the right-size cluster — e.g. RCMN on S57 has
    snap_span 68.3 mm vs DIXI-15CM library span 68.0, and without the
    tolerance 18CM (78.5 mm) would be picked instead. Falls back to
    the LARGEST cluster when none covers the span."""
    observed = peak_span_mm if snap_span_mm is None else max(peak_span_mm, snap_span_mm)
    qualifying = []
    for m_id in cluster_ids:
        offsets = models_dict[m_id]["contact_center_offsets_from_tip_mm"]
        model_span = float(offsets[-1] - offsets[0])
        qualifying.append((m_id, model_span, abs(model_span - observed)))
    if not qualifying:
        return None
    covering = [q for q in qualifying if q[1] >= observed - span_tolerance_mm]
    if covering:
        covering.sort(key=lambda x: x[1])
        return covering[0][0]
    qualifying.sort(key=lambda x: -x[1])
    return qualifying[0][0]


def _pick_minimal_length_am(arc_length_mm, am_ids, models_dict,
                             *, n_peaks: int = 0,
                             target_pitch_mm: float | None = None,
                             pitch_tol_mm: float = PITCH_MATCH_TOL_MM):
    """Smallest depth model whose total_exploration_length covers
    arc_length (within 2 mm) AND contact_count ≥ n_peaks.

    Optional pitch filter mirrors :func:`_pick_minimal_extension_am` —
    when a target pitch is supplied, candidates outside the tolerance
    are excluded so the fallback path stays in the right pitch family."""
    qualifying = []
    for m_id in am_ids:
        m = models_dict[m_id]
        total_len = float(m.get("total_exploration_length_mm") or 0.0)
        if total_len < arc_length_mm - 2.0:
            continue
        if int(m.get("contact_count") or 0) < n_peaks:
            continue
        if target_pitch_mm is not None:
            m_pitch = model_uniform_pitch(m)
            if m_pitch is None or abs(m_pitch - target_pitch_mm) > pitch_tol_mm:
                continue
        qualifying.append((m_id, total_len))
    if not qualifying:
        return None
    qualifying.sort(key=lambda x: x[1])
    return qualifying[0][0]


def classify_by_pitch_decision_tree(
    features: dict,
    *,
    models: Sequence[dict],
    arc_length_mm: float,
    cluster_envelope_margin: float = 0.0,
    candidate_model_ids: Sequence[str] | None = None,
    pitch_match_tol_mm: float = PITCH_MATCH_TOL_MM,
    clean_pitch_cv_max: float = CLEAN_PITCH_CV_MAX,
    min_clear_for_clean: int = MIN_CLEAR_FOR_CLEAN,
    min_peaks_for_regular: int = MIN_PEAKS_FOR_REGULAR,
    pitch_cv_threshold: float = PITCH_CV_THRESHOLD,
    cm_bm_margin: float = CM_BM_MARGIN,
    irregular_cv: float = IRREGULAR_CV,
    length_guard_mm: float = LENGTH_GUARD_MM,
    min_cluster_votes: int = MIN_CLUSTER_VOTES,
) -> dict:
    """Library-driven decision-tree electrode-model picker.

    Inputs:

    * ``features``: dict from :func:`trajectory_peak_features` plus
      whatever extras the caller has. Required keys: ``n_peaks``,
      ``n_clear``, ``pitch_cv``, ``peak_span_mm``, ``snap_span_mm``,
      ``n_large_pitches``.
    * ``models``: list of library model dicts (typically the output of
      :func:`filter_models_for_strategy`).
    * ``arc_length_mm``: intracranial trajectory length (bone-inner →
      fitted tip).
    * ``cluster_envelope_margin``: optional discriminator —
      ``best_cluster_envelope_corr − best_depth_envelope_corr`` over a
      separately-computed metal_width_avg signal. Pass 0.0 when not
      computed.
    * ``candidate_model_ids``: optional restriction list (e.g. the IDs
      surviving a vendor + pitch-strategy filter); when ``None``, all
      ``models`` are eligible.

    Returns a dict with:

    * ``model_id``: predicted model ID (str) or ``None``
    * ``branch``: ``"cluster"`` / ``"pitch_match"`` / ``"regular_AM"`` /
      ``"continuous_AM"`` / ``"none"``
    * ``votes``: dict with ``vote_pitch``, ``vote_envelope``,
      ``vote_length``, ``force_len``, ``cluster_votes``
    * ``feature_snapshot``: copy of ``features`` for the diagnostics
    """
    models_dict = {m["id"]: m for m in models}
    if candidate_model_ids is None:
        candidate_ids = list(models_dict.keys())
    else:
        candidate_ids = [m for m in candidate_model_ids if m in models_dict]
    depth_ids   = [m for m in candidate_ids if model_uniform_pitch(models_dict[m]) is not None]
    cluster_ids = [m for m in candidate_ids if model_uniform_pitch(models_dict[m]) is None]

    n_peaks = int(features.get("n_peaks") or 0)
    n_clear = int(features.get("n_clear") or 0)
    pitch_cv = float(features.get("pitch_cv") or float("inf"))
    peak_span_mm = float(features.get("peak_span_mm") or 0.0)
    snap_span_mm = float(features.get("snap_span_mm") or 0.0)
    n_large_pitches = int(features.get("n_large_pitches") or 0)

    # Cluster-vote machinery.
    max_depth_length = (
        float(max(float(models_dict[m].get("total_exploration_length_mm") or 0.0)
                  for m in depth_ids))
        if depth_ids else 0.0
    )
    force_len = arc_length_mm > (max_depth_length + length_guard_mm)
    vote_pitch    = n_large_pitches >= 2
    vote_envelope = cluster_envelope_margin > cm_bm_margin
    vote_length   = (arc_length_mm > 50.0 and pitch_cv > irregular_cv)
    cluster_votes = int(vote_pitch) + int(vote_envelope) + int(vote_length)
    is_cluster = (force_len or cluster_votes >= min_cluster_votes)

    predicted: str | None = None
    branch = "none"

    if is_cluster and cluster_ids:
        predicted = _pick_minimal_extension_cluster(
            peak_span_mm, cluster_ids, models_dict,
            snap_span_mm=snap_span_mm,
        )
        branch = "cluster"
    else:
        # Depth (uniform-pitch) family. Try pitch_match first when erosion
        # produced a clean signature. The eroded median pitch is also
        # used as a SOFT pitch filter in the fallback paths even when
        # CV is too high to trigger pitch_match — keeps regular_AM /
        # continuous_AM in the right pitch family (AM 3.5 vs MM 3.9/4.8/6.1).
        eroded_clean = (n_clear >= min_clear_for_clean
                        and pitch_cv < clean_pitch_cv_max
                        and peak_span_mm > 0)
        median_pitch = (peak_span_mm / max(1, n_clear - 1)) if n_clear >= 2 else None
        if eroded_clean and depth_ids:
            predicted = _pick_by_pitch_match(
                n_peaks, snap_span_mm, median_pitch,
                depth_ids, models_dict,
                pitch_tol_mm=pitch_match_tol_mm,
            )
            if predicted is not None:
                branch = "pitch_match"
        if predicted is None and depth_ids:
            is_regular = (n_peaks >= min_peaks_for_regular
                          and pitch_cv < pitch_cv_threshold)
            if is_regular:
                predicted = _pick_minimal_extension_am(
                    n_peaks, snap_span_mm, depth_ids, models_dict,
                    target_pitch_mm=median_pitch,
                    pitch_tol_mm=pitch_match_tol_mm,
                )
                if predicted is not None:
                    branch = "regular_AM"
        if predicted is None and depth_ids:
            predicted = _pick_minimal_length_am(
                arc_length_mm, depth_ids, models_dict, n_peaks=n_peaks,
                target_pitch_mm=median_pitch,
                pitch_tol_mm=pitch_match_tol_mm,
            )
            if predicted is not None:
                branch = "continuous_AM"
        if predicted is None and depth_ids and median_pitch is not None:
            # Hard fallback: pitch filter excluded everything. Try once
            # more with no pitch filter so we still emit a model.
            predicted = _pick_minimal_length_am(
                arc_length_mm, depth_ids, models_dict, n_peaks=n_peaks,
            )
            if predicted is not None:
                branch = "continuous_AM_no_pitch"

    return dict(
        model_id=predicted,
        branch=branch,
        votes=dict(
            vote_pitch=vote_pitch,
            vote_envelope=vote_envelope,
            vote_length=vote_length,
            force_len=force_len,
            cluster_votes=cluster_votes,
        ),
        feature_snapshot=dict(features),
    )


def is_relaxed_match(
    predicted: str | None, truth: str | None,
    arc_length_mm: float,
    models: Sequence[dict],
    *,
    candidate_model_ids: Sequence[str] | None = None,
    pitch_match_tol_mm: float = PITCH_MATCH_TOL_MM,
) -> bool:
    """Relaxed-accuracy rule: exact match OR predicted is the next
    contact-count step up within the truth's pitch family AND the
    predicted model's total length > intracranial arc length (so the
    bigger pick overhangs the bone).

    Cluster models on either side require exact match — clusters are
    diagnostic of the trajectory and a wrong cluster pick is never OK.
    """
    if predicted == truth:
        return True
    if predicted is None or truth is None:
        return False
    models_dict = {m["id"]: m for m in models}
    if predicted not in models_dict or truth not in models_dict:
        return False
    pred_pitch  = model_uniform_pitch(models_dict[predicted])
    truth_pitch = model_uniform_pitch(models_dict[truth])
    if pred_pitch is None or truth_pitch is None:
        return False
    if abs(pred_pitch - truth_pitch) > pitch_match_tol_mm:
        return False
    pool = candidate_model_ids if candidate_model_ids is not None else list(models_dict.keys())
    family = sorted(
        [
            m for m in pool
            if m in models_dict
            and model_uniform_pitch(models_dict[m]) is not None
            and abs(model_uniform_pitch(models_dict[m]) - truth_pitch) <= pitch_match_tol_mm
        ],
        key=lambda m: int(models_dict[m].get("contact_count") or 0),
    )
    if predicted not in family or truth not in family:
        return False
    if family.index(predicted) - family.index(truth) != 1:
        return False
    pred_len = float(models_dict[predicted].get("total_exploration_length_mm") or 0.0)
    return pred_len > arc_length_mm


def place_contacts_from_tip(tip_ras, axis_unit, model: dict):
    """Place library-model contacts starting at the trajectory's fitted
    tip and walking inward (against ``axis_unit``, which is oriented
    entry→tip).

    The library's ``contact_center_offsets_from_tip_mm`` are measured
    from the **physical electrode tip** — first entry is typically 1 mm
    (the deepest contact sits 1 mm past the physical tip), then
    increasing toward the most-superficial contact. The snap pipeline's
    fitted tip = center of the deepest detected blob = the deepest
    CONTACT, not the physical tip (plastic, invisible in CT). So we
    re-zero the offsets to the deepest contact before placing:
    ``offsets - offsets[0]`` puts the deepest contact AT the fitted tip,
    with subsequent contacts walking back at the right inter-contact
    spacing.
    """
    import numpy as np
    from .centerline import unit
    offsets = np.asarray(model["contact_center_offsets_from_tip_mm"], dtype=float)
    offsets_from_deepest = offsets - offsets[0]
    direction = -unit(axis_unit)
    return (np.asarray(tip_ras, dtype=float)[None, :]
            + offsets_from_deepest[:, None] * direction[None, :])


__all__ = [
    "PITCH_STRATEGY_VENDORS",
    "VENDOR_ID_PREFIXES",
    "classify_by_count_and_span",
    "classify_by_walker_signature",
    "filter_models_for_strategy",
    "signal_derived_entry_arc",
    "suggest_shortest_covering_model",
    # New library-driven picker (2026-05-18).
    "model_uniform_pitch",
    "trajectory_peak_features",
    "classify_by_pitch_decision_tree",
    "is_relaxed_match",
    "place_contacts_from_tip",
    # Picker thresholds (so callers + tests can tune from one place).
    "CLEAR_PEAK_SIGNAL",
    "CLEAR_PEAK_HEIGHT_FRAC",
    "CLEAR_PEAK_MIN_DIST_MM",
    "LARGE_PITCH_THRESHOLD",
    "CM_BM_MARGIN",
    "STRONG_ENVELOPE_MARGIN",
    "LENGTH_GUARD_MM",
    "IRREGULAR_CV",
    "MIN_CLUSTER_VOTES",
    "PITCH_CV_THRESHOLD",
    "MIN_PEAKS_FOR_REGULAR",
    "PEAK_SPAN_TOLERANCE_MM",
    "CLEAN_PITCH_CV_MAX",
    "MIN_CLEAR_FOR_CLEAN",
    "PITCH_MATCH_TOL_MM",
]
