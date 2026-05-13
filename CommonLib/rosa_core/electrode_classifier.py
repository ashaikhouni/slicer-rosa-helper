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


__all__ = [
    "PITCH_STRATEGY_VENDORS",
    "VENDOR_ID_PREFIXES",
    "classify_by_count_and_span",
    "classify_by_walker_signature",
    "filter_models_for_strategy",
    "signal_derived_entry_arc",
    "suggest_shortest_covering_model",
]
