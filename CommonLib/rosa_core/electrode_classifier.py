"""Electrode-model classification — shared picker for Auto Fit, Guided
Fit, Manual Fit, and Contacts & Trajectory View.

Three scoring strategies, dispatched by ``classify_electrode_model`` based on
which inputs the caller supplies:

1. **PaCER template-correlation** (preferred when a CT volume is available).
   Samples a 1D max-disk intensity profile along the trajectory and scores
   each library candidate by `NCC(profile, model_template) × coverage`,
   where the template places Gaussian bumps at the model's
   `contact_center_offsets_from_tip_mm`. Coverage = fraction of model
   contacts whose expected position falls inside the profile range —
   penalizes longer-but-clipped models so a 10-contact line doesn't tie
   with a 12-contact line whose entry-side overflow gets clipped. Validated
   on T22 (9/9 GT match, median margin 0.134) per
   `tests/deep_core/probe_pacer_picker_t22.py`.

2. **Walker-signature joint scoring** (legacy; used when CT not available
   but walker stats are). Scores by
   `pitch_err·10 + count_err·3.5 + max(0, span_err-2)·1 + max(0, length_err-10)·0.5`.

3. **Length-only with dura tolerance** (last-ditch fallback). Picks the
   shortest model whose total exploration length plus a 10 mm dura tolerance
   covers the observed intracranial length.

All three respect `pitch_strategy` filtering (vendor + pitch set).
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
# PaCER-style template correlation (CT-aware picker)
# ---------------------------------------------------------------------

# Profile sampling — matches the validated 9/9 probe on T22
# (`tests/deep_core/probe_pacer_picker_t22.py`). Wider disks (r=2.5)
# from the robustness sweep absorb more lateral axis error but smear
# the inter-contact gap structure that distinguishes BM/CM electrodes
# (e.g. 15CM's three 5-contact groups vs 18CM's three 6-contact
# groups), which matters more for picker accuracy than the marginal
# lateral robustness gain.
_PACER_PROFILE_STEP_MM = 0.25
_PACER_PROFILE_DISK_RADIUS_MM = 1.0
_PACER_PROFILE_DISK_N_RADII = 2
_PACER_PROFILE_DISK_N_ANGLES = 8
_PACER_PROFILE_REDUCER = "max"

# Tight pads keep the bolt out of the profile — bolt is contiguous bright
# metal that correlates with the first 1-2 contacts of any candidate.
_PACER_PAD_TIP_MM = 1.5
_PACER_PAD_ENTRY_MM = 1.5

# Tip-arc slide: keep candidate tip near the deep end of the profile.
_PACER_TIP_ARC_SEARCH_HALF_WIDTH_MM = 5.0
_PACER_TEMPLATE_SIGMA_MM = 0.6
_PACER_TEMPLATE_OFFSET_STEP_MM = 0.25


def _build_pacer_template(model, profile_arc_mm, tip_arc_mm,
                           arc_lower=None, arc_upper=None):
    """Place a Gaussian bump at each contact position. Returns
    (template, coverage) where coverage = fraction of model contacts
    whose expected arc-position lies inside the valid arc range
    `[arc_lower, arc_upper]` (±3σ slack).

    The valid range defaults to the full profile, but can be tightened
    when an entry-side anchor (bolt-electrode boundary) is available —
    contacts that fall in the bolt-muted region get coverage-penalized,
    so longer templates whose entry-side contacts extend into the
    muted region don't tie with shorter templates that fit in the
    visible region.
    """
    offsets = np.asarray(
        model.get("contact_center_offsets_from_tip_mm", []), dtype=float
    )
    if offsets.size == 0:
        return np.zeros_like(profile_arc_mm), 0.0
    if arc_lower is None:
        arc_lower = float(profile_arc_mm[0])
    if arc_upper is None:
        arc_upper = float(profile_arc_mm[-1])
    positions = tip_arc_mm - offsets
    in_range = (positions >= arc_lower - 3.0 * _PACER_TEMPLATE_SIGMA_MM) & (
        positions <= arc_upper + 3.0 * _PACER_TEMPLATE_SIGMA_MM
    )
    coverage = float(np.count_nonzero(in_range)) / float(positions.size)
    if not np.any(in_range):
        return np.zeros_like(profile_arc_mm), coverage
    template = np.zeros_like(profile_arc_mm)
    sigma_sq = _PACER_TEMPLATE_SIGMA_MM ** 2
    for pos in positions[in_range]:
        template += np.exp(-0.5 * (profile_arc_mm - pos) ** 2 / sigma_sq)
    return template, coverage


def _normalized_cross_correlation(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    if int(finite.sum()) < 5:
        return float("nan")
    a = a[finite] - np.mean(a[finite])
    b = b[finite] - np.mean(b[finite])
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


_PACER_METAL_THRESHOLD_HU = 1000.0
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


def _signal_derived_entry_arc(profile_arc_mm, profile_values):
    """Detect the bolt → electrode boundary by the bolt's sustained-
    bright signature, and return the arc-position where the actual
    electrode region starts.

    Key distinction (per user observation): a single SEEG contact is
    a small bright peak ~2-3 mm wide; a bolt is a large sustained
    bright stretch typically 5-15 mm long. So:

      1. Find runs of metal-bright (HU ≥ `_PACER_BOLT_THRESHOLD_HU`).
      2. The first run whose length ≥ `_PACER_BOLT_MIN_RUN_MM` is
         the BOLT (a contact would be too short).
      3. After the bolt, find the next metal-bright run preceded by
         a dim gap ≥ `_PACER_BOLT_GAP_MIN_MM`. Its start is the
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
    # `arc[bolt_end] + gap_min_mm` so the electrode-side contacts
    # are not muted.
    bolt_end_idx = ends[bolt_idx] - 1  # last sample within bolt run
    if bolt_end_idx >= len(profile_arc_mm) - 1:
        return None
    return float(profile_arc_mm[bolt_end_idx]) + float(_PACER_BOLT_GAP_MIN_MM)


def _signal_derived_tip_arc(profile_arc_mm, profile_values):
    """Return the deepest arc position with intensity above the metal
    threshold, or None when no metal-bright sample exists.

    The deep end of an Auto Fit-detected trajectory frequently
    over-extends the actual deepest contact by 5-20 mm (walker's
    deep-tip refinement can lock onto a halo / skull-bone artifact
    past the real contact). When that happens, `profile_arc[-1]` is
    well past the true tip — sliding the candidate tip in a small
    window around `profile_arc[-1] - PAD_TIP_MM` never reaches the
    actual deepest contact, and the picker is dragged toward longer-
    than-truth models. Anchoring the slide on the *signal-derived*
    tip (deepest bright voxel) self-corrects this without needing a
    pre-trim of the axis upstream.
    """
    arr = np.asarray(profile_values, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return None
    bright = finite & (arr > _PACER_METAL_THRESHOLD_HU)
    if not np.any(bright):
        return None
    last_idx = int(np.where(bright)[0][-1])
    return float(profile_arc_mm[last_idx])


def _score_model_pacer(profile_arc_mm, profile_values, model,
                       expected_tip_arc=None,
                       arc_lower=None, arc_upper=None):
    """Slide tip-arc across a narrow window, score = NCC × coverage.
    Returns (best_score, best_tip_arc_mm, best_coverage).

    `expected_tip_arc` defaults to `profile_arc[-1] - PAD_TIP_MM`.
    `arc_lower` / `arc_upper` constrain the coverage check so longer
    templates whose entry-side contacts extend into a bolt-muted
    region drop coverage proportionally.
    """
    if expected_tip_arc is None:
        expected_tip_arc = profile_arc_mm[-1] - _PACER_PAD_TIP_MM
    arc_lo = expected_tip_arc - _PACER_TIP_ARC_SEARCH_HALF_WIDTH_MM
    arc_hi = expected_tip_arc + _PACER_TIP_ARC_SEARCH_HALF_WIDTH_MM
    candidate_tip_arcs = np.arange(arc_lo, arc_hi, _PACER_TEMPLATE_OFFSET_STEP_MM)
    best_score = -np.inf
    best_tip_arc = float("nan")
    best_coverage = 0.0
    for tip_arc in candidate_tip_arcs:
        tpl, coverage = _build_pacer_template(
            model, profile_arc_mm, tip_arc,
            arc_lower=arc_lower, arc_upper=arc_upper,
        )
        if tpl.max() <= 0.0:
            continue
        ncc = _normalized_cross_correlation(profile_values, tpl)
        if not np.isfinite(ncc):
            continue
        s = ncc * coverage
        if s > best_score:
            best_score = s
            best_tip_arc = float(tip_arc)
            best_coverage = coverage
    return best_score, best_tip_arc, best_coverage


def classify_pacer_template(start_ras, end_ras, ct_volume_kji, ras_to_ijk_mat,
                             models, vendors=("Dixi", "PMT", "Medtronic"),
                             pad_tip_mm=_PACER_PAD_TIP_MM,
                             pad_entry_mm=_PACER_PAD_ENTRY_MM):
    """PaCER-style template-correlation picker.

    Samples a 1D max-disk intensity profile along the trajectory axis
    (extended by `pad_*` on each end), scores each library candidate by
    `NCC × coverage`, returns the highest-scoring model plus its expected
    per-contact RAS positions.

    `start_ras` should be the entry side, `end_ras` the deep tip side.
    Tight pads keep the bolt out of the profile.

    Returns `dict` with `model_id`, `score`, `tip_arc_mm`, `coverage`,
    `contacts_ras` (list of [x,y,z] per contact at the expected position
    along the axis given the matched tip placement), `runner_up_id`,
    `runner_up_score`, `margin`. Returns `None` if no viable candidate.
    """
    # Local import to avoid a circular dep when this module is imported
    # by `contact_peak_fit` itself in some build paths.
    from .contact_peak_fit import sample_axis_profile

    prefixes = _vendor_prefixes(vendors)
    if not prefixes:
        return None
    eligible = [m for m in models if str(m.get("id", "")).startswith(prefixes)]
    if not eligible:
        return None

    s = np.asarray(start_ras, dtype=float).reshape(3)
    e = np.asarray(end_ras, dtype=float).reshape(3)
    direction = e - s
    L = float(np.linalg.norm(direction))
    if L < 1e-3:
        return None
    unit = direction / L
    sample_start = s - float(pad_entry_mm) * unit
    sample_end = e + float(pad_tip_mm) * unit

    try:
        arc_mm, profile = sample_axis_profile(
            volume_kji=ct_volume_kji,
            ras_to_ijk_mat=np.asarray(ras_to_ijk_mat, dtype=float),
            start_ras=sample_start, end_ras=sample_end,
            step_mm=_PACER_PROFILE_STEP_MM,
            disk_radius_mm=_PACER_PROFILE_DISK_RADIUS_MM,
            n_radii=_PACER_PROFILE_DISK_N_RADII,
            n_angles=_PACER_PROFILE_DISK_N_ANGLES,
            reducer=_PACER_PROFILE_REDUCER,
        )
    except Exception:
        return None

    # Bolt-trim: when a bolt's sustained-bright + high-peak signature
    # is detected, mute the profile values BEFORE the electrode entry
    # arc AND tighten the coverage lower bound to the same arc. The
    # mute prevents bolt brightness from contributing to NCC; the
    # coverage-bound penalty drops longer templates that try to
    # "explain" the bolt with their entry-side contacts.
    entry_arc = _signal_derived_entry_arc(arc_mm, profile)
    arc_lower = float(arc_mm[0])
    if entry_arc is not None:
        cutoff = float(entry_arc) - 1.0  # 1 mm safety margin
        mask = arc_mm < cutoff
        if np.any(mask):
            profile = np.array(profile, copy=True)
            profile[mask] = np.nan
        arc_lower = float(entry_arc)
    arc_upper = float(arc_mm[-1])
    # Anchor the candidate tip on the signal (deepest metal-bright
    # voxel) when present, else fall back to the geometry default
    # (`profile_arc[-1] - pad_tip_mm`). Auto-corrects for axis
    # over-extension at the deep end.
    expected_tip = _signal_derived_tip_arc(arc_mm, profile)
    if expected_tip is None:
        expected_tip = float(arc_mm[-1]) - float(pad_tip_mm)
    scored = []
    for m in eligible:
        s_score, tip_arc, cov = _score_model_pacer(
            arc_mm, profile, m, expected_tip_arc=expected_tip,
            arc_lower=arc_lower, arc_upper=arc_upper,
        )
        if not np.isfinite(s_score):
            continue
        scored.append((s_score, m, tip_arc, cov))
    if not scored:
        return None
    scored.sort(key=lambda r: r[0], reverse=True)
    best_score, best_model, best_tip, best_cov = scored[0]
    runner_score = float("nan"); runner_id = ""
    if len(scored) > 1:
        runner_score = float(scored[1][0])
        runner_id = str(scored[1][1].get("id", ""))

    # Compute per-contact RAS positions: walk back from the tip along
    # `-unit` (toward entry) by each contact's offset_from_tip.
    offsets = np.asarray(
        best_model.get("contact_center_offsets_from_tip_mm", []), dtype=float
    )
    # Tip in RAS: arc-length `best_tip` from `sample_start` along `unit`.
    tip_ras = sample_start + float(best_tip) * unit
    contacts_ras = []
    for off in offsets:
        contact_ras = tip_ras - float(off) * unit
        contacts_ras.append([float(contact_ras[0]),
                             float(contact_ras[1]),
                             float(contact_ras[2])])

    return {
        "model_id": str(best_model.get("id", "")),
        "score": float(best_score),
        "tip_arc_mm": float(best_tip),
        "tip_ras": [float(tip_ras[0]), float(tip_ras[1]), float(tip_ras[2])],
        "coverage": float(best_cov),
        "contacts_ras": contacts_ras,
        "runner_up_id": runner_id,
        "runner_up_score": runner_score,
        "margin": float(best_score - runner_score) if np.isfinite(runner_score) else float("nan"),
        "method": "pacer_template",
    }


# ---------------------------------------------------------------------
# Unified entry point — dispatcher
# ---------------------------------------------------------------------

def classify_electrode_model(start_ras, end_ras, *,
                             library=None,
                             models=None,
                             pitch_strategy=None,
                             vendors=None,
                             ct_volume_kji=None,
                             ras_to_ijk_mat=None,
                             walker_signature=None,
                             intracranial_length_mm=None,
                             dura_tolerance_mm=10.0):
    """Single-entry electrode-model picker, used by Auto Fit, Guided Fit,
    Manual Fit, and Contacts & Trajectory View.

    Inputs:
      * `start_ras`, `end_ras` — required. Entry-side and deep-tip-side
        endpoints of the trajectory in RAS.
      * `library` — full electrode library dict; or pass pre-filtered
        `models` directly.
      * `pitch_strategy` — vendor + pitch-set filter
        (`"dixi"` / `"pmt"` / etc.). Applied to the library.
      * `vendors` — vendor prefix filter as a fallback when no
        `pitch_strategy` is given.
      * `ct_volume_kji`, `ras_to_ijk_mat` — when provided, enables PaCER
        template-correlation scoring (preferred). Profile is sampled
        along the trajectory at native resolution.
      * `walker_signature` — `(n, pitch_mm, span_mm)` tuple from a
        prior walker run. Used for legacy walker-signature scoring
        when CT not provided.
      * `intracranial_length_mm` — defaults to `|start − end|`.

    Strategy resolution:
      1. CT volume present → `classify_pacer_template` (preferred).
      2. Walker signature present → `classify_by_walker_signature`.
      3. Else → `suggest_shortest_covering_model` (length-only).

    Returns a dict with at least `model_id`, `score`, `method`. PaCER
    mode also returns `contacts_ras`, `tip_ras`, `margin`. Returns `None`
    if no viable candidate.
    """
    # Resolve model list.
    if models is None:
        if library is None:
            from .electrode_models import load_electrode_library
            library = load_electrode_library()
        models = list(library.get("models") or [])
    models = filter_models_for_strategy(models, pitch_strategy) or models

    # Resolve vendor filter.
    if vendors is None:
        if pitch_strategy:
            vendors = PITCH_STRATEGY_VENDORS.get(
                str(pitch_strategy).lower(),
                tuple(VENDOR_ID_PREFIXES.keys()),
            )
        else:
            vendors = tuple(VENDOR_ID_PREFIXES.keys())

    # Resolve intracranial length. start/end are optional when an
    # explicit length is passed (callers without endpoints, e.g.
    # legacy walker-signature path).
    have_endpoints = start_ras is not None and end_ras is not None
    if intracranial_length_mm is None:
        if not have_endpoints:
            return None
        s = np.asarray(start_ras, dtype=float).reshape(3)
        e = np.asarray(end_ras, dtype=float).reshape(3)
        intracranial_length_mm = float(np.linalg.norm(e - s))

    # PaCER mode (preferred — needs endpoints + a CT volume).
    if (ct_volume_kji is not None and ras_to_ijk_mat is not None
            and have_endpoints):
        result = classify_pacer_template(
            start_ras, end_ras, ct_volume_kji, ras_to_ijk_mat,
            models, vendors=vendors,
        )
        if result is not None:
            result["intracranial_length_mm"] = float(intracranial_length_mm)
            return result
        # Fall through if no candidate scored.

    # Walker-signature mode.
    if walker_signature is not None:
        n_obs, pitch_obs, span_obs = walker_signature
        result = classify_by_walker_signature(
            n_observed=n_obs,
            pitch_observed_mm=pitch_obs,
            contact_span_observed_mm=span_obs,
            intracranial_length_mm=intracranial_length_mm,
            models=models, vendors=vendors,
            dura_tolerance_mm=dura_tolerance_mm,
        )
        if result is not None:
            result["method"] = "walker_signature"
            result["intracranial_length_mm"] = float(intracranial_length_mm)
            return result

    # Length-only fallback.
    result = suggest_shortest_covering_model(
        intracranial_length_mm, models, vendors=vendors,
        dura_tolerance_mm=dura_tolerance_mm,
    )
    if result is not None:
        return {
            "model_id": result["model_id"],
            "score": float(abs(result["gap_mm"])),
            "method": "shortest_covering",
            "model_length_mm": float(result["model_length_mm"]),
            "gap_mm": float(result["gap_mm"]),
            "intracranial_length_mm": float(intracranial_length_mm),
        }
    return None
