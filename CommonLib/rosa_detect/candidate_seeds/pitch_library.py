"""Library bounds + strategy-scoped walker constants.

The v1 walker has bounds (min/max contacts, min/max span, max within-
electrode pitch) derived from the bundled electrode-model library —
filtered per pitch-strategy. The ``_with_strategy_bounds`` decorator +
``_StrategyBoundsScope`` context manager swap module-level constants
in cpfit for the duration of one detection call so the walker sees the
bounds appropriate for the user-chosen strategy.

Lifted from ``contact_pitch_v1_fit.py`` (Session 4 Phase B Move 3).
The scope manager mutates ``contact_pitch_v1_fit.__dict__`` directly so
the walker — which still reads constants as cpfit module-level names —
sees the swapped values without having to be passed them. Will be
cleaned up in Move 7 when the orchestrator owns this state.
"""
from __future__ import annotations

import inspect

from rosa_core.electrode_classifier import PITCH_STRATEGY_VENDORS

from .constants import (
    ANCHOR_TOTAL_OVERSHOOT_MM,
    BOLT_PROTRUSION_MIN_MM,
    SEEG_VENDORS,
    WALKER_GAP_SLACK_MM,
    WALKER_SPAN_OVER_SLACK_MM,
    WALKER_SPAN_UNDER_SLACK_MM,
    _BUNDLED_LIBRARY_BOUNDS_FALLBACK,
)


def model_vendor(model) -> str:
    """Resolve a vendor name for a model entry. Newer entries carry an
    explicit ``vendor`` field; older Dixi / PMT entries didn't, so fall
    back to the id prefix.
    """
    v = str(model.get("vendor") or "").strip()
    if v:
        return v
    mid = str(model.get("id") or "")
    if mid.startswith("DIXI-"):
        return "Dixi"
    if mid.startswith("PMT-"):
        return "PMT"
    if mid.startswith("Medtronic_"):
        return "Medtronic"
    return ""


def compute_library_bounds(vendors=SEEG_VENDORS) -> dict:
    """Walker / length bounds extracted from the electrode library,
    filtered to a set of vendor names.

    Default ``vendors=SEEG_VENDORS`` so that adding a non-SEEG family
    (Medtronic DBS leads with 4 contacts) does not silently lower the
    SEEG walker's ``min_contacts`` from 5 to 4. Pass a different tuple
    (or ``None`` for no filter) when the caller wants bounds tuned to
    a non-SEEG strategy.
    """
    vendor_set = None if vendors is None else {str(v) for v in vendors}
    try:
        from rosa_core.electrode_models import load_electrode_library
        models = list((load_electrode_library() or {}).get("models") or [])
    except Exception:
        models = []
    if not models:
        return dict(_BUNDLED_LIBRARY_BOUNDS_FALLBACK)
    counts, spans = [], []
    all_pitches, regular_pitches = set(), set()
    for m in models:
        if vendor_set is not None and model_vendor(m) not in vendor_set:
            continue
        counts.append(int(m["contact_count"]))
        offsets = [float(x) for x in m["contact_center_offsets_from_tip_mm"]]
        spans.append(offsets[-1] - offsets[0])
        gaps = [round(offsets[i + 1] - offsets[i], 2)
                for i in range(len(offsets) - 1)]
        all_pitches.update(gaps)
        # Smallest gap of each model is its regular intra-block pitch;
        # larger jumps (BM 9 mm, CM 13 mm) are insulation gaps.
        regular_pitches.add(round(min(gaps), 2))
    if not counts:
        return dict(_BUNDLED_LIBRARY_BOUNDS_FALLBACK)
    return {
        "min_contacts": min(counts),
        "max_contacts": max(counts),
        "min_contact_span_mm": min(spans),
        "max_contact_span_mm": max(spans),
        "max_within_electrode_pitch_mm": max(all_pitches),
        "regular_pitches_mm": tuple(sorted(regular_pitches)),
    }


# Module-load-time SEEG library bounds. Cpfit reads ``LIBRARY_BOUNDS``
# from here and derives MIN_LINE_SPAN_MM etc. from it. Strategy-scoped
# overrides happen at call time via ``_StrategyBoundsScope``.
LIBRARY_BOUNDS: dict = compute_library_bounds()
LIBRARY_PITCHES_MM: tuple = LIBRARY_BOUNDS["regular_pitches_mm"]


def library_bounds_for_strategy(strategy_key) -> dict:
    """Return library-derived bounds filtered to the strategy's vendor
    set. Default (unknown / unset strategy) falls back to SEEG bounds.
    """
    key = str(strategy_key or "").strip().lower()
    vendors = PITCH_STRATEGY_VENDORS.get(key)
    if not vendors:
        return LIBRARY_BOUNDS
    return compute_library_bounds(vendors)


def strategy_global_overrides(bounds) -> dict:
    """Translate a library-bounds dict into the module-level walker
    constants the detector reads.

    NOTE: ``MIN_BLOBS_PER_LINE`` is intentionally NOT in this swap — it
    is a geometric chain-formation floor (3), not a library-derived
    bound. Letting strategy bounds shadow it caused the AMC137/LPT
    parity bug: pmt_35 set MIN_BLOBS=8 (smallest PMT model) and rejected
    LPT's chain that had <8 visible blobs.
    """
    return {
        "MIN_LINE_SPAN_MM": float(bounds["min_contact_span_mm"]) - WALKER_SPAN_UNDER_SLACK_MM,
        "MAX_LINE_SPAN_MM": float(bounds["max_contact_span_mm"]) + WALKER_SPAN_OVER_SLACK_MM,
        "MAX_INLIER_GAP_MM": float(bounds["max_within_electrode_pitch_mm"]) + WALKER_GAP_SLACK_MM,
        "MIN_POST_ANCHOR_LEN_MM": float(bounds["min_contact_span_mm"]) + BOLT_PROTRUSION_MIN_MM,
        "MAX_POST_ANCHOR_LEN_MM": float(bounds["max_contact_span_mm"]) + ANCHOR_TOTAL_OVERSHOOT_MM,
    }


class StrategyBoundsScope:
    """Context manager that swaps cpfit's module-level walker constants
    to a strategy's vendor-filtered bounds for the duration of one
    detection call.

    Pipeline calls are single-threaded per Slicer scene, so the scope
    is safe; we restore the SEEG defaults on exit so a Medtronic call
    can't leak into a subsequent Dixi call.

    The scope mutates ``contact_pitch_v1_fit.__dict__`` directly because
    the walker (still living in cpfit during the staged extraction) reads
    its constants as module-level names. After Move 7 (orchestrator
    extraction), this can become an explicit context object passed
    through the stages.
    """

    def __init__(self, strategy_key):
        self.strategy_key = strategy_key
        self._saved = None
        self.bounds = None

    def __enter__(self):
        from .. import contact_pitch_v1_fit as cpfit
        self.bounds = library_bounds_for_strategy(self.strategy_key)
        new_g = strategy_global_overrides(self.bounds)
        g = cpfit.__dict__
        self._saved = {k: g[k] for k in new_g}
        g.update(new_g)
        return self.bounds

    def __exit__(self, exc_type, exc, tb):
        if self._saved is not None:
            from .. import contact_pitch_v1_fit as cpfit
            cpfit.__dict__.update(self._saved)
            self._saved = None
        return False


def with_strategy_bounds(fn):
    """Decorator: swap walker constants to the call's ``pitch_strategy``
    for the duration of the decorated function, then restore the SEEG
    default. Reads ``pitch_strategy`` via ``inspect.signature.bind`` so
    it works for both positional and keyword callers.
    """
    sig = inspect.signature(fn)

    def wrapper(*args, **kwargs):
        try:
            bound = sig.bind_partial(*args, **kwargs)
            strategy = bound.arguments.get("pitch_strategy")
        except TypeError:
            strategy = kwargs.get("pitch_strategy")
        with StrategyBoundsScope(strategy):
            return fn(*args, **kwargs)

    wrapper.__wrapped__ = fn
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


__all__ = [
    "LIBRARY_BOUNDS",
    "LIBRARY_PITCHES_MM",
    "model_vendor",
    "compute_library_bounds",
    "library_bounds_for_strategy",
    "strategy_global_overrides",
    "StrategyBoundsScope",
    "with_strategy_bounds",
]
