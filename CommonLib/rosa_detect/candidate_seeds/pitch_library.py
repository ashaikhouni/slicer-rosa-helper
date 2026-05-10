"""Library bounds + per-strategy walker bounds.

The v1 walker has bounds (min/max contacts, min/max span, max within-
electrode pitch) derived from the bundled electrode-model library and
filtered per pitch-strategy. This module owns the library load and the
``WalkerBounds`` immutable record threaded through the orchestrator
into walker / score / crossing-tip retreat.

The orchestrator computes a single ``WalkerBounds`` on entry from the
caller's pitch_strategy and passes it explicitly to every stage that
needs it; no module-global mutation, no decorator magic.
"""
from __future__ import annotations

from dataclasses import dataclass

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


# Module-load-time SEEG library bounds.
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


@dataclass(frozen=True)
class WalkerBounds:
    """Library-derived strategy-scoped bounds threaded through the
    detector pipeline.

    Field-by-field meaning:

    * ``min_line_span_mm`` / ``max_line_span_mm`` — accept-window for
      the walker's pre-anchor inlier span (shallowest → deepest
      contact). Library min/max contact span ± walker slack.
    * ``max_inlier_gap_mm`` — largest axial gap allowed between
      consecutive walker inliers. Library max within-electrode pitch +
      slack for two consecutive missed contacts.
    * ``min_post_anchor_len_mm`` / ``max_post_anchor_len_mm`` —
      accept-window for the post-anchor total length (bolt-tip → deep
      tip). Library contact span + bolt-protrusion / total-overshoot
      slack.

    Computed from a library-bounds dict by ``WalkerBounds.from_library``.
    """
    min_line_span_mm: float
    max_line_span_mm: float
    max_inlier_gap_mm: float
    min_post_anchor_len_mm: float
    max_post_anchor_len_mm: float

    @classmethod
    def from_library(cls, bounds: dict) -> "WalkerBounds":
        """Apply slack constants to a library-bounds dict and return
        the immutable WalkerBounds record.

        ``MIN_BLOBS_PER_LINE`` is intentionally NOT included — it is a
        geometric chain-formation floor (3), not a library-derived
        bound. Letting strategy bounds shadow it caused the AMC137/LPT
        parity bug: pmt_35 set MIN_BLOBS=8 (smallest PMT model) and
        rejected LPT's chain that had <8 visible blobs.
        """
        return cls(
            min_line_span_mm=(
                float(bounds["min_contact_span_mm"]) - WALKER_SPAN_UNDER_SLACK_MM
            ),
            max_line_span_mm=(
                float(bounds["max_contact_span_mm"]) + WALKER_SPAN_OVER_SLACK_MM
            ),
            max_inlier_gap_mm=(
                float(bounds["max_within_electrode_pitch_mm"]) + WALKER_GAP_SLACK_MM
            ),
            min_post_anchor_len_mm=(
                float(bounds["min_contact_span_mm"]) + BOLT_PROTRUSION_MIN_MM
            ),
            max_post_anchor_len_mm=(
                float(bounds["max_contact_span_mm"]) + ANCHOR_TOTAL_OVERSHOOT_MM
            ),
        )


# Module-load default bounds = SEEG-vendor-filtered library bounds.
# Used as the fallback when a stage gets no explicit bounds (CLI
# defaults, unit tests, probe scripts).
DEFAULT_WALKER_BOUNDS: WalkerBounds = WalkerBounds.from_library(LIBRARY_BOUNDS)


def bounds_for_strategy(strategy_key) -> WalkerBounds:
    """Return the WalkerBounds for the user-chosen pitch strategy.

    Resolves the strategy's vendor filter via
    ``library_bounds_for_strategy`` and applies the slack constants.
    Unknown / unset strategy keys fall back to ``DEFAULT_WALKER_BOUNDS``
    (SEEG-vendor-filtered).
    """
    return WalkerBounds.from_library(library_bounds_for_strategy(strategy_key))


__all__ = [
    "LIBRARY_BOUNDS",
    "LIBRARY_PITCHES_MM",
    "WalkerBounds",
    "DEFAULT_WALKER_BOUNDS",
    "model_vendor",
    "compute_library_bounds",
    "library_bounds_for_strategy",
    "bounds_for_strategy",
]
