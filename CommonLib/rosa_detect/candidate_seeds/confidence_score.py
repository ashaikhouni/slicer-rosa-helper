"""Continuous physical-evidence confidence score for emitted trajectories.

Each emitted trajectory carries a continuous score in [0, 1] assembled
from the same measurements the hard gates already apply: amp_sum,
n_inliers, frangi shaft response, on-library pitch, pre-anchor contact
span, post-anchor length, intracranial depth, bolt-anchor source, and
along-axis metal continuity. Each component saturates at the
corresponding gate threshold (``measure >= threshold`` → 1.0), so a
clean SEEG line scores near 1.0 on every term.

v1 keeps the existing emit-time gates as fallbacks. The score is
attached as metadata (``score``, ``confidence``, ``score_components``)
on every survivor so downstream code can rank, filter, or surface the
weakest emissions for review without changing detection behaviour.

Strategy-scoped span / length bounds (``min_line_span_mm``,
``max_line_span_mm``, ``min_post_anchor_len_mm``,
``max_post_anchor_len_mm``) are passed explicitly via the
:class:`WalkerBounds` argument (defaults to ``DEFAULT_WALKER_BOUNDS``).
"""
from __future__ import annotations

from .constants import (
    FRANGI_LINE_MIN_MEDIAN,
    MIN_BLOBS_PER_LINE,
    SCORE_AMP_SAT,
    SCORE_BOLT_VALUES,
    SCORE_DEPTH_SAT_MM,
    SCORE_HIGH_THRESHOLD,
    SCORE_INTRACRANIAL_SAT_MM,
    SCORE_LENGTH_SHOULDER_MM,
    SCORE_MEDIUM_THRESHOLD,
    SCORE_METAL_CONTINUITY_SAT,
    SCORE_N_INLIERS_OVER_SLACK,
    SCORE_N_INLIERS_SLOPE,
    SCORE_PITCH_TOL_MM,
    SCORE_SPAN_SHOULDER_MM,
    SCORE_WEIGHTS,
)
from .pitch_library import (
    DEFAULT_WALKER_BOUNDS,
    LIBRARY_BOUNDS,
    LIBRARY_PITCHES_MM,
    WalkerBounds,
)


def trapezoid_score(value, lo, hi, shoulder_mm):
    """1.0 inside [lo, hi]; linear falloff to 0 over ``shoulder_mm``."""
    if value < lo - shoulder_mm or value > hi + shoulder_mm:
        return 0.0
    if value < lo:
        return float((value - (lo - shoulder_mm)) / shoulder_mm)
    if value > hi:
        return float(((hi + shoulder_mm) - value) / shoulder_mm)
    return 1.0


def bolt_source_score(src):
    return SCORE_BOLT_VALUES.get(str(src), 0.5)


def compute_trajectory_score(rec, bounds: WalkerBounds | None = None):
    """Return (score, confidence, components) for one trajectory record."""
    if bounds is None:
        bounds = DEFAULT_WALKER_BOUNDS
    MIN_LINE_SPAN_MM = bounds.min_line_span_mm
    MAX_LINE_SPAN_MM = bounds.max_line_span_mm
    MIN_POST_ANCHOR_LEN_MM = bounds.min_post_anchor_len_mm
    MAX_POST_ANCHOR_LEN_MM = bounds.max_post_anchor_len_mm

    components = {}
    is_wire_class = bool(rec.get("wire_class"))

    # ``amp_sum`` and ``n_inliers`` are walker-only signals. Wire-class
    # trajectories come from a bolt-CC PCA fit and have neither — they
    # would force-zero those components and drag the score artificially
    # low. Skip them and let the remaining components (frangi, span,
    # length, depth, intracranial, bolt, metal_continuity) carry the
    # signal.
    if "amp_sum" in rec and not is_wire_class:
        components["amp"] = (
            min(1.0, max(0.0, float(rec["amp_sum"]) / SCORE_AMP_SAT)),
            SCORE_WEIGHTS["amp"],
        )

    if not is_wire_class:
        n = int(rec.get("n_inliers", 0))
        # Lower side: linear ramp from MIN_BLOBS_PER_LINE up by SCORE_N_INLIERS_SLOPE.
        # Upper side: 1.0 up to the library's max contact count, then linear
        # falloff over SCORE_N_INLIERS_OVER_SLACK. n far above the library
        # max means the walker chained a continuous metal structure (the
        # bolt itself, an insulated wire shaft) instead of discrete contacts.
        lib_max = int(LIBRARY_BOUNDS["max_contacts"])
        if n <= MIN_BLOBS_PER_LINE:
            n_score = 0.0
        elif n <= MIN_BLOBS_PER_LINE + SCORE_N_INLIERS_SLOPE:
            n_score = (n - MIN_BLOBS_PER_LINE) / SCORE_N_INLIERS_SLOPE
        elif n <= lib_max:
            n_score = 1.0
        else:
            n_score = max(0.0, 1.0 - (n - lib_max) / SCORE_N_INLIERS_OVER_SLACK)
        components["n_inliers"] = (n_score, SCORE_WEIGHTS["n_inliers"])

    if "frangi_median_mm" in rec:
        components["frangi"] = (
            min(1.0, max(0.0, float(rec["frangi_median_mm"]) / FRANGI_LINE_MIN_MEDIAN)),
            SCORE_WEIGHTS["frangi"],
        )

    if "frac_strong_metal" in rec:
        components["metal_continuity"] = (
            min(1.0, max(0.0, float(rec["frac_strong_metal"]) / SCORE_METAL_CONTINUITY_SAT)),
            SCORE_WEIGHTS["metal_continuity"],
        )

    pitch = rec.get("original_median_pitch_mm")
    if pitch is not None and float(pitch) > 0.0:
        dev = min(abs(float(pitch) - lib) for lib in LIBRARY_PITCHES_MM)
        components["pitch"] = (
            min(1.0, max(0.0, 1.0 - dev / SCORE_PITCH_TOL_MM)),
            SCORE_WEIGHTS["pitch"],
        )

    if "contact_span_mm" in rec:
        components["span"] = (
            trapezoid_score(
                float(rec["contact_span_mm"]),
                MIN_LINE_SPAN_MM, MAX_LINE_SPAN_MM,
                SCORE_SPAN_SHOULDER_MM,
            ),
            SCORE_WEIGHTS["span"],
        )

    length = float(rec.get("length_mm", 0.0))
    if length > 0:
        components["length"] = (
            trapezoid_score(
                length,
                MIN_POST_ANCHOR_LEN_MM, MAX_POST_ANCHOR_LEN_MM,
                SCORE_LENGTH_SHOULDER_MM,
            ),
            SCORE_WEIGHTS["length"],
        )

    dist_max = float(rec.get("dist_max_mm", 0.0))
    components["depth"] = (
        min(1.0, max(0.0, dist_max / SCORE_DEPTH_SAT_MM)),
        SCORE_WEIGHTS["depth"],
    )

    dist_mean = rec.get("dist_mean_mm")
    if dist_mean is not None and float(dist_mean) == float(dist_mean):
        components["intracranial"] = (
            min(1.0, max(0.0, float(dist_mean) / SCORE_INTRACRANIAL_SAT_MM)),
            SCORE_WEIGHTS["intracranial"],
        )

    bolt_src = str(rec.get("bolt_source", "metal"))
    components["bolt"] = (
        bolt_source_score(bolt_src),
        SCORE_WEIGHTS["bolt"],
    )

    weighted = sum(v * w for v, w in components.values())
    total_w = sum(w for _, w in components.values())
    score = weighted / total_w if total_w > 0 else 0.0

    # Confidence policy: high band is reserved for trajectories with BOTH
    # contact-pitch validation AND a real metal bolt CC (bolt_source ==
    # "metal"). Anything missing one or the other caps at medium:
    #
    #   pitch + metal bolt    → high allowed
    #   pitch + synthesized   → cap medium  (CT didn't capture the bolt;
    #                                         the synth fallback is a
    #                                         best-guess on degraded input)
    #   pitch + no anchor     → cap medium  (bolt_source == "none")
    #   bolt CC + no pitch    → cap medium  (wire_class; metal_cc bolt)
    #
    # Wire-class records carry bolt_source == "metal_cc" by construction,
    # so the single ``bolt_src != "metal"`` test covers them too.
    if bolt_src != "metal" and score >= SCORE_HIGH_THRESHOLD:
        score = SCORE_HIGH_THRESHOLD - 0.01

    if score >= SCORE_HIGH_THRESHOLD:
        label = "high"
    elif score >= SCORE_MEDIUM_THRESHOLD:
        label = "medium"
    else:
        label = "low"

    return score, label, {k: float(v) for k, (v, _) in components.items()}


__all__ = [
    "trapezoid_score",
    "bolt_source_score",
    "compute_trajectory_score",
]
