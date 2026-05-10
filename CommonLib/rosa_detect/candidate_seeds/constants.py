"""Tunable knobs for the v1 SEEG shank detector.

Lifted from ``contact_pitch_v1_fit.py`` as part of the 2026-05-10 stage
extraction (``handoff_v3_production_lift_2026-05-09`` Session 4 Phase B).
Each constant has its calibration rationale inline; longer prose lives
in the per-stage memory entries.

Section organization:
  1. Blob extraction (LoG)
  2. Walker / pitch tolerance
  3. Auto-pitch detection
  4. Library bounds
  5. Walker line scoring (span / gap / blob count)
  6. Stage-1 dedup
  7. Deep-tip floor
  8. Bolt anchoring + post-anchor dedup
  9. Synth bolt fallback (axis-to-skull)
 10. Frangi gate (post-anchor)
 11. Post-arbitration loosening
 12. Axis-deep-end refinement
 13. Crossing-tip retreat
 14. Confidence score
 15. Wire-class extension

Constants that depend on the dynamically-loaded electrode library
(`LIBRARY_PITCHES_MM`, `MIN_LINE_SPAN_MM`, etc.) live in
``pitch_library.py`` because their initialization runs only after the
library load.
"""
from __future__ import annotations


# ---------------------------------------------------------------------
# 1. Blob extraction (LoG)
# ---------------------------------------------------------------------

# |LoG| floor for accepting a contact-sized local minimum. Calibrated
# for the Box r=1 (3x3x3) blob extractor + sub-voxel quadratic
# refinement. ~50% of typical per-contact LoG (1062), ~2x cross-scanner
# safety margin. See `contact_pitch_v1_fit` original docstring for the
# full calibration rationale.
LOG_BLOB_THRESHOLD: float = 500.0
LOG_BLOB_MAX_VOXELS: int = 500
LOG_BLOB_SUBVOXEL_DEFAULT: bool = True


# ---------------------------------------------------------------------
# 2. Walker / pitch tolerance
# ---------------------------------------------------------------------

PITCH_MM: float = 3.5
# Inter-contact pitch tolerance. ~2 sigma at the per-peak position
# error budget (~0.21 mm). Probes confirmed dataset recall holds at
# 295/295 down to 0.2 mm; 0.4 keeps cross-scanner safety margin.
PITCH_TOL_MM: float = 0.4
PERP_TOL_MM: float = 1.5
AX_TOL_MM: float = 0.7
MAX_K_STEPS: int = 20


# ---------------------------------------------------------------------
# 3. Auto-pitch detection (`detect_pitch_from_intracranial_blobs`)
# ---------------------------------------------------------------------

PITCH_AUTO_MIN_MM: float = 2.5
PITCH_AUTO_MAX_MM: float = 6.5  # cover Dixi MM09A51 hybrid at 6.1mm pitch.
PITCH_AUTO_MAX_PEAKS: int = 3
PITCH_AUTO_SECONDARY_FRAC: float = 0.30
PITCH_AUTO_PEAK_EXCLUSION_MM: float = 0.6
PITCH_SNAP_MM: float = 0.3


# ---------------------------------------------------------------------
# 4. Library fallback bounds (used when rosa_core unavailable)
# ---------------------------------------------------------------------

SEEG_VENDORS: tuple[str, ...] = ("Dixi", "PMT", "AdTech")

_BUNDLED_LIBRARY_BOUNDS_FALLBACK: dict = {
    "min_contacts": 5,
    "max_contacts": 18,
    "min_contact_span_mm": 14.0,
    "max_contact_span_mm": 78.5,
    "max_within_electrode_pitch_mm": 13.0,
    "regular_pitches_mm": (3.5, 3.9, 3.97, 4.43, 4.8, 6.1),
}


# ---------------------------------------------------------------------
# 5. Walker line scoring (span / gap / blob count slack)
# ---------------------------------------------------------------------

WALKER_SPAN_UNDER_SLACK_MM: float = 2.0   # walker can miss endpoint contacts
WALKER_SPAN_OVER_SLACK_MM: float = 11.5   # walker can chain a few bolt voxels
WALKER_GAP_SLACK_MM: float = 9.0          # 2 consecutive missed contacts at 4.5mm pitch
MIN_BLOBS_PER_LINE: int = 3
MIN_BLOBS_POST_ARBITRATION: int = 4


# ---------------------------------------------------------------------
# 6. Stage-1 dedup
# ---------------------------------------------------------------------

STAGE1_DEDUP_ANGLE_DEG: float = 3.0
STAGE1_DEDUP_PERP_MM: float = 2.0
STAGE1_DEDUP_OVERLAP_FRAC: float = 0.3


# ---------------------------------------------------------------------
# 7. Deep-tip floor
# ---------------------------------------------------------------------

DEEP_TIP_MIN_MM: float = 30.0          # strict floor for long lines
DEEP_TIP_MIN_SHORT_MM: float = 15.0    # short-line relaxation
DEEP_TIP_SHORT_MAX_AVG_PITCH_MM: float = 7.0


# ---------------------------------------------------------------------
# 8. Bolt anchoring + post-anchor dedup
# ---------------------------------------------------------------------

BOLT_PROTRUSION_MIN_MM: float = 16.0      # short PMT bolts protrude ~12mm
ANCHOR_TOTAL_OVERSHOOT_MM: float = 61.5   # long-bolt + thin-wire-PMT slack

POST_ANCHOR_DEDUP_PERP_MM: float = 3.0
POST_ANCHOR_DEDUP_ANG_DEG: float = 8.0    # 5° was too tight: auto-pitch


# ---------------------------------------------------------------------
# 9. Synth bolt fallback (`_axis_to_skull_synth`)
# ---------------------------------------------------------------------

AXIS_SKULL_SYNTH_STEP_MM: float = 0.5
AXIS_SKULL_SYNTH_MAX_OUTWARD_MM: float = 80.0
AXIS_SKULL_SYNTH_BOLT_PROTRUDE_MM: float = 15.0


# ---------------------------------------------------------------------
# 10. Frangi gate (post-anchor)
# ---------------------------------------------------------------------

FRANGI_LINE_MIN_MEDIAN: float = 30.0


# ---------------------------------------------------------------------
# 11. Axis-deep-end refinement (`_refine_deep_end_via_axis_log`)
# ---------------------------------------------------------------------

AXIS_REFINE_STEP_MM: float = 0.5
AXIS_REFINE_MAX_MM: float = 40.0
AXIS_REFINE_MIN_ABS: float = LOG_BLOB_THRESHOLD
AXIS_REFINE_MISS_MM: float = 3.0
DEEP_END_MARGIN_PAST_LAST_CONTACT_MM: float = 5.0


# ---------------------------------------------------------------------
# 12. Crossing-tip retreat
# ---------------------------------------------------------------------

CROSSING_TIP_CLEARANCE_MM: float = 2.0
CROSSING_RETREAT_STEP_MM: float = 0.5


# ---------------------------------------------------------------------
# 13. Confidence score (`_compute_trajectory_score`)
# ---------------------------------------------------------------------

SCORE_WEIGHTS: dict[str, float] = {
    "amp":               1.0,
    "n_inliers":         1.0,
    "frangi":            1.0,
    "pitch":             1.0,
    "span":              1.0,
    "length":            1.0,
    "depth":             1.0,         # was 0.5; SEEG is a depth-electrode
                                       # technique by definition, so depth is
                                       # a load-bearing signal, not a soft
                                       # prior.
    "intracranial":      0.5,
    "bolt":              1.0,
    "metal_continuity":  2.0,  # frac_strong-along-axis: real shanks have
                                # discrete contact-saturating peaks all along
                                # the line (matched p10=0.27, p50=0.65);
                                # cross-shank chains and synth-extended FPs
                                # have frac near 0 (orphan p50=0.01). Weight
                                # 2.0 pushes zero-saturation orphans into LOW
                                # band.
}
SCORE_METAL_CONTINUITY_SAT: float = 0.10
SCORE_HIGH_THRESHOLD: float = 0.80
SCORE_MEDIUM_THRESHOLD: float = 0.50
SCORE_PITCH_TOL_MM: float = 0.25
SCORE_SPAN_SHOULDER_MM: float = 6.0
SCORE_LENGTH_SHOULDER_MM: float = 10.0
SCORE_AMP_SAT: float = 5000.0
SCORE_N_INLIERS_SLOPE: float = 10.0
SCORE_N_INLIERS_OVER_SLACK: float = 12.0
SCORE_DEPTH_SAT_MM: float = 30.0
SCORE_INTRACRANIAL_SAT_MM: float = 10.0
SCORE_BOLT_VALUES: dict[str, float] = {
    "metal":         1.0,   # unified bolt CC (replaces "log" + "hu_rescue")
    "metal_cc":      0.7,   # wire-class: bolt CC extends into brain as
                             # continuous metal; walker found no
                             # contact-pitch line (saturated/merged
                             # contacts), but the CC itself defines the
                             # shank axis. Lower than "metal" because no
                             # contact-pitch validation.
    "synthesized":   0.4,   # axis-to-skull synth fallback
    "none":          0.1,   # no anchor and synth couldn't reach hull
}


# ---------------------------------------------------------------------
# 14. Wire-class extension
# ---------------------------------------------------------------------

WIRE_CLASS_MIN_DEPTH_MM: float = 15.0
WIRE_CLASS_MIN_SPAN_MM: float = 15.0
WIRE_CLASS_MIN_VOXELS: int = 50
WIRE_CLASS_MIN_ELONGATION: float = 0.65


__all__ = [
    "LOG_BLOB_THRESHOLD", "LOG_BLOB_MAX_VOXELS", "LOG_BLOB_SUBVOXEL_DEFAULT",
    "PITCH_MM", "PITCH_TOL_MM", "PERP_TOL_MM", "AX_TOL_MM", "MAX_K_STEPS",
    "PITCH_AUTO_MIN_MM", "PITCH_AUTO_MAX_MM", "PITCH_AUTO_MAX_PEAKS",
    "PITCH_AUTO_SECONDARY_FRAC", "PITCH_AUTO_PEAK_EXCLUSION_MM", "PITCH_SNAP_MM",
    "SEEG_VENDORS", "_BUNDLED_LIBRARY_BOUNDS_FALLBACK",
    "WALKER_SPAN_UNDER_SLACK_MM", "WALKER_SPAN_OVER_SLACK_MM",
    "WALKER_GAP_SLACK_MM", "MIN_BLOBS_PER_LINE", "MIN_BLOBS_POST_ARBITRATION",
    "STAGE1_DEDUP_ANGLE_DEG", "STAGE1_DEDUP_PERP_MM", "STAGE1_DEDUP_OVERLAP_FRAC",
    "DEEP_TIP_MIN_MM", "DEEP_TIP_MIN_SHORT_MM", "DEEP_TIP_SHORT_MAX_AVG_PITCH_MM",
    "BOLT_PROTRUSION_MIN_MM", "ANCHOR_TOTAL_OVERSHOOT_MM",
    "POST_ANCHOR_DEDUP_PERP_MM", "POST_ANCHOR_DEDUP_ANG_DEG",
    "AXIS_SKULL_SYNTH_STEP_MM", "AXIS_SKULL_SYNTH_MAX_OUTWARD_MM",
    "AXIS_SKULL_SYNTH_BOLT_PROTRUDE_MM",
    "FRANGI_LINE_MIN_MEDIAN",
    "AXIS_REFINE_STEP_MM", "AXIS_REFINE_MAX_MM", "AXIS_REFINE_MIN_ABS",
    "AXIS_REFINE_MISS_MM", "DEEP_END_MARGIN_PAST_LAST_CONTACT_MM",
    "CROSSING_TIP_CLEARANCE_MM", "CROSSING_RETREAT_STEP_MM",
    "SCORE_WEIGHTS", "SCORE_METAL_CONTINUITY_SAT",
    "SCORE_HIGH_THRESHOLD", "SCORE_MEDIUM_THRESHOLD", "SCORE_PITCH_TOL_MM",
    "SCORE_SPAN_SHOULDER_MM", "SCORE_LENGTH_SHOULDER_MM",
    "SCORE_AMP_SAT", "SCORE_N_INLIERS_SLOPE", "SCORE_N_INLIERS_OVER_SLACK",
    "SCORE_DEPTH_SAT_MM", "SCORE_INTRACRANIAL_SAT_MM", "SCORE_BOLT_VALUES",
    "WIRE_CLASS_MIN_DEPTH_MM", "WIRE_CLASS_MIN_SPAN_MM",
    "WIRE_CLASS_MIN_VOXELS", "WIRE_CLASS_MIN_ELONGATION",
]
