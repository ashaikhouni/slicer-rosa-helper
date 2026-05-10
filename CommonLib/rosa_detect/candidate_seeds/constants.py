"""Tunable knobs for the v1 SEEG shank detector.

Single source of truth for every tunable in the v1 pipeline. Each
constant has its calibration rationale inline; longer prose lives in
the per-stage memory entries.

Section organization:
  1. Blob extraction (LoG)
  2. Walker / pitch tolerance
  3. Auto-pitch detection
  4. Library fallback bounds
  5. Walker line scoring (span / gap / blob count slack)
  6. Stage-1 dedup
  7. Deep-tip floor
  8. Bolt anchoring + post-anchor dedup
  9. Synth bolt fallback (axis-to-skull)
 10. Frangi gate (post-anchor)
 11. Axis-deep-end refinement
 12. Crossing-tip retreat
 13. Confidence score
 14. Wire-class extension
 15. Stage-1 deep-end extension walker
 16. Axis-profile peak signature refinement
 17. Along-axis sampling step (Frangi / metal-evidence / refine)
 18. Model-suggestion gates

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
# safety margin. Calibration rationale lives in
# `project_contact_pitch_v1_kernel_recalibration_2026-04-27.md`.
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


# ---------------------------------------------------------------------
# 15. Stage-1 deep-end extension walker (`extend_deep_end`)
# ---------------------------------------------------------------------
#
# After the initial walker pass, walk outward from the deepest /
# shallowest inliers to absorb un-claimed contact-sized blobs that the
# pitch-strict walker missed. Bounded so we don't chain across to a
# different shank.

EXTEND_MAX_GAP_MM: float = 14.0     # axis gap beyond which the next
                                     # blob is not the same shank. Up
                                     # to two missed contacts at the
                                     # widest library pitch (6.1 mm)
                                     # plus a slack mm.
EXTEND_PERP_TOL_MM: float = 2.5     # perp tolerance for absorption.
                                     # Walker's PERP_TOL_MM is 1.5 —
                                     # the +1 mm here lets the deep
                                     # end pick up one contact whose
                                     # walker-axis projection is
                                     # imperfect (deep-tip drift).
EXTEND_MAX_EXTRA: int = 20          # cap on absorbed blobs per pass.
                                     # 20 is well past the largest
                                     # library electrode (18 contacts).
EXTEND_MAX_OUTER_ITER: int = 4      # outer-loop refit iterations.
                                     # Converges in 1-2 passes on
                                     # clean cases; 4 is a safety cap.


# ---------------------------------------------------------------------
# 16. Axis-profile peak signature refinement
#     (`refine_signature_via_axis_peaks`)
# ---------------------------------------------------------------------
#
# After bolt anchor, re-derive (n_inliers, median_pitch, contact_span)
# from a 1-D LoG profile sampled along the FIT axis with sub-voxel
# (0.25 mm) steps + trilinear interpolation. Recovers the true contact
# pitch on anisotropic CTs where the walker's NN-spacing is biased
# (S56 case: walker locked to 3.14 mm instead of 3.5 mm).

AXIS_PEAK_STEP_MM: float = 0.25         # sub-voxel sampling step.
AXIS_PEAK_DISK_RADIUS_MM: float = 2.0   # off-axis disk radius;
                                         # contacts saturate to ~2 mm
                                         # on most scanners.
AXIS_PEAK_N_RADII: int = 4
AXIS_PEAK_N_ANGLES: int = 8
AXIS_PEAK_MIN_AMPLITUDE: float = 200.0  # |LoG| floor for a peak.
                                         # Below the contact LoG floor
                                         # (LOG_BLOB_THRESHOLD=500) so
                                         # weak-but-real contacts on
                                         # the tip count.
AXIS_PEAK_MIN_SEPARATION_MM: float = 2.0
AXIS_PEAK_MIN_PEAKS_REQUIRED: int = 4   # below this, the picker
                                         # returns None and the
                                         # caller keeps walker stats.
AXIS_PEAK_SHALLOW_PAD_MM: float = 1.5   # extend sampling past skull
                                         # entry / deep tip in case the
                                         # bolt anchor or deep-end
                                         # refine clipped a contact.
AXIS_PEAK_DEEP_PAD_MM: float = 3.0


# ---------------------------------------------------------------------
# 17. Along-axis sampling step (Frangi / metal-evidence / refine)
# ---------------------------------------------------------------------
#
# Common 0.5 mm step shared by `frangi_along_line_stats`,
# `frac_strong_metal_along_line`, and `refine_deep_end_via_axis_log`.
# 0.5 mm is half the canonical 1 mm voxel (Nyquist for the contact-
# sized features) and matches the AXIS_REFINE_STEP_MM used by the
# deep-end refiner.

ALONG_AXIS_STEP_MM: float = 0.5


# ---------------------------------------------------------------------
# 18. Model-suggestion gates (electrode classifier dispatch)
# ---------------------------------------------------------------------

# Minimum intracranial trajectory length below which the orchestrator
# skips the electrode-model picker. 5 mm is shorter than the smallest
# library electrode's contact span — anything shorter is almost
# certainly not a real shank.
MODEL_SUGGEST_MIN_INTRACRANIAL_MM: float = 5.0


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
    "EXTEND_MAX_GAP_MM", "EXTEND_PERP_TOL_MM", "EXTEND_MAX_EXTRA",
    "EXTEND_MAX_OUTER_ITER",
    "AXIS_PEAK_STEP_MM", "AXIS_PEAK_DISK_RADIUS_MM",
    "AXIS_PEAK_N_RADII", "AXIS_PEAK_N_ANGLES",
    "AXIS_PEAK_MIN_AMPLITUDE", "AXIS_PEAK_MIN_SEPARATION_MM",
    "AXIS_PEAK_MIN_PEAKS_REQUIRED",
    "AXIS_PEAK_SHALLOW_PAD_MM", "AXIS_PEAK_DEEP_PAD_MM",
    "ALONG_AXIS_STEP_MM",
    "MODEL_SUGGEST_MIN_INTRACRANIAL_MM",
]
