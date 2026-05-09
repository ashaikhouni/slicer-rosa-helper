"""Tunable knobs for the staged contact-placement pipeline.

Calibrated values from `notebooks/v1_seeds_v2_placement_qc.ipynb` and the
2026-05-09 multi-subject sweep (11 subjects / 133 GT-matched + 33 orphans).
Each constant has a one-line "why this number" comment so future tuners can
judge whether a change is safe; longer rationale lives in
`memory/project_v3_staged_scoring_2026-05-09.md`.

Don't reorder. Callers import individual names from this module; group by
stage so each section is grepable.
"""
from __future__ import annotations


# ---------------------------------------------------------------------
# Stage A — anchor (bolt-end walker degenerate check)
# ---------------------------------------------------------------------

# When centerline_total - bolt_end_arc < this, the contact zone is too
# short to host any contacts → fall through to bolt_less. Saved AMC137
# cropped-bolt shanks (project_autofit_misses_2026-05-06.md).
DEGENERATE_CONTACT_ZONE_MM: float = 5.0


# ---------------------------------------------------------------------
# Stage B — refine (LoG-centroid centerline snap)
# ---------------------------------------------------------------------

# Why LoG (not raw HU) — see contact_placement_v2 docstring + memory
# project_contact_placement_state_2026-05-05.md. 500 matches stage-1
# LOG_BLOB_THRESHOLD: "metal-bright local minimum" calibration constant.
SNAP_RADIUS_MM: float = 2.0
SNAP_LOG_THRESHOLD: float = 500.0
SNAP_STEP_MM: float = 0.5
SNAP_SMOOTH_WINDOW: int = 5


# ---------------------------------------------------------------------
# Stage C — sample (walker disk-stat sampling)
# ---------------------------------------------------------------------

# Walker arc resolution + contact half-diameter.
WALK_STEP_MM: float = 0.25
WALK_DISK_RADIUS_MM: float = 1.0

# Bolt-to-first-contact gap (matched-filter zone constraint).
WALK_FIRST_CONTACT_MIN_MM: float = 1.0

# Tip slack for axis under-reach. Lets the matched filter evaluate model
# tip positions just past the polynomial endpoint. Only applied when
# bolt_source == "metal" (bolt_less seeds use the entire seed as the
# zone — no extension).
WALK_TIP_PAD_MM: float = 3.0

# Disk-sample HU floor (above this = metal-ish). Used by the legacy
# sample_disk_along_polyline; the staged walker uses the
# WALK_AGGREGATOR-defined statistic instead.
WALK_HU_MIN: float = 1000.0

# Walker disk geometry: 1 + n_radii × n_angles samples per disk.
# 1 + 3 × 12 = 37 samples; gives more stable correlation than
# sample_disk_along_polyline's defaults (1 + 2 × 8 = 17).
WALK_N_RADII: int = 3
WALK_N_ANGLES: int = 12

# Disk aggregator statistic. p90 keeps pick accuracy identical to max
# but raises med_corr ~0.03 by suppressing single-voxel HU spikes. Switch
# to "max" for legacy parity, "p75"/"median" for further smoothing.
WALK_AGGREGATOR: str = "p90"

# LoG total-threshold for the legacy sample_disk_along_polyline LoG mode.
# See sample_disk_along_polyline docstring. Unused by the staged walker.
LOG_TOTAL_THRESHOLD: float = 100.0


# ---------------------------------------------------------------------
# Stage D — pick (matched filter + extent-aware re-pick)
# ---------------------------------------------------------------------

# Margin defer threshold for pick_extent_aware: when the matched-filter
# raw corr top1 - top2 > this, trust the matched filter pick — only
# re-rank ties.
PICK_OVERRIDE_MARGIN: float = 0.05


# ---------------------------------------------------------------------
# Stage F.1 — score_cc_overlap
# ---------------------------------------------------------------------

# Lateral CC-centroid distance over which the score decays linearly to 0.
CC_OVERLAP_PERP_SCALE_MM: float = 5.0
# Beyond this lateral distance, treat as no-match (different shank's bolt).
CC_OVERLAP_MAX_PERP_MM: float = 8.0
# CC centroid must project within bolt_end + this along the centerline
# (not the seed line). The centerline curves and seed-line u over-shoots
# for bolts when the trajectory bends (T18/X03 was the diagnostic case).
CC_OVERLAP_MAX_ARC_PAST_BOLT_MM: float = 10.0


# ---------------------------------------------------------------------
# Stage F.2 — score_compound (composite weights + bands)
# ---------------------------------------------------------------------

# Per-component weights summing to ~1.0 minus the bolt_only_penalty
# subtracted post-sum.
COMPOUND_WEIGHTS: dict[str, float] = {
    "corr":       0.20,   # matched-filter NCC (clipped [0, 1])
    "fft":        0.20,   # per-segment pitch FFT power frac
    "tube":       0.15,   # tube-likeness (top-decile HU within 1mm of centerline)
    "margin":     0.10,   # top1 - top2 (normalized at 0.15)
    "walker":     0.10,   # 1.0 if bolt_source == "metal"
    "cc_overlap": 0.15,   # global bolt CC near seed start (cropped-bolt-aware)
    "seeder":     0.10,   # v1 confidence label → {high:1, medium:0.6, low:0.3}
}
# Composite score thresholds for the 3-tier band assignment.
COMPOUND_BANDS: dict[str, float] = {"high": 0.70, "medium": 0.45}
SEEDER_LABEL_TO_SCORE: dict[str, float] = {
    "high":   1.0,
    "medium": 0.6,
    "low":    0.3,
    "":       0.5,
}
# Bolt-only-fake penalty: real shanks like AMC91 10_stg can have bz_frac
# up to ~0.7 but their FFT compensates. Fakes like AMC91 ei=14/15 sit at
# bz_frac~0.7 with weak FFT — penalize iff FFT can't carry them.
BOLT_ONLY_PENALTY_THRESHOLD: float = 0.5
BOLT_ONLY_PENALTY_MAX: float = 0.20


# ---------------------------------------------------------------------
# Validators (opt-in unseeded-mode filters)
# ---------------------------------------------------------------------

# Below this corr, matched-filter pick is too weak to trust as a real
# shank. Calibrated 2026-05-06 on 6-subject dataset.
MIN_CORR_FOR_REAL_SHANK: float = 0.35
# Per-slot HU floor for unseeded validator. Real-shank slot HU 1500-3000+;
# bone/cross-shank FP chains average 900-1500.
MIN_SLOT_HU_MEAN: float = 1500.0
# Per-slot CC volume cap (90th percentile across slots, mm³). Real
# PMT/DIXI contacts ≤140 mm³; bone-spike chains ≥150 mm³. Calibrated
# 2026-05-08, 6-subject dataset (MATCHED max=142.3 / ORPHAN min=166.1).
MAX_SLOT_CC_VOLUME_P90_MM3: float = 150.0
# CC measurement HU floor + ROI half-extent.
CC_HU_THRESHOLD: float = 1500.0
CC_ROI_HALF_MM: float = 5.0


__all__ = [
    "BOLT_ONLY_PENALTY_MAX",
    "BOLT_ONLY_PENALTY_THRESHOLD",
    "CC_HU_THRESHOLD",
    "CC_OVERLAP_MAX_ARC_PAST_BOLT_MM",
    "CC_OVERLAP_MAX_PERP_MM",
    "CC_OVERLAP_PERP_SCALE_MM",
    "CC_ROI_HALF_MM",
    "COMPOUND_BANDS",
    "COMPOUND_WEIGHTS",
    "DEGENERATE_CONTACT_ZONE_MM",
    "LOG_TOTAL_THRESHOLD",
    "MAX_SLOT_CC_VOLUME_P90_MM3",
    "MIN_CORR_FOR_REAL_SHANK",
    "MIN_SLOT_HU_MEAN",
    "PICK_OVERRIDE_MARGIN",
    "SEEDER_LABEL_TO_SCORE",
    "SNAP_LOG_THRESHOLD",
    "SNAP_RADIUS_MM",
    "SNAP_SMOOTH_WINDOW",
    "SNAP_STEP_MM",
    "WALK_AGGREGATOR",
    "WALK_DISK_RADIUS_MM",
    "WALK_FIRST_CONTACT_MIN_MM",
    "WALK_HU_MIN",
    "WALK_N_ANGLES",
    "WALK_N_RADII",
    "WALK_STEP_MM",
    "WALK_TIP_PAD_MM",
]
