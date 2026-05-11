# Pipeline Constants Reference

Last updated: 2026-05-11

The detection + placement pipelines have two `constants.py` files, one
per package. **They are the single source of truth for tunable knobs**;
this doc mirrors them so you can read the rationale without diving into
code. If a number here disagrees with the code, the code wins — please
update this doc in the same change.

- Detector knobs:
  [`CommonLib/rosa_detect/candidate_seeds/constants.py`](../CommonLib/rosa_detect/candidate_seeds/constants.py)
- Placer knobs:
  [`CommonLib/rosa_core/contact_placement/constants.py`](../CommonLib/rosa_core/contact_placement/constants.py)

A few additional knobs that have grown to module-level status live in
`rosa_core.matched_filter` (`SIGMA_CONTACT_MM_DEFAULT`,
`max_tip_short_mm`, `metal_extent_threshold_frac`); they are documented
inline in that module rather than here because they are matched-filter
internals exposed for advanced callers.

> **Cross-scanner safety margin.** Most thresholds carry a 2× cross-scanner
> safety margin over their per-subject calibration, so they should hold
> across vendors / pulse sequences without retuning. If you are tuning
> for a single subject and getting better numbers, check the dataset-wide
> regression (`tests/deep_core/test_pipeline_dataset_contact_pitch_v1.py
> ::test_dataset_full`) before landing the change. **Don't tune to single
> subjects** — see the `feedback_concept_over_threshold.md` memory.

> **Score-band policy.** The placer collapses a continuous compound score
> into three bands (`high` / `medium` / `low`). `high` is reserved for
> *pitch + REAL bolt* (`bolt_source == "metal"`). Synthesized bolts,
> wire-class fallback, and no-bolt acceptance all cap at `medium`. This is
> categorical, not gradient — never let weak-evidence cases earn `high`
> (see `feedback_score_band_policy.md`).

---

## A) Detector — `rosa_detect.candidate_seeds.constants`

The detector runs in stages; each section maps to one stage.

### A.1 Blob extraction (LoG)

Sub-voxel local-min picker over a 3×3×3 box on the LoG σ=1 volume.

| Constant | Value | Why |
|---|---|---|
| `LOG_BLOB_THRESHOLD` | 500.0 | \|LoG\| floor to accept a contact-sized local minimum. ~50 % of typical per-contact LoG (~1062), 2× cross-scanner safety margin |
| `LOG_BLOB_MAX_VOXELS` | 500 | upper bound on blob CC voxel count — anything larger is a metal smear, not a contact |
| `LOG_BLOB_SUBVOXEL_DEFAULT` | True | quadratic sub-voxel refinement on the LoG volume; turn off for parity probes only |

### A.2 Walker / pitch tolerance

The pitch-strict walker chains contact-sized blobs into a candidate
shank line.

| Constant | Value | Why |
|---|---|---|
| `PITCH_MM` | 3.5 | nominal contact-to-contact pitch (DIXI / PMT canonical) |
| `PITCH_TOL_MM` | 0.4 | inter-pitch tolerance, ~2σ at the per-peak position error budget; 295/295 holds down to 0.2 mm |
| `PERP_TOL_MM` | 1.5 | perpendicular tolerance to the chained axis |
| `AX_TOL_MM` | 0.7 | along-axis tolerance |
| `MAX_K_STEPS` | 20 | walker step cap (longest library electrode = 18 contacts + slack) |

### A.3 Auto-pitch detection

When the walker fails on the canonical 3.5 mm pitch, a 1-D autocorrelation
sweep over `[PITCH_AUTO_MIN_MM, PITCH_AUTO_MAX_MM]` finds the dominant
pitch (covers DIXI MM09A51 hybrid at 6.1 mm).

| Constant | Value |
|---|---|
| `PITCH_AUTO_MIN_MM` | 2.5 |
| `PITCH_AUTO_MAX_MM` | 6.5 |
| `PITCH_AUTO_MAX_PEAKS` | 3 |
| `PITCH_AUTO_SECONDARY_FRAC` | 0.30 |
| `PITCH_AUTO_PEAK_EXCLUSION_MM` | 0.6 |
| `PITCH_SNAP_MM` | 0.3 |

### A.4 Library fallback bounds

Used when `rosa_core` library load fails. Identical shape to the live
library; never the source of truth in production.

| Constant | Value |
|---|---|
| `SEEG_VENDORS` | `("Dixi", "PMT", "AdTech")` |
| `_BUNDLED_LIBRARY_BOUNDS_FALLBACK` | min/max contacts, span bounds, regular pitches — see code |

### A.5 Walker line scoring

Slack thresholds the walker uses to accept/reject a chained line.

| Constant | Value | Why |
|---|---|---|
| `WALKER_SPAN_UNDER_SLACK_MM` | 2.0 | walker can miss endpoint contacts |
| `WALKER_SPAN_OVER_SLACK_MM` | 11.5 | walker can chain a few bolt voxels |
| `WALKER_GAP_SLACK_MM` | 9.0 | 2 consecutive missed contacts at 4.5 mm pitch |
| `MIN_BLOBS_PER_LINE` | 3 | hard floor for a candidate line |
| `MIN_BLOBS_POST_ARBITRATION` | 4 | post-arbitration line must reach ≥4 contacts |

### A.6 Stage-1 dedup

Lines that merge at this geometry are treated as the same shank.

| Constant | Value |
|---|---|
| `STAGE1_DEDUP_ANGLE_DEG` | 3.0 |
| `STAGE1_DEDUP_PERP_MM` | 2.0 |
| `STAGE1_DEDUP_OVERLAP_FRAC` | 0.3 |

### A.7 Deep-tip floor

Lines shorter than these floors are dropped (cross-shank chains masquerading
as real shanks).

| Constant | Value | Why |
|---|---|---|
| `DEEP_TIP_MIN_MM` | 30.0 | strict floor for long lines |
| `DEEP_TIP_MIN_SHORT_MM` | 15.0 | short-line relaxation |
| `DEEP_TIP_SHORT_MAX_AVG_PITCH_MM` | 7.0 | short-line floor only applies when pitch is below this |

### A.8 Bolt anchoring + post-anchor dedup

Anchor the line to the metal bolt CC (when present); then de-dup again.

| Constant | Value | Why |
|---|---|---|
| `BOLT_PROTRUSION_MIN_MM` | 16.0 | short PMT bolts protrude ~12 mm |
| `ANCHOR_TOTAL_OVERSHOOT_MM` | 61.5 | long-bolt + thin-wire-PMT slack |
| `POST_ANCHOR_DEDUP_PERP_MM` | 3.0 | |
| `POST_ANCHOR_DEDUP_ANG_DEG` | 8.0 | 5° was too tight: auto-pitch breaks |

### A.9 Synth bolt fallback

When no bolt CC is found, walk outward from the trajectory until we hit
the skull mask; treat the hit as a synthesized bolt.

| Constant | Value |
|---|---|
| `AXIS_SKULL_SYNTH_STEP_MM` | 0.5 |
| `AXIS_SKULL_SYNTH_MAX_OUTWARD_MM` | 80.0 |
| `AXIS_SKULL_SYNTH_BOLT_PROTRUDE_MM` | 15.0 |

### A.10 Frangi gate (post-anchor)

After anchoring, re-evaluate Frangi tubeness along the extended axis.

| Constant | Value |
|---|---|
| `FRANGI_LINE_MIN_MEDIAN` | 30.0 |

### A.11 Axis-deep-end refinement

Walk the deep tip outward along the LoG profile until the signal drops.

| Constant | Value |
|---|---|
| `AXIS_REFINE_STEP_MM` | 0.5 |
| `AXIS_REFINE_MAX_MM` | 40.0 |
| `AXIS_REFINE_MIN_ABS` | `LOG_BLOB_THRESHOLD` (500) |
| `AXIS_REFINE_MISS_MM` | 3.0 |
| `DEEP_END_MARGIN_PAST_LAST_CONTACT_MM` | 5.0 |

### A.12 Crossing-tip retreat

Two shanks whose deep tips are within this clearance get retreated to
keep the closest contacts apart.

| Constant | Value |
|---|---|
| `CROSSING_TIP_CLEARANCE_MM` | 2.0 |
| `CROSSING_RETREAT_STEP_MM` | 0.5 |

### A.13 Confidence score

Composite score → band (`high` / `medium` / `low`). All components are
clipped/normalized to [0, 1] before the weighted sum.

`SCORE_WEIGHTS`:

| Component | Weight | What it measures |
|---|---|---|
| `amp` | 1.0 | mean LoG amplitude across walker inliers |
| `n_inliers` | 1.0 | walker inlier count, normalized to expected library count |
| `frangi` | 1.0 | median Frangi tubeness along the line |
| `pitch` | 1.0 | median pitch deviation from library's nearest model |
| `span` | 1.0 | line span vs library span shoulder |
| `length` | 1.0 | total line length vs shoulder |
| `depth` | 1.0 | how deep the deepest contact reaches (was 0.5; SEEG is by definition a depth technique) |
| `intracranial` | 0.5 | fraction of inliers inside the brain hull |
| `bolt` | 1.0 | bolt-source category lookup |
| `metal_continuity` | 2.0 | `frac_strong` of the contact-saturating signal along the FULL axis. Real shanks p10=0.27 / p50=0.65; cross-shank chains p50=0.01 |

Saturation / shoulder constants:

| Constant | Value |
|---|---|
| `SCORE_METAL_CONTINUITY_SAT` | 0.10 |
| `SCORE_HIGH_THRESHOLD` | 0.80 |
| `SCORE_MEDIUM_THRESHOLD` | 0.50 |
| `SCORE_PITCH_TOL_MM` | 0.25 |
| `SCORE_SPAN_SHOULDER_MM` | 6.0 |
| `SCORE_LENGTH_SHOULDER_MM` | 10.0 |
| `SCORE_AMP_SAT` | 5000.0 |
| `SCORE_N_INLIERS_SLOPE` | 10.0 |
| `SCORE_N_INLIERS_OVER_SLACK` | 12.0 |
| `SCORE_DEPTH_SAT_MM` | 30.0 |
| `SCORE_INTRACRANIAL_SAT_MM` | 10.0 |

`SCORE_BOLT_VALUES` (categorical band cap, **not gradient — see policy box above**):

| `bolt_source` | Score | Meaning |
|---|---|---|
| `metal` | 1.0 | unified bolt CC found |
| `metal_cc` | 0.7 | wire-class: bolt CC extends into brain as continuous metal; walker found no contact-pitch line |
| `synthesized` | 0.4 | axis-to-skull synth fallback |
| `none` | 0.1 | no anchor and synth couldn't reach hull |

### A.14 Wire-class extension

Special path for shanks where contacts are saturated/merged into one CC.

| Constant | Value |
|---|---|
| `WIRE_CLASS_MIN_DEPTH_MM` | 15.0 |
| `WIRE_CLASS_MIN_SPAN_MM` | 15.0 |
| `WIRE_CLASS_MIN_VOXELS` | 50 |
| `WIRE_CLASS_MIN_ELONGATION` | 0.65 |

### A.15 Stage-1 deep-end extension walker

After the initial walker pass, extend the deep end to absorb un-claimed
contact-sized blobs the pitch-strict walker missed.

| Constant | Value | Why |
|---|---|---|
| `EXTEND_MAX_GAP_MM` | 14.0 | up to two missed contacts at the widest library pitch (6.1 mm) plus slack |
| `EXTEND_PERP_TOL_MM` | 2.5 | Walker `PERP_TOL_MM` is 1.5; +1 mm here lets the deep end pick up one drifted-tip contact |
| `EXTEND_MAX_EXTRA` | 20 | absorbed-blob cap; well past the largest library electrode (18) |
| `EXTEND_MAX_OUTER_ITER` | 4 | refit-iteration cap; converges in 1-2 passes |

### A.16 Axis-profile peak signature refinement

After bolt anchor, re-derive (n_inliers, median_pitch, contact_span)
from a 1-D LoG profile sampled along the FIT axis with sub-voxel steps.
Recovers true pitch on anisotropic CTs.

| Constant | Value |
|---|---|
| `AXIS_PEAK_STEP_MM` | 0.25 |
| `AXIS_PEAK_DISK_RADIUS_MM` | 2.0 |
| `AXIS_PEAK_N_RADII` | 4 |
| `AXIS_PEAK_N_ANGLES` | 8 |
| `AXIS_PEAK_MIN_AMPLITUDE` | 200.0 |
| `AXIS_PEAK_MIN_SEPARATION_MM` | 2.0 |
| `AXIS_PEAK_MIN_PEAKS_REQUIRED` | 4 |
| `AXIS_PEAK_SHALLOW_PAD_MM` | 1.5 |
| `AXIS_PEAK_DEEP_PAD_MM` | 3.0 |

### A.17 Along-axis sampling step

Common 0.5 mm step shared by `frangi_along_line_stats`,
`frac_strong_metal_along_line`, and `refine_deep_end_via_axis_log`. Half
of the canonical 1 mm voxel (Nyquist for contact-sized features).

| Constant | Value |
|---|---|
| `ALONG_AXIS_STEP_MM` | 0.5 |

### A.18 Model-suggestion gates

| Constant | Value | Why |
|---|---|---|
| `MODEL_SUGGEST_MIN_INTRACRANIAL_MM` | 5.0 | shorter than the smallest library electrode's contact span; nothing shorter is plausibly a real shank |

---

## B) Placer — `rosa_core.contact_placement.constants`

The placer runs in lettered stages; each section is one stage.

### B.1 Stage A — anchor (degeneracy check)

| Constant | Value | Why |
|---|---|---|
| `DEGENERATE_CONTACT_ZONE_MM` | 5.0 | `centerline_total − bolt_end_arc < 5 mm` means the contact zone is too short to host any contacts → fall through to bolt-less. Saved AMC137 cropped-bolt shanks |

### B.2 Stage B — refine (LoG-centroid centerline snap)

| Constant | Value |
|---|---|
| `SNAP_RADIUS_MM` | 2.0 |
| `SNAP_LOG_THRESHOLD` | 500.0 (matches `LOG_BLOB_THRESHOLD`) |
| `SNAP_STEP_MM` | 0.5 |
| `SNAP_SMOOTH_WINDOW` | 5 |

### B.3 Stage C — sample (walker disk-stat)

| Constant | Value | Why |
|---|---|---|
| `WALK_STEP_MM` | 0.25 | walker arc resolution |
| `WALK_DISK_RADIUS_MM` | 1.0 | contact half-diameter |
| `WALK_FIRST_CONTACT_MIN_MM` | 1.0 | bolt-to-first-contact gap (matched-filter zone constraint) |
| `WALK_TIP_PAD_MM` | 3.0 | tip slack (only applied when `bolt_source == "metal"`); lets matched filter evaluate model tip just past the polynomial endpoint |
| `WALK_HU_MIN` | 1000.0 | legacy `sample_disk_along_polyline` HU floor (staged walker uses `WALK_AGGREGATOR` instead) |
| `WALK_N_RADII` × `WALK_N_ANGLES` | 3 × 12 | 1 + 36 = 37 samples per disk; more stable correlation than the legacy 1 + 16 |
| `WALK_AGGREGATOR` | `"p90"` | suppresses single-voxel HU spikes; raises matched-filter `med_corr` ~0.03 vs. `"max"` |
| `LOG_TOTAL_THRESHOLD` | 100.0 | legacy `sample_disk_along_polyline` LoG mode threshold |

### B.4 Stage D — pick (matched filter + extent-aware re-pick)

| Constant | Value | Why |
|---|---|---|
| `PICK_OVERRIDE_MARGIN` | 0.05 | when matched-filter raw `top1 − top2` exceeds this, trust the matched-filter pick — only re-rank ties via the extent-aware denominator correction |

### B.5 Stage F.1 — `score_cc_overlap`

Bolt-CC overlap term. Real bolts overlap the seed entry zone; cross-shank chains don't.

| Constant | Value | Why |
|---|---|---|
| `CC_OVERLAP_PERP_SCALE_MM` | 5.0 | linear decay scale for lateral CC-centroid distance |
| `CC_OVERLAP_MAX_PERP_MM` | 8.0 | beyond this, treat as no-match (different shank's bolt) |
| `CC_OVERLAP_MAX_ARC_PAST_BOLT_MM` | 10.0 | CC centroid must project within bolt_end + 10 mm along the centerline (not the seed line; the centerline curves) |

### B.6 Stage F.2 — `score_compound`

Composite weights summing to ~1.0 minus the bolt-only-fake penalty
applied post-sum.

`COMPOUND_WEIGHTS`:

| Component | Weight | What it measures |
|---|---|---|
| `corr` | 0.20 | matched-filter NCC (clipped [0, 1]) |
| `fft` | 0.20 | per-segment pitch FFT power frac |
| `tube` | 0.15 | tube-likeness (top-decile HU within 1 mm of centerline) |
| `margin` | 0.10 | top1 − top2 (normalized at 0.15) |
| `walker` | 0.10 | 1.0 if `bolt_source == "metal"` else 0 |
| `cc_overlap` | 0.15 | global bolt CC near seed start (cropped-bolt-aware) |
| `seeder` | 0.10 | v1 confidence label → `{high: 1, medium: 0.6, low: 0.3}` |

`COMPOUND_BANDS`: `{"high": 0.70, "medium": 0.45}` (3-tier band assignment).

`SEEDER_LABEL_TO_SCORE`: `{"high": 1.0, "medium": 0.6, "low": 0.3, "": 0.5}`.

`BOLT_ONLY_PENALTY_THRESHOLD` (0.5) and `BOLT_ONLY_PENALTY_MAX` (0.20):
penalize shanks with high `bz_frac` (bolt-zone fraction) but weak FFT.
Real shanks like AMC91 10_stg can have `bz_frac` up to ~0.7 because
their FFT compensates; AMC91 ei=14/15 sit at `bz_frac ~0.7` with weak
FFT and get penalized.

### B.7 Validators (opt-in unseeded-mode filters)

These only fire in unseeded mode (`place_seeg(..., seeds=None)`); seeded
runs trust the user's input.

| Constant | Value | Why |
|---|---|---|
| `MIN_CORR_FOR_REAL_SHANK` | 0.35 | matched-filter corr below this is too weak to trust as a real shank (calibrated 2026-05-06 on 6-subject dataset) |
| `MIN_SLOT_HU_MEAN` | 1500.0 | per-slot HU floor; real-shank slots 1500-3000+, bone/cross-shank chains 900-1500 |
| `MAX_SLOT_CC_VOLUME_P90_MM3` | 150.0 | per-slot CC volume cap (90th percentile across slots, mm³). Real PMT/DIXI contacts ≤140 mm³; bone-spike chains ≥150 mm³. Calibrated 2026-05-08 |
| `CC_HU_THRESHOLD` | 1500.0 | CC measurement HU floor |
| `CC_ROI_HALF_MM` | 5.0 | CC measurement ROI half-extent |

---

## C) Cross-volume matcher — `rosa_core.cross_volume_match`

The line-RANSAC matcher (used by `rosa-agent match-ros` and callable as
`rosa_core.cross_volume_match`) has only call-time parameters, no
module-level constants. Defaults validated on s57.ros + T24 CT
(16/17 ROS named, all matched pairs ≤ 11° axis-angle and ≤ 6 mm perp):

| Parameter | Default | Why |
|---|---|---|
| `angle_tol_deg` | 15.0 | axis-angle tolerance for RANSAC + greedy match |
| `ransac_perp_tol_mm` | 8.0 | perp line-to-line tolerance during RANSAC inlier counting |
| `match_perp_tol_mm` | 12.0 | perp tolerance for greedy ROS↔det assignment (slightly looser than RANSAC; refined transform is generally good) |
| `n_iter` | 2000 | RANSAC iteration budget |

---

## How to retune safely

1. **Run the regression first.** `tests/deep_core/
   test_pipeline_dataset_contact_pitch_v1.py::test_dataset_full` is the
   net (~70 s, 22 subjects, asserts recall + orphan budget). Don't relax
   its asserts to make a tuning change pass.
2. **Read the inline calibration prose.** Each constant has a one-line
   "why this number"; longer rationale lives in the `memory/` folder
   files referenced in the comments.
3. **Tune the concept, not the value.** If a single subject fails,
   look for the missing concept (`feedback_concept_over_threshold.md`).
   Per-subject thresholds drift; principled fixes hold.
4. **Land with a memory entry.** When a tuning change is principled and
   validated, add a memory note explaining the change so future Claude
   sessions don't undo it on the next refactor.
