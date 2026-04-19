# contact_pitch_v1 Handoff

Last updated: 2026-04-18. Auto-detect snap-to-library-pitch added this
session (T2 auto: 11/12 → 12/12, 0 FP). Previous session (2026-04-19
wall-clock, earlier in local history) added electrode-model
suggestion, intracranial length, deep-end refinement, crossing-tip
retreat, and multi-pitch walker with auto-detect.

A direct (no-bolt-first) SEEG shank detector. Runs entirely from the
postop CT. Replaces the bolt-first `deep_core_v2` for general use; v2
remains in the registry for backward compatibility.

| subject | matched / GT | FP |
| --- | --- | --- |
| T22 (clean) | 9 / 9 | 0 |
| T2 (clipped) | 12 / 12 | 0 |
| T1 (new) | 12 (12 GT) | 0 visible junk |
| T3 (new) | 14 (14 GT) | 0 visible junk |

## Pipeline overview

Three Slicer scalar volumes are registered alongside the CT for
inspection (`<CT>_ContactPitch_LoG_sigma1`, `..._Frangi_sigma1`,
`..._HeadDistance_mm`, `..._IntracranialMask`, `..._HullMask`,
`..._BoltMask`).

### 1. Preprocessing (one pass over the CT)

- **Hull mask**: `HU ≥ −500` (cast to float32 first so any pixel type
  works), close, fill, largest CC. Defines the head outline.
- **Hull signed distance** `dist_arr`: `SignedMaurerDistanceMap`
  (inside positive). Used everywhere downstream that needs depth.
- **Intracranial mask**: `dist_arr ≥ 10 mm`.
- **LoG σ=1** on raw CT (`LaplacianRecursiveGaussian`). Bright metal
  contacts → strong negative LoG. Universal across scanners.
- **Frangi σ=1** (`ObjectnessMeasure`, objectDimension=1). Used only
  by stage 2 (shaft fallback).

### 2. Stage 1 — blob-pitch (multi-pitch, default 3.5 mm Dixi)

The walker runs once per candidate pitch in ``pitches_mm`` and unions
the resulting hypotheses before dedup/arbitration. The pitch set
comes from the UI's **Pitch strategy** combo:

| Strategy | Walker pitches (mm) | Suggestion vendors |
| --- | --- | --- |
| Dixi (default) | 3.5 | Dixi |
| PMT | 3.5 / 3.97 / 4.43 | PMT |
| Mixed Dixi + PMT | 3.5 / 3.97 / 4.43 | Dixi + PMT |
| Auto-detect pitch | mutual-NN peak from intracranial blob cloud | Dixi + PMT + AdTech |

``detect_pitch_from_intracranial_blobs`` computes mutual-nearest-
neighbour distances across the intracranial blob cloud and returns
the centroid of the dominant mode in `[2.5, 6.0] mm`. Empirically
yields `~3.3 mm` on clean Dixi CTs (true 3.5, small partial-volume
low-bias). ``resolve_pitches_for_strategy("auto", ...)`` then snaps
the raw centroid to the nearest library pitch in
``LIBRARY_PITCHES_MM = (3.5, 3.97, 4.43)`` when within
``PITCH_SNAP_MM = 0.3``. This removes the ~0.2 mm low-bias (walker
sees the nominal pitch, not the biased centroid) and recovers band-
edge shanks that were being dropped at `PITCH_TOL_MM = 0.5` — T2 auto
went from 11/12 to 12/12 with 0 FP.

1. **Regional minima** on LoG σ=1 (SITK `GrayscaleErode` radius 2),
   threshold `LoG ≤ −300`. One marker per contact, even when the skull
   metal is connected through a mega-CC.
2. For each pitch in the strategy, enumerate blob pairs at distances
   `k · pitch ± 0.5 mm` for `k ∈ {1, 2, 3}`.
3. For each pair, walk both directions with `pitch_seed ± {0, ±0.1,
   ±0.2}`. Inliers: `perp ≤ 1.5 mm`, `|proj − k·pitch| ≤ 0.7 mm`.
   Accept walks with `≥ 6 blobs` and `span ∈ [15, 90] mm`.
4. **Stray-blob trim** (was wrong-pitch bridging fix): sort inliers by
   axial projection; iteratively drop endpoints where the gap to the
   next inlier exceeds `MAX_INLIER_GAP_MM = 22`. After trim, any
   internal gap above the threshold is a real bridge → reject.
5. **Dedup**: axis angle ≤ 3°, perp center ≤ 2 mm, span overlap ≥ 30 %.
6. **amp_sum gate**: `amp_sum ≥ 6000`.
7. **Ownership arbitration**: any blob claimed by two distinct lines
   is awarded to the closer-fit line; the loser refits on the reduced
   set (with `MIN_BLOBS_POST_ARBITRATION = 5` floor) and may be
   dropped. This is what gives v3 its depth accuracy on shanks that
   crowd against neighbors (T2 X05/X06 dropped 25–28 mm of bridged
   length).
8. **Deep-end walk**: strongest-first, walk outward in steps from each
   line's deepest inlier and snap unclaimed blobs within
   `max_gap_mm = 14`, perp `≤ 2.5 mm`. Refits the axis after each pass
   and re-runs (up to 4 outer iterations) so the line can "snake"
   along a slightly curved or off-axis shank — without this the LCMN
   deep tip was 27.8 mm too shallow.
9. **Second-pass orphan walker**: re-run the pitch walker on blobs
   not claimed by any surviving line. Recovers electrodes whose
   first-pass hypothesis was a bridging line that arbitration killed.
10. **Deep-tip prior**: inliers' `dist_max ≥ 30 mm`. Cuts vessel /
    skull-base FPs whose deepest inlier barely passes the
    intracranial boundary.

### 3. Stage 2 — Frangi shaft fallback

For shanks like T2 RSAN that appear as a continuous dark bar (LoG of
a uniform tube is flat in the middle), CC + PCA on the residual
Frangi cloud after a 3 mm exclusion around stage-1 lines.

- Cloud = `Frangi σ=1 ≥ 30` ∩ intracranial ∩ ¬exclusion.
- Per CC: PCA in mm. Require `30 ≤ n_vox ≤ 20000`,
  `20 mm ≤ span ≤ 85 mm`, `perp_rms ≤ 3 mm`,
  `span / perp_rms ≥ 5`.
- Hull-endpoint + deep-tip priors as in stage 1.

### 4. Bolt anchor (the FP killer)

A bolt is a connected blob of strong LoG minima that touches the
hull surface. Required for every accepted trajectory.

- **Bolt extraction**: `LoG ≤ −300` → CC. Keep CCs with `n_vox ≥ 20`
  and shallowest voxel `head_distance ≤ 2 mm` (touches/pokes through
  the skull).
- **Anchoring**: for each candidate trajectory, project the bolt CC's
  voxels onto the *shank* axis. Accept if ≥ 15 voxels fall in a
  `tube_radius = 3 mm` cylinder within `[-120, +30] mm` along the
  axis AND the most-outward in-tube voxel has
  `head_distance ≤ 15 mm`. This `15 mm` slack tolerates ≈ 5° axis-fit
  error in stage 1 over a 50 mm bolt walk.
- **Endpoints projected onto the shank axis** so the bolt-tip
  (`start_ras`) and skull-entry (`skull_entry_ras`) are colinear with
  the trajectory line.
- **`skull_entry_ras` = deepest in-tube voxel still in the skull/dura
  band** (`head_distance ≤ 10 mm`). Without this clip, big merged CCs
  put the bone→brain marker at the deepest contact instead of the
  bolt base.

### 5. Combine + dedup + filters

- Combined trajectories (stage 1 ∪ stage 2) sorted: stage-1 first
  (pitch-confirmed > geometric), then by length descending.
- Cross-stage dedup (angle ≤ 15°, perp ≤ 8 mm, midpoint inside other
  segment).
- Trajectories oriented **shallow → deep** based on head-distance.
  Re-orientation runs at the start of every extension iteration so
  refits never lose the orientation invariant.
- **Length sanity** `45 ≤ length ≤ 130 mm` post-anchor. Real SEEG
  total length (bolt + shank) sits inside this band; venous-sinus
  Frangi tubes (160 mm+) and short hardware FPs (<45 mm) die here.
- **Air-sinus rejection**: 25 HU samples along
  `skull_entry → end`; reject if >50 % are below −300 HU. Real shanks
  through ventricles top out around 35 % air; sinus tubes run >70 %.

### 6. Axis-directed deep-end refinement

After bolt anchor + dedup, re-sample LoG σ=1 along each trajectory's
axis past `end_ras` in 0.5 mm steps. Push the deep tip out to the
last on-axis position with `LoG ≤ −300`; stop after 3 mm of weak
signal. Fixes cases where the 3-D regional-minima extractor missed
deep contacts because the per-contact wells merged into a single
long bright LoG CC (T2 RAI's 4 deep contacts, for instance — deep
end went from along=58.5 mm back up to 77.0 mm, matching GT within
1.5 mm). Walks strictly on the original axis; curving / off-axis
snap drifted adjacent shanks' midpoints in testing.

### 7. Crossing-tip retreat

Final cleanup that runs once every trajectory's deep end has been
refined. For each trajectory whose tip sits within
`PERP_TOL_MM + 0.5 = 2.0 mm` of another trajectory's segment (using
proper segment-to-point distance, not infinite-line projection),
walk the tip back along the trajectory's own axis until two
conditions are both satisfied:

1. perpendicular clearance from **every** other segment ≥ 2.0 mm, AND
2. on-axis `|LoG| ≥ 300` — the retreated tip sits on a real contact
   peak instead of floating in the gap past the last contact.

Bounded by `MIN_POST_ANCHOR_LEN_MM`; if retreating that far would
shrink the trajectory below the floor, the tip is left alone and
logged. Tuned on T1 where X05 / X08 cross at ~19° near the midline
with 1.1 mm closest approach — X05's tip was overshooting into X08's
contacts by ~11 mm, now retreated to land on X05's last real contact.

### 8. Post-detection electrode suggestion

Every stage-1 and stage-2 trajectory gets a `suggested_model_id`
stamped onto it via `classify_by_count_and_span` →
`suggest_shortest_covering_model`: shortest electrode in the selected
manufacturers whose `total_exploration_length_mm + 10 mm ≥
intracranial_length_mm`. The 10 mm absorbs the dura margin between
`skull_entry_ras` (inside the dura band) and the first real contact.

Suggestion is advisory — the Contacts & Trajectory View module reads
it via `Rosa.BestModelId` on the line node to pre-populate its
"Electrode Model" dropdown, and the user can override.

## Slicer integration

- Pipeline ID: `contact_pitch_v1`. Class
  `ContactPitchV1Pipeline` in
  `CommonLib/shank_engine/pipelines/contact_pitch_v1.py`. Detector
  logic lives in
  `PostopCTLocalization/postop_ct_localization/contact_pitch_v1_fit.py`.
- Widget tab: **Contact Pitch v1** (next to Deep Core v2). "Run
  Contact Pitch v1" button + a **Pitch strategy** combo that controls
  both the walker's candidate pitches and the suggestion vendor
  filter in one setting (see "Stage 1 — blob-pitch" table). Default
  is Dixi.
- Line nodes are rendered **skull_entry → deep tip** (not bolt tip
  → deep tip). The bolt-tip RAS is still kept as ``bolt_tip_ras`` on
  the trajectory dict for any consumer that needs it. This means
  downstream modules such as **Contacts & Trajectory View** compute
  `trajectory_length_mm` as the intracranial length, matching what
  clinicians expect.
- Each trajectory line node carries `Rosa.BestModelId` +
  `Rosa.BestModelScore` attributes populated by the suggestion step.
  The **Contacts & Trajectory View** module's "Electrode Model"
  dropdown reads these to pre-populate its default assignments.
- After a run, six feature volumes are added to the scene
  (`<CT>_ContactPitch_*`) with percentile-based window/level so
  signed-float volumes (LoG, Frangi, head-distance) display with
  useful contrast.
- The visualizer prefers `skull_entry_ras` from the trajectory dict
  (set by the bolt anchor as the bone→brain transition) over the
  legacy annulus-gradient estimate, so the red marker sits at the
  dura on every line.

## Key files

| File | Role |
| --- | --- |
| `CommonLib/shank_engine/pipelines/contact_pitch_v1.py` | Pipeline wrapper, registers CT loader, stashes feature arrays for the widget |
| `CommonLib/shank_engine/bootstrap.py` | Registers the pipeline |
| `PostopCTLocalization/postop_ct_localization/contact_pitch_v1_fit.py` | Algorithm: blob extraction, walker, dedup, arbitration, extension, second-pass, bolt anchor, length/air filters, orient, orchestration |
| `PostopCTLocalization/postop_ct_localization/deep_core_widget.py` | "Contact Pitch v1" tab + feature volume registration |
| `PostopCTLocalization/postop_ct_localization/deep_core_visualization.py` | Honors `skull_entry_ras` for the red marker |
| `tests/deep_core/test_pipeline_dataset_contact_pitch_v1.py` | Regression test |

## Key constants (end of fit module)

| Name | Value | Role |
| --- | --- | --- |
| `MAX_INLIER_GAP_MM` | 22.0 | Walker rejects an inlier chain with any internal gap greater than this. Caps at ~6 missed Dixi contacts + 1 BM/CM insulation jump, prevents cross-shank bridges (T2 X07 RAMC↔LAMC bridge killed). |
| `AXIS_REFINE_STEP_MM` | 0.5 | Step size for the deep-end axis-LoG walk. |
| `AXIS_REFINE_MAX_MM` | 40.0 | Upper bound on how far refinement can extend past the original `end_ras`. |
| `AXIS_REFINE_MIN_ABS` | 300.0 | Same as `LOG_BLOB_THRESHOLD`. Considers "on a contact" when `abs(LoG) ≥ this`. |
| `AXIS_REFINE_MISS_MM` | 3.0 | Consecutive mm of weak LoG signal → stop walking. |
| `CROSSING_TIP_CLEARANCE_MM` | 2.0 | Post-refinement retreat clearance. `PERP_TOL_MM (1.5) + 0.5` safety margin. |
| `CROSSING_RETREAT_STEP_MM` | 0.5 | Retreat step size. |

## Known issues / next steps

1. **Post-retreat re-extension (as literally written: no-op).**
   `_refine_deep_end_via_axis_log` only reads the LoG volume along a
   trajectory's own axis — it never looks at neighbour segments.
   Re-running refinement on a non-retreated trajectory after a
   neighbour retreats produces an identical result because the LoG
   volume is unchanged. The mechanism described in the prior plan
   (neighbour retreat "frees LoG signal") doesn't apply to axis
   refinement; it would only apply to stage-1 **inlier
   re-arbitration** (the priority-4 variant) which is a larger
   change: after retreat, some blobs previously claimed by the
   retreater become orphans and could be offered to the
   non-retreated trajectory via the walker's ownership arbitration.
   Worth revisiting only when a concrete miscall surfaces that
   would be fixed by re-arbitration.
2. **Skip stage 2 when stage 1 covers everything.** Stage 2 only
   exists for shanks like T2 RSAN that lack visible contacts.
   Measured contribution: T22 → 0 stage-2 survivors (stage-2 was
   redundant); T2 → 1 stage-2 survivor (the RSAN shank, required).
   A conservative heuristic — "no unclaimed real-looking bolt" —
   needs cross-subject tuning we can't do with just T22/T2 in the
   regression suite. Revisit when T1/T3 are added to regression so
   the "real-looking bolt" size threshold can be calibrated across
   ≥4 subjects.
3. **`shallow_err` always reports the bolt-tip distance from the
   GT entry**. The GT entry is the bone→brain interface; my start
   is the bolt outer end. Tests use `skull_entry_ras` for the
   shallow-side comparison instead — that's the right point to
   align against the bone→brain GT.

## Regression baseline

- T22 (default Dixi): ≥ 8 / 9 matched, ≤ 10 FP (currently 9 / 9, 0 FP).
- T2 (default Dixi): ≥ 12 / 12 matched, ≤ 10 FP (currently 12 / 12, 0 FP).
- T2 (auto strategy): ≥ 12 / 12 matched, ≤ 10 FP (currently 12 / 12,
  0 FP; verifies the snap-to-library-pitch path).

Run:

```bash
cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper
/Users/ammar/miniforge3/envs/shankdetect/bin/python3 \
    -m unittest tests.deep_core.test_pipeline_dataset_contact_pitch_v1
```
