# contact_pitch_v1 Handoff

Last updated: 2026-04-18.

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

### 2. Stage 1 — blob-pitch (Dixi 3.5 mm prior)

1. **Regional minima** on LoG σ=1 (SITK `GrayscaleErode` radius 2),
   threshold `LoG ≤ −300`. One marker per contact, even when the skull
   metal is connected through a mega-CC.
2. Enumerate blob pairs at distances `k · 3.5 mm ± 0.5 mm` for
   `k ∈ {1, 2, 3}`.
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

## Slicer integration

- Pipeline ID: `contact_pitch_v1`. Class
  `ContactPitchV1Pipeline` in
  `CommonLib/shank_engine/pipelines/contact_pitch_v1.py`. Detector
  logic lives in
  `PostopCTLocalization/postop_ct_localization/contact_pitch_v1_fit.py`.
- Widget tab: **Contact Pitch v1** (next to Deep Core v2). Single
  "Run" button — uses the currently-selected CT, no config needed.
- After a run, six feature volumes are added to the scene
  (`<CT>_ContactPitch_*`) with percentile-based window/level so
  signed-float volumes (LoG, Frangi, head-distance) display with
  useful contrast.
- The visualizer prefers `skull_entry_ras` from the trajectory dict
  (set by the bolt anchor as the bone→brain transition) over the
  legacy annulus-gradient estimate, so the red marker now sits at the
  dura on every line instead of jumping between air→skin and
  bolt→bone.

## Key files

| File | Role |
| --- | --- |
| `CommonLib/shank_engine/pipelines/contact_pitch_v1.py` | Pipeline wrapper, registers CT loader, stashes feature arrays for the widget |
| `CommonLib/shank_engine/bootstrap.py` | Registers the pipeline |
| `PostopCTLocalization/postop_ct_localization/contact_pitch_v1_fit.py` | Algorithm: blob extraction, walker, dedup, arbitration, extension, second-pass, bolt anchor, length/air filters, orient, orchestration |
| `PostopCTLocalization/postop_ct_localization/deep_core_widget.py` | "Contact Pitch v1" tab + feature volume registration |
| `PostopCTLocalization/postop_ct_localization/deep_core_visualization.py` | Honors `skull_entry_ras` for the red marker |
| `tests/deep_core/test_pipeline_dataset_contact_pitch_v1.py` | Regression test |

## Known issues / next steps

1. **Library-aware pitch walker.** Stage 1 hardcodes Dixi 3.5 mm.
   PMT-16C runs at 4.43 mm and DIXI-BM/CM have 9–13 mm insulation
   gaps between groups. A UI dropdown for electrode family, plus a
   library-driven walker, would generalize beyond Dixi.
2. **Skip stage 2 when stage 1 covers everything.** Stage 2 only
   exists for shanks like T2 RSAN that lack visible contacts. On
   subjects whose shanks all show contacts (T1, T3, most), Frangi
   adds no recall and only generates FPs that we then filter out.
   A "stage-2 needed?" check would save ~1–2 s and a class of FPs.
3. **`shallow_err` always reports the bolt-tip distance from the
   GT entry**. The GT entry is the bone→brain interface; my start
   is the bolt outer end. Tests use `skull_entry_ras` for the
   shallow-side comparison instead — that's the right point to
   align against the bone→brain GT.

## Regression baseline

- T22: ≥ 8 / 9 matched, ≤ 10 FP (currently 8 / 9, 0 FP).
- T2: ≥ 12 / 12 matched, ≤ 10 FP (currently 12 / 12, 0 FP).

Run:

```bash
cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper
/Users/ammar/miniforge3/envs/shankdetect/bin/python3 \
    -m unittest tests.deep_core.test_pipeline_dataset_contact_pitch_v1
```
