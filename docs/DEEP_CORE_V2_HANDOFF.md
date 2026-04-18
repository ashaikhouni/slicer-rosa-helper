# deep_core_v2 Handoff

> **Superseded for general use by `contact_pitch_v1`.** See
> `CONTACT_PITCH_V1_HANDOFF.md`. The two-stage probe documented here
> evolved into the production `contact_pitch_v1` pipeline (with bolt
> anchor, ownership arbitration, deep-end walk, length / air filters).
> `deep_core_v2` (cylinder-RANSAC, bolt-first) remains in the
> registry for backward compatibility.

Last updated: 2026-04-17 (two-stage session).

## Current best detector: `probe_two_stage.py`

Two-stage LoG+Frangi detector. Achieves full recall on both T22 (clean) and
T2 (clipped) in a few seconds — no HU thresholds, no scanner-specific
tuning.

| | T22 (clean) | T2 (clipped) |
| --- | --- | --- |
| recall | **8/8** | **12/12** |
| FP | 21 | 22 |
| runtime | 3.8 s | 9.5 s |

Compare with the previous state:

| method | T22 recall / FP | T2 recall / FP | T2 runtime |
| --- | --- | --- | --- |
| v2 (committed cylinder RANSAC) | 9/9 / low | 0/12 | — |
| v4 probe (Frangi + intracranial + RANSAC) | 8/8 / 27 | 11/12 / 48 (LPRG missed) | 130 s |
| **two-stage (this session)** | **8/8 / 21** | **12/12 / 22** | **9.5 s** |

### Pipeline

1. **Preprocessing**
   - Hull mask (HU ≥ −500, close, fill, largest CC) and
     `intracranial = head_distance ≥ 10 mm` from hull. Same as v4.
   - Frangi σ=1 on raw CT (`ObjectnessMeasure`, objectDimension=1).
   - Laplacian of Gaussian σ=1 on raw CT
     (`LaplacianRecursiveGaussian`).

2. **Stage 1 — blob-pitch (contacts with Dixi 3.5 mm periodicity)**
   - Regional minima on LoG σ=1 with erode radius 2 voxels, then gate
     `LoG ≤ −300`. One marker per contact local minimum (CC approach
     fails here because a mega-CC absorbs the whole skull+electrode
     chain on T2; regional minima don't suffer this).
   - Enumerate blob pairs at distances `k × 3.5 mm ± 0.5 mm` for
     `k ∈ {1, 2, 3}`. Allowing `k=2,3` handles occasional missed contacts.
   - For each pair, walk both directions at local pitch
     `seed_d / round(seed_d / 3.5)`. Multi-pitch search: try
     `pitch_seed ± {0, ±0.1, ±0.2}` mm and keep the walk with most
     inliers. A blob is an inlier if `perp ≤ 1.5 mm` and
     `|proj − k·pitch| ≤ 0.7 mm`.
   - Accept walks with `n_blobs ≥ 6` and `span ∈ [15, 90] mm`.
   - Dedup: axis angle ≤ 3°, perp center distance ≤ 2 mm, span overlap
     ≥ 30 %.
   - FP gate: `amp_sum ≥ 6000` (sum of inlier LoG magnitudes). Real
     shanks' LoG minima are strong; skull/bone spurious lines assemble
     from weak blobs.

3. **Stage 1 exclusion zone**
   - For each accepted stage-1 line, mark a 3 mm-radius tube around it
     in voxel space. Stage 2 will skip these voxels.

4. **Stage 2 — Frangi shaft fallback (pitch-unresolved shanks)**
   - Some shanks (e.g. T2 RSAN) appear on CT as a continuous dark bar
     with no visible contacts — the LoG of a uniform tube is flat in
     the middle, so the contact-comb signature does not exist. Frangi
     tube response, on the other hand, lights up for any locally-tubular
     structure.
   - Cloud = `Frangi σ=1 ≥ 30` ∩ `intracranial` ∩ `¬exclusion_zone`.
     Higher Frangi threshold than stage 1 (10) keeps CCs thin so real
     shafts don't merge with adjacent bone.
   - 3D connected components on the cloud.
   - Per CC: PCA in world-mm. Require
     `30 ≤ n_vox ≤ 20,000`, `20 mm ≤ span ≤ 85 mm`, `perp_rms ≤ 3 mm`,
     and `span / perp_rms ≥ 5`. The `span / perp_rms` geometric aspect
     is robust to CC curvature (unlike λ₁/λ₂, which collapses to ~1 for
     a gently curved tube).
   - Axis = PC1, endpoints = center + [proj_min, proj_max] × PC1.
   - No periodicity check (the whole point of stage 2 is that these
     shanks lack periodicity).

5. **Combine** stage 1 + stage 2 hypotheses. GT matching is greedy 1-to-1
   (angle ≤ 10°, mid-distance ≤ 8 mm).

### How to run

```bash
cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper
/Users/ammar/miniforge3/envs/shankdetect/bin/python3 \
  tests/deep_core/probe_two_stage.py T22   # ~4 s
/Users/ammar/miniforge3/envs/shankdetect/bin/python3 \
  tests/deep_core/probe_two_stage.py T2    # ~10 s
```

## Session story (why two stages)

The path to the current detector, compressed:

1. **Probe A — `probe_bolt_recovery.py`.** Frangi σ=2 + linearity gate
   gives at most 50 % bolt recall with FPs > TPs. Fails because (a) the
   whole skull vault forms a thick non-linear mega-CC that absorbs bolts
   adjacent to it, and (b) the "successful" matches are really shafts,
   not bolts. σ=2 + linearity is not a bolt detector.

2. **Probe B — `probe_contact_recovery.py`.** Frangi σ=1 + 5 mm median-HU
   ∈ [−50, 80] soft-tissue filter. Works on T22 (78 % in-tube pass rate);
   **collapses on T2** (0.3 % in-tube). Not an HU saturation issue —
   contact halos on T2 fill the 5 mm sphere with HU ~500 (p99 clipped at
   3071 is irrelevant). Filter is scanner-dependent.

3. **Probe C — `probe_periodicity_gate.py`.** Windowed FFT with
   SNR ≥ 2 in the 2.5–5 mm pitch band accepts *every* RANSAC line — not
   specific. Periodicity as currently parameterized does not discriminate.

4. **Probe D — `probe_wide_smooth.py`.** Tested wide-Gaussian / p5-quantile
   salvage for the T2 soft-tissue filter. Best variant (p5, 10 mm sphere)
   recovers 16 % of T2 in-tube (vs 0.3 % baseline) but over-prunes T22
   (drops from 78 % to 7.6 %). No scanner-universal filter.

5. **Probe E — `probe_log_frangi.py`.** **LoG σ=1 gives a
   scanner-universal contact signature.** Median LoG at in-GT-tube
   voxels is −530 on both T22 and T2. Every in-tube voxel has LoG
   ≤ −33 on both subjects. Halo-immune because LoG at σ=1 responds to
   sharp peaks, which halos (being smooth) don't produce. Also tested
   Frangi × |LoG| dot product — within a Frangi cloud, LoG does almost
   all the work.

6. **v5-LoG — `probe_detector_v5_log.py`.** Plug the LoG-gated cloud into
   v4-style RANSAC. T22: 8/8 with 5 FPs (great). T2: **8/12 — worse than
   v4** because threshold 300 drops a few weak contacts. LPRG still missed.

7. **Probe F — `probe_log_oracle.py`.** Given the correct GT axis,
   LoG + scipy.find_peaks finds clean Dixi 3.5 mm pitch on 17 of 20
   shanks across both subjects (including **LPRG**: 11 peaks at exactly
   3.5 mm). Remaining three are either short (LSAN, RSAN — too few
   peaks) or partial-volume-fused (T22 L at 6 mm effective). **The signal
   is everywhere; LPRG was never a signal problem — it was a
   voxel-RANSAC axis-finding problem.**

8. **Probe G — `probe_blob_pitch.py`.** Skip voxel RANSAC entirely. Run
   pitch-constrained Hough over LoG regional-minima blobs — for each blob
   pair within ~Dixi pitch, count other blobs at k·pitch positions on the
   line. This directly uses the 3.5 mm prior to construct hypotheses, no
   discretized accumulator needed. **Catches LPRG** (blob-space is 50×
   smaller than voxel-space and has no between-electrode clumping to
   steer RANSAC off-axis). Early version: T22 7/8 FP 4, T2 10/12 FP 26.
   After multi-pitch-seed + perp_tol 1.5 mm + amp_sum gate: T22 8/8 FP 6,
   T2 11/12 FP 12 (only RSAN missed).

9. **Probe H — `probe_two_stage.py`.** RSAN on T2 has no resolvable
   contacts — on CT it's a uniform dark bar, so LoG is flat along it,
   so blob-pitch cannot work by physics. The right tool is the Frangi
   tube detector. Stage 2 = Frangi CC + PCA on the residual cloud
   after a 3 mm exclusion zone around stage-1 lines. Replaces iterative
   RANSAC with direct geometric primitives (CC + PCA) — same detection
   power, 30× faster. **Recovers RSAN at T2, gives 12/12 + 8/8.**

The two-stage decomposition matches physical signal categories: shanks
with pitch-visible contacts (LoG peaks) and shanks without (Frangi tubes).
No single detector can cover both because the LoG of a continuous tube is
zero in the middle by definition.

## Known issues / next steps

1. **Stage-2 FPs (~15 per subject).** Tubular CCs in skull/bone pass
   the aspect filter. Tightening `span ≥ 25 mm` or `aspect_geom ≥ 8`
   would cut most without losing RSAN (span 38, aspect ~41). A geometric
   prior — require one endpoint within 15 mm of hull — is probably
   a cleaner FP killer.
2. **Stage-1 FPs (~10 per subject).** Also tunable; amp_sum threshold
   is the main lever.
3. **Pipeline integration.** `probe_two_stage.py` is standalone. Porting
   into `PostopCTLocalization/.../deep_core_v2_fit.py` hasn't been done;
   when porting, follow `feedback_probe_first.md` (port the probe logic
   literally without adding extra filters).
4. **Non-Dixi vendors.** The 3.5 mm pitch prior is Dixi-specific.
   Ad-Tech (5 mm) and PMT (3.5–5 mm) would need a multi-pitch variant —
   try k=1,2 pitch at both 3.5 and 5.0 in stage 1.

## Probe inventory (this session)

All under `tests/deep_core/`. Standalone, `shankdetect` env.

| Probe | What it tests / shows |
| --- | --- |
| `probe_bolt_recovery.py` | σ=2 + linearity — fails for bolt detection |
| `probe_contact_recovery.py` | soft-tissue filter — fails on T2 halos |
| `probe_diagnose.py` | deep-dives on why Probes A/B fail |
| `probe_periodicity_gate.py` | windowed FFT periodicity — not specific |
| `probe_wide_smooth.py` | wide-kernel halo salvage — not scanner-universal |
| `probe_log_frangi.py` | **LoG σ=1 is scanner-universal for contacts** |
| `probe_detector_v5_log.py` | v4 RANSAC with LoG cloud — T22 big win, T2 tradeoff |
| `probe_log_oracle.py` | **Given correct axis, LoG + find_peaks hits 17/20 shanks** |
| `probe_blob_pitch.py` | Stage 1: pitch-constrained Hough on LoG regional minima |
| `probe_two_stage.py` | **Current best: blob-pitch + Frangi shaft fallback** |

## Ground rules

- v1/v2 pipelines are committed and gated by `test_pipeline_dataset.py` /
  `test_pipeline_dataset_v2.py`. Don't touch them.
- Probes are standalone; they don't go through the `deep_core` pipeline.
- Port to the pipeline only after confirming numbers match the probe
  (see `feedback_probe_first.md`).
- Use `shankdetect` conda env.
- ROSA GT for T22: `contact_label_dataset/rosa_helper_import/T22/
  ROSA_Contacts_final_trajectory_points.csv`. T2 uses the shank TSV via
  `load_ground_truth_shanks`.

## Committed v2 state (for reference)

### Cylinder RANSAC algorithm (the core)

For each bolt candidate:

1. **PCA-refit** the bolt axis on a 2.5 mm-radius tube of bolt-metal
   voxels (HU >= 2000).
2. Build a cylinder from bolt center along inward axis. Depth capped at
   `lib_max + 20 mm`.
3. Extract voxels HU >= 1000 AND head_distance >= 3 mm.
4. RANSAC line fit constrained to within 15° of bolt axis.
5. PCA-refit inliers → refined axis.
6. Density-binned deep tip (5 mm bins, stop after 3 sparse bins).
7. Shallow endpoint at −12 mm from bolt center.

### Dedup

`_dedup_v2_trajectories`: removes near-collinear duplicates (< 15°, < 8 mm
perp) with overlap guard preventing bilateral merge.

### Code layout

| File | Role |
| --- | --- |
| `PostopCTLocalization/.../deep_core_v2_fit.py` | Cylinder RANSAC + density trim |
| `CommonLib/.../deep_core_v2.py` | Pipeline + dedup with overlap guard |
| `tests/.../test_pipeline_dataset_v2.py` | Baselines T22 ≥ 9/9, T1 ≥ 10/12 |
