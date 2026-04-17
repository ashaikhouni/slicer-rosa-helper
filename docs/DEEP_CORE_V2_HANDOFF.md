# deep_core_v2 Handoff

Last updated: 2026-04-17. Probe files uncommitted on `main`, built on `d6e7d9b`.

## What this document is

You are Claude Code continuing work on SEEG shank detection. The v2
cylinder-RANSAC pipeline (committed, `d6e7d9b`) works on high-dynamic-range
CTs but fails entirely on **clipped CTs** (T2, T4) where HU saturates at
3071 and bolt metal becomes indistinguishable from dense bone.

This session produced a **new HU-agnostic detector probe** (`probe_detector_v4.py`)
that solves the clipped-CT problem.

## Orient first

```bash
cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper
git log --oneline -5
git status
ls tests/deep_core/probe_detector_v4.py
ls tests/deep_core/probe_periodicity.py
ls tests/deep_core/probe_frangi_cc.py
```

## Committed state — v2 (d6e7d9b)

Cylinder RANSAC + density-binned depth trimming. Works on high-DR CTs.

- **T22: 9/9** (using shank TSV GT, different from ROSA CSV GT in this session)
- **T1: 11/12** (RHH miss is GT inaccuracy)
- **T2: 0/12**, **T4: 0/16** — bolt detection fails (HU saturation)

See the `## Committed v2 state` section below for algorithm details.

## This session — HU-agnostic detector (v4 probe)

### Breakthrough result

| | T22 (clean) | T2 (clipped) |
| --- | --- | --- |
| v2 | 9/9 | 0/12 |
| **v4 probe** | **8/8** (ROSA GT, 8 trajectories) | **11/12** |

Same detector, no HU thresholds. Only LPRG still missed on T2.

### Why RANSAC beat ridge tracking (v3 probe)

Ridge tracking (v3) used local Hessian eigenvectors for axis direction.
At σ=2 (needed to bridge inter-contact gaps), close-parallel electrodes
merge in the Hessian → eigenvector averages their axes → tracking follows
a line midway between two real electrodes.

RANSAC fits a line globally through separated voxel clusters. Close-parallel
electrodes give separate clusters in Frangi σ=1, so RANSAC finds two
distinct lines. Immune to Hessian averaging.

### Pipeline

All in `tests/deep_core/probe_detector_v4.py`:

1. **Mask**: `hull_mask` (HU ≥ −500, close+fill, largest CC) ∩ `head_distance ≥ 10 mm`
   → intracranial ROI. HU-agnostic (air/tissue boundary is robust under clipping).

2. **Frangi σ=1 (contact scale)**: `ObjectnessMeasure` on Gaussian-smoothed
   CT. Enhances contact-sized tubes.

3. **Voxel cloud**: threshold Frangi σ=1 ≥ 10 inside ROI.

4. **Iterative RANSAC**:
   - Weighted random 2-point samples, 800 iters per line
   - Inlier tolerance 1 mm
   - Min 40 inliers, span 20–85 mm, density ≥ 0.5 inliers/mm
   - PCA refinement after best RANSAC candidate
   - Span clipping if initial fit > 85 mm (slides window, keeps densest region)
   - Exclusion radius 1.5 mm around accepted line
   - Repeat until no line passes thresholds

5. **Dedup**: merge tracks with axis angle < 5°, perp distance < 3 mm,
   span overlap ≥ 40% (keeps longer).

6. **Periodicity confirmation** (`probe_periodicity.py` derived):
   - Cylindrical sample (r=1 mm, 8-point ring, max reducer) along track axis,
     intra-brain only
   - FFT in band 0.2–0.4 cycles/mm (pitch 2.5–5 mm)
   - Accept if pitch ∈ [2.6, 4.8] mm AND SNR ≥ 2, OR track length ≥ 40 mm

7. **Evaluation**: greedy 1-to-1 GT assignment (angle ≤ 10°, mid-d ≤ 8 mm).

### How to run

```bash
# v4 detector probe
/Users/ammar/miniforge3/envs/shankdetect/bin/python3 \
  tests/deep_core/probe_detector_v4.py T22   # ~40s
/Users/ammar/miniforge3/envs/shankdetect/bin/python3 \
  tests/deep_core/probe_detector_v4.py T2    # ~130s (RANSAC over large cloud)

# Periodicity probe (shows 3.5 mm pitch detectable on both clean + clipped CTs)
/Users/ammar/miniforge3/envs/shankdetect/bin/python3 \
  tests/deep_core/probe_periodicity.py T22
```

### Output files

- `/tmp/detected_{T22,T2}_v4.nii.gz` — accepted tracks as labeled tubes
- `/tmp/frangi23_{T22,T2}.nii.gz` — multi-scale Frangi tube response
- `/tmp/gm_{T22,T2}.nii.gz` — gradient magnitude volume

Load alongside CT in Slicer for visual verification.

### Supporting probes (same session)

| Probe | Purpose |
| --- | --- |
| `probe_gm_bolt_entry.py` | Confirms GM survives HU clipping (T2 GM strength ≥ T22) |
| `probe_gm_entry_points.py` | GM peaks cleanly at bolt-through-skull entry points |
| `probe_frangi.py` | Multi-scale Frangi runs in ~1.5s; strong signal on both subjects |
| `probe_frangi_cc.py` | Threshold + CC + linearity filter (superseded by v4) |
| `probe_periodicity.py` | 3.5 mm Dixi contact pitch detectable on all 20 shanks (with cylindrical sampling, intra-brain only, band 0.2–0.4 cyc/mm) |
| `probe_skeleton_branches.py` | Negative result: skeleton of skull+metal is sheet-dominated |
| `probe_detector_v3.py` | Ridge tracking approach (v3); T22 7/8, T2 9/12 |

## Known issues remaining

### 1. LPRG (T2) still missed

Closest v4 track has axis 19° off from GT. Probably same close-parallel
failure even with RANSAC — possibly because LPRG's Frangi σ=1 voxels are
not well-separated from a neighbor.

### 2. Many false positives

v4 on T22 produces 35 accepted tracks for 8 real electrodes; T2 has 59 for
12. Likely skull folds, blood vessels, calcifications that pass periodicity
and length thresholds. Not hurting recall but needs post-filtering before
production.

### 3. T2 runtime ~130 s

RANSAC iterates 800× per line over a large voxel cloud. Each iteration does
O(N) distance-to-line computation. Can optimize with early termination or
smarter initial sampling.

## Next-session plan (updated 2026-04-17 late)

**Architecture target:** v2-style bolt-anchored pipeline with HU-agnostic
detectors replacing the two HU-dependent stages:

- Bolt detection ← Frangi σ=2 + linearity ≥ 15 mm
- Contact signal ← Frangi σ=1 + soft-tissue neighborhood filter inside brain
- Pairing ← unchanged v2 cylinder-RANSAC on contact voxels within each
  bolt's cylinder
- Fallback ← v4-style RANSAC on contact voxels not claimed by any
  bolt-linked trajectory (handles missed bolts)

**But first, two diagnostic probes** — understand each signal in isolation
before combining. Notes are on non-contrast CT, so vessels and dural
sinuses don't appear (high-σ linearity is specific to bolts/shafts).

### Probe A — `probe_bolt_recovery.py`

Frangi σ=2 on raw CT (full volume, no mask). Threshold sweep
{20, 40, 80} → connected components → PCA: require span ≥ 15 mm AND
λ₂/λ₁ ≤ 0.05. Per GT: entry within 10 mm, axis within 15°.
Report TP/FP per threshold. Answers: is a mask needed, or does linearity
alone filter the skull?

### Probe B — `probe_contact_recovery.py`

Frangi σ=1 threshold ≥ 10 + soft-tissue filter (5 mm-radius neighborhood
median HU ∈ [−50, 80]). Run **with and without** intracranial mask.
Report inlier density along each GT axis + total unexplained voxels.
Answers: is the soft-tissue filter alone enough to replace the mask?

**Do not build an end-to-end detector in the diagnostic session.** Just
produce the numbers and NIFTI outputs for Slicer inspection, then discuss
with Ammar before designing v5.

**Ammar's reasoning:** splitting into component probes lets us understand
which signals are strong and which filters are redundant, so v5 can be
designed with minimal parameter hacking.

## Ground rules

- Don't touch v1 or v2 pipelines. They work for clean CTs and tests in
  `tests/deep_core/test_pipeline_dataset.py` / `test_pipeline_dataset_v2.py`
  are locked in.
- v4 probe is standalone — not yet integrated into pipeline. Integration
  happens after the next-session idea is evaluated.
- Use `shankdetect` conda env for anything needing SimpleITK.
- ROSA GT for T22 is at
  `contact_label_dataset/rosa_helper_import/T22/ROSA_Contacts_final_trajectory_points.csv`
  (user edited, preferred over shank TSV).

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
