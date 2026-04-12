# Deep Core Phase B Status — handoff doc

Last updated: 2026-04-12

## What is this document

You are Claude Code continuing work on the deep_core SEEG electrode detection module in the rosa_viewer project. The previous session implemented **Phase B: electrode-template group fitting**. Read this doc first, then `git log --oneline -15` for recent commits, then the files listed below.

Memory files at `~/.claude/projects/-Users-ammar-Dropbox-rosa-viewer/memory/` have user profile and feedback.

## Phase B in one sentence

Phase B replaces the geometric "walk metal until thick" extension stage with template matching: for each proposal axis, slide each library electrode model along the axis, place contacts according to the model's known offsets, and select the best-fitting (model, anchor) combination using group-level non-conflicting assignment.

## Files

**Core implementation:**
- `CommonLib/shank_engine/pipelines/deep_core_v1.py`
  - `DeepCoreV1Pipeline._run_model_fit()` — stage wrapper, loads library, calls `run_model_fit_group`
  - `DeepCoreV1Pipeline._get_electrode_library()` — lazy-loads and caches the library on the pipeline instance
  - `DeepCoreV1Pipeline.run()` — when `cfg.model_fit.enabled=True` (default), calls `_run_model_fit` instead of the legacy `_run_extension` + `_run_final_rejection`
- `PostopCTLocalization/postop_ct_localization/deep_core_model_fit.py` — all model-fit algorithm logic
  - `run_model_fit_group()` — entry point: generates hypotheses, greedy non-conflict assignment, returns accepted proposals
  - `_hypotheses_for_proposal()` — per-proposal loop over models × shift positions
  - `_sample_hu_at_ras()` — HU sampling with `reduction="max"` for box samples or `radius_vox=0` for point samples
  - `_check_lateral_profile()` — validates each contact center is brighter than perimeter samples
  - `_axis_segment_median_hu()` — samples HU densely along the axis segment; filters out "axis crosses tissue between random bright spots" false positives
  - `_depth_at_ras()` — looks up head_distance_map_kji for in-brain classification
  - `filter_models_by_family()` — filters library by ID prefix and `min_contacts`

**Config:**
- `PostopCTLocalization/postop_ct_localization/deep_core_config.py`
  - `DeepCoreModelFitConfig` — all Phase B parameters
  - `DeepCoreConfig.model_fit: DeepCoreModelFitConfig`
  - `DeepCoreMaskConfig.deep_core_shrink_mm: float = 20.0` (bumped from 15)

**Electrode library:**
- `CommonLib/rosa_core/electrode_models.py` — `load_electrode_library()` returns dict with `"models"` list
- `CommonLib/resources/electrodes/electrode_models.json` — 16 models: 9 DIXI + 7 PMT
- Contact offsets are stored in `contact_center_offsets_from_tip_mm` (smallest offset = closest to physical tip = **deepest contact when implanted**)

**Tests:**
- `tests/deep_core/test_pipeline_dataset.py` — regression tests on T1 and T22 from the SEEG dataset
- `tests/deep_core/test_annulus_sampler.py` — fast unit tests (pure Python)

## Config defaults (DeepCoreModelFitConfig)

```python
enabled: bool = True
families: tuple[str, ...] = ("DIXI",)           # DIXI only by default
deep_anchor_search_mm: float = 5.0              # slide ±5mm from proposal's deep endpoint
deep_anchor_step_mm: float = 0.5
hit_hu_threshold: float = 1500.0                # contact "hit" if HU > this
sample_radius_vox: int = 2                      # center sample box = 1mm radius
lateral_offset_mm: float = 2.5                  # perimeter sample offset
min_lateral_drop_hu: float = 500.0              # center must be this much brighter than perimeter
in_brain_min_depth_mm: float = 10.0             # head_distance threshold for "in-brain"
min_in_brain_contacts: int = 6                  # excludes DIXI-5AM; bolt safety
min_in_brain_hit_fraction: float = 0.70         # 70% of in-brain contacts must hit
conflict_radius_mm: float = 2.0                 # two fits can't share contacts within this distance
min_axis_segment_median_hu: float = 600.0       # axis segment median HU floor
```

## Algorithm (condensed)

1. **Input**: proposals from the candidate generator (mix of line atoms and contact chains), each with `start_ras`, `end_ras`, approximate axis.

2. **Per-proposal hypothesis generation** (`_hypotheses_for_proposal`):
   - Determine the deeper endpoint using `head_distance_map_kji`; compute `shallow_axis` pointing from deep tip toward entry.
   - For each eligible model in the family (contact_count ≥ 6 for DIXI):
     - For each `shift_mm` in `[-5, +5]` step 0.5mm:
       - Anchor the deepest contact at `deep_seed + shallow_axis * shift_mm`.
       - Place all contacts at `deepest + delta * shallow_axis` where `delta = offset - min(offsets)`.
       - For each contact position:
         - Sample center HU (box, radius=2 vox, max reduction)
         - Sample lateral HU (point samples at ±2.5mm in 2 perpendicular directions)
         - Check in-brain (`head_distance ≥ 10mm`)
         - "Hit" = in-brain AND center > 1500 HU AND `(center - max_lateral) ≥ 500 HU`
       - Accept if ≥6 in-brain hits AND hit_fraction ≥ 0.70 AND axis_segment_median_HU ≥ 600
     - Emit hypothesis dict with sort_key

3. **Group fitting** (`run_model_fit_group`):
   - Sort all hypotheses across all proposals by `sort_key` descending:
     - `(n_in_brain_hits, -model_span_mm, -contact_count, raw_hu_sum_in_brain)`
   - Greedy loop:
     - Skip if this proposal already has an accepted fit
     - Skip if any predicted contact is within 2mm of an already-accepted contact
     - Otherwise accept the fit and add its contacts to the claimed set
   - Hard-reject proposals with no accepted fit.

4. **Output**: rewritten proposals with exact contact-center endpoints and `best_model_id` metadata. No extension or final_rejection stage runs after this (those are skipped when model_fit is enabled).

## Key design decisions

- **Group fitting, not per-proposal**: prevents two fits from claiming the same physical metal (parallel electrode case).
- **Full-volume HU sampling, not mask-filtered**: lets the algorithm place contacts in the superficial 0-20mm zone where the candidate generator can't see (the deep_core_shrink_mm filter blocks it from the blob search but not from template sampling).
- **In-brain restriction**: bolts (anchoring hardware that extends up to 40mm outside the skull) have zero in-brain extent, so requiring ≥6 in-brain contacts is the bolt-safety mechanism.
- **Shortest-model tiebreak**: when two models fit equally well by hit count, prefer the shorter physical span (then fewer contacts). The user's explicit requirement: "use the shortest electrode that explains the in-brain portion of the electrode."
- **Hard-reject unassigned**: no fallback to geometric extension. If no model fits, the proposal is dropped.
- **Axis-segment median HU check**: prevents false positives where a proposal axis passes through tissue between sparse bright spots that happen to belong to *different* electrodes. A real electrode has bright metal along its entire contact-spanning segment.

## Current results (Phase B baseline, locked in regression tests)

Runtime environment: `conda activate shankdetect` (numpy + SimpleITK + shank_engine).

### T1 (default config)

| Metric | Value |
|---|---|
| GT shanks | 12 |
| Predicted | 12 |
| Loose match (10mm tip / 25° / 20mm shallow) | 8/12 |
| Strict match (4mm tip / 25° / 15mm shallow) | 3/12 |

Matched shanks have **angles <2°** (often <1°) and endpoint errors **1-7mm**. Best-fit models are mostly DIXI-18AM and DIXI-15AM.

### T22 (`mask.metal_threshold_hu=1000`)

| Metric | Value |
|---|---|
| GT shanks | 9 |
| Predicted | 8 |
| Loose match | 4/9 |
| Strict match | 0/9 |

### Comparison to pre-Phase-B (geometric extension)

| | Extension | Phase B | Delta |
|---|---|---|---|
| T1 loose | 11/12 | 8/12 | -3 |
| T1 strict | 4/12 | 3/12 | -1 |
| T22 loose | 7/9 | 4/9 | -3 |
| T22 strict | 2/9 | 0/9 | -2 |

**Phase B scored slightly worse on loose match** because it's stricter about rejecting candidates with no clean model fit. Strict match is roughly flat. The *shape* of the errors improved (angles near-zero, endpoints anchored at real contacts) but some shanks are now missed entirely that the extension stage captured with wider tolerances.

## Known issues / open questions

### Issue 1: Model selection ambiguity on line-appearance electrodes

When the CT shows a continuous bright wire (not discrete beads), DIXI-18AM and DIXI-15CM both score 18/18 and 15/15 respectively at the same anchor position. The sort key prefers DIXI-18AM because it has more hits, but the actual electrode might be DIXI-15CM (15 contacts, 4.5mm spacing vs DIXI-18AM's 18 contacts, 3.5mm spacing).

**Fix candidates:**
- Sample at half-spacing positions (between predicted contacts) — for bead electrodes, between-contact HU is low; for line electrodes, it's high. Use this to choose between models with identical hit counts.
- Explicit spacing estimation from the actual bright-voxel positions on the axis, matched to the closest model spacing.
- Per-trajectory model hint from the user (they know the electrode type at implant time).

### Issue 2: Loose-match recall dropped vs prior baseline

Phase B is stricter about rejecting candidates. Some real shanks that the extension stage matched loosely are now rejected because:
- Not enough in-brain contacts in any model fit
- Axis-segment median check fails (possibly over-strict at 600 HU)

**Fix candidates:**
- Lower `min_in_brain_hit_fraction` from 0.70 → 0.60
- Lower `min_axis_segment_median_hu` from 600 → 400
- Relax the lateral drop check from 500 HU → 300 HU
- Investigate specific missed shanks on T1 (LAMC, LPMC, RAMC, RPMC) to see which check is rejecting them

### Issue 3: Some shanks completely missing from candidates

Phase B only refines existing candidates; it can't create new ones. The candidate generator has its own thresholds that may exclude real shanks (short electrodes, sparse blobs, wrong PCA axis). This is what Phase C would address.

### Issue 4: T22 is worse than T1 and I don't know exactly why

T22 uses a lower metal threshold (1000 vs 1900 HU) so blobs are larger and noisier. The DIXI-18CM model keeps winning in the sort even though DIXI-18AM should be the right answer for the actual electrodes. Probably related to Issue 1.

## How to run and debug

### Environment setup

```bash
conda activate shankdetect
```

### Run regression tests

```bash
cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper
python3 -m unittest tests.deep_core.test_pipeline_dataset -v
```

Takes ~10 seconds. Expects `ROSA_SEEG_DATASET` env var or the default path `/Users/ammar/Dropbox/thalamus_subjects/seeg_localization`.

### Run pipeline on one subject directly

```bash
PYTHONPATH=/Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper/CommonLib:/Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper/PostopCTLocalization:/Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper/tools:$PYTHONPATH \
python3 tools/eval_seeg_localization.py \
  --dataset-root /Users/ammar/Dropbox/thalamus_subjects/seeg_localization \
  --out-dir /tmp/eval_output \
  --pipeline-key deep_core_v1 \
  --subjects T1
```

### Debug a specific proposal's fit

Use the captured-proposal pattern: monkey-patch `pipe._run_model_fit` to capture its input, then call `_hypotheses_for_proposal` directly with the proposal and a single model. See session history for example scripts.

### Disable model_fit (fall back to extension)

Set `"model_fit.enabled": False` in the pipeline config dict. The old extension + final_rejection stages will run instead.

## Useful session commands

```bash
# Show recent deep_core-related commits
git log --oneline -20 -- PostopCTLocalization/postop_ct_localization/deep_core_* CommonLib/shank_engine/pipelines/deep_core_v1.py

# Count deep_core module size
wc -l PostopCTLocalization/postop_ct_localization/deep_core_*.py CommonLib/shank_engine/pipelines/deep_core_v1.py | sort -n

# List all active shank_engine pipelines
grep register_pipeline CommonLib/shank_engine/bootstrap.py
```

## Where things stand

The architecture is clean. Phase B is implemented and running. It gets the **shape of the answer right** (angles, endpoint alignment) but doesn't yet win on strict-match recall because model selection isn't perfect and some proposals get dropped for lacking enough in-brain hits. The next session should focus on:

1. Visually debugging specific failing cases with CT screenshots (that's why the user is switching to VS Code)
2. Fixing the model-selection ambiguity (Issue 1) — probably by adding between-contact HU sampling
3. Relaxing one or two of the validation thresholds to recover the loose-match recall that dropped

## Don't forget

- Don't break the regression tests — they encode the currently-locked baselines
- The Phase B code lives in `deep_core_model_fit.py` (pure Python, no Slicer) — keep it that way so it stays testable
- Respect the user's principle: **shortest electrode that explains the in-brain portion wins ties**. Don't introduce a "longest wins" rule without discussion.
- Bolts can extend up to 40mm outward from the skull. The `min_in_brain_contacts = 6` floor is the bolt-safety mechanism — don't lower it casually.
- The user's preferred workflow is **commits for every working milestone** so they can test in Slicer between changes. Don't pile up uncommitted work.
