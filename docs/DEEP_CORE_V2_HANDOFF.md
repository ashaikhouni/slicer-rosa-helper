# deep_core_v2 Handoff

Last updated: 2026-04-16. Uncommitted changes on `main`, built on `ba1462e`.

## What this document is

You are Claude Code continuing work on the **bolt-first `deep_core_v2`
pipeline**. The cylinder-gather + axis-constrained RANSAC approach now
achieves correct electrode axis identification AND depth determination:

- **T22: 9/9** loose match (all shanks)
- **T1: 10/12** loose match

## Orient first

```bash
cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper
git log --oneline -5
git diff --stat
ls PostopCTLocalization/postop_ct_localization/deep_core_v2_fit.py
ls CommonLib/shank_engine/pipelines/deep_core_v2.py
ls tests/deep_core/probe_cylinder_fit.py
```

## Current state — what works

### Cylinder RANSAC algorithm (the core)

For each bolt candidate:

1. **PCA-refit** the bolt axis on a 2.5 mm-radius tube of bolt-metal
   voxels (HU >= 2000) to get a slightly better prior.
2. Build a cylinder from the bolt center along the inward (deep) axis.
   Depth is capped at `lib_max + 20 mm` (~100 mm) to avoid capturing
   contralateral electrodes.
3. Extract all voxels in the cylinder with **HU >= 1000** and
   **head_distance >= 3 mm** (bright + intracranial).
4. **RANSAC line fit** constrained to lines within **15 deg** of the
   bolt axis. This rejects crossing electrodes.
5. PCA-refit the RANSAC inliers -> refined electrode axis.
6. **Density-binned deep tip** (`_density_trimmed_deep_tip`): walk
   5 mm bins from shallow to deep; a bin is "sparse" if it has < 3
   inliers; stop after 3 consecutive sparse bins (gap_tolerance=2).
   This separates electrode contacts (dense) from contralateral bone
   (sparse, separated by empty bins).
7. Shallow endpoint at `-12 mm` from bolt center (v2-specific offset).

### Dedup

`_dedup_v2_trajectories` removes near-collinear duplicates (< 15 deg
angle, < 8 mm perp distance) keeping the longer one. An **overlap
guard** prevents bilateral electrodes on opposite sides of the head
from being falsely merged: the candidate's midpoint must project
within the kept trajectory's segment (+/- 50% margin).

### Results (per-shank axis angle vs GT)

- T22: all 9 GT shanks matched, axis error 3.6-9.2 deg, worst
  end_error 9.7 mm
- T1: 10/12 GT shanks matched, axis error 0.1-6.2 deg

### Code layout

| File | What changed |
|---|---|
| `PostopCTLocalization/.../deep_core_v2_fit.py` | `_density_trimmed_deep_tip` (density-binned depth walk), `_fit_cylinder_ransac` uses it + 12 mm bolt offset + library-capped cylinder depth. |
| `CommonLib/.../deep_core_v2.py` | `_dedup_v2_trajectories` overlap guard (prevents bilateral electrode merging). |
| `tests/.../test_pipeline_dataset_v2.py` | Baselines raised: T22 >= 9/9, T1 >= 10/12. |

### Probe scripts (not production code)

| File | Purpose |
|---|---|
| `tests/deep_core/probe_cylinder_fit.py` | Gold reference for the algorithm. |
| `tests/deep_core/probe_bolt_axis_profile.py` | HU profile along bolt axis. |

## How to run things

```bash
# v2-only test
/Users/ammar/miniforge3/envs/shankdetect/bin/python3 -m unittest \
  tests.deep_core.test_pipeline_dataset_v2

# Full regression (v1 must stay green)
/Users/ammar/miniforge3/envs/shankdetect/bin/python3 -m unittest \
  tests.deep_core.test_pipeline_dataset \
  tests.deep_core.test_pipeline_dataset_v2

# Cylinder-fit probe on a subject (visual axis quality check)
/Users/ammar/miniforge3/envs/shankdetect/bin/python3 \
  tests/deep_core/probe_cylinder_fit.py T22 /tmp/cylfit_mask_T22.nii.gz
```

Always use the `shankdetect` conda env Python.

## Known issues — what remains

### 1. T1 RHH (end_error 11.7 mm)

RHH's axis and deep tip are correct (start_err=1.1 mm, angle=2.7 deg)
but the shallow-end error (11.7 mm) exceeds the 10 mm match threshold.
The fixed 12 mm bolt offset places the shallow end too deep for this
particular bolt. A data-driven shallow endpoint (derived from the
shallowest dense inlier cluster) might fix this, but the offset is
a trade-off: 8 mm helps RHH but breaks T22's RSFG/LCMN/LAC.

### 2. T1 LAI (undetected)

The bolt is found (bolt 2, 171 bolt-metal inliers) but the cylinder
RANSAC finds only 28 bright intracranial inliers — well below the 150
threshold. The electrode is very dim or the bolt axis is significantly
misaligned (14.8 mm from GT shallow end). Lowering the threshold to
100 does not recover it (only 28 inliers regardless of threshold).

### 3. RAMC+LAMC collinearity (worked around)

RAMC and LAMC are nearly collinear. The cylinder depth cap at
`lib_max + 20 mm` (~100 mm) prevents capturing LAMC contacts from
RAMC's bolt. The density trim brings RAMC's start_error to 14.3 mm
(within the 20 mm threshold). If longer electrodes are added to the
library, the cylinder depth cap will increase — verify that RAMC
doesn't regress.

## Ground rules

- Don't touch v1 or v1+bolts. Those are locked in
  `tests/deep_core/test_pipeline_dataset.py`.
- v2 test baselines can be **raised** as recall improves.
- When adding config knobs, use `getattr(cfg, "v2_*", default)` in
  the fit function. Add to `DeepCoreModelFitConfig` only when they
  need UI exposure.
- Use the `shankdetect` conda env for anything needing SimpleITK/vtk.
