# deep_core_v2 Handoff

Last updated: 2026-04-15. Session ends at commit `d2d36d6`.

## What this document is

You are Claude Code continuing work on the **bolt-first `deep_core_v2`
pipeline**. v1 is shipped and at full recall (12/12 loose on T1 and 9/9
loose on T22 with bolts enabled). v2 is a proof-of-concept that drops
Phase A entirely and relies on RANSAC bolt detection + a direct
bolt-anchored fit. It currently sits at T1 10/12, T22 5/9.

Your job this session: push v2 toward full recall using one of the
directions listed under "Open problems" below. Do not touch v1 or
the v1+bolts integration — those are locked and in regression tests.

## Orient first

```bash
cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper
git log --oneline -10
ls PostopCTLocalization/postop_ct_localization/deep_core_v2*.py
ls CommonLib/shank_engine/pipelines/deep_core_v2.py
ls tests/deep_core/test_pipeline_dataset_v2.py
```

Also read the "Deep Core v2" section of [PHASE_B_STATUS.md](PHASE_B_STATUS.md),
which has the full numbers and the architectural summary.

## Current state — code layout

| File | Purpose |
|---|---|
| [CommonLib/shank_engine/pipelines/deep_core_v2.py](../CommonLib/shank_engine/pipelines/deep_core_v2.py) | v2 pipeline class; inherits v1, overrides `run()` with `mask → bolt_detection → bolt_fit` |
| [PostopCTLocalization/postop_ct_localization/deep_core_v2_fit.py](../PostopCTLocalization/postop_ct_localization/deep_core_v2_fit.py) | Three fit paths: `two_threshold`, `deepest_peak`, `intensity_peaks`. All accept a `BoltCandidate` and return a trajectory dict. |
| [PostopCTLocalization/postop_ct_localization/deep_core_bolt_ransac.py](../PostopCTLocalization/postop_ct_localization/deep_core_bolt_ransac.py) | Bolt detection (RANSAC over bolt-metal voxels). Shared with v1. |
| [PostopCTLocalization/postop_ct_localization/deep_core_config.py](../PostopCTLocalization/postop_ct_localization/deep_core_config.py) | All v2 config knobs live under `DeepCoreModelFitConfig` as `v2_*` fields. |
| [tests/deep_core/test_pipeline_dataset_v2.py](../tests/deep_core/test_pipeline_dataset_v2.py) | Locked baselines: T1 ≥ 10/12 loose, T22 ≥ 5/9 loose. |

## How to run things

```bash
# Full deep_core regression (27 tests, ~45s wall-clock)
cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper
/Users/ammar/miniforge3/envs/shankdetect/bin/python3 -m unittest \
  tests.deep_core.test_pipeline_dataset \
  tests.deep_core.test_axis_reconstruction \
  tests.deep_core.test_bolt_detection \
  tests.deep_core.test_pipeline_dataset_v2

# v2-only (10s)
/Users/ammar/miniforge3/envs/shankdetect/bin/python3 -m unittest \
  tests.deep_core.test_pipeline_dataset_v2
```

Always use the `shankdetect` conda env Python for dataset tests — the
local `.venv` lacks SimpleITK and vtk.

## Current numbers and the v2 ceiling

| Subject | v1 (no bolts) | v1 + bolts | v2 bolt-first |
|---|---|---|---|
| T1 loose (12 GT) | 11 | **12** | 10 |
| T1 strict (12 GT) | 5 | **7** | 3 |
| T22 loose (9 GT) | 9 | **9** | 5 |
| T22 strict (9 GT) | 2 | **2** | 0 |

### Why v2 is worse than v1+bolts

Phase A is the load-bearing recall mechanism on T22-style subjects
where electrode contacts are below ~1500 HU. We verified directly that
v1 with `use_bolt_detection=False` on T22 still reaches 9/9 loose.
Phase A's atom-chain pre-extraction gives Phase B a rich cloud
already spanning the full electrode; bolts are an augmentation.

v2 drops Phase A by design and tries to walk from the bolt alone. On
T1 where electrodes are bright, the walk reaches deep brain easily
(10/12). On T22 where contacts are dim, the walk from the bolt hits
a ~15 mm gap between bolt metal and first-contact metal that no
global HU threshold can bridge without also crossing into bone or
picking up false bright voxels.

## The three v2 fit modes (in `deep_core_v2_fit.py`)

### `two_threshold` (default)

`_fit_two_threshold` → `_find_deep_tip_by_contiguous_metal`.

Walks `-axis` from the bolt center, at each step sampling a
`v2_contact_probe_radius_mm = 2.5 mm` tube around the axis. If any
voxel in the tube exceeds `v2_contact_hu = 400 HU`, it's a hit. Tracks
a rolling gap counter; if the gap exceeds `v2_contact_max_gap_mm = 15 mm`
the walk stops. Returns the deepest hit as the deep tip.

**Best config found**: `hu=400 gap=15`. Anything tighter loses T22,
anything looser contaminates T1 with bone walks.

**Failure mode**: on 5 T22 shanks, the walk stops within ~15 mm of
the bolt because the electrode contacts never reach 400 HU above a
bone-darkened background. The deep tip is then the bolt's own edge
and the trajectory is too short.

### `deepest_peak`

`_fit_deepest_peak` → `_build_intensity_profile` + gap-tolerant
deepest-peak walk.

Builds a 1D max-HU profile along the axis using a 5 mm-diameter disc
perpendicular to it, then walks inward tracking the deepest sample
above `max(peak_hu_floor, peak_rel_frac * profile_max)` under the
same gap tolerance. Slightly better on T22 (one more shank), slightly
worse on T1 (noise peaks from bone get picked as deep tips).

### `intensity_peaks` (most complex, least robust)

`_fit_intensity_peaks` → `_find_profile_peaks` → `_fit_library_to_peaks`.

Detects local maxima in the profile, then tries every library
electrode model's known `contact_center_offsets_from_tip_mm` pattern
aligned at every observed peak, scoring by match count. Picks the
best-fitting model.

**Failed because**: contact spacing drifts 0.25–0.5 mm per contact
under axis imprecision. Over 10+ contacts the expected positions
walk 2.5–5 mm off the observed peaks, so long-electrode fits lose
their tail. Plus the library fit was picking DIXI-18CM anchored at
noise peaks and producing trajectories that extended past the bolt
into air; I added a shallow-end clamp (`max_shallow_t = 2.0 mm`) to
block that, but the core tolerance issue remains.

## Open problems and next directions

The v2 ceiling is set by **how far the inward walk can reach from a
bolt on dim-electrode subjects**. Ideas for breaking that ceiling,
in rough order of effort:

### 1. Frangi / Sato tubularity filter (highest payoff)

Replace the raw HU mask with a `scale ≈ 1.5 mm` line-filter
response. Electrode shanks light up strongly regardless of
absolute HU because they're tubular structures; bone doesn't
because it's a sheet. Walk the filter response instead of the HU
mask.

- `scipy.ndimage.gaussian_filter` + eigenvalue analysis of the local
  structure tensor gives you Frangi's vesselness measure.
- Pre-compute once per volume (seconds on T1/T22).
- Feed into `_find_deep_tip_by_contiguous_metal` as the sample
  function instead of the raw HU tube check.

### 2. Adaptive per-electrode intensity floor

Instead of a global `v2_contact_hu`, compute a per-bolt floor from
the intensity profile: e.g., `floor = max(HU[-10:0])` (the bolt
metal's own HU) × 0.3. This normalizes the threshold to each bolt's
brightness. Should help T22 where the global 400 HU cutoff is too
conservative for some shanks and too aggressive for others.

### 3. Bolt-axis refinement before walking

The 2.5 mm tube is probably too narrow for T22 because the RANSAC
axis has ~0.5-1° error. Over 40 mm that's 0.4-0.7 mm of lateral
drift, close to the tube's half-width. Fix:

- After finding the first N contacts via the walk, refit the axis
  through `bolt_center + first_N_contact_positions`.
- Re-walk from the refined axis.
- Iterate 2-3 times.

Related: rather than refining the axis, try widening the tube to
3.5-4 mm but adding a **second** narrower (1.5 mm) tube and requiring
the wide tube to hit AND the narrow tube to have a brighter sample
somewhere inside.

### 4. v2 → Phase A fallback (pragmatic)

If v2 rejects a bolt's fit, fall back to a v1-style Phase A
restricted to the bolt's deep_core region, seeded with the bolt's
axis as a strong prior. This is a hybrid architecture that would
inherit v1's recall on hard subjects while still being bolt-first on
easy ones. Probably the shortest path to 12/9 with v2 as the primary
path, at the cost of an architectural "escape hatch".

## Ground rules

- Don't touch v1 or v1+bolts.
- Dataset regression tests (`test_pipeline_dataset.py`) must stay
  green — v1 numbers are locked.
- v2 baselines (`test_pipeline_dataset_v2.py`) can be raised as you
  improve the pipeline. Don't lower them.
- When adding new v2 config knobs, put them on `DeepCoreModelFitConfig`
  as `v2_*` fields with `_ui_meta(... advanced=True)` so they don't
  clutter the main widget panel.
- When modifying widget code, update `_DEEP_CORE_CONTROL_ATTRS` in
  `deep_core_widget.py` for every new `_UI_FIELD_ORDER` entry — the
  widget setup crashes hard if a spec has no attribute mapping.
- Use the `shankdetect` conda env Python for anything needing
  SimpleITK or vtk.

## Quick probes to write early

- **Intensity profile dump**: for each T22 GT shank, sample the CT
  along the GT axis and print the HU profile. See what values the
  contacts actually hit. That tells you whether the 400 HU floor is
  reachable at all on those shanks.
- **Axis drift check**: for each T22 bolt, compare the RANSAC axis
  against the GT direction. If drift is > 1°, the tube-walk approach
  is fundamentally limited and you need axis refinement.
- **Longest walk distance per shank**: run `_fit_two_threshold` with
  very permissive config (hu=200, gap=30) and print the `t_deep`
  value for each bolt. That's the ceiling of what the walk can
  possibly achieve before false hits from bone dominate.

## Memory

There's a persistent memory at `~/.claude/projects/-Users-ammar-Dropbox-rosa-viewer/memory/`
with this handoff's location logged. The first thing you should do
is read `MEMORY.md` in that directory and the entries it points to.
