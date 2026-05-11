# User Guide

Last updated: 2026-05-11

A cross-surface guide to working with the SEEG / ROSA Toolkit. Covers the
end-to-end mental model and the data formats that flow between stages,
without going deep on either Slicer UI or CLI flags. Once you know which
surface fits your task, jump to the [`Slicer guide`](SLICER_GUIDE.md) or
the [`CLI guide`](../cli/README.md) for the actual buttons / commands.

## 1) What this toolkit does

You start from a SEEG implant: a patient with depth electrodes placed via
ROSA. You have some combination of:

- a **`.ros`** file (the surgical plan: trajectories + display volumes),
- a **post-op CT** (the as-implanted reality: bolts + electrode contacts),
- one or more **MRI** volumes (pre-op planning, FreeSurfer / atlas
  segmentations, or a target-MRI you want everything aligned to),
- a set of **atlas labelmaps** (FreeSurfer parcellation, THOMAS thalamic
  nuclei, white-matter parcellation).

The toolkit takes you from those raw inputs to:

- **Trajectories** — fitted shank lines on the actual CT, named to match
  the surgeon's plan, scored for confidence.
- **Contacts** — per-electrode contact positions, either at the model's
  nominal pitch ("model-driven") or snapped to CT image peaks
  ("peak-driven").
- **Atlas labels** — what anatomical structure each contact sits in
  (or is closest to), under one or more atlas sources.
- **Exports** — TSV / NIfTI / Slicer Markups / Curry POM in the frame
  of your choice.

## 2) Two surfaces, one algorithm

| Surface | Best for | How you run it |
|---|---|---|
| **3D Slicer extension** | clinical / research with full visual review, manual edits, atlas burns | install via Slicer module path; see [`SLICER_GUIDE.md`](SLICER_GUIDE.md) |
| **`rosa-agent` CLI**    | batch processing, regression, pipelines, scripted reproducibility, headless servers | `pip install .`; see [`CLI guide`](../cli/README.md) |

Both surfaces import the same `CommonLib/` algorithm. Detection on the
same CT goes through the same `rosa_detect.service.run_contact_pitch_v1`
entry point in both places, and the parity is pinned by tests — see
[`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md). So you can iterate on a case
in Slicer, batch-rerun a cohort headless, and trust that they match.

## 3) The end-to-end pipeline (conceptual)

```
   ┌──────────┐
   │  .ros    │── plan-only ─────────────────────┐
   └──────────┘                                  │
   ┌──────────┐                                  ▼
   │ post-op  │── detect ──► trajectories ── (optional) ──► named trajectories
   │   CT     │             (rosa_detect)      cross-volume matching
   └──────────┘                                       │
                                                     ▼
                                         place contacts (rosa_core.contact_placement)
                                                     │
                                                     ▼
                                  label against atlas (rosa_core.atlas_assignment_policy)
                                                     │
                                                     ▼
                                                  exports
```

Each stage is independent — you can start in the middle if you already
have the upstream artifact (e.g. a fitted trajectories TSV from another
tool). The contracts between stages are TSV columns; see
[`CLI guide § Output TSV columns`](../cli/README.md) for the schema.

### Stage A — load

Input: a ROSA case folder (`.ros` + `DICOM/<uid>/<name>.img/.hdr`).
Output: planned trajectories in RAS + a manifest of the displays / matrices.
Surfaces: Slicer **01 Loader**, CLI `rosa-agent load` /
`rosa-agent rosa-to-nifti`.

If you only have an external CT (no .ros) you skip this stage.

### Stage B — detect

Input: a post-op CT (NIfTI / NRRD), optional seeds (planned trajectories).
Output: fitted trajectory lines (one per electrode) with model id,
confidence band, bolt source.

The detector is `rosa_detect.run_contact_pitch_v1`. It runs in three
modes:
- **Auto fit** (no seeds) — direct shank detection from the CT.
- **Guided fit** (planned seeds, no model id) — uses the .ros line as a
  prior; PCA-fits the local CT evidence to refine geometry.
- **Manual fit** — adopts trajectories drawn by hand in the scene.

In Slicer this is **02 Postop CT Localization**; in the CLI it's
`rosa-agent detect` (or just `rosa-agent place` which composes detect +
contact placement).

### Stage C — place contacts

Input: trajectories + the same CT.
Output: per-contact RAS positions, electrode model id, confidence.

Two modes (set in Slicer's **02 Contacts & Trajectory View** or via the
CLI `rosa-agent place` strategy):
- **Model-driven** — contacts at the assigned electrode model's nominal
  pitch along the fitted axis. Trust the model + line, get clean output.
- **Peak-driven** — contacts at CT-image peaks (LoG σ=1, 2 mm disk along
  the axis), then matched against the electrode library. Surfaces curved
  shanks, drifted contacts, or wrong model assignments.

The 5-mode placement dispatcher (`rosa_core.placement_modes.place_seeg`)
lets you fix any subset of inputs:

| Mode | What you fix    | Use case                                                       |
|------|-----------------|----------------------------------------------------------------|
| 1    | nothing         | full auto: detect + place from a bare CT                       |
| 2    | `n_expected`    | "I expect N shanks; pick the top N by score"                   |
| 3    | named expected  | "These named shanks should be present" (surgical plan match)   |
| 4    | seeds + models  | external/manual seeds with vouched electrode models            |
| 5    | seeds only      | external/manual seeds; let the library matcher pick the model  |

### Stage D — label

Input: contacts + atlas sources.
Output: per-contact atlas labels + distances per source plus a unified
`closest_*` summary.

Atlas sources can be in their own RAS (registered inline against your
post-op CT) or already aligned. See [`SLICER_GUIDE.md`](SLICER_GUIDE.md)
for **01 Atlas Sources** and **02 Atlas Labeling**, or
[`CLI guide § Atlas labeling`](../cli/README.md) for the headless flow.

### Stage E — export

In Slicer: **04 Export Center** runs profile-driven exports (contacts only,
trajectories only, atlas only, full bundle …). In the CLI: every
subcommand writes its own QC directory; `rosa-agent pipeline` composes
the full bundle.

## 4) Coordinate frames

Everything in this toolkit works in **RAS millimeters** by default. The
edges of the system that talk to other ecosystems carry their own
conventions:

- **`.ros` / Analyze `.img/.hdr`** are stored in **LPS** — the
  `case_loader` flips them to RAS at load.
- **DICOM** is **LPS** — the export profiles handle the flip.
- **NIfTI** is **RAS** — drops in directly.
- **Curry POM** uses LPS — `curry_export` handles it.

When working across two coordinate frames for the same patient (e.g. a
plan in the ROSA reference frame + a CT post-registered to a different
MRI atlas), see [`CLI guide § match-ros`](../cli/README.md) for the
purely-geometric line-RANSAC matcher that re-aligns the two without
needing the original reference volume.

## 5) Data inputs and where they come from

| Input | Format | Notes |
|---|---|---|
| ROSA case | `<folder>/*.ros` + `DICOM/<uid>/<vol>.img/.hdr` | the surgical plan + the displays it was planned on |
| post-op CT | NIfTI (`.nii.gz`) preferred | RAS; ideally already registered to the planning MRI but `rosa-agent pipeline` can register it inline |
| pre-op MRI / atlas T1 | NIfTI | when atlas labelmaps are in T1 RAS, pass `--atlas-base` to register inline |
| FreeSurfer parc | NIfTI labelmap (`aparc+aseg.nii.gz`) + LUT | LUT path is `$FREESURFER_HOME/FreeSurferColorLUT.txt` |
| THOMAS | folder of per-nucleus masks | typically already aligned to its own MRI |
| WM parc | NIfTI labelmap | optional |

## 6) Output artifacts and where they live

CLI runs (anything via `rosa-agent`) write a **QC directory** with a
stable layout:

```
<output>/
  manifest.json          ← provenance: inputs, runtime, mode args
  trajectories.tsv       ← one row per fitted shank
  contacts.tsv           ← one row per contact
  labels.tsv             ← one row per contact (when --label is on)
  figures/               ← per-trajectory PNG QC plots (when matplotlib is available)
  diagnostics/cmp.tsv    ← internal diagnostics (peak amplitudes, score components, …)
```

Slicer runs publish the same data into MRML scene nodes via workflow
roles (e.g. `AutoFitTrajectoryLines`, `ContactFiducials`,
`FreeSurferLabelMap`); **04 Export Center** writes those out in the
chosen profile.

The TSV column contracts are stable and shared across the two surfaces —
see [`CLI guide § Output TSV columns`](../cli/README.md). External tools
that consume CLI output can also consume Slicer-exported output and vice
versa.

## 7) Choosing a workflow

### Full ROSA case, post-op CT in the same folder
- **Slicer**: 01 Loader → 02 Postop CT Localization → 02 Contacts &
  Trajectory View → 03 Atlas Labeling → 04 Export Center.
- **CLI**: `rosa-agent pipeline /path/to/CASE --ref-volume post --out-dir DIR`.

### Full ROSA case, post-op CT in a different folder
- **Slicer**: 01 Loader, then import the CT and register it under the
  same module.
- **CLI**: `rosa-agent pipeline /path/to/CASE --ct external.nii.gz
  --ref-volume preopMRI --out-dir DIR`. Inline rigid registration
  (Versor3D + Mattes MI, mirrors BRAINSFit) aligns the CT to the chosen
  ROSA reference.

### CT only (no ROSA folder)
- **Slicer**: 01 Loader → import CT under a custom role → 02 Postop CT
  Localization → de novo Auto Fit.
- **CLI**: `rosa-agent place --ct ct.nii.gz --output DIR --library dixi`
  (mode 1) or `rosa-agent detect ... | rosa-agent contacts ...` for the
  staged form.

### Same patient, two coordinate frames
You have a `.ros` plan in one frame and a CT post-registered to a
different target MRI. The reference volume the .ros points at isn't
even on disk anymore.

- **CLI**: `rosa-agent match-ros --rosa-folder PLAN --ct ct.nii.gz
  --output DIR`. Detector runs on the CT (mode 1), then a
  line-geometry RANSAC recovers the rigid transform between the two
  frames purely from the trajectory bundle and renames each detector
  emission with the plan's electrode name.

### External contacts (other tool / manual annotation)
- **Slicer**: 01 Setup → 02 Contact Import → 03 Atlas Labeling → Export.
- **CLI**: `rosa-agent label contacts.tsv --target-volume ct.nii.gz
  --freesurfer ...` skips detection and goes straight to labeling.

### Re-running detection with new tunables
The pipeline knobs (LoG threshold, walker tolerances, score weights,
band thresholds, etc.) live in two `constants.py` files; for a flag-by-flag
reference see [`PIPELINE_CONSTANTS.md`](PIPELINE_CONSTANTS.md). Most
users never touch them; the calibration covers the cross-scanner
variability we've validated against.

## 8) Validation expectations

When the toolkit reports a fit, what does that mean?

- **`band` column** — `high` = real bolt CC + matched-filter pitch
  signature; `medium` = synthesized / wire-class / no-bolt acceptance;
  `low` = rejected (currently unused).
- **`compound_score`** — composite [0..1] across matched-filter
  correlation, FFT pitch power, tube-likeness, walker bolt-source,
  CC overlap, and seeder confidence. Weights and bands are in
  [`PIPELINE_CONSTANTS.md`](PIPELINE_CONSTANTS.md).
- **per-contact drift (peak-driven)** — slots that drifted >1 mm from
  the model nominal are flagged, so curved shafts or mis-assigned models
  are visible.

The full-dataset regression (`tests/deep_core/
test_pipeline_dataset_contact_pitch_v1.py`) asserts recall + an orphan
budget across the SEEG dataset. CLI / Slicer parity is pinned by
boundary tests so changes can't drift between the two surfaces.

## 9) Where to go next

- Step-by-step Slicer screens: [`SLICER_GUIDE.md`](SLICER_GUIDE.md)
- Every `rosa-agent` subcommand + flag + the Python library API:
  [`cli/README.md`](../cli/README.md)
- Tunable knobs reference: [`PIPELINE_CONSTANTS.md`](PIPELINE_CONSTANTS.md)
- Architecture, package layout, parity invariants:
  [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md)
- Install: [`INSTALL.md`](../INSTALL.md)
