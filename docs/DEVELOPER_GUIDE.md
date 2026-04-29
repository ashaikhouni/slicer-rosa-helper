# Developer Guide

Last updated: 2026-04-19

## 1) Architecture Overview

The extension is organized as:
- thin module UIs (`RosaHelper`, `PostopCTLocalization`, `ContactsTrajectoryView`, etc.)
- shared logic/services in `CommonLib`
- shared scene contract via one `RosaWorkflow` MRML parameter node

Core design rule:
- modules exchange data through workflow roles and registry tables, not through direct widget-to-widget state.

## 2) Shared Libraries

Primary packages in `CommonLib`:
- `rosa_core`: pure-python domain logic (parser, transforms, contact
  generation, peak-driven contact detection via `contact_peak_fit`,
  QC, exporters)
- `shank_core`: pure-python CT shank detection utilities (masking, blob extraction)
- `shank_engine`: detection pipeline framework — see `CommonLib/shank_engine/README.md`
- `rosa_workflow`: workflow state/publish/resolve/registry/export services
- `rosa_scene`: Slicer scene services (trajectory/electrode/atlas helpers)
  - includes shared layout orchestration via `layout_service.py`
  - long-axis slice view (`align_slice_to_trajectory(mode='long')`)
    auto-fits FOV to the trajectory span so the whole shank is visible

Key `rosa_core` entry points added this session:
- `detect_contacts_on_axis`, `sample_axis_profile`, `detect_peaks_1d`,
  `fit_best_electrode`, `candidate_ids_for_vendors`,
  `ras_contacts_to_contact_records`, `PeakFitResult` —
  all exported from `rosa_core` for the Contacts & Trajectory View
  module's peak-driven mode. Reuses the LoG σ=1 volume that Auto Fit
  stashes in the scene (`<CT>_ContactPitch_LoG_sigma1`), falls back
  to SimpleITK recomputation when the cached volume is absent.

Import rule:
- modules should import shared services only from `CommonLib/...`
- do not reintroduce module-local bridge imports from `RosaHelper/Lib/...`

## 3) MRML Workflow Contract

Canonical state node:
- `vtkMRMLScriptedModuleNode` named `RosaWorkflow`

Common role families:
- defaults: `BaseVolume`, `PostopCT`
- trajectories: `PlannedTrajectoryLines`, `WorkingTrajectoryLines`, grouped producer roles
- contacts/models: `ContactFiducials`, shaft/contact model roles
- atlas: FreeSurfer/THOMAS/WM source roles
- tables: assignments, QC, atlas assignment

Registry tables:
- `RosaWorkflow_ImageRegistry`
- `RosaWorkflow_TransformRegistry`

Provenance attributes used on managed nodes:
- `Rosa.Managed=1`
- `Rosa.Source`
- `Rosa.Role`
- `Rosa.Space`
- `Rosa.ContextId`

## 4) Module Responsibilities

- `01 Loader`: ROSA load, custom import, registration, default role assignment
- `02 Contact Import`: ingest external contact/trajectory files
- `01 Postop CT Localization`: guided-fit and de novo trajectories
- `02 Contacts & Trajectory View`: contact generation, QC, slice alignment
- `01 Atlas Sources`: load/register/publish atlas assets
- `02 Atlas Labeling`: contact-to-atlas assignment
- `03 Navigation Burn`: THOMAS burn and DICOM export
- `01 Export Center`: profile-driven output export

Keep responsibilities separate. If a feature spans modules, connect through workflow roles.

## 5) Atlas Extension Pattern

To add a new atlas source:
1. Implement provider using typed interface in `CommonLib/rosa_scene/atlas_provider_types.py`.
2. Register provider in `CommonLib/rosa_scene/atlas_provider_registry.py`.
3. Add source loading/publishing in `AtlasSources`.
4. Ensure `AtlasLabeling` can select and run provider.
5. Add tests for provider readiness and assignment schema stability.

## 6) Coding Standards

Docstrings:
- every module class and logic class should have a one-line responsibility docstring
- non-trivial methods should include concise behavior docstrings

Comments:
- add comments only for non-obvious logic, transforms, or coordinate assumptions
- avoid redundant comments that restate code

Dates:
- keep "Last updated" headers in root docs current
- for major interface/contract changes, update this guide and the user guide in the same PR

Compatibility:
- preserve workflow role names unless migration is planned
- if schema changes, update exporters and tests in the same change

## 7) Testing

Pure python tests:

```bash
cd <repo>
<python-env>/bin/python -m unittest discover -s tests/rosa_core -p "test_*.py"
<python-env>/bin/python -m unittest discover -s tests/shank_core -p "test_*.py"
```

Optional pytest run:

```bash
cd <repo>
PYTHONPATH=<python-env>/lib/python3.10/site-packages:$PYTHONPATH python3 -m pytest tests/rosa_core tests/shank_core tests/rosa_scene -q
```

Dataset-gated regressions (need a postop CT dataset on disk; set
`ROSA_SEEG_DATASET` to the top-level directory):

- `tests/deep_core/test_pipeline_dataset_contact_pitch_v1.py` — Auto
  Fit regression. Quick subject-level gates (`test_T22`, `test_T2`,
  `test_T2_auto_strategy`, ~15 s combined) plus the slow full-dataset
  gate `test_dataset_full` (22 subjects, ~70 s) that asserts recall
  and the orphan budget. The full-dataset gate is the regression net for
  refactor work; do not relax its asserts to make a refactor pass.
- `tests/rosa_core/test_contact_peak_fit.py` — peak-driven contact
  detection. Synthetic unit tests + dataset tests asserting T22
  median per-contact error ≤ 1.75 mm and T2 ≤ 1.5 mm vs. the
  subject's `contacts.tsv` ground truth.

For the canonical project state, current pipeline version, and
score-band policy, see [`HANDOFF.md`](HANDOFF.md).

Probes (diagnostic, not asserted):

- `tests/deep_core/probe_contact_peak_filters.py` — compares LoG σ=1,
  raw CT HU, and white top-hat as on-axis contact signals.
- `tests/deep_core/probe_contact_peak_engine.py` — end-to-end
  engine report per GT shank (model id, n matched, residual).

## 8) Adding New Functionality

Recommended sequence:
1. Define workflow inputs/outputs and role/table impact.
2. Add or update pure logic in `rosa_core`/`shank_core` if possible.
3. Add scene/workflow service hooks in `rosa_scene`/`rosa_workflow`.
4. Wire UI controls in exactly one module.
5. Add tests for pure logic and role/table schema stability.
6. Update docs (`USER_GUIDE.md`, this file, and contract docs if needed).
