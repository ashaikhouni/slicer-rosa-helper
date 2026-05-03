# Developer Guide

Last updated: 2026-05-02

## 1) Architecture Overview

The repo ships two independent surfaces that share one algorithm core:

- **Slicer extension** — thin module UIs (`RosaHelper`,
  `PostopCTLocalization`, `ContactsTrajectoryView`, etc.) +
  `CommonLib/rosa_scene` for scene services. Modules exchange data
  through workflow roles + a single `RosaWorkflow` MRML parameter
  node, not through direct widget-to-widget state.
- **Headless CLI** (`cli/rosa_agent/`) — pure-Python `rosa-agent`
  console script. Same algorithm + IO, no Slicer / VTK / Qt deps.
  Installed via `pip install .` at the repo root; runs as
  `rosa-agent pipeline ...` from any cwd. See `cli/README.md`.

Both surfaces import from the same `CommonLib` packages; the
boundary between Slicer-coupled and headless code is enforced by
package layout (see Section 2) and pinned by subprocess-isolated
boundary tests in `tests/deep_core/test_rosa_detect_no_slicer.py`
and `tests/rosa_core/test_lazy_init.py`.

## 2) Shared Libraries

Primary packages in `CommonLib`:

- **`rosa_core`** — pure-Python domain logic. ROS parser + case
  loader + transforms (LPS/RAS), contact placement
  (`contact_peak_fit`, `contact_fit`), atlas-assignment policy +
  shared atlas-index helpers (`atlas_assignment_policy`,
  `atlas_index`), electrode classifier, registration helper
  (`registration.register_rigid_mi` — rigid Versor3D + Mattes MI
  mirroring BRAINSFit), volume sampling primitives. Lazy
  `__init__.py` (PEP 562) so `from rosa_core.X import Y` for pure
  modules doesn't pull NumPy as a side effect.
- **`rosa_detect`** — pure-Python detection algorithm with a sealed
  public seam:
  - `from rosa_detect.service import run_contact_pitch_v1` is the
    ONLY entry point external code uses.
  - `contracts.DetectedTrajectory` (TypedDict) is the public output
    shape; algorithm-private fields are documented as opaque.
  - Lazy `__init__.py` keeps `import rosa_detect` cheap (only
    pure-stdlib types load eagerly; `service` /
    `contact_pitch_v1_fit` / `guided_fit_engine` import on first
    attribute access).
  - **No Slicer / VTK / Qt deps anywhere in this package** — pinned
    by `tests/deep_core/test_rosa_detect_no_slicer.py`.
- **`shank_core`** — CT helpers: masking, blob candidates, IO.
  Pure-Python.
- **`rosa_scene`** — Slicer-only scene services. Trajectory /
  electrode publication, atlas providers (Slicer-VTK-based),
  `RegistrationService` (BRAINSFit), case-loader scene service,
  layout orchestration, **`sitk_volume_adapter`** (the bridge
  between `vtkMRMLScalarVolumeNode` and the SITK-image inputs
  `rosa_detect` consumes — single place where vtk + slicer get
  imported on the algorithm-call path).
- **`rosa_workflow`** — MRML workflow state + publish / resolve /
  registry / export services.

CLI / Slicer parity surfaces — single source of truth for math
both sides go through:

- `rosa_core.case_loader.compose_rosa_display_ijk_to_ras` +
  `centering_translation_4x4` — ROSA volume centering + display↔reference
  composition. CLI (`load_rosa_volume_as_sitk`) and Slicer
  (`rosa_scene.case_loader_service.center_volume`) both delegate.
- `rosa_detect.service.stamp_ijk_to_ras_on_sitk` — LPS-flip stamping
  for an SITK image's origin / direction.
- `rosa_core.atlas_index.{compute_label_centroids, format_atlas_sample,
  parse_freesurfer_lut}` — atlas-provider math shared by Slicer's
  `atlas_providers.py` and CLI's `atlas_provider_headless.py`.
- `rosa_core.contact_peak_fit.compute_log_sigma1_volume` — single
  LoG σ=1 kernel.

Import rule:

- modules should import shared services only from `CommonLib/...`
- do not reintroduce module-local bridge imports from `RosaHelper/Lib/...`
- Slicer modules MUST NOT inline vtk-from-volume-node logic — go
  through `rosa_scene.sitk_volume_adapter` so all detection inputs
  match the parity invariant (see `feedback_cli_slicer_parity.md`).

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

Slicer modules:

- `01 Loader`: ROSA load, custom import, registration, default role assignment
- `02 Contact Import`: ingest external contact/trajectory files
- `01 Postop CT Localization`: guided-fit and de novo trajectories
- `02 Contacts & Trajectory View`: contact generation, QC, slice alignment
- `01 Atlas Sources`: load/register/publish atlas assets
- `02 Atlas Labeling`: contact-to-atlas assignment
- `03 Navigation Burn`: THOMAS burn and DICOM export
- `01 Export Center`: profile-driven output export

Headless surface:

- `cli/rosa_agent` — `rosa-agent` console script (5 subcommands:
  `load`, `detect`, `contacts`, `label`, `pipeline`). Runs the same
  algorithm against ROSA folders, dataset subjects, or external CTs
  with no Slicer install required. See `cli/README.md`.

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
