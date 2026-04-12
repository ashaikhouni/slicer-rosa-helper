# Developer Guide

Last updated: 2026-03-01

## 1) Architecture Overview

The extension is organized as:
- thin module UIs (`RosaHelper`, `PostopCTLocalization`, `ContactsTrajectoryView`, etc.)
- shared logic/services in `CommonLib`
- shared scene contract via one `RosaWorkflow` MRML parameter node

Core design rule:
- modules exchange data through workflow roles and registry tables, not through direct widget-to-widget state.

## 2) Shared Libraries

Primary packages in `CommonLib`:
- `rosa_core`: pure-python domain logic (parser, transforms, contact generation, QC, exporters)
- `shank_core`: pure-python CT shank detection utilities (masking, blob extraction)
- `shank_engine`: detection pipeline framework — see `CommonLib/shank_engine/README.md`
- `rosa_workflow`: workflow state/publish/resolve/registry/export services
- `rosa_scene`: Slicer scene services (trajectory/electrode/atlas helpers)
  - includes shared layout orchestration via `layout_service.py`

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

## 8) Adding New Functionality

Recommended sequence:
1. Define workflow inputs/outputs and role/table impact.
2. Add or update pure logic in `rosa_core`/`shank_core` if possible.
3. Add scene/workflow service hooks in `rosa_scene`/`rosa_workflow`.
4. Wire UI controls in exactly one module.
5. Add tests for pure logic and role/table schema stability.
6. Update docs (`USER_GUIDE.md`, this file, and contract docs if needed).
