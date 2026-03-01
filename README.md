# SEEG / ROSA Toolkit for 3D Slicer

Last updated: 2026-03-01

This repository provides a multi-module 3D Slicer toolkit for:
- loading ROSA cases (`.ros` + Analyze volumes)
- localizing trajectories/contacts from planned data or postop CT
- atlas loading + contact labeling (FreeSurfer, THOMAS, WM)
- nucleus burn-in and DICOM export
- profile-based export bundles for downstream tools

## Module Layout (in Slicer)

- `ROSA.01 Setup`
  - `01 Loader`
  - `02 Contact Import`
- `ROSA.02 Localization`
  - `01 Postop CT Localization`
  - `02 Contacts & Trajectory View`
- `ROSA.03 Atlas`
  - `01 Atlas Sources`
  - `02 Atlas Labeling`
  - `03 Navigation Burn`
- `ROSA.04 Export`
  - `01 Export Center`

## Documentation

- Installation: `INSTALL.md`
- End-user workflows: `docs/USER_GUIDE.md`
- Developer architecture and extension patterns: `docs/DEVELOPER_GUIDE.md`
- Atlas provider extension contract: `ATLAS_SOURCE_CONTRACT.md`
- Refactor roadmap: `ROADMAP.md`
- Execution log by phase: `PROGRESS.md`

## Quick Start

1. Follow `INSTALL.md` and restart Slicer.
2. Open `ROSA.01 Setup -> 01 Loader` and load a ROSA case (or custom volumes).
3. Open `ROSA.02 Localization` modules to fit or detect trajectories and generate contacts.
4. Open `ROSA.03 Atlas` modules if atlas labeling/burn workflows are needed.
5. Export from `ROSA.04 Export -> 01 Export Center`.

## Repository Structure

- `RosaHelper/`: loader module UI (`01 Loader`)
- `ContactImport/`: external contacts/trajectories import module
- `PostopCTLocalization/`: guided fit + de novo CT localization
- `ContactsTrajectoryView/`: contact generation/QC/trajectory view tools
- `AtlasSources/`: atlas loading/registration/publishing
- `AtlasLabeling/`: contact-to-atlas assignment
- `NavigationBurn/`: THOMAS burn + DICOM export
- `ExportCenter/`: profile-based export UI
- `CommonLib/`: shared workflow/core/scene libraries used by all modules
- `tools/`: CLI helpers (`rosa_export.py`, `shank_detect.py`)
- `tests/`: pure-python and service tests

## License

See `LICENSE`.
