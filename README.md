# SEEG / ROSA Toolkit for 3D Slicer

Last updated: 2026-03-01

This repository contains a modular 3D Slicer toolkit for SEEG planning, localization, atlas labeling, and export workflows.

It supports:
- ROSA case loading from `.ros` + Analyze image pairs (`.img/.hdr`)
- custom MRI/CT import and base-space registration
- guided and de novo trajectory localization on postop CT
- contact generation with electrode model assignment and QC metrics
- atlas source loading (FreeSurfer, THOMAS, WM) and contact labeling
- THOMAS nucleus burn into MRI with optional DICOM export
- profile-based data export for downstream analysis/reporting

## Slicer Modules

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

## Architecture (High Level)

- Shared libraries live under `CommonLib/` (`rosa_core`, `shank_core`, `rosa_scene`, `rosa_workflow`).
- Modules interoperate through a shared `RosaWorkflow` MRML contract in-scene.
- Pure algorithmic code is kept separate from UI where possible for easier testing and reuse.

## Documentation

- Install: `INSTALL.md`
- User guide: `docs/USER_GUIDE.md`
- Developer guide: `docs/DEVELOPER_GUIDE.md`

## License

- `LICENSE`
