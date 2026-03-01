# Installation

Last updated: 2026-03-01

## Requirements

- 3D Slicer 5.x
- Local clone of this repository
- Optional Python env for CLI tools and tests:
  - `numpy`
  - `SimpleITK`
  - optional: `antspyx`

Optional environment setup:

```bash
cd <repo>
conda env update -f environment.yml
```

## Install into Slicer (manual module path)

1. Open Slicer.
2. Go to `Settings -> Modules`.
3. Add repository root to `Additional module paths`:
   - `<repo>`
4. Restart Slicer.
5. Verify categories are visible:
   - `ROSA.01 Setup`
   - `ROSA.02 Localization`
   - `ROSA.03 Atlas`
   - `ROSA.04 Export`

## First Validation

1. Open `ROSA.01 Setup -> 01 Loader`.
2. Load a ROSA case folder with `.ros` and `DICOM/*/*.img/.hdr`.
3. Open `ROSA.02 Localization -> 02 Contacts & Trajectory View`.
4. Generate contacts for one or more trajectories.
5. Open `ROSA.04 Export -> 01 Export Center` and run `contacts_only` export.

## Optional CLI Validation

```bash
cd <repo>
python3 tools/phase8_sanity.py
```

## Uninstall

1. Remove repository path from `Settings -> Modules -> Additional module paths`.
2. Restart Slicer.
