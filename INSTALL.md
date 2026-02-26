# Install ROSA Helper in 3D Slicer

## Requirements

- 3D Slicer installed (tested with Slicer 5.x)
- Local copy of this repository
- Python environment for CLI tools:
  - `numpy`
  - `SimpleITK`
- Optional for ANTs-based segmentation integration workflows:
  - `antspyx` (ANTsPy)

Optional environment setup command:

```bash
conda env update -f environment.yml
```

Optional ANTsPy install (only if you use ANTs-based segmentation integration):

```bash
pip install antspyx
```

## Install Steps

1. Open Slicer.
2. Go to `Settings -> Modules`.
3. In `Additional module paths`, add:
   - `<repo>/RosaHelper`
4. Click `OK` and restart Slicer.
5. In the module selector, open category `ROSA` and select `ROSA Helper`.

## First Load Test

1. Open `ROSA Helper`.
2. Select a case folder containing:
   - one `.ros`
   - `DICOM/` with per-series subfolders containing `.img/.hdr`
3. Click `Load ROSA case`.

## Basic Usage Flow

1. In `V1 Contact Labels`, review trajectory rows and electrode model suggestions.
2. Adjust model/tip settings if needed.
3. Click `Generate Contact Fiducials`.
4. Optional: use `Trajectory Slice View` to align Red/Yellow/Green along one trajectory.
5. Click `Export Aligned NIfTI + Coordinates`.

Optional FreeSurfer volumetric atlas load:
- In `FreeSurfer Integration (V1)`, select FreeSurfer subject dir, refresh parcellation list,
  and load `aparc*+aseg` / `aseg` / `wmparc` volumes.
- Keep `Apply FS->ROSA transform to parcellations` enabled when you need ROSA-space labels.
- Keep `Apply LUT to parcellation volumes` enabled for colorized atlas view in slices.
- Optional: enable `Create 3D geometry from parcellations` for 3D closed-surface display.

Optional contact-to-atlas labeling:
- Open `Atlas Contact Labeling (V1)`.
- Select FreeSurfer / THOMAS / WM atlas sources and click `Assign Contacts to Atlas`.
- Export bundle to include `<prefix>_atlas_assignment.csv`.

Default export folder:
- `<case>/RosaHelper_Export`

Exports:
- aligned `.nii.gz` volumes
- `<prefix>_aligned_world_coords.txt` (contact coordinates)

## THOMAS to DICOM (Optional)

If you want a ROSA-importable DICOM with THOMAS labels burned into MRI:

1. Load the original MRI DICOM series in Slicer.
2. Register THOMAS MRI -> ROSA base in `ROSA Helper`.
3. Load THOMAS masks from the THOMAS output folder (`left/` and `right/`).
4. Use the THOMAS panel burn workflow:
   - optional `Nav MRI DICOM dir` -> `Import DICOM MRI` (auto-registers imported MRI to ROSA base)
   - choose `Burn input MRI`, side (`Left/Right/Both`), and nucleus (for example `CM`)
   - click `Register + Burn Nucleus` to create output scalar volume
5. For one-click export, set `DICOM export dir` + series description and click
   `Register + Burn + Export DICOM`.
6. Manual fallback: in Subject Hierarchy, right-click the burned volume and
   `Export to DICOM...` as a new series.

Notes:
- Use the DICOM-based MRI as `Input Volume` for best metadata compatibility.
- Manual fallback remains available with `Segment Editor -> Mask volume`.

## Troubleshooting

- Module not visible:
  - Verify path points to `<repo>/RosaHelper` (not just repo root)
  - Check `Settings -> Modules -> Ignore modules` does not contain `rosahelper`
  - Restart Slicer after path changes

- Volumes missing:
  - Confirm `DICOM/<serie_uid>/<volume>.img` exists for each `VOLUME` entry in `.ros`

- Analyze warnings:
  - ITK Analyze deprecation warnings are expected and do not block loading

- Coordinates look wrong in external viewers:
  - Use the aligned export outputs (`.nii.gz` + aligned coordinate TXT), not raw Analyze files
  - Confirm the external viewer loads the exported NIfTI, not original `.img`
