# Install ROSA Helper in 3D Slicer

## Requirements

- 3D Slicer installed (tested with Slicer 5.x)
- Local copy of this repository

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

## Troubleshooting

- Module not visible:
  - Verify path points to `<repo>/RosaHelper` (not just repo root)
  - Check `Settings -> Modules -> Ignore modules` does not contain `rosahelper`
  - Restart Slicer after path changes

- Volumes missing:
  - Confirm `DICOM/<serie_uid>/<volume>.img` exists for each `VOLUME` entry in `.ros`

- Analyze warnings:
  - ITK Analyze deprecation warnings are expected and do not block loading
