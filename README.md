# ROSA Helper

ROSA Helper is a 3D Slicer scripted module for loading ROSA case folders.

It supports:
- parsing `.ros`
- loading ROSA Analyze volumes (`.img/.hdr`) from `DICOM/<serie_uid>/`
- centering each loaded volume
- applying `TRdicomRdisplay` transforms
- composing transform chains using `[IMAGERY_3DREF]`
- loading trajectories as line markups

## Install (Manual Module)

1. Clone or download this repository.
2. In Slicer, open `Settings -> Modules`.
3. Add this path to `Additional module paths`:
   - `<repo>/RosaHelper`
4. Restart Slicer.
5. Open module `ROSA Helper` (category `ROSA`).

## Repository Layout

- `RosaHelper/`: Slicer scripted module
- `RosaHelper/Lib/rosa_core/`: reusable parser/transform/export code (no Slicer dependency)
- `tools/`: CLI wrappers for offline conversion/export

## CLI Usage

List ROS volumes/trajectories:

```bash
python tools/rosa_export.py list --ros /path/to/case/s54.ros
```

Export trajectories to Markups JSON:

```bash
python tools/rosa_export.py markups \
  --ros /path/to/case/s54.ros \
  --root-volume NCAxT1 \
  --out /tmp/s54_trajectories.mrk.json
```

Export trajectories to FCSV:

```bash
python tools/rosa_export.py fcsv \
  --ros /path/to/case/s54.ros \
  --out /tmp/s54_trajectories.fcsv
```

Export one display transform to ITK `.tfm`:

```bash
python tools/rosa_export.py tfm \
  --ros /path/to/case/s54.ros \
  --volume-name post \
  --root-volume NCAxT1 \
  --ras \
  --out /tmp/post_to_ref.tfm
```

## Case Folder Convention

```text
<case_dir>/
  *.ros
  DICOM/
    <serie_uid>/
      <volume>.img
      <volume>.hdr
```

## Optional: Extension Packaging

This repository also contains CMake files for Slicer extension packaging when a full `Slicer_DIR` build tree is available.

## License

See `LICENSE`.
