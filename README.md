# ROSA Helper

ROSA Helper is a 3D Slicer scripted module for loading ROSA case folders.

It supports:
- parsing `.ros`
- loading ROSA Analyze volumes (`.img/.hdr`) from `DICOM/<serie_uid>/`
- centering each loaded volume
- applying `TRdicomRdisplay` transforms
- composing transform chains using `[IMAGERY_3DREF]`
- loading trajectories as line markups

## ROS File Structure (What We Read)

ROSA `.ros` files are tokenized text files. Each section starts with a bracketed
token, for example:

```text
[TRdicomRdisplay]
<16 matrix values>
[VOLUME]
\DICOM\<serie_uid>\<volume_name>
[IMAGERY_3DREF]
<display index>
```

Relevant tokens used by this project:
- `TRdicomRdisplay`: 4x4 matrix for one display volume
- `VOLUME`: display volume path/name
- `IMAGERY_NAME`: human-readable display label
- `SERIE_UID`: series folder key under `DICOM/`
- `IMAGERY_3DREF`: parent display index for registration chaining
- `TRAJECTORY` and `ELLIPS`: trajectory name + two 3D points

## How ROSA Helper Interprets Transforms

For each display `i`, this project interprets:
- `TRdicomRdisplay(i)` as transform from display `i` to parent display `IMAGERY_3DREF(i)`

Then it composes parent chains into the chosen root display frame:
- `T(i->root) = T(parent->root) * T(i->parent)`

Finally for Slicer application:
1. Convert matrix from LPS to RAS.
2. Optionally invert (module checkbox).
3. Apply to centered loaded volume.

Default root display is the first display in `.ros` order unless a reference
volume is explicitly selected.

Trajectories are interpreted as ROSA/LPS points and converted to RAS for Slicer
markups.

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
