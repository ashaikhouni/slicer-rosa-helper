# ROSA Helper

ROSA Helper is a 3D Slicer scripted module for loading ROSA case folders.

It supports:
- parsing `.ros`
- loading ROSA Analyze volumes (`.img/.hdr`) from `DICOM/<serie_uid>/`
- centering each loaded volume
- applying `TRdicomRdisplay` transforms
- composing transform chains using `[IMAGERY_3DREF]`
- loading trajectories as line markups
- selecting electrode models per trajectory (DIXI AM/BM/CM library)
- auto-suggesting an electrode model from trajectory length (closest within 5 mm)
- generating contact fiducials per electrode (one node per trajectory)
- aligning Red/Yellow/Green slice view to a selected trajectory (`long`/`down`) and centering on it
- exporting aligned NIfTI volumes and contact coordinates together for external tools

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

## Quick Start (Module UI)

1. Install the module (see `INSTALL.md`).
2. Open `ROSA Helper` in Slicer.
3. Set case folder (`<case>/` containing `.ros` and `DICOM/`).
4. Click `Load ROSA case`.
5. Open `V1 Contact Labels`:
   - verify auto-selected electrode models
   - adjust model/tip options as needed
   - click `Generate Contact Fiducials`
6. Optional: open `Trajectory Slice View` and align a slice to one trajectory.
   - the selected slice is reoriented and centered on the trajectory midpoint
7. Click `Export Aligned NIfTI + Coordinates`.

Output default folder:
- `<case>/RosaHelper_Export/`

Output files:
- One `.nii.gz` per loaded/aligned volume
- `<prefix>_aligned_world_coords.txt` with contact coordinates and labels

Coordinate columns in export:
- `x_ras,y_ras,z_ras`: Slicer world RAS (matches exported NIfTI scene)
- `x_lps,y_lps,z_lps`: corresponding LPS values

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
- `RosaHelper/Resources/electrodes/dixi_d08_electrodes.json`: bundled electrode model library (AM/BM/CM)
- `tools/`: CLI wrappers for offline conversion/export

## Electrode Model Library

The bundled DIXI D08 electrode library stores per-model geometry needed for
contact placement:
- model id and type (`AM`, `BM`, `CM`)
- number of contacts and grouping
- contact length and diameter
- insulation distances (intra-contact and inter-group)
- total exploration length
- explicit contact-center offsets from the tip (`contact_center_offsets_from_tip_mm`)

Using explicit center offsets avoids ambiguity for grouped designs (`BM`/`CM`)
and keeps contact generation deterministic.

Current bundled models:
- `DIXI-5AM`, `DIXI-8AM`, `DIXI-10AM`, `DIXI-12AM`, `DIXI-15AM`, `DIXI-18AM`
- `DIXI-15BM`
- `DIXI-15CM`, `DIXI-18CM`

Contact display defaults in Slicer:
- per-electrode node naming: `<prefix>_<trajectory>`
- compact point labels: contact index only (`1`, `2`, ...)
- glyph scale: `2.00`
- text scale: `1.50`

## Recommended Interop Workflow (Freeview/Other Tools)

Analyze `.img/.hdr` orientation handling differs across software. To avoid frame
mismatch:
- do not interchange raw ROSA Analyze files + coordinates directly
- use `Export Aligned NIfTI + Coordinates`
- load exported `.nii.gz` and exported coordinate TXT together in downstream tools

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

Create contact-assignment template (trajectory -> model):

```bash
python tools/rosa_export.py contacts-template \
  --ros /path/to/case/s54.ros \
  --out /tmp/s54_assignments.json
```

Generate contacts from assignments:

```bash
python tools/rosa_export.py contacts-generate \
  --ros /path/to/case/s54.ros \
  --assignments /tmp/s54_assignments.json \
  --out-rosa-json /tmp/s54_contacts_rosa.json \
  --out-fcsv /tmp/s54_contacts_ras.fcsv \
  --out-markups /tmp/s54_contacts_ras.mrk.json
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
