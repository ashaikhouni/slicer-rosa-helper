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
- generating per-electrode 3D models (shaft + contact segments)
- aligning Red/Yellow/Green slice view to a selected trajectory (`long`/`down`) and centering on it
- recalculating contacts/models from edited trajectory entry/target points without creating duplicate nodes
- V1 postop CT auto-fit workflow (detect candidates, fit selected/all, apply fit)
- exporting aligned NIfTI volumes and contact coordinates together for external tools
- loading FreeSurfer volumetric parcellations (`aparc+aseg`, `aparc.a2009s+aseg`, `aparc.DKTatlas+aseg`, `aseg`, `wmparc`)
- loading THOMAS thalamic nuclei masks and optional burn-to-DICOM workflow
- assigning each contact to atlas labels (THOMAS / FreeSurfer / WM) with nearest-voxel and centroid-distance metrics

This extension also includes `Shank Detect` for CT-only workflows (no `.ros` required):
- threshold-based shank trajectory detection from postop CT artifact
- optional exact trajectory-count mode (otherwise up to 30 detections)
- lock/unlock accepted trajectories across reruns
- side-aware default naming (`R##` / `L##`) with editable names
- per-trajectory electrode assignment and contact generation
- row selection auto-aligns Red slice along the selected trajectory

Existing workflows remain valid. New atlas and burn features are additive.

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
   - keep `Create electrode models` enabled to also create 3D model nodes
   - after moving trajectory line control points, click `Update From Edited Trajectories`
6. Optional: open `Trajectory Slice View` and align a slice to one trajectory.
   - the selected slice is reoriented and centered on the trajectory midpoint
7. Optional: open `Auto Align to Postop CT (V1)`:
   - select postop CT
   - `Detect Candidates`
   - `Fit Selected` or `Fit All`
   - `Apply Fit to Trajectories`
   - contacts/models are regenerated from the fitted trajectories
   - you can still manually edit entry/target points or electrode models afterward and click `Update From Edited Trajectories`
8. Click `Export Aligned NIfTI + Coordinates/QC`.
   - if atlas assignment was run, export also includes atlas-label CSV
9. Optional: open `Atlas Contact Labeling (V1)`:
   - click `Refresh Atlas Sources`
   - choose FreeSurfer / THOMAS / WM sources
   - click `Assign Contacts to Atlas`
   - export again to write/update atlas assignment CSV

## Quick Start (Shank Detect)

Use this when you only have a CT with electrode artifact and no ROSA `.ros` metadata.

1. Open module `Shank Detect`.
2. Select postop CT volume.
3. Set detection parameters (threshold, inlier radius, min length, min inliers).
   - optional head-mask gating (`Use head mask filter`) to suppress external wire artifacts
   - use `Min metal depth` / `Max metal depth` to keep only metal points at a plausible depth from the outer head surface
   - defaults: `Min metal depth = 5 mm`, `Max metal depth = 220 mm`
   - optional model-template scoring (`Enable model-template scoring`) to rank/reject lines by expected contact pattern
   - optional mask visualization (`Show metal/head masks in slice views`) to inspect thresholded metal and head gating
4. Click `Preview Masks` to inspect the gated metal mask before running trajectory extraction.
5. Optional: click `Show Depth Curve` to inspect the survival curve `N(depth >= t)` for all head-gated metal points vs kept points.
6. Optional: enable `Use exact trajectory count` and set count.
7. Click `Detect Trajectories`.
   - locked rows are preserved on rerun
   - unlocked rows are replaced by newly detected lines
8. Edit trajectory names if needed; select a row to auto-align Red view along it.
9. Assign electrode model per row (or use `Apply model to all`).
10. Click `Generate Contacts`.
11. Optional: click `Reset Ax/Cor/Sag` to restore standard slice orientations.

Output default folder:
- `<case>/RosaHelper_Export/`

Output files:
- One `.nii.gz` per loaded/aligned volume
- `<prefix>_aligned_world_coords.txt` with contact coordinates and labels
- `<prefix>_planned_trajectory_points.csv` with planned entry/target points
- `<prefix>_qc_metrics.csv` with per-trajectory QC metrics
- `<prefix>_atlas_assignment.csv` with per-contact atlas assignment (when available)

Coordinate columns in export:
- `x_ras,y_ras,z_ras`: Slicer world RAS (matches exported NIfTI scene)
- `x_lps,y_lps,z_lps`: corresponding LPS values

QC CSV columns:
- `trajectory`
- `entry_radial_mm`
- `target_radial_mm`
- `mean_contact_radial_mm`
- `max_contact_radial_mm`
- `rms_contact_radial_mm`
- `angle_deg`
- `matched_contacts`

## Contact Localization Modes

### Manual Contact Localization

Use this when you want full manual control.

1. Load case and trajectories.
2. In `V1 Contact Labels`, set electrode model and tip options per trajectory.
3. Click `Generate Contact Fiducials`.
4. If you edit trajectory entry/target points or change electrode model/tip settings, click `Update From Edited Trajectories`.

`Generate` is typically used the first time.
`Update` is used after edits and recomputes contacts/models in-place.

### Auto Contact Localization (Postop CT V1)

Use this to initialize trajectory placement from postop CT hyperdense contacts.

1. Select postop CT in `Auto Align to Postop CT (V1)`.
2. Run `Detect Candidates`.
3. Run `Fit Selected` or `Fit All`.
4. Run `Apply Fit to Trajectories`.

`Apply Fit to Trajectories` updates trajectory lines and regenerates contacts/models.
After auto-fit, you can continue manual refinement:
- change electrode model assignment
- edit entry/target points
- click `Update From Edited Trajectories` to recompute final contacts/models

## Trajectory QC Metrics

The `Trajectory QC Metrics` panel is populated automatically whenever contacts are
generated or updated. No separate compute button is required.

QC panel is disabled when:
- no generated contacts exist
- planned trajectories (`Plan_*`) are not available

Metrics are computed per trajectory by comparing:
- planned line (`Plan_<trajectory>`)
- final line (`<trajectory>`)
- planned contact centers (from selected model + planned line)
- final contact centers (manual or auto-fit result)

## Atlas Contact Labeling (V1)

`Atlas Contact Labeling (V1)` assigns each generated contact to nearest atlas voxels.

Inputs:
- FreeSurfer volumetric atlas (`FSVOL_*`, for example `FSVOL_aparc+aseg`)
- optional THOMAS segmentation atlas
- optional white-matter atlas (`FSVOL_wmparc`)

Workflow:
1. Generate contacts first.
2. Load FS parcellation volume(s) and optional THOMAS segmentations.
3. Open `Atlas Contact Labeling (V1)` and click `Refresh Atlas Sources`.
4. Select sources from dropdowns and click `Assign Contacts to Atlas`.
5. Export bundle to write `<prefix>_atlas_assignment.csv`.

CSV includes:
- per-atlas labels and distances (`thomas_*`, `freesurfer_*`, `wm_*`)
- `closest_*` columns:
  - nearest structure across all selected atlas sources
  - includes source atlas and both nearest-voxel / centroid distances
- `primary_*` columns:
  - final assignment used by workflow
  - with `Prefer THOMAS when available` enabled: uses THOMAS if present, otherwise closest overall
  - with it disabled: equals `closest_*`

Primary assignment rule:
- chooses the closest voxel match across selected atlas sources
- if `Prefer THOMAS when available` is enabled, THOMAS assignment is used whenever a THOMAS label is present

THOMAS labeling behavior:
- generic whole-thalamus segments are excluded from nearest-label assignment when nuclei are available
- this keeps labels specific to nuclei (for example `CM`, `MD-Pf`) instead of collapsing to `LEFT_THALAMUS` / `RIGHT_THALAMUS`

## FreeSurfer Integration (V1)

`FreeSurfer Integration (V1)` supports aligning a recon-all MRI/surfaces to the
ROSA base frame in the same Slicer scene.

Workflow:
1. Load ROSA case (ROSA base/reference volume is auto-selected when available).
2. Add the exact MRI volume used for `recon-all` to the Slicer scene using Slicer's standard
   `Add Data` workflow (not through ROSA Helper). This is the moving image.
3. In `FreeSurfer Integration (V1)`:
   - set `ROSA base volume` (fixed) and `FreeSurfer MRI` (moving)
   - click `Register FS MRI -> ROSA` (BRAINSFit rigid registration)
4. Set FreeSurfer subject path:
   - subject root containing `surf/` and `label/`, or
   - direct `surf/` path inside your segmentation folder.
5. Optional: enable `Load annotation scalars`, choose an annotation
   (`aparc`, `aparc.a2009s`, `aparc.DKTatlas`, or custom), and optionally set
   a LUT file (for example `FreeSurferColorLUT.txt`).
6. Choose surface set (`pial`, `white`, `pial+white`, `inflated`) and click
   `Load FreeSurfer Surfaces`.
7. Keep `Apply FS->ROSA transform` enabled to bring surfaces into ROSA space.
   Optionally harden surface transforms.
8. Optional volumetric atlas loading from `mri/`:
   - click `Refresh` next to `Parcellation volume`
   - choose one entry (`aparc+aseg.mgz`, `aparc.a2009s+aseg.mgz`, `aparc.DKTatlas+aseg.mgz`, `aseg.mgz`, `wmparc.mgz`) or `all available`
   - click `Load Parcellation Volumes`
   - keep `Apply FS->ROSA transform to parcellations` enabled for ROSA-space alignment
   - keep `Apply LUT to parcellation volumes` enabled for atlas colorized slice display
   - enable `Create 3D geometry from parcellations` to generate closed-surface segmentation nodes in 3D
9. To use parcellations for contact labeling:
   - load at least one `FSVOL_*` atlas volume
   - then run `Atlas Contact Labeling (V1)` assignment

Notes:
- BRAINSFit is expected to be available in standard Slicer installs.
- Surface loading depends on Slicer model IO support for the selected FreeSurfer files.
- Annotation fallback order is:
  `nibabel` `.annot` reader -> SlicerFreeSurfer extension reader (if installed) -> `mris_convert`.
- `nibabel` path does not require a FreeSurfer license.
- Annotation color table priority (same for nibabel and extension paths):
  user-selected LUT -> `FreeSurferLabels` node -> bundled LUT
  (`RosaHelper/Resources/freesurfer/FreeSurferColorLUT20120827.txt`).
- If direct `.pial/.white/.inflated` loading fails, ROSA Helper attempts fallback conversion
  through `mris_convert` (FreeSurfer) and loads the converted VTK surface.
- `mris_convert` fallback may require a valid FreeSurfer license.
- Parcellation volumes are loaded from `mri/` as scalar label volumes named `FSVOL_*`.
- Hardening parcellation transforms is optional and off by default to avoid unintended
  interpolation of label values.
- The same LUT selector is reused for surface annotations and parcellation volume coloring.

Quick checklist:
- The MRI selected in `FreeSurfer MRI` must be the same scan used to build recon-all surfaces.
- Registration (`Register FS MRI -> ROSA`) should be run before loading surfaces.
- The folder selected in `FreeSurfer subject` must contain `surf/` (and `label/` for `.annot` overlays).

## THOMAS Integration (V1)

`THOMAS Thalamus Integration (V1)` supports loading THOMAS left/right structure
masks and bringing them into ROSA base space.

Workflow:
1. Load ROSA case in `ROSA Helper`.
2. Add the MRI used to generate THOMAS (via Slicer's standard `Add Data`).
3. In `THOMAS Thalamus Integration (V1)`:
   - set `ROSA base volume` (fixed) and `THOMAS MRI` (moving)
   - click `Register THOMAS MRI -> ROSA`
4. Set `THOMAS output dir` to the THOMAS subject folder that contains `left/` and `right/`.
5. Click `Load THOMAS Thalamus Masks`.
6. Keep `Apply THOMAS->ROSA transform` enabled to align segmentations to ROSA space.
7. Optional one-click burn workflow in the same panel:
   - optionally set `Nav MRI DICOM dir` and click `Import DICOM MRI`
     (this registers imported DICOM MRI to selected ROSA base and hardens it)
   - set `Burn input MRI`
   - choose `Nucleus side` (`Left`, `Right`, `Both`) and `Nucleus` (for example `CM`)
   - set burn fill value and output volume name
   - click `Register + Burn Nucleus`
8. Optional one-click DICOM export:
   - set `DICOM export dir` and `DICOM series description`
   - click `Register + Burn + Export DICOM` to write classic slice-wise DICOM files
9. Optional atlas labeling integration:
   - after THOMAS masks are loaded, they appear in `Atlas Contact Labeling (V1)` as a source

Notes:
- Loader scans only `left/` and `right/` mask files and skips helper/cropped/resampled/full outputs.
- THOMAS MRI registration is rigid (`BRAINSFit`) and should be reviewed in slice views before downstream export.
- `Register + Burn Nucleus` can auto-run rigid registration from `THOMAS MRI` to `Burn input MRI`
  before creating a burned scalar volume.

## Burn THOMAS Segments Into DICOM (ROSA Navigation)

ROSA imports DICOM. If you need a navigation MRI with thalamus labels burned in:

1. Load the original navigation MRI from DICOM (recommended base for export metadata).
2. Ensure THOMAS segmentation is aligned to that MRI space (via THOMAS registration and transform application/hardening).
3. Open `Segment Editor` and choose the THOMAS segmentation node.
4. Use `Mask volume`:
   - `Input Volume`: the DICOM-based MRI to export
   - `Output Volume`: create a new volume
   - operation:
     - `Fill inside` to highlight selected structure(s), or
     - `Fill outside` to keep only ROI
5. Click `Apply`.
6. Export the new scalar volume as DICOM:
   - in `Data`/Subject Hierarchy, right-click output volume
   - `Export to DICOM...` as a new series (do not overwrite original MRI)
7. Re-import exported DICOM in Slicer and verify alignment/intensity before sending to ROSA.

Module shortcut:
- `Register + Burn Nucleus` creates the burned scalar volume directly from selected
  THOMAS nucleus and side, without manual Segment Editor steps.
- `Auto-register THOMAS MRI -> Burn input` is an advanced fallback and is off by default
  when Burn input MRI is already aligned to ROSA.
- `Register + Burn + Export DICOM` additionally exports the burned result to a classic
  DICOM scalar volume series (one file per slice) under the selected export directory.
- DICOM export remains the same final step via Subject Hierarchy (`Export to DICOM...`).

Important:
- `Mask volume` applies to the currently selected segment. To burn multiple structures, combine segments first or run per segment.
- Use a fill value outside background MRI range for clear visibility (for many T1 scans, bright values around `1000-1500` work well).

## Install (Manual Module)

1. Clone or download this repository.
2. In Slicer, open `Settings -> Modules`.
3. Add this path to `Additional module paths`:
   - `<repo>/RosaHelper`
4. Restart Slicer.
5. Open module `ROSA Helper` (category `ROSA`).
6. Open module `Shank Detect` (category `ROSA`) for CT-only shank detection.

## Repository Layout

- `RosaHelper/`: Slicer scripted module
- `ShankDetect/`: Slicer scripted module for CT-only trajectory detection and contact generation
- `RosaHelper/Lib/rosa_core/`: reusable parser/transform/export code (no Slicer dependency)
- `RosaHelper/Lib/rosa_core/assignments.py`: reusable trajectory-length/model-suggestion helpers
- `RosaHelper/Lib/rosa_core/qc.py`: reusable planned-vs-final QC metric computation
- `RosaHelper/Lib/rosa_slicer/`: Slicer scene/services + widget mixins (`freesurfer_service.py`, `trajectory_scene.py`, `widget_mixin.py`)
- `RosaHelper/Resources/electrodes/dixi_d08_electrodes.json`: bundled electrode model library (AM/BM/CM)
- `RosaHelper/Resources/freesurfer/FreeSurferColorLUT20120827.txt`: bundled FreeSurfer annotation LUT fallback
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
- use `Export Aligned NIfTI + Coordinates/QC`
- load exported `.nii.gz` and exported coordinate TXT together in downstream tools

## CLI Usage

`tools/shank_detect.py` can run with:
- standard Python if `numpy` and `SimpleITK` are installed
- or Slicer Python (`Slicer --python-script ...`) if you prefer to reuse Slicer's bundled environment

Optional dependency:
- `antspyx` is only needed for ANTs-based segmentation integration workflows.

List ROS volumes/trajectories:

```bash
python tools/rosa_export.py list --ros /path/to/case/case.ros
```

Export trajectories to Markups JSON:

```bash
python tools/rosa_export.py markups \
  --ros /path/to/case/case.ros \
  --root-volume NCAxT1 \
  --out /tmp/case_trajectories.mrk.json
```

Export trajectories to FCSV:

```bash
python tools/rosa_export.py fcsv \
  --ros /path/to/case/case.ros \
  --out /tmp/case_trajectories.fcsv
```

Export one display transform to ITK `.tfm`:

```bash
python tools/rosa_export.py tfm \
  --ros /path/to/case/case.ros \
  --volume-name post \
  --root-volume NCAxT1 \
  --ras \
  --out /tmp/post_to_ref.tfm
```

Create contact-assignment template (trajectory -> model):

```bash
python tools/rosa_export.py contacts-template \
  --ros /path/to/case/case.ros \
  --out /tmp/case_assignments.json
```

Generate contacts from assignments:

```bash
python tools/rosa_export.py contacts-generate \
  --ros /path/to/case/case.ros \
  --assignments /tmp/case_assignments.json \
  --out-rosa-json /tmp/case_contacts_rosa.json \
  --out-fcsv /tmp/case_contacts_ras.fcsv \
  --out-markups /tmp/case_contacts_ras.mrk.json
```

Preview CT metal/head-depth masks for shank detection:

```bash
python tools/shank_detect.py preview-masks \
  --ct /path/to/postop_ct.nii.gz \
  --out-dir /tmp/shank_preview \
  --metal-threshold-hu 1800 \
  --use-head-mask \
  --head-mask-threshold-hu -300 \
  --head-mask-close-mm 2.0 \
  --min-metal-depth-mm 5.0 \
  --max-metal-depth-mm 220
```

Run full CT trajectory detection (no Slicer UI):

```bash
python tools/shank_detect.py detect \
  --ct /path/to/postop_ct.nii.gz \
  --out-dir /tmp/shank_detect \
  --metal-threshold-hu 1800 \
  --use-head-mask \
  --head-mask-threshold-hu -300 \
  --head-mask-close-mm 2.0 \
  --min-metal-depth-mm 5.0 \
  --max-metal-depth-mm 220 \
  --max-lines 30
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
