# Slicer Guide

Last updated: 2026-05-11

A per-module reference for the 3D Slicer extension. For the conceptual
end-to-end workflow that's shared with the headless CLI, start with
[`USER_GUIDE.md`](USER_GUIDE.md). For installation, see
[`../INSTALL.md`](../INSTALL.md). For the headless equivalent of any
module, see [`../cli/README.md`](../cli/README.md).

## 1) Typical Workflow

1. Load data in `ROSA.01 Setup`.
2. Localize trajectories and generate contacts in `ROSA.02 Localization`.
3. Optionally load atlas data and assign labels in `ROSA.03 Atlas`.
4. Optionally burn nucleus labels for navigation MRI in `ROSA.03 Atlas -> 03 Navigation Burn`.
5. Export selected outputs from `ROSA.04 Export -> 01 Export Center`.

## 2) Setup Modules

## `01 Loader`

Use this when you have a ROSA folder with `.ros` and `DICOM/<serie_uid>/<volume>.img/.hdr`.

Main actions:
- Load ROSA case.
- Import custom MRI/CT volumes.
- Register imported volumes to base volume.
- Assign default roles (`BaseVolume`, `PostopCT`).

Notes:
- ROSA trajectories are parsed from `.ros` and published into workflow roles.
- Volume/transform provenance is tracked in workflow registries.

## `02 Contact Import`

Use this when contacts or trajectories come from external tools.

Supported formats:
- Contacts: `CSV`, `TSV`, `XLSX`, `POM`
- Trajectories: `CSV`, `TSV`, `XLSX`

Required reference:
- A reference volume is required (from scene or loaded via module).

Required schemas:
- Contacts: `trajectory_name,index,x,y,z`
- Trajectories: `name,ex,ey,ez,tx,ty,tz`

Optional fields:
- Contacts `label`

Coordinate metadata (set in UI):
- coordinate system: `RAS` or `LPS`
- coordinate type: `world` or `voxel`
- units: `mm` or `m`

## 3) Localization Modules

## `01 Postop CT Localization`

Three modes:

- Auto Fit: detect trajectories directly from the postop CT
  (`contact_pitch_v1` pipeline; the production CT-only detector)
- Guided Fit: refine existing (planned) trajectories using the postop CT
- Manual Fit: adopt trajectories drawn by hand in the scene

Key behavior:

- Trajectories are grouped by source (`auto_fit`, `guided_fit`, `manual`,
  `imported_rosa`, `imported_external`, `planned_rosa`).
- Outputs publish to shared workflow roles (`AutoFitTrajectoryLines`,
  `GuidedFitTrajectoryLines`, `ManualTrajectoryLines`, ...).
- Active trajectory source is shared with the Contacts module.

## `02 Contacts & Trajectory View`

Main actions:
- Select trajectory source.
- Assign electrode model/tip options per trajectory.
- Check/uncheck trajectories in `Use` column to control which trajectories generate contacts.
- Choose a **Detection mode**: *Model-driven* (nominal offsets) or
  *Peak-driven* (CT image peaks).
- Generate or update contacts/models.
- View QC metrics and align slice views along a selected trajectory.
- Optional focus layout mode: top FourUp + bottom `long`/`down`
  trajectory views. The blue long-axis view auto-fits its field of
  view to the entire trajectory (entry → deep tip, 1.2× margin); the
  purple down-axis view stays centered on the focus point.

Detection modes:
- *Model-driven (nominal)*: contacts are placed along the fitted
  trajectory at the assigned electrode model's nominal pitch.
  Use this when you trust the model assignment and the line fit.
- *Peak-driven (CT peaks)*: contacts are detected from the postop
  CT by sampling LoG σ=1 along the trajectory axis with a 2 mm disk,
  picking peaks, and matching the peak pattern against the
  electrode library. Contacts are emitted at the detected peak
  positions — so a curved shaft, a drifted contact, or a wrong model
  assignment is visible in the output. Reuses the Auto-Fit-stashed
  `<CT>_ContactPitch_LoG_sigma1` scalar volume when present; computes
  it on-the-fly otherwise.
- When a model is assigned in the table, peak-driven matching is
  restricted to that model. Leaving the model blank lets the engine
  pick the best-fitting model from the library (filtered by the
  *Default model* vendor). Peak-driven falls back to model-driven
  synthesis per trajectory when the engine can't find enough peaks.
- Per-slot drift between peak-detected and nominal positions is
  logged; slots drifting more than 1 mm are flagged so you can spot
  curved shafts or mis-assigned models.

Important:
- `Generate`/`Update` only operate on checked rows.
- At least one checked trajectory with a valid model is required
  (or a valid *Default model* vendor for blank-model peak-driven fits).
- Peak-driven mode requires a `PostopCT` workflow role. Run Auto Fit
  first or assign the post-op CT via the Focus view selector.

## 4) Atlas Modules

## `01 Atlas Sources`

Loads atlas data (FreeSurfer parcellations + surfaces, THOMAS thalamic
nucleus masks) and registers them to the base / postop volume so atlas
labels can be sampled at contact positions downstream. Outputs are
published into the workflow as scene nodes + registry entries — the
**02 Atlas Labeling** module reads those roles, you don't have to
re-select files there.

### Top-level controls

- **Refresh Workflow Inputs** — syncs the module's dropdowns with the
  current scene. Run this after loading a new base volume in **01
  Loader**, or whenever the FS / THOMAS selectors look empty. Also
  rescans the FreeSurfer subject folder for available parcellations.
- **Status log** (read-only) — every operation prints here with a
  prefix: `[fs] ...`, `[thomas] ...`, `[refresh] ...`. Read this when
  something didn't behave as expected.

The body is a 3-tab notebook: **FreeSurfer**, **THOMAS**, **Registry**.

### Tab 1 — FreeSurfer

Three independent sub-flows in the same tab: register the FS MRI to
your base volume, load parcellation volumes, load cortical surfaces.
Sub-flows can run in any order *after* registration; parcellations and
surfaces both consume the FS→ROSA transform when applying it.

#### A. Register the FS MRI to your base volume

1. **ROSA base volume** — the fixed reference (typically the postop CT
   from **01 Loader**, or the preop MRI). Auto-populates on Refresh
   when a workflow base is set.
2. **FreeSurfer MRI** — the moving image. Empty until you load one in
   the next step.
3. **Load FS MRI file** — path to `T1.nii.gz` (or whichever FS
   "input" MRI you want to register). Click **Load MRI Into Scene** to
   pull it in. The volume gets the `AdditionalMRIVolumes` workflow role
   automatically; the FS MRI selector updates to point at the new node.
4. **Init mode** — `useGeometryAlign` (default) initializes BRAINSFit
   from the image headers. Switch to `useMomentsAlign` if the geometry
   init produces a visibly off-axis pre-alignment (rare; usually only
   needed for very different FOVs).
5. **Output transform** — name for the resulting `LinearTransform`
   node. Default `FS_to_ROSA`; you usually don't need to change it.
6. Click **Register FS MRI → ROSA**. Runs rigid Versor3D + Mattes MI
   (the same algorithm as **01 Loader**'s registration). Takes
   30 s-2 min. On success the transform is published with role
   `FSToBaseTransform` and the status log prints the node name.

#### B. Load FreeSurfer parcellations

1. **FreeSurfer subject** — path to the subject's root directory
   (the one that contains `mri/`, `surf/`, `label/`, …). Setting this
   triggers a scan; the parcellation dropdown fills with what was
   found.
2. **Parcellation volume** — `aparc+aseg`, `aparc.a2009s+aseg`,
   `wmparc`, `aseg`, etc. Pick one, or pick **all available** to load
   every parcellation FS produced. **Refresh** rescans if you change
   the subject path.
3. **Apply FS→ROSA transform to parcellations** *(default ON)* —
   leave on unless you have a reason to keep volumes in FS native
   space (debugging, comparing alignments).
4. **Harden parcellation transforms** *(default OFF)* — bakes the
   transform into the volume's IJK→RAS so the volume no longer needs
   the transform node parented above it. Turning this on is one-way;
   don't enable unless you're ready to commit.
5. **Apply LUT to parcellation volumes** *(default ON)* + **Annotation
   LUT** path — point at `$FREESURFER_HOME/FreeSurferColorLUT.txt`.
   Without the LUT the parcellation displays as a grayscale labelmap
   and individual region names won't show on hover.
6. **Create 3D geometry from parcellations** *(default OFF)* — runs
   marching cubes on the labelmap to build 3D segmentation surfaces.
   Slow on whole-brain parcellations; turn on only when you actually
   need the geometry.
7. Click **Load Parcellation Volumes**. `wmparc` gets the
   `WMParcellationVolumes` role; everything else gets
   `FSParcellationVolumes`. **02 Atlas Labeling** consumes both roles.

#### C. Load cortical surfaces *(optional; only needed for visual overlay or surface-projection workflows)*

1. **Surface mode**:
   - **None** — skip surfaces entirely.
   - **FS pial** — load `lh.pial` / `rh.pial` directly from the
     subject's `surf/` folder. Fastest, anatomically accurate.
   - **Volume-derived** — extract surfaces from one of the
     parcellation volumes via marching cubes. Use this when you don't
     have FS-native surfaces but do have a parcellation.
2. **Surface source volume** — only used in *Volume-derived* mode.
   Auto-populates from the most recently loaded `FSParcellationVolumes`.
3. **Surface annotation** — only used in *FS pial* mode. Name of the
   annotation file in `label/` (e.g. `aparc`, `aparc.a2009s`); paints
   the surface with the parcellation colors.
4. **Apply FS→ROSA transform to surfaces** *(default ON)* and
   **Harden surface transforms** *(default ON)* — same semantics as
   for parcellations.
5. **Surface decimate** *(default 0.60)* — reduces the FS pial mesh by
   60 % to keep the 3D view responsive. Ignored for *Volume-derived*.
6. **Load / Generate Surfaces** publishes nodes to
   `FSCorticalSurfaceModels`. **Clear Surfaces** removes them.

#### D. Publish an existing atlas volume

For volumes you already loaded into Slicer manually (drag-and-drop,
DCM import, another module): pick the volume in **Use existing atlas
volume**, then **Publish Selected Atlas Volume**. It infers the role
from the filename (`wmparc*` → `WMParcellationVolumes`, otherwise
`FSParcellationVolumes`) and optionally applies the FS→ROSA transform.

### Tab 2 — THOMAS

Same shape as the FreeSurfer tab, minus the surfaces sub-flow:

1. **Load THOMAS MRI** (`THOMAS_<subject>/<subject>_MRI.nii.gz` or
   similar) — into the scene + workflow as `AdditionalMRIVolumes`.
2. **Register THOMAS MRI → ROSA** — publishes
   `THOMASToBaseTransform`. Default transform name `THOMAS_to_ROSA`.
3. **THOMAS output dir** — the folder that contains `left/` and
   `right/` subdirectories with one NIfTI per thalamic nucleus
   (CM, VA, VL, …). The loader skips the THOMAS-internal `crop*`,
   `resampled*`, `regn_*` variants — only the canonical structure
   masks are picked up.
4. **Apply THOMAS→ROSA transform** *(default ON)* + **Harden loaded
   thalamus transforms** *(default ON)*.
5. **Load THOMAS Thalamus Masks** creates two `Segmentation` nodes
   (`THOMAS_Left_Structures`, `THOMAS_Right_Structures`) — one per
   side, each with one segment per nucleus, auto-colored by nucleus
   type with left/right tinting. Published with role
   `THOMASSegmentations`. **02 Atlas Labeling** and **03 Navigation
   Burn** both read this role.

### Tab 3 — Registry

Read-only view of what's been published. The image registry lists
every registered volume (name, source type, RAS frame, default-flag);
the transform registry lists every transform with from-space /
to-space. **Refresh Registry View** rescans. Use this to confirm a
load actually published, before moving on to **02 Atlas Labeling**.

### Typical FS + THOMAS flow

Roughly in order:

1. Make sure **01 Loader** has set a `BaseVolume` (or `PostopCT`).
2. **Refresh Workflow Inputs** — base volume populates the selectors.
3. *(FreeSurfer tab)* Load the FS MRI → Register → set FS subject path
   → Load parcellations *(LUT enabled, transform-apply on)*.
4. *(optional)* Load cortical surfaces — usually only for visual review,
   not strictly needed for labeling.
5. *(THOMAS tab)* Load the THOMAS MRI → Register → set THOMAS output
   dir → Load masks.
6. *(Registry tab)* Refresh and verify the expected `FS_to_ROSA`,
   `THOMAS_to_ROSA` transforms + the parcellation / segmentation
   nodes are all present.

If you need only one source (FS or THOMAS, not both), skip the other
tab entirely — they don't depend on each other.

### Workflow roles published (for downstream modules)

- `AdditionalMRIVolumes` — FS / THOMAS reference MRI nodes (both tabs publish here)
- `FSToBaseTransform` — `LinearTransform` mapping FS → base RAS (FS tab register)
- `THOMASToBaseTransform` — `LinearTransform` mapping THOMAS → base RAS (THOMAS tab register)
- `FSParcellationVolumes` — parcellation labelmap volumes (FS tab parc load)
- `WMParcellationVolumes` — `wmparc*` labelmaps split out as their own role (FS tab parc load)
- `FSCorticalSurfaceModels` — pial / volume-derived surface models (FS tab surf load)
- `THOMASSegmentations` — left + right structure segmentations (THOMAS tab load)

### Common error states

- **"Select both ROSA base and FS MRI"** — one of the two selectors is
  empty. If the base looks set in another module but is blank here,
  click *Refresh Workflow Inputs* first.
- **"FS→ROSA transform not available"** when loading parcellations or
  surfaces — registration hasn't run yet *and* "Apply FS→ROSA
  transform" is on. Either run the registration step, or uncheck the
  apply box.
- **"No surf/ directory found"** — the subject path you set isn't a
  FreeSurfer subject root. Point at the directory that contains
  `mri/`, `surf/`, `label/`, not at a child of it.
- **"No THOMAS structures found"** — the THOMAS output dir is missing
  `left/` and `right/` subfolders, or those folders contain only the
  intermediate `crop*` / `resampled*` files. Re-run THOMAS or point at
  the right folder.
- **Parcellation loads as grayscale, no region names on hover** —
  *Apply LUT* is on but *Annotation LUT* is empty. Point it at
  `FreeSurferColorLUT.txt`.
- **"Failed to load MRI volume"** — unsupported format. Convert to
  `.nii.gz` and retry.

## `02 Atlas Labeling`

Assigns each contact to selected atlas sources:
- FreeSurfer parcellation
- THOMAS
- White-matter parcellation

Output table includes per-source labels/distances plus unified `closest_*` and `primary_*` fields.

## `03 Navigation Burn`

Burns selected THOMAS nucleus labels into an MRI and optionally exports DICOM.

Typical sequence:
1. Ensure THOMAS sources are loaded/aligned.
2. Choose burn input MRI, side, nucleus, fill value.
3. Run burn.
4. Optional one-step DICOM export.

## 5) Export

## `01 Export Center`

Exports from workflow scene state (not from module-local temporary state).

Select:
- output directory
- filename prefix
- export profile
- optional output frame volume

Profiles:
- `contacts_only`
- `trajectories_only`
- `registered_volumes_only`
- `atlas_only`
- `qc_only`
- `full_bundle`

Common outputs:
- contacts coordinates
- planned/final trajectory CSV
- QC CSV
- atlas assignment CSV (if available)
- aligned NIfTI volumes (profile-dependent)
- manifest JSON

## 6) Coordinate Frames and Interop

- Primary exported XYZ is in the selected export frame volume.
- Atlas labeling semantics come from atlas-native sampling.
- For external tools, prefer exported aligned NIfTI + exported coordinates instead of raw Analyze `.img/.hdr`.

## 7) Example User Paths

- ROSA full case: use `Loader` -> localization -> atlas -> export.
- MRI+CT only (no `.ros`): use `Loader` custom import + `Postop CT Localization` de novo detect.
- External localization file: use `Contact Import` then `Atlas Labeling` and `Export Center`.
- Navigation-only burn: use `Atlas Sources` + `Navigation Burn`.

## 8) Troubleshooting

- Module not visible:
  - verify repo root is in Slicer Additional module paths
  - restart Slicer
- Empty exports:
  - verify required workflow roles exist (contacts/trajectories/volumes for chosen profile)
- Misaligned overlays:
  - verify base volume and transforms in `Atlas Sources` or loader registration steps
- Analyze warnings:
  - ITK Analyze deprecation warnings are expected and non-fatal
