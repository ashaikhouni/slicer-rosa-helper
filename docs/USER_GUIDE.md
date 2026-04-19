# User Guide

Last updated: 2026-04-19

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

Loads and registers atlas sources:
- FreeSurfer (surfaces and parcellation volumes)
- THOMAS masks

Publishes image and transform references into workflow roles.

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
