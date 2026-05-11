# SEEG / ROSA Toolkit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19994662.svg)](https://doi.org/10.5281/zenodo.19994662)

Last updated: 2026-05-11

A modular toolkit for SEEG planning, localization, atlas labeling, and
export workflows. Two surfaces share one algorithm core:

- **3D Slicer extension** — clinical / research workflow with full UI
  (load ROSA case, fit trajectories on postop CT, place contacts,
  label against atlases, export). See [`docs/SLICER_GUIDE.md`](docs/SLICER_GUIDE.md).
- **`rosa-agent` CLI** — headless `pip install`-able command-line agent
  that runs the same pipeline outside Slicer for batch processing,
  regression testing, and reproducible scripting. Eight subcommands:
  `load`, `detect`, `contacts`, `label`, `pipeline`, `place`,
  `rosa-to-nifti`, `match-ros`. See [`cli/README.md`](cli/README.md).

## Capabilities

- ROSA case loading from `.ros` + Analyze image pairs (`.img/.hdr`),
  with a `rosa-to-nifti` CLI command to bake displays + plan into
  ready-to-use NIfTI inputs
- custom MRI/CT import and base-space registration (rigid Versor3D +
  Mattes mutual information; same algorithm in both surfaces)
- guided and de novo trajectory localization on postop CT
- contact generation with electrode model assignment and QC metrics,
  either at the model's nominal pitch ("model-driven") or at CT-image
  peaks along the trajectory ("peak-driven")
- 5-mode staged contact placer (auto / count / named / seeded /
  seeded+model) exposed as `rosa-agent place` and as the
  `rosa_core.placement_modes.place_seeg` library API
- cross-volume trajectory matching: pair a `.ros` plan with a CT in any
  RAS frame (no reference volume / image registration needed) via
  `rosa-agent match-ros` — useful when the post-op CT was registered to
  a different MRI atlas than the one the surgeon planned on
- atlas source loading (FreeSurfer, THOMAS, WM) and contact labeling,
  with optional inline registration of an atlas T1 to the contact
  volume
- THOMAS nucleus burn into MRI with optional DICOM export *(Slicer)*
- profile-based data export for downstream analysis/reporting *(Slicer)*

## Slicer modules

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

## CLI quickstart

```bash
pip install .            # release install (or `pip install -e .` for dev)
rosa-agent --help
```

End-to-end on a ROSA case folder (uses one of the embedded volumes as
the working CT, .ros-planned trajectories as guided-fit seeds):

```bash
rosa-agent pipeline /path/to/ROSA_CASE --ref-volume post --out-dir /tmp/out
```

External CT registered to the ROSA reference frame:

```bash
rosa-agent pipeline /path/to/ROSA_CASE --ct external_ct.nii.gz --out-dir /tmp/out
```

CT-only auto-detection (no ROSA folder, no seeds):

```bash
rosa-agent place --ct postop_ct.nii.gz --output /tmp/out --library dixi
```

Cross-volume naming (same patient, two RAS frames — no registration
needed):

```bash
rosa-agent match-ros --rosa-folder PLAN/ --ct any_frame_ct.nii.gz --output /tmp/out
```

See [`cli/README.md`](cli/README.md) for the full subcommand reference,
TSV column contract, and the Library API (calling `rosa_core` /
`rosa_detect` from Python).

## Architecture (high level)

`CommonLib/` packages, layered:

- **`rosa_core`** — pure-Python domain logic (parser, transforms,
  contact placement, atlas-assignment policy + index, electrode
  classifier, registration helper). Lazy `__init__.py` so headless
  callers don't pull NumPy as a side effect.
- **`rosa_detect`** — pure-Python detection algorithm with a sealed
  public seam (`run_contact_pitch_v1`). No Slicer / VTK / Qt deps —
  pinned by boundary tests.
- **`shank_core`** — CT IO, masking, blob-candidate helpers
  (pure-Python).
- **`rosa_scene`** — Slicer-only scene services (trajectory /
  electrode / atlas providers, registration, layout). Includes
  `sitk_volume_adapter` — the single bridge from a
  `vtkMRMLScalarVolumeNode` to the SITK + 4×4 inputs `rosa_detect`
  consumes.
- **`rosa_workflow`** — Slicer MRML workflow state + publishing.

Slicer modules import only from `CommonLib/`. The CLI imports only
the headless packages (`rosa_core`, `rosa_detect`, `shank_core`); the
Slicer extension and the CLI share parity-critical math via single
sources of truth (volume centering, LPS-flip stamping, atlas index,
LoG kernel) so changes can't drift between the two surfaces.

## Documentation

- Overview + capabilities: this file
- Install: [`INSTALL.md`](INSTALL.md)
- User guide (cross-surface, conceptual workflow): [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md)
- Slicer guide (per-module reference): [`docs/SLICER_GUIDE.md`](docs/SLICER_GUIDE.md)
- CLI guide (every subcommand + Library API): [`cli/README.md`](cli/README.md)
- Pipeline constants reference (all tunable knobs + rationale): [`docs/PIPELINE_CONSTANTS.md`](docs/PIPELINE_CONSTANTS.md)
- Developer guide (architecture, parity invariants, extension points): [`docs/DEVELOPER_GUIDE.md`](docs/DEVELOPER_GUIDE.md)

## License

[`LICENSE`](LICENSE)
