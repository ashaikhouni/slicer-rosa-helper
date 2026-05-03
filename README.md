# SEEG / ROSA Toolkit

Last updated: 2026-05-02

A modular toolkit for SEEG planning, localization, atlas labeling, and
export workflows. Two surfaces share one algorithm core:

- **3D Slicer extension** — clinical / research workflow with full UI
  (load ROSA case, fit trajectories on postop CT, place contacts,
  label against atlases, export).
- **`rosa-agent` CLI** — headless `pip install`-able command-line agent
  that runs the same pipeline (`load` / `detect` / `contacts` / `label`
  / `pipeline`) outside Slicer for batch processing, regression
  testing, and reproducible scripting.

## Capabilities

- ROSA case loading from `.ros` + Analyze image pairs (`.img/.hdr`)
- custom MRI/CT import and base-space registration (rigid Versor3D +
  Mattes mutual information; same algorithm in both surfaces)
- guided and de novo trajectory localization on postop CT
- contact generation with electrode model assignment and QC metrics,
  either at the model's nominal pitch ("model-driven") or at CT-image
  peaks along the trajectory ("peak-driven")
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
rosa-agent detect postop_ct.nii.gz --out trajectories.tsv
rosa-agent contacts trajectories.tsv postop_ct.nii.gz --out contacts.tsv
```

See [`cli/README.md`](cli/README.md) for the full subcommand and TSV
column reference.

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

- Install: [`INSTALL.md`](INSTALL.md)
- User guide: [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md)
- Developer guide: [`docs/DEVELOPER_GUIDE.md`](docs/DEVELOPER_GUIDE.md)
- CLI reference: [`cli/README.md`](cli/README.md)
- Pipeline state + score-band policy: [`docs/HANDOFF.md`](docs/HANDOFF.md)

## License

[`LICENSE`](LICENSE)
