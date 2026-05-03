# Installation

Last updated: 2026-05-02

The repo ships two surfaces. Install whichever you need (or both —
they share `CommonLib/`):

- **Slicer extension** — for the full clinical / research UI.
  Installed by adding the repo to Slicer's module path (no pip
  install needed).
- **`rosa-agent` CLI** — pure-Python `pip install`-able package for
  headless / batch use. Standard Python install.

## Requirements

- For the Slicer extension: 3D Slicer 5.x.
- For the CLI: Python ≥ 3.10. Hard deps install automatically via
  `pip` (`numpy`, `SimpleITK`, `nibabel`, `scipy`).

## Install the Slicer extension (manual module path)

1. Open Slicer.
2. Go to `Settings -> Modules`.
3. Add repository root to `Additional module paths`:
   - `<repo>`
4. Restart Slicer.
5. Verify categories are visible:
   - `ROSA.01 Setup`
   - `ROSA.02 Localization`
   - `ROSA.03 Atlas`
   - `ROSA.04 Export`

## Install the `rosa-agent` CLI

```bash
cd <repo>
pip install .            # release install
pip install -e .         # editable / dev install
```

Creates a `rosa-agent` console script and registers the headless
packages (`rosa_agent`, `rosa_core`, `rosa_detect`, `shank_core`).
Verify:

```bash
rosa-agent --help
```

The Slicer-coupled packages (`rosa_scene`, `rosa_workflow`) are
deliberately NOT installed — they require Slicer's `__main__`
namespace and have no value in a headless install.

See [`cli/README.md`](cli/README.md) for usage.

## Optional dev environment

For running the regression tests + dataset-gated tests:

```bash
cd <repo>
conda env update -f environment.yml
```

## First Validation

1. Open `ROSA.01 Setup -> 01 Loader`.
2. Load a ROSA case folder with `.ros` and `DICOM/*/*.img/.hdr`.
3. Open `ROSA.02 Localization -> 02 Contacts & Trajectory View`.
4. Generate contacts for one or more trajectories.
5. Open `ROSA.04 Export -> 01 Export Center` and run `contacts_only` export.

## Uninstall

1. Remove repository path from `Settings -> Modules -> Additional module paths`.
2. Restart Slicer.
