# SynthStrip — third-party notice

The vendored skull-stripping path (`synthstrip_bundled.py`) runs **SynthStrip**,
FreeSurfer's deep-learning brain extraction tool. We do **not** bundle the
FreeSurfer suite and we do **not** require a FreeSurfer license key.

## What is used, and how

- **Inference script** — the *standalone* `mri_synthstrip` Python script, which
  defines its U-Net inline and depends only on `torch` / `numpy` / `surfa`. It is
  **fetched at runtime** (not committed to this repo) from a pinned FreeSurfer
  commit and verified by SHA-256:
  - commit `0ac5dcccb8b6b875312bcd042258d4590ba39814`
  - sha256 `bbc2ff8f8779862039401b05d5cd6039fb4f3583e0032a793ac9adb3f4521590`
- **Model weights** — openly licensed under **MIT / CC BY-4.0**, fetched and
  checksum-pinned:
  - `synthstrip.1.pt` — sha256 `37417f80…c653e33`, 30,851,709 bytes
  - `synthstrip.nocsf.1.pt` — sha256 `62bf0113…696ecb28`, 30,851,709 bytes
  - source host: `https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/synthstrip/models/`

Assets cache under `~/.cache/rosa-agent/synthstrip/` (override with
`ROSA_SYNTHSTRIP_CACHE`). A desktop/offline build pre-populates that cache so no
network is needed at run time; set `ROSA_SYNTHSTRIP_NO_DOWNLOAD=1` to require it.

## Why fetch instead of vendor in-tree

The script is FreeSurfer-origin code and the weights are ~30 MB. Rather than
commit FreeSurfer-origin code into this MIT/Zenodo-published repository, we
download the pinned, checksum-verified assets at run time (the same pattern
nipreps/fMRIPrep use for FreeSurfer). The **full FreeSurfer suite is never
bundled or redistributed** — its registration EULA forbids third-party transfer.

## Required citation

> A. Hoopes, J. S. Mora, A. V. Dalca, B. Fischl, M. Hoffmann.
> *SynthStrip: Skull-Stripping for Any Brain Image.*
> NeuroImage 206 (2022) 119474. https://doi.org/10.1016/j.neuroimage.2022.119474

## Runtime dependencies and their licenses

- `torch` (PyTorch) — BSD-3-Clause
- `surfa` — MIT
- `numpy` — BSD-3-Clause

Install via the optional extra: `pip install 'rosa-agent[synthstrip]'`.

## Regulatory note

SynthStrip, like all FreeSurfer-origin tools, is for research use and is **not
FDA-cleared**. Any clinical use is at the user's own regulatory risk.
