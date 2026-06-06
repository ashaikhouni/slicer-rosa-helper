# CLI Guide — `rosa-agent`

Last updated: 2026-05-11

A pure-Python command-line agent that runs the SEEG localization
pipeline end-to-end without Slicer. Same algorithm as the Slicer
extension — see [`docs/USER_GUIDE.md`](../docs/USER_GUIDE.md) for the
cross-surface mental model and [`docs/PIPELINE_CONSTANTS.md`](../docs/PIPELINE_CONSTANTS.md)
for tunable knobs.

## Install

```bash
pip install .            # release install
pip install -e .         # editable / dev install
```

The install creates a `rosa-agent` console script and registers the
headless packages (`rosa_agent`, `rosa_core`, `rosa_detect`,
`shank_core`) so they're importable from anywhere — no PYTHONPATH
needed. Run it from any cwd:

```bash
rosa-agent <subcommand> ...
# or equivalently
python -m rosa_agent <subcommand> ...
```

The `pyproject.toml` lives at the repo root (`slicer-rosa-helper/`)
and explicitly excludes the Slicer-coupled packages
(`rosa_scene`, `rosa_workflow`) so the install stays headless.

## Repo-mode (no install)

`python -m rosa_agent ...` from the repo root still works without
`pip install` — the boot path detects an un-installed checkout and
adds `CommonLib/` to `sys.path` as a fallback. This is for dev
iteration only; production use should `pip install`.

## Subcommands at a glance

| Command              | Purpose                                                                                 |
|----------------------|-----------------------------------------------------------------------------------------|
| `load`               | Parse a ROSA case folder into a JSON manifest                                           |
| `detect`             | Run shank detection on a postop CT volume (auto or guided)                              |
| `contacts`           | Place contacts along trajectories using LoG-driven peaks                                |
| `label`              | Assign atlas labels to a contacts TSV                                                   |
| `pipeline`           | Run all four stages end-to-end (load → detect → contacts → label)                       |
| `place`              | 5-mode staged placer (auto / count / named / seeded / seeded+model) on any CT           |
| `fit-rosa`           | Fit electrode models per planned `.ros` trajectory → contacts + plan-vs-fit `qc.tsv`    |
| `rosa-to-nifti`      | Bake a ROSA case folder (.ros + Analyze) into NIfTI volumes + a `seeds.tsv`             |
| `match-trajectories` | Name detector emissions on any-frame CT from a named-trajectory file (line-geom RANSAC) |
| `match-ros`          | `match-trajectories` with the plan parsed from a `.ros` (no images needed)              |
| `export-view`        | Pipeline → browser GLB; FreeSurfer brain + labels optional (CT-only works too)          |
| `view`               | Serve an `export-view` output dir over HTTP and open it in your browser                 |
| `brain-extract`      | Produce an intracranial brain mask via SynthStrip (CT/MRI) or LoG-watershed (CT only)   |
| `deidentify-ros`     | Strip PHI from a `.ros` / case folder (pseudonymise DICOM UIDs, drop raw DICOM/zips)    |

`rosa-agent <subcommand> --help` prints flags for any individual
subcommand.

## Inputs

Coordinates are **RAS millimeters** except inside the JSON manifest's
`planned_trajectories` which exposes both `*_lps` (the raw ROSA frame)
and `*_ras` keys.

## Output TSV columns (the public contract)

These columns are **stable** across the CLI and Slicer surfaces. Add new
columns at the end if you need to extend.

### `trajectories.tsv`

```text
name              str    trajectory label
start_x/y/z       float  RAS mm — bolt-side / outer endpoint
end_x/y/z         float  RAS mm — deep tip
confidence        float  0..1
confidence_label  str    high | medium | low
electrode_model   str    library model id
bolt_source       str    metal | synthesized | wire | none
length_mm         float  end - start
```

### `contacts.tsv`

```text
trajectory       str
label            str    "<trajectory><index>" (e.g. L_AC1)
contact_index    int    1-based
x/y/z            float  RAS mm
peak_detected    int    1 = anchored on detected peak, 0 = model-nominal
electrode_model  str
```

### `labels.tsv`

```text
trajectory                       str
contact_label                    str
contact_index                    int
contact_x/y/z                    float  RAS mm
closest_source                   str    thomas | freesurfer | wm
closest_label                    str
closest_label_value              int
closest_distance_to_voxel_mm     float
thomas_label / *_distance_*      per-source samples
freesurfer_label / *_distance_*
wm_label / *_distance_*
```

---

## End-to-end on the SEEG dataset

```bash
ROSA_SEEG_DATASET=/path/to/seeg_dataset \
    python -m rosa_agent pipeline T22 --out-dir /tmp/T22_cli
```

Outputs:

```text
/tmp/T22_cli/
    trajectories.tsv      ~9 entries
    contacts.tsv          ~117 contacts
    labels.tsv            (when --thomas/--freesurfer is passed)
```

---

## `pipeline` — full end-to-end on a ROSA case folder

Three flavors depending on whether you want detection on a ROSA-embedded
volume, an external volume already aligned to the ROSA frame, or an
external volume that needs registration.

### A. Use a volume from inside the ROSA folder

```bash
python -m rosa_agent pipeline /data/cases/RYAN_ANON \
    --ref-volume postopCT \
    --out-dir /tmp/ryan_cli
```

The named display is loaded from the ROSA folder (Analyze .img/.hdr →
in-memory NIfTI), its `TRdicomRdisplay` matrix is baked into the
SITK image's geometry, and detection runs in the ROSA reference frame.
A NIfTI copy of the working CT is written to `out_dir/ct.nii.gz`
(useful because Analyze isn't a great archival format).

Defaults: `--ref-volume` defaults to the first display in the .ros file.

### B. External CT, already aligned to the ROSA frame

```bash
python -m rosa_agent pipeline /data/cases/RYAN_ANON \
    --ct /data/cases/RYAN_ANON/postop_ct.nii.gz \
    --skip-registration \
    --out-dir /tmp/ryan_cli
```

ROSA-derived seeds are used as guided-fit seeds in the CT frame
without a registration pass. The user's CT is not copied or
transformed — outputs land in the CT frame.

### C. External CT, needs registration to ROSA frame

```bash
python -m rosa_agent pipeline /data/cases/RYAN_ANON \
    --ct /some/external_ct.nii.gz \
    --ref-volume preopMRI \
    --output-frame ct \
    --out-dir /tmp/ryan_cli
```

Rigid Versor3D + Mattes mutual information registration aligns the
external CT to the named ROSA reference (mirrors the BRAINSFit
parameter set the Slicer-side `RegistrationService` uses, so Slicer
and CLI runs on the same pair land in the same place). ROSA-derived
seeds are inverse-transformed into the external CT frame before
detection runs natively in CT frame.

`--output-frame ct` (default): outputs in the external CT frame.
`--output-frame rosa`: outputs are pushed back to the ROSA reference
frame after detection, so they line up with the ROSA-frame planning
geometry.

### Atlas labeling

Two flavors depending on whether the atlas already shares a frame with
your contacts.

**(a) Atlas already in contact-frame RAS** (e.g. parcellation produced
by registering recon-all output back to the postop CT):

```bash
python -m rosa_agent pipeline ... \
    --freesurfer /path/to/aparc+aseg.nii.gz \
    --freesurfer-lut $FREESURFER_HOME/FreeSurferColorLUT.txt \
    --thomas /path/to/thomas_segmentations/
```

**(b) Atlas in T1 RAS — register inline**:

```bash
python -m rosa_agent pipeline ... \
    --freesurfer /path/to/aparc+aseg.nii.gz \
    --freesurfer-lut $FREESURFER_HOME/FreeSurferColorLUT.txt \
    --atlas-base /path/to/T1_recon_input.nii.gz
```

When `--atlas-base` is set, the FS / WM labelmaps are rigidly
registered (Versor3D + Mattes MI, same algorithm as BRAINSFit) and
resampled (nearest-neighbor — labels stay valid integers) onto the
working CT's grid before sampling. THOMAS skips this step (it's
typically already in the same frame as the labelmap it's paired with).

The standalone `label` subcommand takes the same flags plus a required
`--target-volume`:

```bash
python -m rosa_agent label contacts.tsv \
    --freesurfer aparc+aseg.nii.gz \
    --atlas-base T1.nii.gz \
    --target-volume postop_ct.nii.gz \
    --out labels.tsv
```

---

## `place` — 5-mode staged contact placer

`rosa-agent place` exposes the staged placer's five input modes through
one CLI. The mode is implied by which optional flag(s) you pass — the
table below mirrors `rosa_core.placement_modes.place_seeg`'s contract:

| Flag(s) used                                 | Mode | What you fix                  | Use case                                                |
|----------------------------------------------|------|-------------------------------|---------------------------------------------------------|
| (none)                                       | 1    | nothing                       | full auto: detect + place from a bare CT                |
| `--n-expected N`                             | 2    | expected shank count          | "I expect N shanks; pick top N by score"                |
| `--expected E.tsv`                           | 3    | named expected (name + model) | "These named shanks should be present"                  |
| `--seeds S.tsv` (with `electrode_model` col) | 4    | seeds + models                | external/manual seeds with vouched electrode models     |
| `--seeds S.tsv` (no model col)               | 5    | seeds only                    | external/manual seeds; library matcher picks the model  |

Examples:

```bash
# Mode 1 — full auto (no priors)
rosa-agent place --ct ct.nii.gz --output qc/ --library dixi

# Mode 2 — fix the count
rosa-agent place --ct ct.nii.gz --output qc/ --n-expected 8

# Mode 3 — fix the named set
rosa-agent place --ct ct.nii.gz --output qc/ --expected expected.tsv

# Mode 4 — vouched seeds + models
rosa-agent place --ct ct.nii.gz --output qc/ --seeds seeds.tsv  # seeds.tsv has electrode_model column

# Mode 5 — seeds only, library picks the model
rosa-agent place --ct ct.nii.gz --output qc/ --seeds seeds.tsv  # seeds.tsv lacks electrode_model column
```

Seed TSV format: `name`, `start_x/y/z`, `end_x/y/z`, optional
`electrode_model`. The same parser ingests trajectory-row, ex/tx, and
label+xyz pair flavors — see `cli/rosa_agent/io/trajectory_io.py`.

Expected TSV format: simple two-column `name<TAB>model_id`.

Other useful flags:

- `--library KEY` *(default: full library)* — restrict to a vendor strategy: `pmt_35`, `dixi`, `adtech`, ...
- `--sampler {log,hu}` *(default: `log`)* — walker signal source. LoG dominates per the 2026-05-09 sweep.
- `--band-floor {high,medium,low}` *(default: `medium`)* — drop emissions below this band.
- `--snap-angle-deg N` *(default: 12)* — mode 4/5 snap-to-v1 angle tolerance.
- `--snap-perp-mm N` *(default: 8)* — mode 4/5 snap-to-v1 perp tolerance.
- `--no-figures` — skip matplotlib PNG render (TSVs always written).
- `--subject-id ID` — stamped into `manifest.json`.

Output directory layout:

```text
output/
  manifest.json
  trajectories.tsv
  contacts.tsv
  figures/                   # per-trajectory PNGs (skipped if --no-figures)
  diagnostics/cmp.tsv
```

For the constant knobs the placer uses internally (matched-filter
thresholds, score weights, validators), see
[`docs/PIPELINE_CONSTANTS.md § B`](../docs/PIPELINE_CONSTANTS.md#b-placer--rosa_corecontact_placementconstants).

---

## `fit-rosa` — fit electrode models per planned ROSA trajectory

```bash
rosa-agent fit-rosa CASE_DIR/ --postop-ct oarmpost -o out/             # ROS folder
rosa-agent fit-rosa --seeds plan.tsv --postop-ct-path ct.nii.gz -o out/  # seeds + CT
rosa-agent fit-rosa --auto --postop-ct-path ct.nii.gz -o out/          # CT-only, no plan
```

Takes planned trajectories (from a `.ros`, a seed TSV, or inline detection),
snaps each to the imaged shank in the postop CT, picks the electrode model
(matched-filter library picker), and places contacts from the fitted tip.
Exactly one input mode:

- **ROS** — plan = the `.ros` planned trajectories; CT = `--postop-ct <display>` (a display inside the folder).
- **Seeds** — plan = `--seeds plan.tsv`; CT = `--postop-ct-path ct.nii.gz`.
- **Auto** — plan = inline detection (`--auto`); CT = `--postop-ct-path ct.nii.gz`.

Output (`-o DIR`): `manifest.json`, `contacts.tsv` (leading `# reference_frame:`
provenance comment), `trajectories.tsv`, **`qc.tsv`** (plan-vs-fit deviation per
shank — entry/target 3-D error, contact lateral deviation from the planned line,
axis angle; cohort summary in `manifest.qc_summary`), and `figures/` (unless
`--no-figures`).

Useful flags: `--mask-backend {auto,hull,synthstrip,log-watershed}`,
`--library <strategy>`, `--electrodes models.tsv` (force a model per shank),
`--reference-image ref.nii.gz` (emit contacts in another volume's RAS frame).

---

## `rosa-to-nifti` — bake a ROSA folder into NIfTI inputs

Convert a ROSA case folder (`.ros` + `DICOM/<uid>/<name>.img/.hdr`) into
a directory of NIfTI volumes whose IJK→RAS headers match the ROSA
reference frame, plus a `seeds.tsv` of the planned trajectories ready to
feed back through `rosa-agent place --seeds`.

```bash
rosa-agent rosa-to-nifti --rosa-folder s57_rosa/ --output s57_unpacked/

# then, e.g.
rosa-agent place \
    --ct s57_unpacked/post.nii.gz \
    --seeds s57_unpacked/seeds.tsv \
    --output s57_qc/ --library dixi
```

By default every display volume is exported. Pass `--volume NAME` (one
or more times) to export a subset. `--quiet` suppresses per-volume
progress.

Output layout:

```text
output/
  manifest.json              # what was loaded + display-to-reference matrices
  seeds.tsv                  # trajectories in the ROSA reference RAS frame
  <display>.nii.gz           # one per exported display
  ...
```

---

## `match-trajectories` — name detector emissions across coordinate frames

You have a CT and a list of **named** planned trajectories whose coordinates
live in a *different* RAS frame than the CT (a plan exported from another
station, a registration to a different MRI atlas, a routine clinical CT in
scanner coordinates) — or you simply never had a usable `.ros`. This command
runs the detector on the CT, then matches each emission to a planned trajectory
using only line geometry — no image-to-image registration, no reference
volume — and propagates the planned names.

```bash
rosa-agent match-trajectories \
    --plan planned_trajectories.tsv \
    --ct  postop_ct.nii.gz \
    --output match_qc/
```

The plan file is any TSV/CSV the seed reader understands — a header with
`name` + `start_x/y/z` + `end_x/y/z` (or the `label, x, y, z` endpoint-pair
form). Coordinates are RAS, in **any** frame; the matcher recovers the rigid
transform to the CT. At least 3 named trajectories are required (RANSAC).

The `trajectories.tsv` carries the planned electrode names (LCMN, RAMF, ROPE, …)
instead of the detector's CAND-NNN labels. Unmatched plans appear in `match.tsv`
with empty det_name; unmatched detector emissions keep their CAND-NNN names.

How the matcher works (full prose lives in
[`CommonLib/rosa_core/cross_volume_match.py`](../CommonLib/rosa_core/cross_volume_match.py)):

1. Trajectories are represented as **infinite lines** (midpoint + unit
   axis). The plan start/end and the detector start/end can drift by
   several mm at each end, but the underlying line is stable.
2. RANSAC picks 3 random plan lines + 3 random det lines. For each of
   3! orderings × 2³ axis-sign combinations, orthogonal Procrustes on
   the direction triplets gives a rotation `R`; centroid alignment
   gives a translation `t`.
3. Inliers are scored by axis angle + perpendicular line-to-line
   distance — both infinite-line properties, so endpoint drift doesn't
   enter.
4. The best-inlier transform is refined on all inlier pairs and used to
   greedy-match each plan to its closest detector emission.

Validated on s57.ros + T24 CT (same patient, two registrations): 16/17
named, all matched pairs ≤ 11° axis-angle and ≤ 6 mm perpendicular
distance.

Tunable flags (defaults validated as above):

- `--angle-tol-deg N` *(default: 15)* — axis-angle tolerance for RANSAC + greedy match.
- `--ransac-perp-mm N` *(default: 8)* — perp tolerance for RANSAC inliers.
- `--match-perp-mm N` *(default: 12)* — perp tolerance for the greedy plan↔det assignment (slightly looser; refined transform is generally good).
- `--ransac-iter N` *(default: 2000)* — RANSAC iteration budget.
- `--seed N` *(default: 42)* — RANSAC RNG seed.

Output layout:

```text
output/
  manifest.json
  trajectories.tsv             # detector emissions, RENAMED to plan names
  contacts.tsv
  figures/                     # filenames use the plan names too
  diagnostics/cmp.tsv
  match.tsv                    # per-plan match (plan_name, det_name, angle°, perp_mm)
  cross_volume_match.json      # recovered transform + RANSAC diagnostics
```

---

## `match-ros` — `match-trajectories` with a `.ros` plan

A thin wrapper over `match-trajectories` for when the plan is a `.ros`. It
parses the planned trajectories out of the `.ros` (reading only the planned
segments + an LPS→RAS flip — **no image volumes needed**, so it works even when
the ROSA folder is missing its imagery), then runs the identical detect +
line-match operation.

```bash
rosa-agent match-ros \
    --rosa-folder s57_rosa/ \
    --ct path/to/post_registered_t24.nii.gz \
    --output match_qc/

# or pass a bare .ros file directly (no folder needed)
rosa-agent match-ros --ros-file plan.ros --ct ct.nii.gz --output qc/
```

Everything else — matcher internals, tunable flags, output layout — is identical
to `match-trajectories` above.

---

## `export-view` — pipeline + FreeSurfer brain into a browser GLB

`export-view` runs the full pipeline on a ROSA case and packs the
result into a `scene.glb` that you can open in any modern browser
(via the auto-emitted `index.html`). The FreeSurfer recon-all directory
serves two roles in one pass: its `aparc+aseg.mgz` drives the
per-contact anatomical labeling (same `labels.tsv` the `pipeline` /
`label` subcommands emit), and its `surf/?h.pial` surfaces become the
3D brain mesh in the GLB.

```bash
# With a recon: brain mesh + atlas labels + a CT/T1 slice toggle
rosa-agent export-view /path/to/CASE \
    --freesurfer-dir /path/to/Recon \
    --out-dir /tmp/case_view

# CT-only: no recon needed — windowed CT slices + the 3D electrode model
rosa-agent export-view /path/to/CASE --out-dir /tmp/case_view
```

`--freesurfer-dir` is **optional**: omit it for a CT-only view (windowed CT
slices + the 3D electrode model, but no brain mesh and no atlas labels — the
point being you no longer need Slicer just to eyeball a result). When a recon
IS supplied, a **Volume** dropdown in the viewer switches the slice background
between the FreeSurfer T1 and the CT (T1 stays the default).

Required:

- positional: ROSA case folder (or dataset subject id — same as `pipeline`)
- `--out-dir`: output directory

Optional FreeSurfer:

- `--freesurfer-dir`: recon-all subject directory (`surf/`, `mri/`, `label/`) —
  adds the brain mesh + per-contact `labels.tsv`. Omit for CT-only.

The command auto-discovers:

- the parcellation labelmap (`mri/aparc+aseg.mgz`, falling back to
  `aparc.DKTatlas+aseg.mgz` / `aparc.a2009s+aseg.mgz` in that order;
  override with `--parcellation`)
- the FreeSurfer base T1 (`mri/T1.mgz`, falling back to `orig.mgz` /
  `brain.mgz` / `rawavg.mgz`)
- the LUT (explicit `--lut` → `$FREESURFER_HOME/FreeSurferColorLUT.txt`
  → bundled copy packaged in `rosa_core/resources/freesurfer/`)
- the surface annotation (`--annotation aparc` by default; pass an empty
  string to disable per-vertex coloring)

Useful flags:

- `--surfaces pial,white` — comma-separated FS surface kinds. Defaults
  to `pial` (one hemisphere = ~150k vertices, so the GLB stays small).
- `--thomas DIR` — adds a THOMAS thalamic provider so the per-contact
  `labels.tsv` also carries thalamic-segment labels.
- `--contact-radius-mm`, `--trajectory-radius-mm` — geometry sizing.
- `--ct-window lo,hi` — HU window for the CT slice volume (default
  `-150,1500`: brain mid-gray, metal contacts saturate bright).
- All `pipeline` frame flags work too (`--ct`, `--ref-volume`,
  `--seeds`, `--skip-registration`, `--output-frame`).

Output layout:

```text
out_dir/
  trajectories.tsv      # pipeline output
  contacts.tsv          # pipeline output
  labels.tsv            # per-contact FS / WM / THOMAS labels (FS mode only)
  ct.nii.gz             # working CT (only when ROSA-folder mode)
  ct_in_view.nii.gz     # windowed uint8 CT for the in-browser slice viewer
  t1_in_ct.nii.gz       # FS T1 resampled onto the CT grid (FS mode only)
  manifest.json         # pipeline manifest
  scene.glb             # the 3D scene (surfaces + trajectories + contacts)
  scene_meta.json       # volumes + contacts/trajectories the viewer consumes
  index.html            # static three.js viewer (volume selector + slice planes)
  view_manifest.json    # what was loaded + counts
```

To view: serve the directory over HTTP and open `index.html` in a
browser (the three.js ES-module imports + the `scene_meta.json` / volume
`fetch`es all need `http://`, not `file://`):

```bash
cd /tmp/case_view && python -m http.server 8000
# then open http://localhost:8000/
```

The sidebar lists every detected contact with its closest FreeSurfer
/ THOMAS / WM label. The GLB also works in any external glTF viewer
(e.g. <https://gltf-viewer.donmccurdy.com/> via drag-and-drop) — you
just lose the sidebar.

The toolbar also has a **CT MIP** toggle — a rotatable maximum-intensity
projection of the loaded volume (a WebGL raymarch over the same 3D texture the
slice planes use). On a windowed CT the metal contacts are the brightest
voxels, so they read as crisp streaks you can spin to verify trajectories at a
glance; the threshold slider drops soft tissue to isolate bone + electrodes.

How frames line up: contacts and trajectories live in the working CT
RAS frame the `pipeline` runs in. FreeSurfer surfaces are originally
in tkrRAS; the loader converts them to scanner RAS using the T1.mgz
matrices, then rigidly registers the T1 to the working CT (rigid +
Mattes MI, same algorithm `--atlas-base` uses inside `pipeline`) and
applies that transform. The same registration is used to resample
`aparc+aseg.mgz` onto the CT grid for labeling, so contacts and
surfaces share a single alignment.

---

## `view` — serve an export-view directory and open it in a browser

The HTML viewer that `export-view` writes needs to be served over
HTTP, not opened from `file://` — `<script type="importmap">` plus
the `fetch()` calls for `scene.glb` / `scene_meta.json` /
`t1_in_ct.nii.gz` are both CORS-blocked under `file://`. This
subcommand bundles "spin up a local server in the right dir + open
the URL" into one step:

```bash
rosa-agent view /tmp/case_view
```

Picks port 8765 by default; if it's busy, it falls back to whatever
port the OS hands out so you can leave it running across sessions
without port-clash babysitting. Holds the server in the foreground
until you Ctrl-C.

Flags:

- `--port N` — preferred port (default `8765`)
- `--no-open` — print the URL and skip auto-launching the browser
  (useful over SSH or in headless CI checks)

No new dependencies — wraps stdlib `http.server` + `webbrowser`.

### Hosted browser viewer (no server, no install)

The same viewer is published as a static GitHub Pages app at
**`ashaikhouni.github.io/slicer-rosa-helper/viewer/`** (sibling to the `.ros`
de-identifier). Instead of `fetch`-ing from a served directory, it shows a
**drag-and-drop zone**: drop the `scene.glb` + `scene_meta.json` + the
`.nii.gz` volume(s) from an `export-view` output and it renders them entirely
in your browser — **nothing is uploaded, no server runs**. Same render engine
as `export-view`/`view` (it's the one template in "picker" mode); regenerate the
page with `python tools/build_web_viewer.py` after changing the template.

Because the data stays on your machine, this is safe for identifiable cases —
but do **not** commit a real case's `scene.glb`/CT into the public `web/` site
(that would publish PHI). Only synthetic/phantom scenes belong there.

---

## `brain-extract` — intracranial brain mask from CT or MRI

```bash
rosa-agent brain-extract postop_ct.nii.gz -o brain_mask.nii.gz
```

Produces a binary brain mask for any CT or MRI volume. The mask inherits
the input volume's full geometry (size + spacing + origin + direction) so
downstream tooling can resample / overlay without an affine fix-up step.

Two backends with different modality coverage — `--backend auto` (default)
picks the right one for what's available:

| Backend         | Modalities    | Deps to ship                                    | Typical runtime |
|-----------------|---------------|-------------------------------------------------|-----------------|
| `synthstrip`    | CT and MRI    | `mri_synthstrip` binary (FreeSurfer 7.3+)       | 30–90 s CPU     |
| `log-watershed` | **CT only**   | none beyond rosa_core (SITK + numpy + scipy)    | ~3 s            |

### Modality cheat-sheet

| Input you have        | Recommended call                                                      |
|-----------------------|-----------------------------------------------------------------------|
| Postop CT, FS on disk | `brain-extract ct.nii.gz -o mask.nii.gz`  *(auto → synthstrip)*       |
| Postop CT, no FS      | `brain-extract ct.nii.gz -o mask.nii.gz`  *(auto → log-watershed)*    |
| T1/T2/FLAIR MRI       | `brain-extract t1.nii.gz -o mask.nii.gz`  *(auto → synthstrip)*       |
| Pinning the backend   | `brain-extract … --backend synthstrip` *or* `--backend log-watershed` |

`auto` will refuse to use `log-watershed` on MRI input (no air-band voxels
< −500 HU) and exit with code 5 — point at SynthStrip in that case.

### Backend internals

**SynthStrip** is FreeSurfer's deep-learning brain extractor — modality-
agnostic, well-validated in the literature, handles wire/bolt/clip
hardware cleanly. The wrapper:

- Auto-discovers `mri_synthstrip` via, in order: `--synthstrip <path>`,
  `$ROSA_SYNTHSTRIP`, `PATH`, `$FREESURFER_HOME/bin/`, and the usual
  install prefixes (`/Applications/freesurfer/*/bin/`, `/usr/local/
  freesurfer*/`, `/opt/freesurfer*/`).
- Injects `FREESURFER_HOME` into the subprocess env when needed (derived
  from the binary path), so callers don't have to source `SetUpFreeSurfer.sh`.
- Post-writes the mask through SimpleITK with `CopyInformation` so the
  output inherits the CT's q-form geometry — works around the
  qform/sform mismatch SynthStrip otherwise propagates on volumes whose
  NIfTI header carries two inconsistent affines.

Useful flags:

- `--stripped <path>` — also write the skull-stripped volume (input × mask).
- `--no-csf` — tighter SynthStrip mask, excludes CSF voxels at the surface.
- `--threads N` — `-t` passed through to SynthStrip.
- `--timeout S` — subprocess kill after `S` seconds.

**LoG watershed** is a pure-Python CT-only fallback in
[`rosa_detect.primitives.intracranial.log_watershed_brain_mask`](../CommonLib/rosa_detect/primitives/intracranial.py).
No deep learning, no external binary, no per-scan tuning. Recipe:

```text
walls       = (|LoG σ=5 mm| ≥ 10)  AND NOT  (HU ≥ 2000  AND  dist_hull ≥ 10mm)
deep_seed   = (dist_hull ≥ 30mm)  AND NOT  walls
markers     = boundary→extracranial(1), deep_seed→intracranial(2)
brain_mask  = MorphologicalWatershedFromMarkers(-dist(walls), markers) == 2
```

- `|LoG σ=5 mm| ≥ 10` — the binary wall mask validated in Variant F of
  [`notebooks/probes/intracranial_mask_lab.ipynb`](../notebooks/probes/intracranial_mask_lab.ipynb).
  Fires on skull, bolts, and frame; contacts wash out at σ = 5 mm.
- **Intracranial-metal carve**: saturated metal (HU ≥ 2000) deeper than
  10 mm from the head hull is *removed* from the wall mask. Without
  this, high-HU contacts (some scanners report HU > 6000) would fire as
  walls and fragment the brain basin around every electrode.
- **Region-based interior marker**: every voxel ≥ 30 mm from the hull is
  a seed (with a `p99(dist_hull)` fallback if the FOV is too tight).
  Avoids the single-voxel-seed trap where a sulcus next to the deepest
  point isolates the marker from the rest of the brain.

Validated on a 10-subject hand-curated cohort
([`notebooks/cross_modality/03_cross_subject_validation.ipynb`](../notebooks/cross_modality/03_cross_subject_validation.ipynb)):

| metric            | LoG watershed | SynthStrip CT | SynthStrip T1 |
|-------------------|---------------|---------------|---------------|
| mean GT recall    | **97.3 %**    | 94.4 %        | 97.0 %        |
| mean leak %       | 0.50 %        | **0.20 %**    | 0.76 %        |
| mean mask vol     | 1942 cm³      | 1501 cm³      | 1513 cm³      |
| recall wins / 10  | **9**         | 1             | n/a           |
| leak wins / 10    | 0             | **10**        | n/a           |

LoG dominates contact recall; SynthStrip dominates boundary tightness.
For SEEG-style contact localization, **recall is the metric that matters**
— a missed contact is unrecoverable while a few mm³ of bolt inside the
mask is harmless downstream.

### Check whether a backend is available

```bash
rosa-agent brain-extract --check                    # prints which backend auto would pick
rosa-agent brain-extract --check --backend synthstrip   # exit 0 if found, 3 otherwise
rosa-agent brain-extract --check --backend log-watershed  # always exit 0
```

### Exit codes

| Code | Meaning                                                               |
|------|-----------------------------------------------------------------------|
| 0    | mask written                                                          |
| 2    | input missing or argument error                                       |
| 3    | chosen backend not available (only fatal with `--backend synthstrip`) |
| 4    | backend ran but failed                                                |
| 5    | modality mismatch (log-watershed asked to run on a non-CT volume)     |

---

## `deidentify-ros` — strip PHI from a ROSA case

```bash
rosa-agent deidentify-ros CASE_DIR/ --out-dir CLEAN/ --subject-id S01   # whole case folder
rosa-agent deidentify-ros case.ros --out clean.ros --subject-id S01     # single .ros file
```

De-identifies a `.ros` (and, in folder mode, the surrounding case) into a
**fresh** directory — the original is never modified.

- Scrubs `PATIENT_NAME` (→ `--subject-id`), `PATIENT_BIRTHDAY`, `SERIE_DATE`.
- **Pseudonymises** DICOM Series UIDs to a consistent `UID-<sha1>` token rather
  than blanking them, so the display↔series linkage in the `.ros` survives.
- Folder mode also copies the image volumes (renaming each `DICOM/<uid>/` dir to
  its UID token so the clean `.ros` still loads), **drops** `DICOMFiles.zip` (raw
  DICOM = full PHI) and screenshots, and writes a PHI-free planned-trajectory CSV.
- Emits a **private** `<id>_deid_keymap.json` (real name/UID → token) *outside*
  the clean dir for re-linking to your own DICOM. That keymap IS PHI — keep it,
  never share it.

A browser-only version (runs entirely client-side, nothing uploaded) is hosted
at `https://ashaikhouni.github.io/slicer-rosa-helper/deidentify-ros/`.

---

## `load` / `detect` / `contacts` / `label` — staged building blocks

The four-stage CLI lets you run one stage at a time when you already
have an upstream artifact. Useful for reruns + integration with non-CLI
tools.

```bash
# Parse a ROSA case folder into a JSON manifest (no detection)
rosa-agent load /path/to/CASE --out manifest.json

# Detect shanks on a postop CT (auto, no seeds)
rosa-agent detect postop_ct.nii.gz --out trajectories.tsv

# Detect with planned seeds (guided fit)
rosa-agent detect postop_ct.nii.gz --seeds plan.tsv --out trajectories.tsv

# Place contacts along an existing trajectories TSV
rosa-agent contacts trajectories.tsv postop_ct.nii.gz --out contacts.tsv

# Label an existing contacts TSV
rosa-agent label contacts.tsv \
    --target-volume postop_ct.nii.gz \
    --freesurfer aparc+aseg.nii.gz \
    --freesurfer-lut $FREESURFER_HOME/FreeSurferColorLUT.txt \
    --out labels.tsv
```

`rosa-agent pipeline` composes those four stages end-to-end. `rosa-agent
place` is a parallel single-stage placer for the 5-mode dispatcher; it
shares the algorithm core but skips `detect` (mode 1 still runs
detection internally).

---

## Library API — calling from Python

The CLI is a thin wrapper around `rosa_core` + `rosa_detect`. You can
call the same functions directly from a notebook / script.

### Detect on a CT

```python
from rosa_detect.service import run_contact_pitch_v1
import SimpleITK as sitk

img = sitk.ReadImage("postop_ct.nii.gz")
result = run_contact_pitch_v1(img)        # auto mode
for traj in result.trajectories:
    print(traj.name, traj.start_ras, traj.end_ras,
          traj.confidence, traj.confidence_label, traj.bolt_source)
```

`result.trajectories` is `list[DetectedTrajectory]` (TypedDict; the
public contract — see `rosa_detect.contracts`).

### Place contacts (5-mode dispatcher)

```python
from rosa_core.placement_modes import place_seeg, Seed
from rosa_core.contact_placement import sample_neg_log_max
from rosa_core.electrode_classifier import filter_models_for_strategy
from rosa_core.electrode_models import load_electrode_library

library = filter_models_for_strategy(load_electrode_library()["models"], "dixi")

# Mode 1 — full auto from a bare CT (string path or SimpleITK.Image)
batch = place_seeg("postop_ct.nii.gz", library=library, sample_fn=sample_neg_log_max)

# Mode 4 — seeds + vouched electrode models
seeds = [
    Seed(name="LAC", start_ras=[-50, 30, 12], end_ras=[-22, 31, 11], model_id="DIXI-MM12CAGB"),
    Seed(name="RAC", start_ras=[ 50, 30, 12], end_ras=[ 22, 31, 11], model_id="DIXI-MM12CAGB"),
]
batch = place_seeg("postop_ct.nii.gz", seeds=seeds, library=library, sample_fn=sample_neg_log_max)

for t in batch.trajectories:
    print(t.name, t.model_id, t.band, t.compound_score, len(t.contacts_ras))
```

The `PlacementBatch` carries `features` and `bolts` (the LoG / hull /
metal-CC inputs the placer used) so downstream callers (figure
renderers, QC writers, post-pass analyses) don't have to reload the CT
just to get the feature volumes back.

### Cross-volume line matching (no image registration)

```python
from rosa_core.cross_volume_match import cross_volume_match

ros_trajs = [
    {"name": "LAC", "start": [-50, 30, 12], "end": [-22, 31, 11]},
    {"name": "RAC", "start": [ 50, 30, 12], "end": [ 22, 31, 11]},
    # ... (≥ 3 lines)
]
det_trajs = [
    {"name": "CAND-001", "start": [...], "end": [...]},
    # ... (detector emissions, in some other RAS frame)
]

result = cross_volume_match(ros_trajs, det_trajs)

print(result.transform_4x4)              # rigid det -> ros
print(result.refined_inliers, "/", len(det_trajs))
for ros_name, det_name, angle_deg, perp_mm in result.pairs:
    print(ros_name, "<-", det_name, angle_deg, perp_mm)
```

`cross_volume_match` accepts dicts (`{name, start, end}`) or
`TrajectoryLine` (`{name, mid, direction}`). For the algorithm details,
see the module docstring; for the matcher's tuning knobs, see
[`docs/PIPELINE_CONSTANTS.md § C`](../docs/PIPELINE_CONSTANTS.md#c-cross-volume-matcher--rosa_corecross_volume_match).

### Parse a ROSA case folder

```python
from rosa_core import find_ros_file, parse_ros_file, load_rosa_volume_as_sitk

# Plan-only (no image read)
ros_path = find_ros_file("/data/cases/RYAN_ANON")
parsed = parse_ros_file(ros_path)
print(parsed["displays"], parsed["trajectories"])

# Volume + plan (loads one Analyze display, bakes the matrix into SITK)
img, meta = load_rosa_volume_as_sitk("/data/cases/RYAN_ANON", volume_name="post")
print(meta["display_index"], meta["is_reference"], meta["display_to_reference_ras"])
```

The `cli/rosa_agent/io/ros_input.py` helper wraps these so a CLI command
can take a `--rosa-folder` or `--ros-file` flag uniformly:

```python
from rosa_agent.io.ros_input import load_ros_planned_trajectories

plan = load_ros_planned_trajectories(folder="s57_rosa/")
print(plan.ros_file, len(plan.trajectories), plan.trajectories[0])
```

### Atlas labeling (Python)

```python
from rosa_core.atlas_index import (
    compute_label_centroids, format_atlas_sample, parse_freesurfer_lut,
)
# (Provider-agnostic helpers; the CLI label command + Slicer's
# atlas_providers both go through this same module.)
```

### Brain mask (LoG watershed)

```python
import SimpleITK as sitk
from rosa_detect.primitives.intracranial import log_watershed_brain_mask

ct = sitk.ReadImage("postop_ct.nii.gz")
mask = log_watershed_brain_mask(ct)        # SITK image, uint8, same geometry as `ct`
if mask is None:
    raise RuntimeError("watershed degenerate (FOV too tight?)")
sitk.WriteImage(mask, "brain_mask.nii.gz")
```

Default parameters (`sigma_mm=5`, `wall_thr=10`, `metal_hu_floor=2000`,
`intracranial_min_mm=10`, `deep_seed_mm=30`) are validated on the 10-subject
cross-cohort scorecard — see
[`notebooks/cross_modality/03_cross_subject_validation.ipynb`](../notebooks/cross_modality/03_cross_subject_validation.ipynb).

### Brain mask (SynthStrip backend, programmatic)

```python
from rosa_agent.services.synthstrip import (
    find_synthstrip, run_synthstrip, SynthStripNotFound,
)

# Quietly detect availability:
binary = find_synthstrip()
if binary is None:
    ...  # fall through to log_watershed_brain_mask above

# Or just run it:
try:
    mask_path = run_synthstrip("postop_ct.nii.gz", "brain_mask.nii.gz")
except SynthStripNotFound:
    ...  # missing dependency; pick another backend
```

`run_synthstrip` accepts `synthstrip_path=` (override the detection chain),
`stripped_path=` (also write the masked volume), `no_csf=True` (tighter
mask), `threads=N`, and `timeout=S`. The wrapper auto-injects
`FREESURFER_HOME` into the subprocess env and post-fixes the output mask's
affine to match the input.

For the full algorithm seam see
[`DEVELOPER_GUIDE.md § Shared Libraries`](../docs/DEVELOPER_GUIDE.md).

---

## Dependencies

- `numpy`, `SimpleITK`, `nibabel` — required for image IO and detection.
- `scipy` — optional; speeds up the atlas nearest-neighbor query
  (falls back to brute-force NumPy when absent).

The agent imports nothing from Slicer / VTK / Qt.
