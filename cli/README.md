# rosa-agent CLI

A pure-Python command-line agent that runs the SEEG localization
pipeline end-to-end without Slicer.

```bash
python -m rosa_agent <subcommand> ...
```

Subcommands:

| Command   | Purpose                                                     |
|-----------|-------------------------------------------------------------|
| `load`    | Parse a ROSA case folder into a JSON manifest               |
| `detect`  | Run shank detection (auto or guided) on a postop CT volume  |
| `contacts`| Place contacts along trajectories using LoG-driven peaks    |
| `label`   | Assign atlas labels to a contacts TSV                       |
| `pipeline`| Run all four stages end-to-end                              |

## Inputs

The agent assumes the [CommonLib](../CommonLib/) directory is on
`PYTHONPATH`. The package adds it automatically when invoked as
`python -m rosa_agent`.

Coordinates are RAS millimeters except inside the JSON manifest's
`planned_trajectories` which exposes both `*_lps` (the raw ROSA frame)
and `*_ras` keys.

## Output TSV columns (the public contract)

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

These columns are stable. Add new ones at the end if you need to extend.

## Example: end-to-end on the SEEG dataset

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

## Example: ROSA case folder

Three flavors depending on whether you want detection on a ROSA-embedded
volume, an external volume that's already aligned, or an external volume
that needs registration.

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

## Dependencies

* `numpy`, `SimpleITK`, `nibabel` — required for image IO and detection.
* `scipy` — optional; speeds up the atlas nearest-neighbor query
  (falls back to brute-force NumPy when absent).

The agent imports nothing from Slicer / VTK / Qt.
