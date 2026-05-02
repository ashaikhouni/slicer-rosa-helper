# rosa-agent CLI

A pure-Python command-line agent that runs the SEEG localization
pipeline end-to-end without Slicer.

```
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

```
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

```
trajectory       str
label            str    "<trajectory><index>" (e.g. L_AC1)
contact_index    int    1-based
x/y/z            float  RAS mm
peak_detected    int    1 = anchored on detected peak, 0 = model-nominal
electrode_model  str
```

### `labels.tsv`

```
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

```
/tmp/T22_cli/
    trajectories.tsv      ~9 entries
    contacts.tsv          ~117 contacts
    labels.tsv            (when --thomas/--freesurfer is passed)
```

## Example: ROSA case folder

```bash
python -m rosa_agent load /data/cases/RYAN_ANON --out /tmp/ryan_manifest.json
python -m rosa_agent pipeline /data/cases/RYAN_ANON \
    --ct /data/cases/RYAN_ANON/postop_ct.nii.gz \
    --out-dir /tmp/ryan_cli \
    --freesurfer /data/cases/RYAN_ANON/aparc+aseg.nii.gz \
    --freesurfer-lut $FREESURFER_HOME/FreeSurferColorLUT.txt
```

When a ROSA folder is passed, the parsed planned trajectories are used
as guided-fit seeds (RAS-converted from the ROSA-stored LPS).

## Dependencies

* `numpy`, `SimpleITK`, `nibabel` — required for image IO and detection.
* `scipy` — optional; speeds up the atlas nearest-neighbor query
  (falls back to brute-force NumPy when absent).

The agent imports nothing from Slicer / VTK / Qt.
