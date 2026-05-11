"""Generate slicer-rosa-helper/notebooks/v1_seeds_v2_placement_qc.ipynb.

Worked example: how to drive `rosa-agent place` in each of its five
modes against a single CT and inspect the QC results.

Builds the notebook from a Python source so the rendered .ipynb stays
small and reviewable in diffs. Re-run after edits:

    python tools/build_contact_placement_qc_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


def md(*lines: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _to_lines(lines),
    }


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _to_lines(lines),
    }


def _to_lines(lines):
    blob = "\n".join(dedent(s).strip("\n") for s in lines)
    if not blob:
        return []
    parts = blob.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


cells: list[dict] = []


cells.append(md("""
# rosa-agent place — five-mode CLI usage example

`rosa-agent place` is the headless entry point to the SEEG contact
placement pipeline. It accepts one CT and writes a standardized QC
directory (`manifest.json`, `trajectories.tsv`, `contacts.tsv`,
`figures/`, `diagnostics/cmp.tsv`).

The mode is chosen by which optional flags you pass:

| Mode | Flags                                  | What it does                                       |
| ---- | -------------------------------------- | -------------------------------------------------- |
| 1    | _(none)_                               | full auto: detect + place, return all survivors    |
| 2    | `--n-expected N`                       | mode 1, then keep top-N by confidence              |
| 3    | `--expected names.tsv`                 | mode 1, then assign each detection to a named slot |
| 4    | `--seeds seeds.tsv` (with model_id)    | place against caller-provided seeds + models       |
| 5    | `--seeds seeds.tsv` (no model_id)      | place against caller-provided seeds; pick model    |

This notebook runs each mode against a single CT, then loads the QC
output of each run and shows the trajectory tables side by side.

Resume context: rosa_detect/candidate_seeds + rosa_core/contact_placement
own the algorithm; cli/rosa_agent/commands/place.py wraps them.
"""))


cells.append(md("""
## Setup
"""))

cells.append(code("""
%matplotlib inline
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path.cwd()
while REPO.name != "slicer-rosa-helper" and REPO.parent != REPO:
    REPO = REPO.parent
sys.path.insert(0, str(REPO / "CommonLib"))
sys.path.insert(0, str(REPO / "tools"))
"""))


cells.append(md("""
## Subject + CT path

Pick any subject with a CT on disk. AMC subjects + T22 live under
`ROSA_AMC_TESTING_ROOT` (`<SID>/*_CT.nii.gz`); T-series subjects live
under `ROSA_SEEG_DATASET` (use `subjects.tsv` to resolve `ct_path`).

`SUBJECT_ID` is the only knob most readers will change.
"""))

cells.append(code("""
SUBJECT_ID = "T22"

AMC_ROOT  = Path(os.environ.get("ROSA_AMC_TESTING_ROOT", "/Users/ammar/Documents/testing"))
SEEG_ROOT = Path(os.environ.get("ROSA_SEEG_DATASET",   "/Users/ammar/Dropbox/thalamus_subjects/seeg_localization"))


def resolve_ct_and_strategy(sid: str):
    \"\"\"Return (ct_path, strategy_key, gt_shanks_or_None).\"\"\"
    amc_dir = AMC_ROOT / sid
    if amc_dir.is_dir():
        ct = next(iter(amc_dir.glob("*_CT.nii.gz")), None) or next(iter(amc_dir.glob("*.nii.gz")), None)
        if ct is None:
            raise FileNotFoundError(f"no CT found under {amc_dir}")
        strategy = "dixi" if sid == "T22" else "pmt_35"
        return str(ct), strategy, None

    # T-series — resolve via subjects.tsv
    from eval_seeg_localization import iter_subject_rows
    rows = iter_subject_rows(SEEG_ROOT, {sid})
    if not rows:
        raise FileNotFoundError(f"{sid} not in AMC root or T-series manifest")
    row = rows[0]
    ct_path = row.get("source_ct_file") or row["ct_path"]
    return ct_path, "dixi", None


CT_PATH, STRATEGY, _ = resolve_ct_and_strategy(SUBJECT_ID)
print(f"subject : {SUBJECT_ID}")
print(f"CT      : {CT_PATH}")
print(f"strategy: {STRATEGY}")

OUT_ROOT = REPO / "notebooks" / "_qc_output" / SUBJECT_ID
OUT_ROOT.mkdir(parents=True, exist_ok=True)
print(f"qc dirs : {OUT_ROOT}/mode{{1..5}}_qc/")
"""))


cells.append(md("""
## CLI invocation helper

`rosa-agent` is installed by the package's `pyproject.toml`. We invoke
it via `subprocess` so the cell's stdout/stderr captures the CLI's
own progress output verbatim — nothing the notebook does differs from
how the CLI runs in a terminal.
"""))

cells.append(code("""
ROSA_AGENT = shutil.which("rosa-agent") or sys.executable + " -m rosa_agent"


def run_place(*extra_args: str, output_subdir: str, quiet: bool = False) -> dict:
    \"\"\"Invoke `rosa-agent place` with the given extra args; return the
    parsed manifest.json from the output directory.\"\"\"
    out = OUT_ROOT / output_subdir
    if out.exists():
        shutil.rmtree(out)

    cmd = [
        *ROSA_AGENT.split(),
        "place",
        "--ct", CT_PATH,
        "--output", str(out),
        "--library", STRATEGY,
        "--subject-id", SUBJECT_ID,
        *extra_args,
    ]
    if quiet:
        cmd.append("--quiet")

    print("$", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("STDOUT:", proc.stdout)
        print("STDERR:", proc.stderr)
        raise RuntimeError(f"rosa-agent place exited {proc.returncode}")
    if proc.stderr.strip() and not quiet:
        # CLI writes progress to stderr — surface a few last lines so
        # the notebook captures any warnings.
        tail = proc.stderr.strip().splitlines()[-5:]
        print("\\n".join(tail))

    manifest = json.loads((out / "manifest.json").read_text())
    return manifest
"""))


cells.append(md("""
## Mode 1 — auto

No mode flag at all. The pipeline detects every shank it can find
and emits each one with a continuous confidence score and band
(`high` / `medium` / `low`). Use this when you have no prior on
shank count.
"""))

cells.append(code("""
mode1_manifest = run_place(output_subdir="mode1_qc")
mode1_traj = pd.read_csv(OUT_ROOT / "mode1_qc" / "trajectories.tsv", sep="\\t")
print(f"mode 1 emitted {len(mode1_traj)} trajectories")
mode1_traj[["name", "confidence_label", "confidence",
             "electrode_model", "bolt_source", "length_mm"]].head(20)
"""))


cells.append(md("""
## Mode 2 — count

Pass `--n-expected N` to keep only the top-N by confidence. Use this
when you know how many shanks were implanted but have no name list.
"""))

cells.append(code("""
EXPECTED_N = max(1, len(mode1_traj))   # set to your real implant count
mode2_manifest = run_place(
    "--n-expected", str(EXPECTED_N),
    output_subdir=f"mode2_qc",
)
mode2_traj = pd.read_csv(OUT_ROOT / "mode2_qc" / "trajectories.tsv", sep="\\t")
print(f"mode 2 emitted {len(mode2_traj)} trajectories (capped at {EXPECTED_N})")
mode2_traj[["name", "confidence_label", "confidence",
             "electrode_model", "bolt_source", "length_mm"]]
"""))


cells.append(md("""
## Mode 3 — named

Pass `--expected names.tsv` (`name<TAB>model_id`) to assign each
detection to a named slot. Auto-fit still picks the geometry; mode 3
just hands out the names you provide.
"""))

cells.append(code("""
expected_tsv = OUT_ROOT / "expected.tsv"
expected_lines = ["name\\tmodel_id"]
for i in range(min(8, len(mode1_traj))):
    expected_lines.append(f"L_{i+1:02d}\\tDIXI-15CM")
expected_tsv.write_text("\\n".join(expected_lines) + "\\n")

print(expected_tsv.read_text())
mode3_manifest = run_place(
    "--expected", str(expected_tsv),
    output_subdir="mode3_qc",
)
mode3_traj = pd.read_csv(OUT_ROOT / "mode3_qc" / "trajectories.tsv", sep="\\t")
print(f"mode 3 emitted {len(mode3_traj)} trajectories")
mode3_traj[["name", "confidence_label", "confidence",
             "electrode_model", "bolt_source", "length_mm"]]
"""))


cells.append(md("""
## Mode 4 — seeds with model_id

Pass `--seeds seeds.tsv` containing `name`, `start_x/y/z`, `end_x/y/z`,
`electrode_model`. The placer skips detection and aligns to the
caller's seeds, using the named electrode model as a strong prior.

For the demo we synthesize a 1-shank seed file from mode 1's first
emission so the cell runs without external input.
"""))

cells.append(code("""
seeds_tsv_with_model = OUT_ROOT / "seeds_with_model.tsv"
hdr = "name\\tstart_x\\tstart_y\\tstart_z\\tend_x\\tend_y\\tend_z\\telectrode_model"
rows = [hdr]
for _, t in mode1_traj.head(3).iterrows():
    rows.append("\\t".join([
        str(t["name"]),
        f"{t['start_x']:.3f}", f"{t['start_y']:.3f}", f"{t['start_z']:.3f}",
        f"{t['end_x']:.3f}",   f"{t['end_y']:.3f}",   f"{t['end_z']:.3f}",
        str(t.get("electrode_model") or "DIXI-15CM"),
    ]))
seeds_tsv_with_model.write_text("\\n".join(rows) + "\\n")

print(seeds_tsv_with_model.read_text())
mode4_manifest = run_place(
    "--seeds", str(seeds_tsv_with_model),
    output_subdir="mode4_qc",
)
mode4_traj = pd.read_csv(OUT_ROOT / "mode4_qc" / "trajectories.tsv", sep="\\t")
print(f"mode 4 placed {len(mode4_traj)} seeds (1 trajectory per input seed)")
mode4_traj[["name", "confidence_label", "confidence",
             "electrode_model", "bolt_source", "length_mm"]]
"""))


cells.append(md("""
## Mode 5 — seeds without model_id

Same as mode 4, but the seed TSV omits `electrode_model`. The picker
infers the best library model from each seed's geometry (PaCER →
walker signature → length-only fallback).
"""))

cells.append(code("""
seeds_tsv_no_model = OUT_ROOT / "seeds_no_model.tsv"
hdr = "name\\tstart_x\\tstart_y\\tstart_z\\tend_x\\tend_y\\tend_z"
rows = [hdr]
for _, t in mode1_traj.head(3).iterrows():
    rows.append("\\t".join([
        str(t["name"]),
        f"{t['start_x']:.3f}", f"{t['start_y']:.3f}", f"{t['start_z']:.3f}",
        f"{t['end_x']:.3f}",   f"{t['end_y']:.3f}",   f"{t['end_z']:.3f}",
    ]))
seeds_tsv_no_model.write_text("\\n".join(rows) + "\\n")

print(seeds_tsv_no_model.read_text())
mode5_manifest = run_place(
    "--seeds", str(seeds_tsv_no_model),
    output_subdir="mode5_qc",
)
mode5_traj = pd.read_csv(OUT_ROOT / "mode5_qc" / "trajectories.tsv", sep="\\t")
print(f"mode 5 placed {len(mode5_traj)} seeds (electrode_model picked per-seed)")
mode5_traj[["name", "confidence_label", "confidence",
             "electrode_model", "bolt_source", "length_mm"]]
"""))


cells.append(md("""
## Side-by-side comparison

For each mode, summarize: `(n_trajectories, mean_confidence, electrode_model_distribution)`.
"""))

cells.append(code("""
def summarize(name: str, df: pd.DataFrame) -> dict:
    if df.empty:
        return {"mode": name, "n": 0, "mean_conf": float("nan"), "models": ""}
    models = df["electrode_model"].fillna("?").value_counts().to_dict()
    return {
        "mode": name,
        "n": len(df),
        "mean_conf": float(df["confidence"].mean()),
        "models": ", ".join(f"{m}×{c}" for m, c in sorted(models.items(), key=lambda x: -x[1])),
    }

summary = pd.DataFrame([
    summarize("1 (auto)",     mode1_traj),
    summarize("2 (count)",    mode2_traj),
    summarize("3 (named)",    mode3_traj),
    summarize("4 (seeds+model)", mode4_traj),
    summarize("5 (seeds)",    mode5_traj),
])
summary
"""))


cells.append(md("""
## QC artifacts on disk

Each run wrote a self-contained QC directory. Inspect / share these
without re-running detection:

```
notebooks/_qc_output/<SUBJECT_ID>/
  mode1_qc/
    manifest.json        ← run config + per-trajectory band counts
    trajectories.tsv     ← one row per emitted trajectory
    contacts.tsv         ← one row per placed contact
    figures/             ← per-trajectory PNG (CT slab + axis + peaks)
    diagnostics/cmp.tsv  ← per-stage diagnostics (corr, n_covered, ...)
  mode2_qc/
  ...
```

The `figures/` directory is the visual QC: each PNG shows the CT
slab around one trajectory with the placed contacts overlaid.
"""))

cells.append(code("""
import IPython.display as disp

mode = "mode1_qc"
fig_dir = OUT_ROOT / mode / "figures"
if fig_dir.exists():
    pngs = sorted(fig_dir.glob("*.png"))[:3]
    print(f"showing first {len(pngs)} of {len(list(fig_dir.glob('*.png')))} figures from {mode}")
    for p in pngs:
        display(disp.Image(filename=str(p)))
else:
    print(f"no figures (matplotlib missing in CLI env, or --no-figures was passed)")
"""))


cells.append(md("""
## Re-running

Every cell is idempotent — re-running it deletes that mode's QC
directory and writes it fresh. Change `SUBJECT_ID` at the top and
"Run All" to point the whole notebook at a different CT.
"""))


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    out_path = (
        Path(__file__).resolve().parent.parent
        / "notebooks"
        / "v1_seeds_v2_placement_qc.ipynb"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
