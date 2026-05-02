"""rosa-agent pipeline — run load → detect → contacts → label end-to-end.

Two input modes:

* ROSA folder path: ``rosa-agent pipeline /path/to/case --ct /path/to/ct.nii``
  → parses the .ros file into a manifest, then runs detect/contacts/label
  on the supplied CT.

* SEEG dataset subject id (e.g. ``T22``): when the argument is not an
  existing path AND ``$ROSA_SEEG_DATASET`` is set, looks up
  ``$ROSA_SEEG_DATASET/contact_label_dataset/subjects.tsv`` for the
  ``ct_path`` and skips the .ros load (the dataset has no ROSA folder
  per subject — just a registered CT). Useful for the regression
  acceptance test.

Output (``--out-dir DIR``):

    DIR/
        manifest.json     (only when input is a ROSA folder)
        trajectories.tsv
        contacts.tsv
        labels.tsv        (only when atlas providers are configured)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from ..io.manifest import write_manifest
from ..io.trajectory_io import (
    TRAJECTORY_COLUMNS,
    read_seeds_tsv,
    read_tsv_rows,
    write_contacts_tsv,
    write_trajectories_tsv,
)
from .contacts import place_contacts
from .detect import run_auto_detect, run_guided_detect
from .label import LABEL_COLUMNS, _build_providers, label_contacts
from .load import build_manifest_from_ros


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def _resolve_inputs(target: str, ct_override: str | None) -> tuple[Path, dict[str, Any] | None]:
    """Return (ct_path, manifest_or_None) given the user-supplied target."""
    target_path = Path(target).expanduser()

    if target_path.exists() and target_path.is_dir():
        manifest = build_manifest_from_ros(target_path)
        if ct_override:
            return Path(ct_override).expanduser().resolve(), manifest
        # No CT auto-discovery yet (the .ros file references Analyze
        # volumes via tokens, not the postop CT). Require --ct.
        raise SystemExit(
            f"ROSA folder mode: pass --ct <ct_path>; the .ros file "
            f"doesn't directly point at the postop CT."
        )

    # Dataset subject id — look up subjects.tsv.
    dataset_root = os.environ.get("ROSA_SEEG_DATASET")
    if not dataset_root:
        raise SystemExit(
            f"target {target!r} is neither a folder nor a dataset id "
            f"(ROSA_SEEG_DATASET unset)."
        )
    manifest_path = Path(dataset_root) / "contact_label_dataset" / "subjects.tsv"
    if not manifest_path.exists():
        raise SystemExit(
            f"dataset manifest missing: {manifest_path}"
        )
    rows = read_tsv_rows(manifest_path)
    matches = [r for r in rows if str(r.get("subject_id")) == target]
    if not matches:
        raise SystemExit(f"subject {target!r} not in {manifest_path}")
    ct_path = ct_override or matches[0]["ct_path"]
    return Path(ct_path).expanduser().resolve(), None


def run_pipeline(
    target: str,
    *,
    out_dir: str | Path,
    ct_override: str | None = None,
    seeds_path: str | None = None,
    thomas_dir: str | None = None,
    freesurfer_path: str | None = None,
    freesurfer_lut: str | None = None,
    wm_path: str | None = None,
    wm_lut: str | None = None,
) -> dict[str, Any]:
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    ct_path, manifest = _resolve_inputs(target, ct_override)

    if manifest is not None:
        manifest_path = out / "manifest.json"
        write_manifest(manifest_path, manifest)
        _stderr(f"[pipeline] wrote {manifest_path}")

    # Detect.
    if seeds_path:
        seeds = read_seeds_tsv(seeds_path)
        _stderr(f"[pipeline] guided detect with {len(seeds)} seeds")
        trajs = run_guided_detect(ct_path, seeds, run_id=f"pipeline_{target}")
    elif manifest is not None and manifest.get("planned_trajectories"):
        # Use the planned trajectories from the .ros file as seeds.
        seeds = [
            {"name": t["name"], "start_ras": t["start_ras"], "end_ras": t["end_ras"]}
            for t in manifest["planned_trajectories"]
        ]
        _stderr(f"[pipeline] guided detect using {len(seeds)} planned-from-ROSA seeds")
        trajs = run_guided_detect(ct_path, seeds, run_id=f"pipeline_{target}")
    else:
        _stderr("[pipeline] auto detect")
        result = run_auto_detect(ct_path, run_id=f"pipeline_{target}")
        if str(result.get("status")) == "error":
            err = dict(result.get("error") or {})
            raise SystemExit(
                f"[pipeline] detection failed: {err.get('message')} "
                f"(stage={err.get('stage')})"
            )
        trajs = list(result.get("trajectories") or [])

    traj_path = out / "trajectories.tsv"
    n_traj = write_trajectories_tsv(traj_path, trajs)
    _stderr(f"[pipeline] wrote {traj_path} ({n_traj} trajectories)")

    # Contacts.
    traj_seeds = read_seeds_tsv(traj_path)
    contact_groups = place_contacts(ct_path, traj_seeds)
    contacts_path = out / "contacts.tsv"
    n_contacts = write_contacts_tsv(contacts_path, contact_groups)
    _stderr(f"[pipeline] wrote {contacts_path} ({n_contacts} contacts)")

    # Label (skipped silently when no provider configured).
    label_summary = {"written": False, "n_contacts": n_contacts}
    if any([thomas_dir, freesurfer_path, wm_path]):
        contacts = read_tsv_rows(contacts_path)
        providers = _build_providers(
            thomas_dir=thomas_dir,
            freesurfer_path=freesurfer_path,
            freesurfer_lut=freesurfer_lut,
            wm_path=wm_path,
            wm_lut=wm_lut,
        )
        if any(p is not None and p.is_ready() for p in providers.values()):
            from .label import label_contacts as _label_contacts
            label_rows = _label_contacts(contacts, providers)
            labels_path = out / "labels.tsv"
            from ..io.trajectory_io import write_tsv_rows as _write
            _write(labels_path, label_rows, LABEL_COLUMNS)
            label_summary = {"written": True, "n_contacts": len(label_rows)}
            _stderr(f"[pipeline] wrote {labels_path} ({len(label_rows)} contacts)")
        else:
            _stderr("[pipeline] no atlas providers became ready, skipping label step")
    else:
        _stderr("[pipeline] no --thomas/--freesurfer/--wm passed, skipping label step")

    return {
        "ct_path": str(ct_path),
        "out_dir": str(out),
        "n_trajectories": n_traj,
        "n_contacts": n_contacts,
        "labels": label_summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rosa-agent pipeline",
        description="Run load → detect → contacts → label on a ROSA folder or dataset subject.",
    )
    parser.add_argument("target", help="ROSA case folder, OR dataset subject id (e.g. T22)")
    parser.add_argument("--ct", default="", help="CT NIfTI/NRRD path (overrides dataset/manifest lookup)")
    parser.add_argument("--seeds", default="", help="Optional seed TSV → guided fit")
    parser.add_argument("--out-dir", "-o", required=True)
    parser.add_argument("--thomas", default="", help="THOMAS segmentation directory")
    parser.add_argument("--freesurfer", default="", help="FreeSurfer parcellation labelmap")
    parser.add_argument("--freesurfer-lut", default="", help="FreeSurfer LUT")
    parser.add_argument("--wm", default="", help="White-matter labelmap")
    parser.add_argument("--wm-lut", default="", help="WM LUT")
    args = parser.parse_args(argv)

    summary = run_pipeline(
        args.target,
        out_dir=args.out_dir,
        ct_override=args.ct or None,
        seeds_path=args.seeds or None,
        thomas_dir=args.thomas or None,
        freesurfer_path=args.freesurfer or None,
        freesurfer_lut=args.freesurfer_lut or None,
        wm_path=args.wm or None,
        wm_lut=args.wm_lut or None,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
