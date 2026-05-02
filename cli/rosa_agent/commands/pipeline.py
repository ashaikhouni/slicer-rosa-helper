"""rosa-agent pipeline — run load → detect → contacts → label end-to-end.

Three input modes:

* **SEEG dataset subject id** (e.g. ``T22``): when the argument is not
  an existing path AND ``$ROSA_SEEG_DATASET`` is set, look up
  ``$ROSA_SEEG_DATASET/contact_label_dataset/subjects.tsv`` for the
  ``ct_path``. Auto-detect on that CT.

* **ROSA folder path, no external CT**: load one of the displays from
  the ROSA folder as the working CT (controlled by ``--ref-volume``;
  defaults to the first display). Use the .ros planned trajectories
  as guided-fit seeds. Working frame == ROSA reference frame; outputs
  land there.

* **ROSA folder path + external CT** (``--ct external.nii.gz``):
  register the external CT to the chosen ROSA reference volume
  (rigid + Mattes MI, mirroring BRAINSFit), inverse-transform the
  ROSA-derived seeds into the external CT frame, run detection in CT
  frame natively. Default output frame is the external CT frame; pass
  ``--output-frame rosa`` to push outputs back into the ROSA frame.

Output (``--out-dir DIR``):

    DIR/
        trajectories.tsv
        contacts.tsv
        manifest.json         (only when input is a ROSA folder)
        labels.tsv            (only when atlas providers are configured)
        ct.nii.gz             (only when the working CT was converted
                               from Analyze inside the ROSA folder —
                               there's no value in copying a CT that's
                               already a NIfTI on disk)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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


# ---------------------------------------------------------------------
# Frame model
# ---------------------------------------------------------------------


@dataclass
class _PipelineFrame:
    """Resolved working state before detection runs.

    ``working_ct_path`` is always a NIfTI on disk (we write one if the
    source was Analyze). Detection always runs in the working CT's RAS
    frame; ``rosa_to_working_4x4`` carries any RAS->RAS transform
    needed to bring ROSA-derived seeds into that frame (identity when
    the working CT IS the ROSA reference).
    """

    working_ct_path: Path
    rosa_to_working_4x4: np.ndarray
    seeds: list[dict[str, Any]]   # ROSA seeds already pushed into working frame
    manifest: dict[str, Any] | None
    rosa_reference_volume: str | None  # name of ROSA display chosen as reference


def _apply_4x4_to_point(point_ras, m_4x4) -> list[float]:
    h = np.array([float(point_ras[0]), float(point_ras[1]), float(point_ras[2]), 1.0])
    out = (m_4x4 @ h)[:3]
    return [float(v) for v in out]


def _apply_4x4_to_seeds(seeds, m_4x4):
    """Push start/end RAS of each seed through a 4x4."""
    out = []
    for s in seeds:
        out.append({
            **s,
            "start_ras": _apply_4x4_to_point(s["start_ras"], m_4x4),
            "end_ras": _apply_4x4_to_point(s["end_ras"], m_4x4),
        })
    return out


def _apply_4x4_to_trajectories(trajs, m_4x4):
    """Push start/end RAS (and skull_entry / bolt_tip when present) of
    each detected trajectory through a 4x4. Used to push results from
    the working frame back into a different output frame.
    """
    out = []
    for t in trajs:
        new = dict(t)
        if t.get("start_ras") is not None:
            new["start_ras"] = _apply_4x4_to_point(t["start_ras"], m_4x4)
        if t.get("end_ras") is not None:
            new["end_ras"] = _apply_4x4_to_point(t["end_ras"], m_4x4)
        for opt_key in ("skull_entry_ras", "bolt_tip_ras"):
            if t.get(opt_key) is not None:
                new[opt_key] = _apply_4x4_to_point(t[opt_key], m_4x4)
        # path_ras (curved trajectories) — push every control point.
        path = t.get("path_ras")
        if path:
            new["path_ras"] = [
                _apply_4x4_to_point(p, m_4x4) for p in path
            ]
        out.append(new)
    return out


def _apply_4x4_to_contact_groups(groups, m_4x4):
    """Push contact positions through a 4x4."""
    out = []
    for g in groups:
        new = dict(g)
        new["positions_ras"] = [
            _apply_4x4_to_point(p, m_4x4) for p in g.get("positions_ras", [])
        ]
        out.append(new)
    return out


# ---------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------


def _resolve_dataset_subject(target: str, ct_override: str | None) -> Path:
    """Look up a dataset subject id in ROSA_SEEG_DATASET/contact_label_dataset
    and return the CT path. Used by the dataset-id input mode.
    """
    dataset_root = os.environ.get("ROSA_SEEG_DATASET")
    if not dataset_root:
        raise SystemExit(
            f"target {target!r} is neither a folder nor a dataset id "
            f"(ROSA_SEEG_DATASET unset)."
        )
    manifest_path = Path(dataset_root) / "contact_label_dataset" / "subjects.tsv"
    if not manifest_path.exists():
        raise SystemExit(f"dataset manifest missing: {manifest_path}")
    rows = read_tsv_rows(manifest_path)
    matches = [r for r in rows if str(r.get("subject_id")) == target]
    if not matches:
        raise SystemExit(f"subject {target!r} not in {manifest_path}")
    return Path(ct_override or matches[0]["ct_path"]).expanduser().resolve()


def _resolve_pipeline_frame(
    target: str,
    *,
    out_dir: Path,
    ct_override: str | None,
    ref_volume: str | None,
    skip_registration: bool,
) -> _PipelineFrame:
    """Decide on the working CT, load it as NIfTI, and return seeds + the
    transform pushing ROSA RAS -> working CT RAS.
    """
    target_path = Path(target).expanduser()

    if not (target_path.exists() and target_path.is_dir()):
        # Dataset subject id mode — no ROSA folder, no seeds. The CT
        # is already a NIfTI on disk; no value in copying it.
        ct_path = _resolve_dataset_subject(target, ct_override)
        return _PipelineFrame(
            working_ct_path=ct_path,
            rosa_to_working_4x4=np.eye(4),
            seeds=[],
            manifest=None,
            rosa_reference_volume=None,
        )

    # ROSA folder mode.
    from rosa_core import load_rosa_volume_as_sitk
    import SimpleITK as sitk

    # Load the chosen ROSA reference volume (defaults to first display).
    ref_img, ref_meta = load_rosa_volume_as_sitk(
        str(target_path), volume_name=ref_volume,
    )
    rosa_ref_name = ref_meta["loaded_volume"]
    _stderr(f"[pipeline] ROSA reference volume: {rosa_ref_name}")

    # Build a manifest dict (same shape build_manifest_from_ros emits)
    # so the existing manifest writer just works.
    manifest = build_manifest_from_ros(target_path)

    # ROSA seeds, in the ROSA reference frame (RAS).
    rosa_seeds = [
        {"name": t["name"], "start_ras": t["start_ras"], "end_ras": t["end_ras"]}
        for t in ref_meta["trajectories"]
    ]
    _stderr(f"[pipeline] {len(rosa_seeds)} planned trajectories from .ros")

    if not ct_override:
        # No external CT — ROSA reference IS the working CT.
        out_ct = out_dir / "ct.nii.gz"
        sitk.WriteImage(ref_img, str(out_ct))
        _stderr(f"[pipeline] wrote ROSA reference as working CT: {out_ct}")
        return _PipelineFrame(
            working_ct_path=out_ct,
            rosa_to_working_4x4=np.eye(4),
            seeds=rosa_seeds,
            manifest=manifest,
            rosa_reference_volume=rosa_ref_name,
        )

    # External CT supplied — register CT to ROSA reference. The CT
    # itself isn't transformed (we transform seeds instead, per the
    # frame model in run_pipeline's docstring), so there's no value in
    # copying the user's CT into out_dir.
    ct_path = Path(ct_override).expanduser().resolve()
    ct_img = sitk.ReadImage(str(ct_path))

    if skip_registration:
        _stderr(
            "[pipeline] --skip-registration: assuming external CT is "
            "already aligned to the ROSA reference frame"
        )
        rosa_to_working = np.eye(4)
    else:
        from rosa_core.registration import register_rigid_mi
        _stderr(
            f"[pipeline] registering external CT to ROSA reference "
            f"{rosa_ref_name!r} (rigid + Mattes MI)…"
        )
        # fixed = external CT, moving = ROSA reference.
        # We want to push ROSA-frame seeds INTO the CT frame: that's
        # the moving -> fixed direction. moving_to_fixed_ras_4x4 is
        # the SITK transform's *inverse*, which is exactly what we
        # want here (SITK's transform itself maps fixed -> moving for
        # use with sitk.Resample).
        reg_result = register_rigid_mi(
            fixed=ct_img, moving=ref_img,
            logger=_stderr,
        )
        _stderr(
            f"[pipeline] registration done: metric={reg_result.final_metric:+.5f} "
            f"iters={reg_result.n_iterations} ({reg_result.converged_reason})"
        )
        rosa_to_working = reg_result.moving_to_fixed_ras_4x4

    seeds_in_ct = _apply_4x4_to_seeds(rosa_seeds, rosa_to_working)
    return _PipelineFrame(
        # External CT path: working CT IS the user's CT on disk; we
        # don't copy or rewrite it. ``out_ct`` only exists in the
        # other branch (where we wrote the ROSA reference as NIfTI).
        working_ct_path=ct_path,
        rosa_to_working_4x4=rosa_to_working,
        seeds=seeds_in_ct,
        manifest=manifest,
        rosa_reference_volume=rosa_ref_name,
    )


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------


def run_pipeline(
    target: str,
    *,
    out_dir: str | Path,
    ct_override: str | None = None,
    seeds_path: str | None = None,
    ref_volume: str | None = None,
    output_frame: str = "ct",
    skip_registration: bool = False,
    thomas_dir: str | None = None,
    freesurfer_path: str | None = None,
    freesurfer_lut: str | None = None,
    wm_path: str | None = None,
    wm_lut: str | None = None,
    atlas_base_path: str | None = None,
) -> dict[str, Any]:
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    frame = _resolve_pipeline_frame(
        target,
        out_dir=out,
        ct_override=ct_override,
        ref_volume=ref_volume,
        skip_registration=skip_registration,
    )
    ct_path = frame.working_ct_path

    if frame.manifest is not None:
        manifest_path = out / "manifest.json"
        write_manifest(manifest_path, frame.manifest)
        _stderr(f"[pipeline] wrote {manifest_path}")

    # Detection.
    if seeds_path:
        seeds = read_seeds_tsv(seeds_path)
        _stderr(f"[pipeline] guided detect with {len(seeds)} explicit seeds (from {seeds_path})")
        trajs = run_guided_detect(ct_path, seeds, run_id=f"pipeline_{target}")
    elif frame.seeds:
        _stderr(
            f"[pipeline] guided detect using {len(frame.seeds)} ROSA-planned seeds "
            f"(transformed into working frame)"
        )
        trajs = run_guided_detect(ct_path, frame.seeds, run_id=f"pipeline_{target}")
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

    # Contacts (placement always runs in the working CT frame).
    contact_groups = place_contacts(ct_path, [
        {"name": t.get("name", f"T{idx:02d}"),
         "start_ras": t["start_ras"],
         "end_ras": t["end_ras"],
         "electrode_model": t.get("electrode_model") or ""}
        for idx, t in enumerate(trajs, start=1)
    ])

    # Output-frame transform. Working frame is the working CT's RAS;
    # we may want results in the ROSA reference frame instead.
    output_label = "ct"
    if output_frame and output_frame.lower() != "ct":
        if frame.manifest is None:
            _stderr(
                f"[pipeline] --output-frame {output_frame!r} requested but "
                f"input has no ROSA folder — keeping CT frame"
            )
        else:
            # Currently supported: "rosa" (push back to ROSA reference frame).
            # Named-display targeting is deferred to a follow-up.
            requested = output_frame.lower()
            if requested != "rosa":
                _stderr(
                    f"[pipeline] --output-frame {output_frame!r} not yet "
                    f"supported; using 'rosa'"
                )
            working_to_rosa = np.linalg.inv(frame.rosa_to_working_4x4)
            trajs = _apply_4x4_to_trajectories(trajs, working_to_rosa)
            contact_groups = _apply_4x4_to_contact_groups(contact_groups, working_to_rosa)
            output_label = f"rosa({frame.rosa_reference_volume or 'ref'})"
            _stderr(
                f"[pipeline] outputs transformed back to ROSA reference frame "
                f"({frame.rosa_reference_volume!r})"
            )

    # Write outputs.
    traj_path = out / "trajectories.tsv"
    n_traj = write_trajectories_tsv(traj_path, trajs)
    _stderr(f"[pipeline] wrote {traj_path} ({n_traj} trajectories, frame={output_label})")

    contacts_path = out / "contacts.tsv"
    n_contacts = write_contacts_tsv(contacts_path, contact_groups)
    _stderr(f"[pipeline] wrote {contacts_path} ({n_contacts} contacts, frame={output_label})")

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
            atlas_base_path=atlas_base_path,
            # The working CT is the natural target volume — that's the
            # frame the contacts (and any --output-frame transform) live in.
            target_volume_path=str(ct_path) if atlas_base_path else None,
        )
        if any(p is not None and p.is_ready() for p in providers.values()):
            label_rows = label_contacts(contacts, providers)
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
        "output_frame": output_label,
        "rosa_reference_volume": frame.rosa_reference_volume,
        "registration_applied": (
            frame.manifest is not None
            and ct_override is not None
            and not skip_registration
        ),
        "labels": label_summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rosa-agent pipeline",
        description="Run load → detect → contacts → label on a ROSA folder or dataset subject.",
    )
    parser.add_argument(
        "target",
        help="ROSA case folder, OR dataset subject id (e.g. T22)",
    )
    parser.add_argument("--ct", default="", help="External CT NIfTI/NRRD (overrides ROSA-folder reference + dataset lookup)")
    parser.add_argument(
        "--ref-volume", default="",
        help="Name of the ROSA-folder display to use as the reference frame (default: first display)",
    )
    parser.add_argument("--seeds", default="", help="Optional seed TSV — overrides ROSA-derived seeds")
    parser.add_argument(
        "--output-frame", default="ct", choices=("ct", "rosa"),
        help="Frame for output trajectory/contact coordinates (default: ct = working CT frame)",
    )
    parser.add_argument(
        "--skip-registration", action="store_true",
        help="When --ct is supplied alongside a ROSA folder, assume the external CT is "
             "already aligned to the ROSA reference (no rigid registration run)",
    )
    parser.add_argument("--out-dir", "-o", required=True)
    parser.add_argument("--thomas", default="", help="THOMAS segmentation directory")
    parser.add_argument("--freesurfer", default="", help="FreeSurfer parcellation labelmap")
    parser.add_argument("--freesurfer-lut", default="", help="FreeSurfer LUT")
    parser.add_argument("--wm", default="", help="White-matter labelmap")
    parser.add_argument("--wm-lut", default="", help="WM LUT")
    parser.add_argument(
        "--atlas-base", default="",
        help="T1 / base volume the FreeSurfer / WM atlases were reconned on. "
             "When passed, the FS / WM labelmaps are rigidly registered + "
             "resampled onto the working-CT grid before sampling, so atlas "
             "labels align with contacts that live in CT RAS.",
    )
    args = parser.parse_args(argv)

    summary = run_pipeline(
        args.target,
        out_dir=args.out_dir,
        ct_override=args.ct or None,
        seeds_path=args.seeds or None,
        ref_volume=args.ref_volume or None,
        output_frame=args.output_frame,
        skip_registration=bool(args.skip_registration),
        thomas_dir=args.thomas or None,
        freesurfer_path=args.freesurfer or None,
        freesurfer_lut=args.freesurfer_lut or None,
        wm_path=args.wm or None,
        wm_lut=args.wm_lut or None,
        atlas_base_path=args.atlas_base or None,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
