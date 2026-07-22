"""rosa-agent contacts — place contacts along supplied trajectories.

Thin wrapper around ``rosa_core.placement_modes.place_seeg``: reads a
trajectory TSV (one row per shank, RAS endpoints + optional model_id),
runs the staged pipeline, writes contacts to TSV.

Mode dispatch (mirrors ``rosa-agent place``):

  * Every row has ``electrode_model`` set → mode 4 (vouched + force model).
  * Any row missing ``electrode_model`` → mode 5 (snap-to-v1 + library match).

This subcommand is kept for back-compat with ``rosa-agent pipeline`` and
external scripts that produce a ``rosa-agent detect`` TSV. New users should
prefer ``rosa-agent place --seeds traj.tsv --output qc/`` which writes a
full QC directory instead of just the contacts TSV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ..io.trajectory_io import (
    read_seeds_tsv,
    write_contacts_tsv,
)


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def place_contacts(
    ct_path: str | Path,
    trajectories: list[dict[str, Any]],
    *,
    pitch_strategy: str | None = None,
    mask_backend: str = "auto",
    brain_mask: Any = None,
    synthstrip_path: str | None = None,
):
    """Run the staged placement pipeline for each trajectory.

    Returns ``(contact_groups, batch)``:

      * ``contact_groups`` — list of dicts in the format expected by
        ``write_contacts_tsv``::

            {"trajectory": <name>, "electrode_model": <id>,
             "positions_ras": [...], "peak_detected": [...]}

      * ``batch`` — the full ``PlacementBatch`` from ``place_seeg`` so
        downstream callers can render QC figures (``batch.features`` +
        ``batch.bolts`` are the inputs ``rosa_core.qc_figures`` needs
        without re-loading the CT). ``None`` when there were no seeds.

    Mode dispatch:
      * Trajectories with ``electrode_model`` → mode 4 (force the vouched
        model on each).
      * Trajectories missing ``electrode_model`` → mode 5 (snap-to-v1
        candidate, library match).

    Mixed inputs route through mode 5 (the dispatcher's strictness rule
    requires every seed to have a model_id for mode 4); rows that come
    in with a model_id still get model-filtered placement via the snap
    path on the way out.

    Args:
        ct_path: CT volume — file path consumed by SimpleITK.
        trajectories: list of dicts with ``name``, ``start_ras``,
            ``end_ras`` (3-tuples or 3-lists), optional ``electrode_model``.
        pitch_strategy: library subset key (e.g. ``"dixi"``, ``"pmt_35"``).
            ``None`` (default) uses the full bundled library.
    """
    from rosa_core.placement_modes import Seed, place_seeg
    from rosa_core.contact_placement import sample_neg_log_max

    seeds = []
    for traj in trajectories:
        name = str(traj.get("name") or "")
        model = (traj.get("electrode_model") or None)
        seeds.append(Seed(
            name=name,
            start_ras=traj["start_ras"],
            end_ras=traj["end_ras"],
            model_id=str(model) if model else None,
        ))

    if not seeds:
        return [], None

    batch = place_seeg(
        str(ct_path),
        seeds=seeds,
        library=pitch_strategy,
        sample_fn=sample_neg_log_max,
        mask_backend=mask_backend,
        brain_mask=brain_mask,
        synthstrip_path=synthstrip_path,
        progress_logger=_stderr,
    )

    # Sanity bound: place_seeg can DIVERGE for a low-confidence guided-fit seed (a
    # failed snap) and emit contacts hundreds of mm outside the scan. A contact
    # can't lie outside the CT — drop those, so one bad shank doesn't poison the
    # export or blow up the 3-D scene's camera framing. Distance-based (frame-
    # invariant): centre = CT centre in RAS, radius = half-diagonal + 50 mm margin.
    import numpy as np
    import SimpleITK as sitk
    _img = sitk.ReadImage(str(ct_path))
    _sz = np.array(_img.GetSize(), dtype=float)
    _sp = np.array(_img.GetSpacing(), dtype=float)
    _c_lps = np.array(_img.TransformContinuousIndexToPhysicalPoint((_sz / 2.0).tolist()))
    _center = _c_lps * np.array([-1.0, -1.0, 1.0])              # LPS → RAS
    _max_r = 0.5 * float(np.linalg.norm(_sz * _sp)) + 50.0

    out: list[dict[str, Any]] = []
    for placed in batch.trajectories:
        contacts_ras = [list(c) for c in placed.contacts_ras]
        kept = [c for c in contacts_ras
                if float(np.linalg.norm(np.asarray(c, dtype=float) - _center)) <= _max_r]
        if len(kept) != len(contacts_ras):
            _stderr(f"[contacts] {placed.name}: dropped {len(contacts_ras) - len(kept)} "
                    f"contact(s) outside the CT (diverged fit)")
        contacts_ras = kept
        if not contacts_ras:
            _stderr(f"[contacts] {placed.name}: no contacts placed (band={placed.band})")
        out.append({
            "trajectory": placed.name,
            "electrode_model": placed.model_id or "",
            "positions_ras": contacts_ras,
            # The staged placer always emits library-template-anchored
            # contacts (no nominal-vs-detected distinction); mark all as
            # peak_detected for back-compat with the TSV schema.
            "peak_detected": [True] * len(contacts_ras),
        })
    return out, batch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rosa-agent contacts",
        description=(
            "Place contacts along each trajectory using the staged pipeline. "
            "Prefer 'rosa-agent place --seeds traj.tsv --output DIR' for new "
            "workflows (writes a full QC directory)."
        ),
    )
    parser.add_argument("trajectories_tsv", help="Trajectory TSV (rosa-agent detect output)")
    parser.add_argument("ct_path", help="CT NIfTI/NRRD")
    parser.add_argument("--out", "-o", required=True, help="Output contacts TSV")
    parser.add_argument(
        "--library", default=None,
        help="Pitch-strategy / library subset key (e.g. 'dixi', 'pmt_35'). "
             "Default: full library.",
    )
    parser.add_argument(
        "--electrode-types", default=None,
        help="Comma-separated electrode TYPES to constrain the library to "
             "(e.g. 'AM,PMT,CM' — the types a site actually uses). Restricts "
             "model matching to those types; takes precedence over --library. "
             "Default: the full bundled library.",
    )
    parser.add_argument(
        "--mask-backend", choices=("auto", "hull", "log-watershed", "synthstrip"),
        default="auto",
        help="intracranial brain-mask backend for the placement anchor. 'auto' "
             "(default) = SynthStrip-if-available → LoG-watershed; 'hull' = fast "
             "head-distance approximation; 'log-watershed' = CT watershed; "
             "'synthstrip' = force FreeSurfer SynthStrip.")
    parser.add_argument(
        "--synthstrip", default=None,
        help="explicit path to the mri_synthstrip binary (else probed via "
             "$ROSA_SYNTHSTRIP / $FREESURFER_HOME/bin / PATH)")
    parser.add_argument(
        "--brain-mask", default=None,
        help="path to a user-supplied intracranial mask volume "
             "(overrides --mask-backend; resampled to the CT grid)")
    parser.add_argument(
        "--fit-line-to-contacts", action="store_true",
        help="after placing, rewrite the input trajectory TSV so each shank's "
             "line is the PCA fit through its placed contacts (the line 'carries' "
             "its contacts, matching the 3-D scene). Refits in place.")
    args = parser.parse_args(argv)

    brain_mask_img = None
    if args.brain_mask:
        bm_path = Path(args.brain_mask)
        if not bm_path.exists():
            _stderr(f"error: brain mask not found: {bm_path}")
            return 2
        import SimpleITK as sitk
        brain_mask_img = sitk.ReadImage(str(bm_path))
        _stderr(f"[contacts] using user brain mask {bm_path} (overrides --mask-backend)")

    # Library constraint: --electrode-types filters the bundled library to the
    # site's types (an explicit model list place_seeg uses verbatim); it wins
    # over --library. Otherwise --library (strategy key) / full library applies.
    library = args.library
    if args.electrode_types:
        from rosa_core.electrode_models import load_electrode_library
        allowed = {t.strip() for t in args.electrode_types.split(",") if t.strip()}
        models = [m for m in load_electrode_library()["models"]
                  if str(m.get("type") or "") in allowed]
        if not models:
            _stderr(f"error: no electrode models match types {sorted(allowed)}")
            return 2
        library = models
        _stderr(f"[contacts] library constrained to {len(models)} models "
                f"of types {sorted(allowed)}")

    trajs = read_seeds_tsv(args.trajectories_tsv)
    _stderr(f"[contacts] {len(trajs)} trajectories from {args.trajectories_tsv}")
    groups, _batch = place_contacts(
        args.ct_path, trajs, pitch_strategy=library,
        mask_backend=args.mask_backend, brain_mask=brain_mask_img,
        synthstrip_path=args.synthstrip,
    )
    n = write_contacts_tsv(args.out, groups)
    _stderr(f"[contacts] wrote {args.out} ({n} contacts)")

    if args.fit_line_to_contacts:
        # Placement snapped the contacts onto the metal, off the detected seed
        # line — refit each trajectory line to the PCA axis through its contacts
        # so trajectories.tsv (and every downstream viewer) carries them.
        from rosa_core.trajectory_fit import refit_trajectories_file
        n_refit = refit_trajectories_file(args.trajectories_tsv, args.out)
        _stderr(f"[contacts] refit {n_refit} trajectory line(s) in "
                f"{args.trajectories_tsv} to their contacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
