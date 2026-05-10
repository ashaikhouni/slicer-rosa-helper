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
) -> list[dict[str, Any]]:
    """Run the staged placement pipeline for each trajectory.

    Returns a list of contact-group dicts in the format expected by
    ``write_contacts_tsv``::

        {"trajectory": <name>, "electrode_model": <id>,
         "positions_ras": [...], "peak_detected": [...]}

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
        return []

    batch = place_seeg(
        str(ct_path),
        seeds=seeds,
        library=pitch_strategy,
        sample_fn=sample_neg_log_max,
        progress_logger=_stderr,
    )

    out: list[dict[str, Any]] = []
    for placed in batch.trajectories:
        contacts_ras = [list(c) for c in placed.contacts_ras]
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
    return out


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
    args = parser.parse_args(argv)

    trajs = read_seeds_tsv(args.trajectories_tsv)
    _stderr(f"[contacts] {len(trajs)} trajectories from {args.trajectories_tsv}")
    groups = place_contacts(args.ct_path, trajs, pitch_strategy=args.library)
    n = write_contacts_tsv(args.out, groups)
    _stderr(f"[contacts] wrote {args.out} ({n} contacts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
