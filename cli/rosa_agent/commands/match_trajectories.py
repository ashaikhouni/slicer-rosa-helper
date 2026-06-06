"""rosa-agent match-trajectories — name detector emissions on a CT from a
generic named-trajectory file in a (possibly different) coordinate frame.

Use case: you have a CT and a list of NAMED planned trajectories — but the
names live in a *different* RAS frame than the CT (a plan exported from another
station, a registration to a different atlas, a routine scanner-frame CT), or
you simply never had a usable ``.ros``. Run detection on the CT, then propagate
the planned names onto the detected electrodes by matching the two line bundles
geometrically. No image registration, no reference volume.

This is the generic core; ``rosa-agent match-ros`` is the same operation with a
``.ros`` parsed into the plan up front.

Plan file: a TSV/CSV the seed reader understands — a header with
``name`` + ``start_x/y/z`` + ``end_x/y/z`` (or the ``label, x, y, z``
endpoint-pair form). Coordinates are RAS, in ANY frame.

Usage::

    rosa-agent match-trajectories \\
        --plan planned_trajectories.tsv \\
        --ct  postop_ct.nii.gz \\
        --output match_qc/

Output directory (same shape as ``rosa-agent place``):

    output/
      manifest.json              ← provenance + recovered transform + match table
      trajectories.tsv           ← detector emissions, RENAMED to plan names where matched
      contacts.tsv
      figures/                   ← per-trajectory PNGs (filenames use plan names)
      diagnostics/cmp.tsv
      match.tsv                  ← per-plan match (plan_name, det_name, angle, perp_mm)
      cross_volume_match.json    ← recovered det->plan transform + pairs

Unmatched detector emissions keep their ``CAND-NNN`` names; unmatched plans
appear in ``match.tsv`` with an empty det_name.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..match_core import add_match_args, run_trajectory_match


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rosa-agent match-trajectories",
        description=(
            "Name detector emissions on a CT using a generic named-trajectory "
            "file in a different coordinate frame (line-geometry match, no "
            "image registration)."
        ),
    )
    parser.add_argument(
        "--plan", required=True,
        help="named-trajectory TSV/CSV: header with name + start_x/y/z + "
             "end_x/y/z (or the label,x,y,z endpoint-pair form). RAS, any frame.",
    )
    add_match_args(parser)
    args = parser.parse_args(argv)

    log = (lambda _msg: None) if args.quiet else _stderr

    ct_path = Path(args.ct)
    if not ct_path.exists():
        _stderr(f"error: CT not found: {ct_path}")
        return 2

    plan_path = Path(args.plan)
    if not plan_path.exists():
        _stderr(f"error: plan file not found: {plan_path}")
        return 2

    from rosa_agent.io.trajectory_io import read_seeds_tsv
    try:
        rows = read_seeds_tsv(plan_path)
    except (ValueError, OSError) as exc:
        _stderr(f"error: could not read plan {plan_path}: {exc}")
        return 2
    if not rows:
        _stderr(f"error: plan {plan_path} contained no trajectories")
        return 2

    plan_trajs = [
        {"name": r["name"], "start": list(r["start_ras"]), "end": list(r["end_ras"])}
        for r in rows
    ]

    return run_trajectory_match(
        plan_trajs, ct_path, Path(args.output),
        plan_label=plan_path.name,
        library=args.library,
        sampler=args.sampler,
        band_floor=args.band_floor,
        subject_id=args.subject_id,
        no_figures=args.no_figures,
        angle_tol_deg=args.angle_tol_deg,
        ransac_perp_mm=args.ransac_perp_mm,
        match_perp_mm=args.match_perp_mm,
        ransac_iter=args.ransac_iter,
        seed=args.seed,
        log=log,
    )


if __name__ == "__main__":
    raise SystemExit(main())
