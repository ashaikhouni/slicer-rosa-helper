"""rosa-agent match-ros — name detector emissions from a .ros file's plan.

Thin wrapper over the generic ``match-trajectories`` core: it extracts the
planned trajectories from a ``.ros`` (no image read — only the planned segments
plus an LPS->RAS flip, so it works even when the ROSA folder is missing its
image volumes) and then runs the same detect + line-match operation.

Use case: same patient, two coordinate frames. Given a ROSA case folder (or a
bare .ros file) and a CT in some *other* RAS frame (a post-registered CT aligned
to a different MRI atlas, or a routine clinical CT in scanner coordinates),
match each detector-emitted trajectory to a ROS-planned trajectory and propagate
the surgeon's naming. No reference volume from the .ros side is needed.

For a plain named-trajectory file (no .ros), use ``rosa-agent
match-trajectories`` directly — this command just adapts a ``.ros`` into the
same plan bundle.

Usage::

    rosa-agent match-ros \\
        --rosa-folder s57_rosa/ \\
        --ct path/to/post_registered_t24.nii.gz \\
        --output match_qc/

    # or pass a bare .ros file directly
    rosa-agent match-ros --ros-file plan.ros --ct ct.nii.gz --output qc/

Output directory layout is identical to ``rosa-agent match-trajectories``
(``trajectories.tsv`` renamed to the ROS plan names where matched, ``match.tsv``,
``cross_volume_match.json``, ...).
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
        prog="rosa-agent match-ros",
        description=(
            "Name detector emissions on a CT using planned trajectories "
            "from a .ros file in a different coordinate frame."
        ),
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--rosa-folder",
                     help="ROSA case folder (auto-locates the .ros file inside)")
    src.add_argument("--ros-file",
                     help="path to a .ros file directly (skip folder search)")
    add_match_args(parser)
    args = parser.parse_args(argv)

    log = (lambda _msg: None) if args.quiet else _stderr

    ct_path = Path(args.ct)
    if not ct_path.exists():
        _stderr(f"error: CT not found: {ct_path}")
        return 2

    # ------------------------------------------------------------------
    # Parse the .ros plan (no image read). Shares
    # `cli.rosa_agent.io.ros_input` with `rosa-to-nifti`.
    # ------------------------------------------------------------------
    try:
        from rosa_agent.io.ros_input import load_ros_planned_trajectories
    except ImportError as exc:
        _stderr(f"error: rosa_core / rosa_agent unavailable ({exc})")
        return 2
    try:
        plan = load_ros_planned_trajectories(
            folder=args.rosa_folder, file=args.ros_file,
        )
    except (FileNotFoundError, ValueError) as exc:
        _stderr(f"error: {exc}")
        return 2

    return run_trajectory_match(
        plan.trajectories, ct_path, Path(args.output),
        plan_label=plan.ros_file.name,
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
