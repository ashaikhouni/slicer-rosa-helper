"""rosa-agent deidentify-ros — strip PHI from a .ros (and optionally dump
planned trajectories to CSV).

De-identifies patient name / birthday / acquisition date and pseudonymises
DICOM Series UIDs to consistent tokens (so display<->series linkage survives).
Writes a clean ``.ros`` + a PRIVATE re-link keymap. Optionally extracts the
planned trajectories to a (PHI-free) CSV.

    rosa-agent deidentify-ros <ros_file> [--out clean.ros] [--subject-id JK]
        [--keymap key.json | --no-keymap] [--trajectories-csv traj.csv]
        [--blank-uids]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rosa-agent deidentify-ros",
        description="De-identify a ROSA .ros file (PHI out, UID linkage preserved).",
    )
    parser.add_argument("ros_file", help="Path to the identifying .ros file")
    parser.add_argument("--out", default=None,
                        help="Clean .ros output (default <dir>/<subject-id>.ros)")
    parser.add_argument("--subject-id", default=None,
                        help="Replaces the patient name + names the output "
                             "(default: the .ros file's parent folder name)")
    parser.add_argument("--keymap", default=None,
                        help="Where to write the PRIVATE real->token keymap JSON "
                             "(default <out>/<subject-id>_deid_keymap.json). PHI — keep, never share.")
    parser.add_argument("--no-keymap", action="store_true",
                        help="Do not write the re-link keymap.")
    parser.add_argument("--trajectories-csv", nargs="?", default="__none__", const="__auto__",
                        help="Also write planned trajectories to CSV. Bare flag = "
                             "<dir>/<ros-stem>_trajectories.csv; or give a path.")
    parser.add_argument("--blank-uids", action="store_true",
                        help="Blank UIDs to *** instead of pseudonymising (loses series linkage).")
    args = parser.parse_args(argv)

    from rosa_core.ros_deidentify import deidentify_ros_file, write_trajectories_csv

    ros_file = Path(args.ros_file).expanduser()
    if not ros_file.is_file():
        _stderr(f"error: .ros not found: {ros_file}")
        return 2

    keymap_path = False if args.no_keymap else (args.keymap if args.keymap else None)
    out_path, mapping = deidentify_ros_file(
        ros_file, out_path=args.out, subject_id=args.subject_id,
        keymap_path=keymap_path, pseudonymize_uids=not args.blank_uids,
    )
    n_names = len(mapping["names"]); n_uids = len(mapping["uids"]); n_dates = len(mapping["dates"])
    _stderr(f"[deidentify-ros] wrote {out_path}")
    _stderr(f"[deidentify-ros] redacted {n_names} name(s), {n_dates} date(s); "
            f"{'pseudonymised' if not args.blank_uids else 'blanked'} {n_uids} UID(s)")
    if keymap_path is not False:
        _stderr("[deidentify-ros] keymap written (PHI — keep private, do not share)")

    if args.trajectories_csv != "__none__":
        csv_arg = None if args.trajectories_csv == "__auto__" else args.trajectories_csv
        csv_path = write_trajectories_csv(ros_file, out_csv=csv_arg)
        _stderr(f"[deidentify-ros] wrote planned trajectories: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
