"""rosa-agent deidentify-ros — strip PHI from a .ros (and optionally dump
planned trajectories to CSV).

De-identifies patient name / birthday / acquisition date and pseudonymises
DICOM Series UIDs to consistent tokens (so display<->series linkage survives).
Writes a clean ``.ros`` + a PRIVATE re-link keymap. Optionally extracts the
planned trajectories to a (PHI-free) CSV.

    rosa-agent deidentify-ros <ros_file> [--out clean.ros] [--subject-id S01]
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
    parser.add_argument("path", help="A .ros file, OR a case folder (with --out-dir) to "
                                     "de-identify whole: clean .ros + renamed image dirs.")
    parser.add_argument("--out", default=None,
                        help="[file mode] Clean .ros output (default <dir>/<subject-id>.ros)")
    parser.add_argument("--out-dir", default=None,
                        help="[folder mode] Fresh shareable case dir to write (required when "
                             "PATH is a folder). Never mutates the original.")
    parser.add_argument("--subject-id", default=None,
                        help="Replaces the patient name + names the output "
                             "(default: the parent / case folder name)")
    parser.add_argument("--keymap", default=None,
                        help="Where to write the PRIVATE real->token keymap JSON. "
                             "PHI — keep, never share.")
    parser.add_argument("--no-keymap", action="store_true",
                        help="Do not write the re-link keymap.")
    parser.add_argument("--trajectories-csv", nargs="?", default="__none__", const="__auto__",
                        help="[file mode] Also write planned trajectories to CSV. Bare flag = "
                             "<dir>/<ros-stem>_trajectories.csv; or give a path. "
                             "(Folder mode always writes the CSV into --out-dir.)")
    parser.add_argument("--blank-uids", action="store_true",
                        help="Blank UIDs to *** instead of pseudonymising (loses series linkage).")
    args = parser.parse_args(argv)

    from rosa_core.ros_deidentify import (
        deidentify_ros_file, deidentify_ros_folder, write_trajectories_csv,
    )

    path = Path(args.path).expanduser()
    keymap_path = False if args.no_keymap else (args.keymap if args.keymap else None)

    # ---- Folder mode --------------------------------------------------
    if path.is_dir():
        if not args.out_dir:
            _stderr("error: PATH is a folder — give --out-dir for the clean case dir")
            return 2
        out_dir, mapping, summary = deidentify_ros_folder(
            path, args.out_dir, subject_id=args.subject_id,
            keymap_path=keymap_path, pseudonymize_uids=not args.blank_uids,
            trajectories_csv=True, log=_stderr,
        )
        _stderr(f"[deidentify-ros] clean case folder -> {out_dir}")
        _stderr(f"[deidentify-ros] redacted {len(mapping['names'])} name(s), "
                f"{len(mapping['dates'])} date(s); pseudonymised {len(mapping['uids'])} UID(s)")
        if keymap_path is not False:
            _stderr("[deidentify-ros] keymap written OUTSIDE the clean dir (PHI — keep private)")
        return 0

    # ---- Single-file mode ---------------------------------------------
    if not path.is_file():
        _stderr(f"error: not found: {path}")
        return 2
    out_path, mapping = deidentify_ros_file(
        path, out_path=args.out, subject_id=args.subject_id,
        keymap_path=keymap_path, pseudonymize_uids=not args.blank_uids,
    )
    _stderr(f"[deidentify-ros] wrote {out_path}")
    _stderr(f"[deidentify-ros] redacted {len(mapping['names'])} name(s), "
            f"{len(mapping['dates'])} date(s); "
            f"{'pseudonymised' if not args.blank_uids else 'blanked'} {len(mapping['uids'])} UID(s)")
    if keymap_path is not False:
        _stderr("[deidentify-ros] keymap written (PHI — keep private, do not share)")
    if args.trajectories_csv != "__none__":
        csv_arg = None if args.trajectories_csv == "__auto__" else args.trajectories_csv
        csv_path = write_trajectories_csv(path, out_csv=csv_arg)
        _stderr(f"[deidentify-ros] wrote planned trajectories: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
