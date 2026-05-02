"""rosa-agent load — discover the .ros file in a case folder and emit a manifest.

Wraps :func:`rosa_core.case_loader.find_ros_file` +
:func:`rosa_core.ros_parser.parse_ros_file`. Pure Python — no Slicer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rosa_core.case_loader import find_ros_file
from rosa_core.ros_parser import parse_ros_file
from rosa_core.transforms import lps_to_ras_point

from ..io.manifest import write_manifest


def build_manifest_from_ros(case_dir: str | Path) -> dict[str, Any]:
    """Parse a ROSA case folder into the rosa_agent manifest schema."""
    case_path = Path(case_dir).expanduser().resolve()
    ros_path = find_ros_file(str(case_path))
    parsed = parse_ros_file(ros_path)

    displays = []
    for d in parsed.get("displays", []):
        displays.append({
            "index": int(d.get("index", 0)),
            "volume": d.get("volume"),
            "volume_path": d.get("volume_path"),
            "imagery_name": d.get("imagery_name"),
            "serie_uid": d.get("serie_uid"),
            "imagery_3dref": d.get("imagery_3dref"),
            "matrix": d.get("matrix"),
        })

    trajectories = []
    for traj in parsed.get("trajectories", []):
        start_lps = list(traj["start"])
        end_lps = list(traj["end"])
        trajectories.append({
            "name": traj["name"],
            "start_lps": [float(v) for v in start_lps],
            "end_lps": [float(v) for v in end_lps],
            "start_ras": [float(v) for v in lps_to_ras_point(start_lps)],
            "end_ras": [float(v) for v in lps_to_ras_point(end_lps)],
        })

    notes = []
    if parsed.get("trajectory_rejects"):
        notes.append(
            f"{len(parsed['trajectory_rejects'])} trajectory token(s) rejected during parse"
        )

    return {
        "case_dir": str(case_path),
        "ros_file": str(ros_path),
        "displays": displays,
        "reference_volume": (displays[0]["volume"] if displays else None),
        "planned_trajectories": trajectories,
        "token_histogram": parsed.get("token_histogram", {}),
        "notes": notes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rosa-agent load",
        description="Discover a .ros file in a case folder and emit a JSON manifest.",
    )
    parser.add_argument("case_dir", help="Path to the ROSA case folder")
    parser.add_argument("--out", "-o", default="-", help="Manifest output path (default: stdout)")
    args = parser.parse_args(argv)

    manifest = build_manifest_from_ros(args.case_dir)

    if args.out == "-" or args.out == "":
        json.dump({"manifest_version": "1.0", **manifest}, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        write_manifest(args.out, manifest)
        n = len(manifest.get("planned_trajectories") or [])
        print(
            f"[load] wrote {args.out} (displays={len(manifest['displays'])}, "
            f"planned_trajectories={n})",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
