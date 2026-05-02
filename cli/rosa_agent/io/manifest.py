"""JSON case-manifest schema for rosa_agent.

A manifest captures everything ``rosa-agent load`` learned from a ROSA
case folder so downstream commands (detect, contacts, label) can resume
without re-parsing the .ros file.

Schema (top-level dict):

    case_dir          str    absolute path to the input ROSA folder
    ros_file          str    discovered .ros file path
    displays          list   per-volume metadata (volume name, serie_uid,
                             imagery_name, IMAGERY_3DREF, ros->display matrix)
    reference_volume  str    chosen root display volume
    planned_trajectories list of dicts; coords are LPS in the source
                             ROSA frame (matches ros_parser output) plus
                             the same points converted to RAS for
                             convenience.
    notes             list[str]   diagnostic strings

Trajectory dict keys:
    name              str
    start_lps         [x, y, z]
    end_lps           [x, y, z]
    start_ras         [x, y, z]   (lps_to_ras_point of start_lps)
    end_ras           [x, y, z]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MANIFEST_VERSION = "1.0"


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"manifest_version": MANIFEST_VERSION, **manifest}
    out.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def read_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
