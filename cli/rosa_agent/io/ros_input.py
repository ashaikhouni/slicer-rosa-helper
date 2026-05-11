"""Shared ``.ros``-input helpers for the CLI.

Two CLI commands accept ``.ros`` plans as input:

* ``rosa-agent rosa-to-nifti`` — bakes display volumes + plan into NIfTI
* ``rosa-agent match-ros``     — names detector emissions on a different-frame CT

Both need to: locate the ``.ros`` file (from a folder spec or a direct
file path), parse it, and convert planned trajectories from the .ros's
LPS coordinates to RAS so the rest of the CLI can stay in the project's
RAS convention. Centralized here so any new ``.ros``-aware command only
has to call :func:`load_ros_planned_trajectories`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RosPlan:
    """Result of :func:`load_ros_planned_trajectories`.

    ``trajectories`` is a list of ``{name, start, end}`` dicts where
    ``start`` / ``end`` are ``[x, y, z]`` floats in RAS — the same shape
    ``rosa_core.cross_volume_match.cross_volume_match`` and
    ``cli/rosa_agent/io/trajectory_io.read_seeds_tsv`` consume.

    ``raw`` is the full :func:`rosa_core.ros_parser.parse_ros_file` dict
    (display list, transforms, original LPS coords, …) for callers that
    need more than just the named planned segments.
    """

    ros_file: Path
    trajectories: list[dict]
    raw: dict


def resolve_ros_file(folder: str | None, file: str | None) -> Path:
    """Resolve a ``.ros`` path from the CLI's ``(--rosa-folder, --ros-file)``
    pair.

    Exactly one of ``folder`` / ``file`` must be set. Folder mode delegates
    to :func:`rosa_core.case_loader.find_ros_file` so the same .ros-search
    logic the Slicer side uses applies here. Raises ``FileNotFoundError``
    if the resolved path doesn't exist on disk.
    """
    if (folder is None) == (file is None):
        raise ValueError(
            "exactly one of --rosa-folder / --ros-file must be supplied"
        )
    if file is not None:
        ros_path = Path(file)
        if not ros_path.exists():
            raise FileNotFoundError(f"--ros-file not found: {ros_path}")
        return ros_path
    folder_path = Path(folder)  # type: ignore[arg-type]
    if not folder_path.is_dir():
        raise FileNotFoundError(f"--rosa-folder is not a directory: {folder_path}")
    from rosa_core.case_loader import find_ros_file
    return Path(find_ros_file(str(folder_path)))


def load_ros_planned_trajectories(
    folder: str | None = None, file: str | None = None,
) -> RosPlan:
    """Locate, parse, and RAS-convert the planned trajectories from a ``.ros``.

    Pure-text path — no Analyze volume is read. Use this when you only
    need the surgeon's planned segments (e.g. ``match-ros``); use
    :func:`rosa_core.load_rosa_volume_as_sitk` instead when you also
    need an aligned image volume.

    Trajectories with malformed coordinate entries are silently dropped
    (consistent with how :func:`rosa_core.ros_parser.parse_ros_file`
    treats them).
    """
    from rosa_core.ros_parser import parse_ros_file
    from rosa_core.transforms import lps_to_ras_point

    ros_path = resolve_ros_file(folder, file)
    parsed = parse_ros_file(str(ros_path))
    out: list[dict] = []
    for t in (parsed.get("trajectories") or []):
        try:
            s_ras = [float(v) for v in lps_to_ras_point(list(t["start"]))]
            e_ras = [float(v) for v in lps_to_ras_point(list(t["end"]))]
        except (KeyError, TypeError, ValueError):
            continue
        out.append({
            "name": str(t.get("name") or ""),
            "start": s_ras,
            "end": e_ras,
        })
    return RosPlan(ros_file=ros_path, trajectories=out, raw=parsed)


__all__ = ["RosPlan", "resolve_ros_file", "load_ros_planned_trajectories"]
