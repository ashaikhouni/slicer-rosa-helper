"""Export contact / trajectory positions in Curry / Neuroscan
electrode-position file format.

Format
------
Plain ASCII, one electrode per line: ``label x y z``, whitespace
separated. Coordinates in **millimetres, LPS** (Curry's native
coordinate system). This is the simple per-line layout Curry's
electrode-import dialogs expect (also broadly compatible with
EEGLAB ``readlocs`` and other tools that read `.dat` / `.asc`
electrode coordinate files).

Example::

    RAH1 -13.329889 -49.023007 -9.279582
    RAH2 -15.010925 -48.376096 -6.278645
    LAH1 12.443012  -47.881234 -8.991234
    ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from .transforms import lps_to_ras_point


# ---------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------

# A POM point is just (label, x, y, z) where (x, y, z) are RAS in mm.
# `write_curry_pom` flips RAS → LPS at write time.
PomPoint = tuple[str, float, float, float]


# ---------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------

def _format_xyz(x: float, y: float, z: float) -> str:
    """Format a single XYZ row with 6-decimal precision (sub-micron)."""
    return f"{float(x):.6f} {float(y):.6f} {float(z):.6f}"


def _ras_to_lps(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Curry's default coordinate system is LPS — flip x and y signs.

    `lps_to_ras_point` is symmetric (a self-inverse `[-x, -y, z]`
    flip) so we use it in either direction.
    """
    flipped = lps_to_ras_point([float(x), float(y), float(z)])
    return float(flipped[0]), float(flipped[1]), float(flipped[2])


def write_curry_pom(
    path: str | Path,
    points: Sequence[PomPoint],
    *,
    coords_in: str = "ras",
) -> int:
    """Write a Curry electrode-positions file: one ``label x y z`` per
    line, mm, LPS coordinates.

    Parameters
    ----------
    path
        Output file path (typically ``.pom`` or ``.dat``).
    points
        Iterable of ``(label, x, y, z)`` tuples. Coordinates in mm.
    coords_in
        ``"ras"`` (default) — input coords are RAS; flip to LPS at write.
        ``"lps"`` — already LPS, no flip.

    Returns the number of points written.
    """
    rows: list[PomPoint] = list(points)
    if not rows:
        raise ValueError("write_curry_pom: no points to write")
    coords_in_lower = str(coords_in).lower()
    if coords_in_lower not in ("ras", "lps"):
        raise ValueError(f"coords_in must be 'ras' or 'lps', got {coords_in!r}")

    out_path = Path(path)
    lines: list[str] = []
    for label, x, y, z in rows:
        if coords_in_lower == "ras":
            x, y, z = _ras_to_lps(x, y, z)
        # Single-token label (replace any internal whitespace with
        # underscores) so the reader splits each row into exactly
        # 4 fields: name + 3 coordinates.
        clean = str(label).strip().replace(" ", "_") or "_"
        lines.append(f"{clean} {_format_xyz(x, y, z)}")
    lines.append("")  # trailing newline
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return len(rows)


# ---------------------------------------------------------------------
# Helpers — turn contacts / trajectories into POM-point lists
# ---------------------------------------------------------------------

def contacts_to_pom_points(contacts: Iterable[dict]) -> list[PomPoint]:
    """Convert a list of contact dicts to POM points.

    Each contact is expected to carry:
      - ``trajectory`` : str — trajectory name
      - ``index`` : int — 1-based contact index
      - either ``position_ras`` (RAS mm) or ``position_lps`` (LPS mm)
    Label format: ``<TrajName><index>`` (e.g. ``RAH1`` … ``RAH10``).
    """
    out: list[PomPoint] = []
    for c in contacts:
        traj = str(c.get("trajectory") or "").strip() or "?"
        idx = int(c.get("index") or 0)
        if "position_ras" in c and c["position_ras"] is not None:
            ras = c["position_ras"]
        elif "position_lps" in c and c["position_lps"] is not None:
            ras = lps_to_ras_point(c["position_lps"])
        else:
            continue
        if len(ras) < 3:
            continue
        label = f"{traj}{idx}"
        out.append((label, float(ras[0]), float(ras[1]), float(ras[2])))
    return out


def trajectory_endpoints_to_pom_points(
    trajectories: Iterable[dict],
) -> list[PomPoint]:
    """Convert each trajectory to two POM points: entry + tip.

    Trajectories are expected to carry ``start_ras`` / ``end_ras``
    (preferred) or LPS-flavoured ``start`` / ``end`` keys.
    Label format: ``<TrajName>_E`` (entry) and ``<TrajName>_T`` (tip).
    """
    out: list[PomPoint] = []
    for t in trajectories:
        name = str(t.get("name") or "").strip() or "?"
        if "start_ras" in t and t["start_ras"] is not None:
            entry = t["start_ras"]
        elif "start" in t and t["start"] is not None:
            entry = lps_to_ras_point(t["start"])
        else:
            continue
        if "end_ras" in t and t["end_ras"] is not None:
            tip = t["end_ras"]
        elif "end" in t and t["end"] is not None:
            tip = lps_to_ras_point(t["end"])
        else:
            continue
        out.append((f"{name}_E", float(entry[0]), float(entry[1]), float(entry[2])))
        out.append((f"{name}_T", float(tip[0]), float(tip[1]), float(tip[2])))
    return out
