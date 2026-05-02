"""TSV readers/writers for trajectories, contacts, and seeds.

Stable column names (the public contract documented in cli/README.md):

Trajectory TSV ("trajectories.tsv"):
    name              str    trajectory label (e.g. "L_AC")
    start_x/y/z       float  RAS mm — bolt-side / outer endpoint
    end_x/y/z         float  RAS mm — deep tip
    confidence        float  0..1 (when available, "" otherwise)
    confidence_label  str    "high" | "medium" | "low" | ""
    electrode_model   str    library model id (or "")
    bolt_source       str    "metal" | "synthesized" | "wire" | "none" | ""
    length_mm         float  end - start

Contacts TSV ("contacts.tsv"):
    trajectory        str    trajectory name
    label             str    "L_AC1", "L_AC2", ...
    contact_index     int    1-based slot index
    x/y/z             float  RAS mm
    peak_detected     int    1 if anchored on detected peak, 0 if model-nominal
    electrode_model   str    library model id

Seeds TSV (input to detect/contacts; same trajectory-style header).
The simple form (label, x, y, z) is also accepted as a shortcut for
contact-list inputs (label is the trajectory name; (x, y, z) is one
endpoint per row, two consecutive rows = start/end).

Coordinates are always RAS unless the file name encodes otherwise; the
loader does NOT silently flip frames.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Sequence


TRAJECTORY_COLUMNS: tuple[str, ...] = (
    "name",
    "start_x", "start_y", "start_z",
    "end_x", "end_y", "end_z",
    "confidence",
    "confidence_label",
    "electrode_model",
    "bolt_source",
    "length_mm",
)


CONTACT_COLUMNS: tuple[str, ...] = (
    "trajectory",
    "label",
    "contact_index",
    "x", "y", "z",
    "peak_detected",
    "electrode_model",
)


def _sniff_delimiter(sample: str) -> str:
    """Tab-first; commas only when there is no tab."""
    if "\t" in sample:
        return "\t"
    if "," in sample:
        return ","
    return "\t"


def read_tsv_rows(path: str | Path) -> list[dict[str, str]]:
    """Read a (TSV or CSV) file as a list of dicts."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if not text.strip():
        return []
    delim = _sniff_delimiter(text.splitlines()[0])
    reader = csv.DictReader(text.splitlines(), delimiter=delim)
    return [dict(row) for row in reader]


def write_tsv_rows(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    """Write rows to a TSV file (creates parent dirs)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


# ---------------------------------------------------------------------
# Seed parsing — accepts both forms.
# ---------------------------------------------------------------------


def _coerce_float(value: Any) -> float:
    if value in (None, ""):
        raise ValueError("missing numeric value")
    return float(value)


def read_seeds_tsv(path: str | Path) -> list[dict[str, Any]]:
    """Read trajectory seeds from TSV/CSV.

    Two accepted forms:

    1. Trajectory rows — a header containing ``start_x``/``end_x`` (and
       y/z). Returns one dict per row with keys ``name``, ``start_ras``
       (3-tuple), ``end_ras`` (3-tuple).

    2. Endpoint pairs — a header with ``label, x, y, z`` (or just
       ``x, y, z`` plus a name column). Two consecutive rows that share
       a ``label`` are treated as start + end of one trajectory.

    Raises ``ValueError`` if the file matches neither form.
    """
    rows = read_tsv_rows(path)
    if not rows:
        return []
    cols = {c.lower() for c in rows[0].keys()}

    if {"start_x", "end_x"}.issubset(cols):
        out: list[dict[str, Any]] = []
        for r in rows:
            r_lower = {k.lower(): v for k, v in r.items()}
            try:
                start = (
                    _coerce_float(r_lower["start_x"]),
                    _coerce_float(r_lower["start_y"]),
                    _coerce_float(r_lower["start_z"]),
                )
                end = (
                    _coerce_float(r_lower["end_x"]),
                    _coerce_float(r_lower["end_y"]),
                    _coerce_float(r_lower["end_z"]),
                )
            except (KeyError, ValueError) as exc:
                raise ValueError(f"malformed seed row: {r!r} ({exc})") from exc
            name = (
                r_lower.get("name")
                or r_lower.get("shank")
                or r_lower.get("trajectory")
                or f"S{len(out) + 1:02d}"
            )
            out.append({
                "name": str(name),
                "start_ras": start,
                "end_ras": end,
                "electrode_model": r_lower.get("electrode_model") or "",
            })
        return out

    if {"x", "y", "z"}.issubset(cols):
        # Endpoint-pair form: group two rows per label into one seed.
        name_key = "label" if "label" in cols else (
            "name" if "name" in cols else None
        )
        if name_key is None:
            raise ValueError(
                "endpoint TSV needs a 'label' or 'name' column"
            )
        grouped: dict[str, list[tuple[float, float, float]]] = {}
        order: list[str] = []
        for r in rows:
            r_lower = {k.lower(): v for k, v in r.items()}
            label = str(r_lower.get(name_key, "")).strip()
            if not label:
                continue
            try:
                pt = (
                    _coerce_float(r_lower["x"]),
                    _coerce_float(r_lower["y"]),
                    _coerce_float(r_lower["z"]),
                )
            except (KeyError, ValueError) as exc:
                raise ValueError(f"malformed endpoint row: {r!r} ({exc})") from exc
            if label not in grouped:
                grouped[label] = []
                order.append(label)
            grouped[label].append(pt)
        out = []
        for label in order:
            pts = grouped[label]
            if len(pts) < 2:
                raise ValueError(
                    f"endpoint TSV: trajectory {label!r} needs 2 points, got {len(pts)}"
                )
            out.append({
                "name": label,
                "start_ras": pts[0],
                "end_ras": pts[-1],
                "electrode_model": "",
            })
        return out

    raise ValueError(
        f"seed TSV {path}: header must contain start_x/end_x, "
        f"or label+x/y/z (got {sorted(cols)})"
    )


# ---------------------------------------------------------------------
# Trajectory writer — the public contract.
# ---------------------------------------------------------------------


def _fmt_float(value: Any, *, precision: int = 6) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return ""


def trajectory_to_row(traj: dict[str, Any], *, fallback_name: str = "") -> dict[str, str]:
    """Format one DetectedTrajectory as a row matching ``TRAJECTORY_COLUMNS``."""
    start = traj.get("start_ras") or [0.0, 0.0, 0.0]
    end = traj.get("end_ras") or [0.0, 0.0, 0.0]
    bolt = traj.get("bolt") or {}
    bolt_source = (
        traj.get("bolt_source")
        or (bolt.get("source") if isinstance(bolt, dict) else "")
        or ""
    )
    sx, sy, sz = float(start[0]), float(start[1]), float(start[2])
    ex, ey, ez = float(end[0]), float(end[1]), float(end[2])
    length = ((ex - sx) ** 2 + (ey - sy) ** 2 + (ez - sz) ** 2) ** 0.5
    return {
        "name": str(traj.get("name") or fallback_name),
        "start_x": _fmt_float(sx),
        "start_y": _fmt_float(sy),
        "start_z": _fmt_float(sz),
        "end_x": _fmt_float(ex),
        "end_y": _fmt_float(ey),
        "end_z": _fmt_float(ez),
        "confidence": _fmt_float(traj.get("confidence"), precision=4),
        "confidence_label": str(traj.get("confidence_label") or ""),
        "electrode_model": str(traj.get("electrode_model") or ""),
        "bolt_source": str(bolt_source),
        "length_mm": _fmt_float(length, precision=3),
    }


def write_trajectories_tsv(path: str | Path, trajectories: Iterable[dict[str, Any]]) -> int:
    """Write trajectories with stable columns. Returns the count written."""
    rows: list[dict[str, str]] = []
    for idx, traj in enumerate(trajectories, start=1):
        rows.append(trajectory_to_row(traj, fallback_name=f"T{idx:02d}"))
    write_tsv_rows(path, rows, TRAJECTORY_COLUMNS)
    return len(rows)


# ---------------------------------------------------------------------
# Contacts writer.
# ---------------------------------------------------------------------


def write_contacts_tsv(
    path: str | Path,
    contact_groups: Iterable[dict[str, Any]],
) -> int:
    """Write contact rows to TSV.

    Each item in ``contact_groups`` is a dict like::

        {
            "trajectory": "L_AC",
            "electrode_model": "DIXI-15AM",
            "positions_ras": [[x, y, z], ...],
            "peak_detected": [True, False, ...],  # parallel to positions
        }

    Labels are emitted as ``f"{trajectory}{contact_index}"`` with 1-based indexing.
    """
    rows: list[dict[str, str]] = []
    for group in contact_groups:
        traj_name = str(group.get("trajectory") or "")
        model_id = str(group.get("electrode_model") or "")
        positions = list(group.get("positions_ras") or [])
        flags = list(group.get("peak_detected") or [True] * len(positions))
        for idx, pt in enumerate(positions, start=1):
            detected = bool(flags[idx - 1]) if idx - 1 < len(flags) else True
            rows.append({
                "trajectory": traj_name,
                "label": f"{traj_name}{idx}",
                "contact_index": str(idx),
                "x": _fmt_float(pt[0]),
                "y": _fmt_float(pt[1]),
                "z": _fmt_float(pt[2]),
                "peak_detected": "1" if detected else "0",
                "electrode_model": model_id,
            })
    write_tsv_rows(path, rows, CONTACT_COLUMNS)
    return len(rows)
