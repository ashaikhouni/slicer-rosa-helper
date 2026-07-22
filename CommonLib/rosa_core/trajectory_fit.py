"""Make a trajectory line carry its contacts.

Detection/placement produce two decoupled things: a trajectory *line* (the
detected/planned entry→tip seed) and a set of *contacts* (snapped onto the real
metal, typically 1–2 mm off the seed axis). The viewer draws the line and the
contacts separately, so the line reads as "slightly off" its own contacts.

The fix is a single invariant — **the trajectory line is the best-fit (PCA) line
through its contacts** — enforced everywhere contacts are (re)written: the CLI
``pipeline``, the app's ``contacts`` step, and the editor payload. This is the
one algorithm; the callers stay thin (CLI/Slicer parity: geometry lives here).

``fit_line_through_points`` is the primitive; ``refit_trajectories_inplace``
(in-memory dict rows) and ``refit_trajectories_file`` (persisted TSVs) are the
two call shapes.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "fit_line_through_points",
    "refit_trajectories_inplace",
    "refit_trajectories_file",
]


def _principal_axis(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Centroid + unit first principal axis (SVD) of a point cloud."""
    centroid = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    axis = vt[0]
    n = float(np.linalg.norm(axis))
    return centroid, (axis / n if n > 1e-9 else None)


def fit_line_through_points(
    points: Iterable[Sequence[float]],
    ref_start: Sequence[float],
    ref_end: Sequence[float],
    *,
    inlier_mm: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Best-fit line through ``points``, returned as ``(start, end)`` endpoints.

    Intended for **straight** SEEG shanks: the direction is the first principal
    axis (SVD) of the contact cloud, so for a straight electrode the contacts
    lie on the returned line to within their (small) fit residual — they are
    "carried". (A curved/PaCER shank is a chord here; mid-shaft contacts bow off
    it by the arc sagitta — callers should skip the refit for curved models.)

    The fit is made **outlier-resistant**: a single diverged / past-tip snapped
    contact would otherwise tilt the least-squares axis and, via the extent
    below, drag an endpoint out to itself — and a plain LS first pass can't catch
    it (the axis leans toward the outlier, masking its own residual). So we gate
    on perpendicular distance to the **reference line** (the detected/planned
    seed, which is ~parallel to truth — a good direction prior): contacts beyond
    ``inlier_mm`` from it are dropped, then we PCA-fit the survivors. Colinear
    good contacts survive even a mildly angled seed (any subset of colinear
    points fits the same line); a grossly off contact does not. A degenerate
    reference falls back to gating on the first-pass PCA axis.

    The endpoints are oriented and extended so the line:

      * points entry→tip the same way as ``ref_start``→``ref_end`` (PCA sign is
        arbitrary; we pick the sign that agrees with the reference — or, for a
        degenerate reference, a deterministic sign), and
      * spans **both** the reference extent (bolt/entry → planned tip) **and**
        the **inlier** contact extent — ``start``/``end`` reach whichever is
        furthest on each side, so the shaft still meets the bolt yet never stops
        short of a (good) contact.

    Fewer than 2 points, a degenerate cloud, or a degenerate axis → the
    reference endpoints are returned unchanged (nothing to fit).
    """
    pts = np.asarray(list(points), dtype=float)
    rs = np.asarray(ref_start, dtype=float)
    re = np.asarray(ref_end, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 3:
        return rs, re

    # Outlier gate: perpendicular distance to the reference line (base point +
    # unit direction). A degenerate reference has no direction → gate on the
    # first-pass PCA axis instead.
    ref_dir = re - rs
    ref_len = float(np.linalg.norm(ref_dir))
    if ref_len > 1e-9:
        base, u = rs, ref_dir / ref_len
    else:
        base, u = _principal_axis(pts)
        if u is None:
            return rs, re
    rel = pts - base
    resid = np.linalg.norm(rel - np.outer(rel @ u, u), axis=1)
    fit_pts = pts[resid <= float(inlier_mm)]
    if fit_pts.shape[0] < 2:
        fit_pts = pts                               # too aggressive a gate — keep all

    centroid, axis = _principal_axis(fit_pts)
    if axis is None:
        return rs, re

    # Orient entry→tip to match the reference; for a degenerate reference (no
    # entry/tip cue) pick a deterministic sign so repeated runs agree.
    if ref_len > 1e-9:
        if float(np.dot(axis, ref_dir)) < 0.0:
            axis = -axis
    elif axis[int(np.argmax(np.abs(axis)))] < 0.0:
        axis = -axis

    # Project onto the axis (scalar coordinate through the centroid) and take the
    # union of the reference and (inlier) contact extents.
    t_pts = (fit_pts - centroid) @ axis
    t_start = float(np.dot(rs - centroid, axis))
    t_end = float(np.dot(re - centroid, axis))
    lo = min(t_start, t_end, float(t_pts.min()))
    hi = max(t_start, t_end, float(t_pts.max()))
    start = centroid + lo * axis
    end = centroid + hi * axis
    return start, end


def _read_delimited(path: Path) -> tuple[list[str], list[dict]]:
    """Read a TSV/CSV, skipping ``#`` comment lines and sniffing tab-vs-comma
    from the header — so refit accepts the same file variants (comma-delimited
    seeds, a leading ``# reference_frame:`` provenance comment) the rest of the
    pipeline's IO does. Returns ``(fieldnames, rows)``."""
    lines = [ln for ln in path.read_text().splitlines() if not ln.lstrip().startswith("#")]
    if not lines:
        return [], []
    delim = "\t" if "\t" in lines[0] else ","
    reader = csv.DictReader(lines, delimiter=delim)
    return list(reader.fieldnames or []), list(reader)


def _points_by_trajectory(contacts: Iterable[dict]) -> dict[str, list[list[float]]]:
    """Group contact rows (``trajectory``, ``x``, ``y``, ``z``) by shank name."""
    pts: dict[str, list[list[float]]] = defaultdict(list)
    for c in contacts:
        name = str(c.get("trajectory") or "")
        try:
            pts[name].append([float(c["x"]), float(c["y"]), float(c["z"])])
        except (KeyError, TypeError, ValueError):
            continue
    return pts


def refit_trajectories_inplace(
    trajectories: list[dict[str, Any]],
    contact_groups: Iterable[dict[str, Any]],
) -> int:
    """Refit each trajectory dict's ``start_ras``/``end_ras`` to its contacts.

    ``trajectories`` rows carry ``name``, ``start_ras``, ``end_ras`` (as the
    detect/guided output does); ``contact_groups`` carry ``trajectory`` (name)
    and ``positions_ras`` (list of RAS points). Mutates ``trajectories`` and
    returns the number of shanks updated. Frame-agnostic — points and endpoints
    just have to share a frame — so call it *before* any output-frame transform
    and the transform will carry the refit line. Shanks with <2 contacts are
    left untouched.
    """
    pts_by: dict[str, list[list[float]]] = {}
    for g in contact_groups:
        name = str(g.get("trajectory") or "")
        pts_by.setdefault(name, []).extend([list(p) for p in (g.get("positions_ras") or [])])

    n = 0
    for tr in trajectories:
        pts = pts_by.get(str(tr.get("name") or ""), [])
        if len(pts) < 2:
            continue
        start, end = fit_line_through_points(pts, tr["start_ras"], tr["end_ras"])
        tr["start_ras"] = [float(start[0]), float(start[1]), float(start[2])]
        tr["end_ras"] = [float(end[0]), float(end[1]), float(end[2])]
        n += 1
    return n


def refit_trajectories_file(
    traj_tsv: str | Path,
    contacts_tsv: str | Path,
    out: str | Path | None = None,
) -> int:
    """Rewrite ``traj_tsv`` so each line is the PCA fit through its contacts.

    Reads the ``start_x/y/z`` + ``end_x/y/z`` line and every other column of
    ``traj_tsv``, groups ``contacts_tsv`` by ``trajectory``, refits each shank
    with ≥2 contacts (recomputing ``length_mm`` if present), and writes the
    result to ``out`` (default: in place). All non-geometry columns are
    preserved verbatim; shanks with <2 contacts are untouched. Idempotent — a
    line already fitted to its contacts is returned unchanged. Returns the
    number of shanks updated. A traj TSV without ``start_x``/``end_x`` columns
    (a different schema) is left alone (returns 0)."""
    traj_tsv = Path(traj_tsv)
    contacts_tsv = Path(contacts_tsv)
    out = Path(out) if out is not None else traj_tsv

    cols, rows = _read_delimited(traj_tsv)

    geo = ("start_x", "start_y", "start_z", "end_x", "end_y", "end_z")
    if not all(c in cols for c in geo):
        return 0                                    # unfamiliar schema — don't touch it

    pts_by: dict[str, list[list[float]]] = {}
    if contacts_tsv.is_file():
        _, crows = _read_delimited(contacts_tsv)
        pts_by = _points_by_trajectory(crows)

    n = 0
    for row in rows:
        pts = pts_by.get(str(row.get("name") or ""), [])
        if len(pts) < 2:
            continue
        try:
            rs = [float(row["start_x"]), float(row["start_y"]), float(row["start_z"])]
            re = [float(row["end_x"]), float(row["end_y"]), float(row["end_z"])]
        except (TypeError, ValueError):
            continue
        start, end = fit_line_through_points(pts, rs, re)
        row["start_x"], row["start_y"], row["start_z"] = (f"{start[0]:.6f}", f"{start[1]:.6f}", f"{start[2]:.6f}")
        row["end_x"], row["end_y"], row["end_z"] = (f"{end[0]:.6f}", f"{end[1]:.6f}", f"{end[2]:.6f}")
        if "length_mm" in cols:
            row["length_mm"] = f"{float(np.linalg.norm(end - start)):.3f}"
        n += 1

    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(out)
    return n
