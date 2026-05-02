"""Pure-Python primitives shared by atlas providers (CLI + Slicer).

The two atlas-provider families — Slicer's vtk-backed
``rosa_scene.atlas_providers`` and the CLI's nibabel+scipy
``cli/rosa_agent/services/atlas_provider_headless`` — used to carry
their own copies of:

  * per-label centroid computation,
  * sample-result dict formatting,
  * FreeSurfer color-LUT text parsing.

The locator backend (vtkPointLocator vs scipy.cKDTree) and the source
of the labeled-voxel point cloud (slicer.util.arrayFromVolume vs
nibabel.load) legitimately differ between sides — those stay forked.
Everything else lives here so both sides can't drift.

No Slicer / VTK / Qt imports.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def compute_label_centroids(
    points_ras,
    labels,
) -> dict[int, Any]:
    """Return ``{label_value: centroid_ras_3vec}`` for an N×3 point cloud.

    Centroid is the per-label mean of RAS coords. Labels with zero
    points (shouldn't happen if the cloud was built from the labelmap
    itself, but guards against caller bugs) are skipped.

    The centroid is returned as whatever array type the caller passes
    in (numpy array, slice, ...) — no conversion. Callers that need
    a tuple/list can cast at use site.
    """
    import numpy as np

    pts = np.asarray(points_ras, dtype=float)
    lbls = np.asarray(labels, dtype=np.int64)
    out: dict[int, Any] = {}
    if pts.shape[0] == 0:
        return out
    for value in np.unique(lbls):
        mask = lbls == int(value)
        if not mask.any():
            continue
        out[int(value)] = pts[mask].mean(axis=0)
    return out


def distance_to_centroid_mm(point_ras: Sequence[float], centroid) -> float:
    """Euclidean distance from a query RAS point to a centroid RAS point.

    Returns 0.0 when the centroid is None (caller convention: no
    centroid means "treat as on-target"). Otherwise plain L2 in mm.
    """
    if centroid is None:
        return 0.0
    dx = float(point_ras[0]) - float(centroid[0])
    dy = float(point_ras[1]) - float(centroid[1])
    dz = float(point_ras[2]) - float(centroid[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def format_atlas_sample(
    *,
    source_id: str,
    label_value: int,
    label_name: str,
    distance_to_voxel_mm: float,
    distance_to_centroid_mm: float,
    native_ras: Sequence[float],
) -> dict[str, Any]:
    """Return one sample dict matching ``rosa_scene.atlas_provider_types
    .AtlasSampleResult``.

    Single source of truth for the dict shape — the
    ``atlas_assignment_policy.collect_provider_samples`` consumer
    reads ``label`` / ``label_value`` / ``distance_to_voxel_mm`` etc.
    by string key, so any drift between Slicer and CLI providers in
    these key names would silently break label assignment.
    """
    return {
        "source": str(source_id),
        "label": str(label_name),
        "label_value": int(label_value),
        "distance_to_voxel_mm": float(distance_to_voxel_mm),
        "distance_to_centroid_mm": float(distance_to_centroid_mm),
        "native_ras": [
            float(native_ras[0]),
            float(native_ras[1]),
            float(native_ras[2]),
        ],
    }


def parse_freesurfer_lut(path: str | Path) -> dict[int, str]:
    """Parse a FreeSurfer color LUT text file into ``{label_value: name}``.

    Format (each non-comment line): ``<int_id> <name> <R> <G> <B> <A>``.
    Header lines starting with ``#`` and blank lines are skipped.
    Lines that don't start with an integer are silently dropped — the
    caller does not need to pre-clean the file.
    """
    out: dict[int, str] = {}
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            value = int(parts[0])
        except ValueError:
            continue
        out[value] = parts[1]
    return out
