"""Headless atlas providers for rosa_agent contact labeling.

Conforms to the same Provider duck-type the Slicer-side
``atlas_assignment_policy.collect_provider_samples`` expects:

    provider.is_ready() -> bool
    provider.sample_contact(point_world_ras) -> AtlasSampleResult | None

Without Slicer / VTK we use ``nibabel`` to load the labelmap NIfTI and a
``scipy.spatial.cKDTree`` (when scipy is available) to answer
nearest-labeled-voxel queries. Brute-force NumPy is the fallback.

Two concrete providers:

* ``LabelmapAtlasProvider`` — single labelmap volume + LUT (FreeSurfer
  parcellation, white-matter mask, etc.).
* ``ThomasAtlasProvider`` — multiple per-segment labelmaps from a THOMAS
  output directory; each segment's foreground voxels become a labeled
  point cloud the same way the Slicer ``ThomasSegmentationAtlasProvider``
  does after ``ExportSegmentsToLabelmapNode``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence


def _require_numpy():
    import numpy as np
    return np


def _load_label_volume(path: str | Path):
    """Return (label_array, affine_xyz_to_ras_4x4, shape_kji).

    nibabel's ``affine`` maps voxel ``(i, j, k, 1)`` to world coordinates
    in **RAS** (NIfTI's native frame), which is exactly what we need —
    no LPS↔RAS flip required.
    """
    import nibabel as nib
    np = _require_numpy()

    img = nib.load(str(path))
    arr = np.asarray(img.get_fdata(caching="unchanged"), dtype=np.int32)
    affine = np.asarray(img.affine, dtype=float)
    return arr, affine, arr.shape


def _voxel_centers_ras(arr_ijk, affine_ijk_to_ras, label_filter=None):
    """Return (RAS Nx3, label values) for foreground voxels.

    ``arr_ijk`` is indexed ``[i, j, k]`` per nibabel's ``get_fdata``.
    ``label_filter`` (optional set) keeps only voxels with those labels.
    """
    np = _require_numpy()
    if label_filter is None:
        mask = arr_ijk > 0
    else:
        mask = np.isin(arr_ijk, np.asarray(list(label_filter), dtype=arr_ijk.dtype))
    ijk = np.argwhere(mask)
    if ijk.size == 0:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=np.int32)
    labels = arr_ijk[ijk[:, 0], ijk[:, 1], ijk[:, 2]].astype(np.int32)
    h = np.ones((ijk.shape[0], 4), dtype=float)
    h[:, :3] = ijk.astype(float)
    ras = (h @ affine_ijk_to_ras.T)[:, :3]
    return ras, labels


def _build_kdtree(points_ras):
    """Return a callable ``query(point_ras) -> (idx, distance_mm)``."""
    np = _require_numpy()
    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(points_ras)

        def _query(point):
            d, idx = tree.query(np.asarray(point, dtype=float).reshape(3), k=1)
            return int(idx), float(d)

        return _query
    except ImportError:
        pts = np.asarray(points_ras, dtype=float)

        def _query(point):
            p = np.asarray(point, dtype=float).reshape(3)
            d2 = ((pts - p) ** 2).sum(axis=1)
            idx = int(np.argmin(d2))
            return idx, float(math.sqrt(float(d2[idx])))

        return _query


# ---------------------------------------------------------------------
# LUT parsing — FreeSurferColorLUT format (id, name, R, G, B, A).
# ---------------------------------------------------------------------


def parse_freesurfer_lut(path: str | Path) -> dict[int, str]:
    """Parse a FreeSurfer color LUT file into ``{label_value: label_name}``."""
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


# ---------------------------------------------------------------------
# Single-labelmap provider.
# ---------------------------------------------------------------------


class LabelmapAtlasProvider:
    """Atlas provider for one label volume (FreeSurfer parcellation, WM, etc.)."""

    def __init__(
        self,
        source_id: str,
        display_name: str,
        label_path: str | Path,
        *,
        lut_path: str | Path | None = None,
        label_names: dict[int, str] | None = None,
    ) -> None:
        self.source_id = str(source_id)
        self.display_name = str(display_name)
        self.label_path = str(label_path)

        np = _require_numpy()
        self._np = np

        self._arr, self._affine, self._shape = _load_label_volume(label_path)
        self._affine_inv = np.linalg.inv(self._affine)

        ras, labels = _voxel_centers_ras(self._arr, self._affine)
        self._points_ras = ras
        self._labels = labels
        self._query = _build_kdtree(ras) if ras.shape[0] else None

        if label_names is not None:
            self._label_names = dict(label_names)
        elif lut_path is not None:
            self._label_names = parse_freesurfer_lut(lut_path)
        else:
            self._label_names = {}

        # Per-label centroid in RAS for ``distance_to_centroid_mm``.
        self._centroids: dict[int, Any] = {}
        if ras.shape[0]:
            unique = np.unique(labels)
            for value in unique:
                mask = labels == int(value)
                if not mask.any():
                    continue
                self._centroids[int(value)] = ras[mask].mean(axis=0)

    def is_ready(self) -> bool:
        return self._query is not None

    def sample_contact(self, point_world_ras: Sequence[float]) -> dict[str, Any] | None:
        if not self.is_ready():
            return None
        idx, distance_mm = self._query(point_world_ras)
        label_value = int(self._labels[idx])
        label_name = self._label_names.get(label_value, f"Label_{label_value}")
        centroid = self._centroids.get(label_value)
        if centroid is None:
            distance_to_centroid = 0.0
        else:
            dx = float(point_world_ras[0]) - float(centroid[0])
            dy = float(point_world_ras[1]) - float(centroid[1])
            dz = float(point_world_ras[2]) - float(centroid[2])
            distance_to_centroid = math.sqrt(dx * dx + dy * dy + dz * dz)
        return {
            "source": self.source_id,
            "label": str(label_name),
            "label_value": label_value,
            "distance_to_voxel_mm": float(distance_mm),
            "distance_to_centroid_mm": float(distance_to_centroid),
            # Headless agent: native and world coordinates coincide
            # (no per-volume Slicer transform stack).
            "native_ras": [float(v) for v in point_world_ras],
        }


# ---------------------------------------------------------------------
# THOMAS provider — directory of per-segment labelmaps.
# ---------------------------------------------------------------------


class ThomasAtlasProvider:
    """Atlas provider for THOMAS segmentation output directory.

    Each ``*.nii.gz`` (excluding ``thalamus*``) under ``segmentation_dir``
    is treated as one segment. Segment name = file stem; label value is
    auto-assigned starting from 1 in stable filename order. This matches
    the Slicer-side provider's behavior where each segment becomes one
    set of foreground voxels merged into a single nearest-neighbor
    index.
    """

    SKIP_PATTERNS = ("thalamus",)

    def __init__(
        self,
        segmentation_dir: str | Path,
        *,
        source_id: str = "thomas",
        display_name: str = "THOMAS",
    ) -> None:
        self.source_id = str(source_id)
        self.display_name = str(display_name)
        self.segmentation_dir = str(segmentation_dir)

        np = _require_numpy()
        seg_paths = sorted(
            p for p in Path(segmentation_dir).glob("*.nii*")
            if not any(skip in p.name.lower() for skip in self.SKIP_PATTERNS)
        )

        all_pts = []
        all_labels = []
        label_names: dict[int, str] = {}
        next_label = 1
        for path in seg_paths:
            arr, affine, _ = _load_label_volume(path)
            ras, _labels = _voxel_centers_ras(arr, affine)
            if ras.shape[0] == 0:
                continue
            all_pts.append(ras)
            all_labels.append(np.full((ras.shape[0],), next_label, dtype=np.int32))
            label_names[next_label] = path.stem.replace(".nii", "")
            next_label += 1

        if not all_pts:
            self._points_ras = np.empty((0, 3), dtype=float)
            self._labels = np.empty((0,), dtype=np.int32)
            self._query = None
        else:
            self._points_ras = np.vstack(all_pts)
            self._labels = np.concatenate(all_labels)
            self._query = _build_kdtree(self._points_ras)

        self._label_names = label_names
        self._centroids: dict[int, Any] = {}
        if self._points_ras.shape[0]:
            for value in np.unique(self._labels):
                mask = self._labels == int(value)
                if not mask.any():
                    continue
                self._centroids[int(value)] = self._points_ras[mask].mean(axis=0)

    def is_ready(self) -> bool:
        return self._query is not None

    def sample_contact(self, point_world_ras: Sequence[float]) -> dict[str, Any] | None:
        if not self.is_ready():
            return None
        idx, distance_mm = self._query(point_world_ras)
        label_value = int(self._labels[idx])
        label_name = self._label_names.get(label_value, f"Label_{label_value}")
        centroid = self._centroids.get(label_value)
        if centroid is None:
            distance_to_centroid = 0.0
        else:
            dx = float(point_world_ras[0]) - float(centroid[0])
            dy = float(point_world_ras[1]) - float(centroid[1])
            dz = float(point_world_ras[2]) - float(centroid[2])
            distance_to_centroid = math.sqrt(dx * dx + dy * dy + dz * dz)
        return {
            "source": self.source_id,
            "label": str(label_name),
            "label_value": label_value,
            "distance_to_voxel_mm": float(distance_mm),
            "distance_to_centroid_mm": float(distance_to_centroid),
            "native_ras": [float(v) for v in point_world_ras],
        }
