"""NIfTI IO helpers for shank detection CLI."""

from __future__ import annotations

import csv
import os

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import SimpleITK as sitk
except ImportError:  # pragma: no cover
    sitk = None


def _require_numpy():
    if np is None:
        raise RuntimeError("numpy is required")


def _require_sitk():
    if sitk is None:
        raise RuntimeError("SimpleITK is required")


def read_volume(path):
    """Read a scalar image file and return (image, array_kji, spacing_xyz)."""
    _require_sitk()
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    return img, arr, spacing


def write_mask_like(reference_img, mask_kji, out_path):
    """Write binary mask array as image copying geometry from reference image."""
    _require_sitk()
    arr = np.asarray(mask_kji, dtype=np.uint8)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(reference_img)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sitk.WriteImage(out, out_path)


def kji_to_ras_points(reference_img, ijk_kji):
    """Convert KJI voxel indices to RAS points using ITK geometry.

    SimpleITK physical coordinates are LPS, so output is converted to RAS.
    """
    _require_numpy()
    ijk_kji = np.asarray(ijk_kji, dtype=float)
    if ijk_kji.size == 0:
        return np.empty((0, 3), dtype=float)

    # k,j,i -> i,j,k
    ijk_xyz = np.zeros_like(ijk_kji, dtype=float)
    ijk_xyz[:, 0] = ijk_kji[:, 2]
    ijk_xyz[:, 1] = ijk_kji[:, 1]
    ijk_xyz[:, 2] = ijk_kji[:, 0]

    spacing = np.array(reference_img.GetSpacing(), dtype=float)
    origin_lps = np.array(reference_img.GetOrigin(), dtype=float)
    direction = np.array(reference_img.GetDirection(), dtype=float).reshape(3, 3)

    scaled = ijk_xyz * spacing.reshape(1, 3)
    lps = origin_lps.reshape(1, 3) + scaled @ direction.T
    ras = lps.copy()
    ras[:, 0] *= -1.0
    ras[:, 1] *= -1.0
    return ras


def write_points_csv(out_path, points, columns=("x", "y", "z")):
    """Write N x 3 points CSV with configurable header."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(columns))
        for p in pts:
            w.writerow([f"{float(p[0]):.6f}", f"{float(p[1]):.6f}", f"{float(p[2]):.6f}"])
