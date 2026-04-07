"""Blob/connected-component candidate extraction for shank detection."""

from __future__ import annotations

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
        raise RuntimeError("numpy is required for blob candidate extraction")


def _component_labels(mask_kji, fully_connected=True):
    """Return integer component labels in KJI order."""
    _require_numpy()
    mask = np.asarray(mask_kji, dtype=np.uint8)
    if mask.size == 0:
        return np.zeros((0,), dtype=np.int32)
    if sitk is None:
        # Fallback when SimpleITK is unavailable: treat every metal voxel as one component.
        # This preserves functionality, though blob aggregation is effectively disabled.
        out = np.zeros(mask.shape, dtype=np.int32)
        idx = np.argwhere(mask > 0)
        for blob_id, p in enumerate(idx, start=1):
            out[int(p[0]), int(p[1]), int(p[2])] = int(blob_id)
        return out
    cc = sitk.ConnectedComponent(sitk.GetImageFromArray(mask), bool(fully_connected))
    return sitk.GetArrayFromImage(cc).astype(np.int32)


def _blob_elongation_kji(points_kji):
    """Return principal-axis elongation ratio for one blob."""
    _require_numpy()
    if points_kji.shape[0] < 3:
        return 1.0
    x = points_kji.astype(np.float64)
    x = x - np.mean(x, axis=0, keepdims=True)
    cov = (x.T @ x) / max(1, x.shape[0] - 1)
    evals = np.linalg.eigvalsh(cov)
    evals = np.sort(np.maximum(evals, 1e-9))
    return float(evals[-1] / evals[0])


def _blob_pca_features(points_xyz):
    """Return PCA axis/eigenvalues for points in XYZ space."""
    _require_numpy()
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.shape[0] < 3:
        return {
            "axis": np.asarray([0.0, 0.0, 1.0], dtype=float),
            "evals": np.asarray([0.0, 0.0, 0.0], dtype=float),
        }
    centered = pts - np.mean(pts, axis=0, keepdims=True)
    cov = (centered.T @ centered) / max(1, pts.shape[0] - 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)
    evals = np.maximum(evals[order], 1e-9)
    axis = evecs[:, order[-1]]
    n = float(np.linalg.norm(axis))
    if n > 1e-9:
        axis = axis / n
    else:
        axis = np.asarray([0.0, 0.0, 1.0], dtype=float)
    return {"axis": axis.astype(float), "evals": evals.astype(float)}


def extract_blob_candidates(
    metal_mask_kji,
    arr_kji=None,
    depth_map_kji=None,
    ijk_kji_to_ras_fn=None,
    fully_connected=True,
):
    """Extract connected-component blobs and per-blob features."""
    _require_numpy()
    labels = _component_labels(metal_mask_kji, fully_connected=fully_connected)
    if labels.size == 0:
        return {
            "labels_kji": np.zeros((0,), dtype=np.int32),
            "blobs": [],
            "blob_count_total": 0,
        }

    coords = np.argwhere(labels > 0)
    if coords.size == 0:
        return {
            "labels_kji": labels,
            "blobs": [],
            "blob_count_total": 0,
        }
    point_labels = labels[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.int32)
    blob_ids = np.unique(point_labels)
    blob_ids = blob_ids[blob_ids > 0]
    arr_vals = None
    depth_vals = None
    if arr_kji is not None:
        arr_vals = np.asarray(arr_kji, dtype=np.float32)[coords[:, 0], coords[:, 1], coords[:, 2]]
    if depth_map_kji is not None:
        depth_vals = np.asarray(depth_map_kji, dtype=np.float32)[coords[:, 0], coords[:, 1], coords[:, 2]]
    blobs = []
    for blob_id in blob_ids:
        sel = point_labels == int(blob_id)
        idx = coords[sel]
        if idx.size == 0:
            continue
        centroid_kji = np.mean(idx.astype(np.float64), axis=0)
        centroid_ras = None
        pca_axis_ras = None
        pca_evals = np.asarray([0.0, 0.0, 0.0], dtype=float)
        length_mm = 0.0
        diameter_mm = 0.0
        if ijk_kji_to_ras_fn is not None:
            centroid_ras = np.asarray(ijk_kji_to_ras_fn([centroid_kji]), dtype=float).reshape(-1, 3)[0]
            ras_pts = np.asarray(ijk_kji_to_ras_fn(idx.astype(np.float64)), dtype=float).reshape(-1, 3)
            pca = _blob_pca_features(ras_pts)
            pca_axis_ras = pca["axis"]
            pca_evals = pca["evals"]
            if ras_pts.shape[0] >= 2:
                proj = (ras_pts - np.mean(ras_pts, axis=0, keepdims=True)) @ pca_axis_ras.reshape(3)
                length_mm = float(np.max(proj) - np.min(proj))
            diameter_mm = float(2.0 * np.sqrt(max(1e-9, float(np.mean(pca_evals[:2])))))
        else:
            pca = _blob_pca_features(idx.astype(np.float64))
            pca_evals = pca["evals"]
            pca_axis_ras = pca["axis"]
            if idx.shape[0] >= 2:
                proj = (idx.astype(np.float64) - np.mean(idx.astype(np.float64), axis=0, keepdims=True)) @ pca_axis_ras.reshape(3)
                length_mm = float(np.max(proj) - np.min(proj))
            diameter_mm = float(2.0 * np.sqrt(max(1e-9, float(np.mean(pca_evals[:2])))))

        hu_max = None
        hu_q95 = None
        hu_mean = None
        if arr_vals is not None:
            vals = arr_vals[sel]
            if vals.size > 0:
                hu_max = float(np.max(vals))
                hu_q95 = float(np.percentile(vals, 95))
                hu_mean = float(np.mean(vals))

        d_min = None
        d_mean = None
        d_max = None
        if depth_vals is not None:
            dvals = depth_vals[sel]
            if dvals.size > 0:
                d_min = float(np.min(dvals))
                d_mean = float(np.mean(dvals))
                d_max = float(np.max(dvals))

        blobs.append(
            {
                "blob_id": int(blob_id),
                "voxel_count": int(idx.shape[0]),
                "centroid_kji": [float(centroid_kji[0]), float(centroid_kji[1]), float(centroid_kji[2])],
                "centroid_ras": (
                    [float(centroid_ras[0]), float(centroid_ras[1]), float(centroid_ras[2])] if centroid_ras is not None else None
                ),
                "hu_max": hu_max,
                "hu_q95": hu_q95,
                "hu_mean": hu_mean,
                "depth_min": d_min,
                "depth_mean": d_mean,
                "depth_max": d_max,
                "elongation": _blob_elongation_kji(idx),
                "pca_axis_ras": [float(pca_axis_ras[0]), float(pca_axis_ras[1]), float(pca_axis_ras[2])],
                "pca_evals": [float(pca_evals[0]), float(pca_evals[1]), float(pca_evals[2])],
                "length_mm": float(length_mm),
                "diameter_mm": float(diameter_mm),
            }
        )
    return {
        "labels_kji": labels,
        "blobs": blobs,
        "blob_count_total": int(len(blobs)),
    }


def filter_blob_candidates(
    blob_result,
    min_blob_voxels=2,
    max_blob_voxels=250,
    min_blob_peak_hu=None,
    max_blob_elongation=None,
):
    """Filter extracted blobs and classify reject reasons."""
    _require_numpy()
    blobs = list(blob_result.get("blobs", []) or [])
    kept = []
    rejected = []
    reject_small = 0
    reject_large = 0
    reject_intensity = 0
    reject_shape = 0
    for blob in blobs:
        vox = int(blob.get("voxel_count", 0))
        if vox < int(min_blob_voxels):
            reject_small += 1
            rejected.append((blob, "small"))
            continue
        if int(max_blob_voxels) > 0 and vox > int(max_blob_voxels):
            reject_large += 1
            rejected.append((blob, "large"))
            continue
        if min_blob_peak_hu is not None:
            hu_max = blob.get("hu_max", None)
            if hu_max is None or float(hu_max) < float(min_blob_peak_hu):
                reject_intensity += 1
                rejected.append((blob, "intensity"))
                continue
        if max_blob_elongation is not None:
            elong = float(blob.get("elongation", 1.0))
            if elong > float(max_blob_elongation):
                reject_shape += 1
                rejected.append((blob, "shape"))
                continue
        kept.append(blob)

    kept_points_ras = np.asarray(
        [blob["centroid_ras"] for blob in kept if blob.get("centroid_ras") is not None],
        dtype=float,
    ).reshape(-1, 3)
    rejected_points_ras = np.asarray(
        [blob["centroid_ras"] for (blob, _reason) in rejected if blob.get("centroid_ras") is not None],
        dtype=float,
    ).reshape(-1, 3)
    all_points_ras = np.asarray(
        [blob["centroid_ras"] for blob in blobs if blob.get("centroid_ras") is not None],
        dtype=float,
    ).reshape(-1, 3)
    weights = np.asarray([float(max(1.0, np.sqrt(float(blob.get("voxel_count", 1))))) for blob in kept], dtype=float)

    return {
        "kept_blobs": kept,
        "rejected_blobs": rejected,
        "blob_count_kept": int(len(kept)),
        "blob_reject_small": int(reject_small),
        "blob_reject_large": int(reject_large),
        "blob_reject_intensity": int(reject_intensity),
        "blob_reject_shape": int(reject_shape),
        "blob_centroids_all_ras": all_points_ras,
        "blob_centroids_kept_ras": kept_points_ras,
        "blob_centroids_rejected_ras": rejected_points_ras,
        "candidate_points_ras": kept_points_ras,
        "candidate_weights": weights,
        "kept_blob_ids": [int(blob["blob_id"]) for blob in kept],
    }


def build_blob_labelmap(labels_kji, keep_blob_ids=None):
    """Build uint16 component labelmap from integer component labels."""
    _require_numpy()
    labels = np.asarray(labels_kji, dtype=np.int32)
    if labels.size == 0:
        return np.zeros((0,), dtype=np.uint16)
    out = np.zeros(labels.shape, dtype=np.uint16)
    blob_ids = np.unique(labels)
    blob_ids = blob_ids[blob_ids > 0]
    if keep_blob_ids is not None:
        keep_set = {int(v) for v in keep_blob_ids}
        blob_ids = np.asarray([int(v) for v in blob_ids if int(v) in keep_set], dtype=np.int32)
    for out_id, blob_id in enumerate(blob_ids, start=1):
        out[labels == int(blob_id)] = np.uint16(out_id)
    return out
