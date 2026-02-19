"""Masking and depth-gating helpers for CT shank detection.

Active pipeline:
- build head mask from CT intensity + morphology
- threshold metal candidates
- gate candidates by head mask
- gate candidates by signed distance-to-head-surface (depth in mm)
"""

from __future__ import annotations
import time

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
        raise RuntimeError("numpy is required for masking operations")


def _require_sitk():
    if sitk is None:
        raise RuntimeError("SimpleITK is required for masking operations")


def _radius_xyz(mm: float, spacing_xyz) -> list[int]:
    """Convert a physical radius in mm to voxel radius in X/Y/Z."""
    return [
        max(0, int(round(float(mm) / max(1e-6, float(spacing_xyz[0]))))),
        max(0, int(round(float(mm) / max(1e-6, float(spacing_xyz[1]))))),
        max(0, int(round(float(mm) / max(1e-6, float(spacing_xyz[2]))))),
    ]


def largest_component_binary(binary_img):
    """Return largest connected component from a binary SimpleITK image."""
    _require_sitk()
    cc = sitk.ConnectedComponent(binary_img)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    labels = list(stats.GetLabels())
    if not labels:
        return None
    best_label = max(labels, key=lambda lbl: stats.GetPhysicalSize(lbl))
    return sitk.Equal(cc, int(best_label))


def keep_largest_slice_component_kji(mask_kji, axis: int = 0, min_slice_voxels: int = 64):
    """Keep only the largest connected component per slice for one axis."""
    _require_numpy()
    mask = np.asarray(mask_kji, dtype=bool)
    if sitk is None:
        return mask

    out = np.zeros_like(mask, dtype=bool)
    axis = int(axis)
    if axis == 0:
        n_slices = mask.shape[0]
    elif axis == 1:
        n_slices = mask.shape[1]
    else:
        n_slices = mask.shape[2]

    for idx in range(n_slices):
        if axis == 0:
            sl = mask[idx, :, :]
        elif axis == 1:
            sl = mask[:, idx, :]
        else:
            sl = mask[:, :, idx]

        if int(np.count_nonzero(sl)) < int(min_slice_voxels):
            continue
        sl_img = sitk.GetImageFromArray(sl.astype(np.uint8))
        sl_largest = largest_component_binary(sl_img)
        if sl_largest is None:
            continue

        sl_out = sitk.GetArrayFromImage(sl_largest).astype(bool)
        if axis == 0:
            out[idx, :, :] = sl_out
        elif axis == 1:
            out[:, idx, :] = sl_out
        else:
            out[:, :, idx] = sl_out
    return out


def fill_holes_axial_kji(mask_kji):
    """Fill enclosed holes slice-wise in axial plane."""
    _require_numpy()
    mask = np.asarray(mask_kji, dtype=bool)
    if sitk is None:
        return mask

    out = mask.copy()
    for k in range(out.shape[0]):
        sl = out[k]
        if int(np.count_nonzero(sl)) < 32:
            continue
        sl_img = sitk.GetImageFromArray(sl.astype(np.uint8))
        sl_fill = sitk.BinaryFillhole(sl_img)
        out[k] = sitk.GetArrayFromImage(sl_fill).astype(bool)
    return out


def build_head_mask_kji(
    arr_kji,
    spacing_xyz,
    threshold_hu: float = -300.0,
    close_mm: float = 2.0,
    aggressive_cleanup: bool = True,
):
    """Build largest-component head mask in KJI index order."""
    _require_numpy()
    _require_sitk()

    img = sitk.GetImageFromArray(np.asarray(arr_kji, dtype=np.float32))
    img.SetSpacing((float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])))
    mask = sitk.BinaryThreshold(
        img,
        lowerThreshold=float(threshold_hu),
        upperThreshold=1e9,
        insideValue=1,
        outsideValue=0,
    )

    close_mm = max(0.0, float(close_mm))
    if close_mm > 1e-6:
        radius_xyz = _radius_xyz(close_mm, spacing_xyz)
        if any(r > 0 for r in radius_xyz):
            mask = sitk.BinaryMorphologicalClosing(mask, radius_xyz)

    largest = largest_component_binary(mask)
    if largest is None:
        return np.zeros(np.asarray(arr_kji).shape, dtype=bool)

    largest_arr = sitk.GetArrayFromImage(largest).astype(bool)
    largest_arr = fill_holes_axial_kji(largest_arr)
    if bool(aggressive_cleanup):
        largest_arr = keep_largest_slice_component_kji(largest_arr, axis=0, min_slice_voxels=64)
        largest_arr = keep_largest_slice_component_kji(largest_arr, axis=1, min_slice_voxels=64)
        largest_arr = keep_largest_slice_component_kji(largest_arr, axis=2, min_slice_voxels=64)
    else:
        largest_arr = keep_largest_slice_component_kji(largest_arr, axis=0, min_slice_voxels=64)
    return largest_arr


def compute_head_distance_map_kji(head_mask_kji, spacing_xyz):
    """Compute signed head-surface distance map (mm), positive inside mask."""
    _require_numpy()
    _require_sitk()
    mask = np.asarray(head_mask_kji, dtype=np.uint8)
    if mask.size == 0:
        return np.zeros((0,), dtype=np.float32)
    head_img = sitk.GetImageFromArray(mask)
    head_img.SetSpacing((float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])))
    dist_img = sitk.SignedMaurerDistanceMap(
        head_img,
        insideIsPositive=True,
        squaredDistance=False,
        useImageSpacing=True,
    )
    return sitk.GetArrayFromImage(dist_img).astype(np.float32)


def suggest_metal_threshold_hu_from_array(arr_kji):
    """Suggest metal HU threshold from full CT histogram."""
    _require_numpy()
    values = np.asarray(arr_kji, dtype=np.float32).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 1000:
        return 1800.0
    if values.size > 2_000_000:
        rng = np.random.default_rng(0)
        pick = rng.choice(values.size, size=2_000_000, replace=False)
        values = values[pick]
    suggested = float(np.percentile(values, 99.8))
    suggested = float(np.clip(suggested, 1200.0, 3000.0))
    suggested = float(round(suggested / 25.0) * 25.0)
    return suggested


def build_preview_masks(
    arr_kji,
    spacing_xyz,
    threshold,
    use_head_mask=False,
    build_head_mask=True,
    head_mask_threshold_hu=-300.0,
    head_mask_aggressive_cleanup=True,
    head_mask_close_mm=2.0,
    min_metal_depth_mm=5.0,
    max_metal_depth_mm=220.0,
    precomputed_gating_mask_kji=None,
    precomputed_head_distance_map_kji=None,
):
    """Build metal candidates and apply optional head/depth gating.

    Returns dictionary keys consumed by both UI and CLI.
    Depth values are in mm from the outer head surface (inside-positive),
    with practical defaults of 5-220 mm.
    """
    _require_numpy()
    _require_sitk()
    t0 = time.perf_counter()
    t_threshold0 = time.perf_counter()

    arr = np.asarray(arr_kji)
    metal_mask = np.asarray(arr >= float(threshold), dtype=bool)
    candidate_count = int(np.count_nonzero(metal_mask))
    threshold_ms = (time.perf_counter() - t_threshold0) * 1000.0

    gating_mask = None
    inside_method = "none"
    gating_mask_type = "none"

    metal_in_head_count = candidate_count
    depth_kept_count = candidate_count
    dist_head = None
    metal_depth_all_mm = np.empty((0,), dtype=np.float32)
    metal_depth_values_mm = np.empty((0,), dtype=np.float32)

    if bool(use_head_mask) or bool(build_head_mask):
        t_head0 = time.perf_counter()
        if precomputed_gating_mask_kji is not None:
            gating_mask = np.asarray(precomputed_gating_mask_kji, dtype=bool)
        else:
            head_mask = build_head_mask_kji(
                arr_kji=arr,
                spacing_xyz=spacing_xyz,
                threshold_hu=head_mask_threshold_hu,
                close_mm=head_mask_close_mm,
                aggressive_cleanup=head_mask_aggressive_cleanup,
            )
            gating_mask = np.asarray(head_mask, dtype=bool)
        head_mask_ms = (time.perf_counter() - t_head0) * 1000.0
        inside_method = "head_distance"
        gating_mask_type = "head_distance"

        t_dist0 = time.perf_counter()
        if precomputed_head_distance_map_kji is not None:
            dist_head = np.asarray(precomputed_head_distance_map_kji, dtype=np.float32)
        else:
            dist_head = compute_head_distance_map_kji(gating_mask, spacing_xyz=spacing_xyz)
        distance_map_ms = (time.perf_counter() - t_dist0) * 1000.0
    else:
        head_mask_ms = 0.0
        distance_map_ms = 0.0

    in_mask_ijk_kji = np.empty((0, 3), dtype=int)
    head_mask_kept_count = candidate_count
    t_enum0 = time.perf_counter()
    head_gate_ms = 0.0
    depth_gate_ms = 0.0
    if candidate_count > 0:
        ijk_kji = np.argwhere(metal_mask)
        candidate_enum_ms = (time.perf_counter() - t_enum0) * 1000.0
        if gating_mask is not None:
            t_gate0 = time.perf_counter()
            keep = gating_mask[ijk_kji[:, 0], ijk_kji[:, 1], ijk_kji[:, 2]]
            ijk_kji = ijk_kji[keep]
            metal_in_head_count = int(ijk_kji.shape[0])
            head_gate_ms = (time.perf_counter() - t_gate0) * 1000.0
            if dist_head is not None and metal_in_head_count > 0:
                t_depth0 = time.perf_counter()
                depth = dist_head[ijk_kji[:, 0], ijk_kji[:, 1], ijk_kji[:, 2]]
                metal_depth_all_mm = np.asarray(depth, dtype=np.float32)
                metal_depth_values_mm = metal_depth_all_mm.copy()
                min_d = max(0.0, float(min_metal_depth_mm))
                max_d = float(max_metal_depth_mm) if max_metal_depth_mm is not None else float("inf")
                if max_d <= 0.0:
                    max_d = float("inf")
                keep_depth = np.logical_and(metal_depth_values_mm >= min_d, metal_depth_values_mm <= max_d)
                ijk_kji = ijk_kji[keep_depth]
                metal_depth_values_mm = metal_depth_values_mm[keep_depth]
                depth_gate_ms = (time.perf_counter() - t_depth0) * 1000.0
            depth_kept_count = int(ijk_kji.shape[0])
        in_mask_ijk_kji = np.asarray(ijk_kji, dtype=int)
        head_mask_kept_count = int(in_mask_ijk_kji.shape[0])
    else:
        candidate_enum_ms = (time.perf_counter() - t_enum0) * 1000.0

    return {
        "candidate_count": candidate_count,
        "head_mask_kept_count": int(head_mask_kept_count),
        "metal_in_head_count": int(metal_in_head_count),
        "depth_kept_count": int(depth_kept_count),
        "gating_mask_type": gating_mask_type,
        "inside_method": inside_method,
        "metal_mask_kji": metal_mask.astype(np.uint8),
        # Keep both keys for compatibility while shifting terminology to "gating".
        "gating_mask_kji": gating_mask.astype(np.uint8) if gating_mask is not None else None,
        "head_mask_kji": gating_mask.astype(np.uint8) if gating_mask is not None else None,
        "head_distance_map_kji": dist_head,
        "in_mask_ijk_kji": in_mask_ijk_kji,
        "metal_depth_all_mm": metal_depth_all_mm,
        "metal_depth_values_mm": metal_depth_values_mm,
        "profile_ms": {
            "threshold": float(threshold_ms),
            "head_mask": float(head_mask_ms),
            "distance_map": float(distance_map_ms),
            "candidate_enum": float(candidate_enum_ms),
            "head_gate": float(head_gate_ms),
            "depth_gate": float(depth_gate_ms),
            "total": float((time.perf_counter() - t0) * 1000.0),
        },
        "profile_flags": {
            "used_precomputed_gating_mask": bool(precomputed_gating_mask_kji is not None),
            "used_precomputed_distance_map": bool(precomputed_head_distance_map_kji is not None),
        },
    }
