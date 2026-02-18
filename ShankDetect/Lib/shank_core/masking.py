"""Masking helpers for CT shank detection.

This module contains reusable, mostly Slicer-independent utilities for:
- head mask construction
- intracranial (inside-skull) mask construction
- optional skull-ray filtering for metal candidates
- visualization-oriented mask cleanup helpers
"""

from __future__ import annotations

from typing import Optional, Tuple

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


def largest_component_binary(binary_img, exclude_border: bool = False):
    """Return the largest connected component from a binary SimpleITK image.

    If ``exclude_border`` is True, components touching image boundary are ignored.
    Returns ``None`` if no valid component exists.
    """
    _require_sitk()
    cc = sitk.ConnectedComponent(binary_img)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    labels = list(stats.GetLabels())
    if not labels:
        return None

    candidates = []
    size_xyz = binary_img.GetSize()
    dim = int(binary_img.GetDimension())
    for lbl in labels:
        if exclude_border:
            bbox = stats.GetBoundingBox(lbl)
            origin_idx = bbox[:dim]
            extent = bbox[dim:]
            touches_border = False
            for axis in range(dim):
                if origin_idx[axis] <= 0 or (origin_idx[axis] + extent[axis]) >= size_xyz[axis]:
                    touches_border = True
                    break
            if touches_border:
                continue
        candidates.append(lbl)

    if not candidates:
        return None
    best_label = max(candidates, key=lambda lbl: stats.GetPhysicalSize(lbl))
    return sitk.Equal(cc, int(best_label))


def keep_largest_slice_component_kji(mask_kji, axis: int = 0, min_slice_voxels: int = 64):
    """Keep only largest connected component on each slice for selected axis."""
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
    """Fill enclosed holes slice-wise (axial) for conservative cleanup."""
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


def suggest_metal_threshold_hu_from_array(arr_kji):
    """Suggest metal HU threshold using full CT histogram."""
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


def _inside_voxel_count(binary_img) -> int:
    if binary_img is None:
        return 0
    return int(np.count_nonzero(sitk.GetArrayFromImage(binary_img)))


def inside_from_directional_bounds_kji(skull_mask_kji, head_mask_kji):
    """Estimate intracranial region by directional skull bounds in axial slices."""
    _require_numpy()
    skull = np.asarray(skull_mask_kji, dtype=bool)
    if head_mask_kji is None:
        return np.zeros_like(skull, dtype=bool)
    head = np.asarray(head_mask_kji, dtype=bool)

    inside = np.zeros_like(skull, dtype=bool)
    for k in range(skull.shape[0]):
        skull2 = skull[k]
        head2 = head[k]
        if int(np.count_nonzero(head2)) < 256 or int(np.count_nonzero(skull2)) < 64:
            continue

        left_hit = np.maximum.accumulate(skull2, axis=1)
        right_hit = np.maximum.accumulate(skull2[:, ::-1], axis=1)[:, ::-1]
        up_hit = np.maximum.accumulate(skull2, axis=0)
        down_hit = np.maximum.accumulate(skull2[::-1, :], axis=0)[::-1, :]

        bounded = left_hit & right_hit & up_hit & down_hit
        inside2 = bounded & head2
        if int(np.count_nonzero(inside2)) < 256:
            continue
        inside[k] = inside2

    return inside


def inside_from_skull_slices_kji(skull_mask_kji, head_mask_kji=None):
    """Estimate intracranial region by filling skull rings in each axial slice."""
    _require_numpy()
    if sitk is None:
        return np.zeros_like(skull_mask_kji, dtype=bool)

    skull = np.asarray(skull_mask_kji, dtype=bool)
    inside = np.zeros_like(skull, dtype=bool)
    for k in range(skull.shape[0]):
        ring = skull[k]
        if int(np.count_nonzero(ring)) < 64:
            continue
        ring_img = sitk.GetImageFromArray(ring.astype(np.uint8))
        best = None
        best_count = 0
        for close_px in [0, 1, 2, 3, 4, 6, 8]:
            ring_try = ring_img
            if close_px > 0:
                ring_try = sitk.BinaryMorphologicalClosing(ring_try, [int(close_px), int(close_px)])
            filled = sitk.BinaryFillhole(ring_try)
            cavity = sitk.And(filled, sitk.Not(ring_try))
            cavity_largest = largest_component_binary(cavity, exclude_border=True)
            if cavity_largest is None:
                continue
            arr = sitk.GetArrayFromImage(cavity_largest).astype(bool)
            cnt = int(np.count_nonzero(arr))
            if cnt > best_count:
                best = arr
                best_count = cnt
            if cnt > 1500:
                break
        if best is not None:
            inside[k] = best

    if head_mask_kji is not None:
        inside = np.logical_and(inside, np.asarray(head_mask_kji, dtype=bool))
    return inside


def ray_hits_skull_before_boundary(skull2, metal2, y0, x0, dy, dx):
    """Return True if ray hits skull before boundary, with no metal after origin cluster."""
    h, w = skull2.shape
    y = int(y0 + dy)
    x = int(x0 + dx)
    while 0 <= y < h and 0 <= x < w and metal2[y, x]:
        y += dy
        x += dx
    while 0 <= y < h and 0 <= x < w:
        if skull2[y, x]:
            return True
        if metal2[y, x]:
            return False
        y += dy
        x += dx
    return False


def filter_metal_points_inside_skull_rays(ijk_kji, skull_mask_kji, metal_mask_kji):
    """Keep points that are skull-bounded in +/-X and +/-Y on each axial slice."""
    _require_numpy()
    if ijk_kji is None or ijk_kji.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    skull = np.asarray(skull_mask_kji, dtype=bool)
    metal = np.asarray(metal_mask_kji, dtype=bool)

    keep = np.zeros((ijk_kji.shape[0],), dtype=bool)
    for idx in range(ijk_kji.shape[0]):
        k, j, i = [int(v) for v in ijk_kji[idx]]
        if (
            k < 0
            or k >= skull.shape[0]
            or j < 0
            or j >= skull.shape[1]
            or i < 0
            or i >= skull.shape[2]
        ):
            continue
        skull2 = skull[k]
        metal2 = metal[k]
        if not (
            ray_hits_skull_before_boundary(skull2, metal2, j, i, 0, 1)
            and ray_hits_skull_before_boundary(skull2, metal2, j, i, 0, -1)
            and ray_hits_skull_before_boundary(skull2, metal2, j, i, 1, 0)
            and ray_hits_skull_before_boundary(skull2, metal2, j, i, -1, 0)
        ):
            continue
        keep[idx] = True
    return keep


def erode_mask_kji(mask_kji, spacing_xyz, erode_mm: float):
    """Binary-erode a KJI mask by physical distance."""
    _require_numpy()
    _require_sitk()
    img = sitk.GetImageFromArray(np.asarray(mask_kji, dtype=np.uint8))
    img.SetSpacing((float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])))

    erode_mm = max(0.0, float(erode_mm))
    if erode_mm <= 1e-6:
        return np.asarray(mask_kji, dtype=bool)

    radius_xyz = _radius_xyz(erode_mm, spacing_xyz)
    if not any(r > 0 for r in radius_xyz):
        return np.asarray(mask_kji, dtype=bool)
    eroded = sitk.BinaryErode(img, radius_xyz)
    return sitk.GetArrayFromImage(eroded).astype(bool)


def fill_inside_display_holes_kji(inside_mask_kji, metal_mask_kji):
    """Return inside mask with metal removed and enclosed 2D holes filled."""
    _require_numpy()
    inside = np.asarray(inside_mask_kji, dtype=bool)
    metal = np.asarray(metal_mask_kji, dtype=bool)
    inside_nom = np.logical_and(inside, np.logical_not(metal))
    if sitk is None:
        return inside_nom

    filled = np.zeros_like(inside_nom, dtype=bool)
    for k in range(inside_nom.shape[0]):
        sl = inside_nom[k]
        if int(np.count_nonzero(sl)) < 16:
            continue
        sl_img = sitk.GetImageFromArray(sl.astype(np.uint8))
        sl_fill = sitk.BinaryFillhole(sl_img)
        filled[k] = sitk.GetArrayFromImage(sl_fill).astype(bool)
    return filled


def _build_skull_mask_lcc(
    arr_kji,
    spacing_xyz,
    skull_threshold_hu: float,
    metal_exclude_hu: float,
    close_mm: float,
):
    """Build skull shell largest component in SITK image form."""
    img = sitk.GetImageFromArray(np.asarray(arr_kji, dtype=np.float32))
    img.SetSpacing((float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])))

    upper = float(metal_exclude_hu)
    lower = float(skull_threshold_hu)
    if upper <= lower + 1.0:
        upper = lower + 1.0

    skull_raw = sitk.BinaryThreshold(
        img,
        lowerThreshold=lower,
        upperThreshold=upper,
        insideValue=1,
        outsideValue=0,
    )

    close_mm = max(0.0, float(close_mm))
    if close_mm > 1e-6:
        radius_xyz = _radius_xyz(close_mm, spacing_xyz)
        if any(r > 0 for r in radius_xyz):
            skull_raw = sitk.BinaryMorphologicalClosing(skull_raw, radius_xyz)

    skull_lcc = largest_component_binary(skull_raw)
    return img, skull_lcc


def _largest_nonborder_component_closest_to_center(binary_img):
    """Pick robust non-border component near image center (index-space).

    To avoid selecting tiny central speckles, first keep only non-border
    components that are at least 5% of the largest non-border component (or
    at least 100 voxels), then select the closest-to-center among them.
    """
    cc = sitk.ConnectedComponent(binary_img)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    labels = list(stats.GetLabels())
    if not labels:
        return None

    size_xyz = binary_img.GetSize()
    dim = int(binary_img.GetDimension())
    center = [(float(size_xyz[a]) - 1.0) * 0.5 for a in range(dim)]

    nonborder = []
    for lbl in labels:
        bbox = stats.GetBoundingBox(lbl)
        origin_idx = bbox[:dim]
        extent = bbox[dim:]

        touches_border = False
        for axis in range(dim):
            if origin_idx[axis] <= 0 or (origin_idx[axis] + extent[axis]) >= size_xyz[axis]:
                touches_border = True
                break
        if touches_border:
            continue

        comp_center = [float(origin_idx[a]) + 0.5 * float(extent[a]) for a in range(dim)]
        dist = sum((comp_center[a] - center[a]) ** 2 for a in range(dim))
        size_vox = int(stats.GetNumberOfPixels(lbl))
        nonborder.append((int(lbl), size_vox, dist))

    if not nonborder:
        return None

    max_size = max(sz for _, sz, _ in nonborder)
    min_keep = max(100, int(round(0.05 * float(max_size))))
    robust = [x for x in nonborder if x[1] >= min_keep]
    pool = robust if robust else nonborder
    best_label = min(pool, key=lambda x: x[2])[0]
    return sitk.Equal(cc, best_label)


def _build_inside_soft_tissue_mask_kji(
    arr_kji,
    spacing_xyz,
    skull_threshold_hu=300.0,
    metal_exclude_hu=1800.0,
    close_mm=2.0,
    erode_mm=1.0,
    head_threshold_hu=-200.0,
    soft_tissue_low_hu=-100.0,
    soft_tissue_high_hu=180.0,
    aggressive_head_cleanup=True,
    debug_outputs=None,
):
    """Build intracranial mask from central soft tissue component.

    This strategy does not rely on a watertight skull shell. It intersects a soft-tissue
    HU window with the head mask and keeps the central non-border component.
    """
    _require_numpy()
    _require_sitk()

    # Fast head mask for soft-tissue mode: threshold + optional closing + largest component.
    # Avoid expensive per-slice multi-axis cleanup to keep interactive runs responsive.
    img = sitk.GetImageFromArray(np.asarray(arr_kji, dtype=np.float32))
    img.SetSpacing((float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])))
    head_img = sitk.BinaryThreshold(
        img,
        lowerThreshold=float(head_threshold_hu),
        upperThreshold=1e9,
        insideValue=1,
        outsideValue=0,
    )
    if float(close_mm) > 1e-6:
        radius_xyz = _radius_xyz(float(close_mm), spacing_xyz)
        if any(r > 0 for r in radius_xyz):
            head_img = sitk.BinaryMorphologicalClosing(head_img, radius_xyz)
    head_lcc = largest_component_binary(head_img)
    head_arr = (
        sitk.GetArrayFromImage(head_lcc).astype(bool)
        if head_lcc is not None
        else np.zeros(np.asarray(arr_kji).shape, dtype=bool)
    )
    if debug_outputs is not None:
        debug_outputs["soft_head_mask_kji"] = head_arr.astype(np.uint8)
    if int(np.count_nonzero(head_arr)) == 0:
        z = np.zeros(np.asarray(arr_kji).shape, dtype=bool)
        if debug_outputs is not None:
            debug_outputs["soft_window_mask_kji"] = z.astype(np.uint8)
            debug_outputs["soft_window_closed_kji"] = z.astype(np.uint8)
            debug_outputs["soft_selected_component_kji"] = z.astype(np.uint8)
        return z, z, "none"

    arr = np.asarray(arr_kji, dtype=np.float32)
    soft = np.logical_and(arr >= float(soft_tissue_low_hu), arr <= float(soft_tissue_high_hu))
    soft = np.logical_and(soft, head_arr)
    if debug_outputs is not None:
        debug_outputs["soft_window_mask_kji"] = soft.astype(np.uint8)

    soft_img = sitk.GetImageFromArray(soft.astype(np.uint8))
    soft_img.SetSpacing((float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])))
    if float(close_mm) > 1e-6:
        radius_xyz = _radius_xyz(float(close_mm), spacing_xyz)
        if any(r > 0 for r in radius_xyz):
            soft_img = sitk.BinaryMorphologicalClosing(soft_img, radius_xyz)
    if debug_outputs is not None:
        debug_outputs["soft_window_closed_kji"] = sitk.GetArrayFromImage(soft_img).astype(np.uint8)

    inside = _largest_nonborder_component_closest_to_center(soft_img)
    if inside is None:
        z = np.zeros(np.asarray(arr_kji).shape, dtype=bool)
        _, skull_lcc = _build_skull_mask_lcc(arr_kji, spacing_xyz, skull_threshold_hu, metal_exclude_hu, close_mm)
        skull_arr = sitk.GetArrayFromImage(skull_lcc).astype(bool) if skull_lcc is not None else z
        if debug_outputs is not None:
            debug_outputs["soft_selected_component_kji"] = z.astype(np.uint8)
        return z, skull_arr, "none"
    if debug_outputs is not None:
        debug_outputs["soft_selected_component_kji"] = sitk.GetArrayFromImage(inside).astype(np.uint8)

    inside = sitk.BinaryFillhole(inside)
    inside_pre_erode = inside

    if float(erode_mm) > 1e-6:
        erode_radius_xyz = _radius_xyz(float(erode_mm), spacing_xyz)
        if any(r > 0 for r in erode_radius_xyz):
            inside = sitk.BinaryErode(inside, erode_radius_xyz)

    inside_lcc = largest_component_binary(inside, exclude_border=False)
    if inside_lcc is None and float(erode_mm) > 1e-6:
        # If erosion collapses the soft-tissue core, fall back to pre-erode core.
        inside_lcc = largest_component_binary(inside_pre_erode, exclude_border=False)
    if inside_lcc is None:
        z = np.zeros(np.asarray(arr_kji).shape, dtype=bool)
        _, skull_lcc = _build_skull_mask_lcc(arr_kji, spacing_xyz, skull_threshold_hu, metal_exclude_hu, close_mm)
        skull_arr = sitk.GetArrayFromImage(skull_lcc).astype(bool) if skull_lcc is not None else z
        if debug_outputs is not None:
            debug_outputs["soft_final_inside_kji"] = z.astype(np.uint8)
        return z, skull_arr, "none"

    inside_arr = sitk.GetArrayFromImage(inside_lcc).astype(bool)
    if debug_outputs is not None:
        debug_outputs["soft_final_inside_kji"] = inside_arr.astype(np.uint8)
    _, skull_lcc = _build_skull_mask_lcc(arr_kji, spacing_xyz, skull_threshold_hu, metal_exclude_hu, close_mm)
    skull_arr = sitk.GetArrayFromImage(skull_lcc).astype(bool) if skull_lcc is not None else np.zeros_like(inside_arr)
    return inside_arr, skull_arr, "soft_tissue_core"


def build_inside_skull_mask_kji(
    arr_kji,
    spacing_xyz,
    skull_threshold_hu=300.0,
    metal_exclude_hu=1800.0,
    close_mm=2.0,
    erode_mm=1.0,
    head_threshold_hu=-200.0,
    prefer_2d_fill=True,
    strategy="auto",
    head_mask_aggressive_cleanup=True,
    debug_outputs=None,
):
    """Build intracranial mask and skull mask in KJI index order.

    Parameters
    ----------
    strategy:
      - "auto" / "skull_shell": existing skull-shell based method
      - "soft_tissue_core": central soft-tissue component method

    Returns
    -------
    inside_arr, skull_arr, inside_method
    """
    _require_numpy()
    _require_sitk()

    strategy_norm = str(strategy or "auto").strip().lower()
    if strategy_norm in {"soft", "soft_tissue", "soft_tissue_core"}:
        return _build_inside_soft_tissue_mask_kji(
            arr_kji=arr_kji,
            spacing_xyz=spacing_xyz,
            skull_threshold_hu=skull_threshold_hu,
            metal_exclude_hu=metal_exclude_hu,
            close_mm=close_mm,
            erode_mm=erode_mm,
            head_threshold_hu=head_threshold_hu,
            aggressive_head_cleanup=head_mask_aggressive_cleanup,
            debug_outputs=debug_outputs,
        )

    img = sitk.GetImageFromArray(np.asarray(arr_kji, dtype=np.float32))
    img.SetSpacing((float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])))
    upper = float(metal_exclude_hu)
    lower = float(skull_threshold_hu)
    if upper <= lower + 1.0:
        upper = lower + 1.0
    skull_raw = sitk.BinaryThreshold(
        img,
        lowerThreshold=lower,
        upperThreshold=upper,
        insideValue=1,
        outsideValue=0,
    )

    close_mm = max(0.0, float(close_mm))
    min_inside_vox = max(1000, int(0.0005 * float(np.asarray(arr_kji).size)))

    def _extract_interior_from_skull(skull_img):
        skull_lcc = largest_component_binary(skull_img)
        if skull_lcc is None:
            return None, None
        hull = sitk.BinaryFillhole(skull_lcc)
        interior = sitk.And(hull, sitk.Not(skull_lcc))
        if erode_mm > 1e-6:
            erode_radius_xyz = _radius_xyz(erode_mm, spacing_xyz)
            if any(r > 0 for r in erode_radius_xyz):
                interior = sitk.BinaryErode(interior, erode_radius_xyz)
        interior_lcc = largest_component_binary(interior, exclude_border=True)
        return interior_lcc, skull_lcc

    skull_largest = None
    intracranial_largest = None
    close_candidates_mm = [close_mm, close_mm + 1.5, close_mm + 3.0, close_mm + 5.0]
    for close_try_mm in close_candidates_mm:
        skull_try = skull_raw
        if close_try_mm > 1e-6:
            radius_xyz = _radius_xyz(close_try_mm, spacing_xyz)
            if any(r > 0 for r in radius_xyz):
                skull_try = sitk.BinaryMorphologicalClosing(skull_try, radius_xyz)
        interior_try, skull_try_lcc = _extract_interior_from_skull(skull_try)
        if skull_try_lcc is None:
            continue
        skull_largest = skull_try_lcc
        if interior_try is not None and _inside_voxel_count(interior_try) >= min_inside_vox:
            intracranial_largest = interior_try
            break

    if skull_largest is None:
        z = np.zeros(np.asarray(arr_kji).shape, dtype=bool)
        return z, z, "none"

    erode_mm = max(0.0, float(erode_mm))
    head_largest = sitk.BinaryThreshold(
        img,
        lowerThreshold=float(head_threshold_hu),
        upperThreshold=1e9,
        insideValue=1,
        outsideValue=0,
    )
    head_largest = largest_component_binary(head_largest)
    head_arr = sitk.GetArrayFromImage(head_largest).astype(bool) if head_largest is not None else None

    inside_method = "none"
    if intracranial_largest is not None:
        inside_method = "3d_fill"

    if intracranial_largest is None and bool(prefer_2d_fill):
        skull_arr = sitk.GetArrayFromImage(skull_largest).astype(bool)
        inside_slice_arr = inside_from_skull_slices_kji(skull_arr, head_mask_kji=head_arr)
        if int(np.count_nonzero(inside_slice_arr)) > 1000:
            intracranial_slice = sitk.GetImageFromArray(inside_slice_arr.astype(np.uint8))
            intracranial_slice.CopyInformation(img)
            if erode_mm > 1e-6:
                erode_radius_xyz = _radius_xyz(erode_mm, spacing_xyz)
                if any(r > 0 for r in erode_radius_xyz):
                    intracranial_slice = sitk.BinaryErode(intracranial_slice, erode_radius_xyz)
            intracranial_largest = largest_component_binary(intracranial_slice, exclude_border=True)
            if _inside_voxel_count(intracranial_largest) >= min_inside_vox:
                inside_method = "2d_fill"
            else:
                intracranial_largest = None

    if intracranial_largest is None:
        skull_arr = sitk.GetArrayFromImage(skull_largest).astype(bool)
        inside_slice_arr = inside_from_directional_bounds_kji(
            skull_mask_kji=skull_arr,
            head_mask_kji=head_arr,
        )
        if int(np.count_nonzero(inside_slice_arr)) > 1000:
            intracranial_slice = sitk.GetImageFromArray(inside_slice_arr.astype(np.uint8))
            intracranial_slice.CopyInformation(img)
            if erode_mm > 1e-6:
                erode_radius_xyz = _radius_xyz(erode_mm, spacing_xyz)
                if any(r > 0 for r in erode_radius_xyz):
                    intracranial_slice = sitk.BinaryErode(intracranial_slice, erode_radius_xyz)
            intracranial_largest = largest_component_binary(intracranial_slice, exclude_border=True)
            if _inside_voxel_count(intracranial_largest) < min_inside_vox:
                intracranial_largest = None
            else:
                inside_method = "directional_2d"

    if intracranial_largest is None and not bool(prefer_2d_fill):
        skull_arr = sitk.GetArrayFromImage(skull_largest).astype(bool)
        inside_slice_arr = inside_from_skull_slices_kji(skull_arr, head_mask_kji=head_arr)
        if int(np.count_nonzero(inside_slice_arr)) > 1000:
            intracranial_slice = sitk.GetImageFromArray(inside_slice_arr.astype(np.uint8))
            intracranial_slice.CopyInformation(img)
            if erode_mm > 1e-6:
                erode_radius_xyz = _radius_xyz(erode_mm, spacing_xyz)
                if any(r > 0 for r in erode_radius_xyz):
                    intracranial_slice = sitk.BinaryErode(intracranial_slice, erode_radius_xyz)
            intracranial_largest = largest_component_binary(intracranial_slice, exclude_border=True)
            if _inside_voxel_count(intracranial_largest) < min_inside_vox:
                intracranial_largest = None
            else:
                inside_method = "2d_fill"

    if intracranial_largest is None and head_largest is not None:
        for dilate_try_mm in [1.0, 2.0, 3.0, 4.0, 6.0]:
            dilate_radius_xyz = _radius_xyz(dilate_try_mm, spacing_xyz)
            dilate_radius_xyz = [max(1, r) for r in dilate_radius_xyz]
            skull_dil = sitk.BinaryDilate(skull_largest, dilate_radius_xyz)
            intracranial2 = sitk.And(head_largest, sitk.Not(skull_dil))
            if erode_mm > 1e-6:
                erode_radius_xyz = _radius_xyz(erode_mm, spacing_xyz)
                if any(r > 0 for r in erode_radius_xyz):
                    intracranial2 = sitk.BinaryErode(intracranial2, erode_radius_xyz)
            intracranial_largest = largest_component_binary(intracranial2, exclude_border=True)
            if _inside_voxel_count(intracranial_largest) >= min_inside_vox:
                inside_method = "head_minus_skull"
                break
            intracranial_largest = None

    if intracranial_largest is None:
        z = np.zeros(np.asarray(arr_kji).shape, dtype=bool)
        skull_arr = sitk.GetArrayFromImage(skull_largest).astype(bool)
        return z, skull_arr, "none"

    inside_arr = sitk.GetArrayFromImage(intracranial_largest).astype(bool)
    skull_arr = sitk.GetArrayFromImage(skull_largest).astype(bool)
    return inside_arr, skull_arr, inside_method


def build_preview_masks(
    arr_kji,
    spacing_xyz,
    threshold,
    use_head_mask=False,
    build_head_mask=True,
    head_mask_threshold_hu=-300.0,
    head_mask_aggressive_cleanup=True,
    head_mask_close_mm=2.0,
    min_metal_depth_mm=0.0,
    max_metal_depth_mm=999.0,
    debug_soft_outputs=False,
):
    """Build raw preview masks and gated threshold points.

    Returns a dictionary with mask arrays and index-space kept points:
    - ``metal_mask_kji`` (uint8)
    - ``head_mask_kji`` (effective gating mask, uint8 or None)
    - ``skull_mask_kji`` (always None in head-distance mode)
    - ``inside_skull_mask_kji`` (always None in head-distance mode)
    - ``in_mask_ijk_kji`` (N,3 int array in k,j,i)
    """
    _require_numpy()
    _require_sitk()

    arr = np.asarray(arr_kji)
    metal_mask = np.asarray(arr >= float(threshold), dtype=bool)
    candidate_count = int(np.count_nonzero(metal_mask))

    gating_mask = None
    skull_mask = None
    inside_mask = None
    inside_method = "none"
    gating_mask_type = "none"

    debug_outputs = {} if bool(debug_soft_outputs) else None
    metal_in_head_count = candidate_count
    depth_kept_count = candidate_count
    dist_head = None
    metal_depth_all_mm = np.empty((0,), dtype=np.float32)
    metal_depth_values_mm = np.empty((0,), dtype=np.float32)
    if bool(use_head_mask) or bool(build_head_mask):
        # New default path: gate by head mask and head signed distance depth.
        head_mask = build_head_mask_kji(
            arr_kji=arr,
            spacing_xyz=spacing_xyz,
            threshold_hu=head_mask_threshold_hu,
            close_mm=head_mask_close_mm,
            aggressive_cleanup=head_mask_aggressive_cleanup,
        )
        gating_mask = np.asarray(head_mask, dtype=bool)
        inside_method = "head_distance"
        gating_mask_type = "head_distance"

        head_img = sitk.GetImageFromArray(gating_mask.astype(np.uint8))
        head_img.SetSpacing((float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])))
        dist_img = sitk.SignedMaurerDistanceMap(
            head_img,
            insideIsPositive=True,
            squaredDistance=False,
            useImageSpacing=True,
        )
        dist_head = sitk.GetArrayFromImage(dist_img).astype(np.float32)
        if debug_outputs is not None:
            debug_outputs["head_mask_kji"] = gating_mask.astype(np.uint8)

    in_mask_ijk_kji = np.empty((0, 3), dtype=int)
    head_mask_kept_count = candidate_count
    if candidate_count > 0:
        ijk_kji = np.argwhere(metal_mask)
        if gating_mask is not None:
            keep = gating_mask[ijk_kji[:, 0], ijk_kji[:, 1], ijk_kji[:, 2]]
            ijk_kji = ijk_kji[keep]
            metal_in_head_count = int(ijk_kji.shape[0])
            if dist_head is not None and metal_in_head_count > 0:
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
            depth_kept_count = int(ijk_kji.shape[0])
        in_mask_ijk_kji = np.asarray(ijk_kji, dtype=int)
        head_mask_kept_count = int(in_mask_ijk_kji.shape[0])

    out = {
        "candidate_count": candidate_count,
        "head_mask_kept_count": int(head_mask_kept_count),
        "metal_in_head_count": int(metal_in_head_count),
        "depth_kept_count": int(depth_kept_count),
        "gating_mask_type": gating_mask_type,
        "inside_method": inside_method,
        "metal_mask_kji": metal_mask.astype(np.uint8),
        "head_mask_kji": gating_mask.astype(np.uint8) if gating_mask is not None else None,
        "skull_mask_kji": np.asarray(skull_mask, dtype=np.uint8) if skull_mask is not None else None,
        "inside_skull_mask_kji": np.asarray(inside_mask, dtype=np.uint8) if inside_mask is not None else None,
        "in_mask_ijk_kji": in_mask_ijk_kji,
        "metal_depth_all_mm": metal_depth_all_mm,
        "metal_depth_values_mm": metal_depth_values_mm,
    }
    if debug_outputs:
        out["debug_masks_kji"] = debug_outputs
    return out
