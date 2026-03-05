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


def _fill_spans_1d_bool(vec):
    """Fill between first/last True index in a 1D boolean vector."""
    idx = np.flatnonzero(vec)
    if idx.size < 2:
        return vec
    out = vec.copy()
    out[int(idx[0]) : int(idx[-1]) + 1] = True
    return out


def axial_row_col_span_envelope_kji(mask_kji):
    """Build per-axial-slice envelope by row+column span fill + largest component.

    This is a fast, low-cost hole-healing surface used only for distance-map
    computation. It stabilizes depth estimates around metal-streak voids without
    expensive 3D morphology.
    """
    _require_numpy()
    mask = np.asarray(mask_kji, dtype=bool)
    out = np.zeros_like(mask, dtype=bool)
    use_sitk = sitk is not None

    for k in range(mask.shape[0]):
        sl = mask[k, :, :]
        if int(np.count_nonzero(sl)) < 8:
            continue

        row_fill = np.zeros_like(sl, dtype=bool)
        for r in range(sl.shape[0]):
            row_fill[r, :] = _fill_spans_1d_bool(sl[r, :])

        col_fill = np.zeros_like(sl, dtype=bool)
        for c in range(sl.shape[1]):
            col_fill[:, c] = _fill_spans_1d_bool(sl[:, c])

        # Conservative combine: require both row and column support.
        env = np.logical_and(row_fill, col_fill)
        # Guard against accidental over-pruning in small/odd slices.
        if int(np.count_nonzero(env)) < int(np.count_nonzero(sl)):
            env = np.logical_or(env, sl)

        if use_sitk:
            sl_img = sitk.GetImageFromArray(env.astype(np.uint8))
            largest = largest_component_binary(sl_img)
            if largest is not None:
                env = sitk.GetArrayFromImage(largest).astype(bool)

        out[k, :, :] = env
    return out


def build_tissue_cut_distance_and_gating_masks_kji(
    arr_kji,
    spacing_xyz,
    threshold_hu: float = -300.0,
    metal_threshold_hu: float = 1800.0,
    metal_dilate_mm: float = 1.0,
    aggressive_cleanup: bool = True,
):
    """Build dual masks for tissue-cut-no-close depth gating.

    Returns
    -------
    (gating_mask_kji, distance_mask_kji)
      - gating_mask_kji: where metal candidates are allowed to exist
        (`LCC(tissue & ~dilate(metal))` with metal re-added inside tissue LCC).
      - distance_mask_kji: wire-bridge tolerant surface for depth computation
        (`fill_holes(gating_mask_kji)`; single fill-hole + single distance map).
    """
    _require_numpy()
    _require_sitk()

    arr = np.asarray(arr_kji, dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])))

    tissue = sitk.BinaryThreshold(
        img,
        lowerThreshold=float(threshold_hu),
        upperThreshold=1e9,
        insideValue=1,
        outsideValue=0,
    )

    # Main tissue component (patient envelope).
    tissue_lcc = largest_component_binary(tissue)
    if tissue_lcc is None:
        empty = np.zeros(arr.shape, dtype=bool)
        return empty, empty
    tissue_lcc_arr = sitk.GetArrayFromImage(tissue_lcc).astype(bool)

    # Depth surface should be robust to external metal bridges.
    metal_raw = sitk.BinaryThreshold(
        img,
        lowerThreshold=float(metal_threshold_hu),
        upperThreshold=1e9,
        insideValue=1,
        outsideValue=0,
    )
    metal = metal_raw
    dilate_mm = max(0.0, float(metal_dilate_mm))
    if dilate_mm > 1e-6:
        rx = max(0, int(round(dilate_mm / max(1e-6, float(spacing_xyz[0])))))
        ry = max(0, int(round(dilate_mm / max(1e-6, float(spacing_xyz[1])))))
        rz = 0  # keep bridge suppression primarily in-plane to avoid over-carving along z
        if rx > 0 or ry > 0:
            metal = sitk.BinaryDilate(metal, [rx, ry, rz])

    tissue_cut = sitk.And(tissue_lcc, sitk.BinaryNot(metal))
    core = largest_component_binary(tissue_cut)
    if core is None:
        gating_mask = tissue_lcc_arr.copy()
    else:
        gating_mask = sitk.GetArrayFromImage(core).astype(bool)

    # Restore true metal voxels inside patient tissue; this prevents contact dropouts.
    metal_arr = sitk.GetArrayFromImage(metal_raw).astype(bool)
    gating_mask = np.logical_or(gating_mask, np.logical_and(metal_arr, tissue_lcc_arr))

    # Single hole-healing step for distance surface.
    # Use tissue LCC (not metal-cut core) so intracranial depths remain stable
    # around contacts while gating still benefits from bridge suppression.
    distance_mask = sitk.GetArrayFromImage(sitk.BinaryFillhole(tissue_lcc, fullyConnected=False)).astype(bool)

    return gating_mask, distance_mask


def build_outside_air_mask_kji(
    arr_kji,
    air_threshold_hu: float = -500.0,
):
    """Estimate outside-air via border-connected components in an air mask."""
    _require_numpy()
    _require_sitk()

    arr = np.asarray(arr_kji, dtype=np.float32)
    air = np.asarray(arr < float(air_threshold_hu), dtype=np.uint8)
    if int(np.count_nonzero(air)) == 0:
        return np.zeros(arr.shape, dtype=bool)

    cc = sitk.GetArrayFromImage(sitk.ConnectedComponent(sitk.GetImageFromArray(air)))
    if cc.size == 0:
        return np.zeros(arr.shape, dtype=bool)

    border_labels = np.concatenate(
        [
            cc[0, :, :].ravel(),
            cc[-1, :, :].ravel(),
            cc[:, 0, :].ravel(),
            cc[:, -1, :].ravel(),
            cc[:, :, 0].ravel(),
            cc[:, :, -1].ravel(),
        ]
    )
    border_labels = np.unique(border_labels[border_labels > 0])
    if border_labels.size == 0:
        return np.zeros(arr.shape, dtype=bool)
    return np.isin(cc, border_labels)


def build_not_air_lcc_gate_kji(
    arr_kji,
    spacing_xyz,
    air_threshold_hu: float = -350.0,
    erode_radius_vox: int = 1,
    dilate_radius_vox: int = 1,
    gate_margin_mm: float = 0.0,
):
    """Build permissive head gate from not-air mask using LCC after light erosion.

    Steps:
    1) not_air = CT >= air_threshold_hu
    2) erode not_air (light) to disconnect table/bridges
    3) keep largest connected component -> head_core
    4) dilate head_core back -> head_gate
    5) compute dist_inside on head_gate and build metal_gate by margin
    """
    _require_numpy()
    _require_sitk()

    arr = np.asarray(arr_kji, dtype=np.float32)
    not_air = np.asarray(arr >= float(air_threshold_hu), dtype=bool)
    if int(np.count_nonzero(not_air)) == 0:
        empty = np.zeros(arr.shape, dtype=bool)
        return {
            "not_air_mask_kji": empty,
            "not_air_eroded_mask_kji": empty,
            "head_core_mask_kji": empty,
            "head_gate_mask_kji": empty,
            "metal_gate_mask_kji": empty,
            "head_distance_map_kji": np.zeros(arr.shape, dtype=np.float32),
        }

    not_air_img = sitk.GetImageFromArray(not_air.astype(np.uint8))
    not_air_img.SetSpacing((float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])))

    erode_r = max(0, int(erode_radius_vox))
    if erode_r > 0:
        eroded_img = sitk.BinaryErode(not_air_img, [erode_r, erode_r, erode_r])
    else:
        eroded_img = not_air_img
    eroded_arr = sitk.GetArrayFromImage(eroded_img).astype(bool)

    core_img = largest_component_binary(eroded_img)
    if core_img is None:
        core_img = largest_component_binary(not_air_img)
    if core_img is None:
        empty = np.zeros(arr.shape, dtype=bool)
        return {
            "not_air_mask_kji": not_air,
            "not_air_eroded_mask_kji": eroded_arr,
            "head_core_mask_kji": empty,
            "head_gate_mask_kji": empty,
            "metal_gate_mask_kji": empty,
            "head_distance_map_kji": np.zeros(arr.shape, dtype=np.float32),
        }

    core_arr = sitk.GetArrayFromImage(core_img).astype(bool)
    dilate_r = max(0, int(dilate_radius_vox))
    if dilate_r > 0:
        gate_img = sitk.BinaryDilate(core_img, [dilate_r, dilate_r, dilate_r])
    else:
        gate_img = core_img
    gate_arr = sitk.GetArrayFromImage(gate_img).astype(bool)

    dist_inside = compute_head_distance_map_kji(gate_arr, spacing_xyz=spacing_xyz)
    margin_mm = max(0.0, float(gate_margin_mm))
    metal_gate = np.logical_and(gate_arr, np.asarray(dist_inside, dtype=float) >= margin_mm)

    return {
        "not_air_mask_kji": not_air,
        "not_air_eroded_mask_kji": eroded_arr,
        "head_core_mask_kji": core_arr,
        "head_gate_mask_kji": gate_arr,
        "metal_gate_mask_kji": metal_gate,
        "head_distance_map_kji": np.asarray(dist_inside, dtype=np.float32),
    }


def build_head_mask_kji(
    arr_kji,
    spacing_xyz,
    threshold_hu: float = -300.0,
    close_mm: float = 2.0,
    aggressive_cleanup: bool = True,
    method: str = "legacy",
    metal_threshold_hu: float = 1800.0,
    metal_dilate_mm: float = 1.0,
):
    """Build largest-component head mask in KJI index order.

    Parameters
    ----------
    method : {"legacy", "tissue_cut", "tissue_cut_noclose", "outside_air", "not_air_lcc"}
        - legacy: threshold + close + LCC + slice-wise cleanup.
        - tissue_cut: threshold tissue, remove dilated metal bridges, 3D close/fill, LCC.
        - tissue_cut_noclose: tissue-cut gating mask without 3D closing.
        - outside_air: invert border-connected air mask (threshold_hu used as air threshold).
        - not_air_lcc: not-air erosion + LCC + dilation gate.
    """
    _require_numpy()
    _require_sitk()

    method = str(method or "legacy").strip().lower()
    arr = np.asarray(arr_kji, dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])))

    if method == "outside_air":
        outside_air = build_outside_air_mask_kji(arr_kji=arr, air_threshold_hu=threshold_hu)
        return np.logical_not(outside_air)

    if method == "not_air_lcc":
        gate_out = build_not_air_lcc_gate_kji(
            arr_kji=arr,
            spacing_xyz=spacing_xyz,
            air_threshold_hu=threshold_hu,
            erode_radius_vox=1,
            dilate_radius_vox=1,
            gate_margin_mm=0.0,
        )
        return np.asarray(gate_out.get("head_gate_mask_kji"), dtype=bool)

    if method == "tissue_cut_noclose":
        gating_mask, _distance_mask = build_tissue_cut_distance_and_gating_masks_kji(
            arr_kji=arr,
            spacing_xyz=spacing_xyz,
            threshold_hu=threshold_hu,
            metal_threshold_hu=metal_threshold_hu,
            metal_dilate_mm=metal_dilate_mm,
            aggressive_cleanup=aggressive_cleanup,
        )
        return gating_mask

    if method == "tissue_cut":
        tissue = sitk.BinaryThreshold(
            img,
            lowerThreshold=float(threshold_hu),
            upperThreshold=1e9,
            insideValue=1,
            outsideValue=0,
        )
        metal = sitk.BinaryThreshold(
            img,
            lowerThreshold=float(metal_threshold_hu),
            upperThreshold=1e9,
            insideValue=1,
            outsideValue=0,
        )
        dilate_mm = max(0.0, float(metal_dilate_mm))
        if dilate_mm > 1e-6:
            radius_xyz = _radius_xyz(dilate_mm, spacing_xyz)
            if any(r > 0 for r in radius_xyz):
                metal = sitk.BinaryDilate(metal, radius_xyz)

        tissue_cut = sitk.And(tissue, sitk.BinaryNot(metal))
        close_mm = max(0.0, float(close_mm))
        if close_mm > 1e-6:
            radius_xyz = _radius_xyz(close_mm, spacing_xyz)
            if any(r > 0 for r in radius_xyz):
                tissue_cut = sitk.BinaryMorphologicalClosing(tissue_cut, radius_xyz)

        healed = sitk.BinaryFillhole(tissue_cut, fullyConnected=False)
        largest = largest_component_binary(healed)
        if largest is None:
            return np.zeros(arr.shape, dtype=bool)
        largest_arr = sitk.GetArrayFromImage(largest).astype(bool)
        if bool(aggressive_cleanup):
            largest_arr = keep_largest_slice_component_kji(largest_arr, axis=0, min_slice_voxels=64)
        return largest_arr

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
        return np.zeros(arr.shape, dtype=bool)

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
    head_mask_threshold_hu=-500.0,
    head_mask_aggressive_cleanup=True,
    head_mask_close_mm=2.0,
    head_mask_method="outside_air",
    head_mask_metal_dilate_mm=1.0,
    head_gate_erode_vox=1,
    head_gate_dilate_vox=1,
    head_gate_margin_mm=0.0,
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
    distance_surface_mask = None
    depth_window_mask = None
    metal_depth_pass_mask = np.zeros(arr.shape, dtype=bool)
    metal_in_gate_mask = np.zeros(arr.shape, dtype=bool)
    head_method = str(head_mask_method or "legacy").strip().lower()
    not_air_mask = None
    not_air_eroded_mask = None
    head_core_mask = None
    metal_gate_mask = None
    metal_depth_all_mm = np.empty((0,), dtype=np.float32)
    metal_depth_values_mm = np.empty((0,), dtype=np.float32)

    if bool(use_head_mask) or bool(build_head_mask):
        t_head0 = time.perf_counter()
        if precomputed_gating_mask_kji is not None:
            gating_mask = np.asarray(precomputed_gating_mask_kji, dtype=bool)
        else:
            if head_method == "tissue_cut_noclose":
                gating_mask, distance_surface_mask = build_tissue_cut_distance_and_gating_masks_kji(
                    arr_kji=arr,
                    spacing_xyz=spacing_xyz,
                    threshold_hu=head_mask_threshold_hu,
                    metal_threshold_hu=threshold,
                    metal_dilate_mm=head_mask_metal_dilate_mm,
                    aggressive_cleanup=head_mask_aggressive_cleanup,
                )
                gating_mask = np.asarray(gating_mask, dtype=bool)
            elif head_method == "outside_air":
                outside_air = build_outside_air_mask_kji(arr_kji=arr, air_threshold_hu=head_mask_threshold_hu)
                gating_mask = np.logical_not(outside_air)
                # Use a fast per-slice envelope only for distance-surface stabilization.
                distance_surface_mask = axial_row_col_span_envelope_kji(gating_mask)
            elif head_method == "not_air_lcc":
                gate_out = build_not_air_lcc_gate_kji(
                    arr_kji=arr,
                    spacing_xyz=spacing_xyz,
                    air_threshold_hu=head_mask_threshold_hu,
                    erode_radius_vox=int(head_gate_erode_vox),
                    dilate_radius_vox=int(head_gate_dilate_vox),
                    gate_margin_mm=float(head_gate_margin_mm),
                )
                not_air_mask = np.asarray(gate_out.get("not_air_mask_kji"), dtype=bool)
                not_air_eroded_mask = np.asarray(gate_out.get("not_air_eroded_mask_kji"), dtype=bool)
                head_core_mask = np.asarray(gate_out.get("head_core_mask_kji"), dtype=bool)
                gating_mask = np.asarray(gate_out.get("head_gate_mask_kji"), dtype=bool)
                metal_gate_mask = np.asarray(gate_out.get("metal_gate_mask_kji"), dtype=bool)
                dist_head = np.asarray(gate_out.get("head_distance_map_kji"), dtype=np.float32)
            else:
                head_mask = build_head_mask_kji(
                    arr_kji=arr,
                    spacing_xyz=spacing_xyz,
                    threshold_hu=head_mask_threshold_hu,
                    close_mm=head_mask_close_mm,
                    aggressive_cleanup=head_mask_aggressive_cleanup,
                    method=head_mask_method,
                    metal_threshold_hu=threshold,
                    metal_dilate_mm=head_mask_metal_dilate_mm,
                )
                gating_mask = np.asarray(head_mask, dtype=bool)
        head_mask_ms = (time.perf_counter() - t_head0) * 1000.0
        inside_method = "head_distance"
        gating_mask_type = "head_distance"

        t_dist0 = time.perf_counter()
        if precomputed_head_distance_map_kji is not None:
            dist_head = np.asarray(precomputed_head_distance_map_kji, dtype=np.float32)
        elif head_method == "not_air_lcc" and dist_head is not None:
            pass
        elif head_method in ("tissue_cut_noclose", "outside_air"):
            if distance_surface_mask is not None:
                dist_head = compute_head_distance_map_kji(distance_surface_mask, spacing_xyz=spacing_xyz)
            else:
                # Defensive fallback (should be rare with cache/precompute path).
                dist_head = compute_head_distance_map_kji(gating_mask, spacing_xyz=spacing_xyz)
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
        gate_for_metal = metal_gate_mask if metal_gate_mask is not None else gating_mask
        if gate_for_metal is not None:
            t_gate0 = time.perf_counter()
            keep = gate_for_metal[ijk_kji[:, 0], ijk_kji[:, 1], ijk_kji[:, 2]]
            ijk_kji = ijk_kji[keep]
            metal_in_head_count = int(ijk_kji.shape[0])
            head_gate_ms = (time.perf_counter() - t_gate0) * 1000.0
            if metal_in_head_count > 0:
                metal_in_gate_mask[ijk_kji[:, 0], ijk_kji[:, 1], ijk_kji[:, 2]] = True
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
                if gating_mask is not None:
                    depth_window_mask = np.logical_and(
                        np.asarray(gating_mask, dtype=bool),
                        np.logical_and(dist_head >= float(min_d), dist_head <= float(max_d)),
                    )
            depth_kept_count = int(ijk_kji.shape[0])
        in_mask_ijk_kji = np.asarray(ijk_kji, dtype=int)
        head_mask_kept_count = int(in_mask_ijk_kji.shape[0])
        if head_mask_kept_count > 0:
            metal_depth_pass_mask[in_mask_ijk_kji[:, 0], in_mask_ijk_kji[:, 1], in_mask_ijk_kji[:, 2]] = True
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
        "distance_surface_mask_kji": (
            np.asarray(distance_surface_mask, dtype=np.uint8) if distance_surface_mask is not None else None
        ),
        "not_air_mask_kji": np.asarray(not_air_mask, dtype=np.uint8) if not_air_mask is not None else None,
        "not_air_eroded_mask_kji": np.asarray(not_air_eroded_mask, dtype=np.uint8) if not_air_eroded_mask is not None else None,
        "head_core_mask_kji": np.asarray(head_core_mask, dtype=np.uint8) if head_core_mask is not None else None,
        "metal_gate_mask_kji": np.asarray(metal_gate_mask, dtype=np.uint8) if metal_gate_mask is not None else None,
        "metal_in_gate_mask_kji": np.asarray(metal_in_gate_mask, dtype=np.uint8),
        "depth_window_mask_kji": np.asarray(depth_window_mask, dtype=np.uint8) if depth_window_mask is not None else None,
        "metal_depth_pass_mask_kji": np.asarray(metal_depth_pass_mask, dtype=np.uint8),
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
            "head_mask_method": str(head_mask_method),
            "head_gate_erode_vox": int(head_gate_erode_vox),
            "head_gate_dilate_vox": int(head_gate_dilate_vox),
            "head_gate_margin_mm": float(head_gate_margin_mm),
        },
    }
