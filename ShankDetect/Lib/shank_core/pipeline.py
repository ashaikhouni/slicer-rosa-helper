"""High-level detection pipeline combining masking and line fitting."""

from __future__ import annotations

import time
import numpy as np

from .detect import detect_from_preview
from .masking import build_preview_masks


def run_detection(
    arr_kji,
    spacing_xyz,
    threshold,
    ijk_kji_to_ras_fn,
    ras_to_ijk_fn,
    center_ras,
    max_points=300000,
    max_lines=30,
    inlier_radius_mm=1.2,
    min_length_mm=20.0,
    min_inliers=250,
    ransac_iterations=240,
    exclude_segments=None,
    exclude_radius_mm=2.0,
    use_head_mask=False,
    build_head_mask=False,
    head_mask_threshold_hu=-300.0,
    head_mask_aggressive_cleanup=True,
    head_mask_close_mm=2.0,
    min_metal_depth_mm=5.0,
    max_metal_depth_mm=220.0,
    models_by_id=None,
    min_model_score=None,
    precomputed_gating_mask_kji=None,
    precomputed_head_distance_map_kji=None,
):
    """Run end-to-end trajectory detection from CT array + geometry callbacks.

    The caller owns coordinate conversion. This keeps the core reusable across:
    - Slicer (MRML volume geometry)
    - standalone CLI (SimpleITK geometry)
    """
    arr = np.asarray(arr_kji)
    t0 = time.perf_counter()

    t_preview0 = time.perf_counter()
    preview = build_preview_masks(
        arr_kji=arr,
        spacing_xyz=spacing_xyz,
        threshold=threshold,
        use_head_mask=use_head_mask,
        build_head_mask=build_head_mask,
        head_mask_threshold_hu=head_mask_threshold_hu,
        head_mask_aggressive_cleanup=head_mask_aggressive_cleanup,
        head_mask_close_mm=head_mask_close_mm,
        min_metal_depth_mm=min_metal_depth_mm,
        max_metal_depth_mm=max_metal_depth_mm,
        precomputed_gating_mask_kji=precomputed_gating_mask_kji,
        precomputed_head_distance_map_kji=precomputed_head_distance_map_kji,
    )
    preview_ms = (time.perf_counter() - t_preview0) * 1000.0

    t_detect0 = time.perf_counter()
    detect = detect_from_preview(
        arr_kji=arr,
        spacing_xyz=spacing_xyz,
        preview=preview,
        ijk_kji_to_ras_fn=ijk_kji_to_ras_fn,
        ras_to_ijk_fn=ras_to_ijk_fn,
        center_ras=center_ras,
        max_points=max_points,
        max_lines=max_lines,
        inlier_radius_mm=inlier_radius_mm,
        min_length_mm=min_length_mm,
        min_inliers=min_inliers,
        ransac_iterations=ransac_iterations,
        exclude_segments=exclude_segments,
        exclude_radius_mm=exclude_radius_mm,
        models_by_id=models_by_id,
        min_model_score=min_model_score,
    )
    detect_ms = (time.perf_counter() - t_detect0) * 1000.0

    return {
        "candidate_count": int(preview.get("candidate_count", 0)),
        "head_mask_kept_count": int(detect.get("head_mask_kept_count", 0)),
        "gating_mask_type": str(detect.get("gating_mask_type", preview.get("gating_mask_type", "none"))),
        "inside_method": str(detect.get("inside_method", preview.get("inside_method", "none"))),
        "metal_in_head_count": int(detect.get("metal_in_head_count", preview.get("metal_in_head_count", 0))),
        "depth_kept_count": int(detect.get("depth_kept_count", preview.get("depth_kept_count", 0))),
        "gap_reject_count": int(detect.get("gap_reject_count", 0)),
        "duplicate_reject_count": int(detect.get("duplicate_reject_count", 0)),
        "metal_mask_kji": preview.get("metal_mask_kji"),
        "gating_mask_kji": preview.get("gating_mask_kji", preview.get("head_mask_kji")),
        "head_mask_kji": preview.get("head_mask_kji"),
        "head_distance_map_kji": preview.get("head_distance_map_kji"),
        "profile_flags": preview.get("profile_flags", {}),
        "metal_depth_all_mm": detect.get("metal_depth_all_mm"),
        "metal_depth_values_mm": detect.get("metal_depth_values_mm"),
        "in_mask_depth_values_mm": detect.get("in_mask_depth_values_mm"),
        "in_mask_points_ras": detect.get("in_mask_points_ras"),
        "profile_ms": {
            "preview_stage": float(preview_ms),
            "detect_stage": float(detect_ms),
            "total": float((time.perf_counter() - t0) * 1000.0),
            "preview": preview.get("profile_ms", {}),
            "detect": detect.get("profile_ms", {}),
        },
        "lines": detect.get("lines", []),
    }
