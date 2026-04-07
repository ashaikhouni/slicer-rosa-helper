"""High-level detection pipeline combining masking and line fitting."""

from __future__ import annotations

import time
from typing import Any, Callable, TypedDict

import numpy as np

from .detect import detect_from_preview
from .masking import build_preview_masks


class DetectionProfileMs(TypedDict, total=False):
    """Profiling metrics in milliseconds for each pipeline stage."""

    preview_stage: float
    detect_stage: float
    total: float
    preview: dict[str, float]
    detect: dict[str, float]


class DetectionResult(TypedDict, total=False):
    """High-level trajectory detection output payload."""

    candidate_count: int
    head_mask_kept_count: int
    gating_mask_type: str
    inside_method: str
    metal_in_head_count: int
    depth_kept_count: int
    gap_reject_count: int
    duplicate_reject_count: int
    start_zone_reject_count: int
    length_reject_count: int
    inlier_reject_count: int
    candidate_points_total: int
    candidate_points_after_mask: int
    candidate_points_after_depth: int
    effective_min_inliers: int
    effective_inlier_radius_mm: float
    blob_count_total: int
    blob_count_kept: int
    blob_reject_small: int
    blob_reject_large: int
    blob_reject_intensity: int
    blob_reject_shape: int
    fit1_lines_proposed: int
    fit2_lines_kept: int
    rescue_lines_kept: int
    final_lines_kept: int
    assigned_points_after_refine: int
    unassigned_points_after_refine: int
    rescued_points: int
    final_unassigned_points: int
    metal_mask_kji: np.ndarray
    gating_mask_kji: np.ndarray
    head_mask_kji: np.ndarray
    distance_surface_mask_kji: np.ndarray
    not_air_mask_kji: np.ndarray
    not_air_eroded_mask_kji: np.ndarray
    head_core_mask_kji: np.ndarray
    metal_gate_mask_kji: np.ndarray
    metal_in_gate_mask_kji: np.ndarray
    depth_window_mask_kji: np.ndarray
    metal_depth_pass_mask_kji: np.ndarray
    head_distance_map_kji: np.ndarray
    in_mask_ijk_kji: np.ndarray
    blob_labelmap_kji: np.ndarray
    blob_centroids_all_ras: np.ndarray
    blob_centroids_kept_ras: np.ndarray
    blob_centroids_rejected_ras: np.ndarray
    profile_flags: dict[str, Any]
    metal_depth_all_mm: np.ndarray
    metal_depth_values_mm: np.ndarray
    in_mask_depth_values_mm: np.ndarray
    in_mask_points_ras: np.ndarray
    profile_ms: DetectionProfileMs
    lines: list[dict[str, Any]]


def run_detection(
    arr_kji: np.ndarray,
    spacing_xyz: list[float] | tuple[float, float, float],
    threshold: float,
    ijk_kji_to_ras_fn: Callable[[np.ndarray | list[float]], np.ndarray | list[float]],
    ras_to_ijk_fn: Callable[[np.ndarray | list[float]], np.ndarray | list[float]],
    center_ras: list[float] | np.ndarray,
    max_points: int = 300000,
    max_lines: int = 30,
    inlier_radius_mm: float = 1.2,
    min_length_mm: float = 20.0,
    min_inliers: int = 250,
    ransac_iterations: int = 240,
    exclude_segments: list[dict[str, Any]] | None = None,
    exclude_radius_mm: float = 2.0,
    use_head_mask: bool = False,
    build_head_mask: bool = False,
    head_mask_threshold_hu: float = -500.0,
    head_mask_aggressive_cleanup: bool = True,
    head_mask_close_mm: float = 2.0,
    head_mask_method: str = "outside_air",
    head_mask_metal_dilate_mm: float = 1.0,
    head_gate_erode_vox: int = 1,
    head_gate_dilate_vox: int = 1,
    head_gate_margin_mm: float = 0.0,
    min_metal_depth_mm: float = 5.0,
    max_metal_depth_mm: float = 220.0,
    start_zone_window_mm: float = 10.0,
    candidate_mode: str = "voxel",
    min_blob_voxels: int = 2,
    max_blob_voxels: int = 1200,
    min_blob_peak_hu: float | None = None,
    max_blob_elongation: float | None = None,
    enable_rescue_pass: bool = True,
    rescue_min_inliers_scale: float = 0.6,
    rescue_max_lines: int = 6,
    apply_start_zone_prior: bool | None = None,
    models_by_id: dict[str, dict[str, Any]] | None = None,
    min_model_score: float | None = None,
    precomputed_gating_mask_kji: np.ndarray | None = None,
    precomputed_head_distance_map_kji: np.ndarray | None = None,
    include_debug_masks: bool = False,
) -> DetectionResult:
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
        head_mask_method=head_mask_method,
        head_mask_metal_dilate_mm=head_mask_metal_dilate_mm,
        head_gate_erode_vox=head_gate_erode_vox,
        head_gate_dilate_vox=head_gate_dilate_vox,
        head_gate_margin_mm=head_gate_margin_mm,
        min_metal_depth_mm=min_metal_depth_mm,
        max_metal_depth_mm=max_metal_depth_mm,
        precomputed_gating_mask_kji=precomputed_gating_mask_kji,
        precomputed_head_distance_map_kji=precomputed_head_distance_map_kji,
        include_debug_masks=bool(include_debug_masks),
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
        min_metal_depth_mm=min_metal_depth_mm,
        start_zone_window_mm=start_zone_window_mm,
        candidate_mode=candidate_mode,
        min_blob_voxels=min_blob_voxels,
        max_blob_voxels=max_blob_voxels,
        min_blob_peak_hu=min_blob_peak_hu,
        max_blob_elongation=max_blob_elongation,
        enable_rescue_pass=enable_rescue_pass,
        rescue_min_inliers_scale=rescue_min_inliers_scale,
        rescue_max_lines=rescue_max_lines,
        apply_start_zone_prior=apply_start_zone_prior,
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
        "start_zone_reject_count": int(detect.get("start_zone_reject_count", 0)),
        "duplicate_reject_count": int(detect.get("duplicate_reject_count", 0)),
        "length_reject_count": int(detect.get("length_reject_count", 0)),
        "inlier_reject_count": int(detect.get("inlier_reject_count", 0)),
        "candidate_points_total": int(detect.get("candidate_points_total", preview.get("candidate_count", 0))),
        "candidate_points_after_mask": int(detect.get("candidate_points_after_mask", preview.get("metal_in_head_count", 0))),
        "candidate_points_after_depth": int(detect.get("candidate_points_after_depth", preview.get("depth_kept_count", 0))),
        "effective_min_inliers": int(detect.get("effective_min_inliers", min_inliers)),
        "effective_inlier_radius_mm": float(detect.get("effective_inlier_radius_mm", inlier_radius_mm)),
        "blob_count_total": int(detect.get("blob_count_total", 0)),
        "blob_count_kept": int(detect.get("blob_count_kept", 0)),
        "blob_reject_small": int(detect.get("blob_reject_small", 0)),
        "blob_reject_large": int(detect.get("blob_reject_large", 0)),
        "blob_reject_intensity": int(detect.get("blob_reject_intensity", 0)),
        "blob_reject_shape": int(detect.get("blob_reject_shape", 0)),
        "fit1_lines_proposed": int(detect.get("fit1_lines_proposed", 0)),
        "fit2_lines_kept": int(detect.get("fit2_lines_kept", 0)),
        "rescue_lines_kept": int(detect.get("rescue_lines_kept", 0)),
        "final_lines_kept": int(detect.get("final_lines_kept", len(detect.get("lines", [])))),
        "assigned_points_after_refine": int(detect.get("assigned_points_after_refine", 0)),
        "unassigned_points_after_refine": int(detect.get("unassigned_points_after_refine", 0)),
        "rescued_points": int(detect.get("rescued_points", 0)),
        "final_unassigned_points": int(detect.get("final_unassigned_points", 0)),
        "metal_mask_kji": preview.get("metal_mask_kji"),
        "gating_mask_kji": preview.get("gating_mask_kji", preview.get("head_mask_kji")),
        "head_mask_kji": preview.get("head_mask_kji"),
        "distance_surface_mask_kji": preview.get("distance_surface_mask_kji"),
        "not_air_mask_kji": preview.get("not_air_mask_kji"),
        "not_air_eroded_mask_kji": preview.get("not_air_eroded_mask_kji"),
        "head_core_mask_kji": preview.get("head_core_mask_kji"),
        "metal_gate_mask_kji": preview.get("metal_gate_mask_kji"),
        "metal_in_gate_mask_kji": preview.get("metal_in_gate_mask_kji"),
        "depth_window_mask_kji": preview.get("depth_window_mask_kji"),
        "metal_depth_pass_mask_kji": preview.get("metal_depth_pass_mask_kji"),
        "head_distance_map_kji": preview.get("head_distance_map_kji"),
        "in_mask_ijk_kji": preview.get("in_mask_ijk_kji"),
        "blob_labelmap_kji": detect.get("blob_labelmap_kji"),
        "blob_centroids_all_ras": detect.get("blob_centroids_all_ras"),
        "blob_centroids_kept_ras": detect.get("blob_centroids_kept_ras"),
        "blob_centroids_rejected_ras": detect.get("blob_centroids_rejected_ras"),
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
