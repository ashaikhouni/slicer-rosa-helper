"""Boundary adapter from shank_engine pipelines to legacy shank_core detection."""

from __future__ import annotations

from typing import Any

from shank_core.pipeline import run_detection


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)



def run_blob_ransac_v1(ctx: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Run current shank_core detector and normalize adapter payload."""

    arr_kji = ctx["arr_kji"]
    ijk_kji_to_ras_fn = ctx["ijk_kji_to_ras_fn"]
    ras_to_ijk_fn = ctx["ras_to_ijk_fn"]
    spacing_raw = ctx.get("spacing_xyz", None)
    if spacing_raw is None:
        spacing_xyz = (1.0, 1.0, 1.0)
    else:
        spacing_xyz = tuple(spacing_raw)
    center_raw = ctx.get("center_ras", None)
    if center_raw is None:
        center_ras = [0.0, 0.0, 0.0]
    else:
        center_ras = [float(v) for v in center_raw]

    extras = ctx.get("extras", {})
    models_by_id = None
    if isinstance(extras, dict):
        maybe_models = extras.get("models_by_id")
        if isinstance(maybe_models, dict) and maybe_models:
            models_by_id = maybe_models
    if models_by_id is None:
        maybe_cfg_models = config.get("models_by_id", None)
        if isinstance(maybe_cfg_models, dict) and maybe_cfg_models:
            models_by_id = maybe_cfg_models

    raw = run_detection(
        arr_kji=arr_kji,
        spacing_xyz=spacing_xyz,
        threshold=_to_float(config.get("threshold", 1800.0), 1800.0),
        ijk_kji_to_ras_fn=ijk_kji_to_ras_fn,
        ras_to_ijk_fn=ras_to_ijk_fn,
        center_ras=center_ras,
        max_points=_to_int(config.get("max_points", 300000), 300000),
        max_lines=_to_int(config.get("max_lines", 30), 30),
        inlier_radius_mm=_to_float(config.get("inlier_radius_mm", 1.2), 1.2),
        min_length_mm=_to_float(config.get("min_length_mm", 20.0), 20.0),
        min_inliers=_to_int(config.get("min_inliers", 250), 250),
        ransac_iterations=_to_int(config.get("ransac_iterations", 240), 240),
        use_head_mask=bool(config.get("use_head_mask", True)),
        build_head_mask=bool(config.get("build_head_mask", True)),
        head_mask_threshold_hu=_to_float(config.get("head_mask_threshold_hu", -350.0), -350.0),
        head_mask_method=str(config.get("head_mask_method", "not_air_lcc")),
        head_mask_close_mm=_to_float(config.get("head_mask_close_mm", 2.0), 2.0),
        head_mask_aggressive_cleanup=bool(config.get("head_mask_aggressive_cleanup", False)),
        min_metal_depth_mm=_to_float(config.get("min_metal_depth_mm", 5.0), 5.0),
        max_metal_depth_mm=_to_float(config.get("max_metal_depth_mm", 220.0), 220.0),
        start_zone_window_mm=_to_float(config.get("start_zone_window_mm", 10.0), 10.0),
        candidate_mode=str(config.get("candidate_mode", "blob_centroid")),
        min_blob_voxels=_to_int(config.get("min_blob_voxels", 2), 2),
        max_blob_voxels=_to_int(config.get("max_blob_voxels", 1200), 1200),
        min_blob_peak_hu=config.get("min_blob_peak_hu", None),
        enable_rescue_pass=bool(config.get("enable_rescue_pass", True)),
        rescue_min_inliers_scale=_to_float(config.get("rescue_min_inliers_scale", 0.6), 0.6),
        rescue_max_lines=_to_int(config.get("rescue_max_lines", 6), 6),
        models_by_id=models_by_id,
        min_model_score=config.get("min_model_score", None),
    )

    raw_lines = raw.get("lines", None)
    if raw_lines is None:
        lines = []
    else:
        lines = list(raw_lines)
    trajectories: list[dict[str, Any]] = []
    for idx, line in enumerate(lines, start=1):
        name = str(line.get("name") or f"T{idx:02d}")
        trajectories.append(
            {
                "name": name,
                "source": "blob_ransac_v1",
                "start_ras": [float(v) for v in line.get("start_ras", [0.0, 0.0, 0.0])],
                "end_ras": [float(v) for v in line.get("end_ras", [0.0, 0.0, 0.0])],
                "length_mm": float(line.get("length_mm", 0.0)),
                "model_kind": "line",
                "confidence": float(max(0.0, min(1.0, line.get("inside_fraction", 0.0)))),
                "support_count": int(line.get("inlier_count", 0)),
                "support_mass": float(line.get("support_weight", line.get("inlier_count", 0.0))),
            }
        )

    blob_rows: list[dict[str, Any]] = []
    all_ras = raw.get("blob_centroids_all_ras")
    if all_ras is not None:
        try:
            for i, xyz in enumerate(all_ras):
                blob_rows.append(
                    {
                        "blob_index": int(i),
                        "x": float(xyz[0]),
                        "y": float(xyz[1]),
                        "z": float(xyz[2]),
                    }
                )
        except Exception:
            blob_rows = []

    raw_profile = raw.get("profile_ms", None)
    if isinstance(raw_profile, dict):
        profile = raw_profile
    else:
        profile = {}

    diagnostics = {
        "candidate_points_total": int(raw.get("candidate_points_total", 0)),
        "candidate_points_after_mask": int(raw.get("candidate_points_after_mask", 0)),
        "candidate_points_after_depth": int(raw.get("candidate_points_after_depth", 0)),
        "blob_count_total": int(raw.get("blob_count_total", 0)),
        "blob_count_kept": int(raw.get("blob_count_kept", 0)),
        "gap_reject_count": int(raw.get("gap_reject_count", 0)),
        "duplicate_reject_count": int(raw.get("duplicate_reject_count", 0)),
        "start_zone_reject_count": int(raw.get("start_zone_reject_count", 0)),
        "length_reject_count": int(raw.get("length_reject_count", 0)),
        "inlier_reject_count": int(raw.get("inlier_reject_count", 0)),
        "fit1_lines_proposed": int(raw.get("fit1_lines_proposed", 0)),
        "fit2_lines_kept": int(raw.get("fit2_lines_kept", 0)),
        "rescue_lines_kept": int(raw.get("rescue_lines_kept", 0)),
        "final_lines_kept": int(raw.get("final_lines_kept", len(trajectories))),
        "runtime_ms": float(profile.get("total", 0.0)),
    }

    return {
        "raw": raw,
        "trajectories": trajectories,
        "contacts": [],
        "diagnostics": diagnostics,
        "blobs": blob_rows,
    }
