"""Persistence-aware blob support detector.

This pipeline tracks connected components across a small threshold schedule,
scores stable elongated cores by persistence and instability, then fits shafts
from those high-confidence supports.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from shank_core.blob_candidates import extract_blob_candidates
from shank_core.masking import build_preview_masks

from ..artifacts import add_artifact, write_standard_artifacts
from ..contracts import DetectionContext, DetectionResult
from .base import BaseDetectionPipeline


def _normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-9:
        return np.asarray([0.0, 0.0, 1.0], dtype=float)
    return arr / norm


def _weighted_pca_line(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = np.asarray(weights, dtype=float).reshape(-1)
    w = np.maximum(w, 0.0)
    s = float(np.sum(w))
    if s <= 1e-9:
        raise ValueError("zero weights")
    wn = w / s
    center = np.sum(points * wn[:, None], axis=0)
    x = points - center.reshape(1, 3)
    cov = (x.T * wn.reshape(1, -1)) @ x
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)
    evals = np.asarray(evals[order], dtype=float)
    axis = _normalize(evecs[:, order[-1]])
    return center.astype(float), axis.astype(float), evals.astype(float)


def _depth_at_ras_mm(point_ras: np.ndarray, head_depth_kji: np.ndarray | None, ras_to_ijk_fn) -> float | None:
    if head_depth_kji is None or ras_to_ijk_fn is None:
        return None
    ijk = ras_to_ijk_fn(point_ras)
    i = int(round(float(ijk[0])))
    j = int(round(float(ijk[1])))
    k = int(round(float(ijk[2])))
    if k < 0 or j < 0 or i < 0:
        return None
    if k >= head_depth_kji.shape[0] or j >= head_depth_kji.shape[1] or i >= head_depth_kji.shape[2]:
        return None
    val = float(head_depth_kji[k, j, i])
    return val if np.isfinite(val) else None


def _line_distances(points: np.ndarray, p0: np.ndarray, direction_unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rel = points - p0.reshape(1, 3)
    t = rel @ direction_unit.reshape(3)
    closest = p0.reshape(1, 3) + np.outer(t, direction_unit.reshape(3))
    return np.linalg.norm(points - closest, axis=1), t


def _projected_endpoints(points: np.ndarray, center: np.ndarray, axis: np.ndarray, lo_q: float, hi_q: float) -> tuple[np.ndarray, np.ndarray]:
    proj = (points - center.reshape(1, 3)) @ axis.reshape(3)
    lo = float(np.quantile(proj, lo_q))
    hi = float(np.quantile(proj, hi_q))
    start = center + axis * lo
    end = center + axis * hi
    return start.astype(float), end.astype(float)


def _orient_shallow_to_deep(start: np.ndarray, end: np.ndarray, depth_kji: np.ndarray | None, ras_to_ijk_fn, center_ras: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d0 = _depth_at_ras_mm(start, depth_kji, ras_to_ijk_fn)
    d1 = _depth_at_ras_mm(end, depth_kji, ras_to_ijk_fn)
    if d0 is not None and d1 is not None and abs(float(d0) - float(d1)) > 1e-3:
        return (start, end) if float(d0) <= float(d1) else (end, start)
    c0 = float(np.linalg.norm(start - center_ras))
    c1 = float(np.linalg.norm(end - center_ras))
    return (start, end) if c0 >= c1 else (end, start)


def _segment_inside_fraction(start_ras: np.ndarray, end_ras: np.ndarray, mask_kji: np.ndarray | None, ras_to_ijk_fn, step_mm: float = 1.0) -> float:
    if mask_kji is None or ras_to_ijk_fn is None:
        return 1.0
    seg = end_ras - start_ras
    length = float(np.linalg.norm(seg))
    if length <= 1e-9:
        return 0.0
    n = max(2, int(math.ceil(length / max(1e-3, float(step_mm)))) + 1)
    ts = np.linspace(0.0, 1.0, n)
    dims = mask_kji.shape
    inside = 0
    for t in ts:
        p = start_ras + t * seg
        ijk = ras_to_ijk_fn(p)
        i = int(round(float(ijk[0])))
        j = int(round(float(ijk[1])))
        k = int(round(float(ijk[2])))
        if 0 <= k < dims[0] and 0 <= j < dims[1] and 0 <= i < dims[2] and bool(mask_kji[k, j, i]):
            inside += 1
    return float(inside) / float(max(1, n))


def _line_distance(p0: np.ndarray, d0: np.ndarray, p1: np.ndarray, d1: np.ndarray) -> float:
    u = _normalize(d0)
    v = _normalize(d1)
    w0 = p0 - p1
    c = np.cross(u, v)
    cn = float(np.linalg.norm(c))
    if cn <= 1e-6:
        return float(np.linalg.norm(np.cross(w0, u)))
    return float(abs(np.dot(w0, c)) / cn)


def _overlap_edges(labels_high: np.ndarray, labels_low: np.ndarray) -> dict[int, dict[int, int]]:
    mask = (labels_high > 0) & (labels_low > 0)
    out: dict[int, dict[int, int]] = {}
    if not np.any(mask):
        return out
    pairs = np.stack([labels_high[mask].astype(int), labels_low[mask].astype(int)], axis=1)
    for parent_id, child_id in pairs.tolist():
        out.setdefault(int(parent_id), {})
        out[int(parent_id)][int(child_id)] = int(out[int(parent_id)].get(int(child_id), 0)) + 1
    return out


class BlobPersistenceV1Pipeline(BaseDetectionPipeline):
    pipeline_id = "blob_persistence_v1"
    display_name = "Blob Persistence v1"
    pipeline_version = "1.0.0"

    def _build_gating(self, ctx: DetectionContext, cfg: dict[str, Any]) -> dict[str, Any]:
        arr_kji = np.asarray(ctx["arr_kji"])
        spacing_xyz = tuple(ctx.get("spacing_xyz") or (1.0, 1.0, 1.0))
        threshold = float(cfg.get("threshold_schedule_hu", [2600.0, 2200.0, 1800.0, 1500.0, 1200.0])[2])
        preview = build_preview_masks(
            arr_kji=arr_kji,
            spacing_xyz=spacing_xyz,
            threshold=threshold,
            use_head_mask=bool(cfg.get("use_head_mask", True)),
            build_head_mask=bool(cfg.get("build_head_mask", True)),
            head_mask_threshold_hu=float(cfg.get("head_mask_threshold_hu", -500.0)),
            head_mask_aggressive_cleanup=bool(cfg.get("head_mask_aggressive_cleanup", True)),
            head_mask_close_mm=float(cfg.get("head_mask_close_mm", 2.0)),
            head_mask_method=str(cfg.get("head_mask_method", "outside_air")),
            head_mask_metal_dilate_mm=float(cfg.get("head_mask_metal_dilate_mm", 1.0)),
            head_gate_erode_vox=int(cfg.get("head_gate_erode_vox", 1)),
            head_gate_dilate_vox=int(cfg.get("head_gate_dilate_vox", 1)),
            head_gate_margin_mm=float(cfg.get("head_gate_margin_mm", 0.0)),
            min_metal_depth_mm=float(cfg.get("min_metal_depth_mm", 5.0)),
            max_metal_depth_mm=float(cfg.get("max_metal_depth_mm", 220.0)),
            include_debug_masks=False,
        )
        return {
            "gating_mask_kji": np.asarray(preview.get("gating_mask_kji"), dtype=bool),
            "depth_map_kji": np.asarray(preview.get("head_distance_map_kji"), dtype=np.float32) if preview.get("head_distance_map_kji") is not None else None,
        }

    def _extract_levels(self, ctx: DetectionContext, gating: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
        arr_kji = np.asarray(ctx["arr_kji"])
        gating_mask = np.asarray(gating["gating_mask_kji"], dtype=bool)
        depth_map = gating.get("depth_map_kji")
        thresholds = [float(v) for v in list(cfg.get("threshold_schedule_hu", [2600.0, 2200.0, 1800.0, 1500.0, 1200.0]))]
        out = []
        for thr in thresholds:
            metal = np.asarray(arr_kji >= float(thr), dtype=bool)
            in_head = np.logical_and(metal, gating_mask)
            blobs = extract_blob_candidates(
                in_head.astype(np.uint8),
                arr_kji=arr_kji,
                depth_map_kji=depth_map,
                ijk_kji_to_ras_fn=ctx["ijk_kji_to_ras_fn"],
                fully_connected=True,
            )
            out.append(
                {
                    "threshold_hu": float(thr),
                    "labels_kji": np.asarray(blobs["labels_kji"], dtype=np.int32),
                    "blobs": list(blobs.get("blobs") or []),
                }
            )
        return out

    def _build_persistent_support(self, levels: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
        if not levels:
            return []
        min_persistence = int(cfg.get("persistence_min_levels", 3))
        max_axis_change = float(cfg.get("persistence_max_mean_axis_change_deg", 12.0))
        max_diameter_growth = float(cfg.get("persistence_max_diameter_growth_mm", 4.0))
        min_top_length = float(cfg.get("persistence_min_top_length_mm", 2.0))
        supports: list[dict[str, Any]] = []
        next_maps = [_overlap_edges(levels[i]["labels_kji"], levels[i + 1]["labels_kji"]) for i in range(len(levels) - 1)]
        top = levels[0]
        top_map = {int(b["blob_id"]): b for b in top["blobs"]}
        for blob in top["blobs"]:
            if float(blob.get("length_mm") or 0.0) < min_top_length:
                continue
            current = int(blob["blob_id"])
            persistence = 1
            axis_changes = []
            lengths = [float(blob.get("length_mm") or 0.0)]
            diameters = [float(blob.get("diameter_mm") or 0.0)]
            merge_penalty = 0.0
            axis_prev = _normalize(np.asarray(blob.get("pca_axis_ras") or [0.0, 0.0, 1.0], dtype=float))
            current_blob = blob
            for idx, p2c in enumerate(next_maps):
                children = p2c.get(int(current), {})
                if not children:
                    break
                persistence += 1
                if len(children) > 1:
                    merge_penalty += 0.5 * float(len(children) - 1)
                dom_child = max(children.items(), key=lambda kv: int(kv[1]))[0]
                child_blob = next((b for b in levels[idx + 1]["blobs"] if int(b["blob_id"]) == int(dom_child)), None)
                if child_blob is None:
                    break
                axis_child = _normalize(np.asarray(child_blob.get("pca_axis_ras") or [0.0, 0.0, 1.0], dtype=float))
                axis_changes.append(float(np.degrees(np.arccos(np.clip(abs(float(np.dot(axis_prev, axis_child))), 0.0, 1.0)))))
                lengths.append(float(child_blob.get("length_mm") or 0.0))
                diameters.append(float(child_blob.get("diameter_mm") or 0.0))
                axis_prev = axis_child
                current = int(dom_child)
                current_blob = child_blob
            mean_axis_change = float(np.mean(axis_changes) if axis_changes else 0.0)
            diameter_growth = float(diameters[-1] - diameters[0]) if diameters else 0.0
            length_growth = float(lengths[-1] - lengths[0]) if lengths else 0.0
            if persistence < min_persistence:
                continue
            if mean_axis_change > max_axis_change:
                continue
            if diameter_growth > max_diameter_growth:
                continue
            support_weight = (
                float(persistence)
                * float(max(0.5, current_blob.get("length_mm") or 0.0))
                * float(max(1.0, math.sqrt(float(current_blob.get("voxel_count") or 1.0))))
                / (1.0 + merge_penalty)
            )
            supports.append(
                {
                    "point_ras": np.asarray(current_blob.get("centroid_ras") or [0.0, 0.0, 0.0], dtype=float),
                    "axis_ras": _normalize(np.asarray(current_blob.get("pca_axis_ras") or [0.0, 0.0, 1.0], dtype=float)),
                    "support_weight": float(support_weight),
                    "persistence_levels": int(persistence),
                    "mean_axis_change_deg": float(mean_axis_change),
                    "length_growth_mm": float(length_growth),
                    "diameter_growth_mm": float(diameter_growth),
                    "voxel_count": int(current_blob.get("voxel_count") or 0),
                    "source_blob_id": int(current_blob.get("blob_id") or current),
                }
            )
        return supports

    def _seed_lines(self, supports: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
        if not supports:
            return []
        ordered = sorted(supports, key=lambda s: float(s["support_weight"]), reverse=True)
        seeds = []
        for s in ordered:
            seeds.append({"point": np.asarray(s["point_ras"], dtype=float), "axis": _normalize(np.asarray(s["axis_ras"], dtype=float)), "score": float(s["support_weight"])})
        kept: list[dict[str, Any]] = []
        for seed in seeds:
            dup = False
            for prev in kept:
                angle = float(np.degrees(np.arccos(np.clip(abs(float(np.dot(_normalize(seed["axis"]), _normalize(prev["axis"])))), 0.0, 1.0))))
                line_dist = _line_distance(np.asarray(seed["point"], dtype=float), np.asarray(seed["axis"], dtype=float), np.asarray(prev["point"], dtype=float), np.asarray(prev["axis"], dtype=float))
                if angle <= float(cfg.get("seed_nms_angle_deg", 8.0)) and line_dist <= float(cfg.get("seed_nms_line_distance_mm", 2.5)):
                    dup = True
                    break
            if not dup:
                kept.append(seed)
            if len(kept) >= int(cfg.get("max_lines", 30)) * 4:
                break
        return kept

    def _fit_lines(self, seeds: list[dict[str, Any]], supports: list[dict[str, Any]], gating: dict[str, Any], ctx: DetectionContext, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        if not seeds or not supports:
            return []
        points = np.asarray([s["point_ras"] for s in supports], dtype=float).reshape(-1, 3)
        axes = np.asarray([s["axis_ras"] for s in supports], dtype=float).reshape(-1, 3)
        weights = np.asarray([float(s["support_weight"]) for s in supports], dtype=float).reshape(-1)
        active = np.ones((points.shape[0],), dtype=bool)
        lines: list[dict[str, Any]] = []
        max_lines = int(cfg.get("max_lines", 30))
        radius = float(cfg.get("support_fit_radius_mm", 2.0))
        min_axis_agree = float(cfg.get("support_min_axis_agree", 0.7))
        for _ in range(max_lines):
            best = None
            best_score = -float("inf")
            active_idx = np.where(active)[0]
            if active_idx.size < int(cfg.get("min_inliers", 6)):
                break
            pts = points[active_idx]
            pts_axes = axes[active_idx]
            pts_w = weights[active_idx]
            for seed in seeds:
                p0 = np.asarray(seed["point"], dtype=float)
                axis = _normalize(np.asarray(seed["axis"], dtype=float))
                dist, _proj = _line_distances(pts, p0, axis)
                axis_ok = np.abs(np.sum(pts_axes * axis.reshape(1, 3), axis=1)) >= min_axis_agree
                inlier = (dist <= radius) & axis_ok
                if int(np.count_nonzero(inlier)) < int(cfg.get("min_inliers", 6)):
                    continue
                fit_points = pts[inlier]
                fit_w = pts_w[inlier]
                ctr, axis, _ = _weighted_pca_line(fit_points, fit_w)
                start, end = _projected_endpoints(fit_points, ctr, axis, float(cfg.get("endpoint_lo_quantile", 0.05)), float(cfg.get("endpoint_hi_quantile", 0.95)))
                center_ras = np.asarray(ctx.get("center_ras") if ctx.get("center_ras") is not None else [0.0, 0.0, 0.0], dtype=float)
                start, end = _orient_shallow_to_deep(start, end, gating.get("depth_map_kji"), ctx.get("ras_to_ijk_fn"), center_ras)
                axis = _normalize(end - start)
                residuals, proj = _line_distances(fit_points, ctr, axis)
                rms = float(np.sqrt(np.average(residuals ** 2, weights=np.maximum(fit_w, 1e-6))))
                length_mm = float(np.linalg.norm(end - start))
                if length_mm < float(cfg.get("min_length_mm", 20.0)):
                    continue
                entry_depth = _depth_at_ras_mm(start, gating.get("depth_map_kji"), ctx.get("ras_to_ijk_fn"))
                target_depth = _depth_at_ras_mm(end, gating.get("depth_map_kji"), ctx.get("ras_to_ijk_fn"))
                depth_span = 0.0 if entry_depth is None or target_depth is None else float(target_depth - entry_depth)
                if depth_span < float(cfg.get("min_depth_span_mm", 10.0)):
                    continue
                inside_fraction = _segment_inside_fraction(start, end, gating.get("gating_mask_kji"), ctx.get("ras_to_ijk_fn"), step_mm=1.0)
                score = float(np.sum(fit_w)) + 0.04 * length_mm + 2.0 * inside_fraction + 0.02 * depth_span - 1.5 * rms
                if score > best_score:
                    best_score = score
                    best = {
                        "name": f"P{len(lines)+1:03d}",
                        "start_ras": [float(v) for v in start],
                        "end_ras": [float(v) for v in end],
                        "length_mm": float(length_mm),
                        "support_weight": float(np.sum(fit_w)),
                        "inlier_count": int(np.count_nonzero(inlier)),
                        "inside_fraction": float(inside_fraction),
                        "depth_span_mm": float(depth_span),
                        "entry_depth_mm": None if entry_depth is None else float(entry_depth),
                        "target_depth_mm": None if target_depth is None else float(target_depth),
                        "rms_mm": float(rms),
                        "selection_score": float(score),
                        "assigned_global_indices": active_idx[inlier].tolist(),
                    }
            if best is None:
                break
            lines.append(best)
            remove_idx = np.asarray(best.get("assigned_global_indices") or [], dtype=int).reshape(-1)
            if remove_idx.size:
                active[remove_idx] = False
        return lines

    def _select_final(self, lines: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
        ordered = sorted(lines, key=lambda ln: float(ln.get("selection_score", 0.0)), reverse=True)
        kept = []
        target_count = cfg.get("selection_target_count")
        try:
            target_count = None if target_count in (None, "", 0) else int(target_count)
        except Exception:
            target_count = None
        for line in ordered:
            s0 = np.asarray(line["start_ras"], dtype=float)
            e0 = np.asarray(line["end_ras"], dtype=float)
            d0 = _normalize(e0 - s0)
            dup = False
            for prev in kept:
                s1 = np.asarray(prev["start_ras"], dtype=float)
                e1 = np.asarray(prev["end_ras"], dtype=float)
                d1 = _normalize(e1 - s1)
                angle = float(np.degrees(np.arccos(np.clip(abs(float(np.dot(d0, d1))), 0.0, 1.0))))
                line_dist = _line_distance(0.5 * (s0 + e0), d0, 0.5 * (s1 + e1), d1)
                if angle <= float(cfg.get("selection_nms_angle_deg", 8.0)) and line_dist <= float(cfg.get("selection_nms_line_distance_mm", 2.5)):
                    dup = True
                    break
            if dup:
                continue
            kept.append(line)
            if target_count is not None and len(kept) >= target_count:
                break
        return kept

    def run(self, ctx: DetectionContext) -> DetectionResult:
        t_start = time.perf_counter()
        result = self.make_result(ctx)
        diagnostics = self.diagnostics(result)
        cfg = self._config(ctx)
        try:
            if "arr_kji" not in ctx or "ijk_kji_to_ras_fn" not in ctx:
                result["trajectories"] = []
                result["contacts"] = []
                result["warnings"].append("blob_persistence_v1 missing volume context; returning empty result")
                diagnostics.note("blob_persistence_v1 requires arr_kji and ijk_kji_to_ras_fn for detection")
                return self.finalize(result, diagnostics, t_start)

            gating = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="gating", fn=lambda: self._build_gating(ctx, cfg))
            levels = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="multi_threshold_blobs", fn=lambda: self._extract_levels(ctx, gating, cfg))
            supports = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="persistence_support", fn=lambda: self._build_persistent_support(levels, cfg))
            seeds = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="seed_generation", fn=lambda: self._seed_lines(supports, cfg))
            lines = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="line_fitting", fn=lambda: self._fit_lines(seeds, supports, gating, ctx, cfg))
            selected = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="selection", fn=lambda: self._select_final(lines, cfg))

            result["trajectories"] = [
                {
                    "name": str(line.get("name") or f"P{idx:02d}"),
                    "start_ras": list(line.get("start_ras", [0.0, 0.0, 0.0])),
                    "end_ras": list(line.get("end_ras", [0.0, 0.0, 0.0])),
                    "length_mm": float(line.get("length_mm", 0.0)),
                    "confidence": float(min(1.0, max(0.0, float(line.get("selection_score", 0.0)) / 12.0))),
                    "support_count": int(line.get("inlier_count", 0)),
                    "params": {
                        "rms_mm": float(line.get("rms_mm", 0.0)),
                        "inside_fraction": float(line.get("inside_fraction", 0.0)),
                        "depth_span_mm": float(line.get("depth_span_mm", 0.0)),
                        "entry_depth_mm": line.get("entry_depth_mm"),
                        "target_depth_mm": line.get("target_depth_mm"),
                        "selection_score": float(line.get("selection_score", 0.0)),
                    },
                }
                for idx, line in enumerate(selected, start=1)
            ]
            result["contacts"] = []
            result["warnings"].append("contact_detection_not_implemented")

            diagnostics.set_count("threshold_levels", int(len(levels)))
            diagnostics.set_count("persistent_support_count", int(len(supports)))
            diagnostics.set_count("seed_count", int(len(seeds)))
            diagnostics.set_count("proposal_count", int(len(lines)))
            diagnostics.set_count("final_lines_kept", int(len(selected)))
            diagnostics.note("blob_persistence_v1 fits shafts from persistence-stable blob cores")

            writer = self.get_artifact_writer(ctx, result)
            support_rows = [
                [
                    idx + 1,
                    float(s["point_ras"][0]),
                    float(s["point_ras"][1]),
                    float(s["point_ras"][2]),
                    float(s["axis_ras"][0]),
                    float(s["axis_ras"][1]),
                    float(s["axis_ras"][2]),
                    float(s["support_weight"]),
                    int(s["persistence_levels"]),
                    float(s["mean_axis_change_deg"]),
                    float(s["length_growth_mm"]),
                    float(s["diameter_growth_mm"]),
                ]
                for idx, s in enumerate(supports)
            ]
            support_path = writer.write_csv_rows(
                "persistence_supports.csv",
                [
                    "support_id",
                    "point_x",
                    "point_y",
                    "point_z",
                    "axis_x",
                    "axis_y",
                    "axis_z",
                    "support_weight",
                    "persistence_levels",
                    "mean_axis_change_deg",
                    "length_growth_mm",
                    "diameter_growth_mm",
                ],
                support_rows,
            )
            add_artifact(result["artifacts"], kind="support_csv", path=support_path, description="Persistence-stable supports", stage="persistence_support")
            result["artifacts"].extend(
                write_standard_artifacts(
                    writer,
                    result,
                    blobs=[],
                    pipeline_payload={
                        "pipeline_id": self.pipeline_id,
                        "pipeline_version": self.pipeline_version,
                        "counts": {
                            "levels": int(len(levels)),
                            "supports": int(len(supports)),
                            "seeds": int(len(seeds)),
                            "proposals": int(len(lines)),
                            "selected": int(len(selected)),
                        },
                    },
                )
            )
        except Exception as exc:
            self.fail(ctx=ctx, result=result, diagnostics=diagnostics, stage=str(getattr(exc, "stage", "pipeline")), exc=exc)
        return self.finalize(result, diagnostics, t_start)
