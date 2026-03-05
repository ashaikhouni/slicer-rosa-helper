"""Blob-based EM-like de novo detector pipeline.

This implementation is intentionally lightweight and deterministic. It uses:
- shank_core masking + blob extraction
- weighted line seeding
- soft assignment / weighted PCA refinement
- rule-based model selection

It keeps the same `run(ctx)->DetectionResult` contract and emits a legacy-style
payload in `ctx.extras['legacy_result']` so current Slicer UI can consume masks
and line diagnostics without further changes.
"""

from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import Any

import numpy as np

from shank_core.blob_candidates import extract_blob_candidates, filter_blob_candidates
from shank_core.masking import build_preview_masks

from ..artifacts import add_artifact, write_standard_artifacts
from ..contracts import BlobRecord, ContactRecord, DetectionContext, DetectionResult, ShankModel
from .base import BaseDetectionPipeline


def _line_distances(points: np.ndarray, p0: np.ndarray, direction_unit: np.ndarray) -> np.ndarray:
    rel = points - p0.reshape(1, 3)
    t = rel @ direction_unit.reshape(3)
    closest = p0.reshape(1, 3) + np.outer(t, direction_unit.reshape(3))
    return np.linalg.norm(points - closest, axis=1)


def _weighted_pca_line(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    axis = evecs[:, int(np.argmax(evals))]
    n = float(np.linalg.norm(axis))
    if n <= 1e-9:
        raise ValueError("invalid axis")
    axis = axis / n
    return center.astype(float), axis.astype(float)


def _effective_min_inliers(cfg: dict[str, Any], point_count: int) -> int:
    """Scale voxel-tuned min_inliers down for blob-centroid mode."""
    min_inliers_cfg = int(cfg.get("min_inliers", 250))
    scale = float(cfg.get("em_min_inliers_scale", 0.015))
    floor = int(cfg.get("em_min_inliers_floor", 3))
    cap = int(cfg.get("em_min_inliers_cap", 8))
    max_lines = max(1, int(cfg.get("max_lines", 30)))
    density = int(round(float(point_count) / float(max_lines)))
    scaled = int(round(float(min_inliers_cfg) * scale))
    eff = max(floor, scaled, density)
    return int(max(floor, min(cap, eff)))


def _responsibility_matrix(points: np.ndarray, models: list[ShankModel], sigma_mm: float, point_weights: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0 or len(models) == 0:
        return np.zeros((points.shape[0], len(models)), dtype=float)
    sigma = max(1e-6, float(sigma_mm))
    n = points.shape[0]
    k = len(models)
    out = np.zeros((n, k), dtype=float)
    for j, model in enumerate(models):
        p = np.asarray(model.params.get("point_ras", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
        d = np.asarray(model.params.get("direction_ras", [0.0, 0.0, 1.0]), dtype=float).reshape(3)
        dn = float(np.linalg.norm(d))
        if dn <= 1e-9:
            continue
        d /= dn
        dist = _line_distances(points, p, d)
        out[:, j] = np.exp(-0.5 * (dist / sigma) ** 2)
    out = out * point_weights.reshape(-1, 1)
    # Outlier floor so low-prob points don't destabilize normalization.
    outlier = np.full((n, 1), 0.08, dtype=float)
    z = np.sum(out, axis=1, keepdims=True) + outlier
    z = np.maximum(z, 1e-12)
    return out / z


class _EMGating:
    def compute(self, ctx: DetectionContext, state: dict[str, Any]) -> dict[str, Any]:
        arr_kji = np.asarray(ctx["arr_kji"])
        spacing_xyz = tuple(ctx.get("spacing_xyz") or (1.0, 1.0, 1.0))
        cfg = dict(ctx.get("config") or {})
        return build_preview_masks(
            arr_kji=arr_kji,
            spacing_xyz=spacing_xyz,
            threshold=float(cfg.get("threshold", 1800.0)),
            use_head_mask=bool(cfg.get("use_head_mask", True)),
            build_head_mask=bool(cfg.get("build_head_mask", True)),
            head_mask_threshold_hu=float(cfg.get("head_mask_threshold_hu", -500.0)),
            head_mask_aggressive_cleanup=bool(cfg.get("head_mask_aggressive_cleanup", False)),
            head_mask_close_mm=float(cfg.get("head_mask_close_mm", 2.0)),
            head_mask_method=str(cfg.get("head_mask_method", "outside_air")),
            head_mask_metal_dilate_mm=float(cfg.get("head_mask_metal_dilate_mm", 1.0)),
            head_gate_erode_vox=int(cfg.get("head_gate_erode_vox", 1)),
            head_gate_dilate_vox=int(cfg.get("head_gate_dilate_vox", 1)),
            head_gate_margin_mm=float(cfg.get("head_gate_margin_mm", 0.0)),
            min_metal_depth_mm=float(cfg.get("min_metal_depth_mm", 5.0)),
            max_metal_depth_mm=float(cfg.get("max_metal_depth_mm", 220.0)),
        )


class _EMBlobExtractor:
    def extract(self, ctx: DetectionContext, state: dict[str, Any]) -> list[BlobRecord]:
        preview = dict(state.get("preview") or {})
        arr_kji = np.asarray(ctx["arr_kji"])
        metal_mask = preview.get("metal_depth_pass_mask_kji")
        if metal_mask is None:
            metal_mask = preview.get("metal_mask_kji")
        if metal_mask is None:
            return []
        raw = extract_blob_candidates(
            metal_mask_kji=metal_mask,
            arr_kji=arr_kji,
            depth_map_kji=preview.get("head_distance_map_kji"),
            ijk_kji_to_ras_fn=ctx.get("ijk_kji_to_ras_fn"),
            fully_connected=True,
        )
        cfg = dict(ctx.get("config") or {})
        filt = filter_blob_candidates(
            raw,
            min_blob_voxels=int(cfg.get("min_blob_voxels", 2)),
            max_blob_voxels=int(cfg.get("max_blob_voxels", 1200)),
            min_blob_peak_hu=cfg.get("min_blob_peak_hu", None),
            max_blob_elongation=cfg.get("max_blob_elongation", None),
        )
        state["blob_raw"] = raw
        state["blob_filter"] = filt

        out: list[BlobRecord] = []
        for blob in list(filt.get("kept_blobs") or []):
            c_ras = blob.get("centroid_ras")
            c_kji = blob.get("centroid_kji")
            if c_ras is None or c_kji is None:
                continue
            out.append(
                BlobRecord(
                    blob_id=int(blob.get("blob_id", 0)),
                    centroid_ras=(float(c_ras[0]), float(c_ras[1]), float(c_ras[2])),
                    centroid_kji=(float(c_kji[0]), float(c_kji[1]), float(c_kji[2])),
                    voxel_count=int(blob.get("voxel_count", 0)),
                    peak_hu=float(blob.get("hu_max", 0.0) or 0.0),
                    mean_hu=float(blob.get("hu_mean", 0.0) or 0.0),
                    elongation=float(blob.get("elongation", 1.0) or 1.0),
                    depth_min_mm=float(blob.get("depth_min", 0.0) or 0.0),
                    depth_mean_mm=float(blob.get("depth_mean", 0.0) or 0.0),
                    depth_max_mm=float(blob.get("depth_max", 0.0) or 0.0),
                )
            )
        return out


class _EMBlobScorer:
    def score(self, ctx: DetectionContext, blobs: list[BlobRecord], state: dict[str, Any]) -> list[BlobRecord]:
        cfg = dict(ctx.get("config") or {})
        min_depth = float(cfg.get("min_metal_depth_mm", 5.0))
        scored: list[BlobRecord] = []
        for b in blobs:
            # Soft evidence; blob type is not hard-thresholded here.
            elong = float(max(1.0, b.elongation))
            vox = float(max(1, b.voxel_count))
            depth = float(max(0.0, b.depth_mean_mm))
            seg = min(1.0, max(0.0, (elong - 2.0) / 4.0))
            bead = min(1.0, max(0.0, 1.0 - abs(elong - 1.5) / 2.5))
            depth_good = min(1.0, max(0.0, (depth - min_depth) / 8.0))
            quality = min(1.0, max(0.0, 0.4 * bead + 0.4 * seg + 0.2 * depth_good))
            scores = {
                "segment": float(seg),
                "bead": float(bead),
                "depth": float(depth_good),
                "quality": float(quality),
                "weight": float(max(0.5, min(4.0, math.sqrt(vox) * (0.5 + 0.5 * quality)))),
            }
            scored.append(replace(b, scores=scores))
        return scored


class _EMInitializer:
    def initialize(self, ctx: DetectionContext, blobs: list[BlobRecord], state: dict[str, Any]) -> list[ShankModel]:
        if not blobs:
            return []
        cfg = dict(ctx.get("config") or {})
        points = np.asarray([b.centroid_ras for b in blobs], dtype=float).reshape(-1, 3)
        weights = np.asarray([float(b.scores.get("weight", 1.0)) for b in blobs], dtype=float)
        n = points.shape[0]
        if n < 2:
            return []

        radius = float(cfg.get("inlier_radius_mm", 1.2)) * 1.8
        max_lines = int(cfg.get("max_lines", 30))
        # Blob mode has far fewer points than voxel mode.
        eff_min_inliers = _effective_min_inliers(cfg, point_count=n)
        max_reuse = max(1, int(cfg.get("em_seed_point_reuse", 2)))
        rng = np.random.default_rng(0)
        use_count = np.zeros((n,), dtype=np.int32)
        models: list[ShankModel] = []

        while len(models) < max_lines and int(np.count_nonzero(use_count < max_reuse)) >= eff_min_inliers:
            idx_pool = np.where(use_count < max_reuse)[0]
            if idx_pool.size < eff_min_inliers:
                break
            best = None
            iters = min(200, max(40, int(cfg.get("ransac_iterations", 240))))
            for _ in range(iters):
                i, j = rng.choice(idx_pool, size=2, replace=False)
                p0 = points[i]
                d = points[j] - p0
                dn = float(np.linalg.norm(d))
                if dn <= 1e-9:
                    continue
                d /= dn
                dist = _line_distances(points[idx_pool], p0, d)
                mask_local = dist <= radius
                support = float(np.sum(weights[idx_pool][mask_local]))
                count = int(np.count_nonzero(mask_local))
                if count < eff_min_inliers:
                    continue
                proj = (points[idx_pool][mask_local] - p0.reshape(1, 3)) @ d.reshape(3)
                span = float(np.max(proj) - np.min(proj)) if count > 1 else 0.0
                rms = float(np.mean(dist[mask_local])) if count > 0 else 999.0
                score = support + 0.05 * span - 0.2 * rms
                if best is None or score > best[0]:
                    best = (score, idx_pool[mask_local])
            if best is None:
                break

            inliers = np.asarray(best[1], dtype=int)
            center, axis = _weighted_pca_line(points[inliers], weights[inliers])
            proj = (points[inliers] - center.reshape(1, 3)) @ axis.reshape(3)
            t_min = float(np.min(proj))
            t_max = float(np.max(proj))
            models.append(
                ShankModel(
                    shank_id=f"EM{len(models)+1:02d}",
                    kind="line",
                    params={
                        "point_ras": [float(center[0]), float(center[1]), float(center[2])],
                        "direction_ras": [float(axis[0]), float(axis[1]), float(axis[2])],
                        "t_min": t_min,
                        "t_max": t_max,
                    },
                    support={
                        "seed_count": float(inliers.size),
                        "seed_support": float(np.sum(weights[inliers])),
                    },
                    assigned_blob_ids=tuple(int(blobs[i].blob_id) for i in inliers),
                )
            )
            use_count[inliers] += 1

        state["effective_min_inliers"] = int(eff_min_inliers)
        return models


class _EMRefiner:
    def refine(
        self,
        ctx: DetectionContext,
        blobs: list[BlobRecord],
        models: list[ShankModel],
        state: dict[str, Any],
    ) -> tuple[list[ShankModel], dict[str, Any]]:
        if not blobs or not models:
            return models, {"iterations": 0, "final_objective": 0.0}

        cfg = dict(ctx.get("config") or {})
        points = np.asarray([b.centroid_ras for b in blobs], dtype=float).reshape(-1, 3)
        weights = np.asarray([float(b.scores.get("weight", 1.0)) for b in blobs], dtype=float)
        sigma = float(cfg.get("inlier_radius_mm", 1.2)) * 1.5
        iterations = int(cfg.get("em_iterations", 8))
        objective_trace = []
        current = list(models)

        for _ in range(max(1, iterations)):
            resp = _responsibility_matrix(points, current, sigma_mm=sigma, point_weights=weights)
            if resp.size == 0:
                break
            objective_trace.append(float(np.sum(np.max(resp, axis=1))))
            updated: list[ShankModel] = []
            for j, m in enumerate(current):
                wj = resp[:, j]
                mass = float(np.sum(wj))
                if mass <= 1e-6:
                    updated.append(m)
                    continue
                center, axis = _weighted_pca_line(points, wj)
                proj = (points - center.reshape(1, 3)) @ axis.reshape(3)
                keep = wj >= 0.2 * float(np.max(wj))
                if not np.any(keep):
                    keep = wj > 0.0
                if not np.any(keep):
                    keep = np.ones((points.shape[0],), dtype=bool)
                t_min = float(np.min(proj[keep]))
                t_max = float(np.max(proj[keep]))
                updated.append(
                    ShankModel(
                        shank_id=m.shank_id,
                        kind="line",
                        params={
                            "point_ras": [float(center[0]), float(center[1]), float(center[2])],
                            "direction_ras": [float(axis[0]), float(axis[1]), float(axis[2])],
                            "t_min": t_min,
                            "t_max": t_max,
                        },
                        support={
                            **dict(m.support),
                            "em_mass": mass,
                            "em_span_mm": float(max(0.0, t_max - t_min)),
                        },
                        assigned_blob_ids=m.assigned_blob_ids,
                    )
                )
            current = updated

        # Hard assignment for bookkeeping.
        resp = _responsibility_matrix(points, current, sigma_mm=sigma, point_weights=weights)
        hard = np.argmax(resp, axis=1) if resp.size else np.zeros((points.shape[0],), dtype=int)
        max_resp = np.max(resp, axis=1) if resp.size else np.zeros((points.shape[0],), dtype=float)
        hard_threshold = float(cfg.get("em_hard_assignment_min_prob", 0.20))
        for j, m in enumerate(current):
            ids = []
            p = np.asarray(m.params.get("point_ras", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
            d = np.asarray(m.params.get("direction_ras", [0.0, 0.0, 1.0]), dtype=float).reshape(3)
            dn = float(np.linalg.norm(d))
            if dn > 1e-9:
                d /= dn
                dist = _line_distances(points, p, d)
            else:
                dist = np.full((points.shape[0],), 999.0, dtype=float)
            mask = (hard == j) & (max_resp >= hard_threshold) & (dist <= 2.5 * sigma)
            for i in np.where(mask)[0]:
                ids.append(int(blobs[int(i)].blob_id))
            current[j] = replace(current[j], assigned_blob_ids=tuple(ids))

        payload = {
            "iterations": int(max(1, iterations)),
            "final_objective": float(objective_trace[-1] if objective_trace else 0.0),
            "objective_trace": [float(v) for v in objective_trace],
            "hard_assigned": int(np.count_nonzero(max_resp >= hard_threshold)) if resp.size else 0,
        }
        state["responsibilities"] = resp
        state["hard_assignment"] = hard
        state["hard_assignment_prob"] = max_resp
        return current, payload


class _EMSelector:
    def select(
        self,
        ctx: DetectionContext,
        blobs: list[BlobRecord],
        models: list[ShankModel],
        state: dict[str, Any],
    ) -> list[ShankModel]:
        if not models:
            return []
        cfg = dict(ctx.get("config") or {})
        min_len = float(cfg.get("min_length_mm", 20.0))
        seed_min_inliers = int(state.get("effective_min_inliers", _effective_min_inliers(cfg, point_count=len(blobs))))
        min_inliers = int(cfg.get("em_selector_min_inliers", max(3, seed_min_inliers - 1)))
        min_depth = float(cfg.get("min_metal_depth_mm", 5.0))
        start_window = float(cfg.get("start_zone_window_mm", 10.0))
        overlap_thresh = float(cfg.get("em_duplicate_overlap_threshold", 0.80))
        angle_thresh = float(cfg.get("em_duplicate_angle_threshold_deg", 8.0))
        blob_by_id = {int(b.blob_id): b for b in blobs}

        kept = []
        reject_counts = {
            "start_zone_reject_count": 0,
            "length_reject_count": 0,
            "inlier_reject_count": 0,
        }
        for m in models:
            ids = list(m.assigned_blob_ids)
            count = len(ids)
            span = float(m.support.get("em_span_mm", 0.0))
            if count < min_inliers:
                reject_counts["inlier_reject_count"] += 1
                continue
            if span < min_len:
                reject_counts["length_reject_count"] += 1
                continue
            depths = [float(blob_by_id[i].depth_min_mm) for i in ids if i in blob_by_id]
            if depths:
                depths_arr = np.asarray(depths, dtype=float)
                in_start_zone = np.any((depths_arr >= min_depth) & (depths_arr <= (min_depth + start_window)))
                if not bool(in_start_zone):
                    reject_counts["start_zone_reject_count"] += 1
                    continue
            kept.append(m)

        # Remove near-duplicate models by support overlap.
        points = np.asarray([b.centroid_ras for b in blobs], dtype=float).reshape(-1, 3) if blobs else np.zeros((0, 3), dtype=float)
        final: list[ShankModel] = []
        for m in sorted(kept, key=lambda x: float(x.support.get("em_mass", 0.0)), reverse=True):
            p = np.asarray(m.params.get("point_ras", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
            d = np.asarray(m.params.get("direction_ras", [0.0, 0.0, 1.0]), dtype=float).reshape(3)
            dn = float(np.linalg.norm(d))
            if dn <= 1e-9:
                continue
            d /= dn
            clash = False
            for q in final:
                q0 = np.asarray(q.params.get("point_ras", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
                qd = np.asarray(q.params.get("direction_ras", [0.0, 0.0, 1.0]), dtype=float).reshape(3)
                qn = float(np.linalg.norm(qd))
                if qn <= 1e-9:
                    continue
                qd /= qn
                if points.shape[0] == 0:
                    continue
                dm = _line_distances(points, p, d) <= 1.2 * float(cfg.get("inlier_radius_mm", 1.2))
                dq = _line_distances(points, q0, qd) <= 1.2 * float(cfg.get("inlier_radius_mm", 1.2))
                overlap = float(np.count_nonzero(dm & dq)) / float(max(1, np.count_nonzero(dm)))
                ang = math.degrees(math.acos(max(-1.0, min(1.0, abs(float(np.dot(d, qd)))))))
                if overlap >= overlap_thresh and ang <= angle_thresh:
                    clash = True
                    break
            if not clash:
                final.append(m)

        state["selector_reject_counts"] = reject_counts
        return final


class _EMContactDetector:
    def detect(
        self,
        ctx: DetectionContext,
        blobs: list[BlobRecord],
        models: list[ShankModel],
        state: dict[str, Any],
    ) -> list[ContactRecord]:
        # Contact estimation will be added in a dedicated phase.
        return []


class BlobEMV2Pipeline(BaseDetectionPipeline):
    """Second-generation blob detector with EM-like soft assignment."""

    pipeline_id = "blob_em_v2"
    display_name = "Blob EM v2"
    scaffold = False
    pipeline_version = "0.2.0"

    default_components = {
        "gating": _EMGating(),
        "blob_extractor": _EMBlobExtractor(),
        "blob_scorer": _EMBlobScorer(),
        "initializer": _EMInitializer(),
        "shank_refiner": _EMRefiner(),
        "model_selector": _EMSelector(),
        "contact_detector": _EMContactDetector(),
    }

    def _models_to_trajectories(self, models: list[ShankModel], center_ras: np.ndarray) -> list[dict[str, Any]]:
        trajectories = []
        for idx, m in enumerate(models, start=1):
            p = np.asarray(m.params.get("point_ras", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
            d = np.asarray(m.params.get("direction_ras", [0.0, 0.0, 1.0]), dtype=float).reshape(3)
            dn = float(np.linalg.norm(d))
            if dn <= 1e-9:
                continue
            d /= dn
            t0 = float(m.params.get("t_min", -10.0))
            t1 = float(m.params.get("t_max", 10.0))
            a = p + t0 * d
            b = p + t1 * d
            # Fallback orientation: entry farther from center.
            da = float(np.linalg.norm(a - center_ras))
            db = float(np.linalg.norm(b - center_ras))
            entry, target = (a, b) if da >= db else (b, a)
            length_mm = float(np.linalg.norm(target - entry))
            trajectories.append(
                {
                    "name": str(m.shank_id or f"EM{idx:02d}"),
                    "source": self.pipeline_id,
                    "start_ras": [float(entry[0]), float(entry[1]), float(entry[2])],
                    "end_ras": [float(target[0]), float(target[1]), float(target[2])],
                    "length_mm": length_mm,
                    "model_kind": m.kind,
                    "confidence": float(min(1.0, max(0.0, m.support.get("em_mass", 0.0) / 25.0))),
                    "support_count": int(len(m.assigned_blob_ids)),
                    "support_mass": float(m.support.get("em_mass", len(m.assigned_blob_ids))),
                    "params": dict(m.params),
                }
            )
        return trajectories

    def run(self, ctx: DetectionContext) -> DetectionResult:
        t_start = time.perf_counter()
        result = self.make_result(ctx)
        diagnostics = self.diagnostics(result)
        cfg = self._config(ctx)

        if "arr_kji" not in ctx or "ijk_kji_to_ras_fn" not in ctx:
            result["warnings"].append("blob_em_v2 requires array+geometry context; returning empty result")
            result["meta"].setdefault("extras", {})
            result["meta"]["extras"]["scaffold_mode"] = True
            return self.finalize(result, diagnostics, t_start)

        state: dict[str, Any] = {}
        blobs: list[BlobRecord] = []
        models: list[ShankModel] = []
        contacts: list[ContactRecord] = []

        try:
            gating = self.resolve_component(ctx, "gating")
            extractor = self.resolve_component(ctx, "blob_extractor")
            scorer = self.resolve_component(ctx, "blob_scorer")
            initializer = self.resolve_component(ctx, "initializer")
            refiner = self.resolve_component(ctx, "shank_refiner")
            selector = self.resolve_component(ctx, "model_selector")
            contact_detector = self.resolve_component(ctx, "contact_detector")

            preview = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="gating",
                fn=lambda: gating.compute(ctx, state),
            )
            state["preview"] = preview
            diagnostics.set_count("candidate_points_total", int(preview.get("candidate_count", 0)))
            diagnostics.set_count("candidate_points_after_mask", int(preview.get("metal_in_head_count", 0)))
            diagnostics.set_count("candidate_points_after_depth", int(preview.get("depth_kept_count", 0)))
            diagnostics.set_extra("gating_mask_type", str(preview.get("gating_mask_type", "")))
            diagnostics.set_extra("inside_method", str(preview.get("inside_method", "")))

            blobs = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="blob_extraction",
                fn=lambda: extractor.extract(ctx, state),
            )
            diagnostics.set_count("blob_count_total", int((state.get("blob_raw") or {}).get("blob_count_total", len(blobs))))
            filt = dict(state.get("blob_filter") or {})
            diagnostics.set_count("blob_count_kept", int(filt.get("blob_count_kept", len(blobs))))
            diagnostics.set_count("blob_reject_small", int(filt.get("blob_reject_small", 0)))
            diagnostics.set_count("blob_reject_large", int(filt.get("blob_reject_large", 0)))
            diagnostics.set_count("blob_reject_intensity", int(filt.get("blob_reject_intensity", 0)))
            diagnostics.set_count("blob_reject_shape", int(filt.get("blob_reject_shape", 0)))

            blobs = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="blob_scoring",
                fn=lambda: scorer.score(ctx, blobs, state),
            )

            models = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="seed_initialization",
                fn=lambda: initializer.initialize(ctx, blobs, state),
            )
            diagnostics.set_count("fit1_lines_proposed", len(models))
            diagnostics.set_count("effective_min_inliers", int(state.get("effective_min_inliers", max(4, int(round(float(cfg.get("min_inliers", 250)) * 0.08))))))
            diagnostics.set_count("effective_inlier_radius_mm", int(round(float(cfg.get("inlier_radius_mm", 1.2)) * 1000.0)))

            models, refine_payload = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="em_refinement",
                fn=lambda: refiner.refine(ctx, blobs, models, state),
            )
            state["refine_payload"] = dict(refine_payload or {})
            diagnostics.set_extra("em_refinement", state["refine_payload"])
            diagnostics.set_count("fit2_lines_kept", len(models))

            models = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="model_selection",
                fn=lambda: selector.select(ctx, blobs, models, state),
            )
            diagnostics.set_count("final_lines_kept", len(models))
            rej = dict(state.get("selector_reject_counts") or {})
            for k, v in rej.items():
                diagnostics.set_count(str(k), int(v))

            contacts = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="contact_detection",
                fn=lambda: contact_detector.detect(ctx, blobs, models, state),
            )

            center_ras = np.asarray(ctx.get("center_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
            trajectories = self._models_to_trajectories(models, center_ras=center_ras)
            result["trajectories"] = trajectories
            result["contacts"] = [
                {
                    "trajectory_name": c.shank_id,
                    "label": c.contact_id,
                    "position_ras": list(c.position_ras),
                    "confidence": float(c.confidence),
                }
                for c in contacts
            ]

            if trajectories:
                points = np.asarray([b.centroid_ras for b in blobs], dtype=float).reshape(-1, 3) if blobs else np.zeros((0, 3), dtype=float)
                assigned = np.zeros((points.shape[0],), dtype=bool)
                rad = float(cfg.get("inlier_radius_mm", 1.2)) * 1.5
                for t in trajectories:
                    a = np.asarray(t.get("start_ras", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
                    b = np.asarray(t.get("end_ras", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
                    d = b - a
                    dn = float(np.linalg.norm(d))
                    if dn <= 1e-9 or points.shape[0] == 0:
                        continue
                    d /= dn
                    assigned |= (_line_distances(points, a, d) <= rad)
                diagnostics.set_count("assigned_points_after_refine", int(np.count_nonzero(assigned)))
                diagnostics.set_count("final_unassigned_points", int(max(0, points.shape[0] - np.count_nonzero(assigned))))
            else:
                diagnostics.set_count("assigned_points_after_refine", 0)
                diagnostics.set_count("final_unassigned_points", len(blobs))

            # Keep UI compatibility by publishing legacy-shaped payload to ctx extras.
            extras = ctx.get("extras")
            if isinstance(extras, dict):
                preview = dict(state.get("preview") or {})
                filt = dict(state.get("blob_filter") or {})
                legacy_lines = [
                    {
                        "name": t.get("name", ""),
                        "start_ras": list(t.get("start_ras", [0.0, 0.0, 0.0])),
                        "end_ras": list(t.get("end_ras", [0.0, 0.0, 0.0])),
                        "length_mm": float(t.get("length_mm", 0.0)),
                        "inlier_count": int(t.get("support_count", 0)),
                        "support_weight": float(t.get("support_mass", 0.0)),
                        "inside_fraction": float(t.get("confidence", 0.0)),
                        "rms_mm": float(t.get("params", {}).get("rms_mm", 0.0)),
                    }
                    for t in trajectories
                ]
                extras["legacy_result"] = {
                    "candidate_count": int(preview.get("candidate_count", 0)),
                    "head_mask_kept_count": int(preview.get("head_mask_kept_count", 0)),
                    "gating_mask_type": str(preview.get("gating_mask_type", "head_distance")),
                    "inside_method": str(preview.get("inside_method", "head_distance")),
                    "metal_in_head_count": int(preview.get("metal_in_head_count", 0)),
                    "depth_kept_count": int(preview.get("depth_kept_count", 0)),
                    "candidate_points_total": int(preview.get("candidate_count", 0)),
                    "candidate_points_after_mask": int(preview.get("metal_in_head_count", 0)),
                    "candidate_points_after_depth": int(preview.get("depth_kept_count", 0)),
                    "effective_min_inliers": int(diagnostics.diagnostics.get("counts", {}).get("effective_min_inliers", 0)),
                    "effective_inlier_radius_mm": float(cfg.get("inlier_radius_mm", 1.2) * 1.5),
                    "fit1_lines_proposed": int(diagnostics.diagnostics.get("counts", {}).get("fit1_lines_proposed", 0)),
                    "fit2_lines_kept": int(diagnostics.diagnostics.get("counts", {}).get("fit2_lines_kept", 0)),
                    "rescue_lines_kept": 0,
                    "final_lines_kept": int(len(legacy_lines)),
                    "assigned_points_after_refine": int(diagnostics.diagnostics.get("counts", {}).get("assigned_points_after_refine", 0)),
                    "unassigned_points_after_refine": int(diagnostics.diagnostics.get("counts", {}).get("final_unassigned_points", 0)),
                    "rescued_points": 0,
                    "final_unassigned_points": int(diagnostics.diagnostics.get("counts", {}).get("final_unassigned_points", 0)),
                    "gap_reject_count": 0,
                    "duplicate_reject_count": 0,
                    "start_zone_reject_count": int(diagnostics.diagnostics.get("counts", {}).get("start_zone_reject_count", 0)),
                    "length_reject_count": int(diagnostics.diagnostics.get("counts", {}).get("length_reject_count", 0)),
                    "inlier_reject_count": int(diagnostics.diagnostics.get("counts", {}).get("inlier_reject_count", 0)),
                    "blob_count_total": int(diagnostics.diagnostics.get("counts", {}).get("blob_count_total", 0)),
                    "blob_count_kept": int(diagnostics.diagnostics.get("counts", {}).get("blob_count_kept", 0)),
                    "blob_reject_small": int(diagnostics.diagnostics.get("counts", {}).get("blob_reject_small", 0)),
                    "blob_reject_large": int(diagnostics.diagnostics.get("counts", {}).get("blob_reject_large", 0)),
                    "blob_reject_intensity": int(diagnostics.diagnostics.get("counts", {}).get("blob_reject_intensity", 0)),
                    "blob_reject_shape": int(diagnostics.diagnostics.get("counts", {}).get("blob_reject_shape", 0)),
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
                    "blob_labelmap_kji": (state.get("blob_raw") or {}).get("labels_kji"),
                    "blob_centroids_all_ras": filt.get("blob_centroids_all_ras"),
                    "blob_centroids_kept_ras": filt.get("blob_centroids_kept_ras"),
                    "blob_centroids_rejected_ras": filt.get("blob_centroids_rejected_ras"),
                    "in_mask_depth_values_mm": np.asarray([float(b.depth_mean_mm) for b in blobs], dtype=float),
                    "in_mask_points_ras": np.asarray([b.centroid_ras for b in blobs], dtype=float).reshape(-1, 3),
                    "profile_ms": {
                        "total": float(diagnostics.diagnostics.get("timing", {}).get("total_ms", 0.0)),
                        "first_pass": float(diagnostics.diagnostics.get("timing", {}).get("stage.seed_initialization.ms", 0.0)),
                        "refine": float(diagnostics.diagnostics.get("timing", {}).get("stage.em_refinement.ms", 0.0)),
                        "rescue": 0.0,
                    },
                    "lines": legacy_lines,
                }

            writer = self.get_artifact_writer(ctx, result)
            blob_rows = [
                {
                    "blob_id": b.blob_id,
                    "x": b.centroid_ras[0],
                    "y": b.centroid_ras[1],
                    "z": b.centroid_ras[2],
                    "voxels": b.voxel_count,
                    "peak_hu": b.peak_hu,
                    "elongation": b.elongation,
                    "score_quality": float(b.scores.get("quality", 0.0)),
                }
                for b in blobs
            ]
            artifacts = write_standard_artifacts(
                writer,
                result,
                blobs=blob_rows,
                pipeline_payload={
                    "pipeline_id": self.pipeline_id,
                    "pipeline_version": self.pipeline_version,
                    "state_keys": sorted(state.keys()),
                    "refine_payload": state.get("refine_payload", {}),
                },
            )
            result["artifacts"].extend(artifacts)

            if bool(cfg.get("debug_em_payload", False)):
                path = writer.write_json(
                    "em_debug_payload.json",
                    {
                        "diagnostics": diagnostics.diagnostics,
                        "model_count": len(models),
                        "blob_count": len(blobs),
                    },
                )
                add_artifact(
                    result["artifacts"],
                    kind="em_debug_payload",
                    path=path,
                    description="EM debug payload",
                    stage="finalize",
                )

        except Exception as exc:
            self.fail(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage=str(getattr(exc, "stage", "pipeline")),
                exc=exc,
            )

        return self.finalize(result, diagnostics, t_start)
