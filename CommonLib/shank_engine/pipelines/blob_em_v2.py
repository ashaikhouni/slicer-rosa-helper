"""Blob-based shank extraction pipeline (EM-like refinement, shank-only).

Design goals for this pipeline:
- use permissive in-head metal blobs as observations (not hard depth-pass only)
- classify blobs as bead/segment/junk with soft probabilities
- seed from segment-like blobs first; bead-pair fallback when needed
- score hypotheses with mixed evidence (distance + segment-axis alignment)
- accept/peel/repeat for unknown number of shanks
- refine accepted shanks with weighted reassignment + weighted PCA updates
- output shank trajectories only (contacts intentionally unimplemented)
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


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-9:
        return np.asarray([0.0, 0.0, 1.0], dtype=float)
    return v / n


def _line_distances(points: np.ndarray, p0: np.ndarray, direction_unit: np.ndarray) -> np.ndarray:
    rel = points - p0.reshape(1, 3)
    t = rel @ direction_unit.reshape(3)
    closest = p0.reshape(1, 3) + np.outer(t, direction_unit.reshape(3))
    return np.linalg.norm(points - closest, axis=1)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    aa = _normalize(np.asarray(a, dtype=float).reshape(3))
    bb = _normalize(np.asarray(b, dtype=float).reshape(3))
    c = float(np.clip(abs(float(np.dot(aa, bb))), 0.0, 1.0))
    return float(np.degrees(np.arccos(c)))


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
    axis = _normalize(axis)
    return center.astype(float), axis.astype(float)


def _soft_gaussian(x: np.ndarray, sigma: float) -> np.ndarray:
    s = max(1e-6, float(sigma))
    return np.exp(-0.5 * (x / s) ** 2)


def _line_to_line_distance(p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray) -> float:
    """Shortest distance between two infinite 3D lines."""
    u = _normalize(np.asarray(d1, dtype=float).reshape(3))
    v = _normalize(np.asarray(d2, dtype=float).reshape(3))
    w0 = np.asarray(p1, dtype=float).reshape(3) - np.asarray(p2, dtype=float).reshape(3)
    c = np.cross(u, v)
    cn = float(np.linalg.norm(c))
    if cn <= 1e-6:
        # Nearly parallel: use distance from p2 to line (p1, u).
        return float(np.linalg.norm(np.cross(w0, u)))
    return float(abs(np.dot(w0, c)) / cn)


def _blob_axis_array(blobs: list[BlobRecord]) -> np.ndarray:
    if not blobs:
        return np.zeros((0, 3), dtype=float)
    out = np.asarray([b.pca_axis_ras for b in blobs], dtype=float).reshape(-1, 3)
    n = np.linalg.norm(out, axis=1, keepdims=True)
    n = np.maximum(n, 1e-9)
    return out / n


def _supports_for_line(
    points: np.ndarray,
    blobs: list[BlobRecord],
    p0: np.ndarray,
    direction: np.ndarray,
    cfg: dict[str, Any],
    active_weights: np.ndarray,
) -> dict[str, Any]:
    d = _normalize(direction)
    dist = _line_distances(points, p0, d)
    axis_arr = _blob_axis_array(blobs)

    p_bead = np.asarray([float(b.scores.get("p_bead", 0.0)) for b in blobs], dtype=float)
    p_segment = np.asarray([float(b.scores.get("p_segment", 0.0)) for b in blobs], dtype=float)
    p_junk = np.asarray([float(b.scores.get("p_junk", 0.0)) for b in blobs], dtype=float)
    support_weight = np.asarray([float(b.scores.get("support_weight", 1.0)) for b in blobs], dtype=float)
    depth_prior = np.asarray([float(b.scores.get("depth_prior", 1.0)) for b in blobs], dtype=float)

    r_mm = float(cfg.get("inlier_radius_mm", 1.2)) * float(cfg.get("seed_radius_scale", 1.6))
    theta_sigma = float(cfg.get("segment_angle_sigma_deg", 12.0))
    theta_max = float(cfg.get("segment_angle_max_deg", 20.0))

    bead_term = _soft_gaussian(dist, r_mm) * p_bead

    cosv = np.clip(np.abs(axis_arr @ d.reshape(3)), 0.0, 1.0)
    ang = np.degrees(np.arccos(cosv))
    align_term = _soft_gaussian(ang, theta_sigma)
    seg_term = _soft_gaussian(dist, r_mm) * align_term * p_segment

    # Mixed evidence with soft junk suppression and depth prior.
    contrib = (bead_term + seg_term) * depth_prior * (1.0 - 0.65 * p_junk)
    contrib *= support_weight * active_weights

    bead_inlier = (dist <= r_mm) & (p_bead >= 0.20)
    seg_inlier = (dist <= r_mm * 1.2) & (p_segment >= 0.20) & (ang <= theta_max)
    inlier = bead_inlier | seg_inlier

    if int(np.count_nonzero(inlier)) > 0:
        proj = (points[inlier] - p0.reshape(1, 3)) @ d.reshape(3)
        span = float(np.max(proj) - np.min(proj))
    else:
        span = 0.0

    mass_total = float(np.sum(contrib[inlier])) if int(np.count_nonzero(inlier)) > 0 else 0.0
    mass_bead = float(np.sum((bead_term * support_weight * active_weights)[inlier])) if int(np.count_nonzero(inlier)) > 0 else 0.0
    mass_seg = float(np.sum((seg_term * support_weight * active_weights)[inlier])) if int(np.count_nonzero(inlier)) > 0 else 0.0

    return {
        "dist": dist,
        "ang": ang,
        "inlier": inlier,
        "contrib": contrib,
        "mass_total": mass_total,
        "mass_bead": mass_bead,
        "mass_segment": mass_seg,
        "span_mm": span,
    }


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
            include_debug_masks=bool(cfg.get("debug_masks", False)),
        )


class _EMBlobExtractor:
    def _blob_from_points(
        self,
        points_kji: np.ndarray,
        *,
        arr_kji: np.ndarray,
        depth_map_kji: np.ndarray | None,
        ijk_kji_to_ras_fn,
        blob_id: int,
        meta: dict[str, Any],
    ) -> dict[str, Any] | None:
        pts = np.asarray(points_kji, dtype=float).reshape(-1, 3)
        if pts.shape[0] == 0:
            return None
        centroid_kji = np.mean(pts, axis=0)
        if ijk_kji_to_ras_fn is not None:
            ras_pts = np.asarray(ijk_kji_to_ras_fn(pts), dtype=float).reshape(-1, 3)
        else:
            ras_pts = np.stack([pts[:, 2], pts[:, 1], pts[:, 0]], axis=1)
        centroid_ras = np.mean(ras_pts, axis=0)
        centered = ras_pts - centroid_ras.reshape(1, 3)
        if centered.shape[0] >= 3:
            cov = (centered.T @ centered) / max(1, centered.shape[0] - 1)
            evals, evecs = np.linalg.eigh(cov)
            order = np.argsort(evals)
            evals = np.maximum(evals[order], 1e-9)
            axis = _normalize(evecs[:, order[-1]])
        else:
            evals = np.asarray([1e-9, 1e-9, 1e-9], dtype=float)
            axis = np.asarray([0.0, 0.0, 1.0], dtype=float)
        proj = centered @ axis.reshape(3)
        length_mm = float(np.max(proj) - np.min(proj)) if proj.size else 0.0
        diameter_mm = float(2.0 * np.sqrt(max(1e-9, float(np.mean(evals[:2])))))
        elongation = float(evals[-1] / max(1e-9, float(evals[0])))
        p_int = np.asarray(np.round(pts), dtype=int)
        hu = arr_kji[p_int[:, 0], p_int[:, 1], p_int[:, 2]]
        hu_max = float(np.max(hu)) if hu.size else 0.0
        hu_q95 = float(np.percentile(hu, 95)) if hu.size else 0.0
        hu_mean = float(np.mean(hu)) if hu.size else 0.0
        d_min = d_mean = d_max = 0.0
        if depth_map_kji is not None:
            d = np.asarray(depth_map_kji, dtype=float)[p_int[:, 0], p_int[:, 1], p_int[:, 2]]
            if d.size:
                d_min = float(np.min(d))
                d_mean = float(np.mean(d))
                d_max = float(np.max(d))
        return {
            "blob_id": int(blob_id),
            "voxel_count": int(pts.shape[0]),
            "centroid_kji": [float(centroid_kji[0]), float(centroid_kji[1]), float(centroid_kji[2])],
            "centroid_ras": [float(centroid_ras[0]), float(centroid_ras[1]), float(centroid_ras[2])],
            "hu_max": hu_max,
            "hu_q95": hu_q95,
            "hu_mean": hu_mean,
            "depth_min": d_min,
            "depth_mean": d_mean,
            "depth_max": d_max,
            "elongation": elongation,
            "pca_axis_ras": [float(axis[0]), float(axis[1]), float(axis[2])],
            "pca_evals": [float(evals[0]), float(evals[1]), float(evals[2])],
            "length_mm": length_mm,
            "diameter_mm": diameter_mm,
            "meta": dict(meta or {}),
        }

    @staticmethod
    def _electrode_priors_from_cfg(cfg: dict[str, Any], ctx: DetectionContext) -> dict[str, float]:
        prior_values: dict[str, list[float]] = {
            "contact_length_mm": [],
            "diameter_mm": [],
            "pitch_mm": [],
            "gap_mm": [],
        }

        def _append_prior(key: str, value: Any) -> None:
            try:
                f = float(value)
            except Exception:
                return
            if np.isfinite(f) and f > 0.0:
                prior_values[key].append(float(f))

        prior_sources: list[Any] = []
        if isinstance(cfg.get("electrode_priors"), dict):
            prior_sources.append(cfg.get("electrode_priors"))
        if isinstance(ctx.get("electrode_priors"), dict):
            prior_sources.append(ctx.get("electrode_priors"))
        extras = ctx.get("extras")
        if isinstance(extras, dict) and isinstance(extras.get("electrode_priors"), dict):
            prior_sources.append(extras.get("electrode_priors"))

        for src in prior_sources:
            if not isinstance(src, dict):
                continue
            _append_prior("contact_length_mm", src.get("contact_length_mm"))
            _append_prior("contact_length_mm", src.get("electrode_contact_length_mm"))
            _append_prior("diameter_mm", src.get("diameter_mm"))
            _append_prior("diameter_mm", src.get("electrode_diameter_mm"))
            _append_prior("pitch_mm", src.get("pitch_mm"))
            _append_prior("pitch_mm", src.get("contact_pitch_mm"))
            _append_prior("pitch_mm", src.get("contact_separation_mm"))
            _append_prior("gap_mm", src.get("gap_mm"))
            _append_prior("gap_mm", src.get("contact_gap_mm"))

        _append_prior("contact_length_mm", cfg.get("electrode_prior_contact_length_mm"))
        _append_prior("contact_length_mm", cfg.get("axial_support_contact_length_mm"))
        _append_prior("diameter_mm", cfg.get("electrode_prior_diameter_mm"))
        _append_prior("diameter_mm", cfg.get("axial_support_nominal_diameter_mm"))
        _append_prior("pitch_mm", cfg.get("electrode_prior_pitch_mm"))
        _append_prior("pitch_mm", cfg.get("electrode_prior_contact_separation_mm"))
        _append_prior("pitch_mm", cfg.get("axial_support_spacing_mm"))
        _append_prior("gap_mm", cfg.get("electrode_prior_gap_mm"))

        contact_len = float(np.median(prior_values["contact_length_mm"])) if prior_values["contact_length_mm"] else 2.0
        diameter = float(np.median(prior_values["diameter_mm"])) if prior_values["diameter_mm"] else 0.8
        pitch_cfg = float(np.median(prior_values["pitch_mm"])) if prior_values["pitch_mm"] else 3.5
        gap_cfg = float(np.median(prior_values["gap_mm"])) if prior_values["gap_mm"] else max(0.0, pitch_cfg - contact_len)
        spacing = contact_len + gap_cfg if gap_cfg > 0.0 else pitch_cfg
        return {
            "contact_length_mm": max(0.5, contact_len),
            "pitch_mm": max(1.0, pitch_cfg),
            "gap_mm": max(0.0, gap_cfg),
            "spacing_mm": max(1.0, spacing),
            "diameter_mm": max(0.2, diameter),
        }

    def _eligible_for_axial_support_sampling(
        self,
        blob: dict[str, Any],
        cfg: dict[str, Any],
        priors: dict[str, float],
    ) -> tuple[bool, dict[str, Any]]:
        elong = float(blob.get("elongation", 1.0))
        length = float(blob.get("length_mm", 0.0))
        diam = float(blob.get("diameter_mm", 0.0))
        vox = int(blob.get("voxel_count", 0))
        aspect = length / max(diam, 1e-6)
        min_elong = float(cfg.get("axial_support_min_elongation", 2.0))
        min_aspect = float(cfg.get("axial_support_min_aspect_ratio", 3.0))
        min_vox = int(cfg.get("axial_support_min_voxels", 10))
        min_len = float(cfg.get("axial_support_min_length_mm", max(4.0, 2.0 * priors["spacing_mm"])))
        min_diam = float(cfg.get("axial_support_min_diameter_mm", max(0.3, 0.5 * priors["diameter_mm"])))
        max_diam = float(cfg.get("axial_support_max_diameter_mm", max(2.5, 3.0 * priors["diameter_mm"])))
        eligible = (
            vox >= min_vox
            and length >= min_len
            and (elong >= min_elong or aspect >= min_aspect)
            and (min_diam <= diam <= max_diam)
        )
        info = {
            "elongation": float(elong),
            "aspect_ratio": float(aspect),
            "length_mm": float(length),
            "diameter_mm": float(diam),
            "voxel_count": int(vox),
            "min_length_mm": float(min_len),
            "min_voxels": int(min_vox),
            "min_diameter_mm": float(min_diam),
            "max_diameter_mm": float(max_diam),
            "min_elongation": float(min_elong),
            "min_aspect_ratio": float(min_aspect),
        }
        return bool(eligible), info

    def _axial_support_samples_for_blob(
        self,
        *,
        blob: dict[str, Any],
        points_kji: np.ndarray,
        arr_kji: np.ndarray,
        depth_map_kji: np.ndarray | None,
        ijk_kji_to_ras_fn,
        cfg: dict[str, Any],
        priors: dict[str, float],
    ) -> list[dict[str, Any]]:
        pts = np.asarray(points_kji, dtype=float).reshape(-1, 3)
        if pts.shape[0] == 0:
            return []
        if ijk_kji_to_ras_fn is not None:
            ras_pts = np.asarray(ijk_kji_to_ras_fn(pts), dtype=float).reshape(-1, 3)
        else:
            ras_pts = np.stack([pts[:, 2], pts[:, 1], pts[:, 0]], axis=1)
        c = np.asarray(blob.get("centroid_ras", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
        axis = _normalize(np.asarray(blob.get("pca_axis_ras", [0.0, 0.0, 1.0]), dtype=float).reshape(3))
        t = (ras_pts - c.reshape(1, 3)) @ axis.reshape(3)
        if t.size == 0:
            return []
        t_min = float(np.min(t))
        t_max = float(np.max(t))
        span = max(0.0, t_max - t_min)
        spacing = float(cfg.get("axial_support_spacing_mm", priors["spacing_mm"]))
        spacing = max(1.0, spacing)
        window = float(cfg.get("axial_support_window_mm", max(priors["contact_length_mm"], 0.75 * spacing)))
        window = max(0.5, window)
        min_vox_window = int(cfg.get("axial_support_min_window_voxels", 3))
        max_samples = int(cfg.get("axial_support_max_samples_per_blob", 32))
        if span <= 1e-6:
            return []
        n_samples = int(np.floor(span / spacing)) + 1
        n_samples = int(np.clip(n_samples, 2, max_samples))
        if n_samples < 2:
            return []
        t_samples = np.linspace(t_min, t_max, n_samples)
        samples: list[dict[str, Any]] = []
        half_w = 0.5 * window
        for support_idx, t0 in enumerate(t_samples):
            sel = np.where(np.abs(t - float(t0)) <= half_w)[0]
            if sel.size < min_vox_window:
                continue
            local_ras = ras_pts[sel]
            local_kji = pts[sel]
            local_centroid_ras = np.mean(local_ras, axis=0)
            local_centroid_kji = np.mean(local_kji, axis=0)
            p_int = np.asarray(np.round(local_kji), dtype=int)
            hu_vals = arr_kji[p_int[:, 0], p_int[:, 1], p_int[:, 2]]
            hu_max = float(np.max(hu_vals)) if hu_vals.size else float(blob.get("hu_max", 0.0))
            hu_q95 = float(np.percentile(hu_vals, 95)) if hu_vals.size else float(blob.get("hu_q95", 0.0))
            hu_mean = float(np.mean(hu_vals)) if hu_vals.size else float(blob.get("hu_mean", 0.0))
            d_min = d_mean = d_max = 0.0
            if depth_map_kji is not None:
                d_vals = np.asarray(depth_map_kji, dtype=float)[p_int[:, 0], p_int[:, 1], p_int[:, 2]]
                if d_vals.size:
                    d_min = float(np.min(d_vals))
                    d_mean = float(np.mean(d_vals))
                    d_max = float(np.max(d_vals))
            samples.append(
                {
                    "centroid_ras": [float(local_centroid_ras[0]), float(local_centroid_ras[1]), float(local_centroid_ras[2])],
                    "centroid_kji": [float(local_centroid_kji[0]), float(local_centroid_kji[1]), float(local_centroid_kji[2])],
                    "voxel_count": int(sel.size),
                    "hu_max": hu_max,
                    "hu_q95": hu_q95,
                    "hu_mean": hu_mean,
                    "depth_min": float(d_min),
                    "depth_mean": float(d_mean),
                    "depth_max": float(d_max),
                    "local_t_mm": float(np.mean(t[sel]) if sel.size else t0),
                    "support_index": int(support_idx),
                    "sampling_spacing_mm": float(spacing),
                    "sampling_window_mm": float(window),
                }
            )
        return samples

    def _support_observations_for_blob(
        self,
        *,
        blob: dict[str, Any],
        points_kji: np.ndarray,
        arr_kji: np.ndarray,
        depth_map_kji: np.ndarray | None,
        ijk_kji_to_ras_fn,
        cfg: dict[str, Any],
        priors: dict[str, float],
    ) -> list[dict[str, Any]]:
        parent_id = int(blob.get("blob_id", 0))
        eligible, eligibility_info = self._eligible_for_axial_support_sampling(blob, cfg, priors)
        if not eligible:
            one = dict(blob)
            one["meta"] = dict(one.get("meta") or {}) | {
                "parent_blob_id": parent_id,
                "support_kind": "blob_centroid",
                "support_index": 0,
                "support_count_for_parent": 1,
                "support_weight_fraction": 1.0,
                "axial_sampling_eligible": False,
                "axial_sampling_info": eligibility_info,
                "parent_voxel_count": int(blob.get("voxel_count", 0)),
                "parent_length_mm": float(blob.get("length_mm", 0.0)),
                "parent_diameter_mm": float(blob.get("diameter_mm", 0.0)),
                "parent_elongation": float(blob.get("elongation", 1.0)),
                "parent_hu_q95": float(blob.get("hu_q95", 0.0) or 0.0),
                "parent_hu_max": float(blob.get("hu_max", 0.0) or 0.0),
                "parent_depth_mean_mm": float(blob.get("depth_mean", 0.0) or 0.0),
                "parent_pca_evals": [float(v) for v in (blob.get("pca_evals") or [0.0, 0.0, 0.0])],
            }
            return [one]
        samples = self._axial_support_samples_for_blob(
            blob=blob,
            points_kji=points_kji,
            arr_kji=arr_kji,
            depth_map_kji=depth_map_kji,
            ijk_kji_to_ras_fn=ijk_kji_to_ras_fn,
            cfg=cfg,
            priors=priors,
        )
        if not samples:
            one = dict(blob)
            one["meta"] = dict(one.get("meta") or {}) | {
                "parent_blob_id": parent_id,
                "support_kind": "blob_centroid",
                "support_index": 0,
                "support_count_for_parent": 1,
                "support_weight_fraction": 1.0,
                "axial_sampling_eligible": True,
                "axial_sampling_info": eligibility_info,
                "axial_sampling_reason": "no_valid_local_windows",
                "parent_voxel_count": int(blob.get("voxel_count", 0)),
                "parent_length_mm": float(blob.get("length_mm", 0.0)),
                "parent_diameter_mm": float(blob.get("diameter_mm", 0.0)),
                "parent_elongation": float(blob.get("elongation", 1.0)),
                "parent_hu_q95": float(blob.get("hu_q95", 0.0) or 0.0),
                "parent_hu_max": float(blob.get("hu_max", 0.0) or 0.0),
                "parent_depth_mean_mm": float(blob.get("depth_mean", 0.0) or 0.0),
                "parent_pca_evals": [float(v) for v in (blob.get("pca_evals") or [0.0, 0.0, 0.0])],
            }
            return [one]
        n = int(len(samples))
        frac = float(1.0 / max(1, n))
        out: list[dict[str, Any]] = []
        for i, s in enumerate(samples):
            entry = dict(blob)
            entry["centroid_ras"] = list(s["centroid_ras"])
            entry["centroid_kji"] = list(s["centroid_kji"])
            entry["voxel_count"] = int(s["voxel_count"])
            entry["hu_max"] = float(s["hu_max"])
            entry["hu_q95"] = float(s["hu_q95"])
            entry["hu_mean"] = float(s["hu_mean"])
            entry["depth_min"] = float(s["depth_min"])
            entry["depth_mean"] = float(s["depth_mean"])
            entry["depth_max"] = float(s["depth_max"])
            entry["meta"] = dict(entry.get("meta") or {}) | {
                "parent_blob_id": parent_id,
                "support_kind": "axial_sample",
                "support_index": int(i),
                "support_count_for_parent": int(n),
                "support_weight_fraction": float(frac),
                "local_t_mm": float(s["local_t_mm"]),
                "sampling_spacing_mm": float(s["sampling_spacing_mm"]),
                "sampling_window_mm": float(s["sampling_window_mm"]),
                "axial_sampling_eligible": True,
                "axial_sampling_info": eligibility_info,
                "parent_voxel_count": int(blob.get("voxel_count", 0)),
                "parent_length_mm": float(blob.get("length_mm", 0.0)),
                "parent_diameter_mm": float(blob.get("diameter_mm", 0.0)),
                "parent_elongation": float(blob.get("elongation", 1.0)),
                "parent_hu_q95": float(blob.get("hu_q95", 0.0) or 0.0),
                "parent_hu_max": float(blob.get("hu_max", 0.0) or 0.0),
                "parent_depth_mean_mm": float(blob.get("depth_mean", 0.0) or 0.0),
                "parent_pca_evals": [float(v) for v in (blob.get("pca_evals") or [0.0, 0.0, 0.0])],
            }
            out.append(entry)
        return out

    def extract(self, ctx: DetectionContext, state: dict[str, Any]) -> list[BlobRecord]:
        preview = dict(state.get("preview") or {})
        arr_kji = np.asarray(ctx["arr_kji"])
        cfg = dict(ctx.get("config") or {})
        state["support_stats"] = {
            "blobs_total_cc": 0,
            "blobs_eligible_for_axial_sampling": 0,
            "axial_support_samples_generated": 0,
            "support_points_total": 0,
            "support_points_from_compact_blobs": 0,
            "support_points_from_elongated_blobs": 0,
        }
        state["support_sampling"] = {
            "expanded_parent_blob_ids": [],
            "sample_count_per_parent": {},
            "spacing_mm": 0.0,
            "window_mm": 0.0,
            "electrode_priors": {},
        }

        # Blob candidate source:
        # - default: permissive in-head metal mask
        # - optional: hard distance-pass mask for stricter near-air suppression
        # Default to v1 voxel-fit parity: depth-pass metal mask first.
        use_distance_mask = bool(cfg.get("use_distance_mask_for_blob_candidates", True))
        metal_mask = None
        source = ""
        if use_distance_mask:
            metal_mask = preview.get("metal_depth_pass_mask_kji")
            source = "metal_depth_pass"
            if metal_mask is None:
                metal_mask = preview.get("metal_depth_pass")
                source = "metal_depth_pass"

        if metal_mask is None:
            metal_mask = preview.get("metal_in_head_mask_kji")
            source = "metal_in_head"
        if metal_mask is None:
            metal_mask = preview.get("metal_in_head")
        if metal_mask is None:
            metal_mask = preview.get("metal_in_gate_mask_kji")
            source = "metal_in_gate"
        if metal_mask is None:
            metal_mask = preview.get("gating_mask_kji")
            if metal_mask is not None:
                metal_mask = np.logical_and(np.asarray(preview.get("metal_mask_kji"), dtype=bool), np.asarray(metal_mask, dtype=bool))
                source = "gating_and_metal"
        if metal_mask is None:
            metal_mask = preview.get("metal_mask_kji")
            source = "metal_only_fallback"
        if metal_mask is None:
            return []
        state["blob_source"] = source

        raw = extract_blob_candidates(
            metal_mask_kji=np.asarray(metal_mask, dtype=bool),
            arr_kji=arr_kji,
            depth_map_kji=preview.get("head_distance_map_kji"),
            ijk_kji_to_ras_fn=ctx.get("ijk_kji_to_ras_fn"),
            fully_connected=False,
        )
        filt = filter_blob_candidates(
            raw,
            min_blob_voxels=int(cfg.get("min_blob_voxels", 2)),
            # Blob EM v2: do not hard-reject large merged blobs at pre-filter stage.
            # Keep large blobs for downstream split/segment scoring.
            max_blob_voxels=0,
            min_blob_peak_hu=cfg.get("min_blob_peak_hu", None),
            max_blob_elongation=cfg.get("max_blob_elongation", None),
        )
        state["blob_raw"] = raw
        state["blob_filter"] = filt
        support_stats = {
            "blobs_total_cc": int(raw.get("blob_count_total", 0)),
            "blobs_eligible_for_axial_sampling": 0,
            "axial_support_samples_generated": 0,
            "support_points_total": 0,
            "support_points_from_compact_blobs": 0,
            "support_points_from_elongated_blobs": 0,
        }
        parent_sample_counts: dict[int, int] = {}
        raw_labels = raw.get("labels_kji") if isinstance(raw, dict) else None
        labels_kji = np.asarray(raw_labels, dtype=int) if raw_labels is not None else np.asarray([], dtype=int)
        depth_map_kji = preview.get("head_distance_map_kji")
        cfg = dict(ctx.get("config") or {})
        priors = self._electrode_priors_from_cfg(cfg, ctx)
        expanded_blobs: list[dict[str, Any]] = []
        for blob in list(filt.get("kept_blobs") or []):
            parent_id = int(blob.get("blob_id", 0))
            if labels_kji.size > 0:
                pts = np.argwhere(np.asarray(labels_kji, dtype=int) == parent_id)
            else:
                pts = np.asarray([], dtype=float).reshape(0, 3)
            supports = self._support_observations_for_blob(
                blob=blob,
                points_kji=pts,
                arr_kji=arr_kji,
                depth_map_kji=depth_map_kji,
                ijk_kji_to_ras_fn=ctx.get("ijk_kji_to_ras_fn"),
                cfg=cfg,
                priors=priors,
            )
            count_for_parent = int(len(supports))
            parent_sample_counts[parent_id] = count_for_parent
            if count_for_parent > 1:
                support_stats["blobs_eligible_for_axial_sampling"] += 1
                support_stats["axial_support_samples_generated"] += int(count_for_parent)
                support_stats["support_points_from_elongated_blobs"] += int(count_for_parent)
            else:
                support_stats["support_points_from_compact_blobs"] += 1
            expanded_blobs.extend(supports)
        support_stats["support_points_total"] = int(len(expanded_blobs))
        state["support_stats"] = dict(support_stats)
        state["support_sampling"] = {
            "expanded_parent_blob_ids": sorted([int(pid) for pid, n in parent_sample_counts.items() if int(n) > 1]),
            "sample_count_per_parent": {str(int(k)): int(v) for k, v in parent_sample_counts.items()},
            "spacing_mm": float(cfg.get("axial_support_spacing_mm", priors["spacing_mm"])),
            "window_mm": float(cfg.get("axial_support_window_mm", max(priors["contact_length_mm"], 0.75 * priors["spacing_mm"]))),
            "electrode_priors": dict(priors),
        }

        out: list[BlobRecord] = []
        next_support_id = 1
        for blob in expanded_blobs:
            c_ras = blob.get("centroid_ras")
            c_kji = blob.get("centroid_kji")
            if c_ras is None or c_kji is None:
                continue
            out.append(
                BlobRecord(
                    blob_id=int(next_support_id),
                    centroid_ras=(float(c_ras[0]), float(c_ras[1]), float(c_ras[2])),
                    centroid_kji=(float(c_kji[0]), float(c_kji[1]), float(c_kji[2])),
                    voxel_count=int(blob.get("voxel_count", 0)),
                    peak_hu=float(blob.get("hu_max", 0.0) or 0.0),
                    q95_hu=float(blob.get("hu_q95", blob.get("hu_max", 0.0)) or 0.0),
                    mean_hu=float(blob.get("hu_mean", 0.0) or 0.0),
                    pca_axis_ras=tuple(float(v) for v in (blob.get("pca_axis_ras") or [0.0, 0.0, 1.0])),
                    pca_evals=tuple(float(v) for v in (blob.get("pca_evals") or [0.0, 0.0, 0.0])),
                    length_mm=float(blob.get("length_mm", 0.0) or 0.0),
                    diameter_mm=float(blob.get("diameter_mm", 0.0) or 0.0),
                    elongation=float(blob.get("elongation", 1.0) or 1.0),
                    depth_min_mm=float(blob.get("depth_min", 0.0) or 0.0),
                    depth_mean_mm=float(blob.get("depth_mean", 0.0) or 0.0),
                    depth_max_mm=float(blob.get("depth_max", 0.0) or 0.0),
                    meta=dict(blob.get("meta") or {}),
                )
            )
            next_support_id += 1
        return out


class _EMBlobScorer:
    def score(self, ctx: DetectionContext, blobs: list[BlobRecord], state: dict[str, Any]) -> list[BlobRecord]:
        cfg = dict(ctx.get("config") or {})
        min_depth = float(cfg.get("min_metal_depth_mm", 5.0))
        scored: list[BlobRecord] = []

        for b in blobs:
            meta = dict(b.meta or {})
            support_fraction = float(np.clip(meta.get("support_weight_fraction", 1.0), 1e-6, 1.0))
            support_kind = str(meta.get("support_kind", "blob_centroid"))

            elong_raw = float(meta.get("parent_elongation", b.elongation)) if support_kind == "axial_sample" else float(b.elongation)
            vox_raw = float(meta.get("parent_voxel_count", b.voxel_count)) if support_kind == "axial_sample" else float(b.voxel_count)
            hu_raw = float(meta.get("parent_hu_q95", b.q95_hu if b.q95_hu > 0.0 else b.peak_hu)) if support_kind == "axial_sample" else float(
                b.q95_hu if b.q95_hu > 0.0 else b.peak_hu
            )
            length_raw = float(meta.get("parent_length_mm", b.length_mm)) if support_kind == "axial_sample" else float(b.length_mm)
            diam_raw = float(meta.get("parent_diameter_mm", b.diameter_mm)) if support_kind == "axial_sample" else float(b.diameter_mm)
            depth_parent = float(meta.get("parent_depth_mean_mm", b.depth_mean_mm))
            depth_local = float(b.depth_mean_mm)
            depth = float(max(0.0, 0.50 * max(0.0, depth_parent) + 0.50 * max(0.0, depth_local)))
            evals_parent = meta.get("parent_pca_evals", b.pca_evals)

            elong = float(max(1.0, elong_raw))
            vox = float(max(1.0, vox_raw))
            hu = float(max(0.0, hu_raw))
            length = float(max(0.0, length_raw))
            diam = float(max(0.1, diam_raw))

            ev = np.asarray(evals_parent if support_kind == "axial_sample" else b.pca_evals, dtype=float).reshape(3)
            axis_clarity = 0.0
            if float(np.sum(ev)) > 1e-9:
                axis_clarity = float(np.clip(ev[2] / max(1e-9, float(np.sum(ev))), 0.0, 1.0))

            hu_high = float(np.clip((hu - 1700.0) / 1400.0, 0.0, 1.0))
            depth_soft = float(np.clip((depth - min_depth) / 15.0, 0.0, 1.0))
            superficial = float(np.clip((min_depth - depth) / max(1.0, min_depth), 0.0, 1.0))

            compactness = float(np.clip(1.8 / max(1.8, elong), 0.0, 1.0))
            plausible_bead_size = float(np.clip(1.0 - abs(diam - 2.5) / 4.0, 0.0, 1.0))

            seg_shape = float(np.clip((elong - 2.0) / 4.5, 0.0, 1.0))
            seg_len = float(np.clip((length - 2.0) / 10.0, 0.0, 1.0))
            plausible_seg_diam = float(np.clip(1.0 - abs(diam - 2.0) / 8.0, 0.0, 1.0))

            p_segment_raw = 0.38 * seg_shape + 0.26 * seg_len + 0.24 * axis_clarity + 0.12 * plausible_seg_diam
            p_bead_raw = 0.36 * compactness + 0.28 * plausible_bead_size + 0.22 * hu_high + 0.14 * depth_soft

            # Junk: implausible morphology/HU and superficial elongated structures.
            elongated_superficial = float(np.clip(seg_shape * superficial, 0.0, 1.0))
            low_hu = float(np.clip((1500.0 - hu) / 900.0, 0.0, 1.0))
            very_large = float(np.clip((vox - 200.0) / 600.0, 0.0, 1.0))
            p_junk_raw = 0.50 * elongated_superficial + 0.30 * low_hu + 0.20 * very_large

            s = float(p_segment_raw + p_bead_raw + p_junk_raw)
            if s <= 1e-9:
                p_segment = p_bead = 0.33
                p_junk = 0.34
            else:
                p_segment = float(np.clip(p_segment_raw / s, 0.0, 1.0))
                p_bead = float(np.clip(p_bead_raw / s, 0.0, 1.0))
                p_junk = float(np.clip(p_junk_raw / s, 0.0, 1.0))

            support_weight_base = float(
                np.clip(
                    math.sqrt(vox) * (0.25 + 0.75 * (0.55 * p_segment + 0.45 * p_bead)) * (1.0 - 0.30 * p_junk),
                    0.3,
                    12.0,
                )
            )
            support_weight = float(np.clip(support_weight_base * support_fraction, 1e-4, 12.0))
            depth_prior = float(np.clip(0.40 + 0.60 * depth_soft, 0.15, 1.0))

            scores = {
                "p_segment": p_segment,
                "p_bead": p_bead,
                "p_junk": p_junk,
                "support_weight": support_weight,
                "support_weight_base": support_weight_base,
                "support_weight_fraction": support_fraction,
                "depth_prior": depth_prior,
            }
            scored.append(replace(b, scores=scores))

        return scored


class _EMInitializer:
    def _segment_seeds(self, blobs: list[BlobRecord], points: np.ndarray, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        max_lines = max(1, int(cfg.get("max_lines", 30)))
        max_raw = int(cfg.get("max_segment_seeds_before_nms", max(24, 6 * max_lines)))
        min_p_segment = float(cfg.get("segment_seed_min_p_segment", 0.25))
        ranked = sorted(
            [i for i in range(len(blobs)) if float(blobs[i].scores.get("p_segment", 0.0)) >= min_p_segment],
            key=lambda i: (
                float(blobs[i].scores.get("p_segment", 0.0))
                * float(blobs[i].scores.get("support_weight", 1.0))
                * max(0.1, float(blobs[i].length_mm))
            ),
            reverse=True,
        )
        out: list[dict[str, Any]] = []
        for i in ranked[: max_raw]:
            b = blobs[i]
            out.append(
                {
                    "point": np.asarray(points[i], dtype=float).reshape(3),
                    "direction": _normalize(np.asarray(b.pca_axis_ras, dtype=float).reshape(3)),
                    "score": float(
                        float(b.scores.get("p_segment", 0.0))
                        * float(b.scores.get("support_weight", 1.0))
                        * max(0.1, float(b.length_mm))
                    ),
                    "seed_type": "segment",
                    "blob_i": int(i),
                    "blob_j": int(i),
                }
            )
        return out

    def _bead_pair_seeds(
        self,
        blobs: list[BlobRecord],
        points: np.ndarray,
        cfg: dict[str, Any],
        candidate_indices: list[int] | None = None,
        max_raw_override: int | None = None,
    ) -> list[dict[str, Any]]:
        max_lines = max(1, int(cfg.get("max_lines", 30)))
        max_raw = int(max_raw_override if max_raw_override is not None else cfg.get("max_bead_seeds_before_nms", max(20, 4 * max_lines)))
        ranked_all = sorted(
            range(len(blobs)),
            key=lambda i: float(blobs[i].scores.get("p_bead", 0.0)) * float(blobs[i].scores.get("support_weight", 1.0)),
            reverse=True,
        )
        if candidate_indices is not None:
            allowed = {int(i) for i in candidate_indices}
            ranked = [i for i in ranked_all if i in allowed]
        else:
            ranked = ranked_all
        bead_idx = ranked[: min(len(ranked), max(16, 2 * max_lines))]
        out: list[dict[str, Any]] = []
        if len(bead_idx) < 2:
            return out
        pairs: list[tuple[float, int, int]] = []
        for a in range(len(bead_idx)):
            for b in range(a + 1, len(bead_idx)):
                i = bead_idx[a]
                j = bead_idx[b]
                d = np.asarray(points[j] - points[i], dtype=float)
                dn = float(np.linalg.norm(d))
                if dn <= 1e-6:
                    continue
                bi = blobs[i]
                bj = blobs[j]
                w = (
                    float(bi.scores.get("p_bead", 0.0))
                    * float(bi.scores.get("support_weight", 1.0))
                    * float(bj.scores.get("p_bead", 0.0))
                    * float(bj.scores.get("support_weight", 1.0))
                )
                pairs.append((dn * max(1e-6, w), i, j))
        pairs.sort(reverse=True)
        for score, i, j in pairs[:max_raw]:
            out.append(
                {
                    "point": np.asarray(points[i], dtype=float).reshape(3),
                    "direction": _normalize(np.asarray(points[j] - points[i], dtype=float).reshape(3)),
                    "score": float(score),
                    "seed_type": "bead_pair",
                    "blob_i": int(i),
                    "blob_j": int(j),
                }
            )
        return out

    def _nms_seeds(self, seeds: list[dict[str, Any]], cfg: dict[str, Any], max_keep: int) -> list[dict[str, Any]]:
        if not seeds:
            return []
        angle_thresh = float(cfg.get("seed_nms_angle_deg", 6.0))
        line_thresh = float(cfg.get("seed_nms_line_distance_mm", 1.2))
        ranked = sorted(
            seeds,
            key=lambda s: (
                float(s.get("score", 0.0)),
                int(s.get("blob_i", -1)),
                int(s.get("blob_j", -1)),
            ),
            reverse=True,
        )
        kept: list[dict[str, Any]] = []
        for cand in ranked:
            pc = np.asarray(cand.get("point", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
            dc = _normalize(np.asarray(cand.get("direction", [0.0, 0.0, 1.0]), dtype=float).reshape(3))
            suppress = False
            for k in kept:
                pk = np.asarray(k.get("point", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
                dk = _normalize(np.asarray(k.get("direction", [0.0, 0.0, 1.0]), dtype=float).reshape(3))
                ang = _angle_deg(dc, dk)
                line_dist = _line_to_line_distance(pc, dc, pk, dk)
                if ang <= angle_thresh and line_dist <= line_thresh:
                    suppress = True
                    break
            if suppress:
                continue
            kept.append(cand)
            if len(kept) >= int(max_keep):
                break
        return kept

    def _extract_models_from_seeds(
        self,
        *,
        blobs: list[BlobRecord],
        points: np.ndarray,
        seeds: list[dict[str, Any]],
        cfg: dict[str, Any],
        active_weights: np.ndarray,
        max_models: int,
        start_index: int,
        min_inliers: int,
        min_len: float,
        min_mass: float,
        peel_factor: float,
        use_count: np.ndarray,
        max_reuse: int,
    ) -> list[ShankModel]:
        if not seeds:
            return []
        models: list[ShankModel] = []
        while len(models) < max_models and int(np.count_nonzero((active_weights > 1e-6) & (use_count < max_reuse))) >= min_inliers:
            best = None
            for seed in seeds:
                p0 = np.asarray(seed["point"], dtype=float).reshape(3)
                d = _normalize(np.asarray(seed["direction"], dtype=float).reshape(3))
                stat = _supports_for_line(points, blobs, p0, d, cfg, active_weights=active_weights)
                inlier = np.asarray(stat["inlier"], dtype=bool)
                count = int(np.count_nonzero(inlier))
                if count < min_inliers:
                    continue
                if float(stat["span_mm"]) < min_len:
                    continue
                mass = float(stat["mass_total"])
                if mass < min_mass:
                    continue
                score = mass + 0.08 * float(stat["span_mm"]) + 0.02 * float(seed.get("score", 0.0))
                if best is None or score > best[0]:
                    best = (score, p0, d, stat)

            if best is None:
                break

            _, p0, _d0, stat = best
            inlier = np.asarray(stat["inlier"], dtype=bool)
            idx = np.where(inlier)[0].astype(int)
            if idx.size < min_inliers:
                break

            center, axis = _weighted_pca_line(points[idx], np.maximum(active_weights[idx], 1e-6))
            proj = (points[idx] - center.reshape(1, 3)) @ axis.reshape(3)
            t_min = float(np.min(proj))
            t_max = float(np.max(proj))
            model = ShankModel(
                shank_id=f"EM{start_index + len(models):02d}",
                kind="line",
                params={
                    "point_ras": [float(center[0]), float(center[1]), float(center[2])],
                    "direction_ras": [float(axis[0]), float(axis[1]), float(axis[2])],
                    "t_min": t_min,
                    "t_max": t_max,
                },
                support={
                    "support_mass": float(stat["mass_total"]),
                    "bead_support_mass": float(stat["mass_bead"]),
                    "segment_support_mass": float(stat["mass_segment"]),
                    "span_mm": float(stat["span_mm"]),
                    "residual_mm": float(np.mean(stat["dist"][inlier])) if idx.size else 0.0,
                },
                assigned_blob_ids=tuple(int(blobs[i].blob_id) for i in idx),
            )
            models.append(model)
            use_count[idx] += 1
            active_weights[idx] *= peel_factor
        return models

    def _coverage_metrics(self, blobs: list[BlobRecord], models: list[ShankModel]) -> dict[str, Any]:
        total_support = float(
            np.sum([float(b.scores.get("support_weight", 1.0)) * max(0.0, 1.0 - 0.35 * float(b.scores.get("p_junk", 0.0))) for b in blobs])
        )
        by_id = {int(b.blob_id): b for b in blobs}
        assigned_ids: set[int] = set()
        for m in models:
            assigned_ids.update(int(v) for v in m.assigned_blob_ids)
        first_assigned_support = 0.0
        unexplained_high_bead = 0
        unexplained_high_segment = 0
        for bid, b in by_id.items():
            if bid in assigned_ids:
                first_assigned_support += float(b.scores.get("support_weight", 1.0))
            else:
                if float(b.scores.get("p_bead", 0.0)) >= 0.45:
                    unexplained_high_bead += 1
                if float(b.scores.get("p_segment", 0.0)) >= 0.45:
                    unexplained_high_segment += 1
        coverage = float(first_assigned_support / max(1e-9, total_support))
        return {
            "total_support_mass": float(total_support),
            "first_pass_assigned_support_mass": float(first_assigned_support),
            "coverage_ratio": float(coverage),
            "unexplained_high_bead": int(unexplained_high_bead),
            "unexplained_high_segment": int(unexplained_high_segment),
            "assigned_blob_ids": assigned_ids,
        }

    def initialize(self, ctx: DetectionContext, blobs: list[BlobRecord], state: dict[str, Any]) -> list[ShankModel]:
        if not blobs:
            return []
        cfg = dict(ctx.get("config") or {})
        points = np.asarray([b.centroid_ras for b in blobs], dtype=float).reshape(-1, 3)
        n = points.shape[0]
        if n < 2:
            return []

        max_lines = int(cfg.get("max_lines", 30))
        min_inliers = int(cfg.get("seed_min_inliers", max(2, int(round(float(cfg.get("min_inliers", 250)) * 0.012)))))
        min_len = float(cfg.get("min_length_mm", 20.0))
        min_mass = float(cfg.get("seed_min_support_mass", 1.2))
        peel_factor = float(np.clip(cfg.get("seed_peel_factor", 0.25), 0.05, 0.90))

        active_weights = np.asarray([float(b.scores.get("support_weight", 1.0)) for b in blobs], dtype=float)
        use_count = np.zeros((n,), dtype=np.int32)
        max_reuse = int(cfg.get("seed_max_point_reuse", 3))

        seg_mass = float(np.sum([float(b.scores.get("p_segment", 0.0)) for b in blobs]))
        bead_mass = float(np.sum([float(b.scores.get("p_bead", 0.0)) for b in blobs]))

        segment_raw = self._segment_seeds(blobs, points, cfg)
        segment_nms = self._nms_seeds(segment_raw, cfg, max_keep=max(10, 4 * max_lines))

        bead_raw: list[dict[str, Any]] = []
        strong_segment_evidence = (seg_mass >= 0.6 * bead_mass) and (len(segment_raw) >= max(4, max_lines // 5))
        if not strong_segment_evidence:
            bead_raw = self._bead_pair_seeds(blobs, points, cfg)
        bead_nms = self._nms_seeds(bead_raw, cfg, max_keep=max(8, 2 * max_lines))

        first_pass_seeds = list(segment_nms) + list(bead_nms)

        models_first = self._extract_models_from_seeds(
            blobs=blobs,
            points=points,
            seeds=first_pass_seeds,
            cfg=cfg,
            active_weights=active_weights.copy(),
            max_models=max_lines,
            start_index=1,
            min_inliers=min_inliers,
            min_len=min_len,
            min_mass=min_mass,
            peel_factor=peel_factor,
            use_count=use_count.copy(),
            max_reuse=max_reuse,
        )

        coverage = self._coverage_metrics(blobs, models_first)
        coverage_min = float(cfg.get("coverage_min_ratio_for_no_rescue", 0.60))
        rescue_min_p_bead = float(cfg.get("rescue_only_unassigned_min_p_bead", 0.45))
        bead_rescue_triggered = bool(coverage["coverage_ratio"] < coverage_min)

        models_second: list[ShankModel] = []
        bead_rescue_nms: list[dict[str, Any]] = []
        if bead_rescue_triggered and len(models_first) < max_lines:
            assigned = set(coverage["assigned_blob_ids"])
            residual_idx = [
                i
                for i, b in enumerate(blobs)
                if int(b.blob_id) not in assigned and float(b.scores.get("p_bead", 0.0)) >= rescue_min_p_bead
            ]
            rescue_raw = self._bead_pair_seeds(
                blobs,
                points,
                cfg,
                candidate_indices=residual_idx,
                max_raw_override=int(cfg.get("rescue_max_bead_seeds", max(12, 2 * max_lines))),
            )
            bead_rescue_nms = self._nms_seeds(rescue_raw, cfg, max_keep=int(cfg.get("rescue_max_bead_seeds", max(12, 2 * max_lines))))

            residual_weights = np.zeros((n,), dtype=float)
            for i in residual_idx:
                residual_weights[int(i)] = float(blobs[int(i)].scores.get("support_weight", 1.0))

            models_second = self._extract_models_from_seeds(
                blobs=blobs,
                points=points,
                seeds=bead_rescue_nms,
                cfg=cfg,
                active_weights=residual_weights,
                max_models=max(0, max_lines - len(models_first)),
                start_index=len(models_first) + 1,
                min_inliers=min_inliers,
                min_len=min_len,
                min_mass=min_mass,
                peel_factor=peel_factor,
                use_count=np.zeros((n,), dtype=np.int32),
                max_reuse=max_reuse,
            )

        models: list[ShankModel] = list(models_first) + list(models_second)

        state["effective_min_inliers"] = int(min_inliers)
        state["seed_stats"] = {
            "seed_count_segment_raw": int(len(segment_raw)),
            "seed_count_segment_after_nms": int(len(segment_nms)),
            "seed_count_bead_raw": int(len(bead_raw)),
            "seed_count_bead_after_nms": int(len(bead_nms)),
            "seed_count_segment": int(len(segment_nms)),
            "seed_count_bead_pair": int(len(bead_nms)),
            "strong_segment_evidence": bool(strong_segment_evidence),
            "seed_pool_count": int(len(first_pass_seeds)),
            "total_support_mass": float(coverage["total_support_mass"]),
            "first_pass_assigned_support_mass": float(coverage["first_pass_assigned_support_mass"]),
            "coverage_ratio": float(coverage["coverage_ratio"]),
            "bead_rescue_triggered": int(1 if bead_rescue_triggered else 0),
            "seed_count_bead_rescue": int(len(bead_rescue_nms)),
            "shanks_after_first_pass": int(len(models_first)),
            "shanks_after_second_pass": int(len(models)),
        }
        return models


class _EMRefiner:
    def _line_score_matrix(
        self,
        points: np.ndarray,
        blobs: list[BlobRecord],
        models: list[ShankModel],
        cfg: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = points.shape[0]
        k = len(models)
        if n == 0 or k == 0:
            return np.zeros((n, k), dtype=float), np.zeros((n, k), dtype=float), np.zeros((n, k), dtype=float)

        p_bead = np.asarray([float(b.scores.get("p_bead", 0.0)) for b in blobs], dtype=float)
        p_segment = np.asarray([float(b.scores.get("p_segment", 0.0)) for b in blobs], dtype=float)
        p_junk = np.asarray([float(b.scores.get("p_junk", 0.0)) for b in blobs], dtype=float)
        w = np.asarray([float(b.scores.get("support_weight", 1.0)) for b in blobs], dtype=float)
        depth_prior = np.asarray([float(b.scores.get("depth_prior", 1.0)) for b in blobs], dtype=float)
        axes = _blob_axis_array(blobs)

        sigma_d = float(cfg.get("inlier_radius_mm", 1.2)) * float(cfg.get("refine_radius_scale", 1.4))
        sigma_a = float(cfg.get("segment_angle_sigma_deg", 12.0))

        score = np.zeros((n, k), dtype=float)
        dist_mat = np.zeros((n, k), dtype=float)
        ang_mat = np.zeros((n, k), dtype=float)

        for j, m in enumerate(models):
            p0 = np.asarray(m.params.get("point_ras", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
            d = _normalize(np.asarray(m.params.get("direction_ras", [0.0, 0.0, 1.0]), dtype=float).reshape(3))
            dist = _line_distances(points, p0, d)
            dist_like = _soft_gaussian(dist, sigma_d)
            cosv = np.clip(np.abs(axes @ d.reshape(3)), 0.0, 1.0)
            ang = np.degrees(np.arccos(cosv))
            ang_like = _soft_gaussian(ang, sigma_a)
            mixed = (p_bead * dist_like) + (p_segment * dist_like * ang_like)
            score[:, j] = mixed * w * depth_prior * (1.0 - 0.65 * p_junk)
            dist_mat[:, j] = dist
            ang_mat[:, j] = ang

        return score, dist_mat, ang_mat

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
        iterations = int(cfg.get("refine_iterations", 5))
        min_delta = float(cfg.get("refine_min_objective_delta", 1e-4))
        assign_min = float(cfg.get("refine_assign_min_prob", 0.03))

        current = list(models)
        objective_trace: list[float] = []
        prev_obj = None

        for _ in range(max(1, iterations)):
            score, dist_mat, _ang_mat = self._line_score_matrix(points, blobs, current, cfg)
            if score.size == 0:
                break

            p_junk = np.asarray([float(b.scores.get("p_junk", 0.0)) for b in blobs], dtype=float)
            outlier = np.clip(0.01 + 0.15 * p_junk, 0.01, 0.25).reshape(-1, 1)
            z = np.sum(score, axis=1, keepdims=True) + outlier
            z = np.maximum(z, 1e-12)
            resp = score / z
            obj = float(np.sum(np.log(z.reshape(-1))))
            objective_trace.append(obj)
            if prev_obj is not None and (obj < prev_obj - 1e-6):
                break
            if prev_obj is not None and abs(obj - prev_obj) < min_delta:
                break
            prev_obj = obj

            updated: list[ShankModel] = []
            winner = np.argmax(resp, axis=1)
            winner_prob = resp[np.arange(resp.shape[0]), winner]

            for j, m in enumerate(current):
                hard_mask = (winner == j) & (winner_prob >= assign_min)
                # Keep soft weighting for robustness, but suppress cross-line leakage.
                wj = resp[:, j].copy()
                wj[~hard_mask] *= 0.40
                mass = float(np.sum(wj))
                if mass <= 1e-6:
                    updated.append(m)
                    continue

                center, axis = _weighted_pca_line(points, wj)
                proj = (points - center.reshape(1, 3)) @ axis.reshape(3)
                keep = wj >= (0.25 * float(np.max(wj)))
                if not np.any(keep):
                    keep = wj > 0.0
                t_min = float(np.min(proj[keep])) if np.any(keep) else float(np.min(proj))
                t_max = float(np.max(proj[keep])) if np.any(keep) else float(np.max(proj))

                blob_ids = tuple(int(blobs[i].blob_id) for i in np.where(hard_mask)[0])
                bead_mass = float(np.sum(wj * np.asarray([float(b.scores.get("p_bead", 0.0)) for b in blobs], dtype=float)))
                seg_mass = float(np.sum(wj * np.asarray([float(b.scores.get("p_segment", 0.0)) for b in blobs], dtype=float)))
                residual = float(np.average(dist_mat[:, j], weights=np.maximum(wj, 1e-6)))

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
                            "support_mass": float(mass),
                            "bead_support_mass": float(bead_mass),
                            "segment_support_mass": float(seg_mass),
                            "span_mm": float(max(0.0, t_max - t_min)),
                            "residual_mm": float(residual),
                        },
                        assigned_blob_ids=blob_ids,
                    )
                )
            current = updated

        payload = {
            "iterations": int(max(1, iterations)),
            "objective_trace": [float(v) for v in objective_trace],
            "final_objective": float(objective_trace[-1] if objective_trace else 0.0),
        }
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
        min_mass = float(cfg.get("selector_min_support_mass", 1.6))
        overlap_thresh = float(cfg.get("selector_overlap_threshold", 0.75))
        angle_thresh = float(cfg.get("selector_angle_threshold_deg", 12.0))
        line_dist_thresh = float(cfg.get("selector_line_distance_mm", 1.2))

        reject_counts = {
            "start_zone_reject_count": 0,
            "length_reject_count": 0,
            "inlier_reject_count": 0,
            "model_reject_count": 0,
        }

        kept = []
        for m in models:
            span = float(m.support.get("span_mm", 0.0))
            mass = float(m.support.get("support_mass", 0.0))
            if span < min_len:
                reject_counts["length_reject_count"] += 1
                continue
            if mass < min_mass:
                reject_counts["inlier_reject_count"] += 1
                continue
            kept.append(m)

        # Overlap/angle dedup.
        final: list[ShankModel] = []
        kept_sorted = sorted(
            kept,
            key=lambda m: (
                float(m.support.get("support_mass", 0.0)),
                float(m.support.get("segment_support_mass", 0.0)),
                float(m.support.get("span_mm", 0.0)),
            ),
            reverse=True,
        )

        for m in kept_sorted:
            ids_m = set(int(v) for v in m.assigned_blob_ids)
            dm = _normalize(np.asarray(m.params.get("direction_ras", [0.0, 0.0, 1.0]), dtype=float).reshape(3))
            clash = False
            for q in final:
                ids_q = set(int(v) for v in q.assigned_blob_ids)
                if not ids_m:
                    overlap = 0.0
                else:
                    overlap = float(len(ids_m & ids_q)) / float(max(1, len(ids_m)))
                dq = _normalize(np.asarray(q.params.get("direction_ras", [0.0, 0.0, 1.0]), dtype=float).reshape(3))
                ang = _angle_deg(dm, dq)
                pm = np.asarray(m.params.get("point_ras", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
                pq = np.asarray(q.params.get("point_ras", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
                line_dist = _line_to_line_distance(pm, dm, pq, dq)
                if (overlap >= overlap_thresh and ang <= angle_thresh) or (
                    overlap >= 0.25 and ang <= angle_thresh and line_dist <= line_dist_thresh
                ):
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
        # Shank-only pass by design.
        return []


class BlobEMV2Pipeline(BaseDetectionPipeline):
    """Second-generation shank detector with blob mixed-evidence refinement."""

    pipeline_id = "blob_em_v2"
    display_name = "Blob EM v2"
    scaffold = False
    pipeline_version = "0.3.0"

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
            d = _normalize(np.asarray(m.params.get("direction_ras", [0.0, 0.0, 1.0]), dtype=float).reshape(3))
            t0 = float(m.params.get("t_min", -10.0))
            t1 = float(m.params.get("t_max", 10.0))
            a = p + t0 * d
            b = p + t1 * d
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
                    "confidence": float(min(1.0, max(0.0, float(m.support.get("support_mass", 0.0)) / 25.0))),
                    "support_count": int(len(m.assigned_blob_ids)),
                    "support_mass": float(m.support.get("support_mass", 0.0)),
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
            result["warnings"].append("contact_detection_not_implemented")
            return self.finalize(result, diagnostics, t_start)

        state: dict[str, Any] = {}
        blobs: list[BlobRecord] = []
        models: list[ShankModel] = []

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
            diagnostics.set_extra("blob_source", str(state.get("blob_source", "")))
            filt = dict(state.get("blob_filter") or {})
            diagnostics.set_count("blob_count_total", int((state.get("blob_raw") or {}).get("blob_count_total", len(blobs))))
            diagnostics.set_count("blob_count_kept", int(len(blobs)))
            diagnostics.set_count("blobs_total", int((state.get("blob_raw") or {}).get("blob_count_total", len(blobs))))
            diagnostics.set_count("blob_reject_small", int(filt.get("blob_reject_small", 0)))
            diagnostics.set_count("blob_reject_large", int(filt.get("blob_reject_large", 0)))
            diagnostics.set_count("blob_reject_intensity", int(filt.get("blob_reject_intensity", 0)))
            diagnostics.set_count("blob_reject_shape", int(filt.get("blob_reject_shape", 0)))
            for k, v in dict(state.get("support_stats") or {}).items():
                diagnostics.set_count(str(k), int(v))
            diagnostics.set_extra("support_sampling", dict(state.get("support_sampling") or {}))

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
            diagnostics.set_extra("seed_stats", dict(state.get("seed_stats") or {}))
            seed_stats = dict(state.get("seed_stats") or {})
            diagnostics.set_count("seed_count_segment", int(seed_stats.get("seed_count_segment", 0)))
            diagnostics.set_count("seed_count_bead_pair", int(seed_stats.get("seed_count_bead_pair", 0)))
            diagnostics.set_count("seed_count_segment_raw", int(seed_stats.get("seed_count_segment_raw", 0)))
            diagnostics.set_count("seed_count_segment_after_nms", int(seed_stats.get("seed_count_segment_after_nms", 0)))
            diagnostics.set_count("seed_count_bead_raw", int(seed_stats.get("seed_count_bead_raw", 0)))
            diagnostics.set_count("seed_count_bead_after_nms", int(seed_stats.get("seed_count_bead_after_nms", 0)))
            diagnostics.set_count("bead_rescue_triggered", int(seed_stats.get("bead_rescue_triggered", 0)))
            diagnostics.set_count("seed_count_bead_rescue", int(seed_stats.get("seed_count_bead_rescue", 0)))
            diagnostics.set_count("shanks_after_first_pass", int(seed_stats.get("shanks_after_first_pass", 0)))
            diagnostics.set_count("shanks_after_second_pass", int(seed_stats.get("shanks_after_second_pass", 0)))
            diagnostics.set_extra(
                "coverage_metrics",
                {
                    "first_pass_assigned_support_mass": float(seed_stats.get("first_pass_assigned_support_mass", 0.0)),
                    "total_support_mass": float(seed_stats.get("total_support_mass", 0.0)),
                    "coverage_ratio": float(seed_stats.get("coverage_ratio", 0.0)),
                },
            )
            diagnostics.set_count("fit1_lines_proposed", int(len(models)))
            diagnostics.set_count("effective_min_inliers", int(state.get("effective_min_inliers", 0)))
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
            diagnostics.set_count("fit2_lines_kept", int(len(models)))
            diagnostics.set_count("shanks_before_select", int(len(models)))

            models = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="model_selection",
                fn=lambda: selector.select(ctx, blobs, models, state),
            )
            diagnostics.set_count("final_lines_kept", int(len(models)))
            diagnostics.set_count("shanks_final", int(len(models)))
            for k, v in dict(state.get("selector_reject_counts") or {}).items():
                diagnostics.set_count(str(k), int(v))

            _ = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="contact_detection",
                fn=lambda: contact_detector.detect(ctx, blobs, models, state),
            )

            center_raw = ctx.get("center_ras")
            if center_raw is None:
                center_raw = [0.0, 0.0, 0.0]
            center_ras = np.asarray(center_raw, dtype=float).reshape(3)
            trajectories = self._models_to_trajectories(models, center_ras=center_ras)
            result["trajectories"] = trajectories
            result["contacts"] = []
            result["warnings"].append("contact_detection_not_implemented")

            if blobs:
                points = np.asarray([b.centroid_ras for b in blobs], dtype=float).reshape(-1, 3)
                assigned_ids = set()
                for m in models:
                    assigned_ids.update(int(v) for v in m.assigned_blob_ids)
                blob_ids = [int(b.blob_id) for b in blobs]
                assigned = sum(1 for bid in blob_ids if bid in assigned_ids)
                diagnostics.set_count("assigned_points_after_refine", int(assigned))
                diagnostics.set_count("final_unassigned_points", int(max(0, len(blobs) - assigned)))
            else:
                diagnostics.set_count("assigned_points_after_refine", 0)
                diagnostics.set_count("final_unassigned_points", 0)

            # UI compatibility: provide legacy-shaped payload for PostopCTLocalization.
            extras = ctx.get("extras")
            if isinstance(extras, dict):
                preview = dict(state.get("preview") or {})
                blob_raw = dict(state.get("blob_raw") or {})
                blob_filter = dict(state.get("blob_filter") or {})
                labels_kji = blob_raw.get("labels_kji")
                kept_blob_ids = list(blob_filter.get("kept_blob_ids") or [])
                blob_kept_mask_kji = None
                blob_rejected_mask_kji = None
                if labels_kji is not None:
                    labels_arr = np.asarray(labels_kji, dtype=np.int32)
                    if labels_arr.size > 0:
                        all_blob_mask = labels_arr > 0
                        if kept_blob_ids:
                            keep_set = np.asarray([int(v) for v in kept_blob_ids], dtype=np.int32)
                            blob_kept_mask = np.isin(labels_arr, keep_set)
                        else:
                            blob_kept_mask = np.zeros(labels_arr.shape, dtype=bool)
                        blob_rejected_mask = np.logical_and(all_blob_mask, np.logical_not(blob_kept_mask))
                        blob_kept_mask_kji = np.asarray(blob_kept_mask, dtype=np.uint8)
                        blob_rejected_mask_kji = np.asarray(blob_rejected_mask, dtype=np.uint8)
                support_centroids_ras = np.asarray(
                    [list(b.centroid_ras) for b in blobs],
                    dtype=float,
                ).reshape(-1, 3)
                rejected_raw = blob_filter.get("blob_centroids_rejected_ras")
                if rejected_raw is None:
                    rejected_raw = np.empty((0, 3), dtype=float)
                prefilter_rejected_centroids_ras = np.asarray(
                    rejected_raw,
                    dtype=float,
                ).reshape(-1, 3)
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
                    "effective_inlier_radius_mm": float(cfg.get("inlier_radius_mm", 1.2)),
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
                    "blobs_total_cc": int(diagnostics.diagnostics.get("counts", {}).get("blobs_total_cc", 0)),
                    "blobs_eligible_for_axial_sampling": int(
                        diagnostics.diagnostics.get("counts", {}).get("blobs_eligible_for_axial_sampling", 0)
                    ),
                    "axial_support_samples_generated": int(
                        diagnostics.diagnostics.get("counts", {}).get("axial_support_samples_generated", 0)
                    ),
                    "support_points_total": int(diagnostics.diagnostics.get("counts", {}).get("support_points_total", 0)),
                    "support_points_from_compact_blobs": int(
                        diagnostics.diagnostics.get("counts", {}).get("support_points_from_compact_blobs", 0)
                    ),
                    "support_points_from_elongated_blobs": int(
                        diagnostics.diagnostics.get("counts", {}).get("support_points_from_elongated_blobs", 0)
                    ),
                    "support_sampling": diagnostics.diagnostics.get("extras", {}).get("support_sampling", {}),
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
                    # Use support centroids for display/debug so elongated parent
                    # blobs are represented by axial support samples.
                    "blob_centroids_all_ras": support_centroids_ras,
                    "blob_centroids_kept_ras": support_centroids_ras,
                    # Rejected centroids are still from pre-filter rejects.
                    "blob_centroids_rejected_ras": prefilter_rejected_centroids_ras,
                    "blob_kept_mask_kji": blob_kept_mask_kji,
                    "blob_rejected_mask_kji": blob_rejected_mask_kji,
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
                    "q95_hu": b.q95_hu,
                    "elongation": b.elongation,
                    "p_bead": float(b.scores.get("p_bead", 0.0)),
                    "p_segment": float(b.scores.get("p_segment", 0.0)),
                    "p_junk": float(b.scores.get("p_junk", 0.0)),
                    "support_weight": float(b.scores.get("support_weight", 0.0)),
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
