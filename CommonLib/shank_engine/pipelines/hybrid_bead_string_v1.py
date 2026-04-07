"""Hybrid bead+string shank detector.

This pipeline uses multi-threshold blob lineages to derive three primitive
observation types:
- bead-like compact hyperdensities
- string-like elongated cores
- junction-like merge-heavy artifacts

Shank hypotheses are generated primarily from bead-pair chains, then reinforced
with string observations and downweighted by nearby junction observations.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from rosa_core.electrode_models import load_electrode_library
from shank_core.masking import build_preview_masks

from ..artifacts import add_artifact, write_standard_artifacts
from ..contracts import DetectionContext, DetectionResult
from ..lineage_tracking import build_lineages, extract_threshold_levels, summarize_lineages
from .base import BaseDetectionPipeline


@dataclass(frozen=True)
class _Primitive:
    primitive_type: str
    lineage_id: int
    node_index: int
    point_ras: tuple[float, float, float]
    axis_ras: tuple[float, float, float]
    score: float
    length_mm: float
    diameter_mm: float
    depth_mm: float
    threshold_hu: float
    meta: dict[str, Any]


@dataclass(frozen=True)
class _Seed:
    point_ras: tuple[float, float, float]
    direction_ras: tuple[float, float, float]
    score: float
    bead_i: int
    bead_j: int
    chain_indices: tuple[int, ...] = ()
    best_model_id: str | None = None


@dataclass(frozen=True)
class _ElectrodePrior:
    model_id: str
    pitch_mm: float
    diameter_mm: float
    total_length_mm: float
    contact_count: int


def _normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-9:
        return np.asarray([0.0, 0.0, 1.0], dtype=float)
    return arr / norm


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    aa = _normalize(np.asarray(a, dtype=float))
    bb = _normalize(np.asarray(b, dtype=float))
    dot = float(abs(np.dot(aa, bb)))
    dot = min(1.0, max(-1.0, dot))
    return float(np.degrees(np.arccos(dot)))


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


def _line_distance(p0: np.ndarray, d0: np.ndarray, p1: np.ndarray, d1: np.ndarray) -> float:
    u = _normalize(d0)
    v = _normalize(d1)
    w0 = p0 - p1
    c = np.cross(u, v)
    cn = float(np.linalg.norm(c))
    if cn <= 1e-6:
        return float(np.linalg.norm(np.cross(w0, u)))
    return float(abs(np.dot(w0, c)) / cn)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _center_pitch_from_model(model: dict[str, Any]) -> float | None:
    c2c = model.get("center_to_center_separation_mm")
    if c2c is not None:
        val = _safe_float(c2c, 0.0)
        if val > 0.0:
            return val
    offsets = list(model.get("contact_center_offsets_from_tip_mm") or [])
    if len(offsets) >= 2:
        diffs = np.diff(np.asarray(offsets, dtype=float))
        if diffs.size:
            return float(np.median(diffs))
    return None


def _load_family_priors() -> list[_ElectrodePrior]:
    library = load_electrode_library()
    priors: list[_ElectrodePrior] = []
    for model in list(library.get("models") or []):
        pitch = _center_pitch_from_model(model)
        if pitch is None or pitch <= 0.0:
            continue
        priors.append(
            _ElectrodePrior(
                model_id=str(model.get("id") or "unknown"),
                pitch_mm=float(pitch),
                diameter_mm=float(_safe_float(model.get("diameter_mm"), 0.8)),
                total_length_mm=float(_safe_float(model.get("total_exploration_length_mm"), 0.0)),
                contact_count=int(model.get("contact_count") or 0),
            )
        )
    return priors


def _chain_geometry_compatibility(
    projections_mm: np.ndarray,
    diameters_mm: np.ndarray,
    priors: list[_ElectrodePrior],
) -> tuple[float, str | None]:
    proj = np.sort(np.asarray(projections_mm, dtype=float).reshape(-1))
    if proj.size < 2 or not priors:
        return 0.0, None
    raw_gaps = np.diff(proj)
    gaps = raw_gaps[np.logical_and(raw_gaps >= 1.0, raw_gaps <= 8.0)]
    if gaps.size == 0:
        return 0.0, None
    median_gap = float(np.median(gaps))
    span = float(np.max(proj) - np.min(proj))
    obs_count = int(proj.size)
    mean_diam = float(np.mean(np.asarray(diameters_mm, dtype=float).reshape(-1))) if np.asarray(diameters_mm).size else 0.8
    best_score = 0.0
    best_id: str | None = None
    for prior in priors:
        gap_score = 1.0 - _clamp01(abs(median_gap - prior.pitch_mm) / max(1.0, 0.6 * prior.pitch_mm))
        span_score = 1.0 - _clamp01(abs(span - prior.total_length_mm) / max(8.0, 0.8 * prior.total_length_mm))
        diam_score = 1.0 - _clamp01(abs(mean_diam - prior.diameter_mm) / 2.5)
        count_score = _clamp01(float(obs_count) / float(max(2, min(prior.contact_count, 6))))
        score = 0.45 * gap_score + 0.25 * span_score + 0.15 * diam_score + 0.15 * count_score
        if score > best_score:
            best_score = float(score)
            best_id = str(prior.model_id)
    return float(best_score), best_id


class HybridBeadStringV1Pipeline(BaseDetectionPipeline):
    pipeline_id = "hybrid_bead_string_v1"
    display_name = "Hybrid Bead+String v1"
    pipeline_version = "1.0.0"

    def _build_gating(self, ctx: DetectionContext, cfg: dict[str, Any]) -> dict[str, Any]:
        arr_kji = np.asarray(ctx["arr_kji"])
        spacing_xyz = tuple(ctx.get("spacing_xyz") or (1.0, 1.0, 1.0))
        threshold = float(max(list(cfg.get("threshold_schedule_hu", [2600.0, 2200.0, 1800.0, 1500.0, 1200.0]))))
        preview = build_preview_masks(
            arr_kji=arr_kji,
            spacing_xyz=spacing_xyz,
            threshold=threshold,
            use_head_mask=bool(cfg.get("use_head_mask", True)),
            build_head_mask=bool(cfg.get("build_head_mask", True)),
            head_mask_threshold_hu=float(cfg.get("head_mask_threshold_hu", -500.0)),
            head_mask_method=str(cfg.get("head_mask_method", "outside_air")),
            head_gate_erode_vox=int(cfg.get("head_gate_erode_vox", 1)),
            head_gate_dilate_vox=int(cfg.get("head_gate_dilate_vox", 1)),
            head_gate_margin_mm=float(cfg.get("head_gate_margin_mm", 0.0)),
            min_metal_depth_mm=float(cfg.get("min_metal_depth_mm", 5.0)),
            max_metal_depth_mm=float(cfg.get("max_metal_depth_mm", 220.0)),
            include_debug_masks=False,
        )
        return {
            "gating_mask_kji": np.asarray(preview.get("gating_mask_kji"), dtype=bool),
            "head_distance_map_kji": np.asarray(preview.get("head_distance_map_kji"), dtype=np.float32) if preview.get("head_distance_map_kji") is not None else None,
            "metal_in_head_mask_kji": np.asarray(preview.get("metal_in_head_mask_kji"), dtype=bool) if preview.get("metal_in_head_mask_kji") is not None else None,
        }

    def _lineage_state(self, ctx: DetectionContext, gating: dict[str, Any], cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        thresholds = [float(v) for v in list(cfg.get("threshold_schedule_hu", [2600.0, 2200.0, 1800.0, 1500.0, 1200.0]))]
        levels = extract_threshold_levels(
            arr_kji=np.asarray(ctx["arr_kji"], dtype=float),
            gating_mask_kji=np.asarray(gating.get("gating_mask_kji"), dtype=bool),
            depth_map_kji=gating.get("head_distance_map_kji"),
            thresholds_hu=thresholds,
            ijk_kji_to_ras_fn=ctx["ijk_kji_to_ras_fn"],
        )
        lineages = build_lineages(levels)
        summaries = summarize_lineages(lineages, total_levels=len(levels))
        return lineages, summaries

    def _best_string_node(self, lineage: dict[str, Any], summary: dict[str, Any], cfg: dict[str, Any]) -> tuple[int, dict[str, Any]] | None:
        min_depth_mm = float(cfg.get("string_min_depth_mm", 3.0))
        best: tuple[float, int, dict[str, Any]] | None = None
        for idx, node in enumerate(list(lineage.get("nodes") or [])):
            blob = dict(node.get("blob") or {})
            length = float(blob.get("length_mm") or 0.0)
            diameter = float(blob.get("diameter_mm") or 0.0)
            depth = float(blob.get("depth_mean") or 0.0)
            if length < float(cfg.get("string_min_length_mm", 6.0)):
                continue
            if diameter <= 0.0 or diameter > float(cfg.get("string_max_diameter_mm", 5.0)):
                continue
            if depth < min_depth_mm:
                continue
            slender = length / max(diameter, 0.25)
            score = (
                0.45 * float(summary.get("p_core", 0.0))
                + 0.20 * _clamp01((length - 6.0) / 20.0)
                + 0.15 * _clamp01((slender - 3.0) / 10.0)
                + 0.10 * (1.0 - _clamp01(abs(diameter - 1.2) / 3.0))
                + 0.10 * _clamp01((depth - min_depth_mm) / 8.0)
                - 0.20 * float(summary.get("p_junction", 0.0))
            )
            if best is None or score > best[0]:
                best = (score, idx, blob)
        if best is None or best[0] < float(cfg.get("string_min_score", 0.20)):
            return None
        return int(best[1]), best[2]

    def _bead_nodes(self, lineage: dict[str, Any], summary: dict[str, Any], cfg: dict[str, Any]) -> list[tuple[int, dict[str, Any], float]]:
        out: list[tuple[int, dict[str, Any], float]] = []
        for idx, node in enumerate(list(lineage.get("nodes") or [])):
            blob = dict(node.get("blob") or {})
            length = float(blob.get("length_mm") or 0.0)
            diameter = float(blob.get("diameter_mm") or 0.0)
            depth = float(blob.get("depth_mean") or 0.0)
            hu_q95 = float(blob.get("hu_q95") or blob.get("hu_max") or 0.0)
            vox = float(blob.get("voxel_count") or 0.0)
            if depth < float(cfg.get("bead_min_depth_mm", 0.0)):
                continue
            compact_term = 1.0 - _clamp01((length - 5.0) / 8.0)
            diameter_term = 1.0 - _clamp01(abs(diameter - 1.5) / 3.5)
            hu_term = _clamp01((hu_q95 - 1200.0) / 1800.0)
            voxel_term = _clamp01((vox - 1.0) / 20.0)
            score = (
                0.40 * float(summary.get("p_contact", 0.0))
                + 0.20 * compact_term
                + 0.15 * diameter_term
                + 0.15 * hu_term
                + 0.10 * voxel_term
                - 0.10 * float(summary.get("p_junction", 0.0))
            )
            if length > 10.0:
                score -= 0.15
            if diameter > 5.5:
                score -= 0.20
            if score < float(cfg.get("bead_min_score", 0.25)):
                continue
            out.append((int(idx), blob, float(score)))
        return out

    def _junction_node(self, lineage: dict[str, Any], summary: dict[str, Any], cfg: dict[str, Any]) -> tuple[int, dict[str, Any], float] | None:
        nodes = list(lineage.get("nodes") or [])
        if not nodes:
            return None
        best_idx = 0
        best_blob = dict(nodes[0].get("blob") or {})
        best_raw = float(best_blob.get("diameter_mm") or 0.0) + 0.5 * float(best_blob.get("length_mm") or 0.0)
        for idx, node in enumerate(nodes[1:], start=1):
            blob = dict(node.get("blob") or {})
            raw = float(blob.get("diameter_mm") or 0.0) + 0.5 * float(blob.get("length_mm") or 0.0)
            if raw > best_raw:
                best_idx = idx
                best_blob = blob
                best_raw = raw
        merge_term = _clamp01((float(summary.get("merge_count", 0)) + float(summary.get("split_count", 0))) / 3.0)
        growth_term = _clamp01(float(summary.get("diameter_growth_mm", 0.0)) / 5.0)
        axis_term = _clamp01(float(summary.get("mean_axis_change_deg", 0.0)) / 20.0)
        thick_term = _clamp01((float(best_blob.get("diameter_mm") or 0.0) - 2.5) / 4.0)
        score = 0.45 * float(summary.get("p_junction", 0.0)) + 0.20 * merge_term + 0.15 * growth_term + 0.10 * axis_term + 0.10 * thick_term
        if score < float(cfg.get("junction_min_score", 0.30)):
            return None
        return int(best_idx), best_blob, float(score)

    def _dedup_primitives(self, items: list[_Primitive], radius_mm: float) -> list[_Primitive]:
        kept: list[_Primitive] = []
        for obs in sorted(items, key=lambda x: (-float(x.score), x.lineage_id, x.node_index)):
            pt = np.asarray(obs.point_ras, dtype=float)
            if any(float(np.linalg.norm(pt - np.asarray(k.point_ras, dtype=float))) <= float(radius_mm) for k in kept):
                continue
            kept.append(obs)
        return kept

    def _build_primitives(self, lineages: list[dict[str, Any]], summaries: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, list[_Primitive]]:
        summary_by_id = {int(row["lineage_id"]): row for row in summaries}
        strings: list[_Primitive] = []
        beads: list[_Primitive] = []
        junctions: list[_Primitive] = []
        for lineage in lineages:
            lineage_id = int(lineage["lineage_id"])
            summary = dict(summary_by_id.get(lineage_id) or {})
            string_pick = self._best_string_node(lineage, summary, cfg)
            if string_pick is not None:
                node_index, blob = string_pick
                strings.append(_Primitive(
                    primitive_type="string",
                    lineage_id=lineage_id,
                    node_index=int(node_index),
                    point_ras=tuple(float(v) for v in list(blob.get("centroid_ras") or [0.0,0.0,0.0])[:3]),
                    axis_ras=tuple(float(v) for v in list(blob.get("pca_axis_ras") or [0.0,0.0,1.0])[:3]),
                    score=float(max(0.0, summary.get("p_core", 0.0) - 0.25 * summary.get("p_junction", 0.0))),
                    length_mm=float(blob.get("length_mm") or 0.0),
                    diameter_mm=float(blob.get("diameter_mm") or 0.0),
                    depth_mm=float(blob.get("depth_mean") or 0.0),
                    threshold_hu=float(lineage["nodes"][node_index]["threshold_hu"]),
                    meta={"p_core": float(summary.get("p_core", 0.0)), "p_contact": float(summary.get("p_contact", 0.0)), "p_junction": float(summary.get("p_junction", 0.0))},
                ))
            for node_index, blob, score in self._bead_nodes(lineage, summary, cfg):
                beads.append(_Primitive(
                    primitive_type="bead",
                    lineage_id=lineage_id,
                    node_index=int(node_index),
                    point_ras=tuple(float(v) for v in list(blob.get("centroid_ras") or [0.0,0.0,0.0])[:3]),
                    axis_ras=tuple(float(v) for v in list(blob.get("pca_axis_ras") or [0.0,0.0,1.0])[:3]),
                    score=float(score),
                    length_mm=float(blob.get("length_mm") or 0.0),
                    diameter_mm=float(blob.get("diameter_mm") or 0.0),
                    depth_mm=float(blob.get("depth_mean") or 0.0),
                    threshold_hu=float(lineage["nodes"][node_index]["threshold_hu"]),
                    meta={"p_core": float(summary.get("p_core", 0.0)), "p_contact": float(summary.get("p_contact", 0.0)), "p_junction": float(summary.get("p_junction", 0.0))},
                ))
            junction_pick = self._junction_node(lineage, summary, cfg)
            if junction_pick is not None:
                node_index, blob, score = junction_pick
                junctions.append(_Primitive(
                    primitive_type="junction",
                    lineage_id=lineage_id,
                    node_index=int(node_index),
                    point_ras=tuple(float(v) for v in list(blob.get("centroid_ras") or [0.0,0.0,0.0])[:3]),
                    axis_ras=tuple(float(v) for v in list(blob.get("pca_axis_ras") or [0.0,0.0,1.0])[:3]),
                    score=float(score),
                    length_mm=float(blob.get("length_mm") or 0.0),
                    diameter_mm=float(blob.get("diameter_mm") or 0.0),
                    depth_mm=float(blob.get("depth_mean") or 0.0),
                    threshold_hu=float(lineage["nodes"][node_index]["threshold_hu"]),
                    meta={"p_core": float(summary.get("p_core", 0.0)), "p_contact": float(summary.get("p_contact", 0.0)), "p_junction": float(summary.get("p_junction", 0.0))},
                ))
        return {
            "strings": self._dedup_primitives(strings, radius_mm=float(cfg.get("string_dedup_radius_mm", 3.0))),
            "beads": self._dedup_primitives(beads, radius_mm=float(cfg.get("bead_dedup_radius_mm", 2.5))),
            "junctions": self._dedup_primitives(junctions, radius_mm=float(cfg.get("junction_dedup_radius_mm", 4.0))),
        }

    def _pair_seed_score(self, p0: np.ndarray, p1: np.ndarray, beads: list[_Primitive], strings: list[_Primitive], junctions: list[_Primitive], cfg: dict[str, Any]) -> float:
        priors = list(cfg.get("_electrode_family_priors") or [])
        d = _normalize(p1 - p0)
        dist = float(np.linalg.norm(p1 - p0))
        bead_points = np.asarray([b.point_ras for b in beads], dtype=float).reshape(-1, 3)
        bead_scores = np.asarray([float(b.score) for b in beads], dtype=float).reshape(-1)
        bead_diams = np.asarray([float(b.diameter_mm) for b in beads], dtype=float).reshape(-1)
        dd, tt = _line_distances(bead_points, p0, d)
        bead_mask = np.logical_and(dd <= float(cfg.get("bead_support_radius_mm", 2.5)), np.logical_and(tt >= -3.0, tt <= dist + 3.0))
        bead_mass = float(np.sum(bead_scores[bead_mask]))
        geom_score = 0.0
        if np.any(bead_mask):
            geom_score, _model_id = _chain_geometry_compatibility(tt[bead_mask], bead_diams[bead_mask], priors)

        string_mass = 0.0
        if strings:
            string_points = np.asarray([s.point_ras for s in strings], dtype=float).reshape(-1, 3)
            string_scores = np.asarray([float(s.score) for s in strings], dtype=float).reshape(-1)
            string_axes = np.asarray([s.axis_ras for s in strings], dtype=float).reshape(-1, 3)
            sd, st = _line_distances(string_points, p0, d)
            ang = np.asarray([_angle_deg(ax, d) for ax in string_axes], dtype=float)
            mask = np.logical_and.reduce((sd <= float(cfg.get("string_support_radius_mm", 3.0)), st >= -4.0, st <= dist + 4.0, ang <= float(cfg.get("string_support_angle_deg", 18.0))))
            string_mass = float(np.sum(string_scores[mask]))

        junction_penalty = 0.0
        if junctions:
            jpoints = np.asarray([j.point_ras for j in junctions], dtype=float).reshape(-1, 3)
            jscores = np.asarray([float(j.score) for j in junctions], dtype=float).reshape(-1)
            jd, jt = _line_distances(jpoints, p0, d)
            mask = np.logical_and.reduce((jd <= float(cfg.get("junction_penalty_radius_mm", 4.0)), jt >= -2.0, jt <= dist + 2.0))
            junction_penalty = float(np.sum(jscores[mask]))

        span_term = _clamp01((dist - float(cfg.get("min_seed_span_mm", 12.0))) / 30.0)
        return 0.48 * bead_mass + 0.25 * string_mass + 0.22 * geom_score + 0.15 * span_term - 0.45 * junction_penalty

    def _chain_seed_score(
        self,
        *,
        center: np.ndarray,
        axis: np.ndarray,
        chain_indices: list[int],
        beads: list[_Primitive],
        strings: list[_Primitive],
        junctions: list[_Primitive],
        cfg: dict[str, Any],
    ) -> tuple[float, str | None]:
        priors = list(cfg.get("_electrode_family_priors") or [])
        bead_points = np.asarray([b.point_ras for b in beads], dtype=float).reshape(-1, 3)
        bead_scores = np.asarray([float(b.score) for b in beads], dtype=float).reshape(-1)
        bead_diams = np.asarray([float(b.diameter_mm) for b in beads], dtype=float).reshape(-1)
        bd, bt = _line_distances(bead_points, center, axis)
        seed_span = float(cfg.get("seed_support_span_mm", 80.0))
        bead_mask = np.logical_and.reduce((bd <= float(cfg.get("bead_support_radius_mm", 2.5)), bt >= -4.0, bt <= seed_span))
        bead_mass = float(np.sum(bead_scores[bead_mask]))
        geom_score, best_model_id = _chain_geometry_compatibility(bt[np.asarray(chain_indices, dtype=int)], bead_diams[np.asarray(chain_indices, dtype=int)], priors)

        string_mass = 0.0
        if strings:
            string_points = np.asarray([s.point_ras for s in strings], dtype=float).reshape(-1, 3)
            string_scores = np.asarray([float(s.score) for s in strings], dtype=float).reshape(-1)
            string_axes = np.asarray([s.axis_ras for s in strings], dtype=float).reshape(-1, 3)
            sd, st = _line_distances(string_points, center, axis)
            ang = np.asarray([_angle_deg(ax, axis) for ax in string_axes], dtype=float)
            mask = np.logical_and.reduce((sd <= float(cfg.get("string_support_radius_mm", 3.0)), st >= -4.0, st <= seed_span, ang <= float(cfg.get("string_support_angle_deg", 18.0))))
            string_mass = float(np.sum(string_scores[mask]))

        junction_penalty = 0.0
        if junctions:
            jpoints = np.asarray([j.point_ras for j in junctions], dtype=float).reshape(-1, 3)
            jscores = np.asarray([float(j.score) for j in junctions], dtype=float).reshape(-1)
            jd, jt = _line_distances(jpoints, center, axis)
            mask = np.logical_and.reduce((jd <= float(cfg.get("junction_penalty_radius_mm", 4.0)), jt >= -2.0, jt <= seed_span))
            junction_penalty = float(np.sum(jscores[mask]))

        span = 0.0
        if chain_indices:
            chain_pts = bead_points[np.asarray(chain_indices, dtype=int)]
            _, chain_t = _line_distances(chain_pts, center, axis)
            span = float(np.max(chain_t) - np.min(chain_t))
        span_term = _clamp01((span - float(cfg.get("min_seed_span_mm", 12.0))) / 30.0)
        score = 0.42 * bead_mass + 0.25 * string_mass + 0.35 * geom_score + 0.15 * span_term - 0.45 * junction_penalty
        return float(score), best_model_id

    def _build_chain_seeds(self, beads: list[_Primitive], strings: list[_Primitive], junctions: list[_Primitive], cfg: dict[str, Any]) -> list[_Seed]:
        if len(beads) < 3:
            return []
        bead_points = np.asarray([b.point_ras for b in beads], dtype=float).reshape(-1, 3)
        raw: list[_Seed] = []
        neighborhood_radius = float(cfg.get("chain_neighbor_radius_mm", 22.0))
        chain_radius = float(cfg.get("chain_fit_radius_mm", 2.8))
        seed_span_max = float(cfg.get("seed_support_span_mm", 80.0))
        min_chain_beads = int(cfg.get("min_chain_beads", 4))
        for anchor_idx, anchor in enumerate(beads):
            p0 = np.asarray(anchor.point_ras, dtype=float)
            dists = np.linalg.norm(bead_points - p0.reshape(1, 3), axis=1)
            nb_idx = np.where(dists <= neighborhood_radius)[0]
            if nb_idx.size < min_chain_beads:
                continue
            nb_points = bead_points[nb_idx]
            nb_weights = np.asarray([float(beads[int(i)].score) for i in nb_idx], dtype=float)
            try:
                center, axis, _ = _weighted_pca_line(nb_points, nb_weights)
            except Exception:
                continue
            dd, tt = _line_distances(bead_points, center, axis)
            mask = np.logical_and.reduce((dd <= chain_radius, tt >= -4.0, tt <= seed_span_max))
            cand_idx = np.where(mask)[0]
            if cand_idx.size < min_chain_beads:
                continue
            order = cand_idx[np.argsort(tt[cand_idx])]
            chain: list[int] = [int(order[0])]
            prev_t = float(tt[order[0]])
            max_gap = float(cfg.get("chain_max_gap_mm", 8.5))
            min_gap = float(cfg.get("chain_min_gap_mm", 1.0))
            for idx in order[1:].tolist():
                gap = float(tt[idx] - prev_t)
                if gap < min_gap:
                    continue
                if gap > max_gap:
                    continue
                chain.append(int(idx))
                prev_t = float(tt[idx])
            if len(chain) < min_chain_beads:
                continue
            score, best_model_id = self._chain_seed_score(center=center, axis=axis, chain_indices=chain, beads=beads, strings=strings, junctions=junctions, cfg=cfg)
            if score < float(cfg.get("min_seed_score", 1.0)):
                continue
            chain_pts = bead_points[np.asarray(chain, dtype=int)]
            _, chain_t = _line_distances(chain_pts, center, axis)
            start_idx = int(chain[np.argmin(chain_t)])
            end_idx = int(chain[np.argmax(chain_t)])
            raw.append(
                _Seed(
                    point_ras=tuple(float(v) for v in bead_points[start_idx]),
                    direction_ras=tuple(float(v) for v in axis),
                    score=float(score),
                    bead_i=start_idx,
                    bead_j=end_idx,
                    chain_indices=tuple(int(v) for v in chain),
                    best_model_id=best_model_id,
                )
            )
        raw.sort(key=lambda s: float(s.score), reverse=True)
        return raw[: int(cfg.get("max_chain_seeds", 300))]

    def _build_seeds(self, primitives: dict[str, list[_Primitive]], cfg: dict[str, Any]) -> list[_Seed]:
        beads = list(primitives["beads"])
        strings = list(primitives["strings"])
        junctions = list(primitives["junctions"])
        if len(beads) < 2:
            return []
        max_beads = int(cfg.get("max_beads_for_seeding", 180))
        if len(beads) > max_beads:
            beads = sorted(beads, key=lambda b: float(b.score), reverse=True)[:max_beads]
        raw = self._build_chain_seeds(beads, strings, junctions, cfg)
        if not raw:
            max_pairs = int(cfg.get("max_seed_pairs", 1500))
            pair_min = float(cfg.get("min_seed_span_mm", 12.0))
            pair_max = float(cfg.get("max_seed_span_mm", 90.0))
            for i in range(len(beads)):
                pi = np.asarray(beads[i].point_ras, dtype=float)
                for j in range(i + 1, len(beads)):
                    pj = np.asarray(beads[j].point_ras, dtype=float)
                    dist = float(np.linalg.norm(pj - pi))
                    if dist < pair_min or dist > pair_max:
                        continue
                    score = self._pair_seed_score(pi, pj, beads, strings, junctions, cfg)
                    if score < float(cfg.get("min_seed_score", 1.0)):
                        continue
                    raw.append(_Seed(point_ras=tuple(float(v) for v in pi), direction_ras=tuple(float(v) for v in _normalize(pj - pi)), score=float(score), bead_i=i, bead_j=j))
            raw.sort(key=lambda s: float(s.score), reverse=True)
            raw = raw[:max_pairs]
        return self._nms_seeds(raw, cfg)

    def _nms_seeds(self, seeds: list[_Seed], cfg: dict[str, Any]) -> list[_Seed]:
        kept: list[_Seed] = []
        angle_thr = float(cfg.get("seed_nms_angle_deg", 8.0))
        dist_thr = float(cfg.get("seed_nms_line_distance_mm", 3.0))
        max_keep = int(cfg.get("max_seeds_after_nms", 80))
        for seed in seeds:
            p0 = np.asarray(seed.point_ras, dtype=float)
            d0 = np.asarray(seed.direction_ras, dtype=float)
            dup = False
            for prev in kept:
                p1 = np.asarray(prev.point_ras, dtype=float)
                d1 = np.asarray(prev.direction_ras, dtype=float)
                angle = _angle_deg(d0, d1)
                line_dist = _line_distance(p0, d0, p1, d1)
                if angle <= angle_thr and line_dist <= dist_thr:
                    dup = True
                    break
            if dup:
                continue
            kept.append(seed)
            if len(kept) >= max_keep:
                break
        return kept

    def _fit_candidates(self, seeds: list[_Seed], primitives: dict[str, list[_Primitive]], gating: dict[str, Any], ctx: DetectionContext, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        beads = list(primitives["beads"])
        strings = list(primitives["strings"])
        junctions = list(primitives["junctions"])
        priors = list(cfg.get("_electrode_family_priors") or [])
        bead_points = np.asarray([b.point_ras for b in beads], dtype=float).reshape(-1, 3) if beads else np.zeros((0, 3), dtype=float)
        bead_diams = np.asarray([float(b.diameter_mm) for b in beads], dtype=float).reshape(-1) if beads else np.zeros((0,), dtype=float)
        string_points = np.asarray([s.point_ras for s in strings], dtype=float).reshape(-1, 3) if strings else np.zeros((0, 3), dtype=float)
        candidates: list[dict[str, Any]] = []
        for idx, seed in enumerate(seeds, start=1):
            p0 = np.asarray(seed.point_ras, dtype=float)
            d0 = _normalize(np.asarray(seed.direction_ras, dtype=float))
            seed_span = float(cfg.get("seed_support_span_mm", 80.0))
            points: list[np.ndarray] = []
            weights: list[float] = []
            assigned_beads: list[int] = []
            assigned_strings: list[int] = []
            bead_mass = 0.0
            string_mass = 0.0
            if beads:
                bd, bt = _line_distances(bead_points, p0, d0)
                for bidx, bead in enumerate(beads):
                    if float(bd[bidx]) > float(cfg.get("bead_fit_radius_mm", 2.8)):
                        continue
                    if float(bt[bidx]) < -4.0 or float(bt[bidx]) > seed_span:
                        continue
                    base_weight = float(bead.score) * float(cfg.get("bead_weight", 1.0))
                    if bidx in seed.chain_indices:
                        base_weight *= float(cfg.get("chain_member_weight_scale", 1.5))
                    points.append(np.asarray(bead.point_ras, dtype=float))
                    weights.append(base_weight)
                    assigned_beads.append(int(bidx))
                    bead_mass += float(bead.score)
            if strings:
                sd, st = _line_distances(string_points, p0, d0)
                for sidx, string in enumerate(strings):
                    ang = _angle_deg(np.asarray(string.axis_ras, dtype=float), d0)
                    if float(sd[sidx]) > float(cfg.get("string_fit_radius_mm", 3.0)):
                        continue
                    if float(st[sidx]) < -4.0 or float(st[sidx]) > seed_span:
                        continue
                    if ang > float(cfg.get("string_fit_angle_deg", 18.0)):
                        continue
                    points.append(np.asarray(string.point_ras, dtype=float))
                    weights.append(float(string.score) * float(cfg.get("string_weight", 1.6)))
                    assigned_strings.append(int(sidx))
                    string_mass += float(string.score)
            if len(points) < int(cfg.get("min_support_observations", 4)):
                continue
            pts = np.asarray(points, dtype=float).reshape(-1, 3)
            w = np.asarray(weights, dtype=float).reshape(-1)
            try:
                center, axis, _evals = _weighted_pca_line(pts, w)
            except Exception:
                continue
            dists, tt = _line_distances(pts, center, axis)
            rms = float(np.sqrt(np.average(np.square(dists), weights=np.maximum(w, 1e-9)))) if pts.shape[0] else float("inf")
            bead_proj = []
            if assigned_beads:
                bead_pts = bead_points[np.asarray(assigned_beads, dtype=int)]
                _, bead_t = _line_distances(bead_pts, center, axis)
                bead_proj = bead_t.tolist()
            span = 0.0
            if bead_proj:
                span = float(max(bead_proj) - min(bead_proj))
            else:
                span = float(np.max(tt) - np.min(tt)) if pts.shape[0] >= 2 else 0.0
            if span < float(cfg.get("min_line_span_mm", 18.0)):
                continue
            geom_score = 0.0
            best_model_id: str | None = None
            if assigned_beads:
                bead_proj_arr = np.asarray(bead_proj, dtype=float)
                bead_diam_arr = bead_diams[np.asarray(assigned_beads, dtype=int)]
                geom_score, best_model_id = _chain_geometry_compatibility(bead_proj_arr, bead_diam_arr, priors)
            start, end = _projected_endpoints(pts, center, axis, float(cfg.get("endpoint_lo_q", 0.10)), float(cfg.get("endpoint_hi_q", 0.90)))
            start, end = _orient_shallow_to_deep(start, end, gating.get("head_distance_map_kji"), ctx.get("ras_to_ijk_fn"), np.asarray(ctx.get("center_ras") or [0.0, 0.0, 0.0], dtype=float))
            junction_penalty = 0.0
            if junctions:
                jpts = np.asarray([j.point_ras for j in junctions], dtype=float).reshape(-1, 3)
                jd, jt = _line_distances(jpts, center, axis)
                jmask = np.logical_and.reduce((jd <= float(cfg.get("junction_penalty_radius_mm", 4.0)), jt >= (float(np.min(tt)) - 2.0), jt <= (float(np.max(tt)) + 2.0)))
                if np.any(jmask):
                    junction_penalty = float(np.sum(np.asarray([float(j.score) for j in junctions], dtype=float)[jmask]))
            seed_prior = float(_clamp01(float(seed.score) / max(1.0, float(cfg.get("seed_score_norm", 8.0)))))
            selection_score = 0.42 * bead_mass + 0.24 * string_mass + 1.35 * geom_score + 0.18 * seed_prior + 0.10 * _clamp01((span - 15.0) / 30.0) - 0.30 * rms - 0.45 * junction_penalty
            if selection_score < float(cfg.get("min_selection_score", 1.2)):
                continue
            candidates.append({
                "name": f"P{idx:03d}",
                "start_ras": start.astype(float).tolist(),
                "end_ras": end.astype(float).tolist(),
                "length_mm": float(np.linalg.norm(end - start)),
                "direction_ras": axis.astype(float).tolist(),
                "selection_score": float(selection_score),
                "bead_support_mass": float(bead_mass),
                "string_support_mass": float(string_mass),
                "junction_penalty": float(junction_penalty),
                "geometry_score": float(geom_score),
                "best_model_id": best_model_id or seed.best_model_id,
                "support_count": int(len(points)),
                "bead_count": int(len(assigned_beads)),
                "string_count": int(len(assigned_strings)),
                "rms_mm": float(rms),
                "span_mm": float(span),
            })
        return candidates

    def _select_final(self, candidates: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
        ordered = sorted(candidates, key=lambda x: float(x.get("selection_score", 0.0)), reverse=True)
        kept: list[dict[str, Any]] = []
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
                angle = _angle_deg(d0, d1)
                line_dist = _line_distance(0.5 * (s0 + e0), d0, 0.5 * (s1 + e1), d1)
                if angle <= float(cfg.get("selection_nms_angle_deg", 8.0)) and line_dist <= float(cfg.get("selection_nms_line_distance_mm", 3.0)):
                    dup = True
                    break
            if dup:
                continue
            kept.append(line)
            if target_count is not None and len(kept) >= target_count:
                break
            if target_count is None and len(kept) >= int(cfg.get("max_lines", 30)):
                break
        return kept

    def run(self, ctx: DetectionContext) -> DetectionResult:
        t_start = time.perf_counter()
        result = self.make_result(ctx)
        diagnostics = self.diagnostics(result)
        cfg = self._config(ctx)
        try:
            cfg.setdefault("_electrode_family_priors", _load_family_priors())
            if "arr_kji" not in ctx or "ijk_kji_to_ras_fn" not in ctx:
                result["warnings"].append("hybrid_bead_string_v1 missing volume context; returning empty result")
                return self.finalize(result, diagnostics, t_start)
            gating = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="gating", fn=lambda: self._build_gating(ctx, cfg))
            lineages, summaries = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="lineage_tracking", fn=lambda: self._lineage_state(ctx, gating, cfg))
            primitives = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="primitive_extraction", fn=lambda: self._build_primitives(lineages, summaries, cfg))
            seeds = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="bead_chain_seeding", fn=lambda: self._build_seeds(primitives, cfg))
            candidates = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="candidate_fitting", fn=lambda: self._fit_candidates(seeds, primitives, gating, ctx, cfg))
            selected = self.run_stage(ctx=ctx, result=result, diagnostics=diagnostics, stage_name="selection", fn=lambda: self._select_final(candidates, cfg))

            result["trajectories"] = [
                {
                    "name": str(line.get("name") or f"P{idx:02d}"),
                    "start_ras": list(line.get("start_ras", [0.0, 0.0, 0.0])),
                    "end_ras": list(line.get("end_ras", [0.0, 0.0, 0.0])),
                    "length_mm": float(line.get("length_mm", 0.0)),
                    "confidence": float(min(1.0, max(0.0, float(line.get("selection_score", 0.0)) / 12.0))),
                    "support_count": int(line.get("support_count", 0)),
                    "params": {
                        "selection_score": float(line.get("selection_score", 0.0)),
                        "bead_support_mass": float(line.get("bead_support_mass", 0.0)),
                        "string_support_mass": float(line.get("string_support_mass", 0.0)),
                        "junction_penalty": float(line.get("junction_penalty", 0.0)),
                        "geometry_score": float(line.get("geometry_score", 0.0)),
                        "best_model_id": line.get("best_model_id"),
                        "bead_count": int(line.get("bead_count", 0)),
                        "string_count": int(line.get("string_count", 0)),
                        "rms_mm": float(line.get("rms_mm", 0.0)),
                        "span_mm": float(line.get("span_mm", 0.0)),
                    },
                }
                for idx, line in enumerate(selected, start=1)
            ]
            result["contacts"] = []
            result["warnings"].append("contact_detection_not_implemented")

            diagnostics.set_count("threshold_levels", int(len(cfg.get("threshold_schedule_hu", [2600.0,2200.0,1800.0,1500.0,1200.0]))))
            diagnostics.set_count("lineage_count_total", int(len(lineages)))
            diagnostics.set_count("primitive_bead_count", int(len(primitives["beads"])))
            diagnostics.set_count("primitive_string_count", int(len(primitives["strings"])))
            diagnostics.set_count("primitive_junction_count", int(len(primitives["junctions"])))
            diagnostics.set_count("seed_count", int(len(seeds)))
            diagnostics.set_count("candidate_count", int(len(candidates)))
            diagnostics.set_count("final_lines_kept", int(len(selected)))
            diagnostics.set_extra("primitive_counts", {"beads": len(primitives["beads"]), "strings": len(primitives["strings"]), "junctions": len(primitives["junctions"])})
            diagnostics.set_extra("electrode_prior_count", int(len(cfg.get("_electrode_family_priors") or [])))
            diagnostics.note("hybrid_bead_string_v1 seeds line hypotheses from bead chains and reinforces them with string primitives")

            writer = self.get_artifact_writer(ctx, result)
            primitive_path = writer.write_csv_rows(
                "primitive_observations.csv",
                ["primitive_type", "lineage_id", "node_index", "x", "y", "z", "axis_x", "axis_y", "axis_z", "score", "length_mm", "diameter_mm", "depth_mm", "threshold_hu"],
                [
                    [
                        p.primitive_type,
                        int(p.lineage_id),
                        int(p.node_index),
                        float(p.point_ras[0]),
                        float(p.point_ras[1]),
                        float(p.point_ras[2]),
                        float(p.axis_ras[0]),
                        float(p.axis_ras[1]),
                        float(p.axis_ras[2]),
                        float(p.score),
                        float(p.length_mm),
                        float(p.diameter_mm),
                        float(p.depth_mm),
                        float(p.threshold_hu),
                    ]
                    for group in (primitives["beads"], primitives["strings"], primitives["junctions"])
                    for p in group
                ],
            )
            add_artifact(result["artifacts"], kind="primitive_csv", path=primitive_path, description="Hybrid primitive observations", stage="primitive_extraction")
            result["artifacts"].extend(
                write_standard_artifacts(
                    writer,
                    result,
                    blobs=[],
                    pipeline_payload={
                        "pipeline_id": self.pipeline_id,
                        "pipeline_version": self.pipeline_version,
                        "counts": {
                            "lineages": int(len(lineages)),
                            "beads": int(len(primitives["beads"])),
                            "strings": int(len(primitives["strings"])),
                            "junctions": int(len(primitives["junctions"])),
                            "seeds": int(len(seeds)),
                            "candidates": int(len(candidates)),
                            "selected": int(len(selected)),
                        },
                    },
                )
            )
        except Exception as exc:
            self.fail(ctx=ctx, result=result, diagnostics=diagnostics, stage=str(getattr(exc, "stage", "pipeline")), exc=exc)
        return self.finalize(result, diagnostics, t_start)
