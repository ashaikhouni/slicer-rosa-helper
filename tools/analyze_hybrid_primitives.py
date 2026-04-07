#!/usr/bin/env python3
"""Analyze bead/string/junction primitive coverage on the SEEG localization dataset.

This tool is intentionally diagnostic. It does not fit shanks. It builds a
multi-threshold blob-lineage representation, derives three primitive evidence
classes, and measures how well those primitives cover ground-truth shanks.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIB_DIR = os.path.join(REPO_ROOT, "CommonLib")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

from shank_core.masking import build_preview_masks  # noqa: E402
from shank_engine.lineage_tracking import build_lineages, extract_threshold_levels, summarize_lineages  # noqa: E402
from tools.eval_seeg_localization import (  # noqa: E402
    GroundTruthShank,
    _point_to_segment_distance,
    _unit,
    build_detection_context,
    iter_subject_rows,
    load_ground_truth_shanks,
)


@dataclass(frozen=True)
class PrimitiveObservation:
    subject_id: str
    primitive_type: str
    lineage_id: int
    node_index: int
    threshold_hu: float
    score: float
    point_ras: tuple[float, float, float]
    axis_ras: tuple[float, float, float]
    length_mm: float
    diameter_mm: float
    depth_mm: float
    meta: dict[str, Any]


def _write_tsv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    aa = _unit(np.asarray(a, dtype=float))
    bb = _unit(np.asarray(b, dtype=float))
    dot = float(abs(np.dot(aa, bb)))
    dot = min(1.0, max(-1.0, dot))
    return float(np.degrees(np.arccos(dot)))


def _best_string_node(lineage: dict[str, Any], summary: dict[str, Any], *, min_depth_mm: float) -> tuple[int, dict[str, Any]] | None:
    best: tuple[float, int, dict[str, Any]] | None = None
    for idx, node in enumerate(list(lineage.get("nodes") or [])):
        blob = dict(node.get("blob") or {})
        length = float(blob.get("length_mm") or 0.0)
        diameter = float(blob.get("diameter_mm") or 0.0)
        depth = float(blob.get("depth_mean") or 0.0)
        if length < 6.0:
            continue
        if diameter <= 0.0 or diameter > 5.0:
            continue
        if depth < float(min_depth_mm):
            continue
        slender = length / max(diameter, 0.25)
        length_term = _clamp01((length - 6.0) / 20.0)
        slender_term = _clamp01((slender - 3.0) / 10.0)
        diameter_term = 1.0 - _clamp01(abs(diameter - 1.2) / 3.0)
        depth_term = _clamp01((depth - min_depth_mm) / 8.0)
        score = (
            0.45 * float(summary.get("p_core", 0.0))
            + 0.20 * length_term
            + 0.15 * slender_term
            + 0.10 * diameter_term
            + 0.10 * depth_term
            - 0.20 * float(summary.get("p_junction", 0.0))
        )
        if best is None or score > best[0]:
            best = (score, idx, blob)
    if best is None:
        return None
    if best[0] < 0.20:
        return None
    return int(best[1]), best[2]


def _bead_candidates(lineage: dict[str, Any], summary: dict[str, Any], *, min_depth_mm: float) -> list[tuple[int, dict[str, Any], float]]:
    out: list[tuple[int, dict[str, Any], float]] = []
    for idx, node in enumerate(list(lineage.get("nodes") or [])):
        blob = dict(node.get("blob") or {})
        length = float(blob.get("length_mm") or 0.0)
        diameter = float(blob.get("diameter_mm") or 0.0)
        depth = float(blob.get("depth_mean") or 0.0)
        hu_q95 = float(blob.get("hu_q95") or blob.get("hu_max") or 0.0)
        vox = float(blob.get("voxel_count") or 0.0)
        if depth < float(min_depth_mm):
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
        if score < 0.25:
            continue
        out.append((int(idx), blob, float(score)))
    return out


def _junction_node(lineage: dict[str, Any], summary: dict[str, Any]) -> tuple[int, dict[str, Any], float] | None:
    nodes = list(lineage.get("nodes") or [])
    if not nodes:
        return None
    thickest_idx = 0
    thickest_blob = dict(nodes[0].get("blob") or {})
    thickest_score = float(thickest_blob.get("diameter_mm") or 0.0) + 0.5 * float(thickest_blob.get("length_mm") or 0.0)
    for idx, node in enumerate(nodes[1:], start=1):
        blob = dict(node.get("blob") or {})
        score = float(blob.get("diameter_mm") or 0.0) + 0.5 * float(blob.get("length_mm") or 0.0)
        if score > thickest_score:
            thickest_idx = idx
            thickest_blob = blob
            thickest_score = score
    merge_term = _clamp01((float(summary.get("merge_count", 0)) + float(summary.get("split_count", 0))) / 3.0)
    growth_term = _clamp01(float(summary.get("diameter_growth_mm", 0.0)) / 5.0)
    axis_term = _clamp01(float(summary.get("mean_axis_change_deg", 0.0)) / 20.0)
    thick_term = _clamp01((float(thickest_blob.get("diameter_mm") or 0.0) - 2.5) / 4.0)
    score = 0.45 * float(summary.get("p_junction", 0.0)) + 0.20 * merge_term + 0.15 * growth_term + 0.10 * axis_term + 0.10 * thick_term
    if score < 0.30:
        return None
    return int(thickest_idx), thickest_blob, float(score)


def _dedup_points(items: list[PrimitiveObservation], *, radius_mm: float) -> list[PrimitiveObservation]:
    if not items:
        return []
    kept: list[PrimitiveObservation] = []
    for obs in sorted(items, key=lambda x: (-float(x.score), x.lineage_id, x.node_index)):
        pt = np.asarray(obs.point_ras, dtype=float)
        if any(float(np.linalg.norm(pt - np.asarray(k.point_ras, dtype=float))) <= float(radius_mm) for k in kept):
            continue
        kept.append(obs)
    return kept


def _primitive_rows(subject_id: str, lineages: list[dict[str, Any]], summaries: list[dict[str, Any]], *, core_min_depth_mm: float) -> list[PrimitiveObservation]:
    summary_by_id = {int(row["lineage_id"]): row for row in summaries}
    primitives: list[PrimitiveObservation] = []
    bead_candidates: list[PrimitiveObservation] = []

    for lineage in lineages:
        lineage_id = int(lineage["lineage_id"])
        summary = dict(summary_by_id.get(lineage_id) or {})

        string_pick = _best_string_node(lineage, summary, min_depth_mm=core_min_depth_mm)
        if string_pick is not None:
            node_index, blob = string_pick
            primitives.append(
                PrimitiveObservation(
                    subject_id=subject_id,
                    primitive_type="string",
                    lineage_id=lineage_id,
                    node_index=int(node_index),
                    threshold_hu=float(lineage["nodes"][node_index]["threshold_hu"]),
                    score=float(max(0.0, summary.get("p_core", 0.0) - 0.25 * summary.get("p_junction", 0.0))),
                    point_ras=tuple(float(v) for v in list(blob.get("centroid_ras") or [0.0, 0.0, 0.0])[:3]),
                    axis_ras=tuple(float(v) for v in list(blob.get("pca_axis_ras") or [0.0, 0.0, 1.0])[:3]),
                    length_mm=float(blob.get("length_mm") or 0.0),
                    diameter_mm=float(blob.get("diameter_mm") or 0.0),
                    depth_mm=float(blob.get("depth_mean") or 0.0),
                    meta={
                        "p_core": float(summary.get("p_core", 0.0)),
                        "p_contact": float(summary.get("p_contact", 0.0)),
                        "p_junction": float(summary.get("p_junction", 0.0)),
                        "merge_count": int(summary.get("merge_count", 0)),
                        "split_count": int(summary.get("split_count", 0)),
                        "persistence_levels": int(summary.get("persistence_levels", 0)),
                    },
                )
            )

        for node_index, blob, score in _bead_candidates(lineage, summary, min_depth_mm=0.0):
            bead_candidates.append(
                PrimitiveObservation(
                    subject_id=subject_id,
                    primitive_type="bead",
                    lineage_id=lineage_id,
                    node_index=int(node_index),
                    threshold_hu=float(lineage["nodes"][node_index]["threshold_hu"]),
                    score=float(score),
                    point_ras=tuple(float(v) for v in list(blob.get("centroid_ras") or [0.0, 0.0, 0.0])[:3]),
                    axis_ras=tuple(float(v) for v in list(blob.get("pca_axis_ras") or [0.0, 0.0, 1.0])[:3]),
                    length_mm=float(blob.get("length_mm") or 0.0),
                    diameter_mm=float(blob.get("diameter_mm") or 0.0),
                    depth_mm=float(blob.get("depth_mean") or 0.0),
                    meta={
                        "p_core": float(summary.get("p_core", 0.0)),
                        "p_contact": float(summary.get("p_contact", 0.0)),
                        "p_junction": float(summary.get("p_junction", 0.0)),
                        "merge_count": int(summary.get("merge_count", 0)),
                        "split_count": int(summary.get("split_count", 0)),
                    },
                )
            )

        junction_pick = _junction_node(lineage, summary)
        if junction_pick is not None:
            node_index, blob, score = junction_pick
            primitives.append(
                PrimitiveObservation(
                    subject_id=subject_id,
                    primitive_type="junction",
                    lineage_id=lineage_id,
                    node_index=int(node_index),
                    threshold_hu=float(lineage["nodes"][node_index]["threshold_hu"]),
                    score=float(score),
                    point_ras=tuple(float(v) for v in list(blob.get("centroid_ras") or [0.0, 0.0, 0.0])[:3]),
                    axis_ras=tuple(float(v) for v in list(blob.get("pca_axis_ras") or [0.0, 0.0, 1.0])[:3]),
                    length_mm=float(blob.get("length_mm") or 0.0),
                    diameter_mm=float(blob.get("diameter_mm") or 0.0),
                    depth_mm=float(blob.get("depth_mean") or 0.0),
                    meta={
                        "p_core": float(summary.get("p_core", 0.0)),
                        "p_contact": float(summary.get("p_contact", 0.0)),
                        "p_junction": float(summary.get("p_junction", 0.0)),
                        "merge_count": int(summary.get("merge_count", 0)),
                        "split_count": int(summary.get("split_count", 0)),
                        "persistence_levels": int(summary.get("persistence_levels", 0)),
                    },
                )
            )

    primitives.extend(_dedup_points(bead_candidates, radius_mm=2.5))
    return sorted(primitives, key=lambda x: (x.primitive_type, -float(x.score), x.lineage_id, x.node_index))


def _coverage_rows(subject_id: str, gt_shanks: list[GroundTruthShank], primitives: list[PrimitiveObservation], *, string_radius_mm: float, bead_radius_mm: float, junction_radius_mm: float, string_angle_deg: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    strings = [p for p in primitives if p.primitive_type == "string"]
    beads = [p for p in primitives if p.primitive_type == "bead"]
    junctions = [p for p in primitives if p.primitive_type == "junction"]

    for gt in gt_shanks:
        gt_start = np.asarray(gt.start_ras, dtype=float)
        gt_end = np.asarray(gt.end_ras, dtype=float)
        gt_dir = _unit(np.asarray(gt.direction_ras, dtype=float))

        string_hits: list[PrimitiveObservation] = []
        best_string_dist = float("inf")
        best_string_angle = float("inf")
        for obs in strings:
            pt = np.asarray(obs.point_ras, dtype=float)
            dist = _point_to_segment_distance(pt, gt_start, gt_end)
            ang = _angle_deg(gt_dir, np.asarray(obs.axis_ras, dtype=float))
            best_string_dist = min(best_string_dist, float(dist))
            best_string_angle = min(best_string_angle, float(ang))
            if dist <= float(string_radius_mm) and ang <= float(string_angle_deg):
                string_hits.append(obs)

        bead_hits: list[PrimitiveObservation] = []
        best_bead_dist = float("inf")
        for obs in beads:
            pt = np.asarray(obs.point_ras, dtype=float)
            dist = _point_to_segment_distance(pt, gt_start, gt_end)
            best_bead_dist = min(best_bead_dist, float(dist))
            if dist <= float(bead_radius_mm):
                bead_hits.append(obs)

        junction_hits: list[PrimitiveObservation] = []
        best_junction_dist = float("inf")
        for obs in junctions:
            pt = np.asarray(obs.point_ras, dtype=float)
            dist = _point_to_segment_distance(pt, gt_start, gt_end)
            best_junction_dist = min(best_junction_dist, float(dist))
            if dist <= float(junction_radius_mm):
                junction_hits.append(obs)

        rows.append(
            {
                "subject_id": subject_id,
                "gt_shank": gt.shank,
                "contact_count": int(gt.contact_count),
                "string_hit_count": int(len(string_hits)),
                "bead_hit_count": int(len(bead_hits)),
                "junction_near_count": int(len(junction_hits)),
                "has_string_support": int(bool(string_hits)),
                "has_bead_support": int(bool(bead_hits)),
                "has_any_support": int(bool(string_hits or bead_hits)),
                "best_string_dist_mm": "" if not np.isfinite(best_string_dist) else f"{best_string_dist:.4f}",
                "best_string_angle_deg": "" if not np.isfinite(best_string_angle) else f"{best_string_angle:.4f}",
                "best_bead_dist_mm": "" if not np.isfinite(best_bead_dist) else f"{best_bead_dist:.4f}",
                "best_junction_dist_mm": "" if not np.isfinite(best_junction_dist) else f"{best_junction_dist:.4f}",
                "support_pattern": (
                    "beads_on_string" if (string_hits and bead_hits) else "string" if string_hits else "beads" if bead_hits else "none"
                ),
                "junction_risk": "high" if len(junction_hits) >= 2 else "present" if junction_hits else "none",
            }
        )
    return rows


def _lineage_rows(subject_id: str, summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in summaries:
        out = dict(row)
        out["subject_id"] = subject_id
        out["top_centroid_ras"] = json.dumps(list(row.get("top_centroid_ras") or []))
        out["top_axis_ras"] = json.dumps(list(row.get("top_axis_ras") or []))
        rows.append(out)
    return rows


def _primitive_tsv_rows(primitives: list[PrimitiveObservation]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for obs in primitives:
        rows.append(
            {
                "subject_id": obs.subject_id,
                "primitive_type": obs.primitive_type,
                "lineage_id": int(obs.lineage_id),
                "node_index": int(obs.node_index),
                "threshold_hu": f"{float(obs.threshold_hu):.2f}",
                "score": f"{float(obs.score):.4f}",
                "x": f"{float(obs.point_ras[0]):.4f}",
                "y": f"{float(obs.point_ras[1]):.4f}",
                "z": f"{float(obs.point_ras[2]):.4f}",
                "axis_x": f"{float(obs.axis_ras[0]):.6f}",
                "axis_y": f"{float(obs.axis_ras[1]):.6f}",
                "axis_z": f"{float(obs.axis_ras[2]):.6f}",
                "length_mm": f"{float(obs.length_mm):.4f}",
                "diameter_mm": f"{float(obs.diameter_mm):.4f}",
                "depth_mm": f"{float(obs.depth_mm):.4f}",
                "meta_json": json.dumps(obs.meta, sort_keys=True),
            }
        )
    return rows


def _analyze_subject(row: dict[str, str], args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    subject_id = str(row["subject_id"])
    config = {
        "threshold": float(max(args.thresholds_hu)),
        "use_head_mask": True,
        "build_head_mask": True,
        "head_mask_threshold_hu": float(args.head_mask_threshold_hu),
        "head_mask_method": str(args.head_mask_method),
        "head_gate_erode_vox": int(args.head_gate_erode_vox),
        "head_gate_dilate_vox": int(args.head_gate_dilate_vox),
        "head_gate_margin_mm": float(args.head_gate_margin_mm),
        "min_metal_depth_mm": float(args.min_metal_depth_mm),
        "max_metal_depth_mm": float(args.max_metal_depth_mm),
    }
    extras = {"analysis": "hybrid_primitives"}
    ctx, _img = build_detection_context(row["ct_path"], run_id=f"primitive_{subject_id}", config=config, extras=extras)
    masks = build_preview_masks(
        ctx["arr_kji"],
        ctx["spacing_xyz"],
        threshold=float(max(args.thresholds_hu)),
        use_head_mask=True,
        build_head_mask=True,
        head_mask_threshold_hu=float(args.head_mask_threshold_hu),
        head_mask_method=str(args.head_mask_method),
        head_gate_erode_vox=int(args.head_gate_erode_vox),
        head_gate_dilate_vox=int(args.head_gate_dilate_vox),
        head_gate_margin_mm=float(args.head_gate_margin_mm),
        min_metal_depth_mm=float(args.min_metal_depth_mm),
        max_metal_depth_mm=float(args.max_metal_depth_mm),
        include_debug_masks=False,
    )
    gating_mask = np.asarray(masks.get("gating_mask_kji"), dtype=bool)
    if gating_mask.size == 0 or not np.any(gating_mask):
        gating_mask = np.ones_like(ctx["arr_kji"], dtype=bool)
    levels = extract_threshold_levels(
        arr_kji=np.asarray(ctx["arr_kji"], dtype=float),
        gating_mask_kji=gating_mask,
        depth_map_kji=masks.get("head_distance_map_kji"),
        thresholds_hu=[float(v) for v in args.thresholds_hu],
        ijk_kji_to_ras_fn=ctx["ijk_kji_to_ras_fn"],
    )
    lineages = build_lineages(levels)
    summaries = summarize_lineages(lineages, total_levels=len(levels))
    primitives = _primitive_rows(subject_id, lineages, summaries, core_min_depth_mm=float(args.core_min_depth_mm))
    gt_shanks = load_ground_truth_shanks(row["labels_path"], row.get("shanks_path"))
    coverage_rows = _coverage_rows(
        subject_id,
        gt_shanks,
        primitives,
        string_radius_mm=float(args.string_radius_mm),
        bead_radius_mm=float(args.bead_radius_mm),
        junction_radius_mm=float(args.junction_radius_mm),
        string_angle_deg=float(args.string_angle_deg),
    )
    primitive_counts = {kind: sum(1 for p in primitives if p.primitive_type == kind) for kind in ("string", "bead", "junction")}
    subject_summary = {
        "subject_id": subject_id,
        "gt_shanks": len(gt_shanks),
        "lineage_count_total": len(lineages),
        "lineage_count_core_like": int(sum(1 for row in summaries if float(row.get("p_core", 0.0)) >= 0.34)),
        "lineage_count_contact_like": int(sum(1 for row in summaries if float(row.get("p_contact", 0.0)) >= 0.34)),
        "lineage_count_junction_like": int(sum(1 for row in summaries if float(row.get("p_junction", 0.0)) >= 0.34)),
        "primitive_string_count": int(primitive_counts["string"]),
        "primitive_bead_count": int(primitive_counts["bead"]),
        "primitive_junction_count": int(primitive_counts["junction"]),
        "gt_with_string_support": int(sum(int(r["has_string_support"]) for r in coverage_rows)),
        "gt_with_bead_support": int(sum(int(r["has_bead_support"]) for r in coverage_rows)),
        "gt_with_any_support": int(sum(int(r["has_any_support"]) for r in coverage_rows)),
        "gt_with_junction_near": int(sum(1 for r in coverage_rows if int(r["junction_near_count"]) > 0)),
    }
    return subject_summary, _lineage_rows(subject_id, summaries), _primitive_tsv_rows(primitives), coverage_rows


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze bead/string/junction primitive coverage on the SEEG dataset")
    p.add_argument("--dataset-root", required=True, help="Path to seeg_localization root")
    p.add_argument("--out-dir", required=True, help="Directory for analysis outputs")
    p.add_argument("--subjects", default="", help="Comma-separated subject ids to analyze")
    p.add_argument("--thresholds-hu", default="2600,2200,1800,1500,1200", help="Descending HU thresholds")
    p.add_argument("--head-mask-method", default="outside_air", choices=["outside_air", "not_air_lcc"])
    p.add_argument("--head-mask-threshold-hu", type=float, default=-500.0)
    p.add_argument("--head-gate-erode-vox", type=int, default=1)
    p.add_argument("--head-gate-dilate-vox", type=int, default=1)
    p.add_argument("--head-gate-margin-mm", type=float, default=0.0)
    p.add_argument("--min-metal-depth-mm", type=float, default=5.0)
    p.add_argument("--max-metal-depth-mm", type=float, default=220.0)
    p.add_argument("--core-min-depth-mm", type=float, default=3.0)
    p.add_argument("--string-radius-mm", type=float, default=3.0)
    p.add_argument("--string-angle-deg", type=float, default=20.0)
    p.add_argument("--bead-radius-mm", type=float, default=3.0)
    p.add_argument("--junction-radius-mm", type=float, default=4.0)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    args.thresholds_hu = [float(v.strip()) for v in str(args.thresholds_hu).split(",") if v.strip()]
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    subject_filter = {s.strip() for s in str(args.subjects).split(",") if s.strip()} or None
    subject_rows = iter_subject_rows(dataset_root, subject_filter)
    if not subject_rows:
        raise SystemExit("No subjects matched")

    subject_summaries: list[dict[str, Any]] = []
    all_lineages: list[dict[str, Any]] = []
    all_primitives: list[dict[str, Any]] = []
    all_coverage: list[dict[str, Any]] = []

    for row in subject_rows:
        subject_id = str(row["subject_id"])
        summary, lineage_rows, primitive_rows, coverage_rows = _analyze_subject(row, args)
        subject_summaries.append(summary)
        all_lineages.extend(lineage_rows)
        all_primitives.extend(primitive_rows)
        all_coverage.extend(coverage_rows)

        subject_dir = out_dir / subject_id
        subject_dir.mkdir(parents=True, exist_ok=True)
        _write_tsv(subject_dir / "lineages.tsv", lineage_rows, fieldnames=list(lineage_rows[0].keys()) if lineage_rows else ["subject_id"])
        _write_tsv(subject_dir / "primitives.tsv", primitive_rows, fieldnames=list(primitive_rows[0].keys()) if primitive_rows else ["subject_id"])
        _write_tsv(subject_dir / "gt_coverage.tsv", coverage_rows, fieldnames=list(coverage_rows[0].keys()) if coverage_rows else ["subject_id"])
        with open(subject_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True, default=_json_default)

    _write_tsv(out_dir / "subject_summary.tsv", subject_summaries, fieldnames=list(subject_summaries[0].keys()))
    _write_tsv(out_dir / "lineages.tsv", all_lineages, fieldnames=list(all_lineages[0].keys()) if all_lineages else ["subject_id"])
    _write_tsv(out_dir / "primitives.tsv", all_primitives, fieldnames=list(all_primitives[0].keys()) if all_primitives else ["subject_id"])
    _write_tsv(out_dir / "gt_coverage.tsv", all_coverage, fieldnames=list(all_coverage[0].keys()) if all_coverage else ["subject_id"])

    cohort_summary = {
        "subject_count": len(subject_summaries),
        "subjects": [str(row["subject_id"]) for row in subject_summaries],
        "gt_total": int(sum(int(row["gt_shanks"]) for row in subject_summaries)),
        "gt_with_string_support": int(sum(int(row["gt_with_string_support"]) for row in subject_summaries)),
        "gt_with_bead_support": int(sum(int(row["gt_with_bead_support"]) for row in subject_summaries)),
        "gt_with_any_support": int(sum(int(row["gt_with_any_support"]) for row in subject_summaries)),
        "gt_with_junction_near": int(sum(int(row["gt_with_junction_near"]) for row in subject_summaries)),
        "primitive_string_count": int(sum(int(row["primitive_string_count"]) for row in subject_summaries)),
        "primitive_bead_count": int(sum(int(row["primitive_bead_count"]) for row in subject_summaries)),
        "primitive_junction_count": int(sum(int(row["primitive_junction_count"]) for row in subject_summaries)),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(cohort_summary, f, indent=2, sort_keys=True, default=_json_default)


if __name__ == "__main__":
    main()
