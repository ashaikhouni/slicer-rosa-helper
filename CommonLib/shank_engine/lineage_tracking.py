"""Multi-threshold blob lineage tracking and role scoring.

This module provides the persistence layer used by analysis tools and
detectors. It tracks connected-component blobs across descending metal
thresholds, summarizes each lineage, and assigns soft role scores:
- stable shank core
- contact-like
- junction/artifact
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from shank_core.blob_candidates import extract_blob_candidates


def _normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-9:
        return np.asarray([0.0, 0.0, 1.0], dtype=float)
    return arr / norm


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _overlap_edges(labels_high: np.ndarray, labels_low: np.ndarray) -> tuple[dict[int, dict[int, int]], dict[int, dict[int, int]]]:
    mask = (labels_high > 0) & (labels_low > 0)
    parent_to_child: dict[int, dict[int, int]] = {}
    child_to_parent: dict[int, dict[int, int]] = {}
    if not np.any(mask):
        return parent_to_child, child_to_parent
    pairs = np.stack([labels_high[mask].astype(int), labels_low[mask].astype(int)], axis=1)
    for parent_id, child_id in pairs.tolist():
        parent_to_child.setdefault(int(parent_id), {})
        parent_to_child[int(parent_id)][int(child_id)] = int(parent_to_child[int(parent_id)].get(int(child_id), 0)) + 1
        child_to_parent.setdefault(int(child_id), {})
        child_to_parent[int(child_id)][int(parent_id)] = int(child_to_parent[int(child_id)].get(int(parent_id), 0)) + 1
    return parent_to_child, child_to_parent


def extract_threshold_levels(
    *,
    arr_kji: np.ndarray,
    gating_mask_kji: np.ndarray,
    depth_map_kji: np.ndarray | None,
    thresholds_hu: list[float],
    ijk_kji_to_ras_fn,
) -> list[dict[str, Any]]:
    """Extract blob levels over a descending threshold schedule."""
    levels: list[dict[str, Any]] = []
    arr = np.asarray(arr_kji)
    gate = np.asarray(gating_mask_kji, dtype=bool)
    for thr in sorted([float(v) for v in thresholds_hu], reverse=True):
        metal = np.asarray(arr >= float(thr), dtype=bool)
        in_head = np.logical_and(metal, gate)
        blobs = extract_blob_candidates(
            in_head.astype(np.uint8),
            arr_kji=arr,
            depth_map_kji=depth_map_kji,
            ijk_kji_to_ras_fn=ijk_kji_to_ras_fn,
            fully_connected=True,
        )
        levels.append(
            {
                "threshold_hu": float(thr),
                "labels_kji": np.asarray(blobs["labels_kji"], dtype=np.int32),
                "blobs": list(blobs.get("blobs") or []),
            }
        )
    return levels


def build_lineages(levels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build dominant-descendant blob lineages across descending thresholds.

    A lineage may first appear at any threshold level, not only the highest one.
    New roots are created for blobs that do not have any parent at the previous
    higher threshold. Each root then follows its dominant descendant chain.
    """
    if not levels:
        return []
    next_maps = [_overlap_edges(levels[i]["labels_kji"], levels[i + 1]["labels_kji"]) for i in range(len(levels) - 1)]
    child_to_parent_maps = [child_to_parent for (_p2c, child_to_parent) in next_maps]
    blob_maps = [{int(b["blob_id"]): b for b in level.get("blobs") or []} for level in levels]
    lineages: list[dict[str, Any]] = []

    for root_level, level in enumerate(levels):
        parent_map = child_to_parent_maps[root_level - 1] if root_level > 0 else {}
        for blob in level.get("blobs") or []:
            blob_id = int(blob["blob_id"])
            if root_level > 0 and parent_map.get(blob_id):
                continue

            lineage_nodes = [
                {
                    "level_index": int(root_level),
                    "threshold_hu": float(level["threshold_hu"]),
                    "blob": blob,
                }
            ]
            current = int(blob_id)
            merge_count = 0
            split_count = 0
            for level_index in range(root_level, len(levels) - 1):
                p2c, c2p = next_maps[level_index]
                children = p2c.get(int(current), {})
                if not children:
                    break
                if len(children) > 1:
                    split_count += 1
                dom_child = max(children.items(), key=lambda kv: int(kv[1]))[0]
                if len(c2p.get(int(dom_child), {})) > 1:
                    merge_count += 1
                child_blob = blob_maps[level_index + 1].get(int(dom_child))
                if child_blob is None:
                    break
                lineage_nodes.append(
                    {
                        "level_index": int(level_index + 1),
                        "threshold_hu": float(levels[level_index + 1]["threshold_hu"]),
                        "blob": child_blob,
                    }
                )
                current = int(dom_child)

            top_blob = lineage_nodes[0]["blob"]
            last_blob = lineage_nodes[-1]["blob"]
            axes = [_normalize(np.asarray(node["blob"].get("pca_axis_ras") or [0.0, 0.0, 1.0], dtype=float)) for node in lineage_nodes]
            axis_changes = [
                float(np.degrees(np.arccos(np.clip(abs(float(np.dot(axes[i], axes[i + 1]))), 0.0, 1.0))))
                for i in range(len(axes) - 1)
            ]
            centers = [np.asarray(node["blob"].get("centroid_ras") or [0.0, 0.0, 0.0], dtype=float) for node in lineage_nodes]
            centroid_drifts = [float(np.linalg.norm(centers[i + 1] - centers[i])) for i in range(len(centers) - 1)]
            lengths = [float(node["blob"].get("length_mm") or 0.0) for node in lineage_nodes]
            diameters = [float(node["blob"].get("diameter_mm") or 0.0) for node in lineage_nodes]
            lineages.append(
                {
                    "lineage_id": int(len(lineages) + 1),
                    "nodes": lineage_nodes,
                    "root_level_index": int(root_level),
                    "first_threshold_hu": float(lineage_nodes[0]["threshold_hu"]),
                    "last_threshold_hu": float(lineage_nodes[-1]["threshold_hu"]),
                    "persistence_levels": int(len(lineage_nodes)),
                    "merge_count": int(merge_count),
                    "split_count": int(split_count),
                    "mean_axis_change_deg": float(np.mean(axis_changes) if axis_changes else 0.0),
                    "max_axis_change_deg": float(np.max(axis_changes) if axis_changes else 0.0),
                    "mean_centroid_drift_mm": float(np.mean(centroid_drifts) if centroid_drifts else 0.0),
                    "length_growth_mm": float(lengths[-1] - lengths[0]) if lengths else 0.0,
                    "diameter_growth_mm": float(diameters[-1] - diameters[0]) if diameters else 0.0,
                    "top_blob": top_blob,
                    "last_blob": last_blob,
                }
            )
    return lineages


def score_lineage_roles(lineage: dict[str, Any], *, total_levels: int) -> dict[str, float]:
    """Assign soft core/contact/junction role scores to one lineage."""
    persistence = float(lineage.get("persistence_levels", 0)) / float(max(1, total_levels))
    first_thr = float(lineage.get("first_threshold_hu", 0.0))
    last_thr = float(lineage.get("last_threshold_hu", 0.0))
    mean_axis = float(lineage.get("mean_axis_change_deg", 0.0))
    max_axis = float(lineage.get("max_axis_change_deg", 0.0))
    centroid_drift = float(lineage.get("mean_centroid_drift_mm", 0.0))
    length_growth = float(lineage.get("length_growth_mm", 0.0))
    diameter_growth = float(lineage.get("diameter_growth_mm", 0.0))
    merge_count = float(lineage.get("merge_count", 0))
    split_count = float(lineage.get("split_count", 0))
    top_blob = dict(lineage.get("top_blob") or {})
    last_blob = dict(lineage.get("last_blob") or {})
    top_length = float(top_blob.get("length_mm") or 0.0)
    top_diameter = float(top_blob.get("diameter_mm") or 0.0)
    top_vox = float(top_blob.get("voxel_count") or 0.0)
    last_length = float(last_blob.get("length_mm") or 0.0)
    last_diameter = float(last_blob.get("diameter_mm") or 0.0)

    elongated_top = _clamp01((top_length - 2.0) / 10.0)
    elongated_last = _clamp01((last_length - 5.0) / 20.0)
    diameter_plausible = 1.0 - _clamp01(abs(last_diameter - 1.2) / 4.0)
    stable_axis = 1.0 - _clamp01(mean_axis / 20.0)
    stable_centroid = 1.0 - _clamp01(centroid_drift / 6.0)
    low_merge = 1.0 - _clamp01((merge_count + 0.5 * split_count) / 4.0)
    modest_growth = 1.0 - _clamp01(diameter_growth / 6.0)

    p_core = (
        0.30 * persistence
        + 0.18 * elongated_top
        + 0.16 * elongated_last
        + 0.14 * stable_axis
        + 0.08 * stable_centroid
        + 0.08 * low_merge
        + 0.06 * modest_growth
    )
    contact_compact = 1.0 - _clamp01((last_length - 5.0) / 15.0)
    late_appearance = _clamp01((max(first_thr, 1.0) - min(first_thr, last_thr)) / max(first_thr, 1.0))
    p_contact = (
        0.35 * (1.0 - persistence)
        + 0.20 * contact_compact
        + 0.15 * diameter_plausible
        + 0.15 * late_appearance
        + 0.15 * stable_axis
    )
    p_junction = (
        0.28 * _clamp01((merge_count + split_count) / 4.0)
        + 0.20 * _clamp01(mean_axis / 20.0)
        + 0.20 * _clamp01(diameter_growth / 6.0)
        + 0.16 * _clamp01(length_growth / 40.0)
        + 0.16 * (1.0 - stable_centroid)
    )
    total = max(1e-6, p_core + p_contact + p_junction)
    return {
        "p_core": float(p_core / total),
        "p_contact": float(p_contact / total),
        "p_junction": float(p_junction / total),
        "support_priority": float(
            max(0.0, (p_core / total) * (1.0 + 0.25 * top_vox) * (1.0 + 0.10 * top_length))
        ),
    }


def summarize_lineages(lineages: list[dict[str, Any]], *, total_levels: int) -> list[dict[str, Any]]:
    """Attach role scores to each lineage and return summary rows."""
    rows: list[dict[str, Any]] = []
    for lineage in lineages:
        roles = score_lineage_roles(lineage, total_levels=total_levels)
        top_blob = dict(lineage.get("top_blob") or {})
        last_blob = dict(lineage.get("last_blob") or {})
        rows.append(
            {
                "lineage_id": int(lineage["lineage_id"]),
                "first_threshold_hu": float(lineage["first_threshold_hu"]),
                "last_threshold_hu": float(lineage["last_threshold_hu"]),
                "persistence_levels": int(lineage["persistence_levels"]),
                "merge_count": int(lineage["merge_count"]),
                "split_count": int(lineage["split_count"]),
                "mean_axis_change_deg": float(lineage["mean_axis_change_deg"]),
                "mean_centroid_drift_mm": float(lineage["mean_centroid_drift_mm"]),
                "length_growth_mm": float(lineage["length_growth_mm"]),
                "diameter_growth_mm": float(lineage["diameter_growth_mm"]),
                "top_length_mm": float(top_blob.get("length_mm") or 0.0),
                "top_diameter_mm": float(top_blob.get("diameter_mm") or 0.0),
                "last_length_mm": float(last_blob.get("length_mm") or 0.0),
                "last_diameter_mm": float(last_blob.get("diameter_mm") or 0.0),
                "p_core": float(roles["p_core"]),
                "p_contact": float(roles["p_contact"]),
                "p_junction": float(roles["p_junction"]),
                "support_priority": float(roles["support_priority"]),
                "top_centroid_ras": [float(v) for v in list(top_blob.get("centroid_ras") or [0.0, 0.0, 0.0])],
                "top_axis_ras": [float(v) for v in list(top_blob.get("pca_axis_ras") or [0.0, 0.0, 1.0])],
                "top_voxel_count": int(top_blob.get("voxel_count") or 0),
            }
        )
    return rows
