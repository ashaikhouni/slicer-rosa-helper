"""Complex-blob routing and line recovery for Deep Core support extraction."""

try:
    import numpy as np
except ImportError:
    np = None

from .deep_core_config import deep_core_default_config


_DEEP_CORE_DEFAULTS = deep_core_default_config()
_DEFAULT_ANNULUS_CONFIG = _DEEP_CORE_DEFAULTS.annulus
_DEFAULT_INTERNAL_CONFIG = _DEEP_CORE_DEFAULTS.internal


class DeepCoreComplexBlobMixin:
    """Recover narrow line atoms from voxelized complex blobs."""

    def _build_complex_blob_cube_adjacency(
        self,
        nodes,
    ):
        node_list = [dict(node or {}) for node in list(nodes or []) if int(dict(node or {}).get("node_id", -1)) > 0]
        node_map = {int(node.get("node_id")): node for node in node_list}
        adj = {int(node_id): [] for node_id in node_map.keys()}
        node_degrees = {int(node_id): 0 for node_id in node_map.keys()}
        cube_coord_to_node_id = {}
        for node_id, node in node_map.items():
            cube_coord = tuple(int(v) for v in np.asarray(node.get("cube_coord"), dtype=int).reshape(3).tolist())
            cube_coord_to_node_id[cube_coord] = int(node_id)
        added_edges = set()
        for node_id, node in node_map.items():
            cube_coord = tuple(int(v) for v in np.asarray(node.get("cube_coord"), dtype=int).reshape(3).tolist())
            center_i = np.asarray(node.get("center_ras"), dtype=float).reshape(3)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        neighbor_coord = (int(cube_coord[0] + dx), int(cube_coord[1] + dy), int(cube_coord[2] + dz))
                        nb_id = int(cube_coord_to_node_id.get(neighbor_coord, -1))
                        if nb_id <= 0 or nb_id == int(node_id):
                            continue
                        edge_key = tuple(sorted((int(node_id), int(nb_id))))
                        if edge_key in added_edges:
                            continue
                        nb_node = dict(node_map.get(int(nb_id)) or {})
                        if not nb_node:
                            continue
                        center_j = np.asarray(nb_node.get("center_ras"), dtype=float).reshape(3)
                        step = center_j - center_i
                        dist_mm = float(np.linalg.norm(step))
                        if dist_mm <= 1e-6:
                            continue
                        step_dir = step / dist_mm
                        edge_score = float(1.0 - 0.05 * dist_mm)
                        adj[int(node_id)].append(
                            {
                                "neighbor_id": int(nb_id),
                                "score": float(edge_score),
                                "dist_mm": float(dist_mm),
                                "direction_ras": np.asarray(step_dir, dtype=float).reshape(3),
                            }
                        )
                        adj[int(nb_id)].append(
                            {
                                "neighbor_id": int(node_id),
                                "score": float(edge_score),
                                "dist_mm": float(dist_mm),
                                "direction_ras": np.asarray(-step_dir, dtype=float).reshape(3),
                            }
                        )
                        node_degrees[int(node_id)] = int(node_degrees.get(int(node_id), 0)) + 1
                        node_degrees[int(nb_id)] = int(node_degrees.get(int(nb_id), 0)) + 1
                        added_edges.add(edge_key)
        return {
            "node_map": dict(node_map),
            "adj": dict(adj),
            "node_degrees": dict(node_degrees),
        }

    def _extract_complex_blob_paths(
        self,
        slab_nodes,
        provisional_axis_ras,
        sample_spacing_mm=2.5,
        min_coverage=0.90,
        max_gap_bins=1,
        candidate_guard_diameter_mm=4.0,
        parent_blob_id=None,
        volume_node=None,
        head_distance_map_kji=None,
        annulus_reference_values_hu=None,
        annulus_config=None,
        internal_config=None,
        return_debug=False,
    ):
        annulus_cfg = annulus_config if annulus_config is not None else _DEFAULT_ANNULUS_CONFIG
        internal_cfg = internal_config if internal_config is not None else _DEFAULT_INTERNAL_CONFIG
        nodes = list(slab_nodes or [])
        if len(nodes) < 2:
            return {"paths": [], "chain_rows": []} if bool(return_debug) else []
        spacing = float(max(0.5, sample_spacing_mm))
        graph = self._build_complex_blob_cube_adjacency(nodes)
        node_map = dict(graph.get("node_map") or {})
        adj = dict(graph.get("adj") or {})
        node_degrees = dict(graph.get("node_degrees") or {})
        accepted, candidate_seed_node_ids, graph_node_map = self._build_complex_blob_greedy_paths(
            node_map=node_map,
            adj=adj,
            node_degrees=node_degrees,
            provisional_axis_ras=provisional_axis_ras,
            max_route_count=int(internal_cfg.complex_blob_max_route_count),
            max_seed_count=int(internal_cfg.complex_blob_max_seed_count),
            max_steps=max(20, int(2 * len(node_map) + 2)),
            sample_spacing_mm=spacing,
            min_coverage=min_coverage,
            max_gap_bins=max_gap_bins,
            candidate_guard_diameter_mm=float(max(1.0, candidate_guard_diameter_mm)),
            volume_node=volume_node,
            head_distance_map_kji=head_distance_map_kji,
            annulus_reference_values_hu=annulus_reference_values_hu,
            annulus_config=annulus_cfg,
            internal_config=internal_cfg,
        )
        if not bool(return_debug):
            return accepted
        return {
            "paths": list(accepted),
            "chain_rows": self._build_complex_blob_chain_rows(
                parent_blob_id=int(parent_blob_id) if parent_blob_id is not None else -1,
                graph={"node_map": graph_node_map, "node_degrees": node_degrees},
                accepted_paths=accepted,
                candidate_seed_node_ids=candidate_seed_node_ids,
            ),
        }

    def _build_complex_blob_greedy_paths(
        self,
        node_map,
        adj,
        node_degrees,
        provisional_axis_ras,
        max_route_count=6,
        max_seed_count=16,
        max_steps=24,
        sample_spacing_mm=2.5,
        min_coverage=0.90,
        max_gap_bins=1,
        candidate_guard_diameter_mm=4.0,
        volume_node=None,
        head_distance_map_kji=None,
        annulus_reference_values_hu=None,
        annulus_config=None,
        internal_config=None,
    ):
        annulus_cfg = annulus_config if annulus_config is not None else _DEFAULT_ANNULUS_CONFIG
        internal_cfg = internal_config if internal_config is not None else _DEFAULT_INTERNAL_CONFIG
        node_map = {int(node_id): dict(node or {}) for node_id, node in dict(node_map or {}).items() if int(node_id) > 0}
        adj = {int(node_id): list(neighbors or []) for node_id, neighbors in dict(adj or {}).items()}
        node_degrees = {int(node_id): int(val) for node_id, val in dict(node_degrees or {}).items()}
        if len(node_map) < 2:
            return [], [], dict(node_map)
        seed_blocked_node_ids = set()
        accepted = []
        candidate_seed_node_ids = []
        for _iter_idx in range(int(max_route_count)):
            residual_node_ids = [int(node_id) for node_id in node_map.keys() if int(node_id) not in seed_blocked_node_ids]
            residual_axis = np.asarray(provisional_axis_ras, dtype=float).reshape(3)
            if len(residual_node_ids) >= 2:
                residual_centers = np.asarray(
                    [
                        np.asarray(
                            dict(node_map.get(int(node_id)) or {}).get("center_ras")
                            if dict(node_map.get(int(node_id)) or {}).get("center_ras") is not None
                            else [0.0, 0.0, 0.0],
                            dtype=float,
                        ).reshape(3)
                        for node_id in residual_node_ids
                    ],
                    dtype=float,
                ).reshape(-1, 3)
                if residual_centers.shape[0] >= 2:
                    _, residual_axis = self._fit_line_pca(residual_centers, seed_axis=residual_axis)
            residual_axis_norm = float(np.linalg.norm(residual_axis))
            if residual_axis_norm <= 1e-6:
                residual_axis = np.asarray([0.0, 0.0, 1.0], dtype=float)
                residual_axis_norm = 1.0
            residual_axis = residual_axis / residual_axis_norm
            residual_node_degrees = {}
            blocked_seed_set = set(int(v) for v in list(seed_blocked_node_ids or []) if int(v) > 0)
            for node_id in residual_node_ids:
                residual_node_degrees[int(node_id)] = int(
                    sum(
                        1
                        for edge in list(adj.get(int(node_id)) or [])
                        if int(dict(edge or {}).get("neighbor_id", -1)) > 0
                        and int(dict(edge or {}).get("neighbor_id", -1)) not in blocked_seed_set
                    )
                )
            seed_ids = self._complex_blob_seed_node_ids(
                node_map=node_map,
                node_degrees=residual_node_degrees,
                provisional_axis_ras=residual_axis,
                blocked_node_ids=seed_blocked_node_ids,
                max_seed_count=max_seed_count,
                volume_node=volume_node,
                head_distance_map_kji=head_distance_map_kji,
                annulus_reference_values_hu=annulus_reference_values_hu,
                annulus_config=annulus_cfg,
                internal_config=internal_cfg,
            )
            for seed_id in list(seed_ids or []):
                seed_id = int(seed_id)
                if seed_id > 0 and seed_id not in candidate_seed_node_ids:
                    candidate_seed_node_ids.append(seed_id)
            candidates = []
            for seed_id in list(seed_ids or []):
                seed_node = dict(node_map.get(int(seed_id)) or {})
                if not seed_node:
                    continue
                seed_axis_val = seed_node.get("axis_ras")
                if seed_axis_val is None:
                    seed_axis_val = residual_axis
                seed_dir = np.asarray(seed_axis_val, dtype=float).reshape(3)
                seed_dir_norm = float(np.linalg.norm(seed_dir))
                seed_dir = seed_dir / max(seed_dir_norm, 1e-6)
                if float(np.dot(seed_dir, residual_axis)) < 0.0:
                    seed_dir = -seed_dir
                forward = self._trace_complex_blob_directional_path(
                    node_map=node_map,
                    adj=adj,
                    start_id=int(seed_id),
                    initial_direction_ras=seed_dir,
                    max_steps=int(max_steps),
                    blocked_node_ids=None,
                    candidate_guard_diameter_mm=float(candidate_guard_diameter_mm),
                )
                backward = self._trace_complex_blob_directional_path(
                    node_map=node_map,
                    adj=adj,
                    start_id=int(seed_id),
                    initial_direction_ras=-seed_dir,
                    max_steps=int(max_steps),
                    blocked_node_ids=None,
                    candidate_guard_diameter_mm=float(candidate_guard_diameter_mm),
                )
                merged = [int(v) for v in list(reversed(backward[1:]))] + [int(seed_id)] + [int(v) for v in list(forward[1:])]
                merged = [int(v) for idx, v in enumerate(merged) if idx == 0 or int(v) != int(merged[idx - 1])]
                if len(merged) < 2:
                    continue
                merged = self._extend_complex_blob_path_to_exhaustion(
                    node_map=node_map,
                    adj=adj,
                    node_ids=merged,
                    blocked_node_ids=None,
                    max_extension_steps=max(8, int(max_steps)),
                    candidate_guard_diameter_mm=float(candidate_guard_diameter_mm),
                )
                merged = self._trim_complex_blob_path_to_narrow_core(
                    node_map=node_map,
                    node_ids=merged,
                    target_diameter_mm=float(max(1.0, candidate_guard_diameter_mm)),
                    target_rms_mm=float(internal_cfg.complex_blob_target_rms_mm),
                    min_nodes=3,
                )
                path_info = self._complex_blob_path_summary(
                    node_map=node_map,
                    node_ids=merged,
                    sample_spacing_mm=sample_spacing_mm,
                    min_coverage=min_coverage,
                    max_gap_bins=max_gap_bins,
                )
                if path_info is None:
                    continue
                path_info["seed_node_id"] = int(seed_id)
                candidates.append(dict(path_info))
            if not candidates:
                break
            candidates.sort(
                key=lambda item: (
                    -float(item.get("score", 0.0)),
                    -float(item.get("span_mm", 0.0)),
                    -int(len(list(item.get("node_ids") or []))),
                )
            )
            best_candidate = None
            for candidate in list(candidates or []):
                if any(self._complex_blob_paths_overlap(candidate, prev) for prev in accepted):
                    continue
                best_candidate = dict(candidate)
                break
            if best_candidate is None:
                break
            accepted.append(dict(best_candidate))
            seed_blocked_node_ids.update(int(v) for v in list(best_candidate.get("node_ids") or []) if int(v) > 0)
        node_map_out = dict(node_map)
        for node_id, node in list(node_map_out.items()):
            if int(node_id) <= 0:
                continue
            node_copy = dict(node or {})
            node_copy["candidate_seed"] = bool(int(node_id) in set(candidate_seed_node_ids))
            node_map_out[int(node_id)] = node_copy
        return list(accepted), list(candidate_seed_node_ids), node_map_out

    def _complex_blob_seed_node_ids(
        self,
        node_map,
        node_degrees,
        provisional_axis_ras,
        blocked_node_ids=None,
        max_seed_count=16,
        volume_node=None,
        head_distance_map_kji=None,
        annulus_reference_values_hu=None,
        annulus_config=None,
        internal_config=None,
    ):
        annulus_cfg = annulus_config if annulus_config is not None else _DEFAULT_ANNULUS_CONFIG
        internal_cfg = internal_config if internal_config is not None else _DEFAULT_INTERNAL_CONFIG
        node_map = {int(node_id): dict(node or {}) for node_id, node in dict(node_map or {}).items() if int(node_id) > 0}
        blocked = set(int(v) for v in list(blocked_node_ids or []) if int(v) > 0)
        if not node_map:
            return []
        axis = np.asarray(provisional_axis_ras, dtype=float).reshape(3)
        axis_norm = float(np.linalg.norm(axis))
        axis = axis / max(axis_norm, 1e-6)
        centers = {
            int(node_id): np.asarray(
                dict(node or {}).get("center_ras") if dict(node or {}).get("center_ras") is not None else [0.0, 0.0, 0.0],
                dtype=float,
            ).reshape(3)
            for node_id, node in node_map.items()
        }
        all_centers = np.asarray(list(centers.values()), dtype=float).reshape(-1, 3)
        centroid = np.mean(all_centers, axis=0) if all_centers.shape[0] > 0 else np.zeros((3,), dtype=float)
        cheap_ranked = []
        for node_id, center in centers.items():
            if int(node_id) in blocked:
                continue
            node = dict(node_map.get(int(node_id)) or {})
            degree = int(node_degrees.get(int(node_id), len(list(node_map.keys()))))
            proj = float(np.dot(center - centroid, axis))
            shape = dict(node.get("shape") or {})
            diameter_mm = float(shape.get("diameter_mm", float("inf")) or float("inf"))
            elongation = float(shape.get("elongation", 1.0) or 1.0)
            axis_reliable = bool(node.get("axis_reliable"))
            degree_group = 0 if degree <= 1 else (1 if degree == 2 else 2)
            cheap_ranked.append(
                {
                    "node_id": int(node_id),
                    "proj": float(proj),
                    "abs_proj": float(abs(proj)),
                    "degree": int(degree),
                    "degree_group": int(degree_group),
                    "diameter_mm": float(diameter_mm),
                    "elongation": float(elongation),
                    "axis_reliable": bool(axis_reliable),
                }
            )
        if not cheap_ranked:
            return []
        cheap_ranked.sort(
            key=lambda item: (
                float(item.get("diameter_mm", float("inf"))),
                int(item.get("degree_group", 9)),
                -float(item.get("elongation", 0.0)),
                -float(item.get("abs_proj", 0.0)),
                0 if bool(item.get("axis_reliable")) else 1,
                int(item.get("degree", 999)),
                int(item.get("node_id", -1)),
            )
        )
        shortlist_count = int(
            max(
                int(internal_cfg.complex_seed_shortlist_min),
                min(len(cheap_ranked), max(2, int(internal_cfg.complex_seed_shortlist_multiplier) * int(max_seed_count))),
            )
        )
        shortlist_ids = [int(item.get("node_id", -1)) for item in cheap_ranked[:shortlist_count] if int(item.get("node_id", -1)) > 0]
        if volume_node is not None and shortlist_ids:
            for node_id in shortlist_ids:
                node = dict(node_map.get(int(node_id)) or {})
                if not node:
                    continue
                if node.get("annulus_percentile") is not None and node.get("head_depth_mm") is not None:
                    continue
                node_axis_val = node.get("axis_ras")
                if node_axis_val is None:
                    node_axis_val = axis
                node_axis = np.asarray(node_axis_val, dtype=float).reshape(3)
                support_point_val = node.get("support_point_ras")
                if support_point_val is None:
                    support_point_val = node.get("center_ras")
                if support_point_val is None:
                    support_point_val = [0.0, 0.0, 0.0]
                support_point = np.asarray(support_point_val, dtype=float).reshape(3)
                annulus_stats = self._cross_section_annulus_stats_hu(
                    volume_node=volume_node,
                    center_ras=support_point,
                    axis_ras=node_axis,
                    annulus_inner_mm=float(annulus_cfg.cross_section_annulus_inner_mm),
                    annulus_outer_mm=float(annulus_cfg.cross_section_annulus_outer_mm),
                    radial_steps=int(annulus_cfg.annulus_radial_steps),
                    angular_samples=int(annulus_cfg.annulus_angular_samples),
                )
                annulus_median_hu = annulus_stats.get("median_hu")
                node["annulus_mean_hu"] = annulus_stats.get("mean_hu")
                node["annulus_median_hu"] = annulus_median_hu
                node["annulus_sample_count"] = int(annulus_stats.get("sample_count", 0) or 0)
                node["annulus_percentile"] = self._value_percentile_from_sorted(
                    annulus_reference_values_hu,
                    annulus_median_hu,
                )
                node["head_depth_mm"] = self._depth_at_ras_with_volume(
                    volume_node=volume_node,
                    depth_map_kji=head_distance_map_kji,
                    point_ras=support_point,
                )
                node_map[int(node_id)] = node
        ranked = []
        for item in cheap_ranked:
            node = dict(node_map.get(int(item.get("node_id", -1))) or {})
            annulus_percentile = node.get("annulus_percentile")
            if annulus_percentile is None:
                annulus_percentile = 100.0
            head_depth_mm = float(node.get("head_depth_mm", 0.0) or 0.0)
            ranked.append(
                {
                    **dict(item),
                    "annulus_percentile": float(annulus_percentile),
                    "head_depth_mm": float(head_depth_mm),
                }
            )
        ranked.sort(
            key=lambda item: (
                0 if float(item.get("annulus_percentile", 100.0)) <= float(internal_cfg.complex_seed_rank_good_percentile) else (
                    1 if float(item.get("annulus_percentile", 100.0)) <= float(internal_cfg.complex_seed_rank_preferred_percentile) else (
                        2 if float(item.get("annulus_percentile", 100.0)) <= float(internal_cfg.complex_seed_rank_tolerated_percentile) else 3
                    )
                ),
                -float(item.get("head_depth_mm", 0.0)),
                float(item.get("diameter_mm", float("inf"))),
                int(item.get("degree_group", 9)),
                -float(item.get("elongation", 0.0)),
                -float(item.get("abs_proj", 0.0)),
                0 if bool(item.get("axis_reliable")) else 1,
                int(item.get("degree", 999)),
                int(item.get("node_id", -1)),
            )
        )
        preferred_ranked = [
            item
            for item in ranked
            if float(item.get("annulus_percentile", 100.0)) <= float(internal_cfg.complex_seed_preferred_annulus_percentile)
            and float(item.get("head_depth_mm", 0.0)) >= float(internal_cfg.complex_seed_min_head_depth_mm)
        ]
        if preferred_ranked:
            ranked = preferred_ranked
        ordered_ids = [int(item.get("node_id", -1)) for item in ranked if int(item.get("node_id", -1)) > 0]
        if not ordered_ids:
            return []
        pos_ranked = [item for item in ranked if float(item.get("proj", 0.0)) >= 0.0]
        neg_ranked = [item for item in ranked if float(item.get("proj", 0.0)) < 0.0]
        seeds = []
        for partition in (pos_ranked, neg_ranked):
            for item in list(partition[:4]):
                node_id = int(item.get("node_id", -1))
                if node_id > 0 and node_id not in seeds:
                    seeds.append(int(node_id))
        proj_sorted = sorted(
            [dict(item) for item in ranked],
            key=lambda item: float(item.get("proj", 0.0)),
        )
        for item in list(proj_sorted[:2]) + list(proj_sorted[-2:]):
            node_id = int(item.get("node_id", -1))
            if node_id > 0 and node_id not in seeds:
                seeds.append(int(node_id))
        for node_id in ordered_ids:
            if int(node_id) not in seeds:
                seeds.append(int(node_id))
            if len(seeds) >= int(max(2, max_seed_count)):
                break
        return [int(v) for v in seeds[: max(2, int(max_seed_count))]]

    def _build_complex_blob_chain_rows(
        self,
        parent_blob_id,
        graph,
        accepted_paths,
        candidate_seed_node_ids=None,
    ):
        node_map = dict(dict(graph or {}).get("node_map") or {})
        node_degrees = dict(dict(graph or {}).get("node_degrees") or {})
        if not node_map:
            return []
        memberships = {}
        seed_node_ids_by_line_id = {}
        candidate_seed_node_ids = [int(v) for v in list(candidate_seed_node_ids or []) if int(v) > 0]
        accepted_seed_node_ids = set()
        for line_id, path in enumerate(list(accepted_paths or []), start=1):
            node_ids = [int(v) for v in list(dict(path or {}).get("node_ids") or []) if int(v) > 0]
            seed_node_id = int(dict(path or {}).get("seed_node_id", -1))
            if seed_node_id > 0:
                seed_node_ids_by_line_id[int(line_id)] = int(seed_node_id)
                accepted_seed_node_ids.add(int(seed_node_id))
            for line_order, node_id in enumerate(node_ids, start=1):
                memberships.setdefault(int(node_id), []).append((int(line_id), int(line_order)))
        rows = []
        for node_id in sorted(int(v) for v in node_map.keys()):
            node = dict(node_map.get(int(node_id)) or {})
            node_role = str(node.get("node_role") or self._complex_blob_node_role(int(node_degrees.get(int(node_id), 0))))
            support_point = np.asarray(node.get("support_point_ras"), dtype=float).reshape(-1)
            if support_point.size != 3:
                support_point = np.asarray(node.get("center_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
            member_rows = list(memberships.get(int(node_id)) or [(0, 0)])
            for line_id, line_order in member_rows:
                is_seed = bool(int(seed_node_ids_by_line_id.get(int(line_id), -1)) == int(node_id) and int(line_id) > 0)
                rows.append(
                    {
                        "blob_id": int(parent_blob_id),
                        "node_id": int(node_id),
                        "node_role": str(node_role),
                        "line_id": int(line_id),
                        "line_order": int(line_order),
                        "is_seed": bool(is_seed),
                        "bin_id": int(node.get("bin_id", -1)),
                        "degree": int(node_degrees.get(int(node_id), 0)),
                        "support_x_ras": float(support_point[0]),
                        "support_y_ras": float(support_point[1]),
                        "support_z_ras": float(support_point[2]),
                    }
                )
            if int(node_id) in candidate_seed_node_ids and int(node_id) not in accepted_seed_node_ids:
                rows.append(
                    {
                        "blob_id": int(parent_blob_id),
                        "node_id": int(node_id),
                        "node_role": str(node_role),
                        "line_id": 0,
                        "line_order": 0,
                        "is_seed": False,
                        "is_candidate_seed": True,
                        "bin_id": int(node.get("bin_id", -1)),
                        "degree": int(node_degrees.get(int(node_id), 0)),
                        "support_x_ras": float(support_point[0]),
                        "support_y_ras": float(support_point[1]),
                        "support_z_ras": float(support_point[2]),
                    }
                )
        return rows

    def _build_complex_blob_sampling_rows(
        self,
        parent_blob_id,
        graph,
    ):
        node_map = dict(dict(graph or {}).get("node_map") or {})
        node_degrees = dict(dict(graph or {}).get("node_degrees") or {})
        rows = []
        for node_id in sorted(int(v) for v in node_map.keys()):
            node = dict(node_map.get(int(node_id)) or {})
            support_point = np.asarray(node.get("support_point_ras"), dtype=float).reshape(-1)
            if support_point.size != 3:
                support_point = np.asarray(node.get("center_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
            rows.append(
                {
                    "blob_id": int(parent_blob_id),
                    "node_id": int(node_id),
                    "node_role": str(node.get("node_role") or "core"),
                    "line_id": 0,
                    "line_order": 0,
                    "bin_id": int(node.get("bin_id", -1)),
                    "degree": int(node_degrees.get(int(node_id), 0)),
                    "support_x_ras": float(support_point[0]),
                    "support_y_ras": float(support_point[1]),
                    "support_z_ras": float(support_point[2]),
                }
            )
        return rows

    def _extend_complex_blob_path_to_exhaustion(
        self,
        node_map,
        adj,
        node_ids,
        blocked_node_ids=None,
        max_extension_steps=12,
        candidate_guard_diameter_mm=4.0,
    ):
        path = [int(v) for v in list(node_ids or []) if int(v) > 0]
        if len(path) < 2:
            return path
        blocked = set(int(v) for v in list(blocked_node_ids or []) if int(v) > 0 and int(v) not in path)
        for _ in range(int(max_extension_steps)):
            changed = False
            for prepend in (False, True):
                if len(path) < 2:
                    break
                if prepend:
                    current_id = int(path[0])
                    prev_id = int(path[1])
                    travel_dir = np.asarray(node_map[int(current_id)].get("center_ras"), dtype=float).reshape(3) - np.asarray(node_map[int(prev_id)].get("center_ras"), dtype=float).reshape(3)
                else:
                    current_id = int(path[-1])
                    prev_id = int(path[-2])
                    travel_dir = np.asarray(node_map[int(current_id)].get("center_ras"), dtype=float).reshape(3) - np.asarray(node_map[int(prev_id)].get("center_ras"), dtype=float).reshape(3)
                travel_norm = float(np.linalg.norm(travel_dir))
                if travel_norm <= 1e-6:
                    continue
                travel_dir = travel_dir / travel_norm
                current_node = dict(node_map.get(int(current_id)) or {})
                current_role = str(current_node.get("node_role") or "")
                if current_role == "crossing":
                    crossing_search = self._search_crossing_exit_path(
                        node_map=node_map,
                        adj=adj,
                        start_id=int(current_id),
                        prev_id=int(prev_id),
                        visited=set(path),
                        blocked=blocked,
                        frozen_direction_ras=travel_dir,
                        max_hops=4,
                        lateral_max_mm=5.5,
                        cos_forward=float(np.cos(np.deg2rad(65.0))),
                    )
                    if crossing_search is not None:
                        extension = [int(v) for v in list(crossing_search.get("path_node_ids") or []) if int(v) > 0]
                        if extension:
                            if prepend:
                                path = list(reversed(extension)) + path
                            else:
                                path.extend(extension)
                            changed = True
                            continue
                best = self._select_forward_path_neighbor(
                    node_map=node_map,
                    adj=adj,
                    current_id=int(current_id),
                    prev_id=int(prev_id),
                    active_direction_ras=travel_dir,
                    visited=set(path),
                    blocked=blocked,
                    lateral_max_mm=4.5,
                    cos_forward=float(np.cos(np.deg2rad(65.0))),
                )
                if best is None:
                    continue
                next_id = int(best.get("neighbor_id", -1))
                if next_id <= 0:
                    continue
                candidate_path = ([int(next_id)] + path) if prepend else (path + [int(next_id)])
                if not self._complex_blob_path_respects_diameter_guard(
                    node_map=node_map,
                    node_ids=candidate_path,
                    guard_diameter_mm=float(candidate_guard_diameter_mm),
                ):
                    continue
                if prepend:
                    path.insert(0, int(next_id))
                else:
                    path.append(int(next_id))
                changed = True
            if not changed:
                break
        return [int(v) for idx, v in enumerate(path) if idx == 0 or int(v) != int(path[idx - 1])]

    def _select_forward_path_neighbor(
        self,
        node_map,
        adj,
        current_id,
        prev_id,
        active_direction_ras,
        visited,
        blocked,
        lateral_max_mm=4.5,
        cos_forward=0.42,
    ):
        current_node = dict(node_map.get(int(current_id)) or {})
        if not current_node:
            return None
        current_center = np.asarray(current_node.get("center_ras"), dtype=float).reshape(3)
        active_direction = np.asarray(active_direction_ras, dtype=float).reshape(3)
        active_direction = active_direction / max(float(np.linalg.norm(active_direction)), 1e-6)
        forward_candidates = []
        for edge in list(adj.get(int(current_id)) or []):
            nb_id = int(edge.get("neighbor_id", -1))
            if nb_id <= 0 or nb_id == int(prev_id) or nb_id in set(visited or []) or nb_id in set(blocked or []):
                continue
            nb_node = dict(node_map.get(int(nb_id)) or {})
            if not nb_node:
                continue
            nb_center = np.asarray(nb_node.get("center_ras"), dtype=float).reshape(3)
            step = nb_center - current_center
            step_norm = float(np.linalg.norm(step))
            if step_norm <= 1e-6:
                continue
            step_dir = step / step_norm
            forward_mm = float(np.dot(step, active_direction))
            if forward_mm <= 0.0:
                continue
            lateral_vec = step - forward_mm * active_direction
            lateral_mm = float(np.linalg.norm(lateral_vec))
            if lateral_mm > float(lateral_max_mm):
                continue
            align = float(np.dot(step_dir, active_direction))
            if align < float(cos_forward):
                continue
            forward_candidates.append(
                {
                    "neighbor_id": int(nb_id),
                    "neighbor_role": str(nb_node.get("node_role") or ""),
                    "forward_mm": float(forward_mm),
                    "lateral_mm": float(lateral_mm),
                    "step_norm": float(step_norm),
                    "align": float(align),
                    "score": 2.4 * align - 0.10 * float(forward_mm) - 0.18 * float(lateral_mm) + 0.25 * float(edge.get("score", 0.0)),
                }
            )
        if not forward_candidates:
            return None
        forward_candidates.sort(key=lambda item: (float(item.get("forward_mm", 999.0)), float(item.get("lateral_mm", 999.0)), -float(item.get("score", -999.0))))
        return dict(forward_candidates[0])

    def _trace_complex_blob_directional_path(
        self,
        node_map,
        adj,
        start_id,
        initial_direction_ras,
        max_steps=10,
        blocked_node_ids=None,
        candidate_guard_diameter_mm=4.0,
    ):
        start_id = int(start_id)
        if start_id <= 0 or start_id not in node_map:
            return []
        direction = np.asarray(initial_direction_ras, dtype=float).reshape(3)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-6:
            return [int(start_id)]
        direction = direction / direction_norm
        path = [int(start_id)]
        visited = {int(start_id)}
        blocked = set(int(v) for v in list(blocked_node_ids or []) if int(v) > 0 and int(v) != int(start_id))
        prev_id = -1
        current_id = int(start_id)
        cos_forward = float(np.cos(np.deg2rad(65.0)))
        crossing_mode = False
        frozen_crossing_direction = None
        lateral_max_mm = 4.5
        for _ in range(int(max_steps)):
            current_node = dict(node_map.get(int(current_id)) or {})
            if not current_node:
                break
            path_centers = np.asarray(
                [
                    np.asarray(dict(node_map.get(int(node_id)) or {}).get("center_ras"), dtype=float).reshape(3)
                    for node_id in path
                    if dict(node_map.get(int(node_id)) or {}).get("center_ras") is not None
                ],
                dtype=float,
            ).reshape(-1, 3)
            if path_centers.shape[0] >= 2:
                _fit_center, fit_axis = self._fit_line_pca(path_centers, seed_axis=direction)
                fit_axis = np.asarray(fit_axis, dtype=float).reshape(3)
                fit_axis_norm = float(np.linalg.norm(fit_axis))
                if fit_axis_norm > 1e-6:
                    fit_axis = fit_axis / fit_axis_norm
                    if float(np.dot(fit_axis, direction)) < 0.0:
                        fit_axis = -fit_axis
                    direction = fit_axis
            current_center = np.asarray(current_node.get("center_ras"), dtype=float).reshape(3)
            current_role = str(current_node.get("node_role") or "")
            if current_role == "crossing":
                crossing_mode = True
                if frozen_crossing_direction is None:
                    frozen_crossing_direction = np.asarray(direction, dtype=float).reshape(3)
            best_id = None
            best_score = None
            best_dir = None
            active_direction = np.asarray(
                frozen_crossing_direction if crossing_mode and frozen_crossing_direction is not None else direction,
                dtype=float,
            ).reshape(3)
            if crossing_mode:
                crossing_search = self._search_crossing_exit_path(
                    node_map=node_map,
                    adj=adj,
                    start_id=int(current_id),
                    prev_id=int(prev_id),
                    visited=visited,
                    blocked=blocked,
                    frozen_direction_ras=active_direction,
                    max_hops=6,
                    lateral_max_mm=lateral_max_mm,
                    cos_forward=cos_forward,
                )
                if crossing_search is not None:
                    extension = [int(v) for v in list(crossing_search.get("path_node_ids") or []) if int(v) > 0]
                    if extension:
                        candidate_path = path + [int(v) for v in extension]
                        if not self._complex_blob_path_respects_diameter_guard(
                            node_map=node_map,
                            node_ids=candidate_path,
                            guard_diameter_mm=float(candidate_guard_diameter_mm),
                        ):
                            break
                        for next_id in extension:
                            prev_id = int(current_id)
                            current_id = int(next_id)
                            path.append(int(current_id))
                            visited.add(int(current_id))
                        current_node = dict(node_map.get(int(current_id)) or {})
                        current_role = str(current_node.get("node_role") or "")
                        if current_role == "crossing":
                            continue
                        crossing_mode = False
                        frozen_crossing_direction = None
                        path_centers = np.asarray(
                            [
                                np.asarray(dict(node_map.get(int(node_id)) or {}).get("center_ras"), dtype=float).reshape(3)
                                for node_id in path
                                if dict(node_map.get(int(node_id)) or {}).get("center_ras") is not None
                            ],
                            dtype=float,
                        ).reshape(-1, 3)
                        if path_centers.shape[0] >= 2:
                            _fit_center, fit_axis = self._fit_line_pca(path_centers, seed_axis=active_direction)
                            fit_axis = np.asarray(fit_axis, dtype=float).reshape(3)
                            fit_axis_norm = float(np.linalg.norm(fit_axis))
                            if fit_axis_norm > 1e-6:
                                fit_axis = fit_axis / fit_axis_norm
                                if float(np.dot(fit_axis, active_direction)) < 0.0:
                                    fit_axis = -fit_axis
                                direction = fit_axis
                                continue
                        direction = np.asarray(active_direction, dtype=float).reshape(3)
                        continue
            forward_candidates = []
            for edge in list(adj.get(int(current_id)) or []):
                nb_id = int(edge.get("neighbor_id", -1))
                if nb_id <= 0 or nb_id == int(prev_id) or nb_id in visited or nb_id in blocked:
                    continue
                nb_node = dict(node_map.get(int(nb_id)) or {})
                if not nb_node:
                    continue
                nb_center = np.asarray(nb_node.get("center_ras"), dtype=float).reshape(3)
                step = nb_center - current_center
                step_norm = float(np.linalg.norm(step))
                if step_norm <= 1e-6:
                    continue
                step_dir = step / step_norm
                forward_mm = float(np.dot(step, active_direction))
                if forward_mm <= 0.0:
                    continue
                lateral_vec = step - forward_mm * active_direction
                lateral_mm = float(np.linalg.norm(lateral_vec))
                if lateral_mm > float(lateral_max_mm):
                    continue
                forward_candidates.append(
                    {
                        "edge": dict(edge or {}),
                        "neighbor_id": int(nb_id),
                        "neighbor_node": dict(nb_node),
                        "step": np.asarray(step, dtype=float).reshape(3),
                        "step_norm": float(step_norm),
                        "step_dir": np.asarray(step_dir, dtype=float).reshape(3),
                        "forward_mm": float(forward_mm),
                        "lateral_mm": float(lateral_mm),
                    }
                )
            forward_candidates.sort(key=lambda item: (float(item.get("forward_mm", 999.0)), float(item.get("lateral_mm", 999.0)), float(item.get("step_norm", 999.0))))
            for candidate in forward_candidates[:4]:
                edge = dict(candidate.get("edge") or {})
                nb_id = int(candidate.get("neighbor_id", -1))
                nb_node = dict(candidate.get("neighbor_node") or {})
                step_dir = np.asarray(candidate.get("step_dir"), dtype=float).reshape(3)
                align = float(np.dot(step_dir, active_direction))
                if align < cos_forward:
                    continue
                nb_axis_val = nb_node.get("primary_axis_ras")
                nb_axis_align = 0.0
                if nb_axis_val is not None:
                    nb_axis = np.asarray(nb_axis_val, dtype=float).reshape(3)
                    nb_axis_norm = float(np.linalg.norm(nb_axis))
                    if nb_axis_norm > 1e-6:
                        nb_axis = nb_axis / nb_axis_norm
                        nb_axis_align = float(abs(np.dot(step_dir, nb_axis)))
                nb_role = str(nb_node.get("node_role") or "")
                candidate_score = (
                    2.4 * align
                    + 0.35 * nb_axis_align
                    + 0.25 * float(edge.get("score", 0.0))
                    - 0.10 * float(candidate.get("forward_mm", 0.0))
                    - 0.18 * float(candidate.get("lateral_mm", 0.0))
                )
                if crossing_mode and nb_role == "core":
                    candidate_score += 0.35
                if crossing_mode and nb_role == "crossing":
                    candidate_score += 0.10
                if best_score is None or candidate_score > best_score:
                    best_id = int(nb_id)
                    best_score = float(candidate_score)
                    best_dir = np.asarray(step_dir, dtype=float).reshape(3)
            if best_id is None:
                break
            candidate_path = path + [int(best_id)]
            if not self._complex_blob_path_respects_diameter_guard(
                node_map=node_map,
                node_ids=candidate_path,
                guard_diameter_mm=float(candidate_guard_diameter_mm),
            ):
                break
            prev_id = int(current_id)
            current_id = int(best_id)
            path.append(int(current_id))
            visited.add(int(current_id))
            current_node = dict(node_map.get(int(current_id)) or {})
            current_role = str(current_node.get("node_role") or "")
            if current_role == "crossing":
                crossing_mode = True
                if frozen_crossing_direction is None:
                    frozen_crossing_direction = np.asarray(direction, dtype=float).reshape(3)
                continue
            crossing_mode = False
            frozen_crossing_direction = None
            path_centers = np.asarray(
                [
                    np.asarray(dict(node_map.get(int(node_id)) or {}).get("center_ras"), dtype=float).reshape(3)
                    for node_id in path
                    if dict(node_map.get(int(node_id)) or {}).get("center_ras") is not None
                ],
                dtype=float,
            ).reshape(-1, 3)
            if path_centers.shape[0] >= 2:
                _fit_center, fit_axis = self._fit_line_pca(path_centers, seed_axis=best_dir)
                fit_axis = np.asarray(fit_axis, dtype=float).reshape(3)
                fit_axis_norm = float(np.linalg.norm(fit_axis))
                if fit_axis_norm > 1e-6:
                    fit_axis = fit_axis / fit_axis_norm
                    if float(np.dot(fit_axis, best_dir)) < 0.0:
                        fit_axis = -fit_axis
                    direction = fit_axis
                    continue
            direction = np.asarray(best_dir, dtype=float).reshape(3)
        return [int(v) for v in path]

    @staticmethod
    def _complex_blob_path_fit_points(node_map, node_ids):
        node_list = [dict(node_map.get(int(node_id)) or {}) for node_id in list(node_ids or []) if int(node_id) > 0]
        node_list = [node for node in node_list if node]
        if len(node_list) < 2:
            return np.empty((0, 3), dtype=float)
        fit_point_groups = []
        for node in node_list:
            node_role = str(node.get("node_role") or "")
            if node_role == "crossing":
                fit_point_groups.append(np.asarray(node.get("support_point_ras"), dtype=float).reshape(1, 3))
                continue
            node_points = np.asarray(node.get("points_ras"), dtype=float).reshape(-1, 3)
            if node_points.shape[0] == 0:
                fit_point_groups.append(np.asarray(node.get("support_point_ras"), dtype=float).reshape(1, 3))
            else:
                fit_point_groups.append(node_points)
        if not fit_point_groups:
            return np.empty((0, 3), dtype=float)
        return np.asarray(np.vstack(fit_point_groups), dtype=float).reshape(-1, 3)

    def _complex_blob_path_respects_diameter_guard(
        self,
        node_map,
        node_ids,
        guard_diameter_mm=4.0,
    ):
        node_id_list = [int(v) for v in list(node_ids or []) if int(v) > 0]
        if len(node_id_list) < 3:
            return True
        fit_points = self._complex_blob_path_fit_points(node_map=node_map, node_ids=node_id_list)
        if fit_points.shape[0] < 3:
            return True
        support_points = np.asarray(
            [
                np.asarray(dict(node_map.get(int(node_id)) or {}).get("support_point_ras"), dtype=float).reshape(3)
                for node_id in node_id_list
                if dict(node_map.get(int(node_id)) or {}).get("support_point_ras") is not None
            ],
            dtype=float,
        ).reshape(-1, 3)
        if support_points.shape[0] < 2:
            return True
        seed_axis = support_points[-1] - support_points[0]
        _fit_center, fit_axis = self._fit_line_pca(fit_points, seed_axis=seed_axis)
        shape = self._point_cloud_shape_metrics(fit_points, seed_axis=fit_axis)
        diameter_mm = float(shape.get("diameter_mm", 0.0))
        return bool(diameter_mm <= float(max(1.0, guard_diameter_mm)))

    def _complex_blob_path_quality_metrics(
        self,
        node_map,
        node_ids,
    ):
        node_id_list = [int(v) for v in list(node_ids or []) if int(v) > 0]
        if len(node_id_list) < 2:
            return None
        fit_points = self._complex_blob_path_fit_points(node_map=node_map, node_ids=node_id_list)
        if fit_points.shape[0] < 2:
            return None
        support_points = np.asarray(
            [
                np.asarray(dict(node_map.get(int(node_id)) or {}).get("support_point_ras"), dtype=float).reshape(3)
                for node_id in node_id_list
                if dict(node_map.get(int(node_id)) or {}).get("support_point_ras") is not None
            ],
            dtype=float,
        ).reshape(-1, 3)
        if support_points.shape[0] < 2:
            return None
        seed_axis = support_points[-1] - support_points[0]
        fit_center, fit_axis = self._fit_line_pca(fit_points, seed_axis=seed_axis)
        radial = self._radial_distance_to_line(fit_points, fit_center, fit_axis)
        rms_mm = float(np.sqrt(np.mean(radial ** 2))) if radial.size > 0 else float("inf")
        proj = ((fit_points - fit_center.reshape(1, 3)) @ fit_axis.reshape(3, 1)).reshape(-1)
        span_mm = float(np.max(proj) - np.min(proj)) if proj.size > 0 else 0.0
        shape = self._point_cloud_shape_metrics(fit_points, seed_axis=fit_axis)
        diameter_mm = float(shape.get("diameter_mm", 0.0))
        return {
            "diameter_mm": float(diameter_mm),
            "rms_mm": float(rms_mm),
            "span_mm": float(span_mm),
        }

    def _trim_complex_blob_path_to_narrow_core(
        self,
        node_map,
        node_ids,
        target_diameter_mm=5.0,
        target_rms_mm=3.0,
        min_nodes=3,
    ):
        path = [int(v) for v in list(node_ids or []) if int(v) > 0]
        min_nodes = int(max(2, min_nodes))
        if len(path) < min_nodes:
            return path
        best_valid_path = None
        best_valid_metrics = None
        for start_idx in range(0, len(path)):
            for end_idx in range(start_idx + min_nodes, len(path) + 1):
                subpath = list(path[start_idx:end_idx])
                metrics = self._complex_blob_path_quality_metrics(node_map=node_map, node_ids=subpath)
                if metrics is None:
                    continue
                if (
                    float(metrics.get("diameter_mm", float("inf"))) <= float(target_diameter_mm)
                    and float(metrics.get("rms_mm", float("inf"))) <= float(target_rms_mm)
                ):
                    if best_valid_metrics is None or (
                        float(metrics.get("span_mm", 0.0)),
                        -float(metrics.get("diameter_mm", float("inf"))),
                        -float(metrics.get("rms_mm", float("inf"))),
                    ) > (
                        float(best_valid_metrics.get("span_mm", 0.0)),
                        -float(best_valid_metrics.get("diameter_mm", float("inf"))),
                        -float(best_valid_metrics.get("rms_mm", float("inf"))),
                    ):
                        best_valid_path = list(subpath)
                        best_valid_metrics = dict(metrics)
        if best_valid_path is not None:
            return list(best_valid_path)
        best_path = list(path)
        best_metrics = self._complex_blob_path_quality_metrics(node_map=node_map, node_ids=best_path)
        if best_metrics is None:
            return path
        while len(path) > min_nodes:
            metrics = self._complex_blob_path_quality_metrics(node_map=node_map, node_ids=path)
            if metrics is None:
                break
            if (
                float(metrics.get("diameter_mm", float("inf"))) <= float(target_diameter_mm)
                and float(metrics.get("rms_mm", float("inf"))) <= float(target_rms_mm)
            ):
                return list(path)
            front = list(path[1:])
            back = list(path[:-1])
            front_metrics = self._complex_blob_path_quality_metrics(node_map=node_map, node_ids=front)
            back_metrics = self._complex_blob_path_quality_metrics(node_map=node_map, node_ids=back)
            candidates = []
            for cand_path, cand_metrics in ((front, front_metrics), (back, back_metrics)):
                if cand_metrics is None or len(cand_path) < min_nodes:
                    continue
                penalty = (
                    2.0 * max(0.0, float(cand_metrics.get("rms_mm", float("inf"))) - float(target_rms_mm))
                    + max(0.0, float(cand_metrics.get("diameter_mm", float("inf"))) - float(target_diameter_mm))
                    - 0.05 * float(cand_metrics.get("span_mm", 0.0))
                )
                candidates.append((float(penalty), list(cand_path), dict(cand_metrics)))
            if not candidates:
                break
            candidates.sort(key=lambda item: (float(item[0]), -float(item[2].get("span_mm", 0.0))))
            _penalty, path, cand_metrics = candidates[0]
            if (
                best_metrics is None
                or (
                    float(cand_metrics.get("diameter_mm", float("inf"))),
                    float(cand_metrics.get("rms_mm", float("inf"))),
                    -float(cand_metrics.get("span_mm", 0.0)),
                )
                < (
                    float(best_metrics.get("diameter_mm", float("inf"))),
                    float(best_metrics.get("rms_mm", float("inf"))),
                    -float(best_metrics.get("span_mm", 0.0)),
                )
            ):
                best_path = list(path)
                best_metrics = dict(cand_metrics)
        return list(best_path)

    def _search_crossing_exit_path(
        self,
        node_map,
        adj,
        start_id,
        prev_id,
        visited,
        blocked,
        frozen_direction_ras,
        max_hops=3,
        lateral_max_mm=4.5,
        cos_forward=0.42,
    ):
        start_id = int(start_id)
        if start_id <= 0 or start_id not in node_map:
            return None
        frozen_direction = np.asarray(frozen_direction_ras, dtype=float).reshape(3)
        frozen_direction = frozen_direction / max(float(np.linalg.norm(frozen_direction)), 1e-6)
        start_node = dict(node_map.get(int(start_id)) or {})
        start_center = np.asarray(start_node.get("center_ras"), dtype=float).reshape(3)
        best = None
        best_score = None
        current_id = int(start_id)
        current_prev = int(prev_id)
        path = []
        local_visited = set(int(v) for v in list(visited or []) if int(v) > 0)
        local_visited.add(int(start_id))
        for hop_count in range(int(max_hops)):
            current_node = dict(node_map.get(int(current_id)) or {})
            if not current_node:
                break
            current_center = np.asarray(current_node.get("center_ras"), dtype=float).reshape(3)
            crossing_candidates = []
            core_candidates = []
            for edge in list(adj.get(int(current_id)) or []):
                nb_id = int(edge.get("neighbor_id", -1))
                if nb_id <= 0 or nb_id == int(current_prev) or nb_id in local_visited or nb_id in blocked:
                    continue
                nb_node = dict(node_map.get(int(nb_id)) or {})
                if not nb_node:
                    continue
                nb_center = np.asarray(nb_node.get("center_ras"), dtype=float).reshape(3)
                step = nb_center - current_center
                step_norm = float(np.linalg.norm(step))
                if step_norm <= 1e-6:
                    continue
                forward_mm = float(np.dot(step, frozen_direction))
                if forward_mm <= 0.0:
                    continue
                lateral_vec = step - forward_mm * frozen_direction
                lateral_mm = float(np.linalg.norm(lateral_vec))
                if lateral_mm > float(1.6 * lateral_max_mm):
                    continue
                total_step = nb_center - start_center
                total_step_norm = float(np.linalg.norm(total_step))
                if total_step_norm <= 1e-6:
                    continue
                total_forward_mm = float(np.dot(total_step, frozen_direction))
                if total_forward_mm <= 0.0:
                    continue
                total_lateral_vec = total_step - total_forward_mm * frozen_direction
                total_lateral_mm = float(np.linalg.norm(total_lateral_vec))
                if total_lateral_mm > float(1.8 * lateral_max_mm):
                    continue
                nb_role = str(nb_node.get("node_role") or "")
                candidate = {
                    "neighbor_id": int(nb_id),
                    "neighbor_role": str(nb_role),
                    "candidate_path": list(path) + [int(nb_id)],
                    "forward_mm": float(forward_mm),
                    "lateral_mm": float(lateral_mm),
                    "total_forward_mm": float(total_forward_mm),
                    "total_lateral_mm": float(total_lateral_mm),
                }
                if nb_role == "crossing":
                    candidate["score"] = (
                        2.8 * float(total_forward_mm)
                        - 0.65 * float(total_lateral_mm)
                        - 0.10 * float(hop_count)
                    )
                    crossing_candidates.append(candidate)
                    continue
                nb_axis_align = 0.0
                nb_axis_val = nb_node.get("primary_axis_ras")
                if nb_axis_val is not None:
                    nb_axis = np.asarray(nb_axis_val, dtype=float).reshape(3)
                    nb_axis_norm = float(np.linalg.norm(nb_axis))
                    if nb_axis_norm > 1e-6:
                        nb_axis = nb_axis / nb_axis_norm
                        nb_axis_align = float(abs(np.dot(nb_axis, frozen_direction)))
                candidate["nb_axis_align"] = float(nb_axis_align)
                candidate["score"] = (
                    2.4 * float(total_forward_mm)
                    - 0.50 * float(total_lateral_mm)
                    + 0.90 * float(nb_axis_align)
                    - 0.10 * float(hop_count)
                )
                if float(nb_axis_align) >= 0.70 or float(total_forward_mm) >= 2.0:
                    core_candidates.append(candidate)
            if crossing_candidates:
                crossing_candidates.sort(
                    key=lambda item: (
                        -float(item.get("score", -1e9)),
                        -float(item.get("total_forward_mm", -1e9)),
                        float(item.get("total_lateral_mm", 1e9)),
                        float(item.get("lateral_mm", 1e9)),
                    )
                )
                chosen = dict(crossing_candidates[0])
                next_id = int(chosen.get("neighbor_id", -1))
                if next_id <= 0:
                    break
                path.append(int(next_id))
                local_visited.add(int(next_id))
                current_prev = int(current_id)
                current_id = int(next_id)
                continue
            if core_candidates:
                core_candidates.sort(
                    key=lambda item: (
                        -float(item.get("score", -1e9)),
                        -float(item.get("nb_axis_align", -1e9)),
                        -float(item.get("total_forward_mm", -1e9)),
                        float(item.get("total_lateral_mm", 1e9)),
                    )
                )
                chosen = dict(core_candidates[0])
                if best_score is None or float(chosen.get("score", -1e9)) > float(best_score):
                    best = {
                        "path_node_ids": list(chosen.get("candidate_path") or []),
                        "exit_node_id": int(chosen.get("neighbor_id", -1)),
                    }
                    best_score = float(chosen.get("score", -1e9))
                break
            break
        return best

    @staticmethod
    def _complex_blob_node_role(degree):
        deg = int(degree)
        if deg <= 1:
            return "end"
        if deg == 2:
            return "core"
        return "crossing"

    @staticmethod
    def _complex_blob_paths_overlap(a, b):
        node_ids_a = set(int(v) for v in list(dict(a or {}).get("node_ids") or []) if int(v) > 0)
        node_ids_b = set(int(v) for v in list(dict(b or {}).get("node_ids") or []) if int(v) > 0)
        if not node_ids_a or not node_ids_b:
            overlap_ratio = 0.0
        else:
            overlap = int(len(node_ids_a & node_ids_b))
            overlap_ratio = float(overlap / max(1, min(len(node_ids_a), len(node_ids_b))))
            if overlap_ratio >= 0.75:
                return True
        axis_a_val = dict(a or {}).get("axis_ras")
        if axis_a_val is None:
            axis_a_val = [0.0, 0.0, 1.0]
        axis_b_val = dict(b or {}).get("axis_ras")
        if axis_b_val is None:
            axis_b_val = [0.0, 0.0, 1.0]
        axis_a = np.asarray(axis_a_val, dtype=float).reshape(3)
        axis_b = np.asarray(axis_b_val, dtype=float).reshape(3)
        axis_dot = abs(float(np.dot(axis_a / max(np.linalg.norm(axis_a), 1e-6), axis_b / max(np.linalg.norm(axis_b), 1e-6))))
        if axis_dot < 0.97:
            return False
        center_a_val = dict(a or {}).get("center_ras")
        if center_a_val is None:
            center_a_val = [0.0, 0.0, 0.0]
        center_b_val = dict(b or {}).get("center_ras")
        if center_b_val is None:
            center_b_val = [0.0, 0.0, 0.0]
        center_a = np.asarray(center_a_val, dtype=float).reshape(3)
        center_b = np.asarray(center_b_val, dtype=float).reshape(3)
        axis_a = axis_a / max(float(np.linalg.norm(axis_a)), 1e-6)
        axis_b = axis_b / max(float(np.linalg.norm(axis_b)), 1e-6)

        def _point_line_distance(point, line_point, line_axis):
            pt = np.asarray(point, dtype=float).reshape(3)
            lp = np.asarray(line_point, dtype=float).reshape(3)
            la = np.asarray(line_axis, dtype=float).reshape(3)
            proj = float(np.dot(pt - lp, la))
            radial = pt - lp - proj * la
            return float(np.linalg.norm(radial))

        lateral_ab = _point_line_distance(center_b, center_a, axis_a)
        lateral_ba = _point_line_distance(center_a, center_b, axis_b)
        if min(lateral_ab, lateral_ba) > 3.5:
            return False

        start_a_val = dict(a or {}).get("start_ras")
        if start_a_val is None:
            start_a_val = center_a
        end_a_val = dict(a or {}).get("end_ras")
        if end_a_val is None:
            end_a_val = center_a
        start_b_val = dict(b or {}).get("start_ras")
        if start_b_val is None:
            start_b_val = center_b
        end_b_val = dict(b or {}).get("end_ras")
        if end_b_val is None:
            end_b_val = center_b
        start_a = np.asarray(start_a_val, dtype=float).reshape(3)
        end_a = np.asarray(end_a_val, dtype=float).reshape(3)
        start_b = np.asarray(start_b_val, dtype=float).reshape(3)
        end_b = np.asarray(end_b_val, dtype=float).reshape(3)
        a0 = float(np.dot(start_a - center_a, axis_a))
        a1 = float(np.dot(end_a - center_a, axis_a))
        a_min = float(min(a0, a1))
        a_max = float(max(a0, a1))
        b0 = float(np.dot(start_b - center_a, axis_a))
        b1 = float(np.dot(end_b - center_a, axis_a))
        b_min = float(min(b0, b1))
        b_max = float(max(b0, b1))
        overlap_mm = float(max(0.0, min(a_max, b_max) - max(a_min, b_min)))
        span_a = float(np.linalg.norm(end_a - start_a))
        span_b = float(np.linalg.norm(end_b - start_b))
        shorter_span = float(max(1e-6, min(span_a, span_b)))
        if overlap_mm / shorter_span >= 0.60:
            return True
        return bool(overlap_ratio >= 0.60 and min(lateral_ab, lateral_ba) <= 2.5)

    def _complex_blob_path_summary(
        self,
        node_map,
        node_ids,
        sample_spacing_mm=2.5,
        min_coverage=0.90,
        max_gap_bins=1,
    ):
        node_list = [node_map.get(int(node_id)) for node_id in list(node_ids or [])]
        node_list = [node for node in node_list if node is not None]
        if len(node_list) < 2:
            return None
        support_points = np.asarray([node.get("support_point_ras") for node in node_list], dtype=float).reshape(-1, 3)
        fit_point_groups = []
        for node in node_list:
            node_role = str(node.get("node_role") or "")
            if node_role == "crossing":
                fit_point_groups.append(np.asarray(node.get("support_point_ras"), dtype=float).reshape(1, 3))
                continue
            node_points = np.asarray(node.get("points_ras"), dtype=float).reshape(-1, 3)
            if node_points.shape[0] == 0:
                fit_point_groups.append(np.asarray(node.get("support_point_ras"), dtype=float).reshape(1, 3))
            else:
                fit_point_groups.append(node_points)
        fit_points = np.asarray(np.vstack(fit_point_groups), dtype=float).reshape(-1, 3) if fit_point_groups else np.empty((0, 3), dtype=float)
        if fit_points.shape[0] < 2:
            return None
        seed_axis = support_points[-1] - support_points[0]
        fit_center, fit_axis = self._fit_line_pca(fit_points, seed_axis=seed_axis)
        radial = self._radial_distance_to_line(fit_points, fit_center, fit_axis)
        rms = float(np.sqrt(np.mean(radial ** 2)))
        proj_pts = ((fit_points - fit_center.reshape(1, 3)) @ fit_axis.reshape(3, 1)).reshape(-1)
        pmin = float(np.min(proj_pts))
        pmax = float(np.max(proj_pts))
        span = float(pmax - pmin)
        occupancy_bin_mm = float(max(0.5, sample_spacing_mm))
        if all(dict(node or {}).get("cube_coord") is not None for node in node_list):
            occupancy_bin_mm = float(
                self._adaptive_projected_support_bin_mm(
                    proj_vals=proj_pts,
                    base_bin_mm=occupancy_bin_mm,
                    max_scale=2.5,
                )
            )
        occ = self._axial_occupancy_metrics(proj_pts, bin_mm=float(occupancy_bin_mm))
        shape = self._point_cloud_shape_metrics(fit_points, seed_axis=fit_axis)
        diameter_mm = float(shape.get("diameter_mm", 0.0))
        coverage_required = float(min_coverage)
        if (
            span >= 24.0
            and diameter_mm <= 3.5
            and rms <= 1.25
            and int(occ.get("max_gap_bins", 0)) <= int(max_gap_bins)
        ):
            coverage_required = min(float(coverage_required), 0.80)
        if span < float(max(7.0, 2.5 * sample_spacing_mm)):
            return None
        if diameter_mm > 8.0:
            return None
        if rms > 3.0:
            return None
        if len(node_list) < 3:
            if span < 12.0 or diameter_mm > 5.0:
                return None
        elif (
            float(occ.get("coverage", 0.0)) < float(coverage_required)
            or int(occ.get("max_gap_bins", 0)) > int(max_gap_bins)
        ):
            return None
        score = (
            1.0 * span
            + 2.5 * float(len(node_list))
            + 6.0 * float(occ.get("coverage", 0.0))
            - 2.0 * float(rms)
            - 0.7 * float(diameter_mm)
            - 1.2 * float(occ.get("max_gap_bins", 0))
        )
        return {
            "node_ids": [int(node.get("node_id", -1)) for node in node_list if int(node.get("node_id", -1)) > 0],
            "token_points_ras": np.asarray([node.get("support_point_ras") for node in node_list], dtype=float).reshape(-1, 3),
            "axis_ras": np.asarray(fit_axis, dtype=float).reshape(3),
            "center_ras": np.asarray(fit_center, dtype=float).reshape(3),
            "start_ras": np.asarray(fit_center + pmin * fit_axis, dtype=float).reshape(3),
            "end_ras": np.asarray(fit_center + pmax * fit_axis, dtype=float).reshape(3),
            "span_mm": float(span),
            "diameter_mm": float(diameter_mm),
            "elongation": float(max(1.0, span / max(0.5, diameter_mm))),
            "score": float(score),
            "fit_points_ras": np.asarray(fit_points, dtype=float).reshape(-1, 3),
        }
