"""Raw candidate generation from Deep Core support atoms and token graphs."""

try:
    import numpy as np
except ImportError:
    np = None


class DeepCoreCandidateGenerationMixin:
    """Build graph and blob-axis candidate proposals from support inputs."""

    def _build_token_graph(
        self,
        pts_ras,
        blob_ids,
        axis_map,
        elongation_map,
        neighbor_min_mm=1.0,
        neighbor_max_mm=8.0,
        neighbor_axis_angle_deg=30.0,
    ):
        pts = np.asarray(pts_ras, dtype=float).reshape(-1, 3)
        blob_ids = np.asarray(blob_ids, dtype=np.int32).reshape(-1)
        n = int(pts.shape[0])
        if n < 2:
            return {"adj": [[] for _ in range(n)], "seed_edges": []}
        cos_thresh = float(np.cos(np.deg2rad(float(max(0.0, neighbor_axis_angle_deg)))))
        min_sep = float(max(0.0, neighbor_min_mm))
        max_sep = float(max(min_sep, neighbor_max_mm))
        adj = [[] for _ in range(n)]
        seed_edges = []
        for i in range(n - 1):
            delta = pts[(i + 1) :, :] - pts[i].reshape(1, 3)
            dist = np.linalg.norm(delta, axis=1)
            valid = np.flatnonzero(np.logical_and(dist >= min_sep, dist <= max_sep))
            if valid.size == 0:
                continue
            for rel in valid.tolist():
                j = i + 1 + int(rel)
                direction = delta[int(rel)] / max(float(dist[int(rel)]), 1e-6)
                blob_i = int(blob_ids[i])
                blob_j = int(blob_ids[j])
                same_blob = blob_i == blob_j
                compat = False
                if same_blob:
                    compat = bool(float(elongation_map.get(blob_i, 1.0)) >= 2.0)
                for blob_id in (blob_i, blob_j):
                    axis = np.asarray(axis_map.get(int(blob_id), []), dtype=float).reshape(-1)
                    if axis.size != 3:
                        continue
                    axis_norm = float(np.linalg.norm(axis))
                    if axis_norm <= 1e-6:
                        continue
                    axis = axis / axis_norm
                    if abs(float(np.dot(direction, axis))) >= cos_thresh:
                        compat = True
                        break
                if compat:
                    edge_score = float(dist[int(rel)]) + (2.0 if not same_blob else 0.0)
                    adj[i].append((int(j), float(dist[int(rel)]), direction))
                    adj[j].append((int(i), float(dist[int(rel)]), -direction))
                    seed_edges.append((edge_score, int(i), int(j)))
        seed_edges.sort(key=lambda item: float(item[0]), reverse=True)
        return {"adj": adj, "seed_edges": seed_edges}

    def _support_atom_endpoint_candidates(self, node):
        center_val = node.get("center_ras")
        if center_val is None:
            center_val = [0.0, 0.0, 0.0]
        center = np.asarray(center_val, dtype=float).reshape(3)
        start_val = node.get("start_ras")
        if start_val is None:
            start_val = center
        end_val = node.get("end_ras")
        if end_val is None:
            end_val = center
        start = np.asarray(start_val, dtype=float).reshape(3)
        end = np.asarray(end_val, dtype=float).reshape(3)
        axis_val = node.get("axis_ras")
        if axis_val is None:
            axis_val = end - start
        axis = np.asarray(axis_val, dtype=float).reshape(3)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-6:
            axis = np.asarray([0.0, 0.0, 1.0], dtype=float)
        else:
            axis = axis / axis_norm
        port_extension_mm = 0.0
        if str(node.get("kind") or "") == "line" and str(node.get("source_blob_class") or "") == "complex_blob":
            port_extension_mm = float(
                max(
                    1.5,
                    min(
                        0.75 * float(max(0.5, node.get("span_mm", 0.0))),
                        max(2.5, 1.25 * float(max(0.5, node.get("thickness_mm", 0.0)))),
                    ),
                )
            )
        start_dir = start - center
        start_dir_norm = float(np.linalg.norm(start_dir))
        if start_dir_norm > 1e-6:
            start_dir = start_dir / start_dir_norm
        else:
            start_dir = -axis
        end_dir = end - center
        end_dir_norm = float(np.linalg.norm(end_dir))
        if end_dir_norm > 1e-6:
            end_dir = end_dir / end_dir_norm
        else:
            end_dir = axis
        if float(np.linalg.norm(end - start)) <= 1e-6:
            return [
                {
                    "point_ras": np.asarray(center, dtype=float).reshape(3),
                    "outward_axis_ras": np.asarray(axis, dtype=float).reshape(3),
                    "endpoint_name": "center",
                    "is_port": False,
                }
            ]
        candidates = [
            {
                "point_ras": np.asarray(start, dtype=float).reshape(3),
                "outward_axis_ras": np.asarray(start_dir, dtype=float).reshape(3),
                "endpoint_name": "start",
                "is_port": False,
            },
            {
                "point_ras": np.asarray(end, dtype=float).reshape(3),
                "outward_axis_ras": np.asarray(end_dir, dtype=float).reshape(3),
                "endpoint_name": "end",
                "is_port": False,
            },
        ]
        if port_extension_mm > 0.0:
            candidates.extend(
                [
                    {
                        "point_ras": np.asarray(start + start_dir * port_extension_mm, dtype=float).reshape(3),
                        "outward_axis_ras": np.asarray(start_dir, dtype=float).reshape(3),
                        "endpoint_name": "start_port",
                        "is_port": True,
                    },
                    {
                        "point_ras": np.asarray(end + end_dir * port_extension_mm, dtype=float).reshape(3),
                        "outward_axis_ras": np.asarray(end_dir, dtype=float).reshape(3),
                        "endpoint_name": "end_port",
                        "is_port": True,
                    },
                ]
            )
        return candidates

    def _score_blob_node_bridge(
        self,
        node_i,
        node_j,
        bridge_max_mm=16.0,
        neighbor_max_mm=8.0,
        cos_thresh=0.8,
    ):
        axis_i_val = node_i.get("axis_ras")
        if axis_i_val is None:
            axis_i_val = [0.0, 0.0, 1.0]
        axis_j_val = node_j.get("axis_ras")
        if axis_j_val is None:
            axis_j_val = [0.0, 0.0, 1.0]
        axis_i = np.asarray(axis_i_val, dtype=float).reshape(3)
        axis_j = np.asarray(axis_j_val, dtype=float).reshape(3)
        axis_i = axis_i / max(float(np.linalg.norm(axis_i)), 1e-6)
        axis_j = axis_j / max(float(np.linalg.norm(axis_j)), 1e-6)
        support_i = bool(
            node_i.get("axis_reliable")
            or float(node_i.get("span_mm", 0.0)) >= 4.0
            or float(node_i.get("elongation", 1.0)) >= 2.5
            or int(node_i.get("token_count", 0)) >= 3
        )
        support_j = bool(
            node_j.get("axis_reliable")
            or float(node_j.get("span_mm", 0.0)) >= 4.0
            or float(node_j.get("elongation", 1.0)) >= 2.5
            or int(node_j.get("token_count", 0)) >= 3
        )
        same_parent = bool(
            int(node_i.get("parent_blob_id", -1)) > 0
            and int(node_i.get("parent_blob_id", -1)) == int(node_j.get("parent_blob_id", -2))
        )
        kind_i = str(node_i.get("kind") or "")
        kind_j = str(node_j.get("kind") or "")
        source_i = str(node_i.get("source_blob_class") or "")
        source_j = str(node_j.get("source_blob_class") or "")
        line_line_pair = bool(kind_i == "line" and kind_j == "line")
        contact_chain_line_i = bool(kind_i == "line" and source_i == "contact_chain")
        contact_chain_line_j = bool(kind_j == "line" and source_j == "contact_chain")
        contact_chain_pair = bool(contact_chain_line_i and contact_chain_line_j)
        contact_chain_bridge = bool(contact_chain_line_i or contact_chain_line_j)
        effective_bridge_max_mm = float(max(1.0, bridge_max_mm))
        effective_neighbor_max_mm = float(max(1.0, neighbor_max_mm))
        if contact_chain_pair:
            effective_bridge_max_mm = float(max(effective_bridge_max_mm, 28.0))
            effective_neighbor_max_mm = float(max(effective_neighbor_max_mm, 14.0))
        elif line_line_pair:
            effective_bridge_max_mm = float(max(effective_bridge_max_mm, 24.0))
            effective_neighbor_max_mm = float(max(effective_neighbor_max_mm, 12.0))
        elif contact_chain_bridge:
            effective_bridge_max_mm = float(max(effective_bridge_max_mm, 22.0))
            effective_neighbor_max_mm = float(max(effective_neighbor_max_mm, 10.0))
        line_contact_pair = bool(
            (kind_i == "line" and kind_j == "contact")
            or (kind_i == "contact" and kind_j == "line")
        )
        supported_line_contact_pair = bool(
            line_contact_pair
            and not same_parent
            and (
                (kind_i == "line" and source_i in {"line_blob", "complex_blob"})
                or (kind_j == "line" and source_j in {"line_blob", "complex_blob"})
            )
        )
        if supported_line_contact_pair:
            return None
        complex_i = bool(str(node_i.get("source_blob_class") or "") == "complex_blob" and str(node_i.get("kind") or "") == "line")
        complex_j = bool(str(node_j.get("source_blob_class") or "") == "complex_blob" and str(node_j.get("kind") or "") == "line")
        best = None
        for cand_i in self._support_atom_endpoint_candidates(node_i):
            point_i = np.asarray(cand_i.get("point_ras"), dtype=float).reshape(3)
            out_i = np.asarray(cand_i.get("outward_axis_ras"), dtype=float).reshape(3)
            out_i = out_i / max(float(np.linalg.norm(out_i)), 1e-6)
            for cand_j in self._support_atom_endpoint_candidates(node_j):
                point_j = np.asarray(cand_j.get("point_ras"), dtype=float).reshape(3)
                out_j = np.asarray(cand_j.get("outward_axis_ras"), dtype=float).reshape(3)
                out_j = out_j / max(float(np.linalg.norm(out_j)), 1e-6)
                delta = np.asarray(point_j, dtype=float).reshape(3) - np.asarray(point_i, dtype=float).reshape(3)
                dist_mm = float(np.linalg.norm(delta))
                if dist_mm <= 1e-6 or dist_mm > float(effective_bridge_max_mm):
                    continue
                direction = delta / dist_mm
                align_i = abs(float(np.dot(direction, axis_i)))
                align_j = abs(float(np.dot(direction, axis_j)))
                if support_i and align_i < float(cos_thresh):
                    continue
                if support_j and align_j < float(cos_thresh):
                    continue
                endpoint_align_i = float(np.dot(direction, out_i))
                endpoint_align_j = float(np.dot(-direction, out_j))
                if bool(cand_i.get("is_port")) and endpoint_align_i < 0.45:
                    continue
                if bool(cand_j.get("is_port")) and endpoint_align_j < 0.45:
                    continue
                if not support_i and not support_j and dist_mm > float(max(9.0, effective_neighbor_max_mm + 1.0)):
                    continue
                midpoint = 0.5 * (np.asarray(point_i, dtype=float).reshape(3) + np.asarray(point_j, dtype=float).reshape(3))
                lateral_i = float(
                    self._radial_distance_to_line(
                        midpoint.reshape(1, 3),
                        np.asarray(node_i.get("center_ras"), dtype=float).reshape(3),
                        axis_i,
                    )[0]
                )
                lateral_j = float(
                    self._radial_distance_to_line(
                        midpoint.reshape(1, 3),
                        np.asarray(node_j.get("center_ras"), dtype=float).reshape(3),
                        axis_j,
                    )[0]
                )
                lateral_thresh_i = float(max(3.0, 0.75 * float(node_i.get("thickness_mm", 0.0)) + 1.5))
                lateral_thresh_j = float(max(3.0, 0.75 * float(node_j.get("thickness_mm", 0.0)) + 1.5))
                if support_i and lateral_i > lateral_thresh_i:
                    continue
                if support_j and lateral_j > lateral_thresh_j:
                    continue
                axis_agreement = abs(float(np.dot(axis_i, axis_j)))
                strong_line_line = bool(
                    line_line_pair
                    and axis_agreement >= 0.95
                    and endpoint_align_i >= 0.80
                    and endpoint_align_j >= 0.80
                    and min(lateral_i, lateral_j) <= 3.0
                )
                endpoint_pair_bonus = 0.0
                if contact_chain_bridge and endpoint_align_i >= 0.80 and endpoint_align_j >= 0.80 and axis_agreement >= 0.95:
                    endpoint_pair_bonus = 1.2 if contact_chain_pair else 0.6
                elif strong_line_line:
                    endpoint_pair_bonus = 0.6
                dist_penalty = (
                    0.10 if contact_chain_pair
                    else (0.12 if strong_line_line else (0.14 if contact_chain_bridge else 0.18))
                )
                score = (
                    2.5 * float(max(align_i, align_j))
                    + 0.9 * float(axis_agreement)
                    + 0.8 * float(max(0.0, endpoint_align_i))
                    + 0.8 * float(max(0.0, endpoint_align_j))
                    + 0.18 * float(min(node_i.get("span_mm", 0.0), 12.0))
                    + 0.18 * float(min(node_j.get("span_mm", 0.0), 12.0))
                    - float(dist_penalty) * float(dist_mm)
                    - 0.15 * float(min(lateral_i, lateral_j))
                    + (1.0 if same_parent else 0.0)
                    + (0.6 if dist_mm <= float(effective_neighbor_max_mm) else 0.0)
                    + (0.9 if bool(cand_i.get("is_port")) and not bool(cand_j.get("is_port")) else 0.0)
                    + (0.9 if bool(cand_j.get("is_port")) and not bool(cand_i.get("is_port")) else 0.0)
                    + (0.5 if (complex_i or complex_j) and not same_parent else 0.0)
                    + float(endpoint_pair_bonus)
                )
                candidate = {
                    "score": float(score),
                    "dist_mm": float(dist_mm),
                    "direction_ras": np.asarray(direction, dtype=float).reshape(3),
                    "axis_agreement": float(max(align_i, align_j)),
                    "same_parent": bool(same_parent),
                    "is_local_seed": bool(same_parent or dist_mm <= float(neighbor_max_mm)),
                }
                if best is None or float(candidate.get("score", -1e9)) > float(best.get("score", -1e9)):
                    best = candidate
        return best

    def _score_complex_port_bridge(
        self,
        complex_node,
        other_node,
        bridge_max_mm=30.0,
        cos_thresh=0.55,
    ):
        if str(complex_node.get("kind") or "") != "line":
            return None
        if str(complex_node.get("source_blob_class") or "") != "complex_blob":
            return None
        axis_other_val = other_node.get("axis_ras")
        if axis_other_val is None:
            axis_other_val = [0.0, 0.0, 1.0]
        axis_other = np.asarray(axis_other_val, dtype=float).reshape(3)
        axis_other = axis_other / max(float(np.linalg.norm(axis_other)), 1e-6)
        support_other = bool(
            other_node.get("axis_reliable")
            or float(other_node.get("span_mm", 0.0)) >= 4.0
            or float(other_node.get("elongation", 1.0)) >= 2.5
            or int(other_node.get("token_count", 0)) >= 3
        )
        best = None
        complex_candidates = [
            cand for cand in self._support_atom_endpoint_candidates(complex_node) if bool(cand.get("is_port"))
        ]
        if not complex_candidates:
            complex_candidates = list(self._support_atom_endpoint_candidates(complex_node))
        other_candidates = list(self._support_atom_endpoint_candidates(other_node))
        for cand_i in complex_candidates:
            point_i = np.asarray(cand_i.get("point_ras"), dtype=float).reshape(3)
            out_i = np.asarray(cand_i.get("outward_axis_ras"), dtype=float).reshape(3)
            out_i = out_i / max(float(np.linalg.norm(out_i)), 1e-6)
            for cand_j in other_candidates:
                point_j = np.asarray(cand_j.get("point_ras"), dtype=float).reshape(3)
                delta = point_j - point_i
                dist_mm = float(np.linalg.norm(delta))
                if dist_mm <= 1e-6 or dist_mm > float(max(1.0, bridge_max_mm)):
                    continue
                direction = delta / dist_mm
                port_align = float(np.dot(direction, out_i))
                if port_align < 0.35:
                    continue
                align_other = abs(float(np.dot(direction, axis_other)))
                if support_other and align_other < float(cos_thresh):
                    continue
                midpoint = 0.5 * (point_i + point_j)
                lateral_other = float(
                    self._radial_distance_to_line(
                        midpoint.reshape(1, 3),
                        np.asarray(other_node.get("center_ras"), dtype=float).reshape(3),
                        axis_other,
                    )[0]
                )
                lateral_thresh_other = float(max(4.5, 0.90 * float(other_node.get("thickness_mm", 0.0)) + 2.5))
                if support_other and lateral_other > lateral_thresh_other:
                    continue
                score = (
                    3.2 * float(port_align)
                    + 1.8 * float(align_other)
                    + 0.18 * float(min(other_node.get("span_mm", 0.0), 12.0))
                    - 0.14 * float(dist_mm)
                    - 0.10 * float(lateral_other)
                    + (0.8 if str(other_node.get("kind") or "") == "line" else 0.0)
                    + (0.5 if str(other_node.get("kind") or "") == "contact" else 0.0)
                    + (0.8 if not bool(cand_j.get("is_port")) else 0.0)
                )
                candidate = {
                    "score": float(score),
                    "dist_mm": float(dist_mm),
                    "direction_ras": np.asarray(direction, dtype=float).reshape(3),
                    "axis_agreement": float(max(port_align, align_other)),
                    "same_parent": False,
                    "is_local_seed": bool(dist_mm <= 12.0),
                }
                if best is None or float(candidate.get("score", -1e9)) > float(best.get("score", -1e9)):
                    best = candidate
        return best

    def _build_blob_connectivity_graph(
        self,
        pts_ras,
        blob_ids,
        token_atom_ids,
        axis_map,
        elongation_map,
        support_atoms=None,
        parent_blob_id_map=None,
        neighbor_max_mm=8.0,
        bridge_max_mm=16.0,
        bridge_axis_angle_deg=35.0,
    ):
        pts = np.asarray(pts_ras, dtype=float).reshape(-1, 3)
        blob_ids = np.asarray(blob_ids, dtype=np.int32).reshape(-1)
        token_atom_ids = np.asarray(token_atom_ids, dtype=np.int32).reshape(-1)
        n = int(pts.shape[0])
        if n == 0:
            return {"blob_nodes": {}, "adj": {}, "seed_edges": [], "local_seed_edges": [], "bridge_seed_edges": []}

        atom_map = {
            int(atom.get("atom_id", -1)): dict(atom)
            for atom in list(support_atoms or [])
            if int(dict(atom or {}).get("atom_id", -1)) > 0
        }
        blob_nodes = {}
        for atom_id, atom in atom_map.items():
            token_indices = np.flatnonzero(token_atom_ids == int(atom_id)).astype(int)
            if token_indices.size == 0:
                continue
            blob_pts = pts[token_indices]
            axis_val = atom.get("axis_ras")
            if axis_val is None:
                axis_val = axis_map.get(int(atom.get("parent_blob_id", atom_id)))
            axis = np.asarray(axis_val if axis_val is not None else [], dtype=float).reshape(-1)
            if axis.size != 3 or float(np.linalg.norm(axis)) <= 1e-6:
                if blob_pts.shape[0] >= 2:
                    _center, axis = self._fit_line_pca(blob_pts, seed_axis=None)
                else:
                    axis = np.asarray([0.0, 0.0, 1.0], dtype=float)
            axis = np.asarray(axis, dtype=float).reshape(3)
            axis_norm = float(np.linalg.norm(axis))
            axis = axis / axis_norm if axis_norm > 1e-6 else np.asarray([0.0, 0.0, 1.0], dtype=float)
            center_val = atom.get("center_ras")
            if center_val is None:
                center_val = np.mean(blob_pts, axis=0)
            center = np.asarray(center_val, dtype=float).reshape(3)
            start_val = atom.get("start_ras")
            if start_val is None:
                start_val = center
            end_val = atom.get("end_ras")
            if end_val is None:
                end_val = center
            start_ras = np.asarray(start_val, dtype=float).reshape(3)
            end_ras = np.asarray(end_val, dtype=float).reshape(3)
            span_mm = float(atom.get("span_mm") or np.linalg.norm(end_ras - start_ras))
            blob_nodes[int(atom_id)] = {
                "blob_id": int(atom_id),
                "token_indices": token_indices.astype(int).tolist(),
                "points_ras": np.asarray(blob_pts, dtype=float),
                "center_ras": np.asarray(center, dtype=float),
                "axis_ras": np.asarray(axis, dtype=float),
                "start_ras": np.asarray(start_ras, dtype=float),
                "end_ras": np.asarray(end_ras, dtype=float),
                "span_mm": float(span_mm),
                "token_count": int(token_indices.size),
                "elongation": float(atom.get("elongation", elongation_map.get(int(atom.get("parent_blob_id", atom_id)), 1.0))),
                "thickness_mm": float(atom.get("thickness_mm", 0.0)),
                "axis_reliable": bool(atom.get("axis_reliable", token_indices.size >= 2)),
                "parent_blob_id": int(atom.get("parent_blob_id", dict(parent_blob_id_map or {}).get(int(atom_id), int(atom_id)))),
                "kind": str(atom.get("kind") or "support_atom"),
                "source_blob_class": str(atom.get("source_blob_class") or ""),
                "direct_blob_line_atom": bool(atom.get("direct_blob_line_atom", False)),
                "occupancy": dict(atom.get("occupancy") or {}),
            }

        cos_thresh = float(np.cos(np.deg2rad(float(max(0.0, bridge_axis_angle_deg)))))
        pair_best = {}
        ordered_blob_ids = sorted(int(v) for v in blob_nodes.keys())
        for idx_a, blob_i in enumerate(ordered_blob_ids[:-1]):
            node_i = blob_nodes.get(int(blob_i))
            if node_i is None:
                continue
            if int(np.asarray(node_i.get("token_indices") or [], dtype=int).reshape(-1).size) == 0:
                continue
            for blob_j in ordered_blob_ids[(idx_a + 1) :]:
                node_j = blob_nodes.get(int(blob_j))
                if node_j is None:
                    continue
                if int(np.asarray(node_j.get("token_indices") or [], dtype=int).reshape(-1).size) == 0:
                    continue
                bridge = self._score_blob_node_bridge(
                    node_i=node_i,
                    node_j=node_j,
                    bridge_max_mm=float(bridge_max_mm),
                    neighbor_max_mm=float(neighbor_max_mm),
                    cos_thresh=float(cos_thresh),
                )
                if bridge is None:
                    continue
                pair_key = (int(blob_i), int(blob_j))
                prev = pair_best.get(pair_key)
                if prev is None or float(bridge.get("score", -1e9)) > float(prev.get("score", -1e9)):
                    pair_best[pair_key] = dict(bridge)
        for blob_i in ordered_blob_ids:
            node_i = blob_nodes.get(int(blob_i))
            if node_i is None or str(node_i.get("source_blob_class") or "") != "complex_blob" or str(node_i.get("kind") or "") != "line":
                continue
            explicit_candidates = []
            for blob_j in ordered_blob_ids:
                if int(blob_j) == int(blob_i):
                    continue
                node_j = blob_nodes.get(int(blob_j))
                if node_j is None:
                    continue
                if int(node_i.get("parent_blob_id", -1)) == int(node_j.get("parent_blob_id", -2)):
                    continue
                bridge = self._score_complex_port_bridge(
                    complex_node=node_i,
                    other_node=node_j,
                    bridge_max_mm=float(max(24.0, bridge_max_mm + 12.0)),
                    cos_thresh=float(max(0.50, cos_thresh - 0.10)),
                )
                if bridge is None:
                    continue
                explicit_candidates.append((float(bridge.get("score", -1e9)), int(blob_j), dict(bridge)))
            explicit_candidates.sort(key=lambda item: float(item[0]), reverse=True)
            for _score, blob_j, bridge in explicit_candidates[:4]:
                pair_key = tuple(sorted((int(blob_i), int(blob_j))))
                prev = pair_best.get(pair_key)
                if prev is None or float(bridge.get("score", -1e9)) > float(prev.get("score", -1e9)):
                    pair_best[pair_key] = dict(bridge)

        adj = {int(blob_id): [] for blob_id in blob_nodes.keys()}
        seed_edges = []
        local_seed_edges = []
        bridge_seed_edges = []
        for (blob_a, blob_b), info in pair_best.items():
            dir_ab = np.asarray(info.get("direction_ras"), dtype=float).reshape(3)
            dir_norm = float(np.linalg.norm(dir_ab))
            if dir_norm <= 1e-6:
                continue
            dir_ab = dir_ab / dir_norm
            adj[int(blob_a)].append(
                {
                    "neighbor_blob_id": int(blob_b),
                    "score": float(info.get("score", 0.0)),
                    "bridge_dist_mm": float(info.get("dist_mm", 0.0)),
                    "direction_ras": np.asarray(dir_ab, dtype=float),
                    "seed_i": int(info.get("seed_i", -1)),
                    "seed_j": int(info.get("seed_j", -1)),
                    "axis_agreement": float(info.get("axis_agreement", 0.0)),
                    "source_blob_class": str(atom.get("source_blob_class") or ""),
                    "direct_blob_line_atom": bool(atom.get("direct_blob_line_atom", False)),
                }
            )
            adj[int(blob_b)].append(
                {
                    "neighbor_blob_id": int(blob_a),
                    "score": float(info.get("score", 0.0)),
                    "bridge_dist_mm": float(info.get("dist_mm", 0.0)),
                    "direction_ras": np.asarray(-dir_ab, dtype=float),
                    "seed_i": int(info.get("seed_j", -1)),
                    "seed_j": int(info.get("seed_i", -1)),
                    "axis_agreement": float(info.get("axis_agreement", 0.0)),
                }
            )
            seed_edges.append((float(info.get("score", 0.0)), int(blob_a), int(blob_b)))
            if bool(info.get("is_local_seed")):
                local_seed_edges.append((float(info.get("score", 0.0)), int(blob_a), int(blob_b)))
            if int(blob_nodes.get(int(blob_a), {}).get("parent_blob_id", -1)) != int(blob_nodes.get(int(blob_b), {}).get("parent_blob_id", -1)):
                bridge_seed_edges.append((float(info.get("score", 0.0)), int(blob_a), int(blob_b)))
        seed_edges.sort(key=lambda item: float(item[0]), reverse=True)
        local_seed_edges.sort(key=lambda item: float(item[0]), reverse=True)
        bridge_seed_edges.sort(key=lambda item: float(item[0]), reverse=True)
        return {
            "blob_nodes": blob_nodes,
            "adj": adj,
            "seed_edges": seed_edges,
            "local_seed_edges": local_seed_edges,
            "bridge_seed_edges": bridge_seed_edges,
        }

    def _grow_blob_chain_candidate(
        self,
        pts_ras,
        blob_ids,
        blob_graph,
        seed_blob_i,
        seed_blob_j,
        grow_turn_angle_deg=30.0,
        inlier_radius_mm=2.0,
        axial_bin_mm=2.5,
        min_span_mm=16.0,
        min_inlier_count=5,
        parent_blob_id_map=None,
    ):
        pts = np.asarray(pts_ras, dtype=float).reshape(-1, 3)
        blob_ids = np.asarray(blob_ids, dtype=np.int32).reshape(-1)
        blob_nodes = dict(blob_graph.get("blob_nodes") or {})
        adj = dict(blob_graph.get("adj") or {})
        node_i = blob_nodes.get(int(seed_blob_i))
        node_j = blob_nodes.get(int(seed_blob_j))
        if node_i is None or node_j is None:
            return None

        chain = [int(seed_blob_i), int(seed_blob_j)]
        chain_set = {int(seed_blob_i), int(seed_blob_j)}
        self._grow_blob_chain_side(
            blob_graph=blob_graph,
            chain=chain,
            chain_set=chain_set,
            grow_forward=True,
            grow_turn_angle_deg=float(grow_turn_angle_deg),
            inlier_radius_mm=float(inlier_radius_mm),
        )
        self._grow_blob_chain_side(
            blob_graph=blob_graph,
            chain=chain,
            chain_set=chain_set,
            grow_forward=False,
            grow_turn_angle_deg=float(grow_turn_angle_deg),
            inlier_radius_mm=float(inlier_radius_mm),
        )

        if len(chain) < 2:
            return None
        token_indices = []
        for blob_id in chain:
            token_indices.extend(int(v) for v in list(blob_nodes.get(int(blob_id), {}).get("token_indices") or []))
        if len(token_indices) < int(max(3, min_inlier_count)):
            return None
        token_indices = np.asarray(sorted(set(token_indices)), dtype=int)
        chain_pts = pts[token_indices]
        seed_axis = np.asarray(node_j.get("center_ras"), dtype=float) - np.asarray(node_i.get("center_ras"), dtype=float)
        if float(np.linalg.norm(seed_axis)) <= 1e-6:
            seed_axis = np.asarray(node_j.get("axis_ras"), dtype=float)
        fit_center, fit_axis = self._fit_line_pca(chain_pts, seed_axis=seed_axis)
        radial_fit = self._radial_distance_to_line(chain_pts, fit_center, fit_axis)
        rms = float(np.sqrt(np.mean(radial_fit ** 2)))
        if rms > float(max(0.15, 1.35 * inlier_radius_mm)):
            return None
        proj = ((chain_pts - fit_center.reshape(1, 3)) @ fit_axis.reshape(3, 1)).reshape(-1)
        pmin = float(np.min(proj))
        pmax = float(np.max(proj))
        span = float(pmax - pmin)
        if span < float(max(1.0, min_span_mm)):
            return None
        occ = self._axial_occupancy_metrics(proj, bin_mm=float(max(0.5, axial_bin_mm)))
        if occ["occupied_bins"] < 2:
            return None
        if not self._occupancy_supports_line(
            occ,
            min_coverage=0.40,
            max_gap_bins=2,
            max_group_runs=3,
            min_group_occupied_bins=6,
            min_group_run_bins=2,
        ):
            return None
        distinct_blobs = int(len(chain_set))
        if distinct_blobs < 2:
            return None
        parent_blob_ids = sorted(
            set(int(dict(parent_blob_id_map or {}).get(int(blob_id), int(blob_id))) for blob_id in chain)
        )
        distinct_parent_blobs = int(len(parent_blob_ids))
        mean_turn_dot = 1.0
        min_turn_dot = 1.0
        if len(chain) >= 3:
            centroids = np.asarray([blob_nodes[int(blob_id)].get("center_ras") for blob_id in chain], dtype=float).reshape(-1, 3)
            turn_metrics = self._point_chain_turn_metrics(centroids)
            mean_turn_dot = float(turn_metrics.get("mean_dot", 1.0))
            min_turn_dot = float(turn_metrics.get("min_dot", 1.0))
            if min_turn_dot < float(np.cos(np.deg2rad(45.0))):
                return None
        score = (
            1.0 * span
            + 0.8 * float(chain_pts.shape[0])
            + 7.0 * float(distinct_blobs)
            + 2.5 * float(distinct_parent_blobs)
            + 9.0 * float(occ["coverage"])
            - 2.5 * float(rms)
            - 1.5 * float(occ["max_gap_bins"])
            + 5.0 * float(mean_turn_dot)
        )
        if distinct_blobs == 2:
            score -= 4.0
        start_ras = fit_center + fit_axis * pmin
        end_ras = fit_center + fit_axis * pmax
        midpoint_ras = 0.5 * (start_ras + end_ras)
        return {
            "start_ras": [float(v) for v in start_ras],
            "end_ras": [float(v) for v in end_ras],
            "midpoint_ras": [float(v) for v in midpoint_ras],
            "axis_ras": [float(v) for v in fit_axis],
            "center_ras": [float(v) for v in fit_center],
            "span_mm": float(span),
            "inlier_count": int(chain_pts.shape[0]),
            "distinct_blob_count": int(distinct_blobs),
            "distinct_parent_blob_count": int(distinct_parent_blobs),
            "rms_mm": float(rms),
            "coverage": float(occ["coverage"]),
            "occupied_bins": int(occ["occupied_bins"]),
            "total_bins": int(occ["total_bins"]),
            "max_gap_bins": int(occ["max_gap_bins"]),
            "blob_id_list": [int(v) for v in chain],
            "atom_id_list": [int(v) for v in chain],
            "parent_blob_id_list": [int(v) for v in parent_blob_ids],
            "token_indices": [int(v) for v in token_indices.astype(int).tolist()],
            "score": float(score),
        }

    def _grow_blob_chain_side(
        self,
        blob_graph,
        chain,
        chain_set,
        grow_forward=True,
        grow_turn_angle_deg=30.0,
        inlier_radius_mm=2.0,
    ):
        blob_nodes = dict(blob_graph.get("blob_nodes") or {})
        adj = dict(blob_graph.get("adj") or {})
        cos_turn = float(np.cos(np.deg2rad(float(max(0.0, grow_turn_angle_deg)))))
        while True:
            if len(chain) < 2:
                return
            end_blob = int(chain[-1] if grow_forward else chain[0])
            prev_blob = int(chain[-2] if grow_forward else chain[1])
            end_node = blob_nodes.get(end_blob)
            prev_node = blob_nodes.get(prev_blob)
            if end_node is None or prev_node is None:
                return
            local_axis = np.asarray(end_node.get("center_ras"), dtype=float) - np.asarray(prev_node.get("center_ras"), dtype=float)
            local_norm = float(np.linalg.norm(local_axis))
            if local_norm <= 1e-6:
                return
            local_axis = local_axis / local_norm

            chain_token_indices = []
            for blob_id in chain:
                chain_token_indices.extend(int(v) for v in list(blob_nodes.get(int(blob_id), {}).get("token_indices") or []))
            if not chain_token_indices:
                return
            chain_token_indices = np.asarray(sorted(set(chain_token_indices)), dtype=int)
            chain_pts = np.asarray(
                np.vstack([blob_nodes[int(blob_id)].get("points_ras") for blob_id in chain if blob_nodes.get(int(blob_id)) is not None]),
                dtype=float,
            ).reshape(-1, 3)
            fit_center, fit_axis = self._fit_line_pca(
                chain_pts,
                seed_axis=(np.asarray(blob_nodes[int(chain[-1])].get("center_ras"), dtype=float) - np.asarray(blob_nodes[int(chain[0])].get("center_ras"), dtype=float)),
            )
            chain_proj = ((chain_pts - fit_center.reshape(1, 3)) @ fit_axis.reshape(3, 1)).reshape(-1)
            pmin = float(np.min(chain_proj))
            pmax = float(np.max(chain_proj))
            best = None
            best_score = None
            for edge in list(adj.get(end_blob) or []):
                nb = int(edge.get("neighbor_blob_id", -1))
                if nb in chain_set:
                    continue
                nb_node = blob_nodes.get(nb)
                if nb_node is None:
                    continue
                step_dir_val = edge.get("direction_ras")
                if step_dir_val is None:
                    step_dir_val = [0.0, 0.0, 1.0]
                step_dir = np.asarray(step_dir_val, dtype=float).reshape(3)
                step_norm = float(np.linalg.norm(step_dir))
                if step_norm <= 1e-6:
                    continue
                step_dir = step_dir / step_norm
                if float(np.dot(step_dir, local_axis)) < cos_turn:
                    continue
                nb_center = np.asarray(nb_node.get("center_ras"), dtype=float).reshape(3)
                nb_proj = float(np.dot(nb_center - fit_center.reshape(3), fit_axis.reshape(3)))
                if grow_forward and nb_proj < pmax - 1.0:
                    continue
                if (not grow_forward) and nb_proj > pmin + 1.0:
                    continue
                nb_pts = np.asarray(nb_node.get("points_ras"), dtype=float).reshape(-1, 3)
                radial = self._radial_distance_to_line(nb_pts, fit_center, fit_axis)
                radial_med = float(np.median(radial)) if radial.size else float("inf")
                radial_p75 = float(np.percentile(radial, 75.0)) if radial.size else float("inf")
                if radial_med > float(max(0.25, 1.75 * inlier_radius_mm)):
                    continue
                if radial_p75 > float(max(0.5, 2.10 * inlier_radius_mm)):
                    continue
                candidate_score = (
                    2.0 * float(np.dot(step_dir, local_axis))
                    + 1.2 * float(edge.get("axis_agreement", 0.0))
                    + 0.4 * float(min(edge.get("bridge_dist_mm", 0.0), 10.0))
                    + 0.25 * float(min(nb_node.get("token_count", 0), 4))
                    + (0.8 if int(nb_node.get("parent_blob_id", -1)) == int(end_node.get("parent_blob_id", -2)) else 0.0)
                    - 0.5 * float(radial_med)
                    - 0.15 * float(len(list(adj.get(nb) or [])))
                )
                if best_score is None or candidate_score > best_score:
                    best = nb
                    best_score = float(candidate_score)
            if best is None:
                return
            if grow_forward:
                chain.append(int(best))
            else:
                chain.insert(0, int(best))
            chain_set.add(int(best))

    def _complete_token_line_candidate(
        self,
        pts_ras,
        blob_ids,
        candidate,
        token_atom_ids=None,
        inlier_radius_mm=2.0,
        axial_bin_mm=2.5,
        min_span_mm=16.0,
        min_inlier_count=5,
        elongation_map=None,
        parent_blob_id_map=None,
    ):
        pts = np.asarray(pts_ras, dtype=float).reshape(-1, 3)
        blob_ids = np.asarray(blob_ids, dtype=np.int32).reshape(-1)
        if token_atom_ids is None:
            token_atom_ids = np.zeros((pts.shape[0],), dtype=np.int32)
        token_atom_ids = np.asarray(token_atom_ids, dtype=np.int32).reshape(-1)
        candidate_token_count = int(len(list(candidate.get("token_indices") or [])))
        candidate_source_blob_class = str(candidate.get("seed_source_blob_class") or candidate.get("source_blob_class") or "")
        candidate_family = str(candidate.get("proposal_family") or "")
        seeded_from_complex_blob = bool(candidate_source_blob_class == "complex_blob")
        radial_inlier_scale = 1.35 if seeded_from_complex_blob else 1.15
        endpoint_radius_scale = 2.20 if seeded_from_complex_blob else 1.60
        max_extension_mm = 36.0 if seeded_from_complex_blob else None
        center = np.asarray(candidate.get("center_ras") or candidate.get("midpoint_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
        axis = np.asarray(candidate.get("axis_ras") or [0.0, 0.0, 1.0], dtype=float).reshape(3)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-6 or pts.shape[0] == 0:
            return candidate
        axis = axis / axis_norm
        if candidate_family == "blob_axis" and candidate_token_count > 0:
            inlier_mask = np.zeros((pts.shape[0],), dtype=bool)
            for token_idx in [int(v) for v in list(candidate.get("token_indices") or [])]:
                if 0 <= int(token_idx) < int(pts.shape[0]):
                    inlier_mask[int(token_idx)] = True
            if int(np.count_nonzero(inlier_mask)) < int(max(2, min_inlier_count - 1)):
                return candidate
        else:
            radial = self._radial_distance_to_line(pts, center, axis)
            inlier_mask = radial <= float(max(0.1, radial_inlier_scale * inlier_radius_mm))
            if int(np.count_nonzero(inlier_mask)) < int(max(3, min_inlier_count)):
                return candidate
        inlier_pts = pts[inlier_mask]
        inlier_blob_ids = blob_ids[inlier_mask]
        fit_center, fit_axis = self._fit_line_pca(inlier_pts, seed_axis=axis)
        fit_radial = self._radial_distance_to_line(inlier_pts, fit_center, fit_axis)
        rms = float(np.sqrt(np.mean(fit_radial ** 2)))
        if rms > float(max(0.15, 1.25 * inlier_radius_mm)):
            return candidate
        if candidate_family != "blob_axis":
            expanded_mask = self._extend_token_line_inliers(
                pts_ras=pts,
                center_ras=fit_center,
                axis_ras=fit_axis,
                base_mask=inlier_mask,
                inlier_radius_mm=float(inlier_radius_mm),
                axial_bin_mm=float(axial_bin_mm),
                max_extension_mm=max_extension_mm,
                endpoint_radius_scale=float(endpoint_radius_scale),
            )
            if int(np.count_nonzero(expanded_mask)) > int(np.count_nonzero(inlier_mask)):
                inlier_mask = np.asarray(expanded_mask, dtype=bool).reshape(-1)
                inlier_pts = pts[inlier_mask]
                inlier_blob_ids = blob_ids[inlier_mask]
                fit_center, fit_axis = self._fit_line_pca(inlier_pts, seed_axis=fit_axis)
                fit_radial = self._radial_distance_to_line(inlier_pts, fit_center, fit_axis)
                rms = float(np.sqrt(np.mean(fit_radial ** 2)))
                if rms > float(max(0.15, 1.25 * inlier_radius_mm)):
                    return candidate
        proj = ((inlier_pts - fit_center.reshape(1, 3)) @ fit_axis.reshape(3, 1)).reshape(-1)
        pmin = float(np.min(proj))
        pmax = float(np.max(proj))
        span = float(pmax - pmin)
        if span < float(max(1.0, min_span_mm)):
            return candidate
        occ = self._axial_occupancy_metrics(proj, bin_mm=float(max(0.5, axial_bin_mm)))
        distinct_blobs = int(np.unique(inlier_blob_ids).size)
        parent_blob_ids = sorted(
            set(int(dict(parent_blob_id_map or {}).get(int(blob_id), int(blob_id))) for blob_id in np.unique(inlier_blob_ids).astype(int).tolist())
        )
        distinct_parent_blobs = int(len(parent_blob_ids))
        support_gain = int(inlier_pts.shape[0]) - int(max(0, candidate_token_count))
        standard_occupancy = self._occupancy_supports_line(
            occ,
            min_coverage=0.40,
            max_gap_bins=2,
            max_group_runs=3,
            min_group_occupied_bins=6,
            min_group_run_bins=2,
        )
        relaxed_grouped_occupancy = bool(
            not standard_occupancy
            and support_gain >= 4
            and float(rms) <= float(max(0.15, 1.25 * inlier_radius_mm))
            and float(occ.get("coverage", 0.0)) >= 0.50
            and int(occ.get("occupied_bins", 0)) >= int(max(12, min_inlier_count + 6))
            and int(occ.get("max_gap_bins", 0)) <= 4
            and int(occ.get("max_run_bins", 0)) >= 3
            and int(distinct_blobs) >= 6
        )
        if occ["occupied_bins"] < 2 or not (standard_occupancy or relaxed_grouped_occupancy):
            return candidate
        if distinct_blobs < 2:
            dominant_blob = int(np.bincount(inlier_blob_ids.astype(np.int32)).argmax()) if inlier_blob_ids.size else -1
            dominant_elong = float(dict(elongation_map or {}).get(dominant_blob, 1.0))
            if dominant_elong < 4.0 and span < float(max(22.0, min_span_mm + 4.0)):
                return candidate
        score = (
            0.9 * span
            + 0.8 * float(inlier_pts.shape[0])
            + 5.5 * float(distinct_blobs)
            + 1.5 * float(distinct_parent_blobs)
            + 10.0 * float(occ["coverage"])
            - 2.5 * float(rms)
            - 2.0 * float(occ["max_gap_bins"])
        )
        if distinct_blobs == 1:
            score -= 14.0
        start_ras = fit_center + fit_axis * pmin
        end_ras = fit_center + fit_axis * pmax
        midpoint_ras = 0.5 * (start_ras + end_ras)
        completed = dict(candidate)
        completed.update(
            {
                "start_ras": [float(v) for v in start_ras],
                "end_ras": [float(v) for v in end_ras],
                "midpoint_ras": [float(v) for v in midpoint_ras],
                "axis_ras": [float(v) for v in fit_axis],
                "center_ras": [float(v) for v in fit_center],
                "span_mm": float(span),
                "inlier_count": int(inlier_pts.shape[0]),
                "distinct_blob_count": int(distinct_blobs),
                "distinct_parent_blob_count": int(distinct_parent_blobs),
                "rms_mm": float(rms),
                "coverage": float(occ["coverage"]),
                "occupied_bins": int(occ["occupied_bins"]),
                "total_bins": int(occ["total_bins"]),
                "max_gap_bins": int(occ["max_gap_bins"]),
                "blob_id_list": [int(v) for v in np.unique(inlier_blob_ids).astype(int).tolist()],
                "atom_id_list": [
                    int(v)
                    for v in np.unique(token_atom_ids[inlier_mask]).astype(int).tolist()
                    if int(v) > 0
                ],
                "parent_blob_id_list": [int(v) for v in parent_blob_ids],
                "token_indices": [int(v) for v in np.flatnonzero(inlier_mask).astype(int).tolist()],
                "score": float(score),
                "used_relaxed_grouped_occupancy": bool(relaxed_grouped_occupancy),
                "seed_source_blob_class": str(candidate_source_blob_class),
            }
        )
        return completed

    def _extend_token_line_inliers(
        self,
        pts_ras,
        center_ras,
        axis_ras,
        base_mask,
        inlier_radius_mm=2.0,
        axial_bin_mm=2.5,
        max_extension_mm=None,
        endpoint_radius_scale=1.60,
    ):
        pts = np.asarray(pts_ras, dtype=float).reshape(-1, 3)
        base = np.asarray(base_mask, dtype=bool).reshape(-1)
        if pts.shape[0] == 0 or base.shape[0] != pts.shape[0] or int(np.count_nonzero(base)) < 2:
            return base
        center = np.asarray(center_ras, dtype=float).reshape(3)
        axis = np.asarray(axis_ras, dtype=float).reshape(3)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-6:
            return base
        axis = axis / axis_norm
        vec = pts - center.reshape(1, 3)
        proj = (vec @ axis.reshape(3, 1)).reshape(-1)
        radial = np.linalg.norm(vec - proj.reshape(-1, 1) * axis.reshape(1, 3), axis=1)
        base_proj = proj[base]
        if base_proj.size == 0:
            return base
        endpoint_radius = float(max(0.1, float(endpoint_radius_scale) * float(inlier_radius_mm)))
        axial_bin = float(max(0.5, axial_bin_mm))
        max_extension = float(max(12.0, 8.0 * axial_bin) if max_extension_mm is None else max_extension_mm)
        context_mm = float(2.0 * axial_bin)
        beyond_eps = float(0.25 * axial_bin)
        expanded = np.asarray(base, dtype=bool).copy()
        for grow_min in (True, False):
            edge = float(np.min(base_proj) if grow_min else np.max(base_proj))
            if grow_min:
                window = np.logical_and(proj >= edge - max_extension, proj <= edge + context_mm)
                beyond = proj < edge - beyond_eps
            else:
                window = np.logical_and(proj >= edge - context_mm, proj <= edge + max_extension)
                beyond = proj > edge + beyond_eps
            side_mask = np.logical_and(window, radial <= endpoint_radius)
            if not np.any(np.logical_and(side_mask, beyond)):
                continue
            candidate_idx = np.flatnonzero(side_mask).astype(int)
            ext_dist = (edge - proj[candidate_idx]) if grow_min else (proj[candidate_idx] - edge)
            if candidate_idx.size == 0 or not np.any(ext_dist > beyond_eps):
                continue
            candidate_idx = candidate_idx[np.argsort(ext_dist)]
            ext_dist = ext_dist[np.argsort(ext_dist)]
            bin_ids = np.floor(np.maximum(ext_dist, 0.0) / axial_bin).astype(int)
            occupied_bins = sorted(set(int(v) for v in bin_ids.tolist()))
            if not occupied_bins:
                continue
            frontier = 0
            gap_run = 0
            occupied_lookup = set(occupied_bins)
            max_bin = int(max(occupied_bins))
            for bin_id in range(0, max_bin + 1):
                if bin_id in occupied_lookup:
                    frontier = int(bin_id)
                    gap_run = 0
                else:
                    gap_run += 1
                    if gap_run > 1:
                        break
            beyond_bins = [int(v) for v in occupied_bins if int(v) >= 1 and int(v) <= int(frontier)]
            if len(beyond_bins) < 2:
                continue
            keep_local = np.logical_or(ext_dist <= context_mm, bin_ids <= int(frontier))
            keep_idx = candidate_idx[keep_local]
            occ = self._axial_occupancy_metrics(proj[keep_idx], bin_mm=axial_bin)
            if int(occ.get("occupied_bins", 0)) < 3 or int(occ.get("max_gap_bins", 0)) > 2:
                continue
            expanded[keep_idx] = True
        return expanded

    def _build_blob_axis_hypotheses(
        self,
        pts_ras,
        blob_ids,
        token_atom_ids,
        support_atoms,
        elongation_map,
        axial_bin_mm=2.5,
        min_span_mm=16.0,
        min_inlier_count=5,
        parent_blob_id_map=None,
    ):
        pts = np.asarray(pts_ras, dtype=float).reshape(-1, 3)
        blob_ids = np.asarray(blob_ids, dtype=np.int32).reshape(-1)
        token_atom_ids = np.asarray(token_atom_ids, dtype=np.int32).reshape(-1)
        proposals = []
        for atom in list(support_atoms or []):
            atom = dict(atom or {})
            atom_id = int(atom.get("atom_id", 0))
            if atom_id <= 0 or str(atom.get("kind") or "") != "line":
                continue
            source_blob_class = str(atom.get("source_blob_class") or "")
            is_contact_chain_line = bool(source_blob_class == "contact_chain")
            sel = token_atom_ids == int(atom_id)
            required_token_count = int(3 if is_contact_chain_line else max(4, min_inlier_count))
            if int(np.count_nonzero(sel)) < int(required_token_count):
                continue
            parent_blob_id = int(atom.get("parent_blob_id", atom_id))
            elong = float(atom.get("elongation", elongation_map.get(parent_blob_id, 1.0)))
            if elong < float(2.0 if is_contact_chain_line else 4.0):
                continue
            atom_pts = pts[sel]
            axis = np.asarray(atom.get("axis_ras") or [0.0, 0.0, 1.0], dtype=float).reshape(3)
            center = np.asarray(atom.get("center_ras") or np.mean(atom_pts, axis=0), dtype=float).reshape(3)
            fit_center, fit_axis = self._fit_line_pca(atom_pts, seed_axis=axis)
            radial = self._radial_distance_to_line(atom_pts, fit_center, fit_axis)
            rms = float(np.sqrt(np.mean(radial ** 2)))
            if rms > 1.75:
                continue
            proj = ((atom_pts - fit_center.reshape(1, 3)) @ fit_axis.reshape(3, 1)).reshape(-1)
            pmin = float(np.min(proj))
            pmax = float(np.max(proj))
            span = float(pmax - pmin)
            occ = self._axial_occupancy_metrics(proj, bin_mm=float(max(0.5, axial_bin_mm)))
            if int(occ["occupied_bins"]) < int(2 if is_contact_chain_line else 3):
                continue
            if not self._occupancy_supports_line(
                occ,
                min_coverage=0.35 if is_contact_chain_line else 0.50,
                max_gap_bins=3 if is_contact_chain_line else 2,
                max_group_runs=3,
                min_group_occupied_bins=3 if is_contact_chain_line else 6,
                min_group_run_bins=2,
            ):
                continue
            score = (
                0.75 * span
                + 0.7 * float(atom_pts.shape[0])
                + 6.0 * float(occ["coverage"])
                + 1.5 * float(min(elong, 12.0))
                - 2.0 * float(rms)
                - 1.5 * float(occ["max_gap_bins"])
            )
            start_ras = fit_center + fit_axis * pmin
            end_ras = fit_center + fit_axis * pmax
            midpoint_ras = 0.5 * (start_ras + end_ras)
            token_indices = np.flatnonzero(sel).astype(int).tolist()
            proposals.append(
                self._complete_token_line_candidate(
                    pts_ras=pts,
                    blob_ids=blob_ids,
                    token_atom_ids=token_atom_ids,
                    candidate={
                    "start_ras": [float(v) for v in start_ras],
                    "end_ras": [float(v) for v in end_ras],
                    "midpoint_ras": [float(v) for v in midpoint_ras],
                    "axis_ras": [float(v) for v in fit_axis],
                    "center_ras": [float(v) for v in fit_center],
                    "span_mm": float(span),
                    "inlier_count": int(atom_pts.shape[0]),
                    "distinct_blob_count": 1,
                    "rms_mm": float(rms),
                    "coverage": float(occ["coverage"]),
                    "occupied_bins": int(occ["occupied_bins"]),
                    "total_bins": int(occ["total_bins"]),
                    "max_gap_bins": int(occ["max_gap_bins"]),
                    "blob_id_list": [int(parent_blob_id)],
                    "atom_id_list": [int(atom_id)],
                    "token_indices": [int(v) for v in token_indices],
                    "score": float(score),
                    "proposal_family": "blob_axis",
                    "seed_atom_id": int(atom_id),
                    "seed_source_blob_class": str(source_blob_class),
                    },
                    inlier_radius_mm=2.0,
                    axial_bin_mm=float(axial_bin_mm),
                    min_span_mm=0.0,
                    min_inlier_count=int(required_token_count),
                    elongation_map=elongation_map,
                    parent_blob_id_map=parent_blob_id_map,
                )
            )
        return proposals
