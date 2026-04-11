"""Contact-chain recovery helpers for deep-core support extraction."""

try:
    import numpy as np
except ImportError:
    np = None


class DeepCoreContactChainMixin:
    """Recover line-like atoms from sparse contact-only support."""

    def _build_contact_chain_rows_from_atoms(
        self,
        support_atoms,
        max_neighbor_count=4,
    ):
        contact_atoms = [
            dict(atom or {})
            for atom in list(support_atoms or [])
            if str(dict(atom or {}).get("kind") or "") == "contact"
            and int(dict(atom or {}).get("atom_id", -1)) > 0
        ]
        if len(contact_atoms) < 3:
            return {"rows": [], "debug_rows": [], "atoms": []}
        atom_map = {int(atom.get("atom_id")): dict(atom) for atom in contact_atoms}
        center_map = {
            int(atom_id): np.asarray(dict(atom or {}).get("center_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
            for atom_id, atom in atom_map.items()
        }
        atom_ids = sorted(int(v) for v in atom_map.keys())
        adj = {int(atom_id): [] for atom_id in atom_ids}
        edge_pairs = set()
        for atom_id in atom_ids:
            center_i = np.asarray(center_map.get(int(atom_id)), dtype=float).reshape(3)
            candidates = []
            for other_id in atom_ids:
                if int(other_id) == int(atom_id):
                    continue
                center_j = np.asarray(center_map.get(int(other_id)), dtype=float).reshape(3)
                delta = center_j - center_i
                dist_mm = float(np.linalg.norm(delta))
                if dist_mm <= 1e-6 or dist_mm > 24.0:
                    continue
                candidates.append((float(dist_mm), int(other_id), np.asarray(delta / dist_mm, dtype=float).reshape(3)))
            candidates.sort(key=lambda item: float(item[0]))
            for dist_mm, other_id, direction in candidates[: int(max(1, max_neighbor_count))]:
                adj[int(atom_id)].append(
                    {
                        "neighbor_id": int(other_id),
                        "dist_mm": float(dist_mm),
                        "direction_ras": np.asarray(direction, dtype=float).reshape(3),
                    }
                )
                edge_pairs.add(tuple(sorted((int(atom_id), int(other_id)))))

        seed_pairs = []
        debug_rows = []
        for atom_i, atom_j in sorted(list(edge_pairs)):
            seed_info = self._score_contact_seed_pair_debug(
                center_map=center_map,
                atom_ids=atom_ids,
                atom_i=int(atom_i),
                atom_j=int(atom_j),
            )
            if not bool(dict(seed_info or {}).get("accepted", False)):
                debug_rows.append(
                    {
                        "seed_atom_i": int(atom_i),
                        "seed_atom_j": int(atom_j),
                        "stage": "seed",
                        "status": "rejected",
                        "reason": str(dict(seed_info or {}).get("reason") or "insufficient_collinear_support"),
                        "seed_score": "",
                        "dist_mm": dict(seed_info or {}).get("dist_mm", ""),
                        "extra_support": dict(seed_info or {}).get("extra_support", ""),
                        "support_score": dict(seed_info or {}).get("support_score", ""),
                        "chain_len": 0,
                        "chain_atom_ids": "",
                        "chain_id": "",
                    }
                )
                continue
            debug_rows.append(
                {
                    "seed_atom_i": int(atom_i),
                    "seed_atom_j": int(atom_j),
                    "stage": "seed",
                    "status": "accepted",
                    "reason": "",
                    "seed_score": float(seed_info.get("score", 0.0)),
                    "dist_mm": dict(seed_info or {}).get("dist_mm", ""),
                    "extra_support": dict(seed_info or {}).get("extra_support", ""),
                    "support_score": dict(seed_info or {}).get("support_score", ""),
                    "chain_len": 0,
                    "chain_atom_ids": "",
                    "chain_id": "",
                }
            )
            seed_pairs.append((float(seed_info.get("score", 0.0)), int(atom_i), int(atom_j)))
        seed_pairs.sort(key=lambda item: (-float(item[0]), int(min(item[1], item[2])), int(max(item[1], item[2]))))
        candidates = []
        for _seed_score, atom_i, atom_j in seed_pairs:
            grow_info = self._grow_contact_chain_from_pair(
                center_map=center_map,
                adj=adj,
                start_atom_i=int(atom_i),
                start_atom_j=int(atom_j),
                return_debug=True,
            )
            chain_ids = [int(v) for v in list(dict(grow_info or {}).get("atom_ids") or []) if int(v) > 0]
            for grow_row in list(dict(grow_info or {}).get("debug_rows") or []):
                debug_rows.append(
                    {
                        "seed_atom_i": int(atom_i),
                        "seed_atom_j": int(atom_j),
                        "stage": "grow",
                        "status": str(grow_row.get("status") or "info"),
                        "reason": str(grow_row.get("reason") or ""),
                        "seed_score": float(_seed_score),
                        "chain_len": int(len(chain_ids)),
                        "chain_atom_ids": ",".join(str(int(v)) for v in chain_ids),
                        "chain_id": "",
                        "grow_direction": grow_row.get("grow_direction", ""),
                        "stop_atom_id": grow_row.get("stop_atom_id", ""),
                        "prev_atom_id": grow_row.get("prev_atom_id", ""),
                        "rejection_histogram": grow_row.get("rejection_histogram", ""),
                    }
                )
            split_info = self._split_contact_chain_on_large_gaps(
                center_map=center_map,
                atom_ids=chain_ids,
                return_debug=True,
            )
            chain_segments = list(dict(split_info or {}).get("segments") or [])
            for split_row in list(dict(split_info or {}).get("debug_rows") or []):
                debug_rows.append(
                    {
                        "seed_atom_i": int(atom_i),
                        "seed_atom_j": int(atom_j),
                        "stage": "split",
                        "status": str(split_row.get("status") or "info"),
                        "reason": str(split_row.get("reason") or ""),
                        "seed_score": float(_seed_score),
                        "chain_len": int(len(chain_ids)),
                        "chain_atom_ids": ",".join(str(int(v)) for v in chain_ids),
                        "chain_id": "",
                        "break_index": split_row.get("break_index", ""),
                        "break_gap_mm": split_row.get("break_gap_mm", ""),
                        "segment_index": split_row.get("segment_index", ""),
                        "segment_atom_ids": split_row.get("segment_atom_ids", ""),
                    }
                )
            if not chain_segments:
                debug_rows.append(
                    {
                        "seed_atom_i": int(atom_i),
                        "seed_atom_j": int(atom_j),
                        "stage": "split",
                        "status": "rejected",
                        "reason": "no_valid_segments",
                        "seed_score": float(_seed_score),
                        "chain_len": int(len(list(chain_ids or []))),
                        "chain_atom_ids": ",".join(str(int(v)) for v in list(chain_ids or [])),
                        "chain_id": "",
                    }
                )
                continue
            for segment_idx, segment_ids in enumerate(list(chain_segments or []), start=1):
                chain_info = self._score_contact_chain_debug(
                    center_map=center_map,
                    atom_ids=segment_ids,
                )
                if not bool(dict(chain_info or {}).get("accepted", False)):
                    chain_info = dict(chain_info or {})
                    debug_rows.append(
                        {
                            "seed_atom_i": int(atom_i),
                            "seed_atom_j": int(atom_j),
                            "stage": "score",
                            "status": "rejected",
                            "reason": str(chain_info.get("reason") or "chain_score_gate"),
                            "seed_score": float(_seed_score),
                            "chain_len": int(len(list(segment_ids or []))),
                            "chain_atom_ids": ",".join(str(int(v)) for v in list(chain_info.get("atom_ids") or segment_ids or [])),
                            "chain_id": "",
                            "span_mm": chain_info.get("span_mm", ""),
                            "rms_mm": chain_info.get("rms_mm", ""),
                            "rms_limit_mm": chain_info.get("rms_limit_mm", ""),
                            "median_gap_mm": chain_info.get("median_gap_mm", ""),
                            "max_gap_mm": chain_info.get("max_gap_mm", ""),
                            "spacing_cv": chain_info.get("spacing_cv", ""),
                            "mean_dot": chain_info.get("mean_dot", ""),
                            "min_dot": chain_info.get("min_dot", ""),
                            "segment_index": int(segment_idx),
                        }
                    )
                    continue
                chain_info = dict(chain_info)
                chain_info["seed_atom_i"] = int(atom_i)
                chain_info["seed_atom_j"] = int(atom_j)
                chain_info["seed_score"] = float(_seed_score)
                chain_info["segment_index"] = int(segment_idx)
                candidates.append(chain_info)
        candidates.sort(
            key=lambda item: (
                -float(item.get("score", 0.0)),
                -int(len(list(item.get("atom_ids") or []))),
                -float(item.get("span_mm", 0.0)),
            )
        )
        accepted = []
        used_atom_ids = set()
        for cand in candidates:
            cand_atoms = [int(v) for v in list(cand.get("atom_ids") or []) if int(v) > 0]
            if len(cand_atoms) < 3:
                continue
            shared_used = int(len(set(cand_atoms) & set(used_atom_ids)))
            if shared_used > 0 and float(shared_used / max(1, len(cand_atoms))) >= 0.35:
                debug_rows.append(
                    {
                        "seed_atom_i": int(cand.get("seed_atom_i", -1)),
                        "seed_atom_j": int(cand.get("seed_atom_j", -1)),
                        "stage": "selection",
                        "status": "rejected",
                        "reason": "shared_used",
                        "seed_score": float(cand.get("seed_score", 0.0)),
                        "chain_len": int(len(cand_atoms)),
                        "chain_atom_ids": ",".join(str(int(v)) for v in cand_atoms),
                        "chain_id": "",
                        "span_mm": cand.get("span_mm", ""),
                        "rms_mm": cand.get("rms_mm", ""),
                        "rms_limit_mm": cand.get("rms_limit_mm", ""),
                        "median_gap_mm": cand.get("median_gap_mm", ""),
                        "max_gap_mm": cand.get("max_gap_mm", ""),
                        "spacing_cv": cand.get("spacing_cv", ""),
                        "mean_dot": cand.get("mean_dot", ""),
                        "min_dot": cand.get("min_dot", ""),
                    }
                )
                continue
            duplicate = False
            for prev in accepted:
                prev_atoms = set(int(v) for v in list(prev.get("atom_ids") or []) if int(v) > 0)
                overlap = int(len(prev_atoms & set(cand_atoms)))
                if float(overlap / max(1, min(len(prev_atoms), len(cand_atoms)))) >= 0.60:
                    duplicate = True
                    break
            if duplicate:
                debug_rows.append(
                    {
                        "seed_atom_i": int(cand.get("seed_atom_i", -1)),
                        "seed_atom_j": int(cand.get("seed_atom_j", -1)),
                        "stage": "selection",
                        "status": "rejected",
                        "reason": "duplicate_overlap",
                        "seed_score": float(cand.get("seed_score", 0.0)),
                        "chain_len": int(len(cand_atoms)),
                        "chain_atom_ids": ",".join(str(int(v)) for v in cand_atoms),
                        "chain_id": "",
                        "span_mm": cand.get("span_mm", ""),
                        "rms_mm": cand.get("rms_mm", ""),
                        "rms_limit_mm": cand.get("rms_limit_mm", ""),
                        "median_gap_mm": cand.get("median_gap_mm", ""),
                        "max_gap_mm": cand.get("max_gap_mm", ""),
                        "spacing_cv": cand.get("spacing_cv", ""),
                        "mean_dot": cand.get("mean_dot", ""),
                        "min_dot": cand.get("min_dot", ""),
                    }
                )
                continue
            accepted.append(dict(cand))
            used_atom_ids.update(int(v) for v in cand_atoms)
            debug_rows.append(
                {
                    "seed_atom_i": int(cand.get("seed_atom_i", -1)),
                    "seed_atom_j": int(cand.get("seed_atom_j", -1)),
                    "stage": "selection",
                    "status": "accepted",
                    "reason": "",
                    "seed_score": float(cand.get("seed_score", 0.0)),
                    "chain_len": int(len(cand_atoms)),
                    "chain_atom_ids": ",".join(str(int(v)) for v in cand_atoms),
                    "chain_id": int(len(accepted)),
                    "span_mm": cand.get("span_mm", ""),
                    "rms_mm": cand.get("rms_mm", ""),
                    "rms_limit_mm": cand.get("rms_limit_mm", ""),
                    "median_gap_mm": cand.get("median_gap_mm", ""),
                    "max_gap_mm": cand.get("max_gap_mm", ""),
                    "spacing_cv": cand.get("spacing_cv", ""),
                    "mean_dot": cand.get("mean_dot", ""),
                    "min_dot": cand.get("min_dot", ""),
                }
            )

        rows = []
        line_atoms = []
        for chain_id, cand in enumerate(list(accepted or []), start=1):
            chain_atom_ids = [int(v) for v in list(cand.get("atom_ids") or []) if int(v) > 0]
            for chain_order, atom_id in enumerate(chain_atom_ids, start=1):
                atom = dict(atom_map.get(int(atom_id)) or {})
                center = np.asarray(atom.get("center_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
                node_role = "core"
                if int(chain_order) == 1 or int(chain_order) == int(len(chain_atom_ids)):
                    node_role = "edge"
                rows.append(
                    {
                        "chain_id": int(chain_id),
                        "atom_id": int(atom_id),
                        "parent_blob_id": int(atom.get("parent_blob_id", -1)),
                        "node_role": str(node_role),
                        "chain_order": int(chain_order),
                        "support_x_ras": float(center[0]),
                        "support_y_ras": float(center[1]),
                        "support_z_ras": float(center[2]),
                    }
                )
            centers = np.asarray(
                [
                    np.asarray(dict(atom_map.get(int(atom_id)) or {}).get("center_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
                    for atom_id in chain_atom_ids
                ],
                dtype=float,
            ).reshape(-1, 3)
            if centers.shape[0] >= 3:
                fit_center, fit_axis = self._fit_line_pca(centers, seed_axis=centers[-1] - centers[0])
                line_atoms.append(
                    {
                        "atom_ids": [int(v) for v in chain_atom_ids],
                        "center_ras": [float(v) for v in np.asarray(fit_center, dtype=float).reshape(3)],
                        "axis_ras": [float(v) for v in np.asarray(fit_axis, dtype=float).reshape(3)],
                        "score": float(cand.get("score", 0.0)),
                    }
                )
        return {"rows": rows, "debug_rows": debug_rows, "atoms": line_atoms}

    def _score_contact_seed_pair_debug(
        self,
        center_map,
        atom_ids,
        atom_i,
        atom_j,
    ):
        center_i = np.asarray(center_map.get(int(atom_i)), dtype=float).reshape(3)
        center_j = np.asarray(center_map.get(int(atom_j)), dtype=float).reshape(3)
        delta = center_j - center_i
        dist_mm = float(np.linalg.norm(delta))
        if dist_mm <= 1e-6 or dist_mm > 18.0:
            return {
                "accepted": False,
                "reason": "seed_distance_out_of_range",
                "dist_mm": float(dist_mm),
                "extra_support": 0,
                "support_score": 0.0,
            }
        axis = delta / dist_mm
        midpoint = 0.5 * (center_i + center_j)
        proj_i = float(np.dot(center_i - midpoint, axis))
        proj_j = float(np.dot(center_j - midpoint, axis))
        lo = min(proj_i, proj_j) - 16.0
        hi = max(proj_i, proj_j) + 16.0
        extra_support = 0
        support_score = 0.0
        for other_id in list(atom_ids or []):
            other_id = int(other_id)
            if other_id in (int(atom_i), int(atom_j)):
                continue
            center_k = np.asarray(center_map.get(int(other_id)), dtype=float).reshape(3)
            vec = center_k - midpoint
            proj = float(np.dot(vec, axis))
            if proj < lo or proj > hi:
                continue
            radial = float(self._radial_distance_to_line(center_k.reshape(1, 3), midpoint, axis)[0])
            if radial > 3.5:
                continue
            extra_support += 1
            support_score += float(max(0.0, 1.0 - min(1.0, radial / 3.5)))
        if extra_support < 1:
            return {
                "accepted": False,
                "reason": "insufficient_collinear_support",
                "dist_mm": float(dist_mm),
                "extra_support": int(extra_support),
                "support_score": float(support_score),
            }
        score = 4.0 * float(extra_support) + 1.5 * float(support_score) - 0.15 * float(dist_mm)
        return {
            "accepted": True,
            "reason": "",
            "score": float(score),
            "dist_mm": float(dist_mm),
            "extra_support": int(extra_support),
            "support_score": float(support_score),
        }

    def _grow_contact_chain_from_pair(
        self,
        center_map,
        adj,
        start_atom_i,
        start_atom_j,
        return_debug=False,
    ):
        chain = [int(start_atom_i), int(start_atom_j)]
        chain_set = {int(start_atom_i), int(start_atom_j)}
        cos_forward = float(np.cos(np.deg2rad(60.0)))
        cos_recent = float(np.cos(np.deg2rad(70.0)))
        max_step_mm = 10.0
        debug_rows = []
        for grow_forward in (True, False):
            while True:
                current_id = int(chain[-1] if grow_forward else chain[0])
                prev_id = int(chain[-2] if grow_forward else chain[1])
                current_center = np.asarray(center_map.get(int(current_id)), dtype=float).reshape(3)
                prev_center = np.asarray(center_map.get(int(prev_id)), dtype=float).reshape(3)
                travel = current_center - prev_center
                travel_norm = float(np.linalg.norm(travel))
                if travel_norm <= 1e-6:
                    break
                travel = travel / travel_norm
                chain_centers = np.asarray(
                    [np.asarray(center_map.get(int(atom_id)), dtype=float).reshape(3) for atom_id in chain],
                    dtype=float,
                ).reshape(-1, 3)
                fit_center, fit_axis = self._fit_line_pca(chain_centers, seed_axis=travel)
                if float(np.dot(fit_axis, travel)) < 0.0:
                    fit_axis = -fit_axis
                best_neighbor = None
                best_score = None
                rejection_counts = {}
                for edge in list(adj.get(int(current_id)) or []):
                    nb_id = int(edge.get("neighbor_id", -1))
                    if nb_id <= 0 or nb_id in chain_set:
                        if nb_id > 0 and nb_id in chain_set:
                            rejection_counts["already_used"] = int(rejection_counts.get("already_used", 0)) + 1
                        continue
                    nb_center = np.asarray(center_map.get(int(nb_id)), dtype=float).reshape(3)
                    step = nb_center - current_center
                    step_norm = float(np.linalg.norm(step))
                    if step_norm <= 1e-6:
                        rejection_counts["zero_step"] = int(rejection_counts.get("zero_step", 0)) + 1
                        continue
                    if step_norm > float(max_step_mm):
                        rejection_counts["step_too_long"] = int(rejection_counts.get("step_too_long", 0)) + 1
                        continue
                    step_dir = step / step_norm
                    align_fit = float(np.dot(step_dir, fit_axis))
                    if align_fit < float(cos_forward):
                        rejection_counts["fit_alignment"] = int(rejection_counts.get("fit_alignment", 0)) + 1
                        continue
                    align_recent = float(np.dot(step_dir, travel))
                    if align_recent < float(cos_recent):
                        rejection_counts["recent_alignment"] = int(rejection_counts.get("recent_alignment", 0)) + 1
                        continue
                    lateral = float(self._radial_distance_to_line(nb_center.reshape(1, 3), fit_center, fit_axis)[0])
                    if lateral > 4.5:
                        rejection_counts["lateral_offset"] = int(rejection_counts.get("lateral_offset", 0)) + 1
                        continue
                    if grow_forward:
                        trial_centers = np.vstack([chain_centers, nb_center.reshape(1, 3)])
                    else:
                        trial_centers = np.vstack([nb_center.reshape(1, 3), chain_centers])
                    trial_center, trial_axis = self._fit_line_pca(trial_centers, seed_axis=fit_axis)
                    trial_radial = self._radial_distance_to_line(trial_centers, trial_center, trial_axis)
                    trial_rms = float(np.sqrt(np.mean(np.square(trial_radial)))) if trial_radial.size > 0 else 0.0
                    if trial_rms > 2.0:
                        rejection_counts["trial_rms"] = int(rejection_counts.get("trial_rms", 0)) + 1
                        continue
                    candidate_score = (
                        2.6 * float(align_fit)
                        + 0.8 * float(align_recent)
                        - 0.08 * float(step_norm)
                        - 0.30 * float(lateral)
                        - 1.20 * float(trial_rms)
                    )
                    if best_score is None or candidate_score > best_score:
                        best_neighbor = int(nb_id)
                        best_score = float(candidate_score)
                if best_neighbor is None:
                    if return_debug:
                        if rejection_counts:
                            sorted_items = sorted(rejection_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
                            stop_reason = str(sorted_items[0][0])
                            rejection_hist = ",".join(f"{str(key)}:{int(val)}" for key, val in sorted_items)
                        else:
                            stop_reason = "no_neighbor_candidates"
                            rejection_hist = ""
                        debug_rows.append(
                            {
                                "status": "stopped",
                                "reason": str(stop_reason),
                                "grow_direction": "forward" if grow_forward else "backward",
                                "stop_atom_id": int(current_id),
                                "prev_atom_id": int(prev_id),
                                "rejection_histogram": str(rejection_hist),
                            }
                        )
                    break
                if grow_forward:
                    chain.append(int(best_neighbor))
                else:
                    chain.insert(0, int(best_neighbor))
                chain_set.add(int(best_neighbor))
                if return_debug:
                    debug_rows.append(
                        {
                            "status": "extended",
                            "reason": "",
                            "grow_direction": "forward" if grow_forward else "backward",
                            "stop_atom_id": int(best_neighbor),
                            "prev_atom_id": int(current_id),
                            "rejection_histogram": "",
                        }
                    )
        if return_debug:
            return {"atom_ids": [int(v) for v in chain], "debug_rows": list(debug_rows)}
        return [int(v) for v in chain]

    def _split_contact_chain_on_large_gaps(
        self,
        center_map,
        atom_ids,
        return_debug=False,
    ):
        split_gap_mm = 10.0
        chain_ids = [int(v) for v in list(atom_ids or []) if int(v) > 0]
        if len(chain_ids) < 3:
            segments = [chain_ids] if chain_ids else []
            return {"segments": segments, "debug_rows": []} if return_debug else segments
        centers = np.asarray(
            [np.asarray(center_map.get(int(atom_id)), dtype=float).reshape(3) for atom_id in chain_ids],
            dtype=float,
        ).reshape(-1, 3)
        fit_center, fit_axis = self._fit_line_pca(centers, seed_axis=centers[-1] - centers[0])
        proj = ((centers - fit_center.reshape(1, 3)) @ fit_axis.reshape(3, 1)).reshape(-1)
        order = np.argsort(proj)
        sorted_ids = [int(chain_ids[idx]) for idx in order.tolist()]
        proj_sorted = proj[order]
        gap_values = np.asarray(np.diff(proj_sorted), dtype=float).reshape(-1)
        turn_breaks = set(
            int(v) for v in self._contact_chain_persistent_turn_break_indices(centers[order]) if 0 <= int(v) < len(sorted_ids) - 1
        )
        if gap_values.size == 0 and not turn_breaks:
            segments = [sorted_ids]
            return {"segments": segments, "debug_rows": []} if return_debug else segments
        segments = []
        debug_rows = []
        start_idx = 0
        for gap_idx, gap_mm in enumerate(gap_values.tolist()):
            gap_break = bool(float(gap_mm) > float(split_gap_mm))
            turn_break = bool(int(gap_idx) in turn_breaks)
            if gap_break or turn_break:
                segment = [int(v) for v in sorted_ids[start_idx : gap_idx + 1]]
                if len(segment) >= 3:
                    segments.append(segment)
                    if return_debug:
                        debug_rows.append(
                            {
                                "status": "split",
                                "reason": "gap_and_turn" if gap_break and turn_break else ("gap_too_large" if gap_break else "persistent_turn"),
                                "break_index": int(gap_idx),
                                "break_gap_mm": float(gap_mm),
                                "segment_index": int(len(segments)),
                                "segment_atom_ids": ",".join(str(int(v)) for v in segment),
                            }
                        )
                elif return_debug:
                    debug_rows.append(
                        {
                            "status": "split",
                            "reason": "discard_short_segment",
                            "break_index": int(gap_idx),
                            "break_gap_mm": float(gap_mm),
                            "segment_index": "",
                            "segment_atom_ids": ",".join(str(int(v)) for v in segment),
                        }
                    )
                start_idx = int(gap_idx + 1)
        tail = [int(v) for v in sorted_ids[start_idx:]]
        if len(tail) >= 3:
            segments.append(tail)
            if return_debug and (gap_values.size > 0 or turn_breaks):
                debug_rows.append(
                    {
                        "status": "kept_tail",
                        "reason": "",
                        "break_index": "",
                        "break_gap_mm": "",
                        "segment_index": int(len(segments)),
                        "segment_atom_ids": ",".join(str(int(v)) for v in tail),
                    }
                )
        if not segments and len(sorted_ids) >= 3:
            segments.append([int(v) for v in sorted_ids])
        return {"segments": segments, "debug_rows": debug_rows} if return_debug else segments

    @staticmethod
    def _contact_chain_persistent_turn_break_indices(points_ras, break_angle_deg=28.0, coherence_angle_deg=24.0):
        pts = np.asarray(points_ras, dtype=float).reshape(-1, 3)
        if pts.shape[0] < 5:
            return []
        dirs = []
        for idx in range(pts.shape[0] - 1):
            step = pts[int(idx + 1)] - pts[int(idx)]
            norm = float(np.linalg.norm(step))
            if norm <= 1e-6:
                dirs.append(None)
                continue
            dirs.append(step / norm)
        cos_break = float(np.cos(np.deg2rad(float(break_angle_deg))))
        cos_coherent = float(np.cos(np.deg2rad(float(coherence_angle_deg))))
        break_indices = []
        for boundary in range(2, len(dirs) - 1):
            prev_dirs = [d for d in dirs[int(boundary - 2) : int(boundary)] if d is not None]
            next_dirs = [d for d in dirs[int(boundary) : int(boundary + 2)] if d is not None]
            if len(prev_dirs) < 2 or len(next_dirs) < 2:
                continue
            prev_mean = np.sum(np.asarray(prev_dirs, dtype=float).reshape(-1, 3), axis=0)
            next_mean = np.sum(np.asarray(next_dirs, dtype=float).reshape(-1, 3), axis=0)
            prev_norm = float(np.linalg.norm(prev_mean))
            next_norm = float(np.linalg.norm(next_mean))
            if prev_norm <= 1e-6 or next_norm <= 1e-6:
                continue
            prev_mean = prev_mean / prev_norm
            next_mean = next_mean / next_norm
            if float(np.dot(prev_mean, next_mean)) >= float(cos_break):
                continue
            if min(float(np.dot(d, prev_mean)) for d in prev_dirs) < float(cos_coherent):
                continue
            if min(float(np.dot(d, next_mean)) for d in next_dirs) < float(cos_coherent):
                continue
            break_indices.append(int(boundary))
        return break_indices

    def _score_contact_chain_debug(
        self,
        center_map,
        atom_ids,
    ):
        chain_ids = [int(v) for v in list(atom_ids or []) if int(v) > 0]
        if len(chain_ids) < 3:
            return {
                "accepted": False,
                "reason": "too_few_contacts",
                "atom_ids": [int(v) for v in chain_ids],
            }
        centers = np.asarray(
            [np.asarray(center_map.get(int(atom_id)), dtype=float).reshape(3) for atom_id in chain_ids],
            dtype=float,
        ).reshape(-1, 3)
        fit_center, fit_axis = self._fit_line_pca(centers, seed_axis=centers[-1] - centers[0])
        radial = self._radial_distance_to_line(centers, fit_center, fit_axis)
        rms = float(np.sqrt(np.mean(radial ** 2)))
        proj = ((centers - fit_center.reshape(1, 3)) @ fit_axis.reshape(3, 1)).reshape(-1)
        order = np.argsort(proj)
        proj_sorted = proj[order]
        sorted_ids = [int(chain_ids[idx]) for idx in order.tolist()]
        span = float(np.max(proj_sorted) - np.min(proj_sorted))
        gaps = np.diff(proj_sorted)
        gaps = np.asarray(gaps[np.isfinite(gaps)], dtype=float).reshape(-1)
        positive_gaps = np.asarray(gaps[gaps > 1e-3], dtype=float).reshape(-1)
        if positive_gaps.size == 0:
            return {
                "accepted": False,
                "reason": "no_positive_gaps",
                "atom_ids": [int(v) for v in sorted_ids],
                "span_mm": float(span),
                "rms_mm": float(rms),
            }
        median_gap = float(np.median(positive_gaps))
        if median_gap > 14.0:
            return {
                "accepted": False,
                "reason": "median_gap_too_large",
                "atom_ids": [int(v) for v in sorted_ids],
                "span_mm": float(span),
                "rms_mm": float(rms),
                "median_gap_mm": float(median_gap),
            }
        max_gap = float(np.max(positive_gaps))
        spacing_cv = float(np.std(positive_gaps) / max(1e-6, np.mean(positive_gaps))) if positive_gaps.size >= 2 else 0.0
        turn_metrics = self._contact_chain_turn_metrics(centers[order])
        mean_dot = float(turn_metrics.get("mean_dot", 1.0))
        min_dot = float(turn_metrics.get("min_dot", 1.0))
        if min_dot < float(np.cos(np.deg2rad(40.0))):
            return {
                "accepted": False,
                "reason": "turn_too_sharp",
                "atom_ids": [int(v) for v in sorted_ids],
                "span_mm": float(span),
                "rms_mm": float(rms),
                "median_gap_mm": float(median_gap),
                "max_gap_mm": float(max_gap),
                "spacing_cv": float(spacing_cv),
                "mean_dot": float(mean_dot),
                "min_dot": float(min_dot),
            }
        rms_limit = 1.75
        if len(sorted_ids) >= 5:
            rms_limit += 0.50
        if mean_dot >= float(np.cos(np.deg2rad(20.0))):
            rms_limit += 0.75
        elif mean_dot >= float(np.cos(np.deg2rad(28.0))):
            rms_limit += 0.35
        if rms > float(rms_limit):
            return {
                "accepted": False,
                "reason": "rms_too_large",
                "atom_ids": [int(v) for v in sorted_ids],
                "span_mm": float(span),
                "rms_mm": float(rms),
                "rms_limit_mm": float(rms_limit),
                "median_gap_mm": float(median_gap),
                "max_gap_mm": float(max_gap),
                "spacing_cv": float(spacing_cv),
                "mean_dot": float(mean_dot),
                "min_dot": float(min_dot),
            }
        score = (
            6.0 * float(len(sorted_ids))
            + 1.0 * float(span)
            + 8.0 * float(max(0.0, 1.0 - min(1.0, spacing_cv / 0.50)))
            + 5.0 * float(mean_dot)
            - 2.5 * float(rms)
            - 0.4 * float(max_gap)
        )
        return {
            "accepted": True,
            "reason": "",
            "atom_ids": [int(v) for v in sorted_ids],
            "score": float(score),
            "span_mm": float(span),
            "rms_mm": float(rms),
            "rms_limit_mm": float(rms_limit),
            "median_gap_mm": float(median_gap),
            "max_gap_mm": float(max_gap),
            "spacing_cv": float(spacing_cv),
            "mean_dot": float(mean_dot),
            "min_dot": float(min_dot),
        }

    @staticmethod
    def _contact_chain_turn_metrics(points_ras):
        pts = np.asarray(points_ras, dtype=float).reshape(-1, 3)
        if pts.shape[0] < 3:
            return {"mean_dot": 1.0, "min_dot": 1.0}
        dirs = []
        for idx in range(pts.shape[0] - 1):
            step = pts[int(idx + 1)] - pts[int(idx)]
            norm = float(np.linalg.norm(step))
            if norm <= 1e-6:
                continue
            dirs.append(step / norm)
        if len(dirs) < 2:
            return {"mean_dot": 1.0, "min_dot": 1.0}
        dots = [float(np.clip(np.dot(dirs[idx], dirs[idx + 1]), -1.0, 1.0)) for idx in range(len(dirs) - 1)]
        if not dots:
            return {"mean_dot": 1.0, "min_dot": 1.0}
        return {"mean_dot": float(np.mean(dots)), "min_dot": float(np.min(dots))}
