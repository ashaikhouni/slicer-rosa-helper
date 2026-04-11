"""Support atom normalization and geometric helpers for Deep Core candidates."""

try:
    import numpy as np
except ImportError:
    np = None


class DeepCoreCandidateInputMixin:
    """Normalize support atoms and provide shared geometric helpers."""

    def _prepare_support_atom_inputs(
        self,
        support_atoms=None,
        token_points_ras=None,
        token_blob_ids=None,
        token_atom_ids=None,
        blob_axes_ras_by_id=None,
        blob_elongation_by_id=None,
    ):
        axis_map_in = {int(k): np.asarray(v, dtype=float).reshape(3) for k, v in dict(blob_axes_ras_by_id or {}).items()}
        elong_map_in = {int(k): float(v) for k, v in dict(blob_elongation_by_id or {}).items()}
        normalized_atoms = []
        if support_atoms:
            for default_atom_id, atom in enumerate(list(support_atoms or []), start=1):
                normalized = self._normalize_support_atom(atom, default_atom_id=default_atom_id)
                if normalized is not None:
                    normalized_atoms.append(normalized)
        if not normalized_atoms:
            pts = np.asarray(token_points_ras, dtype=float).reshape(-1, 3)
            blob_ids = np.asarray(
                token_blob_ids if token_blob_ids is not None else np.arange(pts.shape[0]),
                dtype=np.int32,
            ).reshape(-1)
            atom_ids = np.asarray(
                token_atom_ids if token_atom_ids is not None else blob_ids,
                dtype=np.int32,
            ).reshape(-1)
            if blob_ids.shape[0] != pts.shape[0]:
                blob_ids = np.arange(pts.shape[0], dtype=np.int32)
            if atom_ids.shape[0] != pts.shape[0]:
                atom_ids = np.asarray(blob_ids, dtype=np.int32).reshape(-1)
            for blob_id in np.unique(blob_ids).tolist():
                blob_id = int(blob_id)
                sel = blob_ids == blob_id
                blob_pts = pts[sel]
                if blob_pts.shape[0] == 0:
                    continue
                axis_val = axis_map_in.get(blob_id)
                if axis_val is None:
                    axis_val = [0.0, 0.0, 1.0]
                axis = np.asarray(axis_val, dtype=float).reshape(3)
                normalized = self._build_legacy_support_atom(
                    atom_id=blob_id,
                    parent_blob_id=blob_id,
                    support_points_ras=blob_pts,
                    axis_ras=axis,
                    elongation=float(elong_map_in.get(blob_id, 1.0)),
                )
                if normalized is not None:
                    normalized_atoms.append(normalized)
            parent_blob_id_map = {int(blob_id): int(blob_id) for blob_id in np.unique(blob_ids).astype(int).tolist()}
            return {
                "support_atoms": list(normalized_atoms),
                "token_points_ras": np.asarray(pts, dtype=float).reshape(-1, 3),
                "token_blob_ids": np.asarray(blob_ids, dtype=np.int32).reshape(-1),
                "token_atom_ids": np.asarray(atom_ids, dtype=np.int32).reshape(-1),
                "axis_map": dict(axis_map_in),
                "elongation_map": dict(elong_map_in),
                "parent_blob_id_map": dict(parent_blob_id_map),
            }
        pts = np.asarray(token_points_ras, dtype=float).reshape(-1, 3)
        blob_ids = np.asarray(
            token_blob_ids if token_blob_ids is not None else np.arange(pts.shape[0]),
            dtype=np.int32,
        ).reshape(-1)
        atom_ids = np.asarray(
            token_atom_ids if token_atom_ids is not None else np.zeros((pts.shape[0],), dtype=np.int32),
            dtype=np.int32,
        ).reshape(-1)
        if blob_ids.shape[0] != pts.shape[0]:
            blob_ids = np.arange(pts.shape[0], dtype=np.int32)
        if atom_ids.shape[0] != pts.shape[0]:
            atom_ids = np.zeros((pts.shape[0],), dtype=np.int32)
        axis_map = dict(axis_map_in)
        elongation_map = dict(elong_map_in)
        parent_blob_id_map = {int(blob_id): int(blob_id) for blob_id in np.unique(blob_ids).astype(int).tolist()}
        for atom in normalized_atoms:
            atom_id = int(atom.get("atom_id", 0))
            if atom_id <= 0:
                continue
            parent_blob_id_map[atom_id] = int(atom.get("parent_blob_id", atom_id))
        return {
            "support_atoms": list(normalized_atoms),
            "token_points_ras": np.asarray(pts, dtype=float).reshape(-1, 3),
            "token_blob_ids": np.asarray(blob_ids, dtype=np.int32).reshape(-1),
            "token_atom_ids": np.asarray(atom_ids, dtype=np.int32).reshape(-1),
            "axis_map": dict(axis_map),
            "elongation_map": dict(elongation_map),
            "parent_blob_id_map": dict(parent_blob_id_map),
        }

    def _normalize_support_atom(self, atom, default_atom_id=1):
        atom_dict = dict(atom or {})
        support_points = np.asarray(
            atom_dict.get("support_points_ras") or atom_dict.get("points_ras") or [],
            dtype=float,
        ).reshape(-1, 3)
        if support_points.shape[0] == 0:
            return None
        atom_id = int(atom_dict.get("atom_id") or atom_dict.get("blob_id") or default_atom_id)
        parent_blob_id = int(atom_dict.get("parent_blob_id") or atom_id)
        axis = np.asarray(atom_dict.get("axis_ras") or [0.0, 0.0, 1.0], dtype=float).reshape(3)
        axis_norm = float(np.linalg.norm(axis))
        axis = axis / max(axis_norm, 1e-6)
        center = np.asarray(atom_dict.get("center_ras") or np.mean(support_points, axis=0), dtype=float).reshape(3)
        proj = ((support_points - center.reshape(1, 3)) @ axis.reshape(3, 1)).reshape(-1)
        if proj.size == 0:
            proj = np.zeros((1,), dtype=float)
        start_ras = np.asarray(atom_dict.get("start_ras") or (center + axis * float(np.min(proj))), dtype=float).reshape(3)
        end_ras = np.asarray(atom_dict.get("end_ras") or (center + axis * float(np.max(proj))), dtype=float).reshape(3)
        radial = self._radial_distance_to_line(support_points, center, axis)
        thickness_mm = float(
            atom_dict.get("thickness_mm")
            or atom_dict.get("diameter_mm")
            or (2.0 * np.percentile(radial, 95.0) if radial.size > 0 else 0.0)
        )
        span_mm = float(atom_dict.get("span_mm") or np.linalg.norm(end_ras - start_ras))
        occupancy = dict(atom_dict.get("occupancy") or {})
        if not occupancy:
            occupancy = self._axial_occupancy_metrics(proj, bin_mm=2.5)
        return {
            "atom_id": int(atom_id),
            "parent_blob_id": int(parent_blob_id),
            "kind": str(atom_dict.get("kind") or "support_atom"),
            "source_blob_class": str(atom_dict.get("source_blob_class") or ""),
            "direct_blob_line_atom": bool(atom_dict.get("direct_blob_line_atom", False)),
            "support_points_ras": np.asarray(support_points, dtype=float).reshape(-1, 3).tolist(),
            "center_ras": [float(v) for v in center],
            "axis_ras": [float(v) for v in axis],
            "axis_reliable": bool(atom_dict.get("axis_reliable", support_points.shape[0] >= 2)),
            "start_ras": [float(v) for v in start_ras],
            "end_ras": [float(v) for v in end_ras],
            "span_mm": float(span_mm),
            "thickness_mm": float(thickness_mm),
            "elongation": float(atom_dict.get("elongation", max(1.0, span_mm / max(0.5, thickness_mm)))),
            "support_point_count": int(atom_dict.get("support_point_count", support_points.shape[0])),
            "occupancy": {
                "occupied_bins": int(occupancy.get("occupied_bins", 0)),
                "total_bins": int(occupancy.get("total_bins", 0)),
                "coverage": float(occupancy.get("coverage", 0.0)),
                "max_gap_bins": int(occupancy.get("max_gap_bins", 0)),
                "run_count": int(occupancy.get("run_count", 0)),
                "run_lengths_bins": [int(v) for v in list(occupancy.get("run_lengths_bins") or [])],
                "max_run_bins": int(occupancy.get("max_run_bins", 0)),
            },
            "node_ids": [int(v) for v in list(atom_dict.get("node_ids") or []) if int(v) > 0],
            "contact_atom_ids": [int(v) for v in list(atom_dict.get("contact_atom_ids") or []) if int(v) > 0],
            "parent_blob_id_list": [int(v) for v in list(atom_dict.get("parent_blob_id_list") or []) if int(v) > 0],
        }

    def _build_legacy_support_atom(
        self,
        atom_id,
        parent_blob_id,
        support_points_ras,
        axis_ras,
        elongation,
    ):
        points = np.asarray(support_points_ras, dtype=float).reshape(-1, 3)
        if points.shape[0] == 0:
            return None
        center = np.mean(points, axis=0)
        axis = np.asarray(axis_ras, dtype=float).reshape(3)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-6:
            _center, axis = self._fit_line_pca(points, seed_axis=[0.0, 0.0, 1.0])
        else:
            axis = axis / axis_norm
        proj = ((points - center.reshape(1, 3)) @ axis.reshape(3, 1)).reshape(-1)
        start_ras = center + axis * float(np.min(proj))
        end_ras = center + axis * float(np.max(proj))
        radial = self._radial_distance_to_line(points, center, axis)
        thickness_mm = float(2.0 * np.percentile(radial, 95.0)) if radial.size > 0 else 0.0
        occupancy = self._axial_occupancy_metrics(proj, bin_mm=2.5)
        return {
            "atom_id": int(atom_id),
            "parent_blob_id": int(parent_blob_id),
            "kind": "legacy_blob",
            "support_points_ras": np.asarray(points, dtype=float).reshape(-1, 3).tolist(),
            "center_ras": [float(v) for v in center],
            "axis_ras": [float(v) for v in axis],
            "axis_reliable": bool(points.shape[0] >= 2),
            "start_ras": [float(v) for v in start_ras],
            "end_ras": [float(v) for v in end_ras],
            "span_mm": float(np.linalg.norm(end_ras - start_ras)),
            "thickness_mm": float(thickness_mm),
            "elongation": float(max(1.0, elongation)),
            "support_point_count": int(points.shape[0]),
            "occupancy": occupancy,
            "node_ids": [],
        }

    def _fit_line_pca(self, pts_ras, seed_axis=None):
        pts = np.asarray(pts_ras, dtype=float).reshape(-1, 3)
        center = np.mean(pts, axis=0)
        centered = pts - center.reshape(1, 3)
        cov = (centered.T @ centered) / max(1, pts.shape[0] - 1)
        evals, evecs = np.linalg.eigh(cov)
        axis = evecs[:, int(np.argmax(evals))]
        axis = np.asarray(axis, dtype=float).reshape(3)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-6:
            axis = np.asarray(seed_axis if seed_axis is not None else [0.0, 0.0, 1.0], dtype=float).reshape(3)
        else:
            axis = axis / axis_norm
        if seed_axis is not None:
            seed = np.asarray(seed_axis, dtype=float).reshape(3)
            if float(np.dot(axis, seed)) < 0.0:
                axis = -axis
        return center.astype(float), axis.astype(float)

    def _radial_distance_to_line(self, pts_ras, center_ras, axis_ras):
        pts = np.asarray(pts_ras, dtype=float).reshape(-1, 3)
        center = np.asarray(center_ras, dtype=float).reshape(3)
        axis = np.asarray(axis_ras, dtype=float).reshape(3)
        vec = pts - center.reshape(1, 3)
        proj = (vec @ axis.reshape(3, 1)).reshape(-1, 1) * axis.reshape(1, 3)
        radial = vec - proj
        return np.linalg.norm(radial, axis=1)

    def _axial_occupancy_metrics(self, proj_vals, bin_mm=2.5):
        proj = np.asarray(proj_vals, dtype=float).reshape(-1)
        if proj.size == 0:
            return {
                "occupied_bins": 0,
                "total_bins": 0,
                "coverage": 0.0,
                "max_gap_bins": 0,
                "run_count": 0,
                "run_lengths_bins": [],
                "max_run_bins": 0,
            }
        pmin = float(np.min(proj))
        pmax = float(np.max(proj))
        span = float(max(0.0, pmax - pmin))
        nbins = max(1, int(np.floor(span / float(max(0.5, bin_mm)))) + 1)
        bin_ids = np.floor((proj - pmin) / float(max(0.5, bin_mm))).astype(int)
        bin_ids = np.clip(bin_ids, 0, nbins - 1)
        occupied = np.unique(bin_ids)
        occupied_bins = int(occupied.size)
        total_bins = int(nbins)
        coverage = float(occupied_bins / max(1, total_bins))
        max_gap = 0
        prev = None
        run_lengths = []
        run_start = None
        run_prev = None
        for idx in occupied.tolist():
            if prev is not None:
                max_gap = max(max_gap, int(idx - prev - 1))
            if run_start is None:
                run_start = int(idx)
                run_prev = int(idx)
            elif int(idx) == int(run_prev) + 1:
                run_prev = int(idx)
            else:
                run_lengths.append(int(run_prev - run_start + 1))
                run_start = int(idx)
                run_prev = int(idx)
            prev = int(idx)
        if run_start is not None and run_prev is not None:
            run_lengths.append(int(run_prev - run_start + 1))
        return {
            "occupied_bins": occupied_bins,
            "total_bins": total_bins,
            "coverage": coverage,
            "max_gap_bins": int(max_gap),
            "run_count": int(len(run_lengths)),
            "run_lengths_bins": [int(v) for v in run_lengths],
            "max_run_bins": int(max(run_lengths) if run_lengths else 0),
        }

    def _occupancy_supports_line(self, 
        occ,
        min_coverage=0.40,
        max_gap_bins=2,
        max_group_runs=3,
        min_group_occupied_bins=6,
        min_group_run_bins=2,
    ):
        coverage = float(occ.get("coverage", 0.0))
        max_gap = int(occ.get("max_gap_bins", 0))
        if coverage >= float(min_coverage) and max_gap <= int(max_gap_bins):
            return True
        run_lengths = [int(v) for v in list(occ.get("run_lengths_bins") or [])]
        strong_runs = [int(v) for v in run_lengths if int(v) >= int(min_group_run_bins)]
        if not strong_runs:
            return False
        run_count = int(len(strong_runs))
        occupied_bins = int(occ.get("occupied_bins", 0))
        return bool(
            run_count >= 2
            and run_count <= int(max_group_runs)
            and occupied_bins >= int(min_group_occupied_bins)
            and max(strong_runs) >= int(min_group_run_bins)
        )

    def _chain_turn_metrics(self, chain_indices, pts_ras):
        chain = np.asarray(chain_indices, dtype=int).reshape(-1)
        pts = np.asarray(pts_ras, dtype=float).reshape(-1, 3)
        if chain.size < 3:
            return {"mean_dot": 1.0, "min_dot": 1.0}
        dirs = []
        for idx in range(chain.size - 1):
            step = pts[int(chain[idx + 1])] - pts[int(chain[idx])]
            norm = float(np.linalg.norm(step))
            if norm <= 1e-6:
                continue
            dirs.append(step / norm)
        if len(dirs) < 2:
            return {"mean_dot": 1.0, "min_dot": 1.0}
        dots = []
        for idx in range(len(dirs) - 1):
            dots.append(float(np.clip(np.dot(dirs[idx], dirs[idx + 1]), -1.0, 1.0)))
        if not dots:
            return {"mean_dot": 1.0, "min_dot": 1.0}
        return {"mean_dot": float(np.mean(dots)), "min_dot": float(np.min(dots))}

    def _point_chain_turn_metrics(self, points_ras):
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
        dots = []
        for idx in range(len(dirs) - 1):
            dots.append(float(np.clip(np.dot(dirs[idx], dirs[idx + 1]), -1.0, 1.0)))
        if not dots:
            return {"mean_dot": 1.0, "min_dot": 1.0}
        return {"mean_dot": float(np.mean(dots)), "min_dot": float(np.min(dots))}

    def _token_line_candidate_quality(self, candidate):
        score = float(candidate.get("score", 0.0))
        coverage = float(candidate.get("coverage", 0.0))
        span = float(candidate.get("span_mm", 0.0))
        inliers = int(candidate.get("inlier_count", 0))
        gap = int(candidate.get("max_gap_bins", 0))
        density = float(inliers) / float(max(1.0, span))
        quality = score + 10.0 * coverage + 4.0 * density - 1.2 * float(gap * gap)
        if gap > 4:
            quality -= 2.0 * float(gap - 4)
        if bool(candidate.get("used_relaxed_grouped_occupancy")):
            quality -= 2.5 * float(max(0.0, span - 62.0))
            quality -= 1.0 * float(max(0, gap - 3))
        return float(quality)
