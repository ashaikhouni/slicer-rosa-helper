"""Support-atom construction and blob tokenization for Deep Core."""

try:
    import numpy as np
except ImportError:
    np = None

from .deep_core_config import deep_core_default_config


_DEEP_CORE_DEFAULTS = deep_core_default_config()
_DEFAULT_ANNULUS_CONFIG = _DEEP_CORE_DEFAULTS.annulus
_DEFAULT_INTERNAL_CONFIG = _DEEP_CORE_DEFAULTS.internal


class DeepCoreAtomBuilderMixin:
    """Turn deep metal blobs into support atoms and tokenized support samples."""

    def _build_support_atom_payload(
        self,
        volume_node,
        labels_kji,
        blobs,
        support_spacing_mm=2.5,
        component_min_elongation=4.0,
        line_atom_diameter_max_mm=2.0,
        line_atom_min_span_mm=10.0,
        line_atom_min_pca_dominance=6.0,
        contact_component_diameter_max_mm=10.0,
        support_cube_size_mm=5.0,
        head_distance_map_kji=None,
        annulus_config=None,
        internal_config=None,
    ):
        annulus_cfg = annulus_config if annulus_config is not None else _DEFAULT_ANNULUS_CONFIG
        internal_cfg = internal_config if internal_config is not None else _DEFAULT_INTERNAL_CONFIG
        labels = np.asarray(labels_kji, dtype=np.int32)
        annulus_reference_values_hu = self._scan_reference_hu_values(
            volume_node=volume_node,
            lower_hu=-500.0,
            upper_hu=float(annulus_cfg.annulus_reference_upper_hu) if annulus_cfg.annulus_reference_upper_hu is not None else None,
        )
        if labels.size == 0:
            return {
                "support_atoms": [],
                "points_ras": np.empty((0, 3), dtype=float),
                "blob_ids": np.zeros((0,), dtype=np.int32),
                "atom_ids": np.zeros((0,), dtype=np.int32),
                "complex_blob_chain_rows": [],
                "contact_chain_rows": [],
                "contact_chain_debug_rows": [],
                "line_blob_points_ras": np.empty((0, 3), dtype=float),
                "contact_blob_points_ras": np.empty((0, 3), dtype=float),
                "complex_blob_points_ras": np.empty((0, 3), dtype=float),
                "axes_ras_by_id": {},
                "elongation_by_id": {},
                "parent_blob_ids_by_id": {},
                "blob_class_by_id": {},
            }
        spacing = float(max(0.5, support_spacing_mm))
        min_elong = float(max(1.0, component_min_elongation))
        support_atoms = []
        sample_points = []
        sample_blob_ids = []
        sample_atom_ids = []
        complex_blob_chain_rows = []
        contact_chain_rows = []
        contact_chain_debug_rows = []
        blob_class_by_id = {}
        class_sample_points = {
            "line_blob": [],
            "contact_blob": [],
            "complex_blob": [],
        }
        axes_ras_by_id = {}
        elongation_by_id = {}
        parent_blob_ids_by_id = {}
        next_atom_id = 1
        for blob in list(blobs or []):
            raw_blob_id = int(blob.get("blob_id") or 0)
            if raw_blob_id <= 0:
                continue
            coords_kji = np.argwhere(labels == raw_blob_id)
            if coords_kji.size == 0:
                continue
            ras_pts = np.asarray(self._ijk_kji_to_ras_points(volume_node, coords_kji.astype(float)), dtype=float).reshape(-1, 3)
            if ras_pts.shape[0] == 0:
                continue
            centroid_ras = blob.get("centroid_ras")
            if centroid_ras is None:
                centroid = np.mean(ras_pts, axis=0)
            else:
                centroid = np.asarray(centroid_ras, dtype=float).reshape(3)
            axis = np.asarray(blob.get("pca_axis_ras") or [0.0, 0.0, 1.0], dtype=float).reshape(3)
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm <= 1e-6:
                axis = np.asarray([0.0, 0.0, 1.0], dtype=float)
            else:
                axis = axis / axis_norm
            shape = self._point_cloud_shape_metrics(ras_pts, seed_axis=axis)
            max_extent_mm = float(max(shape.get("extent_mm", [0.0, 0.0, 0.0]) or [0.0]))
            span_mm = float(shape.get("span_mm", 0.0))
            diameter_mm = float(shape.get("diameter_mm", 0.0))
            elongation = float(shape.get("elongation", blob.get("elongation") or 1.0))
            pca_dominance = float(shape.get("pca_dominance", 1.0))
            shape_center = np.asarray(shape.get("center_ras", centroid), dtype=float).reshape(3)
            shape_axis = np.asarray(shape.get("axis_ras", axis), dtype=float).reshape(3)
            axes_ras_by_id[int(raw_blob_id)] = np.asarray(shape_axis, dtype=float).reshape(3)
            elongation_by_id[int(raw_blob_id)] = float(elongation)
            parent_blob_ids_by_id[int(raw_blob_id)] = int(raw_blob_id)
            is_contact_blob = bool(
                float(max_extent_mm) <= float(contact_component_diameter_max_mm)
                and float(diameter_mm) <= float(contact_component_diameter_max_mm)
            )
            cube_nodes = {"nodes": []}
            if is_contact_blob:
                blob_class = "contact_blob"
                blob_tokens = self._nearest_support_point_ras(ras_pts, centroid).reshape(1, 3)
            else:
                cube_nodes = self._build_blob_cube_nodes(
                    coords_kji=coords_kji,
                    ras_pts=ras_pts,
                    provisional_axis_ras=shape_axis,
                    cube_size_mm=float(max(1.0, support_cube_size_mm)),
                    volume_node=volume_node,
                    head_distance_map_kji=head_distance_map_kji,
                    annulus_reference_values_hu=annulus_reference_values_hu,
                )
                blob_tokens = np.asarray(
                    [node.get("support_point_ras") for node in list(cube_nodes.get("nodes") or [])],
                    dtype=float,
                ).reshape(-1, 3)
                if blob_tokens.shape[0] == 0:
                    blob_tokens = self._nearest_support_point_ras(ras_pts, centroid).reshape(1, 3)
                blob_class = "complex_blob"
                if (
                    float(span_mm) >= float(line_atom_min_span_mm)
                    and float(elongation) >= float(max(1.0, min_elong))
                    and float(pca_dominance) >= float(max(1.0, line_atom_min_pca_dominance))
                    and float(diameter_mm) <= float(line_atom_diameter_max_mm)
                    and self._point_cloud_fits_axis_corridor(
                        blob_tokens,
                        axis_ras=shape_axis,
                        center_ras=shape_center,
                        diameter_max_mm=float(line_atom_diameter_max_mm),
                    )
                    and int(blob_tokens.shape[0]) >= 2
                ):
                    blob_class = "line_blob"
            blob_class_by_id[int(raw_blob_id)] = str(blob_class)

            token_start_idx = int(len(sample_points))
            sample_points.extend(np.asarray(blob_tokens, dtype=float).reshape(-1, 3).tolist())
            sample_blob_ids.extend([int(raw_blob_id)] * int(blob_tokens.shape[0]))
            sample_atom_ids.extend([0] * int(blob_tokens.shape[0]))
            class_sample_points[str(blob_class)].extend(np.asarray(blob_tokens, dtype=float).reshape(-1, 3).tolist())

            if blob_class == "contact_blob":
                atom = self._create_support_atom(
                    atom_id=int(next_atom_id),
                    parent_blob_id=int(raw_blob_id),
                    kind="contact",
                    support_points_ras=blob_tokens,
                    fit_points_ras=ras_pts,
                    axis_ras=shape_axis,
                    center_ras=shape_center,
                    axis_reliable=False,
                    sample_spacing_mm=spacing,
                    extra_fields={"source_blob_class": str(blob_class)},
                )
                if atom is not None:
                    support_atoms.append(atom)
                    parent_blob_ids_by_id[int(next_atom_id)] = int(raw_blob_id)
                    for local_idx in range(int(blob_tokens.shape[0])):
                        sample_atom_ids[token_start_idx + local_idx] = int(next_atom_id)
                    next_atom_id += 1
                continue

            if blob_class == "line_blob":
                atom = self._create_support_atom(
                    atom_id=int(next_atom_id),
                    parent_blob_id=int(raw_blob_id),
                    kind="line",
                    support_points_ras=blob_tokens,
                    fit_points_ras=ras_pts,
                    axis_ras=shape_axis,
                    center_ras=shape_center,
                    axis_reliable=bool(blob_tokens.shape[0] >= 2),
                    sample_spacing_mm=spacing,
                    extra_fields={
                        "source_blob_class": str(blob_class),
                        "direct_blob_line_atom": True,
                    },
                )
                if atom is not None:
                    support_atoms.append(atom)
                    parent_blob_ids_by_id[int(next_atom_id)] = int(raw_blob_id)
                    for local_idx in range(int(blob_tokens.shape[0])):
                        sample_atom_ids[token_start_idx + local_idx] = int(next_atom_id)
                    next_atom_id += 1
                continue

            complex_debug = self._extract_complex_blob_paths(
                slab_nodes=list(cube_nodes.get("nodes") or []),
                provisional_axis_ras=shape_axis,
                sample_spacing_mm=spacing,
                min_coverage=float(internal_cfg.complex_blob_min_coverage),
                max_gap_bins=int(internal_cfg.complex_blob_max_gap_bins),
                candidate_guard_diameter_mm=float(max(1.0, float(internal_cfg.complex_blob_candidate_guard_scale) * line_atom_diameter_max_mm)),
                parent_blob_id=int(raw_blob_id),
                volume_node=volume_node,
                head_distance_map_kji=head_distance_map_kji,
                annulus_reference_values_hu=annulus_reference_values_hu,
                annulus_config=annulus_cfg,
                internal_config=internal_cfg,
                return_debug=True,
            )
            complex_paths = list(dict(complex_debug or {}).get("paths") or [])
            complex_blob_chain_rows.extend(list(dict(complex_debug or {}).get("chain_rows") or []))
            if not complex_paths:
                continue
            node_to_local_token_idx = {
                int(node.get("node_id", -1)): int(idx)
                for idx, node in enumerate(list(cube_nodes.get("nodes") or []))
                if int(node.get("node_id", -1)) > 0
            }
            for path in list(complex_paths or []):
                token_points = np.asarray(path.get("token_points_ras"), dtype=float).reshape(-1, 3)
                if token_points.shape[0] == 0:
                    continue
                node_ids = [int(v) for v in list(path.get("node_ids") or []) if int(v) > 0]
                atom = self._create_support_atom(
                    atom_id=int(next_atom_id),
                    parent_blob_id=int(raw_blob_id),
                    kind="line",
                    support_points_ras=token_points,
                    fit_points_ras=path.get("fit_points_ras"),
                    axis_ras=path.get("axis_ras", shape_axis),
                    center_ras=path.get("center_ras", shape_center),
                    axis_reliable=bool(token_points.shape[0] >= 2),
                    sample_spacing_mm=spacing,
                    node_ids=node_ids,
                    extra_fields={"source_blob_class": str(blob_class)},
                )
                if not self._atom_supports_line(
                    atom,
                    min_coverage=float(internal_cfg.complex_blob_min_coverage),
                    max_gap_bins=int(internal_cfg.complex_blob_max_gap_bins),
                    min_support_points=2,
                ):
                    continue
                if not self._complex_blob_axis_dominance_supports_line(
                    blob_points_ras=ras_pts,
                    atom=atom,
                    axial_bin_mm=spacing,
                    guard_diameter_mm=float(max(1.0, float(internal_cfg.complex_blob_axis_guard_scale) * line_atom_diameter_max_mm)),
                    min_dominant_bin_coverage=float(internal_cfg.complex_blob_min_dominant_bin_coverage),
                    max_nondominant_run_bins=int(internal_cfg.complex_blob_max_nondominant_run_bins),
                    slab_nodes=list(cube_nodes.get("nodes") or []),
                ):
                    continue
                support_atoms.append(atom)
                parent_blob_ids_by_id[int(next_atom_id)] = int(raw_blob_id)
                for node_id in node_ids:
                    local_token_idx = node_to_local_token_idx.get(int(node_id))
                    if local_token_idx is None:
                        continue
                    sample_atom_ids[token_start_idx + int(local_token_idx)] = int(next_atom_id)
                next_atom_id += 1
            continue
        contact_chain_debug = self._build_contact_chain_rows_from_atoms(
            support_atoms=support_atoms,
            max_neighbor_count=4,
        )
        contact_chain_rows = list(dict(contact_chain_debug or {}).get("rows") or [])
        contact_chain_debug_rows = list(dict(contact_chain_debug or {}).get("debug_rows") or [])
        contact_chain_atoms = list(dict(contact_chain_debug or {}).get("atoms") or [])
        contact_token_indices_by_atom_id = {}
        for token_idx, atom_id in enumerate(list(sample_atom_ids or [])):
            atom_id = int(atom_id)
            if atom_id <= 0:
                continue
            contact_token_indices_by_atom_id.setdefault(int(atom_id), []).append(int(token_idx))
        contact_atom_map = {
            int(atom.get("atom_id", -1)): dict(atom)
            for atom in list(support_atoms or [])
            if int(dict(atom or {}).get("atom_id", -1)) > 0 and str(dict(atom or {}).get("kind") or "") == "contact"
        }
        for chain_info in list(contact_chain_atoms or []):
            chain_contact_atom_ids = [int(v) for v in list(dict(chain_info or {}).get("atom_ids") or []) if int(v) > 0]
            if len(chain_contact_atom_ids) < 3:
                continue
            chain_points = []
            parent_blob_id_list = []
            for contact_atom_id in chain_contact_atom_ids:
                contact_atom = dict(contact_atom_map.get(int(contact_atom_id)) or {})
                if not contact_atom:
                    continue
                center = np.asarray(contact_atom.get("center_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
                chain_points.append(np.asarray(center, dtype=float).reshape(3))
                parent_blob_id_list.append(int(contact_atom.get("parent_blob_id", -1)))
            chain_points = np.asarray(chain_points, dtype=float).reshape(-1, 3)
            if chain_points.shape[0] < 3:
                continue
            synthetic_parent_blob_id = int(next_atom_id)
            atom = self._create_support_atom(
                atom_id=int(next_atom_id),
                parent_blob_id=int(synthetic_parent_blob_id),
                kind="line",
                support_points_ras=chain_points,
                fit_points_ras=chain_points,
                axis_ras=dict(chain_info or {}).get("axis_ras"),
                center_ras=dict(chain_info or {}).get("center_ras"),
                axis_reliable=bool(chain_points.shape[0] >= 2),
                sample_spacing_mm=spacing,
                node_ids=[int(v) for v in chain_contact_atom_ids],
                extra_fields={
                    "source_blob_class": "contact_chain",
                    "contact_atom_ids": [int(v) for v in chain_contact_atom_ids],
                    "parent_blob_id_list": [int(v) for v in sorted(set(v for v in parent_blob_id_list if int(v) > 0))],
                },
            )
            if atom is None:
                continue
            support_atoms.append(atom)
            parent_blob_ids_by_id[int(next_atom_id)] = int(synthetic_parent_blob_id)
            for contact_atom_id in chain_contact_atom_ids:
                for token_idx in list(contact_token_indices_by_atom_id.get(int(contact_atom_id)) or []):
                    if 0 <= int(token_idx) < len(sample_atom_ids):
                        sample_atom_ids[int(token_idx)] = int(next_atom_id)
            next_atom_id += 1
        return {
            "support_atoms": list(support_atoms),
            "points_ras": np.asarray(sample_points, dtype=float).reshape(-1, 3),
            "blob_ids": np.asarray(sample_blob_ids, dtype=np.int32).reshape(-1),
            "atom_ids": np.asarray(sample_atom_ids, dtype=np.int32).reshape(-1),
            "complex_blob_chain_rows": list(complex_blob_chain_rows),
            "contact_chain_rows": list(contact_chain_rows),
            "contact_chain_debug_rows": list(contact_chain_debug_rows),
            "line_blob_points_ras": np.asarray(class_sample_points.get("line_blob") or [], dtype=float).reshape(-1, 3),
            "contact_blob_points_ras": np.asarray(class_sample_points.get("contact_blob") or [], dtype=float).reshape(-1, 3),
            "complex_blob_points_ras": np.asarray(class_sample_points.get("complex_blob") or [], dtype=float).reshape(-1, 3),
            "axes_ras_by_id": dict(axes_ras_by_id),
            "elongation_by_id": dict(elongation_by_id),
            "parent_blob_ids_by_id": dict(parent_blob_ids_by_id),
            "blob_class_by_id": dict(blob_class_by_id),
        }

    def _create_support_atom(
        self,
        atom_id,
        parent_blob_id,
        kind,
        support_points_ras,
        fit_points_ras,
        axis_ras,
        center_ras,
        axis_reliable,
        sample_spacing_mm=2.5,
        node_ids=None,
        extra_fields=None,
    ):
        support_points = np.asarray(support_points_ras, dtype=float).reshape(-1, 3)
        if support_points.shape[0] == 0:
            return None
        fit_points = np.asarray(
            fit_points_ras if fit_points_ras is not None else support_points,
            dtype=float,
        ).reshape(-1, 3)
        if fit_points.shape[0] == 0:
            fit_points = np.asarray(support_points, dtype=float).reshape(-1, 3)
        axis_seed = np.asarray(axis_ras if axis_ras is not None else [0.0, 0.0, 1.0], dtype=float).reshape(3)
        center_seed = np.asarray(
            center_ras if center_ras is not None else np.mean(fit_points, axis=0),
            dtype=float,
        ).reshape(3)
        shape = self._point_cloud_shape_metrics(fit_points, seed_axis=axis_seed)
        axis = np.asarray(shape.get("axis_ras", axis_seed), dtype=float).reshape(3)
        axis_norm = float(np.linalg.norm(axis))
        axis = axis / max(axis_norm, 1e-6)
        if float(np.dot(axis, axis_seed)) < 0.0:
            axis = -axis
        center = np.asarray(shape.get("center_ras", center_seed), dtype=float).reshape(3)
        proj_fit = ((fit_points - center.reshape(1, 3)) @ axis.reshape(3, 1)).reshape(-1)
        pmin = float(np.min(proj_fit)) if proj_fit.size else 0.0
        pmax = float(np.max(proj_fit)) if proj_fit.size else 0.0
        if support_points.shape[0] == 1:
            support_point = np.asarray(support_points[0], dtype=float).reshape(3)
            start_ras = support_point.copy()
            end_ras = support_point.copy()
        else:
            start_ras = center + axis * pmin
            end_ras = center + axis * pmax
        proj_support = ((support_points - center.reshape(1, 3)) @ axis.reshape(3, 1)).reshape(-1)
        occupancy_bin_mm = float(max(0.5, sample_spacing_mm))
        source_blob_class = str(dict(extra_fields or {}).get("source_blob_class") or "")
        occupancy_proj_vals = proj_support
        if source_blob_class == "complex_blob":
            occupancy_bin_mm = float(
                self._adaptive_projected_support_bin_mm(
                    proj_vals=proj_fit,
                    base_bin_mm=occupancy_bin_mm,
                    max_scale=2.5,
                )
            )
            occupancy_proj_vals = proj_fit
        support_occ = self._axial_occupancy_metrics(occupancy_proj_vals, bin_mm=float(occupancy_bin_mm))
        span_mm = float(max(float(shape.get("span_mm", 0.0)), float(np.linalg.norm(end_ras - start_ras))))
        diameter_mm = float(shape.get("diameter_mm", 0.0))
        reliable_axis = bool(
            axis_reliable
            or (
                fit_points.shape[0] >= 3
                and span_mm >= float(max(3.0, sample_spacing_mm))
                and diameter_mm <= 8.0
            )
        )
        atom = {
            "atom_id": int(atom_id),
            "parent_blob_id": int(parent_blob_id),
            "kind": str(kind),
            "support_points_ras": np.asarray(support_points, dtype=float).reshape(-1, 3).tolist(),
            "center_ras": [float(v) for v in center],
            "axis_ras": [float(v) for v in axis],
            "axis_reliable": bool(reliable_axis),
            "start_ras": [float(v) for v in np.asarray(start_ras, dtype=float).reshape(3)],
            "end_ras": [float(v) for v in np.asarray(end_ras, dtype=float).reshape(3)],
            "span_mm": float(span_mm),
            "diameter_mm": float(diameter_mm),
            "thickness_mm": float(diameter_mm),
            "elongation": float(max(1.0, shape.get("elongation", 1.0))),
            "support_point_count": int(support_points.shape[0]),
            "raw_point_count": int(fit_points.shape[0]),
            "occupancy": {
                "bin_mm": float(occupancy_bin_mm),
                "occupied_bins": int(support_occ.get("occupied_bins", 0)),
                "total_bins": int(support_occ.get("total_bins", 0)),
                "coverage": float(support_occ.get("coverage", 0.0)),
                "max_gap_bins": int(support_occ.get("max_gap_bins", 0)),
                "run_count": int(support_occ.get("run_count", 0)),
                "run_lengths_bins": [int(v) for v in list(support_occ.get("run_lengths_bins") or [])],
                "max_run_bins": int(support_occ.get("max_run_bins", 0)),
            },
            "node_ids": [int(v) for v in list(node_ids or []) if int(v) > 0],
        }
        for key, value in dict(extra_fields or {}).items():
            atom[str(key)] = value
        return atom

    @staticmethod
    def _adaptive_projected_support_bin_mm(proj_vals, base_bin_mm=2.5, max_scale=2.5):
        base_bin = float(max(0.5, base_bin_mm))
        proj = np.sort(np.asarray(proj_vals, dtype=float).reshape(-1))
        if proj.size < 2:
            return float(base_bin)
        diffs = np.diff(proj)
        diffs = np.asarray(diffs[np.isfinite(diffs)], dtype=float).reshape(-1)
        diffs = diffs[diffs > 1e-3]
        if diffs.size == 0:
            return float(base_bin)
        median_gap = float(np.median(diffs))
        if median_gap <= base_bin:
            return float(base_bin)
        return float(min(base_bin * float(max(1.0, max_scale)), 1.10 * median_gap))

    @staticmethod
    def _atom_supports_line(atom, min_coverage=0.90, max_gap_bins=1, min_support_points=3):
        atom_dict = dict(atom or {})
        support_points = np.asarray(atom_dict.get("support_points_ras") or [], dtype=float).reshape(-1, 3)
        if support_points.shape[0] < int(max(2, min_support_points)):
            return False
        span_mm = float(atom_dict.get("span_mm", 0.0))
        occupancy = dict(atom_dict.get("occupancy") or {})
        coverage = float(occupancy.get("coverage", 0.0))
        max_gap = int(occupancy.get("max_gap_bins", 0))
        if coverage >= float(min_coverage) and max_gap <= int(max_gap_bins):
            return True
        source_blob_class = str(atom_dict.get("source_blob_class") or "")
        diameter_mm = float(atom_dict.get("diameter_mm", 0.0))
        if (
            source_blob_class == "complex_blob"
            and span_mm >= 24.0
            and diameter_mm <= 3.5
            and max_gap <= int(max_gap_bins)
            and coverage >= 0.80
        ):
            return True
        return False

    def _complex_blob_axis_dominance_supports_line(
        self,
        blob_points_ras,
        atom,
        axial_bin_mm=2.5,
        guard_diameter_mm=2.4,
        min_dominant_bin_coverage=0.75,
        max_nondominant_run_bins=1,
        slab_nodes=None,
        near_axis_alignment_cos=0.55,
    ):
        pts = np.asarray(blob_points_ras, dtype=float).reshape(-1, 3)
        atom_dict = dict(atom or {})
        if pts.shape[0] == 0:
            return False
        center = np.asarray(atom_dict.get("center_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
        axis = np.asarray(atom_dict.get("axis_ras") or [0.0, 0.0, 1.0], dtype=float).reshape(3)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-6:
            return False
        axis = axis / axis_norm
        start = np.asarray(atom_dict.get("start_ras") or center, dtype=float).reshape(3)
        end = np.asarray(atom_dict.get("end_ras") or center, dtype=float).reshape(3)
        start_proj = float(np.dot(start - center, axis))
        end_proj = float(np.dot(end - center, axis))
        pmin = float(min(start_proj, end_proj))
        pmax = float(max(start_proj, end_proj))
        axial_bin = float(max(0.5, axial_bin_mm))
        source_blob_class = str(atom_dict.get("source_blob_class") or "")
        support_points = np.asarray(atom_dict.get("support_points_ras") or [], dtype=float).reshape(-1, 3)
        if source_blob_class == "complex_blob" and support_points.shape[0] >= 2:
            proj_support = ((support_points - center.reshape(1, 3)) @ axis.reshape(3, 1)).reshape(-1)
            axial_bin = float(
                self._adaptive_projected_support_bin_mm(
                    proj_vals=proj_support,
                    base_bin_mm=axial_bin,
                    max_scale=2.5,
                )
            )
        if (pmax - pmin) < axial_bin:
            return False
        proj = ((pts - center.reshape(1, 3)) @ axis.reshape(3, 1)).reshape(-1)
        radial = np.linalg.norm(
            (pts - center.reshape(1, 3)) - proj.reshape(-1, 1) * axis.reshape(1, 3),
            axis=1,
        )
        span_mask = np.logical_and(proj >= pmin - 0.5 * axial_bin, proj <= pmax + 0.5 * axial_bin)
        if int(np.count_nonzero(span_mask)) == 0:
            return False
        nbins = max(1, int(np.floor((pmax - pmin) / axial_bin)) + 1)
        guard_radius = 0.5 * float(max(0.5, guard_diameter_mm))
        slab_node_list = list(slab_nodes or [])
        if slab_node_list:
            path_node_ids = {int(v) for v in list(atom_dict.get("node_ids") or []) if int(v) > 0}
            bin_entries = {int(bin_id): [] for bin_id in range(nbins)}
            for node in slab_node_list:
                node_dict = dict(node or {})
                node_id = int(node_dict.get("node_id", -1))
                center_val = np.asarray(node_dict.get("center_ras"), dtype=float).reshape(-1)
                if center_val.size != 3:
                    continue
                node_center = center_val.reshape(3)
                proj_val = float(np.dot(node_center - center, axis))
                if proj_val < pmin - 0.5 * axial_bin or proj_val > pmax + 0.5 * axial_bin:
                    continue
                bin_id = int(np.floor((proj_val - pmin) / axial_bin))
                bin_id = int(np.clip(bin_id, 0, nbins - 1))
                radial_val = float(np.linalg.norm((node_center - center) - proj_val * axis))
                node_axis_val = node_dict.get("axis_ras")
                if node_axis_val is None:
                    node_axis_val = axis
                node_axis = np.asarray(node_axis_val, dtype=float).reshape(3)
                node_axis_norm = float(np.linalg.norm(node_axis))
                axis_align = 1.0
                if node_axis_norm > 1e-6:
                    axis_align = abs(float(np.dot(node_axis / node_axis_norm, axis)))
                point_count = int(np.asarray(node_dict.get("points_ras"), dtype=float).reshape(-1, 3).shape[0])
                bin_entries[int(bin_id)].append(
                    {
                        "node_id": int(node_id),
                        "radial_mm": float(radial_val),
                        "mass": int(max(1, point_count)),
                        "axis_align": float(axis_align),
                        "axis_reliable": bool(node_dict.get("axis_reliable")),
                    }
                )
            dominant_flags = []
            nearest_radials = []
            for bin_id in range(nbins):
                entries = list(bin_entries.get(int(bin_id)) or [])
                if not entries:
                    dominant_flags.append(False)
                    continue
                near_entries = []
                for entry in entries:
                    if float(entry.get("radial_mm", 999.0)) > guard_radius:
                        continue
                    if int(entry.get("node_id", -1)) in path_node_ids:
                        near_entries.append(entry)
                        continue
                    if bool(entry.get("axis_reliable")) and float(entry.get("axis_align", 0.0)) < float(near_axis_alignment_cos):
                        continue
                    near_entries.append(entry)
                if not near_entries:
                    dominant_flags.append(False)
                    continue
                dominant_flags.append(True)
                nearest_radials.append(min(float(entry.get("radial_mm", 999.0)) for entry in near_entries))
            dominant_count = int(sum(1 for flag in dominant_flags if bool(flag)))
            dominant_coverage = float(dominant_count / max(1, len(dominant_flags)))
            max_nondominant_run = 0
            current_run = 0
            for flag in dominant_flags:
                if bool(flag):
                    current_run = 0
                else:
                    current_run += 1
                    max_nondominant_run = max(int(max_nondominant_run), int(current_run))
            median_near_radial = float(np.median(np.asarray(nearest_radials, dtype=float))) if nearest_radials else float("inf")
            p90_near_radial = float(np.percentile(np.asarray(nearest_radials, dtype=float), 90.0)) if nearest_radials else float("inf")
            return bool(
                dominant_coverage >= float(min_dominant_bin_coverage)
                and int(max_nondominant_run) <= int(max_nondominant_run_bins)
                and median_near_radial <= float(guard_radius)
                and p90_near_radial <= float(1.35 * guard_radius)
            )
        proj_span = proj[span_mask]
        radial_span = radial[span_mask]
        bin_ids = np.floor((proj_span - pmin) / axial_bin).astype(int)
        bin_ids = np.clip(bin_ids, 0, nbins - 1)
        dominant_flags = []
        for bin_id in range(nbins):
            sel = bin_ids == int(bin_id)
            if not np.any(sel):
                dominant_flags.append(False)
                continue
            inside_count = int(np.count_nonzero(radial_span[sel] <= guard_radius))
            outside_count = int(np.count_nonzero(radial_span[sel] > guard_radius))
            dominant_flags.append(bool(inside_count > 0 and inside_count >= outside_count))
        dominant_count = int(sum(1 for flag in dominant_flags if bool(flag)))
        dominant_coverage = float(dominant_count / max(1, len(dominant_flags)))
        max_nondominant_run = 0
        current_run = 0
        for flag in dominant_flags:
            if bool(flag):
                current_run = 0
            else:
                current_run += 1
                max_nondominant_run = max(int(max_nondominant_run), int(current_run))
        return bool(
            dominant_coverage >= float(min_dominant_bin_coverage)
            and int(max_nondominant_run) <= int(max_nondominant_run_bins)
        )

    @staticmethod
    def _nearest_support_point_ras(ras_pts, target_ras):
        pts = np.asarray(ras_pts, dtype=float).reshape(-1, 3)
        if pts.shape[0] == 0:
            return np.asarray(target_ras, dtype=float).reshape(3)
        target = np.asarray(target_ras, dtype=float).reshape(3)
        d2 = np.sum((pts - target.reshape(1, 3)) ** 2, axis=1)
        return pts[int(np.argmin(d2))]

    @staticmethod
    def _orthonormal_frame_from_axis(axis_ras):
        axis = np.asarray(axis_ras, dtype=float).reshape(3)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-6:
            axis = np.asarray([0.0, 0.0, 1.0], dtype=float)
        else:
            axis = axis / axis_norm
        anchor = np.asarray([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(axis, anchor))) > 0.9:
            anchor = np.asarray([0.0, 1.0, 0.0], dtype=float)
        ortho1 = np.cross(axis, anchor)
        ortho1_norm = float(np.linalg.norm(ortho1))
        if ortho1_norm <= 1e-6:
            ortho1 = np.asarray([0.0, 1.0, 0.0], dtype=float)
            ortho1_norm = float(np.linalg.norm(ortho1))
        ortho1 = ortho1 / max(ortho1_norm, 1e-6)
        ortho2 = np.cross(axis, ortho1)
        ortho2_norm = float(np.linalg.norm(ortho2))
        ortho2 = ortho2 / max(ortho2_norm, 1e-6)
        return np.column_stack((axis, ortho1, ortho2))

    def _point_cloud_shape_metrics(self, pts_ras, seed_axis=None):
        pts = np.asarray(pts_ras, dtype=float).reshape(-1, 3)
        if pts.shape[0] == 0:
            frame = self._orthonormal_frame_from_axis(seed_axis if seed_axis is not None else [0.0, 0.0, 1.0])
            return {
                "center_ras": np.zeros((3,), dtype=float),
                "axis_ras": np.asarray(frame[:, 0], dtype=float).reshape(3),
                "frame_ras": np.asarray(frame, dtype=float),
                "extent_mm": [0.0, 0.0, 0.0],
                "span_mm": 0.0,
                "diameter_mm": 0.0,
                "elongation": 1.0,
                "pca_dominance": 1.0,
            }
        evals_sorted = np.zeros((3,), dtype=float)
        center = np.mean(pts, axis=0)
        centered = pts - center.reshape(1, 3)
        if pts.shape[0] >= 2:
            cov = (centered.T @ centered) / max(1, pts.shape[0] - 1)
            evals, evecs = np.linalg.eigh(cov)
            order = np.argsort(evals)[::-1]
            evals_sorted = np.asarray(evals[order], dtype=float).reshape(3)
            frame = np.asarray(evecs[:, order], dtype=float).reshape(3, 3)
        else:
            frame = self._orthonormal_frame_from_axis(seed_axis if seed_axis is not None else [0.0, 0.0, 1.0])
        axis = np.asarray(frame[:, 0], dtype=float).reshape(3)
        axis_norm = float(np.linalg.norm(axis))
        axis = axis / max(axis_norm, 1e-6)
        if seed_axis is not None:
            seed = np.asarray(seed_axis, dtype=float).reshape(3)
            seed_norm = float(np.linalg.norm(seed))
            if seed_norm > 1e-6 and float(np.dot(axis, seed / seed_norm)) < 0.0:
                axis = -axis
                frame[:, 0] = axis
        frame[:, 1:] = self._orthonormal_frame_from_axis(axis)[:, 1:]
        proj = centered @ frame
        extent_mm = [
            float(np.max(proj[:, idx]) - np.min(proj[:, idx])) if proj.shape[0] > 0 else 0.0
            for idx in range(3)
        ]
        lateral_radius = np.linalg.norm(proj[:, 1:3], axis=1) if proj.shape[0] > 0 else np.zeros((0,), dtype=float)
        diameter_mm = float(2.0 * np.percentile(lateral_radius, 95.0)) if lateral_radius.size > 0 else 0.0
        span_mm = float(extent_mm[0])
        elongation = float(span_mm / max(0.5, diameter_mm))
        pca_dominance = float(evals_sorted[0] / max(1e-6, evals_sorted[1])) if evals_sorted.size >= 2 else 1.0
        return {
            "center_ras": np.asarray(center, dtype=float).reshape(3),
            "axis_ras": np.asarray(axis, dtype=float).reshape(3),
            "frame_ras": np.asarray(frame, dtype=float).reshape(3, 3),
            "extent_mm": [float(v) for v in extent_mm],
            "span_mm": float(span_mm),
            "diameter_mm": float(diameter_mm),
            "elongation": float(max(1.0, elongation)),
            "pca_dominance": float(max(1.0, pca_dominance)),
        }

    @staticmethod
    def _point_cloud_fits_axis_corridor(pts_ras, axis_ras, center_ras, diameter_max_mm, slack_mm=0.25):
        pts = np.asarray(pts_ras, dtype=float).reshape(-1, 3)
        if pts.shape[0] == 0:
            return False
        axis = np.asarray(axis_ras, dtype=float).reshape(3)
        center = np.asarray(center_ras, dtype=float).reshape(3)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-6:
            return False
        axis = axis / axis_norm
        centered = pts - center.reshape(1, 3)
        axial = centered @ axis.reshape(3, 1)
        radial_vec = centered - axial * axis.reshape(1, 3)
        radial_dist = np.linalg.norm(radial_vec, axis=1)
        corridor_radius_mm = 0.5 * float(max(0.0, diameter_max_mm)) + float(max(0.0, slack_mm))
        return bool(np.all(radial_dist <= corridor_radius_mm))

    def _build_blob_cube_nodes(
        self,
        coords_kji,
        ras_pts,
        provisional_axis_ras,
        cube_size_mm=2.0,
        volume_node=None,
        head_distance_map_kji=None,
        annulus_reference_values_hu=None,
    ):
        coords = np.asarray(coords_kji, dtype=int).reshape(-1, 3)
        pts = np.asarray(ras_pts, dtype=float).reshape(-1, 3)
        if coords.shape[0] == 0 or pts.shape[0] == 0:
            return {"nodes": []}
        cube_size = float(max(0.5, cube_size_mm))
        origin = np.min(pts, axis=0)
        cube_coords = np.floor((pts - origin.reshape(1, 3)) / cube_size).astype(int)
        axis = np.asarray(provisional_axis_ras, dtype=float).reshape(3)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-6:
            axis = np.asarray([0.0, 0.0, 1.0], dtype=float)
        else:
            axis = axis / axis_norm
        cube_to_indices = {}
        for idx, cube_coord in enumerate(cube_coords):
            cube_to_indices.setdefault(tuple(int(v) for v in cube_coord.tolist()), []).append(int(idx))
        cube_items = []
        for cube_coord, indices in cube_to_indices.items():
            comp_coords = coords[np.asarray(indices, dtype=int)]
            comp_ras = pts[np.asarray(indices, dtype=int)]
            cube_coord_arr = np.asarray(cube_coord, dtype=float).reshape(3)
            cube_center = origin + (cube_coord_arr + 0.5) * cube_size
            comp_center = np.mean(comp_ras, axis=0)
            support_point = self._nearest_support_point_ras(comp_ras, comp_center)
            proj_val = float(np.dot(comp_center - origin, axis))
            cube_items.append(
                {
                    "cube_coord": tuple(int(v) for v in cube_coord),
                    "coords_kji": np.asarray(comp_coords, dtype=int).reshape(-1, 3),
                    "points_ras": np.asarray(comp_ras, dtype=float).reshape(-1, 3),
                    "center_ras": np.asarray(comp_center, dtype=float).reshape(3),
                    "support_point_ras": np.asarray(support_point, dtype=float).reshape(3),
                    "proj_val": float(proj_val),
                }
            )
        ordered_items = sorted(
            cube_items,
            key=lambda item: (
                float(item.get("proj_val", 0.0)),
                tuple(int(v) for v in item.get("cube_coord", (0, 0, 0))),
            ),
        )
        nodes = []
        for next_node_id, item in enumerate(ordered_items, start=1):
            comp_ras = np.asarray(item.get("points_ras"), dtype=float).reshape(-1, 3)
            comp_shape = self._point_cloud_shape_metrics(comp_ras, seed_axis=axis)
            reliable_axis = bool(
                comp_ras.shape[0] >= 3
                and float(comp_shape.get("span_mm", 0.0)) >= 0.9 * cube_size
                and float(comp_shape.get("diameter_mm", 0.0)) <= max(8.0, 2.0 * cube_size)
            )
            nodes.append(
                {
                    "node_id": int(next_node_id),
                    "bin_id": int(next_node_id - 1),
                    "cube_coord": [int(v) for v in item.get("cube_coord", (0, 0, 0))],
                    "points_ras": np.asarray(item.get("points_ras"), dtype=float).reshape(-1, 3),
                    "coords_kji": np.asarray(item.get("coords_kji"), dtype=int).reshape(-1, 3),
                    "center_ras": np.asarray(item.get("center_ras"), dtype=float).reshape(3),
                    "support_point_ras": np.asarray(item.get("support_point_ras"), dtype=float).reshape(3),
                    "axis_ras": np.asarray(comp_shape.get("axis_ras") if reliable_axis else axis, dtype=float).reshape(3),
                    "axis_reliable": bool(reliable_axis),
                    "shape": dict(comp_shape or {}),
                }
            )
        return {"nodes": nodes}
