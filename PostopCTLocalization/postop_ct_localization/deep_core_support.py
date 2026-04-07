"""Deep-core support extraction: masks, blobs, and proposal atoms."""

try:
    import numpy as np
except ImportError:
    np = None

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

from __main__ import slicer

from shank_core.blob_candidates import build_blob_labelmap, extract_blob_candidates
from shank_core.masking import compute_head_distance_map_kji, largest_component_binary

class DeepCoreSupportLogicMixin:
    def build_deep_core_debug(
        self,
        volume_node,
        hull_threshold_hu=-500.0,
        hull_clip_hu=1200.0,
        hull_sigma_mm=4.0,
        hull_open_vox=7,
        hull_close_vox=0,
        deep_core_shrink_mm=15.0,
        metal_threshold_hu=1900.0,
        metal_grow_vox=1,
        blob_sample_spacing_mm=2.5,
        blob_min_elongation=4.0,
        blob_line_diameter_max_mm=2.0,
        blob_line_min_span_mm=10.0,
        blob_line_min_pca_dominance=6.0,
        blob_contact_diameter_max_mm=10.0,
        blob_complex_cube_size_mm=5.0,
        show_blob_diagnostics=True,
    ):
        if sitk is None:
            raise RuntimeError("SimpleITK is required for deep-core debug preview.")
        arr_kji = np.asarray(slicer.util.arrayFromVolume(volume_node), dtype=np.float32)
        spacing_xyz = tuple(float(v) for v in volume_node.GetSpacing())

        arr_clip = np.asarray(arr_kji, dtype=np.float32)
        if np.isfinite(float(hull_clip_hu)):
            arr_clip = np.minimum(arr_clip, float(hull_clip_hu))
        hull_img = sitk.GetImageFromArray(arr_clip.astype(np.float32))
        hull_img.SetSpacing((float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])))
        if float(hull_sigma_mm) > 0.0:
            hull_img = sitk.SmoothingRecursiveGaussian(hull_img, float(hull_sigma_mm))
        smoothed_hull_kji = sitk.GetArrayFromImage(hull_img).astype(np.float32)

        binary = sitk.BinaryThreshold(
            hull_img,
            lowerThreshold=float(hull_threshold_hu),
            upperThreshold=float(max(float(np.nanmax(smoothed_hull_kji)), float(hull_threshold_hu) + 1.0)),
            insideValue=1,
            outsideValue=0,
        )
        if int(hull_open_vox) > 0:
            binary = sitk.BinaryMorphologicalOpening(binary, [int(hull_open_vox)] * 3, sitk.sitkBall)
        hull_lcc = largest_component_binary(binary)
        if hull_lcc is None:
            raise RuntimeError("Failed to build a non-air hull mask from the selected CT.")
        if int(hull_close_vox) > 0:
            hull_lcc = sitk.BinaryMorphologicalClosing(hull_lcc, [int(hull_close_vox)] * 3, sitk.sitkBall)
        hull_mask_kji = sitk.GetArrayFromImage(hull_lcc).astype(bool)
        head_distance_map_kji = np.asarray(
            compute_head_distance_map_kji(hull_mask_kji, spacing_xyz=spacing_xyz),
            dtype=np.float32,
        )
        deep_core_mask_kji = np.logical_and(
            hull_mask_kji,
            head_distance_map_kji >= float(max(0.0, deep_core_shrink_mm)),
        )

        metal_threshold_hu = float(metal_threshold_hu)
        metal_mask_kji = np.asarray(arr_kji >= float(metal_threshold_hu), dtype=bool)
        metal_grown_mask_kji = metal_mask_kji.copy()
        if int(metal_grow_vox) > 0:
            metal_img = sitk.GetImageFromArray(metal_mask_kji.astype(np.uint8))
            metal_grown_mask_kji = sitk.GetArrayFromImage(
                sitk.BinaryDilate(metal_img, [int(metal_grow_vox)] * 3, sitk.sitkBall)
            ).astype(bool)

        deep_seed_raw_mask_kji = np.logical_and(metal_mask_kji, deep_core_mask_kji)
        deep_seed_mask_kji = np.logical_and(metal_grown_mask_kji, deep_core_mask_kji)

        smooth_node = self._update_scalar_volume_from_array(
            reference_volume_node=volume_node,
            node_name=f"{volume_node.GetName()}_HullSmooth",
            array_kji=smoothed_hull_kji,
        )
        distance_node = self._update_scalar_volume_from_array(
            reference_volume_node=volume_node,
            node_name=f"{volume_node.GetName()}_HeadDistanceMm",
            array_kji=head_distance_map_kji,
        )

        out = {
            "volume_node_id": volume_node.GetID(),
            "smoothed_hull_volume_node": smooth_node,
            "head_distance_volume_node": distance_node,
            "smoothed_hull_kji": smoothed_hull_kji,
            "head_distance_map_kji": head_distance_map_kji,
            "hull_mask_kji": hull_mask_kji,
            "deep_core_mask_kji": deep_core_mask_kji,
            "metal_mask_kji": metal_mask_kji,
            "metal_grown_mask_kji": metal_grown_mask_kji,
            "deep_seed_raw_mask_kji": deep_seed_raw_mask_kji,
            "deep_seed_mask_kji": deep_seed_mask_kji,
            "stats": {
                "hull_voxels": int(np.count_nonzero(hull_mask_kji)),
                "deep_core_voxels": int(np.count_nonzero(deep_core_mask_kji)),
                "metal_voxels": int(np.count_nonzero(metal_mask_kji)),
                "metal_grown_voxels": int(np.count_nonzero(metal_grown_mask_kji)),
                "deep_seed_raw_voxels": int(np.count_nonzero(deep_seed_raw_mask_kji)),
                "deep_seed_voxels": int(np.count_nonzero(deep_seed_mask_kji)),
                "hull_threshold_hu": float(hull_threshold_hu),
                "hull_clip_hu": float(hull_clip_hu),
                "hull_sigma_mm": float(hull_sigma_mm),
                "hull_open_vox": int(hull_open_vox),
                "hull_close_vox": int(hull_close_vox),
                "deep_core_shrink_mm": float(deep_core_shrink_mm),
                "metal_threshold_hu": float(metal_threshold_hu),
                "metal_grow_vox": int(metal_grow_vox),
                "blob_line_diameter_max_mm": float(blob_line_diameter_max_mm),
                "blob_line_min_span_mm": float(blob_line_min_span_mm),
                "blob_line_min_pca_dominance": float(blob_line_min_pca_dominance),
                "blob_contact_diameter_max_mm": float(blob_contact_diameter_max_mm),
                "blob_complex_cube_size_mm": float(blob_complex_cube_size_mm),
            },
        }

        raw_blob_result = extract_blob_candidates(
            metal_mask_kji=deep_seed_raw_mask_kji,
            arr_kji=arr_kji,
            depth_map_kji=head_distance_map_kji,
            ijk_kji_to_ras_fn=lambda idx: self._ijk_kji_to_ras_points(volume_node, idx),
        )
        grown_blob_result = extract_blob_candidates(
            metal_mask_kji=deep_seed_mask_kji,
            arr_kji=arr_kji,
            depth_map_kji=head_distance_map_kji,
            ijk_kji_to_ras_fn=lambda idx: self._ijk_kji_to_ras_points(volume_node, idx),
        )
        sample_payload = self._build_support_atom_payload(
            volume_node=volume_node,
            labels_kji=raw_blob_result.get("labels_kji"),
            blobs=list(raw_blob_result.get("blobs") or []),
            sample_spacing_mm=float(blob_sample_spacing_mm),
            min_elongation=float(blob_min_elongation),
            line_diameter_max_mm=float(blob_line_diameter_max_mm),
            line_min_span_mm=float(blob_line_min_span_mm),
            line_min_pca_dominance=float(blob_line_min_pca_dominance),
            contact_diameter_max_mm=float(blob_contact_diameter_max_mm),
            complex_cube_size_mm=float(blob_complex_cube_size_mm),
        )
        raw_blobs = [dict(blob or {}) for blob in list(raw_blob_result.get("blobs") or [])]
        out["blob_labelmap_kji"] = build_blob_labelmap(raw_blob_result.get("labels_kji")) if bool(show_blob_diagnostics) else None
        out["blob_centroids_all_ras"] = np.asarray(
            [blob.get("centroid_ras") for blob in raw_blobs if blob.get("centroid_ras") is not None],
            dtype=float,
        ).reshape(-1, 3)
        out["support_atoms"] = list(sample_payload.get("support_atoms") or [])
        kept_parent_blob_ids = {
            int(atom.get("parent_blob_id", -1))
            for atom in list(out["support_atoms"] or [])
            if int(dict(atom or {}).get("parent_blob_id", -1)) > 0
        }
        out["blob_centroids_kept_ras"] = np.asarray(
            [
                blob.get("centroid_ras")
                for blob in raw_blobs
                if blob.get("centroid_ras") is not None and int(blob.get("blob_id", -1)) in kept_parent_blob_ids
            ],
            dtype=float,
        ).reshape(-1, 3)
        out["blob_centroids_rejected_ras"] = np.asarray(
            [
                blob.get("centroid_ras")
                for blob in raw_blobs
                if blob.get("centroid_ras") is not None and int(blob.get("blob_id", -1)) not in kept_parent_blob_ids
            ],
            dtype=float,
        ).reshape(-1, 3)
        out["blob_axes_ras_by_id"] = {
            int(k): [float(v) for v in np.asarray(val, dtype=float).reshape(3).tolist()]
            for k, val in dict(sample_payload.get("axes_ras_by_id") or {}).items()
            if int(k) > 0 and np.asarray(val, dtype=float).reshape(-1).size == 3
        }
        out["blob_elongation_by_id"] = {
            int(k): float(v)
            for k, v in dict(sample_payload.get("elongation_by_id") or {}).items()
            if int(k) > 0
        }
        out["blob_parent_blob_ids_by_id"] = {
            int(k): int(v)
            for k, v in dict(sample_payload.get("parent_blob_ids_by_id") or {}).items()
            if int(k) > 0
        }
        out["blob_sample_points_ras"] = np.asarray(sample_payload.get("points_ras"), dtype=float).reshape(-1, 3)
        out["blob_sample_blob_ids"] = np.asarray(sample_payload.get("blob_ids"), dtype=np.int32).reshape(-1)
        out["blob_sample_atom_ids"] = np.asarray(sample_payload.get("atom_ids"), dtype=np.int32).reshape(-1)
        out["complex_blob_chain_rows"] = list(sample_payload.get("complex_blob_chain_rows") or [])
        out["contact_chain_rows"] = list(sample_payload.get("contact_chain_rows") or [])
        out["contact_chain_debug_rows"] = list(sample_payload.get("contact_chain_debug_rows") or [])
        out["line_blob_sample_points_ras"] = np.asarray(sample_payload.get("line_blob_points_ras"), dtype=float).reshape(-1, 3)
        out["contact_blob_sample_points_ras"] = np.asarray(sample_payload.get("contact_blob_points_ras"), dtype=float).reshape(-1, 3)
        out["complex_blob_sample_points_ras"] = np.asarray(sample_payload.get("complex_blob_points_ras"), dtype=float).reshape(-1, 3)
        out["blob_class_by_id"] = dict(sample_payload.get("blob_class_by_id") or {})
        out["stats"]["deep_seed_sampled_blob_count"] = int(len(dict(out["blob_elongation_by_id"] or {})))
        out["stats"]["deep_seed_atom_count"] = int(len(list(out["support_atoms"] or [])))
        out["stats"]["deep_seed_raw_blob_count"] = int(raw_blob_result.get("blob_count_total", 0))
        out["stats"]["deep_seed_grown_blob_count"] = int(grown_blob_result.get("blob_count_total", 0))
        out["stats"]["deep_seed_sample_count"] = int(np.asarray(out["blob_sample_points_ras"]).reshape(-1, 3).shape[0])
        out["stats"]["deep_seed_line_blob_token_count"] = int(np.asarray(out["line_blob_sample_points_ras"]).reshape(-1, 3).shape[0])
        out["stats"]["deep_seed_contact_blob_token_count"] = int(np.asarray(out["contact_blob_sample_points_ras"]).reshape(-1, 3).shape[0])
        out["stats"]["deep_seed_complex_blob_token_count"] = int(np.asarray(out["complex_blob_sample_points_ras"]).reshape(-1, 3).shape[0])
        out["stats"]["deep_seed_complex_blob_chain_row_count"] = int(len(list(out["complex_blob_chain_rows"] or [])))
        out["stats"]["deep_seed_contact_chain_row_count"] = int(len(list(out["contact_chain_rows"] or [])))
        out["stats"]["deep_seed_contact_chain_debug_row_count"] = int(len(list(out["contact_chain_debug_rows"] or [])))
        return out

    def _build_support_atom_payload(
        self,
        volume_node,
        labels_kji,
        blobs,
        sample_spacing_mm=2.5,
        min_elongation=4.0,
        line_diameter_max_mm=2.0,
        line_min_span_mm=10.0,
        line_min_pca_dominance=6.0,
        contact_diameter_max_mm=10.0,
        complex_cube_size_mm=5.0,
    ):
        labels = np.asarray(labels_kji, dtype=np.int32)
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
        spacing = float(max(0.5, sample_spacing_mm))
        min_elong = float(max(1.0, min_elongation))
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
            slab_nodes = self._build_blob_slab_nodes(
                coords_kji=coords_kji,
                ras_pts=ras_pts,
                center_ras=shape_center,
                provisional_axis_ras=shape_axis,
                sample_spacing_mm=spacing,
            )
            max_components_per_bin = int(max((int(v) for v in dict(slab_nodes.get("bin_counts") or {}).values()), default=0))
            blob_class = self._classify_support_blob(
                max_extent_mm=max_extent_mm,
                span_mm=span_mm,
                diameter_mm=diameter_mm,
                elongation=elongation,
                pca_dominance=pca_dominance,
                max_components_per_bin=max_components_per_bin,
                min_elongation=min_elong,
                line_diameter_max_mm=float(line_diameter_max_mm),
                line_min_span_mm=float(line_min_span_mm),
                line_min_pca_dominance=float(line_min_pca_dominance),
                contact_diameter_max_mm=float(contact_diameter_max_mm),
            )
            blob_class_by_id[int(raw_blob_id)] = str(blob_class)
            if blob_class == "contact_blob":
                blob_tokens = self._nearest_support_point_ras(ras_pts, centroid).reshape(1, 3)
            elif blob_class == "line_blob":
                blob_tokens = self._sample_line_blob_tokens(
                    coords_kji=coords_kji,
                    ras_pts=ras_pts,
                    center_ras=shape_center,
                    axis_ras=shape_axis,
                    sample_spacing_mm=spacing,
                )
                if blob_tokens.shape[0] == 0:
                    blob_tokens = self._nearest_support_point_ras(ras_pts, centroid).reshape(1, 3)
            else:
                cube_nodes = self._build_blob_cube_nodes(
                    coords_kji=coords_kji,
                    ras_pts=ras_pts,
                    provisional_axis_ras=shape_axis,
                    cube_size_mm=float(max(1.0, complex_cube_size_mm)),
                )
                blob_tokens = np.asarray(
                    [node.get("support_point_ras") for node in list(cube_nodes.get("nodes") or [])],
                    dtype=float,
                ).reshape(-1, 3)
                if blob_tokens.shape[0] == 0:
                    blob_tokens = self._nearest_support_point_ras(ras_pts, centroid).reshape(1, 3)

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
                min_coverage=0.90,
                max_gap_bins=1,
                parent_blob_id=int(raw_blob_id),
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
                if not self._atom_supports_line(atom, min_coverage=0.90, max_gap_bins=1, min_support_points=3):
                    continue
                if not self._complex_blob_axis_dominance_supports_line(
                    blob_points_ras=ras_pts,
                    atom=atom,
                    axial_bin_mm=spacing,
                    guard_diameter_mm=float(max(1.0, 1.2 * line_diameter_max_mm)),
                    min_dominant_bin_coverage=0.75,
                    max_nondominant_run_bins=1,
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
            if seed_info is None:
                debug_rows.append(
                    {
                        "seed_atom_i": int(atom_i),
                        "seed_atom_j": int(atom_j),
                        "stage": "seed",
                        "status": "rejected",
                        "reason": "insufficient_collinear_support",
                        "seed_score": "",
                        "chain_len": 0,
                        "chain_atom_ids": "",
                        "chain_id": "",
                    }
                )
                continue
            seed_pairs.append((float(seed_info.get("score", 0.0)), int(atom_i), int(atom_j)))
        seed_pairs.sort(key=lambda item: (-float(item[0]), int(min(item[1], item[2])), int(max(item[1], item[2]))))
        candidates = []
        for _seed_score, atom_i, atom_j in seed_pairs:
            chain_ids = self._grow_contact_chain_from_pair(
                center_map=center_map,
                adj=adj,
                start_atom_i=int(atom_i),
                start_atom_j=int(atom_j),
            )
            chain_segments = self._split_contact_chain_on_large_gaps(
                center_map=center_map,
                atom_ids=chain_ids,
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
            return None
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
            return None
        score = (
            4.0 * float(extra_support)
            + 1.5 * float(support_score)
            - 0.15 * float(dist_mm)
        )
        return {"score": float(score)}

    def _grow_contact_chain_from_pair(
        self,
        center_map,
        adj,
        start_atom_i,
        start_atom_j,
    ):
        chain = [int(start_atom_i), int(start_atom_j)]
        chain_set = {int(start_atom_i), int(start_atom_j)}
        cos_forward = float(np.cos(np.deg2rad(60.0)))
        cos_recent = float(np.cos(np.deg2rad(70.0)))
        # Contact chaining should stay local and should not adapt its jump
        # allowance based on whatever contacts already entered the chain.
        max_step_mm = 10.0
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
                for edge in list(adj.get(int(current_id)) or []):
                    nb_id = int(edge.get("neighbor_id", -1))
                    if nb_id <= 0 or nb_id in chain_set:
                        continue
                    nb_center = np.asarray(center_map.get(int(nb_id)), dtype=float).reshape(3)
                    step = nb_center - current_center
                    step_norm = float(np.linalg.norm(step))
                    if step_norm <= 1e-6:
                        continue
                    if step_norm > float(max_step_mm):
                        continue
                    step_dir = step / step_norm
                    align_fit = float(np.dot(step_dir, fit_axis))
                    if align_fit < float(cos_forward):
                        continue
                    align_recent = float(np.dot(step_dir, travel))
                    if align_recent < float(cos_recent):
                        continue
                    lateral = float(self._radial_distance_to_line(nb_center.reshape(1, 3), fit_center, fit_axis)[0])
                    if lateral > 4.5:
                        continue
                    if grow_forward:
                        trial_centers = np.vstack([chain_centers, nb_center.reshape(1, 3)])
                    else:
                        trial_centers = np.vstack([nb_center.reshape(1, 3), chain_centers])
                    trial_center, trial_axis = self._fit_line_pca(trial_centers, seed_axis=fit_axis)
                    trial_radial = self._radial_distance_to_line(trial_centers, trial_center, trial_axis)
                    trial_rms = float(np.sqrt(np.mean(np.square(trial_radial)))) if trial_radial.size > 0 else 0.0
                    if trial_rms > 2.0:
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
                    break
                if grow_forward:
                    chain.append(int(best_neighbor))
                else:
                    chain.insert(0, int(best_neighbor))
                chain_set.add(int(best_neighbor))
        return [int(v) for v in chain]

    def _split_contact_chain_on_large_gaps(
        self,
        center_map,
        atom_ids,
    ):
        split_gap_mm = 10.0
        chain_ids = [int(v) for v in list(atom_ids or []) if int(v) > 0]
        if len(chain_ids) < 3:
            return [chain_ids] if chain_ids else []
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
            return [sorted_ids]
        segments = []
        start_idx = 0
        for gap_idx, gap_mm in enumerate(gap_values.tolist()):
            if float(gap_mm) > float(split_gap_mm) or int(gap_idx) in turn_breaks:
                segment = [int(v) for v in sorted_ids[start_idx : gap_idx + 1]]
                if len(segment) >= 3:
                    segments.append(segment)
                start_idx = int(gap_idx + 1)
        tail = [int(v) for v in sorted_ids[start_idx:]]
        if len(tail) >= 3:
            segments.append(tail)
        if not segments and len(sorted_ids) >= 3:
            segments.append([int(v) for v in sorted_ids])
        return segments

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
        # Sparse contact chains can have mild curvature or "kissing" endpoints,
        # so accept slightly larger global line RMS when the local turns remain smooth.
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

    def _sample_blob_points_from_components(
        self,
        volume_node,
        labels_kji,
        blobs,
        sample_spacing_mm=2.5,
        min_elongation=4.0,
    ):
        return self._build_support_atom_payload(
            volume_node=volume_node,
            labels_kji=labels_kji,
            blobs=blobs,
            sample_spacing_mm=sample_spacing_mm,
            min_elongation=min_elongation,
        )

    def _append_support_atom(
        self,
        support_atoms,
        next_atom_id,
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
        atom = self._create_support_atom(
            atom_id=int(next_atom_id),
            parent_blob_id=int(parent_blob_id),
            kind=kind,
            support_points_ras=support_points_ras,
            fit_points_ras=fit_points_ras,
            axis_ras=axis_ras,
            center_ras=center_ras,
            axis_reliable=axis_reliable,
            sample_spacing_mm=sample_spacing_mm,
            node_ids=node_ids,
            extra_fields=extra_fields,
        )
        if atom is None:
            return int(next_atom_id)
        support_atoms.append(atom)
        return int(next_atom_id) + 1

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
        if source_blob_class == "complex_blob":
            occupancy_bin_mm = float(
                self._adaptive_projected_support_bin_mm(
                    proj_vals=proj_support,
                    base_bin_mm=occupancy_bin_mm,
                    max_scale=2.5,
                )
            )
        support_occ = self._axial_occupancy_metrics(proj_support, bin_mm=float(occupancy_bin_mm))
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
    def _classify_support_blob(
        max_extent_mm,
        span_mm,
        diameter_mm,
        elongation,
        pca_dominance,
        max_components_per_bin,
        min_elongation=4.0,
        line_diameter_max_mm=2.0,
        line_min_span_mm=10.0,
        line_min_pca_dominance=6.0,
        contact_diameter_max_mm=10.0,
    ):
        if (
            float(max_extent_mm) <= float(contact_diameter_max_mm)
            and float(diameter_mm) <= float(contact_diameter_max_mm)
        ):
            return "contact_blob"
        if (
            float(span_mm) >= float(line_min_span_mm)
            and
            float(elongation) >= float(max(1.0, min_elongation))
            and float(pca_dominance) >= float(max(1.0, line_min_pca_dominance))
            and float(diameter_mm) <= float(line_diameter_max_mm)
            and int(max_components_per_bin) <= 1
        ):
            return "line_blob"
        return "complex_blob"

    @staticmethod
    def _atom_supports_line(atom, min_coverage=0.90, max_gap_bins=1, min_support_points=3):
        atom_dict = dict(atom or {})
        support_points = np.asarray(atom_dict.get("support_points_ras") or [], dtype=float).reshape(-1, 3)
        if support_points.shape[0] < int(max(2, min_support_points)):
            return False
        span_mm = float(atom_dict.get("span_mm", 0.0))
        if span_mm < 6.0:
            return False
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

    @staticmethod
    def _complex_blob_axis_dominance_supports_line(
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
                DeepCoreSupportLogicMixin._adaptive_projected_support_bin_mm(
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
            path_node_ids = {
                int(v)
                for v in list(atom_dict.get("node_ids") or [])
                if int(v) > 0
            }
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
    def _split_points_into_local_components(coords_kji):
        coords = np.asarray(coords_kji, dtype=int).reshape(-1, 3)
        if coords.shape[0] == 0:
            return []
        if coords.shape[0] == 1 or sitk is None:
            return [np.ones((coords.shape[0],), dtype=bool)]
        mins = np.min(coords, axis=0)
        maxs = np.max(coords, axis=0)
        shape = tuple(int(maxs[d] - mins[d] + 1) for d in range(3))
        local = np.zeros(shape, dtype=np.uint8)
        shifted = coords - mins.reshape(1, 3)
        local[shifted[:, 0], shifted[:, 1], shifted[:, 2]] = 1
        cc_img = sitk.ConnectedComponent(sitk.GetImageFromArray(local), True)
        cc = sitk.GetArrayFromImage(cc_img).astype(np.int32)
        labels = cc[shifted[:, 0], shifted[:, 1], shifted[:, 2]]
        out = []
        for label_id in np.unique(labels):
            if int(label_id) <= 0:
                continue
            out.append(labels == int(label_id))
        return out or [np.ones((coords.shape[0],), dtype=bool)]

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

    def _sample_line_blob_tokens(
        self,
        coords_kji,
        ras_pts,
        center_ras,
        axis_ras,
        sample_spacing_mm=2.5,
    ):
        coords = np.asarray(coords_kji, dtype=int).reshape(-1, 3)
        pts = np.asarray(ras_pts, dtype=float).reshape(-1, 3)
        if coords.shape[0] == 0 or pts.shape[0] == 0:
            return np.empty((0, 3), dtype=float)
        center = np.asarray(center_ras, dtype=float).reshape(3)
        axis = np.asarray(axis_ras, dtype=float).reshape(3)
        axis_norm = float(np.linalg.norm(axis))
        axis = axis / max(axis_norm, 1e-6)
        spacing = float(max(0.5, sample_spacing_mm))
        proj = ((pts - center.reshape(1, 3)) @ axis.reshape(3, 1)).reshape(-1)
        pmin = float(np.min(proj))
        pmax = float(np.max(proj))
        if not np.isfinite(pmin) or not np.isfinite(pmax):
            return np.empty((0, 3), dtype=float)
        if (pmax - pmin) < spacing * 1.25:
            return self._nearest_support_point_ras(pts, center).reshape(1, 3)
        nbins = max(1, int(np.floor((pmax - pmin) / spacing)) + 1)
        bin_ids = np.floor((proj - pmin) / spacing).astype(int)
        bin_ids = np.clip(bin_ids, 0, nbins - 1)
        out = []
        for bin_id in range(nbins):
            sel = bin_ids == bin_id
            if not np.any(sel):
                continue
            slab_coords_kji = coords[sel]
            slab_ras_pts = pts[sel]
            for slab_component_mask in self._split_points_into_local_components(slab_coords_kji):
                if not np.any(slab_component_mask):
                    continue
                comp_ras = slab_ras_pts[slab_component_mask]
                comp_center = np.mean(comp_ras, axis=0)
                out.append(self._nearest_support_point_ras(comp_ras, comp_center))
        if not out:
            return np.empty((0, 3), dtype=float)
        return np.asarray(out, dtype=float).reshape(-1, 3)

    def _build_blob_slab_nodes(
        self,
        coords_kji,
        ras_pts,
        center_ras,
        provisional_axis_ras,
        sample_spacing_mm=2.5,
    ):
        coords = np.asarray(coords_kji, dtype=int).reshape(-1, 3)
        pts = np.asarray(ras_pts, dtype=float).reshape(-1, 3)
        if coords.shape[0] == 0 or pts.shape[0] == 0:
            return {"nodes": [], "bin_counts": {}}
        center = np.asarray(center_ras, dtype=float).reshape(3)
        axis = np.asarray(provisional_axis_ras, dtype=float).reshape(3)
        axis_norm = float(np.linalg.norm(axis))
        axis = axis / max(axis_norm, 1e-6)
        spacing = float(max(0.5, sample_spacing_mm))
        proj = ((pts - center.reshape(1, 3)) @ axis.reshape(3, 1)).reshape(-1)
        pmin = float(np.min(proj))
        pmax = float(np.max(proj))
        if not np.isfinite(pmin) or not np.isfinite(pmax):
            return {"nodes": [], "bin_counts": {}}
        nbins = max(1, int(np.floor((pmax - pmin) / spacing)) + 1)
        bin_ids = np.floor((proj - pmin) / spacing).astype(int)
        bin_ids = np.clip(bin_ids, 0, nbins - 1)
        nodes = []
        bin_counts = {}
        next_node_id = 1
        for bin_id in range(nbins):
            sel = bin_ids == bin_id
            if not np.any(sel):
                continue
            slab_coords_kji = coords[sel]
            slab_ras_pts = pts[sel]
            comp_count = 0
            for slab_component_mask in self._split_points_into_local_components(slab_coords_kji):
                if not np.any(slab_component_mask):
                    continue
                comp_count += 1
                comp_coords = slab_coords_kji[slab_component_mask]
                comp_ras = slab_ras_pts[slab_component_mask]
                comp_shape = self._point_cloud_shape_metrics(comp_ras, seed_axis=axis)
                comp_center = np.asarray(comp_shape.get("center_ras"), dtype=float).reshape(3)
                support_point = self._nearest_support_point_ras(comp_ras, comp_center)
                reliable_axis = bool(
                    comp_ras.shape[0] >= 3
                    and float(comp_shape.get("span_mm", 0.0)) >= 2.0
                    and float(comp_shape.get("diameter_mm", 0.0)) <= 8.0
                )
                nodes.append(
                    {
                        "node_id": int(next_node_id),
                        "bin_id": int(bin_id),
                        "points_ras": np.asarray(comp_ras, dtype=float).reshape(-1, 3),
                        "coords_kji": np.asarray(comp_coords, dtype=int).reshape(-1, 3),
                        "center_ras": np.asarray(comp_center, dtype=float).reshape(3),
                        "support_point_ras": np.asarray(support_point, dtype=float).reshape(3),
                        "axis_ras": np.asarray(comp_shape.get("axis_ras") if reliable_axis else axis, dtype=float).reshape(3),
                        "axis_reliable": bool(reliable_axis),
                        "shape": dict(comp_shape or {}),
                    }
                )
                next_node_id += 1
            if comp_count > 0:
                bin_counts[int(bin_id)] = int(comp_count)
        return {"nodes": nodes, "bin_counts": bin_counts}

    def _build_blob_cube_nodes(
        self,
        coords_kji,
        ras_pts,
        provisional_axis_ras,
        cube_size_mm=2.0,
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

    def _classify_complex_blob_nodes(
        self,
        nodes,
    ):
        graph = self._build_complex_blob_cube_adjacency(nodes)
        node_map = dict(graph.get("node_map") or {})
        adj = dict(graph.get("adj") or {})
        node_degrees = dict(graph.get("node_degrees") or {})
        opposite_cos = float(np.cos(np.deg2rad(40.0)))
        shell_min = 2.5
        shell_max = 3.5
        cube_items = []
        for node_id, node in node_map.items():
            cube_coord = np.asarray(dict(node or {}).get("cube_coord"), dtype=int).reshape(3)
            center = np.asarray(dict(node or {}).get("center_ras"), dtype=float).reshape(3)
            cube_items.append((int(node_id), cube_coord, center))
        for node_id, node in node_map.items():
            current_center = np.asarray(dict(node or {}).get("center_ras"), dtype=float).reshape(3)
            current_cube = np.asarray(dict(node or {}).get("cube_coord"), dtype=int).reshape(3)
            shell_neighbor_infos = []
            for other_id, other_cube, other_center in cube_items:
                if int(other_id) == int(node_id):
                    continue
                cube_delta = np.asarray(other_cube - current_cube, dtype=float).reshape(3)
                cube_dist = float(np.linalg.norm(cube_delta))
                if cube_dist < float(shell_min) or cube_dist > float(shell_max):
                    continue
                step = np.asarray(other_center - current_center, dtype=float).reshape(3)
                step_norm = float(np.linalg.norm(step))
                if step_norm <= 1e-6:
                    continue
                shell_neighbor_infos.append(
                    {
                        "neighbor_id": int(other_id),
                        "score": float(1.0 / max(cube_dist, 1e-6)),
                        "dist_mm": float(step_norm),
                        "direction_ras": np.asarray(step / step_norm, dtype=float).reshape(3),
                    }
                )
            neighbor_infos = sorted(
                list(shell_neighbor_infos or []),
                key=lambda info: (float(info.get("dist_mm", 999.0)), -float(info.get("score", 0.0))),
            )
            if not neighbor_infos:
                neighbor_infos = sorted(
                    list(adj.get(int(node_id)) or []),
                    key=lambda info: float(info.get("dist_mm", 999.0)),
                )
            branch_groups = self._cluster_complex_blob_neighbor_directions(neighbor_infos)
            node["branch_groups"] = list(branch_groups)
            node["branch_count"] = int(len(branch_groups))
            primary_axis = None
            axis_strength = 0.0
            if len(branch_groups) == 1:
                dir0_val = dict(branch_groups[0]).get("direction_ras")
                if dir0_val is None:
                    dir0_val = [0.0, 0.0, 1.0]
                dir0 = np.asarray(dir0_val, dtype=float).reshape(3)
                dir0 = dir0 / max(float(np.linalg.norm(dir0)), 1e-6)
                node_role = "edge"
                primary_axis = np.asarray(dir0, dtype=float).reshape(3)
                axis_strength = 1.0
            elif len(branch_groups) == 2:
                dir0_val = dict(branch_groups[0]).get("direction_ras")
                if dir0_val is None:
                    dir0_val = [0.0, 0.0, 1.0]
                dir1_val = dict(branch_groups[1]).get("direction_ras")
                if dir1_val is None:
                    dir1_val = [0.0, 0.0, 1.0]
                dir0 = np.asarray(dir0_val, dtype=float).reshape(3)
                dir1 = np.asarray(dir1_val, dtype=float).reshape(3)
                dir0 = dir0 / max(float(np.linalg.norm(dir0)), 1e-6)
                dir1 = dir1 / max(float(np.linalg.norm(dir1)), 1e-6)
                dot_val = float(np.dot(dir0, dir1))
                if dot_val <= -opposite_cos:
                    axis = dir0 - dir1
                    axis_norm = float(np.linalg.norm(axis))
                    if axis_norm > 1e-6:
                        primary_axis = np.asarray(axis / axis_norm, dtype=float).reshape(3)
                    else:
                        primary_axis = np.asarray(dir0, dtype=float).reshape(3)
                    node_role = "core"
                    axis_strength = float(-dot_val)
                else:
                    node_role = "crossing"
            elif len(branch_groups) <= 0:
                node_role = "edge"
            else:
                node_role = "crossing"
            node["node_role"] = str(node_role)
            node["primary_axis_ras"] = None if primary_axis is None else np.asarray(primary_axis, dtype=float).reshape(3)
            node["axis_strength"] = float(axis_strength)
            node["degree"] = int(node_degrees.get(int(node_id), 0))
        return graph

    def _extract_complex_blob_paths(
        self,
        slab_nodes,
        provisional_axis_ras,
        sample_spacing_mm=2.5,
        min_coverage=0.90,
        max_gap_bins=1,
        candidate_guard_diameter_mm=4.0,
        parent_blob_id=None,
        return_debug=False,
    ):
        nodes = list(slab_nodes or [])
        if len(nodes) < 2:
            return {"paths": [], "chain_rows": []} if bool(return_debug) else []
        spacing = float(max(0.5, sample_spacing_mm))
        graph = self._build_complex_blob_token_graph(
            slab_nodes=nodes,
            provisional_axis_ras=provisional_axis_ras,
            sample_spacing_mm=spacing,
        )
        node_map = dict(graph.get("node_map") or {})
        adj = dict(graph.get("adj") or {})
        accepted = self._build_complex_blob_route_paths(
            node_map=node_map,
            adj=adj,
            max_route_count=6,
            max_seed_count=24,
            max_steps=max(20, int(2 * len(node_map) + 2)),
            sample_spacing_mm=spacing,
            min_coverage=min_coverage,
            max_gap_bins=max_gap_bins,
        )
        if not bool(return_debug):
            return accepted
        return {
            "paths": list(accepted),
            "chain_rows": self._build_complex_blob_chain_rows(
                parent_blob_id=int(parent_blob_id) if parent_blob_id is not None else -1,
                graph=graph,
                accepted_paths=accepted,
            ),
        }

    def _complex_blob_seed_direction_hypotheses(
        self,
        node,
    ):
        node_dict = dict(node or {})
        role = str(node_dict.get("node_role") or "")
        branch_groups = [dict(group or {}) for group in list(node_dict.get("branch_groups") or [])]
        dirs = []
        if role == "edge":
            for group in branch_groups:
                dir_val = group.get("direction_ras")
                if dir_val is None:
                    continue
                direction = np.asarray(dir_val, dtype=float).reshape(3)
                direction_norm = float(np.linalg.norm(direction))
                if direction_norm <= 1e-6:
                    continue
                dirs.append(direction / direction_norm)
        elif role == "core":
            axis_val = node_dict.get("primary_axis_ras")
            if axis_val is not None:
                axis = np.asarray(axis_val, dtype=float).reshape(3)
                axis_norm = float(np.linalg.norm(axis))
                if axis_norm > 1e-6:
                    axis = axis / axis_norm
                    dirs.extend([axis, -axis])
        elif role == "crossing":
            for group in branch_groups:
                dir_val = group.get("direction_ras")
                if dir_val is None:
                    continue
                direction = np.asarray(dir_val, dtype=float).reshape(3)
                direction_norm = float(np.linalg.norm(direction))
                if direction_norm <= 1e-6:
                    continue
                dirs.append(direction / direction_norm)
        dedup = []
        for direction in dirs:
            keep = True
            for prev in dedup:
                if abs(float(np.dot(direction, prev))) >= 0.98:
                    keep = False
                    break
            if keep:
                dedup.append(np.asarray(direction, dtype=float).reshape(3))
        return [np.asarray(v, dtype=float).reshape(3) for v in dedup]

    def _build_complex_blob_route_paths(
        self,
        node_map,
        adj,
        max_route_count=6,
        max_seed_count=24,
        max_steps=24,
        sample_spacing_mm=2.5,
        min_coverage=0.90,
        max_gap_bins=1,
    ):
        blocked_node_ids = set()
        accepted = []
        for _iter_idx in range(int(max_route_count)):
            seeds = []
            for node_id, node in node_map.items():
                node_id = int(node_id)
                if node_id in blocked_node_ids:
                    continue
                node_dict = dict(node or {})
                role = str(node_dict.get("node_role") or "")
                directions = self._complex_blob_seed_direction_hypotheses(node_dict)
                if not directions:
                    continue
                role_priority = 0 if role == "edge" else (1 if role == "core" else 2)
                axis_strength = float(node_dict.get("axis_strength", 0.0))
                degree = int(len(list(adj.get(int(node_id)) or [])))
                for dir_idx, direction in enumerate(directions):
                    seeds.append(
                        (
                            int(role_priority),
                            -float(axis_strength),
                            int(degree),
                            int(node_id),
                            int(dir_idx),
                            np.asarray(direction, dtype=float).reshape(3),
                        )
                    )
            seeds.sort(key=lambda item: (int(item[0]), float(item[1]), int(item[2]), int(item[3]), int(item[4])))
            best_candidate = None
            for role_priority, _neg_axis_strength, _degree, seed_id, _dir_idx, direction in seeds[: max(1, int(max_seed_count))]:
                seed_node = dict(node_map.get(int(seed_id)) or {})
                role = str(seed_node.get("node_role") or "")
                if role == "edge":
                    merged = self._trace_complex_blob_directional_path(
                        node_map=node_map,
                        adj=adj,
                        start_id=int(seed_id),
                        initial_direction_ras=direction,
                        max_steps=int(max_steps),
                        blocked_node_ids=blocked_node_ids,
                    )
                else:
                    forward = self._trace_complex_blob_directional_path(
                        node_map=node_map,
                        adj=adj,
                        start_id=int(seed_id),
                        initial_direction_ras=direction,
                        max_steps=int(max_steps),
                        blocked_node_ids=blocked_node_ids,
                    )
                    backward = self._trace_complex_blob_directional_path(
                        node_map=node_map,
                        adj=adj,
                        start_id=int(seed_id),
                        initial_direction_ras=-np.asarray(direction, dtype=float).reshape(3),
                        max_steps=int(max_steps),
                        blocked_node_ids=blocked_node_ids,
                    )
                    merged = [int(v) for v in list(reversed(backward[1:]))] + [int(seed_id)] + [int(v) for v in list(forward[1:])]
                    merged = [int(v) for idx, v in enumerate(merged) if idx == 0 or int(v) != int(merged[idx - 1])]
                if len(merged) < 3:
                    continue
                merged = self._extend_complex_blob_path_to_exhaustion(
                    node_map=node_map,
                    adj=adj,
                    node_ids=merged,
                    blocked_node_ids=blocked_node_ids,
                    max_extension_steps=max(8, int(max_steps)),
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
                path_info["seed_role"] = str(role)
                if best_candidate is None or float(path_info.get("score", 0.0)) > float(best_candidate.get("score", 0.0)):
                    best_candidate = dict(path_info)
            if best_candidate is None:
                break
            if any(self._complex_blob_paths_overlap(best_candidate, prev) for prev in accepted):
                break
            accepted.append(dict(best_candidate))
            for node_id in list(best_candidate.get("node_ids") or []):
                node_id = int(node_id)
                if node_id <= 0:
                    continue
                node = dict(node_map.get(int(node_id)) or {})
                if str(node.get("node_role") or "") == "crossing":
                    continue
                blocked_node_ids.add(int(node_id))
        return list(accepted)

    def _build_complex_blob_chain_rows(
        self,
        parent_blob_id,
        graph,
        accepted_paths,
    ):
        node_map = dict(dict(graph or {}).get("node_map") or {})
        node_degrees = dict(dict(graph or {}).get("node_degrees") or {})
        if not node_map:
            return []
        memberships = {}
        for line_id, path in enumerate(list(accepted_paths or []), start=1):
            node_ids = [int(v) for v in list(dict(path or {}).get("node_ids") or []) if int(v) > 0]
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
                rows.append(
                    {
                        "blob_id": int(parent_blob_id),
                        "node_id": int(node_id),
                        "node_role": str(node_role),
                        "line_id": int(line_id),
                        "line_order": int(line_order),
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

    def _build_complex_blob_iterative_seed_paths(
        self,
        node_map,
        adj,
        max_seed_count=12,
        max_steps=10,
        sample_spacing_mm=2.5,
        min_coverage=0.90,
        max_gap_bins=1,
    ):
        blocked_node_ids = set()
        out = []
        for _iter_idx in range(8):
            seeds = []
            for node_id, node in node_map.items():
                if int(node_id) in blocked_node_ids:
                    continue
                node_dict = dict(node or {})
                if str(node_dict.get("node_role") or "") != "core":
                    continue
                axis_val = node_dict.get("primary_axis_ras")
                if axis_val is None:
                    continue
                axis = np.asarray(axis_val, dtype=float).reshape(3)
                axis_norm = float(np.linalg.norm(axis))
                if axis_norm <= 1e-6:
                    continue
                seeds.append(
                    (
                        -float(node_dict.get("seed_score", 0.0)),
                        -float(node_dict.get("axis_strength", 0.0)),
                        int(node_id),
                    )
                )
            seeds.sort()
            accepted_this_round = False
            for _neg_seed_score, _neg_axis_strength, seed_id in seeds[: max(1, int(max_seed_count))]:
                if int(seed_id) in blocked_node_ids:
                    continue
                seed_node = dict(node_map.get(int(seed_id)) or {})
                axis = np.asarray(seed_node.get("primary_axis_ras"), dtype=float).reshape(3)
                axis_norm = float(np.linalg.norm(axis))
                if axis_norm <= 1e-6:
                    continue
                axis = axis / axis_norm
                forward = self._trace_complex_blob_directional_path(
                    node_map=node_map,
                    adj=adj,
                    start_id=int(seed_id),
                    initial_direction_ras=axis,
                    max_steps=int(max_steps),
                    blocked_node_ids=None,
                )
                backward = self._trace_complex_blob_directional_path(
                    node_map=node_map,
                    adj=adj,
                    start_id=int(seed_id),
                    initial_direction_ras=-axis,
                    max_steps=int(max_steps),
                    blocked_node_ids=None,
                )
                merged = [int(v) for v in list(reversed(backward[1:]))] + [int(seed_id)] + [int(v) for v in list(forward[1:])]
                merged = [int(v) for idx, v in enumerate(merged) if idx == 0 or int(v) != int(merged[idx - 1])]
                if len(merged) < 3:
                    continue
                merged = self._extend_complex_blob_path_to_exhaustion(
                    node_map=node_map,
                    adj=adj,
                    node_ids=merged,
                    blocked_node_ids=None,
                    max_extension_steps=max(8, int(max_steps)),
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
                out.append(path_info)
                for node_id in list(path_info.get("node_ids") or []):
                    node_id = int(node_id)
                    if node_id <= 0:
                        continue
                    node = dict(node_map.get(int(node_id)) or {})
                    if str(node.get("node_role") or "") == "crossing":
                        continue
                    blocked_node_ids.add(int(node_id))
                accepted_this_round = True
                break
            if not accepted_this_round:
                break
        return out

    def _extend_complex_blob_path_to_exhaustion(
        self,
        node_map,
        adj,
        node_ids,
        blocked_node_ids=None,
        max_extension_steps=12,
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

    def _build_complex_blob_pair_axis_paths(
        self,
        node_map,
        adj,
        anchor_node_ids,
        sample_spacing_mm=2.5,
        candidate_guard_diameter_mm=4.0,
        min_coverage=0.90,
        max_gap_bins=1,
    ):
        anchor_ids = [int(v) for v in list(anchor_node_ids or []) if int(v) > 0]
        if len(anchor_ids) < 2:
            return []
        spacing = float(max(0.5, sample_spacing_mm))
        guard_radius = 0.5 * float(max(1.0, candidate_guard_diameter_mm))
        out = []
        for idx_i, start_id in enumerate(anchor_ids[:-1]):
            start_node = dict(node_map.get(int(start_id)) or {})
            if not start_node:
                continue
            start_center = np.asarray(start_node.get("center_ras"), dtype=float).reshape(3)
            for end_id in anchor_ids[(idx_i + 1):]:
                end_node = dict(node_map.get(int(end_id)) or {})
                if not end_node:
                    continue
                end_center = np.asarray(end_node.get("center_ras"), dtype=float).reshape(3)
                anchor_axis = end_center - start_center
                anchor_span = float(np.linalg.norm(anchor_axis))
                if anchor_span < float(max(7.0, 2.5 * spacing)):
                    continue
                axis = anchor_axis / max(anchor_span, 1e-6)
                allowed_node_ids = []
                for node_id, node in node_map.items():
                    center = np.asarray(node.get("center_ras"), dtype=float).reshape(3)
                    proj = float(np.dot(center - start_center, axis))
                    if proj < -0.5 * spacing or proj > anchor_span + 0.5 * spacing:
                        continue
                    radial = float(self._radial_distance_to_line(center.reshape(1, 3), start_center, axis)[0])
                    if radial <= guard_radius:
                        allowed_node_ids.append(int(node_id))
                if int(start_id) not in allowed_node_ids or int(end_id) not in allowed_node_ids:
                    continue
                path = self._best_complex_blob_graph_path(
                    node_map=node_map,
                    adj=adj,
                    start_id=int(start_id),
                    end_id=int(end_id),
                    seed_axis_ras=anchor_axis,
                    allowed_node_ids=allowed_node_ids,
                )
                if len(path) < 2:
                    continue
                path_info = self._complex_blob_path_summary(
                    node_map=node_map,
                    node_ids=path,
                    sample_spacing_mm=spacing,
                    min_coverage=min_coverage,
                    max_gap_bins=max_gap_bins,
                )
                if path_info is None:
                    continue
                path_info["anchor_node_ids"] = [int(start_id), int(end_id)]
                out.append(path_info)
        return out

    def _trace_complex_blob_directional_path(
        self,
        node_map,
        adj,
        start_id,
        initial_direction_ras,
        max_steps=10,
        blocked_node_ids=None,
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
                fit_center, fit_axis = self._fit_line_pca(path_centers, seed_axis=direction)
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
                            fit_center, fit_axis = self._fit_line_pca(path_centers, seed_axis=active_direction)
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
                step_norm = float(candidate.get("step_norm", 0.0))
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
                fit_center, fit_axis = self._fit_line_pca(path_centers, seed_axis=best_dir)
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

    def _build_complex_blob_token_graph(
        self,
        slab_nodes,
        provisional_axis_ras,
        sample_spacing_mm=2.5,
        role_neighbor_k=4,
    ):
        nodes = list(slab_nodes or [])
        axis = np.asarray(provisional_axis_ras, dtype=float).reshape(3)
        axis_norm = float(np.linalg.norm(axis))
        axis = axis / max(axis_norm, 1e-6)
        spacing = float(max(0.5, sample_spacing_mm))
        node_map = {int(node.get("node_id", -1)): node for node in nodes if int(node.get("node_id", -1)) > 0}
        adj = {int(node_id): [] for node_id in node_map.keys()}
        undirected_neighbors = {int(node_id): set() for node_id in node_map.keys()}
        candidate_neighbors = {int(node_id): [] for node_id in node_map.keys()}
        cube_coord_to_node_id = {}
        use_cube_adjacency = bool(node_map) and all(
            (
                dict(node or {}).get("cube_coord") is not None
                and np.asarray(dict(node or {}).get("cube_coord"), dtype=int).reshape(-1).size == 3
            )
            for node in node_map.values()
        )
        if use_cube_adjacency:
            for node_id, node in node_map.items():
                cube_coord = tuple(int(v) for v in np.asarray(dict(node or {}).get("cube_coord"), dtype=int).reshape(3).tolist())
                cube_coord_to_node_id[cube_coord] = int(node_id)
            added_candidate_edges = set()
            for node_id, node in node_map.items():
                node_i = dict(node or {})
                cube_coord = tuple(int(v) for v in np.asarray(node_i.get("cube_coord"), dtype=int).reshape(3).tolist())
                center_i = np.asarray(node_i.get("center_ras"), dtype=float).reshape(3)
                axis_i = np.asarray(node_i.get("axis_ras"), dtype=float).reshape(3)
                rel_i = bool(node_i.get("axis_reliable"))
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
                            if edge_key in added_candidate_edges:
                                continue
                            node_j = dict(node_map.get(int(nb_id)) or {})
                            if not node_j:
                                continue
                            center_j = np.asarray(node_j.get("center_ras"), dtype=float).reshape(3)
                            step = center_j - center_i
                            dist_mm = float(np.linalg.norm(step))
                            if dist_mm <= 1e-6:
                                continue
                            step_dir = step / dist_mm
                            axis_j = np.asarray(node_j.get("axis_ras"), dtype=float).reshape(3)
                            rel_j = bool(node_j.get("axis_reliable"))
                            edge_score = (
                                1.2
                                + 0.6 * float(max(abs(float(np.dot(step_dir, axis_i))) if rel_i else 0.7, abs(float(np.dot(step_dir, axis_j))) if rel_j else 0.7))
                                + 0.12 * float(min(len(np.asarray(node_i.get("points_ras"), dtype=float).reshape(-1, 3)), 5))
                                + 0.12 * float(min(len(np.asarray(node_j.get("points_ras"), dtype=float).reshape(-1, 3)), 5))
                                - 0.08 * float(dist_mm)
                            )
                            candidate_neighbors[int(node_id)].append(
                                {
                                    "neighbor_id": int(nb_id),
                                    "score": float(edge_score),
                                    "dist_mm": float(dist_mm),
                                    "direction_ras": np.asarray(step_dir, dtype=float).reshape(3),
                                }
                            )
                            candidate_neighbors[int(nb_id)].append(
                                {
                                    "neighbor_id": int(node_id),
                                    "score": float(edge_score),
                                    "dist_mm": float(dist_mm),
                                    "direction_ras": np.asarray(-step_dir, dtype=float).reshape(3),
                                }
                            )
                            added_candidate_edges.add(edge_key)
        else:
            cos_local = float(np.cos(np.deg2rad(55.0)))
            ordered_nodes = sorted(nodes, key=lambda node: int(node.get("node_id", 0)))
            for idx_i, node_i in enumerate(ordered_nodes[:-1]):
                center_i = np.asarray(node_i.get("center_ras"), dtype=float).reshape(3)
                axis_i = np.asarray(node_i.get("axis_ras"), dtype=float).reshape(3)
                rel_i = bool(node_i.get("axis_reliable"))
                for node_j in ordered_nodes[(idx_i + 1) :]:
                    center_j = np.asarray(node_j.get("center_ras"), dtype=float).reshape(3)
                    step = center_j - center_i
                    dist_mm = float(np.linalg.norm(step))
                    if dist_mm <= 1e-6:
                        continue
                    step_dir = step / dist_mm
                    axis_j = np.asarray(node_j.get("axis_ras"), dtype=float).reshape(3)
                    rel_j = bool(node_j.get("axis_reliable"))
                    if rel_i and abs(float(np.dot(step_dir, axis_i))) < cos_local:
                        continue
                    if rel_j and abs(float(np.dot(step_dir, axis_j))) < cos_local:
                        continue
                    edge_score = (
                        1.8
                        + 0.8 * float(max(abs(float(np.dot(step_dir, axis_i))) if rel_i else 0.7, abs(float(np.dot(step_dir, axis_j))) if rel_j else 0.7))
                        + 0.20 * float(min(len(np.asarray(node_i.get("points_ras"), dtype=float).reshape(-1, 3)), 5))
                        + 0.20 * float(min(len(np.asarray(node_j.get("points_ras"), dtype=float).reshape(-1, 3)), 5))
                        - 0.18 * float(dist_mm)
                    )
                    candidate_neighbors[int(node_i.get("node_id", -1))].append(
                        {
                            "neighbor_id": int(node_j.get("node_id", -1)),
                            "score": float(edge_score),
                            "dist_mm": float(dist_mm),
                            "direction_ras": np.asarray(step_dir, dtype=float).reshape(3),
                        }
                    )
                    candidate_neighbors[int(node_j.get("node_id", -1))].append(
                        {
                            "neighbor_id": int(node_i.get("node_id", -1)),
                            "score": float(edge_score),
                            "dist_mm": float(dist_mm),
                            "direction_ras": np.asarray(-step_dir, dtype=float).reshape(3),
                        }
                    )
        role_neighbor_k = int(max(2, min(8, role_neighbor_k)))
        selected_neighbors = {int(node_id): [] for node_id in node_map.keys()}
        for node_id, node in node_map.items():
            ordered_neighbor_infos = sorted(
                list(candidate_neighbors.get(int(node_id)) or []),
                key=lambda info: (float(info.get("dist_mm", 999.0)), -float(info.get("score", 0.0))),
            )
            neighbor_infos = list(ordered_neighbor_infos if use_cube_adjacency else ordered_neighbor_infos[:role_neighbor_k])
            branch_groups = self._cluster_complex_blob_neighbor_directions(neighbor_infos)
            node["branch_groups"] = list(branch_groups)
            node["branch_count"] = int(len(branch_groups))
            role, primary_axis, axis_strength, one_sidedness = self._complex_blob_node_state_from_branch_groups(branch_groups)
            node["node_role"] = str(role)
            node["primary_axis_ras"] = None if primary_axis is None else np.asarray(primary_axis, dtype=float).reshape(3)
            node["axis_strength"] = float(axis_strength)
            node["one_sidedness"] = float(one_sidedness)
            node["seed_score"] = float(max(0.0, axis_strength) + 0.8 * max(0.0, one_sidedness))
            selected_neighbors[int(node_id)] = [dict(info) for info in neighbor_infos]
        added_edges = set()
        for node_id, neighbor_infos in selected_neighbors.items():
            for info in list(neighbor_infos or []):
                nb_id = int(dict(info or {}).get("neighbor_id", -1))
                if int(nb_id) <= 0:
                    continue
                edge_key = tuple(sorted((int(node_id), int(nb_id))))
                if edge_key in added_edges:
                    continue
                node_i = dict(node_map.get(int(node_id)) or {})
                node_j = dict(node_map.get(int(nb_id)) or {})
                if not node_i or not node_j:
                    continue
                center_i = np.asarray(node_i.get("center_ras"), dtype=float).reshape(3)
                center_j = np.asarray(node_j.get("center_ras"), dtype=float).reshape(3)
                step = center_j - center_i
                dist_mm = float(np.linalg.norm(step))
                if dist_mm <= 1e-6:
                    continue
                step_dir = step / dist_mm
                axis_i = np.asarray(node_i.get("axis_ras"), dtype=float).reshape(3)
                axis_j = np.asarray(node_j.get("axis_ras"), dtype=float).reshape(3)
                rel_i = bool(node_i.get("axis_reliable"))
                rel_j = bool(node_j.get("axis_reliable"))
                edge_score = (
                    1.8
                    + 0.8 * float(max(abs(float(np.dot(step_dir, axis_i))) if rel_i else 0.7, abs(float(np.dot(step_dir, axis_j))) if rel_j else 0.7))
                    + 0.20 * float(min(len(np.asarray(node_i.get("points_ras"), dtype=float).reshape(-1, 3)), 5))
                    + 0.20 * float(min(len(np.asarray(node_j.get("points_ras"), dtype=float).reshape(-1, 3)), 5))
                    - 0.18 * float(dist_mm)
                )
                edge_ij = {"neighbor_id": int(nb_id), "score": float(edge_score), "dist_mm": float(dist_mm)}
                edge_ji = {"neighbor_id": int(node_id), "score": float(edge_score), "dist_mm": float(dist_mm)}
                adj[int(node_id)].append(edge_ij)
                adj[int(nb_id)].append(edge_ji)
                undirected_neighbors[int(node_id)].add(int(nb_id))
                undirected_neighbors[int(nb_id)].add(int(node_id))
                added_edges.add(edge_key)
        node_degrees = {int(node_id): int(len(neighbors)) for node_id, neighbors in undirected_neighbors.items()}
        anchor_node_ids = []
        for node_id, node in node_map.items():
            role = str(node.get("node_role") or self._complex_blob_node_role(int(node_degrees.get(int(node_id), 0))))
            if role != "core":
                anchor_node_ids.append(int(node_id))
        if len(anchor_node_ids) < 2:
            ranked = sorted(
                node_map.keys(),
                key=lambda node_id: (0 if str(dict(node_map.get(int(node_id)) or {}).get("node_role") or "") == "end" else 1, int(node_degrees.get(int(node_id), 0))),
            )
            anchor_node_ids = [int(v) for v in ranked[: min(4, len(ranked))]]
        return {
            "node_map": dict(node_map),
            "adj": dict(adj),
            "node_degrees": dict(node_degrees),
            "anchor_node_ids": sorted(set(int(v) for v in anchor_node_ids)),
        }

    @staticmethod
    def _cluster_complex_blob_neighbor_directions(neighbor_infos, same_branch_angle_deg=35.0):
        infos = [dict(info or {}) for info in list(neighbor_infos or []) if int(dict(info or {}).get("neighbor_id", -1)) > 0]
        if not infos:
            return []
        cos_same = float(np.cos(np.deg2rad(float(max(5.0, min(80.0, same_branch_angle_deg))))))
        groups = []
        ordered = sorted(
            infos,
            key=lambda info: (float(info.get("dist_mm", 999.0)), -float(info.get("score", 0.0))),
        )
        for info in ordered:
            direction_val = info.get("direction_ras")
            if direction_val is None:
                direction_val = [0.0, 0.0, 1.0]
            direction = np.asarray(direction_val, dtype=float).reshape(3)
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm <= 1e-6:
                continue
            direction = direction / direction_norm
            best_group = None
            best_dot = None
            for group in groups:
                group_dir_val = group.get("direction_ras")
                if group_dir_val is None:
                    group_dir_val = [0.0, 0.0, 1.0]
                group_dir = np.asarray(group_dir_val, dtype=float).reshape(3)
                dot_val = float(np.dot(direction, group_dir))
                if dot_val < cos_same:
                    continue
                if best_dot is None or dot_val > best_dot:
                    best_dot = dot_val
                    best_group = group
            if best_group is None:
                groups.append(
                    {
                        "direction_ras": np.asarray(direction, dtype=float).reshape(3),
                        "members": [dict(info)],
                    }
                )
                continue
            best_group["members"].append(dict(info))
            weighted = np.zeros((3,), dtype=float)
            for member in list(best_group.get("members") or []):
                member_dir_val = member.get("direction_ras")
                if member_dir_val is None:
                    member_dir_val = [0.0, 0.0, 1.0]
                member_dir = np.asarray(member_dir_val, dtype=float).reshape(3)
                member_norm = float(np.linalg.norm(member_dir))
                if member_norm <= 1e-6:
                    continue
                weighted += member_dir / member_norm
            weighted_norm = float(np.linalg.norm(weighted))
            if weighted_norm > 1e-6:
                best_group["direction_ras"] = np.asarray(weighted / weighted_norm, dtype=float).reshape(3)
        return groups

    @staticmethod
    def _complex_blob_node_state_from_branch_groups(branch_groups, opposite_angle_deg=40.0):
        groups = []
        for group in list(branch_groups or []):
            group_dict = dict(group or {})
            direction_val = group_dict.get("direction_ras")
            if direction_val is None:
                continue
            if np.asarray(direction_val, dtype=float).size == 3:
                groups.append(group_dict)
        branch_count = int(len(groups))
        if branch_count <= 0:
            return ("crossing", None, 0.0, 0.0)
        if branch_count == 1:
            dir0 = np.asarray(groups[0].get("direction_ras"), dtype=float).reshape(3)
            dir0 = dir0 / max(float(np.linalg.norm(dir0)), 1e-6)
            return ("core", dir0, 1.0, 1.0)
        if branch_count >= 3:
            return ("crossing", None, 0.0, 0.0)
        dir0 = np.asarray(groups[0].get("direction_ras"), dtype=float).reshape(3)
        dir1 = np.asarray(groups[1].get("direction_ras"), dtype=float).reshape(3)
        dir0 = dir0 / max(float(np.linalg.norm(dir0)), 1e-6)
        dir1 = dir1 / max(float(np.linalg.norm(dir1)), 1e-6)
        dot_val = float(np.dot(dir0, dir1))
        group_weights = []
        for group in groups[:2]:
            members = [dict(member or {}) for member in list(group.get("members") or [])]
            if not members:
                group_weights.append(1.0)
                continue
            group_weights.append(sum(max(0.0, float(member.get("score", 0.0))) for member in members))
        w0 = float(group_weights[0]) if len(group_weights) > 0 else 1.0
        w1 = float(group_weights[1]) if len(group_weights) > 1 else 1.0
        if dot_val <= -float(np.cos(np.deg2rad(float(max(5.0, min(85.0, opposite_angle_deg)))))):
            axis = dir0 - dir1
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm <= 1e-6:
                axis = dir0
            else:
                axis = axis / axis_norm
            if w1 > w0 and float(np.dot(axis, dir1)) > 0.0:
                axis = -axis
            one_sidedness = float(abs(w0 - w1) / max(1e-6, w0 + w1))
            return ("core", axis, float(-dot_val), one_sidedness)
        return ("crossing", None, 0.0, 0.0)

    @staticmethod
    def _complex_blob_node_role(degree):
        deg = int(degree)
        if deg <= 1:
            return "end"
        if deg == 2:
            return "core"
        return "crossing"

    def _best_complex_blob_graph_path(
        self,
        node_map,
        adj,
        start_id,
        end_id,
        seed_axis_ras=None,
        allowed_node_ids=None,
    ):
        start_node = dict(node_map.get(int(start_id)) or {})
        end_node = dict(node_map.get(int(end_id)) or {})
        if not start_node or not end_node:
            return []
        seed_axis = np.asarray(seed_axis_ras if seed_axis_ras is not None else (np.asarray(end_node.get("center_ras"), dtype=float).reshape(3) - np.asarray(start_node.get("center_ras"), dtype=float).reshape(3)), dtype=float).reshape(3)
        seed_axis_norm = float(np.linalg.norm(seed_axis))
        if seed_axis_norm <= 1e-6:
            return []
        seed_axis = seed_axis / seed_axis_norm
        start_center = np.asarray(start_node.get("center_ras"), dtype=float).reshape(3)
        target = np.asarray(end_node.get("center_ras"), dtype=float).reshape(3)
        allowed = None if allowed_node_ids is None else set(int(v) for v in list(allowed_node_ids or []) if int(v) > 0)
        memo = {}

        def _dfs(node_id, prev_id):
            node_id = int(node_id)
            prev_id = int(prev_id)
            if node_id == int(end_id):
                return (0.0, [int(end_id)])
            memo_key = (int(node_id), int(prev_id))
            if memo_key in memo:
                return memo[memo_key]
            best_score = None
            best_path = []
            node_center = np.asarray(node_map[int(node_id)].get("center_ras"), dtype=float).reshape(3)
            prev_axis = seed_axis
            if int(prev_id) > 0 and int(prev_id) in node_map:
                prev_center = np.asarray(node_map[int(prev_id)].get("center_ras"), dtype=float).reshape(3)
                step_prev = node_center - prev_center
                step_prev_norm = float(np.linalg.norm(step_prev))
                if step_prev_norm > 1e-6:
                    prev_axis = step_prev / step_prev_norm
            curr_to_target = float(np.linalg.norm(target - node_center))
            curr_proj = float(np.dot(node_center - start_center, seed_axis))
            for edge in list(adj.get(int(node_id)) or []):
                nb_id = int(edge.get("neighbor_id", -1))
                if nb_id == int(prev_id):
                    continue
                if allowed is not None and int(nb_id) not in allowed:
                    continue
                nb_node = dict(node_map.get(int(nb_id)) or {})
                if not nb_node:
                    continue
                nb_center = np.asarray(nb_node.get("center_ras"), dtype=float).reshape(3)
                step = nb_center - node_center
                step_norm = float(np.linalg.norm(step))
                if step_norm <= 1e-6:
                    continue
                step_dir = step / step_norm
                if float(np.dot(step_dir, seed_axis)) <= 0.10:
                    continue
                if float(np.dot(step_dir, prev_axis)) <= 0.25:
                    continue
                next_proj = float(np.dot(nb_center - start_center, seed_axis))
                if next_proj <= curr_proj + 0.10 * float(max(1.0, step_norm)):
                    continue
                next_to_target = float(np.linalg.norm(target - nb_center))
                if next_to_target > curr_to_target + float(max(1.0, 0.75 * step_norm)):
                    continue
                tail_score, tail_path = _dfs(int(nb_id), int(node_id))
                if not tail_path:
                    continue
                candidate_score = (
                    float(edge.get("score", 0.0))
                    + float(tail_score)
                    + 0.8 * float(np.dot(step_dir, seed_axis))
                    + 0.5 * float(np.dot(step_dir, prev_axis))
                    - 0.08 * float(next_to_target)
                )
                if best_score is None or candidate_score > best_score:
                    best_score = float(candidate_score)
                    best_path = [int(node_id)] + [int(v) for v in list(tail_path or [])]
            memo[memo_key] = (best_score, best_path) if best_path else (None, [])
            return memo[memo_key]

        _score, path = _dfs(int(start_id), -1)
        if not path or int(path[-1]) != int(end_id):
            return []
        return [int(v) for v in path]

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
        proj_support = ((support_points - fit_center.reshape(1, 3)) @ fit_axis.reshape(3, 1)).reshape(-1)
        occupancy_bin_mm = float(max(0.5, sample_spacing_mm))
        if all(dict(node or {}).get("cube_coord") is not None for node in node_list):
            occupancy_bin_mm = float(
                self._adaptive_projected_support_bin_mm(
                    proj_vals=proj_support,
                    base_bin_mm=occupancy_bin_mm,
                    max_scale=2.5,
                )
            )
        occ = self._axial_occupancy_metrics(proj_support, bin_mm=float(occupancy_bin_mm))
        shape = self._point_cloud_shape_metrics(fit_points, seed_axis=fit_axis)
        diameter_mm = float(shape.get("diameter_mm", 0.0))
        has_crossing = any(str(node.get("node_role") or "") == "crossing" for node in node_list)
        coverage_required = float(min_coverage)
        if (
            has_crossing
            and span >= 24.0
            and diameter_mm <= 3.5
            and rms <= 1.25
            and int(occ.get("max_gap_bins", 0)) <= int(max_gap_bins)
        ):
            coverage_required = min(float(coverage_required), 0.80)
        if span < float(max(7.0, 2.5 * sample_spacing_mm)):
            return None
        if diameter_mm > 8.0:
            return None
        if rms > 2.25:
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
