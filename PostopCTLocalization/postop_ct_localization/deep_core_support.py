"""Deep-core support extraction: masks, connected support, and proposal atoms."""

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

from shank_core.blob_candidates import build_blob_labelmap, extract_blob_candidates
from shank_core.masking import compute_head_distance_map_kji, largest_component_binary
from .deep_core_annulus import DeepCoreAnnulusMixin
from .deep_core_atoms import DeepCoreAtomBuilderMixin
from .deep_core_complex_blob import DeepCoreComplexBlobMixin
from .deep_core_config import (
    DeepCoreMaskConfig,
    DeepCoreSupportConfig,
    deep_core_default_config,
)
from .deep_core_contact_chain import DeepCoreContactChainMixin


_DEEP_CORE_DEFAULTS = deep_core_default_config()
_DEFAULT_MASK_CONFIG = _DEEP_CORE_DEFAULTS.mask
_DEFAULT_SUPPORT_CONFIG = _DEEP_CORE_DEFAULTS.support


class DeepCoreSupportLogicMixin(
    DeepCoreAtomBuilderMixin,
    DeepCoreContactChainMixin,
    DeepCoreComplexBlobMixin,
    DeepCoreAnnulusMixin,
):
    """Own the mask and support stages of the Deep Core pipeline."""

    def _build_deep_core_mask_stage(
        self,
        volume_node,
        mask_config=None,
        volume_accessor=None,
    ):
        """Build the deep-core mask context from one config section.

        When *volume_accessor* is provided, all volume I/O goes through
        the accessor (pure-Python compatible).  Otherwise falls back to
        the legacy Slicer path for backward compatibility.
        """

        if sitk is None:
            raise RuntimeError("SimpleITK is required for deep-core debug preview.")
        cfg = mask_config if mask_config is not None else _DEFAULT_MASK_CONFIG

        if volume_accessor is not None:
            arr_kji = np.asarray(volume_accessor.array_kji(volume_node), dtype=np.float32)
            spacing_xyz = volume_accessor.spacing_xyz(volume_node)
        else:
            from __main__ import slicer  # legacy fallback
            arr_kji = np.asarray(slicer.util.arrayFromVolume(volume_node), dtype=np.float32)
            spacing_xyz = tuple(float(v) for v in volume_node.GetSpacing())

        arr_clip = np.asarray(arr_kji, dtype=np.float32)
        if np.isfinite(float(cfg.hull_clip_hu)):
            arr_clip = np.minimum(arr_clip, float(cfg.hull_clip_hu))
        hull_img = sitk.GetImageFromArray(arr_clip.astype(np.float32))
        hull_img.SetSpacing((float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])))
        if float(cfg.hull_sigma_mm) > 0.0:
            hull_img = sitk.SmoothingRecursiveGaussian(hull_img, float(cfg.hull_sigma_mm))
        smoothed_hull_kji = sitk.GetArrayFromImage(hull_img).astype(np.float32)

        binary = sitk.BinaryThreshold(
            hull_img,
            lowerThreshold=float(cfg.hull_threshold_hu),
            upperThreshold=float(max(float(np.nanmax(smoothed_hull_kji)), float(cfg.hull_threshold_hu) + 1.0)),
            insideValue=1,
            outsideValue=0,
        )
        if int(cfg.hull_open_vox) > 0:
            binary = sitk.BinaryMorphologicalOpening(binary, [int(cfg.hull_open_vox)] * 3, sitk.sitkBall)
        hull_lcc = largest_component_binary(binary)
        if hull_lcc is None:
            raise RuntimeError("Failed to build a non-air hull mask from the selected CT.")
        if int(cfg.hull_close_vox) > 0:
            hull_lcc = sitk.BinaryMorphologicalClosing(hull_lcc, [int(cfg.hull_close_vox)] * 3, sitk.sitkBall)
        hull_mask_kji = sitk.GetArrayFromImage(hull_lcc).astype(bool)
        head_distance_map_kji = np.asarray(
            compute_head_distance_map_kji(hull_mask_kji, spacing_xyz=spacing_xyz),
            dtype=np.float32,
        )
        deep_core_mask_kji = np.logical_and(
            hull_mask_kji,
            head_distance_map_kji >= float(max(0.0, cfg.deep_core_shrink_mm)),
        )

        metal_threshold_hu = float(cfg.metal_threshold_hu)
        metal_mask_kji = np.asarray(arr_kji >= float(metal_threshold_hu), dtype=bool)
        metal_grown_mask_kji = metal_mask_kji.copy()
        if int(cfg.metal_grow_vox) > 0:
            metal_img = sitk.GetImageFromArray(metal_mask_kji.astype(np.uint8))
            metal_grown_mask_kji = sitk.GetArrayFromImage(
                sitk.BinaryDilate(metal_img, [int(cfg.metal_grow_vox)] * 3, sitk.sitkBall)
            ).astype(bool)

        deep_seed_raw_mask_kji = np.logical_and(metal_mask_kji, deep_core_mask_kji)
        deep_seed_mask_kji = np.logical_and(metal_grown_mask_kji, deep_core_mask_kji)

        # Debug visualization volumes — created only when a volume_accessor
        # with update_scalar_volume support is available, or via the legacy
        # mixin path.
        smooth_node = None
        distance_node = None
        if volume_accessor is not None and hasattr(volume_accessor, "update_scalar_volume"):
            smooth_node = volume_accessor.update_scalar_volume(
                reference_volume_node=volume_node,
                node_name=f"{volume_node.GetName()}_HullSmooth",
                array_kji=smoothed_hull_kji,
            )
            distance_node = volume_accessor.update_scalar_volume(
                reference_volume_node=volume_node,
                node_name=f"{volume_node.GetName()}_HeadDistanceMm",
                array_kji=head_distance_map_kji,
            )
        elif hasattr(self, "_update_scalar_volume_from_array"):
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
        return {
            "volume_node_id": volume_node.GetID() if hasattr(volume_node, "GetID") else "",
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
                "hull_threshold_hu": float(cfg.hull_threshold_hu),
                "hull_clip_hu": float(cfg.hull_clip_hu),
                "hull_sigma_mm": float(cfg.hull_sigma_mm),
                "hull_open_vox": int(cfg.hull_open_vox),
                "hull_close_vox": int(cfg.hull_close_vox),
                "deep_core_shrink_mm": float(cfg.deep_core_shrink_mm),
                "metal_threshold_hu": float(cfg.metal_threshold_hu),
                "metal_grow_vox": int(cfg.metal_grow_vox),
            },
        }

    def _build_deep_core_support_stage(
        self,
        volume_node,
        mask_result,
        support_config=None,
        annulus_config=None,
        internal_config=None,
        show_support_diagnostics=True,
        volume_accessor=None,
    ):
        """Extract support atoms and tokenized blob support from the mask stage."""

        support_cfg = support_config if support_config is not None else _DEFAULT_SUPPORT_CONFIG
        annulus_cfg = annulus_config if annulus_config is not None else _DEEP_CORE_DEFAULTS.annulus
        internal_cfg = internal_config if internal_config is not None else _DEEP_CORE_DEFAULTS.internal
        mask_payload = dict(getattr(mask_result, "payload", mask_result) or {})

        if volume_accessor is not None:
            arr_kji = np.asarray(volume_accessor.array_kji(volume_node), dtype=np.float32)
            _ijk_kji_to_ras = lambda idx: volume_accessor.ijk_kji_to_ras_points(volume_node, idx)
        else:
            from __main__ import slicer  # legacy fallback
            arr_kji = np.asarray(slicer.util.arrayFromVolume(volume_node), dtype=np.float32)
            _ijk_kji_to_ras = lambda idx: self._ijk_kji_to_ras_points(volume_node, idx)

        head_distance_map_kji = mask_payload.get("head_distance_map_kji")
        deep_seed_raw_mask_kji = mask_payload.get("deep_seed_raw_mask_kji")
        deep_seed_mask_kji = mask_payload.get("deep_seed_mask_kji")
        raw_blob_result = extract_blob_candidates(
            metal_mask_kji=deep_seed_raw_mask_kji,
            arr_kji=arr_kji,
            depth_map_kji=head_distance_map_kji,
            ijk_kji_to_ras_fn=_ijk_kji_to_ras,
        )
        grown_blob_result = extract_blob_candidates(
            metal_mask_kji=deep_seed_mask_kji,
            arr_kji=arr_kji,
            depth_map_kji=head_distance_map_kji,
            ijk_kji_to_ras_fn=_ijk_kji_to_ras,
        )
        sample_payload = self._build_support_atom_payload(
            volume_node=volume_node,
            labels_kji=raw_blob_result.get("labels_kji"),
            blobs=list(raw_blob_result.get("blobs") or []),
            support_spacing_mm=float(support_cfg.support_spacing_mm),
            component_min_elongation=float(support_cfg.component_min_elongation),
            line_atom_diameter_max_mm=float(support_cfg.line_atom_diameter_max_mm),
            line_atom_min_span_mm=float(support_cfg.line_atom_min_span_mm),
            line_atom_min_pca_dominance=float(support_cfg.line_atom_min_pca_dominance),
            contact_component_diameter_max_mm=float(support_cfg.contact_component_diameter_max_mm),
            support_cube_size_mm=float(support_cfg.support_cube_size_mm),
            head_distance_map_kji=head_distance_map_kji,
            annulus_config=annulus_cfg,
            internal_config=internal_cfg,
        )
        raw_blobs = [dict(blob or {}) for blob in list(raw_blob_result.get("blobs") or [])]
        out = {
            "blob_labelmap_kji": build_blob_labelmap(raw_blob_result.get("labels_kji")) if bool(show_support_diagnostics) else None,
            "blob_centroids_all_ras": np.asarray(
                [blob.get("centroid_ras") for blob in raw_blobs if blob.get("centroid_ras") is not None],
                dtype=float,
            ).reshape(-1, 3),
            "support_atoms": list(sample_payload.get("support_atoms") or []),
        }
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
        stats = dict(mask_payload.get("stats") or {})
        stats.update(
            {
                "support_spacing_mm": float(support_cfg.support_spacing_mm),
                "component_min_elongation": float(support_cfg.component_min_elongation),
                "line_atom_diameter_max_mm": float(support_cfg.line_atom_diameter_max_mm),
                "line_atom_min_span_mm": float(support_cfg.line_atom_min_span_mm),
                "line_atom_min_pca_dominance": float(support_cfg.line_atom_min_pca_dominance),
                "contact_component_diameter_max_mm": float(support_cfg.contact_component_diameter_max_mm),
                "support_cube_size_mm": float(support_cfg.support_cube_size_mm),
                "deep_seed_sampled_blob_count": int(len(dict(out["blob_elongation_by_id"] or {}))),
                "deep_seed_atom_count": int(len(list(out["support_atoms"] or []))),
                "deep_seed_raw_blob_count": int(raw_blob_result.get("blob_count_total", 0)),
                "deep_seed_grown_blob_count": int(grown_blob_result.get("blob_count_total", 0)),
                "deep_seed_sample_count": int(np.asarray(out["blob_sample_points_ras"]).reshape(-1, 3).shape[0]),
                "deep_seed_line_blob_token_count": int(np.asarray(out["line_blob_sample_points_ras"]).reshape(-1, 3).shape[0]),
                "deep_seed_contact_blob_token_count": int(np.asarray(out["contact_blob_sample_points_ras"]).reshape(-1, 3).shape[0]),
                "deep_seed_complex_blob_token_count": int(np.asarray(out["complex_blob_sample_points_ras"]).reshape(-1, 3).shape[0]),
                "deep_seed_complex_blob_chain_row_count": int(len(list(out["complex_blob_chain_rows"] or []))),
                "deep_seed_contact_chain_row_count": int(len(list(out["contact_chain_rows"] or []))),
                "deep_seed_contact_chain_debug_row_count": int(len(list(out["contact_chain_debug_rows"] or []))),
            }
        )
        out["stats"] = stats
        return out
