"""Deep Core trajectory detection pipeline.

Detects SEEG electrode trajectories from postop CT scans using a
deep-core mask, support atom extraction, and graph-based candidate
generation with annulus rejection and extension.

This pipeline reuses the algorithm code in the PostopCTLocalization
module's mixin files, wrapping each stage in the standard
``run_stage()`` framework for timing, diagnostics, and error handling.
"""

from __future__ import annotations

import sys
import time
from typing import Any

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

from shank_core.blob_candidates import build_blob_labelmap, extract_blob_candidates
from shank_core.masking import (
    build_outside_air_mask_kji,
    compute_head_distance_map_kji,
    largest_component_binary,
)

from ..contracts import DetectionContext, DetectionResult
from ..pipelines.base import BaseDetectionPipeline


# ---------------------------------------------------------------------------
# Volume proxy — lets mixin code treat DetectionContext data like a volume_node
# ---------------------------------------------------------------------------

def _get_vtk_module():
    """Return the vtk module if available (Slicer) or None (CLI)."""
    try:
        from __main__ import vtk  # Slicer embeds vtk in __main__
        return vtk
    except ImportError:
        try:
            import vtk  # Standalone vtk install
            return vtk
        except ImportError:
            return None


def _get_slicer_module():
    try:
        from __main__ import slicer
        return slicer
    except ImportError:
        return None


def _slicer_array_kji(volume_node):
    """Extract a numpy array from a Slicer volume node (if available)."""
    slicer = _get_slicer_module()
    if slicer is None:
        raise RuntimeError(
            "No Slicer available; algorithm code received a non-proxy volume_node"
        )
    return np.asarray(slicer.util.arrayFromVolume(volume_node), dtype=float)


def _slicer_ras_to_ijk_fn(volume_node):
    """Return a ras_xyz -> ijk_xyz callable for a Slicer volume."""
    vtk = _get_vtk_module()
    if vtk is None:
        raise RuntimeError(
            "No vtk available; algorithm code received a non-proxy volume_node"
        )
    m_vtk = vtk.vtkMatrix4x4()
    volume_node.GetRASToIJKMatrix(m_vtk)
    mat = np.eye(4, dtype=float)
    for r in range(4):
        for c in range(4):
            mat[r, c] = float(m_vtk.GetElement(r, c))

    def _fn(ras_xyz):
        h = np.array([float(ras_xyz[0]), float(ras_xyz[1]), float(ras_xyz[2]), 1.0], dtype=float)
        return (mat @ h)[:3]

    return _fn


def _slicer_ijk_kji_to_ras(volume_node, ijk_kji):
    """Convert KJI indices to RAS points via the volume's IJK->RAS matrix."""
    vtk = _get_vtk_module()
    if vtk is None:
        raise RuntimeError(
            "No vtk available; algorithm code received a non-proxy volume_node"
        )
    idx = np.asarray(ijk_kji, dtype=float).reshape(-1, 3)
    if idx.size == 0:
        return np.empty((0, 3), dtype=float)
    ijk = np.zeros_like(idx, dtype=float)
    ijk[:, 0] = idx[:, 2]
    ijk[:, 1] = idx[:, 1]
    ijk[:, 2] = idx[:, 0]
    ijk_h = np.concatenate([ijk, np.ones((ijk.shape[0], 1), dtype=float)], axis=1)
    m_vtk = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(m_vtk)
    mat = np.eye(4, dtype=float)
    for r in range(4):
        for c in range(4):
            mat[r, c] = float(m_vtk.GetElement(r, c))
    return (mat @ ijk_h.T).T[:, :3]


class _ContextVolumeProxy:
    """Lightweight object that satisfies what mixin code expects from
    ``volume_node`` — backed entirely by ``DetectionContext`` data.

    The mixin code calls ``volume_node.GetSpacing()``,
    ``slicer.util.arrayFromVolume(volume_node)``, and coordinate
    transforms.  This proxy supplies those without touching Slicer.
    """

    def __init__(self, ctx: DetectionContext):
        self._arr_kji = np.asarray(ctx["arr_kji"], dtype=float)
        self._spacing = tuple(float(v) for v in ctx["spacing_xyz"])
        self._ijk_kji_to_ras_fn = ctx.get("ijk_kji_to_ras_fn")
        self._ras_to_ijk_fn = ctx.get("ras_to_ijk_fn")
        self._name = str(ctx.get("run_id") or "deep_core")
        self._volume_node = ctx.get("extras", {}).get("volume_node")

    def GetSpacing(self):
        return self._spacing

    def GetName(self):
        return self._name

    def GetID(self):
        if self._volume_node is not None and hasattr(self._volume_node, "GetID"):
            return self._volume_node.GetID()
        return self._name

    def GetIJKToRASMatrix(self, vtk_matrix):
        """Populate a vtkMatrix4x4 — only called from legacy paths.
        If the real volume_node is available, delegate to it."""
        if self._volume_node is not None and hasattr(self._volume_node, "GetIJKToRASMatrix"):
            return self._volume_node.GetIJKToRASMatrix(vtk_matrix)
        raise RuntimeError("GetIJKToRASMatrix requires a real Slicer volume node in ctx.extras")

    def GetRASToIJKMatrix(self, vtk_matrix):
        if self._volume_node is not None and hasattr(self._volume_node, "GetRASToIJKMatrix"):
            return self._volume_node.GetRASToIJKMatrix(vtk_matrix)
        raise RuntimeError("GetRASToIJKMatrix requires a real Slicer volume node in ctx.extras")


# ---------------------------------------------------------------------------
# Deep Core config adapter
# ---------------------------------------------------------------------------

def _parse_deep_core_config(cfg: dict[str, Any]):
    """Import and build a DeepCoreConfig from a flat dict."""
    # Lazy import — PostopCTLocalization may not be on sys.path in
    # standalone CLI usage.
    from postop_ct_localization.deep_core_config import (
        DeepCoreConfig,
        deep_core_default_config,
    )

    if isinstance(cfg, DeepCoreConfig):
        return cfg
    defaults = deep_core_default_config()
    if not cfg:
        return defaults
    # If keys are dotted (mask.hull_threshold_hu), use with_updates
    dotted = {k: v for k, v in cfg.items() if "." in str(k)}
    if not dotted:
        return defaults
    # Coerce list values to tuples for tuple fields (e.g. model_fit.families)
    coerced = {}
    for k, v in dotted.items():
        if k == "model_fit.families" and isinstance(v, (list, set)):
            coerced[k] = tuple(str(s) for s in v)
        else:
            coerced[k] = v
    return defaults.with_updates(coerced)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _make_pipeline_class():
    """Build the pipeline class inside a factory to keep mixin imports local."""

    from postop_ct_localization.deep_core_annulus import DeepCoreAnnulusMixin
    from postop_ct_localization.deep_core_atoms import DeepCoreAtomBuilderMixin
    from postop_ct_localization.deep_core_candidate_inputs import DeepCoreCandidateInputMixin
    from postop_ct_localization.deep_core_complex_blob import DeepCoreComplexBlobMixin
    from postop_ct_localization.deep_core_contact_chain import DeepCoreContactChainMixin
    from postop_ct_localization.deep_core_proposals import DeepCoreProposalLogicMixin

    class DeepCoreV1Pipeline(
        BaseDetectionPipeline,
        DeepCoreProposalLogicMixin,
        DeepCoreAtomBuilderMixin,
        DeepCoreCandidateInputMixin,
        DeepCoreContactChainMixin,
        DeepCoreComplexBlobMixin,
        DeepCoreAnnulusMixin,
    ):
        pipeline_id = "deep_core_v1"
        pipeline_version = "1.0.0"

        def _ijk_kji_to_ras_points(self, volume_node, ijk_kji):
            """Bridge for mixin code that calls self._ijk_kji_to_ras_points."""
            if isinstance(volume_node, _ContextVolumeProxy) and volume_node._ijk_kji_to_ras_fn is not None:
                return volume_node._ijk_kji_to_ras_fn(ijk_kji)
            return _slicer_ijk_kji_to_ras(volume_node, ijk_kji)

        def extract_threshold_candidates_lps(
            self,
            volume_node,
            threshold,
            head_mask_threshold_hu=-500.0,
            min_metal_depth_mm=5.0,
            max_metal_depth_mm=220.0,
            head_mask_method="outside_air",
        ):
            """Extract metal candidate points from a CT volume as LPS coords.

            Uses the proxy's cached array + ``ijk_kji_to_ras_fn`` when
            possible; falls back to a real Slicer volume node otherwise.
            """
            from shank_core.masking import build_preview_masks

            # Resolve array, spacing, and an IJK/KJI->RAS transform
            if isinstance(volume_node, _ContextVolumeProxy):
                arr = volume_node._arr_kji
                spacing = volume_node._spacing
                ijk_kji_to_ras_fn = volume_node._ijk_kji_to_ras_fn
                real_node = volume_node._volume_node
            else:
                arr = _slicer_array_kji(volume_node)
                spacing = tuple(float(v) for v in volume_node.GetSpacing())
                ijk_kji_to_ras_fn = None
                real_node = volume_node

            used_threshold = float(threshold)
            best_preview: dict | None = None
            best_count = -1
            while True:
                preview = build_preview_masks(
                    arr_kji=np.asarray(arr, dtype=float),
                    spacing_xyz=spacing,
                    threshold=float(used_threshold),
                    use_head_mask=True,
                    build_head_mask=True,
                    head_mask_threshold_hu=float(head_mask_threshold_hu),
                    head_mask_method=str(head_mask_method),
                    head_gate_erode_vox=1,
                    head_gate_dilate_vox=1,
                    head_gate_margin_mm=0.0,
                    min_metal_depth_mm=float(min_metal_depth_mm),
                    max_metal_depth_mm=float(max_metal_depth_mm),
                    include_debug_masks=False,
                )
                count = int(preview.get("depth_kept_count") or 0)
                if count > best_count:
                    best_preview = preview
                    best_count = count
                if count >= 300000 or used_threshold <= 500.0 + 1e-6:
                    break
                used_threshold = max(500.0, used_threshold - 50.0)
            preview = best_preview if best_preview is not None else {}
            idx = np.argwhere(np.asarray(preview.get("metal_depth_pass_mask_kji"), dtype=bool))
            if idx.size == 0:
                return {
                    "points_lps": np.empty((0, 3), dtype=float),
                    "threshold_hu": float(used_threshold),
                }

            # Convert KJI indices to RAS using whichever transform we have
            if ijk_kji_to_ras_fn is not None:
                ras = np.asarray(ijk_kji_to_ras_fn(idx.astype(float)), dtype=float).reshape(-1, 3)
            elif real_node is not None:
                ras = _slicer_ijk_kji_to_ras(real_node, idx.astype(float))
            else:
                return {
                    "points_lps": np.empty((0, 3), dtype=float),
                    "threshold_hu": float(used_threshold),
                }

            lps = ras.copy()
            lps[:, 0] *= -1.0
            lps[:, 1] *= -1.0
            return {"points_lps": lps, "threshold_hu": float(used_threshold)}

        # -- Cached overrides for hot annulus methods ----------------------
        # The DeepCoreAnnulusMixin shim re-reads the Slicer volume on every
        # call.  These overrides use the already-extracted ctx data when a
        # _ContextVolumeProxy is passed.

        @staticmethod
        def _volume_array_kji(volume_node):
            if isinstance(volume_node, _ContextVolumeProxy):
                return volume_node._arr_kji
            return _slicer_array_kji(volume_node)

        @staticmethod
        def _ras_to_ijk_fn_for_volume(volume_node):
            if isinstance(volume_node, _ContextVolumeProxy) and volume_node._ras_to_ijk_fn is not None:
                return volume_node._ras_to_ijk_fn
            return _slicer_ras_to_ijk_fn(volume_node)

        # Note: the base DeepCoreAnnulusMixin declares some of these as
        # @classmethod / @staticmethod, so callers may use cls.method() or
        # self.method().  All overrides here must work in both call styles
        # by accepting *args and dispatching on argument types.

        @classmethod
        def _scan_reference_hu_values(cls, volume_node, lower_hu=-500.0, upper_hu=2500.0):
            arr_kji = cls._volume_array_kji(volume_node)
            if arr_kji is None:
                return None
            values = np.asarray(arr_kji, dtype=float).reshape(-1)
            keep = np.isfinite(values) & (values > float(lower_hu))
            if upper_hu is not None:
                keep &= values < float(upper_hu)
            ref = np.asarray(values[keep], dtype=float).reshape(-1)
            if ref.size <= 0:
                return None
            ref.sort()
            return ref

        @classmethod
        def _depth_at_ras_with_volume(cls, volume_node, depth_map_kji, point_ras):
            fn = cls._ras_to_ijk_fn_for_volume(volume_node)
            if depth_map_kji is None or fn is None:
                return None
            ijk = fn(point_ras)
            i, j, k = int(round(float(ijk[0]))), int(round(float(ijk[1]))), int(round(float(ijk[2])))
            if k < 0 or j < 0 or i < 0 or k >= depth_map_kji.shape[0] or j >= depth_map_kji.shape[1] or i >= depth_map_kji.shape[2]:
                return None
            val = float(depth_map_kji[k, j, i])
            return val if np.isfinite(val) else None

        @classmethod
        def _cross_section_annulus_stats_hu(
            cls, volume_node, center_ras, axis_ras,
            annulus_inner_mm=3.0, annulus_outer_mm=4.0,
            radial_steps=2, angular_samples=12,
        ):
            arr_kji = cls._volume_array_kji(volume_node)
            fn = cls._ras_to_ijk_fn_for_volume(volume_node)
            if arr_kji is None or fn is None:
                return {"mean_hu": None, "median_hu": None, "sample_count": 0}
            center = np.asarray(center_ras if center_ras is not None else [0, 0, 0], dtype=float).reshape(3)
            _axis, u, v = cls._orthonormal_basis_for_axis(axis_ras)
            radii = np.linspace(float(annulus_inner_mm), float(annulus_outer_mm), int(max(1, radial_steps)))
            angles = np.linspace(0, 2 * np.pi, int(max(4, angular_samples)), endpoint=False)
            samples = []
            for r in radii.tolist():
                for theta in angles.tolist():
                    offset = float(r) * np.cos(float(theta)) * u + float(r) * np.sin(float(theta)) * v
                    ijk = fn(center + offset)
                    i, j, k = int(round(float(ijk[0]))), int(round(float(ijk[1]))), int(round(float(ijk[2])))
                    if 0 <= k < arr_kji.shape[0] and 0 <= j < arr_kji.shape[1] and 0 <= i < arr_kji.shape[2]:
                        val = float(arr_kji[k, j, i])
                        if np.isfinite(val):
                            samples.append(val)
            if not samples:
                return {"mean_hu": None, "median_hu": None, "sample_count": 0}
            sa = np.asarray(samples, dtype=float)
            return {"mean_hu": float(np.mean(sa)), "median_hu": float(np.median(sa)), "sample_count": int(sa.size)}

        @classmethod
        def _cross_section_annulus_mean_ct_hu(
            cls, volume_node, center_ras, axis_ras,
            annulus_inner_mm=3.0, annulus_outer_mm=4.0,
            radial_steps=2, angular_samples=12,
        ):
            return cls._cross_section_annulus_stats_hu(
                volume_node, center_ras, axis_ras,
                annulus_inner_mm, annulus_outer_mm,
                radial_steps, angular_samples,
            )

        # ---------------------------------------------------------------
        # Mask stage
        # ---------------------------------------------------------------

        def _run_mask(self, ctx: DetectionContext, cfg) -> dict[str, Any]:
            if sitk is None:
                raise RuntimeError("SimpleITK is required for deep-core mask stage.")
            dc = _parse_deep_core_config(cfg)
            mcfg = dc.mask

            arr_kji = np.asarray(ctx["arr_kji"], dtype=np.float32)
            spacing_xyz = tuple(float(v) for v in ctx["spacing_xyz"])

            arr_clip = np.asarray(arr_kji, dtype=np.float32)
            if np.isfinite(float(mcfg.hull_clip_hu)):
                arr_clip = np.minimum(arr_clip, float(mcfg.hull_clip_hu))
            hull_img = sitk.GetImageFromArray(arr_clip.astype(np.float32))
            hull_img.SetSpacing(spacing_xyz)
            if float(mcfg.hull_sigma_mm) > 0.0:
                hull_img = sitk.SmoothingRecursiveGaussian(hull_img, float(mcfg.hull_sigma_mm))
            smoothed_hull_kji = sitk.GetArrayFromImage(hull_img).astype(np.float32)

            binary = sitk.BinaryThreshold(
                hull_img,
                lowerThreshold=float(mcfg.hull_threshold_hu),
                upperThreshold=float(max(float(np.nanmax(smoothed_hull_kji)), float(mcfg.hull_threshold_hu) + 1.0)),
                insideValue=1, outsideValue=0,
            )
            if int(mcfg.hull_open_vox) > 0:
                binary = sitk.BinaryMorphologicalOpening(binary, [int(mcfg.hull_open_vox)] * 3, sitk.sitkBall)
            hull_lcc = largest_component_binary(binary)
            if hull_lcc is None:
                raise RuntimeError("Failed to build a non-air hull mask from the selected CT.")
            if int(mcfg.hull_close_vox) > 0:
                hull_lcc = sitk.BinaryMorphologicalClosing(hull_lcc, [int(mcfg.hull_close_vox)] * 3, sitk.sitkBall)
            hull_mask_kji = sitk.GetArrayFromImage(hull_lcc).astype(bool)

            # head_distance source: hull ∪ internal-air. The hull gives us a
            # cleaned structural envelope (robust to flood-fill leaks through
            # foramina/orbits), while internal-air fills sinuses and other
            # trapped air pockets so they don't eat the shrink from the inside.
            outside_air_kji = build_outside_air_mask_kji(
                arr_kji, air_threshold_hu=float(mcfg.hull_threshold_hu)
            )
            if np.count_nonzero(outside_air_kji) > 0:
                air_mask_kji = arr_kji < float(mcfg.hull_threshold_hu)
                internal_air_kji = np.logical_and(air_mask_kji, np.logical_not(outside_air_kji))
                head_mask_for_edt = np.logical_or(hull_mask_kji, internal_air_kji)
            else:
                # Degenerate case (head fills the volume, no boundary air):
                # fall back to hull-only EDT, matching legacy behavior.
                head_mask_for_edt = hull_mask_kji
            head_distance_map_kji = np.asarray(
                compute_head_distance_map_kji(head_mask_for_edt, spacing_xyz=spacing_xyz),
                dtype=np.float32,
            )
            deep_core_mask_kji = np.logical_and(
                hull_mask_kji,
                head_distance_map_kji >= float(max(0.0, mcfg.deep_core_shrink_mm)),
            )

            metal_mask_kji = np.asarray(arr_kji >= float(mcfg.metal_threshold_hu), dtype=bool)
            bolt_metal_mask_kji = np.asarray(
                arr_kji >= float(mcfg.bolt_metal_threshold_hu), dtype=bool
            )
            metal_grown_mask_kji = metal_mask_kji.copy()
            if int(mcfg.metal_grow_vox) > 0:
                metal_img = sitk.GetImageFromArray(metal_mask_kji.astype(np.uint8))
                metal_grown_mask_kji = sitk.GetArrayFromImage(
                    sitk.BinaryDilate(metal_img, [int(mcfg.metal_grow_vox)] * 3, sitk.sitkBall)
                ).astype(bool)

            return {
                "hull_mask_kji": hull_mask_kji,
                "deep_core_mask_kji": deep_core_mask_kji,
                "metal_mask_kji": metal_mask_kji,
                "bolt_metal_mask_kji": bolt_metal_mask_kji,
                "metal_grown_mask_kji": metal_grown_mask_kji,
                "deep_seed_raw_mask_kji": np.logical_and(metal_mask_kji, deep_core_mask_kji),
                "deep_seed_mask_kji": np.logical_and(metal_grown_mask_kji, deep_core_mask_kji),
                "head_distance_map_kji": head_distance_map_kji,
                "smoothed_hull_kji": smoothed_hull_kji,
                "stats": {
                    "hull_voxels": int(np.count_nonzero(hull_mask_kji)),
                    "deep_core_voxels": int(np.count_nonzero(deep_core_mask_kji)),
                    "metal_voxels": int(np.count_nonzero(metal_mask_kji)),
                    "bolt_metal_voxels": int(np.count_nonzero(bolt_metal_mask_kji)),
                },
            }

        # ---------------------------------------------------------------
        # Bolt detection stage (RANSAC on saturation-bright metal)
        # ---------------------------------------------------------------

        def _run_bolt_detection(self, ctx: DetectionContext, mask: dict, cfg) -> dict[str, Any]:
            from postop_ct_localization.deep_core_bolt_ransac import (
                BoltRansacConfig,
                find_bolt_candidates,
            )

            bcfg = cfg.bolt
            if not bool(bcfg.enabled):
                return {"candidates": [], "stats": {"enabled": False}}

            arr_kji = np.asarray(ctx["arr_kji"], dtype=np.float32)
            ijk_fn = ctx.get("ijk_kji_to_ras_fn")
            ras_to_ijk_fn = ctx.get("ras_to_ijk_fn")
            if ijk_fn is None or ras_to_ijk_fn is None:
                return {"candidates": [], "stats": {"enabled": True, "error": "no coordinate fns"}}

            ransac_cfg = BoltRansacConfig(
                span_min_mm=float(bcfg.span_min_mm),
                span_max_mm=float(bcfg.span_max_mm),
                inlier_tol_mm=float(bcfg.inlier_tol_mm),
                min_inliers=int(bcfg.min_inliers),
                fill_frac_min=float(bcfg.fill_frac_min),
                max_gap_mm=float(bcfg.max_gap_mm),
                shell_min_mm=float(bcfg.shell_min_mm),
                shell_max_mm=float(bcfg.shell_max_mm),
                axis_depth_delta_mm=float(bcfg.axis_depth_delta_mm),
                support_overlap_frac=float(bcfg.support_overlap_frac),
                collinear_angle_deg=float(bcfg.collinear_angle_deg),
                collinear_perp_mm=float(bcfg.collinear_perp_mm),
                max_lines=int(bcfg.max_lines),
                n_samples=int(bcfg.n_samples),
            )

            candidates = find_bolt_candidates(
                arr_kji=arr_kji,
                bolt_metal_mask_kji=mask["bolt_metal_mask_kji"],
                head_distance_map_kji=mask["head_distance_map_kji"],
                ijk_kji_to_ras_fn=ijk_fn,
                ras_to_ijk_fn=ras_to_ijk_fn,
                cfg=ransac_cfg,
            )
            return {
                "candidates": candidates,
                "stats": {
                    "enabled": True,
                    "n_candidates": len(candidates),
                    "bolt_metal_voxels": int(mask.get("stats", {}).get("bolt_metal_voxels", 0)),
                },
            }

        # ---------------------------------------------------------------
        # Support stage
        # ---------------------------------------------------------------

        def _run_support(self, ctx: DetectionContext, mask: dict, cfg) -> dict[str, Any]:
            dc = _parse_deep_core_config(cfg)
            scfg = dc.support
            proxy = _ContextVolumeProxy(ctx)
            arr_kji = np.asarray(ctx["arr_kji"], dtype=np.float32)
            ijk_fn = ctx.get("ijk_kji_to_ras_fn")

            raw_blob_result = extract_blob_candidates(
                metal_mask_kji=mask["deep_seed_raw_mask_kji"],
                arr_kji=arr_kji,
                depth_map_kji=mask["head_distance_map_kji"],
                ijk_kji_to_ras_fn=ijk_fn,
            )
            grown_blob_result = extract_blob_candidates(
                metal_mask_kji=mask["deep_seed_mask_kji"],
                arr_kji=arr_kji,
                depth_map_kji=mask["head_distance_map_kji"],
                ijk_kji_to_ras_fn=ijk_fn,
            )
            sample_payload = self._build_support_atom_payload(
                volume_node=proxy,
                labels_kji=raw_blob_result.get("labels_kji"),
                blobs=list(raw_blob_result.get("blobs") or []),
                support_spacing_mm=float(scfg.support_spacing_mm),
                component_min_elongation=float(scfg.component_min_elongation),
                line_atom_diameter_max_mm=float(scfg.line_atom_diameter_max_mm),
                line_atom_min_span_mm=float(scfg.line_atom_min_span_mm),
                line_atom_min_pca_dominance=float(scfg.line_atom_min_pca_dominance),
                contact_component_diameter_max_mm=float(scfg.contact_component_diameter_max_mm),
                support_cube_size_mm=float(scfg.support_cube_size_mm),
                head_distance_map_kji=mask["head_distance_map_kji"],
                annulus_config=dc.annulus,
                internal_config=dc.internal,
            )

            support_atoms = list(sample_payload.get("support_atoms") or [])
            blob_axes = {
                int(k): [float(v) for v in np.asarray(val, dtype=float).reshape(3).tolist()]
                for k, val in dict(sample_payload.get("axes_ras_by_id") or {}).items()
                if int(k) > 0 and np.asarray(val, dtype=float).reshape(-1).size == 3
            }
            blob_elongation = {
                int(k): float(v)
                for k, v in dict(sample_payload.get("elongation_by_id") or {}).items()
                if int(k) > 0
            }

            return {
                "support_atoms": support_atoms,
                "blob_sample_points_ras": np.asarray(sample_payload.get("points_ras"), dtype=float).reshape(-1, 3),
                "blob_sample_blob_ids": np.asarray(sample_payload.get("blob_ids"), dtype=np.int32).reshape(-1),
                "blob_sample_atom_ids": np.asarray(sample_payload.get("atom_ids"), dtype=np.int32).reshape(-1),
                "blob_axes_ras_by_id": blob_axes,
                "blob_elongation_by_id": blob_elongation,
                "blob_class_by_id": dict(sample_payload.get("blob_class_by_id") or {}),
                # Mask fields forwarded for downstream stages
                "head_distance_map_kji": mask["head_distance_map_kji"],
                "metal_grown_mask_kji": mask["metal_grown_mask_kji"],
                "metal_mask_kji": mask["metal_mask_kji"],
                "hull_mask_kji": mask["hull_mask_kji"],
                # Raw data for debug visualization
                "blob_labelmap_kji": build_blob_labelmap(raw_blob_result.get("labels_kji")),
                "complex_blob_chain_rows": list(sample_payload.get("complex_blob_chain_rows") or []),
                "contact_chain_rows": list(sample_payload.get("contact_chain_rows") or []),
                "contact_chain_debug_rows": list(sample_payload.get("contact_chain_debug_rows") or []),
                "line_blob_sample_points_ras": np.asarray(sample_payload.get("line_blob_points_ras"), dtype=float).reshape(-1, 3),
                "contact_blob_sample_points_ras": np.asarray(sample_payload.get("contact_blob_points_ras"), dtype=float).reshape(-1, 3),
                "complex_blob_sample_points_ras": np.asarray(sample_payload.get("complex_blob_points_ras"), dtype=float).reshape(-1, 3),
                "raw_blob_result": raw_blob_result,
                "grown_blob_result": grown_blob_result,
                "stats": {
                    "atom_count": len(support_atoms),
                    "raw_blob_count": int(raw_blob_result.get("blob_count_total", 0)),
                    "grown_blob_count": int(grown_blob_result.get("blob_count_total", 0)),
                },
            }

        # ---------------------------------------------------------------
        # Proposal sub-stages — delegate to mixin methods
        # ---------------------------------------------------------------

        def _run_candidates(self, proxy, support, dc):
            return self._build_deep_core_raw_proposals_stage(
                volume_node=proxy,
                support_result=support,
                mask_config=dc.mask,
                support_config=dc.support,
                proposal_config=dc.proposal,
                annulus_config=dc.annulus,
                internal_config=dc.internal,
            )

        def _run_annulus_rejection(self, proxy, support, proposals, dc):
            return self._apply_pre_extension_annulus_rejection_stage(
                volume_node=proxy,
                support_result=support,
                proposal_payload=proposals,
                annulus_config=dc.annulus,
            )

        def _run_extension(self, proxy, support, proposals, dc):
            return self._extend_deep_core_proposals_stage(
                volume_node=proxy,
                support_result=support,
                proposal_payload=proposals,
                mask_config=dc.mask,
                support_config=dc.support,
                proposal_config=dc.proposal,
                annulus_config=dc.annulus,
                internal_config=dc.internal,
            )

        def _run_final_rejection(self, proxy, support, proposals, dc):
            return self._apply_final_deep_core_rejection_stage(
                volume_node=proxy,
                support_result=support,
                proposal_payload=proposals,
                mask_config=dc.mask,
                proposal_config=dc.proposal,
            )

        # ---------------------------------------------------------------
        # Model fit (Phase B) — replaces extension when enabled
        # ---------------------------------------------------------------

        def _get_electrode_library(self):
            """Load and cache the electrode library on the pipeline instance."""
            lib = getattr(self, "_electrode_library", None)
            if lib is None:
                from rosa_core.electrode_models import load_electrode_library
                lib = load_electrode_library()
                self._electrode_library = lib
            return lib

        def _run_model_fit(self, ctx, support, proposals, dc, bolt_output=None, mask=None):
            """Phase B: trajectory reconstruction (axis refinement, extension,
            bone↔brain interface, rejection). See
            ``docs/PHASE_B_REDESIGN.md``.

            When ``model_fit.use_bolt_detection`` is set and ``bolt_output``
            contains bolt candidates, each bolt is turned into one synthetic
            proposal (``bolt_bridged`` when colinear support atoms are found,
            otherwise ``bolt_only``) and merged into the proposal list. The
            blob_sample_points_ras array is extended with the bolt inlier
            points so bolt-only proposals still have a non-empty initial
            cloud via ``token_indices``.
            """
            from postop_ct_localization.deep_core_model_fit import (
                filter_models_by_family,
                run_model_fit_group,
            )

            mfg = dc.model_fit
            library = self._get_electrode_library()
            models = filter_models_by_family(library, tuple(mfg.families))

            input_proposals = list(proposals.get("proposals") or [])
            for p in input_proposals:
                p.setdefault("source", "atoms_only")

            arr_kji = np.asarray(ctx["arr_kji"], dtype=float)
            ras_to_ijk_fn = ctx["ras_to_ijk_fn"]

            blob_points = support.get("blob_sample_points_ras")
            blob_points_arr = (
                np.asarray(blob_points, dtype=float).reshape(-1, 3)
                if blob_points is not None else np.zeros((0, 3), dtype=float)
            )
            support_atoms = list(support.get("support_atoms") or [])

            bolt_stats = {"bolt_bridged": 0, "bolt_only": 0}
            if bool(mfg.use_bolt_detection) and bolt_output is not None:
                bolt_candidates = list(bolt_output.get("candidates") or [])
                if bolt_candidates:
                    from postop_ct_localization import deep_core_axis_reconstruction as axr

                    # Reconstruct the bolt-metal point cloud once so we can
                    # slice each bolt's inliers and add them to
                    # ``blob_sample_points_ras``.
                    bolt_metal_mask = None
                    if mask is not None:
                        bolt_metal_mask = mask.get("bolt_metal_mask_kji")
                    if bolt_metal_mask is not None:
                        ijk_fn = ctx.get("ijk_kji_to_ras_fn")
                        idx = np.argwhere(np.asarray(bolt_metal_mask, dtype=bool))
                        bolt_cloud = np.asarray(ijk_fn(idx.astype(float)), dtype=float)
                    else:
                        bolt_cloud = np.zeros((0, 3), dtype=float)

                    new_blob_points = [blob_points_arr]
                    next_tok_idx = int(blob_points_arr.shape[0])
                    bridge_tol = float(getattr(mfg, "bolt_bridge_radial_tol_mm", 2.5))
                    for bc in bolt_candidates:
                        inlier_pts = bolt_cloud[bc.support_mask] if bolt_cloud.size else np.zeros((0, 3))
                        if inlier_pts.shape[0] < 3:
                            continue
                        # Build an AxisFit so we can reuse the reabsorb helper
                        # with the bolt's precise RANSAC axis as the seed.
                        fit = axr.refine_axis_from_cloud(inlier_pts, seed_axis=bc.axis_ras)
                        if fit is None:
                            continue
                        # Temporarily override the radial tolerance for the
                        # bridge query so we pick up atoms a bit further from
                        # the axis than the reabsorb default (2.5mm by default).
                        class _Shim:
                            pass
                        shim = _Shim()
                        shim.reabsorb_radial_tol_mm = bridge_tol
                        shim.reabsorb_angle_tol_deg = float(
                            getattr(mfg, "reabsorb_angle_tol_deg", 5.0)
                        )
                        atom_ids = axr.reabsorb_colinear_atoms(
                            fit, support_atoms, shim, already_absorbed_ids=set()
                        )

                        # Stash bolt inlier points at the end of
                        # blob_sample_points_ras and reference them via
                        # token_indices. This is how Phase B's
                        # _gather_initial_cloud receives extra cloud points
                        # without touching support_atoms.
                        token_start = next_tok_idx
                        new_blob_points.append(inlier_pts)
                        next_tok_idx += inlier_pts.shape[0]
                        token_indices = list(range(token_start, next_tok_idx))

                        source = "bolt_bridged" if atom_ids else "bolt_only"
                        t_min = float(fit.t_min)
                        t_max = float(fit.t_max)
                        start_ras = fit.center + fit.axis * t_min
                        end_ras = fit.center + fit.axis * t_max
                        synth = {
                            "source": source,
                            "atom_id_list": [int(v) for v in atom_ids],
                            "token_indices": token_indices,
                            "axis_ras": [float(v) for v in fit.axis],
                            "start_ras": [float(v) for v in start_ras],
                            "end_ras": [float(v) for v in end_ras],
                            "bolt_seed": True,
                            "bolt_center_ras": [float(v) for v in bc.center_ras],
                            "bolt_span_mm": float(bc.span_mm),
                            "bolt_hd_center_mm": float(bc.hd_center_mm),
                            "bolt_fill_frac": float(bc.fill_frac),
                        }
                        input_proposals.append(synth)
                        bolt_stats[source] += 1

                    if next_tok_idx > blob_points_arr.shape[0]:
                        blob_points_arr = np.concatenate(new_blob_points, axis=0)

            result = run_model_fit_group(
                proposals=input_proposals,
                arr_kji=arr_kji,
                ras_to_ijk_fn=ras_to_ijk_fn,
                head_distance_map_kji=support.get("head_distance_map_kji"),
                library_models=models,
                cfg=mfg,
                metal_mask_kji=support.get("metal_mask_kji"),
                hull_mask_kji=support.get("hull_mask_kji"),
                support_atoms=support_atoms,
                blob_sample_points_ras=blob_points_arr,
            )
            out = dict(proposals)
            out["proposals"] = list(result["accepted_proposals"])
            stats_out = dict(result.get("stats") or {})
            stats_out.update(bolt_stats)
            out["model_fit_stats"] = stats_out
            return out

        # ---------------------------------------------------------------
        # Result conversion
        # ---------------------------------------------------------------

        @staticmethod
        def _proposals_to_trajectories(proposal_payload: dict) -> list[dict[str, Any]]:
            """Convert internal proposal dicts to DetectionResult trajectory format."""
            proposals = list(proposal_payload.get("proposals") or [])
            trajectories = []
            for p in proposals:
                start = p.get("start_ras")
                end = p.get("end_ras")
                if start is None or end is None:
                    continue
                t = {
                    "start_ras": [float(v) for v in start],
                    "end_ras": [float(v) for v in end],
                    "span_mm": float(p.get("span_mm", 0.0)),
                    "proposal_family": str(p.get("proposal_family", "")),
                    "inlier_count": int(p.get("inlier_count", 0)),
                    "confidence": float(p.get("score", 0.0)),
                }
                if p.get("axis_ras") is not None:
                    t["axis_ras"] = [float(v) for v in p["axis_ras"]]
                if p.get("bolt_ras") is not None:
                    t["bolt_ras"] = [float(v) for v in p["bolt_ras"]]
                if p.get("intracranial_span_mm") is not None:
                    t["intracranial_span_mm"] = float(p["intracranial_span_mm"])
                if p.get("bolt_extent_mm") is not None:
                    t["bolt_extent_mm"] = float(p["bolt_extent_mm"])
                if p.get("explained_atom_ids") is not None:
                    t["explained_atom_ids"] = [int(v) for v in p["explained_atom_ids"]]
                if p.get("annulus_mean_ct_hu") is not None:
                    t["annulus_mean_ct_hu"] = float(p["annulus_mean_ct_hu"])
                if p.get("best_model_id") is not None:
                    t["best_model_id"] = str(p["best_model_id"])
                    t["best_model_score"] = float(p.get("best_model_score", 0.0))
                if p.get("best_model_n_hits") is not None:
                    t["best_model_n_hits"] = int(p["best_model_n_hits"])
                    t["best_model_in_brain_total"] = int(p.get("best_model_in_brain_total", 0))
                    t["best_model_contact_count"] = int(p.get("best_model_contact_count", 0))
                if p.get("model_contact_positions_ras") is not None:
                    t["model_contact_positions_ras"] = list(p["model_contact_positions_ras"])
                t["_proposal_index"] = len(trajectories)
                trajectories.append(t)
            return trajectories

        # ---------------------------------------------------------------
        # Main entry points
        # ---------------------------------------------------------------

        def run(self, ctx: DetectionContext) -> DetectionResult:
            t_start = time.perf_counter()
            result = self.make_result(ctx)
            diag = self.diagnostics(result)
            cfg = _parse_deep_core_config(self._config(ctx))

            try:
                proxy = _ContextVolumeProxy(ctx)

                mask = self.run_stage(ctx=ctx, result=result, diagnostics=diag,
                    stage_name="mask", fn=lambda: self._run_mask(ctx, cfg))

                bolt = self.run_stage(ctx=ctx, result=result, diagnostics=diag,
                    stage_name="bolt_detection",
                    fn=lambda: self._run_bolt_detection(ctx, mask, cfg))

                support = self.run_stage(ctx=ctx, result=result, diagnostics=diag,
                    stage_name="support", fn=lambda: self._run_support(ctx, mask, cfg))

                proposals = self.run_stage(ctx=ctx, result=result, diagnostics=diag,
                    stage_name="candidate_generation",
                    fn=lambda: self._run_candidates(proxy, support, cfg))

                proposals = self.run_stage(ctx=ctx, result=result, diagnostics=diag,
                    stage_name="annulus_rejection",
                    fn=lambda: self._run_annulus_rejection(proxy, support, proposals, cfg))

                if cfg.model_fit.enabled:
                    proposals = self.run_stage(ctx=ctx, result=result, diagnostics=diag,
                        stage_name="model_fit",
                        fn=lambda: self._run_model_fit(ctx, support, proposals, cfg,
                                                        bolt_output=bolt, mask=mask))
                    # model_fit's group assignment + hard-reject already
                    # handles dedup and rejection; skip final_rejection.
                else:
                    proposals = self.run_stage(ctx=ctx, result=result, diagnostics=diag,
                        stage_name="extension",
                        fn=lambda: self._run_extension(proxy, support, proposals, cfg))
                    proposals = self.run_stage(ctx=ctx, result=result, diagnostics=diag,
                        stage_name="final_rejection",
                        fn=lambda: self._run_final_rejection(proxy, support, proposals, cfg))

                result["trajectories"] = self._proposals_to_trajectories(proposals)

                diag.set_count("proposal_count", len(result["trajectories"]))
                diag.set_count("atom_count", int(support.get("stats", {}).get("atom_count", 0)))

                # Cache heavy data on self — NOT in result (sanitize_result
                # would try to JSON-serialize numpy arrays).
                self._last_mask_output = mask
                self._last_bolt_output = bolt
                self._last_support_output = support
                self._last_proposal_payload = proposals

            except Exception as exc:
                self.fail(ctx=ctx, result=result, diagnostics=diag,
                          stage="unknown", exc=exc)
                # Ensure cached outputs exist even on failure
                if not hasattr(self, "_last_proposal_payload"):
                    self._last_proposal_payload = {"proposals": []}

            return self.finalize(result, diag, t_start)

        def run_debug(self, ctx: DetectionContext) -> DetectionResult:
            """Run mask + support stages only (for visualization)."""
            t_start = time.perf_counter()
            result = self.make_result(ctx)
            diag = self.diagnostics(result)
            cfg = _parse_deep_core_config(self._config(ctx))

            try:
                mask = self.run_stage(ctx=ctx, result=result, diagnostics=diag,
                    stage_name="mask", fn=lambda: self._run_mask(ctx, cfg))

                support = self.run_stage(ctx=ctx, result=result, diagnostics=diag,
                    stage_name="support", fn=lambda: self._run_support(ctx, mask, cfg))

                # Cache heavy data on self — NOT in result["meta"] (sanitize_result
                # would try to JSON-serialize numpy arrays, which is extremely slow).
                self._last_mask_output = mask
                self._last_support_output = support

                diag.set_count("atom_count", int(support.get("stats", {}).get("atom_count", 0)))

            except Exception as exc:
                self.fail(ctx=ctx, result=result, diagnostics=diag,
                          stage="unknown", exc=exc)

            return self.finalize(result, diag, t_start)

    return DeepCoreV1Pipeline


DeepCoreV1Pipeline = _make_pipeline_class()
