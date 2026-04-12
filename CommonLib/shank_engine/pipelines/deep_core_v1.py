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
from shank_core.masking import compute_head_distance_map_kji, largest_component_binary

from ..contracts import DetectionContext, DetectionResult
from ..pipelines.base import BaseDetectionPipeline


# ---------------------------------------------------------------------------
# Volume proxy — lets mixin code treat DetectionContext data like a volume_node
# ---------------------------------------------------------------------------

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
    if dotted:
        return defaults.with_updates(dotted)
    return defaults


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
            from postop_ct_localization.deep_core_volume import SlicerVolumeAccessor
            return SlicerVolumeAccessor().ijk_kji_to_ras_points(volume_node, ijk_kji)

        def extract_threshold_candidates_lps(self, volume_node, threshold, **kwargs):
            """Bridge for mixin code that calls self.extract_threshold_candidates_lps."""
            if isinstance(volume_node, _ContextVolumeProxy) and volume_node._volume_node is not None:
                volume_node = volume_node._volume_node
            from postop_ct_localization.deep_core_volume import SlicerVolumeAccessor
            return SlicerVolumeAccessor().extract_threshold_candidates_lps(
                volume_node=volume_node, threshold=threshold, **kwargs
            )

        # -- Cached overrides for hot annulus methods ----------------------
        # The DeepCoreAnnulusMixin shim creates a new SlicerVolumeAccessor
        # and re-reads the volume on every call.  These overrides use the
        # already-extracted ctx data when a _ContextVolumeProxy is passed.

        @staticmethod
        def _volume_array_kji(volume_node):
            if isinstance(volume_node, _ContextVolumeProxy):
                return volume_node._arr_kji
            from postop_ct_localization.deep_core_volume import SlicerVolumeAccessor
            return SlicerVolumeAccessor().array_kji(volume_node)

        @staticmethod
        def _ras_to_ijk_fn_for_volume(volume_node):
            if isinstance(volume_node, _ContextVolumeProxy) and volume_node._ras_to_ijk_fn is not None:
                return volume_node._ras_to_ijk_fn
            from postop_ct_localization.deep_core_volume import SlicerVolumeAccessor
            return SlicerVolumeAccessor().ras_to_ijk_fn(volume_node)

        def _scan_reference_hu_values(self, volume_node, lower_hu=-500.0, upper_hu=2500.0):
            arr_kji = self._volume_array_kji(volume_node)
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

        def _depth_at_ras_with_volume(self, volume_node, depth_map_kji, point_ras):
            fn = self._ras_to_ijk_fn_for_volume(volume_node)
            if depth_map_kji is None or fn is None:
                return None
            ijk = fn(point_ras)
            i, j, k = int(round(float(ijk[0]))), int(round(float(ijk[1]))), int(round(float(ijk[2])))
            if k < 0 or j < 0 or i < 0 or k >= depth_map_kji.shape[0] or j >= depth_map_kji.shape[1] or i >= depth_map_kji.shape[2]:
                return None
            val = float(depth_map_kji[k, j, i])
            return val if np.isfinite(val) else None

        def _cross_section_annulus_stats_hu(
            self, volume_node, center_ras, axis_ras,
            annulus_inner_mm=3.0, annulus_outer_mm=4.0,
            radial_steps=2, angular_samples=12,
        ):
            arr_kji = self._volume_array_kji(volume_node)
            fn = self._ras_to_ijk_fn_for_volume(volume_node)
            if arr_kji is None or fn is None:
                return {"mean_hu": None, "median_hu": None, "sample_count": 0}
            center = np.asarray(center_ras if center_ras is not None else [0, 0, 0], dtype=float).reshape(3)
            _axis, u, v = self._orthonormal_basis_for_axis(axis_ras)
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

        def _cross_section_annulus_mean_ct_hu(
            self, volume_node, center_ras, axis_ras,
            annulus_inner_mm=3.0, annulus_outer_mm=4.0,
            radial_steps=2, angular_samples=12,
        ):
            return self._cross_section_annulus_stats_hu(
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

            head_distance_map_kji = np.asarray(
                compute_head_distance_map_kji(hull_mask_kji, spacing_xyz=spacing_xyz),
                dtype=np.float32,
            )
            deep_core_mask_kji = np.logical_and(
                hull_mask_kji,
                head_distance_map_kji >= float(max(0.0, mcfg.deep_core_shrink_mm)),
            )

            metal_mask_kji = np.asarray(arr_kji >= float(mcfg.metal_threshold_hu), dtype=bool)
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
                "metal_grown_mask_kji": metal_grown_mask_kji,
                "deep_seed_raw_mask_kji": np.logical_and(metal_mask_kji, deep_core_mask_kji),
                "deep_seed_mask_kji": np.logical_and(metal_grown_mask_kji, deep_core_mask_kji),
                "head_distance_map_kji": head_distance_map_kji,
                "smoothed_hull_kji": smoothed_hull_kji,
                "stats": {
                    "hull_voxels": int(np.count_nonzero(hull_mask_kji)),
                    "deep_core_voxels": int(np.count_nonzero(deep_core_mask_kji)),
                    "metal_voxels": int(np.count_nonzero(metal_mask_kji)),
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
                # Raw data for debug visualization
                "blob_labelmap_kji": build_blob_labelmap(raw_blob_result.get("labels_kji")),
                "complex_blob_chain_rows": list(sample_payload.get("complex_blob_chain_rows") or []),
                "contact_chain_rows": list(sample_payload.get("contact_chain_rows") or []),
                "contact_chain_debug_rows": list(sample_payload.get("contact_chain_debug_rows") or []),
                "line_blob_points_ras": np.asarray(sample_payload.get("line_blob_points_ras"), dtype=float).reshape(-1, 3),
                "contact_blob_points_ras": np.asarray(sample_payload.get("contact_blob_points_ras"), dtype=float).reshape(-1, 3),
                "complex_blob_points_ras": np.asarray(sample_payload.get("complex_blob_points_ras"), dtype=float).reshape(-1, 3),
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
                if p.get("annulus_mean_ct_hu") is not None:
                    t["annulus_mean_ct_hu"] = float(p["annulus_mean_ct_hu"])
                if p.get("best_model_id") is not None:
                    t["best_model_id"] = str(p["best_model_id"])
                    t["best_model_score"] = float(p.get("best_model_score", 0.0))
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

                support = self.run_stage(ctx=ctx, result=result, diagnostics=diag,
                    stage_name="support", fn=lambda: self._run_support(ctx, mask, cfg))

                proposals = self.run_stage(ctx=ctx, result=result, diagnostics=diag,
                    stage_name="candidate_generation",
                    fn=lambda: self._run_candidates(proxy, support, cfg))

                proposals = self.run_stage(ctx=ctx, result=result, diagnostics=diag,
                    stage_name="annulus_rejection",
                    fn=lambda: self._run_annulus_rejection(proxy, support, proposals, cfg))

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
                self._last_support_output = support
                self._last_proposal_payload = proposals

            except Exception as exc:
                self.fail(ctx=ctx, result=result, diagnostics=diag,
                          stage="unknown", exc=exc)

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
