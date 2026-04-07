import math
import json
import os

try:
    import numpy as np
except ImportError:
    np = None

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

from __main__ import ctk, qt, slicer, vtk

from shank_core.blob_candidates import build_blob_labelmap, extract_blob_candidates
from shank_core.io import image_ijk_ras_matrices, kji_to_ras_points, kji_to_ras_points_matrix, ras_to_ijk_float_matrix
from shank_core.masking import build_preview_masks

from .constants import DE_NOVO_MODE_SPECS, GUIDED_SOURCE_OPTIONS

class DeepCoreProposalLogicMixin:
    def build_deep_core_proposals(
        self,
        volume_node,
        token_points_ras,
        token_blob_ids=None,
        token_atom_ids=None,
        blob_axes_ras_by_id=None,
        blob_elongation_by_id=None,
        support_atoms=None,
        neighbor_max_mm=8.0,
        neighbor_min_mm=1.0,
        neighbor_axis_angle_deg=30.0,
        grow_turn_angle_deg=30.0,
        inlier_radius_mm=2.0,
        axial_bin_mm=2.5,
        min_span_mm=16.0,
        min_inlier_count=5,
        min_chain_tokens=5,
        max_seed_edges=1200,
        max_pair_seeds=900,
        max_output_proposals=None,
        guided_threshold_hu=1900.0,
        guided_head_mask_threshold_hu=-500.0,
        guided_roi_radius_mm=5.0,
        guided_max_angle_deg=12.0,
        guided_max_depth_shift_mm=2.0,
        guided_fit_mode="deep_anchor_v2",
        guided_max_residual_mm=2.0,
        head_distance_map_kji=None,
        deep_core_shrink_mm=15.0,
        short_float_reject_span_mm=20.0,
        short_float_edge_tol_mm=2.5,
        guided_candidate_mask_kji=None,
        outward_support_check_span_mm=45.0,
        outward_support_radius_mm=2.5,
        outward_support_search_mm=20.0,
        outward_support_min_extension_mm=6.0,
        outward_support_min_depth_gain_mm=4.0,
    ):
        prepared = self._prepare_support_atom_inputs(
            support_atoms=support_atoms,
            token_points_ras=token_points_ras,
            token_blob_ids=token_blob_ids,
            token_atom_ids=token_atom_ids,
            blob_axes_ras_by_id=blob_axes_ras_by_id,
            blob_elongation_by_id=blob_elongation_by_id,
        )
        pts = np.asarray(prepared.get("token_points_ras"), dtype=float).reshape(-1, 3)
        if pts.shape[0] < 2:
            return {"proposals": [], "candidate_count": 0, "token_count": int(pts.shape[0])}
        blob_ids = np.asarray(prepared.get("token_blob_ids"), dtype=np.int32).reshape(-1)
        token_atom_ids = np.asarray(prepared.get("token_atom_ids"), dtype=np.int32).reshape(-1)
        axis_map = {int(k): np.asarray(v, dtype=float).reshape(3) for k, v in dict(prepared.get("axis_map") or {}).items()}
        elong_map = {int(k): float(v) for k, v in dict(prepared.get("elongation_map") or {}).items()}
        parent_blob_id_map = {int(k): int(v) for k, v in dict(prepared.get("parent_blob_id_map") or {}).items()}
        blob_graph = self._build_blob_connectivity_graph(
            pts_ras=pts,
            blob_ids=blob_ids,
            token_atom_ids=token_atom_ids,
            axis_map=axis_map,
            elongation_map=elong_map,
            support_atoms=prepared.get("support_atoms"),
            parent_blob_id_map=parent_blob_id_map,
            neighbor_max_mm=float(neighbor_max_mm),
            bridge_max_mm=float(max(10.0, neighbor_max_mm + 8.0)),
            bridge_axis_angle_deg=float(max(30.0, neighbor_axis_angle_deg + 5.0)),
        )
        seed_edges = list(blob_graph.get("local_seed_edges") or [])
        blob_seed_edges = list(blob_graph.get("bridge_seed_edges") or [])
        candidates = []
        for _seed_score, blob_i, blob_j in seed_edges[: int(max(1, max_seed_edges))]:
            proposal = self._grow_blob_chain_candidate(
                pts_ras=pts,
                blob_ids=blob_ids,
                blob_graph=blob_graph,
                seed_blob_i=int(blob_i),
                seed_blob_j=int(blob_j),
                grow_turn_angle_deg=float(grow_turn_angle_deg),
                inlier_radius_mm=float(inlier_radius_mm),
                axial_bin_mm=float(axial_bin_mm),
                min_span_mm=float(min_span_mm),
                min_inlier_count=int(min_inlier_count),
                parent_blob_id_map=parent_blob_id_map,
            )
            if proposal is None:
                continue
            proposal = self._complete_token_line_candidate(
                pts_ras=pts,
                blob_ids=blob_ids,
                token_atom_ids=token_atom_ids,
                candidate=proposal,
                inlier_radius_mm=float(inlier_radius_mm),
                axial_bin_mm=float(axial_bin_mm),
                min_span_mm=float(min_span_mm),
                min_inlier_count=int(min_inlier_count),
                elongation_map=elong_map,
                parent_blob_id_map=parent_blob_id_map,
            )
            if proposal is None:
                continue
            proposal["proposal_family"] = "graph"
            candidates.append(proposal)
        for _seed_score, blob_i, blob_j in blob_seed_edges[: int(max(1, max_seed_edges))]:
            proposal = self._grow_blob_chain_candidate(
                pts_ras=pts,
                blob_ids=blob_ids,
                blob_graph=blob_graph,
                seed_blob_i=int(blob_i),
                seed_blob_j=int(blob_j),
                grow_turn_angle_deg=float(grow_turn_angle_deg),
                inlier_radius_mm=float(inlier_radius_mm),
                axial_bin_mm=float(axial_bin_mm),
                min_span_mm=float(min_span_mm),
                min_inlier_count=int(min_inlier_count),
                parent_blob_id_map=parent_blob_id_map,
            )
            if proposal is None:
                continue
            proposal = self._complete_token_line_candidate(
                pts_ras=pts,
                blob_ids=blob_ids,
                token_atom_ids=token_atom_ids,
                candidate=proposal,
                inlier_radius_mm=float(inlier_radius_mm),
                axial_bin_mm=float(axial_bin_mm),
                min_span_mm=float(min_span_mm),
                min_inlier_count=int(min_inlier_count),
                elongation_map=elong_map,
                parent_blob_id_map=parent_blob_id_map,
            )
            if proposal is None:
                continue
            proposal["proposal_family"] = "blob_connectivity"
            candidates.append(proposal)
        blob_candidates = self._build_blob_axis_hypotheses(
            pts_ras=pts,
            blob_ids=blob_ids,
            token_atom_ids=token_atom_ids,
            support_atoms=prepared.get("support_atoms"),
            elongation_map=elong_map,
            axial_bin_mm=float(axial_bin_mm),
            min_span_mm=float(min_span_mm),
            min_inlier_count=int(min_inlier_count),
            parent_blob_id_map=parent_blob_id_map,
        )
        candidates.extend(blob_candidates)
        local_candidate_count = int(len(candidates))
        if local_candidate_count < 8 or (not seed_edges and not blob_seed_edges):
            pair_candidates = self._build_token_pair_hypotheses(
                pts_ras=pts,
                blob_ids=blob_ids,
                axis_map=axis_map,
                elongation_map=elong_map,
                inlier_radius_mm=float(inlier_radius_mm),
                axial_bin_mm=float(axial_bin_mm),
                min_span_mm=float(min_span_mm),
                min_inlier_count=int(min_inlier_count),
                max_pair_seeds=int(max_pair_seeds),
                parent_blob_id_map=parent_blob_id_map,
            )
            completed_pair_candidates = []
            for proposal in list(pair_candidates or []):
                proposal = self._complete_token_line_candidate(
                    pts_ras=pts,
                    blob_ids=blob_ids,
                    token_atom_ids=token_atom_ids,
                    candidate=proposal,
                    inlier_radius_mm=float(inlier_radius_mm),
                    axial_bin_mm=float(axial_bin_mm),
                    min_span_mm=float(min_span_mm),
                    min_inlier_count=int(min_inlier_count),
                    elongation_map=elong_map,
                    parent_blob_id_map=parent_blob_id_map,
                )
                if proposal is None:
                    continue
                proposal["proposal_family"] = "pair_ransac"
                completed_pair_candidates.append(proposal)
            candidates.extend(completed_pair_candidates)
        candidates = self._merge_collinear_token_line_candidates(
            pts_ras=pts,
            blob_ids=blob_ids,
            token_atom_ids=token_atom_ids,
            candidates=candidates,
            inlier_radius_mm=float(inlier_radius_mm),
            axial_bin_mm=float(axial_bin_mm),
            min_span_mm=float(min_span_mm),
            min_inlier_count=int(min_inlier_count),
            elongation_map=elong_map,
            parent_blob_id_map=parent_blob_id_map,
        )
        proposals = self._nms_token_line_candidates(candidates, max_output=max_output_proposals)
        proposals = self._resolve_contact_chain_stitched_claims(
            pts_ras=pts,
            blob_ids=blob_ids,
            token_atom_ids=token_atom_ids,
            proposals=proposals,
            support_atoms=prepared.get("support_atoms"),
            inlier_radius_mm=float(inlier_radius_mm),
            axial_bin_mm=float(axial_bin_mm),
            min_span_mm=float(min_span_mm),
            min_inlier_count=int(min_inlier_count),
            elongation_map=elong_map,
            parent_blob_id_map=parent_blob_id_map,
            max_output=max_output_proposals,
        )
        proposals = self._suppress_stitched_subsumed_blob_axis_proposals(
            proposals=proposals,
        )
        proposals = self._reject_short_floating_proposals(
            volume_node=volume_node,
            proposals=proposals,
            head_distance_map_kji=head_distance_map_kji,
            deep_core_shrink_mm=float(deep_core_shrink_mm),
            min_span_mm=float(short_float_reject_span_mm),
            edge_tol_mm=float(short_float_edge_tol_mm),
        )
        proposals = self._reject_small_proposals_without_outward_support(
            volume_node=volume_node,
            proposals=proposals,
            candidate_mask_kji=guided_candidate_mask_kji,
            head_distance_map_kji=head_distance_map_kji,
            max_checked_span_mm=float(outward_support_check_span_mm),
            support_radius_mm=float(outward_support_radius_mm),
            outward_search_mm=float(outward_support_search_mm),
            min_extension_mm=float(outward_support_min_extension_mm),
            min_depth_gain_mm=float(outward_support_min_depth_gain_mm),
            edge_tol_mm=float(short_float_edge_tol_mm),
        )
        proposals = self._extend_and_assign_models_to_proposals(
            volume_node=volume_node,
            proposals=proposals,
            token_points_ras=pts,
            token_blob_ids=blob_ids,
            threshold_hu=float(guided_threshold_hu),
            head_mask_threshold_hu=float(guided_head_mask_threshold_hu),
            roi_radius_mm=float(guided_roi_radius_mm),
            max_angle_deg=float(guided_max_angle_deg),
            max_depth_shift_mm=float(guided_max_depth_shift_mm),
            fit_mode=str(guided_fit_mode or "deep_anchor_v2"),
            max_residual_mm=float(guided_max_residual_mm),
            head_distance_map_kji=head_distance_map_kji,
            guided_candidate_mask_kji=guided_candidate_mask_kji,
        )
        proposals = self._suppress_stitched_subsumed_blob_axis_proposals(
            proposals=proposals,
        )
        # For deep-core debugging, keep the stack honest: uncovered/dropout pair
        # rescues create many false long lines and obscure whether the local
        # graph-based proposers are actually recovering the trajectory.
        rescue_candidates = []
        dropout_candidates = []
        return {
            "proposals": proposals,
            "candidate_count": int(len(candidates) + len(rescue_candidates) + len(dropout_candidates)),
            "token_count": int(pts.shape[0]),
        }

    @staticmethod
    def _ras_to_ijk_fn_for_volume(volume_node):
        if volume_node is None:
            return None
        if hasattr(volume_node, "GetRASToIJKMatrix"):
            def _fn(point_ras):
                m_vtk = vtk.vtkMatrix4x4()
                volume_node.GetRASToIJKMatrix(m_vtk)
                mat = np.eye(4, dtype=float)
                for r in range(4):
                    for c in range(4):
                        mat[r, c] = float(m_vtk.GetElement(r, c))
                return ras_to_ijk_float_matrix(point_ras, mat)
            return _fn
        if sitk is not None:
            try:
                _ijk_to_ras, ras_to_ijk = image_ijk_ras_matrices(volume_node)
                return lambda point_ras: ras_to_ijk_float_matrix(point_ras, ras_to_ijk)
            except Exception:
                return None
        return None

    @staticmethod
    def _kji_to_ras_points_for_volume(volume_node, ijk_kji):
        idx = np.asarray(ijk_kji, dtype=float).reshape(-1, 3)
        if idx.size == 0:
            return np.empty((0, 3), dtype=float)
        if volume_node is not None and hasattr(volume_node, "GetIJKToRASMatrix"):
            m_vtk = vtk.vtkMatrix4x4()
            volume_node.GetIJKToRASMatrix(m_vtk)
            mat = np.eye(4, dtype=float)
            for r in range(4):
                for c in range(4):
                    mat[r, c] = float(m_vtk.GetElement(r, c))
            return kji_to_ras_points_matrix(idx, mat)
        return np.asarray(kji_to_ras_points(volume_node, idx), dtype=float).reshape(-1, 3)

    @classmethod
    def _depth_at_ras_with_volume(cls, volume_node, depth_map_kji, point_ras):
        ras_to_ijk_fn = cls._ras_to_ijk_fn_for_volume(volume_node)
        if depth_map_kji is None or ras_to_ijk_fn is None:
            return None
        ijk = ras_to_ijk_fn(point_ras)
        i = int(round(float(ijk[0])))
        j = int(round(float(ijk[1])))
        k = int(round(float(ijk[2])))
        if k < 0 or j < 0 or i < 0 or k >= depth_map_kji.shape[0] or j >= depth_map_kji.shape[1] or i >= depth_map_kji.shape[2]:
            return None
        val = float(depth_map_kji[k, j, i])
        return val if np.isfinite(val) else None

    @staticmethod
    def _volume_array_kji(volume_node):
        if volume_node is None:
            return None
        if hasattr(volume_node, "GetIJKToRASMatrix"):
            try:
                return np.asarray(slicer.util.arrayFromVolume(volume_node), dtype=float)
            except Exception:
                return None
        if sitk is not None:
            try:
                return np.asarray(sitk.GetArrayFromImage(volume_node), dtype=float)
            except Exception:
                return None
        return None

    @staticmethod
    def _orthonormal_basis_for_axis(axis_ras):
        axis = np.asarray(axis_ras, dtype=float).reshape(3)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-6:
            axis = np.asarray([0.0, 0.0, 1.0], dtype=float)
            axis_norm = 1.0
        axis = axis / axis_norm
        ref = np.asarray([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(axis, ref))) > 0.90:
            ref = np.asarray([0.0, 1.0, 0.0], dtype=float)
        u = np.cross(axis, ref)
        u_norm = float(np.linalg.norm(u))
        if u_norm <= 1e-6:
            ref = np.asarray([0.0, 0.0, 1.0], dtype=float)
            u = np.cross(axis, ref)
            u_norm = float(np.linalg.norm(u))
        u = u / max(u_norm, 1e-6)
        v = np.cross(axis, u)
        v = v / max(float(np.linalg.norm(v)), 1e-6)
        return axis, u, v

    @classmethod
    def _proposal_annulus_mean_ct_hu(
        cls,
        volume_node,
        proposal,
        annulus_inner_mm=2.5,
        annulus_outer_mm=3.5,
        axial_step_mm=2.0,
        radial_steps=2,
        angular_samples=12,
    ):
        arr_kji = cls._volume_array_kji(volume_node)
        ras_to_ijk_fn = cls._ras_to_ijk_fn_for_volume(volume_node)
        if arr_kji is None or ras_to_ijk_fn is None:
            return {"mean_hu": None, "sample_count": 0}
        start = np.asarray(proposal.get("start_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
        end = np.asarray(proposal.get("end_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
        axis, u, v = cls._orthonormal_basis_for_axis(end - start)
        span_mm = float(np.linalg.norm(end - start))
        if span_mm <= 1e-6:
            return {"mean_hu": None, "sample_count": 0}
        axial_positions = np.arange(0.0, span_mm + 1e-6, float(max(0.5, axial_step_mm)), dtype=float)
        radial_values = np.linspace(float(annulus_inner_mm), float(annulus_outer_mm), int(max(1, radial_steps)))
        angles = np.linspace(0.0, 2.0 * np.pi, int(max(4, angular_samples)), endpoint=False)
        samples = []
        for t in axial_positions.tolist():
            center = start + axis * float(t)
            for radius in radial_values.tolist():
                for theta in angles.tolist():
                    offset = (
                        float(radius) * np.cos(float(theta)) * u
                        + float(radius) * np.sin(float(theta)) * v
                    )
                    point_ras = center + offset
                    ijk = ras_to_ijk_fn(point_ras)
                    i = int(round(float(ijk[0])))
                    j = int(round(float(ijk[1])))
                    k = int(round(float(ijk[2])))
                    if (
                        k < 0 or j < 0 or i < 0
                        or k >= arr_kji.shape[0]
                        or j >= arr_kji.shape[1]
                        or i >= arr_kji.shape[2]
                    ):
                        continue
                    val = float(arr_kji[k, j, i])
                    if np.isfinite(val):
                        samples.append(val)
        if not samples:
            return {"mean_hu": None, "sample_count": 0}
        return {
            "mean_hu": float(np.mean(np.asarray(samples, dtype=float))),
            "sample_count": int(len(samples)),
        }

    @classmethod
    def _reject_short_floating_proposals(
        cls,
        volume_node,
        proposals,
        head_distance_map_kji=None,
        deep_core_shrink_mm=15.0,
        min_span_mm=20.0,
        edge_tol_mm=2.5,
    ):
        proposal_list = [dict(p or {}) for p in list(proposals or [])]
        if not proposal_list or head_distance_map_kji is None:
            return proposal_list
        kept = []
        edge_depth_mm = float(deep_core_shrink_mm)
        tol_mm = float(max(0.5, edge_tol_mm))
        span_cutoff_mm = float(max(1.0, min_span_mm))
        for proposal in proposal_list:
            span_mm = float(proposal.get("span_mm", 0.0) or 0.0)
            if span_mm >= span_cutoff_mm:
                kept.append(proposal)
                continue
            start_ras = np.asarray(proposal.get("start_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
            end_ras = np.asarray(proposal.get("end_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
            start_depth = cls._depth_at_ras_with_volume(volume_node, head_distance_map_kji, start_ras)
            end_depth = cls._depth_at_ras_with_volume(volume_node, head_distance_map_kji, end_ras)
            touches_edge = False
            for depth_val in (start_depth, end_depth):
                if depth_val is None:
                    continue
                if abs(float(depth_val) - edge_depth_mm) <= tol_mm:
                    touches_edge = True
                    break
            if not touches_edge:
                continue
            kept.append(proposal)
        return kept

    def _extract_guided_candidate_points_lps(
        self,
        volume_node,
        threshold_hu,
        head_mask_threshold_hu=-500.0,
        min_metal_depth_mm=5.0,
        max_metal_depth_mm=220.0,
    ):
        if volume_node is None:
            return {"points_lps": np.empty((0, 3), dtype=float), "threshold_hu": float(threshold_hu)}
        if hasattr(volume_node, "GetIJKToRASMatrix"):
            return self.extract_threshold_candidates_lps(
                volume_node=volume_node,
                threshold=float(threshold_hu),
                head_mask_threshold_hu=float(head_mask_threshold_hu),
                min_metal_depth_mm=float(min_metal_depth_mm),
                max_metal_depth_mm=float(max_metal_depth_mm),
            )
        if sitk is None:
            return {"points_lps": np.empty((0, 3), dtype=float), "threshold_hu": float(threshold_hu)}
        arr = sitk.GetArrayFromImage(volume_node)
        spacing_xyz = tuple(float(v) for v in volume_node.GetSpacing())
        used_threshold = float(threshold_hu)
        best_preview = None
        best_count = -1
        while True:
            preview = build_preview_masks(
                arr_kji=np.asarray(arr, dtype=float),
                spacing_xyz=spacing_xyz,
                threshold=float(used_threshold),
                use_head_mask=True,
                build_head_mask=True,
                head_mask_threshold_hu=float(head_mask_threshold_hu),
                head_mask_method="outside_air",
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
            return {"points_lps": np.empty((0, 3), dtype=float), "threshold_hu": float(used_threshold)}
        ras = np.asarray(self._kji_to_ras_points_for_volume(volume_node, idx), dtype=float).reshape(-1, 3)
        lps = ras.copy()
        lps[:, 0] *= -1.0
        lps[:, 1] *= -1.0
        return {"points_lps": lps, "threshold_hu": float(used_threshold)}

    def _guided_candidate_points_lps_from_mask(
        self,
        volume_node,
        candidate_mask_kji,
        head_distance_map_kji=None,
        min_metal_depth_mm=5.0,
        max_metal_depth_mm=220.0,
    ):
        mask = np.asarray(candidate_mask_kji, dtype=bool)
        if head_distance_map_kji is not None:
            depth_map = np.asarray(head_distance_map_kji, dtype=float)
            mask = np.logical_and(mask, np.isfinite(depth_map))
            mask = np.logical_and(mask, depth_map >= float(min_metal_depth_mm))
            if np.isfinite(float(max_metal_depth_mm)):
                mask = np.logical_and(mask, depth_map <= float(max_metal_depth_mm))
        idx = np.argwhere(mask)
        if idx.size == 0:
            return np.empty((0, 3), dtype=float)
        ras = np.asarray(self._kji_to_ras_points_for_volume(volume_node, idx), dtype=float).reshape(-1, 3)
        lps = ras.copy()
        lps[:, 0] *= -1.0
        lps[:, 1] *= -1.0
        return lps

    def _guided_candidate_support_from_mask(
        self,
        volume_node,
        candidate_mask_kji,
        head_distance_map_kji=None,
        min_metal_depth_mm=5.0,
        max_metal_depth_mm=220.0,
    ):
        mask = np.asarray(candidate_mask_kji, dtype=bool)
        if head_distance_map_kji is not None:
            depth_map = np.asarray(head_distance_map_kji, dtype=float)
            mask = np.logical_and(mask, np.isfinite(depth_map))
            mask = np.logical_and(mask, depth_map >= float(min_metal_depth_mm))
            if np.isfinite(float(max_metal_depth_mm)):
                mask = np.logical_and(mask, depth_map <= float(max_metal_depth_mm))
        idx = np.argwhere(mask)
        if idx.size == 0:
            return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)
        ras = np.asarray(self._kji_to_ras_points_for_volume(volume_node, idx), dtype=float).reshape(-1, 3)
        if head_distance_map_kji is None:
            depth_vals = np.zeros((ras.shape[0],), dtype=float)
        else:
            depth_vals = np.asarray(head_distance_map_kji, dtype=float)[idx[:, 0], idx[:, 1], idx[:, 2]].reshape(-1)
        return ras, depth_vals

    @staticmethod
    def _extend_shallow_endpoint_along_axis(
        proposal,
        support_pts_ras,
        support_depths,
        volume_node,
        head_distance_map_kji,
        support_radius_mm=2.5,
        outward_search_mm=120.0,
        inward_search_mm=40.0,
        bin_mm=1.0,
        max_gap_bins=2,
        max_local_diameter_mm=4.0,
        thick_run_bins=3,
    ):
        start_ras = np.asarray(proposal.get("start_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
        end_ras = np.asarray(proposal.get("end_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
        axis_out = end_ras - start_ras
        axis_norm = float(np.linalg.norm(axis_out))
        if axis_norm <= 1e-6:
            axis_out = np.asarray(proposal.get("axis_ras") or [0.0, 0.0, 1.0], dtype=float).reshape(3)
            axis_norm = float(np.linalg.norm(axis_out))
            if axis_norm <= 1e-6:
                axis_out = np.asarray([0.0, 0.0, 1.0], dtype=float)
                axis_norm = 1.0
        axis_out = axis_out / axis_norm
        seed_span_mm = float(np.linalg.norm(end_ras - start_ras))
        rel = np.asarray(support_pts_ras, dtype=float).reshape(-1, 3) - start_ras.reshape(1, 3)
        axial = (rel @ axis_out.reshape(3, 1)).reshape(-1)
        radial = np.linalg.norm(rel - axial.reshape(-1, 1) * axis_out.reshape(1, 3), axis=1)
        support_mask = np.logical_and(
            axial >= -float(max(5.0, inward_search_mm)),
            axial <= float(seed_span_mm + max(5.0, outward_search_mm)),
        )
        support_mask = np.logical_and(support_mask, radial <= float(max(0.5, support_radius_mm)))
        if not np.any(support_mask):
            axis_seed = end_ras - start_ras
            axis_seed = axis_seed / max(float(np.linalg.norm(axis_seed)), 1e-6)
            return {
                "start_ras": start_ras,
                "end_ras": end_ras,
                "axis_ras": axis_seed,
                "extended_mm": 0.0,
                "shallow_depth_mm": DeepCoreProposalLogicMixin._depth_at_ras_with_volume(
                    volume_node, head_distance_map_kji, start_ras
                ),
                "extended_depth_mm": DeepCoreProposalLogicMixin._depth_at_ras_with_volume(
                    volume_node, head_distance_map_kji, end_ras
                ),
            }
        axial_sel = axial[support_mask]
        if axial_sel.size == 0:
            axis_seed = end_ras - start_ras
            axis_seed = axis_seed / max(float(np.linalg.norm(axis_seed)), 1e-6)
            return {
                "start_ras": start_ras,
                "end_ras": end_ras,
                "axis_ras": axis_seed,
                "extended_mm": 0.0,
                "shallow_depth_mm": DeepCoreProposalLogicMixin._depth_at_ras_with_volume(
                    volume_node, head_distance_map_kji, start_ras
                ),
                "extended_depth_mm": DeepCoreProposalLogicMixin._depth_at_ras_with_volume(
                    volume_node, head_distance_map_kji, end_ras
                ),
            }
        radial_sel = radial[support_mask]
        bin_size = float(max(0.5, bin_mm))
        bin_ids = np.floor(axial_sel / bin_size).astype(int)
        bin_to_radial = {}
        for bin_id, radial_val in zip(bin_ids.tolist(), radial_sel.tolist()):
            bin_to_radial.setdefault(int(bin_id), []).append(float(radial_val))
        seed_lo_bin = int(np.floor(0.0 / bin_size))
        seed_hi_bin = int(np.floor(max(seed_span_mm - 1e-6, 0.0) / bin_size))

        def can_include_bin(bin_id):
            radial_vals = np.asarray(bin_to_radial.get(int(bin_id)) or [], dtype=float).reshape(-1)
            if radial_vals.size == 0:
                return False, False
            local_radius_q75 = float(np.percentile(radial_vals, 75.0))
            local_diameter_mm = 2.0 * local_radius_q75
            return True, bool(local_diameter_mm > float(max_local_diameter_mm))

        run_lo = int(seed_lo_bin)
        run_hi = int(seed_hi_bin)

        gap_run = 0
        thick_run = 0
        b = int(seed_lo_bin - 1)
        min_bin = int(np.floor((-float(max(5.0, inward_search_mm))) / bin_size))
        while b >= min_bin:
            has_support, is_thick = can_include_bin(b)
            if has_support and not is_thick:
                run_lo = int(b)
                gap_run = 0
                thick_run = 0
            elif has_support and is_thick:
                thick_run += 1
                if thick_run >= int(max(1, thick_run_bins)):
                    break
            else:
                gap_run += 1
                if gap_run > int(max_gap_bins):
                    break
            b -= 1

        gap_run = 0
        thick_run = 0
        b = int(seed_hi_bin + 1)
        max_bin = int(np.ceil((seed_span_mm + float(max(5.0, outward_search_mm))) / bin_size))
        while b <= max_bin:
            has_support, is_thick = can_include_bin(b)
            if has_support and not is_thick:
                run_hi = int(b)
                gap_run = 0
                thick_run = 0
            elif has_support and is_thick:
                thick_run += 1
                if thick_run >= int(max(1, thick_run_bins)):
                    break
            else:
                gap_run += 1
                if gap_run > int(max_gap_bins):
                    break
            b += 1

        new_start_offset_mm = float(min(0.0, run_lo * bin_size))
        new_end_offset_mm = float(max(seed_span_mm, (run_hi + 1) * bin_size))
        new_start = start_ras + axis_out * new_start_offset_mm
        new_end = start_ras + axis_out * new_end_offset_mm
        axis_final = new_end - new_start
        axis_final_norm = float(np.linalg.norm(axis_final))
        if axis_final_norm <= 1e-6:
            axis_final = end_ras - start_ras
            axis_final_norm = float(np.linalg.norm(axis_final))
        axis_final = axis_final / max(axis_final_norm, 1e-6)
        start_depth = DeepCoreProposalLogicMixin._depth_at_ras_with_volume(volume_node, head_distance_map_kji, new_start)
        end_depth = DeepCoreProposalLogicMixin._depth_at_ras_with_volume(volume_node, head_distance_map_kji, new_end)
        return {
            "start_ras": new_start,
            "end_ras": new_end,
            "axis_ras": axis_final,
            "extended_mm": float(max(0.0, (new_end_offset_mm - new_start_offset_mm) - seed_span_mm)),
            "shallow_depth_mm": None if start_depth is None or end_depth is None else float(min(start_depth, end_depth)),
            "extended_depth_mm": None if start_depth is None or end_depth is None else float(max(start_depth, end_depth)),
        }

    @classmethod
    def _reject_small_proposals_without_outward_support(
        cls,
        volume_node,
        proposals,
        candidate_mask_kji=None,
        head_distance_map_kji=None,
        max_checked_span_mm=45.0,
        support_radius_mm=2.5,
        outward_search_mm=20.0,
        min_extension_mm=6.0,
        min_depth_gain_mm=4.0,
        edge_tol_mm=2.5,
    ):
        proposal_list = [dict(p or {}) for p in list(proposals or [])]
        if not proposal_list or candidate_mask_kji is None or head_distance_map_kji is None:
            return proposal_list
        helper = cls()
        support_pts_ras, support_depths = helper._guided_candidate_support_from_mask(
            volume_node=volume_node,
            candidate_mask_kji=candidate_mask_kji,
            head_distance_map_kji=head_distance_map_kji,
        )
        if support_pts_ras.shape[0] == 0:
            return proposal_list
        kept = []
        for proposal in proposal_list:
            span_mm = float(proposal.get("span_mm", 0.0) or 0.0)
            if span_mm >= float(max_checked_span_mm):
                kept.append(proposal)
                continue
            start_ras = np.asarray(proposal.get("start_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
            end_ras = np.asarray(proposal.get("end_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
            start_depth = cls._depth_at_ras_with_volume(volume_node, head_distance_map_kji, start_ras)
            end_depth = cls._depth_at_ras_with_volume(volume_node, head_distance_map_kji, end_ras)
            if start_depth is None or end_depth is None:
                kept.append(proposal)
                continue
            if min(float(start_depth), float(end_depth)) <= float(edge_tol_mm):
                kept.append(proposal)
                continue
            if float(start_depth) <= float(end_depth):
                shallow_ras = start_ras
                deep_ras = end_ras
                shallow_depth = float(start_depth)
            else:
                shallow_ras = end_ras
                deep_ras = start_ras
                shallow_depth = float(end_depth)
            axis_out = shallow_ras - deep_ras
            axis_norm = float(np.linalg.norm(axis_out))
            if axis_norm <= 1e-6:
                continue
            axis_out = axis_out / axis_norm
            rel = support_pts_ras - shallow_ras.reshape(1, 3)
            axial = (rel @ axis_out.reshape(3, 1)).reshape(-1)
            radial = np.linalg.norm(rel - axial.reshape(-1, 1) * axis_out.reshape(1, 3), axis=1)
            support_mask = np.logical_and(axial >= 0.0, axial <= float(max(2.0, outward_search_mm)))
            support_mask = np.logical_and(support_mask, radial <= float(max(1.0, support_radius_mm)))
            if not np.any(support_mask):
                continue
            ext_axial = axial[support_mask]
            ext_depths = support_depths[support_mask]
            extension_mm = float(np.max(ext_axial))
            min_depth = float(np.min(ext_depths))
            depth_gain = float(shallow_depth - min_depth)
            if extension_mm < float(min_extension_mm):
                continue
            if depth_gain < float(min_depth_gain_mm):
                continue
            kept.append(proposal)
        return kept

    def _extend_and_assign_models_to_proposals(
        self,
        volume_node,
        proposals,
        token_points_ras,
        token_blob_ids,
        threshold_hu,
        head_mask_threshold_hu=-500.0,
        roi_radius_mm=5.0,
        max_angle_deg=12.0,
        max_depth_shift_mm=2.0,
        fit_mode="deep_anchor_v2",
        max_residual_mm=2.0,
        head_distance_map_kji=None,
        guided_candidate_mask_kji=None,
        annulus_reject_mean_hu=200.0,
        annulus_reject_min_samples=120,
    ):
        proposal_list = [dict(p or {}) for p in list(proposals or [])]
        if not proposal_list:
            return []
        if guided_candidate_mask_kji is not None:
            support_pts_ras, support_depths = self._guided_candidate_support_from_mask(
                volume_node=volume_node,
                candidate_mask_kji=guided_candidate_mask_kji,
                head_distance_map_kji=head_distance_map_kji,
                min_metal_depth_mm=0.0,
            )
            candidate_threshold_hu = float(threshold_hu)
        else:
            candidate_payload = self._extract_guided_candidate_points_lps(
                volume_node=volume_node,
                threshold_hu=float(threshold_hu),
                head_mask_threshold_hu=float(head_mask_threshold_hu),
            )
            candidate_points_lps = np.asarray(candidate_payload.get("points_lps"), dtype=float).reshape(-1, 3)
            support_pts_ras = candidate_points_lps.copy()
            support_pts_ras[:, 0] *= -1.0
            support_pts_ras[:, 1] *= -1.0
            support_depths = np.zeros((support_pts_ras.shape[0],), dtype=float)
            candidate_threshold_hu = float(candidate_payload.get("threshold_hu", threshold_hu))
        if np.asarray(support_pts_ras, dtype=float).reshape(-1, 3).shape[0] == 0:
            return proposal_list
        accepted = []
        for proposal in proposal_list:
            grown = self._extend_shallow_endpoint_along_axis(
                proposal=proposal,
                support_pts_ras=support_pts_ras,
                support_depths=support_depths,
                volume_node=volume_node,
                head_distance_map_kji=head_distance_map_kji,
            )
            proposal["start_ras"] = [float(v) for v in np.asarray(grown.get("start_ras"), dtype=float).reshape(3)]
            proposal["end_ras"] = [float(v) for v in np.asarray(grown.get("end_ras"), dtype=float).reshape(3)]
            proposal["axis_ras"] = [float(v) for v in np.asarray(grown.get("axis_ras"), dtype=float).reshape(3)]
            proposal["span_mm"] = float(
                np.linalg.norm(np.asarray(proposal["end_ras"], dtype=float) - np.asarray(proposal["start_ras"], dtype=float))
            )
            proposal["guided_fit_success"] = False
            proposal["guided_fit_mode_used"] = "axis_growth"
            proposal["guided_fit_threshold_hu"] = float(candidate_threshold_hu)
            proposal["axis_growth_extension_mm"] = float(grown.get("extended_mm", 0.0) or 0.0)
            proposal["axis_growth_shallow_depth_mm"] = grown.get("shallow_depth_mm")
            proposal["axis_growth_extended_depth_mm"] = grown.get("extended_depth_mm")
            annulus_stats = self._proposal_annulus_mean_ct_hu(
                volume_node=volume_node,
                proposal=proposal,
            )
            proposal["annulus_mean_ct_hu"] = annulus_stats.get("mean_hu")
            proposal["annulus_sample_count"] = int(annulus_stats.get("sample_count", 0) or 0)
            annulus_mean_hu = proposal.get("annulus_mean_ct_hu")
            annulus_sample_count = int(proposal.get("annulus_sample_count", 0) or 0)
            proposal["annulus_bone_suspicious"] = bool(
                annulus_mean_hu is not None
                and annulus_sample_count >= int(max(1, annulus_reject_min_samples))
                and float(annulus_mean_hu) > float(annulus_reject_mean_hu)
            )
            proposal["best_model_id"] = ""
            proposal["best_model_score"] = 0.0
            accepted.append(proposal)
        return accepted

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

    @staticmethod

    @staticmethod

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

    @staticmethod
    def _support_atom_endpoint_candidates(node):
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

    def _grow_token_chain_candidate(
        self,
        pts_ras,
        blob_ids,
        graph,
        seed_i,
        seed_j,
        grow_turn_angle_deg=30.0,
        inlier_radius_mm=2.0,
        axial_bin_mm=2.5,
        min_span_mm=14.0,
        min_inlier_count=5,
        min_chain_tokens=5,
    ):
        p0 = np.asarray(pts_ras[int(seed_i)], dtype=float).reshape(3)
        p1 = np.asarray(pts_ras[int(seed_j)], dtype=float).reshape(3)
        axis = p1 - p0
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-6:
            return None
        axis = axis / axis_norm
        chain = [int(seed_i), int(seed_j)]
        chain_set = {int(seed_i), int(seed_j)}
        adj = list(graph.get("adj") or [])

        self._grow_token_chain_side(
            pts_ras=pts_ras,
            adj=adj,
            chain=chain,
            chain_set=chain_set,
            grow_forward=True,
            grow_turn_angle_deg=float(grow_turn_angle_deg),
            inlier_radius_mm=float(inlier_radius_mm),
        )
        self._grow_token_chain_side(
            pts_ras=pts_ras,
            adj=adj,
            chain=chain,
            chain_set=chain_set,
            grow_forward=False,
            grow_turn_angle_deg=float(grow_turn_angle_deg),
            inlier_radius_mm=float(inlier_radius_mm),
        )

        if len(chain) < int(max(2, min_chain_tokens)):
            return None
        chain_pts = np.asarray(pts_ras[np.asarray(chain, dtype=int)], dtype=float).reshape(-1, 3)
        fit_center, fit_axis = self._fit_line_pca(chain_pts, seed_axis=axis)
        radial_fit = self._radial_distance_to_line(chain_pts, fit_center, fit_axis)
        if float(np.sqrt(np.mean(radial_fit ** 2))) > float(max(0.1, inlier_radius_mm)):
            return None
        inlier_pts = chain_pts
        inlier_blob_ids = np.asarray(blob_ids[np.asarray(chain, dtype=int)], dtype=np.int32).reshape(-1)
        proj = ((inlier_pts - fit_center.reshape(1, 3)) @ fit_axis.reshape(3, 1)).reshape(-1)
        pmin = float(np.min(proj))
        pmax = float(np.max(proj))
        span = float(pmax - pmin)
        if span < float(max(1.0, min_span_mm)):
            return None
        rms = float(np.sqrt(np.mean(self._radial_distance_to_line(inlier_pts, fit_center, fit_axis) ** 2)))
        occ = self._axial_occupancy_metrics(proj, bin_mm=float(max(0.5, axial_bin_mm)))
        distinct_blobs = int(np.unique(inlier_blob_ids).size)
        if occ["occupied_bins"] < 2:
            return None
        if not self._occupancy_supports_line(
            occ,
            min_coverage=0.45,
            max_gap_bins=2,
            max_group_runs=3,
            min_group_occupied_bins=6,
            min_group_run_bins=2,
        ):
            return None
        if distinct_blobs < 2 and (span < float(max(20.0, min_span_mm + 4.0)) or int(inlier_pts.shape[0]) < int(max(7, min_inlier_count + 2))):
            return None
        turn_metrics = self._chain_turn_metrics(np.asarray(chain, dtype=int), np.asarray(pts_ras, dtype=float))
        if float(turn_metrics["min_dot"]) < float(np.cos(np.deg2rad(40.0))):
            return None
        score = (
            1.0 * span
            + 1.0 * float(inlier_pts.shape[0])
            + 5.0 * float(distinct_blobs)
            + 8.0 * float(occ["coverage"])
            - 2.5 * float(rms)
            - 2.0 * float(occ["max_gap_bins"])
            + 4.0 * float(turn_metrics["mean_dot"])
        )
        if distinct_blobs == 1:
            score -= 18.0
        elif distinct_blobs == 2:
            score -= 6.0
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
            "inlier_count": int(inlier_pts.shape[0]),
            "distinct_blob_count": int(distinct_blobs),
            "rms_mm": float(rms),
            "coverage": float(occ["coverage"]),
            "occupied_bins": int(occ["occupied_bins"]),
            "total_bins": int(occ["total_bins"]),
            "max_gap_bins": int(occ["max_gap_bins"]),
            "blob_id_list": [int(v) for v in np.unique(inlier_blob_ids).astype(int).tolist()],
            "token_indices": [int(v) for v in chain],
            "mean_turn_dot": float(turn_metrics["mean_dot"]),
            "score": float(score),
        }

    def _build_token_pair_hypotheses(
        self,
        pts_ras,
        blob_ids,
        axis_map,
        elongation_map,
        inlier_radius_mm=2.0,
        axial_bin_mm=2.5,
        min_span_mm=16.0,
        min_inlier_count=5,
        max_pair_seeds=600,
        parent_blob_id_map=None,
    ):
        pts = np.asarray(pts_ras, dtype=float).reshape(-1, 3)
        blob_ids = np.asarray(blob_ids, dtype=np.int32).reshape(-1)
        n = int(pts.shape[0])
        if n < 2:
            return []
        pair_candidates = []
        max_pair_seeds = int(max(1, max_pair_seeds))
        for i in range(n - 1):
            delta = pts[(i + 1) :, :] - pts[i].reshape(1, 3)
            dist = np.linalg.norm(delta, axis=1)
            valid = np.flatnonzero(np.logical_and(dist >= 6.0, dist <= 60.0))
            if valid.size == 0:
                continue
            for rel in valid.tolist():
                j = i + 1 + int(rel)
                blob_i = int(blob_ids[i])
                blob_j = int(blob_ids[j])
                same_blob = blob_i == blob_j
                if same_blob and float(elongation_map.get(blob_i, 1.0)) < 5.0:
                    continue
                direction = delta[int(rel)] / max(float(dist[int(rel)]), 1e-6)
                compat = False
                axis_agreement = 0.0
                for blob_id in (blob_i, blob_j):
                    axis = np.asarray(axis_map.get(int(blob_id), []), dtype=float).reshape(-1)
                    if axis.size != 3:
                        continue
                    axis_norm = float(np.linalg.norm(axis))
                    if axis_norm <= 1e-6:
                        continue
                    axis = axis / axis_norm
                    axis_agreement = max(axis_agreement, abs(float(np.dot(direction, axis))))
                    if axis_agreement >= float(np.cos(np.deg2rad(35.0))):
                        compat = True
                        break
                if not compat and not same_blob and float(dist[int(rel)]) < 12.0:
                    continue
                seed_priority = (
                    (2.0 if not same_blob else 0.0)
                    + min(float(dist[int(rel)]), 30.0) / 30.0
                    + (1.0 if compat else -0.75)
                    + 0.75 * float(axis_agreement)
                    + 0.1 * float(elongation_map.get(blob_i, 1.0) + elongation_map.get(blob_j, 1.0))
                )
                pair_candidates.append((seed_priority, int(i), int(j), float(dist[int(rel)])))
        pair_candidates.sort(key=lambda item: float(item[0]), reverse=True)
        pair_candidates = self._select_token_pair_seed_subset(
            pair_candidates=pair_candidates,
            blob_ids=blob_ids,
            elongation_map=elongation_map,
            max_pair_seeds=max_pair_seeds,
        )
        proposals = []
        for _priority, i, j, _dist_mm in pair_candidates:
            proposal = self._score_token_pair_hypothesis(
                pts_ras=pts,
                blob_ids=blob_ids,
                seed_i=int(i),
                seed_j=int(j),
                inlier_radius_mm=float(inlier_radius_mm),
                axial_bin_mm=float(axial_bin_mm),
                min_span_mm=float(min_span_mm),
                min_inlier_count=int(min_inlier_count),
                elongation_map=elongation_map,
                parent_blob_id_map=parent_blob_id_map,
            )
            if proposal is None:
                continue
            proposal["proposal_family"] = "pair_ransac"
            proposals.append(proposal)
        return proposals

    @staticmethod

    @staticmethod

    @staticmethod
    def _token_pair_distance_band(dist_mm):
        dist = float(dist_mm)
        if dist < 18.0:
            return 0
        if dist < 36.0:
            return 1
        return 2

    def _select_token_pair_seed_subset(
        self,
        pair_candidates,
        blob_ids,
        elongation_map,
        max_pair_seeds,
    ):
        ranked = list(pair_candidates or [])
        limit = int(max(1, max_pair_seeds))
        if len(ranked) <= limit:
            return ranked
        blob_ids = np.asarray(blob_ids, dtype=np.int32).reshape(-1)
        elong_map = {int(k): float(v) for k, v in dict(elongation_map or {}).items()}
        selected = []
        selected_keys = set()
        covered_blob_bands = set()

        def candidate_key(item):
            return (int(item[1]), int(item[2]))

        def add_candidate(item):
            key = candidate_key(item)
            if key in selected_keys:
                return False
            selected.append(item)
            selected_keys.add(key)
            _priority, i, j, dist_mm = item
            band = self._token_pair_distance_band(float(dist_mm))
            covered_blob_bands.add((int(blob_ids[int(i)]), int(band)))
            covered_blob_bands.add((int(blob_ids[int(j)]), int(band)))
            return True

        priority_budget = min(len(ranked), max(32, int(round(0.22 * float(limit)))))
        for item in ranked[:priority_budget]:
            if len(selected) >= limit:
                return selected
            add_candidate(item)

        rep_by_blob_band = {}
        for item in ranked:
            key = candidate_key(item)
            if key in selected_keys:
                continue
            _priority, i, j, dist_mm = item
            band = self._token_pair_distance_band(float(dist_mm))
            for blob_id in (int(blob_ids[int(i)]), int(blob_ids[int(j)])):
                blob_band = (int(blob_id), int(band))
                if blob_band in covered_blob_bands or blob_band in rep_by_blob_band:
                    continue
                rep_by_blob_band[blob_band] = item

        coverage_candidates = []
        seen_coverage = set()
        for item in rep_by_blob_band.values():
            key = candidate_key(item)
            if key in selected_keys or key in seen_coverage:
                continue
            seen_coverage.add(key)
            coverage_candidates.append(item)

        def coverage_gain(item):
            _priority, i, j, dist_mm = item
            band = self._token_pair_distance_band(float(dist_mm))
            gain = 0
            for blob_id in (int(blob_ids[int(i)]), int(blob_ids[int(j)])):
                if (int(blob_id), int(band)) not in covered_blob_bands:
                    gain += 1
            return int(gain)

        coverage_candidates.sort(
            key=lambda item: (coverage_gain(item), float(item[0])),
            reverse=True,
        )
        for item in coverage_candidates:
            if len(selected) >= limit:
                return selected
            add_candidate(item)

        fragment_budget = min(int(max(48, round(0.30 * float(limit)))), max(0, limit - len(selected)))
        if fragment_budget > 0:
            fragment_candidates = []
            seen_fragment_pairs = set()
            def fragment_sort_key(cand):
                _priority, i, j, dist_mm = cand
                blob_i = int(blob_ids[int(i)])
                blob_j = int(blob_ids[int(j)])
                min_elong = min(float(elong_map.get(blob_i, 1.0)), float(elong_map.get(blob_j, 1.0)))
                return (float(min_elong), float(dist_mm), -float(cand[0]))

            for item in sorted(ranked, key=fragment_sort_key):
                key = candidate_key(item)
                if key in selected_keys:
                    continue
                _priority, i, j, _dist_mm = item
                blob_i = int(blob_ids[int(i)])
                blob_j = int(blob_ids[int(j)])
                if blob_i == blob_j:
                    continue
                if min(float(elong_map.get(blob_i, 1.0)), float(elong_map.get(blob_j, 1.0))) >= 4.5:
                    continue
                blob_pair = tuple(sorted((blob_i, blob_j)))
                if blob_pair in seen_fragment_pairs:
                    continue
                seen_fragment_pairs.add(blob_pair)
                fragment_candidates.append(item)
            for item in fragment_candidates[:fragment_budget]:
                if len(selected) >= limit:
                    return selected
                add_candidate(item)

        for item in ranked:
            if len(selected) >= limit:
                break
            add_candidate(item)
        return selected

    def _build_uncovered_token_hypotheses(
        self,
        pts_ras,
        blob_ids,
        axis_map,
        elongation_map,
        existing_proposals,
        inlier_radius_mm=2.0,
        axial_bin_mm=2.5,
        min_span_mm=16.0,
        min_inlier_count=5,
    ):
        pts = np.asarray(pts_ras, dtype=float).reshape(-1, 3)
        blob_ids = np.asarray(blob_ids, dtype=np.int32).reshape(-1)
        covered = set()
        for proposal in list(existing_proposals or []):
            covered.update(int(v) for v in list(proposal.get("token_indices") or []))
        uncovered = np.asarray([idx for idx in range(int(pts.shape[0])) if idx not in covered], dtype=np.int32)
        if uncovered.size < max(4, int(min_inlier_count)):
            return []
        sub_pts = pts[uncovered]
        sub_blob_ids = blob_ids[uncovered]
        rescue = self._build_token_pair_hypotheses(
            pts_ras=sub_pts,
            blob_ids=sub_blob_ids,
            axis_map=axis_map,
            elongation_map=elongation_map,
            inlier_radius_mm=float(max(2.0, inlier_radius_mm)),
            axial_bin_mm=float(axial_bin_mm),
            min_span_mm=float(max(12.0, min_span_mm - 4.0)),
            min_inlier_count=int(max(4, min_inlier_count - 1)),
            max_pair_seeds=400,
        )
        out = []
        uncovered_list = uncovered.astype(int).tolist()
        for proposal in list(rescue or []):
            mapped = [int(uncovered_list[int(v)]) for v in list(proposal.get("token_indices") or []) if 0 <= int(v) < len(uncovered_list)]
            if not mapped:
                continue
            proposal = dict(proposal)
            proposal["token_indices"] = mapped
            proposal = self._complete_token_line_candidate(
                pts_ras=pts,
                blob_ids=blob_ids,
                candidate=proposal,
                inlier_radius_mm=float(inlier_radius_mm),
                axial_bin_mm=float(axial_bin_mm),
                min_span_mm=float(min_span_mm),
                min_inlier_count=int(min_inlier_count),
                elongation_map=elongation_map,
            )
            if proposal is None:
                continue
            proposal["proposal_family"] = "uncovered_rescue"
            proposal["score"] = float(proposal.get("score", 0.0)) + 8.0
            out.append(proposal)
        return out

    def _build_dropout_token_hypotheses(
        self,
        pts_ras,
        blob_ids,
        axis_map,
        elongation_map,
        existing_proposals,
        inlier_radius_mm=2.0,
        axial_bin_mm=2.5,
        min_span_mm=16.0,
        min_inlier_count=5,
        max_passes=6,
        max_pair_seeds=250,
    ):
        pts = np.asarray(pts_ras, dtype=float).reshape(-1, 3)
        blob_ids = np.asarray(blob_ids, dtype=np.int32).reshape(-1)
        ranked = sorted(list(existing_proposals or []), key=self._token_line_candidate_quality, reverse=True)
        out = []
        for proposal in ranked[: int(max(1, max_passes))]:
            removed = set(int(v) for v in list(proposal.get("token_indices") or []))
            if not removed:
                continue
            remaining = np.asarray([idx for idx in range(int(pts.shape[0])) if idx not in removed], dtype=np.int32)
            if remaining.size < max(4, int(min_inlier_count)):
                continue
            sub_pts = pts[remaining]
            sub_blob_ids = blob_ids[remaining]
            rescue = self._build_token_pair_hypotheses(
                pts_ras=sub_pts,
                blob_ids=sub_blob_ids,
                axis_map=axis_map,
                elongation_map=elongation_map,
                inlier_radius_mm=float(max(2.0, inlier_radius_mm)),
                axial_bin_mm=float(axial_bin_mm),
                min_span_mm=float(max(12.0, min_span_mm - 2.0)),
                min_inlier_count=int(max(4, min_inlier_count - 1)),
                max_pair_seeds=int(max(32, max_pair_seeds)),
            )
            remaining_list = remaining.astype(int).tolist()
            for candidate in list(rescue or []):
                mapped = [
                    int(remaining_list[int(v)])
                    for v in list(candidate.get("token_indices") or [])
                    if 0 <= int(v) < len(remaining_list)
                ]
                if not mapped:
                    continue
                candidate = dict(candidate)
                candidate["token_indices"] = mapped
                candidate = self._complete_token_line_candidate(
                    pts_ras=pts,
                    blob_ids=blob_ids,
                    candidate=candidate,
                    inlier_radius_mm=float(inlier_radius_mm),
                    axial_bin_mm=float(axial_bin_mm),
                    min_span_mm=float(min_span_mm),
                    min_inlier_count=int(min_inlier_count),
                    elongation_map=elongation_map,
                )
                if candidate is None:
                    continue
                candidate["proposal_family"] = "dropout_rescue"
                candidate["score"] = float(candidate.get("score", 0.0)) + 6.0
                out.append(candidate)
        return out

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
            min_seed_span_mm = float(max(8.0, min_span_mm - 8.0)) if is_contact_chain_line else float(max(18.0, min_span_mm))
            if span < float(min_seed_span_mm):
                continue
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
                    min_span_mm=float(min_seed_span_mm),
                    min_inlier_count=int(required_token_count),
                    elongation_map=elongation_map,
                    parent_blob_id_map=parent_blob_id_map,
                )
            )
        return proposals

    def _score_token_pair_hypothesis(
        self,
        pts_ras,
        blob_ids,
        seed_i,
        seed_j,
        inlier_radius_mm=2.0,
        axial_bin_mm=2.5,
        min_span_mm=16.0,
        min_inlier_count=5,
        elongation_map=None,
        parent_blob_id_map=None,
    ):
        pts = np.asarray(pts_ras, dtype=float).reshape(-1, 3)
        blob_ids = np.asarray(blob_ids, dtype=np.int32).reshape(-1)
        p0 = np.asarray(pts[int(seed_i)], dtype=float).reshape(3)
        p1 = np.asarray(pts[int(seed_j)], dtype=float).reshape(3)
        axis = p1 - p0
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-6:
            return None
        axis = axis / axis_norm
        center = 0.5 * (p0 + p1)
        radial = self._radial_distance_to_line(pts, center, axis)
        inlier_mask = radial <= float(max(0.1, inlier_radius_mm))
        if int(np.count_nonzero(inlier_mask)) < int(max(3, min_inlier_count)):
            return None
        inlier_pts = pts[inlier_mask]
        inlier_blob_ids = blob_ids[inlier_mask]
        fit_center, fit_axis = self._fit_line_pca(inlier_pts, seed_axis=axis)
        fit_radial = self._radial_distance_to_line(inlier_pts, fit_center, fit_axis)
        rms = float(np.sqrt(np.mean(fit_radial ** 2)))
        if rms > float(max(0.15, inlier_radius_mm)):
            return None
        proj = ((inlier_pts - fit_center.reshape(1, 3)) @ fit_axis.reshape(3, 1)).reshape(-1)
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
        distinct_blobs = int(np.unique(inlier_blob_ids).size)
        parent_blob_ids = sorted(
            set(int(dict(parent_blob_id_map or {}).get(int(blob_id), int(blob_id))) for blob_id in np.unique(inlier_blob_ids).astype(int).tolist())
        )
        distinct_parent_blobs = int(len(parent_blob_ids))
        if distinct_blobs < 2:
            dominant_blob = int(np.bincount(inlier_blob_ids.astype(np.int32)).argmax()) if inlier_blob_ids.size else -1
            dominant_elong = float(dict(elongation_map or {}).get(dominant_blob, 1.0))
            if dominant_elong < 5.0:
                return None
            if span < float(max(24.0, min_span_mm + 6.0)) or int(inlier_pts.shape[0]) < int(max(8, min_inlier_count + 2)):
                return None
        score = (
            0.9 * span
            + 0.8 * float(inlier_pts.shape[0])
            + 6.0 * float(distinct_blobs)
            + 1.5 * float(distinct_parent_blobs)
            + 10.0 * float(occ["coverage"])
            - 2.5 * float(rms)
            - 2.0 * float(occ["max_gap_bins"])
        )
        if distinct_blobs == 1:
            score -= 20.0
        start_ras = fit_center + fit_axis * pmin
        end_ras = fit_center + fit_axis * pmax
        midpoint_ras = 0.5 * (start_ras + end_ras)
        token_indices = np.flatnonzero(inlier_mask).astype(int).tolist()
        return {
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
            "parent_blob_id_list": [int(v) for v in parent_blob_ids],
            "token_indices": [int(v) for v in token_indices],
            "score": float(score),
        }

    def _grow_token_chain_side(
        self,
        pts_ras,
        adj,
        chain,
        chain_set,
        grow_forward=True,
        grow_turn_angle_deg=30.0,
        inlier_radius_mm=2.0,
    ):
        cos_turn = float(np.cos(np.deg2rad(float(max(0.0, grow_turn_angle_deg)))))
        while True:
            if len(chain) < 2:
                return
            if grow_forward:
                end_idx = int(chain[-1])
                prev_idx = int(chain[-2])
                local_axis = np.asarray(pts_ras[end_idx], dtype=float) - np.asarray(pts_ras[prev_idx], dtype=float)
            else:
                end_idx = int(chain[0])
                prev_idx = int(chain[1])
                local_axis = np.asarray(pts_ras[end_idx], dtype=float) - np.asarray(pts_ras[prev_idx], dtype=float)
            axis_norm = float(np.linalg.norm(local_axis))
            if axis_norm <= 1e-6:
                return
            local_axis = local_axis / axis_norm
            fit_center, fit_axis = self._fit_line_pca(
                np.asarray(pts_ras[np.asarray(chain, dtype=int)], dtype=float),
                seed_axis=(np.asarray(pts_ras[chain[-1]], dtype=float) - np.asarray(pts_ras[chain[0]], dtype=float)),
            )
            best = None
            best_score = None
            for nb, edge_dist, edge_dir in list(adj[end_idx] or []):
                nb = int(nb)
                if nb in chain_set:
                    continue
                step = np.asarray(pts_ras[nb], dtype=float) - np.asarray(pts_ras[end_idx], dtype=float)
                step_norm = float(np.linalg.norm(step))
                if step_norm <= 1e-6:
                    continue
                step_dir = step / step_norm
                if float(np.dot(step_dir, local_axis)) < cos_turn:
                    continue
                radial = float(self._radial_distance_to_line(np.asarray(pts_ras[nb], dtype=float).reshape(1, 3), fit_center, fit_axis)[0])
                if radial > float(max(0.1, inlier_radius_mm)):
                    continue
                neighbor_degree = len(list(adj[nb] or []))
                candidate_score = (
                    2.0 * float(np.dot(step_dir, local_axis))
                    + 0.5 * float(edge_dist)
                    - 0.8 * radial
                    - 0.15 * float(neighbor_degree)
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

    @staticmethod

    @staticmethod

    @staticmethod
    def _fit_line_pca(pts_ras, seed_axis=None):
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

    @staticmethod

    @staticmethod

    @staticmethod
    def _radial_distance_to_line(pts_ras, center_ras, axis_ras):
        pts = np.asarray(pts_ras, dtype=float).reshape(-1, 3)
        center = np.asarray(center_ras, dtype=float).reshape(3)
        axis = np.asarray(axis_ras, dtype=float).reshape(3)
        vec = pts - center.reshape(1, 3)
        proj = (vec @ axis.reshape(3, 1)).reshape(-1, 1) * axis.reshape(1, 3)
        radial = vec - proj
        return np.linalg.norm(radial, axis=1)

    @staticmethod

    @staticmethod

    @staticmethod
    def _axial_occupancy_metrics(proj_vals, bin_mm=2.5):
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

    @staticmethod

    @staticmethod

    @staticmethod
    def _occupancy_supports_line(
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

    @staticmethod

    @staticmethod

    @staticmethod
    def _chain_turn_metrics(chain_indices, pts_ras):
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

    @staticmethod

    @staticmethod

    @staticmethod
    def _point_chain_turn_metrics(points_ras):
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

    @staticmethod

    @staticmethod

    @staticmethod
    def _token_line_candidate_quality(candidate):
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

    def _merge_collinear_token_line_candidates(
        self,
        pts_ras,
        blob_ids,
        token_atom_ids,
        candidates,
        inlier_radius_mm=2.0,
        axial_bin_mm=2.5,
        min_span_mm=16.0,
        min_inlier_count=5,
        elongation_map=None,
        parent_blob_id_map=None,
    ):
        ordered = sorted(list(candidates or []), key=self._token_line_candidate_quality, reverse=True)
        merged = list(ordered)
        seen_pairs = set()
        extra = []
        for idx_a, cand_a in enumerate(ordered[:-1]):
            source_a = str(cand_a.get("seed_source_blob_class") or cand_a.get("source_blob_class") or "")
            if source_a != "complex_blob":
                continue
            axis_a = np.asarray(cand_a.get("axis_ras") or [0.0, 0.0, 1.0], dtype=float).reshape(3)
            axis_a = axis_a / max(float(np.linalg.norm(axis_a)), 1e-6)
            start_a = np.asarray(cand_a.get("start_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
            end_a = np.asarray(cand_a.get("end_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
            for cand_b in ordered[(idx_a + 1):]:
                pair_key = (
                    tuple(sorted(int(v) for v in list(cand_a.get("token_indices") or [])[:8])),
                    tuple(sorted(int(v) for v in list(cand_b.get("token_indices") or [])[:8])),
                )
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                axis_b = np.asarray(cand_b.get("axis_ras") or [0.0, 0.0, 1.0], dtype=float).reshape(3)
                axis_b = axis_b / max(float(np.linalg.norm(axis_b)), 1e-6)
                axis_dot = abs(float(np.dot(axis_a, axis_b)))
                if axis_dot < 0.92:
                    continue
                start_b = np.asarray(cand_b.get("start_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
                end_b = np.asarray(cand_b.get("end_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
                endpoint_pairs = [
                    (start_a, start_b),
                    (start_a, end_b),
                    (end_a, start_b),
                    (end_a, end_b),
                ]
                endpoint_gap = min(float(np.linalg.norm(p0 - p1)) for p0, p1 in endpoint_pairs)
                if endpoint_gap > 24.0:
                    continue
                lateral_ab = float(
                    self._radial_distance_to_line(
                        np.vstack((start_b.reshape(1, 3), end_b.reshape(1, 3))),
                        np.asarray(cand_a.get("center_ras") or 0.5 * (start_a + end_a), dtype=float).reshape(3),
                        axis_a,
                    ).mean()
                )
                lateral_ba = float(
                    self._radial_distance_to_line(
                        np.vstack((start_a.reshape(1, 3), end_a.reshape(1, 3))),
                        np.asarray(cand_b.get("center_ras") or 0.5 * (start_b + end_b), dtype=float).reshape(3),
                        axis_b,
                    ).mean()
                )
                if min(lateral_ab, lateral_ba) > 4.0:
                    continue
                token_union = sorted(
                    set(int(v) for v in list(cand_a.get("token_indices") or []))
                    | set(int(v) for v in list(cand_b.get("token_indices") or []))
                )
                if len(token_union) <= max(
                    len(list(cand_a.get("token_indices") or [])),
                    len(list(cand_b.get("token_indices") or [])),
                ):
                    continue
                base = dict(cand_a if self._token_line_candidate_quality(cand_a) >= self._token_line_candidate_quality(cand_b) else cand_b)
                base["token_indices"] = [int(v) for v in token_union]
                base["atom_id_list"] = sorted(
                    set(int(v) for v in list(cand_a.get("atom_id_list") or []))
                    | set(int(v) for v in list(cand_b.get("atom_id_list") or []))
                )
                base["proposal_family"] = "merged_collinear"
                completed = self._complete_token_line_candidate(
                    pts_ras=pts_ras,
                    blob_ids=blob_ids,
                    token_atom_ids=token_atom_ids,
                    candidate=base,
                    inlier_radius_mm=float(inlier_radius_mm),
                    axial_bin_mm=float(axial_bin_mm),
                    min_span_mm=float(min_span_mm),
                    min_inlier_count=int(min_inlier_count),
                    elongation_map=elongation_map,
                    parent_blob_id_map=parent_blob_id_map,
                )
                if completed is None:
                    continue
                if float(completed.get("span_mm", 0.0)) <= max(float(cand_a.get("span_mm", 0.0)), float(cand_b.get("span_mm", 0.0))) + 4.0:
                    continue
                extra.append(completed)
        if extra:
            merged.extend(extra)
        return merged

    def _nms_token_line_candidates(self, candidates, max_output=None):
        ordered = sorted(list(candidates or []), key=self._token_line_candidate_quality, reverse=True)
        survivors = []
        for cand in ordered:
            duplicate = False
            for prev in survivors:
                if self._token_line_candidates_overlap(cand, prev):
                    duplicate = True
                    break
            if not duplicate:
                survivors.append(cand)

        kept = []
        covered_tokens = set()
        covered_blobs = set()
        remaining = list(survivors)
        limit = None if max_output is None else int(max_output)
        if limit is not None and limit <= 0:
            limit = None
        while remaining and (limit is None or len(kept) < limit):
            best_idx = None
            best_gain = None
            for idx, cand in enumerate(remaining):
                token_set = set(int(v) for v in list(cand.get("token_indices") or []))
                blob_set = set(int(v) for v in list(cand.get("blob_id_list") or []))
                new_tokens = int(len(token_set - covered_tokens))
                new_blobs = int(len(blob_set - covered_blobs))
                reused_tokens = int(len(token_set & covered_tokens))
                gain = (
                    self._token_line_candidate_quality(cand)
                    + 1.4 * float(new_tokens)
                    + 6.0 * float(new_blobs)
                    - 0.6 * float(reused_tokens)
                )
                if best_gain is None or gain > best_gain:
                    best_idx = int(idx)
                    best_gain = float(gain)
            if best_idx is None:
                break
            best = remaining.pop(int(best_idx))
            kept.append(best)
            covered_tokens.update(int(v) for v in list(best.get("token_indices") or []))
            covered_blobs.update(int(v) for v in list(best.get("blob_id_list") or []))
        return kept

    def _resolve_contact_chain_stitched_claims(
        self,
        pts_ras,
        blob_ids,
        token_atom_ids,
        proposals,
        support_atoms=None,
        inlier_radius_mm=2.0,
        axial_bin_mm=2.5,
        min_span_mm=16.0,
        min_inlier_count=5,
        elongation_map=None,
        parent_blob_id_map=None,
        max_output=None,
    ):
        kept = [dict(p or {}) for p in list(proposals or [])]
        if not kept:
            return []
        pts = np.asarray(pts_ras, dtype=float).reshape(-1, 3)
        blob_ids = np.asarray(blob_ids, dtype=np.int32).reshape(-1)
        token_atom_ids = np.asarray(token_atom_ids, dtype=np.int32).reshape(-1)
        support_atom_map = {
            int(dict(atom or {}).get("atom_id", -1)): dict(atom or {})
            for atom in list(support_atoms or [])
            if int(dict(atom or {}).get("atom_id", -1)) > 0
        }
        stitched = []
        for proposal in kept:
            family = str(proposal.get("proposal_family") or "")
            atom_id_list = [int(v) for v in list(proposal.get("atom_id_list") or []) if int(v) > 0]
            if family in {"blob_connectivity", "merged_collinear"} or len(atom_id_list) >= 2:
                stitched.append(dict(proposal))
        if not stitched:
            return kept

        adjusted = []
        for proposal in kept:
            family = str(proposal.get("proposal_family") or "")
            source = str(proposal.get("seed_source_blob_class") or proposal.get("source_blob_class") or "")
            atom_id_list = [int(v) for v in list(proposal.get("atom_id_list") or []) if int(v) > 0]
            if not (family == "blob_axis" and source == "contact_chain" and len(atom_id_list) == 1):
                adjusted.append(dict(proposal))
                continue
            seed_atom_id = int(atom_id_list[0])
            seed_atom = dict(support_atom_map.get(int(seed_atom_id)) or {})
            raw_blob_members = {int(v) for v in list(seed_atom.get("parent_blob_id_list") or []) if int(v) > 0}
            if not raw_blob_members:
                adjusted.append(dict(proposal))
                continue
            claimed_raw = set()
            for stitched_prop in stitched:
                claimed_raw.update(int(v) for v in list(stitched_prop.get("blob_id_list") or []) if int(v) in raw_blob_members)
            if not claimed_raw:
                adjusted.append(dict(proposal))
                continue
            remaining_raw = set(int(v) for v in raw_blob_members if int(v) not in claimed_raw)
            if len(remaining_raw) <= 2:
                continue
            base_token_indices = [int(v) for v in list(proposal.get("token_indices") or []) if 0 <= int(v) < int(pts.shape[0])]
            trimmed_token_indices = [
                int(idx)
                for idx in base_token_indices
                if int(token_atom_ids[int(idx)]) == int(seed_atom_id) and int(blob_ids[int(idx)]) in remaining_raw
            ]
            if len(trimmed_token_indices) < 3:
                continue
            trimmed = dict(proposal)
            trimmed["token_indices"] = [int(v) for v in trimmed_token_indices]
            trimmed["blob_id_list"] = [int(v) for v in sorted(remaining_raw)]
            trimmed["parent_blob_id_list"] = [int(v) for v in sorted(remaining_raw)]
            completed = self._complete_token_line_candidate(
                pts_ras=pts,
                blob_ids=blob_ids,
                token_atom_ids=token_atom_ids,
                candidate=trimmed,
                inlier_radius_mm=float(inlier_radius_mm),
                axial_bin_mm=float(axial_bin_mm),
                min_span_mm=float(max(8.0, min_span_mm - 8.0)),
                min_inlier_count=int(min(3, len(trimmed_token_indices))),
                elongation_map=elongation_map,
                parent_blob_id_map=parent_blob_id_map,
            )
            if completed is None:
                continue
            completed["atom_id_list"] = [int(seed_atom_id)]
            adjusted.append(dict(completed))
        return self._nms_token_line_candidates(adjusted, max_output=max_output)

    def _suppress_stitched_subsumed_blob_axis_proposals(self, proposals):
        kept = [dict(p or {}) for p in list(proposals or [])]
        if not kept:
            return []
        stitched = []
        for proposal in kept:
            family = str(proposal.get("proposal_family") or "")
            atom_ids = [int(v) for v in list(proposal.get("atom_id_list") or []) if int(v) > 0]
            if family in {"blob_connectivity", "merged_collinear"} or len(atom_ids) >= 2:
                stitched.append(dict(proposal))
        if not stitched:
            return kept

        survivors = []
        for proposal in kept:
            family = str(proposal.get("proposal_family") or "")
            atom_ids = [int(v) for v in list(proposal.get("atom_id_list") or []) if int(v) > 0]
            if not (family == "blob_axis" and len(atom_ids) == 1):
                survivors.append(dict(proposal))
                continue
            seed_atom_id = int(atom_ids[0])
            covered = False
            for stitched_prop in stitched:
                stitched_atoms = {int(v) for v in list(stitched_prop.get("atom_id_list") or []) if int(v) > 0}
                if seed_atom_id not in stitched_atoms:
                    continue
                if not self._token_line_candidates_overlap(proposal, stitched_prop):
                    continue
                if float(stitched_prop.get("span_mm", 0.0)) < float(proposal.get("span_mm", 0.0)) + 4.0:
                    continue
                covered = True
                break
            if covered:
                continue
            survivors.append(dict(proposal))
        return self._suppress_family_level_duplicates(survivors)

    def _suppress_family_level_duplicates(self, proposals):
        kept = [dict(p or {}) for p in list(proposals or [])]
        if len(kept) < 2:
            return kept

        family_priority = {
            "blob_connectivity": 5,
            "merged_collinear": 4,
            "graph": 4,
            "blob_axis": 3,
            "pair_ransac": 2,
            "uncovered_rescue": 1,
            "dropout_rescue": 1,
            "contact_chain": 1,
        }

        ordered = sorted(
            kept,
            key=lambda p: (
                int(family_priority.get(str(p.get("proposal_family") or ""), 0)),
                float(p.get("span_mm", 0.0)),
                float(p.get("score", 0.0)),
            ),
            reverse=True,
        )
        survivors = []
        for proposal in ordered:
            family = str(proposal.get("proposal_family") or "")
            atom_ids = {int(v) for v in list(proposal.get("atom_id_list") or []) if int(v) > 0}
            drop = False
            for prev in survivors:
                prev_family = str(prev.get("proposal_family") or "")
                prev_atoms = {int(v) for v in list(prev.get("atom_id_list") or []) if int(v) > 0}
                lower_or_equal_priority = int(family_priority.get(family, 0)) <= int(family_priority.get(prev_family, 0))
                atom_subset = bool(atom_ids) and atom_ids.issubset(prev_atoms)
                if lower_or_equal_priority and atom_subset:
                    axis_a = np.asarray(proposal.get("axis_ras") or [0.0, 0.0, 1.0], dtype=float).reshape(3)
                    axis_b = np.asarray(prev.get("axis_ras") or [0.0, 0.0, 1.0], dtype=float).reshape(3)
                    dot = float(np.clip(abs(np.dot(axis_a, axis_b)), -1.0, 1.0))
                    angle_deg = float(np.degrees(np.arccos(dot)))
                    mid_a = np.asarray(proposal.get("midpoint_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
                    mid_b = np.asarray(prev.get("midpoint_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
                    lateral_a = self._radial_distance_to_line(mid_a.reshape(1, 3), np.asarray(prev.get("center_ras"), dtype=float), axis_b)[0]
                    lateral_b = self._radial_distance_to_line(mid_b.reshape(1, 3), np.asarray(proposal.get("center_ras"), dtype=float), axis_a)[0]
                    if angle_deg <= 12.0 and min(float(lateral_a), float(lateral_b)) <= 4.0:
                        drop = True
                        break
                if not self._token_line_candidates_overlap(proposal, prev):
                    continue
                same_extent = self._token_line_candidate_extent_similar(proposal, prev)
                loose_extent = self._token_line_candidate_extent_similar(
                    proposal,
                    prev,
                    endpoint_mm_thresh=16.0,
                    span_ratio_thresh=0.60,
                )
                if lower_or_equal_priority and (atom_subset or same_extent or loose_extent):
                    drop = True
                    break
            if not drop:
                survivors.append(dict(proposal))
        return survivors

    @staticmethod

    @staticmethod

    @staticmethod
    def _token_line_candidate_extent_similar(a, b, endpoint_mm_thresh=10.0, span_ratio_thresh=0.75):
        start_a = np.asarray(a.get("start_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
        end_a = np.asarray(a.get("end_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
        start_b = np.asarray(b.get("start_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
        end_b = np.asarray(b.get("end_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
        axis_a = np.asarray(a.get("axis_ras") or [0.0, 0.0, 1.0], dtype=float).reshape(3)
        axis_b = np.asarray(b.get("axis_ras") or [0.0, 0.0, 1.0], dtype=float).reshape(3)
        if float(np.dot(axis_a, axis_b)) < 0.0:
            start_b, end_b = end_b, start_b
        span_a = float(np.linalg.norm(end_a - start_a))
        span_b = float(np.linalg.norm(end_b - start_b))
        span_ratio = float(min(span_a, span_b) / max(1.0, max(span_a, span_b)))
        if span_ratio < float(span_ratio_thresh):
            return False
        start_err = float(np.linalg.norm(start_a - start_b))
        end_err = float(np.linalg.norm(end_a - end_b))
        return bool(start_err <= float(endpoint_mm_thresh) and end_err <= float(endpoint_mm_thresh))

    def _token_line_candidates_overlap(self, a, b, angle_deg_thresh=8.0, lateral_mm_thresh=3.0):
        tokens_a = set(int(v) for v in list(a.get("token_indices") or []))
        tokens_b = set(int(v) for v in list(b.get("token_indices") or []))
        overlap_ratio = 0.0
        if tokens_a and tokens_b:
            overlap = len(tokens_a & tokens_b)
            overlap_ratio = overlap / float(max(1, min(len(tokens_a), len(tokens_b))))
            if overlap / float(max(1, min(len(tokens_a), len(tokens_b)))) >= 0.35:
                if self._token_line_candidate_extent_similar(a, b):
                    return True
        blobs_a = set(int(v) for v in list(a.get("blob_id_list") or []))
        blobs_b = set(int(v) for v in list(b.get("blob_id_list") or []))
        shared_blob_count = len(blobs_a & blobs_b)
        axis_a = np.asarray(a.get("axis_ras") or [0.0, 0.0, 1.0], dtype=float).reshape(3)
        axis_b = np.asarray(b.get("axis_ras") or [0.0, 0.0, 1.0], dtype=float).reshape(3)
        dot = float(np.clip(abs(np.dot(axis_a, axis_b)), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(dot)))
        if angle_deg > float(angle_deg_thresh):
            return False
        mid_a = np.asarray(a.get("midpoint_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
        mid_b = np.asarray(b.get("midpoint_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
        lateral_a = self._radial_distance_to_line(mid_a.reshape(1, 3), np.asarray(b.get("center_ras"), dtype=float), axis_b)[0]
        lateral_b = self._radial_distance_to_line(mid_b.reshape(1, 3), np.asarray(a.get("center_ras"), dtype=float), axis_a)[0]
        if min(float(lateral_a), float(lateral_b)) > float(lateral_mm_thresh):
            return False
        # For deep-core debugging, preserve recall: do not suppress nearby lines
        # unless they reuse support tokens or parent blobs *and* represent the
        # same effective extent. Different-length ownership variants should
        # survive so debug recall does not collapse to stitched supersets.
        if shared_blob_count > 0 and self._token_line_candidate_extent_similar(a, b):
            return True
        shared_blob_ratio = float(shared_blob_count) / float(max(1, min(len(blobs_a), len(blobs_b))))
        span_a = float(a.get("span_mm", 0.0))
        span_b = float(b.get("span_mm", 0.0))
        if span_a <= span_b:
            shorter = a
            longer = b
        else:
            shorter = b
            longer = a
        shorter_span = float(shorter.get("span_mm", 0.0))
        longer_span = float(longer.get("span_mm", 0.0))
        shorter_cov = float(shorter.get("coverage", 0.0))
        longer_cov = float(longer.get("coverage", 0.0))
        shorter_gap = int(shorter.get("max_gap_bins", 0))
        longer_gap = int(longer.get("max_gap_bins", 0))
        # Suppress through-going variants that reuse the same core support but
        # append a sparse tail through a different corridor.
        if (
            max(float(overlap_ratio), float(shared_blob_ratio)) >= 0.70
            and longer_span >= 1.35 * max(1.0, shorter_span)
            and longer_cov + 0.20 <= shorter_cov
            and longer_gap >= max(6, shorter_gap + 4)
        ):
            return True
        return False
