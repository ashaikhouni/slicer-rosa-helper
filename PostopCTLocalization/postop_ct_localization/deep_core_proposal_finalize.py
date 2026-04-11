"""Proposal extension and final pruning for Deep Core."""

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

from shank_core.masking import build_preview_masks
from .deep_core_config import deep_core_default_config


_DEEP_CORE_DEFAULTS = deep_core_default_config()
_DEFAULT_ANNULUS_CONFIG = _DEEP_CORE_DEFAULTS.annulus
_DEFAULT_INTERNAL_CONFIG = _DEEP_CORE_DEFAULTS.internal


class DeepCoreProposalFinalizeMixin:
    """Extend proposals and apply post-extension pruning."""

    def _reject_short_floating_proposals(
        self,
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
            start_depth = self._depth_at_ras_with_volume(volume_node, head_distance_map_kji, start_ras)
            end_depth = self._depth_at_ras_with_volume(volume_node, head_distance_map_kji, end_ras)
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

    def _extract_support_candidate_points_lps(
        self,
        volume_node,
        threshold_hu,
        head_mask_threshold_hu=-500.0,
        min_metal_depth_mm=5.0,
        max_metal_depth_mm=220.0,
    ):
        if volume_node is None:
            return {"points_lps": np.empty((0, 3), dtype=float), "threshold_hu": float(threshold_hu)}
        # Prefer the VolumeAccessor if available on the pipeline.
        vol = getattr(self, "_vol", None)
        if vol is not None and hasattr(vol, "extract_threshold_candidates_lps"):
            return vol.extract_threshold_candidates_lps(
                volume_node=volume_node,
                threshold=float(threshold_hu),
                head_mask_threshold_hu=float(head_mask_threshold_hu),
                min_metal_depth_mm=float(min_metal_depth_mm),
                max_metal_depth_mm=float(max_metal_depth_mm),
            )
        # Legacy path: call through host_logic proxy.
        if hasattr(volume_node, "GetIJKToRASMatrix") and hasattr(self, "extract_threshold_candidates_lps"):
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

    def _support_candidate_points_from_mask(
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

    def _extend_shallow_endpoint_along_axis(self, 
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
                "shallow_depth_mm": self._depth_at_ras_with_volume(
                    volume_node, head_distance_map_kji, start_ras
                ),
                "extended_depth_mm": self._depth_at_ras_with_volume(
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
                "shallow_depth_mm": self._depth_at_ras_with_volume(
                    volume_node, head_distance_map_kji, start_ras
                ),
                "extended_depth_mm": self._depth_at_ras_with_volume(
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

        def support_bin_state(bin_id):
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
            has_support, is_thick = support_bin_state(b)
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
            has_support, is_thick = support_bin_state(b)
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
        start_depth = self._depth_at_ras_with_volume(volume_node, head_distance_map_kji, new_start)
        end_depth = self._depth_at_ras_with_volume(volume_node, head_distance_map_kji, new_end)
        return {
            "start_ras": new_start,
            "end_ras": new_end,
            "axis_ras": axis_final,
            "extended_mm": float(max(0.0, (new_end_offset_mm - new_start_offset_mm) - seed_span_mm)),
            "shallow_depth_mm": None if start_depth is None or end_depth is None else float(min(start_depth, end_depth)),
            "extended_depth_mm": None if start_depth is None or end_depth is None else float(max(start_depth, end_depth)),
        }

    def _reject_small_proposals_without_outward_support(
        self,
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
        support_pts_ras, support_depths = self._support_candidate_points_from_mask(
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
            start_depth = self._depth_at_ras_with_volume(volume_node, head_distance_map_kji, start_ras)
            end_depth = self._depth_at_ras_with_volume(volume_node, head_distance_map_kji, end_ras)
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

    def _finalize_extended_proposals(
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
        line_atom_max_diameter_mm=5.0,
        annulus_flag_percentile=80.0,
        annulus_flag_min_samples=120,
        annulus_reference_upper_hu=2500.0,
        annulus_config=None,
        internal_config=None,
    ):
        annulus_cfg = annulus_config if annulus_config is not None else _DEFAULT_ANNULUS_CONFIG
        internal_cfg = internal_config if internal_config is not None else _DEFAULT_INTERNAL_CONFIG
        proposal_list = [dict(p or {}) for p in list(proposals or [])]
        if not proposal_list:
            return []
        annulus_reference_values_hu = self._scan_reference_hu_values(
            volume_node=volume_node,
            lower_hu=-500.0,
            upper_hu=float(annulus_reference_upper_hu) if annulus_reference_upper_hu is not None else None,
        )
        if guided_candidate_mask_kji is not None:
            support_pts_ras, support_depths = self._support_candidate_points_from_mask(
                volume_node=volume_node,
                candidate_mask_kji=guided_candidate_mask_kji,
                head_distance_map_kji=head_distance_map_kji,
                min_metal_depth_mm=0.0,
            )
            support_threshold_hu = float(threshold_hu)
        else:
            candidate_payload = self._extract_support_candidate_points_lps(
                volume_node=volume_node,
                threshold_hu=float(threshold_hu),
                head_mask_threshold_hu=float(head_mask_threshold_hu),
            )
            candidate_points_lps = np.asarray(candidate_payload.get("points_lps"), dtype=float).reshape(-1, 3)
            support_pts_ras = candidate_points_lps.copy()
            support_pts_ras[:, 0] *= -1.0
            support_pts_ras[:, 1] *= -1.0
            support_depths = np.zeros((support_pts_ras.shape[0],), dtype=float)
            support_threshold_hu = float(candidate_payload.get("threshold_hu", threshold_hu))
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
                support_radius_mm=float(internal_cfg.extension_support_radius_mm),
                outward_search_mm=float(internal_cfg.extension_outward_search_mm),
                inward_search_mm=float(internal_cfg.extension_inward_search_mm),
                bin_mm=float(internal_cfg.extension_bin_mm),
                max_gap_bins=int(internal_cfg.extension_max_gap_bins),
                max_local_diameter_mm=float(max(0.5, 1.1 * float(line_atom_max_diameter_mm))),
                thick_run_bins=int(internal_cfg.extension_thick_run_bins),
            )
            proposal["start_ras"] = [float(v) for v in np.asarray(grown.get("start_ras"), dtype=float).reshape(3)]
            proposal["end_ras"] = [float(v) for v in np.asarray(grown.get("end_ras"), dtype=float).reshape(3)]
            proposal["axis_ras"] = [float(v) for v in np.asarray(grown.get("axis_ras"), dtype=float).reshape(3)]
            proposal["span_mm"] = float(
                np.linalg.norm(np.asarray(proposal["end_ras"], dtype=float) - np.asarray(proposal["start_ras"], dtype=float))
            )
            proposal["support_threshold_hu"] = float(support_threshold_hu)
            proposal["axis_growth_extension_mm"] = float(grown.get("extended_mm", 0.0) or 0.0)
            proposal["axis_growth_shallow_depth_mm"] = grown.get("shallow_depth_mm")
            proposal["axis_growth_extended_depth_mm"] = grown.get("extended_depth_mm")
            proposal = self._annotate_proposal_endpoint_roles(
                volume_node=volume_node,
                proposal=proposal,
                head_distance_map_kji=head_distance_map_kji,
            )
            shallow_annulus_stats = self._proposal_shallow_endpoint_annulus_mean_ct_hu(
                volume_node=volume_node,
                proposal=proposal,
                head_distance_map_kji=head_distance_map_kji,
                annulus_inner_mm=float(annulus_cfg.line_annulus_inner_mm),
                annulus_outer_mm=float(annulus_cfg.line_annulus_outer_mm),
            )
            proposal["shallow_endpoint_annulus_mean_ct_hu"] = shallow_annulus_stats.get("mean_hu")
            proposal["shallow_endpoint_annulus_sample_count"] = int(shallow_annulus_stats.get("sample_count", 0) or 0)
            proposal = self._annotate_single_proposal_with_annulus_stats(
                volume_node=volume_node,
                proposal=proposal,
                reference_values_hu=annulus_reference_values_hu,
                mean_key="annulus_mean_ct_hu",
                median_key="annulus_median_ct_hu",
                sample_key="annulus_sample_count",
                percentile_key="annulus_scan_percentile",
                flag_key="annulus_bone_suspicious",
                annulus_flag_percentile=float(annulus_flag_percentile),
                annulus_flag_min_samples=int(annulus_cfg.annulus_flag_min_samples),
                annulus_config=annulus_cfg,
            )
            proposal["best_model_id"] = ""
            proposal["best_model_score"] = 0.0
            accepted.append(proposal)
        return accepted

