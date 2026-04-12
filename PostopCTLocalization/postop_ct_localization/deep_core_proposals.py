"""Deep-core proposal generation: orchestration and config-backed wrapper."""

import numpy as np

from .deep_core_annulus import DeepCoreAnnulusMixin
from .deep_core_candidate_dedup import DeepCoreCandidateDedupMixin
from .deep_core_candidate_generation import DeepCoreCandidateGenerationMixin
from .deep_core_candidate_inputs import DeepCoreCandidateInputMixin
from .deep_core_config import deep_core_default_config
from .deep_core_proposal_annulus import DeepCoreProposalAnnulusMixin
from .deep_core_proposal_finalize import DeepCoreProposalFinalizeMixin


_DEEP_CORE_DEFAULTS = deep_core_default_config()
_DEFAULT_MASK_CONFIG = _DEEP_CORE_DEFAULTS.mask
_DEFAULT_SUPPORT_CONFIG = _DEEP_CORE_DEFAULTS.support
_DEFAULT_PROPOSAL_CONFIG = _DEEP_CORE_DEFAULTS.proposal
_DEFAULT_ANNULUS_CONFIG = _DEEP_CORE_DEFAULTS.annulus
_DEFAULT_INTERNAL_CONFIG = _DEEP_CORE_DEFAULTS.internal


class DeepCoreProposalLogicMixin(
    DeepCoreCandidateInputMixin,
    DeepCoreCandidateGenerationMixin,
    DeepCoreCandidateDedupMixin,
    DeepCoreProposalFinalizeMixin,
    DeepCoreProposalAnnulusMixin,
    DeepCoreAnnulusMixin,
):
    """Own the raw proposal, rejection, extension, and final pruning stages."""

    def _build_deep_core_raw_proposals_stage(
        self,
        volume_node,
        support_result,
        mask_config=None,
        support_config=None,
        proposal_config=None,
        annulus_config=None,
        internal_config=None,
    ):
        """Build raw candidate proposals before annulus rejection or extension."""

        mask_cfg = mask_config if mask_config is not None else _DEFAULT_MASK_CONFIG
        support_cfg = support_config if support_config is not None else _DEFAULT_SUPPORT_CONFIG
        proposal_cfg = proposal_config if proposal_config is not None else _DEFAULT_PROPOSAL_CONFIG
        _annulus_cfg = annulus_config if annulus_config is not None else _DEFAULT_ANNULUS_CONFIG
        _internal_cfg = internal_config if internal_config is not None else _DEFAULT_INTERNAL_CONFIG
        _resolved = getattr(support_result, "combined_payload", getattr(support_result, "payload", support_result))
        support_payload = _resolved if hasattr(_resolved, "get") else dict(_resolved or {})
        prepared = self._prepare_support_atom_inputs(
            support_atoms=support_payload.get("support_atoms"),
            token_points_ras=support_payload.get("blob_sample_points_ras"),
            token_blob_ids=support_payload.get("blob_sample_blob_ids"),
            token_atom_ids=support_payload.get("blob_sample_atom_ids"),
            blob_axes_ras_by_id=support_payload.get("blob_axes_ras_by_id"),
            blob_elongation_by_id=support_payload.get("blob_elongation_by_id"),
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
            neighbor_max_mm=float(proposal_cfg.neighbor_max_mm),
            bridge_max_mm=float(max(10.0, proposal_cfg.neighbor_max_mm + 8.0)),
            bridge_axis_angle_deg=float(max(30.0, proposal_cfg.neighbor_axis_angle_deg + 5.0)),
        )
        seed_edges = list(blob_graph.get("local_seed_edges") or [])
        blob_seed_edges = list(blob_graph.get("bridge_seed_edges") or [])
        candidates = []
        for family, family_seed_edges in (("graph", seed_edges), ("blob_connectivity", blob_seed_edges)):
            for _seed_score, blob_i, blob_j in family_seed_edges[: int(max(1, proposal_cfg.max_seed_edges))]:
                proposal = self._grow_blob_chain_candidate(
                    pts_ras=pts,
                    blob_ids=blob_ids,
                    blob_graph=blob_graph,
                    seed_blob_i=int(blob_i),
                    seed_blob_j=int(blob_j),
                    grow_turn_angle_deg=float(proposal_cfg.grow_turn_angle_deg),
                    inlier_radius_mm=float(proposal_cfg.inlier_radius_mm),
                    axial_bin_mm=float(proposal_cfg.axial_bin_mm),
                    min_span_mm=float(proposal_cfg.min_span_mm),
                    min_inlier_count=int(proposal_cfg.min_inlier_count),
                    parent_blob_id_map=parent_blob_id_map,
                )
                if proposal is None:
                    continue
                proposal = self._complete_token_line_candidate(
                    pts_ras=pts,
                    blob_ids=blob_ids,
                    token_atom_ids=token_atom_ids,
                    candidate=proposal,
                    inlier_radius_mm=float(proposal_cfg.inlier_radius_mm),
                    axial_bin_mm=float(proposal_cfg.axial_bin_mm),
                    min_span_mm=float(proposal_cfg.min_span_mm),
                    min_inlier_count=int(proposal_cfg.min_inlier_count),
                    elongation_map=elong_map,
                    parent_blob_id_map=parent_blob_id_map,
                )
                if proposal is None:
                    continue
                proposal["proposal_family"] = str(family)
                candidates.append(proposal)
        candidates.extend(
            self._build_blob_axis_hypotheses(
                pts_ras=pts,
                blob_ids=blob_ids,
                token_atom_ids=token_atom_ids,
                support_atoms=prepared.get("support_atoms"),
                elongation_map=elong_map,
                axial_bin_mm=float(proposal_cfg.axial_bin_mm),
                min_span_mm=float(proposal_cfg.min_span_mm),
                min_inlier_count=int(proposal_cfg.min_inlier_count),
                parent_blob_id_map=parent_blob_id_map,
            )
        )
        candidates = self._merge_collinear_token_line_candidates(
            pts_ras=pts,
            blob_ids=blob_ids,
            token_atom_ids=token_atom_ids,
            candidates=candidates,
            inlier_radius_mm=float(proposal_cfg.inlier_radius_mm),
            axial_bin_mm=float(proposal_cfg.axial_bin_mm),
            min_span_mm=float(proposal_cfg.min_span_mm),
            min_inlier_count=int(proposal_cfg.min_inlier_count),
            elongation_map=elong_map,
            parent_blob_id_map=parent_blob_id_map,
        )
        proposals = self._nms_token_line_candidates(candidates, max_output=proposal_cfg.max_output_proposals)
        proposals = self._resolve_contact_chain_stitched_claims(
            pts_ras=pts,
            blob_ids=blob_ids,
            token_atom_ids=token_atom_ids,
            proposals=proposals,
            support_atoms=prepared.get("support_atoms"),
            inlier_radius_mm=float(proposal_cfg.inlier_radius_mm),
            axial_bin_mm=float(proposal_cfg.axial_bin_mm),
            min_span_mm=float(proposal_cfg.min_span_mm),
            min_inlier_count=int(proposal_cfg.min_inlier_count),
            elongation_map=elong_map,
            parent_blob_id_map=parent_blob_id_map,
            max_output=proposal_cfg.max_output_proposals,
        )
        proposals = self._suppress_stitched_subsumed_blob_axis_proposals(proposals=proposals)
        return {
            "proposals": proposals,
            "candidate_count": int(len(candidates)),
            "token_count": int(pts.shape[0]),
            "token_points_ras": pts,
            "token_blob_ids": blob_ids,
        }

    def _apply_pre_extension_annulus_rejection_stage(
        self,
        volume_node,
        support_result,
        proposal_payload,
        annulus_config=None,
    ):
        """Annotate and reject bone-adjacent proposals before extension."""

        annulus_cfg = annulus_config if annulus_config is not None else _DEFAULT_ANNULUS_CONFIG
        payload = dict(proposal_payload or {})
        proposals = self._annotate_proposals_with_annulus_stats(
            volume_node=volume_node,
            proposals=payload.get("proposals"),
            mean_key="pre_extension_annulus_mean_ct_hu",
            median_key="pre_extension_annulus_median_ct_hu",
            sample_key="pre_extension_annulus_sample_count",
            percentile_key="pre_extension_annulus_scan_percentile",
            flag_key="pre_extension_annulus_bone_suspicious",
            annulus_flag_percentile=float(annulus_cfg.pre_extension_annulus_reject_percentile),
            annulus_reference_upper_hu=float(annulus_cfg.annulus_reference_upper_hu),
            annulus_config=annulus_cfg,
        )
        payload["proposals"] = self._reject_bone_adjacency_proposals(
            proposals=proposals,
            flag_key="pre_extension_annulus_bone_suspicious",
        )
        return payload

    def _extend_deep_core_proposals_stage(
        self,
        volume_node,
        support_result,
        proposal_payload,
        mask_config=None,
        support_config=None,
        proposal_config=None,
        annulus_config=None,
        internal_config=None,
    ):
        """Extend surviving proposals and annotate final annulus stats."""

        mask_cfg = mask_config if mask_config is not None else _DEFAULT_MASK_CONFIG
        support_cfg = support_config if support_config is not None else _DEFAULT_SUPPORT_CONFIG
        proposal_cfg = proposal_config if proposal_config is not None else _DEFAULT_PROPOSAL_CONFIG
        annulus_cfg = annulus_config if annulus_config is not None else _DEFAULT_ANNULUS_CONFIG
        internal_cfg = internal_config if internal_config is not None else _DEFAULT_INTERNAL_CONFIG
        payload = dict(proposal_payload or {})
        _resolved = getattr(support_result, "combined_payload", getattr(support_result, "payload", support_result))
        support_payload = _resolved if hasattr(_resolved, "get") else dict(_resolved or {})
        proposals = self._finalize_extended_proposals(
            volume_node=volume_node,
            proposals=payload.get("proposals"),
            token_points_ras=payload.get("token_points_ras"),
            token_blob_ids=payload.get("token_blob_ids"),
            threshold_hu=float(proposal_cfg.guided_threshold_hu if proposal_cfg.guided_threshold_hu is not None else mask_cfg.metal_threshold_hu),
            head_mask_threshold_hu=float(proposal_cfg.guided_head_mask_threshold_hu if proposal_cfg.guided_head_mask_threshold_hu is not None else mask_cfg.hull_threshold_hu),
            roi_radius_mm=float(proposal_cfg.guided_roi_radius_mm),
            max_angle_deg=float(proposal_cfg.guided_max_angle_deg),
            max_depth_shift_mm=float(proposal_cfg.guided_max_depth_shift_mm),
            fit_mode=str(proposal_cfg.guided_fit_mode or "deep_anchor_v2"),
            max_residual_mm=float(proposal_cfg.guided_max_residual_mm),
            head_distance_map_kji=support_payload.get("head_distance_map_kji"),
            guided_candidate_mask_kji=support_payload.get("metal_grown_mask_kji"),
            line_atom_max_diameter_mm=float(support_cfg.line_atom_diameter_max_mm),
            annulus_reference_upper_hu=float(annulus_cfg.annulus_reference_upper_hu),
            annulus_config=annulus_cfg,
            internal_config=internal_cfg,
        )
        payload["proposals"] = proposals
        return payload

    def _apply_final_deep_core_rejection_stage(
        self,
        volume_node,
        support_result,
        proposal_payload,
        mask_config=None,
        proposal_config=None,
    ):
        """Run final float/outward-support pruning after extension."""

        mask_cfg = mask_config if mask_config is not None else _DEFAULT_MASK_CONFIG
        proposal_cfg = proposal_config if proposal_config is not None else _DEFAULT_PROPOSAL_CONFIG
        payload = dict(proposal_payload or {})
        _resolved = getattr(support_result, "combined_payload", getattr(support_result, "payload", support_result))
        support_payload = _resolved if hasattr(_resolved, "get") else dict(_resolved or {})
        proposals = self._reject_short_floating_proposals(
            volume_node=volume_node,
            proposals=payload.get("proposals"),
            head_distance_map_kji=support_payload.get("head_distance_map_kji"),
            deep_core_shrink_mm=float(mask_cfg.deep_core_shrink_mm),
            min_span_mm=float(proposal_cfg.short_float_reject_span_mm),
            edge_tol_mm=float(proposal_cfg.short_float_edge_tol_mm),
        )
        proposals = self._reject_small_proposals_without_outward_support(
            volume_node=volume_node,
            proposals=proposals,
            candidate_mask_kji=support_payload.get("metal_grown_mask_kji"),
            head_distance_map_kji=support_payload.get("head_distance_map_kji"),
            max_checked_span_mm=float(proposal_cfg.outward_support_check_span_mm),
            support_radius_mm=float(proposal_cfg.outward_support_radius_mm),
            outward_search_mm=float(proposal_cfg.outward_support_search_mm),
            min_extension_mm=float(proposal_cfg.outward_support_min_extension_mm),
            min_depth_gain_mm=float(proposal_cfg.outward_support_min_depth_gain_mm),
            edge_tol_mm=float(proposal_cfg.short_float_edge_tol_mm),
        )
        payload["proposals"] = self._suppress_stitched_subsumed_blob_axis_proposals(proposals=proposals)
        return payload

