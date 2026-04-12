"""Proposal annulus scoring and rejection helpers for Deep Core."""

import numpy as np

from shank_core.io import kji_to_ras_points, kji_to_ras_points_matrix

from .deep_core_annulus import DeepCoreAnnulusMixin
from .deep_core_config import deep_core_default_config


_DEEP_CORE_DEFAULTS = deep_core_default_config()
_DEFAULT_ANNULUS_CONFIG = _DEEP_CORE_DEFAULTS.annulus


class DeepCoreProposalAnnulusMixin(DeepCoreAnnulusMixin):
    """Annulus-based proposal annotation and pre-extension rejection."""

    def _kji_to_ras_points_for_volume(self, volume_node, ijk_kji):
        idx = np.asarray(ijk_kji, dtype=float).reshape(-1, 3)
        if idx.size == 0:
            return np.empty((0, 3), dtype=float)
        # If volume_node carries a context fn (proxy), use it directly.
        ctx_fn = getattr(volume_node, "_ijk_kji_to_ras_fn", None)
        if ctx_fn is not None:
            return np.asarray(ctx_fn(idx), dtype=float).reshape(-1, 3)
        # Real Slicer volume path: try via VTK if available.
        if volume_node is not None and hasattr(volume_node, "GetIJKToRASMatrix"):
            try:
                from __main__ import vtk  # noqa: deferred
            except ImportError:
                try:
                    import vtk
                except ImportError:
                    vtk = None
            if vtk is not None:
                m_vtk = vtk.vtkMatrix4x4()
                volume_node.GetIJKToRASMatrix(m_vtk)
                mat = np.eye(4, dtype=float)
                for r in range(4):
                    for c in range(4):
                        mat[r, c] = float(m_vtk.GetElement(r, c))
                return kji_to_ras_points_matrix(idx, mat)
        return np.asarray(kji_to_ras_points(volume_node, idx), dtype=float).reshape(-1, 3)

    @classmethod
    def _line_annulus_mean_ct_hu(
        cls,
        volume_node,
        start_ras,
        axis_ras,
        span_mm,
        annulus_inner_mm=2.5,
        annulus_outer_mm=3.5,
        axial_step_mm=2.0,
        radial_steps=2,
        angular_samples=12,
        axial_start_mm=None,
        axial_end_mm=None,
    ):
        arr_kji = cls._volume_array_kji(volume_node)
        ras_to_ijk_fn = cls._ras_to_ijk_fn_for_volume(volume_node)
        if arr_kji is None or ras_to_ijk_fn is None:
            return {"mean_hu": None, "sample_count": 0}
        start = np.asarray([0.0, 0.0, 0.0] if start_ras is None else start_ras, dtype=float).reshape(3)
        axis, u, v = cls._orthonormal_basis_for_axis(axis_ras)
        span_mm = float(max(0.0, span_mm))
        if span_mm <= 1e-6:
            return {"mean_hu": None, "sample_count": 0}
        axial_lo = 0.0 if axial_start_mm is None else float(np.clip(float(axial_start_mm), 0.0, span_mm))
        axial_hi = span_mm if axial_end_mm is None else float(np.clip(float(axial_end_mm), 0.0, span_mm))
        if axial_hi < axial_lo:
            axial_lo, axial_hi = axial_hi, axial_lo
        axial_positions = np.arange(axial_lo, axial_hi + 1e-6, float(max(0.5, axial_step_mm)), dtype=float)
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
            return {"mean_hu": None, "median_hu": None, "sample_count": 0}
        sample_arr = np.asarray(samples, dtype=float)
        return {
            "mean_hu": float(np.mean(sample_arr)),
            "median_hu": float(np.median(sample_arr)),
            "sample_count": int(sample_arr.size),
        }

    @classmethod
    def _annotate_single_proposal_with_annulus_stats(
        cls,
        volume_node,
        proposal,
        reference_values_hu=None,
        mean_key="annulus_mean_ct_hu",
        median_key="annulus_median_ct_hu",
        sample_key="annulus_sample_count",
        percentile_key="annulus_scan_percentile",
        flag_key="annulus_bone_suspicious",
        annulus_flag_percentile=80.0,
        annulus_flag_min_samples=120,
        sibling_blob_proposal_count=1,
        annulus_config=None,
    ):
        annulus_cfg = annulus_config if annulus_config is not None else _DEFAULT_ANNULUS_CONFIG
        proposal_dict = dict(proposal or {})
        annulus_stats = cls._proposal_annulus_mean_ct_hu(
            volume_node=volume_node,
            proposal=proposal_dict,
            annulus_inner_mm=float(annulus_cfg.line_annulus_inner_mm),
            annulus_outer_mm=float(annulus_cfg.line_annulus_outer_mm),
            radial_steps=int(annulus_cfg.annulus_radial_steps),
            angular_samples=int(annulus_cfg.annulus_angular_samples),
        )
        proposal_dict[str(mean_key)] = annulus_stats.get("mean_hu")
        proposal_dict[str(median_key)] = annulus_stats.get("median_hu")
        proposal_dict[str(sample_key)] = int(annulus_stats.get("sample_count", 0) or 0)
        annulus_median_hu = proposal_dict.get(str(median_key))
        annulus_sample_count = int(proposal_dict.get(str(sample_key), 0) or 0)
        annulus_percentile = cls._value_percentile_from_sorted(reference_values_hu, annulus_median_hu)
        proposal_dict[str(percentile_key)] = annulus_percentile
        proposal_dict[str(flag_key)] = bool(
            annulus_percentile is not None
            and annulus_sample_count >= int(max(1, annulus_flag_min_samples))
            and float(annulus_percentile) >= float(annulus_flag_percentile)
        )
        profile_stats = cls._proposal_annulus_profile_gate_stats(
            volume_node=volume_node,
            proposal=proposal_dict,
            reference_values_hu=reference_values_hu,
            brainlike_percentile=float(annulus_flag_percentile),
            sibling_blob_proposal_count=int(max(1, sibling_blob_proposal_count)),
            annulus_config=annulus_cfg,
        )
        for key, value in dict(profile_stats or {}).items():
            proposal_dict[str(key)] = value
        profile_flag = bool(proposal_dict.get("annulus_profile_bone_suspicious"))
        proposal_dict[str(flag_key)] = bool(proposal_dict.get(str(flag_key)) or profile_flag)
        return proposal_dict

    @staticmethod
    def _boolean_runs(values):
        runs = []
        start = None
        for idx, val in enumerate(list(values or [])):
            if bool(val):
                if start is None:
                    start = int(idx)
            elif start is not None:
                runs.append((int(start), int(idx - 1)))
                start = None
        if start is not None:
            runs.append((int(start), int(len(list(values or [])) - 1)))
        return runs

    @classmethod
    def _proposal_annulus_profile_gate_stats(
        cls,
        volume_node,
        proposal,
        reference_values_hu=None,
        brainlike_percentile=80.0,
        annulus_inner_mm=3.0,
        annulus_outer_mm=4.0,
        axial_step_mm=1.0,
        radial_steps=2,
        angular_samples=12,
        min_samples=8,
        remaining_brain_fraction_req=0.80,
        simple_middle_nonbrain_max_run_mm=1.0,
        multi_middle_nonbrain_max_run_mm=3.0,
        endpoint_margin_mm=2.0,
        sibling_blob_proposal_count=1,
        annulus_config=None,
    ):
        annulus_cfg = annulus_config if annulus_config is not None else _DEFAULT_ANNULUS_CONFIG
        proposal_dict = dict(proposal or {})
        start = np.asarray(proposal_dict.get("start_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
        end = np.asarray(proposal_dict.get("end_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
        axis = end - start
        span_mm = float(np.linalg.norm(axis))
        step_mm = float(max(0.5, axial_step_mm))
        if span_mm <= step_mm:
            return {
                "annulus_profile_sample_count": 0,
                "annulus_profile_brain_fraction": None,
                "annulus_profile_remaining_brain_fraction": None,
                "annulus_profile_endpoint_nonbrain_run_mm": None,
                "annulus_profile_middle_nonbrain_run_mm": None,
                "annulus_profile_allowed_endpoint": "",
                "annulus_profile_bone_suspicious": False,
            }
        axis = axis / max(float(np.linalg.norm(axis)), 1e-6)
        step_mm = float(max(0.5, annulus_cfg.profile_axial_step_mm))
        sample_positions = np.arange(0.0, span_mm + 0.5 * step_mm, step_mm, dtype=float)
        valid_positions = []
        valid_percentiles = []
        for pos_mm in sample_positions.tolist():
            center_ras = start + float(pos_mm) * axis
            ann = cls._cross_section_annulus_mean_ct_hu(
                volume_node=volume_node,
                center_ras=center_ras,
                axis_ras=axis,
                annulus_inner_mm=float(annulus_cfg.cross_section_annulus_inner_mm),
                annulus_outer_mm=float(annulus_cfg.cross_section_annulus_outer_mm),
                radial_steps=int(annulus_cfg.annulus_radial_steps),
                angular_samples=int(annulus_cfg.annulus_angular_samples),
            )
            med = ann.get("median_hu")
            pct = cls._value_percentile_from_sorted(reference_values_hu, med)
            if pct is None:
                continue
            valid_positions.append(float(pos_mm))
            valid_percentiles.append(float(pct))
        if len(valid_percentiles) < int(max(2, annulus_cfg.profile_min_samples)):
            return {
                "annulus_profile_sample_count": int(len(valid_percentiles)),
                "annulus_profile_brain_fraction": None,
                "annulus_profile_remaining_brain_fraction": None,
                "annulus_profile_endpoint_nonbrain_run_mm": None,
                "annulus_profile_middle_nonbrain_run_mm": None,
                "annulus_profile_allowed_endpoint": "",
                "annulus_profile_bone_suspicious": False,
            }
        brainlike = [bool(float(pct) < float(brainlike_percentile)) for pct in valid_percentiles]
        nonbrain_runs = cls._boolean_runs([not val for val in brainlike])
        profile_len = int(len(brainlike))
        start_run = None
        end_run = None
        for run_start, run_end in nonbrain_runs:
            if int(run_start) == 0:
                start_run = (int(run_start), int(run_end))
            if int(run_end) == int(profile_len - 1):
                end_run = (int(run_start), int(run_end))
        allowed_endpoint = ""
        allowed_run = None
        if start_run is not None and end_run is None:
            allowed_endpoint = "start"
            allowed_run = start_run
        elif end_run is not None and start_run is None:
            allowed_endpoint = "end"
            allowed_run = end_run
        elif start_run is not None and end_run is not None:
            start_len = int(start_run[1] - start_run[0] + 1)
            end_len = int(end_run[1] - end_run[0] + 1)
            if start_len >= end_len:
                allowed_endpoint = "start"
                allowed_run = start_run
            else:
                allowed_endpoint = "end"
                allowed_run = end_run
        remaining_flags = list(brainlike)
        endpoint_nonbrain_run_mm = 0.0
        if allowed_run is not None:
            run_start, run_end = allowed_run
            endpoint_nonbrain_run_mm = float(max(1, int(run_end - run_start + 1)) * step_mm)
            for idx in range(int(run_start), int(run_end + 1)):
                if 0 <= int(idx) < len(remaining_flags):
                    remaining_flags[int(idx)] = True
        remaining_brain_fraction = float(np.mean(remaining_flags)) if remaining_flags else 1.0
        middle_nonbrain_runs = cls._boolean_runs([not val for val in remaining_flags])
        middle_nonbrain_run_mm = float(
            max([int(run_end - run_start + 1) for run_start, run_end in middle_nonbrain_runs] or [0]) * step_mm
        )
        max_middle_nonbrain_run_mm = float(
            annulus_cfg.profile_multi_middle_nonbrain_max_run_mm
            if int(max(1, sibling_blob_proposal_count)) > 1
            else annulus_cfg.profile_simple_middle_nonbrain_max_run_mm
        )
        endpoint_margin_bins = int(max(1, round(float(annulus_cfg.profile_endpoint_margin_mm) / step_mm)))
        substantial_middle_run = False
        for run_start, run_end in middle_nonbrain_runs:
            touches_start = int(run_start) <= int(endpoint_margin_bins)
            touches_end = int(run_end) >= int(profile_len - 1 - endpoint_margin_bins)
            if touches_start or touches_end:
                continue
            run_mm = float(max(1, int(run_end - run_start + 1)) * step_mm)
            if run_mm > float(max_middle_nonbrain_run_mm):
                substantial_middle_run = True
                break
        profile_flag = bool(
            remaining_brain_fraction < float(annulus_cfg.profile_remaining_brain_fraction_req)
            or substantial_middle_run
        )
        return {
            "annulus_profile_sample_count": int(len(valid_percentiles)),
            "annulus_profile_brain_fraction": float(np.mean(brainlike)) if brainlike else None,
            "annulus_profile_remaining_brain_fraction": float(remaining_brain_fraction),
            "annulus_profile_endpoint_nonbrain_run_mm": float(endpoint_nonbrain_run_mm),
            "annulus_profile_middle_nonbrain_run_mm": float(middle_nonbrain_run_mm),
            "annulus_profile_allowed_endpoint": str(allowed_endpoint),
            "annulus_profile_bone_suspicious": bool(profile_flag),
        }

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
        axial_start_mm=None,
        axial_end_mm=None,
    ):
        start = np.asarray(proposal.get("start_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
        end = np.asarray(proposal.get("end_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
        axis = end - start
        span_mm = float(np.linalg.norm(axis))
        return cls._line_annulus_mean_ct_hu(
            volume_node=volume_node,
            start_ras=start,
            axis_ras=axis,
            span_mm=span_mm,
            annulus_inner_mm=float(annulus_inner_mm),
            annulus_outer_mm=float(annulus_outer_mm),
            axial_step_mm=float(axial_step_mm),
            radial_steps=int(radial_steps),
            angular_samples=int(angular_samples),
            axial_start_mm=axial_start_mm,
            axial_end_mm=axial_end_mm,
        )

    @classmethod
    def _proposal_shallow_endpoint_annulus_mean_ct_hu(
        cls,
        volume_node,
        proposal,
        head_distance_map_kji=None,
        axial_window_mm=4.0,
        annulus_inner_mm=2.5,
        annulus_outer_mm=3.5,
    ):
        start = np.asarray(proposal.get("start_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
        end = np.asarray(proposal.get("end_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
        span_mm = float(np.linalg.norm(end - start))
        if span_mm <= 1e-6:
            return {"mean_hu": None, "sample_count": 0, "endpoint": None}
        shallow_endpoint = str(proposal.get("shallow_endpoint_name") or "").strip().lower()
        if shallow_endpoint not in {"start", "end"}:
            start_depth = cls._depth_at_ras_with_volume(volume_node, head_distance_map_kji, start)
            end_depth = cls._depth_at_ras_with_volume(volume_node, head_distance_map_kji, end)
            if start_depth is None and end_depth is None:
                return {"mean_hu": None, "sample_count": 0, "endpoint": None}
            shallow_endpoint = "start"
            if start_depth is None:
                shallow_endpoint = "end"
            elif end_depth is None:
                shallow_endpoint = "start"
            elif float(end_depth) < float(start_depth):
                shallow_endpoint = "end"
        if shallow_endpoint not in {"start", "end"}:
            return {"mean_hu": None, "sample_count": 0, "endpoint": None}
        window_mm = float(max(1.0, axial_window_mm))
        if shallow_endpoint == "start":
            stats = cls._proposal_annulus_mean_ct_hu(
                volume_node=volume_node,
                proposal=proposal,
                annulus_inner_mm=float(annulus_inner_mm),
                annulus_outer_mm=float(annulus_outer_mm),
                axial_start_mm=0.0,
                axial_end_mm=min(span_mm, window_mm),
            )
        else:
            stats = cls._proposal_annulus_mean_ct_hu(
                volume_node=volume_node,
                proposal=proposal,
                annulus_inner_mm=float(annulus_inner_mm),
                annulus_outer_mm=float(annulus_outer_mm),
                axial_start_mm=max(0.0, span_mm - window_mm),
                axial_end_mm=span_mm,
            )
        stats = dict(stats or {})
        stats["endpoint"] = shallow_endpoint
        return stats

    @classmethod
    def _annotate_proposal_endpoint_roles(
        cls,
        volume_node,
        proposal,
        head_distance_map_kji=None,
    ):
        proposal_dict = dict(proposal or {})
        start = np.asarray(proposal_dict.get("start_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
        end = np.asarray(proposal_dict.get("end_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
        start_depth = cls._depth_at_ras_with_volume(volume_node, head_distance_map_kji, start)
        end_depth = cls._depth_at_ras_with_volume(volume_node, head_distance_map_kji, end)
        shallow_endpoint = ""
        deep_endpoint = ""
        if start_depth is None and end_depth is None:
            shallow_endpoint = ""
            deep_endpoint = ""
        elif start_depth is None:
            shallow_endpoint = "end"
            deep_endpoint = "start"
        elif end_depth is None:
            shallow_endpoint = "start"
            deep_endpoint = "end"
        elif float(start_depth) <= float(end_depth):
            shallow_endpoint = "start"
            deep_endpoint = "end"
        else:
            shallow_endpoint = "end"
            deep_endpoint = "start"
        proposal_dict["start_depth_mm"] = None if start_depth is None else float(start_depth)
        proposal_dict["end_depth_mm"] = None if end_depth is None else float(end_depth)
        proposal_dict["shallow_endpoint_name"] = str(shallow_endpoint)
        proposal_dict["deep_endpoint_name"] = str(deep_endpoint)
        if shallow_endpoint == "start":
            proposal_dict["shallow_depth_mm"] = proposal_dict["start_depth_mm"]
            proposal_dict["deep_depth_mm"] = proposal_dict["end_depth_mm"]
        elif shallow_endpoint == "end":
            proposal_dict["shallow_depth_mm"] = proposal_dict["end_depth_mm"]
            proposal_dict["deep_depth_mm"] = proposal_dict["start_depth_mm"]
        else:
            proposal_dict["shallow_depth_mm"] = None
            proposal_dict["deep_depth_mm"] = None
        return proposal_dict

    @classmethod
    def _proposal_skull_transition_from_annulus_gradient(
        cls,
        volume_node,
        proposal,
        axial_step_mm=1.0,
        annulus_inner_mm=3.0,
        annulus_outer_mm=4.0,
        radial_steps=2,
        angular_samples=12,
        smoothing_radius=2,
        edge_search_window_mm=18.0,
        edge_search_fraction=0.35,
    ):
        start = np.asarray(proposal.get("start_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
        end = np.asarray(proposal.get("end_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
        span_mm = float(np.linalg.norm(end - start))
        if span_mm <= 2.0:
            return {}
        step_mm = float(max(0.5, axial_step_mm))
        sample_positions = np.arange(0.0, span_mm + 0.5 * step_mm, step_mm, dtype=float)
        search_limit_mm = float(
            max(
                step_mm * 2.0,
                min(float(edge_search_window_mm), float(max(step_mm * 2.0, edge_search_fraction * span_mm))),
            )
        )

        def evaluate_direction(origin_ras, target_ras, origin_name, target_name):
            profile_axis = np.asarray(target_ras, dtype=float).reshape(3) - np.asarray(origin_ras, dtype=float).reshape(3)
            axis_norm = float(np.linalg.norm(profile_axis))
            if axis_norm <= 1e-6:
                return None
            profile_axis = profile_axis / axis_norm
            means = []
            counts = []
            for pos_mm in sample_positions.tolist():
                center_ras = np.asarray(origin_ras, dtype=float).reshape(3) + profile_axis * float(pos_mm)
                stats = cls._cross_section_annulus_mean_ct_hu(
                    volume_node=volume_node,
                    center_ras=center_ras,
                    axis_ras=profile_axis,
                    annulus_inner_mm=float(annulus_inner_mm),
                    annulus_outer_mm=float(annulus_outer_mm),
                    radial_steps=int(radial_steps),
                    angular_samples=int(angular_samples),
                )
                median_hu = stats.get("median_hu")
                means.append(np.nan if median_hu is None else float(median_hu))
                counts.append(int(stats.get("sample_count", 0) or 0))
            means_arr = np.asarray(means, dtype=float)
            finite_mask = np.isfinite(means_arr)
            if int(np.count_nonzero(finite_mask)) < 4:
                return None

            smooth_arr = means_arr.copy()
            radius = int(max(0, smoothing_radius))
            if radius > 0:
                for idx in range(smooth_arr.shape[0]):
                    lo = max(0, idx - radius)
                    hi = min(smooth_arr.shape[0], idx + radius + 1)
                    window = means_arr[lo:hi]
                    window = window[np.isfinite(window)]
                    if window.size > 0:
                        smooth_arr[idx] = float(np.mean(window))

            grads = np.diff(smooth_arr) / float(step_mm)
            if grads.size == 0:
                return None
            finite_grad_mask = np.isfinite(grads)
            pos_grad_mask = np.logical_and(finite_grad_mask, grads > 0.0)
            if not np.any(pos_grad_mask):
                return None
            near_edge_mask = np.asarray(sample_positions[1:] <= float(search_limit_mm), dtype=bool).reshape(-1)
            candidate_mask = np.logical_and(pos_grad_mask, near_edge_mask)
            if not np.any(candidate_mask):
                return None
            jump_idx = int(np.nanargmax(np.where(candidate_mask, grads, np.nan)))
            jump_pos_mm = float(sample_positions[min(jump_idx + 1, sample_positions.shape[0] - 1)])
            jump_point_ras = np.asarray(origin_ras, dtype=float).reshape(3) + profile_axis * jump_pos_mm
            return {
                "point_ras": [float(v) for v in np.asarray(jump_point_ras, dtype=float).reshape(3)],
                "axial_pos_mm": float(jump_pos_mm),
                "jump_hu_per_mm": float(grads[jump_idx]),
                "max_positive_gradient_hu_per_mm": float(grads[jump_idx]),
                "sample_step_mm": float(step_mm),
                "edge_search_limit_mm": float(search_limit_mm),
                "profile_origin_endpoint": str(origin_name),
                "profile_target_endpoint": str(target_name),
                "profile_mean_hu": [None if not np.isfinite(v) else float(v) for v in means_arr.tolist()],
                "profile_smooth_hu": [None if not np.isfinite(v) else float(v) for v in smooth_arr.tolist()],
                "profile_positions_mm": [float(v) for v in sample_positions.tolist()],
                "profile_sample_counts": [int(v) for v in counts],
            }

        start_to_end = evaluate_direction(start, end, "start", "end")
        end_to_start = evaluate_direction(end, start, "end", "start")
        candidates = [c for c in (start_to_end, end_to_start) if c is not None]
        if not candidates:
            return {}
        best = max(candidates, key=lambda item: float(item.get("jump_hu_per_mm", float("-inf"))))
        return dict(best)

    @classmethod
    def _annotate_proposals_with_annulus_stats(
        self,
        volume_node,
        proposals,
        mean_key="annulus_mean_ct_hu",
        median_key="annulus_median_ct_hu",
        sample_key="annulus_sample_count",
        percentile_key="annulus_scan_percentile",
        flag_key="annulus_bone_suspicious",
        annulus_flag_percentile=80.0,
        annulus_flag_min_samples=120,
        annulus_reference_upper_hu=2500.0,
        annulus_config=None,
    ):
        annulus_cfg = annulus_config if annulus_config is not None else _DEFAULT_ANNULUS_CONFIG
        annulus_reference_values_hu = self._scan_reference_hu_values(
            volume_node=volume_node,
            lower_hu=-500.0,
            upper_hu=float(annulus_cfg.annulus_reference_upper_hu) if annulus_cfg.annulus_reference_upper_hu is not None else None,
        )
        blob_proposal_counts = {}
        for proposal in list(proposals or []):
            proposal_dict = dict(proposal or {})
            for blob_id in [int(v) for v in list(proposal_dict.get("parent_blob_id_list") or []) if int(v) > 0]:
                blob_proposal_counts[int(blob_id)] = int(blob_proposal_counts.get(int(blob_id), 0)) + 1
        out = []
        for proposal in list(proposals or []):
            proposal_dict = dict(proposal or {})
            sibling_blob_proposal_count = max(
                [int(blob_proposal_counts.get(int(blob_id), 0)) for blob_id in list(proposal_dict.get("parent_blob_id_list") or []) if int(blob_id) > 0] or [1]
            )
            proposal_dict = self._annotate_single_proposal_with_annulus_stats(
                volume_node=volume_node,
                proposal=proposal_dict,
                reference_values_hu=annulus_reference_values_hu,
                mean_key=mean_key,
                median_key=median_key,
                sample_key=sample_key,
                percentile_key=percentile_key,
                flag_key=flag_key,
                annulus_flag_percentile=float(annulus_flag_percentile),
                annulus_flag_min_samples=int(annulus_cfg.annulus_flag_min_samples),
                sibling_blob_proposal_count=int(sibling_blob_proposal_count),
                annulus_config=annulus_cfg,
            )
            out.append(proposal_dict)
        return out

    def _reject_bone_adjacency_proposals(self, 
        proposals,
        flag_key="annulus_bone_suspicious",
    ):
        kept = []
        for proposal in list(proposals or []):
            proposal_dict = dict(proposal or {})
            if bool(proposal_dict.get(str(flag_key))):
                continue
            kept.append(proposal_dict)
        return kept
