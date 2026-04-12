"""Deep Core Debug workflow — bridges the shank_engine pipeline to the widget.

The detection pipeline is now ``deep_core_v1`` registered in the
``shank_engine`` ``PipelineRegistry``.  This module:

- Builds a ``DetectionContext`` from a Slicer volume node
- Runs the pipeline through the registry
- Converts ``DetectionResult`` to the legacy result types the widget expects
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .deep_core_config import DeepCoreConfig, deep_core_default_config
from .deep_core_proposal_annulus import DeepCoreProposalAnnulusMixin
from .deep_core_visualization import DeepCoreVisualizationLogicMixin
from .deep_core_widget import DeepCoreDebugWidgetMixin


def _build_detection_context(
    logic,
    volume_node,
    config: DeepCoreConfig | None = None,
) -> dict[str, Any]:
    """Build a ``DetectionContext`` dict from a Slicer volume node."""
    from __main__ import slicer

    cfg = config or deep_core_default_config()
    return {
        "run_id": f"deep_core_{volume_node.GetName()}",
        "arr_kji": np.asarray(slicer.util.arrayFromVolume(volume_node), dtype=float),
        "spacing_xyz": tuple(float(v) for v in volume_node.GetSpacing()),
        "ijk_kji_to_ras_fn": lambda idx: logic._ijk_kji_to_ras_points(volume_node, idx),
        "ras_to_ijk_fn": lambda ras: logic._ras_to_ijk_float(volume_node, ras),
        "center_ras": logic._volume_center_ras(volume_node),
        "config": cfg.to_flat_dict(),
        "extras": {"volume_node": volume_node},
    }


def _detection_result_to_legacy_support(volume_node, det_result):
    """Convert a debug ``DetectionResult`` to the legacy
    ``DeepCoreSupportResult`` shape the widget reads."""
    from .deep_core_pipeline import DeepCoreMaskResult, DeepCoreSupportResult

    meta = det_result.get("meta", {})
    mask_output = meta.get("mask_output", {})
    support_output = meta.get("support_output", {})

    mask_payload = dict(mask_output)
    mask_payload["volume_node_id"] = str(volume_node.GetID() if hasattr(volume_node, "GetID") else "")

    mask_result = DeepCoreMaskResult(
        volume_node=volume_node,
        volume_node_id=mask_payload.get("volume_node_id", ""),
        payload=mask_payload,
    )
    return DeepCoreSupportResult(
        mask_result=mask_result,
        payload=support_output,
    )


def _detection_result_to_legacy_proposal(volume_node, det_result, support_result):
    """Convert a full ``DetectionResult`` to the legacy
    ``DeepCoreProposalResult`` shape the widget reads."""
    from .deep_core_pipeline import DeepCoreProposalResult

    # Extract proposals from trajectories — each has _proposal with original dict
    proposals = []
    for t in det_result.get("trajectories", []):
        p = t.get("_proposal")
        if p is not None:
            proposals.append(p)
        else:
            proposals.append(t)

    payload = {
        "proposals": proposals,
        "candidate_count": int(det_result.get("diagnostics", {}).get("counts", {}).get("candidate_count", 0)),
        "token_count": int(det_result.get("diagnostics", {}).get("counts", {}).get("token_count", 0)),
    }
    return DeepCoreProposalResult(
        support_result=support_result,
        payload=payload,
    )


class DeepCoreDebugLogicMixin(
    DeepCoreVisualizationLogicMixin,
    DeepCoreProposalAnnulusMixin,
):
    """Deep-core logic bridge — creates pipeline via registry."""

    def get_deep_core_pipeline(self):
        """Return the ``DeepCoreV1Pipeline`` instance (cached)."""
        pipeline = getattr(self, "_deep_core_pipeline", None)
        if pipeline is None:
            pipeline = self.pipeline_registry.create_pipeline("deep_core_v1")
            self._deep_core_pipeline = pipeline
        return pipeline

    def build_deep_core_context(self, volume_node, config=None):
        """Build a DetectionContext for deep core from a Slicer volume."""
        return _build_detection_context(self, volume_node, config)

    def run_deep_core_debug(self, volume_node, config=None, show_support_diagnostics=True):
        """Run mask + support stages via the pipeline and return legacy result."""
        ctx = self.build_deep_core_context(volume_node, config)
        pipeline = self.get_deep_core_pipeline()
        det_result = pipeline.run_debug(ctx)

        # Create debug viz volumes if requested
        if show_support_diagnostics:
            mask_output = det_result.get("meta", {}).get("mask_output", {})
            if mask_output:
                try:
                    from .deep_core_volume import SlicerVolumeAccessor
                    accessor = SlicerVolumeAccessor()
                    accessor.update_scalar_volume(
                        reference_volume_node=volume_node,
                        node_name=f"{volume_node.GetName()}_HullSmooth",
                        array_kji=mask_output.get("smoothed_hull_kji"),
                    )
                    accessor.update_scalar_volume(
                        reference_volume_node=volume_node,
                        node_name=f"{volume_node.GetName()}_HeadDistanceMm",
                        array_kji=mask_output.get("head_distance_map_kji"),
                    )
                except Exception:
                    pass

        return _detection_result_to_legacy_support(volume_node, det_result)

    def run_deep_core_proposals(self, volume_node, config=None, debug_result=None):
        """Run full pipeline via the pipeline and return legacy result."""
        ctx = self.build_deep_core_context(volume_node, config)
        pipeline = self.get_deep_core_pipeline()
        det_result = pipeline.run(ctx)

        # Build legacy support result for the wrapper
        support_result = debug_result
        if support_result is None:
            support_result = _detection_result_to_legacy_support(volume_node, det_result)

        return _detection_result_to_legacy_proposal(volume_node, det_result, support_result)
