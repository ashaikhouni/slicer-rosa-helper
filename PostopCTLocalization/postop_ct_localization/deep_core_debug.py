"""Deep Core Debug workflow — bridges the shank_engine pipeline to the widget.

The detection pipeline is now ``deep_core_v1`` registered in the
``shank_engine`` ``PipelineRegistry``.  This module:

- Builds a ``DetectionContext`` from a Slicer volume node
- Runs the pipeline through the registry
- Converts ``DetectionResult`` to legacy result types the widget expects
- Provides legacy result dataclasses (DeepCoreMaskResult, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .deep_core_config import DeepCoreConfig, deep_core_default_config
from .deep_core_proposal_annulus import DeepCoreProposalAnnulusMixin
from .deep_core_visualization import DeepCoreVisualizationLogicMixin
from .deep_core_widget import DeepCoreDebugWidgetMixin


# ---------------------------------------------------------------------------
# Legacy result types — consumed by deep_core_widget.py
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeepCoreMaskResult:
    """Mask stage output for widget consumption."""
    volume_node: Any
    volume_node_id: str
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def stats(self) -> dict[str, Any]:
        return dict(self.payload.get("stats") or {})


@dataclass(frozen=True)
class DeepCoreSupportResult:
    """Support stage output for widget consumption."""
    mask_result: DeepCoreMaskResult
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def volume_node(self) -> Any:
        return self.mask_result.volume_node

    @property
    def volume_node_id(self) -> str:
        return self.mask_result.volume_node_id

    @property
    def stats(self) -> dict[str, Any]:
        return dict(self.payload.get("stats") or {})


@dataclass(frozen=True)
class DeepCoreProposalResult:
    """Proposal stage output for widget consumption."""
    support_result: DeepCoreSupportResult
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def volume_node(self) -> Any:
        return self.support_result.volume_node

    @property
    def volume_node_id(self) -> str:
        return self.support_result.volume_node_id


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Result conversion
# ---------------------------------------------------------------------------

def _detection_result_to_legacy_support(volume_node, det_result, pipeline=None):
    """Convert a debug ``DetectionResult`` to legacy widget types."""
    mask_output = getattr(pipeline, "_last_mask_output", None) or {}
    support_output = getattr(pipeline, "_last_support_output", None) or {}

    mask_payload = dict(mask_output)
    mask_payload["volume_node_id"] = str(
        volume_node.GetID() if hasattr(volume_node, "GetID") else ""
    )

    mask_result = DeepCoreMaskResult(
        volume_node=volume_node,
        volume_node_id=mask_payload.get("volume_node_id", ""),
        payload=mask_payload,
    )

    # Merge mask stats into support stats so widget sees hull_voxels etc.
    support_payload = dict(support_output)
    merged_stats = dict(mask_output.get("stats") or {})
    merged_stats.update(dict(support_output.get("stats") or {}))
    support_payload["stats"] = merged_stats

    return DeepCoreSupportResult(
        mask_result=mask_result,
        payload=support_payload,
    )


def _detection_result_to_legacy_proposal(volume_node, det_result, support_result, pipeline=None):
    """Convert a full ``DetectionResult`` to legacy widget types."""
    proposal_payload = getattr(pipeline, "_last_proposal_payload", None) or {}
    proposals = list(proposal_payload.get("proposals") or [])

    payload = {
        "proposals": proposals,
        "candidate_count": int(proposal_payload.get("candidate_count", 0)),
        "token_count": int(proposal_payload.get("token_count", 0)),
    }
    return DeepCoreProposalResult(
        support_result=support_result,
        payload=payload,
    )


# ---------------------------------------------------------------------------
# Logic mixin
# ---------------------------------------------------------------------------

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
        self._last_deep_core_ctx = ctx
        self._last_deep_core_ctx_volume_id = str(
            volume_node.GetID() if hasattr(volume_node, "GetID") else ""
        )
        pipeline = self.get_deep_core_pipeline()
        det_result = pipeline.run_debug(ctx)

        if show_support_diagnostics:
            mask_output = getattr(pipeline, "_last_mask_output", None) or {}
            if mask_output:
                try:
                    from __main__ import slicer, vtk

                    def _update_vol(ref_node, name, arr):
                        scene = slicer.mrmlScene
                        node = None
                        for n in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
                            if n.GetName() == str(name):
                                node = n
                                break
                        if node is None:
                            node = scene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", str(name))
                        m = vtk.vtkMatrix4x4()
                        ref_node.GetIJKToRASMatrix(m)
                        node.SetIJKToRASMatrix(m)
                        slicer.util.updateVolumeFromArray(node, np.asarray(arr))
                        node.Modified()

                    _update_vol(volume_node, f"{volume_node.GetName()}_HullSmooth",
                                mask_output.get("smoothed_hull_kji"))
                    _update_vol(volume_node, f"{volume_node.GetName()}_HeadDistanceMm",
                                mask_output.get("head_distance_map_kji"))
                except Exception:
                    pass

        return _detection_result_to_legacy_support(volume_node, det_result, pipeline=pipeline)

    def run_deep_core_proposals(self, volume_node, config=None, debug_result=None):
        """Run full pipeline via the pipeline and return legacy result."""
        cached_vid = getattr(self, "_last_deep_core_ctx_volume_id", None)
        current_vid = str(volume_node.GetID() if hasattr(volume_node, "GetID") else "")
        if cached_vid == current_vid and getattr(self, "_last_deep_core_ctx", None) is not None:
            ctx = self._last_deep_core_ctx
            cfg = config or deep_core_default_config()
            ctx["config"] = cfg.to_flat_dict()
        else:
            ctx = self.build_deep_core_context(volume_node, config)
        pipeline = self.get_deep_core_pipeline()
        det_result = pipeline.run(ctx)

        if det_result.get("status") == "error":
            err = det_result.get("error", {})
            notes = det_result.get("diagnostics", {}).get("notes", [])
            msg = err.get("message", "unknown error") if err else "unknown error"
            raise RuntimeError(f"[deep-core pipeline] {msg} (notes: {notes})")

        support_result = debug_result
        if support_result is None:
            support_result = _detection_result_to_legacy_support(
                volume_node, det_result, pipeline=pipeline
            )

        return _detection_result_to_legacy_proposal(
            volume_node, det_result, support_result, pipeline=pipeline
        )
