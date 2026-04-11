"""Compatibility aggregator for the Deep Core Debug workflow.

The implementation is split by responsibility:
- ``deep_core_stages.py``: Composed stage classes (MaskStage, SupportStage, ProposalStage)
- ``deep_core_pipeline.py``: Pipeline orchestrator with legacy result wrappers
- ``deep_core_volume.py``: VolumeAccessor protocol and SlicerVolumeAccessor
- ``deep_core_visualization.py``: MRML node display helpers
- ``deep_core_widget.py``: Deep Core Debug tab UI and user actions

The mixin-based inheritance (DeepCoreSupportLogicMixin, etc.) is retained
on ``DeepCoreDebugLogicMixin`` only so that the visualization mixin and
any remaining legacy callers still resolve their ``self._`` methods.  New
code should use ``get_deep_core_pipeline()`` exclusively.
"""

from .deep_core_pipeline import DeepCorePipeline
from .deep_core_proposals import DeepCoreProposalLogicMixin
from .deep_core_support import DeepCoreSupportLogicMixin
from .deep_core_visualization import DeepCoreVisualizationLogicMixin
from .deep_core_volume import SlicerVolumeAccessor
from .deep_core_widget import DeepCoreDebugWidgetMixin


class DeepCoreDebugLogicMixin(
    DeepCoreSupportLogicMixin,
    DeepCoreProposalLogicMixin,
    DeepCoreVisualizationLogicMixin,
):
    """Composed deep-core logic mixin."""

    def get_deep_core_pipeline(self):
        """Return a cached Deep Core pipeline using composed stage classes."""

        pipeline = getattr(self, "_deep_core_pipeline", None)
        if pipeline is None:
            accessor = SlicerVolumeAccessor()
            pipeline = DeepCorePipeline(
                volume_accessor=accessor,
                host_logic=self,
            )
            self._deep_core_pipeline = pipeline
        return pipeline
