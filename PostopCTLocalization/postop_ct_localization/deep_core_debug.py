"""Compatibility aggregator for the Deep Core Debug workflow.

The implementation is intentionally split by responsibility:
- `deep_core_widget.py`: Deep Core Debug tab UI and user actions
- `deep_core_support.py`: mask building and blob/token support extraction
- `deep_core_proposals.py`: local graph proposal generation, completion, and NMS
- `deep_core_visualization.py`: MRML nodes, labelmaps, display helpers, and diagnostics
"""

from .deep_core_proposals import DeepCoreProposalLogicMixin
from .deep_core_support import DeepCoreSupportLogicMixin
from .deep_core_visualization import DeepCoreVisualizationLogicMixin
from .deep_core_widget import DeepCoreDebugWidgetMixin


class DeepCoreDebugLogicMixin(
    DeepCoreSupportLogicMixin,
    DeepCoreProposalLogicMixin,
    DeepCoreVisualizationLogicMixin,
):
    """Composed deep-core logic mixin."""

    pass
