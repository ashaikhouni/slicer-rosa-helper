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

from rosa_core import (
    load_electrode_library,
    lps_to_ras_point,
    model_map,
    suggest_model_id_for_trajectory,
    trajectory_length_mm,
)
from rosa_core.contact_fit import fit_electrode_axis_and_tip
from rosa_scene import ElectrodeSceneService, LayoutService, TrajectoryFocusController, TrajectorySceneService
from shank_core.blob_candidates import build_blob_labelmap, extract_blob_candidates
from shank_core.masking import build_preview_masks, compute_head_distance_map_kji, largest_component_binary
from shank_engine import PipelineRegistry, register_builtin_pipelines
from rosa_workflow import WorkflowPublisher, WorkflowState
from rosa_workflow.workflow_registry import table_to_dict_rows

from .constants import GUIDED_SOURCE_OPTIONS

class ManualFitWidgetMixin:
    def _build_manual_fit_tab(self):
        """Build a lightweight manual-mode tab for scene-authored trajectory lines."""
        tab = qt.QWidget()
        self.modeTabs.addTab(tab, "Manual Fit")
        form = qt.QFormLayout(tab)
        help_text = qt.QLabel(
            "Draw line markups in the scene, then sync them into the Manual trajectory set."
        )
        help_text.wordWrap = True
        form.addRow(help_text)

        sync_button = qt.QPushButton("Sync Manual Trajectories From Scene")
        sync_button.clicked.connect(self.onSyncManualTrajectoriesClicked)
        form.addRow(sync_button)

        activate_button = qt.QPushButton("Switch Trajectory Source To Manual")
        activate_button.clicked.connect(self.onSwitchToManualSourceClicked)
        form.addRow(activate_button)

    def onSyncManualTrajectoriesClicked(self):
        """Publish scene-authored line markups into the Manual trajectory set."""
        count = self.logic.sync_manual_trajectories_to_workflow(workflow_node=self.workflowNode)
        self.log(f"[manual] synced {count} manual trajectory nodes")
        self._set_workflow_active_source("manual")
        self._set_guided_source_combo("manual")
        self.onRefreshClicked()

    def onSwitchToManualSourceClicked(self):
        """Switch current source filter to Manual trajectories without mutating nodes."""
        self._set_workflow_active_source("manual")
        self._set_guided_source_combo("manual")
        self.onRefreshClicked()
