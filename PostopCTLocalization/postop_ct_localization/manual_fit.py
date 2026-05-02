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

_ORIENTATION_AUTO = "auto"
_ORIENTATION_ENTRY_FIRST = "entry_first"
_ORIENTATION_TARGET_FIRST = "target_first"

_SEED_SOURCE_OPTIONS = (
    ("Auto Fit", "auto_fit"),
    ("Guided Fit", "guided_fit"),
    ("Imported ROSA", "imported_rosa"),
    ("Imported External", "imported_external"),
)


class ManualFitWidgetMixin:
    def _build_manual_fit_tab(self):
        """Build a lightweight manual-mode tab for scene-authored trajectory lines."""
        tab = qt.QWidget()
        self.modeTabs.addTab(tab, "Manual Fit")
        form = qt.QFormLayout(tab)
        form.setFieldGrowthPolicy(qt.QFormLayout.AllNonFixedFieldsGrow)
        help_text = qt.QLabel(
            "Manual Fit: optionally Seed from a source (clone Auto/"
            "Guided/Imported into the Manual set as a starting "
            "point), then in Slicer's Markups module draw a Line "
            "markup along each missing shank — one control point at "
            "entry, one at the deep tip. Click Sync From Scene to "
            "publish; orientation rule below decides which end is "
            "entry. Hover any button for details."
        )
        help_text.wordWrap = True
        help_text.setMinimumWidth(0)
        help_text.setSizePolicy(
            qt.QSizePolicy.Preferred, qt.QSizePolicy.MinimumExpanding,
        )
        form.addRow(help_text)

        self.manualFitSeedSourceCombo = qt.QComboBox()
        for label, key in _SEED_SOURCE_OPTIONS:
            self.manualFitSeedSourceCombo.addItem(label, key)
        self.manualFitSeedSourceCombo.toolTip = (
            "Which trajectory set to clone into Manual when Seed is pressed. "
            "Existing manual lines with the same name are not overwritten — "
            "re-pressing Seed only adds lines that aren't already in Manual."
        )
        self.manualFitSeedSourceCombo.setMinimumContentsLength(8)
        self.manualFitSeedSourceCombo.setSizeAdjustPolicy(
            qt.QComboBox.AdjustToMinimumContentsLengthWithIcon,
        )
        self.manualFitSeedSourceCombo.setSizePolicy(
            qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed,
        )
        form.addRow("Seed source:", self.manualFitSeedSourceCombo)

        seed_button = qt.QPushButton("Seed From Source")
        seed_button.toolTip = (
            "Clone the chosen source's trajectories into the Manual set as new "
            "line markups. Original source lines are not modified."
        )
        seed_button.clicked.connect(self.onSeedManualFromSourceClicked)
        form.addRow(seed_button)

        delete_selected_button = qt.QPushButton("Delete Selected")
        delete_selected_button.toolTip = (
            "Remove every manual-group row currently selected in the Trajectory "
            "table above. Non-manual selections are ignored."
        )
        delete_selected_button.clicked.connect(self.onDeleteSelectedManualClicked)
        form.addRow(delete_selected_button)

        clear_all_button = qt.QPushButton("Clear All Manual")
        clear_all_button.toolTip = (
            "Delete every line in the Manual trajectory set. Confirms before "
            "removing — non-Manual sources (Auto Fit, Imported, etc.) are not affected."
        )
        clear_all_button.clicked.connect(self.onClearAllManualClicked)
        form.addRow(clear_all_button)

        self.manualFitOrientationCombo = qt.QComboBox()
        self.manualFitOrientationCombo.addItem("Auto (skull-distance)", _ORIENTATION_AUTO)
        self.manualFitOrientationCombo.addItem("Entry → Target", _ORIENTATION_ENTRY_FIRST)
        self.manualFitOrientationCombo.addItem("Target → Entry", _ORIENTATION_TARGET_FIRST)
        self.manualFitOrientationCombo.toolTip = (
            "How to decide which endpoint is the bolt/entry on Sync. "
            "Auto samples the postop CT head-surface distance at both ends; "
            "the shallower end becomes the entry. Falls back to "
            "Entry → Target if no postop CT is registered."
        )
        self.manualFitOrientationCombo.setMinimumContentsLength(0)
        self.manualFitOrientationCombo.setSizePolicy(
            qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed,
        )
        form.addRow("Orientation:", self.manualFitOrientationCombo)

        # Restrict the model library used for the per-line PaCER picker.
        # On Sync, every manual line gets `Rosa.BestModelId` stamped from
        # whatever model wins under this strategy — so picking
        # "Dixi AM (3.5 mm)" prevents the picker from suggesting a PMT
        # or Medtronic model on a Dixi-only case.
        from rosa_core.electrode_classifier import PITCH_STRATEGY_OPTIONS
        self.manualFitPitchStrategyCombo = qt.QComboBox()
        for label, key in PITCH_STRATEGY_OPTIONS:
            self.manualFitPitchStrategyCombo.addItem(label, key)
        self.manualFitPitchStrategyCombo.setCurrentIndex(0)  # default: Dixi AM
        self.manualFitPitchStrategyCombo.setToolTip(
            "Restrict the model library used by the per-line "
            "electrode-model picker on Sync. Vendor + pitch-set filter."
        )
        self.manualFitPitchStrategyCombo.setMinimumContentsLength(8)
        self.manualFitPitchStrategyCombo.setSizeAdjustPolicy(
            qt.QComboBox.AdjustToMinimumContentsLengthWithIcon,
        )
        self.manualFitPitchStrategyCombo.setSizePolicy(
            qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed,
        )
        form.addRow("Pitch strategy:", self.manualFitPitchStrategyCombo)

        sync_button = qt.QPushButton("Sync From Scene")
        sync_button.clicked.connect(self.onSyncManualTrajectoriesClicked)
        form.addRow(sync_button)

        swap_button = qt.QPushButton("Swap Entry/Target")
        swap_button.clicked.connect(self.onSwapManualEndpointsClicked)
        form.addRow(swap_button)

        activate_button = qt.QPushButton("Switch Source → Manual")
        activate_button.clicked.connect(self.onSwitchToManualSourceClicked)
        form.addRow(activate_button)

    def onSyncManualTrajectoriesClicked(self):
        """Publish scene-authored line markups into the Manual trajectory set."""
        orientation = self.manualFitOrientationCombo.currentData
        if not isinstance(orientation, str) or not orientation:
            orientation = _ORIENTATION_AUTO
        strategy = self.manualFitPitchStrategyCombo.currentData
        if not isinstance(strategy, str) or not strategy:
            strategy = "auto"
        count, reoriented = self.logic.sync_manual_trajectories_to_workflow(
            workflow_node=self.workflowNode,
            orientation=orientation,
            pitch_strategy=strategy,
        )
        self.log(
            f"[manual] synced {count} manual trajectory nodes "
            f"({reoriented} reoriented, mode={orientation}, "
            f"strategy={strategy})"
        )
        self._set_workflow_active_source("manual")
        self._set_guided_source_combo("manual")
        self.onRefreshClicked()

    def onSwapManualEndpointsClicked(self):
        """Swap E/T on every manual line in the scene (one-shot bulk override)."""
        n = self.logic.swap_manual_trajectory_endpoints(workflow_node=self.workflowNode)
        self.log(f"[manual] swapped entry/target on {n} manual trajectory nodes")
        self.onRefreshClicked()

    def onSeedManualFromSourceClicked(self):
        """Clone the chosen source's trajectories into the Manual set."""
        source = self.manualFitSeedSourceCombo.currentData
        if not isinstance(source, str) or not source:
            return
        added, skipped = self.logic.seed_manual_from_source(
            source_key=source, workflow_node=self.workflowNode,
        )
        self.log(
            f"[manual] seeded from {source}: {added} added, "
            f"{skipped} skipped (name already in Manual)"
        )
        self.onRefreshClicked()

    def onDeleteSelectedManualClicked(self):
        """Remove manual-group trajectories currently selected in the table."""
        rows = self._selected_table_rows() if hasattr(self, "_selected_table_rows") else []
        names = []
        for row in rows:
            if 0 <= row < len(getattr(self, "loadedTrajectories", [])):
                traj = self.loadedTrajectories[row]
                if str(traj.get("group", "")).strip().lower() != "manual":
                    continue
                name = str(traj.get("name", "")).strip()
                if name:
                    names.append(name)
        if not names:
            qt.QMessageBox.information(
                slicer.util.mainWindow(),
                "Manual Fit",
                "Select one or more Manual-group rows in the Trajectory table first.",
            )
            return
        removed = self.logic.remove_trajectories_by_name(names, source_key="manual")
        self.log(f"[manual] deleted {removed} selected manual trajectory nodes")
        self.onRefreshClicked()

    def onClearAllManualClicked(self):
        """Delete every line in the Manual trajectory set after confirmation."""
        manual_rows = self.logic.trajectory_scene.collect_working_trajectory_rows(
            groups=["manual"]
        )
        if not manual_rows:
            self.log("[manual] no manual trajectories to clear")
            return
        ret = qt.QMessageBox.question(
            slicer.util.mainWindow(),
            "Clear Manual Trajectories",
            f"Delete all {len(manual_rows)} manual trajectories?\n\n"
            "This removes only the Manual set; Auto Fit / Imported / Guided "
            "sources are not affected.",
            qt.QMessageBox.Yes | qt.QMessageBox.No,
            qt.QMessageBox.No,
        )
        if ret != qt.QMessageBox.Yes:
            return
        names = [str(r.get("name", "")).strip() for r in manual_rows]
        names = [n for n in names if n]
        removed = self.logic.remove_trajectories_by_name(names, source_key="manual")
        self.log(f"[manual] cleared {removed} manual trajectory nodes")
        self.onRefreshClicked()

    def onSwitchToManualSourceClicked(self):
        """Switch current source filter to Manual trajectories without mutating nodes."""
        self._set_workflow_active_source("manual")
        self._set_guided_source_combo("manual")
        self.onRefreshClicked()
