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
        help_text = qt.QLabel(
            "<b>Manual Fit workflow</b>"
            "<ol style='margin-top:4px; margin-bottom:4px;'>"
            "<li>(Optional) Pick a source below and press <b>Seed Manual From Source</b> "
            "to copy its trajectories into the Manual set as a starting point — "
            "e.g. clone the Auto Fit output, then draw extra lines for any shanks "
            "Auto Fit missed.</li>"
            "<li>In Slicer's <b>Markups</b> module, activate the <b>Line</b> "
            "tool and draw two control points along each shank you want to add — "
            "one near the bolt/entry, one near the deep tip. Click order doesn't matter "
            "when <i>Auto</i> orientation is selected below.</li>"
            "<li>Edit any line by dragging its control points in the slice or 3D views — "
            "seeded clones are independent of the source and your edits stay on the "
            "Manual copy. Rename a line in the Markups module to give it a meaningful "
            "name (e.g. <i>RAH</i>).</li>"
            "<li>Delete unwanted lines by selecting their rows in the Trajectory "
            "table above and pressing <b>Delete Selected Manual Trajectories</b>, "
            "or wipe the whole Manual set with <b>Clear All Manual Trajectories</b>.</li>"
            "<li>Press <b>Sync Manual Trajectories From Scene</b>. All lines "
            "(seeded + drawn) are tagged as the <i>Manual</i> trajectory set, "
            "each one's two endpoints are ordered per the selected rule, and "
            "the entry control point is labelled <b>E</b> while the deep tip is "
            "labelled <b>T</b>.</li>"
            "<li>If a line ends up flipped (E and T swapped), press "
            "<b>Swap Entry/Target On All Manual Lines</b>.</li>"
            "<li>Press <b>Switch Trajectory Source To Manual</b> to make the "
            "Manual set the active source for downstream tools.</li>"
            "</ol>"
        )
        help_text.wordWrap = True
        help_text.textFormat = qt.Qt.RichText
        form.addRow(help_text)

        self.manualFitSeedSourceCombo = qt.QComboBox()
        for label, key in _SEED_SOURCE_OPTIONS:
            self.manualFitSeedSourceCombo.addItem(label, key)
        self.manualFitSeedSourceCombo.toolTip = (
            "Which trajectory set to clone into Manual when Seed is pressed. "
            "Existing manual lines with the same name are not overwritten — "
            "re-pressing Seed only adds lines that aren't already in Manual."
        )
        form.addRow("Seed source:", self.manualFitSeedSourceCombo)

        seed_button = qt.QPushButton("Seed Manual From Source")
        seed_button.toolTip = (
            "Clone the chosen source's trajectories into the Manual set as new "
            "line markups. Original source lines are not modified."
        )
        seed_button.clicked.connect(self.onSeedManualFromSourceClicked)
        form.addRow(seed_button)

        delete_selected_button = qt.QPushButton("Delete Selected Manual Trajectories")
        delete_selected_button.toolTip = (
            "Remove every manual-group row currently selected in the Trajectory "
            "table above. Non-manual selections are ignored."
        )
        delete_selected_button.clicked.connect(self.onDeleteSelectedManualClicked)
        form.addRow(delete_selected_button)

        clear_all_button = qt.QPushButton("Clear All Manual Trajectories")
        clear_all_button.toolTip = (
            "Delete every line in the Manual trajectory set. Confirms before "
            "removing — non-Manual sources (Auto Fit, Imported, etc.) are not affected."
        )
        clear_all_button.clicked.connect(self.onClearAllManualClicked)
        form.addRow(clear_all_button)

        self.manualFitOrientationCombo = qt.QComboBox()
        self.manualFitOrientationCombo.addItem(
            "Auto (closer to skull surface = entry)", _ORIENTATION_AUTO
        )
        self.manualFitOrientationCombo.addItem(
            "Click order: entry first, target second", _ORIENTATION_ENTRY_FIRST
        )
        self.manualFitOrientationCombo.addItem(
            "Click order: target first, entry second", _ORIENTATION_TARGET_FIRST
        )
        self.manualFitOrientationCombo.toolTip = (
            "How to decide which endpoint is the bolt/entry on Sync. "
            "Auto samples the postop CT head-surface distance at both ends; "
            "the shallower end becomes the entry. Falls back to 'entry first' "
            "if no postop CT is registered."
        )
        form.addRow("Orientation rule:", self.manualFitOrientationCombo)

        sync_button = qt.QPushButton("Sync Manual Trajectories From Scene")
        sync_button.clicked.connect(self.onSyncManualTrajectoriesClicked)
        form.addRow(sync_button)

        swap_button = qt.QPushButton("Swap Entry/Target On All Manual Lines")
        swap_button.clicked.connect(self.onSwapManualEndpointsClicked)
        form.addRow(swap_button)

        activate_button = qt.QPushButton("Switch Trajectory Source To Manual")
        activate_button.clicked.connect(self.onSwitchToManualSourceClicked)
        form.addRow(activate_button)

    def onSyncManualTrajectoriesClicked(self):
        """Publish scene-authored line markups into the Manual trajectory set."""
        orientation = self.manualFitOrientationCombo.currentData
        if not isinstance(orientation, str) or not orientation:
            orientation = _ORIENTATION_AUTO
        count, reoriented = self.logic.sync_manual_trajectories_to_workflow(
            workflow_node=self.workflowNode, orientation=orientation,
        )
        self.log(
            f"[manual] synced {count} manual trajectory nodes "
            f"({reoriented} reoriented, mode={orientation})"
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
