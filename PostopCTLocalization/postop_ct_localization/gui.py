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

class PostopCTLocalizationWidgetBaseMixin:
    def setup(self):
        super().setup()
        self.logic = self._create_logic()
        self.workflowState = self.logic.workflow_state
        self.workflowPublisher = self.logic.workflow_publish
        self.workflowNode = self.workflowState.resolve_or_create_workflow_node()

        self.modelsById = {}
        self.modelIds = []
        self.loadedTrajectories = []
        self.assignmentMap = {}
        self._syncingGuidedSourceCombo = False
        self._pendingGuidedFollow = False
        self._updatingGuidedTable = False
        self._renamingGuidedTrajectory = False
        self._syncingModeTabs = False
        self._lastDeepCoreProposalResult = None

        form = qt.QFormLayout()
        self.layout.addLayout(form)

        self.ctSelector = slicer.qMRMLNodeComboBox()
        self.ctSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.ctSelector.noneEnabled = True
        self.ctSelector.addEnabled = False
        self.ctSelector.removeEnabled = False
        self.ctSelector.setMRMLScene(slicer.mrmlScene)
        self.ctSelector.setToolTip("Postop CT used for Auto Fit / Guided Fit.")
        self.ctSelector.currentNodeChanged.connect(self.onCtSelectorChanged)
        form.addRow("Postop CT", self.ctSelector)

        topRow = qt.QHBoxLayout()
        self.refreshButton = qt.QPushButton("Refresh Workflow Inputs")
        self.refreshButton.clicked.connect(self.onRefreshClicked)
        topRow.addWidget(self.refreshButton)
        self.applyFocusLayoutButton = qt.QPushButton("Apply Focus Layout (2x3)")
        self.applyFocusLayoutButton.clicked.connect(self.onApplyFocusLayoutClicked)
        topRow.addWidget(self.applyFocusLayoutButton)
        topRow.addStretch(1)
        form.addRow(topRow)

        self.summaryLabel = qt.QLabel("Workflow not scanned yet.")
        self.summaryLabel.wordWrap = True
        form.addRow("Workflow summary", self.summaryLabel)

        self.modeTabs = qt.QTabWidget()
        self.modeTabs.currentChanged.connect(self.onModeTabChanged)
        self.layout.addWidget(self.modeTabs)
        self._build_guided_fit_tab()
        self._build_contact_pitch_v1_tab()
        self._build_manual_fit_tab()
        self._build_shared_trajectory_ui()

        self.statusText = qt.QPlainTextEdit()
        self.statusText.setReadOnly(True)
        self.statusText.setMaximumBlockCount(2000)
        self.layout.addWidget(self.statusText)
        self.layout.addStretch(1)

        self._load_electrode_library()
        self.onRefreshClicked()

    def log(self, message):
        self.statusText.appendPlainText(str(message))
        self.statusText.ensureCursorVisible()
        try:
            slicer.app.processEvents()
        except Exception:
            pass

    def _build_shared_trajectory_ui(self):
        """Build shared trajectory source + table UI visible for both tabs."""
        section = ctk.ctkCollapsibleButton()
        section.text = "Trajectory Set"
        section.collapsed = False
        self.layout.addWidget(section)
        form = qt.QFormLayout(section)

        self.guidedSourceCombo = qt.QComboBox()
        for key, label in GUIDED_SOURCE_OPTIONS:
            self.guidedSourceCombo.addItem(label, key)
        self.guidedSourceCombo.currentIndexChanged.connect(self.onGuidedSourceChanged)
        form.addRow("Trajectory source", self.guidedSourceCombo)

        # Confidence filter — auto-checks "Use" only for trajectories at
        # or above the chosen band. Auto-Fit emissions carry confidence
        # via Rosa.Confidence / Rosa.ConfidenceLabel attributes; manual
        # / imported trajectories have no confidence and always check.
        self.confidenceFilterCombo = qt.QComboBox()
        for key, label in (
            ("all", "Show all (no filter)"),
            ("low", "Low and above (>= 0)"),
            ("medium", "Medium and above (>= 0.50)"),
            ("high", "High only (>= 0.80)"),
        ):
            self.confidenceFilterCombo.addItem(label, key)
        self.confidenceFilterCombo.setCurrentIndex(0)
        self.confidenceFilterCombo.currentIndexChanged.connect(
            self.onConfidenceFilterChanged
        )
        form.addRow("Confidence filter", self.confidenceFilterCombo)

        self.guidedTrajectoryTable = qt.QTableWidget()
        self.guidedTrajectoryTable.setColumnCount(4)
        self.guidedTrajectoryTable.setHorizontalHeaderLabels(
            ["Mark", "Trajectory", "Length (mm)", "Confidence"]
        )
        self.guidedTrajectoryTable.horizontalHeader().setSectionResizeMode(0, qt.QHeaderView.ResizeToContents)
        self.guidedTrajectoryTable.horizontalHeader().setSectionResizeMode(1, qt.QHeaderView.Stretch)
        self.guidedTrajectoryTable.horizontalHeader().setSectionResizeMode(2, qt.QHeaderView.ResizeToContents)
        self.guidedTrajectoryTable.horizontalHeader().setSectionResizeMode(3, qt.QHeaderView.ResizeToContents)
        self.guidedTrajectoryTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        # ExtendedSelection: Ctrl-click to add to selection, Shift-click
        # to extend a range. Combined with the "Mark Selected" button
        # below, lets users select N rows and toggle them in bulk
        # without iterating one checkbox at a time.
        self.guidedTrajectoryTable.setSelectionMode(qt.QAbstractItemView.ExtendedSelection)
        self.guidedTrajectoryTable.cellClicked.connect(self.onGuidedTrajectoryTableCellClicked)
        self.guidedTrajectoryTable.currentCellChanged.connect(self.onGuidedTrajectoryTableCurrentCellChanged)
        self.guidedTrajectoryTable.itemChanged.connect(self.onGuidedTrajectoryItemChanged)
        form.addRow(self.guidedTrajectoryTable)

        # Bulk-mark actions. Trajectories start UNCHECKED so a stray
        # click on "Remove Marked Trajectories" does nothing — users
        # have to explicitly mark first. Multi-select + the "Mark
        # Selected" button is the fast path for marking multiple rows.
        actions_row = qt.QHBoxLayout()
        self.markSelectedButton = qt.QPushButton("Mark Selected")
        self.markSelectedButton.setToolTip(
            "Check the Mark column for every currently-selected row. "
            "Use Ctrl/Shift-click to select multiple rows."
        )
        self.markSelectedButton.clicked.connect(self.onMarkSelectedClicked)
        actions_row.addWidget(self.markSelectedButton)

        self.markAllButton = qt.QPushButton("Mark All Visible")
        self.markAllButton.setToolTip(
            "Check the Mark column for every visible row "
            "(rows hidden by the Confidence filter are skipped)."
        )
        self.markAllButton.clicked.connect(self.onMarkAllClicked)
        actions_row.addWidget(self.markAllButton)

        self.unmarkAllButton = qt.QPushButton("Unmark All")
        self.unmarkAllButton.clicked.connect(self.onUnmarkAllClicked)
        actions_row.addWidget(self.unmarkAllButton)

        self.removeCheckedButton = qt.QPushButton("Remove Marked Trajectories")
        self.removeCheckedButton.clicked.connect(self.onRemoveCheckedClicked)
        actions_row.addWidget(self.removeCheckedButton)

        actions_row.addStretch(1)
        form.addRow(actions_row)

    def _load_electrode_library(self):
        try:
            data = load_electrode_library()
            self.modelsById = model_map(data)
            self.modelIds = sorted(self.modelsById.keys())
        except Exception as exc:
            self.modelsById = {}
            self.modelIds = []
            self.log(f"[electrodes] failed to load model library: {exc}")
            return
        self.log(f"[electrodes] loaded {len(self.modelIds)} models")

    @staticmethod
    def _clean_table_text(value, default=""):
        """Normalize MRML table text cells that may include wrapped quotes."""
        if value is None:
            return default
        text = str(value).strip()
        if not text:
            return default
        if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
            text = text[1:-1].strip()
        return text or default

    @classmethod
    def _safe_float(cls, value, default=0.0):
        """Parse numeric table cells robustly (e.g., '\"0.0\"')."""
        if value is None:
            return float(default)
        if isinstance(value, (int, float)):
            return float(value)
        text = cls._clean_table_text(value, default="")
        if not text:
            return float(default)
        try:
            return float(text)
        except Exception:
            try:
                return float(text.strip(" \"'"))
            except Exception:
                return float(default)

    def _assignment_map_from_workflow(self):
        amap = {}
        table = self.workflowNode.GetNodeReference("ElectrodeAssignmentTable")
        for row in table_to_dict_rows(table):
            traj = self._clean_table_text(row.get("trajectory"), default="")
            model_id = self._clean_table_text(row.get("model_id"), default="")
            if not traj or not model_id:
                continue
            amap[traj] = {
                "trajectory": traj,
                "model_id": model_id,
                "tip_at": self._clean_table_text(row.get("tip_at"), default="target"),
                "tip_shift_mm": self._safe_float(row.get("tip_shift_mm"), default=0.0),
                "xyz_offset_mm": [0.0, 0.0, 0.0],
            }
        return amap

    def _selected_guided_trajectory_name(self):
        row = self.guidedTrajectoryTable.currentRow()
        if row < 0:
            selected = self.guidedTrajectoryTable.selectedItems()
            if selected:
                row = selected[0].row()
        if row < 0 and self.guidedTrajectoryTable.rowCount > 0:
            row = 0
        if row < 0:
            return ""
        item = self.guidedTrajectoryTable.item(row, 1)
        return str(item.text() if item else "").strip()

    def _selected_guided_trajectory(self):
        name = self._selected_guided_trajectory_name()
        if not name:
            return None
        for traj in self.loadedTrajectories:
            if str(traj.get("name", "")) == name:
                return traj
        return None

    def _center_on_guided_selection(self):
        selected = self._selected_guided_trajectory()
        scope_ids = [traj.get("node_id", "") for traj in self.loadedTrajectories if traj.get("node_id")]
        self.logic.focus_controller.focus_selected(
            trajectory=selected,
            scope_node_ids=scope_ids,
            jump_cardinal=True,
            align_focus_views=bool(self.logic.layout_service.has_focus_slice_views()),
            focus="entry",
        )

    def _schedule_guided_follow(self):
        if self._pendingGuidedFollow:
            return
        self._pendingGuidedFollow = True
        qt.QTimer.singleShot(0, self._run_scheduled_guided_follow)

    def _run_scheduled_guided_follow(self):
        self._pendingGuidedFollow = False
        self._center_on_guided_selection()

    def _selected_guided_source_key(self):
        data = self.guidedSourceCombo.currentData
        value = data() if callable(data) else data
        return str(value or "working")

    def _sync_ct_selector_from_workflow(self):
        postop_nodes = self.workflowState.role_nodes("PostopCT", workflow_node=self.workflowNode)
        if postop_nodes:
            self.ctSelector.setCurrentNode(postop_nodes[0])

    def _resolve_base_postop_nodes(self):
        base_nodes = self.workflowState.role_nodes("BaseVolume", workflow_node=self.workflowNode)
        postop_nodes = self.workflowState.role_nodes("PostopCT", workflow_node=self.workflowNode)
        base_node = base_nodes[0] if base_nodes else None
        postop_node = postop_nodes[0] if postop_nodes else None
        if postop_node is None:
            postop_node = self.ctSelector.currentNode()
        return base_node, postop_node

    def _apply_primary_slice_layers(self):
        base_node, postop_node = self._resolve_base_postop_nodes()
        if base_node is not None and postop_node is not None:
            bg_id = base_node.GetID()
            fg_id = postop_node.GetID()
            fg_opacity = 0.5
        elif base_node is not None:
            bg_id = base_node.GetID()
            fg_id = ""
            fg_opacity = 0.0
        elif postop_node is not None:
            bg_id = postop_node.GetID()
            fg_id = ""
            fg_opacity = 0.0
        else:
            return
        for composite in slicer.util.getNodesByClass("vtkMRMLSliceCompositeNode"):
            composite.SetBackgroundVolumeID(bg_id)
            composite.SetForegroundVolumeID(fg_id)
            composite.SetForegroundOpacity(fg_opacity)

    def _workflow_active_source(self):
        if self.workflowNode is None:
            return ""
        return str(self.workflowNode.GetParameter("ActiveTrajectorySource") or "").strip().lower()

    def _set_workflow_active_source(self, source_key):
        if self.workflowNode is None:
            return
        self.workflowNode.SetParameter("ActiveTrajectorySource", str(source_key or "").strip().lower())

    def _set_guided_source_combo(self, source_key):
        idx = self.guidedSourceCombo.findData(str(source_key or "").strip().lower())
        if idx < 0 or idx == self.guidedSourceCombo.currentIndex:
            return
        self._syncingGuidedSourceCombo = True
        try:
            self.guidedSourceCombo.setCurrentIndex(idx)
        finally:
            self._syncingGuidedSourceCombo = False

    def _sync_guided_source_from_workflow(self):
        key = self._workflow_active_source()
        if not key:
            return
        self._set_guided_source_combo(key)

    def _role_has_nodes(self, role):
        return len(self.workflowState.role_nodes(role, workflow_node=self.workflowNode)) > 0

    def _default_source_when_unset(self):
        # Preferred startup source is imported ROSA if available.
        if self._role_has_nodes("ImportedTrajectoryLines"):
            return "imported_rosa"
        if self._role_has_nodes("GuidedFitTrajectoryLines"):
            return "guided_fit"
        if self._role_has_nodes("AutoFitTrajectoryLines"):
            return "auto_fit"
        if self._role_has_nodes("ImportedExternalTrajectoryLines"):
            return "imported_external"
        if self._role_has_nodes("PlannedTrajectoryLines"):
            return "planned_rosa"
        if self._role_has_nodes("WorkingTrajectoryLines"):
            return "working"
        return "working"

    def _apply_guided_source_visibility(self, source_key):
        key = str(source_key or "working").strip().lower()
        group_map = {
            "working": ["imported_rosa", "imported_external", "manual", "guided_fit", "auto_fit"],
            "imported_rosa": ["imported_rosa"],
            "imported_external": ["imported_external"],
            "manual": ["manual"],
            "guided_fit": ["guided_fit"],
            "auto_fit": ["auto_fit"],
            "planned_rosa": ["planned_rosa"],
        }
        groups = group_map.get(key, group_map["working"])
        self.logic.trajectory_scene.show_only_groups(groups)

    def _refresh_summary(self):
        n_total = len(self.loadedTrajectories)
        # Show confidence-band counts when any trajectory carries one
        # (i.e. Auto-Fit results); manual / imported trajectories have
        # no confidence and are reported as untagged.
        bands = {"high": 0, "medium": 0, "low": 0}
        n_untagged = 0
        for traj in self.loadedTrajectories:
            lab = str(traj.get("confidence_label") or "").strip().lower()
            if lab in bands:
                bands[lab] += 1
            else:
                n_untagged += 1
        if n_untagged == n_total:
            band_text = ""
        else:
            parts = []
            for k in ("high", "medium", "low"):
                if bands[k]:
                    parts.append(f"{bands[k]} {k}")
            if n_untagged:
                parts.append(f"{n_untagged} untagged")
            band_text = " (" + ", ".join(parts) + ")" if parts else ""
        self.summaryLabel.setText(
            f"trajectories={n_total}{band_text}, "
            f"assignments={len(self.assignmentMap)}"
        )

    def onRefreshClicked(self):
        self.workflowNode = self.workflowState.resolve_or_create_workflow_node()
        self._sync_ct_selector_from_workflow()
        if not self._workflow_active_source():
            self._set_workflow_active_source(self._default_source_when_unset())
        self._sync_guided_source_from_workflow()
        source_key = self._selected_guided_source_key()
        self.loadedTrajectories = self.logic.collect_trajectories_by_source(
            source_key=source_key,
            workflow_node=self.workflowNode,
        )
        self.assignmentMap = self._assignment_map_from_workflow()
        self._populate_guided_trajectory_table()
        if hasattr(self, "_refresh_guided_seed_source_combo"):
            self._refresh_guided_seed_source_combo()
        self._apply_guided_source_visibility(source_key)
        self._apply_primary_slice_layers()
        self._refresh_summary()
        self._schedule_guided_follow()
        self.log(
            f"[refresh] source={source_key} trajectories={len(self.loadedTrajectories)} "
            f"assignments={len(self.assignmentMap)}"
        )

    def enter(self):
        """Ensure base+postop overlay and focus behavior each time module is opened."""
        parent_enter = getattr(super(), "enter", None)
        if callable(parent_enter):
            parent_enter()
        self.workflowNode = self.workflowState.resolve_or_create_workflow_node()
        self.logic.layout_service.sanitize_focus_layout_state()
        self._sync_ct_selector_from_workflow()
        self._apply_primary_slice_layers()
        self._schedule_guided_follow()

    def onGuidedSourceChanged(self, _idx):
        if self._syncingGuidedSourceCombo:
            return
        self._set_workflow_active_source(self._selected_guided_source_key())
        self.onRefreshClicked()

    def onModeTabChanged(self, _idx):
        """No per-tab state to persist after the de-novo mode tabs were removed."""
        return

    def onApplyFocusLayoutClicked(self):
        self.logic.layout_service.sanitize_focus_layout_state()
        if not self.logic.layout_service.apply_trajectory_focus_layout():
            self.log("[focus] failed to apply trajectory focus layout")
            return
        self.log("[focus] trajectory focus layout applied")
        self._schedule_guided_follow()

    def onCtSelectorChanged(self, node):
        """Update display layers when CT selection changes."""
        self._lastDeepCoreProposalResult = None
        self.logic.trajectory_scene.remove_preview_lines()
        self._apply_primary_slice_layers()

    def onGuidedTrajectoryTableCellClicked(self, _row, _col):
        self._schedule_guided_follow()

    def onGuidedTrajectoryTableCurrentCellChanged(self, _currentRow, _currentColumn, _previousRow, _previousColumn):
        self._schedule_guided_follow()
