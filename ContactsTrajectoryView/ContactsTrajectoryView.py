"""3D Slicer module for contact generation and trajectory-oriented viewing.

Last updated: 2026-03-01
"""

import os
import sys

from __main__ import ctk, qt, slicer, vtk
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
)

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_CANDIDATES = [
    os.path.join(os.path.dirname(MODULE_DIR), "CommonLib"),
    os.path.join(MODULE_DIR, "CommonLib"),
]
for path in PATH_CANDIDATES:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

from rosa_core import (
    compute_qc_metrics,
    electrode_length_mm,
    generate_contacts,
    load_electrode_library,
    model_map,
    suggest_model_id_for_trajectory,
    trajectory_length_mm,
)
from rosa_scene import (
    ElectrodeSceneService,
    LayoutService,
    TrajectoryFocusController,
    TrajectorySceneService,
    widget_current_text,
)
from rosa_workflow import WorkflowPublisher, WorkflowState

TRAJECTORY_SOURCE_OPTIONS = [
    ("working", "Working (active)"),
    ("imported_rosa", "Imported ROSA"),
    ("imported_external", "Imported External"),
    ("manual", "Manual (scene)"),
    ("guided_fit", "Guided Fit"),
    ("auto_fit", "Auto Fit"),
    ("planned_rosa", "Planned ROSA"),
]


class ContactsTrajectoryView(ScriptedLoadableModule):
    """Slicer module metadata for contact + trajectory workflows."""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "02 Contacts & Trajectory View"
        self.parent.categories = ["ROSA.02 Localization"]
        self.parent.dependencies = []
        self.parent.contributors = ["Ammar Shaikhouni", "Codex"]
        self.parent.helpText = (
            "Generate electrode contacts from trajectory lines and align slice views to trajectories "
            "using the shared RosaWorkflow MRML contract."
        )


class ContactsTrajectoryViewWidget(ScriptedLoadableModuleWidget):
    """UI for electrode model assignment, contact generation, and slice alignment."""

    def setup(self):
        super().setup()

        self.logic = ContactsTrajectoryViewLogic()
        self.workflowState = self.logic.workflow_state
        self.workflowNode = self.workflowState.resolve_or_create_workflow_node()

        self.modelsById = {}
        self.modelIds = []
        self.loadedTrajectories = []
        self.lastGeneratedContacts = []
        self.lastAssignments = {"schema_version": "1.0", "assignments": []}
        self.lastQCMetricsRows = []
        self._syncingSourceCombo = False
        self._syncingFocusControls = False
        self._syncingVolumeSelectors = False
        self._pendingFollow = False
        self._pendingFocusLayoutApply = False
        self._updatingContactTable = False
        self._renamingTrajectory = False
        self._workflowObserverTag = None
        self._workflowObserverNode = None
        self._workflowRefreshPending = False
        self._workflowRefreshInFlight = False

        top_form = qt.QFormLayout()
        self.layout.addLayout(top_form)

        refresh_row = qt.QHBoxLayout()
        self.refreshButton = qt.QPushButton("Refresh Workflow Inputs")
        self.refreshButton.clicked.connect(self.onRefreshClicked)
        refresh_row.addWidget(self.refreshButton)
        self.showPlannedCheck = qt.QCheckBox("Show planned trajectories")
        self.showPlannedCheck.setChecked(False)
        self.showPlannedCheck.toggled.connect(self.onShowPlannedToggled)
        refresh_row.addWidget(self.showPlannedCheck)
        refresh_row.addStretch(1)
        top_form.addRow(refresh_row)

        self.summaryLabel = qt.QLabel("Workflow not scanned yet.")
        self.summaryLabel.wordWrap = True
        top_form.addRow("Workflow summary", self.summaryLabel)

        self.trajectorySourceCombo = qt.QComboBox()
        for key, label in TRAJECTORY_SOURCE_OPTIONS:
            self.trajectorySourceCombo.addItem(label, key)
        self.trajectorySourceCombo.currentIndexChanged.connect(self.onTrajectorySourceChanged)
        top_form.addRow("Trajectory source", self.trajectorySourceCombo)

        self._build_contact_ui()
        self._build_qc_ui()
        self._build_slice_view_ui()

        self.statusText = qt.QPlainTextEdit()
        self.statusText.setReadOnly(True)
        self.statusText.setMaximumBlockCount(1500)
        self.layout.addWidget(self.statusText)
        self.layout.addStretch(1)

        self._load_electrode_library()
        self.logic.layout_service.sanitize_focus_layout_state()
        self._ensure_workflow_observer()
        self.onRefreshClicked()

    def enter(self):
        """Refresh module state when entering this module."""
        self._ensure_workflow_observer()
        self.onRefreshClicked()

    def cleanup(self):
        self._remove_workflow_observer()
        parent_cleanup = getattr(super(), "cleanup", None)
        if callable(parent_cleanup):
            parent_cleanup()

    def _remove_workflow_observer(self):
        node = getattr(self, "_workflowObserverNode", None)
        tag = getattr(self, "_workflowObserverTag", None)
        if node is not None and tag is not None:
            try:
                node.RemoveObserver(tag)
            except Exception:
                pass
        self._workflowObserverNode = None
        self._workflowObserverTag = None

    def _ensure_workflow_observer(self):
        node = self.workflowState.resolve_or_create_workflow_node()
        if node is None:
            self._remove_workflow_observer()
            return
        if node is getattr(self, "_workflowObserverNode", None) and getattr(self, "_workflowObserverTag", None) is not None:
            return
        self._remove_workflow_observer()
        self._workflowObserverNode = node
        self._workflowObserverTag = node.AddObserver(vtk.vtkCommand.ModifiedEvent, self._on_workflow_node_modified)

    def _on_workflow_node_modified(self, caller=None, event=None):
        if self._workflowRefreshPending or self._workflowRefreshInFlight:
            return
        self._workflowRefreshPending = True
        qt.QTimer.singleShot(0, self._refresh_from_workflow_change)

    def _refresh_from_workflow_change(self):
        self._workflowRefreshPending = False
        if self._workflowRefreshInFlight:
            return
        self._workflowRefreshInFlight = True
        try:
            self._sync_source_combo_from_workflow()
            self.onRefreshClicked()
        finally:
            self._workflowRefreshInFlight = False

    def _workflow_active_source(self):
        if self.workflowNode is None:
            return ""
        return str(self.workflowNode.GetParameter("ActiveTrajectorySource") or "").strip().lower()

    def _set_workflow_active_source(self, source_key):
        if self.workflowNode is None:
            return
        key = str(source_key or "").strip().lower()
        self.workflowNode.SetParameter("ActiveTrajectorySource", key)

    def _sync_source_combo_from_workflow(self):
        key = self._workflow_active_source()
        if not key:
            return
        idx = self.trajectorySourceCombo.findData(key)
        if idx < 0 or idx == self.trajectorySourceCombo.currentIndex:
            return
        self._syncingSourceCombo = True
        try:
            self.trajectorySourceCombo.setCurrentIndex(idx)
        finally:
            self._syncingSourceCombo = False

    def _workflow_param_bool(self, key, default=False):
        if self.workflowNode is None:
            return bool(default)
        value = str(self.workflowNode.GetParameter(key) or "").strip().lower()
        if not value:
            return bool(default)
        return value in ("1", "true", "yes", "on")

    def _set_workflow_param_bool(self, key, enabled):
        if self.workflowNode is None:
            return
        self.workflowNode.SetParameter(key, "true" if bool(enabled) else "false")

    def _sync_focus_controls_from_workflow(self):
        autofollow = self._workflow_param_bool("TrajectoryFocusAutoFollow", True)
        self._syncingFocusControls = True
        try:
            self.autoFollowTrajectoryCheck.setChecked(bool(autofollow))
        finally:
            self._syncingFocusControls = False

    def _sync_focus_volume_selectors_from_workflow(self):
        if self.workflowNode is None:
            return
        self._syncingVolumeSelectors = True
        try:
            self.focusBaseSelector.setCurrentNode(self.workflowNode.GetNodeReference("BaseVolume"))
            self.focusPostopSelector.setCurrentNode(self.workflowNode.GetNodeReference("PostopCT"))
        finally:
            self._syncingVolumeSelectors = False

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

    def log(self, message):
        """Append one message line to module log."""
        self.statusText.appendPlainText(str(message))
        self.statusText.ensureCursorVisible()
        try:
            slicer.app.processEvents()
        except Exception:
            pass

    def _widget_value(self, widget):
        if widget is None:
            return 0.0
        value_attr = getattr(widget, "value", 0.0)
        value = value_attr() if callable(value_attr) else value_attr
        return float(value)

    def _build_contact_ui(self):
        """Create contact generation controls."""
        section = ctk.ctkCollapsibleButton()
        section.text = "Contact Labels"
        section.collapsed = False
        self.layout.addWidget(section)
        form = qt.QFormLayout(section)

        self.contactTable = qt.QTableWidget()
        self.contactTable.setColumnCount(7)
        self.contactTable.setHorizontalHeaderLabels(
            [
                "Use",
                "Trajectory",
                "Traj Length (mm)",
                "Electrode Model",
                "Elec Length (mm)",
                "Tip At",
                "Tip Shift (mm)",
            ]
        )
        self.contactTable.horizontalHeader().setSectionResizeMode(0, qt.QHeaderView.ResizeToContents)
        self.contactTable.horizontalHeader().setSectionResizeMode(1, qt.QHeaderView.ResizeToContents)
        self.contactTable.horizontalHeader().setSectionResizeMode(2, qt.QHeaderView.ResizeToContents)
        self.contactTable.horizontalHeader().setSectionResizeMode(3, qt.QHeaderView.Stretch)
        self.contactTable.horizontalHeader().setSectionResizeMode(4, qt.QHeaderView.ResizeToContents)
        self.contactTable.horizontalHeader().setSectionResizeMode(5, qt.QHeaderView.ResizeToContents)
        self.contactTable.horizontalHeader().setSectionResizeMode(6, qt.QHeaderView.ResizeToContents)
        self.contactTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.contactTable.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.contactTable.cellClicked.connect(self.onContactTableCellClicked)
        self.contactTable.currentCellChanged.connect(self.onContactTableCurrentCellChanged)
        self.contactTable.itemChanged.connect(self.onContactTableItemChanged)
        form.addRow(self.contactTable)

        defaults_row = qt.QHBoxLayout()
        self.defaultModelCombo = qt.QComboBox()
        self.defaultModelCombo.setMinimumContentsLength(14)
        defaults_row.addWidget(self.defaultModelCombo)
        self.applyModelAllButton = qt.QPushButton("Apply model to all")
        self.applyModelAllButton.clicked.connect(self.onApplyModelAllClicked)
        defaults_row.addWidget(self.applyModelAllButton)
        defaults_row.addStretch(1)
        form.addRow("Default model", defaults_row)

        self.contactsNodeNameEdit = qt.QLineEdit("ROSA_Contacts")
        form.addRow("Output node prefix", self.contactsNodeNameEdit)

        self.createModelsCheck = qt.QCheckBox("Create electrode models")
        self.createModelsCheck.setChecked(True)
        form.addRow("Model option", self.createModelsCheck)

        button_row = qt.QHBoxLayout()
        self.generateContactsButton = qt.QPushButton("Generate Contact Fiducials")
        self.generateContactsButton.clicked.connect(self.onGenerateContactsClicked)
        self.generateContactsButton.setEnabled(False)
        button_row.addWidget(self.generateContactsButton)
        self.updateContactsButton = qt.QPushButton("Update From Edited Trajectories")
        self.updateContactsButton.clicked.connect(self.onUpdateContactsClicked)
        self.updateContactsButton.setEnabled(False)
        button_row.addWidget(self.updateContactsButton)
        form.addRow(button_row)

    def _build_qc_ui(self):
        """Create QC metric table."""
        self.qcSection = ctk.ctkCollapsibleButton()
        self.qcSection.text = "Trajectory QC Metrics"
        self.qcSection.collapsed = True
        self.layout.addWidget(self.qcSection)
        qf = qt.QFormLayout(self.qcSection)

        self.qcStatusLabel = qt.QLabel("QC disabled: generate contacts first.")
        self.qcStatusLabel.wordWrap = True
        qf.addRow(self.qcStatusLabel)

        self.qcTable = qt.QTableWidget()
        self.qcTable.setColumnCount(8)
        self.qcTable.setHorizontalHeaderLabels(
            [
                "Trajectory",
                "Entry RE (mm)",
                "Target RE (mm)",
                "Mean RE (mm)",
                "Max RE (mm)",
                "RMS RE (mm)",
                "Angle (deg)",
                "N",
            ]
        )
        self.qcTable.horizontalHeader().setSectionResizeMode(0, qt.QHeaderView.ResizeToContents)
        for col in range(1, 8):
            self.qcTable.horizontalHeader().setSectionResizeMode(col, qt.QHeaderView.ResizeToContents)
        self.qcTable.setSelectionMode(qt.QAbstractItemView.NoSelection)
        qf.addRow(self.qcTable)
        self._set_qc_enabled(False, "QC disabled: generate contacts first.")

    def _build_slice_view_ui(self):
        """Create trajectory-focus layout controls."""
        section = ctk.ctkCollapsibleButton()
        section.text = "Trajectory Focus View"
        section.collapsed = False
        self.layout.addWidget(section)
        form = qt.QFormLayout(section)

        volumes_row = qt.QHBoxLayout()
        self.focusBaseSelector = slicer.qMRMLNodeComboBox()
        self.focusBaseSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.focusBaseSelector.noneEnabled = True
        self.focusBaseSelector.addEnabled = False
        self.focusBaseSelector.removeEnabled = False
        self.focusBaseSelector.renameEnabled = False
        self.focusBaseSelector.setMRMLScene(slicer.mrmlScene)
        volumes_row.addWidget(self.focusBaseSelector, 1)

        self.focusPostopSelector = slicer.qMRMLNodeComboBox()
        self.focusPostopSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.focusPostopSelector.noneEnabled = True
        self.focusPostopSelector.addEnabled = False
        self.focusPostopSelector.removeEnabled = False
        self.focusPostopSelector.renameEnabled = False
        self.focusPostopSelector.setMRMLScene(slicer.mrmlScene)
        volumes_row.addWidget(self.focusPostopSelector, 1)

        self.applyFocusVolumesButton = qt.QPushButton("Set Base/Postop")
        self.applyFocusVolumesButton.clicked.connect(self.onApplyFocusVolumesClicked)
        volumes_row.addWidget(self.applyFocusVolumesButton)
        form.addRow("Base / Postop", volumes_row)

        focus_row = qt.QHBoxLayout()
        self.applyFocusLayoutButton = qt.QPushButton("Apply Focus Layout (2x3)")
        self.applyFocusLayoutButton.clicked.connect(self.onApplyFocusLayoutClicked)
        focus_row.addWidget(self.applyFocusLayoutButton)
        self.restoreLayoutButton = qt.QPushButton("Restore Previous Layout")
        self.restoreLayoutButton.clicked.connect(self.onResetFocusLayoutClicked)
        focus_row.addWidget(self.restoreLayoutButton)
        focus_row.addStretch(1)
        form.addRow(focus_row)

        self.autoFollowTrajectoryCheck = qt.QCheckBox("Auto-follow selected trajectory")
        self.autoFollowTrajectoryCheck.setChecked(True)
        self.autoFollowTrajectoryCheck.toggled.connect(self.onAutoFollowToggled)
        form.addRow(self.autoFollowTrajectoryCheck)

    def _set_readonly_text_item(self, row, column, text):
        item = qt.QTableWidgetItem(str(text))
        item.setFlags(item.flags() & ~qt.Qt.ItemIsEditable)
        self.contactTable.setItem(row, column, item)

    def _set_editable_trajectory_item(self, row, text):
        item = qt.QTableWidgetItem(str(text))
        item.setFlags(item.flags() | qt.Qt.ItemIsEditable)
        item.setData(qt.Qt.UserRole, str(text))
        self.contactTable.setItem(row, 1, item)

    def _set_qc_enabled(self, enabled, message=""):
        self.qcSection.setEnabled(bool(enabled))
        self.qcTable.setEnabled(bool(enabled))
        if message:
            self.qcStatusLabel.setText(message)

    def _set_qc_table_item(self, row, column, text):
        item = qt.QTableWidgetItem(str(text))
        item.setFlags(item.flags() & ~qt.Qt.ItemIsEditable)
        self.qcTable.setItem(row, column, item)

    def _build_model_combo(self):
        combo = qt.QComboBox()
        combo.setMinimumContentsLength(14)
        combo.addItem("")
        for model_id in self.modelIds:
            combo.addItem(model_id)
        return combo

    def _build_tip_at_combo(self):
        combo = qt.QComboBox()
        combo.addItems(["target", "entry"])
        return combo

    def _build_tip_shift_spinbox(self):
        spin = qt.QDoubleSpinBox()
        spin.setDecimals(2)
        spin.setRange(-50.0, 50.0)
        spin.setSingleStep(0.25)
        spin.setValue(0.0)
        spin.setSuffix(" mm")
        return spin

    def _build_use_checkbox(self):
        check = qt.QCheckBox()
        check.setChecked(True)
        return check

    def _row_is_selected(self, row):
        check = self.contactTable.cellWidget(row, 0)
        return bool(check and check.checked)

    def _electrode_length_mm(self, model_id):
        model = self.modelsById.get(model_id, {})
        return electrode_length_mm(model)

    def _bind_model_length_update(self, model_combo, row):
        if hasattr(model_combo, "currentTextChanged"):
            model_combo.currentTextChanged.connect(
                lambda _text, row_index=row: self._update_electrode_length_cell(row_index)
            )
        else:
            model_combo.currentIndexChanged.connect(
                lambda _idx, row_index=row: self._update_electrode_length_cell(row_index)
            )

    def _update_electrode_length_cell(self, row):
        model_combo = self.contactTable.cellWidget(row, 3)
        model_id = widget_current_text(model_combo).strip()
        length_text = ""
        if model_id:
            length_text = f"{self._electrode_length_mm(model_id):.2f}"
        self._set_readonly_text_item(row, 4, length_text)

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
        self.defaultModelCombo.clear()
        self.defaultModelCombo.addItem("")
        for model_id in self.modelIds:
            self.defaultModelCombo.addItem(model_id)
        self.log(f"[electrodes] loaded {len(self.modelIds)} models")

    def _populate_contact_table(self, trajectories):
        self._updatingContactTable = True
        self.contactTable.setRowCount(0)
        auto_assigned = 0
        for row, traj in enumerate(trajectories):
            self.contactTable.insertRow(row)
            self.contactTable.setCellWidget(row, 0, self._build_use_checkbox())
            self._set_editable_trajectory_item(row, traj["name"])
            self._set_readonly_text_item(row, 2, f"{trajectory_length_mm(traj):.2f}")

            model_combo = self._build_model_combo()
            self._bind_model_length_update(model_combo, row)
            suggested_model = str(traj.get("best_model_id") or "").strip()
            if not suggested_model:
                suggested_model = suggest_model_id_for_trajectory(
                    trajectory=traj,
                    models_by_id=self.modelsById,
                    model_ids=self.modelIds,
                    tolerance_mm=5.0,
                )
            if suggested_model:
                idx = model_combo.findText(suggested_model)
                if idx >= 0:
                    model_combo.setCurrentIndex(idx)
                    auto_assigned += 1
            self.contactTable.setCellWidget(row, 3, model_combo)
            self._set_readonly_text_item(row, 4, "")
            self._update_electrode_length_cell(row)
            self.contactTable.setCellWidget(row, 5, self._build_tip_at_combo())
            self.contactTable.setCellWidget(row, 6, self._build_tip_shift_spinbox())

        enabled = bool(trajectories) and bool(self.modelsById)
        self.generateContactsButton.setEnabled(enabled)
        self.updateContactsButton.setEnabled(enabled)
        self.applyModelAllButton.setEnabled(enabled)
        if trajectories:
            self.log(f"[contacts] ready for {len(trajectories)} trajectories")
            self.log(f"[contacts] auto-assigned models for {auto_assigned}/{len(trajectories)}")
            self.contactTable.selectRow(0)
        else:
            self.log("[contacts] no trajectories available")
        self._updatingContactTable = False

    def onContactTableItemChanged(self, item):
        if self._updatingContactTable or self._renamingTrajectory:
            return
        if item is None or item.column() != 1:
            return
        row = int(item.row())
        if row < 0 or row >= len(self.loadedTrajectories):
            return

        old_name = str(item.data(qt.Qt.UserRole) or self.loadedTrajectories[row].get("name", "")).strip()
        new_name = (item.text() or "").strip()
        if not old_name:
            return
        if not new_name:
            self._renamingTrajectory = True
            item.setText(old_name)
            self._renamingTrajectory = False
            return
        if new_name == old_name:
            return

        for r in range(self.contactTable.rowCount):
            if r == row:
                continue
            other = self.contactTable.item(r, 1)
            if other and (other.text() or "").strip().lower() == new_name.lower():
                self._renamingTrajectory = True
                item.setText(old_name)
                self._renamingTrajectory = False
                qt.QMessageBox.warning(
                    slicer.util.mainWindow(),
                    "Contacts & Trajectory View",
                    f"Trajectory name '{new_name}' already exists.",
                )
                return

        node_id = self.loadedTrajectories[row].get("node_id", "")
        renamed = self.logic.rename_trajectory(
            node_id=node_id,
            new_name=new_name,
        )
        if not renamed:
            self._renamingTrajectory = True
            item.setText(old_name)
            self._renamingTrajectory = False
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "Contacts & Trajectory View",
                f"Failed to rename trajectory '{old_name}'.",
            )
            return

        self.loadedTrajectories[row]["name"] = new_name
        item.setData(qt.Qt.UserRole, new_name)
        self.log(f"[trajectory] renamed {old_name} -> {new_name}")
        self._refresh_summary()
        self._schedule_follow_selected_trajectory()

    def _collect_assignments(self):
        rows = []
        for row in range(self.contactTable.rowCount):
            if not self._row_is_selected(row):
                continue
            traj_item = self.contactTable.item(row, 1)
            if not traj_item:
                continue
            model_combo = self.contactTable.cellWidget(row, 3)
            tip_at_combo = self.contactTable.cellWidget(row, 5)
            tip_shift_spin = self.contactTable.cellWidget(row, 6)
            model_id = widget_current_text(model_combo).strip()
            if not model_id:
                continue
            rows.append(
                {
                    "trajectory": traj_item.text(),
                    "model_id": model_id,
                    "tip_at": widget_current_text(tip_at_combo) or "target",
                    "tip_shift_mm": self._widget_value(tip_shift_spin),
                    "xyz_offset_mm": [0.0, 0.0, 0.0],
                }
            )
        return {"schema_version": "1.0", "assignments": rows}

    def _build_trajectory_map_with_scene_overrides(self):
        out = {}
        for traj in self.loadedTrajectories:
            node = slicer.mrmlScene.GetNodeByID(traj.get("node_id", ""))
            scene_traj = self.logic.trajectory_scene.trajectory_from_line_node(traj.get("name", ""), node)
            out[traj["name"]] = scene_traj if scene_traj is not None else traj
        return out

    def _compute_qc_rows(self):
        if not self.lastGeneratedContacts:
            return [], "QC disabled: generate contacts first."
        planned_map = self.logic.collect_planned_trajectory_map(workflow_node=self.workflowNode)
        if not planned_map:
            return [], "QC disabled: no planned trajectories (Plan_*) found."
        if not self.lastAssignments.get("assignments"):
            return [], "QC disabled: no electrode assignments available."

        final_map = self._build_trajectory_map_with_scene_overrides()
        try:
            planned_contacts = generate_contacts(
                list(planned_map.values()),
                self.modelsById,
                self.lastAssignments,
            )
        except Exception as exc:
            return [], f"QC disabled: failed to generate planned contacts ({exc})."

        rows = compute_qc_metrics(
            planned_trajectories_by_name=planned_map,
            final_trajectories_by_name=final_map,
            planned_contacts=planned_contacts,
            final_contacts=self.lastGeneratedContacts,
            include_unmatched_planned=True,
        )
        if not rows:
            return [], "QC disabled: no planned trajectories available for comparison."
        return rows, f"QC metrics computed for {len(rows)} planned trajectories."

    def _fmt_qc_metric(self, value):
        if value is None:
            return "NA"
        try:
            return f"{float(value):.2f}"
        except Exception:
            return "NA"

    def _refresh_qc_metrics(self):
        rows, status = self._compute_qc_rows()
        self.lastQCMetricsRows = rows
        self.qcTable.setRowCount(0)
        if not rows:
            self._set_qc_enabled(False, status)
            return
        self._set_qc_enabled(True, status)
        self.qcTable.setRowCount(len(rows))
        for r, row in enumerate(rows):
            self._set_qc_table_item(r, 0, row["trajectory"])
            self._set_qc_table_item(r, 1, self._fmt_qc_metric(row.get("entry_radial_mm")))
            self._set_qc_table_item(r, 2, self._fmt_qc_metric(row.get("target_radial_mm")))
            self._set_qc_table_item(r, 3, self._fmt_qc_metric(row.get("mean_contact_radial_mm")))
            self._set_qc_table_item(r, 4, self._fmt_qc_metric(row.get("max_contact_radial_mm")))
            self._set_qc_table_item(r, 5, self._fmt_qc_metric(row.get("rms_contact_radial_mm")))
            self._set_qc_table_item(r, 6, self._fmt_qc_metric(row.get("angle_deg")))
            self._set_qc_table_item(r, 7, str(row["matched_contacts"]))

    def _refresh_summary(self):
        planned = self.logic.collect_planned_trajectory_map(workflow_node=self.workflowNode)
        source_key = self.trajectorySourceCombo.currentData
        source_key = source_key() if callable(source_key) else source_key
        summary = (
            f"source={source_key or 'working'}, "
            f"loaded trajectories={len(self.loadedTrajectories)}, "
            f"planned trajectories={len(planned)}, "
            f"contacts={len(self.lastGeneratedContacts)}"
        )
        self.summaryLabel.setText(summary)

    def _apply_source_visibility(self, source_key):
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
        # Planned visibility remains user-controlled except when planned source is selected directly.
        if key == "planned_rosa":
            self.logic.electrode_scene.set_planned_trajectory_visibility(True)
        else:
            self.logic.electrode_scene.set_planned_trajectory_visibility(bool(self.showPlannedCheck.checked))

    def onRefreshClicked(self):
        self.workflowNode = self.workflowState.resolve_or_create_workflow_node()
        self.logic.layout_service.sanitize_focus_layout_state()
        if not self._workflow_active_source():
            self._set_workflow_active_source(self._default_source_when_unset())
        self._sync_focus_controls_from_workflow()
        self._sync_source_combo_from_workflow()
        self._sync_focus_volume_selectors_from_workflow()
        source_key = self._selected_source_key()
        self.loadedTrajectories = self.logic.collect_trajectories_by_source(
            source_key=source_key,
            workflow_node=self.workflowNode,
        )
        self.lastGeneratedContacts = []
        self.lastAssignments = {"schema_version": "1.0", "assignments": []}
        self.lastQCMetricsRows = []
        self._populate_contact_table(self.loadedTrajectories)
        self._apply_source_visibility(source_key)
        self._refresh_qc_metrics()
        self._refresh_summary()
        self.log(f"[refresh] source={source_key} trajectories={len(self.loadedTrajectories)}")
        if not self.logic.layout_service.has_focus_slice_views():
            self._schedule_apply_focus_layout()
        self._apply_focus_slice_layers()
        self._schedule_follow_selected_trajectory()

    def _selected_source_key(self):
        data = self.trajectorySourceCombo.currentData
        value = data() if callable(data) else data
        return str(value or "working")

    def onTrajectorySourceChanged(self, _idx):
        if self._syncingSourceCombo:
            return
        self._set_workflow_active_source(self._selected_source_key())
        self.onRefreshClicked()

    def onApplyModelAllClicked(self):
        model_id = widget_current_text(self.defaultModelCombo).strip()
        if not model_id:
            return
        for row in range(self.contactTable.rowCount):
            combo = self.contactTable.cellWidget(row, 3)
            if combo:
                idx = combo.findText(model_id)
                if idx >= 0:
                    combo.setCurrentIndex(idx)

    def _run_contact_generation(self, log_context="generate", allow_last_assignments=False):
        selected_rows = [row for row in range(self.contactTable.rowCount) if self._row_is_selected(row)]
        if not selected_rows:
            raise ValueError("Select at least one trajectory in the Use column.")

        assignments = self._collect_assignments()
        if not assignments["assignments"]:
            if allow_last_assignments and self.lastAssignments.get("assignments"):
                assignments = self.lastAssignments
                self.log(f"[contacts:{log_context}] using last non-empty assignments")
            else:
                raise ValueError("Select electrode models for the selected trajectories.")
        else:
            self.lastAssignments = assignments

        traj_map = self._build_trajectory_map_with_scene_overrides()
        ordered_names = []
        for row in selected_rows:
            item = self.contactTable.item(row, 1)
            if item:
                ordered_names.append(item.text())
        selected_trajectories = [traj_map[name] for name in ordered_names if name in traj_map]
        contacts = generate_contacts(selected_trajectories, self.modelsById, assignments)

        node_prefix = self.contactsNodeNameEdit.text.strip() or "ROSA_Contacts"
        contact_nodes = self.logic.electrode_scene.create_contacts_fiducials_nodes_by_trajectory(
            contacts,
            node_prefix=node_prefix,
        )
        model_nodes = {}
        if self.createModelsCheck.checked:
            model_nodes = self.logic.electrode_scene.create_electrode_models_by_trajectory(
                contacts=contacts,
                trajectories_by_name=traj_map,
                models_by_id=self.modelsById,
                node_prefix=node_prefix,
            )

        self.lastGeneratedContacts = contacts
        assignment_rows = list(assignments.get("assignments", []))
        for row in assignment_rows:
            traj_name = row.get("trajectory", "")
            traj = traj_map.get(traj_name)
            row["trajectory_length_mm"] = 0.0 if traj is None else trajectory_length_mm(traj)
            row["electrode_length_mm"] = electrode_length_mm(self.modelsById.get(row.get("model_id", ""), {}))
            row["source"] = "contacts"

        self._refresh_qc_metrics()
        qc_rows_for_publish = []
        for row in self.lastQCMetricsRows:
            out = dict(row)
            for key in (
                "entry_radial_mm",
                "target_radial_mm",
                "mean_contact_radial_mm",
                "max_contact_radial_mm",
                "rms_contact_radial_mm",
                "angle_deg",
            ):
                if out.get(key) is None:
                    out[key] = "NA"
            qc_rows_for_publish.append(out)
        self.logic.electrode_scene.publish_contacts_outputs(
            contact_nodes_by_traj=contact_nodes,
            model_nodes_by_traj=model_nodes,
            assignment_rows=assignment_rows,
            qc_rows=qc_rows_for_publish,
            workflow_node=self.workflowNode,
        )
        self.log(
            f"[contacts:{log_context}] updated {len(contacts)} points across {len(contact_nodes)} electrode nodes"
        )
        if model_nodes:
            self.log(f"[models:{log_context}] updated {len(model_nodes)} electrode model pairs")
        if self.lastQCMetricsRows:
            self.log(f"[qc:{log_context}] computed metrics for {len(self.lastQCMetricsRows)} trajectories")
        self._refresh_summary()

    def onGenerateContactsClicked(self):
        if not self.loadedTrajectories:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "Contacts & Trajectory View",
                "No trajectories are available in workflow/scene.",
            )
            return
        try:
            self._run_contact_generation(log_context="generate")
        except Exception as exc:
            self.log(f"[contacts] error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Contacts & Trajectory View", str(exc))

    def onUpdateContactsClicked(self):
        if not self.loadedTrajectories:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "Contacts & Trajectory View",
                "No trajectories are available in workflow/scene.",
            )
            return
        if not self.lastGeneratedContacts:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "Contacts & Trajectory View",
                "Generate contacts once first, then use update.",
            )
            return
        try:
            self._run_contact_generation(log_context="update")
        except Exception as exc:
            self.log(f"[contacts:update] error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Contacts & Trajectory View", str(exc))

    def _selected_trajectory_name_from_table(self):
        row = self.contactTable.currentRow()
        if row < 0:
            selected = self.contactTable.selectedItems()
            if selected:
                row = selected[0].row()
        if row < 0 and self.contactTable.rowCount > 0:
            row = 0
        if row < 0:
            return ""
        item = self.contactTable.item(row, 1)
        return (item.text() if item else "").strip()

    def _selected_trajectory(self):
        traj_name = self._selected_trajectory_name_from_table()
        if not traj_name:
            return None
        traj_map = self._build_trajectory_map_with_scene_overrides()
        return traj_map.get(traj_name)

    def _highlight_selected_trajectory(self):
        scope_ids = [traj.get("node_id", "") for traj in self.loadedTrajectories if traj.get("node_id")]
        selected = self._selected_trajectory()
        self.logic.focus_controller.focus_selected(
            trajectory=selected,
            scope_node_ids=scope_ids,
            jump_cardinal=False,
            align_focus_views=False,
        )

    def _resolve_primary_volumes(self):
        base_nodes = self.workflowState.role_nodes("BaseVolume", workflow_node=self.workflowNode)
        post_nodes = self.workflowState.role_nodes("PostopCT", workflow_node=self.workflowNode)
        base_node = base_nodes[0] if base_nodes else None
        post_node = post_nodes[0] if post_nodes else None
        return base_node, post_node

    def _apply_focus_slice_layers(self):
        base_node, post_node = self._resolve_primary_volumes()
        if post_node is not None and base_node is not None:
            bg_node, fg_node, fg_opacity = post_node, base_node, 0.5
        elif post_node is not None:
            bg_node, fg_node, fg_opacity = post_node, None, 0.0
        elif base_node is not None:
            bg_node, fg_node, fg_opacity = base_node, None, 0.0
        else:
            bg_node, fg_node, fg_opacity = None, None, 0.0

        for view_name in (
            "Red",
            "Yellow",
            "Green",
            self.logic.layout_service.TRAJECTORY_LONG_VIEW,
            self.logic.layout_service.TRAJECTORY_DOWN_VIEW,
        ):
            self.logic.electrode_scene.set_slice_view_layers(
                slice_view=view_name,
                background_node=bg_node,
                foreground_node=fg_node,
                foreground_opacity=fg_opacity,
            )

    def _align_focus_views_for_selected(self, align_trajectory_views=True):
        trajectory = self._selected_trajectory()
        scope_ids = [traj.get("node_id", "") for traj in self.loadedTrajectories if traj.get("node_id")]
        try:
            return bool(
                self.logic.focus_controller.focus_selected(
                    trajectory=trajectory,
                    scope_node_ids=scope_ids,
                    jump_cardinal=True,
                    align_focus_views=bool(align_trajectory_views),
                    focus="entry",
                )
            )
        except Exception as exc:
            self.log(f"[focus] alignment fallback: {exc}")
            return False

    def _follow_selected_trajectory_if_enabled(self):
        if self._syncingFocusControls:
            return
        self._highlight_selected_trajectory()
        if not bool(self.autoFollowTrajectoryCheck.checked):
            return
        align_trajectory_views = bool(self.logic.layout_service.has_focus_slice_views())
        self._align_focus_views_for_selected(align_trajectory_views=align_trajectory_views)

    def _schedule_follow_selected_trajectory(self):
        if self._pendingFollow:
            return
        self._pendingFollow = True
        qt.QTimer.singleShot(0, self._run_scheduled_follow)

    def _run_scheduled_follow(self):
        self._pendingFollow = False
        self._follow_selected_trajectory_if_enabled()

    def _schedule_apply_focus_layout(self):
        if self._pendingFocusLayoutApply:
            return
        self._pendingFocusLayoutApply = True
        qt.QTimer.singleShot(0, self._run_scheduled_apply_focus_layout)

    def _run_scheduled_apply_focus_layout(self):
        self._pendingFocusLayoutApply = False
        self.onApplyFocusLayoutClicked()

    def onContactTableCellClicked(self, _row, _col):
        self._schedule_follow_selected_trajectory()

    def onContactTableCurrentCellChanged(self, _currentRow, _currentColumn, _previousRow, _previousColumn):
        self._schedule_follow_selected_trajectory()

    def onAutoFollowToggled(self, checked):
        self._set_workflow_param_bool("TrajectoryFocusAutoFollow", bool(checked))
        self._highlight_selected_trajectory()
        if bool(checked):
            self._schedule_follow_selected_trajectory()

    def onApplyFocusVolumesClicked(self):
        base_node = self.focusBaseSelector.currentNode()
        postop_node = self.focusPostopSelector.currentNode()
        if base_node is not None:
            self.logic.workflow_publish.set_default_role("BaseVolume", base_node, workflow_node=self.workflowNode)
        if postop_node is not None:
            self.logic.workflow_publish.set_default_role("PostopCT", postop_node, workflow_node=self.workflowNode)
        else:
            self.logic.workflow_publish.set_default_role("PostopCT", None, workflow_node=self.workflowNode)
        self._apply_focus_slice_layers()
        self._schedule_follow_selected_trajectory()
        self.log(
            "[focus] set defaults: "
            f"base={(base_node.GetName() if base_node else 'unset')}, "
            f"postop={(postop_node.GetName() if postop_node else 'unset')}"
        )

    def onApplyFocusLayoutClicked(self):
        self.logic.layout_service.sanitize_focus_layout_state()
        if not self.logic.layout_service.apply_trajectory_focus_layout():
            self.log("[focus] failed to apply trajectory focus layout")
            return
        self._apply_focus_slice_layers()
        self._schedule_follow_selected_trajectory()
        self.log("[focus] trajectory focus layout applied")

    def onResetFocusLayoutClicked(self):
        if self.logic.layout_service.restore_previous_layout():
            self.log("[focus] restored previous layout")

    def onAlignSliceClicked(self):
        # Deprecated manual align action retained for backward compatibility.
        self._follow_selected_trajectory_if_enabled()

    def onShowPlannedToggled(self, checked):
        self.logic.electrode_scene.set_planned_trajectory_visibility(bool(checked))


class ContactsTrajectoryViewLogic(ScriptedLoadableModuleLogic):
    """Logic for extracting trajectories and publishing contact scene outputs."""

    def __init__(self):
        super().__init__()
        self.workflow_state = WorkflowState()
        self.workflow_publish = WorkflowPublisher(self.workflow_state)
        self.trajectory_scene = TrajectorySceneService()
        self.layout_service = LayoutService()
        self.electrode_scene = ElectrodeSceneService(
            workflow_state=self.workflow_state,
            workflow_publish=self.workflow_publish,
        )
        self.focus_controller = TrajectoryFocusController(
            trajectory_scene=self.trajectory_scene,
            electrode_scene=self.electrode_scene,
            layout_service=self.layout_service,
        )

    def _collect_trajectories_from_role(self, role, workflow_node=None):
        """Return trajectories from one workflow role."""
        wf = workflow_node or self.workflow_state.resolve_or_create_workflow_node()
        trajectories = []
        role_nodes = self.workflow_state.role_nodes(role, workflow_node=wf)
        for node in role_nodes:
            traj = self.trajectory_scene.trajectory_from_line_node("", node)
            if traj is not None:
                trajectories.append(traj)
        trajectories.sort(key=lambda item: item.get("name", ""))
        return trajectories

    def collect_trajectories_by_source(self, source_key="working", workflow_node=None):
        """Return trajectories for one selected source group."""
        wf = workflow_node or self.workflow_state.resolve_or_create_workflow_node()
        source = str(source_key or "working").strip().lower()

        if source == "working":
            trajectories = self._collect_trajectories_from_role("WorkingTrajectoryLines", workflow_node=wf)
            if trajectories:
                return trajectories
            rows = self.trajectory_scene.collect_working_trajectory_rows(
                groups=["imported_rosa", "imported_external", "manual", "guided_fit", "auto_fit"]
            )
            fallback = []
            for row in rows:
                node = slicer.mrmlScene.GetNodeByID(row["node_id"])
                if node is None:
                    continue
                traj = self.trajectory_scene.trajectory_from_line_node(row["name"], node)
                if traj is not None:
                    fallback.append(traj)
            fallback.sort(key=lambda item: item.get("name", ""))
            return fallback

        if source == "imported_rosa":
            return self._collect_trajectories_from_role("ImportedTrajectoryLines", workflow_node=wf)
        if source == "guided_fit":
            return self._collect_trajectories_from_role("GuidedFitTrajectoryLines", workflow_node=wf)
        if source == "auto_fit":
            return self._collect_trajectories_from_role("AutoFitTrajectoryLines", workflow_node=wf)
        if source == "imported_external":
            return self._collect_trajectories_from_role("ImportedExternalTrajectoryLines", workflow_node=wf)
        if source == "planned_rosa":
            return self._collect_trajectories_from_role("PlannedTrajectoryLines", workflow_node=wf)
        if source == "manual":
            rows = self.trajectory_scene.collect_working_trajectory_rows(groups=["manual"])
            nodes = []
            trajectories = []
            for row in rows:
                node = slicer.mrmlScene.GetNodeByID(row["node_id"])
                if node is None:
                    continue
                self.trajectory_scene.set_trajectory_metadata(
                    node=node,
                    trajectory_name=row["name"],
                    group="manual",
                    origin=node.GetAttribute("Rosa.TrajectoryOrigin") or "manual",
                )
                nodes.append(node)
                traj = self.trajectory_scene.trajectory_from_line_node(row["name"], node)
                if traj is not None:
                    trajectories.append(traj)
            self.workflow_publish.publish_nodes(
                role="ManualTrajectoryLines",
                nodes=nodes,
                source="manual",
                space_name="ROSA_BASE",
                workflow_node=wf,
            )
            self.trajectory_scene.place_trajectory_nodes_in_hierarchy(
                context_id=self.workflow_state.context_id(workflow_node=wf),
                nodes=nodes,
            )
            trajectories.sort(key=lambda item: item.get("name", ""))
            return trajectories

        return []

    def collect_planned_trajectory_map(self, workflow_node=None):
        """Return planned trajectories from role if available, else from Plan_* line nodes."""
        wf = workflow_node or self.workflow_state.resolve_or_create_workflow_node()
        planned = {}
        role_nodes = self.workflow_state.role_nodes("PlannedTrajectoryLines", workflow_node=wf)
        for node in role_nodes:
            name = self.trajectory_scene.logical_name_from_node(node) or (node.GetName() or "")
            traj = self.trajectory_scene.trajectory_from_line_node(name, node)
            if traj is not None:
                planned[name] = traj
        if planned:
            return planned
        return self.trajectory_scene.collect_planned_trajectory_map()

    def rename_trajectory(self, node_id, new_name):
        """Rename one working trajectory node (planned trajectories are left unchanged)."""
        node = slicer.mrmlScene.GetNodeByID(str(node_id or ""))
        if node is None:
            return False
        return bool(self.trajectory_scene.rename_trajectory_node(node, new_name))
