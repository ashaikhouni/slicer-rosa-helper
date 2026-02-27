"""3D Slicer module for contact generation and trajectory-oriented viewing."""

import os
import sys

from __main__ import ctk, qt, slicer
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
    lps_to_ras_point,
    model_map,
    suggest_model_id_for_trajectory,
    trajectory_length_mm,
)
from rosa_scene import ElectrodeSceneService, TrajectorySceneService
from rosa_workflow import WorkflowPublisher, WorkflowState

TRAJECTORY_SOURCE_OPTIONS = [
    ("working", "Working (active)"),
    ("imported_rosa", "Imported ROSA"),
    ("manual", "Manual (scene)"),
    ("guided_fit", "Guided Fit"),
    ("de_novo", "De Novo"),
    ("planned_rosa", "Planned ROSA"),
]


class ContactsTrajectoryView(ScriptedLoadableModule):
    """Slicer module metadata for contact + trajectory workflows."""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "Contacts & Trajectory View"
        self.parent.categories = ["ROSA"]
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
        self.onRefreshClicked()

    def log(self, message):
        """Append one message line to module log."""
        self.statusText.appendPlainText(str(message))
        self.statusText.ensureCursorVisible()
        try:
            slicer.app.processEvents()
        except Exception:
            pass

    def _widget_text(self, widget):
        if widget is None:
            return ""
        text_attr = getattr(widget, "currentText", "")
        return text_attr() if callable(text_attr) else text_attr

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
        self.contactTable.setColumnCount(6)
        self.contactTable.setHorizontalHeaderLabels(
            [
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
        self.contactTable.horizontalHeader().setSectionResizeMode(2, qt.QHeaderView.Stretch)
        self.contactTable.horizontalHeader().setSectionResizeMode(3, qt.QHeaderView.ResizeToContents)
        self.contactTable.horizontalHeader().setSectionResizeMode(4, qt.QHeaderView.ResizeToContents)
        self.contactTable.horizontalHeader().setSectionResizeMode(5, qt.QHeaderView.ResizeToContents)
        self.contactTable.setSelectionMode(qt.QAbstractItemView.NoSelection)
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
        """Create controls to align a slice view to a selected trajectory."""
        section = ctk.ctkCollapsibleButton()
        section.text = "Trajectory Slice View"
        section.collapsed = False
        self.layout.addWidget(section)
        form = qt.QFormLayout(section)

        self.trajectorySelector = qt.QComboBox()
        form.addRow("Trajectory", self.trajectorySelector)

        self.sliceViewSelector = qt.QComboBox()
        self.sliceViewSelector.addItems(["Red", "Yellow", "Green"])
        form.addRow("Slice view", self.sliceViewSelector)

        self.sliceModeSelector = qt.QComboBox()
        self.sliceModeSelector.addItems(["long", "down"])
        form.addRow("Mode", self.sliceModeSelector)

        self.alignSliceButton = qt.QPushButton("Align Slice to Trajectory")
        self.alignSliceButton.clicked.connect(self.onAlignSliceClicked)
        self.alignSliceButton.setEnabled(False)
        form.addRow(self.alignSliceButton)

    def _set_readonly_text_item(self, row, column, text):
        item = qt.QTableWidgetItem(str(text))
        item.setFlags(item.flags() & ~qt.Qt.ItemIsEditable)
        self.contactTable.setItem(row, column, item)

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
        model_combo = self.contactTable.cellWidget(row, 2)
        model_id = self._widget_text(model_combo).strip()
        length_text = ""
        if model_id:
            length_text = f"{self._electrode_length_mm(model_id):.2f}"
        self._set_readonly_text_item(row, 3, length_text)

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
        self.contactTable.setRowCount(0)
        auto_assigned = 0
        for row, traj in enumerate(trajectories):
            self.contactTable.insertRow(row)
            self._set_readonly_text_item(row, 0, traj["name"])
            self._set_readonly_text_item(row, 1, f"{trajectory_length_mm(traj):.2f}")

            model_combo = self._build_model_combo()
            self._bind_model_length_update(model_combo, row)
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
            self.contactTable.setCellWidget(row, 2, model_combo)
            self._set_readonly_text_item(row, 3, "")
            self._update_electrode_length_cell(row)
            self.contactTable.setCellWidget(row, 4, self._build_tip_at_combo())
            self.contactTable.setCellWidget(row, 5, self._build_tip_shift_spinbox())

        enabled = bool(trajectories) and bool(self.modelsById)
        self.generateContactsButton.setEnabled(enabled)
        self.updateContactsButton.setEnabled(enabled)
        self.applyModelAllButton.setEnabled(enabled)
        if trajectories:
            self.log(f"[contacts] ready for {len(trajectories)} trajectories")
            self.log(f"[contacts] auto-assigned models for {auto_assigned}/{len(trajectories)}")
        else:
            self.log("[contacts] no trajectories available")

    def _populate_trajectory_selector(self, trajectories):
        self.trajectorySelector.clear()
        for traj in trajectories:
            group = str(traj.get("group", "") or "unknown")
            label = f"[{group}] {traj['name']}"
            self.trajectorySelector.addItem(label, traj["name"])
        self.alignSliceButton.setEnabled(bool(trajectories))

    def _collect_assignments(self):
        rows = []
        for row in range(self.contactTable.rowCount):
            traj_item = self.contactTable.item(row, 0)
            if not traj_item:
                continue
            model_combo = self.contactTable.cellWidget(row, 2)
            tip_at_combo = self.contactTable.cellWidget(row, 4)
            tip_shift_spin = self.contactTable.cellWidget(row, 5)
            model_id = self._widget_text(model_combo).strip()
            if not model_id:
                continue
            rows.append(
                {
                    "trajectory": traj_item.text(),
                    "model_id": model_id,
                    "tip_at": self._widget_text(tip_at_combo) or "target",
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
        )
        if not rows:
            return [], "QC disabled: no matching planned/final trajectories with contacts."
        return rows, f"QC metrics computed for {len(rows)} trajectories."

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
            self._set_qc_table_item(r, 1, f"{row['entry_radial_mm']:.2f}")
            self._set_qc_table_item(r, 2, f"{row['target_radial_mm']:.2f}")
            self._set_qc_table_item(r, 3, f"{row['mean_contact_radial_mm']:.2f}")
            self._set_qc_table_item(r, 4, f"{row['max_contact_radial_mm']:.2f}")
            self._set_qc_table_item(r, 5, f"{row['rms_contact_radial_mm']:.2f}")
            self._set_qc_table_item(r, 6, f"{row['angle_deg']:.2f}")
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

    def onRefreshClicked(self):
        self.workflowNode = self.workflowState.resolve_or_create_workflow_node()
        source_key = self._selected_source_key()
        self.loadedTrajectories = self.logic.collect_trajectories_by_source(
            source_key=source_key,
            workflow_node=self.workflowNode,
        )
        self.lastGeneratedContacts = []
        self.lastAssignments = {"schema_version": "1.0", "assignments": []}
        self.lastQCMetricsRows = []
        self._populate_contact_table(self.loadedTrajectories)
        self._populate_trajectory_selector(self.loadedTrajectories)
        self._refresh_qc_metrics()
        self._refresh_summary()
        self.log(f"[refresh] source={source_key} trajectories={len(self.loadedTrajectories)}")

    def _selected_source_key(self):
        data = self.trajectorySourceCombo.currentData
        value = data() if callable(data) else data
        return str(value or "working")

    def onTrajectorySourceChanged(self, _idx):
        self.onRefreshClicked()

    def onApplyModelAllClicked(self):
        model_id = self._widget_text(self.defaultModelCombo).strip()
        if not model_id:
            return
        for row in range(self.contactTable.rowCount):
            combo = self.contactTable.cellWidget(row, 2)
            if combo:
                idx = combo.findText(model_id)
                if idx >= 0:
                    combo.setCurrentIndex(idx)

    def _run_contact_generation(self, log_context="generate", allow_last_assignments=False):
        assignments = self._collect_assignments()
        if not assignments["assignments"]:
            if allow_last_assignments and self.lastAssignments.get("assignments"):
                assignments = self.lastAssignments
                self.log(f"[contacts:{log_context}] using last non-empty assignments")
            else:
                raise ValueError("Select at least one electrode model in the assignment table.")
        else:
            self.lastAssignments = assignments

        traj_map = self._build_trajectory_map_with_scene_overrides()
        ordered_names = []
        for row in range(self.contactTable.rowCount):
            item = self.contactTable.item(row, 0)
            if item:
                ordered_names.append(item.text())
        self.loadedTrajectories = [traj_map[name] for name in ordered_names if name in traj_map]
        contacts = generate_contacts(self.loadedTrajectories, self.modelsById, assignments)

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
        self.logic.electrode_scene.publish_contacts_outputs(
            contact_nodes_by_traj=contact_nodes,
            model_nodes_by_traj=model_nodes,
            assignment_rows=assignment_rows,
            qc_rows=self.lastQCMetricsRows,
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

    def onAlignSliceClicked(self):
        data = self.trajectorySelector.currentData
        traj_name = data() if callable(data) else data
        traj_name = str(traj_name or "").strip()
        if not traj_name:
            traj_name = self._widget_text(self.trajectorySelector).strip()
        if not traj_name:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Contacts & Trajectory View", "No trajectory selected.")
            return

        traj_map = self._build_trajectory_map_with_scene_overrides()
        trajectory = traj_map.get(traj_name)
        if trajectory is None:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "Contacts & Trajectory View",
                f"Trajectory '{traj_name}' not found.",
            )
            return

        start_ras = lps_to_ras_point(trajectory["start"])
        end_ras = lps_to_ras_point(trajectory["end"])
        slice_view = self._widget_text(self.sliceViewSelector) or "Red"
        mode = self._widget_text(self.sliceModeSelector) or "long"

        try:
            self.logic.electrode_scene.align_slice_to_trajectory(
                start_ras=start_ras,
                end_ras=end_ras,
                slice_view=slice_view,
                mode=mode,
            )
        except Exception as exc:
            self.log(f"[slice] error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Contacts & Trajectory View", str(exc))
            return

        self.log(f"[slice] aligned {slice_view} view to {traj_name} ({mode})")

    def onShowPlannedToggled(self, checked):
        self.logic.electrode_scene.set_planned_trajectory_visibility(bool(checked))


class ContactsTrajectoryViewLogic(ScriptedLoadableModuleLogic):
    """Logic for extracting trajectories and publishing contact scene outputs."""

    def __init__(self):
        super().__init__()
        self.workflow_state = WorkflowState()
        self.workflow_publish = WorkflowPublisher(self.workflow_state)
        self.trajectory_scene = TrajectorySceneService()
        self.electrode_scene = ElectrodeSceneService(
            workflow_state=self.workflow_state,
            workflow_publish=self.workflow_publish,
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
                groups=["imported_rosa", "manual", "guided_fit", "de_novo"]
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
        if source == "de_novo":
            return self._collect_trajectories_from_role("DeNovoTrajectoryLines", workflow_node=wf)
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
