"""Widget mixin for ROSA Helper UI workflows.

This file groups UI section builders and feature handlers so the module
entrypoint stays focused on wiring.
"""

import os

from __main__ import ctk, qt, slicer

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


class RosaHelperWidgetMixin:
    """UI helper and workflow handlers mixed into `RosaHelperWidget`."""

    def _build_contact_ui(self):
        """Create V1 contact labeling controls."""
        self.contactSection = ctk.ctkCollapsibleButton()
        self.contactSection.text = "V1 Contact Labels"
        self.contactSection.collapsed = True
        self.layout.addWidget(self.contactSection)

        contact_layout = qt.QFormLayout(self.contactSection)

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
        contact_layout.addRow(self.contactTable)

        defaults_row = qt.QHBoxLayout()
        self.defaultModelCombo = qt.QComboBox()
        self.defaultModelCombo.setMinimumContentsLength(14)
        defaults_row.addWidget(self.defaultModelCombo)
        self.applyModelAllButton = qt.QPushButton("Apply model to all")
        self.applyModelAllButton.clicked.connect(self.onApplyModelAllClicked)
        defaults_row.addWidget(self.applyModelAllButton)
        defaults_row.addStretch(1)
        contact_layout.addRow("Default model", defaults_row)

        self.contactsNodeNameEdit = qt.QLineEdit("ROSA_Contacts")
        contact_layout.addRow("Output markups", self.contactsNodeNameEdit)

        self.createModelsCheck = qt.QCheckBox("Create electrode models")
        self.createModelsCheck.setChecked(True)
        self.createModelsCheck.setToolTip(
            "Create per-electrode 3D model nodes (shaft + contacts) after contact generation."
        )
        contact_layout.addRow("Model option", self.createModelsCheck)

        self.generateContactsButton = qt.QPushButton("Generate Contact Fiducials")
        self.generateContactsButton.clicked.connect(self.onGenerateContactsClicked)
        self.generateContactsButton.setEnabled(False)
        contact_layout.addRow(self.generateContactsButton)

        self.updateContactsButton = qt.QPushButton("Update From Edited Trajectories")
        self.updateContactsButton.clicked.connect(self.onUpdateContactsClicked)
        self.updateContactsButton.setEnabled(False)
        self.updateContactsButton.setToolTip(
            "Recompute contacts/models from current trajectory line markups without creating new nodes."
        )
        contact_layout.addRow(self.updateContactsButton)

        self.bundleExportDirEdit = qt.QLineEdit()
        self.bundleExportDirEdit.setPlaceholderText("Optional (defaults to <case>/RosaHelper_Export)")
        contact_layout.addRow("Aligned export folder", self.bundleExportDirEdit)

        self.exportBundleButton = qt.QPushButton("Export Aligned NIfTI + Coordinates/QC")
        self.exportBundleButton.clicked.connect(self.onExportAlignedBundleClicked)
        self.exportBundleButton.setEnabled(False)
        contact_layout.addRow(self.exportBundleButton)

    def _build_trajectory_view_ui(self):
        """Create controls to align a slice view to a selected trajectory."""
        self.trajectoryViewSection = ctk.ctkCollapsibleButton()
        self.trajectoryViewSection.text = "Trajectory Slice View"
        self.trajectoryViewSection.collapsed = True
        self.layout.addWidget(self.trajectoryViewSection)

        view_layout = qt.QFormLayout(self.trajectoryViewSection)

        self.trajectorySelector = qt.QComboBox()
        view_layout.addRow("Trajectory", self.trajectorySelector)

        self.sliceViewSelector = qt.QComboBox()
        self.sliceViewSelector.addItems(["Red", "Yellow", "Green"])
        view_layout.addRow("Slice view", self.sliceViewSelector)

        self.sliceModeSelector = qt.QComboBox()
        self.sliceModeSelector.addItems(["long", "down"])
        self.sliceModeSelector.setToolTip(
            "'long': slice along electrode shaft, 'down': view along shaft axis."
        )
        view_layout.addRow("Mode", self.sliceModeSelector)

        self.showPlannedCheck = qt.QCheckBox("Show planned trajectories")
        self.showPlannedCheck.setChecked(False)
        self.showPlannedCheck.toggled.connect(self.onShowPlannedToggled)
        view_layout.addRow(self.showPlannedCheck)

        self.alignSliceButton = qt.QPushButton("Align Slice to Trajectory")
        self.alignSliceButton.setEnabled(False)
        self.alignSliceButton.clicked.connect(self.onAlignSliceClicked)
        view_layout.addRow(self.alignSliceButton)

    def _build_freesurfer_ui(self):
        """Create controls for FreeSurfer MRI registration and surface import."""
        self.freesurferSection = ctk.ctkCollapsibleButton()
        self.freesurferSection.text = "FreeSurfer Integration (V1)"
        self.freesurferSection.collapsed = True
        self.layout.addWidget(self.freesurferSection)

        fs_layout = qt.QFormLayout(self.freesurferSection)

        self.fsFixedSelector = slicer.qMRMLNodeComboBox()
        self.fsFixedSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.fsFixedSelector.noneEnabled = True
        self.fsFixedSelector.addEnabled = False
        self.fsFixedSelector.removeEnabled = False
        self.fsFixedSelector.setMRMLScene(slicer.mrmlScene)
        self.fsFixedSelector.setToolTip("ROSA base/reference volume (fixed image)")
        fs_layout.addRow("ROSA base volume", self.fsFixedSelector)

        self.fsMovingSelector = slicer.qMRMLNodeComboBox()
        self.fsMovingSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.fsMovingSelector.noneEnabled = True
        self.fsMovingSelector.addEnabled = False
        self.fsMovingSelector.removeEnabled = False
        self.fsMovingSelector.setMRMLScene(slicer.mrmlScene)
        self.fsMovingSelector.setToolTip("MRI used for recon-all (moving image)")
        fs_layout.addRow("FreeSurfer MRI", self.fsMovingSelector)

        self.fsInitModeCombo = qt.QComboBox()
        self.fsInitModeCombo.addItems(["useGeometryAlign", "useMomentsAlign"])
        self.fsInitModeCombo.setCurrentText("useGeometryAlign")
        fs_layout.addRow("Init mode", self.fsInitModeCombo)

        self.fsTransformNameEdit = qt.QLineEdit("FS_to_ROSA")
        fs_layout.addRow("Output transform", self.fsTransformNameEdit)

        self.fsRegisterButton = qt.QPushButton("Register FS MRI -> ROSA")
        self.fsRegisterButton.clicked.connect(self.onRegisterFSMRIToRosaClicked)
        fs_layout.addRow(self.fsRegisterButton)

        self.fsSubjectDirSelector = ctk.ctkPathLineEdit()
        self.fsSubjectDirSelector.filters = ctk.ctkPathLineEdit.Dirs
        self.fsSubjectDirSelector.setToolTip(
            "FreeSurfer subject directory (contains surf/) or direct surf/ directory"
        )
        fs_layout.addRow("FreeSurfer subject", self.fsSubjectDirSelector)

        self.fsSurfaceSetCombo = qt.QComboBox()
        self.fsSurfaceSetCombo.addItems(["pial", "white", "pial+white", "inflated"])
        self.fsSurfaceSetCombo.setCurrentText("pial")
        fs_layout.addRow("Surface set", self.fsSurfaceSetCombo)

        self.fsUseAnnotationCheck = qt.QCheckBox("Load annotation scalars")
        self.fsUseAnnotationCheck.setChecked(True)
        fs_layout.addRow(self.fsUseAnnotationCheck)

        self.fsAnnotationCombo = qt.QComboBox()
        self.fsAnnotationCombo.addItems(["aparc", "aparc.a2009s", "aparc.DKTatlas", "custom"])
        self.fsAnnotationCombo.setCurrentText("aparc")
        fs_layout.addRow("Annotation", self.fsAnnotationCombo)

        self.fsCustomAnnotationEdit = qt.QLineEdit()
        self.fsCustomAnnotationEdit.setPlaceholderText("Custom annotation name (without lh./rh. and .annot)")
        fs_layout.addRow("Custom annot", self.fsCustomAnnotationEdit)

        self.fsLUTPathSelector = ctk.ctkPathLineEdit()
        self.fsLUTPathSelector.filters = ctk.ctkPathLineEdit.Files
        self.fsLUTPathSelector.setToolTip(
            "Optional color LUT text file (for example FreeSurferColorLUT.txt). "
            "If empty, a default FreeSurfer color table node is used when available."
        )
        fs_layout.addRow("Annotation LUT", self.fsLUTPathSelector)

        self.fsApplyTransformCheck = qt.QCheckBox("Apply FS->ROSA transform")
        self.fsApplyTransformCheck.setChecked(True)
        fs_layout.addRow(self.fsApplyTransformCheck)

        self.fsHardenSurfaceCheck = qt.QCheckBox("Harden loaded surface transforms")
        self.fsHardenSurfaceCheck.setChecked(False)
        fs_layout.addRow(self.fsHardenSurfaceCheck)

        self.fsLoadSurfacesButton = qt.QPushButton("Load FreeSurfer Surfaces")
        self.fsLoadSurfacesButton.clicked.connect(self.onLoadFSSurfacesClicked)
        fs_layout.addRow(self.fsLoadSurfacesButton)

    def _build_qc_ui(self):
        """Create QC table that updates automatically after contact generation."""
        self.qcSection = ctk.ctkCollapsibleButton()
        self.qcSection.text = "Trajectory QC Metrics"
        self.qcSection.collapsed = True
        self.layout.addWidget(self.qcSection)

        qc_layout = qt.QFormLayout(self.qcSection)
        self.qcStatusLabel = qt.QLabel("QC metrics are unavailable until contacts are generated.")
        qc_layout.addRow(self.qcStatusLabel)

        self.qcTable = qt.QTableWidget()
        self.qcTable.setColumnCount(8)
        self.qcTable.setHorizontalHeaderLabels(
            [
                "Trajectory",
                "Entry Radial (mm)",
                "Target Radial (mm)",
                "Mean Contact Radial (mm)",
                "Max Contact Radial (mm)",
                "RMS Contact Radial (mm)",
                "Angle (deg)",
                "Matched Contacts",
            ]
        )
        self.qcTable.horizontalHeader().setSectionResizeMode(0, qt.QHeaderView.ResizeToContents)
        for col in range(1, 8):
            self.qcTable.horizontalHeader().setSectionResizeMode(col, qt.QHeaderView.ResizeToContents)
        self.qcTable.setSelectionMode(qt.QAbstractItemView.NoSelection)
        self.qcTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        qc_layout.addRow(self.qcTable)
        self._set_qc_enabled(False, "QC metrics are unavailable until contacts are generated.")

    def _build_autofit_ui(self):
        """Create controls for automatic postop CT alignment of trajectories."""
        self.autoFitSection = ctk.ctkCollapsibleButton()
        self.autoFitSection.text = "Auto Align to Postop CT (V1)"
        self.autoFitSection.collapsed = True
        self.layout.addWidget(self.autoFitSection)

        form = qt.QFormLayout(self.autoFitSection)

        self.postopCTSelector = slicer.qMRMLNodeComboBox()
        self.postopCTSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.postopCTSelector.noneEnabled = True
        self.postopCTSelector.addEnabled = False
        self.postopCTSelector.removeEnabled = False
        self.postopCTSelector.setMRMLScene(slicer.mrmlScene)
        form.addRow("Postop CT", self.postopCTSelector)

        self.autoThresholdSpin = qt.QDoubleSpinBox()
        self.autoThresholdSpin.setRange(-2000.0, 6000.0)
        self.autoThresholdSpin.setDecimals(1)
        self.autoThresholdSpin.setValue(1800.0)
        self.autoThresholdSpin.setSingleStep(50.0)
        form.addRow("Threshold (HU)", self.autoThresholdSpin)

        self.resetCTWindowButton = qt.QPushButton("Reset CT Window")
        self.resetCTWindowButton.clicked.connect(self.onResetCTWindowClicked)
        form.addRow(self.resetCTWindowButton)

        self.autoRoiRadiusSpin = qt.QDoubleSpinBox()
        self.autoRoiRadiusSpin.setRange(0.5, 20.0)
        self.autoRoiRadiusSpin.setDecimals(2)
        self.autoRoiRadiusSpin.setValue(2.5)
        self.autoRoiRadiusSpin.setSingleStep(0.25)
        self.autoRoiRadiusSpin.setSuffix(" mm")
        form.addRow("ROI radius", self.autoRoiRadiusSpin)

        self.autoMaxAngleSpin = qt.QDoubleSpinBox()
        self.autoMaxAngleSpin.setRange(0.0, 90.0)
        self.autoMaxAngleSpin.setDecimals(1)
        self.autoMaxAngleSpin.setValue(12.0)
        self.autoMaxAngleSpin.setSingleStep(0.5)
        self.autoMaxAngleSpin.setSuffix(" deg")
        form.addRow("Max angle", self.autoMaxAngleSpin)

        self.autoMaxDepthShiftSpin = qt.QDoubleSpinBox()
        self.autoMaxDepthShiftSpin.setRange(0.0, 50.0)
        self.autoMaxDepthShiftSpin.setDecimals(2)
        self.autoMaxDepthShiftSpin.setValue(20.0)
        self.autoMaxDepthShiftSpin.setSingleStep(0.5)
        self.autoMaxDepthShiftSpin.setSuffix(" mm")
        form.addRow("Max depth shift", self.autoMaxDepthShiftSpin)

        self.autoFitTrajectorySelector = qt.QComboBox()
        form.addRow("Selected trajectory", self.autoFitTrajectorySelector)

        btn_row = qt.QHBoxLayout()
        self.detectCandidatesButton = qt.QPushButton("Detect Candidates")
        self.detectCandidatesButton.clicked.connect(self.onDetectCandidatesClicked)
        btn_row.addWidget(self.detectCandidatesButton)
        self.fitSelectedButton = qt.QPushButton("Fit Selected")
        self.fitSelectedButton.clicked.connect(self.onFitSelectedClicked)
        btn_row.addWidget(self.fitSelectedButton)
        self.fitAllButton = qt.QPushButton("Fit All")
        self.fitAllButton.clicked.connect(self.onFitAllClicked)
        btn_row.addWidget(self.fitAllButton)
        form.addRow(btn_row)

        self.applyFitButton = qt.QPushButton("Apply Fit to Trajectories")
        self.applyFitButton.clicked.connect(self.onApplyFitClicked)
        form.addRow(self.applyFitButton)
        self._set_autofit_buttons_enabled(False)

    def _set_autofit_buttons_enabled(self, enabled):
        """Enable/disable auto-fit action buttons."""
        enabled = bool(enabled)
        self.fitSelectedButton.setEnabled(enabled)
        self.fitAllButton.setEnabled(enabled)
        self.applyFitButton.setEnabled(enabled and bool(self.autoFitResults))

    def _autofit_candidate_count(self):
        """Return number of detected postop CT candidate points."""
        candidates = self.autoFitCandidatesLPS
        if candidates is None:
            return 0
        size_attr = getattr(candidates, "shape", None)
        if size_attr is not None:
            return int(candidates.shape[0])
        return len(candidates)

    def _load_electrode_library(self):
        """Load bundled electrode models used by V1 contact generation."""
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

    def _build_model_combo(self):
        """Create a combo box populated with electrode model IDs."""
        combo = qt.QComboBox()
        combo.addItem("")
        for model_id in self.modelIds:
            combo.addItem(model_id)
        return combo

    def _build_tip_at_combo(self):
        """Create tip anchor selector for contact placement."""
        combo = qt.QComboBox()
        combo.addItems(["target", "entry"])
        return combo

    def _build_tip_shift_spinbox(self):
        """Create spinbox for tip shift in millimeters."""
        spin = qt.QDoubleSpinBox()
        spin.setDecimals(2)
        spin.setRange(-50.0, 50.0)
        spin.setSingleStep(0.25)
        spin.setValue(0.0)
        spin.setSuffix(" mm")
        return spin

    def _widget_text(self, widget):
        """Return current text from Qt widgets across Python bindings."""
        if widget is None:
            return ""
        text_attr = getattr(widget, "currentText", "")
        return text_attr() if callable(text_attr) else text_attr

    def _widget_value(self, widget):
        """Return numeric value from Qt widgets across Python bindings."""
        if widget is None:
            return 0.0
        value_attr = getattr(widget, "value", 0.0)
        value = value_attr() if callable(value_attr) else value_attr
        return float(value)

    def _set_readonly_text_item(self, row, column, text):
        """Set a read-only table item string value."""
        item = qt.QTableWidgetItem(text)
        item.setFlags(item.flags() & ~qt.Qt.ItemIsEditable)
        self.contactTable.setItem(row, column, item)

    def _set_qc_enabled(self, enabled, message=""):
        """Enable/disable QC section and update status message."""
        self.qcSection.setEnabled(bool(enabled))
        self.qcTable.setEnabled(bool(enabled))
        if message:
            self.qcStatusLabel.setText(message)

    def _set_qc_table_item(self, row, column, text):
        """Set one read-only item in QC table."""
        item = qt.QTableWidgetItem(str(text))
        item.setFlags(item.flags() & ~qt.Qt.ItemIsEditable)
        self.qcTable.setItem(row, column, item)

    def _collect_planned_trajectory_map(self):
        """Read planned line markups (`Plan_*`) into trajectory dictionaries."""
        return self.logic.trajectory_scene.collect_planned_trajectory_map()

    def _compute_qc_rows(self):
        """Compute per-trajectory QC metrics using planned vs current contacts/lines."""
        if not self.lastGeneratedContacts:
            return [], "QC disabled: generate contacts first."

        planned_map = self._collect_planned_trajectory_map()
        if not planned_map:
            return [], "QC disabled: no planned trajectories (Plan_*) found."

        if not self.lastAssignments.get("assignments"):
            return [], "QC disabled: no electrode assignments available."

        final_map = self._build_trajectory_map_with_scene_overrides()
        assignments = self.lastAssignments

        try:
            planned_contacts = generate_contacts(list(planned_map.values()), self.modelsById, assignments)
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
        """Refresh QC table state from current trajectories, assignments, and contacts."""
        rows, status = self._compute_qc_rows()
        self.lastQCMetricsRows = rows
        self.qcTable.setRowCount(0)
        if not rows:
            self._set_qc_enabled(False, status)
            return

        self._set_qc_enabled(True, status)
        self.qcTable.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            self._set_qc_table_item(row_index, 0, row["trajectory"])
            self._set_qc_table_item(row_index, 1, f"{row['entry_radial_mm']:.2f}")
            self._set_qc_table_item(row_index, 2, f"{row['target_radial_mm']:.2f}")
            self._set_qc_table_item(row_index, 3, f"{row['mean_contact_radial_mm']:.2f}")
            self._set_qc_table_item(row_index, 4, f"{row['max_contact_radial_mm']:.2f}")
            self._set_qc_table_item(row_index, 5, f"{row['rms_contact_radial_mm']:.2f}")
            self._set_qc_table_item(row_index, 6, f"{row['angle_deg']:.2f}")
            self._set_qc_table_item(row_index, 7, str(row["matched_contacts"]))

    def _electrode_length_mm(self, model_id):
        """Return electrode exploration length in millimeters for model ID."""
        model = self.modelsById.get(model_id, {})
        return electrode_length_mm(model)

    def _update_electrode_length_cell(self, row):
        """Refresh per-row electrode length after model selection change."""
        model_combo = self.contactTable.cellWidget(row, 2)
        model_id = self._widget_text(model_combo).strip()
        length_text = ""
        if model_id:
            length_text = f"{self._electrode_length_mm(model_id):.2f}"
        self._set_readonly_text_item(row, 3, length_text)

    def _bind_model_length_update(self, model_combo, row):
        """Bind combo change signal to row-specific electrode length update."""
        if hasattr(model_combo, "currentTextChanged"):
            model_combo.currentTextChanged.connect(
                lambda _text, row_index=row: self._update_electrode_length_cell(row_index)
            )
        else:
            model_combo.currentIndexChanged.connect(
                lambda _idx, row_index=row: self._update_electrode_length_cell(row_index)
            )

    def _trajectory_length_mm(self, trajectory):
        """Return Euclidean distance between entry/start and target/end in millimeters."""
        return trajectory_length_mm(trajectory)

    def _suggest_model_id_for_trajectory(self, trajectory):
        """Select model with closest length, constrained to +/- 5 mm from trajectory."""
        return suggest_model_id_for_trajectory(
            trajectory=trajectory,
            models_by_id=self.modelsById,
            model_ids=self.modelIds,
            tolerance_mm=5.0,
        )

    def _populate_contact_table(self, trajectories):
        """Fill assignment table with one row per trajectory."""
        self.contactTable.setRowCount(0)
        auto_assigned = 0
        for row, traj in enumerate(trajectories):
            self.contactTable.insertRow(row)
            self._set_readonly_text_item(row, 0, traj["name"])
            traj_len = self._trajectory_length_mm(traj)
            self._set_readonly_text_item(row, 1, f"{traj_len:.2f}")
            model_combo = self._build_model_combo()
            self._bind_model_length_update(model_combo, row)
            suggested_model = self._suggest_model_id_for_trajectory(traj)
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
        self.exportBundleButton.setEnabled(False)
        if trajectories:
            self.log(f"[contacts] ready for {len(trajectories)} trajectories")
            self.log(f"[contacts] auto-assigned models for {auto_assigned}/{len(trajectories)}")
        else:
            self.log("[contacts] no trajectories found in case")

    def _populate_trajectory_selector(self, trajectories):
        """Populate trajectory selector used for slice alignment."""
        self.trajectorySelector.clear()
        for traj in trajectories:
            self.trajectorySelector.addItem(traj["name"])
        self.alignSliceButton.setEnabled(bool(trajectories))

    def _populate_autofit_trajectory_selector(self, trajectories):
        """Populate selected-trajectory dropdown in auto-fit section."""
        self.autoFitTrajectorySelector.clear()
        for traj in trajectories:
            self.autoFitTrajectorySelector.addItem(traj["name"])

    def _trajectory_by_name(self, name):
        """Find trajectory dictionary by name."""
        for traj in self.loadedTrajectories:
            if traj["name"] == name:
                return traj
        return None

    def _find_line_markup_node(self, name):
        """Return trajectory line markup node by exact name."""
        return self.logic.trajectory_scene.find_line_markup_node(name)

    def _trajectory_from_line_node(self, name, node):
        """Extract trajectory start/end from a line node as ROSA/LPS points."""
        return self.logic.trajectory_scene.trajectory_from_line_node(name, node)

    def _build_trajectory_map_with_scene_overrides(self):
        """Return trajectory map using current scene markups when available."""
        return self.logic.trajectory_scene.build_trajectory_map_with_scene_overrides(
            self.loadedTrajectories
        )

    def _refresh_trajectory_lengths_from_scene(self):
        """Refresh trajectory length column from currently edited line markups."""
        traj_map = self._build_trajectory_map_with_scene_overrides()
        for row in range(self.contactTable.rowCount):
            name_item = self.contactTable.item(row, 0)
            if not name_item:
                continue
            name = name_item.text()
            traj = traj_map.get(name)
            if traj is None:
                continue
            self._set_readonly_text_item(row, 1, f"{self._trajectory_length_mm(traj):.2f}")

    def _run_contact_generation(self, log_context="generate", allow_last_assignments=False):
        """Compute contacts from table assignments and current trajectory markup positions."""
        assignments = self._collect_assignments()
        if not assignments["assignments"]:
            if allow_last_assignments and self.lastAssignments.get("assignments"):
                assignments = self.lastAssignments
                self.log(f"[contacts:{log_context}] using last non-empty electrode assignments")
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
        self._refresh_trajectory_lengths_from_scene()

        try:
            contacts = generate_contacts(self.loadedTrajectories, self.modelsById, assignments)
        except Exception as exc:
            raise ValueError(str(exc))

        node_prefix = self.contactsNodeNameEdit.text.strip() or "ROSA_Contacts"
        nodes = self.logic.create_contacts_fiducials_nodes_by_trajectory(
            contacts,
            node_prefix=node_prefix,
        )

        model_nodes = {}
        if self.createModelsCheck.checked:
            model_nodes = self.logic.create_electrode_models_by_trajectory(
                contacts=contacts,
                trajectories_by_name=traj_map,
                models_by_id=self.modelsById,
                node_prefix=node_prefix,
            )

        self.lastGeneratedContacts = contacts
        self.exportBundleButton.setEnabled(True)
        self.log(
            f"[contacts:{log_context}] updated {len(contacts)} points across {len(nodes)} electrode nodes"
        )
        if model_nodes:
            self.log(f"[models:{log_context}] updated {len(model_nodes)} electrode model pairs")
        self._refresh_qc_metrics()
        if self.lastQCMetricsRows:
            self.log(f"[qc:{log_context}] computed metrics for {len(self.lastQCMetricsRows)} trajectories")

    def _assignment_map(self):
        """Return mapping `trajectory_name -> assignment_row` for selected models."""
        amap = {}
        for row in self._collect_assignments().get("assignments", []):
            amap[row["trajectory"]] = row
        return amap

    def _set_preview_line(self, trajectory_name, start_lps, end_lps):
        """Create/update a preview line showing the fitted trajectory."""
        self.logic.trajectory_scene.set_preview_line(
            trajectory_name=trajectory_name,
            start_lps=start_lps,
            end_lps=end_lps,
            node_prefix="AutoFit_",
        )

    def _remove_autofit_preview_lines(self, trajectory_names=None):
        """Remove AutoFit preview line nodes."""
        self.logic.trajectory_scene.remove_preview_lines(
            trajectory_names=trajectory_names,
            node_prefix="AutoFit_",
        )

    def _fit_trajectories(self, names):
        """Run auto-fit for selected trajectory names and store successful results."""
        if self._autofit_candidate_count() == 0:
            raise ValueError("No CT candidates available. Run 'Detect Candidates' first.")
        try:
            import importlib
            import rosa_core.contact_fit as contact_fit

            contact_fit = importlib.reload(contact_fit)
            fit_electrode_axis_and_tip = contact_fit.fit_electrode_axis_and_tip
        except Exception as exc:
            raise ValueError(f"Auto-fit dependency unavailable: {exc}")

        traj_map = self._build_trajectory_map_with_scene_overrides()
        assign_map = self._assignment_map()
        roi_radius = self._widget_value(self.autoRoiRadiusSpin)
        max_angle = self._widget_value(self.autoMaxAngleSpin)
        max_shift = self._widget_value(self.autoMaxDepthShiftSpin)

        success_count = 0
        for name in names:
            traj = traj_map.get(name)
            if traj is None:
                self.log(f"[autofit] {name}: skipped (trajectory not found)")
                continue
            assignment = assign_map.get(name)
            if assignment is None:
                self.log(f"[autofit] {name}: skipped (no electrode model assigned)")
                continue
            model = self.modelsById.get(assignment["model_id"])
            if not model:
                self.log(f"[autofit] {name}: skipped (unknown model {assignment['model_id']})")
                continue

            fit = fit_electrode_axis_and_tip(
                candidate_points_lps=self.autoFitCandidatesLPS,
                planned_entry_lps=traj["start"],
                planned_target_lps=traj["end"],
                contact_offsets_mm=model["contact_center_offsets_from_tip_mm"],
                tip_at=assignment.get("tip_at", "target"),
                roi_radius_mm=roi_radius,
                max_angle_deg=max_angle,
                max_depth_shift_mm=max_shift,
            )
            if fit.get("success"):
                self.autoFitResults[name] = fit
                success_count += 1
                self._set_preview_line(name, fit["entry_lps"], fit["target_lps"])
                self.log(
                    "[autofit] {name}: angle={a:.2f} deg depth={s:.2f} mm lateral={l:.2f} mm "
                    "residual={r:.2f} mm points={n} slabs={sl}/{si}".format(
                        name=name,
                        a=float(fit.get("angle_deg", 0.0)),
                        s=float(fit.get("tip_shift_mm", 0.0)),
                        l=float(fit.get("lateral_shift_mm", 0.0)),
                        r=float(fit.get("residual_mm", 0.0)),
                        n=int(fit.get("points_in_roi", 0)),
                        sl=int(fit.get("slab_centroids", 0)),
                        si=int(fit.get("slab_inliers", 0)),
                    )
                )
            else:
                self.log(f"[autofit] {name}: failed ({fit.get('reason', 'unknown')})")

        self.applyFitButton.setEnabled(bool(self.autoFitResults))
        self.log(f"[autofit] fitted {success_count}/{len(names)} trajectories")

    def onDetectCandidatesClicked(self):
        """Extract postop CT high-intensity candidate points for auto-fit."""
        volume_node = self.postopCTSelector.currentNode()
        if volume_node is None:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "ROSA Helper", "Select a postop CT volume.")
            return
        threshold = self._widget_value(self.autoThresholdSpin)
        try:
            points_lps = self.logic.extract_threshold_candidates_lps(volume_node, threshold=threshold)
        except Exception as exc:
            self.log(f"[autofit] detect error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "ROSA Helper", str(exc))
            return

        self._remove_autofit_preview_lines()
        self.autoFitCandidatesLPS = points_lps
        self.autoFitResults = {}
        self.autoFitCandidateVolumeNodeID = volume_node.GetID()
        count = int(points_lps.shape[0]) if hasattr(points_lps, "shape") else len(points_lps)
        self._set_autofit_buttons_enabled(count > 0)
        self.log(f"[autofit] detected {count} candidate points at threshold {threshold:.1f}")

        try:
            self.logic.show_volume_in_all_slice_views(volume_node)
            self.logic.apply_ct_window_from_threshold(volume_node, threshold=threshold)
        except Exception as exc:
            self.log(f"[autofit] display warning: {exc}")

    def onResetCTWindowClicked(self):
        """Reset selected postop CT window/level to Slicer auto preset."""
        volume_node = self.postopCTSelector.currentNode()
        if volume_node is None:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "ROSA Helper", "Select a postop CT volume.")
            return
        try:
            self.logic.reset_ct_window(volume_node)
            self.log("[autofit] CT window reset to auto")
        except Exception as exc:
            self.log(f"[autofit] CT window reset warning: {exc}")

    def onFitSelectedClicked(self):
        """Fit only the currently selected trajectory using detected candidates."""
        name = self._widget_text(self.autoFitTrajectorySelector).strip()
        if not name:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "ROSA Helper", "Select a trajectory to fit.")
            return
        try:
            self._fit_trajectories([name])
        except Exception as exc:
            self.log(f"[autofit] error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "ROSA Helper", str(exc))

    def onFitAllClicked(self):
        """Fit all trajectories that have an assigned electrode model."""
        names = [row.get("trajectory") for row in self._collect_assignments().get("assignments", [])]
        names = [n for n in names if n]
        if not names:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "ROSA Helper",
                "Assign at least one electrode model before fitting.",
            )
            return
        try:
            self._fit_trajectories(names)
        except Exception as exc:
            self.log(f"[autofit] error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "ROSA Helper", str(exc))

    def onApplyFitClicked(self):
        """Apply successful auto-fit results to trajectory line markups and refresh contacts."""
        if not self.autoFitResults:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "ROSA Helper",
                "No successful fit results to apply.",
            )
            return

        applied = 0
        for name, fit in self.autoFitResults.items():
            if not fit.get("success"):
                continue
            node = self._find_line_markup_node(name)
            if node is None or node.GetNumberOfControlPoints() < 2:
                continue
            start_ras = lps_to_ras_point(fit["entry_lps"])
            end_ras = lps_to_ras_point(fit["target_lps"])
            node.SetNthControlPointPositionWorld(0, start_ras[0], start_ras[1], start_ras[2])
            node.SetNthControlPointPositionWorld(1, end_ras[0], end_ras[1], end_ras[2])
            applied += 1

        self.log(f"[autofit] applied fitted trajectories: {applied}")
        self._remove_autofit_preview_lines()
        self.autoFitResults = {}
        self.applyFitButton.setEnabled(False)
        try:
            self._run_contact_generation(log_context="autofit", allow_last_assignments=True)
        except Exception as exc:
            self.log(f"[autofit] contact update error: {exc}")
            qt.QMessageBox.critical(
                slicer.util.mainWindow(),
                "ROSA Helper",
                f"Applied fit to trajectories, but contact update failed:\n{exc}",
            )

    def onApplyModelAllClicked(self):
        """Apply selected default electrode model to all rows."""
        model_id = self._widget_text(self.defaultModelCombo).strip()
        if not model_id:
            return
        for row in range(self.contactTable.rowCount):
            combo = self.contactTable.cellWidget(row, 2)
            if combo:
                idx = combo.findText(model_id)
                if idx >= 0:
                    combo.setCurrentIndex(idx)

    def _collect_assignments(self):
        """Collect non-empty assignments from table rows."""
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

    def onGenerateContactsClicked(self):
        """Generate contact fiducials from trajectory assignments."""
        if not self.loadedTrajectories:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "ROSA Helper", "Load a ROSA case first.")
            return

        try:
            self._run_contact_generation(log_context="generate")
        except Exception as exc:
            self.log(f"[contacts] error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "ROSA Helper", str(exc))
            return

    def onUpdateContactsClicked(self):
        """Recalculate contact and model nodes from edited trajectory markups."""
        if not self.loadedTrajectories:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "ROSA Helper", "Load a ROSA case first.")
            return
        if not self.lastGeneratedContacts:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "ROSA Helper",
                "Generate contacts once first, then use update for edited trajectories.",
            )
            return

        try:
            self._run_contact_generation(log_context="update")
        except Exception as exc:
            self.log(f"[contacts:update] error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "ROSA Helper", str(exc))
            return

    def onExportAlignedBundleClicked(self):
        """Export aligned volumes as NIfTI and contact coordinates in same frame."""
        if not self.lastGeneratedContacts:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "ROSA Helper",
                "Generate contacts first.",
            )
            return
        if not self.loadedVolumeNodeIDs:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "ROSA Helper",
                "No loaded volumes available to export.",
            )
            return

        case_dir = self.caseDirSelector.currentPath
        out_dir = self.bundleExportDirEdit.text.strip()
        if not out_dir:
            if not case_dir:
                qt.QMessageBox.warning(
                    slicer.util.mainWindow(),
                    "ROSA Helper",
                    "Case folder is not set.",
                )
                return
            out_dir = os.path.join(case_dir, "RosaHelper_Export")

        node_prefix = self.contactsNodeNameEdit.text.strip() or "ROSA_Contacts"
        self._refresh_qc_metrics()
        planned_map = self._collect_planned_trajectory_map()
        try:
            result = self.logic.export_aligned_bundle(
                volume_node_ids=self.loadedVolumeNodeIDs,
                contacts=self.lastGeneratedContacts,
                out_dir=out_dir,
                node_prefix=node_prefix,
                planned_trajectories=planned_map,
                qc_rows=self.lastQCMetricsRows,
            )
        except Exception as exc:
            self.log(f"[bundle] export warning: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "ROSA Helper", str(exc))
            return

        self.log(
            f"[bundle] exported {result['volume_count']} NIfTI volumes "
            f"and coordinate/QC files to {result['out_dir']}"
        )

    def onAlignSliceClicked(self):
        """Align selected slice view to selected trajectory."""
        traj_name = self._widget_text(self.trajectorySelector).strip()
        if not traj_name:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "ROSA Helper", "No trajectory selected.")
            return

        trajectory = self._trajectory_by_name(traj_name)
        if trajectory is None:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "ROSA Helper",
                f"Trajectory '{traj_name}' not found in loaded case.",
            )
            return

        start_ras = lps_to_ras_point(trajectory["start"])
        end_ras = lps_to_ras_point(trajectory["end"])
        slice_view = self._widget_text(self.sliceViewSelector) or "Red"
        mode = self._widget_text(self.sliceModeSelector) or "long"

        try:
            self.logic.align_slice_to_trajectory(start_ras, end_ras, slice_view=slice_view, mode=mode)
        except Exception as exc:
            self.log(f"[slice] error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "ROSA Helper", str(exc))
            return

        self.log(f"[slice] aligned {slice_view} view to {traj_name} ({mode})")

    def onShowPlannedToggled(self, checked):
        """Toggle visibility of stored planned trajectory lines."""
        self.logic.set_planned_trajectory_visibility(bool(checked))

    def _preselect_freesurfer_reference_volume(self):
        """Default FS fixed volume selector to loaded ROSA reference volume."""
        if self.referenceVolumeName and self.referenceVolumeName in self.loadedVolumeNodeIDs:
            node_id = self.loadedVolumeNodeIDs[self.referenceVolumeName]
            self.fsFixedSelector.setCurrentNodeID(node_id)
            return
        if self.loadedVolumeNodeIDs:
            first_name = sorted(self.loadedVolumeNodeIDs.keys())[0]
            self.fsFixedSelector.setCurrentNodeID(self.loadedVolumeNodeIDs[first_name])

    def _get_or_create_fs_transform_node(self):
        """Return existing or newly created linear transform for FS->ROSA mapping."""
        name = self.fsTransformNameEdit.text.strip() or "FS_to_ROSA"
        node = self.logic._find_node_by_name(name, "vtkMRMLLinearTransformNode")
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", name)
        return node

    def _selected_fs_annotation_name(self):
        """Return selected FreeSurfer annotation name or None when disabled."""
        if not bool(self.fsUseAnnotationCheck.checked):
            return None
        value = (self._widget_text(self.fsAnnotationCombo) or "").strip()
        if not value:
            return None
        if value.lower() == "custom":
            custom = self.fsCustomAnnotationEdit.text.strip()
            return custom or None
        return value

    def onRegisterFSMRIToRosaClicked(self):
        """Run linear registration from FreeSurfer MRI to selected ROSA base volume."""
        fixed_node = self.fsFixedSelector.currentNode()
        moving_node = self.fsMovingSelector.currentNode()
        if fixed_node is None or moving_node is None:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "ROSA Helper",
                "Select both ROSA base volume and FreeSurfer MRI volume.",
            )
            return
        if fixed_node.GetID() == moving_node.GetID():
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "ROSA Helper",
                "Fixed and moving volumes must be different nodes.",
            )
            return

        transform_node = self._get_or_create_fs_transform_node()
        init_mode = self._widget_text(self.fsInitModeCombo) or "useGeometryAlign"
        self.log(
            f"[fs] registration start: moving={moving_node.GetName()} -> fixed={fixed_node.GetName()} "
            f"(init={init_mode})"
        )
        try:
            self.logic.run_brainsfit_rigid_registration(
                fixed_volume_node=fixed_node,
                moving_volume_node=moving_node,
                output_transform_node=transform_node,
                initialize_mode=init_mode,
                logger=self.log,
            )
        except Exception as exc:
            self.log(f"[fs] registration error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "ROSA Helper", str(exc))
            return

        self.fsToRosaTransformNodeID = transform_node.GetID()
        self.log(f"[fs] registration done: transform={transform_node.GetName()}")

    def onLoadFSSurfacesClicked(self):
        """Load FreeSurfer surfaces and optionally apply FS->ROSA transform."""
        subject_dir = self.fsSubjectDirSelector.currentPath
        if not subject_dir:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "ROSA Helper",
                "Select FreeSurfer subject directory.",
            )
            return

        surface_set = self._widget_text(self.fsSurfaceSetCombo) or "pial"
        annotation_name = self._selected_fs_annotation_name()
        lut_path = self.fsLUTPathSelector.currentPath.strip() if self.fsLUTPathSelector.currentPath else ""
        apply_transform = bool(self.fsApplyTransformCheck.checked)
        transform_node = None
        if apply_transform:
            if self.fsToRosaTransformNodeID:
                transform_node = slicer.mrmlScene.GetNodeByID(self.fsToRosaTransformNodeID)
            if transform_node is None:
                name = self.fsTransformNameEdit.text.strip() or "FS_to_ROSA"
                transform_node = self.logic._find_node_by_name(name, "vtkMRMLLinearTransformNode")
            if transform_node is None:
                qt.QMessageBox.warning(
                    slicer.util.mainWindow(),
                    "ROSA Helper",
                    "FS->ROSA transform is not available. Run registration first or disable transform application.",
                )
                return

        try:
            result = self.logic.load_freesurfer_surfaces(
                subject_dir=subject_dir,
                surface_set=surface_set,
                annotation_name=annotation_name,
                color_lut_path=lut_path or None,
                logger=self.log,
            )
        except Exception as exc:
            self.log(f"[fs] surface load error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "ROSA Helper", str(exc))
            return
        loaded_nodes = result.get("loaded_nodes", [])
        missing_paths = result.get("missing_surface_paths", [])
        failed_paths = result.get("failed_surface_paths", [])
        missing_annotation_paths = result.get("missing_annotation_paths", [])
        annotated_count = int(result.get("annotated_nodes", 0))
        color_node_name = result.get("color_node_name", "")

        if apply_transform and transform_node is not None and loaded_nodes:
            self.logic.apply_transform_to_model_nodes(
                model_nodes=loaded_nodes,
                transform_node=transform_node,
                harden=bool(self.fsHardenSurfaceCheck.checked),
            )
            self.log(
                f"[fs] applied transform {transform_node.GetName()} to {len(loaded_nodes)} surfaces"
            )

        if missing_paths:
            self.log(f"[fs] missing surface files: {len(missing_paths)}")
            for path in missing_paths:
                self.log(f"[fs] missing: {path}")
        if failed_paths:
            self.log(f"[fs] failed to load surfaces: {len(failed_paths)}")
            for path in failed_paths:
                self.log(f"[fs] failed: {path}")
        if missing_annotation_paths:
            self.log(f"[fs] missing annotation files: {len(missing_annotation_paths)}")
            for path in missing_annotation_paths:
                self.log(f"[fs] missing annot: {path}")
        if annotation_name:
            self.log(
                f"[fs] annotation mode '{annotation_name}' applied to {annotated_count}/{len(loaded_nodes)} loaded surfaces"
            )
            if color_node_name:
                self.log(f"[fs] annotation color node: {color_node_name}")

        self.log(f"[fs] loaded {len(loaded_nodes)} surface models")
