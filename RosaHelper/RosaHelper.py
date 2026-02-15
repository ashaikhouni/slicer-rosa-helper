"""3D Slicer scripted module entrypoint for ROSA Helper.

This UI-oriented layer delegates parsing and transform composition to `rosa_core`.
It focuses on scene operations:
- load Analyze volumes
- center volumes
- apply composed display transforms
- create trajectory line markups
"""

import os
import sys
import math
try:
    import numpy as np
except ImportError:
    np = None

from __main__ import ctk, qt, slicer, vtk
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
)

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.join(MODULE_DIR, "Lib")
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

from rosa_core import (
    build_effective_matrices,
    choose_reference_volume,
    compute_qc_metrics,
    electrode_length_mm,
    find_ros_file,
    generate_contacts,
    invert_4x4,
    lps_to_ras_matrix,
    lps_to_ras_point,
    load_electrode_library,
    model_map,
    parse_ros_file,
    resolve_analyze_volume,
    resolve_reference_index,
    suggest_model_id_for_trajectory,
    trajectory_length_mm,
)
from rosa_slicer.freesurfer_service import FreeSurferService
from rosa_slicer.trajectory_scene import TrajectorySceneService


class RosaHelper(ScriptedLoadableModule):
    """Slicer module metadata container."""

    def __init__(self, parent):
        """Initialize static module metadata shown in Slicer UI."""
        super().__init__(parent)
        self.parent.title = "ROSA Helper"
        self.parent.categories = ["ROSA"]
        self.parent.dependencies = []
        self.parent.contributors = ["Ammar Shaikhouni", "Codex"]
        self.parent.helpText = "Load a ROSA case folder into Slicer and apply ROSA transforms."


class RosaHelperWidget(ScriptedLoadableModuleWidget):
    """Qt widget for selecting a case folder and loading it into the scene."""

    def setup(self):
        """Create module UI controls and wire actions."""
        super().setup()

        self.logic = RosaHelperLogic()
        self.loadedTrajectories = []
        self.lastGeneratedContacts = []
        self.lastAssignments = {"schema_version": "1.0", "assignments": []}
        self.lastQCMetricsRows = []
        self.loadedVolumeNodeIDs = {}
        self.referenceVolumeName = None
        self.autoFitCandidatesLPS = []
        self.autoFitResults = {}
        self.autoFitCandidateVolumeNodeID = None
        self.fsToRosaTransformNodeID = None
        self.modelsById = {}
        self.modelIds = []

        form = qt.QFormLayout()
        self.layout.addLayout(form)

        self.caseDirSelector = ctk.ctkPathLineEdit()
        self.caseDirSelector.filters = ctk.ctkPathLineEdit.Dirs
        self.caseDirSelector.setToolTip("Case folder containing .ros and DICOM/")
        form.addRow("Case folder", self.caseDirSelector)

        self.referenceEdit = qt.QLineEdit()
        self.referenceEdit.setPlaceholderText("Optional (auto-detect if blank)")
        self.referenceEdit.setToolTip(
            "Root display volume name. If blank, the first ROS display is used."
        )
        form.addRow("Reference volume", self.referenceEdit)

        self.invertCheck = qt.QCheckBox("Invert TRdicomRdisplay")
        self.invertCheck.setChecked(False)
        self.invertCheck.setToolTip(
            "Invert the composed transform before applying. Use only for datasets"
            " where ROS matrices are known to be reversed."
        )
        form.addRow("Transform option", self.invertCheck)

        self.hardenCheck = qt.QCheckBox("Harden transforms")
        self.hardenCheck.setChecked(True)
        form.addRow("Scene option", self.hardenCheck)

        self.markupsCheck = qt.QCheckBox("Load trajectories")
        self.markupsCheck.setChecked(True)
        form.addRow("Trajectory option", self.markupsCheck)

        self.loadButton = qt.QPushButton("Load ROSA case")
        self.loadButton.clicked.connect(self.onLoadClicked)
        self.layout.addWidget(self.loadButton)

        self.statusText = qt.QPlainTextEdit()
        self.statusText.setReadOnly(True)
        self.statusText.setMaximumBlockCount(1000)
        self.layout.addWidget(self.statusText)

        self._build_contact_ui()
        self._build_freesurfer_ui()
        self._build_qc_ui()
        self._build_trajectory_view_ui()
        self._build_autofit_ui()
        self._load_electrode_library()

        self.layout.addStretch(1)

    def log(self, msg):
        """Append status text to the module log panel and stdout."""
        self.statusText.appendPlainText(msg)
        print(msg)

    def onLoadClicked(self):
        """Validate inputs and run the load pipeline."""
        case_dir = self.caseDirSelector.currentPath
        if not case_dir:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "ROSA Helper", "Please select a case folder")
            return

        reference = self.referenceEdit.text.strip() or None

        try:
            summary = self.logic.load_case(
                case_dir=case_dir,
                reference=reference,
                invert=self.invertCheck.checked,
                harden=self.hardenCheck.checked,
                load_trajectories=self.markupsCheck.checked,
                show_planned=self.showPlannedCheck.checked,
                logger=self.log,
            )
        except Exception as exc:
            self.log(f"[error] {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "ROSA Helper", str(exc))
            return

        self.loadedTrajectories = summary["trajectories"]
        self.lastGeneratedContacts = []
        self.lastAssignments = {"schema_version": "1.0", "assignments": []}
        self.lastQCMetricsRows = []
        self.loadedVolumeNodeIDs = summary.get("loaded_volume_node_ids", {})
        self.referenceVolumeName = summary.get("reference_volume")
        self.autoFitCandidatesLPS = []
        self.autoFitResults = {}
        self.autoFitCandidateVolumeNodeID = None
        self.fsToRosaTransformNodeID = None
        self._populate_contact_table(self.loadedTrajectories)
        self._populate_trajectory_selector(self.loadedTrajectories)
        self._populate_autofit_trajectory_selector(self.loadedTrajectories)
        self._refresh_qc_metrics()
        self._set_autofit_buttons_enabled(False)
        self._preselect_freesurfer_reference_volume()
        self.log(
            f"[done] loaded {summary['loaded_volumes']} volumes, "
            f"created {summary['trajectory_count']} trajectories"
        )

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


class RosaHelperLogic(ScriptedLoadableModuleLogic):
    """Core scene-loading logic used by UI and headless `run()` entrypoint."""

    def __init__(self):
        """Initialize service delegates used by Slicer-specific workflows."""
        super().__init__()
        self.fs_service = FreeSurferService(module_dir=MODULE_DIR)
        self.trajectory_scene = TrajectorySceneService()

    def load_case(
        self,
        case_dir,
        reference=None,
        invert=False,
        harden=True,
        load_trajectories=True,
        show_planned=False,
        logger=None,
    ):
        """Load a ROSA case directory into the current Slicer scene.

        Parameters
        ----------
        case_dir: str
            Folder containing one `.ros` file and a `DICOM` subfolder.
        reference: str | None
            Optional root volume name used for chain composition.
        invert: bool
            Invert composed transforms before applying.
        harden: bool
            Harden applied transforms into volume geometry.
        load_trajectories: bool
            Create line markups from ROS trajectories.
        show_planned: bool
            Whether planned trajectory backup lines are visible after load.
        logger: callable | None
            Optional callback used for status messages.
        """

        def log(msg):
            """Forward log messages to callback when provided."""
            if logger:
                logger(msg)
            else:
                print(msg)

        case_dir = os.path.abspath(case_dir)
        ros_path = find_ros_file(case_dir)
        analyze_root = os.path.join(case_dir, "DICOM")

        if not os.path.isdir(analyze_root):
            raise ValueError(f"Analyze root not found: {analyze_root}")

        parsed = parse_ros_file(ros_path)
        displays = parsed["displays"]
        trajectories = parsed["trajectories"]

        if not displays:
            raise ValueError("No TRdicomRdisplay/VOLUME entries found in ROS file")

        reference_volume = choose_reference_volume(displays, preferred=reference)
        root_index = resolve_reference_index(displays, reference_volume)
        effective_lps = build_effective_matrices(displays, root_index=root_index)
        if invert:
            effective_used_lps = [invert_4x4(m) for m in effective_lps]
        else:
            effective_used_lps = effective_lps
        log(f"[ros] {ros_path}")
        log(f"[ref] {reference_volume}")

        loaded_count = 0
        loaded_volume_node_ids = {}

        for i, disp in enumerate(displays):
            vol_name = disp["volume"]
            img_path = resolve_analyze_volume(analyze_root, disp)
            if not img_path:
                log(f"[skip] missing Analyze .img for {vol_name}")
                continue

            vol_node = self._load_volume(img_path)
            if vol_node is None:
                log(f"[skip] failed to load {img_path}")
                continue

            loaded_count += 1
            loaded_volume_node_ids[vol_name] = vol_node.GetID()
            vol_node.SetName(vol_name)
            self._center_volume(vol_node)
            log(f"[load] {vol_name}")
            log(f"[center] {vol_name}")

            if vol_name != reference_volume:
                matrix_ras = lps_to_ras_matrix(effective_used_lps[i])
                tnode = self._apply_transform(vol_node, matrix_ras)
                ref_idx = disp.get("imagery_3dref", root_index)
                log(
                    f"[xform] {vol_name} {'inv ' if invert else ''}TRdicomRdisplay "
                    f"(ref idx {ref_idx} -> root idx {root_index})"
                )
                if harden:
                    slicer.vtkSlicerTransformLogic().hardenTransform(vol_node)
                    slicer.mrmlScene.RemoveNode(tnode)
                    log(f"[harden] {vol_name}")
            else:
                log(f"[xform] {vol_name} reference (none)")

        if load_trajectories and trajectories:
            self._add_trajectories(trajectories, logger=log, show_planned=show_planned)

        return {
            "loaded_volumes": loaded_count,
            "loaded_volume_node_ids": loaded_volume_node_ids,
            "reference_volume": reference_volume,
            "trajectory_count": len(trajectories) if load_trajectories else 0,
            "trajectories": trajectories,
        }

    def _load_volume(self, path):
        """Load a scalar volume by path and return the MRML node."""
        try:
            result = slicer.util.loadVolume(path, returnNode=True)
            if isinstance(result, tuple):
                ok, node = result
                return node if ok else None
            return result
        except TypeError:
            return slicer.util.loadVolume(path)

    def _center_volume(self, volume_node):
        """Center volume origin in Slicer (equivalent to Volumes->Center Volume)."""
        logic = slicer.modules.volumes.logic()
        if logic and hasattr(logic, "CenterVolume"):
            logic.CenterVolume(volume_node)
            return

        ijk_to_ras = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(ijk_to_ras)
        dims = volume_node.GetImageData().GetDimensions()
        center_ijk = [(dims[0] - 1) / 2.0, (dims[1] - 1) / 2.0, (dims[2] - 1) / 2.0, 1.0]
        center_ras = [0.0, 0.0, 0.0, 0.0]
        ijk_to_ras.MultiplyPoint(center_ijk, center_ras)
        for i in range(3):
            ijk_to_ras.SetElement(i, 3, ijk_to_ras.GetElement(i, 3) - center_ras[i])
        volume_node.SetIJKToRASMatrix(ijk_to_ras)

    def _apply_transform(self, volume_node, matrix4x4):
        """Create and assign a linear transform node from a 4x4 matrix."""
        tnode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode")
        vtk_mat = vtk.vtkMatrix4x4()
        for r in range(4):
            for c in range(4):
                vtk_mat.SetElement(r, c, matrix4x4[r][c])
        tnode.SetMatrixTransformToParent(vtk_mat)
        volume_node.SetAndObserveTransformNodeID(tnode.GetID())
        return tnode

    def _vtk_matrix_to_numpy(self, vtk_matrix4x4):
        """Convert vtkMatrix4x4 to a NumPy 4x4 array."""
        out = np.eye(4, dtype=float)
        for r in range(4):
            for c in range(4):
                out[r, c] = float(vtk_matrix4x4.GetElement(r, c))
        return out

    def extract_threshold_candidates_lps(self, volume_node, threshold, max_points=300000):
        """Extract thresholded CT candidate points in LPS coordinates."""
        if np is None:
            raise RuntimeError("NumPy is required for auto-fit candidate extraction.")
        arr = slicer.util.arrayFromVolume(volume_node)  # K, J, I
        idx = np.argwhere(arr >= float(threshold))
        if idx.size == 0:
            return np.empty((0, 3), dtype=float)

        if idx.shape[0] > int(max_points):
            rng = np.random.default_rng(0)
            keep = rng.choice(idx.shape[0], size=int(max_points), replace=False)
            idx = idx[keep]

        # Convert argwhere KJI indices to IJK homogeneous coordinates.
        n = idx.shape[0]
        ijk_h = np.ones((n, 4), dtype=float)
        ijk_h[:, 0] = idx[:, 2].astype(float)  # I
        ijk_h[:, 1] = idx[:, 1].astype(float)  # J
        ijk_h[:, 2] = idx[:, 0].astype(float)  # K

        ijk_to_ras_vtk = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(ijk_to_ras_vtk)
        ijk_to_ras = self._vtk_matrix_to_numpy(ijk_to_ras_vtk)
        ras_h = ijk_h @ ijk_to_ras.T
        ras = ras_h[:, :3]
        lps = ras.copy()
        lps[:, 0] *= -1.0
        lps[:, 1] *= -1.0
        return lps

    def show_volume_in_all_slice_views(self, volume_node):
        """Set provided volume as background in Red/Yellow/Green slice views."""
        if volume_node is None:
            return
        volume_id = volume_node.GetID()
        for composite in slicer.util.getNodesByClass("vtkMRMLSliceCompositeNode"):
            composite.SetBackgroundVolumeID(volume_id)

    def apply_ct_window_from_threshold(self, volume_node, threshold):
        """Apply CT window/level preset centered around the current detection threshold."""
        if volume_node is None:
            return
        display = volume_node.GetDisplayNode()
        if display is None:
            return
        lower = float(threshold) - 250.0
        upper = float(threshold) + 2200.0
        display.AutoWindowLevelOff()
        display.SetWindow(max(upper - lower, 1.0))
        display.SetLevel((upper + lower) * 0.5)

    def reset_ct_window(self, volume_node):
        """Reset CT window/level to Slicer auto mode for selected volume."""
        if volume_node is None:
            return
        display = volume_node.GetDisplayNode()
        if display is None:
            return
        display.AutoWindowLevelOn()

    def run_brainsfit_rigid_registration(
        self,
        fixed_volume_node,
        moving_volume_node,
        output_transform_node,
        initialize_mode="useGeometryAlign",
        logger=None,
    ):
        """Delegate FreeSurfer MRI->ROSA rigid registration to service layer."""
        return self.fs_service.run_brainsfit_rigid_registration(
            fixed_volume_node=fixed_volume_node,
            moving_volume_node=moving_volume_node,
            output_transform_node=output_transform_node,
            initialize_mode=initialize_mode,
            logger=logger,
        )

    def load_freesurfer_surfaces(
        self,
        subject_dir,
        surface_set="pial",
        annotation_name=None,
        color_lut_path=None,
        logger=None,
    ):
        """Delegate FreeSurfer surface and annotation loading to service layer."""
        return self.fs_service.load_freesurfer_surfaces(
            subject_dir=subject_dir,
            surface_set=surface_set,
            annotation_name=annotation_name,
            color_lut_path=color_lut_path,
            logger=logger,
        )

    def apply_transform_to_model_nodes(self, model_nodes, transform_node, harden=False):
        """Delegate model transform application/hardening to service layer."""
        return self.fs_service.apply_transform_to_model_nodes(
            model_nodes=model_nodes,
            transform_node=transform_node,
            harden=harden,
        )

    def _add_trajectories(self, trajectories, logger=None, show_planned=False):
        """Create editable trajectory lines and hidden planned backups."""
        for traj in trajectories:
            start_ras = lps_to_ras_point(traj["start"])
            end_ras = lps_to_ras_point(traj["end"])
            # Editable working trajectory.
            node = self._find_node_by_name(traj["name"], "vtkMRMLMarkupsLineNode")
            if node is None:
                node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", traj["name"])
            node.RemoveAllControlPoints()
            node.AddControlPoint(vtk.vtkVector3d(*start_ras))
            node.AddControlPoint(vtk.vtkVector3d(*end_ras))
            node.SetNthControlPointLabel(0, f"{traj['name']}_start")
            node.SetNthControlPointLabel(1, f"{traj['name']}_end")

            # Immutable planned backup trajectory.
            plan_name = f"Plan_{traj['name']}"
            plan_node = self._find_node_by_name(plan_name, "vtkMRMLMarkupsLineNode")
            if plan_node is None:
                plan_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", plan_name)
            plan_node.RemoveAllControlPoints()
            plan_node.AddControlPoint(vtk.vtkVector3d(*start_ras))
            plan_node.AddControlPoint(vtk.vtkVector3d(*end_ras))
            plan_node.SetNthControlPointLabel(0, f"{traj['name']}_plan_start")
            plan_node.SetNthControlPointLabel(1, f"{traj['name']}_plan_end")
            plan_node.SetLocked(True)
            plan_display = plan_node.GetDisplayNode()
            if plan_display:
                plan_display.SetVisibility(bool(show_planned))
                plan_display.SetColor(0.65, 0.65, 0.65)
                plan_display.SetSelectedColor(0.65, 0.65, 0.65)
                plan_display.SetLineThickness(0.35)
                if hasattr(plan_display, "SetPointLabelsVisibility"):
                    plan_display.SetPointLabelsVisibility(False)

        if logger:
            logger(f"[markups] created {len(trajectories)} line trajectories")

    def set_planned_trajectory_visibility(self, visible):
        """Show or hide all planned backup trajectory lines."""
        for node in slicer.util.getNodesByClass("vtkMRMLMarkupsLineNode"):
            name = node.GetName() or ""
            if not name.startswith("Plan_"):
                continue
            display = node.GetDisplayNode()
            if display:
                display.SetVisibility(bool(visible))

    def create_contacts_fiducials_node(self, contacts, node_name="ROSA_Contacts"):
        """Create a fiducial markups node from contact list in ROSA/LPS space."""
        node = self._find_node_by_name(node_name, "vtkMRMLMarkupsFiducialNode")
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", node_name)
        else:
            node.RemoveAllControlPoints()
        for contact in contacts:
            ras = lps_to_ras_point(contact["position_lps"])
            point_index = node.AddControlPoint(vtk.vtkVector3d(*ras))
            # Keep labels compact in-view: show contact index only.
            contact_index = contact.get("index", point_index + 1)
            node.SetNthControlPointLabel(point_index, str(contact_index))

        display_node = node.GetDisplayNode()
        if display_node:
            display_node.SetGlyphScale(2.00)
            display_node.SetTextScale(1.50)
        return node

    def create_contacts_fiducials_nodes_by_trajectory(self, contacts, node_prefix="ROSA_Contacts"):
        """Create one fiducial node per trajectory so visibility can be toggled independently."""
        by_traj = {}
        for contact in contacts:
            traj = contact.get("trajectory", "unknown")
            by_traj.setdefault(traj, []).append(contact)

        nodes = {}
        for traj_name in sorted(by_traj.keys()):
            node_name = f"{node_prefix}_{traj_name}"
            nodes[traj_name] = self.create_contacts_fiducials_node(
                by_traj[traj_name],
                node_name=node_name,
            )
        return nodes

    def _find_node_by_name(self, node_name, class_name):
        """Return first node with exact name and class, or None."""
        for node in slicer.util.getNodesByClass(class_name):
            if node.GetName() == node_name:
                return node
        return None

    def _tube_polydata(self, p0, p1, radius_mm, sides=24):
        """Build capped tube polydata between two 3D points."""
        line = vtk.vtkLineSource()
        line.SetPoint1(float(p0[0]), float(p0[1]), float(p0[2]))
        line.SetPoint2(float(p1[0]), float(p1[1]), float(p1[2]))

        tube = vtk.vtkTubeFilter()
        tube.SetInputConnection(line.GetOutputPort())
        tube.SetRadius(float(radius_mm))
        tube.SetNumberOfSides(int(sides))
        tube.CappingOn()
        tube.Update()

        out = vtk.vtkPolyData()
        out.DeepCopy(tube.GetOutput())
        return out

    def create_electrode_models_by_trajectory(
        self,
        contacts,
        trajectories_by_name,
        models_by_id,
        node_prefix="ROSA_Contacts",
    ):
        """Create per-trajectory model nodes for electrode shaft and contact segments."""
        by_traj = {}
        for contact in contacts:
            traj = contact.get("trajectory", "")
            by_traj.setdefault(traj, []).append(contact)

        created = {}
        for traj_name in sorted(by_traj.keys()):
            group = sorted(by_traj[traj_name], key=lambda c: int(c.get("index", 0)))
            if not group:
                continue

            model_id = group[0].get("model_id", "")
            if model_id not in models_by_id:
                continue
            model = models_by_id[model_id]
            offsets = list(model.get("contact_center_offsets_from_tip_mm", []))
            if not offsets:
                continue

            p_first = list(group[0]["position_lps"])
            if len(group) >= 2:
                p_last = list(group[-1]["position_lps"])
                axis = self._vunit(self._vsub(p_last, p_first))
            else:
                trajectory = trajectories_by_name.get(traj_name)
                if trajectory is None:
                    continue
                tip_at = (group[0].get("tip_at") or "target").lower()
                if tip_at == "entry":
                    axis = self._vunit(self._vsub(trajectory["end"], trajectory["start"]))
                else:
                    axis = self._vunit(self._vsub(trajectory["start"], trajectory["end"]))

            tip = self._vsub(p_first, self._vmul(axis, float(offsets[0])))
            shaft_len = float(model.get("total_exploration_length_mm", 0.0))
            shaft_end = self._vadd(tip, self._vmul(axis, shaft_len))
            radius = float(model.get("diameter_mm", 0.8)) / 2.0
            contact_len = float(model.get("contact_length_mm", 2.0))

            shaft_poly = self._tube_polydata(
                lps_to_ras_point(tip),
                lps_to_ras_point(shaft_end),
                radius_mm=radius,
                sides=24,
            )

            append = vtk.vtkAppendPolyData()
            for contact in group:
                center = contact["position_lps"]
                p0 = self._vsub(center, self._vmul(axis, contact_len / 2.0))
                p1 = self._vadd(center, self._vmul(axis, contact_len / 2.0))
                segment = self._tube_polydata(
                    lps_to_ras_point(p0),
                    lps_to_ras_point(p1),
                    radius_mm=radius,
                    sides=24,
                )
                append.AddInputData(segment)
            append.Update()
            contact_poly = vtk.vtkPolyData()
            contact_poly.DeepCopy(append.GetOutput())

            shaft_name = f"{node_prefix}_{traj_name}_shaft"
            contacts_name = f"{node_prefix}_{traj_name}_contacts"

            shaft_node = self._find_node_by_name(shaft_name, "vtkMRMLModelNode")
            if shaft_node is None:
                shaft_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", shaft_name)
            shaft_node.SetAndObservePolyData(shaft_poly)
            shaft_node.CreateDefaultDisplayNodes()
            shaft_display = shaft_node.GetDisplayNode()
            if shaft_display:
                shaft_display.SetColor(0.80, 0.80, 0.80)
                shaft_display.SetOpacity(0.40)
                if hasattr(shaft_display, "SetLineWidth"):
                    shaft_display.SetLineWidth(7)
                if hasattr(shaft_display, "SetSliceIntersectionThickness"):
                    shaft_display.SetSliceIntersectionThickness(7)
                if hasattr(shaft_display, "SetVisibility2D"):
                    shaft_display.SetVisibility2D(True)
                elif hasattr(shaft_display, "SetSliceIntersectionVisibility"):
                    shaft_display.SetSliceIntersectionVisibility(True)

            contacts_node = self._find_node_by_name(contacts_name, "vtkMRMLModelNode")
            if contacts_node is None:
                contacts_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", contacts_name)
            contacts_node.SetAndObservePolyData(contact_poly)
            contacts_node.CreateDefaultDisplayNodes()
            contacts_display = contacts_node.GetDisplayNode()
            if contacts_display:
                contacts_display.SetColor(1.00, 0.95, 0.20)
                contacts_display.SetOpacity(1.00)
                if hasattr(contacts_display, "SetLineWidth"):
                    contacts_display.SetLineWidth(7)
                if hasattr(contacts_display, "SetSliceIntersectionThickness"):
                    contacts_display.SetSliceIntersectionThickness(7)
                if hasattr(contacts_display, "SetVisibility2D"):
                    contacts_display.SetVisibility2D(True)
                elif hasattr(contacts_display, "SetSliceIntersectionVisibility"):
                    contacts_display.SetSliceIntersectionVisibility(True)

            created[traj_name] = {"shaft": shaft_node, "contacts": contacts_node}

        return created

    def _safe_filename(self, text):
        """Return filesystem-safe filename stem."""
        safe = []
        for ch in str(text):
            if ch.isalnum() or ch in ("-", "_", "."):
                safe.append(ch)
            else:
                safe.append("_")
        stem = "".join(safe).strip("._")
        return stem or "volume"

    def export_aligned_bundle(
        self,
        volume_node_ids,
        contacts,
        out_dir,
        node_prefix="ROSA_Contacts",
        planned_trajectories=None,
        qc_rows=None,
    ):
        """Export aligned scene volumes, coordinates, planned trajectories, and QC metrics."""
        os.makedirs(out_dir, exist_ok=True)

        saved_paths = []
        for volume_name in sorted(volume_node_ids.keys()):
            node_id = volume_node_ids[volume_name]
            node = slicer.mrmlScene.GetNodeByID(node_id)
            if node is None:
                continue
            filename = f"{self._safe_filename(volume_name)}.nii.gz"
            out_path = os.path.join(out_dir, filename)
            ok = slicer.util.saveNode(node, out_path)
            if not ok:
                raise RuntimeError(f"Failed to save volume '{volume_name}' to {out_path}")
            saved_paths.append(out_path)

        coord_path = os.path.join(out_dir, f"{node_prefix}_aligned_world_coords.txt")
        lines = []
        lines.append("# ROSA Helper aligned export")
        lines.append("# coordinate_system: SLICER_WORLD_RAS (x_ras,y_ras,z_ras)")
        lines.append("# alternate_columns: LPS (x_lps,y_lps,z_lps)")
        lines.append("# columns: trajectory,label,index,x_ras,y_ras,z_ras,x_lps,y_lps,z_lps,model_id")

        def _sort_key(c):
            """Sort contacts by trajectory then by ascending index."""
            return (str(c.get("trajectory", "")), int(c.get("index", 0)))

        for contact in sorted(contacts, key=_sort_key):
            p_lps = contact["position_lps"]
            p_ras = lps_to_ras_point(p_lps)
            lines.append(
                "{traj},{label},{idx},{x_ras:.6f},{y_ras:.6f},{z_ras:.6f},"
                "{x_lps:.6f},{y_lps:.6f},{z_lps:.6f},{model}".format(
                    traj=contact.get("trajectory", ""),
                    label=contact.get("label", ""),
                    idx=int(contact.get("index", 0)),
                    x_ras=float(p_ras[0]),
                    y_ras=float(p_ras[1]),
                    z_ras=float(p_ras[2]),
                    x_lps=float(p_lps[0]),
                    y_lps=float(p_lps[1]),
                    z_lps=float(p_lps[2]),
                    model=contact.get("model_id", ""),
                )
            )
        with open(coord_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        planned_path = os.path.join(out_dir, f"{node_prefix}_planned_trajectory_points.csv")
        planned_rows = []
        planned_map = planned_trajectories or {}
        for traj_name in sorted(planned_map.keys()):
            traj = planned_map[traj_name]
            for point_type, p_lps in (("entry", traj["start"]), ("target", traj["end"])):
                p_ras = lps_to_ras_point(p_lps)
                planned_rows.append(
                    "{traj},{ptype},{x_ras:.6f},{y_ras:.6f},{z_ras:.6f},{x_lps:.6f},{y_lps:.6f},{z_lps:.6f}".format(
                        traj=str(traj_name),
                        ptype=point_type,
                        x_ras=float(p_ras[0]),
                        y_ras=float(p_ras[1]),
                        z_ras=float(p_ras[2]),
                        x_lps=float(p_lps[0]),
                        y_lps=float(p_lps[1]),
                        z_lps=float(p_lps[2]),
                    )
                )
        with open(planned_path, "w", encoding="utf-8") as f:
            f.write("trajectory,point_type,x_ras,y_ras,z_ras,x_lps,y_lps,z_lps\n")
            if planned_rows:
                f.write("\n".join(planned_rows) + "\n")

        qc_path = os.path.join(out_dir, f"{node_prefix}_qc_metrics.csv")
        with open(qc_path, "w", encoding="utf-8") as f:
            f.write(
                "trajectory,entry_radial_mm,target_radial_mm,mean_contact_radial_mm,"
                "max_contact_radial_mm,rms_contact_radial_mm,angle_deg,matched_contacts\n"
            )
            for row in qc_rows or []:
                f.write(
                    "{trajectory},{entry:.6f},{target:.6f},{mean:.6f},{maxv:.6f},{rms:.6f},{angle:.6f},{matched}\n".format(
                        trajectory=str(row.get("trajectory", "")),
                        entry=float(row.get("entry_radial_mm", 0.0)),
                        target=float(row.get("target_radial_mm", 0.0)),
                        mean=float(row.get("mean_contact_radial_mm", 0.0)),
                        maxv=float(row.get("max_contact_radial_mm", 0.0)),
                        rms=float(row.get("rms_contact_radial_mm", 0.0)),
                        angle=float(row.get("angle_deg", 0.0)),
                        matched=int(row.get("matched_contacts", 0)),
                    )
                )

        return {
            "out_dir": out_dir,
            "volume_count": len(saved_paths),
            "volume_paths": saved_paths,
            "coordinates_path": coord_path,
            "planned_trajectories_path": planned_path,
            "qc_metrics_path": qc_path,
        }

    def _vsub(self, a, b):
        """Return vector subtraction `a - b`."""
        return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]

    def _vadd(self, a, b):
        """Return vector addition `a + b`."""
        return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]

    def _vmul(self, a, s):
        """Return scalar multiplication `a * s`."""
        return [a[0] * s, a[1] * s, a[2] * s]

    def _vdot(self, a, b):
        """Return dot product between two 3D vectors."""
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def _vcross(self, a, b):
        """Return cross product `a x b` for 3D vectors."""
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]

    def _vnorm(self, a):
        """Return Euclidean norm of a 3D vector."""
        return math.sqrt(self._vdot(a, a))

    def _vunit(self, a):
        """Return normalized vector and validate non-zero length."""
        n = self._vnorm(a)
        if n <= 1e-9:
            raise ValueError("Zero-length trajectory vector")
        return [a[0] / n, a[1] / n, a[2] / n]

    def align_slice_to_trajectory(self, start_ras, end_ras, slice_view="Red", mode="long"):
        """Align a slice node to a trajectory using two RAS points."""
        direction = self._vunit(self._vsub(end_ras, start_ras))
        center = self._vmul(self._vadd(start_ras, end_ras), 0.5)

        up = [0.0, 0.0, 1.0]
        if abs(self._vdot(direction, up)) > 0.9:
            up = [0.0, 1.0, 0.0]

        x_axis = self._vunit(self._vcross(up, direction))
        y_axis = self._vunit(self._vcross(direction, x_axis))

        mode = (mode or "long").lower()
        if mode == "down":
            normal = direction
            transverse = y_axis
        else:
            normal = x_axis
            transverse = direction

        lm = slicer.app.layoutManager()
        if lm is None:
            raise RuntimeError("Slicer layout manager is not available")
        slice_widget = lm.sliceWidget(slice_view)
        if slice_widget is None:
            raise ValueError(f"Unknown slice view '{slice_view}'")
        slice_node = slice_widget.mrmlSliceNode()
        if slice_node is None:
            raise RuntimeError(f"Slice node not found for view '{slice_view}'")

        slice_node.SetSliceToRASByNTP(
            normal[0],
            normal[1],
            normal[2],
            transverse[0],
            transverse[1],
            transverse[2],
            center[0],
            center[1],
            center[2],
            0,
        )
        # Force slice plane to pass through trajectory center, then pan to keep it visible.
        if hasattr(slice_node, "JumpSliceByOffsetting"):
            slice_node.JumpSliceByOffsetting(center[0], center[1], center[2])
        if hasattr(slice_node, "JumpSliceByCentering"):
            slice_node.JumpSliceByCentering(center[0], center[1], center[2])


def run(case_dir, reference=None, invert=False, harden=True, load_trajectories=True):
    """Headless convenience entrypoint for scripted smoke tests."""
    return RosaHelperLogic().load_case(
        case_dir=case_dir,
        reference=reference,
        invert=invert,
        harden=harden,
        load_trajectories=load_trajectories,
    )
