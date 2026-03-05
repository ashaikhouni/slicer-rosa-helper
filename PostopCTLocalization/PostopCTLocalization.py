"""3D Slicer module for postop CT localization workflows.

Modes:
- Guided Fit: refine planned trajectories on postop CT.
- De Novo Detect: detect trajectories directly from CT artifact.

Last updated: 2026-03-01
"""

import os
import sys
import json

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
PATH_CANDIDATES = [
    os.path.join(os.path.dirname(MODULE_DIR), "CommonLib"),
    os.path.join(MODULE_DIR, "CommonLib"),
]
for path in PATH_CANDIDATES:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

from rosa_core import (
    fit_electrode_axis_and_tip,
    load_electrode_library,
    lps_to_ras_point,
    model_map,
    suggest_model_id_for_trajectory,
    trajectory_length_mm,
)
from rosa_scene import ElectrodeSceneService, LayoutService, TrajectoryFocusController, TrajectorySceneService
from shank_engine import PipelineRegistry, register_builtin_pipelines
from rosa_workflow import WorkflowPublisher, WorkflowState
from rosa_workflow.workflow_registry import table_to_dict_rows

GUIDED_SOURCE_OPTIONS = [
    ("working", "Working (active)"),
    ("imported_rosa", "Imported ROSA"),
    ("imported_external", "Imported External"),
    ("manual", "Manual (scene)"),
    ("guided_fit", "Guided Fit"),
    ("de_novo", "De Novo"),
    ("planned_rosa", "Planned ROSA"),
]


class PostopCTLocalization(ScriptedLoadableModule):
    """Slicer metadata for unified postop CT localization module."""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "01 Postop CT Localization"
        self.parent.categories = ["ROSA.02 Localization"]
        self.parent.dependencies = []
        self.parent.contributors = ["Ammar Shaikhouni", "Codex"]
        self.parent.helpText = (
            "Unified postop CT localization: Guided Fit (planned trajectories) and "
            "De Novo Detect (CT-only)."
        )


class PostopCTLocalizationWidget(ScriptedLoadableModuleWidget):
    """Widget exposing guided-fit and de-novo CT localization modes."""

    def setup(self):
        super().setup()
        self.logic = PostopCTLocalizationLogic()
        self.workflowState = self.logic.workflow_state
        self.workflowPublisher = self.logic.workflow_publish
        self.workflowNode = self.workflowState.resolve_or_create_workflow_node()

        self.modelsById = {}
        self.modelIds = []
        self.loadedTrajectories = []
        self.assignmentMap = {}
        self.candidatesLPS = np.empty((0, 3), dtype=float) if np is not None else []
        self.fitResults = {}
        self._syncingGuidedSourceCombo = False
        self._syncingPipelineCombo = False
        self._pendingGuidedFollow = False
        self._updatingGuidedTable = False
        self._renamingGuidedTrajectory = False

        form = qt.QFormLayout()
        self.layout.addLayout(form)

        self.ctSelector = slicer.qMRMLNodeComboBox()
        self.ctSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.ctSelector.noneEnabled = True
        self.ctSelector.addEnabled = False
        self.ctSelector.removeEnabled = False
        self.ctSelector.setMRMLScene(slicer.mrmlScene)
        self.ctSelector.setToolTip("Postop CT used for guided fit / de novo detection.")
        self.ctSelector.currentNodeChanged.connect(lambda _node: self._apply_primary_slice_layers())
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
        self.layout.addWidget(self.modeTabs)
        self._build_guided_fit_tab()
        self._build_de_novo_tab()
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

    def _build_guided_fit_tab(self):
        tab = qt.QWidget()
        self.modeTabs.addTab(tab, "Guided Fit")
        form = qt.QFormLayout(tab)

        self.guidedThresholdSpin = qt.QDoubleSpinBox()
        self.guidedThresholdSpin.setRange(300.0, 4000.0)
        self.guidedThresholdSpin.setDecimals(1)
        self.guidedThresholdSpin.setValue(1800.0)
        self.guidedThresholdSpin.setSuffix(" HU")
        form.addRow("Metal threshold", self.guidedThresholdSpin)

        detectRow = qt.QHBoxLayout()
        self.detectCandidatesButton = qt.QPushButton("Detect Candidates")
        self.detectCandidatesButton.clicked.connect(self.onDetectCandidatesClicked)
        detectRow.addWidget(self.detectCandidatesButton)
        self.resetWindowButton = qt.QPushButton("Reset CT Window")
        self.resetWindowButton.clicked.connect(self.onResetCTWindowClicked)
        detectRow.addWidget(self.resetWindowButton)
        detectRow.addStretch(1)
        form.addRow(detectRow)

        self.guidedRoiRadiusSpin = qt.QDoubleSpinBox()
        self.guidedRoiRadiusSpin.setRange(1.0, 6.0)
        self.guidedRoiRadiusSpin.setDecimals(2)
        self.guidedRoiRadiusSpin.setValue(3.0)
        self.guidedRoiRadiusSpin.setSuffix(" mm")
        form.addRow("ROI radius", self.guidedRoiRadiusSpin)

        self.guidedMaxAngleSpin = qt.QDoubleSpinBox()
        self.guidedMaxAngleSpin.setRange(1.0, 25.0)
        self.guidedMaxAngleSpin.setDecimals(1)
        self.guidedMaxAngleSpin.setValue(12.0)
        self.guidedMaxAngleSpin.setSuffix(" deg")
        form.addRow("Max angle deviation", self.guidedMaxAngleSpin)

        self.guidedMaxDepthShiftSpin = qt.QDoubleSpinBox()
        self.guidedMaxDepthShiftSpin.setRange(1.0, 50.0)
        self.guidedMaxDepthShiftSpin.setDecimals(1)
        self.guidedMaxDepthShiftSpin.setValue(20.0)
        self.guidedMaxDepthShiftSpin.setSuffix(" mm")
        form.addRow("Max depth shift", self.guidedMaxDepthShiftSpin)

        fitRow = qt.QHBoxLayout()
        self.fitSelectedButton = qt.QPushButton("Fit Checked")
        self.fitSelectedButton.clicked.connect(self.onFitSelectedClicked)
        fitRow.addWidget(self.fitSelectedButton)
        self.fitAllButton = qt.QPushButton("Fit All")
        self.fitAllButton.clicked.connect(self.onFitAllClicked)
        fitRow.addWidget(self.fitAllButton)
        self.applyFitButton = qt.QPushButton("Apply Fit to Trajectories")
        self.applyFitButton.clicked.connect(self.onApplyFitClicked)
        self.applyFitButton.setEnabled(False)
        self.applyFitButton.setVisible(False)
        fitRow.addWidget(self.applyFitButton)
        form.addRow(fitRow)

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

        self.guidedTrajectoryTable = qt.QTableWidget()
        self.guidedTrajectoryTable.setColumnCount(3)
        self.guidedTrajectoryTable.setHorizontalHeaderLabels(["Use", "Trajectory", "Length (mm)"])
        self.guidedTrajectoryTable.horizontalHeader().setSectionResizeMode(0, qt.QHeaderView.ResizeToContents)
        self.guidedTrajectoryTable.horizontalHeader().setSectionResizeMode(1, qt.QHeaderView.Stretch)
        self.guidedTrajectoryTable.horizontalHeader().setSectionResizeMode(2, qt.QHeaderView.ResizeToContents)
        self.guidedTrajectoryTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.guidedTrajectoryTable.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.guidedTrajectoryTable.cellClicked.connect(self.onGuidedTrajectoryTableCellClicked)
        self.guidedTrajectoryTable.currentCellChanged.connect(self.onGuidedTrajectoryTableCurrentCellChanged)
        self.guidedTrajectoryTable.itemChanged.connect(self.onGuidedTrajectoryItemChanged)
        form.addRow(self.guidedTrajectoryTable)

        actions_row = qt.QHBoxLayout()
        self.removeCheckedButton = qt.QPushButton("Remove Checked Trajectories")
        self.removeCheckedButton.clicked.connect(self.onRemoveCheckedClicked)
        actions_row.addWidget(self.removeCheckedButton)
        actions_row.addStretch(1)
        form.addRow(actions_row)

    def _build_de_novo_tab(self):
        tab = qt.QWidget()
        self.modeTabs.addTab(tab, "De Novo Detect")
        form = qt.QFormLayout(tab)

        self.deNovoPipelineCombo = qt.QComboBox()
        self.deNovoPipelineCombo.setToolTip("Detection engine pipeline for de novo CT localization.")
        for entry in self.logic.available_de_novo_pipelines():
            label = str(entry.get("display_name") or entry.get("key") or "pipeline")
            if bool(entry.get("scaffold", False)):
                label = f"{label} (scaffold)"
            self.deNovoPipelineCombo.addItem(label, str(entry.get("key") or ""))
        self.deNovoPipelineCombo.currentIndexChanged.connect(self.onDeNovoPipelineChanged)
        default_index = self.deNovoPipelineCombo.findData(self.logic.default_de_novo_pipeline_key)
        if default_index >= 0:
            self.deNovoPipelineCombo.setCurrentIndex(default_index)
        form.addRow("Detection pipeline", self.deNovoPipelineCombo)

        self.deNovoThresholdSpin = qt.QDoubleSpinBox()
        self.deNovoThresholdSpin.setRange(300.0, 4000.0)
        self.deNovoThresholdSpin.setDecimals(1)
        self.deNovoThresholdSpin.setValue(1800.0)
        self.deNovoThresholdSpin.setSuffix(" HU")
        form.addRow("Metal threshold", self.deNovoThresholdSpin)

        self.deNovoInlierRadiusSpin = qt.QDoubleSpinBox()
        self.deNovoInlierRadiusSpin.setRange(0.3, 6.0)
        self.deNovoInlierRadiusSpin.setDecimals(2)
        self.deNovoInlierRadiusSpin.setValue(1.2)
        self.deNovoInlierRadiusSpin.setSuffix(" mm")
        form.addRow("Inlier radius", self.deNovoInlierRadiusSpin)

        self.deNovoMinLengthSpin = qt.QDoubleSpinBox()
        self.deNovoMinLengthSpin.setRange(5.0, 200.0)
        self.deNovoMinLengthSpin.setDecimals(1)
        self.deNovoMinLengthSpin.setValue(20.0)
        self.deNovoMinLengthSpin.setSuffix(" mm")
        form.addRow("Min trajectory length", self.deNovoMinLengthSpin)

        self.deNovoMinInliersSpin = qt.QSpinBox()
        self.deNovoMinInliersSpin.setRange(25, 20000)
        self.deNovoMinInliersSpin.setValue(250)
        form.addRow("Min inliers", self.deNovoMinInliersSpin)

        self.deNovoMaxPointsSpin = qt.QSpinBox()
        self.deNovoMaxPointsSpin.setRange(5000, 1000000)
        self.deNovoMaxPointsSpin.setSingleStep(5000)
        self.deNovoMaxPointsSpin.setValue(300000)
        form.addRow("Max sampled points", self.deNovoMaxPointsSpin)

        self.deNovoRansacSpin = qt.QSpinBox()
        self.deNovoRansacSpin.setRange(20, 2000)
        self.deNovoRansacSpin.setValue(240)
        form.addRow("RANSAC iterations", self.deNovoRansacSpin)

        self.deNovoMaxLinesSpin = qt.QSpinBox()
        self.deNovoMaxLinesSpin.setRange(1, 50)
        self.deNovoMaxLinesSpin.setValue(30)
        form.addRow("Max lines", self.deNovoMaxLinesSpin)

        self.deNovoUseExactCountCheck = qt.QCheckBox("Use exact trajectory count")
        self.deNovoUseExactCountCheck.setChecked(False)
        self.deNovoUseExactCountCheck.toggled.connect(self._onUseExactToggled)
        form.addRow(self.deNovoUseExactCountCheck)

        self.deNovoExactCountSpin = qt.QSpinBox()
        self.deNovoExactCountSpin.setRange(1, 50)
        self.deNovoExactCountSpin.setValue(10)
        self.deNovoExactCountSpin.setEnabled(False)
        form.addRow("Exact count", self.deNovoExactCountSpin)

        self.deNovoUseHeadMaskCheck = qt.QCheckBox("Use head-depth gating")
        self.deNovoUseHeadMaskCheck.setChecked(True)
        form.addRow(self.deNovoUseHeadMaskCheck)

        self.deNovoHeadThresholdSpin = qt.QDoubleSpinBox()
        self.deNovoHeadThresholdSpin.setRange(-1200.0, 500.0)
        self.deNovoHeadThresholdSpin.setDecimals(1)
        self.deNovoHeadThresholdSpin.setValue(-500.0)
        self.deNovoHeadThresholdSpin.setSuffix(" HU")
        form.addRow("Head threshold", self.deNovoHeadThresholdSpin)

        self.deNovoHeadMaskMethodCombo = qt.QComboBox()
        self.deNovoHeadMaskMethodCombo.addItem("Outside air (default)", "outside_air")
        self.deNovoHeadMaskMethodCombo.addItem("Not-air LCC", "not_air_lcc")
        self.deNovoHeadMaskMethodCombo.addItem("Legacy", "legacy")
        self.deNovoHeadMaskMethodCombo.addItem("Tissue cut", "tissue_cut")
        self.deNovoHeadMaskMethodCombo.addItem("Tissue cut (no close)", "tissue_cut_noclose")
        form.addRow("Head mask method", self.deNovoHeadMaskMethodCombo)

        self.deNovoMinDepthSpin = qt.QDoubleSpinBox()
        self.deNovoMinDepthSpin.setRange(0.0, 100.0)
        self.deNovoMinDepthSpin.setDecimals(2)
        self.deNovoMinDepthSpin.setValue(5.0)
        self.deNovoMinDepthSpin.setSuffix(" mm")
        form.addRow("Min metal depth", self.deNovoMinDepthSpin)

        self.deNovoMaxDepthSpin = qt.QDoubleSpinBox()
        self.deNovoMaxDepthSpin.setRange(0.0, 300.0)
        self.deNovoMaxDepthSpin.setDecimals(2)
        self.deNovoMaxDepthSpin.setValue(220.0)
        self.deNovoMaxDepthSpin.setSuffix(" mm")
        form.addRow("Max metal depth", self.deNovoMaxDepthSpin)

        self.deNovoStartZoneWindowSpin = qt.QDoubleSpinBox()
        self.deNovoStartZoneWindowSpin.setRange(0.0, 50.0)
        self.deNovoStartZoneWindowSpin.setDecimals(2)
        self.deNovoStartZoneWindowSpin.setValue(10.0)
        self.deNovoStartZoneWindowSpin.setSuffix(" mm")
        self.deNovoStartZoneWindowSpin.setToolTip(
            "Each detected shank must include at least one support point in "
            "[Min metal depth, Min metal depth + this window]."
        )
        form.addRow("Start-zone window", self.deNovoStartZoneWindowSpin)

        self.deNovoCandidateModeCombo = qt.QComboBox()
        self.deNovoCandidateModeCombo.addItem("Voxel", "voxel")
        self.deNovoCandidateModeCombo.addItem("Blob centroid", "blob_centroid")
        form.addRow("Candidate mode", self.deNovoCandidateModeCombo)

        self.deNovoMinBlobVoxelsSpin = qt.QSpinBox()
        self.deNovoMinBlobVoxelsSpin.setRange(1, 10000)
        self.deNovoMinBlobVoxelsSpin.setValue(2)
        form.addRow("Min blob voxels", self.deNovoMinBlobVoxelsSpin)

        self.deNovoMaxBlobVoxelsSpin = qt.QSpinBox()
        self.deNovoMaxBlobVoxelsSpin.setRange(1, 20000)
        self.deNovoMaxBlobVoxelsSpin.setValue(1200)
        form.addRow("Max blob voxels", self.deNovoMaxBlobVoxelsSpin)

        self.deNovoMinBlobPeakHuEdit = qt.QLineEdit()
        self.deNovoMinBlobPeakHuEdit.setPlaceholderText("optional (empty = disabled)")
        form.addRow("Min blob peak HU", self.deNovoMinBlobPeakHuEdit)

        self.deNovoEnableRescueCheck = qt.QCheckBox("Enable rescue pass")
        self.deNovoEnableRescueCheck.setChecked(True)
        form.addRow(self.deNovoEnableRescueCheck)

        self.deNovoRescueScaleSpin = qt.QDoubleSpinBox()
        self.deNovoRescueScaleSpin.setRange(0.1, 1.0)
        self.deNovoRescueScaleSpin.setDecimals(2)
        self.deNovoRescueScaleSpin.setSingleStep(0.05)
        self.deNovoRescueScaleSpin.setValue(0.6)
        form.addRow("Rescue min-inlier scale", self.deNovoRescueScaleSpin)

        self.deNovoRescueMaxLinesSpin = qt.QSpinBox()
        self.deNovoRescueMaxLinesSpin.setRange(0, 30)
        self.deNovoRescueMaxLinesSpin.setValue(6)
        form.addRow("Rescue max lines", self.deNovoRescueMaxLinesSpin)

        self.deNovoUseModelScoreCheck = qt.QCheckBox("Use model-template scoring")
        self.deNovoUseModelScoreCheck.setChecked(True)
        self.deNovoUseModelScoreCheck.toggled.connect(self._onUseModelScoreToggled)
        form.addRow(self.deNovoUseModelScoreCheck)

        self.deNovoMinModelScoreSpin = qt.QDoubleSpinBox()
        self.deNovoMinModelScoreSpin.setRange(-1.0, 5.0)
        self.deNovoMinModelScoreSpin.setDecimals(2)
        self.deNovoMinModelScoreSpin.setSingleStep(0.05)
        self.deNovoMinModelScoreSpin.setValue(0.10)
        form.addRow("Min model score", self.deNovoMinModelScoreSpin)
        self._onUseModelScoreToggled(bool(self.deNovoUseModelScoreCheck.checked))

        self.deNovoDebugDiagnosticsCheck = qt.QCheckBox("Debug diagnostics (nodes + JSON)")
        self.deNovoDebugDiagnosticsCheck.setChecked(False)
        form.addRow(self.deNovoDebugDiagnosticsCheck)

        self.deNovoReplaceCheck = qt.QCheckBox("Replace existing working trajectories")
        self.deNovoReplaceCheck.setChecked(True)
        form.addRow(self.deNovoReplaceCheck)

        self.deNovoDetectButton = qt.QPushButton("Detect Trajectories")
        self.deNovoDetectButton.clicked.connect(self.onDeNovoDetectClicked)
        form.addRow(self.deNovoDetectButton)

    def _onUseExactToggled(self, checked):
        self.deNovoExactCountSpin.setEnabled(bool(checked))

    def _onUseModelScoreToggled(self, checked):
        self.deNovoMinModelScoreSpin.setEnabled(bool(checked))

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

    @staticmethod
    def _optional_float_from_text(text):
        """Parse optional float value from text; empty string -> None."""
        raw = str(text or "").strip()
        if not raw:
            return None
        return float(raw)

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

    def _populate_guided_trajectory_table(self):
        self._updatingGuidedTable = True
        try:
            self.guidedTrajectoryTable.setRowCount(0)
            for row, traj in enumerate(self.loadedTrajectories):
                self.guidedTrajectoryTable.insertRow(row)

                use_check = qt.QCheckBox()
                use_check.setChecked(True)
                use_check.setStyleSheet("margin-left:8px; margin-right:8px;")
                self.guidedTrajectoryTable.setCellWidget(row, 0, use_check)

                traj_name = str(traj.get("name", ""))
                name_item = qt.QTableWidgetItem(traj_name)
                name_item.setFlags(name_item.flags() | qt.Qt.ItemIsEditable)
                name_item.setData(qt.Qt.UserRole, traj_name)
                self.guidedTrajectoryTable.setItem(row, 1, name_item)

                length_item = qt.QTableWidgetItem(f"{trajectory_length_mm(traj):.2f}")
                length_item.setFlags(length_item.flags() & ~qt.Qt.ItemIsEditable)
                self.guidedTrajectoryTable.setItem(row, 2, length_item)

            if self.guidedTrajectoryTable.rowCount > 0:
                self.guidedTrajectoryTable.selectRow(0)
        finally:
            self._updatingGuidedTable = False

    def onGuidedTrajectoryItemChanged(self, item):
        if self._updatingGuidedTable or self._renamingGuidedTrajectory:
            return
        if item is None or item.column() != 1:
            return
        row = int(item.row())
        if row < 0 or row >= len(self.loadedTrajectories):
            return

        old_name = str(item.data(qt.Qt.UserRole) or self.loadedTrajectories[row].get("name", "")).strip()
        new_name = str(item.text() or "").strip()
        if not old_name:
            return
        if not new_name:
            self._renamingGuidedTrajectory = True
            item.setText(old_name)
            self._renamingGuidedTrajectory = False
            return
        if new_name == old_name:
            return

        for r in range(self.guidedTrajectoryTable.rowCount):
            if r == row:
                continue
            other = self.guidedTrajectoryTable.item(r, 1)
            if other and str(other.text() or "").strip().lower() == new_name.lower():
                self._renamingGuidedTrajectory = True
                item.setText(old_name)
                self._renamingGuidedTrajectory = False
                qt.QMessageBox.warning(
                    slicer.util.mainWindow(),
                    "Postop CT Localization",
                    f"Trajectory name '{new_name}' already exists.",
                )
                return

        node_id = self.loadedTrajectories[row].get("node_id", "")
        if not self.logic.rename_trajectory(node_id=node_id, new_name=new_name):
            self._renamingGuidedTrajectory = True
            item.setText(old_name)
            self._renamingGuidedTrajectory = False
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "Postop CT Localization",
                f"Failed to rename trajectory '{old_name}'.",
            )
            return

        self.loadedTrajectories[row]["name"] = new_name
        item.setData(qt.Qt.UserRole, new_name)
        self.log(f"[trajectory] renamed {old_name} -> {new_name}")
        self._refresh_summary()
        self._schedule_guided_follow()

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

    def _workflow_de_novo_pipeline_key(self):
        if self.workflowNode is None:
            return ""
        return str(self.workflowNode.GetParameter("DeNovoPipelineKey") or "").strip()

    def _set_workflow_de_novo_pipeline_key(self, pipeline_key):
        if self.workflowNode is None:
            return
        self.workflowNode.SetParameter("DeNovoPipelineKey", str(pipeline_key or "").strip())

    def _set_de_novo_pipeline_combo(self, pipeline_key):
        key = str(pipeline_key or "").strip()
        idx = self.deNovoPipelineCombo.findData(key)
        if idx < 0 or idx == self.deNovoPipelineCombo.currentIndex:
            return
        self._syncingPipelineCombo = True
        try:
            self.deNovoPipelineCombo.setCurrentIndex(idx)
        finally:
            self._syncingPipelineCombo = False

    def _sync_de_novo_pipeline_from_workflow(self):
        key = self._workflow_de_novo_pipeline_key()
        if not key:
            key = self.logic.default_de_novo_pipeline_key
            self._set_workflow_de_novo_pipeline_key(key)
        self._set_de_novo_pipeline_combo(key)

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
        if self._role_has_nodes("DeNovoTrajectoryLines"):
            return "de_novo"
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
            "working": ["imported_rosa", "imported_external", "manual", "guided_fit", "de_novo"],
            "imported_rosa": ["imported_rosa"],
            "imported_external": ["imported_external"],
            "manual": ["manual"],
            "guided_fit": ["guided_fit"],
            "de_novo": ["de_novo"],
            "planned_rosa": ["planned_rosa"],
        }
        groups = group_map.get(key, group_map["working"])
        self.logic.trajectory_scene.show_only_groups(groups)

    def _refresh_summary(self):
        self.summaryLabel.setText(
            f"working trajectories={len(self.loadedTrajectories)}, "
            f"assignments={len(self.assignmentMap)}, "
            f"candidate points={(len(self.candidatesLPS) if np is None else int(getattr(self.candidatesLPS, 'shape', [0])[0]))}"
        )

    def onRefreshClicked(self):
        self.workflowNode = self.workflowState.resolve_or_create_workflow_node()
        self._sync_ct_selector_from_workflow()
        if not self._workflow_active_source():
            self._set_workflow_active_source(self._default_source_when_unset())
        self._sync_guided_source_from_workflow()
        self._sync_de_novo_pipeline_from_workflow()
        source_key = self._selected_guided_source_key()
        self.loadedTrajectories = self.logic.collect_trajectories_by_source(
            source_key=source_key,
            workflow_node=self.workflowNode,
        )
        self.assignmentMap = self._assignment_map_from_workflow()
        self._populate_guided_trajectory_table()
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

    def onDeNovoPipelineChanged(self, _idx):
        if self._syncingPipelineCombo:
            return
        pipeline_data = self.deNovoPipelineCombo.currentData
        pipeline_key = pipeline_data() if callable(pipeline_data) else pipeline_data
        self._set_workflow_de_novo_pipeline_key(str(pipeline_key or self.logic.default_de_novo_pipeline_key))

    def onApplyFocusLayoutClicked(self):
        self.logic.layout_service.sanitize_focus_layout_state()
        if not self.logic.layout_service.apply_trajectory_focus_layout():
            self.log("[focus] failed to apply trajectory focus layout")
            return
        self.log("[focus] trajectory focus layout applied")
        self._schedule_guided_follow()

    def onGuidedTrajectoryTableCellClicked(self, _row, _col):
        self._schedule_guided_follow()

    def onGuidedTrajectoryTableCurrentCellChanged(self, _currentRow, _currentColumn, _previousRow, _previousColumn):
        self._schedule_guided_follow()

    def onDetectCandidatesClicked(self):
        volume_node = self.ctSelector.currentNode()
        if volume_node is None:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Postop CT Localization", "Select a CT volume.")
            return
        threshold = float(self.guidedThresholdSpin.value)
        try:
            points_lps = self.logic.extract_threshold_candidates_lps(
                volume_node=volume_node,
                threshold=threshold,
            )
        except Exception as exc:
            self.log(f"[guided] detect error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Postop CT Localization", str(exc))
            return

        self.candidatesLPS = points_lps
        self.fitResults = {}
        self.applyFitButton.setEnabled(False)

        self.logic.register_postop_ct(volume_node, workflow_node=self.workflowNode)
        self._apply_primary_slice_layers()
        self.logic.apply_ct_window_from_threshold(volume_node, threshold=threshold)
        n = int(points_lps.shape[0]) if np is not None else len(points_lps)
        self.log(f"[guided] detected {n} candidate points at threshold {threshold:.1f}")
        self._refresh_summary()

    def onResetCTWindowClicked(self):
        volume_node = self.ctSelector.currentNode()
        if volume_node is None:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Postop CT Localization", "Select a CT volume.")
            return
        self.logic.reset_ct_window(volume_node)
        self.log("[guided] CT window reset")

    def _assignment_for_trajectory(self, traj):
        row = self.assignmentMap.get(traj["name"])
        if row and row.get("model_id") in self.modelsById:
            return row
        model_id = suggest_model_id_for_trajectory(
            trajectory=traj,
            models_by_id=self.modelsById,
            model_ids=self.modelIds,
            tolerance_mm=5.0,
        )
        if not model_id:
            # Model-free guided fit: still allow trajectory axis/depth fitting.
            return {
                "trajectory": traj["name"],
                "model_id": "",
                "tip_at": "target",
                "tip_shift_mm": 0.0,
                "xyz_offset_mm": [0.0, 0.0, 0.0],
            }
        return {
            "trajectory": traj["name"],
            "model_id": model_id,
            "tip_at": "target",
            "tip_shift_mm": 0.0,
            "xyz_offset_mm": [0.0, 0.0, 0.0],
        }

    def _fit_names(self, names):
        if np is not None and int(self.candidatesLPS.shape[0]) == 0:
            raise ValueError("No CT candidates. Run 'Detect Candidates' first.")

        traj_map = {traj["name"]: traj for traj in self.loadedTrajectories}
        success = 0
        applied_nodes = []
        self.logic.trajectory_scene.remove_preview_lines(node_prefix="AutoFit_")
        for name in names:
            traj = traj_map.get(name)
            if traj is None:
                self.log(f"[guided] {name}: skipped (missing trajectory)")
                continue
            assignment = self._assignment_for_trajectory(traj)
            if assignment is None:
                self.log(f"[guided] {name}: skipped (no electrode model)")
                continue
            model = self.modelsById.get(assignment["model_id"])
            offsets = model["contact_center_offsets_from_tip_mm"] if model else []
            fit = fit_electrode_axis_and_tip(
                candidate_points_lps=self.candidatesLPS,
                planned_entry_lps=traj["start"],
                planned_target_lps=traj["end"],
                contact_offsets_mm=offsets,
                tip_at=assignment.get("tip_at", "target"),
                roi_radius_mm=float(self.guidedRoiRadiusSpin.value),
                max_angle_deg=float(self.guidedMaxAngleSpin.value),
                max_depth_shift_mm=float(self.guidedMaxDepthShiftSpin.value),
            )
            if fit.get("success"):
                self.fitResults[name] = fit
                start_ras = lps_to_ras_point(fit["entry_lps"])
                end_ras = lps_to_ras_point(fit["target_lps"])
                existing = self.logic.trajectory_scene.find_line_by_group_and_name(name, "guided_fit")
                node = self.logic.trajectory_scene.create_or_update_trajectory_line(
                    name=name,
                    start_ras=start_ras,
                    end_ras=end_ras,
                    node_id=None if existing is None else existing.GetID(),
                    node_name=f"Guided_{name}",
                    group="guided_fit",
                    origin="postop_ct_guided_fit",
                )
                applied_nodes.append(node)
                success += 1
                self.log(
                    "[guided] {name}: angle={a:.2f} deg depth={s:.2f} mm lateral={l:.2f} mm residual={r:.2f} mm".format(
                        name=name,
                        a=float(fit.get("angle_deg", 0.0)),
                        s=float(fit.get("tip_shift_mm", 0.0)),
                        l=float(fit.get("lateral_shift_mm", 0.0)),
                        r=float(fit.get("residual_mm", 0.0)),
                    )
                )
            else:
                self.log(f"[guided] {name}: failed ({fit.get('reason', 'unknown')})")
        if applied_nodes:
            self.workflowPublisher.publish_nodes(
                role="GuidedFitTrajectoryLines",
                nodes=applied_nodes,
                source="postop_ct_guided_fit",
                space_name="ROSA_BASE",
                workflow_node=self.workflowNode,
            )
            self.workflowPublisher.publish_nodes(
                role="WorkingTrajectoryLines",
                nodes=applied_nodes,
                source="postop_ct_guided_fit",
                space_name="ROSA_BASE",
                workflow_node=self.workflowNode,
            )
            self.logic.trajectory_scene.place_trajectory_nodes_in_hierarchy(
                context_id=self.workflowState.context_id(workflow_node=self.workflowNode),
                nodes=applied_nodes,
            )
            self.logic.trajectory_scene.show_only_groups(["guided_fit"])
            self._set_workflow_active_source("guided_fit")
            self._set_guided_source_combo("guided_fit")
            self.log(f"[guided] applied fitted trajectories: {len(applied_nodes)}")
            self.log(
                "[contacts] guided trajectories updated. Use module 'Contacts & Trajectory View' "
                "with source 'Guided Fit' to generate contacts and QC."
            )
            self.onRefreshClicked()

        self.applyFitButton.setEnabled(False)
        self.log(f"[guided] fitted {success}/{len(names)} trajectories")

    def onFitSelectedClicked(self):
        names = self._checked_guided_trajectory_names()
        if not names:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "Postop CT Localization",
                "Check at least one trajectory in the table.",
            )
            return
        try:
            self._fit_names(names)
        except Exception as exc:
            self.log(f"[guided] error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Postop CT Localization", str(exc))

    def onFitAllClicked(self):
        names = [traj["name"] for traj in self.loadedTrajectories]
        if not names:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Postop CT Localization", "No trajectories available.")
            return
        try:
            self._fit_names(names)
        except Exception as exc:
            self.log(f"[guided] error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Postop CT Localization", str(exc))

    def onApplyFitClicked(self):
        qt.QMessageBox.information(
            slicer.util.mainWindow(),
            "Postop CT Localization",
            "Guided fit is now applied directly when you click 'Fit Selected' or 'Fit All'.",
        )

    def _checked_guided_trajectory_names(self):
        names = []
        for row in range(self.guidedTrajectoryTable.rowCount):
            check = self.guidedTrajectoryTable.cellWidget(row, 0)
            if check is None or not bool(check.checked):
                continue
            item = self.guidedTrajectoryTable.item(row, 1)
            if item is None:
                continue
            name = str(item.text() or "").strip()
            if name:
                names.append(name)
        return names

    def onRemoveCheckedClicked(self):
        names = self._checked_guided_trajectory_names()
        if not names:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "Postop CT Localization",
                "Check at least one trajectory to remove.",
            )
            return
        removed = self.logic.remove_trajectories_by_name(names=names, source_key=self._selected_guided_source_key())
        for name in names:
            self.fitResults.pop(name, None)
        self.log(f"[trajectory] removed {removed}/{len(names)} trajectories")
        self.onRefreshClicked()

    def onDeNovoDetectClicked(self):
        volume_node = self.ctSelector.currentNode()
        if volume_node is None:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Postop CT Localization", "Select a CT volume.")
            return
        try:
            min_blob_peak_hu = self._optional_float_from_text(self.deNovoMinBlobPeakHuEdit.text)
        except Exception:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "Postop CT Localization",
                "Min blob peak HU must be numeric or empty.",
            )
            return

        max_lines = int(self.deNovoExactCountSpin.value) if self.deNovoUseExactCountCheck.checked else int(
            self.deNovoMaxLinesSpin.value
        )
        candidate_mode_data = self.deNovoCandidateModeCombo.currentData
        candidate_mode = candidate_mode_data() if callable(candidate_mode_data) else candidate_mode_data
        candidate_mode = str(candidate_mode or "voxel")
        head_mask_method_data = self.deNovoHeadMaskMethodCombo.currentData
        head_mask_method = head_mask_method_data() if callable(head_mask_method_data) else head_mask_method_data
        head_mask_method = str(head_mask_method or "outside_air")
        pipeline_data = self.deNovoPipelineCombo.currentData
        pipeline_key = pipeline_data() if callable(pipeline_data) else pipeline_data
        pipeline_key = str(
            pipeline_key
            or self._workflow_de_novo_pipeline_key()
            or self.logic.default_de_novo_pipeline_key
        )
        self.logic.register_postop_ct(volume_node, workflow_node=self.workflowNode)

        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        slicer.app.processEvents()
        try:
            result = self.logic.detect_de_novo(
                volume_node=volume_node,
                threshold=float(self.deNovoThresholdSpin.value),
                max_points=int(self.deNovoMaxPointsSpin.value),
                max_lines=max_lines,
                inlier_radius_mm=float(self.deNovoInlierRadiusSpin.value),
                min_length_mm=float(self.deNovoMinLengthSpin.value),
                min_inliers=int(self.deNovoMinInliersSpin.value),
                ransac_iterations=int(self.deNovoRansacSpin.value),
                use_head_mask=bool(self.deNovoUseHeadMaskCheck.checked),
                head_mask_threshold_hu=float(self.deNovoHeadThresholdSpin.value),
                head_mask_method=head_mask_method,
                min_metal_depth_mm=float(self.deNovoMinDepthSpin.value),
                max_metal_depth_mm=float(self.deNovoMaxDepthSpin.value),
                start_zone_window_mm=float(self.deNovoStartZoneWindowSpin.value),
                candidate_mode=candidate_mode,
                min_blob_voxels=int(self.deNovoMinBlobVoxelsSpin.value),
                max_blob_voxels=int(self.deNovoMaxBlobVoxelsSpin.value),
                min_blob_peak_hu=min_blob_peak_hu,
                max_blob_elongation=None,
                enable_rescue_pass=bool(self.deNovoEnableRescueCheck.checked),
                rescue_min_inliers_scale=float(self.deNovoRescueScaleSpin.value),
                rescue_max_lines=int(self.deNovoRescueMaxLinesSpin.value),
                use_model_score=bool(self.deNovoUseModelScoreCheck.checked),
                min_model_score=float(self.deNovoMinModelScoreSpin.value),
                models_by_id=(self.modelsById if bool(self.deNovoUseModelScoreCheck.checked) else None),
                pipeline_key=pipeline_key,
            )
        except Exception as exc:
            self.log(f"[denovo] failed: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Postop CT Localization", str(exc))
            return
        finally:
            qt.QApplication.restoreOverrideCursor()
            slicer.app.processEvents()

        try:
            self.logic.show_metal_and_head_masks(
                volume_node=volume_node,
                metal_mask_kji=result.get("metal_mask_kji"),
                head_mask_kji=result.get("gating_mask_kji"),
                head_distance_map_kji=result.get("head_distance_map_kji"),
                distance_surface_mask_kji=result.get("distance_surface_mask_kji"),
                not_air_mask_kji=result.get("not_air_mask_kji"),
                not_air_eroded_mask_kji=result.get("not_air_eroded_mask_kji"),
                head_core_mask_kji=result.get("head_core_mask_kji"),
                metal_gate_mask_kji=result.get("metal_gate_mask_kji"),
                metal_in_gate_mask_kji=result.get("metal_in_gate_mask_kji"),
                depth_window_mask_kji=result.get("depth_window_mask_kji"),
                metal_depth_pass_mask_kji=result.get("metal_depth_pass_mask_kji"),
            )
            self.log("[denovo] displayed head/metal mask overlay")
        except Exception as exc:
            self.log(f"[denovo] mask display warning: {exc}")

        params = {
            "pipeline_key": pipeline_key,
            "threshold_hu": float(self.deNovoThresholdSpin.value),
            "candidate_mode": candidate_mode,
            "inlier_radius_mm": float(self.deNovoInlierRadiusSpin.value),
            "min_length_mm": float(self.deNovoMinLengthSpin.value),
            "min_inliers": int(self.deNovoMinInliersSpin.value),
            "ransac_iterations": int(self.deNovoRansacSpin.value),
            "max_lines": int(max_lines),
            "min_depth_mm": float(self.deNovoMinDepthSpin.value),
            "max_depth_mm": float(self.deNovoMaxDepthSpin.value),
            "head_mask_method": head_mask_method,
            "start_zone_window_mm": float(self.deNovoStartZoneWindowSpin.value),
            "min_blob_voxels": int(self.deNovoMinBlobVoxelsSpin.value),
            "max_blob_voxels": int(self.deNovoMaxBlobVoxelsSpin.value),
            "min_blob_peak_hu": min_blob_peak_hu,
            "enable_rescue_pass": bool(self.deNovoEnableRescueCheck.checked),
            "rescue_min_inliers_scale": float(self.deNovoRescueScaleSpin.value),
            "rescue_max_lines": int(self.deNovoRescueMaxLinesSpin.value),
            "use_model_score": bool(self.deNovoUseModelScoreCheck.checked),
            "min_model_score": float(self.deNovoMinModelScoreSpin.value),
        }
        diagnostics = self.logic._build_detection_diagnostics(result=result, params=params)

        self.log(
            "[denovo] mode={mode} candidates={cand} after-mask={after_mask} after-depth={after_depth} "
            "min_inliers={min_inliers} inlier_radius={inlier_radius:.2f} fit1={fit1} fit2={fit2} rescue={rescue} "
            "final={final} unassigned={unassigned} total={total:.1f}ms".format(
                mode=str(diagnostics.get("candidate_mode", "voxel")),
                cand=int(diagnostics.get("candidate_points_total", 0)),
                after_mask=int(diagnostics.get("candidate_points_after_mask", 0)),
                after_depth=int(diagnostics.get("candidate_points_after_depth", 0)),
                min_inliers=int(diagnostics.get("effective_min_inliers", 0)),
                inlier_radius=float(diagnostics.get("effective_inlier_radius_mm", 0.0)),
                fit1=int(diagnostics.get("fit1_lines_proposed", 0)),
                fit2=int(diagnostics.get("fit2_lines_kept", 0)),
                rescue=int(diagnostics.get("rescue_lines_kept", 0)),
                final=int(diagnostics.get("final_lines_kept", 0)),
                unassigned=int(diagnostics.get("final_unassigned_points", 0)),
                total=float(diagnostics.get("profile_ms", {}).get("total", 0.0)),
            )
        )
        try:
            if bool(self.deNovoDebugDiagnosticsCheck.checked):
                self.logic.show_blob_diagnostics(
                    volume_node=volume_node,
                    blob_labelmap_kji=result.get("blob_labelmap_kji"),
                    blob_centroids_all_ras=result.get("blob_centroids_all_ras"),
                    blob_centroids_kept_ras=result.get("blob_centroids_kept_ras"),
                    blob_centroids_rejected_ras=result.get("blob_centroids_rejected_ras"),
                )
            else:
                # Always show included blob centroids in 3D, keep all/rejected
                # hidden unless debug mode is enabled.
                self.logic.show_blob_diagnostics(
                    volume_node=volume_node,
                    blob_labelmap_kji=None,
                    blob_centroids_all_ras=np.empty((0, 3), dtype=float),
                    blob_centroids_kept_ras=result.get("blob_centroids_kept_ras"),
                    blob_centroids_rejected_ras=np.empty((0, 3), dtype=float),
                )
        except Exception as exc:
            self.log(f"[denovo] blob diagnostics warning: {exc}")

        if bool(self.deNovoDebugDiagnosticsCheck.checked):
            try:
                out_json = self.logic.write_detection_diagnostics_json(
                    volume_node=volume_node,
                    diagnostics=diagnostics,
                )
                self.log(
                    "[denovo:debug] blobs(total={total}, kept={kept}, reject_small={small}, reject_large={large}, "
                    "reject_intensity={intensity}, reject_shape={shape})".format(
                        total=int(diagnostics.get("blob_count_total", 0)),
                        kept=int(diagnostics.get("blob_count_kept", 0)),
                        small=int(diagnostics.get("blob_reject_small", 0)),
                        large=int(diagnostics.get("blob_reject_large", 0)),
                        intensity=int(diagnostics.get("blob_reject_intensity", 0)),
                        shape=int(diagnostics.get("blob_reject_shape", 0)),
                    )
                )
                self.log(f"[denovo:debug] diagnostics json: {out_json}")
            except Exception as exc:
                self.log(f"[denovo] debug diagnostics warning: {exc}")

        lines = result.get("lines", [])
        if not lines:
            self.log("[denovo] no trajectories detected")
            return

        rows = self.logic.upsert_detected_lines(
            lines=lines,
            replace_existing=bool(self.deNovoReplaceCheck.checked),
        )
        self.logic.publish_working_rows(
            rows,
            workflow_node=self.workflowNode,
            role="DeNovoTrajectoryLines",
            source="postop_ct_denovo",
        )
        self.logic.trajectory_scene.show_only_groups(["de_novo"])
        self._set_workflow_active_source("de_novo")
        self._set_guided_source_combo("de_novo")
        self.log(
            f"[denovo] start-zone-reject={int(result.get('start_zone_reject_count', 0))}, "
            f"new_lines={len(lines)}, total_rows={len(rows)}"
        )
        self.log(
            "[contacts] trajectories updated. Use module 'Contacts & Trajectory View' "
            "with source 'Guided Fit' or 'De Novo' to generate contacts and QC."
        )
        self.onRefreshClicked()


class PostopCTLocalizationLogic(ScriptedLoadableModuleLogic):
    """Logic for guided-fit and de-novo postop CT localization."""

    def __init__(self):
        super().__init__()
        if np is None:
            raise RuntimeError("numpy is required for Postop CT Localization.")
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
        self.pipeline_registry = PipelineRegistry()
        register_builtin_pipelines(self.pipeline_registry)
        self.default_de_novo_pipeline_key = self._resolve_default_de_novo_pipeline_key()

    def _resolve_default_de_novo_pipeline_key(self):
        keys = list(self.pipeline_registry.keys())
        if not keys:
            return ""
        preferred = "blob_ransac_v1"
        return preferred if preferred in keys else str(keys[0])

    def available_de_novo_pipelines(self):
        """Return display metadata for registered de novo engine pipelines."""
        entries = []
        for key in self.pipeline_registry.keys():
            display_name = key
            scaffold = False
            try:
                pipeline = self.pipeline_registry.create_pipeline(key)
                display_name = str(getattr(pipeline, "display_name", key) or key)
                scaffold = bool(getattr(pipeline, "scaffold", False))
            except Exception:
                pass
            entries.append(
                {
                    "key": str(key),
                    "display_name": display_name,
                    "scaffold": scaffold,
                }
            )
        return entries

    def register_postop_ct(self, volume_node, workflow_node=None):
        self.workflow_publish.register_volume(
            volume_node=volume_node,
            source_type="import",
            source_path="",
            space_name="ROSA_BASE",
            role="AdditionalCTVolumes",
            is_default_postop=True,
            workflow_node=workflow_node,
        )

    def _engine_result_to_legacy_detection_result(self, engine_result, ctx_extras, config):
        """Map canonical engine result to legacy de-novo payload shape.

        The widget currently expects legacy keys (`lines`, mask arrays, reject
        counters). For `blob_ransac_v1`, we preserve the exact legacy payload via
        `ctx_extras['legacy_result']`. Other pipelines receive a minimal mapped
        payload that still supports trajectory ingestion + diagnostics.
        """

        legacy = ctx_extras.get("legacy_result") if isinstance(ctx_extras, dict) else None
        if isinstance(legacy, dict):
            return legacy

        diagnostics = dict((engine_result.get("diagnostics") or {}))
        counts = dict(diagnostics.get("counts") or {})
        timing = dict(diagnostics.get("timing") or {})

        lines = []
        for idx, traj in enumerate(list(engine_result.get("trajectories") or []), start=1):
            start = list(traj.get("start_ras", []))
            end = list(traj.get("end_ras", []))
            params = dict(traj.get("params") or {})
            if len(start) != 3:
                start = list(params.get("start_ras", [0.0, 0.0, 0.0]))
            if len(end) != 3:
                end = list(params.get("end_ras", [0.0, 0.0, 0.0]))
            if len(start) != 3:
                start = [0.0, 0.0, 0.0]
            if len(end) != 3:
                end = [0.0, 0.0, 0.0]
            start_np = np.asarray(start, dtype=float).reshape(3)
            end_np = np.asarray(end, dtype=float).reshape(3)
            lines.append(
                {
                    "name": str(traj.get("name") or f"T{idx:02d}"),
                    "start_ras": [float(v) for v in start_np],
                    "end_ras": [float(v) for v in end_np],
                    "length_mm": float(np.linalg.norm(end_np - start_np)),
                    "inlier_count": int(traj.get("support_count", 0)),
                    "support_weight": float(traj.get("support_mass", traj.get("support_count", 0.0))),
                    "inside_fraction": float(traj.get("confidence", 0.0)),
                    "rms_mm": float(params.get("rms_mm", 0.0)),
                    "depth_span_mm": float(params.get("depth_span_mm", 0.0)),
                    "best_model_id": str(params.get("best_model_id", "")),
                    "best_model_score": params.get("best_model_score", None),
                }
            )

        return {
            "candidate_count": int(counts.get("candidate_points_total", 0)),
            "head_mask_kept_count": int(counts.get("candidate_points_after_mask", 0)),
            "gating_mask_type": str(diagnostics.get("extras", {}).get("gating_mask_type", "engine")),
            "inside_method": str(diagnostics.get("extras", {}).get("inside_method", "engine")),
            "metal_in_head_count": int(counts.get("candidate_points_after_mask", 0)),
            "depth_kept_count": int(counts.get("candidate_points_after_depth", 0)),
            "gap_reject_count": int(counts.get("gap_reject_count", 0)),
            "duplicate_reject_count": int(counts.get("duplicate_reject_count", 0)),
            "start_zone_reject_count": int(counts.get("start_zone_reject_count", 0)),
            "length_reject_count": int(counts.get("length_reject_count", 0)),
            "inlier_reject_count": int(counts.get("inlier_reject_count", 0)),
            "candidate_points_total": int(counts.get("candidate_points_total", 0)),
            "candidate_points_after_mask": int(counts.get("candidate_points_after_mask", 0)),
            "candidate_points_after_depth": int(counts.get("candidate_points_after_depth", 0)),
            "effective_min_inliers": int(config.get("min_inliers", 0)),
            "effective_inlier_radius_mm": float(config.get("inlier_radius_mm", 0.0)),
            "blob_count_total": int(counts.get("blob_count_total", 0)),
            "blob_count_kept": int(counts.get("blob_count_kept", 0)),
            "blob_reject_small": int(counts.get("blob_reject_small", 0)),
            "blob_reject_large": int(counts.get("blob_reject_large", 0)),
            "blob_reject_intensity": int(counts.get("blob_reject_intensity", 0)),
            "blob_reject_shape": int(counts.get("blob_reject_shape", 0)),
            "fit1_lines_proposed": int(counts.get("fit1_lines_proposed", 0)),
            "fit2_lines_kept": int(counts.get("fit2_lines_kept", 0)),
            "rescue_lines_kept": int(counts.get("rescue_lines_kept", 0)),
            "final_lines_kept": int(counts.get("final_lines_kept", len(lines))),
            "assigned_points_after_refine": int(counts.get("assigned_points_after_refine", 0)),
            "unassigned_points_after_refine": int(counts.get("unassigned_points_after_refine", 0)),
            "rescued_points": int(counts.get("rescued_points", 0)),
            "final_unassigned_points": int(counts.get("final_unassigned_points", 0)),
            "profile_ms": {
                "total": float(timing.get("total_ms", 0.0)),
                "first_pass": float(timing.get("stage.seed_initialization.ms", 0.0)),
                "refine": float(timing.get("stage.em_refinement.ms", 0.0)),
                "rescue": float(timing.get("stage.model_selection.ms", 0.0)),
            },
            "lines": lines,
        }

    def rename_trajectory(self, node_id, new_name):
        """Rename one trajectory line node while preserving group metadata."""
        node = slicer.mrmlScene.GetNodeByID(str(node_id or ""))
        if node is None:
            return False
        return bool(self.trajectory_scene.rename_trajectory_node(node, new_name))

    def remove_trajectories_by_name(self, names, source_key="working"):
        """Delete trajectories by logical name from current scene/source scope."""
        source = str(source_key or "working").strip().lower()
        allowed_groups = {
            "working": {"imported_rosa", "imported_external", "manual", "guided_fit", "de_novo"},
            "imported_rosa": {"imported_rosa"},
            "imported_external": {"imported_external"},
            "manual": {"manual"},
            "guided_fit": {"guided_fit"},
            "de_novo": {"de_novo"},
            "planned_rosa": {"planned_rosa"},
        }.get(source, {"imported_rosa", "imported_external", "manual", "guided_fit", "de_novo"})

        target_names = {str(name).strip() for name in (names or []) if str(name).strip()}
        if not target_names:
            return 0
        removed = 0
        for node in list(slicer.util.getNodesByClass("vtkMRMLMarkupsLineNode")):
            group = self.trajectory_scene.infer_group_from_node(node)
            if group not in allowed_groups:
                continue
            logical_name = self.trajectory_scene.logical_name_from_node(node)
            if logical_name not in target_names:
                continue
            slicer.mrmlScene.RemoveNode(node)
            removed += 1
        return removed

    def _get_or_create_labelmap_node(self, node_name):
        node = None
        try:
            node = slicer.util.getNode(node_name)
        except Exception:
            node = None
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", node_name)
            node.CreateDefaultDisplayNodes()
        return node

    def _update_labelmap_from_mask(self, reference_volume_node, node_name, mask_kji):
        if mask_kji is None:
            return None
        node = self._get_or_create_labelmap_node(node_name)
        arr = np.asarray(mask_kji)
        if arr.dtype == np.bool_:
            arr = arr.astype(np.uint8)
        elif np.issubdtype(arr.dtype, np.integer):
            # Keep integer label depth for multi-component debug maps.
            if arr.max() > np.iinfo(np.uint8).max:
                arr = arr.astype(np.uint16)
            elif arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        slicer.util.updateVolumeFromArray(node, arr)
        m = vtk.vtkMatrix4x4()
        reference_volume_node.GetIJKToRASMatrix(m)
        node.SetIJKToRASMatrix(m)
        display = node.GetDisplayNode()
        if display:
            try:
                color = slicer.util.getNode("GenericAnatomyColors")
                display.SetAndObserveColorNodeID(color.GetID())
            except Exception:
                pass
            display.SetVisibility(True)
        return node

    def show_metal_and_head_masks(
        self,
        volume_node,
        metal_mask_kji=None,
        head_mask_kji=None,
        head_distance_map_kji=None,
        distance_surface_mask_kji=None,
        not_air_mask_kji=None,
        not_air_eroded_mask_kji=None,
        head_core_mask_kji=None,
        metal_gate_mask_kji=None,
        metal_in_gate_mask_kji=None,
        depth_window_mask_kji=None,
        metal_depth_pass_mask_kji=None,
    ):
        """Display troubleshooting masks for de novo shank detection."""
        if metal_mask_kji is None:
            raise ValueError("metal_mask_kji is required for mask visualization")

        metal_bool = np.asarray(metal_mask_kji, dtype=bool)
        self._update_labelmap_from_mask(
            reference_volume_node=volume_node,
            node_name=f"{volume_node.GetName()}_MetalMask",
            mask_kji=metal_mask_kji,
        )
        if head_mask_kji is not None:
            self._update_labelmap_from_mask(
                reference_volume_node=volume_node,
                node_name=f"{volume_node.GetName()}_HeadMask",
                mask_kji=head_mask_kji,
            )
        distance_map_inside_mask = None
        if head_distance_map_kji is not None:
            distance_map_inside_mask = np.asarray(head_distance_map_kji, dtype=float) > 0.0
            self._update_labelmap_from_mask(
                reference_volume_node=volume_node,
                node_name=f"{volume_node.GetName()}_DistanceMapMask",
                mask_kji=distance_map_inside_mask,
            )
        if distance_surface_mask_kji is not None:
            self._update_labelmap_from_mask(
                reference_volume_node=volume_node,
                node_name=f"{volume_node.GetName()}_DistanceSurfaceMask",
                mask_kji=distance_surface_mask_kji,
            )
        if not_air_mask_kji is not None:
            self._update_labelmap_from_mask(
                reference_volume_node=volume_node,
                node_name=f"{volume_node.GetName()}_NotAirMask",
                mask_kji=not_air_mask_kji,
            )
        if not_air_eroded_mask_kji is not None:
            self._update_labelmap_from_mask(
                reference_volume_node=volume_node,
                node_name=f"{volume_node.GetName()}_NotAirErodedMask",
                mask_kji=not_air_eroded_mask_kji,
            )
        if head_core_mask_kji is not None:
            self._update_labelmap_from_mask(
                reference_volume_node=volume_node,
                node_name=f"{volume_node.GetName()}_HeadCoreMask",
                mask_kji=head_core_mask_kji,
            )
        if metal_gate_mask_kji is not None:
            self._update_labelmap_from_mask(
                reference_volume_node=volume_node,
                node_name=f"{volume_node.GetName()}_MetalGateMask",
                mask_kji=metal_gate_mask_kji,
            )
        if metal_in_gate_mask_kji is not None:
            self._update_labelmap_from_mask(
                reference_volume_node=volume_node,
                node_name=f"{volume_node.GetName()}_MetalInGateMask",
                mask_kji=metal_in_gate_mask_kji,
            )
        if depth_window_mask_kji is not None:
            self._update_labelmap_from_mask(
                reference_volume_node=volume_node,
                node_name=f"{volume_node.GetName()}_DepthWindowMask",
                mask_kji=depth_window_mask_kji,
            )
        if metal_depth_pass_mask_kji is not None:
            self._update_labelmap_from_mask(
                reference_volume_node=volume_node,
                node_name=f"{volume_node.GetName()}_DepthPassMetalMask",
                mask_kji=metal_depth_pass_mask_kji,
            )

        # Overlay label IDs (for troubleshooting):
        # 1=head/gating, 2=metal, 3=depth window, 4=row/column distance surface,
        # 5=metal kept after depth gate, 6=distance-map inside mask, 7=metal-in-gate.
        combo = np.zeros_like(np.asarray(metal_mask_kji, dtype=np.uint8), dtype=np.uint8)
        if head_mask_kji is not None:
            combo[np.asarray(head_mask_kji, dtype=bool)] = 1
        if distance_map_inside_mask is not None:
            combo[np.asarray(distance_map_inside_mask, dtype=bool)] = 6
        if distance_surface_mask_kji is not None:
            combo[np.asarray(distance_surface_mask_kji, dtype=bool)] = 4
        if depth_window_mask_kji is not None:
            combo[np.asarray(depth_window_mask_kji, dtype=bool)] = 3
        combo[metal_bool] = 2
        if metal_in_gate_mask_kji is not None:
            combo[np.asarray(metal_in_gate_mask_kji, dtype=bool)] = 7
        if metal_depth_pass_mask_kji is not None:
            combo[np.asarray(metal_depth_pass_mask_kji, dtype=bool)] = 5
        combo_node = self._update_labelmap_from_mask(
            reference_volume_node=volume_node,
            node_name=f"{volume_node.GetName()}_MaskOverlay",
            mask_kji=combo,
        )
        slicer.util.setSliceViewerLayers(
            background=volume_node,
            foreground=None,
            foregroundOpacity=0.0,
            label=combo_node,
            labelOpacity=0.55,
        )

    def _get_or_create_fiducial_node(self, node_name, color_rgb):
        node = None
        try:
            node = slicer.util.getNode(node_name)
        except Exception:
            node = None
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", node_name)
            node.CreateDefaultDisplayNodes()
            node.SetAttribute("Rosa.Managed", "1")
        display = node.GetDisplayNode()
        if display is not None:
            display.SetColor(float(color_rgb[0]), float(color_rgb[1]), float(color_rgb[2]))
            display.SetSelectedColor(float(color_rgb[0]), float(color_rgb[1]), float(color_rgb[2]))
            display.SetGlyphScale(1.0)
            display.SetTextScale(0.8)
            display.SetVisibility(True)
            if hasattr(display, "SetPointLabelsVisibility"):
                display.SetPointLabelsVisibility(False)
            # Keep default slice behavior: points are only shown when the
            # current slice intersects them.
            if hasattr(display, "SetSliceProjection"):
                display.SetSliceProjection(False)
        return node

    def _set_fiducials_from_ras_points(self, node, points_ras):
        node.RemoveAllControlPoints()
        pts = np.asarray(points_ras, dtype=float).reshape(-1, 3)
        for p in pts:
            node.AddControlPoint(vtk.vtkVector3d(float(p[0]), float(p[1]), float(p[2])))
        node.SetLocked(True)
        return int(pts.shape[0])

    def show_blob_diagnostics(
        self,
        volume_node,
        blob_labelmap_kji=None,
        blob_centroids_all_ras=None,
        blob_centroids_kept_ras=None,
        blob_centroids_rejected_ras=None,
    ):
        """Create/update blob debug overlays and centroid markups."""
        if blob_labelmap_kji is not None and np.asarray(blob_labelmap_kji).size > 0:
            self._update_labelmap_from_mask(
                reference_volume_node=volume_node,
                node_name=f"{volume_node.GetName()}_BlobLabelMap",
                mask_kji=blob_labelmap_kji,
            )
        all_node = self._get_or_create_fiducial_node(
            f"{volume_node.GetName()}_BlobCentroids_All",
            color_rgb=(1.0, 1.0, 0.0),
        )
        kept_node = self._get_or_create_fiducial_node(
            f"{volume_node.GetName()}_BlobCentroids_Kept",
            color_rgb=(0.1, 0.9, 0.1),
        )
        rej_node = self._get_or_create_fiducial_node(
            f"{volume_node.GetName()}_BlobCentroids_Rejected",
            color_rgb=(0.95, 0.25, 0.25),
        )
        self._set_fiducials_from_ras_points(all_node, blob_centroids_all_ras)
        self._set_fiducials_from_ras_points(kept_node, blob_centroids_kept_ras)
        self._set_fiducials_from_ras_points(rej_node, blob_centroids_rejected_ras)

    @staticmethod
    def _build_detection_diagnostics(result, params):
        """Build stable diagnostics payload for logs/tests/debug export."""
        profile = dict(result.get("profile_ms", {}) or {})
        summary = {
            "candidate_mode": str((params or {}).get("candidate_mode", "voxel")),
            "candidate_points_total": int(result.get("candidate_points_total", 0)),
            "candidate_points_after_mask": int(result.get("candidate_points_after_mask", 0)),
            "candidate_points_after_depth": int(result.get("candidate_points_after_depth", 0)),
            "effective_min_inliers": int(result.get("effective_min_inliers", 0)),
            "effective_inlier_radius_mm": float(result.get("effective_inlier_radius_mm", 0.0)),
            "blob_count_total": int(result.get("blob_count_total", 0)),
            "blob_count_kept": int(result.get("blob_count_kept", 0)),
            "blob_reject_small": int(result.get("blob_reject_small", 0)),
            "blob_reject_large": int(result.get("blob_reject_large", 0)),
            "blob_reject_intensity": int(result.get("blob_reject_intensity", 0)),
            "blob_reject_shape": int(result.get("blob_reject_shape", 0)),
            "fit1_lines_proposed": int(result.get("fit1_lines_proposed", 0)),
            "fit2_lines_kept": int(result.get("fit2_lines_kept", 0)),
            "rescue_lines_kept": int(result.get("rescue_lines_kept", 0)),
            "final_lines_kept": int(result.get("final_lines_kept", len(result.get("lines", []) or []))),
            "assigned_points_after_refine": int(result.get("assigned_points_after_refine", 0)),
            "unassigned_points_after_refine": int(result.get("unassigned_points_after_refine", 0)),
            "rescued_points": int(result.get("rescued_points", 0)),
            "final_unassigned_points": int(result.get("final_unassigned_points", 0)),
            "gap_reject_count": int(result.get("gap_reject_count", 0)),
            "duplicate_reject_count": int(result.get("duplicate_reject_count", 0)),
            "start_zone_reject_count": int(result.get("start_zone_reject_count", 0)),
            "length_reject_count": int(result.get("length_reject_count", 0)),
            "inlier_reject_count": int(result.get("inlier_reject_count", 0)),
            "profile_ms": {
                "setup": float(profile.get("setup", 0.0)),
                "subsample": float(profile.get("subsample", 0.0)),
                "depth_map": float(profile.get("depth_map", 0.0)),
                "ras_convert": float(profile.get("ras_convert", 0.0)),
                "blob_stage": float(profile.get("blob_stage", 0.0)),
                "exclude": float(profile.get("exclude", 0.0)),
                "fit1_stage": float(profile.get("first_pass", 0.0)),
                "fit2_stage": float(profile.get("refine", 0.0)),
                "rescue_stage": float(profile.get("rescue", 0.0)),
                "total": float(profile.get("total", 0.0)),
            },
            "params": dict(params or {}),
        }
        lines = list(result.get("lines", []) or [])
        summary["lines"] = [
            {
                "index": int(i + 1),
                "start_ras": list(line.get("start_ras", [0.0, 0.0, 0.0])),
                "end_ras": list(line.get("end_ras", [0.0, 0.0, 0.0])),
                "length_mm": float(line.get("length_mm", 0.0)),
                "inlier_count": int(line.get("inlier_count", 0)),
                "support_weight": float(line.get("support_weight", 0.0)),
                "rms_mm": float(line.get("rms_mm", 0.0)),
                "inside_fraction": float(line.get("inside_fraction", 0.0)),
                "depth_span_mm": float(line.get("depth_span_mm", 0.0)),
                "best_model_id": str(line.get("best_model_id", "")),
                "best_model_score": (
                    None if line.get("best_model_score", None) is None else float(line.get("best_model_score"))
                ),
            }
            for i, line in enumerate(lines)
        ]
        return summary

    def write_detection_diagnostics_json(self, volume_node, diagnostics):
        """Persist debug diagnostics JSON under /tmp for troubleshooting."""
        out_dir = os.path.join("/tmp", "rosa_postopct_debug")
        os.makedirs(out_dir, exist_ok=True)
        safe_name = str(volume_node.GetName() or "volume").replace(os.sep, "_")
        out_path = os.path.join(out_dir, f"{safe_name}_denovo_diagnostics.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2)
        return out_path

    def _vtk_matrix_to_numpy(self, vtk_matrix4x4):
        out = np.eye(4, dtype=float)
        for r in range(4):
            for c in range(4):
                out[r, c] = float(vtk_matrix4x4.GetElement(r, c))
        return out

    def extract_threshold_candidates_lps(self, volume_node, threshold, max_points=300000):
        arr = slicer.util.arrayFromVolume(volume_node)  # K,J,I
        idx = np.argwhere(arr >= float(threshold))
        if idx.size == 0:
            return np.empty((0, 3), dtype=float)
        if idx.shape[0] > int(max_points):
            rng = np.random.default_rng(0)
            keep = rng.choice(idx.shape[0], size=int(max_points), replace=False)
            idx = idx[keep]
        n = idx.shape[0]
        ijk_h = np.ones((n, 4), dtype=float)
        ijk_h[:, 0] = idx[:, 2]
        ijk_h[:, 1] = idx[:, 1]
        ijk_h[:, 2] = idx[:, 0]
        m_vtk = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(m_vtk)
        m = self._vtk_matrix_to_numpy(m_vtk)
        ras = (ijk_h @ m.T)[:, :3]
        lps = ras.copy()
        lps[:, 0] *= -1.0
        lps[:, 1] *= -1.0
        return lps

    def show_volume_in_all_slice_views(self, volume_node):
        if volume_node is None:
            return
        volume_id = volume_node.GetID()
        app_logic = slicer.app.applicationLogic() if hasattr(slicer.app, "applicationLogic") else None
        if app_logic is not None:
            sel = app_logic.GetSelectionNode()
            if sel is not None:
                sel.SetReferenceActiveVolumeID(volume_id)
                app_logic.PropagateVolumeSelection(0)
        for composite in slicer.util.getNodesByClass("vtkMRMLSliceCompositeNode"):
            composite.SetBackgroundVolumeID(volume_id)
        bounds = [0.0] * 6
        volume_node.GetRASBounds(bounds)
        cx = 0.5 * (bounds[0] + bounds[1])
        cy = 0.5 * (bounds[2] + bounds[3])
        cz = 0.5 * (bounds[4] + bounds[5])
        lm = slicer.app.layoutManager()
        if lm is None:
            return
        for view_name in ("Red", "Yellow", "Green"):
            widget = lm.sliceWidget(view_name)
            if widget is None:
                continue
            logic = widget.sliceLogic()
            if logic is not None:
                logic.FitSliceToAll()
            slice_node = widget.mrmlSliceNode()
            if slice_node is not None:
                try:
                    slice_node.JumpSliceByCentering(cx, cy, cz)
                except Exception:
                    pass

    def apply_ct_window_from_threshold(self, volume_node, threshold):
        display = volume_node.GetDisplayNode()
        if display is None:
            return
        lower = float(threshold) - 250.0
        upper = float(threshold) + 2200.0
        display.AutoWindowLevelOff()
        display.SetWindow(max(upper - lower, 1.0))
        display.SetLevel((upper + lower) * 0.5)

    def reset_ct_window(self, volume_node):
        display = volume_node.GetDisplayNode()
        if display is not None:
            display.AutoWindowLevelOn()

    def reset_standard_slice_views(self):
        lm = slicer.app.layoutManager()
        if lm is None:
            return
        layout_node = lm.layoutLogic().GetLayoutNode()
        if layout_node is not None:
            layout_node.SetViewArrangement(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        for view_name, orientation in (("Red", "Axial"), ("Yellow", "Sagittal"), ("Green", "Coronal")):
            widget = lm.sliceWidget(view_name)
            if widget is None:
                continue
            slice_node = widget.mrmlSliceNode()
            if slice_node:
                slice_node.SetOrientation(orientation)
            logic = widget.sliceLogic()
            if logic:
                logic.FitSliceToAll()

    def _collect_trajectories_from_role(self, role, workflow_node=None):
        wf = workflow_node or self.workflow_state.resolve_or_create_workflow_node()
        trajectories = []
        for node in self.workflow_state.role_nodes(role, workflow_node=wf):
            traj = self.trajectory_scene.trajectory_from_line_node("", node)
            if traj is not None:
                trajectories.append(traj)
        trajectories.sort(key=lambda item: item.get("name", ""))
        return trajectories

    def collect_trajectories_by_source(self, source_key="working", workflow_node=None):
        wf = workflow_node or self.workflow_state.resolve_or_create_workflow_node()
        source = str(source_key or "working").strip().lower()

        if source == "working":
            trajectories = self._collect_trajectories_from_role("WorkingTrajectoryLines", workflow_node=wf)
            if trajectories:
                return trajectories
            rows = self.trajectory_scene.collect_working_trajectory_rows(
                groups=["imported_rosa", "imported_external", "manual", "guided_fit", "de_novo"]
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

        role_map = {
            "imported_rosa": "ImportedTrajectoryLines",
            "imported_external": "ImportedExternalTrajectoryLines",
            "guided_fit": "GuidedFitTrajectoryLines",
            "de_novo": "DeNovoTrajectoryLines",
            "planned_rosa": "PlannedTrajectoryLines",
        }
        if source in role_map:
            return self._collect_trajectories_from_role(role_map[source], workflow_node=wf)

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

    def _ijk_kji_to_ras_points(self, volume_node, ijk_kji):
        # Accept list or ndarray; normalize to (N,3) array in KJI order.
        idx = np.asarray(ijk_kji, dtype=float).reshape(-1, 3)
        ijk = np.zeros_like(idx, dtype=float)
        ijk[:, 0] = idx[:, 2]
        ijk[:, 1] = idx[:, 1]
        ijk[:, 2] = idx[:, 0]
        ijk_h = np.concatenate([ijk, np.ones((ijk.shape[0], 1), dtype=float)], axis=1)
        m_vtk = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(m_vtk)
        m = self._vtk_matrix_to_numpy(m_vtk)
        ras_h = (m @ ijk_h.T).T
        return ras_h[:, :3]

    def _ras_to_ijk_float(self, volume_node, ras_xyz):
        ras_h = np.array([float(ras_xyz[0]), float(ras_xyz[1]), float(ras_xyz[2]), 1.0], dtype=float)
        m_vtk = vtk.vtkMatrix4x4()
        volume_node.GetRASToIJKMatrix(m_vtk)
        m = self._vtk_matrix_to_numpy(m_vtk)
        ijk = m @ ras_h
        return ijk[:3]

    def _volume_center_ras(self, volume_node):
        image_data = volume_node.GetImageData()
        dims = image_data.GetDimensions()
        center_ijk = np.array([0.5 * (dims[0] - 1), 0.5 * (dims[1] - 1), 0.5 * (dims[2] - 1), 1.0], dtype=float)
        m_vtk = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(m_vtk)
        m = self._vtk_matrix_to_numpy(m_vtk)
        return (m @ center_ijk)[:3]

    def detect_de_novo(
        self,
        volume_node,
        threshold,
        max_points,
        max_lines,
        inlier_radius_mm,
        min_length_mm,
        min_inliers,
        ransac_iterations,
        use_head_mask,
        head_mask_threshold_hu,
        head_mask_method,
        min_metal_depth_mm,
        max_metal_depth_mm,
        start_zone_window_mm,
        candidate_mode="voxel",
        min_blob_voxels=2,
        max_blob_voxels=1200,
        min_blob_peak_hu=None,
        max_blob_elongation=None,
        enable_rescue_pass=True,
        rescue_min_inliers_scale=0.6,
        rescue_max_lines=6,
        use_model_score=True,
        min_model_score=0.10,
        models_by_id=None,
        pipeline_key=None,
    ):
        config = dict(
            threshold=float(threshold),
            max_points=int(max_points),
            max_lines=int(max_lines),
            inlier_radius_mm=float(inlier_radius_mm),
            min_length_mm=float(min_length_mm),
            min_inliers=int(min_inliers),
            ransac_iterations=int(ransac_iterations),
            use_head_mask=bool(use_head_mask),
            build_head_mask=bool(use_head_mask),
            head_mask_threshold_hu=float(head_mask_threshold_hu),
            head_mask_aggressive_cleanup=False,
            head_mask_close_mm=2.0,
            head_mask_method=str(head_mask_method or "outside_air"),
            head_mask_metal_dilate_mm=1.0,
            head_gate_erode_vox=1,
            head_gate_dilate_vox=1,
            head_gate_margin_mm=0.0,
            min_metal_depth_mm=float(min_metal_depth_mm),
            max_metal_depth_mm=float(max_metal_depth_mm),
            candidate_mode=str(candidate_mode or "voxel"),
            min_blob_voxels=int(min_blob_voxels),
            max_blob_voxels=int(max_blob_voxels),
            min_blob_peak_hu=min_blob_peak_hu,
            max_blob_elongation=max_blob_elongation,
            enable_rescue_pass=bool(enable_rescue_pass),
            rescue_min_inliers_scale=float(rescue_min_inliers_scale),
            rescue_max_lines=int(rescue_max_lines),
            min_model_score=(float(min_model_score) if bool(use_model_score) else None),
            pipeline_key=str(pipeline_key or self.default_de_novo_pipeline_key),
        )
        ctx_extras = {}
        if bool(use_model_score) and isinstance(models_by_id, dict) and models_by_id:
            ctx_extras["models_by_id"] = models_by_id
        ctx = {
            "run_id": f"denovo_{volume_node.GetName()}",
            "arr_kji": slicer.util.arrayFromVolume(volume_node),
            "spacing_xyz": volume_node.GetSpacing(),
            "ijk_kji_to_ras_fn": lambda idx: self._ijk_kji_to_ras_points(volume_node, idx),
            "ras_to_ijk_fn": lambda ras: self._ras_to_ijk_float(volume_node, ras),
            "center_ras": self._volume_center_ras(volume_node),
            "config": config,
            "extras": ctx_extras,
        }
        selected_pipeline_key = str(config.get("pipeline_key") or self.default_de_novo_pipeline_key)
        available = set(self.pipeline_registry.keys())
        if selected_pipeline_key not in available:
            selected_pipeline_key = self.default_de_novo_pipeline_key
        engine_result = self.pipeline_registry.run(selected_pipeline_key, ctx)
        if str(engine_result.get("status", "ok")).lower() == "error":
            error = dict(engine_result.get("error") or {})
            stage = str(error.get("stage") or "pipeline")
            message = str(error.get("message") or "Detection pipeline failed")
            raise RuntimeError(f"{message} (stage={stage}, pipeline={selected_pipeline_key})")
        return self._engine_result_to_legacy_detection_result(engine_result, ctx_extras, config)

    def _next_side_names(self, lines, existing_names, midline_x_ras=0.0):
        used = set(existing_names)
        r_count = max([int(n[1:]) for n in used if len(n) == 3 and n.startswith("R") and n[1:].isdigit()] or [0])
        l_count = max([int(n[1:]) for n in used if len(n) == 3 and n.startswith("L") and n[1:].isdigit()] or [0])
        names = []
        for line in lines:
            mid_x = 0.5 * (float(line["start_ras"][0]) + float(line["end_ras"][0]))
            if mid_x >= float(midline_x_ras):
                r_count += 1
                name = f"R{r_count:02d}"
            else:
                l_count += 1
                name = f"L{l_count:02d}"
            while name in used:
                if name.startswith("R"):
                    r_count += 1
                    name = f"R{r_count:02d}"
                else:
                    l_count += 1
                    name = f"L{l_count:02d}"
            used.add(name)
            names.append(name)
        return names

    def upsert_detected_lines(self, lines, replace_existing=True):
        existing_rows = self.trajectory_scene.collect_working_trajectory_rows(groups=["de_novo"])
        if bool(replace_existing):
            for row in existing_rows:
                node = slicer.mrmlScene.GetNodeByID(row["node_id"])
                if node is not None:
                    slicer.mrmlScene.RemoveNode(node)
            existing_rows = []

        existing_names = [row["name"] for row in existing_rows]
        names = self._next_side_names(lines, existing_names=existing_names, midline_x_ras=0.0)
        new_rows = list(existing_rows)
        for i, line in enumerate(lines):
            name = names[i]
            node_name = self.trajectory_scene.build_node_name(name, "de_novo")
            node = self.trajectory_scene.create_or_update_trajectory_line(
                name=name,
                start_ras=line["start_ras"],
                end_ras=line["end_ras"],
                node_id=None,
                group="de_novo",
                origin="postop_ct_denovo",
                node_name=node_name,
            )
            new_rows.append(
                {
                    "name": name,
                    "node_name": node.GetName() or node_name,
                    "node_id": node.GetID(),
                    "group": "de_novo",
                    "start_ras": [float(line["start_ras"][0]), float(line["start_ras"][1]), float(line["start_ras"][2])],
                    "end_ras": [float(line["end_ras"][0]), float(line["end_ras"][1]), float(line["end_ras"][2])],
                }
            )
        new_rows.sort(key=lambda r: r["name"])
        return new_rows

    def publish_working_rows(self, rows, workflow_node=None, role="WorkingTrajectoryLines", source="postop_ct_localization"):
        nodes = []
        for row in rows:
            node = slicer.mrmlScene.GetNodeByID(row.get("node_id", ""))
            if node is not None:
                nodes.append(node)
        if role and role != "WorkingTrajectoryLines":
            self.workflow_publish.publish_nodes(
                role=role,
                nodes=nodes,
                source=source,
                space_name="ROSA_BASE",
                workflow_node=workflow_node,
            )
        self.workflow_publish.publish_nodes(
            role="WorkingTrajectoryLines",
            nodes=nodes,
            source=source,
            space_name="ROSA_BASE",
            workflow_node=workflow_node,
        )
        wf = workflow_node or self.workflow_state.resolve_or_create_workflow_node()
        self.trajectory_scene.place_trajectory_nodes_in_hierarchy(
            context_id=self.workflow_state.context_id(workflow_node=wf),
            nodes=nodes,
        )
