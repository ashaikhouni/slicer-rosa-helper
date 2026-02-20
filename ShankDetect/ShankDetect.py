"""3D Slicer module for CT-only electrode shank/trajectory detection.

Workflow:
- detect linear shank artifacts in a postop CT volume
- create editable trajectory line markups
- assign electrode models per trajectory
- generate contact fiducials from selected models
"""

import math
import os
import re
import sys
import inspect
from collections import OrderedDict

try:
    import numpy as np
except ImportError:
    np = None
try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

from __main__ import ctk, qt, slicer, vtk
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
)

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
SIBLING_ROSA_LIB = os.path.join(os.path.dirname(MODULE_DIR), "RosaHelper", "Lib")
LOCAL_LIB = os.path.join(MODULE_DIR, "Lib")
for path in [LOCAL_LIB, SIBLING_ROSA_LIB]:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

from rosa_core import (
    generate_contacts,
    load_electrode_library,
    lps_to_ras_point,
    model_map,
    suggest_model_id_for_trajectory,
    trajectory_length_mm,
)
from shank_core import masking as shank_masking
from shank_core import pipeline as shank_pipeline


class ShankDetect(ScriptedLoadableModule):
    """Slicer metadata for CT shank detection module."""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "Shank Detect"
        self.parent.categories = ["ROSA"]
        self.parent.dependencies = []
        self.parent.contributors = ["Ammar Shaikhouni", "Codex"]
        self.parent.helpText = (
            "Detect electrode shank trajectories directly from CT artifact, then assign "
            "electrode models and generate contacts."
        )


class ShankDetectWidget(ScriptedLoadableModuleWidget):
    """Widget for CT artifact-based trajectory detection and contact generation."""

    def setup(self):
        super().setup()

        self.logic = ShankDetectLogic()
        self.modelsById = {}
        self.modelIds = []
        self._updatingTable = False
        self._syncingThresholdWidgets = False
        self._headPreviewPass = 0
        self._headPreviewKey = None
        self._lastMaskResult = None
        self._cleanupLegacyDepthPlotNodes()

        self.statusText = qt.QPlainTextEdit()
        self.statusText.setReadOnly(True)
        self.statusText.setMaximumBlockCount(2000)

        form = qt.QFormLayout()
        self.layout.addLayout(form)

        self.ctSelector = slicer.qMRMLNodeComboBox()
        self.ctSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.ctSelector.noneEnabled = True
        self.ctSelector.addEnabled = False
        self.ctSelector.removeEnabled = False
        self.ctSelector.setMRMLScene(slicer.mrmlScene)
        self.ctSelector.setToolTip("Postop CT volume with electrode artifact")
        form.addRow("CT volume", self.ctSelector)

        self.thresholdSpin = qt.QDoubleSpinBox()
        self.thresholdSpin.setRange(600.0, 4000.0)
        self.thresholdSpin.setDecimals(1)
        self.thresholdSpin.setValue(1800.0)
        self.thresholdSpin.setSingleStep(25.0)
        self.thresholdSpin.setSuffix(" HU")
        self.thresholdSlider = qt.QSlider(qt.Qt.Horizontal)
        self.thresholdSlider.setRange(600, 4000)
        self.thresholdSlider.setSingleStep(25)
        self.thresholdSlider.setPageStep(100)
        self.thresholdSlider.setValue(1800)
        self.autoMetalThresholdButton = qt.QPushButton("Auto")
        self.autoMetalThresholdButton.setToolTip("Suggest metal threshold from full CT histogram.")
        metalRow = qt.QHBoxLayout()
        metalRow.addWidget(self.thresholdSpin)
        metalRow.addWidget(self.thresholdSlider, 1)
        metalRow.addWidget(self.autoMetalThresholdButton)
        form.addRow("Metal threshold", metalRow)

        self.inlierRadiusSpin = qt.QDoubleSpinBox()
        self.inlierRadiusSpin.setRange(0.2, 6.0)
        self.inlierRadiusSpin.setDecimals(2)
        self.inlierRadiusSpin.setValue(1.2)
        self.inlierRadiusSpin.setSingleStep(0.1)
        self.inlierRadiusSpin.setSuffix(" mm")
        form.addRow("Inlier radius", self.inlierRadiusSpin)

        self.minLengthSpin = qt.QDoubleSpinBox()
        self.minLengthSpin.setRange(5.0, 200.0)
        self.minLengthSpin.setDecimals(1)
        self.minLengthSpin.setValue(20.0)
        self.minLengthSpin.setSingleStep(1.0)
        self.minLengthSpin.setSuffix(" mm")
        form.addRow("Min trajectory length", self.minLengthSpin)

        self.minInliersSpin = qt.QSpinBox()
        self.minInliersSpin.setRange(25, 20000)
        self.minInliersSpin.setValue(250)
        form.addRow("Min inliers", self.minInliersSpin)

        self.maxPointsSpin = qt.QSpinBox()
        self.maxPointsSpin.setRange(5000, 1000000)
        self.maxPointsSpin.setSingleStep(5000)
        self.maxPointsSpin.setValue(300000)
        form.addRow("Max sampled points", self.maxPointsSpin)

        self.ransacIterationsSpin = qt.QSpinBox()
        self.ransacIterationsSpin.setRange(20, 2000)
        self.ransacIterationsSpin.setValue(240)
        form.addRow("RANSAC iterations", self.ransacIterationsSpin)

        self.useHeadMaskCheck = qt.QCheckBox("Use head mask filter")
        self.useHeadMaskCheck.setChecked(True)
        form.addRow(self.useHeadMaskCheck)

        self.headMaskThresholdSpin = qt.QDoubleSpinBox()
        self.headMaskThresholdSpin.setRange(-1200.0, 1000.0)
        self.headMaskThresholdSpin.setDecimals(1)
        self.headMaskThresholdSpin.setValue(-500.0)
        self.headMaskThresholdSpin.setSingleStep(25.0)
        self.headMaskThresholdSpin.setSuffix(" HU")
        self.headMaskThresholdSlider = qt.QSlider(qt.Qt.Horizontal)
        self.headMaskThresholdSlider.setRange(-1200, 1000)
        self.headMaskThresholdSlider.setSingleStep(25)
        self.headMaskThresholdSlider.setPageStep(100)
        self.headMaskThresholdSlider.setValue(-500)
        self.previewHeadMaskButton = qt.QPushButton("Refine + Fill")
        self.previewHeadMaskButton.setToolTip(
            "Run full head-mask refinement (closing + fill holes). Repeated clicks increase closing radius to "
            "progressively bridge residual holes."
        )
        self.resetHeadMaskButton = qt.QPushButton("Reset Refine")
        self.resetHeadMaskButton.setToolTip(
            "Reset refinement pass counter and clear head-mask overlay. Use this before trying a new threshold."
        )
        headRow = qt.QHBoxLayout()
        headRow.addWidget(self.headMaskThresholdSpin)
        headRow.addWidget(self.headMaskThresholdSlider, 1)
        headRow.addWidget(self.previewHeadMaskButton)
        headRow.addWidget(self.resetHeadMaskButton)
        form.addRow("Head mask threshold", headRow)

        self.headMaskCloseSpin = qt.QDoubleSpinBox()
        self.headMaskCloseSpin.setRange(0.0, 15.0)
        self.headMaskCloseSpin.setDecimals(1)
        self.headMaskCloseSpin.setValue(2.0)
        self.headMaskCloseSpin.setSingleStep(0.5)
        self.headMaskCloseSpin.setSuffix(" mm")
        form.addRow("Head mask closing", self.headMaskCloseSpin)

        self.headMaskMethodCombo = qt.QComboBox()
        self.headMaskMethodCombo.addItem("Legacy cleanup", "legacy")
        self.headMaskMethodCombo.addItem("Tissue-cut bridge suppression", "tissue_cut")
        self.headMaskMethodCombo.addItem("Tissue-cut (no closing)", "tissue_cut_noclose")
        self.headMaskMethodCombo.addItem("Outside-air distance", "outside_air")
        self.headMaskMethodCombo.setCurrentIndex(3)
        self.headMaskMethodCombo.setToolTip(
            "Choose head-mask construction method. Use tissue-cut (no closing) to benchmark speed/point-count impact."
        )
        form.addRow("Head mask method", self.headMaskMethodCombo)

        self.headMaskMetalDilateSpin = qt.QDoubleSpinBox()
        self.headMaskMetalDilateSpin.setRange(0.0, 5.0)
        self.headMaskMetalDilateSpin.setDecimals(2)
        self.headMaskMetalDilateSpin.setValue(1.0)
        self.headMaskMetalDilateSpin.setSingleStep(0.25)
        self.headMaskMetalDilateSpin.setSuffix(" mm")
        self.headMaskMetalDilateSpin.setToolTip(
            "Only for tissue-cut method: dilate metal before subtraction to break external wire bridges."
        )
        form.addRow("Metal bridge dilate", self.headMaskMetalDilateSpin)

        self.minMetalDepthSpin = qt.QDoubleSpinBox()
        self.minMetalDepthSpin.setRange(0.0, 100.0)
        self.minMetalDepthSpin.setDecimals(2)
        self.minMetalDepthSpin.setValue(5.0)
        self.minMetalDepthSpin.setSingleStep(0.25)
        self.minMetalDepthSpin.setSuffix(" mm")
        self.minMetalDepthSpin.setToolTip("Minimum depth from outer head surface for metal points.")
        form.addRow("Min metal depth", self.minMetalDepthSpin)

        self.maxMetalDepthSpin = qt.QDoubleSpinBox()
        self.maxMetalDepthSpin.setRange(0.0, 300.0)
        self.maxMetalDepthSpin.setDecimals(2)
        self.maxMetalDepthSpin.setValue(220.0)
        self.maxMetalDepthSpin.setSingleStep(0.5)
        self.maxMetalDepthSpin.setSuffix(" mm")
        self.maxMetalDepthSpin.setToolTip("Maximum depth from outer head surface for metal points.")
        form.addRow("Max metal depth", self.maxMetalDepthSpin)

        self.aggressiveHeadCleanupCheck = qt.QCheckBox("Aggressive 3-plane island cleanup")
        self.aggressiveHeadCleanupCheck.setChecked(False)
        self.aggressiveHeadCleanupCheck.setToolTip(
            "Use stronger island removal (axial/coronal/sagittal). Slower but may improve difficult masks."
        )
        form.addRow(self.aggressiveHeadCleanupCheck)

        self.useModelScoreCheck = qt.QCheckBox("Enable model-template scoring")
        self.useModelScoreCheck.setChecked(True)
        form.addRow(self.useModelScoreCheck)

        self.modelScoreMinSpin = qt.QDoubleSpinBox()
        self.modelScoreMinSpin.setRange(-20.0, 20.0)
        self.modelScoreMinSpin.setDecimals(2)
        self.modelScoreMinSpin.setValue(0.10)
        self.modelScoreMinSpin.setSingleStep(0.05)
        form.addRow("Min model score", self.modelScoreMinSpin)

        self.showMasksCheck = qt.QCheckBox("Show metal/head masks in slice views")
        self.showMasksCheck.setChecked(True)
        form.addRow(self.showMasksCheck)

        self.showInMaskPointsCheck = qt.QCheckBox("Show in-mask candidate voxels (red)")
        self.showInMaskPointsCheck.setChecked(True)
        self.showInMaskPointsCheck.setToolTip(
            "Show the metal voxels kept after depth/head gating as a red overlay."
        )
        self.maxInMaskPointsSpin = qt.QSpinBox()
        self.maxInMaskPointsSpin.setRange(500, 30000)
        self.maxInMaskPointsSpin.setValue(2500)
        self.maxInMaskPointsSpin.setSingleStep(500)
        self.maxInMaskPointsSpin.setToolTip(
            "Displayed red candidates are hard-capped internally to keep Slicer responsive."
        )
        inMaskRow = qt.QHBoxLayout()
        inMaskRow.addWidget(self.showInMaskPointsCheck)
        inMaskRow.addWidget(qt.QLabel("Max shown"))
        inMaskRow.addWidget(self.maxInMaskPointsSpin)
        form.addRow(inMaskRow)

        self.useExactCountCheck = qt.QCheckBox("Use exact trajectory count")
        self.useExactCountCheck.setChecked(False)
        self.useExactCountCheck.toggled.connect(self._onUseExactCountToggled)
        form.addRow(self.useExactCountCheck)

        self.exactCountSpin = qt.QSpinBox()
        self.exactCountSpin.setRange(2, 30)
        self.exactCountSpin.setValue(10)
        self.exactCountSpin.setEnabled(False)
        form.addRow("Exact count", self.exactCountSpin)

        detectRow = qt.QHBoxLayout()
        self.previewMasksButton = qt.QPushButton("Preview Masks")
        self.previewMasksButton.clicked.connect(self.onPreviewMasksClicked)
        detectRow.addWidget(self.previewMasksButton)
        self.detectButton = qt.QPushButton("Detect Trajectories")
        self.detectButton.clicked.connect(self.onDetectClicked)
        detectRow.addWidget(self.detectButton)
        self.resetViewsButton = qt.QPushButton("Reset Ax/Cor/Sag")
        self.resetViewsButton.clicked.connect(self.onResetViewsClicked)
        detectRow.addWidget(self.resetViewsButton)
        detectRow.addStretch(1)
        form.addRow(detectRow)

        self.trajectorySection = ctk.ctkCollapsibleButton()
        self.trajectorySection.text = "Detected Trajectories"
        self.trajectorySection.collapsed = False
        self.layout.addWidget(self.trajectorySection)

        trajLayout = qt.QFormLayout(self.trajectorySection)
        self.trajectoryTable = qt.QTableWidget()
        self.trajectoryTable.setColumnCount(7)
        self.trajectoryTable.setHorizontalHeaderLabels(
            [
                "Lock",
                "Name",
                "Length (mm)",
                "Auto model",
                "Score",
                "Electrode model",
                "Elec length (mm)",
            ]
        )
        self.trajectoryTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.trajectoryTable.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.trajectoryTable.horizontalHeader().setSectionResizeMode(0, qt.QHeaderView.ResizeToContents)
        self.trajectoryTable.horizontalHeader().setSectionResizeMode(1, qt.QHeaderView.ResizeToContents)
        self.trajectoryTable.horizontalHeader().setSectionResizeMode(2, qt.QHeaderView.ResizeToContents)
        self.trajectoryTable.horizontalHeader().setSectionResizeMode(3, qt.QHeaderView.ResizeToContents)
        self.trajectoryTable.horizontalHeader().setSectionResizeMode(4, qt.QHeaderView.ResizeToContents)
        self.trajectoryTable.horizontalHeader().setSectionResizeMode(5, qt.QHeaderView.Stretch)
        self.trajectoryTable.horizontalHeader().setSectionResizeMode(6, qt.QHeaderView.ResizeToContents)
        self.trajectoryTable.itemChanged.connect(self.onTableItemChanged)
        self.trajectoryTable.itemSelectionChanged.connect(self.onTrajectorySelectionChanged)
        trajLayout.addRow(self.trajectoryTable)

        applyRow = qt.QHBoxLayout()
        self.defaultModelCombo = qt.QComboBox()
        self.defaultModelCombo.setMinimumContentsLength(14)
        applyRow.addWidget(self.defaultModelCombo)
        self.applyModelAllButton = qt.QPushButton("Apply model to all")
        self.applyModelAllButton.clicked.connect(self.onApplyModelAllClicked)
        applyRow.addWidget(self.applyModelAllButton)
        applyRow.addStretch(1)
        trajLayout.addRow("Default model", applyRow)

        self.contactPrefixEdit = qt.QLineEdit("CTShankContacts")
        trajLayout.addRow("Contacts node prefix", self.contactPrefixEdit)

        self.generateContactsButton = qt.QPushButton("Generate Contacts")
        self.generateContactsButton.setEnabled(False)
        self.generateContactsButton.clicked.connect(self.onGenerateContactsClicked)
        trajLayout.addRow(self.generateContactsButton)

        self._metalPreviewTimer = qt.QTimer()
        self._metalPreviewTimer.setSingleShot(True)
        self._metalPreviewTimer.setInterval(150)
        self._metalPreviewTimer.timeout.connect(self.onPreviewMetalThreshold)

        self._headPreviewTimer = qt.QTimer()
        self._headPreviewTimer.setSingleShot(True)
        self._headPreviewTimer.setInterval(180)
        self._headPreviewTimer.timeout.connect(self.onPreviewHeadMaskFast)

        self.thresholdSpin.valueChanged.connect(self._onMetalThresholdSpinChanged)
        self.thresholdSlider.valueChanged.connect(self._onMetalThresholdSliderChanged)
        self.autoMetalThresholdButton.clicked.connect(self.onAutoSuggestMetalThresholdClicked)
        self.headMaskThresholdSpin.valueChanged.connect(self._onHeadThresholdSpinChanged)
        self.headMaskThresholdSlider.valueChanged.connect(self._onHeadThresholdSliderChanged)
        self.headMaskMethodCombo.currentIndexChanged.connect(self._onHeadMaskMethodChanged)
        self.previewHeadMaskButton.clicked.connect(self.onPreviewHeadMaskClicked)
        self.resetHeadMaskButton.clicked.connect(self.onResetHeadMaskPreviewClicked)

        self.layout.addWidget(self.statusText)
        self.layout.addStretch(1)

        self._loadElectrodeLibrary()
        self._onHeadMaskMethodChanged(self.headMaskMethodCombo.currentIndex)

    def log(self, msg):
        """Append status message to UI and console."""
        self.statusText.appendPlainText(msg)
        print(msg)

    def _cleanupLegacyDepthPlotNodes(self):
        """Remove legacy depth-curve plot nodes from older module versions."""
        legacy_prefixes = (
            "ShankDetectDepthPlotView",
            "_DepthCurveTable",
            "_DepthCurveChart",
            "_DepthCurveAll",
            "_DepthCurveKept",
        )
        to_remove = []
        for node in slicer.util.getNodes("*").values():
            name = str(node.GetName() or "")
            if any(name == p or name.endswith(p) for p in legacy_prefixes):
                to_remove.append(node)
        for node in to_remove:
            slicer.mrmlScene.RemoveNode(node)

    def _syncThresholdWidgets(self, spin, slider, value):
        """Set linked spin/slider value without recursive signal loops."""
        if self._syncingThresholdWidgets:
            return
        self._syncingThresholdWidgets = True
        try:
            spin.blockSignals(True)
            slider.blockSignals(True)
            spin.setValue(float(value))
            slider.setValue(int(round(float(value))))
        finally:
            slider.blockSignals(False)
            spin.blockSignals(False)
            self._syncingThresholdWidgets = False

    def _onMetalThresholdSpinChanged(self, value):
        """Sync metal threshold slider and trigger debounced preview."""
        self._syncThresholdWidgets(self.thresholdSpin, self.thresholdSlider, value)
        self._scheduleMetalPreview()

    def _onMetalThresholdSliderChanged(self, value):
        """Sync metal threshold spin and trigger debounced preview."""
        self._syncThresholdWidgets(self.thresholdSpin, self.thresholdSlider, value)
        self._scheduleMetalPreview()

    def _onHeadThresholdSpinChanged(self, value):
        """Sync head-threshold slider to spin value."""
        self._syncThresholdWidgets(self.headMaskThresholdSpin, self.headMaskThresholdSlider, value)
        self._headPreviewPass = 0
        self._headPreviewKey = None
        self._scheduleHeadPreview()

    def _onHeadThresholdSliderChanged(self, value):
        """Sync head-threshold spin to slider value."""
        self._syncThresholdWidgets(self.headMaskThresholdSpin, self.headMaskThresholdSlider, value)
        self._headPreviewPass = 0
        self._headPreviewKey = None
        self._scheduleHeadPreview()

    def _currentHeadMaskMethod(self):
        """Return normalized head-mask method from UI combo."""
        data = None
        try:
            data_attr = self.headMaskMethodCombo.currentData
            data = data_attr() if callable(data_attr) else data_attr
        except Exception:
            data = None
        method = str(data).strip().lower() if data else "legacy"
        if method not in ("legacy", "tissue_cut", "tissue_cut_noclose", "outside_air"):
            method = "outside_air"
        return method

    def _onHeadMaskMethodChanged(self, _index):
        """Toggle method-specific controls and invalidate cached preview state."""
        method = self._currentHeadMaskMethod()
        is_tissue_cut = str(method) in ("tissue_cut", "tissue_cut_noclose")
        self.headMaskMetalDilateSpin.setEnabled(is_tissue_cut)
        self.headMaskCloseSpin.setEnabled(str(method) not in ("tissue_cut_noclose", "outside_air"))
        self._headPreviewPass = 0
        self._headPreviewKey = None
        self._scheduleHeadPreview()

    def _scheduleMetalPreview(self):
        """Debounced fast metal-only mask preview for threshold tuning."""
        if self.ctSelector.currentNode() is None:
            return
        self._metalPreviewTimer.start()

    def _scheduleHeadPreview(self):
        """Debounced fast head-mask-only preview for threshold tuning."""
        if self.ctSelector.currentNode() is None:
            return
        self._headPreviewTimer.start()

    def onAutoSuggestMetalThresholdClicked(self):
        """Estimate metal threshold from CT histogram and apply it to controls."""
        volume_node = self.ctSelector.currentNode()
        if volume_node is None:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Shank Detect", "Select a CT volume first.")
            return
        try:
            suggested = self.logic.suggest_metal_threshold_hu(
                volume_node=volume_node,
                head_threshold_hu=float(self.headMaskThresholdSpin.value),
                head_close_mm=float(self.headMaskCloseSpin.value),
                head_aggressive_cleanup=bool(self.aggressiveHeadCleanupCheck.checked),
            )
        except Exception as exc:
            self.log(f"[threshold] auto suggestion failed: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Shank Detect", str(exc))
            return
        self._syncThresholdWidgets(self.thresholdSpin, self.thresholdSlider, suggested)
        self.log(f"[threshold] suggested metal threshold: {float(suggested):.1f} HU")
        self._scheduleMetalPreview()

    def onPreviewMetalThreshold(self):
        """Fast preview of thresholded metal mask for live threshold tuning."""
        volume_node = self.ctSelector.currentNode()
        if volume_node is None:
            return
        try:
            arr_kji = slicer.util.arrayFromVolume(volume_node)
            threshold = float(self.thresholdSpin.value)
            metal_mask = np.asarray(arr_kji >= threshold, dtype=np.uint8)
            self.logic.show_metal_and_head_masks(
                volume_node=volume_node,
                metal_mask_kji=metal_mask,
                head_mask_kji=None,
            )
        except Exception as exc:
            self.log(f"[metal] preview failed: {exc}")

    def onPreviewHeadMaskClicked(self):
        """Preview head-mask threshold tuning without running full detection pipeline."""
        volume_node = self.ctSelector.currentNode()
        if volume_node is None:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Shank Detect", "Select a CT volume first.")
            return
        try:
            key = (
                str(volume_node.GetID()),
                round(float(self.headMaskThresholdSpin.value), 2),
                round(float(self.headMaskCloseSpin.value), 2),
                self._currentHeadMaskMethod(),
                round(float(self.headMaskMetalDilateSpin.value), 2),
                round(float(self.thresholdSpin.value), 2),
            )
            if key == self._headPreviewKey:
                self._headPreviewPass += 1
            else:
                self._headPreviewKey = key
                self._headPreviewPass = 0

            close_mm = float(self.headMaskCloseSpin.value) + float(self._headPreviewPass) * 1.0
            arr_kji = slicer.util.arrayFromVolume(volume_node)
            head_mask = self.logic._build_head_mask_kji(
                volume_node=volume_node,
                arr_kji=arr_kji,
                threshold_hu=float(self.headMaskThresholdSpin.value),
                close_mm=close_mm,
                aggressive_cleanup=bool(self.aggressiveHeadCleanupCheck.checked),
                method=self._currentHeadMaskMethod(),
                metal_threshold_hu=float(self.thresholdSpin.value),
                metal_dilate_mm=float(self.headMaskMetalDilateSpin.value),
            )
            metal_mask = np.asarray(arr_kji >= float(self.thresholdSpin.value), dtype=np.uint8)
            self.logic.show_metal_and_head_masks(
                volume_node=volume_node,
                metal_mask_kji=metal_mask,
                head_mask_kji=head_mask.astype(np.uint8),
            )
            self.log(
                f"[head] preview threshold={float(self.headMaskThresholdSpin.value):.1f} HU "
                f"close={close_mm:.1f}mm pass={int(self._headPreviewPass)} "
                f"voxels={int(np.count_nonzero(head_mask))}"
            )
        except Exception as exc:
            self.log(f"[head] preview failed: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Shank Detect", str(exc))

    def onResetHeadMaskPreviewClicked(self):
        """Reset iterative head-mask preview and clear head-mask overlay."""
        self._headPreviewPass = 0
        self._headPreviewKey = None
        volume_node = self.ctSelector.currentNode()
        if volume_node is None:
            return
        try:
            arr_kji = slicer.util.arrayFromVolume(volume_node)
            metal_mask = np.asarray(arr_kji >= float(self.thresholdSpin.value), dtype=np.uint8)
            self.logic.show_metal_and_head_masks(
                volume_node=volume_node,
                metal_mask_kji=metal_mask,
                head_mask_kji=None,
            )
            self.log("[head] reset preview pass and cleared head-mask overlay")
        except Exception as exc:
            self.log(f"[head] reset failed: {exc}")

    def onPreviewHeadMaskFast(self):
        """Fast live head-mask preview while slider is moving (threshold-only)."""
        volume_node = self.ctSelector.currentNode()
        if volume_node is None:
            return
        try:
            arr_kji = slicer.util.arrayFromVolume(volume_node)
            head_mask = np.asarray(arr_kji >= float(self.headMaskThresholdSpin.value), dtype=np.uint8)
            metal_mask = np.asarray(arr_kji >= float(self.thresholdSpin.value), dtype=np.uint8)
            self.logic.show_metal_and_head_masks(
                volume_node=volume_node,
                metal_mask_kji=metal_mask,
                head_mask_kji=head_mask,
            )
        except Exception as exc:
            self.log(f"[head] fast preview failed: {exc}")

    def _onUseExactCountToggled(self, checked):
        self.exactCountSpin.setEnabled(bool(checked))

    def _loadElectrodeLibrary(self):
        """Load DIXI electrode library used for per-trajectory assignment."""
        try:
            lib = load_electrode_library()
            self.modelsById = model_map(lib)
            self.modelIds = sorted(self.modelsById.keys())
        except Exception as exc:
            self.modelsById = {}
            self.modelIds = []
            self.log(f"[electrodes] failed to load library: {exc}")
            return

        self.defaultModelCombo.clear()
        self.defaultModelCombo.addItem("")
        for model_id in self.modelIds:
            self.defaultModelCombo.addItem(model_id)
        self.log(f"[electrodes] loaded {len(self.modelIds)} models")

    def _buildModelCombo(self):
        combo = qt.QComboBox()
        combo.addItem("")
        for model_id in self.modelIds:
            combo.addItem(model_id)
        return combo

    def _setReadOnlyItem(self, row, col, text):
        item = qt.QTableWidgetItem(str(text))
        item.setFlags(item.flags() & ~qt.Qt.ItemIsEditable)
        self.trajectoryTable.setItem(row, col, item)

    def _modelLengthMm(self, model_id):
        model = self.modelsById.get(model_id, {})
        return float(model.get("total_exploration_length_mm", 0.0))

    def _bindModelLengthUpdate(self, combo, row):
        def _update(_unused=None, row_index=row):
            model_id = combo.currentText.strip()
            text = ""
            if model_id:
                text = f"{self._modelLengthMm(model_id):.2f}"
            self._setReadOnlyItem(row_index, 6, text)

        if hasattr(combo, "currentTextChanged"):
            combo.currentTextChanged.connect(_update)
        else:
            combo.currentIndexChanged.connect(_update)

    def _lineNodeFromRow(self, row):
        name_item = self.trajectoryTable.item(row, 1)
        if name_item is None:
            return None
        node_id = name_item.data(qt.Qt.UserRole)
        if not node_id:
            return None
        return slicer.mrmlScene.GetNodeByID(node_id)

    def _rowTrajectory(self, row):
        node = self._lineNodeFromRow(row)
        if node is None or node.GetNumberOfControlPoints() < 2:
            return None
        p0 = [0.0, 0.0, 0.0]
        p1 = [0.0, 0.0, 0.0]
        node.GetNthControlPointPositionWorld(0, p0)
        node.GetNthControlPointPositionWorld(1, p1)
        return {
            "row": row,
            "node_id": node.GetID(),
            "name": (self.trajectoryTable.item(row, 1).text() or node.GetName()).strip(),
            "start_ras": [float(p0[0]), float(p0[1]), float(p0[2])],
            "end_ras": [float(p1[0]), float(p1[1]), float(p1[2])],
            "length_mm": trajectory_length_mm({"start": p0, "end": p1}),
            "locked": self.trajectoryTable.item(row, 0).checkState() == qt.Qt.Checked,
            "auto_model_id": self.trajectoryTable.item(row, 3).text().strip()
            if self.trajectoryTable.item(row, 3)
            else "",
            "auto_model_score": float(self.trajectoryTable.item(row, 4).text())
            if self.trajectoryTable.item(row, 4) and self.trajectoryTable.item(row, 4).text().strip()
            else None,
            "model_id": self.trajectoryTable.cellWidget(row, 5).currentText.strip()
            if self.trajectoryTable.cellWidget(row, 5)
            else "",
        }

    def _collectTrajectoriesFromTable(self):
        rows = []
        for row in range(self.trajectoryTable.rowCount):
            info = self._rowTrajectory(row)
            if info is not None:
                rows.append(info)
        return rows

    def _nextSiteAwareNames(self, lines, existing_names, midline_x_ras=0.0):
        """Assign side-aware names (R##/L##) using a midline X in RAS."""
        used = set(existing_names)
        counts = {"R": 0, "L": 0}
        pattern = re.compile(r"^([RL])(\d+)$", re.IGNORECASE)
        for name in existing_names:
            m = pattern.match(name.strip())
            if not m:
                continue
            side = m.group(1).upper()
            counts[side] = max(counts[side], int(m.group(2)))

        assigned = []
        for line in lines:
            mid_x = 0.5 * (float(line["start_ras"][0]) + float(line["end_ras"][0]))
            side = "R" if mid_x >= float(midline_x_ras) else "L"
            idx = counts[side] + 1
            name = f"{side}{idx:02d}"
            while name in used:
                idx += 1
                name = f"{side}{idx:02d}"
            counts[side] = idx
            used.add(name)
            assigned.append(name)
        return assigned

    def _populateTrajectoryTable(self, trajectories):
        """Populate/editable trajectory table from scene trajectory rows."""
        self._updatingTable = True
        self.trajectoryTable.setRowCount(0)

        for row, traj in enumerate(trajectories):
            self.trajectoryTable.insertRow(row)

            lock_item = qt.QTableWidgetItem("")
            lock_item.setFlags(lock_item.flags() | qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
            lock_item.setCheckState(qt.Qt.Checked if traj.get("locked") else qt.Qt.Unchecked)
            self.trajectoryTable.setItem(row, 0, lock_item)

            name_item = qt.QTableWidgetItem(traj["name"])
            name_item.setFlags(name_item.flags() | qt.Qt.ItemIsEditable)
            name_item.setData(qt.Qt.UserRole, traj["node_id"])
            self.trajectoryTable.setItem(row, 1, name_item)

            self._setReadOnlyItem(row, 2, f"{float(traj['length_mm']):.2f}")
            self._setReadOnlyItem(row, 3, traj.get("auto_model_id", ""))
            score = traj.get("auto_model_score", None)
            self._setReadOnlyItem(row, 4, f"{float(score):.2f}" if score is not None else "")

            model_combo = self._buildModelCombo()
            self.trajectoryTable.setCellWidget(row, 5, model_combo)
            model_id = traj.get("model_id", "").strip()
            if not model_id:
                model_id = (traj.get("auto_model_id", "") or "").strip()
            if not model_id:
                model_id = suggest_model_id_for_trajectory(
                    {"start": traj["start_ras"], "end": traj["end_ras"]},
                    self.modelsById,
                    model_ids=self.modelIds,
                    tolerance_mm=5.0,
                )
            if model_id:
                idx = model_combo.findText(model_id)
                if idx >= 0:
                    model_combo.setCurrentIndex(idx)
            self._bindModelLengthUpdate(model_combo, row)
            self._setReadOnlyItem(row, 6, "")
            if model_combo.currentText.strip():
                self._setReadOnlyItem(row, 6, f"{self._modelLengthMm(model_combo.currentText.strip()):.2f}")

            node = slicer.mrmlScene.GetNodeByID(traj.get("node_id", ""))
            if node is not None and node.IsA("vtkMRMLMarkupsLineNode"):
                display = node.GetDisplayNode()
                if display is not None and hasattr(display, "SetPropertiesLabelVisibility"):
                    display.SetPropertiesLabelVisibility(False)

        self._updatingTable = False
        has_rows = self.trajectoryTable.rowCount > 0
        self.generateContactsButton.setEnabled(has_rows)
        self.applyModelAllButton.setEnabled(has_rows)

    def onTableItemChanged(self, item):
        """Handle in-table edits (lock toggle and trajectory rename)."""
        if self._updatingTable or item is None:
            return
        if item.column() != 1:
            return

        row = item.row()
        node = self._lineNodeFromRow(row)
        if node is None:
            return

        new_name = item.text().strip()
        if not new_name:
            self._updatingTable = True
            item.setText(node.GetName())
            self._updatingTable = False
            return

        for other_row in range(self.trajectoryTable.rowCount):
            if other_row == row:
                continue
            other_item = self.trajectoryTable.item(other_row, 1)
            if other_item and other_item.text().strip() == new_name:
                self.log(f"[name] duplicate name '{new_name}' is not allowed")
                self._updatingTable = True
                item.setText(node.GetName())
                self._updatingTable = False
                return

        node.SetName(new_name)

    def onTrajectorySelectionChanged(self):
        """Align Red slice (long-axis) to selected trajectory row."""
        row = self.trajectoryTable.currentRow()
        if row < 0:
            return
        traj = self._rowTrajectory(row)
        if traj is None:
            return

        try:
            self.logic.align_slice_to_trajectory(
                start_ras=traj["start_ras"],
                end_ras=traj["end_ras"],
                slice_view="Red",
                mode="long",
            )
            self.log(f"[slice] aligned Red view to {traj['name']} (long)")
        except Exception as exc:
            self.log(f"[slice] alignment warning: {exc}")

    def onResetViewsClicked(self):
        """Reset Red/Yellow/Green orientations to standard Ax/Cor/Sag."""
        try:
            self.logic.reset_standard_slice_views()
            self.log("[slice] reset to Axial/Coronal/Sagittal")
        except Exception as exc:
            self.log(f"[slice] reset warning: {exc}")

    def onApplyModelAllClicked(self):
        """Apply selected default electrode model to all trajectories."""
        model_id = self.defaultModelCombo.currentText.strip()
        if not model_id:
            return
        for row in range(self.trajectoryTable.rowCount):
            combo = self.trajectoryTable.cellWidget(row, 5)
            if combo is None:
                continue
            idx = combo.findText(model_id)
            if idx >= 0:
                combo.setCurrentIndex(idx)
                self._setReadOnlyItem(row, 6, f"{self._modelLengthMm(model_id):.2f}")

    def _collectDetectionSettings(self):
        """Read detection UI settings into one dictionary."""
        return {
            "threshold": float(self.thresholdSpin.value),
            "inlier_radius_mm": float(self.inlierRadiusSpin.value),
            "min_length_mm": float(self.minLengthSpin.value),
            "min_inliers": int(self.minInliersSpin.value),
            "max_points": int(self.maxPointsSpin.value),
            "ransac_iterations": int(self.ransacIterationsSpin.value),
            "use_head_mask": bool(self.useHeadMaskCheck.checked),
            "head_mask_threshold_hu": float(self.headMaskThresholdSpin.value),
            "head_mask_close_mm": float(self.headMaskCloseSpin.value),
            "head_mask_method": self._currentHeadMaskMethod(),
            "head_mask_metal_dilate_mm": float(self.headMaskMetalDilateSpin.value),
            "min_metal_depth_mm": float(self.minMetalDepthSpin.value),
            "max_metal_depth_mm": float(self.maxMetalDepthSpin.value),
            "head_mask_aggressive_cleanup": bool(self.aggressiveHeadCleanupCheck.checked),
            "use_model_score": bool(self.useModelScoreCheck.checked),
            "min_model_score": float(self.modelScoreMinSpin.value),
            "show_masks": bool(self.showMasksCheck.checked),
            "show_in_mask_points": bool(self.showInMaskPointsCheck.checked),
            "max_in_mask_points": int(self.maxInMaskPointsSpin.value),
        }

    def _displayMasksFromResult(self, volume_node, result):
        """Render metal/head masks and log head-depth gating stats."""
        metal = result.get("metal_mask_kji")
        gating = result.get("head_mask_kji")
        self.logic.show_metal_and_head_masks(
            volume_node=volume_node,
            metal_mask_kji=metal,
            head_mask_kji=gating,
        )
        mask_type = str(result.get("gating_mask_type", "none"))
        candidate_count = int(result.get("candidate_count", 0))
        mask_kept = int(result.get("head_mask_kept_count", candidate_count))
        metal_vox = int(np.count_nonzero(metal)) if metal is not None else 0
        gating_vox = int(np.count_nonzero(gating)) if gating is not None else 0
        inside_method = str(result.get("inside_method", "unknown"))
        metal_in_head = int(result.get("metal_in_head_count", mask_kept))
        depth_kept = int(result.get("depth_kept_count", mask_kept))
        self.log(
            f"[mask] displayed mask overlay (head + metal); "
            f"mask={mask_type}, in-mask={mask_kept}/{candidate_count}, "
            f"inside_method={inside_method}, method={result.get('profile_flags', {}).get('head_mask_method', 'legacy')}, "
            f"metal_in_head={metal_in_head}, depth_kept={depth_kept}, "
            f"voxels(metal={metal_vox}, gating={gating_vox})"
        )

    def _displayInMaskPointsFromResult(self, volume_node, result, max_points=6000):
        """Render in-mask candidate voxels as red overlay for QC."""
        in_mask_ijk_kji = result.get("in_mask_ijk_kji")
        if in_mask_ijk_kji is None:
            self.logic.clear_in_mask_points_node(volume_node)
            return
        shown, total = self.logic.show_in_mask_points_node(
            volume_node=volume_node,
            in_mask_ijk_kji=in_mask_ijk_kji,
            max_points=max_points,
        )
        self.log(f"[mask] in-mask candidate voxels shown: {shown}/{total}")

    def onPreviewMasksClicked(self):
        """Compute and show raw head-distance mask construction without trajectory fitting."""
        volume_node = self.ctSelector.currentNode()
        if volume_node is None:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Shank Detect", "Select a CT volume first.")
            return

        settings = self._collectDetectionSettings()
        try:
            result = self.logic.preview_masks(
                volume_node=volume_node,
                threshold=settings["threshold"],
                use_head_mask=settings["use_head_mask"],
                build_head_mask=settings["use_head_mask"],
                head_mask_threshold_hu=settings["head_mask_threshold_hu"],
                head_mask_aggressive_cleanup=settings["head_mask_aggressive_cleanup"],
                head_mask_close_mm=settings["head_mask_close_mm"],
                head_mask_method=settings["head_mask_method"],
                head_mask_metal_dilate_mm=settings["head_mask_metal_dilate_mm"],
                min_metal_depth_mm=settings["min_metal_depth_mm"],
                max_metal_depth_mm=settings["max_metal_depth_mm"],
            )
        except Exception as exc:
            self.log(f"[mask] failed: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Shank Detect", str(exc))
            return

        try:
            self._displayMasksFromResult(volume_node, result)
        except Exception as exc:
            self.log(f"[mask] visualization warning: {exc}")
        self._lastMaskResult = result
        try:
            if settings["show_in_mask_points"]:
                self._displayInMaskPointsFromResult(
                    volume_node,
                    result,
                    max_points=settings["max_in_mask_points"],
                )
            else:
                self.logic.clear_in_mask_points_node(volume_node)
        except Exception as exc:
            self.log(f"[mask] in-mask points warning: {exc}")
        profile = result.get("profile_ms", {})
        if profile:
            self.log(
                "[profile] preview "
                f"total={float(profile.get('total', 0.0)):.1f}ms "
                f"(threshold={float(profile.get('threshold', 0.0)):.1f}, "
                f"head_mask={float(profile.get('head_mask', 0.0)):.1f}, "
                f"distance={float(profile.get('distance_map', 0.0)):.1f}, "
                f"enum={float(profile.get('candidate_enum', 0.0)):.1f}, "
                f"gate={float(profile.get('head_gate', 0.0)):.1f}, "
                f"depth={float(profile.get('depth_gate', 0.0)):.1f})"
            )
        flags = result.get("profile_flags", {})
        if flags:
            self.log(
                "[profile] preview cache "
                f"head_cache_hit={bool(flags.get('head_cache_hit', False))} "
                f"gating={bool(flags.get('used_precomputed_gating_mask', False))} "
                f"distance={bool(flags.get('used_precomputed_distance_map', False))}"
            )
        self.log("[depth] preview complete")

    def onDetectClicked(self):
        """Detect CT shank trajectories and populate editable table."""
        volume_node = self.ctSelector.currentNode()
        if volume_node is None:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Shank Detect", "Select a CT volume first.")
            return

        rows = self._collectTrajectoriesFromTable()
        locked = [row for row in rows if row.get("locked")]
        unlocked = [row for row in rows if not row.get("locked")]

        if unlocked:
            self.logic.remove_nodes_by_ids([row["node_id"] for row in unlocked])

        use_exact = bool(self.useExactCountCheck.checked)
        if use_exact:
            target_total = int(self.exactCountSpin.value)
            if len(locked) > target_total:
                self.log(
                    f"[detect] {len(locked)} locked trajectories already exceed exact target {target_total}. "
                    "Unlock some rows or increase exact target."
                )
                self._populateTrajectoryTable(locked)
                return
            max_new = target_total - len(locked)
        else:
            max_new = max(0, 30 - len(locked))

        if max_new <= 0:
            self.log("[detect] no free slots to detect new trajectories (max reached by locked rows)")
            self._populateTrajectoryTable(locked)
            return

        settings = self._collectDetectionSettings()
        self.log("[detect] running...")
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        slicer.app.processEvents()

        exclude_segments = [
            {"start_ras": row["start_ras"], "end_ras": row["end_ras"]}
            for row in locked
            if row.get("start_ras") is not None
        ]

        try:
            result = self.logic.detect_trajectory_lines(
                volume_node=volume_node,
                threshold=settings["threshold"],
                max_points=settings["max_points"],
                max_lines=max_new,
                inlier_radius_mm=settings["inlier_radius_mm"],
                min_length_mm=settings["min_length_mm"],
                min_inliers=settings["min_inliers"],
                ransac_iterations=settings["ransac_iterations"],
                exclude_segments=exclude_segments,
                exclude_radius_mm=max(2.0, settings["inlier_radius_mm"] * 1.25),
                use_head_mask=settings["use_head_mask"],
                build_head_mask=settings["use_head_mask"],
                head_mask_threshold_hu=settings["head_mask_threshold_hu"],
                head_mask_aggressive_cleanup=settings["head_mask_aggressive_cleanup"],
                head_mask_close_mm=settings["head_mask_close_mm"],
                head_mask_method=settings["head_mask_method"],
                head_mask_metal_dilate_mm=settings["head_mask_metal_dilate_mm"],
                min_metal_depth_mm=settings["min_metal_depth_mm"],
                max_metal_depth_mm=settings["max_metal_depth_mm"],
                models_by_id=self.modelsById if settings["use_model_score"] else None,
                min_model_score=settings["min_model_score"] if settings["use_model_score"] else None,
            )
        except Exception as exc:
            self.log(f"[detect] failed: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Shank Detect", str(exc))
            return
        finally:
            qt.QApplication.restoreOverrideCursor()
            slicer.app.processEvents()

        detected = result.get("lines", [])
        candidate_count = int(result.get("candidate_count", 0))
        mask_kept = int(result.get("head_mask_kept_count", candidate_count))
        mask_type = str(result.get("gating_mask_type", "none"))
        inside_method = str(result.get("inside_method", "unknown"))
        metal_in_head = int(result.get("metal_in_head_count", mask_kept))
        depth_kept = int(result.get("depth_kept_count", mask_kept))
        gap_reject = int(result.get("gap_reject_count", 0))
        dup_reject = int(result.get("duplicate_reject_count", 0))
        self.log(
            f"[detect] candidates: {candidate_count}, in-mask: {mask_kept}, "
            f"mask={mask_type}, inside_method={inside_method}, "
            f"metal_in_head={metal_in_head}, depth_kept={depth_kept}, "
            f"gap_reject={gap_reject}, dup_reject={dup_reject}, "
            f"new lines: {len(detected)}"
        )

        if settings["show_masks"]:
            try:
                self._displayMasksFromResult(volume_node, result)
            except Exception as exc:
                self.log(f"[mask] visualization warning: {exc}")
        self._lastMaskResult = result
        try:
            if settings["show_in_mask_points"]:
                self._displayInMaskPointsFromResult(
                    volume_node,
                    result,
                    max_points=settings["max_in_mask_points"],
                )
            else:
                self.logic.clear_in_mask_points_node(volume_node)
        except Exception as exc:
            self.log(f"[mask] in-mask points warning: {exc}")
        profile = result.get("profile_ms", {})
        if profile:
            p_prev = profile.get("preview", {}) if isinstance(profile.get("preview"), dict) else {}
            p_det = profile.get("detect", {}) if isinstance(profile.get("detect"), dict) else {}
            self.log(
                "[profile] detect "
                f"total={float(profile.get('total', 0.0)):.1f}ms "
                f"(preview_stage={float(profile.get('preview_stage', 0.0)):.1f}, "
                f"detect_stage={float(profile.get('detect_stage', 0.0)):.1f})"
            )
            self.log(
                "[profile] detect detail "
                f"preview(head_mask={float(p_prev.get('head_mask', 0.0)):.1f}, "
                f"distance={float(p_prev.get('distance_map', 0.0)):.1f}) "
                f"fit(first_pass={float(p_det.get('first_pass', 0.0)):.1f}, "
                f"refine={float(p_det.get('refine', 0.0)):.1f})"
            )
        flags = result.get("profile_flags", {})
        if flags:
            self.log(
                "[profile] detect cache "
                f"head_cache_hit={bool(flags.get('head_cache_hit', False))} "
                f"gating={bool(flags.get('used_precomputed_gating_mask', False))} "
                f"distance={bool(flags.get('used_precomputed_distance_map', False))}"
            )
        self.log("[depth] detect complete")

        existing_names = [row["name"] for row in locked]
        midline_x = 0.0
        try:
            head_mask = result.get("head_mask_kji")
            if head_mask is not None:
                center_ras = self.logic.mask_centroid_ras(volume_node, head_mask)
                if center_ras is not None:
                    midline_x = float(center_ras[0])
        except Exception as exc:
            self.log(f"[name] midline estimate warning: {exc}")
        self.log(f"[name] side split midline X (RAS): {midline_x:.2f} mm")
        new_names = self._nextSiteAwareNames(detected, existing_names, midline_x_ras=midline_x)

        new_rows = []
        for idx, line in enumerate(detected):
            name = new_names[idx]
            node = self.logic.create_or_update_trajectory_line(
                name=name,
                start_ras=line["start_ras"],
                end_ras=line["end_ras"],
                node_id=None,
            )
            new_rows.append(
                {
                    "name": name,
                    "node_id": node.GetID(),
                    "start_ras": line["start_ras"],
                    "end_ras": line["end_ras"],
                    "length_mm": float(line["length_mm"]),
                    "auto_model_id": line.get("best_model_id", ""),
                    "auto_model_score": line.get("best_model_score", None),
                    "locked": False,
                    "model_id": "",
                }
            )

        combined = []
        for row in locked:
            node = slicer.mrmlScene.GetNodeByID(row["node_id"])
            if node is None:
                continue
            p0 = [0.0, 0.0, 0.0]
            p1 = [0.0, 0.0, 0.0]
            node.GetNthControlPointPositionWorld(0, p0)
            node.GetNthControlPointPositionWorld(1, p1)
            combined.append(
                {
                    "name": row["name"],
                    "node_id": row["node_id"],
                    "start_ras": [float(p0[0]), float(p0[1]), float(p0[2])],
                    "end_ras": [float(p1[0]), float(p1[1]), float(p1[2])],
                    "length_mm": trajectory_length_mm({"start": p0, "end": p1}),
                    "auto_model_id": row.get("auto_model_id", ""),
                    "auto_model_score": row.get("auto_model_score", None),
                    "locked": True,
                    "model_id": row.get("model_id", ""),
                }
            )
        combined.extend(new_rows)
        combined = sorted(combined, key=lambda r: r["name"])
        self._populateTrajectoryTable(combined)

        if use_exact and len(combined) != int(self.exactCountSpin.value):
            self.log(
                f"[detect] warning: exact target {int(self.exactCountSpin.value)} not met, "
                f"found {len(combined)} total trajectories"
            )

        if self.trajectoryTable.rowCount > 0:
            self.trajectoryTable.selectRow(0)

    def onGenerateContactsClicked(self):
        """Generate contact fiducials for assigned trajectories."""
        rows = self._collectTrajectoriesFromTable()
        if not rows:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Shank Detect", "No trajectories available.")
            return

        trajectories_lps = []
        assignments_rows = []
        missing_models = []
        for row in rows:
            model_id = row.get("model_id", "")
            if not model_id:
                continue
            if model_id not in self.modelsById:
                missing_models.append(model_id)
                continue

            trajectories_lps.append(
                {
                    "name": row["name"],
                    "start": lps_to_ras_point(row["start_ras"]),
                    "end": lps_to_ras_point(row["end_ras"]),
                }
            )
            assignments_rows.append(
                {
                    "trajectory": row["name"],
                    "model_id": model_id,
                    "tip_at": "target",
                    "tip_shift_mm": 0.0,
                    "xyz_offset_mm": [0.0, 0.0, 0.0],
                }
            )

        if missing_models:
            qt.QMessageBox.critical(
                slicer.util.mainWindow(),
                "Shank Detect",
                "Unknown model IDs in table: " + ", ".join(sorted(set(missing_models))),
            )
            return

        if not assignments_rows:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "Shank Detect",
                "Assign at least one electrode model before generating contacts.",
            )
            return

        assignments = {"schema_version": "1.0", "assignments": assignments_rows}

        try:
            contacts = generate_contacts(trajectories_lps, self.modelsById, assignments)
        except Exception as exc:
            self.log(f"[contacts] generation failed: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Shank Detect", str(exc))
            return

        prefix = self.contactPrefixEdit.text.strip() or "CTShankContacts"
        nodes = self.logic.create_contacts_fiducials_nodes_by_trajectory(contacts, node_prefix=prefix)
        self.log(f"[contacts] generated {len(contacts)} contact points across {len(nodes)} nodes")


class ShankDetectLogic(ScriptedLoadableModuleLogic):
    """Logic for CT artifact thresholding, line fitting, and visualization."""

    def __init__(self):
        super().__init__()
        # Cache expensive head-mask + distance-map computations by volume/settings.
        self._head_distance_cache = OrderedDict()
        self._max_head_cache_entries = 4
        # Bump when head-mask/distance algorithms change to invalidate stale cache payloads.
        self._head_cache_schema_version = 5

    def _head_cache_key(
        self,
        volume_node,
        head_mask_threshold_hu,
        head_mask_close_mm,
        head_mask_aggressive_cleanup,
        head_mask_method,
        metal_threshold_hu,
        head_mask_metal_dilate_mm,
    ):
        """Build a stable cache key for head-mask-dependent computations."""
        image_data = volume_node.GetImageData()
        dims = tuple(int(x) for x in image_data.GetDimensions()) if image_data is not None else (0, 0, 0)
        spacing = tuple(round(float(s), 6) for s in volume_node.GetSpacing())
        return (
            int(self._head_cache_schema_version),
            str(volume_node.GetID()),
            dims,
            spacing,
            round(float(head_mask_threshold_hu), 2),
            round(float(head_mask_close_mm), 2) if str(head_mask_method or "legacy") != "tissue_cut_noclose" else 0.0,
            bool(head_mask_aggressive_cleanup),
            str(head_mask_method or "legacy"),
            round(float(metal_threshold_hu), 2)
            if str(head_mask_method or "legacy") in ("tissue_cut", "tissue_cut_noclose")
            else 0.0,
            round(float(head_mask_metal_dilate_mm), 2),
        )

    def _get_head_distance_cache(
        self,
        volume_node,
        arr_kji,
        head_mask_threshold_hu,
        head_mask_close_mm,
        head_mask_aggressive_cleanup,
        head_mask_method,
        metal_threshold_hu,
        head_mask_metal_dilate_mm,
    ):
        """Return cached head gating mask + distance map, computing them if needed.

        Returns
        -------
        (cache_payload, cache_hit)
        """
        key = self._head_cache_key(
            volume_node,
            head_mask_threshold_hu=head_mask_threshold_hu,
            head_mask_close_mm=head_mask_close_mm,
            head_mask_aggressive_cleanup=head_mask_aggressive_cleanup,
            head_mask_method=head_mask_method,
            metal_threshold_hu=metal_threshold_hu,
            head_mask_metal_dilate_mm=head_mask_metal_dilate_mm,
        )
        cached = self._head_distance_cache.get(key)
        if cached is not None:
            self._head_distance_cache.move_to_end(key)
            return cached, True

        self._ensure_masking_api()
        method = str(head_mask_method or "legacy").strip().lower()
        if method == "tissue_cut_noclose":
            gating_mask, distance_surface_mask = shank_masking.build_tissue_cut_distance_and_gating_masks_kji(
                arr_kji=arr_kji,
                spacing_xyz=volume_node.GetSpacing(),
                threshold_hu=head_mask_threshold_hu,
                metal_threshold_hu=metal_threshold_hu,
                metal_dilate_mm=head_mask_metal_dilate_mm,
                aggressive_cleanup=head_mask_aggressive_cleanup,
            )
            distance_map = shank_masking.compute_head_distance_map_kji(
                distance_surface_mask,
                spacing_xyz=volume_node.GetSpacing(),
            )
        else:
            gating_mask = shank_masking.build_head_mask_kji(
                arr_kji=arr_kji,
                spacing_xyz=volume_node.GetSpacing(),
                threshold_hu=head_mask_threshold_hu,
                close_mm=head_mask_close_mm,
                aggressive_cleanup=head_mask_aggressive_cleanup,
                method=method,
                metal_threshold_hu=metal_threshold_hu,
                metal_dilate_mm=head_mask_metal_dilate_mm,
            )
            distance_map = shank_masking.compute_head_distance_map_kji(
                gating_mask,
                spacing_xyz=volume_node.GetSpacing(),
            )
        cached = {
            "gating_mask_kji": np.asarray(gating_mask, dtype=np.uint8),
            "head_distance_map_kji": np.asarray(distance_map, dtype=np.float32),
        }
        self._head_distance_cache[key] = cached
        while len(self._head_distance_cache) > int(self._max_head_cache_entries):
            self._head_distance_cache.popitem(last=False)
        return cached, False

    def _ensure_masking_api(self):
        """Ensure `shank_core.masking` exposes required API after hot-reload."""
        global shank_masking  # pylint: disable=global-statement

        import importlib

        try:
            shank_masking = importlib.reload(shank_masking)
        except Exception:
            import shank_core.masking as _masking

            shank_masking = _masking

        has_api = (
            hasattr(shank_masking, "build_preview_masks")
            and hasattr(shank_masking, "compute_head_distance_map_kji")
            and hasattr(shank_masking, "build_head_mask_kji")
            and hasattr(shank_masking, "build_tissue_cut_distance_and_gating_masks_kji")
        )
        has_precompute_args = False
        has_method_args = False
        if has_api:
            try:
                params = inspect.signature(shank_masking.build_preview_masks).parameters
                has_precompute_args = "precomputed_gating_mask_kji" in params and "precomputed_head_distance_map_kji" in params
                head_params = inspect.signature(shank_masking.build_head_mask_kji).parameters
                has_method_args = (
                    "method" in head_params
                    and "metal_threshold_hu" in head_params
                    and "metal_dilate_mm" in head_params
                )
            except Exception:
                has_precompute_args = False
                has_method_args = False
        if not has_api or not has_precompute_args or not has_method_args:
            module_path = getattr(shank_masking, "__file__", "<unknown>")
            raise RuntimeError(
                "Loaded shank_core.masking is stale and missing required APIs "
                f"(module={module_path}). Reload module paths in Slicer and restart once."
            )

    def _ensure_pipeline_api(self):
        """Ensure `shank_core.pipeline` exposes required API after hot-reload."""
        global shank_pipeline, shank_masking  # pylint: disable=global-statement

        import importlib

        try:
            # Reload masking first, then pipeline, because pipeline imports masking symbols at module import time.
            shank_masking = importlib.reload(shank_masking)
            # Reload detect core too, so pipeline binds the latest profiling + fitting code.
            detect_mod = importlib.import_module("shank_core.detect")
            importlib.reload(detect_mod)
            shank_pipeline = importlib.reload(shank_pipeline)
        except Exception:
            import shank_core.pipeline as _pipeline

            shank_pipeline = _pipeline

        has_api = hasattr(shank_pipeline, "run_detection")
        has_precompute_args = False
        has_method_args = False
        if has_api:
            try:
                params = inspect.signature(shank_pipeline.run_detection).parameters
                has_precompute_args = "precomputed_gating_mask_kji" in params and "precomputed_head_distance_map_kji" in params
                has_method_args = "head_mask_method" in params and "head_mask_metal_dilate_mm" in params
            except Exception:
                has_precompute_args = False
                has_method_args = False
        if not has_api or not has_precompute_args or not has_method_args:
            module_path = getattr(shank_pipeline, "__file__", "<unknown>")
            raise RuntimeError(
                "Loaded shank_core.pipeline is stale and missing run_detection "
                f"(module={module_path}). Reload module paths in Slicer and restart once."
            )

    def _ijk_kji_to_ras_points(self, volume_node, ijk_kji):
        """Convert `(k,j,i)` voxel indices to RAS points."""
        ijk = np.zeros_like(ijk_kji, dtype=float)
        ijk[:, 0] = ijk_kji[:, 2]
        ijk[:, 1] = ijk_kji[:, 1]
        ijk[:, 2] = ijk_kji[:, 0]

        ijk_h = np.concatenate([ijk, np.ones((ijk.shape[0], 1), dtype=float)], axis=1)

        m_vtk = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(m_vtk)
        m = np.array([[m_vtk.GetElement(r, c) for c in range(4)] for r in range(4)], dtype=float)
        ras_h = (m @ ijk_h.T).T
        return ras_h[:, :3]

    def mask_centroid_ras(self, volume_node, mask_kji):
        """Return centroid of a binary KJI mask in RAS coordinates, or None."""
        if mask_kji is None:
            return None
        idx = np.argwhere(np.asarray(mask_kji, dtype=bool))
        if idx.size == 0:
            return None
        kji_center = idx.mean(axis=0)
        ijk_h = np.array([kji_center[2], kji_center[1], kji_center[0], 1.0], dtype=float)
        m_vtk = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(m_vtk)
        m = np.array([[m_vtk.GetElement(r, c) for c in range(4)] for r in range(4)], dtype=float)
        ras = m @ ijk_h
        return ras[:3]

    def _build_head_mask_kji(
        self,
        volume_node,
        arr_kji,
        threshold_hu=-300.0,
        close_mm=2.0,
        aggressive_cleanup=True,
        method="legacy",
        metal_threshold_hu=1800.0,
        metal_dilate_mm=1.0,
    ):
        """Build largest-component head mask in KJI index order."""
        self._ensure_masking_api()
        return shank_masking.build_head_mask_kji(
            arr_kji=arr_kji,
            spacing_xyz=volume_node.GetSpacing(),
            threshold_hu=threshold_hu,
            close_mm=close_mm,
            aggressive_cleanup=aggressive_cleanup,
            method=method,
            metal_threshold_hu=metal_threshold_hu,
            metal_dilate_mm=metal_dilate_mm,
        )

    def suggest_metal_threshold_hu(
        self,
        volume_node,
        head_threshold_hu=-300.0,
        head_close_mm=2.0,
        head_aggressive_cleanup=True,
    ):
        """Suggest metal HU threshold from full CT histogram.

        Parameters are kept for API compatibility with existing callers.
        """
        self._ensure_masking_api()
        arr_kji = slicer.util.arrayFromVolume(volume_node)
        return shank_masking.suggest_metal_threshold_hu_from_array(arr_kji)

    def _get_or_create_labelmap_node(self, node_name):
        """Return existing or newly-created labelmap node by name."""
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
        """Write one binary mask array into a labelmap node aligned to reference volume."""
        if mask_kji is None:
            return None
        node = self._get_or_create_labelmap_node(node_name)
        arr = np.asarray(mask_kji, dtype=np.uint8)
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
    ):
        """Create/update head + metal mask labelmaps and overlay them in slice views."""
        if metal_mask_kji is None:
            raise ValueError("metal_mask_kji is required for mask visualization")
        metal_bool = np.asarray(metal_mask_kji, dtype=bool)
        metal_node = self._update_labelmap_from_mask(
            reference_volume_node=volume_node,
            node_name=f"{volume_node.GetName()}_MetalMask",
            mask_kji=metal_mask_kji,
        )
        head_node = None
        if head_mask_kji is not None:
            head_node = self._update_labelmap_from_mask(
                reference_volume_node=volume_node,
                node_name=f"{volume_node.GetName()}_HeadMask",
                mask_kji=head_mask_kji,
            )

        combo = np.zeros_like(np.asarray(metal_mask_kji, dtype=np.uint8), dtype=np.uint8)
        if head_mask_kji is not None:
            combo[np.asarray(head_mask_kji, dtype=bool)] = 1
        combo[metal_bool] = 2
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
        return {
            "metal_node": metal_node,
            "head_node": head_node,
            "overlay_node": combo_node,
        }

    def clear_in_mask_points_node(self, volume_node):
        """Remove in-mask overlay/legacy point preview nodes for a volume if present."""
        if volume_node is None:
            return
        for node_name in (f"{volume_node.GetName()}_InMaskPoints", f"{volume_node.GetName()}_InMaskOverlay"):
            try:
                node = slicer.util.getNode(node_name)
            except Exception:
                node = None
            if node is not None:
                slicer.mrmlScene.RemoveNode(node)

    def _get_or_create_inmask_color_node(self):
        """Return a compact color table for red in-mask voxel overlay."""
        node_name = "ShankDetectInMaskColors"
        try:
            node = slicer.util.getNode(node_name)
        except Exception:
            node = None
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode", node_name)
            node.SetTypeToUser()
            node.SetNumberOfColors(2)
            node.SetColor(0, "Background", 0.0, 0.0, 0.0, 0.0)
            node.SetColor(1, "InMask", 1.0, 0.0, 0.0, 1.0)
            node.HideFromEditorsOff()
        return node

    def show_in_mask_points_node(self, volume_node, in_mask_ijk_kji, max_points=6000):
        """Show in-mask candidate voxels as a red overlay labelmap.

        Returns
        -------
        (shown_voxels, total_candidates)
        """
        if volume_node is None:
            return (0, 0)
        if in_mask_ijk_kji is None:
            self.clear_in_mask_points_node(volume_node)
            return (0, 0)

        idx = np.asarray(in_mask_ijk_kji, dtype=int).reshape(-1, 3)
        total = int(idx.shape[0])
        if total == 0:
            self.clear_in_mask_points_node(volume_node)
            return (0, 0)

        arr_shape = tuple(int(s) for s in slicer.util.arrayFromVolume(volume_node).shape)
        mask = np.zeros(arr_shape, dtype=np.uint8)

        # Keep compatibility with existing "Max shown" UI by optional subsampling.
        max_points = max(100, int(max_points))
        use_idx = idx
        if total > max_points:
            rng = np.random.default_rng(0)
            pick = rng.choice(total, size=max_points, replace=False)
            use_idx = idx[pick]

        valid = (
            (use_idx[:, 0] >= 0)
            & (use_idx[:, 1] >= 0)
            & (use_idx[:, 2] >= 0)
            & (use_idx[:, 0] < arr_shape[0])
            & (use_idx[:, 1] < arr_shape[1])
            & (use_idx[:, 2] < arr_shape[2])
        )
        use_idx = use_idx[valid]
        if use_idx.size == 0:
            self.clear_in_mask_points_node(volume_node)
            return (0, total)
        mask[use_idx[:, 0], use_idx[:, 1], use_idx[:, 2]] = 1

        if sitk is not None:
            # Make selected voxels easier to see in slice views.
            img = sitk.GetImageFromArray(mask)
            img = sitk.BinaryDilate(img, [1, 1, 0])
            mask = sitk.GetArrayFromImage(img).astype(np.uint8)

        node = self._update_labelmap_from_mask(
            reference_volume_node=volume_node,
            node_name=f"{volume_node.GetName()}_InMaskOverlay",
            mask_kji=mask,
        )
        display = node.GetDisplayNode()
        if display:
            color = self._get_or_create_inmask_color_node()
            display.SetAndObserveColorNodeID(color.GetID())
            display.SetVisibility(True)

        slicer.util.setSliceViewerLayers(
            background=volume_node,
            foreground=None,
            foregroundOpacity=0.0,
            label=node,
            labelOpacity=0.85,
        )
        return (int(np.count_nonzero(mask)), total)

    def _ras_to_ijk_float(self, volume_node, ras_xyz):
        """Convert one RAS point to continuous IJK coordinates."""
        ras_h = np.array([float(ras_xyz[0]), float(ras_xyz[1]), float(ras_xyz[2]), 1.0], dtype=float)
        m_vtk = vtk.vtkMatrix4x4()
        volume_node.GetRASToIJKMatrix(m_vtk)
        m = np.array([[m_vtk.GetElement(r, c) for c in range(4)] for r in range(4)], dtype=float)
        ijk = m @ ras_h
        return ijk[:3]

    def preview_masks(
        self,
        volume_node,
        threshold,
        use_head_mask=False,
        build_head_mask=True,
        head_mask_threshold_hu=-500.0,
        head_mask_aggressive_cleanup=True,
        head_mask_close_mm=2.0,
        head_mask_method="outside_air",
        head_mask_metal_dilate_mm=1.0,
        min_metal_depth_mm=5.0,
        max_metal_depth_mm=220.0,
    ):
        """Build and preview head-distance masks only (no line fitting)."""
        self._ensure_masking_api()
        arr_kji = slicer.util.arrayFromVolume(volume_node)
        precomputed_gating_mask_kji = None
        precomputed_head_distance_map_kji = None
        cache_hit = False
        if bool(use_head_mask) or bool(build_head_mask):
            cached, cache_hit = self._get_head_distance_cache(
                volume_node=volume_node,
                arr_kji=arr_kji,
                head_mask_threshold_hu=head_mask_threshold_hu,
                head_mask_close_mm=head_mask_close_mm,
                head_mask_aggressive_cleanup=head_mask_aggressive_cleanup,
                head_mask_method=head_mask_method,
                metal_threshold_hu=threshold,
                head_mask_metal_dilate_mm=head_mask_metal_dilate_mm,
            )
            precomputed_gating_mask_kji = cached["gating_mask_kji"]
            precomputed_head_distance_map_kji = cached["head_distance_map_kji"]
        result = shank_masking.build_preview_masks(
            arr_kji=arr_kji,
            spacing_xyz=volume_node.GetSpacing(),
            threshold=threshold,
            use_head_mask=use_head_mask,
            build_head_mask=build_head_mask,
            head_mask_threshold_hu=head_mask_threshold_hu,
            head_mask_aggressive_cleanup=head_mask_aggressive_cleanup,
            head_mask_close_mm=head_mask_close_mm,
            head_mask_method=head_mask_method,
            head_mask_metal_dilate_mm=head_mask_metal_dilate_mm,
            min_metal_depth_mm=min_metal_depth_mm,
            max_metal_depth_mm=max_metal_depth_mm,
            precomputed_gating_mask_kji=precomputed_gating_mask_kji,
            precomputed_head_distance_map_kji=precomputed_head_distance_map_kji,
        )
        result_flags = result.get("profile_flags") if isinstance(result.get("profile_flags"), dict) else {}
        result_flags["head_cache_hit"] = bool(cache_hit)
        result["profile_flags"] = result_flags
        ijk_kji = result.get("in_mask_ijk_kji")
        if ijk_kji is None or int(ijk_kji.shape[0]) == 0:
            in_mask_points_ras = np.empty((0, 3), dtype=float)
        else:
            in_mask_points_ras = self._ijk_kji_to_ras_points(volume_node, ijk_kji)
        result["in_mask_points_ras"] = in_mask_points_ras
        result["lines"] = []
        result.setdefault("gap_reject_count", 0)
        result.setdefault("duplicate_reject_count", 0)
        return result

    def _volume_center_ras(self, volume_node):
        """Approximate volume center in RAS from voxel extents."""
        image_data = volume_node.GetImageData()
        dims = image_data.GetDimensions()
        center_ijk = np.array([0.5 * (dims[0] - 1), 0.5 * (dims[1] - 1), 0.5 * (dims[2] - 1), 1.0])
        m_vtk = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(m_vtk)
        m = np.array([[m_vtk.GetElement(r, c) for c in range(4)] for r in range(4)], dtype=float)
        ras = m @ center_ijk
        return ras[:3]

    def detect_trajectory_lines(
        self,
        volume_node,
        threshold,
        max_points=300000,
        max_lines=30,
        inlier_radius_mm=1.2,
        min_length_mm=20.0,
        min_inliers=250,
        ransac_iterations=240,
        exclude_segments=None,
        exclude_radius_mm=2.0,
        use_head_mask=False,
        build_head_mask=False,
        head_mask_threshold_hu=-500.0,
        head_mask_aggressive_cleanup=True,
        head_mask_close_mm=2.0,
        head_mask_method="outside_air",
        head_mask_metal_dilate_mm=1.0,
        min_metal_depth_mm=5.0,
        max_metal_depth_mm=220.0,
        models_by_id=None,
        min_model_score=None,
    ):
        """Detect multiple linear trajectories in CT threshold candidates."""
        if np is None:
            raise RuntimeError("numpy is required for trajectory detection")

        self._ensure_masking_api()
        self._ensure_pipeline_api()

        arr_kji = slicer.util.arrayFromVolume(volume_node)
        center_ras = self._volume_center_ras(volume_node)
        precomputed_gating_mask_kji = None
        precomputed_head_distance_map_kji = None
        cache_hit = False
        if bool(use_head_mask) or bool(build_head_mask):
            cached, cache_hit = self._get_head_distance_cache(
                volume_node=volume_node,
                arr_kji=arr_kji,
                head_mask_threshold_hu=head_mask_threshold_hu,
                head_mask_close_mm=head_mask_close_mm,
                head_mask_aggressive_cleanup=head_mask_aggressive_cleanup,
                head_mask_method=head_mask_method,
                metal_threshold_hu=threshold,
                head_mask_metal_dilate_mm=head_mask_metal_dilate_mm,
            )
            precomputed_gating_mask_kji = cached["gating_mask_kji"]
            precomputed_head_distance_map_kji = cached["head_distance_map_kji"]

        # Delegate detection math to reusable core pipeline; this keeps UI thin.
        result = shank_pipeline.run_detection(
            arr_kji=arr_kji,
            spacing_xyz=volume_node.GetSpacing(),
            threshold=threshold,
            ijk_kji_to_ras_fn=lambda ijk_kji: self._ijk_kji_to_ras_points(volume_node, ijk_kji),
            ras_to_ijk_fn=lambda ras_xyz: self._ras_to_ijk_float(volume_node, ras_xyz),
            center_ras=center_ras,
            max_points=max_points,
            max_lines=max_lines,
            inlier_radius_mm=inlier_radius_mm,
            min_length_mm=min_length_mm,
            min_inliers=min_inliers,
            ransac_iterations=ransac_iterations,
            exclude_segments=exclude_segments,
            exclude_radius_mm=exclude_radius_mm,
            use_head_mask=use_head_mask,
            build_head_mask=build_head_mask,
            head_mask_threshold_hu=head_mask_threshold_hu,
            head_mask_aggressive_cleanup=head_mask_aggressive_cleanup,
            head_mask_close_mm=head_mask_close_mm,
            head_mask_method=head_mask_method,
            head_mask_metal_dilate_mm=head_mask_metal_dilate_mm,
            min_metal_depth_mm=min_metal_depth_mm,
            max_metal_depth_mm=max_metal_depth_mm,
            models_by_id=models_by_id,
            min_model_score=min_model_score,
            precomputed_gating_mask_kji=precomputed_gating_mask_kji,
            precomputed_head_distance_map_kji=precomputed_head_distance_map_kji,
        )
        result_flags = result.get("profile_flags") if isinstance(result.get("profile_flags"), dict) else {}
        result_flags["head_cache_hit"] = bool(cache_hit)
        result["profile_flags"] = result_flags
        return result

    def remove_nodes_by_ids(self, node_ids):
        """Remove nodes by ID if present in scene."""
        for node_id in node_ids:
            if not node_id:
                continue
            node = slicer.mrmlScene.GetNodeByID(node_id)
            if node is not None:
                slicer.mrmlScene.RemoveNode(node)

    def create_or_update_trajectory_line(self, name, start_ras, end_ras, node_id=None):
        """Create or update a line markup trajectory node."""
        node = None
        if node_id:
            node = slicer.mrmlScene.GetNodeByID(node_id)
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", name)
            node.CreateDefaultDisplayNodes()
        else:
            node.SetName(name)

        node.RemoveAllControlPoints()
        node.AddControlPoint(vtk.vtkVector3d(*start_ras))
        node.AddControlPoint(vtk.vtkVector3d(*end_ras))
        display = node.GetDisplayNode()
        if display:
            display.SetColor(1.0, 1.0, 0.0)
            display.SetSelectedColor(1.0, 0.85, 0.2)
            display.SetLineThickness(0.5)
            display.SetPointLabelsVisibility(True)
            # Keep control-point labels (e.g., R02-1/R02-2) but hide auto length text.
            if hasattr(display, "SetPropertiesLabelVisibility"):
                display.SetPropertiesLabelVisibility(False)
        return node

    def create_contacts_fiducials_nodes_by_trajectory(self, contacts, node_prefix="CTShankContacts"):
        """Create/update one fiducial list per trajectory from contact coordinates."""
        grouped = {}
        for contact in contacts:
            traj = contact.get("trajectory")
            if not traj:
                continue
            grouped.setdefault(traj, []).append(contact)

        out = {}
        for traj_name in sorted(grouped.keys()):
            node_name = f"{node_prefix}_{traj_name}"
            node = None
            try:
                node = slicer.util.getNode(node_name)
            except Exception:
                node = None
            if node is None:
                node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", node_name)
                node.CreateDefaultDisplayNodes()

            node.RemoveAllControlPoints()
            for contact in sorted(grouped[traj_name], key=lambda c: int(c.get("index", 0))):
                xyz = lps_to_ras_point(contact["position_lps"])
                node.AddControlPoint(vtk.vtkVector3d(*xyz), contact["label"])

            display = node.GetDisplayNode()
            if display:
                display.SetGlyphTypeFromString("Sphere3D")
                display.SetGlyphScale(2.0)
                display.SetTextScale(1.5)
                display.SetColor(1.0, 1.0, 0.2)
                display.SetSelectedColor(1.0, 0.8, 0.1)

            out[traj_name] = node

        return out

    def _vsub(self, a, b):
        return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]

    def _vadd(self, a, b):
        return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]

    def _vmul(self, a, s):
        return [a[0] * s, a[1] * s, a[2] * s]

    def _vdot(self, a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def _vcross(self, a, b):
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]

    def _vnorm(self, a):
        return math.sqrt(self._vdot(a, a))

    def _vunit(self, a):
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
        if hasattr(slice_node, "JumpSliceByOffsetting"):
            slice_node.JumpSliceByOffsetting(center[0], center[1], center[2])
        if hasattr(slice_node, "JumpSliceByCentering"):
            slice_node.JumpSliceByCentering(center[0], center[1], center[2])

    def reset_standard_slice_views(self):
        """Reset Red/Yellow/Green to Axial/Sagittal/Coronal orientation."""
        lm = slicer.app.layoutManager()
        if lm is None:
            raise RuntimeError("Slicer layout manager is not available")
        layout_node = slicer.app.layoutManager().layoutLogic().GetLayoutNode()
        if layout_node is not None:
            layout_node.SetViewArrangement(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)

        config = [
            ("Red", "Axial"),
            ("Yellow", "Sagittal"),
            ("Green", "Coronal"),
        ]
        for view_name, orientation in config:
            widget = lm.sliceWidget(view_name)
            if widget is None:
                continue
            slice_node = widget.mrmlSliceNode()
            if slice_node is None:
                continue
            slice_node.SetOrientation(orientation)
            logic = widget.sliceLogic()
            if logic is not None:
                logic.FitSliceToAll()
