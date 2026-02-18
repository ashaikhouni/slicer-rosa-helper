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
        self.headMaskThresholdSpin.setValue(-300.0)
        self.headMaskThresholdSpin.setSingleStep(25.0)
        self.headMaskThresholdSpin.setSuffix(" HU")
        self.headMaskThresholdSlider = qt.QSlider(qt.Qt.Horizontal)
        self.headMaskThresholdSlider.setRange(-1200, 1000)
        self.headMaskThresholdSlider.setSingleStep(25)
        self.headMaskThresholdSlider.setPageStep(100)
        self.headMaskThresholdSlider.setValue(-300)
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

        self.minMetalDepthSpin = qt.QDoubleSpinBox()
        self.minMetalDepthSpin.setRange(0.0, 100.0)
        self.minMetalDepthSpin.setDecimals(2)
        self.minMetalDepthSpin.setValue(0.0)
        self.minMetalDepthSpin.setSingleStep(0.25)
        self.minMetalDepthSpin.setSuffix(" mm")
        self.minMetalDepthSpin.setToolTip("Minimum depth from outer head surface for metal points.")
        form.addRow("Min metal depth", self.minMetalDepthSpin)

        self.maxMetalDepthSpin = qt.QDoubleSpinBox()
        self.maxMetalDepthSpin.setRange(0.0, 200.0)
        self.maxMetalDepthSpin.setDecimals(2)
        self.maxMetalDepthSpin.setValue(200.0)
        self.maxMetalDepthSpin.setSingleStep(0.5)
        self.maxMetalDepthSpin.setSuffix(" mm")
        self.maxMetalDepthSpin.setToolTip("Maximum depth from outer head surface for metal points.")
        form.addRow("Max metal depth", self.maxMetalDepthSpin)

        self.aggressiveHeadCleanupCheck = qt.QCheckBox("Aggressive 3-plane island cleanup")
        self.aggressiveHeadCleanupCheck.setChecked(True)
        self.aggressiveHeadCleanupCheck.setToolTip(
            "Use stronger island removal (axial/coronal/sagittal) for head-mask Refine + Fill."
        )
        form.addRow(self.aggressiveHeadCleanupCheck)

        self.minInsideFractionSpin = qt.QDoubleSpinBox()
        self.minInsideFractionSpin.setRange(0.0, 1.0)
        self.minInsideFractionSpin.setDecimals(2)
        self.minInsideFractionSpin.setValue(0.70)
        self.minInsideFractionSpin.setSingleStep(0.05)
        form.addRow("Min in-head fraction", self.minInsideFractionSpin)

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

        self.showInMaskPointsCheck = qt.QCheckBox("Show in-mask candidate points (red)")
        self.showInMaskPointsCheck.setChecked(True)
        self.showInMaskPointsCheck.setToolTip(
            "Show the metal points kept after depth/head gating (sampled for responsiveness)."
        )
        self.maxInMaskPointsSpin = qt.QSpinBox()
        self.maxInMaskPointsSpin.setRange(500, 30000)
        self.maxInMaskPointsSpin.setValue(2500)
        self.maxInMaskPointsSpin.setSingleStep(500)
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
        self.showDepthCurveButton = qt.QPushButton("Show Depth Curve")
        self.showDepthCurveButton.clicked.connect(self.onShowDepthCurveClicked)
        detectRow.addWidget(self.showDepthCurveButton)
        self.detectButton = qt.QPushButton("Detect Trajectories")
        self.detectButton.clicked.connect(self.onDetectClicked)
        detectRow.addWidget(self.detectButton)
        self.resetViewsButton = qt.QPushButton("Reset Ax/Cor/Sag")
        self.resetViewsButton.clicked.connect(self.onResetViewsClicked)
        detectRow.addWidget(self.resetViewsButton)
        detectRow.addStretch(1)
        form.addRow(detectRow)

        self.depthCurveSection = ctk.ctkCollapsibleButton()
        self.depthCurveSection.text = "Depth Curve"
        self.depthCurveSection.collapsed = False
        self.layout.addWidget(self.depthCurveSection)
        depthLayout = qt.QVBoxLayout(self.depthCurveSection)
        self.depthPlotWidget = None
        self.depthPlotViewNode = None
        self._setupDepthPlotWidget(depthLayout)

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
        self.previewHeadMaskButton.clicked.connect(self.onPreviewHeadMaskClicked)
        self.resetHeadMaskButton.clicked.connect(self.onResetHeadMaskPreviewClicked)

        self.layout.addWidget(self.statusText)
        self.layout.addStretch(1)

        self._loadElectrodeLibrary()

    def log(self, msg):
        """Append status message to UI and console."""
        self.statusText.appendPlainText(msg)
        print(msg)

    def _setupDepthPlotWidget(self, parent_layout):
        """Create an embedded MRML plot widget for depth curves."""
        try:
            if hasattr(slicer, "qMRMLPlotWidget"):
                plot_widget = slicer.qMRMLPlotWidget()
            else:
                import qSlicerPlotsModuleWidgetsPythonQt as plots_widgets

                plot_widget = plots_widgets.qMRMLPlotWidget()
            plot_widget.setMRMLScene(slicer.mrmlScene)
            plot_widget.setMinimumHeight(180)
            view_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotViewNode", "ShankDetectDepthPlotView")
            if hasattr(plot_widget, "setMRMLPlotViewNode"):
                plot_widget.setMRMLPlotViewNode(view_node)
            elif hasattr(plot_widget, "setMRMLViewNode"):
                plot_widget.setMRMLViewNode(view_node)
            self.depthPlotWidget = plot_widget
            self.depthPlotViewNode = view_node
            parent_layout.addWidget(plot_widget)
        except Exception:
            self.depthPlotWidget = qt.QLabel("Embedded plot view unavailable in this Slicer build.")
            self.depthPlotWidget.setWordWrap(True)
            parent_layout.addWidget(self.depthPlotWidget)
            self.depthPlotViewNode = None

    def _showDepthChart(self, chart_node):
        """Display chart in embedded plot widget when available."""
        if chart_node is None or self.depthPlotViewNode is None:
            return False
        try:
            self.depthPlotViewNode.SetPlotChartNodeID(chart_node.GetID())
            return True
        except Exception:
            return False

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

    def _nextSiteAwareNames(self, lines, existing_names):
        """Assign side-aware names (R##/L##) for new detected lines."""
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
            side = "R" if mid_x >= 0.0 else "L"
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
            "min_metal_depth_mm": float(self.minMetalDepthSpin.value),
            "max_metal_depth_mm": float(self.maxMetalDepthSpin.value),
            "head_mask_aggressive_cleanup": bool(self.aggressiveHeadCleanupCheck.checked),
            "min_inside_fraction": float(self.minInsideFractionSpin.value),
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
            f"inside_method={inside_method}, metal_in_head={metal_in_head}, depth_kept={depth_kept}, "
            f"voxels(metal={metal_vox}, gating={gating_vox})"
        )

    def _displayInMaskPointsFromResult(self, volume_node, result, max_points=6000):
        """Render sampled in-mask candidate points as red fiducials for QC."""
        points_ras = result.get("in_mask_points_ras")
        if points_ras is None:
            self.logic.clear_in_mask_points_node(volume_node)
            return
        shown, total = self.logic.show_in_mask_points_node(
            volume_node=volume_node,
            points_ras=points_ras,
            max_points=max_points,
        )
        self.log(f"[mask] in-mask candidate points shown: {shown}/{total}")

    def onShowDepthCurveClicked(self):
        """Show depth survival curve from last preview/detect result."""
        volume_node = self.ctSelector.currentNode()
        if volume_node is None:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Shank Detect", "Select a CT volume first.")
            return
        if self._lastMaskResult is None:
            qt.QMessageBox.information(
                slicer.util.mainWindow(),
                "Shank Detect",
                "Run Preview Masks or Detect Trajectories first to compute depth values.",
            )
            return
        self._showDepthCurveFromResult(volume_node, self._lastMaskResult)

    def _showDepthCurveFromResult(self, volume_node, result):
        """Plot survival curve N(depth >= t) for head-gated metal depths."""
        depths_all = np.asarray(result.get("metal_depth_all_mm"), dtype=float).reshape(-1)
        if depths_all.size == 0:
            self.log("[depth] no depth values available for curve")
            return
        depths_kept = np.asarray(result.get("metal_depth_values_mm"), dtype=float).reshape(-1)
        depths_all = depths_all[np.isfinite(depths_all)]
        depths_kept = depths_kept[np.isfinite(depths_kept)]
        if depths_all.size == 0:
            self.log("[depth] no finite depth values available")
            return

        max_depth = float(max(1.0, np.percentile(depths_all, 99.5)))
        t = np.linspace(0.0, max_depth, 201)
        s_all = np.sort(depths_all)
        y_all = s_all.size - np.searchsorted(s_all, t, side="left")
        if depths_kept.size > 0:
            s_kept = np.sort(depths_kept)
            y_kept = s_kept.size - np.searchsorted(s_kept, t, side="left")
        else:
            y_kept = np.zeros_like(y_all)

        base = volume_node.GetName()
        table_name = f"{base}_DepthCurveTable"
        chart_name = f"{base}_DepthCurveChart"
        all_name = f"{base}_DepthCurveAll"
        kept_name = f"{base}_DepthCurveKept"

        def _get_or_create(class_name, node_name):
            try:
                node = slicer.util.getNode(node_name)
                if node is not None and node.IsA(class_name):
                    return node
            except Exception:
                pass
            return slicer.mrmlScene.AddNewNodeByClass(class_name, node_name)

        table_node = _get_or_create("vtkMRMLTableNode", table_name)
        table = table_node.GetTable()
        table.Initialize()

        x_col = vtk.vtkFloatArray()
        x_col.SetName("DepthMm")
        all_col = vtk.vtkFloatArray()
        all_col.SetName("N_ge_t_all")
        kept_col = vtk.vtkFloatArray()
        kept_col.SetName("N_ge_t_kept")
        table.AddColumn(x_col)
        table.AddColumn(all_col)
        table.AddColumn(kept_col)
        table.SetNumberOfRows(int(t.size))
        for i in range(int(t.size)):
            table.SetValue(i, 0, float(t[i]))
            table.SetValue(i, 1, float(y_all[i]))
            table.SetValue(i, 2, float(y_kept[i]))

        all_series = _get_or_create("vtkMRMLPlotSeriesNode", all_name)
        all_series.SetAndObserveTableNodeID(table_node.GetID())
        all_series.SetXColumnName("DepthMm")
        all_series.SetYColumnName("N_ge_t_all")
        all_series.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeLine)
        all_series.SetMarkerStyle(slicer.vtkMRMLPlotSeriesNode.MarkerStyleNone)
        all_series.SetColor(0.6, 0.6, 0.6)

        kept_series = _get_or_create("vtkMRMLPlotSeriesNode", kept_name)
        kept_series.SetAndObserveTableNodeID(table_node.GetID())
        kept_series.SetXColumnName("DepthMm")
        kept_series.SetYColumnName("N_ge_t_kept")
        kept_series.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeLine)
        kept_series.SetMarkerStyle(slicer.vtkMRMLPlotSeriesNode.MarkerStyleNone)
        kept_series.SetColor(1.0, 0.2, 0.2)

        chart = _get_or_create("vtkMRMLPlotChartNode", chart_name)
        chart.RemoveAllPlotSeriesNodeIDs()
        chart.AddAndObservePlotSeriesNodeID(all_series.GetID())
        chart.AddAndObservePlotSeriesNodeID(kept_series.GetID())
        chart.SetTitle(f"{base}: Depth Survival Curve")
        chart.SetXAxisTitle("Depth from head surface (mm)")
        chart.SetYAxisTitle("N(depth >= t)")

        if self._showDepthChart(chart):
            self.log(f"[depth] curve shown in module: total={depths_all.size}, kept={depths_kept.size}")
        else:
            slicer.modules.plots.logic().ShowChartInLayout(chart)
            self.log(f"[depth] curve shown in plot layout: total={depths_all.size}, kept={depths_kept.size}")

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
        self.log("[depth] preview complete; click 'Show Depth Curve' to plot.")

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
                min_metal_depth_mm=settings["min_metal_depth_mm"],
                max_metal_depth_mm=settings["max_metal_depth_mm"],
                min_inside_fraction=settings["min_inside_fraction"],
                models_by_id=self.modelsById if settings["use_model_score"] else None,
                min_model_score=settings["min_model_score"] if settings["use_model_score"] else None,
            )
        except Exception as exc:
            self.log(f"[detect] failed: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Shank Detect", str(exc))
            return

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
        self.log("[depth] detect complete; click 'Show Depth Curve' to plot.")

        existing_names = [row["name"] for row in locked]
        new_names = self._nextSiteAwareNames(detected, existing_names)

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

    def _ensure_masking_api(self):
        """Ensure `shank_core.masking` exposes required API after hot-reload."""
        global shank_masking  # pylint: disable=global-statement

        if hasattr(shank_masking, "build_preview_masks"):
            return

        import importlib

        try:
            shank_masking = importlib.reload(shank_masking)
        except Exception:
            import shank_core.masking as _masking

            shank_masking = _masking

        if not hasattr(shank_masking, "build_preview_masks"):
            module_path = getattr(shank_masking, "__file__", "<unknown>")
            raise RuntimeError(
                "Loaded shank_core.masking is stale and missing build_preview_masks "
                f"(module={module_path}). Reload module paths in Slicer and restart once."
            )

    def extract_threshold_points_ras(self, volume_node, threshold, max_points=300000):
        """Return sampled threshold voxels as RAS points from a scalar volume."""
        if np is None:
            raise RuntimeError("numpy is required for threshold candidate extraction")

        arr_kji = slicer.util.arrayFromVolume(volume_node)
        ijk_kji = self._sample_threshold_ijk_kji(arr_kji, threshold=threshold, max_points=max_points)
        if ijk_kji.size == 0:
            return np.empty((0, 3), dtype=float)

        return self._ijk_kji_to_ras_points(volume_node, ijk_kji)

    def _sample_threshold_ijk_kji(self, arr_kji, threshold, max_points=300000):
        """Return sampled `(k,j,i)` index array for thresholded voxels."""
        ijk_kji = np.argwhere(arr_kji >= float(threshold))
        if ijk_kji.size == 0:
            return ijk_kji.reshape(0, 3)

        n = int(ijk_kji.shape[0])
        if n > int(max_points):
            rng = np.random.default_rng(0)
            pick = rng.choice(n, size=int(max_points), replace=False)
            ijk_kji = ijk_kji[pick]
        return ijk_kji

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

    def _build_head_mask_kji(
        self,
        volume_node,
        arr_kji,
        threshold_hu=-300.0,
        close_mm=2.0,
        aggressive_cleanup=True,
    ):
        """Build largest-component head mask in KJI index order."""
        self._ensure_masking_api()
        return shank_masking.build_head_mask_kji(
            arr_kji=arr_kji,
            spacing_xyz=volume_node.GetSpacing(),
            threshold_hu=threshold_hu,
            close_mm=close_mm,
            aggressive_cleanup=aggressive_cleanup,
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
        """Remove the in-mask point preview node for a volume if present."""
        if volume_node is None:
            return
        node_name = f"{volume_node.GetName()}_InMaskPoints"
        try:
            node = slicer.util.getNode(node_name)
        except Exception:
            node = None
        if node is not None:
            slicer.mrmlScene.RemoveNode(node)

    def show_in_mask_points_node(self, volume_node, points_ras, max_points=6000):
        """Show sampled gated candidate points as a red fiducial cloud.

        Returns
        -------
        (shown_count, total_count)
        """
        if volume_node is None:
            return (0, 0)
        if points_ras is None:
            self.clear_in_mask_points_node(volume_node)
            return (0, 0)

        pts = np.asarray(points_ras, dtype=float).reshape(-1, 3)
        total = int(pts.shape[0])
        if total == 0:
            self.clear_in_mask_points_node(volume_node)
            return (0, 0)

        max_points = max(100, int(max_points))
        if total > max_points:
            rng = np.random.default_rng(0)
            pick = rng.choice(total, size=max_points, replace=False)
            pts = pts[pick]

        node_name = f"{volume_node.GetName()}_InMaskPoints"
        try:
            node = slicer.util.getNode(node_name)
        except Exception:
            node = None
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", node_name)
            node.CreateDefaultDisplayNodes()
        else:
            node.SetName(node_name)

        node.RemoveAllControlPoints()
        for p in pts:
            node.AddControlPoint(vtk.vtkVector3d(float(p[0]), float(p[1]), float(p[2])))

        display = node.GetDisplayNode()
        if display:
            display.SetGlyphTypeFromString("Sphere3D")
            display.SetGlyphScale(2.0)
            display.SetTextScale(0.0)
            display.SetColor(1.0, 0.0, 0.0)
            display.SetSelectedColor(1.0, 0.25, 0.25)
            if hasattr(display, "SetPointLabelsVisibility"):
                display.SetPointLabelsVisibility(False)

        return (int(pts.shape[0]), total)

    def _ras_to_ijk_float(self, volume_node, ras_xyz):
        """Convert one RAS point to continuous IJK coordinates."""
        ras_h = np.array([float(ras_xyz[0]), float(ras_xyz[1]), float(ras_xyz[2]), 1.0], dtype=float)
        m_vtk = vtk.vtkMatrix4x4()
        volume_node.GetRASToIJKMatrix(m_vtk)
        m = np.array([[m_vtk.GetElement(r, c) for c in range(4)] for r in range(4)], dtype=float)
        ijk = m @ ras_h
        return ijk[:3]

    def _segment_inside_mask_fraction(self, volume_node, start_ras, end_ras, mask_kji, step_mm=1.0):
        """Fraction of sampled segment points that fall inside `mask_kji`."""
        start = np.asarray(start_ras, dtype=float)
        end = np.asarray(end_ras, dtype=float)
        seg = end - start
        length = float(np.linalg.norm(seg))
        if length <= 1e-9:
            return 0.0
        n = max(2, int(math.ceil(length / max(1e-3, float(step_mm)))) + 1)
        ts = np.linspace(0.0, 1.0, n)

        dims = mask_kji.shape  # (k, j, i)
        inside = 0
        total = 0
        for t in ts:
            p = start + t * seg
            ijk = self._ras_to_ijk_float(volume_node, p)
            i = int(round(float(ijk[0])))
            j = int(round(float(ijk[1])))
            k = int(round(float(ijk[2])))
            total += 1
            if 0 <= k < dims[0] and 0 <= j < dims[1] and 0 <= i < dims[2]:
                if bool(mask_kji[k, j, i]):
                    inside += 1
        return float(inside) / float(max(1, total))

    def _clamp_segment_to_mask(self, volume_node, start_ras, end_ras, mask_kji, step_mm=0.5):
        """Clip segment endpoints to first/last in-mask samples along the segment."""
        start = np.asarray(start_ras, dtype=float)
        end = np.asarray(end_ras, dtype=float)
        seg = end - start
        length = float(np.linalg.norm(seg))
        if length <= 1e-9:
            return None
        n = max(2, int(math.ceil(length / max(1e-3, float(step_mm)))) + 1)
        ts = np.linspace(0.0, 1.0, n)
        pts = start.reshape(1, 3) + np.outer(ts, seg)

        dims = mask_kji.shape  # (k,j,i)
        inside = np.zeros((n,), dtype=bool)
        for idx in range(n):
            ijk = self._ras_to_ijk_float(volume_node, pts[idx])
            i = int(round(float(ijk[0])))
            j = int(round(float(ijk[1])))
            k = int(round(float(ijk[2])))
            if 0 <= k < dims[0] and 0 <= j < dims[1] and 0 <= i < dims[2]:
                inside[idx] = bool(mask_kji[k, j, i])
        idxs = np.where(inside)[0]
        if idxs.size < 2:
            return None
        p0 = pts[int(idxs[0])]
        p1 = pts[int(idxs[-1])]
        if float(np.linalg.norm(p1 - p0)) <= 1e-3:
            return None
        return [float(p0[0]), float(p0[1]), float(p0[2])], [float(p1[0]), float(p1[1]), float(p1[2])]

    def _sample_local_max_hu(self, arr_kji, ijk_float, radius_vox_ijk):
        """Sample local max HU near one continuous IJK location."""
        i0 = int(round(float(ijk_float[0])))
        j0 = int(round(float(ijk_float[1])))
        k0 = int(round(float(ijk_float[2])))
        ri, rj, rk = radius_vox_ijk
        i1 = max(0, i0 - ri)
        i2 = min(arr_kji.shape[2] - 1, i0 + ri)
        j1 = max(0, j0 - rj)
        j2 = min(arr_kji.shape[1] - 1, j0 + rj)
        k1 = max(0, k0 - rk)
        k2 = min(arr_kji.shape[0] - 1, k0 + rk)
        if i1 > i2 or j1 > j2 or k1 > k2:
            return float("-inf")
        patch = arr_kji[k1 : k2 + 1, j1 : j2 + 1, i1 : i2 + 1]
        return float(np.max(patch)) if patch.size else float("-inf")

    def _sample_axis_profile_hu(
        self,
        volume_node,
        arr_kji,
        tip_ras,
        entry_ras,
        step_mm=0.5,
        radial_sample_mm=0.8,
    ):
        """Sample local-max HU profile from tip towards entry along trajectory axis."""
        tip = np.asarray(tip_ras, dtype=float)
        entry = np.asarray(entry_ras, dtype=float)
        axis = entry - tip
        length = float(np.linalg.norm(axis))
        if length <= 1e-6:
            return np.empty((0,), dtype=float), np.empty((0,), dtype=float)
        axis = axis / length

        spacing = volume_node.GetSpacing()  # xyz
        ri = max(0, int(round(float(radial_sample_mm) / max(1e-6, float(spacing[0])))))
        rj = max(0, int(round(float(radial_sample_mm) / max(1e-6, float(spacing[1])))))
        rk = max(0, int(round(float(radial_sample_mm) / max(1e-6, float(spacing[2])))))
        radius_vox_ijk = (ri, rj, rk)

        n = max(2, int(math.floor(length / max(1e-6, float(step_mm)))) + 1)
        t = np.linspace(0.0, length, n)
        vals = np.zeros((n,), dtype=float)
        for idx, dist in enumerate(t):
            p = tip + dist * axis
            ijk = self._ras_to_ijk_float(volume_node, p)
            vals[idx] = self._sample_local_max_hu(arr_kji, ijk, radius_vox_ijk)
        return t, vals

    def _score_model_template_on_profile(self, t_mm, hu_vals, model, line_length_mm):
        """Return best template score for one model across small depth shifts."""
        if t_mm.size < 8:
            return {"score": float("-inf"), "shift_mm": 0.0}
        offsets = [float(x) for x in model.get("contact_center_offsets_from_tip_mm", [])]
        if not offsets:
            return {"score": float("-inf"), "shift_mm": 0.0}

        mean = float(np.mean(hu_vals))
        std = float(np.std(hu_vals))
        if std <= 1e-6:
            return {"score": float("-inf"), "shift_mm": 0.0}
        z = (hu_vals - mean) / std
        t_max = float(np.max(t_mm))

        shifts = np.arange(-6.0, 6.01, 0.5)
        best_score = float("-inf")
        best_shift = 0.0
        for shift in shifts:
            centers = [o + float(shift) for o in offsets]
            centers = [c for c in centers if 0.0 <= c <= t_max]
            if len(centers) < max(3, int(0.4 * len(offsets))):
                continue
            contact_vals = np.interp(centers, t_mm, z)

            gap_vals = []
            for i in range(len(centers) - 1):
                gap_vals.append(0.5 * (centers[i] + centers[i + 1]))
            if gap_vals:
                gap_vals = np.interp(np.asarray(gap_vals, dtype=float), t_mm, z)
                gap_mean = float(np.mean(gap_vals))
            else:
                gap_mean = 0.0
            contact_mean = float(np.mean(contact_vals))

            score = contact_mean - gap_mean
            model_len = float(model.get("total_exploration_length_mm", line_length_mm))
            length_mismatch = abs(float(line_length_mm) - model_len)
            score -= max(0.0, length_mismatch - 5.0) * 0.05

            if score > best_score:
                best_score = score
                best_shift = float(shift)
        return {"score": best_score, "shift_mm": best_shift}

    def _select_best_model_for_line(self, volume_node, arr_kji, line, models_by_id):
        """Evaluate electrode model templates on one candidate line profile."""
        t_mm, hu_vals = self._sample_axis_profile_hu(
            volume_node=volume_node,
            arr_kji=arr_kji,
            tip_ras=line["end_ras"],
            entry_ras=line["start_ras"],
            step_mm=0.5,
            radial_sample_mm=0.8,
        )
        if t_mm.size < 8:
            return {"best_model_id": "", "best_model_score": None, "best_model_shift_mm": None}

        best_id = ""
        best_score = float("-inf")
        best_shift = 0.0
        for model_id in sorted(models_by_id.keys()):
            score = self._score_model_template_on_profile(
                t_mm=t_mm,
                hu_vals=hu_vals,
                model=models_by_id[model_id],
                line_length_mm=float(line["length_mm"]),
            )
            if score["score"] > best_score:
                best_score = float(score["score"])
                best_shift = float(score["shift_mm"])
                best_id = model_id
        if not best_id:
            return {"best_model_id": "", "best_model_score": None, "best_model_shift_mm": None}
        return {
            "best_model_id": best_id,
            "best_model_score": float(best_score),
            "best_model_shift_mm": float(best_shift),
        }

    def _model_max_center_gap_mm(self, model):
        """Return maximum adjacent contact-center spacing for one electrode model."""
        offsets = [float(x) for x in model.get("contact_center_offsets_from_tip_mm", [])]
        if len(offsets) < 2:
            return 0.0
        return float(max(offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1)))

    def _max_empty_gap_mm_from_line_inliers(self, t_vals, lo, hi, bin_mm=0.5):
        """Estimate largest empty gap along fitted line span from inlier projections."""
        span = float(hi - lo)
        if span <= max(1e-6, float(bin_mm)):
            return 0.0
        t = np.asarray(t_vals, dtype=float)
        t = t[np.isfinite(t)]
        t = t[(t >= float(lo)) & (t <= float(hi))]
        if t.size < 2:
            return span

        n_bins = max(2, int(math.ceil(span / max(1e-6, float(bin_mm)))))
        occ = np.zeros((n_bins,), dtype=np.uint8)
        u = t - float(lo)
        idx = np.floor(u / max(1e-6, float(bin_mm))).astype(int)
        idx = np.clip(idx, 0, n_bins - 1)
        occ[idx] = 1

        # Bridge 1-bin pinholes, keep only true long empty stretches.
        occ_dil = occ.copy()
        occ_dil[1:] = np.maximum(occ_dil[1:], occ[:-1])
        occ_dil[:-1] = np.maximum(occ_dil[:-1], occ[1:])
        occ = occ_dil
        occ_idx = np.where(occ > 0)[0]
        if occ_idx.size < 2:
            return span
        gaps_bins = np.diff(occ_idx) - 1
        max_gap_bins = int(np.max(gaps_bins)) if gaps_bins.size else 0
        return float(max(0, max_gap_bins)) * float(bin_mm)

    def _line_distances(self, points, p0, direction_unit):
        """Distance of points to an infinite 3D line defined by p0 + t*d."""
        rel = points - p0
        t = rel @ direction_unit
        closest = p0 + np.outer(t, direction_unit)
        return np.linalg.norm(points - closest, axis=1)

    def _fit_axis_pca(self, points):
        """Principal-axis line fit using PCA."""
        center = np.mean(points, axis=0)
        x = points - center
        cov = x.T @ x / max(points.shape[0] - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, int(np.argmax(eigvals))]
        n = np.linalg.norm(axis)
        if n <= 1e-9:
            raise ValueError("failed to estimate principal axis")
        return center, axis / n

    def _ransac_fit_line(self, points, distance_threshold_mm=1.2, min_inliers=250, iterations=240):
        """Robust line fit from point cloud using pair-sampling RANSAC."""
        n = int(points.shape[0])
        if n < 2:
            return None

        rng = np.random.default_rng(0)
        best_mask = None
        best_count = -1
        best_score = float("inf")

        for _ in range(int(iterations)):
            i, j = rng.choice(n, size=2, replace=False)
            p0 = points[i]
            d = points[j] - p0
            dn = float(np.linalg.norm(d))
            if dn <= 1e-9:
                continue
            d = d / dn
            dist = self._line_distances(points, p0, d)
            mask = dist <= float(distance_threshold_mm)
            count = int(np.count_nonzero(mask))
            if count <= 0:
                continue
            score = float(np.mean(dist[mask]))
            if count > best_count or (count == best_count and score < best_score):
                best_mask = mask
                best_count = count
                best_score = score

        if best_mask is None or best_count < int(min_inliers):
            return None

        inliers = points[best_mask]
        center, axis = self._fit_axis_pca(inliers)
        dist = self._line_distances(inliers, center, axis)
        rms = float(np.sqrt(np.mean(dist**2))) if dist.size else 0.0
        return {
            "inlier_mask": best_mask,
            "center": center,
            "axis": axis,
            "inlier_count": int(best_count),
            "rms_mm": rms,
        }

    def _distance_to_segment_mask(self, points, start, end, radius_mm):
        """Boolean mask for points within radius of a finite 3D line segment."""
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        seg = end - start
        seg_len2 = float(np.dot(seg, seg))
        if seg_len2 <= 1e-9:
            d = np.linalg.norm(points - start.reshape(1, 3), axis=1)
            return d <= float(radius_mm)

        rel = points - start.reshape(1, 3)
        t = (rel @ seg) / seg_len2
        t = np.clip(t, 0.0, 1.0)
        closest = start.reshape(1, 3) + np.outer(t, seg)
        d = np.linalg.norm(points - closest, axis=1)
        return d <= float(radius_mm)

    def _distance_to_segment(self, points, start, end):
        """Distance from many 3D points to one finite segment."""
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        seg = end - start
        seg_len2 = float(np.dot(seg, seg))
        if seg_len2 <= 1e-9:
            return np.linalg.norm(points - start.reshape(1, 3), axis=1)
        rel = points - start.reshape(1, 3)
        t = (rel @ seg) / seg_len2
        t = np.clip(t, 0.0, 1.0)
        closest = start.reshape(1, 3) + np.outer(t, seg)
        return np.linalg.norm(points - closest, axis=1)

    def _distance_to_line_with_extension(self, points, start, end, extension_mm=8.0):
        """Distance to line axis with finite extent extended at both ends."""
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        seg = end - start
        length = float(np.linalg.norm(seg))
        if length <= 1e-9:
            d = np.linalg.norm(points - start.reshape(1, 3), axis=1)
            return d, np.zeros((points.shape[0],), dtype=bool)
        u = seg / length
        rel = points - start.reshape(1, 3)
        t = rel @ u
        closest = start.reshape(1, 3) + np.outer(t, u)
        d = np.linalg.norm(points - closest, axis=1)
        ext = float(max(0.0, extension_mm))
        valid = (t >= -ext) & (t <= (length + ext))
        return d, valid

    def _point_to_segment_distance(self, point, start, end):
        """Distance from one 3D point to one finite segment."""
        p = np.asarray(point, dtype=float)
        s = np.asarray(start, dtype=float)
        e = np.asarray(end, dtype=float)
        v = e - s
        vv = float(np.dot(v, v))
        if vv <= 1e-9:
            return float(np.linalg.norm(p - s))
        t = float(np.dot(p - s, v) / vv)
        t = max(0.0, min(1.0, t))
        c = s + t * v
        return float(np.linalg.norm(p - c))

    def _line_quality_tuple(self, line):
        """Sortable quality tuple for choosing between duplicate line hypotheses."""
        model_score = line.get("best_model_score", None)
        has_model = 1 if model_score is not None else 0
        model_val = float(model_score) if model_score is not None else -1e9
        return (
            has_model,
            model_val,
            float(line.get("inside_fraction", 0.0)),
            float(line.get("inlier_count", 0.0)),
            -float(line.get("rms_mm", 999.0)),
        )

    def _are_lines_duplicate(self, line_a, line_b, distance_mm=1.8, angle_deg=3.0):
        """Return True if two detected lines likely represent the same shank."""
        a0 = np.asarray(line_a["start_ras"], dtype=float)
        a1 = np.asarray(line_a["end_ras"], dtype=float)
        b0 = np.asarray(line_b["start_ras"], dtype=float)
        b1 = np.asarray(line_b["end_ras"], dtype=float)

        da = a1 - a0
        db = b1 - b0
        na = float(np.linalg.norm(da))
        nb = float(np.linalg.norm(db))
        if na <= 1e-6 or nb <= 1e-6:
            return False
        ua = da / na
        ub = db / nb
        ang = float(np.degrees(np.arccos(np.clip(abs(np.dot(ua, ub)), 0.0, 1.0))))
        if ang > float(angle_deg):
            return False

        d = [
            self._point_to_segment_distance(a0, b0, b1),
            self._point_to_segment_distance(a1, b0, b1),
            self._point_to_segment_distance(b0, a0, a1),
            self._point_to_segment_distance(b1, a0, a1),
        ]
        return float(np.mean(d)) <= float(distance_mm)

    def _suppress_duplicate_lines(self, lines, distance_mm=1.8, angle_deg=3.0):
        """Non-max suppression for near-duplicate shank detections."""
        if not lines:
            return [], 0
        ordered = sorted(lines, key=self._line_quality_tuple, reverse=True)
        kept = []
        rejected = 0
        for cand in ordered:
            dup = False
            for k in kept:
                if self._are_lines_duplicate(k, cand, distance_mm=distance_mm, angle_deg=angle_deg):
                    dup = True
                    break
            if dup:
                rejected += 1
                continue
            kept.append(cand)
        return kept, rejected

    def _suppress_conflicting_lines_by_support(
        self,
        lines,
        support_points,
        support_radius_mm=1.2,
        overlap_threshold=0.60,
    ):
        """Suppress lines that share most supporting points with a better line."""
        if not lines or support_points is None or support_points.shape[0] == 0:
            return list(lines), 0

        ordered = sorted(lines, key=self._line_quality_tuple, reverse=True)
        kept = []
        kept_masks = []
        rejected = 0
        for cand in ordered:
            c_mask = self._distance_to_segment_mask(
                support_points,
                start=cand["start_ras"],
                end=cand["end_ras"],
                radius_mm=support_radius_mm,
            )
            c_count = int(np.count_nonzero(c_mask))
            if c_count <= 0:
                continue

            conflict = False
            for k_mask in kept_masks:
                k_count = int(np.count_nonzero(k_mask))
                inter = int(np.count_nonzero(np.logical_and(c_mask, k_mask)))
                denom = max(1, min(c_count, k_count))
                if (float(inter) / float(denom)) >= float(overlap_threshold):
                    conflict = True
                    break
            if conflict:
                rejected += 1
                continue
            kept.append(cand)
            kept_masks.append(c_mask)

        return kept, rejected

    def _point_near_mask(self, volume_node, point_ras, mask_kji, radius_mm=2.0):
        """Return True if point is within radius of any positive mask voxel."""
        if mask_kji is None:
            return False
        p = np.asarray(self._ras_to_ijk_float(volume_node, point_ras), dtype=float)
        i0, j0, k0 = p
        sx, sy, sz = volume_node.GetSpacing()
        ri = max(1, int(round(float(radius_mm) / max(1e-6, float(sx)))))
        rj = max(1, int(round(float(radius_mm) / max(1e-6, float(sy)))))
        rk = max(1, int(round(float(radius_mm) / max(1e-6, float(sz)))))
        i1 = max(0, int(math.floor(i0 - ri)))
        i2 = min(mask_kji.shape[2] - 1, int(math.ceil(i0 + ri)))
        j1 = max(0, int(math.floor(j0 - rj)))
        j2 = min(mask_kji.shape[1] - 1, int(math.ceil(j0 + rj)))
        k1 = max(0, int(math.floor(k0 - rk)))
        k2 = min(mask_kji.shape[0] - 1, int(math.ceil(k0 + rk)))
        if i1 > i2 or j1 > j2 or k1 > k2:
            return False
        patch = mask_kji[k1 : k2 + 1, j1 : j2 + 1, i1 : i2 + 1]
        return bool(np.any(patch))

    def _refine_lines_exclusive(
        self,
        volume_node,
        arr_kji,
        support_points,
        raw_lines,
        inlier_radius_mm,
        min_length_mm,
        min_inliers,
        gating_mask_kji,
        skull_mask_kji,
        min_inside_fraction,
        models_by_id=None,
        min_model_score=None,
    ):
        """Second-pass exclusive assignment: each support point belongs to at most one line."""
        if not raw_lines or support_points is None or support_points.shape[0] == 0:
            return [], {"gap_reject_count": 0, "duplicate_reject_count": 0}

        n = support_points.shape[0]
        m = len(raw_lines)
        dist_mat = np.full((n, m), np.inf, dtype=float)
        for li, line in enumerate(raw_lines):
            d, valid = self._distance_to_line_with_extension(
                support_points,
                start=line["start_ras"],
                end=line["end_ras"],
                extension_mm=8.0,
            )
            d[~valid] = np.inf
            dist_mat[:, li] = d

        assign_radius_mm = max(1.5, float(inlier_radius_mm) * 1.25)
        best_idx = np.argmin(dist_mat, axis=1)
        best_dist = dist_mat[np.arange(n), best_idx]
        assigned = np.full((n,), -1, dtype=int)
        assigned[best_dist <= assign_radius_mm] = best_idx[best_dist <= assign_radius_mm]

        center_ras = self._volume_center_ras(volume_node)
        kept = []
        gap_reject_count = 0

        for li in range(m):
            pts = support_points[assigned == li]
            if pts.shape[0] < int(min_inliers):
                continue
            center, axis = self._fit_axis_pca(pts)
            t = (pts - center.reshape(1, 3)) @ axis
            if t.size < 2:
                continue
            lo = float(np.percentile(t, 2.0))
            hi = float(np.percentile(t, 98.0))
            length = float(hi - lo)
            if length < float(min_length_mm):
                continue

            p0 = center + axis * lo
            p1 = center + axis * hi
            d0 = float(np.linalg.norm(p0 - center_ras))
            d1 = float(np.linalg.norm(p1 - center_ras))
            if d0 < d1:
                entry_ras = p1
                target_ras = p0
            else:
                entry_ras = p0
                target_ras = p1

            inside_fraction = 1.0
            if gating_mask_kji is not None:
                inside_fraction = self._segment_inside_mask_fraction(
                    volume_node=volume_node,
                    start_ras=entry_ras,
                    end_ras=target_ras,
                    mask_kji=gating_mask_kji,
                    step_mm=1.0,
                )
                if inside_fraction < float(min_inside_fraction):
                    continue
                clamped = self._clamp_segment_to_mask(
                    volume_node=volume_node,
                    start_ras=entry_ras,
                    end_ras=target_ras,
                    mask_kji=gating_mask_kji,
                    step_mm=0.5,
                )
                if clamped is None:
                    continue
                entry_ras, target_ras = clamped
                length = float(np.linalg.norm(np.asarray(entry_ras) - np.asarray(target_ras)))
                if length < float(min_length_mm):
                    continue

            if skull_mask_kji is not None:
                if not self._point_near_mask(
                    volume_node=volume_node,
                    point_ras=entry_ras,
                    mask_kji=skull_mask_kji,
                    radius_mm=3.0,
                ):
                    continue

            rms = float(np.sqrt(np.mean(self._line_distances(pts, center, axis) ** 2))) if pts.shape[0] else 0.0
            line = {
                "start_ras": [float(entry_ras[0]), float(entry_ras[1]), float(entry_ras[2])],
                "end_ras": [float(target_ras[0]), float(target_ras[1]), float(target_ras[2])],
                "length_mm": float(length),
                "inlier_count": int(pts.shape[0]),
                "rms_mm": float(rms),
                "inside_fraction": float(inside_fraction),
            }

            if models_by_id:
                model_fit = self._select_best_model_for_line(
                    volume_node=volume_node,
                    arr_kji=arr_kji,
                    line=line,
                    models_by_id=models_by_id,
                )
                line.update(model_fit)
                best_model_id = str(model_fit.get("best_model_id") or "")
                if min_model_score is not None and model_fit.get("best_model_score") is not None:
                    if float(model_fit["best_model_score"]) < float(min_model_score):
                        continue

                observed_gap = self._max_empty_gap_mm_from_line_inliers(t_vals=t, lo=lo, hi=hi, bin_mm=0.5)
                if best_model_id and best_model_id in models_by_id:
                    max_model_gap = self._model_max_center_gap_mm(models_by_id[best_model_id])
                    allowed_gap = float(max_model_gap) + 3.0
                else:
                    allowed_gap = 14.0
                line["max_observed_gap_mm"] = float(observed_gap)
                line["max_allowed_gap_mm"] = float(allowed_gap)
                if observed_gap > allowed_gap:
                    gap_reject_count += 1
                    continue

            kept.append(line)

        kept, duplicate_reject_count = self._suppress_conflicting_lines_by_support(
            lines=kept,
            support_points=support_points,
            support_radius_mm=max(1.0, float(inlier_radius_mm)),
            overlap_threshold=0.60,
        )
        return kept, {
            "gap_reject_count": int(gap_reject_count),
            "duplicate_reject_count": int(duplicate_reject_count),
        }

    def preview_masks(
        self,
        volume_node,
        threshold,
        use_head_mask=False,
        build_head_mask=True,
        head_mask_threshold_hu=-300.0,
        head_mask_aggressive_cleanup=True,
        head_mask_close_mm=2.0,
        min_metal_depth_mm=0.0,
        max_metal_depth_mm=999.0,
    ):
        """Build and preview head-distance masks only (no line fitting)."""
        self._ensure_masking_api()
        arr_kji = slicer.util.arrayFromVolume(volume_node)
        result = shank_masking.build_preview_masks(
            arr_kji=arr_kji,
            spacing_xyz=volume_node.GetSpacing(),
            threshold=threshold,
            use_head_mask=use_head_mask,
            build_head_mask=build_head_mask,
            head_mask_threshold_hu=head_mask_threshold_hu,
            head_mask_aggressive_cleanup=head_mask_aggressive_cleanup,
            head_mask_close_mm=head_mask_close_mm,
            min_metal_depth_mm=min_metal_depth_mm,
            max_metal_depth_mm=max_metal_depth_mm,
        )
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
        head_mask_threshold_hu=-300.0,
        head_mask_aggressive_cleanup=True,
        head_mask_close_mm=2.0,
        min_metal_depth_mm=0.0,
        max_metal_depth_mm=999.0,
        min_inside_fraction=0.70,
        models_by_id=None,
        min_model_score=None,
    ):
        """Detect multiple linear trajectories in CT threshold candidates."""
        if np is None:
            raise RuntimeError("numpy is required for trajectory detection")

        self._ensure_masking_api()
        arr_kji = slicer.util.arrayFromVolume(volume_node)
        preview = shank_masking.build_preview_masks(
            arr_kji=arr_kji,
            spacing_xyz=volume_node.GetSpacing(),
            threshold=threshold,
            use_head_mask=use_head_mask,
            build_head_mask=build_head_mask,
            head_mask_threshold_hu=head_mask_threshold_hu,
            head_mask_aggressive_cleanup=head_mask_aggressive_cleanup,
            head_mask_close_mm=head_mask_close_mm,
            min_metal_depth_mm=min_metal_depth_mm,
            max_metal_depth_mm=max_metal_depth_mm,
        )
        candidate_count = int(preview.get("candidate_count", 0))
        if candidate_count == 0:
            return {
                "candidate_count": 0,
                "head_mask_kept_count": 0,
                "inside_method": "none",
                "metal_in_head_count": 0,
                "depth_kept_count": 0,
                "gap_reject_count": 0,
                "duplicate_reject_count": 0,
                "metal_mask_kji": (arr_kji >= float(threshold)).astype(np.uint8),
                "head_mask_kji": None,
                "skull_mask_kji": None,
                "inside_skull_mask_kji": None,
                "metal_depth_all_mm": np.empty((0,), dtype=float),
                "metal_depth_values_mm": np.empty((0,), dtype=float),
                "in_mask_points_ras": np.empty((0, 3), dtype=float),
                "lines": [],
            }

        gating_mask_kji = preview.get("head_mask_kji")
        inside_skull_mask_kji = preview.get("inside_skull_mask_kji")
        skull_mask_kji = preview.get("skull_mask_kji")
        inside_method = str(preview.get("inside_method", "none"))
        gating_mask_type = str(preview.get("gating_mask_type", "none"))

        ijk_all = np.asarray(preview.get("in_mask_ijk_kji"), dtype=int).reshape(-1, 3)
        head_mask_kept_count_full = int(ijk_all.shape[0])
        metal_in_head_count = int(preview.get("metal_in_head_count", head_mask_kept_count_full))
        depth_kept_count = int(preview.get("depth_kept_count", head_mask_kept_count_full))
        metal_depth_all_mm = np.asarray(preview.get("metal_depth_all_mm"), dtype=float).reshape(-1)
        metal_depth_values_mm = np.asarray(preview.get("metal_depth_values_mm"), dtype=float).reshape(-1)
        if head_mask_kept_count_full > int(max_points):
            rng = np.random.default_rng(0)
            pick = rng.choice(head_mask_kept_count_full, size=int(max_points), replace=False)
            ijk_kji = ijk_all[pick]
        else:
            ijk_kji = ijk_all

        head_mask_kept_count = int(head_mask_kept_count_full)
        in_mask_points_ras = np.empty((0, 3), dtype=float)
        points = self._ijk_kji_to_ras_points(volume_node, ijk_kji)
        in_mask_points_ras = points.copy()
        if head_mask_kept_count == 0:
            return {
                "candidate_count": candidate_count,
                "head_mask_kept_count": 0,
                "gating_mask_type": gating_mask_type,
                "inside_method": inside_method,
                "metal_in_head_count": metal_in_head_count,
                "depth_kept_count": depth_kept_count,
                "gap_reject_count": 0,
                "duplicate_reject_count": 0,
                "metal_mask_kji": (arr_kji >= float(threshold)).astype(np.uint8),
                "head_mask_kji": gating_mask_kji.astype(np.uint8) if gating_mask_kji is not None else None,
                "skull_mask_kji": skull_mask_kji.astype(np.uint8) if skull_mask_kji is not None else None,
                "inside_skull_mask_kji": inside_skull_mask_kji.astype(np.uint8)
                if inside_skull_mask_kji is not None
                else None,
                "metal_depth_all_mm": metal_depth_all_mm,
                "metal_depth_values_mm": metal_depth_values_mm,
                "in_mask_points_ras": in_mask_points_ras,
                "lines": [],
            }

        remaining = points

        if exclude_segments:
            keep = np.ones((remaining.shape[0],), dtype=bool)
            for seg in exclude_segments:
                seg_mask = self._distance_to_segment_mask(
                    remaining,
                    seg["start_ras"],
                    seg["end_ras"],
                    radius_mm=exclude_radius_mm,
                )
                keep &= ~seg_mask
            remaining = remaining[keep]

        lines = []
        support_points = remaining.copy()
        center_ras = self._volume_center_ras(volume_node)

        for _ in range(int(max_lines)):
            if remaining.shape[0] < int(min_inliers):
                break

            fit = self._ransac_fit_line(
                remaining,
                distance_threshold_mm=inlier_radius_mm,
                min_inliers=min_inliers,
                iterations=ransac_iterations,
            )
            if fit is None:
                break

            mask = fit["inlier_mask"]
            inliers = remaining[mask]
            center = fit["center"]
            axis = fit["axis"]

            t = (inliers - center.reshape(1, 3)) @ axis
            if t.size < 2:
                remaining = remaining[~mask]
                continue

            lo = float(np.percentile(t, 2.0))
            hi = float(np.percentile(t, 98.0))
            length = hi - lo
            if length < float(min_length_mm):
                remaining = remaining[~mask]
                continue

            p_start = center + axis * lo
            p_end = center + axis * hi

            # Heuristic orientation: target/deep point is closer to volume center.
            d_start = float(np.linalg.norm(p_start - center_ras))
            d_end = float(np.linalg.norm(p_end - center_ras))
            if d_start < d_end:
                entry_ras = p_end
                target_ras = p_start
            else:
                entry_ras = p_start
                target_ras = p_end

            inside_fraction = 1.0
            if gating_mask_kji is not None:
                inside_fraction = self._segment_inside_mask_fraction(
                    volume_node=volume_node,
                    start_ras=entry_ras,
                    end_ras=target_ras,
                    mask_kji=gating_mask_kji,
                    step_mm=1.0,
                )
                if inside_fraction < float(min_inside_fraction):
                    remaining = remaining[~mask]
                    continue
                clamped = self._clamp_segment_to_mask(
                    volume_node=volume_node,
                    start_ras=entry_ras,
                    end_ras=target_ras,
                    mask_kji=gating_mask_kji,
                    step_mm=0.5,
                )
                if clamped is None:
                    remaining = remaining[~mask]
                    continue
                entry_ras, target_ras = clamped
                length = float(np.linalg.norm(np.asarray(entry_ras) - np.asarray(target_ras)))
                if length < float(min_length_mm):
                    remaining = remaining[~mask]
                    continue

            line = {
                "start_ras": [float(entry_ras[0]), float(entry_ras[1]), float(entry_ras[2])],
                "end_ras": [float(target_ras[0]), float(target_ras[1]), float(target_ras[2])],
                "length_mm": float(length),
                "inlier_count": int(fit["inlier_count"]),
                "rms_mm": float(fit["rms_mm"]),
                "inside_fraction": float(inside_fraction),
            }

            lines.append(line)

            remaining = remaining[~mask]
        lines, refine_stats = self._refine_lines_exclusive(
            volume_node=volume_node,
            arr_kji=arr_kji,
            support_points=support_points,
            raw_lines=lines,
            inlier_radius_mm=float(inlier_radius_mm),
            min_length_mm=float(min_length_mm),
            min_inliers=int(min_inliers),
            gating_mask_kji=gating_mask_kji,
            skull_mask_kji=skull_mask_kji,
            min_inside_fraction=float(min_inside_fraction),
            models_by_id=models_by_id,
            min_model_score=min_model_score,
        )
        gap_reject_count = int(refine_stats.get("gap_reject_count", 0))
        duplicate_reject_count = int(refine_stats.get("duplicate_reject_count", 0))
        lines = sorted(
            lines,
            key=lambda line: (
                float(line.get("best_model_score", 0.0)) if line.get("best_model_score") is not None else -1e9,
                float(line.get("inside_fraction", 0.0)),
                float(line.get("inlier_count", 0)),
                -float(line.get("rms_mm", 999.0)),
            ),
            reverse=True,
        )
        return {
            "candidate_count": candidate_count,
            "head_mask_kept_count": head_mask_kept_count,
            "gating_mask_type": gating_mask_type,
            "inside_method": inside_method,
            "metal_in_head_count": metal_in_head_count,
            "depth_kept_count": depth_kept_count,
            "gap_reject_count": int(gap_reject_count),
            "duplicate_reject_count": int(duplicate_reject_count),
            "metal_mask_kji": (arr_kji >= float(threshold)).astype(np.uint8),
            "head_mask_kji": gating_mask_kji.astype(np.uint8) if gating_mask_kji is not None else None,
            "skull_mask_kji": skull_mask_kji.astype(np.uint8) if skull_mask_kji is not None else None,
            "inside_skull_mask_kji": inside_skull_mask_kji.astype(np.uint8)
            if inside_skull_mask_kji is not None
            else None,
            "metal_depth_all_mm": metal_depth_all_mm,
            "metal_depth_values_mm": metal_depth_values_mm,
            "in_mask_points_ras": in_mask_points_ras,
            "lines": lines,
        }

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
            display.SetColor(1.0, 0.2, 0.2)
            display.SetSelectedColor(1.0, 0.4, 0.4)
            display.SetLineThickness(0.5)
            display.SetPointLabelsVisibility(True)
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
