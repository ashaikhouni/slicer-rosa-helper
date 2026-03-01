"""Navigation burn module for THOMAS nucleus burn + DICOM export."""

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

from rosa_workflow import WorkflowPublisher, WorkflowState
from rosa_scene import (
    AtlasCoreService,
    get_or_create_linear_transform,
    preselect_base_volume,
    widget_current_text,
)


class NavigationBurn(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "Navigation Burn"
        self.parent.categories = ["ROSA"]
        self.parent.dependencies = []
        self.parent.contributors = ["Ammar Shaikhouni", "Codex"]
        self.parent.helpText = "Burn THOMAS nucleus into MRI and export DICOM for navigation."


class NavigationBurnWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        super().setup()
        self.logic = NavigationBurnLogic()
        self.workflowState = self.logic.workflow_state
        self.workflowPublisher = self.logic.workflow_publish
        self.workflowNode = self.workflowState.resolve_or_create_workflow_node()
        self.lastBurnedVolumeNodeID = None
        self.navDicomToRosaTransformNodeID = None

        header = qt.QFormLayout()
        self.layout.addLayout(header)
        self.refreshButton = qt.QPushButton("Refresh Workflow Inputs")
        self.refreshButton.clicked.connect(self.onRefreshClicked)
        header.addRow(self.refreshButton)

        self.tabs = qt.QTabWidget()
        self.layout.addWidget(self.tabs)
        self._build_burn_tab()
        self._build_export_tab()

        self.statusText = qt.QPlainTextEdit()
        self.statusText.setReadOnly(True)
        self.statusText.setMaximumBlockCount(2000)
        self.layout.addWidget(self.statusText)
        self.layout.addStretch(1)
        self.onRefreshClicked()

    def log(self, message):
        self.statusText.appendPlainText(str(message))
        self.statusText.ensureCursorVisible()
        try:
            slicer.app.processEvents()
        except Exception:
            pass

    def _build_burn_tab(self):
        tab = qt.QWidget()
        self.tabs.addTab(tab, "Burn Volume")
        form = qt.QFormLayout(tab)

        self.navDicomDirSelector = ctk.ctkPathLineEdit()
        self.navDicomDirSelector.filters = ctk.ctkPathLineEdit.Dirs
        form.addRow("Nav MRI DICOM dir", self.navDicomDirSelector)

        self.importNavDicomButton = qt.QPushButton("Import DICOM MRI")
        self.importNavDicomButton.clicked.connect(self.onImportNavDicomClicked)
        form.addRow(self.importNavDicomButton)

        self.burnInputSelector = slicer.qMRMLNodeComboBox()
        self.burnInputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.burnInputSelector.noneEnabled = True
        self.burnInputSelector.addEnabled = False
        self.burnInputSelector.removeEnabled = False
        self.burnInputSelector.setMRMLScene(slicer.mrmlScene)
        form.addRow("Burn input MRI", self.burnInputSelector)

        self.sideCombo = qt.QComboBox()
        self.sideCombo.addItems(["Left", "Right", "Both"])
        self.sideCombo.setCurrentText("Both")
        form.addRow("Nucleus side", self.sideCombo)

        nucleus_row = qt.QHBoxLayout()
        self.nucleusCombo = qt.QComboBox()
        self.nucleusCombo.setEditable(True)
        self.nucleusCombo.addItem("CM")
        self.nucleusCombo.setCurrentText("CM")
        nucleus_row.addWidget(self.nucleusCombo)
        self.refreshNucleusButton = qt.QPushButton("Refresh")
        self.refreshNucleusButton.clicked.connect(self._refresh_nucleus_combo)
        nucleus_row.addWidget(self.refreshNucleusButton)
        form.addRow("Nucleus", nucleus_row)

        self.fillSpin = qt.QDoubleSpinBox()
        self.fillSpin.setRange(-32768.0, 32767.0)
        self.fillSpin.setDecimals(1)
        self.fillSpin.setSingleStep(50.0)
        self.fillSpin.setValue(1200.0)
        self.fillSpin.setSuffix(" HU")
        form.addRow("Burn fill value", self.fillSpin)

        self.outputNameEdit = qt.QLineEdit("THOMAS_Burned_MRI")
        form.addRow("Output volume name", self.outputNameEdit)

        self.burnButton = qt.QPushButton("Run Burn")
        self.burnButton.clicked.connect(self.onBurnClicked)
        form.addRow(self.burnButton)

    def _build_export_tab(self):
        tab = qt.QWidget()
        self.tabs.addTab(tab, "DICOM Export")
        form = qt.QFormLayout(tab)

        self.exportVolumeSelector = slicer.qMRMLNodeComboBox()
        self.exportVolumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.exportVolumeSelector.noneEnabled = True
        self.exportVolumeSelector.addEnabled = False
        self.exportVolumeSelector.removeEnabled = False
        self.exportVolumeSelector.setMRMLScene(slicer.mrmlScene)
        form.addRow("Volume to export", self.exportVolumeSelector)

        self.referenceSelector = slicer.qMRMLNodeComboBox()
        self.referenceSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.referenceSelector.noneEnabled = True
        self.referenceSelector.addEnabled = False
        self.referenceSelector.removeEnabled = False
        self.referenceSelector.setMRMLScene(slicer.mrmlScene)
        self.referenceSelector.setToolTip("DICOM patient/study context source volume.")
        form.addRow("Reference DICOM volume", self.referenceSelector)

        self.exportDirSelector = ctk.ctkPathLineEdit()
        self.exportDirSelector.filters = ctk.ctkPathLineEdit.Dirs
        form.addRow("Export directory", self.exportDirSelector)

        self.seriesDescriptionEdit = qt.QLineEdit()
        self.seriesDescriptionEdit.setPlaceholderText("THOMAS_CM_BOTH_BURNED")
        form.addRow("Series description", self.seriesDescriptionEdit)

        self.exportButton = qt.QPushButton("Export Selected Volume to DICOM")
        self.exportButton.clicked.connect(self.onExportClicked)
        form.addRow(self.exportButton)

        self.burnAndExportButton = qt.QPushButton("Run Burn + Export DICOM")
        self.burnAndExportButton.clicked.connect(self.onBurnAndExportClicked)
        form.addRow(self.burnAndExportButton)

    def _get_or_create_nav_dicom_transform_node(self):
        return get_or_create_linear_transform("NavMRI_DICOM_to_ROSA")

    def _workflow_segmentation_nodes(self):
        wf = self.workflowState.resolve_or_create_workflow_node()
        return self.workflowState.role_nodes("THOMASSegmentations", workflow_node=wf)

    def _refresh_nucleus_combo(self):
        seg_nodes = self._workflow_segmentation_nodes()
        nuclei = self.logic.core.collect_thomas_nuclei(seg_nodes)
        current = (widget_current_text(self.nucleusCombo) or "").strip()
        self.nucleusCombo.clear()
        if nuclei:
            for nucleus in nuclei:
                self.nucleusCombo.addItem(nucleus)
            if current and current in nuclei:
                self.nucleusCombo.setCurrentText(current)
            else:
                self.nucleusCombo.setCurrentText("CM" if "CM" in nuclei else nuclei[0])
        else:
            self.nucleusCombo.addItem("CM")
            self.nucleusCombo.setCurrentText(current or "CM")

    def _preselect_base_volume(self):
        wf = self.workflowState.resolve_or_create_workflow_node()
        preselect_base_volume(selector=self.burnInputSelector, workflow_node=wf)
        if self.referenceSelector.currentNode() is None:
            preselect_base_volume(selector=self.referenceSelector, workflow_node=wf)

    def onRefreshClicked(self):
        self.workflowNode = self.workflowState.resolve_or_create_workflow_node()
        self._preselect_base_volume()
        self._refresh_nucleus_combo()
        self.log(f"[refresh] THOMAS segmentations: {len(self._workflow_segmentation_nodes())}")

    def onImportNavDicomClicked(self):
        dicom_dir = (self.navDicomDirSelector.currentPath or "").strip()
        if not dicom_dir:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Navigation Burn", "Select DICOM directory first.")
            return
        try:
            volume_node = self.logic.core.load_dicom_scalar_volume_from_directory(
                dicom_dir=dicom_dir,
                logger=self.log,
            )
        except Exception as exc:
            self.log(f"[thomas] DICOM import error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Navigation Burn", str(exc))
            return

        base_node = self.workflowNode.GetNodeReference("BaseVolume")
        if base_node is not None and base_node.GetID() != volume_node.GetID():
            transform_node = self._get_or_create_nav_dicom_transform_node()
            self.log(
                f"[thomas] nav DICOM registration start: moving={volume_node.GetName()} -> "
                f"fixed={base_node.GetName()} (init=useGeometryAlign)"
            )
            try:
                self.logic.core.run_brainsfit_rigid_registration(
                    fixed_volume_node=base_node,
                    moving_volume_node=volume_node,
                    output_transform_node=transform_node,
                    initialize_mode="useGeometryAlign",
                    logger=self.log,
                )
                self.logic.core.apply_transform_to_nodes(
                    nodes=[volume_node],
                    transform_node=transform_node,
                    harden=True,
                )
            except Exception as exc:
                self.log(f"[thomas] nav DICOM registration error: {exc}")
                qt.QMessageBox.critical(slicer.util.mainWindow(), "Navigation Burn", str(exc))
                return
            self.navDicomToRosaTransformNodeID = transform_node.GetID()
            self.workflowPublisher.register_transform(
                transform_node=transform_node,
                from_space="DICOM_NATIVE",
                to_space="ROSA_BASE",
                transform_type="linear",
                status="active",
                workflow_node=self.workflowNode,
            )
            self.log(f"[thomas] nav DICOM registration done: {volume_node.GetName()} aligned to {base_node.GetName()}")
        else:
            self.log("[thomas] nav DICOM registration skipped (no BaseVolume set or same node)")

        self.workflowPublisher.register_volume(
            volume_node=volume_node,
            source_type="dicom",
            source_path=dicom_dir,
            space_name="ROSA_BASE",
            role="AdditionalMRIVolumes",
            workflow_node=self.workflowNode,
        )
        self.burnInputSelector.setCurrentNode(volume_node)
        self.referenceSelector.setCurrentNode(volume_node)
        self.log(f"[thomas] DICOM MRI ready for burn/export: {volume_node.GetName()}")

    def _run_burn(self):
        burn_input_node = self.burnInputSelector.currentNode()
        if burn_input_node is None:
            raise ValueError("Select Burn input MRI volume.")

        seg_nodes = self._workflow_segmentation_nodes()
        burn_seg_nodes = list(seg_nodes)
        temp_volume_nodes = []
        burn_input_for_burn = burn_input_node

        volumes_logic = slicer.modules.volumes.logic()
        transform_logic = slicer.vtkSlicerTransformLogic()
        if volumes_logic is None:
            raise RuntimeError("Volumes logic is unavailable.")

        if not burn_seg_nodes:
            raise RuntimeError("No THOMAS segmentations found in workflow. Load in Atlas Sources first.")

        # Keep original scene nodes untouched when they carry parent transforms.
        if burn_input_node.GetTransformNodeID():
            burn_input_for_burn = volumes_logic.CloneVolume(
                slicer.mrmlScene,
                burn_input_node,
                "__THOMAS_BURN_FIXED",
            )
            temp_volume_nodes.append(burn_input_for_burn)
            transform_logic.hardenTransform(burn_input_for_burn)
            self.log(f"[thomas] using hardened temp fixed volume: {burn_input_for_burn.GetName()}")

        try:
            side = (widget_current_text(self.sideCombo) or "Both").strip()
            nucleus = (widget_current_text(self.nucleusCombo) or "").strip()
            fill_value = float(self.fillSpin.value)
            output_name = self.outputNameEdit.text.strip() or "THOMAS_Burned_MRI"
            out_volume = self.logic.core.burn_thomas_nucleus_to_volume(
                segmentation_nodes=burn_seg_nodes,
                input_volume_node=burn_input_for_burn,
                nucleus=nucleus,
                side=side,
                fill_value=fill_value,
                output_name=output_name,
                logger=self.log,
            )
        finally:
            for node in temp_volume_nodes:
                if node is not None and node.GetScene() is not None:
                    slicer.mrmlScene.RemoveNode(node)

        self.logic.core.place_node_under_same_study(out_volume, burn_input_node, logger=self.log)
        self.logic.core.show_volume_in_all_slice_views(out_volume)
        self.workflowPublisher.register_volume(
            volume_node=out_volume,
            source_type="derived",
            source_path="",
            space_name="ROSA_BASE",
            role="DerivedVolumes",
            is_derived=True,
            derived_from_node_id=burn_input_node.GetID() if burn_input_node is not None else "",
            workflow_node=self.workflowNode,
        )
        self.lastBurnedVolumeNodeID = out_volume.GetID()
        self.exportVolumeSelector.setCurrentNode(out_volume)
        self.log(f"[thomas] burn complete: {out_volume.GetName()}")
        return out_volume

    def _series_description_default(self):
        nucleus = (widget_current_text(self.nucleusCombo) or "NUCLEUS").strip().upper()
        side = (widget_current_text(self.sideCombo) or "BOTH").strip().upper()
        return f"THOMAS_{nucleus}_{side}_BURNED"

    def _export_volume(self, volume_node):
        if volume_node is None:
            raise ValueError("Select a volume to export.")
        reference_volume = self.referenceSelector.currentNode()
        if reference_volume is None:
            reference_volume = self.burnInputSelector.currentNode()
        if reference_volume is None:
            raise ValueError("Select reference DICOM volume.")
        export_dir = (self.exportDirSelector.currentPath or "").strip()
        if not export_dir:
            raise ValueError("Select export directory.")
        series_description = (self.seriesDescriptionEdit.text or "").strip()
        if not series_description:
            series_description = self._series_description_default()
            self.seriesDescriptionEdit.setText(series_description)

        series_dir = self.logic.core.export_scalar_volume_to_dicom_series(
            volume_node=volume_node,
            reference_volume_node=reference_volume,
            export_dir=export_dir,
            series_description=series_description,
            modality="MR",
            logger=self.log,
        )
        self.log(f"[thomas] DICOM export complete: {series_dir}")
        return series_dir

    def onBurnClicked(self):
        try:
            self._run_burn()
        except Exception as exc:
            self.log(f"[thomas] burn failed: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Navigation Burn", str(exc))

    def onExportClicked(self):
        node = self.exportVolumeSelector.currentNode()
        if node is None and self.lastBurnedVolumeNodeID:
            node = slicer.mrmlScene.GetNodeByID(self.lastBurnedVolumeNodeID)
        try:
            self._export_volume(node)
        except Exception as exc:
            self.log(f"[thomas] export failed: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Navigation Burn", str(exc))

    def onBurnAndExportClicked(self):
        try:
            out_volume = self._run_burn()
            self._export_volume(out_volume)
        except Exception as exc:
            self.log(f"[thomas] burn/export failed: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Navigation Burn", str(exc))


class NavigationBurnLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        super().__init__()
        self.core = AtlasCoreService(module_dir=MODULE_DIR)
        self.workflow_state = WorkflowState()
        self.workflow_publish = WorkflowPublisher()
