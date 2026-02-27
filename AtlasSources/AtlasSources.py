"""Atlas source loading module (FreeSurfer + THOMAS + registry view)."""

import importlib.util
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

from rosa_workflow import WorkflowState, WorkflowPublisher
from rosa_workflow.workflow_registry import table_to_dict_rows

_ROSA_HELPER_LOGIC_INSTANCE = None


def _get_rosa_helper_logic_instance():
    """Load and cache `RosaHelperLogic` from module source."""
    global _ROSA_HELPER_LOGIC_INSTANCE
    if _ROSA_HELPER_LOGIC_INSTANCE is not None:
        return _ROSA_HELPER_LOGIC_INSTANCE

    helper_path = os.path.join(os.path.dirname(MODULE_DIR), "RosaHelper", "RosaHelper.py")
    spec = importlib.util.spec_from_file_location("_rosahelper_logic_bridge", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load RosaHelper logic bridge.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _ROSA_HELPER_LOGIC_INSTANCE = module.RosaHelperLogic()
    return _ROSA_HELPER_LOGIC_INSTANCE


class AtlasSources(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "Atlas Sources"
        self.parent.categories = ["ROSA"]
        self.parent.dependencies = []
        self.parent.contributors = ["Ammar Shaikhouni", "Codex"]
        self.parent.helpText = "Load/register FreeSurfer and THOMAS atlas sources into RosaWorkflow."


class AtlasSourcesWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        super().setup()
        self.logic = AtlasSourcesLogic()
        self.workflowState = self.logic.workflow_state
        self.workflowPublisher = self.logic.workflow_publish
        self.workflowNode = self.workflowState.resolve_or_create_workflow_node()

        self.fsToRosaTransformNodeID = None
        self.thomasToRosaTransformNodeID = None

        top_form = qt.QFormLayout()
        self.layout.addLayout(top_form)
        self.refreshButton = qt.QPushButton("Refresh Workflow Inputs")
        self.refreshButton.clicked.connect(self.onRefreshClicked)
        top_form.addRow(self.refreshButton)

        self.tabs = qt.QTabWidget()
        self.layout.addWidget(self.tabs)
        self._build_freesurfer_tab()
        self._build_thomas_tab()
        self._build_registry_tab()

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

    def _build_freesurfer_tab(self):
        tab = qt.QWidget()
        self.tabs.addTab(tab, "FreeSurfer")
        form = qt.QFormLayout(tab)

        self.fsFixedSelector = slicer.qMRMLNodeComboBox()
        self.fsFixedSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.fsFixedSelector.noneEnabled = True
        self.fsFixedSelector.addEnabled = False
        self.fsFixedSelector.removeEnabled = False
        self.fsFixedSelector.setMRMLScene(slicer.mrmlScene)
        form.addRow("ROSA base volume", self.fsFixedSelector)

        self.fsMovingSelector = slicer.qMRMLNodeComboBox()
        self.fsMovingSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.fsMovingSelector.noneEnabled = True
        self.fsMovingSelector.addEnabled = False
        self.fsMovingSelector.removeEnabled = False
        self.fsMovingSelector.setMRMLScene(slicer.mrmlScene)
        form.addRow("FreeSurfer MRI", self.fsMovingSelector)

        self.fsLoadMRIPathSelector = ctk.ctkPathLineEdit()
        self.fsLoadMRIPathSelector.filters = ctk.ctkPathLineEdit.Files
        form.addRow("Load FS MRI file", self.fsLoadMRIPathSelector)
        self.fsLoadMRIButton = qt.QPushButton("Load MRI Into Scene")
        self.fsLoadMRIButton.clicked.connect(self.onLoadFSMRIClicked)
        form.addRow(self.fsLoadMRIButton)

        self.fsInitModeCombo = qt.QComboBox()
        self.fsInitModeCombo.addItems(["useGeometryAlign", "useMomentsAlign"])
        self.fsInitModeCombo.setCurrentText("useGeometryAlign")
        form.addRow("Init mode", self.fsInitModeCombo)

        self.fsTransformNameEdit = qt.QLineEdit("FS_to_ROSA")
        form.addRow("Output transform", self.fsTransformNameEdit)

        self.fsRegisterButton = qt.QPushButton("Register FS MRI -> ROSA")
        self.fsRegisterButton.clicked.connect(self.onRegisterFSMRIToRosaClicked)
        form.addRow(self.fsRegisterButton)

        self.fsSubjectDirSelector = ctk.ctkPathLineEdit()
        self.fsSubjectDirSelector.filters = ctk.ctkPathLineEdit.Dirs
        if hasattr(self.fsSubjectDirSelector, "currentPathChanged"):
            self.fsSubjectDirSelector.currentPathChanged.connect(lambda _p: self._refresh_fs_parcellation_combo())
        form.addRow("FreeSurfer subject", self.fsSubjectDirSelector)

        parc_row = qt.QHBoxLayout()
        self.fsParcellationCombo = qt.QComboBox()
        self.fsParcellationCombo.addItem("all available")
        parc_row.addWidget(self.fsParcellationCombo)
        self.fsRefreshParcellationButton = qt.QPushButton("Refresh")
        self.fsRefreshParcellationButton.clicked.connect(self._refresh_fs_parcellation_combo)
        parc_row.addWidget(self.fsRefreshParcellationButton)
        form.addRow("Parcellation volume", parc_row)

        self.fsApplyTransformVolumesCheck = qt.QCheckBox("Apply FS->ROSA transform to parcellations")
        self.fsApplyTransformVolumesCheck.setChecked(True)
        form.addRow(self.fsApplyTransformVolumesCheck)

        self.fsHardenVolumeCheck = qt.QCheckBox("Harden parcellation transforms")
        self.fsHardenVolumeCheck.setChecked(False)
        form.addRow(self.fsHardenVolumeCheck)

        self.fsApplyParcellationLUTCheck = qt.QCheckBox("Apply LUT to parcellation volumes")
        self.fsApplyParcellationLUTCheck.setChecked(True)
        form.addRow(self.fsApplyParcellationLUTCheck)

        self.fsCreateParcellation3DCheck = qt.QCheckBox("Create 3D geometry from parcellations")
        self.fsCreateParcellation3DCheck.setChecked(False)
        form.addRow(self.fsCreateParcellation3DCheck)

        self.fsLUTPathSelector = ctk.ctkPathLineEdit()
        self.fsLUTPathSelector.filters = ctk.ctkPathLineEdit.Files
        form.addRow("Annotation LUT", self.fsLUTPathSelector)

        self.fsLoadParcellationButton = qt.QPushButton("Load Parcellation Volumes")
        self.fsLoadParcellationButton.clicked.connect(self.onLoadFSParcellationsClicked)
        form.addRow(self.fsLoadParcellationButton)

        self.fsExistingAtlasSelector = slicer.qMRMLNodeComboBox()
        self.fsExistingAtlasSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.fsExistingAtlasSelector.noneEnabled = True
        self.fsExistingAtlasSelector.addEnabled = False
        self.fsExistingAtlasSelector.removeEnabled = False
        self.fsExistingAtlasSelector.setMRMLScene(slicer.mrmlScene)
        self.fsExistingAtlasSelector.setToolTip(
            "Select an already loaded atlas/parcellation volume to publish as FreeSurfer atlas source."
        )
        form.addRow("Use existing atlas volume", self.fsExistingAtlasSelector)

        self.fsPublishExistingAtlasButton = qt.QPushButton("Publish Selected Atlas Volume")
        self.fsPublishExistingAtlasButton.clicked.connect(self.onPublishExistingFSAtlasClicked)
        form.addRow(self.fsPublishExistingAtlasButton)

    def _build_thomas_tab(self):
        tab = qt.QWidget()
        self.tabs.addTab(tab, "THOMAS")
        form = qt.QFormLayout(tab)

        self.thomasFixedSelector = slicer.qMRMLNodeComboBox()
        self.thomasFixedSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.thomasFixedSelector.noneEnabled = True
        self.thomasFixedSelector.addEnabled = False
        self.thomasFixedSelector.removeEnabled = False
        self.thomasFixedSelector.setMRMLScene(slicer.mrmlScene)
        form.addRow("ROSA base volume", self.thomasFixedSelector)

        self.thomasMovingSelector = slicer.qMRMLNodeComboBox()
        self.thomasMovingSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.thomasMovingSelector.noneEnabled = True
        self.thomasMovingSelector.addEnabled = False
        self.thomasMovingSelector.removeEnabled = False
        self.thomasMovingSelector.setMRMLScene(slicer.mrmlScene)
        form.addRow("THOMAS MRI", self.thomasMovingSelector)

        self.thomasLoadMRIPathSelector = ctk.ctkPathLineEdit()
        self.thomasLoadMRIPathSelector.filters = ctk.ctkPathLineEdit.Files
        form.addRow("Load THOMAS MRI file", self.thomasLoadMRIPathSelector)
        self.thomasLoadMRIButton = qt.QPushButton("Load THOMAS MRI Into Scene")
        self.thomasLoadMRIButton.clicked.connect(self.onLoadThomasMRIClicked)
        form.addRow(self.thomasLoadMRIButton)

        self.thomasInitModeCombo = qt.QComboBox()
        self.thomasInitModeCombo.addItems(["useGeometryAlign", "useMomentsAlign"])
        self.thomasInitModeCombo.setCurrentText("useGeometryAlign")
        form.addRow("Init mode", self.thomasInitModeCombo)

        self.thomasTransformNameEdit = qt.QLineEdit("THOMAS_to_ROSA")
        form.addRow("Output transform", self.thomasTransformNameEdit)

        self.thomasRegisterButton = qt.QPushButton("Register THOMAS MRI -> ROSA")
        self.thomasRegisterButton.clicked.connect(self.onRegisterThomasMRIToRosaClicked)
        form.addRow(self.thomasRegisterButton)

        self.thomasMaskDirSelector = ctk.ctkPathLineEdit()
        self.thomasMaskDirSelector.filters = ctk.ctkPathLineEdit.Dirs
        form.addRow("THOMAS output dir", self.thomasMaskDirSelector)

        self.thomasApplyTransformCheck = qt.QCheckBox("Apply THOMAS->ROSA transform")
        self.thomasApplyTransformCheck.setChecked(True)
        form.addRow(self.thomasApplyTransformCheck)

        self.thomasHardenCheck = qt.QCheckBox("Harden loaded thalamus transforms")
        self.thomasHardenCheck.setChecked(True)
        form.addRow(self.thomasHardenCheck)

        self.thomasLoadMasksButton = qt.QPushButton("Load THOMAS Thalamus Masks")
        self.thomasLoadMasksButton.clicked.connect(self.onLoadThomasMasksClicked)
        form.addRow(self.thomasLoadMasksButton)

    def _build_registry_tab(self):
        tab = qt.QWidget()
        self.tabs.addTab(tab, "Registry")
        layout = qt.QVBoxLayout(tab)
        self.registryText = qt.QPlainTextEdit()
        self.registryText.setReadOnly(True)
        layout.addWidget(self.registryText)
        self.registryRefreshButton = qt.QPushButton("Refresh Registry View")
        self.registryRefreshButton.clicked.connect(self.refresh_registry_view)
        layout.addWidget(self.registryRefreshButton)

    def _widget_text(self, widget):
        text_attr = getattr(widget, "currentText", "")
        return text_attr() if callable(text_attr) else text_attr

    def _find_node_by_name(self, node_name, class_name):
        for node in slicer.util.getNodesByClass(class_name):
            if node.GetName() == node_name:
                return node
        return None

    def _get_or_create_transform_node(self, name):
        node = self._find_node_by_name(name, "vtkMRMLLinearTransformNode")
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", name)
        return node

    def _refresh_fs_parcellation_combo(self):
        current = (self._widget_text(self.fsParcellationCombo) or "all available").strip()
        self.fsParcellationCombo.clear()
        self.fsParcellationCombo.addItem("all available")
        subject_dir = self.fsSubjectDirSelector.currentPath
        if not subject_dir:
            return
        try:
            candidates = self.logic.core.list_freesurfer_parcellation_candidates(subject_dir)
        except Exception as exc:
            self.log(f"[fs] parcellation scan error: {exc}")
            return
        # Compatibility: service may return either:
        # - dict with {"available": [{"name": ..., "path": ...}, ...]}
        # - list of strings / list of dict entries
        entries = []
        if isinstance(candidates, dict):
            entries = candidates.get("available", []) or []
        elif isinstance(candidates, (list, tuple)):
            entries = list(candidates)

        for entry in entries:
            if isinstance(entry, str):
                name = entry.strip()
            elif isinstance(entry, dict):
                name = str(entry.get("name", "")).strip()
            else:
                name = str(entry).strip()
            if name:
                self.fsParcellationCombo.addItem(name)
        idx = self.fsParcellationCombo.findText(current)
        self.fsParcellationCombo.setCurrentIndex(idx if idx >= 0 else 0)

    def _preselect_base_volume(self, selector):
        if selector is None:
            return
        wf = self.workflowState.resolve_or_create_workflow_node()
        base = wf.GetNodeReference("BaseVolume")
        if base is not None:
            selector.setCurrentNode(base)

    def onRefreshClicked(self):
        self.workflowNode = self.workflowState.resolve_or_create_workflow_node()
        # Keep registry default flags synced with workflow single-role references.
        base = self.workflowNode.GetNodeReference("BaseVolume")
        postop = self.workflowNode.GetNodeReference("PostopCT")
        if base is not None:
            self.workflowPublisher.set_default_role("BaseVolume", base, workflow_node=self.workflowNode)
        if postop is not None:
            self.workflowPublisher.set_default_role("PostopCT", postop, workflow_node=self.workflowNode)
        self._preselect_base_volume(self.fsFixedSelector)
        self._preselect_base_volume(self.thomasFixedSelector)
        self._refresh_fs_parcellation_combo()
        self.refresh_registry_view()
        self.log("[refresh] workflow inputs refreshed")

    def refresh_registry_view(self):
        wf = self.workflowState.resolve_or_create_workflow_node()
        lines = []
        image_table = wf.GetNodeReference("ImageRegistryTable")
        if image_table is not None:
            lines.append("ImageRegistry:")
            for row in table_to_dict_rows(image_table):
                lines.append(
                    "  - {label} | {source_type} | role-base={is_default_base} role-postop={is_default_postop_ct}".format(
                        **row
                    )
                )
        transform_table = wf.GetNodeReference("TransformRegistryTable")
        if transform_table is not None:
            lines.append("")
            lines.append("TransformRegistry:")
            for row in table_to_dict_rows(transform_table):
                lines.append(
                    f"  - {row.get('transform_node_id','')} | {row.get('from_space','')} -> {row.get('to_space','')}"
                )
        if not lines:
            lines = ["No registry data available."]
        self.registryText.setPlainText("\n".join(lines))

    def onRegisterFSMRIToRosaClicked(self):
        fixed_node = self.fsFixedSelector.currentNode()
        moving_node = self.fsMovingSelector.currentNode()
        if fixed_node is None or moving_node is None:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Atlas Sources", "Select both ROSA base and FS MRI.")
            return
        # Ensure selected FS MRI is published even if user picked it from scene dropdown.
        storage = moving_node.GetStorageNode() if hasattr(moving_node, "GetStorageNode") else None
        source_path = storage.GetFileName() if storage is not None else ""
        self.workflowPublisher.register_volume(
            volume_node=moving_node,
            source_type="freesurfer",
            source_path=source_path or "",
            space_name="FS_NATIVE",
            role="AdditionalMRIVolumes",
            workflow_node=self.workflowNode,
        )
        transform_node = self._get_or_create_transform_node(self.fsTransformNameEdit.text.strip() or "FS_to_ROSA")
        init_mode = self._widget_text(self.fsInitModeCombo) or "useGeometryAlign"
        self.log(
            f"[fs] registration start: moving={moving_node.GetName()} -> fixed={fixed_node.GetName()} (init={init_mode})"
        )
        try:
            self.logic.core.run_brainsfit_rigid_registration(
                fixed_volume_node=fixed_node,
                moving_volume_node=moving_node,
                output_transform_node=transform_node,
                initialize_mode=init_mode,
                logger=self.log,
            )
        except Exception as exc:
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Atlas Sources", str(exc))
            self.log(f"[fs] registration error: {exc}")
            return
        self.fsToRosaTransformNodeID = transform_node.GetID()
        self.workflowPublisher.register_transform(
            transform_node=transform_node,
            from_space="FS_NATIVE",
            to_space="ROSA_BASE",
            transform_type="linear",
            status="active",
            role="FSToBaseTransform",
            workflow_node=self.workflowNode,
        )
        self.log(f"[fs] registration done: transform={transform_node.GetName()}")
        self.refresh_registry_view()

    def onLoadFSMRIClicked(self):
        path = (self.fsLoadMRIPathSelector.currentPath or "").strip()
        if not path:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Atlas Sources", "Select an MRI file to load.")
            return
        if not os.path.isfile(path):
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Atlas Sources", f"File not found:\n{path}")
            return
        try:
            loaded = slicer.util.loadVolume(path)
        except Exception as exc:
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Atlas Sources", str(exc))
            self.log(f"[fs] MRI load error: {exc}")
            return
        if loaded is None:
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Atlas Sources", "Failed to load MRI volume.")
            return
        node = loaded
        if isinstance(loaded, tuple):
            ok, node = loaded
            if not ok:
                qt.QMessageBox.critical(slicer.util.mainWindow(), "Atlas Sources", "Failed to load MRI volume.")
                return
        self.fsMovingSelector.setCurrentNode(node)
        self.workflowPublisher.register_volume(
            volume_node=node,
            source_type="freesurfer",
            source_path=path,
            space_name="FS_NATIVE",
            role="AdditionalMRIVolumes",
            workflow_node=self.workflowNode,
        )
        self.log(f"[fs] loaded MRI volume: {node.GetName()}")
        self.refresh_registry_view()

    def _current_fs_transform_for_volumes(self):
        transform_node = None
        if self.fsToRosaTransformNodeID:
            transform_node = slicer.mrmlScene.GetNodeByID(self.fsToRosaTransformNodeID)
        if transform_node is None:
            transform_node = self._find_node_by_name(
                self.fsTransformNameEdit.text.strip() or "FS_to_ROSA",
                "vtkMRMLLinearTransformNode",
            )
        return transform_node

    def _publish_fs_parcellation_node(self, node, source_path=""):
        if node is None:
            return
        name = (node.GetName() or "").lower()
        role = "WMParcellationVolumes" if "wmparc" in name else "FSParcellationVolumes"
        self.workflowPublisher.register_volume(
            volume_node=node,
            source_type="freesurfer",
            source_path=source_path,
            space_name="ROSA_BASE" if bool(self.fsApplyTransformVolumesCheck.checked) else "FS_NATIVE",
            role=role,
            workflow_node=self.workflowNode,
        )

    def onPublishExistingFSAtlasClicked(self):
        node = self.fsExistingAtlasSelector.currentNode()
        if node is None:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Atlas Sources", "Select a volume to publish.")
            return
        if bool(self.fsApplyTransformVolumesCheck.checked):
            transform_node = self._current_fs_transform_for_volumes()
            if transform_node is None:
                qt.QMessageBox.warning(
                    slicer.util.mainWindow(),
                    "Atlas Sources",
                    "FS->ROSA transform is not available. Run registration first or disable transform application.",
                )
                return
            self.logic.core.apply_transform_to_nodes(
                nodes=[node],
                transform_node=transform_node,
                harden=bool(self.fsHardenVolumeCheck.checked),
            )
            self.log(f"[fs] applied transform {transform_node.GetName()} to selected atlas volume")

        self._publish_fs_parcellation_node(node=node, source_path="")
        self.log(f"[fs] published existing atlas volume: {node.GetName()}")
        self.refresh_registry_view()

    def onLoadFSParcellationsClicked(self):
        subject_dir = self.fsSubjectDirSelector.currentPath
        if not subject_dir:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Atlas Sources", "Select FreeSurfer subject directory.")
            return
        selected = (self._widget_text(self.fsParcellationCombo) or "").strip()
        selected_names = None if selected == "all available" else [selected]
        lut_path = self.fsLUTPathSelector.currentPath.strip() if self.fsLUTPathSelector.currentPath else ""
        try:
            result = self.logic.core.load_freesurfer_parcellation_volumes(
                subject_dir=subject_dir,
                selected_names=selected_names,
                color_lut_path=lut_path or None,
                apply_color_table=bool(self.fsApplyParcellationLUTCheck.checked),
                create_3d_geometry=bool(self.fsCreateParcellation3DCheck.checked),
                logger=self.log,
            )
        except Exception as exc:
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Atlas Sources", str(exc))
            self.log(f"[fs] parcellation load error: {exc}")
            return

        loaded_nodes = result.get("loaded_nodes", [])
        loaded_paths = result.get("loaded_paths", [])
        if bool(self.fsApplyTransformVolumesCheck.checked):
            transform_node = self._current_fs_transform_for_volumes()
            if transform_node is not None and loaded_nodes:
                self.logic.core.apply_transform_to_nodes(
                    nodes=loaded_nodes,
                    transform_node=transform_node,
                    harden=bool(self.fsHardenVolumeCheck.checked),
                )
                self.log(f"[fs] applied transform {transform_node.GetName()} to {len(loaded_nodes)} parcellations")

        for node, path in zip(loaded_nodes, loaded_paths):
            self._publish_fs_parcellation_node(node=node, source_path=path)
        self.log(f"[fs] loaded {len(loaded_nodes)} parcellation volumes")
        self.refresh_registry_view()

    def onRegisterThomasMRIToRosaClicked(self):
        fixed_node = self.thomasFixedSelector.currentNode()
        moving_node = self.thomasMovingSelector.currentNode()
        if fixed_node is None or moving_node is None:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Atlas Sources", "Select both ROSA base and THOMAS MRI.")
            return
        storage = moving_node.GetStorageNode() if hasattr(moving_node, "GetStorageNode") else None
        source_path = storage.GetFileName() if storage is not None else ""
        self.workflowPublisher.register_volume(
            volume_node=moving_node,
            source_type="thomas",
            source_path=source_path or "",
            space_name="THOMAS_NATIVE",
            role="AdditionalMRIVolumes",
            workflow_node=self.workflowNode,
        )
        transform_node = self._get_or_create_transform_node(self.thomasTransformNameEdit.text.strip() or "THOMAS_to_ROSA")
        init_mode = self._widget_text(self.thomasInitModeCombo) or "useGeometryAlign"
        self.log(
            f"[thomas] registration start: moving={moving_node.GetName()} -> fixed={fixed_node.GetName()} (init={init_mode})"
        )
        try:
            self.logic.core.run_brainsfit_rigid_registration(
                fixed_volume_node=fixed_node,
                moving_volume_node=moving_node,
                output_transform_node=transform_node,
                initialize_mode=init_mode,
                logger=self.log,
            )
        except Exception as exc:
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Atlas Sources", str(exc))
            self.log(f"[thomas] registration error: {exc}")
            return
        self.thomasToRosaTransformNodeID = transform_node.GetID()
        self.workflowPublisher.register_transform(
            transform_node=transform_node,
            from_space="THOMAS_NATIVE",
            to_space="ROSA_BASE",
            transform_type="linear",
            status="active",
            role="THOMASToBaseTransform",
            workflow_node=self.workflowNode,
        )
        self.log(f"[thomas] registration done: transform={transform_node.GetName()}")
        self.refresh_registry_view()

    def onLoadThomasMRIClicked(self):
        path = (self.thomasLoadMRIPathSelector.currentPath or "").strip()
        if not path:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Atlas Sources", "Select a THOMAS MRI file to load.")
            return
        if not os.path.isfile(path):
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Atlas Sources", f"File not found:\n{path}")
            return
        try:
            loaded = slicer.util.loadVolume(path)
        except Exception as exc:
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Atlas Sources", str(exc))
            self.log(f"[thomas] MRI load error: {exc}")
            return
        if loaded is None:
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Atlas Sources", "Failed to load THOMAS MRI volume.")
            return
        node = loaded
        if isinstance(loaded, tuple):
            ok, node = loaded
            if not ok:
                qt.QMessageBox.critical(
                    slicer.util.mainWindow(),
                    "Atlas Sources",
                    "Failed to load THOMAS MRI volume.",
                )
                return
        self.thomasMovingSelector.setCurrentNode(node)
        self.workflowPublisher.register_volume(
            volume_node=node,
            source_type="thomas",
            source_path=path,
            space_name="THOMAS_NATIVE",
            role="AdditionalMRIVolumes",
            workflow_node=self.workflowNode,
        )
        self.log(f"[thomas] loaded MRI volume: {node.GetName()}")
        self.refresh_registry_view()

    def onLoadThomasMasksClicked(self):
        thomas_dir = self.thomasMaskDirSelector.currentPath
        if not thomas_dir:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Atlas Sources", "Select THOMAS output directory.")
            return
        try:
            result = self.logic.core.load_thomas_thalamus_masks(
                thomas_dir=thomas_dir,
                logger=self.log,
            )
        except Exception as exc:
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Atlas Sources", str(exc))
            self.log(f"[thomas] load error: {exc}")
            return
        loaded_nodes = result.get("loaded_nodes", [])
        if bool(self.thomasApplyTransformCheck.checked):
            transform_node = None
            if self.thomasToRosaTransformNodeID:
                transform_node = slicer.mrmlScene.GetNodeByID(self.thomasToRosaTransformNodeID)
            if transform_node is None:
                transform_node = self._find_node_by_name(
                    self.thomasTransformNameEdit.text.strip() or "THOMAS_to_ROSA",
                    "vtkMRMLLinearTransformNode",
                )
            if transform_node is not None and loaded_nodes:
                self.logic.core.apply_transform_to_nodes(
                    nodes=loaded_nodes,
                    transform_node=transform_node,
                    harden=bool(self.thomasHardenCheck.checked),
                )
                self.log(f"[thomas] applied transform {transform_node.GetName()} to {len(loaded_nodes)} segmentations")
        self.workflowPublisher.publish_nodes(
            role="THOMASSegmentations",
            nodes=loaded_nodes,
            source="thomas",
            space_name="ROSA_BASE" if bool(self.thomasApplyTransformCheck.checked) else "THOMAS_NATIVE",
            workflow_node=self.workflowNode,
        )
        self.log(f"[thomas] loaded {len(loaded_nodes)} THOMAS segmentation nodes")
        self.refresh_registry_view()


class AtlasSourcesLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        super().__init__()
        self.core = _get_rosa_helper_logic_instance()
        self.workflow_state = WorkflowState()
        self.workflow_publish = WorkflowPublisher(self.workflow_state)
