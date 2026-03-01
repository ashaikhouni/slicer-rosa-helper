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
import importlib
import hashlib
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
LIB_CANDIDATES = [
    os.path.join(os.path.dirname(MODULE_DIR), "CommonLib"),  # source tree shared libs
    os.path.join(MODULE_DIR, "CommonLib"),  # packaged extension shared libs
]
for _lib_dir in LIB_CANDIDATES:
    if os.path.isdir(_lib_dir) and _lib_dir not in sys.path:
        sys.path.insert(0, _lib_dir)

import rosa_core as _rosa_core_mod
_rosa_core_mod = importlib.reload(_rosa_core_mod)

from rosa_core import (
    lps_to_ras_point,
)
from rosa_scene import electrode_scene as _electrode_scene_mod
from rosa_scene import freesurfer_service as _freesurfer_service_mod
from rosa_scene import trajectory_scene as _trajectory_scene_mod
from rosa_scene import case_loader_service as _case_loader_service_mod
from rosa_workflow import export_bundle as _export_bundle_mod
from rosa_workflow import workflow_state as _workflow_state_mod
from rosa_workflow import workflow_publish as _workflow_publish_mod

# Hot-reload helper submodules for dev iteration without full app restart.
_freesurfer_service_mod = importlib.reload(_freesurfer_service_mod)
_trajectory_scene_mod = importlib.reload(_trajectory_scene_mod)
_electrode_scene_mod = importlib.reload(_electrode_scene_mod)
_case_loader_service_mod = importlib.reload(_case_loader_service_mod)
_export_bundle_mod = importlib.reload(_export_bundle_mod)
_workflow_state_mod = importlib.reload(_workflow_state_mod)
_workflow_publish_mod = importlib.reload(_workflow_publish_mod)
FreeSurferService = _freesurfer_service_mod.FreeSurferService
TrajectorySceneService = _trajectory_scene_mod.TrajectorySceneService
ElectrodeSceneService = _electrode_scene_mod.ElectrodeSceneService
CaseLoaderService = _case_loader_service_mod.CaseLoaderService
WorkflowState = _workflow_state_mod.WorkflowState
WorkflowPublisher = _workflow_publish_mod.WorkflowPublisher
export_aligned_bundle_service = _export_bundle_mod.export_aligned_bundle


class RosaHelper(ScriptedLoadableModule):
    """Slicer module metadata container."""

    def __init__(self, parent):
        """Initialize static module metadata shown in Slicer UI."""
        super().__init__(parent)
        self.parent.title = "Loader"
        self.parent.categories = ["ROSA"]
        self.parent.dependencies = []
        self.parent.contributors = ["Ammar Shaikhouni", "Codex"]
        self.parent.helpText = (
            "Load ROSA folders or custom volumes into the shared workflow scene, "
            "set Base/Postop defaults, and register volumes."
        )


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
        self.lastAtlasAssignmentRows = []
        self.loadedVolumeNodeIDs = {}
        self.loadedVolumeSourcePaths = {}
        self.referenceVolumeName = None
        self.autoFitCandidatesLPS = []
        self.autoFitResults = {}
        self.autoFitCandidateVolumeNodeID = None
        self.fsToRosaTransformNodeID = None
        self.fsParcellationVolumeNodeIDs = []
        self.fsParcellationSegNodeIDs = []
        self.thomasToRosaTransformNodeID = None
        self.thomasDicomToRosaTransformNodeID = None
        self.thomasImportedDicomNodeID = None
        self.thomasSegmentationNodeIDs = []
        self.modelsById = {}
        self.modelIds = []
        self.customVolumeSourcePaths = {}
        self.workflowNode = self.logic.workflow_state.resolve_or_create_workflow_node()

        tabs = qt.QTabWidget()
        self.layout.addWidget(tabs)

        rosa_tab = qt.QWidget()
        rosa_form = qt.QFormLayout(rosa_tab)
        tabs.addTab(rosa_tab, "ROSA Load")

        self.caseDirSelector = ctk.ctkPathLineEdit()
        self.caseDirSelector.filters = ctk.ctkPathLineEdit.Dirs
        self.caseDirSelector.setToolTip("Case folder containing .ros and DICOM/")
        rosa_form.addRow("Case folder", self.caseDirSelector)

        self.referenceEdit = qt.QLineEdit()
        self.referenceEdit.setPlaceholderText("Optional (auto-detect if blank)")
        self.referenceEdit.setToolTip(
            "Root display volume name. If blank, the first ROS display is used."
        )
        rosa_form.addRow("Reference volume", self.referenceEdit)

        self.invertCheck = qt.QCheckBox("Invert TRdicomRdisplay")
        self.invertCheck.setChecked(False)
        self.invertCheck.setToolTip(
            "Invert the composed transform before applying. Use only for datasets"
            " where ROS matrices are known to be reversed."
        )
        rosa_form.addRow("Transform option", self.invertCheck)

        self.hardenCheck = qt.QCheckBox("Harden transforms")
        self.hardenCheck.setChecked(True)
        rosa_form.addRow("Scene option", self.hardenCheck)

        self.markupsCheck = qt.QCheckBox("Load trajectories")
        self.markupsCheck.setChecked(True)
        rosa_form.addRow("Trajectory option", self.markupsCheck)

        self.loadButton = qt.QPushButton("Load ROSA case")
        self.loadButton.clicked.connect(self.onLoadClicked)
        rosa_form.addRow(self.loadButton)

        custom_tab = qt.QWidget()
        custom_form = qt.QFormLayout(custom_tab)
        tabs.addTab(custom_tab, "Custom Import")

        self.customVolumePath = ctk.ctkPathLineEdit()
        self.customVolumePath.filters = ctk.ctkPathLineEdit.Files
        self.customVolumePath.setToolTip("Import one MRI/CT volume (NIfTI, NRRD, Analyze, ...).")
        custom_form.addRow("Volume file", self.customVolumePath)

        self.customVolumeNameEdit = qt.QLineEdit()
        self.customVolumeNameEdit.setPlaceholderText("Optional display name")
        custom_form.addRow("Volume name", self.customVolumeNameEdit)

        roles_row = qt.QHBoxLayout()
        self.customSetBaseCheck = qt.QCheckBox("Set as Base")
        self.customSetPostopCheck = qt.QCheckBox("Set as Postop CT")
        roles_row.addWidget(self.customSetBaseCheck)
        roles_row.addWidget(self.customSetPostopCheck)
        custom_form.addRow("Default roles", roles_row)

        self.customImportButton = qt.QPushButton("Import Volume")
        self.customImportButton.clicked.connect(self.onCustomImportClicked)
        custom_form.addRow(self.customImportButton)

        self.customBaseSelector = slicer.qMRMLNodeComboBox()
        self.customBaseSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.customBaseSelector.noneEnabled = False
        self.customBaseSelector.addEnabled = False
        self.customBaseSelector.removeEnabled = False
        self.customBaseSelector.setMRMLScene(slicer.mrmlScene)
        custom_form.addRow("Base volume", self.customBaseSelector)

        self.customPostopSelector = slicer.qMRMLNodeComboBox()
        self.customPostopSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.customPostopSelector.noneEnabled = True
        self.customPostopSelector.addEnabled = False
        self.customPostopSelector.removeEnabled = False
        self.customPostopSelector.setMRMLScene(slicer.mrmlScene)
        custom_form.addRow("Postop CT", self.customPostopSelector)

        self.applyDefaultsButton = qt.QPushButton("Apply Base/Postop Roles")
        self.applyDefaultsButton.clicked.connect(self.onApplyCustomDefaultsClicked)
        custom_form.addRow(self.applyDefaultsButton)

        self.registerFixedSelector = slicer.qMRMLNodeComboBox()
        self.registerFixedSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.registerFixedSelector.noneEnabled = False
        self.registerFixedSelector.addEnabled = False
        self.registerFixedSelector.removeEnabled = False
        self.registerFixedSelector.setMRMLScene(slicer.mrmlScene)
        custom_form.addRow("Register fixed", self.registerFixedSelector)

        self.registerMovingSelector = slicer.qMRMLNodeComboBox()
        self.registerMovingSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.registerMovingSelector.noneEnabled = False
        self.registerMovingSelector.addEnabled = False
        self.registerMovingSelector.removeEnabled = False
        self.registerMovingSelector.setMRMLScene(slicer.mrmlScene)
        custom_form.addRow("Register moving", self.registerMovingSelector)

        self.registerInitModeCombo = qt.QComboBox()
        self.registerInitModeCombo.addItem("useGeometryAlign")
        self.registerInitModeCombo.addItem("useMomentsAlign")
        self.registerInitModeCombo.addItem("Off")
        custom_form.addRow("Init mode", self.registerInitModeCombo)

        reg_role_row = qt.QHBoxLayout()
        self.registerOutputAsBaseCheck = qt.QCheckBox("Output as Base")
        self.registerOutputAsPostopCheck = qt.QCheckBox("Output as Postop CT")
        reg_role_row.addWidget(self.registerOutputAsBaseCheck)
        reg_role_row.addWidget(self.registerOutputAsPostopCheck)
        custom_form.addRow("After register", reg_role_row)

        self.registerVolumesButton = qt.QPushButton("Register + Harden Output")
        self.registerVolumesButton.clicked.connect(self.onRegisterCustomVolumesClicked)
        custom_form.addRow(self.registerVolumesButton)

        self.statusText = qt.QPlainTextEdit()
        self.statusText.setReadOnly(True)
        self.statusText.setMaximumBlockCount(1000)
        self.layout.addWidget(self.statusText)

        # Atlas/burn workflows were extracted into dedicated modules:
        # AtlasSources, AtlasLabeling, NavigationBurn.

        self.layout.addStretch(1)

    def log(self, msg):
        """Append status text to the module log panel and stdout."""
        self.statusText.appendPlainText(msg)
        print(msg)

    def onLoadClicked(self):
        """Validate inputs and run the load pipeline."""
        case_dir = self.caseDirSelector.currentPath
        if not case_dir:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Loader", "Please select a case folder")
            return

        reference = self.referenceEdit.text.strip() or None

        try:
            summary = self.logic.load_case(
                case_dir=case_dir,
                reference=reference,
                invert=self.invertCheck.checked,
                harden=self.hardenCheck.checked,
                load_trajectories=self.markupsCheck.checked,
                show_planned=False,
                logger=self.log,
            )
        except Exception as exc:
            self.log(f"[error] {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Loader", str(exc))
            return

        self.loadedTrajectories = summary["trajectories"]
        self.lastGeneratedContacts = []
        self.lastAssignments = {"schema_version": "1.0", "assignments": []}
        self.lastQCMetricsRows = []
        self.lastAtlasAssignmentRows = []
        self.loadedVolumeNodeIDs = summary.get("loaded_volume_node_ids", {})
        self.loadedVolumeSourcePaths = summary.get("loaded_volume_source_paths", {})
        self.referenceVolumeName = summary.get("reference_volume")
        self.autoFitCandidatesLPS = []
        self.autoFitResults = {}
        self.autoFitCandidateVolumeNodeID = None
        self.fsToRosaTransformNodeID = None
        self.fsParcellationVolumeNodeIDs = []
        self.fsParcellationSegNodeIDs = []
        self.thomasToRosaTransformNodeID = None
        self.thomasDicomToRosaTransformNodeID = None
        self.thomasImportedDicomNodeID = None
        self.thomasSegmentationNodeIDs = []
        self.log(
            f"[done] loaded {summary['loaded_volumes']} volumes, "
            f"created {summary['trajectory_count']} trajectories"
        )
        self._publish_loaded_case_to_workflow(summary)

    def _infer_registry_role_for_volume(self, volume_node):
        """Infer workflow multi-volume role for one imported scalar volume."""
        name = (volume_node.GetName() or "").lower()
        if "ct" in name:
            return "AdditionalCTVolumes"
        return "AdditionalMRIVolumes"

    def _infer_source_type_for_volume(self, volume_node):
        """Infer source type hint for one selected volume when not explicit."""
        source = (volume_node.GetAttribute("Rosa.Source") or "").strip().lower()
        return source or "import"

    def _register_volume_in_workflow(
        self,
        volume_node,
        source_type=None,
        source_path=None,
        is_default_base=False,
        is_default_postop=False,
        is_derived=False,
        derived_from_node_id="",
    ):
        """Upsert one volume into image registry + relevant workflow roles."""
        if volume_node is None:
            return None
        wf = self.logic.workflow_state.resolve_or_create_workflow_node()
        self.workflowNode = wf
        role = self._infer_registry_role_for_volume(volume_node)
        src_type = source_type or self._infer_source_type_for_volume(volume_node)
        src_path = source_path or self.customVolumeSourcePaths.get(volume_node.GetID(), "")
        published = self.logic.workflow_publish.register_volume(
            volume_node=volume_node,
            source_type=src_type,
            source_path=src_path,
            space_name="ROSA_BASE",
            role=role,
            is_default_base=bool(is_default_base),
            is_default_postop=bool(is_default_postop),
            is_derived=bool(is_derived),
            derived_from_node_id=derived_from_node_id or "",
            workflow_node=wf,
        )
        if published is not None and src_path:
            published.SetAttribute("Rosa.SourcePath", src_path)
            self.customVolumeSourcePaths[published.GetID()] = src_path
        return published

    def onCustomImportClicked(self):
        """Import one user-selected volume and publish to workflow registry/roles."""
        path = (self.customVolumePath.currentPath or "").strip()
        if not path:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Loader", "Select a volume file to import.")
            return
        if not os.path.isfile(path):
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Loader", f"File not found:\n{path}")
            return
        if self.customSetBaseCheck.checked and self.customSetPostopCheck.checked:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "Loader",
                "Select at most one default role during import (Base or Postop CT).",
            )
            return

        node = self.logic._load_volume(path)
        if node is None:
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Loader", f"Failed to load volume:\n{path}")
            return

        custom_name = (self.customVolumeNameEdit.text or "").strip()
        if custom_name:
            node.SetName(custom_name)
        node.SetAttribute("Rosa.SourcePath", path)
        self.customVolumeSourcePaths[node.GetID()] = path

        published = self._register_volume_in_workflow(
            volume_node=node,
            source_type="import",
            source_path=path,
            is_default_base=bool(self.customSetBaseCheck.checked),
            is_default_postop=bool(self.customSetPostopCheck.checked),
        )
        if published is None:
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Loader", "Failed to publish imported volume.")
            return

        if self.customSetBaseCheck.checked:
            self.customBaseSelector.setCurrentNode(published)
            self.registerFixedSelector.setCurrentNode(published)
        if self.customSetPostopCheck.checked:
            self.customPostopSelector.setCurrentNode(published)

        self.log(f"[custom] imported {published.GetName()} ({path})")
        self.customVolumeNameEdit.clear()
        self.customSetBaseCheck.setChecked(False)
        self.customSetPostopCheck.setChecked(False)

    def onApplyCustomDefaultsClicked(self):
        """Set current base/postop defaults from scene-selected volumes."""
        base_node = self.customBaseSelector.currentNode()
        postop_node = self.customPostopSelector.currentNode()
        if base_node is None:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Loader", "Select a base volume.")
            return
        if postop_node is not None and postop_node.GetID() == base_node.GetID():
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "Loader",
                "Base and Postop CT cannot reference the same volume.",
            )
            return

        self._register_volume_in_workflow(volume_node=base_node, is_default_base=True)
        if postop_node is not None:
            self._register_volume_in_workflow(volume_node=postop_node, is_default_postop=True)
        self.log(
            f"[custom] defaults updated: base={base_node.GetName()}, "
            f"postop={(postop_node.GetName() if postop_node else 'unset')}"
        )

    def onRegisterCustomVolumesClicked(self):
        """Register moving->fixed, harden in-place, and keep a single output volume."""
        fixed = self.registerFixedSelector.currentNode()
        moving = self.registerMovingSelector.currentNode()
        if fixed is None or moving is None:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Loader", "Select both fixed and moving volumes.")
            return
        if fixed.GetID() == moving.GetID():
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Loader", "Fixed and moving volumes must differ.")
            return
        if self.registerOutputAsBaseCheck.checked and self.registerOutputAsPostopCheck.checked:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "Loader",
                "Select at most one output default role (Base or Postop CT).",
            )
            return

        init_mode = self.registerInitModeCombo.currentText
        transform_name = f"{moving.GetName()}_to_{fixed.GetName()}"
        transform_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", transform_name)
        try:
            ok = self.logic.run_brainsfit_rigid_registration(
                fixed_volume_node=fixed,
                moving_volume_node=moving,
                output_transform_node=transform_node,
                initialize_mode=init_mode,
                logger=self.log,
            )
            if not ok:
                raise RuntimeError("BRAINSFit registration failed.")

            moving_name_before = moving.GetName()
            moving.SetAndObserveTransformNodeID(transform_node.GetID())
            slicer.vtkSlicerTransformLogic().hardenTransform(moving)
            moving.SetAndObserveTransformNodeID(None)

            # Keep native->base transform for later coordinate transfer and show it under workflow folder.
            if hasattr(transform_node, "SetHideFromEditors"):
                transform_node.SetHideFromEditors(False)
            transform_node.SetAttribute("Rosa.Managed", "1")
            transform_node.SetAttribute("Rosa.Source", "import_registration")
            transform_node.SetAttribute("Rosa.Space", "ROSA_BASE")
            transform_node.SetAttribute("Rosa.NativeToBaseForNodeID", moving.GetID())
            transform_node.SetAttribute("Rosa.NativeToBaseForNodeName", moving_name_before)

            moving_source = moving.GetAttribute("Rosa.SourcePath") or self.customVolumeSourcePaths.get(
                moving.GetID(), ""
            )
            published = self._register_volume_in_workflow(
                volume_node=moving,
                source_type="import",
                source_path=moving_source,
                is_default_base=bool(self.registerOutputAsBaseCheck.checked),
                is_default_postop=bool(self.registerOutputAsPostopCheck.checked),
            )
            if published is None:
                raise RuntimeError("Failed to publish hardened registered volume.")

            self.logic.workflow_publish.register_transform(
                transform_node=transform_node,
                from_space=f"{moving_name_before}_NATIVE",
                to_space="ROSA_BASE",
                transform_type="linear",
                status="active",
                role=None,
                workflow_node=self.workflowNode,
            )
            published.SetAttribute("Rosa.NativeToBaseTransformID", transform_node.GetID())

            if self.registerOutputAsBaseCheck.checked:
                self.customBaseSelector.setCurrentNode(published)
                self.registerFixedSelector.setCurrentNode(published)
            if self.registerOutputAsPostopCheck.checked:
                self.customPostopSelector.setCurrentNode(published)
            self.registerMovingSelector.setCurrentNode(published)

            self.log(
                f"[custom] registered {moving.GetName()} -> {fixed.GetName()} "
                f"and hardened in-place ({published.GetName()}); "
                f"native->base transform kept as hidden metadata ({transform_node.GetName()})"
            )
            self.registerOutputAsBaseCheck.setChecked(False)
            self.registerOutputAsPostopCheck.setChecked(False)
        except Exception as exc:
            if transform_node is not None and slicer.mrmlScene.GetNodeByID(transform_node.GetID()) is not None:
                slicer.mrmlScene.RemoveNode(transform_node)
            self.log(f"[custom] registration failed: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Loader", str(exc))

    def _publish_loaded_case_to_workflow(self, summary):
        """Publish loaded ROSA volumes and trajectories into shared workflow roles."""
        wf = self.logic.workflow_state.resolve_or_create_workflow_node()
        self.workflowNode = wf

        ros_nodes = []
        for volume_name, node_id in sorted((self.loadedVolumeNodeIDs or {}).items()):
            node = slicer.mrmlScene.GetNodeByID(node_id)
            if node is None:
                continue
            source_path = (self.loadedVolumeSourcePaths or {}).get(volume_name, "")
            published = self.logic.workflow_publish.register_volume(
                volume_node=node,
                source_type="rosa",
                source_path=source_path,
                space_name="ROSA_BASE",
                role="ROSAVolumes",
                is_default_base=(volume_name == self.referenceVolumeName),
                workflow_node=wf,
            )
            if published is not None:
                ros_nodes.append(published)

        if ros_nodes:
            self.logic.workflow_publish.publish_nodes(
                role="ROSAVolumes",
                nodes=ros_nodes,
                source="rosa",
                space_name="ROSA_BASE",
                workflow_node=wf,
            )
            if wf.GetNodeReference("BaseVolume") is None:
                self.logic.workflow_publish.set_default_base_volume(ros_nodes[0], workflow_node=wf)

        # Publish ROSA native->base transforms into transform registry.
        for rec in summary.get("ros_transform_records", []) or []:
            transform_node = slicer.mrmlScene.GetNodeByID(rec.get("transform_node_id", ""))
            if transform_node is None:
                continue
            self.logic.workflow_publish.register_transform(
                transform_node=transform_node,
                from_space=rec.get("from_space", ""),
                to_space=rec.get("to_space", "ROSA_BASE"),
                transform_type="linear",
                status="active",
                role=None,
                workflow_node=wf,
            )
            vol_name = rec.get("volume_name", "")
            vol_node_id = (self.loadedVolumeNodeIDs or {}).get(vol_name, "")
            vol_node = slicer.mrmlScene.GetNodeByID(vol_node_id) if vol_node_id else None
            if vol_node is not None:
                vol_node.SetAttribute("Rosa.NativeToBaseTransformID", transform_node.GetID())

        planned_nodes = []
        imported_nodes = []
        for traj in self.loadedTrajectories or []:
            name = traj.get("name", "")
            if not name:
                continue
            node = self.logic.trajectory_scene.find_line_by_group_and_name(name, "imported_rosa")
            if node is None:
                node = self.logic._find_node_by_name(name, "vtkMRMLMarkupsLineNode")
            if node is not None:
                imported_nodes.append(node)
            plan_node = self.logic.trajectory_scene.find_line_by_group_and_name(name, "planned_rosa")
            if plan_node is None:
                plan_node = self.logic._find_node_by_name(f"Plan_{name}", "vtkMRMLMarkupsLineNode")
            if plan_node is not None:
                planned_nodes.append(plan_node)

        self.logic.workflow_publish.publish_nodes(
            role="WorkingTrajectoryLines",
            nodes=imported_nodes,
            source="rosa",
            space_name="ROSA_BASE",
            workflow_node=wf,
        )
        self.logic.workflow_publish.publish_nodes(
            role="ImportedTrajectoryLines",
            nodes=imported_nodes,
            source="rosa",
            space_name="ROSA_BASE",
            workflow_node=wf,
        )
        self.logic.workflow_publish.publish_nodes(
            role="PlannedTrajectoryLines",
            nodes=planned_nodes,
            source="rosa",
            space_name="ROSA_BASE",
            workflow_node=wf,
        )
        self.logic.trajectory_scene.place_trajectory_nodes_in_hierarchy(
            context_id=self.logic.workflow_state.context_id(workflow_node=wf),
            nodes=(imported_nodes + planned_nodes),
        )
        self.logic.trajectory_scene.show_only_groups(["imported_rosa"])




class RosaHelperLogic(ScriptedLoadableModuleLogic):
    """Core scene-loading logic used by UI and headless `run()` entrypoint."""

    def __init__(self):
        """Initialize service delegates used by Slicer-specific workflows."""
        super().__init__()
        self.case_loader = CaseLoaderService()
        self.workflow_state = WorkflowState()
        self.workflow_publish = WorkflowPublisher(self.workflow_state)
        self.fs_service = FreeSurferService(module_dir=MODULE_DIR)
        self.trajectory_scene = TrajectorySceneService()
        self.electrode_scene = ElectrodeSceneService(
            workflow_state=self.workflow_state,
            workflow_publish=self.workflow_publish,
        )

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

        return self.case_loader.load_case(
            case_dir=case_dir,
            reference=reference,
            invert=invert,
            harden=harden,
            load_trajectories=load_trajectories,
            show_planned=show_planned,
            add_trajectories_fn=self._add_trajectories,
            logger=logger,
        )

    def _load_volume(self, path):
        """Load a scalar volume by path and return the MRML node."""
        return self.case_loader.load_volume(path)

    def _center_volume(self, volume_node):
        """Center volume origin in Slicer (equivalent to Volumes->Center Volume)."""
        self.case_loader.center_volume(volume_node)

    def _apply_transform(self, volume_node, matrix4x4):
        """Create and assign a linear transform node from a 4x4 matrix."""
        return self.case_loader.apply_transform(volume_node, matrix4x4)

    def _vtk_matrix_to_numpy(self, vtk_matrix4x4):
        """Convert vtkMatrix4x4 to a NumPy 4x4 array."""
        return self.case_loader.vtk_matrix_to_numpy(vtk_matrix4x4)

    def extract_threshold_candidates_lps(self, volume_node, threshold, max_points=300000):
        """Extract thresholded CT candidate points in LPS coordinates."""
        return self.case_loader.extract_threshold_candidates_lps(
            volume_node=volume_node,
            threshold=threshold,
            max_points=max_points,
        )

    def show_volume_in_all_slice_views(self, volume_node):
        """Set provided volume as background in Red/Yellow/Green slice views."""
        self.case_loader.show_volume_in_all_slice_views(volume_node)

    def apply_ct_window_from_threshold(self, volume_node, threshold):
        """Apply CT window/level preset centered around the current detection threshold."""
        self.case_loader.apply_ct_window_from_threshold(volume_node, threshold)

    def reset_ct_window(self, volume_node):
        """Reset CT window/level to Slicer auto mode for selected volume."""
        self.case_loader.reset_ct_window(volume_node)

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

    def list_freesurfer_parcellation_candidates(self, subject_dir):
        """Return detected FreeSurfer parcellation files under mri/."""
        return self.fs_service.freesurfer_parcellation_candidates(subject_dir)

    def load_freesurfer_parcellation_volumes(
        self,
        subject_dir,
        selected_names=None,
        color_lut_path=None,
        apply_color_table=True,
        create_3d_geometry=False,
        logger=None,
    ):
        """Load selected FreeSurfer parcellation volumes from recon-all mri/."""
        return self.fs_service.load_freesurfer_parcellation_volumes(
            subject_dir=subject_dir,
            selected_names=selected_names,
            color_lut_path=color_lut_path,
            apply_color_table=apply_color_table,
            create_3d_geometry=create_3d_geometry,
            logger=logger,
        )

    def apply_transform_to_model_nodes(self, model_nodes, transform_node, harden=False):
        """Delegate model transform application/hardening to service layer."""
        return self.fs_service.apply_transform_to_model_nodes(
            model_nodes=model_nodes,
            transform_node=transform_node,
            harden=harden,
        )

    def apply_transform_to_nodes(self, nodes, transform_node, harden=False):
        """Apply one linear transform to any transformable MRML nodes."""
        if transform_node is None:
            raise ValueError("Transform node is required.")
        transform_id = transform_node.GetID()
        for node in nodes or []:
            if node is None:
                continue
            if hasattr(node, "SetAndObserveTransformNodeID"):
                node.SetAndObserveTransformNodeID(transform_id)
                if bool(harden):
                    slicer.vtkSlicerTransformLogic().hardenTransform(node)
        return nodes

    def _load_label_volume_node(self, path):
        """Load one NIfTI mask as labelmap volume node."""
        try:
            result = slicer.util.loadLabelVolume(path, returnNode=True)
            if isinstance(result, tuple):
                ok, node = result
                return node if ok else None
            return result
        except Exception:
            try:
                result = slicer.util.loadVolume(path, properties={"labelmap": True}, returnNode=True)
                if isinstance(result, tuple):
                    ok, node = result
                    return node if ok else None
                return result
            except Exception:
                return None

    def _infer_thomas_side(self, path):
        """Infer side tag from path/name."""
        text = (path or "").lower()
        if "/left/" in text or text.endswith("/left") or "lh" in os.path.basename(text):
            return "left"
        if "/right/" in text or text.endswith("/right") or "rh" in os.path.basename(text):
            return "right"
        return "unknown"

    def _thomas_segment_name_from_path(self, path):
        """Build display segment name from THOMAS file name."""
        base = os.path.basename(path)
        name = base.replace(".nii.gz", "").replace(".nii", "")
        side = self._infer_thomas_side(path)
        if name.lower().endswith("_l") or name.lower().endswith("_r"):
            name = name[:-2]
        # Common THOMAS naming: "11-CM", "6_VLPd", "CL_L", "1-THALAMUS"
        # Prefer anatomical token over numeric index for readability.
        token = name
        if "-" in token:
            token = token.split("-", 1)[1]
        elif "_" in token:
            parts = token.split("_")
            if parts and parts[0].isdigit():
                token = "_".join(parts[1:]) or token
        token = token.strip("_- ") or name
        return f"{side.upper()}_{token}"

    def _thomas_color_for_label(self, segment_name, side):
        """Return deterministic RGB color for THOMAS segment label."""
        key = (segment_name or "UNKNOWN").upper()
        base_map = {
            "THALAMUS": (0.90, 0.80, 0.20),
            "CM": (0.95, 0.35, 0.35),
            "VA": (0.95, 0.55, 0.20),
            "VLA": (0.90, 0.55, 0.45),
            "VL": (0.85, 0.50, 0.55),
            "VLP": (0.85, 0.45, 0.70),
            "VLPD": (0.75, 0.45, 0.80),
            "VLPV": (0.75, 0.55, 0.85),
            "VPL": (0.45, 0.70, 0.95),
            "PUL": (0.40, 0.75, 0.70),
            "LGN": (0.35, 0.75, 0.55),
            "MGN": (0.35, 0.65, 0.50),
            "MD-PF": (0.60, 0.70, 0.45),
            "HB": (0.45, 0.85, 0.45),
            "MTT": (0.60, 0.85, 0.45),
            "AV": (0.75, 0.75, 0.35),
        }
        token = key.split("_", 1)[-1] if "_" in key else key
        color = base_map.get(token)
        if color is None:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            val = int(digest[:8], 16)
            # Stable fallback palette sampling in HSV-like space.
            r = 0.35 + ((val >> 16) & 0xFF) / 255.0 * 0.55
            g = 0.35 + ((val >> 8) & 0xFF) / 255.0 * 0.55
            b = 0.35 + (val & 0xFF) / 255.0 * 0.55
            color = (r, g, b)

        # Side tint to make L/R distinguishable when same nucleus is shown.
        if side == "left":
            return (max(0.0, color[0] * 0.90), min(1.0, color[1] * 1.03), min(1.0, color[2] * 1.08))
        if side == "right":
            return (min(1.0, color[0] * 1.08), min(1.0, color[1] * 1.02), max(0.0, color[2] * 0.90))
        return color

    def _style_thomas_segmentation(self, seg_node, side):
        """Apply standard display and hemisphere color for THOMAS segmentations."""
        if seg_node is None:
            return
        seg_node.CreateDefaultDisplayNodes()
        display = seg_node.GetDisplayNode()
        if display:
            display.SetVisibility(True)
            if hasattr(display, "SetVisibility2D"):
                display.SetVisibility2D(True)
            if hasattr(display, "SetVisibility3D"):
                display.SetVisibility3D(True)
            if hasattr(display, "SetOpacity2DFill"):
                display.SetOpacity2DFill(0.25)
            if hasattr(display, "SetOpacity2DOutline"):
                display.SetOpacity2DOutline(1.0)
            if hasattr(display, "SetOpacity3D"):
                display.SetOpacity3D(0.55)
            if hasattr(display, "SetPreferredDisplayRepresentationName2D"):
                display.SetPreferredDisplayRepresentationName2D("Binary labelmap")
            if hasattr(display, "SetPreferredDisplayRepresentationName3D"):
                display.SetPreferredDisplayRepresentationName3D("Closed surface")

    def _find_thomas_mask_paths(self, thomas_dir):
        """Collect THOMAS structure masks under left/right dirs (top-level only)."""
        root = os.path.abspath(thomas_dir)
        if not os.path.isdir(root):
            raise ValueError(f"THOMAS output directory not found: {thomas_dir}")
        by_side = {"left": [], "right": []}
        skipped = []
        excluded_tokens = (
            "crop",
            "resampled",
            "ocrop",
            "thomasfull",
            "regn_",
            "sthomas",
            "mask_inp",
        )
        for side in ("left", "right"):
            side_dir = os.path.join(root, side)
            if not os.path.isdir(side_dir):
                continue
            files = sorted(os.listdir(side_dir))
            for fname in files:
                full = os.path.join(side_dir, fname)
                if not os.path.isfile(full):
                    continue
                lower = fname.lower()
                if not (lower.endswith(".nii") or lower.endswith(".nii.gz")):
                    continue
                if any(tok in lower for tok in excluded_tokens):
                    skipped.append(full)
                    continue
                by_side[side].append(full)
        return by_side, skipped

    def load_thomas_thalamus_masks(
        self,
        thomas_dir,
        logger=None,
        replace_existing=True,
        node_name_prefix="THOMAS_",
    ):
        """Load THOMAS left/right structure masks into segmentation nodes."""
        if not hasattr(slicer.modules, "segmentations"):
            raise RuntimeError("Segmentations module is not available in this Slicer install.")
        seg_logic = slicer.modules.segmentations.logic()
        if seg_logic is None:
            raise RuntimeError("Segmentations logic is unavailable.")

        by_side, skipped = self._find_thomas_mask_paths(thomas_dir)
        total_candidates = sum(len(v) for v in by_side.values())
        if total_candidates == 0:
            raise ValueError(
                "No THOMAS structure masks found under left/right directories "
                "(top-level only; EXTRAS/cropped/resampled files are ignored)."
            )

        loaded_nodes = []
        loaded_paths = []
        failed_paths = []
        for side in ("left", "right"):
            paths = by_side.get(side, [])
            if not paths:
                continue
            seg_name = f"{node_name_prefix}{side.capitalize()}_Structures"
            if replace_existing:
                existing = self._find_node_by_name(seg_name, "vtkMRMLSegmentationNode")
                if existing is not None:
                    slicer.mrmlScene.RemoveNode(existing)
            seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", seg_name)
            seg_node.CreateDefaultDisplayNodes()
            self._style_thomas_segmentation(seg_node, side)
            segmentation = seg_node.GetSegmentation()
            for path in paths:
                label_node = self._load_label_volume_node(path)
                if label_node is None:
                    failed_paths.append(path)
                    continue
                pre_count = segmentation.GetNumberOfSegments()
                seg_logic.ImportLabelmapToSegmentationNode(label_node, seg_node)
                post_count = segmentation.GetNumberOfSegments()
                if post_count <= pre_count:
                    failed_paths.append(path)
                    slicer.mrmlScene.RemoveNode(label_node)
                    continue

                new_segment_id = segmentation.GetNthSegmentID(post_count - 1)
                new_segment = segmentation.GetSegment(new_segment_id)
                if new_segment is not None:
                    seg_label = self._thomas_segment_name_from_path(path)
                    new_segment.SetName(seg_label)
                    color = self._thomas_color_for_label(seg_label, side)
                    new_segment.SetColor(float(color[0]), float(color[1]), float(color[2]))

                loaded_paths.append(path)
                slicer.mrmlScene.RemoveNode(label_node)
                if logger:
                    logger(f"[thomas] loaded structure: {path}")

            if segmentation.GetNumberOfSegments() > 0:
                loaded_nodes.append(seg_node)
            else:
                slicer.mrmlScene.RemoveNode(seg_node)

        return {
            "loaded_nodes": loaded_nodes,
            "loaded_mask_paths": loaded_paths,
            "failed_mask_paths": failed_paths,
            "missing_mask_paths": [],
            "skipped_mask_paths": skipped,
        }

    def load_dicom_scalar_volume_from_directory(self, dicom_dir, logger=None):
        """Load one scalar DICOM series from a directory.

        The directory is expected to contain a single series (or a sub-tree where one
        series dominates). The highest-confidence scalar loadable is selected.
        """
        root = os.path.abspath(dicom_dir)
        if not os.path.isdir(root):
            raise ValueError(f"DICOM directory not found: {dicom_dir}")

        files = []
        for walk_root, _dirs, names in os.walk(root):
            for name in names:
                if name.startswith("."):
                    continue
                path = os.path.join(walk_root, name)
                if os.path.isfile(path):
                    files.append(path)
        files.sort()
        if not files:
            raise ValueError(f"No files found under DICOM directory: {dicom_dir}")

        node = None
        try:
            from DICOMScalarVolumePlugin import DICOMScalarVolumePluginClass

            plugin = DICOMScalarVolumePluginClass()
            loadables = plugin.examine([files]) or []
            if loadables:
                loadables.sort(
                    key=lambda l: (float(getattr(l, "confidence", 0.0)), len(getattr(l, "files", []) or [])),
                    reverse=True,
                )
                chosen = loadables[0]
                chosen.selected = True
                if logger:
                    logger(
                        f"[thomas] DICOM candidate: '{getattr(chosen, 'name', 'series')}' "
                        f"confidence={float(getattr(chosen, 'confidence', 0.0)):.2f} "
                        f"files={len(getattr(chosen, 'files', []) or [])}"
                    )
                before_ids = {n.GetID() for n in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")}
                loaded = plugin.load(chosen)
                if hasattr(loaded, "GetID"):
                    node = loaded
                else:
                    after_nodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
                    for after_node in after_nodes:
                        if after_node.GetID() not in before_ids:
                            node = after_node
                            break
        except Exception as exc:
            if logger:
                logger(f"[thomas] DICOM plugin import/examine failed, fallback to direct load: {exc}")

        if node is None:
            # Fallback: try loading files directly until one succeeds.
            for path in files:
                try:
                    result = slicer.util.loadVolume(path, returnNode=True)
                    if isinstance(result, tuple):
                        ok, candidate = result
                        if ok and candidate is not None:
                            node = candidate
                            break
                    elif result is not None:
                        node = result
                        break
                except Exception:
                    continue

        if node is None:
            raise RuntimeError(
                "Failed to load DICOM scalar volume from directory. "
                "Select a single-series folder and try again."
            )

        if logger:
            logger(f"[thomas] loaded DICOM MRI: {node.GetName()}")
        return node

    def export_scalar_volume_to_dicom_series(
        self,
        volume_node,
        reference_volume_node,
        export_dir,
        series_description,
        modality="MR",
        logger=None,
    ):
        """Export scalar volume as classic DICOM series (one file per slice).

        Parameters
        ----------
        volume_node: vtkMRMLScalarVolumeNode
            Volume to export.
        reference_volume_node: vtkMRMLScalarVolumeNode
            Reference DICOM volume used to inherit patient/study context.
        export_dir: str
            Destination directory.
        series_description: str
            DICOM SeriesDescription.
        modality: str
            DICOM modality tag (default `MR`).
        """
        if volume_node is None:
            raise ValueError("Volume node is required for DICOM export.")
        if reference_volume_node is None:
            raise ValueError("Reference volume node is required for DICOM export.")

        out_root = os.path.abspath(export_dir)
        os.makedirs(out_root, exist_ok=True)

        sh_node = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        if sh_node is None:
            raise RuntimeError("Subject hierarchy is unavailable.")

        # Ensure output node is placed under the same study/patient as reference.
        self.place_node_under_same_study(volume_node, reference_volume_node, logger=logger)

        volume_item = sh_node.GetItemByDataNode(volume_node)
        reference_item = sh_node.GetItemByDataNode(reference_volume_node)
        if volume_item == 0:
            raise RuntimeError("Output volume is not in subject hierarchy.")
        if reference_item == 0:
            raise RuntimeError("Reference volume is not in subject hierarchy.")

        reference_study_item = sh_node.GetItemParent(reference_item)
        reference_patient_item = sh_node.GetItemParent(reference_study_item) if reference_study_item else 0
        if reference_study_item:
            sh_node.SetItemParent(volume_item, reference_study_item)

        from DICOMScalarVolumePlugin import DICOMScalarVolumePluginClass

        plugin = DICOMScalarVolumePluginClass()
        exportables = plugin.examineForExport(volume_item) or []
        if not exportables:
            raise RuntimeError("DICOM scalar volume export is unavailable for selected output volume.")
        exportable = exportables[0]
        exportable.directory = out_root
        exportable.setTag("SeriesDescription", series_description or "THOMAS_BURNED")
        if modality:
            exportable.setTag("Modality", modality)

        constants = slicer.vtkMRMLSubjectHierarchyConstants
        patient_tags = [
            constants.GetDICOMPatientNameTagName,
            constants.GetDICOMPatientIDTagName,
            constants.GetDICOMPatientBirthDateTagName,
            constants.GetDICOMPatientSexTagName,
            constants.GetDICOMPatientCommentsTagName,
        ]
        study_tags = [
            constants.GetDICOMStudyIDTagName,
            constants.GetDICOMStudyDateTagName,
            constants.GetDICOMStudyTimeTagName,
            constants.GetDICOMStudyDescriptionTagName,
        ]
        for getter in patient_tags:
            tag_name = getter()
            value = sh_node.GetItemAttribute(reference_patient_item, tag_name) if reference_patient_item else ""
            if value:
                exportable.setTag(tag_name, value)
        for getter in study_tags:
            tag_name = getter()
            value = sh_node.GetItemAttribute(reference_study_item, tag_name) if reference_study_item else ""
            if value:
                exportable.setTag(tag_name, value)

        if logger:
            logger(
                f"[thomas] DICOM export start: volume={volume_node.GetName()} "
                f"series='{exportable.tag('SeriesDescription')}' out={out_root}"
            )
        err = plugin.export([exportable])
        if err:
            raise RuntimeError(err)

        series_dir = os.path.join(out_root, f"ScalarVolume_{int(volume_item)}")
        if logger:
            logger(f"[thomas] DICOM export wrote series directory: {series_dir}")
        return series_dir

    def place_node_under_same_study(self, node, reference_node, logger=None):
        """Place node under the same subject-hierarchy study item as reference node."""
        if node is None or reference_node is None:
            return False
        sh_node = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        if sh_node is None:
            return False
        node_item = sh_node.GetItemByDataNode(node)
        ref_item = sh_node.GetItemByDataNode(reference_node)
        if node_item == 0 or ref_item == 0:
            return False
        ref_study = sh_node.GetItemParent(ref_item)
        if not ref_study:
            return False
        sh_node.SetItemParent(node_item, ref_study)
        if logger:
            logger(
                f"[thomas] moved {node.GetName()} under study of {reference_node.GetName()}"
            )
        return True

    def _thomas_nucleus_from_segment_name(self, name):
        """Extract normalized nucleus token from segment name (for example LEFT_CM -> CM)."""
        text = (name or "").strip().upper()
        if text.startswith("LEFT_"):
            return text[5:]
        if text.startswith("RIGHT_"):
            return text[6:]
        return text

    def _thomas_side_from_segment_name(self, name, node_name=""):
        """Infer side label from segment and node names."""
        segment_text = (name or "").upper()
        node_text = (node_name or "").upper()
        if segment_text.startswith("LEFT_") or node_text.startswith("THOMAS_LEFT"):
            return "left"
        if segment_text.startswith("RIGHT_") or node_text.startswith("THOMAS_RIGHT"):
            return "right"
        return "unknown"

    def collect_thomas_nuclei(self, segmentation_nodes):
        """Return sorted unique nucleus names present in THOMAS segmentations."""
        nuclei = set()
        for seg_node in segmentation_nodes or []:
            if seg_node is None:
                continue
            segmentation = seg_node.GetSegmentation()
            if segmentation is None:
                continue
            for i in range(segmentation.GetNumberOfSegments()):
                seg_id = segmentation.GetNthSegmentID(i)
                segment = segmentation.GetSegment(seg_id)
                if segment is None:
                    continue
                nucleus = self._thomas_nucleus_from_segment_name(segment.GetName())
                if nucleus:
                    nuclei.add(nucleus)
        return sorted(nuclei)

    def burn_thomas_nucleus_to_volume(
        self,
        segmentation_nodes,
        input_volume_node,
        nucleus,
        side="Both",
        fill_value=1200.0,
        output_name="THOMAS_Burned_MRI",
        logger=None,
    ):
        """Burn selected THOMAS nucleus voxels into a cloned scalar volume.

        Parameters
        ----------
        segmentation_nodes: list[vtkMRMLSegmentationNode]
            Loaded THOMAS segmentation nodes (left/right).
        input_volume_node: vtkMRMLScalarVolumeNode
            Destination MRI volume whose intensities are modified at selected voxels.
        nucleus: str
            Target nucleus token (for example `CM`, `THALAMUS`, `VLPD`).
        side: str
            `Left`, `Right`, or `Both`.
        fill_value: float
            Scalar value written at selected voxels.
        output_name: str
            Name of cloned output volume.
        """
        if np is None:
            raise RuntimeError("NumPy is required for burn workflow.")
        if input_volume_node is None:
            raise ValueError("Input volume node is required.")
        nucleus_token = (nucleus or "").strip().upper()
        if not nucleus_token:
            raise ValueError("Nucleus name is required.")

        side_text = (side or "Both").strip().lower()
        if side_text not in ("left", "right", "both"):
            raise ValueError("Side must be Left, Right, or Both.")
        target_sides = {"left", "right"} if side_text == "both" else {side_text}

        selected_segments = []
        for seg_node in segmentation_nodes or []:
            if seg_node is None:
                continue
            segmentation = seg_node.GetSegmentation()
            if segmentation is None:
                continue
            node_name = seg_node.GetName() or ""
            for i in range(segmentation.GetNumberOfSegments()):
                seg_id = segmentation.GetNthSegmentID(i)
                segment = segmentation.GetSegment(seg_id)
                if segment is None:
                    continue
                seg_name = segment.GetName() or ""
                seg_side = self._thomas_side_from_segment_name(seg_name, node_name=node_name)
                seg_nucleus = self._thomas_nucleus_from_segment_name(seg_name)
                if seg_side in target_sides and seg_nucleus == nucleus_token:
                    selected_segments.append((seg_node, seg_id, seg_name))

        if not selected_segments:
            available = ", ".join(self.collect_thomas_nuclei(segmentation_nodes))
            raise ValueError(
                f"No THOMAS segments matched nucleus '{nucleus_token}' and side '{side}'. "
                f"Available nuclei: {available or 'none'}"
            )

        volumes_logic = slicer.modules.volumes.logic()
        if volumes_logic is None:
            raise RuntimeError("Volumes logic is unavailable.")
        out_volume = volumes_logic.CloneVolume(slicer.mrmlScene, input_volume_node, output_name)
        if out_volume is None:
            raise RuntimeError("Failed to create output burn volume.")

        seg_logic = slicer.modules.segmentations.logic()
        if seg_logic is None:
            raise RuntimeError("Segmentations logic is unavailable.")

        out_arr = slicer.util.arrayFromVolume(out_volume)
        fill_cast = np.asarray([fill_value], dtype=out_arr.dtype)[0]
        total_voxels = 0

        for seg_node, seg_id, seg_name in selected_segments:
            labelmap_node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLLabelMapVolumeNode",
                f"__tmp_{seg_node.GetName()}_{seg_id}",
            )
            ids = vtk.vtkStringArray()
            ids.InsertNextValue(seg_id)
            ok = seg_logic.ExportSegmentsToLabelmapNode(
                seg_node,
                ids,
                labelmap_node,
                input_volume_node,
            )
            if not ok:
                slicer.mrmlScene.RemoveNode(labelmap_node)
                raise RuntimeError(f"Failed exporting segment '{seg_name}' to labelmap.")
            label_arr = slicer.util.arrayFromVolume(labelmap_node)
            mask = label_arr > 0
            voxels = int(mask.sum())
            if voxels > 0:
                out_arr[mask] = fill_cast
                total_voxels += voxels
            slicer.mrmlScene.RemoveNode(labelmap_node)

        slicer.util.arrayFromVolumeModified(out_volume)
        if logger:
            logger(
                f"[thomas] burned nucleus {nucleus_token} ({side}) using {len(selected_segments)} segment(s), "
                f"voxels={total_voxels}, fill={float(fill_cast)} -> {out_volume.GetName()}"
            )
        return out_volume

    def _add_trajectories(self, trajectories, logger=None, show_planned=False):
        """Create editable trajectory lines and hidden planned backups."""
        for traj in trajectories:
            start_ras = lps_to_ras_point(traj["start"])
            end_ras = lps_to_ras_point(traj["end"])
            # Editable working trajectory.
            node = self.trajectory_scene.create_or_update_trajectory_line(
                name=traj["name"],
                start_ras=start_ras,
                end_ras=end_ras,
                node_name=traj["name"],
                group="imported_rosa",
                origin="rosa",
            )

            # Immutable planned backup trajectory.
            plan_node = self.trajectory_scene.create_or_update_trajectory_line(
                name=traj["name"],
                start_ras=start_ras,
                end_ras=end_ras,
                node_name=f"Plan_{traj['name']}",
                group="planned_rosa",
                origin="rosa",
            )
            plan_display = plan_node.GetDisplayNode()
            if plan_display:
                plan_display.SetVisibility(bool(show_planned))

        if logger:
            logger(f"[markups] created {len(trajectories)} line trajectories")

    def set_planned_trajectory_visibility(self, visible):
        """Compatibility wrapper delegating to shared scene service."""
        return self.electrode_scene.set_planned_trajectory_visibility(visible)

    def create_contacts_fiducials_node(self, contacts, node_name="ROSA_Contacts"):
        """Compatibility wrapper delegating to shared scene service."""
        return self.electrode_scene.create_contacts_fiducials_node(
            contacts=contacts,
            node_name=node_name,
        )

    def create_contacts_fiducials_nodes_by_trajectory(self, contacts, node_prefix="ROSA_Contacts"):
        """Compatibility wrapper delegating to shared scene service."""
        return self.electrode_scene.create_contacts_fiducials_nodes_by_trajectory(
            contacts=contacts,
            node_prefix=node_prefix,
        )

    def _find_node_by_name(self, node_name, class_name):
        """Return first node with exact name and class, or None."""
        for node in slicer.util.getNodesByClass(class_name):
            if node.GetName() == node_name:
                return node
        return None

    def create_or_update_table_node(self, node_name, columns, rows):
        """Compatibility wrapper delegating to shared scene service."""
        return self.electrode_scene.create_or_update_table_node(
            node_name=node_name,
            columns=columns,
            rows=rows,
        )

    def _ensure_subject_hierarchy_folder(self, parent_item_id, folder_name):
        """Create/reuse one subject-hierarchy folder under parent."""
        sh_node = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        if sh_node is None:
            return 0
        if hasattr(sh_node, "GetItemChildWithName"):
            existing = sh_node.GetItemChildWithName(parent_item_id, folder_name)
            if existing:
                return existing
        return sh_node.CreateFolderItem(parent_item_id, folder_name)

    def place_electrode_nodes_in_hierarchy(
        self,
        context_id,
        contact_nodes_by_traj,
        model_nodes_by_traj,
    ):
        """Compatibility wrapper delegating to shared scene service."""
        return self.electrode_scene.place_electrode_nodes_in_hierarchy(
            context_id=context_id,
            contact_nodes_by_traj=contact_nodes_by_traj,
            model_nodes_by_traj=model_nodes_by_traj,
        )

    def publish_contacts_outputs(
        self,
        contact_nodes_by_traj,
        model_nodes_by_traj,
        assignment_rows,
        qc_rows,
        workflow_node=None,
    ):
        """Compatibility wrapper delegating to shared scene service."""
        return self.electrode_scene.publish_contacts_outputs(
            contact_nodes_by_traj=contact_nodes_by_traj,
            model_nodes_by_traj=model_nodes_by_traj,
            assignment_rows=assignment_rows,
            qc_rows=qc_rows,
            workflow_node=workflow_node,
        )

    def publish_atlas_assignment_rows(self, atlas_rows, workflow_node=None):
        """Publish atlas assignment rows into workflow table role."""
        wf = workflow_node or self.workflow_state.resolve_or_create_workflow_node()
        columns = [
            "trajectory",
            "contact_label",
            "contact_index",
            "x_ras",
            "y_ras",
            "z_ras",
            "closest_source",
            "closest_label",
            "closest_label_value",
            "closest_distance_to_voxel_mm",
            "closest_distance_to_centroid_mm",
            "primary_source",
            "primary_label",
            "primary_label_value",
            "primary_distance_to_voxel_mm",
            "primary_distance_to_centroid_mm",
            "thomas_label",
            "thomas_label_value",
            "thomas_distance_to_voxel_mm",
            "thomas_distance_to_centroid_mm",
            "freesurfer_label",
            "freesurfer_label_value",
            "freesurfer_distance_to_voxel_mm",
            "freesurfer_distance_to_centroid_mm",
            "wm_label",
            "wm_label_value",
            "wm_distance_to_voxel_mm",
            "wm_distance_to_centroid_mm",
            "thomas_native_x_ras",
            "thomas_native_y_ras",
            "thomas_native_z_ras",
            "freesurfer_native_x_ras",
            "freesurfer_native_y_ras",
            "freesurfer_native_z_ras",
            "wm_native_x_ras",
            "wm_native_y_ras",
            "wm_native_z_ras",
        ]
        rows = []
        for row in atlas_rows or []:
            p_ras = row.get("contact_ras", [0.0, 0.0, 0.0])
            th_native = row.get("thomas_native_ras", [0.0, 0.0, 0.0])
            fs_native = row.get("freesurfer_native_ras", [0.0, 0.0, 0.0])
            wm_native = row.get("wm_native_ras", [0.0, 0.0, 0.0])
            rows.append(
                {
                    "trajectory": row.get("trajectory", ""),
                    "contact_label": row.get("contact_label", ""),
                    "contact_index": row.get("contact_index", 0),
                    "x_ras": float(p_ras[0]),
                    "y_ras": float(p_ras[1]),
                    "z_ras": float(p_ras[2]),
                    "closest_source": row.get("closest_source", ""),
                    "closest_label": row.get("closest_label", ""),
                    "closest_label_value": row.get("closest_label_value", 0),
                    "closest_distance_to_voxel_mm": row.get("closest_distance_to_voxel_mm", 0.0),
                    "closest_distance_to_centroid_mm": row.get("closest_distance_to_centroid_mm", 0.0),
                    "primary_source": row.get("primary_source", ""),
                    "primary_label": row.get("primary_label", ""),
                    "primary_label_value": row.get("primary_label_value", 0),
                    "primary_distance_to_voxel_mm": row.get("primary_distance_to_voxel_mm", 0.0),
                    "primary_distance_to_centroid_mm": row.get("primary_distance_to_centroid_mm", 0.0),
                    "thomas_label": row.get("thomas_label", ""),
                    "thomas_label_value": row.get("thomas_label_value", 0),
                    "thomas_distance_to_voxel_mm": row.get("thomas_distance_to_voxel_mm", 0.0),
                    "thomas_distance_to_centroid_mm": row.get("thomas_distance_to_centroid_mm", 0.0),
                    "freesurfer_label": row.get("freesurfer_label", ""),
                    "freesurfer_label_value": row.get("freesurfer_label_value", 0),
                    "freesurfer_distance_to_voxel_mm": row.get("freesurfer_distance_to_voxel_mm", 0.0),
                    "freesurfer_distance_to_centroid_mm": row.get("freesurfer_distance_to_centroid_mm", 0.0),
                    "wm_label": row.get("wm_label", ""),
                    "wm_label_value": row.get("wm_label_value", 0),
                    "wm_distance_to_voxel_mm": row.get("wm_distance_to_voxel_mm", 0.0),
                    "wm_distance_to_centroid_mm": row.get("wm_distance_to_centroid_mm", 0.0),
                    "thomas_native_x_ras": float(th_native[0]),
                    "thomas_native_y_ras": float(th_native[1]),
                    "thomas_native_z_ras": float(th_native[2]),
                    "freesurfer_native_x_ras": float(fs_native[0]),
                    "freesurfer_native_y_ras": float(fs_native[1]),
                    "freesurfer_native_z_ras": float(fs_native[2]),
                    "wm_native_x_ras": float(wm_native[0]),
                    "wm_native_y_ras": float(wm_native[1]),
                    "wm_native_z_ras": float(wm_native[2]),
                }
            )
        table = self.create_or_update_table_node(
            node_name="Rosa_AtlasAssignments",
            columns=columns,
            rows=rows,
        )
        self.workflow_state.set_single_role("AtlasAssignmentTable", table, workflow_node=wf)
        self.workflow_state.tag_node(
            table,
            role="AtlasAssignmentTable",
            source="atlas",
            space="ROSA_BASE",
            signature=table.GetID(),
            workflow_node=wf,
        )

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
        """Compatibility wrapper delegating to shared scene service."""
        return self.electrode_scene.create_electrode_models_by_trajectory(
            contacts=contacts,
            trajectories_by_name=trajectories_by_name,
            models_by_id=models_by_id,
            node_prefix=node_prefix,
        )

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
        final_trajectories=None,
        qc_rows=None,
        atlas_rows=None,
        output_frame_node=None,
        export_profile="full_bundle",
    ):
        """Export aligned scene outputs through shared workflow export service."""
        return export_aligned_bundle_service(
            volume_node_ids=volume_node_ids,
            contacts=contacts,
            out_dir=out_dir,
            node_prefix=node_prefix,
            planned_trajectories=planned_trajectories,
            final_trajectories=final_trajectories,
            qc_rows=qc_rows,
            atlas_rows=atlas_rows,
            output_frame_node=output_frame_node,
            export_profile=export_profile,
        )

    def _vtk_matrix_to_numpy_4x4(self, vtk_matrix):
        """Return 4x4 NumPy matrix copied from vtkMatrix4x4."""
        out = np.eye(4, dtype=float)
        for r in range(4):
            for c in range(4):
                out[r, c] = float(vtk_matrix.GetElement(r, c))
        return out

    def _volume_ijk_to_world_matrix(self, volume_node):
        """Return IJK->world-RAS 4x4 matrix, including linear parent transform if present."""
        ijk_to_ras_vtk = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(ijk_to_ras_vtk)
        ijk_to_world = self._vtk_matrix_to_numpy_4x4(ijk_to_ras_vtk)

        parent = volume_node.GetParentTransformNode()
        if parent is None:
            return ijk_to_world
        parent_to_world_vtk = vtk.vtkMatrix4x4()
        if not parent.GetMatrixTransformToWorld(parent_to_world_vtk):
            return ijk_to_world
        parent_to_world = self._vtk_matrix_to_numpy_4x4(parent_to_world_vtk)
        return parent_to_world @ ijk_to_world

    def _world_to_node_ras_matrix(self, node):
        """Return 4x4 matrix mapping world RAS into node-local RAS coordinates."""
        if node is None:
            return np.eye(4, dtype=float)
        parent = node.GetParentTransformNode()
        if parent is None:
            return np.eye(4, dtype=float)
        world_to_local_vtk = vtk.vtkMatrix4x4()
        if not parent.GetMatrixTransformFromWorld(world_to_local_vtk):
            return np.eye(4, dtype=float)
        return self._vtk_matrix_to_numpy_4x4(world_to_local_vtk)

    def _world_to_node_ras_point(self, node, point_world_ras):
        """Map world RAS point into node-local RAS coordinates."""
        mat = self._world_to_node_ras_matrix(node)
        vec = np.array([float(point_world_ras[0]), float(point_world_ras[1]), float(point_world_ras[2]), 1.0])
        out = mat @ vec
        return [float(out[0]), float(out[1]), float(out[2])]

    def _volume_label_lookup_name(self, volume_node, label_value):
        """Resolve label name via display color node when available."""
        if volume_node is None:
            return ""
        display = volume_node.GetDisplayNode()
        if display is None:
            return ""
        color_node = display.GetColorNode()
        if color_node is None:
            return ""
        try:
            name = color_node.GetColorName(int(label_value))
        except Exception:
            name = ""
        if not name:
            return ""
        return str(name)

    def _build_volume_label_index(self, volume_node):
        """Build point-locator index for non-zero voxels in a label volume."""
        if np is None:
            raise RuntimeError("NumPy is required for atlas assignment.")
        if volume_node is None:
            return None
        arr = slicer.util.arrayFromVolume(volume_node)  # KJI
        idx = np.argwhere(arr > 0)
        if idx.size == 0:
            return None

        labels = arr[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.int32)
        ijk_h = np.ones((idx.shape[0], 4), dtype=float)
        ijk_h[:, 0] = idx[:, 2].astype(float)
        ijk_h[:, 1] = idx[:, 1].astype(float)
        ijk_h[:, 2] = idx[:, 0].astype(float)

        ijk_to_world = self._volume_ijk_to_world_matrix(volume_node)
        ras = ijk_h @ ijk_to_world.T
        ras = ras[:, :3]

        points = vtk.vtkPoints()
        points.SetNumberOfPoints(int(ras.shape[0]))
        for i in range(int(ras.shape[0])):
            points.SetPoint(i, float(ras[i, 0]), float(ras[i, 1]), float(ras[i, 2]))
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        locator = vtk.vtkPointLocator()
        locator.SetDataSet(poly)
        locator.BuildLocator()

        centroids = {}
        label_values = np.unique(labels)
        for label_value in label_values:
            mask = labels == int(label_value)
            xyz = ras[mask]
            if xyz.size == 0:
                continue
            centroids[int(label_value)] = xyz.mean(axis=0)

        label_names = {}
        for value in label_values:
            name = self._volume_label_lookup_name(volume_node, int(value))
            label_names[int(value)] = name or f"Label_{int(value)}"

        return {
            "locator": locator,
            "points": points,
            "labels": labels,
            "centroids": centroids,
            "label_names": label_names,
        }

    def _build_segmentation_label_index(
        self,
        segmentation_nodes,
        reference_volume_node,
        skip_generic_thalamus=True,
    ):
        """Build point-locator index from segmentation voxels for nearest-segment lookup.

        For THOMAS, generic whole-thalamus masks can overlap all nuclei and dominate
        nearest-neighbor lookup. By default we skip those segments so contacts map to
        specific nuclei; fallback to full set if nothing remains.
        """
        if np is None:
            raise RuntimeError("NumPy is required for atlas assignment.")
        if not segmentation_nodes:
            return None
        if reference_volume_node is None:
            raise ValueError("Reference volume node is required for segmentation atlas indexing.")
        if not hasattr(slicer.modules, "segmentations"):
            return None
        seg_logic = slicer.modules.segmentations.logic()
        if seg_logic is None:
            return None

        all_points = []
        all_labels = []
        label_names = {}
        label_value = 1
        for seg_node in segmentation_nodes:
            if seg_node is None:
                continue
            segmentation = seg_node.GetSegmentation()
            if segmentation is None:
                continue
            for i in range(segmentation.GetNumberOfSegments()):
                seg_id = segmentation.GetNthSegmentID(i)
                segment = segmentation.GetSegment(seg_id)
                if segment is None:
                    continue
                seg_name = segment.GetName() or ""
                if skip_generic_thalamus:
                    nucleus = self._thomas_nucleus_from_segment_name(seg_name)
                    if nucleus == "THALAMUS":
                        continue
                label_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "__tmp_atlas_seg")
                ids = vtk.vtkStringArray()
                ids.InsertNextValue(seg_id)
                ok = seg_logic.ExportSegmentsToLabelmapNode(seg_node, ids, label_node, reference_volume_node)
                if not ok:
                    slicer.mrmlScene.RemoveNode(label_node)
                    continue
                arr = slicer.util.arrayFromVolume(label_node)
                idx = np.argwhere(arr > 0)
                if idx.size == 0:
                    slicer.mrmlScene.RemoveNode(label_node)
                    continue
                ijk_h = np.ones((idx.shape[0], 4), dtype=float)
                ijk_h[:, 0] = idx[:, 2].astype(float)
                ijk_h[:, 1] = idx[:, 1].astype(float)
                ijk_h[:, 2] = idx[:, 0].astype(float)
                ijk_to_ras_vtk = vtk.vtkMatrix4x4()
                label_node.GetIJKToRASMatrix(ijk_to_ras_vtk)
                ijk_to_ras = self._vtk_matrix_to_numpy_4x4(ijk_to_ras_vtk)
                ras = ijk_h @ ijk_to_ras.T
                ras = ras[:, :3]
                all_points.append(ras)
                all_labels.append(np.full((ras.shape[0],), int(label_value), dtype=np.int32))
                label_names[int(label_value)] = seg_name or f"Segment_{label_value}"
                label_value += 1
                slicer.mrmlScene.RemoveNode(label_node)

        if not all_points:
            # Fallback when only generic masks are available.
            if skip_generic_thalamus:
                return self._build_segmentation_label_index(
                    segmentation_nodes=segmentation_nodes,
                    reference_volume_node=reference_volume_node,
                    skip_generic_thalamus=False,
                )
            return None

        ras_all = np.vstack(all_points)
        labels_all = np.concatenate(all_labels)

        points = vtk.vtkPoints()
        points.SetNumberOfPoints(int(ras_all.shape[0]))
        for i in range(int(ras_all.shape[0])):
            points.SetPoint(i, float(ras_all[i, 0]), float(ras_all[i, 1]), float(ras_all[i, 2]))
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        locator = vtk.vtkPointLocator()
        locator.SetDataSet(poly)
        locator.BuildLocator()

        centroids = {}
        for value in np.unique(labels_all):
            xyz = ras_all[labels_all == int(value)]
            if xyz.size == 0:
                continue
            centroids[int(value)] = xyz.mean(axis=0)

        return {
            "locator": locator,
            "points": points,
            "labels": labels_all,
            "centroids": centroids,
            "label_names": label_names,
        }

    def _query_label_index(self, index_cache, point_ras):
        """Return nearest label match info from precomputed index cache."""
        if not index_cache:
            return {
                "label": "",
                "label_value": 0,
                "distance_to_voxel_mm": 0.0,
                "distance_to_centroid_mm": 0.0,
            }
        locator = index_cache["locator"]
        point_id = int(locator.FindClosestPoint(float(point_ras[0]), float(point_ras[1]), float(point_ras[2])))
        nearest = [0.0, 0.0, 0.0]
        index_cache["points"].GetPoint(point_id, nearest)
        dx = float(point_ras[0]) - float(nearest[0])
        dy = float(point_ras[1]) - float(nearest[1])
        dz = float(point_ras[2]) - float(nearest[2])
        dv = math.sqrt(dx * dx + dy * dy + dz * dz)

        label_value = int(index_cache["labels"][point_id])
        label_name = index_cache["label_names"].get(label_value, f"Label_{label_value}")
        centroid = index_cache["centroids"].get(label_value)
        if centroid is None:
            dc = 0.0
        else:
            cx = float(point_ras[0]) - float(centroid[0])
            cy = float(point_ras[1]) - float(centroid[1])
            cz = float(point_ras[2]) - float(centroid[2])
            dc = math.sqrt(cx * cx + cy * cy + cz * cz)

        return {
            "label": label_name,
            "label_value": label_value,
            "distance_to_voxel_mm": float(dv),
            "distance_to_centroid_mm": float(dc),
        }

    def assign_contacts_to_atlases(
        self,
        contacts,
        freesurfer_volume_node=None,
        thomas_segmentation_nodes=None,
        wm_volume_node=None,
        reference_volume_node=None,
        prefer_thomas=True,
    ):
        """Assign each contact to nearest voxel label in selected atlas sources.

        Returns a list of rows suitable for CSV export.
        """
        if np is None:
            raise RuntimeError("NumPy is required for atlas assignment.")

        fs_idx = self._build_volume_label_index(freesurfer_volume_node) if freesurfer_volume_node else None
        wm_idx = self._build_volume_label_index(wm_volume_node) if wm_volume_node else None
        th_idx = None
        if thomas_segmentation_nodes:
            th_idx = self._build_segmentation_label_index(thomas_segmentation_nodes, reference_volume_node)
        th_native_node = None
        if thomas_segmentation_nodes:
            for node in thomas_segmentation_nodes:
                if node is not None:
                    th_native_node = node
                    break

        rows = []
        for contact in contacts or []:
            point_ras = lps_to_ras_point(contact.get("position_lps", [0.0, 0.0, 0.0]))
            th_native_ras = (
                self._world_to_node_ras_point(th_native_node, point_ras)
                if th_native_node is not None
                else [0.0, 0.0, 0.0]
            )
            fs_native_ras = (
                self._world_to_node_ras_point(freesurfer_volume_node, point_ras)
                if freesurfer_volume_node is not None
                else [0.0, 0.0, 0.0]
            )
            wm_native_ras = (
                self._world_to_node_ras_point(wm_volume_node, point_ras)
                if wm_volume_node is not None
                else [0.0, 0.0, 0.0]
            )
            th = self._query_label_index(th_idx, point_ras) if th_idx else None
            fs = self._query_label_index(fs_idx, point_ras) if fs_idx else None
            wm = self._query_label_index(wm_idx, point_ras) if wm_idx else None

            choices = []
            if fs is not None and fs.get("label"):
                choices.append(("freesurfer", fs))
            if th is not None and th.get("label"):
                choices.append(("thomas", th))
            if wm is not None and wm.get("label"):
                choices.append(("wm", wm))

            closest_source = ""
            closest = {"label": "", "label_value": 0, "distance_to_voxel_mm": 0.0, "distance_to_centroid_mm": 0.0}
            if choices:
                closest_source, closest = min(
                    choices,
                    key=lambda item: float(item[1].get("distance_to_voxel_mm", float("inf"))),
                )

            primary_source = ""
            primary = {"label": "", "label_value": 0, "distance_to_voxel_mm": 0.0, "distance_to_centroid_mm": 0.0}
            if prefer_thomas and th is not None and th.get("label"):
                primary_source, primary = "thomas", th
            else:
                primary_source, primary = closest_source, closest

            rows.append(
                {
                    "trajectory": contact.get("trajectory", ""),
                    "contact_label": contact.get("label", ""),
                    "contact_index": int(contact.get("index", 0)),
                    "contact_ras": [float(point_ras[0]), float(point_ras[1]), float(point_ras[2])],
                    "closest_source": closest_source,
                    "closest_label": closest.get("label", ""),
                    "closest_label_value": int(closest.get("label_value", 0)),
                    "closest_distance_to_voxel_mm": float(closest.get("distance_to_voxel_mm", 0.0)),
                    "closest_distance_to_centroid_mm": float(closest.get("distance_to_centroid_mm", 0.0)),
                    "primary_source": primary_source,
                    "primary_label": primary.get("label", ""),
                    "primary_label_value": int(primary.get("label_value", 0)),
                    "primary_distance_to_voxel_mm": float(primary.get("distance_to_voxel_mm", 0.0)),
                    "primary_distance_to_centroid_mm": float(primary.get("distance_to_centroid_mm", 0.0)),
                    "thomas_label": "" if th is None else th.get("label", ""),
                    "thomas_label_value": 0 if th is None else int(th.get("label_value", 0)),
                    "thomas_distance_to_voxel_mm": 0.0 if th is None else float(th.get("distance_to_voxel_mm", 0.0)),
                    "thomas_distance_to_centroid_mm": 0.0 if th is None else float(th.get("distance_to_centroid_mm", 0.0)),
                    "freesurfer_label": "" if fs is None else fs.get("label", ""),
                    "freesurfer_label_value": 0 if fs is None else int(fs.get("label_value", 0)),
                    "freesurfer_distance_to_voxel_mm": 0.0 if fs is None else float(fs.get("distance_to_voxel_mm", 0.0)),
                    "freesurfer_distance_to_centroid_mm": 0.0 if fs is None else float(fs.get("distance_to_centroid_mm", 0.0)),
                    "wm_label": "" if wm is None else wm.get("label", ""),
                    "wm_label_value": 0 if wm is None else int(wm.get("label_value", 0)),
                    "wm_distance_to_voxel_mm": 0.0 if wm is None else float(wm.get("distance_to_voxel_mm", 0.0)),
                    "wm_distance_to_centroid_mm": 0.0 if wm is None else float(wm.get("distance_to_centroid_mm", 0.0)),
                    "thomas_native_ras": th_native_ras,
                    "freesurfer_native_ras": fs_native_ras,
                    "wm_native_ras": wm_native_ras,
                }
            )

        rows.sort(key=lambda r: (str(r.get("trajectory", "")), int(r.get("contact_index", 0))))
        return rows

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
        """Compatibility wrapper delegating to shared scene service."""
        return self.electrode_scene.align_slice_to_trajectory(
            start_ras=start_ras,
            end_ras=end_ras,
            slice_view=slice_view,
            mode=mode,
        )


def run(case_dir, reference=None, invert=False, harden=True, load_trajectories=True):
    """Headless convenience entrypoint for scripted smoke tests."""
    return RosaHelperLogic().load_case(
        case_dir=case_dir,
        reference=reference,
        invert=invert,
        harden=harden,
        load_trajectories=load_trajectories,
    )
