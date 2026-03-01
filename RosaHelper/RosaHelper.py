"""3D Slicer scripted module entrypoint for Loader.

This module is the case/volume loader front-end:
- load ROSA cases into shared workflow roles
- import custom MRI/CT volumes
- set default Base/Postop roles
- run optional rigid registration for custom imports
"""

import os
import sys
from __main__ import ctk, qt, slicer
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

from rosa_core import (
    lps_to_ras_point,
)
from rosa_scene.case_loader_service import CaseLoaderService
from rosa_scene.freesurfer_service import FreeSurferService
from rosa_scene.scene_utils import find_node_by_name
from rosa_scene.trajectory_scene import TrajectorySceneService
from rosa_workflow.workflow_publish import WorkflowPublisher
from rosa_workflow.workflow_state import WorkflowState


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
        self.loadedVolumeNodeIDs = {}
        self.loadedVolumeSourcePaths = {}
        self.referenceVolumeName = None
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
        self.loadedVolumeNodeIDs = summary.get("loaded_volume_node_ids", {})
        self.loadedVolumeSourcePaths = summary.get("loaded_volume_source_paths", {})
        self.referenceVolumeName = summary.get("reference_volume")
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
                node = find_node_by_name(name, "vtkMRMLMarkupsLineNode")
            if node is not None:
                imported_nodes.append(node)
            plan_node = self.logic.trajectory_scene.find_line_by_group_and_name(name, "planned_rosa")
            if plan_node is None:
                plan_node = find_node_by_name(f"Plan_{name}", "vtkMRMLMarkupsLineNode")
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
    """Loader-only logic used by UI and headless `run()` entrypoint."""

    def __init__(self):
        """Initialize loader and workflow services."""
        super().__init__()
        self.case_loader = CaseLoaderService()
        self.workflow_state = WorkflowState()
        self.workflow_publish = WorkflowPublisher(self.workflow_state)
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
        """Load one ROSA case into the current scene via CaseLoaderService."""
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
        """Apply CT window/level preset centered around the current threshold."""
        self.case_loader.apply_ct_window_from_threshold(volume_node, threshold)

    def reset_ct_window(self, volume_node):
        """Reset CT window/level to Slicer auto mode."""
        self.case_loader.reset_ct_window(volume_node)

    def run_brainsfit_rigid_registration(
        self,
        fixed_volume_node,
        moving_volume_node,
        output_transform_node,
        initialize_mode="useGeometryAlign",
        logger=None,
    ):
        """Run rigid BRAINSFit registration through shared FreeSurfer service."""
        return self.fs_service.run_brainsfit_rigid_registration(
            fixed_volume_node=fixed_volume_node,
            moving_volume_node=moving_volume_node,
            output_transform_node=output_transform_node,
            initialize_mode=initialize_mode,
            logger=logger,
        )

    def _add_trajectories(self, trajectories, logger=None, show_planned=False):
        """Create editable imported trajectories and optional planned backups."""
        for traj in trajectories:
            start_ras = lps_to_ras_point(traj["start"])
            end_ras = lps_to_ras_point(traj["end"])
            self.trajectory_scene.create_or_update_trajectory_line(
                name=traj["name"],
                start_ras=start_ras,
                end_ras=end_ras,
                node_name=traj["name"],
                group="imported_rosa",
                origin="rosa",
            )
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


def run(case_dir, reference=None, invert=False, harden=True, load_trajectories=True):
    """Headless convenience entrypoint for scripted smoke tests."""
    return RosaHelperLogic().load_case(
        case_dir=case_dir,
        reference=reference,
        invert=invert,
        harden=harden,
        load_trajectories=load_trajectories,
    )
