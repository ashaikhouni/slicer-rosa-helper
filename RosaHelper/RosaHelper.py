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
    find_ros_file,
    invert_4x4,
    lps_to_ras_matrix,
    lps_to_ras_point,
    parse_ros_file,
    resolve_analyze_volume,
    resolve_reference_index,
)
from rosa_slicer.freesurfer_service import FreeSurferService
from rosa_slicer.trajectory_scene import TrajectorySceneService
from rosa_slicer.widget_mixin import RosaHelperWidgetMixin


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


class RosaHelperWidget(RosaHelperWidgetMixin, ScriptedLoadableModuleWidget):
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
