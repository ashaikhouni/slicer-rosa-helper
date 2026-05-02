"""3D Slicer module for workflow-based export bundle generation.

Last updated: 2026-03-01
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
PATH_CANDIDATES = [
    os.path.join(os.path.dirname(MODULE_DIR), "CommonLib"),  # source tree shared libs
    os.path.join(MODULE_DIR, "CommonLib"),  # packaged extension shared libs
]
for path in PATH_CANDIDATES:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

import numpy as np

from rosa_workflow import WorkflowState
from rosa_workflow.export_bundle import (
    _world_to_node_ras_matrix,  # internal helper, reused so Curry uses
    collect_contacts_from_workflow,  # the same frame-transform semantics
    collect_export_inputs_from_workflow,  # as the bundle export.
    export_aligned_bundle,
)
from rosa_workflow.export_profiles import get_export_profile, profile_names

# Curry .pom export: writer + helpers live in rosa_core, which is
# pure ASCII / no Slicer deps so it's exercised by the headless test
# suite too. Bound to a button in this module.
from rosa_core.curry_export import (
    contacts_to_pom_points,
    trajectory_endpoints_to_pom_points,
    write_curry_pom,
)


# Trajectory roles by source key. Mirrors the source picker in CTV /
# PostopCT so the same vocabulary works everywhere.
_CURRY_TRAJECTORY_ROLE_BY_SOURCE = {
    "auto_fit":          "AutoFitTrajectoryLines",
    "guided_fit":        "GuidedFitTrajectoryLines",
    "manual":            "ManualTrajectoryLines",
    "imported_rosa":     "ImportedTrajectoryLines",
    "imported_external": "ImportedExternalTrajectoryLines",
    "planned_rosa":      "PlannedTrajectoryLines",
    "working":           "WorkingTrajectoryLines",
}


def _collect_trajectories_from_role(workflow_node, role):
    """Read trajectory line nodes for a given workflow role and convert
    each to a dict with `name` / `start_ras` / `end_ras`."""
    state = WorkflowState()
    nodes = state.role_nodes(role, workflow_node=workflow_node) or []
    out = []
    for node in nodes:
        if node is None or not node.IsA("vtkMRMLMarkupsLineNode"):
            continue
        if int(node.GetNumberOfControlPoints()) < 2:
            continue
        name = (node.GetAttribute("Rosa.TrajectoryName") or "").strip()
        if not name:
            name = (node.GetName() or "").strip()
        if not name:
            continue
        p0 = [0.0, 0.0, 0.0]
        p1 = [0.0, 0.0, 0.0]
        node.GetNthControlPointPositionWorld(0, p0)
        node.GetNthControlPointPositionWorld(1, p1)
        out.append({
            "name": name,
            "start_ras": [float(p0[0]), float(p0[1]), float(p0[2])],
            "end_ras":   [float(p1[0]), float(p1[1]), float(p1[2])],
        })
    out.sort(key=lambda t: t.get("name", ""))
    return out


class ExportCenter(ScriptedLoadableModule):
    """Slicer metadata for workflow-driven export module."""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "01 Export Center"
        self.parent.categories = ["ROSA.04 Export"]
        self.parent.dependencies = []
        self.parent.contributors = ["Ammar Shaikhouni", "Codex"]
        self.parent.helpText = (
            "Export workflow artifacts (contacts, trajectories, QC, atlas, volumes) "
            "from shared RosaWorkflow MRML state."
        )


class ExportCenterWidget(ScriptedLoadableModuleWidget):
    """UI for exporting bundle profiles from shared workflow roles."""

    def setup(self):
        super().setup()
        self.logic = ExportCenterLogic()
        self.workflowState = WorkflowState()
        self.workflowNode = self.workflowState.resolve_or_create_workflow_node()

        form = qt.QFormLayout()
        self.layout.addLayout(form)

        self.outputDirEdit = ctk.ctkPathLineEdit()
        self.outputDirEdit.filters = ctk.ctkPathLineEdit.Dirs
        self.outputDirEdit.setToolTip("Destination directory for exported files.")
        self.outputDirEdit.currentPath = ""
        form.addRow("Output directory", self.outputDirEdit)

        self.prefixEdit = qt.QLineEdit(
            self.workflowNode.GetParameter("DefaultExportPrefix") or "ROSA_Contacts"
        )
        self.prefixEdit.setToolTip("Filename prefix used for exported artifact files.")
        form.addRow("Filename prefix", self.prefixEdit)

        self.profileCombo = qt.QComboBox()
        for name in profile_names():
            self.profileCombo.addItem(name)
        default_idx = self.profileCombo.findText("full_bundle")
        if default_idx >= 0:
            self.profileCombo.setCurrentIndex(default_idx)
        form.addRow("Export profile", self.profileCombo)

        self.frameSelector = slicer.qMRMLNodeComboBox()
        self.frameSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.frameSelector.noneEnabled = True
        self.frameSelector.addEnabled = False
        self.frameSelector.removeEnabled = False
        self.frameSelector.setMRMLScene(slicer.mrmlScene)
        self.frameSelector.setToolTip(
            "Coordinate frame for primary exported XYZ columns. Defaults to workflow base volume."
        )
        form.addRow("Export coordinate frame", self.frameSelector)

        self.summaryLabel = qt.QLabel("Workflow inputs not scanned yet.")
        self.summaryLabel.wordWrap = True
        form.addRow("Workflow summary", self.summaryLabel)

        row = qt.QHBoxLayout()
        self.refreshButton = qt.QPushButton("Refresh Workflow Inputs")
        self.refreshButton.clicked.connect(self.onRefreshClicked)
        row.addWidget(self.refreshButton)
        self.exportButton = qt.QPushButton("Export Bundle")
        self.exportButton.clicked.connect(self.onExportClicked)
        row.addWidget(self.exportButton)
        form.addRow(row)

        # ---- Curry .pom export -----------------------------------
        # Workflow: load Curry's base MRI in Slicer, register it to
        # the ROSA base T1 with the registration kept as an
        # unhardened parent transform, then pick that volume here as
        # the Curry MRI reference. The export inverts its parent
        # transform so the .pom is in CurryT1's scanner-native LPS —
        # the same frame Curry will use to display the volume.
        curry_section = ctk.ctkCollapsibleButton()
        curry_section.text = "Curry .pom export"
        curry_section.collapsed = False
        self.layout.addWidget(curry_section)
        curry_form = qt.QFormLayout(curry_section)

        curry_help = qt.QLabel(
            "1. Load the Curry-side MRI (use Load… or pick an "
            "already-loaded volume). 2. Click <b>Register</b> — runs "
            "rigid BRAINSFit in-place against the ROSA base T1 and "
            "applies the transform as a parent (no hardening). "
            "3. Click <b>Export</b>."
        )
        curry_help.wordWrap = True
        curry_help.setSizePolicy(
            qt.QSizePolicy.Preferred, qt.QSizePolicy.MinimumExpanding,
        )
        curry_form.addRow(curry_help)

        # Curry MRI reference: pick an already-loaded volume, or use
        # the Load button to bring one in from disk in one click.
        ref_row = qt.QHBoxLayout()
        self.curryReferenceSelector = slicer.qMRMLNodeComboBox()
        self.curryReferenceSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.curryReferenceSelector.noneEnabled = True
        self.curryReferenceSelector.addEnabled = False
        self.curryReferenceSelector.removeEnabled = False
        self.curryReferenceSelector.renameEnabled = False
        self.curryReferenceSelector.setMRMLScene(slicer.mrmlScene)
        self.curryReferenceSelector.setToolTip(
            "Curry's reference MRI — the volume Curry will display. "
            "Output coordinates are in this volume's scanner-native LPS frame."
        )
        ref_row.addWidget(self.curryReferenceSelector, 1)
        self.curryLoadVolumeButton = qt.QPushButton("Load…")
        self.curryLoadVolumeButton.setToolTip(
            "Open a file dialog to load the Curry MRI volume "
            "(NIfTI / NRRD / DICOM). The newly loaded volume is "
            "auto-selected as the Curry MRI reference."
        )
        self.curryLoadVolumeButton.clicked.connect(self.onCurryLoadVolumeClicked)
        ref_row.addWidget(self.curryLoadVolumeButton)
        curry_form.addRow("Curry MRI reference", ref_row)

        self.curryRegisterButton = qt.QPushButton("Register Curry MRI → ROSA base")
        self.curryRegisterButton.setToolTip(
            "Open Slicer's General Registration (BRAINS) module with "
            "the ROSA base T1 as fixed image and the Curry MRI as "
            "moving. Run the registration there, then return here. "
            "Important: keep the resulting transform applied to the "
            "Curry MRI — do NOT harden it."
        )
        self.curryRegisterButton.clicked.connect(self.onCurryRegisterClicked)
        curry_form.addRow(self.curryRegisterButton)

        self.currySourceCombo = qt.QComboBox()
        for label, key in (
            ("Contacts (per fiducial)",       "contacts"),
            ("Auto Fit endpoints",            "auto_fit"),
            ("Guided Fit endpoints",          "guided_fit"),
            ("Manual Fit endpoints",          "manual"),
            ("Imported ROSA endpoints",       "imported_rosa"),
            ("Imported External endpoints",   "imported_external"),
            ("Planned ROSA endpoints",        "planned_rosa"),
            ("Working endpoints",             "working"),
        ):
            self.currySourceCombo.addItem(label, key)
        self.currySourceCombo.setCurrentIndex(0)
        self.currySourceCombo.setToolTip(
            "What to export: per-fiducial contact positions, or two "
            "endpoints (entry / tip) per trajectory from any source."
        )
        curry_form.addRow("Source", self.currySourceCombo)

        self.curryExportButton = qt.QPushButton("Export → Curry .pom")
        self.curryExportButton.setToolTip(
            "Write the selected source's positions to a Curry "
            "placement (.pom) file in LPS mm coordinates, in the "
            "Curry MRI reference frame."
        )
        self.curryExportButton.clicked.connect(self.onExportCurryClicked)
        curry_form.addRow(self.curryExportButton)
        # ---- end Curry export -----------------------------------

        self.statusText = qt.QPlainTextEdit()
        self.statusText.setReadOnly(True)
        self.statusText.setMaximumBlockCount(1500)
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

    def _profile_requirements_ok(self, profile, data):
        need_contacts = bool(
            profile.get("include_contacts", False)
            or profile.get("include_qc", False)
            or profile.get("include_atlas", False)
        )
        if need_contacts and not data.get("contacts"):
            return False, "Profile requires contacts but none are available in workflow."
        need_traj = bool(profile.get("include_planned", False) or profile.get("include_final", False))
        if need_traj and not (data.get("planned_trajectories") or data.get("final_trajectories")):
            return False, "Profile requires trajectories but none are available in workflow."
        if profile.get("include_volumes", False) and not data.get("volume_node_ids"):
            return False, "Profile requires volumes but image registry is empty."
        return True, ""

    def onRefreshClicked(self):
        try:
            self.workflowNode = self.workflowState.resolve_or_create_workflow_node()
            data = collect_export_inputs_from_workflow(
                workflow_node=self.workflowNode,
                output_frame_node=self.frameSelector.currentNode(),
            )
        except Exception as exc:
            self.summaryLabel.setText(f"Failed to read workflow inputs: {exc}")
            self.exportButton.setEnabled(False)
            self.log(f"[export] refresh failed: {exc}")
            return

        volumes = len(data.get("volume_node_ids", {}))
        contacts = len(data.get("contacts", []))
        planned = len(data.get("planned_trajectories", {}))
        final = len(data.get("final_trajectories", {}))
        qc = len(data.get("qc_rows", []))
        atlas = len(data.get("atlas_rows", []))
        summary = (
            f"volumes={volumes}, contacts={contacts}, planned={planned}, "
            f"final={final}, qc={qc}, atlas={atlas}"
        )
        self.summaryLabel.setText(summary)
        profile_name = self.profileCombo.currentText.strip() or "full_bundle"
        profile, _resolved = get_export_profile(profile_name)
        ok, reason = self._profile_requirements_ok(profile, data)
        self.exportButton.setEnabled(ok)
        if not ok:
            self.log(f"[export] disabled: {reason}")

    def onExportClicked(self):
        output_dir = self.outputDirEdit.currentPath.strip()
        if not output_dir:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Export Center", "Select an output directory.")
            return

        self.workflowNode = self.workflowState.resolve_or_create_workflow_node()
        profile_name = self.profileCombo.currentText.strip() or "full_bundle"
        profile, resolved_profile = get_export_profile(profile_name)
        data = collect_export_inputs_from_workflow(
            workflow_node=self.workflowNode,
            output_frame_node=self.frameSelector.currentNode(),
        )
        ok, reason = self._profile_requirements_ok(profile, data)
        if not ok:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Export Center", reason)
            return

        prefix = self.prefixEdit.text.strip() or (
            self.workflowNode.GetParameter("DefaultExportPrefix") or "ROSA_Contacts"
        )
        try:
            result = export_aligned_bundle(
                volume_node_ids=data.get("volume_node_ids", {}),
                contacts=data.get("contacts", []),
                out_dir=output_dir,
                node_prefix=prefix,
                planned_trajectories=data.get("planned_trajectories", {}),
                final_trajectories=data.get("final_trajectories", {}),
                qc_rows=data.get("qc_rows", []),
                atlas_rows=data.get("atlas_rows", []),
                output_frame_node=data.get("output_frame_node", None),
                export_profile=resolved_profile,
            )
        except Exception as exc:
            self.log(f"[export] failed: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Export Center", str(exc))
            return

        self.log(
            f"[export] profile={resolved_profile} wrote {result.get('volume_count', 0)} volume(s) "
            f"to {result.get('out_dir', output_dir)}"
        )
        self.log(f"[export] manifest: {result.get('manifest_path', '')}")
        self.onRefreshClicked()

    def onCurryLoadVolumeClicked(self):
        """Open a file dialog and load the chosen volume; auto-select
        it as the Curry MRI reference."""
        path = qt.QFileDialog.getOpenFileName(
            slicer.util.mainWindow(),
            "Load Curry MRI volume",
            str(qt.QDir.homePath()),
            "Volumes (*.nii *.nii.gz *.nrrd *.mha *.mhd *.img *.hdr);;All files (*)",
        )
        if not path:
            return
        try:
            node = slicer.util.loadVolume(path)
        except Exception as exc:
            qt.QMessageBox.critical(
                slicer.util.mainWindow(), "Curry export",
                f"Failed to load volume: {exc}",
            )
            return
        if node is None:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(), "Curry export",
                "Volume loaded but no node returned — pick it from "
                "the combo manually.",
            )
            return
        try:
            self.curryReferenceSelector.setCurrentNode(node)
        except Exception:
            pass
        self.log(f"[curry] loaded Curry MRI: {path} → node '{node.GetName()}'")

    def onCurryRegisterClicked(self):
        """Run rigid BRAINSFit registration of the Curry MRI to the
        ROSA base T1 in-place. Result is a transform node *applied*
        as the Curry MRI's parent (not hardened) — exactly what the
        Curry export needs to map points to scanner-native LPS.
        """
        self.workflowNode = self.workflowState.resolve_or_create_workflow_node()
        ros_base = self.workflowNode.GetNodeReference("BaseVolume")
        curry_ref = self.curryReferenceSelector.currentNode()
        if ros_base is None:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(), "Curry export",
                "ROSA base T1 not registered in workflow yet — load "
                "via RosaHelper first.",
            )
            return
        if curry_ref is None:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(), "Curry export",
                "Pick or load the Curry MRI volume first (use Load…).",
            )
            return
        if curry_ref.GetID() == ros_base.GetID():
            qt.QMessageBox.warning(
                slicer.util.mainWindow(), "Curry export",
                "Curry MRI and ROSA base T1 are the same volume.",
            )
            return
        # Same BRAINSFit path RosaHelper uses for custom-volume
        # registration, but we leave the transform APPLIED (not
        # hardened) so the export can invert it on the way out.
        from rosa_scene.registration_service import RegistrationService
        reg = RegistrationService()
        transform_name = f"{curry_ref.GetName()}_to_{ros_base.GetName()}_curry"
        # Re-use any existing transform with this name (idempotent
        # re-runs); else create a new one.
        transform_node = slicer.mrmlScene.GetFirstNodeByName(transform_name)
        if transform_node is None or not transform_node.IsA("vtkMRMLLinearTransformNode"):
            transform_node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLLinearTransformNode", transform_name,
            )
        try:
            reg.run_brainsfit_rigid_registration(
                fixed_volume_node=ros_base,
                moving_volume_node=curry_ref,
                output_transform_node=transform_node,
                initialize_mode="useGeometryAlign",
                logger=self.log,
            )
        except Exception as exc:
            qt.QMessageBox.critical(
                slicer.util.mainWindow(), "Curry export",
                f"Registration failed: {exc}",
            )
            return
        # Apply (not harden) so the Curry export can invert the
        # registration on its way out and produce coordinates in
        # the Curry MRI's scanner-native frame.
        curry_ref.SetAndObserveTransformNodeID(transform_node.GetID())
        # Tag the transform so it's discoverable later.
        try:
            transform_node.SetAttribute("Rosa.Managed", "1")
            transform_node.SetAttribute("Rosa.Source", "curry_export_registration")
            transform_node.SetAttribute(
                "Rosa.RegisteredVolumeID", curry_ref.GetID(),
            )
        except Exception:
            pass
        self.log(
            f"[curry] registered {curry_ref.GetName()} → "
            f"{ros_base.GetName()} (transform '{transform_name}' "
            f"applied as parent — DO NOT harden)"
        )
        qt.QMessageBox.information(
            slicer.util.mainWindow(), "Curry export",
            f"Registered {curry_ref.GetName()} → {ros_base.GetName()}.\n\n"
            "The transform is APPLIED (not hardened) so the export can "
            "map points back into the Curry MRI's native frame. Click "
            "'Export → Curry .pom' to write the file.",
        )

    def onExportCurryClicked(self):
        """Write the selected source's positions to a Curry .pom file."""
        source_key = ""
        try:
            data = self.currySourceCombo.currentData
            source_key = str(data() if callable(data) else data or "").strip()
        except Exception:
            source_key = ""
        if not source_key:
            return
        self.workflowNode = self.workflowState.resolve_or_create_workflow_node()
        # Validate the Curry MRI reference selection.
        curry_ref = self.curryReferenceSelector.currentNode()
        if curry_ref is None:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(), "Curry export",
                "Pick a Curry MRI reference volume first. Load Curry's "
                "MRI in Slicer, register it to the ROSA base T1 (don't "
                "harden the registration), then select it in the "
                "'Curry MRI reference' combo.",
            )
            return
        # Sanity check: the chosen volume should have a parent transform
        # — that's how the export gets back to the volume's native frame.
        # If it's None the user likely hardened the registration; the
        # export would silently land in world / ROSA frame instead of
        # CurryT1's native frame.
        if curry_ref.GetParentTransformNode() is None:
            ans = qt.QMessageBox.question(
                slicer.util.mainWindow(), "Curry export",
                f"'{curry_ref.GetName()}' has no parent transform.\n\n"
                "If you registered this volume to the ROSA base and "
                "hardened the registration, the export will be in the "
                "ROSA / world frame, NOT the Curry MRI's scanner-"
                "native frame.\n\nProceed anyway?",
                qt.QMessageBox.Yes | qt.QMessageBox.No,
                qt.QMessageBox.No,
            )
            if ans != qt.QMessageBox.Yes:
                return
        try:
            points = self._collect_curry_points(source_key, curry_ref)
        except Exception as exc:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(), "Curry export",
                f"Could not gather points: {exc}",
            )
            return
        if not points:
            qt.QMessageBox.information(
                slicer.util.mainWindow(), "Curry export",
                f"No points to export for source '{source_key}'.",
            )
            return
        prefix = (self.prefixEdit.text or "").strip() or "ROSA_Contacts"
        out_dir = (self.outputDirEdit.currentPath or "").strip() or str(qt.QDir.homePath())
        default_path = f"{out_dir}/{prefix}_{source_key}.pom"
        path = qt.QFileDialog.getSaveFileName(
            slicer.util.mainWindow(),
            "Export to Curry .pom",
            default_path,
            "Curry placement files (*.pom);;All files (*)",
        )
        if not path:
            return
        try:
            n = write_curry_pom(path, points, coords_in="ras")
        except Exception as exc:
            qt.QMessageBox.critical(
                slicer.util.mainWindow(), "Curry export",
                f"Write failed: {exc}",
            )
            return
        frame_name = curry_ref.GetName() or "world RAS"
        self.log(
            f"[curry] wrote {n} points (source={source_key}, "
            f"frame={frame_name}, output=LPS mm) to {path}"
        )
        qt.QMessageBox.information(
            slicer.util.mainWindow(), "Curry export",
            f"Wrote {n} points to:\n{path}\n\n"
            f"Source: {source_key}\n"
            f"Frame: {frame_name}\n"
            f"Coordinates: LPS mm",
        )

    def _collect_curry_points(self, source_key, curry_ref):
        """Return a list of (label, x, y, z) tuples in CURRY MRI's
        native RAS for export.

        Markups in Slicer live in WORLD RAS. To get coordinates in
        Curry MRI's native frame (the frame Curry will display the
        volume in), apply `_world_to_node_ras_matrix(curry_ref) @ p`
        which inverts the registration parent transform.
        """
        world_to_frame = _world_to_node_ras_matrix(curry_ref)

        def _xform(x, y, z):
            v = np.array([float(x), float(y), float(z), 1.0])
            out = world_to_frame @ v
            return float(out[0]), float(out[1]), float(out[2])

        if source_key == "contacts":
            contacts = collect_contacts_from_workflow(self.workflowNode)
            raw = contacts_to_pom_points(contacts)
        else:
            role = _CURRY_TRAJECTORY_ROLE_BY_SOURCE.get(source_key)
            if not role:
                raise ValueError(f"unknown source: {source_key}")
            trajectories = _collect_trajectories_from_role(
                self.workflowNode, role,
            )
            raw = trajectory_endpoints_to_pom_points(trajectories)
        return [(label, *_xform(x, y, z)) for label, x, y, z in raw]


class ExportCenterLogic(ScriptedLoadableModuleLogic):
    """Thin logic wrapper; export behavior lives in shared workflow service."""

    pass
