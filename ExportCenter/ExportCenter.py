"""3D Slicer module for workflow-based export bundle generation."""

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

from rosa_workflow import WorkflowState
from rosa_workflow.export_bundle import collect_export_inputs_from_workflow, export_aligned_bundle
from rosa_workflow.export_profiles import get_export_profile, profile_names


class ExportCenter(ScriptedLoadableModule):
    """Slicer metadata for workflow-driven export module."""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "Export Center"
        self.parent.categories = ["ROSA"]
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


class ExportCenterLogic(ScriptedLoadableModuleLogic):
    """Thin logic wrapper; export behavior lives in shared workflow service."""

    pass
