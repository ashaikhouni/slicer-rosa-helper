"""Atlas labeling module consuming shared workflow contacts and atlas sources."""

import os
import sys

from __main__ import qt, slicer
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

from rosa_core import lps_to_ras_point
from rosa_workflow import WorkflowState
from rosa_scene import AtlasAssignmentService, AtlasUtils


class AtlasLabeling(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "Atlas Labeling"
        self.parent.categories = ["ROSA"]
        self.parent.dependencies = []
        self.parent.contributors = ["Ammar Shaikhouni", "Codex"]
        self.parent.helpText = "Assign generated contacts to FreeSurfer/THOMAS/WM atlases."


class AtlasLabelingWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        super().setup()
        self.logic = AtlasLabelingLogic()
        self.workflowState = self.logic.workflow_state
        self.workflowNode = self.workflowState.resolve_or_create_workflow_node()
        self.lastRows = []

        form = qt.QFormLayout()
        self.layout.addLayout(form)
        self.refreshButton = qt.QPushButton("Refresh Atlas Sources")
        self.refreshButton.clicked.connect(self.onRefreshClicked)
        form.addRow(self.refreshButton)

        self.atlasFSVolumeCombo = qt.QComboBox()
        self.atlasFSVolumeCombo.addItem("(none)")
        form.addRow("FreeSurfer atlas", self.atlasFSVolumeCombo)

        self.atlasThomasCombo = qt.QComboBox()
        self.atlasThomasCombo.addItem("(none)")
        form.addRow("Thalamus atlas", self.atlasThomasCombo)

        self.atlasWMVolumeCombo = qt.QComboBox()
        self.atlasWMVolumeCombo.addItem("(none)")
        form.addRow("White matter atlas", self.atlasWMVolumeCombo)

        self.referenceSelector = slicer.qMRMLNodeComboBox()
        self.referenceSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.referenceSelector.noneEnabled = True
        self.referenceSelector.addEnabled = False
        self.referenceSelector.removeEnabled = False
        self.referenceSelector.setMRMLScene(slicer.mrmlScene)
        form.addRow("Reference volume", self.referenceSelector)

        self.assignButton = qt.QPushButton("Assign Contacts to Atlas")
        self.assignButton.clicked.connect(self.onAssignClicked)
        form.addRow(self.assignButton)

        self.statusText = qt.QPlainTextEdit()
        self.statusText.setReadOnly(True)
        self.statusText.setMaximumBlockCount(1000)
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

    def _refresh_combo_from_nodes(self, combo, nodes, all_token=False):
        current = combo.currentData if hasattr(combo, "currentData") else None
        current_val = current() if callable(current) else current
        combo.clear()
        combo.addItem("(none)", "")
        if all_token and nodes:
            combo.addItem("(all)", "__ALL__")
        for node in nodes:
            combo.addItem(node.GetName(), node.GetID())
        idx = combo.findData(current_val)
        combo.setCurrentIndex(idx if idx >= 0 else 0)

    def onRefreshClicked(self):
        self.workflowNode = self.workflowState.resolve_or_create_workflow_node()
        fs_nodes = self.workflowState.role_nodes("FSParcellationVolumes", workflow_node=self.workflowNode)
        wm_nodes = self.workflowState.role_nodes("WMParcellationVolumes", workflow_node=self.workflowNode)
        th_nodes = self.workflowState.role_nodes("THOMASSegmentations", workflow_node=self.workflowNode)
        self._refresh_combo_from_nodes(self.atlasFSVolumeCombo, fs_nodes, all_token=False)
        self._refresh_combo_from_nodes(self.atlasWMVolumeCombo, wm_nodes, all_token=False)
        self._refresh_combo_from_nodes(self.atlasThomasCombo, th_nodes, all_token=True)
        base = self.workflowNode.GetNodeReference("BaseVolume")
        if base is not None:
            self.referenceSelector.setCurrentNode(base)
        self.log(
            f"[refresh] atlas sources: fs={len(fs_nodes)} wm={len(wm_nodes)} thomas={len(th_nodes)}"
        )

    def _selected_volume(self, combo):
        data = combo.currentData if hasattr(combo, "currentData") else None
        node_id = data() if callable(data) else data
        if not node_id:
            return None
        return slicer.mrmlScene.GetNodeByID(str(node_id))

    def _selected_thomas_nodes(self):
        data = self.atlasThomasCombo.currentData if hasattr(self.atlasThomasCombo, "currentData") else None
        node_id = data() if callable(data) else data
        if not node_id:
            return []
        if str(node_id) == "__ALL__":
            return self.workflowState.role_nodes("THOMASSegmentations", workflow_node=self.workflowNode)
        node = slicer.mrmlScene.GetNodeByID(str(node_id))
        return [node] if node is not None else []

    def _recover_contacts_from_workflow_nodes(self):
        contact_nodes = self.workflowState.role_nodes("ContactFiducials", workflow_node=self.workflowNode)
        recovered = []
        for node in contact_nodes:
            traj_name = (node.GetAttribute("Rosa.TrajectoryName") or "").strip()
            if not traj_name:
                node_name = node.GetName() or ""
                if "_" in node_name:
                    traj_name = node_name.rsplit("_", 1)[-1]
                else:
                    traj_name = node_name or "UNK"
            for i in range(node.GetNumberOfControlPoints()):
                p_ras = [0.0, 0.0, 0.0]
                node.GetNthControlPointPositionWorld(i, p_ras)
                p_lps = lps_to_ras_point(p_ras)
                label = node.GetNthControlPointLabel(i) or f"{traj_name}{i + 1}"
                recovered.append(
                    {
                        "trajectory": traj_name,
                        "index": i + 1,
                        "label": label,
                        "position_lps": p_lps,
                    }
                )
        recovered.sort(key=lambda c: (str(c.get("trajectory", "")), int(c.get("index", 0))))
        return recovered

    def onAssignClicked(self):
        contacts = self._recover_contacts_from_workflow_nodes()
        if not contacts:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Atlas Labeling", "No workflow contacts found.")
            return
        fs_node = self._selected_volume(self.atlasFSVolumeCombo)
        wm_node = self._selected_volume(self.atlasWMVolumeCombo)
        th_nodes = self._selected_thomas_nodes()
        if fs_node is None and wm_node is None and not th_nodes:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Atlas Labeling", "Select at least one atlas source.")
            return
        ref_node = self.referenceSelector.currentNode()
        if ref_node is None:
            ref_node = self.workflowNode.GetNodeReference("BaseVolume")
        try:
            rows = self.logic.assignment_service.assign_contacts_to_atlases(
                contacts=contacts,
                freesurfer_volume_node=fs_node,
                thomas_segmentation_nodes=th_nodes,
                wm_volume_node=wm_node,
                reference_volume_node=ref_node,
            )
            self.logic.assignment_service.publish_atlas_assignment_rows(
                atlas_rows=rows,
                workflow_node=self.workflowNode,
            )
        except Exception as exc:
            self.log(f"[atlas] assignment error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Atlas Labeling", str(exc))
            return
        self.lastRows = rows
        self.log(f"[atlas] assigned {len(rows)} contacts")


class AtlasLabelingLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        super().__init__()
        self.workflow_state = WorkflowState()
        self.utils = AtlasUtils()
        self.assignment_service = AtlasAssignmentService(utils=self.utils, workflow_state=self.workflow_state)
