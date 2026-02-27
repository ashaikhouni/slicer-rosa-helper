"""3D Slicer module for postop CT localization workflows.

Modes:
- Guided Fit: refine planned trajectories on postop CT.
- De Novo Detect: detect trajectories directly from CT artifact.
"""

import os
import sys

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
PATH_CANDIDATES = [
    os.path.join(os.path.dirname(MODULE_DIR), "CommonLib"),
    os.path.join(MODULE_DIR, "CommonLib"),
    os.path.join(os.path.dirname(MODULE_DIR), "ShankDetect", "Lib"),
    os.path.join(MODULE_DIR, "ShankDetect", "Lib"),
]
for path in PATH_CANDIDATES:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

from rosa_core import (
    fit_electrode_axis_and_tip,
    load_electrode_library,
    lps_to_ras_point,
    model_map,
    suggest_model_id_for_trajectory,
)
from rosa_scene import ElectrodeSceneService, TrajectorySceneService
from rosa_workflow import WorkflowPublisher, WorkflowState
from rosa_workflow.workflow_registry import table_to_dict_rows
from shank_core import pipeline as shank_pipeline


class PostopCTLocalization(ScriptedLoadableModule):
    """Slicer metadata for unified postop CT localization module."""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "Postop CT Localization"
        self.parent.categories = ["ROSA"]
        self.parent.dependencies = []
        self.parent.contributors = ["Ammar Shaikhouni", "Codex"]
        self.parent.helpText = (
            "Unified postop CT localization: Guided Fit (planned trajectories) and "
            "De Novo Detect (CT-only)."
        )


class PostopCTLocalizationWidget(ScriptedLoadableModuleWidget):
    """Widget exposing guided-fit and de-novo CT localization modes."""

    def setup(self):
        super().setup()
        self.logic = PostopCTLocalizationLogic()
        self.workflowState = self.logic.workflow_state
        self.workflowPublisher = self.logic.workflow_publish
        self.workflowNode = self.workflowState.resolve_or_create_workflow_node()

        self.modelsById = {}
        self.modelIds = []
        self.loadedTrajectories = []
        self.assignmentMap = {}
        self.candidatesLPS = np.empty((0, 3), dtype=float) if np is not None else []
        self.fitResults = {}

        form = qt.QFormLayout()
        self.layout.addLayout(form)

        self.ctSelector = slicer.qMRMLNodeComboBox()
        self.ctSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.ctSelector.noneEnabled = True
        self.ctSelector.addEnabled = False
        self.ctSelector.removeEnabled = False
        self.ctSelector.setMRMLScene(slicer.mrmlScene)
        self.ctSelector.setToolTip("Postop CT used for guided fit / de novo detection.")
        form.addRow("Postop CT", self.ctSelector)

        topRow = qt.QHBoxLayout()
        self.refreshButton = qt.QPushButton("Refresh Workflow Inputs")
        self.refreshButton.clicked.connect(self.onRefreshClicked)
        topRow.addWidget(self.refreshButton)
        self.resetViewsButton = qt.QPushButton("Reset Ax/Cor/Sag")
        self.resetViewsButton.clicked.connect(self.onResetViewsClicked)
        topRow.addWidget(self.resetViewsButton)
        topRow.addStretch(1)
        form.addRow(topRow)

        self.summaryLabel = qt.QLabel("Workflow not scanned yet.")
        self.summaryLabel.wordWrap = True
        form.addRow("Workflow summary", self.summaryLabel)

        self.modeTabs = qt.QTabWidget()
        self.layout.addWidget(self.modeTabs)
        self._build_guided_fit_tab()
        self._build_de_novo_tab()

        self.statusText = qt.QPlainTextEdit()
        self.statusText.setReadOnly(True)
        self.statusText.setMaximumBlockCount(2000)
        self.layout.addWidget(self.statusText)
        self.layout.addStretch(1)

        self._load_electrode_library()
        self.onRefreshClicked()

    def log(self, message):
        self.statusText.appendPlainText(str(message))
        self.statusText.ensureCursorVisible()
        try:
            slicer.app.processEvents()
        except Exception:
            pass

    def _build_guided_fit_tab(self):
        tab = qt.QWidget()
        self.modeTabs.addTab(tab, "Guided Fit")
        form = qt.QFormLayout(tab)

        self.guidedThresholdSpin = qt.QDoubleSpinBox()
        self.guidedThresholdSpin.setRange(300.0, 4000.0)
        self.guidedThresholdSpin.setDecimals(1)
        self.guidedThresholdSpin.setValue(1800.0)
        self.guidedThresholdSpin.setSuffix(" HU")
        form.addRow("Metal threshold", self.guidedThresholdSpin)

        detectRow = qt.QHBoxLayout()
        self.detectCandidatesButton = qt.QPushButton("Detect Candidates")
        self.detectCandidatesButton.clicked.connect(self.onDetectCandidatesClicked)
        detectRow.addWidget(self.detectCandidatesButton)
        self.resetWindowButton = qt.QPushButton("Reset CT Window")
        self.resetWindowButton.clicked.connect(self.onResetCTWindowClicked)
        detectRow.addWidget(self.resetWindowButton)
        detectRow.addStretch(1)
        form.addRow(detectRow)

        self.guidedTrajectorySelector = qt.QComboBox()
        form.addRow("Selected trajectory", self.guidedTrajectorySelector)

        self.guidedRoiRadiusSpin = qt.QDoubleSpinBox()
        self.guidedRoiRadiusSpin.setRange(1.0, 6.0)
        self.guidedRoiRadiusSpin.setDecimals(2)
        self.guidedRoiRadiusSpin.setValue(3.0)
        self.guidedRoiRadiusSpin.setSuffix(" mm")
        form.addRow("ROI radius", self.guidedRoiRadiusSpin)

        self.guidedMaxAngleSpin = qt.QDoubleSpinBox()
        self.guidedMaxAngleSpin.setRange(1.0, 25.0)
        self.guidedMaxAngleSpin.setDecimals(1)
        self.guidedMaxAngleSpin.setValue(12.0)
        self.guidedMaxAngleSpin.setSuffix(" deg")
        form.addRow("Max angle deviation", self.guidedMaxAngleSpin)

        self.guidedMaxDepthShiftSpin = qt.QDoubleSpinBox()
        self.guidedMaxDepthShiftSpin.setRange(1.0, 50.0)
        self.guidedMaxDepthShiftSpin.setDecimals(1)
        self.guidedMaxDepthShiftSpin.setValue(20.0)
        self.guidedMaxDepthShiftSpin.setSuffix(" mm")
        form.addRow("Max depth shift", self.guidedMaxDepthShiftSpin)

        fitRow = qt.QHBoxLayout()
        self.fitSelectedButton = qt.QPushButton("Fit Selected")
        self.fitSelectedButton.clicked.connect(self.onFitSelectedClicked)
        fitRow.addWidget(self.fitSelectedButton)
        self.fitAllButton = qt.QPushButton("Fit All")
        self.fitAllButton.clicked.connect(self.onFitAllClicked)
        fitRow.addWidget(self.fitAllButton)
        self.applyFitButton = qt.QPushButton("Apply Fit to Trajectories")
        self.applyFitButton.clicked.connect(self.onApplyFitClicked)
        self.applyFitButton.setEnabled(False)
        fitRow.addWidget(self.applyFitButton)
        form.addRow(fitRow)

    def _build_de_novo_tab(self):
        tab = qt.QWidget()
        self.modeTabs.addTab(tab, "De Novo Detect")
        form = qt.QFormLayout(tab)

        self.deNovoThresholdSpin = qt.QDoubleSpinBox()
        self.deNovoThresholdSpin.setRange(300.0, 4000.0)
        self.deNovoThresholdSpin.setDecimals(1)
        self.deNovoThresholdSpin.setValue(1800.0)
        self.deNovoThresholdSpin.setSuffix(" HU")
        form.addRow("Metal threshold", self.deNovoThresholdSpin)

        self.deNovoInlierRadiusSpin = qt.QDoubleSpinBox()
        self.deNovoInlierRadiusSpin.setRange(0.3, 6.0)
        self.deNovoInlierRadiusSpin.setDecimals(2)
        self.deNovoInlierRadiusSpin.setValue(1.2)
        self.deNovoInlierRadiusSpin.setSuffix(" mm")
        form.addRow("Inlier radius", self.deNovoInlierRadiusSpin)

        self.deNovoMinLengthSpin = qt.QDoubleSpinBox()
        self.deNovoMinLengthSpin.setRange(5.0, 200.0)
        self.deNovoMinLengthSpin.setDecimals(1)
        self.deNovoMinLengthSpin.setValue(20.0)
        self.deNovoMinLengthSpin.setSuffix(" mm")
        form.addRow("Min trajectory length", self.deNovoMinLengthSpin)

        self.deNovoMinInliersSpin = qt.QSpinBox()
        self.deNovoMinInliersSpin.setRange(25, 20000)
        self.deNovoMinInliersSpin.setValue(250)
        form.addRow("Min inliers", self.deNovoMinInliersSpin)

        self.deNovoMaxPointsSpin = qt.QSpinBox()
        self.deNovoMaxPointsSpin.setRange(5000, 1000000)
        self.deNovoMaxPointsSpin.setSingleStep(5000)
        self.deNovoMaxPointsSpin.setValue(300000)
        form.addRow("Max sampled points", self.deNovoMaxPointsSpin)

        self.deNovoRansacSpin = qt.QSpinBox()
        self.deNovoRansacSpin.setRange(20, 2000)
        self.deNovoRansacSpin.setValue(240)
        form.addRow("RANSAC iterations", self.deNovoRansacSpin)

        self.deNovoMaxLinesSpin = qt.QSpinBox()
        self.deNovoMaxLinesSpin.setRange(1, 50)
        self.deNovoMaxLinesSpin.setValue(30)
        form.addRow("Max lines", self.deNovoMaxLinesSpin)

        self.deNovoUseExactCountCheck = qt.QCheckBox("Use exact trajectory count")
        self.deNovoUseExactCountCheck.setChecked(False)
        self.deNovoUseExactCountCheck.toggled.connect(self._onUseExactToggled)
        form.addRow(self.deNovoUseExactCountCheck)

        self.deNovoExactCountSpin = qt.QSpinBox()
        self.deNovoExactCountSpin.setRange(1, 50)
        self.deNovoExactCountSpin.setValue(10)
        self.deNovoExactCountSpin.setEnabled(False)
        form.addRow("Exact count", self.deNovoExactCountSpin)

        self.deNovoUseHeadMaskCheck = qt.QCheckBox("Use head-depth gating")
        self.deNovoUseHeadMaskCheck.setChecked(True)
        form.addRow(self.deNovoUseHeadMaskCheck)

        self.deNovoHeadThresholdSpin = qt.QDoubleSpinBox()
        self.deNovoHeadThresholdSpin.setRange(-1200.0, 500.0)
        self.deNovoHeadThresholdSpin.setDecimals(1)
        self.deNovoHeadThresholdSpin.setValue(-500.0)
        self.deNovoHeadThresholdSpin.setSuffix(" HU")
        form.addRow("Head threshold", self.deNovoHeadThresholdSpin)

        self.deNovoMinDepthSpin = qt.QDoubleSpinBox()
        self.deNovoMinDepthSpin.setRange(0.0, 100.0)
        self.deNovoMinDepthSpin.setDecimals(2)
        self.deNovoMinDepthSpin.setValue(5.0)
        self.deNovoMinDepthSpin.setSuffix(" mm")
        form.addRow("Min metal depth", self.deNovoMinDepthSpin)

        self.deNovoMaxDepthSpin = qt.QDoubleSpinBox()
        self.deNovoMaxDepthSpin.setRange(0.0, 300.0)
        self.deNovoMaxDepthSpin.setDecimals(2)
        self.deNovoMaxDepthSpin.setValue(220.0)
        self.deNovoMaxDepthSpin.setSuffix(" mm")
        form.addRow("Max metal depth", self.deNovoMaxDepthSpin)

        self.deNovoReplaceCheck = qt.QCheckBox("Replace existing working trajectories")
        self.deNovoReplaceCheck.setChecked(True)
        form.addRow(self.deNovoReplaceCheck)

        self.deNovoDetectButton = qt.QPushButton("Detect Trajectories")
        self.deNovoDetectButton.clicked.connect(self.onDeNovoDetectClicked)
        form.addRow(self.deNovoDetectButton)

    def _onUseExactToggled(self, checked):
        self.deNovoExactCountSpin.setEnabled(bool(checked))

    def _load_electrode_library(self):
        try:
            data = load_electrode_library()
            self.modelsById = model_map(data)
            self.modelIds = sorted(self.modelsById.keys())
        except Exception as exc:
            self.modelsById = {}
            self.modelIds = []
            self.log(f"[electrodes] failed to load model library: {exc}")
            return
        self.log(f"[electrodes] loaded {len(self.modelIds)} models")

    def _assignment_map_from_workflow(self):
        amap = {}
        table = self.workflowNode.GetNodeReference("ElectrodeAssignmentTable")
        for row in table_to_dict_rows(table):
            traj = (row.get("trajectory") or "").strip()
            model_id = (row.get("model_id") or "").strip()
            if not traj or not model_id:
                continue
            amap[traj] = {
                "trajectory": traj,
                "model_id": model_id,
                "tip_at": (row.get("tip_at") or "target").strip() or "target",
                "tip_shift_mm": float(row.get("tip_shift_mm") or 0.0),
                "xyz_offset_mm": [0.0, 0.0, 0.0],
            }
        return amap

    def _populate_guided_trajectory_selector(self):
        self.guidedTrajectorySelector.clear()
        for traj in self.loadedTrajectories:
            self.guidedTrajectorySelector.addItem(traj["name"])

    def _refresh_summary(self):
        self.summaryLabel.setText(
            f"working trajectories={len(self.loadedTrajectories)}, "
            f"assignments={len(self.assignmentMap)}, "
            f"candidate points={(len(self.candidatesLPS) if np is None else int(getattr(self.candidatesLPS, 'shape', [0])[0]))}"
        )

    def onRefreshClicked(self):
        self.workflowNode = self.workflowState.resolve_or_create_workflow_node()
        self.loadedTrajectories = self.logic.collect_working_trajectories(workflow_node=self.workflowNode)
        self.assignmentMap = self._assignment_map_from_workflow()
        self._populate_guided_trajectory_selector()
        self._refresh_summary()
        self.log(
            f"[refresh] trajectories={len(self.loadedTrajectories)} assignments={len(self.assignmentMap)}"
        )

    def onResetViewsClicked(self):
        try:
            self.logic.reset_standard_slice_views()
            self.log("[slice] reset to Axial/Coronal/Sagittal")
        except Exception as exc:
            self.log(f"[slice] reset warning: {exc}")

    def onDetectCandidatesClicked(self):
        volume_node = self.ctSelector.currentNode()
        if volume_node is None:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Postop CT Localization", "Select a CT volume.")
            return
        threshold = float(self.guidedThresholdSpin.value)
        try:
            points_lps = self.logic.extract_threshold_candidates_lps(
                volume_node=volume_node,
                threshold=threshold,
            )
        except Exception as exc:
            self.log(f"[guided] detect error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Postop CT Localization", str(exc))
            return

        self.candidatesLPS = points_lps
        self.fitResults = {}
        self.applyFitButton.setEnabled(False)

        self.logic.register_postop_ct(volume_node, workflow_node=self.workflowNode)
        self.logic.show_volume_in_all_slice_views(volume_node)
        self.logic.apply_ct_window_from_threshold(volume_node, threshold=threshold)
        n = int(points_lps.shape[0]) if np is not None else len(points_lps)
        self.log(f"[guided] detected {n} candidate points at threshold {threshold:.1f}")
        self._refresh_summary()

    def onResetCTWindowClicked(self):
        volume_node = self.ctSelector.currentNode()
        if volume_node is None:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Postop CT Localization", "Select a CT volume.")
            return
        self.logic.reset_ct_window(volume_node)
        self.log("[guided] CT window reset")

    def _assignment_for_trajectory(self, traj):
        row = self.assignmentMap.get(traj["name"])
        if row and row.get("model_id") in self.modelsById:
            return row
        model_id = suggest_model_id_for_trajectory(
            trajectory=traj,
            models_by_id=self.modelsById,
            model_ids=self.modelIds,
            tolerance_mm=5.0,
        )
        if not model_id:
            return None
        return {
            "trajectory": traj["name"],
            "model_id": model_id,
            "tip_at": "target",
            "tip_shift_mm": 0.0,
            "xyz_offset_mm": [0.0, 0.0, 0.0],
        }

    def _fit_names(self, names):
        if np is not None and int(self.candidatesLPS.shape[0]) == 0:
            raise ValueError("No CT candidates. Run 'Detect Candidates' first.")

        traj_map = {traj["name"]: traj for traj in self.loadedTrajectories}
        success = 0
        for name in names:
            traj = traj_map.get(name)
            if traj is None:
                self.log(f"[guided] {name}: skipped (missing trajectory)")
                continue
            assignment = self._assignment_for_trajectory(traj)
            if assignment is None:
                self.log(f"[guided] {name}: skipped (no electrode model)")
                continue
            model = self.modelsById.get(assignment["model_id"])
            fit = fit_electrode_axis_and_tip(
                candidate_points_lps=self.candidatesLPS,
                planned_entry_lps=traj["start"],
                planned_target_lps=traj["end"],
                contact_offsets_mm=model["contact_center_offsets_from_tip_mm"],
                tip_at=assignment.get("tip_at", "target"),
                roi_radius_mm=float(self.guidedRoiRadiusSpin.value),
                max_angle_deg=float(self.guidedMaxAngleSpin.value),
                max_depth_shift_mm=float(self.guidedMaxDepthShiftSpin.value),
            )
            if fit.get("success"):
                self.fitResults[name] = fit
                self.logic.trajectory_scene.set_preview_line(
                    trajectory_name=name,
                    start_lps=fit["entry_lps"],
                    end_lps=fit["target_lps"],
                    node_prefix="AutoFit_",
                )
                success += 1
                self.log(
                    "[guided] {name}: angle={a:.2f} deg depth={s:.2f} mm lateral={l:.2f} mm residual={r:.2f} mm".format(
                        name=name,
                        a=float(fit.get("angle_deg", 0.0)),
                        s=float(fit.get("tip_shift_mm", 0.0)),
                        l=float(fit.get("lateral_shift_mm", 0.0)),
                        r=float(fit.get("residual_mm", 0.0)),
                    )
                )
            else:
                self.log(f"[guided] {name}: failed ({fit.get('reason', 'unknown')})")
        self.applyFitButton.setEnabled(bool(self.fitResults))
        self.log(f"[guided] fitted {success}/{len(names)} trajectories")

    def onFitSelectedClicked(self):
        name = self.guidedTrajectorySelector.currentText.strip()
        if not name:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Postop CT Localization", "Select a trajectory.")
            return
        try:
            self._fit_names([name])
        except Exception as exc:
            self.log(f"[guided] error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Postop CT Localization", str(exc))

    def onFitAllClicked(self):
        names = [traj["name"] for traj in self.loadedTrajectories]
        if not names:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Postop CT Localization", "No trajectories available.")
            return
        try:
            self._fit_names(names)
        except Exception as exc:
            self.log(f"[guided] error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Postop CT Localization", str(exc))

    def onApplyFitClicked(self):
        if not self.fitResults:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Postop CT Localization", "No fit results to apply.")
            return
        applied_nodes = []
        for name, fit in self.fitResults.items():
            node = self.logic.trajectory_scene.find_line_markup_node(name)
            if node is None or node.GetNumberOfControlPoints() < 2:
                continue
            start_ras = lps_to_ras_point(fit["entry_lps"])
            end_ras = lps_to_ras_point(fit["target_lps"])
            node.SetNthControlPointPositionWorld(0, start_ras[0], start_ras[1], start_ras[2])
            node.SetNthControlPointPositionWorld(1, end_ras[0], end_ras[1], end_ras[2])
            applied_nodes.append(node)

        self.logic.trajectory_scene.remove_preview_lines(node_prefix="AutoFit_")
        self.fitResults = {}
        self.applyFitButton.setEnabled(False)
        if applied_nodes:
            self.workflowPublisher.publish_nodes(
                role="WorkingTrajectoryLines",
                nodes=applied_nodes,
                source="postop_ct_guided_fit",
                space_name="ROSA_BASE",
                workflow_node=self.workflowNode,
            )
        self.log(f"[guided] applied fitted trajectories: {len(applied_nodes)}")
        self.onRefreshClicked()

    def onDeNovoDetectClicked(self):
        volume_node = self.ctSelector.currentNode()
        if volume_node is None:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Postop CT Localization", "Select a CT volume.")
            return

        max_lines = int(self.deNovoExactCountSpin.value) if self.deNovoUseExactCountCheck.checked else int(
            self.deNovoMaxLinesSpin.value
        )
        self.logic.register_postop_ct(volume_node, workflow_node=self.workflowNode)

        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        slicer.app.processEvents()
        try:
            result = self.logic.detect_de_novo(
                volume_node=volume_node,
                threshold=float(self.deNovoThresholdSpin.value),
                max_points=int(self.deNovoMaxPointsSpin.value),
                max_lines=max_lines,
                inlier_radius_mm=float(self.deNovoInlierRadiusSpin.value),
                min_length_mm=float(self.deNovoMinLengthSpin.value),
                min_inliers=int(self.deNovoMinInliersSpin.value),
                ransac_iterations=int(self.deNovoRansacSpin.value),
                use_head_mask=bool(self.deNovoUseHeadMaskCheck.checked),
                head_mask_threshold_hu=float(self.deNovoHeadThresholdSpin.value),
                min_metal_depth_mm=float(self.deNovoMinDepthSpin.value),
                max_metal_depth_mm=float(self.deNovoMaxDepthSpin.value),
            )
        except Exception as exc:
            self.log(f"[denovo] failed: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Postop CT Localization", str(exc))
            return
        finally:
            qt.QApplication.restoreOverrideCursor()
            slicer.app.processEvents()

        lines = result.get("lines", [])
        if not lines:
            self.log("[denovo] no trajectories detected")
            return

        rows = self.logic.upsert_detected_lines(
            lines=lines,
            replace_existing=bool(self.deNovoReplaceCheck.checked),
        )
        self.logic.publish_working_rows(rows, workflow_node=self.workflowNode)
        self.log(
            f"[denovo] candidates={int(result.get('candidate_count', 0))}, "
            f"in-mask={int(result.get('head_mask_kept_count', 0))}, "
            f"new_lines={len(lines)}, total_rows={len(rows)}"
        )
        self.onRefreshClicked()


class PostopCTLocalizationLogic(ScriptedLoadableModuleLogic):
    """Logic for guided-fit and de-novo postop CT localization."""

    def __init__(self):
        super().__init__()
        if np is None:
            raise RuntimeError("numpy is required for Postop CT Localization.")
        self.workflow_state = WorkflowState()
        self.workflow_publish = WorkflowPublisher(self.workflow_state)
        self.trajectory_scene = TrajectorySceneService()
        self.electrode_scene = ElectrodeSceneService(
            workflow_state=self.workflow_state,
            workflow_publish=self.workflow_publish,
        )

    def register_postop_ct(self, volume_node, workflow_node=None):
        self.workflow_publish.register_volume(
            volume_node=volume_node,
            source_type="import",
            source_path="",
            space_name="ROSA_BASE",
            role="AdditionalCTVolumes",
            is_default_postop=True,
            workflow_node=workflow_node,
        )

    def _vtk_matrix_to_numpy(self, vtk_matrix4x4):
        out = np.eye(4, dtype=float)
        for r in range(4):
            for c in range(4):
                out[r, c] = float(vtk_matrix4x4.GetElement(r, c))
        return out

    def extract_threshold_candidates_lps(self, volume_node, threshold, max_points=300000):
        arr = slicer.util.arrayFromVolume(volume_node)  # K,J,I
        idx = np.argwhere(arr >= float(threshold))
        if idx.size == 0:
            return np.empty((0, 3), dtype=float)
        if idx.shape[0] > int(max_points):
            rng = np.random.default_rng(0)
            keep = rng.choice(idx.shape[0], size=int(max_points), replace=False)
            idx = idx[keep]
        n = idx.shape[0]
        ijk_h = np.ones((n, 4), dtype=float)
        ijk_h[:, 0] = idx[:, 2]
        ijk_h[:, 1] = idx[:, 1]
        ijk_h[:, 2] = idx[:, 0]
        m_vtk = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(m_vtk)
        m = self._vtk_matrix_to_numpy(m_vtk)
        ras = (ijk_h @ m.T)[:, :3]
        lps = ras.copy()
        lps[:, 0] *= -1.0
        lps[:, 1] *= -1.0
        return lps

    def show_volume_in_all_slice_views(self, volume_node):
        volume_id = volume_node.GetID()
        for composite in slicer.util.getNodesByClass("vtkMRMLSliceCompositeNode"):
            composite.SetBackgroundVolumeID(volume_id)

    def apply_ct_window_from_threshold(self, volume_node, threshold):
        display = volume_node.GetDisplayNode()
        if display is None:
            return
        lower = float(threshold) - 250.0
        upper = float(threshold) + 2200.0
        display.AutoWindowLevelOff()
        display.SetWindow(max(upper - lower, 1.0))
        display.SetLevel((upper + lower) * 0.5)

    def reset_ct_window(self, volume_node):
        display = volume_node.GetDisplayNode()
        if display is not None:
            display.AutoWindowLevelOn()

    def reset_standard_slice_views(self):
        lm = slicer.app.layoutManager()
        if lm is None:
            return
        layout_node = lm.layoutLogic().GetLayoutNode()
        if layout_node is not None:
            layout_node.SetViewArrangement(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        for view_name, orientation in (("Red", "Axial"), ("Yellow", "Sagittal"), ("Green", "Coronal")):
            widget = lm.sliceWidget(view_name)
            if widget is None:
                continue
            slice_node = widget.mrmlSliceNode()
            if slice_node:
                slice_node.SetOrientation(orientation)
            logic = widget.sliceLogic()
            if logic:
                logic.FitSliceToAll()

    def collect_working_trajectories(self, workflow_node=None):
        wf = workflow_node or self.workflow_state.resolve_or_create_workflow_node()
        trajectories = []
        for node in self.workflow_state.role_nodes("WorkingTrajectoryLines", workflow_node=wf):
            traj = self.trajectory_scene.trajectory_from_line_node(node.GetName(), node)
            if traj is not None:
                trajectories.append(traj)
        if not trajectories:
            for row in self.trajectory_scene.collect_working_trajectory_rows():
                node = slicer.mrmlScene.GetNodeByID(row["node_id"])
                traj = self.trajectory_scene.trajectory_from_line_node(row["name"], node)
                if traj is not None:
                    trajectories.append(traj)
        trajectories.sort(key=lambda t: t.get("name", ""))
        return trajectories

    def _ijk_kji_to_ras_points(self, volume_node, ijk_kji):
        ijk = np.zeros_like(ijk_kji, dtype=float)
        ijk[:, 0] = ijk_kji[:, 2]
        ijk[:, 1] = ijk_kji[:, 1]
        ijk[:, 2] = ijk_kji[:, 0]
        ijk_h = np.concatenate([ijk, np.ones((ijk.shape[0], 1), dtype=float)], axis=1)
        m_vtk = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(m_vtk)
        m = self._vtk_matrix_to_numpy(m_vtk)
        ras_h = (m @ ijk_h.T).T
        return ras_h[:, :3]

    def _ras_to_ijk_float(self, volume_node, ras_xyz):
        ras_h = np.array([float(ras_xyz[0]), float(ras_xyz[1]), float(ras_xyz[2]), 1.0], dtype=float)
        m_vtk = vtk.vtkMatrix4x4()
        volume_node.GetRASToIJKMatrix(m_vtk)
        m = self._vtk_matrix_to_numpy(m_vtk)
        ijk = m @ ras_h
        return ijk[:3]

    def _volume_center_ras(self, volume_node):
        image_data = volume_node.GetImageData()
        dims = image_data.GetDimensions()
        center_ijk = np.array([0.5 * (dims[0] - 1), 0.5 * (dims[1] - 1), 0.5 * (dims[2] - 1), 1.0], dtype=float)
        m_vtk = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(m_vtk)
        m = self._vtk_matrix_to_numpy(m_vtk)
        return (m @ center_ijk)[:3]

    def detect_de_novo(
        self,
        volume_node,
        threshold,
        max_points,
        max_lines,
        inlier_radius_mm,
        min_length_mm,
        min_inliers,
        ransac_iterations,
        use_head_mask,
        head_mask_threshold_hu,
        min_metal_depth_mm,
        max_metal_depth_mm,
    ):
        result = shank_pipeline.run_detection(
            arr_kji=slicer.util.arrayFromVolume(volume_node),
            spacing_xyz=volume_node.GetSpacing(),
            threshold=float(threshold),
            ijk_kji_to_ras_fn=lambda idx: self._ijk_kji_to_ras_points(volume_node, idx),
            ras_to_ijk_fn=lambda ras: self._ras_to_ijk_float(volume_node, ras),
            center_ras=self._volume_center_ras(volume_node),
            max_points=int(max_points),
            max_lines=int(max_lines),
            inlier_radius_mm=float(inlier_radius_mm),
            min_length_mm=float(min_length_mm),
            min_inliers=int(min_inliers),
            ransac_iterations=int(ransac_iterations),
            use_head_mask=bool(use_head_mask),
            build_head_mask=bool(use_head_mask),
            head_mask_threshold_hu=float(head_mask_threshold_hu),
            head_mask_aggressive_cleanup=False,
            head_mask_close_mm=2.0,
            head_mask_method="outside_air",
            head_mask_metal_dilate_mm=1.0,
            min_metal_depth_mm=float(min_metal_depth_mm),
            max_metal_depth_mm=float(max_metal_depth_mm),
            models_by_id=None,
            min_model_score=None,
        )
        return result

    def _next_side_names(self, lines, existing_names, midline_x_ras=0.0):
        used = set(existing_names)
        r_count = max([int(n[1:]) for n in used if len(n) == 3 and n.startswith("R") and n[1:].isdigit()] or [0])
        l_count = max([int(n[1:]) for n in used if len(n) == 3 and n.startswith("L") and n[1:].isdigit()] or [0])
        names = []
        for line in lines:
            mid_x = 0.5 * (float(line["start_ras"][0]) + float(line["end_ras"][0]))
            if mid_x >= float(midline_x_ras):
                r_count += 1
                name = f"R{r_count:02d}"
            else:
                l_count += 1
                name = f"L{l_count:02d}"
            while name in used:
                if name.startswith("R"):
                    r_count += 1
                    name = f"R{r_count:02d}"
                else:
                    l_count += 1
                    name = f"L{l_count:02d}"
            used.add(name)
            names.append(name)
        return names

    def upsert_detected_lines(self, lines, replace_existing=True):
        existing_rows = self.trajectory_scene.collect_working_trajectory_rows()
        if bool(replace_existing):
            for row in existing_rows:
                node = slicer.mrmlScene.GetNodeByID(row["node_id"])
                if node is not None:
                    slicer.mrmlScene.RemoveNode(node)
            existing_rows = []

        existing_names = [row["name"] for row in existing_rows]
        names = self._next_side_names(lines, existing_names=existing_names, midline_x_ras=0.0)
        new_rows = list(existing_rows)
        for i, line in enumerate(lines):
            name = names[i]
            node = self.trajectory_scene.create_or_update_trajectory_line(
                name=name,
                start_ras=line["start_ras"],
                end_ras=line["end_ras"],
                node_id=None,
            )
            new_rows.append(
                {
                    "name": name,
                    "node_id": node.GetID(),
                    "start_ras": [float(line["start_ras"][0]), float(line["start_ras"][1]), float(line["start_ras"][2])],
                    "end_ras": [float(line["end_ras"][0]), float(line["end_ras"][1]), float(line["end_ras"][2])],
                }
            )
        new_rows.sort(key=lambda r: r["name"])
        return new_rows

    def publish_working_rows(self, rows, workflow_node=None):
        nodes = []
        for row in rows:
            node = slicer.mrmlScene.GetNodeByID(row.get("node_id", ""))
            if node is not None:
                nodes.append(node)
        self.workflow_publish.publish_nodes(
            role="WorkingTrajectoryLines",
            nodes=nodes,
            source="postop_ct_localization",
            space_name="ROSA_BASE",
            workflow_node=workflow_node,
        )

