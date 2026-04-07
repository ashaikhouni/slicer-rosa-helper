import json
import os

try:
    import numpy as np
except ImportError:
    np = None

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

from __main__ import ctk, qt, slicer, vtk

from rosa_core import (
    load_electrode_library,
    lps_to_ras_point,
    model_map,
    suggest_model_id_for_trajectory,
    trajectory_length_mm,
)
from rosa_core.contact_fit import fit_electrode_axis_and_tip
from rosa_scene import ElectrodeSceneService, LayoutService, TrajectoryFocusController, TrajectorySceneService
from shank_core.blob_candidates import build_blob_labelmap, extract_blob_candidates
from shank_core.masking import build_preview_masks, compute_head_distance_map_kji, largest_component_binary
from shank_engine import PipelineRegistry, register_builtin_pipelines
from rosa_workflow import WorkflowPublisher, WorkflowState
from rosa_workflow.workflow_registry import table_to_dict_rows

from .constants import DE_NOVO_MODE_SPECS, GUIDED_SOURCE_OPTIONS

class GuidedFitWidgetMixin:
    def _build_guided_fit_tab(self):
        tab = qt.QWidget()
        self.modeTabs.addTab(tab, "Guided Fit")
        form = qt.QFormLayout(tab)

        threshold_widget, guided_threshold_slider, guided_threshold_spin = self._build_hu_threshold_selector(
            default_hu=600.0
        )
        self.guidedThresholdSlider = guided_threshold_slider
        self.guidedThresholdSpin = guided_threshold_spin

        self.guidedHeadThresholdSpin = qt.QDoubleSpinBox()
        self.guidedHeadThresholdSpin.setRange(-1200.0, 500.0)
        self.guidedHeadThresholdSpin.setDecimals(1)
        self.guidedHeadThresholdSpin.setValue(-500.0)
        self.guidedHeadThresholdSpin.setSuffix(" HU")

        self.guidedMinDepthSpin = qt.QDoubleSpinBox()
        self.guidedMinDepthSpin.setRange(0.0, 100.0)
        self.guidedMinDepthSpin.setDecimals(2)
        self.guidedMinDepthSpin.setValue(5.0)
        self.guidedMinDepthSpin.setSuffix(" mm")

        self.guidedMaxDepthSpin = qt.QDoubleSpinBox()
        self.guidedMaxDepthSpin.setRange(0.0, 300.0)
        self.guidedMaxDepthSpin.setDecimals(2)
        self.guidedMaxDepthSpin.setValue(220.0)
        self.guidedMaxDepthSpin.setSuffix(" mm")

        guided_info = qt.QLabel(
            "Adaptive guided fit: threshold, ROI, and sparse-support retries are automatic."
        )
        guided_info.wordWrap = True
        form.addRow(guided_info)

        detectRow = qt.QHBoxLayout()
        self.detectCandidatesButton = qt.QPushButton("Detect Candidates")
        self.detectCandidatesButton.clicked.connect(self.onDetectCandidatesClicked)
        detectRow.addWidget(self.detectCandidatesButton)
        self.resetWindowButton = qt.QPushButton("Reset CT Window")
        self.resetWindowButton.clicked.connect(self.onResetCTWindowClicked)
        detectRow.addWidget(self.resetWindowButton)
        detectRow.addStretch(1)
        form.addRow(detectRow)

        self.guidedRoiRadiusSpin = qt.QDoubleSpinBox()
        self.guidedRoiRadiusSpin.setRange(1.0, 6.0)
        self.guidedRoiRadiusSpin.setDecimals(2)
        self.guidedRoiRadiusSpin.setValue(5.0)
        self.guidedRoiRadiusSpin.setSuffix(" mm")

        self.guidedMaxAngleSpin = qt.QDoubleSpinBox()
        self.guidedMaxAngleSpin.setRange(1.0, 25.0)
        self.guidedMaxAngleSpin.setDecimals(1)
        self.guidedMaxAngleSpin.setValue(12.0)
        self.guidedMaxAngleSpin.setSuffix(" deg")

        self.guidedMaxDepthShiftSpin = qt.QDoubleSpinBox()
        self.guidedMaxDepthShiftSpin.setRange(1.0, 50.0)
        self.guidedMaxDepthShiftSpin.setDecimals(1)
        self.guidedMaxDepthShiftSpin.setValue(2.0)
        self.guidedMaxDepthShiftSpin.setSuffix(" mm")

        self.guidedFitModeCombo = qt.QComboBox()
        self.guidedFitModeCombo.addItem("Deep-anchored (v2)", "deep_anchor_v2")

        fitRow = qt.QHBoxLayout()
        self.fitSelectedButton = qt.QPushButton("Fit Checked")
        self.fitSelectedButton.clicked.connect(self.onFitSelectedClicked)
        fitRow.addWidget(self.fitSelectedButton)
        self.fitAllButton = qt.QPushButton("Fit All")
        self.fitAllButton.clicked.connect(self.onFitAllClicked)
        fitRow.addWidget(self.fitAllButton)
        self.applyFitButton = qt.QPushButton("Apply Fit to Trajectories")
        self.applyFitButton.clicked.connect(self.onApplyFitClicked)
        self.applyFitButton.setEnabled(False)
        self.applyFitButton.setVisible(False)
        fitRow.addWidget(self.applyFitButton)
        form.addRow(fitRow)

    def _populate_guided_trajectory_table(self):
        self._updatingGuidedTable = True
        try:
            self.guidedTrajectoryTable.setRowCount(0)
            for row, traj in enumerate(self.loadedTrajectories):
                self.guidedTrajectoryTable.insertRow(row)

                use_check = qt.QCheckBox()
                use_check.setChecked(True)
                use_check.setStyleSheet("margin-left:8px; margin-right:8px;")
                self.guidedTrajectoryTable.setCellWidget(row, 0, use_check)

                traj_name = str(traj.get("name", ""))
                name_item = qt.QTableWidgetItem(traj_name)
                name_item.setFlags(name_item.flags() | qt.Qt.ItemIsEditable)
                name_item.setData(qt.Qt.UserRole, traj_name)
                self.guidedTrajectoryTable.setItem(row, 1, name_item)

                length_item = qt.QTableWidgetItem(f"{trajectory_length_mm(traj):.2f}")
                length_item.setFlags(length_item.flags() & ~qt.Qt.ItemIsEditable)
                self.guidedTrajectoryTable.setItem(row, 2, length_item)

            if self.guidedTrajectoryTable.rowCount > 0:
                self.guidedTrajectoryTable.selectRow(0)
        finally:
            self._updatingGuidedTable = False

    def onGuidedTrajectoryItemChanged(self, item):
        if self._updatingGuidedTable or self._renamingGuidedTrajectory:
            return
        if item is None or item.column() != 1:
            return
        row = int(item.row())
        if row < 0 or row >= len(self.loadedTrajectories):
            return

        old_name = str(item.data(qt.Qt.UserRole) or self.loadedTrajectories[row].get("name", "")).strip()
        new_name = str(item.text() or "").strip()
        if not old_name:
            return
        if not new_name:
            self._renamingGuidedTrajectory = True
            item.setText(old_name)
            self._renamingGuidedTrajectory = False
            return
        if new_name == old_name:
            return

        for r in range(self.guidedTrajectoryTable.rowCount):
            if r == row:
                continue
            other = self.guidedTrajectoryTable.item(r, 1)
            if other and str(other.text() or "").strip().lower() == new_name.lower():
                self._renamingGuidedTrajectory = True
                item.setText(old_name)
                self._renamingGuidedTrajectory = False
                qt.QMessageBox.warning(
                    slicer.util.mainWindow(),
                    "Postop CT Localization",
                    f"Trajectory name '{new_name}' already exists.",
                )
                return

        node_id = self.loadedTrajectories[row].get("node_id", "")
        if not self.logic.rename_trajectory(node_id=node_id, new_name=new_name):
            self._renamingGuidedTrajectory = True
            item.setText(old_name)
            self._renamingGuidedTrajectory = False
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "Postop CT Localization",
                f"Failed to rename trajectory '{old_name}'.",
            )
            return

        self.loadedTrajectories[row]["name"] = new_name
        item.setData(qt.Qt.UserRole, new_name)
        self.log(f"[trajectory] renamed {old_name} -> {new_name}")
        self._refresh_summary()
        self._schedule_guided_follow()

    def onDetectCandidatesClicked(self):
        volume_node = self.ctSelector.currentNode()
        if volume_node is None:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Postop CT Localization", "Select a CT volume.")
            return
        threshold = float(self.guidedThresholdSpin.value)
        try:
            detect_payload = self.logic.extract_threshold_candidates_lps(
                volume_node=volume_node,
                threshold=threshold,
                head_mask_threshold_hu=float(self.guidedHeadThresholdSpin.value),
                min_metal_depth_mm=float(self.guidedMinDepthSpin.value),
                max_metal_depth_mm=float(self.guidedMaxDepthSpin.value),
            )
        except Exception as exc:
            self.log(f"[guided] detect error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Postop CT Localization", str(exc))
            return
        points_lps = np.asarray(detect_payload["points_lps"], dtype=float)
        used_threshold = float(detect_payload.get("threshold_hu") or threshold)
        self.candidatesLPS = points_lps
        self.fitResults = {}
        self.applyFitButton.setEnabled(False)

        self.logic.register_postop_ct(volume_node, workflow_node=self.workflowNode)
        self._apply_primary_slice_layers()
        self.guidedThresholdSpin.setValue(used_threshold)
        self.logic.apply_ct_window_from_threshold(volume_node, threshold=used_threshold)
        n = int(points_lps.shape[0]) if np is not None else len(points_lps)
        self.log(f"[guided] detected {n} candidate points at threshold {used_threshold:.1f}")
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
            # Model-free guided fit: still allow trajectory axis/depth fitting.
            return {
                "trajectory": traj["name"],
                "model_id": "",
                "tip_at": "target",
                "tip_shift_mm": 0.0,
                "xyz_offset_mm": [0.0, 0.0, 0.0],
            }
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
        applied_nodes = []
        self.logic.trajectory_scene.remove_preview_lines(node_prefix="AutoFit_")
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
            offsets = model["contact_center_offsets_from_tip_mm"] if model else []
            fit = None
            roi_values = []
            for roi_mm in (float(self.guidedRoiRadiusSpin.value), 8.0, 10.0):
                if any(abs(roi_mm - existing) < 1e-6 for existing in roi_values):
                    continue
                roi_values.append(roi_mm)
            for roi_mm in roi_values:
                fit = fit_electrode_axis_and_tip(
                    candidate_points_lps=self.candidatesLPS,
                    planned_entry_lps=traj["start"],
                    planned_target_lps=traj["end"],
                    contact_offsets_mm=offsets,
                    tip_at=assignment.get("tip_at", "target"),
                    roi_radius_mm=float(roi_mm),
                    max_angle_deg=float(self.guidedMaxAngleSpin.value),
                    max_depth_shift_mm=float(self.guidedMaxDepthShiftSpin.value),
                    fit_mode=str(self.guidedFitModeCombo.currentData or "deep_anchor_v2"),
                )
                fit["roi_radius_mm_attempted"] = float(roi_mm)
                if bool(fit.get("success")):
                    break
                reason = str(fit.get("reason") or "").lower()
                sparse_failure = (
                    "too few" in reason
                    or "no candidate" in reason
                    or "no compact clusters" in reason
                    or int(fit.get("points_in_roi") or 0) < 300
                )
                if not sparse_failure:
                    break
            if fit.get("success"):
                self.fitResults[name] = fit
                start_ras = lps_to_ras_point(fit["entry_lps"])
                end_ras = lps_to_ras_point(fit["target_lps"])
                existing = self.logic.trajectory_scene.find_line_by_group_and_name(name, "guided_fit")
                node = self.logic.trajectory_scene.create_or_update_trajectory_line(
                    name=name,
                    start_ras=start_ras,
                    end_ras=end_ras,
                    node_id=None if existing is None else existing.GetID(),
                    node_name=f"Guided_{name}",
                    group="guided_fit",
                    origin="postop_ct_guided_fit",
                )
                applied_nodes.append(node)
                success += 1
                self.log(
                    "[guided] {name}: mode={m} angle={a:.2f} deg depth={s:.2f} mm lateral={l:.2f} mm residual={r:.2f} mm".format(
                        name=name,
                        m=str(fit.get("fit_mode_used") or ""),
                        a=float(fit.get("angle_deg", 0.0)),
                        s=float(fit.get("tip_shift_mm", 0.0)),
                        l=float(fit.get("lateral_shift_mm", 0.0)),
                        r=float(fit.get("residual_mm", 0.0)),
                    )
                )
                if str(fit.get("fit_mode_used") or "") == "deep_anchor_v2":
                    self.log(
                        "[guided] {name}: anchor={src} source={source} mode={mode} morph={morph} terminal_cluster={cluster} candidates={count}".format(
                            name=name,
                            src=str(fit.get("deep_anchor_source") or ""),
                            source=str(fit.get("terminal_blob_source") or ""),
                            mode=str(fit.get("terminal_anchor_mode") or ""),
                            morph=str(fit.get("local_terminal_morphology") or ""),
                            cluster=str(fit.get("terminal_blob_selected_cluster_id") or ""),
                            count=int(fit.get("terminal_blob_candidate_count") or 0),
                        )
                    )
            else:
                self.log(f"[guided] {name}: failed ({fit.get('reason', 'unknown')})")
        if applied_nodes:
            self.workflowPublisher.publish_nodes(
                role="GuidedFitTrajectoryLines",
                nodes=applied_nodes,
                source="postop_ct_guided_fit",
                space_name="ROSA_BASE",
                workflow_node=self.workflowNode,
            )
            self.workflowPublisher.publish_nodes(
                role="WorkingTrajectoryLines",
                nodes=applied_nodes,
                source="postop_ct_guided_fit",
                space_name="ROSA_BASE",
                workflow_node=self.workflowNode,
            )
            self.logic.trajectory_scene.place_trajectory_nodes_in_hierarchy(
                context_id=self.workflowState.context_id(workflow_node=self.workflowNode),
                nodes=applied_nodes,
            )
            self.logic.trajectory_scene.show_only_groups(["guided_fit"])
            self._set_workflow_active_source("guided_fit")
            self._set_guided_source_combo("guided_fit")
            self.log(f"[guided] applied fitted trajectories: {len(applied_nodes)}")
            self.log(
                "[contacts] guided trajectories updated. Use module 'Contacts & Trajectory View' "
                "with source 'Guided Fit' to generate contacts and QC."
            )
            self.onRefreshClicked()

        self.applyFitButton.setEnabled(False)
        self.log(f"[guided] fitted {success}/{len(names)} trajectories")

    def onFitSelectedClicked(self):
        names = self._checked_guided_trajectory_names()
        if not names:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "Postop CT Localization",
                "Check at least one trajectory in the table.",
            )
            return
        try:
            self._fit_names(names)
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
        qt.QMessageBox.information(
            slicer.util.mainWindow(),
            "Postop CT Localization",
            "Guided fit is now applied directly when you click 'Fit Selected' or 'Fit All'.",
        )

    def _checked_guided_trajectory_names(self):
        names = []
        for row in range(self.guidedTrajectoryTable.rowCount):
            check = self.guidedTrajectoryTable.cellWidget(row, 0)
            if check is None or not bool(check.checked):
                continue
            item = self.guidedTrajectoryTable.item(row, 1)
            if item is None:
                continue
            name = str(item.text() or "").strip()
            if name:
                names.append(name)
        return names

    def onRemoveCheckedClicked(self):
        names = self._checked_guided_trajectory_names()
        if not names:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "Postop CT Localization",
                "Check at least one trajectory to remove.",
            )
            return
        removed = self.logic.remove_trajectories_by_name(names=names, source_key=self._selected_guided_source_key())
        for name in names:
            self.fitResults.pop(name, None)
        self.log(f"[trajectory] removed {removed}/{len(names)} trajectories")
        self.onRefreshClicked()
