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

class DeNovoWidgetMixin:
    def _build_de_novo_mode_tabs(self):
        """Create separate de novo tabs so each fit mode exposes only relevant controls."""
        for spec in DE_NOVO_MODE_SPECS:
            self._build_de_novo_mode_tab(spec)

    def _build_de_novo_mode_tab(self, mode_spec):
        """Build one de novo mode tab and persist its controls by pipeline key."""
        pipeline_key = str(mode_spec.get("pipeline_key") or "").strip()
        show_blob_controls = bool(mode_spec.get("show_blob_controls"))
        tab = qt.QWidget()
        tab_index = self.modeTabs.addTab(tab, str(mode_spec.get("tab_label") or pipeline_key))
        self._deNovoTabIndexByPipeline[pipeline_key] = int(tab_index)
        form = qt.QFormLayout(tab)
        controls = {}
        self._deNovoControlsByPipeline[pipeline_key] = controls
        controls["pipeline_key"] = pipeline_key
        controls["show_blob_controls"] = show_blob_controls

        threshold_widget, threshold_slider, threshold_spin = self._build_hu_threshold_selector(default_hu=1800.0)
        form.addRow("Metal threshold", threshold_widget)
        controls["threshold_slider"] = threshold_slider
        controls["threshold_spin"] = threshold_spin

        inlier_radius_spin = qt.QDoubleSpinBox()
        inlier_radius_spin.setRange(0.3, 6.0)
        inlier_radius_spin.setDecimals(2)
        inlier_radius_spin.setValue(1.2)
        inlier_radius_spin.setSuffix(" mm")
        form.addRow("Inlier radius", inlier_radius_spin)
        controls["inlier_radius_spin"] = inlier_radius_spin

        min_length_spin = qt.QDoubleSpinBox()
        min_length_spin.setRange(5.0, 200.0)
        min_length_spin.setDecimals(1)
        min_length_spin.setValue(20.0)
        min_length_spin.setSuffix(" mm")
        form.addRow("Min trajectory length", min_length_spin)
        controls["min_length_spin"] = min_length_spin

        min_inliers_spin = qt.QSpinBox()
        # Single user-facing inlier control shared by all de novo pipelines.
        min_inliers_spin.setRange(1, 1000)
        min_inliers_spin.setValue(250)
        form.addRow("Min inliers", min_inliers_spin)
        controls["min_inliers_spin"] = min_inliers_spin

        max_points_spin = qt.QSpinBox()
        max_points_spin.setRange(5000, 1000000)
        max_points_spin.setSingleStep(5000)
        max_points_spin.setValue(300000)
        form.addRow("Max sampled points", max_points_spin)
        controls["max_points_spin"] = max_points_spin

        ransac_spin = qt.QSpinBox()
        ransac_spin.setRange(20, 2000)
        ransac_spin.setValue(240)
        form.addRow("RANSAC iterations", ransac_spin)
        controls["ransac_spin"] = ransac_spin

        max_lines_spin = qt.QSpinBox()
        max_lines_spin.setRange(1, 50)
        max_lines_spin.setValue(30)
        form.addRow("Max lines", max_lines_spin)
        controls["max_lines_spin"] = max_lines_spin

        use_exact_count_check = qt.QCheckBox("Use exact trajectory count")
        use_exact_count_check.setChecked(False)
        form.addRow(use_exact_count_check)
        controls["use_exact_count_check"] = use_exact_count_check

        exact_count_spin = qt.QSpinBox()
        exact_count_spin.setRange(1, 50)
        exact_count_spin.setValue(10)
        exact_count_spin.setEnabled(False)
        form.addRow("Exact count", exact_count_spin)
        controls["exact_count_spin"] = exact_count_spin
        use_exact_count_check.toggled.connect(
            lambda checked, c=controls: self._onUseExactToggled(c, checked)
        )

        use_head_mask_check = qt.QCheckBox("Use head-depth gating")
        use_head_mask_check.setChecked(True)
        form.addRow(use_head_mask_check)
        controls["use_head_mask_check"] = use_head_mask_check

        head_threshold_spin = qt.QDoubleSpinBox()
        head_threshold_spin.setRange(-1200.0, 500.0)
        head_threshold_spin.setDecimals(1)
        head_threshold_spin.setValue(-500.0)
        head_threshold_spin.setSuffix(" HU")
        form.addRow("Head threshold", head_threshold_spin)
        controls["head_threshold_spin"] = head_threshold_spin

        head_mask_method_combo = qt.QComboBox()
        head_mask_method_combo.addItem("Outside air (default)", "outside_air")
        head_mask_method_combo.addItem("Not-air LCC", "not_air_lcc")
        form.addRow("Head mask method", head_mask_method_combo)
        controls["head_mask_method_combo"] = head_mask_method_combo

        min_depth_spin = qt.QDoubleSpinBox()
        min_depth_spin.setRange(0.0, 100.0)
        min_depth_spin.setDecimals(2)
        min_depth_spin.setValue(5.0)
        min_depth_spin.setSuffix(" mm")
        form.addRow("Min metal depth", min_depth_spin)
        controls["min_depth_spin"] = min_depth_spin

        max_depth_spin = qt.QDoubleSpinBox()
        max_depth_spin.setRange(0.0, 300.0)
        max_depth_spin.setDecimals(2)
        max_depth_spin.setValue(220.0)
        max_depth_spin.setSuffix(" mm")
        form.addRow("Max metal depth", max_depth_spin)
        controls["max_depth_spin"] = max_depth_spin

        start_zone_spin = qt.QDoubleSpinBox()
        start_zone_spin.setRange(0.0, 50.0)
        start_zone_spin.setDecimals(2)
        start_zone_spin.setValue(10.0)
        start_zone_spin.setSuffix(" mm")
        start_zone_spin.setToolTip(
            "Each detected shank must include at least one support point in "
            "[Min metal depth, Min metal depth + this window]."
        )
        form.addRow("Start-zone window", start_zone_spin)
        controls["start_zone_spin"] = start_zone_spin

        enable_rescue_check = qt.QCheckBox("Enable rescue pass")
        enable_rescue_check.setChecked(True)
        form.addRow(enable_rescue_check)
        controls["enable_rescue_check"] = enable_rescue_check

        rescue_scale_spin = qt.QDoubleSpinBox()
        rescue_scale_spin.setRange(0.1, 1.0)
        rescue_scale_spin.setDecimals(2)
        rescue_scale_spin.setSingleStep(0.05)
        rescue_scale_spin.setValue(0.6)
        form.addRow("Rescue min-inlier scale", rescue_scale_spin)
        controls["rescue_scale_spin"] = rescue_scale_spin

        rescue_max_lines_spin = qt.QSpinBox()
        rescue_max_lines_spin.setRange(0, 30)
        rescue_max_lines_spin.setValue(6)
        form.addRow("Rescue max lines", rescue_max_lines_spin)
        controls["rescue_max_lines_spin"] = rescue_max_lines_spin

        use_model_score_check = qt.QCheckBox("Use model-template scoring")
        use_model_score_check.setChecked(True)
        form.addRow(use_model_score_check)
        controls["use_model_score_check"] = use_model_score_check

        min_model_score_spin = qt.QDoubleSpinBox()
        min_model_score_spin.setRange(-1.0, 5.0)
        min_model_score_spin.setDecimals(2)
        min_model_score_spin.setSingleStep(0.05)
        min_model_score_spin.setValue(0.10)
        form.addRow("Min model score", min_model_score_spin)
        controls["min_model_score_spin"] = min_model_score_spin
        use_model_score_check.toggled.connect(
            lambda checked, c=controls: self._onUseModelScoreToggled(c, checked)
        )
        self._onUseModelScoreToggled(controls, bool(use_model_score_check.checked))

        debug_diagnostics_check = qt.QCheckBox("Debug diagnostics (nodes + JSON)")
        debug_diagnostics_check.setChecked(False)
        form.addRow(debug_diagnostics_check)
        controls["debug_diagnostics_check"] = debug_diagnostics_check

        replace_check = qt.QCheckBox("Replace existing working trajectories")
        replace_check.setChecked(True)
        form.addRow(replace_check)
        controls["replace_check"] = replace_check

        # Blob-v2-only controls.
        blob_group = ctk.ctkCollapsibleButton()
        blob_group.text = "Blob Fit Options"
        blob_group.collapsed = False
        form.addRow(blob_group)
        blob_form = qt.QFormLayout(blob_group)

        distance_mask_check = qt.QCheckBox("Use distance-gated mask for blob candidates")
        distance_mask_check.setChecked(True)
        distance_mask_check.setToolTip(
            "Use metal depth-pass mask (after min/max depth gating) for blob candidate extraction."
        )
        blob_form.addRow(distance_mask_check)
        controls["distance_mask_check"] = distance_mask_check

        min_blob_voxels_spin = qt.QSpinBox()
        min_blob_voxels_spin.setRange(1, 10000)
        min_blob_voxels_spin.setValue(2)
        blob_form.addRow("Min blob voxels", min_blob_voxels_spin)
        controls["min_blob_voxels_spin"] = min_blob_voxels_spin

        max_blob_voxels_spin = qt.QSpinBox()
        max_blob_voxels_spin.setRange(1, 20000)
        max_blob_voxels_spin.setValue(1200)
        blob_form.addRow("Max blob voxels", max_blob_voxels_spin)
        controls["max_blob_voxels_spin"] = max_blob_voxels_spin

        min_blob_peak_hu_edit = qt.QLineEdit()
        min_blob_peak_hu_edit.setPlaceholderText("optional (empty = disabled)")
        blob_form.addRow("Min blob peak HU", min_blob_peak_hu_edit)
        controls["min_blob_peak_hu_edit"] = min_blob_peak_hu_edit

        use_electrode_priors_check = qt.QCheckBox("Use electrode priors for axial support sampling")
        use_electrode_priors_check.setChecked(True)
        use_electrode_priors_check.setToolTip(
            "Auto-tune axial support sampling parameters from loaded electrode models "
            "(diameter/contact length/contact spacing)."
        )
        blob_form.addRow(use_electrode_priors_check)
        controls["use_electrode_priors_check"] = use_electrode_priors_check

        controls["blob_group"] = blob_group
        blob_group.setVisible(show_blob_controls)
        blob_group.setEnabled(show_blob_controls)
        if not show_blob_controls:
            distance_mask_check.setChecked(False)
            use_electrode_priors_check.setChecked(False)

        detect_button = qt.QPushButton("Detect Trajectories")
        detect_button.clicked.connect(
            lambda _checked=False, pk=pipeline_key: self.onDeNovoDetectClicked(pipeline_key=pk)
        )
        form.addRow(detect_button)
        controls["detect_button"] = detect_button

    @staticmethod
    def _onUseExactToggled(controls, checked):
        exact_count_spin = controls.get("exact_count_spin")
        if exact_count_spin is not None:
            exact_count_spin.setEnabled(bool(checked))

    @staticmethod
    def _onUseModelScoreToggled(controls, checked):
        min_model_score_spin = controls.get("min_model_score_spin")
        if min_model_score_spin is not None:
            min_model_score_spin.setEnabled(bool(checked))

    def _threshold_widget_pairs(self):
        """Yield all active threshold spin/slider pairs in this module."""
        if hasattr(self, "guidedThresholdSpin") and hasattr(self, "guidedThresholdSlider"):
            yield self.guidedThresholdSpin, self.guidedThresholdSlider
        for controls in self._deNovoControlsByPipeline.values():
            spin = controls.get("threshold_spin")
            slider = controls.get("threshold_slider")
            if spin is not None and slider is not None:
                yield spin, slider

    def _sync_threshold_ranges_from_ct(self, volume_node=None):
        """Expand HU threshold selector ranges to current CT dynamic range."""
        node = volume_node or self.ctSelector.currentNode()
        if node is None or np is None:
            return
        try:
            arr = slicer.util.arrayFromVolume(node)
            lo = float(np.nanmin(arr))
            hi = float(np.nanmax(arr))
        except Exception:
            return
        if not np.isfinite(lo) or not np.isfinite(hi):
            return
        if hi <= lo:
            hi = lo + 1.0
        lo_i = int(np.floor(lo))
        hi_i = int(np.ceil(hi))
        for spin, slider in self._threshold_widget_pairs():
            try:
                current = float(spin.value)
                spin.setRange(float(lo_i), float(hi_i))
                slider.setRange(int(lo_i), int(hi_i))
                clamped = float(np.clip(current, float(lo_i), float(hi_i)))
                block_spin = spin.blockSignals(True)
                block_slider = slider.blockSignals(True)
                try:
                    spin.setValue(clamped)
                    slider.setValue(int(round(clamped)))
                finally:
                    spin.blockSignals(block_spin)
                    slider.blockSignals(block_slider)
            except Exception:
                # Slicer module reload can leave stale PythonQt widget wrappers behind.
                # Ignore destroyed threshold widgets and continue with the live ones.
                continue

    def onDeNovoDetectClicked(self, pipeline_key=None):
        """Run selected de novo pipeline (voxel/blob) and publish detected trajectories."""
        volume_node = self.ctSelector.currentNode()
        if volume_node is None:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Postop CT Localization", "Select a CT volume.")
            return
        selected_pipeline_key = str(
            pipeline_key
            or self._pipeline_key_for_current_mode_tab()
            or self._workflow_de_novo_pipeline_key()
            or self.logic.default_de_novo_pipeline_key
        ).strip()
        controls = self._controls_for_pipeline(selected_pipeline_key)
        uses_blob_pipeline = bool(controls.get("show_blob_controls"))

        try:
            min_blob_peak_hu = self._optional_float_from_text(
                controls.get("min_blob_peak_hu_edit").text if controls.get("min_blob_peak_hu_edit") is not None else ""
            )
        except Exception:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "Postop CT Localization",
                "Min blob peak HU must be numeric or empty.",
            )
            return

        max_lines = int(controls["exact_count_spin"].value) if controls["use_exact_count_check"].checked else int(
            controls["max_lines_spin"].value
        )
        head_mask_method_data = controls["head_mask_method_combo"].currentData
        head_mask_method = head_mask_method_data() if callable(head_mask_method_data) else head_mask_method_data
        head_mask_method = str(head_mask_method or "outside_air")
        self._set_workflow_de_novo_pipeline_key(selected_pipeline_key)
        self.logic.register_postop_ct(volume_node, workflow_node=self.workflowNode)

        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        slicer.app.processEvents()
        pass_models = bool(controls["use_model_score_check"].checked) or (
            uses_blob_pipeline and bool(controls.get("use_electrode_priors_check") and controls["use_electrode_priors_check"].checked)
        )
        try:
            detect_payload = self.logic.detect_de_novo(
                volume_node=volume_node,
                threshold=float(controls["threshold_spin"].value),
                max_points=int(controls["max_points_spin"].value),
                max_lines=max_lines,
                inlier_radius_mm=float(controls["inlier_radius_spin"].value),
                min_length_mm=float(controls["min_length_spin"].value),
                min_inliers=int(controls["min_inliers_spin"].value),
                ransac_iterations=int(controls["ransac_spin"].value),
                use_head_mask=bool(controls["use_head_mask_check"].checked),
                head_mask_threshold_hu=float(controls["head_threshold_spin"].value),
                head_mask_method=head_mask_method,
                min_metal_depth_mm=float(controls["min_depth_spin"].value),
                max_metal_depth_mm=float(controls["max_depth_spin"].value),
                start_zone_window_mm=float(controls["start_zone_spin"].value),
                min_blob_voxels=(int(controls["min_blob_voxels_spin"].value) if uses_blob_pipeline else 2),
                max_blob_voxels=(int(controls["max_blob_voxels_spin"].value) if uses_blob_pipeline else 1200),
                min_blob_peak_hu=(min_blob_peak_hu if uses_blob_pipeline else None),
                max_blob_elongation=None,
                use_distance_mask_for_blobs=(
                    uses_blob_pipeline
                    and bool(controls.get("distance_mask_check") and controls["distance_mask_check"].checked)
                ),
                enable_rescue_pass=bool(controls["enable_rescue_check"].checked),
                rescue_min_inliers_scale=float(controls["rescue_scale_spin"].value),
                rescue_max_lines=int(controls["rescue_max_lines_spin"].value),
                use_model_score=bool(controls["use_model_score_check"].checked),
                use_electrode_priors=(
                    uses_blob_pipeline
                    and bool(controls.get("use_electrode_priors_check") and controls["use_electrode_priors_check"].checked)
                ),
                min_model_score=float(controls["min_model_score_spin"].value),
                models_by_id=(self.modelsById if pass_models else None),
                pipeline_key=selected_pipeline_key,
                debug_masks=bool(controls["debug_diagnostics_check"].checked),
            )
        except Exception as exc:
            self.log(f"[denovo] failed: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Postop CT Localization", str(exc))
            return
        finally:
            qt.QApplication.restoreOverrideCursor()
            slicer.app.processEvents()

        engine_result = dict((detect_payload or {}).get("engine_result") or {})
        raw_payload = dict((detect_payload or {}).get("raw_payload") or {})
        prior_summary = dict((detect_payload or {}).get("prior_summary") or {})
        if prior_summary:
            self.log(
                "[denovo] electrode priors: diam={diam:.2f}mm contact={contact:.2f}mm "
                "pitch={pitch:.2f}mm gap={gap:.2f}mm spacing={spacing:.2f}mm window={window:.2f}mm maxSamples={maxs}".format(
                    diam=float(prior_summary.get("electrode_prior_diameter_mm", 0.0)),
                    contact=float(prior_summary.get("electrode_prior_contact_length_mm", 0.0)),
                    pitch=float(prior_summary.get("electrode_prior_pitch_mm", 0.0)),
                    gap=float(prior_summary.get("electrode_prior_gap_mm", 0.0)),
                    spacing=float(prior_summary.get("axial_support_spacing_mm", 0.0)),
                    window=float(prior_summary.get("axial_support_window_mm", 0.0)),
                    maxs=int(prior_summary.get("axial_support_max_samples_per_blob", 0)),
                )
            )

        if raw_payload:
            try:
                self.logic.show_metal_and_head_masks(
                    volume_node=volume_node,
                    metal_mask_kji=raw_payload.get("metal_mask_kji"),
                    head_mask_kji=raw_payload.get("gating_mask_kji"),
                    head_distance_map_kji=raw_payload.get("head_distance_map_kji"),
                    distance_surface_mask_kji=raw_payload.get("distance_surface_mask_kji"),
                    not_air_mask_kji=raw_payload.get("not_air_mask_kji"),
                    not_air_eroded_mask_kji=raw_payload.get("not_air_eroded_mask_kji"),
                    head_core_mask_kji=raw_payload.get("head_core_mask_kji"),
                    metal_gate_mask_kji=raw_payload.get("metal_gate_mask_kji"),
                    metal_in_gate_mask_kji=raw_payload.get("metal_in_gate_mask_kji"),
                    depth_window_mask_kji=raw_payload.get("depth_window_mask_kji"),
                    metal_depth_pass_mask_kji=raw_payload.get("metal_depth_pass_mask_kji"),
                    blob_kept_mask_kji=raw_payload.get("blob_kept_mask_kji"),
                    blob_rejected_mask_kji=raw_payload.get("blob_rejected_mask_kji"),
                )
                self.log("[denovo] displayed head/metal mask overlay")
            except Exception as exc:
                self.log(f"[denovo] mask display warning: {exc}")

        params = {
            "pipeline_key": selected_pipeline_key,
            "threshold_hu": float(controls["threshold_spin"].value),
            "candidate_mode": "voxel",
            "inlier_radius_mm": float(controls["inlier_radius_spin"].value),
            "min_length_mm": float(controls["min_length_spin"].value),
            "min_inliers": int(controls["min_inliers_spin"].value),
            "ransac_iterations": int(controls["ransac_spin"].value),
            "max_lines": int(max_lines),
            "min_depth_mm": float(controls["min_depth_spin"].value),
            "max_depth_mm": float(controls["max_depth_spin"].value),
            "head_mask_method": head_mask_method,
            "start_zone_window_mm": float(controls["start_zone_spin"].value),
            "min_blob_voxels": int(controls["min_blob_voxels_spin"].value) if uses_blob_pipeline else 0,
            "max_blob_voxels": int(controls["max_blob_voxels_spin"].value) if uses_blob_pipeline else 0,
            "min_blob_peak_hu": min_blob_peak_hu,
            "use_distance_mask_for_blobs": (
                uses_blob_pipeline
                and bool(controls.get("distance_mask_check") and controls["distance_mask_check"].checked)
            ),
            "enable_rescue_pass": bool(controls["enable_rescue_check"].checked),
            "rescue_min_inliers_scale": float(controls["rescue_scale_spin"].value),
            "rescue_max_lines": int(controls["rescue_max_lines_spin"].value),
            "use_model_score": bool(controls["use_model_score_check"].checked),
            "use_electrode_priors": (
                uses_blob_pipeline
                and bool(controls.get("use_electrode_priors_check") and controls["use_electrode_priors_check"].checked)
            ),
            "min_model_score": float(controls["min_model_score_spin"].value),
        }
        diagnostics = self.logic._build_detection_diagnostics(engine_result=engine_result, params=params)

        self.log(
            "[denovo] mode={mode} candidates={cand} after-mask={after_mask} after-depth={after_depth} "
            "min_inliers={min_inliers} inlier_radius={inlier_radius:.2f} fit1={fit1} fit2={fit2} rescue={rescue} "
            "final={final} unassigned={unassigned} total={total:.1f}ms".format(
                mode=str(diagnostics.get("candidate_mode", "voxel")),
                cand=int(diagnostics.get("candidate_points_total", 0)),
                after_mask=int(diagnostics.get("candidate_points_after_mask", 0)),
                after_depth=int(diagnostics.get("candidate_points_after_depth", 0)),
                min_inliers=int(diagnostics.get("effective_min_inliers", 0)),
                inlier_radius=float(diagnostics.get("effective_inlier_radius_mm", 0.0)),
                fit1=int(diagnostics.get("fit1_lines_proposed", 0)),
                fit2=int(diagnostics.get("fit2_lines_kept", 0)),
                rescue=int(diagnostics.get("rescue_lines_kept", 0)),
                final=int(diagnostics.get("final_lines_kept", 0)),
                unassigned=int(diagnostics.get("final_unassigned_points", 0)),
                total=float(diagnostics.get("profile_ms", {}).get("total", 0.0)),
            )
        )
        try:
            blob_all = raw_payload.get("blob_centroids_all_ras")
            if blob_all is None:
                blob_all = np.empty((0, 3), dtype=float)
            blob_kept = raw_payload.get("blob_centroids_kept_ras")
            if blob_kept is None:
                blob_kept = np.empty((0, 3), dtype=float)
            blob_rejected = raw_payload.get("blob_centroids_rejected_ras")
            if blob_rejected is None:
                blob_rejected = np.empty((0, 3), dtype=float)
            if bool(controls["debug_diagnostics_check"].checked):
                self.logic.show_blob_diagnostics(
                    volume_node=volume_node,
                    blob_labelmap_kji=raw_payload.get("blob_labelmap_kji"),
                    blob_centroids_all_ras=blob_all,
                    blob_centroids_kept_ras=blob_kept,
                    blob_centroids_rejected_ras=blob_rejected,
                )
            else:
                # Always show included blob centroids in 3D, keep all/rejected
                # hidden unless debug mode is enabled.
                self.logic.show_blob_diagnostics(
                    volume_node=volume_node,
                    blob_labelmap_kji=None,
                    blob_centroids_all_ras=np.empty((0, 3), dtype=float),
                    blob_centroids_kept_ras=blob_kept,
                    blob_centroids_rejected_ras=np.empty((0, 3), dtype=float),
                )
        except Exception as exc:
            self.log(f"[denovo] blob diagnostics warning: {exc}")

        if bool(controls["debug_diagnostics_check"].checked):
            try:
                out_json = self.logic.write_detection_diagnostics_json(
                    volume_node=volume_node,
                    diagnostics=diagnostics,
                )
                self.log(
                    "[denovo:debug] blobs(total={total}, kept={kept}, reject_small={small}, reject_large={large}, "
                    "reject_intensity={intensity}, reject_shape={shape})".format(
                        total=int(diagnostics.get("blob_count_total", 0)),
                        kept=int(diagnostics.get("blob_count_kept", 0)),
                        small=int(diagnostics.get("blob_reject_small", 0)),
                        large=int(diagnostics.get("blob_reject_large", 0)),
                        intensity=int(diagnostics.get("blob_reject_intensity", 0)),
                        shape=int(diagnostics.get("blob_reject_shape", 0)),
                    )
                )
                if str(selected_pipeline_key) == "blob_em_v2":
                    self.log(
                        "[denovo:debug] support(blobs_total_cc={total_cc}, eligible={eligible}, samples={samples}, "
                        "support_total={total}, compact={compact}, elongated={elongated})".format(
                            total_cc=int(diagnostics.get("blobs_total_cc", 0)),
                            eligible=int(diagnostics.get("blobs_eligible_for_axial_sampling", 0)),
                            samples=int(diagnostics.get("axial_support_samples_generated", 0)),
                            total=int(diagnostics.get("support_points_total", 0)),
                            compact=int(diagnostics.get("support_points_from_compact_blobs", 0)),
                            elongated=int(diagnostics.get("support_points_from_elongated_blobs", 0)),
                        )
                    )
                self.log(f"[denovo:debug] diagnostics json: {out_json}")
            except Exception as exc:
                self.log(f"[denovo] debug diagnostics warning: {exc}")

        lines = self.logic._engine_trajectories_to_lines(engine_result)
        if not lines:
            self.log("[denovo] no trajectories detected")
            return

        rows = self.logic.upsert_detected_lines(
            lines=lines,
            replace_existing=bool(controls["replace_check"].checked),
        )
        self.logic.publish_working_rows(
            rows,
            workflow_node=self.workflowNode,
            role="DeNovoTrajectoryLines",
            source="postop_ct_denovo",
        )
        self.logic.trajectory_scene.show_only_groups(["de_novo"])
        self._set_workflow_active_source("de_novo")
        self._set_guided_source_combo("de_novo")
        self.log(
            f"[denovo] start-zone-reject={int(diagnostics.get('start_zone_reject_count', 0))}, "
            f"new_lines={len(lines)}, total_rows={len(rows)}"
        )
        self.log(
            "[contacts] trajectories updated. Use module 'Contacts & Trajectory View' "
            "with source 'Guided Fit' or 'De Novo' to generate contacts and QC."
        )
        self.onRefreshClicked()


class DeNovoLogicMixin:
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
        head_mask_method,
        min_metal_depth_mm,
        max_metal_depth_mm,
        start_zone_window_mm,
        min_blob_voxels=2,
        max_blob_voxels=1200,
        min_blob_peak_hu=None,
        max_blob_elongation=None,
        use_distance_mask_for_blobs=False,
        enable_rescue_pass=True,
        rescue_min_inliers_scale=0.6,
        rescue_max_lines=6,
        use_model_score=True,
        use_electrode_priors=True,
        min_model_score=0.10,
        models_by_id=None,
        pipeline_key=None,
        debug_masks=False,
    ):
        config = dict(
            threshold=float(threshold),
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
            head_mask_method=str(head_mask_method or "outside_air"),
            head_mask_metal_dilate_mm=1.0,
            head_gate_erode_vox=1,
            head_gate_dilate_vox=1,
            head_gate_margin_mm=0.0,
            min_metal_depth_mm=float(min_metal_depth_mm),
            max_metal_depth_mm=float(max_metal_depth_mm),
            candidate_mode="voxel",
            min_blob_voxels=int(min_blob_voxels),
            max_blob_voxels=int(max_blob_voxels),
            min_blob_peak_hu=min_blob_peak_hu,
            max_blob_elongation=max_blob_elongation,
            use_distance_mask_for_blob_candidates=bool(use_distance_mask_for_blobs),
            enable_rescue_pass=bool(enable_rescue_pass),
            rescue_min_inliers_scale=float(rescue_min_inliers_scale),
            rescue_max_lines=int(rescue_max_lines),
            min_model_score=(float(min_model_score) if bool(use_model_score) else None),
            pipeline_key=str(pipeline_key or self.default_de_novo_pipeline_key),
            debug_masks=bool(debug_masks),
        )
        if bool(use_electrode_priors):
            priors = self._electrode_prior_support_params(models_by_id)
            if priors:
                config.update(priors)
        ctx_extras = {}
        if bool(use_model_score) and isinstance(models_by_id, dict) and models_by_id:
            ctx_extras["models_by_id"] = models_by_id
        ctx = {
            "run_id": f"denovo_{volume_node.GetName()}",
            "arr_kji": slicer.util.arrayFromVolume(volume_node),
            "spacing_xyz": volume_node.GetSpacing(),
            "ijk_kji_to_ras_fn": lambda idx: self._ijk_kji_to_ras_points(volume_node, idx),
            "ras_to_ijk_fn": lambda ras: self._ras_to_ijk_float(volume_node, ras),
            "center_ras": self._volume_center_ras(volume_node),
            "config": config,
            "extras": ctx_extras,
        }
        selected_pipeline_key = str(config.get("pipeline_key") or self.default_de_novo_pipeline_key)
        available = set(self.pipeline_registry.keys())
        if selected_pipeline_key not in available:
            selected_pipeline_key = self.default_de_novo_pipeline_key
        engine_result = self.pipeline_registry.run(selected_pipeline_key, ctx)
        if str(engine_result.get("status", "ok")).lower() == "error":
            error = dict(engine_result.get("error") or {})
            stage = str(error.get("stage") or "pipeline")
            message = str(error.get("message") or "Detection pipeline failed")
            raise RuntimeError(f"{message} (stage={stage}, pipeline={selected_pipeline_key})")
        raw_payload = {}
        if isinstance(ctx_extras, dict):
            maybe_raw = ctx_extras.get("legacy_result")
            if isinstance(maybe_raw, dict):
                raw_payload = maybe_raw
        prior_summary = {}
        if bool(use_electrode_priors):
            for key in (
                "electrode_prior_diameter_mm",
                "electrode_prior_contact_length_mm",
                "electrode_prior_pitch_mm",
                "electrode_prior_gap_mm",
                "electrode_prior_contact_separation_mm",
                "electrode_prior_max_exploration_mm",
                "axial_support_spacing_mm",
                "axial_support_window_mm",
                "axial_support_max_samples_per_blob",
            ):
                if key in config:
                    prior_summary[key] = config.get(key)
        return {
            "engine_result": engine_result,
            "raw_payload": raw_payload,
            "pipeline_key": selected_pipeline_key,
            "prior_summary": prior_summary,
        }

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
