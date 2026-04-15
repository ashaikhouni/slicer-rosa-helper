import os
import csv

try:
    import numpy as np
except ImportError:
    np = None

from __main__ import ctk, qt, slicer

from .deep_core_config import deep_core_default_config, deep_core_ui_specs


_DEEP_CORE_CONTROL_ATTRS = {
    "mask.hull_threshold_hu": "deepCoreHullThresholdSpin",
    "mask.hull_clip_hu": "deepCoreHullClipSpin",
    "mask.hull_sigma_mm": "deepCoreHullSigmaSpin",
    "mask.hull_open_vox": "deepCoreHullOpenSpin",
    "mask.hull_close_vox": "deepCoreHullCloseSpin",
    "mask.deep_core_shrink_mm": "deepCoreShrinkSpin",
    "mask.metal_threshold_hu": "deepCoreMetalThresholdSpin",
    "mask.bolt_metal_threshold_hu": "deepCoreBoltMetalThresholdSpin",
    "mask.metal_grow_vox": "deepCoreMetalGrowSpin",
    "support.support_spacing_mm": "deepCoreSupportSpacingSpin",
    "support.component_min_elongation": "deepCoreComponentMinElongationSpin",
    "support.line_atom_diameter_max_mm": "deepCoreLineAtomDiameterSpin",
    "support.line_atom_min_span_mm": "deepCoreLineAtomMinSpanSpin",
    "support.line_atom_min_pca_dominance": "deepCoreLineAtomPCADominanceSpin",
    "support.contact_component_diameter_max_mm": "deepCoreContactComponentDiameterSpin",
    "support.support_cube_size_mm": "deepCoreSupportCubeSizeSpin",
    "annulus.pre_extension_annulus_reject_percentile": "deepCorePreExtensionAnnulusRejectSpin",
    "annulus.cross_section_annulus_inner_mm": "deepCoreAnnulusInnerSpin",
    "annulus.cross_section_annulus_outer_mm": "deepCoreAnnulusOuterSpin",
    "annulus.profile_remaining_brain_fraction_req": "deepCoreProfileRemainingBrainFractionSpin",
    "annulus.profile_simple_middle_nonbrain_max_run_mm": "deepCoreSimpleMiddleNonbrainRunSpin",
    "annulus.profile_multi_middle_nonbrain_max_run_mm": "deepCoreMultiMiddleNonbrainRunSpin",
    "annulus.profile_endpoint_margin_mm": "deepCoreProfileEndpointMarginSpin",
    "proposal.outward_support_check_span_mm": "deepCoreOutwardSupportCheckSpanSpin",
    "proposal.outward_support_radius_mm": "deepCoreOutwardSupportRadiusSpin",
    "proposal.outward_support_search_mm": "deepCoreOutwardSupportSearchSpin",
    "proposal.outward_support_min_extension_mm": "deepCoreOutwardSupportMinExtensionSpin",
    "proposal.outward_support_min_depth_gain_mm": "deepCoreOutwardSupportMinDepthGainSpin",
    "internal.complex_seed_preferred_annulus_percentile": "deepCoreSeedPreferredAnnulusSpin",
    "internal.complex_seed_min_head_depth_mm": "deepCoreSeedMinHeadDepthSpin",
    "model_fit.use_bolt_detection": "deepCoreUseBoltDetectionCheck",
    "model_fit.bolt_bridge_radial_tol_mm": "deepCoreBoltBridgeRadialTolSpin",
    "model_fit.bolt_endpoint_offset_mm": "deepCoreBoltEndpointOffsetSpin",
    "bolt.enabled": "deepCoreBoltEnabledCheck",
    "bolt.span_min_mm": "deepCoreBoltSpanMinSpin",
    "bolt.span_max_mm": "deepCoreBoltSpanMaxSpin",
    "bolt.inlier_tol_mm": "deepCoreBoltInlierTolSpin",
    "bolt.min_inliers": "deepCoreBoltMinInliersSpin",
    "bolt.fill_frac_min": "deepCoreBoltFillFracMinSpin",
    "bolt.max_gap_mm": "deepCoreBoltMaxGapSpin",
    "bolt.shell_min_mm": "deepCoreBoltShellMinSpin",
    "bolt.shell_max_mm": "deepCoreBoltShellMaxSpin",
    "bolt.axis_depth_delta_mm": "deepCoreBoltAxisDepthDeltaSpin",
    "bolt.support_overlap_frac": "deepCoreBoltSupportOverlapSpin",
    "bolt.collinear_angle_deg": "deepCoreBoltCollinearAngleSpin",
    "bolt.collinear_perp_mm": "deepCoreBoltCollinearPerpSpin",
    "bolt.max_lines": "deepCoreBoltMaxLinesSpin",
    "bolt.n_samples": "deepCoreBoltNSamplesSpin",
}

class DeepCoreDebugWidgetMixin:
    def _current_deep_core_support_result(self, volume_node=None):
        result = getattr(self, "_lastDeepCoreDebugResult", None)
        if result is None:
            return None
        current_volume = volume_node if volume_node is not None else self.ctSelector.currentNode()
        if current_volume is None or str(result.volume_node_id) != str(current_volume.GetID()):
            return None
        return result

    def _current_deep_core_proposal_result(self, volume_node=None):
        result = getattr(self, "_lastDeepCoreProposalResult", None)
        if result is None:
            return None
        current_volume = volume_node if volume_node is not None else self.ctSelector.currentNode()
        if current_volume is None or str(result.volume_node_id) != str(current_volume.GetID()):
            return None
        return result

    def _make_deep_core_numeric_control(self, field_spec):
        control_kind = str(field_spec.control or "double")
        if control_kind == "bool":
            control = qt.QCheckBox()
            control.setChecked(bool(field_spec.default))
        elif control_kind == "int":
            control = qt.QSpinBox()
            control.setRange(int(field_spec.minimum), int(field_spec.maximum))
            control.setValue(int(field_spec.default))
            if str(field_spec.suffix or ""):
                control.setSuffix(str(field_spec.suffix))
        else:
            control = qt.QDoubleSpinBox()
            control.setRange(float(field_spec.minimum), float(field_spec.maximum))
            control.setDecimals(int(field_spec.decimals))
            control.setValue(float(field_spec.default))
            if str(field_spec.suffix or ""):
                control.setSuffix(str(field_spec.suffix))
        if str(field_spec.description or ""):
            control.setToolTip(str(field_spec.description))
        return control

    def _deep_core_control_value(self, field_spec):
        control = getattr(self, _DEEP_CORE_CONTROL_ATTRS[str(field_spec.path)])
        control_kind = str(field_spec.control or "double")
        if control_kind == "bool":
            return bool(control.isChecked())
        if control_kind == "int":
            return int(control.value)
        return float(control.value)

    def _deep_core_config_from_ui(self):
        config = deep_core_default_config()
        updates = {}
        for spec in deep_core_ui_specs(advanced=None):
            updates[str(spec.path)] = self._deep_core_control_value(spec)
        return config.with_updates(updates)

    def _log_deep_core_config(self, config):
        parts = [f"{key}={value}" for key, value in sorted(dict(config.to_flat_dict()).items())]
        self.log("[deep-core] config " + ", ".join(parts))

    def _build_deep_core_debug_tab(self):
        tab = qt.QWidget()
        self.modeTabs.addTab(tab, "Deep Core Debug")
        form = qt.QFormLayout(tab)

        help_text = qt.QLabel(
            "Build a smoothed non-air hull, shrink it to a deep core, then intersect it with a "
            "metal mask to inspect deep seed support before de novo proposal generation."
        )
        help_text.wordWrap = True
        form.addRow(help_text)

        for spec in deep_core_ui_specs(advanced=False):
            control = self._make_deep_core_numeric_control(spec)
            setattr(self, _DEEP_CORE_CONTROL_ATTRS[str(spec.path)], control)
            form.addRow(str(spec.label), control)

        advanced_section = ctk.ctkCollapsibleButton()
        advanced_section.text = "Advanced"
        advanced_section.collapsed = True
        advanced_form = qt.QFormLayout(advanced_section)
        for spec in deep_core_ui_specs(advanced=True):
            control = self._make_deep_core_numeric_control(spec)
            setattr(self, _DEEP_CORE_CONTROL_ATTRS[str(spec.path)], control)
            advanced_form.addRow(str(spec.label), control)
        form.addRow(advanced_section)

        self.deepCoreShowSupportCheck = qt.QCheckBox("Show deep-seed support diagnostics")
        self.deepCoreShowSupportCheck.setChecked(True)
        form.addRow(self.deepCoreShowSupportCheck)
        self.deepCoreSupportLegendLabel = qt.QLabel(
            "Debug markers: yellow=raw component centroids, dark blue=line-atom support samples, "
            "green=contact support samples, orange=complex-path support samples, gold=support atom centers, "
            "cyan=support atom lines."
        )
        self.deepCoreSupportLegendLabel.wordWrap = True
        form.addRow(self.deepCoreSupportLegendLabel)

        self.deepCoreProposalDisplayModeCombo = qt.QComboBox()
        self.deepCoreProposalDisplayModeCombo.addItem("All proposal families", "all")
        self.deepCoreProposalDisplayModeCombo.addItem("Local families", "local_only")
        self.deepCoreProposalDisplayModeCombo.addItem("Graph + blob-connectivity only", "graph_blob_only")
        form.addRow("Proposal display", self.deepCoreProposalDisplayModeCombo)

        self.deepCoreProposalLegendLabel = qt.QLabel(
            "Colors: graph=green, blob-connectivity=cyan, blob-axis=orange."
        )
        self.deepCoreProposalLegendLabel.wordWrap = True
        form.addRow(self.deepCoreProposalLegendLabel)

        button_row = qt.QHBoxLayout()
        build_button = qt.QPushButton("Build Deep-Core Debug Masks")
        build_button.clicked.connect(self.onBuildDeepCoreDebugClicked)
        button_row.addWidget(build_button)
        proposal_button = qt.QPushButton("Build Proposal Lines")
        proposal_button.clicked.connect(self.onBuildDeepCoreProposalsClicked)
        button_row.addWidget(proposal_button)
        button_row.addStretch(1)
        form.addRow(button_row)

        self.deepCoreProposalDisplayModeCombo.currentIndexChanged.connect(self.onDeepCoreProposalDisplayModeChanged)

    def _deep_core_display_mode_key(self):
        return str(self.deepCoreProposalDisplayModeCombo.currentData or "all")

    @staticmethod
    def _filter_deep_core_proposals_for_display(proposals, mode_key="all"):
        family_map = {
            "all": None,
            "local_only": {"graph", "blob_connectivity", "blob_axis"},
            "graph_blob_only": {"graph", "blob_connectivity"},
        }
        keep_families = family_map.get(str(mode_key or "all"))
        proposal_list = list(proposals or [])
        if keep_families is None:
            return proposal_list
        return [proposal for proposal in proposal_list if str(proposal.get("proposal_family") or "") in keep_families]

    @staticmethod
    def _format_deep_core_proposal_family_counts(proposals):
        proposal_list = list(proposals or [])
        if not proposal_list:
            return "none"
        ordered_families = (
            "graph",
            "blob_connectivity",
            "blob_axis",
        )
        counts = {}
        for proposal in proposal_list:
            family = str(proposal.get("proposal_family") or "unknown")
            counts[family] = int(counts.get(family, 0)) + 1
        parts = []
        for family in ordered_families:
            count = int(counts.pop(family, 0))
            if count > 0:
                parts.append(f"{family}={count}")
        for family in sorted(counts.keys()):
            parts.append(f"{family}={int(counts[family])}")
        return ", ".join(parts) if parts else "none"

    def _refresh_deep_core_proposal_display(self, log_summary=False):
        volume_node = self.ctSelector.currentNode()
        proposal_result = self._current_deep_core_proposal_result(volume_node=volume_node)
        if volume_node is None or proposal_result is None:
            self._lastDeepCoreProposalNodes = []
            return []
        proposals = list(proposal_result.payload.get("proposals") or [])
        mode_key = self._deep_core_display_mode_key()
        displayed = self._filter_deep_core_proposals_for_display(proposals, mode_key=mode_key)
        self._lastDeepCoreProposalNodes = list(
            self.logic.show_deep_core_proposals(volume_node=volume_node, proposals=displayed) or []
        )
        if log_summary:
            self.log(
                "[deep-core] display mode={mode} shown={shown}/{total} families={families}".format(
                    mode=mode_key,
                    shown=len(displayed),
                    total=len(proposals),
                    families=self._format_deep_core_proposal_family_counts(displayed),
                )
            )
        return displayed

    def onDeepCoreProposalDisplayModeChanged(self, _idx):
        self._refresh_deep_core_proposal_display(log_summary=True)

    def onBuildDeepCoreDebugClicked(self):
        volume_node = self.ctSelector.currentNode()
        if volume_node is None:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Postop CT Localization", "Select a CT volume.")
            return
        try:
            config = self._deep_core_config_from_ui()
            self._log_deep_core_config(config)
            support_result = self.logic.run_deep_core_debug(
                volume_node=volume_node,
                config=config,
                show_support_diagnostics=bool(self.deepCoreShowSupportCheck.checked),
            )
        except Exception as exc:
            self.log(f"[deep-core] error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Postop CT Localization", str(exc))
            return

        self._lastDeepCoreDebugResult = support_result
        self._lastDeepCoreProposalResult = None
        self.logic.trajectory_scene.remove_preview_lines()
        self.logic.register_postop_ct(volume_node, workflow_node=self.workflowNode)
        self._apply_primary_slice_layers()
        mask_payload = dict(support_result.mask_result.payload or {})
        support_payload = dict(support_result.payload or {})

        try:
            self.logic.show_metal_and_head_masks(
                volume_node=volume_node,
                metal_mask_kji=mask_payload.get("metal_mask_kji"),
                head_mask_kji=mask_payload.get("hull_mask_kji"),
                head_distance_map_kji=mask_payload.get("head_distance_map_kji"),
                head_core_mask_kji=mask_payload.get("deep_core_mask_kji"),
                metal_gate_mask_kji=mask_payload.get("metal_grown_mask_kji"),
                metal_in_gate_mask_kji=mask_payload.get("deep_seed_mask_kji"),
                metal_depth_pass_mask_kji=mask_payload.get("deep_seed_raw_mask_kji"),
            )
            self.log("[deep-core] displayed deep-core mask overlay")
        except Exception as exc:
            self.log(f"[deep-core] mask display warning: {exc}")

        if bool(self.deepCoreShowSupportCheck.checked):
            try:
                self.logic.show_blob_diagnostics(
                    volume_node=volume_node,
                    blob_labelmap_kji=support_payload.get("blob_labelmap_kji"),
                    blob_centroids_all_ras=support_payload.get("blob_centroids_all_ras"),
                    blob_centroids_kept_ras=support_payload.get("blob_sample_points_ras"),
                    blob_centroids_rejected_ras=np.empty((0, 3), dtype=float),
                    line_blob_points_ras=support_payload.get("line_blob_sample_points_ras"),
                    contact_blob_points_ras=support_payload.get("contact_blob_sample_points_ras"),
                    complex_blob_points_ras=support_payload.get("complex_blob_sample_points_ras"),
                    complex_blob_chain_rows=support_payload.get("complex_blob_chain_rows"),
                    contact_chain_rows=support_payload.get("contact_chain_rows"),
                )
                blob_label_node = None
                try:
                    blob_label_node = slicer.util.getNode(f"{volume_node.GetName()}_BlobLabelMap")
                except Exception:
                    blob_label_node = None
                if blob_label_node is not None:
                    slicer.util.setSliceViewerLayers(
                        background=volume_node,
                        foreground=None,
                        foregroundOpacity=0.0,
                        label=blob_label_node,
                        labelOpacity=0.55,
                )
                self.log("[deep-core] displayed deep-seed blob diagnostics")
            except Exception as exc:
                self.log(f"[deep-core] blob display warning: {exc}")
            try:
                self.logic.show_deep_core_support_atoms(
                    volume_node=volume_node,
                    support_atoms=support_payload.get("support_atoms"),
                )
                self.log(
                    "[deep-core] displayed support atoms in 3D (atoms={count})".format(
                        count=len(list(support_payload.get("support_atoms") or [])),
                    )
                )
            except Exception as exc:
                self.log(f"[deep-core] support-atom display warning: {exc}")
        else:
            try:
                self.logic.show_deep_core_support_atoms(
                    volume_node=volume_node,
                    support_atoms=[],
                )
            except Exception:
                pass

        stats = dict(support_result.stats or {})
        self.log(
            "[deep-core] hull={hull} core={core} metal={metal} grown={grown} deepSeed={seed} "
            "blobs={blobs} samples={samples} metalThr={threshold:.1f}HU".format(
                hull=int(stats.get("hull_voxels", 0)),
                core=int(stats.get("deep_core_voxels", 0)),
                metal=int(stats.get("metal_voxels", 0)),
                grown=int(stats.get("metal_grown_voxels", 0)),
                seed=int(stats.get("deep_seed_voxels", 0)),
                blobs=int(stats.get("deep_seed_raw_blob_count", 0)),
                samples=int(stats.get("deep_seed_sample_count", 0)),
                threshold=float(stats.get("metal_threshold_hu", 0.0)),
            )
        )
        chain_rows = list(support_payload.get("complex_blob_chain_rows") or [])
        if chain_rows:
            try:
                chain_path = self._write_complex_blob_chain_tsv(volume_node, chain_rows)
                self.log(
                    "[deep-core] complex-blob chain rows={rows} tsv={path}".format(
                        rows=len(chain_rows),
                        path=chain_path,
                    )
                )
            except Exception as exc:
                self.log(f"[deep-core] complex-blob chain dump warning: {exc}")
        contact_chain_debug_rows = list(support_payload.get("contact_chain_debug_rows") or [])
        if contact_chain_debug_rows:
            try:
                debug_path = self._write_contact_chain_debug_tsv(volume_node, contact_chain_debug_rows)
                self.log(
                    "[deep-core] contact-chain debug rows={rows} tsv={path}".format(
                        rows=len(contact_chain_debug_rows),
                        path=debug_path,
                    )
                )
            except Exception as exc:
                self.log(f"[deep-core] contact-chain debug dump warning: {exc}")
        smooth_node = mask_payload.get("smoothed_hull_volume_node")
        distance_node = mask_payload.get("head_distance_volume_node")
        if smooth_node is not None:
            self.log(f"[deep-core] smoothed hull volume: {smooth_node.GetName()}")
        if distance_node is not None:
            self.log(f"[deep-core] head distance volume: {distance_node.GetName()}")

    def _write_complex_blob_chain_tsv(self, volume_node, chain_rows):
        rows = list(chain_rows or [])
        if not rows:
            return ""
        temp_root = str(getattr(slicer.app, "temporaryPath", "/tmp") or "/tmp")
        os.makedirs(temp_root, exist_ok=True)
        safe_name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(volume_node.GetName() or "volume"))
        out_path = os.path.join(temp_root, f"{safe_name}_complex_blob_chains.tsv")
        fieldnames = [
            "blob_id",
            "node_id",
            "node_role",
            "line_id",
            "line_order",
            "bin_id",
            "degree",
            "support_x_ras",
            "support_y_ras",
            "support_z_ras",
        ]
        with open(out_path, "w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for row in rows:
                row_dict = dict(row or {})
                writer.writerow({key: row_dict.get(key, "") for key in fieldnames})
        return out_path

    def _write_contact_chain_debug_tsv(self, volume_node, debug_rows):
        rows = list(debug_rows or [])
        if not rows:
            return ""
        temp_root = str(getattr(slicer.app, "temporaryPath", "/tmp") or "/tmp")
        os.makedirs(temp_root, exist_ok=True)
        safe_name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(volume_node.GetName() or "volume"))
        out_path = os.path.join(temp_root, f"{safe_name}_contact_chain_debug.tsv")
        fieldnames = [
            "seed_atom_i",
            "seed_atom_j",
            "stage",
            "status",
            "reason",
            "seed_score",
            "dist_mm",
            "extra_support",
            "support_score",
            "chain_len",
            "chain_atom_ids",
            "chain_id",
            "span_mm",
            "rms_mm",
            "rms_limit_mm",
            "median_gap_mm",
            "max_gap_mm",
            "spacing_cv",
            "mean_dot",
            "min_dot",
            "grow_direction",
            "stop_atom_id",
            "prev_atom_id",
            "rejection_histogram",
            "break_index",
            "break_gap_mm",
            "segment_index",
            "segment_atom_ids",
        ]
        with open(out_path, "w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for row in rows:
                row_dict = dict(row or {})
                writer.writerow({key: row_dict.get(key, "") for key in fieldnames})
        return out_path

    def onBuildDeepCoreProposalsClicked(self):
        volume_node = self.ctSelector.currentNode()
        if volume_node is None:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Postop CT Localization", "Select a CT volume.")
            return
        support_result = self._current_deep_core_support_result(volume_node=volume_node)
        if support_result is None:
            qt.QMessageBox.information(
                slicer.util.mainWindow(),
                "Postop CT Localization",
                "Build Deep-Core Debug Masks first for the current CT.",
            )
            return
        try:
            config = self._deep_core_config_from_ui()
            self._log_deep_core_config(config)
            if support_result is None:
                support_result = self.logic.run_deep_core_debug(
                    volume_node=volume_node,
                    config=config,
                    show_support_diagnostics=False,
                )
                self._lastDeepCoreDebugResult = support_result
            proposal_result = self.logic.run_deep_core_proposals(
                volume_node=volume_node,
                config=config,
                debug_result=support_result,
            )
        except Exception as exc:
            self.log(f"[deep-core] proposal error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Postop CT Localization", str(exc))
            return

        self._lastDeepCoreProposalResult = proposal_result
        proposals = list(proposal_result.payload.get("proposals") or [])
        displayed = self._refresh_deep_core_proposal_display(log_summary=False)
        proposal_nodes = list(getattr(self, "_lastDeepCoreProposalNodes", []) or [])
        if proposal_nodes:
            rows = []
            for node in proposal_nodes:
                traj = self.logic.trajectory_scene.trajectory_from_line_node("", node)
                if traj is None:
                    continue
                rows.append(
                    {
                        "name": str(traj.get("name") or ""),
                        "node_name": str(traj.get("node_name") or node.GetName() or ""),
                        "node_id": str(traj.get("node_id") or node.GetID() or ""),
                        "group": str(traj.get("group") or "autofit_preview"),
                        "start_ras": list(traj.get("start") or [0.0, 0.0, 0.0]),
                        "end_ras": list(traj.get("end") or [0.0, 0.0, 0.0]),
                    }
                )
            if rows:
                self.logic.publish_working_rows(
                    rows,
                    workflow_node=self.workflowNode,
                    role="DeepCoreTrajectoryLines",
                    source="postop_ct_deep_core",
                )
                self._set_workflow_active_source("deep_core")
                self._set_guided_source_combo("deep_core")
                self.onRefreshClicked()
        self.log(
            "[deep-core] proposals={count}/{total} mode={mode} candidates={cand} tokens={tok} families={families}".format(
                count=len(displayed),
                total=len(proposals),
                mode=self._deep_core_display_mode_key(),
                cand=int(proposal_result.payload.get("candidate_count", 0)),
                tok=int(proposal_result.payload.get("token_count", 0)),
                families=self._format_deep_core_proposal_family_counts(proposals),
            )
        )
        for idx, proposal in enumerate(displayed[:12], start=1):
            self.log(
                "[deep-core] P{idx:02d} family={family} span={span:.1f}mm tokens={tokens} blobs={blobs} rms={rms:.2f} score={score:.1f}".format(
                    idx=idx,
                    family=str(proposal.get("proposal_family") or "unknown"),
                    span=float(proposal.get("span_mm", 0.0)),
                    tokens=int(proposal.get("inlier_count", 0)),
                    blobs=int(proposal.get("distinct_blob_count", 0)),
                    rms=float(proposal.get("rms_mm", 0.0)),
                    score=float(proposal.get("score", 0.0)),
                )
            )
