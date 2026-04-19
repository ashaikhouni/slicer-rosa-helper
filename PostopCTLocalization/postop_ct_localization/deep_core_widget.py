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
    "model_fit.v2_contact_hu": "deepCoreV2ContactHuSpin",
    "model_fit.v2_contact_probe_radius_mm": "deepCoreV2ContactProbeRadiusSpin",
    "model_fit.v2_contact_max_gap_mm": "deepCoreV2ContactMaxGapSpin",
    "model_fit.v2_profile_step_mm": "deepCoreV2ProfileStepSpin",
    "model_fit.v2_profile_disc_radius_mm": "deepCoreV2ProfileDiscRadiusSpin",
    "model_fit.v2_profile_min_peak_sep_mm": "deepCoreV2ProfileMinPeakSepSpin",
    "model_fit.v2_profile_peak_hu_floor": "deepCoreV2ProfilePeakHuFloorSpin",
    "model_fit.v2_profile_peak_rel_frac": "deepCoreV2ProfilePeakRelFracSpin",
    "model_fit.v2_profile_peak_match_tol_mm": "deepCoreV2ProfilePeakMatchTolSpin",
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

    def _build_deep_core_v2_tab(self):
        tab = qt.QWidget()
        self.modeTabs.addTab(tab, "Deep Core v2")
        form = qt.QFormLayout(tab)

        help_text = qt.QLabel(
            "Bolt-first pipeline: detect bolts via RANSAC on bright metal, "
            "then fit electrode axes via cylinder RANSAC on intracranial contacts."
        )
        help_text.wordWrap = True
        form.addRow(help_text)

        button_row = qt.QHBoxLayout()
        run_button = qt.QPushButton("Run Deep Core v2")
        run_button.clicked.connect(self.onRunDeepCoreV2Clicked)
        button_row.addWidget(run_button)
        button_row.addStretch(1)
        form.addRow(button_row)

    def _register_contact_pitch_feature_volumes(self, reference_volume_node, features):
        """Register LoG / Frangi / head-distance / masks / bolts as Slicer
        scalar volumes named ``<CT>_ContactPitch_<feature>`` so they can
        be inspected in the 3D + slice views.
        """
        if not features or reference_volume_node is None:
            return
        base = reference_volume_node.GetName() or "ContactPitch"
        feature_labels = (
            ("log_sigma1", "LoG_sigma1", True),
            ("frangi_sigma1", "Frangi_sigma1", True),
            ("head_distance", "HeadDistance_mm", True),
            ("intracranial", "IntracranialMask", False),
            ("hull", "HullMask", False),
            ("bolt_mask", "BoltMask", False),
        )
        registered = []
        for key, label, percentile_wl in feature_labels:
            arr = features.get(key)
            if arr is None:
                continue
            try:
                node = self.logic._update_scalar_volume_from_array(
                    reference_volume_node, f"{base}_ContactPitch_{label}", arr,
                )
            except Exception as exc:
                self.log(f"[contact-pitch-v1] skipped feature {label}: {exc}")
                continue
            if node is None:
                continue
            if percentile_wl and np is not None:
                self._set_percentile_window_level(node, arr)
            registered.append(label)
        if registered:
            self.log(f"[contact-pitch-v1] feature volumes: {', '.join(registered)}")

    @staticmethod
    def _set_percentile_window_level(node, array):
        """Override auto-W/L with a [2, 98] percentile-based window so
        signed-float feature volumes (LoG, Frangi, head-distance) show
        useful contrast instead of being crushed by outliers.
        """
        if np is None or node is None:
            return
        try:
            finite = np.asarray(array, dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size < 100:
                return
            p2, p98 = np.percentile(finite, [2.0, 98.0])
            window = float(p98 - p2)
            level = 0.5 * float(p98 + p2)
            if window <= 1e-6:
                return
            display = node.GetDisplayNode()
            if display is None:
                return
            try:
                display.AutoWindowLevelOff()
            except Exception:
                pass
            display.SetWindow(window)
            display.SetLevel(level)
        except Exception:
            pass

    def _build_contact_pitch_v1_tab(self):
        tab = qt.QWidget()
        self.modeTabs.addTab(tab, "Contact Pitch v1")
        form = qt.QFormLayout(tab)

        help_text = qt.QLabel(
            "Direct shank detection (no bolt stage). Stage 1: LoG \u03c3=1 "
            "regional-minima blobs + Dixi 3.5 mm pitch walk. Stage 2: Frangi "
            "tube fallback (CC + PCA) for shanks without resolvable contacts. "
            "Trajectories are tagged stage1/stage2 and span the detected "
            "contact range (not bolt entry to deep tip)."
        )
        help_text.wordWrap = True
        form.addRow(help_text)

        button_row = qt.QHBoxLayout()
        run_button = qt.QPushButton("Run Contact Pitch v1")
        run_button.clicked.connect(self.onRunContactPitchV1Clicked)
        button_row.addWidget(run_button)
        button_row.addStretch(1)
        form.addRow(button_row)

        # Manufacturer filter for the electrode-type suggestion. Detection
        # itself is unchanged; only the ``suggested_model_id`` stamped on
        # each trajectory line is affected. User can mix manufacturers or
        # tick none to skip suggestions entirely. Vendors are derived
        # from the bundled electrode library so new entries appear
        # automatically.
        self.contactPitchVendorChecks = {}
        vendor_row = qt.QHBoxLayout()
        vendors = self._discover_contact_pitch_vendors()
        default_on = {"Dixi"}
        for vendor in vendors:
            box = qt.QCheckBox(vendor)
            box.setChecked(vendor in default_on)
            self.contactPitchVendorChecks[vendor] = box
            vendor_row.addWidget(box)
        vendor_row.addStretch(1)
        form.addRow("Manufacturers:", vendor_row)

        # Inline progress UI so the user sees live feedback without
        # having to scroll to the status text panel.  The detector
        # emits ~12 checkpoint messages; setting max to 12 gives a
        # sensible determinate bar.  The status label shows the
        # latest message verbatim.
        self.contactPitchProgressBar = qt.QProgressBar()
        self.contactPitchProgressBar.setRange(0, 12)
        self.contactPitchProgressBar.setValue(0)
        self.contactPitchProgressBar.setTextVisible(True)
        self.contactPitchProgressBar.setFormat("step %v / %m")
        form.addRow("Progress:", self.contactPitchProgressBar)
        self.contactPitchStatusLabel = qt.QLabel("idle")
        self.contactPitchStatusLabel.wordWrap = True
        form.addRow("Status:", self.contactPitchStatusLabel)

    # Map from vendor label shown in the UI to the model-id prefix it
    # corresponds to in ``electrode_models.json``. Keep this mapping here
    # rather than extracting a raw prefix list so the UI stays clean even
    # if library entries grow id suffixes like "DIXI-"/"PMT-" etc.
    _CONTACT_PITCH_VENDOR_LABEL_BY_PREFIX = {
        "DIXI-": "Dixi",
        "PMT-": "PMT",
        "AD-": "AdTech",
        "ADTECH-": "AdTech",
    }

    def _discover_contact_pitch_vendors(self):
        """Return the sorted list of manufacturer labels present in the
        bundled electrode library. Unknown prefixes fall back to the
        token before the first ``-`` capitalized."""
        try:
            from rosa_core.electrode_models import load_electrode_library
            library = load_electrode_library()
        except Exception:
            return ["Dixi", "PMT"]
        found = set()
        for model in library.get("models") or []:
            mid = str(model.get("id") or "")
            label = None
            for prefix, candidate in self._CONTACT_PITCH_VENDOR_LABEL_BY_PREFIX.items():
                if mid.startswith(prefix):
                    label = candidate
                    break
            if label is None and "-" in mid:
                label = mid.split("-", 1)[0].title()
            if label:
                found.add(label)
        ordered = sorted(found, key=lambda s: s.lower())
        return ordered or ["Dixi", "PMT"]

    def _selected_contact_pitch_vendors(self):
        vendors = []
        for label, box in (self.contactPitchVendorChecks or {}).items():
            if box.isChecked():
                vendors.append(label)
        return tuple(vendors)

    def onRunContactPitchV1Clicked(self):
        volume_node = self.ctSelector.currentNode()
        if volume_node is None:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Postop CT Localization", "Select a CT volume.")
            return
        # Reset inline progress UI at the top of the tab so the user
        # can see it without scrolling.
        self.contactPitchProgressBar.setValue(0)
        self.contactPitchStatusLabel.setText("starting…")
        try:
            slicer.app.processEvents()
        except Exception:
            pass
        try:
            self.log("[contact-pitch-v1] running two-stage LoG+Frangi detector...")
            pipeline = self.logic.pipeline_registry.create_pipeline("contact_pitch_v1")
            ctx = self.logic.build_deep_core_context(volume_node, config=None)
            # Forward the manufacturer-filter selection so the fit module
            # only emits suggested electrode ids from the chosen vendors.
            # Empty selection → no suggestions (advisory field stays blank).
            ctx["contact_pitch_v1_vendors"] = list(
                self._selected_contact_pitch_vendors()
            )

            # Live progress: forward each pipeline checkpoint to the
            # inline progress bar + status label + pump Qt events so
            # the UI repaints during the ~10–20 s detection.
            def _progress(msg):
                self.contactPitchStatusLabel.setText(str(msg))
                cur = int(self.contactPitchProgressBar.value)
                mx = int(self.contactPitchProgressBar.maximum)
                self.contactPitchProgressBar.setValue(min(cur + 1, mx))
                self.log(f"[contact-pitch-v1]   {msg}")
                try:
                    slicer.app.processEvents()
                except Exception:
                    pass
            ctx["logger"] = _progress

            det_result = pipeline.run(ctx)
            if det_result.get("status") == "error":
                raise RuntimeError(det_result.get("error", {}).get("message", "unknown"))
            trajectories = list(det_result.get("trajectories") or [])
            counts = det_result.get("diagnostics", {}).get("counts", {})
            self.log(
                f"[contact-pitch-v1] {len(trajectories)} trajectories "
                f"(stage1={counts.get('stage1_count', 0)}, "
                f"stage2={counts.get('stage2_count', 0)})"
            )

            features = getattr(pipeline, "_last_feature_arrays", None) or {}
            self._register_contact_pitch_feature_volumes(volume_node, features)

            self.logic.trajectory_scene.remove_preview_lines()
            self.logic.register_postop_ct(volume_node, workflow_node=self.workflowNode)
            # Render the line from skull_entry → deep_tip (the
            # intracranial portion only) so downstream modules such as
            # Contacts & Trajectory View compute trajectory length as
            # intracranial rather than bolt-tip-to-deep-tip. Keep the
            # original bolt-tip endpoint as an extra field for any
            # consumer that still wants it.
            render_trajectories = []
            for t in trajectories:
                r = dict(t)
                se = r.get("skull_entry_ras")
                if se is not None and len(list(se)) >= 3:
                    r["bolt_tip_ras"] = list(r.get("start_ras") or [])
                    r["start_ras"] = [float(v) for v in list(se)[:3]]
                render_trajectories.append(r)
            nodes = self.logic.show_deep_core_proposals(
                volume_node=volume_node, proposals=render_trajectories
            ) or []
            self._lastDeepCoreProposalNodes = nodes

            # Stamp the per-trajectory electrode suggestion onto each
            # line node (when present) so downstream modules such as
            # Contacts & Trajectory View can pick up a default
            # electrode model to fit against. Order of ``nodes`` mirrors
            # the ``trajectories`` list produced by the pipeline.
            suggestion_log = []
            n_suggested = 0
            for idx, node in enumerate(nodes):
                if idx >= len(trajectories):
                    break
                traj = trajectories[idx]
                suggested = str(traj.get("suggested_model_id") or "")
                node_name = node.GetName()
                n_obs = int(traj.get("n_inliers") or 0)
                span_mm = float(traj.get("contact_span_mm") or 0.0)
                intra_mm = float(traj.get("intracranial_length_mm") or 0.0)
                source = str(traj.get("source") or "unknown")
                if suggested:
                    try:
                        # Rosa.BestModelId is the attribute read by
                        # trajectory_scene.trajectory_from_line_node and
                        # consumed by the Contacts & Trajectory View
                        # module's "Electrode Model" dropdown as
                        # ``traj["best_model_id"]``. Set it so our
                        # suggestion populates that dropdown.
                        node.SetAttribute("Rosa.BestModelId", suggested)
                        score = float(traj.get("suggested_model_score") or 0.0)
                        node.SetAttribute("Rosa.BestModelScore", f"{score:.3f}")
                        method = str(traj.get("suggested_model_method") or "")
                        if method:
                            node.SetAttribute("Rosa.SuggestedElectrodeMethod", method)
                    except Exception:
                        pass
                    method_str = str(traj.get("suggested_model_method") or "")
                    method_tag = f" [{method_str}]" if method_str else ""
                    suggestion_log.append(
                        f"  {node_name}: {suggested}{method_tag} "
                        f"(src={source}, n={n_obs}, span={span_mm:.1f} mm, "
                        f"intracranial={intra_mm:.1f} mm)"
                    )
                    n_suggested += 1
                else:
                    # Classification now runs for every trajectory
                    # (stage-1 and stage-2) via the intracranial-length
                    # shortest-covering rule. Missing suggestions mean
                    # either the vendor filter is empty or no model in
                    # the chosen vendors is long enough to cover the
                    # observed intracranial length.
                    if not self._selected_contact_pitch_vendors():
                        reason = "no manufacturer ticked"
                    elif intra_mm < 5.0:
                        reason = "intracranial length too short"
                    else:
                        reason = "no model in selected vendors covers intracranial length"
                    suggestion_log.append(
                        f"  {node_name}: \u2014 ({reason}; "
                        f"src={source}, intracranial={intra_mm:.1f} mm)"
                    )
            if suggestion_log:
                self.log(
                    f"[contact-pitch-v1] suggested electrodes: "
                    f"{n_suggested}/{len(nodes)}"
                )
                for line in suggestion_log:
                    self.log(line)

            if nodes:
                rows = []
                for ni, node in enumerate(nodes):
                    traj = self.logic.trajectory_scene.trajectory_from_line_node("", node)
                    if traj is None:
                        continue
                    row = {
                        "name": str(traj.get("name") or ""),
                        "node_name": str(traj.get("node_name") or node.GetName() or ""),
                        "node_id": str(traj.get("node_id") or node.GetID() or ""),
                        "group": str(traj.get("group") or "autofit_preview"),
                        "start_ras": list(traj.get("start") or [0.0, 0.0, 0.0]),
                        "end_ras": list(traj.get("end") or [0.0, 0.0, 0.0]),
                    }
                    if ni < len(trajectories):
                        det = trajectories[ni]
                        suggested = str(det.get("suggested_model_id") or "")
                        if suggested:
                            row["suggested_model_id"] = suggested
                        if det.get("intracranial_length_mm") is not None:
                            row["intracranial_length_mm"] = float(det["intracranial_length_mm"])
                    rows.append(row)
                if rows:
                    self.logic.publish_working_rows(
                        rows,
                        workflow_node=self.workflowNode,
                        role="DeepCoreTrajectoryLines",
                        source="postop_ct_contact_pitch",
                    )
                    self._set_workflow_active_source("deep_core")
                    self._set_guided_source_combo("deep_core")
                    self.onRefreshClicked()

            self.log(f"[contact-pitch-v1] published {len(nodes)} trajectory lines to workflow")
            # Finalize progress UI on success.
            self.contactPitchProgressBar.setValue(self.contactPitchProgressBar.maximum)
            self.contactPitchStatusLabel.setText(
                f"done — {len(trajectories)} trajectories "
                f"(stage1={counts.get('stage1_count', 0)}, "
                f"stage2={counts.get('stage2_count', 0)})"
            )
        except Exception as exc:
            self.log(f"[contact-pitch-v1] error: {exc}")
            self.contactPitchStatusLabel.setText(f"error: {exc}")
            import traceback; traceback.print_exc()
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Postop CT Localization", str(exc))

    def onRunDeepCoreV2Clicked(self):
        volume_node = self.ctSelector.currentNode()
        if volume_node is None:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Postop CT Localization", "Select a CT volume.")
            return
        try:
            config = self._deep_core_config_from_ui()
            self.log("[deep-core-v2] running bolt-first pipeline...")
            pipeline = self.logic.pipeline_registry.create_pipeline("deep_core_v2")
            ctx = self.logic.build_deep_core_context(volume_node, config)
            det_result = pipeline.run(ctx)
            if det_result.get("status") == "error":
                raise RuntimeError(det_result.get("error", {}).get("message", "unknown"))
            trajectories = list(det_result.get("trajectories") or [])
            self.log(f"[deep-core-v2] {len(trajectories)} trajectories detected")

            self.logic.trajectory_scene.remove_preview_lines()
            self.logic.register_postop_ct(volume_node, workflow_node=self.workflowNode)
            nodes = self.logic.show_deep_core_proposals(
                volume_node=volume_node, proposals=trajectories
            ) or []
            self._lastDeepCoreProposalNodes = nodes

            if nodes:
                rows = []
                for node in nodes:
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

            self.log(f"[deep-core-v2] published {len(nodes)} trajectory lines to workflow")
        except Exception as exc:
            self.log(f"[deep-core-v2] error: {exc}")
            import traceback; traceback.print_exc()
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Postop CT Localization", str(exc))

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
