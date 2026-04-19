"""Contact Pitch v1 (Auto Fit) widget tab.

Hosts the Slicer UI for the ``contact_pitch_v1`` detection pipeline —
the production direct-shank SEEG detector. Kept under the legacy
filename ``deep_core_widget.py`` so external imports stay stable, but
the debug/v1/v2 scaffolding this module used to host has been removed.
"""

try:
    import numpy as np
except ImportError:
    np = None

from __main__ import qt, slicer


class ContactPitchV1WidgetMixin:
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
        self.modeTabs.addTab(tab, "Auto Fit")
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
        run_button = qt.QPushButton("Run Auto Fit")
        run_button.clicked.connect(self.onRunContactPitchV1Clicked)
        button_row.addWidget(run_button)
        button_row.addStretch(1)
        form.addRow(button_row)

        # Pitch strategy controls both the walker's candidate pitches
        # and the manufacturer filter for the electrode-type suggestion.
        # "Dixi" runs the legacy single 3.5 mm walker. "PMT" adds
        # PMT's 3.97 mm (16B) and 4.43 mm (16C) variants. "Mixed"
        # unions Dixi + PMT. "Auto-detect" estimates pitch from the
        # intracranial blob cloud's mutual-NN histogram before stage 1
        # runs — useful when the manufacturer is unknown or mixed.
        self.contactPitchStrategyCombo = qt.QComboBox()
        for label, key in (
            ("Dixi (3.5 mm)", "dixi"),
            ("PMT (3.5 / 3.97 / 4.43 mm)", "pmt"),
            ("Mixed Dixi + PMT", "mixed"),
            ("Auto-detect pitch", "auto"),
        ):
            self.contactPitchStrategyCombo.addItem(label, key)
        self.contactPitchStrategyCombo.setCurrentIndex(0)  # default: Dixi
        self.contactPitchStrategyCombo.setToolTip(
            "Walker pitch set + suggestion vendor filter. Detection "
            "results are identical across strategies that share the "
            "same pitch set; only the suggested electrode model id "
            "differs."
        )
        form.addRow("Pitch strategy:", self.contactPitchStrategyCombo)

        self.contactPitchProgressBar = qt.QProgressBar()
        self.contactPitchProgressBar.setRange(0, 12)
        self.contactPitchProgressBar.setValue(0)
        self.contactPitchProgressBar.setTextVisible(True)
        self.contactPitchProgressBar.setFormat("step %v / %m")
        form.addRow("Progress:", self.contactPitchProgressBar)
        self.contactPitchStatusLabel = qt.QLabel("idle")
        self.contactPitchStatusLabel.wordWrap = True
        form.addRow("Status:", self.contactPitchStatusLabel)

    # Vendor sets implied by each strategy. Mirrors
    # ``PITCH_STRATEGY_VENDORS`` in ``contact_pitch_v1_fit`` — duplicated
    # here so the widget can log sensible messages without importing
    # the fit module.
    _CONTACT_PITCH_STRATEGY_VENDORS = {
        "dixi":  ("Dixi",),
        "pmt":   ("PMT",),
        "mixed": ("Dixi", "PMT"),
        "auto":  ("Dixi", "PMT", "AdTech"),
    }

    def _selected_contact_pitch_strategy(self):
        data = self.contactPitchStrategyCombo.currentData
        return str(data or "dixi")

    def _selected_contact_pitch_vendors(self):
        return self._CONTACT_PITCH_STRATEGY_VENDORS.get(
            self._selected_contact_pitch_strategy(), ("Dixi",),
        )

    def onRunContactPitchV1Clicked(self):
        volume_node = self.ctSelector.currentNode()
        if volume_node is None:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Postop CT Localization", "Select a CT volume.")
            return
        self.contactPitchProgressBar.setValue(0)
        self.contactPitchStatusLabel.setText("starting…")
        try:
            slicer.app.processEvents()
        except Exception:
            pass
        try:
            self.log("[contact-pitch-v1] running two-stage LoG+Frangi detector...")
            pipeline = self.logic.pipeline_registry.create_pipeline("contact_pitch_v1")
            ctx = self.logic.build_detection_context(volume_node)
            strategy_key = self._selected_contact_pitch_strategy()
            ctx["contact_pitch_v1_pitch_strategy"] = strategy_key
            ctx["contact_pitch_v1_vendors"] = list(
                self._selected_contact_pitch_vendors()
            )

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
            # Render the line from skull_entry → deep_tip (intracranial
            # portion only) so downstream modules such as Contacts &
            # Trajectory View compute trajectory length as intracranial
            # rather than bolt-tip-to-deep-tip. Keep the original
            # bolt-tip endpoint as an extra field for consumers that
            # still want it.
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
                        # Rosa.BestModelId is read by
                        # trajectory_scene.trajectory_from_line_node and
                        # consumed by the Contacts & Trajectory View
                        # module's "Electrode Model" dropdown.
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
