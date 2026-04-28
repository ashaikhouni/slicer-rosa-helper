"""Guided Fit widget: snap planned / seeded trajectories to the
imaged shank using the LoG-based engine in :mod:`guided_fit_engine`.

The widget is self-contained — seed source dropdown, seed list, and
fit buttons all live inside the tab so there's no ambiguity about
which trajectories the Fit buttons act on. The shared "Trajectory
Set" table below the tabs is a view of the active workflow source
and is not consulted by Guided Fit.

Fit output matches Auto Fit's shape: bolt tip + skull entry + deep
tip, with the rendered line drawn from ``skull_entry → end`` so
downstream modules see intracranial length.
"""
try:
    import numpy as np
except ImportError:
    np = None

from __main__ import qt, slicer

from rosa_core import lps_to_ras_point, trajectory_length_mm

from . import contact_pitch_v1_fit as cpfit
from . import guided_fit_engine as gfe


# Seed sources the Guided Fit tab offers. Excludes "working" and
# "guided_fit" — neither is a sensible seed (working is a merged view;
# guided_fit is the *output* of this tab, not its input).
_GUIDED_SEED_SOURCES = (
    ("auto_fit", "Auto Fit"),
    ("imported_rosa", "Imported ROSA"),
    ("imported_external", "Imported External"),
    ("manual", "Manual (scene)"),
    ("planned_rosa", "Planned ROSA"),
)


class GuidedFitWidgetMixin:
    def _build_guided_fit_tab(self):
        tab = qt.QWidget()
        self.modeTabs.addTab(tab, "Guided Fit")
        layout = qt.QVBoxLayout(tab)

        info = qt.QLabel(
            "Snap seeded trajectories to the actual imaged shank. The "
            "engine collects LoG \u03c3=1 blobs inside a cylindrical ROI "
            "around each seed axis, fits the main line via "
            "amplitude-weighted PCA (wide \u2192 tight re-fit), then "
            "anchors the shallow end to the bolt CC and refines the "
            "deep end via axis-LoG. Same scanner-agnostic signal as "
            "Auto Fit; output shape is the same too (bolt tip + skull "
            "entry + deep tip)."
        )
        info.wordWrap = True
        layout.addWidget(info)

        form = qt.QFormLayout()
        layout.addLayout(form)

        # Seed source. Entries are "<Label> (<N>)" with empty sources
        # disabled so the user can only pick real seeds.
        self.guidedSeedSourceCombo = qt.QComboBox()
        self.guidedSeedSourceCombo.setToolTip(
            "Trajectory source to use as the seed. Each entry shows "
            "the current count of published trajectories for that "
            "source. Changing the selection repopulates the seed "
            "list below."
        )
        self.guidedSeedSourceCombo.currentIndexChanged.connect(
            self.onGuidedSeedSourceChanged
        )
        form.addRow("Seed source:", self.guidedSeedSourceCombo)

        self.guidedRefreshSeedsButton = qt.QPushButton("Refresh seed counts")
        self.guidedRefreshSeedsButton.clicked.connect(
            self.onGuidedRefreshSeedsClicked
        )
        form.addRow(self.guidedRefreshSeedsButton)

        # Dedicated seed table inside the tab — deliberately NOT the
        # shared trajectory table below the tabs, so the user can see
        # at a glance which trajectories the Fit buttons will act on.
        self.guidedSeedTable = qt.QTableWidget()
        self.guidedSeedTable.setColumnCount(4)
        self.guidedSeedTable.setHorizontalHeaderLabels(
            ["Fit", "Name", "Length (mm)", "Status"]
        )
        self.guidedSeedTable.horizontalHeader().setStretchLastSection(True)
        self.guidedSeedTable.verticalHeader().setVisible(False)
        self.guidedSeedTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.guidedSeedTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.guidedSeedTable.setMinimumHeight(140)
        layout.addWidget(self.guidedSeedTable)

        # Fit knobs.
        self.guidedRoiRadiusSpin = qt.QDoubleSpinBox()
        self.guidedRoiRadiusSpin.setRange(1.0, 10.0)
        self.guidedRoiRadiusSpin.setDecimals(2)
        self.guidedRoiRadiusSpin.setSingleStep(0.25)
        self.guidedRoiRadiusSpin.setValue(float(gfe.DEFAULT_ROI_RADIUS_MM))
        self.guidedRoiRadiusSpin.setSuffix(" mm")
        self.guidedRoiRadiusSpin.setToolTip(
            "Perpendicular distance from the seed axis within which "
            "LoG blobs are accepted for the initial PCA. After the "
            "rough axis is found, a tight 1.5 mm cylinder around "
            "that axis drops cross-shank contamination before the "
            "final fit."
        )
        form.addRow("ROI radius:", self.guidedRoiRadiusSpin)

        self.guidedMaxAngleSpin = qt.QDoubleSpinBox()
        self.guidedMaxAngleSpin.setRange(1.0, 30.0)
        self.guidedMaxAngleSpin.setDecimals(1)
        self.guidedMaxAngleSpin.setValue(float(gfe.DEFAULT_MAX_ANGLE_DEG))
        self.guidedMaxAngleSpin.setSuffix(" deg")
        self.guidedMaxAngleSpin.setToolTip(
            "Maximum tilt of the fitted axis vs the seed axis."
        )
        form.addRow("Max axis tilt:", self.guidedMaxAngleSpin)

        self.guidedMaxLateralShiftSpin = qt.QDoubleSpinBox()
        self.guidedMaxLateralShiftSpin.setRange(0.5, 20.0)
        self.guidedMaxLateralShiftSpin.setDecimals(2)
        self.guidedMaxLateralShiftSpin.setValue(
            float(gfe.DEFAULT_MAX_LATERAL_SHIFT_MM)
        )
        self.guidedMaxLateralShiftSpin.setSuffix(" mm")
        self.guidedMaxLateralShiftSpin.setToolTip(
            "Maximum lateral shift of the fitted midpoint vs the seed midpoint."
        )
        form.addRow("Max lateral shift:", self.guidedMaxLateralShiftSpin)

        fitRow = qt.QHBoxLayout()
        self.fitSelectedButton = qt.QPushButton("Fit Checked")
        self.fitSelectedButton.clicked.connect(self.onFitSelectedClicked)
        fitRow.addWidget(self.fitSelectedButton)
        self.fitAllButton = qt.QPushButton("Fit All")
        self.fitAllButton.clicked.connect(self.onFitAllClicked)
        fitRow.addWidget(self.fitAllButton)
        fitRow.addStretch(1)
        form.addRow(fitRow)

        self.guidedFitStatusLabel = qt.QLabel("idle")
        self.guidedFitStatusLabel.wordWrap = True
        form.addRow("Status:", self.guidedFitStatusLabel)

        # Seed cache. Only trajectories for the currently-selected seed
        # source are held here. Fit buttons iterate this list.
        self._guidedSeedTrajectories = []
        # Per-seed fit status keyed as
        # ``self._guidedSeedStatusBySource[source_key][seed_name] =
        # (ok: bool, reason: str)``. Survives table refreshes so the
        # ✓/✗ markers stay visible after a fit run; cleared only when
        # the user picks a different seed source.
        self._guidedSeedStatusBySource = {}

    # ---- Seed source combo + seed list --------------------------------

    def _refresh_guided_seed_source_combo(self):
        """Populate the Guided Fit seed combo with ``<Label> (<N>)``
        entries. Disables sources with zero trajectories so the user
        can only pick real seeds. Preserves the prior selection when
        possible.
        """
        if not hasattr(self, "guidedSeedSourceCombo"):
            return
        prev_key = ""
        try:
            prev_key = str(self.guidedSeedSourceCombo.currentData or "").strip().lower()
        except Exception:
            prev_key = ""
        self.guidedSeedSourceCombo.blockSignals(True)
        try:
            self.guidedSeedSourceCombo.clear()
            for key, label in _GUIDED_SEED_SOURCES:
                try:
                    count = int(self.logic.count_trajectories_by_source(
                        key, workflow_node=self.workflowNode,
                    ))
                except Exception:
                    count = 0
                self.guidedSeedSourceCombo.addItem(f"{label} ({count})", key)
                if count == 0:
                    idx = self.guidedSeedSourceCombo.count - 1
                    model_item = self.guidedSeedSourceCombo.model().item(idx)
                    if model_item is not None:
                        model_item.setEnabled(False)
            restored = False
            if prev_key:
                idx = self.guidedSeedSourceCombo.findData(prev_key)
                if idx >= 0:
                    self.guidedSeedSourceCombo.setCurrentIndex(idx)
                    restored = True
            if not restored:
                for idx in range(self.guidedSeedSourceCombo.count):
                    model_item = self.guidedSeedSourceCombo.model().item(idx)
                    if model_item is None or model_item.isEnabled():
                        self.guidedSeedSourceCombo.setCurrentIndex(idx)
                        break
        finally:
            self.guidedSeedSourceCombo.blockSignals(False)
        self._refresh_guided_seed_table()

    def _selected_guided_seed_source(self):
        data = self.guidedSeedSourceCombo.currentData
        return str(data or "").strip().lower()

    def _refresh_guided_seed_table(self):
        """Repopulate the seed table from whichever source is picked
        in the Guided Fit seed combo.
        """
        key = self._selected_guided_seed_source()
        try:
            seeds = list(self.logic.collect_trajectories_by_source(
                key, workflow_node=self.workflowNode,
            ))
        except Exception:
            seeds = []
        self._guidedSeedTrajectories = seeds

        self.guidedSeedTable.setRowCount(0)
        for row, traj in enumerate(seeds):
            self.guidedSeedTable.insertRow(row)
            use_check = qt.QCheckBox()
            use_check.setChecked(True)
            use_check.setStyleSheet("margin-left:8px; margin-right:8px;")
            self.guidedSeedTable.setCellWidget(row, 0, use_check)

            name_item = qt.QTableWidgetItem(str(traj.get("name", "")))
            self.guidedSeedTable.setItem(row, 1, name_item)

            length_item = qt.QTableWidgetItem(f"{trajectory_length_mm(traj):.2f}")
            self.guidedSeedTable.setItem(row, 2, length_item)

            status_item = qt.QTableWidgetItem("—")
            status_item.setTextAlignment(qt.Qt.AlignCenter)
            self.guidedSeedTable.setItem(row, 3, status_item)

        # Re-apply any remembered statuses from a prior fit of this
        # seed source so the ✓ / ✗ markers survive a table rebuild.
        stash = self._guidedSeedStatusBySource.get(key) or {}
        if stash:
            for row, traj in enumerate(seeds):
                entry = stash.get(str(traj.get("name", "")))
                if entry is None:
                    continue
                ok, reason = entry
                self._paint_status_cell(row, ok, reason)

    def onGuidedSeedSourceChanged(self, _idx):
        # Switching to a different seed source invalidates the prior
        # fit statuses we might have shown for the previous source;
        # clear them so stale markers don't bleed across sources.
        current = self._selected_guided_seed_source()
        if current:
            for key in list(self._guidedSeedStatusBySource.keys()):
                if key != current:
                    self._guidedSeedStatusBySource.pop(key, None)
        self._refresh_guided_seed_table()

    def onGuidedRefreshSeedsClicked(self):
        self._refresh_guided_seed_source_combo()

    def _checked_guided_seed_indices(self):
        out = []
        for row in range(int(self.guidedSeedTable.rowCount)):
            cell = self.guidedSeedTable.cellWidget(row, 0)
            if cell is not None and bool(cell.isChecked()):
                out.append(row)
        return out

    # ``_populate_guided_trajectory_table`` is still called by gui.py's
    # ``onRefreshClicked`` for the shared trajectory table below the
    # tabs (it collects from ``self.loadedTrajectories``). Left as a
    # stub here because the shared table lives on the widget mixin
    # class and is reused across tabs; Guided Fit no longer depends on
    # it.
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

                # Confidence column. Auto-Fit emissions carry numeric
                # ``confidence`` + a discrete ``confidence_label``;
                # manual / imported trajectories show "—".
                conf = traj.get("confidence")
                conf_label = str(traj.get("confidence_label") or "").strip().lower()
                if conf is None:
                    conf_text = "—"
                else:
                    band = conf_label or "—"
                    conf_text = f"{float(conf):.2f}  ({band})"
                conf_item = qt.QTableWidgetItem(conf_text)
                conf_item.setFlags(conf_item.flags() & ~qt.Qt.ItemIsEditable)
                # Color-code the band so users can spot weak emissions
                # at a glance without reading the score.
                if conf_label == "high":
                    conf_item.setForeground(qt.QBrush(qt.QColor("#1e7d32")))
                elif conf_label == "medium":
                    conf_item.setForeground(qt.QBrush(qt.QColor("#e8a000")))
                elif conf_label == "low":
                    conf_item.setForeground(qt.QBrush(qt.QColor("#c62828")))
                self.guidedTrajectoryTable.setItem(row, 3, conf_item)

            if self.guidedTrajectoryTable.rowCount > 0:
                self.guidedTrajectoryTable.selectRow(0)
            # Apply any active confidence filter (hides rows below band).
            self._apply_confidence_filter()
        finally:
            self._updatingGuidedTable = False

    def _apply_confidence_filter(self):
        """Hide rows whose trajectory's confidence falls below the
        threshold chosen in the Confidence-filter combo. Manual /
        imported trajectories (no confidence set) always show — the
        filter only narrows Auto-Fit results.
        """
        combo = getattr(self, "confidenceFilterCombo", None)
        if combo is None:
            return
        key = combo.itemData(combo.currentIndex)
        if key in (None, "all"):
            min_score = float("-inf")
        elif key == "high":
            min_score = 0.80
        elif key == "medium":
            min_score = 0.50
        else:  # "low" treated as "any score >= 0"
            min_score = 0.0
        for row, traj in enumerate(self.loadedTrajectories):
            conf = traj.get("confidence")
            if conf is None:
                hidden = False  # untagged trajectories always visible
            else:
                hidden = float(conf) < min_score
            self.guidedTrajectoryTable.setRowHidden(row, hidden)

    def onConfidenceFilterChanged(self, _idx):
        self._apply_confidence_filter()

    def onGuidedTrajectoryItemChanged(self, item):
        if self._updatingGuidedTable or self._renamingGuidedTrajectory:
            return
        if item is None or item.column() != 1:
            return
        row = int(item.row())
        if row < 0 or row >= len(self.loadedTrajectories):
            return
        new_name = str(item.text() or "").strip()
        traj = self.loadedTrajectories[row]
        old_name = str(traj.get("name", "")).strip()
        if not new_name or new_name == old_name:
            return
        self._renamingGuidedTrajectory = True
        try:
            ok = bool(self.logic.rename_trajectory(traj.get("node_id", ""), new_name))
        finally:
            self._renamingGuidedTrajectory = False
        if ok:
            traj["name"] = new_name
            item.setData(qt.Qt.UserRole, new_name)
            self.log(f"[guided] renamed {old_name} → {new_name}")
            self.onRefreshClicked()
        else:
            self._updatingGuidedTable = True
            try:
                item.setText(old_name)
            finally:
                self._updatingGuidedTable = False

    def _checked_guided_trajectory_names(self):
        """Kept for compatibility with the shared ``Remove checked``
        action, which still reads the shared trajectory table.
        """
        names = []
        for row in range(int(self.guidedTrajectoryTable.rowCount)):
            cell = self.guidedTrajectoryTable.cellWidget(row, 0)
            if cell is not None and bool(cell.isChecked()):
                item = self.guidedTrajectoryTable.item(row, 1)
                names.append(str(item.text() if item else "").strip())
        return [n for n in names if n]

    def onRemoveCheckedClicked(self):
        names = self._checked_guided_trajectory_names()
        if not names:
            qt.QMessageBox.information(
                slicer.util.mainWindow(),
                "Postop CT Localization",
                "Check at least one trajectory to remove.",
            )
            return
        source_key = self._selected_guided_source_key()
        removed = self.logic.remove_trajectories_by_name(names, source_key=source_key)
        self.log(f"[guided] removed {removed} trajectories from source '{source_key}'")
        self.onRefreshClicked()

    # ---- Fitting ------------------------------------------------------

    def _guided_fit_volume_matrices(self, volume_node):
        import SimpleITK as sitk
        arr_kji = np.asarray(
            slicer.util.arrayFromVolume(volume_node), dtype=np.float32
        )
        img = sitk.GetImageFromArray(arr_kji)
        img.SetSpacing(tuple(float(v) for v in volume_node.GetSpacing()))

        try:
            import vtk
        except ImportError:
            from __main__ import vtk
        m = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(m)
        ijk_to_ras = np.array([
            [float(m.GetElement(r, c)) for c in range(4)] for r in range(4)
        ], dtype=float)
        m2 = vtk.vtkMatrix4x4()
        volume_node.GetRASToIJKMatrix(m2)
        ras_to_ijk = np.array([
            [float(m2.GetElement(r, c)) for c in range(4)] for r in range(4)
        ], dtype=float)
        return img, ijk_to_ras, ras_to_ijk

    @staticmethod
    def _seed_start_end_ras(traj):
        """``trajectory_from_line_node`` stores ``start``/``end`` in
        LPS for ROSA interop. Convert to RAS for the LoG engine.
        """
        start = traj.get("start") or [0.0, 0.0, 0.0]
        end = traj.get("end") or [0.0, 0.0, 0.0]
        return (
            np.asarray(lps_to_ras_point(start), dtype=float),
            np.asarray(lps_to_ras_point(end), dtype=float),
        )

    def _paint_status_cell(self, seed_index, ok, reason=""):
        """Draw the ✓ / ✗ into the Status cell of one seed row."""
        if seed_index < 0 or seed_index >= int(self.guidedSeedTable.rowCount):
            return
        item = self.guidedSeedTable.item(seed_index, 3)
        if item is None:
            item = qt.QTableWidgetItem()
            self.guidedSeedTable.setItem(seed_index, 3, item)
        item.setTextAlignment(qt.Qt.AlignCenter)
        if ok:
            item.setText("\u2713")
            item.setForeground(qt.QBrush(qt.QColor(40, 160, 60)))
            item.setToolTip("Fitted")
        else:
            item.setText("\u2717")
            item.setForeground(qt.QBrush(qt.QColor(200, 50, 50)))
            item.setToolTip(str(reason or "fit rejected"))

    def _set_seed_status(self, seed_index, ok, reason=""):
        """Update the Status cell AND remember the outcome so it
        survives table refreshes until the seed source changes.
        """
        self._paint_status_cell(seed_index, ok, reason)
        key = self._selected_guided_seed_source()
        if not key or seed_index < 0:
            return
        if seed_index >= len(self._guidedSeedTrajectories):
            return
        name = str(self._guidedSeedTrajectories[seed_index].get("name", ""))
        if not name:
            return
        stash = self._guidedSeedStatusBySource.setdefault(key, {})
        stash[name] = (bool(ok), str(reason or ""))

    def _fit_seeds(self, seed_rows):
        """``seed_rows`` is a list of (row_index, trajectory) tuples so
        we can write results back to the table in the matching cells.
        """
        volume_node = self.ctSelector.currentNode()
        if volume_node is None:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "Postop CT Localization",
                "Select a CT volume.",
            )
            return
        if not seed_rows:
            self.log("[guided] nothing to fit (check at least one seed)")
            return

        seed_source_key = self._selected_guided_seed_source()

        self.guidedFitStatusLabel.setText("preprocessing LoG / hull / bolts …")
        try:
            slicer.app.processEvents()
        except Exception:
            pass

        try:
            img, ijk_to_ras, ras_to_ijk = self._guided_fit_volume_matrices(volume_node)
            features = gfe.compute_features(img, ijk_to_ras)
        except Exception as exc:
            self.log(f"[guided] preprocessing failed: {exc}")
            qt.QMessageBox.critical(
                slicer.util.mainWindow(), "Postop CT Localization", str(exc),
            )
            self.guidedFitStatusLabel.setText(f"error: {exc}")
            return

        roi_mm = float(self.guidedRoiRadiusSpin.value)
        max_angle = float(self.guidedMaxAngleSpin.value)
        max_lat = float(self.guidedMaxLateralShiftSpin.value)

        self.logic.register_postop_ct(volume_node, workflow_node=self.workflowNode)
        self.logic.trajectory_scene.remove_preview_lines(node_prefix="Guided_")

        # Reset remembered statuses for this seed source so a fresh fit
        # run paints an accurate per-row picture (no stale ✓/✗).
        if seed_source_key:
            self._guidedSeedStatusBySource.pop(seed_source_key, None)

        # Two-phase: first collect fit records, then run crossing-tip
        # retreat across all records, then emit the scene nodes with
        # the post-retreat geometry. Mirrors Auto Fit's flow.
        fit_records = []
        missed_names = []
        for row, traj in seed_rows:
            name = str(traj.get("name", "")) or "?"
            seed_start, seed_end = self._seed_start_end_ras(traj)
            try:
                fit = gfe.fit_trajectory(
                    planned_start_ras=seed_start,
                    planned_end_ras=seed_end,
                    features=features,
                    ijk_to_ras_mat=ijk_to_ras,
                    ras_to_ijk_mat=ras_to_ijk,
                    roi_radius_mm=roi_mm,
                    max_angle_deg=max_angle,
                    max_lateral_shift_mm=max_lat,
                )
            except Exception as exc:
                self.log(f"[guided] {name}: fit crashed ({exc})")
                self._set_seed_status(row, False, f"crash: {exc}")
                missed_names.append(name)
                continue

            if not bool(fit.get("success")):
                reason = str(fit.get("reason", "unknown"))
                self.log(f"[guided] {name}: failed ({reason})")
                self._set_seed_status(row, False, reason)
                missed_names.append(name)
                continue

            if "skull_entry_ras" in fit:
                line_start = np.asarray(fit["skull_entry_ras"], dtype=float)
            else:
                line_start = np.asarray(fit["start_ras"], dtype=float)
            line_end = np.asarray(fit["end_ras"], dtype=float)

            fit_records.append({
                "name": name,
                "seed_row": row,
                "fit": fit,
                # start_ras / end_ras are the keys _retreat_crossing_tips
                # reads and writes. Using skull_entry as the shallow end
                # and deep tip as the deep end keeps the line colinear
                # with the intracranial segment.
                "start_ras": line_start,
                "end_ras": line_end,
            })

        # Crossing-tip retreat across all fitted records. Walks any
        # deep tip that sits within 2 mm of another fitted segment
        # back along its own axis until clearance is restored AND the
        # retreated tip sits on a real contact peak, matching Auto
        # Fit's post-refinement retreat. Without this pass, two
        # crossing seeds fit independently can have their tips land
        # inside each other's contact tubes.
        if fit_records:
            try:
                cpfit._retreat_crossing_tips(
                    fit_records,
                    log_arr=features["log"],
                    ras_to_ijk_mat=ras_to_ijk,
                    logger=self.log,
                )
            except Exception as exc:
                self.log(f"[guided] crossing-tip retreat failed: {exc}")

        applied_nodes = []
        for rec in fit_records:
            name = rec["name"]
            row = rec["seed_row"]
            fit = rec["fit"]
            line_start = rec["start_ras"]
            line_end = rec["end_ras"]

            existing = self.logic.trajectory_scene.find_line_by_group_and_name(
                name, "guided_fit",
            )
            node = self.logic.trajectory_scene.create_or_update_trajectory_line(
                name=name,
                start_ras=[float(v) for v in line_start],
                end_ras=[float(v) for v in line_end],
                node_id=None if existing is None else existing.GetID(),
                node_name=f"Guided_{name}",
                group="guided_fit",
                origin="postop_ct_guided_fit",
            )
            applied_nodes.append(node)
            self._set_seed_status(row, True)
            anchored = "bolt" if fit.get("bolt_anchored") else "no-bolt"
            self.log(
                "[guided] {n}: angle={a:.2f}°  lat={l:.2f} mm  "
                "n={ni}  len={L:.1f} mm  {anc}".format(
                    n=name,
                    a=float(fit.get("angle_deg", 0.0)),
                    l=float(fit.get("lateral_shift_mm", 0.0)),
                    ni=int(fit.get("n_inliers", 0)),
                    L=float(np.linalg.norm(line_end - line_start)),
                    anc=anchored,
                )
            )

        fitted = len(applied_nodes)
        total = len(seed_rows)

        if applied_nodes:
            self.workflowPublisher.publish_nodes(
                role="GuidedFitTrajectoryLines",
                nodes=applied_nodes,
                source="postop_ct_guided_fit",
                space_name="ROSA_BASE",
                workflow_node=self.workflowNode,
            )
            # Merge (don't replace) into WorkingTrajectoryLines so the
            # seed trajectories that fed Guided Fit stay visible in the
            # workflow. ``publish_nodes`` replaces the role's node list,
            # which silently dropped the seeds after a Guided Fit run.
            existing_working = self.workflowPublisher.state.role_nodes(
                "WorkingTrajectoryLines", workflow_node=self.workflowNode,
            )
            seen_ids = set()
            merged = []
            for n in list(existing_working) + list(applied_nodes):
                if n is None:
                    continue
                nid = n.GetID()
                if nid in seen_ids:
                    continue
                seen_ids.add(nid)
                merged.append(n)
            self.workflowPublisher.publish_nodes(
                role="WorkingTrajectoryLines",
                nodes=merged,
                source="postop_ct_guided_fit",
                space_name="ROSA_BASE",
                workflow_node=self.workflowNode,
            )
            self.logic.trajectory_scene.place_trajectory_nodes_in_hierarchy(
                context_id=self.workflowState.context_id(
                    workflow_node=self.workflowNode
                ),
                nodes=applied_nodes,
            )
            # When some seeds missed, show BOTH the fitted group and
            # the original seed-source group so the user can see at
            # a glance which planned lines didn't get snapped (they
            # render in the seed source's color; fits render cyan).
            if missed_names and seed_source_key:
                groups = ["guided_fit", seed_source_key]
            else:
                groups = ["guided_fit"]
            self.logic.trajectory_scene.show_only_groups(groups)
            self._set_workflow_active_source("guided_fit")
            self._set_guided_source_combo("guided_fit")
            self._refresh_guided_seed_source_combo()
            self.log(f"[guided] applied {fitted} trajectories")
            self.onRefreshClicked()

        if missed_names:
            preview = ", ".join(missed_names[:6])
            if len(missed_names) > 6:
                preview += f", …(+{len(missed_names) - 6})"
            self.guidedFitStatusLabel.setText(
                f"done — fitted {fitted}/{total}; missed: {preview}"
            )
            self.log(f"[guided] missed {len(missed_names)}/{total}: {preview}")
        else:
            self.guidedFitStatusLabel.setText(
                f"done — fitted {fitted}/{total} seeds"
            )

    def onFitSelectedClicked(self):
        rows = self._checked_guided_seed_indices()
        if not rows:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "Postop CT Localization",
                "Check at least one seed trajectory.",
            )
            return
        seed_rows = [(r, self._guidedSeedTrajectories[r]) for r in rows]
        self._fit_seeds(seed_rows)

    def onFitAllClicked(self):
        seed_rows = list(enumerate(self._guidedSeedTrajectories))
        if not seed_rows:
            qt.QMessageBox.information(
                slicer.util.mainWindow(),
                "Postop CT Localization",
                "No seed trajectories. Pick a source with a non-zero count.",
            )
            return
        self._fit_seeds(seed_rows)
