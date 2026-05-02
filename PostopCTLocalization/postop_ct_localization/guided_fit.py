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

from rosa_detect import contact_pitch_v1_fit as cpfit
from rosa_detect import guided_fit_engine as gfe


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
        info.setMinimumWidth(0)
        info.setSizePolicy(
            qt.QSizePolicy.Preferred, qt.QSizePolicy.MinimumExpanding,
        )
        layout.addWidget(info)

        form = qt.QFormLayout()
        form.setFieldGrowthPolicy(qt.QFormLayout.AllNonFixedFieldsGrow)
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
            ["Fit", "Name", "Len mm", "Status"]
        )
        seed_header = self.guidedSeedTable.horizontalHeader()
        for col in range(self.guidedSeedTable.columnCount):
            seed_header.setSectionResizeMode(col, qt.QHeaderView.Interactive)
        for col, width in (
            (0, 40),    # Fit
            (1, 110),   # Name
            (2, 70),    # Len mm
            (3, 90),    # Status
        ):
            self.guidedSeedTable.setColumnWidth(col, width)
        seed_header.setStretchLastSection(False)
        self.guidedSeedTable.verticalHeader().setVisible(False)
        self.guidedSeedTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.guidedSeedTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.guidedSeedTable.setMinimumHeight(140)
        self.guidedSeedTable.setMinimumWidth(120)
        self.guidedSeedTable.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAsNeeded)
        self.guidedSeedTable.setSizePolicy(
            qt.QSizePolicy.Expanding, qt.QSizePolicy.Preferred,
        )
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
        self.guidedRoiRadiusSpin.setMinimumWidth(0)
        self.guidedRoiRadiusSpin.setSizePolicy(
            qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed,
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
        self.guidedMaxAngleSpin.setMinimumWidth(0)
        self.guidedMaxAngleSpin.setSizePolicy(
            qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed,
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
        self.guidedMaxLateralShiftSpin.setMinimumWidth(0)
        self.guidedMaxLateralShiftSpin.setSizePolicy(
            qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed,
        )
        form.addRow("Max lateral shift:", self.guidedMaxLateralShiftSpin)

        # Local pitch-strategy combo so the user can restrict the
        # picker library directly from Guided Fit (used to require
        # flipping to the Auto Fit tab to change it).
        from rosa_core.electrode_classifier import PITCH_STRATEGY_OPTIONS
        self.guidedPitchStrategyCombo = qt.QComboBox()
        for label, key in PITCH_STRATEGY_OPTIONS:
            self.guidedPitchStrategyCombo.addItem(label, key)
        self.guidedPitchStrategyCombo.setCurrentIndex(0)  # default: Dixi AM
        self.guidedPitchStrategyCombo.setToolTip(
            "Restrict the model library used by the per-trajectory "
            "PaCER picker on Fit. Vendor + pitch-set filter."
        )
        self.guidedPitchStrategyCombo.setMinimumContentsLength(8)
        self.guidedPitchStrategyCombo.setSizeAdjustPolicy(
            qt.QComboBox.AdjustToMinimumContentsLengthWithIcon,
        )
        self.guidedPitchStrategyCombo.setSizePolicy(
            qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed,
        )
        form.addRow("Pitch strategy:", self.guidedPitchStrategyCombo)

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
        self.guidedFitStatusLabel.setMinimumWidth(0)
        self.guidedFitStatusLabel.setSizePolicy(
            qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred,
        )
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
                # Default UNCHECKED. The "Remove Marked Trajectories"
                # button only acts on checked rows, so a stray click
                # on a fresh table is harmless. Users mark rows
                # explicitly via the Mark Selected / Mark All buttons.
                use_check.setChecked(False)
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

    # ---- Bulk-mark actions for the trajectory table ------------------

    def _selected_table_rows(self):
        """Indices of rows currently selected in the trajectory table.
        Uses the selection model so Ctrl/Shift-click multi-selection
        works correctly. Falls back to currentRow() when nothing is
        selected.
        """
        sel_model = self.guidedTrajectoryTable.selectionModel()
        rows = set()
        if sel_model is not None:
            for idx in sel_model.selectedRows():
                rows.add(int(idx.row()))
        if not rows:
            cur = int(self.guidedTrajectoryTable.currentRow())
            if cur >= 0:
                rows.add(cur)
        return sorted(rows)

    def _set_marked(self, rows, marked):
        """Set the Mark checkbox state on the given table rows. Skips
        hidden rows (so confidence-filtered ones aren't accidentally
        marked by Mark All).
        """
        for row in rows:
            if self.guidedTrajectoryTable.isRowHidden(row):
                continue
            cell = self.guidedTrajectoryTable.cellWidget(row, 0)
            if cell is not None:
                cell.setChecked(bool(marked))

    def onMarkSelectedClicked(self):
        rows = self._selected_table_rows()
        if not rows:
            return
        self._set_marked(rows, True)

    def onMarkAllClicked(self):
        all_rows = list(range(int(self.guidedTrajectoryTable.rowCount)))
        self._set_marked(all_rows, True)

    def onUnmarkAllClicked(self):
        # Force-clear ALL rows (including hidden ones) so a confidence
        # filter change doesn't leak previously-marked rows back into
        # the next "Remove Marked" action.
        for row in range(int(self.guidedTrajectoryTable.rowCount)):
            cell = self.guidedTrajectoryTable.cellWidget(row, 0)
            if cell is not None:
                cell.setChecked(False)

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

    @staticmethod
    def _parse_baseline_ras_attr(text):
        """Decode an `x,y,z` MRML attr written by deep_core_visualization
        at Auto Fit publish time. Returns ``None`` if the attribute is
        missing, malformed, or holds fewer than three coordinates.
        """
        if not text:
            return None
        try:
            parts = [float(v) for v in str(text).split(",")]
        except Exception:
            return None
        if len(parts) < 3:
            return None
        return parts[:3]

    def onRevertToAutoFitClicked(self):
        rows = self._selected_table_rows()
        if not rows:
            qt.QMessageBox.information(
                slicer.util.mainWindow(),
                "Postop CT Localization",
                "Select at least one trajectory row to revert.",
            )
            return
        reverted = 0
        skipped = 0
        for row in rows:
            if row < 0 or row >= len(self.loadedTrajectories):
                skipped += 1
                continue
            traj = self.loadedTrajectories[row]
            node_id = str(traj.get("node_id") or "")
            if not node_id:
                skipped += 1
                continue
            node = slicer.mrmlScene.GetNodeByID(node_id)
            if node is None or node.GetNumberOfControlPoints() < 2:
                skipped += 1
                continue
            start_ras = self._parse_baseline_ras_attr(
                node.GetAttribute("Rosa.AutoFitStartRas")
            )
            end_ras = self._parse_baseline_ras_attr(
                node.GetAttribute("Rosa.AutoFitEndRas")
            )
            if start_ras is None or end_ras is None:
                skipped += 1
                continue
            node.SetNthControlPointPosition(
                0, float(start_ras[0]), float(start_ras[1]), float(start_ras[2])
            )
            node.SetNthControlPointPosition(
                1, float(end_ras[0]), float(end_ras[1]), float(end_ras[2])
            )
            reverted += 1
        self.log(
            f"[auto-fit] reverted {reverted} trajectory/ies to baseline "
            f"({skipped} skipped — no Rosa.AutoFit*Ras stamp)"
        )
        self.onRefreshClicked()

    # ---- Fitting ------------------------------------------------------

    def _guided_fit_volume_matrices(self, volume_node):
        # Single source of truth for the LPS-flip + matrix bundling
        # lives in the Slicer-side adapter (rosa_scene). The detection
        # algorithm package (rosa_detect) is boundary-clean of vtk /
        # slicer; it consumes plain (img, i2r, r2i) tuples.
        from rosa_scene.sitk_volume_adapter import image_from_volume_node

        return image_from_volume_node(volume_node)

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
            features = gfe.compute_features(img, ijk_to_ras, ras_to_ijk)
            # ``compute_features`` may have resampled the volume to the
            # canonical grid; pull the canonical img + matrices back so
            # subsequent ``fit_trajectory`` calls operate on the same
            # grid the LoG / Frangi kernels were computed on.
            img = features["img"]
            ijk_to_ras = features["ijk_to_ras_mat"]
            ras_to_ijk = features["ras_to_ijk_mat"]
        except Exception as exc:
            self.log(f"[guided] preprocessing failed: {exc}")
            qt.QMessageBox.critical(
                slicer.util.mainWindow(), "Postop CT Localization", str(exc),
            )
            self.guidedFitStatusLabel.setText(f"error: {exc}")
            return

        # Diagnostic: surface the blob-cloud size + RAS extent so a
        # systematic frame mismatch (planned trajectories in a
        # different frame than the postop CT) is visible at a glance.
        # When every seed reports "0 < N" blobs in ROI, this line
        # tells whether the engine extracted ANY blobs (LoG bad) or
        # extracted plenty but they're nowhere near the seeds (frame
        # mismatch).
        try:
            _pts = features.get("blob_pts_ras")
            _vol_name = volume_node.GetName() if volume_node is not None else "?"
            if _pts is not None and len(_pts) > 0:
                _arr = np.asarray(_pts, dtype=float)
                _mn = _arr.min(axis=0)
                _mx = _arr.max(axis=0)
                self.log(
                    f"[guided] LoG blob cloud: n={int(_arr.shape[0])} "
                    f"on '{_vol_name}'; RAS bbox "
                    f"x=[{_mn[0]:+.1f},{_mx[0]:+.1f}] "
                    f"y=[{_mn[1]:+.1f},{_mx[1]:+.1f}] "
                    f"z=[{_mn[2]:+.1f},{_mx[2]:+.1f}]"
                )
            else:
                self.log(
                    f"[guided] LoG blob cloud is EMPTY on '{_vol_name}' "
                    "— no metal-bright voxels in the postop CT (wrong "
                    "volume? CT not loaded? HU rescaled?)."
                )
        except Exception as exc:
            self.log(f"[guided] blob-cloud diagnostic failed: {exc}")

        roi_mm = float(self.guidedRoiRadiusSpin.value)
        max_angle = float(self.guidedMaxAngleSpin.value)
        max_lat = float(self.guidedMaxLateralShiftSpin.value)

        self.logic.register_postop_ct(volume_node, workflow_node=self.workflowNode)
        self.logic.trajectory_scene.remove_preview_lines(node_prefix="Guided_")

        # Reset remembered statuses for this seed source so a fresh fit
        # run paints an accurate per-row picture (no stale ✓/✗).
        if seed_source_key:
            self._guidedSeedStatusBySource.pop(seed_source_key, None)

        # Phase 2 (match-against-auto): if Auto Fit has already published
        # trajectories in the workflow, prefer matching seeds to those —
        # the user gets walker-validated, post-anchor-Frangi-gated,
        # wire-class-aware results inherited verbatim. Falls back to the
        # PCA fit (Phase 1) when no auto trajectory falls within the
        # seed's tolerance window, or when Auto Fit hasn't been run yet.
        try:
            auto_trajs = self.logic.collect_trajectories_by_source(
                "auto_fit", workflow_node=self.workflowNode,
            )
        except Exception:
            auto_trajs = []
        if auto_trajs:
            self.log(
                f"[guided] match-against-auto enabled — {len(auto_trajs)} "
                "Auto Fit trajectories available"
            )
        else:
            self.log(
                "[guided] no Auto Fit trajectories in workflow; "
                "running PCA fit only (tip: run Auto Fit first to "
                "inherit walker + score)"
            )

        # Diagnostic: log the seed-cloud RAS bbox so a frame mismatch
        # vs. the blob cloud's bbox is immediately visible. If seeds
        # are systematically offset from the LoG cloud, every fit
        # will report "0 blobs in ROI" — but the user can see the
        # offset here without inspecting individual rows.
        try:
            _seed_centers = np.asarray([
                0.5 * (np.asarray(self._seed_start_end_ras(t)[0], dtype=float)
                       + np.asarray(self._seed_start_end_ras(t)[1], dtype=float))
                for _r, t in seed_rows
            ], dtype=float)
            if _seed_centers.shape[0] > 0:
                _smn = _seed_centers.min(axis=0)
                _smx = _seed_centers.max(axis=0)
                self.log(
                    f"[guided] seed midpoint cloud: n={_seed_centers.shape[0]} "
                    f"RAS bbox "
                    f"x=[{_smn[0]:+.1f},{_smx[0]:+.1f}] "
                    f"y=[{_smn[1]:+.1f},{_smx[1]:+.1f}] "
                    f"z=[{_smn[2]:+.1f},{_smx[2]:+.1f}]"
                )
        except Exception as exc:
            self.log(f"[guided] seed-cloud diagnostic failed: {exc}")

        # Two-phase: first collect fit records, then run crossing-tip
        # retreat across all records, then emit the scene nodes with
        # the post-retreat geometry. Mirrors Auto Fit's flow.
        fit_records = []
        missed_names = []
        for row, traj in seed_rows:
            name = str(traj.get("name", "")) or "?"
            seed_start, seed_end = self._seed_start_end_ras(traj)
            fit = None
            # Try matching against Auto Fit first.
            if auto_trajs:
                try:
                    fit = gfe.match_seed_to_auto_traj(
                        planned_start_ras=seed_start,
                        planned_end_ras=seed_end,
                        auto_trajs=auto_trajs,
                        max_angle_deg=max_angle,
                        max_lateral_shift_mm=max_lat,
                    )
                except Exception as exc:
                    self.log(f"[guided] {name}: match-auto crashed ({exc})")
                    fit = None
            # Fall back to PCA fit when no auto match.
            if fit is None:
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

            for warn in fit.get("warnings") or ():
                self.log(f"[guided] {name}: warning — {warn}")

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
            # Inherit the postop CT's parent transform so the line
            # displays in the same world frame the CT is shown in. The
            # fit RAS coords are already in the volume node's local
            # IJK→RAS frame; if the volume has a separate parent
            # transform, the line needs the same transform applied
            # at display time. (Auto Fit does this via
            # ``_copy_parent_transform`` in deep_core_visualization;
            # Guided Fit was missing the equivalent.)
            try:
                if volume_node is not None and node is not None:
                    node.SetAndObserveTransformNodeID(
                        volume_node.GetTransformNodeID()
                    )
            except Exception:
                pass
            # Stamp Auto-Fit-equivalent confidence + bolt-source attrs so
            # the Slicer confidence filter combo and Trajectory Set table
            # treat guided-fit lines the same as auto-fit lines.
            try:
                conf_val = fit.get("confidence")
                if conf_val is not None:
                    node.SetAttribute("Rosa.Confidence", f"{float(conf_val):.3f}")
                conf_label = fit.get("confidence_label")
                if conf_label:
                    node.SetAttribute("Rosa.ConfidenceLabel", str(conf_label))
                bolt_src = fit.get("bolt_source")
                if bolt_src:
                    node.SetAttribute("Rosa.BoltSource", str(bolt_src))
                # Phase 2 audit trail: when a guided line was inherited
                # from an auto-fit detection, record which auto traj it
                # came from. Useful when the user sees identical
                # AutoFit_X and Guided_X lines and wonders why.
                if bool(fit.get("matched_auto_source")):
                    node.SetAttribute("Rosa.MatchedAutoSource", "1")
                    matched_name = str(fit.get("matched_auto_name") or "")
                    if matched_name:
                        node.SetAttribute("Rosa.MatchedAutoName", matched_name)
            except Exception:
                pass
            # Unified electrode-model picker (same code path as Auto Fit
            # and Manual Fit). PaCER template-correlation against the
            # canonical CT volume is the preferred mode; walker signature
            # / length-only are fallbacks. Phase 2 match-against-auto
            # already inherits the matched auto trajectory's attributes
            # via the auto-source path; only PCA fallback needs to pick
            # here.
            try:
                from rosa_core.electrode_classifier import classify_electrode_model
                walker_sig = None
                n_obs = int(fit.get("n_inliers") or 0)
                pitch_obs = float(fit.get("original_median_pitch_mm") or 0.0)
                span_obs = float(fit.get("contact_span_mm") or 0.0)
                if n_obs > 0 and pitch_obs > 0.0:
                    walker_sig = (n_obs, pitch_obs, span_obs)
                strategy = "auto"
                combo = getattr(self, "guidedPitchStrategyCombo", None)
                if combo is not None:
                    data = combo.currentData
                    if isinstance(data, str) and data:
                        strategy = data
                else:
                    fallback = getattr(self, "_selected_contact_pitch_strategy", None)
                    if callable(fallback):
                        strategy = str(fallback() or "auto")
                pick = classify_electrode_model(
                    start_ras=line_start, end_ras=line_end,
                    pitch_strategy=strategy,
                    ct_volume_kji=features.get("ct_arr_kji"),
                    ras_to_ijk_mat=ras_to_ijk,
                    walker_signature=walker_sig,
                )
                if pick is not None:
                    node.SetAttribute("Rosa.BestModelId", str(pick.get("model_id") or ""))
                    method = str(pick.get("method") or "")
                    if method:
                        node.SetAttribute("Rosa.SuggestedElectrodeMethod", method)
            except Exception as exc:
                self.log(f"[guided] {name}: model-pick warning — {exc}")
            applied_nodes.append(node)
            self._set_seed_status(row, True)
            anchored = "bolt" if fit.get("bolt_anchored") else "no-bolt"
            source_tag = "auto-match" if fit.get("matched_auto_source") else "pca-fit"
            self.log(
                "[guided] {n}: {src}  angle={a:.2f}°  lat={l:.2f} mm  "
                "n={ni}  len={L:.1f} mm  {anc}".format(
                    n=name,
                    src=source_tag,
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
