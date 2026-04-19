"""Guided Fit widget: snap planned/seeded trajectories to the imaged
shank using the LoG-based engine in :mod:`guided_fit_engine`.

Seed sources come from the workflow: Auto Fit, Imported ROSA,
Imported External, Manual, Planned ROSA. A dropdown at the top of the
tab lists each source with its current trajectory count so the user
knows what's available before fitting.
"""
try:
    import numpy as np
except ImportError:
    np = None

from __main__ import qt, slicer

from rosa_core import trajectory_length_mm

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
        form = qt.QFormLayout(tab)

        guided_info = qt.QLabel(
            "Snap seeded trajectories to the actual imaged shank using "
            "LoG \u03c3=1 blobs inside a cylindrical ROI around each seed "
            "axis. Same scanner-agnostic signal as Auto Fit."
        )
        guided_info.wordWrap = True
        form.addRow(guided_info)

        # Seed-source dropdown. Entries are "<Label> (<N>)" and the
        # current ``ActiveTrajectorySource`` drives the shared table
        # below the tabs, so picking a source here also updates what
        # the user sees checked.
        self.guidedSeedSourceCombo = qt.QComboBox()
        self.guidedSeedSourceCombo.setToolTip(
            "Trajectory source to use as the seed for guided fit. "
            "Each entry shows the number of trajectories currently "
            "published under that source."
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

        # Fit knobs. Defaults match the engine's defaults.
        self.guidedRoiRadiusSpin = qt.QDoubleSpinBox()
        self.guidedRoiRadiusSpin.setRange(1.0, 6.0)
        self.guidedRoiRadiusSpin.setDecimals(2)
        self.guidedRoiRadiusSpin.setSingleStep(0.25)
        self.guidedRoiRadiusSpin.setValue(float(gfe.DEFAULT_ROI_RADIUS_MM))
        self.guidedRoiRadiusSpin.setSuffix(" mm")
        self.guidedRoiRadiusSpin.setToolTip(
            "Perpendicular distance from the seed axis within which "
            "LoG blobs are accepted for the PCA fit."
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

    # ---- Seed-source combo --------------------------------------------

    def _refresh_guided_seed_source_combo(self):
        """Populate the Guided Fit seed combo with ``<Label> (<N>)``
        entries and preserve the user's current selection when possible.
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
                # Disable empty sources so the user can only pick real seeds.
                if count == 0:
                    idx = self.guidedSeedSourceCombo.count - 1
                    model_item = self.guidedSeedSourceCombo.model().item(idx)
                    if model_item is not None:
                        model_item.setEnabled(False)
            # Reinstate previous selection; if it's empty/unavailable,
            # default to the first non-empty source.
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

    def _selected_guided_seed_source(self):
        data = self.guidedSeedSourceCombo.currentData
        return str(data or "").strip().lower()

    def onGuidedSeedSourceChanged(self, _idx):
        """When the user picks a seed source, drive the shared
        trajectory table + source combo to match so the table below
        shows the seeds the user is about to fit.
        """
        key = self._selected_guided_seed_source()
        if not key:
            return
        self._set_workflow_active_source(key)
        self._set_guided_source_combo(key)
        self.onRefreshClicked()

    def onGuidedRefreshSeedsClicked(self):
        self._refresh_guided_seed_source_combo()
        self.onRefreshClicked()

    # ---- Table helpers (unchanged) ------------------------------------

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
        """Return (sitk_image, ijk_to_ras, ras_to_ijk) for the volume.

        Builds an SITK image from the array so the LoG pipeline works
        in Slicer regardless of whether the volume has an on-disk path.
        """
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

    def _fit_names(self, names):
        volume_node = self.ctSelector.currentNode()
        if volume_node is None:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "Postop CT Localization",
                "Select a CT volume.",
            )
            return
        traj_map = {traj["name"]: traj for traj in self.loadedTrajectories}
        targets = [traj_map[n] for n in names if n in traj_map]
        if not targets:
            self.log("[guided] nothing to fit (check at least one seed)")
            return

        self.guidedFitStatusLabel.setText("preprocessing LoG / hull …")
        try:
            slicer.app.processEvents()
        except Exception:
            pass

        try:
            img, ijk_to_ras, ras_to_ijk = self._guided_fit_volume_matrices(volume_node)
            features = gfe.compute_features(img, ras_to_ijk)
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

        applied_nodes = []
        success = 0
        for traj in targets:
            name = str(traj.get("name", ""))
            start_ras = traj.get("start") or [0.0, 0.0, 0.0]
            end_ras = traj.get("end") or [0.0, 0.0, 0.0]
            # The trajectory records store RAS already (via
            # trajectory_from_line_node); no LPS conversion needed.
            try:
                fit = gfe.fit_trajectory(
                    planned_start_ras=start_ras,
                    planned_end_ras=end_ras,
                    features=features,
                    ijk_to_ras_mat=ijk_to_ras,
                    ras_to_ijk_mat=ras_to_ijk,
                    roi_radius_mm=roi_mm,
                    max_angle_deg=max_angle,
                    max_lateral_shift_mm=max_lat,
                )
            except Exception as exc:
                self.log(f"[guided] {name}: fit crashed ({exc})")
                continue

            if not bool(fit.get("success")):
                self.log(
                    f"[guided] {name}: failed ({fit.get('reason', 'unknown')})"
                )
                continue

            existing = self.logic.trajectory_scene.find_line_by_group_and_name(
                name, "guided_fit",
            )
            node = self.logic.trajectory_scene.create_or_update_trajectory_line(
                name=name,
                start_ras=fit["start_ras"],
                end_ras=fit["end_ras"],
                node_id=None if existing is None else existing.GetID(),
                node_name=f"Guided_{name}",
                group="guided_fit",
                origin="postop_ct_guided_fit",
            )
            applied_nodes.append(node)
            success += 1
            self.log(
                "[guided] {name}: angle={a:.2f}° lat={l:.2f} mm n={n} "
                "len={L:.1f} mm".format(
                    name=name,
                    a=float(fit.get("angle_deg", 0.0)),
                    l=float(fit.get("lateral_shift_mm", 0.0)),
                    n=int(fit.get("n_inliers", 0)),
                    L=float(fit.get("length_mm", 0.0)),
                )
            )

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
                context_id=self.workflowState.context_id(
                    workflow_node=self.workflowNode
                ),
                nodes=applied_nodes,
            )
            self.logic.trajectory_scene.show_only_groups(["guided_fit"])
            self._set_workflow_active_source("guided_fit")
            self._set_guided_source_combo("guided_fit")
            self._refresh_guided_seed_source_combo()
            self.log(f"[guided] applied {len(applied_nodes)} trajectories")
            self.onRefreshClicked()

        self.guidedFitStatusLabel.setText(
            f"done — fitted {success}/{len(targets)} seeds"
        )

    def onFitSelectedClicked(self):
        names = self._checked_guided_trajectory_names()
        if not names:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "Postop CT Localization",
                "Check at least one trajectory in the table.",
            )
            return
        self._fit_names(names)

    def onFitAllClicked(self):
        names = [str(traj.get("name", "")) for traj in self.loadedTrajectories]
        names = [n for n in names if n]
        if not names:
            qt.QMessageBox.information(
                slicer.util.mainWindow(),
                "Postop CT Localization",
                "No seed trajectories loaded.",
            )
            return
        self._fit_names(names)

    # ``onApplyFitClicked`` and ``onDetectCandidatesClicked`` existed
    # under the legacy HU-threshold guided-fit flow. They no longer
    # have targets; kept as no-ops only if older code paths still
    # reference them (they don't after this rewrite).
