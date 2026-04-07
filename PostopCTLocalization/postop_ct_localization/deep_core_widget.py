import json
import os
import csv

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

class DeepCoreDebugWidgetMixin:
    @staticmethod
    def _deep_core_model_family_key(model_id):
        text = str(model_id or "").strip().upper()
        if not text:
            return ""
        parts = text.split("-", 1)
        prefix = parts[0]
        if len(parts) == 1:
            return prefix
        suffix = parts[1]
        digits = []
        letters = []
        for ch in suffix:
            if ch.isdigit() and not letters:
                digits.append(ch)
            elif ch.isalpha():
                letters.append(ch)
        family_suffix = "".join(letters).strip().upper()
        return f"{prefix}-{family_suffix}" if family_suffix else prefix

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

        self.deepCoreHullThresholdSpin = qt.QDoubleSpinBox()
        self.deepCoreHullThresholdSpin.setRange(-1200.0, 500.0)
        self.deepCoreHullThresholdSpin.setDecimals(1)
        self.deepCoreHullThresholdSpin.setValue(-500.0)
        self.deepCoreHullThresholdSpin.setSuffix(" HU")
        form.addRow("Hull threshold", self.deepCoreHullThresholdSpin)

        self.deepCoreHullClipSpin = qt.QDoubleSpinBox()
        self.deepCoreHullClipSpin.setRange(100.0, 4000.0)
        self.deepCoreHullClipSpin.setDecimals(1)
        self.deepCoreHullClipSpin.setValue(1200.0)
        self.deepCoreHullClipSpin.setSuffix(" HU")
        form.addRow("Hull clip", self.deepCoreHullClipSpin)

        self.deepCoreHullSigmaSpin = qt.QDoubleSpinBox()
        self.deepCoreHullSigmaSpin.setRange(0.0, 20.0)
        self.deepCoreHullSigmaSpin.setDecimals(2)
        self.deepCoreHullSigmaSpin.setValue(4.0)
        self.deepCoreHullSigmaSpin.setSuffix(" mm")
        form.addRow("Hull Gaussian", self.deepCoreHullSigmaSpin)

        self.deepCoreHullOpenSpin = qt.QSpinBox()
        self.deepCoreHullOpenSpin.setRange(0, 30)
        self.deepCoreHullOpenSpin.setValue(7)
        self.deepCoreHullOpenSpin.setSuffix(" vox")
        form.addRow("Hull opening", self.deepCoreHullOpenSpin)

        self.deepCoreHullCloseSpin = qt.QSpinBox()
        self.deepCoreHullCloseSpin.setRange(0, 30)
        self.deepCoreHullCloseSpin.setValue(0)
        self.deepCoreHullCloseSpin.setSuffix(" vox")
        form.addRow("Hull closing", self.deepCoreHullCloseSpin)

        self.deepCoreShrinkSpin = qt.QDoubleSpinBox()
        self.deepCoreShrinkSpin.setRange(0.0, 40.0)
        self.deepCoreShrinkSpin.setDecimals(2)
        self.deepCoreShrinkSpin.setValue(15.0)
        self.deepCoreShrinkSpin.setSuffix(" mm")
        form.addRow("Deep-core shrink", self.deepCoreShrinkSpin)

        self.deepCoreMetalThresholdSpin = qt.QDoubleSpinBox()
        self.deepCoreMetalThresholdSpin.setRange(-1200.0, 4000.0)
        self.deepCoreMetalThresholdSpin.setDecimals(1)
        self.deepCoreMetalThresholdSpin.setValue(1900.0)
        self.deepCoreMetalThresholdSpin.setSuffix(" HU")
        form.addRow("Metal threshold", self.deepCoreMetalThresholdSpin)

        self.deepCoreMetalGrowSpin = qt.QSpinBox()
        self.deepCoreMetalGrowSpin.setRange(0, 6)
        self.deepCoreMetalGrowSpin.setValue(1)
        self.deepCoreMetalGrowSpin.setSuffix(" vox")
        form.addRow("Metal grow", self.deepCoreMetalGrowSpin)

        self.deepCoreBlobSampleSpacingSpin = qt.QDoubleSpinBox()
        self.deepCoreBlobSampleSpacingSpin.setRange(0.5, 10.0)
        self.deepCoreBlobSampleSpacingSpin.setDecimals(2)
        self.deepCoreBlobSampleSpacingSpin.setValue(2.5)
        self.deepCoreBlobSampleSpacingSpin.setSuffix(" mm")
        form.addRow("Blob sample spacing", self.deepCoreBlobSampleSpacingSpin)

        self.deepCoreBlobMinElongationSpin = qt.QDoubleSpinBox()
        self.deepCoreBlobMinElongationSpin.setRange(1.0, 50.0)
        self.deepCoreBlobMinElongationSpin.setDecimals(2)
        self.deepCoreBlobMinElongationSpin.setValue(4.0)
        form.addRow("Blob min elongation", self.deepCoreBlobMinElongationSpin)

        self.deepCoreLineDiameterSpin = qt.QDoubleSpinBox()
        self.deepCoreLineDiameterSpin.setRange(0.2, 10.0)
        self.deepCoreLineDiameterSpin.setDecimals(2)
        self.deepCoreLineDiameterSpin.setValue(2.0)
        self.deepCoreLineDiameterSpin.setSuffix(" mm")
        form.addRow("Direct line max diameter", self.deepCoreLineDiameterSpin)

        self.deepCoreLineMinSpanSpin = qt.QDoubleSpinBox()
        self.deepCoreLineMinSpanSpin.setRange(1.0, 50.0)
        self.deepCoreLineMinSpanSpin.setDecimals(2)
        self.deepCoreLineMinSpanSpin.setValue(10.0)
        self.deepCoreLineMinSpanSpin.setSuffix(" mm")
        form.addRow("Direct line min span", self.deepCoreLineMinSpanSpin)

        self.deepCoreLinePCADominanceSpin = qt.QDoubleSpinBox()
        self.deepCoreLinePCADominanceSpin.setRange(1.0, 50.0)
        self.deepCoreLinePCADominanceSpin.setDecimals(2)
        self.deepCoreLinePCADominanceSpin.setValue(6.0)
        form.addRow("Direct line PCA dominance", self.deepCoreLinePCADominanceSpin)

        self.deepCoreContactDiameterSpin = qt.QDoubleSpinBox()
        self.deepCoreContactDiameterSpin.setRange(0.5, 10.0)
        self.deepCoreContactDiameterSpin.setDecimals(2)
        self.deepCoreContactDiameterSpin.setValue(10.0)
        self.deepCoreContactDiameterSpin.setSuffix(" mm")
        form.addRow("Contact max diameter", self.deepCoreContactDiameterSpin)

        self.deepCoreComplexCubeSizeSpin = qt.QDoubleSpinBox()
        self.deepCoreComplexCubeSizeSpin.setRange(1.0, 10.0)
        self.deepCoreComplexCubeSizeSpin.setDecimals(2)
        self.deepCoreComplexCubeSizeSpin.setValue(5.0)
        self.deepCoreComplexCubeSizeSpin.setSuffix(" mm")
        form.addRow("Complex cube size", self.deepCoreComplexCubeSizeSpin)

        self.deepCoreShowBlobCheck = qt.QCheckBox("Show deep-seed blob diagnostics")
        self.deepCoreShowBlobCheck.setChecked(True)
        form.addRow(self.deepCoreShowBlobCheck)
        self.deepCoreSupportLegendLabel = qt.QLabel(
            "Debug markers: yellow=raw blob centroids, dark blue=line-blob tokens, "
            "green=contact-blob tokens, orange=complex-blob tokens, gold=support atom centers, "
            "cyan=support atom lines."
        )
        self.deepCoreSupportLegendLabel.wordWrap = True
        form.addRow(self.deepCoreSupportLegendLabel)

        self.deepCoreProposalDisplayModeCombo = qt.QComboBox()
        self.deepCoreProposalDisplayModeCombo.addItem("All proposal families", "all")
        self.deepCoreProposalDisplayModeCombo.addItem("Local families", "local_only")
        self.deepCoreProposalDisplayModeCombo.addItem("Graph + blob-connectivity only", "graph_blob_only")
        self.deepCoreProposalDisplayModeCombo.addItem("Pair fallback only", "pair_only")
        self.deepCoreProposalDisplayModeCombo.addItem("Rescue only", "rescue_only")
        form.addRow("Proposal display", self.deepCoreProposalDisplayModeCombo)

        self.deepCoreProposalLegendLabel = qt.QLabel(
            "Colors: graph=green, blob-connectivity=cyan, blob-axis=orange, "
            "pair fallback=magenta, uncovered rescue=red, dropout rescue=yellow."
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
            "local_only": {"graph", "blob_connectivity", "blob_axis", "contact_chain"},
            "graph_blob_only": {"graph", "blob_connectivity"},
            "pair_only": {"pair_ransac"},
            "rescue_only": {"uncovered_rescue", "dropout_rescue"},
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
            "contact_chain",
            "pair_ransac",
            "uncovered_rescue",
            "dropout_rescue",
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
        payload = dict(self._lastDeepCoreProposalPayload or {})
        volume_node = self.ctSelector.currentNode()
        if volume_node is None or not payload or payload.get("volume_node_id") != volume_node.GetID():
            self._lastDeepCoreProposalNodes = []
            return []
        proposals = list(payload.get("proposals") or [])
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
            payload = self.logic.build_deep_core_debug(
                volume_node=volume_node,
                hull_threshold_hu=float(self.deepCoreHullThresholdSpin.value),
                hull_clip_hu=float(self.deepCoreHullClipSpin.value),
                hull_sigma_mm=float(self.deepCoreHullSigmaSpin.value),
                hull_open_vox=int(self.deepCoreHullOpenSpin.value),
                hull_close_vox=int(self.deepCoreHullCloseSpin.value),
                deep_core_shrink_mm=float(self.deepCoreShrinkSpin.value),
                metal_threshold_hu=float(self.deepCoreMetalThresholdSpin.value),
                metal_grow_vox=int(self.deepCoreMetalGrowSpin.value),
                blob_sample_spacing_mm=float(self.deepCoreBlobSampleSpacingSpin.value),
                blob_min_elongation=float(self.deepCoreBlobMinElongationSpin.value),
                blob_line_diameter_max_mm=float(self.deepCoreLineDiameterSpin.value),
                blob_line_min_span_mm=float(self.deepCoreLineMinSpanSpin.value),
                blob_line_min_pca_dominance=float(self.deepCoreLinePCADominanceSpin.value),
                blob_contact_diameter_max_mm=float(self.deepCoreContactDiameterSpin.value),
                blob_complex_cube_size_mm=float(self.deepCoreComplexCubeSizeSpin.value),
                show_blob_diagnostics=bool(self.deepCoreShowBlobCheck.checked),
            )
        except Exception as exc:
            self.log(f"[deep-core] error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Postop CT Localization", str(exc))
            return

        self._lastDeepCorePayload = dict(payload or {})
        self._lastDeepCoreProposalPayload = None
        self.logic.trajectory_scene.remove_preview_lines()
        self.logic.register_postop_ct(volume_node, workflow_node=self.workflowNode)
        self._apply_primary_slice_layers()

        try:
            self.logic.show_metal_and_head_masks(
                volume_node=volume_node,
                metal_mask_kji=payload.get("metal_mask_kji"),
                head_mask_kji=payload.get("hull_mask_kji"),
                head_distance_map_kji=payload.get("head_distance_map_kji"),
                head_core_mask_kji=payload.get("deep_core_mask_kji"),
                metal_gate_mask_kji=payload.get("metal_grown_mask_kji"),
                metal_in_gate_mask_kji=payload.get("deep_seed_mask_kji"),
                metal_depth_pass_mask_kji=payload.get("deep_seed_raw_mask_kji"),
            )
            self.log("[deep-core] displayed deep-core mask overlay")
        except Exception as exc:
            self.log(f"[deep-core] mask display warning: {exc}")

        if bool(self.deepCoreShowBlobCheck.checked):
            try:
                self.logic.show_blob_diagnostics(
                    volume_node=volume_node,
                    blob_labelmap_kji=payload.get("blob_labelmap_kji"),
                    blob_centroids_all_ras=payload.get("blob_centroids_all_ras"),
                    blob_centroids_kept_ras=payload.get("blob_sample_points_ras"),
                    blob_centroids_rejected_ras=np.empty((0, 3), dtype=float),
                    line_blob_points_ras=payload.get("line_blob_sample_points_ras"),
                    contact_blob_points_ras=payload.get("contact_blob_sample_points_ras"),
                    complex_blob_points_ras=payload.get("complex_blob_sample_points_ras"),
                    complex_blob_chain_rows=payload.get("complex_blob_chain_rows"),
                    contact_chain_rows=payload.get("contact_chain_rows"),
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
                    support_atoms=payload.get("support_atoms"),
                )
                self.log(
                    "[deep-core] displayed support atoms in 3D (atoms={count})".format(
                        count=len(list(payload.get("support_atoms") or [])),
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

        stats = dict(payload.get("stats") or {})
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
        chain_rows = list(payload.get("complex_blob_chain_rows") or [])
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
        contact_chain_debug_rows = list(payload.get("contact_chain_debug_rows") or [])
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
        smooth_node = payload.get("smoothed_hull_volume_node")
        distance_node = payload.get("head_distance_volume_node")
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
        payload = dict(self._lastDeepCorePayload or {})
        if not payload or payload.get("volume_node_id") != volume_node.GetID():
            qt.QMessageBox.information(
                slicer.util.mainWindow(),
                "Postop CT Localization",
                "Build Deep-Core Debug Masks first for the current CT.",
            )
            return
        try:
            proposal_payload = self.logic.build_deep_core_proposals(
                volume_node=volume_node,
                token_points_ras=payload.get("blob_sample_points_ras"),
                token_blob_ids=payload.get("blob_sample_blob_ids"),
                token_atom_ids=payload.get("blob_sample_atom_ids"),
                blob_axes_ras_by_id=payload.get("blob_axes_ras_by_id"),
                blob_elongation_by_id=payload.get("blob_elongation_by_id"),
                support_atoms=payload.get("support_atoms"),
                guided_threshold_hu=float(self.deepCoreMetalThresholdSpin.value),
                guided_head_mask_threshold_hu=float(self.deepCoreHullThresholdSpin.value),
                head_distance_map_kji=payload.get("head_distance_map_kji"),
                deep_core_shrink_mm=float(self.deepCoreShrinkSpin.value),
                guided_candidate_mask_kji=payload.get("metal_grown_mask_kji"),
            )
        except Exception as exc:
            self.log(f"[deep-core] proposal error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Postop CT Localization", str(exc))
            return

        self._lastDeepCoreProposalPayload = dict(proposal_payload or {})
        self._lastDeepCoreProposalPayload["volume_node_id"] = volume_node.GetID()
        proposals = list(self._lastDeepCoreProposalPayload.get("proposals") or [])
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
                cand=int(proposal_payload.get("candidate_count", 0)),
                tok=int(proposal_payload.get("token_count", 0)),
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
