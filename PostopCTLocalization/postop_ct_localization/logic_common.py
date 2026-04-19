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

class PostopCTLocalizationLogicBaseMixin:
    def __init__(self):
        super().__init__()
        if np is None:
            raise RuntimeError("numpy is required for Postop CT Localization.")
        self.workflow_state = WorkflowState()
        self.workflow_publish = WorkflowPublisher(self.workflow_state)
        self.trajectory_scene = TrajectorySceneService()
        self.layout_service = LayoutService()
        self.electrode_scene = ElectrodeSceneService(
            workflow_state=self.workflow_state,
            workflow_publish=self.workflow_publish,
        )
        self.focus_controller = TrajectoryFocusController(
            trajectory_scene=self.trajectory_scene,
            electrode_scene=self.electrode_scene,
            layout_service=self.layout_service,
        )
        self.pipeline_registry = PipelineRegistry()
        register_builtin_pipelines(self.pipeline_registry)
        self.default_de_novo_pipeline_key = self._resolve_default_de_novo_pipeline_key()

    def _resolve_default_de_novo_pipeline_key(self):
        keys = list(self.pipeline_registry.keys())
        if not keys:
            return ""
        preferred = "contact_pitch_v1"
        return preferred if preferred in keys else str(keys[0])

    def available_de_novo_pipelines(self):
        """Return display metadata for registered de novo engine pipelines."""
        entries = []
        for key in self.pipeline_registry.keys():
            display_name = key
            scaffold = False
            expose_in_ui = True
            try:
                pipeline = self.pipeline_registry.create_pipeline(key)
                display_name = str(getattr(pipeline, "display_name", key) or key)
                scaffold = bool(getattr(pipeline, "scaffold", False))
                expose_in_ui = bool(getattr(pipeline, "expose_in_ui", True))
            except Exception:
                pass
            if not expose_in_ui:
                continue
            entries.append(
                {
                    "key": str(key),
                    "display_name": display_name,
                    "scaffold": scaffold,
                }
            )
        return entries

    def build_detection_context(self, volume_node):
        """Build a minimal ``DetectionContext`` dict for ``contact_pitch_v1``.

        The pipeline only needs ``arr_kji``, ``spacing_xyz``, and the
        Slicer ``volume_node`` (to pull the true IJK↔RAS matrices). All
        other keys are left to the caller to add (e.g. pitch strategy,
        vendors, logger).
        """
        return {
            "run_id": f"contact_pitch_{volume_node.GetName()}",
            "arr_kji": np.asarray(slicer.util.arrayFromVolume(volume_node), dtype=float),
            "spacing_xyz": tuple(float(v) for v in volume_node.GetSpacing()),
            "extras": {"volume_node": volume_node},
        }

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

    def sync_manual_trajectories_to_workflow(self, workflow_node=None):
        """Tag/publish scene-authored line markups as manual trajectories."""
        wf = workflow_node or self.workflow_state.resolve_or_create_workflow_node()
        rows = self.trajectory_scene.collect_working_trajectory_rows(groups=["manual"])
        nodes = []
        for row in rows:
            node = slicer.mrmlScene.GetNodeByID(row.get("node_id", ""))
            if node is None:
                continue
            self.trajectory_scene.set_trajectory_metadata(
                node=node,
                trajectory_name=row.get("name", ""),
                group="manual",
                origin=node.GetAttribute("Rosa.TrajectoryOrigin") or "manual_scene",
            )
            nodes.append(node)
        self.workflow_publish.publish_nodes(
            role="ManualTrajectoryLines",
            nodes=nodes,
            source="manual_scene",
            space_name="ROSA_BASE",
            workflow_node=wf,
        )
        self.workflow_publish.publish_nodes(
            role="WorkingTrajectoryLines",
            nodes=nodes,
            source="manual_scene",
            space_name="ROSA_BASE",
            workflow_node=wf,
        )
        self.trajectory_scene.place_trajectory_nodes_in_hierarchy(
            context_id=self.workflow_state.context_id(workflow_node=wf),
            nodes=nodes,
        )
        return len(nodes)

    @staticmethod

    @staticmethod
    def _engine_trajectories_to_lines(engine_result):
        """Map canonical engine trajectories to local line-row schema."""
        lines = []
        for idx, traj in enumerate(list(engine_result.get("trajectories") or []), start=1):
            start = list(traj.get("start_ras", []))
            end = list(traj.get("end_ras", []))
            params = dict(traj.get("params") or {})
            if len(start) != 3:
                start = list(params.get("start_ras", [0.0, 0.0, 0.0]))
            if len(end) != 3:
                end = list(params.get("end_ras", [0.0, 0.0, 0.0]))
            if len(start) != 3:
                start = [0.0, 0.0, 0.0]
            if len(end) != 3:
                end = [0.0, 0.0, 0.0]
            start_np = np.asarray(start, dtype=float).reshape(3)
            end_np = np.asarray(end, dtype=float).reshape(3)
            lines.append(
                {
                    "name": str(traj.get("name") or f"T{idx:02d}"),
                    "start_ras": [float(v) for v in start_np],
                    "end_ras": [float(v) for v in end_np],
                    "length_mm": float(np.linalg.norm(end_np - start_np)),
                    "inlier_count": int(traj.get("support_count", 0)),
                    "support_weight": float(traj.get("support_mass", traj.get("support_count", 0.0))),
                    "inside_fraction": float(traj.get("confidence", 0.0)),
                    "rms_mm": float(params.get("rms_mm", 0.0)),
                    "depth_span_mm": float(params.get("depth_span_mm", 0.0)),
                    "best_model_id": str(params.get("best_model_id", "")),
                    "best_model_score": params.get("best_model_score", None),
                }
            )
        return lines

    def rename_trajectory(self, node_id, new_name):
        """Rename one trajectory line node while preserving group metadata."""
        node = slicer.mrmlScene.GetNodeByID(str(node_id or ""))
        if node is None:
            return False
        return bool(self.trajectory_scene.rename_trajectory_node(node, new_name))

    def remove_trajectories_by_name(self, names, source_key="working"):
        """Delete trajectories by logical name from current scene/source scope."""
        source = str(source_key or "working").strip().lower()
        allowed_groups = {
            "working": {"imported_rosa", "imported_external", "manual", "guided_fit", "deep_core", "de_novo", "autofit_preview"},
            "imported_rosa": {"imported_rosa"},
            "imported_external": {"imported_external"},
            "manual": {"manual"},
            "guided_fit": {"guided_fit"},
            "deep_core": {"autofit_preview"},
            "de_novo": {"de_novo"},
            "planned_rosa": {"planned_rosa"},
        }.get(source, {"imported_rosa", "imported_external", "manual", "guided_fit", "deep_core", "de_novo", "autofit_preview"})

        target_names = {str(name).strip() for name in (names or []) if str(name).strip()}
        if not target_names:
            return 0
        removed = 0
        for node in list(slicer.util.getNodesByClass("vtkMRMLMarkupsLineNode")):
            group = self.trajectory_scene.infer_group_from_node(node)
            if group not in allowed_groups:
                continue
            logical_name = self.trajectory_scene.logical_name_from_node(node)
            if logical_name not in target_names:
                continue
            slicer.mrmlScene.RemoveNode(node)
            removed += 1
        return removed

    def _collect_trajectories_from_role(self, role, workflow_node=None):
        wf = workflow_node or self.workflow_state.resolve_or_create_workflow_node()
        trajectories = []
        for node in self.workflow_state.role_nodes(role, workflow_node=wf):
            traj = self.trajectory_scene.trajectory_from_line_node("", node)
            if traj is not None:
                trajectories.append(traj)
        trajectories.sort(key=lambda item: item.get("name", ""))
        return trajectories

    def collect_trajectories_by_source(self, source_key="working", workflow_node=None):
        wf = workflow_node or self.workflow_state.resolve_or_create_workflow_node()
        source = str(source_key or "working").strip().lower()

        if source == "working":
            trajectories = self._collect_trajectories_from_role("WorkingTrajectoryLines", workflow_node=wf)
            if trajectories:
                return trajectories
            rows = self.trajectory_scene.collect_working_trajectory_rows(
                groups=["imported_rosa", "imported_external", "manual", "guided_fit", "deep_core", "de_novo", "autofit_preview"]
            )
            fallback = []
            for row in rows:
                node = slicer.mrmlScene.GetNodeByID(row["node_id"])
                if node is None:
                    continue
                traj = self.trajectory_scene.trajectory_from_line_node(row["name"], node)
                if traj is not None:
                    fallback.append(traj)
            fallback.sort(key=lambda item: item.get("name", ""))
            return fallback

        role_map = {
            "imported_rosa": "ImportedTrajectoryLines",
            "imported_external": "ImportedExternalTrajectoryLines",
            "guided_fit": "GuidedFitTrajectoryLines",
            "deep_core": "DeepCoreTrajectoryLines",
            "de_novo": "DeNovoTrajectoryLines",
            "planned_rosa": "PlannedTrajectoryLines",
        }
        if source in role_map:
            return self._collect_trajectories_from_role(role_map[source], workflow_node=wf)

        if source == "manual":
            rows = self.trajectory_scene.collect_working_trajectory_rows(groups=["manual"])
            nodes = []
            trajectories = []
            for row in rows:
                node = slicer.mrmlScene.GetNodeByID(row["node_id"])
                if node is None:
                    continue
                self.trajectory_scene.set_trajectory_metadata(
                    node=node,
                    trajectory_name=row["name"],
                    group="manual",
                    origin=node.GetAttribute("Rosa.TrajectoryOrigin") or "manual",
                )
                nodes.append(node)
                traj = self.trajectory_scene.trajectory_from_line_node(row["name"], node)
                if traj is not None:
                    trajectories.append(traj)
            self.workflow_publish.publish_nodes(
                role="ManualTrajectoryLines",
                nodes=nodes,
                source="manual",
                space_name="ROSA_BASE",
                workflow_node=wf,
            )
            self.trajectory_scene.place_trajectory_nodes_in_hierarchy(
                context_id=self.workflow_state.context_id(workflow_node=wf),
                nodes=nodes,
            )
            trajectories.sort(key=lambda item: item.get("name", ""))
            return trajectories

        return []

    def _ijk_kji_to_ras_points(self, volume_node, ijk_kji):
        # Accept list or ndarray; normalize to (N,3) array in KJI order.
        idx = np.asarray(ijk_kji, dtype=float).reshape(-1, 3)
        ijk = np.zeros_like(idx, dtype=float)
        ijk[:, 0] = idx[:, 2]
        ijk[:, 1] = idx[:, 1]
        ijk[:, 2] = idx[:, 0]
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

    @staticmethod

    @staticmethod
    def _electrode_prior_support_params(models_by_id):
        """Derive axial-support sampling priors from loaded electrode model geometry."""
        if not isinstance(models_by_id, dict) or not models_by_id:
            return {}
        diameters = []
        contact_lengths = []
        pitches = []
        gaps = []
        total_lengths = []
        for model in list(models_by_id.values()):
            if not isinstance(model, dict):
                continue
            try:
                diameters.append(float(model.get("diameter_mm")))
            except Exception:
                pass
            try:
                contact_lengths.append(float(model.get("contact_length_mm")))
            except Exception:
                pass
            offsets = model.get("contact_center_offsets_from_tip_mm")
            if isinstance(offsets, (list, tuple)) and len(offsets) >= 2:
                vals = []
                for off in offsets:
                    try:
                        vals.append(float(off))
                    except Exception:
                        pass
                vals = sorted(vals)
                for i in range(1, len(vals)):
                    d = float(vals[i] - vals[i - 1])
                    if d > 0.0:
                        pitches.append(d)
                try:
                    contact_len = float(model.get("contact_length_mm"))
                    if contact_len > 0.0:
                        for p in pitches[-max(1, len(vals) - 1) :]:
                            gaps.append(max(0.0, float(p) - contact_len))
                except Exception:
                    pass
            try:
                total_lengths.append(float(model.get("total_exploration_length_mm")))
            except Exception:
                pass

        if not diameters:
            return {}
        diameter_mm = float(np.median(np.asarray(diameters, dtype=float)))
        contact_length_mm = float(np.median(np.asarray(contact_lengths, dtype=float))) if contact_lengths else 2.0
        pitch_mm = float(np.median(np.asarray(pitches, dtype=float))) if pitches else 3.5
        gap_mm = float(np.median(np.asarray(gaps, dtype=float))) if gaps else max(0.0, pitch_mm - contact_length_mm)
        spacing_mm = float(max(1.0, contact_length_mm + gap_mm if gap_mm > 0.0 else pitch_mm))
        max_exploration_mm = float(np.max(np.asarray(total_lengths, dtype=float))) if total_lengths else 60.0
        max_samples = int(np.clip(np.ceil(max_exploration_mm / max(spacing_mm, 1e-3)) + 2, 12, 48))
        return {
            "electrode_prior_diameter_mm": diameter_mm,
            "electrode_prior_contact_length_mm": contact_length_mm,
            "electrode_prior_pitch_mm": pitch_mm,
            "electrode_prior_gap_mm": gap_mm,
            "electrode_prior_contact_separation_mm": spacing_mm,
            "electrode_prior_max_exploration_mm": max_exploration_mm,
            "axial_support_spacing_mm": spacing_mm,
            "axial_support_window_mm": float(max(contact_length_mm, 0.75 * spacing_mm)),
            "axial_support_min_diameter_mm": float(max(0.4, 0.5 * diameter_mm)),
            "axial_support_max_diameter_mm": float(max(2.5, 3.0 * diameter_mm)),
            "axial_support_min_aspect_ratio": 2.0,
            "axial_support_min_length_mm": float(max(3.0, 1.5 * spacing_mm)),
            "axial_support_max_samples_per_blob": max_samples,
            "axial_support_min_window_voxels": 3,
        }

    def upsert_detected_lines(self, lines, replace_existing=True):
        existing_rows = self.trajectory_scene.collect_working_trajectory_rows(groups=["de_novo"])
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
            node_name = self.trajectory_scene.build_node_name(name, "de_novo")
            node = self.trajectory_scene.create_or_update_trajectory_line(
                name=name,
                start_ras=line["start_ras"],
                end_ras=line["end_ras"],
                node_id=None,
                group="de_novo",
                origin="postop_ct_denovo",
                node_name=node_name,
            )
            new_rows.append(
                {
                    "name": name,
                    "node_name": node.GetName() or node_name,
                    "node_id": node.GetID(),
                    "group": "de_novo",
                    "start_ras": [float(line["start_ras"][0]), float(line["start_ras"][1]), float(line["start_ras"][2])],
                    "end_ras": [float(line["end_ras"][0]), float(line["end_ras"][1]), float(line["end_ras"][2])],
                }
            )
        new_rows.sort(key=lambda r: r["name"])
        return new_rows

    def publish_working_rows(self, rows, workflow_node=None, role="WorkingTrajectoryLines", source="postop_ct_localization"):
        nodes = []
        for row in rows:
            node = slicer.mrmlScene.GetNodeByID(row.get("node_id", ""))
            if node is not None:
                nodes.append(node)
        if role and role != "WorkingTrajectoryLines":
            self.workflow_publish.publish_nodes(
                role=role,
                nodes=nodes,
                source=source,
                space_name="ROSA_BASE",
                workflow_node=workflow_node,
            )
        self.workflow_publish.publish_nodes(
            role="WorkingTrajectoryLines",
            nodes=nodes,
            source=source,
            space_name="ROSA_BASE",
            workflow_node=workflow_node,
        )
        wf = workflow_node or self.workflow_state.resolve_or_create_workflow_node()
        self.trajectory_scene.place_trajectory_nodes_in_hierarchy(
            context_id=self.workflow_state.context_id(workflow_node=wf),
            nodes=nodes,
        )
