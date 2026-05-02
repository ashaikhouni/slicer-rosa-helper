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

from .constants import GUIDED_SOURCE_OPTIONS

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

    def build_detection_context(self, volume_node):
        """Build a minimal ``DetectionContext`` dict for ``contact_pitch_v1``.

        When the volume has a storage node pointing at a readable NIfTI,
        populate ``ctx['ct']`` with the file path so the pipeline reads
        via ``sitk.ReadImage`` — the identical path the CLI regression
        uses. This sidesteps two sources of drift between Slicer and
        CLI runs on the same file:

          * ``slicer.util.arrayFromVolume`` returning a re-scaled /
            re-oriented copy of the NIfTI voxel data (scl_slope/scl_inter
            handling, axis reorientation).
          * ``volume_node.GetIJKToRASMatrix`` returning a matrix that
            Slicer may have flipped to a canonical RAS diagonal on load.

        ``arr_kji`` / ``spacing_xyz`` stay on the context as a fallback
        for any consumer that expects them (and for volumes that don't
        have an on-disk source, e.g. scene-authored scalar volumes).
        """
        from types import SimpleNamespace

        ct_ref = None
        try:
            storage = volume_node.GetStorageNode()
            src = storage.GetFileName() if storage is not None else ""
            if src and os.path.exists(src):
                ct_ref = SimpleNamespace(
                    volume_id=volume_node.GetName(), path=str(src),
                )
        except Exception:
            ct_ref = None

        ctx: dict = {
            "run_id": f"contact_pitch_{volume_node.GetName()}",
            "arr_kji": np.asarray(slicer.util.arrayFromVolume(volume_node), dtype=float),
            "spacing_xyz": tuple(float(v) for v in volume_node.GetSpacing()),
            "extras": {"volume_node": volume_node},
        }
        if ct_ref is not None:
            ctx["ct"] = ct_ref
        return ctx

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

    MANUAL_ORIENTATION_AUTO = "auto"
    MANUAL_ORIENTATION_ENTRY_FIRST = "entry_first"
    MANUAL_ORIENTATION_TARGET_FIRST = "target_first"

    def sync_manual_trajectories_to_workflow(
        self, workflow_node=None, orientation="auto", pitch_strategy="auto",
    ):
        """Tag/publish scene-authored line markups as manual trajectories.

        ``orientation`` controls how each line's two control points are
        ordered before the canonical (cp0=Entry, cp1=Target) labels are
        stamped:

          * ``"auto"`` (default) — sample the postop CT head-distance map
            at both endpoints; the shallower (closer-to-skull) end becomes
            cp0 / Entry. Falls back to ``"entry_first"`` when no postop CT
            volume is registered.
          * ``"entry_first"`` — trust the user's click order as drawn.
          * ``"target_first"`` — swap cp0/cp1 unconditionally.

        Returns ``(n_synced, n_reoriented)``.
        """
        wf = workflow_node or self.workflow_state.resolve_or_create_workflow_node()
        rows = self.trajectory_scene.collect_working_trajectory_rows(groups=["manual"])
        mode = self._normalize_manual_orientation(orientation)
        volume_node, dist_map = (None, None)
        if mode == self.MANUAL_ORIENTATION_AUTO:
            volume_node = self._find_postop_ct_volume_node(workflow_node=wf)
            dist_map = self._ensure_manual_fit_head_distance_map(volume_node)
            if dist_map is None:
                # Hull distance unavailable — preserve user click order rather
                # than silently mis-orient.
                mode = self.MANUAL_ORIENTATION_ENTRY_FIRST
        nodes = []
        n_reoriented = 0
        # Pre-load the postop CT once so we can run the unified electrode-
        # model picker on every synced manual line. PaCER template
        # correlation needs (volume_kji, ras_to_ijk_mat, start_ras,
        # end_ras); Manual Fit's user-drawn endpoints are the seed.
        ct_arr_kji = None
        ras_to_ijk_for_pick = None
        try:
            ct_volume_for_pick = volume_node or self._find_postop_ct_volume_node(
                workflow_node=wf,
            )
            if ct_volume_for_pick is not None:
                import SimpleITK as _sitk
                from shank_core.io import image_ijk_ras_matrices
                # Pull the canonical-resampled volume from the same
                # `prepare_volume` path that detection uses, so manual
                # picks stay consistent with Auto / Guided picks.
                from postop_ct_localization.contact_pitch_v1_fit import prepare_volume
                _img = _sitk.GetImageFromArray(
                    slicer.util.arrayFromVolume(ct_volume_for_pick)
                )
                _i2r, _r2i = image_ijk_ras_matrices(ct_volume_for_pick)
                _img, _i2r, _r2i = prepare_volume(_img, _i2r, _r2i)
                ct_arr_kji = _sitk.GetArrayFromImage(_img).astype("float32")
                ras_to_ijk_for_pick = _r2i
        except Exception:
            ct_arr_kji = None
            ras_to_ijk_for_pick = None
        for row in rows:
            node = slicer.mrmlScene.GetNodeByID(row.get("node_id", ""))
            if node is None:
                continue
            if self._apply_manual_orientation(node, mode, volume_node, dist_map):
                n_reoriented += 1
            self.trajectory_scene.set_trajectory_metadata(
                node=node,
                trajectory_name=row.get("name", ""),
                group="manual",
                origin=node.GetAttribute("Rosa.TrajectoryOrigin") or "manual_scene",
            )
            node.SetAttribute("Rosa.ManualOrientation", mode)
            # Stamp the unified picker's model id so Contacts &
            # Trajectory View can read `Rosa.BestModelId` directly
            # instead of falling back to length-only guessing on every
            # populate.
            try:
                if ct_arr_kji is not None:
                    traj = self.trajectory_scene.trajectory_from_line_node("", node)
                    if traj is not None:
                        from rosa_core.electrode_classifier import classify_electrode_model
                        pick = classify_electrode_model(
                            start_ras=traj["start_ras"],
                            end_ras=traj["end_ras"],
                            pitch_strategy=str(pitch_strategy or "auto"),
                            ct_volume_kji=ct_arr_kji,
                            ras_to_ijk_mat=ras_to_ijk_for_pick,
                        )
                        if pick is not None:
                            node.SetAttribute(
                                "Rosa.BestModelId", str(pick.get("model_id") or ""),
                            )
                            method = str(pick.get("method") or "")
                            if method:
                                node.SetAttribute(
                                    "Rosa.SuggestedElectrodeMethod", method,
                                )
            except Exception:
                pass
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
        return len(nodes), n_reoriented

    def seed_manual_from_source(self, source_key, workflow_node=None):
        """Clone trajectories from another group into the Manual set.

        Lets the user start from Auto Fit / Guided Fit / imported lines and
        then add missing shanks by drawing extra line markups, instead of
        re-deriving everything from scratch. Existing manual lines with the
        same logical name are skipped — re-clicking Seed is idempotent and
        does not duplicate. The source nodes are NOT modified; new nodes
        are created in the Manual group with origin = ``seeded_from_<src>``.

        Returns ``(n_added, n_skipped)``.
        """
        wf = workflow_node or self.workflow_state.resolve_or_create_workflow_node()
        src = str(source_key or "").strip().lower()
        if src not in {
            "auto_fit", "guided_fit", "imported_rosa",
            "imported_external", "planned_rosa",
        }:
            return 0, 0
        existing_manual_names = {
            str(row.get("name", "")).strip()
            for row in self.trajectory_scene.collect_working_trajectory_rows(
                groups=["manual"]
            )
        }
        source_rows = self.trajectory_scene.collect_working_trajectory_rows(
            groups=[src]
        )
        nodes = []
        n_added = 0
        n_skipped = 0
        for row in source_rows:
            name = str(row.get("name", "")).strip()
            if not name:
                continue
            if name in existing_manual_names:
                n_skipped += 1
                continue
            start = row.get("start_ras")
            end = row.get("end_ras")
            if not start or not end or len(start) < 3 or len(end) < 3:
                continue
            node = self.trajectory_scene.create_or_update_trajectory_line(
                name=name,
                start_ras=start,
                end_ras=end,
                node_id=None,
                group="manual",
                origin=f"seeded_from_{src}",
                node_name=None,
            )
            if node is None:
                continue
            node.SetAttribute("Rosa.SeededFrom", src)
            nodes.append(node)
            existing_manual_names.add(name)
            n_added += 1
        if nodes:
            self.workflow_publish.publish_nodes(
                role="ManualTrajectoryLines",
                nodes=nodes,
                source=f"manual_seeded_from_{src}",
                space_name="ROSA_BASE",
                workflow_node=wf,
            )
            self.workflow_publish.publish_nodes(
                role="WorkingTrajectoryLines",
                nodes=nodes,
                source=f"manual_seeded_from_{src}",
                space_name="ROSA_BASE",
                workflow_node=wf,
            )
            self.trajectory_scene.place_trajectory_nodes_in_hierarchy(
                context_id=self.workflow_state.context_id(workflow_node=wf),
                nodes=nodes,
            )
        return n_added, n_skipped

    def swap_manual_trajectory_endpoints(self, workflow_node=None):
        """Swap cp0/cp1 on every manual line in the scene. Returns swap count.

        Get-out-of-jail card for users who left ``orientation`` on
        ``"auto"`` and got the result wrong (e.g. shank entered through a
        thin region the head mask classified as outside-air, so the bolt
        sampled with negative depth).
        """
        wf = workflow_node or self.workflow_state.resolve_or_create_workflow_node()
        rows = self.trajectory_scene.collect_working_trajectory_rows(groups=["manual"])
        n = 0
        for row in rows:
            node = slicer.mrmlScene.GetNodeByID(row.get("node_id", ""))
            if node is None:
                continue
            if not self._swap_line_control_points(node):
                continue
            self.trajectory_scene.set_trajectory_metadata(
                node=node,
                trajectory_name=row.get("name", ""),
                group="manual",
                origin=node.GetAttribute("Rosa.TrajectoryOrigin") or "manual_scene",
            )
            n += 1
        return n

    @classmethod
    def _normalize_manual_orientation(cls, orientation):
        mode = str(orientation or "").strip().lower()
        valid = {
            cls.MANUAL_ORIENTATION_AUTO,
            cls.MANUAL_ORIENTATION_ENTRY_FIRST,
            cls.MANUAL_ORIENTATION_TARGET_FIRST,
        }
        return mode if mode in valid else cls.MANUAL_ORIENTATION_AUTO

    def _find_postop_ct_volume_node(self, workflow_node=None):
        """Resolve the registered postop CT volume node, or None."""
        wf = workflow_node or self.workflow_state.resolve_or_create_workflow_node()
        for role in ("PostopCT", "AdditionalCTVolumes"):
            nodes = self.workflow_state.role_nodes(role, workflow_node=wf)
            for node in nodes:
                if node is not None:
                    return node
        return None

    def _ensure_manual_fit_head_distance_map(self, volume_node):
        """Lazy-build and cache the head-distance map (kji, mm) for ``volume_node``."""
        if volume_node is None:
            return None
        cache = getattr(self, "_manual_fit_head_dist_cache", None)
        cache_key = volume_node.GetID()
        if cache and cache.get("key") == cache_key:
            return cache.get("dist_map_kji")
        try:
            arr = np.asarray(slicer.util.arrayFromVolume(volume_node), dtype=float)
        except Exception:
            return None
        spacing = tuple(float(v) for v in volume_node.GetSpacing())
        masks = build_preview_masks(
            arr_kji=arr,
            spacing_xyz=spacing,
            threshold=1500.0,
            use_head_mask=False,
            build_head_mask=True,
        )
        dist_map = masks.get("head_distance_map_kji")
        if dist_map is None:
            return None
        self._manual_fit_head_dist_cache = {
            "key": cache_key,
            "dist_map_kji": np.asarray(dist_map, dtype=np.float32),
        }
        return self._manual_fit_head_dist_cache["dist_map_kji"]

    def _apply_manual_orientation(self, node, mode, volume_node, dist_map):
        """Reorient one manual line per ``mode``. Returns True if cp0/cp1 swapped."""
        if node is None or node.GetNumberOfControlPoints() < 2:
            return False
        if mode == self.MANUAL_ORIENTATION_ENTRY_FIRST:
            return False
        if mode == self.MANUAL_ORIENTATION_TARGET_FIRST:
            return self._swap_line_control_points(node)
        # auto
        if dist_map is None or volume_node is None:
            return False
        p0_world = [0.0, 0.0, 0.0]
        p1_world = [0.0, 0.0, 0.0]
        node.GetNthControlPointPositionWorld(0, p0_world)
        node.GetNthControlPointPositionWorld(1, p1_world)
        d0 = self._sample_head_distance_at_world(volume_node, dist_map, p0_world)
        d1 = self._sample_head_distance_at_world(volume_node, dist_map, p1_world)
        if d0 <= d1:
            return False
        return self._swap_line_control_points(node)

    def _sample_head_distance_at_world(self, volume_node, dist_map, world_xyz):
        """Sample head-distance map at a markup-world point. World is LPS in this scene."""
        ras = lps_to_ras_point([float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2])])
        ijk = self._ras_to_ijk_float(volume_node, ras)
        K, J, I = dist_map.shape
        i = int(np.clip(round(float(ijk[0])), 0, I - 1))
        j = int(np.clip(round(float(ijk[1])), 0, J - 1))
        k = int(np.clip(round(float(ijk[2])), 0, K - 1))
        return float(dist_map[k, j, i])

    def _swap_line_control_points(self, node):
        """Swap cp0 and cp1 positions on a markup line node. Returns True on success."""
        if node is None or node.GetNumberOfControlPoints() < 2:
            return False
        p0 = [0.0, 0.0, 0.0]
        p1 = [0.0, 0.0, 0.0]
        node.GetNthControlPointPositionWorld(0, p0)
        node.GetNthControlPointPositionWorld(1, p1)
        node.RemoveAllControlPoints()
        node.AddControlPoint(vtk.vtkVector3d(*p1))
        node.AddControlPoint(vtk.vtkVector3d(*p0))
        return True

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
            "working": {"imported_rosa", "imported_external", "manual", "guided_fit", "auto_fit"},
            "imported_rosa": {"imported_rosa"},
            "imported_external": {"imported_external"},
            "manual": {"manual"},
            "guided_fit": {"guided_fit"},
            "auto_fit": {"auto_fit"},
            "planned_rosa": {"planned_rosa"},
        }.get(source, {"imported_rosa", "imported_external", "manual", "guided_fit", "auto_fit"})

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

    def count_trajectories_by_source(self, source_key, workflow_node=None):
        """Cheap count of trajectories available under one source key.

        Used by Guided Fit to populate its seed-source dropdown with
        ``"<Label> (<N>)"`` entries without loading every trajectory's
        markup node.
        """
        return len(
            self.collect_trajectories_by_source(source_key, workflow_node=workflow_node)
        )

    def collect_trajectories_by_source(self, source_key="working", workflow_node=None):
        wf = workflow_node or self.workflow_state.resolve_or_create_workflow_node()
        source = str(source_key or "working").strip().lower()

        if source == "working":
            trajectories = self._collect_trajectories_from_role("WorkingTrajectoryLines", workflow_node=wf)
            if trajectories:
                return trajectories
            rows = self.trajectory_scene.collect_working_trajectory_rows(
                groups=["imported_rosa", "imported_external", "manual", "guided_fit", "auto_fit"]
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
            "auto_fit": "AutoFitTrajectoryLines",
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
