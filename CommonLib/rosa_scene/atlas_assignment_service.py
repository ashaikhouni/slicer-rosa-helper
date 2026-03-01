"""Atlas contact labeling and assignment table publishing service."""

from __future__ import annotations

import math

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from __main__ import slicer, vtk

from rosa_core import lps_to_ras_point
from rosa_workflow import WorkflowState

from .electrode_scene import ElectrodeSceneService


class AtlasAssignmentService:
    """Assign contacts to atlas labels and publish workflow table rows."""

    def __init__(self, utils, workflow_state=None):
        self.utils = utils
        self.workflow_state = workflow_state or WorkflowState()
        self.electrode_scene = ElectrodeSceneService(workflow_state=self.workflow_state)

    @staticmethod
    def _thomas_nucleus_from_segment_name(name):
        text = (name or "").strip().upper()
        if text.startswith("LEFT_"):
            return text[5:]
        if text.startswith("RIGHT_"):
            return text[6:]
        return text

    def _volume_label_lookup_name(self, volume_node, label_value):
        if volume_node is None:
            return ""
        display = volume_node.GetDisplayNode()
        if display is None:
            return ""
        color_node = display.GetColorNode()
        if color_node is None:
            return ""
        try:
            name = color_node.GetColorName(int(label_value))
        except Exception:
            name = ""
        return str(name) if name else ""

    def _build_volume_label_index(self, volume_node):
        if np is None:
            raise RuntimeError("NumPy is required for atlas assignment.")
        if volume_node is None:
            return None

        arr = slicer.util.arrayFromVolume(volume_node)
        idx = np.argwhere(arr > 0)
        if idx.size == 0:
            return None

        labels = arr[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.int32)
        ijk_h = np.ones((idx.shape[0], 4), dtype=float)
        ijk_h[:, 0] = idx[:, 2].astype(float)
        ijk_h[:, 1] = idx[:, 1].astype(float)
        ijk_h[:, 2] = idx[:, 0].astype(float)

        ijk_to_world = self.utils.volume_ijk_to_world_matrix(volume_node)
        ras = (ijk_h @ ijk_to_world.T)[:, :3]

        points = vtk.vtkPoints()
        points.SetNumberOfPoints(int(ras.shape[0]))
        for i in range(int(ras.shape[0])):
            points.SetPoint(i, float(ras[i, 0]), float(ras[i, 1]), float(ras[i, 2]))
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        locator = vtk.vtkPointLocator()
        locator.SetDataSet(poly)
        locator.BuildLocator()

        centroids = {}
        label_values = np.unique(labels)
        for label_value in label_values:
            xyz = ras[labels == int(label_value)]
            if xyz.size:
                centroids[int(label_value)] = xyz.mean(axis=0)

        label_names = {}
        for value in label_values:
            name = self._volume_label_lookup_name(volume_node, int(value))
            label_names[int(value)] = name or f"Label_{int(value)}"

        return {
            "locator": locator,
            "points": points,
            "labels": labels,
            "centroids": centroids,
            "label_names": label_names,
        }

    def _build_segmentation_label_index(self, segmentation_nodes, reference_volume_node, skip_generic_thalamus=True):
        if np is None:
            raise RuntimeError("NumPy is required for atlas assignment.")
        if not segmentation_nodes:
            return None
        if reference_volume_node is None:
            raise ValueError("Reference volume node is required for segmentation atlas indexing.")
        if not hasattr(slicer.modules, "segmentations"):
            return None

        seg_logic = slicer.modules.segmentations.logic()
        if seg_logic is None:
            return None

        all_points = []
        all_labels = []
        label_names = {}
        label_value = 1

        for seg_node in segmentation_nodes:
            if seg_node is None:
                continue
            segmentation = seg_node.GetSegmentation()
            if segmentation is None:
                continue
            for i in range(segmentation.GetNumberOfSegments()):
                seg_id = segmentation.GetNthSegmentID(i)
                segment = segmentation.GetSegment(seg_id)
                if segment is None:
                    continue
                seg_name = segment.GetName() or ""
                if skip_generic_thalamus and self._thomas_nucleus_from_segment_name(seg_name) == "THALAMUS":
                    continue

                label_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "__tmp_atlas_seg")
                ids = vtk.vtkStringArray()
                ids.InsertNextValue(seg_id)
                ok = seg_logic.ExportSegmentsToLabelmapNode(seg_node, ids, label_node, reference_volume_node)
                if not ok:
                    slicer.mrmlScene.RemoveNode(label_node)
                    continue

                arr = slicer.util.arrayFromVolume(label_node)
                idx = np.argwhere(arr > 0)
                if idx.size == 0:
                    slicer.mrmlScene.RemoveNode(label_node)
                    continue

                ijk_h = np.ones((idx.shape[0], 4), dtype=float)
                ijk_h[:, 0] = idx[:, 2].astype(float)
                ijk_h[:, 1] = idx[:, 1].astype(float)
                ijk_h[:, 2] = idx[:, 0].astype(float)
                ijk_to_ras_vtk = vtk.vtkMatrix4x4()
                label_node.GetIJKToRASMatrix(ijk_to_ras_vtk)
                ijk_to_ras = self.utils.vtk_matrix_to_numpy_4x4(ijk_to_ras_vtk)
                ras = (ijk_h @ ijk_to_ras.T)[:, :3]

                all_points.append(ras)
                all_labels.append(np.full((ras.shape[0],), int(label_value), dtype=np.int32))
                label_names[int(label_value)] = seg_name or f"Segment_{label_value}"
                label_value += 1
                slicer.mrmlScene.RemoveNode(label_node)

        if not all_points:
            if skip_generic_thalamus:
                return self._build_segmentation_label_index(
                    segmentation_nodes=segmentation_nodes,
                    reference_volume_node=reference_volume_node,
                    skip_generic_thalamus=False,
                )
            return None

        ras_all = np.vstack(all_points)
        labels_all = np.concatenate(all_labels)

        points = vtk.vtkPoints()
        points.SetNumberOfPoints(int(ras_all.shape[0]))
        for i in range(int(ras_all.shape[0])):
            points.SetPoint(i, float(ras_all[i, 0]), float(ras_all[i, 1]), float(ras_all[i, 2]))
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        locator = vtk.vtkPointLocator()
        locator.SetDataSet(poly)
        locator.BuildLocator()

        centroids = {}
        for value in np.unique(labels_all):
            xyz = ras_all[labels_all == int(value)]
            if xyz.size:
                centroids[int(value)] = xyz.mean(axis=0)

        return {
            "locator": locator,
            "points": points,
            "labels": labels_all,
            "centroids": centroids,
            "label_names": label_names,
        }

    @staticmethod
    def _query_label_index(index_cache, point_ras):
        if not index_cache:
            return {
                "label": "",
                "label_value": 0,
                "distance_to_voxel_mm": 0.0,
                "distance_to_centroid_mm": 0.0,
            }

        locator = index_cache["locator"]
        point_id = int(locator.FindClosestPoint(float(point_ras[0]), float(point_ras[1]), float(point_ras[2])))
        nearest = [0.0, 0.0, 0.0]
        index_cache["points"].GetPoint(point_id, nearest)
        dx = float(point_ras[0]) - float(nearest[0])
        dy = float(point_ras[1]) - float(nearest[1])
        dz = float(point_ras[2]) - float(nearest[2])
        dv = math.sqrt(dx * dx + dy * dy + dz * dz)

        label_value = int(index_cache["labels"][point_id])
        label_name = index_cache["label_names"].get(label_value, f"Label_{label_value}")
        centroid = index_cache["centroids"].get(label_value)
        dc = 0.0
        if centroid is not None:
            cx = float(point_ras[0]) - float(centroid[0])
            cy = float(point_ras[1]) - float(centroid[1])
            cz = float(point_ras[2]) - float(centroid[2])
            dc = math.sqrt(cx * cx + cy * cy + cz * cz)

        return {
            "label": label_name,
            "label_value": label_value,
            "distance_to_voxel_mm": float(dv),
            "distance_to_centroid_mm": float(dc),
        }

    def assign_contacts_to_atlases(
        self,
        contacts,
        freesurfer_volume_node=None,
        thomas_segmentation_nodes=None,
        wm_volume_node=None,
        reference_volume_node=None,
        prefer_thomas=True,
    ):
        if np is None:
            raise RuntimeError("NumPy is required for atlas assignment.")

        fs_idx = self._build_volume_label_index(freesurfer_volume_node) if freesurfer_volume_node else None
        wm_idx = self._build_volume_label_index(wm_volume_node) if wm_volume_node else None

        th_idx = None
        if thomas_segmentation_nodes:
            th_idx = self._build_segmentation_label_index(thomas_segmentation_nodes, reference_volume_node)

        th_native_node = None
        if thomas_segmentation_nodes:
            for node in thomas_segmentation_nodes:
                if node is not None:
                    th_native_node = node
                    break

        rows = []
        for contact in contacts or []:
            point_ras = lps_to_ras_point(contact.get("position_lps", [0.0, 0.0, 0.0]))
            th_native_ras = self.utils.world_to_node_ras_point(th_native_node, point_ras) if th_native_node is not None else [0.0, 0.0, 0.0]
            fs_native_ras = self.utils.world_to_node_ras_point(freesurfer_volume_node, point_ras) if freesurfer_volume_node is not None else [0.0, 0.0, 0.0]
            wm_native_ras = self.utils.world_to_node_ras_point(wm_volume_node, point_ras) if wm_volume_node is not None else [0.0, 0.0, 0.0]

            th = self._query_label_index(th_idx, point_ras) if th_idx else None
            fs = self._query_label_index(fs_idx, point_ras) if fs_idx else None
            wm = self._query_label_index(wm_idx, point_ras) if wm_idx else None

            choices = []
            if fs is not None and fs.get("label"):
                choices.append(("freesurfer", fs))
            if th is not None and th.get("label"):
                choices.append(("thomas", th))
            if wm is not None and wm.get("label"):
                choices.append(("wm", wm))

            closest_source = ""
            closest = {"label": "", "label_value": 0, "distance_to_voxel_mm": 0.0, "distance_to_centroid_mm": 0.0}
            if choices:
                closest_source, closest = min(choices, key=lambda item: float(item[1].get("distance_to_voxel_mm", float("inf"))))

            primary_source = ""
            primary = {"label": "", "label_value": 0, "distance_to_voxel_mm": 0.0, "distance_to_centroid_mm": 0.0}
            if prefer_thomas and th is not None and th.get("label"):
                primary_source, primary = "thomas", th
            else:
                primary_source, primary = closest_source, closest

            rows.append(
                {
                    "trajectory": contact.get("trajectory", ""),
                    "contact_label": contact.get("label", ""),
                    "contact_index": int(contact.get("index", 0)),
                    "contact_ras": [float(point_ras[0]), float(point_ras[1]), float(point_ras[2])],
                    "closest_source": closest_source,
                    "closest_label": closest.get("label", ""),
                    "closest_label_value": int(closest.get("label_value", 0)),
                    "closest_distance_to_voxel_mm": float(closest.get("distance_to_voxel_mm", 0.0)),
                    "closest_distance_to_centroid_mm": float(closest.get("distance_to_centroid_mm", 0.0)),
                    "primary_source": primary_source,
                    "primary_label": primary.get("label", ""),
                    "primary_label_value": int(primary.get("label_value", 0)),
                    "primary_distance_to_voxel_mm": float(primary.get("distance_to_voxel_mm", 0.0)),
                    "primary_distance_to_centroid_mm": float(primary.get("distance_to_centroid_mm", 0.0)),
                    "thomas_label": "" if th is None else th.get("label", ""),
                    "thomas_label_value": 0 if th is None else int(th.get("label_value", 0)),
                    "thomas_distance_to_voxel_mm": 0.0 if th is None else float(th.get("distance_to_voxel_mm", 0.0)),
                    "thomas_distance_to_centroid_mm": 0.0 if th is None else float(th.get("distance_to_centroid_mm", 0.0)),
                    "freesurfer_label": "" if fs is None else fs.get("label", ""),
                    "freesurfer_label_value": 0 if fs is None else int(fs.get("label_value", 0)),
                    "freesurfer_distance_to_voxel_mm": 0.0 if fs is None else float(fs.get("distance_to_voxel_mm", 0.0)),
                    "freesurfer_distance_to_centroid_mm": 0.0 if fs is None else float(fs.get("distance_to_centroid_mm", 0.0)),
                    "wm_label": "" if wm is None else wm.get("label", ""),
                    "wm_label_value": 0 if wm is None else int(wm.get("label_value", 0)),
                    "wm_distance_to_voxel_mm": 0.0 if wm is None else float(wm.get("distance_to_voxel_mm", 0.0)),
                    "wm_distance_to_centroid_mm": 0.0 if wm is None else float(wm.get("distance_to_centroid_mm", 0.0)),
                    "thomas_native_ras": th_native_ras,
                    "freesurfer_native_ras": fs_native_ras,
                    "wm_native_ras": wm_native_ras,
                }
            )

        rows.sort(key=lambda r: (str(r.get("trajectory", "")), int(r.get("contact_index", 0))))
        return rows

    def publish_atlas_assignment_rows(self, atlas_rows, workflow_node=None):
        wf = workflow_node or self.workflow_state.resolve_or_create_workflow_node()
        columns = [
            "trajectory", "contact_label", "contact_index", "x_ras", "y_ras", "z_ras",
            "closest_source", "closest_label", "closest_label_value", "closest_distance_to_voxel_mm", "closest_distance_to_centroid_mm",
            "primary_source", "primary_label", "primary_label_value", "primary_distance_to_voxel_mm", "primary_distance_to_centroid_mm",
            "thomas_label", "thomas_label_value", "thomas_distance_to_voxel_mm", "thomas_distance_to_centroid_mm",
            "freesurfer_label", "freesurfer_label_value", "freesurfer_distance_to_voxel_mm", "freesurfer_distance_to_centroid_mm",
            "wm_label", "wm_label_value", "wm_distance_to_voxel_mm", "wm_distance_to_centroid_mm",
            "thomas_native_x_ras", "thomas_native_y_ras", "thomas_native_z_ras",
            "freesurfer_native_x_ras", "freesurfer_native_y_ras", "freesurfer_native_z_ras",
            "wm_native_x_ras", "wm_native_y_ras", "wm_native_z_ras",
        ]
        rows = []
        for row in atlas_rows or []:
            p_ras = row.get("contact_ras", [0.0, 0.0, 0.0])
            th_native = row.get("thomas_native_ras", [0.0, 0.0, 0.0])
            fs_native = row.get("freesurfer_native_ras", [0.0, 0.0, 0.0])
            wm_native = row.get("wm_native_ras", [0.0, 0.0, 0.0])
            rows.append(
                {
                    "trajectory": row.get("trajectory", ""),
                    "contact_label": row.get("contact_label", ""),
                    "contact_index": row.get("contact_index", 0),
                    "x_ras": float(p_ras[0]),
                    "y_ras": float(p_ras[1]),
                    "z_ras": float(p_ras[2]),
                    "closest_source": row.get("closest_source", ""),
                    "closest_label": row.get("closest_label", ""),
                    "closest_label_value": row.get("closest_label_value", 0),
                    "closest_distance_to_voxel_mm": row.get("closest_distance_to_voxel_mm", 0.0),
                    "closest_distance_to_centroid_mm": row.get("closest_distance_to_centroid_mm", 0.0),
                    "primary_source": row.get("primary_source", ""),
                    "primary_label": row.get("primary_label", ""),
                    "primary_label_value": row.get("primary_label_value", 0),
                    "primary_distance_to_voxel_mm": row.get("primary_distance_to_voxel_mm", 0.0),
                    "primary_distance_to_centroid_mm": row.get("primary_distance_to_centroid_mm", 0.0),
                    "thomas_label": row.get("thomas_label", ""),
                    "thomas_label_value": row.get("thomas_label_value", 0),
                    "thomas_distance_to_voxel_mm": row.get("thomas_distance_to_voxel_mm", 0.0),
                    "thomas_distance_to_centroid_mm": row.get("thomas_distance_to_centroid_mm", 0.0),
                    "freesurfer_label": row.get("freesurfer_label", ""),
                    "freesurfer_label_value": row.get("freesurfer_label_value", 0),
                    "freesurfer_distance_to_voxel_mm": row.get("freesurfer_distance_to_voxel_mm", 0.0),
                    "freesurfer_distance_to_centroid_mm": row.get("freesurfer_distance_to_centroid_mm", 0.0),
                    "wm_label": row.get("wm_label", ""),
                    "wm_label_value": row.get("wm_label_value", 0),
                    "wm_distance_to_voxel_mm": row.get("wm_distance_to_voxel_mm", 0.0),
                    "wm_distance_to_centroid_mm": row.get("wm_distance_to_centroid_mm", 0.0),
                    "thomas_native_x_ras": float(th_native[0]),
                    "thomas_native_y_ras": float(th_native[1]),
                    "thomas_native_z_ras": float(th_native[2]),
                    "freesurfer_native_x_ras": float(fs_native[0]),
                    "freesurfer_native_y_ras": float(fs_native[1]),
                    "freesurfer_native_z_ras": float(fs_native[2]),
                    "wm_native_x_ras": float(wm_native[0]),
                    "wm_native_y_ras": float(wm_native[1]),
                    "wm_native_z_ras": float(wm_native[2]),
                }
            )

        table = self.electrode_scene.create_or_update_table_node(
            node_name="Rosa_AtlasAssignments",
            columns=columns,
            rows=rows,
        )
        self.workflow_state.set_single_role("AtlasAssignmentTable", table, workflow_node=wf)
        self.workflow_state.tag_node(
            table,
            role="AtlasAssignmentTable",
            source="atlas",
            space="ROSA_BASE",
            signature=table.GetID(),
            workflow_node=wf,
        )
