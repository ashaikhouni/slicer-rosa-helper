"""Concrete atlas-provider implementations for contact labeling."""

from __future__ import annotations

import math
from typing import Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from __main__ import slicer, vtk

from .atlas_provider_types import AtlasSampleResult


def _query_index(index_cache, point_ras):
    if not index_cache:
        return None

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
        "label": str(label_name),
        "label_value": int(label_value),
        "distance_to_voxel_mm": float(dv),
        "distance_to_centroid_mm": float(dc),
    }


class VolumeLabelAtlasProvider:
    """Atlas provider for labeled scalar volumes (FS parcellation, WM, etc.)."""

    def __init__(self, source_id, display_name, volume_node, utils):
        self.source_id = str(source_id)
        self.display_name = str(display_name)
        self.volume_node = volume_node
        self.utils = utils
        self.world_to_native_matrix = self.utils.world_to_node_ras_matrix(self.volume_node) if self.volume_node is not None else None
        self.index_cache = self._build_index(volume_node)

    def is_ready(self):
        return self.index_cache is not None and self.volume_node is not None

    def _volume_label_lookup_name(self, label_value):
        if self.volume_node is None:
            return ""
        display = self.volume_node.GetDisplayNode()
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

    def _build_index(self, volume_node):
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
            name = self._volume_label_lookup_name(int(value))
            label_names[int(value)] = name or f"Label_{int(value)}"

        return {
            "locator": locator,
            "points": points,
            "labels": labels,
            "centroids": centroids,
            "label_names": label_names,
        }

    def sample_contact(self, point_world_ras: Sequence[float]) -> AtlasSampleResult | None:
        if not self.is_ready():
            return None
        q = _query_index(self.index_cache, point_world_ras)
        if q is None:
            return None
        native_ras = self.utils.world_to_node_ras_point_with_matrix(self.world_to_native_matrix, point_world_ras)
        return {
            "source": self.source_id,
            "label": q.get("label", ""),
            "label_value": int(q.get("label_value", 0)),
            "distance_to_voxel_mm": float(q.get("distance_to_voxel_mm", 0.0)),
            "distance_to_centroid_mm": float(q.get("distance_to_centroid_mm", 0.0)),
            "native_ras": [float(native_ras[0]), float(native_ras[1]), float(native_ras[2])],
        }


class ThomasSegmentationAtlasProvider:
    """Atlas provider for THOMAS segmentation nodes."""

    def __init__(self, segmentation_nodes, reference_volume_node, utils):
        self.source_id = "thomas"
        self.display_name = "THOMAS"
        self.segmentation_nodes = list(segmentation_nodes or [])
        self.reference_volume_node = reference_volume_node
        self.utils = utils
        self.native_node = next((node for node in self.segmentation_nodes if node is not None), None)
        self.world_to_native_matrix = self.utils.world_to_node_ras_matrix(self.native_node) if self.native_node is not None else None
        self.index_cache = self._build_index(skip_generic_thalamus=True)

    def is_ready(self):
        return self.index_cache is not None and self.native_node is not None

    @staticmethod
    def _thomas_nucleus_from_segment_name(name):
        text = (name or "").strip().upper()
        if text.startswith("LEFT_"):
            return text[5:]
        if text.startswith("RIGHT_"):
            return text[6:]
        return text

    def _build_index(self, skip_generic_thalamus=True):
        if np is None:
            raise RuntimeError("NumPy is required for atlas assignment.")
        if not self.segmentation_nodes:
            return None
        if self.reference_volume_node is None:
            raise ValueError("Reference volume node is required for THOMAS assignment.")
        if not hasattr(slicer.modules, "segmentations"):
            return None

        seg_logic = slicer.modules.segmentations.logic()
        if seg_logic is None:
            return None

        all_points = []
        all_labels = []
        label_names = {}
        label_value = 1

        for seg_node in self.segmentation_nodes:
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
                ok = seg_logic.ExportSegmentsToLabelmapNode(seg_node, ids, label_node, self.reference_volume_node)
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
                return self._build_index(skip_generic_thalamus=False)
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

    def sample_contact(self, point_world_ras: Sequence[float]) -> AtlasSampleResult | None:
        if not self.is_ready():
            return None
        q = _query_index(self.index_cache, point_world_ras)
        if q is None:
            return None
        native_ras = self.utils.world_to_node_ras_point_with_matrix(self.world_to_native_matrix, point_world_ras)
        return {
            "source": self.source_id,
            "label": q.get("label", ""),
            "label_value": int(q.get("label_value", 0)),
            "distance_to_voxel_mm": float(q.get("distance_to_voxel_mm", 0.0)),
            "distance_to_centroid_mm": float(q.get("distance_to_centroid_mm", 0.0)),
            "native_ras": [float(native_ras[0]), float(native_ras[1]), float(native_ras[2])],
        }
