"""Shared geometry and scene utility helpers for atlas services."""

from __future__ import annotations

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from __main__ import slicer, vtk


class AtlasUtils:
    """Small utility helpers reused by atlas-specific services."""

    def find_node_by_name(self, node_name, class_name):
        for node in slicer.util.getNodesByClass(class_name):
            if node.GetName() == node_name:
                return node
        return None

    def vtk_matrix_to_numpy_4x4(self, vtk_matrix):
        out = np.eye(4, dtype=float)
        for r in range(4):
            for c in range(4):
                out[r, c] = float(vtk_matrix.GetElement(r, c))
        return out

    def volume_ijk_to_world_matrix(self, volume_node):
        ijk_to_ras_vtk = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(ijk_to_ras_vtk)
        ijk_to_world = self.vtk_matrix_to_numpy_4x4(ijk_to_ras_vtk)

        parent = volume_node.GetParentTransformNode()
        if parent is None:
            return ijk_to_world
        parent_to_world_vtk = vtk.vtkMatrix4x4()
        if not parent.GetMatrixTransformToWorld(parent_to_world_vtk):
            return ijk_to_world
        parent_to_world = self.vtk_matrix_to_numpy_4x4(parent_to_world_vtk)
        return parent_to_world @ ijk_to_world

    def world_to_node_ras_matrix(self, node):
        if node is None:
            return np.eye(4, dtype=float)
        parent = node.GetParentTransformNode()
        if parent is None:
            return np.eye(4, dtype=float)
        world_to_local_vtk = vtk.vtkMatrix4x4()
        if not parent.GetMatrixTransformFromWorld(world_to_local_vtk):
            return np.eye(4, dtype=float)
        return self.vtk_matrix_to_numpy_4x4(world_to_local_vtk)

    def world_to_node_ras_point(self, node, point_world_ras):
        mat = self.world_to_node_ras_matrix(node)
        vec = np.array([float(point_world_ras[0]), float(point_world_ras[1]), float(point_world_ras[2]), 1.0])
        out = mat @ vec
        return [float(out[0]), float(out[1]), float(out[2])]

    def show_volume_in_all_slice_views(self, volume_node):
        if volume_node is None:
            return
        volume_id = volume_node.GetID()
        for composite in slicer.util.getNodesByClass("vtkMRMLSliceCompositeNode"):
            composite.SetBackgroundVolumeID(volume_id)
