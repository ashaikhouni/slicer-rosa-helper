"""Shared geometry and scene utility helpers for atlas services."""

from __future__ import annotations

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from __main__ import slicer, vtk

from .scene_utils import find_node_by_name


class AtlasUtils:
    """Small utility helpers reused by atlas-specific services."""

    def find_node_by_name(self, node_name, class_name):
        """Return first scene node that matches exact name and class."""
        return find_node_by_name(node_name=node_name, class_name=class_name)

    def vtk_matrix_to_numpy_4x4(self, vtk_matrix):
        """Convert a ``vtkMatrix4x4`` to a NumPy ``(4,4)`` float array."""
        out = np.eye(4, dtype=float)
        for r in range(4):
            for c in range(4):
                out[r, c] = float(vtk_matrix.GetElement(r, c))
        return out

    def volume_ijk_to_world_matrix(self, volume_node):
        """Return voxel-IJK to world-RAS matrix, including any parent transform."""
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
        """Return world-RAS to node-local-RAS transform matrix for one node."""
        if node is None:
            return np.eye(4, dtype=float)
        parent = node.GetParentTransformNode()
        if parent is None:
            return np.eye(4, dtype=float)
        world_to_local_vtk = vtk.vtkMatrix4x4()
        if not parent.GetMatrixTransformFromWorld(world_to_local_vtk):
            return np.eye(4, dtype=float)
        return self.vtk_matrix_to_numpy_4x4(world_to_local_vtk)

    def world_to_node_ras_point_with_matrix(self, world_to_node_matrix, point_world_ras):
        """Map one world-RAS point into node-local RAS using a precomputed matrix."""
        mat = np.eye(4, dtype=float) if world_to_node_matrix is None else world_to_node_matrix
        vec = np.array([float(point_world_ras[0]), float(point_world_ras[1]), float(point_world_ras[2]), 1.0])
        out = mat @ vec
        return [float(out[0]), float(out[1]), float(out[2])]

    def world_to_node_ras_point(self, node, point_world_ras):
        """Map one world-RAS point into node-local RAS for sampling in node space."""
        mat = self.world_to_node_ras_matrix(node)
        return self.world_to_node_ras_point_with_matrix(mat, point_world_ras)

    def show_volume_in_all_slice_views(self, volume_node):
        """Set one scalar volume as background in all slice composite nodes."""
        if volume_node is None:
            return
        volume_id = volume_node.GetID()
        for composite in slicer.util.getNodesByClass("vtkMRMLSliceCompositeNode"):
            composite.SetBackgroundVolumeID(volume_id)
