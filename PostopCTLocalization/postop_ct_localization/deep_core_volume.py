"""Volume access protocol for Deep Core.

This module defines the boundary between Deep Core algorithm code and the
Slicer runtime.  Algorithm stages receive a ``VolumeAccessor`` and never
import ``slicer`` or ``vtk`` directly, which makes them testable with a
pure-numpy mock accessor.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VolumeAccessor(Protocol):
    """Minimal interface for reading CT volume data.

    Algorithm code calls these methods instead of touching Slicer APIs.
    """

    def array_kji(self, volume_node: Any) -> np.ndarray:
        """Return the volume as a float64 numpy array in K,J,I order."""
        ...

    def spacing_xyz(self, volume_node: Any) -> tuple[float, float, float]:
        """Return voxel spacing in (X, Y, Z) millimetres."""
        ...

    def ijk_kji_to_ras_points(
        self, volume_node: Any, ijk_kji: np.ndarray
    ) -> np.ndarray:
        """Convert an (N, 3) array of K,J,I indices to RAS coordinates."""
        ...

    def ras_to_ijk_fn(self, volume_node: Any):
        """Return a callable ``ras_xyz -> ijk_xyz`` for *volume_node*."""
        ...

    def ijk_to_ras_matrix(self, volume_node: Any) -> np.ndarray:
        """Return the 4x4 IJK-to-RAS homogeneous matrix."""
        ...


# ---------------------------------------------------------------------------
# Slicer implementation
# ---------------------------------------------------------------------------

def _vtk_matrix_to_numpy(vtk_matrix4x4) -> np.ndarray:
    """Convert a vtkMatrix4x4 to a (4, 4) numpy array."""
    out = np.eye(4, dtype=float)
    for r in range(4):
        for c in range(4):
            out[r, c] = float(vtk_matrix4x4.GetElement(r, c))
    return out


class SlicerVolumeAccessor:
    """``VolumeAccessor`` backed by the 3D Slicer runtime."""

    def array_kji(self, volume_node: Any) -> np.ndarray:
        from __main__ import slicer  # noqa: deferred import
        return np.asarray(slicer.util.arrayFromVolume(volume_node), dtype=float)

    def spacing_xyz(self, volume_node: Any) -> tuple[float, float, float]:
        s = volume_node.GetSpacing()
        return (float(s[0]), float(s[1]), float(s[2]))

    def ijk_kji_to_ras_points(
        self, volume_node: Any, ijk_kji: np.ndarray
    ) -> np.ndarray:
        idx = np.asarray(ijk_kji, dtype=float).reshape(-1, 3)
        if idx.size == 0:
            return np.empty((0, 3), dtype=float)
        # KJI -> IJK
        ijk = np.zeros_like(idx, dtype=float)
        ijk[:, 0] = idx[:, 2]
        ijk[:, 1] = idx[:, 1]
        ijk[:, 2] = idx[:, 0]
        ijk_h = np.concatenate(
            [ijk, np.ones((ijk.shape[0], 1), dtype=float)], axis=1
        )
        mat = self.ijk_to_ras_matrix(volume_node)
        ras_h = (mat @ ijk_h.T).T
        return ras_h[:, :3]

    def ras_to_ijk_fn(self, volume_node: Any):
        from __main__ import vtk  # noqa: deferred import
        m_vtk = vtk.vtkMatrix4x4()
        volume_node.GetRASToIJKMatrix(m_vtk)
        mat = _vtk_matrix_to_numpy(m_vtk)

        def _ras_to_ijk(ras_xyz):
            ras_h = np.array(
                [float(ras_xyz[0]), float(ras_xyz[1]), float(ras_xyz[2]), 1.0],
                dtype=float,
            )
            return (mat @ ras_h)[:3]

        return _ras_to_ijk

    def ijk_to_ras_matrix(self, volume_node: Any) -> np.ndarray:
        from __main__ import vtk  # noqa: deferred import
        m_vtk = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(m_vtk)
        return _vtk_matrix_to_numpy(m_vtk)

    def extract_threshold_candidates_lps(
        self,
        volume_node: Any,
        threshold: float,
        head_mask_threshold_hu: float = -500.0,
        min_metal_depth_mm: float = 5.0,
        max_metal_depth_mm: float = 220.0,
        head_mask_method: str = "outside_air",
    ) -> dict[str, Any]:
        """Extract metal candidate points from a CT volume.

        This was formerly on DeepCoreVisualizationLogicMixin; it is algorithm
        logic that belongs on the volume accessor.
        """
        from shank_core.masking import build_preview_masks

        arr = self.array_kji(volume_node)
        spacing_xyz = self.spacing_xyz(volume_node)
        used_threshold = float(threshold)
        best_preview: dict | None = None
        best_count = -1
        while True:
            preview = build_preview_masks(
                arr_kji=np.asarray(arr, dtype=float),
                spacing_xyz=spacing_xyz,
                threshold=float(used_threshold),
                use_head_mask=True,
                build_head_mask=True,
                head_mask_threshold_hu=float(head_mask_threshold_hu),
                head_mask_method=str(head_mask_method),
                head_gate_erode_vox=1,
                head_gate_dilate_vox=1,
                head_gate_margin_mm=0.0,
                min_metal_depth_mm=float(min_metal_depth_mm),
                max_metal_depth_mm=float(max_metal_depth_mm),
                include_debug_masks=False,
            )
            count = int(preview.get("depth_kept_count") or 0)
            if count > best_count:
                best_preview = preview
                best_count = count
            if count >= 300000 or used_threshold <= 500.0 + 1e-6:
                break
            used_threshold = max(500.0, used_threshold - 50.0)
        preview = best_preview if best_preview is not None else {}
        idx = np.argwhere(
            np.asarray(preview.get("metal_depth_pass_mask_kji"), dtype=bool)
        )
        if idx.size == 0:
            return {
                "points_lps": np.empty((0, 3), dtype=float),
                "threshold_hu": float(used_threshold),
            }
        n = idx.shape[0]
        ijk_h = np.ones((n, 4), dtype=float)
        ijk_h[:, 0] = idx[:, 2]
        ijk_h[:, 1] = idx[:, 1]
        ijk_h[:, 2] = idx[:, 0]
        mat = self.ijk_to_ras_matrix(volume_node)
        ras = (ijk_h @ mat.T)[:, :3]
        lps = ras.copy()
        lps[:, 0] *= -1.0
        lps[:, 1] *= -1.0
        return {"points_lps": lps, "threshold_hu": float(used_threshold)}

    def update_scalar_volume(
        self,
        reference_volume_node: Any,
        node_name: str,
        array_kji: np.ndarray,
    ) -> Any:
        """Create or update a scalar volume node for debug visualization.

        Returns the MRML node.
        """
        from __main__ import slicer, vtk  # noqa: deferred import

        scene = slicer.mrmlScene
        node = None
        for n in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
            if n.GetName() == str(node_name):
                node = n
                break
        if node is None:
            node = scene.AddNewNodeByClass(
                "vtkMRMLScalarVolumeNode", str(node_name)
            )
        # Copy geometry from reference
        m = vtk.vtkMatrix4x4()
        reference_volume_node.GetIJKToRASMatrix(m)
        node.SetIJKToRASMatrix(m)
        slicer.util.updateVolumeFromArray(node, np.asarray(array_kji))
        node.Modified()
        return node
