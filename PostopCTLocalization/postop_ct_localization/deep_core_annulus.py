"""Annulus sampling and volume-geometry helpers for Deep Core.

All volume access goes through the ``VolumeAccessor`` protocol so this
module has no Slicer or VTK imports and is testable with a mock accessor.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class AnnulusSampler:
    """Geometric and CT annulus sampling helpers.

    Constructed with a ``VolumeAccessor`` (see ``deep_core_volume.py``).
    """

    def __init__(self, volume_accessor):
        self._vol = volume_accessor

    # -- volume convenience wrappers ----------------------------------------

    def volume_array_kji(self, volume_node: Any) -> np.ndarray | None:
        if volume_node is None:
            return None
        try:
            return np.asarray(self._vol.array_kji(volume_node), dtype=float)
        except Exception:
            return None

    def ras_to_ijk_fn(self, volume_node: Any):
        if volume_node is None:
            return None
        try:
            return self._vol.ras_to_ijk_fn(volume_node)
        except Exception:
            return None

    # -- pure geometry helpers (no volume access) ---------------------------

    @staticmethod
    def orthonormal_basis_for_axis(axis_ras):
        axis = np.asarray(axis_ras, dtype=float).reshape(3)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-6:
            axis = np.asarray([0.0, 0.0, 1.0], dtype=float)
            axis_norm = 1.0
        axis = axis / axis_norm
        ref = np.asarray([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(axis, ref))) > 0.90:
            ref = np.asarray([0.0, 1.0, 0.0], dtype=float)
        u = np.cross(axis, ref)
        u_norm = float(np.linalg.norm(u))
        if u_norm <= 1e-6:
            ref = np.asarray([0.0, 0.0, 1.0], dtype=float)
            u = np.cross(axis, ref)
            u_norm = float(np.linalg.norm(u))
        u = u / max(u_norm, 1e-6)
        v = np.cross(axis, u)
        v = v / max(float(np.linalg.norm(v)), 1e-6)
        return axis, u, v

    @staticmethod
    def value_percentile_from_sorted(sorted_values, value):
        if sorted_values is None or value is None:
            return None
        arr = np.asarray(sorted_values, dtype=float).reshape(-1)
        if arr.size <= 0:
            return None
        idx = int(np.searchsorted(arr, float(value), side="right"))
        return float(100.0 * idx / max(1, arr.size))

    # -- volume sampling methods -------------------------------------------

    def scan_reference_hu_values(
        self,
        volume_node: Any,
        lower_hu: float = -500.0,
        upper_hu: float | None = 2500.0,
    ):
        arr_kji = self.volume_array_kji(volume_node)
        if arr_kji is None:
            return None
        values = np.asarray(arr_kji, dtype=float).reshape(-1)
        keep = np.isfinite(values) & (values > float(lower_hu))
        if upper_hu is not None:
            keep &= values < float(upper_hu)
        ref = np.asarray(values[keep], dtype=float).reshape(-1)
        if ref.size <= 0:
            return None
        ref.sort()
        return ref

    def depth_at_ras(
        self, volume_node: Any, depth_map_kji: np.ndarray, point_ras
    ):
        fn = self.ras_to_ijk_fn(volume_node)
        if depth_map_kji is None or fn is None:
            return None
        ijk = fn(point_ras)
        i = int(round(float(ijk[0])))
        j = int(round(float(ijk[1])))
        k = int(round(float(ijk[2])))
        if (
            k < 0 or j < 0 or i < 0
            or k >= depth_map_kji.shape[0]
            or j >= depth_map_kji.shape[1]
            or i >= depth_map_kji.shape[2]
        ):
            return None
        val = float(depth_map_kji[k, j, i])
        return val if np.isfinite(val) else None

    def cross_section_annulus_stats_hu(
        self,
        volume_node: Any,
        center_ras,
        axis_ras,
        annulus_inner_mm: float = 3.0,
        annulus_outer_mm: float = 4.0,
        radial_steps: int = 2,
        angular_samples: int = 12,
    ):
        arr_kji = self.volume_array_kji(volume_node)
        fn = self.ras_to_ijk_fn(volume_node)
        if arr_kji is None or fn is None:
            return {"mean_hu": None, "median_hu": None, "sample_count": 0}
        center = np.asarray(
            center_ras if center_ras is not None else [0.0, 0.0, 0.0],
            dtype=float,
        ).reshape(3)
        _axis, u, v = self.orthonormal_basis_for_axis(axis_ras)
        radial_values = np.linspace(
            float(annulus_inner_mm),
            float(annulus_outer_mm),
            int(max(1, radial_steps)),
        )
        angles = np.linspace(
            0.0, 2.0 * np.pi, int(max(4, angular_samples)), endpoint=False
        )
        samples = []
        for radius in radial_values.tolist():
            for theta in angles.tolist():
                offset = float(radius) * np.cos(float(theta)) * u + float(
                    radius
                ) * np.sin(float(theta)) * v
                point_ras = center + offset
                ijk = fn(point_ras)
                i = int(round(float(ijk[0])))
                j = int(round(float(ijk[1])))
                k = int(round(float(ijk[2])))
                if (
                    k < 0 or j < 0 or i < 0
                    or k >= arr_kji.shape[0]
                    or j >= arr_kji.shape[1]
                    or i >= arr_kji.shape[2]
                ):
                    continue
                val = float(arr_kji[k, j, i])
                if np.isfinite(val):
                    samples.append(val)
        if not samples:
            return {"mean_hu": None, "median_hu": None, "sample_count": 0}
        sample_arr = np.asarray(samples, dtype=float)
        return {
            "mean_hu": float(np.mean(sample_arr)),
            "median_hu": float(np.median(sample_arr)),
            "sample_count": int(sample_arr.size),
        }

    def cross_section_annulus_mean_ct_hu(
        self,
        volume_node: Any,
        center_ras,
        axis_ras,
        annulus_inner_mm: float = 3.0,
        annulus_outer_mm: float = 4.0,
        radial_steps: int = 2,
        angular_samples: int = 12,
    ):
        return self.cross_section_annulus_stats_hu(
            volume_node=volume_node,
            center_ras=center_ras,
            axis_ras=axis_ras,
            annulus_inner_mm=annulus_inner_mm,
            annulus_outer_mm=annulus_outer_mm,
            radial_steps=radial_steps,
            angular_samples=angular_samples,
        )


# ---------------------------------------------------------------------------
# Legacy mixin shim
# ---------------------------------------------------------------------------

class DeepCoreAnnulusMixin:
    """Backward-compatible mixin that delegates to an AnnulusSampler.

    During the transition period, the mixin creates a per-call
    ``AnnulusSampler`` using the ``SlicerVolumeAccessor``. Once all callers
    are migrated to the composed pipeline, this class can be removed.
    """

    @staticmethod
    def _orthonormal_basis_for_axis(axis_ras):
        return AnnulusSampler.orthonormal_basis_for_axis(axis_ras)

    @staticmethod
    def _value_percentile_from_sorted(sorted_values, value):
        return AnnulusSampler.value_percentile_from_sorted(sorted_values, value)

    @classmethod
    def _scan_reference_hu_values(cls, volume_node, lower_hu=-500.0, upper_hu=2500.0):
        from .deep_core_volume import SlicerVolumeAccessor
        sampler = AnnulusSampler(SlicerVolumeAccessor())
        return sampler.scan_reference_hu_values(volume_node, lower_hu, upper_hu)

    @staticmethod
    def _volume_array_kji(volume_node):
        from .deep_core_volume import SlicerVolumeAccessor
        sampler = AnnulusSampler(SlicerVolumeAccessor())
        return sampler.volume_array_kji(volume_node)

    @staticmethod
    def _ras_to_ijk_fn_for_volume(volume_node):
        from .deep_core_volume import SlicerVolumeAccessor
        sampler = AnnulusSampler(SlicerVolumeAccessor())
        return sampler.ras_to_ijk_fn(volume_node)

    @classmethod
    def _depth_at_ras_with_volume(cls, volume_node, depth_map_kji, point_ras):
        from .deep_core_volume import SlicerVolumeAccessor
        sampler = AnnulusSampler(SlicerVolumeAccessor())
        return sampler.depth_at_ras(volume_node, depth_map_kji, point_ras)

    @classmethod
    def _cross_section_annulus_stats_hu(
        cls, volume_node, center_ras, axis_ras,
        annulus_inner_mm=3.0, annulus_outer_mm=4.0,
        radial_steps=2, angular_samples=12,
    ):
        from .deep_core_volume import SlicerVolumeAccessor
        sampler = AnnulusSampler(SlicerVolumeAccessor())
        return sampler.cross_section_annulus_stats_hu(
            volume_node, center_ras, axis_ras,
            annulus_inner_mm, annulus_outer_mm,
            radial_steps, angular_samples,
        )

    @classmethod
    def _cross_section_annulus_mean_ct_hu(
        cls, volume_node, center_ras, axis_ras,
        annulus_inner_mm=3.0, annulus_outer_mm=4.0,
        radial_steps=2, angular_samples=12,
    ):
        from .deep_core_volume import SlicerVolumeAccessor
        sampler = AnnulusSampler(SlicerVolumeAccessor())
        return sampler.cross_section_annulus_mean_ct_hu(
            volume_node, center_ras, axis_ras,
            annulus_inner_mm, annulus_outer_mm,
            radial_steps, angular_samples,
        )
