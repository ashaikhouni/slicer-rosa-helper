"""Unit tests for :mod:`deep_core_axis_reconstruction` helpers.

These tests use synthetic volumes (identity RAS↔IJK transform, 1mm
isotropic spacing) and synthetic point clouds. They verify the helper
contracts in isolation; integration with the real pipeline is tested
in ``test_pipeline_dataset.py``.
"""

from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "PostopCTLocalization"))

from postop_ct_localization import deep_core_axis_reconstruction as axr


def _identity_ras_to_ijk(ras_xyz):
    """1mm isotropic identity transform (RAS xyz ↔ IJK xyz)."""
    return np.asarray(ras_xyz, dtype=float).reshape(3)


@dataclass
class _Cfg:
    # Axis refinement
    axis_fit_max_residual_mm: float = 1.2
    # Reabsorption
    reabsorb_radial_tol_mm: float = 1.5
    reabsorb_angle_tol_deg: float = 5.0
    reabsorb_axial_window_mm: float = 5.0
    # Extension
    extension_step_mm: float = 0.5
    extension_tube_radius_mm: float = 1.5
    extension_max_gap_mm: float = 3.0
    extension_termination_gap_mm: float = 5.0
    extension_head_distance_floor_mm: float = -1.0
    # HU classification
    lateral_hu_ring_radius_mm: float = 3.5
    lateral_hu_ring_samples: int = 8
    hu_air_max: float = -500.0
    hu_brain_max: float = 150.0
    hu_bone_max: float = 1800.0
    hu_classification_smoothing: int = 3


def _make_line_cloud(start, direction, length_mm, n_points, jitter=0.0, rng=None):
    rng = rng or np.random.default_rng(0)
    t = np.linspace(0.0, float(length_mm), int(n_points))
    d = np.asarray(direction, dtype=float).reshape(3)
    d = d / np.linalg.norm(d)
    base = np.asarray(start, dtype=float).reshape(1, 3) + t.reshape(-1, 1) * d.reshape(1, 3)
    if jitter > 0.0:
        base = base + rng.normal(scale=jitter, size=base.shape)
    return base


class RefineAxisTests(unittest.TestCase):
    def test_straight_line_recovers_axis(self):
        pts = _make_line_cloud([10.0, 20.0, 30.0], [0.0, 0.0, 1.0], 40.0, 81, jitter=0.05)
        fit = axr.refine_axis_from_cloud(pts, seed_axis=[0.0, 0.0, 1.0])
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(abs(float(fit.axis[2])), 1.0, places=2)
        self.assertLess(fit.residual_rms_mm, 0.2)
        self.assertAlmostEqual(fit.t_max - fit.t_min, 40.0, delta=0.3)
        self.assertEqual(fit.point_count, 81)

    def test_tilted_line(self):
        direction = np.array([1.0, 1.0, 2.0])
        direction = direction / np.linalg.norm(direction)
        pts = _make_line_cloud([5.0, 5.0, 5.0], direction, 30.0, 61)
        fit = axr.refine_axis_from_cloud(pts, seed_axis=direction)
        self.assertIsNotNone(fit)
        cos = abs(float(fit.axis @ direction))
        self.assertGreater(cos, 0.999)

    def test_empty_cloud_returns_none(self):
        self.assertIsNone(axr.refine_axis_from_cloud(np.zeros((0, 3))))


class ReabsorbTests(unittest.TestCase):
    def _fit(self):
        return axr.AxisFit(
            center=np.array([0.0, 0.0, 10.0]),
            axis=np.array([0.0, 0.0, 1.0]),
            residual_rms_mm=0.05,
            residual_median_mm=0.05,
            elongation=50.0,
            t_min=-10.0,
            t_max=10.0,
            point_count=40,
        )

    def test_colinear_atom_absorbed(self):
        fit = self._fit()
        atom = {
            "atom_id": 7,
            "kind": "line",
            "axis_reliable": True,
            "center_ras": [0.4, 0.2, 12.0],  # on-axis, inside axial window
            "axis_ras": [0.0, 0.0, 1.0],
            "support_points_ras": [],
        }
        ids = axr.reabsorb_colinear_atoms(fit, [atom], _Cfg())
        self.assertEqual(ids, [7])

    def test_off_axis_rejected(self):
        fit = self._fit()
        atom = {
            "atom_id": 8,
            "kind": "line",
            "axis_reliable": True,
            "center_ras": [3.0, 0.0, 12.0],  # 3mm off axis
            "axis_ras": [0.0, 0.0, 1.0],
        }
        ids = axr.reabsorb_colinear_atoms(fit, [atom], _Cfg())
        self.assertEqual(ids, [])

    def test_misaligned_axis_rejected(self):
        fit = self._fit()
        atom = {
            "atom_id": 9,
            "kind": "line",
            "axis_reliable": True,
            "center_ras": [0.0, 0.0, 12.0],
            "axis_ras": [1.0, 0.0, 0.0],  # orthogonal
        }
        ids = axr.reabsorb_colinear_atoms(fit, [atom], _Cfg())
        self.assertEqual(ids, [])

    def test_already_absorbed_skipped(self):
        fit = self._fit()
        atom = {
            "atom_id": 7,
            "kind": "line",
            "axis_reliable": True,
            "center_ras": [0.0, 0.0, 12.0],
            "axis_ras": [0.0, 0.0, 1.0],
        }
        ids = axr.reabsorb_colinear_atoms(fit, [atom], _Cfg(), already_absorbed_ids={7})
        self.assertEqual(ids, [])


class MetalProfileTests(unittest.TestCase):
    def test_profile_covers_range(self):
        pts = _make_line_cloud([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 20.0, 41)
        center = np.array([0.0, 0.0, 10.0])
        axis = np.array([0.0, 0.0, 1.0])
        t_min, t_max, profile, step = axr.build_metal_profile(pts, axis, center, step_mm=0.5)
        self.assertAlmostEqual(t_max - t_min, 20.0, delta=0.1)
        self.assertEqual(step, 0.5)
        self.assertGreater(int(profile.sum()), 30)


class ExtendAxisTests(unittest.TestCase):
    def _identity_fit(self, t_min=5.0, t_max=15.0):
        return axr.AxisFit(
            center=np.array([20.0, 20.0, 0.0]),
            axis=np.array([0.0, 0.0, 1.0]),
            residual_rms_mm=0.05,
            residual_median_mm=0.05,
            elongation=50.0,
            t_min=t_min,
            t_max=t_max,
            point_count=20,
        )

    def test_extends_into_metal_rind(self):
        # Mask: a vertical metal rod at (i=20, j=20) from k=0 to k=40.
        mask = np.zeros((50, 50, 50), dtype=bool)
        mask[0:40, 18:23, 18:23] = True
        # Head distance: positive everywhere (inside hull).
        hd = np.full_like(mask, 10.0, dtype=float)
        fit = self._identity_fit(t_min=5.0, t_max=15.0)
        t_deep, t_shallow = axr.extend_axis_along_mask(
            fit, mask, hd, _identity_ras_to_ijk, _Cfg()
        )
        # The fit center is at z=0; axis is +z. Metal runs k∈[0,40)
        # → RAS z∈[0,40). So walking from t_min=5 deep (-z) should
        # stop near z=0, and shallow (+z) should extend past 15 toward z=39.
        self.assertLess(t_deep, 2.0)
        self.assertGreater(t_shallow, 30.0)

    def test_head_distance_floor_stops_extension(self):
        mask = np.ones((50, 50, 50), dtype=bool)
        # Head distance: drops below floor beyond z=20.
        hd = np.full((50, 50, 50), 5.0, dtype=float)
        hd[21:, :, :] = -5.0
        fit = self._identity_fit(t_min=5.0, t_max=15.0)
        _, t_shallow = axr.extend_axis_along_mask(
            fit, mask, hd, _identity_ras_to_ijk, _Cfg()
        )
        self.assertLess(t_shallow, 22.0)

    def test_no_metal_no_extension(self):
        mask = np.zeros((50, 50, 50), dtype=bool)
        hd = np.full_like(mask, 10.0, dtype=float)
        fit = self._identity_fit(t_min=5.0, t_max=15.0)
        t_deep, t_shallow = axr.extend_axis_along_mask(
            fit, mask, hd, _identity_ras_to_ijk, _Cfg()
        )
        self.assertAlmostEqual(t_deep, 5.0, places=5)
        self.assertAlmostEqual(t_shallow, 15.0, places=5)


class HUClassificationTests(unittest.TestCase):
    def _axis_fit(self):
        return axr.AxisFit(
            center=np.array([25.0, 25.0, 25.0]),
            axis=np.array([0.0, 0.0, 1.0]),
            residual_rms_mm=0.05,
            residual_median_mm=0.05,
            elongation=50.0,
            t_min=-10.0,
            t_max=10.0,
            point_count=20,
        )

    def test_brain_to_bone_interface(self):
        arr = np.full((60, 60, 60), 30.0, dtype=float)  # brain HU
        # arr is kji-indexed: axes[0]=k (IJK z). Bone at high z → large k.
        arr[40:, :, :] = 1000.0
        fit = self._axis_fit()
        t_values = np.arange(-10.0, 20.0, 0.5)
        classes = axr.classify_tissue_along_axis(fit, t_values, arr, _identity_ras_to_ijk, _Cfg())
        t_interface = axr.find_bone_brain_interface(classes, t_values)
        self.assertIsNotNone(t_interface)
        # Brain covers z∈[0,40), bone z>=40. Shallowest brain sample
        # is the last t before bone → t ≈ 15 - step.
        self.assertAlmostEqual(float(t_interface), 14.5, delta=1.0)

    def test_no_bone_returns_shallowest_brain(self):
        arr = np.full((60, 60, 60), 30.0, dtype=float)
        fit = self._axis_fit()
        t_values = np.arange(-10.0, 10.0, 0.5)
        classes = axr.classify_tissue_along_axis(fit, t_values, arr, _identity_ras_to_ijk, _Cfg())
        t_interface = axr.find_bone_brain_interface(classes, t_values)
        self.assertIsNotNone(t_interface)
        self.assertAlmostEqual(float(t_interface), float(t_values[-1]), delta=0.01)

    def test_all_bone_returns_none(self):
        arr = np.full((60, 60, 60), 1000.0, dtype=float)
        fit = self._axis_fit()
        t_values = np.arange(-10.0, 10.0, 0.5)
        classes = axr.classify_tissue_along_axis(fit, t_values, arr, _identity_ras_to_ijk, _Cfg())
        self.assertIsNone(axr.find_bone_brain_interface(classes, t_values))

    def test_brain_span_mm(self):
        classes = np.array([axr.BRAIN] * 20 + [axr.BONE] * 5 + [axr.BRAIN] * 10, dtype=np.int8)
        span = axr.intracranial_brain_span_mm(classes, step_mm=0.5)
        self.assertAlmostEqual(span, 10.0)


class LibrarySpanTests(unittest.TestCase):
    _LIBRARY = [
        {"id": "DIXI-5AM", "total_exploration_length_mm": 16.0},
        {"id": "DIXI-8AM", "total_exploration_length_mm": 26.5},
        {"id": "DIXI-15CM", "total_exploration_length_mm": 68.0},
    ]

    def test_picks_closest(self):
        mid, delta = axr.library_span_match(27.0, self._LIBRARY, tolerance_mm=5.0)
        self.assertEqual(mid, "DIXI-8AM")
        self.assertLess(delta, 1.0)

    def test_out_of_range(self):
        mid, delta = axr.library_span_match(200.0, self._LIBRARY, tolerance_mm=5.0)
        self.assertEqual(mid, "DIXI-15CM")
        self.assertGreater(delta, 100.0)

    def test_range(self):
        lo, hi = axr.library_span_range(self._LIBRARY)
        self.assertEqual((lo, hi), (16.0, 68.0))


if __name__ == "__main__":
    unittest.main()
