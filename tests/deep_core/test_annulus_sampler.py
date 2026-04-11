"""Tests for AnnulusSampler (pure Python, no Slicer dependency)."""

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))
sys.path.insert(0, str(REPO_ROOT / "PostopCTLocalization"))

from postop_ct_localization.deep_core_annulus import AnnulusSampler


class MockVolumeAccessor:
    """Pure-numpy VolumeAccessor for testing."""

    def __init__(self, arr_kji, spacing_xyz=(1.0, 1.0, 1.0)):
        self._arr = np.asarray(arr_kji, dtype=float)
        self._spacing = tuple(float(s) for s in spacing_xyz)

    def array_kji(self, volume_node):
        return self._arr.copy()

    def spacing_xyz(self, volume_node):
        return self._spacing

    def ijk_kji_to_ras_points(self, volume_node, ijk_kji):
        # Identity transform: KJI -> RAS as (I, J, K)
        idx = np.asarray(ijk_kji, dtype=float).reshape(-1, 3)
        ras = np.zeros_like(idx)
        ras[:, 0] = idx[:, 2]  # I -> X (RAS)
        ras[:, 1] = idx[:, 1]  # J -> Y (RAS)
        ras[:, 2] = idx[:, 0]  # K -> Z (RAS)
        return ras

    def ras_to_ijk_fn(self, volume_node):
        def _ras_to_ijk(ras_xyz):
            return np.array(
                [float(ras_xyz[0]), float(ras_xyz[1]), float(ras_xyz[2])],
                dtype=float,
            )
        return _ras_to_ijk

    def ijk_to_ras_matrix(self, volume_node):
        return np.eye(4, dtype=float)


class TestOrthonormalBasis(unittest.TestCase):
    def test_z_axis(self):
        axis, u, v = AnnulusSampler.orthonormal_basis_for_axis([0, 0, 1])
        np.testing.assert_allclose(axis, [0, 0, 1], atol=1e-10)
        self.assertAlmostEqual(float(np.dot(axis, u)), 0.0, places=10)
        self.assertAlmostEqual(float(np.dot(axis, v)), 0.0, places=10)
        self.assertAlmostEqual(float(np.dot(u, v)), 0.0, places=10)
        self.assertAlmostEqual(float(np.linalg.norm(u)), 1.0, places=10)
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=10)

    def test_x_axis(self):
        axis, u, v = AnnulusSampler.orthonormal_basis_for_axis([1, 0, 0])
        np.testing.assert_allclose(axis, [1, 0, 0], atol=1e-10)
        self.assertAlmostEqual(float(np.dot(axis, u)), 0.0, places=10)

    def test_diagonal_axis(self):
        axis, u, v = AnnulusSampler.orthonormal_basis_for_axis([1, 1, 1])
        self.assertAlmostEqual(float(np.linalg.norm(axis)), 1.0, places=10)
        self.assertAlmostEqual(float(np.dot(axis, u)), 0.0, places=10)
        self.assertAlmostEqual(float(np.dot(axis, v)), 0.0, places=10)

    def test_zero_axis_fallback(self):
        axis, u, v = AnnulusSampler.orthonormal_basis_for_axis([0, 0, 0])
        np.testing.assert_allclose(axis, [0, 0, 1], atol=1e-10)


class TestValuePercentile(unittest.TestCase):
    def test_basic(self):
        sorted_values = np.array([10, 20, 30, 40, 50], dtype=float)
        self.assertAlmostEqual(
            AnnulusSampler.value_percentile_from_sorted(sorted_values, 25.0),
            40.0,
        )

    def test_below_all(self):
        sorted_values = np.array([10, 20, 30], dtype=float)
        self.assertAlmostEqual(
            AnnulusSampler.value_percentile_from_sorted(sorted_values, 5.0),
            0.0,
        )

    def test_above_all(self):
        sorted_values = np.array([10, 20, 30], dtype=float)
        self.assertAlmostEqual(
            AnnulusSampler.value_percentile_from_sorted(sorted_values, 100.0),
            100.0,
        )

    def test_none_inputs(self):
        self.assertIsNone(AnnulusSampler.value_percentile_from_sorted(None, 5.0))
        self.assertIsNone(AnnulusSampler.value_percentile_from_sorted([1, 2], None))


class TestScanReferenceHU(unittest.TestCase):
    def test_filters_by_range(self):
        # 10x10x10 volume with known values
        arr = np.full((10, 10, 10), 100.0, dtype=float)
        arr[0, 0, 0] = -600.0  # below lower_hu, should be excluded
        arr[1, 1, 1] = 3000.0  # above upper_hu, should be excluded

        accessor = MockVolumeAccessor(arr)
        sampler = AnnulusSampler(accessor)
        ref = sampler.scan_reference_hu_values("dummy_node", lower_hu=-500.0, upper_hu=2500.0)
        self.assertIsNotNone(ref)
        # All 100.0 values should be kept (1000 - 2 = 998)
        self.assertEqual(ref.shape[0], 998)
        np.testing.assert_allclose(ref, 100.0)

    def test_empty_volume(self):
        arr = np.full((5, 5, 5), -700.0, dtype=float)  # all below threshold
        accessor = MockVolumeAccessor(arr)
        sampler = AnnulusSampler(accessor)
        ref = sampler.scan_reference_hu_values("dummy_node", lower_hu=-500.0)
        self.assertIsNone(ref)


class TestDepthAtRAS(unittest.TestCase):
    def test_in_bounds(self):
        depth_map = np.arange(27, dtype=float).reshape(3, 3, 3)
        accessor = MockVolumeAccessor(np.zeros((3, 3, 3)))
        sampler = AnnulusSampler(accessor)
        # RAS (1, 1, 1) -> IJK (1, 1, 1) -> KJI [1, 1, 1] -> value = 13
        val = sampler.depth_at_ras("dummy", depth_map, [1.0, 1.0, 1.0])
        self.assertAlmostEqual(val, 13.0)

    def test_out_of_bounds(self):
        depth_map = np.zeros((3, 3, 3), dtype=float)
        accessor = MockVolumeAccessor(np.zeros((3, 3, 3)))
        sampler = AnnulusSampler(accessor)
        val = sampler.depth_at_ras("dummy", depth_map, [10.0, 10.0, 10.0])
        self.assertIsNone(val)


class TestCrossSectionAnnulus(unittest.TestCase):
    def test_uniform_volume(self):
        arr = np.full((20, 20, 20), 500.0, dtype=float)
        accessor = MockVolumeAccessor(arr)
        sampler = AnnulusSampler(accessor)
        result = sampler.cross_section_annulus_stats_hu(
            "dummy",
            center_ras=[10, 10, 10],
            axis_ras=[0, 0, 1],
            annulus_inner_mm=2.0,
            annulus_outer_mm=3.0,
            radial_steps=2,
            angular_samples=8,
        )
        self.assertIsNotNone(result["mean_hu"])
        self.assertAlmostEqual(result["mean_hu"], 500.0)
        self.assertGreater(result["sample_count"], 0)


if __name__ == "__main__":
    unittest.main()
