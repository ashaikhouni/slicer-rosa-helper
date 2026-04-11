"""Tests for VolumeAccessor protocol and implementations."""

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))
sys.path.insert(0, str(REPO_ROOT / "PostopCTLocalization"))

from postop_ct_localization.deep_core_volume import VolumeAccessor


class MockVolumeAccessor:
    """Minimal accessor that satisfies the protocol."""

    def __init__(self, arr_kji, spacing=(1.0, 1.0, 1.0)):
        self._arr = np.asarray(arr_kji, dtype=float)
        self._spacing = spacing

    def array_kji(self, volume_node):
        return self._arr.copy()

    def spacing_xyz(self, volume_node):
        return self._spacing

    def ijk_kji_to_ras_points(self, volume_node, ijk_kji):
        idx = np.asarray(ijk_kji, dtype=float).reshape(-1, 3)
        ras = np.zeros_like(idx)
        ras[:, 0] = idx[:, 2]
        ras[:, 1] = idx[:, 1]
        ras[:, 2] = idx[:, 0]
        return ras

    def ras_to_ijk_fn(self, volume_node):
        def _fn(ras):
            return np.array([float(ras[0]), float(ras[1]), float(ras[2])])
        return _fn

    def ijk_to_ras_matrix(self, volume_node):
        return np.eye(4, dtype=float)


class TestProtocolCompliance(unittest.TestCase):
    def test_mock_satisfies_protocol(self):
        accessor = MockVolumeAccessor(np.zeros((3, 3, 3)))
        self.assertIsInstance(accessor, VolumeAccessor)

    def test_missing_method_fails_protocol(self):
        class Incomplete:
            def array_kji(self, v):
                return np.zeros((3, 3, 3))
        self.assertNotIsInstance(Incomplete(), VolumeAccessor)


class TestMockAccessorBehavior(unittest.TestCase):
    def test_array_kji(self):
        arr = np.arange(27, dtype=float).reshape(3, 3, 3)
        accessor = MockVolumeAccessor(arr)
        result = accessor.array_kji("dummy")
        np.testing.assert_array_equal(result, arr)
        # Should return a copy
        result[0, 0, 0] = 999
        self.assertNotEqual(accessor.array_kji("dummy")[0, 0, 0], 999)

    def test_spacing(self):
        accessor = MockVolumeAccessor(np.zeros((3, 3, 3)), spacing=(0.5, 0.5, 1.0))
        self.assertEqual(accessor.spacing_xyz("dummy"), (0.5, 0.5, 1.0))

    def test_ijk_kji_to_ras(self):
        accessor = MockVolumeAccessor(np.zeros((3, 3, 3)))
        kji = np.array([[2, 1, 0]], dtype=float)  # K=2, J=1, I=0
        ras = accessor.ijk_kji_to_ras_points("dummy", kji)
        # With identity: I->X, J->Y, K->Z
        np.testing.assert_allclose(ras, [[0, 1, 2]])

    def test_ijk_kji_to_ras_empty(self):
        accessor = MockVolumeAccessor(np.zeros((3, 3, 3)))
        kji = np.empty((0, 3), dtype=float)
        ras = accessor.ijk_kji_to_ras_points("dummy", kji)
        self.assertEqual(ras.shape, (0, 3))


if __name__ == "__main__":
    unittest.main()
