"""Tests for the small standalone helpers: ``cc_volume`` and ``snap``.

These don't run a full pipeline — they exercise the helpers in isolation
on tiny synthetic volumes.
"""
from __future__ import annotations

import unittest

import numpy as np

from rosa_core.contact_placement import (
    slot_cc_volume_mm3,
    snap_centerline_to_centroid,
)


class SlotCcVolumeMm3Tests(unittest.TestCase):
    def setUp(self):
        # 16×16×16 volume, identity 1mm IJK→RAS.
        self.r2i = np.eye(4)

    def test_no_metal_voxels_returns_zero(self):
        arr = np.zeros((16, 16, 16), dtype=np.float32)
        v = slot_cc_volume_mm3(
            arr, self.r2i, np.array([8, 8, 8], dtype=float),
            spacing_xyz=(1.0, 1.0, 1.0), hu_threshold=1500.0, roi_half_mm=5.0,
        )
        self.assertEqual(v, 0.0)

    def test_isolated_voxel_returns_one_voxel_volume(self):
        arr = np.zeros((16, 16, 16), dtype=np.float32)
        arr[8, 8, 8] = 2500.0
        v = slot_cc_volume_mm3(
            arr, self.r2i, np.array([8, 8, 8], dtype=float),
            spacing_xyz=(1.0, 1.0, 1.0), hu_threshold=1500.0, roi_half_mm=5.0,
        )
        self.assertEqual(v, 1.0)  # 1 voxel × 1 mm³

    def test_3x3x3_block_returns_27_voxel_volume(self):
        arr = np.zeros((16, 16, 16), dtype=np.float32)
        arr[7:10, 7:10, 7:10] = 2500.0
        v = slot_cc_volume_mm3(
            arr, self.r2i, np.array([8, 8, 8], dtype=float),
            spacing_xyz=(1.0, 1.0, 1.0), hu_threshold=1500.0, roi_half_mm=5.0,
        )
        self.assertEqual(v, 27.0)

    def test_anisotropic_spacing_scales_volume(self):
        # 1×1 voxel at 0.5×0.5×2 mm → 0.5 mm³.
        arr = np.zeros((16, 16, 16), dtype=np.float32)
        arr[8, 8, 8] = 2500.0
        v = slot_cc_volume_mm3(
            arr, self.r2i, np.array([8, 8, 8], dtype=float),
            spacing_xyz=(0.5, 0.5, 2.0), hu_threshold=1500.0, roi_half_mm=5.0,
        )
        self.assertAlmostEqual(v, 0.5, places=6)

    def test_two_disconnected_blocks_only_returns_one(self):
        # Slot is in the first block; the second block is unreachable from
        # the slot and should be excluded by the CC label.
        arr = np.zeros((24, 16, 16), dtype=np.float32)
        arr[3:5, 7:10, 7:10] = 2500.0    # block 1: 18 voxels
        arr[18:21, 7:10, 7:10] = 2500.0  # block 2: 27 voxels (far away)
        v = slot_cc_volume_mm3(
            arr, self.r2i, np.array([8, 4, 8], dtype=float),  # in block 1
            spacing_xyz=(1.0, 1.0, 1.0), hu_threshold=1500.0, roi_half_mm=5.0,
        )
        self.assertEqual(v, 18.0)


class SnapCenterlineToCentroidTests(unittest.TestCase):
    """Snap pulls the centerline toward LoG-bright voxels.

    Builds a thin tube of bright LoG voxels offset 1mm in y from the input
    centerline; verifies the snap moves the centerline closer to it.
    """

    def setUp(self):
        # 32×16×8 volume. LoG-bright tube along axis at y=4 (offset +1 from
        # centerline at y=3).
        self.shape = (8, 16, 32)
        self.log = np.zeros(self.shape, dtype=np.float32)
        # Set the LoG to a strong NEGATIVE value at the offset tube — the
        # snap negates LoG when computing the centroid weight, so negative-
        # LoG voxels read as "metal-bright".
        for x in range(2, 30):
            self.log[4, 4, x] = -1500.0  # bright tube at y=4 (centerline at y=3)
        self.r2i = np.eye(4)
        # Centerline at y=3.
        self.centerline = np.array([[2, 3, 4], [29, 3, 4]], dtype=float)

    def test_snap_moves_centerline_toward_bright_track(self):
        snapped = snap_centerline_to_centroid(
            self.centerline, self.log, self.r2i,
            snap_radius_mm=2.0, step_mm=0.5,
            log_threshold=500.0, n_radii=2, n_angles=8, smooth_window=1,
        )
        # In RAS = (x, y, z): bright tube at y=4, centerline at y=3.
        # After snap, the y-coords should shift toward y=4 (positive direction)
        # at the interior arcs.
        n = snapped.shape[0]
        interior = snapped[n // 4:3 * n // 4, 1]
        self.assertGreater(float(interior.mean()), 3.0)

    def test_snap_no_metal_returns_centerline_unchanged(self):
        # All-zero LoG → no voxels above threshold → snap returns centerline.
        empty = np.zeros(self.shape, dtype=np.float32)
        snapped = snap_centerline_to_centroid(
            self.centerline, empty, self.r2i,
            snap_radius_mm=2.0, step_mm=1.0,
            log_threshold=500.0, n_radii=2, n_angles=8, smooth_window=1,
        )
        # All snapped points stay at y=3 (the input centerline's y-coord).
        np.testing.assert_allclose(snapped[:, 1], 3.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
