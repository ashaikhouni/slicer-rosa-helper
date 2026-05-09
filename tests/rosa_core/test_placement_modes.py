"""Unit tests for ``rosa_core.placement_modes`` — the public 5-mode dispatcher.

Session 1 implements only mode 4 (placement-only with user-vouched seeds +
model_id). Modes 1, 2, 3, 5 raise ``NotImplementedError`` until Session 2.
"""
from __future__ import annotations

import unittest

import numpy as np

from rosa_core.placement_modes import (
    PlacedTrajectory,
    PlacementBatch,
    Seed,
    place_seeg,
)


class SeedDataclassTests(unittest.TestCase):
    def test_endpoints_coerced_to_float_arrays(self):
        s = Seed(name="L1", start_ras=[0, 1, 2], end_ras=(10, 11, 12))
        self.assertIsInstance(s.start_ras, np.ndarray)
        self.assertIsInstance(s.end_ras, np.ndarray)
        self.assertEqual(s.start_ras.dtype, np.float64)
        np.testing.assert_allclose(s.start_ras, [0, 1, 2])
        np.testing.assert_allclose(s.end_ras, [10, 11, 12])

    def test_model_id_default_none(self):
        s = Seed(name="L1", start_ras=[0, 0, 0], end_ras=[1, 0, 0])
        self.assertIsNone(s.model_id)


class ModeDispatchTests(unittest.TestCase):
    """Routes-only — verifies the dispatcher picks the right mode and rejects
    incompatible argument combos. No actual placement runs (CT loaders + the
    candidate-seed generator are Session 2)."""

    def setUp(self):
        # Synthetic minimal features dict — the dispatcher needs something
        # truthy to pass through, but mode != 4 raises before features is read.
        self.features = {
            "ct_arr_kji": np.zeros((4, 4, 4), dtype=np.float32),
            "ras_to_ijk_mat": np.eye(4),
            "ijk_to_ras_mat": np.eye(4),
        }

    def test_mode_1_not_implemented(self):
        with self.assertRaises(NotImplementedError) as cm:
            place_seeg(None, features=self.features)
        self.assertIn("mode 1", str(cm.exception))

    def test_mode_2_not_implemented(self):
        with self.assertRaises(NotImplementedError) as cm:
            place_seeg(None, n_expected=8, features=self.features)
        self.assertIn("mode 2", str(cm.exception))

    def test_mode_3_not_implemented(self):
        with self.assertRaises(NotImplementedError) as cm:
            place_seeg(None, expected=[("L1", "DIXI-15")], features=self.features)
        self.assertIn("mode 3", str(cm.exception))

    def test_mode_5_not_implemented(self):
        # Seeds without model_id → mode 5.
        seeds = [Seed(name="L1", start_ras=[0, 0, 0], end_ras=[1, 0, 0])]
        with self.assertRaises(NotImplementedError) as cm:
            place_seeg(None, seeds=seeds, features=self.features)
        self.assertIn("mode 5", str(cm.exception))

    def test_mode_4_requires_features(self):
        seeds = [Seed(name="L1", start_ras=[0, 0, 0], end_ras=[1, 0, 0],
                      model_id="DIXI-15CM")]
        with self.assertRaises(NotImplementedError) as cm:
            place_seeg(None, seeds=seeds)  # no features
        self.assertIn("features", str(cm.exception).lower())

    def test_seeds_plus_expected_rejected(self):
        seeds = [Seed(name="L1", start_ras=[0, 0, 0], end_ras=[1, 0, 0])]
        with self.assertRaises(ValueError):
            place_seeg(None, seeds=seeds, expected=[("L1", "DIXI-15")],
                       features=self.features)

    def test_seeds_plus_n_expected_rejected(self):
        seeds = [Seed(name="L1", start_ras=[0, 0, 0], end_ras=[1, 0, 0])]
        with self.assertRaises(ValueError):
            place_seeg(None, seeds=seeds, n_expected=4, features=self.features)


class Mode4SyntheticTests(unittest.TestCase):
    """End-to-end mode-4 run on the synthetic CT fixture from
    ``test_contact_placement_stages``. Verifies the dispatcher returns a
    well-formed ``PlacementBatch`` with one ``PlacedTrajectory`` per seed."""

    def setUp(self):
        # Inline the synthetic fixture (avoid cross-test-module imports).
        K, J, I = 64, 32, 32
        arr = np.zeros((K, J, I), dtype=np.float32)
        z0, y0, x0 = 5, 16, 16
        n_contacts, pitch_mm = 10, 3.5
        for n in range(n_contacts):
            z = int(round(z0 + n * pitch_mm))
            for dz in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        zi, yi, xi = z + dz, y0 + dy, x0 + dx
                        if 0 <= zi < K and 0 <= yi < J and 0 <= xi < I:
                            arr[zi, yi, xi] = 2500.0
        log = np.where(arr > 1000.0, -arr, np.zeros_like(arr)).astype(np.float32)
        self.features = {
            "ct_arr_kji":     arr,
            "log":            log,
            "ras_to_ijk_mat": np.eye(4),
            "ijk_to_ras_mat": np.eye(4),
        }
        self.library = [{
            "id": "SYNTH-10",
            "vendor": "test",
            "n_contacts": n_contacts,
            "pitch_mm": pitch_mm,
            "contact_center_offsets_from_tip_mm": [n * pitch_mm for n in range(n_contacts)],
            "active_length_mm": (n_contacts - 1) * pitch_mm + pitch_mm,
            "diameter_mm": 1.3,
        }]
        self.seeds = [Seed(
            name="SYNTH-L1",
            start_ras=np.array([x0, y0, z0 - 5.0], dtype=float),
            end_ras=np.array([x0, y0, z0 + n_contacts * pitch_mm + 5.0], dtype=float),
            model_id="SYNTH-10",
        )]

    def test_mode_4_returns_placement_batch(self):
        # End-to-end shape check on the synthetic fixture. Doesn't assert
        # ``model_id == "SYNTH-10"`` — synthetic CTs don't always cross the
        # matched-filter corr floor; the AMC88/T18 integration tests pin the
        # real numbers.
        batch = place_seeg(None, seeds=self.seeds,
                           library=self.library, features=self.features)
        self.assertIsInstance(batch, PlacementBatch)
        self.assertEqual(len(batch.trajectories), 1)
        traj = batch.trajectories[0]
        self.assertIsInstance(traj, PlacedTrajectory)
        self.assertEqual(traj.name, "SYNTH-L1")
        self.assertIn(traj.band, {"high", "medium", "low"})
        self.assertIsInstance(traj.score_components, dict)

    def test_mode_4_diagnostics_record_mode(self):
        batch = place_seeg(None, seeds=self.seeds,
                           library=self.library, features=self.features)
        self.assertEqual(batch.diagnostics["mode"], 4)
        self.assertEqual(batch.diagnostics["n_seeds"], 1)

    def test_output_dir_carried_through(self):
        # output_dir is stored on the result without writing files yet
        # (Session 3 wires the writer).
        from pathlib import Path
        batch = place_seeg(None, seeds=self.seeds,
                           library=self.library, features=self.features,
                           output_dir="/tmp/test_qc_dir")
        self.assertEqual(batch.qc_dir, Path("/tmp/test_qc_dir"))


if __name__ == "__main__":
    unittest.main()
