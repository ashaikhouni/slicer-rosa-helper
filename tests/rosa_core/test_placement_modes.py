"""Unit tests for ``rosa_core.placement_modes`` — the public 5-mode dispatcher.

Session 2 wires all 5 modes. Mode-level integration tests (real datasets)
live in ``test_contact_placement_integration.py``; this file exercises the
dispatcher's mode routing + arg-validation + the synthetic CT smoke path.
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


# ---------------------------------------------------------------------
# Synthetic-CT fixture (parallel-tube; mirrors test_contact_placement_runner)
# ---------------------------------------------------------------------


def _synthetic_features():
    """A bright periodic tube along axis 0; identity 1mm IJK→RAS.

    The features dict matches what
    ``rosa_detect.guided_fit_engine.compute_features`` would produce — a
    SimpleITK image included so candidate-seed generation (which calls
    ``run_two_stage_detection``) doesn't crash on missing ``img``.
    """
    import SimpleITK as sitk
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
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    head_distance = np.full(arr.shape, 50.0, dtype=np.float32)
    return {
        "img":            img,
        "ct_arr_kji":     arr,
        "log":            log,
        "frangi":         np.zeros_like(arr),
        "head_distance":  head_distance,
        "ras_to_ijk_mat": np.eye(4),
        "ijk_to_ras_mat": np.eye(4),
    }


def _synthetic_library():
    pitch_mm = 3.5
    n = 10
    return [{
        "id": "SYNTH-10",
        "vendor": "test",
        "n_contacts": n,
        "pitch_mm": pitch_mm,
        "contact_center_offsets_from_tip_mm": [k * pitch_mm for k in range(n)],
        "active_length_mm": (n - 1) * pitch_mm + pitch_mm,
        "diameter_mm": 1.3,
    }]


def _synthetic_seed(model_id="SYNTH-10"):
    z0, y0, x0 = 5, 16, 16
    return Seed(
        name="SYNTH-L1",
        start_ras=np.array([x0, y0, z0 - 5.0], dtype=float),
        end_ras=np.array([x0, y0, z0 + 10 * 3.5 + 5.0], dtype=float),
        model_id=model_id,
    )


# ---------------------------------------------------------------------
# Seed dataclass — extended in Session 2 with seeder_* fields
# ---------------------------------------------------------------------


class SeedDataclassTests(unittest.TestCase):
    def test_endpoints_coerced_to_float_arrays(self):
        s = Seed(name="L1", start_ras=[0, 1, 2], end_ras=(10, 11, 12))
        self.assertIsInstance(s.start_ras, np.ndarray)
        self.assertIsInstance(s.end_ras, np.ndarray)
        self.assertEqual(s.start_ras.dtype, np.float64)
        np.testing.assert_allclose(s.start_ras, [0, 1, 2])
        np.testing.assert_allclose(s.end_ras, [10, 11, 12])

    def test_default_optional_fields(self):
        s = Seed(name="L1", start_ras=[0, 0, 0], end_ras=[1, 0, 0])
        self.assertIsNone(s.model_id)
        self.assertEqual(s.seeder_label, "")
        self.assertEqual(s.seeder_confidence, 0.0)
        self.assertIsNone(s.seeder_model)

    def test_custom_seeder_fields(self):
        s = Seed(name="L1", start_ras=[0, 0, 0], end_ras=[1, 0, 0],
                 seeder_label="high", seeder_confidence=0.8,
                 seeder_model="DIXI-15CM")
        self.assertEqual(s.seeder_label, "high")
        self.assertEqual(s.seeder_confidence, 0.8)
        self.assertEqual(s.seeder_model, "DIXI-15CM")


# ---------------------------------------------------------------------
# Mode dispatch — argument validation
# ---------------------------------------------------------------------


class ModeDispatchValidationTests(unittest.TestCase):
    def setUp(self):
        self.features = _synthetic_features()
        self.bolts = []
        self.library = _synthetic_library()

    def test_seeds_plus_expected_rejected(self):
        seeds = [Seed(name="L1", start_ras=[0, 0, 0], end_ras=[1, 0, 0])]
        with self.assertRaises(ValueError):
            place_seeg(None, seeds=seeds, expected=[("L1", "DIXI-15")],
                       features=self.features, bolts=self.bolts)

    def test_seeds_plus_n_expected_rejected(self):
        seeds = [Seed(name="L1", start_ras=[0, 0, 0], end_ras=[1, 0, 0])]
        with self.assertRaises(ValueError):
            place_seeg(None, seeds=seeds, n_expected=4,
                       features=self.features, bolts=self.bolts)

    def test_no_ct_no_features_rejected(self):
        # Without features pre-supplied, place_seeg needs a CT to load.
        seeds = [_synthetic_seed()]
        with self.assertRaises(ValueError) as cm:
            place_seeg(None, seeds=seeds, library=self.library)
        self.assertIn("CT", str(cm.exception))


# ---------------------------------------------------------------------
# Mode-4 (placement-only with model_id) — synthetic CT smoke
# ---------------------------------------------------------------------


class Mode4SyntheticTests(unittest.TestCase):
    """Mode 4: caller supplies seeds with model_id; no library search."""

    def setUp(self):
        self.features = _synthetic_features()
        self.bolts = []
        self.library = _synthetic_library()
        self.seeds = [_synthetic_seed("SYNTH-10")]

    def test_mode_4_returns_placement_batch(self):
        batch = place_seeg(None, seeds=self.seeds,
                           library=self.library, features=self.features,
                           bolts=self.bolts)
        self.assertIsInstance(batch, PlacementBatch)
        self.assertEqual(len(batch.trajectories), 1)
        traj = batch.trajectories[0]
        self.assertIsInstance(traj, PlacedTrajectory)
        self.assertEqual(traj.name, "SYNTH-L1")
        self.assertIn(traj.band, {"high", "medium", "low"})
        self.assertIsInstance(traj.score_components, dict)

    def test_mode_4_diagnostics_record_mode(self):
        batch = place_seeg(None, seeds=self.seeds,
                           library=self.library, features=self.features,
                           bolts=self.bolts)
        self.assertEqual(batch.diagnostics["mode"], 4)
        self.assertEqual(batch.diagnostics["n_input_seeds"], 1)
        self.assertEqual(batch.diagnostics["n_emitted"], 1)

    def test_output_dir_carried_through(self):
        from pathlib import Path
        batch = place_seeg(None, seeds=self.seeds,
                           library=self.library, features=self.features,
                           bolts=self.bolts, output_dir="/tmp/test_qc_dir")
        self.assertEqual(batch.qc_dir, Path("/tmp/test_qc_dir"))


# ---------------------------------------------------------------------
# Mode-5 (seeded, no model_id) — synthetic CT smoke
# ---------------------------------------------------------------------


class Mode5SyntheticTests(unittest.TestCase):
    """Mode 5: seeds without model_id; snaps to closest mode-1 emission.

    Synthetic CT typically yields zero v1 candidates (too clean for stage1
    to pick anything up). Mode 5 then drops all seeds (no candidate to snap
    to). This is the documented behavior — caller sees ``n_seeds_unmatched``
    in diagnostics. Real-CT mode-5 behavior is exercised by the AMC88
    integration test.
    """

    def setUp(self):
        self.features = _synthetic_features()
        self.bolts = []
        self.library = _synthetic_library()
        self.seeds = [_synthetic_seed(model_id=None)]

    def test_mode_5_routes_correctly(self):
        batch = place_seeg(None, seeds=self.seeds,
                           library=self.library, features=self.features,
                           bolts=self.bolts)
        self.assertEqual(batch.diagnostics["mode"], 5)
        self.assertIn("mode5", batch.diagnostics)

    def test_mode_5_diag_reports_snap_outcome(self):
        batch = place_seeg(None, seeds=self.seeds,
                           library=self.library, features=self.features,
                           bolts=self.bolts)
        d = batch.diagnostics["mode5"]
        for k in ("n_input_seeds", "n_candidates_generated",
                  "n_seeds_matched", "n_seeds_unmatched",
                  "n_candidates_dropped",
                  "snap_angle_tol_deg", "snap_perp_tol_mm"):
            self.assertIn(k, d)
        self.assertEqual(d["n_input_seeds"], 1)
        # Unmatched + matched = input total.
        self.assertEqual(
            d["n_seeds_matched"] + d["n_seeds_unmatched"],
            d["n_input_seeds"],
        )

    def test_mode_5_snap_tolerances_recorded(self):
        # The configured tolerances surface in diagnostics so QC can see them.
        batch = place_seeg(None, seeds=self.seeds,
                           library=self.library, features=self.features,
                           bolts=self.bolts,
                           snap_angle_tol_deg=2.5, snap_perp_tol_mm=0.5)
        d = batch.diagnostics["mode5"]
        self.assertEqual(d["snap_angle_tol_deg"], 2.5)
        self.assertEqual(d["snap_perp_tol_mm"], 0.5)
        # Sanity: matched + unmatched = input total.
        self.assertEqual(
            d["n_seeds_matched"] + d["n_seeds_unmatched"], d["n_input_seeds"],
        )


# ---------------------------------------------------------------------
# Mode-1/2/3 dispatch shape — these need a real-ish CT for the
# candidate-seed generator (v1 stage1) to emit anything; the synthetic
# fixture is too clean. Smoke-check that the dispatcher routes correctly
# and returns a PlacementBatch even when no candidates are produced.
# ---------------------------------------------------------------------


class AutoModeDispatchTests(unittest.TestCase):
    """Modes 1/2/3 — confirm routing. Empty-result OK for the synthetic CT."""

    def setUp(self):
        self.features = _synthetic_features()
        self.bolts = []
        self.library = _synthetic_library()

    def test_mode_1_routes_to_auto(self):
        batch = place_seeg(None, library=self.library,
                           features=self.features, bolts=self.bolts)
        self.assertEqual(batch.diagnostics["mode"], 1)
        self.assertIsInstance(batch.trajectories, list)
        # band_floor default = "medium" for mode 1.
        self.assertEqual(batch.diagnostics["band_floor"], "medium")

    def test_mode_2_routes_with_n_expected(self):
        batch = place_seeg(None, n_expected=3, library=self.library,
                           features=self.features, bolts=self.bolts)
        self.assertEqual(batch.diagnostics["mode"], 2)
        # Mode 2 returns up to N (could be fewer if v1 emits less).
        self.assertLessEqual(len(batch.trajectories), 3)

    def test_mode_3_routes_with_expected(self):
        batch = place_seeg(None,
                           expected=[("L1", "SYNTH-10")],
                           library=self.library,
                           features=self.features, bolts=self.bolts)
        self.assertEqual(batch.diagnostics["mode"], 3)


# ---------------------------------------------------------------------
# Helper-level tests
# ---------------------------------------------------------------------


class FilterByBandTests(unittest.TestCase):
    def test_band_floor_filters_by_rank(self):
        from rosa_core.placement_modes import _filter_by_band
        from rosa_core.contact_placement import PlacementCtx
        from dataclasses import replace

        def _ctx_with_band(band):
            ctx = PlacementCtx(
                seed_start=np.zeros(3),
                seed_end=np.array([10, 0, 0], dtype=float),
                features={}, library_models=[],
            )
            return replace(ctx, score_components={"band": band})

        pairs = [
            ("h", _ctx_with_band("high")),
            ("m", _ctx_with_band("medium")),
            ("l", _ctx_with_band("low")),
        ]
        kept = _filter_by_band(pairs, "medium")
        self.assertEqual([n for n, _ in kept], ["h", "m"])

        kept_high = _filter_by_band(pairs, "high")
        self.assertEqual([n for n, _ in kept_high], ["h"])


class AssignByExpectedTests(unittest.TestCase):
    @staticmethod
    def _ctx(model, score):
        from rosa_core.contact_placement import PlacementCtx
        from dataclasses import replace
        ctx = PlacementCtx(
            seed_start=np.zeros(3),
            seed_end=np.array([10, 0, 0], dtype=float),
            features={}, library_models=[],
        )
        return replace(ctx, score_components={
            "model_id": model, "compound_score": score, "band": "high",
        })

    def test_assignment_picks_best_match_per_expected(self):
        from rosa_core.placement_modes import _assign_by_expected

        pairs = [
            ("CAND-001", self._ctx("DIXI-15", 0.8)),
            ("CAND-002", self._ctx("DIXI-12", 0.7)),
            ("CAND-003", self._ctx("DIXI-15", 0.6)),
        ]
        out, diag = _assign_by_expected(pairs, [
            ("L1", "DIXI-15"),
            ("L2", "DIXI-12"),
        ])
        self.assertEqual([n for n, _ in out], ["L1", "L2"])
        self.assertEqual(out[0][1].score_components["model_id"], "DIXI-15")
        self.assertEqual(out[1][1].score_components["model_id"], "DIXI-12")
        self.assertEqual(diag["duplicate_models"], {})

    def test_duplicate_model_ids_emit_warning_and_diag(self):
        """Per user 2026-05-09: duplicate model_ids in expected can't be
        disambiguated from CT alone. Greedy assignment among same-model
        candidates is arbitrary; the dispatcher warns the caller."""
        import warnings
        from rosa_core.placement_modes import _assign_by_expected

        pairs = [
            ("CAND-001", self._ctx("DIXI-15", 0.9)),
            ("CAND-002", self._ctx("DIXI-15", 0.7)),
        ]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out, diag = _assign_by_expected(pairs, [
                ("L1", "DIXI-15"),
                ("L2", "DIXI-15"),
            ])

        self.assertEqual(len(out), 2)
        self.assertEqual(diag["duplicate_models"], {"DIXI-15": 2})
        # Warning surfaced.
        self.assertTrue(
            any("duplicate model_ids" in str(w.message) for w in caught),
            f"expected duplicate-model warning; got {[str(w.message) for w in caught]}",
        )

    def test_no_match_falls_back_to_highest_score_unassigned(self):
        from rosa_core.placement_modes import _assign_by_expected

        pairs = [
            ("CAND-001", self._ctx("DIXI-12", 0.8)),
            ("CAND-002", self._ctx("DIXI-12", 0.7)),
        ]
        out, diag = _assign_by_expected(pairs, [
            ("L1", "DIXI-15"),  # no DIXI-15 candidate
        ])
        self.assertEqual(len(out), 1)
        # Fallback used.
        self.assertEqual(diag["per_name_outcome"][0]["outcome"], "fallback_no_model_match")


if __name__ == "__main__":
    unittest.main()
