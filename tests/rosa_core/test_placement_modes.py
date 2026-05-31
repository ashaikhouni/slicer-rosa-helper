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

    def test_empty_seeds_rejected(self):
        # An empty seed list almost always means "the upstream filter
        # dropped everything" — falling through to mode 1 hides that.
        with self.assertRaises(ValueError) as cm:
            place_seeg(None, seeds=[], features=self.features, bolts=self.bolts)
        self.assertIn("seeds", str(cm.exception).lower())

    def test_empty_expected_rejected(self):
        with self.assertRaises(ValueError) as cm:
            place_seeg(None, expected=[], features=self.features, bolts=self.bolts)
        self.assertIn("expected", str(cm.exception).lower())

    def test_zero_n_expected_rejected(self):
        with self.assertRaises(ValueError):
            place_seeg(None, n_expected=0, features=self.features, bolts=self.bolts)

    def test_negative_n_expected_rejected(self):
        with self.assertRaises(ValueError):
            place_seeg(None, n_expected=-3, features=self.features, bolts=self.bolts)

    def test_mode4_unknown_model_id_rejected(self):
        # Vouched-model semantics: mode 4 must NOT silently fall back to
        # the full library when the user's model_id isn't there.
        seeds = [_synthetic_seed(model_id="DOES-NOT-EXIST")]
        with self.assertRaises(ValueError) as cm:
            place_seeg(None, seeds=seeds,
                       features=self.features, bolts=self.bolts,
                       library=self.library)
        msg = str(cm.exception)
        self.assertIn("DOES-NOT-EXIST", msg)
        self.assertIn("library", msg.lower())


# ---------------------------------------------------------------------
# Mode-4 (placement-only with model_id) — synthetic CT smoke
# ---------------------------------------------------------------------


class Mode4SyntheticTests(unittest.TestCase):
    """Mode 4: caller supplies seeds with model_id.

    Mode 4 (post-2026-05-09) snaps user seeds to v1 candidate emissions
    (inheriting bolt-anchored geometry), then forces the matched filter
    to the vouched model. Falls back to per-seed placement when the snap
    finds no candidate within tolerance. Synthetic CT typically yields
    zero v1 candidates → all seeds take the fallback path.
    """

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
        self.assertIn("mode4", batch.diagnostics)
        d = batch.diagnostics["mode4"]
        for k in ("n_input_seeds", "n_candidates_generated",
                  "n_snapped_passthrough", "n_snapped_re_placed",
                  "n_fallback_per_seed", "snap_angle_tol_deg",
                  "snap_perp_tol_mm"):
            self.assertIn(k, d)
        # Sum of paths = total seeds.
        self.assertEqual(
            d["n_snapped_passthrough"] + d["n_snapped_re_placed"]
            + d["n_fallback_per_seed"],
            d["n_input_seeds"],
        )

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
    """Mode 5: seeds without model_id; snaps to closest mode-1 emission,
    falls back to per-seed placement when the snap finds no candidate
    within tolerance.

    Synthetic CT typically yields zero v1 candidates (too clean for stage1
    to pick anything up). Mode 5 then takes the fallback path for every
    seed — caller sees ``n_seeds_fallback == n_input_seeds`` in
    diagnostics. Real-CT mode-5 behavior is exercised by the AMC88 / T18
    integration tests.
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
                  "n_seeds_snapped", "n_seeds_fallback",
                  "n_candidates_dropped",
                  "snap_angle_tol_deg", "snap_perp_tol_mm"):
            self.assertIn(k, d)
        self.assertEqual(d["n_input_seeds"], 1)
        # Snapped + fallback = input total (every seed accounted for).
        self.assertEqual(
            d["n_seeds_snapped"] + d["n_seeds_fallback"],
            d["n_input_seeds"],
        )

    def test_mode_5_snap_tolerances_recorded(self):
        batch = place_seeg(None, seeds=self.seeds,
                           library=self.library, features=self.features,
                           bolts=self.bolts,
                           snap_angle_tol_deg=2.5, snap_perp_tol_mm=0.5)
        d = batch.diagnostics["mode5"]
        self.assertEqual(d["snap_angle_tol_deg"], 2.5)
        self.assertEqual(d["snap_perp_tol_mm"], 0.5)
        # Snapped + fallback = input total.
        self.assertEqual(
            d["n_seeds_snapped"] + d["n_seeds_fallback"], d["n_input_seeds"],
        )

    def test_mode_5_returns_one_traj_per_input_seed(self):
        """Per user 2026-05-10: never silently drop a user-supplied seed.
        Fallback path runs per-seed placement when v1 misses."""
        batch = place_seeg(None, seeds=self.seeds,
                           library=self.library, features=self.features,
                           bolts=self.bolts)
        self.assertEqual(len(batch.trajectories), len(self.seeds))


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


class SnapFlowAutoEngineTests(unittest.TestCase):
    """The auto modes (1/2/3) run the fit-rosa snap-flow engine
    (snap_fit_to_ctxs) — the single canonical engine since the staged two_pass
    was retired. Synthetic smoke: the public contract holds on the snap-flow
    path; the dataset accuracy parity is the manual A-B
    (tools/_ab_placeseeg_auto.py)."""

    def setUp(self):
        self.features = _synthetic_features()
        self.bolts = []
        self.library = _synthetic_library()

    def test_mode1_snap_flow_runs_and_contract_holds(self):
        batch = place_seeg(None, features=self.features, bolts=self.bolts,
                           library=self.library, band_floor="low")
        self.assertIsInstance(batch, PlacementBatch)
        self.assertEqual(batch.diagnostics["mode"], 1)
        self.assertEqual(batch.diagnostics["auto_engine"], "snap_flow")
        # Whatever it emits must be well-formed PlacedTrajectory records.
        for t in batch.trajectories:
            self.assertIsInstance(t, PlacedTrajectory)
            self.assertIn(t.band, {"high", "medium", "low"})
            self.assertIsInstance(t.score_components, dict)
            self.assertIn(t.bolt_source, {"metal", "bolt_less"})


class ForceModelOnSnappedCtxTests(unittest.TestCase):
    """Mode-4's snap-flow forced re-place (_force_model_on_snapped_ctx) must
    INHERIT the matched emission's bolt_source, not hard-code 'metal'. A user
    seed can match an auto emission that is itself a bolt_less fallback (snap
    failed on the v1 candidate → raw-seed geometry); labeling that 'metal'
    would falsely set s_walker / the cropped-bolt s_cc_overlap and inflate the
    band, and report bolt_source='metal' for geometry with no metal anchor."""

    def setUp(self):
        self.features = _synthetic_features()
        self.library = _synthetic_library()
        self.seed = _synthetic_seed()

    def _cand(self, bolt_source):
        # _force_model_on_snapped_ctx only reads seed_start/seed_end/bolt_source.
        from types import SimpleNamespace
        return SimpleNamespace(
            seed_start=self.seed.start_ras,
            seed_end=self.seed.end_ras,
            bolt_source=bolt_source,
        )

    def test_metal_emission_stays_metal_and_places(self):
        from rosa_core.placement_modes import _force_model_on_snapped_ctx
        ctx = _force_model_on_snapped_ctx(
            self._cand("metal"), "SYNTH-10",
            features=self.features, library_models=self.library,
            bolts=[], seed=self.seed)
        self.assertEqual(ctx.bolt_source, "metal")
        self.assertGreater(len(ctx.placed_ras), 0)  # vouched model forced+placed

    def test_bolt_less_emission_stays_bolt_less(self):
        from rosa_core.placement_modes import _force_model_on_snapped_ctx
        ctx = _force_model_on_snapped_ctx(
            self._cand("bolt_less"), "SYNTH-10",
            features=self.features, library_models=self.library,
            bolts=[], seed=self.seed)
        # The fix: bolt_source is propagated, NOT forced to 'metal'.
        self.assertEqual(ctx.bolt_source, "bolt_less")
        # Still forces + places the vouched model (never drop the seed).
        self.assertGreater(len(ctx.placed_ras), 0)


if __name__ == "__main__":
    unittest.main()
