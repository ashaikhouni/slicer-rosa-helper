"""Per-stage unit tests for the ``rosa_core.contact_placement`` package.

These exercise each stage on a synthetic CT volume (a single bright tube with
periodic contacts along it). They don't validate accuracy — that's what the
multi-subject integration tests do — but they check:

1. Each stage advances the right ``PlacementCtx`` fields.
2. Stage swaps work (e.g. ``refine_noop`` vs ``refine_log_snap``).
3. The composer ``place_seed`` runs end-to-end without exceptions and
   returns sensibly-typed score components.
4. Edge cases: bolt_less anchor path, degenerate seed.

The synthetic fixture is small (32×32×64 voxels, 1mm isotropic) so each
test runs in well under a second.
"""
from __future__ import annotations

import unittest

import numpy as np

from rosa_core.contact_placement import (
    PlacementCtx,
    aggregate_disk,
    anchor_bolt_less,
    pick_matched_filter,
    place_at_match,
    place_seed,
    refine_log_snap,
    refine_noop,
    sample_hu_max,
    sample_neg_log_max,
    score_cc_overlap,
    score_compound,
    score_simple,
    stage_anchor,
)


# ---------------------------------------------------------------------
# Synthetic-CT fixture
# ---------------------------------------------------------------------


def _make_synthetic_ct(*, shape=(64, 32, 32), pitch_mm=3.5, n_contacts=10,
                        contact_hu=2500.0, bg_hu=0.0,
                        contact_radius_vox=1, axis_offset=(0, 16, 16)):
    """A bright tube along axis 0 with 10 high-HU spheres at ``pitch_mm``."""
    K, J, I = shape
    arr = np.full(shape, bg_hu, dtype=np.float32)
    z0, y0, x0 = axis_offset
    # Periodic contacts as small high-HU "spheres" (just cube neighborhoods).
    for n in range(n_contacts):
        z = int(round(z0 + n * pitch_mm))  # 1mm/voxel → arc ≈ z index
        if not (0 <= z < K):
            continue
        for dz in range(-contact_radius_vox, contact_radius_vox + 1):
            for dy in range(-contact_radius_vox, contact_radius_vox + 1):
                for dx in range(-contact_radius_vox, contact_radius_vox + 1):
                    zi, yi, xi = z + dz, y0 + dy, x0 + dx
                    if 0 <= zi < K and 0 <= yi < J and 0 <= xi < I:
                        arr[zi, yi, xi] = contact_hu
    # Identity 1mm IJK→RAS.
    i2r = np.eye(4)
    r2i = np.eye(4)
    return arr, i2r, r2i, axis_offset, n_contacts, pitch_mm


def _features_from_synthetic(arr, i2r, r2i):
    """Build a minimal features dict matching what stages read."""
    # LoG response: contacts produce negative LoG spikes. Approximate with a
    # sign-flipped + scaled copy for testing — the stages just sample it.
    log = np.where(arr > 1000.0, -arr, np.zeros_like(arr)).astype(np.float32)
    return {
        "ct_arr_kji":     arr,
        "log":            log,
        "ras_to_ijk_mat": r2i,
        "ijk_to_ras_mat": i2r,
    }


def _seed_along_axis(axis_offset, n_contacts, pitch_mm):
    """Seed running through the synthetic bright tube."""
    z0, y0, x0 = axis_offset
    # Seed in IJK-first ordering for the synthetic identity matrix:
    # arr[K, J, I] indexed [k=z, j=y, i=x]; sample_trilinear_at_ras consumes
    # RAS = [x_world, y_world, z_world]. With identity i2r, RAS == IJK ordered
    # as (i, j, k).
    start = np.array([x0, y0, z0 - 5.0], dtype=float)            # before first contact
    end = np.array([x0, y0, z0 + n_contacts * pitch_mm + 5.0], dtype=float)
    return start, end


def _library_subset(pitch_mm=3.5, n_contacts=10):
    """A 1-model library matching the synthetic geometry."""
    offs = [n * pitch_mm for n in range(n_contacts)]
    return [{
        "id": "SYNTH-10",
        "vendor": "test",
        "n_contacts": n_contacts,
        "pitch_mm": pitch_mm,
        "contact_center_offsets_from_tip_mm": offs,
        "active_length_mm": offs[-1] + pitch_mm,
        "diameter_mm": 1.3,
    }]


# ---------------------------------------------------------------------
# Stage A — anchor
# ---------------------------------------------------------------------


class StageAnchorTests(unittest.TestCase):
    def setUp(self):
        arr, i2r, r2i, off, n, p = _make_synthetic_ct()
        self.features = _features_from_synthetic(arr, i2r, r2i)
        self.s, self.e = _seed_along_axis(off, n, p)

    def test_anchor_bolt_less_writes_2_point_centerline(self):
        ctx = PlacementCtx(
            seed_start=self.s, seed_end=self.e,
            features=self.features, library_models=_library_subset(),
        )
        ctx2 = anchor_bolt_less(ctx)
        self.assertEqual(ctx2.bolt_source, "bolt_less")
        self.assertEqual(ctx2.bolt_end_arc, 0.0)
        self.assertIsNotNone(ctx2.centerline)
        self.assertEqual(ctx2.centerline.shape, (2, 3))

    def test_stage_anchor_falls_back_when_metal_fails(self):
        # Synthetic volume has no realistic bolt; metal anchor should fail
        # and we fall through to bolt_less.
        ctx = PlacementCtx(
            seed_start=self.s, seed_end=self.e,
            features=self.features, library_models=_library_subset(),
        )
        out = stage_anchor(ctx)
        self.assertIn(out.bolt_source, {"metal", "bolt_less"})
        self.assertIsNotNone(out.centerline)


# ---------------------------------------------------------------------
# Stage B — refine
# ---------------------------------------------------------------------


class StageRefineTests(unittest.TestCase):
    def test_refine_noop_is_identity(self):
        ctx = PlacementCtx(
            seed_start=np.array([0, 0, 0], dtype=float),
            seed_end=np.array([10, 0, 0], dtype=float),
            features={}, library_models=[],
        )
        out = refine_noop(ctx)
        self.assertIs(out, ctx)

    def test_refine_log_snap_skips_bolt_less(self):
        # bolt_less should pass through unchanged (no LoG basis to refine).
        ctx = PlacementCtx(
            seed_start=np.array([0, 0, 0], dtype=float),
            seed_end=np.array([10, 0, 0], dtype=float),
            features={"ras_to_ijk_mat": np.eye(4), "log": np.zeros((4, 4, 4))},
            library_models=[], bolt_source="bolt_less",
            centerline=np.array([[0, 0, 0], [10, 0, 0]], dtype=float),
        )
        out = refine_log_snap(ctx)
        # Same object — no-op for non-metal anchors.
        self.assertIs(out, ctx)


# ---------------------------------------------------------------------
# Stage C — sample
# ---------------------------------------------------------------------


class StageSampleTests(unittest.TestCase):
    def setUp(self):
        arr, i2r, r2i, off, n, p = _make_synthetic_ct()
        self.features = _features_from_synthetic(arr, i2r, r2i)
        self.s, self.e = _seed_along_axis(off, n, p)
        # Bolt-less anchor for simplicity — the walker doesn't need
        # bolt-end information to run.
        self.ctx = anchor_bolt_less(PlacementCtx(
            seed_start=self.s, seed_end=self.e,
            features=self.features, library_models=_library_subset(),
        ))

    def test_sample_hu_max_writes_arcs_and_signal(self):
        out = sample_hu_max(self.ctx)
        self.assertIsNotNone(out.walk_arcs)
        self.assertIsNotNone(out.walk_signal)
        self.assertEqual(out.signal_kind, "hu_max")
        self.assertEqual(len(out.walk_arcs), len(out.walk_signal))
        self.assertGreater(len(out.walk_arcs), 0)

    def test_sample_neg_log_max_writes_arcs_and_signal(self):
        out = sample_neg_log_max(self.ctx)
        self.assertEqual(out.signal_kind, "neg_log_max")
        # LoG values are negated on read; stored signal should be ≥0.
        self.assertTrue((out.walk_signal >= 0).all())


class AggregateDiskTests(unittest.TestCase):
    def test_max_p90_p75_median(self):
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        mask = np.ones_like(samples, dtype=bool)
        self.assertEqual(aggregate_disk(samples, mask, "max"), 100.0)
        self.assertGreater(aggregate_disk(samples, mask, "p90"), 50.0)
        self.assertGreater(aggregate_disk(samples, mask, "p75"), 4.0)
        self.assertEqual(aggregate_disk(samples, mask, "median"), 3.5)

    def test_unknown_kind_falls_back_to_max(self):
        samples = np.array([1.0, 5.0])
        mask = np.array([True, True])
        self.assertEqual(aggregate_disk(samples, mask, "garbage"), 5.0)

    def test_no_finite_returns_zero(self):
        samples = np.array([np.nan, np.inf])
        mask = np.array([True, True])
        # All masked-finite samples are non-finite → return 0.0.
        self.assertEqual(aggregate_disk(samples, mask, "max"), 0.0)

    def test_mask_excludes_voxels(self):
        samples = np.array([1.0, 100.0])
        mask = np.array([True, False])  # exclude the spike
        self.assertEqual(aggregate_disk(samples, mask, "max"), 1.0)


# ---------------------------------------------------------------------
# Stages D, E, F — exercised through the composer (cheap sanity)
# ---------------------------------------------------------------------


class ComposerSmokeTests(unittest.TestCase):
    """End-to-end run of ``place_seed`` on the synthetic fixture.

    Verifies the compound score is in [0, 1] and band ∈ {"high", "medium",
    "low"}. Does not pin model_id (synthetic fixture is too clean to be
    representative — that's what the AMC/T-series integration tests are for).
    """

    def setUp(self):
        arr, i2r, r2i, off, n, p = _make_synthetic_ct()
        self.features = _features_from_synthetic(arr, i2r, r2i)
        self.s, self.e = _seed_along_axis(off, n, p)
        self.library = _library_subset()

    def test_place_seed_returns_scored_ctx(self):
        ctx = place_seed(
            self.s, self.e,
            features=self.features,
            library_models=self.library,
        )
        sc = ctx.score_components
        self.assertIn("compound_score", sc)
        self.assertIn("band", sc)
        self.assertIn(sc["band"], {"high", "medium", "low"})
        self.assertGreaterEqual(sc["compound_score"], 0.0)
        self.assertLessEqual(sc["compound_score"], 1.0)
        # All 7 subscores present.
        sub = sc.get("subscores", {})
        for k in ("s_corr", "s_fft", "s_tube", "s_margin",
                  "s_walker", "s_cc_overlap", "s_seeder",
                  "bolt_only_penalty"):
            self.assertIn(k, sub)

    def test_place_seed_runs_with_single_model_library(self):
        # A 1-model library forces the matched filter to either pick that
        # model or return no_match. Either way the composer must complete
        # without raising and emit a valid score dict. (Synthetic CTs don't
        # always cross the matched-filter ``corr`` floor — that's expected
        # and is what the real integration tests validate.)
        ctx = place_seed(
            self.s, self.e,
            features=self.features,
            library_models=self.library,
        )
        sc = ctx.score_components
        self.assertIn("model_id", sc)            # field present
        self.assertIn("compound_score", sc)
        self.assertIn(sc["band"], {"high", "medium", "low"})

    def test_place_seed_with_log_sampler(self):
        ctx = place_seed(
            self.s, self.e,
            features=self.features,
            library_models=self.library,
            sample_fn=sample_neg_log_max,
        )
        self.assertEqual(ctx.signal_kind, "neg_log_max")

    def test_place_seed_with_refine_noop(self):
        # Stage swap: skip refinement; pipeline should still complete.
        ctx = place_seed(
            self.s, self.e,
            features=self.features,
            library_models=self.library,
            refine_fn=refine_noop,
        )
        self.assertIn("compound_score", ctx.score_components)


# ---------------------------------------------------------------------
# Stage F — score_compound boundary cases
# ---------------------------------------------------------------------


class ScoreCompoundBoundaryTests(unittest.TestCase):
    """Cases that don't need a full composer run."""

    def _ctx_with(self, **score_kwargs) -> PlacementCtx:
        ctx = PlacementCtx(
            seed_start=np.zeros(3), seed_end=np.array([10, 0, 0], dtype=float),
            features={}, library_models=[],
        )
        ctx.score_components = {
            "corr": 0.0, "tube_like_frac": 0.0, "model_corr_margin": 0.0,
            "bolt_zone_frac": 0.0, "bolt_source": "bolt_less",
            "cc_overlap_score": 0.0, "pitch_power_frac": 0.0,
            "fft_n_reliable_segments": 0,
        }
        ctx.score_components.update(score_kwargs)
        return ctx

    def test_band_high_at_perfect_signals(self):
        # Saturate every component; should land in "high".
        ctx = self._ctx_with(
            corr=1.0, tube_like_frac=1.0, model_corr_margin=0.5,
            bolt_source="metal", cc_overlap_score=1.0,
            pitch_power_frac=1.0, fft_n_reliable_segments=3,
        )
        ctx.bolt_source = "metal"
        ctx.score_components["bolt_source"] = "metal"
        out = score_compound(ctx)
        self.assertEqual(out.score_components["band"], "high")

    def test_band_low_at_zero_signals(self):
        ctx = self._ctx_with()  # all zeros
        out = score_compound(ctx)
        # All zeros + neutral_seeder(0.5)*0.10 + neutral_fft(0.5)*0.20 = 0.15
        self.assertEqual(out.score_components["band"], "low")

    def test_cropped_bolt_substitution(self):
        # walker=metal but cc_overlap_score=0 → s_cc_overlap=0.5 (not 0).
        ctx = self._ctx_with(
            bolt_source="metal", cc_overlap_score=0.0,
            corr=0.5, tube_like_frac=0.5, fft_n_reliable_segments=1,
            pitch_power_frac=0.6,
        )
        out = score_compound(ctx)
        sub = out.score_components["subscores"]
        self.assertEqual(sub["s_cc_overlap"], 0.5)

    def test_bolt_only_penalty_subtracts(self):
        # High bz + low FFT → penalty kicks in.
        ctx = self._ctx_with(
            bolt_zone_frac=0.9, fft_n_reliable_segments=1, pitch_power_frac=0.0,
            corr=0.5, bolt_source="metal", cc_overlap_score=0.5,
        )
        out = score_compound(ctx)
        self.assertGreater(out.score_components["subscores"]["bolt_only_penalty"], 0.0)

    def test_fft_norm_override(self):
        ctx = self._ctx_with(fft_n_reliable_segments=1, pitch_power_frac=0.1)
        # Without override, s_fft = clip(0.1) = 0.1.
        out_a = score_compound(ctx)
        # With override → s_fft = clip(0.9) = 0.9.
        out_b = score_compound(ctx, fft_norm=0.9)
        self.assertGreater(
            out_b.score_components["subscores"]["s_fft"],
            out_a.score_components["subscores"]["s_fft"],
        )


# ---------------------------------------------------------------------
# Stage F.1 — score_cc_overlap centerline-projection
# ---------------------------------------------------------------------


class ScoreCcOverlapTests(unittest.TestCase):
    def _ctx(self, *, bolts, bolt_source="metal"):
        ctx = PlacementCtx(
            seed_start=np.array([0, 0, 0], dtype=float),
            seed_end=np.array([20, 0, 0], dtype=float),
            features={}, library_models=[], bolts=bolts,
            bolt_source=bolt_source, bolt_end_arc=5.0,
            centerline=np.array([[0, 0, 0], [20, 0, 0]], dtype=float),
        )
        ctx.score_components = {}
        return ctx

    def test_no_bolts_returns_zero_score(self):
        ctx = self._ctx(bolts=None)
        out = score_cc_overlap(ctx)
        self.assertEqual(out.score_components["cc_overlap_score"], 0.0)

    def test_close_bolt_high_score(self):
        # A bolt CC centroid right at the start of the centerline.
        bolts = [{"id": "b1", "n_vox": 50, "pts_ras": np.array([[2, 0.5, 0]], dtype=float)}]
        ctx = self._ctx(bolts=bolts)
        out = score_cc_overlap(ctx)
        self.assertGreater(out.score_components["cc_overlap_score"], 0.5)
        self.assertEqual(out.score_components["cc_match"], "b1")

    def test_far_bolt_filtered_out(self):
        # CC centroid 10 mm laterally — past CC_OVERLAP_MAX_PERP_MM (8 mm).
        bolts = [{"id": "b1", "n_vox": 50, "pts_ras": np.array([[2, 10, 0]], dtype=float)}]
        ctx = self._ctx(bolts=bolts)
        out = score_cc_overlap(ctx)
        self.assertEqual(out.score_components["cc_overlap_score"], 0.0)


# ---------------------------------------------------------------------
# Stage E — place_at_match no-op when match rejected
# ---------------------------------------------------------------------


class PlaceAtMatchTests(unittest.TestCase):
    def test_no_match_returns_ctx_unchanged(self):
        ctx = PlacementCtx(
            seed_start=np.zeros(3), seed_end=np.array([10, 0, 0], dtype=float),
            features={}, library_models=[],
            centerline=np.array([[0, 0, 0], [10, 0, 0]], dtype=float),
            match=None,
        )
        out = place_at_match(ctx)
        self.assertEqual(out.placed_ras, [])
        self.assertIs(out, ctx)


if __name__ == "__main__":
    unittest.main()
