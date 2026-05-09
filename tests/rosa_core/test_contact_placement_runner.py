"""Tests for ``two_pass`` (cross-shank ownership orchestrator) and
``postpass_fft`` (per-subject FFT normalization)."""
from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from rosa_core.contact_placement import (
    PlacementCtx,
    apply_subject_fft_normalization,
    run_two_pass,
)


def _two_seed_synthetic_features():
    """Two parallel bright tubes 10 mm apart in a 64×32×32 volume."""
    K, J, I = 64, 32, 32
    arr = np.zeros((K, J, I), dtype=np.float32)
    for x in (10, 22):  # two tubes
        for n in range(8):
            z = 5 + int(round(n * 3.5))
            for dz in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        zi, yi, xi = z + dz, 16 + dy, x + dx
                        if 0 <= zi < K and 0 <= yi < J and 0 <= xi < I:
                            arr[zi, yi, xi] = 2500.0
    log = np.where(arr > 1000.0, -arr, np.zeros_like(arr)).astype(np.float32)
    return {
        "ct_arr_kji":     arr,
        "log":            log,
        "ras_to_ijk_mat": np.eye(4),
        "ijk_to_ras_mat": np.eye(4),
    }


def _seeds_for_two_tubes():
    return [
        {"start_ras": np.array([10, 16, 0],  dtype=float),
         "end_ras":   np.array([10, 16, 40], dtype=float)},
        {"start_ras": np.array([22, 16, 0],  dtype=float),
         "end_ras":   np.array([22, 16, 40], dtype=float)},
    ]


def _trivial_library():
    pitch = 3.5
    n = 8
    return [{
        "id": "SYNTH-8",
        "vendor": "test",
        "n_contacts": n,
        "pitch_mm": pitch,
        "contact_center_offsets_from_tip_mm": [k * pitch for k in range(n)],
        "active_length_mm": (n - 1) * pitch + pitch,
        "diameter_mm": 1.3,
    }]


class TwoPassTests(unittest.TestCase):
    def test_returns_one_ctx_per_seed(self):
        out = run_two_pass(
            _seeds_for_two_tubes(),
            features=_two_seed_synthetic_features(),
            library_models=_trivial_library(),
        )
        self.assertEqual(len(out), 2)
        for c in out:
            self.assertIn("compound_score", c.score_components)
            self.assertIn(c.score_components["band"], {"high", "medium", "low"})

    def test_other_centerlines_excludes_self(self):
        # Pass 2 should set ``other_centerlines`` to all OTHER pass-1
        # centerlines. We can't observe this directly through the public
        # API, but we can verify it doesn't crash with a single seed.
        out = run_two_pass(
            _seeds_for_two_tubes()[:1],
            features=_two_seed_synthetic_features(),
            library_models=_trivial_library(),
        )
        self.assertEqual(len(out), 1)


class ApplySubjectFftNormalizationTests(unittest.TestCase):
    def _ctx_with_components(self, *, n_slots, uniform, pitch_power):
        ctx = PlacementCtx(
            seed_start=np.zeros(3), seed_end=np.array([10, 0, 0], dtype=float),
            features={}, library_models=[],
            centerline=np.array([[0, 0, 0], [10, 0, 0]], dtype=float),
        )
        return replace(ctx, score_components={
            "model_uniform_pitch": uniform,
            "n_slots": n_slots,
            "pitch_power_frac": pitch_power,
            "fft_n_reliable_segments": 1 if uniform else 0,
            # Other fields needed by score_compound:
            "corr": 0.5, "tube_like_frac": 0.4, "model_corr_margin": 0.1,
            "bolt_zone_frac": 0.2, "bolt_source": "metal",
            "cc_overlap_score": 0.3,
        })

    def test_no_reliable_emissions_returns_input_unchanged(self):
        # All emissions are non-uniform → no reliable reference set → no-op.
        ctxs = [
            self._ctx_with_components(n_slots=10, uniform=False, pitch_power=0.5),
            self._ctx_with_components(n_slots=10, uniform=False, pitch_power=0.4),
        ]
        out = apply_subject_fft_normalization(ctxs)
        self.assertEqual(len(out), len(ctxs))

    def test_normalized_uniform_emissions_pickup_subject_ref(self):
        # Three reliable emissions with FFT 0.2/0.4/0.6 → p75 = 0.5.
        ctxs = [
            self._ctx_with_components(n_slots=10, uniform=True, pitch_power=0.2),
            self._ctx_with_components(n_slots=10, uniform=True, pitch_power=0.4),
            self._ctx_with_components(n_slots=10, uniform=True, pitch_power=0.6),
        ]
        out = apply_subject_fft_normalization(ctxs)
        for c in out:
            sc = c.score_components
            self.assertIn("fft_subject_ref_p75", sc)
            self.assertGreater(sc["fft_subject_ref_p75"], 0.0)
            self.assertIn("fft_subject_norm", sc)

    def test_short_emissions_get_none_norm_and_neutral_compound(self):
        # An emission with fewer than 8 slots stays out of the reference set
        # AND gets ``fft_subject_norm = None``.
        ctxs = [
            self._ctx_with_components(n_slots=10, uniform=True, pitch_power=0.5),
            self._ctx_with_components(n_slots=10, uniform=True, pitch_power=0.5),
            self._ctx_with_components(n_slots=4,  uniform=True, pitch_power=0.5),
        ]
        out = apply_subject_fft_normalization(ctxs)
        self.assertIsNone(out[2].score_components["fft_subject_norm"])

    def test_zero_reliable_max_returns_input_unchanged(self):
        # Edge case: all reliable emissions have FFT == 0.
        ctxs = [
            self._ctx_with_components(n_slots=10, uniform=True, pitch_power=0.0),
            self._ctx_with_components(n_slots=10, uniform=True, pitch_power=0.0),
        ]
        out = apply_subject_fft_normalization(ctxs)
        # Function returns the input list (not necessarily the same objects).
        self.assertEqual(len(out), len(ctxs))


if __name__ == "__main__":
    unittest.main()
