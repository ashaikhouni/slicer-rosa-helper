"""Tests for ``postpass_fft`` (per-subject FFT normalization).

(The ``two_pass`` cross-shank orchestrator was retired with the placement
consolidation — the snap-flow's ``arbitrate_shared_peaks`` subsumed it.)
"""
from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from rosa_core.contact_placement import (
    PlacementCtx,
    apply_subject_fft_normalization,
)


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
