"""Per-subject FFT normalization post-pass.

Cross-subject AMC88 vs AMC91 vs T-series have very different absolute FFT
power levels (AMC88 matched p75 ≈ 0.5; AMC91 matched p75 ≈ 0.9). A single
global threshold for ``s_fft`` over-rewards AMC91 and under-rewards AMC88.

Post-pass: take the subject's p75 of FFT scores **across reliable emissions**
(uniform-pitch, n_contacts ≥ 8) and divide each emission's FFT by it. After
this, every subject's matched cluster lands near ``s_fft ≈ 1.0`` and orphans
below.

Reliability gate keeps CM/BM and short-electrode emissions out of the
reference set — their FFTs are structurally weak and would drag the p75 down.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .context import PlacementCtx
from .stage_f_score import score_compound


def apply_subject_fft_normalization(ctx_list: list[PlacementCtx]) -> list[PlacementCtx]:
    """Recompute compound score per emission with FFT normalized to subject p75.

    Returns a new list — does not mutate the input contexts.

    No-op when no reliable emissions exist or all reference FFTs are below
    1e-6 (returns the input list as-is). The non-reliable emissions get
    ``fft_norm=None`` which routes ``score_compound`` to the neutral 0.5
    fallback (preserves the per-emission reliability gate semantics).
    """
    reliable = [
        c.score_components.get("pitch_power_frac", 0.0) for c in ctx_list
        if c.score_components.get("model_uniform_pitch")
        and int(c.score_components.get("n_slots", 0)) >= 8
    ]
    if not reliable or max(reliable) <= 1e-6:
        return list(ctx_list)
    fft_ref = float(np.percentile(reliable, 75))
    out: list[PlacementCtx] = []
    for c in ctx_list:
        sc = c.score_components
        if sc.get("model_uniform_pitch") and int(sc.get("n_slots", 0)) >= 8:
            normed = float(sc.get("pitch_power_frac", 0.0)) / max(fft_ref, 1e-6)
        else:
            normed = None  # → score_compound uses neutral 0.5 (fft_reliable=False)
        sc_aug = dict(sc)
        sc_aug["fft_subject_ref_p75"] = fft_ref
        sc_aug["fft_subject_norm"] = normed
        c2 = replace(c, score_components=sc_aug)
        out.append(score_compound(c2, fft_norm=normed))
    return out


__all__ = ["apply_subject_fft_normalization"]
