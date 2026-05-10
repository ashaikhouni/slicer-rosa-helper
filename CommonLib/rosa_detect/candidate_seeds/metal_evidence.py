"""Per-voxel metal-evidence volume + along-axis fraction-strong sampler.

Used by v1's bolt-anchoring stage to find bolt CCs (as a volume that
combines |LoG| and HU evidence) and by the post-anchor scoring to
measure metal continuity along the trajectory axis.
"""
from __future__ import annotations

import numpy as np

from rosa_core.volume_sampling import clip_to_voxel, iter_axis_points, ras_to_ijk_pt

from ..primitives.bolt_anchor import (
    HU_BOLT_NORMALIZER,
    LOG_BOLT_NORMALIZER,
    METAL_BOLT_THRESHOLD,
)


def compute_metal_evidence_volume(log_arr, ct_arr):
    """Per-voxel metal-evidence volume:

        evidence(v) = max(|LoG(v)|/LOG_BOLT_NORMALIZER,
                          max(0, HU(v))/HU_BOLT_NORMALIZER)

    Returned as float32. ``ct_arr`` may be ``None`` (LoG-only fallback).
    """
    log_norm = np.abs(log_arr) / float(LOG_BOLT_NORMALIZER)
    if ct_arr is None:
        return log_norm.astype(np.float32, copy=False)
    hu_norm = np.maximum(0.0, ct_arr) / float(HU_BOLT_NORMALIZER)
    return np.maximum(log_norm, hu_norm).astype(np.float32, copy=False)


def frac_strong_metal_along_line(start_ras, end_ras, log_arr, ct_arr,
                                  ras_to_ijk_mat, step_mm: float = 0.5) -> float:
    """Fraction of axis samples whose per-voxel metal evidence saturates
    (>= ``METAL_BOLT_THRESHOLD``).

    Real SEEG shanks have many contact-saturating voxels along their axis
    (matched p10 ≈ 0.27, p50 ≈ 0.65). Hull-skimming bone-assembled chains
    and synth-extended FPs have near-zero saturation (orphan p50 ≈ 0.01).
    """
    L = float(np.linalg.norm(np.asarray(end_ras) - np.asarray(start_ras)))
    if L < step_mm:
        return 0.0
    n_strong = 0
    n = 0
    for _t, p in iter_axis_points(start_ras, end_ras, step_mm):
        i, j, k = ras_to_ijk_pt(ras_to_ijk_mat, p)
        kc, jc, ic = clip_to_voxel(log_arr.shape, i, j, k)
        log_norm = abs(float(log_arr[kc, jc, ic])) / LOG_BOLT_NORMALIZER
        hu_norm = max(0.0, float(ct_arr[kc, jc, ic])) / HU_BOLT_NORMALIZER
        if max(log_norm, hu_norm) >= METAL_BOLT_THRESHOLD:
            n_strong += 1
        n += 1
    return float(n_strong) / float(n)


__all__ = ["compute_metal_evidence_volume", "frac_strong_metal_along_line"]
