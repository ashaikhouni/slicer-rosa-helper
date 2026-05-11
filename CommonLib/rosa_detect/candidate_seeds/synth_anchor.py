"""Axis-to-skull synth anchor.

Fallback for stage-1 lines whose bolt CC couldn't be found: walk
outward from the shallowest line endpoint along the axis until the
hull surface is crossed, then synthesize a skull_entry / bolt_tip
pair so downstream code (extension + emission) can still anchor on
the line.
"""
from __future__ import annotations

import numpy as np

from ..primitives.bolt_anchor import BOLT_BASE_MAX_DIST_MM
from .constants import (
    AXIS_SKULL_SYNTH_BOLT_PROTRUDE_MM,
    AXIS_SKULL_SYNTH_MAX_OUTWARD_MM,
    AXIS_SKULL_SYNTH_STEP_MM,
)


def axis_to_skull_synth(shallow_ras, deep_ras, dist_arr, ras_to_ijk_mat,
                        step_mm=AXIS_SKULL_SYNTH_STEP_MM,
                        max_outward_mm=AXIS_SKULL_SYNTH_MAX_OUTWARD_MM,
                        bolt_protrude_mm=AXIS_SKULL_SYNTH_BOLT_PROTRUDE_MM,
                        skull_band_mm=BOLT_BASE_MAX_DIST_MM):
    """Synthesize a skull_entry_ras + bolt_tip_ras for a strong stage-1
    line whose bolt CC couldn't be found. Walk outward from
    ``shallow_ras`` along the axis (shallow → outside) until the hull
    surface is crossed; return the skull-band position as
    skull_entry, and a position ``bolt_protrude_mm`` further out as a
    synthetic bolt_tip. Returns (None, None) when the axis doesn't
    cross the hull within ``max_outward_mm`` (e.g., bolt outside the
    CT acquisition window but axis still misses the skull — CT is
    windowed out in that direction).
    """
    s = np.asarray(shallow_ras, dtype=float)
    e = np.asarray(deep_ras, dtype=float)
    d = s - e
    L = float(np.linalg.norm(d))
    if L < 1e-3:
        return None, None
    axis_out = d / L
    K, J, I = dist_arr.shape

    def _sample(p):
        h = np.array([float(p[0]), float(p[1]), float(p[2]), 1.0])
        ijk = (ras_to_ijk_mat @ h)[:3]
        i = int(np.clip(round(ijk[0]), 0, I - 1))
        j = int(np.clip(round(ijk[1]), 0, J - 1))
        k = int(np.clip(round(ijk[2]), 0, K - 1))
        return float(dist_arr[k, j, i])

    n_steps = int(max_outward_mm / step_mm)
    skull_entry = None
    for idx in range(0, n_steps + 1):
        p = s + idx * step_mm * axis_out
        d_at = _sample(p)
        # skull_entry = outermost position still inside the skull/dura
        # band (0 < dist ≤ skull_band_mm). Stop when we cross outside
        # the hull (dist < 0).
        if 0.0 <= d_at <= skull_band_mm:
            skull_entry = p
        elif d_at < 0.0:
            break
    if skull_entry is None:
        return None, None
    bolt_tip = skull_entry + float(bolt_protrude_mm) * axis_out
    return skull_entry, bolt_tip


__all__ = ["axis_to_skull_synth"]
