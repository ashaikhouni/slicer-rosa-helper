"""Post-anchor crossing-tip retreat.

When a deep tip ends up inside another trajectory's contact-acceptance
tube, the refinement has walked past the real electrode end into the
neighbour's contacts. :func:`retreat_crossing_tips` walks each such
tip back along its own axis until clearance is restored AND the
retreated tip sits on a real contact peak.

The ``min_length_mm`` floor is the strategy-scoped
``min_post_anchor_len_mm`` from :class:`WalkerBounds`; the orchestrator
passes it explicitly. When omitted, falls back to
``DEFAULT_WALKER_BOUNDS.min_post_anchor_len_mm``.
"""
from __future__ import annotations

import numpy as np

from ..primitives.geometry import min_perp_to_other_segments
from .constants import (
    AXIS_REFINE_MIN_ABS,
    CROSSING_RETREAT_STEP_MM,
    CROSSING_TIP_CLEARANCE_MM,
)
from .pitch_library import DEFAULT_WALKER_BOUNDS


def retreat_crossing_tips(anchored,
                          log_arr=None,
                          ras_to_ijk_mat=None,
                          clearance_mm=CROSSING_TIP_CLEARANCE_MM,
                          step_mm=CROSSING_RETREAT_STEP_MM,
                          min_length_mm: float | None = None,
                          contact_abs_log=AXIS_REFINE_MIN_ABS,
                          logger=None):
    """For every trajectory whose deep tip sits inside another's
    contact-acceptance tube (perp < ``clearance_mm`` from another
    segment), walk the tip back along its own axis until

    1. clearance from every other segment is ≥ ``clearance_mm``; AND
    2. the retreated tip sits on a real contact peak
       (``|LoG| ≥ contact_abs_log`` at the on-axis sample) — so it
       snaps to the deep edge of the last detected contact rather than
       floating in the empty gap past it.

    Aborts on a given trajectory if retreat would shrink it below
    ``min_length_mm`` (defaults to
    ``DEFAULT_WALKER_BOUNDS.min_post_anchor_len_mm``).
    """
    if min_length_mm is None:
        min_length_mm = DEFAULT_WALKER_BOUNDS.min_post_anchor_len_mm

    segs = []
    for rec in anchored:
        s = np.asarray(rec.get("start_ras"), dtype=float)
        e = np.asarray(rec.get("end_ras"), dtype=float)
        d = e - s
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            segs.append(None)
            continue
        segs.append({"rec": rec, "s": s, "e": e, "a": d / L, "L": L})

    have_log = log_arr is not None and ras_to_ijk_mat is not None
    if have_log:
        K, J, I = log_arr.shape

        def _log_at(p):
            h = np.array([float(p[0]), float(p[1]), float(p[2]), 1.0])
            ijk = (ras_to_ijk_mat @ h)[:3]
            i = int(np.clip(round(ijk[0]), 0, I - 1))
            j = int(np.clip(round(ijk[1]), 0, J - 1))
            k = int(np.clip(round(ijk[2]), 0, K - 1))
            return float(log_arr[k, j, i])
    else:
        def _log_at(p):
            return -float("inf")  # disables the contact-snap rule

    for i, seg in enumerate(segs):
        if seg is None:
            continue
        rec = seg["rec"]
        e = seg["e"]; a = seg["a"]; s = seg["s"]; L = seg["L"]
        clearance = min_perp_to_other_segments(e, segs, i)
        if clearance >= clearance_mm:
            continue
        # Retreat along -a step-by-step until both clearance is
        # restored AND the retreated tip sits on a real contact peak.
        # Bound the retreat so the trajectory doesn't shrink below
        # MIN_POST_ANCHOR_LEN_MM, which would make the bolt anchor
        # inconsistent with the rest of the gating.
        max_retreat = max(0.0, L - min_length_mm)
        n_steps = int(max_retreat / step_mm)
        retreated_mm = 0.0
        new_end = e
        for step in range(1, n_steps + 1):
            dist = step * step_mm
            candidate = e - dist * a
            if min_perp_to_other_segments(candidate, segs, i) < clearance_mm:
                continue  # still inside another shank's tube
            if have_log and _log_at(candidate) > -contact_abs_log:
                continue  # off a contact — keep retreating to snap to blob edge
            new_end = candidate
            retreated_mm = dist
            break
        if retreated_mm > 0.0:
            rec["end_ras"] = new_end
            rec["length_mm"] = float(np.linalg.norm(new_end - s))
            d_new = new_end - s
            L_new = float(np.linalg.norm(d_new))
            if L_new > 1e-6:
                segs[i] = {"rec": rec, "s": s, "e": new_end,
                           "a": d_new / L_new, "L": L_new}
            if logger is not None:
                try:
                    logger(
                        f"  retreated tip {i}: {retreated_mm:.1f} mm "
                        f"(clearance {clearance:.2f} → "
                        f"{min_perp_to_other_segments(new_end, segs, i):.2f} mm, "
                        f"snapped to contact peak)"
                    )
                except Exception:
                    pass
        elif logger is not None:
            try:
                logger(
                    f"  tip {i} crosses another shank (clearance "
                    f"{clearance:.2f} mm) but cannot retreat without "
                    f"shrinking below {min_length_mm} mm; left alone"
                )
            except Exception:
                pass


__all__ = ["retreat_crossing_tips"]
