"""Cross-stage trajectory dedup.

Two rules, applied in order across the assembled trajectory list:

1. **Same bolt anchor + same physical line → same shank.** Different
   walker passes finding disjoint blob ranges along one electrode
   collapse to the longest survivor.
2. **Inlier-subset → same shank.** A candidate whose blob inliers are
   a subset of an already-kept trajectory's inliers is a duplicate.
   Disjoint inlier sets that anchored to *different* bolts (or to
   merged bolts at different lines) are physically distinct shanks
   and both survive.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .constants import (
    POST_ANCHOR_DEDUP_ANG_DEG,
    POST_ANCHOR_DEDUP_PERP_MM,
)


def dedup_trajectories(trajectories,
                       perp_mm=POST_ANCHOR_DEDUP_PERP_MM,
                       ang_deg=POST_ANCHOR_DEDUP_ANG_DEG):
    """Remove duplicate trajectories.

    Two rules, in order:

      1. **Same bolt anchor + same physical line → same shank.**
         Trajectories that anchored to the same bolt CC AND share
         the same physical axis (small inter-axis angle AND small
         midpoint-to-axis perpendicular) are different walker passes
         finding disjoint blob ranges along one electrode
         (T5 LCMN, T3 LAI classes). The line-to-line proximity check
         stays robust under walker-axis variation: a 1-3° tilt over
         a 50-80 mm reach moves the start/end points by 5-15 mm but
         leaves the midpoint perp within ~1 mm. Distinct shanks
         that happen to share a merged ``bolt_id`` (T1's mega-CC,
         T22's adjacent bolts) have either non-parallel axes or
         non-coincident midpoints, so they survive.

      2. **Inlier-subset → same shank.** The candidate's blob
         inliers are a subset of an already-kept trajectory's
         inliers — drop, the survivor still claims every blob.
         Disjoint inlier sets that anchored to *different* bolts
         (or to merged bolts at different lines) are physically
         distinct shanks (T9 RAC next to LAC, T23 CWB next to its
         sibling) and both survive.
    """
    kept: list[dict[str, Any]] = []
    kept_inliers: list[frozenset] = []
    kept_bolt_ids: list[int] = []
    kept_axes: list[np.ndarray] = []
    kept_mids: list[np.ndarray] = []
    cos_ang_tol = float(np.cos(np.radians(ang_deg)))
    for t in trajectories:
        s = np.asarray(t["start_ras"], dtype=float)
        e = np.asarray(t["end_ras"], dtype=float)
        d = e - s
        length = float(np.linalg.norm(d))
        if length < 1e-6:
            continue
        t_axis = d / length
        t_mid = 0.5 * (s + e)
        t_inliers = frozenset(int(b) for b in (t.get("inlier_idx") or []))
        t_bolt_id = int(t.get("bolt_id", -1))
        # Rule 1 — same bolt CC + same physical line.
        if t_bolt_id >= 0:
            is_dup = False
            for ki, k_bolt_id in enumerate(kept_bolt_ids):
                if k_bolt_id != t_bolt_id:
                    continue
                cos_ang = abs(float(t_axis @ kept_axes[ki]))
                if cos_ang < cos_ang_tol:
                    continue  # not parallel enough
                # Midpoint-to-other-axis perpendicular.
                delta = t_mid - kept_mids[ki]
                perp = float(np.linalg.norm(
                    delta - (delta @ kept_axes[ki]) * kept_axes[ki]
                ))
                if perp <= perp_mm:
                    is_dup = True
                    break
            if is_dup:
                continue
        # Rule 2 — does any kept k subsume t (t.inliers ⊆ k.inliers)?
        # If so, drop t outright. Disjoint inlier sets on different
        # bolts are physically distinct shanks; both survive.
        is_dup_subset = False
        if t_inliers:
            for ki, k_inliers in enumerate(kept_inliers):
                if k_inliers and t_inliers.issubset(k_inliers):
                    is_dup_subset = True
                    break
        if is_dup_subset:
            continue
        # Pass 2b — does t subsume any kept k? Drop those k (no
        # orphans: all their blobs are in t).
        if t_inliers:
            survivors_idx = [
                ki for ki, k_inliers in enumerate(kept_inliers)
                if not (k_inliers and k_inliers.issubset(t_inliers))
            ]
            if len(survivors_idx) != len(kept):
                kept = [kept[ki] for ki in survivors_idx]
                kept_inliers = [kept_inliers[ki] for ki in survivors_idx]
                kept_bolt_ids = [kept_bolt_ids[ki] for ki in survivors_idx]
                kept_axes = [kept_axes[ki] for ki in survivors_idx]
                kept_mids = [kept_mids[ki] for ki in survivors_idx]
        kept.append(t)
        kept_inliers.append(t_inliers)
        kept_bolt_ids.append(t_bolt_id)
        kept_axes.append(t_axis)
        kept_mids.append(t_mid)
    return kept


__all__ = ["dedup_trajectories"]
