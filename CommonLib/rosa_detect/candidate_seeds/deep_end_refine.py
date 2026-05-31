"""Deep-end refinement for stage-1 lines.

Two functions, both operate on a single trajectory record after the
walker + bolt anchor have produced a tentative ``end_ras``:

* :func:`refine_deep_end_via_axis_log` — disk-sample LoG along the
  axis past ``end_ras`` (forward extension) to capture deep contacts
  that merged into one bright shaft and got missed by the 3D blob
  extractor (T2 RAI).
* :func:`clip_deep_end_to_inliers` — cap ``end_ras`` so it sits no
  more than ``DEEP_END_MARGIN_PAST_LAST_CONTACT_MM`` past the deepest
  walker inlier (catches walker / extension over-reach).
"""
from __future__ import annotations

import numpy as np

from .constants import (
    AXIS_REFINE_MAX_MM,
    AXIS_REFINE_MIN_ABS,
    AXIS_REFINE_MISS_MM,
    AXIS_REFINE_STEP_MM,
    DEEP_END_MARGIN_PAST_LAST_CONTACT_MM,
)


def refine_deep_end_via_axis_log(rec, log_arr, ras_to_ijk_mat,
                                 *, others=None,
                                 step_mm=AXIS_REFINE_STEP_MM,
                                 max_extend_mm=AXIS_REFINE_MAX_MM,
                                 min_abs_log=AXIS_REFINE_MIN_ABS,
                                 miss_mm=AXIS_REFINE_MISS_MM):
    """Normalize ``end_ras`` so it sits on the last real contact peak
    along the trajectory axis.

    Two passes share the same stopping rule (``miss_mm`` consecutive
    mm of LoG above ``-min_abs_log``):

      * Forward: walk outward past the current end. Fixes cases where
        individual deep-contact LoG minima merged into one bright
        shaft and the 3-D blob extractor missed them (T2 RAI).
      * Backward (clip-back): when the current end sits past the last
        real contact (walker / extension / anchor over-reach, e.g.
        subject-137 L_3's thin-wire PMT), walk INWARD until the first
        strong-LoG position and clip end to it.

    When ``others`` is provided (a list of other trajectory records),
    the disk sampler skips voxels that sit closer to any neighbour's
    axis than to ours (a local closest-axis ownership test). This
    prevents the disk-sampling at radii 1.5/2.5 mm from leaking into a
    kissing-partner's contacts and driving spurious extension into a
    gap (T1/6,3 + T1/7,4).

    Returns a new ``end_ras`` point or ``None`` when neither direction
    finds a contact peak (end is far from any contact signal).
    """
    start = np.asarray(rec.get("start_ras"), dtype=float)
    end = np.asarray(rec.get("end_ras"), dtype=float)
    d = end - start
    L = float(np.linalg.norm(d))
    if L < 1e-3:
        return None
    axis = d / L
    K, J, I = log_arr.shape

    # Build an orthonormal frame perpendicular to axis for disk
    # sampling. The walker's fitted axis can be 1-2° off from the
    # true shank axis; without a disk search, deep contacts at the
    # end of the chain that sit 1-3 mm off the walker-axis get missed.
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = helper - np.dot(helper, axis) * axis
    u_n = float(np.linalg.norm(u))
    if u_n < 1e-9:
        helper = np.array([0.0, 1.0, 0.0])
        u = helper - np.dot(helper, axis) * axis
        u_n = float(np.linalg.norm(u))
    u = u / max(u_n, 1e-9)
    v = np.cross(axis, u)

    def _sample(p):
        h = np.array([float(p[0]), float(p[1]), float(p[2]), 1.0])
        ijk = (ras_to_ijk_mat @ h)[:3]
        i = int(np.clip(round(ijk[0]), 0, I - 1))
        j = int(np.clip(round(ijk[1]), 0, J - 1))
        k = int(np.clip(round(ijk[2]), 0, K - 1))
        return float(log_arr[k, j, i])

    # Disk sampling: center + 8 angles × 2 radii = 17 samples per step.
    # Matches the peak-driven Contacts & Trajectory View engine's
    # perp-sampling pattern.
    disk_offsets = [(0.0, 0.0)]
    for radius in (1.5, 2.5):
        for ang in np.linspace(0.0, 2.0 * np.pi, num=8, endpoint=False):
            disk_offsets.append((radius * float(np.cos(ang)),
                                  radius * float(np.sin(ang))))

    # Build neighbor segments for ownership masking (skip disk voxels
    # closer to a partner's axis than to ours).
    other_seg_pairs = []
    if others:
        for o in others:
            o_s = np.asarray(o.get("start_ras"), dtype=float)
            o_e = np.asarray(o.get("end_ras"), dtype=float)
            if np.linalg.norm(o_e - o_s) > 1e-9:
                other_seg_pairs.append((o_s, o_e))

    def _perp_to_others(p):
        best = np.inf
        for o_s, o_e in other_seg_pairs:
            ab = o_e - o_s
            LL2 = float(np.dot(ab, ab))
            if LL2 < 1e-18:
                continue
            t = float(np.clip(np.dot(p - o_s, ab) / LL2, 0.0, 1.0))
            d = float(np.linalg.norm(p - (o_s + t * ab)))
            if d < best:
                best = d
        return best

    def _disk_hit(p_axis):
        for dr_u, dr_v in disk_offsets:
            p = p_axis + dr_u * u + dr_v * v
            if other_seg_pairs:
                dist_self = float(np.hypot(dr_u, dr_v))
                if _perp_to_others(p) < dist_self:
                    continue  # voxel is closer to a neighbour — skip
            if _sample(p) <= -min_abs_log:
                return True
        return False

    n_steps = int(max_extend_mm / step_mm)
    miss_steps_allowed = max(1, int(miss_mm / step_mm))
    last_hit = None
    if _disk_hit(end):
        last_hit = end.copy()
    consecutive_miss = 0
    for s in range(1, n_steps + 1):
        p = end + s * step_mm * axis
        if _disk_hit(p):
            last_hit = p
            consecutive_miss = 0
        else:
            consecutive_miss += 1
            if consecutive_miss >= miss_steps_allowed:
                break
    return last_hit


def clip_deep_end_to_inliers(rec,
                             margin_mm=DEEP_END_MARGIN_PAST_LAST_CONTACT_MM):
    """Clip ``end_ras`` back so it sits no more than ``margin_mm`` past
    the deepest walker inlier projected onto the shank axis.

    No-op when the trajectory has no ``inlier_ras``.
    """
    inliers = rec.get("inlier_ras")
    if inliers is None or len(inliers) == 0:
        return None
    start = np.asarray(rec.get("start_ras"), dtype=float)
    end = np.asarray(rec.get("end_ras"), dtype=float)
    d = end - start
    L = float(np.linalg.norm(d))
    if L < 1e-3:
        return None
    axis = d / L
    pts = np.asarray(inliers, dtype=float)
    proj = (pts - start) @ axis
    max_proj = float(proj.max())
    ceiling = max_proj + float(margin_mm)
    if L <= ceiling:
        return None
    return start + ceiling * axis


__all__ = [
    "refine_deep_end_via_axis_log",
    "clip_deep_end_to_inliers",
]
