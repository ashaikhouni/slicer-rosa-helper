"""Shared-CC tip refinement.

When two trajectories' deep tips converge on the same LoG connected
component — typical of two antiparallel shanks whose deepest contacts
sit at the colinear midline — the walker has absorbed the other
shank's deepest contact as its own (a cross-shank bridge). Because the
two shanks cross at a small angle, each tip sits ~3 mm perpendicular
to the other segment, just outside the
``CROSSING_TIP_CLEARANCE_MM = 2.0`` trigger of
:func:`retreat_crossing_tips` — so geometric retreat doesn't engage.

This stage resolves those by perp-ownership splitting of the shared CC:

  1. Identify trajectory pairs whose tips are within ``r_conflict_mm``.
  2. For each pair, locate the LoG CC visited deepest by both
     trajectories. If the deepest visited label differs, no refinement.
  3. Split the shared CC into sub-contacts via local |LoG| maxima
     (one peak per contact-sized window). Each voxel is assigned to
     its nearest peak.
  4. Each sub-contact's owner is the trajectory with the smaller
     perp-to-axis distance (margin ``perp_margin_mm``).
  5. Each trajectory's new ``end_ras`` becomes the deep edge of its
     deepest owned sub-contact.

Verified on T1 X04/X07 (DIXI-15CM antiparallel midline pair): retreats
X04 tip by 5.6 mm and X07 tip by 6.3 mm, snapping each to its real
deepest contact.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import center_of_mass
from scipy.ndimage import label as ndi_label
from scipy.ndimage import maximum_filter

from .constants import LOG_BLOB_THRESHOLD

SHARED_CC_CONTACT_ABS_LOG = LOG_BLOB_THRESHOLD
SHARED_CC_R_OWN_MM = 3.0
SHARED_CC_PEAK_NEIGHBORHOOD = 3
SHARED_CC_OWNER_PERP_MARGIN_MM = 0.3
SHARED_CC_R_CONFLICT_MM = 5.0


def _closest_pt_on_seg(p, a, b):
    p = np.asarray(p, float); a = np.asarray(a, float); b = np.asarray(b, float)
    ab = b - a; L = float(np.linalg.norm(ab))
    if L < 1e-9:
        return a, 0.0, float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / (L * L), 0.0, 1.0))
    return a + t * ab, t * L, float(np.linalg.norm(p - (a + t * ab)))


def _build_chain(seg, cc_centroids, r_perp):
    s = seg["s"]; e = seg["e"]; L = seg["L"]
    chain = []
    for lab, c in cc_centroids.items():
        _, arc, perp = _closest_pt_on_seg(c, s, e)
        if perp <= r_perp and 0.0 <= arc <= L:
            chain.append((arc, lab, c))
    chain.sort()
    return chain


def _split_cc_by_local_max(abs_log, label_vol, target_lab, i2r,
                           sA, eA, sB, eB,
                           peak_size, contact_abs_log, perp_margin):
    aA = (eA - sA) / np.linalg.norm(eA - sA)
    aB = (eB - sB) / np.linalg.norm(eB - sB)
    mask_lab = (label_vol == target_lab)
    abs_log_in = np.where(mask_lab, abs_log, 0.0)
    maxf = maximum_filter(abs_log_in, size=peak_size)
    is_peak = mask_lab & (abs_log_in == maxf) & (abs_log_in >= contact_abs_log)
    peak_lbl, peak_n = ndi_label(is_peak)
    if peak_n == 0:
        return []
    cc_voxels = np.argwhere(mask_lab)
    peaks_by_lbl = {pl: np.argwhere(peak_lbl == pl)
                    for pl in range(1, peak_n + 1)}
    assignment = np.zeros_like(label_vol, dtype=np.int32)
    for vk, vj, vi in cc_voxels:
        best_d = np.inf; best_l = 0
        for pl, pv in peaks_by_lbl.items():
            d = ((pv - np.array([vk, vj, vi])) ** 2).sum(axis=1).min()
            if d < best_d:
                best_d = d; best_l = pl
        assignment[vk, vj, vi] = best_l
    out = []
    for pl in range(1, peak_n + 1):
        coords = np.argwhere(assignment == pl)
        if len(coords) == 0:
            continue
        h_all = np.column_stack([coords[:, 2], coords[:, 1], coords[:, 0],
                                  np.ones(len(coords))])
        ras_all = (h_all @ i2r.T)[:, :3]
        centroid = ras_all.mean(axis=0)
        arcs_A = (ras_all - sA) @ aA
        arcs_B = (ras_all - sB) @ aB
        _, _, perp_A = _closest_pt_on_seg(centroid, sA, eA)
        _, _, perp_B = _closest_pt_on_seg(centroid, sB, eB)
        if perp_A < perp_B - perp_margin:
            owner = "A"
        elif perp_B < perp_A - perp_margin:
            owner = "B"
        else:
            owner = "equal"
        out.append(dict(
            end_arc_A=float(arcs_A.max()),
            end_arc_B=float(arcs_B.max()),
            owner=owner,
        ))
    return out


def _retreat_loser_to_deepest_inlier(rec, seg):
    """For the trajectory that owns no sub-contact in a shared CC, its
    ``end_ras`` has drifted past the last real inlier into the winner's
    tip region (via ``DEEP_END_MARGIN_PAST_LAST_CONTACT_MM``). Retreat
    ``end_ras`` along the trajectory's own axis to the projection of
    its deepest inlier — no stripping needed since the loser's inlier
    set is already its true contact chain. Returns the retreat distance
    in mm (0 if no retreat applied).
    """
    inliers = rec.get("inlier_ras")
    if inliers is None:
        return 0.0
    inliers = np.asarray(inliers, dtype=float)
    if inliers.size == 0:
        return 0.0
    arcs = (inliers - seg["s"]) @ seg["a"]
    new_arc = float(arcs.max())
    if new_arc >= seg["L"] - 1e-3:
        return 0.0  # already at or past deepest inlier — nothing to do
    new_end = seg["s"] + new_arc * seg["a"]
    rec["end_ras"] = new_end
    rec["length_mm"] = float(np.linalg.norm(new_end - seg["s"]))
    return float(seg["L"] - new_arc)


def _strip_inliers_past(rec, s_ras, axis, max_arc, tol_mm=0.5):
    inliers = rec.get("inlier_ras")
    if inliers is None:
        return
    inliers = np.asarray(inliers, dtype=float)
    if inliers.size == 0:
        return
    arcs = (inliers - s_ras) @ axis
    keep = arcs <= max_arc + tol_mm
    if bool(np.all(keep)):
        return
    rec["inlier_ras"] = inliers[keep]


def _refresh_seg(rec, s):
    e = np.asarray(rec["end_ras"], dtype=float)
    d = e - s
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return None
    return {"rec": rec, "s": s, "e": e, "a": d / L, "L": L}


def refine_shared_cc_tips(anchored, log_arr, ras_to_ijk_mat,
                          r_conflict_mm=SHARED_CC_R_CONFLICT_MM,
                          r_own_mm=SHARED_CC_R_OWN_MM,
                          contact_abs_log=SHARED_CC_CONTACT_ABS_LOG,
                          peak_size=SHARED_CC_PEAK_NEIGHBORHOOD,
                          perp_margin_mm=SHARED_CC_OWNER_PERP_MARGIN_MM,
                          logger=None):
    """Resolve cross-shank tip bridges via shared-CC ownership splitting.

    See module docstring for the algorithm. Mutates ``anchored`` in
    place — updates each refined record's ``end_ras`` and ``length_mm``.
    """
    if log_arr is None or ras_to_ijk_mat is None:
        return
    r2i = np.asarray(ras_to_ijk_mat, dtype=float)
    i2r = np.linalg.inv(r2i)

    segs = []
    for rec in anchored:
        s = np.asarray(rec.get("start_ras"), dtype=float)
        seg = _refresh_seg(rec, s)
        segs.append(seg)

    # Compute global LoG CCs once.
    mask = log_arr <= -contact_abs_log
    label_vol, n_cc = ndi_label(mask)
    if n_cc == 0:
        return
    coms = center_of_mass(mask, label_vol, range(1, n_cc + 1))
    cc_centroids = {}
    for li, (k, j, i) in enumerate(coms, start=1):
        h = np.array([i, j, k, 1.0])
        cc_centroids[li] = (i2r @ h)[:3]
    abs_log = -log_arr

    seen = set()
    for ai, sa in enumerate(segs):
        if sa is None:
            continue
        for bi, sb in enumerate(segs):
            if bi == ai or sb is None:
                continue
            key = tuple(sorted([ai, bi]))
            if key in seen:
                continue
            _, _, perp_a_on_b = _closest_pt_on_seg(sa["e"], sb["s"], sb["e"])
            _, _, perp_b_on_a = _closest_pt_on_seg(sb["e"], sa["s"], sa["e"])
            if min(perp_a_on_b, perp_b_on_a) > r_conflict_mm:
                continue
            seen.add(key)

            chA = _build_chain(sa, cc_centroids, r_own_mm)
            chB = _build_chain(sb, cc_centroids, r_own_mm)
            if not chA or not chB:
                continue
            lA = chA[-1][1]; lB = chB[-1][1]
            if lA != lB:
                continue  # tips don't share a CC

            subs = _split_cc_by_local_max(
                abs_log, label_vol, lA, i2r,
                sa["s"], sa["e"], sb["s"], sb["e"],
                peak_size=peak_size,
                contact_abs_log=contact_abs_log,
                perp_margin=perp_margin_mm,
            )
            if not subs:
                continue
            a_subs = [s for s in subs if s["owner"] in ("A", "equal")]
            b_subs = [s for s in subs if s["owner"] in ("B", "equal")]

            # Asymmetric case (e.g. T2 ti=6/LAMC ↔ ti=10/RAMC midline
            # pair): the shared CC has only one peak and it's owned
            # outright by one trajectory; the loser has no owned
            # sub-contact. Strip the loser's inliers that fall inside
            # the shared CC voxels (they're really the winner's contact
            # that the walker absorbed across the midline), then
            # retreat the loser's end_ras to its new deepest remaining
            # inlier's axial projection.
            if not a_subs or not b_subs:
                retreat_mm = 0.0
                loser_idx = None
                if a_subs and not b_subs:
                    loser_idx = bi
                    retreat_mm = _retreat_loser_to_deepest_inlier(
                        anchored[bi], sb)
                elif b_subs and not a_subs:
                    loser_idx = ai
                    retreat_mm = _retreat_loser_to_deepest_inlier(
                        anchored[ai], sa)
                if loser_idx is not None and retreat_mm > 0.0:
                    segs[loser_idx] = _refresh_seg(
                        anchored[loser_idx],
                        np.asarray(anchored[loser_idx]["start_ras"], float),
                    )
                if logger is not None:
                    try:
                        logger(
                            f"  shared-CC asymmetric ({ai},{bi}): "
                            f"loser=ti={loser_idx}, retreated "
                            f"{retreat_mm:.2f} mm to deepest inlier "
                            f"(CC label={lA})"
                        )
                    except Exception:
                        pass
                continue

            a_deep = max(a_subs, key=lambda s: s["end_arc_A"])
            b_deep = max(b_subs, key=lambda s: s["end_arc_B"])
            new_eA = sa["s"] + a_deep["end_arc_A"] * sa["a"]
            new_eB = sb["s"] + b_deep["end_arc_B"] * sb["a"]
            dA = a_deep["end_arc_A"] - sa["L"]
            dB = b_deep["end_arc_B"] - sb["L"]

            anchored[ai]["end_ras"] = new_eA
            anchored[ai]["length_mm"] = float(np.linalg.norm(new_eA - sa["s"]))
            anchored[bi]["end_ras"] = new_eB
            anchored[bi]["length_mm"] = float(np.linalg.norm(new_eB - sb["s"]))

            # Strip wrong-shank inliers: any inlier whose projection onto
            # the trajectory's own axis sits past the new tip's arc is
            # past the deepest owned sub-contact, i.e. it was an inlier
            # from the other shank that the walker / extend stage
            # absorbed across the midline.
            _strip_inliers_past(anchored[ai], sa["s"], sa["a"],
                                a_deep["end_arc_A"])
            _strip_inliers_past(anchored[bi], sb["s"], sb["a"],
                                b_deep["end_arc_B"])

            segs[ai] = _refresh_seg(anchored[ai], sa["s"])
            segs[bi] = _refresh_seg(anchored[bi], sb["s"])

            if logger is not None:
                try:
                    logger(
                        f"  shared-CC refine pair ({ai},{bi}): "
                        f"dA={dA:+.2f} mm, dB={dB:+.2f} mm "
                        f"(CC label {lA}, split into {len(subs)} subs)"
                    )
                except Exception:
                    pass


__all__ = ["refine_shared_cc_tips"]
