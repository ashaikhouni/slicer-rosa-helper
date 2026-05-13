"""Stage 1 orchestrator: walker + arbitration + extension + dedup.

Composes the lower-level walker (``walker.py``) into the full stage-1
candidate-seed pipeline. ``run_stage1`` is the public entry point used
by ``run_two_stage_detection`` (still in cpfit during the staged
extraction).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..primitives.geometry import orient_shallow_to_deep
from .blob_extraction import extract_blobs
from .constants import (
    DEEP_TIP_MIN_MM,
    DEEP_TIP_MIN_SHORT_MM,
    DEEP_TIP_SHORT_MAX_AVG_PITCH_MM,
    EXTEND_MAX_EXTRA,
    EXTEND_MAX_GAP_MM,
    EXTEND_MAX_OUTER_ITER,
    EXTEND_PERP_TOL_MM,
    FRANGI_LINE_MIN_MEDIAN,
    LOG_BLOB_MAX_VOXELS,
    LOG_BLOB_THRESHOLD,
    MIN_BLOBS_PER_LINE,
    MIN_BLOBS_POST_ARBITRATION,
    PITCH_MM,
    PITCH_TOL_MM,
    STAGE1_DEDUP_ANGLE_DEG,
    STAGE1_DEDUP_OVERLAP_FRAC,
    STAGE1_DEDUP_PERP_MM,
)
from .frangi_sampling import frangi_along_line_stats, median_inlier_pitch
from .pitch_library import DEFAULT_WALKER_BOUNDS, WalkerBounds
from .walker import refit_line_from_inliers, walk_line


def arbitrate_blob_ownership(stage1_lines, pts_c, amps_c,
                              bounds: WalkerBounds | None = None):
    """Resolve inlier contention. For each blob claimed by multiple
    lines, award it to the line whose axis is closest (smallest perp
    distance). Lines re-fit on reduced inlier sets; those below the
    MIN_BLOBS / MIN_SPAN floors are dropped.
    """
    if len(stage1_lines) < 2:
        return stage1_lines

    claims: dict[int, list[tuple[int, float]]] = {}
    for li, line in enumerate(stage1_lines):
        axis = np.asarray(line["axis"], dtype=float)
        center = np.asarray(line["center"], dtype=float)
        for bi in line["inlier_idx"]:
            diff = pts_c[bi] - center
            along = float(np.dot(diff, axis))
            perp = diff - along * axis
            perp_d = float(np.linalg.norm(perp))
            claims.setdefault(int(bi), []).append((li, perp_d))

    keep_sets = [set(l["inlier_idx"]) for l in stage1_lines]
    for bi, owners in claims.items():
        if len(owners) <= 1:
            continue
        owners.sort(key=lambda x: x[1])
        for li, _ in owners[1:]:
            keep_sets[li].discard(bi)

    kept: list[dict[str, Any]] = []
    for li, line in enumerate(stage1_lines):
        new_inliers = sorted(keep_sets[li])
        if len(new_inliers) < MIN_BLOBS_POST_ARBITRATION:
            continue
        new_line = dict(line)
        new_line["inlier_idx"] = new_inliers
        refit = refit_line_from_inliers(
            new_line, pts_c, amps_c, min_blobs=MIN_BLOBS_POST_ARBITRATION,
            bounds=bounds,
        )
        if refit is not None:
            kept.append(refit)
    return kept


def second_pass_orphan_walker(existing_lines, pts_c, amps_c,
                               pitches_mm=(PITCH_MM,),
                               bounds: WalkerBounds | None = None):
    """Re-run the pitch walker on blobs not claimed by any surviving
    line. Recovers electrodes whose only first-pass hypothesis was a
    bridging line that arbitration killed.
    """
    claimed: set[int] = set()
    for l in existing_lines:
        for bi in l["inlier_idx"]:
            claimed.add(int(bi))
    orphan_idx = [bi for bi in range(len(pts_c)) if bi not in claimed]
    if len(orphan_idx) < MIN_BLOBS_PER_LINE:
        return []

    orphan_pts = pts_c[orphan_idx]
    orphan_amps = amps_c[orphan_idx]
    dist = np.sqrt(np.sum((orphan_pts[:, None, :] - orphan_pts[None, :, :]) ** 2, axis=2))

    new_hyps: list[dict[str, Any]] = []
    for pitch in pitches_mm:
        pair_mask = np.zeros_like(dist, dtype=bool)
        for mult in (1, 2, 3):
            lo = mult * pitch - PITCH_TOL_MM
            hi = mult * pitch + PITCH_TOL_MM
            pair_mask |= (dist >= lo) & (dist <= hi)
        iu, ju = np.where(np.triu(pair_mask, k=1))
        for pi, pj in zip(iu, ju):
            h = walk_line(int(pi), int(pj), orphan_pts, orphan_amps,
                           pitch_mm=pitch, bounds=bounds)
            if h is None:
                continue
            h["inlier_idx"] = [int(orphan_idx[i]) for i in h["inlier_idx"]]
            h["seed_pitch_mm"] = float(pitch)
            new_hyps.append(h)
    if not new_hyps:
        return []
    new_hyps.sort(key=lambda h: -h["n_blobs"])
    new_lines = dedup_stage1_lines(new_hyps)
    return new_lines


def extend_deep_end(line, pts_c, amps_c, claimed_blobs,
                     dist_arr=None, ras_to_ijk_mat=None,
                     max_gap_mm: float = EXTEND_MAX_GAP_MM,
                     perp_tol_mm: float = EXTEND_PERP_TOL_MM,
                     max_extra: int = EXTEND_MAX_EXTRA,
                     max_outer_iter: int = EXTEND_MAX_OUTER_ITER,
                     bounds: WalkerBounds | None = None):
    """Walk outward from the current deepest AND shallowest inliers,
    snapping to unclaimed blobs within ``max_gap_mm`` along the axis.
    Refits axis after each pass; iterates until convergence.
    """
    for _outer in range(max_outer_iter):
        s_ras = np.asarray(line["start_ras"], dtype=float)
        e_ras = np.asarray(line["end_ras"], dtype=float)
        if dist_arr is not None and ras_to_ijk_mat is not None:
            s_ras, e_ras = orient_shallow_to_deep(
                s_ras, e_ras, dist_arr, ras_to_ijk_mat,
            )
            line["start_ras"] = s_ras
            line["end_ras"] = e_ras
        d_ras = e_ras - s_ras
        L_line = float(np.linalg.norm(d_ras))
        if L_line < 1e-6:
            break
        axis = d_ras / L_line  # shallow → deep
        center = np.asarray(line["center"], dtype=float)
        inliers = list(line["inlier_idx"])
        n_pre = len(inliers)
        diffs_all = pts_c - center
        along_all = diffs_all @ axis
        perp_all = np.linalg.norm(
            diffs_all - np.outer(along_all, axis), axis=1,
        )
        # Deep-side walk.
        deep_proj = float(((pts_c[inliers] - center) @ axis).max())
        for _ in range(max_extra):
            candidate_mask = (
                (along_all > deep_proj)
                & (along_all - deep_proj <= max_gap_mm)
                & (perp_all <= perp_tol_mm)
            )
            cand = [int(bi) for bi in np.where(candidate_mask)[0]
                    if int(bi) not in claimed_blobs
                    and int(bi) not in set(inliers)]
            if not cand:
                break
            best = max(cand, key=lambda bi: float(amps_c[bi]))
            inliers.append(best)
            claimed_blobs.add(best)
            deep_proj = float(along_all[best])
        # Shallow-side walk.
        shallow_proj = float(((pts_c[inliers] - center) @ axis).min())
        for _ in range(max_extra):
            candidate_mask = (
                (along_all < shallow_proj)
                & (shallow_proj - along_all <= max_gap_mm)
                & (perp_all <= perp_tol_mm)
            )
            cand = [int(bi) for bi in np.where(candidate_mask)[0]
                    if int(bi) not in claimed_blobs
                    and int(bi) not in set(inliers)]
            if not cand:
                break
            best = max(cand, key=lambda bi: float(amps_c[bi]))
            inliers.append(best)
            claimed_blobs.add(best)
            shallow_proj = float(along_all[best])
        if len(inliers) == n_pre:
            break  # converged
        line["inlier_idx"] = sorted(inliers)
        refit = refit_line_from_inliers(line, pts_c, amps_c, bounds=bounds)
        if refit is None:
            break
        line = refit
    return line


def extend_deep_end_two_pass(lines, pts_c, amps_c,
                              dist_arr=None, ras_to_ijk_mat=None,
                              max_gap_mm: float = EXTEND_MAX_GAP_MM,
                              perp_tol_mm: float = EXTEND_PERP_TOL_MM,
                              max_extra: int = EXTEND_MAX_EXTRA,
                              max_outer_iter: int = EXTEND_MAX_OUTER_ITER,
                              bounds: WalkerBounds | None = None,
                              perp_margin_mm: float = 0.5):
    """Two-pass deep-end extension with conflict resolution.

    Replaces the per-line greedy ``extend_deep_end`` loop that mutated
    a shared ``claimed_blobs`` set in amp-sum order. The greedy version
    let the first-processed line grab kissing-partner contacts that
    sat within ``perp_tol_mm`` of its axis (T8/6 stealing T8/9's tip;
    T9/8,11; T18/4,6; T1/6 similar) — then refit the axis to include
    them, making the leak invisible to downstream perp-based checks.

    Two-pass design:
      Pass 1 (propose, no exclusion): each line independently identifies
        its next deep-side and shallow-side candidate blob, ignoring
        already-claimed blobs from other lines.
      Pass 2 (resolve & apply): for each blob proposed by >1 line, pick
        the owner using
          (a) perp-to-axis: line whose axis passes closer wins when the
              perp difference exceeds ``perp_margin_mm``;
          (b) chain-pitch tiebreaker (colinear-kissing case): the
              candidate arc closest to that line's deepest-inlier-arc
              + typical_pitch wins.
      Refit every line whose inliers changed. Loop until no new claims.

    Single-claim blobs (one line wants them) are assigned to their sole
    claimant — orphan-grab semantics for CM/BM cluster gaps are
    preserved. Conflict resolution only kicks in when 2+ lines reach.
    """
    n = len(lines)
    if n == 0:
        return lines
    pts = np.asarray(pts_c, dtype=float)
    amps = np.asarray(amps_c, dtype=float)

    def _state(line):
        s = np.asarray(line["start_ras"], dtype=float)
        e = np.asarray(line["end_ras"], dtype=float)
        d = e - s
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            return None
        axis = d / L
        center = np.asarray(line["center"], dtype=float)
        diffs = pts - center
        along = diffs @ axis
        perp = np.linalg.norm(
            diffs - np.outer(along, axis), axis=1,
        )
        return {"s": s, "e": e, "axis": axis, "center": center,
                "along": along, "perp": perp}

    def _typical_pitch(line, state):
        inliers = list(line["inlier_idx"])
        if len(inliers) < 2:
            return 3.5
        arcs = sorted(float(state["along"][int(bi)]) for bi in inliers)
        gaps = [arcs[i + 1] - arcs[i] for i in range(len(arcs) - 1)]
        gaps_plausible = [g for g in gaps if 1.5 <= g <= 8.0]
        if gaps_plausible:
            return float(np.median(gaps_plausible))
        return float(np.median(gaps)) if gaps else 3.5

    for outer in range(max_outer_iter):
        states = []
        for line in lines:
            if dist_arr is not None and ras_to_ijk_mat is not None:
                s_ras = np.asarray(line["start_ras"], dtype=float)
                e_ras = np.asarray(line["end_ras"], dtype=float)
                s_ras, e_ras = orient_shallow_to_deep(
                    s_ras, e_ras, dist_arr, ras_to_ijk_mat,
                )
                line["start_ras"] = s_ras
                line["end_ras"] = e_ras
            states.append(_state(line))

        proposals = {}
        for li, line in enumerate(lines):
            st = states[li]
            if st is None: continue
            inliers = set(line["inlier_idx"])
            inlier_arcs_self = [float(st["along"][int(bi)]) for bi in inliers]
            if not inlier_arcs_self:
                continue
            deep_proj = max(inlier_arcs_self)
            shallow_proj = min(inlier_arcs_self)

            # Deep candidate
            mask_d = ((st["along"] > deep_proj)
                      & (st["along"] - deep_proj <= max_gap_mm)
                      & (st["perp"] <= perp_tol_mm))
            cand_d = [int(bi) for bi in np.where(mask_d)[0]
                       if int(bi) not in inliers]
            if cand_d:
                best = max(cand_d, key=lambda bi: float(amps[bi]))
                proposals.setdefault(best, []).append(
                    (li, "deep", float(st["along"][best]),
                     float(st["perp"][best]))
                )
            # Shallow candidate
            mask_s = ((st["along"] < shallow_proj)
                      & (shallow_proj - st["along"] <= max_gap_mm)
                      & (st["perp"] <= perp_tol_mm))
            cand_s = [int(bi) for bi in np.where(mask_s)[0]
                       if int(bi) not in inliers]
            if cand_s:
                best = max(cand_s, key=lambda bi: float(amps[bi]))
                proposals.setdefault(best, []).append(
                    (li, "shallow", float(st["along"][best]),
                     float(st["perp"][best]))
                )

        if not proposals:
            break

        any_added = False
        for blob_idx, claimants in proposals.items():
            if len(claimants) == 1:
                li, side, arc, perp = claimants[0]
                if int(blob_idx) not in set(lines[li]["inlier_idx"]):
                    lines[li]["inlier_idx"].append(int(blob_idx))
                    any_added = True
                continue
            sorted_by_perp = sorted(claimants, key=lambda c: c[3])
            best = sorted_by_perp[0]
            runner = sorted_by_perp[1]
            if runner[3] - best[3] >= perp_margin_mm:
                winner = best
            else:
                best_score = float("inf")
                winner = best
                for c in claimants:
                    li, side, arc, _ = c
                    pitch = _typical_pitch(lines[li], states[li])
                    inliers = list(lines[li]["inlier_idx"])
                    arcs_self = [float(states[li]["along"][int(bi)]) for bi in inliers]
                    if not arcs_self: continue
                    if side == "deep":
                        own_extreme = max(arcs_self)
                        expected = own_extreme + pitch
                    else:
                        own_extreme = min(arcs_self)
                        expected = own_extreme - pitch
                    score = abs(arc - expected)
                    if score < best_score:
                        best_score = score; winner = c
            wli, _, _, _ = winner
            if int(blob_idx) not in set(lines[wli]["inlier_idx"]):
                lines[wli]["inlier_idx"].append(int(blob_idx))
                any_added = True

        if not any_added:
            break

        new_lines = []
        for line in lines:
            line["inlier_idx"] = sorted(line["inlier_idx"])
            refit = refit_line_from_inliers(line, pts_c, amps_c, bounds=bounds)
            if refit is None:
                new_lines.append(line)
            else:
                new_lines.append(refit)
        lines = new_lines

    return lines


def dedup_stage1_lines(lines):
    """Remove near-duplicate stage-1 walker hypotheses.

    For every (i, j), i < j, a duplicate requires all three of:
      * axis angle ≤ STAGE1_DEDUP_ANGLE_DEG
      * perpendicular center offset ≤ STAGE1_DEDUP_PERP_MM
      * overlap / shorter ≥ STAGE1_DEDUP_OVERLAP_FRAC
    """
    n = len(lines)
    if n < 2:
        return list(lines)

    axes = np.stack([np.asarray(l["axis"], dtype=float) for l in lines])
    centers = np.stack([np.asarray(l["center"], dtype=float) for l in lines])
    span_lo = np.array([float(l["span_lo"]) for l in lines])
    span_hi = np.array([float(l["span_hi"]) for l in lines])
    n_blobs = np.array([int(l["n_blobs"]) for l in lines])

    dots = np.clip(np.abs(axes @ axes.T), 0.0, 1.0)
    ang_ok = np.degrees(np.arccos(dots)) <= STAGE1_DEDUP_ANGLE_DEG

    M = axes @ centers.T
    axes_dot_center = np.einsum("ik,ik->i", axes, centers)
    par = M - axes_dot_center[:, None]

    C2 = np.einsum("ik,ik->i", centers, centers)
    cc = centers @ centers.T
    d2 = C2[:, None] + C2[None, :] - 2.0 * cc
    perp_sq = np.maximum(0.0, d2 - par * par)
    perp_ok = perp_sq <= (STAGE1_DEDUP_PERP_MM * STAGE1_DEDUP_PERP_MM)

    b_lo = par + span_lo[None, :]
    b_hi = par + span_hi[None, :]
    a_lo = span_lo[:, None]
    a_hi = span_hi[:, None]
    overlap = np.maximum(0.0, np.minimum(a_hi, b_hi) - np.maximum(a_lo, b_lo))
    a_len = (span_hi - span_lo)[:, None]
    b_len = (span_hi - span_lo)[None, :]
    shorter = np.minimum(a_len, b_len)
    safe_shorter = np.where(shorter > 1e-6, shorter, 1.0)
    frac = overlap / safe_shorter
    overlap_ok = (shorter > 1e-6) & (frac >= STAGE1_DEDUP_OVERLAP_FRAC)

    hit = ang_ok & perp_ok & overlap_ok
    np.fill_diagonal(hit, False)

    alive = np.ones(n, dtype=bool)
    for i in range(n):
        if not alive[i]:
            continue
        row = hit[i].copy()
        row[: i + 1] = False
        row &= alive
        if not row.any():
            continue
        if np.any(row & (n_blobs > n_blobs[i])):
            alive[i] = False
            continue
        alive[row] = False

    return [lines[i] for i in range(n) if alive[i]]


def run_stage1(log_arr, kji_to_ras_fn, dist_arr, ras_to_ijk_mat,
                pitches_mm=None, frangi_arr=None,
                bounds: WalkerBounds | None = None):
    """Blob-pitch detector on the LoG σ=1 field.

    Returns ``(lines, pts_c)`` where pts_c are the contact-sized blob
    RAS positions used for stage-1 exclusion construction downstream.
    """
    if bounds is None:
        bounds = DEFAULT_WALKER_BOUNDS
    if pitches_mm is None or len(tuple(pitches_mm)) == 0:
        pitches_mm = (PITCH_MM,)
    pitches_mm = tuple(float(p) for p in pitches_mm)
    blobs = extract_blobs(log_arr, threshold=LOG_BLOB_THRESHOLD)
    if not blobs:
        return [], np.empty((0, 3), dtype=float)
    pts_ras = np.array([kji_to_ras_fn(b["kji"]) for b in blobs])
    amps = np.array([b["amp"] for b in blobs], dtype=float)
    n_vox = np.array([b["n_vox"] for b in blobs], dtype=int)
    contact_mask = n_vox <= LOG_BLOB_MAX_VOXELS
    pts_c = pts_ras[contact_mask]
    amps_c = amps[contact_mask]
    if pts_c.shape[0] < 2:
        return [], pts_c

    dist = np.sqrt(np.sum((pts_c[:, None, :] - pts_c[None, :, :]) ** 2, axis=2))

    hyps = []
    for pitch in pitches_mm:
        pair_mask = np.zeros_like(dist, dtype=bool)
        for mult in (1, 2, 3):
            lo = mult * pitch - PITCH_TOL_MM
            hi = mult * pitch + PITCH_TOL_MM
            pair_mask |= (dist >= lo) & (dist <= hi)
        iu, ju = np.where(np.triu(pair_mask, k=1))
        for pi, pj in zip(iu, ju):
            h = walk_line(int(pi), int(pj), pts_c, amps_c, pitch_mm=pitch,
                           bounds=bounds)
            if h is not None:
                h["seed_pitch_mm"] = float(pitch)
                hyps.append(h)
    hyps.sort(key=lambda h: -h["n_blobs"])
    lines = dedup_stage1_lines(hyps)
    if frangi_arr is not None:
        for l in lines:
            fmean, fmed = frangi_along_line_stats(
                l["start_ras"], l["end_ras"], frangi_arr, ras_to_ijk_mat,
            )
            l["frangi_mean_mm"] = fmean
            l["frangi_median_mm"] = fmed
        lines = [l for l in lines
                 if l.get("frangi_median_mm", 0.0) >= FRANGI_LINE_MIN_MEDIAN]
    for l in lines:
        l.setdefault("original_span_mm", float(l.get("span_mm", 0.0)))
        if "original_median_pitch_mm" not in l:
            l["original_median_pitch_mm"] = median_inlier_pitch(
                pts_c[l["inlier_idx"]], l["axis"],
            )
    lines = arbitrate_blob_ownership(lines, pts_c, amps_c, bounds=bounds)

    # Two-pass deep-end extension with conflict resolution. Replaces
    # the per-line greedy ``extend_deep_end`` loop that mutated a
    # shared ``claimed`` set in amp-sum order, letting the
    # first-processed line steal kissing-partner contacts (T8/6 leak).
    lines = extend_deep_end_two_pass(
        lines, pts_c, amps_c,
        dist_arr=dist_arr, ras_to_ijk_mat=ras_to_ijk_mat,
        bounds=bounds,
    )

    second_pass_lines = second_pass_orphan_walker(
        lines, pts_c, amps_c, pitches_mm=pitches_mm, bounds=bounds,
    )
    if second_pass_lines and frangi_arr is not None:
        kept_sp = []
        for nl in second_pass_lines:
            fmean, fmed = frangi_along_line_stats(
                nl["start_ras"], nl["end_ras"], frangi_arr, ras_to_ijk_mat,
            )
            nl["frangi_mean_mm"] = fmean
            nl["frangi_median_mm"] = fmed
            if fmed >= FRANGI_LINE_MIN_MEDIAN:
                kept_sp.append(nl)
        second_pass_lines = kept_sp
    if second_pass_lines:
        # Rebuild `claimed` from current stage1 lines (the two-pass
        # extend above doesn't keep one), then add the second-pass
        # orphan walker's own inliers before letting them extend.
        claimed: set[int] = set()
        for l in lines:
            for bi in l["inlier_idx"]:
                claimed.add(int(bi))
        for nl in second_pass_lines:
            for bi in nl["inlier_idx"]:
                claimed.add(int(bi))
            nl.setdefault("original_span_mm", float(nl.get("span_mm", 0.0)))
            if "original_median_pitch_mm" not in nl:
                nl["original_median_pitch_mm"] = median_inlier_pitch(
                    pts_c[nl["inlier_idx"]], nl["axis"],
                )
        second_pass_lines = [
            extend_deep_end(nl, pts_c, amps_c, claimed,
                             dist_arr=dist_arr, ras_to_ijk_mat=ras_to_ijk_mat,
                             bounds=bounds)
            for nl in second_pass_lines
        ]
        lines = lines + second_pass_lines
        lines.sort(key=lambda l: -l["n_blobs"])
        lines = dedup_stage1_lines(lines)

    K, J, I = dist_arr.shape
    kept = []
    for l in lines:
        inlier_ras = pts_c[l["inlier_idx"]]
        h = np.concatenate([inlier_ras, np.ones((inlier_ras.shape[0], 1))],
                           axis=1)
        ijk = (ras_to_ijk_mat @ h.T).T[:, :3]
        ii = np.clip(np.round(ijk[:, 0]).astype(int), 0, I - 1)
        jj = np.clip(np.round(ijk[:, 1]).astype(int), 0, J - 1)
        kk = np.clip(np.round(ijk[:, 2]).astype(int), 0, K - 1)
        inlier_dists = dist_arr[kk, jj, ii]
        l["dist_min_mm"] = float(inlier_dists.min())
        l["dist_max_mm"] = float(inlier_dists.max())
        l["dist_mean_mm"] = float(inlier_dists.mean())
        median_pitch = float(l.get(
            "original_median_pitch_mm",
            (float(l.get("original_span_mm", l.get("span_mm", 0.0)))
             / max(1, int(l.get("n_blobs", 2)) - 1)),
        ))
        looks_like_seeg = median_pitch <= DEEP_TIP_SHORT_MAX_AVG_PITCH_MM
        min_dist = DEEP_TIP_MIN_SHORT_MM if looks_like_seeg else DEEP_TIP_MIN_MM
        if l["dist_max_mm"] < min_dist:
            continue
        l["start_ras"], l["end_ras"] = orient_shallow_to_deep(
            l["start_ras"], l["end_ras"], dist_arr, ras_to_ijk_mat,
        )
        kept.append(l)
    return kept, pts_c


__all__ = [
    "arbitrate_blob_ownership",
    "second_pass_orphan_walker",
    "extend_deep_end",
    "extend_deep_end_two_pass",
    "dedup_stage1_lines",
    "run_stage1",
]
