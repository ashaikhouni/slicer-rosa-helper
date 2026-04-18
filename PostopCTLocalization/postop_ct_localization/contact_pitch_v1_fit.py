"""Contact-pitch v1 detector: two-stage LoG+Frangi SEEG shank detection.

Ported directly from ``tests/deep_core/probe_two_stage.py`` + supporting
probes. No gates or filters beyond what the probe uses.

Pipeline:
  * preprocessing: hull mask, intracranial mask, hull signed-distance,
    LoG sigma=1, Frangi sigma=1.
  * stage 1 — blob-pitch: LoG regional-minima blobs + Dixi 3.5 mm pitch
    Hough-style walk, amp_sum gate, deep-tip prior.
  * stage 1 exclusion zone (3 mm tube around every accepted stage-1 line).
  * stage 2 — Frangi shaft fallback: Frangi>=30 cloud minus the exclusion
    zone, CC + PCA, span/aspect filters, hull-endpoint + deep-tip prior.
  * combine stage 1 + stage 2 hypotheses.

See ``slicer-rosa-helper/docs/DEEP_CORE_V2_HANDOFF.md`` for the session
story that produced this design.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


# ---- Config (match probe_two_stage.py + probe_blob_pitch.py) ----------

INTRACRANIAL_MIN_DISTANCE_MM = 10.0
FRANGI_STAGE1_SIGMA = 1.0
LOG_SIGMA_MM = 1.0
LOG_BLOB_THRESHOLD = 300.0
LOG_BLOB_MAX_VOXELS = 500

PITCH_MM = 3.5
PITCH_TOL_MM = 0.5
PERP_TOL_MM = 1.5
AX_TOL_MM = 0.7
MAX_K_STEPS = 20
MIN_BLOBS_PER_LINE = 6
MIN_LINE_SPAN_MM = 15.0
MAX_LINE_SPAN_MM = 90.0
AMP_SUM_MIN = 6000.0
# Maximum allowed gap between consecutive inlier contacts along the
# axis. The walker trims single stray outliers whose gap to the rest
# exceeds this, then rejects any line whose remaining internal gap still
# exceeds it. Set to a high value (effectively off) here — blob
# ownership arbitration (not yet implemented) is the cleaner fix for
# cross-shank bridges. Pre-trim threshold still prunes the worst stray
# blobs to keep the axis clean for bolt anchoring.
MAX_INLIER_GAP_MM = 999.0

STAGE1_DEDUP_ANGLE_DEG = 3.0
STAGE1_DEDUP_PERP_MM = 2.0
STAGE1_DEDUP_OVERLAP_FRAC = 0.3

STAGE2_FRANGI_THR = 30.0
STAGE2_MIN_VOXELS = 30
STAGE2_MAX_VOXELS = 20000
STAGE2_MIN_SPAN_MM = 20.0
STAGE2_MAX_SPAN_MM = 85.0
STAGE2_MAX_PERP_RMS_MM = 3.0
STAGE2_MIN_ASPECT_GEOM = 5.0

FALLBACK_EXCLUSION_MM = 3.0
HULL_ENDPOINT_MAX_MM = 15.0
DEEP_TIP_MIN_MM = 30.0   # bumped from 25 — many sinus / skull-base FPs
                         # have inliers barely past the intracranial
                         # boundary (dist_max 25–28 mm).
# Air-sinus rejection: along the intracranial portion of every trajectory
# (skull_entry → deep tip), sample CT HU at AIR_SAMPLE_COUNT points;
# if more than AIR_FRAC_MAX of those samples are below AIR_HU_THRESHOLD
# (typical air ≈ −1000 HU), reject. Real electrodes traverse brain
# parenchyma (~30 HU) with metal spikes; sinus tubes are hollow.
AIR_HU_THRESHOLD = -300.0
AIR_FRAC_MAX = 0.50  # real shanks crossing ventricles can hit ~35% air
                     # (CSF + small voids). Sinus FPs run >70%.
AIR_SAMPLE_COUNT = 25

# Post-anchor length bounds. Real SEEG = ~25–80 mm shank + ~15–25 mm
# bolt protrusion. Catches stage-2 venous-sinus / vessel false positives
# (>130 mm) and short skull-base hardware FPs (<45 mm).
MIN_POST_ANCHOR_LEN_MM = 45.0
MAX_POST_ANCHOR_LEN_MM = 130.0

# Cross-stage dedup (applied AFTER bolt anchoring).
CROSS_STAGE_DEDUP_ANGLE_DEG = 15.0
CROSS_STAGE_DEDUP_PERP_MM = 8.0

# Bolt detection from LoG σ=1 cloud.
# A bolt CC is any connected blob of bright-metal LoG minima that reaches
# near/outside the hull surface. We deliberately skip the CC's own PCA
# axis — bolt CCs can be giant merged chains (electrode + contacts) or
# tiny fragments, neither of which gives a reliable axis. Instead, the
# anchor step tests whether enough of each bolt CC's voxels sit in a
# narrow tube around the candidate shank axis.
BOLT_LOG_THRESHOLD = 300.0          # |LoG| magnitude gate (matches contact)
BOLT_MIN_VOXELS = 20                # drop tiny (isolated-contact) CCs
BOLT_HULL_PROXIMITY_MM = 2.0        # CC must touch / poke through hull
BOLT_TUBE_RADIUS_MM = 3.0            # shank-axis tube for bolt-voxel count
BOLT_MIN_TUBE_VOXELS = 15            # min CC voxels in tube to accept bolt
BOLT_MAX_INWARD_ALONG_MM = 30.0     # bolt may extend up to this far past
                                    # the shallowest contact (CCs often
                                    # include bolt + first few contacts)
BOLT_SEARCH_OUTWARD_MM = 120.0      # max outward distance to look
BOLT_SHALLOW_HULL_PROX_MM = 15.0    # shallowest tube voxel must reach
                                    # at least this far from hull. The
                                    # bolt CC itself must touch the hull
                                    # (dist_min ≤ 2, set at extract
                                    # time), so this is really an upper
                                    # bound for "the bolt voxels visible
                                    # to the shank tube go OUTWARD-ish".
                                    # 15 mm tolerates ~5° axis-fit error
                                    # in stage 1 over a 50 mm bolt walk.
BOLT_BASE_MAX_DIST_MM = 10.0        # skull_entry_ras (= bolt base, the
                                    # bone→brain transition) must come
                                    # from a tube voxel still in the
                                    # skull/dura band (head_distance ≤
                                    # this). Without this clip, merged
                                    # CCs that include deep contacts
                                    # would put the marker at the deep
                                    # tip instead.


# ---- Preprocessing ----------------------------------------------------

def build_masks(img):
    import SimpleITK as sitk
    # Cast to float so the threshold range is unrestricted by pixel type
    # (some CTs come in as uint16 / int16 where 1e9 overflows ITK's
    # type-coerced upper bound).
    img_f = sitk.Cast(img, sitk.sitkFloat32)
    thr = sitk.BinaryThreshold(img_f, lowerThreshold=-500.0, upperThreshold=1e9,
                                insideValue=1, outsideValue=0)
    cc = sitk.RelabelComponent(sitk.ConnectedComponent(thr), sortByObjectSize=True)
    largest = sitk.Equal(cc, 1)
    closed = sitk.BinaryMorphologicalClosing(largest, kernelRadius=(3, 3, 3))
    hull = sitk.BinaryFillhole(closed)
    dist = sitk.SignedMaurerDistanceMap(
        hull, insideIsPositive=True, squaredDistance=False, useImageSpacing=True,
    )
    dist_arr = sitk.GetArrayFromImage(dist).astype(np.float32)
    hull_arr = sitk.GetArrayFromImage(hull).astype(bool)
    intracranial = dist_arr >= INTRACRANIAL_MIN_DISTANCE_MM
    return hull_arr, intracranial, dist_arr


def frangi_single(img, sigma):
    import SimpleITK as sitk
    sm = sitk.SmoothingRecursiveGaussian(img, sigma=float(sigma))
    ob = sitk.ObjectnessMeasure(sm, objectDimension=1, brightObject=True)
    return sitk.GetArrayFromImage(ob).astype(np.float32)


def log_sigma(img, sigma_mm):
    import SimpleITK as sitk
    log = sitk.LaplacianRecursiveGaussian(img, sigma=float(sigma_mm))
    return sitk.GetArrayFromImage(log).astype(np.float32)


# ---- Stage 1: blob-pitch ---------------------------------------------

def _unit(v):
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def _sample_dist_at_ras(dist_arr, ras_to_ijk_mat, ras_xyz):
    """Look up head_distance at a RAS point (nearest voxel)."""
    K, J, I = dist_arr.shape
    h = np.array([float(ras_xyz[0]), float(ras_xyz[1]), float(ras_xyz[2]), 1.0])
    ijk = (ras_to_ijk_mat @ h)[:3]
    i = int(np.clip(round(ijk[0]), 0, I - 1))
    j = int(np.clip(round(ijk[1]), 0, J - 1))
    k = int(np.clip(round(ijk[2]), 0, K - 1))
    return float(dist_arr[k, j, i])


def _orient_shallow_to_deep(start_ras, end_ras, dist_arr, ras_to_ijk_mat):
    """Return (shallow_ras, deep_ras) so the shallower end (smaller
    head_distance = closer to hull surface) comes first. Disambiguates
    PCA axis direction so downstream visualization can color shallow vs
    deep consistently.
    """
    d_start = _sample_dist_at_ras(dist_arr, ras_to_ijk_mat, start_ras)
    d_end = _sample_dist_at_ras(dist_arr, ras_to_ijk_mat, end_ras)
    if d_start <= d_end:
        return np.asarray(start_ras, dtype=float), np.asarray(end_ras, dtype=float)
    return np.asarray(end_ras, dtype=float), np.asarray(start_ras, dtype=float)


def extract_blobs(log_arr, threshold=LOG_BLOB_THRESHOLD):
    """Regional-minima blob extraction. Each contact (local LoG minimum)
    becomes one blob. Uses SITK grayscale erode to find local minima in a
    ~1 mm radius, then thresholds by absolute LoG value.
    """
    import SimpleITK as sitk
    erode = sitk.GrayscaleErode(
        sitk.GetImageFromArray(log_arr),
        kernelRadius=[2, 2, 2],
    )
    eroded = sitk.GetArrayFromImage(erode).astype(np.float32)
    is_local_min = (log_arr <= eroded + 1e-4)
    strong = is_local_min & (log_arr <= -abs(threshold))
    kk, jj, ii = np.where(strong)
    blobs = []
    for k, j, i in zip(kk, jj, ii):
        val = float(log_arr[k, j, i])
        blobs.append(dict(
            kji=np.array([float(k), float(j), float(i)]),
            amp=-val, n_vox=1,
        ))
    return blobs


def _walk_with_pitch(axis, anchor, pts, amps, pitch, perp_tol, ax_tol, max_k):
    diffs = pts - anchor
    proj = diffs @ axis
    perp = diffs - np.outer(proj, axis)
    perp_d = np.linalg.norm(perp, axis=1)
    within_perp = perp_d <= perp_tol
    inliers = set()
    for k in range(-max_k, max_k + 1):
        target = k * pitch
        ax_resid = np.abs(proj - target)
        cand = np.where(within_perp & (ax_resid <= ax_tol))[0]
        if cand.size == 0:
            continue
        best_blob = cand[np.argmax(amps[cand])]
        inliers.add(int(best_blob))
    if not inliers:
        return None
    return dict(inliers=inliers, pitch=pitch, n_inliers=len(inliers))


def _walk_line(seed_idx, neighbor_idx, pts, amps):
    p0 = pts[seed_idx]
    p1 = pts[neighbor_idx]
    seed_d = float(np.linalg.norm(p1 - p0))
    k_seed = max(1, int(round(seed_d / PITCH_MM)))
    pitch_seed = seed_d / k_seed
    if not (PITCH_MM - PITCH_TOL_MM <= pitch_seed <= PITCH_MM + PITCH_TOL_MM):
        return None
    axis = (p1 - p0) / seed_d
    best = None
    for dp in (-0.2, -0.1, 0.0, 0.1, 0.2):
        pitch_try = pitch_seed + dp
        if not (PITCH_MM - PITCH_TOL_MM <= pitch_try <= PITCH_MM + PITCH_TOL_MM):
            continue
        r = _walk_with_pitch(axis, p0, pts, amps, pitch_try,
                             PERP_TOL_MM, AX_TOL_MM, MAX_K_STEPS)
        if r is None:
            continue
        if best is None or r["n_inliers"] > best["n_inliers"]:
            best = r
    if best is None:
        return None
    inliers = list(best["inliers"])
    if len(inliers) < MIN_BLOBS_PER_LINE:
        return None
    inlier_pts = pts[inliers]
    c = inlier_pts.mean(axis=0)
    X = inlier_pts - c
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    axis_ref = _unit(Vt[0])
    proj_ref = X @ axis_ref

    # Contiguity: real electrodes are contact chains. A run that stitched
    # a stray blob onto one end via a huge k-multiple looks like
    # [big_gap, 3.5, 3.5, ...] when sorted. Trim extreme outliers by
    # iteratively dropping endpoints whose adjacent gap exceeds
    # MAX_INLIER_GAP_MM. After trimming, any remaining internal gap
    # greater than MAX_INLIER_GAP_MM is a true cross-shank bridge → reject.
    sort_idx = np.argsort(proj_ref)
    sorted_proj = proj_ref[sort_idx].tolist()
    sorted_inliers = [inliers[i] for i in sort_idx]
    while len(sorted_inliers) > MIN_BLOBS_PER_LINE:
        front_gap = sorted_proj[1] - sorted_proj[0]
        back_gap = sorted_proj[-1] - sorted_proj[-2]
        if max(front_gap, back_gap) <= MAX_INLIER_GAP_MM:
            break
        if front_gap >= back_gap:
            sorted_proj.pop(0)
            sorted_inliers.pop(0)
        else:
            sorted_proj.pop()
            sorted_inliers.pop()
    if len(sorted_inliers) < MIN_BLOBS_PER_LINE:
        return None
    # After trimming, any internal gap > threshold is a true bridge.
    gaps_after = np.diff(sorted_proj)
    if gaps_after.size > 0 and float(gaps_after.max()) > MAX_INLIER_GAP_MM:
        return None
    # Re-fit on the pruned set.
    inliers = sorted_inliers
    inlier_pts = pts[inliers]
    c = inlier_pts.mean(axis=0)
    X = inlier_pts - c
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    axis_ref = _unit(Vt[0])
    proj_ref = X @ axis_ref
    lo, hi = float(proj_ref.min()), float(proj_ref.max())
    span = hi - lo
    if span < MIN_LINE_SPAN_MM or span > MAX_LINE_SPAN_MM:
        return None
    return dict(
        axis=axis_ref, center=c, inlier_idx=inliers,
        span_mm=span, span_lo=lo, span_hi=hi,
        start_ras=c + lo * axis_ref, end_ras=c + hi * axis_ref,
        n_blobs=len(inliers),
        amp_sum=float(np.sum([amps[i] for i in inliers])),
    )


MIN_BLOBS_POST_ARBITRATION = 5  # looser floor after arbitration, which
                                # can legitimately shave 1–2 blobs from a
                                # real electrode sharing boundary contacts


def _refit_line_from_inliers(line, pts_c, amps_c, min_blobs=MIN_BLOBS_PER_LINE):
    """Recompute axis / span / endpoints / amp_sum from the current
    ``inlier_idx`` list. Mutates ``line`` and returns it. Returns None
    if the line has too few inliers or too short a span after refit.
    """
    inliers = list(line["inlier_idx"])
    if len(inliers) < min_blobs:
        return None
    inlier_pts = pts_c[inliers]
    c = inlier_pts.mean(axis=0)
    X = inlier_pts - c
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    axis_ref = _unit(Vt[0])
    proj_ref = X @ axis_ref
    lo = float(proj_ref.min())
    hi = float(proj_ref.max())
    span = hi - lo
    if span < MIN_LINE_SPAN_MM or span > MAX_LINE_SPAN_MM:
        return None
    line["axis"] = axis_ref
    line["center"] = c
    line["span_lo"] = lo
    line["span_hi"] = hi
    line["span_mm"] = span
    line["start_ras"] = c + lo * axis_ref
    line["end_ras"] = c + hi * axis_ref
    line["n_blobs"] = len(inliers)
    line["inlier_idx"] = inliers
    line["amp_sum"] = float(np.sum(amps_c[inliers]))
    return line


def _arbitrate_blob_ownership(stage1_lines, pts_c, amps_c):
    """Resolve inlier contention. For each blob claimed by multiple
    lines, award it to the line whose axis is closest (smallest perp
    distance). Lines re-fit on reduced inlier sets; those below the
    MIN_BLOBS / MIN_SPAN floors are dropped.

    This is the ownership arbitration the user asked for: stage-1's
    pitch walker can harvest the same blob for two crossing electrodes;
    the closer-fitting electrode keeps it.
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
        refit = _refit_line_from_inliers(
            new_line, pts_c, amps_c, min_blobs=MIN_BLOBS_POST_ARBITRATION,
        )
        if refit is not None:
            kept.append(refit)
    return kept


def _second_pass_orphan_walker(existing_lines, pts_c, amps_c):
    """Re-run the pitch walker on blobs not claimed by any surviving
    line. Recovers electrodes whose only first-pass hypothesis was a
    bridging line that got dropped by arbitration (e.g. T22 L_1).
    Returns NEW lines (may be empty); they still need to pass the
    standard amp_sum / dist / bolt-anchor gates downstream.
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
    N = orphan_pts.shape[0]
    dist = np.sqrt(np.sum((orphan_pts[:, None, :] - orphan_pts[None, :, :]) ** 2, axis=2))
    pair_mask = np.zeros_like(dist, dtype=bool)
    for mult in (1, 2, 3):
        lo = mult * PITCH_MM - PITCH_TOL_MM
        hi = mult * PITCH_MM + PITCH_TOL_MM
        pair_mask |= (dist >= lo) & (dist <= hi)
    iu, ju = np.where(np.triu(pair_mask, k=1))

    new_hyps: list[dict[str, Any]] = []
    for pi, pj in zip(iu, ju):
        h = _walk_line(int(pi), int(pj), orphan_pts, orphan_amps)
        if h is None:
            continue
        # Remap orphan-local indices back to original pts_c indices.
        h["inlier_idx"] = [int(orphan_idx[i]) for i in h["inlier_idx"]]
        new_hyps.append(h)
    if not new_hyps:
        return []
    new_hyps.sort(key=lambda h: -h["n_blobs"])
    new_lines = _dedup_stage1_lines(new_hyps)
    new_lines = [l for l in new_lines if l.get("amp_sum", 0.0) >= AMP_SUM_MIN]
    return new_lines


def _extend_deep_end(line, pts_c, amps_c, claimed_blobs,
                      dist_arr=None, ras_to_ijk_mat=None,
                      max_gap_mm=14.0,
                      perp_tol_mm=2.5,
                      max_extra=20,
                      max_outer_iter=4):
    """Walk outward from the current deepest inlier, snapping to unclaimed
    blobs within ``max_gap_mm`` of the previous contact along the axis.
    Refits the axis after each pass and re-runs until no more blobs
    can be added — this lets the axis "snake" along a slightly curved
    or off-line electrode and pick up contacts that would have been
    just outside the original PCA axis's tube.

    The 10 mm gap covers up to ~3 missed 3.5 mm-pitch contacts OR a
    DIXI-BM 9 mm insulation jump. Tighter "4 mm at the brain tip"
    behavior emerges naturally because the deepest contacts on real
    electrodes ARE close-spaced.
    """
    for _outer in range(max_outer_iter):
        # Walk in the shallow -> deep direction. After every refit the
        # PCA endpoints can be in either order; if dist_arr is provided,
        # re-orient first so end_ras is the deep side.
        s_ras = np.asarray(line["start_ras"], dtype=float)
        e_ras = np.asarray(line["end_ras"], dtype=float)
        if dist_arr is not None and ras_to_ijk_mat is not None:
            s_ras, e_ras = _orient_shallow_to_deep(
                s_ras, e_ras, dist_arr, ras_to_ijk_mat,
            )
            line["start_ras"] = s_ras
            line["end_ras"] = e_ras
        d_ras = e_ras - s_ras
        L_line = float(np.linalg.norm(d_ras))
        if L_line < 1e-6:
            break
        axis = d_ras / L_line  # shallow -> deep
        center = np.asarray(line["center"], dtype=float)
        inliers = list(line["inlier_idx"])
        n_pre = len(inliers)
        diffs_all = pts_c - center
        along_all = diffs_all @ axis
        perp_all = np.linalg.norm(
            diffs_all - np.outer(along_all, axis), axis=1,
        )
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
        if len(inliers) == n_pre:
            break  # converged
        line["inlier_idx"] = sorted(inliers)
        refit = _refit_line_from_inliers(line, pts_c, amps_c)
        if refit is None:
            break
        line = refit
    return line


def _dedup_stage1_lines(lines):
    if len(lines) < 2:
        return lines
    keep = [True] * len(lines)
    for i in range(len(lines)):
        if not keep[i]:
            continue
        a = lines[i]
        for j in range(i + 1, len(lines)):
            if not keep[j]:
                continue
            b = lines[j]
            ang = float(np.degrees(np.arccos(
                np.clip(abs(np.dot(a["axis"], b["axis"])), 0, 1))))
            if ang > STAGE1_DEDUP_ANGLE_DEG:
                continue
            dv = b["center"] - a["center"]
            par = dv @ a["axis"]
            perp = dv - par * a["axis"]
            perp_d = float(np.linalg.norm(perp))
            if perp_d > STAGE1_DEDUP_PERP_MM:
                continue
            b_lo = par + b["span_lo"]; b_hi = par + b["span_hi"]
            overlap = max(0.0, min(a["span_hi"], b_hi) - max(a["span_lo"], b_lo))
            shorter = min(a["span_hi"] - a["span_lo"], b["span_hi"] - b["span_lo"])
            if shorter > 1e-6 and overlap / shorter >= STAGE1_DEDUP_OVERLAP_FRAC:
                if a["n_blobs"] >= b["n_blobs"]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break
    return [lines[i] for i in range(len(lines)) if keep[i]]


def run_stage1(log_arr, kji_to_ras_fn, dist_arr, ras_to_ijk_mat):
    """Blob-pitch detector on the LoG σ=1 field.
    Returns (lines, pts_c) where pts_c are the contact-sized blob RAS
    positions used for stage-1 exclusion construction downstream.
    """
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
    pair_mask = np.zeros_like(dist, dtype=bool)
    for mult in (1, 2, 3):
        lo = mult * PITCH_MM - PITCH_TOL_MM
        hi = mult * PITCH_MM + PITCH_TOL_MM
        pair_mask |= (dist >= lo) & (dist <= hi)
    iu, ju = np.where(np.triu(pair_mask, k=1))

    hyps = []
    for pi, pj in zip(iu, ju):
        h = _walk_line(int(pi), int(pj), pts_c, amps_c)
        if h is not None:
            hyps.append(h)
    hyps.sort(key=lambda h: -h["n_blobs"])
    # Dedup near-duplicate walker hypotheses FIRST (same physical
    # electrode from different seed pairs). Amp_sum gate next, so
    # arbitration runs only on strong, distinct lines — prevents an
    # otherwise-strong line from being dropped because arbitration
    # shaved its amplitude below the gate threshold.
    lines = _dedup_stage1_lines(hyps)
    lines = [l for l in lines if l.get("amp_sum", 0.0) >= AMP_SUM_MIN]
    # Ownership arbitration: if two distinct lines share an inlier, the
    # closer-fit one keeps it. Re-fits axes after reducing inliers and
    # drops any line whose remaining count falls below MIN_BLOBS.
    lines = _arbitrate_blob_ownership(lines, pts_c, amps_c)

    # Deep-end walk: extend each surviving line with unclaimed blobs
    # found within 4 mm of the current deep tip. Strongest-first so
    # high-confidence lines claim their blobs before weaker ones.
    claimed: set[int] = set()
    for l in lines:
        for bi in l["inlier_idx"]:
            claimed.add(int(bi))
    lines.sort(key=lambda l: -float(l.get("amp_sum", 0.0)))
    lines = [
        _extend_deep_end(l, pts_c, amps_c, claimed,
                         dist_arr=dist_arr, ras_to_ijk_mat=ras_to_ijk_mat)
        for l in lines
    ]

    # Second-pass walker on orphan blobs (those not claimed by any
    # surviving line after arbitration + deep-end extension). Recovers
    # electrodes whose first-pass hypothesis was a bridging line that
    # arbitration killed.
    second_pass_lines = _second_pass_orphan_walker(lines, pts_c, amps_c)
    if second_pass_lines:
        # Extend deep ends on second-pass lines too.
        for nl in second_pass_lines:
            for bi in nl["inlier_idx"]:
                claimed.add(int(bi))
        second_pass_lines = [
            _extend_deep_end(nl, pts_c, amps_c, claimed,
                             dist_arr=dist_arr, ras_to_ijk_mat=ras_to_ijk_mat)
            for nl in second_pass_lines
        ]
        lines = lines + second_pass_lines
        # Final dedup catches any near-duplicates between first-pass
        # and second-pass lines.
        lines.sort(key=lambda l: -l["n_blobs"])
        lines = _dedup_stage1_lines(lines)

    # Deep-tip prior: real shanks' deepest inlier sits ≥ DEEP_TIP_MIN_MM
    # head-distance from hull. Skull-/bone-assembled spurious lines have
    # every inlier at head_distance ≤ 0.
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
        if l["dist_max_mm"] < DEEP_TIP_MIN_MM:
            continue
        l["start_ras"], l["end_ras"] = _orient_shallow_to_deep(
            l["start_ras"], l["end_ras"], dist_arr, ras_to_ijk_mat,
        )
        kept.append(l)
    return kept, pts_c


# ---- Stage 1 exclusion zone ------------------------------------------

def compute_exclusion_mask(cloud_mask_shape, lines_ras, ras_to_ijk_mat,
                           radius_mm=FALLBACK_EXCLUSION_MM, step_mm=1.0):
    """Mark voxels within `radius_mm` of any stage-1 line. Stage 2 skips
    these voxels so already-detected shanks aren't re-detected as Frangi
    tubes.
    """
    excl = np.zeros(cloud_mask_shape, dtype=bool)
    K, J, I = cloud_mask_shape
    for line in lines_ras:
        start = np.asarray(line["start_ras"], dtype=float)
        end = np.asarray(line["end_ras"], dtype=float)
        axis = end - start
        L = float(np.linalg.norm(axis))
        if L < 1e-3:
            continue
        axis = axis / L
        n_steps = int(L / step_mm) + 1
        for k_step in range(n_steps):
            p_ras = start + k_step * step_mm * axis
            h = np.array([p_ras[0], p_ras[1], p_ras[2], 1.0])
            ijk = (ras_to_ijk_mat @ h)[:3]
            k, j, i = int(round(ijk[2])), int(round(ijk[1])), int(round(ijk[0]))
            r = int(np.ceil(radius_mm / 0.5))
            k0, k1 = max(0, k - r), min(K, k + r + 1)
            j0, j1 = max(0, j - r), min(J, j + r + 1)
            i0, i1 = max(0, i - r), min(I, i + r + 1)
            excl[k0:k1, j0:j1, i0:i1] = True
    return excl


# ---- Stage 2: Frangi shaft fallback -----------------------------------

def run_stage2(frangi_s1, intracranial_mask, exclusion_mask, spacing_xyz,
               dist_arr, ijk_to_ras_mat, ras_to_ijk_mat):
    """Fast shaft detector: CC + PCA on residual Frangi cloud. A continuous
    dark shaft with CT-invisible contacts (e.g. RSAN on T2) shows up as a
    single elongated CC in the Frangi tube response; PCA gives the axis
    directly. Same span/aspect + hull-endpoint + deep-tip priors as in
    run_stage1.
    """
    import SimpleITK as sitk
    cloud_mask = (frangi_s1 >= STAGE2_FRANGI_THR) & intracranial_mask & ~exclusion_mask
    bin_img = sitk.GetImageFromArray(cloud_mask.astype(np.uint8))
    cc_filt = sitk.ConnectedComponentImageFilter()
    cc_filt.SetFullyConnected(True)
    cc_img = cc_filt.Execute(bin_img)
    cc_arr = sitk.GetArrayFromImage(cc_img)

    sx, sy, sz = float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])
    lines = []
    flat = cc_arr.ravel()
    order = np.argsort(flat)
    flat_sorted = flat[order]
    changes = np.where(np.diff(flat_sorted) != 0)[0] + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [flat_sorted.size]])
    for s, e in zip(starts, ends):
        lab = flat_sorted[s]
        if lab == 0:
            continue
        idxs = order[s:e]
        n_vox = idxs.size
        if n_vox < STAGE2_MIN_VOXELS or n_vox > STAGE2_MAX_VOXELS:
            continue
        kk, jj, ii = np.unravel_index(idxs, cc_arr.shape)
        pts_mm = np.stack([ii * sx, jj * sy, kk * sz], axis=1).astype(np.float64)
        c = pts_mm.mean(axis=0)
        X = pts_mm - c
        cov = X.T @ X / max(1, n_vox - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]
        pc1 = eigvecs[:, 0]
        proj = X @ pc1
        lo, hi = float(proj.min()), float(proj.max())
        span = hi - lo
        if span < STAGE2_MIN_SPAN_MM or span > STAGE2_MAX_SPAN_MM:
            continue
        perp = X - np.outer(proj, pc1)
        perp_rms = float(np.sqrt(np.mean(np.sum(perp * perp, axis=1))))
        if perp_rms > STAGE2_MAX_PERP_RMS_MM:
            continue
        aspect_geom = span / max(perp_rms, 1e-3)
        if aspect_geom < STAGE2_MIN_ASPECT_GEOM:
            continue
        cc_dist_min = float(dist_arr[kk, jj, ii].min())
        cc_dist_max = float(dist_arr[kk, jj, ii].max())
        if cc_dist_min > HULL_ENDPOINT_MAX_MM:
            continue
        if cc_dist_max < DEEP_TIP_MIN_MM:
            continue
        kji_pts = np.stack([kk, jj, ii], axis=1).astype(np.float64)
        kji_c = kji_pts.mean(axis=0)
        X_kji = kji_pts - kji_c
        _, _, Vt = np.linalg.svd(X_kji, full_matrices=False)
        kji_axis = _unit(Vt[0])
        kji_proj = X_kji @ kji_axis
        kji_lo, kji_hi = float(kji_proj.min()), float(kji_proj.max())
        start_kji = kji_c + kji_lo * kji_axis
        end_kji = kji_c + kji_hi * kji_axis
        start_ijk = np.array([start_kji[2], start_kji[1], start_kji[0], 1.0])
        end_ijk = np.array([end_kji[2], end_kji[1], end_kji[0], 1.0])
        start_ras = (ijk_to_ras_mat @ start_ijk)[:3]
        end_ras = (ijk_to_ras_mat @ end_ijk)[:3]
        start_ras, end_ras = _orient_shallow_to_deep(
            start_ras, end_ras, dist_arr, ras_to_ijk_mat,
        )
        lines.append(dict(
            start_ras=start_ras, end_ras=end_ras,
            length_mm=span, n_inliers=int(n_vox),
            dist_min_mm=cc_dist_min, dist_max_mm=cc_dist_max,
        ))
    return lines, cc_arr


def _trajectory_air_fraction(start_ras, end_ras, ct_arr_kji, ras_to_ijk_mat,
                              n_samples=AIR_SAMPLE_COUNT,
                              air_hu_threshold=AIR_HU_THRESHOLD):
    """Sample CT HU at evenly-spaced points along [start_ras, end_ras]
    and return the fraction of samples below air_hu_threshold. Used to
    reject trajectories that traverse air sinuses / mastoid air cells.
    """
    s = np.asarray(start_ras, dtype=float)
    e = np.asarray(end_ras, dtype=float)
    samples = np.linspace(s, e, n_samples)
    K, J, I = ct_arr_kji.shape
    air_n = 0
    valid = 0
    for p in samples:
        h = np.array([float(p[0]), float(p[1]), float(p[2]), 1.0])
        ijk = (ras_to_ijk_mat @ h)[:3]
        i = int(round(float(ijk[0])))
        j = int(round(float(ijk[1])))
        k = int(round(float(ijk[2])))
        if 0 <= k < K and 0 <= j < J and 0 <= i < I:
            valid += 1
            if float(ct_arr_kji[k, j, i]) < air_hu_threshold:
                air_n += 1
    if valid == 0:
        return 0.0
    return air_n / valid


# ---- Bolt detection + anchoring --------------------------------------

def extract_bolt_candidates(log_arr, dist_arr, ijk_to_ras_mat, spacing_xyz,
                             threshold=BOLT_LOG_THRESHOLD,
                             min_voxels=BOLT_MIN_VOXELS,
                             hull_proximity_mm=BOLT_HULL_PROXIMITY_MM,
                             ras_to_ijk_mat=None):
    """Find bolt candidate CCs in the LoG σ=1 cloud.

    Any CC of bright-metal LoG minima that touches or pokes through the
    hull surface is a bolt candidate. No axis/linearity check here —
    the anchor step uses the shank's axis, not the CC's.

    Returns (bolts, bolt_mask). Each bolt dict has: pts_ras, n_vox,
    dist_min_mm, dist_max_mm.
    """
    import SimpleITK as sitk
    cloud = (log_arr <= -abs(threshold)).astype(np.uint8)
    bin_img = sitk.GetImageFromArray(cloud)
    cc_filt = sitk.ConnectedComponentImageFilter()
    cc_filt.SetFullyConnected(True)
    cc_img = cc_filt.Execute(bin_img)
    cc_arr = sitk.GetArrayFromImage(cc_img)

    bolts: list[dict[str, Any]] = []
    flat = cc_arr.ravel()
    order = np.argsort(flat)
    flat_sorted = flat[order]
    changes = np.where(np.diff(flat_sorted) != 0)[0] + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [flat_sorted.size]])
    bolt_label_mask = np.zeros_like(cc_arr, dtype=np.uint8)

    for s, e in zip(starts, ends):
        lab = flat_sorted[s]
        if lab == 0:
            continue
        idxs = order[s:e]
        n_vox = idxs.size
        if n_vox < min_voxels:
            continue
        kk, jj, ii = np.unravel_index(idxs, cc_arr.shape)
        cc_dist_min = float(dist_arr[kk, jj, ii].min())
        cc_dist_max = float(dist_arr[kk, jj, ii].max())
        # Bolt must poke through / sit at the hull surface. Shafts buried
        # entirely inside brain don't qualify (no bolt is inside brain).
        if cc_dist_min > hull_proximity_mm:
            continue

        all_ijk = np.stack([ii, jj, kk], axis=1).astype(np.float64)
        h = np.concatenate([all_ijk, np.ones((n_vox, 1))], axis=1)
        pts_ras = (ijk_to_ras_mat @ h.T).T[:, :3]
        pts_dist = dist_arr[kk, jj, ii].astype(np.float32)
        bolt_label_mask[kk, jj, ii] = 1
        bolts.append(dict(
            pts_ras=pts_ras,
            pts_dist=pts_dist,
            n_vox=int(n_vox),
            dist_min_mm=cc_dist_min,
            dist_max_mm=cc_dist_max,
        ))

    return bolts, bolt_label_mask


def anchor_trajectory_to_bolt(traj_start_ras, traj_end_ras, bolts,
                               tube_radius_mm=BOLT_TUBE_RADIUS_MM,
                               min_tube_voxels=BOLT_MIN_TUBE_VOXELS,
                               max_inward_mm=BOLT_MAX_INWARD_ALONG_MM,
                               search_outward_mm=BOLT_SEARCH_OUTWARD_MM,
                               shallow_hull_prox_mm=BOLT_SHALLOW_HULL_PROX_MM,
                               base_max_dist_mm=BOLT_BASE_MAX_DIST_MM):
    """Project each bolt CC's voxels onto the shank axis. If enough of
    them fall in a tube (perp ≤ tube_radius) within the shallow-side
    search range AND the most-outward tube voxel sits near the hull
    surface, that CC is the bolt. Returns
    (new_start_ras, skull_entry_ras, bolt) or (None, None, None).

    skull_entry_ras = innermost tube voxel (largest ``along``), a good
    proxy for where bone ends and brain begins along the trajectory.
    """
    traj_start = np.asarray(traj_start_ras, dtype=float)
    traj_end = np.asarray(traj_end_ras, dtype=float)
    d = traj_end - traj_start
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return None, None, None
    shank_axis = d / L  # shallow -> deep

    best = None
    best_n = 0
    best_shallow = None
    best_entry = None
    for bolt in bolts:
        diffs = bolt["pts_ras"] - traj_start
        along = diffs @ shank_axis
        perp_vec = diffs - np.outer(along, shank_axis)
        perp = np.linalg.norm(perp_vec, axis=1)
        in_tube = (
            (perp <= tube_radius_mm)
            & (along <= max_inward_mm)
            & (along >= -search_outward_mm)
        )
        n_in = int(in_tube.sum())
        if n_in < min_tube_voxels:
            continue
        tube_idxs = np.where(in_tube)[0]
        along_tube = along[tube_idxs]
        shallow_local = int(tube_idxs[np.argmin(along_tube)])
        # Shallow (outermost tube) voxel MUST reach the hull surface.
        # This stops a giant merged bolt-chain from anchoring deep-brain
        # candidates via its tail voxels.
        pts_dist = bolt.get("pts_dist")
        if pts_dist is not None:
            shallow_dist = float(pts_dist[shallow_local])
        else:
            shallow_dist = 0.0
        if shallow_dist > shallow_hull_prox_mm:
            continue
        # skull_entry = deepest tube voxel STILL in skull/dura band.
        # Without the dist clip, big merged CCs (bolt + contacts) put
        # the marker at the deepest contact instead of the bolt base.
        if pts_dist is not None:
            in_skull_band = np.where(
                (pts_dist[tube_idxs] <= base_max_dist_mm)
            )[0]
            if in_skull_band.size > 0:
                deep_local = int(tube_idxs[
                    in_skull_band[np.argmax(along_tube[in_skull_band])]
                ])
            else:
                deep_local = int(tube_idxs[np.argmax(along_tube)])
        else:
            deep_local = int(tube_idxs[np.argmax(along_tube)])
        if n_in > best_n:
            best = bolt
            best_n = n_in
            # Project chosen voxels onto the SHANK axis so endpoints
            # stay strictly colinear with the trajectory line. Without
            # this, the markers can sit up to tube_radius_mm (3 mm) off
            # the axis since they're picked from voxel centers.
            shallow_along = float(along[shallow_local])
            deep_along = float(along[deep_local])
            best_shallow = traj_start + shallow_along * shank_axis
            best_entry = traj_start + deep_along * shank_axis

    if best is None:
        return None, None, None
    return best_shallow, best_entry, best


# ---- Cross-stage dedup -----------------------------------------------

def _dedup_trajectories(trajectories,
                        angle_deg=CROSS_STAGE_DEDUP_ANGLE_DEG,
                        perp_mm=CROSS_STAGE_DEDUP_PERP_MM):
    """Remove near-collinear duplicates across stages. Keeps the longer
    line of each pair when their axes align and one midpoint projects
    inside the other segment. Ported from ``deep_core_v2._dedup_v2_trajectories``.
    """
    kept: list[dict[str, Any]] = []
    for t in trajectories:
        s = np.asarray(t["start_ras"], dtype=float)
        e = np.asarray(t["end_ras"], dtype=float)
        d = e - s
        length = float(np.linalg.norm(d))
        if length < 1e-6:
            continue
        axis = d / length
        is_dup = False
        for ki, k in enumerate(kept):
            ks = np.asarray(k["start_ras"], dtype=float)
            ke = np.asarray(k["end_ras"], dtype=float)
            kd = ke - ks
            klen = float(np.linalg.norm(kd))
            if klen < 1e-6:
                continue
            kaxis = kd / klen
            cos = abs(float(np.dot(axis, kaxis)))
            ang = float(np.degrees(np.arccos(min(1.0, cos))))
            if ang > angle_deg:
                continue
            mid = 0.5 * (s + e)
            v = mid - ks
            along = float(np.dot(v, kaxis))
            perp = v - along * kaxis
            pd = float(np.linalg.norm(perp))
            # Reject as duplicate only if this midpoint projects within
            # the kept segment (with margin). Prevents bilateral shanks
            # on opposite sides of the head from being merged.
            margin = 0.5 * klen
            if along < -margin or along > klen + margin:
                continue
            if pd <= perp_mm:
                is_dup = True
                if length > klen:
                    kept[ki] = t
                break
        if not is_dup:
            kept.append(t)
    return kept


# ---- Orchestration ----------------------------------------------------

def _kji_to_ras_fn_from_matrix(ijk_to_ras_mat):
    m = np.asarray(ijk_to_ras_mat, dtype=float)

    def _fn(kji):
        if kji.ndim == 1:
            i, j, k = float(kji[2]), float(kji[1]), float(kji[0])
            return (m @ np.array([i, j, k, 1.0]))[:3]
        ijk = np.stack([kji[:, 2], kji[:, 1], kji[:, 0]], axis=1)
        h = np.concatenate([ijk, np.ones((ijk.shape[0], 1))], axis=1)
        return (m @ h.T).T[:, :3]
    return _fn


def run_two_stage_detection(img, ijk_to_ras_mat, ras_to_ijk_mat,
                             return_features=False, progress_logger=None):
    """Run the full two-stage detector on a SITK image.

    Args:
        img: SimpleITK image (raw CT).
        ijk_to_ras_mat: 4x4 numpy matrix.
        ras_to_ijk_mat: 4x4 numpy matrix.
        return_features: if True, return (trajectories, feature_arrays)
            where feature_arrays is a dict with the LoG, Frangi, hull
            head-distance, intracranial and hull arrays (KJI-order).
        progress_logger: optional callable(message: str) invoked at each
            major checkpoint. The Slicer widget passes a callback that
            updates the status panel and runs `app.processEvents()` so
            the UI doesn't appear hung during the ~10–20 s detection.

    Returns:
        list[dict] or (list[dict], dict): trajectories list (always) and
        optionally a feature_arrays dict for debugging / visualization.
    """
    def _log(msg):
        if progress_logger is not None:
            try:
                progress_logger(msg)
            except Exception:
                pass

    ijk_to_ras_mat = np.asarray(ijk_to_ras_mat, dtype=float)
    ras_to_ijk_mat = np.asarray(ras_to_ijk_mat, dtype=float)
    import SimpleITK as sitk
    _log("preprocessing: hull, head-distance, intracranial mask…")
    ct_arr_kji = sitk.GetArrayFromImage(img).astype(np.float32)
    hull, intracranial, dist_arr = build_masks(img)
    _log("preprocessing: LoG σ=1…")
    log1 = log_sigma(img, sigma_mm=LOG_SIGMA_MM)
    _log("preprocessing: Frangi σ=1…")
    frangi_s1 = frangi_single(img, sigma=FRANGI_STAGE1_SIGMA)
    kji_to_ras = _kji_to_ras_fn_from_matrix(ijk_to_ras_mat)

    _log("stage 1: blob-pitch walker (this is the slow step)…")
    stage1_lines, _pts_blobs = run_stage1(
        log1, kji_to_ras, dist_arr, ras_to_ijk_mat,
    )
    _log(f"stage 1: {len(stage1_lines)} candidate lines after walk + arbitrate + extend")
    excl = compute_exclusion_mask(
        frangi_s1.shape, stage1_lines, ras_to_ijk_mat,
    )
    _log("stage 2: Frangi shaft fallback…")
    stage2_lines, _cc_arr = run_stage2(
        frangi_s1, intracranial, excl, img.GetSpacing(),
        dist_arr, ijk_to_ras_mat, ras_to_ijk_mat,
    )
    _log(f"stage 2: {len(stage2_lines)} candidate lines")

    _log("bolt extraction…")
    bolts, bolt_mask = extract_bolt_candidates(
        log1, dist_arr, ijk_to_ras_mat, img.GetSpacing(),
    )
    _log(f"bolt extraction: {len(bolts)} bolt candidates")

    def _assemble(l, source):
        rec = dict(
            start_ras=np.asarray(l["start_ras"], dtype=float),
            end_ras=np.asarray(l["end_ras"], dtype=float),
            shallow_endpoint_name="start",
            deep_endpoint_name="end",
            source=source,
            length_mm=float(l.get("span_mm", l.get("length_mm", 0.0))),
            n_inliers=int(l.get("n_blobs", l.get("n_inliers", 0))),
            dist_min_mm=float(l.get("dist_min_mm", float("nan"))),
            dist_max_mm=float(l.get("dist_max_mm", float("nan"))),
        )
        if source == "stage1":
            rec["amp_sum"] = float(l.get("amp_sum", 0.0))
        return rec

    # Anchor each candidate to a bolt BEFORE dedup. No bolt == not a
    # real electrode. Then apply length and air-sinus filters; both
    # catch stage-2 false positives that look nothing like real shanks.
    def _anchor_or_reject(rec):
        new_start, skull_entry, bolt = anchor_trajectory_to_bolt(
            rec["start_ras"], rec["end_ras"], bolts,
        )
        if new_start is None:
            return None
        rec["start_ras"] = np.asarray(new_start, dtype=float)
        if skull_entry is not None:
            rec["skull_entry_ras"] = np.asarray(skull_entry, dtype=float)
        rec["length_mm"] = float(np.linalg.norm(rec["end_ras"] - rec["start_ras"]))
        # Length sanity: real SEEG total length (bolt + shank) is bounded.
        if (rec["length_mm"] < MIN_POST_ANCHOR_LEN_MM
                or rec["length_mm"] > MAX_POST_ANCHOR_LEN_MM):
            return None
        # Air-sinus rejection along the intracranial portion.
        intracranial_start = rec.get("skull_entry_ras", rec["start_ras"])
        air_frac = _trajectory_air_fraction(
            intracranial_start, rec["end_ras"], ct_arr_kji, ras_to_ijk_mat,
        )
        if air_frac > AIR_FRAC_MAX:
            return None
        rec["air_fraction"] = float(air_frac)
        rec["bolt_n_vox"] = int(bolt["n_vox"])
        rec["bolt_dist_min_mm"] = float(bolt["dist_min_mm"])
        return rec

    _log("anchoring + length/air filters…")
    anchored: list[dict[str, Any]] = []
    for l in stage1_lines:
        rec = _anchor_or_reject(_assemble(l, "stage1"))
        if rec is not None:
            anchored.append(rec)
    for l in stage2_lines:
        rec = _anchor_or_reject(_assemble(l, "stage2"))
        if rec is not None:
            anchored.append(rec)
    _log(f"anchoring: {len(anchored)} survived")

    # Sort: prefer stage1 over stage2, then longer length. Stage1 is
    # pitch-confirmed; stage2 is a geometric fallback. Dedup keeps the
    # first survivor of each cluster.
    def _sort_key(rec):
        return (0 if rec["source"] == "stage1" else 1,
                -float(rec.get("length_mm", 0.0)))
    anchored.sort(key=_sort_key)
    anchored = _dedup_trajectories(anchored)
    _log(f"final dedup: {len(anchored)} trajectories")

    # Convert to JSON-safe dicts (tuples of floats).
    trajectories: list[dict[str, Any]] = []
    for rec in anchored:
        out = dict(rec)
        out["start_ras"] = [float(x) for x in rec["start_ras"]]
        out["end_ras"] = [float(x) for x in rec["end_ras"]]
        if "skull_entry_ras" in rec:
            out["skull_entry_ras"] = [float(x) for x in rec["skull_entry_ras"]]
        trajectories.append(out)

    if return_features:
        features = {
            "log_sigma1": log1,
            "frangi_sigma1": frangi_s1,
            "head_distance": dist_arr,
            "intracranial": intracranial.astype(np.uint8),
            "hull": hull.astype(np.uint8),
            "bolt_mask": bolt_mask,
        }
        return trajectories, features
    return trajectories
