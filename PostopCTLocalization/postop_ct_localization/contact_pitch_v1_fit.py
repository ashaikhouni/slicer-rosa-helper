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

# CT HU ceiling for scanner-invariance. Metal (titanium bolts, Pt/Ir
# contacts) saturates in the 2000-3000 HU range on every standard-
# reconstruction CT we have (22/23 Tx subjects + AMC099 + ct88).
# Outlier encodings — one subject in our dataset (T1) reaches HU=18966
# because of a different reconstruction pipeline — produce a LoG
# response proportionally larger (LoG is a linear operator) and shift
# the contact-detection distribution to a different effective
# percentile, which would force the fixed LOG_BLOB_THRESHOLD to
# behave inconsistently. Clipping the input once here makes the LoG /
# Frangi response scanner-invariant so the fixed thresholds
# downstream stay universal. 3000 sits safely above real metal
# saturation on all observed scans while clamping only the encoding
# outliers.
HU_CLIP_MAX = 3000.0

PITCH_MM = 3.5
PITCH_TOL_MM = 0.5
PERP_TOL_MM = 1.5
AX_TOL_MM = 0.7
MAX_K_STEPS = 20
MIN_BLOBS_PER_LINE = 5          # 5-contact short superficial depth
                                 # electrodes (ASF, short frontal/parietal
                                 # families) have exactly 5 visible
                                 # contacts — a floor of 6 silently drops
                                 # them. See T21 L_8/L_9/L_13.
MIN_LINE_SPAN_MM = 12.0          # 5 × 3.5 mm pitch ≈ 14 mm nominal; leave
                                 # slack for sub-pitch drift.
MAX_LINE_SPAN_MM = 90.0
AMP_SUM_MIN = 5000.0             # 5 contacts × ~1000 amp each.
# Maximum allowed gap between consecutive inlier contacts along the
# axis. The walker trims single stray outliers whose gap to the rest
# exceeds this, then rejects any line whose remaining internal gap still
# exceeds it. 22 mm allows up to ~6 consecutive missed contacts at the
# 3.5 mm Dixi pitch, and 1 insulation jump on BM (9 mm) or CM (13 mm)
# families. Higher values let the walker bridge contacts from two
# separate electrodes into a single cross-midline line (seen on T2 X07
# where a 6-blob line joined RAMC and LAMC), so we cap here.
MAX_INLIER_GAP_MM = 22.0

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
DEEP_TIP_MIN_MM = 30.0          # strict floor for long lines (where
                                # sinus / skull-base tube FPs hide).
DEEP_TIP_MIN_SHORT_MM = 15.0    # short-line relaxation: superficial
                                # top-of-skull depths (T21 L_8/L_9/L_13)
                                # only reach ~15-20 mm intracranial.
DEEP_TIP_SHORT_MAX_AVG_PITCH_MM = 7.0
                                # Deep-tip prior discriminator: walker
                                # lines whose pre-extend inter-contact
                                # gap averages ≤ this get the relaxed
                                # DEEP_TIP_MIN_SHORT_MM=15 floor; any
                                # wider avg pitch means "not a real
                                # SEEG chain" → strict DEEP_TIP_MIN_MM
                                # floor. 7 mm covers Dixi (3.5),
                                # PMT 16B/C (3.97 / 4.43), and
                                # over-extension slack up to 2× nominal
                                # pitch. Cross-shank bridges + sinus
                                # FPs land well above 7 mm avg.
# Minimum mean intracranial depth of walker inliers. Real SEEG
# contacts sit 5-50 mm inside the hull surface; ghost "contacts"
# produced by bone / skull artifacts cluster at hull_dist 0-3 mm.
# Trajectories whose inliers are on average in bone (this threshold
# fails) are rejected post-anchor — catches cases like T1 X12 where
# a 7-inlier line was assembled out of bone-bright spots along an
# axis diverging from a real bolt.
MIN_INLIER_DIST_MEAN_MM = 5.0
# Air-sinus rejection: along the intracranial portion of every trajectory
# (skull_entry → deep tip), sample CT HU at AIR_SAMPLE_COUNT points;
# if more than AIR_FRAC_MAX of those samples are below AIR_HU_THRESHOLD
# (typical air ≈ −1000 HU), reject. Real electrodes traverse brain
# parenchyma (~30 HU) with metal spikes; sinus tubes are hollow.
AIR_HU_THRESHOLD = -300.0
# Bone rejection: same sampling, but counts points with HU above
# ``BONE_HU_THRESHOLD``. Brain parenchyma sits at roughly 20-50 HU;
# skull bone is 500-1500 HU. A real electrode passes metal-contact
# spikes (> 1500 HU) embedded in brain; a spurious "trajectory"
# threaded through bone sits at 300-1500 HU throughout. Threshold
# conservatively above parenchyma and below metal so ONLY the
# in-bone-not-metal case trips the gate.
BONE_HU_THRESHOLD = 300.0
BONE_HU_METAL_CEILING = 1500.0
BONE_FRAC_MAX = 0.75            # Empirically: real SEEG across T1 /
                                # T2 / T22 / T21 has in-path bone
                                # fraction ≤ 0.64 (T21 worst case).
                                # T1 X12 FP (the cross-bone ghost
                                # line) had 0.84. A 0.75 ceiling
                                # cleanly separates them. Moderate-
                                # bone FPs that sneak past this are
                                # caught downstream by the joint
                                # ``(bone ≥ 0.5) AND (n_in < 9) AND
                                # (amp_per < 1200)`` rule.
DEEP_TIP_NODATA_CHECK_LEN_MM = 20.0
DEEP_TIP_NODATA_HU_THRESHOLD = -1000.0
DEEP_TIP_NODATA_FRAC_MAX = 0.30
                                # Volume-boundary FP filter: sample
                                # HU along the last
                                # DEEP_TIP_NODATA_CHECK_LEN_MM of
                                # trajectory. If >
                                # DEEP_TIP_NODATA_FRAC_MAX of samples
                                # read as no-data (HU ≤ -1000, which
                                # is the NIfTI background value for
                                # voxels outside the oblique CT
                                # acquisition region), the deep end
                                # extends past the scanned volume.
                                # Real trajectories end inside
                                # brain tissue (HU 0-80); FP lines
                                # that project past the FOV edge
                                # (subject-137 X15) have their last
                                # cm+ in the HU=-1024 background.

BONE_SKIM_MIN_BONE_FRAC = 0.65  # Joint "cross-bone skim" FP filter.
BONE_SKIM_MAX_N_INLIERS = 6     # Real SEEG shanks with bone ≥ 0.65
BONE_SKIM_MAX_AMP_PER_IN = 1200.0
                                # (T21 worst at 0.64) have many
                                # inliers AND saturate amp; cross-
                                # bone skim FPs stitch a handful of
                                # sparse bone blobs with weak amp.
                                # All three conditions together catch
                                # the FP without touching real
                                # high-bone shanks. Catches subject-
                                # 137 X2 (bone 0.68, 6 inliers, 692
                                # amp/inlier). The more moderate-
                                # bone FP (X3 at bone 0.36) is close
                                # enough to real high-bone shanks
                                # (T22 at 0.36 with 7 inliers) that
                                # we can't separate them with these
                                # metrics alone.
AIR_FRAC_MAX = 0.50  # real shanks crossing ventricles can hit ~35% air
                     # (CSF + small voids). Sinus FPs run >70%.
AIR_SAMPLE_COUNT = 25

# Post-anchor length bounds. Real SEEG = ~25–80 mm shank + ~15–25 mm
# bolt protrusion. Catches stage-2 venous-sinus / vessel false positives
# (>130 mm) and short skull-base hardware FPs (<45 mm).
MIN_POST_ANCHOR_LEN_MM = 30.0   # short superficial depths (~20 mm
                                 # intracranial + ~10–15 mm bolt). PMT
                                 # bolts run ~12 mm so AMC099 L_5 lands
                                 # at 34 mm total; a 35 mm floor
                                 # rejected it by 0.5 mm. Real 5-contact
                                 # shanks have ≥ 15 mm intracranial, and
                                 # the walker's 5-blob floor + avg-pitch
                                 # gate block fragment / hardware FPs
                                 # that might slip under a 30 mm cap.
MAX_POST_ANCHOR_LEN_MM = 140.0  # Was 130. Bumped to recover subject-137
                                # L_3 (thin-wire PMT: 84 mm shank + wire
                                # gap + bolt → anchored length 132 mm).
                                # Still catches venous sinus / vessel
                                # FPs (typically > 160 mm).

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
BOLT_HU_RESCUE_THRESHOLD = 2000.0   # HU threshold for the per-line
                                    # bolt rescue pool. Titanium bolts
                                    # saturate well above 2000 HU while
                                    # cortical bone peaks around 1500;
                                    # the gap cleanly separates the two.
                                    # Used only on strong stage-1 lines
                                    # whose primary LoG-based anchor
                                    # failed (T4 / ct88 pattern where
                                    # the bolt's LoG signature falls
                                    # below BOLT_LOG_THRESHOLD = 800).
                                    # Tube-filtered per line before
                                    # being offered to the anchor, so
                                    # the secondary pool never leaks
                                    # into global detection.
BOLT_HU_RESCUE_HULL_PROX_MM = 5.0   # Loosened hull-proximity gate for
                                    # the HU rescue pool. Thin-wire PMT
                                    # designs place the bolt-like dense
                                    # structure 2-5 mm inside the hull
                                    # rather than poking through; the
                                    # 2 mm primary gate would reject
                                    # them. Per-line tube filtering
                                    # still keeps far-away CCs out.
BOLT_HU_RESCUE_TUBE_RADIUS_MM = 5.0 # Wider tube radius for the HU
                                    # rescue anchor. The walker's axis
                                    # can drift 1-3° from the true
                                    # shank axis, which over a 40-60 mm
                                    # outward reach (thin-wire PMT)
                                    # adds 1-3 mm lateral offset at
                                    # the bolt. The primary 3 mm tube
                                    # would miss bolts that are really
                                    # on-axis but appear 3-5 mm off
                                    # due to walker-axis tilt.
AXIS_SKULL_SYNTH_STEP_MM = 0.5      # Third-tier rescue: when neither
AXIS_SKULL_SYNTH_MAX_OUTWARD_MM = 80.0
AXIS_SKULL_SYNTH_BOLT_PROTRUDE_MM = 15.0
                                    # the LoG nor HU bolt pool finds
                                    # an anchor AND the stage-1 line
                                    # has strong-SEEG-chain evidence,
                                    # walk outward along the walker
                                    # axis until it crosses the hull
                                    # boundary. Use the crossing as a
                                    # synthetic skull_entry_ras and
                                    # place the synthetic bolt_tip
                                    # BOLT_PROTRUDE_MM further out.
                                    # Recovers T4-class subjects
                                    # whose bolts sit outside the CT
                                    # acquisition window.
BOLT_RESCUE_MIN_N_INLIERS = 10      # Min walker inliers on the stage-1 line (HU rescue)
BOLT_RESCUE_MIN_ORIG_SPAN_MM = 25.0 # Min pre-extend contact span (mm)
BOLT_RESCUE_MIN_DIST_MAX_MM = 30.0  # Min inlier depth (mm). Real shanks
                                    # penetrate at least 30 mm into the
                                    # brain; shallower "lines" are
                                    # typically bolt/skull artifact
                                    # chains that happen to be long
                                    # enough to trigger the rescue
                                    # candidate check. Tighter than
                                    # BOLT_RESCUE_MIN_N_INLIERS alone
                                    # because n=10 shallow chains can
                                    # form on the skull surface.

BOLT_LOG_THRESHOLD = 800.0          # |LoG| magnitude gate for bolt CCs.
                                    # Higher than LOG_BLOB_THRESHOLD
                                    # (300, for contacts) because
                                    # titanium bolts are much denser
                                    # than platinum/iridium contacts
                                    # — real bolts hit |LoG| > 1000
                                    # easily, scalp / skin metal and
                                    # partial-volume artifacts sit
                                    # between 300-700. A strict
                                    # bolt threshold naturally splits
                                    # what would otherwise be a
                                    # whole-head mega-CC into
                                    # discrete per-shank bolts (T1
                                    # went from one 191 k-voxel CC
                                    # to 20+ per-shank CCs).
BOLT_MIN_VOXELS = 20                # drop tiny (isolated-contact) CCs
BOLT_HULL_PROXIMITY_MM = 2.0        # CC must touch / poke through hull
BOLT_TUBE_RADIUS_MM = 3.0            # shank-axis tube for bolt-voxel count
BOLT_MIN_TUBE_VOXELS = 15            # min CC voxels in tube to accept bolt
BOLT_MAX_INWARD_ALONG_MM = 30.0     # bolt may extend up to this far past
                                    # the shallowest contact (CCs often
                                    # include bolt + first few contacts)
BOLT_SEARCH_OUTWARD_MM = 60.0       # max outward distance (mm) from
                                    # the walker's shallow contact
                                    # within which a bolt CC can be
                                    # accepted. Real SEEG bolts
                                    # protrude ~15-40 mm past the
                                    # skull; combined with ~15 mm
                                    # between shallowest contact and
                                    # skull, a 60 mm reach covers
                                    # every realistic case. Was 120
                                    # mm — that let the anchor grab
                                    # a far-away cross-shank bolt
                                    # and assemble a cross-brain
                                    # "trajectory" out of a short
                                    # walker fragment plus a
                                    # distant bolt (T1 stage-1 #19
                                    # became a 120 mm FP this way).
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


# ---- Pitch strategy + auto-detection ---------------------------------

# Candidate electrode-pitch set per UI strategy. The walker runs once
# per pitch in this set; hypotheses across pitches are unioned before
# dedup/arbitration so multi-family cases (e.g. Dixi + PMT on the
# same scan) get both families detected without the user picking a
# single pitch.
#
# For the "auto" strategy, pitches are estimated at runtime from the
# intracranial blob cloud's mutual-NN distance distribution (see
# ``detect_pitch_from_intracranial_blobs``). On a clean Dixi case the
# auto detector returns ≈ 3.3 mm; the surrounding ±0.5 mm tolerance in
# the walker absorbs the sub-bin localization bias.
PITCH_STRATEGY_PITCHES_MM = {
    "dixi":    (3.5,),
    "pmt_35":  (3.5,),           # PMT 2102-XX-091 family — same pitch as Dixi
    "pmt":     (3.5, 3.97, 4.43),
    "mixed":   (3.5, 3.97, 4.43),
}
PITCH_STRATEGY_VENDORS = {
    "dixi":    ("Dixi",),
    "pmt_35":  ("PMT",),
    "pmt":     ("PMT",),
    "mixed":   ("Dixi", "PMT"),
    "auto":    ("Dixi", "PMT", "AdTech"),
}

PITCH_AUTO_MIN_MM = 2.5
PITCH_AUTO_MAX_MM = 6.0

# Mutual-NN centroid sits ~0.2 mm low of the true pitch (partial-volume
# localization bias). When the auto detector lands within this tolerance
# of a known library pitch, snap to it so the walker sees the nominal
# value instead of the biased estimate.
LIBRARY_PITCHES_MM = (3.5, 3.97, 4.43)
PITCH_SNAP_MM = 0.3


def detect_pitch_from_intracranial_blobs(pts_c, dist_arr, ras_to_ijk_mat,
                                           min_mm=PITCH_AUTO_MIN_MM,
                                           max_mm=PITCH_AUTO_MAX_MM):
    """Estimate electrode pitch from the mutual-nearest-neighbour
    distances of the intracranial blob cloud.

    Mutual-NN pairs (A's 1-NN is B AND B's 1-NN is A) are dominated by
    same-shank adjacent contacts, so their distribution peaks at the
    true electrode pitch. Non-mutual neighbours (cross-shank, bolt→
    contact, noise) don't show up in this distribution. Empirical
    centroid tends to sit ~0.2 mm low of the nominal pitch (partial-
    volume localization bias) — within the walker's ±0.5 mm tolerance.

    Returns a list with the detected pitch, or an empty list when the
    blob cloud is too sparse or the histogram is flat.
    """
    if pts_c is None or len(pts_c) < 10:
        return []
    pts_arr = np.asarray(pts_c, dtype=float)
    hd = np.array([
        _sample_dist_at_ras(dist_arr, ras_to_ijk_mat, p) for p in pts_arr
    ])
    pts_ic = pts_arr[hd >= INTRACRANIAL_MIN_DISTANCE_MM]
    if pts_ic.shape[0] < 10:
        return []
    D = np.sqrt(np.sum((pts_ic[:, None, :] - pts_ic[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(D, np.inf)
    nn_idx = np.argmin(D, axis=1)
    mutual_dists = []
    seen = set()
    for i in range(pts_ic.shape[0]):
        j = int(nn_idx[i])
        if int(nn_idx[j]) == i:
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            mutual_dists.append(float(D[i, j]))
    if len(mutual_dists) < 5:
        return []
    m = np.asarray(mutual_dists)
    m = m[(m >= min_mm) & (m <= max_mm)]
    if m.size < 5:
        return []
    bins = np.arange(min_mm, max_mm + 0.25, 0.25)
    hist, edges = np.histogram(m, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if hist.max() <= 0:
        return []
    peak_idx = int(np.argmax(hist))
    peak_mm = float(centers[peak_idx])
    window = np.abs(centers - peak_mm) <= 0.6
    denom = float(np.sum(hist[window]))
    if denom <= 0:
        return []
    centroid = float(np.sum(centers[window] * hist[window]) / denom)
    return [round(centroid, 2)]


def _snap_to_library_pitch(pitch_mm, library=LIBRARY_PITCHES_MM,
                             tol_mm=PITCH_SNAP_MM):
    """If ``pitch_mm`` is within ``tol_mm`` of a known library pitch,
    return the nominal library value; otherwise return ``pitch_mm``
    unchanged. Removes the ~0.2 mm low-bias from the mutual-NN centroid.
    """
    pitch_mm = float(pitch_mm)
    best = None
    for lib in library:
        d = abs(pitch_mm - float(lib))
        if d <= tol_mm and (best is None or d < best[0]):
            best = (d, float(lib))
    return best[1] if best is not None else pitch_mm


def resolve_pitches_for_strategy(strategy, pts_c=None,
                                    dist_arr=None, ras_to_ijk_mat=None):
    """Map a UI ``strategy`` string to a concrete tuple of walker
    pitches. Falls back to the Dixi default when auto-detection fails
    or the strategy is unrecognised.
    """
    key = str(strategy or "dixi").lower()
    if key in PITCH_STRATEGY_PITCHES_MM:
        return PITCH_STRATEGY_PITCHES_MM[key]
    if key == "auto":
        if pts_c is None or dist_arr is None or ras_to_ijk_mat is None:
            return (PITCH_MM,)
        detected = detect_pitch_from_intracranial_blobs(
            pts_c, dist_arr, ras_to_ijk_mat,
        )
        if detected:
            snapped = tuple(_snap_to_library_pitch(p) for p in detected)
            return snapped
        return (PITCH_MM,)
    return (PITCH_MM,)


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


# Sub-voxel centroid refinement on LoG blob minima. Without it, blob
# positions snap to voxel-integer grid and consecutive-contact distances
# collapse onto √(integer) values (√14 = 3.74, √38 = 6.16 …). For
# specific axis orientations those values fall outside the walker's
# seed-pair pitch band ([3.0, 4.0] ∪ [6.5, 7.5] ∪ [10.0, 11.0]) and the
# shank is lost. Refined positions land within ~0.1 mm of true centres
# and collapse observed distances back toward 3.5 / 7.0 / 10.5 mm. Probes
# can flip this off to compare.
LOG_BLOB_SUBVOXEL_DEFAULT = True


def extract_blobs(log_arr, threshold=LOG_BLOB_THRESHOLD, sub_voxel=None):
    """Regional-minima blob extraction. Each contact (local LoG minimum)
    becomes one blob. Uses SITK grayscale erode to find local minima in a
    ~1 mm radius, then thresholds by absolute LoG value.

    ``sub_voxel``: when True, refine each minimum's position to sub-voxel
    accuracy via a 1-D quadratic fit along each axis in the 3×3×3
    neighbourhood. This counteracts voxel-grid aliasing in blob-pair
    distances — without it, consecutive-contact distances snap to
    sqrt-integer values (√14 = 3.74 for well-aligned axes, √24 = 4.90
    for others), putting some shanks outside the walker's seed-pair
    pitch band despite being standard Dixi 3.5 mm pitch.
    """
    import SimpleITK as sitk
    if sub_voxel is None:
        sub_voxel = LOG_BLOB_SUBVOXEL_DEFAULT
    erode = sitk.GrayscaleErode(
        sitk.GetImageFromArray(log_arr),
        kernelRadius=[2, 2, 2],
    )
    eroded = sitk.GetArrayFromImage(erode).astype(np.float32)
    is_local_min = (log_arr <= eroded + 1e-4)
    strong = is_local_min & (log_arr <= -abs(threshold))
    kk, jj, ii = np.where(strong)
    blobs = []
    K, J, I = log_arr.shape
    for k, j, i in zip(kk, jj, ii):
        val = float(log_arr[k, j, i])
        if sub_voxel and 0 < k < K - 1 and 0 < j < J - 1 and 0 < i < I - 1:
            # Quadratic vertex along each axis: offset = 0.5·(f⁻ − f⁺) / (f⁻ − 2·f⁰ + f⁺)
            # Clip to [-0.5, 0.5] so the refined position stays inside the voxel.
            fi_m = float(log_arr[k, j, i - 1]); fi_p = float(log_arr[k, j, i + 1])
            fj_m = float(log_arr[k, j - 1, i]); fj_p = float(log_arr[k, j + 1, i])
            fk_m = float(log_arr[k - 1, j, i]); fk_p = float(log_arr[k + 1, j, i])
            def _vtx(fm, f0, fp):
                denom = fm - 2.0 * f0 + fp
                if abs(denom) < 1e-6:
                    return 0.0
                off = 0.5 * (fm - fp) / denom
                return max(-0.5, min(0.5, off))
            di = _vtx(fi_m, val, fi_p)
            dj = _vtx(fj_m, val, fj_p)
            dk = _vtx(fk_m, val, fk_p)
            blobs.append(dict(
                kji=np.array([float(k) + dk, float(j) + dj, float(i) + di]),
                amp=-val, n_vox=1,
            ))
        else:
            blobs.append(dict(
                kji=np.array([float(k), float(j), float(i)]),
                amp=-val, n_vox=1,
            ))
    return blobs


def _walk_with_pitch_precomputed(proj, within_perp, amps, pitch, ax_tol, max_k):
    """Pitch-matching step given pre-computed per-blob axis projection and
    perp-tolerance mask (``_walk_line`` computes these once per seed pair
    and reuses across the 5 pitch perturbations — the axis and perp mask
    don't change, only the per-k targets do).

    Vectorized: each blob's natural slot is ``k = round(proj / pitch)``.
    A blob is accepted when its perp tolerance holds, ``|k| ≤ max_k`` and
    ``|proj − k·pitch| ≤ ax_tol``. For each surviving k slot, keep the
    single blob with the highest amplitude. Replaces the 41-iteration
    Python loop over k that dominated the walker after the per-pair
    factoring (2·MAX_K_STEPS + 1 iterations × 54k calls on T2).
    """
    k_nearest = np.rint(proj / pitch).astype(np.int64)
    target = k_nearest * pitch
    ax_resid = np.abs(proj - target)
    valid = within_perp & (ax_resid <= ax_tol) & (np.abs(k_nearest) <= max_k)
    if not np.any(valid):
        return None
    idx_valid = np.where(valid)[0]
    k_valid = k_nearest[idx_valid]
    amps_valid = amps[idx_valid]
    order = np.argsort(k_valid, kind="stable")
    sorted_k = k_valid[order]
    sorted_idx = idx_valid[order]
    sorted_amps = amps_valid[order]
    # Group breakpoints: indices where k changes (plus 0 and len as ends).
    change = np.where(np.diff(sorted_k) != 0)[0] + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [sorted_k.size]])
    inliers = set()
    for s, e in zip(starts, ends):
        local_best = int(np.argmax(sorted_amps[s:e]))
        inliers.add(int(sorted_idx[s + local_best]))
    return dict(inliers=inliers, pitch=pitch, n_inliers=len(inliers))


def _walk_line(seed_idx, neighbor_idx, pts, amps, pitch_mm=PITCH_MM):
    p0 = pts[seed_idx]
    p1 = pts[neighbor_idx]
    seed_d = float(np.linalg.norm(p1 - p0))
    k_seed = max(1, int(round(seed_d / pitch_mm)))
    pitch_seed = seed_d / k_seed
    if not (pitch_mm - PITCH_TOL_MM <= pitch_seed <= pitch_mm + PITCH_TOL_MM):
        return None
    axis = (p1 - p0) / seed_d
    # Per-pair precompute: the axis is fixed across pitch perturbations,
    # so projection + perp-tolerance only need to be computed once.
    # Perp-distance test uses perp² = |diffs|² − proj² to avoid the
    # np.outer(proj, axis) allocation that dominated the walker in
    # profiling (~11 s of 17 s wall on T2).
    diffs = pts - p0
    proj = diffs @ axis
    d2 = np.einsum("ij,ij->i", diffs, diffs)
    perp_sq = d2 - proj * proj
    within_perp = perp_sq <= (PERP_TOL_MM * PERP_TOL_MM)
    best = None
    for dp in (-0.2, -0.1, 0.0, 0.1, 0.2):
        pitch_try = pitch_seed + dp
        if not (pitch_mm - PITCH_TOL_MM <= pitch_try <= pitch_mm + PITCH_TOL_MM):
            continue
        r = _walk_with_pitch_precomputed(
            proj, within_perp, amps, pitch_try, AX_TOL_MM, MAX_K_STEPS,
        )
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


def _median_inlier_pitch(pts, axis):
    """Median consecutive spacing of ``pts`` projected onto ``axis``.

    Robust to a single far-away outlier that would skew mean-based
    avg_pitch (original_span_mm / (n-1)). When the walker absorbs a
    spurious blob at one end, the mean pitch inflates past the
    ``looks_like_seeg`` threshold even though most of the inliers sit
    on a regular 3.5 mm chain; median collapses back to the true
    pitch.
    """
    pts = np.asarray(pts, dtype=float)
    axis = np.asarray(axis, dtype=float)
    if pts.shape[0] < 2:
        return float("inf")
    c = pts.mean(axis=0)
    proj = np.sort((pts - c) @ axis)
    diffs = np.diff(proj)
    if diffs.size == 0:
        return float("inf")
    return float(np.median(diffs))


MIN_BLOBS_POST_ARBITRATION = 4  # looser floor after arbitration, which
                                # can legitimately shave 1–2 blobs from a
                                # real electrode sharing boundary contacts.
                                # One below MIN_BLOBS_PER_LINE so a
                                # 5-contact shank can survive a single
                                # arbitrated contact without being killed.


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


def _second_pass_orphan_walker(existing_lines, pts_c, amps_c,
                                 pitches_mm=(PITCH_MM,)):
    """Re-run the pitch walker on blobs not claimed by any surviving
    line. Recovers electrodes whose only first-pass hypothesis was a
    bridging line that got dropped by arbitration (e.g. T22 L_1).
    Returns NEW lines (may be empty); they still need to pass the
    standard amp_sum / dist / bolt-anchor gates downstream.

    ``pitches_mm`` is the set of candidate electrode pitches to walk.
    Defaults to the legacy single Dixi 3.5 mm pitch.
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
            h = _walk_line(int(pi), int(pj), orphan_pts, orphan_amps,
                            pitch_mm=pitch)
            if h is None:
                continue
            h["inlier_idx"] = [int(orphan_idx[i]) for i in h["inlier_idx"]]
            h["seed_pitch_mm"] = float(pitch)
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
    """Walk outward from the current deepest **and** shallowest inliers,
    snapping to unclaimed blobs within ``max_gap_mm`` of the last contact
    along the axis. Refits the axis after each pass and re-runs until no
    more blobs can be added — this lets the axis "snake" along a slightly
    curved or off-line electrode and pick up contacts that would have
    been just outside the original PCA axis's tube.

    ``max_gap_mm`` of 14 mm covers ~3 missed 3.5 mm-pitch contacts or a
    BM 9 mm insulation jump. Tighter "4 mm at the brain tip" behaviour
    emerges naturally because deep contacts on real electrodes ARE
    close-spaced.

    Walking both directions matters: when the initial walker locks onto
    the middle of an electrode (for instance because arbitration stripped
    its tip contacts or because the first seed pair was mid-shank) a deep-
    only extension will undershoot the deep tip by however much the
    shallow side was already short. Symmetric extension fixes this — one
    pass grabs whichever contacts the initial line missed, the refit
    recenters the axis, and the next outer iteration can chase the other
    side.
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
        # Shallow-side walk. ``shallow_proj`` tracks the shallowest
        # inlier; candidates sit at a smaller ``along``, closer to the
        # bolt. Bolt blobs naturally end up here — that is fine since
        # they will be rejected by the bolt-anchor step downstream if
        # they do not belong on this shank.
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
        refit = _refit_line_from_inliers(line, pts_c, amps_c)
        if refit is None:
            break
        line = refit
    return line


def _dedup_stage1_lines(lines):
    """Remove near-duplicate stage-1 walker hypotheses.

    For every (i, j), i < j, a duplicate requires all three of:
      * axis angle ≤ STAGE1_DEDUP_ANGLE_DEG
      * perpendicular center offset ≤ STAGE1_DEDUP_PERP_MM
      * overlap / shorter ≥ STAGE1_DEDUP_OVERLAP_FRAC (shorter > 1e-6)

    When a pair qualifies: if ``n_blobs[i] >= n_blobs[j]`` drop j,
    else drop i (and i stops comparing). A dropped line cannot trigger
    further drops — this is what stops every subset being collapsed to
    a single survivor. Returns surviving lines in the input order.

    Vectorized: O(N²) pairwise matrices + a Python loop with numpy
    inner ops, replacing the original pure-Python double loop that
    took 5 s/call on ~1500 hypotheses (AMC099).
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

    # par[i, j] = (centers[j] - centers[i]) · axes[i]
    M = axes @ centers.T
    axes_dot_center = np.einsum("ik,ik->i", axes, centers)
    par = M - axes_dot_center[:, None]

    # perp²[i, j] = |centers[j] - centers[i]|² - par[i, j]²
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
        # Original branch: a.n_blobs >= b.n_blobs → kill j; else kill
        # i and break. Under callers' pre-sort by -n_blobs the second
        # branch is unreachable, but handle it for general safety: if
        # any surviving j in ``row`` has strictly more blobs than i,
        # the original would break at the first such j and leave later
        # j's untouched — so i dies and no further kills from i.
        if np.any(row & (n_blobs > n_blobs[i])):
            alive[i] = False
            continue
        alive[row] = False

    return [lines[i] for i in range(n) if alive[i]]


def run_stage1(log_arr, kji_to_ras_fn, dist_arr, ras_to_ijk_mat,
                pitches_mm=None):
    """Blob-pitch detector on the LoG σ=1 field.
    Returns (lines, pts_c) where pts_c are the contact-sized blob RAS
    positions used for stage-1 exclusion construction downstream.

    ``pitches_mm`` is a sequence of candidate electrode pitches. One
    walker pass runs per pitch; hypotheses across pitches are unioned
    before dedup / arbitration. Default is the legacy single
    ``[PITCH_MM]`` (Dixi 3.5 mm).
    """
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

    # Run one seed-pair + walker pass per candidate pitch, then union
    # hypotheses. Near-duplicates (same shank found at two pitches)
    # are killed by the subsequent ``_dedup_stage1_lines`` call.
    hyps = []
    for pitch in pitches_mm:
        pair_mask = np.zeros_like(dist, dtype=bool)
        for mult in (1, 2, 3):
            lo = mult * pitch - PITCH_TOL_MM
            hi = mult * pitch + PITCH_TOL_MM
            pair_mask |= (dist >= lo) & (dist <= hi)
        iu, ju = np.where(np.triu(pair_mask, k=1))
        for pi, pj in zip(iu, ju):
            h = _walk_line(int(pi), int(pj), pts_c, amps_c, pitch_mm=pitch)
            if h is not None:
                h["seed_pitch_mm"] = float(pitch)
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
    # Stash the pre-extend span so the deep-tip prior downstream can
    # tell a genuinely-short walker line from a short line grown by
    # absorbing neighbouring-shank contacts (T21 L_13: span 19→46).
    claimed: set[int] = set()
    for l in lines:
        for bi in l["inlier_idx"]:
            claimed.add(int(bi))
        l.setdefault("original_span_mm", float(l.get("span_mm", 0.0)))
        if "original_median_pitch_mm" not in l:
            l["original_median_pitch_mm"] = _median_inlier_pitch(
                pts_c[l["inlier_idx"]], l["axis"],
            )
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
    second_pass_lines = _second_pass_orphan_walker(
        lines, pts_c, amps_c, pitches_mm=pitches_mm,
    )
    if second_pass_lines:
        # Extend deep ends on second-pass lines too.
        for nl in second_pass_lines:
            for bi in nl["inlier_idx"]:
                claimed.add(int(bi))
            nl.setdefault("original_span_mm", float(nl.get("span_mm", 0.0)))
            if "original_median_pitch_mm" not in nl:
                nl["original_median_pitch_mm"] = _median_inlier_pitch(
                    pts_c[nl["inlier_idx"]], nl["axis"],
                )
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
    # head-distance from hull. Skull-/bone-assembled spurious lines
    # have every inlier at head_distance ≤ 0. Lines that look like a
    # real SEEG chain (avg pre-extend pitch ≤ DEEP_TIP_SHORT_MAX_AVG_PITCH_MM)
    # use the relaxed DEEP_TIP_MIN_SHORT_MM floor so superficial
    # top-of-skull depths (T21 L_8 / L_9 / L_13) and laterally-placed
    # full-length shanks with shallow deep tips (AMC099 L_10, L_31) are
    # not dropped; longer lines keep the strict 30 mm floor to
    # continue killing sinus / skull-base tube FPs.
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
        # "Looks like a real SEEG chain" → relaxed deep-tip floor. The
        # discriminator is the pre-extend MEDIAN pitch: real electrodes
        # sit in a narrow 3–7 mm pitch band; cross-shank bridges and
        # sinus / vessel FPs land outside it. Median (not mean) so one
        # walker-absorbed outlier blob doesn't re-classify a genuine
        # chain as cross-shank (T14 RMMF: 6 inliers, mean pitch 8.5
        # from one far blob, median pitch 3.7 — the real chain).
        # ``original_median_pitch_mm`` is the pre-extend value so
        # ``_extend_deep_end`` absorption can't skew the statistic.
        median_pitch = float(l.get(
            "original_median_pitch_mm",
            (float(l.get("original_span_mm", l.get("span_mm", 0.0)))
             / max(1, int(l.get("n_blobs", 2)) - 1)),
        ))
        looks_like_seeg = median_pitch <= DEEP_TIP_SHORT_MAX_AVG_PITCH_MM
        min_dist = DEEP_TIP_MIN_SHORT_MM if looks_like_seeg else DEEP_TIP_MIN_MM
        if l["dist_max_mm"] < min_dist:
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


def _trajectory_hu_fractions(start_ras, end_ras, ct_arr_kji, ras_to_ijk_mat,
                               n_samples=AIR_SAMPLE_COUNT,
                               air_hu_threshold=AIR_HU_THRESHOLD,
                               bone_hu_threshold=BONE_HU_THRESHOLD,
                               bone_hu_metal_ceiling=BONE_HU_METAL_CEILING,
                               end_pad_mm=3.0):
    """Sample CT HU at evenly-spaced points along [start_ras, end_ras]
    and return (air_fraction, bone_fraction). Air = HU <
    ``air_hu_threshold`` (sinuses, mastoid cells). Bone = HU between
    ``bone_hu_threshold`` and ``bone_hu_metal_ceiling`` (skull bone,
    but NOT metal contacts — those sit above the ceiling). A real
    electrode passes through brain parenchyma with metal contact
    spikes; a spurious path through bone has most samples in the
    bone range.

    ``end_pad_mm`` trims the sampling segment inward from both
    endpoints. The first few mm past ``skull_entry_ras`` cross
    dura and CSF with transitional HU; the last few mm near the
    deep tip may straddle the tip bloom. Cropping both ends
    focuses the sample on the clean brain / bone separation
    zone and widens the real-vs-FP gap.
    """
    s = np.asarray(start_ras, dtype=float)
    e = np.asarray(end_ras, dtype=float)
    d = e - s
    L = float(np.linalg.norm(d))
    if L > 2 * end_pad_mm + 1e-6:
        axis = d / L
        s = s + end_pad_mm * axis
        e = e - end_pad_mm * axis
    samples = np.linspace(s, e, n_samples)
    K, J, I = ct_arr_kji.shape
    air_n = 0
    bone_n = 0
    valid = 0
    for p in samples:
        h = np.array([float(p[0]), float(p[1]), float(p[2]), 1.0])
        ijk = (ras_to_ijk_mat @ h)[:3]
        i = int(round(float(ijk[0])))
        j = int(round(float(ijk[1])))
        k = int(round(float(ijk[2])))
        if 0 <= k < K and 0 <= j < J and 0 <= i < I:
            valid += 1
            hu = float(ct_arr_kji[k, j, i])
            if hu < air_hu_threshold:
                air_n += 1
            elif bone_hu_threshold <= hu < bone_hu_metal_ceiling:
                bone_n += 1
    if valid == 0:
        return 0.0, 0.0
    return air_n / valid, bone_n / valid


def _deep_tip_nodata_fraction(start_ras, end_ras, ct_arr_kji, ras_to_ijk_mat,
                                check_len_mm=DEEP_TIP_NODATA_CHECK_LEN_MM,
                                hu_threshold=DEEP_TIP_NODATA_HU_THRESHOLD,
                                n_samples=11):
    """Sample the last ``check_len_mm`` of the trajectory (measuring
    inward from ``end_ras`` toward ``start_ras``) and return the
    fraction of samples with HU ≤ ``hu_threshold`` (no-data voxels
    outside the oblique CT acquisition region, typically -1024 in
    NIfTI background).

    Real trajectories end inside brain tissue (HU ~0-80); FPs whose
    deep tip projects past the CT field of view have most of their
    last cm+ sitting in -1024 background.
    """
    s = np.asarray(start_ras, dtype=float)
    e = np.asarray(end_ras, dtype=float)
    d = e - s
    L = float(np.linalg.norm(d))
    if L < 1.0:
        return 0.0
    axis = d / L
    # Start of the check segment = max(0, L - check_len_mm) mm from s.
    seg_len = float(min(check_len_mm, L))
    seg_start = e - seg_len * axis
    samples = np.linspace(seg_start, e, n_samples)
    K, J, I = ct_arr_kji.shape
    nodata = 0
    valid = 0
    for p in samples:
        h = np.array([float(p[0]), float(p[1]), float(p[2]), 1.0])
        ijk = (ras_to_ijk_mat @ h)[:3]
        i = int(round(float(ijk[0])))
        j = int(round(float(ijk[1])))
        k = int(round(float(ijk[2])))
        if 0 <= k < K and 0 <= j < J and 0 <= i < I:
            valid += 1
            if float(ct_arr_kji[k, j, i]) <= hu_threshold:
                nodata += 1
        else:
            # Sample is literally outside the image bounds → counts
            # as no-data for our purposes.
            valid += 1
            nodata += 1
    if valid == 0:
        return 0.0
    return nodata / valid


def _trajectory_air_fraction(start_ras, end_ras, ct_arr_kji, ras_to_ijk_mat,
                              n_samples=AIR_SAMPLE_COUNT,
                              air_hu_threshold=AIR_HU_THRESHOLD):
    """Back-compat wrapper: returns only the air fraction. Prefer
    ``_trajectory_hu_fractions`` in new code.
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
                             ras_to_ijk_mat=None,
                             ct_arr=None,
                             hu_threshold=None):
    """Find bolt candidate CCs in the LoG σ=1 cloud, or (when ``ct_arr``
    + ``hu_threshold`` are supplied) in the raw-HU cloud.

    LoG mode (default): CCs of ``log_arr ≤ -|threshold|``. HU mode:
    CCs of ``ct_arr ≥ hu_threshold``. Either way, every CC must touch
    or poke through the hull surface — bolts sit at the skull; shafts
    buried entirely inside brain don't qualify.

    Returns (bolts, bolt_mask). Each bolt dict has: pts_ras, n_vox,
    dist_min_mm, dist_max_mm.
    """
    import SimpleITK as sitk
    if ct_arr is not None and hu_threshold is not None:
        cloud = (ct_arr >= float(hu_threshold)).astype(np.uint8)
    else:
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
    next_bolt_id = 0

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
            id=next_bolt_id,
            pts_ras=pts_ras,
            pts_dist=pts_dist,
            n_vox=int(n_vox),
            dist_min_mm=cc_dist_min,
            dist_max_mm=cc_dist_max,
        ))
        next_bolt_id += 1

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
    # Stash the in-tube voxel count so callers that try both
    # orientations (shallow→deep ambiguity for trajectories whose
    # endpoints have similar hull distance) can compare strength.
    best = dict(best)
    best["tube_n_vox"] = int(best_n)
    return best_shallow, best_entry, best


# ---- Deep-end axis refinement ----------------------------------------

# Axis-directed refinement thresholds. ``|LoG| >= AXIS_REFINE_MIN_ABS``
# is what we consider "on a contact / bright shaft"; once that signal
# drops below the threshold for ``AXIS_REFINE_MISS_MM`` consecutive mm
# along the axis, the shank has ended. Tuned on T2 RAI where the deep
# 4 contacts merge into one LoG CC and only show up as a 1-D oscillation
# along the axis.
AXIS_REFINE_STEP_MM = 0.5
AXIS_REFINE_MAX_MM = 40.0    # never extend further than this past end_ras
AXIS_REFINE_MIN_ABS = 300.0  # match LOG_BLOB_THRESHOLD (contact signal)
AXIS_REFINE_MISS_MM = 3.0    # 3 mm of LoG above -threshold → stop
DEEP_END_MARGIN_PAST_LAST_CONTACT_MM = 5.0
                             # Hard cap on how far the deep end can
                             # sit past the deepest real contact
                             # (walker inlier). No SEEG electrode has
                             # a large gap past the last contact, so
                             # anything further is walker/extension
                             # over-reach. 5 mm ≈ one contact pitch
                             # gives slack for walker-axis drift.

# Post-refinement crossing-tip retreat. If a deep tip ends up inside
# another trajectory's contact-acceptance tube (perp ≤ this), the
# refinement has walked past the real electrode end into the neighbour's
# contacts. Retreat the tip along its own axis until the perp clearance
# is at least this. 2.0 mm = walker's PERP_TOL_MM (1.5) + 0.5 mm safety
# margin, so a tip outside this is unambiguously not inside any other
# shank's blob-acceptance radius.
CROSSING_TIP_CLEARANCE_MM = 2.0
CROSSING_RETREAT_STEP_MM = 0.5


def _min_perp_to_other_segments(p, segs, skip_idx):
    """Minimum perpendicular distance from point ``p`` to any other
    segment in ``segs`` (skipping the one at ``skip_idx``). Uses
    segment-to-point distance (clamped along-projection), not infinite
    line, so crossing shanks compare only where they actually live.
    """
    best = float("inf")
    for i, seg in enumerate(segs):
        if i == skip_idx:
            continue
        v = p - seg["s"]
        along = float(v @ seg["a"])
        along_c = max(0.0, min(seg["L"], along))
        proj = seg["s"] + along_c * seg["a"]
        d = float(np.linalg.norm(p - proj))
        if d < best:
            best = d
    return best


def _retreat_crossing_tips(anchored,
                             log_arr=None,
                             ras_to_ijk_mat=None,
                             clearance_mm=CROSSING_TIP_CLEARANCE_MM,
                             step_mm=CROSSING_RETREAT_STEP_MM,
                             min_length_mm=MIN_POST_ANCHOR_LEN_MM,
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
    ``min_length_mm``.
    """
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
        clearance = _min_perp_to_other_segments(e, segs, i)
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
            if _min_perp_to_other_segments(candidate, segs, i) < clearance_mm:
                continue  # still inside another shank's tube
            if have_log and _log_at(candidate) > -contact_abs_log:
                continue  # off a contact — keep retreating to snap to blob edge
            new_end = candidate
            retreated_mm = dist
            break
        if retreated_mm > 0.0:
            rec["end_ras"] = new_end
            rec["length_mm"] = float(np.linalg.norm(new_end - s))
            # Update segment cache so subsequent trajectories see the
            # retreated tip (their clearance check against ``i`` uses
            # the shrunken segment — a fairer starting point).
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
                        f"{_min_perp_to_other_segments(new_end, segs, i):.2f} mm, "
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


def _refine_deep_end_via_axis_log(rec, log_arr, ras_to_ijk_mat,
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

    def _disk_hit(p_axis):
        for dr_u, dr_v in disk_offsets:
            p = p_axis + dr_u * u + dr_v * v
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


# (STRONG_CONTACT_AMP_MIN removed — per-subject LoG amplitudes vary
# too widely to share a single threshold. T4 saturates at ~1500
# while T22 hits 2200+; a fixed 1000 floor excluded most T4 inliers,
# collapsing the clip ceiling. Walker inliers are already vetted by
# the pitch + geometry tests, so the clip trusts them all.)


def _axis_to_skull_synth(shallow_ras, deep_ras, dist_arr, ras_to_ijk_mat,
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


def _clip_deep_end_to_inliers(rec, log_arr=None, ras_to_ijk_mat=None,
                                margin_mm=DEEP_END_MARGIN_PAST_LAST_CONTACT_MM):
    """Clip ``end_ras`` back so it sits no more than ``margin_mm`` past
    the deepest walker inlier projected onto the shank axis. Walker
    inliers are already vetted by pitch + geometry tests in the
    walker + extension stages, so a separate LoG-amp filter is
    redundant and brittle across subjects with varying saturation
    (T4 saturates at ~1500, T22 at 2200+). All inliers count.

    No-op when the trajectory has no ``inlier_ras`` (e.g., stage-2
    Frangi trajectories that skip the walker).
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
                             return_features=False, progress_logger=None,
                             suggestion_vendors=None,
                             pitch_strategy=None,
                             pitches_mm=None):
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
    # Scanner-invariance pre-clip: cap HU at HU_CLIP_MAX so the LoG /
    # Frangi response is consistent across scans regardless of how the
    # CT reconstruction encodes metal saturation. Does not affect hull
    # detection (thresholded at HU ≥ -500) or any of the downstream HU
    # filters (all below HU_CLIP_MAX).
    img = sitk.Clamp(img, lowerBound=-1024.0, upperBound=HU_CLIP_MAX)
    _log("preprocessing: hull, head-distance, intracranial mask…")
    ct_arr_kji = sitk.GetArrayFromImage(img).astype(np.float32)
    # Input fingerprint — lets us compare Slicer vs CLI runs byte-for-byte.
    # If Slicer returns a different trajectory count, the most common
    # causes are (a) HU rescaling (NIfTI scl_slope/scl_inter applied
    # differently) and (b) IJK→RAS matrix mismatch; this trace exposes both.
    try:
        _sp = img.GetSpacing()
        _dg = [ijk_to_ras_mat[i, i] for i in range(3)]
        _org = [ijk_to_ras_mat[i, 3] for i in range(3)]
        _log(
            f"input fingerprint: shape={ct_arr_kji.shape} "
            f"HU[min/mean/max]={ct_arr_kji.min():.1f}/"
            f"{ct_arr_kji.mean():.1f}/{ct_arr_kji.max():.1f} "
            f"spacing={tuple(f'{s:.4f}' for s in _sp)} "
            f"ijk2ras_diag={tuple(f'{d:+.4f}' for d in _dg)} "
            f"origin={tuple(f'{o:+.2f}' for o in _org)}"
        )
    except Exception:
        pass
    hull, intracranial, dist_arr = build_masks(img)
    _log("preprocessing: LoG σ=1…")
    log1 = log_sigma(img, sigma_mm=LOG_SIGMA_MM)
    _log("preprocessing: Frangi σ=1…")
    frangi_s1 = frangi_single(img, sigma=FRANGI_STAGE1_SIGMA)
    kji_to_ras = _kji_to_ras_fn_from_matrix(ijk_to_ras_mat)

    # Resolve walker pitches from the caller's strategy. Explicit
    # ``pitches_mm`` override takes precedence (used by unit tests and
    # power users). Otherwise fall back to strategy lookup — "auto"
    # auto-detects pitch from the intracranial blob cloud here so
    # stage-1 sees the right pitches from its first pass.
    if pitches_mm is not None and len(tuple(pitches_mm)) > 0:
        resolved_pitches = tuple(float(p) for p in pitches_mm)
    elif pitch_strategy is not None:
        strat_key = str(pitch_strategy).lower()
        if strat_key == "auto":
            _log("auto-detect pitch: extracting blobs…")
            _blobs_preview = extract_blobs(log1, threshold=LOG_BLOB_THRESHOLD)
            _pts_preview = (
                np.array([kji_to_ras(b["kji"]) for b in _blobs_preview])
                if _blobs_preview
                else np.empty((0, 3), dtype=float)
            )
            if _pts_preview.shape[0] > 0:
                _n_vox_preview = np.array(
                    [b["n_vox"] for b in _blobs_preview], dtype=int,
                )
                _pts_c_preview = _pts_preview[_n_vox_preview <= LOG_BLOB_MAX_VOXELS]
            else:
                _pts_c_preview = _pts_preview
            _raw_detected = detect_pitch_from_intracranial_blobs(
                _pts_c_preview, dist_arr, ras_to_ijk_mat,
            ) if _pts_c_preview.shape[0] > 0 else []
            resolved_pitches = resolve_pitches_for_strategy(
                "auto",
                pts_c=_pts_c_preview,
                dist_arr=dist_arr,
                ras_to_ijk_mat=ras_to_ijk_mat,
            )
            if _raw_detected:
                _log(
                    f"auto-detect pitch: raw={[f'{p:.2f}' for p in _raw_detected]} "
                    f"→ snapped={[f'{p:.2f}' for p in resolved_pitches]} mm"
                )
            else:
                _log(
                    f"auto-detect pitch: using {[f'{p:.2f}' for p in resolved_pitches]} mm (fallback)"
                )
        else:
            resolved_pitches = resolve_pitches_for_strategy(strat_key)
    else:
        resolved_pitches = (PITCH_MM,)

    _log(
        f"stage 1: blob-pitch walker — pitches={[f'{p:.2f}' for p in resolved_pitches]} mm"
    )
    stage1_lines, pts_blobs = run_stage1(
        log1, kji_to_ras, dist_arr, ras_to_ijk_mat,
        pitches_mm=resolved_pitches,
    )
    _log(f"stage 1: {len(stage1_lines)} candidate lines after walk + arbitrate + extend")
    # Attach inlier RAS coords AND LoG amplitudes to each stage-1 line
    # so post-anchor refinement can clip the deep end to the last
    # STRONG real contact (weak/noisy blobs added by extension don't
    # count as legit deep endpoints).
    import numpy as _np
    # Re-derive LoG amplitudes at each contact-sized blob position —
    # pts_blobs is already the contact-filtered cloud, so indexing
    # matches line["inlier_idx"].
    try:
        K, J, I = log1.shape
        h_all = _np.concatenate([pts_blobs, _np.ones((pts_blobs.shape[0], 1))], axis=1)
        ijk_all = (ras_to_ijk_mat @ h_all.T).T[:, :3]
        ii = _np.clip(_np.round(ijk_all[:, 0]).astype(int), 0, I - 1)
        jj = _np.clip(_np.round(ijk_all[:, 1]).astype(int), 0, J - 1)
        kk = _np.clip(_np.round(ijk_all[:, 2]).astype(int), 0, K - 1)
        blob_amps = _np.abs(log1[kk, jj, ii]).astype(_np.float32)
    except Exception:
        blob_amps = None
    for l in stage1_lines:
        try:
            l["inlier_ras"] = _np.asarray(pts_blobs[l["inlier_idx"]], dtype=float)
            if blob_amps is not None:
                l["inlier_amps"] = _np.asarray(blob_amps[l["inlier_idx"]], dtype=float)
        except Exception:
            l["inlier_ras"] = None
            l["inlier_amps"] = None
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
    # Secondary HU-based bolt pool. Titanium bolts saturate above 2000
    # HU regardless of their LoG-σ=1 response, so this catches bolts
    # that the LoG extractor misses (e.g., T4, ct88 subjects with
    # weaker LoG bolt signature). Used only as a per-line rescue: the
    # anchor retries against this pool, tube-filtered to the specific
    # shank, when the primary LoG pool yielded no anchor for a
    # high-confidence stage-1 line. Never offered globally.
    rescue_bolts_hu, _ = extract_bolt_candidates(
        log1, dist_arr, ijk_to_ras_mat, img.GetSpacing(),
        ct_arr=ct_arr_kji, hu_threshold=BOLT_HU_RESCUE_THRESHOLD,
        hull_proximity_mm=BOLT_HU_RESCUE_HULL_PROX_MM,
    )
    _log(f"bolt rescue pool (HU ≥ {BOLT_HU_RESCUE_THRESHOLD:.0f}): "
         f"{len(rescue_bolts_hu)} candidates")

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
            dist_mean_mm=float(l.get("dist_mean_mm", float("nan"))),
        )
        if source == "stage1":
            rec["amp_sum"] = float(l.get("amp_sum", 0.0))
            # Preserve the pre-anchor inlier span — the actual distance
            # between the shallowest and deepest detected contacts. The
            # post-anchor ``length_mm`` overwrites this with the
            # bolt-tip → deep-tip length, so downstream classifiers
            # need this field to see the true contact span.
            rec["contact_span_mm"] = float(l.get("span_mm", 0.0))
            # Pre-extend walker span, used by the HU rescue's "strong
            # stage-1" gate.
            rec["original_span_mm"] = float(
                l.get("original_span_mm", l.get("span_mm", 0.0))
            )
            rec["original_median_pitch_mm"] = float(l.get(
                "original_median_pitch_mm",
                (float(l.get("original_span_mm", l.get("span_mm", 0.0)))
                 / max(1, int(l.get("n_blobs", 2)) - 1)),
            ))
            # Inlier RAS coords + amplitudes — used by the deep-end
            # refinement to clip at last-STRONG-contact + small margin
            # (independent of LoG thresholds).
            inlier_ras = l.get("inlier_ras")
            if inlier_ras is not None:
                rec["inlier_ras"] = np.asarray(inlier_ras, dtype=float)
            inlier_amps = l.get("inlier_amps")
            if inlier_amps is not None:
                rec["inlier_amps"] = np.asarray(inlier_amps, dtype=float)
        return rec

    def _filter_bolts_near_axis(bolt_list, start_ras, end_ras,
                                   tube_radius_mm=BOLT_TUBE_RADIUS_MM,
                                   outward_mm=BOLT_SEARCH_OUTWARD_MM,
                                   inward_mm=BOLT_MAX_INWARD_ALONG_MM):
        """Keep only bolt CCs whose voxels sit within the shank's tube.
        Per-line spatial gate for the HU rescue so the secondary pool
        never acts globally.
        """
        s = np.asarray(start_ras, dtype=float)
        e = np.asarray(end_ras, dtype=float)
        d = e - s
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            return []
        axis = d / L
        out = []
        for b in bolt_list:
            pts = b.get("pts_ras")
            if pts is None or len(pts) == 0:
                continue
            diffs = np.asarray(pts) - s
            along = diffs @ axis
            perp_vec = diffs - np.outer(along, axis)
            perp = np.linalg.norm(perp_vec, axis=1)
            in_tube = (
                (perp <= tube_radius_mm)
                & (along <= inward_mm)
                & (along >= -outward_mm)
            )
            if int(in_tube.sum()) >= BOLT_MIN_TUBE_VOXELS:
                out.append(b)
        return out

    def _is_rescue_candidate(rec):
        """Strong-SEEG-chain gate for the HU bolt rescue.

        A real SEEG electrode's walker pre-extension line has Dixi/PMT
        pitch (3-5 mm). Uses the pre-extend MEDIAN pitch — robust to
        one walker-absorbed outlier that would skew a mean-based
        statistic past the 7 mm cap even when most inliers sit on a
        regular chain. Falls back to min(span_pre, span_post)/(n-1)
        when median isn't available.
        """
        if rec.get("source") != "stage1":
            return False
        n = int(rec.get("n_inliers", 0))
        if n < BOLT_RESCUE_MIN_N_INLIERS:
            return False
        dist_max = float(rec.get("dist_max_mm", 0.0))
        if dist_max < BOLT_RESCUE_MIN_DIST_MAX_MM:
            return False
        span_post = float(rec.get("contact_span_mm", rec.get("length_mm", 0.0)))
        span_pre = float(rec.get("original_span_mm", span_post))
        span_for_pitch = min(span_pre, span_post) if span_pre > 0 else span_post
        fallback_avg = span_for_pitch / (n - 1) if n > 1 else float("inf")
        median_pitch = float(rec.get("original_median_pitch_mm", fallback_avg))
        if median_pitch > DEEP_TIP_SHORT_MAX_AVG_PITCH_MM:
            return False
        return True

    # Anchor each candidate to a bolt BEFORE dedup. No bolt == not a
    # real electrode. Then apply length and air-sinus filters; both
    # catch stage-2 false positives that look nothing like real shanks.
    def _anchor_or_reject(rec):
        # ``_orient_shallow_to_deep`` upstream uses hull head-distance
        # to pick which endpoint is the shallow one, but that's
        # ambiguous for trajectories whose deep tip sits in a deep
        # sulcus as close to its local hull surface as the bolt side
        # (T22 LGR: orbital-floor tip is ~10 mm from hull, skull-top
        # bolt is ~15 mm from hull — orientation flipped). Let the
        # bolt CC decide by trying both orientations and keeping the
        # one whose bolt anchor has more tube voxels. Falls back to
        # either non-None result when only one orientation anchors.
        def _try_anchor(bolt_pool):
            fwd = anchor_trajectory_to_bolt(
                rec["start_ras"], rec["end_ras"], bolt_pool,
            )
            bwd = anchor_trajectory_to_bolt(
                rec["end_ras"], rec["start_ras"], bolt_pool,
            )
            fwd_n = int(fwd[2].get("tube_n_vox", 0)) if fwd[2] is not None else 0
            bwd_n = int(bwd[2].get("tube_n_vox", 0)) if bwd[2] is not None else 0
            return fwd, bwd, fwd_n, bwd_n

        fwd, bwd, fwd_n, bwd_n = _try_anchor(bolts)
        # HU rescue: only when the primary LoG pool finds nothing AND
        # the stage-1 line looks unambiguously like a real SEEG chain.
        # The rescue bolts are tube-filtered to this specific shank so
        # the secondary HU pool never anchors anything off-axis.
        if fwd_n == 0 and bwd_n == 0 and _is_rescue_candidate(rec):
            local = _filter_bolts_near_axis(
                rescue_bolts_hu, rec["start_ras"], rec["end_ras"],
                tube_radius_mm=BOLT_HU_RESCUE_TUBE_RADIUS_MM,
            )
            if local:
                # Widened anchor tube too: walker axis can drift 1-3°
                # from true shank axis, pushing the bolt up to 3-5 mm
                # perpendicular even when it's really on-axis.
                def _try_anchor_wide(bolt_pool):
                    fwd = anchor_trajectory_to_bolt(
                        rec["start_ras"], rec["end_ras"], bolt_pool,
                        tube_radius_mm=BOLT_HU_RESCUE_TUBE_RADIUS_MM,
                    )
                    bwd = anchor_trajectory_to_bolt(
                        rec["end_ras"], rec["start_ras"], bolt_pool,
                        tube_radius_mm=BOLT_HU_RESCUE_TUBE_RADIUS_MM,
                    )
                    fwd_n = int(fwd[2].get("tube_n_vox", 0)) if fwd[2] is not None else 0
                    bwd_n = int(bwd[2].get("tube_n_vox", 0)) if bwd[2] is not None else 0
                    return fwd, bwd, fwd_n, bwd_n
                fwd, bwd, fwd_n, bwd_n = _try_anchor_wide(local)

        if bwd_n > fwd_n:
            # Orientation was wrong; flip before writing results back.
            rec["start_ras"], rec["end_ras"] = (
                np.asarray(rec["end_ras"], dtype=float),
                np.asarray(rec["start_ras"], dtype=float),
            )
            new_start, skull_entry, bolt = bwd
        else:
            new_start, skull_entry, bolt = fwd
        # Axis-to-skull synthetic rescue: when both the LoG and HU
        # bolt pools failed AND the walker line has strong-SEEG-chain
        # evidence, synthesize a skull_entry + bolt_tip by walking
        # the walker axis outward until it crosses the hull surface.
        # Recovers T4-class subjects whose bolts sit outside the CT
        # acquisition window (so no bolt CC at any threshold).
        bolt_from_synth = None
        if new_start is None and _is_rescue_candidate(rec):
            s0, e0 = _orient_shallow_to_deep(
                rec["start_ras"], rec["end_ras"],
                dist_arr, ras_to_ijk_mat,
            )
            synth_skull, synth_tip = _axis_to_skull_synth(
                s0, e0, dist_arr, ras_to_ijk_mat,
            )
            if synth_skull is not None:
                rec["start_ras"] = np.asarray(s0, dtype=float)
                rec["end_ras"] = np.asarray(e0, dtype=float)
                new_start = synth_tip
                skull_entry = synth_skull
                bolt_from_synth = {"n_vox": 0, "dist_min_mm": float("nan"),
                                    "id": -1}
        if new_start is None:
            return None
        if bolt_from_synth is not None:
            bolt = bolt_from_synth
            rec["bolt_source"] = "axis_synth"
        rec["start_ras"] = np.asarray(new_start, dtype=float)
        if skull_entry is not None:
            rec["skull_entry_ras"] = np.asarray(skull_entry, dtype=float)
        rec["length_mm"] = float(np.linalg.norm(rec["end_ras"] - rec["start_ras"]))
        # Length sanity: real SEEG total length (bolt + shank) is bounded.
        if (rec["length_mm"] < MIN_POST_ANCHOR_LEN_MM
                or rec["length_mm"] > MAX_POST_ANCHOR_LEN_MM):
            return None
        # Air-sinus AND in-bone rejection along the intracranial portion.
        # Real electrodes cross brain parenchyma (HU ≈ 20-50) punctuated
        # by metal contact spikes; sinus tubes are hollow (HU air); FP
        # trajectories threaded through bone sit at HU 300-1500
        # throughout. Check both fractions in one pass.
        intracranial_start = rec.get("skull_entry_ras", rec["start_ras"])
        air_frac, bone_frac = _trajectory_hu_fractions(
            intracranial_start, rec["end_ras"], ct_arr_kji, ras_to_ijk_mat,
        )
        if air_frac > AIR_FRAC_MAX:
            return None
        if bone_frac > BONE_FRAC_MAX:
            return None
        # Volume-boundary check: trajectories that extend past the
        # scanned CT FOV (oblique acquisition → -1024 HU in the
        # extrapolated region) have their deep portion sitting in
        # background voxels. Real trajectories end in brain tissue.
        deep_nodata = _deep_tip_nodata_fraction(
            rec.get("skull_entry_ras", rec["start_ras"]),
            rec["end_ras"], ct_arr_kji, ras_to_ijk_mat,
        )
        if deep_nodata > DEEP_TIP_NODATA_FRAC_MAX:
            return None
        # Cross-bone skim FP filter: bone-heavy path + few inliers +
        # low per-inlier amp. Real SEEG shanks that pass through
        # some bone still have enough inliers (≥ 9) and saturate
        # amp (≥ 1200 per inlier); skim-FPs stitch a handful of
        # sparse bone blobs with weak amplitudes (subject-137 X2,
        # X3 examples).
        if rec.get("source") == "stage1":
            n_in = int(rec.get("n_inliers", 0))
            amp_per = float(rec.get("amp_sum", 0.0)) / max(1, n_in)
            if (bone_frac >= BONE_SKIM_MIN_BONE_FRAC
                    and n_in <= BONE_SKIM_MAX_N_INLIERS
                    and amp_per < BONE_SKIM_MAX_AMP_PER_IN):
                return None
        rec["air_fraction"] = float(air_frac)
        rec["bone_fraction"] = float(bone_frac)
        rec["bolt_n_vox"] = int(bolt["n_vox"])
        rec["bolt_dist_min_mm"] = float(bolt["dist_min_mm"])
        rec["bolt_id"] = int(bolt.get("id", -1))
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

    # Axis-directed deep-end refinement. The 3D regional-minima blob
    # extractor misses contacts when the per-contact LoG wells merge
    # into one continuous CC (seen on T2 X06 / RAI, where the deep 3–4
    # contacts sit inside a single long bright shaft and don't produce
    # distinct 3D minima). Sample the LoG profile 1-dimensionally along
    # the trajectory axis and push ``end_ras`` out to the last real
    # contact peak. Stage-2 trajectories share the problem (they are
    # explicitly the no-per-contact case) so we refine both.
    for rec in anchored:
        new_end = _refine_deep_end_via_axis_log(
            rec, log1, ras_to_ijk_mat,
        )
        if new_end is not None:
            rec["end_ras"] = new_end
        # Hard cap: end must sit within DEEP_END_MARGIN_PAST_LAST_CONTACT_MM
        # of the deepest walker inlier. No SEEG electrode has a long gap
        # past its last contact; anything further is over-reach.
        clipped = _clip_deep_end_to_inliers(rec)
        if clipped is not None:
            rec["end_ras"] = clipped
        if new_end is not None or clipped is not None:
            rec["length_mm"] = float(
                np.linalg.norm(
                    np.asarray(rec["end_ras"]) - np.asarray(rec["start_ras"])
                )
            )

    # Crossing-tip retreat: after all trajectories have been extended,
    # pull back any tip that lives inside another's contact-acceptance
    # tube. Runs only at the final stage so every axis has settled
    # before we decide which tip is the intruder. Passing the LoG
    # volume lets the retreat additionally snap the pulled-back tip to
    # the deep edge of the last real contact instead of floating in the
    # gap between contacts.
    _log("crossing-tip retreat…")
    _retreat_crossing_tips(
        anchored,
        log_arr=log1,
        ras_to_ijk_mat=ras_to_ijk_mat,
        logger=_log,
    )

    # Intracranial-only length (skull entry → deep tip). The existing
    # ``length_mm`` is bolt-tip → deep-tip and includes ~15–25 mm of bolt
    # protrusion outside the skull; downstream displays/clinical reporting
    # want the part that actually sits inside the brain.
    for rec in anchored:
        entry = np.asarray(
            rec.get("skull_entry_ras", rec.get("start_ras")),
            dtype=float,
        )
        end = np.asarray(rec["end_ras"], dtype=float)
        rec["intracranial_length_mm"] = float(np.linalg.norm(end - entry))

    # Suggested electrode model per stage-1 trajectory. Uses the
    # pre-anchor contact span + inlier count against the library,
    # filtered by the caller's ``suggestion_vendors`` selection (or
    # all known vendors when not specified). Advisory only — downstream
    # modules such as Contacts & Trajectory View do the actual contact
    # fitting and the user can override this suggestion.
    if suggestion_vendors is not None:
        vendors_for_suggest = tuple(suggestion_vendors)
    elif pitch_strategy is not None:
        strat_key = str(pitch_strategy).lower()
        vendors_for_suggest = PITCH_STRATEGY_VENDORS.get(
            strat_key, tuple(VENDOR_ID_PREFIXES.keys()),
        )
    else:
        vendors_for_suggest = tuple(VENDOR_ID_PREFIXES.keys())
    if not vendors_for_suggest:
        _log("no vendors selected; skipping electrode suggestions")
        _models = []
    else:
        try:
            from rosa_core.electrode_models import load_electrode_library
            _library = load_electrode_library()
            _models = list(_library.get("models") or [])
        except Exception as exc:
            _log(f"electrode library load failed ({exc}); no suggestions emitted")
            _models = []
    if _models:
        n_suggested = 0
        for rec in anchored:
            # Shortest electrode that covers the intracranial length is
            # a more robust primary signal than walker-inlier count/span:
            # it is anchored on skull_entry→deep_tip geometry, which the
            # bolt anchor produces reliably even when the walker misses
            # contacts (e.g. the T22-X02 case where the walker stopped
            # at 7/15 contacts but the intracranial length still
            # reflects the real 15CM shank). Same rule for stage-1 and
            # stage-2 — only their input metric differs in practice.
            intra = float(rec.get("intracranial_length_mm") or 0.0)
            if intra < 5.0:
                continue
            best = suggest_shortest_covering_model(
                intra, _models, vendors=vendors_for_suggest,
            )
            if best is None:
                continue
            rec["suggested_model_id"] = str(best["model_id"])
            rec["suggested_model_method"] = "shortest_covering"
            rec["suggested_model_length_mm"] = float(best["model_length_mm"])
            rec["suggested_model_gap_mm"] = float(best["gap_mm"])
            # Score proxy: how snug the fit is — smaller is tighter.
            # (Mostly for logging/tie-break; Contacts & Trajectory View
            # only reads ``best_model_id`` to populate its dropdown.)
            rec["suggested_model_score"] = float(abs(best["gap_mm"]))
            n_suggested += 1
        _log(
            f"suggested electrodes: {n_suggested} trajectories "
            f"(vendors={'+'.join(vendors_for_suggest)})"
        )

    # Convert to JSON-safe dicts (tuples of floats).
    trajectories: list[dict[str, Any]] = []
    for rec in anchored:
        out = dict(rec)
        out["start_ras"] = [float(x) for x in rec["start_ras"]]
        out["end_ras"] = [float(x) for x in rec["end_ras"]]
        if "skull_entry_ras" in rec:
            out["skull_entry_ras"] = [float(x) for x in rec["skull_entry_ras"]]
        trajectories.append(out)

    # Per-trajectory fingerprint — compact trace makes it easy to diff
    # Slicer-run results against a CLI run to spot which specific
    # trajectory disappeared / shifted when subject-level totals don't match.
    try:
        _log(f"trajectory summary ({len(trajectories)} kept):")
        for _i, _t in enumerate(trajectories):
            _se = _t.get("skull_entry_ras") or _t.get("start_ras") or [0.0, 0.0, 0.0]
            _en = _t.get("end_ras") or [0.0, 0.0, 0.0]
            _src = str(_t.get("source") or "?")
            _n = int(_t.get("n_inliers") or 0)
            _sp = float(_t.get("contact_span_mm") or 0.0)
            _log(
                f"  [{_i:02d}] src={_src} n={_n} span={_sp:.1f}mm "
                f"skull_entry=({_se[0]:+.1f},{_se[1]:+.1f},{_se[2]:+.1f}) "
                f"deep_tip=({_en[0]:+.1f},{_en[1]:+.1f},{_en[2]:+.1f})"
            )
    except Exception:
        pass

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


# ---- Post-detection electrode classification -------------------------

VENDOR_ID_PREFIXES = {"Dixi": "DIXI-", "PMT": "PMT-"}


def _vendor_prefixes(vendors):
    return tuple(
        VENDOR_ID_PREFIXES[v] for v in (vendors or ()) if v in VENDOR_ID_PREFIXES
    )


def suggest_shortest_covering_model(intracranial_length_mm, models,
                                     vendors=("Dixi",),
                                     dura_tolerance_mm=10.0):
    """Pick the shortest electrode whose full exploration span plus a
    dura tolerance covers the given intracranial length.

    ``intracranial_length_mm`` is ``|skull_entry_ras − end_ras|`` — the
    distance from the bone/dura band (where ``skull_entry_ras`` is
    placed) to the detected deep tip. Because ``skull_entry_ras`` sits
    inside the skull/dura band (``head_distance ≤ 10 mm``) rather than
    exactly at the first contact, the observed length overstates the
    electrode's active length by roughly 5–10 mm of soft-tissue margin.
    ``dura_tolerance_mm`` absorbs that offset so a 15CM (70 mm active
    length) still matches a 76 mm observed intracranial span.

    Primary rule: smallest ``total_exploration_length_mm`` for which
    ``total + dura_tolerance_mm ≥ intracranial_length_mm``.

    Returns ``{"model_id", "model_length_mm", "gap_mm"}`` or ``None``
    if no model is long enough (or no vendor selected).
    """
    prefixes = _vendor_prefixes(vendors)
    if not prefixes:
        return None
    L = float(intracranial_length_mm)
    tol = float(dura_tolerance_mm)
    best = None
    for model in models:
        mid = str(model.get("id") or "")
        if not mid.startswith(prefixes):
            continue
        total = model.get("total_exploration_length_mm")
        if total is None:
            offsets = model.get("contact_center_offsets_from_tip_mm") or []
            if len(offsets) < 2:
                continue
            total = float(offsets[-1]) - float(offsets[0])
        total = float(total)
        if total + tol < L:
            continue
        if best is None or total < best["model_length_mm"]:
            best = {
                "model_id": mid,
                "model_length_mm": total,
                "gap_mm": total - L,
            }
    return best


def classify_by_count_and_span(n_observed, span_observed_mm,
                                models, vendors=("Dixi",),
                                count_weight_mm=3.5):
    """Pick the electrode model best explained by an observed
    (contact-count, span) pair.

    Score = ``|N_model − n_observed| * count_weight_mm +
    |span_model − span_observed_mm|``. ``count_weight_mm=3.5`` means
    1 missing contact is worth ~1 pitch of span error, so models
    that match on count dominate.

    ``span_model = offsets[-1] − offsets[0]``, the tip-contact-to-deep-
    contact span (NOT the full electrode exploration length).

    ``vendors`` filters models by vendor-id prefix ("Dixi" → "DIXI-",
    "PMT" → "PMT-"). Models outside the prefix set are skipped.

    Returns ``{"model_id", "score", "count_err", "span_err"}`` or
    ``None`` if no candidate survives the vendor filter.
    """
    prefixes = _vendor_prefixes(vendors)
    if not prefixes:
        return None
    best = None
    for model in models:
        mid = str(model.get("id") or "")
        if not mid.startswith(prefixes):
            continue
        offsets = model.get("contact_center_offsets_from_tip_mm") or []
        if len(offsets) < 2:
            continue
        n_model = int(model.get("contact_count") or len(offsets))
        span_model = float(offsets[-1]) - float(offsets[0])
        count_err = abs(int(n_observed) - n_model)
        span_err = abs(float(span_observed_mm) - span_model)
        score = count_err * float(count_weight_mm) + span_err
        if best is None or score < best["score"]:
            best = {
                "model_id": mid,
                "score": float(score),
                "count_err": int(count_err),
                "span_err": float(span_err),
                "n_model": n_model,
                "span_model_mm": span_model,
            }
    return best
