"""Bolt detection + per-trajectory bolt anchoring.

Two pieces of the contact_pitch_v1 pipeline:

  ``extract_bolt_candidates`` — find connected metal-evidence CCs that
  reach the hull surface and could plausibly be skull bolts.

  ``anchor_trajectory_to_bolt`` — project each candidate's voxels onto
  one trajectory's shank axis; the CC with the most voxels inside a
  tube around the axis becomes the trajectory's bolt anchor and pins
  ``skull_entry_ras`` at the deepest tube voxel still in the
  skull/dura band.

Plus the metal-evidence + bolt-CC tuning constants both functions
share. Imported by the orchestrator (``run_two_stage_detection`` in
``contact_pitch_v1_fit``) and by ``guided_fit_engine`` (which uses
the unified metal-evidence path for its own bolt extraction).

No Slicer / VTK / Qt dependencies.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------
# Bolt detection: unified metal-evidence pass.
#
# A bolt CC is any connected blob of saturating metal-evidence voxels
# that reaches near/outside the hull surface. Per-voxel evidence is the
# normalized maximum of |LoG| and HU:
#
#     evidence(v) = max(|LoG(v)| / LOG_BOLT_NORMALIZER,
#                       max(0, HU(v)) / HU_BOLT_NORMALIZER)
#
# The CC threshold (METAL_BOLT_THRESHOLD = 1.0) means "either LoG saturates
# at the contact-bolt level (|LoG| ≥ 800) OR HU saturates at the
# titanium-bolt level (HU ≥ 2000)". One pass replaces the historical
# 3-tier cascade (LoG@800 → HU@2000 → axis-synth) — when the CC pass
# finds no anchor and the axis crosses the hull, axis-synth still kicks
# in as the only remaining fallback (covers bolts outside the CT FOV).
#
# Why per-voxel-max rather than two separate CC passes:
#   - Titanium bolts saturate HU first; platinum-iridium contacts saturate
#     LoG first. Both signals are "definitely metal" so taking the max
#     captures any bolt regardless of vendor and the hull-touch + tube
#     filters keep skull-only CCs out.
#   - One pass produces stable per-shank CCs. Today's split caused
#     duplicate CCs (LoG bolt + HU bolt of the same physical bolt) which
#     the orientation-flip step had to disambiguate.
LOG_BOLT_NORMALIZER = 800.0         # |LoG| level at which platinum-iridium
                                    # bolts saturate. Higher than
                                    # ``LOG_BLOB_THRESHOLD = 300`` (used
                                    # for contacts) — bolts are denser
                                    # than contacts. Below this, scalp
                                    # metal and partial-volume artifacts
                                    # contribute. Used as the LoG arm of
                                    # the unified bolt evidence and of
                                    # the score-side metal_continuity
                                    # term so a single change here
                                    # tightens / loosens both at once.
HU_BOLT_NORMALIZER = 2000.0         # HU level at which titanium bolts
                                    # saturate. Cortical bone peaks
                                    # around 1500 HU so 2000 cleanly
                                    # separates the two. Used as the HU
                                    # arm of the unified bolt evidence
                                    # and of the score-side
                                    # metal_continuity term.
METAL_BOLT_THRESHOLD = 1.0          # Unified normalized cutoff. 1.0 means
                                    # "saturates in at least one modality".
                                    # Calibrated 2026-04-28 across the
                                    # 22-subject dataset: 295/295 matched
                                    # anchor-or-synth, 4 must-preserve
                                    # shanks behave correctly (T4 RSFG
                                    # anchors via HU; T3 LAI / T4 LCMN /
                                    # T4 LSFG fall through to synth as
                                    # their bolts sit outside CT FOV).
                                    # Lowering admits scalp CC mega-merges
                                    # (94k+ voxels at 0.85, 442k at 0.5);
                                    # raising drops thin LoG bolts.

# Bolt CC + tube-search tuning ----------------------------------------

BOLT_MIN_VOXELS = 20                # drop tiny (isolated-contact) CCs
BOLT_HULL_PROXIMITY_MM = 5.0        # CC must reach within this many mm
                                    # of the hull surface. 5 mm covers
                                    # thin-wire PMT designs whose bolt-
                                    # like dense structure sits 2-5 mm
                                    # inside the hull rather than poking
                                    # through. Per-line tube filtering
                                    # still keeps far-away CCs out, so
                                    # the loose value doesn't admit
                                    # skull-band noise.
BOLT_TUBE_RADIUS_MM = 5.0           # shank-axis tube for bolt-voxel count.
                                    # 5 mm tolerates 1-3° walker-axis
                                    # drift, which can put real bolts
                                    # 3-5 mm laterally offset over a
                                    # 40-60 mm outward reach. The 3 mm
                                    # value used pre-unification missed
                                    # T4 / thin-wire-PMT bolts.
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


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def extract_bolt_candidates(log_arr, dist_arr, ijk_to_ras_mat, spacing_xyz,
                             threshold=LOG_BOLT_NORMALIZER,
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
        # When no tube voxel reaches the skull band (e.g. cropped
        # scans where the bolt is partly outside FOV, or the tube
        # axis grazes the bolt CC), set deep_local=None so the caller
        # leaves ``skull_entry_ras`` unset and falls back to
        # ``start_ras`` for intracranial-length reporting — preferable
        # to placing a bogus skull-entry deep inside the head.
        deep_local = None
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
        if n_in > best_n:
            best = bolt
            best_n = n_in
            # Project chosen voxels onto the SHANK axis so endpoints
            # stay strictly colinear with the trajectory line. Without
            # this, the markers can sit up to tube_radius_mm off the
            # axis since they're picked from voxel centers.
            shallow_along = float(along[shallow_local])
            best_shallow = traj_start + shallow_along * shank_axis
            if deep_local is None:
                best_entry = None
            else:
                deep_along = float(along[deep_local])
                best_entry = traj_start + deep_along * shank_axis

    if best is None:
        return None, None, None
    # Stash the in-tube voxel count so callers that try both
    # orientations (shallow→deep ambiguity for trajectories whose
    # endpoints have similar hull distance) can compare strength.
    best = dict(best)
    best["tube_n_vox"] = int(best_n)
    return best_shallow, best_entry, best
