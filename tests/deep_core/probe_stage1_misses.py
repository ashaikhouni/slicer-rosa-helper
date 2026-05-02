"""Walk through each of the 6 CLEAN shanks that stage 1 misses, and
diagnose why the walker doesn't emit a line there.

For each miss:
  1. Snap the GT via guided fit → reference axis.
  2. Dump LoG blobs inside a 2 mm tube around the snapped axis —
     their axial depth, amplitude, perp distance.
  3. Check whether those blobs sit on the walker's contact-sized list
     (pts_c after the n_vox ≤ LOG_BLOB_MAX_VOXELS gate).
  4. Build the pairwise-distance matrix between in-tube blobs and
     count how many pairs fall inside walker's pitch windows
     (multipliers 1/2/3 × pitch ± tol). If zero pairs qualify, the
     walker can't even seed on this shank.
  5. For each in-tube blob, find whether some OTHER stage-1 line
     claimed it as an inlier — arbitration theft.
  6. Try "force-walking" the in-tube blobs: pick the two strongest
     within pitch window, call ``_walk_line`` directly and report
     the resulting n_blobs / amp_sum. If walk succeeds and amp_sum
     passes AMP_SUM_MIN, the miss is a dedup or arbitration kill,
     not a walker refusal.

Shanks to inspect (from probe_stage1_vs_snapped_gt.py):
    T7 LSFG, T8 LSFG, T11 RAI, T12 L^CMN, T20 LASF, T24 RPRG.

Run
---
    /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_stage1_misses.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper")
sys.path.insert(0, str(ROOT / "CommonLib"))
sys.path.insert(0, str(ROOT / "PostopCTLocalization"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import SimpleITK as sitk

from rosa_detect import contact_pitch_v1_fit as cpfit
from rosa_detect import guided_fit_engine as gfe
from eval_seeg_localization import (
    iter_subject_rows, image_ijk_ras_matrices,
    load_reference_ground_truth_shanks,
)

DATASET_ROOT = Path("/Users/ammar/Dropbox/thalamus_subjects/seeg_localization")

MISS_LIST = [
    ("T7", "LSFG"),
    ("T8", "LSFG"),
    ("T11", "RAI"),
    ("T12", "L^CMN"),
    ("T20", "LASF"),
    ("T24", "RPRG"),
]

TUBE_RADIUS_MM = 2.0
STRONG_AMP = 500.0


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def _axis(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return _unit(b - a), float(np.linalg.norm(b - a))


def _blobs_in_tube(pts_ras, amps, start, end, radius_mm):
    u, L = _axis(start, end)
    if L < 1e-3:
        return np.array([], dtype=int)
    v = pts_ras - np.asarray(start, dtype=float)[None, :]
    axial = v @ u
    perp = v - axial[:, None] * u[None, :]
    perp_d = np.linalg.norm(perp, axis=1)
    keep = (axial >= -2.0) & (axial <= L + 2.0) & (perp_d <= radius_mm)
    idx = np.where(keep)[0]
    order = np.argsort(axial[idx])
    return idx[order], axial[idx[order]], perp_d[idx[order]]


def _pitch_pair_count(in_tube_pts, pitches):
    """Count blob pairs whose separation falls inside ANY pitch band
    (mult 1/2/3 × pitch ± PITCH_TOL_MM). Also return a flat list of
    qualifying pairs (i, j, dist, pitch, mult)."""
    tol = cpfit.PITCH_TOL_MM
    n = len(in_tube_pts)
    if n < 2:
        return 0, []
    dist = np.sqrt(np.sum(
        (in_tube_pts[:, None, :] - in_tube_pts[None, :, :]) ** 2, axis=2,
    ))
    pair_hits = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(dist[i, j])
            for p in pitches:
                for mult in (1, 2, 3):
                    if abs(d - mult * p) <= tol:
                        pair_hits.append((i, j, d, p, mult))
                        break
    return len(pair_hits), pair_hits


def _walk_one(seed_i, seed_j, pts_c, amps_c, pitch):
    """Direct call to the internal walker."""
    return cpfit._walk_line(int(seed_i), int(seed_j), pts_c, amps_c,
                              pitch_mm=float(pitch))


def _investigate(sid, shank_name, row, features, ijk2ras, ras2ijk, log1,
                  dist_arr, kji_to_ras, pts_c, amps_c, nvox_c):
    print(f"\n{'#'*110}\n# {sid}  {shank_name}\n{'#'*110}")

    # Locate GT shank — use redone for T4; standard otherwise.
    if sid == "T4":
        raise RuntimeError("T4 not in miss list")
    gt_shanks, _ = load_reference_ground_truth_shanks(row, None)
    match = next((g for g in gt_shanks if g.shank == shank_name), None)
    if match is None:
        print(f"  ! GT shank {shank_name} not found in subject")
        return
    snap = gfe.fit_trajectory(
        np.asarray(match.start_ras), np.asarray(match.end_ras),
        features, ijk2ras, ras2ijk,
    )
    if not snap.get("success"):
        print(f"  ! snap failed: {snap.get('reason')}")
        return
    start = np.asarray(snap["start_ras"], dtype=float)
    end = np.asarray(snap["end_ras"], dtype=float)
    u, L = _axis(start, end)
    print(
        f"  snapped axis: start={start.round(1).tolist()}  "
        f"end={end.round(1).tolist()}  length={L:.1f} mm  "
        f"(snap: n_in={snap['n_inliers']}, lat={snap['lateral_shift_mm']:.2f}, "
        f"ang={snap['angle_deg']:.2f})"
    )

    # In-tube blobs from the full features cloud (pre-contact-size filter).
    pts_all = features["blob_pts_ras"]
    amps_all = features["blob_amps"]
    idx_all, axial_all, perp_all = _blobs_in_tube(
        pts_all, amps_all, start, end, TUBE_RADIUS_MM,
    )
    print(f"\n  in-tube blobs (ALL, n_vox unfiltered): {len(idx_all)}")

    # Now intersect with contact-sized subset pts_c. Need to find which
    # pts_c rows correspond to the in-tube blobs.
    # pts_c is a SUBSET of blobs with n_vox ≤ LOG_BLOB_MAX_VOXELS.
    # Build a lookup: blob_idx_in_full → blob_idx_in_pts_c.
    # We can match by position: pts_c = pts_ras[contact_mask].
    # But we don't have pts_ras here — it's built inside run_stage1.
    # Instead, redo the same extraction.
    blobs_full = cpfit.extract_blobs(log1, threshold=cpfit.LOG_BLOB_THRESHOLD)
    # Order matches pts_all / amps_all as computed by guided_fit_engine too.
    n_vox_arr = np.array([b["n_vox"] for b in blobs_full], dtype=int)
    contact_mask = n_vox_arr <= cpfit.LOG_BLOB_MAX_VOXELS
    # Build mapping: pts_all index → pts_c index (or -1 if filtered out).
    full_to_c = np.full(len(blobs_full), -1, dtype=int)
    full_to_c[contact_mask] = np.arange(int(contact_mask.sum()))
    # Sanity: pts_c.shape[0] should match sum(contact_mask)
    # and amps_c should match amps_all[contact_mask].

    in_tube_c = [int(full_to_c[i]) for i in idx_all if full_to_c[i] >= 0]
    filtered_out = [int(i) for i in idx_all if full_to_c[i] < 0]
    print(
        f"  in-tube blobs in contact-sized subset (pts_c): "
        f"{len(in_tube_c)} / {len(idx_all)}  "
        f"(filtered by size: {len(filtered_out)})"
    )

    if not in_tube_c:
        print("  → walker has NOTHING to work with here; no contact-sized blobs.")
        return

    # Dump the contact-sized in-tube blobs with head-distance.
    K, J, I = dist_arr.shape
    print(
        f"\n  {'#':>2s} {'depth':>7s} {'|LoG|':>8s} {'perp':>6s} {'head_d':>7s}  region"
    )
    intracranial_idx = []
    for rank, ci in enumerate(in_tube_c):
        pt = pts_c[ci]
        v = pt - start
        d_ax = float(v @ u)
        d_pp = float(np.linalg.norm(v - d_ax * u))
        amp = float(amps_c[ci])
        # Head-distance at the blob's voxel.
        h = np.array([pt[0], pt[1], pt[2], 1.0])
        ijk = (ras2ijk @ h)[:3]
        ii = int(np.clip(round(ijk[0]), 0, I - 1))
        jj = int(np.clip(round(ijk[1]), 0, J - 1))
        kk = int(np.clip(round(ijk[2]), 0, K - 1))
        head_d = float(dist_arr[kk, jj, ii])
        intra = head_d >= cpfit.INTRACRANIAL_MIN_DISTANCE_MM
        if intra:
            intracranial_idx.append(rank)
        region = "INTRA" if intra else ("hull" if head_d >= -2 else "extra")
        print(f"  {rank:>2d} {d_ax:>7.2f} {amp:>8.1f} {d_pp:>6.2f} {head_d:>7.2f}  {region}")

    print(f"\n  intracranial-only tube indices: {intracranial_idx}")
    if len(intracranial_idx) >= 2:
        tube_pts_intra = pts_c[[in_tube_c[r] for r in intracranial_idx]]
        n_pairs_i, pair_hits_i = _pitch_pair_count(tube_pts_intra, [3.5])
        print(
            f"  pitch pairs (intracranial-only, pitch=3.5, ±{cpfit.PITCH_TOL_MM}, "
            f"mult 1/2/3): {n_pairs_i}"
        )
        for i, j, d, p, m in pair_hits_i[:8]:
            actual_i = intracranial_idx[i]; actual_j = intracranial_idx[j]
            print(f"    intra-pair({actual_i},{actual_j}): d={d:.2f}mm  pitch=3.5×{m}")

    # Check pitch-pair seeding.
    tube_pts = pts_c[in_tube_c]
    pitches = cpfit.resolve_pitches_for_strategy("dixi")
    n_pairs, pair_hits = _pitch_pair_count(tube_pts, pitches)
    print(
        f"\n  pitch-qualifying pairs (±{cpfit.PITCH_TOL_MM} mm, mult 1/2/3, "
        f"pitches {list(pitches)}): {n_pairs}"
    )
    if n_pairs == 0:
        print("  → NO SEED PAIRS. Walker cannot even start on this shank.")
        return
    if n_pairs <= 6:
        for i, j, d, p, m in pair_hits[:6]:
            print(f"    pair({i},{j}): d={d:.2f}mm  pitch={p:.1f}×{m}")
    else:
        for i, j, d, p, m in pair_hits[:3]:
            print(f"    pair({i},{j}): d={d:.2f}mm  pitch={p:.1f}×{m}")
        print(f"    ... and {n_pairs - 3} more")

    # Force-walk the strongest qualifying pair.
    amps_tube = np.array([amps_c[i] for i in in_tube_c])
    best = None
    for i, j, d, p, m in pair_hits:
        score = amps_tube[i] + amps_tube[j]
        if best is None or score > best[-1]:
            best = (i, j, d, p, m, score)
    bi, bj, bd, bp, bm, bsc = best
    gi, gj = in_tube_c[bi], in_tube_c[bj]
    h = _walk_one(gi, gj, pts_c, amps_c, bp)
    print(
        f"\n  FORCE-WALK strongest pair: pts_c[{gi}]-pts_c[{gj}] "
        f"(tube idx {bi}-{bj}), d={bd:.2f} mm, pitch={bp}×{bm}, "
        f"amp_sum={bsc:.0f}"
    )
    if h is None:
        print("    → _walk_line returned None (< MIN_BLOBS_PER_LINE inliers).")
    else:
        print(
            f"    → n_blobs={h['n_blobs']} span={h['span_mm']:.2f} mm  "
            f"amp_sum={h['amp_sum']:.0f}  "
            f"(AMP_SUM_MIN={cpfit.AMP_SUM_MIN:.0f}; "
            f"MIN_BLOBS_PER_LINE={cpfit.MIN_BLOBS_PER_LINE})"
        )
        passes_amp = h['amp_sum'] >= cpfit.AMP_SUM_MIN
        passes_min = h['n_blobs'] >= cpfit.MIN_BLOBS_PER_LINE
        print(
            f"    gates: amp_sum {'PASS' if passes_amp else 'FAIL'}, "
            f"n_blobs {'PASS' if passes_min else 'FAIL'}"
        )

    # Now run the FULL stage-1 and check which lines claim our in-tube blobs.
    stage1_lines, _pts = cpfit.run_stage1(
        log1, kji_to_ras, dist_arr, ras2ijk, pitches_mm=pitches,
    )
    tube_set = set(in_tube_c)
    claims = []
    for li, line in enumerate(stage1_lines):
        stolen = [int(x) for x in line["inlier_idx"] if int(x) in tube_set]
        if stolen:
            t_axis = _unit(np.asarray(line["end_ras"]) - np.asarray(line["start_ras"]))
            ang = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(u, t_axis)))))))
            t_mid = 0.5 * (np.asarray(line["start_ras"]) + np.asarray(line["end_ras"]))
            mid = 0.5 * (start + end)
            d_mid = t_mid - mid
            d_mid_perp = float(np.linalg.norm(d_mid - (d_mid @ u) * u))
            claims.append((li, len(stolen), ang, d_mid_perp, int(line["n_blobs"])))
    if claims:
        print(f"\n  stage-1 lines that claim in-tube blobs:")
        for li, n_stolen, ang, mid_d, nb in claims:
            print(
                f"    line[{li}]  claims {n_stolen} in-tube blobs  "
                f"ang={ang:.2f}°  mid_perp={mid_d:.2f}mm  n_blobs={nb}"
            )
    else:
        print("\n  NO stage-1 line claims any in-tube blob → walker doesn't produce "
              "a line on this shank at all.")


def main():
    subjects = sorted({sid for sid, _ in MISS_LIST})
    rows_by_sid = {}
    for row in iter_subject_rows(DATASET_ROOT, set(subjects)):
        rows_by_sid[str(row["subject_id"])] = row

    for sid in subjects:
        row = rows_by_sid.get(sid)
        if row is None:
            print(f"\n[!] {sid} not in manifest")
            continue
        ct_path = row["ct_path"]
        img = sitk.ReadImage(ct_path)
        img_c = sitk.Clamp(img, lowerBound=-1024.0,
                            upperBound=cpfit.HU_CLIP_MAX)
        ijk2ras, ras2ijk = image_ijk_ras_matrices(img_c)
        ijk2ras = np.asarray(ijk2ras, dtype=float)
        ras2ijk = np.asarray(ras2ijk, dtype=float)
        features = gfe.compute_features(img_c, ijk2ras)
        log1 = features["log"]
        dist_arr = features["head_distance"]
        kji_to_ras = cpfit._kji_to_ras_fn_from_matrix(ijk2ras)

        # Rebuild pts_c exactly as run_stage1 does.
        blobs = cpfit.extract_blobs(log1, threshold=cpfit.LOG_BLOB_THRESHOLD)
        pts_ras = np.array([kji_to_ras(b["kji"]) for b in blobs])
        amps_ras = np.array([b["amp"] for b in blobs], dtype=float)
        nvox = np.array([b["n_vox"] for b in blobs], dtype=int)
        contact_mask = nvox <= cpfit.LOG_BLOB_MAX_VOXELS
        pts_c = pts_ras[contact_mask]
        amps_c = amps_ras[contact_mask]
        nvox_c = nvox[contact_mask]

        for sid2, shank in MISS_LIST:
            if sid2 != sid:
                continue
            _investigate(
                sid, shank, row, features, ijk2ras, ras2ijk,
                log1, dist_arr, kji_to_ras,
                pts_c, amps_c, nvox_c,
            )


if __name__ == "__main__":
    main()
