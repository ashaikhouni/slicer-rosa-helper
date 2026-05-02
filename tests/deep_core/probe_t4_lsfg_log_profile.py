"""Dump the LoG σ=1 blobs (as extracted by extract_blobs — 3D local
minima thresholded at LOG_BLOB_THRESHOLD) that lie within a tube around
each of T4's redone-GT trajectories. This is what the walker actually
sees.

For each GT shank we report:
- number of blobs inside a 2 mm tube around the axis
- their axial positions (depth along the axis), LoG amplitudes, voxel
  sizes, and spacings between consecutive axial-positions
- whether spacings cluster around any single pitch

Wire-class signature should be either:
  (a) very few blobs (no local minima along the metal)
  (b) many blobs with irregular spacing and weak amplitudes, or
  (c) regular-looking spacing that's an aliasing artifact of the voxel
      grid (detectable by unusually tight variance and spacing
      matching half-voxel or voxel multiples)

Run
---
    /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_t4_lsfg_log_profile.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper")
sys.path.insert(0, str(ROOT / "CommonLib"))
sys.path.insert(0, str(ROOT / "PostopCTLocalization"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import SimpleITK as sitk

from rosa_detect import contact_pitch_v1_fit as cpfit
from eval_seeg_localization import iter_subject_rows, image_ijk_ras_matrices

DATASET_ROOT = Path("/Users/ammar/Dropbox/thalamus_subjects/seeg_localization")
NEW_GT_CSV = (
    DATASET_ROOT / "contact_label_dataset" / "rosa_helper_import" / "T4"
    / "ROSA_Contacts_T4_final_trajectory_points.csv"
)
TUBE_RADIUS_MM = 2.0


def _load_new_gt():
    by = defaultdict(dict)
    with NEW_GT_CSV.open() as f:
        for r in csv.DictReader(f):
            pt = np.array(
                [float(r["x_world_ras"]), float(r["y_world_ras"]),
                 float(r["z_world_ras"])], dtype=float,
            )
            by[r["trajectory"]][r["point_type"]] = pt
    return {s: (d["entry"], d["target"])
            for s, d in by.items() if "entry" in d and "target" in d}


def _axis(a, b):
    d = b - a
    L = float(np.linalg.norm(d))
    return d / L, L


def _blobs_in_tube(blob_pts_ras, blob_amps, blob_nvox, start, end, radius_mm):
    """Return blobs within ``radius_mm`` perpendicular distance of the axis
    AND between start and end inclusive (axial projection).
    Returns: (axial_depth[:], amps[:], nvox[:]) sorted by depth.
    """
    u, L = _axis(start, end)
    v = blob_pts_ras - start[None, :]
    axial = v @ u
    perp = v - axial[:, None] * u[None, :]
    perp_d = np.linalg.norm(perp, axis=1)
    within = (axial >= -2.0) & (axial <= L + 2.0) & (perp_d <= radius_mm)
    axs = axial[within]; amp = blob_amps[within]; nvx = blob_nvox[within]
    order = np.argsort(axs)
    return axs[order], amp[order], nvx[order], perp_d[within][order]


def _spacing_stats(depths, amps, strong_amp_min):
    """Consider only 'strong' blobs (amp ≥ strong_amp_min) for spacing — these
    are the ones the walker locks onto."""
    mask = amps >= strong_amp_min
    strong_d = depths[mask]
    if strong_d.size < 2:
        return None
    dedup_d = []
    for d in strong_d:
        if not dedup_d or abs(d - dedup_d[-1]) > 1.2:
            dedup_d.append(float(d))
    if len(dedup_d) < 2:
        return None
    sp = np.diff(np.array(dedup_d))
    return dict(
        n_strong=int(mask.sum()),
        n_dedup=len(dedup_d),
        spacings=sp.tolist(),
        med=float(np.median(sp)),
        mean=float(np.mean(sp)),
        std=float(np.std(sp)),
        min=float(np.min(sp)),
        max=float(np.max(sp)),
    )


def main():
    rows = iter_subject_rows(DATASET_ROOT, {"T4"})
    ct_path = rows[0]["ct_path"]

    gts = _load_new_gt()
    print(f"T4 GT trajectories: {len(gts)}")

    img = sitk.ReadImage(ct_path)
    img_clamp = sitk.Clamp(img, lowerBound=-1024.0, upperBound=cpfit.HU_CLIP_MAX)
    ijk_to_ras_mat, ras_to_ijk_mat = image_ijk_ras_matrices(img_clamp)
    ijk_to_ras_mat = np.asarray(ijk_to_ras_mat, dtype=float)
    ras_to_ijk_mat = np.asarray(ras_to_ijk_mat, dtype=float)
    kji_to_ras_fn = cpfit._kji_to_ras_fn_from_matrix(ijk_to_ras_mat)

    log_kji = cpfit.log_sigma(img_clamp, sigma_mm=cpfit.LOG_SIGMA_MM)
    print(f"CT spacing: {img.GetSpacing()}  shape: {log_kji.shape}")
    print(f"LoG range: [{log_kji.min():.1f}, {log_kji.max():.1f}]  "
          f"LOG_BLOB_THRESHOLD={cpfit.LOG_BLOB_THRESHOLD}  "
          f"LOG_BLOB_MAX_VOXELS={cpfit.LOG_BLOB_MAX_VOXELS}")

    blobs = cpfit.extract_blobs(log_kji, threshold=cpfit.LOG_BLOB_THRESHOLD)
    print(f"extract_blobs: {len(blobs)} raw blobs total")
    if not blobs:
        return

    pts_ras = np.array([kji_to_ras_fn(b["kji"]) for b in blobs])
    amps = np.array([b["amp"] for b in blobs], dtype=float)
    nvox = np.array([b["n_vox"] for b in blobs], dtype=int)

    # Contact-sized blob filter — same one the walker applies.
    contact_mask = nvox <= cpfit.LOG_BLOB_MAX_VOXELS
    pts_c = pts_ras[contact_mask]
    amps_c = amps[contact_mask]
    nvox_c = nvox[contact_mask]
    print(
        f"Contact-sized subset (n_vox ≤ {cpfit.LOG_BLOB_MAX_VOXELS}): "
        f"{pts_c.shape[0]}"
    )

    # ---- Per-shank summary --------------------------------
    STRONG = 500.0  # amp ≥ 500 is walker's "strong" territory
    print(
        f"\n{'='*110}\nPer-shank blob summary (tube r={TUBE_RADIUS_MM}mm). "
        f"'strong' = |LoG| ≥ {STRONG}\n{'='*110}"
    )
    hdr = (
        f"{'shank':6s} {'len':>5s} {'n_tot':>5s} {'n_str':>5s} "
        f"{'amp_p50':>7s} {'amp_p90':>7s} {'n_dedup':>7s} "
        f"{'sp_med':>6s} {'sp_mean':>7s} {'sp_std':>6s} {'sp_min':>6s} {'sp_max':>6s}"
    )
    print(hdr)
    details = {}
    for name in sorted(gts.keys()):
        s, e = gts[name]
        _, L = _axis(s, e)
        depths, amp_in, nv_in, perp_in = _blobs_in_tube(
            pts_c, amps_c, nvox_c, s, e, TUBE_RADIUS_MM,
        )
        st = _spacing_stats(depths, amp_in, STRONG)
        details[name] = dict(depths=depths, amps=amp_in, nvox=nv_in,
                              perp=perp_in, stats=st, L=L, start=s, end=e)
        if amp_in.size:
            p50 = float(np.percentile(amp_in, 50))
            p90 = float(np.percentile(amp_in, 90))
        else:
            p50 = p90 = float("nan")
        n_strong = int((amp_in >= STRONG).sum())
        if st:
            print(
                f"{name:6s} {L:5.1f} {amp_in.size:5d} {n_strong:5d} "
                f"{p50:7.1f} {p90:7.1f} {st['n_dedup']:7d} "
                f"{st['med']:6.2f} {st['mean']:7.2f} {st['std']:6.2f} "
                f"{st['min']:6.2f} {st['max']:6.2f}"
            )
        else:
            print(
                f"{name:6s} {L:5.1f} {amp_in.size:5d} {n_strong:5d} "
                f"{p50:7.1f} {p90:7.1f} {'-':>7s} "
                f"{'-':>6s} {'-':>7s} {'-':>6s} {'-':>6s} {'-':>6s}"
            )

    # ---- Flag ------------------------------
    print(
        f"\n{'='*110}\nWire-class heuristic: few blobs OR low amplitudes OR "
        f"irregular spacing\n{'='*110}"
    )
    print(
        f"{'shank':6s} {'n_str':>5s} {'amp_p50':>7s} {'sp_std':>6s} {'sp_cv':>6s}  flag"
    )
    for name in sorted(gts.keys()):
        d = details[name]
        amp = d["amps"]
        st = d["stats"]
        p50 = float(np.percentile(amp, 50)) if amp.size else 0.0
        n_str = int((amp >= STRONG).sum())
        sp_std = st["std"] if st else float("nan")
        sp_cv = (sp_std / st["mean"]) if st and st["mean"] > 0 else float("nan")
        flag = ""
        if n_str <= 4: flag += "FEW_STRONG "
        if p50 < 400:  flag += "WEAK_MEDIAN "
        if st and sp_std > 1.0:   flag += "IRREGULAR "
        if st and (st["min"] < 2.5 or st["max"] > 6.0): flag += "OFF_PITCH "
        print(
            f"{name:6s} {n_str:>5d} {p50:>7.1f} "
            f"{sp_std:>6.2f} {sp_cv:>6.2f}  {flag}"
        )

    # ---- Focus: LSFG full blob listing -----------
    print(f"\n{'='*110}\nLSFG blob listing\n{'='*110}")
    d = details["LSFG"]
    print(
        f"  {'idx':>3s} {'depth':>7s} {'|LoG|':>8s} {'n_vox':>5s} {'perp':>6s}"
    )
    for i in range(d["depths"].size):
        print(
            f"  {i:>3d} {d['depths'][i]:>7.2f} {d['amps'][i]:>8.1f} "
            f"{int(d['nvox'][i]):>5d} {d['perp'][i]:>6.2f}"
        )
    if d["stats"]:
        print(
            f"\n  strong-blob spacings (after 1.2mm dedup): "
            f"{[round(x, 2) for x in d['stats']['spacings']]}"
        )

    # ---- Compare: full listing for a clean contact shank as baseline --
    for baseline in ("LMFG", "RMFG", "LCMN"):
        if baseline not in details:
            continue
        print(f"\n{'='*110}\n{baseline} blob listing (baseline)\n{'='*110}")
        d = details[baseline]
        print(
            f"  {'idx':>3s} {'depth':>7s} {'|LoG|':>8s} {'n_vox':>5s} {'perp':>6s}"
        )
        for i in range(d["depths"].size):
            print(
                f"  {i:>3d} {d['depths'][i]:>7.2f} {d['amps'][i]:>8.1f} "
                f"{int(d['nvox'][i]):>5d} {d['perp'][i]:>6.2f}"
            )
        if d["stats"]:
            print(
                f"\n  strong-blob spacings: "
                f"{[round(x, 2) for x in d['stats']['spacings']]}"
            )
        break


if __name__ == "__main__":
    main()
