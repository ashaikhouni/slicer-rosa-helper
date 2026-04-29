"""S56 stage-by-stage detection funnel.

Runs contact_pitch_v1 on S56's raw CT and dumps the count at every
pipeline stage so we can see where shanks are being lost (preprocessing
threshold? walker seed pruning? bolt anchor? score gate?).

Also exports LoG blob centers as a CSV the user can load into Slicer
as fiducial markups to visually compare against the missing shanks
they identify in the post-op CT.

Run:
    cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper && \\
        /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_s56_funnel.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper")
sys.path.insert(0, str(ROOT / "CommonLib"))
sys.path.insert(0, str(ROOT / "PostopCTLocalization"))

import numpy as np
import SimpleITK as sitk

from postop_ct_localization import contact_pitch_v1_fit as cpfit
from shank_core.io import image_ijk_ras_matrices

S56_CT = Path("/Users/ammar/Documents/Data/imaging/S56/Post_CT/S56_CT.nii.gz")
OUT_DIR = Path("/tmp/s56_funnel")
OUT_DIR.mkdir(exist_ok=True)


def main():
    img = sitk.ReadImage(str(S56_CT))
    print(f"S56 raw: spacing={img.GetSpacing()}, size={img.GetSize()}")

    log_lines: list[str] = []

    def _logger(msg):
        log_lines.append(str(msg))
        print(f"  [log] {msg}")

    # Replicate the canonical resample manually so we can probe the
    # post-resample voxel array and IJK→RAS matrix the detector sees.
    spacing_in = img.GetSpacing()
    if min(float(s) for s in spacing_in) < cpfit.CANONICAL_SPACING_MM * 0.95:
        size_in = img.GetSize()
        target_size = [
            max(1, int(round(size_in[i] * float(spacing_in[i]) / cpfit.CANONICAL_SPACING_MM)))
            for i in range(3)
        ]
        rs = sitk.ResampleImageFilter()
        rs.SetOutputSpacing((cpfit.CANONICAL_SPACING_MM,) * 3)
        rs.SetSize(target_size)
        rs.SetOutputOrigin(img.GetOrigin())
        rs.SetOutputDirection(img.GetDirection())
        rs.SetInterpolator(sitk.sitkLinear)
        rs.SetDefaultPixelValue(-1024)
        img_canonical = rs.Execute(img)
        if cpfit.RAW_RESAMPLE_GAUSSIAN_SIGMA_MM > 0:
            img_canonical = sitk.SmoothingRecursiveGaussian(
                img_canonical, cpfit.RAW_RESAMPLE_GAUSSIAN_SIGMA_MM,
            )
    else:
        img_canonical = img

    img_canonical = sitk.Clamp(
        img_canonical, lowerBound=-1024.0, upperBound=cpfit.HU_CLIP_MAX,
    )
    print(f"  canonical: spacing={img_canonical.GetSpacing()}, size={img_canonical.GetSize()}")

    ct_arr = sitk.GetArrayFromImage(img_canonical).astype(np.float32)
    ijk2ras_can, ras2ijk_can = image_ijk_ras_matrices(img_canonical)

    # Stage A — preprocessing volumes.
    hull, intracranial, dist_arr = cpfit.build_masks(img_canonical)
    print(f"\n[A] preprocessing")
    print(f"  hull voxels        = {int(hull.sum()):>10d}")
    print(f"  intracranial voxels= {int(intracranial.sum()):>10d}")
    print(f"  dist max (mm)      = {float(dist_arr.max()):.1f}")

    # Stage B — LoG blobs.
    log1 = cpfit.log_sigma(img_canonical, sigma_mm=cpfit.LOG_SIGMA_MM)
    blobs = cpfit.extract_blobs(log1, threshold=cpfit.LOG_BLOB_THRESHOLD)
    print(f"\n[B] LoG σ=1 blob extraction (|LoG| ≥ {cpfit.LOG_BLOB_THRESHOLD})")
    print(f"  total blobs              = {len(blobs):>10d}")

    def _kji_to_ras(kji):
        ijk = np.array([float(kji[2]), float(kji[1]), float(kji[0]), 1.0])
        ras = ijk2ras_can @ ijk
        return ras[:3]

    pts_all = np.array([_kji_to_ras(b["kji"]) for b in blobs]) if blobs else np.empty((0, 3))
    n_vox_all = np.array([b["n_vox"] for b in blobs], dtype=int) if blobs else np.empty((0,), dtype=int)
    contact_mask = n_vox_all <= cpfit.LOG_BLOB_MAX_VOXELS
    pts_contact = pts_all[contact_mask] if len(pts_all) else pts_all
    print(f"  contact-sized (n_vox ≤ {cpfit.LOG_BLOB_MAX_VOXELS}) = {int(contact_mask.sum()):>10d}")

    # Sample dist_arr at each contact-sized blob to count intracranial blobs.
    if len(pts_contact):
        h = np.concatenate([pts_contact, np.ones((len(pts_contact), 1))], axis=1)
        ijk = (ras2ijk_can @ h.T).T[:, :3]
        K, J, I = dist_arr.shape
        ii = np.clip(np.round(ijk[:, 0]).astype(int), 0, I - 1)
        jj = np.clip(np.round(ijk[:, 1]).astype(int), 0, J - 1)
        kk = np.clip(np.round(ijk[:, 2]).astype(int), 0, K - 1)
        depth = dist_arr[kk, jj, ii]
        n_intra = int(np.sum(depth >= cpfit.INTRACRANIAL_MIN_DISTANCE_MM))
    else:
        depth = np.empty((0,), dtype=np.float32)
        n_intra = 0
    print(f"  intracranial (depth ≥ {cpfit.INTRACRANIAL_MIN_DISTANCE_MM} mm) = {n_intra:>10d}")

    # Export blob centers for Slicer overlay.
    fcsv = OUT_DIR / "s56_log_blobs.fcsv"
    with open(fcsv, "w") as f:
        f.write("# Markups fiducial file version = 4.11\n")
        f.write("# CoordinateSystem = LPS\n")
        f.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n")
        for i, p in enumerate(pts_contact):
            # pts_contact is in RAS (kji_to_ras_fn returns RAS); convert to LPS for Slicer fcsv default.
            x_lps, y_lps, z_lps = -float(p[0]), -float(p[1]), float(p[2])
            depth_mm = float(depth[i]) if i < len(depth) else 0.0
            f.write(
                f"vtkMRMLMarkupsFiducialNode_{i},{x_lps:.3f},{y_lps:.3f},{z_lps:.3f},"
                f"0,0,0,1,1,1,0,blob_{i},depth={depth_mm:.1f}mm,\n"
            )
    print(f"  wrote {fcsv} ({len(pts_contact)} blob centers — load in Slicer to overlay)")

    # Stage C — full pipeline run with logger.
    print(f"\n[C] full run_two_stage_detection (auto pitch)")
    trajs, features = cpfit.run_two_stage_detection(
        img, ijk2ras_can, ras2ijk_can,
        return_features=True,
        progress_logger=_logger,
        pitch_strategy="auto",
    )
    print(f"\n  final trajectories: {len(trajs)}")

    # Score-band breakdown.
    bands = {"high": 0, "medium": 0, "low": 0}
    for t in trajs:
        cl = str(t.get("confidence_label", "?"))
        bands[cl] = bands.get(cl, 0) + 1
    print(f"  bands: high={bands['high']}  medium={bands['medium']}  low={bands['low']}")

    # Bolt-source breakdown.
    bs_count: dict[str, int] = {}
    for t in trajs:
        bs = str(t.get("bolt_source", "?"))
        bs_count[bs] = bs_count.get(bs, 0) + 1
    print(f"  bolt sources: {dict(sorted(bs_count.items()))}")

    # Per-trajectory dump sorted by Z.
    print(f"\n  per-trajectory dump (sorted by Z midpoint, descending):")
    rows = []
    for t in trajs:
        s = np.asarray(t["start_ras"], dtype=float)
        e = np.asarray(t["end_ras"], dtype=float)
        mid = 0.5 * (s + e)
        rows.append((mid[2], s, e, mid, t))
    rows.sort(key=lambda r: -r[0])
    for _, s, e, mid, t in rows:
        print(
            f"    conf={float(t.get('confidence', 0)):.2f}({str(t.get('confidence_label', '?'))[:6]:>6s})  "
            f"bolt={str(t.get('bolt_source', '?')):>11s}  "
            f"n={int(t.get('n_inliers', 0)):>2d}  "
            f"pitch={float(t.get('original_median_pitch_mm', 0)):.2f}  "
            f"len={float(np.linalg.norm(e - s)):.1f}  "
            f"mid_RAS=({mid[0]:+.1f},{mid[1]:+.1f},{mid[2]:+.1f})"
        )

    # How many contact blobs ended up as trajectory inliers?
    used_blob_idx = set()
    for t in trajs:
        for i in t.get("inlier_idx", []):
            used_blob_idx.add(int(i))
    n_used = len(used_blob_idx)
    print(f"\n  blob accounting:")
    print(f"    contact-sized blobs in feature cloud = {int(contact_mask.sum())}")
    print(f"    blobs used as trajectory inliers      = {n_used}")
    print(f"    orphan blobs (in cloud, no traj)      = {int(contact_mask.sum()) - n_used}")
    print(f"\n  full progress log captured in {len(log_lines)} lines:")
    for ln in log_lines:
        print(f"    > {ln}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
