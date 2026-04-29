"""S56 missed-shank funnel against user-provided manual GT.

User confirmed S56 ground truth = 16 shanks (L_2, L_3, X01..X14). Auto
detector finds 14 (all X-shanks). L_2 and L_3 are missed. Both are
short (~34-38 mm) and nearly horizontal (Δz ≤ 1 mm) — temporal-lobe
lateral-entry signature.

This probe answers: at what stage do L_2 / L_3 fall out?

  - Are there enough LoG blobs along the GT line? (preprocessing /
    threshold problem if not)
  - Do those blobs cluster at SEEG pitch (3.5 mm)? (walker seed
    problem if not)
  - Did the walker emit a candidate line that got rejected later?

For each GT shank (missed AND matched), reports:
  - count of contact-sized intracranial blobs within 2 mm of the line
  - their longitudinal positions, LoG amps, NN distances
  - pitch histogram of consecutive-NN distances
  - nearest detected trajectory by midpoint distance + axis angle

Run:
    cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper && \\
        /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_s56_missed_shanks.py
"""
from __future__ import annotations

import csv
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
GT_CSV = Path("/Users/ammar/Dropbox/tmp/S56_final_trajectory_points.csv")

PROXIMITY_MM = 2.5  # max perpendicular distance from GT line to count a blob


def _load_gt():
    """Return {name: (entry_ras, target_ras)} from the manual CSV."""
    rows = list(csv.DictReader(open(GT_CSV)))
    by_name: dict[str, dict[str, np.ndarray]] = {}
    for r in rows:
        name = r["trajectory"]
        kind = r["point_type"]
        ras = np.array([float(r["x_world_ras"]), float(r["y_world_ras"]), float(r["z_world_ras"])])
        by_name.setdefault(name, {})[kind] = ras
    out = {}
    for name, ends in by_name.items():
        if "entry" in ends and "target" in ends:
            out[name] = (ends["entry"], ends["target"])
    return out


def _project_blobs_on_line(blob_ras, entry, target):
    """Return (n_close, longitudinal_pos_mm, perp_dist_mm, idx_close)
    for blobs whose perp dist to the line ≤ PROXIMITY_MM and whose
    longitudinal position is between entry (0) and target (length)."""
    if len(blob_ras) == 0:
        return 0, np.empty((0,)), np.empty((0,)), np.empty((0,), dtype=int)
    axis = target - entry
    L = float(np.linalg.norm(axis))
    if L < 1e-6:
        return 0, np.empty((0,)), np.empty((0,)), np.empty((0,), dtype=int)
    u = axis / L
    rel = blob_ras - entry  # (N, 3)
    long_pos = rel @ u  # (N,)
    perp_vec = rel - np.outer(long_pos, u)
    perp_dist = np.linalg.norm(perp_vec, axis=1)
    keep = (perp_dist <= PROXIMITY_MM) & (long_pos >= -3.0) & (long_pos <= L + 3.0)
    idx = np.where(keep)[0]
    return int(keep.sum()), long_pos[idx], perp_dist[idx], idx


def main():
    img = sitk.ReadImage(str(S56_CT))
    print(f"S56: spacing={img.GetSpacing()}, size={img.GetSize()}")
    gt = _load_gt()
    print(f"GT shanks: {sorted(gt.keys())} ({len(gt)} total)")

    # Run pipeline preprocessing + blob extraction so we can probe directly.
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
        img_can = rs.Execute(img)
        if cpfit.RAW_RESAMPLE_GAUSSIAN_SIGMA_MM > 0:
            img_can = sitk.SmoothingRecursiveGaussian(
                img_can, cpfit.RAW_RESAMPLE_GAUSSIAN_SIGMA_MM,
            )
    else:
        img_can = img
    img_can = sitk.Clamp(img_can, lowerBound=-1024.0, upperBound=cpfit.HU_CLIP_MAX)
    ijk2ras_can, ras2ijk_can = image_ijk_ras_matrices(img_can)

    hull, intra, dist_arr = cpfit.build_masks(img_can)
    log1 = cpfit.log_sigma(img_can, sigma_mm=cpfit.LOG_SIGMA_MM)

    # All contact-sized blobs in RAS, with amps + intracranial filter.
    all_blobs = cpfit.extract_blobs(log1, threshold=cpfit.LOG_BLOB_THRESHOLD)
    pts_ras = np.array(
        [(ijk2ras_can @ np.array([b["kji"][2], b["kji"][1], b["kji"][0], 1.0]))[:3]
         for b in all_blobs]
    ) if all_blobs else np.empty((0, 3))
    n_vox = np.array([b["n_vox"] for b in all_blobs], dtype=int) if all_blobs else np.empty((0,), dtype=int)
    contact_mask = n_vox <= cpfit.LOG_BLOB_MAX_VOXELS
    pts_c = pts_ras[contact_mask]
    # Sample LoG amp at each contact-sized blob.
    if len(pts_c):
        h = np.concatenate([pts_c, np.ones((len(pts_c), 1))], axis=1)
        ijk = (ras2ijk_can @ h.T).T[:, :3]
        K, J, I = log1.shape
        ii = np.clip(np.round(ijk[:, 0]).astype(int), 0, I - 1)
        jj = np.clip(np.round(ijk[:, 1]).astype(int), 0, J - 1)
        kk = np.clip(np.round(ijk[:, 2]).astype(int), 0, K - 1)
        amps = np.abs(log1[kk, jj, ii]).astype(float)
        depths = dist_arr[kk, jj, ii].astype(float)
    else:
        amps = np.empty((0,))
        depths = np.empty((0,))
    print(f"\nblob cloud: {len(pts_c)} contact-sized blobs, "
          f"{int((depths >= cpfit.INTRACRANIAL_MIN_DISTANCE_MM).sum())} intracranial "
          f"(depth ≥ {cpfit.INTRACRANIAL_MIN_DISTANCE_MM} mm)")

    # Run the full detector to get the auto trajectories for matching.
    trajs, _ = cpfit.run_two_stage_detection(
        img, ijk2ras_can, ras2ijk_can,
        return_features=True, pitch_strategy="auto",
    )
    print(f"detector emitted: {len(trajs)} trajectories")

    # Match GT to detector by midpoint distance + axis angle.
    def _unit(v):
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-9 else np.array([0., 0., 1.])

    def _match_one(entry, target):
        gt_axis = _unit(target - entry)
        gt_mid = 0.5 * (entry + target)
        best = None
        for ti, t in enumerate(trajs):
            ts = np.asarray(t["start_ras"], dtype=float)
            te = np.asarray(t["end_ras"], dtype=float)
            t_axis = _unit(te - ts)
            ang = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(gt_axis, t_axis)))))))
            t_mid = 0.5 * (ts + te)
            mid_d = float(np.linalg.norm(t_mid - gt_mid))
            score = ang + mid_d
            if best is None or score < best[0]:
                best = (score, ti, ang, mid_d, t)
        return best

    print(f"\n=== per-GT-shank diagnostic (proximity ≤ {PROXIMITY_MM} mm) ===")
    for name in sorted(gt.keys()):
        entry, target = gt[name]
        L = float(np.linalg.norm(target - entry))
        n_close, longp, perpd, idx = _project_blobs_on_line(pts_c, entry, target)
        amps_close = amps[idx] if len(idx) else np.empty((0,))
        depths_close = depths[idx] if len(idx) else np.empty((0,))
        # NN spacing along axis
        if len(longp) >= 2:
            sorted_long = np.sort(longp)
            nn_diffs = np.diff(sorted_long)
            nn_med = float(np.median(nn_diffs))
            nn_min = float(nn_diffs.min())
            nn_max = float(nn_diffs.max())
        else:
            nn_med = nn_min = nn_max = float("nan")
        # Best detector match
        best = _match_one(entry, target)
        if best is None:
            match_str = "no detector trajectories"
        else:
            score, ti, ang, mid_d, t = best
            tag = "MATCH" if (ang <= 15.0 and mid_d <= 15.0) else "MISS "
            match_str = (
                f"{tag} → traj#{ti} ang={ang:.1f}° mid_d={mid_d:.1f}mm "
                f"pitch={float(t.get('original_median_pitch_mm',0)):.2f} "
                f"n={int(t.get('n_inliers',0))}"
            )
        print(
            f"\n  {name:5s} L={L:5.1f}mm Δz={abs(target[2]-entry[2]):.2f}mm  →  {match_str}"
        )
        print(
            f"    blobs near line: n={n_close}  "
            f"intracranial(d≥{cpfit.INTRACRANIAL_MIN_DISTANCE_MM:.0f}mm)={int((depths_close >= cpfit.INTRACRANIAL_MIN_DISTANCE_MM).sum())}"
        )
        if n_close > 0:
            print(
                f"    long pos range : [{longp.min():+.1f}, {longp.max():+.1f}] mm "
                f"(line length {L:.1f} mm)"
            )
            print(
                f"    LoG amps       : min={amps_close.min():.0f}  med={np.median(amps_close):.0f}  "
                f"max={amps_close.max():.0f}  (LOG_BLOB_THRESHOLD={cpfit.LOG_BLOB_THRESHOLD})"
            )
            print(
                f"    head depth     : min={depths_close.min():.1f}  med={np.median(depths_close):.1f}  "
                f"max={depths_close.max():.1f} mm"
            )
            print(
                f"    NN spacing     : min={nn_min:.2f}  med={nn_med:.2f}  max={nn_max:.2f} mm  "
                f"(library 3.50 mm)"
            )
        else:
            print(f"    NO BLOBS within {PROXIMITY_MM} mm of GT line — preprocessing miss")

    return 0


if __name__ == "__main__":
    sys.exit(main())
