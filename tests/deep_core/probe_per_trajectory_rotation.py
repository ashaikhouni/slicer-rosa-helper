"""Per-trajectory rotational-invariance probe.

Instead of rotating the whole volume (which has bbox-edge issues),
extract a slab around each candidate trajectory's axis, rotate the
slab, run blob extraction + walker on the slab, and test whether a
chain still emerges. Real shanks survive (contacts are dominant
peaks regardless of voxel orientation); voxel-aliased FPs fail.

Per-trajectory cost ~0.1-0.5 s; total overhead modest.

Run:
    cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper && \\
        /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_per_trajectory_rotation.py
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

from postop_ct_localization import contact_pitch_v1_fit as cpfit
from shank_core.io import image_ijk_ras_matrices
from eval_seeg_localization import (
    iter_subject_rows,
    load_reference_ground_truth_shanks,
)


DATASET_ROOT = Path("/Users/ammar/Dropbox/thalamus_subjects/seeg_localization")

# Slab dimensions in mm relative to the trajectory.
SLAB_PERP_MM = 12.0    # +/- this far perp to axis
SLAB_PAD_MM = 10.0     # pad along axis past start/end
SLAB_VOXEL_MM = 1.0    # canonical 1mm voxels
ROTATION_DEG = 30.0    # rotation about the shank axis


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def _build_orthonormal_frame(axis_unit):
    """Return (axis, perp1, perp2) right-handed orthonormal frame."""
    a = np.asarray(axis_unit, dtype=float)
    a = a / np.linalg.norm(a)
    # Pick the world axis least aligned with `a` for cross-product stability.
    if abs(a[0]) < abs(a[1]) and abs(a[0]) < abs(a[2]):
        seed = np.array([1.0, 0.0, 0.0])
    elif abs(a[1]) < abs(a[2]):
        seed = np.array([0.0, 1.0, 0.0])
    else:
        seed = np.array([0.0, 0.0, 1.0])
    p1 = np.cross(a, seed); p1 = p1 / np.linalg.norm(p1)
    p2 = np.cross(a, p1);   p2 = p2 / np.linalg.norm(p2)
    return a, p1, p2


def _resample_slab_around_trajectory(
    img: sitk.Image,
    start_ras: np.ndarray,
    end_ras: np.ndarray,
    rotation_deg: float = 0.0,
) -> tuple[sitk.Image, np.ndarray, np.ndarray]:
    """Resample CT into a local slab whose axes are aligned with the
    trajectory: axis-0 (i) perpendicular1, axis-1 (j) perpendicular2,
    axis-2 (k) along trajectory.

    Returns (slab_sitk, slab_to_world_4x4_ras, world_to_slab_4x4_ras).

    The local IJK origin sits at the slab's corner (perp1=-PERP, perp2=-PERP,
    along=-PAD). With ``rotation_deg`` non-zero, the perp1/perp2 frame
    rotates about the shank axis by that angle.
    """
    s = np.asarray(start_ras, dtype=float)
    e = np.asarray(end_ras, dtype=float)
    L = float(np.linalg.norm(e - s))
    if L < 1e-3:
        return None, None, None
    axis, p1, p2 = _build_orthonormal_frame((e - s) / L)
    # Apply rotation about the axis to the perpendicular frame.
    if rotation_deg != 0.0:
        a = float(np.deg2rad(rotation_deg))
        c, sn = float(np.cos(a)), float(np.sin(a))
        p1_new = c * p1 + sn * p2
        p2_new = -sn * p1 + c * p2
        p1, p2 = p1_new, p2_new

    # Slab dimensions in voxels.
    nperp = int(round(2 * SLAB_PERP_MM / SLAB_VOXEL_MM)) + 1
    nalong = int(round((L + 2 * SLAB_PAD_MM) / SLAB_VOXEL_MM)) + 1
    size_xyz = (nperp, nperp, nalong)

    # Slab origin in RAS is the corner of the slab.
    slab_origin_ras = (
        s
        - SLAB_PAD_MM * axis
        - SLAB_PERP_MM * p1
        - SLAB_PERP_MM * p2
    )

    # Build slab IJK->RAS matrix: columns are slab axes scaled by spacing.
    slab_ijk_to_ras = np.eye(4)
    slab_ijk_to_ras[:3, 0] = p1 * SLAB_VOXEL_MM
    slab_ijk_to_ras[:3, 1] = p2 * SLAB_VOXEL_MM
    slab_ijk_to_ras[:3, 2] = axis * SLAB_VOXEL_MM
    slab_ijk_to_ras[:3, 3] = slab_origin_ras

    # SITK uses LPS internally. Build the SITK origin/direction in LPS.
    slab_origin_lps = np.array([-slab_origin_ras[0], -slab_origin_ras[1], slab_origin_ras[2]])
    p1_lps = np.array([-p1[0], -p1[1], p1[2]])
    p2_lps = np.array([-p2[0], -p2[1], p2[2]])
    axis_lps = np.array([-axis[0], -axis[1], axis[2]])
    direction_lps = np.column_stack([p1_lps, p2_lps, axis_lps])

    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing((SLAB_VOXEL_MM, SLAB_VOXEL_MM, SLAB_VOXEL_MM))
    rs.SetSize(size_xyz)
    rs.SetOutputOrigin(tuple(float(v) for v in slab_origin_lps))
    rs.SetOutputDirection(tuple(float(v) for v in direction_lps.flatten()))
    rs.SetInterpolator(sitk.sitkLinear)
    rs.SetDefaultPixelValue(-1024)
    slab = rs.Execute(img)

    world_to_slab = np.linalg.inv(slab_ijk_to_ras)
    return slab, slab_ijk_to_ras, world_to_slab


def _walk_chain_in_slab(slab_arr_kji: np.ndarray, slab_ijk_to_ras: np.ndarray,
                         start_ras: np.ndarray, end_ras: np.ndarray,
                         expected_pitch_mm: float) -> dict:
    """Run LoG-blob extraction on the slab and walk a chain along the
    trajectory's axial direction (slab z = trajectory axis). Return
    {'n_inliers', 'span_mm'} or None if no chain.
    """
    import SimpleITK as sitk_local
    slab_sitk = sitk_local.GetImageFromArray(slab_arr_kji)
    slab_sitk.SetSpacing((SLAB_VOXEL_MM, SLAB_VOXEL_MM, SLAB_VOXEL_MM))
    # HU clip + LoG sigma=1, same as production preprocessing.
    slab_clamped = sitk_local.Clamp(slab_sitk, lowerBound=-1024.0, upperBound=cpfit.HU_CLIP_MAX)
    log_sitk = sitk_local.LaplacianRecursiveGaussian(
        sitk_local.Cast(slab_clamped, sitk_local.sitkFloat32), sigma=cpfit.LOG_SIGMA_MM,
    )
    log_arr = sitk_local.GetArrayFromImage(log_sitk)
    blobs = cpfit.extract_blobs(log_arr, threshold=cpfit.LOG_BLOB_THRESHOLD)
    if blobs is None or len(blobs) == 0:
        return None
    # extract_blobs returns list of dicts with 'kji' (k, j, i order),
    # 'amp', 'n_vox'. Convert to IJK then RAS.
    pts_kji = np.array([np.asarray(b["kji"], dtype=float) for b in blobs])
    pts_ijk = np.stack([pts_kji[:, 2], pts_kji[:, 1], pts_kji[:, 0]], axis=1)
    h = np.concatenate([pts_ijk, np.ones((pts_ijk.shape[0], 1))], axis=1)
    pts_ras = (slab_ijk_to_ras @ h.T).T[:, :3]

    # Inlier criterion: blobs within SLAB_PERP_MM/2 of the trajectory axis,
    # within (length + pad)/2 of the trajectory midpoint.
    s = np.asarray(start_ras, dtype=float)
    e = np.asarray(end_ras, dtype=float)
    axis = (e - s) / max(1e-9, float(np.linalg.norm(e - s)))
    diffs = pts_ras - s
    along = diffs @ axis
    perp_vec = diffs - np.outer(along, axis)
    perp = np.linalg.norm(perp_vec, axis=1)
    mask = (perp <= cpfit.PERP_TOL_MM) & (along >= -SLAB_PAD_MM) & (along <= float(np.linalg.norm(e - s)) + SLAB_PAD_MM)
    n_inliers = int(mask.sum())
    if n_inliers < cpfit.MIN_BLOBS_PER_LINE:
        return None
    span = float(along[mask].max() - along[mask].min())
    return {"n_inliers": n_inliers, "span_mm": span, "blob_count": int(pts_ijk.shape[0])}


def main():
    rows = list(iter_subject_rows(DATASET_ROOT, {"T1"}))
    if not rows:
        print("ERROR: T1 not found"); return 1
    row = rows[0]
    gt, _ = load_reference_ground_truth_shanks(row)
    img = sitk.ReadImage(str(row["ct_path"]))
    print(f"T1: {len(gt)} GT shanks, input spacing={img.GetSpacing()}")

    # Run the production pipeline once to get candidate trajectories.
    ijk_to_ras_mat, ras_to_ijk_mat = image_ijk_ras_matrices(img)
    trajs = cpfit.run_two_stage_detection(
        img, ijk_to_ras_mat, ras_to_ijk_mat,
        return_features=False, pitch_strategy="auto",
    )
    print(f"\n=== production pipeline emitted {len(trajs)} trajectories ===")

    # Match to GT for labeling.
    def _greedy(gt_, trajs_):
        pairs=[]
        for gi,g in enumerate(gt_):
            gs=np.asarray(g.start_ras,dtype=float); ge=np.asarray(g.end_ras,dtype=float)
            ga=_unit(ge-gs); gm=0.5*(gs+ge)
            for ti,t in enumerate(trajs_):
                ts=np.asarray(t['start_ras'],dtype=float); te=np.asarray(t['end_ras'],dtype=float)
                ta=_unit(te-ts); tm=0.5*(ts+te)
                ang=float(np.degrees(np.arccos(min(1.0,abs(float(np.dot(ga,ta)))))))
                d=gm-tm; p=d-(d@ta)*ta; mid=float(np.linalg.norm(p))
                if ang<=10 and mid<=8: pairs.append((ang+mid,gi,ti))
        pairs.sort(); used_g=set(); used_t=set(); m={}
        for s,gi,ti in pairs:
            if gi in used_g or ti in used_t: continue
            used_g.add(gi); used_t.add(ti); m[ti]=str(gt_[gi].shank)
        return m
    matched = _greedy(gt, trajs)

    # For each trajectory, run the per-shank rotation test.
    print(f"\n{'idx':>3s} {'kind':>8s} {'name':>10s} {'n_inl_orig':>10s}  {'n_inl_at_0deg':>13s}  {'n_inl_at_30deg':>14s}  {'verdict':>10s}  meta")
    for ti, t in enumerate(trajs):
        kind = "matched" if ti in matched else "orphan"
        name = matched.get(ti, "")
        s_ras = np.asarray(t["start_ras"], dtype=float)
        e_ras = np.asarray(t["end_ras"], dtype=float)
        pitch = float(t.get("original_median_pitch_mm", 3.5))
        n_orig = int(t.get("n_inliers", 0))

        # 0-deg slab as a sanity check (should reproduce the original chain).
        slab0, slab_to_ras_0, _ = _resample_slab_around_trajectory(img, s_ras, e_ras, rotation_deg=0.0)
        slab0_arr = sitk.GetArrayFromImage(slab0)
        chain0 = _walk_chain_in_slab(slab0_arr, slab_to_ras_0, s_ras, e_ras, pitch)
        n0 = chain0["n_inliers"] if chain0 else 0

        # Rotated slab.
        slab_r, slab_to_ras_r, _ = _resample_slab_around_trajectory(img, s_ras, e_ras, rotation_deg=ROTATION_DEG)
        slab_r_arr = sitk.GetArrayFromImage(slab_r)
        chain_r = _walk_chain_in_slab(slab_r_arr, slab_to_ras_r, s_ras, e_ras, pitch)
        n_r = chain_r["n_inliers"] if chain_r else 0

        verdict = "INVARIANT" if (chain_r and n_r >= cpfit.MIN_BLOBS_PER_LINE) else "drops"
        meta = (f"conf={float(t.get('confidence',0)):.2f}({t.get('confidence_label','?'):>6s}) "
                f"bolt={t.get('bolt_source','?'):>10s} pitch={pitch:.2f}")
        print(f"{ti:>3d} {kind:>8s} {name:>10s} {n_orig:>10d}  {n0:>13d}  {n_r:>14d}  {verdict:>10s}  {meta}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
