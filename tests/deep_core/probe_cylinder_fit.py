"""Probe: gather bright intracranial voxels in a wide cylinder around each
bolt axis, RANSAC-fit a line through them, and output a label mask.

No step-by-step tracking. Just: collect bright voxels → fit line → done.

Usage:
  /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
    tests/deep_core/probe_cylinder_fit.py [SUBJECT_ID] [OUT_NIFTI]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))
sys.path.insert(0, str(REPO_ROOT / "PostopCTLocalization"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

DATASET_ROOT = Path(
    os.environ.get(
        "ROSA_SEEG_DATASET",
        "/Users/ammar/Dropbox/thalamus_subjects/seeg_localization",
    )
)


def _unit(v):
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def _pca_axis(points):
    c = points.mean(axis=0)
    X = points - c
    cov = X.T @ X / max(1, X.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    return _unit(eigvecs[:, int(np.argmax(eigvals))]), c


def _angle_deg(a, b):
    a = _unit(a); b = _unit(b)
    c = float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _ras_batch_to_kji(pts_ras, ras_to_ijk_mat):
    pts = np.asarray(pts_ras, dtype=float).reshape(-1, 3)
    h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    ijk_h = (ras_to_ijk_mat @ h.T).T
    return np.rint(np.stack([ijk_h[:, 2], ijk_h[:, 1], ijk_h[:, 0]], axis=1)).astype(int)


def _orient_axis_inward(axis, center, hd_map, ras_to_ijk_mat):
    def hd(pt):
        kji = _ras_batch_to_kji(pt.reshape(1, 3), ras_to_ijk_mat)[0]
        s = hd_map.shape
        if not (0 <= kji[0] < s[0] and 0 <= kji[1] < s[1] and 0 <= kji[2] < s[2]):
            return float("nan")
        return float(hd_map[kji[0], kji[1], kji[2]])
    hp = hd(center + 20.0 * axis)
    hm = hd(center - 20.0 * axis)
    if np.isfinite(hp) and np.isfinite(hm) and hm > hp:
        return -axis
    return axis


def _refit_axis_from_tube(cloud_ras, axis0, center0, *, half_span_mm, radius_mm):
    v = cloud_ras - center0
    proj = v @ axis0
    perp = v - np.outer(proj, axis0)
    dist = np.linalg.norm(perp, axis=1)
    keep = (dist < radius_mm) & (proj > -half_span_mm) & (proj < half_span_mm)
    if int(keep.sum()) < 8:
        return axis0, center0
    axis_ref, center_ref = _pca_axis(cloud_ras[keep])
    if np.dot(axis_ref, axis0) < 0:
        axis_ref = -axis_ref
    return axis_ref, center_ref


def ransac_line_fit(points, *, n_iter=2000, inlier_tol_mm=2.0, min_inliers=5,
                    rng_seed=42, prior_axis=None, max_angle_from_prior_deg=15.0):
    """RANSAC line fit on (N,3) points. Returns (axis, center, inlier_mask).

    If prior_axis is given, only accept candidate lines within
    max_angle_from_prior_deg of the prior direction.
    """
    n = points.shape[0]
    if n < 2:
        return _unit(np.array([0, 0, 1])), points.mean(axis=0), np.ones(n, dtype=bool)
    rng = np.random.default_rng(rng_seed)
    best_n = 0
    best_mask = np.zeros(n, dtype=bool)
    best_axis = np.array([0.0, 0.0, 1.0])
    cos_limit = float(np.cos(np.radians(max_angle_from_prior_deg))) if prior_axis is not None else 0.0
    for _ in range(n_iter):
        i, j = rng.choice(n, size=2, replace=False)
        d = points[j] - points[i]
        L = float(np.linalg.norm(d))
        if L < 1.0:
            continue
        axis = d / L
        if prior_axis is not None:
            if abs(float(np.dot(axis, prior_axis))) < cos_limit:
                continue
        v = points - points[i]
        proj = v @ axis
        perp = v - np.outer(proj, axis)
        dist = np.linalg.norm(perp, axis=1)
        mask = dist < inlier_tol_mm
        n_in = int(mask.sum())
        if n_in > best_n:
            best_n = n_in
            best_mask = mask
            best_axis = axis
    if best_n >= min_inliers:
        inlier_pts = points[best_mask]
        best_axis, _ = _pca_axis(inlier_pts)
    center = points[best_mask].mean(axis=0) if best_n > 0 else points.mean(axis=0)
    return best_axis, center, best_mask


def cylinder_fit_bolt(
    *, bolt_center_ras, bolt_axis_in,
    arr_kji, ijk_to_ras_mat, ras_to_ijk_mat, head_distance_map_kji,
    cylinder_radius_mm=10.0,
    cylinder_depth_mm=150.0,
    hu_floor=1000.0,
    head_distance_min_mm=3.0,
    ransac_inlier_tol_mm=2.0,
):
    """Gather bright intracranial voxels in a wide cylinder along the bolt
    axis, RANSAC-fit a line, return axis + deep tip + inlier voxels."""
    shape = arr_kji.shape
    axis = _unit(bolt_axis_in)

    # Build bounding box in KJI for the cylinder.
    from itertools import product
    u = np.array([1, 0, 0], dtype=float)
    if abs(np.dot(axis, u)) > 0.9:
        u = np.array([0, 1, 0], dtype=float)
    corners = []
    for t in (0.0, cylinder_depth_mm):
        for du in (-cylinder_radius_mm, cylinder_radius_mm):
            for dv in (-cylinder_radius_mm, cylinder_radius_mm):
                p = bolt_center_ras + t * axis + du * u + dv * np.cross(axis, u)
                corners.append(p)
    corners = np.stack(corners)
    ckji = _ras_batch_to_kji(corners, ras_to_ijk_mat)
    k_lo = max(0, int(ckji[:, 0].min()) - 2)
    k_hi = min(shape[0], int(ckji[:, 0].max()) + 3)
    j_lo = max(0, int(ckji[:, 1].min()) - 2)
    j_hi = min(shape[1], int(ckji[:, 1].max()) + 3)
    i_lo = max(0, int(ckji[:, 2].min()) - 2)
    i_hi = min(shape[2], int(ckji[:, 2].max()) + 3)

    # Extract subvolume.
    subvol = arr_kji[k_lo:k_hi, j_lo:j_hi, i_lo:i_hi]
    hd_sub = head_distance_map_kji[k_lo:k_hi, j_lo:j_hi, i_lo:i_hi]

    # Find bright + intracranial voxels.
    bright_intra = (subvol >= hu_floor) & (hd_sub >= head_distance_min_mm)
    if not bright_intra.any():
        return {"n_bright": 0, "axis": axis, "center": bolt_center_ras,
                "deep_tip_ras": bolt_center_ras, "deep_mm": 0.0,
                "n_inliers": 0, "inlier_kji": np.zeros((0, 3), dtype=int)}

    # Convert bright voxels to RAS.
    kk, jj, ii = np.where(bright_intra)
    kk_abs = kk + k_lo
    jj_abs = jj + j_lo
    ii_abs = ii + i_lo
    ijk_flat = np.stack([ii_abs, jj_abs, kk_abs], axis=1).astype(float)
    h = np.concatenate([ijk_flat, np.ones((ijk_flat.shape[0], 1))], axis=1)
    ras_pts = (ijk_to_ras_mat @ h.T).T[:, :3]

    # Filter to within cylinder radius of bolt axis.
    v = ras_pts - bolt_center_ras
    proj = v @ axis
    perp = v - np.outer(proj, axis)
    perp_dist = np.linalg.norm(perp, axis=1)
    in_cyl = (perp_dist <= cylinder_radius_mm) & (proj >= -5.0) & (proj <= cylinder_depth_mm)
    cyl_pts = ras_pts[in_cyl]
    cyl_kji_abs = np.stack([kk_abs[in_cyl], jj_abs[in_cyl], ii_abs[in_cyl]], axis=1)

    if cyl_pts.shape[0] < 3:
        return {"n_bright": int(cyl_pts.shape[0]), "axis": axis,
                "center": bolt_center_ras, "deep_tip_ras": bolt_center_ras,
                "deep_mm": 0.0, "n_inliers": 0,
                "inlier_kji": np.zeros((0, 3), dtype=int)}

    # RANSAC line fit, constrained to within 15° of bolt axis.
    fit_axis, fit_center, inlier_mask = ransac_line_fit(
        cyl_pts, inlier_tol_mm=ransac_inlier_tol_mm, min_inliers=5,
        prior_axis=axis, max_angle_from_prior_deg=15.0,
    )

    # Orient fit_axis same as bolt axis (inward).
    if np.dot(fit_axis, axis) < 0:
        fit_axis = -fit_axis

    # Deep tip = deepest inlier projection along fit_axis from bolt center.
    inlier_pts = cyl_pts[inlier_mask]
    inlier_kji = cyl_kji_abs[inlier_mask]
    if inlier_pts.shape[0] > 0:
        proj_in = (inlier_pts - bolt_center_ras) @ fit_axis
        deep_idx = int(np.argmax(proj_in))
        deep_mm = float(proj_in[deep_idx])
        deep_tip = inlier_pts[deep_idx]
    else:
        deep_mm = 0.0
        deep_tip = bolt_center_ras

    return {
        "n_bright": int(cyl_pts.shape[0]),
        "axis": fit_axis,
        "center": fit_center,
        "deep_tip_ras": deep_tip,
        "deep_mm": deep_mm,
        "n_inliers": int(inlier_mask.sum()),
        "inlier_kji": inlier_kji,
        "all_cyl_kji": cyl_kji_abs,
    }


def run(subject_id, out_path):
    import SimpleITK as sitk
    from shank_engine import PipelineRegistry, register_builtin_pipelines
    from shank_core.io import image_ijk_ras_matrices
    from eval_seeg_localization import build_detection_context, iter_subject_rows

    rows = iter_subject_rows(DATASET_ROOT, {subject_id})
    if not rows:
        raise SystemExit(f"subject {subject_id} not in manifest")
    row = rows[0]
    registry = PipelineRegistry()
    register_builtin_pipelines(registry)
    ctx, src_img = build_detection_context(
        row["ct_path"], run_id=f"cylfit_{subject_id}", config={}, extras={}
    )
    pipeline = registry.create_pipeline("deep_core_v2")
    pipeline.run_debug(ctx)
    mask = pipeline._last_mask_output
    bolt_out = pipeline._last_bolt_output

    arr_kji = np.asarray(ctx["arr_kji"], dtype=float)
    ras_to_ijk_fn = ctx["ras_to_ijk_fn"]
    ijk_kji_to_ras_fn = ctx["ijk_kji_to_ras_fn"]
    bolt_metal_mask = np.asarray(mask["bolt_metal_mask_kji"], dtype=bool)
    head_distance_map = mask["head_distance_map_kji"]

    ijk_to_ras_mat, ras_to_ijk_mat = image_ijk_ras_matrices(src_img)
    ijk_to_ras_mat = np.asarray(ijk_to_ras_mat, dtype=float)
    ras_to_ijk_mat = np.asarray(ras_to_ijk_mat, dtype=float)

    idx = np.argwhere(bolt_metal_mask)
    cloud_ras = np.asarray(ijk_kji_to_ras_fn(idx.astype(float)), dtype=float)

    label_arr = np.zeros(arr_kji.shape, dtype=np.uint16)

    print(f"# subject={subject_id} bolts={len(bolt_out['candidates'])}")
    print(f"# bolt label n_bright n_inliers deep_mm axis_vs_bolt_deg")

    for bi, bolt in enumerate(bolt_out["candidates"]):
        label = bi + 1
        c0 = np.asarray(bolt.center_ras, dtype=float)
        a0 = _unit(np.asarray(bolt.axis_ras, dtype=float))
        half_span = float(bolt.span_mm) / 2.0 + 1.0
        a_ref, c_ref = _refit_axis_from_tube(
            cloud_ras, a0, c0, half_span_mm=half_span, radius_mm=2.5,
        )
        a_in = _orient_axis_inward(a_ref, c_ref, head_distance_map, ras_to_ijk_mat)

        result = cylinder_fit_bolt(
            bolt_center_ras=c_ref,
            bolt_axis_in=a_in,
            arr_kji=arr_kji,
            ijk_to_ras_mat=ijk_to_ras_mat,
            ras_to_ijk_mat=ras_to_ijk_mat,
            head_distance_map_kji=head_distance_map,
        )

        ang = _angle_deg(a_in, result["axis"])
        print(
            f"  bolt {bi:2d} label={label:2d} n_bright={result['n_bright']:5d} "
            f"n_inliers={result['n_inliers']:4d} deep={result['deep_mm']:5.1f}mm "
            f"axis_delta={ang:.1f}deg"
        )

        # Paint bolt support.
        v = cloud_ras - c_ref
        proj = v @ a_in
        perp = v - np.outer(proj, a_in)
        dist = np.linalg.norm(perp, axis=1)
        in_tube = (dist < 2.5) & (proj > -half_span - 1.0) & (proj < half_span + 1.0)
        bolt_kji = _ras_batch_to_kji(cloud_ras[in_tube], ras_to_ijk_mat)
        s = label_arr.shape
        ok = (bolt_kji[:, 0] >= 0) & (bolt_kji[:, 0] < s[0]) & \
             (bolt_kji[:, 1] >= 0) & (bolt_kji[:, 1] < s[1]) & \
             (bolt_kji[:, 2] >= 0) & (bolt_kji[:, 2] < s[2])
        bk = bolt_kji[ok]
        label_arr[bk[:, 0], bk[:, 1], bk[:, 2]] = label

        # Paint RANSAC inlier voxels.
        if result["inlier_kji"].shape[0] > 0:
            ik = result["inlier_kji"]
            ok2 = (ik[:, 0] >= 0) & (ik[:, 0] < s[0]) & \
                  (ik[:, 1] >= 0) & (ik[:, 1] < s[1]) & \
                  (ik[:, 2] >= 0) & (ik[:, 2] < s[2])
            ik = ik[ok2]
            label_arr[ik[:, 0], ik[:, 1], ik[:, 2]] = label

    out_img = sitk.GetImageFromArray(label_arr.astype(np.uint16))
    out_img.CopyInformation(src_img)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out_img, str(out_path))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    subject = sys.argv[1] if len(sys.argv) > 1 else "T22"
    out_path = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/cylfit_mask_{subject}.nii.gz"
    run(subject, out_path)
