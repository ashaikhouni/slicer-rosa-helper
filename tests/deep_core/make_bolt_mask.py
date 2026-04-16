"""Build a Slicer-loadable label map showing detected bolts + their
forward 5mm-diameter cylindrical axes for a given subject.

Each bolt gets a unique label index (1..N). Both the actual bolt-metal
support voxels AND a 2.5mm-radius cylinder extending +150 mm inward
along the (PCA-refit) bolt axis are painted with that index, so in
Slicer you can see "this bolt + this cylinder belong together".

Usage:
  /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
    tests/deep_core/make_bolt_mask.py [SUBJECT_ID] [OUT_NIFTI]

Defaults: SUBJECT_ID=T22, OUT_NIFTI=/tmp/bolt_mask_<subject>.nii.gz

Load alongside the original CT in Slicer to inspect.
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


# --- geometry (same as the probe) -------------------------------------------

def _unit(v):
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def _perp_frame(axis):
    a = _unit(axis)
    helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _unit(helper - np.dot(helper, a) * a)
    v = _unit(np.cross(a, u))
    return u, v


def _pca_axis(points):
    c = points.mean(axis=0)
    X = points - c
    cov = X.T @ X / max(1, X.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    return _unit(eigvecs[:, int(np.argmax(eigvals))]), c


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


def _orient_axis_inward(axis, center, head_distance_map_kji, ras_to_ijk_fn):
    shape = head_distance_map_kji.shape

    def hd(pt):
        ijk = np.asarray(ras_to_ijk_fn(pt), dtype=float)
        kji = (int(round(ijk[2])), int(round(ijk[1])), int(round(ijk[0])))
        if not (0 <= kji[0] < shape[0] and 0 <= kji[1] < shape[1] and 0 <= kji[2] < shape[2]):
            return float("nan")
        return float(head_distance_map_kji[kji[0], kji[1], kji[2]])

    hp = hd(center + 20.0 * axis)
    hm = hd(center - 20.0 * axis)
    if np.isfinite(hp) and np.isfinite(hm) and hm > hp:
        return -axis
    return axis


# --- cylinder painting -------------------------------------------------------

def _filled_disc_offsets_ras(axis, radius_mm, step_mm=0.4):
    u, v = _perp_frame(axis)
    coords = np.arange(-radius_mm, radius_mm + 0.5 * step_mm, step_mm)
    pts = []
    r2 = radius_mm * radius_mm
    for du in coords:
        for dv in coords:
            if du * du + dv * dv <= r2:
                pts.append(du * u + dv * v)
    return np.stack(pts, axis=0) if pts else np.zeros((1, 3))


def _ras_batch_to_kji(pts_ras, ras_to_ijk_mat):
    """Batched RAS -> nearest-int KJI using the 4x4 matrix directly."""
    pts = np.asarray(pts_ras, dtype=float).reshape(-1, 3)
    h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    ijk_h = (ras_to_ijk_mat @ h.T).T  # (N, 4) but only :3 needed
    ijk = ijk_h[:, :3]
    kji = np.rint(np.stack([ijk[:, 2], ijk[:, 1], ijk[:, 0]], axis=1)).astype(int)
    return kji


def _paint_cylinder(
    label_arr_kji,
    label_value,
    *,
    center_ras,
    axis_ras,
    ras_to_ijk_mat,
    radius_mm,
    t_start_mm,
    t_end_mm,
    axial_step_mm=0.4,
):
    offsets = _filled_disc_offsets_ras(axis_ras, radius_mm, step_mm=0.4)
    shape = label_arr_kji.shape
    n_steps = int(round((t_end_mm - t_start_mm) / axial_step_mm)) + 1
    ts = t_start_mm + np.arange(n_steps) * axial_step_mm
    for t in ts:
        center_pt = center_ras + float(t) * axis_ras
        pts = center_pt[None, :] + offsets  # (M, 3)
        kji = _ras_batch_to_kji(pts, ras_to_ijk_mat)
        ok = (
            (kji[:, 0] >= 0) & (kji[:, 0] < shape[0])
            & (kji[:, 1] >= 0) & (kji[:, 1] < shape[1])
            & (kji[:, 2] >= 0) & (kji[:, 2] < shape[2])
        )
        kji = kji[ok]
        if kji.size == 0:
            continue
        cur = label_arr_kji[kji[:, 0], kji[:, 1], kji[:, 2]]
        free = cur == 0
        if not free.any():
            continue
        kji_free = kji[free]
        label_arr_kji[kji_free[:, 0], kji_free[:, 1], kji_free[:, 2]] = label_value


# --- driver ------------------------------------------------------------------

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
        row["ct_path"], run_id=f"mask_{subject_id}", config={}, extras={}
    )

    pipeline = registry.create_pipeline("deep_core_v2")
    pipeline.run_debug(ctx)
    mask = pipeline._last_mask_output
    bolt_out = pipeline._last_bolt_output

    arr_kji = np.asarray(ctx["arr_kji"], dtype=float)
    ras_to_ijk_fn = ctx["ras_to_ijk_fn"]
    ijk_kji_to_ras_fn = ctx["ijk_kji_to_ras_fn"]
    _ijk_to_ras_mat, ras_to_ijk_mat = image_ijk_ras_matrices(src_img)
    ras_to_ijk_mat = np.asarray(ras_to_ijk_mat, dtype=float)
    bolt_metal_mask = np.asarray(mask["bolt_metal_mask_kji"], dtype=bool)
    head_distance_map = mask["head_distance_map_kji"]

    label_arr = np.zeros(arr_kji.shape, dtype=np.uint16)

    idx = np.argwhere(bolt_metal_mask)
    cloud_ras = np.asarray(ijk_kji_to_ras_fn(idx.astype(float)), dtype=float)

    print(f"# subject={subject_id} bolts={len(bolt_out['candidates'])} bolt_metal_voxels={cloud_ras.shape[0]}")

    for bi, bolt in enumerate(bolt_out["candidates"]):
        label = bi + 1  # 1..N
        c0 = np.asarray(bolt.center_ras, dtype=float)
        a0 = _unit(np.asarray(bolt.axis_ras, dtype=float))
        half_span = float(bolt.span_mm) / 2.0 + 1.0

        a_ref, c_ref = _refit_axis_from_tube(
            cloud_ras, a0, c0, half_span_mm=half_span, radius_mm=2.5,
        )
        a_in = _orient_axis_inward(a_ref, c_ref, head_distance_map, ras_to_ijk_fn)

        # Paint the actual bolt support voxels (the detected metal cloud)
        # within this bolt's tube. Bolt = wider tube to make it visible.
        v = cloud_ras - c_ref
        proj = v @ a_in
        perp = v - np.outer(proj, a_in)
        dist = np.linalg.norm(perp, axis=1)
        in_tube = (dist < 2.5) & (proj > -half_span - 1.0) & (proj < half_span + 1.0)
        bolt_pts_ras = cloud_ras[in_tube]
        bolt_kji = _ras_batch_to_kji(bolt_pts_ras, ras_to_ijk_mat)
        shape = label_arr.shape
        ok = (
            (bolt_kji[:, 0] >= 0) & (bolt_kji[:, 0] < shape[0])
            & (bolt_kji[:, 1] >= 0) & (bolt_kji[:, 1] < shape[1])
            & (bolt_kji[:, 2] >= 0) & (bolt_kji[:, 2] < shape[2])
        )
        bolt_kji = bolt_kji[ok]
        label_arr[bolt_kji[:, 0], bolt_kji[:, 1], bolt_kji[:, 2]] = label

        # Paint the forward cylinder: from t=-half_span (start of bolt deep
        # end) to t=+150 mm inward.
        _paint_cylinder(
            label_arr, label,
            center_ras=c_ref, axis_ras=a_in,
            ras_to_ijk_mat=ras_to_ijk_mat,
            radius_mm=2.5,
            t_start_mm=-half_span,
            t_end_mm=150.0,
        )

        n_painted = int((label_arr == label).sum())
        print(
            f"  bolt {bi:2d} label={label:2d} center={c_ref.round(1).tolist()} "
            f"span={bolt.span_mm:.1f}mm n_voxels_painted={n_painted}"
        )

    # Build SITK image with the source CT geometry.
    # arr_kji is (K, J, I) which matches SITK's GetArrayFromImage convention,
    # so we can pass it directly to GetImageFromArray.
    out_img = sitk.GetImageFromArray(label_arr.astype(np.uint16))
    out_img.CopyInformation(src_img)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out_img, str(out_path))
    print(f"\nwrote {out_path}  shape={label_arr.shape}  unique_labels={sorted(set(np.unique(label_arr).tolist()))}")


def main():
    subject = sys.argv[1] if len(sys.argv) > 1 else "T22"
    out_path = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/bolt_mask_{subject}.nii.gz"
    run(subject, out_path)


if __name__ == "__main__":
    main()
