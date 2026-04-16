"""Slab-MIP centroid tracker probe.

Walks each bolt inward by snapping to the centroid of bright voxels in
a 5mm-thick × 4mm-radius slab perpendicular to the local axis. The
local axis is recomputed from the recent trail (last ~5mm of centroids)
so the cylinder follows the actual electrode shaft instead of drifting
when bolt PCA is slightly off-shaft.

Outputs:
  - stdout: per-bolt trail length, total bend, gap count
  - NIfTI label map: each bolt's tracked tube (1.5mm radius around the
    centroid polyline) with a unique label index. Load alongside the CT
    in Slicer to compare against /tmp/bolt_mask_T22.nii.gz.

Usage:
  /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
    tests/deep_core/probe_track_bolt.py [SUBJECT_ID] [OUT_NIFTI]

Defaults: SUBJECT_ID=T22, OUT=/tmp/track_mask_<subject>.nii.gz
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


# --- geometry helpers --------------------------------------------------------

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


def _angle_deg(a, b):
    a = _unit(a); b = _unit(b)
    c = float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _ras_batch_to_kji(pts_ras, ras_to_ijk_mat):
    pts = np.asarray(pts_ras, dtype=float).reshape(-1, 3)
    h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    ijk_h = (ras_to_ijk_mat @ h.T).T
    ijk = ijk_h[:, :3]
    kji = np.rint(np.stack([ijk[:, 2], ijk[:, 1], ijk[:, 0]], axis=1)).astype(int)
    return kji


# --- slab sampling -----------------------------------------------------------

def _cluster_bright_in_slab(
    center_ras, axis,
    *, arr_kji, ijk_to_ras_mat, ras_to_ijk_mat,
    slab_radius_mm, slab_half_thickness_mm, hu_floor,
    min_cluster_voxels=3,
):
    """Find connected bright clusters inside a slab perpendicular to axis.

    Builds a KJI subvolume covering the slab, masks voxels inside the slab
    and above hu_floor, runs 3D 26-connected labeling, and returns a list
    of dicts per cluster with centroid_ras, n_voxels, max_hu.
    """
    axis_n = _unit(axis)
    u, v = _perp_frame(axis_n)

    # Conservative KJI bounding box from slab corners.
    corners = []
    for sa in (-slab_half_thickness_mm, slab_half_thickness_mm):
        for su in (-slab_radius_mm, slab_radius_mm):
            for sv in (-slab_radius_mm, slab_radius_mm):
                corners.append(center_ras + sa * axis_n + su * u + sv * v)
    corners_ras = np.stack(corners, axis=0)
    ckji = _ras_batch_to_kji(corners_ras, ras_to_ijk_mat)
    k_lo = int(ckji[:, 0].min()) - 1
    k_hi = int(ckji[:, 0].max()) + 2
    j_lo = int(ckji[:, 1].min()) - 1
    j_hi = int(ckji[:, 1].max()) + 2
    i_lo = int(ckji[:, 2].min()) - 1
    i_hi = int(ckji[:, 2].max()) + 2
    shape = arr_kji.shape
    k_lo = max(0, k_lo); j_lo = max(0, j_lo); i_lo = max(0, i_lo)
    k_hi = min(shape[0], k_hi); j_hi = min(shape[1], j_hi); i_hi = min(shape[2], i_hi)
    if k_lo >= k_hi or j_lo >= j_hi or i_lo >= i_hi:
        return []

    subvol = arr_kji[k_lo:k_hi, j_lo:j_hi, i_lo:i_hi]
    kk = np.arange(k_lo, k_hi)
    jj = np.arange(j_lo, j_hi)
    ii = np.arange(i_lo, i_hi)
    KK, JJ, II = np.meshgrid(kk, jj, ii, indexing="ij")
    ijk_flat = np.stack([II.ravel(), JJ.ravel(), KK.ravel()], axis=1).astype(float)
    h = np.concatenate([ijk_flat, np.ones((ijk_flat.shape[0], 1))], axis=1)
    ras_flat = (ijk_to_ras_mat @ h.T).T[:, :3]
    ras = ras_flat.reshape(KK.shape + (3,))

    rel = ras - center_ras
    axial = np.einsum("...i,i->...", rel, axis_n)
    perp_vec = rel - axial[..., None] * axis_n
    perp_dist = np.linalg.norm(perp_vec, axis=-1)
    in_slab = (np.abs(axial) <= slab_half_thickness_mm) & (perp_dist <= slab_radius_mm)
    bright = in_slab & (subvol >= hu_floor)
    if not bright.any():
        return []

    # 26-connected labeling via flood-fill (no scipy dependency).
    labels = np.zeros(bright.shape, dtype=np.int32)
    n_labels = 0
    dk, dj, di = bright.shape
    for kk_ in range(dk):
        for jj_ in range(dj):
            for ii_ in range(di):
                if bright[kk_, jj_, ii_] and labels[kk_, jj_, ii_] == 0:
                    n_labels += 1
                    stack = [(kk_, jj_, ii_)]
                    labels[kk_, jj_, ii_] = n_labels
                    while stack:
                        ck, cj, ci = stack.pop()
                        for nk in range(max(0, ck - 1), min(dk, ck + 2)):
                            for nj in range(max(0, cj - 1), min(dj, cj + 2)):
                                for ni in range(max(0, ci - 1), min(di, ci + 2)):
                                    if bright[nk, nj, ni] and labels[nk, nj, ni] == 0:
                                        labels[nk, nj, ni] = n_labels
                                        stack.append((nk, nj, ni))

    clusters = []
    for li in range(1, n_labels + 1):
        mask = labels == li
        n_vox = int(mask.sum())
        if n_vox < min_cluster_voxels:
            continue
        cluster_ras = ras[mask]
        cluster_hu = subvol[mask]
        clusters.append({
            "centroid": cluster_ras.mean(axis=0),
            "n_voxels": n_vox,
            "max_hu": float(cluster_hu.max()),
        })
    return clusters


def _schedule_lookup(schedule, depth):
    for max_d, val in schedule:
        if depth <= max_d:
            return val
    return schedule[-1][1]


# --- the tracker ------------------------------------------------------------

def _hd_at(pt_ras, head_distance_map_kji, ras_to_ijk_mat):
    kji = _ras_batch_to_kji(np.asarray(pt_ras).reshape(1, 3), ras_to_ijk_mat)[0]
    shape = head_distance_map_kji.shape
    if not (0 <= kji[0] < shape[0] and 0 <= kji[1] < shape[1] and 0 <= kji[2] < shape[2]):
        return float("nan")
    return float(head_distance_map_kji[kji[0], kji[1], kji[2]])


def track_bolt(
    *, bolt_center_ras, bolt_axis_in,
    arr_kji, ras_to_ijk_mat, ijk_to_ras_mat, head_distance_map_kji,
    step_mm=1.0,
    slab_radius_schedule=((5.0, 20.0), (20.0, 10.0), (float("inf"), 5.0)),
    max_perp_schedule=((5.0, 6.0), (20.0, 4.0), (float("inf"), 2.5)),
    slab_half_thickness_mm=2.5,
    hu_floor=1100.0,
    hu_saturation_floor=1400.0,
    min_cluster_voxels=2,
    max_walk_mm=110.0,
    gap_mm=15.0,
    lever_back_mm=10.0,
    cum_max_deg=12.0,
    axis_freeze_mm=10.0,
    head_distance_min_mm=3.0,
    head_distance_check_after_mm=5.0,
    axial_accept_min_mm=-2.0,
    axial_accept_max_mm=6.0,
    debug=False,
):
    """Walk inward from bolt_center using wide-slab cluster selection.

    At each step: sample a slab perpendicular to the current trail
    direction, find bright connected clusters, filter to clusters within
    max_perp_mm of the predicted axis line, pick the one with minimum
    perpendicular distance (the straightest continuation). Snap the next
    point's lateral component to that cluster.
    """
    n_steps = int(round(max_walk_mm / step_mm))
    gap_steps_max = max(1, int(round(gap_mm / step_mm)))
    lever_back_steps = max(1, int(round(lever_back_mm / step_mm)))

    trail = [np.asarray(bolt_center_ras, dtype=float).copy()]
    axes = [_unit(bolt_axis_in).copy()]
    measured = [True]
    a0 = axes[0].copy()

    gap_run = 0
    deepest_measured_idx = 0
    n_candidates_log = []

    for k in range(1, n_steps + 1):
        t_walked_mm = (k - 1) * step_mm

        if len(trail) > lever_back_steps:
            anchor = trail[-1 - lever_back_steps]
            recent_dir = _unit(trail[-1] - anchor)
        else:
            recent_dir = axes[-1]

        p_pred = trail[-1] + step_mm * recent_dir

        slab_r = _schedule_lookup(slab_radius_schedule, t_walked_mm)
        max_perp = _schedule_lookup(max_perp_schedule, t_walked_mm)

        clusters = _cluster_bright_in_slab(
            p_pred, recent_dir,
            arr_kji=arr_kji,
            ijk_to_ras_mat=ijk_to_ras_mat,
            ras_to_ijk_mat=ras_to_ijk_mat,
            slab_radius_mm=slab_r,
            slab_half_thickness_mm=slab_half_thickness_mm,
            hu_floor=hu_floor,
            min_cluster_voxels=min_cluster_voxels,
        )

        # Filter: saturation floor, axial window, perpendicular distance.
        candidates = []
        for c in clusters:
            if c["max_hu"] < hu_saturation_floor:
                continue
            rel = c["centroid"] - p_pred
            axial = float(rel @ recent_dir)
            if axial < axial_accept_min_mm or axial > axial_accept_max_mm:
                continue
            perp_vec = rel - axial * recent_dir
            perp_dist = float(np.linalg.norm(perp_vec))
            if perp_dist > max_perp:
                continue
            candidates.append({
                "cluster": c,
                "axial": axial,
                "perp_dist": perp_dist,
                "perp_vec": perp_vec,
            })

        n_candidates_log.append(len(candidates))

        best = None
        if candidates:
            # Pick minimum perpendicular distance; tie-break on higher max_hu.
            best = min(
                candidates,
                key=lambda x: (x["perp_dist"], -x["cluster"]["max_hu"]),
            )

            # Head-distance gate once past entry.
            if t_walked_mm >= head_distance_check_after_mm:
                hd = _hd_at(
                    best["cluster"]["centroid"], head_distance_map_kji, ras_to_ijk_mat,
                )
                if not np.isfinite(hd) or hd < head_distance_min_mm:
                    best = None

        # Debug: first 15 steps of debug bolt.
        if debug and k <= 15:
            print(
                f"    step {k:3d} t={t_walked_mm:.0f}mm slab_r={slab_r:.0f} "
                f"nclusters={len(clusters)} ncand={len(candidates)} gap={gap_run}"
            )
            for ci, c in enumerate(candidates):
                tag = "**" if (best is not None and c is best) else "  "
                print(
                    f"      {tag}c{ci}: perp={c['perp_dist']:.2f} axial={c['axial']:.1f} "
                    f"nvox={c['cluster']['n_voxels']} maxhu={c['cluster']['max_hu']:.0f}"
                )

        if best is not None:
            # Advance by step along recent_dir, then apply lateral snap only.
            p_new = p_pred + best["perp_vec"]
            gap_run = 0
            deepest_measured_idx = k
            measured_here = True
        else:
            p_new = p_pred
            measured_here = False
            gap_run += 1
            if gap_run > gap_steps_max:
                break

        trail.append(p_new)
        measured.append(measured_here)

        if len(trail) > lever_back_steps + 1:
            anchor2 = trail[-1 - lever_back_steps]
            new_axis = _unit(p_new - anchor2)
        else:
            new_axis = axes[-1]

        cum_ang = _angle_deg(a0, new_axis)
        if cum_ang > cum_max_deg and cum_ang > 1e-6:
            t = cum_max_deg / cum_ang
            new_axis = _unit((1.0 - t) * a0 + t * new_axis)

        axes.append(new_axis)

    return {
        "trail": np.asarray(trail, dtype=float),
        "axes": np.asarray(axes, dtype=float),
        "measured": np.asarray(measured, dtype=bool),
        "deepest_measured_idx": deepest_measured_idx,
        "deepest_measured_mm": deepest_measured_idx * step_mm,
        "final_axis": axes[-1],
        "total_bend_deg": _angle_deg(a0, axes[-1]),
        "n_candidates_log": n_candidates_log,
    }


# --- mask painting -----------------------------------------------------------

def _paint_segment(label_arr_kji, label, p0, p1, *, radius_mm, ras_to_ijk_mat,
                   axial_step_mm=0.4):
    seg = p1 - p0
    L = float(np.linalg.norm(seg))
    if L < 1e-6:
        n = 1
    else:
        n = max(1, int(round(L / axial_step_mm)))
    a = seg / max(L, 1e-9)
    u, v = _perp_frame(a)
    coords = np.arange(-radius_mm, radius_mm + 0.25, 0.4)
    disc = []
    r2 = radius_mm * radius_mm
    for du in coords:
        for dv in coords:
            if du * du + dv * dv <= r2:
                disc.append(du * u + dv * v)
    disc = np.stack(disc, axis=0)
    shape = label_arr_kji.shape
    for i in range(n + 1):
        t = (i / n) if n > 0 else 0.0
        c = p0 + t * seg
        pts = c[None, :] + disc
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
        label_arr_kji[kji_free[:, 0], kji_free[:, 1], kji_free[:, 2]] = label


def paint_trail(label_arr_kji, label, trail, *, ras_to_ijk_mat, radius_mm=1.5):
    for i in range(len(trail) - 1):
        _paint_segment(
            label_arr_kji, label, trail[i], trail[i + 1],
            radius_mm=radius_mm, ras_to_ijk_mat=ras_to_ijk_mat,
        )


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
        row["ct_path"], run_id=f"track_{subject_id}", config={}, extras={}
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
    print(f"# columns: bolt label trail_len_mm deepest_meas_mm n_measured n_total total_bend_deg")

    bi_debug = 1  # LCMN bolt — enable step-by-step diagnostics
    for bi, bolt in enumerate(bolt_out["candidates"]):
        label = bi + 1
        c0 = np.asarray(bolt.center_ras, dtype=float)
        a0 = _unit(np.asarray(bolt.axis_ras, dtype=float))
        half_span = float(bolt.span_mm) / 2.0 + 1.0

        a_ref, c_ref = _refit_axis_from_tube(
            cloud_ras, a0, c0, half_span_mm=half_span, radius_mm=2.5,
        )
        a_in = _orient_axis_inward(a_ref, c_ref, head_distance_map, ras_to_ijk_fn)

        # Start from the bolt's deep edge so the slab doesn't
        # pick up bolt-metal clusters.
        bolt_deep_edge = c_ref + (float(bolt.span_mm) / 2.0 + 1.0) * a_in
        result = track_bolt(
            bolt_center_ras=bolt_deep_edge,
            bolt_axis_in=a_in,
            arr_kji=arr_kji,
            ras_to_ijk_mat=ras_to_ijk_mat,
            ijk_to_ras_mat=ijk_to_ras_mat,
            head_distance_map_kji=head_distance_map,
            debug=(bi == bi_debug),
        )

        trail = result["trail"]
        n_total = len(trail)
        n_meas = int(result["measured"].sum())
        trail_len = float(np.sum(np.linalg.norm(np.diff(trail, axis=0), axis=1))) if n_total > 1 else 0.0

        print(
            f"  bolt {bi:2d} label={label:2d} trail_len={trail_len:6.1f}mm "
            f"deepest_meas={result['deepest_measured_mm']:5.0f}mm "
            f"measured={n_meas:3d}/{n_total:3d} bend={result['total_bend_deg']:.2f}deg"
        )

        # Paint the bolt-metal tube (so we still see the bolt in Slicer).
        v = cloud_ras - c_ref
        proj = v @ a_in
        perp = v - np.outer(proj, a_in)
        dist = np.linalg.norm(perp, axis=1)
        in_tube = (dist < 2.5) & (proj > -half_span - 1.0) & (proj < half_span + 1.0)
        bolt_kji = _ras_batch_to_kji(cloud_ras[in_tube], ras_to_ijk_mat)
        shape = label_arr.shape
        ok = (
            (bolt_kji[:, 0] >= 0) & (bolt_kji[:, 0] < shape[0])
            & (bolt_kji[:, 1] >= 0) & (bolt_kji[:, 1] < shape[1])
            & (bolt_kji[:, 2] >= 0) & (bolt_kji[:, 2] < shape[2])
        )
        bolt_kji = bolt_kji[ok]
        label_arr[bolt_kji[:, 0], bolt_kji[:, 1], bolt_kji[:, 2]] = label

        # Paint the tracked trail as a thin tube (1.5mm radius) so it's
        # visually distinct from the static cylinder in /tmp/bolt_mask_T22.
        paint_trail(label_arr, label, trail, ras_to_ijk_mat=ras_to_ijk_mat, radius_mm=1.5)

    out_img = sitk.GetImageFromArray(label_arr.astype(np.uint16))
    out_img.CopyInformation(src_img)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out_img, str(out_path))
    print(f"\nwrote {out_path}")


def main():
    subject = sys.argv[1] if len(sys.argv) > 1 else "T22"
    out_path = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/track_mask_{subject}.nii.gz"
    run(subject, out_path)


if __name__ == "__main__":
    main()
