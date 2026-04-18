"""Diagnostic: understand WHY Probe A and Probe B failed.

Probe A: Why do some bolts miss the span>=15mm AND lam2/lam1<=0.05 filter?
  - Find every σ=2 CC that touches a GT bolt (within 8 mm of GT entry along axis)
  - Print its span, ratio, n_vox, endpoint-to-entry distance, axis-angle
  - See whether the CC is too small, non-linear (ratio), or misaligned

Probe B: Why does the soft-tissue filter destroy T2 coverage?
  - Take the FULL Frangi σ=1 >= 10 cloud (pre-soft-filter). Measure GT tube
    coverage. Confirm RAS conversion is correct for T2.
  - For voxels inside ANY GT tube, print distribution of 5mm-median-HU.
    If it's mostly >80 on T2, the bloom hypothesis is confirmed.
  - Same for T22 for comparison.

Usage:
  /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
    tests/deep_core/probe_diagnose.py [T22|T2] [A|B|AB]
"""
from __future__ import annotations

import csv
import os
import sys
import time
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


def _line_axis_len(start, end):
    d = end - start
    n = float(np.linalg.norm(d))
    return (d / n) if n > 1e-9 else np.array([0.0, 0.0, 1.0]), n


def load_gt(subject_id):
    csv_path = (
        DATASET_ROOT / "contact_label_dataset" / "rosa_helper_import"
        / subject_id / "ROSA_Contacts_final_trajectory_points.csv"
    )
    if csv_path.exists():
        traj = {}
        with open(csv_path) as f:
            r = csv.DictReader(f)
            for row in r:
                name = row["trajectory"]
                pt = np.array(
                    [float(row["x_world_ras"]), float(row["y_world_ras"]),
                     float(row["z_world_ras"])], dtype=float,
                )
                traj.setdefault(name, {})[row["point_type"]] = pt
        out = []
        for name, pair in traj.items():
            if "entry" in pair and "target" in pair:
                L = float(np.linalg.norm(pair["target"] - pair["entry"]))
                out.append((name, pair["entry"], pair["target"], L))
        return out
    from eval_seeg_localization import iter_subject_rows, load_ground_truth_shanks
    rows = iter_subject_rows(DATASET_ROOT, {subject_id})
    if not rows:
        return []
    row = rows[0]
    shanks = load_ground_truth_shanks(row["labels_path"], row.get("shanks_path"))
    return [
        (s.shank, np.asarray(s.start_ras, dtype=float),
         np.asarray(s.end_ras, dtype=float), float(s.span_mm))
        for s in shanks
    ]


def frangi_sigma(img, sigma):
    import SimpleITK as sitk
    sm = sitk.SmoothingRecursiveGaussian(img, sigma=float(sigma))
    ob = sitk.ObjectnessMeasure(sm, objectDimension=1, brightObject=True)
    return sitk.GetArrayFromImage(ob).astype(np.float32)


def kji_to_ras_fn(img):
    from shank_core.io import image_ijk_ras_matrices
    m, _ = image_ijk_ras_matrices(img)
    m = np.asarray(m, dtype=float)
    def _f(kji):
        if kji.ndim == 1:
            i, j, k = float(kji[2]), float(kji[1]), float(kji[0])
            return (m @ np.array([i, j, k, 1.0]))[:3]
        ijk = np.stack([kji[:, 2], kji[:, 1], kji[:, 0]], axis=1)
        h = np.concatenate([ijk, np.ones((ijk.shape[0], 1))], axis=1)
        return (m @ h.T).T[:, :3]
    return _f


def analyze_probeA(subject_id):
    import SimpleITK as sitk
    ct_path = DATASET_ROOT / "post_registered_ct" / f"{subject_id}_post_registered.nii.gz"
    img = sitk.ReadImage(str(ct_path))
    spacing = img.GetSpacing()
    to_ras = kji_to_ras_fn(img)

    print(f"\n# ===== Probe A diagnostic: {subject_id} =====")
    t0 = time.time()
    frangi_s2 = frangi_sigma(img, sigma=2.0)
    print(f"# frangi sigma=2: max={frangi_s2.max():.1f}  "
          f"p99={np.percentile(frangi_s2, 99):.1f}  "
          f"p999={np.percentile(frangi_s2, 99.9):.1f}  "
          f"({time.time()-t0:.1f}s)")

    gt = load_gt(subject_id)

    # Sample each GT entry point: what is Frangi σ=2 response along the
    # first 20 mm of the trajectory (skin to brain)?
    print("\n## Frangi σ=2 profile along first 30 mm of each GT trajectory")
    print(f"{'shank':10s} {'max_f2':>7s} {'mean_f2':>7s} "
          f"{'peak@mm':>8s} {'mm>=20':>7s} {'mm>=40':>7s}")
    from shank_core.io import image_ijk_ras_matrices
    _, ras_to_ijk = image_ijk_ras_matrices(img)
    ras_to_ijk = np.asarray(ras_to_ijk, dtype=float)

    def sample_at_ras(pt_ras):
        h = np.array([pt_ras[0], pt_ras[1], pt_ras[2], 1.0])
        ijk = (ras_to_ijk @ h)[:3]
        # IJK -> KJI for array index
        k, j, i = ijk[2], ijk[1], ijk[0]
        ks, js, ins = frangi_s2.shape
        if not (0 <= k < ks - 1 and 0 <= j < js - 1 and 0 <= i < ins - 1):
            return float("nan")
        k0, j0, i0 = int(k), int(j), int(i)
        return float(frangi_s2[k0, j0, i0])

    gt_profiles = []
    for (name, g_s, g_e, _) in gt:
        axis, gL = _line_axis_len(g_s, g_e)
        # Sample from -5 mm (outside skin) to +30 mm (into brain) relative
        # to GT entry.
        vals = []
        for t in np.arange(-5.0, 30.1, 0.5):
            pt = g_s + t * axis
            vals.append(sample_at_ras(pt))
        vals = np.asarray(vals, dtype=float)
        fin = vals[np.isfinite(vals)]
        if fin.size == 0:
            print(f"{name:10s} -- outside volume")
            continue
        peak_idx = int(np.nanargmax(vals))
        peak_mm = -5.0 + 0.5 * peak_idx
        print(
            f"{name:10s} {fin.max():7.1f} {fin.mean():7.1f} "
            f"{peak_mm:8.1f} {int((fin >= 20).sum()):>7d} "
            f"{int((fin >= 40).sum()):>7d}"
        )
        gt_profiles.append((name, vals))

    # For each missed shank, find the CCs whose centroid lies within a 10mm
    # radius of the GT entry. Print their properties.
    print("\n## At threshold=20: CCs near each GT entry (within 10 mm)")
    thr = 20.0
    bin_arr = (frangi_s2 >= thr)
    bin_img = sitk.GetImageFromArray(bin_arr.astype(np.uint8))
    bin_img.CopyInformation(img)
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)
    cc_img = cc_filter.Execute(bin_img)
    cc_arr = sitk.GetArrayFromImage(cc_img)
    n_cc = int(cc_arr.max())
    print(f"# n_cc={n_cc}")

    sx, sy, sz = float(spacing[0]), float(spacing[1]), float(spacing[2])

    # Precompute per-CC info
    print(f"\n{'shank':10s} {'cc':>5s} {'n_vox':>6s} {'span':>6s} "
          f"{'ratio':>6s} {'ang':>6s} {'d_entry':>8s}")
    for (name, g_s, g_e, _) in gt:
        axis, gL = _line_axis_len(g_s, g_e)
        # Find CCs within 12 mm of entry
        # Instead of iterating labels, take a box around entry in voxel space
        # and grab unique labels.
        # Convert entry to ijk
        h = np.array([g_s[0], g_s[1], g_s[2], 1.0])
        ijk_e = (ras_to_ijk @ h)[:3]
        k_e, j_e, i_e = ijk_e[2], ijk_e[1], ijk_e[0]
        r_vox = (int(np.ceil(14.0 / sz)), int(np.ceil(14.0 / sy)),
                 int(np.ceil(14.0 / sx)))
        K, J, I = cc_arr.shape
        k0 = max(0, int(k_e) - r_vox[0]); k1 = min(K, int(k_e) + r_vox[0] + 1)
        j0 = max(0, int(j_e) - r_vox[1]); j1 = min(J, int(j_e) + r_vox[1] + 1)
        i0 = max(0, int(i_e) - r_vox[2]); i1 = min(I, int(i_e) + r_vox[2] + 1)
        box = cc_arr[k0:k1, j0:j1, i0:i1]
        labs = np.unique(box[box > 0])
        for lab in labs:
            kk, jj, ii = np.where(cc_arr == lab)
            if kk.size < 4:
                continue
            # mm-space PCA
            pts = np.stack([ii * sx, jj * sy, kk * sz], axis=1).astype(np.float64)
            c = pts.mean(axis=0)
            X = pts - c
            cov = X.T @ X / max(1, pts.shape[0] - 1)
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = eigvals[::-1]
            eigvecs = eigvecs[:, ::-1]
            pc1 = eigvecs[:, 0]
            l1, l2 = float(eigvals[0]), float(eigvals[1])
            proj = X @ pc1
            span = float(proj.max() - proj.min())
            ratio = l2 / l1 if l1 > 1e-12 else 1.0
            # Endpoints in RAS
            kji_pts = np.stack([kk, jj, ii], axis=1).astype(np.float64)
            kji_c = kji_pts.mean(axis=0)
            X_kji = kji_pts - kji_c
            _, _, Vt_kji = np.linalg.svd(X_kji, full_matrices=False)
            kji_axis = _unit(Vt_kji[0])
            kji_proj = X_kji @ kji_axis
            kji_lo = kji_c + float(kji_proj.min()) * kji_axis
            kji_hi = kji_c + float(kji_proj.max()) * kji_axis
            p_lo_ras = to_ras(kji_lo)
            p_hi_ras = to_ras(kji_hi)
            ax_ras, _ = _line_axis_len(p_lo_ras, p_hi_ras)
            ang = float(np.degrees(np.arccos(np.clip(abs(np.dot(axis, ax_ras)), 0, 1))))
            d_entry = float(min(np.linalg.norm(p_lo_ras - g_s),
                                np.linalg.norm(p_hi_ras - g_s)))
            tag = ""
            if span >= 15.0 and ratio <= 0.05:
                tag = " KEPT"
            print(
                f"{name:10s} {int(lab):>5d} {int(kk.size):>6d} "
                f"{span:>6.1f} {ratio:>6.3f} {ang:>6.2f} "
                f"{d_entry:>8.2f}{tag}"
            )


def analyze_probeB(subject_id):
    import SimpleITK as sitk
    ct_path = DATASET_ROOT / "post_registered_ct" / f"{subject_id}_post_registered.nii.gz"
    img = sitk.ReadImage(str(ct_path))
    spacing = img.GetSpacing()
    raw_ct = sitk.GetArrayFromImage(img).astype(np.float32)
    to_ras = kji_to_ras_fn(img)

    print(f"\n# ===== Probe B diagnostic: {subject_id} =====")
    print(f"# raw CT: shape={raw_ct.shape}  min={raw_ct.min():.0f}  "
          f"max={raw_ct.max():.0f}  p99={np.percentile(raw_ct, 99):.0f}")

    t0 = time.time()
    frangi_s1 = frangi_sigma(img, sigma=1.0)
    print(f"# frangi sigma=1: max={frangi_s1.max():.1f}  ({time.time()-t0:.1f}s)")

    # Full cloud (pre-soft-filter)
    frangi_mask = frangi_s1 >= 10.0
    kk, jj, ii = np.where(frangi_mask)
    n0 = int(kk.size)
    print(f"# frangi>=10: {n0} voxels")

    pts_kji = np.stack([kk, jj, ii], axis=1).astype(np.float64)
    pts_ras = to_ras(pts_kji)
    print(f"# RAS bbox: "
          f"x=[{pts_ras[:,0].min():.1f},{pts_ras[:,0].max():.1f}] "
          f"y=[{pts_ras[:,1].min():.1f},{pts_ras[:,1].max():.1f}] "
          f"z=[{pts_ras[:,2].min():.1f},{pts_ras[:,2].max():.1f}]")

    gt = load_gt(subject_id)
    all_entries = np.array([g[1] for g in gt])
    print(f"# GT entries bbox: "
          f"x=[{all_entries[:,0].min():.1f},{all_entries[:,0].max():.1f}] "
          f"y=[{all_entries[:,1].min():.1f},{all_entries[:,1].max():.1f}] "
          f"z=[{all_entries[:,2].min():.1f},{all_entries[:,2].max():.1f}]")

    # Compute GT tube membership for full cloud
    any_tube_full = np.zeros(n0, dtype=bool)
    per_shank_full = []
    for (name, g_s, g_e, L) in gt:
        axis, gL = _line_axis_len(g_s, g_e)
        diffs = pts_ras - g_s[None, :]
        proj = diffs @ axis
        perp = diffs - np.outer(proj, axis)
        dist = np.linalg.norm(perp, axis=1)
        in_tube = (proj >= 0.0) & (proj <= gL) & (dist <= 2.0)
        per_shank_full.append((name, int(in_tube.sum()), gL, in_tube))
        any_tube_full = any_tube_full | in_tube
    n_tube_full = int(any_tube_full.sum())
    print(f"# full cloud (pre soft-filter) in GT tubes: {n_tube_full}")
    print(f"{'shank':10s} {'L':>6s} {'n_tube':>7s} {'vox/mm':>7s}")
    for name, n, L, _ in per_shank_full:
        print(f"{name:10s} {L:6.1f} {n:>7d} {(n/max(L,1e-6)):7.2f}")

    if n_tube_full == 0:
        print("!! No Frangi voxels at all in GT tubes. Coord-system issue?")
        return

    # Neighborhood-median distribution for voxels INSIDE GT tubes vs OUTSIDE
    def sphere_offsets(spacing_xyz, radius_mm):
        sx, sy, sz = float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])
        rk = int(np.ceil(radius_mm / sz))
        rj = int(np.ceil(radius_mm / sy))
        ri = int(np.ceil(radius_mm / sx))
        offs = []
        for dk in range(-rk, rk + 1):
            for dj in range(-rj, rj + 1):
                for di in range(-ri, ri + 1):
                    d2 = (dk * sz) ** 2 + (dj * sy) ** 2 + (di * sx) ** 2
                    if d2 <= radius_mm ** 2:
                        offs.append((dk, dj, di))
        return offs

    def neighborhood_median(kk_s, jj_s, ii_s, offs):
        dk = np.array([o[0] for o in offs], dtype=np.int32)
        dj = np.array([o[1] for o in offs], dtype=np.int32)
        di = np.array([o[2] for o in offs], dtype=np.int32)
        K, J, I = raw_ct.shape
        N = kk_s.shape[0]
        out = np.empty(N, dtype=np.float32)
        chunk = 4000
        for start in range(0, N, chunk):
            end = min(N, start + chunk)
            kc = kk_s[start:end][None, :] + dk[:, None]
            jc = jj_s[start:end][None, :] + dj[:, None]
            ic = ii_s[start:end][None, :] + di[:, None]
            valid = (
                (kc >= 0) & (kc < K) & (jc >= 0) & (jc < J) & (ic >= 0) & (ic < I)
            )
            kcC = np.clip(kc, 0, K - 1)
            jcC = np.clip(jc, 0, J - 1)
            icC = np.clip(ic, 0, I - 1)
            vals = raw_ct[kcC, jcC, icC].astype(np.float32)
            vals = np.where(valid, vals, np.nan)
            out[start:end] = np.nanmedian(vals, axis=0)
        return out

    offs = sphere_offsets(spacing, 5.0)
    print(f"\n# 5mm sphere: {len(offs)} voxels")

    # In-tube voxels: subsample up to 5000 for speed
    in_idx = np.where(any_tube_full)[0]
    out_idx = np.where(~any_tube_full)[0]
    if in_idx.size > 5000:
        in_idx = np.random.default_rng(0).choice(in_idx, 5000, replace=False)
    if out_idx.size > 5000:
        out_idx = np.random.default_rng(0).choice(out_idx, 5000, replace=False)

    t0 = time.time()
    med_in = neighborhood_median(kk[in_idx], jj[in_idx], ii[in_idx], offs)
    med_out = neighborhood_median(kk[out_idx], jj[out_idx], ii[out_idx], offs)
    print(f"# computed medians for {in_idx.size} in-tube + "
          f"{out_idx.size} out-of-tube voxels ({time.time()-t0:.1f}s)")

    def desc(arr, label):
        print(
            f"  {label}: n={arr.size} "
            f"min={arr.min():.0f} p5={np.percentile(arr,5):.0f} "
            f"p25={np.percentile(arr,25):.0f} "
            f"med={np.median(arr):.0f} "
            f"p75={np.percentile(arr,75):.0f} "
            f"p95={np.percentile(arr,95):.0f} max={arr.max():.0f}  "
            f"frac[-50,80]={float(((arr>=-50)&(arr<=80)).mean())*100:.1f}%"
        )

    print("\n## neighborhood (5 mm) median HU distribution:")
    desc(med_in, "in GT tubes  ")
    desc(med_out, "out of tubes ")

    # Why are in-tube medians >80 on T2? Let's look at a specific contact.
    print("\n## Sample a specific GT shank in detail (first shank in list):")
    name, g_s, g_e, L = gt[0]
    axis, gL = _line_axis_len(g_s, g_e)
    # Sample along GT axis every 1mm, print raw CT and 5mm neighborhood median
    print(f"# shank={name}  L={L:.1f}")
    print(f"{'mm':>5s} {'raw':>6s} {'n_neigh_>80':>11s} {'median':>7s}")
    from shank_core.io import image_ijk_ras_matrices
    _, ras_to_ijk = image_ijk_ras_matrices(img)
    ras_to_ijk = np.asarray(ras_to_ijk, dtype=float)

    dk = np.array([o[0] for o in offs], dtype=np.int32)
    dj = np.array([o[1] for o in offs], dtype=np.int32)
    di = np.array([o[2] for o in offs], dtype=np.int32)
    K, J, I = raw_ct.shape
    for t in np.arange(0.0, min(L, 40.0) + 0.01, 2.0):
        p = g_s + t * axis
        h = np.array([p[0], p[1], p[2], 1.0])
        ijk = (ras_to_ijk @ h)[:3]
        k0, j0, i0 = int(round(ijk[2])), int(round(ijk[1])), int(round(ijk[0]))
        if not (0 <= k0 < K and 0 <= j0 < J and 0 <= i0 < I):
            continue
        raw_v = float(raw_ct[k0, j0, i0])
        kcC = np.clip(k0 + dk, 0, K - 1)
        jcC = np.clip(j0 + dj, 0, J - 1)
        icC = np.clip(i0 + di, 0, I - 1)
        valid = ((k0 + dk >= 0) & (k0 + dk < K) & (j0 + dj >= 0) & (j0 + dj < J)
                 & (i0 + di >= 0) & (i0 + di < I))
        vals = raw_ct[kcC, jcC, icC].astype(np.float32)
        vals = vals[valid]
        n_hi = int((vals > 80).sum())
        med = float(np.median(vals))
        print(f"{t:>5.1f} {raw_v:>6.0f} {n_hi:>11d} {med:>7.0f}")


def main(subject_id, which):
    if "A" in which:
        analyze_probeA(subject_id)
    if "B" in which:
        analyze_probeB(subject_id)


if __name__ == "__main__":
    subj = sys.argv[1] if len(sys.argv) > 1 else "T22"
    which = sys.argv[2] if len(sys.argv) > 2 else "AB"
    main(subj, which)
