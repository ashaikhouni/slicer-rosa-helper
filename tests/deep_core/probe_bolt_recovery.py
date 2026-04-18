"""Probe A: bolt recovery via Frangi sigma=2 + PCA linearity.

Goal: measure how reliably "Frangi sigma=2 + straight >=15 mm" identifies
bolt positions on T22 (clean) and T2 (clipped). No HU thresholds. No mask,
so we can see if linearity alone filters the skull.

Method:
  1. Frangi sigma=2 on raw CT (full volume, no mask).
  2. Sweep threshold in {20, 40, 80}; 3D connected components on thresholded
     voxels.
  3. Per CC: PCA; require span >= 15 mm AND lam2/lam1 <= 0.05 (linear).
  4. Eval: for each GT shank, is there a candidate CC whose nearer endpoint
     is within 10 mm of GT entry AND axis within 15 deg of GT direction?
  5. Report per threshold: TP count, FP count.

Usage:
  /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
    tests/deep_core/probe_bolt_recovery.py [T22|T2]
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

THRESHOLDS = (20.0, 40.0, 80.0)
SPAN_MIN_MM = 15.0
RATIO_MAX = 0.05

MATCH_ANGLE_DEG = 15.0
MATCH_ENTRY_MM = 10.0


def _unit(v):
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def frangi_sigma(img, sigma):
    import SimpleITK as sitk
    sm = sitk.SmoothingRecursiveGaussian(img, sigma=float(sigma))
    ob = sitk.ObjectnessMeasure(sm, objectDimension=1, brightObject=True)
    return sitk.GetArrayFromImage(ob).astype(np.float32)


def analyze_ccs(labeled_arr, spacing_xyz):
    """Per-CC PCA in world-mm. spacing is (sx, sy, sz); array is KJI."""
    n = int(labeled_arr.max())
    sx, sy, sz = float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])
    results = []
    for lab in range(1, n + 1):
        kk, jj, ii = np.where(labeled_arr == lab)
        if kk.size < 4:
            continue
        pts_ijk = np.stack([ii * sx, jj * sy, kk * sz], axis=1).astype(np.float64)
        c = pts_ijk.mean(axis=0)
        X = pts_ijk - c
        cov = X.T @ X / max(1, pts_ijk.shape[0] - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]
        l1, l2, _ = float(eigvals[0]), float(eigvals[1]), float(eigvals[2])
        pc1 = eigvecs[:, 0]
        proj = X @ pc1
        span_mm = float(proj.max() - proj.min())
        ratio = l2 / l1 if l1 > 1e-12 else 1.0
        # endpoints in IJK-mm (stamped into image-coord mm, pre-direction)
        p_lo = c + float(proj.min()) * pc1
        p_hi = c + float(proj.max()) * pc1
        # Store kji of centroid too for later RAS conversion
        kji_pts = np.stack([kk, jj, ii], axis=1).astype(np.float64)
        kji_c = kji_pts.mean(axis=0)
        X_kji = kji_pts - kji_c
        _, _, Vt_kji = np.linalg.svd(X_kji, full_matrices=False)
        kji_axis = _unit(Vt_kji[0])
        kji_proj = X_kji @ kji_axis
        kji_lo = kji_c + float(kji_proj.min()) * kji_axis
        kji_hi = kji_c + float(kji_proj.max()) * kji_axis
        results.append(dict(
            label=lab, n_vox=int(kk.size),
            span_mm=span_mm, ratio=ratio, l1=l1, l2=l2,
            kji_c=kji_c, kji_lo=kji_lo, kji_hi=kji_hi,
        ))
    return results


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


def _line_axis(start, end):
    d = end - start
    n = float(np.linalg.norm(d))
    return (d / n) if n > 1e-9 else np.array([0.0, 0.0, 1.0]), n


def kji_to_ras(kji_point, ijk_to_ras_mat):
    k, j, i = float(kji_point[0]), float(kji_point[1]), float(kji_point[2])
    h = np.array([i, j, k, 1.0])
    return (ijk_to_ras_mat @ h)[:3]


def match_candidates_to_gt(candidates_ras, gt):
    """Greedy 1-to-1: each GT gets best matching candidate (if any)."""
    pairs = []
    for gi, (_, g_s, g_e, _) in enumerate(gt):
        g_axis, _ = _line_axis(g_s, g_e)
        for ti, c in enumerate(candidates_ras):
            c_axis = c["axis_ras"]
            ang = float(np.degrees(np.arccos(np.clip(abs(np.dot(g_axis, c_axis)), 0, 1))))
            if ang > MATCH_ANGLE_DEG:
                continue
            d_lo = float(np.linalg.norm(c["p_lo_ras"] - g_s))
            d_hi = float(np.linalg.norm(c["p_hi_ras"] - g_s))
            d_entry = min(d_lo, d_hi)
            if d_entry > MATCH_ENTRY_MM:
                continue
            score = ang + d_entry
            pairs.append((score, gi, ti, ang, d_entry))
    pairs.sort(key=lambda p: p[0])
    gt_assigned = {}
    used = set()
    for _, gi, ti, ang, de in pairs:
        if gi in gt_assigned or ti in used:
            continue
        gt_assigned[gi] = dict(ti=ti, ang=ang, d_entry=de)
        used.add(ti)
    return gt_assigned, used


def run(subject_id):
    import SimpleITK as sitk
    from shank_core.io import image_ijk_ras_matrices

    ct_path = DATASET_ROOT / "post_registered_ct" / f"{subject_id}_post_registered.nii.gz"
    print(f"# subject={subject_id}  ct={ct_path.name}")
    img = sitk.ReadImage(str(ct_path))
    spacing = img.GetSpacing()
    ijk_to_ras_mat, _ = image_ijk_ras_matrices(img)
    ijk_to_ras_mat = np.asarray(ijk_to_ras_mat, dtype=float)

    t0 = time.time()
    frangi_s2 = frangi_sigma(img, sigma=2.0)
    print(
        f"# frangi sigma=2: {time.time()-t0:.1f}s  "
        f"max={frangi_s2.max():.1f}  "
        f"p99={np.percentile(frangi_s2, 99):.1f}  "
        f"p999={np.percentile(frangi_s2, 99.9):.1f}"
    )

    gt = load_gt(subject_id)
    print(f"# GT trajectories: {len(gt)}")

    # Accepted CCs (keep spatial labels for combined output)
    accepted_all = {}  # thr -> list of candidate dicts
    best_thr = None
    best_score = None  # (tp, -fp) ordering

    for thr in THRESHOLDS:
        bin_arr = (frangi_s2 >= thr)
        n_bin = int(bin_arr.sum())
        bin_img = sitk.GetImageFromArray(bin_arr.astype(np.uint8))
        bin_img.CopyInformation(img)
        cc_filter = sitk.ConnectedComponentImageFilter()
        cc_filter.SetFullyConnected(True)
        cc_img = cc_filter.Execute(bin_img)
        cc_arr = sitk.GetArrayFromImage(cc_img)
        n_cc = int(cc_arr.max())

        infos = analyze_ccs(cc_arr, spacing)
        kept = [
            d for d in infos
            if d["span_mm"] >= SPAN_MIN_MM and d["ratio"] <= RATIO_MAX
        ]

        # Convert endpoints to RAS and axis
        candidates_ras = []
        for d in kept:
            p_lo_ras = kji_to_ras(d["kji_lo"], ijk_to_ras_mat)
            p_hi_ras = kji_to_ras(d["kji_hi"], ijk_to_ras_mat)
            axis_ras, length_ras = _line_axis(p_lo_ras, p_hi_ras)
            candidates_ras.append(dict(
                label=d["label"], n_vox=d["n_vox"],
                span_mm=d["span_mm"], ratio=d["ratio"],
                p_lo_ras=p_lo_ras, p_hi_ras=p_hi_ras,
                axis_ras=axis_ras, length_ras=length_ras,
            ))

        gt_assigned, used = match_candidates_to_gt(candidates_ras, gt)
        tp = len(gt_assigned)
        fp = len(candidates_ras) - len(used)

        print(
            f"\n## threshold={thr:.0f}  "
            f"above_thr_vox={n_bin}  n_cc={n_cc}  "
            f"kept(span>=15,ratio<=0.05)={len(kept)}  "
            f"TP={tp}/{len(gt)}  FP={fp}"
        )
        # Per-GT breakdown
        missed = []
        for gi, (name, _, _, _) in enumerate(gt):
            if gi in gt_assigned:
                a = gt_assigned[gi]
                print(
                    f"  {name:10s}  MATCH  cand={a['ti']+1:>3d}  "
                    f"ang={a['ang']:.2f}deg  d_entry={a['d_entry']:.2f}mm"
                )
            else:
                missed.append(name)
                # find closest candidate among ALL (kept) for diagnostic
                if candidates_ras:
                    g_s, g_e = gt[gi][1], gt[gi][2]
                    g_axis, _ = _line_axis(g_s, g_e)
                    best = None
                    for ti, c in enumerate(candidates_ras):
                        ang = float(np.degrees(np.arccos(
                            np.clip(abs(np.dot(g_axis, c["axis_ras"])), 0, 1))))
                        d = min(
                            float(np.linalg.norm(c["p_lo_ras"] - g_s)),
                            float(np.linalg.norm(c["p_hi_ras"] - g_s)),
                        )
                        score = ang + d
                        if best is None or score < best["score"]:
                            best = dict(ti=ti, ang=ang, d=d, score=score)
                    b = best
                    print(
                        f"  {name:10s}  miss   best cand={b['ti']+1:>3d}  "
                        f"ang={b['ang']:.2f}deg  d_entry={b['d']:.2f}mm"
                    )
                else:
                    print(f"  {name:10s}  miss   (no kept candidates)")

        accepted_all[thr] = dict(
            candidates=candidates_ras, cc_arr=cc_arr, kept=kept,
            tp=tp, fp=fp,
        )

        score = (tp, -fp)
        if best_score is None or score > best_score:
            best_score = score
            best_thr = thr

    print(f"\n## summary")
    print(f"{'thr':>5s} {'kept':>5s} {'TP':>3s} {'FP':>4s}")
    for thr in THRESHOLDS:
        a = accepted_all[thr]
        print(f"{thr:>5.0f} {len(a['candidates']):>5d} {a['tp']:>3d} {a['fp']:>4d}")
    print(f"# best-balanced threshold: {best_thr:.0f}  "
          f"(TP={accepted_all[best_thr]['tp']}, FP={accepted_all[best_thr]['fp']})")

    # Write NIFTI of best-threshold kept candidates (sequential labels)
    a = accepted_all[best_thr]
    out = np.zeros_like(a["cc_arr"], dtype=np.uint16)
    for i, d in enumerate(a["kept"], start=1):
        out[a["cc_arr"] == d["label"]] = i
    out_img = sitk.GetImageFromArray(out)
    out_img.CopyInformation(img)
    out_path = f"/tmp/bolt_candidates_{subject_id}.nii.gz"
    sitk.WriteImage(out_img, out_path)
    print(f"# wrote {out_path} (threshold={best_thr:.0f})")


if __name__ == "__main__":
    subj = sys.argv[1] if len(sys.argv) > 1 else "T22"
    run(subj)
