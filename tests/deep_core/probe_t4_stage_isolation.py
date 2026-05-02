"""Diagnose why stage 2 emits zero on T4.

Runs stage 1 and stage 2 independently on T4, then compares to the redone
T4 GT. For each GT shank reports (a) stage-1's best match, (b) stage-2's
best match when given the real stage-1 exclusion mask, (c) stage-2's best
match when given NO exclusion mask. For LSFG specifically, also dumps
Frangi-cloud statistics in a tube around the true GT axis (raw, intracranial-
gated, and exclusion-gated) so we can tell whether stage 2 is being blocked
by exclusion vs not seeing the shaft signal at all.

Three possible verdicts:
  (a) stage 2 sees LSFG but stage-1 exclusion masks it out → widen / refine
      exclusion or let stage 2 propose even inside exclusion.
  (b) stage 2 sees LSFG cloud but filters drop it (span/aspect/hull) →
      tune the Frangi-CC filter stack.
  (c) stage 2 doesn't see LSFG at all (Frangi < STAGE2_FRANGI_THR on axis) →
      Frangi shaft response is too weak; need a different shaft feature.

Run
---
    /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_t4_stage_isolation.py
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
from eval_seeg_localization import build_detection_context, iter_subject_rows

DATASET_ROOT = Path("/Users/ammar/Dropbox/thalamus_subjects/seeg_localization")
NEW_GT_CSV = (
    DATASET_ROOT / "contact_label_dataset" / "rosa_helper_import" / "T4"
    / "ROSA_Contacts_T4_final_trajectory_points.csv"
)

MATCH_ANGLE_DEG = 10.0
MATCH_MID_MM = 8.0


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def _axis_midpoint(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return _unit(b - a), 0.5 * (a + b)


def _angle_deg(u, v):
    return float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(u, v)))))))


def _perp_mid(g_mid, t_mid, t_axis):
    v = g_mid - t_mid
    perp = v - (v @ t_axis) * t_axis
    return float(np.linalg.norm(perp))


def _load_new_gt():
    by_shank = defaultdict(dict)
    with NEW_GT_CSV.open() as f:
        for row in csv.DictReader(f):
            shank = row["trajectory"]
            pt = np.array(
                [float(row["x_world_ras"]),
                 float(row["y_world_ras"]),
                 float(row["z_world_ras"])],
                dtype=float,
            )
            by_shank[shank][row["point_type"]] = pt
    return {
        s: (d["entry"], d["target"])
        for s, d in by_shank.items()
        if "entry" in d and "target" in d
    }


def _best_match(trajs, g_axis, g_mid):
    """Return dict with best-scoring trajectory against one GT axis."""
    best = None
    for ti, t in enumerate(trajs):
        t_axis, t_mid = _axis_midpoint(t["start_ras"], t["end_ras"])
        ang = _angle_deg(g_axis, t_axis)
        mid_d = _perp_mid(g_mid, t_mid, t_axis)
        score = ang + mid_d
        if best is None or score < best["score"]:
            best = dict(
                ti=ti, ang=ang, mid_d=mid_d, score=score,
                n_inl=int(t.get("n_inliers", 0)),
                length=float(t.get("length_mm", 0.0)),
            )
    return best


def _tube_voxel_count(axis_start, axis_end, bool_arr_kji, ras_to_ijk_mat,
                      radius_mm, step_mm=0.5):
    """Count True voxels in a radius-mm tube around the [start,end] RAS axis.
    Samples every step_mm along the axis; at each step, counts voxels in a
    cubic window of side (2*radius_vox+1). Simple & matches the exclusion-mask
    logic so stats are directly comparable.
    """
    K, J, I = bool_arr_kji.shape
    s = np.asarray(axis_start, dtype=float)
    e = np.asarray(axis_end, dtype=float)
    axis = e - s
    L = float(np.linalg.norm(axis))
    if L < 1e-3:
        return 0
    axis /= L
    n_steps = int(L / step_mm) + 1
    seen = set()
    r = int(np.ceil(radius_mm / 0.5))
    for k_step in range(n_steps):
        p = s + k_step * step_mm * axis
        h = np.array([p[0], p[1], p[2], 1.0])
        ijk = (ras_to_ijk_mat @ h)[:3]
        kc = int(round(ijk[2])); jc = int(round(ijk[1])); ic = int(round(ijk[0]))
        k0, k1 = max(0, kc - r), min(K, kc + r + 1)
        j0, j1 = max(0, jc - r), min(J, jc + r + 1)
        i0, i1 = max(0, ic - r), min(I, ic + r + 1)
        if k1 <= k0 or j1 <= j0 or i1 <= i0:
            continue
        sub = bool_arr_kji[k0:k1, j0:j1, i0:i1]
        # Count True voxels (not unique — fast and sufficient for comparison).
        seen.add((k0, k1, j0, j1, i0, i1, int(sub.sum())))
    # sum counts per step
    return sum(s[-1] for s in seen)


def _frangi_profile_along_axis(axis_start, axis_end, frangi_kji, ras_to_ijk_mat,
                                step_mm=0.5):
    """Return (distances_mm, frangi_values) sampled at the nearest voxel."""
    K, J, I = frangi_kji.shape
    s = np.asarray(axis_start, dtype=float)
    e = np.asarray(axis_end, dtype=float)
    d = e - s
    L = float(np.linalg.norm(d))
    if L < 1e-3:
        return np.array([]), np.array([])
    axis = d / L
    n = int(L / step_mm) + 1
    dists = np.arange(n) * step_mm
    vals = np.zeros(n, dtype=np.float32)
    for idx in range(n):
        p = s + dists[idx] * axis
        h = np.array([p[0], p[1], p[2], 1.0])
        ijk = (ras_to_ijk_mat @ h)[:3]
        kc = int(round(ijk[2])); jc = int(round(ijk[1])); ic = int(round(ijk[0]))
        if 0 <= kc < K and 0 <= jc < J and 0 <= ic < I:
            vals[idx] = float(frangi_kji[kc, jc, ic])
    return dists, vals


def _assemble_s1_with_inliers(stage1_lines, pts_blobs, log1, ras_to_ijk_mat):
    """Attach inlier_ras and inlier_amps like run_two_stage_detection does —
    exclusion-mask compute expects start/end RAS only, but downstream helpers
    expect the extra fields."""
    K, J, I = log1.shape
    if pts_blobs.shape[0] > 0:
        h = np.concatenate([pts_blobs, np.ones((pts_blobs.shape[0], 1))], axis=1)
        ijk_all = (ras_to_ijk_mat @ h.T).T[:, :3]
        ii = np.clip(np.round(ijk_all[:, 0]).astype(int), 0, I - 1)
        jj = np.clip(np.round(ijk_all[:, 1]).astype(int), 0, J - 1)
        kk = np.clip(np.round(ijk_all[:, 2]).astype(int), 0, K - 1)
        blob_amps = np.abs(log1[kk, jj, ii]).astype(np.float32)
    else:
        blob_amps = None
    for l in stage1_lines:
        try:
            l["inlier_ras"] = np.asarray(pts_blobs[l["inlier_idx"]], dtype=float)
            if blob_amps is not None:
                l["inlier_amps"] = np.asarray(
                    blob_amps[l["inlier_idx"]], dtype=float)
        except Exception:
            pass


def main():
    rows = iter_subject_rows(DATASET_ROOT, {"T4"})
    assert rows, "T4 not in manifest"
    ct_path = rows[0]["ct_path"]

    gts = _load_new_gt()
    print(f"Loaded T4 redone GT: {len(gts)} shanks — {sorted(gts.keys())}")
    print(f"CT: {ct_path}")

    # Preprocess exactly as run_two_stage_detection does.
    img = sitk.ReadImage(ct_path)
    img = sitk.Clamp(img, lowerBound=-1024.0, upperBound=cpfit.HU_CLIP_MAX)
    from eval_seeg_localization import image_ijk_ras_matrices
    ijk_to_ras_mat, ras_to_ijk_mat = image_ijk_ras_matrices(img)
    ijk_to_ras_mat = np.asarray(ijk_to_ras_mat, dtype=float)
    ras_to_ijk_mat = np.asarray(ras_to_ijk_mat, dtype=float)

    ct_arr_kji = sitk.GetArrayFromImage(img).astype(np.float32)
    print(
        f"CT shape={ct_arr_kji.shape} "
        f"HU[min/mean/max]={ct_arr_kji.min():.0f}/"
        f"{ct_arr_kji.mean():.0f}/{ct_arr_kji.max():.0f}"
    )

    hull, intracranial, dist_arr = cpfit.build_masks(img)
    log1 = cpfit.log_sigma(img, sigma_mm=cpfit.LOG_SIGMA_MM)
    frangi_s1 = cpfit.frangi_single(img, sigma=cpfit.FRANGI_STAGE1_SIGMA)
    kji_to_ras = cpfit._kji_to_ras_fn_from_matrix(ijk_to_ras_mat)

    print(f"\nFrangi range: [{frangi_s1.min():.2f}, {frangi_s1.max():.2f}], "
          f"threshold={cpfit.STAGE2_FRANGI_THR}")
    n_fr_pos = int((frangi_s1 >= cpfit.STAGE2_FRANGI_THR).sum())
    n_fr_ic = int(((frangi_s1 >= cpfit.STAGE2_FRANGI_THR) & intracranial).sum())
    print(f"Frangi >= thr: {n_fr_pos:,} voxels  "
          f"(intracranial gated: {n_fr_ic:,})")

    # ---- Stage 1 ------------------------------------------------------
    stage1_lines, pts_blobs = cpfit.run_stage1(
        log1, kji_to_ras, dist_arr, ras_to_ijk_mat,
        pitches_mm=cpfit.resolve_pitches_for_strategy("dixi"),
    )
    _assemble_s1_with_inliers(stage1_lines, pts_blobs, log1, ras_to_ijk_mat)
    print(f"\nstage 1: {len(stage1_lines)} lines")

    # ---- Stage 1 exclusion mask -------------------------------------
    excl_s1 = cpfit.compute_exclusion_mask(
        frangi_s1.shape, stage1_lines, ras_to_ijk_mat,
    )
    excl_empty = np.zeros_like(excl_s1, dtype=bool)
    print(f"stage-1 exclusion mask: {int(excl_s1.sum()):,} voxels True")

    # ---- Stage 2, two variants -------------------------------------
    stage2_lines_real, cc_arr_real = cpfit.run_stage2(
        frangi_s1, intracranial, excl_s1, img.GetSpacing(),
        dist_arr, ijk_to_ras_mat, ras_to_ijk_mat,
    )
    stage2_lines_noexcl, cc_arr_noexcl = cpfit.run_stage2(
        frangi_s1, intracranial, excl_empty, img.GetSpacing(),
        dist_arr, ijk_to_ras_mat, ras_to_ijk_mat,
    )
    print(
        f"stage 2 (with s1 exclusion):  {len(stage2_lines_real)} lines"
    )
    print(
        f"stage 2 (WITHOUT exclusion):  {len(stage2_lines_noexcl)} lines"
    )

    # ---- Per-GT best-match table ------------------------------------
    def _fmt_best(b):
        if b is None:
            return "      -        -      -         -"
        match = b["ang"] <= MATCH_ANGLE_DEG and b["mid_d"] <= MATCH_MID_MM
        tag = "MATCH" if match else ("near" if b["ang"] <= 15 and b["mid_d"] <= 13 else "miss")
        return (
            f"t{b['ti']:>2d} ang={b['ang']:5.2f}° mid={b['mid_d']:5.2f}mm "
            f"n_in={b['n_inl']:>3d} len={b['length']:5.1f}  {tag:>5s}"
        )

    print(
        f"\n{'='*110}\nPER-GT best match per stage (match window: "
        f"angle<={MATCH_ANGLE_DEG}, mid<={MATCH_MID_MM})\n{'='*110}"
    )
    header = (
        f"{'shank':8s}  "
        f"{'STAGE1':45s}  {'STAGE2 (s1 excl)':45s}  {'STAGE2 (no excl)':45s}"
    )
    print(header)
    s1_matched = 0
    s2_real_matched = 0
    s2_noexcl_matched = 0
    per_shank = {}
    for shank in sorted(gts.keys()):
        g_s, g_e = gts[shank]
        g_axis, g_mid = _axis_midpoint(g_s, g_e)
        b1 = _best_match(stage1_lines, g_axis, g_mid)
        b2r = _best_match(stage2_lines_real, g_axis, g_mid)
        b2n = _best_match(stage2_lines_noexcl, g_axis, g_mid)
        per_shank[shank] = (b1, b2r, b2n)
        if b1 and b1["ang"] <= MATCH_ANGLE_DEG and b1["mid_d"] <= MATCH_MID_MM:
            s1_matched += 1
        if b2r and b2r["ang"] <= MATCH_ANGLE_DEG and b2r["mid_d"] <= MATCH_MID_MM:
            s2_real_matched += 1
        if b2n and b2n["ang"] <= MATCH_ANGLE_DEG and b2n["mid_d"] <= MATCH_MID_MM:
            s2_noexcl_matched += 1
        print(f"{shank:8s}  {_fmt_best(b1):45s}  {_fmt_best(b2r):45s}  {_fmt_best(b2n):45s}")

    print(
        f"\nTotals:  stage1 matched {s1_matched}/{len(gts)}  |  "
        f"stage2 (with excl) {s2_real_matched}/{len(gts)}  |  "
        f"stage2 (no excl) {s2_noexcl_matched}/{len(gts)}"
    )

    # ---- Focus on LSFG -------------------------------------------
    print(f"\n{'='*110}\nLSFG FOCUS — wire-class suspect\n{'='*110}")
    if "LSFG" not in gts:
        print("  LSFG not in GT")
        return
    lsfg_s, lsfg_e = gts["LSFG"]
    g_axis, g_mid = _axis_midpoint(lsfg_s, lsfg_e)
    print(f"  entry={lsfg_s}  target={lsfg_e}")
    print(f"  axis_length={np.linalg.norm(lsfg_e - lsfg_s):.2f} mm")

    # How much Frangi-cloud does LSFG's axis actually sit in?
    fr_mask = frangi_s1 >= cpfit.STAGE2_FRANGI_THR
    fr_ic = fr_mask & intracranial
    fr_s1ex = fr_ic & ~excl_s1
    for radius in (1.0, 2.0, 3.0, 4.0):
        n_raw = _tube_voxel_count(lsfg_s, lsfg_e, fr_mask, ras_to_ijk_mat, radius)
        n_ic = _tube_voxel_count(lsfg_s, lsfg_e, fr_ic, ras_to_ijk_mat, radius)
        n_ex = _tube_voxel_count(lsfg_s, lsfg_e, fr_s1ex, ras_to_ijk_mat, radius)
        print(
            f"  Frangi>={cpfit.STAGE2_FRANGI_THR} tube r={radius:.1f}mm:  "
            f"raw={n_raw:>5d}  intracranial={n_ic:>5d}  "
            f"after s1 excl={n_ex:>5d}"
        )

    # Frangi profile along the true axis.
    dists, frangi_vals = _frangi_profile_along_axis(
        lsfg_s, lsfg_e, frangi_s1, ras_to_ijk_mat, step_mm=0.5,
    )
    ic_axis = np.zeros_like(dists, dtype=bool)
    ex_axis = np.zeros_like(dists, dtype=bool)
    K, J, I = intracranial.shape
    s = np.asarray(lsfg_s, dtype=float)
    e = np.asarray(lsfg_e, dtype=float)
    L = float(np.linalg.norm(e - s))
    axis = (e - s) / L
    for idx, d in enumerate(dists):
        p = s + d * axis
        h = np.array([p[0], p[1], p[2], 1.0])
        ijk = (ras_to_ijk_mat @ h)[:3]
        kc = int(round(ijk[2])); jc = int(round(ijk[1])); ic = int(round(ijk[0]))
        if 0 <= kc < K and 0 <= jc < J and 0 <= ic < I:
            ic_axis[idx] = bool(intracranial[kc, jc, ic])
            ex_axis[idx] = bool(excl_s1[kc, jc, ic])

    print(
        f"\n  Frangi profile along LSFG axis (500µm steps, "
        f"len={dists.size} samples):"
    )
    print(
        f"    range=[{frangi_vals.min():.2f}, {frangi_vals.max():.2f}]  "
        f"mean={frangi_vals.mean():.2f}  "
        f">= thr ({cpfit.STAGE2_FRANGI_THR}): {int((frangi_vals >= cpfit.STAGE2_FRANGI_THR).sum())}/{dists.size}"
    )
    print(
        f"    intracranial samples: {int(ic_axis.sum())}/{dists.size}  "
        f"stage-1-excl-covered: {int(ex_axis.sum())}/{dists.size}"
    )
    # Break down by 5mm bins along the axis.
    print("\n  depth[mm]  frangi_min  frangi_mean  frangi_max  %>=thr  %intra  %excl")
    n_bins = max(1, int(np.ceil(L / 5.0)))
    for bi in range(n_bins):
        lo = bi * 5.0
        hi = min(L, (bi + 1) * 5.0)
        mask = (dists >= lo) & (dists < hi if bi < n_bins - 1 else dists <= hi)
        if not mask.any():
            continue
        v = frangi_vals[mask]
        ic_b = ic_axis[mask]
        ex_b = ex_axis[mask]
        print(
            f"  {lo:4.1f}-{hi:4.1f}   {v.min():8.2f}   {v.mean():9.2f}   "
            f"{v.max():8.2f}   "
            f"{100.0 * float((v >= cpfit.STAGE2_FRANGI_THR).mean()):5.1f}%  "
            f"{100.0 * float(ic_b.mean()):5.1f}%  "
            f"{100.0 * float(ex_b.mean()):5.1f}%"
        )

    # Which stage-1 line (if any) is covering LSFG's region via exclusion?
    print("\n  Stage-1 line(s) whose exclusion overlaps LSFG axis:")
    for idx, l in enumerate(stage1_lines):
        t_axis, t_mid = _axis_midpoint(l["start_ras"], l["end_ras"])
        ang = _angle_deg(g_axis, t_axis)
        mid_d = _perp_mid(g_mid, t_mid, t_axis)
        if mid_d < 6.0:  # close enough to matter for LSFG exclusion
            print(
                f"    s1[{idx}]  ang={ang:5.2f}°  mid_d={mid_d:5.2f}mm  "
                f"n_in={int(l.get('n_blobs', 0))}  "
                f"span={float(l.get('span_mm', 0)):.1f}mm  "
                f"start={np.asarray(l['start_ras']).round(1).tolist()}  "
                f"end={np.asarray(l['end_ras']).round(1).tolist()}"
            )

    # Which stage-2 CC (no-excl run) best lines up with LSFG, even if filtered?
    print("\n  Stage-2 CCs (no-excl run) near LSFG axis (all CCs before filters):")
    cloud_noexcl = (frangi_s1 >= cpfit.STAGE2_FRANGI_THR) & intracranial
    cc_img = sitk.ConnectedComponentImageFilter()
    cc_img.SetFullyConnected(True)
    cc = sitk.GetArrayFromImage(cc_img.Execute(
        sitk.GetImageFromArray(cloud_noexcl.astype(np.uint8))))
    # Collect CCs that touch LSFG's 3-mm tube.
    tube = np.zeros_like(cloud_noexcl, dtype=bool)
    r_vox = int(np.ceil(3.0 / 0.5))
    n_steps = int(L / 0.5) + 1
    for k_step in range(n_steps):
        p = s + k_step * 0.5 * axis
        h = np.array([p[0], p[1], p[2], 1.0])
        ijk = (ras_to_ijk_mat @ h)[:3]
        kc = int(round(ijk[2])); jc = int(round(ijk[1])); ic = int(round(ijk[0]))
        k0, k1 = max(0, kc - r_vox), min(K, kc + r_vox + 1)
        j0, j1 = max(0, jc - r_vox), min(J, jc + r_vox + 1)
        i0, i1 = max(0, ic - r_vox), min(I, ic + r_vox + 1)
        tube[k0:k1, j0:j1, i0:i1] = True
    touching = np.unique(cc[tube])
    touching = [lab for lab in touching.tolist() if lab != 0]
    print(f"    {len(touching)} CC(s) touching LSFG 3-mm tube")
    sx, sy, sz = img.GetSpacing()
    for lab in touching[:10]:
        vox_mask = (cc == lab)
        n_vox = int(vox_mask.sum())
        kk, jj, ii = np.where(vox_mask)
        if n_vox == 0:
            continue
        pts_mm = np.stack([ii * sx, jj * sy, kk * sz], axis=1).astype(float)
        c = pts_mm.mean(axis=0)
        X = pts_mm - c
        cov = X.T @ X / max(1, n_vox - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = eigvals[::-1]; eigvecs = eigvecs[:, ::-1]
        pc1 = eigvecs[:, 0]
        proj = X @ pc1
        span = float(proj.max() - proj.min())
        perp = X - np.outer(proj, pc1)
        perp_rms = float(np.sqrt(np.mean(np.sum(perp * perp, axis=1))))
        aspect = span / max(perp_rms, 1e-3)
        print(
            f"    cc={lab:>5d}  n_vox={n_vox:>6d}  span={span:5.1f}mm  "
            f"perp_rms={perp_rms:5.2f}mm  aspect={aspect:5.1f}  "
            f"pass_vox={(cpfit.STAGE2_MIN_VOXELS <= n_vox <= cpfit.STAGE2_MAX_VOXELS)}"
            f"  pass_span={(cpfit.STAGE2_MIN_SPAN_MM <= span <= cpfit.STAGE2_MAX_SPAN_MM)}"
            f"  pass_perp={(perp_rms <= cpfit.STAGE2_MAX_PERP_RMS_MM)}"
            f"  pass_asp={(aspect >= cpfit.STAGE2_MIN_ASPECT_GEOM)}"
        )


if __name__ == "__main__":
    main()
