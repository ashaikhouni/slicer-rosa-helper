"""Per-inlier HU probe to discriminate metal-contact peaks from bone-air
interface peaks.

T1 has medium-band orphans visualized to be passing through mastoid /
temporal-bone air cells. The walker locks onto LoG peaks at bone-air
interfaces (high contrast → strong negative LoG) but those peaks are
NOT metal contacts: their underlying HU is variable (interface region
between bone ~1500 HU and air ~-1000 HU), unlike real Pt/Ir contacts
which saturate at metal HU (1500-3000+).

This probe runs the pipeline on every dataset subject, then for each
inlier of every emission samples the CT HU value at the inlier's RAS
position. We compare matched-line inlier HU distribution vs
orphan-line inlier HU distribution to see whether HU at the inlier
discriminates real contacts from bone-air FPs.

Run:
    cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper && \\
        /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_inlier_hu.py
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
EXCLUDE = {"T17", "T19", "T21"}

MATCH_ANGLE_DEG = 10.0
MATCH_MID_MM = 8.0


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def _greedy_match(gt_shanks, trajs):
    pairs = []
    for gi, g in enumerate(gt_shanks):
        g_s = np.asarray(g.start_ras, dtype=float)
        g_e = np.asarray(g.end_ras, dtype=float)
        g_axis = _unit(g_e - g_s)
        g_mid = 0.5 * (g_s + g_e)
        for ti, t in enumerate(trajs):
            t_s = np.asarray(t["start_ras"], dtype=float)
            t_e = np.asarray(t["end_ras"], dtype=float)
            t_axis = _unit(t_e - t_s)
            ang = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(g_axis, t_axis)))))))
            t_mid = 0.5 * (t_s + t_e)
            d = g_mid - t_mid
            p = d - (d @ t_axis) * t_axis
            mid_d = float(np.linalg.norm(p))
            if ang <= MATCH_ANGLE_DEG and mid_d <= MATCH_MID_MM:
                pairs.append((ang + mid_d, gi, ti, ang, mid_d))
    pairs.sort(key=lambda p: p[0])
    used_g, used_t = set(), set()
    matched_ti = {}
    for _s, gi, ti, ang, mid_d in pairs:
        if gi in used_g or ti in used_t:
            continue
        used_g.add(gi); used_t.add(ti)
        matched_ti[ti] = (str(gt_shanks[gi].shank), ang, mid_d)
    return matched_ti


def _sample_hu_at_inliers(inlier_ras, ct_arr_kji, ras_to_ijk_mat):
    """Sample CT HU at each inlier RAS position. Returns array of HU."""
    if inlier_ras is None or len(inlier_ras) == 0:
        return np.array([], dtype=float)
    pts = np.asarray(inlier_ras, dtype=float)
    h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    ijk = (ras_to_ijk_mat @ h.T).T[:, :3]
    K, J, I = ct_arr_kji.shape
    ii = np.clip(np.round(ijk[:, 0]).astype(int), 0, I - 1)
    jj = np.clip(np.round(ijk[:, 1]).astype(int), 0, J - 1)
    kk = np.clip(np.round(ijk[:, 2]).astype(int), 0, K - 1)
    return ct_arr_kji[kk, jj, ii].astype(float)


def main():
    rows = iter_subject_rows(DATASET_ROOT, None)
    rows = [r for r in rows if str(r["subject_id"]) not in EXCLUDE]
    rows.sort(key=lambda r: str(r["subject_id"]))

    matched_hu_per_inlier: list[float] = []
    matched_min_hu_per_line: list[float] = []
    orphan_hu_per_inlier: list[float] = []
    orphan_min_hu_per_line: list[tuple[float, str, str]] = []  # (min_hu, subj, info)

    print(f"running pipeline on {len(rows)} subjects (auto strategy)\n")
    for row in rows:
        sid = str(row["subject_id"])
        gt, _ = load_reference_ground_truth_shanks(row)
        # Run pipeline directly so we can also access the CT array for HU sampling.
        img = sitk.ReadImage(str(row["ct_path"]))
        ct_arr = sitk.GetArrayFromImage(img).astype(np.float32)
        ijk_to_ras_mat, ras_to_ijk_mat = image_ijk_ras_matrices(img)
        trajs = cpfit.run_two_stage_detection(
            img, ijk_to_ras_mat, ras_to_ijk_mat,
            return_features=False, pitch_strategy="auto",
        )
        matched = _greedy_match(gt, trajs)
        n_match = len(matched)
        n_orph = len(trajs) - n_match
        print(f"  {sid}: {n_match}/{len(gt)} matched, {n_orph} orphans")
        for ti, t in enumerate(trajs):
            inlier_ras = t.get("inlier_ras")
            if inlier_ras is None:
                continue
            hus = _sample_hu_at_inliers(inlier_ras, ct_arr, ras_to_ijk_mat)
            if hus.size == 0:
                continue
            line_min = float(hus.min())
            if ti in matched:
                matched_hu_per_inlier.extend(hus.tolist())
                matched_min_hu_per_line.append(line_min)
            else:
                orphan_hu_per_inlier.extend(hus.tolist())
                info = (
                    f"conf={float(t.get('confidence',0.0)):.2f} "
                    f"bolt={t.get('bolt_source','?')} n={int(t.get('n_inliers',0))} "
                    f"frangi={float(t.get('frangi_median_mm',0.0)):.0f} "
                    f"pitch={float(t.get('original_median_pitch_mm',0.0)):.2f}"
                )
                orphan_min_hu_per_line.append((line_min, sid, info))

    arr_m = np.asarray(matched_hu_per_inlier, dtype=float)
    arr_o = np.asarray(orphan_hu_per_inlier, dtype=float)

    print(f"\n=== per-inlier HU distribution ===")
    print(f"  matched: n={arr_m.size}")
    if arr_m.size:
        for q in (1, 5, 10, 25, 50, 75, 90, 95, 99):
            print(f"    p{q:>2d} = {np.percentile(arr_m, q):>7.0f}")
        print(f"    min = {arr_m.min():>7.0f}")
        print(f"    max = {arr_m.max():>7.0f}")
    print(f"  orphan:  n={arr_o.size}")
    if arr_o.size:
        for q in (1, 5, 10, 25, 50, 75, 90, 95, 99):
            print(f"    p{q:>2d} = {np.percentile(arr_o, q):>7.0f}")
        print(f"    min = {arr_o.min():>7.0f}")
        print(f"    max = {arr_o.max():>7.0f}")

    arr_mm = np.asarray(matched_min_hu_per_line, dtype=float)
    arr_om = np.asarray([t[0] for t in orphan_min_hu_per_line], dtype=float)

    print(f"\n=== per-line MIN-HU among inliers (the weakest contact on each line) ===")
    print(f"  matched n={arr_mm.size}")
    if arr_mm.size:
        for q in (1, 5, 10, 25, 50):
            print(f"    p{q:>2d} = {np.percentile(arr_mm, q):>7.0f}")
        print(f"    min = {arr_mm.min():>7.0f}")
    print(f"  orphan  n={arr_om.size}")
    if arr_om.size:
        for q in (50, 75, 90, 95, 99):
            print(f"    p{q:>2d} = {np.percentile(arr_om, q):>7.0f}")
        print(f"    max = {arr_om.max():>7.0f}")

    print(f"\n=== orphan lines sorted by min-HU descending (highest = closest to looking real) ===")
    orphan_min_hu_per_line.sort(key=lambda x: -x[0])
    for hu, sid, info in orphan_min_hu_per_line:
        print(f"  {sid:>3s}  min_inlier_HU={hu:>7.0f}  {info}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
