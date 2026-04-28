"""Inspect every T1 orphan trajectory in detail.

For each orphan: full geometry, score components, bolt_source, and a
1D LoG profile along the axis at 0.5 mm steps. The intent is to
identify a specific orphan from a Slicer screenshot by its endpoint
positions, and explain why the pipeline emitted it.

Run:
    cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper && \\
        /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_t1_orphans_inspect.py
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
from scipy.ndimage import map_coordinates

from postop_ct_localization import contact_pitch_v1_fit as cpfit
from shank_engine import PipelineRegistry, register_builtin_pipelines
from shank_core.io import image_ijk_ras_matrices
from eval_seeg_localization import (
    build_detection_context, iter_subject_rows,
    load_reference_ground_truth_shanks,
)

DATASET_ROOT = Path("/Users/ammar/Dropbox/thalamus_subjects/seeg_localization")
SAMPLE_STEP_MM = 0.5


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def _greedy_match(gt_shanks, trajs):
    pairs = []
    for gi, g in enumerate(gt_shanks):
        gs, ge = np.asarray(g.start_ras, dtype=float), np.asarray(g.end_ras, dtype=float)
        ga = _unit(ge - gs); gm = 0.5 * (gs + ge)
        for ti, t in enumerate(trajs):
            ts, te = np.asarray(t["start_ras"], dtype=float), np.asarray(t["end_ras"], dtype=float)
            ta = _unit(te - ts); tm = 0.5 * (ts + te)
            ang = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(ga, ta)))))))
            d = gm - tm; mid = float(np.linalg.norm(d - (d @ ta) * ta))
            if ang <= 10 and mid <= 8:
                pairs.append((ang + mid, gi, ti))
    pairs.sort(); used_g, used_t = set(), set(); m = {}
    for _s, gi, ti in pairs:
        if gi in used_g or ti in used_t: continue
        used_g.add(gi); used_t.add(ti); m[ti] = str(gt_shanks[gi].shank)
    return m


def _canonical_log_and_ct(raw_img):
    """Reproduce production preprocessing; return log_arr_kji, ct_arr_kji,
    ras_to_ijk_mat for the canonical-grid space.
    """
    spacing = raw_img.GetSpacing()
    if min(float(s) for s in spacing) < cpfit.CANONICAL_SPACING_MM * 0.95:
        size_in = raw_img.GetSize()
        target = (cpfit.CANONICAL_SPACING_MM,)*3
        target_size = [
            max(1, int(round(size_in[i] * float(spacing[i]) / cpfit.CANONICAL_SPACING_MM)))
            for i in range(3)
        ]
        rs = sitk.ResampleImageFilter()
        rs.SetOutputSpacing(target); rs.SetSize(target_size)
        rs.SetOutputOrigin(raw_img.GetOrigin())
        rs.SetOutputDirection(raw_img.GetDirection())
        rs.SetInterpolator(sitk.sitkLinear)
        rs.SetDefaultPixelValue(-1024)
        img = rs.Execute(raw_img)
        if getattr(cpfit, "RAW_RESAMPLE_GAUSSIAN_SIGMA_MM", 0.0) > 0:
            img = sitk.SmoothingRecursiveGaussian(img, cpfit.RAW_RESAMPLE_GAUSSIAN_SIGMA_MM)
    else:
        img = raw_img
    img_clamped = sitk.Clamp(img, lowerBound=-1024.0, upperBound=cpfit.HU_CLIP_MAX)
    log_sitk = sitk.LaplacianRecursiveGaussian(
        sitk.Cast(img_clamped, sitk.sitkFloat32), sigma=cpfit.LOG_SIGMA_MM,
    )
    log_arr = sitk.GetArrayFromImage(log_sitk)
    ct_arr = sitk.GetArrayFromImage(img).astype(np.float32)
    _, ras_to_ijk = image_ijk_ras_matrices(img)
    return log_arr, ct_arr, np.asarray(ras_to_ijk, dtype=float)


def _sample_along(arr, ras_to_ijk, start, end, step_mm=SAMPLE_STEP_MM):
    s = np.asarray(start, dtype=float); e = np.asarray(end, dtype=float)
    L = float(np.linalg.norm(e - s))
    n = max(8, int(round(L / step_mm)) + 1)
    t = np.linspace(0.0, 1.0, n)
    pts = s[None, :] + t[:, None] * (e - s)[None, :]
    h = np.concatenate([pts, np.ones((n, 1))], axis=1)
    ijk = (ras_to_ijk @ h.T).T[:, :3]
    kji = ijk[:, [2, 1, 0]]
    samples = map_coordinates(arr, kji.T, order=1, mode='nearest')
    return samples.astype(float), L, n


def main():
    rows = iter_subject_rows(DATASET_ROOT, {"T1"})
    if not rows:
        print("ERROR: T1 not found"); return 1
    row = rows[0]
    gt, _ = load_reference_ground_truth_shanks(row)

    registry = PipelineRegistry()
    register_builtin_pipelines(registry)
    ctx, raw_img = build_detection_context(
        row["ct_path"], run_id="probe_t1_orphans", config={}, extras={},
    )
    ctx["contact_pitch_v1_pitch_strategy"] = "auto"
    result = registry.run("contact_pitch_v1", ctx)
    trajs = list(result.get("trajectories") or [])
    matched = _greedy_match(gt, trajs)

    log_arr, ct_arr, ras_to_ijk = _canonical_log_and_ct(raw_img)

    print(f"T1: {len(trajs)} emissions ({len(matched)} matched, "
          f"{len(trajs) - len(matched)} orphans)\n")

    orphan_idxs = [ti for ti in range(len(trajs)) if ti not in matched]
    for orph_n, ti in enumerate(orphan_idxs):
        t = trajs[ti]
        s = np.asarray(t["start_ras"], dtype=float)
        e = np.asarray(t["end_ras"], dtype=float)
        mid = 0.5 * (s + e)
        axis = _unit(e - s)
        L = float(np.linalg.norm(e - s))

        log_samples, _, n_samples = _sample_along(log_arr, ras_to_ijk, s, e)
        ct_samples, _, _ = _sample_along(ct_arr, ras_to_ijk, s, e)

        # Detect contact peaks: count voxels with |LoG| > 500 along axis.
        n_above_thr = int(np.sum(np.abs(log_samples) > 500))
        n_above_strong = int(np.sum(np.abs(log_samples) > 1000))

        print(f"=== ORPHAN #{orph_n + 1}  (traj idx {ti}) ===")
        print(f"  start_ras = [{s[0]:+7.2f}, {s[1]:+7.2f}, {s[2]:+7.2f}]")
        print(f"  end_ras   = [{e[0]:+7.2f}, {e[1]:+7.2f}, {e[2]:+7.2f}]")
        print(f"  midpoint  = [{mid[0]:+7.2f}, {mid[1]:+7.2f}, {mid[2]:+7.2f}]")
        print(f"  axis      = [{axis[0]:+5.3f}, {axis[1]:+5.3f}, {axis[2]:+5.3f}]")
        print(f"  length    = {L:.2f} mm  (n_inliers={int(t.get('n_inliers', 0))})")
        print(f"  bolt_source={t.get('bolt_source','?'):>10s}  "
              f"conf={float(t.get('confidence',0)):.3f}({str(t.get('confidence_label','?')):>6s})")
        print(f"  pitch     = {float(t.get('original_median_pitch_mm', 0)):.2f} mm  "
              f"contact_span={float(t.get('contact_span_mm', 0)):.1f} mm  "
              f"dist_max={float(t.get('dist_max_mm', 0)):.1f} mm")
        print(f"  amp_sum   = {float(t.get('amp_sum', 0)):.0f}  "
              f"frangi_med={float(t.get('frangi_median_mm', 0)):.0f}")

        # Score component breakdown (recompute since trajectory may not store it)
        score, label, components = cpfit._compute_trajectory_score(t)
        print(f"  score components:")
        for k, v in sorted(components.items()):
            print(f"    {k:>14s} = {v:.3f}")

        print(f"  axis sampled @ {SAMPLE_STEP_MM}mm: n={n_samples} samples")
        print(f"    |LoG|>500: {n_above_thr} samples ({100*n_above_thr/n_samples:.0f}%)  "
              f"|LoG|>1000: {n_above_strong} samples ({100*n_above_strong/n_samples:.0f}%)")
        print(f"    LoG p10/p50/p90 = {np.percentile(log_samples, 10):+.0f} / "
              f"{np.percentile(log_samples, 50):+.0f} / "
              f"{np.percentile(log_samples, 90):+.0f}")
        print(f"    HU  p10/p50/p90 = {np.percentile(ct_samples, 10):+.0f} / "
              f"{np.percentile(ct_samples, 50):+.0f} / "
              f"{np.percentile(ct_samples, 90):+.0f}")

        # Spatial profile of |LoG| along axis (compact)
        n_bins = 20
        bin_edges = np.linspace(0, n_samples, n_bins + 1, dtype=int)
        bin_max = []
        for k in range(n_bins):
            seg = log_samples[bin_edges[k]:bin_edges[k+1]]
            bin_max.append(float(np.abs(seg).max()) if seg.size else 0.0)
        bar_chars = []
        for v in bin_max:
            if v > 1000: bar_chars.append("#")
            elif v > 500: bar_chars.append("=")
            elif v > 200: bar_chars.append("-")
            else: bar_chars.append(".")
        print(f"    |LoG| along axis (start→end, 20 bins):  {''.join(bar_chars)}")
        bin_max_hu = []
        for k in range(n_bins):
            seg = ct_samples[bin_edges[k]:bin_edges[k+1]]
            bin_max_hu.append(float(seg.max()) if seg.size else 0.0)
        bar_hu = []
        for v in bin_max_hu:
            if v > 2000: bar_hu.append("#")
            elif v > 1000: bar_hu.append("=")
            elif v > 100: bar_hu.append("-")
            else: bar_hu.append(".")
        print(f"    HU max  along axis (start→end, 20 bins):  {''.join(bar_hu)}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
