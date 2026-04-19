"""Probe: on-axis filters for contact peak detection.

For each GT shank trajectory in a subject:
  * sample candidate 1-D profiles along the axis at 0.25 mm steps
  * pick peaks (local minima for LoG, local maxima for CT-HU / top-hat)
  * match peaks to GT contact arc-lengths within ``MATCH_TOL_MM``
  * report median per-contact error, match-rate, extra peaks

Candidates:
  1. LoG sigma=1 (signed) — contacts minima
  2. Raw CT HU — contacts maxima
  3. White top-hat HU — contacts maxima (background subtracted)

Usage:
  /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
    tests/deep_core/probe_contact_peak_filters.py [T22|T2]
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

STEP_MM = 0.25
MATCH_TOL_MM = 1.5
PEAK_MIN_SEPARATION_MM = 2.0


def _unit(v):
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def _sample_trilinear(arr, kji_float):
    k, j, i = kji_float
    s = arr.shape
    if not (0 <= k < s[0] - 1 and 0 <= j < s[1] - 1 and 0 <= i < s[2] - 1):
        return float("nan")
    k0, j0, i0 = int(k), int(j), int(i)
    dk, dj, di = k - k0, j - j0, i - i0
    v000 = arr[k0, j0, i0]; v001 = arr[k0, j0, i0 + 1]
    v010 = arr[k0, j0 + 1, i0]; v011 = arr[k0, j0 + 1, i0 + 1]
    v100 = arr[k0 + 1, j0, i0]; v101 = arr[k0 + 1, j0, i0 + 1]
    v110 = arr[k0 + 1, j0 + 1, i0]; v111 = arr[k0 + 1, j0 + 1, i0 + 1]
    c00 = v000 * (1 - di) + v001 * di
    c01 = v010 * (1 - di) + v011 * di
    c10 = v100 * (1 - di) + v101 * di
    c11 = v110 * (1 - di) + v111 * di
    c0 = c00 * (1 - dj) + c01 * dj
    c1 = c10 * (1 - dj) + c11 * dj
    return float(c0 * (1 - dk) + c1 * dk)


def _orthonormal_basis(direction_unit):
    any_vec = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(direction_unit, any_vec)) > 0.9:
        any_vec = np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(direction_unit, any_vec))
    v = _unit(np.cross(direction_unit, u))
    return u, v


def sample_line_profile(arr_kji, ras_to_ijk_mat, start_ras, direction_unit,
                        n_steps, step_mm, cyl_radius_mm=0.0, n_ring=6,
                        reducer="center"):
    """Sample a 1-D profile along the axis, optionally taking the reducer
    over a perpendicular ring at each step (more robust when the axis
    is 0.5-1 mm off the true shank)."""
    u = v = None
    if cyl_radius_mm > 0.0:
        u, v = _orthonormal_basis(direction_unit)

    vals = np.empty(n_steps, dtype=float)
    for i in range(n_steps):
        center = start_ras + i * step_mm * direction_unit
        if cyl_radius_mm <= 0.0:
            h = np.array([center[0], center[1], center[2], 1.0])
            ijk = ras_to_ijk_mat @ h
            kji = np.array([ijk[2], ijk[1], ijk[0]], dtype=float)
            vals[i] = _sample_trilinear(arr_kji, kji)
        else:
            samples = []
            for k in range(n_ring):
                ang = 2.0 * np.pi * k / n_ring
                p = center + cyl_radius_mm * (np.cos(ang) * u + np.sin(ang) * v)
                h = np.array([p[0], p[1], p[2], 1.0])
                ijk = ras_to_ijk_mat @ h
                kji = np.array([ijk[2], ijk[1], ijk[0]], dtype=float)
                samples.append(_sample_trilinear(arr_kji, kji))
            h = np.array([center[0], center[1], center[2], 1.0])
            ijk = ras_to_ijk_mat @ h
            kji = np.array([ijk[2], ijk[1], ijk[0]], dtype=float)
            samples.append(_sample_trilinear(arr_kji, kji))
            s = np.asarray(samples, dtype=float)
            s = s[np.isfinite(s)]
            if s.size == 0:
                vals[i] = float("nan")
            elif reducer == "max":
                vals[i] = float(s.max())
            elif reducer == "min":
                vals[i] = float(s.min())
            else:  # center
                vals[i] = float(samples[-1]) if np.isfinite(samples[-1]) else (
                    float(s.mean())
                )
    return vals


def find_peaks(profile, polarity, step_mm, min_separation_mm=PEAK_MIN_SEPARATION_MM,
               min_amplitude=None):
    """Return peak arc-length positions (mm from start).

    polarity='min': local minima (LoG).
    polarity='max': local maxima (CT-HU / top-hat).
    """
    x = np.asarray(profile, dtype=float).copy()
    n = x.size
    if n < 3:
        return []
    if polarity == "min":
        # Flip sign so we can use max-seeking.
        y = -x
    else:
        y = x
    # Local maxima:
    is_peak = np.zeros(n, dtype=bool)
    is_peak[1:-1] = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])
    idx = np.where(is_peak)[0]
    if min_amplitude is not None:
        idx = idx[y[idx] >= min_amplitude]
    if idx.size == 0:
        return []
    # Sort by amplitude desc; greedy non-max suppression by arc-length.
    order = idx[np.argsort(-y[idx])]
    selected = []
    min_gap_steps = max(1, int(round(min_separation_mm / step_mm)))
    for i in order:
        if all(abs(int(i) - int(j)) >= min_gap_steps for j in selected):
            selected.append(int(i))
    selected.sort()
    return [float(i) * step_mm for i in selected]


def match_peaks_to_gt(peaks_mm, gt_positions_mm, tol_mm=MATCH_TOL_MM):
    """Greedy nearest match. Returns (errors_mm, n_matched, n_extra)."""
    peaks = sorted(peaks_mm)
    gt = sorted(gt_positions_mm)
    used = set()
    errors = []
    for p in peaks:
        best_i, best_d = -1, None
        for i, g in enumerate(gt):
            if i in used:
                continue
            d = abs(p - g)
            if d <= tol_mm and (best_d is None or d < best_d):
                best_i, best_d = i, d
        if best_i >= 0:
            used.add(best_i)
            errors.append(best_d)
    n_extra = max(0, len(peaks) - len(errors))
    return errors, len(errors), n_extra


def white_top_hat_hu(img, radius_mm):
    import SimpleITK as sitk
    spacing = img.GetSpacing()
    radius_vox = [int(max(1, round(radius_mm / s))) for s in spacing]
    opened = sitk.GrayscaleMorphologicalOpening(
        sitk.Cast(img, sitk.sitkFloat32),
        kernelRadius=radius_vox,
    )
    wth = sitk.Subtract(sitk.Cast(img, sitk.sitkFloat32), opened)
    return sitk.GetArrayFromImage(wth).astype(np.float32)


def run(subject_id):
    import SimpleITK as sitk
    from shank_core.io import image_ijk_ras_matrices
    from eval_seeg_localization import iter_subject_rows, load_ground_truth_shanks

    rows = iter_subject_rows(DATASET_ROOT, {subject_id})
    if not rows:
        print(f"# no rows for {subject_id}")
        return
    row = rows[0]
    ct_path = row["ct_path"]
    shanks = load_ground_truth_shanks(row["labels_path"], row.get("shanks_path"))
    print(f"# subject={subject_id} ct={ct_path} n_shanks={len(shanks)}")

    img = sitk.ReadImage(ct_path)
    ct_arr = sitk.GetArrayFromImage(img).astype(np.float32)
    _, ras_to_ijk_mat = image_ijk_ras_matrices(img)
    ras_to_ijk_mat = np.asarray(ras_to_ijk_mat, dtype=float)

    print("# computing LoG sigma=1 ...")
    log_arr = sitk.GetArrayFromImage(
        sitk.LaplacianRecursiveGaussian(img, sigma=1.0)
    ).astype(np.float32)
    print("# computing white top-hat radius=3mm ...")
    wth_arr = white_top_hat_hu(img, radius_mm=3.0)

    # (name, array, polarity, amp_threshold, cyl_radius_mm, reducer).
    # Reducer=min for LoG (we want the deepest negative); max for CT/WTH.
    filters = [
        ("LoG_cyl2_min", log_arr, "min", 100.0, 2.0, "min"),
        ("CT_cyl2_max", ct_arr, "max", 500.0, 2.0, "max"),
        ("WTH_cyl1_max", wth_arr, "max", 150.0, 1.0, "max"),
        ("WTH_cyl2_max", wth_arr, "max", 150.0, 2.0, "max"),
    ]

    hdr = (
        f"{'shank':10s} {'nGT':>3s} "
        + " ".join(
            f"{fname:>18s}_med  n  ext"
            for fname, *_rest in filters
        )
    )
    print(hdr)
    medians_by_filter = {fname: [] for fname, *_rest in filters}
    match_rates_by_filter = {fname: [] for fname, *_rest in filters}

    debug_shank = os.environ.get("PROBE_DEBUG_SHANK", "").strip()
    for gt in shanks:
        start = np.asarray(gt.start_ras, dtype=float)
        end = np.asarray(gt.end_ras, dtype=float)
        axis_vec = end - start
        L = float(np.linalg.norm(axis_vec))
        if L < 10.0:
            continue
        du = axis_vec / L
        # Re-center axis on the actual contact cloud (PCA) to rule out
        # axis drift when the shanks.tsv line is slightly off.
        gt_pts_all = np.asarray(gt.contacts_ras, dtype=float)
        centroid = gt_pts_all.mean(axis=0)
        centered = gt_pts_all - centroid
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        pca_du = _unit(vh[0])
        # Keep it oriented start->end.
        if np.dot(pca_du, du) < 0:
            pca_du = -pca_du
        pca_proj = centered @ pca_du
        pca_start = centroid + pca_du * float(pca_proj.min())
        pca_end = centroid + pca_du * float(pca_proj.max())
        # Choose PCA axis to remove shanks-file drift.
        start = pca_start
        end = pca_end
        du = pca_du
        L = float(np.linalg.norm(end - start))
        # Extend sampling a little past both ends to check for extras.
        margin_mm = 5.0
        total_len = L + 2 * margin_mm
        n_steps = int(total_len / STEP_MM) + 1
        sample_start = start - margin_mm * du
        gt_pts = np.asarray(gt.contacts_ras, dtype=float)
        # GT arc-length = projection onto axis, expressed from sample_start.
        gt_arc = [(float(np.dot(p - sample_start, du))) for p in gt_pts]
        gt_arc = [a for a in gt_arc if 0.0 <= a <= total_len]

        row_cells = [f"{gt.shank:10s}", f"{len(gt_arc):3d}"]
        for fname, arr, polarity, amp, cyl, red in filters:
            prof = sample_line_profile(
                arr, ras_to_ijk_mat, sample_start, du, n_steps, STEP_MM,
                cyl_radius_mm=cyl, reducer=red,
            )
            peaks_mm = find_peaks(prof, polarity, STEP_MM, min_amplitude=amp)
            if debug_shank and gt.shank == debug_shank:
                print(f"\n  # {fname} profile stats: "
                      f"min={np.nanmin(prof):.1f} max={np.nanmax(prof):.1f} "
                      f"gt_arc={[f'{a:.1f}' for a in gt_arc]}")
                print(f"  # {fname} peaks={[f'{p:.1f}' for p in peaks_mm]}")
                prof_at_gt = []
                for a in gt_arc:
                    idx = int(round(a / STEP_MM))
                    if 0 <= idx < len(prof):
                        prof_at_gt.append(prof[idx])
                print(f"  # {fname} values_at_GT={[f'{v:.1f}' for v in prof_at_gt]}")
            errors, n_match, n_extra = match_peaks_to_gt(peaks_mm, gt_arc)
            med = float(np.median(errors)) if errors else float("nan")
            medians_by_filter[fname].append(med if errors else float("nan"))
            match_rate = n_match / max(1, len(gt_arc))
            match_rates_by_filter[fname].append(match_rate)
            row_cells.append(f"{med:10.2f}" if errors else f"{'-':>10s}")
            row_cells.append(f"{n_match:>3d}/{len(gt_arc):<3d}")
            row_cells.append(f"{n_extra:>6d}")
        print(" ".join(row_cells))

    print("# summary")
    for fname, *_rest in filters:
        meds = [v for v in medians_by_filter[fname] if np.isfinite(v)]
        rates = match_rates_by_filter[fname]
        med_all = float(np.median(meds)) if meds else float("nan")
        rate_all = float(np.mean(rates)) if rates else float("nan")
        print(
            f"# {fname}: median_per_contact_err={med_all:.2f}mm  "
            f"shank_mean_match_rate={rate_all:.2%}"
        )


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else "T22"
    run(sid)


if __name__ == "__main__":
    main()
