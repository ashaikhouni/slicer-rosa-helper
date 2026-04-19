"""Smoke-test the ``rosa_core.contact_peak_fit`` engine against GT.

Runs ``detect_contacts_on_axis`` on every GT shank for a subject. Uses
the PCA line through contact positions as the axis input (stand-in for
Auto-Fit output), computes LoG σ=1 with SimpleITK, and reports:

  * assigned model id vs expected (loose — shanks don't carry model id)
  * median per-contact position error (RAS) vs GT positions

Usage:
  /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
    tests/deep_core/probe_contact_peak_engine.py [T22|T2]
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


def _pca_endpoints(pts):
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    du = _unit(vh[0])
    proj = centered @ du
    p_min = centroid + du * float(proj.min())
    p_max = centroid + du * float(proj.max())
    return p_min, p_max, du


def run(subject_id):
    import SimpleITK as sitk
    from shank_core.io import image_ijk_ras_matrices
    from eval_seeg_localization import iter_subject_rows, load_ground_truth_shanks

    from rosa_core import load_electrode_library, model_map
    from rosa_core.contact_peak_fit import (
        candidate_ids_for_vendors,
        detect_contacts_on_axis,
    )

    rows = iter_subject_rows(DATASET_ROOT, {subject_id})
    if not rows:
        print(f"# no rows for {subject_id}")
        return
    row = rows[0]
    ct_path = row["ct_path"]
    shanks = load_ground_truth_shanks(row["labels_path"], row.get("shanks_path"))
    print(f"# subject={subject_id} ct={ct_path} n_shanks={len(shanks)}")

    img = sitk.ReadImage(ct_path)
    _, ras_to_ijk_mat = image_ijk_ras_matrices(img)
    ras_to_ijk_mat = np.asarray(ras_to_ijk_mat, dtype=float)

    print("# computing LoG sigma=1 ...")
    log_arr = sitk.GetArrayFromImage(
        sitk.LaplacianRecursiveGaussian(img, sigma=1.0)
    ).astype(np.float32)

    library = load_electrode_library()
    models_by_id = model_map(library)
    candidate_ids = candidate_ids_for_vendors(
        models_by_id, vendors=["DIXI", "PMT"],
    )

    print(
        f"{'shank':10s} {'nGT':>3s} {'modelId':>14s} {'nSlots':>6s} "
        f"{'nMatched':>8s} {'coverage':>8s} {'meanRes':>8s} "
        f"{'medErrRAS':>9s} {'maxErrRAS':>9s} {'rejected_reason':>40s}"
    )

    pos_errors_all = []
    for gt in shanks:
        pts = np.asarray(gt.contacts_ras, dtype=float)
        if pts.shape[0] < 3:
            continue
        # Use the GT shanks.tsv start/end (contact-PCA endpoints) as
        # the axis. Extend by 1.5 mm on the shallow side (smaller than
        # inter-contact pitch so spurious bolt-side peaks don't become
        # "slot 0" in the fit) and 3 mm on the deep side (space past
        # the deepest contact for the tip position).
        s_ras = np.asarray(gt.start_ras, dtype=float)
        e_ras = np.asarray(gt.end_ras, dtype=float)
        du = _unit(e_ras - s_ras)
        start = s_ras - du * 1.5
        end = e_ras + du * 3.0

        result = detect_contacts_on_axis(
            start, end, log_arr, ras_to_ijk_mat,
            models_by_id, candidate_ids=candidate_ids,
        )
        debug_shank = os.environ.get("PROBE_DEBUG_SHANK", "").strip()
        if debug_shank and gt.shank == debug_shank:
            from rosa_core.contact_peak_fit import (
                sample_axis_profile, detect_peaks_1d,
            )
            arc_mm, prof = sample_axis_profile(
                log_arr, ras_to_ijk_mat, start, end,
            )
            peaks = detect_peaks_1d(prof, arc_mm[1] - arc_mm[0])
            axis_len = float(np.linalg.norm(end - start))
            du = (end - start) / axis_len
            gt_arc = [float(np.dot(p - start, du)) for p in gt.contacts_ras]
            print(f"  # axis_len_mm={axis_len:.2f}")
            print(f"  # peaks_mm ({len(peaks)}): {[f'{p:.1f}' for p in peaks]}")
            print(f"  # gt_arc ({len(gt_arc)}): {[f'{p:.1f}' for p in gt_arc]}")
            print(f"  # detected_ras_arc:")
            for i, (pr, det) in enumerate(zip(result.positions_ras, result.peak_detected)):
                parc = float(np.dot(np.asarray(pr) - start, du))
                print(f"     slot {i+1}: arc={parc:.1f} detected={det}")
        expected = gt.contact_count

        if not result.model_id:
            print(
                f"{gt.shank:10s} {expected:3d} {'-':>14s} {'-':>6s} "
                f"{'-':>8s} {'-':>8s} {'-':>8s} {'-':>9s} {'-':>9s} "
                f"{result.rejected_reason[:40]:>40s}"
            )
            continue

        # Score only slots that actually came from a detected peak —
        # nominal-fallback slots can be far from any real contact when
        # the model has more slots than the subject's electrode.
        det_pts_all = np.asarray(result.positions_ras, dtype=float)
        mask = np.asarray(result.peak_detected, dtype=bool)
        det_pts = det_pts_all[mask]
        if det_pts.size == 0:
            per_errors = []
        else:
            d = np.linalg.norm(det_pts[:, None, :] - pts[None, :, :], axis=2)
            used = set()
            per_errors = []
            for i in range(det_pts.shape[0]):
                order = np.argsort(d[i])
                for j in order:
                    if int(j) not in used:
                        used.add(int(j))
                        per_errors.append(float(d[i, int(j)]))
                        break
        per_arr = np.asarray(per_errors, dtype=float) if per_errors else np.array([float("nan")])
        med_err = float(np.nanmedian(per_arr)) if per_errors else float("nan")
        max_err = float(np.nanmax(per_arr)) if per_errors else float("nan")
        pos_errors_all.extend(per_errors)
        cov = result.n_matched / max(1, result.n_model_slots)
        print(
            f"{gt.shank:10s} {expected:3d} {result.model_id:>14s} "
            f"{result.n_model_slots:>6d} {result.n_matched:>8d} "
            f"{cov:>8.2%} {result.mean_residual_mm:>7.2f}mm "
            f"{med_err:>8.2f}mm {max_err:>8.2f}mm {'':>40s}"
        )
    if pos_errors_all:
        arr = np.asarray(pos_errors_all, dtype=float)
        print(
            f"# overall median per-contact RAS error: {float(np.median(arr)):.2f} mm "
            f"(n={arr.size})"
        )


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else "T22"
    run(sid)


if __name__ == "__main__":
    main()
