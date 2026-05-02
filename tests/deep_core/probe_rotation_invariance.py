"""Rotational-invariance probe.

Apply a rigid rotation to T1's CT, run the full pipeline on both the
original and the rotated volume, transform the rotated trajectories
back into the original frame, and compare. A perfectly orientation-
invariant pipeline emits the same set of trajectories (modulo
numerical drift from the resampling).

Hypothesis: the high-band (real-shank) emissions are stable; the
medium-band (FP) tail shifts because the LoG-blob extractor's
3x3x3 Box kernel is axis-aligned in voxel space, so under rotation
the suppression neighbourhood sees different voxel offsets.

Run:
    cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper && \\
        /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_rotation_invariance.py
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

from rosa_detect import contact_pitch_v1_fit as cpfit
from shank_core.io import image_ijk_ras_matrices
from eval_seeg_localization import (
    iter_subject_rows,
    load_reference_ground_truth_shanks,
)


DATASET_ROOT = Path("/Users/ammar/Dropbox/thalamus_subjects/seeg_localization")

ROTATIONS_DEG = [0.0, 15.0, 30.0, 45.0, 90.0]


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def _rotate_volume(img: sitk.Image, angle_deg: float, axis: tuple = (0, 0, 1)) -> tuple[sitk.Image, sitk.Transform]:
    """Apply a rigid rotation about the volume center. Returns the
    rotated image (same spacing, larger bbox to fit corners) and the
    transform mapping original RAS -> rotated RAS.
    """
    if angle_deg == 0:
        return img, sitk.Transform(3, sitk.sitkIdentity)

    sz = img.GetSize()
    sp = img.GetSpacing()
    org = np.asarray(img.GetOrigin(), dtype=float)
    dr = np.asarray(img.GetDirection(), dtype=float).reshape(3, 3)

    # Find physical center of input volume.
    center_ijk = np.array([(sz[0]-1)/2., (sz[1]-1)/2., (sz[2]-1)/2.])
    center_lps = org + dr @ (center_ijk * np.asarray(sp))

    # Build Euler rotation about that center.
    angle_rad = float(np.deg2rad(angle_deg))
    tx = sitk.Euler3DTransform()
    tx.SetCenter(tuple(center_lps.tolist()))
    if axis == (1, 0, 0):
        tx.SetRotation(angle_rad, 0.0, 0.0)
    elif axis == (0, 1, 0):
        tx.SetRotation(0.0, angle_rad, 0.0)
    else:
        tx.SetRotation(0.0, 0.0, angle_rad)

    # Resample with the SAME spacing into a bbox large enough to fit
    # the rotated corners. We expand the output size by ~1.4x so 45 deg
    # rotation doesn't crop.
    expand = 1.6
    out_size = [int(round(sz[i] * expand)) for i in range(3)]
    # Center the output on the same physical center.
    out_origin_ijk = np.array([(out_size[0]-1)/2., (out_size[1]-1)/2., (out_size[2]-1)/2.])
    new_origin_lps = center_lps - dr @ (out_origin_ijk * np.asarray(sp))

    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing(sp)
    rs.SetSize(out_size)
    rs.SetOutputOrigin(tuple(new_origin_lps.tolist()))
    rs.SetOutputDirection(tuple(dr.flatten().tolist()))
    rs.SetInterpolator(sitk.sitkLinear)
    rs.SetDefaultPixelValue(-1024)
    # SITK uses inverse mapping: T maps fixed (output) -> moving (input).
    rs.SetTransform(tx.GetInverse())
    out = rs.Execute(img)
    return out, tx


def _run_pipeline(img: sitk.Image) -> list:
    ijk_to_ras_mat, ras_to_ijk_mat = image_ijk_ras_matrices(img)
    return cpfit.run_two_stage_detection(
        img, ijk_to_ras_mat, ras_to_ijk_mat,
        return_features=False, pitch_strategy="auto",
    )


def _greedy_match_two_traj_sets(trajs_a, trajs_b, ang_tol=5.0, mid_tol=5.0):
    """Greedy 1:1 match between two trajectory sets in the SAME RAS
    frame. Returns dict a_idx -> b_idx for matched pairs.
    """
    pairs = []
    for ai, ta in enumerate(trajs_a):
        as_ = np.asarray(ta["start_ras"], dtype=float)
        ae = np.asarray(ta["end_ras"], dtype=float)
        a_axis = _unit(ae - as_)
        a_mid = 0.5 * (as_ + ae)
        for bi, tb in enumerate(trajs_b):
            bs = np.asarray(tb["start_ras"], dtype=float)
            be = np.asarray(tb["end_ras"], dtype=float)
            b_axis = _unit(be - bs)
            b_mid = 0.5 * (bs + be)
            ang = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(a_axis, b_axis)))))))
            d = a_mid - b_mid
            p = d - (d @ b_axis) * b_axis
            mid = float(np.linalg.norm(p))
            if ang <= ang_tol and mid <= mid_tol:
                pairs.append((ang + mid, ai, bi))
    pairs.sort()
    used_a, used_b = set(), set()
    matches = {}
    for _s, ai, bi in pairs:
        if ai in used_a or bi in used_b:
            continue
        used_a.add(ai); used_b.add(bi)
        matches[ai] = bi
    return matches


def _ras_back_to_original(ras_in_rotated, transform):
    """Map a RAS point in the rotated frame back to the original frame.

    SITK transforms are LPS-internal; inputs are RAS. Convert RAS->LPS,
    apply transform.GetInverse() (rotated_LPS -> original_LPS), then
    LPS->RAS.
    """
    p_ras = np.asarray(ras_in_rotated, dtype=float)
    p_lps = np.array([-p_ras[0], -p_ras[1], p_ras[2]])
    p_lps_orig = transform.GetInverse().TransformPoint(tuple(p_lps.tolist()))
    return np.array([-p_lps_orig[0], -p_lps_orig[1], p_lps_orig[2]])


def main():
    rows = list(iter_subject_rows(DATASET_ROOT, {"T1"}))
    if not rows:
        print("ERROR: T1 not found"); return 1
    row = rows[0]
    gt, _ = load_reference_ground_truth_shanks(row)
    img_orig = sitk.ReadImage(str(row["ct_path"]))
    print(f"T1: {len(gt)} GT shanks, input spacing={img_orig.GetSpacing()}, size={img_orig.GetSize()}")

    # Reference: original orientation.
    print(f"\n=== running pipeline on UNROTATED T1 ===")
    trajs_orig = _run_pipeline(img_orig)
    print(f"  emitted {len(trajs_orig)} trajectories")

    print(f"\n{'angle':>5s} {'emit':>5s} {'matched_orig':>12s} {'extra_in_rot':>12s} {'missing_in_rot':>14s}  {'comments':<30s}")
    for angle in ROTATIONS_DEG:
        if angle == 0.0:
            print(f"{angle:>5.0f} {len(trajs_orig):>5d} {'(reference)':>12s} {'-':>12s} {'-':>14s}")
            continue

        img_rot, tx = _rotate_volume(img_orig, angle, axis=(0, 0, 1))
        try:
            trajs_rot_in_rot = _run_pipeline(img_rot)
        except Exception as exc:
            print(f"{angle:>5.0f} ERROR: {exc}")
            continue

        # Bring rotated trajectories back to original RAS for matching.
        trajs_rot_in_orig = []
        for t in trajs_rot_in_rot:
            new_t = dict(t)
            new_t["start_ras"] = _ras_back_to_original(t["start_ras"], tx)
            new_t["end_ras"] = _ras_back_to_original(t["end_ras"], tx)
            trajs_rot_in_orig.append(new_t)

        # Pair-match the two sets in the original frame.
        m = _greedy_match_two_traj_sets(trajs_orig, trajs_rot_in_orig, ang_tol=5.0, mid_tol=5.0)
        n_matched = len(m)
        n_extra = len(trajs_rot_in_orig) - n_matched  # in rotated but not in orig
        n_missing = len(trajs_orig) - n_matched       # in orig but not in rotated
        print(f"{angle:>5.0f} {len(trajs_rot_in_rot):>5d} {n_matched:>12d} {n_extra:>12d} {n_missing:>14d}")

        # Dump the smallest angle (15) AND 90 deg lines that diverge.
        if angle in (ROTATIONS_DEG[1], 90.0):
            print(f"\n  --- detail at {angle} deg ---")
            unmatched_orig = [i for i in range(len(trajs_orig)) if i not in m]
            unmatched_rot = [i for i in range(len(trajs_rot_in_orig)) if i not in m.values()]
            print(f"\n  unmatched-in-orig (lines that DISAPPEAR under rotation):")
            for i in unmatched_orig:
                t = trajs_orig[i]
                print(f"    [{i:>2d}] conf={float(t.get('confidence',0)):.2f}({t.get('confidence_label','?'):>6s}) "
                      f"bolt={t.get('bolt_source','?'):>10s} n={int(t.get('n_inliers',0)):>2d} "
                      f"pitch={float(t.get('original_median_pitch_mm',0)):.2f}")
            print(f"\n  unmatched-in-rotated (lines that APPEAR only after rotation):")
            for i in unmatched_rot:
                t = trajs_rot_in_orig[i]
                print(f"    [{i:>2d}] conf={float(t.get('confidence',0)):.2f}({t.get('confidence_label','?'):>6s}) "
                      f"bolt={t.get('bolt_source','?'):>10s} n={int(t.get('n_inliers',0)):>2d} "
                      f"pitch={float(t.get('original_median_pitch_mm',0)):.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
