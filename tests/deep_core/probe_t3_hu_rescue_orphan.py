"""Diagnose the T3 high-band hu_rescue orphan that drops under rotation.

Per `project_contact_pitch_v1_dataset_rotation_2026-04-27.md`, T3 has
3 orphans of which 1 high-band (bolt=hu_rescue, n=8, pitch=3.77,
conf=0.86) drops under 30° per-trajectory rotation. The other 2 T3
orphans (1 high-band log + 1 medium none) survive.

Decide: is this drop correct (true FP we want demoted) or wrong
(real shank with unusual geometry)?

Diagnostics:
  1. List the orphan's full record (geometry, inlier blobs)
  2. Find the nearest GT shanks by midpoint distance — is any GT
     within an extended match window (15° / 15 mm) but missed by the
     production 10° / 8 mm window?
  3. Compare against the surviving T3 high-band orphan (bolt=log,
     n=12, pitch=3.61, conf=0.95) — what differs structurally?
  4. Slab analysis: how many LoG peaks survive at 0° vs 30°? Where
     do they sit relative to the trajectory axis?

Run:
    cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper && \\
        /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_t3_hu_rescue_orphan.py
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
from shank_engine import PipelineRegistry, register_builtin_pipelines
from eval_seeg_localization import (
    build_detection_context,
    iter_subject_rows,
    load_reference_ground_truth_shanks,
)

DATASET_ROOT = Path("/Users/ammar/Dropbox/thalamus_subjects/seeg_localization")

SLAB_PERP_MM = 12.0
SLAB_PAD_MM = 10.0
SLAB_VOXEL_MM = 1.0


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def _build_orthonormal_frame(axis_unit):
    a = np.asarray(axis_unit, dtype=float)
    a = a / np.linalg.norm(a)
    if abs(a[0]) < abs(a[1]) and abs(a[0]) < abs(a[2]):
        seed = np.array([1.0, 0.0, 0.0])
    elif abs(a[1]) < abs(a[2]):
        seed = np.array([0.0, 1.0, 0.0])
    else:
        seed = np.array([0.0, 0.0, 1.0])
    p1 = np.cross(a, seed); p1 = p1 / np.linalg.norm(p1)
    p2 = np.cross(a, p1);   p2 = p2 / np.linalg.norm(p2)
    return a, p1, p2


def _slab(img, start_ras, end_ras, rotation_deg=0.0):
    s = np.asarray(start_ras, dtype=float)
    e = np.asarray(end_ras, dtype=float)
    L = float(np.linalg.norm(e - s))
    axis, p1, p2 = _build_orthonormal_frame((e - s) / L)
    if rotation_deg != 0.0:
        a = float(np.deg2rad(rotation_deg))
        c, sn = float(np.cos(a)), float(np.sin(a))
        p1, p2 = c * p1 + sn * p2, -sn * p1 + c * p2

    nperp = int(round(2 * SLAB_PERP_MM / SLAB_VOXEL_MM)) + 1
    nalong = int(round((L + 2 * SLAB_PAD_MM) / SLAB_VOXEL_MM)) + 1
    slab_origin_ras = s - SLAB_PAD_MM * axis - SLAB_PERP_MM * p1 - SLAB_PERP_MM * p2

    M = np.eye(4)
    M[:3, 0] = p1 * SLAB_VOXEL_MM
    M[:3, 1] = p2 * SLAB_VOXEL_MM
    M[:3, 2] = axis * SLAB_VOXEL_MM
    M[:3, 3] = slab_origin_ras

    origin_lps = np.array([-slab_origin_ras[0], -slab_origin_ras[1], slab_origin_ras[2]])
    p1_lps = np.array([-p1[0], -p1[1], p1[2]])
    p2_lps = np.array([-p2[0], -p2[1], p2[2]])
    axis_lps = np.array([-axis[0], -axis[1], axis[2]])
    direction_lps = np.column_stack([p1_lps, p2_lps, axis_lps])

    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing((SLAB_VOXEL_MM,)*3)
    rs.SetSize((nperp, nperp, nalong))
    rs.SetOutputOrigin(tuple(float(v) for v in origin_lps))
    rs.SetOutputDirection(tuple(float(v) for v in direction_lps.flatten()))
    rs.SetInterpolator(sitk.sitkLinear)
    rs.SetDefaultPixelValue(-1024)
    slab = rs.Execute(img)
    if min(float(s_) for s_ in img.GetSpacing()) < cpfit.CANONICAL_SPACING_MM * 0.95 \
            and getattr(cpfit, "RAW_RESAMPLE_GAUSSIAN_SIGMA_MM", 0.0) > 0:
        slab = sitk.SmoothingRecursiveGaussian(slab, cpfit.RAW_RESAMPLE_GAUSSIAN_SIGMA_MM)
    return slab, M


def _slab_blobs_along_axis(img, start_ras, end_ras, rotation_deg):
    """Return list of (along_mm, perp_mm, log_amp) for each blob in the slab."""
    slab, M = _slab(img, start_ras, end_ras, rotation_deg)
    slab_clamped = sitk.Clamp(slab, lowerBound=-1024.0, upperBound=cpfit.HU_CLIP_MAX)
    log_sitk = sitk.LaplacianRecursiveGaussian(
        sitk.Cast(slab_clamped, sitk.sitkFloat32), sigma=cpfit.LOG_SIGMA_MM,
    )
    log_arr = sitk.GetArrayFromImage(log_sitk)
    blobs = cpfit.extract_blobs(log_arr, threshold=cpfit.LOG_BLOB_THRESHOLD)
    if not blobs:
        return []
    s = np.asarray(start_ras, dtype=float)
    e = np.asarray(end_ras, dtype=float)
    axis = (e - s) / max(1e-9, float(np.linalg.norm(e - s)))
    rows = []
    for b in blobs:
        kji = np.asarray(b["kji"], dtype=float)
        ijk = np.array([kji[2], kji[1], kji[0], 1.0])
        ras = (M @ ijk)[:3]
        d = ras - s
        along = float(d @ axis)
        perp = float(np.linalg.norm(d - along * axis))
        rows.append((along, perp, float(b["amp"]), float(b.get("n_vox", 0))))
    rows.sort()
    return rows


def main():
    rows = iter_subject_rows(DATASET_ROOT, {"T3"})
    if not rows:
        print("ERROR: T3 not in manifest"); return 1
    row = rows[0]

    gt, _ = load_reference_ground_truth_shanks(row)
    registry = PipelineRegistry()
    register_builtin_pipelines(registry)
    ctx, raw_img = build_detection_context(
        row["ct_path"], run_id="probe_t3_hu_rescue", config={}, extras={},
    )
    ctx["contact_pitch_v1_pitch_strategy"] = "auto"
    result = registry.run("contact_pitch_v1", ctx)
    trajs = list(result.get("trajectories") or [])
    print(f"T3: {len(gt)} GT shanks, pipeline emitted {len(trajs)} trajectories")

    # Greedy match to identify orphans (10°/8mm).
    pairs = []
    for gi, g in enumerate(gt):
        gs, ge = np.asarray(g.start_ras, dtype=float), np.asarray(g.end_ras, dtype=float)
        ga = _unit(ge - gs); gm = 0.5 * (gs + ge)
        for ti, t in enumerate(trajs):
            ts, te = np.asarray(t["start_ras"], dtype=float), np.asarray(t["end_ras"], dtype=float)
            ta = _unit(te - ts); tm = 0.5 * (ts + te)
            ang = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(ga, ta)))))))
            d = gm - tm; mid = float(np.linalg.norm(d - (d @ ta) * ta))
            if ang <= 10 and mid <= 8:
                pairs.append((ang + mid, gi, ti, ang, mid))
    pairs.sort(); used_g, used_t = set(), set(); matched = {}
    for _, gi, ti, ang, mid in pairs:
        if gi in used_g or ti in used_t: continue
        used_g.add(gi); used_t.add(ti)
        matched[ti] = (str(gt[gi].shank), ang, mid)
    print(f"matched: {len(matched)}/{len(gt)}")

    # Identify the two high-band orphans of interest.
    target_drop = None  # T3 hu_rescue
    target_keep = None  # T3 log
    for ti, t in enumerate(trajs):
        if ti in matched: continue
        if str(t.get("confidence_label", "?")) != "high": continue
        bolt = str(t.get("bolt_source", "?"))
        if bolt == "hu_rescue" and target_drop is None:
            target_drop = (ti, t)
        elif bolt == "log" and target_keep is None:
            target_keep = (ti, t)
    if target_drop is None:
        print("ERROR: could not find T3 hu_rescue high-band orphan"); return 1
    print(f"\nDROP target (hu_rescue): traj #{target_drop[0]}")
    print(f"KEEP target (log):       traj #{target_keep[0] if target_keep else '?'}")

    def _dump(label, t):
        s = np.asarray(t["start_ras"], dtype=float)
        e = np.asarray(t["end_ras"], dtype=float)
        print(f"\n=== {label} ===")
        print(f"  start_ras = {s.tolist()}")
        print(f"  end_ras   = {e.tolist()}")
        print(f"  midpoint  = {(0.5*(s+e)).tolist()}")
        print(f"  length    = {float(np.linalg.norm(e-s)):.2f} mm")
        print(f"  axis      = {_unit(e-s).tolist()}")
        for k in ("n_inliers", "amp_sum", "frangi_median_mm",
                  "original_median_pitch_mm", "contact_span_mm",
                  "dist_max_mm", "dist_mean_mm", "bolt_source",
                  "confidence", "confidence_label"):
            v = t.get(k)
            print(f"  {k:30s} = {v}")

    _dump("DROP candidate (T3 hu_rescue)", target_drop[1])
    if target_keep:
        _dump("KEEP reference (T3 log)", target_keep[1])

    # Question 1: is there a GT shank near the orphan's midpoint that
    # the production 10°/8mm window missed?
    print("\n=== nearest GT shanks to DROP candidate (sorted by midpoint distance) ===")
    drop = target_drop[1]
    ts = np.asarray(drop["start_ras"], dtype=float)
    te = np.asarray(drop["end_ras"], dtype=float)
    ta = _unit(te - ts); tm = 0.5 * (ts + te)
    rows_near = []
    for g in gt:
        gs = np.asarray(g.start_ras, dtype=float)
        ge = np.asarray(g.end_ras, dtype=float)
        ga = _unit(ge - gs); gm = 0.5 * (gs + ge)
        ang = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(ga, ta)))))))
        # Mid-point perpendicular distance to the trajectory axis.
        d = gm - tm
        perp = float(np.linalg.norm(d - (d @ ta) * ta))
        # Centroid-to-centroid distance
        cdist = float(np.linalg.norm(d))
        rows_near.append((cdist, ang, perp, str(g.shank), gm.tolist()))
    rows_near.sort()
    print(f"{'shank':>10s}  {'cent_d':>7s}  {'ang°':>5s}  {'perp':>5s}  midpoint_ras")
    for cd, ang, perp, name, gm in rows_near[:5]:
        flag = ""
        if ang <= 10 and perp <= 8:
            flag = " (inside production match window!)"
        elif ang <= 15 and perp <= 15:
            flag = " (inside extended window 15/15)"
        print(f"{name:>10s}  {cd:>7.2f}  {ang:>5.1f}  {perp:>5.1f}  "
              f"[{gm[0]:.1f},{gm[1]:.1f},{gm[2]:.1f}]{flag}")

    # Question 2: does any GT shank sit on the orphan's axis?
    # If yes, the orphan IS that GT shank but the production window
    # rejected it because of a long start/end offset.

    # Question 3: slab blob analysis at 0° and 30°.
    for label, t in (("DROP candidate (T3 hu_rescue)", target_drop[1]),
                     ("KEEP reference (T3 log)", target_keep[1] if target_keep else None)):
        if t is None: continue
        s = np.asarray(t["start_ras"], dtype=float)
        e = np.asarray(t["end_ras"], dtype=float)
        L = float(np.linalg.norm(e - s))
        print(f"\n=== slab LoG blobs along axis: {label} (L={L:.1f} mm) ===")
        for ang in (0.0, 30.0):
            blobs = _slab_blobs_along_axis(raw_img, s, e, rotation_deg=ang)
            blobs_in = [b for b in blobs if b[1] <= cpfit.PERP_TOL_MM
                        and -SLAB_PAD_MM <= b[0] <= L + SLAB_PAD_MM]
            print(f"\n  rotation={ang:>4.1f}°: total slab blobs={len(blobs)}, "
                  f"inliers (perp<={cpfit.PERP_TOL_MM}, along in [-{SLAB_PAD_MM:.0f}, L+{SLAB_PAD_MM:.0f}])={len(blobs_in)}")
            if blobs:
                print(f"    {'along':>6s} {'perp':>5s} {'amp':>6s} {'n_vox':>5s}")
                for along, perp, amp, n_vox in blobs[:30]:
                    mark = "*" if perp <= cpfit.PERP_TOL_MM and -SLAB_PAD_MM <= along <= L + SLAB_PAD_MM else " "
                    print(f"  {mark} {along:>6.1f} {perp:>5.2f} {amp:>6.0f} {n_vox:>5.0f}")
                if len(blobs) > 30:
                    print(f"    ... ({len(blobs) - 30} more)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
