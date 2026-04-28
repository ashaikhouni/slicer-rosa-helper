"""Dataset-wide per-trajectory rotational-invariance probe.

Extension of `probe_per_trajectory_rotation.py` (which only ran on T1)
to all 22 evaluable subjects. For each candidate trajectory emitted by
the production pipeline, extract a slab around the shank axis, rotate
the slab 30° about the shank axis, re-run blob extraction + chain
test, and record whether the chain survives.

Hypothesis under test (from
`project_contact_pitch_v1_rotation_invariance.md`): voxel-grid-aliasing
FPs flicker under rotation while real shanks remain invariant. T1 showed
4/7 medium-band orphans drop and all 12 matched preserved. Need to know
if this generalizes — if so, `rotation_robustness` is a strong score
component candidate.

Output for each subject:
  - matched_invariant / matched_dropped
  - orphan_invariant / orphan_dropped (broken down by confidence band)

Aggregated:
  - matched survival rate (should be ~100 %)
  - per-band orphan drop rate
  - what the new orphan tally would be if rotation = required gate

Run:
    cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper && \\
        /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_dataset_rotation_invariance.py
"""
from __future__ import annotations

import sys
import time
from collections import Counter, defaultdict
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
EXCLUDE = {"T17", "T19", "T21"}

MATCH_ANGLE_DEG = 10.0
MATCH_MID_MM = 8.0

# Slab geometry (matches probe_per_trajectory_rotation.py).
SLAB_PERP_MM = 12.0
SLAB_PAD_MM = 10.0
SLAB_VOXEL_MM = 1.0
ROTATION_DEG = 30.0


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
                pairs.append((ang + mid_d, gi, ti))
    pairs.sort(key=lambda p: p[0])
    used_g, used_t = set(), set()
    matched_ti = {}
    for _s, gi, ti in pairs:
        if gi in used_g or ti in used_t:
            continue
        used_g.add(gi); used_t.add(ti)
        matched_ti[ti] = str(gt_shanks[gi].shank)
    return matched_ti


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


def _resample_slab_around_trajectory(img, start_ras, end_ras, rotation_deg=0.0):
    """Slab whose K-axis aligns with shank axis; perp1/perp2 rotated by
    ``rotation_deg`` about that axis."""
    s = np.asarray(start_ras, dtype=float)
    e = np.asarray(end_ras, dtype=float)
    L = float(np.linalg.norm(e - s))
    if L < 1e-3:
        return None, None
    axis, p1, p2 = _build_orthonormal_frame((e - s) / L)
    if rotation_deg != 0.0:
        a = float(np.deg2rad(rotation_deg))
        c, sn = float(np.cos(a)), float(np.sin(a))
        p1_new = c * p1 + sn * p2
        p2_new = -sn * p1 + c * p2
        p1, p2 = p1_new, p2_new

    nperp = int(round(2 * SLAB_PERP_MM / SLAB_VOXEL_MM)) + 1
    nalong = int(round((L + 2 * SLAB_PAD_MM) / SLAB_VOXEL_MM)) + 1
    size_xyz = (nperp, nperp, nalong)

    slab_origin_ras = (
        s - SLAB_PAD_MM * axis - SLAB_PERP_MM * p1 - SLAB_PERP_MM * p2
    )

    slab_ijk_to_ras = np.eye(4)
    slab_ijk_to_ras[:3, 0] = p1 * SLAB_VOXEL_MM
    slab_ijk_to_ras[:3, 1] = p2 * SLAB_VOXEL_MM
    slab_ijk_to_ras[:3, 2] = axis * SLAB_VOXEL_MM
    slab_ijk_to_ras[:3, 3] = slab_origin_ras

    # SITK uses LPS internally.
    slab_origin_lps = np.array([-slab_origin_ras[0], -slab_origin_ras[1], slab_origin_ras[2]])
    p1_lps = np.array([-p1[0], -p1[1], p1[2]])
    p2_lps = np.array([-p2[0], -p2[1], p2[2]])
    axis_lps = np.array([-axis[0], -axis[1], axis[2]])
    direction_lps = np.column_stack([p1_lps, p2_lps, axis_lps])

    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing((SLAB_VOXEL_MM, SLAB_VOXEL_MM, SLAB_VOXEL_MM))
    rs.SetSize(size_xyz)
    rs.SetOutputOrigin(tuple(float(v) for v in slab_origin_lps))
    rs.SetOutputDirection(tuple(float(v) for v in direction_lps.flatten()))
    rs.SetInterpolator(sitk.sitkLinear)
    rs.SetDefaultPixelValue(-1024)
    slab = rs.Execute(img)

    # Mirror production's anti-aliasing: when raw input is sub-mm, the
    # canonical resample applies a Gaussian sigma=0.7 mm post-resample
    # to suppress sub-mm aliasing. Apply the same here so the slab's
    # LoG response matches production's.
    raw_min_spacing = min(float(s_) for s_ in img.GetSpacing())
    if raw_min_spacing < cpfit.CANONICAL_SPACING_MM * 0.95 \
            and getattr(cpfit, "RAW_RESAMPLE_GAUSSIAN_SIGMA_MM", 0.0) > 0:
        slab = sitk.SmoothingRecursiveGaussian(
            slab, cpfit.RAW_RESAMPLE_GAUSSIAN_SIGMA_MM
        )

    return slab, slab_ijk_to_ras


def _walk_chain_in_slab(slab_arr_kji, slab_ijk_to_ras, start_ras, end_ras):
    slab_sitk = sitk.GetImageFromArray(slab_arr_kji)
    slab_sitk.SetSpacing((SLAB_VOXEL_MM, SLAB_VOXEL_MM, SLAB_VOXEL_MM))
    slab_clamped = sitk.Clamp(slab_sitk, lowerBound=-1024.0, upperBound=cpfit.HU_CLIP_MAX)
    log_sitk = sitk.LaplacianRecursiveGaussian(
        sitk.Cast(slab_clamped, sitk.sitkFloat32), sigma=cpfit.LOG_SIGMA_MM,
    )
    log_arr = sitk.GetArrayFromImage(log_sitk)
    blobs = cpfit.extract_blobs(log_arr, threshold=cpfit.LOG_BLOB_THRESHOLD)
    if not blobs:
        return None
    pts_kji = np.array([np.asarray(b["kji"], dtype=float) for b in blobs])
    pts_ijk = np.stack([pts_kji[:, 2], pts_kji[:, 1], pts_kji[:, 0]], axis=1)
    h = np.concatenate([pts_ijk, np.ones((pts_ijk.shape[0], 1))], axis=1)
    pts_ras = (slab_ijk_to_ras @ h.T).T[:, :3]

    s = np.asarray(start_ras, dtype=float)
    e = np.asarray(end_ras, dtype=float)
    L = float(np.linalg.norm(e - s))
    axis = (e - s) / max(1e-9, L)
    diffs = pts_ras - s
    along = diffs @ axis
    perp_vec = diffs - np.outer(along, axis)
    perp = np.linalg.norm(perp_vec, axis=1)
    mask = (
        (perp <= cpfit.PERP_TOL_MM)
        & (along >= -SLAB_PAD_MM)
        & (along <= L + SLAB_PAD_MM)
    )
    n_inliers = int(mask.sum())
    if n_inliers < cpfit.MIN_BLOBS_PER_LINE:
        return None
    span = float(along[mask].max() - along[mask].min())
    return {"n_inliers": n_inliers, "span_mm": span}


def _classify(traj, img):
    s_ras = np.asarray(traj["start_ras"], dtype=float)
    e_ras = np.asarray(traj["end_ras"], dtype=float)
    slab_r, slab_to_ras_r = _resample_slab_around_trajectory(
        img, s_ras, e_ras, rotation_deg=ROTATION_DEG,
    )
    if slab_r is None:
        return False, 0
    slab_r_arr = sitk.GetArrayFromImage(slab_r)
    chain = _walk_chain_in_slab(slab_r_arr, slab_to_ras_r, s_ras, e_ras)
    return (chain is not None), (chain["n_inliers"] if chain else 0)


def main():
    registry = PipelineRegistry()
    register_builtin_pipelines(registry)

    rows = iter_subject_rows(DATASET_ROOT, None)
    rows = [r for r in rows if str(r["subject_id"]) not in EXCLUDE]
    rows.sort(key=lambda r: int(str(r["subject_id"]).lstrip("T")))

    # Per-subject results.
    per_subject = []
    # Aggregate buckets keyed by (kind, band). kind in {"matched","orphan"}.
    agg_total = defaultdict(int)
    agg_inv = defaultdict(int)
    # Per-orphan dump for inspection.
    orphan_rows = []

    print(f"running pipeline + per-trajectory rotation probe on "
          f"{len(rows)} subjects (rotation={ROTATION_DEG}°)\n")
    t_start_all = time.time()

    for row in rows:
        subject_id = str(row["subject_id"])
        t_sub = time.time()

        gt, _ = load_reference_ground_truth_shanks(row)
        ctx, raw_img = build_detection_context(
            row["ct_path"],
            run_id=f"probe_rot_{subject_id}",
            config={},
            extras={},
        )
        ctx["contact_pitch_v1_pitch_strategy"] = "auto"
        result = registry.run("contact_pitch_v1", ctx)
        if str(result.get("status", "ok")).lower() == "error":
            err = dict(result.get("error") or {})
            print(f"  {subject_id}: PIPELINE ERROR {err.get('message')}")
            continue
        trajs = list(result.get("trajectories") or [])
        matched = _greedy_match(gt, trajs)

        n_matched_inv = 0
        n_matched_drop = 0
        n_orph_by_band = Counter()
        n_orph_inv_by_band = Counter()

        for ti, t in enumerate(trajs):
            band = str(t.get("confidence_label", "?"))
            inv, n_r = _classify(t, raw_img)
            if ti in matched:
                kind = "matched"
                if inv:
                    n_matched_inv += 1
                else:
                    n_matched_drop += 1
            else:
                kind = "orphan"
                n_orph_by_band[band] += 1
                if inv:
                    n_orph_inv_by_band[band] += 1
                orphan_rows.append({
                    "subject": subject_id,
                    "band": band,
                    "bolt_source": str(t.get("bolt_source", "?")),
                    "n_inliers": int(t.get("n_inliers", 0)),
                    "n_inliers_rot": n_r,
                    "pitch_mm": float(t.get("original_median_pitch_mm", 0.0)),
                    "confidence": float(t.get("confidence", 0.0)),
                    "invariant": bool(inv),
                })
            agg_total[(kind, band)] += 1
            if inv:
                agg_inv[(kind, band)] += 1

        n_orph = sum(n_orph_by_band.values())
        n_orph_inv = sum(n_orph_inv_by_band.values())
        per_subject.append({
            "subject": subject_id,
            "gt": len(gt),
            "matched": len(matched),
            "matched_inv": n_matched_inv,
            "matched_drop": n_matched_drop,
            "orphans": n_orph,
            "orphan_inv": n_orph_inv,
            "orphan_drop": n_orph - n_orph_inv,
            "orph_by_band": dict(n_orph_by_band),
            "orph_inv_by_band": dict(n_orph_inv_by_band),
        })
        elapsed = time.time() - t_sub
        print(
            f"  {subject_id:>4s}: gt={len(gt):>2d}  matched={len(matched):>2d} "
            f"(inv={n_matched_inv:>2d}, drop={n_matched_drop:>2d})  "
            f"orphans={n_orph:>2d} (inv={n_orph_inv}, drop={n_orph - n_orph_inv})  "
            f"[{elapsed:.1f}s]"
        )

    elapsed_all = time.time() - t_start_all
    print(f"\nTotal probe wall time: {elapsed_all:.1f}s")

    # ------------- Aggregate report -------------
    print("\n" + "=" * 70)
    print("AGGREGATE: rotation-survival rates by (kind, band)")
    print("=" * 70)
    print(f"{'kind':>8s} {'band':>8s}  {'n_total':>8s}  {'n_inv':>8s}  {'inv_rate':>9s}")
    keys_sorted = sorted(agg_total.keys(), key=lambda k: (k[0], k[1]))
    for k in keys_sorted:
        tot = agg_total[k]
        inv = agg_inv.get(k, 0)
        rate = inv / tot if tot else 0.0
        kind, band = k
        print(f"{kind:>8s} {band:>8s}  {tot:>8d}  {inv:>8d}  {rate:>9.1%}")

    # Bottom-line: matched preserved? orphans dropped?
    tot_matched = sum(v for (k, _), v in agg_total.items() if k == "matched")
    inv_matched = sum(v for (k, _), v in agg_inv.items() if k == "matched")
    tot_orphan = sum(v for (k, _), v in agg_total.items() if k == "orphan")
    inv_orphan = sum(v for (k, _), v in agg_inv.items() if k == "orphan")
    print()
    print(f"matched: {inv_matched}/{tot_matched} preserved "
          f"({inv_matched/max(1,tot_matched):.1%}); "
          f"{tot_matched - inv_matched} would be lost if rotation=required")
    print(f"orphans: {tot_orphan - inv_orphan}/{tot_orphan} would be dropped "
          f"({(tot_orphan - inv_orphan)/max(1,tot_orphan):.1%})")

    # Score-component preview: assuming the gate kept everything currently
    # in 'high' regardless and only filtered medium/low.
    high_match_lost = sum(
        v_inv_diff for (k, b), v_inv_diff in (
            (key, agg_total[key] - agg_inv.get(key, 0)) for key in agg_total
        )
        if k == "matched" and b == "high"
    )
    med_orph_dropped = sum(
        agg_total[k] - agg_inv.get(k, 0)
        for k in agg_total if k == ("orphan", "medium")
    )
    print(f"\nIf gate = 'medium-band orphans must be rotation-invariant':")
    print(f"  matched-high lost: {high_match_lost} (should be 0)")
    print(f"  medium-band orphans dropped: {med_orph_dropped}")

    # ------------- Per-orphan dump -------------
    print("\n" + "=" * 70)
    print(f"All {len(orphan_rows)} orphans, sorted by (band, invariant, n_rot)")
    print("=" * 70)
    print(f"{'subj':>4s} {'band':>6s} {'inv':>4s} {'bolt':>11s} "
          f"{'n_orig':>6s} {'n_rot':>5s} {'pitch':>5s} {'conf':>5s}")
    band_order = {"high": 0, "medium": 1, "low": 2, "?": 3}
    orphan_rows.sort(key=lambda r: (
        band_order.get(r["band"], 99),
        not r["invariant"],
        -r["n_inliers_rot"],
    ))
    for r in orphan_rows:
        marker = "INV" if r["invariant"] else "DROP"
        print(
            f"{r['subject']:>4s} {r['band']:>6s} {marker:>4s} "
            f"{r['bolt_source']:>11s} "
            f"{r['n_inliers']:>6d} {r['n_inliers_rot']:>5d} "
            f"{r['pitch_mm']:>5.2f} {r['confidence']:>5.2f}"
        )

    # ------------- Per-subject summary -------------
    print("\n" + "=" * 70)
    print("Per-subject survival summary")
    print("=" * 70)
    print(f"{'subj':>4s} {'gt':>3s} {'mat':>3s} {'mInv':>4s} {'mDrop':>5s} "
          f"{'orph':>4s} {'oInv':>4s} {'oDrop':>5s}")
    for s in per_subject:
        print(
            f"{s['subject']:>4s} {s['gt']:>3d} {s['matched']:>3d} "
            f"{s['matched_inv']:>4d} {s['matched_drop']:>5d} "
            f"{s['orphans']:>4d} {s['orphan_inv']:>4d} {s['orphan_drop']:>5d}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
