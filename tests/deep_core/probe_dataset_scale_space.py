"""Dataset-wide scale-space invariance probe.

For each candidate trajectory, re-run blob extraction on the slab at
LoG sigma in {0.8, 1.0, 1.2} mm. Real contacts produce peaks at all
three sigmas (sharp 3D objects with size ~1 mm); voxel-aliased FPs
typically only respond at the sigma that aligns with the cube kernel.

Hypothesis (option C in `project_contact_pitch_v1_rotation_invariance.md`):
scale-space invariance is the natural complement to rotation
invariance — should kill the rotation survivors (k=2-pitch T1 medium
orphans, T3 medium n=5).

Output mirrors the rotation probe so we can compose the two signals.

Run:
    cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper && \\
        /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_dataset_scale_space.py
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

SLAB_PERP_MM = 12.0
SLAB_PAD_MM = 10.0
SLAB_VOXEL_MM = 1.0

# Scale-space sigmas: production sigma = 1.0 (cpfit.LOG_SIGMA_MM).
# Bracket it with one sigma below and one above (~20 % bandwidth).
SIGMAS_MM = (0.8, 1.0, 1.2)


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
            if ang <= MATCH_ANGLE_DEG and mid <= MATCH_MID_MM:
                pairs.append((ang + mid, gi, ti))
    pairs.sort(); used_g, used_t = set(), set(); m = {}
    for _s, gi, ti in pairs:
        if gi in used_g or ti in used_t: continue
        used_g.add(gi); used_t.add(ti); m[ti] = str(gt_shanks[gi].shank)
    return m


def _build_orthonormal_frame(axis_unit):
    a = np.asarray(axis_unit, dtype=float); a = a / np.linalg.norm(a)
    if abs(a[0]) < abs(a[1]) and abs(a[0]) < abs(a[2]):
        seed = np.array([1.0, 0.0, 0.0])
    elif abs(a[1]) < abs(a[2]):
        seed = np.array([0.0, 1.0, 0.0])
    else:
        seed = np.array([0.0, 0.0, 1.0])
    p1 = np.cross(a, seed); p1 = p1 / np.linalg.norm(p1)
    p2 = np.cross(a, p1);   p2 = p2 / np.linalg.norm(p2)
    return a, p1, p2


def _slab(img, start_ras, end_ras):
    """Slab around the trajectory at canonical 1mm spacing.
    No rotation — scale-space alone does not require it.
    """
    s = np.asarray(start_ras, dtype=float)
    e = np.asarray(end_ras, dtype=float)
    L = float(np.linalg.norm(e - s))
    if L < 1e-3:
        return None, None
    axis, p1, p2 = _build_orthonormal_frame((e - s) / L)
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


def _chain_at_sigma(slab, M, start_ras, end_ras, sigma_mm):
    """Run blob extraction on the slab at the given LoG sigma. Test
    whether enough on-axis inliers form a chain.
    """
    slab_clamped = sitk.Clamp(slab, lowerBound=-1024.0, upperBound=cpfit.HU_CLIP_MAX)
    log_sitk = sitk.LaplacianRecursiveGaussian(
        sitk.Cast(slab_clamped, sitk.sitkFloat32), sigma=float(sigma_mm),
    )
    log_arr = sitk.GetArrayFromImage(log_sitk)
    blobs = cpfit.extract_blobs(log_arr, threshold=cpfit.LOG_BLOB_THRESHOLD)
    if not blobs:
        return 0
    pts_kji = np.array([np.asarray(b["kji"], dtype=float) for b in blobs])
    pts_ijk = np.stack([pts_kji[:, 2], pts_kji[:, 1], pts_kji[:, 0]], axis=1)
    h = np.concatenate([pts_ijk, np.ones((pts_ijk.shape[0], 1))], axis=1)
    pts_ras = (M @ h.T).T[:, :3]

    s = np.asarray(start_ras, dtype=float); e = np.asarray(end_ras, dtype=float)
    L = float(np.linalg.norm(e - s))
    axis = (e - s) / max(1e-9, L)
    diffs = pts_ras - s
    along = diffs @ axis
    perp_vec = diffs - np.outer(along, axis)
    perp = np.linalg.norm(perp_vec, axis=1)
    mask = (perp <= cpfit.PERP_TOL_MM) & (along >= -SLAB_PAD_MM) & (along <= L + SLAB_PAD_MM)
    return int(mask.sum())


def _classify(t, raw_img):
    """Return (n_at_each_sigma_tuple, all_pass)."""
    s_ras = np.asarray(t["start_ras"], dtype=float)
    e_ras = np.asarray(t["end_ras"], dtype=float)
    slab, M = _slab(raw_img, s_ras, e_ras)
    if slab is None:
        return (0,) * len(SIGMAS_MM), False
    counts = []
    for sigma in SIGMAS_MM:
        n = _chain_at_sigma(slab, M, s_ras, e_ras, sigma)
        counts.append(n)
    all_pass = all(n >= cpfit.MIN_BLOBS_PER_LINE for n in counts)
    return tuple(counts), all_pass


def main():
    registry = PipelineRegistry()
    register_builtin_pipelines(registry)

    rows = iter_subject_rows(DATASET_ROOT, None)
    rows = [r for r in rows if str(r["subject_id"]) not in EXCLUDE]
    rows.sort(key=lambda r: int(str(r["subject_id"]).lstrip("T")))

    per_subject = []
    agg_total = defaultdict(int)
    agg_inv = defaultdict(int)
    orphan_rows = []
    matched_rows = []

    print(f"running pipeline + scale-space probe on {len(rows)} subjects "
          f"(sigmas={SIGMAS_MM} mm)\n")
    t_start_all = time.time()

    for row in rows:
        subject_id = str(row["subject_id"])
        t_sub = time.time()
        gt, _ = load_reference_ground_truth_shanks(row)
        ctx, raw_img = build_detection_context(
            row["ct_path"], run_id=f"probe_scale_{subject_id}",
            config={}, extras={},
        )
        ctx["contact_pitch_v1_pitch_strategy"] = "auto"
        result = registry.run("contact_pitch_v1", ctx)
        if str(result.get("status", "ok")).lower() == "error":
            print(f"  {subject_id}: PIPELINE ERROR")
            continue
        trajs = list(result.get("trajectories") or [])
        matched = _greedy_match(gt, trajs)

        n_matched_inv = 0; n_matched_drop = 0
        n_orph_by_band = Counter(); n_orph_inv_by_band = Counter()
        for ti, t in enumerate(trajs):
            band = str(t.get("confidence_label", "?"))
            counts, inv = _classify(t, raw_img)
            kind = "matched" if ti in matched else "orphan"
            agg_total[(kind, band)] += 1
            if inv: agg_inv[(kind, band)] += 1
            if kind == "matched":
                if inv: n_matched_inv += 1
                else: n_matched_drop += 1
                matched_rows.append({
                    "subject": subject_id, "shank": matched[ti], "band": band,
                    "counts": counts, "invariant": bool(inv),
                    "n_orig": int(t.get("n_inliers", 0)),
                })
            else:
                n_orph_by_band[band] += 1
                if inv: n_orph_inv_by_band[band] += 1
                orphan_rows.append({
                    "subject": subject_id, "band": band,
                    "bolt_source": str(t.get("bolt_source", "?")),
                    "n_orig": int(t.get("n_inliers", 0)),
                    "counts": counts,
                    "pitch_mm": float(t.get("original_median_pitch_mm", 0.0)),
                    "confidence": float(t.get("confidence", 0.0)),
                    "invariant": bool(inv),
                })

        n_orph = sum(n_orph_by_band.values())
        n_orph_inv = sum(n_orph_inv_by_band.values())
        per_subject.append({
            "subject": subject_id, "gt": len(gt), "matched": len(matched),
            "matched_inv": n_matched_inv, "matched_drop": n_matched_drop,
            "orphans": n_orph, "orphan_inv": n_orph_inv,
            "orphan_drop": n_orph - n_orph_inv,
        })
        elapsed = time.time() - t_sub
        print(f"  {subject_id:>4s}: gt={len(gt):>2d}  matched={len(matched):>2d} "
              f"(inv={n_matched_inv:>2d}, drop={n_matched_drop:>2d})  "
              f"orphans={n_orph:>2d} (inv={n_orph_inv}, drop={n_orph - n_orph_inv})  "
              f"[{elapsed:.1f}s]")

    print(f"\nTotal probe wall time: {time.time() - t_start_all:.1f}s")

    # Aggregate
    print("\n" + "=" * 70)
    print("AGGREGATE: scale-space-survival rates by (kind, band)")
    print("=" * 70)
    print(f"{'kind':>8s} {'band':>8s}  {'n_total':>8s}  {'n_inv':>8s}  {'inv_rate':>9s}")
    for k in sorted(agg_total.keys()):
        tot = agg_total[k]; inv = agg_inv.get(k, 0)
        print(f"{k[0]:>8s} {k[1]:>8s}  {tot:>8d}  {inv:>8d}  {inv/max(1,tot):>9.1%}")

    tot_m = sum(v for (k, _), v in agg_total.items() if k == "matched")
    inv_m = sum(v for (k, _), v in agg_inv.items() if k == "matched")
    tot_o = sum(v for (k, _), v in agg_total.items() if k == "orphan")
    inv_o = sum(v for (k, _), v in agg_inv.items() if k == "orphan")
    print(f"\nmatched: {inv_m}/{tot_m} preserved ({inv_m/max(1,tot_m):.1%})")
    print(f"orphans: {tot_o - inv_o}/{tot_o} dropped ({(tot_o - inv_o)/max(1,tot_o):.1%})")

    # Per-orphan dump
    print("\n" + "=" * 70)
    print(f"All {len(orphan_rows)} orphans, sorted by (band, invariant)")
    print("=" * 70)
    print(f"{'subj':>4s} {'band':>6s} {'inv':>4s} {'bolt':>11s} "
          f"{'n_orig':>6s} {'n@.8':>5s} {'n@1':>4s} {'n@1.2':>5s} {'pitch':>5s} {'conf':>5s}")
    band_order = {"high": 0, "medium": 1, "low": 2, "?": 3}
    orphan_rows.sort(key=lambda r: (band_order.get(r["band"], 99), not r["invariant"]))
    for r in orphan_rows:
        marker = "INV" if r["invariant"] else "DROP"
        c = r["counts"]
        print(f"{r['subject']:>4s} {r['band']:>6s} {marker:>4s} {r['bolt_source']:>11s} "
              f"{r['n_orig']:>6d} {c[0]:>5d} {c[1]:>4d} {c[2]:>5d} "
              f"{r['pitch_mm']:>5.2f} {r['confidence']:>5.2f}")

    # Matched failures (should be 0 or near 0)
    matched_failures = [r for r in matched_rows if not r["invariant"]]
    if matched_failures:
        print(f"\n=== MATCHED LINES THAT DROP UNDER SCALE-SPACE ({len(matched_failures)}) ===")
        print("These are real shanks that the gate would have lost — recall-cost!")
        print(f"{'subj':>4s} {'shank':>10s} {'band':>6s} {'n_orig':>6s} {'n@.8':>5s} {'n@1':>4s} {'n@1.2':>5s}")
        for r in matched_failures:
            c = r["counts"]
            print(f"{r['subject']:>4s} {r['shank']:>10s} {r['band']:>6s} "
                  f"{r['n_orig']:>6d} {c[0]:>5d} {c[1]:>4d} {c[2]:>5d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
