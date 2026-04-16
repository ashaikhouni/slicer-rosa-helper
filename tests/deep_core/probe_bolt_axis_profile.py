"""Probe: bolt axis quality + inward HU profile dump.

For each detected bolt on a subject:
  1. Re-fit the bolt axis via PCA on bolt-metal voxels lying within a
     5 mm-diameter tube of the RANSAC axis (broader than the 3 mm
     RANSAC inlier tube). This lets us see whether the RANSAC axis is
     tube-support-tight or has bias.
  2. Match the bolt to the nearest GT shank (by bolt center -> GT
     start_ras) and report the angle between refit axis and GT
     direction.
  3. Walk +5 mm to -150 mm (inward) along the refit axis from the bolt
     center in 1 mm steps. At each step gather voxels within a 5 mm
     radius perpendicular disc and emit median / p90 / max HU plus
     count of voxels above 400 HU.

Usage:
  /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
    tests/deep_core/probe_bolt_axis_profile.py [SUBJECT_ID] [OUT_DIR]

Defaults: SUBJECT_ID=T22, OUT_DIR prints to stdout.
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


# --- geometry helpers --------------------------------------------------------

def _unit(v):
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def _perp_frame(axis):
    a = _unit(axis)
    helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _unit(helper - np.dot(helper, a) * a)
    v = _unit(np.cross(a, u))
    return u, v


def _pca_axis(points):
    c = points.mean(axis=0)
    X = points - c
    cov = X.T @ X / max(1, X.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    return _unit(axis), c


def _refit_axis_from_tube(cloud_ras, axis0, center0, *, half_span_mm, radius_mm):
    """PCA on cloud points within ``radius_mm`` perp + ``half_span_mm`` axial of the seed line."""
    v = cloud_ras - center0
    proj = v @ axis0
    perp = v - np.outer(proj, axis0)
    dist = np.linalg.norm(perp, axis=1)
    keep = (dist < radius_mm) & (proj > -half_span_mm) & (proj < half_span_mm)
    if int(keep.sum()) < 8:
        return axis0, center0, int(keep.sum())
    axis_ref, center_ref = _pca_axis(cloud_ras[keep])
    if np.dot(axis_ref, axis0) < 0:
        axis_ref = -axis_ref
    return axis_ref, center_ref, int(keep.sum())


def _angle_deg(a, b):
    a = _unit(a); b = _unit(b)
    c = float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _perp_dist_to_line(point, line_pt, line_dir):
    line_dir = _unit(line_dir)
    v = np.asarray(point, dtype=float) - np.asarray(line_pt, dtype=float)
    return float(np.linalg.norm(v - np.dot(v, line_dir) * line_dir))


def _match_bolts_to_gt(bolts_info, gt_shanks, *, max_perp_mm=8.0, max_angle_deg=15.0):
    """Greedy one-to-one match: each GT shank gets at most one bolt.

    Score = perp_distance_mm + 1.5 * angle_deg. Lower is better. A pair is
    eligible only if perp <= max_perp_mm AND angle <= max_angle_deg.
    Returns dict gt_shank_name -> bolt_index (or absent if unmatched).
    """
    pairs = []  # (score, gt_idx, bolt_idx)
    for gi, gt in enumerate(gt_shanks):
        gt_start = np.asarray(gt.start_ras, dtype=float)
        gt_dir = np.asarray(gt.direction_ras, dtype=float)
        for bi, info in enumerate(bolts_info):
            perp = _perp_dist_to_line(info["center"], gt_start, gt_dir)
            ang = _angle_deg(info["axis"], gt_dir)
            if perp <= max_perp_mm and ang <= max_angle_deg:
                pairs.append((perp + 1.5 * ang, perp, ang, gi, bi))
    pairs.sort()
    used_gt = set()
    used_bolt = set()
    gt_to_bolt = {}
    bolt_to_gt = {}
    for score, perp, ang, gi, bi in pairs:
        if gi in used_gt or bi in used_bolt:
            continue
        used_gt.add(gi)
        used_bolt.add(bi)
        gt_to_bolt[gt_shanks[gi].shank] = (bi, perp, ang)
        bolt_to_gt[bi] = (gt_shanks[gi].shank, perp, ang)
    return gt_to_bolt, bolt_to_gt


def _walk_extent(rows, *, accept_fn, gap_mm=15.0, step_mm=1.0):
    """Walk t>=0 inward; return deepest t where accept_fn(row)==True under
    a contiguous-gap tolerance of gap_mm.
    """
    gap_steps = max(1, int(round(gap_mm / step_mm)))
    deepest = 0.0
    gap_run = 0
    seen_any = False
    for r in sorted(rows, key=lambda r: r[0]):
        t = r[0]
        if t < 0.0:
            continue
        if accept_fn(r):
            deepest = float(t)
            gap_run = 0
            seen_any = True
        else:
            gap_run += 1
            if gap_run > gap_steps:
                break
    return deepest if seen_any else float("nan")


def _walk_grid(rows, gap_mm=15.0):
    """Multi-criterion walk extents: max_hu>=H for H in {400,700,1000} and
    n_above_400>=K for K in {1,2,3}.
    """
    out = {}
    for hu in (400.0, 700.0, 1000.0):
        out[f"max>={int(hu)}"] = _walk_extent(
            rows, accept_fn=lambda r, h=hu: np.isfinite(r[4]) and r[4] >= h, gap_mm=gap_mm,
        )
    for k in (1, 2, 3):
        out[f"n400>={k}"] = _walk_extent(
            rows, accept_fn=lambda r, kk=k: r[5] >= kk, gap_mm=gap_mm,
        )
    return out


# --- profile sampling --------------------------------------------------------

def _disc_offset_grid(axis, radius_mm, step_mm=0.75):
    u, v = _perp_frame(axis)
    coords = np.arange(-radius_mm, radius_mm + 0.5 * step_mm, step_mm)
    pts = []
    r2 = radius_mm * radius_mm
    for du in coords:
        for dv in coords:
            if du * du + dv * dv <= r2:
                pts.append(du * u + dv * v)
    return np.stack(pts, axis=0)


def _sample_disc_voxels(center_pt_ras, offsets, arr_kji, ras_to_ijk_fn):
    """Return unique-voxel HU values inside a perpendicular disc."""
    shape = arr_kji.shape
    seen = set()
    hus = []
    for off in offsets:
        ijk = np.asarray(ras_to_ijk_fn(center_pt_ras + off), dtype=float)
        kji = (int(round(ijk[2])), int(round(ijk[1])), int(round(ijk[0])))
        if kji in seen:
            continue
        if not (0 <= kji[0] < shape[0] and 0 <= kji[1] < shape[1] and 0 <= kji[2] < shape[2]):
            continue
        seen.add(kji)
        hus.append(float(arr_kji[kji[0], kji[1], kji[2]]))
    return np.asarray(hus, dtype=float)


def profile_along_axis(
    *, center_ras, axis_ras, arr_kji, ras_to_ijk_fn,
    outward_mm=5.0, inward_mm=150.0, step_mm=1.0, radius_mm=5.0,
):
    offsets = _disc_offset_grid(axis_ras, radius_mm, step_mm=0.75)
    ts = np.arange(-inward_mm, outward_mm + 0.5 * step_mm, step_mm)
    rows = []
    for t in ts:
        pt = center_ras + t * axis_ras
        hus = _sample_disc_voxels(pt, offsets, arr_kji, ras_to_ijk_fn)
        if hus.size == 0:
            rows.append((float(t), 0, np.nan, np.nan, np.nan, 0))
            continue
        med = float(np.median(hus))
        p90 = float(np.percentile(hus, 90))
        mx = float(hus.max())
        n400 = int((hus >= 400.0).sum())
        rows.append((float(t), int(hus.size), med, p90, mx, n400))
    return rows


# --- driver ------------------------------------------------------------------

def _orient_axis_inward(axis, center, head_distance_map_kji, ras_to_ijk_fn):
    """Flip axis so +axis points DEEP (higher head_distance)."""
    shape = head_distance_map_kji.shape

    def hd(pt):
        ijk = np.asarray(ras_to_ijk_fn(pt), dtype=float)
        kji = (int(round(ijk[2])), int(round(ijk[1])), int(round(ijk[0])))
        if not (0 <= kji[0] < shape[0] and 0 <= kji[1] < shape[1] and 0 <= kji[2] < shape[2]):
            return float("nan")
        return float(head_distance_map_kji[kji[0], kji[1], kji[2]])

    hp = hd(center + 20.0 * axis)
    hm = hd(center - 20.0 * axis)
    if np.isfinite(hp) and np.isfinite(hm) and hm > hp:
        return -axis
    return axis


def run_probe(subject_id, out_dir=None, *, probe_radius_mm=2.5):
    from shank_engine import PipelineRegistry, register_builtin_pipelines
    from eval_seeg_localization import (
        build_detection_context,
        iter_subject_rows,
        load_reference_ground_truth_shanks,
    )

    rows = iter_subject_rows(DATASET_ROOT, {subject_id})
    if not rows:
        raise SystemExit(f"subject {subject_id} not in manifest")
    row = rows[0]
    gt_shanks, _ = load_reference_ground_truth_shanks(row)

    registry = PipelineRegistry()
    register_builtin_pipelines(registry)

    ctx, _ = build_detection_context(
        row["ct_path"], run_id=f"probe_{subject_id}", config={}, extras={}
    )

    pipeline = registry.create_pipeline("deep_core_v2")
    pipeline.run_debug(ctx)
    mask = pipeline._last_mask_output
    bolt_out = pipeline._last_bolt_output

    arr_kji = np.asarray(ctx["arr_kji"], dtype=float)
    ras_to_ijk_fn = ctx["ras_to_ijk_fn"]
    ijk_kji_to_ras_fn = ctx["ijk_kji_to_ras_fn"]
    bolt_metal_mask = np.asarray(mask["bolt_metal_mask_kji"], dtype=bool)
    head_distance_map = mask["head_distance_map_kji"]

    idx = np.argwhere(bolt_metal_mask)
    cloud_ras = np.asarray(ijk_kji_to_ras_fn(idx.astype(float)), dtype=float)
    print(f"# subject={subject_id}  bolt_metal_voxels={cloud_ras.shape[0]}  bolts={len(bolt_out['candidates'])}  gt_shanks={len(gt_shanks)}")

    out_dir_path = Path(out_dir) if out_dir else None
    if out_dir_path:
        out_dir_path.mkdir(parents=True, exist_ok=True)

    # Pass 1: refit axes and collect per-bolt info.
    bolts_info = []
    for bi, bolt in enumerate(bolt_out["candidates"]):
        c0 = np.asarray(bolt.center_ras, dtype=float)
        a0 = _unit(np.asarray(bolt.axis_ras, dtype=float))
        half_span = float(bolt.span_mm) / 2.0 + 1.0
        a_ref, c_ref, n_keep = _refit_axis_from_tube(
            cloud_ras, a0, c0, half_span_mm=half_span, radius_mm=2.5,
        )
        a_in = _orient_axis_inward(a_ref, c_ref, head_distance_map, ras_to_ijk_fn)
        bolts_info.append({
            "bi": bi, "bolt": bolt, "center": c_ref, "axis": a_in,
            "axis_orig": a0, "n_refit": n_keep,
            "ransac_vs_refit_deg": _angle_deg(a_in, a0),
        })

    # Pass 2: greedy one-to-one GT <-> bolt matching.
    gt_to_bolt, bolt_to_gt = _match_bolts_to_gt(
        bolts_info, gt_shanks, max_perp_mm=8.0, max_angle_deg=15.0,
    )

    # Pass 3: per-bolt diagnostics + profile dump + walk-extent summary.
    summary_rows = []
    for info in bolts_info:
        bi = info["bi"]
        bolt = info["bolt"]
        c_ref = info["center"]
        a_in = info["axis"]

        match = bolt_to_gt.get(bi)
        if match:
            gt_name, perp, ang = match
            tag = f"TP {gt_name} (perp={perp:.1f}mm, ang={ang:.1f}deg)"
        else:
            tag = "FP/dup"

        print(
            f"\n## bolt {bi:2d}  center={c_ref.round(1).tolist()}  span={bolt.span_mm:.1f}mm  "
            f"inliers={bolt.n_inliers}  refit_vox={info['n_refit']}  "
            f"ransac_vs_refit={info['ransac_vs_refit_deg']:.1f}deg  -> {tag}"
        )

        # Profile: -5..+150 mm, +axis = deep.
        offsets = _disc_offset_grid(a_in, radius_mm=probe_radius_mm, step_mm=0.5)
        step_mm = 1.0
        ts = np.arange(-5.0, 150.0 + 0.5 * step_mm, step_mm)
        rows_out = []
        for t in ts:
            pt = c_ref + t * a_in
            hus = _sample_disc_voxels(pt, offsets, arr_kji, ras_to_ijk_fn)
            if hus.size == 0:
                rows_out.append((float(t), 0, float("nan"), float("nan"), float("nan"), 0))
            else:
                rows_out.append((
                    float(t), int(hus.size), float(np.median(hus)),
                    float(np.percentile(hus, 90)), float(hus.max()),
                    int((hus >= 400.0).sum()),
                ))

        # Walk-extent grid (gap=15mm, several accept criteria).
        extents = _walk_grid(rows_out, gap_mm=15.0)
        ext_str = "  ".join(f"{k}={v:.0f}" for k, v in extents.items())
        print(f"   gap=15mm  {ext_str}")

        summary_rows.append({
            "bi": bi, "tag": tag, "extents": extents,
            "ransac_vs_refit": info["ransac_vs_refit_deg"],
        })

        header = "t_mm,n_vox,med_hu,p90_hu,max_hu,n_above_400"
        if out_dir_path:
            fname = out_dir_path / f"{subject_id}_bolt{bi:02d}.csv"
            with open(fname, "w") as fh:
                fh.write(header + "\n")
                for r in rows_out:
                    fh.write(",".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in r) + "\n")

    # Final report: which GT shanks were/weren't matched.
    matched_names = set(gt_to_bolt.keys())
    print("\n# GT match summary")
    for gt in gt_shanks:
        if gt.shank in matched_names:
            bi, perp, ang = gt_to_bolt[gt.shank]
            print(f"   MATCHED  {gt.shank:8s} -> bolt {bi:2d}  perp={perp:.1f}mm  ang={ang:.1f}deg")
        else:
            print(f"   MISSED   {gt.shank:8s} (no bolt within perp<=8mm and ang<=15deg)")

    print(f"\n# Walk-extent grid per matched bolt  (probe_radius={probe_radius_mm}mm, gap=15mm)")
    crits = list(summary_rows[0]["extents"].keys()) if summary_rows else []
    print("   bolt  gt_shank        " + "  ".join(f"{c:>10s}" for c in crits))
    for s in summary_rows:
        if s["tag"].startswith("TP"):
            shank_short = s["tag"].split()[1]
            cells = "  ".join(f"{s['extents'][c]:>10.0f}" for c in crits)
            print(f"   {s['bi']:>4d}  {shank_short:<14s} {cells}")
    print("   (FP/dup bolts:)")
    for s in summary_rows:
        if not s["tag"].startswith("TP"):
            cells = "  ".join(f"{s['extents'][c]:>10.0f}" for c in crits)
            print(f"   {s['bi']:>4d}  {'(FP/dup)':<14s} {cells}")


def main():
    subject = sys.argv[1] if len(sys.argv) > 1 else "T22"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    radius = float(sys.argv[3]) if len(sys.argv) > 3 else 2.5
    run_probe(subject, out_dir=out_dir, probe_radius_mm=radius)


if __name__ == "__main__":
    main()
