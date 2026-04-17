"""Probe v4: HU-agnostic electrode detector using RANSAC on Frangi voxels.

Replaces ridge tracking (v3) with global line fitting:
  1. Same mask (head_distance >= 10 mm intracranial).
  2. Frangi sigma=1 (contact-scale response).
  3. Threshold -> voxel cloud, weighted by Frangi intensity.
  4. Iteratively: RANSAC-fit a line through the cloud; accept if inlier count
     and span meet thresholds; remove inliers within a tube; repeat.
  5. Same periodicity confirmation + dedup + GT evaluation as v3.

No Hessian. No local-axis computation. Line fits are global, immune to
close-parallel axis-averaging that plagued ridge tracking at sigma=2.

Usage:
  /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
    tests/deep_core/probe_detector_v4.py [T22|T2]
"""
from __future__ import annotations

import csv
import os
import sys
import time
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

# ----- Config -----
INTRACRANIAL_MIN_DISTANCE_MM = 10.0
FRANGI_S1_THR = 10.0         # voxel cloud threshold (on sigma=1 Frangi)
RANSAC_TOL_MM = 1.0           # inlier distance tolerance
RANSAC_ITER = 800             # trials per line
RANSAC_MIN_INLIERS = 40       # minimum inliers to accept a line
RANSAC_MIN_SPAN_MM = 20.0
RANSAC_MAX_SPAN_MM = 85.0     # physical max electrode length
RANSAC_EXCLUSION_MM = 1.5     # tube radius zeroed after each line

PITCH_STEP_MM = 0.25
PITCH_BAND_LO = 0.2
PITCH_BAND_HI = 0.4
PITCH_MIN_MM = 2.6
PITCH_MAX_MM = 4.8
PITCH_MIN_SNR = 2.0
PITCH_MIN_TRACK_LEN_MM = 22.0
LONG_TRACK_BYPASS_MM = 40.0

DEDUP_ANGLE_DEG = 5.0
DEDUP_PERP_MM = 3.0
DEDUP_OVERLAP_FRAC = 0.4

MATCH_ANGLE_DEG = 10.0
MATCH_MIDPOINT_MM = 8.0
# ------------------


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


def _sample_mask(mask, kji_float):
    k = int(round(kji_float[0])); j = int(round(kji_float[1])); i = int(round(kji_float[2]))
    s = mask.shape
    if 0 <= k < s[0] and 0 <= j < s[1] and 0 <= i < s[2]:
        return bool(mask[k, j, i])
    return False


# ----- Masks -----
def build_masks(img):
    import SimpleITK as sitk
    thr = sitk.BinaryThreshold(img, lowerThreshold=-500.0, upperThreshold=1e9,
                                insideValue=1, outsideValue=0)
    cc = sitk.RelabelComponent(sitk.ConnectedComponent(thr), sortByObjectSize=True)
    largest = sitk.Equal(cc, 1)
    closed = sitk.BinaryMorphologicalClosing(largest, kernelRadius=(3, 3, 3))
    hull = sitk.BinaryFillhole(closed)
    dist = sitk.SignedMaurerDistanceMap(
        hull, insideIsPositive=True, squaredDistance=False, useImageSpacing=True,
    )
    dist_arr = sitk.GetArrayFromImage(dist).astype(np.float32)
    hull_arr = sitk.GetArrayFromImage(hull).astype(bool)
    return hull_arr, dist_arr >= INTRACRANIAL_MIN_DISTANCE_MM


def frangi_single(img, sigma):
    import SimpleITK as sitk
    sm = sitk.SmoothingRecursiveGaussian(img, sigma=float(sigma))
    ob = sitk.ObjectnessMeasure(sm, objectDimension=1, brightObject=True)
    return sitk.GetArrayFromImage(ob).astype(np.float32)


# ----- RANSAC -----
def ransac_one_line(pts, weights, tol_mm, n_iter, min_inliers, min_span, rng):
    n = pts.shape[0]
    if n < min_inliers:
        return None
    p = weights / weights.sum()
    best = None
    for _ in range(n_iter):
        idx = rng.choice(n, size=2, replace=False, p=p)
        p1, p2 = pts[idx[0]], pts[idx[1]]
        d = p2 - p1
        dn = float(np.linalg.norm(d))
        if dn < 3.0:
            continue
        axis = d / dn
        diffs = pts - p1
        proj = diffs @ axis
        perp = diffs - np.outer(proj, axis)
        dists = np.linalg.norm(perp, axis=1)
        inliers = dists < tol_mm
        n_in = int(inliers.sum())
        if n_in < min_inliers:
            continue
        in_proj = proj[inliers]
        span = float(in_proj.max() - in_proj.min())
        if span < min_span:
            continue
        score = float(weights[inliers].sum())
        if best is None or score > best["score"]:
            best = dict(axis=axis, p1=p1, inliers=inliers, n=n_in,
                        span=span, score=score)

    if best is None:
        return None
    # PCA refinement on inliers
    inlier_pts = pts[best["inliers"]]
    c = inlier_pts.mean(axis=0)
    X = inlier_pts - c
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    axis = _unit(Vt[0])
    diffs = pts - c
    proj = diffs @ axis
    perp = diffs - np.outer(proj, axis)
    dists = np.linalg.norm(perp, axis=1)
    inliers = dists < tol_mm
    if int(inliers.sum()) < min_inliers:
        return None
    in_proj = proj[inliers]
    span_lo, span_hi = float(in_proj.min()), float(in_proj.max())
    span = span_hi - span_lo
    if span < min_span:
        return None

    # If span exceeds physical max, clip to densest max-span window.
    if span > RANSAC_MAX_SPAN_MM:
        # sliding-window density along the line
        sorted_proj = np.sort(in_proj)
        best_count = 0
        best_center = 0.5 * (span_lo + span_hi)
        w = RANSAC_MAX_SPAN_MM
        step = 2.0
        for t in np.arange(span_lo + 0.5 * w, span_hi - 0.5 * w + 1e-6, step):
            cnt = int(np.sum((sorted_proj >= t - 0.5 * w) & (sorted_proj <= t + 0.5 * w)))
            if cnt > best_count:
                best_count = cnt
                best_center = float(t)
        span_lo = best_center - 0.5 * RANSAC_MAX_SPAN_MM
        span_hi = best_center + 0.5 * RANSAC_MAX_SPAN_MM
        # restrict inliers to this window
        window_ok = (proj >= span_lo) & (proj <= span_hi)
        inliers = inliers & window_ok
        if int(inliers.sum()) < min_inliers:
            return None
        in_proj = proj[inliers]
        if in_proj.size == 0:
            return None
        span_lo, span_hi = float(in_proj.min()), float(in_proj.max())
        span = span_hi - span_lo
        if span < min_span:
            return None

    # Inlier density check: real electrodes have dense inliers; skull-
    # spanning lines have sparse ones.
    if span > 0 and (int(inliers.sum()) / span) < 0.5:
        return None

    return dict(axis=axis, center=c, inlier_mask=inliers,
                span_mm=span, span_lo=span_lo, span_hi=span_hi,
                n_inliers=int(inliers.sum()))


def iterative_ransac(frangi_s1, intracranial_mask):
    # voxel cloud
    mask = (frangi_s1 >= FRANGI_S1_THR) & intracranial_mask
    kk, jj, ii = np.where(mask)
    pts = np.stack([kk.astype(float), jj.astype(float), ii.astype(float)], axis=1)
    weights = frangi_s1[mask].astype(float)
    print(f"  cloud voxels: {pts.shape[0]}  "
          f"weight_max={weights.max():.1f}  mean={weights.mean():.1f}")
    rng = np.random.default_rng(0)
    lines = []
    active = np.ones(pts.shape[0], dtype=bool)
    for iteration in range(60):
        if int(active.sum()) < RANSAC_MIN_INLIERS:
            break
        active_pts = pts[active]
        active_w = weights[active]
        result = ransac_one_line(
            active_pts, active_w, tol_mm=RANSAC_TOL_MM, n_iter=RANSAC_ITER,
            min_inliers=RANSAC_MIN_INLIERS, min_span=RANSAC_MIN_SPAN_MM, rng=rng,
        )
        if result is None:
            break
        # Build polyline for this line
        step = 0.5
        n_steps = int(result["span_mm"] / step) + 1
        t = np.linspace(result["span_lo"], result["span_hi"], n_steps)
        polyline = result["center"] + t[:, None] * result["axis"]
        lines.append({
            "polyline_kji": polyline,
            "length_mm": float(result["span_mm"]),
            "n_inliers": int(result["n_inliers"]),
            "axis": result["axis"],
            "center": result["center"],
        })
        # Remove voxels within EXCLUSION tube from active pool
        diffs = active_pts - result["center"]
        proj = diffs @ result["axis"]
        perp = diffs - np.outer(proj, result["axis"])
        dists = np.linalg.norm(perp, axis=1)
        to_remove = (dists < RANSAC_EXCLUSION_MM) & (proj >= result["span_lo"] - 2.0) & (proj <= result["span_hi"] + 2.0)
        active_idx = np.where(active)[0]
        active[active_idx[to_remove]] = False

    return lines


# ----- Dedup -----
def _track_axis_and_span(polyline):
    center = polyline.mean(axis=0)
    X = polyline - center
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    axis = _unit(Vt[0])
    proj = X @ axis
    return axis, center, float(proj.min()), float(proj.max())


def dedup_tracks(tracks):
    if len(tracks) < 2:
        return tracks
    info = []
    for t in tracks:
        axis, center, lo, hi = _track_axis_and_span(t["polyline_kji"])
        info.append(dict(axis=axis, center=center, lo=lo, hi=hi,
                          length=t["length_mm"], t=t))
    keep = [True] * len(info)
    for i in range(len(info)):
        if not keep[i]:
            continue
        ai = info[i]
        for j in range(i + 1, len(info)):
            if not keep[j]:
                continue
            aj = info[j]
            ang = float(np.degrees(np.arccos(np.clip(abs(np.dot(ai["axis"], aj["axis"])), 0, 1))))
            if ang > DEDUP_ANGLE_DEG:
                continue
            dv = aj["center"] - ai["center"]
            par = dv @ ai["axis"]
            perp = dv - par * ai["axis"]
            perp_d = float(np.linalg.norm(perp))
            if perp_d > DEDUP_PERP_MM:
                continue
            j_lo = par + aj["lo"]; j_hi = par + aj["hi"]
            overlap = max(0.0, min(ai["hi"], j_hi) - max(ai["lo"], j_lo))
            shorter = min(ai["hi"] - ai["lo"], aj["hi"] - aj["lo"])
            if shorter > 1e-6 and (overlap / shorter) >= DEDUP_OVERLAP_FRAC:
                if ai["length"] >= aj["length"]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break
    return [info[i]["t"] for i in range(len(info)) if keep[i]]


# ----- Periodicity -----
def _orthonormal_basis(d):
    any_vec = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(d, any_vec)) > 0.9:
        any_vec = np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(d, any_vec))
    v = _unit(np.cross(d, u))
    return u, v


def periodicity_along_polyline(frangi_s1, polyline_kji, intracranial_mask):
    if polyline_kji.shape[0] < 3:
        return float("nan"), float("nan")
    center = polyline_kji.mean(axis=0)
    X = polyline_kji - center
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    axis = _unit(Vt[0])
    proj = X @ axis
    lo, hi = float(proj.min()), float(proj.max())
    if (hi - lo) < 15.0:
        return float("nan"), float("nan")
    n_steps = int((hi - lo) / PITCH_STEP_MM) + 1
    u, v = _orthonormal_basis(axis)
    vals = np.empty(n_steps, dtype=float)
    for i in range(n_steps):
        c = center + (lo + i * PITCH_STEP_MM) * axis
        if not _sample_mask(intracranial_mask, c):
            vals[i] = np.nan
            continue
        samples = [_sample_trilinear(frangi_s1, c)]
        for k in range(8):
            ang = 2.0 * np.pi * k / 8
            off = 1.0 * (np.cos(ang) * u + np.sin(ang) * v)
            samples.append(_sample_trilinear(frangi_s1, c + off))
        s = np.asarray(samples, dtype=float)
        s = s[np.isfinite(s)]
        vals[i] = float(s.max()) if s.size else np.nan
    vals = vals[np.isfinite(vals)]
    if vals.size < 20:
        return float("nan"), float("nan")
    vals = vals - vals.mean()
    spec = np.fft.rfft(vals)
    power = (np.abs(spec) ** 2) / vals.size
    freqs = np.fft.rfftfreq(vals.size, d=PITCH_STEP_MM)
    mask = (freqs > PITCH_BAND_LO) & (freqs < PITCH_BAND_HI)
    if not mask.any():
        return float("nan"), float("nan")
    band = power[mask]; bfreq = freqs[mask]
    pi = int(np.argmax(band))
    peak_pitch = 1.0 / float(bfreq[pi]) if bfreq[pi] > 0 else float("nan")
    peak_power = float(band[pi])
    excl = np.zeros_like(band, dtype=bool)
    excl[max(0, pi - 3):pi + 4] = True
    bg = band[~excl]
    bg_power = float(bg.mean()) if bg.size else 0.0
    snr = peak_power / bg_power if bg_power > 0 else float("inf")
    return peak_pitch, snr


# ----- GT + matching (same as v3) -----
def load_gt(subject_id):
    csv_path = (
        DATASET_ROOT / "contact_label_dataset" / "rosa_helper_import"
        / subject_id / "ROSA_Contacts_final_trajectory_points.csv"
    )
    if csv_path.exists():
        traj = {}
        with open(csv_path) as f:
            r = csv.DictReader(f)
            for row in r:
                name = row["trajectory"]
                pt = np.array(
                    [float(row["x_world_ras"]), float(row["y_world_ras"]),
                     float(row["z_world_ras"])], dtype=float,
                )
                traj.setdefault(name, {})[row["point_type"]] = pt
        out = []
        for name, pair in traj.items():
            if "entry" in pair and "target" in pair:
                L = float(np.linalg.norm(pair["target"] - pair["entry"]))
                out.append((name, pair["entry"], pair["target"], L))
        return out
    from eval_seeg_localization import iter_subject_rows, load_ground_truth_shanks
    rows = iter_subject_rows(DATASET_ROOT, {subject_id})
    if not rows:
        return []
    row = rows[0]
    shanks = load_ground_truth_shanks(row["labels_path"], row.get("shanks_path"))
    return [
        (s.shank, np.asarray(s.start_ras, dtype=float),
         np.asarray(s.end_ras, dtype=float), float(s.span_mm))
        for s in shanks
    ]


def _line_axis(start, end):
    d = end - start
    return _unit(d), float(np.linalg.norm(d))


def match_tracks_to_gt(tracks_ras, gt):
    pairs = []
    for gi, (name, g_s, g_e, g_L) in enumerate(gt):
        g_axis, _ = _line_axis(g_s, g_e)
        g_mid = 0.5 * (g_s + g_e)
        for ti, t in enumerate(tracks_ras):
            t_s, t_e = t["start_ras"], t["end_ras"]
            t_axis, _ = _line_axis(t_s, t_e)
            ang = float(np.degrees(np.arccos(np.clip(abs(np.dot(g_axis, t_axis)), 0, 1))))
            t_mid = 0.5 * (t_s + t_e)
            v = g_mid - t_mid
            perp = v - (v @ t_axis) * t_axis
            mid_d = float(np.linalg.norm(perp))
            if ang <= MATCH_ANGLE_DEG and mid_d <= MATCH_MIDPOINT_MM:
                pairs.append((ang + mid_d, gi, ti, ang, mid_d))
    pairs.sort(key=lambda p: p[0])
    gt_assigned = {}
    track_used = set()
    for score, gi, ti, ang, md in pairs:
        if gi in gt_assigned or ti in track_used:
            continue
        gt_assigned[gi] = dict(track_index=ti, angle_deg=ang, mid_d_mm=md)
        track_used.add(ti)
    results = []
    for gi, (name, _, _, g_L) in enumerate(gt):
        if gi in gt_assigned:
            a = gt_assigned[gi]
            results.append(dict(gt_name=name, gt_len=g_L, matched=True, **a))
        else:
            results.append(dict(gt_name=name, gt_len=g_L, matched=False,
                                track_index=-1, angle_deg=float("nan"),
                                mid_d_mm=float("nan")))
    return results


def save_label_volume(tracks, shape, img, out_path):
    import SimpleITK as sitk
    label = np.zeros(shape, dtype=np.uint16)
    for i, t in enumerate(tracks, start=1):
        for p in t["polyline_kji"]:
            kc, jc, ic = int(round(p[0])), int(round(p[1])), int(round(p[2]))
            for dk in range(-1, 2):
                for dj in range(-1, 2):
                    for di in range(-1, 2):
                        k, j, ii = kc + dk, jc + dj, ic + di
                        if 0 <= k < shape[0] and 0 <= j < shape[1] and 0 <= ii < shape[2]:
                            if dk * dk + dj * dj + di * di <= 1:
                                label[k, j, ii] = i
    out_img = sitk.GetImageFromArray(label)
    out_img.CopyInformation(img)
    sitk.WriteImage(out_img, out_path)


def run(subject_id):
    import SimpleITK as sitk
    from shank_core.io import image_ijk_ras_matrices

    ct_path = DATASET_ROOT / "post_registered_ct" / f"{subject_id}_post_registered.nii.gz"
    print(f"# subject={subject_id}")
    print(f"# ct={ct_path}")
    img = sitk.ReadImage(str(ct_path))

    t0 = time.time()
    hull, intracranial = build_masks(img)
    print(f"# masks: hull={hull.sum()} intracranial={intracranial.sum()} "
          f"({time.time()-t0:.1f}s)")

    t0 = time.time()
    frangi_s1 = frangi_single(img, sigma=1.0)
    print(f"# frangi sigma=1: max={frangi_s1.max():.1f} "
          f"p99={np.percentile(frangi_s1, 99):.1f} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    print("# iterative RANSAC:")
    tracks = iterative_ransac(frangi_s1, intracranial)
    n_pre = len(tracks)
    tracks = dedup_tracks(tracks)
    print(f"# lines: {n_pre} before dedup, {len(tracks)} after ({time.time()-t0:.1f}s)")

    # Convert to RAS
    ijk_to_ras_mat, _ = image_ijk_ras_matrices(img)
    ijk_to_ras_mat = np.asarray(ijk_to_ras_mat, dtype=float)

    def kji_to_ras(kji):
        ijk = np.stack([kji[:, 2], kji[:, 1], kji[:, 0]], axis=1)
        h = np.concatenate([ijk, np.ones((ijk.shape[0], 1))], axis=1)
        return (ijk_to_ras_mat @ h.T).T[:, :3]

    tracks_ras = []
    for t in tracks:
        pts_ras = kji_to_ras(t["polyline_kji"])
        tracks_ras.append({
            "polyline_ras": pts_ras,
            "start_ras": pts_ras[0],
            "end_ras": pts_ras[-1],
            "length_mm": t["length_mm"],
            "n_inliers": t["n_inliers"],
            "polyline_kji": t["polyline_kji"],
        })

    # Periodicity confirmation
    accepted = []
    for t, ts in zip(tracks, tracks_ras):
        pitch, snr = periodicity_along_polyline(frangi_s1, t["polyline_kji"], intracranial)
        periodic_ok = (
            ts["length_mm"] >= PITCH_MIN_TRACK_LEN_MM
            and np.isfinite(pitch)
            and PITCH_MIN_MM <= pitch <= PITCH_MAX_MM
            and snr >= PITCH_MIN_SNR
        )
        passes = periodic_ok or ts["length_mm"] >= LONG_TRACK_BYPASS_MM
        ts["pitch_mm"] = pitch
        ts["snr"] = snr
        ts["passes"] = passes
        if passes:
            accepted.append(ts)

    print(f"\n# tracks = {len(tracks)}, passing periodicity/length = {len(accepted)}\n")
    print(f"{'#':>3s} {'len_mm':>7s} {'n_in':>5s} {'pitch':>7s} {'snr':>7s} {'pass':>5s}")
    for i, ts in enumerate(tracks_ras):
        print(f"{i+1:>3d} {ts['length_mm']:>7.1f} {ts['n_inliers']:>5d} "
              f"{ts['pitch_mm']:>7.2f} {ts['snr']:>7.1f} "
              f"{'YES' if ts['passes'] else 'no':>5s}")

    gt = load_gt(subject_id)
    print(f"\n# GT trajectories = {len(gt)}")
    matches = match_tracks_to_gt(accepted, gt)
    n_matched = sum(1 for m in matches if m["matched"])
    print(f"# matched = {n_matched}/{len(gt)}")

    if n_matched < len(gt):
        print("\n# MISSED GT -- closest among ALL tracks (pre-filter):")
        print(f"{'shank':10s} {'track':>6s} {'len':>6s} {'angle':>7s} "
              f"{'mid_d':>7s} {'pitch':>7s} {'snr':>7s} {'pass':>6s}")
        for m in matches:
            if m["matched"]:
                continue
            g = next(x for x in gt if x[0] == m["gt_name"])
            _, g_s, g_e, _ = g
            g_axis, _ = _line_axis(g_s, g_e)
            g_mid = 0.5 * (g_s + g_e)
            best = None
            for ti, t in enumerate(tracks_ras):
                t_s, t_e = t["start_ras"], t["end_ras"]
                t_axis, _ = _line_axis(t_s, t_e)
                ang = float(np.degrees(np.arccos(np.clip(abs(np.dot(g_axis, t_axis)), 0, 1))))
                t_mid = 0.5 * (t_s + t_e)
                v = g_mid - t_mid
                perp = v - (v @ t_axis) * t_axis
                mid_d = float(np.linalg.norm(perp))
                score = ang + mid_d
                if best is None or score < best["score"]:
                    best = dict(ti=ti, ang=ang, mid_d=mid_d, score=score,
                                length=t["length_mm"],
                                pitch=t.get("pitch_mm", float("nan")),
                                snr=t.get("snr", float("nan")),
                                passed=t.get("passes", False))
            if best is None:
                print(f"{m['gt_name']:10s}  (no tracks)")
            else:
                print(
                    f"{m['gt_name']:10s} {best['ti']+1:>6d} {best['length']:>6.1f} "
                    f"{best['ang']:>7.2f} {best['mid_d']:>7.2f} "
                    f"{best['pitch']:>7.2f} {best['snr']:>7.1f} "
                    f"{'YES' if best['passed'] else 'no':>6s}"
                )

    print(f"\n{'shank':10s} {'gt_len':>7s} {'matched':>8s} {'angle':>7s} {'mid_d':>7s} {'track':>6s}")
    for m in matches:
        print(
            f"{m['gt_name']:10s} {m['gt_len']:7.1f} "
            f"{'YES' if m['matched'] else 'no':>8s} "
            f"{m.get('angle_deg', float('nan')):7.2f} "
            f"{m.get('mid_d_mm', float('nan')):7.2f} "
            f"{m.get('track_index', -1)+1:>6d}"
        )

    out_label = f"/tmp/detected_{subject_id}_v4.nii.gz"
    save_label_volume(accepted, frangi_s1.shape, img, out_label)
    print(f"\n# wrote {out_label}")


if __name__ == "__main__":
    subj = sys.argv[1] if len(sys.argv) > 1 else "T22"
    run(subj)
