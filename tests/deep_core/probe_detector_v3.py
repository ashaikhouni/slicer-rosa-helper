"""Probe v3: HU-agnostic electrode detector using iterative ridge tracking.

Pipeline:
  1. Build head mask + intracranial ROI (head_distance >= 10 mm).
  2. Frangi multi-scale (sigma=[1,2,3]) max -> seed/continue volume.
  3. Frangi sigma=1 -> for periodicity confirmation.
  4. Hessian at sigma=2 (6 finite-difference components) -> for ridge axis.
  5. Iterative greedy:
       seed = argmax of remaining Frangi inside mask above T_seed.
       Track from seed forward & backward using Hessian minor eigenvector.
       Step size 0.5 mm; terminate on angle change > 10 deg, Frangi < T_cont,
       or mask exit. Zero out a tube of radius 2 mm around the track, repeat.
  6. Periodicity confirmation: cylindrical sample of Frangi sigma=1 along
     polyline; accept iff peak pitch in [3.0, 4.2] mm and SNR >= 5, OR
     track length >= 40 mm.
  7. Evaluate vs ground truth (ROSA CSV for T22, shank TSV for others).

Assumes 1 mm isotropic voxel spacing (true for our dataset).

Usage:
  /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
    tests/deep_core/probe_detector_v3.py [T22|T2]
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

# ---------------------------- Config --------------------------------
T_SEED = 30.0           # min Frangi for starting a new track
T_CONTINUE = 15.0       # min Frangi for continuing a track
STEP_MM = 0.5
MAX_STEPS = 400         # hard cap per direction (200 mm)
MAX_ANGLE_DEG = 10.0    # per-step angle change limit
EXCLUSION_RADIUS_MM = 2.0
MIN_TRACK_LEN_MM = 15.0
PITCH_MIN_TRACK_LEN_MM = 22.0  # min length to accept via periodicity

PITCH_STEP_MM = 0.25
PITCH_BAND_LO = 0.2     # cycles/mm  (pitch 5 mm)
PITCH_BAND_HI = 0.4     # cycles/mm  (pitch 2.5 mm)
PITCH_MIN_MM = 2.6
PITCH_MAX_MM = 4.8
PITCH_MIN_SNR = 2.0
DEDUP_ANGLE_DEG = 5.0
DEDUP_PERP_MM = 3.0
DEDUP_OVERLAP_FRAC = 0.4
LONG_TRACK_BYPASS_MM = 40.0

MATCH_ANGLE_DEG = 10.0
MATCH_MIDPOINT_MM = 8.0

INTRACRANIAL_MIN_DISTANCE_MM = 10.0
# --------------------------------------------------------------------


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
    v000 = arr[k0, j0, i0]
    v001 = arr[k0, j0, i0 + 1]
    v010 = arr[k0, j0 + 1, i0]
    v011 = arr[k0, j0 + 1, i0 + 1]
    v100 = arr[k0 + 1, j0, i0]
    v101 = arr[k0 + 1, j0, i0 + 1]
    v110 = arr[k0 + 1, j0 + 1, i0]
    v111 = arr[k0 + 1, j0 + 1, i0 + 1]
    c00 = v000 * (1 - di) + v001 * di
    c01 = v010 * (1 - di) + v011 * di
    c10 = v100 * (1 - di) + v101 * di
    c11 = v110 * (1 - di) + v111 * di
    c0 = c00 * (1 - dj) + c01 * dj
    c1 = c10 * (1 - dj) + c11 * dj
    return float(c0 * (1 - dk) + c1 * dk)


def _sample_mask(mask, kji_float):
    k, j, i = int(round(kji_float[0])), int(round(kji_float[1])), int(round(kji_float[2]))
    s = mask.shape
    if 0 <= k < s[0] and 0 <= j < s[1] and 0 <= i < s[2]:
        return bool(mask[k, j, i])
    return False


# ---------------------------- Masks --------------------------------
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
    intracranial = dist_arr >= INTRACRANIAL_MIN_DISTANCE_MM
    return hull_arr, intracranial, dist_arr


# ---------------------------- Frangi --------------------------------
def frangi_multiscale_max(img, sigmas):
    import SimpleITK as sitk
    max_arr = None
    for s in sigmas:
        sm = sitk.SmoothingRecursiveGaussian(img, sigma=float(s))
        ob = sitk.ObjectnessMeasure(sm, objectDimension=1, brightObject=True)
        a = sitk.GetArrayFromImage(ob).astype(np.float32)
        max_arr = a if max_arr is None else np.maximum(max_arr, a)
    return max_arr


def frangi_single(img, sigma):
    import SimpleITK as sitk
    sm = sitk.SmoothingRecursiveGaussian(img, sigma=float(sigma))
    ob = sitk.ObjectnessMeasure(sm, objectDimension=1, brightObject=True)
    return sitk.GetArrayFromImage(ob).astype(np.float32)


# ---------------------------- Hessian --------------------------------
def hessian_components_kji(img, sigma):
    """Pre-smooth CT at sigma, then central-difference 6 Hessian components.

    KJI ordering: axis 0 = k (z), axis 1 = j (y), axis 2 = i (x).
    Returns (Hkk, Hjj, Hii, Hkj, Hki, Hji) as float32 arrays.
    """
    import SimpleITK as sitk
    sm = sitk.SmoothingRecursiveGaussian(img, sigma=float(sigma))
    f = sitk.GetArrayFromImage(sm).astype(np.float32)

    Hkk = np.zeros_like(f); Hjj = np.zeros_like(f); Hii = np.zeros_like(f)
    Hkj = np.zeros_like(f); Hki = np.zeros_like(f); Hji = np.zeros_like(f)

    Hkk[1:-1, :, :] = f[2:, :, :] - 2 * f[1:-1, :, :] + f[:-2, :, :]
    Hjj[:, 1:-1, :] = f[:, 2:, :] - 2 * f[:, 1:-1, :] + f[:, :-2, :]
    Hii[:, :, 1:-1] = f[:, :, 2:] - 2 * f[:, :, 1:-1] + f[:, :, :-2]

    Hkj[1:-1, 1:-1, :] = 0.25 * (
        f[2:, 2:, :] - f[2:, :-2, :] - f[:-2, 2:, :] + f[:-2, :-2, :]
    )
    Hki[1:-1, :, 1:-1] = 0.25 * (
        f[2:, :, 2:] - f[2:, :, :-2] - f[:-2, :, 2:] + f[:-2, :, :-2]
    )
    Hji[:, 1:-1, 1:-1] = 0.25 * (
        f[:, 2:, 2:] - f[:, 2:, :-2] - f[:, :-2, 2:] + f[:, :-2, :-2]
    )
    return Hkk, Hjj, Hii, Hkj, Hki, Hji


def sample_hessian(H, kji_float):
    Hkk, Hjj, Hii, Hkj, Hki, Hji = H
    hkk = _sample_trilinear(Hkk, kji_float)
    hjj = _sample_trilinear(Hjj, kji_float)
    hii = _sample_trilinear(Hii, kji_float)
    hkj = _sample_trilinear(Hkj, kji_float)
    hki = _sample_trilinear(Hki, kji_float)
    hji = _sample_trilinear(Hji, kji_float)
    return np.array([
        [hkk, hkj, hki],
        [hkj, hjj, hji],
        [hki, hji, hii],
    ], dtype=float)


def tube_axis_from_hessian(Hmat):
    """Return unit vector along the tube axis (smallest abs eigenvalue).
    Returns None if Hessian is degenerate / nan.
    """
    if not np.all(np.isfinite(Hmat)):
        return None
    try:
        eigvals, eigvecs = np.linalg.eigh(Hmat)
    except np.linalg.LinAlgError:
        return None
    idx = int(np.argmin(np.abs(eigvals)))
    v = eigvecs[:, idx]
    n = float(np.linalg.norm(v))
    if n < 1e-9 or not np.all(np.isfinite(v)):
        return None
    return v / n


# ---------------------------- Tracking --------------------------------
def track_from_seed(seed_kji, frangi, H, intracranial_mask):
    max_angle_rad = np.radians(MAX_ANGLE_DEG)
    seed = np.asarray(seed_kji, dtype=float)
    Hm = sample_hessian(H, seed)
    axis0 = tube_axis_from_hessian(Hm)
    if axis0 is None:
        return np.asarray([seed], dtype=float)

    def walk(sign):
        wp = []
        pos = seed.copy()
        prev_axis = sign * axis0
        for _ in range(MAX_STEPS):
            Hm = sample_hessian(H, pos)
            axis = tube_axis_from_hessian(Hm)
            if axis is None:
                break
            if np.dot(axis, prev_axis) < 0:
                axis = -axis
            cos_a = float(np.clip(np.dot(axis, prev_axis), -1.0, 1.0))
            if np.arccos(abs(cos_a)) > max_angle_rad:
                break
            new_pos = pos + STEP_MM * axis  # 1 mm iso -> direct step
            if not _sample_mask(intracranial_mask, new_pos):
                break
            fr = _sample_trilinear(frangi, new_pos)
            if not np.isfinite(fr) or fr < T_CONTINUE:
                break
            pos = new_pos
            prev_axis = axis
            wp.append(pos.copy())
        return wp

    fwd = walk(+1.0)
    bwd = walk(-1.0)
    poly = list(reversed(bwd)) + [seed.copy()] + fwd
    return np.asarray(poly, dtype=float)


def zero_tube_around(frangi, polyline_kji, radius):
    """Zero out voxels within `radius` of any point in polyline."""
    shape = frangi.shape
    r = int(np.ceil(radius)) + 1
    r2 = radius * radius
    for p in polyline_kji:
        kc, jc, ic = int(round(p[0])), int(round(p[1])), int(round(p[2]))
        k_lo, k_hi = max(0, kc - r), min(shape[0], kc + r + 1)
        j_lo, j_hi = max(0, jc - r), min(shape[1], jc + r + 1)
        i_lo, i_hi = max(0, ic - r), min(shape[2], ic + r + 1)
        kk, jj, ii = np.meshgrid(
            np.arange(k_lo, k_hi) - kc,
            np.arange(j_lo, j_hi) - jc,
            np.arange(i_lo, i_hi) - ic,
            indexing="ij",
        )
        d2 = kk * kk + jj * jj + ii * ii
        block = frangi[k_lo:k_hi, j_lo:j_hi, i_lo:i_hi]
        block[d2 <= r2] = 0.0


def _track_axis_and_span(polyline):
    center = polyline.mean(axis=0)
    X = polyline - center
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    axis = _unit(Vt[0])
    proj = X @ axis
    lo, hi = float(proj.min()), float(proj.max())
    return axis, center, proj, lo, hi


def dedup_tracks(tracks):
    """Merge near-parallel tracks that overlap spatially. Keep the longer one."""
    if len(tracks) < 2:
        return tracks
    info = []
    for t in tracks:
        poly = t["polyline_kji"]
        axis, center, proj, lo, hi = _track_axis_and_span(poly)
        info.append(dict(axis=axis, center=center, lo=lo, hi=hi,
                          length=t["length_mm"], t=t))

    keep_mask = [True] * len(info)
    for i in range(len(info)):
        if not keep_mask[i]:
            continue
        ai = info[i]
        for j in range(i + 1, len(info)):
            if not keep_mask[j]:
                continue
            aj = info[j]
            ang = float(np.degrees(np.arccos(np.clip(abs(np.dot(ai["axis"], aj["axis"])), 0, 1))))
            if ang > DEDUP_ANGLE_DEG:
                continue
            # perpendicular distance between the two line centers
            dv = aj["center"] - ai["center"]
            par = dv @ ai["axis"]
            perp = dv - par * ai["axis"]
            perp_d = float(np.linalg.norm(perp))
            if perp_d > DEDUP_PERP_MM:
                continue
            # axial overlap: project j's endpoints onto i's axis, compare with i's extent
            j_lo = par + aj["lo"]
            j_hi = par + aj["hi"]
            overlap_lo = max(ai["lo"], j_lo)
            overlap_hi = min(ai["hi"], j_hi)
            overlap = max(0.0, overlap_hi - overlap_lo)
            shorter = min(ai["hi"] - ai["lo"], aj["hi"] - aj["lo"])
            if shorter <= 1e-6:
                continue
            overlap_frac = overlap / shorter
            if overlap_frac < DEDUP_OVERLAP_FRAC:
                continue
            # merge: drop the shorter
            if ai["length"] >= aj["length"]:
                keep_mask[j] = False
            else:
                keep_mask[i] = False
                break
    return [info[i]["t"] for i in range(len(info)) if keep_mask[i]]


def iterative_detect(frangi_max, hessian, intracranial_mask,
                     frangi_s1=None, hessian_s1=None,
                     t_seed_secondary=50.0, secondary_exclusion_mm=1.0):
    """Primary pass on frangi_max (hessian at sigma=2 for direction).
    Optional secondary pass on frangi_s1 with hessian_s1 for direction and
    tighter exclusion -- catches close-parallel neighbors whose axis is
    confused at sigma=2.
    """
    remaining = frangi_max.copy()
    remaining[~intracranial_mask] = 0.0
    tracks = []

    for iteration in range(200):
        max_val = float(remaining.max())
        if max_val < T_SEED:
            break
        seed_kji = np.unravel_index(int(np.argmax(remaining)), remaining.shape)
        poly = track_from_seed(seed_kji, remaining, hessian, intracranial_mask)
        if poly.shape[0] < 2:
            remaining[seed_kji] = 0.0
            continue
        seg_lens = np.linalg.norm(np.diff(poly, axis=0), axis=1)
        length_mm = float(seg_lens.sum())
        zero_tube_around(remaining, poly, radius=EXCLUSION_RADIUS_MM)
        if length_mm >= MIN_TRACK_LEN_MM:
            tracks.append({"polyline_kji": poly, "length_mm": length_mm, "pass": "primary"})

    if frangi_s1 is None:
        return tracks

    # Secondary pass using Frangi sigma=1 -- separates close-parallel electrodes.
    # Zero out s1 around existing tracks with TIGHT exclusion (~1 mm),
    # then iteratively seed + track using the sigma=2 Hessian (still good direction).
    s1_remaining = frangi_s1.copy()
    s1_remaining[~intracranial_mask] = 0.0
    for t in tracks:
        zero_tube_around(s1_remaining, t["polyline_kji"],
                         radius=secondary_exclusion_mm)

    secondary_hessian = hessian_s1 if hessian_s1 is not None else hessian
    for iteration in range(200):
        max_val = float(s1_remaining.max())
        if max_val < t_seed_secondary:
            break
        seed_kji = np.unravel_index(int(np.argmax(s1_remaining)), s1_remaining.shape)
        poly = track_from_seed(seed_kji, s1_remaining, secondary_hessian, intracranial_mask)
        if poly.shape[0] < 2:
            s1_remaining[seed_kji] = 0.0
            continue
        seg_lens = np.linalg.norm(np.diff(poly, axis=0), axis=1)
        length_mm = float(seg_lens.sum())
        zero_tube_around(s1_remaining, poly, radius=secondary_exclusion_mm)
        if length_mm >= MIN_TRACK_LEN_MM:
            tracks.append({"polyline_kji": poly, "length_mm": length_mm, "pass": "secondary"})

    return tracks


# ---------------------------- Periodicity --------------------------------
def _orthonormal_basis(d):
    any_vec = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(d, any_vec)) > 0.9:
        any_vec = np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(d, any_vec))
    v = _unit(np.cross(d, u))
    return u, v


def periodicity_along_polyline(frangi_s1, polyline_kji, intracranial_mask):
    """Sample Frangi sigma=1 with a 1mm cylinder along the polyline.
    Return (peak_pitch_mm, snr).
    """
    if polyline_kji.shape[0] < 3:
        return float("nan"), float("nan")
    # Build a straight resampling axis from PC1 (handles gentle curvature OK)
    center = polyline_kji.mean(axis=0)
    X = polyline_kji - center
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    axis = _unit(Vt[0])
    proj = X @ axis
    lo, hi = float(proj.min()), float(proj.max())
    length_mm = hi - lo
    if length_mm < 15.0:
        return float("nan"), float("nan")
    n_steps = int(length_mm / PITCH_STEP_MM) + 1
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
    band = power[mask]
    bfreq = freqs[mask]
    pi = int(np.argmax(band))
    peak_pitch = 1.0 / float(bfreq[pi]) if bfreq[pi] > 0 else float("nan")
    peak_power = float(band[pi])
    excl = np.zeros_like(band, dtype=bool)
    excl[max(0, pi - 3):pi + 4] = True
    bg = band[~excl]
    bg_power = float(bg.mean()) if bg.size else 0.0
    snr = peak_power / bg_power if bg_power > 0 else float("inf")
    return peak_pitch, snr


# ---------------------------- GT loading --------------------------------
def load_gt(subject_id):
    """Return list of (name, start_ras, end_ras, length_mm)."""
    # Prefer ROSA CSV if present
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
                     float(row["z_world_ras"])],
                    dtype=float,
                )
                traj.setdefault(name, {})[row["point_type"]] = pt
        out = []
        for name, pair in traj.items():
            if "entry" in pair and "target" in pair:
                L = float(np.linalg.norm(pair["target"] - pair["entry"]))
                out.append((name, pair["entry"], pair["target"], L))
        return out
    # Fallback: shank TSV
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


# ---------------------------- Evaluation --------------------------------
def _line_axis(start, end):
    d = end - start
    L = float(np.linalg.norm(d))
    return _unit(d), L


def match_tracks_to_gt(tracks_ras, gt):
    """Greedy 1-to-1 assignment by combined (angle + midpoint distance).

    Returns list of GT matches. Each track used at most once.
    """
    # Build all candidate pairs with eligibility
    pairs = []  # (score, gt_i, track_i, angle, mid_d)
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


# ---------------------------- Output --------------------------------
def save_label_volume(tracks, shape, img, out_path, spacing_ras_to_ijk=None):
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


# ---------------------------- Main --------------------------------
def run(subject_id):
    import SimpleITK as sitk
    from shank_core.io import image_ijk_ras_matrices

    ct_path = DATASET_ROOT / "post_registered_ct" / f"{subject_id}_post_registered.nii.gz"
    print(f"# subject={subject_id}")
    print(f"# ct={ct_path}")

    img = sitk.ReadImage(str(ct_path))
    spacing = img.GetSpacing()
    if not np.allclose(spacing, [1.0, 1.0, 1.0]):
        print(f"! warning: non-iso spacing {spacing}; tracking assumes 1mm iso")

    t0 = time.time()
    hull, intracranial, dist = build_masks(img)
    print(f"# masks: hull={hull.sum()}  intracranial={intracranial.sum()}  "
          f"({time.time()-t0:.1f}s)")

    t0 = time.time()
    frangi_max = frangi_multiscale_max(img, sigmas=(1.0, 2.0, 3.0))
    frangi_s1 = frangi_single(img, sigma=1.0)
    print(f"# frangi: max={frangi_max.max():.1f} p99={np.percentile(frangi_max, 99):.1f} "
          f"({time.time()-t0:.1f}s)")

    t0 = time.time()
    H = hessian_components_kji(img, sigma=2.0)
    H_s1 = hessian_components_kji(img, sigma=1.0)
    print(f"# hessian sigma=2 + sigma=1: ({time.time()-t0:.1f}s)")

    t0 = time.time()
    tracks = iterative_detect(frangi_max, H, intracranial, frangi_s1=frangi_s1,
                              hessian_s1=H_s1,
                              t_seed_secondary=25.0, secondary_exclusion_mm=0.8)
    n_primary = sum(1 for t in tracks if t.get("pass") == "primary")
    n_secondary = sum(1 for t in tracks if t.get("pass") == "secondary")
    n_pre_dedup = len(tracks)
    tracks = dedup_tracks(tracks)
    print(f"# iterative tracking: {n_pre_dedup} tracks "
          f"(primary={n_primary}, secondary={n_secondary}), "
          f"after dedup={len(tracks)} ({time.time()-t0:.1f}s)")

    # Convert polylines to RAS and compute endpoints
    ijk_to_ras_mat, ras_to_ijk_mat = image_ijk_ras_matrices(img)
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
            "polyline_kji": t["polyline_kji"],
        })

    # Periodicity confirmation
    accepted = []
    for i, (t, ts) in enumerate(zip(tracks, tracks_ras)):
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
    print(f"{'#':>3s} {'len_mm':>7s} {'pitch':>7s} {'snr':>7s} {'pass':>5s}")
    for i, ts in enumerate(tracks_ras):
        print(f"{i+1:>3d} {ts['length_mm']:>7.1f} "
              f"{ts['pitch_mm']:>7.2f} {ts['snr']:>7.1f} "
              f"{'YES' if ts['passes'] else 'no':>5s}")

    # Evaluate vs GT
    gt = load_gt(subject_id)
    print(f"\n# GT trajectories = {len(gt)}")
    matches = match_tracks_to_gt(accepted, gt)
    n_matched = sum(1 for m in matches if m["matched"])
    print(f"# matched = {n_matched}/{len(gt)}")

    # Diagnostic: for each missed GT, find closest track among ALL (not just accepted)
    if n_matched < len(gt):
        print("\n# MISSED GT -- closest track among ALL tracks (before filtering):")
        print(f"{'shank':10s} {'track':>6s} {'len_mm':>7s} {'angle':>7s} "
              f"{'mid_d':>7s} {'pitch':>7s} {'snr':>7s} {'passed':>8s}")
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
                                length=t["length_mm"], pitch=t.get("pitch_mm", float("nan")),
                                snr=t.get("snr", float("nan")),
                                passed=t.get("passes", False))
            if best is None:
                print(f"{m['gt_name']:10s}  (no tracks)")
            else:
                print(
                    f"{m['gt_name']:10s} {best['ti']+1:>6d} {best['length']:>7.1f} "
                    f"{best['ang']:>7.2f} {best['mid_d']:>7.2f} "
                    f"{best['pitch']:>7.2f} {best['snr']:>7.1f} "
                    f"{'YES' if best['passed'] else 'no':>8s}"
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

    # Save label volume
    out_label = f"/tmp/detected_{subject_id}_v3.nii.gz"
    save_label_volume(accepted, frangi_max.shape, img, out_label)
    print(f"\n# wrote {out_label}")


if __name__ == "__main__":
    subj = sys.argv[1] if len(sys.argv) > 1 else "T22"
    run(subj)
