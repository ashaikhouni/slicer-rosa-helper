"""Probe: periodicity of the SEEG contact chain along GT trajectories.

Along each ROSA (entry, target) axis, sample CT intensity at fine steps and
check for a periodic modulation at the expected inter-contact spacing (~3.5
mm). Compute:
  - autocorrelation of the profile
  - FFT power spectrum
  - report peak frequency and periodic SNR

Also sample a deliberately-OFF axis (shifted 2 mm perpendicular) as a
negative control — periodicity should collapse.

Tests both T22 (clean) and T2 (clipped) -- but only T22 has ROSA GT.
For T2 we reuse the shank TSV start/end as a proxy.

Usage:
  /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
    tests/deep_core/probe_periodicity.py [T22|T2]
"""
from __future__ import annotations

import csv
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
EXPECTED_PITCH_MM = 3.5


def _unit(v):
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def _ras_to_kji(pt_ras, ras_to_ijk_mat):
    h = np.array([pt_ras[0], pt_ras[1], pt_ras[2], 1.0])
    ijk = ras_to_ijk_mat @ h
    return np.array([ijk[2], ijk[1], ijk[0]], dtype=float)


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


def _orthonormal_basis(direction_unit):
    any_vec = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(direction_unit, any_vec)) > 0.9:
        any_vec = np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(direction_unit, any_vec))
    v = _unit(np.cross(direction_unit, u))
    return u, v


def sample_profile(arr_kji, ras_to_ijk_mat, start_ras, direction_unit, n_steps,
                   step_mm, perp_offset_mm=0.0, cyl_radius_mm=0.0, n_ring=8,
                   reducer="mean"):
    """Sample along the axis. If cyl_radius_mm>0, sample a disk of radius
    cyl_radius_mm perpendicular to the axis at each step and reduce with
    `reducer` ('mean' or 'max').

    `perp_offset_mm` shifts the whole line perpendicularly (for off-axis
    negative control).
    """
    u, v = _orthonormal_basis(direction_unit)
    shift = u * perp_offset_mm

    vals = np.empty(n_steps, dtype=float)
    for i in range(n_steps):
        center = start_ras + i * step_mm * direction_unit + shift
        if cyl_radius_mm <= 0.0:
            kji = _ras_to_kji(center, ras_to_ijk_mat)
            vals[i] = _sample_trilinear(arr_kji, kji)
        else:
            # Center + ring of n_ring samples at cyl_radius_mm
            samples = [_sample_trilinear(arr_kji, _ras_to_kji(center, ras_to_ijk_mat))]
            for k in range(n_ring):
                ang = 2.0 * np.pi * k / n_ring
                offset = cyl_radius_mm * (np.cos(ang) * u + np.sin(ang) * v)
                p = center + offset
                samples.append(_sample_trilinear(arr_kji, _ras_to_kji(p, ras_to_ijk_mat)))
            s = np.asarray(samples, dtype=float)
            s = s[np.isfinite(s)]
            if s.size == 0:
                vals[i] = float("nan")
            elif reducer == "max":
                vals[i] = float(s.max())
            else:
                vals[i] = float(s.mean())
    return vals


def periodic_snr(profile, step_mm, expected_pitch_mm):
    """FFT power at the expected pitch vs average background power.

    Return (peak_pitch_mm, snr, periodic_power, background_power).
    """
    x = np.asarray(profile, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 20:
        return float("nan"), float("nan"), float("nan"), float("nan")
    x = x - np.mean(x)
    n = x.size
    # rfft
    spec = np.fft.rfft(x)
    power = (np.abs(spec) ** 2) / n
    freqs = np.fft.rfftfreq(n, d=step_mm)  # cycles / mm
    # Tight search around Dixi 3.5mm pitch: 2.5-5mm -> 0.2-0.4 cycles/mm
    mask = (freqs > 0.2) & (freqs < 0.4)
    if not mask.any():
        return float("nan"), float("nan"), float("nan"), float("nan")
    band_freqs = freqs[mask]
    band_power = power[mask]
    peak_i = int(np.argmax(band_power))
    peak_freq = float(band_freqs[peak_i])
    peak_pitch = 1.0 / peak_freq if peak_freq > 0 else float("nan")
    peak_power = float(band_power[peak_i])
    # Background: exclude 10 bins around the peak and within pitch 3-4mm
    exclude = np.zeros_like(band_power, dtype=bool)
    exclude[max(0, peak_i - 3):peak_i + 4] = True
    bg = band_power[~exclude]
    bg_power = float(bg.mean()) if bg.size > 0 else 0.0
    snr = peak_power / bg_power if bg_power > 0 else float("inf")
    return peak_pitch, snr, peak_power, bg_power


def load_rosa(subject_id):
    csv_path = (
        DATASET_ROOT / "contact_label_dataset" / "rosa_helper_import"
        / subject_id / "ROSA_Contacts_final_trajectory_points.csv"
    )
    if not csv_path.exists():
        return None
    traj = {}
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            name = row["trajectory"]
            pt = np.array(
                [float(row["x_world_ras"]), float(row["y_world_ras"]), float(row["z_world_ras"])],
                dtype=float,
            )
            traj.setdefault(name, {})[row["point_type"]] = pt
    return traj


def load_tsv_shanks(subject_id):
    """Fallback for T2 and others: use the shank GT TSV."""
    from eval_seeg_localization import iter_subject_rows, load_ground_truth_shanks
    rows = iter_subject_rows(DATASET_ROOT, {subject_id})
    if not rows:
        return None
    row = rows[0]
    return load_ground_truth_shanks(row["labels_path"], row.get("shanks_path"))


def run(subject_id):
    import SimpleITK as sitk
    from shank_core.io import image_ijk_ras_matrices

    ct_path = DATASET_ROOT / "post_registered_ct" / f"{subject_id}_post_registered.nii.gz"
    print(f"# subject={subject_id}  ct={ct_path}")

    img = sitk.ReadImage(str(ct_path))
    ct_arr = sitk.GetArrayFromImage(img).astype(np.float32)
    _, ras_to_ijk_mat = image_ijk_ras_matrices(img)
    ras_to_ijk_mat = np.asarray(ras_to_ijk_mat, dtype=float)

    # Compute Frangi sigma=1 (contact-scale).
    print("# computing Frangi sigma=1 ...")
    sm = sitk.SmoothingRecursiveGaussian(img, sigma=1.0)
    ob = sitk.ObjectnessMeasure(sm, objectDimension=1, brightObject=True)
    fr1_arr = sitk.GetArrayFromImage(ob).astype(np.float32)

    # Frangi sigma=[2,3] if cached
    fr23_arr = None
    fr23_path = Path(f"/tmp/frangi23_{subject_id}.nii.gz")
    if fr23_path.exists():
        fr23_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(fr23_path))).astype(np.float32)

    # Decide trajectory source.
    # For each trajectory we want a sampling range that excludes the
    # bolt/skull zone. For ROSA (entry, target): skip 20 mm from entry.
    # For shank TSV: entries start at contact 1 which is already intra-brain,
    # so use start..end directly.
    rosa = load_rosa(subject_id)
    pairs = []  # (name, sample_start_ras, direction_unit, sample_len_mm)
    if rosa:
        for name, pair in rosa.items():
            if "entry" not in pair or "target" not in pair:
                continue
            entry = pair["entry"]
            target = pair["target"]
            du = _unit(target - entry)
            full_len = float(np.linalg.norm(target - entry))
            # skip first 20 mm (bolt+skull zone), sample to target+5mm
            sample_start = entry + 20.0 * du
            sample_len = (full_len - 20.0) + 5.0
            if sample_len < 15.0:
                continue
            pairs.append((name, sample_start, du, sample_len))
    else:
        shanks = load_tsv_shanks(subject_id)
        if not shanks:
            print("# no GT available")
            return
        for gt in shanks:
            start = np.asarray(gt.start_ras, dtype=float)
            end = np.asarray(gt.end_ras, dtype=float)
            du = _unit(end - start)
            L = float(np.linalg.norm(end - start))
            pairs.append((gt.shank, start, du, L + 5.0))

    print(
        f"# trajectories={len(pairs)} step={STEP_MM}mm pitch_target={EXPECTED_PITCH_MM}mm"
    )
    print("# cylindrical sampling: radius=1mm, 8-point ring, reducer=max")
    print("# search band: pitch 2.5-5mm\n")
    print(
        f"{'shank':8s} {'len_mm':>7s} "
        f"{'CT_pitch':>9s} {'CT_snr':>7s} "
        f"{'Fr1_pitch':>10s} {'Fr1_snr':>8s} "
        f"{'Fr1_off_snr':>11s}"
        f"{' Fr23_pitch' if fr23_arr is not None else ''}"
        f"{' Fr23_snr' if fr23_arr is not None else ''}"
    )

    for name, sample_start, du, sample_len in pairs:
        n_steps = int(sample_len / STEP_MM) + 1

        ct_prof = sample_profile(
            ct_arr, ras_to_ijk_mat, sample_start, du, n_steps, STEP_MM,
            cyl_radius_mm=1.0, reducer="max",
        )
        fr1_prof = sample_profile(
            fr1_arr, ras_to_ijk_mat, sample_start, du, n_steps, STEP_MM,
            cyl_radius_mm=1.0, reducer="max",
        )
        fr1_off = sample_profile(
            fr1_arr, ras_to_ijk_mat, sample_start, du, n_steps, STEP_MM,
            perp_offset_mm=3.0, cyl_radius_mm=1.0, reducer="max",
        )

        ct_pitch, ct_snr, _, _ = periodic_snr(ct_prof, STEP_MM, EXPECTED_PITCH_MM)
        fr1_pitch, fr1_snr, _, _ = periodic_snr(fr1_prof, STEP_MM, EXPECTED_PITCH_MM)
        _, fr1_off_snr, _, _ = periodic_snr(fr1_off, STEP_MM, EXPECTED_PITCH_MM)

        line = (
            f"{name:8s} {sample_len:7.1f} "
            f"{ct_pitch:9.2f} {ct_snr:7.1f} "
            f"{fr1_pitch:10.2f} {fr1_snr:8.1f} "
            f"{fr1_off_snr:11.1f}"
        )
        if fr23_arr is not None:
            fr23_prof = sample_profile(
                fr23_arr, ras_to_ijk_mat, sample_start, du, n_steps, STEP_MM,
                cyl_radius_mm=1.0, reducer="max",
            )
            fr23_pitch, fr23_snr, _, _ = periodic_snr(fr23_prof, STEP_MM, EXPECTED_PITCH_MM)
            line += f" {fr23_pitch:10.2f} {fr23_snr:8.1f}"
        print(line)


if __name__ == "__main__":
    subj = sys.argv[1] if len(sys.argv) > 1 else "T22"
    run(subj)
