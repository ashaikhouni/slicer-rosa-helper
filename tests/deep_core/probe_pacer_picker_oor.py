"""PaCER OOR axis refit — does it earn its keep on T22 vs wider-disk alone?

Phase 2 follow-up to ``probe_pacer_picker_robustness.py``. The wider-
disk picker handled most of the realistic Auto-Fit noise budget (1-2
mm lateral, 1-3° tilt) at 0.85+ match rate. Two cases it can't
recover from:
  1. Bending shanks (brain shift) — straight axis can't fit a curve.
  2. Hard tilt ≥3° / lateral ≥3 mm — disk eventually misses the metal.

PaCER's OOR (Optimal Oblique Resampling) handles both: at each axial
step it samples a transverse grid, threshold-weighted centroids the
metal, and re-fits a polynomial through the centroid skeleton. The
refined polynomial then carries the picker.

Implementation here is intentionally cheap:
  - Single pass (no degree-8 internal then degree-3 final).
  - Loosened SEEG settings: 0.5 mm along-axis, 0.25 mm in-plane on a
    2.0 mm radius disk (~80 transverse points × 160 axial steps ≈
    12k samples per trajectory). Vectorized via numpy where possible.
  - Polynomial degree 2 (gentle bend; SEEG shanks rarely curve more).

Compare per-trajectory:
  A. straight axis from GT (phase-1 baseline; should be 9/9)
  B. perturbed axis (lateral=2 mm, tilt=2°)
  C. perturbed axis + OOR refit
  D. perturbed axis (lateral=3 mm, tilt=5°) — vanilla cliff
  E. (D) + OOR refit

If C ≥ B and E recovers most of D's loss, OOR earns its keep.

Run:
    cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper && \\
      /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
      tests/deep_core/probe_pacer_picker_oor.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "CommonLib"))
sys.path.insert(0, str(ROOT / "tests" / "deep_core"))

from rosa_core.contact_peak_fit import sample_axis_profile  # noqa: E402
from rosa_core.electrode_models import load_electrode_library  # noqa: E402

from probe_pacer_picker_t22 import (  # noqa: E402
    GT_FILE,
    CT_FILE,
    PROFILE_STEP_MM,
    PROFILE_REDUCER,
    TEMPLATE_OFFSET_STEP_MM,
    TEMPLATE_SIGMA_MM,
    PAD_TIP_MM,
    PAD_ENTRY_MM,
    TIP_ARC_SEARCH_HALF_WIDTH_MM,
    axis_from_gt,
    build_template,
    load_ct_volume,
    load_gt_trajectories,
    normalized_cross_correlation,
)
from probe_pacer_picker_robustness import perturb_axis  # noqa: E402


# OOR settings — loosened from PaCER for SEEG.
OOR_AXIAL_STEP_MM = 0.5
OOR_DISK_RADIUS_MM = 2.0
OOR_GRID_STEP_MM = 0.25
OOR_THRESHOLD_HU = 1500.0    # metal is bright; tissue and bone are below
OOR_POLY_DEGREE = 1          # straight shank baseline; raise to 2 once bending fixture is available

# Wider-disk picker config (winning baseline from robustness sweep).
PICKER_DISK_RADIUS_MM = 2.5
PICKER_DISK_N_RADII = 4
PICKER_DISK_N_ANGLES = 8


# ---------------------------------------------------------------------
# OOR axis refit
# ---------------------------------------------------------------------

def _orthonormal_basis(direction_unit):
    any_vec = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(direction_unit, any_vec)) > 0.9:
        any_vec = np.array([0.0, 1.0, 0.0])
    u = np.cross(direction_unit, any_vec)
    u /= np.linalg.norm(u)
    v = np.cross(direction_unit, u)
    v /= np.linalg.norm(v)
    return u, v


def _build_disk_offsets(direction_unit, radius_mm, grid_step_mm):
    """Return (M, 3) array of in-plane offsets covering a disk of given radius
    on a regular Cartesian grid in the transverse plane."""
    u, v = _orthonormal_basis(direction_unit)
    n = int(radius_mm / grid_step_mm)
    coords = np.arange(-n, n + 1) * grid_step_mm
    pts = []
    for du in coords:
        for dv in coords:
            if du * du + dv * dv <= radius_mm * radius_mm + 1e-9:
                pts.append(du * u + dv * v)
    return np.array(pts, dtype=float)


def _trilinear_batch(volume_kji, points_kji):
    """Batched trilinear interpolation. ``points_kji`` is (N, 3) in (k, j, i)
    order matching ``volume_kji.shape``. Returns (N,) with NaN out-of-bounds.
    """
    arr = volume_kji
    s = arr.shape
    p = np.asarray(points_kji, dtype=float)
    k, j, i = p[:, 0], p[:, 1], p[:, 2]
    valid = (
        (k >= 0) & (k < s[0] - 1) &
        (j >= 0) & (j < s[1] - 1) &
        (i >= 0) & (i < s[2] - 1)
    )
    out = np.full(p.shape[0], np.nan, dtype=float)
    if not np.any(valid):
        return out
    k = k[valid]; j = j[valid]; i = i[valid]
    k0 = k.astype(np.int64); j0 = j.astype(np.int64); i0 = i.astype(np.int64)
    dk = k - k0; dj = j - j0; di = i - i0
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
    vals = c0 * (1 - dk) + c1 * dk
    out[valid] = vals
    return out


def _ras_to_kji_batch(ras_to_ijk_mat, points_ras):
    """Convert (N, 3) RAS points to (N, 3) (k, j, i) per `_sample_trilinear`'s
    convention (volume is k-j-i indexed; matrix is i-j-k row order)."""
    pts = np.asarray(points_ras, dtype=float)
    h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=float)])
    ijk = (ras_to_ijk_mat @ h.T).T  # (N, 4)
    out = np.empty((pts.shape[0], 3), dtype=float)
    out[:, 0] = ijk[:, 2]  # k
    out[:, 1] = ijk[:, 1]  # j
    out[:, 2] = ijk[:, 0]  # i
    return out


def refine_axis_oor(volume_kji, ras_to_ijk_mat, start_ras, end_ras):
    """PaCER-style OOR axis refit. Returns parametric polynomial coefficients
    for x(t), y(t), z(t) with t ∈ [0, 1] along the refined arc.

    At each axial step along the seed (start, end) line, sample a transverse
    disk on a regular Cartesian grid, threshold to keep bright metal, and
    take the threshold-weighted 3D centroid as the refined skeleton point.
    Fit a degree-OOR_POLY_DEGREE polynomial to the skeleton.
    """
    s = np.asarray(start_ras, dtype=float)
    e = np.asarray(end_ras, dtype=float)
    direction = e - s
    L = float(np.linalg.norm(direction))
    if L < 1e-3:
        raise ValueError("axis too short")
    direction_unit = direction / L
    disk_offsets_ras = _build_disk_offsets(
        direction_unit, OOR_DISK_RADIUS_MM, OOR_GRID_STEP_MM,
    )  # (M, 3)
    n_steps = int(L / OOR_AXIAL_STEP_MM) + 1
    arc_mm = np.arange(n_steps, dtype=float) * OOR_AXIAL_STEP_MM
    centers_ras = s[None, :] + arc_mm[:, None] * direction_unit[None, :]  # (N, 3)
    # Sample (N, M) intensities by tiling centers + disk offsets.
    M = disk_offsets_ras.shape[0]
    sample_ras = (
        centers_ras[:, None, :] + disk_offsets_ras[None, :, :]
    ).reshape(-1, 3)  # (N*M, 3)
    sample_kji = _ras_to_kji_batch(ras_to_ijk_mat, sample_ras)
    intensities = _trilinear_batch(volume_kji, sample_kji).reshape(n_steps, M)
    # Threshold-weighted centroid in RAS at each axial step.
    weights = np.where(intensities >= OOR_THRESHOLD_HU, intensities - OOR_THRESHOLD_HU, 0.0)
    weights[~np.isfinite(weights)] = 0.0
    sample_ras_grid = sample_ras.reshape(n_steps, M, 3)
    sum_w = weights.sum(axis=1)
    valid_step = sum_w > 1e-6
    if valid_step.sum() < OOR_POLY_DEGREE + 2:
        # Not enough metal-bearing axial steps; fall back to seed.
        return None
    centroid_ras = (
        (sample_ras_grid * weights[:, :, None]).sum(axis=1) / np.where(sum_w > 0, sum_w, 1.0)[:, None]
    )
    skeleton = centroid_ras[valid_step]
    # Re-parameterize t ∈ [0, 1] by cumulative arc length along the skeleton.
    diffs = np.diff(skeleton, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    if cum[-1] < 1e-6:
        return None
    t_params = cum / cum[-1]
    # Fit polynomial per axis.
    coeffs = np.zeros((3, OOR_POLY_DEGREE + 1), dtype=float)
    for ax in range(3):
        coeffs[ax] = np.polyfit(t_params, skeleton[:, ax], OOR_POLY_DEGREE)
    return coeffs, cum[-1]


def sample_polynomial_axis(coeffs, total_arc_mm, n_steps):
    """Evaluate the polynomial at evenly-spaced t (NOT arc-uniform, but close
    for gentle curves at low degree). Returns (n_steps, 3) RAS points."""
    t = np.linspace(0.0, 1.0, n_steps)
    pts = np.empty((n_steps, 3), dtype=float)
    for ax in range(3):
        pts[:, ax] = np.polyval(coeffs[ax], t)
    return pts


def sample_curved_profile(volume_kji, ras_to_ijk_mat, coeffs, total_arc_mm, *,
                          step_mm=PROFILE_STEP_MM,
                          disk_radius_mm=PICKER_DISK_RADIUS_MM,
                          n_radii=PICKER_DISK_N_RADII,
                          n_angles=PICKER_DISK_N_ANGLES,
                          reducer=PROFILE_REDUCER):
    """Sample a 1D profile along a polynomial axis, mimicking the API of
    sample_axis_profile but for a curve instead of a straight line.

    Returns (arc_mm, profile_values).
    """
    n_steps = int(total_arc_mm / step_mm) + 1
    arc_mm = np.arange(n_steps, dtype=float) * step_mm
    pts = sample_polynomial_axis(coeffs, total_arc_mm, n_steps)
    # Tangent direction at each step via finite differences.
    tangents = np.zeros_like(pts)
    tangents[1:-1] = pts[2:] - pts[:-2]
    tangents[0] = pts[1] - pts[0]
    tangents[-1] = pts[-1] - pts[-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    tangents = tangents / norms
    # Per-step transverse disk; reduce.
    profile = np.full(n_steps, np.nan, dtype=float)
    for idx in range(n_steps):
        u, v = _orthonormal_basis(tangents[idx])
        samples_ras = [pts[idx]]
        if disk_radius_mm > 0 and n_radii > 0 and n_angles > 0:
            for r_idx in range(1, n_radii + 1):
                r = disk_radius_mm * r_idx / n_radii
                for a_idx in range(n_angles):
                    ang = 2.0 * np.pi * a_idx / n_angles
                    samples_ras.append(pts[idx] + r * (np.cos(ang) * u + np.sin(ang) * v))
        sample_arr = np.array(samples_ras, dtype=float)
        sample_kji = _ras_to_kji_batch(ras_to_ijk_mat, sample_arr)
        vals = _trilinear_batch(volume_kji, sample_kji)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        if reducer == "max":
            profile[idx] = float(vals.max())
        elif reducer == "min":
            profile[idx] = float(vals.min())
        else:
            profile[idx] = float(vals.mean())
    return arc_mm, profile


# ---------------------------------------------------------------------
# Picker
# ---------------------------------------------------------------------

def score_model(profile_arc_mm, profile_values, model):
    expected_tip = profile_arc_mm[-1] - PAD_TIP_MM
    arc_lo = expected_tip - TIP_ARC_SEARCH_HALF_WIDTH_MM
    arc_hi = expected_tip + TIP_ARC_SEARCH_HALF_WIDTH_MM
    candidate_tip_arcs = np.arange(arc_lo, arc_hi, TEMPLATE_OFFSET_STEP_MM)
    best_score = -np.inf
    for tip_arc in candidate_tip_arcs:
        tpl, coverage = build_template(model, profile_arc_mm, tip_arc)
        if tpl.max() <= 0.0:
            continue
        ncc = normalized_cross_correlation(profile_values, tpl)
        if not np.isfinite(ncc):
            continue
        s = ncc * coverage
        if s > best_score:
            best_score = s
    return best_score


def pick_straight(volume_kji, ras_to_ijk, start_ras, end_ras, models):
    arc_mm, profile = sample_axis_profile(
        volume_kji=volume_kji,
        ras_to_ijk_mat=ras_to_ijk,
        start_ras=start_ras, end_ras=end_ras,
        step_mm=PROFILE_STEP_MM,
        disk_radius_mm=PICKER_DISK_RADIUS_MM,
        n_radii=PICKER_DISK_N_RADII,
        n_angles=PICKER_DISK_N_ANGLES,
        reducer=PROFILE_REDUCER,
    )
    best_id = ""
    best_score = -np.inf
    for m in models:
        s = score_model(arc_mm, profile, m)
        if s > best_score:
            best_score = s
            best_id = m["id"]
    return best_id, best_score


def pick_oor(volume_kji, ras_to_ijk, start_ras, end_ras, models):
    refit = refine_axis_oor(volume_kji, ras_to_ijk, start_ras, end_ras)
    if refit is None:
        return pick_straight(volume_kji, ras_to_ijk, start_ras, end_ras, models)
    coeffs, total_arc_mm = refit
    if OOR_POLY_DEGREE == 1:
        # Refined-line path: polynomial is a*t + b. Endpoints at t=0 and t=1
        # bracket the metal-bearing region; extend by PAD_ENTRY / PAD_TIP so
        # profile_arc extends past both ends (matches straight-pick convention
        # the picker's expected_tip relies on).
        end_pt = sample_polynomial_axis(coeffs, total_arc_mm, 2)
        p0, p1 = end_pt[0], end_pt[1]
        direction = p1 - p0
        L = float(np.linalg.norm(direction))
        if L < 1e-3:
            return pick_straight(volume_kji, ras_to_ijk, start_ras, end_ras, models)
        unit = direction / L
        new_start = p0 - PAD_ENTRY_MM * unit
        new_end = p1 + PAD_TIP_MM * unit
        return pick_straight(volume_kji, ras_to_ijk, new_start, new_end, models)
    # Curved (degree >= 2) — sample via custom curved profile sampler.
    arc_mm, profile = sample_curved_profile(
        volume_kji, ras_to_ijk, coeffs, total_arc_mm,
    )
    best_id = ""
    best_score = -np.inf
    for m in models:
        s = score_model(arc_mm, profile, m)
        if s > best_score:
            best_score = s
            best_id = m["id"]
    return best_id, best_score


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------

NOISE_LEVELS = [
    ("clean",   0.0, 0.0),
    ("mild",    1.0, 1.0),
    ("typical", 2.0, 2.0),
    ("hard",    3.0, 3.0),
    ("worst",   3.0, 5.0),
]
N_SEEDS = 8


def main() -> int:
    if not GT_FILE.exists() or not CT_FILE.exists():
        print(f"missing fixture: {GT_FILE} / {CT_FILE}", file=sys.stderr)
        return 1

    print("Loading T22 GT + CT…")
    gts = load_gt_trajectories(GT_FILE)
    volume_kji, ras_to_ijk, _ = load_ct_volume(CT_FILE)
    library = load_electrode_library()
    models = [m for m in library["models"] if m["id"].startswith("DIXI-")]
    print(f"  {len(gts)} GT trajectories, {len(models)} DIXI candidates")

    rng_master = np.random.default_rng(0xc0ffee)

    print("\n=== straight (wider-disk) vs OOR refit (curved profile) ===")
    print(f"{'level':<10} {'lat':>5} {'tilt':>5}  "
          f"{'straight':>9}  {'OOR':>9}  {'Δ':>6}  {'time/pick(ms)':>14}")
    for label, lateral, tilt in NOISE_LEVELS:
        if lateral == 0.0 and tilt == 0.0:
            n_seeds_here = 1
        else:
            n_seeds_here = N_SEEDS
        seed_rng = np.random.default_rng(rng_master.integers(0, 2**32))
        n_total = 0
        n_match_straight = 0
        n_match_oor = 0
        t_oor = 0.0
        for _ in range(n_seeds_here):
            for gt in gts:
                s_ras, e_ras = axis_from_gt(gt)
                sp, ep = perturb_axis(s_ras, e_ras, lateral, tilt, seed_rng)
                pid_s, _ = pick_straight(volume_kji, ras_to_ijk, sp, ep, models)
                t0 = time.time()
                pid_o, _ = pick_oor(volume_kji, ras_to_ijk, sp, ep, models)
                t_oor += time.time() - t0
                n_total += 1
                if pid_s == gt.model_id:
                    n_match_straight += 1
                if pid_o == gt.model_id:
                    n_match_oor += 1
        rate_s = n_match_straight / max(1, n_total)
        rate_o = n_match_oor / max(1, n_total)
        delta = rate_o - rate_s
        ms_per_pick = 1000.0 * t_oor / max(1, n_total)
        print(f"{label:<10} {lateral:>4.1f}m {tilt:>4.1f}°  "
              f"{rate_s:>9.2f}  {rate_o:>9.2f}  {delta:>+6.2f}  {ms_per_pick:>14.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
