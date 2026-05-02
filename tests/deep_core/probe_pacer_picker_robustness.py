"""Robustness sweep: how much axis error breaks the PaCER-style picker?

Phase 1 (`probe_pacer_picker_t22.py`) used the GT-derived axis and
matched 9/9. Real Auto Fit / Manual Fit axes carry lateral offset and
tilt error; this probe perturbs the GT axis at controlled noise levels
and re-runs the picker. If match rate stays at 9/9 within the noise
budget Auto Fit actually produces (~0.5-2 mm lateral, ~1-3° tilt), we
don't need PaCER's polynomial OOR axis refit. Otherwise we do.

Two configurations tried:
  - "vanilla" — same settings as phase 1 (disk r=1 mm, 2 rings × 8 angles)
  - "wider"   — disk r=2.5 mm, 4 rings × 8 angles

Wider disk is the cheap robustness lever. If it recovers tolerance to
1-2 mm without OOR, OOR is unnecessary. If wider-disk picks up
cross-shank contamination, OOR is the next move.

Per (lateral_mm, tilt_deg, config): N_SEEDS random perturbations × 9
trajectories = 9*N picks. Match rate aggregated.

Run:
    cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper && \\
      /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
      tests/deep_core/probe_pacer_picker_robustness.py
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

# Reuse phase-1 helpers verbatim.
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
    GtTrajectory,
    axis_from_gt,
    build_template,
    load_ct_volume,
    load_gt_trajectories,
    normalized_cross_correlation,
)


LATERAL_LEVELS_MM = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
TILT_LEVELS_DEG = [0.0, 1.0, 2.0, 3.0, 5.0]
N_SEEDS = 8

DISK_CONFIGS = {
    "vanilla": dict(disk_radius_mm=1.0, n_radii=2, n_angles=8),
    "wider":   dict(disk_radius_mm=2.5, n_radii=4, n_angles=8),
}


# ---------------------------------------------------------------------
# Axis perturbation
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


def perturb_axis(start_ras, end_ras, lateral_mm, tilt_deg, rng):
    """Apply lateral parallel-translation + tilt to the (start, end) axis.

    Lateral: shifts both endpoints by ``lateral_mm`` in a random
    direction in the transverse plane.
    Tilt: rotates both endpoints symmetrically around the midpoint by
    ``tilt_deg`` around a random transverse axis. End-to-end angular
    error of the resulting line equals tilt_deg.
    """
    s = np.asarray(start_ras, dtype=float)
    e = np.asarray(end_ras, dtype=float)
    direction = e - s
    L = float(np.linalg.norm(direction))
    if L < 1e-3:
        return s.copy(), e.copy()
    direction_unit = direction / L
    u_basis, v_basis = _orthonormal_basis(direction_unit)

    # Lateral parallel translation.
    if lateral_mm > 0.0:
        ang = rng.uniform(0.0, 2.0 * np.pi)
        offset = lateral_mm * (np.cos(ang) * u_basis + np.sin(ang) * v_basis)
    else:
        offset = np.zeros(3, dtype=float)

    # Tilt around midpoint.
    midpoint = 0.5 * (s + e)
    if tilt_deg > 0.0:
        ang_tilt = rng.uniform(0.0, 2.0 * np.pi)
        rot_axis = np.cos(ang_tilt) * u_basis + np.sin(ang_tilt) * v_basis
        rot_axis /= np.linalg.norm(rot_axis)
        theta = np.deg2rad(tilt_deg)
        c = np.cos(theta)
        si = np.sin(theta)
        new_dir = (
            direction_unit * c
            + np.cross(rot_axis, direction_unit) * si
            + rot_axis * np.dot(rot_axis, direction_unit) * (1.0 - c)
        )
        new_dir /= np.linalg.norm(new_dir)
        new_s = midpoint - 0.5 * L * new_dir + offset
        new_e = midpoint + 0.5 * L * new_dir + offset
    else:
        new_s = s + offset
        new_e = e + offset
    return new_s, new_e


# ---------------------------------------------------------------------
# Picker (re-implemented here so disk config is configurable)
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


def pick_model(volume_kji, ras_to_ijk, start_ras, end_ras, models, disk_kwargs):
    arc_mm, profile = sample_axis_profile(
        volume_kji=volume_kji,
        ras_to_ijk_mat=ras_to_ijk,
        start_ras=start_ras, end_ras=end_ras,
        step_mm=PROFILE_STEP_MM,
        reducer=PROFILE_REDUCER,
        **disk_kwargs,
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

    for cfg_name, disk_kwargs in DISK_CONFIGS.items():
        print(f"\n=== config: {cfg_name} (disk r={disk_kwargs['disk_radius_mm']} mm, "
              f"{disk_kwargs['n_radii']} rings × {disk_kwargs['n_angles']} angles) ===")
        # Header: tilts as columns
        cols = "  ".join(f"{t:5.1f}°" for t in TILT_LEVELS_DEG)
        header = "lat / tilt"
        print(f"{header:<10} {cols}")
        t_start = time.time()
        for lateral in LATERAL_LEVELS_MM:
            cells = []
            for tilt in TILT_LEVELS_DEG:
                if lateral == 0.0 and tilt == 0.0:
                    n_seeds_here = 1  # deterministic at zero noise
                else:
                    n_seeds_here = N_SEEDS
                seed_rng = np.random.default_rng(rng_master.integers(0, 2**32))
                n_total = 0
                n_match = 0
                for _ in range(n_seeds_here):
                    for gt in gts:
                        s, e = axis_from_gt(gt)
                        sp, ep = perturb_axis(s, e, lateral, tilt, seed_rng)
                        try:
                            picked, _score = pick_model(
                                volume_kji, ras_to_ijk, sp, ep, models, disk_kwargs,
                            )
                        except Exception:
                            picked = ""
                        n_total += 1
                        if picked == gt.model_id:
                            n_match += 1
                rate = n_match / max(1, n_total)
                cells.append(f"{rate:5.2f}")
            print(f"{lateral:>4.1f} mm   " + "  ".join(cells))
        print(f"  ({time.time() - t_start:.1f} s)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
