"""Auto Fit + per-trajectory PaCER picker DEBUG dump.

For every detected trajectory on T22:
  - bolt-detection result (entry_arc) and tip-detection result
  - profile stats
  - top-5 candidate model scores with NCC × coverage breakdown

Lets us see exactly why the picker chooses what it does on each shank
(in particular the misses on T22: X02 → 18CM, X08 → 8AM, X09 → 12AM).

Run:
    cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper && \\
      /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
      tests/deep_core/probe_pacer_picker_t22_debug.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "CommonLib"))
sys.path.insert(0, str(ROOT / "PostopCTLocalization"))
sys.path.insert(0, str(ROOT / "tools"))

from rosa_detect.service import run_contact_pitch_v1
from eval_seeg_localization import build_detection_context, iter_subject_rows
from rosa_core.contact_peak_fit import sample_axis_profile
from rosa_core.electrode_classifier import (
    _PACER_PAD_TIP_MM, _PACER_PAD_ENTRY_MM,
    _PACER_PROFILE_STEP_MM, _PACER_PROFILE_REDUCER,
    _PACER_PROFILE_DISK_RADIUS_MM, _PACER_PROFILE_DISK_N_RADII,
    _PACER_PROFILE_DISK_N_ANGLES,
    _build_pacer_template, _normalized_cross_correlation,
    _signal_derived_entry_arc, _signal_derived_tip_arc,
    _PACER_TIP_ARC_SEARCH_HALF_WIDTH_MM, _PACER_TEMPLATE_OFFSET_STEP_MM,
    filter_models_for_strategy,
)
from probe_pacer_picker_t22_autofit import (  # noqa: E402
    DATASET_ROOT, GT_FILE, load_gt_trajectories, match_trajectories,
)
from rosa_core.electrode_models import load_electrode_library
import SimpleITK as sitk


def score_one(arc_mm, profile, model, expected_tip,
              arc_lower=None, arc_upper=None):
    arc_lo = expected_tip - _PACER_TIP_ARC_SEARCH_HALF_WIDTH_MM
    arc_hi = expected_tip + _PACER_TIP_ARC_SEARCH_HALF_WIDTH_MM
    candidates = np.arange(arc_lo, arc_hi, _PACER_TEMPLATE_OFFSET_STEP_MM)
    best_score = -np.inf
    best_ncc = float("nan")
    best_cov = 0.0
    best_tip = float("nan")
    for tip in candidates:
        tpl, cov = _build_pacer_template(
            model, arc_mm, tip,
            arc_lower=arc_lower, arc_upper=arc_upper,
        )
        if tpl.max() <= 0.0:
            continue
        ncc = _normalized_cross_correlation(profile, tpl)
        if not np.isfinite(ncc):
            continue
        s = ncc * cov
        if s > best_score:
            best_score = s
            best_ncc = ncc
            best_cov = cov
            best_tip = float(tip)
    return best_score, best_ncc, best_cov, best_tip


def main() -> int:
    print("Loading T22 GT…")
    gt = load_gt_trajectories(GT_FILE)
    print("Running Auto Fit on T22…")
    rows = iter_subject_rows(DATASET_ROOT, {"T22"})
    ctx, _ = build_detection_context(
        rows[0]["ct_path"], run_id="contact_pitch_T22_debug", config={}, extras={},
    )
    ctx["contact_pitch_v1_pitch_strategy"] = "dixi"
    result = run_contact_pitch_v1(ctx)
    trajs = list(result.get("trajectories") or [])
    matched = match_trajectories(gt, trajs)
    print(f"Matched {len(matched)}/{len(gt)}")

    library = load_electrode_library()
    models = filter_models_for_strategy(
        list(library.get("models") or []), "dixi",
    )

    # Re-load CT once to sample the profile per trajectory.
    img = sitk.ReadImage(rows[0]["ct_path"])
    ct_arr = sitk.GetArrayFromImage(img).astype("float32")
    spacing = np.array(img.GetSpacing(), dtype=float)
    direction = np.array(img.GetDirection(), dtype=float).reshape(3, 3)
    origin = np.array(img.GetOrigin(), dtype=float)
    M_lps = direction @ np.diag(spacing)
    M_inv = np.linalg.inv(M_lps)
    flip = np.diag([-1.0, -1.0, 1.0])
    ras_to_ijk = np.eye(4)
    ras_to_ijk[:3, :3] = M_inv @ flip
    ras_to_ijk[:3, 3] = -M_inv @ origin

    for gi, ti in sorted(matched, key=lambda p: gt[p[0]]["name"]):
        g = gt[gi]
        t = trajs[ti]
        truth = g["model_id"]
        picked = str(t.get("suggested_model_id") or "")
        marker = "OK " if picked == truth else "MISS"
        s_ras = np.asarray(t.get("skull_entry_ras") or t["start_ras"], dtype=float)
        e_ras = np.asarray(t["end_ras"], dtype=float)
        L = float(np.linalg.norm(e_ras - s_ras))
        print(f"\n=== {g['name']} truth={truth} picked={picked} {marker} "
              f"(L={L:.2f} mm, n_in={t.get('n_inliers')}, span={t.get('contact_span_mm'):.1f}) ===")

        unit = (e_ras - s_ras) / L
        sample_start = s_ras - _PACER_PAD_ENTRY_MM * unit
        sample_end = e_ras + _PACER_PAD_TIP_MM * unit
        arc_mm, profile = sample_axis_profile(
            volume_kji=ct_arr, ras_to_ijk_mat=ras_to_ijk,
            start_ras=sample_start, end_ras=sample_end,
            step_mm=_PACER_PROFILE_STEP_MM,
            disk_radius_mm=_PACER_PROFILE_DISK_RADIUS_MM,
            n_radii=_PACER_PROFILE_DISK_N_RADII,
            n_angles=_PACER_PROFILE_DISK_N_ANGLES,
            reducer=_PACER_PROFILE_REDUCER,
        )
        finite = np.isfinite(profile)
        print(f"  profile len={len(arc_mm)}  arc_max={arc_mm[-1]:.2f}  "
              f"hu_min={float(profile[finite].min()):.0f}  "
              f"hu_max={float(profile[finite].max()):.0f}  "
              f"hu_p95={float(np.percentile(profile[finite], 95)):.0f}")
        entry_arc = _signal_derived_entry_arc(arc_mm, profile)
        if entry_arc is not None:
            mask = arc_mm < (entry_arc - 1.0)
            if np.any(mask):
                profile_clean = np.array(profile, copy=True)
                profile_clean[mask] = np.nan
                profile = profile_clean
        tip_arc = _signal_derived_tip_arc(arc_mm, profile)
        expected_tip = tip_arc if tip_arc is not None else (arc_mm[-1] - _PACER_PAD_TIP_MM)
        print(f"  entry_arc={entry_arc}  tip_arc={tip_arc}  expected_tip={expected_tip:.2f}")
        # Score every model — pass the same arc_lower/arc_upper as
        # the production dispatcher so debug scores match production.
        arc_lower = float(entry_arc) if entry_arc is not None else float(arc_mm[0])
        arc_upper = float(arc_mm[-1])
        scored = []
        for m in models:
            s, ncc, cov, tip = score_one(
                arc_mm, profile, m, expected_tip,
                arc_lower=arc_lower, arc_upper=arc_upper,
            )
            if np.isfinite(s):
                scored.append((s, m["id"], ncc, cov, tip))
        scored.sort(key=lambda r: r[0], reverse=True)
        print("  top-5 candidates:")
        for s, mid, ncc, cov, tip in scored[:5]:
            tag = "  <-- TRUTH" if mid == truth else ("  <-- PICK" if mid == picked else "")
            print(f"    {mid:<13} score={s:>6.3f}  ncc={ncc:>6.3f}  "
                  f"cov={cov:.2f}  tip={tip:>6.2f}{tag}")
        # On misses, dump the profile as ASCII histogram so we can see
        # whether actual contact peaks line up with the template.
        if picked != truth:
            arr = np.asarray(profile, dtype=float)
            arc = np.asarray(arc_mm, dtype=float)
            mn = float(np.nanmin(arr)); mx = float(np.nanmax(arr))
            rng = max(1.0, mx - mn)
            print(f"  profile (HU range {mn:.0f}–{mx:.0f}):")
            # 0.5 mm step keeps 3.5 mm pitch peaks resolvable.
            step = max(1, int(round(0.5 / _PACER_PROFILE_STEP_MM)))
            for i in range(0, len(arc), step):
                v = arr[i]
                if not np.isfinite(v):
                    bar = "  (muted)"
                else:
                    n_bars = int(round(40 * (v - mn) / rng))
                    bar = "#" * max(0, n_bars)
                print(f"    arc {arc[i]:>6.2f}  {('--' if not np.isfinite(v) else f'{v:>6.0f}')}  {bar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
