"""PaCER-style template-correlation electrode-model picker probe on T22.

Tests whether template correlation against each library model cleanly
identifies the right electrode model from the 1-D intensity profile
along the GT trajectory axis.

Phase 1 (this probe): straight axis derived from GT contacts (no
polynomial OOR refit yet). Isolates the model-ID question from the
axis-fit question. If correlation cleanly separates the right model
from neighbors here, the principle works and we can layer the OOR
refit on top in Phase 2.

T22 GT (from `tests/data/T22/T22_aligned_world_coords.txt`):
    X01-X04 = DIXI-15CM (15 contacts, 3-group structural gaps)
    X05     = DIXI-15AM (15 contacts uniform 3.5 mm)
    X06     = DIXI-12AM (12 contacts uniform 3.5 mm)
    X07-X09 = DIXI-10AM (10 contacts uniform 3.5 mm)

Run:
    cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper && \\
      /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
      tests/deep_core/probe_pacer_picker_t22.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "CommonLib"))

from rosa_core.contact_peak_fit import sample_axis_profile  # noqa: E402
from rosa_core.electrode_models import load_electrode_library  # noqa: E402


T22_DIR = ROOT / "tests" / "data" / "T22"
GT_FILE = T22_DIR / "T22_aligned_world_coords.txt"
CT_FILE = T22_DIR / "T22_ct.nii.gz"

# Profile-sampling defaults — native CT, max-reducer over a small
# transverse disk so a slightly-off axis still hits the metal contact.
PROFILE_STEP_MM = 0.25
PROFILE_DISK_RADIUS_MM = 1.0
PROFILE_DISK_N_RADII = 2
PROFILE_DISK_N_ANGLES = 8
PROFILE_REDUCER = "max"

# Template construction.
TEMPLATE_SIGMA_MM = 0.6        # half-contact-width-ish; bumps roughly 1.5 mm FWHM
TEMPLATE_OFFSET_STEP_MM = 0.25  # axial slide resolution
TEMPLATE_OFFSET_PAD_MM = 5.0    # search 5 mm past either profile end

# Axis padding off the most-tip / most-entry GT contact when defining
# the straight sampling axis. Tight pads keep the bolt (which sits
# 10-30 mm beyond the most-superficial active contact) OUT of the
# profile — the bolt is a contiguous bright metal strip that
# correlates equally with the first few contacts of any candidate
# model, dragging the slide to a degenerate tip_arc in the entry pad.
PAD_TIP_MM = 1.5
PAD_ENTRY_MM = 1.5

# Tip-arc slide window: keep the candidate tip near the deep end of
# the profile (the deepest GT contact sits at profile_arc[-1] -
# PAD_TIP_MM). Sliding the tip across the entry side allows degenerate
# matches where only the first 1-2 model contacts overlap a single
# bolt-bright region.
TIP_ARC_SEARCH_HALF_WIDTH_MM = 5.0


# ---------------------------------------------------------------------
# GT loading
# ---------------------------------------------------------------------

@dataclass
class GtTrajectory:
    name: str
    model_id: str
    contacts_ras: np.ndarray  # (N, 3) RAS, ordered by index ascending


def load_gt_trajectories(path: Path) -> list[GtTrajectory]:
    by_traj: dict[str, list[tuple[int, list[float], str]]] = defaultdict(list)
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split(",")
            if len(cols) < 13:
                continue
            traj, _label, idx = cols[0], cols[1], int(cols[2])
            x, y, z = float(cols[6]), float(cols[7]), float(cols[8])  # world RAS
            model = cols[12]
            by_traj[traj].append((idx, [x, y, z], model))
    out: list[GtTrajectory] = []
    for name, rows in sorted(by_traj.items()):
        rows.sort()
        models = {m for _, _, m in rows}
        if len(models) != 1:
            raise ValueError(f"{name}: mixed model_ids {models}")
        contacts = np.array([r[1] for r in rows], dtype=float)
        out.append(GtTrajectory(name=name, model_id=models.pop(), contacts_ras=contacts))
    return out


# ---------------------------------------------------------------------
# CT loading
# ---------------------------------------------------------------------

def load_ct_volume(path: Path):
    """Return (volume_kji, ras_to_ijk_mat 4x4, spacing_ijk_mm)."""
    import SimpleITK as sitk
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # (k, j, i) per SITK convention
    spacing = np.array(img.GetSpacing(), dtype=float)        # (i, j, k)
    direction = np.array(img.GetDirection(), dtype=float).reshape(3, 3)
    origin = np.array(img.GetOrigin(), dtype=float)          # LPS
    # SITK origin/direction live in LPS. Convert RAS → LPS via a sign flip
    # on x and y, then LPS → IJK via inv(direction · diag(spacing)).
    M_lps = direction @ np.diag(spacing)
    M_inv = np.linalg.inv(M_lps)
    flip = np.diag([-1.0, -1.0, 1.0])
    ras_to_ijk = np.eye(4)
    ras_to_ijk[:3, :3] = M_inv @ flip
    ras_to_ijk[:3, 3] = -M_inv @ origin
    return arr, ras_to_ijk, spacing


# ---------------------------------------------------------------------
# Axis derivation
# ---------------------------------------------------------------------

def axis_from_gt(gt: GtTrajectory):
    """Return (start_ras, end_ras) for sampling.

    GT index 1 is the deepest contact (the tip); index N is the most
    superficial (entry-side). Profile arc grows from start (entry side,
    extended) to end (deep tip, slightly extended).
    """
    deep_tip = gt.contacts_ras[0]
    entry_side = gt.contacts_ras[-1]
    direction = deep_tip - entry_side
    L = float(np.linalg.norm(direction))
    if L < 1e-3:
        raise ValueError(f"{gt.name}: degenerate axis")
    direction_unit = direction / L
    start = entry_side - PAD_ENTRY_MM * direction_unit
    end = deep_tip + PAD_TIP_MM * direction_unit
    return start, end


# ---------------------------------------------------------------------
# Template + scoring
# ---------------------------------------------------------------------

def build_template(model: dict, profile_arc_mm: np.ndarray,
                   tip_arc_mm: float):
    """Place a Gaussian bump at each contact position.

    ``tip_arc_mm`` is the candidate arc-length where the model's tip
    (offset_from_tip = 0) sits along the profile. Each contact at
    ``offset_from_tip`` lives at arc-length ``tip_arc_mm − offset`` —
    closer to entry as the offset grows (the tip is at the deep end of
    the profile, profile starts at the entry side).

    Returns ``(template, coverage)`` where ``coverage`` is the fraction
    of model contacts whose expected position lies inside the profile.
    Longer models whose entry-side contacts fall off the profile drop
    coverage proportionally — used by the caller to penalize template
    clipping (otherwise a 15AM and a 10AM that both have 10 visible
    contacts would tie on raw NCC).
    """
    offsets = np.asarray(
        model.get("contact_center_offsets_from_tip_mm", []), dtype=float
    )
    if offsets.size == 0:
        return np.zeros_like(profile_arc_mm), 0.0
    positions = tip_arc_mm - offsets
    in_range = (positions >= profile_arc_mm[0] - 3.0 * TEMPLATE_SIGMA_MM) & (
        positions <= profile_arc_mm[-1] + 3.0 * TEMPLATE_SIGMA_MM
    )
    coverage = float(np.count_nonzero(in_range)) / float(positions.size)
    if not np.any(in_range):
        return np.zeros_like(profile_arc_mm), coverage
    template = np.zeros_like(profile_arc_mm)
    sigma_sq = TEMPLATE_SIGMA_MM ** 2
    for pos in positions[in_range]:
        template += np.exp(-0.5 * (profile_arc_mm - pos) ** 2 / sigma_sq)
    return template, coverage


def normalized_cross_correlation(profile: np.ndarray,
                                 template: np.ndarray) -> float:
    a = np.asarray(profile, dtype=float)
    b = np.asarray(template, dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 5:
        return float("nan")
    a = a[finite] - np.mean(a[finite])
    b = b[finite] - np.mean(b[finite])
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def score_model(profile_arc_mm: np.ndarray, profile_values: np.ndarray,
                model: dict):
    """Slide tip-arc-position over the profile, return best (score, tip_arc).

    Search window is centred on the deep end of the profile (where the
    GT-derived axis places the tip), with TIP_ARC_SEARCH_HALF_WIDTH_MM
    of slack on either side to absorb tip placement error.
    """
    expected_tip = profile_arc_mm[-1] - PAD_TIP_MM
    arc_lo = expected_tip - TIP_ARC_SEARCH_HALF_WIDTH_MM
    arc_hi = expected_tip + TIP_ARC_SEARCH_HALF_WIDTH_MM
    candidate_tip_arcs = np.arange(arc_lo, arc_hi, TEMPLATE_OFFSET_STEP_MM)
    best_score = -np.inf
    best_tip_arc = float("nan")
    best_coverage = 0.0
    for tip_arc in candidate_tip_arcs:
        tpl, coverage = build_template(model, profile_arc_mm, tip_arc)
        if tpl.max() <= 0.0:
            continue
        ncc = normalized_cross_correlation(profile_values, tpl)
        if not np.isfinite(ncc):
            continue
        # Coverage-weighted score: a 15AM template clipped to its first
        # 10 contacts (because the profile only spans 10 contacts of
        # truth) loses 1/3 of its contacts → coverage 0.67 → its score
        # is held below an actually-fitting 10AM. Without this term,
        # any longer model whose entry-side overflow gets clipped ties
        # exactly with the shorter model that fits.
        s = ncc * coverage
        if s > best_score:
            best_score = s
            best_tip_arc = float(tip_arc)
            best_coverage = coverage
    return best_score, best_tip_arc, best_coverage


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------

def main() -> int:
    if not GT_FILE.exists():
        print(f"missing {GT_FILE}", file=sys.stderr)
        return 1
    if not CT_FILE.exists():
        print(f"missing {CT_FILE}", file=sys.stderr)
        return 1

    print("Loading T22 GT + CT…")
    gts = load_gt_trajectories(GT_FILE)
    print(f"  {len(gts)} GT trajectories: " + ", ".join(
        f"{g.name}={g.model_id}({len(g.contacts_ras)})" for g in gts
    ))

    volume_kji, ras_to_ijk, spacing = load_ct_volume(CT_FILE)
    print(f"  CT shape (k,j,i)={volume_kji.shape} spacing(i,j,k)=({spacing[0]:.3f},{spacing[1]:.3f},{spacing[2]:.3f})")
    print(f"  HU range [{int(volume_kji.min())}, {int(volume_kji.max())}]")

    library = load_electrode_library()
    candidate_models = [
        m for m in library.get("models", [])
        if str(m.get("id", "")).startswith("DIXI-")
    ]
    print(f"  {len(candidate_models)} DIXI candidate models in library")

    # Score each GT trajectory against the full DIXI subset.
    print("\n--- per-trajectory results ---")
    print(f"{'traj':<5} {'GT':<11} {'best':<11} {'score':>6} {'tip_arc':>8}  "
          f"{'#2':<11} {'s2':>6}  {'margin':>7}  match")
    n_match = 0
    margins = []
    for gt in gts:
        start, end = axis_from_gt(gt)
        L = float(np.linalg.norm(end - start))
        arc_mm, profile = sample_axis_profile(
            volume_kji=volume_kji,
            ras_to_ijk_mat=ras_to_ijk,
            start_ras=start, end_ras=end,
            step_mm=PROFILE_STEP_MM,
            disk_radius_mm=PROFILE_DISK_RADIUS_MM,
            n_radii=PROFILE_DISK_N_RADII,
            n_angles=PROFILE_DISK_N_ANGLES,
            reducer=PROFILE_REDUCER,
        )
        scored = []
        for m in candidate_models:
            s, tip_arc, cov = score_model(arc_mm, profile, m)
            scored.append((s, m["id"], tip_arc, cov))
        scored.sort(key=lambda r: (r[0] if np.isfinite(r[0]) else -np.inf), reverse=True)
        best_s, best_id, best_tip, _ = scored[0]
        runner_s, runner_id, _, _ = (
            scored[1] if len(scored) > 1 else (float("nan"), "-", float("nan"), 0.0)
        )
        margin = best_s - runner_s if np.isfinite(best_s) and np.isfinite(runner_s) else float("nan")
        margins.append(margin)
        is_match = best_id == gt.model_id
        if is_match:
            n_match += 1
        marker = "OK " if is_match else "MISS"
        print(f"{gt.name:<5} {gt.model_id:<11} {best_id:<11} {best_s:>6.3f} "
              f"{best_tip:>8.2f}  {runner_id:<11} {runner_s:>6.3f}  {margin:>7.3f}  {marker}")

    print(f"\n{n_match}/{len(gts)} GT models matched")
    finite_margins = [m for m in margins if np.isfinite(m)]
    if finite_margins:
        print(f"score margin (best - runner-up): "
              f"min={min(finite_margins):.3f}  median={np.median(finite_margins):.3f}  "
              f"max={max(finite_margins):.3f}")

    # Verbose dump for one trajectory: top-5 candidates each.
    print("\n--- top-5 candidates per trajectory (sanity check) ---")
    for gt in gts:
        start, end = axis_from_gt(gt)
        arc_mm, profile = sample_axis_profile(
            volume_kji=volume_kji,
            ras_to_ijk_mat=ras_to_ijk,
            start_ras=start, end_ras=end,
            step_mm=PROFILE_STEP_MM,
            disk_radius_mm=PROFILE_DISK_RADIUS_MM,
            n_radii=PROFILE_DISK_N_RADII,
            n_angles=PROFILE_DISK_N_ANGLES,
            reducer=PROFILE_REDUCER,
        )
        scored = []
        for m in candidate_models:
            s, tip_arc, cov = score_model(arc_mm, profile, m)
            scored.append((s, m["id"], tip_arc, cov))
        scored.sort(key=lambda r: (r[0] if np.isfinite(r[0]) else -np.inf), reverse=True)
        print(f"  {gt.name} (GT={gt.model_id}):")
        for s, mid, tip, cov in scored[:5]:
            tag = "  <-- GT" if mid == gt.model_id else ""
            print(f"    {mid:<13} score={s:>6.3f} tip_arc={tip:>6.2f} cov={cov:.2f}{tag}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
