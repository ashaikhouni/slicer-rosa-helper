"""V2 contact placement: matched-filter scoring on a walker-disk-stat signal.

Composes already-shipped primitives:
  - ``estimate_bolt_end_from_metal_mass`` → bolt-end arc + polynomial centerline
  - ``sample_disk_along_polyline`` → walker max-disk-stat signal along centerline
  - ``matched_filter_pick`` → Pearson cross-correlation library scoring

Replaces the older RANSAC + LoG-blob pipeline in ``contact_placement.py``
with a single-knob (σ_contact ≈ 1 mm) matcher that doesn't depend on
detected blob clusters.

Bolt-less fallback: when ``estimate_bolt_end_from_metal_mass`` returns
``None`` or collapses the contact zone (bolt_end ≥ centerline length −
``DEGENERATE_CONTACT_ZONE_MM``), treat the seed axis itself as the
contact zone (``bolt_end_arc_mm = 0``, no tip extension, no synth bolt).
This recovers shanks whose bolt is cropped at the CT FOV edge (per
``project_autofit_misses_2026-05-06.md``: AMC137 LI/LPT/RI/RU).

The matched filter's correlation score acts as the **trajectory
validator**: high ``corr_score`` (≥ ``min_corr``) = real shank, low
score = drop. No separate "must have bolt" / "must pass Frangi
median" gates needed (per user 2026-05-06).

This module is **additive** — does not modify ``contact_placement.py``.
Callers can adopt incrementally.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .matched_filter import (
    SIGMA_CONTACT_MM_DEFAULT,
    MatchedFilterResult,
    matched_filter_pick,
)


# ---------------------------------------------------------------------
# Defaults — physical, not tuned
# ---------------------------------------------------------------------

WALK_STEP_MM = 0.25                      # walker arc resolution
WALK_DISK_RADIUS_MM = 1.0                # contact half-diameter
WALK_FIRST_CONTACT_MIN_MM = 1.0          # bolt-to-first-contact gap
WALK_TIP_PAD_MM = 3.0                    # tip slack for axis under-reach (standard mode)
WALK_HU_MIN = 1000.0                     # disk-sample min HU (above this = metal-ish)

# Centerline snap to local metal centroid (Stage 3 in the staged-walker
# pipeline). Recenters the polynomial centerline on local high-HU
# centroids; fixes ~5-10% of standard-mode placement errors when the
# auto-fit / polynomial axis is laterally offset by 1-2 mm from the
# actual electrode (project_staged_walker_2026-05-05.md).
SNAP_RADIUS_MM = 2.0
SNAP_HU_THRESHOLD = 1000.0
SNAP_STEP_MM = 0.5
SNAP_SMOOTH_WINDOW = 5

DEGENERATE_CONTACT_ZONE_MM = 5.0         # if cl_max - bolt_end < this, treat as bolt-less

# Walker disk-stat sampling. n_radii × n_angles + 1 = sample count per
# disk. The probe-tested config (3 × 12 + 1 = 37 samples) gives more
# stable correlation than `sample_disk_along_polyline`'s defaults
# (2 × 8 + 1 = 17 samples) — the matched filter Pearson correlation
# is sensitive to sample noise at low signal arcs.
WALK_N_RADII = 3
WALK_N_ANGLES = 12

# Validator threshold. Below this, the matcher's pick is too weak to
# trust as a real shank. Calibrated 2026-05-06 on the 6-subject dataset:
# real shanks score 0.5-0.95, AMC91/SuraceContacts (atypical) scores
# 0.66, AMC137 cropped-bolt-less shanks score 0.37-0.69. Setting the
# threshold to 0.35 keeps real shanks in and rejects no-signal cases.
MIN_CORR_FOR_REAL_SHANK = 0.35

# Per-slot HU floor for the unseeded validator. Real-shank slot HU is
# 1500-3000+ across all contacts; cross-shank/bone FP chains average
# 900-1500. 1500 cleanly drops 7 of 12 GENUINE_FP orphans on the dataset
# at the cost of one TP (AMC91 / hetero shank with mean=1285). Set to
# None to disable.
MIN_SLOT_HU_MEAN = 1500.0

# Per-slot connected-component volume cap. A real PMT/DIXI contact is a
# 1.3 mm × 2 mm cylinder of platinum-iridium ≈ 2.6 mm³ of saturating-HU
# metal. Adjacent contacts and the wire connecting them inflate the
# saturating-HU CC up to ~140 mm³ within a 5 mm half-extent ROI. Bone-
# spike chains, surgical clips, and multi-shank wire bundles do not
# obey this physical bound — at least one slot lands in unbounded bone.
#
# We measure the 90th-percentile of per-slot CC volumes and cap it at
# 150 mm³ — calibrated 2026-05-08 on the 6-subject dataset:
#
#   MATCHED (n=64): vol_p90 max=142.3 mm³  (real shanks)
#   ORPHAN  (n=9):  vol_p90 min=166.1 mm³  (bone chains)
#
# This is direction A from the 2026-05-07 v2 handoff. The walker self-
# aligns to peaks so per-slot HU thresholds share its blind spot, but
# CC volume is a topological measurement the walker cannot fake.
MAX_SLOT_CC_VOLUME_P90_MM3 = 150.0
CC_HU_THRESHOLD = 1500.0
CC_ROI_HALF_MM = 5.0


# ---------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------


@dataclass
class PlacementV2Result:
    """Result of ``place_contacts_for_seed_v2``.

    ``success=False`` indicates the matcher couldn't find a model OR the
    correlation was below ``min_corr_for_real_shank``. ``corr_score``
    is the Pearson correlation against the winning library template.
    """

    success: bool
    model_id: str | None
    placed_ras: list[list[float]]
    centerline_ras: list[list[float]] | None
    corr_score: float
    bolt_end_arc_mm: float
    bolt_source: str   # "metal" | "synthesized" | "bolt_less" | "none"
    n_placed: int
    rejected_reason: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Polyline helpers (also used by the walker)
# ---------------------------------------------------------------------


def _polyline_segments(polyline: np.ndarray):
    """Returns (starts, dirs, lens, cum_start) for a (K,3) polyline."""
    P = np.asarray(polyline, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] < 2:
        raise ValueError("polyline must be (K,3) with K>=2")
    diffs = np.diff(P, axis=0)
    lens = np.linalg.norm(diffs, axis=1)
    keep = lens > 1e-9
    if not keep.any():
        raise ValueError("polyline has zero arc length")
    starts = P[:-1][keep]
    diffs = diffs[keep]
    lens = lens[keep]
    dirs = diffs / lens[:, None]
    cum_start = np.concatenate([[0.0], np.cumsum(lens[:-1])])
    return starts, dirs, lens, cum_start


def _polyline_pos_at_arc(polyline: np.ndarray, arc_mm: float) -> np.ndarray:
    """Position on the polyline at the given arc length."""
    starts, dirs, lens, cum_start = _polyline_segments(polyline)
    total = float(cum_start[-1] + lens[-1])
    if arc_mm <= 0.0:
        return starts[0].copy()
    if arc_mm >= total:
        return starts[-1] + lens[-1] * dirs[-1]
    i = int(np.searchsorted(cum_start + lens, arc_mm, side="right"))
    i = min(i, len(starts) - 1)
    t = arc_mm - cum_start[i]
    return starts[i] + t * dirs[i]


def _project_to_polyline_arc(polyline: np.ndarray, point_ras: np.ndarray) -> float:
    """Arc-length of the closest polyline point to ``point_ras``."""
    starts, dirs, lens, cum_start = _polyline_segments(polyline)
    pt = np.asarray(point_ras, dtype=float)
    best_d = np.inf
    best_arc = 0.0
    for i in range(len(starts)):
        a = starts[i]; L = lens[i]; u = dirs[i]
        t = float(np.clip((pt - a) @ u, 0.0, L))
        proj = a + t * u
        d = float(np.linalg.norm(pt - proj))
        if d < best_d:
            best_d = d
            best_arc = float(cum_start[i] + t)
    return best_arc


def _straight_centerline(start: np.ndarray, end: np.ndarray, n_points: int = 64) -> np.ndarray:
    """Discretize a straight start→end segment into ``n_points``."""
    s = np.asarray(start, dtype=float)
    e = np.asarray(end, dtype=float)
    ts = np.linspace(0.0, 1.0, n_points)
    return np.array([s + t * (e - s) for t in ts])


def _extend_centerline_tail(centerline: np.ndarray, extra_mm: float) -> np.ndarray:
    """Extend the centerline past its deep endpoint by ``extra_mm`` along
    the local tail tangent. Lets the walker sample signal slightly past
    the auto-fit axis tip — allows the matched filter to evaluate model
    tip positions that fall just past the polynomial endpoint.
    """
    if extra_mm <= 1e-6:
        return centerline
    cl = np.asarray(centerline, dtype=float)
    tail_dir = cl[-1] - cl[-2]
    tail_len = float(np.linalg.norm(tail_dir))
    if tail_len < 1e-9:
        return cl
    tail_unit = tail_dir / tail_len
    new_tip = cl[-1] + tail_unit * float(extra_mm)
    return np.vstack([cl, new_tip[None, :]])


def _polyline_pos_tan(polyline: np.ndarray, arc_mm: float):
    """Position + unit tangent on the polyline at the given arc."""
    starts, dirs, lens, cum_start = _polyline_segments(polyline)
    total = float(cum_start[-1] + lens[-1])
    if arc_mm <= 0.0:
        return starts[0].copy(), dirs[0]
    if arc_mm >= total:
        return starts[-1] + lens[-1] * dirs[-1], dirs[-1]
    i = int(np.searchsorted(cum_start + lens, arc_mm, side="right"))
    i = min(i, len(starts) - 1)
    t = arc_mm - cum_start[i]
    return starts[i] + t * dirs[i], dirs[i]


def _ortho_uv(tangent: np.ndarray):
    any_v = np.array([1.0, 0.0, 0.0]) if abs(tangent[0]) <= 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(tangent, any_v); u /= np.linalg.norm(u)
    v = np.cross(tangent, u);     v /= np.linalg.norm(v)
    return u, v


def _snap_centerline_to_centroid(
    centerline: np.ndarray, ct_arr_kji, r2i,
    *, snap_radius_mm: float = SNAP_RADIUS_MM,
    step_mm: float = SNAP_STEP_MM,
    hu_threshold: float = SNAP_HU_THRESHOLD,
    n_radii: int = 4, n_angles: int = 16,
    smooth_window: int = SNAP_SMOOTH_WINDOW,
) -> np.ndarray:
    """Recenter ``centerline`` arc-by-arc on the local high-HU centroid.

    Sample a perpendicular disk around each arc-position; weight each
    in-disk voxel by ``max(0, HU - hu_threshold)``; shift the arc
    position to the weighted centroid. Smooth the resulting polyline
    with a uniform filter (window ``smooth_window``).

    Lifted from the staged-walker notebook with no logic changes.
    """
    from .volume_sampling import sample_trilinear_batch
    from scipy.ndimage import uniform_filter1d

    starts, dirs, lens, cum_start = _polyline_segments(centerline)
    total_arc = float(cum_start[-1] + lens[-1])
    arcs = np.arange(0.0, total_arc + 0.5 * step_mm, step_mm)
    snapped = np.zeros((len(arcs), 3), dtype=float)
    # Pre-compute disk offset templates in (u, v) basis once.
    n_per_disk = n_radii * n_angles
    off_u = np.zeros(n_per_disk, dtype=float)
    off_v = np.zeros(n_per_disk, dtype=float)
    idx = 0
    for r_i in range(1, n_radii + 1):
        rr = snap_radius_mm * r_i / n_radii
        for a_i in range(n_angles):
            ang = 2.0 * np.pi * a_i / n_angles
            off_u[idx] = rr * np.cos(ang)
            off_v[idx] = rr * np.sin(ang)
            idx += 1
    for ai, t in enumerate(arcs):
        center, tangent = _polyline_pos_tan(centerline, float(t))
        u, v = _ortho_uv(tangent)
        # (n_per_disk, 3) batch of perpendicular-disk sample points.
        pts = (center[None, :]
               + off_u[:, None] * u[None, :]
               + off_v[:, None] * v[None, :])
        hus = sample_trilinear_batch(ct_arr_kji, r2i, pts)
        valid = np.isfinite(hus) & (hus > hu_threshold)
        if np.any(valid):
            w = hus[valid] - hu_threshold
            mu = float((w * off_u[valid]).sum() / w.sum())
            mv = float((w * off_v[valid]).sum() / w.sum())
            snapped[ai] = center + mu * u + mv * v
        else:
            snapped[ai] = center
    if smooth_window > 1:
        snapped = uniform_filter1d(snapped, size=smooth_window, axis=0, mode="nearest")
    return snapped


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------


def _slot_cc_volume_mm3(
    ct_arr_kji: np.ndarray, r2i: np.ndarray, slot_ras: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    hu_threshold: float = CC_HU_THRESHOLD,
    roi_half_mm: float = CC_ROI_HALF_MM,
) -> float:
    """Volume (mm³) of the saturating-HU connected component containing
    ``slot_ras`` (or the nearest above-threshold voxel) within a ROI cube
    of half-extent ``roi_half_mm``.

    Returns 0.0 if no above-threshold voxel exists in the ROI at all.
    """
    from scipy.ndimage import label as _cc_label

    pt_h = np.array([slot_ras[0], slot_ras[1], slot_ras[2], 1.0])
    ijk = (r2i @ pt_h)[:3]
    i_idx = int(round(ijk[0])); j_idx = int(round(ijk[1])); k_idx = int(round(ijk[2]))

    sx, sy, sz = float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])
    half_i = max(2, int(np.ceil(roi_half_mm / sx)))
    half_j = max(2, int(np.ceil(roi_half_mm / sy)))
    half_k = max(2, int(np.ceil(roi_half_mm / sz)))

    K, J, I = ct_arr_kji.shape
    k_lo = max(0, k_idx - half_k); k_hi = min(K, k_idx + half_k + 1)
    j_lo = max(0, j_idx - half_j); j_hi = min(J, j_idx + half_j + 1)
    i_lo = max(0, i_idx - half_i); i_hi = min(I, i_idx + half_i + 1)
    if k_hi <= k_lo or j_hi <= j_lo or i_hi <= i_lo:
        return 0.0

    roi = ct_arr_kji[k_lo:k_hi, j_lo:j_hi, i_lo:i_hi]
    mask = roi >= hu_threshold
    if not mask.any():
        return 0.0
    labels, _ = _cc_label(mask)

    rk = int(np.clip(k_idx - k_lo, 0, mask.shape[0] - 1))
    rj = int(np.clip(j_idx - j_lo, 0, mask.shape[1] - 1))
    ri = int(np.clip(i_idx - i_lo, 0, mask.shape[2] - 1))
    slot_label = int(labels[rk, rj, ri])
    if slot_label == 0:
        ks, js, is_ = np.where(mask)
        d = (ks - rk) ** 2 + (js - rj) ** 2 + (is_ - ri) ** 2
        n = int(np.argmin(d))
        slot_label = int(labels[ks[n], js[n], is_[n]])
    cc_voxels = int((labels == slot_label).sum())
    return cc_voxels * (sx * sy * sz)


def place_contacts_for_seed_v2(
    seed_start_ras,
    seed_end_ras,
    *,
    features: dict,
    library_models: Sequence[dict],
    sigma_contact_mm: float = SIGMA_CONTACT_MM_DEFAULT,
    min_corr_for_real_shank: float = MIN_CORR_FOR_REAL_SHANK,
    min_slot_hu_mean: float | None = None,
    max_slot_cc_volume_p90_mm3: float | None = None,
    cc_hu_threshold: float = CC_HU_THRESHOLD,
    cc_roi_half_mm: float = CC_ROI_HALF_MM,
    add_valley_anti_template: bool = False,
    valley_anti_alpha: float = 1.0,
    walk_step_mm: float = WALK_STEP_MM,
    walk_disk_radius_mm: float = WALK_DISK_RADIUS_MM,
    first_contact_min_mm: float = WALK_FIRST_CONTACT_MIN_MM,
    walk_tip_pad_mm: float = WALK_TIP_PAD_MM,
    walk_hu_min: float = WALK_HU_MIN,
) -> PlacementV2Result:
    """Place contacts on a seed trajectory using the matched-filter pipeline.

    Args:
        seed_start_ras, seed_end_ras: trajectory endpoints in RAS.
        features: dict from ``rosa_detect.guided_fit_engine.compute_features``
            providing at minimum ``ct_arr_kji``, ``ras_to_ijk_mat``,
            ``ijk_to_ras_mat``, ``head_distance``.
        library_models: candidate electrode models.
        sigma_contact_mm: matched-filter Gaussian σ (≈ contact half-length).
        min_corr_for_real_shank: drop placements with corr below this.
        min_slot_hu_mean: optional opt-in HU floor for the unseeded
            validator. Set to ``MIN_SLOT_HU_MEAN`` (1500) on the
            unseeded path; ``None`` (default) on the seeded path.
        max_slot_cc_volume_p90_mm3: optional opt-in cap on the 90th
            percentile of per-slot saturating-HU connected-component
            volumes. Drops bone-spike chains and surgical-clip artifacts
            that the matched filter can't distinguish from real shanks.
            Set to ``MAX_SLOT_CC_VOLUME_P90_MM3`` (150) on the unseeded
            path; ``None`` (default) on the seeded path.
        cc_hu_threshold, cc_roi_half_mm: HU floor and ROI half-extent
            for the CC measurement.
        walk_*: walker disk-stat sampling knobs.

    Returns:
        ``PlacementV2Result``. ``success=False`` if the matcher rejects
        or the correlation is below the threshold.
    """
    from .contact_placement import estimate_bolt_end_from_metal_mass
    from .contact_placement import sample_disk_along_polyline

    s = np.asarray(seed_start_ras, dtype=float)
    e = np.asarray(seed_end_ras, dtype=float)
    seed_len = float(np.linalg.norm(e - s))
    if seed_len < 1e-3:
        return PlacementV2Result(
            success=False, model_id=None, placed_ras=[], centerline_ras=None,
            corr_score=0.0, bolt_end_arc_mm=0.0, bolt_source="none",
            n_placed=0, rejected_reason="seed_zero_length",
        )

    ct_arr = features["ct_arr_kji"]
    r2i = np.asarray(features["ras_to_ijk_mat"], dtype=float)

    # Stage 1+2: bolt-end estimate.
    #
    # ``estimate_bolt_end_from_metal_mass`` walks forward from the metal-
    # mass peak assuming the bolt is at the seed start. When the seed is
    # reversed (bolt at end), the walk falls off the end and returns
    # None — v2 then falls into bolt_less mode unnecessarily, with worse
    # centerline snap and contacts placed in low-HU positions.
    #
    # Stage1 chains are emitted in arbitrary direction (LoG-blob walk
    # order, not ROSA convention) and several GT files label contacts
    # deep-to-superficial — both produce reversed seeds. So always try
    # both directions and take whichever yields a valid bolt_end with a
    # non-degenerate contact zone. Verified 2026-05-08 on AMC135/hetero-
    # mid-M and AMC137/LPT, both recoverable via seed-flip.
    def _try_bolt_end(seed_start, seed_end):
        try:
            be = estimate_bolt_end_from_metal_mass(
                seed_start, seed_end, features=features,
                library_models=library_models,
            )
        except Exception:
            return None, None
        be_arc = be.get("bolt_end_arc_mm")
        cp = be.get("centerline")
        if be_arc is None or cp is None:
            return None, None
        # Reject degenerate (no contact zone left after the bolt).
        cp_total = float(np.linalg.norm(np.diff(cp, axis=0), axis=1).sum())
        if cp_total - float(be_arc) < DEGENERATE_CONTACT_ZONE_MM:
            return None, None
        return be_arc, cp

    bolt_end_straight, cp = _try_bolt_end(s, e)
    if bolt_end_straight is None:
        # Try reversed seed.
        be_rev, cp_rev = _try_bolt_end(e, s)
        if be_rev is not None:
            # Adopt the reversed seed for the rest of the pipeline.
            s, e = e, s
            seed_len = float(np.linalg.norm(e - s))
            bolt_end_straight, cp = be_rev, cp_rev

    # Decide bolt-less fallback vs standard.
    use_fallback = bolt_end_straight is None or cp is None

    if use_fallback:
        centerline = _straight_centerline(s, e)
        bolt_source = "bolt_less"
        bolt_end_cl_arc = 0.0
        max_extend = 0.0
    else:
        centerline_poly = np.asarray(cp, dtype=float)
        # Stage 3: snap polynomial centerline to local high-HU centroid
        # (recovers ~5-10% of placements where the polynomial axis is
        # 1-2 mm off the actual electrode axis).
        centerline = _snap_centerline_to_centroid(centerline_poly, ct_arr, r2i)
        u_str = (e - s) / seed_len
        bolt_pt = s + float(bolt_end_straight) * u_str
        bolt_end_cl_arc = _project_to_polyline_arc(centerline, bolt_pt)
        bolt_source = "metal"
        max_extend = walk_tip_pad_mm

    # Centerline arc-length total (un-extended; this is the "true"
    # contact-zone length the matcher considers via profile_end_arc).
    cl_max = float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum())

    # Stage 4: walker disk-stat sampling. Extend the centerline tail by
    # ``walk_tip_pad_mm`` so the walker samples signal slightly past the
    # auto-fit axis tip — lets the matched filter evaluate model tip
    # positions that fall just past the polynomial endpoint.
    centerline_for_walker = _extend_centerline_tail(centerline, max_extend)
    arcs, max_b, _total_b = sample_disk_along_polyline(
        ct_arr, r2i, centerline_for_walker,
        polarity="positive", step_mm=walk_step_mm,
        disk_radius_mm=walk_disk_radius_mm,
        n_radii=WALK_N_RADII, n_angles=WALK_N_ANGLES,
        total_threshold=walk_hu_min,
    )
    signal = max_b

    # Stage 5: matched-filter library pick.
    match: MatchedFilterResult = matched_filter_pick(
        arcs, signal, library_models,
        bolt_end_arc=bolt_end_cl_arc,
        first_contact_min_mm=first_contact_min_mm,
        profile_end_arc=cl_max,
        max_extend_tip_mm=max_extend,
        sigma_contact_mm=sigma_contact_mm,
        add_valley_anti_template=add_valley_anti_template,
        valley_anti_alpha=valley_anti_alpha,
    )

    if match.best_model_id is None:
        return PlacementV2Result(
            success=False, model_id=None, placed_ras=[], centerline_ras=centerline.tolist(),
            corr_score=0.0, bolt_end_arc_mm=bolt_end_cl_arc, bolt_source=bolt_source,
            n_placed=0, rejected_reason="no_model",
            diagnostics={"cl_max_mm": cl_max, "in_zone_p75": float(np.percentile(signal, 75))},
        )

    # Validator gate: matched-filter score is the trajectory check.
    if match.corr < float(min_corr_for_real_shank):
        return PlacementV2Result(
            success=False, model_id=match.best_model_id, placed_ras=[],
            centerline_ras=centerline.tolist(),
            corr_score=match.corr, bolt_end_arc_mm=bolt_end_cl_arc,
            bolt_source=bolt_source, n_placed=match.n_slots,
            rejected_reason=f"corr_below_threshold({match.corr:.3f}<{min_corr_for_real_shank:.3f})",
        )

    # Convert slot arcs to RAS placements along the centerline.
    placed = [_polyline_pos_at_arc(centerline, float(a)).tolist()
               for a in match.slot_arcs]

    # Per-slot HU floor: real-shank slot HU is consistently 1500-3000+;
    # cross-shank/bone FP chains average lower. Optional opt-in filter
    # (default off in seeded mode where caller has a known-good axis).
    if min_slot_hu_mean is not None:
        from .volume_sampling import sample_trilinear_at_ras
        slot_hus = []
        for pt in placed:
            hu = float(sample_trilinear_at_ras(ct_arr, r2i, np.asarray(pt, dtype=float)))
            if np.isfinite(hu): slot_hus.append(hu)
        slot_hu_mean = float(np.mean(slot_hus)) if slot_hus else 0.0
        if slot_hu_mean < float(min_slot_hu_mean):
            return PlacementV2Result(
                success=False, model_id=match.best_model_id, placed_ras=[],
                centerline_ras=centerline.tolist(),
                corr_score=match.corr, bolt_end_arc_mm=bolt_end_cl_arc,
                bolt_source=bolt_source, n_placed=match.n_slots,
                rejected_reason=(
                    f"slot_hu_mean_below_threshold"
                    f"({slot_hu_mean:.0f}<{min_slot_hu_mean:.0f})"
                ),
                diagnostics={"slot_hu_mean": slot_hu_mean},
            )

    # Per-slot CC-volume cap: bone-spike chains and surgical-clip
    # artifacts have at least one slot whose saturating-HU connected
    # component spans 150+ mm³ (vs ≤140 mm³ for real shanks).
    if max_slot_cc_volume_p90_mm3 is not None and placed:
        spacing_xyz = features["img"].GetSpacing()
        slot_volumes = [
            _slot_cc_volume_mm3(
                ct_arr, r2i, np.asarray(pt, dtype=float), spacing_xyz,
                hu_threshold=float(cc_hu_threshold),
                roi_half_mm=float(cc_roi_half_mm),
            )
            for pt in placed
        ]
        slot_volumes_arr = np.asarray(slot_volumes, dtype=float)
        slot_cc_p90 = float(np.percentile(slot_volumes_arr, 90))
        if slot_cc_p90 > float(max_slot_cc_volume_p90_mm3):
            return PlacementV2Result(
                success=False, model_id=match.best_model_id, placed_ras=[],
                centerline_ras=centerline.tolist(),
                corr_score=match.corr, bolt_end_arc_mm=bolt_end_cl_arc,
                bolt_source=bolt_source, n_placed=match.n_slots,
                rejected_reason=(
                    f"slot_cc_volume_p90_above_threshold"
                    f"({slot_cc_p90:.1f}>{max_slot_cc_volume_p90_mm3:.1f})"
                ),
                diagnostics={"slot_cc_volume_p90_mm3": slot_cc_p90},
            )

    return PlacementV2Result(
        success=True,
        model_id=match.best_model_id,
        placed_ras=placed,
        centerline_ras=centerline.tolist(),
        corr_score=match.corr,
        bolt_end_arc_mm=bolt_end_cl_arc,
        bolt_source=bolt_source,
        n_placed=match.n_slots,
        rejected_reason="",
        diagnostics={
            "cl_max_mm": cl_max,
            "tip_arc_mm": float(match.tip_arc) if match.tip_arc is not None else float("nan"),
            "n_covered": match.n_covered,
        },
    )


__all__ = [
    "MIN_CORR_FOR_REAL_SHANK",
    "MAX_SLOT_CC_VOLUME_P90_MM3",
    "MIN_SLOT_HU_MEAN",
    "PlacementV2Result",
    "place_contacts_for_seed_v2",
]
