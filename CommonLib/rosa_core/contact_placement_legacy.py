"""Contact placement on a CT volume given seed trajectories.

Place contacts along each seed trajectory (auto-fit / guided-fit /
manual proposal). The pipeline, validated on AMC88 / AMC91 / AMC135 /
AMC136 / AMC137 / T22 (49/66 exact-count, 0.40 mm median error,
98.5% on-metal):

  1. Sample HU profile along the seed axis to find the bolt → contact
     entry transition (``signal_derived_entry_arc``).
  2. Filter 3D LoG blobs (precomputed by ``compute_features``) into a
     corridor around the seed axis, optionally restricted to blobs
     assigned to this trajectory's owner (cross-shank disambiguation).
  3. Partition corridor blobs into bolt-side vs contact-side via the
     largest along-axis gap that leaves enough contacts on the deep
     side (``min_contact_side`` floor — guards against stray-blob
     past-tip mis-snap).
  4. Bent-loop RANSAC: pick the best-fitting library model, refit a
     polynomial through inliers, re-arc-length under the curve,
     iterate until the inlier set stabilizes.
  5. Place contacts at library-defined offsets along the fitted curve.

Curved seeds (``path_ras`` polyline ≥ 3 points) are accepted; arc-length
is computed along the polyline instead of the straight segment. The
bent-loop polynomial refit is unchanged — the seed curve is a better
initial parameterization, not a constraint.

This is the v2 placement path (multi-shank-aware, bent-loop, library-
matched). The older 1D peak-only ``contact_peak_fit.detect_contacts_on_axis``
remains for callers that don't want the full feature pipeline.

Public surface:
    place_contacts_for_trajectories(img, ..., trajectories) -> Batch
    place_contacts_for_axis(start, end, *, features, ...) -> Result
    assign_axis_owners(blob_pts_ras, axes) -> ndarray
    ContactPlacementConfig, ContactPlacementResult, ContactPlacementBatch

Boundary-clean of Slicer / VTK / Qt. Caller (Slicer or CLI) is
responsible for getting the SimpleITK image + IJK↔RAS matrices.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ContactPlacementConfig:
    """Tunables for the bent-RANSAC placement pipeline.

    Defaults match the validated ``ransac_all_subjects.ipynb``
    constants (49/66 exact-count, 0.40 mm median error). Overriding
    individual fields requires evidence that the new value preserves
    or improves the dataset regression — see the dataset gate in
    ``tests/rosa_core/test_contact_placement.py``.
    """

    # HU profile sampling along the axis.
    profile_step_mm: float = 0.25
    profile_disk_radius_mm: float = 1.0
    profile_n_radii: int = 2
    profile_n_angles: int = 8
    profile_pad_bolt_mm: float = 0.0
    profile_pad_tip_mm: float = 5.0

    # Corridor that filters the 3D blob cloud to "near this axis".
    corridor_radius_mm: float = 6.0
    peak_hu_min: float = 500.0
    along_axis_lo_pad_mm: float = 2.0  # accept blobs from -2 mm before bolt

    # Cross-shank ownership (when caller computes one for the batch).
    ownership_max_perp_mm: float = 4.0

    # Bolt / contact partition. Cascade has four tiers (HU profile
    # signature → hull proximity → library-aware gap → no cut); the
    # legacy ``gap_only`` path is kept for ablation. See
    # ``_partition_bolt_from_contacts`` for the full rationale.
    #
    # Default is ``gap_only`` because the validated 49/66 dataset
    # number was achieved with that path. The cascade is conceptually
    # correct (handles 15CM/15BM/18CM design gaps and hull-proximate
    # bolts) but currently regresses 9 shanks against gap_only —
    # primarily because ``signal_derived_entry_arc`` (Tier 1) returns
    # values on real subjects that pass the library plausibility check
    # yet cut sub-optimally. Use ``notebooks/bolt_partition_qc.ipynb``
    # to diagnose per-shank tier behavior; flip the default to
    # ``cascade`` once Tier 1 is tuned to ≥ 49/67.
    partition_strategy: Literal["cascade", "gap_only"] = "gap_only"
    partition_gap_thresh_mm: float = 5.0
    partition_min_contact_side: int = 4
    partition_keep_buffer_mm: float = 1.0
    bolt_hull_proximity_mm: float = 3.0
    library_check_min_inliers: int = 3
    library_check_inlier_tol_mm: float = 1.5
    library_check_pair_tol_mm: float = 1.0
    library_check_min_coverage: float = 0.6

    # Tier 1A (metal-mass-derived bolt-end). Computed by
    # estimate_bolt_end_from_metal_mass when ``features`` and the
    # axis endpoints are passed to ``_partition_bolt_from_contacts``.
    # Validated in ``notebooks/bolt_partition_qc.ipynb`` on AMC88+T22.
    partition_metal_mass_tier: bool = True
    metal_mass_plateau_frac: float = 0.50
    metal_mass_max_gap_mm: float = 6.0
    metal_mass_padding_mm: float = 0.5

    # RANSAC library match.
    ransac_pair_tol_mm: float = 0.6
    ransac_inlier_tol_mm: float = 1.0
    ransac_max_extend_tip_mm: float = 5.0
    ransac_phantom_penalty: float = 1.0
    ransac_bolt_free_pass: bool = True

    # Bent loop.
    max_bent_iterations: int = 4

    # Curve fit.
    curve_fit: Literal["polynomial", "spline"] = "polynomial"
    poly_deg: int = 2


# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------


@dataclass
class ContactPlacementResult:
    """Output of placement on one trajectory."""

    name: str
    success: bool
    model_id: str
    positions_ras: list[list[float]]
    placement_kind: Literal["polynomial", "spline", "straight", "failed"]
    n_iter: int
    score: float
    sign: float
    tip_arc_mm: float
    eff_entry_arc_mm: float | None
    n_inliers: int
    rejected_reason: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContactPlacementBatch:
    """Output of placement across many trajectories on one volume."""

    results: list[ContactPlacementResult]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def by_name(self) -> dict[str, ContactPlacementResult]:
        return {r.name: r for r in self.results}


# ---------------------------------------------------------------------
# Curve fit (polynomial + spline)
# ---------------------------------------------------------------------
#
# Both return a triple ``(sp, t2s, ctrl)``:
#   ``sp(tau)``    : map sampled axis-arcs ``tau`` to RAS positions on
#                    the curve. Clipped to the sampled range.
#   ``t2s(tau)``   : map ``tau`` (axis arc-length) to ``s`` (curve arc-
#                    length). The bent loop calls this on every blob's
#                    arc to re-parameterize the corridor under the
#                    fitted curve.
#   ``ctrl``       : ``(ts, pts, t_d, s_d)`` — original control points
#                    plus dense ``t``/``s`` samples for QC plotting.
#
# Polynomial deg-2 is the production default (validated +3 exact-count
# vs cubic spline on AMC88/91/135/136/137/T22, 49 vs 46). Spline kept
# for QC ablation.


def _fit_polynomial_through_points(pts, arcs, deg=2):
    """Per-axis deg-N polynomial fit. Same contract as the spline fn."""
    pts = np.asarray(pts, dtype=float)
    arcs = np.asarray(arcs, dtype=float)
    if len(arcs) < deg + 1:
        return None, None, None
    o = np.argsort(arcs)
    ts = arcs[o]
    pts = pts[o]
    keep = np.concatenate([[True], np.diff(ts) > 1e-3])
    ts = ts[keep]
    pts = pts[keep]
    if len(ts) < deg + 1:
        return None, None, None
    cx = np.polyfit(ts, pts[:, 0], deg)
    cy = np.polyfit(ts, pts[:, 1], deg)
    cz = np.polyfit(ts, pts[:, 2], deg)
    t_d = np.linspace(ts[0], ts[-1], 4000)
    xyz = np.column_stack([
        np.polyval(cx, t_d),
        np.polyval(cy, t_d),
        np.polyval(cz, t_d),
    ])
    s_d = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    )

    def sp(tau):
        tc = np.clip(np.asarray(tau, float), ts[0], ts[-1])
        return np.column_stack([
            np.polyval(cx, tc),
            np.polyval(cy, tc),
            np.polyval(cz, tc),
        ])

    def t2s(tau):
        tau = np.asarray(tau, float)
        s = np.interp(tau, t_d, s_d)
        # Linear extrapolation outside the sample range — keeps the
        # bent loop well-defined when blobs sit just past the inlier
        # span.
        s = np.where(tau < ts[0], tau - ts[0], s)
        s = np.where(tau > ts[-1], s_d[-1] + (tau - ts[-1]), s)
        return s

    return sp, t2s, (ts, pts, t_d, s_d)


def _fit_spline_through_points(pts, arcs):
    """Cubic spline through (arc, position) samples. Kept for ablation;
    polynomial deg-2 is the production default."""
    from scipy.interpolate import CubicSpline

    pts = np.asarray(pts, dtype=float)
    arcs = np.asarray(arcs, dtype=float)
    if len(arcs) < 4:
        return None, None, None
    o = np.argsort(arcs)
    ts = arcs[o]
    pts = pts[o]
    keep = np.concatenate([[True], np.diff(ts) > 1e-3])
    ts = ts[keep]
    pts = pts[keep]
    if len(ts) < 4:
        return None, None, None
    sx = CubicSpline(ts, pts[:, 0])
    sy = CubicSpline(ts, pts[:, 1])
    sz = CubicSpline(ts, pts[:, 2])
    t_d = np.linspace(ts[0], ts[-1], 4000)
    xyz = np.column_stack([sx(t_d), sy(t_d), sz(t_d)])
    s_d = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    )

    def sp(tau):
        tc = np.clip(np.asarray(tau, float), ts[0], ts[-1])
        return np.column_stack([sx(tc), sy(tc), sz(tc)])

    def t2s(tau):
        tau = np.asarray(tau, float)
        s = np.interp(tau, t_d, s_d)
        s = np.where(tau < ts[0], tau - ts[0], s)
        s = np.where(tau > ts[-1], s_d[-1] + (tau - ts[-1]), s)
        return s

    return sp, t2s, (ts, pts, t_d, s_d)


def _curve_fit_dispatch(pts, arcs, *, kind, poly_deg):
    if kind == "polynomial":
        return _fit_polynomial_through_points(pts, arcs, deg=poly_deg)
    if kind == "spline":
        return _fit_spline_through_points(pts, arcs)
    raise ValueError(f"unknown curve_fit kind: {kind!r}")


# ---------------------------------------------------------------------
# Polyline arc-length helpers (curved seeds)
# ---------------------------------------------------------------------


def _polyline_segments(path_ras: np.ndarray):
    """Return per-segment start, direction-unit, length for a polyline.

    ``path_ras`` shape ``(K, 3)`` with ``K >= 2``. Zero-length segments
    are dropped (consecutive duplicate points).
    """
    P = np.asarray(path_ras, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] < 2:
        raise ValueError("path_ras must be (K,3) with K>=2")
    diffs = P[1:] - P[:-1]
    lens = np.linalg.norm(diffs, axis=1)
    keep = lens > 1e-9
    if not keep.any():
        raise ValueError("path_ras has zero arc length")
    starts = P[:-1][keep]
    units = (diffs[keep] / lens[keep, None])
    seglens = lens[keep]
    cum_start = np.concatenate([[0.0], np.cumsum(seglens)[:-1]])
    return starts, units, seglens, cum_start


def _project_to_polyline(points: np.ndarray, path_ras: np.ndarray):
    """Return (arc, perp) for each query point against the polyline.

    For each query, picks the closest segment by perpendicular distance;
    arc is the cumulative-along-path distance to the projection point;
    perp is the perpendicular distance from the query to the projection.

    Extrapolation: the FIRST segment accepts along < 0 (query past the
    proximal end of the path) and the LAST segment accepts along > L
    (query past the distal tip). Interior segments are strictly
    clamped — a query above an L-bend stays on the elbow, not on a
    phantom extension.

    For straight 2-point paths (single segment), this collapses to the
    standard ``along = (q - start) @ unit`` projection with full
    extrapolation in both directions.
    """
    starts, units, seglens, cum_start = _polyline_segments(path_ras)
    Q = np.asarray(points, dtype=float)
    n = Q.shape[0]
    n_seg = len(starts)
    best_arc = np.full(n, np.inf, dtype=float)
    best_perp = np.full(n, np.inf, dtype=float)
    for k, (s, u, L, c0) in enumerate(zip(starts, units, seglens, cum_start)):
        d = Q - s
        along = d @ u
        # Per-segment extrapolation policy.
        lo = -np.inf if k == 0 else 0.0
        hi = np.inf if k == n_seg - 1 else float(L)
        along_used = np.clip(along, lo, hi)
        proj = s + along_used[:, None] * u
        perp = np.linalg.norm(Q - proj, axis=1)
        better = perp < best_perp
        best_perp = np.where(better, perp, best_perp)
        best_arc = np.where(better, c0 + along_used, best_arc)

    return best_arc, best_perp


def _polyline_total_length_mm(path_ras: np.ndarray) -> float:
    """Total polyline arc length (sum of segment lengths)."""
    P = np.asarray(path_ras, dtype=float)
    if P.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))


def _resolve_seed_path(
    start_ras, end_ras, path_ras=None,
) -> tuple[np.ndarray, float]:
    """Return (path, total_length_mm). path is (K,3) with K>=2.

    Straight seeds collapse to ``[start, end]``. Curved seeds use the
    given polyline. Both downstream consumers — corridor projection and
    initial-arc parameterization — work on a uniform polyline view.
    """
    if path_ras is not None:
        P = np.asarray(path_ras, dtype=float)
        if P.ndim == 2 and P.shape[0] >= 2 and P.shape[1] == 3:
            L = _polyline_total_length_mm(P)
            if L >= 1e-3:
                return P, L
    s = np.asarray(start_ras, dtype=float).reshape(3)
    e = np.asarray(end_ras, dtype=float).reshape(3)
    P = np.stack([s, e], axis=0)
    L = float(np.linalg.norm(e - s))
    return P, L


def _polyline_axis_at(path_ras: np.ndarray, arc: float) -> np.ndarray:
    """Sample the polyline at arc-length ``arc`` (RAS point)."""
    starts, units, seglens, cum_start = _polyline_segments(path_ras)
    if arc <= 0:
        return starts[0] + arc * units[0]
    total = float(cum_start[-1] + seglens[-1])
    if arc >= total:
        return starts[-1] + (arc - cum_start[-1]) * units[-1]
    # Find the segment containing ``arc``.
    idx = int(np.searchsorted(cum_start + seglens, arc, side="right"))
    idx = min(idx, len(starts) - 1)
    along = arc - cum_start[idx]
    return starts[idx] + along * units[idx]


# ---------------------------------------------------------------------
# Bolt / contact partition (tiered cascade)
# ---------------------------------------------------------------------
#
# Job: identify the rightmost-arc bolt-material boundary so RANSAC can
# match library offsets against contact peaks without bolt blobs
# inflating the inlier count.
#
# The original "largest gap" heuristic conflated three different gap
# types:
#   1. Bolt → contact transition (5-30 mm, what we want to find)
#   2. Intra-electrode pitch (3-12 mm)
#   3. Intra-electrode design gap (DIXI 15CM/15BM/18CM, 12-30 mm — by
#      device design, NOT a bolt boundary)
# It only worked when (1) > (3) > (2). For long-gap electrodes this is
# accidental — when the bolt sits close to the proximal contact (e.g.
# cropped scan, no skull margin) the design gap exceeds the bolt-
# contact gap and the heuristic cuts mid-shank.
#
# The cascade addresses this by routing through the discriminator that
# ACTUALLY identifies bolt material:
#   Tier 1 — HU profile signature (``signal_derived_entry_arc``).
#     Sustained-bright run ≥ 1.5 mm with peak ≥ 2400 HU. Strongest
#     signal when the bolt is fully imaged.
#   Tier 2 — Hull proximity. Bolt material sits within ~3 mm of the
#     hull boundary (head_distance ≈ 0); contact material is past the
#     dura. Works when the HU profile heuristic misfires (e.g. blob
#     cloud is sparse along axis but features['head_distance'] is
#     dense).
#   Tier 3 — Library-aware gap walk. For each candidate gap (largest
#     first), accept the cut iff the contact-side blob arcs plausibly
#     match SOME library model's offset structure. A 15CM design gap,
#     if used as a cut, leaves a 7-8-blob distal-only pattern that
#     matches no library model — so the cut is rejected and the walk
#     continues to a wider gap.
#   Tier 4 — No cut. Feed the full corridor into RANSAC; the
#     phantom_bolt_side penalty handles bolt blobs at the cost of
#     some scoring noise.
#
# The legacy ``gap_only`` path is preserved behind the
# ``ContactPlacementConfig.partition_strategy`` flag for ablation.


def _plausibly_matches_library(
    contact_side_arcs: np.ndarray,
    library_models: Sequence[dict] | None,
    *,
    min_inliers: int,
    pair_tol_mm: float,
    inlier_tol_mm: float,
    min_coverage: float = 0.6,
) -> bool:
    """Tier 3 plausibility check: does the candidate contact-side blob
    pattern align with SOME library model AS A FULL ELECTRODE? Used
    to reject mid-electrode design-gap cuts.

    Two gates:
      * ``inl >= min_inliers`` — at least this many slots aligned (anti-
        noise threshold).
      * ``inl / model_slots >= min_coverage`` — the matched portion
        covers most of the model. Without coverage, a 7-arc subset of
        a 15-contact DIXI 15CM would pass (the RANSAC happily aligns
        7/15 slots) and we'd accept a wrong design-gap cut.

    Implemented as a relaxed-tolerance RANSAC pass over the library
    with ``phantom_penalty=0`` (we want to know whether any FULL
    pattern matches, not the best phantom-penalized score). Returns
    False on empty inputs.
    """
    if library_models is None or len(library_models) == 0:
        return False
    arcs = np.asarray(contact_side_arcs, dtype=float)
    if arcs.size < min_inliers:
        return False
    rb = _ransac_pick_library(
        arcs, list(library_models),
        entry_arc=None,
        profile_end_arc=float(arcs.max()) + 50.0,
        pair_tol_mm=pair_tol_mm,
        inlier_tol_mm=inlier_tol_mm,
        max_extend_tip_mm=20.0,
        bolt_free_pass=True,
        phantom_penalty=0.0,
    )
    if rb is None:
        return False
    inl = int(rb.get("inl", 0))
    ns = max(1, int(rb.get("ns", 1)))
    if inl < int(min_inliers):
        return False
    coverage = inl / ns
    return coverage >= float(min_coverage)


def _partition_bolt_from_contacts(
    arcs: Sequence[float],
    *,
    entry_arc: float | None = None,
    head_distance_at_blob: np.ndarray | None = None,
    library_models: Sequence[dict] | None = None,
    config: ContactPlacementConfig | None = None,
    # Tier 1A inputs (NEW): metal-mass-derived bolt-end via
    # ``estimate_bolt_end_from_metal_mass``. Optional — when missing,
    # the cascade falls back to Tier 1 (signal_derived_entry_arc).
    features: dict | None = None,
    bolt_outer_ras: np.ndarray | None = None,
    deep_end_ras: np.ndarray | None = None,
) -> tuple[float | None, str]:
    """Tiered cascade: metal-mass → HU profile → hull proximity → library-aware gap.

    Each tier proposes a cut; we accept the FIRST cut whose contact-
    side passes the library plausibility check. This guards against
    a tier returning a slightly-off value that mis-cuts the contact
    array — the library check rejects such proposals and the cascade
    falls through to the next tier.

    Returns ``(eff_entry_arc, tier_used)``. ``eff_entry_arc=None`` plus
    ``tier_used="no_cut"`` means no boundary identified — caller should
    feed the full corridor into RANSAC.

    Tiers (in order of preference):
      1A. ``metal_mass`` — total HU mass per perpendicular disk along
          the trajectory; bolt-end where mass drops below
          plateau_frac × peak after envelope smoothing with σ =
          library median pitch. See ``estimate_bolt_end_from_metal_mass``.
          Most robust on multi-peak bolts (nut + bone-collar) where
          Tier 1 misfires.
      1B. ``hu_profile`` — ``signal_derived_entry_arc`` on a max-disk
          HU profile (legacy, still used as fallback).
      2.  ``hull_proximity`` — proximal-most blob with head distance
          past the dura tolerance.
      3.  ``library_aware_gap`` — largest qualifying gap whose contact-
          side blob arrangement plausibly matches some library model.
      4.  ``no_cut`` — no boundary identified.
    """
    cfg = config or ContactPlacementConfig()
    arr = np.asarray(arcs, dtype=float)
    if arr.size < 2:
        return entry_arc, "trivial"
    s_sorted = np.sort(arr)

    def library_passes(cut_value: float) -> bool:
        right = s_sorted[s_sorted >= float(cut_value) - cfg.partition_keep_buffer_mm]
        if right.size < int(cfg.library_check_min_inliers):
            return False
        return _plausibly_matches_library(
            right, library_models,
            min_inliers=cfg.library_check_min_inliers,
            pair_tol_mm=cfg.library_check_pair_tol_mm,
            inlier_tol_mm=cfg.library_check_inlier_tol_mm,
            min_coverage=cfg.library_check_min_coverage,
        )

    # Build ordered proposals. Each is (cut, tier_label).
    proposals: list[tuple[float, str]] = []

    # Tier 1A: metal-mass-derived bolt-end (NEW).
    if (cfg.partition_metal_mass_tier
            and features is not None
            and bolt_outer_ras is not None
            and deep_end_ras is not None):
        try:
            mm_result = estimate_bolt_end_from_metal_mass(
                bolt_outer_ras, deep_end_ras,
                features=features,
                library_models=library_models,
                plateau_frac=cfg.metal_mass_plateau_frac,
                max_gap_mm=cfg.metal_mass_max_gap_mm,
                padding_mm=cfg.metal_mass_padding_mm,
            )
            mm_bolt_end = mm_result.get("bolt_end_arc_mm")
            if mm_bolt_end is not None:
                proposals.append((float(mm_bolt_end), "metal_mass"))
        except Exception as exc:
            # Defensive: a sampling failure (out-of-FOV axis, etc.)
            # should fall through to other tiers, not crash placement.
            # Surface the failure as a warning so wiring bugs (like a
            # kwarg mismatch) don't hide silently — the previous bare
            # `pass` masked a `padding_mm` argument-routing bug for an
            # entire dataset run.
            warnings.warn(
                f"metal_mass tier failed: {exc!r}",
                RuntimeWarning, stacklevel=2,
            )

    # Tier 1B: HU profile signature.
    if entry_arc is not None:
        proposals.append((float(entry_arc), "hu_profile"))

    # Tier 2: hull proximity. Proximal-most blob past the dura (head
    # distance > bolt-hull tolerance) is the contact-side boundary.
    if (head_distance_at_blob is not None
            and len(head_distance_at_blob) == len(arr)):
        hd = np.asarray(head_distance_at_blob, dtype=float)
        order = np.argsort(arr)
        contact_mask_sorted = (
            hd[order] > float(cfg.bolt_hull_proximity_mm)
        )
        if contact_mask_sorted.any():
            first_idx = int(np.argmax(contact_mask_sorted))
            proposals.append(
                (float(arr[order[first_idx]]), "hull_proximity")
            )

    # Tier 3: library-aware gap walk. Each candidate gap (largest
    # first, ≥ gap_thresh_mm, contact-side count ≥ min_contact_side)
    # adds one proposal.
    gaps = np.diff(s_sorted)
    candidates = sorted(
        [(float(gaps[i]), int(i)) for i in range(len(gaps))
         if gaps[i] >= float(cfg.partition_gap_thresh_mm)],
        key=lambda t: -t[0],
    )
    for _g, i in candidates:
        if (len(s_sorted) - i - 1) < int(cfg.partition_min_contact_side):
            continue
        proposals.append(
            (float(s_sorted[i + 1]), "library_aware_gap")
        )

    # If we have no library to validate against, accept the first
    # proposal in cascade order without checking. This mirrors the
    # legacy gap_only behaviour for environments that don't supply
    # library models (rare; library is loaded lazily by callers).
    if library_models is None or len(library_models) == 0:
        if proposals:
            return proposals[0]
        return entry_arc, ("trivial" if entry_arc is not None else "no_cut")

    for cut, tier in proposals:
        if library_passes(cut):
            return cut, tier

    # No proposal passed the library check. Fall back to entry_arc
    # without validation — better to attempt placement against a
    # weakly-supported boundary than to drop the trajectory.
    if entry_arc is not None:
        return float(entry_arc), "hu_profile_unvalidated"
    return None, "no_cut"


def _partition_bolt_contacts_by_arc(
    arcs: Sequence[float],
    entry_arc: float | None,
    *, gap_thresh_mm: float, min_contact_side: int,
) -> float | None:
    """Legacy "largest gap" heuristic. Kept as the ``gap_only`` strategy
    for ablation against the cascade.

    Behaviour:
      - The largest along-axis gap >= ``gap_thresh_mm`` whose contact-
        side has at least ``min_contact_side`` blobs is the cut.
      - Falls back to ``entry_arc`` when no qualifying gap exists.

    Conceptually wrong for long-gap electrodes (15CM/15BM/18CM) when
    the bolt-to-proximal gap is smaller than the design gap. See the
    cascade docstring above for the corrected discriminator.
    """
    arr = np.sort(np.asarray(arcs, dtype=float))
    if len(arr) < 2:
        return entry_arc
    gaps = np.diff(arr)
    if len(gaps) == 0:
        return entry_arc
    candidates = sorted(
        [(float(gaps[i]), int(i)) for i in range(len(gaps))
         if gaps[i] >= float(gap_thresh_mm)],
        key=lambda t: -t[0],
    )
    for _g, i in candidates:
        if (len(arr) - i - 1) >= int(min_contact_side):
            return float(arr[i + 1])
    return entry_arc


# ---------------------------------------------------------------------
# RANSAC library match
# ---------------------------------------------------------------------


def _ransac_pick_library(
    peaks: np.ndarray,
    models: list[dict],
    *,
    entry_arc: float | None,
    profile_end_arc: float,
    pair_tol_mm: float,
    inlier_tol_mm: float,
    max_extend_tip_mm: float,
    bolt_free_pass: bool,
    phantom_penalty: float,
) -> dict | None:
    """Brute-force RANSAC over library models.

    For each model, try every ordered pair of detected peaks as the
    (slot_i, slot_j) anchors and both tip orientations (sign=±1). Score
    by inlier count minus phantom penalties. Return the best (model_id,
    score, tip_arc, sign).

    Phantom penalty: a slot whose nominal position has no peak within
    ``inlier_tol_mm`` adds 1 phantom (down-weighted by
    ``phantom_penalty``). Slots that fall past the profile end by more
    than ``max_extend_tip_mm`` disqualify the candidate entirely
    (electrode would extend off the imaged volume).
    """
    peaks = np.asarray(sorted(peaks), dtype=float)
    n = len(peaks)
    if n < 2:
        return None
    best: dict | None = None
    for model in models:
        offsets = np.asarray(
            model.get("contact_center_offsets_from_tip_mm") or [],
            dtype=float,
        )
        ns = offsets.size
        if ns < 2:
            continue
        bl: dict | None = None
        for sign in (+1.0, -1.0):
            for ia in range(n):
                for ib in range(ia + 1, n):
                    pa = float(peaks[ia])
                    pb = float(peaks[ib])
                    for si in range(ns):
                        for sj in range(ns):
                            if si == sj:
                                continue
                            tip = pa - sign * offsets[si]
                            pj = tip + sign * offsets[sj]
                            if abs(pj - pb) > pair_tol_mm:
                                continue
                            inl = pi_count = pb_count = pe_count = 0
                            disq = False
                            for sk in range(ns):
                                pk = tip + sign * offsets[sk]
                                d = float(np.min(np.abs(peaks - pk)))
                                if d <= inlier_tol_mm:
                                    inl += 1
                                elif (entry_arc is not None
                                      and pk < float(entry_arc)):
                                    pb_count += 1
                                elif pk > profile_end_arc + max_extend_tip_mm:
                                    disq = True
                                    break
                                elif pk > profile_end_arc:
                                    pe_count += 1
                                else:
                                    pi_count += 1
                            if disq:
                                continue
                            score = (
                                float(inl)
                                - phantom_penalty * pi_count
                                - phantom_penalty * pe_count
                            )
                            if bl is None or score > bl["score"]:
                                bl = {
                                    "model_id": str(model["id"]),
                                    "score": score,
                                    "inl": inl,
                                    "phantom_interior": pi_count,
                                    "phantom_bolt_side": pb_count,
                                    "phantom_past_end": pe_count,
                                    "ns": ns,
                                    "tip_arc": tip,
                                    "sign": sign,
                                }
        if bl is not None and (best is None or bl["score"] > best["score"]):
            best = bl
    # ``bolt_free_pass`` is currently informational. Future work: when
    # ``entry_arc is None``, weighting the phantom_bolt_side category
    # may want to differ; for now the score function treats it the same.
    _ = bolt_free_pass
    return best


def _identify_inliers(
    peak_arcs: np.ndarray,
    ransac_best: dict,
    models: list[dict],
    tol: float,
) -> list[int]:
    """Match each model slot to its nearest peak (greedy, no double-
    assignment). Returns indices into ``peak_arcs``."""
    if ransac_best is None:
        return []
    model = next(m for m in models if str(m["id"]) == ransac_best["model_id"])
    offsets = np.asarray(model["contact_center_offsets_from_tip_mm"])
    tip = ransac_best["tip_arc"]
    sign = ransac_best["sign"]
    arr = np.asarray(peak_arcs)
    used: set[int] = set()
    inl: list[int] = []
    for off in offsets:
        d = np.abs(arr - (tip + sign * off))
        for c in np.argsort(d):
            ci = int(c)
            if ci in used:
                continue
            if d[ci] <= tol:
                used.add(ci)
                inl.append(ci)
            break
    return sorted(set(inl))


# ---------------------------------------------------------------------
# Cross-shank ownership
# ---------------------------------------------------------------------


def assign_axis_owners(
    blob_pts_ras: np.ndarray,
    axes: Sequence[dict],
    *,
    max_perp_mm: float = 4.0,
    max_extend_tip_mm: float = 5.0,
    along_axis_lo_pad_mm: float = 2.0,
) -> np.ndarray:
    """Assign each blob to its closest axis by perpendicular distance.

    ``axes`` is a list of dicts with ``start_ras`` and ``end_ras``
    (and optionally ``path_ras`` for curved seeds). Returns an integer
    array of axis indices into ``axes``; ``-1`` means unowned.

    A blob is unowned when its perpendicular to every axis exceeds
    ``max_perp_mm``, or it falls outside every axis's along-extent
    (with the configured pads for curved-tip extension).

    Used by ``place_contacts_for_trajectories`` once per CT to set up
    the owner masks fed to the per-axis call. Also useable by callers
    that want a different placement strategy (e.g. notebooks doing QC).
    """
    pts = np.asarray(blob_pts_ras, dtype=float)
    n = pts.shape[0]
    n_ax = len(axes)
    if n_ax == 0 or n == 0:
        return np.full(n, -1, dtype=int)
    perp_mat = np.full((n, n_ax), np.inf, dtype=float)
    for k, ax in enumerate(axes):
        path, L = _resolve_seed_path(
            ax["start_ras"], ax["end_ras"], ax.get("path_ras"),
        )
        if L < 1e-3:
            continue
        arc, perp = _project_to_polyline(pts, path)
        in_range = (
            (arc >= -float(along_axis_lo_pad_mm))
            & (arc <= L + float(max_extend_tip_mm))
        )
        perp_mat[:, k] = np.where(in_range, perp, np.inf)
    oi = np.argmin(perp_mat, axis=1)
    od = perp_mat[np.arange(n), oi]
    owners = np.where(od > max_perp_mm, -1, oi).astype(int)
    return owners


# ---------------------------------------------------------------------
# Per-axis placement
# ---------------------------------------------------------------------


def _failed_result(name: str, reason: str) -> ContactPlacementResult:
    return ContactPlacementResult(
        name=name,
        success=False,
        model_id="",
        positions_ras=[],
        placement_kind="failed",
        n_iter=0,
        score=0.0,
        sign=0.0,
        tip_arc_mm=0.0,
        eff_entry_arc_mm=None,
        n_inliers=0,
        rejected_reason=reason,
        diagnostics={},
    )


def _filter_models(library_models, model_id):
    """Restrict candidate library to the assigned model when given."""
    if model_id:
        m = [m for m in library_models if str(m.get("id")) == str(model_id)]
        return m or list(library_models)
    return list(library_models)


def place_contacts_for_axis(
    start_ras,
    end_ras,
    *,
    features: dict,
    library_models: list[dict],
    path_ras: list | np.ndarray | None = None,
    model_id: str | None = None,
    owner_mask: np.ndarray | None = None,
    name: str = "",
    config: ContactPlacementConfig | None = None,
) -> ContactPlacementResult:
    """Place contacts on one trajectory axis.

    Args:
        start_ras, end_ras: seed axis endpoints in RAS (mm).
        features: dict from ``rosa_detect.guided_fit_engine.compute_features``.
            Required keys: ``ct_arr_kji``, ``ras_to_ijk_mat``,
            ``blob_pts_ras``, ``blob_amps``.
        library_models: list of electrode-model dicts to consider.
        path_ras: optional curved-seed polyline (K, 3) with K>=3. When
            given, arc-length is computed along the polyline.
        model_id: when set, restricts RANSAC to this single model.
        owner_mask: optional boolean array over ``blob_pts_ras``;
            True where the blob is assigned to this trajectory.
        config: ``ContactPlacementConfig``; defaults if None.

    Returns:
        ``ContactPlacementResult``.
    """
    cfg = config or ContactPlacementConfig()
    name = str(name or "")

    from .contact_peak_fit import sample_axis_profile
    from .electrode_classifier import signal_derived_entry_arc

    ct_arr = np.asarray(features["ct_arr_kji"], dtype=np.float32)
    r2i = np.asarray(features["ras_to_ijk_mat"], dtype=float)
    blobs = np.asarray(features["blob_pts_ras"], dtype=float)
    if blobs.size == 0:
        return _failed_result(name, "no blobs in volume")

    path, axis_len_mm = _resolve_seed_path(start_ras, end_ras, path_ras)
    if axis_len_mm < 1e-3:
        return _failed_result(name, "seed has zero length")

    # Profile sampling: use the polyline endpoints as the profile
    # interval. Curved seeds → profile sampled along the straight
    # line between start and end of the polyline (the bolt-detection
    # heuristic only needs the entry-region HU response, which lives
    # near the proximal end and is approximately straight in that
    # region). Curved-tip refinement happens in the bent loop.
    s_pad = path[0] - cfg.profile_pad_bolt_mm * (
        (path[1] - path[0])
        / max(1e-9, float(np.linalg.norm(path[1] - path[0])))
    )
    e_pad = path[-1] + cfg.profile_pad_tip_mm * (
        (path[-1] - path[-2])
        / max(1e-9, float(np.linalg.norm(path[-1] - path[-2])))
    )
    arc_p, prof_p = sample_axis_profile(
        ct_arr, r2i, s_pad, e_pad,
        step_mm=cfg.profile_step_mm,
        disk_radius_mm=cfg.profile_disk_radius_mm,
        n_radii=cfg.profile_n_radii,
        n_angles=cfg.profile_n_angles,
        reducer="max",
    )
    entry_arc = signal_derived_entry_arc(arc_p, prof_p)
    profile_end_arc = float(arc_p[-1]) if len(arc_p) else axis_len_mm

    # Project blobs to the seed polyline → (arc, perp).
    arc_pj, perp_d = _project_to_polyline(blobs, path)

    # HU at each blob — used for the corridor's metal gate. Sample
    # trilinearly; finite-mask filters out-of-volume points.
    from .volume_sampling import sample_trilinear_at_ras
    hu_at = np.array(
        [float(sample_trilinear_at_ras(ct_arr, r2i, p)) for p in blobs],
        dtype=float,
    )

    in_corr = (
        (perp_d < cfg.corridor_radius_mm)
        & (arc_pj > -cfg.along_axis_lo_pad_mm)
        & (arc_pj < axis_len_mm + cfg.ransac_max_extend_tip_mm)
        & np.isfinite(hu_at)
        & (hu_at >= cfg.peak_hu_min)
    )
    if owner_mask is not None:
        owner_mask = np.asarray(owner_mask, dtype=bool)
        if owner_mask.shape[0] != blobs.shape[0]:
            return _failed_result(name, "owner_mask shape mismatch")
        in_corr = in_corr & owner_mask

    # Bolt / contact partition.
    candidate_models = _filter_models(library_models, model_id)
    in_corr_idx = np.where(in_corr)[0]
    arcs_in_corr = arc_pj[in_corr_idx]
    head_distance_arr = features.get("head_distance")
    head_distance_at_corr_blobs: np.ndarray | None = None
    if head_distance_arr is not None and in_corr_idx.size > 0:
        head_distance_at_corr_blobs = np.array(
            [
                float(sample_trilinear_at_ras(
                    head_distance_arr, r2i, blobs[bi]
                ))
                for bi in in_corr_idx
            ],
            dtype=float,
        )
    if cfg.partition_strategy == "cascade":
        eff_entry, partition_tier = _partition_bolt_from_contacts(
            arcs_in_corr,
            entry_arc=entry_arc,
            head_distance_at_blob=head_distance_at_corr_blobs,
            library_models=candidate_models,
            config=cfg,
            # Tier 1A inputs: features + axis endpoints (in straight-
            # axis frame, same frame the partition's arcs live in)
            # so estimate_bolt_end_from_metal_mass can sample HU mass
            # along the trajectory and detect the bolt-end.
            features=features,
            bolt_outer_ras=path[0],
            deep_end_ras=path[-1],
        )
    else:
        eff_entry = _partition_bolt_contacts_by_arc(
            arcs_in_corr, entry_arc,
            gap_thresh_mm=cfg.partition_gap_thresh_mm,
            min_contact_side=cfg.partition_min_contact_side,
        )
        partition_tier = "gap_only"
    if eff_entry is not None:
        cont_keep = in_corr & (
            arc_pj >= float(eff_entry) - cfg.partition_keep_buffer_mm
        )
    else:
        cont_keep = in_corr

    cp = blobs[cont_keep]
    carcs = arc_pj[cont_keep]
    o = np.argsort(carcs)
    cp = cp[o]
    carcs = carcs[o]
    # Bolt-side blobs (in corridor but before the partition cut) — kept
    # in diagnostics for QC plotters that visualize bolt vs. contact
    # arrangement.
    if eff_entry is not None:
        bolt_keep = in_corr & (
            arc_pj < float(eff_entry) - cfg.partition_keep_buffer_mm
        )
    else:
        bolt_keep = np.zeros_like(in_corr)
    bp = blobs[bolt_keep]
    barcs = arc_pj[bolt_keep]
    ob = np.argsort(barcs)
    bp = bp[ob]
    barcs = barcs[ob]

    diagnostics = {
        "profile_arc_mm": np.asarray(arc_p, dtype=float),
        "profile_values": np.asarray(prof_p, dtype=float),
        "axis_len_mm": float(axis_len_mm),
        "entry_arc_mm": (None if entry_arc is None else float(entry_arc)),
        "partition_tier": partition_tier,
        "n_corridor_blobs": int(in_corr.sum()),
        "n_contact_side_blobs": int(cont_keep.sum()),
        # Per-axis intermediates for QC. ``corridor_blobs_ras`` and
        # ``corridor_arcs_mm`` are the contact-side blobs sorted by
        # arc; ``bolt_blobs_ras`` / ``bolt_arcs_mm`` are the bolt-side
        # ones. Numpy arrays — diagnostics is intentionally NOT
        # JSON-safe; consumers who need JSON cast as appropriate.
        "corridor_blobs_ras": cp,
        "corridor_arcs_mm": carcs,
        "bolt_blobs_ras": bp,
        "bolt_arcs_mm": barcs,
    }

    if len(carcs) < 2:
        result = _failed_result(name, "too few contact-side blobs")
        result.eff_entry_arc_mm = (
            None if eff_entry is None else float(eff_entry)
        )
        result.diagnostics = diagnostics
        return result

    # Bent-loop RANSAC: pick → fit curve → re-arc-length → repeat.
    arcs_l = carcs.copy()
    rb: dict | None = None
    inl: list[int] = []
    sp_pos = None
    sp_ctrl = None
    pset: frozenset[int] = frozenset()
    n_iter = 0
    for it in range(cfg.max_bent_iterations):
        n_iter = it + 1
        rb = _ransac_pick_library(
            arcs_l, candidate_models,
            entry_arc=entry_arc,
            profile_end_arc=profile_end_arc + cfg.ransac_max_extend_tip_mm,
            pair_tol_mm=cfg.ransac_pair_tol_mm,
            inlier_tol_mm=cfg.ransac_inlier_tol_mm,
            max_extend_tip_mm=cfg.ransac_max_extend_tip_mm,
            bolt_free_pass=cfg.ransac_bolt_free_pass,
            phantom_penalty=cfg.ransac_phantom_penalty,
        )
        inl = _identify_inliers(
            arcs_l, rb, candidate_models, cfg.ransac_inlier_tol_mm,
        )
        ins = frozenset(inl)
        if ins == pset:
            break
        pset = ins
        if len(inl) < 4:
            break
        s_, t_, ctrl_ = _curve_fit_dispatch(
            cp[inl], carcs[inl],
            kind=cfg.curve_fit, poly_deg=cfg.poly_deg,
        )
        if s_ is None:
            break
        sp_pos = s_
        sp_ctrl = ctrl_
        arcs_l = t_(carcs)

    if rb is None:
        result = _failed_result(name, "no library model matched")
        result.eff_entry_arc_mm = (
            None if eff_entry is None else float(eff_entry)
        )
        result.n_iter = n_iter
        result.diagnostics = diagnostics
        return result

    model = next(
        m for m in candidate_models if str(m["id"]) == rb["model_id"]
    )
    offsets = np.asarray(
        model["contact_center_offsets_from_tip_mm"], dtype=float,
    )
    tip_s = float(rb["tip_arc"])
    sign = float(rb["sign"])

    if sp_pos is not None and sp_ctrl is not None:
        ts_c, _, t_d, s_d = sp_ctrl
        out = []
        for off in offsets:
            tau = float(np.interp(tip_s + sign * off, s_d, t_d))
            out.append(sp_pos(np.array([tau]))[0].tolist())
        positions = out
        kind: Literal["polynomial", "spline", "straight", "failed"] = (
            "polynomial" if cfg.curve_fit == "polynomial" else "spline"
        )
    else:
        # Straight-axis fallback: walk the seed polyline at arc tip+off.
        out = []
        total = float(axis_len_mm)
        for off in offsets:
            arc = tip_s + sign * off
            arc = max(min(arc, total + cfg.ransac_max_extend_tip_mm),
                      -cfg.along_axis_lo_pad_mm)
            out.append(_polyline_axis_at(path, arc).tolist())
        positions = out
        kind = "straight"

    return ContactPlacementResult(
        name=name,
        success=True,
        model_id=str(rb["model_id"]),
        positions_ras=positions,
        placement_kind=kind,
        n_iter=n_iter,
        score=float(rb["score"]),
        sign=sign,
        tip_arc_mm=tip_s,
        eff_entry_arc_mm=(
            None if eff_entry is None else float(eff_entry)
        ),
        n_inliers=int(rb["inl"]),
        rejected_reason="",
        diagnostics={
            **diagnostics,
            "phantom_interior": int(rb.get("phantom_interior", 0)),
            "phantom_bolt_side": int(rb.get("phantom_bolt_side", 0)),
            "phantom_past_end": int(rb.get("phantom_past_end", 0)),
            "inlier_indices_in_corridor": list(inl),
            "curve_position_fn": sp_pos,            # callable, may be None
            "curve_arc_param": sp_ctrl,             # (ts, pts, t_d, s_d) or None
        },
    )


# ---------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------


def _resolve_library_models(
    library_models: list[dict] | None,
    library_strategy: str | None,
):
    """Return the candidate library list for the batch.

    Priority: explicit ``library_models`` > strategy filter > full
    library. Strategy filtering uses
    ``electrode_classifier.filter_models_for_strategy`` so vendor
    families collapse the same way Auto Fit does.
    """
    if library_models is not None and len(library_models):
        return list(library_models)
    from .electrode_models import load_electrode_library
    library = load_electrode_library()
    models = list(library["models"])
    if library_strategy:
        from .electrode_classifier import filter_models_for_strategy
        models = filter_models_for_strategy(models, library_strategy) or models
    return models


def place_contacts_for_trajectories(
    img,
    ijk_to_ras_mat,
    ras_to_ijk_mat,
    *,
    trajectories: list[dict],
    library_strategy: str | None = None,
    library_models: list[dict] | None = None,
    config: ContactPlacementConfig | None = None,
    features: dict | None = None,
) -> ContactPlacementBatch:
    """Batch placement: one CT volume + many trajectory seeds → contacts.

    Args:
        img: SimpleITK image (raw CT).
        ijk_to_ras_mat: 4x4 IJK→RAS matrix.
        ras_to_ijk_mat: 4x4 RAS→IJK matrix.
        trajectories: list of dicts. Each MUST carry ``start_ras`` and
            ``end_ras``. Optional keys: ``name``, ``electrode_model``
            (mode-5 restrict), ``path_ras`` (curved seed polyline).
        library_strategy: ``"dixi"``, ``"pmt_35"``, etc. None = full library.
        library_models: explicit override; takes precedence over strategy.
        config: ``ContactPlacementConfig``; defaults if None.
        features: precomputed features dict (from
            ``guided_fit_engine.compute_features``). Skips one
            preprocessing pass when caller already has it (Slicer
            reuses Auto Fit's feature volumes).

    Returns:
        ``ContactPlacementBatch`` with one ``ContactPlacementResult``
        per input trajectory (in input order). Trajectories that fail
        placement carry ``success=False`` with ``rejected_reason``.
    """
    cfg = config or ContactPlacementConfig()

    if features is None:
        from rosa_detect.guided_fit_engine import compute_features
        features = compute_features(img, ijk_to_ras_mat, ras_to_ijk_mat)

    candidate_models = _resolve_library_models(library_models, library_strategy)

    # Subject-scope owner assignment so cross-shank blobs don't double-
    # count on neighboring axes.
    blobs = np.asarray(features["blob_pts_ras"], dtype=float)
    axes = [
        {
            "start_ras": t["start_ras"],
            "end_ras": t["end_ras"],
            "path_ras": t.get("path_ras"),
        }
        for t in trajectories
    ]
    owners = assign_axis_owners(
        blobs, axes,
        max_perp_mm=cfg.ownership_max_perp_mm,
        max_extend_tip_mm=cfg.ransac_max_extend_tip_mm,
        along_axis_lo_pad_mm=cfg.along_axis_lo_pad_mm,
    )

    results: list[ContactPlacementResult] = []
    for k, traj in enumerate(trajectories):
        owner_mask = (owners == k)
        result = place_contacts_for_axis(
            traj["start_ras"], traj["end_ras"],
            features=features,
            library_models=candidate_models,
            path_ras=traj.get("path_ras"),
            model_id=traj.get("electrode_model"),
            owner_mask=owner_mask,
            name=str(traj.get("name") or f"traj_{k}"),
            config=cfg,
        )
        results.append(result)

    return ContactPlacementBatch(
        results=results,
        diagnostics={
            "n_trajectories": len(trajectories),
            "n_blobs": int(blobs.shape[0]),
            "n_owned_blobs": int((owners >= 0).sum()),
            "library_strategy": library_strategy,
            "n_candidate_models": len(candidate_models),
        },
    )


# ---------------------------------------------------------------------
# Bolt-end estimation from total metal mass
# ---------------------------------------------------------------------
#
# Independent of the bent-RANSAC placement pipeline above. These are
# the validated building blocks for partition Tier 1 (bolt-end /
# entry-arc detection):
#
#   centerline = polynomial fit through brightness-mass centroids of
#                perpendicular disks along the trajectory
#                (refine_axis_via_centroid)
#   profile    = total HU mass per perpendicular disk along the
#                centerline AND straight axis; element-wise max gives
#                the bolt-mass signal regardless of which line is
#                closer to the bolt
#                (sample_disk_along_polyline)
#   bolt-end   = walk forward from peak; first sustained drop below
#                plateau_frac × peak after Gaussian smoothing with
#                σ = library median pitch (envelope filter that
#                cancels shaft contact oscillations)
#                (entry_arc_from_metal_mass)
#
# Validated in ``notebooks/bolt_partition_qc.ipynb`` on AMC88 + T22.
# Headline: T22 traj_1 (a multi-peak bolt where the legacy
# signal_derived_entry_arc misfires) lands within 1 mm of the true
# bolt-end with this approach.


def median_library_pitch_mm(library_models) -> float | None:
    """Median spacing between consecutive contacts across the strategy's
    models. Used as the optimal envelope-smoothing σ in
    ``entry_arc_from_metal_mass``: a Gaussian of σ = pitch averages
    over exactly one full contact-oscillation period, cancelling each
    shaft contact peak with its adjacent valley while leaving the
    bolt's sustained-high mass untouched.

    Returns None when no library models are provided (caller can fall
    back to a fixed default).
    """
    pitches: list[float] = []
    for m in (library_models or []):
        offsets = np.asarray(
            m.get("contact_center_offsets_from_tip_mm") or [],
            dtype=float,
        )
        if offsets.size >= 2:
            sorted_offsets = np.sort(offsets)
            pitches.extend(np.diff(sorted_offsets).tolist())
    if not pitches:
        return None
    return float(np.median(pitches))


def _brightness(volume_values, polarity: str):
    """Convert raw volume values to "brightness" (positive at metal).

    HU: brightness = HU value (positive for metal).
    LoG sigma=1: brightness = -LoG (LoG is NEGATIVE at metal-bright spots).
    """
    if polarity == "positive":
        return volume_values
    if polarity == "negative":
        return -np.asarray(volume_values, dtype=float)
    raise ValueError(f"unknown polarity {polarity!r}")


def refine_axis_via_centroid(
    volume_arr,
    ras_to_ijk_mat,
    start_ras,
    end_ras,
    *,
    polarity: str = "positive",
    arc_step_mm: float = 0.25,
    cross_radius_mm: float = 6.0,
    cross_pixel_mm: float = 0.5,
    filter_value: float = 500.0,
    weight_offset: float = 1500.0,
    poly_deg: int = 2,
):
    """Polynomial-fit centerline through brightness-mass-weighted
    centroids of perpendicular disks along the straight axis.

    For each arc bin along the straight axis, the per-bin centroid is
    the brightness-weighted center of mass within a radius-
    ``cross_radius_mm`` perpendicular disk. We fit two polynomials —
    du(arc) and dv(arc) of degree ``poly_deg`` — through the valid
    centroid track in the straight-axis perpendicular basis, then
    sample densely.

    Defaults are tuned for HU (positive polarity, filter HU ≥ 500,
    weight = max(0, HU − 1500)). For LoG sigma=1 pass
    ``polarity="negative"`` with ``filter_value=100, weight_offset=200``.

    Returns the refined polyline as Kx3 RAS, or None when fewer than
    ``poly_deg + 1`` cross-sections contain enough metal.
    """
    import numpy as np

    s = np.asarray(start_ras, dtype=float)
    e = np.asarray(end_ras, dtype=float)
    L = float(np.linalg.norm(e - s))
    if L < 5.0:
        return None
    axis = (e - s) / L
    any_v = (
        np.array([1.0, 0, 0])
        if abs(axis[0]) <= 0.9
        else np.array([0, 1.0, 0])
    )
    u_perp = np.cross(axis, any_v); u_perp /= np.linalg.norm(u_perp)
    v_perp = np.cross(axis, u_perp); v_perp /= np.linalg.norm(v_perp)

    grid_axis = np.arange(
        -cross_radius_mm, cross_radius_mm + cross_pixel_mm, cross_pixel_mm,
    )
    DU, DV = np.meshgrid(grid_axis, grid_axis)
    in_disk = DU ** 2 + DV ** 2 <= cross_radius_mm ** 2
    DU = DU[in_disk]; DV = DV[in_disk]

    arcs = np.arange(0, L + 0.001, arc_step_mm)
    n = len(arcs)
    du_track = np.full(n, np.nan, dtype=float)
    dv_track = np.full(n, np.nan, dtype=float)

    from .volume_sampling import sample_trilinear_batch

    # Pre-compute the disk offsets in 3D as a (M, 3) template that we
    # reuse per arc by adding the arc center.
    M = len(DU)
    disk_offsets = (DU[:, None] * u_perp[None, :]
                    + DV[:, None] * v_perp[None, :])  # (M, 3)

    for i, t in enumerate(arcs):
        center = s + t * axis
        pts = center[None, :] + disk_offsets   # (M, 3)
        raw = sample_trilinear_batch(volume_arr, ras_to_ijk_mat, pts)
        bright = _brightness(raw, polarity)
        mask = np.isfinite(bright) & (bright >= filter_value)
        if int(mask.sum()) < 3:
            continue
        w = np.maximum(bright[mask] - weight_offset, 0.0)
        if w.sum() < 1e-6:
            continue
        du_track[i] = float(np.sum(DU[mask] * w) / np.sum(w))
        dv_track[i] = float(np.sum(DV[mask] * w) / np.sum(w))

    valid = np.isfinite(du_track) & np.isfinite(dv_track)
    if int(valid.sum()) < poly_deg + 1:
        return None

    cu = np.polyfit(arcs[valid], du_track[valid], poly_deg)
    cv = np.polyfit(arcs[valid], dv_track[valid], poly_deg)
    du_smooth = np.polyval(cu, arcs)
    dv_smooth = np.polyval(cv, arcs)

    polyline = np.zeros((n, 3), dtype=float)
    for i, t in enumerate(arcs):
        center = s + t * axis
        polyline[i] = (
            center + du_smooth[i] * u_perp + dv_smooth[i] * v_perp
        )
    return polyline


def sample_disk_along_polyline(
    volume_arr,
    ras_to_ijk_mat,
    polyline,
    *,
    polarity: str = "positive",
    step_mm: float = 0.25,
    disk_radius_mm: float = 1.5,
    n_radii: int = 2,
    n_angles: int = 8,
    total_threshold: float = 1500.0,
):
    """Walk a polyline; per arc bin emit (arc, max_brightness, total_brightness).

    ``max_brightness``: max brightness in the perpendicular disk (the
        same signal ``signal_derived_entry_arc`` consumes for max-disk
        profiles).
    ``total_brightness``: Σ max(0, brightness − total_threshold) over
        disk voxels — the integrated metal mass per disk. Sustained-
        high through bolt + bone (HU) or bolt + wire + contacts (LoG);
        the sharp drop where the integral crosses bolt-end is what
        ``entry_arc_from_metal_mass`` keys on.

    For a STRAIGHT polyline (2-point input ``[start, end]``), this
    walks perpendicular to a fixed axis. For a curved polyline it
    walks perpendicular to the local tangent, allowing curved seeds.
    """
    import numpy as np

    starts, dirs, lens, cum_start = _polyline_segments(polyline)
    total = float(cum_start[-1] + lens[-1])
    arcs = np.arange(0, total + 0.5 * step_mm, step_mm)
    max_b = np.zeros(len(arcs), dtype=float)
    total_b = np.zeros(len(arcs), dtype=float)

    from .volume_sampling import sample_trilinear_batch

    # Pre-compute disk-offset templates (in u-v perpendicular basis).
    # Each per-arc offset is offset_u * u + offset_v * v in 3D.
    n_per_disk = 1 + n_radii * n_angles
    offset_u = np.zeros(n_per_disk, dtype=float)
    offset_v = np.zeros(n_per_disk, dtype=float)
    idx = 1
    for r_idx in range(1, n_radii + 1):
        rr = disk_radius_mm * r_idx / n_radii
        for a_idx in range(n_angles):
            ang = 2 * np.pi * a_idx / n_angles
            offset_u[idx] = rr * np.cos(ang)
            offset_v[idx] = rr * np.sin(ang)
            idx += 1

    for ai, t in enumerate(arcs):
        center, tangent = _polyline_at_arc_with_tangent(
            starts, dirs, lens, cum_start, float(t),
        )
        any_v = (
            np.array([1.0, 0, 0])
            if abs(tangent[0]) <= 0.9
            else np.array([0, 1.0, 0])
        )
        u = np.cross(tangent, any_v); u /= np.linalg.norm(u)
        v = np.cross(tangent, u);     v /= np.linalg.norm(v)
        # Build (n_per_disk, 3) sample-point batch in one numpy expression.
        pts = (center[None, :]
               + offset_u[:, None] * u[None, :]
               + offset_v[:, None] * v[None, :])
        samples = sample_trilinear_batch(volume_arr, ras_to_ijk_mat, pts)
        s_arr = _brightness(samples, polarity)
        finite = s_arr[np.isfinite(s_arr)]
        max_b[ai] = float(finite.max()) if finite.size else 0.0
        total_b[ai] = float(np.maximum(finite - total_threshold, 0.0).sum())
    return arcs, max_b, total_b


def _polyline_at_arc_with_tangent(starts, dirs, lens, cum_start, arc):
    """Position + tangent at arc-length on the polyline. Helper for
    sample_disk_along_polyline (which needs both)."""
    if arc <= 0:
        return starts[0] + arc * dirs[0], dirs[0]
    total = cum_start[-1] + lens[-1]
    if arc >= total:
        return starts[-1] + (arc - cum_start[-1]) * dirs[-1], dirs[-1]
    idx = int(np.searchsorted(cum_start + lens, arc, side="right"))
    idx = min(idx, len(starts) - 1)
    return starts[idx] + (arc - cum_start[idx]) * dirs[idx], dirs[idx]


def entry_arc_from_metal_mass(
    arcs,
    mass,
    *,
    smooth_sigma_mm: float = 3.5,
    plateau_frac: float = 0.50,
    padding_mm: float = 0.5,
    max_gap_mm: float = 6.0,
    threshold_mode: str = "plateau",
    abs_threshold: float = 500.0,
):
    """Detect bolt-end / entry arc from a metal-mass profile.

    Algorithm:
      1. Gaussian-smooth the mass profile with σ = ``smooth_sigma_mm``.
         When σ matches the library's median contact pitch, this
         averages each shaft contact peak with its adjacent valley,
         pulling the shaft envelope down to ~50% of the bolt's
         sustained-high mass.
      2. Find the global max of the smoothed profile (the bolt's
         bone-collar peak).
      3. Threshold:
            "plateau":  threshold = plateau_frac × peak
            "absolute": threshold = abs_threshold (peak-independent)
      4. Walk forward from the peak. Track the FIRST arc where mass
         drops below threshold. If mass stays below threshold for
         ``max_gap_mm`` of arc, the remembered drop is the bolt-end.
         If mass climbs back above threshold within ``max_gap_mm``,
         the dip was an internal bolt feature (e.g., nut-to-bolt-body
         gap); reset and continue.

    Returns the entry arc (float, mm), or None when no plateau
    detected (mass profile flat, peak below threshold, etc).
    """
    import numpy as np

    arcs = np.asarray(arcs, dtype=float)
    mass = np.asarray(mass, dtype=float)
    if len(arcs) < 5:
        return None
    finite = np.isfinite(mass)
    if not finite.any():
        return None
    step_mm = float(arcs[1] - arcs[0]) if len(arcs) >= 2 else 0.25

    if smooth_sigma_mm > 0:
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(
            np.where(finite, mass, 0.0),
            sigma=max(0.5, smooth_sigma_mm / step_mm),
            mode="nearest",
        )
    else:
        smoothed = np.where(finite, mass, 0.0)

    peak_val = float(smoothed.max())
    if peak_val < 1e-3:
        return None

    if threshold_mode == "absolute":
        threshold = float(abs_threshold)
        if peak_val <= threshold:
            return None
    elif threshold_mode == "plateau":
        threshold = float(plateau_frac) * peak_val
    else:
        raise ValueError(f"unknown threshold_mode {threshold_mode!r}")

    peak_idx = int(np.argmax(smoothed))
    max_below_bins = max(1, int(round(max_gap_mm / step_mm)))

    last_drop = None
    bins_below = 0
    for i in range(peak_idx, len(smoothed)):
        if smoothed[i] >= threshold:
            bins_below = 0
            last_drop = None
        else:
            if last_drop is None:
                last_drop = float(arcs[i])
            bins_below += 1
            if bins_below > max_below_bins:
                return last_drop + float(padding_mm)
    if last_drop is not None:
        return last_drop + float(padding_mm)
    return None


def estimate_bolt_end_from_metal_mass(
    bolt_outer_ras,
    deep_end_ras,
    *,
    features: dict,
    library_models=None,
    cross_radius_mm: float = 6.0,
    signal_disk_radius_mm: float = 1.5,
    plateau_frac: float = 0.50,
    max_gap_mm: float = 6.0,
    padding_mm: float = 0.5,
    threshold_mode: str = "plateau",
    abs_threshold: float = 500.0,
    smooth_sigma_fallback_mm: float = 3.5,
    poly_deg: int = 2,
) -> dict:
    """High-level: bolt-end arc from total HU mass along the trajectory.

    Pipeline (per ``bolt_partition_qc.ipynb``):
      1. Polynomial centerline via ``refine_axis_via_centroid`` on the
         volume's LoG (when available, since LoG sees the wire that
         connects bolt to contacts) or HU otherwise.
      2. Total HU mass per perpendicular disk sampled along BOTH the
         straight axis (``[bolt_outer_ras, deep_end_ras]``) and the
         polynomial centerline. Element-wise max combines the two —
         picks up the bolt regardless of which line is closer to it.
      3. Smooth with σ = library median pitch (or
         ``smooth_sigma_fallback_mm`` if no library) and detect the
         bolt-end via ``entry_arc_from_metal_mass``.

    Args:
        bolt_outer_ras: shallow end of the trajectory axis (RAS, mm).
            The bolt's outer-most tube voxel is a good choice.
        deep_end_ras: deep end. The bolt-side end of the search
            interval — usually the trajectory's deep tip + small pad.
        features: dict from ``rosa_detect.guided_fit_engine.compute_features``.
            Must contain ``ct_arr_kji``, ``ras_to_ijk_mat``; will use
            ``log`` if available for centerline estimation.
        library_models: list of electrode-model dicts; provides the
            median pitch for envelope smoothing. When None, falls back
            to ``smooth_sigma_fallback_mm``.

    Returns:
        dict with:
          "bolt_end_arc_mm": detected arc (mm from bolt_outer_ras) or
              None if not detected.
          "centerline": refined polyline (Kx3 RAS) or None.
          "diagnostics": {
              "arcs_mm", "total_mass_combined", "total_mass_straight",
              "total_mass_polyline", "smooth_sigma_mm", "threshold",
              "peak_mass", "peak_arc_mm",
          }
    """
    import numpy as np

    ct_arr = np.asarray(features["ct_arr_kji"], dtype=np.float32)
    r2i = np.asarray(features["ras_to_ijk_mat"], dtype=float)
    log_arr = features.get("log")

    s = np.asarray(bolt_outer_ras, dtype=float)
    e = np.asarray(deep_end_ras, dtype=float)

    # Centerline. Prefer LoG (sees the wire) when available; HU as
    # fallback.
    if log_arr is not None:
        centerline = refine_axis_via_centroid(
            log_arr, r2i, s, e,
            polarity="negative",
            cross_radius_mm=cross_radius_mm,
            filter_value=100.0, weight_offset=200.0,
            poly_deg=poly_deg,
        )
    else:
        centerline = refine_axis_via_centroid(
            ct_arr, r2i, s, e,
            polarity="positive",
            cross_radius_mm=cross_radius_mm,
            filter_value=500.0, weight_offset=1500.0,
            poly_deg=poly_deg,
        )

    # Total HU mass on STRAIGHT axis.
    straight_polyline = np.stack([s, e])
    arc_s, _max_s, total_s = sample_disk_along_polyline(
        ct_arr, r2i, straight_polyline,
        polarity="positive",
        disk_radius_mm=signal_disk_radius_mm,
        total_threshold=1500.0,
    )

    # Total HU mass on CENTERLINE (when available).
    if centerline is not None:
        arc_r, _max_r, total_r = sample_disk_along_polyline(
            ct_arr, r2i, centerline,
            polarity="positive",
            disk_radius_mm=signal_disk_radius_mm,
            total_threshold=1500.0,
        )
        # Re-grid polyline mass onto straight-axis u-coordinate.
        u_r = _polyline_arc_to_u(centerline, arc_r, s, (e - s) / np.linalg.norm(e - s))
        total_polyline_on_u = np.interp(arc_s, u_r, total_r, left=0.0, right=0.0)
        total_combined = np.maximum(total_s, total_polyline_on_u)
    else:
        total_polyline_on_u = np.zeros_like(arc_s)
        total_combined = total_s

    # σ = library median pitch (envelope filter cancelling shaft
    # oscillations); fallback to fixed value if no library.
    pitch_mm = median_library_pitch_mm(library_models)
    sigma_mm = pitch_mm if pitch_mm else smooth_sigma_fallback_mm

    bolt_end_arc = entry_arc_from_metal_mass(
        arc_s, total_combined,
        smooth_sigma_mm=sigma_mm,
        plateau_frac=plateau_frac,
        padding_mm=padding_mm,
        max_gap_mm=max_gap_mm,
        threshold_mode=threshold_mode,
        abs_threshold=abs_threshold,
    )

    peak_idx = int(np.argmax(total_combined)) if total_combined.size else 0
    peak_mass = float(total_combined[peak_idx]) if total_combined.size else 0.0
    peak_arc = float(arc_s[peak_idx]) if total_combined.size else 0.0
    threshold_used = (
        plateau_frac * peak_mass if threshold_mode == "plateau" else abs_threshold
    )
    return {
        "bolt_end_arc_mm": bolt_end_arc,
        "centerline": centerline,
        "diagnostics": {
            "arcs_mm": arc_s,
            "total_mass_combined": total_combined,
            "total_mass_straight": total_s,
            "total_mass_polyline": total_polyline_on_u,
            "smooth_sigma_mm": sigma_mm,
            "threshold": threshold_used,
            "peak_mass": peak_mass,
            "peak_arc_mm": peak_arc,
        },
    }


def _polyline_arc_to_u(polyline, arcs, slab_origin_ras, slab_axis):
    """Convert polyline arc-length positions to slab u-coordinate
    (projection onto the straight axis from slab_origin_ras).
    """
    starts, dirs, lens, cum_start = _polyline_segments(polyline)
    out = np.zeros(len(arcs), dtype=float)
    for i, t in enumerate(arcs):
        if t <= 0:
            pos = starts[0] + t * dirs[0]
        elif t >= cum_start[-1] + lens[-1]:
            pos = starts[-1] + (t - cum_start[-1]) * dirs[-1]
        else:
            idx = int(np.searchsorted(cum_start + lens, t, side="right"))
            idx = min(idx, len(starts) - 1)
            pos = starts[idx] + (t - cum_start[idx]) * dirs[idx]
        out[i] = float(
            (pos - np.asarray(slab_origin_ras, float))
            @ np.asarray(slab_axis, float)
        )
    return out


__all__ = [
    "ContactPlacementConfig",
    "ContactPlacementResult",
    "ContactPlacementBatch",
    "place_contacts_for_axis",
    "place_contacts_for_trajectories",
    "assign_axis_owners",
    # Bolt-end estimation building blocks (notebook → production).
    "median_library_pitch_mm",
    "refine_axis_via_centroid",
    "sample_disk_along_polyline",
    "entry_arc_from_metal_mass",
    "estimate_bolt_end_from_metal_mass",
]
