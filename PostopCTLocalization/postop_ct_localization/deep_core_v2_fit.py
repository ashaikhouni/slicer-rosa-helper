"""Bolt-first trajectory emission for deep_core_v2.

Two independent fit paths per bolt candidate — the pipeline calls
``fit_bolt_trajectory`` which dispatches based on
``model_fit.v2_fit_mode``:

- ``"two_threshold"`` (default): walks the axis inward from the bolt
  center through a **loose** metal mask (``mask.contact_probe_hu``,
  default 1000 HU) with a gap tolerance, stopping at the deepest
  sample that's still contiguous with the bolt. Validates
  intracranial span against the electrode library.

- ``"intensity_peaks"``: builds a 1D max-HU profile along the axis
  using a 5 mm-diameter disc perpendicular to the axis, detects peaks
  (local maxima above a fraction of the profile max), and fits each
  library electrode's known contact-spacing pattern to the observed
  peak positions. Best-scoring model determines the deep tip and the
  best_model_id. No thresholding of individual voxels is needed — the
  fit uses the relative peak structure.

Both paths take the bolt's RANSAC axis and center as ground truth (no
PCA refit) and anchor the shallow endpoint to the bolt position via
``model_fit.bolt_endpoint_offset_mm``.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from . import deep_core_axis_reconstruction as axr


# Fixed axial walk extent (mm from bolt center, + is scalp, - is deep).
BOLT_INWARD_WALK_MM = 110.0
BOLT_OUTWARD_WALK_MM = 5.0


# ---------------------------------------------------------------------------
# Axis orientation
# ---------------------------------------------------------------------------

def _orient_bolt_axis_toward_scalp(
    axis_ras: np.ndarray,
    center_ras: np.ndarray,
    head_distance_map_kji: np.ndarray,
    ras_to_ijk_fn,
) -> np.ndarray:
    """Flip the axis so +axis points toward the scalp (lower head_distance)."""
    shape = head_distance_map_kji.shape

    def hd(pt):
        ijk = np.asarray(ras_to_ijk_fn(pt), dtype=float)
        kji = np.rint([ijk[2], ijk[1], ijk[0]]).astype(int)
        if any(kji[ax] < 0 or kji[ax] >= shape[ax] for ax in range(3)):
            return float("nan")
        return float(head_distance_map_kji[kji[0], kji[1], kji[2]])

    hd_plus = hd(center_ras + 20.0 * axis_ras)
    hd_minus = hd(center_ras - 20.0 * axis_ras)
    # Out-of-bounds (NaN) means outside the head → toward the scalp.
    if not np.isfinite(hd_minus) and np.isfinite(hd_plus):
        # -axis goes outside → -axis is scalp-ward → flip
        return -axis_ras
    if not np.isfinite(hd_plus) and np.isfinite(hd_minus):
        # +axis goes outside → +axis is already scalp-ward → no flip
        return axis_ras
    if np.isfinite(hd_plus) and np.isfinite(hd_minus) and hd_minus < hd_plus:
        return -axis_ras
    return axis_ras


# ---------------------------------------------------------------------------
# Lateral tube sampling helpers
# ---------------------------------------------------------------------------

def _build_disc_offsets(axis: np.ndarray, radius_mm: float, n_samples: int) -> np.ndarray:
    """Return (N, 3) ring offsets perpendicular to ``axis`` at ``radius_mm``."""
    u, v = axr._perp_frame(axis)
    angles = np.linspace(0.0, 2.0 * np.pi, max(3, n_samples), endpoint=False)
    return np.stack(
        [radius_mm * (np.cos(a) * u + np.sin(a) * v) for a in angles],
        axis=0,
    )


def _build_filled_disc_offsets(axis: np.ndarray, radius_mm: float) -> np.ndarray:
    """Return (N, 3) offsets filling a disc of ``radius_mm`` on a ~1mm grid."""
    u, v = axr._perp_frame(axis)
    step = 1.0
    coords = np.arange(-radius_mm, radius_mm + 0.5 * step, step)
    pts = []
    for du in coords:
        for dv in coords:
            if du * du + dv * dv <= radius_mm * radius_mm:
                pts.append(du * u + dv * v)
    if not pts:
        pts.append(np.zeros(3))
    return np.stack(pts, axis=0)


def _max_hu_in_disc(
    center_pt_ras: np.ndarray,
    offsets: np.ndarray,
    arr_kji: np.ndarray,
    ras_to_ijk_fn,
) -> float:
    shape = arr_kji.shape
    best = -np.inf
    for off in offsets:
        s = center_pt_ras + off
        ijk = np.asarray(ras_to_ijk_fn(s), dtype=float)
        kji = np.rint([ijk[2], ijk[1], ijk[0]]).astype(int)
        if any(kji[ax] < 0 or kji[ax] >= shape[ax] for ax in range(3)):
            continue
        hu = float(arr_kji[kji[0], kji[1], kji[2]])
        if hu > best:
            best = hu
    return best


# ---------------------------------------------------------------------------
# Approach A: two-threshold walk
# ---------------------------------------------------------------------------

def _find_deep_tip_by_contiguous_metal(
    center_ras: np.ndarray,
    axis_ras: np.ndarray,
    arr_kji: np.ndarray,
    ras_to_ijk_fn,
    *,
    contact_hu: float,
    tube_radius_mm: float,
    inward_walk_mm: float,
    step_mm: float,
    max_gap_mm: float,
) -> float | None:
    """Walk from the bolt center along ``-axis`` (deep) and return the
    deepest axial ``t`` (negative) that is still connected to the bolt
    via metal samples above ``contact_hu`` within ``tube_radius_mm``
    of the axis, under a ``max_gap_mm`` contiguity rule.

    Returns ``None`` if no metal is contiguous with the bolt.
    """
    offsets = _build_disc_offsets(axis_ras, tube_radius_mm, n_samples=12)
    n_steps = int(round(inward_walk_mm / step_mm))
    deepest_t = None
    gap_run = 0
    max_gap_steps = max(1, int(round(max_gap_mm / step_mm)))
    for i in range(n_steps + 1):
        t = -float(i) * step_mm
        pt = center_ras + t * axis_ras
        max_hu = _max_hu_in_disc(pt, offsets, arr_kji, ras_to_ijk_fn)
        if max_hu >= contact_hu:
            deepest_t = t
            gap_run = 0
        else:
            gap_run += 1
            if gap_run > max_gap_steps:
                break
    return deepest_t


def _ransac_line_fit(points, *, n_iter=2000, inlier_tol_mm=2.0, min_inliers=5,
                     rng_seed=42, prior_axis=None, max_angle_deg=15.0):
    n = points.shape[0]
    if n < 2:
        return np.array([0., 0., 1.]), points.mean(axis=0), np.ones(n, dtype=bool)
    rng = np.random.default_rng(rng_seed)
    best_n = 0
    best_mask = np.zeros(n, dtype=bool)
    best_axis = np.array([0., 0., 1.])
    cos_limit = float(np.cos(np.radians(max_angle_deg))) if prior_axis is not None else 0.0
    for _ in range(n_iter):
        i, j = rng.choice(n, size=2, replace=False)
        d = points[j] - points[i]
        L = float(np.linalg.norm(d))
        if L < 1.0:
            continue
        axis = d / L
        if prior_axis is not None and abs(float(np.dot(axis, prior_axis))) < cos_limit:
            continue
        v = points - points[i]
        proj = v @ axis
        perp = v - np.outer(proj, axis)
        dist = np.linalg.norm(perp, axis=1)
        mask = dist < inlier_tol_mm
        n_in = int(mask.sum())
        if n_in > best_n:
            best_n = n_in
            best_mask = mask
            best_axis = axis
    if best_n >= min_inliers:
        inliers = points[best_mask]
        c = inliers.mean(axis=0)
        X = inliers - c
        cov = X.T @ X / max(1, X.shape[0] - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        best_axis = eigvecs[:, int(np.argmax(eigvals))]
        n = float(np.linalg.norm(best_axis))
        if n > 1e-9:
            best_axis = best_axis / n
    center = points[best_mask].mean(axis=0) if best_n > 0 else points.mean(axis=0)
    return best_axis, center, best_mask


def _contiguous_deep_tip(proj_values, *, max_gap_mm=15.0, step_mm=1.0):
    """Given sorted (ascending, i.e. deep→shallow) projection values of
    inliers, find the deepest value still contiguous with the shallow end
    under a gap tolerance. Returns the deepest t in the contiguous run.
    """
    if len(proj_values) == 0:
        return None
    vals = np.sort(proj_values)[::-1]  # shallow-first (most positive = most scalp-ward)
    deepest = float(vals[0])
    for i in range(1, len(vals)):
        gap = float(vals[i - 1] - vals[i])
        if gap > max_gap_mm:
            break
        deepest = float(vals[i])
    return deepest


def _density_trimmed_deep_tip(proj_values, *, bin_mm=5.0, min_bin_count=3,
                               gap_tolerance=2):
    """Walk inlier-projection bins shallow-to-deep and stop when density drops.

    Bins with fewer than *min_bin_count* inliers are "sparse".  Up to
    *gap_tolerance* consecutive sparse bins are bridged; the walk stops
    once ``gap_tolerance + 1`` consecutive sparse bins are seen.

    Returns the deepest inlier projection in the dense region, or
    ``None`` if no dense region is found.
    """
    vals = np.asarray(proj_values, dtype=float)
    if vals.size == 0:
        return None

    shallow_start = 5.0          # small margin above bolt center
    deep_limit = float(vals.min()) - bin_mm
    n_bins = max(1, int(np.ceil((shallow_start - deep_limit) / bin_mm)))

    last_dense_edge = None
    consecutive_sparse = 0
    started = False

    for i in range(n_bins):
        shallow_edge = shallow_start - i * bin_mm
        deep_edge = shallow_edge - bin_mm
        count = int(np.sum((vals <= shallow_edge) & (vals > deep_edge)))

        if count >= min_bin_count:
            started = True
            last_dense_edge = deep_edge
            consecutive_sparse = 0
        elif started:
            consecutive_sparse += 1
            if consecutive_sparse > gap_tolerance:
                break

    if last_dense_edge is None:
        return None

    dense_vals = vals[vals >= last_dense_edge]
    if dense_vals.size == 0:
        return None
    return float(dense_vals.min())


def _gather_bright_cylinder_ras(
    center_ras, axis_inward, arr_kji, head_distance_map_kji, ras_to_ijk_fn,
    *, radius_mm, depth_mm, hu_floor, hd_min_mm, step_mm=1.0,
):
    """Gather RAS positions of bright, intracranial voxels in a cylinder."""
    offsets = _build_filled_disc_offsets(axis_inward, radius_mm)
    n_steps = int(round(depth_mm / step_mm))
    shape = arr_kji.shape
    seen = set()
    pts = []
    for i in range(n_steps + 1):
        t = float(i) * step_mm
        c_pt = center_ras + t * axis_inward
        for off in offsets:
            pt = c_pt + off
            ijk = np.asarray(ras_to_ijk_fn(pt), dtype=float)
            kji = (int(round(ijk[2])), int(round(ijk[1])), int(round(ijk[0])))
            if kji in seen:
                continue
            seen.add(kji)
            if not (0 <= kji[0] < shape[0] and 0 <= kji[1] < shape[1] and 0 <= kji[2] < shape[2]):
                continue
            if arr_kji[kji[0], kji[1], kji[2]] >= hu_floor and head_distance_map_kji[kji[0], kji[1], kji[2]] >= hd_min_mm:
                pts.append(pt.copy())
    return np.asarray(pts, dtype=float).reshape(-1, 3) if pts else np.zeros((0, 3))


def _fit_cylinder_ransac(
    bolt,
    fit: "axr.AxisFit",
    arr_kji: np.ndarray,
    head_distance_map_kji: np.ndarray,
    ras_to_ijk_fn,
    library_models: list[dict[str, Any]],
    library_span_bounds: tuple[float, float],
    cfg,
    *,
    ijk_kji_to_ras_fn=None,
) -> dict[str, Any] | None:
    """Exact port of probe_cylinder_fit: gather bright intracranial voxels
    in a wide cylinder, RANSAC a line constrained to the bolt axis, use
    inlier extent as trajectory endpoints."""
    if ijk_kji_to_ras_fn is None:
        return None

    cyl_radius = float(getattr(cfg, "v2_cylinder_radius_mm", 10.0))
    lib_min, lib_max = library_span_bounds
    cyl_depth = float(getattr(cfg, "v2_cylinder_depth_mm",
                               lib_max + 20.0 if lib_max > 0.0 else 150.0))
    hu_floor = float(getattr(cfg, "v2_cylinder_hu_floor", 1000.0))
    hd_min = float(getattr(cfg, "v2_cylinder_hd_min_mm", 3.0))
    ransac_tol = float(getattr(cfg, "v2_ransac_inlier_tol_mm", 2.0))
    ransac_max_angle = float(getattr(cfg, "v2_ransac_max_angle_deg", 15.0))
    step_mm = float(getattr(cfg, "extension_step_mm", 0.5))

    axis_scalp = fit.axis  # +axis = toward scalp
    axis_deep = -axis_scalp
    center = fit.center
    shape = arr_kji.shape

    # --- PCA refit on bolt-metal voxels in a wider tube (matches probe) ---
    # Gather very bright voxels (bolt metal > 2000 HU) near the bolt
    # center within a 2.5mm-radius tube and re-derive the axis via PCA.
    half_span = float(bolt.span_mm) / 2.0 + 1.0
    bolt_subvol_r = int(np.ceil(half_span + 5.0))
    c_ijk_raw = np.asarray(ras_to_ijk_fn(center), dtype=float)
    c_kji_int = np.array([int(round(c_ijk_raw[2])), int(round(c_ijk_raw[1])), int(round(c_ijk_raw[0]))])
    bk0 = max(0, c_kji_int[0] - bolt_subvol_r)
    bk1 = min(shape[0], c_kji_int[0] + bolt_subvol_r + 1)
    bj0 = max(0, c_kji_int[1] - bolt_subvol_r)
    bj1 = min(shape[1], c_kji_int[1] + bolt_subvol_r + 1)
    bi0 = max(0, c_kji_int[2] - bolt_subvol_r)
    bi1 = min(shape[2], c_kji_int[2] + bolt_subvol_r + 1)
    bolt_sub = arr_kji[bk0:bk1, bj0:bj1, bi0:bi1]
    bk, bj, bi = np.where(bolt_sub >= 2000.0)
    if bk.size >= 8:
        bolt_kji = np.stack([bk + bk0, bj + bj0, bi + bi0], axis=1).astype(float)
        bolt_ras = np.asarray(ijk_kji_to_ras_fn(bolt_kji), dtype=float)
        rel = bolt_ras - center
        proj_b = rel @ axis_deep
        perp_b = rel - np.outer(proj_b, axis_deep)
        dist_b = np.linalg.norm(perp_b, axis=1)
        in_tube = (dist_b < 2.5) & (np.abs(proj_b) < half_span)
        if int(in_tube.sum()) >= 8:
            tube_pts = bolt_ras[in_tube]
            c_tube = tube_pts.mean(axis=0)
            X = tube_pts - c_tube
            cov = X.T @ X / max(1, X.shape[0] - 1)
            eigvals, eigvecs = np.linalg.eigh(cov)
            a_refit = eigvecs[:, int(np.argmax(eigvals))]
            n = float(np.linalg.norm(a_refit))
            if n > 1e-9:
                a_refit = a_refit / n
                if np.dot(a_refit, axis_scalp) < 0:
                    a_refit = -a_refit
                axis_scalp = a_refit
                axis_deep = -axis_scalp
                center = c_tube

    # --- gather bright intracranial voxels in a wide cylinder ---
    u = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(axis_deep, u))) > 0.9:
        u = np.array([0.0, 1.0, 0.0])
    v_perp = np.cross(axis_deep, u)
    corners = []
    for t in (0.0, cyl_depth):
        for du in (-cyl_radius, cyl_radius):
            for dv in (-cyl_radius, cyl_radius):
                corners.append(center + t * axis_deep + du * u + dv * v_perp)
    corners = np.stack(corners)
    c_ijk = np.array([np.asarray(ras_to_ijk_fn(c), dtype=float) for c in corners])
    c_kji = np.rint(np.stack([c_ijk[:, 2], c_ijk[:, 1], c_ijk[:, 0]], axis=1)).astype(int)
    k_lo = max(0, int(c_kji[:, 0].min()) - 2)
    k_hi = min(shape[0], int(c_kji[:, 0].max()) + 3)
    j_lo = max(0, int(c_kji[:, 1].min()) - 2)
    j_hi = min(shape[1], int(c_kji[:, 1].max()) + 3)
    i_lo = max(0, int(c_kji[:, 2].min()) - 2)
    i_hi = min(shape[2], int(c_kji[:, 2].max()) + 3)
    if k_lo >= k_hi or j_lo >= j_hi or i_lo >= i_hi:
        return None

    subvol = arr_kji[k_lo:k_hi, j_lo:j_hi, i_lo:i_hi]
    hd_sub = head_distance_map_kji[k_lo:k_hi, j_lo:j_hi, i_lo:i_hi]
    bright_intra = (subvol >= hu_floor) & (hd_sub >= hd_min)
    if not bright_intra.any():
        return None

    kk, jj, ii = np.where(bright_intra)
    kji_abs = np.stack([kk + k_lo, jj + j_lo, ii + i_lo], axis=1).astype(float)
    ras_pts = np.asarray(ijk_kji_to_ras_fn(kji_abs), dtype=float)

    rel = ras_pts - center
    proj = rel @ axis_deep
    perp = rel - np.outer(proj, axis_deep)
    perp_dist = np.linalg.norm(perp, axis=1)
    in_cyl = (perp_dist <= cyl_radius) & (proj >= -5.0) & (proj <= cyl_depth)
    cyl_pts = ras_pts[in_cyl]
    if cyl_pts.shape[0] < 5:
        return None

    # --- RANSAC line fit constrained to bolt axis ---
    fit_axis, _, inlier_mask = _ransac_line_fit(
        cyl_pts, inlier_tol_mm=ransac_tol, min_inliers=5,
        prior_axis=axis_deep, max_angle_deg=ransac_max_angle,
    )
    if int(inlier_mask.sum()) < 5:
        return None

    if np.dot(fit_axis, axis_scalp) < 0:
        fit_axis = -fit_axis

    new_fit = axr.AxisFit(
        center=center, axis=fit_axis,
        residual_rms_mm=0.0, residual_median_mm=0.0, elongation=1.0,
        t_min=fit.t_min, t_max=fit.t_max,
        point_count=int(inlier_mask.sum()),
    )

    # --- deep tip via density-binned walk (avoids contralateral bone) ---
    inlier_pts = cyl_pts[inlier_mask]
    n_inliers = int(inlier_mask.sum())
    proj_in = (inlier_pts - center) @ fit_axis
    if proj_in.size == 0 or float(proj_in.min()) >= 0.0:
        return None

    # FP bolts have very few inliers — real electrodes have 150+
    # contact voxels within 2mm of a consistent line.
    min_inliers = int(getattr(cfg, "v2_min_ransac_inliers", 150))
    if n_inliers < min_inliers:
        return None

    density_bin = float(getattr(cfg, "v2_depth_trim_bin_mm", 5.0))
    density_min = int(getattr(cfg, "v2_depth_trim_min_count", 3))
    density_gap = int(getattr(cfg, "v2_depth_trim_gap_bins", 2))
    t_deep_intra = _density_trimmed_deep_tip(
        proj_in, bin_mm=density_bin, min_bin_count=density_min,
        gap_tolerance=density_gap,
    )
    if t_deep_intra is None:
        return None
    bolt_offset = float(getattr(cfg, "v2_cylinder_bolt_offset_mm", 12.0))
    t_interface = -bolt_offset
    if t_interface <= t_deep_intra:
        t_interface = min(0.0, t_deep_intra + 1.0)
    intracranial_span = float(t_interface - t_deep_intra)

    span_tol = float(getattr(cfg, "library_span_tolerance_mm", 5.0))
    best_model_id, _ = axr.library_span_match(
        intracranial_span, library_models, tolerance_mm=span_tol
    )

    return _emit_trajectory(
        bolt, new_fit, t_deep_intra, t_interface,
        intracranial_span, intracranial_span, best_model_id,
        source="bolt_v2_cylinder",
    )


def _fit_two_threshold(
    bolt,
    fit: "axr.AxisFit",
    arr_kji: np.ndarray,
    head_distance_map_kji: np.ndarray,
    ras_to_ijk_fn,
    library_models: list[dict[str, Any]],
    library_span_bounds: tuple[float, float],
    cfg,
) -> dict[str, Any] | None:
    contact_hu = float(getattr(cfg, "v2_contact_hu", 1000.0))
    tube_radius = float(getattr(cfg, "v2_contact_probe_radius_mm", 2.5))
    max_gap_mm = float(getattr(cfg, "v2_contact_max_gap_mm", 8.0))
    step_mm = float(getattr(cfg, "extension_step_mm", 0.5))

    t_deep = _find_deep_tip_by_contiguous_metal(
        fit.center, fit.axis, arr_kji, ras_to_ijk_fn,
        contact_hu=contact_hu,
        tube_radius_mm=tube_radius,
        inward_walk_mm=BOLT_INWARD_WALK_MM,
        step_mm=step_mm,
        max_gap_mm=max_gap_mm,
    )
    if t_deep is None:
        return None

    # Quick brain sanity via lateral HU ring classification between the
    # deep tip and the bolt center. Rejects lines where the walk was
    # entirely in bone (dental metal, frame screws that happened to
    # line up with a bolt).
    ts = np.arange(t_deep, 0.0 + step_mm * 0.5, step_mm)
    classes = axr.classify_tissue_along_axis(
        fit, ts, arr_kji, ras_to_ijk_fn, cfg
    )
    brain_span_mm = axr.intracranial_brain_span_mm(classes, step_mm)
    min_span = float(getattr(cfg, "min_intracranial_span_mm", 15.0))
    if brain_span_mm < min_span:
        return None

    bolt_offset = float(getattr(cfg, "bolt_endpoint_offset_mm", 8.0))
    t_interface = -bolt_offset
    t_deep_intra = float(t_deep)
    if t_interface <= t_deep_intra:
        t_interface = min(0.0, t_deep_intra + 1.0)

    intracranial_span = float(t_interface - t_deep_intra)
    if intracranial_span < min_span:
        return None

    # Library span gate (soft trim).
    lib_min, lib_max = library_span_bounds
    span_tol = float(getattr(cfg, "library_span_tolerance_mm", 5.0))
    if lib_max > 0.0:
        if intracranial_span < lib_min - span_tol:
            return None
        if intracranial_span > lib_max + span_tol:
            t_deep_intra = float(t_interface - (lib_max + span_tol))
            intracranial_span = float(t_interface - t_deep_intra)

    best_model_id, _ = axr.library_span_match(
        intracranial_span, library_models, tolerance_mm=span_tol
    )

    return _emit_trajectory(
        bolt, fit, t_deep_intra, t_interface,
        intracranial_span, brain_span_mm, best_model_id,
        source="bolt_v2_walk",
    )


# ---------------------------------------------------------------------------
# Approach B: intensity peaks + library fit
# ---------------------------------------------------------------------------

def _build_intensity_profile(
    center_ras: np.ndarray,
    axis_ras: np.ndarray,
    arr_kji: np.ndarray,
    ras_to_ijk_fn,
    *,
    disc_radius_mm: float,
    inward_mm: float,
    outward_mm: float,
    step_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_values, profile) where profile[i] = max HU in the disc
    perpendicular to ``axis_ras`` at center + t_values[i] * axis_ras.
    """
    offsets = _build_filled_disc_offsets(axis_ras, disc_radius_mm)
    t_values = np.arange(-inward_mm, outward_mm + 0.5 * step_mm, step_mm)
    profile = np.full(t_values.shape[0], -np.inf, dtype=np.float64)
    for i, t in enumerate(t_values):
        pt = center_ras + float(t) * axis_ras
        profile[i] = _max_hu_in_disc(pt, offsets, arr_kji, ras_to_ijk_fn)
    return t_values, profile


def _find_profile_peaks(
    profile: np.ndarray,
    *,
    min_value: float,
    min_separation_samples: int,
) -> np.ndarray:
    """Return indices of local maxima in ``profile`` with value >=
    ``min_value`` separated by at least ``min_separation_samples``
    samples. Greedy peak picking: take the globally-largest sample,
    mask out its neighborhood, repeat.
    """
    mask = np.isfinite(profile) & (profile >= float(min_value))
    if not np.any(mask):
        return np.zeros(0, dtype=int)
    scratch = profile.copy()
    scratch[~mask] = -np.inf
    picked: list[int] = []
    sep = int(max(1, min_separation_samples))
    while True:
        idx = int(np.argmax(scratch))
        if not np.isfinite(scratch[idx]) or scratch[idx] < min_value:
            break
        picked.append(idx)
        lo = max(0, idx - sep)
        hi = min(scratch.size, idx + sep + 1)
        scratch[lo:hi] = -np.inf
    picked.sort()
    return np.asarray(picked, dtype=int)


def _library_contact_spacings(model: dict[str, Any]) -> np.ndarray | None:
    """Return the ascending contact offsets (mm) from the deepest
    contact for one library model, or ``None`` if the model has no
    usable geometry. Offsets are relative to the deepest contact
    (``offsets[0] == 0``).
    """
    # Preferred schema: per-model ``contact_center_offsets_from_tip_mm``.
    from_tip = model.get("contact_center_offsets_from_tip_mm")
    if from_tip is not None:
        arr = np.asarray(from_tip, dtype=float).reshape(-1)
        if arr.size >= 2:
            arr = np.sort(arr)
            return arr - float(arr[0])
    # Back-compat shapes.
    contacts = model.get("contact_offsets_mm")
    if contacts is not None:
        arr = np.asarray(contacts, dtype=float).reshape(-1)
        if arr.size >= 2:
            arr = np.sort(arr)
            return arr - float(arr[0])
    contact_count = model.get("contact_count") or model.get("n_contacts")
    spacing = (
        model.get("contact_spacing_mm")
        or model.get("inter_contact_distance_mm")
        or model.get("center_to_center_separation_mm")
    )
    if contact_count is not None and spacing is not None:
        n = int(contact_count)
        s = float(spacing)
        if n >= 2 and s > 0.0:
            return np.asarray([i * s for i in range(n)], dtype=float)
    return None


def _fit_library_to_peaks(
    peak_t: np.ndarray,
    library_models: list[dict[str, Any]],
    *,
    tolerance_mm: float,
    max_shallow_t: float = 2.0,
) -> tuple[dict[str, Any] | None, float, float, float] | None:
    """For each library model, try aligning its known contact offsets
    to the observed peak positions. Returns the best ``(model,
    t_deepest, t_shallowest, score)`` or ``None`` if no model covers
    any peaks.

    ``score`` is the absolute number of the model's contacts that
    found a peak within ``tolerance_mm``, minus a small penalty for
    RMS alignment error. Absolute (not fractional) so longer
    electrodes aren't penalized relative to short ones.
    """
    if peak_t.size == 0:
        return None
    peaks = np.sort(peak_t.astype(float))
    best = None
    for model in library_models or []:
        spacings = _library_contact_spacings(model)
        if spacings is None:
            continue
        # Try each observed peak as the candidate deepest-contact
        # anchor. For each anchor, compute the model's expected peak
        # positions (anchor + spacings) and count matches.
        for anchor_t in peaks:
            expected = anchor_t + spacings
            match_count = 0
            sq_err = 0.0
            matched_obs: set[int] = set()
            matched_expected_idx: list[int] = []
            for exp_i, exp in enumerate(expected):
                dists = np.abs(peaks - exp)
                for j in np.argsort(dists):
                    if int(j) in matched_obs:
                        continue
                    if dists[j] > tolerance_mm:
                        break
                    matched_obs.add(int(j))
                    sq_err += float(dists[j]) ** 2
                    match_count += 1
                    matched_expected_idx.append(exp_i)
                    break
            if match_count < 2:
                continue
            rms_err = float(np.sqrt(sq_err / match_count))
            coverage = match_count / float(len(spacings))
            # Absolute-count-first scoring with a small penalty for rms
            # error and a tiny bonus for high coverage (breaks ties in
            # favor of the model that uses all its contacts).
            score = float(match_count) - 0.3 * rms_err + 0.1 * coverage
            t_deepest = anchor_t
            t_shallowest = anchor_t + float(spacings[-1])
            # Shallow end must be inside the bolt region — the first
            # electrode contact sits at or below the skull surface,
            # which along the scalp-oriented axis is at t ~ 0 to
            # -(skull thickness). Any fit whose shallow end extends
            # outside the bolt (``t > max_shallow_t``) is a
            # long-library-matched-to-scattered-noise artifact.
            if t_shallowest > max_shallow_t:
                continue
            # Require the MATCHED peaks to span most of the model's
            # expected extent — otherwise a model whose contacts fall
            # in a narrow subrange of ``peaks`` will out-score a
            # tighter fit by brute match count. ``matched_expected_idx``
            # holds the model indices that found a peak; require
            # first_matched_contact ~= 0 and last_matched_contact ~=
            # n_contacts-1 within a small tolerance.
            if matched_expected_idx:
                first_matched = min(matched_expected_idx)
                last_matched = max(matched_expected_idx)
                if first_matched > 1:
                    continue
                if last_matched < len(spacings) - 2:
                    continue
            if best is None or score > best[3]:
                best = (model, t_deepest, t_shallowest, score,
                        match_count, rms_err, coverage)
    if best is None:
        return None
    model, t_deepest, t_shallowest, score, _, _, _ = best
    return model, t_deepest, t_shallowest, score


def _fit_deepest_peak(
    bolt,
    fit: "axr.AxisFit",
    arr_kji: np.ndarray,
    head_distance_map_kji: np.ndarray,
    ras_to_ijk_fn,
    library_models: list[dict[str, Any]],
    library_span_bounds: tuple[float, float],
    cfg,
) -> dict[str, Any] | None:
    """Approach C: build the intensity profile and use the deepest
    qualifying peak as the deep tip. Shallow end is anchored at
    ``-bolt_endpoint_offset_mm`` exactly like v1+bolts.

    This skips library matching entirely — the library is only used
    to validate the final intracranial span. The premise is that
    finding a single deepest contact is much easier than matching a
    whole contact pattern, and it's all we actually need to define
    the trajectory's two endpoints.
    """
    step_mm = float(getattr(cfg, "v2_profile_step_mm", 0.25))
    disc_radius = float(getattr(cfg, "v2_profile_disc_radius_mm", 2.5))
    peak_hu_floor = float(getattr(cfg, "v2_profile_peak_hu_floor", 800.0))
    peak_rel_frac = float(getattr(cfg, "v2_profile_peak_rel_frac", 0.35))
    max_gap_mm = float(getattr(cfg, "v2_contact_max_gap_mm", 8.0))
    bolt_offset = float(getattr(cfg, "bolt_endpoint_offset_mm", 8.0))

    t_values, profile = _build_intensity_profile(
        fit.center, fit.axis, arr_kji, ras_to_ijk_fn,
        disc_radius_mm=disc_radius,
        inward_mm=BOLT_INWARD_WALK_MM,
        outward_mm=BOLT_OUTWARD_WALK_MM,
        step_mm=step_mm,
    )

    finite = profile[np.isfinite(profile)]
    if finite.size == 0:
        return None
    profile_max = float(np.max(finite))
    min_value = max(peak_hu_floor, peak_rel_frac * profile_max)

    # Walk from the bolt (t ~ 0) inward (decreasing t). Track the
    # deepest sample where profile >= min_value under a gap tolerance:
    # if we haven't seen a qualifying sample in the last ``max_gap_mm``
    # steps, stop walking. This gives us the deep tip as the deepest
    # contact-like peak contiguous (within gap tolerance) with the
    # bolt through the electrode shaft.
    bolt_center_idx = int(np.argmin(np.abs(t_values)))
    max_gap_steps = max(1, int(round(max_gap_mm / step_mm)))
    deepest_idx = None
    gap_run = 0
    for i in range(bolt_center_idx, -1, -1):
        val = profile[i]
        if np.isfinite(val) and val >= min_value:
            deepest_idx = i
            gap_run = 0
        else:
            gap_run += 1
            if gap_run > max_gap_steps:
                break
    if deepest_idx is None:
        return None

    t_deep_intra = float(t_values[deepest_idx])
    t_interface = -bolt_offset
    if t_interface <= t_deep_intra:
        t_interface = min(0.0, t_deep_intra + 1.0)

    intracranial_span = float(t_interface - t_deep_intra)
    min_span = float(getattr(cfg, "min_intracranial_span_mm", 15.0))
    if intracranial_span < min_span:
        return None

    # Brain sanity check.
    ts = np.arange(t_deep_intra, t_interface + step_mm * 0.5, step_mm)
    classes = axr.classify_tissue_along_axis(
        fit, ts, arr_kji, ras_to_ijk_fn, cfg
    )
    brain_span_mm = axr.intracranial_brain_span_mm(classes, step_mm)
    if brain_span_mm < min_span:
        return None

    # Library span gate (soft trim).
    lib_min, lib_max = library_span_bounds
    span_tol = float(getattr(cfg, "library_span_tolerance_mm", 5.0))
    if lib_max > 0.0:
        if intracranial_span < lib_min - span_tol:
            return None
        if intracranial_span > lib_max + span_tol:
            t_deep_intra = float(t_interface - (lib_max + span_tol))
            intracranial_span = float(t_interface - t_deep_intra)

    best_model_id, _ = axr.library_span_match(
        intracranial_span, library_models, tolerance_mm=span_tol
    )

    return _emit_trajectory(
        bolt, fit, t_deep_intra, t_interface,
        intracranial_span, brain_span_mm, best_model_id,
        source="bolt_v2_deepest",
    )


def _fit_intensity_peaks(
    bolt,
    fit: "axr.AxisFit",
    arr_kji: np.ndarray,
    head_distance_map_kji: np.ndarray,
    ras_to_ijk_fn,
    library_models: list[dict[str, Any]],
    library_span_bounds: tuple[float, float],
    cfg,
) -> dict[str, Any] | None:
    step_mm = float(getattr(cfg, "v2_profile_step_mm", 0.25))
    disc_radius = float(getattr(cfg, "v2_profile_disc_radius_mm", 2.5))
    min_peak_sep_mm = float(getattr(cfg, "v2_profile_min_peak_sep_mm", 2.5))
    peak_hu_floor = float(getattr(cfg, "v2_profile_peak_hu_floor", 800.0))
    peak_rel_frac = float(getattr(cfg, "v2_profile_peak_rel_frac", 0.35))
    match_tol_mm = float(getattr(cfg, "v2_profile_peak_match_tol_mm", 1.5))

    t_values, profile = _build_intensity_profile(
        fit.center, fit.axis, arr_kji, ras_to_ijk_fn,
        disc_radius_mm=disc_radius,
        inward_mm=BOLT_INWARD_WALK_MM,
        outward_mm=BOLT_OUTWARD_WALK_MM,
        step_mm=step_mm,
    )

    # Adaptive min peak value: the larger of the absolute floor and a
    # fraction of the profile max. For dim electrodes the floor
    # dominates; for bright ones the relative fraction dominates.
    finite = profile[np.isfinite(profile)]
    if finite.size == 0:
        return None
    profile_max = float(np.max(finite))
    min_peak_value = max(peak_hu_floor, peak_rel_frac * profile_max)
    min_sep = int(round(min_peak_sep_mm / step_mm))
    peak_idx = _find_profile_peaks(profile, min_value=min_peak_value,
                                   min_separation_samples=min_sep)
    if peak_idx.size < 2:
        return None
    peak_t = t_values[peak_idx]

    # Only keep peaks on the inward side of the bolt (strictly negative
    # t — we're looking for contacts, not the bolt metal itself).
    # A small positive margin allows the shallow-most contact to sit
    # just barely inside the bolt span.
    peak_t = peak_t[peak_t <= 0.5]
    if peak_t.size < 2:
        return None

    fit_result = _fit_library_to_peaks(
        peak_t, library_models, tolerance_mm=match_tol_mm
    )
    if fit_result is None:
        return None
    model, t_deepest, t_shallowest, score = fit_result

    # Contact-row span == distance between deepest and shallowest
    # matched contact centers. For Approach B we do NOT apply the v1
    # ``min_intracranial_span_mm`` gate — that was calibrated to the
    # deep_core brain-entry-to-deepest-contact span, which is longer
    # than the contact row itself. The library span gate still rejects
    # nonsense fits.
    intracranial_span = float(t_shallowest - t_deepest)

    # Brain sanity check. We sample from the deepest matched peak up
    # to the bolt, not just the contact row, so the brain-span check
    # catches trajectories that never enter the brain proper.
    sanity_t_lo = min(t_deepest, -20.0)
    sanity_t_hi = max(0.0, t_shallowest)
    ts = np.arange(sanity_t_lo, sanity_t_hi + step_mm * 0.5, step_mm)
    classes = axr.classify_tissue_along_axis(
        fit, ts, arr_kji, ras_to_ijk_fn, cfg
    )
    brain_span_mm = axr.intracranial_brain_span_mm(classes, step_mm)
    min_brain = 8.0
    if brain_span_mm < min_brain:
        return None

    return _emit_trajectory(
        bolt, fit, t_deepest, t_shallowest,
        intracranial_span, brain_span_mm,
        str(model.get("id", "")),
        source="bolt_v2_peaks",
    )


# ---------------------------------------------------------------------------
# Shared trajectory emission
# ---------------------------------------------------------------------------

def _emit_trajectory(
    bolt,
    fit,
    t_deep_intra: float,
    t_interface: float,
    intracranial_span: float,
    brain_span_mm: float,
    best_model_id: str,
    *,
    source: str,
) -> dict[str, Any]:
    t_bolt_outer = float(bolt.span_mm) * 0.5
    start_ras = fit.center + fit.axis * float(t_deep_intra)
    end_ras = fit.center + fit.axis * float(t_interface)
    bolt_ras = fit.center + fit.axis * t_bolt_outer
    return {
        "source": source,
        "start_ras": [float(v) for v in start_ras],
        "end_ras": [float(v) for v in end_ras],
        "bolt_ras": [float(v) for v in bolt_ras],
        "axis_ras": [float(v) for v in fit.axis],
        "bolt_center_ras": [float(v) for v in bolt.center_ras],
        "span_mm": float(intracranial_span),
        "intracranial_span_mm": float(intracranial_span),
        "bolt_extent_mm": float(max(0.0, t_bolt_outer - t_interface)),
        "brain_span_mm": float(brain_span_mm),
        "axis_residual_rms_mm": 0.0,
        "axis_residual_median_mm": 0.0,
        "best_model_id": str(best_model_id or ""),
        "model_fit_passed": True,
        "bolt_seed": True,
        "bolt_span_mm": float(bolt.span_mm),
        "bolt_hd_center_mm": float(bolt.hd_center_mm),
        "bolt_fill_frac": float(bolt.fill_frac),
        "atom_id_list": [],
        "explained_atom_ids": [],
    }


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def fit_bolt_trajectory(
    bolt,
    *,
    arr_kji: np.ndarray,
    head_distance_map_kji: np.ndarray,
    ras_to_ijk_fn,
    ijk_kji_to_ras_fn=None,
    library_models: list[dict[str, Any]],
    library_span_bounds: tuple[float, float],
    cfg,
) -> dict[str, Any] | None:
    """Turn one bolt candidate into a trajectory dict (or ``None``)."""
    axis = np.asarray(bolt.axis_ras, dtype=float).reshape(3)
    axis_n = float(np.linalg.norm(axis))
    if axis_n < 1e-9:
        return None
    axis = axis / axis_n
    center = np.asarray(bolt.center_ras, dtype=float).reshape(3)
    axis = _orient_bolt_axis_toward_scalp(
        axis, center, head_distance_map_kji, ras_to_ijk_fn
    )

    fit = axr.AxisFit(
        center=center,
        axis=axis,
        residual_rms_mm=0.0,
        residual_median_mm=0.0,
        elongation=1.0,
        t_min=-float(bolt.span_mm) * 0.5,
        t_max=+float(bolt.span_mm) * 0.5,
        point_count=int(bolt.n_inliers),
    )

    mode = str(getattr(cfg, "v2_fit_mode", "cylinder_ransac")).strip().lower()
    if mode == "cylinder_ransac":
        return _fit_cylinder_ransac(
            bolt, fit, arr_kji, head_distance_map_kji, ras_to_ijk_fn,
            library_models, library_span_bounds, cfg,
            ijk_kji_to_ras_fn=ijk_kji_to_ras_fn,
        )
    if mode == "intensity_peaks":
        return _fit_intensity_peaks(
            bolt, fit, arr_kji, head_distance_map_kji, ras_to_ijk_fn,
            library_models, library_span_bounds, cfg,
        )
    if mode == "two_threshold":
        return _fit_two_threshold(
            bolt, fit, arr_kji, head_distance_map_kji, ras_to_ijk_fn,
            library_models, library_span_bounds, cfg,
        )
    return _fit_deepest_peak(
        bolt, fit, arr_kji, head_distance_map_kji, ras_to_ijk_fn,
        library_models, library_span_bounds, cfg,
    )


def run_bolt_fit_group(
    bolts,
    *,
    arr_kji: np.ndarray,
    head_distance_map_kji: np.ndarray,
    ras_to_ijk_fn,
    ijk_kji_to_ras_fn=None,
    library_models: list[dict[str, Any]],
    cfg,
) -> dict[str, Any]:
    """Fit every bolt candidate independently. Returns the same shape as
    ``run_model_fit_group`` so the v2 pipeline can swap it in without
    touching trajectory conversion downstream.
    """
    lib_bounds = axr.library_span_range(library_models)
    accepted: list[dict[str, Any]] = []
    rejected = 0
    for bolt in (bolts or []):
        traj = fit_bolt_trajectory(
            bolt,
            arr_kji=arr_kji,
            head_distance_map_kji=head_distance_map_kji,
            ras_to_ijk_fn=ras_to_ijk_fn,
            ijk_kji_to_ras_fn=ijk_kji_to_ras_fn,
            library_models=library_models,
            library_span_bounds=lib_bounds,
            cfg=cfg,
        )
        if traj is None:
            rejected += 1
            continue
        accepted.append(traj)
    return {
        "accepted_proposals": accepted,
        "stats": {
            "input_count": int(len(bolts or [])),
            "accepted_count": int(len(accepted)),
            "rejected": int(rejected),
        },
    }
