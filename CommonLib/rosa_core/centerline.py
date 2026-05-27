"""Per-trajectory centerline signal computation.

Given an entry/tip endpoint pair (RAS) and a CT volume, samples a 2-D HU
disk per arc step along the trajectory's fitted axis and returns
arc-relative signal arrays used by the seeded-fit + electrode-classifier
pipeline:

* ``bone_width_mm``         — effective diameter of voxels with HU ≥ bone_hu
* ``metal_width_mm``        — effective diameter of voxels with HU ≥ metal_hu
* ``metal_width_eroded_mm`` — same after one binary erosion (separates
  contact peaks from continuous wire/bolt mass)
* ``ring_metal_total``      — Σ max(0, HU − metal_hu) over an annulus
  ``[ring_inner_mm, disk_radius_mm]``; suppresses metal at the axis,
  emphasises bolt-body ring HU
* ``disk_metal_excess``     — Σ max(0, HU − metal_hu) over the full disk
* ``disk_nonmetal_sum``     — Σ max(0, HU) over voxels < metal_hu
* ``max_hu``                — disk-max HU per arc step

Plus two convenience builders for derived signals used by the picker:

* :func:`build_detrended_ratio` — 100·disk_metal_excess/disk_nonmetal_sum
  with a running-min baseline removed
* :func:`build_signals_along_arc` — convenience wrapper that produces a
  dict of all the signals the picker scores against

Constants are exposed at module level so callers can tune them per case
without editing the function signatures.

Ported from ``notebooks/seeded_fit/_build_starter.py`` 2026-05-18; same
algorithm validated on S57 (16/16) and S54 (13/15 strict, 15/15 relaxed).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .volume_sampling import sample_trilinear_batch


# ---------------------------------------------------------------------
# Centerline sampling constants (per cell-3 of _build_starter.py).
# ---------------------------------------------------------------------

PROFILE_DISK_RADIUS_MM = 5.0    # 10 mm diameter disc
PROFILE_RING_INNER_MM  = 1.5    # exclude this radius for the "ring" sum
PROFILE_PAD_ENTRY_MM   = 15.0   # walk this far OUTSIDE the fitted entry
PROFILE_PAD_TIP_MM     = 5.0    # walk this far PAST the fitted tip
PROFILE_STEP_MM        = 0.3
PROFILE_GRID_STEP_MM   = 0.5    # 2D disk sample-grid step
PROFILE_EROSION_ITERS  = 1      # binary erosions on metal mask before width calc
HU_BONE_THRESHOLD      = 300.0
HU_METAL_THRESHOLD     = 1500.0
RATIO_DETREND_HALFWIDTH_MM = 7.0


# ---------------------------------------------------------------------
# Small linear-algebra utilities (kept local to avoid pulling in
# rosa_detect.primitives.geometry, which adds an HU-rescue layer that
# isn't needed for centerline math).
# ---------------------------------------------------------------------


def unit(v) -> np.ndarray:
    """Unit vector; returns the input unchanged if its norm is < 1e-9."""
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def orthonormal_basis_for_axis(axis) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane perpendicular to ``axis``."""
    u = unit(axis)
    seed = np.eye(3)[int(np.argmin(np.abs(u)))]
    p1 = unit(np.cross(u, seed))
    p2 = unit(np.cross(u, p1))
    return p1, p2


# ---------------------------------------------------------------------
# Intensity-threshold helpers.
# ---------------------------------------------------------------------


def otsu_threshold(values, *, n_bins: int = 256,
                   floor: float = 300.0,
                   fallback: float = HU_METAL_THRESHOLD) -> float:
    """Otsu split of ``values`` (HU samples) clamped above ``floor``.

    Used to choose a per-trajectory metal threshold from the intracranial
    HU distribution — more robust than the fixed ``HU_METAL_THRESHOLD``
    when bolt/contact HU varies (kid CTs, low-dose, etc.).
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v >= floor)]
    if v.size < 32:
        return float(fallback)
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1.0:
        return float(fallback)
    counts, edges = np.histogram(v, bins=n_bins, range=(lo, hi))
    total = counts.sum()
    if total == 0:
        return float(fallback)
    centers = 0.5 * (edges[:-1] + edges[1:])
    omega_cum = np.cumsum(counts) / total
    mu_cum    = np.cumsum(counts * centers) / total
    mu_total  = mu_cum[-1]
    denom = omega_cum * (1.0 - omega_cum)
    denom[denom < 1e-9] = 1e-9
    sigma_b2 = (mu_total * omega_cum - mu_cum) ** 2 / denom
    idx = int(np.argmax(sigma_b2))
    return float(centers[idx])


# ---------------------------------------------------------------------
# Intracranial HU sampler — for per-trajectory Otsu threshold choice.
# ---------------------------------------------------------------------


def collect_intracranial_hu_along_trajectory(
    entry_ras, tip_ras, axis_unit, *,
    ct_vol: np.ndarray,
    ras_to_ijk_mat: np.ndarray,
    intracranial_mask_arr: np.ndarray | None,
    tube_radius_mm: float = 3.0,
    step_mm: float = 0.5,
) -> np.ndarray:
    """Sample HU inside a tube around the trajectory; keep only voxels
    that fall inside ``intracranial_mask_arr``.

    Returns a 1-D HU array (empty when the trajectory is too short or no
    mask was supplied). Use as input to :func:`otsu_threshold` to pick a
    per-trajectory metal HU floor that reflects the actual electrode
    density on this scan.
    """
    s = np.asarray(entry_ras, dtype=float)
    e = np.asarray(tip_ras,   dtype=float)
    L = float(np.linalg.norm(e - s))
    if L < 5.0 or intracranial_mask_arr is None:
        return np.array([], dtype=float)
    axis  = unit(e - s)
    perp1, perp2 = orthonormal_basis_for_axis(axis)
    n_d = int(2 * tube_radius_mm / step_mm) + 1
    u_arr = np.linspace(-tube_radius_mm, tube_radius_mm, n_d)
    UU, VV = np.meshgrid(u_arr, u_arr, indexing="ij")
    in_disk = (UU ** 2 + VV ** 2) <= tube_radius_mm ** 2
    u_in, v_in = UU[in_disk], VV[in_disk]
    n_arcs = int(L / step_mm) + 1
    arc_arr = np.linspace(0.0, L, n_arcs)
    ct_f32   = ct_vol.astype(np.float32, copy=False)
    mask_f32 = intracranial_mask_arr.astype(np.float32)
    chunks = []
    for arc in arc_arr:
        center = s + arc * axis
        pts = (center[None, :]
               + u_in[:, None] * perp1[None, :]
               + v_in[:, None] * perp2[None, :])
        vals      = sample_trilinear_batch(ct_f32,   ras_to_ijk_mat, pts)
        mask_vals = sample_trilinear_batch(mask_f32, ras_to_ijk_mat, pts)
        keep = (mask_vals > 0.5) & np.isfinite(vals)
        if keep.any():
            chunks.append(vals[keep])
    return np.concatenate(chunks) if chunks else np.array([], dtype=float)


# ---------------------------------------------------------------------
# Main centerline sampler.
# ---------------------------------------------------------------------


def sample_trajectory_profile(
    entry_ras, tip_ras, axis_unit, *,
    ct_vol: np.ndarray,
    ras_to_ijk_mat: np.ndarray,
    pad_entry_mm: float = PROFILE_PAD_ENTRY_MM,
    pad_tip_mm: float = PROFILE_PAD_TIP_MM,
    step_mm: float = PROFILE_STEP_MM,
    disk_radius_mm: float = PROFILE_DISK_RADIUS_MM,
    ring_inner_mm: float = PROFILE_RING_INNER_MM,
    grid_step_mm: float = PROFILE_GRID_STEP_MM,
    bone_hu: float = HU_BONE_THRESHOLD,
    metal_hu: float = HU_METAL_THRESHOLD,
    erosion_iters: int = PROFILE_EROSION_ITERS,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Walk a centerline from ``entry − pad_entry_mm`` (outside skull) to
    ``tip + pad_tip_mm`` (past the deepest contact), sampling a 2D HU disk
    per step.

    Returns ``(arc_rel, prof)`` where ``arc_rel`` is arc distance relative
    to ``entry_ras`` (negative = outside skull, 0 = entry, positive = into
    brain; tip lives at ``|tip − entry|``) and ``prof`` is a dict of per-arc
    arrays — see module docstring for the keys.
    """
    from scipy.ndimage import binary_erosion as _binary_erosion

    s = np.asarray(entry_ras, dtype=float)
    e = np.asarray(tip_ras,   dtype=float)
    axis_in = unit(axis_unit)
    perp1, perp2 = orthonormal_basis_for_axis(axis_in)
    entry_to_tip = float(np.linalg.norm(e - s))

    n = int(2 * disk_radius_mm / grid_step_mm) + 1
    u_arr = np.linspace(-disk_radius_mm, disk_radius_mm, n)
    UU, VV = np.meshgrid(u_arr, u_arr, indexing="ij")
    r2 = UU ** 2 + VV ** 2
    in_disk = r2 <= disk_radius_mm ** 2
    in_ring = (r2 >= ring_inner_mm ** 2) & in_disk
    voxel_area_mm2 = grid_step_mm ** 2
    u_grid_flat = UU.ravel()
    v_grid_flat = VV.ravel()

    arc_start = -pad_entry_mm
    arc_end   = entry_to_tip + pad_tip_mm
    n_arcs    = int((arc_end - arc_start) / step_mm) + 1
    arc_rel   = np.linspace(arc_start, arc_end, n_arcs)

    bone_width         = np.zeros(n_arcs, dtype=float)
    metal_width        = np.zeros(n_arcs, dtype=float)
    metal_width_eroded = np.zeros(n_arcs, dtype=float)
    ring_total         = np.zeros(n_arcs, dtype=float)
    disk_metal_excess  = np.zeros(n_arcs, dtype=float)
    disk_nonmetal_sum  = np.zeros(n_arcs, dtype=float)
    max_hu             = np.zeros(n_arcs, dtype=float)
    ct_f32 = ct_vol.astype(np.float32, copy=False)

    for i, arc in enumerate(arc_rel):
        center = s + arc * axis_in
        pts_grid = (center[None, :]
                    + u_grid_flat[:, None] * perp1[None, :]
                    + v_grid_flat[:, None] * perp2[None, :])
        vals_grid_flat = sample_trilinear_batch(ct_f32, ras_to_ijk_mat, pts_grid)
        vals_grid = vals_grid_flat.reshape(UU.shape)
        finite_in_disk = vals_grid[in_disk]
        finite_in_disk = finite_in_disk[np.isfinite(finite_in_disk)]
        max_hu[i] = float(finite_in_disk.max()) if finite_in_disk.size else float("nan")
        bone_area  = float((finite_in_disk >= bone_hu).sum())  * voxel_area_mm2
        metal_area = float((finite_in_disk >= metal_hu).sum()) * voxel_area_mm2
        bone_width[i]  = 2.0 * np.sqrt(bone_area  / np.pi)
        metal_width[i] = 2.0 * np.sqrt(metal_area / np.pi)
        finite_grid_mask = np.isfinite(vals_grid)
        metal_mask_2d = (vals_grid >= metal_hu) & in_disk & finite_grid_mask
        if erosion_iters > 0 and metal_mask_2d.any():
            eroded = _binary_erosion(metal_mask_2d, iterations=erosion_iters)
        else:
            eroded = metal_mask_2d
        metal_width_eroded[i] = 2.0 * np.sqrt(float(eroded.sum()) * voxel_area_mm2 / np.pi)
        in_ring_finite = in_ring & finite_grid_mask
        if in_ring_finite.any():
            vals_ring = vals_grid[in_ring_finite]
            ring_total[i] = float(np.maximum(vals_ring - metal_hu, 0.0).sum())
        disk_metal_excess[i] = float(np.maximum(finite_in_disk - metal_hu, 0.0).sum())
        nonmetal_vals = finite_in_disk[finite_in_disk < metal_hu]
        disk_nonmetal_sum[i] = float(np.maximum(nonmetal_vals, 0.0).sum())
    return arc_rel, dict(
        bone_width_mm=bone_width,
        metal_width_mm=metal_width,
        metal_width_eroded_mm=metal_width_eroded,
        ring_metal_total=ring_total,
        disk_metal_excess=disk_metal_excess,
        disk_nonmetal_sum=disk_nonmetal_sum,
        max_hu=max_hu,
    )


# ---------------------------------------------------------------------
# Derived-signal builders used by the picker.
# ---------------------------------------------------------------------


def build_detrended_ratio(prof: dict[str, np.ndarray],
                          *, step_mm: float = PROFILE_STEP_MM,
                          halfwidth_mm: float = RATIO_DETREND_HALFWIDTH_MM) -> np.ndarray:
    """``100 · disk_metal_excess / max(disk_nonmetal_sum, 1)`` with a
    running-min baseline removed on a ±halfwidth_mm window. Suppresses
    slow scan-wide intensity drifts and leaves contact peaks intact."""
    from scipy.ndimage import minimum_filter1d as _mf1
    me = np.asarray(prof["disk_metal_excess"], dtype=float)
    nm = np.asarray(prof["disk_nonmetal_sum"], dtype=float)
    ratio = 100.0 * me / np.maximum(nm, 1.0)
    win_n = max(3, int(2 * halfwidth_mm / step_mm))
    if len(ratio) <= win_n:
        return np.zeros_like(ratio)
    baseline = _mf1(ratio, size=win_n, mode="nearest")
    return np.maximum(ratio - baseline, 0.0)


def build_signals_along_arc(
    arc_rel: np.ndarray, prof: dict[str, np.ndarray],
    *, step_mm: float = PROFILE_STEP_MM,
) -> dict[str, np.ndarray]:
    """Build the set of named signals used by ``score_contact_fit`` and the
    discrimination heatmap. Convenience wrapper — every signal is also
    accessible directly from ``prof`` except ``detrended_ratio`` and a
    rolling 2-mm-window cluster envelope (``metal_width_avg``)."""
    from scipy.ndimage import uniform_filter1d as _uf1
    me            = np.asarray(prof["disk_metal_excess"], dtype=float)
    width         = np.asarray(prof["metal_width_mm"], dtype=float)
    width_eroded  = np.asarray(prof["metal_width_eroded_mm"], dtype=float)
    ring_total    = np.asarray(prof["ring_metal_total"], dtype=float)
    bone_width    = np.asarray(prof["bone_width_mm"], dtype=float)
    detrended     = build_detrended_ratio(prof, step_mm=step_mm)
    # ±2 mm rolling mean of metal_width — emphasises CM/BM contact clusters
    # (5 contacts pull the local mean up; gaps drop it).
    win_n = max(3, int(2 * 2.0 / step_mm))
    width_avg = _uf1(width, size=win_n, mode="nearest")
    log_neg = np.maximum(-me, 0.0)  # placeholder; raw LoG-neg requires the
                                    # full feature volume — leave as zero
                                    # array here so callers can plug in
                                    # their own sampled signal.
    return dict(
        bone_width_mm=bone_width,
        metal_width_mm=width,
        metal_width_eroded_mm=width_eroded,
        metal_width_avg=width_avg,
        ring_metal_total=ring_total,
        disk_metal_excess=me,
        detrended_ratio=detrended,
        log_neg=log_neg,
    )


__all__ = [
    "unit",
    "orthonormal_basis_for_axis",
    "otsu_threshold",
    "collect_intracranial_hu_along_trajectory",
    "sample_trajectory_profile",
    "build_detrended_ratio",
    "build_signals_along_arc",
    # Constants (so CLI/tests can read defaults without re-defining).
    "PROFILE_DISK_RADIUS_MM",
    "PROFILE_RING_INNER_MM",
    "PROFILE_PAD_ENTRY_MM",
    "PROFILE_PAD_TIP_MM",
    "PROFILE_STEP_MM",
    "PROFILE_GRID_STEP_MM",
    "PROFILE_EROSION_ITERS",
    "HU_BONE_THRESHOLD",
    "HU_METAL_THRESHOLD",
    "RATIO_DETREND_HALFWIDTH_MM",
]
