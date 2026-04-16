"""Pure-numpy helpers for Phase B trajectory reconstruction.

These helpers are the building blocks of the redesigned
``run_model_fit_group`` in :mod:`deep_core_model_fit`. They are
deliberately free of Slicer / vtk / SimpleITK imports so they can be
unit-tested in isolation against synthetic volumes and synthetic atom
point clouds.

See ``docs/PHASE_B_REDESIGN.md`` for the algorithmic context. The
function list and contracts match the "Resume point for next session"
checklist in that document.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Small internal utilities
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if n <= 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return v / n


def _perp_frame(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two unit vectors perpendicular to ``axis`` and to each other."""
    a = _unit(axis)
    helper = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(a, helper))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=float)
    u = helper - float(np.dot(helper, a)) * a
    u = _unit(u)
    v = np.cross(a, u)
    v = _unit(v)
    return u, v


def _project_axial(points_ras: np.ndarray, center: np.ndarray, axis: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_ras, dtype=float).reshape(-1, 3)
    if pts.size == 0:
        return np.zeros((0,), dtype=float)
    return ((pts - center.reshape(1, 3)) @ axis.reshape(3, 1)).reshape(-1)


def _project_radial(points_ras: np.ndarray, center: np.ndarray, axis: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_ras, dtype=float).reshape(-1, 3)
    if pts.size == 0:
        return np.zeros((0,), dtype=float)
    centered = pts - center.reshape(1, 3)
    axial = (centered @ axis.reshape(3, 1)) * axis.reshape(1, 3)
    radial = centered - axial
    return np.linalg.norm(radial, axis=1)


def _sample_mask_nearest(
    mask_kji: np.ndarray,
    ras_to_ijk_fn: Callable[[np.ndarray], Sequence[float]],
    points_ras: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbour sample a 3D mask at a batch of RAS points.

    Returns a boolean array. Out-of-bounds and non-finite samples read
    as ``False``.
    """
    pts = np.asarray(points_ras, dtype=float).reshape(-1, 3)
    out = np.zeros((pts.shape[0],), dtype=bool)
    if mask_kji is None or pts.shape[0] == 0:
        return out
    kmax, jmax, imax = int(mask_kji.shape[0]), int(mask_kji.shape[1]), int(mask_kji.shape[2])
    for idx in range(pts.shape[0]):
        ijk = ras_to_ijk_fn(pts[idx])
        i = int(round(float(ijk[0])))
        j = int(round(float(ijk[1])))
        k = int(round(float(ijk[2])))
        if 0 <= k < kmax and 0 <= j < jmax and 0 <= i < imax:
            out[idx] = bool(mask_kji[k, j, i])
    return out


def _sample_array_nearest(
    arr_kji: np.ndarray,
    ras_to_ijk_fn: Callable[[np.ndarray], Sequence[float]],
    points_ras: np.ndarray,
    fill: float = np.nan,
) -> np.ndarray:
    """Nearest-neighbour sample a 3D scalar volume at a batch of RAS points."""
    pts = np.asarray(points_ras, dtype=float).reshape(-1, 3)
    out = np.full((pts.shape[0],), float(fill), dtype=float)
    if arr_kji is None or pts.shape[0] == 0:
        return out
    kmax, jmax, imax = int(arr_kji.shape[0]), int(arr_kji.shape[1]), int(arr_kji.shape[2])
    for idx in range(pts.shape[0]):
        ijk = ras_to_ijk_fn(pts[idx])
        i = int(round(float(ijk[0])))
        j = int(round(float(ijk[1])))
        k = int(round(float(ijk[2])))
        if 0 <= k < kmax and 0 <= j < jmax and 0 <= i < imax:
            val = float(arr_kji[k, j, i])
            if np.isfinite(val):
                out[idx] = val
    return out


# ---------------------------------------------------------------------------
# Step 1 — axis refinement
# ---------------------------------------------------------------------------

@dataclass
class AxisFit:
    center: np.ndarray
    axis: np.ndarray  # unit vector
    residual_rms_mm: float
    residual_median_mm: float
    elongation: float
    t_min: float
    t_max: float
    point_count: int


def refine_axis_from_cloud(points_ras: np.ndarray, seed_axis: np.ndarray | None = None) -> AxisFit | None:
    """PCA line fit through a RAS point cloud.

    Returns ``None`` if the cloud is empty or degenerate. The returned
    ``axis`` is a unit vector; ``center`` is the cloud mean; residuals
    are perpendicular distances from each point to the fitted line.
    """
    pts = np.asarray(points_ras, dtype=float).reshape(-1, 3)
    if pts.shape[0] < 2:
        return None
    center = pts.mean(axis=0)
    centered = pts - center.reshape(1, 3)
    cov = (centered.T @ centered) / max(1, pts.shape[0] - 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = np.asarray(evals[order], dtype=float)
    evecs = np.asarray(evecs[:, order], dtype=float)
    axis = _unit(evecs[:, 0])
    if seed_axis is not None:
        seed = _unit(np.asarray(seed_axis, dtype=float).reshape(3))
        if float(np.dot(axis, seed)) < 0.0:
            axis = -axis
    axial = (centered @ axis.reshape(3, 1)).reshape(-1)
    lateral = centered - axial.reshape(-1, 1) * axis.reshape(1, 3)
    radial = np.linalg.norm(lateral, axis=1)
    rms = float(np.sqrt(np.mean(radial * radial))) if radial.size else 0.0
    median = float(np.median(radial)) if radial.size else 0.0
    elongation = float(np.sqrt(max(evals[0], 1e-12) / max(evals[1], 1e-12)))
    return AxisFit(
        center=np.asarray(center, dtype=float).reshape(3),
        axis=axis,
        residual_rms_mm=rms,
        residual_median_mm=median,
        elongation=elongation,
        t_min=float(np.min(axial)) if axial.size else 0.0,
        t_max=float(np.max(axial)) if axial.size else 0.0,
        point_count=int(pts.shape[0]),
    )


# ---------------------------------------------------------------------------
# Step 2 — colinear atom reabsorption
# ---------------------------------------------------------------------------

def reabsorb_colinear_atoms(
    fit: AxisFit,
    atom_pool: list[dict[str, Any]],
    cfg: Any,
    already_absorbed_ids: set[int] | None = None,
) -> list[int]:
    """Return the atom IDs in ``atom_pool`` that are colinear with ``fit``.

    An atom qualifies if its center lies within ``reabsorb_radial_tol_mm``
    perpendicular of the refined line. For line atoms with a reliable
    own axis, an angular filter is also applied. No axial distance
    restriction is enforced — the whole point of reabsorption is to
    recover atoms the previous phase didn't chain together, which for
    multi-group electrodes (e.g. DIXI-15CM) means reaching contact
    atoms 30mm+ outside the cluster's initial extent. Already-absorbed
    atoms are never re-reported.
    """
    radial_tol = float(getattr(cfg, "reabsorb_radial_tol_mm", 1.5))
    angle_tol_deg = float(getattr(cfg, "reabsorb_angle_tol_deg", 5.0))
    cos_tol = float(np.cos(np.deg2rad(max(0.0, angle_tol_deg))))
    already = set(int(v) for v in (already_absorbed_ids or set()))
    out: list[int] = []
    for atom in atom_pool or []:
        aid = int(atom.get("atom_id", -1))
        if aid < 0 or aid in already:
            continue
        center = np.asarray(atom.get("center_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
        delta = center - fit.center
        axial = float(delta @ fit.axis)
        radial = np.linalg.norm(delta - axial * fit.axis)
        if float(radial) > radial_tol:
            continue
        kind = str(atom.get("kind") or "")
        axis_reliable = bool(atom.get("axis_reliable", False))
        if axis_reliable and kind == "line":
            atom_axis_raw = np.asarray(atom.get("axis_ras") or [0.0, 0.0, 1.0], dtype=float).reshape(3)
            atom_axis = _unit(atom_axis_raw)
            cos_ang = abs(float(atom_axis @ fit.axis))
            if cos_ang < cos_tol:
                continue
        out.append(aid)
    return out


def atom_perp_to_line(
    atom: dict[str, Any],
    fit: AxisFit,
) -> float:
    """Median perpendicular distance of an atom's support points from
    the fit line. Returns ``inf`` if the atom has no points."""
    pts = np.asarray(atom.get("support_points_ras") or [], dtype=float).reshape(-1, 3)
    if pts.shape[0] == 0:
        return float("inf")
    radial = _project_radial(pts, fit.center, fit.axis)
    return float(np.median(radial))


def _refit_subset(
    keep: list[int],
    atom_by_id: dict[int, dict[str, Any]],
    seed_axis: np.ndarray,
) -> AxisFit | None:
    cloud = []
    for aid in keep:
        atom = atom_by_id.get(int(aid))
        if atom is None:
            continue
        pts = np.asarray(atom.get("support_points_ras") or [], dtype=float).reshape(-1, 3)
        if pts.size:
            cloud.append(pts)
    if not cloud:
        return None
    combined = np.concatenate(cloud, axis=0)
    return refine_axis_from_cloud(combined, seed_axis=seed_axis)


def prune_outliers_loo(
    atom_ids: list[int],
    atom_by_id: dict[int, dict[str, Any]],
    fit: AxisFit,
    cfg: Any,
    *,
    min_keep: int = 1,
    good_residual_mm: float | None = None,
    improvement_factor: float = 0.5,
) -> tuple[list[int], AxisFit]:
    """Greedy leave-one-out outlier pruning by refit residual.

    At each iteration, try removing each remaining atom in turn,
    refit on the rest, and keep the removal that gives the largest
    residual drop. Stop when the current fit's residual is already
    good (< ``good_residual_mm``), when no removal gives at least
    ``1 - improvement_factor`` proportional improvement, or when
    only ``min_keep`` atoms are left.

    More robust than per-atom perpendicular distance pruning when the
    bad atom contributes evenly to the combined fit (symmetric
    bridges, 2-atom proposals where one atom is noise). The
    perpendicular-distance approach fails on those because both
    atoms get similar median perp values.

    Returns ``(kept_ids, refit_fit)``.
    """
    tol = float(getattr(cfg, "axis_fit_max_residual_mm", 1.8))
    good = float(good_residual_mm) if good_residual_mm is not None else 0.5 * tol
    keep = [int(a) for a in atom_ids]
    cur_fit = fit
    while len(keep) > int(min_keep):
        if cur_fit.residual_median_mm <= good:
            break
        best_subset: list[int] | None = None
        best_fit: AxisFit | None = None
        best_res = cur_fit.residual_median_mm
        for i, _excluded in enumerate(keep):
            subset = keep[:i] + keep[i + 1:]
            test_fit = _refit_subset(subset, atom_by_id, cur_fit.axis)
            if test_fit is None:
                continue
            if test_fit.residual_median_mm < best_res:
                best_res = test_fit.residual_median_mm
                best_subset = subset
                best_fit = test_fit
        if best_subset is None or best_fit is None:
            break
        if best_res >= cur_fit.residual_median_mm * improvement_factor:
            break
        keep = best_subset
        cur_fit = best_fit
    return keep, cur_fit


def prune_outlier_atoms(
    atom_ids: list[int],
    atom_by_id: dict[int, dict[str, Any]],
    fit: AxisFit,
    cfg: Any,
    *,
    perp_tol_mm: float | None = None,
    min_keep: int = 1,
) -> list[int]:
    """Iteratively drop atoms whose points sit far off the fit line.

    A bridged proposal — one whose ``atom_id_list`` mixes atoms from
    two parallel shanks, or one clean atom with a stray noise atom —
    produces a PCA fit that splits the difference between the two
    lines. Each individual atom's points then sit well off the
    resulting fit. By measuring per-atom perpendicular distance and
    dropping the worst outlier when it's above the tolerance, the fit
    can be recovered to one of the two real shanks.

    Drops the worst atom whenever ``worst > tol`` (absolute rule),
    which works for any cluster size including 2-atom proposals where
    statistical outlier detection breaks down. Stops when ``worst <=
    tol`` or when only ``min_keep`` atoms remain.

    Returns the kept atom ID list. Refitting is the caller's job.
    """
    tol = float(perp_tol_mm) if perp_tol_mm is not None else float(
        getattr(cfg, "axis_fit_max_residual_mm", 1.8)
    )
    keep = [int(a) for a in atom_ids]
    while len(keep) > int(min_keep):
        scores = []
        for aid in keep:
            atom = atom_by_id.get(int(aid))
            if atom is None:
                scores.append(float("inf"))
                continue
            scores.append(atom_perp_to_line(atom, fit))
        if not scores:
            break
        worst_idx = int(np.argmax(scores))
        worst = float(scores[worst_idx])
        if worst <= tol:
            break
        keep.pop(worst_idx)
        # Refit on remaining atoms' point clouds
        cloud = []
        for aid in keep:
            atom = atom_by_id.get(int(aid))
            if atom is None:
                continue
            pts = np.asarray(atom.get("support_points_ras") or [], dtype=float).reshape(-1, 3)
            if pts.size:
                cloud.append(pts)
        if not cloud:
            break
        combined = np.concatenate(cloud, axis=0)
        refined = refine_axis_from_cloud(combined, seed_axis=fit.axis)
        if refined is None:
            break
        fit = refined
    return keep


def gather_atom_cloud(
    atom_pool: list[dict[str, Any]],
    atom_ids: Sequence[int],
) -> np.ndarray:
    """Concatenate ``support_points_ras`` across the given atom IDs."""
    wanted = set(int(v) for v in atom_ids or [])
    bucket: list[np.ndarray] = []
    for atom in atom_pool or []:
        if int(atom.get("atom_id", -1)) not in wanted:
            continue
        pts = np.asarray(atom.get("support_points_ras") or [], dtype=float).reshape(-1, 3)
        if pts.size:
            bucket.append(pts)
    if not bucket:
        return np.zeros((0, 3), dtype=float)
    return np.concatenate(bucket, axis=0)


# ---------------------------------------------------------------------------
# Step 3 — metal profile along the refined axis
# ---------------------------------------------------------------------------

def build_metal_profile(
    points_ras: np.ndarray,
    axis: np.ndarray,
    center: np.ndarray,
    *,
    step_mm: float = 0.5,
    pad_mm: float = 0.0,
) -> tuple[float, float, np.ndarray, float]:
    """Project an atom point cloud onto ``axis`` into a boolean profile.

    Returns ``(t_min, t_max, profile, step_mm)``. The profile bin at
    index ``i`` covers ``[t_min + i*step_mm, t_min + (i+1)*step_mm)``.
    """
    a = _unit(axis)
    c = np.asarray(center, dtype=float).reshape(3)
    axial = _project_axial(points_ras, c, a)
    if axial.size == 0:
        return 0.0, 0.0, np.zeros((0,), dtype=bool), float(step_mm)
    t_min = float(np.min(axial) - pad_mm)
    t_max = float(np.max(axial) + pad_mm)
    n = int(max(1, np.ceil((t_max - t_min) / max(1e-6, step_mm))))
    profile = np.zeros((n,), dtype=bool)
    bins = np.floor((axial - t_min) / max(1e-6, step_mm)).astype(int)
    bins = np.clip(bins, 0, n - 1)
    profile[bins] = True
    return t_min, t_max, profile, float(step_mm)


# ---------------------------------------------------------------------------
# Step 4 — metal-mask extension
# ---------------------------------------------------------------------------

def _tube_offsets(axis: np.ndarray, radius_mm: float, n_radial: int = 4) -> np.ndarray:
    """Sample offsets covering a short perpendicular tube of given radius.

    Returns an ``(M, 3)`` array of offset vectors (including the axis
    center as the first row). ``n_radial`` angular samples on a single
    ring is enough for a ~1-2mm tube; no radial layers.
    """
    if radius_mm <= 0.0 or n_radial <= 0:
        return np.zeros((1, 3), dtype=float)
    u, v = _perp_frame(axis)
    angles = np.linspace(0.0, 2.0 * np.pi, int(n_radial), endpoint=False)
    ring = np.stack(
        [float(radius_mm) * (np.cos(a) * u + np.sin(a) * v) for a in angles],
        axis=0,
    )
    return np.concatenate([np.zeros((1, 3), dtype=float), ring], axis=0)


def _metal_present_at(
    t: float,
    center: np.ndarray,
    axis: np.ndarray,
    offsets: np.ndarray,
    metal_mask_kji: np.ndarray,
    ras_to_ijk_fn: Callable[[np.ndarray], Sequence[float]],
) -> bool:
    p0 = center + axis * float(t)
    samples = p0.reshape(1, 3) + offsets
    hits = _sample_mask_nearest(metal_mask_kji, ras_to_ijk_fn, samples)
    return bool(np.any(hits))


def _head_distance_at(
    t: float,
    center: np.ndarray,
    axis: np.ndarray,
    head_distance_map_kji: np.ndarray | None,
    ras_to_ijk_fn: Callable[[np.ndarray], Sequence[float]],
) -> float:
    if head_distance_map_kji is None:
        return float("inf")
    p = (center + axis * float(t)).reshape(1, 3)
    vals = _sample_array_nearest(head_distance_map_kji, ras_to_ijk_fn, p, fill=np.nan)
    v = float(vals[0])
    if not np.isfinite(v):
        return float("-inf")
    return v


def sample_head_distance_profile(
    fit: AxisFit,
    t_values: np.ndarray,
    head_distance_map_kji: np.ndarray | None,
    ras_to_ijk_fn: Callable[[np.ndarray], Sequence[float]],
) -> np.ndarray:
    """Sample ``head_distance`` at each axial offset ``t`` along the axis.

    Returns an array of the same length as ``t_values``. Out-of-volume
    samples read as ``-inf``.
    """
    out = np.full((len(t_values),), float("-inf"), dtype=float)
    if head_distance_map_kji is None:
        return out
    t_arr = np.asarray(t_values, dtype=float).reshape(-1)
    for idx, t in enumerate(t_arr.tolist()):
        p = fit.center + fit.axis * float(t)
        vals = _sample_array_nearest(
            head_distance_map_kji, ras_to_ijk_fn, p.reshape(1, 3), fill=np.nan
        )
        v = float(vals[0])
        if np.isfinite(v):
            out[idx] = v
    return out


def find_brain_entry_from_outside(
    classes: np.ndarray,
    t_values: np.ndarray,
    *,
    smoothing_window: int = 3,
    min_brain_run: int = 3,
) -> float | None:
    """Walk the lateral-HU class array from the SHALLOW end (highest
    ``t``) toward the deep end and return the first sample where we
    enter a sustained brain run.

    "Brain entry from outside" is anatomically meaningful: walking
    inward from the bolt tip, we pass through air → bolt/skull →
    brain. The shallowest sample whose lateral ring is brain — and
    that has at least ``min_brain_run`` more brain samples deeper
    than it — marks the shank's entry into the intracranial brain
    compartment. Air, bone, and metal samples shallower than this
    point are skipped.

    This avoids the calibrated head_distance threshold: we let the
    surrounding tissue tell us where brain begins, instead of
    assuming a fixed offset from the scalp.

    Returns ``None`` if no sustained brain run is found.
    """
    if classes.size == 0:
        return None
    smoothed = _smooth_classes(np.asarray(classes, dtype=np.int8), smoothing_window)
    t_arr = np.asarray(t_values, dtype=float).reshape(-1)
    n = int(smoothed.size)
    need = int(max(1, min_brain_run))
    # Walk shallow → deep (high t → low t).
    for idx in range(n - 1, -1, -1):
        if int(smoothed[idx]) != BRAIN:
            continue
        # Require at least ``need`` brain samples at idx and deeper
        # (lower indices, since classes are indexed deep→shallow by
        # construction in classify_tissue_along_axis).
        lo = max(0, idx - (need - 1))
        window = smoothed[lo:idx + 1]
        if int(np.count_nonzero(window == BRAIN)) >= need:
            return float(t_arr[idx])
    return None


def find_intracranial_exit_by_head_distance(
    t_values: np.ndarray,
    head_distance_profile: np.ndarray,
    *,
    threshold_mm: float = 5.0,
) -> float | None:
    """Find the shallow-side ``t`` at which ``head_distance`` drops
    below ``threshold_mm``.

    Walks deep→shallow. Returns the first ``t`` whose head_distance is
    ``< threshold_mm`` AND stays below it for the rest of the walk.
    This marks the point where the axis is leaving the intracranial
    compartment (burr hole, dura, cortex, whichever exits first).
    Returns ``None`` if the axis never drops below the threshold.
    """
    t_arr = np.asarray(t_values, dtype=float).reshape(-1)
    hd = np.asarray(head_distance_profile, dtype=float).reshape(-1)
    if t_arr.size == 0:
        return None
    below_threshold = hd < float(threshold_mm)
    if not np.any(below_threshold):
        return None
    # Walk from shallow back to deep; find the last index that is
    # STILL above threshold. The interface is the next sample after
    # that (or the first sample if all are below).
    last_above = -1
    for idx in range(int(t_arr.size) - 1, -1, -1):
        if hd[idx] >= float(threshold_mm):
            last_above = idx
            break
    interface_idx = last_above + 1
    if interface_idx >= int(t_arr.size):
        return None
    return float(t_arr[interface_idx])


EXIT_SCALP = "exit_scalp"
EXIT_GAP = "exit_gap"
EXIT_VOLUME = "exit_volume"


@dataclass
class ExtensionWalk:
    last_metal_t: float
    reason: str
    walked_mm: float
    min_head_distance: float  # smallest hd value sampled along the walk


def walk_metal_mask(
    fit: AxisFit,
    start_t: float,
    direction: int,
    metal_mask_kji: np.ndarray | None,
    head_distance_map_kji: np.ndarray | None,
    ras_to_ijk_fn: Callable[[np.ndarray], Sequence[float]],
    cfg: Any,
) -> ExtensionWalk:
    """Walk from ``start_t`` along ``direction * axis``, extending while
    metal is present within a narrow perpendicular tube.

    ``direction`` is ``+1`` or ``-1``. Termination reason is one of
    ``EXIT_SCALP`` (head-distance floor crossed), ``EXIT_GAP`` (a
    contiguous empty run reached ``extension_termination_gap_mm``), or
    ``EXIT_VOLUME`` (axis left the sampled volume).
    """
    step = float(getattr(cfg, "extension_step_mm", 0.5))
    tube_radius = float(getattr(cfg, "extension_tube_radius_mm", 1.5))
    max_gap = float(getattr(cfg, "extension_max_gap_mm", 3.0))
    term_gap = float(getattr(cfg, "extension_termination_gap_mm", 5.0))
    floor = float(getattr(cfg, "extension_head_distance_floor_mm", -1.0))
    offsets = _tube_offsets(fit.axis, tube_radius, n_radial=6)

    if metal_mask_kji is None:
        return ExtensionWalk(float(start_t), EXIT_GAP, 0.0, float("inf"))
    t = float(start_t)
    last_metal_t = float(start_t)
    gap_run = 0.0
    max_walk_mm = 200.0
    walked = 0.0
    reason = EXIT_GAP
    min_hd = float("inf")
    while walked < max_walk_mm:
        t = t + float(direction) * step
        walked += step
        hd = _head_distance_at(t, fit.center, fit.axis, head_distance_map_kji, ras_to_ijk_fn)
        if hd == float("-inf"):
            reason = EXIT_VOLUME
            break
        if hd < min_hd:
            min_hd = hd
        if hd < floor:
            reason = EXIT_SCALP
            break
        present = _metal_present_at(
            t, fit.center, fit.axis, offsets, metal_mask_kji, ras_to_ijk_fn
        )
        if present:
            last_metal_t = t
            gap_run = 0.0
            continue
        gap_run += step
        if gap_run >= term_gap:
            reason = EXIT_GAP
            break
    return ExtensionWalk(float(last_metal_t), reason, float(walked), float(min_hd))


def extend_axis_along_mask(
    fit: AxisFit,
    metal_mask_kji: np.ndarray | None,
    head_distance_map_kji: np.ndarray | None,
    ras_to_ijk_fn: Callable[[np.ndarray], Sequence[float]],
    cfg: Any,
) -> tuple[float, float]:
    """Convenience wrapper returning ``(t_deep_extended, t_shallow_extended)``.

    "Deep" is the ``-axis`` direction; "shallow" is ``+axis``. The caller
    is responsible for orienting the axis via :func:`orient_axis_by_scalp_exit`.
    """
    deep_walk = walk_metal_mask(
        fit, fit.t_min, -1, metal_mask_kji, head_distance_map_kji, ras_to_ijk_fn, cfg
    )
    shallow_walk = walk_metal_mask(
        fit, fit.t_max, +1, metal_mask_kji, head_distance_map_kji, ras_to_ijk_fn, cfg
    )
    return float(deep_walk.last_metal_t), float(shallow_walk.last_metal_t)


def orient_axis_by_scalp_exit(
    fit: AxisFit,
    metal_mask_kji: np.ndarray | None,
    head_distance_map_kji: np.ndarray | None,
    ras_to_ijk_fn: Callable[[np.ndarray], Sequence[float]],
    cfg: Any,
) -> tuple[AxisFit, bool]:
    """Orient the axis so that ``+axis`` points toward the bolt / scalp.

    Walks the metal mask in both directions and picks the orientation
    where at least one walk terminates with ``EXIT_SCALP``. If both
    directions exit the scalp, the orientation with the longer walked
    distance wins (longer bolt protrusion ⇒ more convincing shallow
    end). If neither direction reaches the scalp, the axis is left
    unchanged.

    Returns ``(fit_or_flipped, has_scalp_exit)``. ``has_scalp_exit`` is
    True iff at least one direction's metal-mask walk terminated with
    ``EXIT_SCALP``. Real electrodes always reach the scalp through
    their bolt; a fit without a scalp exit is either fully inside
    brain (not an electrode) or has the axis pointing so badly that
    the walk can't find the bolt (e.g. a bridged proposal spanning
    two parallel shanks).
    """
    plus = walk_metal_mask(
        fit, fit.t_max, +1, metal_mask_kji, head_distance_map_kji, ras_to_ijk_fn, cfg
    )
    minus = walk_metal_mask(
        fit, fit.t_min, -1, metal_mask_kji, head_distance_map_kji, ras_to_ijk_fn, cfg
    )
    # Scalp-exit detection: a walk "found the bolt" if either it
    # explicitly terminated past the scalp (EXIT_SCALP) OR its
    # exploration reached a head_distance below the scalp-detection
    # threshold at any point along the walk. The latter rescues
    # cases where the metal mask is sparse near the bolt entry: the
    # walk terminates on a gap a few mm short of the skin, but the
    # walk's footprint clearly passed through the scalp region.
    scalp_detect = float(getattr(cfg, "scalp_exit_detect_head_distance_mm", 5.0))
    plus_reached_scalp = plus.min_head_distance < scalp_detect
    minus_reached_scalp = minus.min_head_distance < scalp_detect
    plus_exits = plus.reason == EXIT_SCALP or plus_reached_scalp
    minus_exits = minus.reason == EXIT_SCALP or minus_reached_scalp
    has_scalp_exit = plus_exits or minus_exits
    flip = False
    if plus_exits and minus_exits:
        flip = minus.walked_mm > plus.walked_mm
    elif minus_exits and not plus_exits:
        flip = True
    if not flip:
        return fit, has_scalp_exit
    return (
        AxisFit(
            center=fit.center.copy(),
            axis=-fit.axis,
            residual_rms_mm=fit.residual_rms_mm,
            residual_median_mm=fit.residual_median_mm,
            elongation=fit.elongation,
            t_min=-fit.t_max,
            t_max=-fit.t_min,
            point_count=fit.point_count,
        ),
        has_scalp_exit,
    )


# ---------------------------------------------------------------------------
# Step 5 — lateral HU tissue classification
# ---------------------------------------------------------------------------

AIR = 0
BRAIN = 1
BONE = 2
METAL = 3

_CLASS_NAMES = {AIR: "air", BRAIN: "brain", BONE: "bone", METAL: "metal"}


def _classify_hu(median_hu: float, cfg: Any) -> int:
    air_max = float(getattr(cfg, "hu_air_max", -500.0))
    brain_max = float(getattr(cfg, "hu_brain_max", 150.0))
    bone_max = float(getattr(cfg, "hu_bone_max", 1800.0))
    if median_hu < air_max:
        return AIR
    if median_hu < brain_max:
        return BRAIN
    if median_hu < bone_max:
        return BONE
    return METAL


def classify_tissue_along_axis(
    fit: AxisFit,
    t_values: np.ndarray,
    arr_kji: np.ndarray,
    ras_to_ijk_fn: Callable[[np.ndarray], Sequence[float]],
    cfg: Any,
) -> np.ndarray:
    """Classify each axial sample using a lateral HU ring median.

    Returns an int array with one class code per ``t`` value. Samples
    whose ring is entirely out-of-volume are marked ``AIR``.
    """
    ring_radius = float(getattr(cfg, "lateral_hu_ring_radius_mm", 3.5))
    ring_samples = int(getattr(cfg, "lateral_hu_ring_samples", 8))
    u, v = _perp_frame(fit.axis)
    angles = np.linspace(0.0, 2.0 * np.pi, max(3, ring_samples), endpoint=False)
    ring = np.stack(
        [ring_radius * (np.cos(a) * u + np.sin(a) * v) for a in angles],
        axis=0,
    )
    out = np.zeros((len(t_values),), dtype=np.int8)
    for idx, t in enumerate(np.asarray(t_values, dtype=float).reshape(-1).tolist()):
        center_pt = fit.center + fit.axis * float(t)
        samples_ras = center_pt.reshape(1, 3) + ring
        vals = _sample_array_nearest(arr_kji, ras_to_ijk_fn, samples_ras, fill=np.nan)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            out[idx] = AIR
            continue
        out[idx] = _classify_hu(float(np.median(finite)), cfg)
    return out


def _smooth_classes(classes: np.ndarray, window: int) -> np.ndarray:
    """Replace runs shorter than ``window`` with their right-hand neighbour
    class. This is a one-pass smoother that absorbs single-voxel noise
    without blurring real transitions.
    """
    w = int(max(1, window))
    if w <= 1 or classes.size == 0:
        return classes.copy()
    out = classes.copy()
    n = out.size
    i = 0
    while i < n:
        j = i
        while j < n and out[j] == out[i]:
            j += 1
        run_len = j - i
        if run_len < w and i > 0 and j < n:
            out[i:j] = out[j]
        i = j
    return out


def find_bone_brain_interface(
    classes: np.ndarray,
    t_values: np.ndarray,
    *,
    smoothing_window: int = 3,
    min_brain_run: int = 3,
    sustained_bone_run: int = 3,
    require_exit_after_bone: bool = True,
) -> float | None:
    """Find the first sustained brain→bone transition walking deep→shallow.

    A valid interface requires:

    - at least ``min_brain_run`` contiguous brain samples (air counts
      as neutral, tolerating sinus intrusions) immediately before the
      transition,
    - at least ``sustained_bone_run`` bone samples immediately after,
      and
    - when ``require_exit_after_bone`` is True, the bone run must end
      in air or the end of the sampled range (i.e., the axis must
      actually *exit* past the skull). This rejects beam-hardening
      streaks around contact metal, which show as short bone runs
      surrounded by brain on both sides.

    Returns the ``t`` at the first qualifying bone sample, or the
    shallowest brain sample when no transition qualifies (the shank
    never exits brain in the sampled range).
    """
    if classes.size == 0:
        return None
    smoothed = _smooth_classes(np.asarray(classes, dtype=np.int8), smoothing_window)
    t_arr = np.asarray(t_values, dtype=float).reshape(-1)
    n = int(smoothed.size)
    need_brain = int(max(1, min_brain_run))
    need_bone = int(max(1, sustained_bone_run))
    brain_run = 0
    idx = 0
    while idx < n:
        c = int(smoothed[idx])
        if c == BRAIN:
            brain_run += 1
            idx += 1
            continue
        if c == AIR:
            idx += 1
            continue
        if c == BONE and brain_run >= need_brain:
            # Measure the contiguous bone run (air tolerated inside).
            run_end = idx
            while run_end < n and int(smoothed[run_end]) in (BONE, AIR):
                run_end += 1
            bone_len = int(np.count_nonzero(smoothed[idx:run_end] == BONE))
            if bone_len >= need_bone:
                exits = (run_end >= n) or int(smoothed[run_end]) != BRAIN
                if exits or not require_exit_after_bone:
                    return float(t_arr[idx])
            # Spurious bone streak: skip it and continue the brain run.
            idx = run_end
            continue
        if c == BONE:
            brain_run = 0
            idx += 1
            continue
        # METAL — reset.
        brain_run = 0
        idx += 1
    # No qualifying transition found. Return the shallowest brain
    # sample so short in-brain shanks still produce an endpoint.
    brain_idxs = np.flatnonzero(smoothed == BRAIN)
    if brain_idxs.size == 0:
        return None
    return float(t_arr[int(brain_idxs[-1])])


def deep_end_of_brain_run(
    classes: np.ndarray,
    t_values: np.ndarray,
    interface_t: float,
    *,
    smoothing_window: int = 3,
) -> float | None:
    """Return the deepest brain sample in the contiguous brain run
    that ends just deep of ``interface_t``.

    Walks from the sample at ``interface_t`` (exclusive) deeper while
    the class remains brain or air (tolerating sinus intrusions).
    Used to pin the deep intracranial endpoint on observed brain
    rather than whatever the metal-mask extension happened to reach.
    """
    if classes.size == 0:
        return None
    smoothed = _smooth_classes(np.asarray(classes, dtype=np.int8), smoothing_window)
    t_arr = np.asarray(t_values, dtype=float).reshape(-1)
    # Find index at or just below interface_t.
    idx_arr = np.searchsorted(t_arr, float(interface_t), side="left")
    idx = int(idx_arr) - 1
    if idx < 0:
        return None
    while idx >= 0 and int(smoothed[idx]) not in (BRAIN, AIR):
        idx -= 1
    if idx < 0:
        return None
    deepest = idx
    while idx >= 0 and int(smoothed[idx]) in (BRAIN, AIR):
        if int(smoothed[idx]) == BRAIN:
            deepest = idx
        idx -= 1
    return float(t_arr[deepest])


def intracranial_brain_span_mm(
    classes: np.ndarray,
    step_mm: float,
) -> float:
    """Longest contiguous run of brain-classified samples, expressed in mm."""
    if classes.size == 0:
        return 0.0
    best = 0
    run = 0
    for c in classes.tolist():
        if int(c) == BRAIN:
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return float(best) * float(step_mm)


# ---------------------------------------------------------------------------
# Step 6 — library span recognizer gate
# ---------------------------------------------------------------------------

def library_span_match(
    intracranial_span_mm: float,
    library_models: list[dict[str, Any]],
    tolerance_mm: float,
) -> tuple[str | None, float]:
    """Return (best_model_id, best_abs_delta_mm) whose span is closest
    to ``intracranial_span_mm``. If the closest delta exceeds
    ``tolerance_mm`` the model id is still returned (the caller decides
    whether to reject); the second element lets the caller gate on it.
    """
    best_id: str | None = None
    best_delta = float("inf")
    for model in library_models or []:
        span = _library_model_span(model)
        if span is None:
            continue
        delta = abs(float(span) - float(intracranial_span_mm))
        if delta < best_delta:
            best_delta = delta
            best_id = str(model.get("id", ""))
    return best_id, float(best_delta) if best_id is not None else float("inf")


def _library_model_span(model: dict[str, Any]) -> float | None:
    for key in ("total_exploration_length_mm", "total_span_mm", "span_mm"):
        v = model.get(key)
        if v is not None:
            return float(v)
    return None


def library_model_contact_span_mm(model: dict[str, Any]) -> float:
    """Return the span from deepest contact center to shallowest contact
    center for a library model. This is what the evaluation metric
    measures — GT ``start_ras`` / ``end_ras`` are contact centers, not
    the skull boundary.
    """
    offsets = model.get("contact_center_offsets_from_tip_mm")
    if offsets:
        vals = [float(v) for v in offsets]
        if len(vals) >= 2:
            return float(max(vals) - min(vals))
    span = _library_model_span(model)
    return float(span) if span is not None else 0.0


def library_model_first_contact_offset_mm(model: dict[str, Any]) -> float:
    """Distance from the electrode tip to the center of the deepest contact."""
    offsets = model.get("contact_center_offsets_from_tip_mm")
    if offsets:
        vals = [float(v) for v in offsets]
        if vals:
            return float(min(vals))
    return 0.0


def library_span_range(library_models: list[dict[str, Any]]) -> tuple[float, float]:
    spans: list[float] = []
    for model in library_models or []:
        span = _library_model_span(model)
        if span is None:
            continue
        spans.append(float(span))
    if not spans:
        return 0.0, 0.0
    return float(min(spans)), float(max(spans))
