"""Phase B: electrode-template group fitting for Deep Core proposals.

Replaces the geometric extension stage. For each candidate proposal,
slides each library electrode model along the proposal axis and scores
per-contact HU samples in the full CT volume. Selects a globally
non-conflicting set of (proposal, model, anchor) assignments via greedy
score-ordered allocation. Hard-rejects proposals with no model fit.

Design notes
------------
- Sampling uses the raw ``arr_kji`` (not the deep_core-filtered metal
  mask) so contacts in the superficial 0-15mm zone are visible.
- Scoring is restricted to contacts inside the brain (head_distance ≥
  ``in_brain_min_depth_mm``).  Bolt-induced false positives are
  prevented by requiring at least ``min_in_brain_contacts`` (default 6)
  in-brain hits — bolts have zero in-brain extent.
- Sort key: ``(n_in_brain_hits, -contact_count, raw_hu_sum)`` so that
  the *shortest* model that explains the in-brain portion wins ties.
- Conflict radius (default 2 mm) prevents two fits from claiming the
  same physical contact.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# HU sampling helpers
# ---------------------------------------------------------------------------

def _orthonormal_basis(axis_ras: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.asarray(axis_ras, dtype=float).reshape(3)
    n = float(np.linalg.norm(axis))
    if n <= 1e-9:
        axis = np.array([0.0, 0.0, 1.0])
        n = 1.0
    axis = axis / n
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(axis, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, ref)
    u = u / max(float(np.linalg.norm(u)), 1e-9)
    v = np.cross(axis, u)
    v = v / max(float(np.linalg.norm(v)), 1e-9)
    return axis, u, v


def _sample_hu_at_ras(
    arr_kji: np.ndarray,
    ras_to_ijk_fn,
    ras_point: np.ndarray,
    radius_vox: int,
    *,
    reduction: str = "max",
) -> float:
    """Sample HU around ras_point.

    radius_vox=0 → point sample (single voxel).
    Otherwise samples a (2*r+1)^3 box and returns max (default) or mean.
    'max' is good for contact-finding (catches a bright voxel anywhere
    in the box even with slight axis offset). 'mean' is good for lateral
    perimeter sampling (avoids single bright voxels dominating).
    """
    ijk = ras_to_ijk_fn(ras_point)
    i, j, k = int(round(float(ijk[0]))), int(round(float(ijk[1]))), int(round(float(ijk[2])))
    K, J, I = arr_kji.shape
    if not (0 <= k < K and 0 <= j < J and 0 <= i < I):
        return float("nan")
    if radius_vox <= 0:
        return float(arr_kji[k, j, i])
    k0, k1 = max(0, k - radius_vox), min(K, k + radius_vox + 1)
    j0, j1 = max(0, j - radius_vox), min(J, j + radius_vox + 1)
    i0, i1 = max(0, i - radius_vox), min(I, i + radius_vox + 1)
    box = arr_kji[k0:k1, j0:j1, i0:i1]
    if reduction == "mean":
        return float(np.mean(box))
    return float(np.max(box))


def _sample_hu_at_positions(
    arr_kji: np.ndarray,
    ras_to_ijk_fn,
    positions_ras: np.ndarray,
    radius_vox: int,
) -> np.ndarray:
    out = np.zeros(positions_ras.shape[0], dtype=float)
    for idx in range(positions_ras.shape[0]):
        out[idx] = _sample_hu_at_ras(arr_kji, ras_to_ijk_fn, positions_ras[idx], radius_vox)
    return out


def _axis_segment_median_hu(
    arr_kji: np.ndarray,
    ras_to_ijk_fn,
    deepest_ras: np.ndarray,
    shallowest_ras: np.ndarray,
    *,
    step_mm: float = 0.5,
) -> float:
    """Sample HU at fine intervals along the axis segment from deepest
    to shallowest contact. Returns the MEDIAN.

    For a real electrode the segment is mostly metal-or-insulation
    (median ~800-2000+ HU). For a false fit through tissue with
    random bright spots at contact positions, the median is ~30 HU
    (brain) or ~100 HU (CSF).
    """
    seg = shallowest_ras - deepest_ras
    span = float(np.linalg.norm(seg))
    if span <= 1e-6:
        return float("nan")
    n = max(2, int(round(span / float(step_mm))) + 1)
    samples = []
    for i in range(n):
        t = i / float(n - 1)
        pos = deepest_ras + seg * t
        hu = _sample_hu_at_ras(arr_kji, ras_to_ijk_fn, pos, 0)  # point sample
        if np.isfinite(hu):
            samples.append(hu)
    if not samples:
        return float("nan")
    return float(np.median(samples))


def _check_lateral_profile(
    arr_kji: np.ndarray,
    ras_to_ijk_fn,
    positions_ras: np.ndarray,
    axis_ras: np.ndarray,
    *,
    lateral_offset_mm: float,
    min_drop_hu: float,
    radius_vox: int,
) -> np.ndarray:
    """For each contact, check center-vs-lateral-perimeter HU drop.

    Returns boolean array (n_contacts,) where True means the position
    is centered on the proposal axis (i.e., the bright voxel is on the
    axis, not off to one side).

    Implementation notes:
    - Center sample uses a box (max) to catch contacts slightly off-axis.
    - Lateral samples use POINT samples (no box) so they don't catch
      voxels of the central contact via box overlap.
    - Lateral offset must be larger than ``radius_vox`` voxels to avoid
      box overlap with the center.
    """
    _axis, u, v = _orthonormal_basis(axis_ras)
    n = positions_ras.shape[0]
    ok = np.zeros(n, dtype=bool)
    for idx in range(n):
        center = positions_ras[idx]
        center_hu = _sample_hu_at_ras(arr_kji, ras_to_ijk_fn, center, radius_vox, reduction="max")
        if not np.isfinite(center_hu):
            continue
        lateral_max = -np.inf
        for direction in (u, -u, v, -v):
            p = center + direction * lateral_offset_mm
            # Point sample (radius_vox=0) so we don't pick up the central contact
            hu = _sample_hu_at_ras(arr_kji, ras_to_ijk_fn, p, 0)
            if np.isfinite(hu) and hu > lateral_max:
                lateral_max = float(hu)
        if not np.isfinite(lateral_max):
            continue
        # The on-axis sample must be substantially brighter than the lateral perimeter.
        if (center_hu - lateral_max) >= float(min_drop_hu):
            ok[idx] = True
    return ok


def _depth_at_ras(
    head_distance_map_kji: np.ndarray,
    ras_to_ijk_fn,
    ras_point,
) -> float | None:
    if head_distance_map_kji is None:
        return None
    ijk = ras_to_ijk_fn(ras_point)
    i, j, k = int(round(float(ijk[0]))), int(round(float(ijk[1]))), int(round(float(ijk[2])))
    K, J, I = head_distance_map_kji.shape
    if not (0 <= k < K and 0 <= j < J and 0 <= i < I):
        return None
    val = float(head_distance_map_kji[k, j, i])
    return val if np.isfinite(val) else None


# ---------------------------------------------------------------------------
# Library filtering
# ---------------------------------------------------------------------------

def filter_models_by_family(
    library: dict[str, Any],
    families: tuple[str, ...],
    min_contacts: int,
) -> list[dict[str, Any]]:
    """Return library models whose id starts with one of the family
    prefixes (case-insensitive) AND has ``contact_count >= min_contacts``.
    """
    if not families:
        models = list(library.get("models") or [])
    else:
        fams = tuple(f.upper().rstrip("-") + "-" for f in families)
        models = [
            m for m in (library.get("models") or [])
            if any(str(m.get("id", "")).upper().startswith(f) for f in fams)
        ]
    return [m for m in models if int(m.get("contact_count", 0)) >= int(min_contacts)]


# ---------------------------------------------------------------------------
# Per-proposal hypothesis generation
# ---------------------------------------------------------------------------

def _hypotheses_for_proposal(
    proposal: dict[str, Any],
    arr_kji: np.ndarray,
    ras_to_ijk_fn,
    head_distance_map_kji: np.ndarray,
    library_models: list[dict[str, Any]],
    cfg,
) -> list[dict[str, Any]]:
    """Build all (model, anchor) hypotheses for one proposal."""
    start = np.asarray(proposal.get("start_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
    end = np.asarray(proposal.get("end_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
    axis_vec = end - start
    axis_norm = float(np.linalg.norm(axis_vec))
    if axis_norm <= 1e-6:
        return []
    axis = axis_vec / axis_norm

    # Pick the deeper endpoint as the search anchor.
    # shallow_axis must point FROM the deep tip TOWARD the entry (shallow end).
    d_start = _depth_at_ras(head_distance_map_kji, ras_to_ijk_fn, start)
    d_end = _depth_at_ras(head_distance_map_kji, ras_to_ijk_fn, end)
    if (d_end or -1.0) >= (d_start or -1.0):
        # 'end' is deeper -> shallow direction is from end to start
        deep_seed = end
        shallow_axis = (start - end) / np.linalg.norm(start - end)
    else:
        deep_seed = start
        shallow_axis = (end - start) / np.linalg.norm(end - start)

    hyps = []
    for model in library_models:
        offsets = np.asarray(model["contact_center_offsets_from_tip_mm"], dtype=float)
        # Smallest offset = closest to physical tip = deepest contact
        deepest_offset = float(offsets.min())
        deltas = offsets - deepest_offset  # >= 0, larger = shallower

        for shift_mm in np.arange(
            -float(cfg.deep_anchor_search_mm),
            float(cfg.deep_anchor_search_mm) + 1e-6,
            float(cfg.deep_anchor_step_mm),
        ):
            deepest = deep_seed + shallow_axis * float(shift_mm)
            # Place all contacts: deepest is delta=0, shallowest is delta=max
            positions = deepest[None, :] + deltas[:, None] * shallow_axis[None, :]

            contact_hu = _sample_hu_at_positions(
                arr_kji, ras_to_ijk_fn, positions, int(cfg.sample_radius_vox)
            )
            lateral_ok = _check_lateral_profile(
                arr_kji, ras_to_ijk_fn, positions, shallow_axis,
                lateral_offset_mm=float(cfg.lateral_offset_mm),
                min_drop_hu=float(cfg.min_lateral_drop_hu),
                radius_vox=int(cfg.sample_radius_vox),
            )

            # In-brain mask per contact
            in_brain_mask = np.zeros(positions.shape[0], dtype=bool)
            for idx in range(positions.shape[0]):
                d = _depth_at_ras(head_distance_map_kji, ras_to_ijk_fn, positions[idx])
                if d is not None and d >= float(cfg.in_brain_min_depth_mm):
                    in_brain_mask[idx] = True

            n_in_brain_total = int(in_brain_mask.sum())
            if n_in_brain_total < int(cfg.min_in_brain_contacts):
                continue

            hits_mask = (contact_hu > float(cfg.hit_hu_threshold)) & lateral_ok & in_brain_mask
            n_in_brain_hits = int(hits_mask.sum())

            if n_in_brain_hits < int(cfg.min_in_brain_contacts):
                continue
            if n_in_brain_total > 0 and n_in_brain_hits / n_in_brain_total < float(cfg.min_in_brain_hit_fraction):
                continue

            # Axis-segment median HU check: rejects fits whose axis
            # passes through tissue between sparse hits from unrelated
            # electrodes. A real electrode has high median HU along its
            # full contact-spanning segment.
            seg_median_hu = _axis_segment_median_hu(
                arr_kji, ras_to_ijk_fn, positions[0], positions[-1], step_mm=0.5,
            )
            if not np.isfinite(seg_median_hu):
                continue
            if seg_median_hu < float(cfg.min_axis_segment_median_hu):
                continue

            raw_hu_sum_in_brain = float(np.sum(
                np.clip(contact_hu[in_brain_mask] - float(cfg.hit_hu_threshold), 0.0, None)
            ))

            # Model span = total length from deepest to shallowest contact
            model_span_mm = float(offsets.max() - offsets.min())

            # Sort key: more in-brain hits wins (covers more real
            # electrode); shorter physical span wins ties (the user's
            # "shortest model that explains the in-brain portion" rule
            # — distinguishes DIXI-18AM 60mm vs DIXI-18CM 81mm at equal
            # hit count); brightness final tiebreaker.
            sort_key = (
                n_in_brain_hits,
                -model_span_mm,
                -int(model["contact_count"]),
                raw_hu_sum_in_brain,
            )

            hyps.append({
                "sort_key": sort_key,
                "model_id": str(model["id"]),
                "model": model,
                "n_in_brain_hits": n_in_brain_hits,
                "n_in_brain_total": n_in_brain_total,
                "n_total_contacts": int(model["contact_count"]),
                "contact_positions_ras": positions,
                "in_brain_mask": in_brain_mask,
                "deepest_contact_ras": positions[0],
                "shallowest_contact_ras": positions[-1],
                "axis_ras": shallow_axis.copy(),
                "raw_hu_sum_in_brain": raw_hu_sum_in_brain,
            })

    return hyps


# ---------------------------------------------------------------------------
# Group fitting
# ---------------------------------------------------------------------------

def run_model_fit_group(
    proposals: list[dict[str, Any]],
    arr_kji: np.ndarray,
    ras_to_ijk_fn,
    head_distance_map_kji: np.ndarray,
    library_models: list[dict[str, Any]],
    cfg,
) -> dict[str, Any]:
    """Group-fit electrode templates to a list of proposals.

    Returns a dict with:
        ``accepted_proposals``: list of (rewritten) proposal dicts that
            had a successful, non-conflicting model fit
        ``stats``: counts and per-stage diagnostics
    """
    if not proposals or not library_models:
        return {
            "accepted_proposals": [],
            "stats": {
                "input_count": int(len(proposals or [])),
                "hypotheses_generated": 0,
                "accepted_count": 0,
                "rejected_unassigned": int(len(proposals or [])),
            },
        }

    # Stage 1: generate hypotheses for every proposal
    all_hypotheses: list[tuple[int, dict[str, Any]]] = []
    for p_idx, prop in enumerate(proposals):
        for hyp in _hypotheses_for_proposal(
            prop, arr_kji, ras_to_ijk_fn, head_distance_map_kji, library_models, cfg
        ):
            all_hypotheses.append((p_idx, hyp))

    # Stage 2: greedy non-conflicting assignment
    all_hypotheses.sort(key=lambda x: x[1]["sort_key"], reverse=True)

    accepted_by_proposal: dict[int, dict[str, Any]] = {}
    accepted_contact_centers: list[list[float]] = []
    conflict_radius_sq = float(cfg.conflict_radius_mm) ** 2

    for p_idx, hyp in all_hypotheses:
        if p_idx in accepted_by_proposal:
            continue
        if accepted_contact_centers:
            accepted_arr = np.asarray(accepted_contact_centers, dtype=float)
            conflict = False
            for c in hyp["contact_positions_ras"]:
                deltas = accepted_arr - c[None, :]
                d2 = np.sum(deltas * deltas, axis=1)
                if (d2 < conflict_radius_sq).any():
                    conflict = True
                    break
            if conflict:
                continue
        accepted_by_proposal[p_idx] = hyp
        for c in hyp["contact_positions_ras"]:
            accepted_contact_centers.append([float(c[0]), float(c[1]), float(c[2])])

    # Stage 3: rewrite accepted proposals; HARD REJECT unassigned
    accepted_props: list[dict[str, Any]] = []
    for p_idx, prop in enumerate(proposals):
        if p_idx not in accepted_by_proposal:
            continue
        hyp = accepted_by_proposal[p_idx]
        new_prop = dict(prop)
        new_prop["start_ras"] = [float(v) for v in hyp["deepest_contact_ras"]]
        new_prop["end_ras"] = [float(v) for v in hyp["shallowest_contact_ras"]]
        new_prop["axis_ras"] = [float(v) for v in hyp["axis_ras"]]
        new_prop["span_mm"] = float(np.linalg.norm(
            hyp["shallowest_contact_ras"] - hyp["deepest_contact_ras"]
        ))
        new_prop["best_model_id"] = hyp["model_id"]
        new_prop["best_model_n_hits"] = int(hyp["n_in_brain_hits"])
        new_prop["best_model_in_brain_total"] = int(hyp["n_in_brain_total"])
        new_prop["best_model_contact_count"] = int(hyp["n_total_contacts"])
        new_prop["best_model_score"] = float(hyp["raw_hu_sum_in_brain"])
        new_prop["model_contact_positions_ras"] = [
            [float(v) for v in p] for p in hyp["contact_positions_ras"]
        ]
        new_prop["model_in_brain_mask"] = [bool(b) for b in hyp["in_brain_mask"]]
        new_prop["model_fit_passed"] = True
        accepted_props.append(new_prop)

    return {
        "accepted_proposals": accepted_props,
        "stats": {
            "input_count": int(len(proposals)),
            "hypotheses_generated": int(len(all_hypotheses)),
            "accepted_count": int(len(accepted_props)),
            "rejected_unassigned": int(len(proposals) - len(accepted_props)),
        },
    }
