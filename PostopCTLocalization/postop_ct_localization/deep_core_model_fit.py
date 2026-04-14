"""Phase B: trajectory reconstruction via axis refinement and
metal-mask extension.

See ``docs/PHASE_B_REDESIGN.md`` for the full design rationale. The
high-level flow for each incoming proposal is:

    1. Refine the axis via PCA on the proposal's point cloud.
    2. Reabsorb any colinear atoms from the full atom pool.
    3. Build a metal profile along the refined axis.
    4. Extend the axis along the raw metal mask past the deep_core
       shrink rind, stopping at a sustained empty run or when the axis
       crosses the scalp into external air.
    5. Classify tissue at lateral HU rings to locate the bone↔brain
       interface (the clinically meaningful shallow endpoint).
    6. Apply rejection gates (intracranial extent, library span).
    7. Resolve group conflicts by refined-axis proximity.
    8. Emit the trajectory with observed endpoints.

The library of electrode models acts as a soft recognizer gate — it
does not generate endpoints. Contact placement is handled by a later
stage.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import deep_core_axis_reconstruction as axr


# ---------------------------------------------------------------------------
# Family filtering (kept for call-site compatibility)
# ---------------------------------------------------------------------------

def filter_models_by_family(
    library: dict[str, Any],
    families: tuple[str, ...],
    min_contacts: int = 0,
) -> list[dict[str, Any]]:
    """Return library models whose id starts with one of the family
    prefixes (case-insensitive)."""
    if not families:
        models = list(library.get("models") or [])
    else:
        fams = tuple(f.upper().rstrip("-") + "-" for f in families)
        models = [
            m for m in (library.get("models") or [])
            if any(str(m.get("id", "")).upper().startswith(f) for f in fams)
        ]
    if int(min_contacts) > 0:
        models = [m for m in models if int(m.get("contact_count", 0)) >= int(min_contacts)]
    return models


# ---------------------------------------------------------------------------
# Per-proposal fit
# ---------------------------------------------------------------------------

def _gather_initial_cloud(
    prop: dict[str, Any],
    atom_by_id: dict[int, dict[str, Any]],
    blob_sample_points_ras: np.ndarray | None,
    blocked_atom_ids: set[int] | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Union of per-atom support points and the proposal's tokenized
    points. Atoms in ``blocked_atom_ids`` (already claimed by a
    higher-priority fit) are excluded from the cloud. Returns the
    cloud and the list of atom IDs that actually contributed to it.
    """
    blocked = set(int(v) for v in (blocked_atom_ids or set()))
    bucket: list[np.ndarray] = []
    used_ids: list[int] = []
    for aid in list(prop.get("atom_id_list") or []):
        aid_int = int(aid)
        if aid_int in blocked:
            continue
        atom = atom_by_id.get(aid_int)
        if atom is None:
            continue
        pts = np.asarray(atom.get("support_points_ras") or [], dtype=float).reshape(-1, 3)
        if pts.size:
            bucket.append(pts)
            used_ids.append(aid_int)
    if blob_sample_points_ras is not None:
        tok_idx = list(prop.get("token_indices") or [])
        if tok_idx:
            tok_arr = np.asarray(blob_sample_points_ras, dtype=float).reshape(-1, 3)
            sel = np.asarray(tok_idx, dtype=int)
            sel = sel[(sel >= 0) & (sel < tok_arr.shape[0])]
            if sel.size:
                bucket.append(tok_arr[sel])
    if not bucket:
        return np.zeros((0, 3), dtype=float), used_ids
    return np.concatenate(bucket, axis=0), used_ids


def _seed_axis_from_proposal(prop: dict[str, Any]) -> np.ndarray:
    """Prefer the proposal's published axis; fall back to end - start."""
    a = prop.get("axis_ras")
    if a is not None:
        arr = np.asarray(a, dtype=float).reshape(3)
        if float(np.linalg.norm(arr)) > 1e-6:
            return arr
    s = np.asarray(prop.get("start_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
    e = np.asarray(prop.get("end_ras") or [0.0, 0.0, 1.0], dtype=float).reshape(3)
    d = e - s
    if float(np.linalg.norm(d)) <= 1e-6:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return d


def _fit_one_proposal(
    prop: dict[str, Any],
    *,
    arr_kji: np.ndarray,
    metal_mask_kji: np.ndarray | None,
    head_distance_map_kji: np.ndarray | None,
    ras_to_ijk_fn,
    atom_pool: list[dict[str, Any]],
    atom_by_id: dict[int, dict[str, Any]],
    blob_sample_points_ras: np.ndarray | None,
    library_models: list[dict[str, Any]],
    library_span_bounds: tuple[float, float],
    cfg,
    blocked_atom_ids: set[int] | None = None,
) -> dict[str, Any] | None:
    """Return an accepted-trajectory dict or ``None`` if the proposal
    is rejected. ``blocked_atom_ids`` is the set of atom IDs already
    claimed by a higher-priority accepted fit; such atoms are invisible
    to both the initial cloud and reabsorption.
    """
    seed_axis = _seed_axis_from_proposal(prop)
    blocked = set(int(v) for v in (blocked_atom_ids or set()))
    initial_cloud, initial_atom_list = _gather_initial_cloud(
        prop, atom_by_id, blob_sample_points_ras, blocked_atom_ids=blocked
    )
    if initial_cloud.shape[0] < 3:
        return None

    fit = axr.refine_axis_from_cloud(initial_cloud, seed_axis=seed_axis)
    if fit is None:
        return None
    max_residual = float(getattr(cfg, "axis_fit_max_residual_mm", 1.2))

    # Step 1.5 — outlier-atom pruning. If the proposal's atom_id_list
    # bundles atoms from two parallel shanks (a bridged proposal), the
    # initial PCA fit splits the difference. Drop atoms whose points
    # sit far off the fit line and refit. This recovers RAMC-class
    # cases where Phase A's chaining grabbed a stray atom from a
    # neighbour. No-op when all atoms agree.
    if len(initial_atom_list) >= 3:
        kept_ids = axr.prune_outlier_atoms(
            initial_atom_list, atom_by_id, fit, cfg, min_keep=2
        )
        if len(kept_ids) < len(initial_atom_list):
            cloud_kept: list[np.ndarray] = []
            for aid in kept_ids:
                atom = atom_by_id.get(int(aid))
                if atom is None:
                    continue
                pts = np.asarray(
                    atom.get("support_points_ras") or [], dtype=float
                ).reshape(-1, 3)
                if pts.size:
                    cloud_kept.append(pts)
            if cloud_kept:
                pruned_cloud = np.concatenate(cloud_kept, axis=0)
                pruned_fit = axr.refine_axis_from_cloud(pruned_cloud, seed_axis=fit.axis)
                if pruned_fit is not None and pruned_fit.residual_median_mm < fit.residual_median_mm:
                    fit = pruned_fit
                    initial_cloud = pruned_cloud
                    initial_atom_list = kept_ids

    if fit.residual_median_mm > max_residual:
        return None

    # Step 2 — colinear atom reabsorption (excluding blocked atoms).
    initial_atom_ids = set(initial_atom_list)
    skip_ids = initial_atom_ids | blocked
    reabsorbed = axr.reabsorb_colinear_atoms(
        fit, atom_pool, cfg, already_absorbed_ids=skip_ids
    )
    if reabsorbed:
        extra = axr.gather_atom_cloud(atom_pool, reabsorbed)
        if extra.size:
            combined = np.concatenate([initial_cloud, extra], axis=0)
            refined = axr.refine_axis_from_cloud(combined, seed_axis=fit.axis)
            if refined is not None and refined.residual_median_mm <= max_residual:
                fit = refined
                initial_cloud = combined
    explained_atom_ids = sorted(initial_atom_ids | set(int(v) for v in reabsorbed))

    # Step 3 — metal profile (diagnostic only for now; the extension
    # step relies on the mask rather than this profile).
    t_min_cloud, t_max_cloud, _profile, _step = axr.build_metal_profile(
        initial_cloud, fit.axis, fit.center, step_mm=0.5
    )
    # Prefer the cloud-derived extent (slightly tighter than PCA
    # projected extent) for the extension seed.
    fit.t_min = float(min(fit.t_min, t_min_cloud))
    fit.t_max = float(max(fit.t_max, t_max_cloud))

    # Orient the axis so that +axis points toward the bolt / scalp.
    # A real electrode's bolt always pokes through the skin, so at
    # least one direction's metal walk must terminate with
    # EXIT_SCALP. A fit that reaches the scalp in *neither* direction
    # is either (a) not a real electrode (trajectory stays inside
    # brain) or (b) has an axis so badly tilted — typically because
    # the initial cloud bridged two parallel shanks — that the bolt
    # can't be found. Reject it either way.
    fit, has_scalp_exit = axr.orient_axis_by_scalp_exit(
        fit, metal_mask_kji, head_distance_map_kji, ras_to_ijk_fn, cfg
    )
    if not has_scalp_exit:
        return None

    # Step 4 — metal-mask extension (post-orientation).
    t_deep_ext, t_shallow_ext = axr.extend_axis_along_mask(
        fit, metal_mask_kji, head_distance_map_kji, ras_to_ijk_fn, cfg
    )

    # Step 5 — shallow intracranial endpoint via head_distance
    # threshold. This is a calibration to typical bolt drive depths
    # rather than a true physical signal — the actual contact
    # placement that defines GT_end will be done in a Phase C stage
    # and Phase B only needs to be in the right neighborhood. The
    # bolt tip itself is emitted separately as ``bolt_ras`` below
    # so Phase C can refine from it.
    step_mm = float(getattr(cfg, "extension_step_mm", 0.5))
    if t_shallow_ext - t_deep_ext < step_mm:
        return None
    t_values = np.arange(t_deep_ext, t_shallow_ext + step_mm * 0.5, step_mm)
    classes = axr.classify_tissue_along_axis(
        fit, t_values, arr_kji, ras_to_ijk_fn, cfg
    )
    hd_profile = axr.sample_head_distance_profile(
        fit, t_values, head_distance_map_kji, ras_to_ijk_fn
    )
    exit_threshold = float(getattr(cfg, "intracranial_exit_head_distance_mm", 15.0))
    t_interface = axr.find_intracranial_exit_by_head_distance(
        t_values, hd_profile, threshold_mm=exit_threshold
    )
    if t_interface is None:
        t_interface = float(t_shallow_ext)

    # Deep endpoint = the deepest observed metal voxel. Earlier
    # iterations tried to pin this to the deepest brain sample
    # preceding ``t_interface`` to defend against skull-base overshoot,
    # but that defense mis-fires when beam-hardening streaks inside
    # the contact region get classified as bone and break up the
    # deep brain run. Metal extension already bounds the deep side by
    # the actual contact bloom, which is enough.
    t_deep_intra = float(t_deep_ext)

    # Step 6 — rejection gates.
    intracranial_span = float(t_interface - t_deep_intra)
    min_span = float(getattr(cfg, "min_intracranial_span_mm", 15.0))
    if intracranial_span < min_span:
        return None
    brain_span = axr.intracranial_brain_span_mm(classes, step_mm)
    if brain_span < min_span:
        return None
    lib_min, lib_max = library_span_bounds
    span_tol = float(getattr(cfg, "library_span_tolerance_mm", 5.0))
    if lib_max > 0.0:
        if intracranial_span < lib_min - span_tol:
            return None
        if intracranial_span > lib_max + span_tol:
            # Cap the shallow endpoint at the longest library span plus
            # tolerance. The extension may have walked through a
            # noisy metal mask — trim rather than reject so a plausibly
            # correct axis still produces a trajectory.
            t_interface = float(t_deep_intra + (lib_max + span_tol))
            intracranial_span = float(t_interface - t_deep_intra)

    best_model_id, _best_delta = axr.library_span_match(
        intracranial_span, library_models, tolerance_mm=span_tol
    )

    t_start = float(t_deep_intra)
    t_end = float(t_interface)
    start_ras = fit.center + fit.axis * t_start
    end_ras = fit.center + fit.axis * t_end
    bolt_ras = fit.center + fit.axis * float(t_shallow_ext)

    out = dict(prop)
    out["start_ras"] = [float(v) for v in start_ras]
    out["end_ras"] = [float(v) for v in end_ras]
    out["bolt_ras"] = [float(v) for v in bolt_ras]
    out["axis_ras"] = [float(v) for v in fit.axis]
    out["span_mm"] = float(intracranial_span)
    out["intracranial_span_mm"] = float(intracranial_span)
    out["bolt_extent_mm"] = float(max(0.0, t_shallow_ext - t_interface))
    out["explained_atom_ids"] = list(explained_atom_ids)
    out["axis_residual_rms_mm"] = float(fit.residual_rms_mm)
    out["axis_residual_median_mm"] = float(fit.residual_median_mm)
    out["best_model_id"] = best_model_id or ""
    out["model_fit_passed"] = True
    # Drop stale fields from the old template-matching implementation.
    for stale in (
        "model_contact_positions_ras",
        "best_model_n_hits",
        "best_model_in_brain_total",
        "best_model_contact_count",
        "best_model_score",
        "model_fit_stub",
    ):
        out.pop(stale, None)
    # Internal hooks for conflict resolution.
    out["_fit_center_ras"] = fit.center.tolist()
    out["_fit_axis_ras"] = fit.axis.tolist()
    out["_fit_t_deep"] = float(t_deep_ext)
    out["_fit_t_shallow"] = float(t_shallow_ext)
    out["_fit_t_interface"] = float(t_interface)
    return out


# ---------------------------------------------------------------------------
# Step 7 — group conflict resolution
# ---------------------------------------------------------------------------

def _axes_conflict(
    a: dict[str, Any],
    b: dict[str, Any],
    radius_mm: float,
) -> bool:
    """Two fits conflict if they share an atom or their refined axes
    come within ``radius_mm`` perpendicular distance over any overlap
    of their axial extents.
    """
    ids_a = set(int(v) for v in list(a.get("explained_atom_ids") or []))
    ids_b = set(int(v) for v in list(b.get("explained_atom_ids") or []))
    if ids_a & ids_b:
        return True
    # Sample each trajectory's segment and check the minimum
    # perpendicular distance of the shorter segment's endpoints to the
    # other's (finite) axis. Using segments rather than infinite lines
    # avoids false conflicts between shanks that would only cross
    # outside their physical extents.
    sa = np.asarray(a.get("start_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
    ea = np.asarray(a.get("end_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
    sb = np.asarray(b.get("start_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
    eb = np.asarray(b.get("end_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
    da_seg = ea - sa
    db_seg = eb - sb
    la = float(np.linalg.norm(da_seg))
    lb = float(np.linalg.norm(db_seg))
    if la < 1e-6 or lb < 1e-6:
        return False
    da_hat = da_seg / la
    db_hat = db_seg / lb

    def _perp_to_seg(pt: np.ndarray, s0: np.ndarray, d_hat: np.ndarray, length: float) -> float:
        rel = pt - s0
        t = float(np.clip(float(rel @ d_hat), 0.0, length))
        foot = s0 + d_hat * t
        return float(np.linalg.norm(pt - foot))

    dists = [
        _perp_to_seg(sb, sa, da_hat, la),
        _perp_to_seg(eb, sa, da_hat, la),
        _perp_to_seg(sa, sb, db_hat, lb),
        _perp_to_seg(ea, sb, db_hat, lb),
    ]
    return float(min(dists)) <= radius_mm


def _resolve_conflicts(
    accepted: list[dict[str, Any]],
    radius_mm: float,
) -> list[dict[str, Any]]:
    ranked = sorted(
        accepted,
        key=lambda p: (
            -float(p.get("intracranial_span_mm", 0.0)),
            float(p.get("axis_residual_rms_mm", 1e9)),
        ),
    )
    kept: list[dict[str, Any]] = []
    for cand in ranked:
        if any(_axes_conflict(cand, k, radius_mm) for k in kept):
            continue
        kept.append(cand)
    return kept


# ---------------------------------------------------------------------------
# Entry point used by deep_core_v1._run_model_fit
# ---------------------------------------------------------------------------

def run_model_fit_group(
    proposals: list[dict[str, Any]],
    arr_kji: np.ndarray,
    ras_to_ijk_fn,
    head_distance_map_kji: np.ndarray,
    library_models: list[dict[str, Any]],
    cfg,
    *,
    metal_mask_kji: np.ndarray | None = None,
    hull_mask_kji: np.ndarray | None = None,
    support_atoms: list[dict[str, Any]] | None = None,
    blob_sample_points_ras: np.ndarray | None = None,
) -> dict[str, Any]:
    atom_pool = list(support_atoms or [])
    atom_by_id = {int(a.get("atom_id", -1)): a for a in atom_pool if int(a.get("atom_id", -1)) >= 0}
    lib_bounds = axr.library_span_range(library_models)

    # Sort proposals by priority: most atoms first (richer evidence =
    # better chance of producing a good fit that can then claim its
    # atoms), then longest cluster span. Later proposals see a smaller
    # atom pool (claimed atoms are blocked) which prevents bridged
    # fits from surviving — if a long clean shank fits first, its
    # atoms are gone, and a bridged proposal spanning that shank plus
    # a neighbour will fail the scalp-exit gate with its remaining
    # atoms.
    def _priority_key(prop: dict[str, Any]) -> tuple:
        atoms = list(prop.get("atom_id_list") or [])
        s = prop.get("start_ras")
        e = prop.get("end_ras")
        span = 0.0
        if s is not None and e is not None:
            span = float(
                np.linalg.norm(
                    np.asarray(e, dtype=float).reshape(3)
                    - np.asarray(s, dtype=float).reshape(3)
                )
            )
        return (-int(len(atoms)), -float(span))

    ordered = sorted(list(proposals or []), key=_priority_key)

    claimed: set[int] = set()
    accepted: list[dict[str, Any]] = []
    n_rejected = 0

    for prop in ordered:
        fitted = _fit_one_proposal(
            prop,
            arr_kji=arr_kji,
            metal_mask_kji=metal_mask_kji,
            head_distance_map_kji=head_distance_map_kji,
            ras_to_ijk_fn=ras_to_ijk_fn,
            atom_pool=atom_pool,
            atom_by_id=atom_by_id,
            blob_sample_points_ras=blob_sample_points_ras,
            library_models=library_models,
            library_span_bounds=lib_bounds,
            cfg=cfg,
            blocked_atom_ids=claimed,
        )
        if fitted is None:
            n_rejected += 1
            continue
        accepted.append(fitted)
        for aid in fitted.get("explained_atom_ids") or []:
            claimed.add(int(aid))

    # Strip internal fit hooks before returning to the pipeline.
    for p in accepted:
        for k in (
            "_fit_center_ras",
            "_fit_axis_ras",
            "_fit_t_deep",
            "_fit_t_shallow",
            "_fit_t_interface",
        ):
            p.pop(k, None)

    return {
        "accepted_proposals": accepted,
        "stats": {
            "input_count": int(len(proposals or [])),
            "accepted_count": int(len(accepted)),
            "rejected": int(n_rejected),
            "claimed_atom_count": int(len(claimed)),
        },
    }
