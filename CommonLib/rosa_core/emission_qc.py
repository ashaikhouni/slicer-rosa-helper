"""Per-emission QC for unified-pipeline output.

Given the emissions from ``detect_and_place_unified`` and the original
``features`` dict, build per-emission diagnostic data so callers can
visualize / audit / triage:

  * `EmissionQC` dataclass with everything one needs for a slab + profile
    figure: walker arcs + signal, bolt-end arc, placed-contact arcs,
    placed-contact HU sample, per-library-model corr breakdown.
  * `build_subject_qc(...)` runs all of the above for a list of
    emissions in one shared compute pass.
  * `render_emission_figure(...)` produces a 3-panel matplotlib figure
    (CT slab + walker profile + per-model corr bar chart).
  * `emission_qc_to_dict(...)` serializes one QC entry to a JSON-safe
    dict for the CLI.

Same module powers QC notebooks AND the ``rosa-agent qc`` subcommand.
The notebook calls ``build_subject_qc`` + ``render_emission_figure``;
the CLI calls the same functions and dumps the dicts to JSON + PNGs.

Distinct from ``rosa_core.qc``, which is planned-vs-final-trajectory QC
(a different concern: how close the ROSA-planned trajectory was to the
post-op contacts). This module is for the auto-detection output.
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass
class PerModelCorr:
    """Score one library model would have given on this emission's seed."""
    model_id: str
    n_slots: int
    n_covered: int
    corr: float


@dataclass
class EmissionQC:
    """Comprehensive QC for one emitted trajectory.

    All RAS positions are in the canonical-feature-grid frame (the same
    frame as the input ``features`` dict). All arc-along values are mm.
    """

    emission_idx: int

    # Best-pick result (from the emission itself).
    model_id: str | None
    corr: float
    n_placed: int
    bolt_source: str
    bolt_end_arc_mm: float

    # Seed (post bolt-anchor) and centerline (snap-on-LoG output).
    seed_start_ras: list[float]
    seed_end_ras: list[float]
    centerline_ras: list[list[float]]    # snapped polyline
    placed_ras: list[list[float]]
    placed_contact_arcs_mm: list[float]  # arc along centerline_ras

    # Sampled HU at each placed contact center (placement-quality QC).
    placed_hu: list[float]
    placed_hu_mean: float
    placed_hu_min: float

    # Walker disk-stat profile along the snapped centerline (this is
    # the matched-filter signal). arcs and signal have equal length.
    walker_arcs_mm: list[float]
    walker_max_hu: list[float]

    # Score breakdown across every library model (sorted by corr desc).
    per_model_corr: list[PerModelCorr]

    # GT context (optional; populated when caller supplies a match).
    gt_match_name: str | None = None
    expected_n_contacts: int | None = None
    gt_contacts_ras: list[list[float]] | None = None

    diagnostics: dict[str, Any] = field(default_factory=dict)


def _walker_profile(centerline, ct_arr_kji, r2i, *,
                      step_mm: float, disk_radius_mm: float,
                      n_radii: int, n_angles: int, hu_min: float):
    """Sample max-disk HU along ``centerline``. Mirrors v2's walker call."""
    from .contact_placement import sample_disk_along_polyline
    arcs, max_b, _total = sample_disk_along_polyline(
        ct_arr_kji, r2i, np.asarray(centerline, float),
        polarity="positive", step_mm=step_mm,
        disk_radius_mm=disk_radius_mm,
        n_radii=n_radii, n_angles=n_angles,
        total_threshold=hu_min,
    )
    return arcs, max_b


def build_emission_qc(
    emission,
    *,
    emission_idx: int,
    features: dict,
    library_models: Sequence[dict],
    gt_contacts_ras: np.ndarray | None = None,
    gt_match_name: str | None = None,
    expected_n_contacts: int | None = None,
) -> EmissionQC:
    """Build comprehensive QC for one emitted trajectory.

    Re-runs ``estimate_bolt_end_from_metal_mass`` + the snap pass on
    the emission's seed to recover the centerline + walker profile
    that ``place_contacts_for_seed_v2`` used internally, then scores
    every library model against that signal so callers can see the
    full per-model corr breakdown (not just the picked model).
    """
    from .contact_placement import estimate_bolt_end_from_metal_mass
    from .contact_placement_v2 import (
        _snap_centerline_to_centroid, _extend_centerline_tail,
        _project_to_polyline_arc,
        WALK_DISK_RADIUS_MM, WALK_HU_MIN, WALK_N_RADII, WALK_N_ANGLES,
        WALK_TIP_PAD_MM, WALK_STEP_MM,
    )
    from .matched_filter import matched_filter_pick

    es = np.asarray(emission.start_ras, dtype=float)
    ee = np.asarray(emission.end_ras, dtype=float)
    seed_dir = (ee - es) / max(np.linalg.norm(ee - es), 1e-9)

    ct_arr = features["ct_arr_kji"]
    log_arr = features.get("log")
    r2i = np.asarray(features["ras_to_ijk_mat"], dtype=float)

    # Re-build the centerline + bolt-end on the emission's seed.
    centerline = None
    bolt_end_arc_mm = float("nan")
    walker_arcs = walker_max = np.array([], dtype=float)
    placed_arcs: list[float] = []
    try:
        be = estimate_bolt_end_from_metal_mass(
            es, ee, features=features, library_models=library_models,
        )
        be_arc = be.get("bolt_end_arc_mm")
        cp = be.get("centerline")
        if be_arc is not None and cp is not None:
            cp = np.asarray(cp, dtype=float)
            if log_arr is not None:
                centerline = _snap_centerline_to_centroid(cp, log_arr, r2i)
            else:
                centerline = cp
            cl_walker = _extend_centerline_tail(centerline, WALK_TIP_PAD_MM)
            walker_arcs, walker_max = _walker_profile(
                cl_walker, ct_arr, r2i,
                step_mm=WALK_STEP_MM,
                disk_radius_mm=WALK_DISK_RADIUS_MM,
                n_radii=WALK_N_RADII, n_angles=WALK_N_ANGLES,
                hu_min=WALK_HU_MIN,
            )
            bolt_end_arc_mm = float(_project_to_polyline_arc(
                centerline, es + float(be_arc) * seed_dir,
            ))
            for p in (emission.placed_ras or []):
                placed_arcs.append(float(_project_to_polyline_arc(
                    centerline, np.asarray(p, dtype=float),
                )))
    except Exception as exc:
        warnings.warn(
            f"emission_qc.build_emission_qc: profile/centerline rebuild "
            f"failed ({exc}); per-model corr breakdown will be empty",
            stacklevel=2,
        )

    # Sample HU at each placed contact center.
    placed_hu: list[float] = []
    if emission.placed_ras:
        from .volume_sampling import sample_trilinear_at_ras
        for p in emission.placed_ras:
            try:
                hu = float(sample_trilinear_at_ras(
                    ct_arr, r2i, np.asarray(p, float),
                ))
            except Exception:
                hu = float("nan")
            placed_hu.append(hu if np.isfinite(hu) else float("nan"))
    placed_hu_arr = np.asarray(placed_hu, float) if placed_hu else np.array([np.nan])
    placed_hu_mean = float(np.nanmean(placed_hu_arr))
    placed_hu_min  = float(np.nanmin(placed_hu_arr))

    # Per-model corr breakdown — score each library model separately
    # against the same walker signal.
    per_model: list[PerModelCorr] = []
    if walker_arcs.size and centerline is not None:
        cl_max = float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum())
        for m in library_models:
            try:
                mres = matched_filter_pick(
                    walker_arcs, walker_max, [m],
                    bolt_end_arc=bolt_end_arc_mm,
                    profile_end_arc=cl_max,
                    max_extend_tip_mm=WALK_TIP_PAD_MM,
                )
            except Exception:
                continue
            per_model.append(PerModelCorr(
                model_id=str(m.get("id") or ""),
                n_slots=int(mres.n_slots),
                n_covered=int(mres.n_covered),
                corr=float(mres.corr),
            ))
        per_model.sort(key=lambda x: -x.corr)

    return EmissionQC(
        emission_idx=int(emission_idx),
        model_id=emission.model_id,
        corr=float(emission.corr_score),
        n_placed=int(emission.n_placed),
        bolt_source=str(emission.bolt_source),
        bolt_end_arc_mm=bolt_end_arc_mm,
        seed_start_ras=[float(x) for x in es],
        seed_end_ras=[float(x) for x in ee],
        centerline_ras=([[float(x) for x in p] for p in centerline]
                          if centerline is not None else []),
        placed_ras=[[float(x) for x in p] for p in (emission.placed_ras or [])],
        placed_contact_arcs_mm=placed_arcs,
        placed_hu=placed_hu,
        placed_hu_mean=placed_hu_mean,
        placed_hu_min=placed_hu_min,
        walker_arcs_mm=[float(a) for a in walker_arcs],
        walker_max_hu=[float(v) for v in walker_max],
        per_model_corr=per_model,
        gt_match_name=gt_match_name,
        expected_n_contacts=expected_n_contacts,
        gt_contacts_ras=([[float(x) for x in p] for p in gt_contacts_ras]
                          if gt_contacts_ras is not None and len(gt_contacts_ras) else None),
        diagnostics={
            "anchored": bool(getattr(emission, "anchored", False)),
            "synthed":  bool(getattr(emission, "synthed", False)),
        },
    )


def build_subject_qc(
    emissions: Sequence,
    *,
    features: dict,
    library_models: Sequence[dict],
    gt_match_idx: dict[str, int] | None = None,
    gt_meta: dict[str, dict] | None = None,
) -> list[EmissionQC]:
    """Build QC for every emission. Optional GT cross-ref:
    ``gt_match_idx`` maps GT name → emission index; ``gt_meta`` maps
    GT name → ``{contacts_ras, expected_n}``.
    """
    qc_list: list[EmissionQC] = []
    inv_match: dict[int, str] = {}
    if gt_match_idx is not None:
        inv_match = {ei: name for name, ei in gt_match_idx.items()}
    for ei, emit in enumerate(emissions):
        gt_name = inv_match.get(ei)
        gt_contacts = expected_n = None
        if gt_name is not None and gt_meta is not None and gt_name in gt_meta:
            gt_contacts = gt_meta[gt_name].get("contacts_ras")
            expected_n = gt_meta[gt_name].get("expected_n")
        qc_list.append(build_emission_qc(
            emit, emission_idx=ei,
            features=features, library_models=library_models,
            gt_contacts_ras=(np.asarray(gt_contacts, float)
                              if gt_contacts is not None else None),
            gt_match_name=gt_name,
            expected_n_contacts=expected_n,
        ))
    return qc_list


def emission_qc_to_dict(qc: EmissionQC) -> dict[str, Any]:
    """JSON-safe representation of an EmissionQC."""
    return asdict(qc)


# ---------------------------------------------------------------------
# Figure rendering — for notebooks. Imports matplotlib lazily so the
# data-only API stays import-cheap.
# ---------------------------------------------------------------------


def render_emission_figure(
    qc: EmissionQC,
    *,
    features: dict,
    bolts: Sequence[dict] | None = None,
    slab_perp_half_mm: float = 12.0,
    slab_thickness_half_mm: float = 2.0,
    slab_step_mm: float = 0.5,
    title_extra: str = "",
):
    """Render a 3-panel QC figure for one emission:

      Top:    CT slab (max-IP, ±slab_thickness_half_mm) through the
              emission axis, with GT contacts (if any), bolt CC voxels,
              placed contacts, and the bolt_end marker overlaid.
      Middle: walker disk-stat profile along the centerline with
              bolt_end + placed-contact arcs marked.
      Bottom: per-library-model corr bar chart, picked model highlighted.
    """
    import matplotlib.pyplot as plt
    from .volume_sampling import sample_trilinear_at_ras

    ct_arr = features["ct_arr_kji"]
    r2i = np.asarray(features["ras_to_ijk_mat"], dtype=float)
    es = np.asarray(qc.seed_start_ras, dtype=float)
    ee = np.asarray(qc.seed_end_ras, dtype=float)
    seed_dir = (ee - es) / max(np.linalg.norm(ee - es), 1e-9)
    span = float(np.linalg.norm(ee - es))

    world_up = np.array([0.0, 0.0, 1.0])
    v_perp = world_up - (world_up @ seed_dir) * seed_dir
    if np.linalg.norm(v_perp) < 1e-6:
        v_perp = np.array([0.0, 1.0, 0.0])
    v_perp /= np.linalg.norm(v_perp)
    third = np.cross(seed_dir, v_perp)
    third /= np.linalg.norm(third)

    U_RANGE = (-15.0, span + 15.0)
    V_RANGE = (-slab_perp_half_mm, slab_perp_half_mm)
    u_grid = np.arange(U_RANGE[0], U_RANGE[1] + slab_step_mm, slab_step_mm)
    v_grid = np.arange(V_RANGE[0], V_RANGE[1] + slab_step_mm, slab_step_mm)
    t_off = np.arange(-slab_thickness_half_mm,
                        slab_thickness_half_mm + slab_step_mm,
                        slab_step_mm)
    slab = np.full((v_grid.size, u_grid.size), np.nan)
    for t in t_off:
        for vi, vv in enumerate(v_grid):
            for ui, uu in enumerate(u_grid):
                p = es + uu * seed_dir + vv * v_perp + t * third
                val = sample_trilinear_at_ras(ct_arr, r2i, p)
                if not np.isnan(val):
                    cur = slab[vi, ui]
                    slab[vi, ui] = float(val) if np.isnan(cur) else max(cur, float(val))

    def _proj(pts):
        rel = np.asarray(pts, float) - es
        return rel @ seed_dir, rel @ v_perp

    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1.0, 0.9], hspace=0.35)
    ax_slab = fig.add_subplot(gs[0])
    ax_prof = fig.add_subplot(gs[1])
    ax_bar = fig.add_subplot(gs[2])

    extent = [u_grid[0], u_grid[-1], v_grid[0], v_grid[-1]]
    finite = slab[np.isfinite(slab)]
    if finite.size:
        vmin, vmax = float(np.percentile(finite, 1)), float(np.percentile(finite, 99.5))
    else:
        vmin, vmax = -100.0, 2500.0
    ax_slab.imshow(slab, extent=extent, origin="lower", aspect="auto",
                    cmap="gray", vmin=vmin, vmax=vmax)
    ax_slab.axhline(0, color="cyan", lw=0.5, ls="--", alpha=0.5,
                     label="emission axis (perp=0)")
    ax_slab.axvspan(0, span, color="blue", alpha=0.04,
                     label=f"emission seed ({span:.1f} mm)")

    if qc.gt_contacts_ras:
        gtc_along, gtc_v = _proj(qc.gt_contacts_ras)
        ax_slab.scatter(gtc_along, gtc_v, marker="x", color="orange", s=70, lw=2,
                         label=f"GT {qc.gt_match_name} contacts (n={len(qc.gt_contacts_ras)})",
                         zorder=12)

    if bolts is not None:
        cc_pts = []
        for b in bolts:
            for p in np.asarray(b["pts_ras"], float):
                rel = p - es
                uu = rel @ seed_dir
                vv = rel @ v_perp
                tt = rel @ third
                if (abs(tt) <= slab_thickness_half_mm + 0.5
                    and U_RANGE[0] - 2 <= uu <= U_RANGE[1] + 2
                    and V_RANGE[0] <= vv <= V_RANGE[1]):
                    cc_pts.append((uu, vv))
        if cc_pts:
            cc = np.asarray(cc_pts, float)
            ax_slab.scatter(cc[:, 0], cc[:, 1], marker=".", color="magenta",
                             s=4, alpha=0.5,
                             label=f"bolt CC voxels ({cc.shape[0]})", zorder=8)

    if qc.placed_ras:
        placed_along, placed_v = _proj(qc.placed_ras)
        ax_slab.scatter(placed_along, placed_v - 4.0, marker="v", color="red",
                         s=40, label=f"placed ({qc.model_id}, n={qc.n_placed})",
                         zorder=11)

    if np.isfinite(qc.bolt_end_arc_mm):
        ax_slab.axvline(qc.bolt_end_arc_mm, color="red", lw=1.2,
                         label=f"bolt_end_arc={qc.bolt_end_arc_mm:.1f}mm")

    title = f"emission #{qc.emission_idx}"
    if qc.gt_match_name is not None:
        title += f" ↔ GT {qc.gt_match_name}"
        if qc.expected_n_contacts is not None:
            title += f" (expected {qc.expected_n_contacts} contacts)"
    else:
        title += "  ↔  ORPHAN"
    title += f"  →  {qc.model_id}  corr={qc.corr:.3f}"
    if title_extra:
        title += f"  ·  {title_extra}"
    ax_slab.set_title(title, fontsize=10)
    ax_slab.set_xlim(U_RANGE)
    ax_slab.set_ylim(V_RANGE)
    ax_slab.set_ylabel("perp from emission axis (mm)")
    ax_slab.legend(loc="lower right", fontsize=7, ncol=2)

    if qc.walker_arcs_mm:
        ax_prof.plot(qc.walker_arcs_mm, qc.walker_max_hu,
                      color="steelblue", lw=1.0,
                      label="walker disk-stat (HU max)")
        ax_prof.axhline(1500, color="red", ls=":", lw=0.6, label="HU=1500")
        if np.isfinite(qc.bolt_end_arc_mm):
            ax_prof.axvline(qc.bolt_end_arc_mm, color="red", lw=1.2,
                             label=f"bolt_end_arc={qc.bolt_end_arc_mm:.1f}mm")
        for arc in qc.placed_contact_arcs_mm:
            ax_prof.axvline(arc, color="red", alpha=0.35, lw=0.7)
        ax_prof.set_xlim(0, max(qc.walker_arcs_mm))
        ax_prof.set_xlabel("arc along snapped centerline (mm)")
        ax_prof.set_ylabel("HU max")
        ax_prof.legend(loc="upper right", fontsize=7)
        ax_prof.grid(True, alpha=0.2)
    else:
        ax_prof.text(0.5, 0.5, "profile build failed",
                      ha="center", va="center", transform=ax_prof.transAxes,
                      fontsize=10, color="gray")
        ax_prof.set_axis_off()

    if qc.per_model_corr:
        ids = [pm.model_id for pm in qc.per_model_corr]
        corrs = [pm.corr for pm in qc.per_model_corr]
        n_cov = [f"{pm.n_covered}/{pm.n_slots}" for pm in qc.per_model_corr]
        colors = ["#cc3333" if pm.model_id == qc.model_id else "#888888"
                   for pm in qc.per_model_corr]
        x = np.arange(len(ids))
        ax_bar.bar(x, corrs, color=colors, alpha=0.8)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(ids, rotation=45, ha="right", fontsize=7)
        for xi, (c, nc) in enumerate(zip(corrs, n_cov)):
            ax_bar.text(xi, c + 0.01, f"{c:.2f}\n{nc}",
                         ha="center", va="bottom", fontsize=6)
        ax_bar.set_ylabel("matched-filter corr")
        ax_bar.set_ylim(0, max(1.0, max(corrs) * 1.15))
        ax_bar.set_title("per-model corr breakdown (red = picked, slots = covered/total)",
                          fontsize=9)
        ax_bar.grid(True, axis="y", alpha=0.2)
    else:
        ax_bar.text(0.5, 0.5, "no per-model corr available",
                     ha="center", va="center", transform=ax_bar.transAxes,
                     fontsize=10, color="gray")
        ax_bar.set_axis_off()

    return fig
