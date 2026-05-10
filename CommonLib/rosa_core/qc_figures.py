"""Headless matplotlib emission figures for QC output.

Lifted from the notebook's ``render_emission`` cell
(slicer-rosa-helper/notebooks/v1_seeds_v2_placement_qc.ipynb,
the visualization block at the bottom). Adapted to:

  * Force the Agg backend (no GUI).
  * Take a ``PlacedTrajectory`` instead of a notebook ``PlacementCtx``.
  * Skip cleanly when matplotlib isn't installed.

The figure is the same 3-panel layout the notebook produces:

  * Top:    CT slab (HU max-IP overlaid with -LoG heatmap), centerline,
            placed contact markers, bolt-end vertical line, GT scatter.
  * Middle: walker profile (-LoG p90-disk along centerline).
  * Bottom: per-library-model corr bar chart (when available).

Used by ``rosa_core.qc_output.write_qc_directory`` when ``write_figures``
is on (default).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np


def render_all_figures(
    trajectories: Iterable, output_dir: str | Path, *,
    features: dict, bolts: list[dict] | None = None,
    gt_contacts_by_name: dict[str, np.ndarray] | None = None,
) -> int:
    """Render one PNG per trajectory into ``output_dir``.

    Filenames: ``001_<name>.png``, ``002_<name>.png``, ... — the index
    keeps lexicographic-sortable ordering. Returns the count rendered.

    Returns 0 if matplotlib isn't available (caller's choice to log/warn).
    """
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return 0

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for idx, traj in enumerate(trajectories, start=1):
        path = out / f"{idx:03d}_{_safe_name(traj.name)}.png"
        try:
            render_placed_trajectory_figure(
                traj, path,
                features=features, bolts=bolts,
                gt_contacts=(gt_contacts_by_name or {}).get(traj.name),
            )
            count += 1
        except Exception as exc:  # noqa: BLE001
            # Per-figure rendering failures shouldn't break the whole batch;
            # write a small breadcrumb so caller can debug.
            (out / f"{idx:03d}_{_safe_name(traj.name)}.error.txt").write_text(
                f"render_placed_trajectory_figure failed: {type(exc).__name__}: {exc}\n",
            )
    return count


def render_placed_trajectory_figure(
    traj, path: str | Path, *,
    features: dict, bolts: list[dict] | None = None,
    gt_contacts: np.ndarray | None = None,
) -> None:
    """Write a 3-panel QC figure for one ``PlacedTrajectory`` to ``path``.

    ``features`` is the full feature dict (``ct_arr_kji``, ``log``,
    ``ras_to_ijk_mat``); same shape as what
    ``rosa_detect.guided_fit_engine.compute_features`` produces.
    """
    import matplotlib
    matplotlib.use("Agg")  # idempotent
    import matplotlib.pyplot as plt

    from .volume_sampling import sample_trilinear_at_ras

    es = np.asarray(traj.start_ras, dtype=float)
    ee = np.asarray(traj.end_ras, dtype=float)
    r2i = np.asarray(features["ras_to_ijk_mat"], dtype=float)
    ct_arr = features["ct_arr_kji"]
    log_arr = features.get("log")

    slab, slab_log, u_grid, v_grid, seed_dir, v_perp, _third, span = _build_slab(
        es, ee, ct_arr, r2i, log_arr=log_arr,
    )

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.7, 0.9, 0.8], hspace=0.45)
    ax_slab = fig.add_subplot(gs[0])
    ax_prof = fig.add_subplot(gs[1])
    ax_bar = fig.add_subplot(gs[2])

    extent = [u_grid[0], u_grid[-1], v_grid[0], v_grid[-1]]
    if slab_log is not None:
        finite_log = slab_log[np.isfinite(slab_log)]
        if finite_log.size:
            log_lo = float(np.percentile(finite_log, 5))
            log_hi = float(np.percentile(finite_log, 99))
            ax_slab.imshow(slab_log, extent=extent, origin="lower", aspect="auto",
                            cmap="viridis", vmin=log_lo, vmax=log_hi)
    finite = slab[np.isfinite(slab)]
    hu_vmin, hu_vmax = (
        (float(np.percentile(finite, 1)), float(np.percentile(finite, 99.5)))
        if finite.size else (-100.0, 2500.0)
    )
    ax_slab.imshow(slab, extent=extent, origin="lower", aspect="auto",
                    cmap="gray", vmin=hu_vmin, vmax=hu_vmax, alpha=0.55)
    ax_slab.axhline(0, color="cyan", lw=0.5, ls="--", alpha=0.5,
                     label="seed (perp=0)")
    ax_slab.axvspan(0, span, color="blue", alpha=0.04)

    def _proj(pts):
        rel = np.asarray(pts, float) - es
        return rel @ seed_dir, rel @ v_perp

    cl = traj.centerline_ras
    if cl is not None and len(cl) >= 2:
        cu, cv = _proj(cl)
        ax_slab.plot(cu, cv, color="#22c55e", lw=1.4, alpha=0.9, zorder=9,
                      label=f"centerline ({traj.bolt_source}, {len(cl)} pts)")

    if bolts:
        SLAB_THICK_HALF = 2.0
        _, _, third = _slab_basis(es, ee)
        cc_pts = []
        for b in bolts:
            for p in np.asarray(b["pts_ras"], dtype=float):
                rel = p - es
                uu = rel @ seed_dir
                vv = rel @ v_perp
                tt = rel @ third
                if (abs(tt) <= SLAB_THICK_HALF + 0.5
                    and -2 <= uu <= span + 17
                    and v_grid[0] <= vv <= v_grid[-1]):
                    cc_pts.append((uu, vv))
        if cc_pts:
            cc = np.asarray(cc_pts, dtype=float)
            ax_slab.scatter(cc[:, 0], cc[:, 1], marker=".", color="magenta",
                             s=4, alpha=0.5, label=f"bolt CC ({cc.shape[0]})",
                             zorder=7)

    if gt_contacts is not None and len(gt_contacts):
        u, v = _proj(gt_contacts)
        ax_slab.scatter(u, v, marker="x", color="orange", s=70, lw=2,
                         label=f"GT (n={len(gt_contacts)})", zorder=12)

    if traj.contacts_ras:
        u, v = _proj(traj.contacts_ras)
        ax_slab.scatter(u, v, marker="o", color="red", s=30, alpha=0.85,
                         label=f"placed ({traj.model_id}, n={len(traj.contacts_ras)})",
                         zorder=11)

    if traj.bolt_source == "metal" and traj.bolt_end_arc_mm > 0:
        ax_slab.axvline(traj.bolt_end_arc_mm, color="red", lw=1.2,
                         label=f"bolt_end={traj.bolt_end_arc_mm:.1f}mm")

    ax_slab.set_xlim(u_grid[0], u_grid[-1])
    ax_slab.set_ylim(v_grid[0], v_grid[-1])
    ax_slab.set_ylabel("perp (mm)")
    ax_slab.legend(loc="lower right", fontsize=7, ncol=2)

    # Walker profile — placed contact tick marks (no walker signal stored
    # on PlacedTrajectory; only the placed positions are surfaced).
    if traj.contacts_ras and cl is not None:
        cl_arr = np.asarray(cl, dtype=float)
        for p in traj.contacts_ras:
            arc = _project_to_polyline_arc_local(cl_arr, np.asarray(p, dtype=float))
            ax_prof.axvline(arc, color="#1f77b4", alpha=0.6, lw=0.8)
    ax_prof.set_xlim(0, span if cl is None else max(span, _polyline_total_arc(cl)))
    ax_prof.set_xlabel("arc along centerline (mm)")
    ax_prof.set_ylabel("placed contacts (tick marks)")
    ax_prof.grid(True, alpha=0.2)
    if traj.bolt_source == "metal" and traj.bolt_end_arc_mm > 0:
        ax_prof.axvline(traj.bolt_end_arc_mm, color="red", lw=1.0,
                         label=f"bolt_end={traj.bolt_end_arc_mm:.1f}mm")
        ax_prof.legend(loc="upper right", fontsize=8)

    # Per-model corr bar — read from score_components.per_model_corr stash.
    pm = (traj.score_components or {}).get("per_model_corr") or []
    _bar(ax_bar, list(pm), traj.model_id,
         f"per-model corr  →  picked {traj.model_id}")

    title = (
        f"{traj.name}    "
        f"[{traj.bolt_source}]    "
        f"band={traj.band}    "
        f"compound={traj.compound_score:.3f}"
    )
    fig.suptitle(title, fontsize=10, y=0.99)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Internals (slab build + per-model bar — lifted from notebook)
# ---------------------------------------------------------------------


def _slab_basis(es: np.ndarray, ee: np.ndarray):
    seed_dir = (ee - es) / max(np.linalg.norm(ee - es), 1e-9)
    world_up = np.array([0.0, 0.0, 1.0])
    v_perp = world_up - (world_up @ seed_dir) * seed_dir
    if np.linalg.norm(v_perp) < 1e-6:
        v_perp = np.array([0.0, 1.0, 0.0])
    v_perp /= np.linalg.norm(v_perp)
    third = np.cross(seed_dir, v_perp); third /= np.linalg.norm(third)
    return seed_dir, v_perp, third


def _build_slab(
    es, ee, ct_arr, r2i, *, log_arr=None,
    perp_half: float = 12.0, thick_half: float = 2.0, step: float = 0.5,
):
    """Max-IP slab through the seed plane. Returns (slab_hu, slab_log,
    u_grid, v_grid, seed_dir, v_perp, third, span)."""
    from .volume_sampling import sample_trilinear_at_ras

    seed_dir, v_perp, third = _slab_basis(es, ee)
    span = float(np.linalg.norm(ee - es))
    u_grid = np.arange(-15.0, span + 15.0 + step, step)
    v_grid = np.arange(-perp_half, perp_half + step, step)
    t_off = np.arange(-thick_half, thick_half + step, step)
    slab_hu = np.full((v_grid.size, u_grid.size), np.nan)
    slab_log = np.full((v_grid.size, u_grid.size), np.nan) if log_arr is not None else None

    for t in t_off:
        for vi, vv in enumerate(v_grid):
            for ui, uu in enumerate(u_grid):
                p = es + uu * seed_dir + vv * v_perp + t * third
                val = sample_trilinear_at_ras(ct_arr, r2i, p)
                if not np.isnan(val):
                    cur = slab_hu[vi, ui]
                    slab_hu[vi, ui] = float(val) if np.isnan(cur) else max(cur, float(val))
                if log_arr is not None:
                    lv = sample_trilinear_at_ras(log_arr, r2i, p)
                    if not np.isnan(lv):
                        nlv = -float(lv)
                        cur = slab_log[vi, ui]
                        slab_log[vi, ui] = nlv if np.isnan(cur) else max(cur, nlv)
    return slab_hu, slab_log, u_grid, v_grid, seed_dir, v_perp, third, span


def _bar(ax, per_model, picked: str | None, title: str) -> None:
    if not per_model:
        ax.set_axis_off(); ax.set_title(title + "  (no data)", fontsize=9); return
    ids = [t[0] for t in per_model]
    corrs = [t[3] for t in per_model]
    cov = [f"{t[2]}/{t[1]}" for t in per_model]
    colors = ["#cc3333" if i == picked else "#888888" for i in ids]
    x = np.arange(len(ids))
    ax.bar(x, corrs, color=colors, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=7)
    ymax = max(1.0, max(corrs) * 1.15)
    ax.set_ylim(min(0, min(corrs) - 0.05), ymax)
    for xi, (c, nc) in enumerate(zip(corrs, cov)):
        ax.text(xi, c + 0.01, f"{c:.2f}\n{nc}", ha="center", va="bottom", fontsize=6)
    ax.axhline(0, color="black", lw=0.4)
    ax.grid(True, axis="y", alpha=0.2)
    ax.set_title(title, fontsize=9)


def _project_to_polyline_arc_local(polyline: np.ndarray, pt: np.ndarray) -> float:
    """Inline arc-projection (avoid pulling the contact_placement package
    into the figure renderer's hot path). Closed-form per-segment."""
    p = polyline.astype(float)
    if len(p) < 2:
        return 0.0
    a = p[:-1]; b = p[1:]
    ab = b - a
    ab_len2 = np.einsum("ij,ij->i", ab, ab)
    seg_lens = np.sqrt(ab_len2)
    cum_start = np.concatenate([[0.0], np.cumsum(seg_lens[:-1])])
    best_d, best_arc = np.inf, 0.0
    for i in range(len(a)):
        L = seg_lens[i]
        if L < 1e-9: continue
        u = ab[i] / L
        t = float(np.clip((pt - a[i]) @ u, 0.0, L))
        proj = a[i] + t * u
        d = float(np.linalg.norm(pt - proj))
        if d < best_d:
            best_d = d
            best_arc = float(cum_start[i] + t)
    return best_arc


def _polyline_total_arc(polyline) -> float:
    p = np.asarray(polyline, dtype=float)
    if len(p) < 2: return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def _safe_name(name: str) -> str:
    """File-safe slug (allow alnum, dot, dash, underscore)."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(name))


__all__ = ["render_all_figures", "render_placed_trajectory_figure"]
