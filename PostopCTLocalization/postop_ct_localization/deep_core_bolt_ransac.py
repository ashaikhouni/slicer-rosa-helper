"""RANSAC bolt detection for deep_core.

A bolt is the solid stainless-steel SEEG anchor that holds each electrode at
the skull entry. On CT it shows as a ~10-40 mm continuous saturation-bright
cylinder embedded in the scalp, distinct from the dimmer electrode contacts.
Detecting bolts gives us a per-electrode anchor with an accurate PCA axis
before the electrode even enters the brain.

This module fits line candidates to the bright metal point cloud via RANSAC,
then filters them with:

  - span gate (BOLT_SPAN_MIN_MM .. BOLT_SPAN_MAX_MM) — solid, not shank-long
  - fill fraction along the line >= fill_frac_min (rejects contact-ring trains)
  - max gap along the line <= max_gap_mm (rejects broken lines)
  - shallow-shell position (head_distance at center in [shell_min, shell_max])
  - axis-depth gradient (probing inward along the axis must reach deeper head
    than the center, which excludes dental/jaw metal that runs along the skin)
  - dedup via support overlap + collinear-along-axis (drops fragments of the
    same bolt or same electrode when RANSAC finds them twice)

All operations are pure numpy; no Slicer or vtk dependencies. SimpleITK is
only used for CC labeling of the shallow-shell mask; importing is lazy so
callers in non-SITK environments can still load the module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Sequence

import numpy as np


@dataclass
class BoltCandidate:
    """One accepted bolt line returned by ``find_bolt_candidates``."""

    center_ras: np.ndarray           # shape (3,)
    axis_ras: np.ndarray             # shape (3,), unit vector
    span_mm: float                   # length of the line along the axis
    n_inliers: int                   # inlier count on the original cloud
    hd_center_mm: float              # head_distance at the center
    hd_inward_max_mm: float          # max head_distance probed along the axis
    fill_frac: float                 # fraction of span sampled points with metal
    max_gap_mm: float                # longest contiguous gap along the span
    support_mask: np.ndarray = field(  # boolean mask over the original point cloud
        default_factory=lambda: np.zeros(0, dtype=bool)
    )


@dataclass
class BoltRansacConfig:
    """Parameters for ``find_bolt_candidates``.

    The defaults are the values we validated on T1 and T22 with the detection
    probe. Callers in pipeline stages build this by copying from a
    ``DeepCoreBoltConfig``.
    """

    # Shape gates
    span_min_mm: float = 10.0
    span_max_mm: float = 40.0
    inlier_tol_mm: float = 1.5
    min_inliers: int = 15
    fill_step_mm: float = 0.5
    fill_sample_r_mm: float = 1.5
    fill_frac_min: float = 0.80
    max_gap_mm: float = 3.0

    # Position gates
    shell_min_mm: float = -5.0
    shell_max_mm: float = 35.0
    axis_depth_probe_steps_mm: Sequence[float] = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
    axis_depth_delta_mm: float = 8.0

    # Dedup
    support_overlap_frac: float = 0.70
    collinear_angle_deg: float = 10.0
    collinear_perp_mm: float = 5.0

    # RANSAC control
    max_lines: int = 40
    n_samples: int = 4000
    min_sample_sep_mm: float = 8.0
    rng_seed: int = 0


def _pca_axis(points: np.ndarray) -> np.ndarray:
    c = points.mean(axis=0)
    X = points - c
    cov = X.T @ X / max(1, X.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    n = float(np.linalg.norm(axis))
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0])
    return axis / n


def _evaluate_line(
    axis: np.ndarray,
    p0: np.ndarray,
    cloud: np.ndarray,
    cfg: BoltRansacConfig,
) -> dict | None:
    """Evaluate a candidate line. Returns None if it fails any gate."""
    v = cloud - p0
    proj = v @ axis
    perp = v - np.outer(proj, axis)
    dist = np.linalg.norm(perp, axis=1)
    mask = dist < cfg.inlier_tol_mm
    n_in = int(mask.sum())
    if n_in < cfg.min_inliers:
        return None
    proj_in = proj[mask]
    t_min = float(proj_in.min())
    t_max = float(proj_in.max())
    span = t_max - t_min
    if span < cfg.span_min_mm or span > cfg.span_max_mm:
        return None

    # Fill test: sample the span at FILL_STEP_MM and require each step to have
    # an inlier within FILL_SAMPLE_R_MM of its axial position.
    n_steps = int(round(span / cfg.fill_step_mm))
    if n_steps < 4:
        return None
    ts = t_min + (np.arange(n_steps) + 0.5) * (span / n_steps)
    sorted_proj = np.sort(proj_in)
    idx = np.searchsorted(sorted_proj, ts)
    hits = np.zeros(n_steps, dtype=bool)
    for i in range(n_steps):
        left = idx[i] - 1
        right = idx[i]
        best_d = np.inf
        if 0 <= left < sorted_proj.size:
            best_d = min(best_d, abs(ts[i] - sorted_proj[left]))
        if 0 <= right < sorted_proj.size:
            best_d = min(best_d, abs(ts[i] - sorted_proj[right]))
        if best_d < cfg.fill_sample_r_mm:
            hits[i] = True
    frac = float(hits.mean())
    if frac < cfg.fill_frac_min:
        return None
    max_gap = 0
    cur = 0
    for h in hits:
        if not h:
            cur += 1
            if cur > max_gap:
                max_gap = cur
        else:
            cur = 0
    max_gap_mm = max_gap * (span / n_steps)
    if max_gap_mm > cfg.max_gap_mm:
        return None
    return {
        "mask": mask,
        "inliers": cloud[mask],
        "n_in": n_in,
        "span": span,
        "frac": frac,
        "max_gap_mm": max_gap_mm,
    }


def _ransac_lines(cloud: np.ndarray, cfg: BoltRansacConfig) -> List[dict]:
    """Iteratively peel line candidates from the metal cloud."""
    if cloud.shape[0] < cfg.min_inliers:
        return []
    rng = np.random.default_rng(cfg.rng_seed)
    remaining = cloud.copy()
    results: List[dict] = []
    for _ in range(cfg.max_lines):
        n = remaining.shape[0]
        if n < cfg.min_inliers:
            break
        best = None
        i_idx = rng.integers(0, n, size=cfg.n_samples)
        j_idx = rng.integers(0, n, size=cfg.n_samples)
        P1 = remaining[i_idx]
        P2 = remaining[j_idx]
        D = P2 - P1
        sep = np.linalg.norm(D, axis=1)
        good = (sep >= cfg.min_sample_sep_mm) & (sep <= cfg.span_max_mm)
        if not good.any():
            break
        P1g = P1[good]
        Dg = D[good]
        sep_g = sep[good]
        axes = Dg / sep_g[:, None]
        for k in range(axes.shape[0]):
            res = _evaluate_line(axes[k], P1g[k], remaining, cfg)
            if res is None:
                continue
            score = res["n_in"] * res["frac"] - 5.0 * res["max_gap_mm"]
            if best is None or score > best[0]:
                best = (score, res, axes[k], P1g[k])
        if best is None:
            break
        _, res, _axis, _p0 = best
        inliers = res["inliers"]
        axis_ref = _pca_axis(inliers)
        center = inliers.mean(axis=0)
        proj_ref = (inliers - center) @ axis_ref
        results.append(
            {
                "axis": axis_ref,
                "center": center,
                "n": int(inliers.shape[0]),
                "span": float(proj_ref.max() - proj_ref.min()),
                "frac": res["frac"],
                "max_gap_mm": res["max_gap_mm"],
            }
        )
        remaining = remaining[~res["mask"]]
    return results


def _ras_point_to_kji(
    point_ras: np.ndarray,
    ras_to_ijk_fn: Callable,
) -> np.ndarray:
    """Convert one RAS point to integer KJI indices."""
    ijk = np.asarray(ras_to_ijk_fn(point_ras), dtype=float).reshape(3)
    return np.rint([ijk[2], ijk[1], ijk[0]]).astype(int)


def _hd_at(
    point_ras: np.ndarray,
    head_distance_map_kji: np.ndarray,
    ras_to_ijk_fn: Callable,
) -> float:
    kji = _ras_point_to_kji(point_ras, ras_to_ijk_fn)
    shape = head_distance_map_kji.shape
    for ax in range(3):
        if kji[ax] < 0 or kji[ax] >= shape[ax]:
            return float("nan")
    return float(head_distance_map_kji[kji[0], kji[1], kji[2]])


def _annotate_depth(
    lines: List[dict],
    head_distance_map_kji: np.ndarray,
    ras_to_ijk_fn: Callable,
    cfg: BoltRansacConfig,
) -> None:
    for L in lines:
        hd = _hd_at(L["center"], head_distance_map_kji, ras_to_ijk_fn)
        L["hd_center"] = hd if np.isfinite(hd) else -1.0
        hd_best = L["hd_center"]
        for sign in (+1.0, -1.0):
            for t in cfg.axis_depth_probe_steps_mm:
                pt = L["center"] + sign * t * L["axis"]
                v = _hd_at(pt, head_distance_map_kji, ras_to_ijk_fn)
                if np.isfinite(v) and v > hd_best:
                    hd_best = v
        L["hd_inward"] = hd_best


def _position_filter(lines: List[dict], cfg: BoltRansacConfig) -> List[dict]:
    kept = []
    for L in lines:
        hd = L["hd_center"]
        if not (cfg.shell_min_mm <= hd <= cfg.shell_max_mm):
            continue
        if (L["hd_inward"] - hd) < cfg.axis_depth_delta_mm:
            continue
        kept.append(L)
    return kept


def _compute_support_mask(L: dict, cloud: np.ndarray, cfg: BoltRansacConfig) -> None:
    v = cloud - L["center"]
    proj = v @ L["axis"]
    perp = v - np.outer(proj, L["axis"])
    dist = np.linalg.norm(perp, axis=1)
    half_span = L["span"] / 2.0 + 1.0
    in_span = (proj >= -half_span) & (proj <= half_span)
    mask = (dist < cfg.inlier_tol_mm) & in_span
    L["support_mask"] = mask
    L["support_n"] = int(mask.sum())


def _dedup(lines: List[dict], cfg: BoltRansacConfig) -> List[dict]:
    """Drop same-bolt fragments and same-electrode contact-row segments."""
    # Sort shallow-first (prefer bolts) then by support size.
    lines = sorted(lines, key=lambda L: (L["hd_center"], -L["support_n"]))
    deduped: List[dict] = []
    for L in lines:
        is_dup = False
        for K in deduped:
            inter = int(np.logical_and(L["support_mask"], K["support_mask"]).sum())
            denom = min(L["support_n"], K["support_n"])
            ov = (inter / denom) if denom > 0 else 0.0
            if ov >= cfg.support_overlap_frac:
                is_dup = True
                break
            cos = abs(float(np.dot(L["axis"], K["axis"])))
            cos = min(1.0, max(-1.0, cos))
            ang = float(np.degrees(np.arccos(cos)))
            if ang > cfg.collinear_angle_deg:
                continue
            v = L["center"] - K["center"]
            perp = v - np.dot(v, K["axis"]) * K["axis"]
            pd = float(np.linalg.norm(perp))
            if pd <= cfg.collinear_perp_mm:
                is_dup = True
                break
        if not is_dup:
            deduped.append(L)
    return deduped


def find_bolt_candidates(
    *,
    arr_kji: np.ndarray,
    bolt_metal_mask_kji: np.ndarray,
    head_distance_map_kji: np.ndarray,
    ijk_kji_to_ras_fn: Callable,
    ras_to_ijk_fn: Callable,
    cfg: BoltRansacConfig | None = None,
) -> List[BoltCandidate]:
    """Detect bolt candidates in a CT volume.

    Parameters
    ----------
    arr_kji:
        The raw CT array (K, J, I). Only used to trim the cloud when the bolt
        metal mask is unusually empty; filtering operates on masks.
    bolt_metal_mask_kji:
        Boolean mask of voxels above the bolt HU threshold, produced by the
        mask stage.
    head_distance_map_kji:
        Signed head-distance (mm) from the outer scalp, produced by the mask
        stage. Should be the post-sinus-composition version so shell filtering
        makes sense near sinuses.
    ijk_kji_to_ras_fn:
        Converts (N, 3) KJI integer indices to (N, 3) RAS coordinates (mm).
    ras_to_ijk_fn:
        Converts a single (3,) RAS point to a (3,) IJK float tuple.
    cfg:
        Tuning parameters. Defaults match the values validated on T1 and T22.
    """
    cfg = cfg or BoltRansacConfig()
    bolt_mask = np.asarray(bolt_metal_mask_kji, dtype=bool)
    if not bolt_mask.any():
        return []

    idx = np.argwhere(bolt_mask)
    cloud = np.asarray(ijk_kji_to_ras_fn(idx.astype(float)), dtype=float)
    if cloud.shape[0] < cfg.min_inliers:
        return []

    lines = _ransac_lines(cloud, cfg)
    if not lines:
        return []

    _annotate_depth(lines, head_distance_map_kji, ras_to_ijk_fn, cfg)
    kept = _position_filter(lines, cfg)
    for L in kept:
        _compute_support_mask(L, cloud, cfg)
    deduped = _dedup(kept, cfg)

    results: List[BoltCandidate] = []
    for L in deduped:
        results.append(
            BoltCandidate(
                center_ras=np.asarray(L["center"], dtype=float),
                axis_ras=np.asarray(L["axis"], dtype=float),
                span_mm=float(L["span"]),
                n_inliers=int(L["support_n"]),
                hd_center_mm=float(L["hd_center"]),
                hd_inward_max_mm=float(L["hd_inward"]),
                fill_frac=float(L["frac"]),
                max_gap_mm=float(L["max_gap_mm"]),
                support_mask=np.asarray(L["support_mask"], dtype=bool),
            )
        )
    return results
