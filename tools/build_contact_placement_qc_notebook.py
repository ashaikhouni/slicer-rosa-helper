"""Generate slicer-rosa-helper/notebooks/v1_seeds_v2_placement_qc.ipynb."""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


def md(*lines: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _to_lines(lines),
    }


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _to_lines(lines),
    }


def _to_lines(lines):
    """Normalize a tuple of multi-line strings into nbformat's list-of-lines."""
    blob = "\n".join(dedent(s).strip("\n") for s in lines)
    if not blob:
        return []
    parts = blob.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


cells: list[dict] = []

cells.append(md("""
# v1 seeds → v2 placement QC

Layer `place_contacts_for_seed_v2` on top of v1 auto trajectory emissions and
study the score surface (corr, n_covered/n_slots, bolt_source, placed_hu_*,
per-model margin) for matched + orphan emissions. **No gating** — `min_corr=0`,
`min_slot_hu_mean=0`, `max_slot_cc_volume_p90_mm3=None` — we want to see what
bone / cross-shank chains look like in the score so we can design components
that separate them.

Change `SUBJECT_ID` below and re-run all cells.

Resume context: `handoff_contact_scoring_2026-05-09_v2.md`.
"""))

cells.append(code("""
%matplotlib inline
import os, sys
from pathlib import Path

REPO = Path.cwd()
while REPO.name != "slicer-rosa-helper" and REPO.parent != REPO:
    REPO = REPO.parent
sys.path.insert(0, str(REPO / "CommonLib"))
sys.path.insert(0, str(REPO / "tools"))

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
"""))

cells.append(md("""
## Subject + matching tolerances
"""))

cells.append(code("""
SUBJECT_ID = "AMC88"   # AMC88 / AMC91 / AMC135 / AMC136 / AMC137 / T22 / T1..T25
ANGLE_TOL_DEG = 12.0   # GT-axis match tolerance
PERP_TOL_MM   = 8.0
"""))

cells.append(md("""
## Resolve subject paths + GT

AMC subjects + T22 live under `ROSA_AMC_TESTING_ROOT` with `<SID>/*_CT.nii.gz`
and `<SID>/Electrodes/*.dat`. T-series subjects live under `ROSA_SEEG_DATASET`
and use the `subjects.tsv` manifest.
"""))

cells.append(code("""
AMC_ROOT = Path(os.environ.get("ROSA_AMC_TESTING_ROOT", "/Users/ammar/Documents/testing"))
SEEG_ROOT = Path(os.environ.get("ROSA_SEEG_DATASET", "/Users/ammar/Dropbox/thalamus_subjects/seeg_localization"))


def _gt_axis_from_contacts(contacts):
    pts = np.asarray(contacts, dtype=float)
    cm = pts.mean(axis=0)
    cn = pts - cm
    _, _, vh = np.linalg.svd(cn, full_matrices=False)
    d = vh[0] / np.linalg.norm(vh[0])
    pr = cn @ d
    if pr[0] > pr[-1]:
        d = -d
        pr = -pr
    return cm + d * float(pr.min()), cm + d * float(pr.max())


def _load_dat(path):
    out = []
    for ln in Path(path).read_text().splitlines():
        s = ln.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) != 3:
            break
        try:
            out.append([float(parts[0]), float(parts[1]), float(parts[2])])
        except ValueError:
            break
    return np.asarray(out, dtype=float)


def _parse_curated_gt(path: Path):
    \"\"\"Parse rosa_helper-aligned export → list of GT shanks. Coordinates
    are in WORLD_RAS (columns 6-8). Each row is one contact; rows grouped
    by `trajectory` form a shank.\"\"\"
    out_by_traj: dict[str, list[dict]] = {}
    for ln in path.read_text().splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split(",")
        if len(parts) < 9:
            continue
        try:
            x, y, z = float(parts[6]), float(parts[7]), float(parts[8])
        except ValueError:
            continue
        traj = parts[0]
        model = parts[12] if len(parts) >= 13 else None
        out_by_traj.setdefault(traj, []).append({"x": x, "y": y, "z": z, "model": model})
    out = []
    for name, contacts in sorted(out_by_traj.items()):
        if len(contacts) < 2:
            continue
        pts = np.asarray([[c["x"], c["y"], c["z"]] for c in contacts], dtype=float)
        s, e = _gt_axis_from_contacts(pts)
        out.append({"name": name, "contacts_ras": pts,
                    "start_ras": s, "end_ras": e,
                    "model_id": contacts[0].get("model")})
    return out


def resolve_subject(sid: str):
    amc_dir = AMC_ROOT / sid
    if amc_dir.is_dir():
        ct = next(iter(amc_dir.glob("*_CT.nii.gz")), None) or next(iter(amc_dir.glob("*.nii.gz")), None)
        elec = amc_dir / "Electrodes"
        if not elec.is_dir():
            elec = amc_dir / "electrodes"
        gt = []
        for f in sorted(elec.glob("*.dat")):
            if f.stem.lower() == "elecpointset":
                continue
            pts = _load_dat(f)
            if pts.shape[0] >= 2:
                s, e = _gt_axis_from_contacts(pts)
                gt.append({"name": f.stem, "contacts_ras": pts, "start_ras": s, "end_ras": e})
        strategy = "dixi" if sid == "T22" else "pmt_35"
        return str(ct), gt, strategy

    # T-series: prefer user-curated GT (rosa_helper_import/<SID>/<SID>_GT_aligned_world_coords.txt)
    # over the auto-snapped manifest GT. The curated GT lives in the
    # post_registered_ct frame, so load that CT (column source_ct_file in
    # subjects.tsv) instead of ct_path.
    from eval_seeg_localization import iter_subject_rows, load_reference_ground_truth_shanks
    rows = iter_subject_rows(SEEG_ROOT, {sid})
    if not rows:
        raise FileNotFoundError(f"{sid} not found in AMC root or T-series manifest")
    row = rows[0]
    curated = SEEG_ROOT / "contact_label_dataset" / "rosa_helper_import" / sid / f"{sid}_GT_aligned_world_coords.txt"
    if curated.exists():
        ct_path = row.get("source_ct_file") or row["ct_path"]
        gt = _parse_curated_gt(curated)
        print(f"[gt] using curated rosa_helper export: {len(gt)} shanks from {curated.name}")
        return ct_path, gt, "dixi"

    shanks, _ = load_reference_ground_truth_shanks(row)
    gt = [{"name": s.shank,
           "contacts_ras": np.asarray(s.contacts_ras, dtype=float),
           "start_ras": np.asarray(s.start_ras, dtype=float),
           "end_ras":   np.asarray(s.end_ras,   dtype=float)} for s in shanks]
    return row["ct_path"], gt, "dixi"


CT_PATH, GT, STRATEGY = resolve_subject(SUBJECT_ID)
print(f"{SUBJECT_ID}: CT={CT_PATH}")
print(f"          GT shanks: {len(GT)}    strategy: {STRATEGY}")
"""))

cells.append(md("""
## Compute features + extract bolts + load library
"""))

cells.append(code("""
from shank_core.io import image_ijk_ras_matrices
from rosa_detect import guided_fit_engine as gfe
from rosa_detect.candidate_seeds.metal_evidence import compute_metal_evidence_volume
from rosa_detect.primitives.bolt_anchor import (
    extract_bolt_candidates, METAL_BOLT_THRESHOLD, BOLT_HULL_PROXIMITY_MM,
)
from rosa_core import load_electrode_library
from rosa_core.electrode_classifier import filter_models_for_strategy

img = sitk.ReadImage(CT_PATH)
i2r_in, r2i_in = image_ijk_ras_matrices(img)
features = gfe.compute_features(img, np.asarray(i2r_in), np.asarray(r2i_in))
i2r = np.asarray(features["ijk_to_ras_mat"])
r2i = np.asarray(features["ras_to_ijk_mat"])

metal_evidence = compute_metal_evidence_volume(features["log"], features["ct_arr_kji"])
bolts, _ = extract_bolt_candidates(
    features["log"], features["head_distance"], i2r, img.GetSpacing(),
    ras_to_ijk_mat=r2i, ct_arr=metal_evidence,
    hu_threshold=METAL_BOLT_THRESHOLD, hull_proximity_mm=BOLT_HULL_PROXIMITY_MM,
)

library = load_electrode_library()
lib_models = filter_models_for_strategy(library["models"], STRATEGY)
print(f"bolts: {len(bolts)}   library models ({STRATEGY}): {len(lib_models)}")
"""))

cells.append(md("""
## Run v1 detection (production path, untouched)
"""))

cells.append(code("""
from rosa_detect.service import run_contact_pitch_v1_with_features

ctx = {"img": features["img"], "ijk_to_ras_4x4": i2r, "ras_to_ijk_4x4": r2i}
det, _ = run_contact_pitch_v1_with_features(ctx)
v1_trajs = list(det.get("trajectories") or [])
print(f"v1 emitted {len(v1_trajs)} trajectories  (status={det.get('status')})")
for ti, t in enumerate(v1_trajs):
    print(f"  [{ti:2d}] {t.get('confidence_label','?'):>6s}  "
          f"model={t.get('electrode_model','-')!s:<14s}  "
          f"start={np.round(t['start_ras'],1).tolist()}  "
          f"end={np.round(t['end_ras'],1).tolist()}")
"""))

cells.append(md("""
## Staged contact placement

Re-implementation of v2's flow as a composition of small named stages, each a
function that takes a `PlacementCtx` and returns an updated one. No bidirectional
retry — the seed is assumed to be entry→target (the seeder's contract). Bolt-less
is a first-class anchor outcome, not a fallback exception.

Stages: **anchor → refine → sample → pick → place → score**.

Swap implementations by passing different stage functions to the composer
(e.g. `sample_fn=sample_neg_log_max` for LoG instead of HU).
"""))

cells.append(code("""
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from rosa_core.contact_placement import sample_disk_along_polyline, estimate_bolt_end_from_metal_mass
from rosa_core.matched_filter import matched_filter_pick, MatchedFilterResult
from rosa_core.contact_placement import (
    WALK_STEP_MM, WALK_DISK_RADIUS_MM, WALK_N_RADII, WALK_N_ANGLES, WALK_TIP_PAD_MM, WALK_HU_MIN,
    DEGENERATE_CONTACT_ZONE_MM,
    snap_centerline_to_centroid as _snap_centerline_to_centroid,
    extend_centerline_tail as _extend_centerline_tail,
    project_to_polyline_arc as _project_to_polyline_arc,
)


@dataclass
class PlacementCtx:
    seed_start: np.ndarray
    seed_end: np.ndarray
    features: dict
    library_models: list[dict]
    # Optional global bolt CC list (subject-level, from the stage-1
    # `extract_bolt_candidates` pass). Independent of the per-emission
    # metal-mass walker that produces `bolt_end_arc` — used as a
    # confidence-only "reliable bolt present?" signal in score_compound.
    bolts: list[dict] | None = None
    # Other emissions' centerlines (post-anchor + post-refine) for the
    # cross-shank ownership mask in Stage C. When set, each disk voxel
    # is owned by the centerline it's closest to; voxels owned by another
    # shank are zeroed before aggregating per-arc walker stats.
    other_centerlines: list[np.ndarray] | None = None
    # Seeder-side metadata (carried through from the v1 trajectory dict so
    # the compound score can fuse seeder confidence with the placement-side
    # signals — same direction as the user's 5-mode input API).
    seeder_confidence: float = 0.0
    seeder_label: str = ""                 # "high" | "medium" | "low" | ""
    seeder_model: str | None = None
    centerline: np.ndarray | None = None
    bolt_end_arc: float = 0.0
    bolt_source: str = "unknown"           # "metal" | "bolt_less" — walker outcome only
    walk_arcs: np.ndarray | None = None
    walk_signal: np.ndarray | None = None
    signal_kind: str = ""                  # "hu_max" | "neg_log_max" | ...
    match: MatchedFilterResult | None = None
    placed_ras: list[np.ndarray] = field(default_factory=list)
    score_components: dict[str, Any] = field(default_factory=dict)
"""))

cells.append(md("""
### Stage A — anchor

Two distinct walker concepts in this codebase (don't conflate):

- The **bolt-end walker** (`estimate_bolt_end_from_metal_mass`) walks the seed
  looking for where the metal-mass tail drops. **Only runs for metal anchors.**
  `anchor_bolt_less` skips it entirely — there's no bolt to walk to.
- The **disk-stat sampler** (Stage C) walks the centerline emitting a 1D HU/LoG
  signal for the matched filter. Runs for both anchor types — it's not finding
  a bolt, it's collecting the placement signal.

`anchor_metal` runs the bolt-end walker + v2's degenerate-contact-zone reject.
`anchor_bolt_less` is the explicit straight-seed fallback: the entire emitter
seed becomes the centerline, `bolt_end_arc=0.0`, `max_extend=0.0`. The matched
filter scores across the whole centerline. **No reverse retry** — seeders
emit canonical entry→target direction; manual-mode flips are the caller's
problem to fix upstream.
"""))

cells.append(code("""
def anchor_metal(ctx: PlacementCtx) -> PlacementCtx | None:
    try:
        be = estimate_bolt_end_from_metal_mass(
            ctx.seed_start, ctx.seed_end,
            features=ctx.features, library_models=ctx.library_models,
        )
    except Exception:
        return None
    be_arc = be.get("bolt_end_arc_mm")
    cp = be.get("centerline")
    if be_arc is None or cp is None:
        return None
    cp = np.asarray(cp, dtype=float)
    cp_total = float(np.linalg.norm(np.diff(cp, axis=0), axis=1).sum())
    if cp_total - float(be_arc) < DEGENERATE_CONTACT_ZONE_MM:
        return None
    return replace(ctx, centerline=cp, bolt_end_arc=float(be_arc), bolt_source="metal")


def anchor_bolt_less(ctx: PlacementCtx) -> PlacementCtx:
    cl = np.vstack([ctx.seed_start, ctx.seed_end])
    return replace(ctx, centerline=cl, bolt_end_arc=0.0, bolt_source="bolt_less")


def stage_anchor(ctx: PlacementCtx) -> PlacementCtx:
    return anchor_metal(ctx) or anchor_bolt_less(ctx)
"""))

cells.append(md("""
### Stage B — refine centerline

For metal anchors: snap polynomial centerline to the LoG-centroid track
(recovers axes 1–2 mm off the actual electrode). For bolt-less: no-op (we have
no reliable basis to refine a 2-point straight seed yet — research follow-up).
"""))

cells.append(code("""
def refine_log_snap(ctx: PlacementCtx) -> PlacementCtx:
    \"\"\"LoG centroid snap. When `ctx.other_centerlines` is set, mask LoG
    voxels closer to a neighboring centerline before computing the
    centroid — prevents the snap from drifting toward a passing shank.\"\"\"
    if ctx.bolt_source != "metal":
        return ctx
    log_arr = ctx.features.get("log")
    if log_arr is None:
        return ctx
    r2i = np.asarray(ctx.features["ras_to_ijk_mat"], dtype=float)
    if not ctx.other_centerlines:
        snapped = _snap_centerline_to_centroid(ctx.centerline, log_arr, r2i)
        return replace(ctx, centerline=np.asarray(snapped, dtype=float))
    snapped = _snap_centerline_owned(ctx.centerline, log_arr, r2i,
                                       others=ctx.other_centerlines)
    return replace(ctx, centerline=np.asarray(snapped, dtype=float))


def _snap_centerline_owned(centerline, log_arr_kji, r2i, *, others,
                              snap_radius_mm: float = 4.0,
                              step_mm: float = 0.5,
                              log_threshold: float = 500.0,
                              n_radii: int = 4, n_angles: int = 16,
                              smooth_window: int = 5) -> np.ndarray:
    \"\"\"Ownership-aware variant of `_snap_centerline_to_centroid`. Same
    centroid-of-bright-LoG logic, but at each arc step we discard disk
    voxels that sit closer to a neighbor's centerline than ours. Defaults
    mirror v2's `_snap_centerline_to_centroid` defaults.\"\"\"
    from rosa_core.volume_sampling import sample_trilinear_batch
    from scipy.ndimage import uniform_filter1d

    cl = np.asarray(centerline, dtype=float)
    diffs = np.diff(cl, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(cum[-1])
    arcs = np.arange(0.0, total + 0.5 * step_mm, step_mm)
    snapped = np.zeros((len(arcs), 3), dtype=float)

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
    dist_self = np.sqrt(off_u ** 2 + off_v ** 2)
    others_arr = [np.asarray(o, dtype=float) for o in others if o is not None and len(o) >= 2]

    for ai, t in enumerate(arcs):
        i = int(np.searchsorted(cum, t, side="right") - 1)
        i = max(0, min(i, len(diffs) - 1))
        t_frac = (t - cum[i]) / max(seg_lens[i], 1e-9)
        center = cl[i] + t_frac * diffs[i]
        tangent = diffs[i] / max(seg_lens[i], 1e-9)
        any_v = np.array([1.0, 0.0, 0.0]) if abs(tangent[0]) <= 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(tangent, any_v); u /= max(np.linalg.norm(u), 1e-9)
        v = np.cross(tangent, u);     v /= max(np.linalg.norm(v), 1e-9)
        pts = center[None, :] + off_u[:, None] * u[None, :] + off_v[:, None] * v[None, :]

        dist_other = np.full(n_per_disk, np.inf, dtype=float)
        for ocl in others_arr:
            dist_other = np.minimum(dist_other, _min_dist_pts_to_polyline(pts, ocl))
        owned = dist_self <= dist_other

        log_vals = sample_trilinear_batch(log_arr_kji, r2i, pts)
        sig = -log_vals
        valid = np.isfinite(sig) & (sig > log_threshold) & owned
        if np.any(valid):
            w = sig[valid] - log_threshold
            mu = float((w * off_u[valid]).sum() / w.sum())
            mv = float((w * off_v[valid]).sum() / w.sum())
            snapped[ai] = center + mu * u + mv * v
        else:
            snapped[ai] = center
    if smooth_window > 1:
        snapped = uniform_filter1d(snapped, size=smooth_window, axis=0, mode="nearest")
    return snapped


def refine_noop(ctx: PlacementCtx) -> PlacementCtx:
    return ctx
"""))

cells.append(md("""
### Stage C — sample disk-stat signal along centerline

Two interchangeable implementations: HU max-disk (positive polarity, `ct_arr_kji`)
and −LoG max-disk (negative polarity, `log`). Both return per-arc max-brightness
in a perpendicular disk. Plug-and-play swap point.
"""))

cells.append(code("""
LOG_TOTAL_THRESHOLD = 100.0  # see sample_disk_along_polyline docstring (LoG mode)
WALK_AGGREGATOR = "p90"   # "max" | "p90" | "p75" | "median". p90 keeps pick
                            # accuracy identical to max but raises med_corr ~0.03
                            # by suppressing single-voxel HU spikes.


def _min_dist_pts_to_polyline(pts: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    \"\"\"Return min distance from each point in ``pts`` (n, 3) to any
    segment of ``polyline`` (m, 3). Vectorized closed-form distance to
    each segment, then min across segments.\"\"\"
    if len(polyline) < 2:
        return np.full(pts.shape[0], np.inf, dtype=float)
    p = polyline.astype(float)
    a = p[:-1]                        # (m-1, 3) segment starts
    b = p[1:]                         # (m-1, 3) segment ends
    ab = b - a                        # (m-1, 3)
    ab_len2 = np.einsum("ij,ij->i", ab, ab)  # (m-1,)
    # Project each pt onto each segment.
    ap = pts[:, None, :] - a[None, :, :]  # (n, m-1, 3)
    t = np.einsum("nmi,mi->nm", ap, ab) / np.maximum(ab_len2[None, :], 1e-12)
    t = np.clip(t, 0.0, 1.0)
    closest = a[None, :, :] + t[..., None] * ab[None, :, :]  # (n, m-1, 3)
    diffs = pts[:, None, :] - closest                         # (n, m-1, 3)
    dists = np.sqrt(np.einsum("nmi,nmi->nm", diffs, diffs))   # (n, m-1)
    return dists.min(axis=1)


def _aggregate_disk(samples: np.ndarray, mask: np.ndarray, kind: str) -> float:
    finite = np.isfinite(samples) & mask
    if not finite.any():
        return 0.0
    s = samples[finite]
    if kind == "max":
        return float(s.max())
    if kind == "p90":
        return float(np.percentile(s, 90))
    if kind == "p75":
        return float(np.percentile(s, 75))
    if kind == "median":
        return float(np.median(s))
    return float(s.max())


def _walk(ctx: PlacementCtx, *, volume, polarity: str, total_threshold: float):
    \"\"\"Walk the centerline emitting per-arc disk stats (default p90 — see
    WALK_AGGREGATOR). When ``ctx.other_centerlines`` is set, voxels closer
    to another shank's centerline than ours are zeroed before aggregating
    — the cross-shank ownership mask. Eliminates passing-shank artifacts.\"\"\"
    from rosa_core.volume_sampling import sample_trilinear_batch
    r2i = np.asarray(ctx.features["ras_to_ijk_mat"], dtype=float)
    cl = np.asarray(ctx.centerline, dtype=float)
    max_extend = WALK_TIP_PAD_MM if ctx.bolt_source == "metal" else 0.0
    cl_walk = _extend_centerline_tail(cl, max_extend) if max_extend > 0 else cl

    diffs = np.diff(cl_walk, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(cum[-1])
    arcs = np.arange(0.0, total + 0.5 * WALK_STEP_MM, WALK_STEP_MM)
    out = np.zeros(len(arcs), dtype=float)

    n_per = 1 + WALK_N_RADII * WALK_N_ANGLES
    off_u = np.zeros(n_per, dtype=float)
    off_v = np.zeros(n_per, dtype=float)
    idx = 1
    for r_idx in range(1, WALK_N_RADII + 1):
        rr = WALK_DISK_RADIUS_MM * r_idx / WALK_N_RADII
        for a_idx in range(WALK_N_ANGLES):
            ang = 2.0 * np.pi * a_idx / WALK_N_ANGLES
            off_u[idx] = rr * np.cos(ang)
            off_v[idx] = rr * np.sin(ang)
            idx += 1

    others = [np.asarray(ocl, dtype=float) for ocl in (ctx.other_centerlines or [])
                if ocl is not None and len(ocl) >= 2]
    dist_self = np.sqrt(off_u ** 2 + off_v ** 2)

    for ai, t in enumerate(arcs):
        i = int(np.searchsorted(cum, t, side="right") - 1)
        i = max(0, min(i, len(diffs) - 1))
        t_frac = (t - cum[i]) / max(seg_lens[i], 1e-9)
        center = cl_walk[i] + t_frac * diffs[i]
        tangent = diffs[i] / max(seg_lens[i], 1e-9)
        any_v = np.array([1.0, 0.0, 0.0]) if abs(tangent[0]) <= 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(tangent, any_v); u /= max(np.linalg.norm(u), 1e-9)
        v = np.cross(tangent, u);     v /= max(np.linalg.norm(v), 1e-9)
        pts = center[None, :] + off_u[:, None] * u[None, :] + off_v[:, None] * v[None, :]

        if others:
            dist_other = np.full(n_per, np.inf, dtype=float)
            for ocl in others:
                dist_other = np.minimum(dist_other, _min_dist_pts_to_polyline(pts, ocl))
            owned = dist_self <= dist_other
        else:
            owned = np.ones(n_per, dtype=bool)

        samples = sample_trilinear_batch(volume, r2i, pts)
        if polarity == "negative":
            samples = -samples
        out[ai] = _aggregate_disk(samples, owned, WALK_AGGREGATOR)

    return arcs, out


def sample_hu_max(ctx: PlacementCtx) -> PlacementCtx:
    arcs, sig = _walk(ctx, volume=ctx.features["ct_arr_kji"],
                      polarity="positive", total_threshold=WALK_HU_MIN)
    return replace(ctx, walk_arcs=arcs, walk_signal=sig, signal_kind="hu_max")


def sample_neg_log_max(ctx: PlacementCtx) -> PlacementCtx:
    arcs, sig = _walk(ctx, volume=ctx.features["log"],
                      polarity="negative", total_threshold=LOG_TOTAL_THRESHOLD)
    return replace(ctx, walk_arcs=arcs, walk_signal=sig, signal_kind="neg_log_max")
"""))

cells.append(md("""
### Stage D — pick library model

Pearson NCC against the library comb-template (existing `matched_filter_pick`).
Returns the full `MatchedFilterResult` — winning model_id, n_slots, n_covered,
slot arcs, corr.
"""))

cells.append(code("""
def pick_matched_filter(ctx: PlacementCtx) -> PlacementCtx:
    cl = np.asarray(ctx.centerline, dtype=float)
    cl_max = float(np.linalg.norm(np.diff(cl, axis=0), axis=1).sum())
    max_extend = WALK_TIP_PAD_MM if ctx.bolt_source == "metal" else 0.0
    res = matched_filter_pick(
        ctx.walk_arcs, ctx.walk_signal, ctx.library_models,
        bolt_end_arc=ctx.bolt_end_arc,
        profile_end_arc=cl_max,
        max_extend_tip_mm=max_extend,
    )
    return replace(ctx, match=res)
"""))

cells.append(md("""
### Stage D.2 — re-pick by extent-weighted score

Raw matched-filter corr ranks templates by *local* fit, so a short electrode
fitting a contiguous slice of a long signal can beat the full-extent template
by 0.02–0.05. T18/X11 (15-peak DIXI-15AM signal): raw corr picks DIXI-5AM
or DIXI-18CM. Re-rank by `corr × min(1, model_extent / contact_zone_extent)`
— if the extent-weighted top differs, re-run matched_filter on that model so
its `slot_arcs` (used by Stage E) align with the better template.
"""))

cells.append(code("""
# Peak-count detection retired 2026-05-09: replaced by `corr × √(n_cov/max_n_cov)`
# denominator-corrected re-rank in `pick_extent_aware`. The dn correction
# fixes the same NCC-denominator-bias case (T18/X11 short-template-cherry-picks
# clean section) without find_peaks/gradient detection. Cleaner architecture,
# +1 model accuracy on T18 (HU 12→13 / 13).


def pick_extent_aware(ctx: PlacementCtx) -> PlacementCtx:
    \"\"\"Re-rank by `corr × √(n_covered / max_n_covered)` — denominator
    correction. Compensates for Pearson NCC's bias toward shorter templates:
    when ||t|| ∝ √n_slots, a 5-slot template aligned with 5 of 15 visible
    peaks gets an unfair denominator boost. Multiplying by √(n_cov/max_n_cov)
    re-normalizes — equivalent to evaluating each template against the
    longest template's denominator.

    Margin defer: when the matched-filter top has clear margin (>0.05),
    trust it — only re-rank ties.

    Validated on 7 subjects (79 GT shanks): plain matched filter
    78.5%/82.3% (HU/LoG); +dn 79.7%/83.5%. Replaces the prior peak-count
    re-rank, which was simpler-on-paper but regressed T18 HU 12→11.
    \"\"\"
    if ctx.match is None or ctx.walk_arcs is None or ctx.centerline is None:
        return ctx
    pmc = per_model_corrs(ctx)
    if len(pmc) < 2:
        return ctx

    PICK_OVERRIDE_MARGIN = 0.05
    if pmc[0][3] - pmc[1][3] > PICK_OVERRIDE_MARGIN:
        return ctx

    max_cov = max(t[2] for t in pmc)
    if max_cov == 0:
        return ctx
    weighted = [(t[0], t[3] * float(np.sqrt(t[2] / max_cov))) for t in pmc]
    weighted.sort(key=lambda x: -x[1])
    preferred_id = weighted[0][0]
    if preferred_id == ctx.match.best_model_id:
        return ctx

    lookup = {str(m.get("id") or ""): m for m in ctx.library_models}
    preferred_model = lookup.get(preferred_id)
    if preferred_model is None:
        return ctx
    cl = np.asarray(ctx.centerline, dtype=float)
    cl_total = float(np.linalg.norm(np.diff(cl, axis=0), axis=1).sum())
    max_extend = WALK_TIP_PAD_MM if ctx.bolt_source == "metal" else 0.0
    new_match = matched_filter_pick(
        ctx.walk_arcs, ctx.walk_signal, [preferred_model],
        bolt_end_arc=ctx.bolt_end_arc,
        profile_end_arc=cl_total,
        max_extend_tip_mm=max_extend,
    )
    return replace(ctx, match=new_match)
"""))

cells.append(md("""
### Stage E — place contacts on centerline

Project picked slot arcs onto the centerline → RAS. Trivial.
"""))

cells.append(code("""
def _polyline_at_arc(polyline: np.ndarray, arc: float) -> np.ndarray:
    diffs = np.diff(polyline, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    arc = float(np.clip(arc, 0.0, cum[-1]))
    i = int(np.searchsorted(cum, arc, side="right") - 1)
    i = min(max(i, 0), len(diffs) - 1)
    t = (arc - cum[i]) / max(seg_lens[i], 1e-9)
    return polyline[i] + t * diffs[i]


def place_at_match(ctx: PlacementCtx) -> PlacementCtx:
    if ctx.match is None or ctx.match.slot_arcs is None or ctx.match.best_model_id is None:
        return ctx
    cl = np.asarray(ctx.centerline, dtype=float)
    placed = [_polyline_at_arc(cl, float(a)) for a in ctx.match.slot_arcs]
    return replace(ctx, placed_ras=placed)
"""))

cells.append(md("""
### Stage F — score

Compose components into a per-emission diagnostic dict. **Not gates** — these
are signals for the eventual confidence band. The composite `band` here is a
strawman to be tuned once we've eyeballed multiple subjects.

Components:
- `corr` — matched-filter NCC (primary)
- `n_covered_frac` — fraction of model slots inside the contact zone
- `bolt_zone_frac` — `bolt_end_arc / centerline_length`. High = mostly-bolt.
  ei=10's canonical failure mode.
- `bolt_source` — "metal" / "bolt_less"
- `placed_hu_mean` — average HU at placed contact centers
"""))

cells.append(code("""
from rosa_core.volume_sampling import sample_trilinear_at_ras


def per_model_corrs(ctx: PlacementCtx) -> list[tuple]:
    \"\"\"Score every library model against this ctx's signal. Output:
    list of (model_id, n_slots, n_covered, corr) sorted by descending corr.\"\"\"
    if ctx.walk_arcs is None or ctx.walk_signal is None:
        return []
    cl = np.asarray(ctx.centerline, dtype=float)
    cl_max = float(np.linalg.norm(np.diff(cl, axis=0), axis=1).sum())
    max_extend = WALK_TIP_PAD_MM if ctx.bolt_source == "metal" else 0.0
    out = []
    for m in ctx.library_models:
        try:
            r = matched_filter_pick(
                ctx.walk_arcs, ctx.walk_signal, [m],
                bolt_end_arc=ctx.bolt_end_arc,
                profile_end_arc=cl_max,
                max_extend_tip_mm=max_extend,
            )
            out.append((str(m.get("id") or ""), int(r.n_slots), int(r.n_covered), float(r.corr)))
        except Exception:
            continue
    out.sort(key=lambda t: -t[3])
    return out


def _model_uniform_pitch(model: dict) -> tuple[float | None, bool]:
    \"\"\"Return (pitch_mm, uniform) for one library model. Uniform = all
    consecutive contact spacings within 0.1 mm.\"\"\"
    offs = model.get("contact_center_offsets_from_tip_mm")
    if not offs or len(offs) < 2:
        return None, False
    diffs = np.diff(np.sort(np.asarray(offs, dtype=float)))
    pitch = float(np.mean(diffs))
    return pitch, bool(np.all(np.abs(diffs - pitch) < 0.1))


def _zone_modulation(arcs: np.ndarray, sig: np.ndarray, bolt_end: float, cl_max: float) -> tuple[float, float, np.ndarray, np.ndarray]:
    \"\"\"Restrict signal to the contact-zone window [bolt_end, cl_max] and
    return (cv, ptp_mod, zone_arcs, zone_signal). cv = stddev/mean,
    ptp_mod = (max-min)/mean.\"\"\"
    mask = (arcs >= bolt_end) & (arcs <= cl_max)
    z_arcs = arcs[mask]
    z_sig = sig[mask]
    if z_sig.size < 4:
        return 0.0, 0.0, z_arcs, z_sig
    mean_s = float(np.mean(z_sig))
    if mean_s <= 1e-6:
        return 0.0, 0.0, z_arcs, z_sig
    cv = float(np.std(z_sig) / mean_s)
    ptp = float((np.max(z_sig) - np.min(z_sig)) / mean_s)
    return cv, ptp, z_arcs, z_sig


def _tube_like_frac(ctx: PlacementCtx, *,
                       percentile: float = 90.0,
                       tube_radius_mm: float = 1.0,
                       perp_max_mm: float = 2.5,
                       perp_step_mm: float = 0.5,
                       arc_step_mm: float = 1.0) -> float:
    \"\"\"Fraction of *high-HU* disk voxels (top ``100−percentile`` % by HU,
    sampled across the entire contact-zone disk) that sit within
    ``tube_radius_mm`` of the centerline.

    Self-calibrating — no fixed HU threshold; the cutoff is the top-decile
    of *this emission's* own disk samples. Cross-scanner HU-shift safe.

    Real shanks: top-HU voxels are the contacts → all near centerline → ~1.0.
    Bone-spike / bolt-only fakes: top-HU voxels are radially scattered → ~0.3-0.5.
    Anchor-independent (works for metal AND bolt_less).
    \"\"\"
    if ctx.centerline is None:
        return 0.0
    cl = np.asarray(ctx.centerline, dtype=float)
    if len(cl) < 2:
        return 0.0

    r2i = np.asarray(ctx.features["ras_to_ijk_mat"], dtype=float)
    ct_arr = ctx.features["ct_arr_kji"]
    diffs = np.diff(cl, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(cum[-1])

    zone_start = float(ctx.bolt_end_arc)
    zone_end = total
    if zone_end - zone_start < 1.0:
        return 0.0

    offsets = np.arange(-perp_max_mm, perp_max_mm + perp_step_mm, perp_step_mm)
    radials = np.sqrt(offsets[:, None] ** 2 + offsets[None, :] ** 2).ravel()

    from rosa_core.volume_sampling import sample_trilinear_batch
    arcs = np.arange(zone_start, zone_end, arc_step_mm)

    # Collect (radial_dist, hu) for every disk voxel across every arc step.
    all_radials = []
    all_hus = []
    for arc in arcs:
        i = int(np.searchsorted(cum, arc, side="right") - 1)
        i = max(0, min(i, len(diffs) - 1))
        t_frac = (arc - cum[i]) / max(seg_lens[i], 1e-9)
        center = cl[i] + t_frac * diffs[i]
        tangent = diffs[i] / max(seg_lens[i], 1e-9)
        any_v = np.array([1.0, 0, 0]) if abs(tangent[0]) <= 0.9 else np.array([0, 1.0, 0])
        u = np.cross(tangent, any_v); u /= max(np.linalg.norm(u), 1e-9)
        v = np.cross(tangent, u);     v /= max(np.linalg.norm(v), 1e-9)

        pts = (center[None, None, :]
               + offsets[:, None, None] * u[None, None, :]
               + offsets[None, :, None] * v[None, None, :]).reshape(-1, 3)
        samples = sample_trilinear_batch(ct_arr, r2i, pts).ravel()
        finite = np.isfinite(samples)
        if finite.any():
            all_radials.append(radials[finite])
            all_hus.append(samples[finite])

    if not all_hus:
        return 0.0
    hus = np.concatenate(all_hus)
    rs  = np.concatenate(all_radials)
    cutoff = float(np.percentile(hus, percentile))
    high = hus >= cutoff
    if not high.any():
        return 0.0
    return float(np.mean(rs[high] <= tube_radius_mm))


def _pitch_power_frac(z_arcs: np.ndarray, z_sig: np.ndarray, pitch_mm: float) -> float:
    \"\"\"FFT power in a 3-bin band around f=1/pitch divided by total spectral
    power (excluding DC). Detrend + Hann window first. Used for one
    contiguous signal slice; the per-segment helper below calls it per
    cluster on multi-cluster (CM/BM) electrodes.\"\"\"
    if z_sig.size < 16 or pitch_mm <= 0:
        return 0.0
    from scipy.signal import detrend
    step = float(z_arcs[1] - z_arcs[0]) if z_arcs.size >= 2 else WALK_STEP_MM
    sig = detrend(z_sig) * np.hanning(z_sig.size)
    spec = np.abs(np.fft.rfft(sig)) ** 2
    freqs = np.fft.rfftfreq(z_sig.size, d=step)
    target = 1.0 / pitch_mm
    if freqs[-1] < target:
        return 0.0  # signal too short to resolve the pitch frequency
    idx = int(np.argmin(np.abs(freqs - target)))
    band = float(spec[max(0, idx - 1):idx + 2].sum())
    total = float(spec[1:].sum())  # exclude DC
    return band / total if total > 0 else 0.0


def _pitch_power_frac_per_segment(z_arcs: np.ndarray, z_sig: np.ndarray,
                                     slot_arcs: np.ndarray, pitch_mm: float, *,
                                     gap_thresh_mm: float = 5.0,
                                     min_slots: int = 4) -> tuple[float, int, int]:
    \"\"\"Per-cluster FFT for CM/BM (multi-segment) electrodes.

    Splits ``slot_arcs`` by gaps > ``gap_thresh_mm`` into clusters. For each
    cluster with ≥ ``min_slots`` contacts AND ≥ 16 walker samples, computes
    pitch FFT power frac and averages across reliable clusters.

    Uniform-pitch electrodes have a single cluster → reduces to single FFT.

    Returns ``(mean_power, n_reliable_segments, n_segments_total)``.
    \"\"\"
    if len(slot_arcs) < 2 or pitch_mm <= 0:
        return 0.0, 0, 0
    slots = np.sort(np.asarray(slot_arcs, dtype=float))
    diffs = np.diff(slots)
    cluster_breaks = np.where(diffs > gap_thresh_mm)[0]
    cluster_starts = np.concatenate([[0], cluster_breaks + 1])
    cluster_ends   = np.concatenate([cluster_breaks, [len(slots) - 1]])
    n_segments = len(cluster_starts)

    powers = []
    for cs, ce in zip(cluster_starts, cluster_ends):
        n_in_cluster = int(ce - cs + 1)
        if n_in_cluster < min_slots:
            continue
        first = float(slots[cs]) - 0.5 * pitch_mm
        last  = float(slots[ce]) + 0.5 * pitch_mm
        mask = (z_arcs >= first) & (z_arcs <= last)
        seg_arcs = z_arcs[mask]
        seg_sig  = z_sig[mask]
        if seg_sig.size < 16:
            continue
        powers.append(_pitch_power_frac(seg_arcs, seg_sig, pitch_mm))

    if not powers:
        return 0.0, 0, n_segments
    return float(np.mean(powers)), len(powers), n_segments


def score_simple(ctx: PlacementCtx) -> PlacementCtx:
    cl = np.asarray(ctx.centerline, dtype=float)
    cl_total = float(np.linalg.norm(np.diff(cl, axis=0), axis=1).sum()) if len(cl) >= 2 else 0.0
    bolt_zone_frac = ctx.bolt_end_arc / cl_total if cl_total > 0 else 0.0

    # placed-contact HU
    r2i = np.asarray(ctx.features["ras_to_ijk_mat"], dtype=float)
    ct_arr = ctx.features["ct_arr_kji"]
    placed_hu = []
    for p in ctx.placed_ras or []:
        try:
            v = float(sample_trilinear_at_ras(ct_arr, r2i, np.asarray(p, dtype=float)))
            if np.isfinite(v):
                placed_hu.append(v)
        except Exception:
            pass

    # Per-library-model corrs (includes the picked model). Used for uniformity
    # and stashed for the renderer + the comparison df so we don't recompute.
    pmc = per_model_corrs(ctx)
    if pmc:
        top1 = pmc[0][3]
        top2 = pmc[1][3] if len(pmc) > 1 else 0.0
        rest = [t[3] for t in pmc[1:]]
        # Uniformity: mean(rest) / top1. Approaches 1 when every model scores
        # the same (flat plateau, tie-break pick); → 0 when one model dominates.
        uniformity = float(np.mean(rest) / top1) if (rest and top1 > 1e-6) else 0.0
        uniformity = float(np.clip(uniformity, 0.0, 1.0))
        margin = float(top1 - top2)
    else:
        uniformity, margin = 1.0, 0.0

    # Zone modulation (CV + peak-to-peak / mean) on the contact-zone walker signal.
    cv = ptp_mod = 0.0
    pitch_power = 0.0
    pitch_mm = None
    uniform_pitch = False
    fft_n_reliable_segments = 0
    fft_n_total_segments = 0
    if ctx.walk_arcs is not None and ctx.walk_signal is not None:
        cv, ptp_mod, z_arcs, z_sig = _zone_modulation(
            ctx.walk_arcs, ctx.walk_signal, ctx.bolt_end_arc, cl_total,
        )
        m = ctx.match
        picked = next((mm for mm in ctx.library_models
                        if str(mm.get("id") or "") == (m.best_model_id if m else "")), None)
        if picked is not None:
            pitch_mm, uniform_pitch = _model_uniform_pitch(picked)
            # Per-segment FFT — handles CM/BM multi-cluster models. For
            # uniform-pitch electrodes this reduces to a single-segment FFT
            # (equivalent to the previous global FFT).
            if pitch_mm and z_sig.size and m is not None and m.slot_arcs is not None and len(m.slot_arcs) >= 2:
                pitch_power, fft_n_reliable_segments, fft_n_total_segments = (
                    _pitch_power_frac_per_segment(z_arcs, z_sig, m.slot_arcs, pitch_mm)
                )

    # Tube-likeness — fraction of contact-zone arcs where the brightest
    # voxel sits within 1mm of centerline. Anchor-independent (works for
    # both metal and bolt_less). Real shanks are 1mm-thin tubes.
    tube_frac = _tube_like_frac(ctx) if ctx.centerline is not None else 0.0

    m = ctx.match
    cps = {
        "corr":                  float(m.corr) if m else 0.0,
        "n_slots":               int(m.n_slots) if m else 0,
        "n_covered":             int(m.n_covered) if m else 0,
        "n_covered_frac":        float(m.n_covered) / m.n_slots if m and m.n_slots else 0.0,
        "bolt_zone_frac":        float(bolt_zone_frac),
        "bolt_source":           ctx.bolt_source,
        "placed_hu_mean":        float(np.mean(placed_hu)) if placed_hu else float("nan"),
        "placed_hu_min":         float(np.min(placed_hu))  if placed_hu else float("nan"),
        "n_placed":              len(ctx.placed_ras or []),
        "model_id":              m.best_model_id if m else None,
        # New components (this commit):
        "model_corr_uniformity": uniformity,
        "model_corr_margin":     margin,
        "zone_cv":               float(cv),
        "zone_ptp_mod":          float(ptp_mod),
        "pitch_power_frac":      float(pitch_power),
        "fft_n_segments":        int(fft_n_total_segments),
        "fft_n_reliable_segments": int(fft_n_reliable_segments),
        "pitch_mm":              float(pitch_mm) if pitch_mm else None,
        "model_uniform_pitch":   bool(uniform_pitch),
        "tube_like_frac":        float(tube_frac),
        # Stashed for the renderer + cmp_df so they don't recompute.
        "per_model_corr":        pmc,
    }
    return replace(ctx, score_components=cps)
"""))

cells.append(md("""
### Stage F.1 — score_cc_overlap

Independent confidence signal: does the seed actually intersect a hull-anchored
bolt CC, or just have metal-mass along its line? Walker tells us `bolt_end_arc`
(geometry — required for the matched filter). CC overlap tells us "is the
metal at the seed start a real bolt cluster touching the skull?" The two can
disagree — a seed running through a saturated-bone region passes the walker
test but has no hull-anchored CC nearby. That's the AMC91 ei=14 / ei=15
mostly-bolt-fake mode.

Cheap O(n_bolts) check: nearest CC centroid lateral distance to the seed line,
constrained to seeds where the projection lands near the seed start.
"""))

cells.append(code("""
CC_OVERLAP_PERP_SCALE_MM = 5.0     # lateral distance over which the score decays to 0
CC_OVERLAP_MAX_PERP_MM = 8.0       # beyond this, treat as no-match
CC_OVERLAP_MAX_ARC_PAST_BOLT_MM = 10.0  # centroid must project within bolt_end+this
                                         # along the *centerline* (not the seed)


def _project_to_polyline(pt: np.ndarray, polyline: np.ndarray) -> tuple[float, float]:
    \"\"\"Return (arc_along_polyline, perp_distance) for the closest point on
    polyline to pt. Vectorized over polyline segments.\"\"\"
    p = polyline.astype(float)
    if len(p) < 2:
        return 0.0, float(np.linalg.norm(pt - p[0])) if len(p) else float("inf")
    a = p[:-1]
    b = p[1:]
    ab = b - a
    ab_len2 = np.einsum("ij,ij->i", ab, ab)
    ap = pt - a
    t = np.einsum("ij,ij->i", ap, ab) / np.maximum(ab_len2, 1e-12)
    t = np.clip(t, 0.0, 1.0)
    closest = a + t[:, None] * ab
    diffs = pt - closest
    dists = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))
    seg_idx = int(np.argmin(dists))
    perp = float(dists[seg_idx])
    seg_lens = np.sqrt(ab_len2)
    cum_start = np.concatenate([[0.0], np.cumsum(seg_lens[:-1])])
    arc = float(cum_start[seg_idx] + t[seg_idx] * seg_lens[seg_idx])
    return arc, perp


def score_cc_overlap(ctx: PlacementCtx) -> PlacementCtx:
    cps = dict(ctx.score_components)
    cps.update({
        "cc_match": None, "cc_dist_mm": float("nan"),
        "cc_n_voxels": 0, "cc_overlap_score": 0.0,
    })
    if not ctx.bolts:
        return replace(ctx, score_components=cps)

    # Project CC centroids onto the CENTERLINE polyline (snapped/refined),
    # not the straight seed line — the centerline curves and seed-line u
    # over-shoots for bolts when the trajectory bends. T18/X03 had its
    # bolt CC centroid at seed-line u=39mm (excluded by any reasonable
    # u-band), but on the centerline its arc-along is ~22mm with perp~4mm.
    cl = np.asarray(ctx.centerline, dtype=float) if ctx.centerline is not None else np.vstack([ctx.seed_start, ctx.seed_end])
    cl_total = float(np.linalg.norm(np.diff(cl, axis=0), axis=1).sum()) if len(cl) >= 2 else 0.0
    arc_max = float(ctx.bolt_end_arc) + CC_OVERLAP_MAX_ARC_PAST_BOLT_MM \
              if ctx.bolt_source == "metal" else cl_total

    best_id, best_dist, best_n, best_arc = None, float("inf"), 0, float("nan")
    for b in ctx.bolts:
        pts = np.asarray(b.get("pts_ras"), dtype=float)
        if pts.size == 0:
            continue
        center = pts.mean(axis=0)
        arc_along, perp = _project_to_polyline(center, cl)
        if arc_along > arc_max:
            continue   # CC is past where the bolt could plausibly be
        if perp > CC_OVERLAP_MAX_PERP_MM:
            continue   # too far laterally — different shank's bolt
        if perp < best_dist:
            best_dist = perp
            best_id = b.get("id")
            best_n = int(b.get("n_vox") or 0)
            best_arc = arc_along

    if best_id is not None:
        overlap = max(0.0, 1.0 - best_dist / CC_OVERLAP_PERP_SCALE_MM)
        cps.update({
            "cc_match":         best_id,
            "cc_dist_mm":       best_dist,
            "cc_arc_along_mm":  best_arc,
            "cc_n_voxels":      best_n,
            "cc_overlap_score": float(overlap),
        })
    return replace(ctx, score_components=cps)
"""))

cells.append(md("""
### Stage F.2 — score_compound

Fuse per-emission signals into a single composite + 3-tier confidence band.
Inputs: matched-filter `corr`, FFT pitch power, peak-to-peak modulation,
matched-filter margin, bolt-source, and the **seeder's confidence label**
(carried in from v1 — `high`/`medium`/`low`).

Each component contributes a normalized [0, 1] sub-score; the composite is a
weighted sum with hand-set weights. The weights are tunable in this cell —
the goal is to eyeball the bands across subjects and adjust until matched
emissions cluster `high`, fakes cluster `low`. Do **not** treat this as
production-ready scoring yet.
"""))

cells.append(code("""
COMPOUND_WEIGHTS = {
    "corr":       0.20,   # matched-filter NCC (clipped to [0, 1])
    "fft":        0.20,   # per-segment pitch FFT power frac
    "tube":       0.15,   # tube-likeness (high HU within 1mm of centerline)
    "margin":     0.10,   # top1 - top2 (normalized at 0.15)
    "walker":     0.10,   # 1.0 if walker found a bolt-end (metal source)
    "cc_overlap": 0.15,   # global bolt CC near seed start (cropped-bolt-aware)
    "seeder":     0.10,   # v1 confidence label → {high:1, medium:0.6, low:0.3}
}
COMPOUND_BANDS = {"high": 0.70, "medium": 0.45}
SEEDER_LABEL_TO_SCORE = {"high": 1.0, "medium": 0.6, "low": 0.3, "": 0.5}
BOLT_ONLY_PENALTY_THRESHOLD = 0.5     # bz_frac above which to start penalizing
BOLT_ONLY_PENALTY_MAX = 0.20          # max penalty subtracted from composite


def score_compound(ctx: PlacementCtx, *, fft_norm: float | None = None) -> PlacementCtx:
    \"\"\"Compose a composite score + band from per-emission components.

    fft_norm: optional override for s_fft (the per-subject normalization
    post-pass passes subject-relative value). Reliability gate now uses
    per-segment FFT — `fft_n_reliable_segments >= 1` (handles CM/BM where
    each cluster is FFT'd independently). When unreliable (no segment had
    ≥4 slots and ≥16 walker samples), s_fft defaults to neutral 0.5.

    Cropped-bolt CC fix: when the walker found a bolt-end (bolt_source=metal)
    but cc_overlap_score is 0 (CC out of FOV), don't penalize — substitute
    neutral 0.5. Walker presence is sufficient bolt evidence.
    \"\"\"
    sc = ctx.score_components
    bz = sc.get("bolt_zone_frac", 0.0)
    fft_reliable = int(sc.get("fft_n_reliable_segments", 0)) >= 1

    if not fft_reliable:
        s_fft = 0.5
    elif fft_norm is not None:
        s_fft = float(np.clip(fft_norm, 0.0, 1.0))
    else:
        s_fft = float(np.clip(sc.get("pitch_power_frac", 0.0), 0.0, 1.0))

    # Cropped-bolt-aware CC overlap: walker says bolt → don't punish missing CC.
    raw_cc = float(np.clip(sc.get("cc_overlap_score", 0.0), 0.0, 1.0))
    walker_metal = sc.get("bolt_source") == "metal"
    if walker_metal and raw_cc == 0.0:
        s_cc = 0.5  # cropped or out-of-FOV: walker compensates
    else:
        s_cc = raw_cc

    sub = {
        "s_corr":       float(np.clip(sc.get("corr", 0.0), 0.0, 1.0)),
        "s_fft":        s_fft,
        "s_tube":       float(np.clip(sc.get("tube_like_frac", 0.0), 0.0, 1.0)),
        "s_margin":     float(np.clip(sc.get("model_corr_margin", 0.0) / 0.15, 0.0, 1.0)),
        "s_walker":     1.0 if walker_metal else 0.0,
        "s_cc_overlap": s_cc,
        "s_seeder":     SEEDER_LABEL_TO_SCORE.get(ctx.seeder_label, 0.5),
    }

    # Mostly-bolt penalty: real shanks like AMC91 10_stg can have bz_frac up
    # to ~0.7 but their FFT compensates. Fakes like AMC91 ei=14/15 sit at
    # bz_frac~0.7 with weak FFT — penalize iff fft_log can't carry them.
    if bz > BOLT_ONLY_PENALTY_THRESHOLD and sub["s_fft"] < 0.5:
        bolt_only_penalty = BOLT_ONLY_PENALTY_MAX * min(1.0,
            (bz - BOLT_ONLY_PENALTY_THRESHOLD) / (1.0 - BOLT_ONLY_PENALTY_THRESHOLD))
    else:
        bolt_only_penalty = 0.0

    composite = sum(COMPOUND_WEIGHTS[k] * sub[f"s_{k}"]
                     for k in ("corr", "fft", "tube", "margin", "walker", "cc_overlap", "seeder")
                     ) - bolt_only_penalty
    composite = float(np.clip(composite, 0.0, 1.0))
    band = ("high"   if composite >= COMPOUND_BANDS["high"] else
            "medium" if composite >= COMPOUND_BANDS["medium"] else
            "low")

    cps = dict(sc)
    cps["compound_score"] = composite
    cps["band"] = band
    cps["fft_reliable"] = fft_reliable
    cps["subscores"] = {**sub, "bolt_only_penalty": float(bolt_only_penalty)}
    return replace(ctx, score_components=cps)
"""))

cells.append(md("""
### Composer

Sequence the stages. Pass `sample_fn` (and any other stage swap) explicitly to
make the comparison points obvious at the call site.
"""))

cells.append(code("""
def place_v3(seed_start, seed_end, *, features, library_models,
             bolts: list[dict] | None = None,
             seeder_confidence: float = 0.0, seeder_label: str = "",
             seeder_model: str | None = None,
             refine_fn: Callable = refine_log_snap,
             sample_fn: Callable = sample_hu_max) -> PlacementCtx:
    ctx = PlacementCtx(
        seed_start=np.asarray(seed_start, dtype=float),
        seed_end=np.asarray(seed_end, dtype=float),
        features=features, library_models=library_models,
        bolts=bolts,
        seeder_confidence=float(seeder_confidence),
        seeder_label=str(seeder_label or ""),
        seeder_model=seeder_model,
    )
    ctx = stage_anchor(ctx)
    ctx = refine_fn(ctx)
    ctx = sample_fn(ctx)
    ctx = pick_matched_filter(ctx)
    ctx = pick_extent_aware(ctx)
    ctx = place_at_match(ctx)
    ctx = score_simple(ctx)
    ctx = score_cc_overlap(ctx)
    ctx = score_compound(ctx)
    return ctx
"""))

cells.append(md("""
## Run staged placement on every v1 trajectory (HU + LoG)
"""))

cells.append(code("""
# Two-pass: anchor + refine all emissions first so we can compute
# cross-shank ownership masks before sampling. Stage C reads
# ctx.other_centerlines and zeros voxels closer to a neighbor's snapped
# centerline (handles passing-shank interference, e.g. T18/X03).
#
# A pass-1.5 ownership-aware re-refine was tried 2026-05-09: helped
# T18 (HU +1) but regressed AMC135/T1/T2/T3/T4 (cumulative −7 HU). The
# voxel-ownership rule mis-attributes real shank voxels when initial
# centerlines aren't perfectly accurate. Reverted.
def _run_pass1(traj):
    es = np.asarray(traj["start_ras"], dtype=float)
    ee = np.asarray(traj["end_ras"],   dtype=float)
    ctx = PlacementCtx(
        seed_start=es, seed_end=ee,
        features=features, library_models=lib_models, bolts=bolts,
        seeder_confidence=float(traj.get("confidence") or 0.0),
        seeder_label=str(traj.get("confidence_label") or ""),
        seeder_model=traj.get("electrode_model"),
    )
    ctx = stage_anchor(ctx)
    ctx = refine_log_snap(ctx)
    return ctx


pass1_ctxs = [_run_pass1(t) for t in v1_trajs]
all_centerlines = [np.asarray(c.centerline, float) for c in pass1_ctxs if c.centerline is not None]


def _run_pass2(ei: int, sample_fn) -> PlacementCtx:
    base = pass1_ctxs[ei]
    others = [cl for j, cl in enumerate(all_centerlines) if j != ei]
    ctx = replace(base, other_centerlines=others)
    ctx = sample_fn(ctx)
    ctx = pick_matched_filter(ctx)
    ctx = pick_extent_aware(ctx)
    ctx = place_at_match(ctx)
    ctx = score_simple(ctx)
    ctx = score_cc_overlap(ctx)
    ctx = score_compound(ctx)
    return ctx


ctx_hu_list  = [_run_pass2(ei, sample_hu_max)      for ei in range(len(pass1_ctxs))]
ctx_log_list = [_run_pass2(ei, sample_neg_log_max) for ei in range(len(pass1_ctxs))]
n_metal     = sum(1 for c in ctx_hu_list if c.bolt_source == "metal")
n_bolt_less = sum(1 for c in ctx_hu_list if c.bolt_source == "bolt_less")
print(f"placed: {len(ctx_hu_list)} emissions  (metal={n_metal}, bolt_less={n_bolt_less})")
"""))

cells.append(md("""
## Post-pass — per-subject FFT normalization

Cross-subject AMC88 vs AMC91 vs T-series have very different absolute FFT power
levels (AMC88 matched p75 ≈ 0.5; AMC91 matched p75 ≈ 0.9). A single global
threshold for `s_fft` over-rewards AMC91 and under-rewards AMC88. Post-pass:
take the subject's p75 of FFT scores **across reliable emissions** (uniform-
pitch, n_contacts ≥ 8) and divide each emission's FFT by it. After this, every
subject's matched cluster lands near `s_fft ≈ 1.0` and orphans below.

Reliability gate keeps CM/BM and short-electrode emissions out of the
reference set — their FFTs are structurally weak and would drag the p75 down.
"""))

cells.append(code("""
def apply_subject_fft_normalization(ctx_list: list[PlacementCtx]) -> list[PlacementCtx]:
    \"\"\"Recompute compound score per emission with FFT normalized to the
    subject's reliable-FFT p75. Returns a new list (ctx is immutable-ish so
    score_compound returns new ctx objects).\"\"\"
    reliable = [c.score_components.get("pitch_power_frac", 0.0) for c in ctx_list
                if c.score_components.get("model_uniform_pitch")
                and int(c.score_components.get("n_slots", 0)) >= 8]
    if not reliable or max(reliable) <= 1e-6:
        return list(ctx_list)
    fft_ref = float(np.percentile(reliable, 75))
    out = []
    for c in ctx_list:
        sc = c.score_components
        if sc.get("model_uniform_pitch") and int(sc.get("n_slots", 0)) >= 8:
            normed = float(sc.get("pitch_power_frac", 0.0)) / max(fft_ref, 1e-6)
        else:
            normed = None  # → score_compound uses neutral 0.5 (fft_reliable=False)
        # Stash for the cmp_df / debugging.
        sc_aug = dict(sc)
        sc_aug["fft_subject_ref_p75"] = fft_ref
        sc_aug["fft_subject_norm"] = normed
        c2 = replace(c, score_components=sc_aug)
        out.append(score_compound(c2, fft_norm=normed))
    return out


ctx_hu_list  = apply_subject_fft_normalization(ctx_hu_list)
ctx_log_list = apply_subject_fft_normalization(ctx_log_list)

# Sanity print.
hu_ref  = ctx_hu_list[0].score_components.get("fft_subject_ref_p75")
log_ref = ctx_log_list[0].score_components.get("fft_subject_ref_p75")
print(f"subject FFT p75 (HU) = {hu_ref:.3f}    (LoG) = {log_ref:.3f}")
"""))

cells.append(md("""
## GT axis match (greedy)
"""))

cells.append(code("""
def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])

pairs = []
for gi, g in enumerate(GT):
    g_s = np.asarray(g["start_ras"]); g_e = np.asarray(g["end_ras"])
    g_axis = _unit(g_e - g_s); g_mid = 0.5 * (g_s + g_e)
    for ei, t in enumerate(v1_trajs):
        e_s = np.asarray(t["start_ras"]); e_e = np.asarray(t["end_ras"])
        e_axis = _unit(e_e - e_s); e_mid = 0.5 * (e_s + e_e)
        ang = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(g_axis, e_axis)))))))
        v = g_mid - e_mid
        perp = v - float(np.dot(v, e_axis)) * e_axis
        mid_d = float(np.linalg.norm(perp))
        if ang <= ANGLE_TOL_DEG and mid_d <= PERP_TOL_MM:
            pairs.append((ang + mid_d, gi, ei))
pairs.sort(key=lambda p: p[0])
used_g, used_e = set(), set()
gt_for_emission: dict[int, str] = {}
for _s, gi, ei in pairs:
    if gi in used_g or ei in used_e:
        continue
    used_g.add(gi); used_e.add(ei)
    gt_for_emission[ei] = GT[gi]["name"]
print(f"GT match: {len(used_g)}/{len(GT)} matched")
"""))

cells.append(md("""
## Comparison table — HU vs LoG side-by-side
"""))

cells.append(code("""
def _r(x, n=3):
    return round(x, n) if x is not None and np.isfinite(x) else None


rows = []
for ei, (cu, cl) in enumerate(zip(ctx_hu_list, ctx_log_list)):
    sh = cu.score_components; sl = cl.score_components
    rows.append({
        "ei":          ei,
        "gt":          gt_for_emission.get(ei, "ORPHAN"),
        "seeder":      cu.seeder_label,
        "bolt_src":    sh["bolt_source"],
        "cc_dist":     _r(sh.get("cc_dist_mm"), 2),
        "bz_frac":     _r(sh.get("bolt_zone_frac")),
        "tube_hu":     _r(sh.get("tube_like_frac")),
        "fft_segs":    f'{sh.get("fft_n_reliable_segments", 0)}/{sh.get("fft_n_segments", 0)}',
        "model_hu":    sh["model_id"],
        "corr_hu":     _r(sh["corr"]),
        "uni_hu":      _r(sh.get("model_corr_uniformity")),
        "margin_hu":   _r(sh.get("model_corr_margin")),
        "ptp_hu":      _r(sh.get("zone_ptp_mod")),
        "fft_hu":      _r(sh["pitch_power_frac"]),
        "compound_hu": _r(sh["compound_score"]),
        "band_hu":     sh["band"],
        "model_log":   sl["model_id"],
        "corr_log":    _r(sl["corr"]),
        "fft_log":     _r(sl["pitch_power_frac"]),
        "tube_log":    _r(sl.get("tube_like_frac")),
        "compound_log":_r(sl["compound_score"]),
        "band_log":    sl["band"],
        "agree?":      sh["model_id"] == sl["model_id"],
        "hu_mean":     int(sh["placed_hu_mean"]) if np.isfinite(sh["placed_hu_mean"]) else None,
    })
df = pd.DataFrame(rows).sort_values(["gt", "compound_hu"], ascending=[True, False])
df
"""))

cells.append(md("""
## Per-emission figures

CT slab + walker profile (HU + −LoG twin axis) + per-model corr bars (HU left,
LoG right). One figure per emission; bolt-less emissions show `[bolt_less]` in
the title and have no bolt_end marker.
"""))

cells.append(code("""
from rosa_core.volume_sampling import sample_trilinear_at_ras


def _score_models(ctx: PlacementCtx) -> list[tuple]:
    \"\"\"Read precomputed per-model corrs from score_simple's stash.\"\"\"
    return list(ctx.score_components.get("per_model_corr") or [])


def _slab_basis(es, ee):
    seed_dir = (ee - es) / max(np.linalg.norm(ee - es), 1e-9)
    world_up = np.array([0.0, 0.0, 1.0])
    v_perp = world_up - (world_up @ seed_dir) * seed_dir
    if np.linalg.norm(v_perp) < 1e-6:
        v_perp = np.array([0.0, 1.0, 0.0])
    v_perp /= np.linalg.norm(v_perp)
    third = np.cross(seed_dir, v_perp); third /= np.linalg.norm(third)
    return seed_dir, v_perp, third


def _build_slab(es, ee, ct_arr, r2i, *, log_arr=None,
                  perp_half=12.0, thick_half=2.0, step=0.5):
    \"\"\"Max-IP slab through the seed plane. When `log_arr` is provided,
    also returns a parallel −LoG slab (positive at metal-bright minima)
    for heatmap overlay.\"\"\"
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
                        nlv = -float(lv)  # negate: positive at metal-bright
                        cur = slab_log[vi, ui]
                        slab_log[vi, ui] = nlv if np.isnan(cur) else max(cur, nlv)
    return slab_hu, slab_log, u_grid, v_grid, seed_dir, v_perp, third, span


def _bar(ax, per_model, picked, title):
    if not per_model:
        ax.set_axis_off(); ax.set_title(title + "  (no data)", fontsize=9); return
    ids   = [t[0] for t in per_model]; corrs = [t[3] for t in per_model]
    cov   = [f"{t[2]}/{t[1]}" for t in per_model]
    colors = ["#cc3333" if i == picked else "#888888" for i in ids]
    x = np.arange(len(ids))
    ax.bar(x, corrs, color=colors, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=7)
    ymax = max(1.0, max(corrs) * 1.15)
    ax.set_ylim(min(0, min(corrs) - 0.05), ymax)
    for xi, (c, nc) in enumerate(zip(corrs, cov)):
        ax.text(xi, c + 0.01, f"{c:.2f}\\n{nc}", ha="center", va="bottom", fontsize=6)
    ax.axhline(0, color="black", lw=0.4)
    ax.grid(True, axis="y", alpha=0.2)
    ax.set_title(title, fontsize=9)


def render_emission(ei, ctx_hu, ctx_log, gt_name, *, features, bolts, GT_lookup):
    es = ctx_hu.seed_start; ee = ctx_hu.seed_end
    r2i = np.asarray(features["ras_to_ijk_mat"], dtype=float)
    ct_arr = features["ct_arr_kji"]

    log_arr = features.get("log")
    slab, slab_log, u_grid, v_grid, seed_dir, v_perp, _third, span = _build_slab(
        es, ee, ct_arr, r2i, log_arr=log_arr)

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.7, 0.9, 0.8], hspace=0.45)
    ax_slab = fig.add_subplot(gs[0])
    ax_prof = fig.add_subplot(gs[1])
    ax_bar  = fig.add_subplot(gs[2])

    extent = [u_grid[0], u_grid[-1], v_grid[0], v_grid[-1]]
    # Layered: −LoG heatmap as the background (calibration-invariant
    # contrast across the whole slab — shows where periodic metal-bright
    # minima exist), then HU grayscale on top with alpha so the saturated
    # metal punches through but tissue lets the LoG colormap breathe.
    if slab_log is not None:
        finite_log = slab_log[np.isfinite(slab_log)]
        if finite_log.size:
            log_lo = float(np.percentile(finite_log, 5))
            log_hi = float(np.percentile(finite_log, 99))
            ax_slab.imshow(slab_log, extent=extent, origin="lower", aspect="auto",
                            cmap="viridis", vmin=log_lo, vmax=log_hi)
    finite = slab[np.isfinite(slab)]
    hu_vmin, hu_vmax = ((float(np.percentile(finite, 1)), float(np.percentile(finite, 99.5)))
                         if finite.size else (-100.0, 2500.0))
    ax_slab.imshow(slab, extent=extent, origin="lower", aspect="auto",
                    cmap="gray", vmin=hu_vmin, vmax=hu_vmax, alpha=0.55)
    ax_slab.axhline(0, color="cyan", lw=0.5, ls="--", alpha=0.5,
                     label="seed (perp=0)")
    ax_slab.axvspan(0, span, color="blue", alpha=0.04)

    def _proj(pts):
        rel = np.asarray(pts, float) - es
        return rel @ seed_dir, rel @ v_perp

    # Refined centerline (post-anchor + refine stage) projected onto slab.
    # For metal anchors this is the LoG-snapped polynomial — its deviation
    # from perp=0 shows how much the snap moved the axis off the straight
    # seed. Bolt-less anchors are 2-point straight seeds → flat at perp=0.
    cl = np.asarray(ctx_hu.centerline, dtype=float)
    if len(cl) >= 2:
        cu, cv = _proj(cl)
        cl_label = f"centerline ({ctx_hu.bolt_source}, {len(cl)} pts)"
        ax_slab.plot(cu, cv, color="#22c55e", lw=1.4, alpha=0.9, zorder=9, label=cl_label)

    if bolts:
        # Only keep bolt-CC voxels actually within the slab's out-of-plane
        # thickness band (±slab_thickness_half + 0.5 mm) — without this gate
        # other shanks' bolts project through the 2D slab and dust the figure.
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
            ax_slab.scatter(cc[:, 0], cc[:, 1], marker=".", color="magenta", s=4, alpha=0.5,
                             label=f"bolt CC ({cc.shape[0]})", zorder=7)

    # All placement markers at their TRUE perp positions on the slab —
    # overlapping markers mean HU and LoG placements agree on this contact.
    if gt_name and gt_name in GT_lookup:
        gtc = GT_lookup[gt_name]
        u, v = _proj(gtc)
        ax_slab.scatter(u, v, marker="x", color="orange", s=70, lw=2,
                         label=f"GT {gt_name} (n={len(gtc)})", zorder=12)
    if ctx_hu.placed_ras:
        u, v = _proj(ctx_hu.placed_ras)
        ax_slab.scatter(u, v, marker="o", color="red", s=30, alpha=0.85,
                         label=f"placed HU ({ctx_hu.score_components['model_id']}, n={len(ctx_hu.placed_ras)})",
                         zorder=11)
    if ctx_log.placed_ras:
        u, v = _proj(ctx_log.placed_ras)
        ax_slab.scatter(u, v, marker="s", color="#1f77b4", s=28, alpha=0.85,
                         edgecolors="white", linewidths=0.4,
                         label=f"placed LoG ({ctx_log.score_components['model_id']}, n={len(ctx_log.placed_ras)})",
                         zorder=11)

    if ctx_hu.bolt_source == "metal":
        ax_slab.axvline(ctx_hu.bolt_end_arc, color="red", lw=1.2,
                         label=f"bolt_end={ctx_hu.bolt_end_arc:.1f}mm")

    ax_slab.set_xlim(u_grid[0], u_grid[-1])
    ax_slab.set_ylim(v_grid[0], v_grid[-1])
    ax_slab.set_ylabel("perp (mm)")
    ax_slab.legend(loc="lower right", fontsize=7, ncol=2)

    # Walker profile — LoG only (HU saturates and adds little; LoG is the
    # primary signal for picking).
    if ctx_log.walk_arcs is not None:
        ax_prof.plot(ctx_log.walk_arcs, ctx_log.walk_signal, color="#d97706", lw=1.2,
                       label="−LoG p90-disk")
        ax_prof.fill_between(ctx_log.walk_arcs, ctx_log.walk_signal,
                              color="#d97706", alpha=0.15)
        ax_prof.set_ylabel("−LoG")
    if ctx_log.bolt_source == "metal":
        ax_prof.axvline(ctx_log.bolt_end_arc, color="red", lw=1.0,
                         label=f"bolt_end={ctx_log.bolt_end_arc:.1f}mm")
    # Project placed contacts onto the centerline arc for tick marks.
    if ctx_log.placed_ras and ctx_log.centerline is not None:
        cl_arr = np.asarray(ctx_log.centerline, dtype=float)
        for p in ctx_log.placed_ras:
            arc = float(_project_to_polyline_arc(cl_arr, np.asarray(p, dtype=float)))
            ax_prof.axvline(arc, color="#1f77b4", alpha=0.35, lw=0.6)
    ax_prof.set_xlabel("arc along centerline (mm)")
    ax_prof.grid(True, alpha=0.2)
    ax_prof.legend(loc="upper right", fontsize=8)

    # Per-model corr bars — LoG only.
    pm_log = _score_models(ctx_log)
    _bar(ax_bar, pm_log, ctx_log.score_components["model_id"],
            f"−LoG per-model corr  →  picked {ctx_log.score_components['model_id']}")

    src = ctx_log.bolt_source
    title = f"emission #{ei}  ↔  {gt_name or 'ORPHAN'}    [{src}]"
    if ctx_hu.score_components["model_id"] != ctx_log.score_components["model_id"]:
        title += f"    HU↔LoG disagree (HU={ctx_hu.score_components['model_id']})"
    fig.suptitle(title, fontsize=10, y=0.99)
    return fig


GT_lookup = {g["name"]: g["contacts_ras"] for g in GT}
for ei, (cu, cl) in enumerate(zip(ctx_hu_list, ctx_log_list)):
    fig = render_emission(ei, cu, cl, gt_for_emission.get(ei),
                          features=features, bolts=bolts, GT_lookup=GT_lookup)
    plt.show()
    plt.close(fig)
"""))

cells.append(md("""
## Score-distribution scatter — matched vs orphan

Three projections of the score surface. Look for clusters that separate the two
classes. Bone-spike / clip-FP cases will sit in different regions than real
shanks; those are the leverage points for new score components.
"""))

cells.append(code("""
matched_df = df[df["gt"] != "ORPHAN"]
orph_df    = df[df["gt"] == "ORPHAN"]
G, R = "#2a9d8f", "#e76f51"

panels = [
    ("corr_hu",   "uni_hu",    "corr × model_corr_uniformity",  None,  None),
    ("corr_hu",   "ptp_hu",    "corr × zone peak-to-peak / mean", None, None),
    ("corr_hu",   "fft_hu",    "corr × pitch FFT power frac",    None,  0.10),
    ("corr_hu",   "bz_frac",   "corr × bolt_zone_frac",          None,  None),
    ("corr_hu",   "hu_mean",   "corr × placed HU mean",          0.35,  1500),
    ("corr_hu",   "corr_log",  "HU corr × LoG corr",             None,  None),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (xk, yk, title, vline, hline) in zip(axes.flat, panels):
    ax.scatter(matched_df[xk], matched_df[yk], color=G, label=f"matched (n={len(matched_df)})", s=40)
    ax.scatter(orph_df[xk],    orph_df[yk],    color=R, label=f"orphan (n={len(orph_df)})",   s=40, marker="x")
    ax.set_xlabel(xk); ax.set_ylabel(yk); ax.set_title(title, fontsize=10)
    if vline is not None:
        ax.axvline(vline, color="gray", lw=0.6, ls="--")
    if hline is not None:
        ax.axhline(hline, color="gray", lw=0.6, ls=":")
    if (xk, yk) == ("corr_hu", "corr_log"):
        ax.plot([-0.1, 1.0], [-0.1, 1.0], color="gray", lw=0.5, ls="--")
    ax.grid(alpha=0.2); ax.legend(fontsize=8)
fig.suptitle(f"{SUBJECT_ID}: score-surface separation (matched vs orphan)", y=1.00)
fig.tight_layout()
plt.show()
"""))

cells.append(md("""
## Observations (fill in by hand)

- Bolt-less orphans: which score components separate them from real shanks?
- HU vs LoG disagreements: when does the LoG signal change the picked model?
- High `bolt_zone_frac` cases: are they all orphans, or do real shanks ever go above 0.5?
- Cases where a real shank's `corr_hu` < 0.35 — what makes them weak?
- Stages we should swap next? (refine: try HU snap, no-snap; sample: try
  integrated HU, σ-bank LoG; pick: try peak-pick + library)
"""))


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path("/Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper/notebooks/v1_seeds_v2_placement_qc.ipynb")
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({out.stat().st_size} bytes, {len(cells)} cells)")
