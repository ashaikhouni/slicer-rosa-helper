"""Metal-evidence calibration probe (Concept Swap #2).

Test the proposed unified bolt-anchor pipeline by replacing the 3-tier
cascade (LoG@800 -> HU@2000 -> axis-synth) with ONE CC extraction on
a unified per-voxel metal-evidence volume:

    metal_evidence(voxel) = max(|LoG(v)| / 800, max(0, HU(v)) / 2000)

The unified function then does:
  1. CC extraction on metal_evidence >= T_BOLT (e.g. T_BOLT = 1.0).
  2. Hull-touch filter (existing `extract_bolt_candidates` logic).
  3. Per-line tube anchor with unified tube radius (e.g. 5 mm).
  4. Synth fallback if no CC anchors.

This probe simulates step 1-3 against EVERY emitted trajectory and
reports anchor success / fail per tier, with focus on the 4
must-preserve shanks (T4 RSFG hu_rescue + T3 LAI / T4 LCMN / T4 LSFG
axis_synth). The simulation uses the trajectory's walker INLIERS as
the proxy for the pre-anchor walker line (shallowest_inlier -> mid
of inliers -> deepest_inlier; PCA axis from inliers).

Usage:
    cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper && \\
        /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_metal_evidence_calibration.py [SUBJECT|ALL]
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper")
sys.path.insert(0, str(ROOT / "CommonLib"))
sys.path.insert(0, str(ROOT / "PostopCTLocalization"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np

from postop_ct_localization import contact_pitch_v1_fit as cpfit
from shank_engine import PipelineRegistry, register_builtin_pipelines
from eval_seeg_localization import (
    build_detection_context, iter_subject_rows,
    load_reference_ground_truth_shanks,
)

DATASET_ROOT = Path("/Users/ammar/Dropbox/thalamus_subjects/seeg_localization")
EXCLUDE = {"T17", "T19", "T21"}

LOG_NORM = 800.0
HU_NORM = 2000.0

# Candidate unified-anchor thresholds. T_BOLT = 1.0 means
# |LoG| >= 800 OR HU >= 2000 — exactly today's two cascade thresholds
# expressed as a single normalized cutoff. Below 1.0 admits weaker
# bolts (HU just under 2000 or LoG just under 800). Above 1.0 demands
# both modalities saturate together.
T_BOLT_SWEEP = (0.5, 0.7, 0.85, 1.0, 1.25, 1.5)

# Unified hull-proximity tolerance (mm). Today's cascade splits this
# 2.0 (LoG) vs 5.0 (HU rescue). The unified pass needs ONE value that
# accepts thin-wire-PMT bolts (which sit 2-5 mm inside hull) without
# eating skull-band noise. 5 mm matches today's HU-rescue value.
HULL_PROX_MM = 5.0

# Unified tube radius (mm). Today's cascade splits 3.0 (LoG) vs 5.0
# (HU rescue). Walker-axis drift up to 3 mm over a 50 mm bolt walk
# argues for the wider value. 5 mm matches today's HU-rescue value.
TUBE_RADIUS_MM = 5.0

# How far outward from the walker shallow endpoint to search for the
# bolt CC along the axis. Same as today's BOLT_SEARCH_OUTWARD_MM.
SEARCH_OUTWARD_MM = 60.0

MUST_PRESERVE = {
    ("T4", "RSFG"),
    ("T3", "LAI"),
    ("T4", "LCMN"),
    ("T4", "LSFG"),
}


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def _greedy_match(gt_shanks, trajs):
    pairs = []
    for gi, g in enumerate(gt_shanks):
        gs = np.asarray(g.start_ras, dtype=float)
        ge = np.asarray(g.end_ras, dtype=float)
        ga = _unit(ge - gs)
        gm = 0.5 * (gs + ge)
        for ti, t in enumerate(trajs):
            ts = np.asarray(t["start_ras"], dtype=float)
            te = np.asarray(t["end_ras"], dtype=float)
            ta = _unit(te - ts)
            tm = 0.5 * (ts + te)
            ang = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(ga, ta)))))))
            d = gm - tm
            mid = float(np.linalg.norm(d - (d @ ta) * ta))
            if ang <= 10 and mid <= 8:
                pairs.append((ang + mid, gi, ti))
    pairs.sort()
    used_g, used_t = set(), set()
    m = {}
    for _s, gi, ti in pairs:
        if gi in used_g or ti in used_t:
            continue
        used_g.add(gi)
        used_t.add(ti)
        m[ti] = str(gt_shanks[gi].shank)
    return m


def _hook_volumes():
    """Capture log_arr / ct_arr / dist_arr / ijk_to_ras_mat / spacing
    via monkey-patching ``extract_bolt_candidates`` (called twice — once
    with ct_arr=None for LoG, once with ct_arr for HU rescue).
    """
    state = {}
    real = cpfit.extract_bolt_candidates

    def patched(log_arr, dist_arr, ijk_to_ras_mat, spacing_xyz, **kw):
        state["log_arr"] = log_arr
        state["dist_arr"] = dist_arr
        state["ijk_to_ras_mat"] = np.asarray(ijk_to_ras_mat, dtype=float)
        state["spacing"] = tuple(float(s) for s in spacing_xyz)
        if "ct_arr" in kw and kw["ct_arr"] is not None:
            state["ct_arr"] = kw["ct_arr"]
        return real(log_arr, dist_arr, ijk_to_ras_mat, spacing_xyz, **kw)

    cpfit.extract_bolt_candidates = patched

    def restore():
        cpfit.extract_bolt_candidates = real

    return state, restore


def _build_metal_evidence_volume(log_arr, ct_arr):
    """metal_evidence(v) = max(|LoG(v)|/LOG_NORM, max(0,HU(v))/HU_NORM)

    Returned as float32 to keep the CC threshold cheap.
    """
    log_norm = np.abs(log_arr) / float(LOG_NORM)
    if ct_arr is None:
        return log_norm.astype(np.float32, copy=False)
    hu_norm = np.maximum(0.0, ct_arr) / float(HU_NORM)
    return np.maximum(log_norm, hu_norm).astype(np.float32, copy=False)


def _walker_proxy(traj, dist_arr, ras_to_ijk_mat):
    """Approximate walker pre-anchor (start, end) from the trajectory's
    inliers. Returns (walker_start, walker_end) where walker_start is
    the shallowest inlier and walker_end is the deepest inlier.

    Falls back to (start_ras, end_ras) when inliers are absent.
    """
    inliers = traj.get("inlier_ras")
    s = np.asarray(traj["start_ras"], dtype=float)
    e = np.asarray(traj["end_ras"], dtype=float)
    if inliers is None or len(inliers) == 0:
        return s, e
    pts = np.asarray(inliers, dtype=float)
    K, J, I = dist_arr.shape
    h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    ijk = (ras_to_ijk_mat @ h.T).T[:, :3]
    ic = np.clip(np.round(ijk[:, 0]).astype(int), 0, I - 1)
    jc = np.clip(np.round(ijk[:, 1]).astype(int), 0, J - 1)
    kc = np.clip(np.round(ijk[:, 2]).astype(int), 0, K - 1)
    d = dist_arr[kc, jc, ic].astype(np.float32)
    shallow_idx = int(np.argmin(d))
    deep_idx = int(np.argmax(d))
    walker_start = pts[shallow_idx]
    walker_end = pts[deep_idx]
    return walker_start, walker_end


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "ALL"
    subjects_filter = None if arg == "ALL" else {arg}
    rows = iter_subject_rows(DATASET_ROOT, subjects_filter)
    rows = [r for r in rows if str(r["subject_id"]) not in EXCLUDE]
    rows.sort(key=lambda r: int(str(r["subject_id"]).lstrip("T")))
    if not rows:
        print(f"ERROR: no subjects found for filter {subjects_filter}")
        return 1

    state, restore = _hook_volumes()
    registry = PipelineRegistry()
    register_builtin_pipelines(registry)

    # collected_records[T] is a list of records (one per emitted trajectory)
    collected = {T: [] for T in T_BOLT_SWEEP}
    # Trajectory metadata (subject, ti, name, matched, current bolt_source).
    traj_meta = []

    try:
        for row in rows:
            sid = str(row["subject_id"])
            gt, _ = load_reference_ground_truth_shanks(row)
            ctx, _img = build_detection_context(
                row["ct_path"], run_id=f"probe_metalcal_{sid}",
                config={}, extras={},
            )
            ctx["contact_pitch_v1_pitch_strategy"] = "auto"
            result = registry.run("contact_pitch_v1", ctx)
            trajs = list(result.get("trajectories") or [])
            matched = _greedy_match(gt, trajs)

            log_arr = state.get("log_arr")
            ct_arr = state.get("ct_arr")
            dist_arr = state.get("dist_arr")
            ijk_to_ras = state.get("ijk_to_ras_mat")
            spacing = state.get("spacing")
            if log_arr is None or ijk_to_ras is None:
                print(f"  {sid}: WARNING — no captured volumes, skipping")
                continue
            ras_to_ijk = np.linalg.inv(ijk_to_ras)
            metal_vol = _build_metal_evidence_volume(log_arr, ct_arr)

            # Per-threshold bolt CC extraction. Re-use the existing
            # `extract_bolt_candidates` machinery in HU-mode by passing
            # metal_vol as `ct_arr` and threshold T as `hu_threshold`.
            bolts_by_T = {}
            for T in T_BOLT_SWEEP:
                bolts, _ = cpfit.extract_bolt_candidates(
                    log_arr, dist_arr, ijk_to_ras, spacing,
                    ct_arr=metal_vol, hu_threshold=T,
                    hull_proximity_mm=HULL_PROX_MM,
                )
                bolts_by_T[T] = bolts

            for ti, t in enumerate(trajs):
                traj_meta.append({
                    "subject": sid,
                    "ti": ti,
                    "name": matched.get(ti, ""),
                    "matched": ti in matched,
                    "bolt_source": str(t.get("bolt_source", "?")),
                    "n_inliers": int(t.get("n_inliers", 0)),
                    "conf": float(t.get("confidence", 0.0)),
                    "conf_label": str(t.get("confidence_label", "?")),
                })
                walker_s, walker_e = _walker_proxy(t, dist_arr, ras_to_ijk)
                # Try anchoring forward + backward; pick orientation
                # with stronger anchor (matches existing _try_anchor).
                for T in T_BOLT_SWEEP:
                    bolts = bolts_by_T[T]
                    fwd = cpfit.anchor_trajectory_to_bolt(
                        walker_s, walker_e, bolts,
                        tube_radius_mm=TUBE_RADIUS_MM,
                        search_outward_mm=SEARCH_OUTWARD_MM,
                    )
                    bwd = cpfit.anchor_trajectory_to_bolt(
                        walker_e, walker_s, bolts,
                        tube_radius_mm=TUBE_RADIUS_MM,
                        search_outward_mm=SEARCH_OUTWARD_MM,
                    )
                    fwd_n = int(fwd[2].get("tube_n_vox", 0)) if fwd[2] else 0
                    bwd_n = int(bwd[2].get("tube_n_vox", 0)) if bwd[2] else 0
                    if max(fwd_n, bwd_n) > 0:
                        bolt = fwd[2] if fwd_n >= bwd_n else bwd[2]
                        anchor_status = "anchored"
                        n_vox = max(fwd_n, bwd_n)
                        bolt_dist_min = float(bolt.get("dist_min_mm", float("nan")))
                        bolt_n_total = int(bolt.get("n_vox", 0))
                    else:
                        anchor_status = "fall_through"
                        n_vox = 0
                        bolt_dist_min = float("nan")
                        bolt_n_total = 0
                    collected[T].append({
                        "anchor_status": anchor_status,
                        "tube_n_vox": n_vox,
                        "bolt_dist_min": bolt_dist_min,
                        "bolt_n_total": bolt_n_total,
                    })
            print(f"  {sid:>4s}: {len(matched)}/{len(gt)} matched, "
                  f"{len(trajs) - len(matched)} orphans, "
                  f"|bolts|@T=1.0={len(bolts_by_T[1.0])}")
    finally:
        restore()

    # --- per-must-preserve detail -----------------------------------
    print("\n=== MUST-PRESERVE shanks (4) — unified anchor outcome per T ===")
    print("axis_synth shanks SHOULD fall through (bolt outside CT FOV).")
    print("hu_rescue / log shanks SHOULD anchor.\n")
    print(f"{'subj':>4s} {'name':>5s}  {'today':>11s}  ", end="")
    for T in T_BOLT_SWEEP:
        print(f"  T={T:.2f}", end="")
    print()
    for idx, meta in enumerate(traj_meta):
        key = (meta["subject"], meta["name"]) if meta["name"] else None
        if key not in MUST_PRESERVE:
            continue
        print(f"{meta['subject']:>4s} {meta['name']:>5s}  "
              f"{meta['bolt_source']:>11s}  ", end="")
        for T in T_BOLT_SWEEP:
            r = collected[T][idx]
            tag = "ANCH" if r["anchor_status"] == "anchored" else "fall"
            print(f"  {tag:>6s}", end="")
        print()
        # Per-threshold detail
        for T in T_BOLT_SWEEP:
            r = collected[T][idx]
            if r["anchor_status"] == "anchored":
                print(f"        T={T:.2f}: tube_n_vox={r['tube_n_vox']:>4d}  "
                      f"bolt_n_total={r['bolt_n_total']:>5d}  "
                      f"dist_min={r['bolt_dist_min']:+.1f} mm")
            else:
                print(f"        T={T:.2f}: fall through")

    # --- summary -----------------------------------------------------
    print(f"\n=== summary: matched / orphan / must-preserve anchor counts ===")
    matched_indices = [i for i, m in enumerate(traj_meta) if m["matched"]]
    orphan_indices = [i for i, m in enumerate(traj_meta) if not m["matched"]]
    mp_indices = [
        i for i, m in enumerate(traj_meta)
        if (m["subject"], m["name"]) in MUST_PRESERVE
    ]
    print(f"\ntotals: {len(matched_indices)} matched, "
          f"{len(orphan_indices)} orphans, {len(mp_indices)} must-preserve")
    print(f"\n{'thresh':>6s}  {'matched anch':>14s}  "
          f"{'orphan anch':>14s}  {'must-pres anch':>15s}")
    for T in T_BOLT_SWEEP:
        m_anch = sum(
            1 for i in matched_indices
            if collected[T][i]["anchor_status"] == "anchored"
        )
        o_anch = sum(
            1 for i in orphan_indices
            if collected[T][i]["anchor_status"] == "anchored"
        )
        mp_anch = sum(
            1 for i in mp_indices
            if collected[T][i]["anchor_status"] == "anchored"
        )
        print(
            f"{T:>6.2f}  {m_anch:>4d}/{len(matched_indices):<4d} "
            f"({100*m_anch/max(1,len(matched_indices)):>4.1f}%)  "
            f"{o_anch:>4d}/{len(orphan_indices):<4d} "
            f"({100*o_anch/max(1,len(orphan_indices)):>4.1f}%)  "
            f"{mp_anch:>3d}/{len(mp_indices):<3d}"
        )

    # --- breakdown per current bolt_source ---------------------------
    # For each current tier, how many shanks anchor under unified at each T?
    sources = sorted({m["bolt_source"] for m in traj_meta})
    print(f"\n=== anchor count by today's bolt_source (matched shanks only) ===")
    print(f"\n{'T':>6s}", end="")
    for src in sources:
        n_total = sum(
            1 for i in matched_indices if traj_meta[i]["bolt_source"] == src
        )
        print(f"  {src:>11s} (/{n_total:>3d})", end="")
    print()
    for T in T_BOLT_SWEEP:
        print(f"{T:>6.2f}", end="")
        for src in sources:
            n_anch = sum(
                1 for i in matched_indices
                if traj_meta[i]["bolt_source"] == src
                and collected[T][i]["anchor_status"] == "anchored"
            )
            n_total = sum(
                1 for i in matched_indices
                if traj_meta[i]["bolt_source"] == src
            )
            print(f"        {n_anch:>3d}/{n_total:<3d}     ", end="")
        print()

    # --- matched shanks that DON'T anchor at chosen T (1.0) ---------
    print(f"\n=== matched shanks that DO NOT anchor at T=1.0 ===")
    print(f"(These need synth fallback under unified pass.)\n")
    for src in ("log", "hu_rescue"):
        miss = [
            i for i in matched_indices
            if traj_meta[i]["bolt_source"] == src
            and collected[1.0][i]["anchor_status"] != "anchored"
        ]
        if not miss:
            print(f"  bolt_source={src}: none missed at T=1.0")
            continue
        print(f"  bolt_source={src}: {len(miss)} missed:")
        print(f"    {'subj':>4s} {'name':>5s} "
              f"{'T=0.5':>5s} {'T=0.7':>5s} {'T=0.85':>6s} {'T=1.0':>5s}")
        for i in miss:
            m = traj_meta[i]
            tags = [
                "anch" if collected[T][i]["anchor_status"] == "anchored"
                else "fall"
                for T in (0.5, 0.7, 0.85, 1.0)
            ]
            print(f"    {m['subject']:>4s} {m['name']:>5s} "
                  f"{tags[0]:>5s} {tags[1]:>5s} {tags[2]:>6s} {tags[3]:>5s}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
