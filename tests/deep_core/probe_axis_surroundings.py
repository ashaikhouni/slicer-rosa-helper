"""Tube-surroundings probe — is the shank perpendicular tube embedded
in intracranial space + low-LoG (homogeneous) tissue?

Question
--------
Concept-swap #5 retires DEEP_TIP_MIN_MM and friends. The intent: a
real electrode is mostly *surrounded* by brain parenchyma. Two
problems with the absolute-HU framing of that intent:

  1. Between contacts ON-axis the trajectory passes through the
     insulated wire (~200-800 HU partial-volume), not parenchyma —
     so absolute-HU axis samples don't read as parenchyma even on
     real shanks.
  2. The CT HU encoding is scanner-variant (cf. ``HU_CLIP_MAX = 3000``
     introduced for metal saturation; brain HU has the same problem
     in the other direction). T4's "deep brain" voxels read at
     ~400-500 HU, not 20-50.

Two scanner-invariant tube tests instead:

  - **Intracranial fraction**: sample a perpendicular ring around the
    axis against the existing hull head-distance field. Real shank:
    deep-half tube ~ 100% inside intracranial space. Bone-skim FP:
    tube hangs off the skull surface.
  - **LoG homogeneity**: same ring against the LoG-σ=1 array. Brain
    parenchyma is locally homogeneous so |LoG| is near zero outside
    contact voxels. Wire-through-bone or texture-aliasing FPs sit in
    a band where |LoG| is non-trivial.

This probe runs the pipeline (with ``return_features=True``), then
samples the perpendicular tube against ``head_distance`` and
``log_sigma1``. Joint distribution of matched-vs-orphan written to
/tmp/probe_axis_surroundings.tsv.

Run
---
    cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper && \\
        /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_axis_surroundings.py
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

ROOT = Path("/Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper")
sys.path.insert(0, str(ROOT / "CommonLib"))
sys.path.insert(0, str(ROOT / "PostopCTLocalization"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import SimpleITK as sitk

from postop_ct_localization import contact_pitch_v1_fit as cpfit
from eval_seeg_localization import (
    iter_subject_rows,
    load_reference_ground_truth_shanks,
)
from shank_core.io import image_ijk_ras_matrices

DATASET_ROOT = Path(
    os.environ.get(
        "ROSA_SEEG_DATASET",
        "/Users/ammar/Dropbox/thalamus_subjects/seeg_localization",
    )
)
EXCLUDE = {"T17", "T19", "T21"}

MATCH_ANGLE_DEG = 10.0
MATCH_MID_MM = 8.0

# Tube geometry. Inner radius covers the SEEG contact (~0.4 mm)
# + insulation/wire (~0.5 mm) + a partial-volume halo (~1 mm) so
# bright contact voxels don't bleed into the tube samples. Outer
# radius is the maximum distance we still treat as "shank
# surroundings" before we're sampling unrelated tissue.
TUBE_INNER_R_MM = 2.0
TUBE_OUTER_R_MM = 4.0

N_PERP_SAMPLES = 8           # ring positions per axial step
AXIAL_STEP_MM = 1.0
INLIER_HALO_MM = 1.5         # axial halo around each inlier where
                              # contact metal still scatters into the
                              # tube; skip those axial slabs

OUT_TSV = Path("/tmp/probe_axis_surroundings.tsv")


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def _ortho_basis(axis):
    """Two unit vectors perpendicular to ``axis`` and to each other."""
    ref = np.array([1.0, 0.0, 0.0])
    if abs(axis[0]) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, ref)
    u /= max(np.linalg.norm(u), 1e-9)
    v = np.cross(axis, u)
    return u, v


def _sample_tube(start_ras, end_ras, ras_to_ijk_mat,
                  arrays,
                  inlier_ras=None,
                  inner_r=TUBE_INNER_R_MM,
                  outer_r=TUBE_OUTER_R_MM,
                  n_perp=N_PERP_SAMPLES,
                  axial_step=AXIAL_STEP_MM,
                  inlier_halo=INLIER_HALO_MM,
                  start_pad_mm=2.0,
                  end_pad_mm=2.0):
    """Sample each (named) volumetric array inside a perpendicular
    tube around the axis, skipping inlier-halo axial slabs.

    ``arrays`` is a dict[name -> kji-order numpy array], all same shape.
    Returns dict[name -> 1-D numpy array of samples].
    """
    s = np.asarray(start_ras, dtype=float)
    e = np.asarray(end_ras, dtype=float)
    L = float(np.linalg.norm(e - s))
    out = {k: [] for k in arrays}
    if L < 5.0:
        return {k: np.array([]) for k in arrays}
    axis = (e - s) / L
    u, v = _ortho_basis(axis)

    inlier_along = np.array([])
    if inlier_ras is not None and len(inlier_ras) > 0:
        inl = np.asarray(inlier_ras, dtype=float)
        inlier_along = (inl - s) @ axis

    radii = np.linspace(inner_r, outer_r, 3)
    thetas = np.arange(n_perp) * (2.0 * np.pi / n_perp)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    any_arr = next(iter(arrays.values()))
    nz, ny, nx = any_arr.shape
    d = start_pad_mm
    while d < L - end_pad_mm:
        if inlier_along.size and float(np.min(np.abs(inlier_along - d))) < inlier_halo:
            d += axial_step
            continue
        center = s + d * axis
        for r in radii:
            for ct, st in zip(cos_t, sin_t):
                p_ras = center + r * (ct * u + st * v)
                ras_h = np.array([p_ras[0], p_ras[1], p_ras[2], 1.0])
                ijk = ras_to_ijk_mat @ ras_h
                i = int(round(float(ijk[0])))
                j = int(round(float(ijk[1])))
                k = int(round(float(ijk[2])))
                if 0 <= k < nz and 0 <= j < ny and 0 <= i < nx:
                    for name, arr in arrays.items():
                        out[name].append(float(arr[k, j, i]))
        d += axial_step
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def _log_summary(log_samples):
    """LoG-σ=1 magnitude summary in the perpendicular tube.

    Real shank in brain parenchyma: tissue is locally homogeneous so
    |LoG| sits low in the tube (most voxels < 100; tail driven by
    small CSF / vessel structures). Bone-skim or wire-class FP:
    tissue around the axis has texture, so |LoG| sits higher.

    LoG is scanner-invariant (the upstream pipeline applies the
    HU clip at HU_CLIP_MAX = 3000, so the LoG response shape is
    consistent across subjects). Returns a dict of percentiles +
    fractions above sample thresholds.
    """
    if log_samples.size == 0:
        nan = float("nan")
        return {
            "abs_p50": nan, "abs_p75": nan, "abs_p90": nan,
            "frac_le_50": nan, "frac_le_100": nan, "frac_le_200": nan,
        }
    a = np.abs(log_samples)
    return {
        "abs_p50": float(np.percentile(a, 50)),
        "abs_p75": float(np.percentile(a, 75)),
        "abs_p90": float(np.percentile(a, 90)),
        "frac_le_50":  float((a <= 50).sum())  / float(a.size),
        "frac_le_100": float((a <= 100).sum()) / float(a.size),
        "frac_le_200": float((a <= 200).sum()) / float(a.size),
    }


def _greedy_match(gt_shanks, trajs):
    pairs = []
    for gi, g in enumerate(gt_shanks):
        g_s = np.asarray(g.start_ras, dtype=float)
        g_e = np.asarray(g.end_ras, dtype=float)
        g_axis = _unit(g_e - g_s)
        g_mid = 0.5 * (g_s + g_e)
        for ti, t in enumerate(trajs):
            t_s = np.asarray(t["start_ras"], dtype=float)
            t_e = np.asarray(t["end_ras"], dtype=float)
            t_axis = _unit(t_e - t_s)
            ang = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(g_axis, t_axis)))))))
            t_mid = 0.5 * (t_s + t_e)
            d = g_mid - t_mid
            mid_d = float(np.linalg.norm(d - (d @ t_axis) * t_axis))
            if ang <= MATCH_ANGLE_DEG and mid_d <= MATCH_MID_MM:
                pairs.append((ang + mid_d, gi, ti))
    pairs.sort(key=lambda p: p[0])
    used_g, used_t = set(), set()
    matched_ti = {}
    for _s, gi, ti in pairs:
        if gi in used_g or ti in used_t:
            continue
        used_g.add(gi)
        used_t.add(ti)
        matched_ti[ti] = str(gt_shanks[gi].shank)
    return matched_ti


def main():
    rows = iter_subject_rows(DATASET_ROOT, None)
    rows = [r for r in rows if str(r["subject_id"]) not in EXCLUDE]
    rows.sort(key=lambda r: str(r["subject_id"]))

    fieldnames = [
        "subject_id", "label", "matched_shank", "trajectory_idx",
        "source", "bolt_source",
        "n_inliers", "length_mm",
        "n_samples",
        "log_abs_p50", "log_abs_p75", "log_abs_p90",
        "log_frac_le_50", "log_frac_le_100", "log_frac_le_200",
        "deep_log_abs_p50", "deep_log_abs_p75", "deep_log_abs_p90",
        "deep_log_frac_le_50", "deep_log_frac_le_100", "deep_log_frac_le_200",
        "confidence", "confidence_label",
    ]
    out_rows = []
    print(f"running pipeline + tube probe on {len(rows)} subjects")
    for row in rows:
        subject_id = str(row["subject_id"])
        gt, _ = load_reference_ground_truth_shanks(row)
        img = sitk.ReadImage(row["ct_path"])
        ijk_to_ras, ras_to_ijk = image_ijk_ras_matrices(img)
        trajs, features = cpfit.run_two_stage_detection(
            img, ijk_to_ras, ras_to_ijk,
            return_features=True,
            pitch_strategy="auto",
        )
        arrays = {"log": np.asarray(features["log_sigma1"])}
        matched = _greedy_match(gt, trajs)
        n_match = len(matched)
        n_orph = len(trajs) - n_match
        print(f"  {subject_id}: {n_match}/{len(gt)} matched, "
              f"{n_orph} orphans, {len(trajs)} emitted")
        for ti, t in enumerate(trajs):
            inlier_ras = t.get("inlier_ras")
            full = _sample_tube(
                t["start_ras"], t["end_ras"], ras_to_ijk, arrays,
                inlier_ras=inlier_ras,
            )
            mid_ras = 0.5 * (np.asarray(t["start_ras"]) + np.asarray(t["end_ras"]))
            deep_half = _sample_tube(
                mid_ras, np.asarray(t["end_ras"]), ras_to_ijk, arrays,
                inlier_ras=inlier_ras,
                start_pad_mm=0.0,
            )
            full_log = _log_summary(full["log"])
            deep_log = _log_summary(deep_half["log"])
            label = "matched" if ti in matched else "orphan"
            row_out = {
                "subject_id": subject_id,
                "label": label,
                "matched_shank": matched.get(ti, ""),
                "trajectory_idx": ti,
                "source": str(t.get("source") or ""),
                "bolt_source": str(t.get("bolt_source") or ""),
                "n_inliers": int(t.get("n_inliers") or 0),
                "length_mm": float(t.get("length_mm") or 0.0),
                "n_samples": int(full["log"].size),
                "confidence": float(t.get("confidence") or float("nan")),
                "confidence_label": str(t.get("confidence_label") or ""),
            }
            for k, v in full_log.items():
                row_out[f"log_{k}"] = v
            for k, v in deep_log.items():
                row_out[f"deep_log_{k}"] = v
            out_rows.append(row_out)

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {len(out_rows)} rows to {OUT_TSV}")

    matched_rows = [r for r in out_rows if r["label"] == "matched"]
    orphan_rows = [r for r in out_rows if r["label"] == "orphan"]
    print(f"\n=== n: matched={len(matched_rows)}  orphan={len(orphan_rows)} ===")

    def _pct(rows, key):
        a = np.array([float(r[key]) for r in rows], dtype=float)
        a = a[~np.isnan(a)]
        if a.size == 0:
            return "nan"
        return (
            f"p10/p50/p90 = {np.percentile(a, 10):.2f} / "
            f"{np.percentile(a, 50):.2f} / {np.percentile(a, 90):.2f}"
        )

    for key in ("log_abs_p50", "log_abs_p75", "log_abs_p90",
                "log_frac_le_50", "log_frac_le_100", "log_frac_le_200",
                "deep_log_abs_p50", "deep_log_abs_p75", "deep_log_abs_p90",
                "deep_log_frac_le_50", "deep_log_frac_le_100", "deep_log_frac_le_200"):
        print(f"  {key:>24s} matched: {_pct(matched_rows, key)}")
        print(f"  {key:>24s}  orphan: {_pct(orphan_rows, key)}")

    # Rank orphans by deep_log_abs_p50 — high values mean textured
    # surroundings (bone, wire, vasculature), low values mean
    # homogeneous brain.
    print("\n=== orphans ranked by deep_log_abs_p50 (descending — bad first) ===")
    rows_sorted = sorted(
        orphan_rows,
        key=lambda r: -float(r["deep_log_abs_p50"]) if r["deep_log_abs_p50"] == r["deep_log_abs_p50"] else 0.0,
    )
    for r in rows_sorted:
        print(
            f"  {r['subject_id']:>4s} t{int(r['trajectory_idx']):>2d}: "
            f"deep |LoG| p50/p75/p90 = "
            f"{float(r['deep_log_abs_p50']):>5.0f}/"
            f"{float(r['deep_log_abs_p75']):>5.0f}/"
            f"{float(r['deep_log_abs_p90']):>5.0f}  "
            f"frac_le_100={float(r['deep_log_frac_le_100']):.2f}  "
            f"conf={float(r['confidence']):.2f}({r['confidence_label']:>6s}) "
            f"src={r['source']:>6s} bolt={r['bolt_source']:>10s} "
            f"n={int(r['n_inliers']):>2d}"
        )

    # Real shanks with HIGH deep |LoG| — sanity check: if matched lines
    # share textured surroundings, a high-|LoG| gate would falsely
    # demote them.
    print("\n=== matched top-15 by deep_log_abs_p50 (worst homogeneity) ===")
    matched_sorted = sorted(
        matched_rows,
        key=lambda r: -float(r["deep_log_abs_p50"]) if r["deep_log_abs_p50"] == r["deep_log_abs_p50"] else 0.0,
    )
    for r in matched_sorted[:15]:
        print(
            f"  {r['subject_id']:>4s} {r['matched_shank']:>8s}: "
            f"deep |LoG| p50/p75/p90 = "
            f"{float(r['deep_log_abs_p50']):>5.0f}/"
            f"{float(r['deep_log_abs_p75']):>5.0f}/"
            f"{float(r['deep_log_abs_p90']):>5.0f}  "
            f"frac_le_100={float(r['deep_log_frac_le_100']):.2f}  "
            f"conf={float(r['confidence']):.2f}({r['confidence_label']:>6s})"
        )


if __name__ == "__main__":
    main()
