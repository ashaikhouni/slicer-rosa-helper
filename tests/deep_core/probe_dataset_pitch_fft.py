"""Pitch-FFT invariance probe.

For each candidate trajectory, sample LoG values along the trajectory
axis at fine spacing (0.5 mm), FFT, find the dominant spatial
frequency in the physical-pitch band [2.5, 9 mm], and compare
against the trajectory's emitted pitch.

Hypothesis: rotation + scale-space invariance can't distinguish
"real shank walked every-other-contact" (k=2 walker miss) from a
truly real shank — both are physical periodic structure. Pitch-FFT
catches it: the FFT fundamental sits at the TRUE contact pitch
(3.4 mm), not at the walker's claimed pitch (6.8 mm).

Library pitches: (3.5, 3.9, 3.97, 4.43, 4.8, 6.1) mm. So 6.7-6.9 mm
emissions are NOT library pitches and are immediately suspicious.

Per-trajectory output:
  - emitted_pitch  (walker's claim, from `original_median_pitch_mm`)
  - dominant_pitch (FFT peak in the physical-pitch band)
  - ratio          = dominant / emitted
      ratio ≈ 1.0 → walker correct
      ratio ≈ 0.5 → walker missed alternate contacts (k=2 case)
  - peak_power_norm = dominant peak power / median spectrum power

Run:
    cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper && \\
        /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_dataset_pitch_fft.py
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path("/Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper")
sys.path.insert(0, str(ROOT / "CommonLib"))
sys.path.insert(0, str(ROOT / "PostopCTLocalization"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import map_coordinates

from postop_ct_localization import contact_pitch_v1_fit as cpfit
from shank_engine import PipelineRegistry, register_builtin_pipelines
from shank_core.io import image_ijk_ras_matrices
from eval_seeg_localization import (
    build_detection_context,
    iter_subject_rows,
    load_reference_ground_truth_shanks,
)


DATASET_ROOT = Path("/Users/ammar/Dropbox/thalamus_subjects/seeg_localization")
EXCLUDE = {"T17", "T19", "T21"}

MATCH_ANGLE_DEG = 10.0
MATCH_MID_MM = 8.0

SAMPLE_STEP_MM = 0.5    # well below the smallest library pitch (3.5 mm)
PITCH_RANGE_MM = (2.0, 13.0)  # covers libraries 3.5-6.1 + k=2 multiples + small-pitch tail


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def _greedy_match(gt_shanks, trajs):
    pairs = []
    for gi, g in enumerate(gt_shanks):
        gs, ge = np.asarray(g.start_ras, dtype=float), np.asarray(g.end_ras, dtype=float)
        ga = _unit(ge - gs); gm = 0.5 * (gs + ge)
        for ti, t in enumerate(trajs):
            ts, te = np.asarray(t["start_ras"], dtype=float), np.asarray(t["end_ras"], dtype=float)
            ta = _unit(te - ts); tm = 0.5 * (ts + te)
            ang = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(ga, ta)))))))
            d = gm - tm; mid = float(np.linalg.norm(d - (d @ ta) * ta))
            if ang <= MATCH_ANGLE_DEG and mid <= MATCH_MID_MM:
                pairs.append((ang + mid, gi, ti))
    pairs.sort(); used_g, used_t = set(), set(); m = {}
    for _s, gi, ti in pairs:
        if gi in used_g or ti in used_t: continue
        used_g.add(gi); used_t.add(ti); m[ti] = str(gt_shanks[gi].shank)
    return m


def _canonical_log_volume(raw_img):
    """Reproduce the production canonical-resample + anti-alias + LoG
    pipeline once per subject. Returns (log_arr_kji, ras_to_ijk_mat).
    """
    spacing = raw_img.GetSpacing()
    if min(float(s) for s in spacing) < cpfit.CANONICAL_SPACING_MM * 0.95:
        size_in = raw_img.GetSize()
        target_spacing = (cpfit.CANONICAL_SPACING_MM,)*3
        target_size = [
            max(1, int(round(size_in[i] * float(spacing[i]) / cpfit.CANONICAL_SPACING_MM)))
            for i in range(3)
        ]
        rs = sitk.ResampleImageFilter()
        rs.SetOutputSpacing(target_spacing)
        rs.SetSize(target_size)
        rs.SetOutputOrigin(raw_img.GetOrigin())
        rs.SetOutputDirection(raw_img.GetDirection())
        rs.SetInterpolator(sitk.sitkLinear)
        rs.SetDefaultPixelValue(-1024)
        img = rs.Execute(raw_img)
        if getattr(cpfit, "RAW_RESAMPLE_GAUSSIAN_SIGMA_MM", 0.0) > 0:
            img = sitk.SmoothingRecursiveGaussian(img, cpfit.RAW_RESAMPLE_GAUSSIAN_SIGMA_MM)
    else:
        img = raw_img

    img = sitk.Clamp(img, lowerBound=-1024.0, upperBound=cpfit.HU_CLIP_MAX)
    log_sitk = sitk.LaplacianRecursiveGaussian(
        sitk.Cast(img, sitk.sitkFloat32), sigma=cpfit.LOG_SIGMA_MM,
    )
    log_arr = sitk.GetArrayFromImage(log_sitk)
    _, ras_to_ijk = image_ijk_ras_matrices(img)
    return log_arr, np.asarray(ras_to_ijk, dtype=float)


def _sample_log_along_axis(log_arr_kji, ras_to_ijk_mat, start_ras, end_ras):
    """Sample LoG at SAMPLE_STEP_MM intervals from start_ras to end_ras.
    Returns (samples, n_samples, length_mm).
    """
    s = np.asarray(start_ras, dtype=float)
    e = np.asarray(end_ras, dtype=float)
    L = float(np.linalg.norm(e - s))
    if L < SAMPLE_STEP_MM * 8:  # need at least 8 samples for a meaningful FFT
        return None, 0, L
    n = max(8, int(round(L / SAMPLE_STEP_MM)) + 1)
    t = np.linspace(0.0, 1.0, n)
    pts_ras = s[None, :] + t[:, None] * (e - s)[None, :]
    h = np.concatenate([pts_ras, np.ones((n, 1))], axis=1)
    pts_ijk = (ras_to_ijk_mat @ h.T).T[:, :3]
    pts_kji = pts_ijk[:, [2, 1, 0]]
    samples = map_coordinates(log_arr_kji, pts_kji.T, order=1, mode='nearest')
    return samples.astype(float), n, L


def _dominant_pitch_in_band(samples, step_mm, pitch_range_mm=PITCH_RANGE_MM):
    """Return (dominant_pitch_mm, peak_power_normalized) or (None, 0).
    Detrends (linear), Hann-windows, FFTs, finds peak in physical-pitch band.
    """
    n = len(samples)
    if n < 8:
        return None, 0.0
    # Detrend
    x = np.arange(n, dtype=float)
    a, b = np.polyfit(x, samples, 1)
    samples_det = samples - (a * x + b)
    # Hann window
    samples_w = samples_det * np.hanning(n)
    # FFT
    spectrum = np.abs(np.fft.rfft(samples_w))
    freqs = np.fft.rfftfreq(n, d=step_mm)
    # Restrict to physical pitch band
    band_mask = (freqs >= 1.0/pitch_range_mm[1]) & (freqs <= 1.0/pitch_range_mm[0])
    if not band_mask.any():
        return None, 0.0
    band_spec = spectrum[band_mask]
    band_freqs = freqs[band_mask]
    peak_idx = int(np.argmax(band_spec))
    peak_freq = float(band_freqs[peak_idx])
    if peak_freq <= 0:
        return None, 0.0
    dominant_pitch = 1.0 / peak_freq
    # Normalize peak power against the median spectrum power (in band)
    median_band = float(np.median(band_spec))
    peak_power_norm = float(band_spec[peak_idx]) / max(1e-9, median_band)
    return dominant_pitch, peak_power_norm


def main():
    registry = PipelineRegistry()
    register_builtin_pipelines(registry)
    rows = iter_subject_rows(DATASET_ROOT, None)
    rows = [r for r in rows if str(r["subject_id"]) not in EXCLUDE]
    rows.sort(key=lambda r: int(str(r["subject_id"]).lstrip("T")))

    matched_rows = []
    orphan_rows = []
    print(f"running pipeline + pitch-FFT probe on {len(rows)} subjects "
          f"(step={SAMPLE_STEP_MM} mm, band={PITCH_RANGE_MM} mm)\n")
    t_all = time.time()

    for row in rows:
        subject_id = str(row["subject_id"])
        t_sub = time.time()
        gt, _ = load_reference_ground_truth_shanks(row)
        ctx, raw_img = build_detection_context(
            row["ct_path"], run_id=f"probe_fft_{subject_id}",
            config={}, extras={},
        )
        ctx["contact_pitch_v1_pitch_strategy"] = "auto"
        result = registry.run("contact_pitch_v1", ctx)
        if str(result.get("status", "ok")).lower() == "error":
            print(f"  {subject_id}: PIPELINE ERROR")
            continue
        trajs = list(result.get("trajectories") or [])
        matched = _greedy_match(gt, trajs)
        log_arr, ras_to_ijk = _canonical_log_volume(raw_img)

        n_match = 0
        for ti, t in enumerate(trajs):
            samples, n, L = _sample_log_along_axis(
                log_arr, ras_to_ijk,
                t["start_ras"], t["end_ras"],
            )
            if samples is None:
                continue
            dom_pitch, peak_norm = _dominant_pitch_in_band(samples, SAMPLE_STEP_MM)
            emitted = float(t.get("original_median_pitch_mm", 0.0))
            if dom_pitch is None or emitted <= 0:
                ratio = float("nan")
            else:
                ratio = dom_pitch / emitted
            band = str(t.get("confidence_label", "?"))
            row_out = {
                "subject": subject_id,
                "band": band,
                "bolt_source": str(t.get("bolt_source", "?")),
                "n_orig": int(t.get("n_inliers", 0)),
                "length_mm": L,
                "n_samples": n,
                "emitted_pitch": emitted,
                "dominant_pitch": dom_pitch if dom_pitch else float("nan"),
                "peak_power_norm": peak_norm,
                "ratio": ratio,
                "confidence": float(t.get("confidence", 0.0)),
                "pitch_dev_to_lib": (
                    min(abs(emitted - lp) for lp in cpfit.LIBRARY_PITCHES_MM)
                    if emitted > 0 else float("nan")
                ),
            }
            if ti in matched:
                row_out["shank"] = matched[ti]
                matched_rows.append(row_out)
                n_match += 1
            else:
                orphan_rows.append(row_out)

        print(f"  {subject_id:>4s}: {n_match}/{len(gt)} matched, "
              f"{len(trajs) - n_match} orphans  [{time.time()-t_sub:.1f}s]")

    print(f"\nTotal probe wall time: {time.time()-t_all:.1f}s")

    # ----- Aggregate stats: matched vs orphan distributions -----
    def _stats(rows, key):
        vals = np.array([r[key] for r in rows if not (isinstance(r[key], float) and np.isnan(r[key]))])
        if vals.size == 0: return None
        return {
            "n": int(vals.size),
            "p10": float(np.percentile(vals, 10)),
            "p50": float(np.percentile(vals, 50)),
            "p90": float(np.percentile(vals, 90)),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }

    print("\n" + "=" * 74)
    print("Aggregate distributions (matched vs orphan)")
    print("=" * 74)
    print(f"{'metric':<22s}  {'kind':>8s}  {'n':>4s}  {'p10':>7s} {'p50':>7s} {'p90':>7s}  {'min':>7s} {'max':>7s}")
    for key in ("ratio", "peak_power_norm", "pitch_dev_to_lib"):
        for kind, recs in (("matched", matched_rows), ("orphan", orphan_rows)):
            s = _stats(recs, key)
            if s is None: continue
            print(f"{key:<22s}  {kind:>8s}  {s['n']:>4d}  {s['p10']:>7.3f} {s['p50']:>7.3f} {s['p90']:>7.3f}  {s['min']:>7.3f} {s['max']:>7.3f}")

    # ----- Per-orphan dump -----
    print("\n" + "=" * 74)
    print(f"All {len(orphan_rows)} orphans, sorted by (band, |ratio - 1| desc)")
    print("=" * 74)
    print(f"{'subj':>4s} {'band':>6s} {'bolt':>11s} {'emitted':>8s} {'dom':>7s} "
          f"{'ratio':>6s} {'pkN':>5s} {'libdev':>6s} {'n':>3s} {'conf':>5s}")
    band_order = {"high": 0, "medium": 1, "low": 2, "?": 3}
    orphan_rows.sort(key=lambda r: (
        band_order.get(r["band"], 99),
        -abs(r.get("ratio", 1.0) - 1.0) if not np.isnan(r["ratio"]) else 0
    ))
    for r in orphan_rows:
        ratio = r["ratio"]
        flag = ""
        if not np.isnan(ratio):
            if 0.85 <= ratio <= 1.15:
                flag = "  ok"
            elif 0.40 <= ratio <= 0.60:
                flag = "  K=2 walker-miss"
            elif 0.30 <= ratio <= 0.36:
                flag = "  K=3 walker-miss?"
            else:
                flag = "  off"
        print(f"{r['subject']:>4s} {r['band']:>6s} {r['bolt_source']:>11s} "
              f"{r['emitted_pitch']:>8.2f} {r['dominant_pitch']:>7.2f} "
              f"{ratio:>6.2f} {r['peak_power_norm']:>5.2f} "
              f"{r['pitch_dev_to_lib']:>6.2f} "
              f"{r['n_orig']:>3d} {r['confidence']:>5.2f}{flag}")

    # ----- Targeted summary: surviving 5 from rotation+scale-space -----
    print("\n" + "=" * 74)
    print("Targets: orphans that PASSED both rotation and scale-space tests")
    print("=" * 74)
    targets = [
        ("T1", 3.73, "log"),
        ("T1", 6.75, "none"),
        ("T1", 6.89, "none"),
        ("T3", 3.61, "log"),
        ("T3", 3.68, "none"),
    ]
    for subj, pitch, bolt in targets:
        candidates = [
            r for r in orphan_rows
            if r["subject"] == subj
            and abs(r["emitted_pitch"] - pitch) < 0.10
            and r["bolt_source"] == bolt
        ]
        if candidates:
            r = candidates[0]
            ratio = r["ratio"]
            verdict = ("WALKER-MISS (drop)" if not np.isnan(ratio) and 0.40 <= ratio <= 0.60
                       else "OK (likely real)" if not np.isnan(ratio) and 0.85 <= ratio <= 1.15
                       else "off-band")
            print(f"  {subj} bolt={bolt:>4s} emitted={pitch:.2f} -> dom={r['dominant_pitch']:.2f} "
                  f"ratio={ratio:.2f} pkN={r['peak_power_norm']:.2f}  ==> {verdict}")
        else:
            print(f"  {subj} bolt={bolt:>4s} emitted={pitch:.2f}  (not found in this run)")

    # ----- Matched discrimination preview -----
    matched_ratios = np.array([
        r["ratio"] for r in matched_rows
        if not np.isnan(r["ratio"])
    ])
    n_matched = matched_ratios.size
    n_walker_miss = int(np.sum((matched_ratios >= 0.40) & (matched_ratios <= 0.60)))
    n_ok = int(np.sum((matched_ratios >= 0.85) & (matched_ratios <= 1.15)))
    print(f"\nMatched ratio distribution: ok-band [0.85, 1.15] = {n_ok}/{n_matched} "
          f"({n_ok/n_matched:.1%}), walker-miss [0.40, 0.60] = {n_walker_miss}/{n_matched} "
          f"({n_walker_miss/n_matched:.1%})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
