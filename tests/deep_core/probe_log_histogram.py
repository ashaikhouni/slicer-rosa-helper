"""Per-subject |LoG| distribution probe for contact_pitch_v1.

Question
--------
Today ``LOG_BLOB_THRESHOLD = 300`` and ``BOLT_LOG_THRESHOLD = 800`` are
fixed across all subjects. Is the |LoG| histogram actually similar
across subjects (so fixed thresholds are fine), or does it vary enough
that a fixed threshold samples different percentiles on different
subjects (i.e. magic numbers pretending to be universal)?

Method
------
For each subject in the SEEG dataset (plus AMC099 / ct88 where
present):
  1. Load CT via SITK, build hull + intracranial mask + LoG sigma=1.
  2. Restrict to intracranial voxels; compute the ``contact-amplitude''
     signal ``A = max(0, -LoG)`` (matches what ``extract_blobs`` uses —
     only the dark-spot side of the LoG field).
  3. Report distribution quantiles + voxel counts above current
     thresholds + regional-minima blob counts at several thresholds.
  4. Summarize ratios across subjects to see the magic-number damage.

If the p99.9 (or equivalently ``blobs_300 / blobs_1000``) ratio across
subjects exceeds ~2×, fixed thresholds don't generalize and we need
per-subject auto-threshold (probably percentile- or blob-count-based).

Run
---
    /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_log_histogram.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path("/Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper")
sys.path.insert(0, str(ROOT / "CommonLib"))
sys.path.insert(0, str(ROOT / "PostopCTLocalization"))

import numpy as np
import SimpleITK as sitk

from postop_ct_localization.contact_pitch_v1_fit import (
    build_masks, log_sigma, extract_blobs,
    LOG_SIGMA_MM, LOG_BLOB_THRESHOLD, HU_CLIP_MAX,
)


DATASET_ROOT = Path(os.environ.get(
    "ROSA_SEEG_DATASET",
    "/Users/ammar/Dropbox/thalamus_subjects/seeg_localization",
))
CT_DIR = DATASET_ROOT / "contact_label_dataset" / "ct"

# Subjects in the Tx manifest, minus T19 / T21 per memory
# (``reference_seeg_dataset.md``).
TX_SUBJECTS = [
    "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10",
    "T11", "T12", "T13", "T14", "T15", "T16", "T17", "T18",
    "T20", "T22", "T23", "T24", "T25",
]

EXTRA_SUBJECTS = [
    ("AMC099",
     "/Users/ammar/Dropbox/MRI-Pipeline/inbox/brunner/FREESURFER/AMC099/NIfTI/CT/CT.nii.gz"),
    ("ct88", "/Users/ammar/Dropbox/tmp/ct88.nii.gz"),
]

# Thresholds we want to inspect.
THRESHOLDS = [300.0, 500.0, 800.0, 1000.0, 1500.0]
# Quantiles on A = max(0, -LoG) restricted to intracranial voxels.
QUANTILES = [0.990, 0.995, 0.999, 0.9995, 0.9999, 0.99999]


def _subject_ct_path(subject: str) -> Path | None:
    p = CT_DIR / f"{subject}_ct.nii.gz"
    return p if p.exists() else None


def _probe_subject(subject: str, ct_path: Path) -> dict:
    """Return one row of measurements for one subject."""
    t0 = time.perf_counter()
    img = sitk.ReadImage(str(ct_path))
    # Mirror the scanner-invariance clip that run_two_stage_detection
    # applies, so this probe measures the distribution the walker
    # actually sees.
    img = sitk.Clamp(img, lowerBound=-1024.0, upperBound=HU_CLIP_MAX)
    hull, intracranial, _dist = build_masks(img)
    log1 = log_sigma(img, sigma_mm=LOG_SIGMA_MM)
    # Contact-amplitude signal — only dark-side LoG matters for contacts.
    # Restrict to intracranial voxels so skull / air noise does not skew
    # the distribution.
    intra_mask = intracranial.astype(bool)
    amp = np.where(log1 < 0.0, -log1, 0.0).astype(np.float32)
    amp_intra = amp[intra_mask]
    n_intra = int(amp_intra.size)
    # Also report CT HU stats: if HU distribution varies across subjects
    # (e.g. one scan saturates metal at 3000 HU, another at 8000), LoG
    # response scales linearly with that variation — which would explain
    # per-subject LoG distribution spread WITHOUT subject-variable
    # contact density. In that case, the principled fix is HU clipping
    # before LoG, not per-subject threshold.
    ct_arr = sitk.GetArrayFromImage(img).astype(np.float32)
    ct_intra = ct_arr[intra_mask]
    # Global HU max too — metal contacts outside the hull (bolts) matter
    # for bolt detection.
    hu_quantiles_intra = np.quantile(
        ct_intra, [0.9, 0.99, 0.999, 0.9999, 1.0],
    ) if ct_intra.size > 0 else np.zeros(5)
    hu_global_max = float(ct_arr.max())

    qs = np.quantile(amp_intra, QUANTILES) if n_intra > 0 else np.zeros(len(QUANTILES))
    vox_ge = {}
    for thr in THRESHOLDS:
        vox_ge[thr] = int(np.count_nonzero(amp_intra >= thr))

    # Regional-minima blob counts at each threshold (full volume, not
    # just intracranial — extract_blobs handles that). These are the
    # actual "contact candidate" counts the walker would see at each
    # threshold.
    blobs_count = {}
    for thr in THRESHOLDS:
        blobs = extract_blobs(log1, threshold=thr)
        blobs_count[thr] = len(blobs)

    wall = time.perf_counter() - t0
    return {
        "subject": subject,
        "n_intra": n_intra,
        "quantiles": qs,
        "vox_ge": vox_ge,
        "blobs_count": blobs_count,
        "hu_q_intra": hu_quantiles_intra,
        "hu_global_max": hu_global_max,
        "wall_s": wall,
    }


def _fmt(x, width=10, prec=1):
    if isinstance(x, (int, np.integer)):
        return f"{x:{width}d}"
    return f"{x:{width}.{prec}f}"


def main():
    rows = []

    for subj in TX_SUBJECTS:
        p = _subject_ct_path(subj)
        if p is None:
            print(f"{subj}: NOT FOUND at {CT_DIR / f'{subj}_ct.nii.gz'}")
            continue
        r = _probe_subject(subj, p)
        rows.append(r)
        q99_9 = r["quantiles"][2]
        print(
            f"{subj:9s} wall={r['wall_s']:4.1f}s "
            f"n_intra={r['n_intra']:>9d} "
            f"p99.9={q99_9:7.1f}  "
            f"blobs@300={r['blobs_count'][300.0]:>5d}  "
            f"blobs@800={r['blobs_count'][800.0]:>5d}  "
            f"blobs@1000={r['blobs_count'][1000.0]:>5d}"
        )

    for subj, ct_path in EXTRA_SUBJECTS:
        p = Path(ct_path)
        if not p.exists():
            print(f"{subj}: NOT FOUND at {p}")
            continue
        r = _probe_subject(subj, p)
        rows.append(r)
        q99_9 = r["quantiles"][2]
        print(
            f"{subj:9s} wall={r['wall_s']:4.1f}s "
            f"n_intra={r['n_intra']:>9d} "
            f"p99.9={q99_9:7.1f}  "
            f"blobs@300={r['blobs_count'][300.0]:>5d}  "
            f"blobs@800={r['blobs_count'][800.0]:>5d}  "
            f"blobs@1000={r['blobs_count'][1000.0]:>5d}"
        )

    if not rows:
        print("no subjects processed.")
        return

    # ---- Full table ----
    print()
    print("=" * 140)
    print("PER-SUBJECT QUANTILES (of A = max(0, -LoG) inside intracranial mask)")
    print("=" * 140)
    header = (
        f"{'subject':9s}  {'p99.0':>8s} {'p99.5':>8s} {'p99.9':>8s} "
        f"{'p99.95':>8s} {'p99.99':>8s} {'p99.999':>9s}  "
        f"{'v>=300':>9s} {'v>=500':>9s} {'v>=800':>9s} {'v>=1000':>9s} {'v>=1500':>9s}  "
        f"{'b@300':>6s} {'b@500':>6s} {'b@800':>6s} {'b@1000':>6s} {'b@1500':>6s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        q = r["quantiles"]
        v = r["vox_ge"]
        b = r["blobs_count"]
        print(
            f"{r['subject']:9s}  "
            f"{q[0]:8.1f} {q[1]:8.1f} {q[2]:8.1f} {q[3]:8.1f} {q[4]:8.1f} {q[5]:9.1f}  "
            f"{v[300.0]:9d} {v[500.0]:9d} {v[800.0]:9d} {v[1000.0]:9d} {v[1500.0]:9d}  "
            f"{b[300.0]:6d} {b[500.0]:6d} {b[800.0]:6d} {b[1000.0]:6d} {b[1500.0]:6d}"
        )

    # ---- Cross-subject ratio summary ----
    def _col_stats(values):
        arr = np.array(values, dtype=float)
        return float(np.min(arr)), float(np.median(arr)), float(np.max(arr))

    print()
    print("=" * 140)
    print("CROSS-SUBJECT SPREAD — if max/min > 2.0, fixed thresholds are subject-dependent")
    print("=" * 140)
    print(f"{'metric':20s}  {'min':>12s} {'median':>12s} {'max':>12s} {'max/min':>10s}")
    print("-" * 72)

    def _emit_row(label, values):
        mn, med, mx = _col_stats(values)
        ratio = mx / mn if mn > 0 else float("inf")
        print(
            f"{label:20s}  {mn:12.1f} {med:12.1f} {mx:12.1f} {ratio:10.2f}"
        )

    qs = np.array([r["quantiles"] for r in rows])
    for qi, q in enumerate(QUANTILES):
        _emit_row(f"p{q * 100:.3f}", qs[:, qi].tolist())

    for thr in THRESHOLDS:
        _emit_row(f"blobs @ {thr:.0f}", [r["blobs_count"][thr] for r in rows])

    # ---- HU-vs-LoG correlation: does subject LoG spread track CT HU spread? ----
    print()
    print("=" * 140)
    print("HU RANGE vs LoG p99.9 — if they correlate, LoG spread is driven by HU scaling, not contact density")
    print("=" * 140)
    print(
        f"{'subject':9s}  {'HU_intra_p99':>12s} {'HU_intra_p99.9':>14s} "
        f"{'HU_intra_p99.99':>15s} {'HU_intra_max':>12s} {'HU_global_max':>14s}  "
        f"{'LoG_p99.9':>10s}  {'LoG/HU_p99.9':>13s}"
    )
    for r in rows:
        hu_q = r["hu_q_intra"]
        log_p999 = r["quantiles"][2]
        # Use HU_intra_p99 (hu_q[0]) as the "contact HU floor" proxy. If
        # LoG_p99.9 / HU_intra_p99 is constant across subjects, LoG is
        # just tracking HU.
        hu_p99 = hu_q[0]
        ratio = log_p999 / hu_p99 if hu_p99 > 0 else float("nan")
        print(
            f"{r['subject']:9s}  "
            f"{hu_q[0]:12.0f} {hu_q[1]:14.0f} {hu_q[2]:15.0f} "
            f"{hu_q[4]:12.0f} {r['hu_global_max']:14.0f}  "
            f"{log_p999:10.1f}  {ratio:13.3f}"
        )

    # Spread summary on HU stats.
    print()
    print(f"{'metric':20s}  {'min':>12s} {'median':>12s} {'max':>12s} {'max/min':>10s}")
    print("-" * 72)
    _emit_row("HU_intra_p99",   [r["hu_q_intra"][0] for r in rows])
    _emit_row("HU_intra_p99.9", [r["hu_q_intra"][1] for r in rows])
    _emit_row("HU_intra_max",   [r["hu_q_intra"][4] for r in rows])
    _emit_row("HU_global_max",  [r["hu_global_max"] for r in rows])

    # Contact-count calibration: how many blobs at the current 300
    # threshold vs the expected contact count? Read from subjects.tsv.
    try:
        import csv
        manifest = {}
        with (DATASET_ROOT / "contact_label_dataset" / "subjects.tsv").open() as f:
            rd = csv.DictReader(f, delimiter="\t")
            for row in rd:
                manifest[row["subject_id"]] = int(row["n_contacts"])
        print()
        print("=" * 140)
        print("BLOBS-vs-GT-CONTACTS — ratio blobs_300 / gt_contacts per subject")
        print("=" * 140)
        print(f"{'subject':9s}  {'gt_contacts':>12s} {'blobs@300':>10s} "
              f"{'blobs@500':>10s} {'blobs@800':>10s}   "
              f"{'b300/gt':>10s} {'b500/gt':>10s} {'b800/gt':>10s}")
        for r in rows:
            gt = manifest.get(r["subject"])
            if gt is None:
                continue
            b3 = r["blobs_count"][300.0]
            b5 = r["blobs_count"][500.0]
            b8 = r["blobs_count"][800.0]
            print(
                f"{r['subject']:9s}  {gt:12d} {b3:10d} {b5:10d} {b8:10d}   "
                f"{b3 / gt:10.2f} {b5 / gt:10.2f} {b8 / gt:10.2f}"
            )
    except Exception as exc:
        print(f"(manifest GT comparison skipped: {exc})")


if __name__ == "__main__":
    main()
