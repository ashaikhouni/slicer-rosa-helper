"""One-shot baseline probe: run contact_pitch_v1 on all 22 dataset
subjects (T17 / T19 / T21 excluded per the localization memo) and
print per-subject matched / orphans + dataset totals. Used to seed
the assertions in ``test_pipeline_dataset_contact_pitch_v1.test_dataset_full``.

Not part of CI; run by hand when the orphan budget needs to be
re-baselined after a deliberate algorithm change.
"""
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))
sys.path.insert(0, str(REPO_ROOT / "PostopCTLocalization"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import numpy as np
from eval_seeg_localization import (
    build_detection_context,
    iter_subject_rows,
    load_reference_ground_truth_shanks,
)
from rosa_detect.service import run_contact_pitch_v1


DATASET_ROOT = Path(
    os.environ.get(
        "ROSA_SEEG_DATASET",
        "/Users/ammar/Dropbox/thalamus_subjects/seeg_localization",
    )
)
EXCLUDED = {"T17", "T19", "T21"}
MAX_ANGLE_DEG = 10.0
MAX_MID_MM = 8.0


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def _match_one(row, registry):
    sid = str(row["subject_id"])
    gt, _ = load_reference_ground_truth_shanks(row)
    ctx, _ = build_detection_context(
        row["ct_path"], run_id=f"contact_pitch_{sid}", config={}, extras={}
    )
    result = run_contact_pitch_v1(ctx)
    assert result.get("status") == "ok", f"{sid}: {result.get('error')}"
    trajs = list(result.get("trajectories") or [])

    pairs = []
    for gi, g in enumerate(gt):
        g_s = np.asarray(g.start_ras, dtype=float)
        g_e = np.asarray(g.end_ras, dtype=float)
        g_axis = _unit(g_e - g_s)
        g_mid = 0.5 * (g_s + g_e)
        for ti, t in enumerate(trajs):
            t_s = np.asarray(t["start_ras"], dtype=float)
            t_e = np.asarray(t["end_ras"], dtype=float)
            t_axis = _unit(t_e - t_s)
            ang = float(np.degrees(np.arccos(
                min(1.0, abs(float(np.dot(g_axis, t_axis)))))))
            t_mid = 0.5 * (t_s + t_e)
            v = g_mid - t_mid
            perp = v - float(np.dot(v, t_axis)) * t_axis
            mid_d = float(np.linalg.norm(perp))
            if ang <= MAX_ANGLE_DEG and mid_d <= MAX_MID_MM:
                pairs.append((ang + mid_d, gi, ti))
    pairs.sort(key=lambda p: p[0])
    used_gt: set[int] = set()
    used_tr: set[int] = set()
    for _s, gi, ti in pairs:
        if gi in used_gt or ti in used_tr:
            continue
        used_gt.add(gi)
        used_tr.add(ti)
    n_matched = len(used_gt)
    n_total = len(gt)
    n_fp = len(trajs) - n_matched
    return n_matched, n_total, n_fp


def main():
    rows = [
        r for r in iter_subject_rows(DATASET_ROOT, None)
        if str(r["subject_id"]) not in EXCLUDED
    ]
    rows.sort(key=lambda r: int(str(r["subject_id"]).lstrip("T")))
    print(f"running contact_pitch_v1 on {len(rows)} subjects "
          f"(excluded: {sorted(EXCLUDED)})")
    t0 = time.time()
    total_matched = total_gt = total_orphans = 0
    per_subject = []
    for row in rows:
        sid = str(row["subject_id"])
        ts = time.time()
        n_matched, n_total, n_fp = _match_one(row, registry)
        dt = time.time() - ts
        total_matched += n_matched
        total_gt += n_total
        total_orphans += n_fp
        miss = n_total - n_matched
        print(f"  {sid:5s} matched {n_matched:2d}/{n_total:2d}"
              f"  orphans {n_fp:3d}"
              f"  miss {miss}"
              f"  ({dt:.1f}s)")
        per_subject.append((sid, n_matched, n_total, n_fp, miss, dt))
    elapsed = time.time() - t0
    print()
    print(f"DATASET total: matched {total_matched}/{total_gt}  "
          f"orphans {total_orphans}  elapsed {elapsed:.1f}s")
    bad = [s for s in per_subject if s[4] > 0]
    if bad:
        print(f"\nrecall regressions on {len(bad)} subjects:")
        for sid, m, n, _fp, miss, _dt in bad:
            print(f"  {sid}: missed {miss} ({m}/{n})")


if __name__ == "__main__":
    main()
