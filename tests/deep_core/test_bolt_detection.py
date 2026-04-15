"""Bolt detection stage tests.

Runs the mask + bolt_detection stages on T1 and T22 and asserts that each GT
shank has a bolt candidate aligned with its trajectory. Locks the baseline
established during the bolt design phase:

  - T1: all 12 GT shanks matched by a bolt within angle<=15°, perp<=8mm
  - T22: at least 8 of 9 GT shanks matched (RIFG is a borderline case and
    may or may not clear the strict gate depending on RANSAC RNG drift)

Also caps false-positive counts at the values we observed in the probe so
regressions will flag a mask/config drift that produces many spurious lines.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))
sys.path.insert(0, str(REPO_ROOT / "PostopCTLocalization"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

DATASET_ROOT = Path(
    os.environ.get(
        "ROSA_SEEG_DATASET",
        "/Users/ammar/Dropbox/thalamus_subjects/seeg_localization",
    )
)
SUBJECTS_TSV = DATASET_ROOT / "contact_label_dataset" / "subjects.tsv"
DATASET_AVAILABLE = SUBJECTS_TSV.exists()


def _try_imports():
    try:
        import SimpleITK  # noqa: F401
        from postop_ct_localization.deep_core_bolt_ransac import (  # noqa: F401
            find_bolt_candidates,
        )
        from shank_engine.pipelines.deep_core_v1 import DeepCoreV1Pipeline  # noqa: F401
        return True
    except ImportError:
        return False


DEPS_AVAILABLE = _try_imports()


def _match_bolts_to_gt(bolts, gt, ang_tol_deg: float = 15.0, perp_tol_mm: float = 8.0):
    """Greedy best-first assignment of bolts to GT shanks."""
    used: set[int] = set()
    hits: list[tuple[str, float, float]] = []
    rows = []
    for shank in gt:
        s = np.asarray(shank.start_ras, dtype=float)
        e = np.asarray(shank.end_ras, dtype=float)
        axis_gt = e - s
        n = np.linalg.norm(axis_gt)
        if n < 1e-6:
            continue
        axis_gt /= n
        scores = []
        for i, b in enumerate(bolts):
            v = b.center_ras - s
            perp = v - np.dot(v, axis_gt) * axis_gt
            pd = float(np.linalg.norm(perp))
            cos = abs(float(np.dot(b.axis_ras, axis_gt)))
            cos = min(1.0, max(-1.0, cos))
            ang = float(np.degrees(np.arccos(cos)))
            ep = min(
                float(np.linalg.norm(b.center_ras - s)),
                float(np.linalg.norm(b.center_ras - e)),
            )
            scores.append((pd + 1.5 * ang + 0.2 * ep, ang, pd, ep, i))
        rows.append((shank.shank, scores))

    ordered = sorted(range(len(rows)), key=lambda i: min(s[0] for s in rows[i][1]) if rows[i][1] else 1e9)
    for ri in ordered:
        _, scores = rows[ri]
        if not scores:
            continue
        best = None
        for sc in scores:
            if sc[4] in used:
                continue
            if best is None or sc[0] < best[0]:
                best = sc
        if best is None:
            continue
        score, ang, pd, ep, idx = best
        if ang <= ang_tol_deg and pd <= perp_tol_mm:
            used.add(idx)
            hits.append((rows[ri][0], ang, pd))
    return hits


@unittest.skipUnless(
    DATASET_AVAILABLE,
    f"SEEG dataset not found at {SUBJECTS_TSV}. Set ROSA_SEEG_DATASET to override.",
)
@unittest.skipUnless(
    DEPS_AVAILABLE,
    "SimpleITK / deep_core_bolt_ransac / shank_engine not importable.",
)
class DeepCoreBoltDetectionTests(unittest.TestCase):
    """Validate the bolt detection stage on T1 and T22."""

    def _run_bolt_stage(self, subject_id: str, config_overrides=None):
        from eval_seeg_localization import (
            build_detection_context,
            iter_subject_rows,
            load_reference_ground_truth_shanks,
        )
        from shank_engine.pipelines.deep_core_v1 import DeepCoreV1Pipeline
        from postop_ct_localization.deep_core_config import (
            deep_core_default_config,
        )

        rows = iter_subject_rows(DATASET_ROOT, {subject_id})
        self.assertEqual(len(rows), 1, f"Subject {subject_id} not in manifest")
        row = rows[0]
        gt, _ = load_reference_ground_truth_shanks(row)

        config = dict(config_overrides or {})
        ctx, _ = build_detection_context(
            row["ct_path"], run_id=f"bolt_{subject_id}", config=config, extras={}
        )
        cfg = deep_core_default_config()
        if config:
            cfg = cfg.with_updates(config)

        pipeline = DeepCoreV1Pipeline()
        mask = pipeline._run_mask(ctx, cfg)
        bolt = pipeline._run_bolt_detection(ctx, mask, cfg)
        return gt, bolt["candidates"], mask, bolt.get("stats", {})

    def test_T1_bolts_cover_all_shanks(self):
        gt, bolts, mask, stats = self._run_bolt_stage("T1")
        self.assertEqual(len(gt), 12, "T1 GT count drifted")
        self.assertGreaterEqual(
            len(bolts), 12, f"T1 bolt count below GT ({len(bolts)})"
        )
        self.assertLessEqual(
            len(bolts), 30, f"T1 bolt count too high (possible mask drift): {len(bolts)}"
        )
        hits = _match_bolts_to_gt(bolts, gt)
        self.assertEqual(
            len(hits), 12,
            f"T1 bolts missed GT shanks: {[s.shank for s in gt] } vs hits {[h[0] for h in hits]}",
        )
        fps = len(bolts) - len(hits)
        self.assertLessEqual(
            fps, 15,
            f"T1 bolt false positives exceeded baseline: {fps}",
        )

    def test_T22_bolts_cover_all_shanks(self):
        gt, bolts, mask, stats = self._run_bolt_stage("T22")
        self.assertEqual(len(gt), 9, "T22 GT count drifted")
        self.assertGreaterEqual(
            len(bolts), 9, f"T22 bolt count below GT ({len(bolts)})"
        )
        self.assertLessEqual(
            len(bolts), 25, f"T22 bolt count too high: {len(bolts)}"
        )
        hits = _match_bolts_to_gt(bolts, gt)
        # T22 RIFG is borderline; require at least 8 of 9 to hit the strict gate.
        self.assertGreaterEqual(
            len(hits), 8,
            f"T22 bolt hits below baseline: {len(hits)}/9  "
            f"(hits={[h[0] for h in hits]})",
        )
        fps = len(bolts) - len(hits)
        self.assertLessEqual(
            fps, 15,
            f"T22 bolt false positives exceeded baseline: {fps}",
        )

    def test_bolt_disabled_returns_empty(self):
        gt, bolts, mask, stats = self._run_bolt_stage(
            "T22", config_overrides={"bolt.enabled": False}
        )
        self.assertEqual(bolts, [])
        self.assertFalse(stats.get("enabled", True))


if __name__ == "__main__":
    unittest.main()
