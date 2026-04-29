"""End-to-end regression for the contact_pitch_v1 detector.

Matches the probe's axis-based acceptance criterion (``angle ≤ 10°``
and ``mid_d ≤ 8 mm``). Start/end errors are deliberately NOT gated:
stage-1 lines span the contact range, which is typically offset from
the GT bolt-entry/deep-tip endpoints.

Two test surfaces:

  - Per-subject quick gates (``test_T22``, ``test_T2``,
    ``test_T2_auto_strategy``) — fast iteration for code changes
    on Auto Fit. Combined runtime ≈ 15 s.
  - Full-dataset gate (``test_dataset_full``) — runs all 22 subjects
    (T17 / T19 / T21 excluded; their GT is unreliable per
    ``feedback_gt_completeness.md``), asserts recall + orphan budget.
    Runtime ≈ 70 s. This is the regression net for refactor work.

See ``slicer-rosa-helper/docs/HANDOFF.md`` for current pipeline
state, recall numbers, and the score-band policy.
"""
import os
import sys
import unittest
from pathlib import Path

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
        import numpy  # noqa: F401
        import SimpleITK  # noqa: F401
        from shank_engine import (  # noqa: F401
            PipelineRegistry,
            register_builtin_pipelines,
        )
        return True
    except ImportError:
        return False


DEPS_AVAILABLE = _try_imports()


@unittest.skipUnless(
    DATASET_AVAILABLE,
    f"SEEG dataset not found at {SUBJECTS_TSV}. Set ROSA_SEEG_DATASET to override.",
)
@unittest.skipUnless(
    DEPS_AVAILABLE,
    "numpy/SimpleITK/shank_engine not importable in this environment.",
)
class ContactPitchV1DatasetRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from shank_engine import PipelineRegistry, register_builtin_pipelines

        cls.registry = PipelineRegistry()
        register_builtin_pipelines(cls.registry)

    def _run_subject(self, subject_id, *, max_angle_deg=10.0,
                      max_mid_mm=8.0, deep_core_config_overrides=None,
                      pitch_strategy=None):
        """Run v3 pipeline on a subject and match predictions to GT using
        the probe's axis-midpoint criterion (angle + perpendicular midpoint
        distance). Greedy 1-to-1 assignment.
        """
        import numpy as np
        from eval_seeg_localization import (
            build_detection_context,
            iter_subject_rows,
            load_reference_ground_truth_shanks,
        )

        rows = iter_subject_rows(DATASET_ROOT, {subject_id})
        self.assertEqual(len(rows), 1, f"Subject {subject_id} not in manifest")
        row = rows[0]

        gt, _ = load_reference_ground_truth_shanks(row)
        config = dict(deep_core_config_overrides or {})
        run_tag = f"contact_pitch_{subject_id}"
        if pitch_strategy:
            run_tag = f"{run_tag}_{pitch_strategy}"
        ctx, _ = build_detection_context(
            row["ct_path"], run_id=run_tag, config=config, extras={}
        )
        if pitch_strategy:
            ctx["contact_pitch_v1_pitch_strategy"] = pitch_strategy
        result = self.registry.run("contact_pitch_v1", ctx)
        self.assertEqual(
            result.get("status"), "ok",
            f"Pipeline failed: {result.get('error')}",
        )
        trajs = list(result.get("trajectories") or [])

        def _unit(v):
            n = float(np.linalg.norm(v))
            return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])

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
                if ang <= max_angle_deg and mid_d <= max_mid_mm:
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
        n_fp = len(trajs) - n_matched
        return gt, trajs, n_matched, n_fp

    # FP caps loosened from 10 to 20 when the LoG-blob extractor switched
    # from Ball r=2 (81-voxel kernel, max diagonal reach 2.45 voxels) to
    # Box r=1 (3×3×3 cube, max corner reach √3 voxels). The smaller
    # kernel admits more local-min candidates per subject — recovering
    # LSFG-class shanks that the wider Ball missed via diagonal-corner
    # interference — at the cost of more walker-emitted orphans. The
    # primary recall guard (n_matched ≥ N_GT) is unchanged; the FP cap
    # is now a runaway guardrail rather than an "FPs ≤ 10" target.
    # Score framework downstream classifies most extra orphans as
    # medium/low confidence.
    def test_T22(self):
        gt, trajs, n_matched, n_fp = self._run_subject("T22")
        self.assertEqual(len(gt), 9, "T22 GT count drifted")
        self.assertGreaterEqual(
            n_matched, 8, f"contact_pitch_v1 match regressed: {n_matched}/{len(gt)}"
        )
        self.assertLessEqual(n_fp, 20, f"contact_pitch_v1 FP count regressed: {n_fp}")

    def test_T2(self):
        gt, trajs, n_matched, n_fp = self._run_subject("T2")
        self.assertGreaterEqual(
            n_matched, 12, f"contact_pitch_v1 match regressed: {n_matched}/{len(gt)}"
        )
        self.assertLessEqual(n_fp, 20, f"contact_pitch_v1 FP count regressed: {n_fp}")

    def test_T2_auto_strategy(self):
        # Auto-detect pitch snaps the mutual-NN centroid (~3.3 mm on a
        # true-3.5 mm Dixi case) to the nearest library pitch within
        # 0.3 mm. Without the snap, auto lost one shank at the band
        # edge (11/12); with the snap it matches the Dixi default.
        gt, trajs, n_matched, n_fp = self._run_subject(
            "T2", pitch_strategy="auto",
        )
        self.assertGreaterEqual(
            n_matched, 12,
            f"contact_pitch_v1 auto-strategy match regressed: {n_matched}/{len(gt)}",
        )
        self.assertLessEqual(
            n_fp, 20,
            f"contact_pitch_v1 auto-strategy FP count regressed: {n_fp}",
        )

    # Subjects with GT considered unreliable per
    # ``feedback_gt_completeness.md`` — excluded from the full-dataset
    # gate so orphan-vs-real-shank ambiguity doesn't poison the budget.
    DATASET_EXCLUDED_SUBJECTS = frozenset({"T17", "T19", "T21"})

    # Full-dataset gate baseline (pipeline 1.0.29, 2026-04-29):
    #   295 / 295 matched, 15 orphans, ~70 s on a 22-subject run.
    # Bumping this without an accompanying memory entry explaining
    # WHY the budget moved is a regression smell — see
    # ``feedback_gt_completeness.md``: orphans are FPs, not "shanks
    # GT missed", so the budget should drop or hold over time, never
    # silently grow.
    DATASET_REQUIRED_RECALL = 295
    DATASET_ORPHAN_BUDGET = 20

    def test_dataset_full(self):
        from eval_seeg_localization import iter_subject_rows

        rows = [
            r for r in iter_subject_rows(DATASET_ROOT, None)
            if str(r["subject_id"]) not in self.DATASET_EXCLUDED_SUBJECTS
        ]
        self.assertGreater(
            len(rows), 0,
            f"no subjects in manifest at {SUBJECTS_TSV}",
        )

        total_matched = 0
        total_gt = 0
        total_orphans = 0
        # Per-subject failures are logged so a regression report shows
        # WHICH subject(s) lost shanks, not just the dataset totals.
        per_subject = []
        for row in rows:
            sid = str(row["subject_id"])
            gt, trajs, n_matched, n_fp = self._run_subject(sid)
            n_total = len(gt)
            total_matched += n_matched
            total_gt += n_total
            total_orphans += n_fp
            per_subject.append((sid, n_matched, n_total, n_fp))

        # Total recall over the dataset. Because each subject's matched
        # count is bounded by its GT count, ``total_matched ≥ total_gt``
        # implies every subject hit 100 % of its GT — there is no way
        # for one subject's loss to be hidden by another's gain.
        if total_matched < self.DATASET_REQUIRED_RECALL:
            misses = [
                f"{sid}: {m}/{n}"
                for sid, m, n, _ in per_subject if m < n
            ]
            self.fail(
                f"contact_pitch_v1 dataset recall regressed: "
                f"{total_matched}/{total_gt} matched (need ≥ "
                f"{self.DATASET_REQUIRED_RECALL}). Per-subject misses: "
                + ", ".join(misses)
            )
        self.assertGreaterEqual(total_gt, self.DATASET_REQUIRED_RECALL)

        # Orphan budget. Tightening this asserts the score framework
        # is doing its job — every orphan above the budget is an FP we
        # don't have a story for.
        if total_orphans > self.DATASET_ORPHAN_BUDGET:
            sorted_offenders = sorted(
                per_subject, key=lambda r: r[3], reverse=True,
            )[:6]
            offenders = [
                f"{sid}: {fp} orphans"
                for sid, _m, _n, fp in sorted_offenders if fp > 0
            ]
            self.fail(
                f"contact_pitch_v1 dataset orphan count regressed: "
                f"{total_orphans} > budget {self.DATASET_ORPHAN_BUDGET}. "
                f"Top offenders: " + ", ".join(offenders)
            )


if __name__ == "__main__":
    unittest.main()
