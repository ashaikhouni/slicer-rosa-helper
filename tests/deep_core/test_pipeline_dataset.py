"""End-to-end regression tests for deep_core_v1 against the SEEG dataset.

These tests run the full pipeline on real subjects from the
``thalamus_subjects/seeg_localization`` dataset and assert that
detection metrics match the locked-in baselines.

Skipped when the dataset isn't available.
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
        from shank_engine import PipelineRegistry, register_builtin_pipelines  # noqa: F401
        return True
    except ImportError:
        return False


DEPS_AVAILABLE = _try_imports()


@unittest.skipUnless(
    DATASET_AVAILABLE,
    f"SEEG dataset not found at {SUBJECTS_TSV}. "
    "Set ROSA_SEEG_DATASET to override.",
)
@unittest.skipUnless(
    DEPS_AVAILABLE,
    "numpy/SimpleITK/shank_engine not importable in this environment.",
)
class DeepCoreV1DatasetRegressionTests(unittest.TestCase):
    """Run deep_core_v1 on real subjects and assert detection baselines."""

    @classmethod
    def setUpClass(cls):
        from shank_engine import PipelineRegistry, register_builtin_pipelines

        cls.registry = PipelineRegistry()
        register_builtin_pipelines(cls.registry)

    def _run_subject(self, subject_id, deep_core_config_overrides=None):
        """Run the pipeline on one subject; return (gt, pred, summary_loose)."""
        from eval_seeg_localization import (
            build_detection_context,
            iter_subject_rows,
            load_reference_ground_truth_shanks,
            match_shanks,
            predicted_shanks_from_result,
        )

        rows = iter_subject_rows(DATASET_ROOT, {subject_id})
        self.assertEqual(len(rows), 1, f"Subject {subject_id} not in manifest")
        row = rows[0]

        gt, _ = load_reference_ground_truth_shanks(row)
        config = dict(deep_core_config_overrides or {})
        ctx, _ = build_detection_context(
            row["ct_path"], run_id=f"test_{subject_id}", config=config, extras={}
        )
        result = self.registry.run("deep_core_v1", ctx)
        self.assertEqual(
            result.get("status"), "ok",
            f"Pipeline failed: {result.get('error')}"
        )
        pred = predicted_shanks_from_result(result)

        _, loose = match_shanks(
            gt, pred,
            match_distance_mm=10.0,
            match_angle_deg=25.0,
            match_start_mm=20.0,
        )
        _, strict = match_shanks(
            gt, pred,
            match_distance_mm=4.0,
            match_angle_deg=25.0,
            match_start_mm=15.0,
        )
        return gt, pred, loose, strict

    def test_T1_default_config(self):
        """T1 with default Phase B config.

        Locked at the post-redesign baseline (commit 18c1dde).
        Pre-redesign was 8/12 loose, 3/12 strict; the redesign
        delivered the LPMC/RPMC/LAMC/RHH/RCMN/RAI/LAI fixes that
        were predicted in ``docs/PHASE_B_REDESIGN.md``. The single
        remaining unpaired shank is RAMC, whose predicted trajectory
        is borderline (end_error ~10.5mm — 0.5mm over the loose
        gate); it will be recovered when a Phase C contact-placement
        stage refines the shallow endpoint from the emitted
        ``bolt_ras`` anchor.
        """
        gt, pred, loose, strict = self._run_subject("T1")
        self.assertEqual(len(gt), 12, "T1 GT count drifted")
        self.assertGreaterEqual(len(pred), 10, "Predicted count regressed")
        self.assertLessEqual(len(pred), 16, "Too many predictions")
        n_loose = int(loose.get("matched", 0))
        n_strict = int(strict.get("matched", 0))
        self.assertGreaterEqual(n_loose, 11, f"Loose match regressed: {n_loose}/{len(gt)}")
        self.assertGreaterEqual(n_strict, 5, f"Strict match regressed: {n_strict}/{len(gt)}")

    def test_T22_metal_threshold_1100(self):
        """T22 with metal_threshold=1100: locked post-redesign baseline.

        Pre-redesign was 4/9 loose, 1/9 strict. The 1100 HU threshold
        (vs the default 1900) is needed because T22's electrode metal
        is dimmer than T1's; with 1100 every GT shank gets a clean
        proposal and Phase B matches all 9.
        """
        gt, pred, loose, strict = self._run_subject(
            "T22",
            deep_core_config_overrides={"mask.metal_threshold_hu": 1100.0},
        )
        self.assertEqual(len(gt), 9, "T22 GT count drifted")
        self.assertGreaterEqual(len(pred), 8, "Predicted count regressed")
        self.assertLessEqual(len(pred), 14, "Too many predictions")
        n_loose = int(loose.get("matched", 0))
        n_strict = int(strict.get("matched", 0))
        self.assertGreaterEqual(n_loose, 9, f"Loose match regressed: {n_loose}/{len(gt)}")
        self.assertGreaterEqual(n_strict, 2, f"Strict match regressed: {n_strict}/{len(gt)}")


if __name__ == "__main__":
    unittest.main()
