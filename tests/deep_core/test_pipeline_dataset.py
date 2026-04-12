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

        # Loose match: 10 mm tip tolerance, 25° angle, 20 mm shallow tolerance
        _, summary = match_shanks(
            gt, pred,
            match_distance_mm=10.0,
            match_angle_deg=25.0,
            match_start_mm=20.0,
        )
        return gt, pred, summary

    def test_T1_default_config(self):
        """T1 with default deep_core config: locked baseline."""
        gt, pred, summary = self._run_subject("T1")
        self.assertEqual(len(gt), 12, "T1 GT count drifted")
        # Locked baseline: ~15 predicted, 11/12 matched under loose match
        self.assertGreaterEqual(len(pred), 12, "Predicted count regressed")
        self.assertLessEqual(len(pred), 20, "Too many predictions (precision regression)")
        matched = int(summary.get("matched", 0))
        self.assertGreaterEqual(matched, 10, f"Only matched {matched}/{len(gt)} shanks (regression)")

    def test_T22_metal_threshold_1000(self):
        """T22 with metal_threshold=1000: locked baseline."""
        gt, pred, summary = self._run_subject(
            "T22",
            deep_core_config_overrides={"mask.metal_threshold_hu": 1000.0},
        )
        self.assertEqual(len(gt), 9, "T22 GT count drifted")
        self.assertGreaterEqual(len(pred), 8, "Predicted count regressed")
        self.assertLessEqual(len(pred), 15, "Too many predictions")
        matched = int(summary.get("matched", 0))
        # Locked baseline: 7/9 matched with these settings
        self.assertGreaterEqual(matched, 6, f"Only matched {matched}/{len(gt)} shanks")


if __name__ == "__main__":
    unittest.main()
