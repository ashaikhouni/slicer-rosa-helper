"""End-to-end regression for the bolt-first deep_core_v2 pipeline.

v2 drops Phase A proposal generation and the support/annulus stages.
Every trajectory comes from a RANSAC bolt candidate whose axis is
extended inward via a loose-threshold metal walk to find the deepest
contact, and whose shallow endpoint is anchored to the bolt center
via ``model_fit.bolt_endpoint_offset_mm``.

This is a **proof-of-concept** pipeline. It does not match v1+bolts'
full recall because for dim-electrode subjects (like T22) the walk
from the bolt can't bridge the bolt-to-contact gap on every shank.
Baselines here reflect the current ceiling of the bolt-only path:

  - T1:  loose >= 10/12
  - T22: loose >=  5/9

Users who need full recall should use v1 with
``model_fit.use_bolt_detection=True`` (tested in
``test_pipeline_dataset.py``).
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
class DeepCoreV2DatasetRegressionTests(unittest.TestCase):
    """Run deep_core_v2 on real subjects and assert bolt-first baselines."""

    @classmethod
    def setUpClass(cls):
        from shank_engine import PipelineRegistry, register_builtin_pipelines

        cls.registry = PipelineRegistry()
        register_builtin_pipelines(cls.registry)

    def _run_subject(self, subject_id, deep_core_config_overrides=None):
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
            row["ct_path"], run_id=f"v2_{subject_id}", config=config, extras={}
        )
        result = self.registry.run("deep_core_v2", ctx)
        self.assertEqual(
            result.get("status"), "ok",
            f"Pipeline failed: {result.get('error')}",
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

    def test_T1_bolt_first(self):
        """T1 with deep_core_v2. Baseline: 10/12 loose."""
        gt, pred, loose, strict = self._run_subject("T1")
        self.assertEqual(len(gt), 12, "T1 GT count drifted")
        n_loose = int(loose.get("matched", 0))
        self.assertGreaterEqual(
            n_loose, 10, f"v2 loose match regressed: {n_loose}/{len(gt)}"
        )

    def test_T22_bolt_first(self):
        """T22 with deep_core_v2. Baseline: 5/9 loose.

        The 4 shanks v2 misses on T22 all have dim electrode contacts
        that don't survive the loose metal mask at 400 HU with a 15 mm
        gap tolerance. v1+bolts recovers them via Phase A's atom
        proposals; v2 drops Phase A by design.
        """
        gt, pred, loose, strict = self._run_subject("T22")
        self.assertEqual(len(gt), 9, "T22 GT count drifted")
        n_loose = int(loose.get("matched", 0))
        self.assertGreaterEqual(
            n_loose, 5, f"v2 loose match regressed: {n_loose}/{len(gt)}"
        )


if __name__ == "__main__":
    unittest.main()
