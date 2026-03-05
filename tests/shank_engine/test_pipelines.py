import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))

from shank_engine import BlobRecord, PipelineRegistry, register_builtin_pipelines  # noqa: E402



def _kji_to_ras_points_identity(idx):
    arr = np.asarray(idx, dtype=float).reshape(-1, 3)
    return np.stack([arr[:, 2], arr[:, 1], arr[:, 0]], axis=1)



def _ras_to_ijk_identity(ras_xyz):
    xyz = np.asarray(ras_xyz, dtype=float).reshape(-1)
    return np.asarray([xyz[0], xyz[1], xyz[2]], dtype=float)


class _OverrideBlobExtractor:
    def extract(self, ctx, state):
        return [
            BlobRecord(
                blob_id=1,
                centroid_ras=(0.0, 0.0, 0.0),
                centroid_kji=(0.0, 0.0, 0.0),
                voxel_count=7,
            )
        ]


class _OverrideGating:
    def compute(self, ctx, state):
        return {
            "candidate_count": 1,
            "metal_in_head_count": 1,
            "depth_kept_count": 1,
            "gating_mask_type": "test",
            "inside_method": "test",
        }


class PipelinesTests(unittest.TestCase):
    def test_all_pipelines_share_run_signature(self):
        reg = PipelineRegistry()
        register_builtin_pipelines(reg)
        for key in reg.keys():
            pipe = reg.create_pipeline(key)
            self.assertTrue(callable(getattr(pipe, "run", None)))

    def test_blob_em_v2_scaffold_schema(self):
        reg = PipelineRegistry()
        register_builtin_pipelines(reg)
        pipe = reg.create_pipeline("blob_em_v2")
        out = pipe.run({"run_id": "r1", "config": {}})
        self.assertEqual(out["pipeline_id"], "blob_em_v2")
        self.assertEqual(out["status"], "ok")
        self.assertIn("diagnostics", out)
        self.assertIn("warnings", out)
        self.assertIn("timing", out["diagnostics"])

    def test_blob_em_v2_component_override(self):
        reg = PipelineRegistry()
        register_builtin_pipelines(reg)
        pipe = reg.create_pipeline("blob_em_v2")
        out = pipe.run(
            {
                "run_id": "r1-override",
                "config": {},
                "arr_kji": np.zeros((2, 2, 2), dtype=np.float32),
                "ijk_kji_to_ras_fn": _kji_to_ras_points_identity,
                "components": {
                    "gating": _OverrideGating(),
                    "blob_extractor": _OverrideBlobExtractor(),
                },
            }
        )
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["diagnostics"]["counts"].get("blob_count_total"), 1)

    def test_blob_ransac_v1_adapter_runs_minimal_case(self):
        reg = PipelineRegistry()
        register_builtin_pipelines(reg)
        pipe = reg.create_pipeline("blob_ransac_v1")

        arr = np.full((24, 24, 24), -1000.0, dtype=np.float32)
        arr[6:18, 6:18, 6:18] = 40.0
        arr[12, 12, 10] = 2500.0

        ctx = {
            "run_id": "r2",
            "arr_kji": arr,
            "spacing_xyz": (1.0, 1.0, 1.0),
            # Regression coverage: numpy arrays must not be evaluated as bool.
            "center_ras": np.asarray([12.0, 12.0, 12.0], dtype=float),
            "ijk_kji_to_ras_fn": _kji_to_ras_points_identity,
            "ras_to_ijk_fn": _ras_to_ijk_identity,
            "config": {
                "threshold": 1800.0,
                "max_lines": 5,
                "min_inliers": 2,
                "candidate_mode": "blob_centroid",
                "head_mask_method": "outside_air",
                "min_metal_depth_mm": 0.0,
            },
        }

        out = pipe.run(ctx)
        self.assertEqual(out["pipeline_id"], "blob_ransac_v1")
        self.assertIn(out["status"], ("ok", "error"))
        self.assertIn("diagnostics", out)


if __name__ == "__main__":
    unittest.main()
