import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_PATH = REPO_ROOT / "tools" / "analyze_blob_segment_consensus.py"

spec = importlib.util.spec_from_file_location("analyze_blob_segment_consensus", TOOLS_PATH)
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
sys.modules.setdefault("analyze_blob_segment_consensus", module)
spec.loader.exec_module(module)


class BlobSegmentConsensusAnalysisTests(unittest.TestCase):
    def test_tiny_component_stays_blob_not_merged(self):
        kind, score = module._classify_component(
            {"voxel_count": 1, "depth_mean": 6.0},
            {"length_mm": 0.0, "diameter_mm": 0.0, "residual_rms_mm": 0.0, "aspect_ratio": 0.0},
            {"grown_support_ratio": 27.0, "grown_axis_stability_deg": 83.0},
        )
        self.assertEqual(kind, "blob")
        self.assertGreaterEqual(score, 0.0)

    def test_line_like_component_classifies_as_segment(self):
        kind, _score = module._classify_component(
            {"voxel_count": 40, "depth_mean": 10.0},
            {"length_mm": 16.0, "diameter_mm": 1.4, "residual_rms_mm": 0.8, "aspect_ratio": 8.0},
            {"grown_support_ratio": 4.0, "grown_axis_stability_deg": 4.0},
        )
        self.assertEqual(kind, "segment")

    def test_crossing_blob_decomposes_into_multiple_lines(self):
        pts = []
        for x in np.linspace(-12.0, 12.0, 49):
            pts.append((x, 0.0, 0.0))
        for y in np.linspace(-12.0, 12.0, 49):
            pts.append((0.0, y, 0.0))
        pts = np.asarray(pts, dtype=float)
        decomp = module._decompose_merged_blob(
            pts,
            {
                "decomp_min_points": 20,
                "decomp_max_lines": 4,
                "decomp_line_radius_mm": 1.25,
                "decomp_min_fraction_per_line": 0.15,
                "decomp_min_points_per_line": 10,
                "decomp_fit_sample_points": 2000,
                "decomp_max_residual_mm": 0.8,
            },
        )
        self.assertGreaterEqual(len(decomp["sub_lines"]), 2)
        self.assertGreater(decomp["explained_fraction"], 0.80)

    def test_blob_consensus_prefers_stronger_multi_blob_corridor(self):
        priors = [module.ElectrodePrior("test", 4.0, 0.8, 40.0, 10)]
        blobs = [
            module.PrimitiveRecord("S", 1800.0, 0, 1, 0, "blob", 1.0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0, 0.8, 0.1, 4.0, 1.0, 0.0, 4, {}),
            module.PrimitiveRecord("S", 1800.0, 0, 2, 0, "blob", 1.0, (4.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0, 0.8, 0.1, 5.0, 1.0, 0.0, 4, {}),
            module.PrimitiveRecord("S", 1800.0, 0, 3, 0, "blob", 1.0, (8.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0, 0.8, 0.1, 6.0, 1.0, 0.0, 4, {}),
            module.PrimitiveRecord("S", 1800.0, 0, 4, 0, "blob", 0.4, (8.0, 1.2, 0.0), (1.0, 0.0, 0.0), 1.0, 0.8, 0.1, 6.0, 1.0, 0.0, 4, {}),
        ]
        raw_hyps, _support_rows, accepted, assigned = module._build_blob_hypotheses(
            subject_id="S",
            blob_primitives=blobs,
            gt_shanks=[],
            priors=priors,
            gating={"head_distance_map_kji": None},
            ctx={"ras_to_ijk_fn": None, "center_ras": [0.0, 0.0, 0.0]},
            cfg={
                "max_blob_neighbors": 8,
                "hypothesis_min_pair_distance_mm": 3.0,
                "hypothesis_max_pair_distance_mm": 12.0,
                "hypothesis_support_radius_mm": 1.5,
                "hypothesis_support_span_pad_mm": 1.0,
                "hypothesis_min_support_blobs": 3,
                "min_pitch_score": 0.0,
                "entry_depth_good_mm": 12.0,
                "consensus_overlap_drop": 0.55,
                "consensus_angle_deg": 8.0,
                "consensus_line_distance_mm": 2.0,
                "consensus_max_lines": 10,
            },
        )
        self.assertGreater(len(raw_hyps), 1)
        self.assertEqual(len(accepted), 1)
        self.assertTrue(any(int(row["ambiguous"]) == 1 for row in assigned))

    def test_segment_absorption_does_not_create_duplicate_rescue_hypothesis(self):
        hyp = module.HypothesisRecord(
            subject_id="S",
            hypothesis_id="H0001",
            source="blob_pair",
            start_ras=(0.0, 0.0, 0.0),
            end_ras=(20.0, 0.0, 0.0),
            direction_ras=(1.0, 0.0, 0.0),
            score=1.0,
            support_mass=3.0,
            pitch_score=0.8,
            skull_score=0.7,
            support_blob_ids=("B1", "B2", "B3"),
            seed_blob_ids=("B1", "B2"),
            best_model_id="test",
            meta={},
        )
        absorption_rows, rescue_hyps = module._segment_absorption(
            subject_id="S",
            segment_rows=[
                {
                    "blob_key": "SEG1",
                    "source_kind": "segment",
                    "threshold_hu": "1800.0",
                    "x": "10.0",
                    "y": "0.4",
                    "z": "0.0",
                    "axis_x": "1.0",
                    "axis_y": "0.0",
                    "axis_z": "0.0",
                    "length_mm": "8.0",
                    "diameter_mm": "1.0",
                    "score": "0.8",
                }
            ],
            accepted_hypotheses=[hyp],
            priors=[module.ElectrodePrior("test", 4.0, 0.8, 40.0, 10)],
            cfg={"segment_absorb_distance_mm": 3.2, "segment_absorb_angle_deg": 16.0},
        )
        self.assertEqual(len(absorption_rows), 1)
        self.assertEqual(absorption_rows[0]["absorbed"], 1)
        self.assertEqual(len(rescue_hyps), 0)


if __name__ == "__main__":
    unittest.main()
