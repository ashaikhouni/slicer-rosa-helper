"""Tests for typed stage output dataclasses."""

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))
sys.path.insert(0, str(REPO_ROOT / "PostopCTLocalization"))

from postop_ct_localization.deep_core_stages import (
    MaskStageOutput,
    ProposalStageOutput,
    SupportStageOutput,
)


class TestMaskStageOutput(unittest.TestCase):
    def _make_mask(self):
        shape = (5, 5, 5)
        return MaskStageOutput(
            hull_mask_kji=np.ones(shape, dtype=bool),
            deep_core_mask_kji=np.zeros(shape, dtype=bool),
            metal_mask_kji=np.zeros(shape, dtype=bool),
            metal_grown_mask_kji=np.zeros(shape, dtype=bool),
            deep_seed_raw_mask_kji=np.zeros(shape, dtype=bool),
            deep_seed_mask_kji=np.zeros(shape, dtype=bool),
            head_distance_map_kji=np.zeros(shape, dtype=np.float32),
            smoothed_hull_kji=np.zeros(shape, dtype=np.float32),
            stats={"hull_voxels": 125},
        )

    def test_to_payload_keys(self):
        mask = self._make_mask()
        payload = mask.to_payload()
        expected_keys = {
            "hull_mask_kji", "deep_core_mask_kji", "metal_mask_kji",
            "metal_grown_mask_kji", "deep_seed_raw_mask_kji",
            "deep_seed_mask_kji", "head_distance_map_kji",
            "smoothed_hull_kji", "stats",
        }
        self.assertEqual(set(payload.keys()), expected_keys)

    def test_stats_in_payload(self):
        mask = self._make_mask()
        payload = mask.to_payload()
        self.assertEqual(payload["stats"]["hull_voxels"], 125)


class TestSupportStageOutput(unittest.TestCase):
    def _make_support(self):
        shape = (5, 5, 5)
        mask = MaskStageOutput(
            hull_mask_kji=np.ones(shape, dtype=bool),
            deep_core_mask_kji=np.zeros(shape, dtype=bool),
            metal_mask_kji=np.zeros(shape, dtype=bool),
            metal_grown_mask_kji=np.zeros(shape, dtype=bool),
            deep_seed_raw_mask_kji=np.zeros(shape, dtype=bool),
            deep_seed_mask_kji=np.zeros(shape, dtype=bool),
            head_distance_map_kji=np.ones(shape, dtype=np.float32) * 20.0,
            smoothed_hull_kji=np.zeros(shape, dtype=np.float32),
            stats={},
        )
        return SupportStageOutput(
            mask=mask,
            support_atoms=[{"atom_id": 1, "parent_blob_id": 1}],
            blob_sample_points_ras=np.array([[1, 2, 3]], dtype=float),
            blob_sample_blob_ids=np.array([1], dtype=np.int32),
            blob_sample_atom_ids=np.array([1], dtype=np.int32),
            blob_axes_ras_by_id={1: [0, 0, 1]},
            blob_elongation_by_id={1: 5.0},
            blob_class_by_id={1: "line_blob"},
            stats={"deep_seed_atom_count": 1},
        )

    def test_head_distance_from_mask(self):
        support = self._make_support()
        np.testing.assert_allclose(support.head_distance_map_kji, 20.0)

    def test_metal_grown_from_mask(self):
        support = self._make_support()
        self.assertFalse(support.metal_grown_mask_kji.any())

    def test_to_payload_includes_mask_keys(self):
        support = self._make_support()
        payload = support.to_payload()
        # Should have mask keys
        self.assertIn("hull_mask_kji", payload)
        self.assertIn("head_distance_map_kji", payload)
        # And support keys
        self.assertIn("support_atoms", payload)
        self.assertIn("blob_sample_points_ras", payload)
        self.assertIn("blob_axes_ras_by_id", payload)

    def test_to_payload_support_atoms(self):
        support = self._make_support()
        payload = support.to_payload()
        self.assertEqual(len(payload["support_atoms"]), 1)
        self.assertEqual(payload["support_atoms"][0]["atom_id"], 1)


class TestProposalStageOutput(unittest.TestCase):
    def test_to_payload(self):
        shape = (3, 3, 3)
        mask = MaskStageOutput(
            hull_mask_kji=np.ones(shape, dtype=bool),
            deep_core_mask_kji=np.zeros(shape, dtype=bool),
            metal_mask_kji=np.zeros(shape, dtype=bool),
            metal_grown_mask_kji=np.zeros(shape, dtype=bool),
            deep_seed_raw_mask_kji=np.zeros(shape, dtype=bool),
            deep_seed_mask_kji=np.zeros(shape, dtype=bool),
            head_distance_map_kji=np.zeros(shape, dtype=np.float32),
            smoothed_hull_kji=np.zeros(shape, dtype=np.float32),
            stats={},
        )
        support = SupportStageOutput(
            mask=mask,
            support_atoms=[],
            blob_sample_points_ras=np.empty((0, 3), dtype=float),
            blob_sample_blob_ids=np.array([], dtype=np.int32),
            blob_sample_atom_ids=np.array([], dtype=np.int32),
            blob_axes_ras_by_id={},
            blob_elongation_by_id={},
            blob_class_by_id={},
            stats={},
        )
        proposals = [
            {"start_ras": [0, 0, 0], "end_ras": [10, 0, 0], "proposal_family": "graph"},
        ]
        result = ProposalStageOutput(
            support=support,
            proposals=proposals,
            candidate_count=5,
            token_count=20,
        )
        payload = result.to_payload()
        self.assertEqual(len(payload["proposals"]), 1)
        self.assertEqual(payload["candidate_count"], 5)
        self.assertEqual(payload["token_count"], 20)


if __name__ == "__main__":
    unittest.main()
