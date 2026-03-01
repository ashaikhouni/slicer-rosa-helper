import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib" / "rosa_scene"))

from atlas_assignment_policy import (  # noqa: E402
    build_assignment_row,
    choose_closest_sample,
    collect_provider_samples,
)


class _MockProvider:
    def __init__(self, ready, sample):
        self._ready = bool(ready)
        self._sample = sample
        self.calls = 0

    def is_ready(self):
        return self._ready

    def sample_contact(self, _point_world_ras):
        self.calls += 1
        return self._sample


class AtlasAssignmentPolicyTests(unittest.TestCase):
    def test_collect_provider_samples_respects_readiness(self):
        ready = _MockProvider(
            True,
            {
                "source": "freesurfer",
                "label": "ctx-lh-precuneus",
                "label_value": 100,
                "distance_to_voxel_mm": 1.2,
                "distance_to_centroid_mm": 3.1,
                "native_ras": [1.0, 2.0, 3.0],
            },
        )
        not_ready = _MockProvider(False, {"label": "SHOULD_NOT_BE_USED"})

        samples = collect_provider_samples([0.0, 0.0, 0.0], {"freesurfer": ready, "thomas": not_ready})
        self.assertEqual(ready.calls, 1)
        self.assertEqual(not_ready.calls, 0)
        self.assertEqual(samples["freesurfer"]["label"], "ctx-lh-precuneus")
        self.assertIsNone(samples["thomas"])

    def test_choose_closest_sample_prefers_smallest_distance(self):
        source, sample = choose_closest_sample(
            {
                "freesurfer": {"label": "A", "distance_to_voxel_mm": 1.1, "label_value": 1},
                "thomas": {"label": "B", "distance_to_voxel_mm": 2.3, "label_value": 2},
                "wm": {"label": "", "distance_to_voxel_mm": 0.2, "label_value": 3},
            }
        )
        self.assertEqual(source, "freesurfer")
        self.assertEqual(sample["label"], "A")

    def test_build_assignment_row_schema_is_stable(self):
        contact = {"trajectory": "R01", "index": 3, "label": "R013"}
        point_ras = [10.0, -2.5, 5.5]
        samples = {
            "freesurfer": {
                "source": "freesurfer",
                "label": "ctx-rh-insula",
                "label_value": 2035,
                "distance_to_voxel_mm": 0.8,
                "distance_to_centroid_mm": 6.4,
                "native_ras": [11.0, -2.0, 5.0],
            }
        }
        closest_source, closest = choose_closest_sample(samples)
        row = build_assignment_row(contact, point_ras, samples, closest_source, closest)

        expected_keys = {
            "trajectory",
            "contact_label",
            "contact_index",
            "contact_ras",
            "closest_source",
            "closest_label",
            "closest_label_value",
            "closest_distance_to_voxel_mm",
            "closest_distance_to_centroid_mm",
            "primary_source",
            "primary_label",
            "primary_label_value",
            "primary_distance_to_voxel_mm",
            "primary_distance_to_centroid_mm",
            "thomas_label",
            "thomas_label_value",
            "thomas_distance_to_voxel_mm",
            "thomas_distance_to_centroid_mm",
            "freesurfer_label",
            "freesurfer_label_value",
            "freesurfer_distance_to_voxel_mm",
            "freesurfer_distance_to_centroid_mm",
            "wm_label",
            "wm_label_value",
            "wm_distance_to_voxel_mm",
            "wm_distance_to_centroid_mm",
            "thomas_native_ras",
            "freesurfer_native_ras",
            "wm_native_ras",
        }
        self.assertEqual(set(row.keys()), expected_keys)
        self.assertEqual(row["closest_source"], "freesurfer")
        self.assertEqual(row["primary_source"], row["closest_source"])
        self.assertEqual(row["freesurfer_label"], "ctx-rh-insula")
        self.assertEqual(row["thomas_label"], "")
        self.assertEqual(row["wm_label"], "")


if __name__ == "__main__":
    unittest.main()
