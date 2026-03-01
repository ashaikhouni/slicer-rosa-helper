import copy
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))

from rosa_core.electrode_models import (  # noqa: E402
    default_electrode_library_path,
    load_electrode_library,
    model_map,
    validate_electrode_library,
)


class ElectrodeModelsTests(unittest.TestCase):
    def test_load_default_library(self):
        path = default_electrode_library_path()
        self.assertTrue(path.exists())
        data = load_electrode_library(path)
        self.assertIn("models", data)
        self.assertGreater(len(data["models"]), 0)

    def test_model_map(self):
        data = {"models": [{"id": "A", "contact_count": 1, "contact_center_offsets_from_tip_mm": [0.0], "type": "AM", "contact_length_mm": 2.0, "diameter_mm": 0.8, "total_exploration_length_mm": 10.0}]}
        m = model_map(data)
        self.assertIn("A", m)
        self.assertEqual(m["A"]["type"], "AM")

    def test_validate_missing_fields_raises(self):
        bad = {"models": [{"id": "A"}]}
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            validate_electrode_library(bad)

    def test_validate_duplicate_id_raises(self):
        base = load_electrode_library(default_electrode_library_path())
        bad = copy.deepcopy(base)
        bad["models"] = [bad["models"][0], copy.deepcopy(bad["models"][0])]
        with self.assertRaisesRegex(ValueError, "Duplicate model id"):
            validate_electrode_library(bad)

    def test_validate_contact_count_offsets_mismatch_raises(self):
        base = load_electrode_library(default_electrode_library_path())
        bad = copy.deepcopy(base)
        bad["models"][0]["contact_center_offsets_from_tip_mm"] = [0.0]
        with self.assertRaisesRegex(ValueError, "contact_count"):
            validate_electrode_library(bad)

    def test_validate_non_increasing_offsets_raise(self):
        base = load_electrode_library(default_electrode_library_path())
        bad = copy.deepcopy(base)
        count = int(bad["models"][0]["contact_count"])
        bad["models"][0]["contact_center_offsets_from_tip_mm"] = [0.0] * count
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            validate_electrode_library(bad)


if __name__ == "__main__":
    unittest.main()
