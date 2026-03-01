import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))

from rosa_core.contacts import (  # noqa: E402
    build_assignment_template,
    build_contacts_markups,
    contacts_to_fcsv_rows,
    generate_contacts,
    load_assignments,
)


class ContactsTests(unittest.TestCase):
    def setUp(self):
        self.trajectories = [{"name": "RHH", "start": [0, 0, 0], "end": [10, 0, 0]}]
        self.models_by_id = {
            "DIXI-2": {
                "id": "DIXI-2",
                "contact_center_offsets_from_tip_mm": [0.0, 3.5],
            }
        }

    def test_build_assignment_template(self):
        template = build_assignment_template(self.trajectories, default_model_id="DIXI-2")
        self.assertEqual(template["schema_version"], "1.0")
        self.assertEqual(template["assignments"][0]["trajectory"], "RHH")
        self.assertEqual(template["assignments"][0]["model_id"], "DIXI-2")
        self.assertEqual(template["assignments"][0]["tip_at"], "target")

    def test_load_assignments_supports_both_formats(self):
        with tempfile.TemporaryDirectory() as td:
            standard = Path(td) / "assignments_standard.json"
            shorthand = Path(td) / "assignments_shorthand.json"
            standard.write_text(
                json.dumps({"schema_version": "1.0", "assignments": [{"trajectory": "RHH", "model_id": "DIXI-2"}]}),
                encoding="utf-8",
            )
            shorthand.write_text(json.dumps({"RHH": "DIXI-2"}), encoding="utf-8")

            a_standard = load_assignments(str(standard))
            a_shorthand = load_assignments(str(shorthand))

            self.assertEqual(a_standard["assignments"][0]["model_id"], "DIXI-2")
            self.assertEqual(a_shorthand["assignments"][0]["trajectory"], "RHH")
            self.assertEqual(a_shorthand["assignments"][0]["tip_at"], "target")

    def test_generate_contacts_in_target_tip_mode(self):
        assignments = {
            "assignments": [
                {
                    "trajectory": "RHH",
                    "model_id": "DIXI-2",
                    "tip_at": "target",
                    "tip_shift_mm": 0.0,
                    "xyz_offset_mm": [0.0, 0.0, 0.0],
                }
            ]
        }
        contacts = generate_contacts(self.trajectories, self.models_by_id, assignments)
        self.assertEqual(len(contacts), 2)
        self.assertEqual(contacts[0]["label"], "RHH1")
        self.assertEqual(contacts[0]["position_lps"], [10.0, 0.0, 0.0])
        self.assertEqual(contacts[1]["position_lps"], [6.5, 0.0, 0.0])

    def test_generate_contacts_with_tip_shift_and_xyz_offset(self):
        assignments = {
            "assignments": [
                {
                    "trajectory": "RHH",
                    "model_id": "DIXI-2",
                    "tip_at": "target",
                    "tip_shift_mm": 2.0,
                    "xyz_offset_mm": [0.0, 1.0, 0.0],
                }
            ]
        }
        contacts = generate_contacts(self.trajectories, self.models_by_id, assignments)
        self.assertEqual(contacts[0]["position_lps"], [8.0, 1.0, 0.0])
        self.assertEqual(contacts[1]["position_lps"], [4.5, 1.0, 0.0])

    def test_generate_contacts_raises_on_missing_references(self):
        assignments = {"assignments": [{"trajectory": "MISSING", "model_id": "DIXI-2"}]}
        with self.assertRaisesRegex(ValueError, "Missing references"):
            generate_contacts(self.trajectories, self.models_by_id, assignments)

    def test_contacts_to_fcsv_rows_and_markups(self):
        contacts = [
            {
                "trajectory": "RHH",
                "model_id": "DIXI-2",
                "index": 1,
                "label": "RHH1",
                "position_lps": [10.0, 2.0, 3.0],
            }
        ]
        rows = contacts_to_fcsv_rows(contacts, to_ras=True)
        self.assertEqual(rows[0]["xyz"], [-10.0, -2.0, 3.0])

        markups = build_contacts_markups(contacts, to_ras=False, node_name="node")
        self.assertEqual(markups["markups"][0]["coordinateSystem"], "LPS")
        self.assertEqual(markups["markups"][0]["controlPoints"][0]["label"], "RHH1")


if __name__ == "__main__":
    unittest.main()
