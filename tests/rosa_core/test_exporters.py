import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))

from rosa_core.exporters import (  # noqa: E402
    build_fcsv_rows,
    build_markups_document,
    build_markups_lines,
    save_fcsv,
    save_markups_json,
)


class ExportersTests(unittest.TestCase):
    def setUp(self):
        self.traj = [{"name": "RHH", "start": [0, 1, 2], "end": [10, 11, 12]}]

    def test_build_markups_lines_in_ras(self):
        markups = build_markups_lines(self.traj, to_ras=True)
        cp0 = markups[0]["controlPoints"][0]["position"]
        cp1 = markups[0]["controlPoints"][1]["position"]
        self.assertEqual(cp0, [0, -1, 2])
        self.assertEqual(cp1, [-10, -11, 12])
        self.assertEqual(markups[0]["coordinateSystem"], "RAS")

    def test_build_markups_lines_with_display_to_dicom(self):
        affine = [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        markups = build_markups_lines(self.traj, to_ras=False, display_to_dicom=affine)
        cp0 = markups[0]["controlPoints"][0]["position"]
        self.assertEqual(cp0, [1.0, 3.0, 5.0])
        self.assertEqual(markups[0]["coordinateSystem"], "LPS")

    def test_build_fcsv_rows(self):
        rows = build_fcsv_rows(self.traj, to_ras=False, same_label_pair=False)
        self.assertEqual(rows[0]["label"], "RHH_entry")
        self.assertEqual(rows[1]["label"], "RHH_target")

        rows_same = build_fcsv_rows(self.traj, to_ras=False, same_label_pair=True)
        self.assertEqual(rows_same[0]["label"], "RHH")
        self.assertEqual(rows_same[1]["label"], "RHH")

    def test_save_fcsv_and_markups_json(self):
        with tempfile.TemporaryDirectory() as td:
            fcsv_path = Path(td) / "traj.fcsv"
            json_path = Path(td) / "traj.mrk.json"

            rows = build_fcsv_rows(self.traj, to_ras=False)
            save_fcsv(fcsv_path, rows)
            fcsv_text = fcsv_path.read_text(encoding="utf-8")
            self.assertIn("# Markups fiducial file version = 4.11", fcsv_text)
            self.assertIn("RHH_entry", fcsv_text)

            markups = build_markups_lines(self.traj, to_ras=False)
            save_markups_json(json_path, markups)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["@schema"].split("/")[-1], "markups-schema-v1.0.0.json")

    def test_build_markups_document(self):
        markups = build_markups_lines(self.traj, to_ras=False)
        doc = build_markups_document(markups)
        self.assertIn("markups", doc)
        self.assertEqual(len(doc["markups"]), 1)


if __name__ == "__main__":
    unittest.main()
