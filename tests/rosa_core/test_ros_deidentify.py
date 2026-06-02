"""Unit tests for rosa_core.ros_deidentify — PHI removal + UID pseudonymisation
+ planned-trajectory CSV. No dataset / SimpleITK needed."""
from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))

try:
    from rosa_core import ros_deidentify as rd
    IMPORTABLE = True
except ImportError:
    IMPORTABLE = False

# A synthetic .ros mirroring the real layout (CRLF; two displays sharing a UID
# between SERIE_UID and VOLUME path; trajectories split across TRAJECTORY/ELLIPS).
_UID1 = "1.2.826.0.1.3680043.2.1290.1111"
_UID2 = "1.2.826.0.1.3680043.2.1290.2222"
_ROS_LINES = [
    "[BEGIN]",
    "[APPLICATION]", "BRAIN",
    "[IMAGERY_NAME]", "T1",
    "[PATIENT_NAME]", "DOE^JOHN^EZEKIEL",
    "[PATIENT_BIRTHDAY]", "19700101",
    "[SERIE_UID]", _UID1,
    "[SERIE_DATE]", "20260101",
    "[VOLUME]", rf"\DICOM\{_UID1}\T1",
    "[IMAGERY_NAME]", "post",
    "[PATIENT_NAME]", "DOE^JOHN^EZEKIEL",
    "[SERIE_UID]", _UID2,
    "[VOLUME]", rf"\DICOM\{_UID2}\post",
    "[TRAJECTORY]", "2",
    "RSPL 0 123 1 -11.4076 36.2768 62.1561 1 -3.1271 31.0647 37.2853 200.0 2.0",
    "[SECURITY_ZONE]", "0.00 10.00 10.00",
    "[ELLIPS]", "0",
    "RCUN 0 456 1 27.0000 -63.8000 30.1000 1 -10.6000 -55.0000 15.3000 200.0 2.0",
    "[SECURITY_ZONE]", "0.00 10.00 10.00",
    "[ELLIPS]", "0",
]
_ROS_TEXT = "\r\n".join(_ROS_LINES) + "\r\n"


@unittest.skipUnless(IMPORTABLE, "rosa_core not importable")
class DeidentifyTextTests(unittest.TestCase):
    def setUp(self):
        self.clean, self.mapping = rd.deidentify_ros_text(_ROS_TEXT, subject_id="SUBJ")

    def test_phi_removed(self):
        self.assertNotIn("DOE^JOHN^EZEKIEL", self.clean)
        self.assertNotIn("19700101", self.clean)   # birthday
        self.assertNotIn("20260101", self.clean)   # serie date
        self.assertNotIn(_UID1, self.clean)
        self.assertNotIn(_UID2, self.clean)

    def test_subject_id_substituted(self):
        self.assertIn("SUBJ", self.clean)
        self.assertEqual(self.mapping["names"], {"DOE^JOHN^EZEKIEL": "SUBJ"})

    def test_uid_pseudonymised_consistently_preserves_linkage(self):
        # Each distinct UID -> one token; SAME token in SERIE_UID value AND its
        # VOLUME path, so display<->series linkage survives.
        tok1 = self.mapping["uids"][_UID1]
        tok2 = self.mapping["uids"][_UID2]
        self.assertNotEqual(tok1, tok2)
        self.assertEqual(self.clean.count(tok1), 2)   # SERIE_UID + VOLUME
        self.assertEqual(self.clean.count(tok2), 2)
        self.assertIn(rf"\DICOM\{tok1}\T1", self.clean)
        self.assertIn(rf"\DICOM\{tok2}\post", self.clean)

    def test_trajectories_and_geometry_preserved(self):
        self.assertIn("RSPL 0 123 1 -11.4076 36.2768 62.1561", self.clean)
        self.assertIn("RCUN 0 456 1 27.0000 -63.8000 30.1000", self.clean)
        self.assertIn("0.00 10.00 10.00", self.clean)
        self.assertIn("T1", self.clean)
        self.assertIn("post", self.clean)

    def test_crlf_preserved(self):
        self.assertIn("\r\n", self.clean)
        self.assertNotIn("\n\n", self.clean.replace("\r\n", "\n"))

    def test_dates_captured_in_keymap(self):
        tags = {t for t, _ in self.mapping["dates"]}
        self.assertEqual(tags, {"PATIENT_BIRTHDAY", "SERIE_DATE"})

    def test_blank_uids_mode(self):
        clean, _ = rd.deidentify_ros_text(_ROS_TEXT, subject_id="SUBJ", pseudonymize_uids=False)
        self.assertNotIn(_UID1, clean)
        self.assertIn("***", clean)


@unittest.skipUnless(IMPORTABLE, "rosa_core not importable")
class TrajectoryCsvTests(unittest.TestCase):
    def test_extract_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            ros = Path(tmp) / "case.ros"
            ros.write_text(_ROS_TEXT, encoding="utf-8")
            trajs = rd.extract_trajectories(ros)
            names = [t["name"] for t in trajs]
            self.assertEqual(names, ["RSPL", "RCUN"])
            self.assertAlmostEqual(trajs[0]["entry"][0], -11.4076, places=3)
            self.assertAlmostEqual(trajs[0]["target"][2], 37.2853, places=3)

            csv_path = rd.write_trajectories_csv(ros)
            rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
            self.assertEqual([r["name"] for r in rows], ["RSPL", "RCUN"])
            self.assertIn("length_mm", rows[0])
            self.assertGreater(float(rows[0]["length_mm"]), 0.0)


@unittest.skipUnless(IMPORTABLE, "rosa_core not importable")
class DeidentifyFileTests(unittest.TestCase):
    def test_file_roundtrip_and_keymap(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            # Parent folder name becomes the default subject id.
            sub = Path(tmp) / "JK"
            sub.mkdir()
            ros = sub / "DOE JOHN 20260101.ros"   # identifying filename
            ros.write_text(_ROS_TEXT, encoding="utf-8")
            out, mapping = rd.deidentify_ros_file(ros)
            self.assertEqual(out.name, "JK.ros")    # output name has no PHI
            clean = out.read_text(encoding="utf-8")
            self.assertNotIn("DOE^JOHN^EZEKIEL", clean)
            self.assertIn("JK", clean)              # subject id from folder
            keymap = out.with_name("JK_deid_keymap.json")
            self.assertTrue(keymap.is_file())
            km = json.loads(keymap.read_text())
            self.assertEqual(km["names"], {"DOE^JOHN^EZEKIEL": "JK"})
            self.assertEqual(len(km["uids"]), 2)


if __name__ == "__main__":
    unittest.main()
