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
    "[PATIENT_NAME]", "DOE^JANE^Q",
    "[PATIENT_BIRTHDAY]", "19700101",
    "[SERIE_UID]", _UID1,
    "[SERIE_DATE]", "20260101",
    "[VOLUME]", rf"\DICOM\{_UID1}\T1",
    "[IMAGERY_NAME]", "post",
    "[PATIENT_NAME]", "DOE^JANE^Q",
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
        self.assertNotIn("DOE^JANE^Q", self.clean)
        self.assertNotIn("19700101", self.clean)   # birthday
        self.assertNotIn("20260101", self.clean)   # serie date
        self.assertNotIn(_UID1, self.clean)
        self.assertNotIn(_UID2, self.clean)

    def test_subject_id_substituted(self):
        self.assertIn("SUBJ", self.clean)
        self.assertEqual(self.mapping["names"], {"DOE^JANE^Q": "SUBJ"})

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
            sub = Path(tmp) / "S99"
            sub.mkdir()
            ros = sub / "TEST CASE 20260101.ros"   # identifying-style filename
            ros.write_text(_ROS_TEXT, encoding="utf-8")
            out, mapping = rd.deidentify_ros_file(ros)
            self.assertEqual(out.name, "S99.ros")   # output name has no PHI
            self.assertIn(b"\r\n", out.read_bytes())  # CRLF preserved, not collapsed to LF
            clean = out.read_text(encoding="utf-8")
            self.assertNotIn("DOE^JANE^Q", clean)
            self.assertIn("S99", clean)             # subject id from folder
            keymap = out.with_name("S99_deid_keymap.json")
            self.assertTrue(keymap.is_file())
            km = json.loads(keymap.read_text())
            self.assertEqual(km["names"], {"DOE^JANE^Q": "S99"})
            self.assertEqual(len(km["uids"]), 2)


@unittest.skipUnless(IMPORTABLE, "rosa_core not importable")
class DeidentifyFolderTests(unittest.TestCase):
    def _make_case(self, root: Path) -> Path:
        case = root / "S07"
        (case / "DICOM" / _UID1).mkdir(parents=True)
        (case / "DICOM" / _UID2).mkdir(parents=True)
        (case / "ORIGINAL.ros").write_text(_ROS_TEXT, encoding="utf-8")
        # images (clean) + a raw-DICOM zip (PHI) + a screenshot (PHI risk)
        for uid, nm in ((_UID1, "T1"), (_UID2, "post")):
            d = case / "DICOM" / uid
            (d / f"{nm}.img").write_bytes(b"\x00" * 16)
            (d / f"{nm}.hdr").write_bytes(b"\x00" * 348)
            (d / "DICOMFiles.zip").write_bytes(b"PK\x03\x04rawdicom")
        (case / "screenshot.png").write_bytes(b"\x89PNG")
        (case / "qc").mkdir()
        (case / "qc" / "fig.jpg").write_bytes(b"\xff\xd8")
        return case

    def test_folder_deid(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            case = self._make_case(tmp)
            out = tmp / "S07_clean"
            out_dir, mapping, summary = rd.deidentify_ros_folder(case, out)

            # clean .ros present + PHI-free + still parses
            ros_out = out_dir / "S07.ros"
            self.assertTrue(ros_out.is_file())
            self.assertIn(b"\r\n", ros_out.read_bytes())   # CRLF preserved
            clean = ros_out.read_text()
            self.assertNotIn("DOE^JANE^Q", clean)
            self.assertNotIn(_UID1, clean)

            # DICOM dirs renamed to the .ros's UID tokens; linkage holds
            tok1, tok2 = mapping["uids"][_UID1], mapping["uids"][_UID2]
            self.assertTrue((out_dir / "DICOM" / tok1 / "T1.img").is_file())
            self.assertTrue((out_dir / "DICOM" / tok2 / "post.hdr").is_file())
            self.assertIn(rf"\DICOM\{tok2}\post", clean)
            self.assertFalse((out_dir / "DICOM" / _UID1).exists())  # original UID name gone

            # zips dropped, screenshots not copied
            self.assertEqual(list(out_dir.rglob("*.zip")), [])
            self.assertEqual(list(out_dir.rglob("*.png")), [])
            self.assertEqual(list(out_dir.rglob("*.jpg")), [])
            self.assertEqual(summary["zips_dropped"], 2)
            self.assertGreaterEqual(summary["dirs_renamed"], 2)

            # trajectory CSV inside the shareable dir; keymap OUTSIDE it
            self.assertTrue((out_dir / "S07_trajectories.csv").is_file())
            self.assertEqual(list(out_dir.rglob("*keymap*")), [])
            self.assertTrue((out_dir.with_name(out_dir.name + "_deid_keymap.json")).is_file())

            # original untouched
            self.assertTrue((case / "DICOM" / _UID1 / "DICOMFiles.zip").is_file())


if __name__ == "__main__":
    unittest.main()
