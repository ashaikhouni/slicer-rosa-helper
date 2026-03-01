import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))

from rosa_core.ros_parser import extract_tokens, parse_ros_file, parse_ros_text  # noqa: E402


ROS_SAMPLE = """
[TRdicomRdisplay]
1 0 0 0
0 1 0 0
0 0 1 0
0 0 0 1
[VOLUME]
/DICOM/1.2.3.4/NCAxT1
[TRdicomRdisplay]
0.98 0.01 0.0 5.0
0.00 1.00 0.0 -2.0
0.00 0.00 1.0 1.5
0.00 0.00 0.0 1.0
[VOLUME]
\\DICOM\\9.8.7.6\\post
[IMAGERY_NAME]
NCAxT1
[SERIE_UID]
1.2.3.4
[IMAGERY_3DREF]
0
[IMAGERY_NAME]
post
[SERIE_UID]
9.8.7.6
[IMAGERY_3DREF]
1
[TRAJECTORY]
ignored
RHH 0 0 0 10 11 12 0 20 21 22
[ELLIPS]
ignored
LHH 0 0 0 1 2 3 0 4 5 6
[END]
"""


class RosParserTests(unittest.TestCase):
    def test_extract_tokens(self):
        tokens = extract_tokens(ROS_SAMPLE)
        self.assertGreater(len(tokens), 0)
        self.assertEqual(tokens[0]["token"], "TRdicomRdisplay")
        self.assertEqual(tokens[1]["token"], "VOLUME")

    def test_parse_ros_text(self):
        parsed = parse_ros_text(ROS_SAMPLE)
        self.assertEqual(len(parsed["displays"]), 2)
        self.assertEqual(parsed["displays"][0]["volume"], "NCAxT1")
        self.assertEqual(parsed["displays"][0]["imagery_name"], "NCAxT1")
        self.assertEqual(parsed["displays"][0]["serie_uid"], "1.2.3.4")
        self.assertEqual(parsed["displays"][0]["imagery_3dref"], 0)
        self.assertEqual(parsed["displays"][1]["volume"], "post")
        # Backslashes are normalized by parser.
        self.assertIn("/DICOM/9.8.7.6/post", parsed["displays"][1]["volume_path"])

        self.assertEqual(len(parsed["trajectories"]), 2)
        self.assertEqual(parsed["trajectories"][0]["name"], "RHH")
        self.assertEqual(parsed["trajectories"][0]["start"], [10.0, 11.0, 12.0])
        self.assertEqual(parsed["trajectories"][0]["end"], [20.0, 21.0, 22.0])

    def test_parse_ros_file(self):
        with tempfile.TemporaryDirectory() as td:
            ros_path = Path(td) / "sample.ros"
            ros_path.write_text(ROS_SAMPLE, encoding="utf-8")
            parsed = parse_ros_file(ros_path)
            self.assertEqual(parsed["ros_path"], str(ros_path))
            self.assertEqual(len(parsed["displays"]), 2)


if __name__ == "__main__":
    unittest.main()
