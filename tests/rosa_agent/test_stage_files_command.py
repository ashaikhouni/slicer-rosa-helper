"""``rosa-agent stage-files`` — frozen-safe file staging for the app's import job.

Replaces a ``python -c`` copy step that broke in the frozen sidecar (there
``sys.executable`` is the packaged binary, understood only as serve/engine) and
was not cross-platform. These tests lock the copy behavior and the guards.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "cli"))

from rosa_agent.commands.stage_files import main as stage_main  # noqa: E402


class StageFilesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.src_a = self.tmp / "src_contacts.tsv"
        self.src_b = self.tmp / "src_traj.tsv"
        self.src_a.write_text("trajectory\tx\ty\tz\nA\t1\t2\t3\n", encoding="utf-8")
        self.src_b.write_text("name\tstart_x\nA\t1\n", encoding="utf-8")
        self.out = self.tmp / "job"

    def test_copies_into_out_dir_with_given_names(self):
        rc = stage_main(["--out-dir", str(self.out),
                         "--copy", str(self.src_a), "contacts.tsv",
                         "--copy", str(self.src_b), "trajectories.tsv"])
        self.assertEqual(rc, 0)
        self.assertTrue((self.out / "contacts.tsv").is_file())
        self.assertTrue((self.out / "trajectories.tsv").is_file())
        self.assertEqual((self.out / "contacts.tsv").read_text(encoding="utf-8"),
                         self.src_a.read_text(encoding="utf-8"))

    def test_creates_missing_out_dir(self):
        deep = self.out / "nested" / "dir"
        rc = stage_main(["--out-dir", str(deep), "--copy", str(self.src_a), "contacts.tsv"])
        self.assertEqual(rc, 0)
        self.assertTrue((deep / "contacts.tsv").is_file())

    def test_rejects_name_with_path_separator(self):
        rc = stage_main(["--out-dir", str(self.out),
                         "--copy", str(self.src_a), "sub/contacts.tsv"])
        self.assertEqual(rc, 2)                      # NAME must be a bare filename

    def test_missing_source_errors(self):
        rc = stage_main(["--out-dir", str(self.out),
                         "--copy", str(self.tmp / "nope.tsv"), "contacts.tsv"])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
