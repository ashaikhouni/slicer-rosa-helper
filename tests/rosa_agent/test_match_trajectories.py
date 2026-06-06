"""Input-guard tests for `rosa-agent match-trajectories`.

These exercise the cheap pre-detection guards (no CT read / no detection): a
missing CT, a missing plan file, and a malformed plan all exit 2 with a clean
message rather than leaking a traceback past the heavy pipeline boundary. The
full detect+match path needs a real CT and is covered by the cross_volume_match
algorithm test + manual runs.
"""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "cli"))
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))


def _try_imports() -> bool:
    try:
        from rosa_agent.commands import match_trajectories  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(_try_imports(), "rosa_agent.commands.match_trajectories not importable")
class MatchTrajectoriesGuardTests(unittest.TestCase):
    def _run(self, argv):
        from rosa_agent.commands.match_trajectories import main as mt_main
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            rc = mt_main(argv)
        return rc, buf.getvalue()

    def test_missing_ct(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            plan = tmp / "plan.tsv"
            plan.write_text("name\tstart_x\tstart_y\tstart_z\tend_x\tend_y\tend_z\n"
                            "A\t0\t0\t0\t0\t0\t40\n")
            rc, err = self._run([
                "--plan", str(plan), "--ct", str(tmp / "nope.nii.gz"),
                "--output", str(tmp / "out"),
            ])
            self.assertEqual(rc, 2)
            self.assertIn("ct not found", err.lower())

    def test_missing_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ct = tmp / "ct.nii.gz"
            ct.write_bytes(b"")  # exists() passes; plan check fails first
            rc, err = self._run([
                "--plan", str(tmp / "nope.tsv"), "--ct", str(ct),
                "--output", str(tmp / "out"),
            ])
            self.assertEqual(rc, 2)
            self.assertIn("plan file not found", err.lower())

    def test_malformed_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ct = tmp / "ct.nii.gz"
            ct.write_bytes(b"")
            plan = tmp / "plan.tsv"
            plan.write_text("foo\tbar\n1\t2\n")  # header read_seeds_tsv rejects
            rc, err = self._run([
                "--plan", str(plan), "--ct", str(ct),
                "--output", str(tmp / "out"),
            ])
            self.assertEqual(rc, 2)
            self.assertIn("could not read plan", err.lower())


if __name__ == "__main__":
    unittest.main()
