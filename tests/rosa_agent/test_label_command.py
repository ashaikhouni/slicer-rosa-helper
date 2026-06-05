"""label command: an explicitly-requested atlas that fails to load must ERROR
(exit non-zero), not silently write an empty labels file + exit 0 — which is
dangerous in batch runs (e.g. a typo'd FreeSurfer labelmap path)."""

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
        from rosa_agent.commands import label  # noqa: F401
        return True
    except ImportError:
        return False


def _write_contacts(tmp: Path) -> Path:
    p = tmp / "contacts.tsv"
    p.write_text(
        "trajectory\tlabel\tcontact_index\tx\ty\tz\tpeak_detected\telectrode_model\n"
        "L1\tL1_1\t1\t1.0\t2.0\t3.0\t1\tDIXI-8AM\n"
    )
    return p


@unittest.skipUnless(_try_imports(), "rosa_agent.commands.label not importable")
class LabelMissingAtlasTests(unittest.TestCase):
    def _run(self, argv):
        from rosa_agent.commands.label import main as label_main
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            rc = label_main(argv)
        return rc, buf.getvalue()

    def test_missing_freesurfer_atlas_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            contacts = _write_contacts(tmp)
            out = tmp / "labels.tsv"
            rc, err = self._run([
                str(contacts), "--out", str(out),
                "--freesurfer", str(tmp / "does_not_exist.nii.gz"),
            ])
            self.assertEqual(rc, 2)            # explicit-but-missing atlas → error
            self.assertIn("error:", err.lower())
            self.assertIn("--freesurfer", err)

    def test_no_atlas_requested_writes_empty_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            contacts = _write_contacts(tmp)
            out = tmp / "labels.tsv"
            rc, _err = self._run([str(contacts), "--out", str(out)])
            self.assertEqual(rc, 0)            # nothing requested → empty output, exit 0
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
