"""Unit tests for the deepmriprep segmentation service wrapper.

No deepmriprep/torch needed — the subprocess is mocked. Locks the discovery
(libomp + MPS-fallback flags on the probe), the embedded runner's validity, and
the not-found path. A real tissue+atlas run is validated E2E elsewhere.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))

try:
    from rosa_detect.services import deepmriprep_seg as dm
    HAVE = True
except Exception:  # noqa: BLE001
    HAVE = False


@unittest.skipUnless(HAVE, "rosa_detect unavailable")
class DeepmriprepSegTests(unittest.TestCase):
    def test_embedded_runner_formats_to_valid_python(self):
        code = dm._RUNNER.format(
            t1=repr("/x/t1.nii.gz"),
            outputs=repr({"p0": "/x/out/p0.nii.gz",
                          "neuromorphometrics": "/x/out/neuromorphometrics.nii.gz"}),
            no_gpu=True)
        compile(code, "<runner>", "exec")   # raises SyntaxError if malformed
        self.assertIn("run_all=False", code)              # only the needed steps run
        self.assertIn("PYTORCH", "".join(dm._PROBE_ENV))  # sanity: fallback flag key present

    def test_find_probe_sets_libomp_and_mps_flags(self):
        with mock.patch.dict(os.environ, {"ROSA_DEEPMRIPREP_PYTHON": sys.executable}):
            ok = mock.Mock(returncode=0)
            with mock.patch.object(dm.subprocess, "run", return_value=ok) as m_run:
                dm.find_deepmriprep()
        env = m_run.call_args.kwargs.get("env") or {}
        self.assertEqual(env.get("KMP_DUPLICATE_LIB_OK"), "TRUE")
        self.assertEqual(env.get("PYTORCH_ENABLE_MPS_FALLBACK"), "1")

    def test_find_none_when_nothing_imports(self):
        with mock.patch.dict(os.environ, {"ROSA_DEEPMRIPREP_PYTHON": ""}, clear=False):
            with mock.patch.object(dm.subprocess, "run", return_value=mock.Mock(returncode=1)):
                self.assertIsNone(dm.find_deepmriprep())

    def test_available_reflects_find(self):
        with mock.patch.object(dm, "find_deepmriprep", return_value="/x/py"):
            self.assertTrue(dm.deepmriprep_available())
        with mock.patch.object(dm, "find_deepmriprep", return_value=None):
            self.assertFalse(dm.deepmriprep_available())

    def test_run_raises_when_not_found(self):
        with mock.patch.object(dm, "find_deepmriprep", return_value=None):
            with self.assertRaises(dm.DeepmriprepNotFound):
                dm.run_deepmriprep("t1.nii.gz", "/out")


if __name__ == "__main__":
    unittest.main()
