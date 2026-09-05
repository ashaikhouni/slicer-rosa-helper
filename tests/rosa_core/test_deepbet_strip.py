"""Unit tests for the deepbet brain-extraction service wrapper.

These do NOT require deepbet / torch to be installed — they exercise the
interpreter-discovery and not-found paths (the subprocess is mocked), which is
all the CLI wiring depends on. A real strip is validated end-to-end elsewhere.
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
    from rosa_detect.services import deepbet_strip as ds
    HAVE = True
except Exception:  # noqa: BLE001
    HAVE = False


@unittest.skipUnless(HAVE, "rosa_detect unavailable")
class DeepbetStripTests(unittest.TestCase):
    def setUp(self):
        # These tests exercise external-interpreter discovery independently of
        # the bundled ONNX fallback installed on the developer's machine.
        fallback = mock.patch("rosa_detect.services.deepbet_onnx.deepbet_onnx_available", return_value=False)
        fallback.start()
        self.addCleanup(fallback.stop)

    def test_bundled_fallback_runs_when_external_backend_is_missing(self):
        with mock.patch.object(ds, "find_deepbet", return_value=None), \
             mock.patch("rosa_detect.services.deepbet_onnx.deepbet_onnx_available", return_value=True), \
             mock.patch("rosa_detect.services.deepbet_onnx.run_deepbet_onnx", return_value=Path("out.nii.gz")) as run:
            self.assertEqual(ds.run_deepbet("in.nii.gz", "out.nii.gz"), Path("out.nii.gz"))
            run.assert_called_once()

    def test_find_deepbet_none_when_nothing_imports(self):
        # Neutralise the env var (empty → skipped), make every probe "fail to
        # import" → no candidate qualifies → None.
        with mock.patch.dict(os.environ, {"ROSA_DEEPBET_PYTHON": ""}, clear=False):
            fail = mock.Mock(returncode=1)
            with mock.patch.object(ds.subprocess, "run", return_value=fail):
                self.assertIsNone(ds.find_deepbet())

    def test_find_deepbet_returns_candidate_that_imports(self):
        # An env-var interpreter that "imports deepbet" (returncode 0) is chosen.
        with mock.patch.dict(os.environ, {"ROSA_DEEPBET_PYTHON": sys.executable}):
            ok = mock.Mock(returncode=0)
            with mock.patch.object(ds.subprocess, "run", return_value=ok):
                py = ds.find_deepbet()
        self.assertIsNotNone(py)

    def test_find_deepbet_probe_sets_libomp_flag(self):
        # The probe must pass KMP_DUPLICATE_LIB_OK=TRUE so a deepbet that shares
        # an env with an MKL numpy (torch + numpy) doesn't abort on OMP Error #15
        # and get mis-reported as unavailable.
        with mock.patch.dict(os.environ, {"ROSA_DEEPBET_PYTHON": sys.executable}):
            ok = mock.Mock(returncode=0)
            with mock.patch.object(ds.subprocess, "run", return_value=ok) as m_run:
                ds.find_deepbet()
        env = m_run.call_args.kwargs.get("env") or {}
        self.assertEqual(env.get("KMP_DUPLICATE_LIB_OK"), "TRUE")

    def test_run_deepbet_raises_when_not_found(self):
        with mock.patch.object(ds, "find_deepbet", return_value=None):
            with self.assertRaises(ds.DeepbetNotFound):
                ds.run_deepbet("in.nii.gz", "out.nii.gz")

    def test_available_reflects_find(self):
        with mock.patch.object(ds, "find_deepbet", return_value="/x/py"):
            self.assertTrue(ds.deepbet_available())
        with mock.patch.object(ds, "find_deepbet", return_value=None):
            self.assertFalse(ds.deepbet_available())


if __name__ == "__main__":
    unittest.main()
