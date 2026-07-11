"""Unit tests for the FastSurfer seg-only backend wrapper.

No FastSurfer/torch needed — these lock the device-selection glue, in
particular the ``ROSA_FASTSURFER_DEVICE`` override a CPU-only env requires
(the arm64 parent would otherwise pick ``mps``, and FastSurfer's
``find_device('mps')`` *raises* on a build without an MPS backend). A real
seg run is validated E2E elsewhere.
"""
from __future__ import annotations

import os
import platform
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))

try:
    from rosa_detect.services import fastsurfer_seg as fs
    HAVE = True
except Exception:  # noqa: BLE001
    HAVE = False


@unittest.skipUnless(HAVE, "rosa_detect unavailable")
class FastsurferDeviceTests(unittest.TestCase):
    def test_explicit_device_wins_over_env_override(self):
        with mock.patch.dict(os.environ, {"ROSA_FASTSURFER_DEVICE": "cpu"}):
            self.assertEqual(fs._pick_device("mps"), "mps")
            self.assertEqual(fs._pick_device("cuda"), "cuda")

    def test_env_override_forces_device_on_auto(self):
        with mock.patch.dict(os.environ, {"ROSA_FASTSURFER_DEVICE": "cpu"}):
            self.assertEqual(fs._pick_device("auto"), "cpu")

    def test_auto_without_override_uses_platform_guess(self):
        env = {k: v for k, v in os.environ.items() if k != "ROSA_FASTSURFER_DEVICE"}
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch.object(platform, "system", return_value="Darwin"), \
                 mock.patch.object(platform, "machine", return_value="arm64"):
                self.assertEqual(fs._pick_device("auto"), "mps")
            with mock.patch.object(platform, "system", return_value="Linux"), \
                 mock.patch.object(platform, "machine", return_value="x86_64"):
                self.assertEqual(fs._pick_device("auto"), "cpu")


if __name__ == "__main__":
    unittest.main()
