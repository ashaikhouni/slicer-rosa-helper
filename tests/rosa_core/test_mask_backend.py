"""Unit tests for the shared brain-mask backend selector — specifically the
non-throwing full-volume fallback that keeps the place/export CLIs (and any
caller) tolerant of degenerate / non-CT inputs on the lenient ``auto`` default,
while an EXPLICITLY-requested backend that fails still raises (strict).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))

try:
    import numpy as np
    import SimpleITK as sitk
    from rosa_detect.services import mask_backend as mb
    HAVE_DEPS = True
except Exception:  # noqa: BLE001
    HAVE_DEPS = False


@unittest.skipUnless(HAVE_DEPS, "numpy / SimpleITK / rosa_detect unavailable")
class MaskBackendFallbackTests(unittest.TestCase):
    def setUp(self):
        self.img = sitk.GetImageFromArray(np.zeros((8, 8, 8), dtype=np.float32))

    def test_auto_falls_back_to_full_volume_when_all_backends_fail(self):
        # Simulate synthstrip + log-watershed both failing (the no-SynthStrip
        # CI path on a degenerate phantom). auto must NOT raise — it returns a
        # full-volume permissive mask so downstream just finds nothing.
        with mock.patch.object(mb, "select_brain_mask_to_path", return_value=None):
            mask, used = mb.compute_intracranial_mask(self.img, backend="auto")
        self.assertEqual(used, "full-volume")
        self.assertEqual(mask.shape, (8, 8, 8))
        self.assertTrue(bool(mask.all()))
        self.assertEqual(mask.dtype, np.bool_)

    def test_explicit_backend_still_raises_on_failure(self):
        # An explicitly-requested backend that fails is a real error (a real
        # subject should surface a mis-configured mask loudly), not a silent
        # full-volume fallback.
        with mock.patch.object(mb, "select_brain_mask_to_path", return_value=None):
            for backend in ("synthstrip", "log-watershed"):
                with self.assertRaises(RuntimeError):
                    mb.compute_intracranial_mask(self.img, backend=backend)

    def test_provided_brain_mask_overrides_backends(self):
        # An explicit override short-circuits the backends entirely.
        override = np.ones((8, 8, 8), dtype=bool)
        called = {"n": 0}

        def _should_not_run(*_a, **_k):
            called["n"] += 1
            return None
        with mock.patch.object(mb, "select_brain_mask_to_path", _should_not_run):
            mask, used = mb.compute_intracranial_mask(
                self.img, backend="auto", brain_mask=override,
            )
        self.assertEqual(used, "provided")
        self.assertTrue(bool(mask.all()))
        self.assertEqual(called["n"], 0)   # backends never consulted


@unittest.skipUnless(HAVE_DEPS, "numpy / SimpleITK / rosa_detect unavailable")
class HullBackendTests(unittest.TestCase):
    """The 'hull' backend is the cheap build_masks intracranial — a fast,
    dependency-free approximation (no temp file, no external binary)."""

    def setUp(self):
        # -1000 HU background with a +100 HU tissue cube → build_masks yields a
        # clean hull + a non-empty intracranial core (head-distance ≥ 10 mm).
        arr = np.full((32, 32, 32), -1000.0, dtype=np.float32)
        arr[4:28, 4:28, 4:28] = 100.0
        self.img = sitk.GetImageFromArray(arr)
        self.img.SetSpacing((1.0, 1.0, 1.0))

    def test_hull_matches_build_masks_and_skips_selector(self):
        from rosa_detect.primitives.preprocessing import build_masks
        # 'hull' short-circuits before the file-based selector — never runs it.
        with mock.patch.object(mb, "select_brain_mask_to_path") as m:
            mask, used = mb.compute_intracranial_mask(self.img, backend="hull")
            m.assert_not_called()
        self.assertEqual(used, "hull")
        self.assertEqual(mask.dtype, np.bool_)
        self.assertEqual(mask.shape, (32, 32, 32))
        self.assertTrue(bool(mask.any()))
        _hull, intra, _dist = build_masks(self.img)
        self.assertTrue(np.array_equal(mask, intra.astype(bool)))

    def test_provided_override_beats_hull(self):
        # An explicit brain_mask still wins over backend="hull".
        override = np.ones((32, 32, 32), dtype=bool)
        mask, used = mb.compute_intracranial_mask(
            self.img, backend="hull", brain_mask=override,
        )
        self.assertEqual(used, "provided")
        self.assertTrue(bool(mask.all()))

    def test_select_to_path_hull_writes_mask_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            ct_p = Path(td) / "ct.nii.gz"
            m_p = Path(td) / "mask.nii.gz"
            sitk.WriteImage(self.img, str(ct_p))
            used = mb.select_brain_mask_to_path(ct_p, m_p, backend="hull")
            self.assertEqual(used, "hull")
            self.assertTrue(m_p.exists())
            mask = sitk.GetArrayFromImage(sitk.ReadImage(str(m_p)))
            self.assertTrue(bool(mask.any()))


if __name__ == "__main__":
    unittest.main()
