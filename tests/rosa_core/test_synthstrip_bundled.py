"""Unit tests for the vendored standalone SynthStrip (``synthstrip_bundled``).

These exercise the pure-stdlib orchestration — cache resolution, checksum
verification, the no-download / wrong-checksum guards, csf/nocsf weight
selection, and the clear errors when the runtime lacks torch — WITHOUT needing
torch, surfa, or any network. The actual neural skull-strip is covered by a
separate, opt-in end-to-end check that runs only where torch+surfa are present.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))

# Module imports with stdlib only — no torch — so this never skips.
from rosa_detect.services import synthstrip_bundled as sb  # noqa: E402


class ConstantsTests(unittest.TestCase):
    def test_pinned_checksums_are_sane(self):
        for asset in (sb._SCRIPT, sb._WEIGHTS["default"], sb._WEIGHTS["nocsf"]):
            self.assertRegex(asset.sha256, r"^[0-9a-f]{64}$")
            self.assertTrue(asset.url.startswith("https://"))
        # Weights are the documented ~30 MB; both variants the same size.
        self.assertEqual(sb._WEIGHTS["default"].size, 30851709)
        self.assertEqual(sb._WEIGHTS["nocsf"].size, 30851709)
        # csf and nocsf are genuinely different weights.
        self.assertNotEqual(sb._WEIGHTS["default"].sha256,
                            sb._WEIGHTS["nocsf"].sha256)


class CacheDirTests(unittest.TestCase):
    def test_respects_explicit_cache_env(self):
        with mock.patch.dict("os.environ", {sb._CACHE_ENV: "/tmp/rosa_ss_cache"}):
            self.assertEqual(sb.cache_dir(), Path("/tmp/rosa_ss_cache"))

    def test_falls_back_to_xdg_then_home(self):
        with mock.patch.dict("os.environ", {"XDG_CACHE_HOME": "/tmp/xdg"}, clear=False):
            with mock.patch.dict("os.environ", {sb._CACHE_ENV: ""}, clear=False):
                # _CACHE_ENV empty -> XDG path used
                import os
                os.environ.pop(sb._CACHE_ENV, None)
                self.assertEqual(sb.cache_dir(),
                                 Path("/tmp/xdg") / "rosa-agent" / "synthstrip")


class ChecksumTests(unittest.TestCase):
    def test_sha256_and_have_roundtrip(self):
        import hashlib, tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "blob.bin"
            p.write_bytes(b"rosa-synthstrip-test")
            digest = hashlib.sha256(b"rosa-synthstrip-test").hexdigest()
            self.assertEqual(sb._sha256(p), digest)
            self.assertTrue(sb._have(p, digest))
            self.assertFalse(sb._have(p, "00" * 32))
            self.assertFalse(sb._have(Path(td) / "missing.bin", digest))


class FetchTests(unittest.TestCase):
    def test_no_download_when_disabled_and_absent(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "synthstrip.1.pt"
            with mock.patch.dict("os.environ", {sb._NO_DOWNLOAD_ENV: "1"}):
                with self.assertRaises(sb.BundledSynthStripUnavailable):
                    sb._fetch(sb._WEIGHTS["default"], dest)

    def test_checksum_mismatch_raises_and_does_not_poison_cache(self):
        import io, tempfile
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "synthstrip.1.pt"
            # urlopen returns bytes whose sha256 != the pinned value.
            fake = io.BytesIO(b"not the real weights")
            cm = mock.MagicMock()
            cm.__enter__.return_value = fake
            cm.__exit__.return_value = False
            with mock.patch.object(sb.urllib.request, "urlopen", return_value=cm):
                with self.assertRaises(sb.BundledSynthStripError):
                    sb._fetch(sb._WEIGHTS["default"], dest)
            # The bad bytes must NOT have been committed to the cache path.
            self.assertFalse(dest.exists())

    def test_cached_valid_asset_is_returned_without_download(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            content = b"already here"
            import hashlib
            asset = sb._Asset(filename="x.bin", url="https://example/x.bin",
                              sha256=hashlib.sha256(content).hexdigest(), size=len(content))
            dest = Path(td) / "x.bin"
            dest.write_bytes(content)
            # urlopen must never be called for an already-valid cache entry.
            with mock.patch.object(sb.urllib.request, "urlopen",
                                   side_effect=AssertionError("should not download")):
                self.assertEqual(sb._fetch(asset, dest), dest)


class EnsureAssetsTests(unittest.TestCase):
    def test_selects_nocsf_weights_when_requested(self):
        captured = []

        def _fake_fetch(asset, dest):
            captured.append(asset.filename)
            return dest
        with mock.patch.object(sb, "_fetch", _fake_fetch):
            sb.ensure_assets(no_csf=True)
        self.assertIn("synthstrip.nocsf.1.pt", captured)
        self.assertNotIn("synthstrip.1.pt", captured)

    def test_selects_default_weights_otherwise(self):
        captured = []
        with mock.patch.object(sb, "_fetch", lambda a, d: captured.append(a.filename) or d):
            sb.ensure_assets(no_csf=False)
        self.assertIn("synthstrip.1.pt", captured)
        self.assertNotIn("synthstrip.nocsf.1.pt", captured)


class RuntimeGuardTests(unittest.TestCase):
    def test_run_raises_unavailable_without_torch(self):
        sb._runtime_ok.cache_clear()
        with mock.patch.object(sb, "runtime_ok", return_value=False):
            with self.assertRaises(sb.BundledSynthStripUnavailable):
                sb.run_bundled_synthstrip("in.nii.gz", "mask.nii.gz")

    def test_available_false_without_torch(self):
        with mock.patch.object(sb, "runtime_ok", return_value=False):
            self.assertFalse(sb.available())


_T22_CT = REPO_ROOT / "tests" / "data" / "T22" / "T22_ct.nii.gz"


@unittest.skipUnless(
    sb.runtime_ok() and _T22_CT.is_file(),
    "needs torch+surfa and the (gitignored) T22 CT — runs locally, skips in CI",
)
class EndToEndStripTests(unittest.TestCase):
    """Real skull-strip through the vendored standalone SynthStrip — fetches the
    pinned script + weights and runs the network on CPU. Only runs where torch +
    surfa + a CT volume are present (never in hosted CI)."""

    def test_strips_a_real_ct(self):
        import tempfile
        import numpy as np
        import SimpleITK as sitk

        with tempfile.TemporaryDirectory() as td:
            mask_p = Path(td) / "mask.nii.gz"
            out = sb.run_bundled_synthstrip(_T22_CT, mask_p, timeout=600)
            self.assertTrue(out.is_file())

            ct = sitk.ReadImage(str(_T22_CT))
            mask = sitk.ReadImage(str(out))
            # Geometry inherited from the input (fix_mask_geometry).
            self.assertEqual(mask.GetSize(), ct.GetSize())
            arr = sitk.GetArrayFromImage(mask)
            # Binary, non-trivial brain fraction (a real head CT strips to a
            # sizeable but not whole-volume mask).
            uniq = set(np.unique(arr).tolist())
            self.assertTrue(uniq.issubset({0, 1}))
            frac = float((arr > 0).mean())
            self.assertGreater(frac, 0.02)
            self.assertLess(frac, 0.80)


if __name__ == "__main__":
    unittest.main()
