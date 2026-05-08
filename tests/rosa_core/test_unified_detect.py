"""Unit tests for ``rosa_core.unified_detect``.

Heavy-volume integration is covered by probe scripts on the SEEG
dataset (see project_unified_pipeline_m9_2026-05-08.md). These tests
pin:
  * the lazy export contract (no eager rosa_detect / SimpleITK pull)
  * the dataclass + module constants
  * the small pure helpers (``_axis_dup``, ``_is_genuine_seeg_chain``,
    ``_refine_axis_via_bolt``)
"""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

try:
    import numpy as np
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMONLIB = REPO_ROOT / "CommonLib"


class LazyExportTests(unittest.TestCase):
    """``import rosa_core`` doesn't drag rosa_detect/SITK into sys.modules.

    The unified_detect module *uses* both, but the lazy __getattr__
    contract says the package init shouldn't pull them until someone
    actually accesses ``rosa_core.detect_and_place_unified``.
    """

    def test_import_rosa_core_does_not_import_rosa_detect(self):
        code = (
            "import sys\n"
            "import rosa_core\n"
            "loaded_rosa_detect = any(m == 'rosa_detect' or m.startswith('rosa_detect.')\n"
            "                          for m in sys.modules)\n"
            "loaded_sitk = 'SimpleITK' in sys.modules\n"
            "print('rosa_detect' if loaded_rosa_detect else 'no_rosa_detect')\n"
            "print('sitk' if loaded_sitk else 'no_sitk')\n"
        )
        env = {"PYTHONPATH": str(COMMONLIB), "PATH": ""}
        proc = subprocess.run([sys.executable, "-c", code],
                                capture_output=True, text=True, env=env)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        out = proc.stdout.splitlines()
        self.assertIn("no_rosa_detect", out)
        self.assertIn("no_sitk", out)

    def test_lazy_attrs_resolve(self):
        """The three exported names from unified_detect resolve."""
        code = (
            "import rosa_core\n"
            "print(rosa_core.MIN_BLOBS_PER_LINE_UNIFIED)\n"
            "print(rosa_core.UnifiedTrajectory.__name__)\n"
            "print(rosa_core.detect_and_place_unified.__name__)\n"
        )
        env = {"PYTHONPATH": str(COMMONLIB), "PATH": ""}
        proc = subprocess.run([sys.executable, "-c", code],
                                capture_output=True, text=True, env=env)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        out = proc.stdout.splitlines()
        self.assertEqual(out[0], "4")
        self.assertEqual(out[1], "UnifiedTrajectory")
        self.assertEqual(out[2], "detect_and_place_unified")


@unittest.skipUnless(DEPS_AVAILABLE, "numpy not available")
class HelpersTests(unittest.TestCase):
    """Pure-numeric helpers — testable without a CT volume."""

    def setUp(self):
        sys.path.insert(0, str(COMMONLIB))
        from rosa_core import unified_detect
        self.mod = unified_detect

    def tearDown(self):
        cl = str(COMMONLIB)
        if cl in sys.path:
            sys.path.remove(cl)

    def test_axis_dup_parallel_close(self):
        """Two near-parallel axes within perp tol → duplicates."""
        s1 = np.array([0.0, 0.0, 0.0]); e1 = np.array([20.0, 0.0, 0.0])
        s2 = np.array([0.0, 1.5, 0.0]); e2 = np.array([20.0, 1.5, 0.0])
        self.assertTrue(self.mod._axis_dup(s1, e1, s2, e2, 4.0, 12.0))

    def test_axis_dup_too_far_perp(self):
        """Same direction but lateral drift > perp tol → not dup."""
        s1 = np.array([0.0, 0.0, 0.0]); e1 = np.array([20.0, 0.0, 0.0])
        s2 = np.array([0.0, 5.0, 0.0]); e2 = np.array([20.0, 5.0, 0.0])
        self.assertFalse(self.mod._axis_dup(s1, e1, s2, e2, 4.0, 12.0))

    def test_axis_dup_too_far_angle(self):
        """Same midpoint, very different direction → not dup."""
        s1 = np.array([0.0, 0.0, 0.0]); e1 = np.array([20.0, 0.0, 0.0])
        # 30° tilt
        s2 = np.array([0.0, 0.0, 0.0])
        e2 = np.array([20.0 * np.cos(np.radians(30.0)),
                        20.0 * np.sin(np.radians(30.0)), 0.0])
        self.assertFalse(self.mod._axis_dup(s1, e1, s2, e2, 4.0, 12.0))

    def test_axis_dup_direction_free(self):
        """``_axis_dup`` should treat reversed seeds as the same axis."""
        s1 = np.array([0.0, 0.0, 0.0]); e1 = np.array([20.0, 0.0, 0.0])
        s2 = np.array([20.0, 1.0, 0.0]); e2 = np.array([0.0, 1.0, 0.0])
        self.assertTrue(self.mod._axis_dup(s1, e1, s2, e2, 4.0, 12.0))

    def test_axis_dup_zero_length(self):
        """Degenerate input must not crash; should report not-dup."""
        s1 = np.array([0.0, 0.0, 0.0]); e1 = np.array([0.0, 0.0, 0.0])
        s2 = np.array([1.0, 0.0, 0.0]); e2 = np.array([20.0, 0.0, 0.0])
        self.assertFalse(self.mod._axis_dup(s1, e1, s2, e2, 4.0, 12.0))

    def test_is_genuine_seeg_chain_passes_strong_chain(self):
        """A typical SEEG chain (n=8, deep, in-band pitch) passes."""
        chain = {"n_inliers": 8, "dist_max_mm": 60.0,
                 "contact_span_mm": 24.5, "original_span_mm": 24.5,
                 "original_median_pitch_mm": 3.5}
        self.assertTrue(self.mod._is_genuine_seeg_chain(
            chain, min_blobs=5, min_dist_max=30.0, max_pitch=7.0))

    def test_is_genuine_seeg_chain_rejects_short(self):
        """Below min_blobs → False."""
        chain = {"n_inliers": 3, "dist_max_mm": 60.0,
                 "contact_span_mm": 24.5, "original_median_pitch_mm": 3.5}
        self.assertFalse(self.mod._is_genuine_seeg_chain(
            chain, min_blobs=5, min_dist_max=30.0, max_pitch=7.0))

    def test_is_genuine_seeg_chain_rejects_shallow(self):
        """``dist_max_mm`` below ``min_dist_max`` → False."""
        chain = {"n_inliers": 8, "dist_max_mm": 12.0,
                 "contact_span_mm": 24.5, "original_median_pitch_mm": 3.5}
        self.assertFalse(self.mod._is_genuine_seeg_chain(
            chain, min_blobs=5, min_dist_max=30.0, max_pitch=7.0))

    def test_is_genuine_seeg_chain_rejects_wrong_pitch(self):
        """Median pitch above ``max_pitch`` → False (cross-shank chain)."""
        chain = {"n_inliers": 8, "dist_max_mm": 60.0,
                 "contact_span_mm": 70.0, "original_median_pitch_mm": 10.0}
        self.assertFalse(self.mod._is_genuine_seeg_chain(
            chain, min_blobs=5, min_dist_max=30.0, max_pitch=7.0))

    def test_refine_axis_via_bolt_no_anchor_returns_original(self):
        """When the anchor function returns None for both directions,
        the axis is unchanged and ``anchored=False``.
        """
        def anchor_fn_none(s, e, bolts):
            return (None, None, None)
        s = np.array([10.0, 20.0, 30.0])
        e = np.array([50.0, 25.0, 35.0])
        ns, ne, anchored = self.mod._refine_axis_via_bolt(
            s, e, bolts=[], anchor_fn=anchor_fn_none,
        )
        np.testing.assert_array_equal(ns, s)
        np.testing.assert_array_equal(ne, e)
        self.assertFalse(anchored)

    def test_refine_axis_via_bolt_picks_higher_tube_count(self):
        """When both forward and reverse anchor, pick whichever tube
        count is higher.
        """
        s = np.array([0.0, 0.0, 0.0])
        e = np.array([40.0, 0.0, 0.0])
        # Fake anchor function: forward returns 100 voxels, reverse 200.
        def anchor_fn(p_start, p_end, bolts):
            if np.allclose(p_start, s):
                return (np.array([-5.0, 0.0, 0.0]), None, {"tube_n_vox": 100})
            return (np.array([45.0, 0.0, 0.0]), None, {"tube_n_vox": 200})
        ns, ne, anchored = self.mod._refine_axis_via_bolt(
            s, e, bolts=[], anchor_fn=anchor_fn,
        )
        self.assertTrue(anchored)
        # Reverse won → new_start sits past the original deep end.
        np.testing.assert_array_equal(ns, np.array([45.0, 0.0, 0.0]))
        np.testing.assert_array_equal(ne, s)


@unittest.skipUnless(DEPS_AVAILABLE, "numpy not available")
class UnifiedTrajectoryDataclassTests(unittest.TestCase):
    """Pin the public surface of the result dataclass."""

    def setUp(self):
        sys.path.insert(0, str(COMMONLIB))
        from rosa_core import UnifiedTrajectory
        self.UnifiedTrajectory = UnifiedTrajectory

    def tearDown(self):
        cl = str(COMMONLIB)
        if cl in sys.path:
            sys.path.remove(cl)

    def test_minimal_construction(self):
        t = self.UnifiedTrajectory(
            start_ras=[0.0, 0.0, 0.0], end_ras=[10.0, 0.0, 0.0],
            model_id="PMT-8", corr_score=0.7,
            placed_ras=[[0.0, 0.0, 0.0]],
            centerline_ras=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            bolt_source="metal", n_placed=1,
            anchored=True, synthed=False,
        )
        self.assertEqual(t.model_id, "PMT-8")
        self.assertEqual(t.diagnostics, {})


if __name__ == "__main__":
    unittest.main()
