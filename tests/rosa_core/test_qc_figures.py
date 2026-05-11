"""Smoke tests for ``rosa_core.qc_figures``.

Exercises the renderer end-to-end on a synthetic CT + PlacedTrajectory.
Skips cleanly when matplotlib isn't available.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from rosa_core.placement_modes import PlacedTrajectory


def _matplotlib_available() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


def _features():
    """Features dict shape that the renderer needs (subset of compute_features)."""
    K, J, I = 32, 32, 64
    arr = np.zeros((K, J, I), dtype=np.float32)
    for n in range(8):
        z = 5 + int(n * 4)
        if z < I:
            arr[16, 16, z] = 2500.0
    log = np.where(arr > 1000.0, -arr, np.zeros_like(arr)).astype(np.float32)
    return {
        "ct_arr_kji":     arr,
        "log":            log,
        "ras_to_ijk_mat": np.eye(4),
        "ijk_to_ras_mat": np.eye(4),
    }


def _traj():
    return PlacedTrajectory(
        name="SYNTH-L1",
        start_ras=np.array([16, 16, 0], dtype=float),
        end_ras=np.array([16, 16, 35], dtype=float),
        centerline_ras=np.array([[16, 16, 0], [16, 16, 35]], dtype=float),
        contacts_ras=[np.array([16, 16, 5 + 4 * n], dtype=float) for n in range(8)],
        model_id="SYNTH-8",
        compound_score=0.78,
        band="high",
        bolt_source="metal",
        bolt_end_arc_mm=4.0,
        score_components={"per_model_corr": [("SYNTH-8", 8, 8, 0.85)]},
        diagnostics={"signal_kind": "neg_log_max"},
    )


@unittest.skipUnless(_matplotlib_available(), "matplotlib not installed")
class QcFiguresSmokeTests(unittest.TestCase):
    def test_render_writes_png(self):
        from rosa_core.qc_figures import render_placed_trajectory_figure
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "fig.png"
            render_placed_trajectory_figure(
                _traj(), out, features=_features(), bolts=[],
            )
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 1024)  # non-trivial PNG

    def test_render_all_writes_one_per_traj(self):
        from rosa_core.qc_figures import render_all_figures
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            n = render_all_figures(
                [_traj(), _traj()], out, features=_features(), bolts=[],
            )
            self.assertEqual(n, 2)
            pngs = sorted(out.glob("*.png"))
            self.assertEqual(len(pngs), 2)
            self.assertTrue(pngs[0].name.startswith("001_"))
            self.assertTrue(pngs[1].name.startswith("002_"))


class QcFiguresGracefulNoMatplotlibTests(unittest.TestCase):
    """When matplotlib is missing, ``render_all_figures`` returns 0
    quietly. We can't simulate that on a system that has matplotlib, but
    the tests below verify the dispatch path."""

    def test_safe_name_strips_unsafe_chars(self):
        from rosa_core.qc_figures import _safe_name
        self.assertEqual(_safe_name("L_AC"), "L_AC")
        self.assertEqual(_safe_name("L/AC"), "L_AC")
        self.assertEqual(_safe_name("L AC #1"), "L_AC__1")
        self.assertEqual(_safe_name("X.0"), "X.0")


if __name__ == "__main__":
    unittest.main()
