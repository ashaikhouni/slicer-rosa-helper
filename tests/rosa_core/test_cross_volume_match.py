"""Pin the cross-frame trajectory matcher that powers `match-trajectories` /
`match-ros`: given a NAMED plan line bundle and detector lines in a *different*
RAS frame, it must recover the rigid transform from line geometry alone and pair
each plan with its true counterpart — with no image registration.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))


def _try_imports() -> bool:
    try:
        import numpy  # noqa: F401
        from rosa_core import cross_volume_match  # noqa: F401
        return True
    except ImportError:
        return False


DEPS_AVAILABLE = _try_imports()


@unittest.skipUnless(DEPS_AVAILABLE, "numpy / rosa_core not importable")
class CrossVolumeMatchTests(unittest.TestCase):
    def _rigid(self, theta_deg, t):
        import numpy as np
        c, s = np.cos(np.radians(theta_deg)), np.sin(np.radians(theta_deg))
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return R, np.asarray(t, float)

    def test_recovers_pairs_across_a_rigid_transform(self):
        import numpy as np
        from rosa_core.cross_volume_match import cross_volume_match

        # A named plan bundle with varied directions (RANSAC needs non-parallel
        # lines to fix the rotation).
        plan = [
            {"name": "RAM", "start": [0.0, 0.0, 0.0], "end": [0.0, 0.0, 40.0]},
            {"name": "RHH", "start": [20.0, 0.0, 0.0], "end": [20.0, 10.0, 40.0]},
            {"name": "LOF", "start": [-20.0, 5.0, 0.0], "end": [-15.0, 5.0, 45.0]},
            {"name": "LAC", "start": [10.0, -15.0, 0.0], "end": [5.0, -10.0, 42.0]},
        ]
        # Detector lines = plan pushed through a known rigid transform into a
        # different frame, with detector-style names.
        R, t = self._rigid(25.0, [7.0, -3.0, 11.0])
        det = []
        for tr in plan:
            s = R @ np.asarray(tr["start"], float) + t
            e = R @ np.asarray(tr["end"], float) + t
            det.append({"name": f"CAND-{tr['name']}", "start": s.tolist(), "end": e.tolist()})

        result = cross_volume_match(plan, det, seed=0)

        # Every plan got its true counterpart.
        got = {p: d for p, d, _, _ in result.pairs}
        self.assertEqual(got["RAM"], "CAND-RAM")
        self.assertEqual(got["RHH"], "CAND-RHH")
        self.assertEqual(got["LOF"], "CAND-LOF")
        self.assertEqual(got["LAC"], "CAND-LAC")
        self.assertEqual(sum(1 for _, d, _, _ in result.pairs if d), 4)
        # Tight geometric residuals on a clean synthetic bundle.
        for _p, d, ang, perp in result.pairs:
            if d:
                self.assertLess(ang, 2.0)
                self.assertLess(perp, 1.0)

    def test_unmatched_plan_kept_with_empty_det(self):
        import numpy as np
        from rosa_core.cross_volume_match import cross_volume_match

        plan = [
            {"name": "A", "start": [0.0, 0.0, 0.0], "end": [0.0, 0.0, 40.0]},
            {"name": "B", "start": [20.0, 0.0, 0.0], "end": [20.0, 10.0, 40.0]},
            {"name": "C", "start": [-20.0, 5.0, 0.0], "end": [-15.0, 5.0, 45.0]},
            {"name": "D", "start": [10.0, -15.0, 0.0], "end": [5.0, -10.0, 42.0]},
        ]
        R, t = self._rigid(15.0, [4.0, 1.0, -6.0])
        # Detector sees only A, B, C (D was not implanted / not detected).
        det = []
        for tr in plan[:3]:
            s = R @ np.asarray(tr["start"], float) + t
            e = R @ np.asarray(tr["end"], float) + t
            det.append({"name": f"d{tr['name']}", "start": s.tolist(), "end": e.tolist()})

        result = cross_volume_match(plan, det, seed=0)
        got = {p: d for p, d, _, _ in result.pairs}
        self.assertEqual(got["A"], "dA")
        self.assertIsNone(got["D"])  # no detector counterpart → kept, det empty


if __name__ == "__main__":
    unittest.main()
