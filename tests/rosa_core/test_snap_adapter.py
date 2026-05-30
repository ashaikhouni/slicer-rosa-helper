"""Tests for rosa_core.contact_placement.snap_adapter.snap_chain_to_ctx — the
chain -> PlacementCtx bridge that feeds the fit-rosa snap-flow into the staged
scorers. Validates the full ctx contract Stage-F + the PlacedTrajectory output
need, end-to-end through scoring, on a synthetic metal comb.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))

try:
    import numpy as np
    from rosa_core.electrode_models import load_electrode_library
    from rosa_core.electrode_classifier import filter_models_for_strategy
    from rosa_core.contact_placement import (
        score_cc_overlap, score_compound, score_simple, snap_chain_to_ctx,
    )
    HAVE_DEPS = True
except Exception:  # noqa: BLE001
    HAVE_DEPS = False

_PITCH = 3.5
_X = _Y = 25          # comb runs up the z-axis at (x=25, y=25)
_Z0 = 12.0
_N = 10               # 10 contacts -> expect a ~10-contact AM pick


def _synthetic_features(n_contacts=_N, vol=60):
    """A clean metal comb up the z-axis: bright LoG-neg blobs every 3.5 mm,
    all-ones intracranial mask, identity RAS<->IJK. arr is [k,j,i] so a RAS
    point (x,y,z) samples arr[z,y,x]."""
    log = np.zeros((vol, vol, vol), dtype=np.float32)
    ct = np.zeros((vol, vol, vol), dtype=np.float32)
    zs = [_Z0 + i * _PITCH for i in range(n_contacts)]
    for z in zs:
        k = int(round(z))
        log[k, _Y - 2:_Y + 3, _X - 2:_X + 3] = -1500.0   # metal => negative LoG
        ct[k, _Y - 2:_Y + 3, _X - 2:_X + 3] = 3000.0     # bright metal HU
    mask = np.ones((vol, vol, vol), dtype=np.float32)
    feats = {
        "log": log,
        "ct_arr_kji": ct,
        "ras_to_ijk_mat": np.eye(4),
        "intracranial_mask": mask,
    }
    return feats, zs


def _chain(zs):
    pts = np.array([[_X, _Y, z] for z in zs], dtype=float)
    return {
        "axis": np.array([0.0, 0.0, 1.0]),
        "entry_ras": pts[0].copy(),
        "tip_ras": pts[-1].copy(),
        "kept_pts": pts,
    }


@unittest.skipUnless(HAVE_DEPS, "numpy / rosa_core unavailable")
class SnapAdapterTests(unittest.TestCase):
    def setUp(self):
        self.lib = filter_models_for_strategy(
            load_electrode_library()["models"], "dixi",
        )
        self.feats, self.zs = _synthetic_features()
        self.chain = _chain(self.zs)

    def test_ctx_contract_fields_populated(self):
        ctx = snap_chain_to_ctx(
            self.chain, features=self.feats, library_models=self.lib,
        )
        # Geometry + signal.
        self.assertEqual(np.asarray(ctx.centerline).shape, (2, 3))
        self.assertIsNotNone(ctx.walk_arcs)
        self.assertIsNotNone(ctx.walk_signal)
        self.assertEqual(len(ctx.walk_arcs), len(ctx.walk_signal))
        self.assertGreater(len(ctx.walk_arcs), 10)
        self.assertEqual(ctx.signal_kind, "neg_log_max")
        # Anchor + source.
        self.assertEqual(ctx.bolt_source, "metal")
        self.assertEqual(ctx.bolt_end_arc, 0.0)   # all-ones mask -> in-brain at entry
        # A pick happened on the clean comb.
        self.assertIsNotNone(ctx.match)
        self.assertIsNotNone(ctx.match.best_model_id)
        self.assertTrue(ctx.match.best_model_id.endswith("AM"))  # uniform comb
        # Contacts placed from the fitted tip.
        self.assertGreaterEqual(len(ctx.placed_ras), 8)
        # Deepest placed contact sits at/near the fitted tip.
        tip = np.asarray(self.chain["tip_ras"])
        dmin = min(float(np.linalg.norm(np.asarray(p) - tip)) for p in ctx.placed_ras)
        self.assertLess(dmin, 1.0)

    def test_scorers_run_end_to_end(self):
        ctx = snap_chain_to_ctx(
            self.chain, features=self.feats, library_models=self.lib,
            seeder_label="high", seeder_confidence=0.9,
        )
        ctx = score_simple(ctx)
        ctx = score_cc_overlap(ctx)
        ctx = score_compound(ctx)
        sc = ctx.score_components
        self.assertIn("compound_score", sc)
        self.assertGreaterEqual(float(sc["compound_score"]), 0.0)
        self.assertLessEqual(float(sc["compound_score"]), 1.0)
        self.assertIn("band", sc)
        self.assertIn(sc["band"], {"high", "medium", "low"})
        self.assertIn("model_id", sc)

    def test_forced_model_skips_floor_and_places_it(self):
        ctx = snap_chain_to_ctx(
            self.chain, features=self.feats, library_models=self.lib,
            forced_model_id="DIXI-15AM",
        )
        self.assertEqual(ctx.match.best_model_id, "DIXI-15AM")
        self.assertEqual(len(ctx.placed_ras), 15)   # exactly the forced model

    def test_poor_signal_still_yields_wellformed_ctx(self):
        # Empty volume -> no metal; the adapter must still produce a ctx the
        # scorers can run on (graceful, not a crash).
        vol = 60
        feats = {
            "log": np.zeros((vol, vol, vol), dtype=np.float32),
            "ct_arr_kji": np.zeros((vol, vol, vol), dtype=np.float32),
            "ras_to_ijk_mat": np.eye(4),
            "intracranial_mask": np.ones((vol, vol, vol), dtype=np.float32),
        }
        ctx = snap_chain_to_ctx(self.chain, features=feats, library_models=self.lib)
        self.assertEqual(np.asarray(ctx.centerline).shape, (2, 3))
        self.assertEqual(ctx.bolt_source, "metal")
        ctx = score_compound(score_cc_overlap(score_simple(ctx)))
        self.assertIn("compound_score", ctx.score_components)


if __name__ == "__main__":
    unittest.main()
