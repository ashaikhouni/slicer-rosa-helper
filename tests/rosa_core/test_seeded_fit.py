"""Targeted regression for ``rosa_core.seeded_fit`` pass-2 arbitration.

The rule under test is leave-one-out + jump-gate attribution
(:func:`arbitrate_shared_peaks`): a tail contact is reassigned away from a
chain only when it (1) lies beyond a ``> JUMP_GATE_PITCH x pitch`` jump from
the chain body — a contiguous tip is never peeled — and (2) with itself left
out, sits closer to a neighbouring chain's detected axis, within that
neighbour's contact extent, than to the line through the chain's remaining
contacts.

Two synthetic smoke cases:

1. **Crossing**: two electrodes cross at ~90° with neighbouring (but
   distinct) contacts ~1.5 mm apart in 3D. Every contact is contiguous with
   its own array (no jump), so arbitration keeps both chains intact.

2. **T-end**: shank A over-reaches onto shank B's body — a spurious peak
   two pitches past A's real tip, landing on a B contact. It is a jump
   (eligible) and is closer to B's axis than to A's leave-one-out spine, so
   it is dropped from A.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))

try:
    import numpy as np
    from rosa_core.seeded_fit import (
        snap_via_signal_walk, arbitrate_shared_peaks,
    )
    HAVE_DEPS = True
except Exception:  # noqa: BLE001
    HAVE_DEPS = False


def _build_volume(contact_centers, *, shape=(80, 80, 80),
                   contact_sigma=0.7, peak_amp=1500.0):
    """Render Gaussian blobs at each RAS contact center. ``vol[k,j,i]``
    holds the value at RAS ``(x=i, y=j, z=k)`` so ``ras_to_ijk = eye(4)``
    works directly with ``sample_trilinear_batch``."""
    vol = np.zeros(shape, dtype=np.float32)
    k_idx, j_idx, i_idx = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]),
        indexing="ij",
    )
    coords = np.stack([i_idx, j_idx, k_idx], axis=-1).astype(np.float32)
    inv_2s2 = 1.0 / (2.0 * contact_sigma * contact_sigma)
    for c in contact_centers:
        d2 = ((coords - c) ** 2).sum(axis=-1)
        vol += peak_amp * np.exp(-d2 * inv_2s2).astype(np.float32)
    return vol


def _mk_chain(pts, axis):
    """Minimal chain dict in the shape arbitration consumes."""
    return {
        "axis": np.asarray(axis, float),
        "centroid": pts.mean(axis=0),
        "kept_pts": pts,
        "kept_along": (pts - pts.mean(axis=0)) @ np.asarray(axis, float),
        "entry_ras": pts[0],
        "tip_ras":   pts[-1],
        "extent_mm": float(np.linalg.norm(pts[-1] - pts[0])),
        "median_pitch_mm": 3.5,
        "pitch_mode_mm":   3.5,
        "n_peaks": len(pts),
    }


@unittest.skipUnless(HAVE_DEPS, "numpy / rosa_core unavailable")
class TestArbitrate(unittest.TestCase):
    """Leave-one-out + jump-gate ``arbitrate_shared_peaks``."""

    def test_crossing_keeps_both_chains_contacts(self):
        """Two electrodes crossing at ~90° each have contacts ~1.5 mm
        apart in 3D at the crossing. Every contact is contiguous with its
        own array (no jump), so arbitration must keep both chains."""
        A = np.array([[8 + k * 3.5, 40.0, 40.0] for k in range(10)])
        cross_x = A[5, 0]
        B = np.array([[cross_x, 38.0 + k * 3.5, 41.5] for k in range(10)])
        vol = _build_volume(np.concatenate([A, B]))
        ras_to_ijk = np.eye(4, dtype=float)
        chain_A = snap_via_signal_walk(
            A[0] - 1.5 * (A[1] - A[0]), A[-1] + 1.5 * (A[-1] - A[-2]),
            signal_vol=vol, threshold=80.0, ras_to_ijk=ras_to_ijk,
        )
        chain_B = snap_via_signal_walk(
            B[0] - 1.5 * (B[1] - B[0]), B[-1] + 1.5 * (B[-1] - B[-2]),
            signal_vol=vol, threshold=80.0, ras_to_ijk=ras_to_ijk,
        )
        self.assertIsNotNone(chain_A)
        self.assertIsNotNone(chain_B)
        n_A_before = len(chain_A["kept_pts"])
        n_B_before = len(chain_B["kept_pts"])
        new_chains, _ = arbitrate_shared_peaks([chain_A, chain_B])
        self.assertGreaterEqual(len(new_chains[0]["kept_pts"]), n_A_before - 1,
                                "crossing should keep A's contacts")
        self.assertGreaterEqual(len(new_chains[1]["kept_pts"]), n_B_before - 1,
                                "crossing should keep B's contacts")

    def test_tend_drops_jumped_peak_onto_bar(self):
        """T-end: shank A (along +z at x=20, y=40) over-reaches with a
        spurious peak two pitches past its real tip, landing on a contact
        of shank B (along +x at z=50). The peak is a jump (> gate) and is
        closer to B's axis than to A's leave-one-out spine, so it is
        dropped from A; A keeps exactly its 8 real contacts."""
        A_pts = np.array([[20.0, 40.0, 20.0 + k * 3.5] for k in range(8)])
        # Spurious over-reach at B's contact (24.5, 40, 50): z=50 is ~1.6
        # pitch past A's real tip (z=44.5) -> a jump; perp to A's axis = 4.5
        # mm (off A) but perp to B's axis = 0 (on B).
        A_pts = np.vstack([A_pts, np.array([24.5, 40.0, 50.0])])
        B_pts = np.array([[k * 3.5, 40.0, 50.0] for k in range(15)])
        chain_A = _mk_chain(A_pts, [0.0, 0.0, 1.0])
        chain_B = _mk_chain(B_pts, [1.0, 0.0, 0.0])

        new_chains, n_conflicts = arbitrate_shared_peaks([chain_A, chain_B])
        new_A, new_B = new_chains
        self.assertIsNotNone(new_A)
        self.assertIsNotNone(new_B)
        self.assertEqual(len(new_A["kept_pts"]), 8,
                         "A should drop the jumped peak that landed on B")
        self.assertGreaterEqual(n_conflicts, 1,
                                "expected at least one reassignment")

    def test_contiguous_tip_near_neighbour_is_protected(self):
        """A genuine deep tip that is contiguous (one pitch from its
        neighbour) must NOT be peeled even when a neighbour's axis passes
        close — the jump-gate protects it."""
        # A along +z; its real tip at z=44.5 is contiguous (one pitch from
        # z=41.0). B runs along +x right next to A's tip (z=44.5, y=41).
        A_pts = np.array([[20.0, 40.0, 20.0 + k * 3.5] for k in range(8)])
        B_pts = np.array([[k * 3.5, 41.0, 44.5] for k in range(15)])
        chain_A = _mk_chain(A_pts, [0.0, 0.0, 1.0])
        chain_B = _mk_chain(B_pts, [1.0, 0.0, 0.0])
        new_chains, _ = arbitrate_shared_peaks([chain_A, chain_B])
        self.assertEqual(len(new_chains[0]["kept_pts"]), 8,
                         "contiguous tip must be protected from peeling")


@unittest.skipUnless(HAVE_DEPS, "numpy / rosa_core unavailable")
class TestSeedOffsetRescue(unittest.TestCase):
    """The tube-max fallback in ``snap_via_signal_walk`` rescues a seed that
    is laterally offset from the real electrode (plan-vs-implant mismatch)."""

    def test_offaxis_seed_is_rescued_and_recentered(self):
        # Electrode along +z at x=40, y=40 (pitch 3.5, 10 contacts).
        C = np.array([[40.0, 40.0, 12.0 + k * 3.5] for k in range(10)])
        vol = _build_volume(C)               # contact_sigma 0.7
        ras_to_ijk = np.eye(4, dtype=float)
        # Seed offset +2.5 mm in x: the bare centerline samples ~2.5 mm off
        # the blobs (value << threshold), so without the fallback it finds
        # < 4 peaks and returns None. The tube-max fallback finds the
        # off-axis contacts; localization then re-centers onto the electrode.
        off = np.array([2.5, 0.0, 0.0])
        start = C[0] + off - 1.5 * (C[1] - C[0])
        end = C[-1] + off + 1.5 * (C[-1] - C[-2])
        chain = snap_via_signal_walk(
            start, end, signal_vol=vol, threshold=80.0, ras_to_ijk=ras_to_ijk,
        )
        self.assertIsNotNone(chain, "off-axis seed should be rescued by the tube fallback")
        self.assertGreaterEqual(len(chain["kept_pts"]), 8)
        # Re-centered onto the real electrode (x ~ 40, not the seed's 42.5).
        mean_x = float(np.asarray(chain["kept_pts"])[:, 0].mean())
        self.assertLess(abs(mean_x - 40.0), 0.8,
                        "rescued chain must re-center onto the real electrode axis")

    def test_onaxis_seed_does_not_trigger_fallback(self):
        """An on-axis seed snaps via the centerline exactly as before — the
        fallback is a no-op here (guards the zero-regression property)."""
        C = np.array([[40.0, 40.0, 12.0 + k * 3.5] for k in range(10)])
        vol = _build_volume(C)
        chain = snap_via_signal_walk(
            C[0] - 1.5 * (C[1] - C[0]), C[-1] + 1.5 * (C[-1] - C[-2]),
            signal_vol=vol, threshold=80.0, ras_to_ijk=np.eye(4, dtype=float),
        )
        self.assertIsNotNone(chain)
        self.assertGreaterEqual(len(chain["kept_pts"]), 9)


if __name__ == "__main__":
    unittest.main()
