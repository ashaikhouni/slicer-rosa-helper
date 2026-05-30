"""Unit tests for the shared electrode-model pick — especially the
covering-floor, which is the behavioral payoff of the contact-placement
consolidation (it is what flips T18/X11's under-pick once the staged pipeline
feeds a real snap chain) yet had no direct coverage.

Pins three contracts:
  * ``model_family`` routing (uniform / cluster / mm).
  * ``_count_resolved_proximal`` — resolved-contact counting (prominence +
    distal-of-anchor + deep-tip contiguity), incl. the chain dict shape
    ``{kept_pts, entry_ras, axis}``.
  * ``_smallest_covering`` — bump UP within the same family, cap at largest,
    never cross family.
  * end-to-end: the floor fires when the matcher under-picks a uniform model,
    and ``chain=None`` disables it entirely.
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
    from rosa_core.model_pick import (
        _count_resolved_proximal,
        _smallest_covering,
        model_family,
        pick_electrode_model,
    )
    HAVE_DEPS = True
except Exception:  # noqa: BLE001
    HAVE_DEPS = False

_STEP = 0.3
_PITCH = 3.5


def _comb(peak_arcs, *, lo=0.0, hi=50.0, amp=1.0, base=0.05, sigma=0.6):
    """A clean periodic LoG-neg-like profile: tall Gaussian bumps on a low
    baseline at each contact arc. Returns ``(arcs, sig)``."""
    arcs = np.arange(lo, hi, _STEP)
    sig = np.full_like(arcs, base, dtype=float)
    for a in peak_arcs:
        sig = sig + amp * np.exp(-0.5 * ((arcs - a) / sigma) ** 2)
    return arcs, sig


def _chain(peak_arcs, *, axis=(1.0, 0.0, 0.0), entry=(0.0, 0.0, 0.0)):
    """A snap chain dict ``{kept_pts, entry_ras, axis}`` whose contacts lie at
    ``peak_arcs`` along ``axis`` from ``entry`` (so projection recovers arcs)."""
    axis = np.asarray(axis, dtype=float)
    entry = np.asarray(entry, dtype=float)
    pts = np.array([entry + a * axis for a in peak_arcs], dtype=float)
    return {"kept_pts": pts, "entry_ras": entry, "axis": axis}


@unittest.skipUnless(HAVE_DEPS, "numpy / rosa_core unavailable")
class ModelFamilyTests(unittest.TestCase):
    def setUp(self):
        self.lib = {m["id"]: m for m in load_electrode_library()["models"]}

    def test_am_is_uniform(self):
        self.assertEqual(model_family(self.lib["DIXI-15AM"]), "uniform")

    def test_cm_is_cluster(self):
        self.assertEqual(model_family(self.lib["DIXI-18CM"]), "cluster")

    def test_mm_is_mm(self):
        mm = next((mid for mid in self.lib if "MM" in mid), None)
        if mm is None:
            self.skipTest("no MM model in library")
        self.assertEqual(model_family(self.lib[mm]), "mm")

    def test_empty_is_unknown(self):
        self.assertEqual(model_family({}), "?")


@unittest.skipUnless(HAVE_DEPS, "numpy / rosa_core unavailable")
class CountResolvedTests(unittest.TestCase):
    def test_counts_all_contiguous_distal_peaks(self):
        arcs_pk = [8.0 + i * _PITCH for i in range(12)]   # 12 contacts
        arcs, sig = _comb(arcs_pk)
        n = _count_resolved_proximal(_chain(arcs_pk), arcs, sig, anchor=5.0)
        self.assertEqual(n, 12)

    def test_excludes_peaks_proximal_of_anchor(self):
        arcs_pk = [8.0 + i * _PITCH for i in range(12)]
        arcs, sig = _comb(arcs_pk)
        # Anchor past the first three contacts → only 9 are distal.
        anchor = arcs_pk[3] - 0.5
        n = _count_resolved_proximal(_chain(arcs_pk), arcs, sig, anchor=anchor)
        self.assertEqual(n, 9)

    def test_jump_gate_breaks_contiguity_at_deep_tip(self):
        # One isolated shallow peak + a contiguous deep run of 3.
        deep = [20.0, 23.5, 27.0]
        arcs_pk = [5.0, *deep]
        arcs, sig = _comb(arcs_pk, hi=32.0)
        n = _count_resolved_proximal(_chain(arcs_pk), arcs, sig, anchor=3.0)
        self.assertEqual(n, 3)   # the shallow peak is past a >1.3*pitch jump

    def test_two_or_fewer_points_returns_len(self):
        arcs, sig = _comb([10.0])
        n = _count_resolved_proximal(_chain([10.0]), arcs, sig, anchor=0.0)
        self.assertEqual(n, 1)


@unittest.skipUnless(HAVE_DEPS, "numpy / rosa_core unavailable")
class SmallestCoveringTests(unittest.TestCase):
    def setUp(self):
        self.lib = {m["id"]: m for m in load_electrode_library()["models"]}

    def test_bumps_to_smallest_covering_am(self):
        # AM ladder: 5/8/10/12/15/18.
        self.assertEqual(_smallest_covering("DIXI-5AM", 12, self.lib), "DIXI-12AM")
        self.assertEqual(_smallest_covering("DIXI-8AM", 13, self.lib), "DIXI-15AM")
        self.assertEqual(_smallest_covering("DIXI-10AM", 16, self.lib), "DIXI-18AM")

    def test_caps_at_largest_when_none_covers(self):
        self.assertEqual(_smallest_covering("DIXI-5AM", 99, self.lib), "DIXI-18AM")

    def test_never_crosses_family(self):
        # CM ladder is 15/18 — a CM under-count stays CM, never jumps to AM.
        self.assertEqual(_smallest_covering("DIXI-15CM", 16, self.lib), "DIXI-18CM")
        self.assertTrue(_smallest_covering("DIXI-15CM", 16, self.lib).endswith("CM"))

    def test_returns_exact_model_on_exact_count(self):
        # _smallest_covering is unconditional "smallest model >= n" — the
        # "only bump if the pick under-covers" guard is in pick_electrode_model,
        # not here. An exact ladder count returns that model.
        self.assertEqual(_smallest_covering("DIXI-5AM", 15, self.lib), "DIXI-15AM")
        self.assertEqual(_smallest_covering("DIXI-5AM", 11, self.lib), "DIXI-12AM")


@unittest.skipUnless(HAVE_DEPS, "numpy / rosa_core unavailable")
class CoveringFloorEndToEndTests(unittest.TestCase):
    """The floor fires when the matcher under-picks a uniform model relative to
    the resolved-contact count; ``chain=None`` disables it."""

    def setUp(self):
        self.lib = {m["id"]: m for m in load_electrode_library()["models"]}
        self.am = [self.lib[k] for k in
                   ("DIXI-5AM", "DIXI-8AM", "DIXI-10AM", "DIXI-12AM",
                    "DIXI-15AM", "DIXI-18AM")]
        # 12-contact comb; restrict the matcher to short candidates so it
        # under-picks, then the floor (using the full library) must rescue.
        self.arcs_pk = [8.0 + i * _PITCH for i in range(12)]
        self.arcs, self.sig = _comb(self.arcs_pk)
        self.short_candidates = [self.lib["DIXI-5AM"], self.lib["DIXI-8AM"]]

    def test_floor_fires_on_underpick(self):
        pred, branch, _mf, diag = pick_electrode_model(
            self.arcs, self.sig, self.short_candidates, self.lib,
            bolt_end_arc=5.0, profile_end_arc=50.0,
            chain=_chain(self.arcs_pk),
        )
        self.assertEqual(branch, "matched_filter_covering_floor")
        self.assertEqual(self.lib[pred]["contact_count"], 12)   # bumped to 12AM
        self.assertEqual(pred, "DIXI-12AM")
        self.assertEqual(diag["covering_floor_n_resolved"], 12)

    def test_chain_none_disables_floor(self):
        pred, branch, _mf, diag = pick_electrode_model(
            self.arcs, self.sig, self.short_candidates, self.lib,
            bolt_end_arc=5.0, profile_end_arc=50.0,
            chain=None,
        )
        self.assertNotEqual(branch, "matched_filter_covering_floor")
        # Without the floor the pick stays among the (short) candidates.
        self.assertIn(pred, {"DIXI-5AM", "DIXI-8AM"})
        self.assertNotIn("covering_floor_from", diag)


if __name__ == "__main__":
    unittest.main()
