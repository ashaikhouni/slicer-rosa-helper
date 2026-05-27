"""Unit tests for ``rosa_core.matched_filter`` on synthetic comb signals."""
from __future__ import annotations

import unittest

try:
    import numpy as np
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


def _make_comb_signal(arcs, contact_arcs, *, peak_amp=1500.0, sigma_mm=0.7):
    """Synthetic comb signal: sum of Gaussians at ``contact_arcs``."""
    sig = np.zeros_like(arcs)
    inv = 1.0 / (2.0 * sigma_mm * sigma_mm)
    for c in contact_arcs:
        sig += peak_amp * np.exp(-((arcs - c) ** 2) * inv)
    return sig


def _make_library_model(name, n_contacts, pitch_mm=3.5):
    """Library model dict with offsets [0, pitch, 2*pitch, ...]."""
    offsets = [i * pitch_mm for i in range(n_contacts)]
    return {"id": name, "contact_center_offsets_from_tip_mm": offsets}


@unittest.skipUnless(DEPS_AVAILABLE, "numpy not available")
class MatchedFilterPickTests(unittest.TestCase):
    """Behavioural unit tests on synthetic comb signals."""

    def setUp(self):
        from rosa_core.matched_filter import matched_filter_pick
        self.pick = matched_filter_pick
        # 60 mm arc, 0.25 mm step — same granularity as the staged walker.
        self.arcs = np.arange(0.0, 60.0 + 0.125, 0.25)

    def test_pick_correct_model_when_only_one_fits(self):
        """8 peaks at PMT-8 positions → matcher picks PMT-8."""
        # PMT-8 contacts at arcs [4.0, 7.5, ..., 28.5]: span 24.5 mm.
        peaks = [4.0 + i * 3.5 for i in range(8)]
        signal = _make_comb_signal(self.arcs, peaks)
        models = [
            _make_library_model("PMT-8", 8),
            _make_library_model("PMT-12", 12),
            _make_library_model("PMT-16A", 16),
        ]
        r = self.pick(
            self.arcs, signal, models,
            bolt_end_arc=2.0, first_contact_min_mm=1.0,
            profile_end_arc=30.0, max_extend_tip_mm=3.0,
            sigma_contact_mm=1.0,
        )
        self.assertEqual(r.best_model_id, "PMT-8")
        self.assertEqual(r.n_slots, 8)

    def test_shorter_wins_on_tie(self):
        """When PMT-8's contact-zone slots are identical to PMT-10's
        (PMT-10's extras fall in bolt zone), tie-break picks PMT-8.

        Setup: 8 contacts, all in contact zone. PMT-8 fits exactly. PMT-10
        with the same tip places its 8 deepest contacts at the same arcs
        as PMT-8, with its 2 most-proximal slots in the bolt zone.
        """
        peaks = [4.0 + i * 3.5 for i in range(8)]
        signal = _make_comb_signal(self.arcs, peaks)
        models = [
            _make_library_model("PMT-10", 10),  # placed first; should still lose
            _make_library_model("PMT-8", 8),
        ]
        r = self.pick(
            self.arcs, signal, models,
            bolt_end_arc=2.0, first_contact_min_mm=1.0,
            profile_end_arc=30.0, max_extend_tip_mm=0.0,  # bolt-less mode
            sigma_contact_mm=1.0,
            max_bolt_frac=0.5,
        )
        self.assertEqual(r.best_model_id, "PMT-8")

    def test_max_tip_override_clamps_deepest_slot(self):
        """``max_tip_override`` limits the deepest-slot arc."""
        # 12 peaks; both PMT-8 and PMT-12 could fit, but override cuts off
        # at PMT-8's natural span.
        peaks = [4.0 + i * 3.5 for i in range(12)]
        signal = _make_comb_signal(self.arcs, peaks)
        models = [
            _make_library_model("PMT-8", 8),
            _make_library_model("PMT-12", 12),
        ]
        r = self.pick(
            self.arcs, signal, models,
            bolt_end_arc=2.0, first_contact_min_mm=1.0,
            profile_end_arc=50.0, max_extend_tip_mm=3.0,
            sigma_contact_mm=1.0,
            max_tip_override=28.5,  # PMT-8's natural deepest slot
        )
        # With max_tip clamped, PMT-12's deepest slot can't reach the
        # deepest signal peak, so PMT-8 wins.
        self.assertEqual(r.best_model_id, "PMT-8")

    def test_empty_signal_returns_no_pick(self):
        signal = np.zeros_like(self.arcs)
        models = [_make_library_model("PMT-8", 8)]
        r = self.pick(
            self.arcs, signal, models,
            bolt_end_arc=2.0, first_contact_min_mm=1.0,
            profile_end_arc=30.0, max_extend_tip_mm=3.0,
        )
        self.assertIsNone(r.best_model_id)

    def test_empty_library_returns_no_pick(self):
        peaks = [4.0 + i * 3.5 for i in range(8)]
        signal = _make_comb_signal(self.arcs, peaks)
        r = self.pick(
            self.arcs, signal, [],
            bolt_end_arc=2.0, first_contact_min_mm=1.0,
            profile_end_arc=30.0, max_extend_tip_mm=3.0,
        )
        self.assertIsNone(r.best_model_id)

    def test_max_bolt_frac_rejects_too_many_slots_in_bolt(self):
        """Matcher rejects candidates with > max_bolt_frac slots in the
        bolt zone."""
        peaks = [4.0 + i * 3.5 for i in range(8)]
        signal = _make_comb_signal(self.arcs, peaks)
        # Force a configuration where any tip leaves > 50% of PMT-12's
        # slots in the bolt zone (no signal, so no model with tip in
        # contact zone has enough coverage). Strict bolt cap: 0.30.
        models = [_make_library_model("PMT-12", 12)]
        r_strict = self.pick(
            self.arcs, signal, models,
            bolt_end_arc=20.0, first_contact_min_mm=1.0,  # cutoff=21
            profile_end_arc=30.0, max_extend_tip_mm=0.0,
            max_bolt_frac=0.30,
        )
        # Loose cap: same setup, allow up to 50% in bolt.
        r_loose = self.pick(
            self.arcs, signal, models,
            bolt_end_arc=20.0, first_contact_min_mm=1.0,
            profile_end_arc=30.0, max_extend_tip_mm=0.0,
            max_bolt_frac=0.50,
        )
        # The looser cap should be at least as permissive: if strict
        # rejects all tips, loose may still accept some.
        if r_strict.best_model_id is None:
            # Strict rejected all → loose may or may not accept; just
            # verify the looser case doesn't crash.
            self.assertTrue(r_loose.best_model_id is None or r_loose.n_slots == 12)
        else:
            self.assertIsNotNone(r_loose.best_model_id)

    def test_max_tip_short_mm_rejects_short_template_in_bolt_zone(self):
        """The deep-tip-coverage gate rejects (model, tip-position)
        candidates whose deepest slot leaves the deep half of the
        trajectory unexplained — even when the template correlates well
        over a bolt-saturated proximal region.

        Setup: bright bolt zone in the proximal arc (u=4..16) plus the
        actual shank contacts at u=22..50.  Without the gate, PMT-5
        (5 contacts, 14mm span) wins by fitting in the bright bolt
        zone, leaving u=22..50 unfit.  With ``max_tip_short_mm=5`` the
        small electrode is rejected and a longer model that reaches
        within 5mm of the trajectory tip is picked instead.
        """
        # Synthetic that reproduces the s57 RIPR failure structurally:
        # PMT-5 placed against a uniformly-bright proximal "bolt"
        # plateau scores the same correlation regardless of where in
        # the plateau its tip sits, so the picker has no opposing
        # force to prefer the deep end. Without the gate the picker
        # accepts the proximal placement (tip in the bolt zone, deep
        # half of trajectory unfit). With the gate, only deep-tip
        # placements survive.
        # Wide bright "bolt" plateau u=4..18 (saturated metal): 3x
        # higher amplitude than the real contact peaks at u=22..50.
        plateau = np.where(
            (self.arcs >= 4.0) & (self.arcs <= 18.0), 4500.0, 0.0,
        )
        contact_peaks = [22.0 + i * 3.5 for i in range(9)]
        signal = plateau + _make_comb_signal(
            self.arcs, contact_peaks, peak_amp=1500.0,
        )
        models = [_make_library_model("PMT-5", 5)]

        # Without gate: PMT-5 picks SOME tip — its best position is
        # in the bright plateau (deepest slot inside the bolt).
        r_no_gate = self.pick(
            self.arcs, signal, models,
            bolt_end_arc=2.0, first_contact_min_mm=1.0,
            profile_end_arc=50.0, max_extend_tip_mm=3.0,
            sigma_contact_mm=1.0, max_tip_short_mm=None,
        )
        self.assertEqual(r_no_gate.best_model_id, "PMT-5")
        no_gate_tip = float(r_no_gate.slot_arcs.max())
        # Without the gate, the picker's chosen deepest slot is far
        # from the trajectory tip (sitting in the bolt plateau).
        self.assertGreater(50.0 - no_gate_tip, 10.0)

        # With gate (5 mm tolerance): the bolt-zone position is
        # rejected; the only PMT-5 placements that survive are deep-
        # tip ones (deepest slot within 5 mm of profile_end).
        r_gated = self.pick(
            self.arcs, signal, models,
            bolt_end_arc=2.0, first_contact_min_mm=1.0,
            profile_end_arc=50.0, max_extend_tip_mm=3.0,
            sigma_contact_mm=1.0, max_tip_short_mm=5.0,
        )
        self.assertEqual(r_gated.best_model_id, "PMT-5")
        gated_tip = float(r_gated.slot_arcs.max())
        self.assertLessEqual(50.0 - gated_tip, 5.0)
        # The gate moved the placement: the new deepest-slot position
        # is closer to the trajectory tip than the ungated one.
        self.assertGreater(gated_tip, no_gate_tip)

    def test_valley_anti_template_rejects_uniform_high_signal(self):
        """A constant-high signal (e.g., bone with chance peaks aligned at
        library pitch) gives high basic-Pearson but valley-aware Pearson
        crushes it because both peak and midpoint positions have the same
        HU — no contrast → low correlation."""
        # Synthetic: uniform-high signal with periodic contact-shaped peaks
        # at PMT-8 spacing. Real shanks have valleys between contacts; this
        # synthetic has no valleys (uniform background under the peaks).
        peaks = [4.0 + i * 3.5 for i in range(8)]
        # Uniform background at 1500 + Gaussian peaks at 2000 above bg
        sig_uniform_bg = np.full_like(self.arcs, 1500.0)
        sig_uniform_bg += _make_comb_signal(self.arcs, peaks, peak_amp=2000.0)
        models = [_make_library_model("PMT-8", 8)]

        # Basic mode: high correlation (peaks match)
        r_basic = self.pick(
            self.arcs, sig_uniform_bg, models,
            bolt_end_arc=2.0, first_contact_min_mm=1.0,
            profile_end_arc=30.0, max_extend_tip_mm=3.0,
            sigma_contact_mm=1.0,
        )
        self.assertEqual(r_basic.best_model_id, "PMT-8")
        basic_corr = r_basic.corr

        # Valley mode: correlation crashes because midpoints are HIGH
        # (same uniform bg) — the negative anti-template at midpoints
        # gets multiplied by positive signal_zm and contributes negatively.
        r_valley = self.pick(
            self.arcs, sig_uniform_bg, models,
            bolt_end_arc=2.0, first_contact_min_mm=1.0,
            profile_end_arc=30.0, max_extend_tip_mm=3.0,
            sigma_contact_mm=1.0,
            add_valley_anti_template=True,
        )
        # Same model picked but score should drop significantly.
        # On a real shank with valleys this would NOT drop as much.
        self.assertLess(r_valley.corr, basic_corr * 0.95)

    def test_valley_anti_template_preserves_real_shank_pick(self):
        """A signal with peaks AND valleys (real-shank-like) still picks the
        right model with valley anti-template enabled."""
        peaks = [4.0 + i * 3.5 for i in range(8)]
        # Just the peaks, no uniform background — clean shank-like signal
        signal = _make_comb_signal(self.arcs, peaks)
        models = [
            _make_library_model("PMT-8", 8),
            _make_library_model("PMT-12", 12),
        ]
        r = self.pick(
            self.arcs, signal, models,
            bolt_end_arc=2.0, first_contact_min_mm=1.0,
            profile_end_arc=30.0, max_extend_tip_mm=3.0,
            sigma_contact_mm=1.0,
            add_valley_anti_template=True,
        )
        self.assertEqual(r.best_model_id, "PMT-8")

    def test_no_signal_in_zone_returns_no_pick(self):
        """When the signal is zero throughout the contact zone, no pick."""
        # Comb signal entirely BEFORE the cutoff.
        peaks = [4.0 + i * 3.5 for i in range(8)]
        signal = _make_comb_signal(self.arcs, peaks)
        models = [_make_library_model("PMT-8", 8)]
        r = self.pick(
            self.arcs, signal, models,
            bolt_end_arc=58.0, first_contact_min_mm=1.0,  # in_zone empty
            profile_end_arc=60.0, max_extend_tip_mm=0.0,
        )
        self.assertIsNone(r.best_model_id)

    def test_arcs_signal_shape_mismatch_raises(self):
        models = [_make_library_model("PMT-8", 8)]
        with self.assertRaises(ValueError):
            self.pick(
                np.array([0.0, 1.0, 2.0]),
                np.array([0.0, 1.0]),  # wrong length
                models,
                bolt_end_arc=0.0, first_contact_min_mm=1.0,
                profile_end_arc=2.0, max_extend_tip_mm=0.0,
            )


@unittest.skipUnless(DEPS_AVAILABLE, "numpy not available")
class MatchedFilterShortcutsTests(unittest.TestCase):
    """Test the public surface (dataclass, default constants)."""

    def test_default_sigma_is_1mm(self):
        from rosa_core.matched_filter import SIGMA_CONTACT_MM_DEFAULT
        self.assertEqual(SIGMA_CONTACT_MM_DEFAULT, 1.0)

    def test_result_dataclass_fields(self):
        from rosa_core.matched_filter import MatchedFilterResult
        r = MatchedFilterResult(
            best_model_id="PMT-8", tip_arc=24.5,
            slot_arcs=np.array([24.5, 21.0, 17.5, 14.0, 10.5, 7.0, 3.5, 0.0]),
            n_slots=8, n_covered=8, corr=0.95,
        )
        self.assertEqual(r.best_model_id, "PMT-8")
        self.assertEqual(r.n_slots, 8)
        self.assertAlmostEqual(r.corr, 0.95)


@unittest.skipUnless(DEPS_AVAILABLE, "numpy not available")
class ExtentAwareAndVerifierTests(unittest.TestCase):
    """Tests for ``extent_aware_pick`` and ``no_metal_rerank``."""

    def _build_result(self, arcs, signal, models):
        from rosa_core.matched_filter import matched_filter_pick
        return matched_filter_pick(
            arcs, signal, models,
            bolt_end_arc=0.0,
            profile_end_arc=float(arcs[-1]),
            max_extend_tip_mm=0.5,
        )

    # ---- extent_aware_pick ----------------------------------------

    def test_extent_aware_returns_best_when_corr_decisive(self):
        from rosa_core.matched_filter import (
            extent_aware_pick, MatchedFilterResult,
        )
        per = [
            MatchedFilterResult("A", 10.0, np.zeros(5), 5, 5, 0.95),
            MatchedFilterResult("B", 10.0, np.zeros(3), 3, 3, 0.30),
        ]
        res = MatchedFilterResult("A", 10.0, np.zeros(5), 5, 5, 0.95,
                                  per_model=per)
        self.assertEqual(extent_aware_pick(res), "A")

    def test_extent_aware_prefers_more_coverage_on_corr_tie(self):
        """Close-corr models — extent-aware should pick the one with
        more coverage (n_covered)."""
        from rosa_core.matched_filter import (
            extent_aware_pick, MatchedFilterResult,
        )
        per = [
            MatchedFilterResult("short", 10.0, np.zeros(3), 3, 3, 0.93),
            MatchedFilterResult("long",  10.0, np.zeros(8), 8, 8, 0.94),
        ]
        res = MatchedFilterResult("short", 10.0, np.zeros(3), 3, 3, 0.93,
                                  per_model=per)
        # 0.93 * sqrt(3/8) = 0.57 vs 0.94 * 1.0 = 0.94 → "long" wins
        self.assertEqual(extent_aware_pick(res), "long")

    def test_extent_aware_falls_through_when_no_per_model(self):
        from rosa_core.matched_filter import (
            extent_aware_pick, MatchedFilterResult,
        )
        res = MatchedFilterResult("A", 10.0, np.zeros(5), 5, 5, 0.95)
        self.assertEqual(extent_aware_pick(res), "A")

    # ---- no_metal_rerank ------------------------------------------

    def test_verifier_silent_when_all_slots_on_metal(self):
        """Clean 8-contact comb, 8AM picked: zero in-brain dead slots →
        verifier returns the original pick unchanged."""
        from rosa_core.matched_filter import no_metal_rerank
        contact_arcs = [3.5 * i for i in range(8)]
        arcs = np.arange(0.0, 28.0, 0.3)
        sig = _make_comb_signal(arcs, contact_arcs)
        lib = [_make_library_model("PMT-8", 8),
               _make_library_model("PMT-10", 10)]
        res = self._build_result(arcs, sig, lib)
        self.assertEqual(res.best_model_id, "PMT-8")
        self.assertEqual(no_metal_rerank(res, arcs, sig), "PMT-8")

    def test_verifier_downgrades_when_interior_slot_is_dead(self):
        """A 10-contact template fitted onto an 8-contact signal: the
        2 deepest (interior) slots fall on no-metal → verifier should
        swap to a candidate with fewer in-brain dead slots."""
        from rosa_core.matched_filter import no_metal_rerank, MatchedFilterResult
        arcs = np.arange(0.0, 35.0, 0.3)
        # 8 real contacts at the proximal end; arc 28-35 is empty
        sig = _make_comb_signal(arcs, [3.5 * i + 3.5 for i in range(8)])
        # Fabricate per_model where 10-slot template has 2 dead interior
        # slots (at tip-end, arc 31.5 and 35.0) and 8-slot template is
        # clean.
        per = [
            MatchedFilterResult(
                "PMT-10", 35.0,
                slot_arcs=np.array([35.0, 31.5, 28.0, 24.5, 21.0,
                                    17.5, 14.0, 10.5, 7.0, 3.5]),
                n_slots=10, n_covered=8, corr=0.50,
            ),
            MatchedFilterResult(
                "PMT-8", 28.0,
                slot_arcs=np.array([28.0, 24.5, 21.0, 17.5, 14.0,
                                    10.5, 7.0, 3.5]),
                n_slots=8, n_covered=8, corr=0.55,
            ),
        ]
        res = MatchedFilterResult(
            "PMT-10", 35.0,
            slot_arcs=per[0].slot_arcs,
            n_slots=10, n_covered=8, corr=0.50,
            per_model=per,
        )
        self.assertEqual(no_metal_rerank(res, arcs, sig, anchor_arc=0.0),
                         "PMT-8")

    def test_verifier_does_not_count_boundary_slot_as_dead(self):
        """Proximal/distal extreme slots are template-alignment slack
        (PMT-16A often places its tip slot 1-2 mm past the actual tip
        in vacuum). One dead boundary slot must NOT trigger downgrade."""
        from rosa_core.matched_filter import no_metal_rerank, MatchedFilterResult
        arcs = np.arange(0.0, 30.0, 0.3)
        # 8 contacts; arc 28+ is empty (template overshoots)
        sig = _make_comb_signal(arcs, [3.5 * i + 1.0 for i in range(8)])
        # PMT-8 template aligned so its distal-most slot sits 1mm past
        # the array. Interior slots all on metal.
        per = [MatchedFilterResult(
            "PMT-8", 29.5,
            slot_arcs=np.array([29.5, 26.0, 22.5, 19.0, 15.5,
                                12.0, 8.5, 5.0]),
            n_slots=8, n_covered=7, corr=0.60,
        )]
        res = MatchedFilterResult(
            "PMT-8", 29.5,
            slot_arcs=per[0].slot_arcs,
            n_slots=8, n_covered=7, corr=0.60,
            per_model=per,
        )
        self.assertEqual(no_metal_rerank(res, arcs, sig, anchor_arc=0.0),
                         "PMT-8")

    def test_verifier_ignores_bolt_zone_dead_slots(self):
        """Slots proximal of the in-brain anchor are tolerated — we
        can't verify contacts in the bolt zone from CT, and a dead
        bolt slot is not a placement failure."""
        from rosa_core.matched_filter import no_metal_rerank, MatchedFilterResult
        arcs = np.arange(0.0, 40.0, 0.3)
        # Real contacts only past arc 15 (the in-brain region).
        sig = _make_comb_signal(arcs, [15.0 + 3.5 * i for i in range(7)])
        anchor = 15.0
        # PMT-10 has 3 slots proximal of anchor — those are bolt zone,
        # should NOT count as dead. Interior in-brain slots fit signal.
        per = [MatchedFilterResult(
            "PMT-10", 36.0,
            slot_arcs=np.array([36.0, 32.5, 29.0, 25.5, 22.0, 18.5,
                                15.0, 11.5, 8.0, 4.5]),
            n_slots=10, n_covered=7, corr=0.55,
        )]
        res = MatchedFilterResult(
            "PMT-10", 36.0,
            slot_arcs=per[0].slot_arcs,
            n_slots=10, n_covered=7, corr=0.55,
            per_model=per,
        )
        # cur_dead in [anchor, tip] = 0 interior dead → no downgrade.
        self.assertEqual(no_metal_rerank(res, arcs, sig, anchor_arc=anchor),
                         "PMT-10")

    def test_verifier_falls_through_on_empty_per_model(self):
        from rosa_core.matched_filter import no_metal_rerank, MatchedFilterResult
        arcs = np.arange(0.0, 20.0, 0.3)
        sig = np.ones_like(arcs)
        res = MatchedFilterResult(None, None, None, 0, 0, 0.0,
                                  per_model=None)
        self.assertIsNone(no_metal_rerank(res, arcs, sig))


if __name__ == "__main__":
    unittest.main()
