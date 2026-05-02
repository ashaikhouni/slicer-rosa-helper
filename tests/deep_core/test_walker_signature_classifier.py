"""Unit tests for ``classify_by_walker_signature`` electrode model picker.

Walker-signature uses observed (n, pitch, span, length) jointly to
disambiguate models that share covering length but differ on pitch
or count — the failure mode of ``suggest_shortest_covering_model``
that this classifier replaces in the production pipeline.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))
sys.path.insert(0, str(REPO_ROOT / "PostopCTLocalization"))

LIBRARY_PATH = (
    REPO_ROOT / "CommonLib" / "resources" / "electrodes" / "electrode_models.json"
)


class WalkerSignatureClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rosa_detect.contact_pitch_v1_fit import (
            classify_by_walker_signature,
            suggest_shortest_covering_model,
        )
        cls.classify = staticmethod(classify_by_walker_signature)
        cls.shortest = staticmethod(suggest_shortest_covering_model)
        cls.models = json.loads(LIBRARY_PATH.read_text())["models"]

    def _id(self, n, pitch, span, length, vendors=("Dixi", "PMT")):
        out = self.classify(
            n_observed=n, pitch_observed_mm=pitch,
            contact_span_observed_mm=span,
            intracranial_length_mm=length,
            models=self.models, vendors=vendors,
        )
        self.assertIsNotNone(out)
        return out["model_id"]

    # PMT 16-contact family: same vendor + count, different pitch.
    # Old shortest-covering picked the wrong family every time.
    def test_pmt_16a_at_3_5(self):
        self.assertEqual(self._id(16, 3.50, 52.5, 56.0), "PMT-16A")

    def test_pmt_16b_at_3_97(self):
        self.assertEqual(self._id(16, 3.97, 59.5, 63.0), "PMT-16B")

    def test_pmt_16c_at_4_43(self):
        self.assertEqual(self._id(16, 4.43, 66.5, 70.0), "PMT-16C")

    # DIXI 15CM vs 18AM: different counts, similar covering length.
    def test_dixi_15cm_picks_15(self):
        self.assertEqual(self._id(15, 3.50, 68.0, 72.0), "DIXI-15CM")

    def test_dixi_18am_picks_18(self):
        self.assertEqual(self._id(18, 3.50, 59.5, 63.0), "DIXI-18AM")

    def test_dixi_15bm_picks_15bm(self):
        # 15 contacts at 3.5 mm pitch, total span 60 — distinguished
        # from 18AM only by count.
        self.assertEqual(self._id(15, 3.50, 60.0, 64.0), "DIXI-15BM")

    # DIXI MM family: non-uniform pitch, distinguished by median pitch.
    def test_dixi_mm09a33_at_3_9(self):
        self.assertEqual(self._id(9, 3.90, 31.2, 35.0), "DIXI-MM09A33")

    def test_dixi_mm09a40_at_4_8(self):
        self.assertEqual(self._id(9, 4.80, 38.4, 42.0), "DIXI-MM09A40")

    # Wire-class fallback signature: pitch=0, n=0, span=0 — score
    # function should not crash and length-only matching should pick
    # the model whose total length is closest.
    def test_wire_class_signature_does_not_crash(self):
        out = self.classify(
            n_observed=0, pitch_observed_mm=0.0,
            contact_span_observed_mm=0.0,
            intracranial_length_mm=70.0,
            models=self.models, vendors=("Dixi", "PMT"),
        )
        self.assertIsNotNone(out)

    # Vendor filter respects "Dixi" — PMT models excluded.
    def test_vendor_filter_excludes_pmt(self):
        out = self.classify(
            n_observed=16, pitch_observed_mm=3.97,
            contact_span_observed_mm=59.5,
            intracranial_length_mm=63.0,
            models=self.models, vendors=("Dixi",),
        )
        # PMT-16B is the perfect match but excluded; classifier picks
        # the closest DIXI candidate (DIXI-18AM by count proximity at
        # 3.5 pitch) but score is meaningfully worse than the unfiltered
        # PMT-16B match. This test asserts only that vendor filter
        # routes correctly and a Dixi id comes out.
        self.assertIsNotNone(out)
        self.assertTrue(out["model_id"].startswith("DIXI"))

    def test_disambiguation_beats_shortest_covering_on_pmt_16b(self):
        """Old shortest-covering on PMT-16B's signature picks PMT-16A.
        New classifier picks PMT-16B. Direct head-to-head."""
        sig = self.classify(
            n_observed=16, pitch_observed_mm=3.97,
            contact_span_observed_mm=59.5,
            intracranial_length_mm=63.0,
            models=self.models, vendors=("Dixi", "PMT"),
        )
        sho = self.shortest(63.0, self.models, vendors=("Dixi", "PMT"))
        self.assertEqual(sig["model_id"], "PMT-16B")
        self.assertNotEqual(sho["model_id"], "PMT-16B")


class StrategyLibraryFilterTests(unittest.TestCase):
    """``filter_models_for_strategy`` is the layer that enforces
    pitch-strategy → library-family restriction beyond the vendor
    prefix. When the user picks "Dixi AM (3.5 mm)" the suggestion
    must come from the 3.5 mm DIXI family only, not from DIXI-MM
    (3.9 / 4.8 / 6.1 mm).
    """

    @classmethod
    def setUpClass(cls):
        from rosa_detect.contact_pitch_v1_fit import (
            filter_models_for_strategy, classify_by_walker_signature,
        )
        cls.filter = staticmethod(filter_models_for_strategy)
        cls.classify = staticmethod(classify_by_walker_signature)
        cls.models = json.loads(LIBRARY_PATH.read_text())["models"]

    def _ids(self, strategy):
        return {m["id"] for m in self.filter(self.models, strategy)}

    def test_dixi_strategy_includes_all_3_5_mm_dixi(self):
        """Dixi 3.5 mm strategy passes the full DIXI 3.5 mm family —
        AM, BM, and CM variants — and excludes only DIXI-MM (which
        rides 3.9 / 4.8 / 6.1 mm pitches)."""
        ids = self._ids("dixi")
        # AM family at 3.5 mm pitch.
        self.assertIn("DIXI-5AM", ids)
        self.assertIn("DIXI-15AM", ids)
        self.assertIn("DIXI-18AM", ids)
        # BM / CM long-shaft variants — also 3.5 mm pitch.
        self.assertIn("DIXI-15BM", ids)
        self.assertIn("DIXI-15CM", ids)
        self.assertIn("DIXI-18CM", ids)
        # MM family excluded — different pitch.
        self.assertNotIn("DIXI-MM09A33", ids)
        self.assertNotIn("DIXI-MM09A40", ids)
        # PMT excluded by vendor.
        self.assertFalse(any(i.startswith("PMT") for i in ids))

    def test_dixi_mm_strategy_excludes_am_family(self):
        ids = self._ids("dixi_mm")
        self.assertIn("DIXI-MM09A33", ids)
        self.assertNotIn("DIXI-15AM", ids)
        self.assertNotIn("DIXI-15CM", ids)

    def test_dixi_all_includes_both_families(self):
        ids = self._ids("dixi_all")
        self.assertIn("DIXI-15CM", ids)
        self.assertIn("DIXI-MM09A33", ids)

    def test_pmt_35_excludes_pmt_16b_and_16c(self):
        ids = self._ids("pmt_35")
        # PMT-16B (3.97 mm) and PMT-16C (4.43 mm) excluded by pitch.
        self.assertNotIn("PMT-16B", ids)
        self.assertNotIn("PMT-16C", ids)
        # PMT-8 / 10 / 12 / 14 / 16A all sit at 3.5 mm.
        self.assertIn("PMT-16A", ids)

    def test_pmt_strategy_includes_full_pmt_family(self):
        ids = self._ids("pmt")
        self.assertIn("PMT-16A", ids)
        self.assertIn("PMT-16B", ids)
        self.assertIn("PMT-16C", ids)

    def test_auto_strategy_returns_all_models(self):
        out = self.filter(self.models, "auto")
        self.assertEqual(len(out), len(self.models))

    def test_dixi_strategy_picks_15cm_not_mm_on_3_5_walker(self):
        """End-to-end: a 3.5 mm 15-contact walker signature on the
        Dixi strategy should pick DIXI-15CM and never DIXI-MM09A33
        (which exists in the library but has 3.9 mm pitch and is
        excluded by the strategy library filter)."""
        filtered = self.filter(self.models, "dixi")
        out = self.classify(
            n_observed=15, pitch_observed_mm=3.50,
            contact_span_observed_mm=68.0,
            intracranial_length_mm=72.0,
            models=filtered, vendors=("Dixi",),
        )
        self.assertEqual(out["model_id"], "DIXI-15CM")


if __name__ == "__main__":
    unittest.main()
