"""Smoke tests for ``rosa_core.contact_placement.context.PlacementCtx``.

The dataclass itself is data-only (no behavior). These tests guard against
accidental field-rename / default-change drift; the actual stage logic is
exercised by the per-stage tests.
"""
from __future__ import annotations

import unittest
from dataclasses import fields, replace

import numpy as np

from rosa_core.contact_placement import PlacementCtx


class PlacementCtxFieldsTests(unittest.TestCase):
    """Pin the field set so accidental renames trip a fast unit test."""

    def setUp(self):
        self.ctx = PlacementCtx(
            seed_start=np.array([0, 0, 0], dtype=float),
            seed_end=np.array([10, 0, 0], dtype=float),
            features={"ct_arr_kji": None, "ras_to_ijk_mat": np.eye(4)},
            library_models=[],
        )

    def test_required_fields_present(self):
        # Every field a stage writes must exist on the dataclass.
        names = {f.name for f in fields(self.ctx)}
        required = {
            "seed_start", "seed_end", "features", "library_models",
            "bolts",
            "seeder_confidence", "seeder_label", "seeder_model",
            "centerline", "bolt_end_arc", "bolt_source",
            "walk_arcs", "walk_signal", "signal_kind",
            "match", "placed_ras", "score_components",
        }
        missing = required - names
        self.assertFalse(missing, f"PlacementCtx is missing fields: {missing}")

    def test_defaults_for_init_fields(self):
        # Stages assume these defaults; if they change, downstream stages
        # need updates.
        self.assertEqual(self.ctx.bolt_source, "unknown")
        self.assertEqual(self.ctx.bolt_end_arc, 0.0)
        self.assertEqual(self.ctx.signal_kind, "")
        self.assertEqual(self.ctx.seeder_label, "")
        self.assertEqual(self.ctx.seeder_confidence, 0.0)
        self.assertIsNone(self.ctx.bolts)
        self.assertIsNone(self.ctx.centerline)
        self.assertIsNone(self.ctx.walk_arcs)
        self.assertIsNone(self.ctx.walk_signal)
        self.assertIsNone(self.ctx.match)
        self.assertEqual(self.ctx.placed_ras, [])
        self.assertEqual(self.ctx.score_components, {})

    def test_replace_returns_new_instance(self):
        # Stages use ``dataclasses.replace`` to write fields; verify the
        # original is untouched.
        ctx2 = replace(self.ctx, bolt_source="metal", bolt_end_arc=12.5)
        self.assertEqual(self.ctx.bolt_source, "unknown")    # unchanged
        self.assertEqual(ctx2.bolt_source, "metal")
        self.assertEqual(ctx2.bolt_end_arc, 12.5)


if __name__ == "__main__":
    unittest.main()
