"""Unit tests for the slice-axis refit helper.

The helper takes the placed-contact RAS positions plus the seed line
endpoints and produces a refit ``(start_ras, end_ras)`` pair so the
focus slice plane passes through the actual contacts. Used by
``TrajectoryFocusController.focus_selected`` when CTV passes
``placed_contacts_ras=...``.

Importing the controller module loads its sibling Slicer-only modules,
so we stub them before exec.
"""
from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))


def _load_focus_module():
    """Import only the focus controller module, stubbing Slicer deps."""
    fake_core = types.ModuleType("rosa_core")
    fake_core.lps_to_ras_point = lambda p: list(p)
    sys.modules.setdefault("rosa_core", fake_core)
    pkg = types.ModuleType("rosa_scene")
    pkg.__path__ = []
    sys.modules.setdefault("rosa_scene", pkg)
    for sub, attr_name in [
        ("electrode_scene", "ElectrodeSceneService"),
        ("layout_service", "LayoutService"),
        ("trajectory_scene", "TrajectorySceneService"),
    ]:
        m = types.ModuleType(f"rosa_scene.{sub}")
        setattr(m, attr_name, object)
        sys.modules.setdefault(f"rosa_scene.{sub}", m)
        setattr(pkg, sub, m)
    spec = importlib.util.spec_from_file_location(
        "rosa_scene.trajectory_focus_controller",
        str(REPO_ROOT / "CommonLib" / "rosa_scene" / "trajectory_focus_controller.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rosa_scene.trajectory_focus_controller"] = mod
    spec.loader.exec_module(mod)
    return mod


_focus = _load_focus_module()


class RefitAxisThroughContactsTests(unittest.TestCase):
    def test_straight_contacts_offset_from_seed_refit_to_contact_axis(self):
        """5 contacts on the x-axis at y=0; seed offset to y=2 — the
        refit axis should land at y=0 (so the slice passes through
        contacts, not the seed line)."""
        contacts = [(i * 3.5, 0.0, 0.0) for i in range(5)]
        out = _focus._refit_axis_through_contacts(
            contacts, fallback_start=(0.0, 2.0, 0.0), fallback_end=(20.0, 2.0, 0.0),
        )
        self.assertIsNotNone(out)
        start, end = out
        np.testing.assert_allclose(start, [0.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(end, [14.0, 0.0, 0.0], atol=1e-6)

    def test_reversed_seed_flips_orientation(self):
        contacts = [(i * 3.5, 0.0, 0.0) for i in range(5)]
        out = _focus._refit_axis_through_contacts(
            contacts, fallback_start=(20.0, 2.0, 0.0), fallback_end=(0.0, 2.0, 0.0),
        )
        self.assertIsNotNone(out)
        start, end = out
        # When seed runs end→start (-x direction), refit should match.
        np.testing.assert_allclose(start, [14.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(end, [0.0, 0.0, 0.0], atol=1e-6)

    def test_single_contact_returns_none(self):
        out = _focus._refit_axis_through_contacts(
            [(0.0, 0.0, 0.0)],
            fallback_start=(0.0, 2.0, 0.0), fallback_end=(20.0, 2.0, 0.0),
        )
        self.assertIsNone(out)

    def test_coincident_contacts_return_none(self):
        out = _focus._refit_axis_through_contacts(
            [(0.0, 0.0, 0.0)] * 5,
            fallback_start=(0.0, 2.0, 0.0), fallback_end=(20.0, 2.0, 0.0),
        )
        self.assertIsNone(out)

    def test_curved_contacts_pca_best_fit(self):
        """Slightly bent electrode (cubic curve along main axis). The
        refit should give the principal axis (≈ x-direction) and span
        from first to last contact projection. Some contacts will sit
        off the slice plane — that's the limitation, not a bug."""
        contacts = [
            (i * 3.5, 0.05 * (i - 2.0) ** 2, 0.0)  # curved in y
            for i in range(5)
        ]
        out = _focus._refit_axis_through_contacts(
            contacts, fallback_start=(0.0, 0.0, 0.0), fallback_end=(20.0, 0.0, 0.0),
        )
        self.assertIsNotNone(out)
        start, end = np.asarray(out[0]), np.asarray(out[1])
        # Principal axis ≈ x.
        axis = end - start
        axis_unit = axis / np.linalg.norm(axis)
        self.assertGreater(abs(float(axis_unit[0])), 0.95)

    def test_three_d_contacts_yz_offset(self):
        """Contacts on a line in 3D (not aligned with any cardinal axis):
        verify start/end span the full contact range projected onto the
        principal axis."""
        # Line direction (1, 1, 1)/sqrt(3); contacts at t=0..4 along it.
        d = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
        origin = np.array([5.0, 5.0, 5.0])
        contacts = [(origin + t * 3.5 * d).tolist() for t in range(5)]
        out = _focus._refit_axis_through_contacts(
            contacts,
            fallback_start=tuple((origin + 1.0).tolist()),
            fallback_end=tuple((origin + 20.0 * d + 1.0).tolist()),
        )
        self.assertIsNotNone(out)
        start, end = np.asarray(out[0]), np.asarray(out[1])
        # Length along the diagonal = 4 * 3.5 = 14.
        self.assertAlmostEqual(float(np.linalg.norm(end - start)), 14.0, places=4)


if __name__ == "__main__":
    unittest.main()
