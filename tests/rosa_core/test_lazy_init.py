"""Pin that ``rosa_core/__init__.py`` doesn't pull heavy deps eagerly.

The pure-Python policy modules (``atlas_assignment_policy``,
``transforms``, ``ros_parser``, ``types``) must be importable without
NumPy / SimpleITK / scipy installed. The package init used to do
``from .contact_fit import ...`` etc. eagerly, which dragged numpy in
as a side effect of any submodule import (a regression caught by the
user 2026-05-02). The fix is a PEP 562 ``__getattr__``-driven init.

We assert the invariant in a subprocess (clean ``sys.modules``) so a
test that earlier in the run imported numpy can't mask a regression
here.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMONLIB = REPO_ROOT / "CommonLib"


def _run_in_subprocess(code: str) -> tuple[int, str, str]:
    """Run ``code`` with COMMONLIB on PYTHONPATH, capture (rc, stdout, stderr)."""
    env_pythonpath = str(COMMONLIB)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": env_pythonpath, "PATH": ""},
    )
    return proc.returncode, proc.stdout, proc.stderr


class LazyInitTests(unittest.TestCase):
    def test_atlas_assignment_policy_imports_without_numpy_side_effect(self):
        """``from rosa_core.atlas_assignment_policy import X`` must NOT
        cause numpy / scipy / SimpleITK to land in sys.modules. The
        policy is pure-Python and used by tests that don't depend on
        numerical libs.
        """
        code = textwrap.dedent("""
            import sys
            from rosa_core.atlas_assignment_policy import (
                build_assignment_row, choose_closest_sample, collect_provider_samples,
            )
            tainted = sorted(
                m for m in sys.modules
                if m == "numpy" or m.startswith("numpy.")
                   or m == "scipy" or m.startswith("scipy.")
                   or m == "SimpleITK" or m.startswith("SimpleITK.")
                   or m == "nibabel" or m.startswith("nibabel.")
            )
            print("TAINTED:", tainted)
            print("OK" if not tainted else "LEAK")
        """).strip()
        rc, stdout, stderr = _run_in_subprocess(code)
        self.assertEqual(
            rc, 0,
            f"subprocess crashed: stdout={stdout!r} stderr={stderr!r}",
        )
        self.assertIn("OK", stdout, f"unexpected leak: {stdout} / {stderr}")

    def test_transforms_imports_without_numpy_side_effect(self):
        """``from rosa_core.transforms import X`` is also pure-Python."""
        code = textwrap.dedent("""
            import sys
            from rosa_core.transforms import lps_to_ras_point, lps_to_ras_matrix
            tainted = sorted(
                m for m in sys.modules
                if m == "numpy" or m.startswith("numpy.")
                   or m == "scipy" or m.startswith("scipy.")
                   or m == "SimpleITK" or m.startswith("SimpleITK.")
            )
            print("TAINTED:", tainted)
            print("OK" if not tainted else "LEAK")
        """).strip()
        rc, stdout, stderr = _run_in_subprocess(code)
        self.assertEqual(rc, 0, f"subprocess crashed: {stdout=!r} {stderr=!r}")
        self.assertIn("OK", stdout, f"unexpected leak: {stdout} / {stderr}")

    def test_top_level_lazy_export_still_works(self):
        """``from rosa_core import lps_to_ras_point`` must still work — the
        15+ Slicer / tools call sites depend on this convenience.
        """
        code = textwrap.dedent("""
            from rosa_core import lps_to_ras_point
            assert callable(lps_to_ras_point)
            assert lps_to_ras_point([1.0, 2.0, 3.0]) == [-1.0, -2.0, 3.0]
            print("OK")
        """).strip()
        rc, stdout, stderr = _run_in_subprocess(code)
        self.assertEqual(rc, 0, f"subprocess crashed: {stdout=!r} {stderr=!r}")
        self.assertIn("OK", stdout, f"top-level export broke: {stdout} / {stderr}")

    def test_unknown_attribute_raises_loud_error(self):
        """Unknown name should raise loudly, matching eager-import
        behavior — Python wraps ``__getattr__`` AttributeError into
        ``ImportError`` for the ``from X import Y`` syntax (same as
        the eager-import version did), and into ``AttributeError`` for
        plain ``rosa_core.bogus`` access.
        """
        code = textwrap.dedent("""
            # from-import form — Python wraps to ImportError.
            try:
                from rosa_core import this_does_not_exist
                print("UNEXPECTED_NO_RAISE")
            except ImportError as exc:
                from_import_ok = True
            else:
                from_import_ok = False

            # Attribute access — raises AttributeError directly.
            import rosa_core
            try:
                rosa_core.also_does_not_exist
                attr_ok = False
            except AttributeError:
                attr_ok = True

            print("OK" if (from_import_ok and attr_ok) else "FAIL")
        """).strip()
        rc, stdout, stderr = _run_in_subprocess(code)
        self.assertEqual(rc, 0, f"subprocess crashed: {stdout=!r} {stderr=!r}")
        self.assertIn("OK", stdout, f"wrong exception shape: {stdout} / {stderr}")


if __name__ == "__main__":
    unittest.main()
