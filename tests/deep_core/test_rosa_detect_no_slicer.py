"""Pin that ``rosa_detect`` has no Slicer / VTK / Qt deps.

The package's docstring claims "no Slicer / VTK / Qt dependencies in
any module here — only NumPy, SciPy, SimpleITK". Until the 2026-05-02
boundary-cleanup, ``rosa_detect.service`` lazy-imported ``vtk`` for
its ``volume_node_geometry`` / ``image_from_volume_node`` helpers,
making the claim a half-truth. Those helpers now live in the proper
home: ``rosa_scene.sitk_volume_adapter`` (Slicer-side adapter that
translates volume nodes into the plain inputs ``rosa_detect`` actually
consumes).

Subprocess-isolated to defeat earlier test runs that may have imported
vtk for other reasons.
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
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(COMMONLIB), "PATH": ""},
    )
    return proc.returncode, proc.stdout, proc.stderr


class RosaDetectBoundaryTests(unittest.TestCase):
    def test_import_rosa_detect_does_not_pull_vtk_or_slicer(self):
        """``import rosa_detect`` (and reaching into service) must NOT
        cause vtk / slicer / Qt to land in sys.modules.
        """
        code = textwrap.dedent("""
            import sys
            import rosa_detect                    # package init
            from rosa_detect.service import (     # the file the user flagged
                run_contact_pitch_v1,
                stamp_ijk_to_ras_on_sitk,
                load_image_and_matrices,
            )
            from rosa_detect import (
                guided_fit_engine,                # also full module, just in case
                contact_pitch_v1_fit,
            )
            tainted = sorted(
                m for m in sys.modules
                if m == "vtk" or m.startswith("vtk.")
                   or m == "slicer" or m.startswith("slicer.")
                   or m == "qt" or m.startswith("qt.")
                   or m == "ctk" or m.startswith("ctk.")
                   or m.startswith("MRMLCorePython")
                   or m.startswith("qSlicer")
            )
            print("TAINTED:", tainted)
            print("OK" if not tainted else "LEAK")
        """).strip()
        rc, stdout, stderr = _run_in_subprocess(code)
        self.assertEqual(rc, 0, f"subprocess crashed: {stdout=!r} {stderr=!r}")
        self.assertIn("OK", stdout, f"unexpected leak: {stdout} / {stderr}")

    def test_import_rosa_detect_does_not_pull_numpy_eagerly(self):
        """``import rosa_detect`` (alone — no submodule access) must NOT
        cause numpy / scipy / SimpleITK to land in sys.modules.

        The package's docstring says it depends on those numerical libs,
        but eager-loading them in __init__ means ``import rosa_detect``
        for type-only access (``DetectionContext``, ``DetectedTrajectory``)
        fails in any minimal environment that ships only the lightweight
        types but not the full scientific stack. The lazy ``__getattr__``
        in __init__.py defers heavy imports until the heavy attribute is
        actually used; this test pins that.
        """
        code = textwrap.dedent("""
            import sys

            # Reach into pure-Python types only — lazy __getattr__ should
            # not fire for these (they're eager-imported from contracts /
            # diagnostics, both stdlib).
            import rosa_detect
            from rosa_detect import (
                DetectionContext, DetectedTrajectory, DetectionResult,
                trajectory_path_points,
            )

            tainted = sorted(
                m for m in sys.modules
                if m == "numpy" or m.startswith("numpy.")
                   or m == "scipy" or m.startswith("scipy.")
                   or m == "SimpleITK" or m.startswith("SimpleITK.")
                   or m.startswith("rosa_detect.service")
                   or m.startswith("rosa_detect.contact_pitch_v1_fit")
                   or m.startswith("rosa_detect.guided_fit_engine")
            )
            # Show count + first few so a leak is debuggable.
            print(f"TAINTED_COUNT: {len(tainted)}")
            print(f"TAINTED_SAMPLE: {tainted[:5]}")
            print("OK" if not tainted else "LEAK")
        """).strip()
        rc, stdout, stderr = _run_in_subprocess(code)
        self.assertEqual(rc, 0, f"subprocess crashed: {stdout=!r} {stderr=!r}")
        self.assertIn("OK", stdout, f"eager heavy import: {stdout} / {stderr}")

    def test_lazy_submodule_access_still_works(self):
        """Accessing a heavy attribute via ``rosa_detect.<name>`` should
        trigger the lazy import on demand. Pin both the
        ``rosa_detect.run_contact_pitch_v1`` callable form and
        ``rosa_detect.contact_pitch_v1_fit`` whole-module form.
        """
        code = textwrap.dedent("""
            import sys
            import rosa_detect

            # Whole-module form (some probes / tests do this).
            mod = rosa_detect.contact_pitch_v1_fit
            assert mod.__name__ == "rosa_detect.contact_pitch_v1_fit"

            # Callable form (Slicer / CLI entry).
            fn = rosa_detect.run_contact_pitch_v1
            assert callable(fn)

            # After lazy access, the submodule is loaded.
            assert "rosa_detect.contact_pitch_v1_fit" in sys.modules
            assert "rosa_detect.service" in sys.modules
            print("OK")
        """).strip()
        rc, stdout, stderr = _run_in_subprocess(code)
        self.assertEqual(rc, 0, f"subprocess crashed: {stdout=!r} {stderr=!r}")
        self.assertIn("OK", stdout, f"lazy access broke: {stdout} / {stderr}")


if __name__ == "__main__":
    unittest.main()
