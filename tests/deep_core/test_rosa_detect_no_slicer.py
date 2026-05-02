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


if __name__ == "__main__":
    unittest.main()
