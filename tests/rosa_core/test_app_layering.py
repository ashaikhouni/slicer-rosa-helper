"""Architecture guard: the ENGINE must never import the APP.

The desktop app (``app/rosa_service``) depends on the engine (``rosa_core`` /
``rosa_detect`` / ``rosa_agent`` / ``shank_core``) — never the reverse.
Enforcing the dependency direction now keeps the eventual "app → its own repo"
split mechanical instead of a rewrite. Pure static scan (no FastAPI), so it
runs in hosted CI where the app deps aren't installed.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENGINE_ROOTS = [
    REPO / "CommonLib" / "rosa_core",
    REPO / "CommonLib" / "rosa_detect",
    REPO / "CommonLib" / "shank_core",
    REPO / "cli" / "rosa_agent",
]
# `import rosa_service` / `from rosa_service ...` (and the reserved rosa_app).
_APP_IMPORT = re.compile(r"^\s*(?:from|import)\s+(?:rosa_service|rosa_app)\b", re.M)


class EngineDoesNotImportAppTests(unittest.TestCase):
    def test_no_engine_file_imports_the_app(self):
        offenders = []
        for root in ENGINE_ROOTS:
            if not root.exists():
                continue
            for py in root.rglob("*.py"):
                if "__pycache__" in py.parts:
                    continue
                if _APP_IMPORT.search(py.read_text(encoding="utf-8", errors="replace")):
                    offenders.append(str(py.relative_to(REPO)))
        self.assertEqual(
            offenders, [],
            "engine files import the app package (forbidden — the app depends "
            f"on the engine, not the reverse): {offenders}",
        )


if __name__ == "__main__":
    unittest.main()
