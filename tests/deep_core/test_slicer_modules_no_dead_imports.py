"""Slicer-side import hygiene smoke test.

Walks every .py file under the Slicer module directories
(PostopCTLocalization, ContactsTrajectoryView, ExportCenter,
NavigationBurn, ContactImport, AtlasLabeling, RosaHelper,
AtlasSources) and checks two things:

1. The file parses (catches syntax errors).
2. None of its imports reference modules that no longer exist.

The motivation is the 2026-05-10 cpfit deletion: three Slicer files
still imported `from rosa_detect.contact_pitch_v1_fit import ...` and
would have ImportError'd at first use, but the existing test suite
covers only the headless rosa_detect / rosa_core modules so the
breakage went unnoticed until the next manual Slicer launch. This
test pins "no Slicer file imports a deleted symbol" so the same
class of regression breaks CI immediately.

Bare AST walk — no `import` is actually executed (Slicer modules
do `from __main__ import qt, slicer` at top level which can't run
in a normal Python process). The check resolves each `import` /
`from X import` statement against an explicit allowlist of removed /
renamed modules.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


SLICER_MODULE_DIRS: tuple[str, ...] = (
    "PostopCTLocalization",
    "ContactsTrajectoryView",
    "ExportCenter",
    "NavigationBurn",
    "ContactImport",
    "AtlasLabeling",
    "RosaHelper",
    "AtlasSources",
)


# Modules / symbol-paths that have been removed or renamed and must
# not appear as import targets in any Slicer-side .py file. Entries
# match the leading prefix of an `import target` or
# `from target import …` statement.
DEAD_IMPORT_PREFIXES: tuple[str, ...] = (
    # The whole cpfit module was deleted on 2026-05-10. Functions /
    # constants moved to candidate_seeds.* / primitives.*.
    "rosa_detect.contact_pitch_v1_fit",
    # contact_pitch_v1_fit-as-cpfit aliases (e.g.
    # `from rosa_detect import contact_pitch_v1_fit as cpfit`).
    # The module-name portion is still
    # rosa_detect.contact_pitch_v1_fit, caught above; this entry
    # documents the alias form for readability.
    # Older retired packages — historical, never re-introduce.
    "shank_engine",
    "shank_core.detect",
    "shank_core.pipeline",
)


def _iter_slicer_python_files():
    for d in SLICER_MODULE_DIRS:
        base = REPO_ROOT / d
        if not base.is_dir():
            continue
        for p in base.rglob("*.py"):
            yield p


def _imports_in(tree: ast.AST):
    """Yield (lineno, target_string) for every import statement in `tree`.

    Captures both ``import X`` and ``from X import Y``; for the
    second form the target string is the dotted module name (X), not
    the imported symbol (Y).
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue  # relative imports are local — not a cross-pkg risk
            if node.module:
                yield node.lineno, node.module


class SlicerModuleImportHygieneTests(unittest.TestCase):
    def test_every_slicer_py_file_parses(self):
        failures: list[str] = []
        for p in _iter_slicer_python_files():
            try:
                ast.parse(p.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                failures.append(f"{p.relative_to(REPO_ROOT)}: {exc}")
        self.assertEqual(failures, [], "Slicer-side .py files failed to parse")

    def test_no_dead_imports_in_slicer_modules(self):
        """No Slicer file may import from a removed module path.

        Any failure here means a refactor deleted / renamed something
        in the headless code without updating the Slicer-side imports
        that called it. Failure message lists the offending file +
        line + target so you can fix it directly.
        """
        violations: list[str] = []
        for p in _iter_slicer_python_files():
            try:
                tree = ast.parse(p.read_text(encoding="utf-8"))
            except SyntaxError:
                continue  # parse_test will surface it
            rel = p.relative_to(REPO_ROOT)
            for lineno, target in _imports_in(tree):
                for dead in DEAD_IMPORT_PREFIXES:
                    if target == dead or target.startswith(dead + "."):
                        violations.append(f"{rel}:{lineno}: imports {target!r}")
                        break
        self.assertEqual(
            violations, [],
            "Slicer-side modules import removed / renamed targets:\n  "
            + "\n  ".join(violations),
        )


if __name__ == "__main__":
    unittest.main()
