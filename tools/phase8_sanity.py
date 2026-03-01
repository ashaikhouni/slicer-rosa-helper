#!/usr/bin/env python3
"""Phase 8 cleanup sanity checks.

Checks:
- removed compatibility bridge files are absent
- no source imports the removed `rosa_slicer.workflow` path
- all Python files compile
"""

from __future__ import annotations

import os
import py_compile
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BRIDGE_DIR = ROOT / "RosaHelper" / "Lib" / "rosa_slicer" / "workflow"
SKIP_DIR_NAMES = {".git", "__pycache__"}
SKIP_SUFFIXES = {".pyc"}


def _iter_python_files():
    for path in ROOT.rglob("*.py"):
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        if path.suffix in SKIP_SUFFIXES:
            continue
        yield path


def check_bridge_removed():
    if not BRIDGE_DIR.exists():
        return []
    files = [p for p in BRIDGE_DIR.rglob("*") if p.is_file()]
    return files


def check_legacy_import_references():
    offenders = []
    pattern = re.compile(r"rosa_slicer\.workflow")
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        if path.suffix.lower() not in {".py", ".md", ".txt", ".rst"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if pattern.search(text):
            offenders.append(path)
    return offenders


def check_py_compile():
    failures = []
    count = 0
    for path in _iter_python_files():
        try:
            py_compile.compile(str(path), doraise=True)
            count += 1
        except Exception as exc:
            failures.append((path, str(exc)))
    return count, failures


def main() -> int:
    bridge_files = check_bridge_removed()
    legacy_refs = check_legacy_import_references()
    compiled_count, compile_failures = check_py_compile()

    print(f"[phase8] python files compiled: {compiled_count}")
    print(f"[phase8] bridge files remaining: {len(bridge_files)}")
    for path in bridge_files:
        print(f"  - {path}")
    print(f"[phase8] legacy bridge references: {len(legacy_refs)}")
    for path in legacy_refs:
        print(f"  - {path}")
    print(f"[phase8] compile failures: {len(compile_failures)}")
    for path, err in compile_failures[:20]:
        print(f"  - {path}: {err}")

    # Allow roadmap/progress docs to mention legacy path historically.
    allowed_legacy_ref_files = {
        ROOT / "PROGRESS.md",
        ROOT / "tools" / "phase8_sanity.py",
    }
    unexpected_legacy_refs = [p for p in legacy_refs if p not in allowed_legacy_ref_files]

    if bridge_files or compile_failures or unexpected_legacy_refs:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
