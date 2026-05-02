"""Subcommand dispatcher for ``rosa-agent`` / ``python -m rosa_agent``.

Two ways to run:

* **Installed (recommended)**: ``pip install .`` (or ``-e .``) at the
  repo root creates a ``rosa-agent`` console script and makes
  ``rosa_agent`` / ``rosa_core`` / ``rosa_detect`` / ``shank_core``
  importable normally — no PYTHONPATH gymnastics needed.

* **Repo-mode (legacy / dev iteration)**: when the package isn't
  installed, the boot path below detects that ``rosa_core`` isn't on
  sys.path and falls back to injecting ``<repo>/CommonLib`` so the
  CLI still works against a fresh checkout. This fallback is for
  developer convenience only; production use should ``pip install``.

Subcommands::

    rosa-agent load     <ros_folder>           [--out manifest.json]
    rosa-agent detect   <ct.nii> [--seeds tsv]  --out trajectories.tsv
    rosa-agent contacts <traj.tsv> <ct.nii>     --out contacts.tsv
    rosa-agent label    <contacts.tsv> [...]    --out labels.tsv
    rosa-agent pipeline <ros_folder|subj_id> --out-dir DIR [...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_commonlib_importable() -> None:
    """Repo-mode fallback for un-installed checkouts.

    When the package has been ``pip install``-ed, ``rosa_core`` is
    already importable and we leave sys.path alone. When running from
    a checkout WITHOUT install (e.g. ``python -m rosa_agent ...`` from
    cli/ on a fresh clone) and ``rosa_core`` isn't on sys.path, look
    for ``<repo>/CommonLib`` next to the cli/ package and inject it.
    """
    try:
        import rosa_core  # noqa: F401  (probe import — discarded)
        return
    except ImportError:
        pass

    here = Path(__file__).resolve()
    # cli/rosa_agent/main.py → repo root is parents[2].
    common = here.parents[2] / "CommonLib"
    if common.exists() and str(common) not in sys.path:
        sys.path.insert(0, str(common))


_ensure_commonlib_importable()


SUBCOMMANDS = {
    "load":     "rosa_agent.commands.load",
    "detect":   "rosa_agent.commands.detect",
    "contacts": "rosa_agent.commands.contacts",
    "label":    "rosa_agent.commands.label",
    "pipeline": "rosa_agent.commands.pipeline",
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        parser = argparse.ArgumentParser(prog="rosa-agent")
        parser.add_argument(
            "subcommand", choices=sorted(SUBCOMMANDS.keys()),
            help="One of: " + ", ".join(sorted(SUBCOMMANDS.keys())),
        )
        parser.print_help()
        return 0 if argv else 1

    subcmd = argv[0]
    if subcmd not in SUBCOMMANDS:
        print(
            f"unknown subcommand {subcmd!r}; choose from "
            f"{', '.join(sorted(SUBCOMMANDS.keys()))}",
            file=sys.stderr,
        )
        return 2

    import importlib
    module = importlib.import_module(SUBCOMMANDS[subcmd])
    return int(module.main(argv[1:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
