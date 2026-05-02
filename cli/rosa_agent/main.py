"""Subcommand dispatcher for ``python -m rosa_agent``.

Top-level usage::

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


# ``cli/`` is on sys.path when invoked as ``python -m rosa_agent`` (the
# package is at ``cli/rosa_agent``). The CommonLib path is added here
# so end users don't need to set PYTHONPATH manually.
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]  # cli/rosa_agent/main.py -> repo root
_LIB = _REPO / "CommonLib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))


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
