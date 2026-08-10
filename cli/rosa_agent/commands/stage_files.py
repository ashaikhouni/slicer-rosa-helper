"""rosa-agent stage-files — copy inputs into a job dir (frozen-safe staging).

The app's ``import`` job stages a pre-computed localization (``contacts.tsv`` +
``trajectories.tsv``) into the job workdir so downstream steps read the same
paths as a pipeline job. This is a real subcommand — NOT a ``python -c`` step —
because in the frozen multi-call sidecar ``sys.executable`` is the packaged
binary, which only understands ``serve`` / ``engine`` (so ``rosa-sidecar.exe -c
...`` fails); a subcommand routes through the ``engine`` dispatch on every OS.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rosa-agent stage-files",
        description="Copy SRC files into --out-dir under given names (frozen-safe).",
    )
    parser.add_argument("--out-dir", required=True, help="destination directory (created if missing)")
    parser.add_argument(
        "--copy", action="append", nargs=2, metavar=("SRC", "NAME"), default=[],
        help="copy SRC to <out-dir>/NAME; repeatable. NAME is a bare filename, "
             "not a path (kept in the job dir).",
    )
    args = parser.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for src, name in args.copy:
        if Path(name).name != name:
            print(f"error: --copy NAME must be a bare filename, got {name!r}", file=sys.stderr)
            return 2
        src_p = Path(src)
        if not src_p.is_file():
            print(f"error: source not found: {src}", file=sys.stderr)
            return 2
        shutil.copyfile(src_p, out / name)
        print(f"[stage-files] {src} -> {out / name}", file=sys.stderr)
    print(f"staged {len(args.copy)} file(s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
