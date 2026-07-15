"""Multi-call entry point for the frozen ROSA sidecar (one PyInstaller binary).

One binary plays two roles so the FastAPI service can re-invoke the compute
engine as a subprocess of *itself* (no separate python needed in the packaged
app):

    rosa-sidecar serve            → run the FastAPI service (prints {"url","port"})
    rosa-sidecar engine <subcmd>  → run the rosa_agent CLI (detect/contacts/label/…)

PyInstaller sets ``sys.frozen = True`` and ``sys.executable = <this binary>``, so
``app/rosa_service/jobs.py`` spawns ``[sys.executable, "engine", <subcmd>, …]`` in
the packaged app (see ``_engine_base``), which lands back here in ``engine`` mode.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    # Live, per-line stdout so the service's SSE log stream stays responsive
    # (the frozen engine has no `-u`; reconfigure instead of relying on env).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except Exception:  # noqa: BLE001 — best effort; older/patched streams
            pass

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: rosa-sidecar {serve|engine} ...", file=sys.stderr)
        return 2

    mode, rest = argv[0], argv[1:]
    if mode == "serve":
        from rosa_service.__main__ import main as serve_main
        return int(serve_main(rest) or 0)
    if mode == "engine":
        from rosa_agent.main import main as engine_main
        return int(engine_main(rest) or 0)

    print(f"unknown mode {mode!r}; use 'serve' or 'engine'", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
