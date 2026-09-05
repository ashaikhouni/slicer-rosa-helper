"""Launch the local service. ``python -m rosa_service`` or ``rosa-app-serve``.

Binds **127.0.0.1 only** (PHI stays on the machine). Picks a free port unless
``--port`` is given, and prints the chosen URL as one JSON line on stdout so a
parent process (the Electron shell, later) can discover where the backend came
up before opening the window.
"""
from __future__ import annotations

# Single-thread the numeric stack IN THE SERVICE PROCESS, before numpy loads.
# The service does small numpy/scipy/LAPACK work — match-ros's Procrustes SVDs,
# the editor reslice — across the event loop AND the request threadpool at once.
# In the frozen macOS app, multiple OpenMP/BLAS runtimes (scipy's OpenBLAS, SITK)
# double-init under that concurrency and SIGSEGV the whole sidecar mid-request
# (crash → respawn → the UI hangs on the pending request). One thread each avoids
# it and costs nothing here (these ops are tiny). The heavy compute/ML engine
# runs as SEPARATE subprocesses that get full threading back (see jobs.py).
import os as _os
for _tvar in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_tvar, "1")

import argparse
import json
import socket


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="rosa-app-serve", description="ROSA app local service (localhost only)",
    )
    p.add_argument("--host", default="127.0.0.1",
                   help="bind host — keep on localhost (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=0,
                   help="port (0 = pick a free one)")
    args = p.parse_args(argv)

    import uvicorn
    port = args.port or _free_port()
    # Machine-readable discovery line for the parent (Electron) process.
    print(json.dumps({"url": f"http://{args.host}:{port}/", "port": port}), flush=True)
    uvicorn.run("rosa_service.app:app", host=args.host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
