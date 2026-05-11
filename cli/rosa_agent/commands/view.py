"""rosa-agent view — serve an export-view output dir + open a browser.

Thin ergonomic wrapper around stdlib's ``http.server`` for the "I just
ran ``export-view`` and want to look at it" case. The HTML viewer the
export command writes requires HTTP delivery — ``<script type=
'importmap'>`` + ``fetch()`` of ``scene.glb`` / ``scene_meta.json`` /
``t1_in_ct.nii.gz`` both fail under ``file://`` — so this turns the
two-step "run http.server in the right directory, then open the URL"
into one command.

Holds the server in the foreground until Ctrl-C. Picks a free port
when the preferred one is busy so you can leave the command running
across sessions without clashing with other dev servers.

No new pip dependencies — ``http.server`` and ``webbrowser`` are both
stdlib.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import socket
import sys
import threading
import webbrowser
from pathlib import Path


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def _pick_port(preferred: int) -> int:
    """Return ``preferred`` if it binds; otherwise let the OS pick one.

    Reads the actual bound port via ``getsockname`` rather than echoing
    ``preferred`` back, so passing ``0`` (the "any free port" sentinel
    that ``socket.bind`` understands) returns a real port too.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return s.getsockname()[1]
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(
    directory: Path,
    *,
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    """Serve ``directory`` over HTTP until Ctrl-C; optionally pop the browser."""
    if not directory.is_dir():
        raise SystemExit(f"directory not found: {directory}")
    if not (directory / "index.html").is_file():
        _stderr(
            f"[view] WARNING: {directory}/index.html not found — "
            f"is this an export-view output?"
        )

    actual_port = _pick_port(port)
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler,
        directory=str(directory),
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", actual_port), handler)

    url = f"http://localhost:{actual_port}/"
    _stderr(f"[view] serving {directory}")
    _stderr(f"[view] open  {url}")
    _stderr("[view] Ctrl-C to stop")

    if open_browser:
        # Defer slightly so the server is accepting connections by the
        # time the browser fetches the page.
        threading.Timer(0.4, lambda: webbrowser.open_new_tab(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _stderr("\n[view] shutting down")
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rosa-agent view",
        description=(
            "Serve an export-view output directory over HTTP "
            "and open it in your default browser."
        ),
    )
    parser.add_argument(
        "directory",
        help="Path to an export-view output dir (the one containing scene.glb + index.html).",
    )
    parser.add_argument(
        "--port", type=int, default=8765,
        help="Preferred port; falls back to an OS-assigned free port when busy (default: 8765).",
    )
    parser.add_argument(
        "--no-open", action="store_true",
        help="Don't auto-open the browser; just print the URL.",
    )
    args = parser.parse_args(argv)

    serve(
        Path(args.directory).expanduser().resolve(),
        port=int(args.port),
        open_browser=not bool(args.no_open),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
