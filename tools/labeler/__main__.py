"""Entry point — launches the labeler Flask server + opens browser."""
from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m labeler",
        description="SEEG GT labeler — Flask + browser tool",
    )
    parser.add_argument(
        "--dataset", required=True,
        help="path to the dataset root (contains ct/, rosa_helper_import/, T{N}/, ...)",
    )
    parser.add_argument(
        "--port", type=int, default=5057,
        help="local port (default 5057 — picked to avoid common collisions)",
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="don't auto-open the browser tab",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="enable Flask debug mode (auto-reload + verbose errors)",
    )
    args = parser.parse_args(argv)

    dataset_dir = Path(args.dataset).expanduser().resolve()
    if not dataset_dir.is_dir():
        print(f"error: dataset not found: {dataset_dir}", file=sys.stderr)
        return 2

    # Make rosa_core importable (assume tools/labeler/ sibling of CommonLib/).
    repo_root = Path(__file__).resolve().parents[2]
    common = repo_root / "CommonLib"
    if common.exists() and str(common) not in sys.path:
        sys.path.insert(0, str(common))

    from .app import create_app  # local import — keeps argparse fast
    app = create_app(dataset_dir)

    url = f"http://localhost:{args.port}/"
    print(f"[labeler] serving {dataset_dir.name} at {url}", flush=True)

    if not args.no_browser:
        def _open():
            time.sleep(0.7)  # let Flask bind the port first
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    # use_reloader=False — reloader spawns a child that re-imports the module,
    # which would re-open the browser and trip up the in-memory state cache.
    app.run(host="127.0.0.1", port=args.port,
            debug=args.debug, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
