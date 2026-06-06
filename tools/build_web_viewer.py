"""Regenerate web/viewer/index.html — the GitHub Pages drag-and-drop viewer.

Single source of truth: the viewer template in
``rosa_agent.commands.export_view`` (``_HTML_TEMPLATE``), rendered in *picker*
mode. The same template powers ``rosa-agent export-view`` / ``view`` in *served*
mode, so the render engine never diverges between the two.

Run after changing the viewer template:

    python tools/build_web_viewer.py

The committed ``web/viewer/index.html`` is what GitHub Pages serves (deployed
from ``web/`` like ``web/deidentify-ros/``). It loads the user's OWN files
client-side — nothing is uploaded, no server.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "cli"))
sys.path.insert(0, str(REPO / "CommonLib"))

from rosa_agent.commands.export_view import render_viewer_html  # noqa: E402

_NOTE = (
    "<!-- GENERATED from cli/rosa_agent/commands/export_view.py "
    "(_HTML_TEMPLATE, picker mode) by tools/build_web_viewer.py.\n"
    "     DO NOT EDIT BY HAND — re-run the script after changing the template. -->\n"
)


def main() -> int:
    out = REPO / "web" / "viewer" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    html = render_viewer_html(title="ROSA / SEEG viewer", mode="picker")
    html = html.replace("<!doctype html>", "<!doctype html>\n" + _NOTE, 1)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({len(html)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
