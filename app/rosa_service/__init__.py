"""rosa_service — the desktop app's local FastAPI service.

This package is the boundary layer between the web/Electron UI and the
``rosa-agent`` engine: the UI talks only to this HTTP+JSON API, never to the
engine CLI directly. Keep imports one-directional — this app may import the
engine (``rosa_core`` / ``rosa_agent``); the engine must never import this.
"""

__all__ = ["create_app"]

from .app import create_app
