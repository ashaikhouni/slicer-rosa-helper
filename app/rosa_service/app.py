"""FastAPI service — the desktop app's contract over the rosa-agent engine.

Scaffold stage: a ``/healthz`` endpoint that proves the engine dependency
resolves. Job endpoints (create/status/cancel), the editable ReviewDoc DTO,
SSE log streaming, and static viewer serving arrive in later phases. The
contract is versioned under ``/api/{API_VERSION}`` so the UI can pin it while
the engine evolves underneath.
"""
from __future__ import annotations

from fastapi import FastAPI

API_VERSION = "v1"


def _engine_info() -> dict:
    """Report the engine link — proves ``rosa-agent`` is importable + its version."""
    try:
        import importlib.metadata as md
        version = md.version("rosa-agent")
    except Exception:  # noqa: BLE001
        version = "unknown"
    try:
        import rosa_core  # noqa: F401  — the engine import must succeed
        engine_ok = True
    except Exception:  # noqa: BLE001
        engine_ok = False
    return {
        "engine": "rosa-agent",
        "engine_version": version,
        "engine_import_ok": engine_ok,
    }


def create_app() -> FastAPI:
    app = FastAPI(title="ROSA app service", version=API_VERSION)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "api": API_VERSION, **_engine_info()}

    return app


# Module-level app for ``uvicorn rosa_service.app:app``.
app = create_app()
