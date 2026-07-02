# ROSA desktop app (`rosa-app`)

The clinician-facing desktop app: a local **FastAPI service** + (later) an
**Electron shell** + a **web UI** (wizard: drop → confirm → run → review & edit
→ export), over the headless **`rosa-agent` engine**.

## Why it lives here (for now)

This app is developed **in-repo** as a subdirectory while the engine ↔ app
contract is still being designed — so engine + app changes can land atomically
and the API can emerge from real use. It is deliberately isolated (its own
package, `pyproject.toml`, and tests) so it splits cleanly into its own repo
once the contract stabilises. All app code lives under `app/`, so
`git subtree split --prefix app` extracts it with history when that time comes.

**Hard rule (enforced by `tests/rosa_core/test_app_layering.py`):** the app
depends on the engine; the **engine never imports the app**. Keep it that way —
it's what makes the future split mechanical.

## Architecture

```
engine (repo root)                 app/ (this dir)
  rosa_core / rosa_detect  ◀────────  rosa_service/  (FastAPI = the UI's contract)
  rosa_agent (CLI)                     web/          (wizard + review/edit UI, later)
        │  published as `rosa-agent`   electron/     (desktop shell, later)
        └──────────────── depends on ──┘
```
The UI talks **only** to the `rosa_service` HTTP+JSON API, never to the engine
CLI directly — so the engine can change without breaking the UI.

## Dev setup (in-repo, editable)

Install the engine first, then this app (order matters — the app declares
`rosa-agent` as a dependency, satisfied by the local editable engine):

```bash
pip install -e .           # from repo root: the rosa-agent engine
pip install -e ./app       # this app (rosa-app)
```

Run the local service:

```bash
rosa-app-serve             # binds 127.0.0.1, picks a free port, prints the URL as JSON
python -m rosa_service     # equivalent
```

Test the app (needs the `[test]` extra — `fastapi` + `httpx`):

```bash
pip install -e './app[test]'
pytest app/tests
```

## Non-negotiables (carried from the plan)

- Bind **127.0.0.1** only; PHI never leaves the machine; no telemetry / no
  runtime network calls in the shipped app (weights pre-seeded).
- Jobs run as subprocesses (cancellable, crash-isolated); every run writes an
  auditable manifest.
- Research-only / not FDA-cleared.

## Status

Scaffold: `/healthz` proving the engine link. Job endpoints, the ReviewDoc DTO,
SSE logs, static viewer serving, and the Electron shell land in later phases.
