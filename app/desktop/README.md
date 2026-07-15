# ROSA desktop shell (Electron)

A thin Electron shell that turns the server-side ROSA app into a **server-free
desktop app**. It does **not** fork the UI or the engine — it spawns the existing
`rosa_service` sidecar (which prints one JSON `{url, port}` line, then serves the
UI on `127.0.0.1`), health-checks it, and loads that localhost URL in a window.
The renderer runs `app/rosa_service/web/` byte-for-byte; a small
`desktop-shim.js` (loaded only under Electron) swaps browser uploads for native
file dialogs via the `window.rosaNative` bridge (see `preload.js`).

See the full plan: `merry-crafting-gizmo` (Electron packaging plan).

## Run in dev (against the repo's Python)

Requires a Python with `rosa_service` + `rosa_agent` importable (the repo's
`shankdetect`/`.venv`). No frozen binary needed — the shell spawns
`<python> -m rosa_service` with `app/` on `PYTHONPATH`.

```bash
cd app/desktop
npm install
# point at a python that has the packages (and, optionally, an existing case store)
ROSA_SIDECAR_PYTHON=/path/to/env/bin/python \
ROSA_APP_WORKDIR=/path/to/cases \
npm start
```

## Environment

| Var | Purpose |
|---|---|
| `ROSA_SIDECAR_MODE` | `dev` (spawn `python -m rosa_service`) or `packaged` (frozen binary). Auto-detected by binary presence otherwise. |
| `ROSA_SIDECAR_PYTHON` | Python interpreter for dev mode (default `python3`). |
| `ROSA_APP_WORKDIR` | Case/job store. Defaults to `<userData>/cases`. |
| `ROSA_FASTSURFER_*`, `ROSA_DEEPMRIPREP_PYTHON`, … | BYO heavy backends — passed through to the sidecar. |

## Packaged mode (later)

electron-builder bundles a PyInstaller one-dir sidecar under
`resources/rosa-sidecar/` (launcher `rosa-sidecar`); `main.js` then spawns
`rosa-sidecar serve`. See the plan's Workstream 1/2.
