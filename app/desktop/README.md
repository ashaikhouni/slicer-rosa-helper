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

## Freeze the sidecar (PyInstaller)

`sidecar_main.py` is a **multi-call** entry: one binary runs both the service
(`rosa-sidecar serve`, prints `{url,port}`) and the engine
(`rosa-sidecar engine <subcmd>`). `app/rosa_service/jobs.py::_engine_base()`
spawns `[sys.executable, "engine", …]` when frozen, so the service re-invokes
itself — no separate Python in the packaged app.

Build in a **torch-free** venv (torch must stay out — it double-inits libomp
with SimpleITK and balloons the bundle):

```bash
python -m venv /tmp/rosa-freeze && . /tmp/rosa-freeze/bin/activate
pip install -e ".[mesh]" -e ./app matplotlib pyinstaller   # from repo root
pyinstaller app/desktop/rosa-sidecar.spec --noconfirm \
  --distpath /tmp/rosa-freeze/dist --workpath /tmp/rosa-freeze/build --clean
```

Result: `dist/rosa-sidecar/` (~307 MB, zero torch). Smoke it:

```bash
BIN=/tmp/rosa-freeze/dist/rosa-sidecar/rosa-sidecar
"$BIN" engine --help                 # all subcommands dispatch
"$BIN" serve                         # prints {url,port}; GET /healthz → 200
```

The spec (`rosa-sidecar.spec`) defeats the dynamic/lazy imports PyInstaller
can't see (the importlib subcommand dispatch + PEP-562 `__getattr__` lazy
submodules), collects SimpleITK's native libs + the package data (bundled
atlases, vendored three.js, the web UI), and **excludes** torch and the other
heavy ML backends.

## Build the .dmg (electron-builder)

Three steps: freeze the sidecar (above), stage it, package.

```bash
# 1. freeze (above) → dist/rosa-sidecar/
# 2. stage it where electron-builder expects it + ad-hoc sign (arm64 needs a sig)
mkdir -p app/desktop/resources
cp -R /tmp/rosa-freeze/dist/rosa-sidecar app/desktop/resources/rosa-sidecar
codesign --force --deep --sign - app/desktop/resources/rosa-sidecar/rosa-sidecar
# 3. package → release/ROSA-<ver>-arm64.dmg
cd app/desktop && npm install && npm run dist
```

`electron-builder.yml` bundles `resources/rosa-sidecar` as `extraResources`
(→ `Contents/Resources/rosa-sidecar/`, where `main.js` finds it) and
`after-pack.js` deep ad-hoc signs the whole bundle so it launches on Apple
Silicon. Result: `release/ROSA-<ver>-arm64.dmg` (~188 MB, torch-free, no Python
needed). Research/internal build = **un-notarized**, so first launch is
**right-click → Open**. `release/` and `resources/rosa-sidecar/` are gitignored.

Packaged, the app spawns `rosa-sidecar serve` from resources and stores cases in
`<userData>/cases`. Heavy backends (FastSurfer, …) stay BYO via `ROSA_*` env.

## Build for Windows (NSIS installer)

PyInstaller **cannot cross-compile**, so the sidecar must be frozen **on
Windows** — a `windows-latest` GitHub Actions runner or a Windows VM. The same
`rosa-sidecar.spec` works (its collectors pick up `.dll`/`.pyd` as they do
`.dylib` on macOS); all bundled deps ship `win_amd64` wheels. On a Windows box
with Python 3.10–3.12 + Node:

```powershell
# 1. freeze the torch-free sidecar → dist\rosa-sidecar\rosa-sidecar.exe
python -m venv C:\rosa-freeze
C:\rosa-freeze\Scripts\Activate.ps1
# From the REPO ROOT — the spec collects rosa_core/rosa_agent/rosa_service/
# rosa_detect/shank_core at build time, so they must be importable (an editable
# install, exactly as the macOS steps do). Without this the freeze SUCCEEDS but
# silently omits every engine subcommand + all rosa_core resources → a broken exe.
pip install -e ".[mesh]" -e ./app matplotlib pyinstaller pyinstaller-hooks-contrib onnxruntime
pyinstaller app\desktop\rosa-sidecar.spec --distpath C:\rosa-freeze\dist `
  --workpath C:\rosa-freeze\build --clean
# 2. stage it where electron-builder expects it (no code-signing needed)
mkdir app\desktop\resources
Copy-Item -Recurse C:\rosa-freeze\dist\rosa-sidecar app\desktop\resources\rosa-sidecar
# 3. package → release\ROSA Setup <ver>.exe
cd app\desktop; npm install; npm run dist:win
```

No ad-hoc sign step (that's macOS-only; `after-pack.js` no-ops off darwin).
Un-signed → users get a **SmartScreen** prompt (analogous to the un-notarized
macOS build); add an Authenticode cert to silence it. The core torch-free app is
fully functional on Windows; the optional heavy backends (FastSurfer, THOMAS,
SynthStrip) have no native Windows build and stay unavailable there.
