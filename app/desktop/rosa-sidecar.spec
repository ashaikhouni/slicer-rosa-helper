# PyInstaller spec — ROSA multi-call sidecar (one-dir, torch-free).
#
# Freezes app/desktop/sidecar_main.py into a single binary that runs BOTH the
# FastAPI service (`rosa-sidecar serve`) and the engine CLI (`rosa-sidecar
# engine <subcmd>`). Build from a torch-free venv:
#     pyinstaller app/desktop/rosa-sidecar.spec --noconfirm
#
# Key jobs: (1) defeat the dynamic/lazy imports PyInstaller can't see — the
# importlib subcommand dispatch in rosa_agent.main and the PEP-562 __getattr__
# lazy submodules in rosa_core/rosa_detect; (2) collect SimpleITK's native libs;
# (3) ship the package data (bundled atlases, vendored three.js, the web UI);
# (4) keep torch and the other heavy ML backends OUT.

from pathlib import Path
from PyInstaller.utils.hooks import (
    collect_submodules, collect_data_files, collect_dynamic_libs, collect_all,
)

REPO = Path(SPECPATH).resolve().parents[1]  # app/desktop -> app -> repo root

hidden, datas, binaries = [], [], []


def _submodules(pkg):
    try:
        return collect_submodules(pkg)
    except Exception:
        return []


def _tree(src_dir, dest_prefix):
    """Collect a directory tree by ABSOLUTE path → (src, dest_dir) tuples.

    Unlike ``collect_data_files(pkg, ...)`` this doesn't need ``pkg`` importable
    at build time, so it survives a wiped editable install (a /tmp-venv cleanup
    once silently dropped the whole web UI here). ``dest_prefix`` mirrors the
    package layout so runtime paths are unchanged.
    """
    src_dir = Path(src_dir)
    out = []
    for f in src_dir.rglob("*"):
        if f.is_file():
            out.append((str(f), str(Path(dest_prefix) / f.parent.relative_to(src_dir))))
    return out


# 1) Our packages — recursively, to cover importlib dispatch + PEP-562 lazy loaders.
for pkg in ("rosa_agent", "rosa_core", "rosa_detect", "shank_core", "rosa_service"):
    hidden += _submodules(pkg)

# 2) The uvicorn/FastAPI stack (lifecycle/protocol modules are imported by string).
uv_d, uv_b, uv_h = collect_all("uvicorn")
datas += uv_d; binaries += uv_b; hidden += uv_h
for pkg in ("pydantic", "pydantic_core", "starlette", "fastapi", "anyio",
            "click", "h11", "multipart", "sniffio"):
    hidden += _submodules(pkg)

# 3) Package data: bundled atlases + LUTs, vendored three.js, the web UI.
datas += collect_data_files("rosa_core", includes=["resources/**/*"])
datas += collect_data_files("rosa_agent", includes=["commands/viewer_assets/**/*"])
# The web UI is the SPA that IS the app — collect it by absolute path so a
# non-importable rosa_service (e.g. a wiped editable install) can never silently
# drop it (which 404s every page). Falls back to collect_data_files if present.
_web = _tree(REPO / "app" / "rosa_service" / "web", "rosa_service/web")
datas += _web or collect_data_files("rosa_service", includes=["web/**/*"])

# 4) SimpleITK native libs + skimage lazy submodules.
binaries += collect_dynamic_libs("SimpleITK")
datas += collect_data_files("SimpleITK")
hidden += _submodules("skimage")

# 4b) onnxruntime — torch-free ONNX inference for the bundled deepbet skull-strip
# (its native libonnxruntime .dylib + capi come via collect_all). The deepbet
# ONNX weights themselves ride along in rosa_core/resources (collected above).
# We only use InferenceSession, so drop the optional tooling (transformers model
# zoo, quantization, training, tools) — smaller bundle, faster analysis/import.
ort_d, ort_b, ort_h = collect_all("onnxruntime")
_ort_drop = ("transformers", "tools", "quantization", "training")
ort_h = [h for h in ort_h if not any(f"onnxruntime.{d}" in h for d in _ort_drop)]
ort_d = [(s, dd) for (s, dd) in ort_d
         if not any(f"onnxruntime/{d}" in str(dd).replace("\\", "/") for d in _ort_drop)]
datas += ort_d; binaries += ort_b; hidden += ort_h

EXCLUDES = [
    "torch", "torchvision", "torchaudio", "tensorflow", "keras",
    "surfa", "deepbet", "fastsurfer", "deepmriprep", "monai",
    "tkinter", "PyQt5", "PyQt6", "PySide2", "PySide6",
    "IPython", "jupyter", "jupyterlab", "notebook", "pytest",
    "wandb", "nvidia", "triton",
]

a = Analysis(
    [str(REPO / "app" / "desktop" / "sidecar_main.py")],
    pathex=[str(REPO / "app"), str(REPO / "cli"), str(REPO / "CommonLib")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    excludes=EXCLUDES,
    hookspath=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="rosa-sidecar",
    console=True,
    disable_windowed_traceback=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="rosa-sidecar")
