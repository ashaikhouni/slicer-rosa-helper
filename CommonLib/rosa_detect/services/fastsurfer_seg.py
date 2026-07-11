"""FastSurfer seg-only backend — a whole-brain anatomical segmentation from a T1.

Runs FastSurfer's VINN segmentation (``FastSurferCNN/run_prediction.py``, no
FreeSurfer, no surface pipeline) to produce an ``aparc.DKTatlas+aseg.deep``
segmentation. That learned segmentation drives two things the classical Otsu
path can't do well:

  * a clean cortical **surface** (dura is simply unlabelled), and
  * subject-specific **labels** (cortex DKT + subcortical aseg) for contacts.

Detection mirrors the SynthStrip backend: it's an *external* runtime (torch +
monai) invoked as a subprocess, auto-detected and optional — callers fall back
to Otsu / the MNI atlases when it isn't present. Configure via env:

  * ``ROSA_FASTSURFER_DIR``    — the FastSurfer checkout (has FastSurferCNN/).
  * ``ROSA_FASTSURFER_PYTHON`` — a python with torch + monai (its env).

or a bundled default (``<dir>`` beside this package, filled in at packaging).
The runtime freezes into a ~545 MB PyInstaller bundle (verified, runs on MPS),
so this can ship in-app or stay BYO — same code path either way.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable


_ASEG_REL = ("mri", "aparc.DKTatlas+aseg.deep.mgz")


def find_fastsurfer() -> tuple[Path | None, str | None]:
    """Return ``(fastsurfer_dir, python)`` if a usable FastSurfer is found, else
    ``(None, None)``. Checks ``ROSA_FASTSURFER_DIR`` / ``ROSA_FASTSURFER_PYTHON``,
    then ``FASTSURFER_HOME``, then a bundled ``vendor/FastSurfer`` sibling."""
    candidates = [
        os.environ.get("ROSA_FASTSURFER_DIR"),
        os.environ.get("FASTSURFER_HOME"),
        str(Path(__file__).resolve().parents[3] / "vendor" / "FastSurfer"),
    ]
    py = os.environ.get("ROSA_FASTSURFER_PYTHON") or shutil.which("python3") or shutil.which("python")
    for c in candidates:
        if not c:
            continue
        d = Path(c)
        if (d / "FastSurferCNN" / "run_prediction.py").is_file():
            return d, py
    return None, None


def _pick_device(device: str) -> str:
    """Choose the inference device WITHOUT importing torch in this (parent)
    process — the parent uses SimpleITK, and torch + SITK both load libomp,
    which segfaults on a double-init. torch lives only in the FastSurfer
    subprocess, which validates the device itself; here we just guess by
    platform (Apple Silicon → mps, else cpu — override with device=).

    ``ROSA_FASTSURFER_DEVICE`` forces the choice, which is required when the
    FastSurfer env is a CPU-only torch build (e.g. the ``fastsurfer_cpu`` env
    from FastSurfer's ``*_env_cpu.yml``): the parent may be arm64 and would
    otherwise pick ``mps``, but ``find_device('mps')`` *raises* on a build
    without an MPS backend rather than falling back to CPU."""
    if device == "auto":
        device = os.environ.get("ROSA_FASTSURFER_DEVICE", "auto")
    if device != "auto":
        return device
    import platform
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mps"
    return "cpu"


def run_fastsurfer_seg(
    t1_path: str | Path,
    out_dir: str | Path,
    *,
    sid: str = "subject",
    device: str = "auto",
    timeout_s: float = 1200.0,
    log: Callable[[str], None] = lambda _m: None,
) -> Path | None:
    """Run FastSurfer VINN seg-only on ``t1_path``; return the aseg path.

    Reuses a cached aseg under ``out_dir/sid/`` when present (segmentation is
    ~30 s on GPU but atlas-independent, so once per case). Returns ``None`` if
    FastSurfer isn't available or the run fails — callers must fall back.
    """
    aseg = Path(out_dir) / sid / Path(*_ASEG_REL)
    if aseg.is_file():
        log(f"[fastsurfer] reusing cached segmentation ({aseg.name})")
        return aseg

    fdir, py = find_fastsurfer()
    if fdir is None or py is None:
        log("[fastsurfer] not found — set ROSA_FASTSURFER_DIR + ROSA_FASTSURFER_PYTHON "
            "(falling back to the classical path)")
        return None

    dev = _pick_device(device)
    cmd = [str(py), str(fdir / "FastSurferCNN" / "run_prediction.py"),
           "--t1", str(t1_path), "--sid", sid, "--sd", str(out_dir), "--device", dev]
    # A few ops FastSurfer uses (view-aggregation) have no MPS kernel; let torch
    # fall those back to CPU instead of raising. Harmless on cpu/cuda. Without
    # this, an mps run dies partway; with it, mps matches cpu to ~7 voxels and
    # runs ~15x faster (verified: 100% overall label agreement).
    env = {**os.environ, "PYTHONPATH": str(fdir), "KMP_DUPLICATE_LIB_OK": "TRUE",
           "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    log(f"[fastsurfer] VINN seg-only on {Path(t1_path).name} (device={dev})…")
    try:
        subprocess.run(cmd, check=True, env=env, timeout=timeout_s,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:  # noqa: BLE001
        log(f"[fastsurfer] run failed ({exc}); falling back")
        return None
    if aseg.is_file():
        log(f"[fastsurfer] segmentation ready ({aseg.name})")
        return aseg
    log("[fastsurfer] run produced no aseg; falling back")
    return None


__all__ = ["find_fastsurfer", "run_fastsurfer_seg"]
