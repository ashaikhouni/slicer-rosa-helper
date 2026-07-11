"""deepbet brain-extraction backend — a fast, MIT-licensed T1 skull-strip.

``deepbet`` (https://github.com/wwu-mmll/deepbet, MIT) is a small CNN that removes
non-brain voxels from a **T1-weighted MRI**. On Apple Silicon it runs on Metal
(MPS) in well under a second — vs FreeSurfer's ``mri_synthstrip`` which is
CUDA-or-CPU (so CPU-locked, ~100 s, on our Macs). Validated agreement with
SynthStrip on our data: Dice ~0.97 (slightly tighter — less dura).

**T1 only.** deepbet is trained on T1-weighted MRI and does NOT reliably strip
CT — the CT intracranial mask stays on SynthStrip / log-watershed / hull. Use
this for the MRI-derived mask route (``place --brain-mask`` / ``brain-extract``
on a preop MRI, then register into the CT frame).

Like the FastSurfer backend, deepbet pulls in torch, so it runs as a **subprocess**:
torch + SimpleITK both load libomp and double-initialise → SIGSEGV in a shared
process. We never import deepbet/torch in-process here. Configure via:

  * ``ROSA_DEEPBET_PYTHON`` — a python that can ``import deepbet`` (its own env).

or fall through to the current interpreter when it happens to have deepbet.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional


class DeepbetNotFound(FileNotFoundError):
    """Raised when no python with an importable ``deepbet`` can be located."""


def find_deepbet(deepbet_python: str | Path | None = None) -> Optional[str]:
    """Return a python interpreter that can ``import deepbet``, else ``None``.

    Checks, in order: the explicit ``deepbet_python`` arg, ``$ROSA_DEEPBET_PYTHON``,
    then the current ``sys.executable``. Each candidate is probed in a subprocess
    (``python -c "import deepbet"``) so we never import torch in *this* process.

    The probe sets ``KMP_DUPLICATE_LIB_OK=TRUE`` (as :func:`run_deepbet` does):
    when deepbet lives in an env that also has an MKL numpy (e.g. torch + numpy in
    one conda env), a bare ``import deepbet`` → ``import torch`` aborts with
    "OMP: Error #15 … libomp.dylib already initialized". Without the flag the
    probe would report a perfectly usable interpreter as unavailable.
    """
    probe_env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    seen: set[str] = set()
    for cand in (deepbet_python, os.environ.get("ROSA_DEEPBET_PYTHON"), sys.executable):
        if not cand:
            continue
        py = shutil.which(str(cand)) or str(cand)
        if py in seen or not Path(py).exists():
            continue
        seen.add(py)
        try:
            r = subprocess.run([py, "-c", "import deepbet"],
                               capture_output=True, timeout=120, env=probe_env)
            if r.returncode == 0:
                return py
        except Exception:  # noqa: BLE001 — a bad candidate just isn't it
            continue
    return None


def run_deepbet(
    input_path: str | Path,
    mask_path: str | Path,
    *,
    deepbet_python: str | Path | None = None,
    no_gpu: bool = False,
    timeout: float | None = None,
    log: Callable[[str], None] = lambda _m: None,
) -> Path:
    """Run deepbet on a **T1 MRI** ``input_path``; write the brain mask to
    ``mask_path`` and return it.

    ``no_gpu=False`` (default) lets deepbet use MPS/CUDA (fast); pass ``True`` to
    force CPU. Runs in a subprocess (torch isolation). Raises ``DeepbetNotFound``
    when no deepbet-capable python exists, or ``CalledProcessError`` / ``TimeoutExpired``.
    """
    py = find_deepbet(deepbet_python)
    if py is None:
        raise DeepbetNotFound(
            "deepbet not found. `pip install deepbet` in an env and set "
            "ROSA_DEEPBET_PYTHON to its python (deepbet is T1-only)."
        )
    input_path = Path(input_path).expanduser().resolve()
    mask_path = Path(mask_path).expanduser().resolve()
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    code = ("from deepbet import run_bet; "
            f"run_bet([{str(input_path)!r}], mask_paths=[{str(mask_path)!r}], "
            f"no_gpu={bool(no_gpu)})")
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    log(f"[deepbet] stripping {input_path.name} (no_gpu={no_gpu})…")
    subprocess.run([py, "-c", code], check=True, timeout=timeout, env=env,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Align the mask geometry to the input (shared with the SynthStrip backend).
    from .synthstrip import fix_mask_geometry
    fix_mask_geometry(input_path, mask_path)
    return mask_path


def deepbet_available(deepbet_python: str | Path | None = None) -> bool:
    """Convenience: True when a deepbet-capable python is reachable."""
    return find_deepbet(deepbet_python) is not None


__all__ = ["DeepbetNotFound", "find_deepbet", "run_deepbet", "deepbet_available"]
