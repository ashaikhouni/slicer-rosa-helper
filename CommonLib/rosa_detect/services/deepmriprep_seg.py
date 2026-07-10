"""deepmriprep segmentation backend — GM/WM/CSF tissue + native-space atlases.

``deepmriprep`` (https://github.com/wwu-mmll/deepmriprep, MIT) runs a T1 through
deepbet (strip) → affine register → a patch 3D-UNet tissue segmentation
(``p0`` label / ``p1`` GM / ``p2`` WM / ``p3`` CSF) → nonlinear warp → a stack of
**native-space atlas parcellations** (neuromorphometrics ≈ aparc, Schaefer,
Hammers, thalamic nuclei, …). We use it two ways:

  * **Surface** — feed the ``p0`` GM+WM region as the ``brain_tissue`` support of
    :func:`rosa_core.brain_mesh.gyral_surface_from_mri` (same hook the FastSurfer
    aseg uses), so the T1 isocontour is meshed inside deepmriprep's learned
    tissue — a crisp alternative to the FastSurfer recon or the Otsu fallback.
  * **Labeling** — its native atlas labelmaps drop straight into the existing
    ``--atlas-labelmap`` / ``atlas_vertex_colors`` coloring + contact labeling,
    no MNI warp needed.

**Mac caveat.** Only the deepbet *strip* is Metal-friendly; the tissue 3D-UNet
uses ``aten::slow_conv3d_forward``, unimplemented on MPS. So the subprocess sets
``PYTORCH_ENABLE_MPS_FALLBACK=1`` (that op runs on CPU) — it works but is
CPU-bound (~minutes), not a speed win over FastSurfer on Apple hardware. On
CUDA it is fast. ``KMP_DUPLICATE_LIB_OK=TRUE`` avoids the torch+MKL libomp abort.

Like deepbet/FastSurfer, torch runs in a **subprocess** (never in the SITK
process). Configure the interpreter via ``ROSA_DEEPMRIPREP_PYTHON`` (a python
that can ``import deepmriprep``), else the current interpreter is probed.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable, Optional

# The atlases worth warping for labeling (deepmriprep offers ~14; these are the
# useful cortical/subcortical ones). neuromorphometrics ≈ FreeSurfer aparc+aseg.
DEFAULT_ATLASES = (
    "neuromorphometrics",
    "Schaefer2018_200Parcels_17Networks_order",
    "thalamic_nuclei",
)

# Tissue maps always produced (p0 label drives the surface support).
TISSUE_KEYS = ("p0", "p1", "p2", "p3")

_PROBE_ENV = {"KMP_DUPLICATE_LIB_OK": "TRUE", "PYTORCH_ENABLE_MPS_FALLBACK": "1"}


class DeepmriprepNotFound(FileNotFoundError):
    """Raised when no python with an importable ``deepmriprep`` can be located."""


def find_deepmriprep(deepmriprep_python: str | Path | None = None) -> Optional[str]:
    """Return a python interpreter that can ``import deepmriprep``, else ``None``.

    Checks the explicit arg, ``$ROSA_DEEPMRIPREP_PYTHON``, then ``sys.executable``.
    Each candidate is probed in a subprocess with ``KMP_DUPLICATE_LIB_OK=TRUE``
    (torch + MKL numpy in one env otherwise aborts the bare import on libomp).
    """
    probe_env = {**os.environ, **_PROBE_ENV}
    seen: set[str] = set()
    for cand in (deepmriprep_python,
                 os.environ.get("ROSA_DEEPMRIPREP_PYTHON"), sys.executable):
        if not cand:
            continue
        py = shutil.which(str(cand)) or str(cand)
        if py in seen or not Path(py).exists():
            continue
        seen.add(py)
        try:
            r = subprocess.run([py, "-c", "import deepmriprep"],
                               capture_output=True, timeout=180, env=probe_env)
            if r.returncode == 0:
                return py
        except Exception:  # noqa: BLE001 — a bad candidate just isn't it
            continue
    return None


def deepmriprep_available(deepmriprep_python: str | Path | None = None) -> bool:
    """True when a deepmriprep-capable python is reachable."""
    return find_deepmriprep(deepmriprep_python) is not None


def run_deepmriprep(
    t1_path: str | Path,
    out_dir: str | Path,
    *,
    deepmriprep_python: str | Path | None = None,
    atlases: Iterable[str] = DEFAULT_ATLASES,
    no_gpu: bool = False,
    timeout: float | None = None,
    log: Callable[[str], None] = lambda _m: None,
) -> dict[str, Path]:
    """Run deepmriprep on a **T1** and write the tissue maps (``p0``..``p3``) plus
    the requested native-space ``atlases`` into ``out_dir`` as NIfTI.

    Returns a dict ``{name: path}`` of every native-space output written (matched
    to the T1 grid). Runs in a subprocess (torch isolation) with the MPS-fallback
    + libomp flags. Raises :class:`DeepmriprepNotFound` when no capable python
    exists, or ``CalledProcessError`` / ``TimeoutExpired``.
    """
    py = find_deepmriprep(deepmriprep_python)
    if py is None:
        raise DeepmriprepNotFound(
            "deepmriprep not found. `pip install deepmriprep` in an env and set "
            "ROSA_DEEPMRIPREP_PYTHON to its python."
        )
    t1_path = Path(t1_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = sorted(set(TISSUE_KEYS) | set(atlases))

    code = _RUNNER.format(t1=repr(str(t1_path)), out=repr(str(out_dir)),
                          keep=repr(keep), no_gpu=bool(no_gpu))
    env = {**os.environ, **_PROBE_ENV}
    log(f"[deepmriprep] segmenting {t1_path.name} (no_gpu={no_gpu}, CPU tissue seg — minutes)…")
    subprocess.run([py, "-c", code], check=True, timeout=timeout, env=env,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    written: dict[str, Path] = {}
    for name in keep:
        p = out_dir / f"{name}.nii.gz"
        if p.is_file():
            written[name] = p
    return written


# Subprocess body: run the full deepmriprep pipeline, save only the wanted
# native-grid outputs. skip_unprocessed=False so a real error surfaces (non-zero
# exit) instead of being silently swallowed.
_RUNNER = (
    "import nibabel as nib\n"
    "from deepmriprep.preprocess import Preprocess\n"
    "t1, out, keep = {t1}, {out}, set({keep})\n"
    "ref = nib.load(t1).shape\n"
    "res = Preprocess(no_gpu={no_gpu}).run(t1, output_paths=None, run_all=True, skip_unprocessed=False)\n"
    "for k, v in (res or {{}}).items():\n"
    "    if k in keep and hasattr(v, 'affine') and getattr(v, 'shape', None) == ref:\n"
    "        nib.save(v, f'{{out}}/{{k}}.nii.gz')\n"
)


__all__ = [
    "DeepmriprepNotFound", "find_deepmriprep", "deepmriprep_available",
    "run_deepmriprep", "DEFAULT_ATLASES", "TISSUE_KEYS",
]
