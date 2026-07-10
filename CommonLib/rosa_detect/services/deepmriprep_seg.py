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
    atlases: Iterable[str] = (),
    tissue: bool = True,
    no_gpu: bool = False,
    timeout: float | None = None,
    log: Callable[[str], None] = lambda _m: None,
) -> dict[str, Path]:
    """Run deepmriprep on a **T1** and write only the requested native-space
    outputs into ``out_dir``: the tissue maps (``p0``..``p3``) when ``tissue`` is
    set, plus each atlas in ``atlases``.

    Only the pipeline steps NEEDED for those outputs run (deepmriprep's
    ``run(output_paths=…, run_all=False)`` gates on ``needed_steps`` and warps
    only the requested atlases) — so the surface path (``tissue`` only, no
    ``atlases``) skips the nonlinear warp + all 14 atlas registrations, and
    labeling warps just the one atlas asked for. Returns ``{name: path}`` for
    every output written. Subprocess (torch isolation) + MPS-fallback/libomp
    flags. Raises :class:`DeepmriprepNotFound` / ``CalledProcessError`` /
    ``TimeoutExpired``.
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

    wanted: dict[str, str] = {}
    if tissue:
        for k in TISSUE_KEYS:            # p0 drives the surface; p1..p3 are cheap extras
            wanted[k] = str(out_dir / f"{k}.nii.gz")
    for a in atlases:
        wanted[a] = str(out_dir / f"{a}.nii.gz")
    if not wanted:                       # never a no-op; at least the tissue label
        wanted["p0"] = str(out_dir / "p0.nii.gz")

    code = _RUNNER.format(t1=repr(str(t1_path)), outputs=repr(wanted),
                          no_gpu=bool(no_gpu))
    env = {**os.environ, **_PROBE_ENV}
    what = "tissue" + (f"+{len(atlases)} atlas(es)" if atlases else " only")
    log(f"[deepmriprep] {what} on {t1_path.name} (no_gpu={no_gpu}, CPU seg — minutes)…")
    subprocess.run([py, "-c", code], check=True, timeout=timeout, env=env,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return {name: Path(p) for name, p in wanted.items() if Path(p).is_file()}


# Subprocess body: run ONLY the steps needed for the requested outputs (run_all
# =False → needed_steps(output_paths)); save_output writes them to the given
# paths. skip_unprocessed=False so a real error surfaces (non-zero exit) instead
# of being silently swallowed.
_RUNNER = (
    "from deepmriprep.preprocess import Preprocess\n"
    "Preprocess(no_gpu={no_gpu}).run({t1}, output_paths={outputs}, "
    "run_all=False, skip_unprocessed=False)\n"
)


__all__ = [
    "DeepmriprepNotFound", "find_deepmriprep", "deepmriprep_available",
    "run_deepmriprep", "DEFAULT_ATLASES", "TISSUE_KEYS",
]
