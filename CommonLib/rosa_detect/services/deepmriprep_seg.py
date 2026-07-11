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

# Every atlas deepmriprep can warp to native space → (display name, coverage,
# license tier). `license_tier` is informational (all fine for academic use with
# citation); the picker shows it like the bundled atlases' non-commercial tag.
DEEPMRIPREP_ATLAS_INFO: dict[str, tuple[str, str, str]] = {
    "neuromorphometrics": ("Neuromorphometrics", "Whole-brain cortical + subcortical (≈ aparc+aseg)", "noncommercial"),
    "Schaefer2018_100Parcels_17Networks_order": ("Schaefer-100 (17nw)", "Cortical networks, 100 parcels", "permissive"),
    "Schaefer2018_200Parcels_17Networks_order": ("Schaefer-200 (17nw)", "Cortical networks, 200 parcels", "permissive"),
    "Schaefer2018_400Parcels_17Networks_order": ("Schaefer-400 (17nw)", "Cortical networks, 400 parcels", "permissive"),
    "Schaefer2018_600Parcels_17Networks_order": ("Schaefer-600 (17nw)", "Cortical networks, 600 parcels", "permissive"),
    "aal3": ("AAL3", "Automated Anatomical Labeling v3", "noncommercial"),
    "hammers": ("Hammers", "Hammersmith cortical + subcortical", "noncommercial"),
    "julichbrain": ("Jülich-Brain", "Cytoarchitectonic probabilistic", "noncommercial"),
    "anatomy3": ("SPM Anatomy v3", "Cytoarchitectonic maps", "noncommercial"),
    "lpba40": ("LPBA40", "LONI Probabilistic Brain Atlas", "noncommercial"),
    "mori": ("JHU (Mori)", "White-matter / deep GM", "permissive"),
    "suit": ("SUIT", "Cerebellum", "noncommercial"),
    "thalamic_nuclei": ("Thalamic nuclei", "Thalamic subnuclei", "noncommercial"),
    "thalamus": ("Thalamus", "Thalamus", "noncommercial"),
    "cobra": ("CoBrA", "Subcortical + cerebellar", "noncommercial"),
    "ibsr": ("IBSR", "Internet Brain Segmentation Repository", "noncommercial"),
}


def parse_deepmriprep_lut(csv_path: str | Path) -> dict[int, str]:
    """``{label_id: region_name}`` from a deepmriprep atlas CSV.

    deepmriprep ships one CSV per atlas, semicolon-delimited, with ``ROIid`` +
    ``ROIname`` columns (neuromorphometrics has a leading unnamed index column).
    Reads pure text — safe in the SITK process (no torch import).
    """
    import csv as _csv
    out: dict[int, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = _csv.reader(f, delimiter=";")
        header = next(reader, None)
        if not header:
            return out
        cols = [h.strip() for h in header]
        try:
            i_id, i_name = cols.index("ROIid"), cols.index("ROIname")
        except ValueError:
            return out
        for row in reader:
            if len(row) <= max(i_id, i_name):
                continue
            try:
                lid = int(float(row[i_id]))
            except (ValueError, TypeError):
                continue
            out[lid] = row[i_name].strip()
    return out

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
# of being silently swallowed. Then copy each atlas's label CSV next to its
# labelmap (deepmriprep ships them under data/templates/<name>.csv) so callers
# have both the labelmap and its id→name LUT in one place.
_RUNNER = (
    "import os, shutil, numpy as np, deepmriprep, torch, nibabel as nib\n"
    "from deepmriprep.preprocess import Preprocess\n"
    # MPS can't run the tissue 3D-UNet (unimplemented op, even with the fallback),
    # so force CPU unless real CUDA is present; the caller's no_gpu still wins up.
    "nog = {no_gpu} or not torch.cuda.is_available()\n"
    "outs = {outputs}\n"
    "res = Preprocess(no_gpu=nog).run({t1}, output_paths=outs, "
    "run_all=False, skip_unprocessed=False)\n"
    # deepmriprep's save_output re-encodes labelmaps as lossy uint8 + a fractional
    # scl_slope → boundary label artifacts (e.g. 42↔44 read back as an invalid 43).
    # Re-save each from the in-memory result with a clean header: labelmaps (p0 +
    # atlases) as int16, tissue probabilities (p1..p3) as float32.
    "for name, p in outs.items():\n"
    "    v = (res or {{}}).get(name)\n"
    "    if v is None or not hasattr(v, 'affine'):\n"
    "        continue\n"
    "    arr = np.asanyarray(v.dataobj)\n"
    "    if name in ('p1', 'p2', 'p3'):\n"
    "        nib.save(nib.Nifti1Image(arr.astype('float32'), v.affine), p)\n"
    "    else:\n"
    "        nib.save(nib.Nifti1Image(np.rint(arr).astype('int16'), v.affine), p)\n"
    "tdir = os.path.join(os.path.dirname(deepmriprep.__file__), 'data', 'templates')\n"
    "for name, p in outs.items():\n"
    "    csv = os.path.join(tdir, name + '.csv')\n"
    "    if os.path.isfile(csv):\n"
    "        shutil.copy(csv, (p[:-7] if p.endswith('.nii.gz') else p) + '.csv')\n"
)


__all__ = [
    "DeepmriprepNotFound", "find_deepmriprep", "deepmriprep_available",
    "run_deepmriprep", "parse_deepmriprep_lut",
    "DEFAULT_ATLASES", "TISSUE_KEYS", "DEEPMRIPREP_ATLAS_INFO",
]
