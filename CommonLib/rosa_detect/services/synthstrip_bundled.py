"""Vendored standalone SynthStrip — skull-strip without a FreeSurfer install.

FreeSurfer's SynthStrip ships as a *standalone* PyTorch script
(``mri_synthstrip``) that defines its own U-Net inline and depends only on
``torch`` / ``numpy`` / ``surfa`` — no FreeSurfer suite, no ``FREESURFER_HOME``,
no license key. This module fetches that script plus the openly-licensed model
weights (each pinned by SHA-256) into a local cache on first use and runs them
via the current interpreter, so the project can skull-strip CT/MRI with **zero
FreeSurfer dependency**.

Why fetch instead of commit
---------------------------
The script is FreeSurfer-origin code and the weights are ~30 MB. Rather than
vendor FreeSurfer code into this MIT/Zenodo repo, we download the pinned,
checksum-verified assets at run time (the nipreps pattern) — or accept them
pre-placed in the cache by an installer. The desktop bundle pre-populates the
cache so no network is needed when the clinician runs it. Set
``ROSA_SYNTHSTRIP_NO_DOWNLOAD=1`` to forbid network and require a pre-placed
cache; set ``ROSA_SYNTHSTRIP_CACHE`` to relocate it.

Runtime requirement
-------------------
``torch`` + ``surfa`` must be importable by the interpreter that runs the script
(``pip install 'rosa-agent[synthstrip]'``). **This module imports only stdlib**,
so importing it never pulls torch — the heavy deps live only in the subprocess
that runs the fetched script.

Licensing
---------
The SynthStrip weights are released under MIT / CC BY-4.0. Cite Hoopes et al.,
*SynthStrip: Skull-Stripping for Any Brain Image*, NeuroImage 206 (2022) 119474,
doi:10.1016/j.neuroimage.2022.119474. See ``SYNTHSTRIP_NOTICE.md`` in this
directory.
"""
from __future__ import annotations

import functools
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


class BundledSynthStripError(RuntimeError):
    """The vendored SynthStrip ran but failed, or an asset was bad."""


class BundledSynthStripUnavailable(BundledSynthStripError):
    """The vendored SynthStrip can't run here — no ``torch``/``surfa`` in the
    runtime, or assets are missing with downloads disabled."""


@dataclass(frozen=True)
class _Asset:
    filename: str
    url: str
    sha256: str
    size: int  # bytes; 0 = not pinned (text script — sha256 is the gate)


# Pinned to a specific FreeSurfer commit (not a moving branch) for
# reproducibility. The standalone script defines ``StripModel`` inline and
# imports only torch / numpy / surfa — no FreeSurfer modules.
_SCRIPT = _Asset(
    filename="mri_synthstrip",
    url=("https://raw.githubusercontent.com/freesurfer/freesurfer/"
         "0ac5dcccb8b6b875312bcd042258d4590ba39814/mri_synthstrip/mri_synthstrip"),
    sha256="bbc2ff8f8779862039401b05d5cd6039fb4f3583e0032a793ac9adb3f4521590",
    size=0,
)

# Model weights — openly licensed (MIT / CC BY-4.0), served from the FreeSurfer
# distribution host. Both variants are the same size; the sha256 distinguishes
# them. ``default`` keeps surrounding CSF; ``nocsf`` excludes it.
_WEIGHTS = {
    "default": _Asset(
        filename="synthstrip.1.pt",
        url=("https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/"
             "synthstrip/models/synthstrip.1.pt"),
        sha256="37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33",
        size=30851709,
    ),
    "nocsf": _Asset(
        filename="synthstrip.nocsf.1.pt",
        url=("https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/"
             "synthstrip/models/synthstrip.nocsf.1.pt"),
        sha256="62bf01137c45b5f0cc04d59dbaed5b9ac138b3f25b766c062a7c1a0d696ecb28",
        size=30851709,
    ),
}

_CACHE_ENV = "ROSA_SYNTHSTRIP_CACHE"
_NO_DOWNLOAD_ENV = "ROSA_SYNTHSTRIP_NO_DOWNLOAD"


def cache_dir() -> Path:
    """Directory holding the cached script + weights.

    Honours ``$ROSA_SYNTHSTRIP_CACHE``, then ``$XDG_CACHE_HOME``, then
    ``~/.cache``; final segment is ``rosa-agent/synthstrip``.
    """
    env = os.environ.get(_CACHE_ENV)
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".cache"
    return root / "rosa-agent" / "synthstrip"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _have(path: Path, sha256: str) -> bool:
    return path.is_file() and _sha256(path) == sha256


def _downloads_allowed() -> bool:
    val = os.environ.get(_NO_DOWNLOAD_ENV, "").strip().lower()
    return val not in ("1", "true", "yes", "on")


def _fetch(asset: _Asset, dest: Path) -> Path:
    """Return a checksum-verified local copy of ``asset`` at ``dest``.

    No-op if already cached and valid. Downloads when missing and permitted;
    re-fetches a corrupt cache entry. Verifies SHA-256 before committing the
    file (atomic rename), so a wrong URL or truncated download fails loudly
    rather than poisoning the cache.
    """
    if _have(dest, asset.sha256):
        return dest
    if dest.exists():
        dest.unlink()  # present but wrong checksum — stale/corrupt
    if not _downloads_allowed():
        raise BundledSynthStripUnavailable(
            f"{asset.filename} not cached at {dest} and downloads are disabled "
            f"({_NO_DOWNLOAD_ENV} set). Place it there manually "
            f"(sha256 {asset.sha256}) or allow downloads."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), prefix=".dl_")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with urllib.request.urlopen(asset.url, timeout=120) as r, open(tmp, "wb") as out:
            shutil.copyfileobj(r, out)
        got = _sha256(tmp)
        if got != asset.sha256:
            raise BundledSynthStripError(
                f"{asset.filename} checksum mismatch from {asset.url}: "
                f"expected {asset.sha256}, got {got}"
            )
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()
    return dest


def ensure_assets(*, no_csf: bool = False) -> Tuple[Path, Path]:
    """Ensure the script + the right weights are cached and verified.

    Returns ``(script_path, weights_path)``. Raises
    :class:`BundledSynthStripUnavailable` when an asset is missing and downloads
    are disabled, or :class:`BundledSynthStripError` on a checksum mismatch.
    """
    cdir = cache_dir()
    script = _fetch(_SCRIPT, cdir / _SCRIPT.filename)
    w = _WEIGHTS["nocsf" if no_csf else "default"]
    weights = _fetch(w, cdir / w.filename)
    return script, weights


@functools.lru_cache(maxsize=8)
def _runtime_ok(python: str) -> bool:
    try:
        r = subprocess.run(
            [python, "-c", "import torch, surfa"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60,
        )
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def runtime_ok(python: str | None = None) -> bool:
    """True when the interpreter that will run the script can import torch+surfa.

    Cached per interpreter path (process lifetime).
    """
    return _runtime_ok(python or sys.executable)


def available(python: str | None = None) -> bool:
    """Best-effort: would the vendored SynthStrip run here?

    True when (a) the runtime has torch+surfa, and (b) the assets are either
    already cached or downloadable. Does **not** download — keeps the ``auto``
    mask path cheap; the actual fetch happens in :func:`run_bundled_synthstrip`.
    """
    if not runtime_ok(python):
        return False
    cdir = cache_dir()
    cached = ((cdir / _SCRIPT.filename).is_file()
              and (cdir / _WEIGHTS["default"].filename).is_file())
    return cached or _downloads_allowed()


def run_bundled_synthstrip(
    input_path: str | Path,
    mask_path: str | Path,
    *,
    stripped_path: str | Path | None = None,
    no_csf: bool = False,
    border: float | None = None,
    threads: int | None = None,
    timeout: float | None = None,
    python: str | None = None,
) -> Path:
    """Skull-strip a NIfTI volume with the vendored standalone SynthStrip.

    No FreeSurfer install, no ``FREESURFER_HOME`` — weights are passed via the
    script's ``--model`` flag, which bypasses FreeSurfer's env lookup entirely.

    Args:
        input_path: input volume (.nii/.nii.gz/.mgz).
        mask_path: where to write the binary brain mask.
        stripped_path: optional skull-stripped volume (script ``-o``).
        no_csf: use the CSF-excluding weights (parenchyma-only mask).
        border: mask border threshold in mm (script ``-b``; default 1).
        threads: PyTorch CPU threads (script ``-t``).
        timeout: seconds before killing the subprocess.
        python: interpreter to run the script with (default ``sys.executable``;
            in the frozen desktop app this is the bundled interpreter).

    Returns:
        Path to the written mask.

    Raises:
        BundledSynthStripUnavailable: runtime lacks torch/surfa, or assets are
            missing with downloads disabled.
        BundledSynthStripError: the strip subprocess failed, or an asset was bad.
    """
    py = python or sys.executable
    if not runtime_ok(py):
        raise BundledSynthStripUnavailable(
            "bundled SynthStrip needs torch + surfa in the runtime — "
            "install with: pip install 'rosa-agent[synthstrip]'"
        )

    # ensure_assets picks the csf/nocsf weights; with --model supplied the
    # script's own --no-csf weight-selection branch is bypassed, so we don't
    # pass --no-csf to it (its only documented effect is choosing the weights).
    script, weights = ensure_assets(no_csf=no_csf)

    input_path = Path(input_path).expanduser().resolve()
    mask_path = Path(mask_path).expanduser().resolve()
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [py, str(script), "-i", str(input_path), "-m", str(mask_path),
           "--model", str(weights)]
    if stripped_path is not None:
        stripped_path = Path(stripped_path).expanduser().resolve()
        stripped_path.parent.mkdir(parents=True, exist_ok=True)
        cmd += ["-o", str(stripped_path)]
    if border is not None:
        cmd += ["-b", str(border)]
    if threads is not None and threads > 0:
        cmd += ["-t", str(int(threads))]

    proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise BundledSynthStripError(
            f"bundled SynthStrip exited {proc.returncode}: "
            f"{proc.stderr.decode('utf-8', errors='replace')[:600]}"
        )

    # Reuse the system-backend's geometry fix-up (q-form/s-form mismatch).
    from .synthstrip import fix_mask_geometry
    fix_mask_geometry(input_path, mask_path)
    return mask_path


__all__ = [
    "BundledSynthStripError",
    "BundledSynthStripUnavailable",
    "cache_dir",
    "ensure_assets",
    "runtime_ok",
    "available",
    "run_bundled_synthstrip",
]
