"""SynthStrip wrapper — locates and invokes FreeSurfer's `mri_synthstrip`.

SynthStrip is FreeSurfer's deep-learning brain extraction tool. It's
modality-agnostic (works on CT and MRI), ships with FreeSurfer 7.3+, and
runs as a binary so we don't need to import PyTorch.

This module:

* **Locates** the `mri_synthstrip` binary across the usual install paths
  without making FreeSurfer a hard dependency.
* **Runs** it on a NIfTI input and returns the path to the output mask.
* **Fails clearly** when the binary isn't found — callers can either ask
  the user for a path or silently skip the step.

Detection priority (highest first):

1. Explicit ``synthstrip_path`` argument.
2. ``$ROSA_SYNTHSTRIP`` environment variable.
3. ``which mri_synthstrip`` (anything on PATH).
4. ``$FREESURFER_HOME/bin/mri_synthstrip``.
5. Glob: ``/Applications/freesurfer/*/bin/mri_synthstrip``,
   ``/usr/local/freesurfer*/bin/mri_synthstrip``,
   ``/opt/freesurfer*/bin/mri_synthstrip``.

A caller that wants to fail quietly:

    >>> from rosa_detect.services.synthstrip import find_synthstrip
    >>> binary = find_synthstrip()
    >>> if binary is None:
    ...     # SynthStrip not installed; skip
    ...     pass

A caller that wants to error out:

    >>> from rosa_detect.services.synthstrip import run_synthstrip, SynthStripNotFound
    >>> try:
    ...     mask_path = run_synthstrip("ct.nii.gz", "out_mask.nii.gz")
    ... except SynthStripNotFound:
    ...     # tell the user to set ROSA_SYNTHSTRIP or pass --synthstrip
    ...     pass
"""
from __future__ import annotations

import os
import shutil
import subprocess
from glob import glob
from pathlib import Path
from typing import Optional


class SynthStripNotFound(FileNotFoundError):
    """Raised when SynthStrip can't be located."""


_SEARCH_GLOBS = (
    "/Applications/freesurfer/*/bin/mri_synthstrip",
    "/usr/local/freesurfer/*/bin/mri_synthstrip",
    "/usr/local/freesurfer*/bin/mri_synthstrip",
    "/opt/freesurfer/*/bin/mri_synthstrip",
    "/opt/freesurfer*/bin/mri_synthstrip",
)


def find_synthstrip(synthstrip_path: str | Path | None = None) -> Optional[Path]:
    """Locate the ``mri_synthstrip`` binary.

    Args:
        synthstrip_path: explicit binary path or directory containing
            it. Takes precedence over all other detection paths.

    Returns:
        Path to a working binary, or ``None`` if nothing was found.
    """
    # 1. Explicit argument
    if synthstrip_path:
        p = Path(synthstrip_path).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return p
        # Allow passing a directory
        candidate = p / "mri_synthstrip"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate

    # 2. Env var
    env_path = os.environ.get("ROSA_SYNTHSTRIP")
    if env_path:
        p = Path(env_path).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return p

    # 3. PATH
    on_path = shutil.which("mri_synthstrip")
    if on_path:
        return Path(on_path)

    # 4. FREESURFER_HOME
    fs_home = os.environ.get("FREESURFER_HOME")
    if fs_home:
        p = Path(fs_home) / "bin" / "mri_synthstrip"
        if p.is_file() and os.access(p, os.X_OK):
            return p

    # 5. Globs
    for pattern in _SEARCH_GLOBS:
        for hit in sorted(glob(pattern)):
            p = Path(hit)
            if p.is_file() and os.access(p, os.X_OK):
                return p

    return None


def run_synthstrip(
    input_path: str | Path,
    mask_path: str | Path,
    *,
    synthstrip_path: str | Path | None = None,
    stripped_path: str | Path | None = None,
    no_csf: bool = False,
    threads: int | None = None,
    timeout: float | None = None,
) -> Path:
    """Run SynthStrip on a NIfTI volume.

    Args:
        input_path: path to the input volume (.nii/.nii.gz/.mgz — anything FS reads).
        mask_path: path where SynthStrip should write the binary brain mask.
        synthstrip_path: explicit path/dir for the binary; falls through
            to ``find_synthstrip``'s detection chain.
        stripped_path: optional path for the skull-stripped volume.
            (SynthStrip's ``-o``; mask is the binary, stripped is the
            volume × mask.) Omit if you only need the mask.
        no_csf: pass ``--no-csf`` to SynthStrip — tighter mask that
            excludes CSF voxels at the surface. Useful when you want
            parenchyma-only.
        threads: optional thread count (``-t``).
        timeout: seconds before killing the subprocess. ``None`` = wait
            indefinitely.

    Returns:
        Path to the written mask file.

    Raises:
        SynthStripNotFound: when the binary can't be located.
        subprocess.CalledProcessError: when the subprocess exits non-zero.
        subprocess.TimeoutExpired: when ``timeout`` is exceeded.
    """
    binary = find_synthstrip(synthstrip_path)
    if binary is None:
        raise SynthStripNotFound(
            "mri_synthstrip not found. Install FreeSurfer 7.3+ or "
            "pass --synthstrip <path>, set ROSA_SYNTHSTRIP, or set "
            "FREESURFER_HOME."
        )

    input_path = Path(input_path).expanduser().resolve()
    mask_path  = Path(mask_path).expanduser().resolve()
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [str(binary), "-i", str(input_path), "-m", str(mask_path)]
    if stripped_path is not None:
        stripped_path = Path(stripped_path).expanduser().resolve()
        stripped_path.parent.mkdir(parents=True, exist_ok=True)
        cmd += ["-o", str(stripped_path)]
    if no_csf:
        cmd += ["--no-csf"]
    if threads is not None and threads > 0:
        cmd += ["-t", str(int(threads))]

    # SynthStrip refuses to run unless FREESURFER_HOME is set ("freesurfer
    # has not been properly sourced"). Derive it from the binary path —
    # the binary lives at $FREESURFER_HOME/bin/mri_synthstrip — and inject
    # into the subprocess env unless the caller already has it set.
    env = dict(os.environ)
    if "FREESURFER_HOME" not in env:
        fs_home = binary.resolve().parent.parent
        if (fs_home / "bin" / "mri_synthstrip").exists():
            env["FREESURFER_HOME"] = str(fs_home)

    subprocess.run(cmd, check=True, timeout=timeout, env=env,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # SynthStrip occasionally writes the output mask with the volume's
    # s-form affine while the input CT it read used q-form (e.g., AMC136
    # in our cohort has qform_code=1, sform_code=2, q-vs-s differ by 50mm).
    # Force the mask to inherit the input image's full geometry so that
    # downstream tools that read the input's q-form get an aligned mask.
    try:
        import SimpleITK as sitk
        ref_img = sitk.ReadImage(str(input_path))
        mask_img = sitk.ReadImage(str(mask_path))
        if mask_img.GetSize() == ref_img.GetSize():
            mask_img.CopyInformation(ref_img)
            sitk.WriteImage(mask_img, str(mask_path))
    except Exception:
        # Best-effort; if SITK isn't available or the mask shape doesn't
        # match, leave the file as SynthStrip wrote it.
        pass

    return mask_path


def synthstrip_available(synthstrip_path: str | Path | None = None) -> bool:
    """Convenience: True when SynthStrip can be invoked."""
    return find_synthstrip(synthstrip_path) is not None
