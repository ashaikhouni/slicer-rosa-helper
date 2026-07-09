"""Brain-mask backend selector — the single source of the intracranial mask.

``backend="auto"`` tries SynthStrip first (best anchoring; what the user added)
and falls through to the LoG-watershed recipe when the SynthStrip binary isn't
installed. ``backend="synthstrip"`` / ``"log-watershed"`` force one; an explicit
``brain_mask`` array/image overrides everything.

Two entry points over the same selector:
  * :func:`select_brain_mask_to_path` — file-based (CLI fit-rosa, cached).
  * :func:`compute_intracranial_mask` — in-memory for a SimpleITK image
    (place_seeg / Slicer); writes a temp CT, runs the selector, reads the mask.

Lifted out of ``cli/rosa_agent/commands/fit_rosa._compute_brain_mask`` 2026-05-30
so the CLI and the Slicer/place_seeg path share ONE mask backend (the gap that
made place_seeg under-anchor vs fit-rosa). The watershed recipe lives in
``rosa_detect.primitives.intracranial``; the SynthStrip + watershed service
wrappers are siblings in this package.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable


def select_brain_mask_to_path(
    ct_path: str | Path,
    mask_path: str | Path,
    *,
    backend: str = "auto",
    synthstrip_path: str | None = None,
    log: Callable[[str], None] = lambda _m: None,
) -> str | None:
    """Write the intracranial mask to ``mask_path``; return the backend name used
    (``"synthstrip"`` / ``"log-watershed"``) or ``None`` if all backends failed.

    For ``backend="auto"``, tries SynthStrip first and falls through to
    LoG-watershed when SynthStrip isn't installed or fails.
    """
    from .synthstrip import (
        SynthStripNotFound, find_synthstrip, run_synthstrip,
    )
    from .log_watershed import (
        LogWatershedFailed, NotACTVolume, run_log_watershed,
    )

    def _try_synthstrip() -> bool:
        # 1. FreeSurfer-installed binary, if present (fastest; no torch needed).
        try:
            run_synthstrip(
                input_path=ct_path, mask_path=mask_path,
                synthstrip_path=synthstrip_path,
            )
            return True
        except SynthStripNotFound:
            pass  # no system binary — fall through to the vendored standalone
        except subprocess.CalledProcessError as exc:
            # Binary present but failed — a real error; don't silently swap
            # backends. Surface and let the caller fall back to watershed.
            log(f"[mask] synthstrip exit {exc.returncode}; "
                f"stderr: {(exc.stderr or b'').decode('utf-8', errors='replace')[:400]}")
            return False
        except Exception as exc:  # noqa: BLE001
            log(f"[mask] synthstrip raised: {exc}")
            return False

        # 2. Vendored standalone SynthStrip (torch + surfa; no FreeSurfer).
        #    Only reachable when no system binary exists.
        try:
            from .synthstrip_bundled import (
                run_bundled_synthstrip, BundledSynthStripUnavailable,
            )
            run_bundled_synthstrip(input_path=ct_path, mask_path=mask_path)
            return True
        except BundledSynthStripUnavailable as exc:
            log(f"[mask] bundled synthstrip unavailable: {exc}")
            return False
        except Exception as exc:  # noqa: BLE001
            log(f"[mask] bundled synthstrip failed: {exc}")
            return False

    def _try_log_watershed() -> bool:
        try:
            run_log_watershed(input_path=ct_path, mask_path=mask_path)
            return True
        except (LogWatershedFailed, NotACTVolume) as exc:
            log(f"[mask] log-watershed failed: {exc}")
            return False

    def _try_deepbet() -> bool:
        # Fast MIT T1 strip (subprocess; torch isolated from SITK). T1-ONLY, so
        # it's an explicit backend on an MRI — never part of 'auto' (which runs
        # on the CT). Used by the MRI-derived mask route.
        try:
            from .deepbet_strip import run_deepbet, DeepbetNotFound
            run_deepbet(input_path=ct_path, mask_path=mask_path, log=log)
            return True
        except DeepbetNotFound as exc:
            log(f"[mask] deepbet unavailable: {exc}")
            return False
        except subprocess.CalledProcessError as exc:
            log(f"[mask] deepbet exit {exc.returncode}; "
                f"stderr: {(exc.stderr or b'').decode('utf-8', errors='replace')[:400]}")
            return False
        except Exception as exc:  # noqa: BLE001
            log(f"[mask] deepbet raised: {exc}")
            return False

    def _try_hull() -> bool:
        # Poor-man's intracranial mask: cheap build_masks (head-distance ≥ 10 mm),
        # no external binary, never artifacts on metal. Coarse boundary.
        try:
            import SimpleITK as sitk
            from ..primitives.preprocessing import build_masks
            img = sitk.ReadImage(str(ct_path))
            _hull, intra, _dist = build_masks(img)
            mask_img = sitk.GetImageFromArray(intra.astype("uint8"))
            mask_img.CopyInformation(img)
            sitk.WriteImage(mask_img, str(mask_path))
            return True
        except Exception as exc:  # noqa: BLE001
            log(f"[mask] hull backend failed: {exc}")
            return False

    if backend == "deepbet":
        return "deepbet" if _try_deepbet() else None
    if backend == "hull":
        return "hull" if _try_hull() else None
    if backend == "synthstrip":
        return "synthstrip" if _try_synthstrip() else None
    if backend == "log-watershed":
        return "log-watershed" if _try_log_watershed() else None
    # auto: SynthStrip *binary* if installed, else log-watershed — unchanged.
    # The vendored standalone (torch) is deliberately NOT auto-activated here:
    # it would mean a 30 MB download + slow NN inference the moment torch is
    # importable, surprising callers and destabilising the default mask. The
    # vendored path is reachable only via an explicit ``backend="synthstrip"``
    # (which routes through ``_try_synthstrip``'s system→vendored fallback).
    if find_synthstrip(synthstrip_path) is not None:
        if _try_synthstrip():
            return "synthstrip"
        log("[mask] synthstrip available but failed; falling back to log-watershed")
    else:
        log("[mask] synthstrip binary not found; using log-watershed "
            "(force the vendored standalone with backend='synthstrip')")
    return "log-watershed" if _try_log_watershed() else None


def compute_intracranial_mask(
    img: Any,
    *,
    backend: str = "auto",
    synthstrip_path: str | None = None,
    brain_mask: Any = None,
    log: Callable[[str], None] | None = None,
) -> tuple[Any, str]:
    """In-memory intracranial mask for a SimpleITK image.

    Returns ``(mask_kji_bool, backend_used)`` where ``mask_kji_bool`` is a numpy
    bool array in SimpleITK [k, j, i] order, aligned to ``img``'s grid.

    ``brain_mask`` (numpy array or SimpleITK image) short-circuits the backends —
    the caller-provided override. ``backend="hull"`` returns the cheap
    ``build_masks`` intracranial (head-distance ≥ 10 mm) — a fast, dependency-free
    approximation, no temp file or external binary. Otherwise runs
    :func:`select_brain_mask_to_path` over a temp CT (so SynthStrip's binary,
    which is file-based, works) and reads the mask back; resamples to ``img`` if
    the backend changed the grid.

    Raises ``RuntimeError`` when every backend fails (degenerate FOV / non-CT).
    """
    import numpy as np
    import SimpleITK as sitk
    _log = log or (lambda _m: None)

    if brain_mask is not None:
        if isinstance(brain_mask, sitk.Image):
            if brain_mask.GetSize() != img.GetSize():
                brain_mask = sitk.Resample(
                    brain_mask, img, sitk.Transform(),
                    sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
                )
            return sitk.GetArrayFromImage(brain_mask).astype(bool), "provided"
        return np.asarray(brain_mask).astype(bool), "provided"

    if backend == "hull":
        # Poor-man's intracranial: the same array detection uses by default.
        # Cheap (threshold + close + fillhole + distance), never artifacts on
        # metal, no FreeSurfer dependency — at the cost of a coarse boundary
        # (head silhouette − 10 mm includes skull/dura). Already KJI-order.
        from ..primitives.preprocessing import build_masks
        _hull, intra, _dist = build_masks(img)
        return intra.astype(bool), "hull"

    with tempfile.TemporaryDirectory(prefix="rosa_mask_") as td:
        ct_p = Path(td) / "ct.nii.gz"
        m_p = Path(td) / "mask.nii.gz"
        sitk.WriteImage(img, str(ct_p))
        used = select_brain_mask_to_path(
            ct_p, m_p, backend=backend,
            synthstrip_path=synthstrip_path, log=_log,
        )
        if used is None:
            # Every backend failed. On the lenient default (``backend="auto"``)
            # this is treated as a recoverable degenerate/non-CT input (e.g. an
            # all-zero synthetic phantom the watershed rejects, with no
            # SynthStrip installed): fall back to a FULL-VOLUME permissive mask
            # so downstream candidate generation simply finds nothing and the
            # caller still produces its normal (empty) outputs — instead of a
            # hard crash. An EXPLICITLY requested backend that fails still raises
            # (a real subject should surface a mis-configured mask loudly).
            if backend == "auto":
                _log("[mask] all backends failed on backend='auto'; using a "
                     "full-volume permissive mask (degenerate / non-CT input)")
                shape = sitk.GetArrayFromImage(img).shape
                return np.ones(shape, dtype=bool), "full-volume"
            raise RuntimeError(
                "intracranial mask: all backends failed "
                f"(backend={backend!r}); FOV may be degenerate or non-CT."
            )
        mask_img = sitk.ReadImage(str(m_p))
        if mask_img.GetSize() != img.GetSize():
            mask_img = sitk.Resample(
                mask_img, img, sitk.Transform(),
                sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
            )
        return sitk.GetArrayFromImage(mask_img).astype(bool), used


__all__ = ["select_brain_mask_to_path", "compute_intracranial_mask"]
