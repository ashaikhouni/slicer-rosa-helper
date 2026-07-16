"""AC-PC reorientation for a standardized clinician slice view.

The head is rarely axis-aligned in the scanner bore, so native slices show a
tilted brain — hard to read as standard neuroanatomical planes. This estimates
the AC-PC orientation and reslices the images along it, so axial / coronal /
sagittal read the way a clinician expects.

Two deliberate choices, per the design discussion:

* **The registration is only a *ruler for the tilt*, not a resampling target.**
  A quick, coarse rigid fit of the MRI to the AC-PC-aligned MNI template gives
  the rotation; we then reslice the *native* pixels along it. Nothing is ever
  shown in MNI space — that (nonlinear) warp is a labeling concern.

* **Native intensities, upright, head-centred.** The patient's own CT + MRI are
  resampled (rigid = no deformation) onto the AC-PC-aligned template grid at
  1 mm, padded symmetrically to include the skull. The head stays centred and
  reads as standard planes; nothing is warped to MNI.

The MRI drives the fit (same modality as the template → robust); the CT rides
along on the already-cached rigid ``t1_to_ct``, so no CT↔template cross-modality
registration is needed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from .bundled_atlases import resolve as _resolve_atlas
from .registration import load_transform, register_rigid_mi, resample_volume

def default_template_path() -> Path:
    """The bundled MNI152 T1w template (AC-PC aligned) used as the orientation ruler."""
    return _resolve_atlas(None).template_path


def _acpc_grid(template, *, pad_mm: float = 25.0):
    """The template's AC-PC grid, padded SYMMETRICALLY to include the skull.

    The template is skull-stripped (brain-tight), so resampling onto it alone
    clips the cranium. Pad it a fixed amount on every side: symmetric padding
    keeps the head centred (so it reads as upright, cleanly framed — an
    asymmetric, CT-bbox-fitted pad instead shoves the head off-centre and the
    rotated acquisition-box edge then *looks* like a tilt). ``sitk.ConstantPad``
    preserves the template's exact geometry + voxel-axis storage, so the result
    renders identically to the (known-good) template grid, just larger.
    """
    import SimpleITK as sitk

    tsp = template.GetSpacing()
    pad = [int(round(pad_mm / tsp[k])) for k in range(3)]
    return sitk.ConstantPad(template, pad, pad, 0.0)


def reorient_to_acpc(
    ct_path: str | Path,
    mri_path: str | Path,
    t1_to_ct_path: str | Path,
    out_ct: str | Path,
    out_mri: str | Path,
    *,
    template_path: str | Path | None = None,
    brain_mask_path: str | Path | None = None,
    logger: Optional[Callable[[str], None]] = None,
) -> dict:
    """Reslice CT + MRI along AC-PC (1 mm, head-centred, skull included), write both.

    Args:
        ct_path: postop CT (native frame).
        mri_path: preop MRI (T1, native frame) — drives the rotation fit.
        t1_to_ct_path: cached rigid transform from case creation, computed with
            ``fixed=CT, moving=T1`` (maps CT→T1); inverted here for T1→CT.
        out_ct / out_mri: where to write ``ct_in_acpc`` / ``mri_in_acpc``.
        template_path: AC-PC-aligned MNI template used as the tilt ruler.
        brain_mask_path: optional native-T1 brain mask — masking the skull off
            the MRI keeps the rigid fit from being pulled by the (template-
            absent) cranium.
        logger: optional ``logger(str)`` progress callback.

    Returns:
        ``{"metric", "n_iterations", "out_ct", "out_mri"}``.
    """
    import SimpleITK as sitk

    tmpl_path = Path(template_path) if template_path is not None else default_template_path()
    template = sitk.ReadImage(str(tmpl_path), sitk.sitkFloat32)
    t1 = sitk.ReadImage(str(mri_path), sitk.sitkFloat32)
    ct = sitk.ReadImage(str(ct_path), sitk.sitkFloat32)

    # The template is skull-stripped; mask the MRI to brain so the cranium (with
    # no template counterpart) can't steer the mutual-information fit.
    moving = t1
    if brain_mask_path and Path(brain_mask_path).is_file():
        mask = sitk.ReadImage(str(brain_mask_path))
        mask = sitk.Resample(mask, t1, sitk.Transform(), sitk.sitkNearestNeighbor,
                             0, mask.GetPixelID())
        moving = sitk.Mask(t1, sitk.Cast(mask != 0, sitk.sitkUInt8))
        if logger is not None:
            logger("[acpc] masked MRI to brain for the tilt fit")

    # Rigid fit — the ORIENTATION only. fixed=template, moving=MRI → transform
    # maps template→MRI (resampling convention). moments init aligns brain
    # centres-of-mass, robust for brain↔brain. Uses the full multi-res pyramid: a
    # coarse-only "quick" fit left a visible residual tilt, and a standardized
    # view has to actually be straight (this is still a ~5s one-time, cached fit).
    if logger is not None:
        logger("[acpc] rigid MRI → MNI template (AC-PC axis) …")
    res = register_rigid_mi(template, moving, init_mode="moments", logger=logger)
    tpl_to_t1 = res.transform

    # CT reslicing transform: template → T1 (the fit) → CT (inverse of the cached
    # CT→T1). SITK applies the LAST-added transform FIRST, so add T1→CT first
    # (applied second) and template→T1 last (applied first).
    t1_to_ct = load_transform(str(t1_to_ct_path))   # maps CT→T1
    tpl_to_ct = sitk.CompositeTransform(3)
    tpl_to_ct.AddTransform(t1_to_ct.GetInverse())   # T1→CT  (applied second)
    tpl_to_ct.AddTransform(tpl_to_t1)               # tpl→T1 (applied first)

    # AC-PC-oriented output grid: the brain-centred template, padded to include
    # the skull (head-centred, upright, clean framing).
    grid = _acpc_grid(template)

    # Store int16 — CT (HU) and MRI intensities are integer-valued, so this is
    # lossless and roughly halves the cache vs float32.
    mri_in_acpc = resample_volume(t1, tpl_to_t1, reference=grid, interp="linear")
    sitk.WriteImage(sitk.Cast(sitk.Round(mri_in_acpc), sitk.sitkInt16), str(out_mri))
    ct_in_acpc = resample_volume(ct, tpl_to_ct, reference=grid, interp="linear")
    sitk.WriteImage(sitk.Cast(sitk.Round(ct_in_acpc), sitk.sitkInt16), str(out_ct))

    if logger is not None:
        logger(f"[acpc] resliced CT/MRI into AC-PC (metric={res.final_metric:+.4f}, "
               f"{res.n_iterations} iters, grid {grid.GetSize()})")
    return {"metric": float(res.final_metric), "n_iterations": int(res.n_iterations),
            "out_ct": str(out_ct), "out_mri": str(out_mri)}


__all__ = ["reorient_to_acpc", "default_template_path"]
