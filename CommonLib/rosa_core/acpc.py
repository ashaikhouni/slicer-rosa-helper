"""Rigid AC-PC reorientation for slice QC.

The head is rarely axis-aligned in the scanner bore, so native-frame slices
show a tilted brain — hard to read as standard neuroanatomical planes. Rigidly
register the patient's MRI to the AC-PC-aligned MNI152 template and resample
BOTH the CT and the MRI onto that grid: the anatomy comes upright in axial /
coronal / sagittal planes.

Rigid (6-DOF) on purpose: it reorients without deforming, so electrode shafts
stay straight and inter-contact distances are preserved. (The atlas warp is
affine/nonlinear — fine for label placement, but it shears or bends geometry, so
it is the wrong tool for a view you might measure an electrode in.)

The MRI drives the fit (same modality as the template → robust); the CT rides
along on the already-cached rigid ``t1_to_ct`` transform. So no CT↔template
cross-modality registration is needed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from .bundled_atlases import resolve as _resolve_atlas
from .registration import load_transform, register_rigid_mi, resample_volume


def default_template_path() -> Path:
    """The bundled MNI152 T1w template (AC-PC aligned) used as the reorient grid."""
    return _resolve_atlas(None).template_path


def _isotropic_reference(template, mm: float):
    """A resampling grid: the template's origin/orientation at ``mm`` spacing.

    Keeps the QC output at a fixed resolution regardless of the default atlas's
    template spacing (the bundled 2009c template is already 1 mm; a coarser
    template would otherwise give soft QC planes). Same AC-PC orientation + FOV;
    only the output grid changes — the rigid transform is unaffected.
    """
    import SimpleITK as sitk

    sp = template.GetSpacing()
    sz = template.GetSize()
    new_size = [max(1, int(round(sz[i] * sp[i] / mm))) for i in range(3)]
    ref = sitk.Image(new_size, template.GetPixelID())
    ref.SetOrigin(template.GetOrigin())
    ref.SetDirection(template.GetDirection())
    ref.SetSpacing((mm, mm, mm))
    return ref


def reorient_to_acpc(
    ct_path: str | Path,
    mri_path: str | Path,
    t1_to_ct_path: str | Path,
    out_ct: str | Path,
    out_mri: str | Path,
    *,
    template_path: str | Path | None = None,
    brain_mask_path: str | Path | None = None,
    out_mm: float = 1.0,
    logger: Optional[Callable[[str], None]] = None,
) -> dict:
    """Resample CT + MRI onto the AC-PC-aligned MNI grid (rigid), write both.

    Args:
        ct_path: postop CT (native frame).
        mri_path: preop MRI (T1, native frame) — drives the fit.
        t1_to_ct_path: cached rigid transform from case creation. Computed with
            ``fixed=CT, moving=T1`` (see brain-extract ``--save-transform``), so
            it maps CT→T1; we invert it for T1→CT.
        out_ct / out_mri: where to write ``ct_in_acpc`` / ``mri_in_acpc``.
        template_path: MNI template to reorient to (default: the bundled one).
        brain_mask_path: optional native-T1 brain mask — masking the skull off
            the MRI keeps the rigid fit from being pulled by the (template-
            absent) cranium. Ignored if missing.
        logger: optional ``logger(str)`` progress callback.

    Returns:
        ``{"metric": float, "n_iterations": int, "out_ct": str, "out_mri": str}``.
    """
    import SimpleITK as sitk

    tmpl_path = Path(template_path) if template_path is not None else default_template_path()
    template = sitk.ReadImage(str(tmpl_path), sitk.sitkFloat32)
    t1 = sitk.ReadImage(str(mri_path), sitk.sitkFloat32)
    ct = sitk.ReadImage(str(ct_path), sitk.sitkFloat32)

    # The template is skull-stripped; mask the MRI to brain so the cranium (which
    # has no counterpart in the template) can't steer the mutual-information fit.
    moving = t1
    if brain_mask_path and Path(brain_mask_path).is_file():
        mask = sitk.ReadImage(str(brain_mask_path))
        mask = sitk.Resample(mask, t1, sitk.Transform(), sitk.sitkNearestNeighbor,
                             0, mask.GetPixelID())
        moving = sitk.Mask(t1, sitk.Cast(mask != 0, sitk.sitkUInt8))
        if logger is not None:
            logger("[acpc] masked MRI to brain for the rigid fit")

    # fixed=template, moving=MRI → transform maps template→MRI (a resampling
    # transform: for each template-grid point, where to sample the MRI). moments
    # init aligns brain centres-of-mass — robust for brain↔brain.
    if logger is not None:
        logger("[acpc] rigid MRI → MNI template (AC-PC) …")
    res = register_rigid_mi(template, moving, init_mode="moments", logger=logger)
    tpl_to_t1 = res.transform

    # Slice onto a finer isotropic grid than the 2 mm template so the QC is crisp.
    grid = _isotropic_reference(template, float(out_mm))

    # MRI onto the AC-PC grid.
    mri_in_acpc = resample_volume(t1, tpl_to_t1, reference=grid, interp="linear")
    sitk.WriteImage(mri_in_acpc, str(out_mri))

    # CT onto the AC-PC grid: template → T1 (rigid fit) → CT (inverse of the
    # cached CT→T1). SITK applies the LAST-added transform FIRST, so add T1→CT
    # first (applied second) and template→T1 last (applied first).
    t1_to_ct = load_transform(str(t1_to_ct_path))   # maps CT→T1
    tpl_to_ct = sitk.CompositeTransform(3)
    tpl_to_ct.AddTransform(t1_to_ct.GetInverse())   # T1→CT  (applied second)
    tpl_to_ct.AddTransform(tpl_to_t1)               # tpl→T1 (applied first)
    ct_in_acpc = resample_volume(ct, tpl_to_ct, reference=grid, interp="linear")
    sitk.WriteImage(ct_in_acpc, str(out_ct))

    if logger is not None:
        logger(f"[acpc] wrote AC-PC CT/MRI (metric={res.final_metric:+.4f}, "
               f"{res.n_iterations} iters)")
    return {"metric": float(res.final_metric), "n_iterations": int(res.n_iterations),
            "out_ct": str(out_ct), "out_mri": str(out_mri)}


__all__ = ["reorient_to_acpc", "default_template_path"]
