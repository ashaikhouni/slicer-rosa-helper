"""ROSA-case import helpers: middle-slice thumbnails + CT/MRI auto-guess.

A ROSA export folder holds several *display* volumes (post-op CT, pre-op MRI, …).
After ``rosa-to-nifti`` bakes them to NIfTI, the import UI shows each with a
middle-slice thumbnail and an auto-guessed role (CT vs MRI) for the clinician to
confirm — they must pick a post-op CT and a non-contrast T1.
"""
from __future__ import annotations

from pathlib import Path


def guess_modality(nii_path) -> str:
    """Heuristic role from intensities: CT is Hounsfield (air ≈ −1000, bone/metal
    ≫ 0, strong negatives); MRI has no strong negatives. Returns ct / mri / other."""
    import numpy as np
    import nibabel as nib
    try:
        data = np.asanyarray(nib.load(str(nii_path)).dataobj)
    except Exception:  # noqa: BLE001
        return "other"
    if data.size == 0:
        return "other"
    lo = float(np.percentile(data, 1))
    # Hounsfield air (≈ −1000) is the reliable tell: a CT has a large air
    # background; a magnitude MRI (T1) has no strongly-negative intensities.
    if lo < -300:
        return "ct"
    if lo > -100:
        return "mri"
    return "other"


def slice_png(nii_path, size: int = 150) -> str:
    """The volume's middle axial slice as a base64 PNG data URI (a thumbnail so
    the clinician can tell CT from MRI at a glance)."""
    import base64
    import io
    import numpy as np
    import nibabel as nib
    from PIL import Image
    data = np.asanyarray(nib.load(str(nii_path)).dataobj)
    while data.ndim > 3:
        data = data[..., 0]
    if data.ndim < 3 or min(data.shape) == 0:
        return ""
    sl = np.asarray(data[:, :, data.shape[2] // 2], dtype=float)
    sl = np.rot90(sl)                                     # roughly upright
    lo, hi = np.percentile(sl, 2), np.percentile(sl, 98)
    if hi <= lo:
        hi = lo + 1.0
    g = np.clip((sl - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    im = Image.fromarray(g, "L")
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def enumerate_displays(staging_dir) -> list:
    """Every baked display volume in ``staging_dir`` with a role guess + thumbnail."""
    out = []
    for nii in sorted(Path(staging_dir).glob("*.nii.gz")):
        out.append({
            "name": nii.name[:-len(".nii.gz")],
            "path": str(nii),
            "guess": guess_modality(nii),
            "png": slice_png(nii),
        })
    return out


__all__ = ["guess_modality", "slice_png", "enumerate_displays"]
