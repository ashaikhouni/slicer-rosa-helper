"""Reformat a volume along anatomical (AC-PC / MNI) axes *in place*.

The head is rarely axis-aligned in the scanner, so native slices are oblique.
:mod:`rosa_core.acpc` reslices onto the MNI *template grid* (a new world frame),
which is right for a standalone AC-PC view but would move every overlay. Here we
instead keep the **original world** and only rotate the sampling grid: the output
volume occupies the same physical space as the input, but its voxel i/j/k axes are
the anatomical R/A/S axes. Grid-stepping it therefore yields true axial / coronal
/ sagittal cuts, while contacts, atlas overlays and cut planes — all addressed in
world RAS — stay consistent with **no** per-overlay transform.

The anatomical orientation is taken (rigidly) from the CT→MNI matrix the app
already caches for atlas labeling: the nearest rotation to its linear part
(scale/shear discarded, so the patient's true scale is preserved).
"""
from __future__ import annotations

import numpy as np

__all__ = ["rigid_rotation_from_affine", "reslice_along_anatomical_axes"]

# RAS <-> LPS axis flip (SITK stores geometry in LPS; our matrices are RAS).
_F = np.diag([-1.0, -1.0, 1.0])


def rigid_rotation_from_affine(matrix4x4) -> np.ndarray:
    """The proper rotation (3×3, det = +1) nearest to a 4×4 affine's linear part.

    Polar decomposition via SVD: ``L = U Σ Vᵀ`` → ``R = U Vᵀ`` (the closest
    orthonormal matrix), with a sign fix so ``det R = +1``. Strips the affine's
    anisotropic scale/shear, keeping only orientation.
    """
    L = np.asarray(matrix4x4, dtype=float)[:3, :3]
    U, _, Vt = np.linalg.svd(L)
    R = U @ Vt
    if np.linalg.det(R) < 0:                      # reflection → flip the weakest axis
        U = U.copy()
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def reslice_along_anatomical_axes(sitk_img, ct_to_mni_ras_4x4, *, spacing_mm: float = 1.0):
    """Reformat ``sitk_img`` so its voxel axes are the anatomical R/A/S axes,
    **keeping the same physical world**.

    ``ct_to_mni_ras_4x4`` maps CT-frame RAS → MNI RAS (e.g. ``cohort.ct_to_mni_matrix``).
    We take its rigid rotation ``R`` (CT→MNI); the MNI/anatomical axes expressed in
    CT-world RAS are then the columns of ``Rᵀ``. We build an output grid in the
    input's own world with those axis directions, sized to cover the input, and
    resample (identity transform — same world). ``_slice_volume_meta`` will read the
    correct ``vox_to_ras`` off the result, so overlays line up automatically.

    Returns a resampled ``SimpleITK.Image`` (same pixel type as the input).
    """
    import SimpleITK as sitk

    R = rigid_rotation_from_affine(ct_to_mni_ras_4x4)     # CT→MNI rotation (RAS)
    axes_ras = R.T                                        # columns = anatomical axes in CT-world RAS
    direction_lps = _F @ axes_ras                         # SITK direction cosines (LPS), det = +1

    # Physical corners of the input (LPS world) projected onto the anatomical axes,
    # to size/place a grid that just covers the head.
    sz = sitk_img.GetSize()
    corners_idx = [(x, y, z) for x in (0, sz[0]) for y in (0, sz[1]) for z in (0, sz[2])]
    corners = np.array([sitk_img.TransformContinuousIndexToPhysicalPoint([float(c) for c in ci])
                        for ci in corners_idx])           # (8,3) LPS
    proj = corners @ direction_lps                        # coord of each corner along each anat axis
    lo, hi = proj.min(axis=0), proj.max(axis=0)

    sp = float(spacing_mm)
    out_size = [int(np.ceil((hi[k] - lo[k]) / sp)) + 1 for k in range(3)]
    origin_lps = direction_lps @ lo                       # world point of output voxel (0,0,0)

    ref = sitk.Image(int(out_size[0]), int(out_size[1]), int(out_size[2]), sitk_img.GetPixelID())
    ref.SetSpacing((sp, sp, sp))
    ref.SetDirection([float(v) for v in direction_lps.flatten()])
    ref.SetOrigin([float(v) for v in origin_lps])
    return sitk.Resample(sitk_img, ref, sitk.Transform(), sitk.sitkLinear, 0.0, sitk_img.GetPixelID())
