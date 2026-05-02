"""CT volume preprocessing for the contact_pitch_v1 detection pipeline.

Both Auto Fit's ``run_two_stage_detection`` and Guided Fit's
``compute_features`` (in ``guided_fit_engine``) call ``prepare_volume``
first so they see the same canonical volume. Any drift between the two
paths is a P0 parity bug per ``feedback_cli_slicer_parity.md``.

The four entry points exported here are also imported by external
research probes via the module-level alias
``rosa_detect.contact_pitch_v1_fit`` — that file keeps re-exporting
the symbols for backwards compatibility. Once the rest of the
contact_pitch_v1_fit monolith is split this preprocessing module
becomes the canonical home; re-exports can then be removed.
"""

from __future__ import annotations

import numpy as np


# ---- Pipeline-wide tuning constants ---------------------------------

INTRACRANIAL_MIN_DISTANCE_MM = 10.0
FRANGI_STAGE1_SIGMA = 1.0
LOG_SIGMA_MM = 1.0
CANONICAL_SPACING_MM = 1.0
HU_CLIP_MAX = 3000.0


# ---- Public preprocessing API ---------------------------------------


def prepare_volume(img, ijk_to_ras_mat, ras_to_ijk_mat=None):
    """Canonicalize a CT volume so every contact_pitch_v1 consumer sees
    the same input regardless of scanner spacing or HU range.

    Three transformations applied IN ORDER:

      1. **Resample** to ``CANONICAL_SPACING_MM`` if any input axis is
         finer than ``0.95 × canonical`` (typical raw post-op CT at
         0.4-0.7 mm). Native 1 mm CTs skip this. Bilinear interp,
         default pixel value -1024.
      2. **Anisotropic Gaussian anti-alias** per axis:
         ``σ[i] = max(0, canonical - s_in[i])``. Axes that aren't
         downsampled get σ=0 — no smoothing where there's no aliasing
         risk. Without this the raw input's sub-mm-grid aliasing
         survives into the LoG and the blob extractor reports ghost
         contacts (21 vs 14 stage-1 emissions on raw T7 vs registered
         T7); isotropic σ=0.7 over-smoothed near-canonical axes and
         lost S56's two horizontal SEEG shanks.
      3. **HU clamp** to ``[-1024, HU_CLIP_MAX]`` so the LoG / Frangi
         response is consistent across scans regardless of how the CT
         reconstruction encodes metal saturation.

    Returns ``(img_canon, ijk_to_ras_canon, ras_to_ijk_canon)``. When
    no resampling is needed the matrices pass through unchanged
    (apart from being cast to float64 ndarrays); when resampling
    happens the matrices are recomputed from the new grid via
    ``shank_core.io.image_ijk_ras_matrices``. ``ras_to_ijk_mat`` may
    be omitted in the latter case — it'll be derived from the new
    image; in the no-resample case the caller's matrix is preserved
    as-is for numerical-parity reasons (otherwise we'd take an
    inverse here that diverges from the Slicer-supplied inverse by
    floating-point noise).
    """
    import SimpleITK as sitk
    from shank_core.io import image_ijk_ras_matrices

    ijk_to_ras_mat = np.asarray(ijk_to_ras_mat, dtype=float)

    spacing = img.GetSpacing()
    needs_resample = min(float(s) for s in spacing) < CANONICAL_SPACING_MM * 0.95
    if needs_resample:
        size_in = img.GetSize()
        target_spacing = (CANONICAL_SPACING_MM, CANONICAL_SPACING_MM, CANONICAL_SPACING_MM)
        target_size = [
            max(1, int(round(size_in[i] * float(spacing[i]) / CANONICAL_SPACING_MM)))
            for i in range(3)
        ]
        rs = sitk.ResampleImageFilter()
        rs.SetOutputSpacing(target_spacing)
        rs.SetSize(target_size)
        rs.SetOutputOrigin(img.GetOrigin())
        rs.SetOutputDirection(img.GetDirection())
        rs.SetInterpolator(sitk.sitkLinear)
        rs.SetDefaultPixelValue(-1024)
        img = rs.Execute(img)
        sigmas_per_axis = tuple(
            float(max(0.0, CANONICAL_SPACING_MM - float(s)))
            for s in spacing
        )
        if max(sigmas_per_axis) > 0:
            img = sitk.SmoothingRecursiveGaussian(img, sigmas_per_axis)
        ijk_to_ras_mat, ras_to_ijk_mat = image_ijk_ras_matrices(img)
        ijk_to_ras_mat = np.asarray(ijk_to_ras_mat, dtype=float)
        ras_to_ijk_mat = np.asarray(ras_to_ijk_mat, dtype=float)
    else:
        if ras_to_ijk_mat is None:
            ras_to_ijk_mat = np.linalg.inv(ijk_to_ras_mat)
        ras_to_ijk_mat = np.asarray(ras_to_ijk_mat, dtype=float)

    img = sitk.Clamp(img, lowerBound=-1024.0, upperBound=HU_CLIP_MAX)
    return img, ijk_to_ras_mat, ras_to_ijk_mat


def build_masks(img):
    """Hull mask + intracranial mask + signed-distance map from CT.

    Returns ``(hull_arr, intracranial_mask, dist_arr)``:

    - ``hull_arr``: bool array, True inside the closed head silhouette.
    - ``intracranial_mask``: bool array, True where ``dist_arr`` is at
      least ``INTRACRANIAL_MIN_DISTANCE_MM`` (avoids skull/bolt voxels).
    - ``dist_arr``: float32 signed Maurer distance from the hull
      surface (positive inside).
    """
    import SimpleITK as sitk
    # Cast to float so the threshold range is unrestricted by pixel type
    # (some CTs come in as uint16 / int16 where 1e9 overflows ITK's
    # type-coerced upper bound).
    img_f = sitk.Cast(img, sitk.sitkFloat32)
    thr = sitk.BinaryThreshold(img_f, lowerThreshold=-500.0, upperThreshold=1e9,
                                insideValue=1, outsideValue=0)
    cc = sitk.RelabelComponent(sitk.ConnectedComponent(thr), sortByObjectSize=True)
    largest = sitk.Equal(cc, 1)
    closed = sitk.BinaryMorphologicalClosing(largest, kernelRadius=(3, 3, 3))
    hull = sitk.BinaryFillhole(closed)
    dist = sitk.SignedMaurerDistanceMap(
        hull, insideIsPositive=True, squaredDistance=False, useImageSpacing=True,
    )
    dist_arr = sitk.GetArrayFromImage(dist).astype(np.float32)
    hull_arr = sitk.GetArrayFromImage(hull).astype(bool)
    intracranial = dist_arr >= INTRACRANIAL_MIN_DISTANCE_MM
    return hull_arr, intracranial, dist_arr


def frangi_single(img, sigma):
    """Frangi vesselness response at one scale (line-like detector)."""
    import SimpleITK as sitk
    sm = sitk.SmoothingRecursiveGaussian(img, sigma=float(sigma))
    ob = sitk.ObjectnessMeasure(sm, objectDimension=1, brightObject=True)
    return sitk.GetArrayFromImage(ob).astype(np.float32)


def log_sigma(img, sigma_mm):
    """Laplacian-of-Gaussian response at the given σ (mm).

    Local minima of this volume are the contact-blob candidates the
    pitch walker operates on; ``sigma_mm = 1.0`` matches a typical
    SEEG contact and is the production value (``LOG_SIGMA_MM``).
    """
    import SimpleITK as sitk
    log = sitk.LaplacianRecursiveGaussian(img, sigma=float(sigma_mm))
    return sitk.GetArrayFromImage(log).astype(np.float32)
