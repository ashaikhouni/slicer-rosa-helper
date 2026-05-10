"""LoG regional-minima blob extraction (Stage 1's input)."""
from __future__ import annotations

import numpy as np

from .constants import LOG_BLOB_SUBVOXEL_DEFAULT, LOG_BLOB_THRESHOLD


def extract_blobs(log_arr, threshold=LOG_BLOB_THRESHOLD, sub_voxel=None):
    """Regional-minima blob extraction. Each contact (local LoG minimum)
    becomes one blob. Uses SITK grayscale erode with a 3×3×3 Box kernel
    (26-connectivity, max reach √3 ≈ 1.73 voxels), then thresholds by
    absolute LoG value.

    Why this kernel:
      The local-min suppression neighbourhood must be wider than the
      LoG within-peak shoulders (≈1.5-2 mm FWHM at σ=1 mm) so each
      contact gives one detection, but strictly narrower than the
      smallest library contact pitch (3.5 mm) so adjacent contacts on
      the same shank both survive as distinct local minima. The
      previous SITK Ball at radius 2 had diagonal reach √6 ≈ 2.45
      voxels — fine on most subjects at 1 mm voxels but failed on the
      `(±1, ±1, ±2)` voxel-offset family where adjacent contacts on a
      shank happen to grid-snap (T7 LSFG: 5/8 contacts detected, 2 of
      those 5 then failed the walker's 0.5 mm pitch tolerance, line
      rejected).

      A 3×3×3 Box (corner reach √3 ≈ 1.73 mm at 1 mm voxels) sits
      cleanly between the within-peak FWHM and the contact pitch and
      is voxel-size-invariant up to spacing ≈ pitch/√3 ≈ 2 mm. On the
      dataset this recovers all LSFG-class shanks and preserves
      T25 LITG which other tighter kernels happen to lose.

    ``sub_voxel``: when True, refine each minimum's position to sub-voxel
    accuracy via a 1-D quadratic fit along each axis in the 3×3×3
    neighbourhood. This counteracts voxel-grid aliasing in blob-pair
    distances.
    """
    import SimpleITK as sitk
    if sub_voxel is None:
        sub_voxel = LOG_BLOB_SUBVOXEL_DEFAULT
    erode = sitk.GrayscaleErode(
        sitk.GetImageFromArray(log_arr),
        kernelRadius=[1, 1, 1],
        kernelType=sitk.sitkBox,
    )
    eroded = sitk.GetArrayFromImage(erode).astype(np.float32)
    is_local_min = (log_arr <= eroded + 1e-4)
    strong = is_local_min & (log_arr <= -abs(threshold))
    kk, jj, ii = np.where(strong)
    blobs = []
    K, J, I = log_arr.shape
    for k, j, i in zip(kk, jj, ii):
        val = float(log_arr[k, j, i])
        if sub_voxel and 0 < k < K - 1 and 0 < j < J - 1 and 0 < i < I - 1:
            fi_m = float(log_arr[k, j, i - 1]); fi_p = float(log_arr[k, j, i + 1])
            fj_m = float(log_arr[k, j - 1, i]); fj_p = float(log_arr[k, j + 1, i])
            fk_m = float(log_arr[k - 1, j, i]); fk_p = float(log_arr[k + 1, j, i])

            def _vtx(fm, f0, fp):
                denom = fm - 2.0 * f0 + fp
                if abs(denom) < 1e-6:
                    return 0.0
                off = 0.5 * (fm - fp) / denom
                return max(-0.5, min(0.5, off))
            di = _vtx(fi_m, val, fi_p)
            dj = _vtx(fj_m, val, fj_p)
            dk = _vtx(fk_m, val, fk_p)
            blobs.append(dict(
                kji=np.array([float(k) + dk, float(j) + dj, float(i) + di]),
                amp=-val, n_vox=1,
            ))
        else:
            blobs.append(dict(
                kji=np.array([float(k), float(j), float(i)]),
                amp=-val, n_vox=1,
            ))
    return blobs


def extract_blob_cloud_ras(log_arr, ijk_to_ras_mat, threshold=LOG_BLOB_THRESHOLD):
    """Return RAS centroids + amplitudes of every LoG regional minimum
    strong enough to be a contact candidate. Public wrapper around
    :func:`extract_blobs` so Auto Fit, Guided Fit, and Contacts &
    Trajectory View can share one entry point for "blobs as RAS points".
    """
    blobs = extract_blobs(log_arr, threshold=threshold)
    if not blobs:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)
    kji = np.array([b["kji"] for b in blobs], dtype=float)
    amps = np.array([b["amp"] for b in blobs], dtype=float)
    ij_k = np.stack([kji[:, 2], kji[:, 1], kji[:, 0]], axis=1)
    h = np.concatenate([ij_k, np.ones((ij_k.shape[0], 1))], axis=1)
    ras = (np.asarray(ijk_to_ras_mat, dtype=float) @ h.T).T[:, :3]
    return ras, amps


__all__ = ["extract_blobs", "extract_blob_cloud_ras"]
