"""Per-slot saturating-HU connected-component volume measurement.

Used by the unseeded-mode validator to drop bone-spike chains and surgical-clip
artifacts that the matched filter can't distinguish from real shanks.

Real PMT/DIXI contacts are 1.3 mm × 2 mm cylinders of platinum-iridium ≈
2.6 mm³ of saturating-HU metal. Adjacent contacts and the wire connecting
them inflate the saturating-HU CC up to ~140 mm³ within a 5 mm half-extent
ROI. Bone-spike chains, surgical clips, and multi-shank wire bundles do not
obey this physical bound — at least one slot lands in unbounded bone.
"""
from __future__ import annotations

import numpy as np

from .constants import CC_HU_THRESHOLD, CC_ROI_HALF_MM


def slot_cc_volume_mm3(
    ct_arr_kji: np.ndarray, r2i: np.ndarray, slot_ras: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    hu_threshold: float = CC_HU_THRESHOLD,
    roi_half_mm: float = CC_ROI_HALF_MM,
) -> float:
    """Volume (mm³) of the saturating-HU CC containing ``slot_ras``.

    Falls back to the nearest above-threshold voxel within an ROI cube of
    half-extent ``roi_half_mm``. Returns 0.0 if no above-threshold voxel
    exists in the ROI.
    """
    from scipy.ndimage import label as _cc_label

    pt_h = np.array([slot_ras[0], slot_ras[1], slot_ras[2], 1.0])
    ijk = (r2i @ pt_h)[:3]
    i_idx = int(round(ijk[0])); j_idx = int(round(ijk[1])); k_idx = int(round(ijk[2]))

    sx, sy, sz = float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])
    half_i = max(2, int(np.ceil(roi_half_mm / sx)))
    half_j = max(2, int(np.ceil(roi_half_mm / sy)))
    half_k = max(2, int(np.ceil(roi_half_mm / sz)))

    K, J, I = ct_arr_kji.shape
    k_lo = max(0, k_idx - half_k); k_hi = min(K, k_idx + half_k + 1)
    j_lo = max(0, j_idx - half_j); j_hi = min(J, j_idx + half_j + 1)
    i_lo = max(0, i_idx - half_i); i_hi = min(I, i_idx + half_i + 1)
    if k_hi <= k_lo or j_hi <= j_lo or i_hi <= i_lo:
        return 0.0

    roi = ct_arr_kji[k_lo:k_hi, j_lo:j_hi, i_lo:i_hi]
    mask = roi >= hu_threshold
    if not mask.any():
        return 0.0
    labels, _ = _cc_label(mask)

    rk = int(np.clip(k_idx - k_lo, 0, mask.shape[0] - 1))
    rj = int(np.clip(j_idx - j_lo, 0, mask.shape[1] - 1))
    ri = int(np.clip(i_idx - i_lo, 0, mask.shape[2] - 1))
    slot_label = int(labels[rk, rj, ri])
    if slot_label == 0:
        ks, js, is_ = np.where(mask)
        d = (ks - rk) ** 2 + (js - rj) ** 2 + (is_ - ri) ** 2
        n = int(np.argmin(d))
        slot_label = int(labels[ks[n], js[n], is_[n]])
    cc_voxels = int((labels == slot_label).sum())
    return cc_voxels * (sx * sy * sz)


__all__ = ["slot_cc_volume_mm3"]
