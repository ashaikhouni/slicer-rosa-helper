"""LoG-watershed brain-mask backend.

CT-only fallback for SynthStrip — no deep-learning dep, runs in ~3 s on a
typical postop CT. Wraps :func:`rosa_detect.primitives.intracranial.log_watershed_brain_mask`
with the same disk-IO contract the SynthStrip wrapper exposes: input path
in, mask path out, mask inherits the source volume's full geometry.

See ``notebooks/cross_modality/03_cross_subject_validation.ipynb`` for the
cohort scorecard (mean recall 97.3 %, mean leak 0.5 %).
"""

from __future__ import annotations

from pathlib import Path


class LogWatershedFailed(RuntimeError):
    """Raised when the LoG watershed degenerates (no interior marker placeable)."""


class NotACTVolume(ValueError):
    """Raised when log-watershed is asked to run on a non-CT volume.

    The recipe relies on HU calibration (bone band, metal floor, hull
    distance from a −500 HU silhouette). It produces meaningless masks on
    MRI / PET / other non-HU modalities.
    """


def looks_like_ct(image, *, air_hu: float = -500.0, min_air_frac: float = 0.01) -> bool:
    """Heuristic: returns True if `image` appears to be a CT volume.

    CT volumes have abundant air voxels (< −500 HU). MRI / PET / SUV volumes
    do not have negative intensity at all. We require at least 1 % of the
    volume to be below `air_hu` for the call to count as CT.
    """
    import numpy as np
    import SimpleITK as sitk

    arr = sitk.GetArrayViewFromImage(image)
    air_frac = float((arr < air_hu).mean())
    return air_frac >= min_air_frac


def run_log_watershed(
    input_path: str | Path,
    mask_path: str | Path,
    *,
    sigma_mm: float = 5.0,
    wall_thr: float = 10.0,
    metal_hu_floor: float = 2000.0,
    intracranial_min_mm: float = 10.0,
    deep_seed_mm: float = 30.0,
) -> Path:
    """Compute a LoG-watershed brain mask and write it to disk.

    Args:
        input_path: path to the input CT volume (NIfTI / Analyze / MGZ — any
            SITK-readable format).
        mask_path: path where the brain mask should be written.
        sigma_mm / wall_thr / metal_hu_floor / intracranial_min_mm /
        deep_seed_mm: forwarded to
            :func:`rosa_detect.primitives.intracranial.log_watershed_brain_mask`.

    Returns:
        Path to the written mask.

    Raises:
        LogWatershedFailed: when the algorithm couldn't place an interior
            marker (degenerate FOV, all-air volume, etc).
    """
    import SimpleITK as sitk

    from rosa_detect.primitives.intracranial import log_watershed_brain_mask

    input_path = Path(input_path).expanduser().resolve()
    mask_path  = Path(mask_path).expanduser().resolve()
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    ct = sitk.ReadImage(str(input_path))
    if not looks_like_ct(ct):
        raise NotACTVolume(
            f"{input_path.name} does not look like a CT volume "
            f"(no air-band voxels < -500 HU). The LoG-watershed backend is "
            f"calibrated to CT intensity and produces meaningless masks on "
            f"MRI / PET / other modalities. Use `--backend synthstrip` for "
            f"non-CT inputs."
        )
    mask_img = log_watershed_brain_mask(
        ct,
        sigma_mm=sigma_mm,
        wall_thr=wall_thr,
        metal_hu_floor=metal_hu_floor,
        intracranial_min_mm=intracranial_min_mm,
        deep_seed_mm=deep_seed_mm,
    )
    if mask_img is None:
        raise LogWatershedFailed(
            f"LoG watershed could not place a deep-brain seed on {input_path}. "
            f"This typically means the FOV is too tight or the volume isn't a head CT."
        )

    sitk.WriteImage(mask_img, str(mask_path))
    return mask_path
