"""Intracranial brain-mask generation for CT (and CT-like) volumes.

The :func:`log_watershed_brain_mask` recipe was validated on a 10-subject
hand-curated cohort (T1, T2, T18, T22, T25 + AMC88, AMC91, AMC135, AMC136,
AMC137) — mean GT contact recall 97.3 %, mean leak 0.5 % of mask volume.
See ``notebooks/cross_modality/03_cross_subject_validation.ipynb`` for the
per-subject scorecard.

Three knobs are combined into the recipe:

1. **LoG wall mask** — ``|LoG σ=5 mm| ≥ 10`` fires on every ~5 mm-scale dense
   edge (skull, bolts, frame). Validated by ``probes/intracranial_mask_lab.ipynb``
   Variant F across T1/T2/T18/T22.
2. **Intracranial-metal carve** — saturated metal (HU ≥ 2000) deeper than
   10 mm from the head hull is *removed* from the wall mask. Without this,
   high-contrast contacts (HU 6000–15000 on some scanners) fire as walls
   and fragment the brain basin.
3. **Region interior marker** — every voxel with ``dist_hull ≥ 30 mm`` is a
   marker. Robust against single-voxel-seed traps. Falls back to ``p99(dist_hull)``
   when the FOV is too tight to have many >30 mm-deep voxels.

The watershed itself is SimpleITK's ``MorphologicalWatershedFromMarkers`` with
two label classes: the volume boundary (label 1 — extracranial) and the deep
seed (label 2 — intracranial). Both flood through the negated wall-distance
field, meeting at the wall ridges.
"""

from __future__ import annotations

from typing import Any

from .preprocessing import build_masks, log_sigma


# Defaults validated on the 10-subject cohort. Anatomically motivated, not
# per-scan tuned.
DEFAULT_SIGMA_MM            = 5.0
DEFAULT_WALL_THR            = 10.0
DEFAULT_METAL_HU_FLOOR      = 2000.0
DEFAULT_INTRACRANIAL_MIN_MM = 10.0
DEFAULT_DEEP_SEED_MM        = 30.0


def log_watershed_brain_mask(
    ct: Any,
    *,
    sigma_mm: float = DEFAULT_SIGMA_MM,
    wall_thr: float = DEFAULT_WALL_THR,
    metal_hu_floor: float = DEFAULT_METAL_HU_FLOOR,
    intracranial_min_mm: float = DEFAULT_INTRACRANIAL_MIN_MM,
    deep_seed_mm: float = DEFAULT_DEEP_SEED_MM,
) -> Any:
    """LoG-watershed intracranial brain mask for a CT volume.

    Args:
        ct: SimpleITK image (CT in HU). Geometry (origin/direction/spacing)
            is preserved on the output mask.
        sigma_mm: LoG kernel sigma in mm. Default 5 mm picks up skull-scale
            edges; smaller values risk treating individual contacts as walls.
        wall_thr: threshold on ``|LoG σ|`` above which a voxel is a wall.
            Default 10 is the empirically validated value from Variant F.
        metal_hu_floor: HU above which voxels count as "saturated metal" for
            the intracranial-metal carve. Default 2000.
        intracranial_min_mm: hull-distance threshold for "intracranial". Metal
            voxels deeper than this are carved out of the wall mask. Skull-
            embedded bolts (hull-distance < this) stay as walls.
        deep_seed_mm: hull-distance threshold for the deep interior marker.
            Default 30 mm gives thousands of guaranteed-brain seed voxels.
            Falls back to ``p99(dist_hull)`` if the FOV is too tight.

    Returns:
        ``sitk.Image`` (uint8, 0/1) with the same geometry as ``ct``. ``None``
        is returned if no interior marker can be placed (degenerate FOV) —
        callers should treat this as "mask generation failed".
    """
    import SimpleITK as sitk
    import numpy as np

    post_a = sitk.GetArrayFromImage(ct).astype(np.float32)
    _, _, dist_hull = build_masks(ct)
    log5 = log_sigma(ct, sigma_mm=sigma_mm)
    walls_raw = np.abs(log5) >= wall_thr

    # Carve intracranial saturated metal out of the wall mask so contacts /
    # deep bolt tips / wires don't fragment the brain basin.
    intracranial_metal = (post_a >= metal_hu_floor) & (dist_hull >= intracranial_min_mm)
    walls = walls_raw & ~intracranial_metal

    # Region-based interior marker — many seeds, not one trapped voxel.
    deep_seed = (dist_hull >= deep_seed_mm) & ~walls
    if deep_seed.sum() < 100:
        thresh = float(np.percentile(dist_hull, 99))
        deep_seed = (dist_hull >= thresh) & ~walls

    markers = np.zeros(post_a.shape, dtype=np.uint8)
    markers[ 0, :, :] = 1; markers[-1, :, :] = 1
    markers[ :, 0, :] = 1; markers[ :,-1, :] = 1
    markers[ :, :, 0] = 1; markers[ :, :,-1] = 1
    markers[deep_seed] = 2

    if (markers == 2).sum() < 50:
        return None

    walls_img = sitk.GetImageFromArray(walls.astype(np.uint8))
    walls_img.CopyInformation(ct)
    dist_w = sitk.GetArrayFromImage(
        sitk.SignedMaurerDistanceMap(
            walls_img,
            insideIsPositive=False,
            squaredDistance=False,
            useImageSpacing=True,
        ),
    ).astype(np.float32)
    neg_img = sitk.GetImageFromArray(-dist_w)
    neg_img.CopyInformation(ct)
    markers_img = sitk.GetImageFromArray(markers)
    markers_img.CopyInformation(ct)

    ws = sitk.GetArrayFromImage(
        sitk.MorphologicalWatershedFromMarkers(
            sitk.Cast(neg_img, sitk.sitkFloat32),
            markers_img,
            markWatershedLine=False,
            fullyConnected=False,
        ),
    )
    brain = (ws == 2) & ~walls

    mask_img = sitk.GetImageFromArray(brain.astype(np.uint8))
    mask_img.CopyInformation(ct)
    return mask_img
