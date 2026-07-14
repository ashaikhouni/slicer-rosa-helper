"""Import a THOMAS thalamic segmentation (Su et al.) into a single combined
labelmap + color LUT the viewer can mesh and tint.

THOMAS writes one binary mask per nucleus into ``left/`` and ``right/``
subdirectories, named ``<THOMAS#>-<Name>.nii.gz`` (e.g. ``12-MD-Pf.nii.gz``),
all in the patient-T1 frame. This module reads the canonical NON-overlapping set
(skipping the whole-thalamus ``1`` and the ``VL`` / ``VLPd/v`` / ``GP``
aggregates that ENCLOSE the sub-nuclei — meshing those would re-create the
occlusion the deep-structure view exists to avoid) and combines them into one
int16 labelmap: left = THOMAS#, right = THOMAS# + 100. Left/right of a nucleus
share a color (anatomy convention).

Pure + headless: no SITK, no viewer, no filesystem writes — the caller decides
where the labelmap goes and how it reaches the CT frame (see the ``import-thomas``
CLI, which registers the T1 to the CT and warps this labelmap through it).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# label# -> (name, (r,g,b) in 0..1). The canonical THOMAS nuclei, minus the
# aggregates (1-THALAMUS, 4567-VL, 6_VLPd, 6_VLPv, 33-GP) which overlap the
# sub-nuclei they contain.
THOMAS_NUCLEI: dict[int, tuple[str, tuple[float, float, float]]] = {
    2:  ("AV",    (1.00, 0.84, 0.10)),
    4:  ("VA",    (1.00, 0.55, 0.05)),
    5:  ("VLa",   (1.00, 0.45, 0.38)),
    6:  ("VLP",   (0.90, 0.18, 0.18)),
    7:  ("VPL",   (0.87, 0.22, 0.62)),
    8:  ("Pul",   (0.60, 0.32, 0.82)),
    9:  ("LGN",   (0.05, 0.62, 0.60)),
    10: ("MGN",   (0.22, 0.80, 0.85)),
    11: ("CM",    (0.20, 0.90, 0.32)),
    12: ("MD-Pf", (0.22, 0.48, 0.96)),
    13: ("Hb",    (0.72, 0.86, 0.22)),
    14: ("MTT",   (0.70, 0.72, 0.76)),
    26: ("Acc",   (0.92, 0.62, 0.72)),
    27: ("Cau",   (0.36, 0.56, 0.92)),
    28: ("Cla",   (0.52, 0.72, 0.52)),
    29: ("GPe",   (0.82, 0.72, 0.42)),
    30: ("GPi",   (0.76, 0.56, 0.30)),
    31: ("Put",   (0.86, 0.46, 0.52)),
    32: ("RN",    (0.62, 0.20, 0.20)),
    34: ("Amy",   (0.92, 0.76, 0.52)),
}
RIGHT_OFFSET = 100   # right-hemisphere label = THOMAS# + 100


def _hemi_masks(hemi_dir: Path) -> dict[int, Path]:
    """Map THOMAS# -> mask file for one hemisphere, keeping only the canonical
    nuclei (exact ``<num>-<Name>``; the ``6_VLPd`` style subdivisions and other
    aggregates are ignored)."""
    out: dict[int, Path] = {}
    if not hemi_dir.is_dir():
        return out
    for num in THOMAS_NUCLEI:
        # a file whose prefix (up to the first '-') is exactly this number, and
        # whose separator is '-' (excludes 6_VLPd, which uses '_').
        cands = [
            f for f in hemi_dir.glob(f"{num}-*.nii.gz")
            if f.name[: len(str(num))] == str(num) and f.name[len(str(num))] == "-"
        ]
        if cands:
            out[num] = sorted(cands)[0]
    return out


def find_reference_t1(thomas_dir: Path) -> Path | None:
    """The intensity T1 the segmentation lives in — used to register into CT.
    THOMAS drops it at the top of the output dir as ``T1.nii.gz``."""
    for name in ("T1.nii.gz", "t1.nii.gz", "ref.nii.gz"):
        p = thomas_dir / name
        if p.is_file():
            return p
    return None


def build_thomas_labelmap(thomas_dir: str | Path) -> tuple[Any, dict[int, list], Path | None]:
    """Read a THOMAS output dir and return ``(labelmap_img, lut, t1_path)``.

    * ``labelmap_img`` — a ``nibabel.Nifti1Image`` (int16) in the T1 frame,
      left = THOMAS#, right = THOMAS# + 100.
    * ``lut`` — ``{label: [name, [r, g, b]]}`` (0..1 colors), left/right of a
      nucleus sharing a color; names suffixed ``-L`` / ``-R``.
    * ``t1_path`` — the reference T1 (for registration), or ``None``.

    Raises ``FileNotFoundError`` if the dir has no ``left/``/``right/`` masks.
    """
    import numpy as np
    import nibabel as nib

    thomas_dir = Path(thomas_dir)
    left = _hemi_masks(thomas_dir / "left")
    right = _hemi_masks(thomas_dir / "right")
    if not left and not right:
        raise FileNotFoundError(
            f"no THOMAS nucleus masks under {thomas_dir}/left or /right "
            "(expected files like '12-MD-Pf.nii.gz')")

    ref_img = None
    for num, path in (list(left.items()) + list(right.items())):
        ref_img = nib.load(str(path))
        break
    lab = np.zeros(ref_img.shape, dtype=np.int16)
    lut: dict[int, list] = {}

    def _stamp(masks: dict[int, Path], offset: int, suffix: str) -> None:
        for num, path in masks.items():
            m = np.asanyarray(nib.load(str(path)).dataobj) > 0
            lab[m] = num + offset
            name, rgb = THOMAS_NUCLEI[num]
            lut[num + offset] = [f"{name}-{suffix}", [float(c) for c in rgb]]

    _stamp(left, 0, "L")
    _stamp(right, RIGHT_OFFSET, "R")

    labelmap_img = nib.Nifti1Image(lab, ref_img.affine)
    return labelmap_img, lut, find_reference_t1(thomas_dir)
