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


def _readable(p: Path) -> bool:
    """A real, non-empty file — guards against 0-byte cloud-storage placeholders
    (e.g. a THOMAS output still syncing on Dropbox) before we try to load it."""
    try:
        return p.is_file() and p.stat().st_size > 0
    except OSError:
        return False


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


def _hemi_binary_masks(hemi_dir: Path, nib, np):
    """Return ``({custom_label: bool_mask_3d}, affine)`` for one hemisphere.

    Prefers the single combined ``thomasfull_{L,R}.nii.gz`` labelmap when THOMAS
    dropped one — it carries the *documented* custom nucleus codes (AV=2, Pul=8,
    CM=11, Cla=28 …), so we read the whole hemisphere from one file instead of
    ~20 per-nucleus masks. Falls back to the individual masks otherwise.
    """
    combined = None
    for nm in ("thomasfull_L.nii.gz", "thomasfull_R.nii.gz",
               "thomasfull_l.nii.gz", "thomasfull_r.nii.gz"):
        p = hemi_dir / nm
        if _readable(p):
            try:
                combined = nib.load(str(p))
                break
            except Exception:                      # noqa: BLE001
                # A 0-byte / half-synced placeholder (common on Dropbox) or a
                # corrupt file: don't crash — fall back to the per-nucleus masks.
                combined = None
    if combined is not None:
        arr = np.asanyarray(combined.dataobj)
        masks = {int(v): (arr == v) for v in np.unique(arr)
                 if int(v) in THOMAS_NUCLEI}
        if masks:
            return masks, combined.affine
    masks, affine = {}, None
    for num, path in _hemi_masks(hemi_dir).items():
        if not _readable(path):
            continue
        try:
            im = nib.load(str(path))
        except Exception:                          # noqa: BLE001
            continue
        affine = im.affine
        masks[num] = np.asanyarray(im.dataobj) > 0
    return masks, affine


def find_reference_t1(thomas_dir: Path) -> Path | None:
    """The intensity image the segmentation lives in — used to register into CT.

    THOMAS is usually built off a high-thalamic-contrast sequence and drops it at
    the top of the output dir; accept the common names (a plain ``T1``, or an
    ``FGATIR`` / ``WMnMPRAGE`` build). When THOMAS was run off an image that is
    NOT stored beside its output, the caller must supply the path explicitly
    (``import-thomas --t1`` / ``burn-thomas --t1``)."""
    for name in ("T1.nii.gz", "t1.nii.gz", "ref.nii.gz",
                 "fgatir.nii.gz", "FGATIR.nii.gz",
                 "WMnMPRAGE.nii.gz", "wmnmprage.nii.gz", "mprage.nii.gz"):
        p = thomas_dir / name
        if _readable(p):
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
    # Prefer the combined per-hemisphere labelmap (thomasfull_{L,R}); fall back
    # to the individual per-nucleus masks. Both use the documented custom codes.
    left, left_aff = _hemi_binary_masks(thomas_dir / "left", nib, np)
    right, right_aff = _hemi_binary_masks(thomas_dir / "right", nib, np)
    if not left and not right:
        raise FileNotFoundError(
            f"no THOMAS nuclei under {thomas_dir}/left or /right "
            "(expected a thomasfull_{L,R}.nii.gz or masks like '12-MD-Pf.nii.gz')")

    affine = left_aff if left_aff is not None else right_aff
    shape = next(iter((left or right).values())).shape
    lab = np.zeros(shape, dtype=np.int16)
    lut: dict[int, list] = {}

    def _stamp(masks: dict[int, Any], offset: int, suffix: str) -> None:
        for num, mask in masks.items():
            lab[mask] = num + offset
            name, rgb = THOMAS_NUCLEI[num]
            lut[num + offset] = [f"{name}-{suffix}", [float(c) for c in rgb]]

    _stamp(left, 0, "L")
    _stamp(right, RIGHT_OFFSET, "R")

    labelmap_img = nib.Nifti1Image(lab, affine)
    return labelmap_img, lut, find_reference_t1(thomas_dir)
