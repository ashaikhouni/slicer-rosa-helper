"""Burn a labelmap into a DICOM series' pixel data and write it back as a NEW
DICOM series that keeps the source patient/study identity + frame of reference.

This is the headless analogue of the Slicer navigation-burn path
(``rosa_scene.thomas_service.burn_thomas_nucleus_to_volume`` +
``rosa_scene.dicom_io_service.export_scalar_volume_to_dicom_series``), which
burns a THOMAS nucleus into a volume's intensity and exports it as a DICOM
series for surgical navigation. Here everything runs with **SimpleITK only** —
no Slicer, no MRML scene, no pydicom — so it works in the CLI/frozen sidecar.

The write mirrors the canonical SimpleITK "read-modify-write" recipe: carry the
patient/study-level tags forward, derive per-slice geometry from the (possibly
resampled) image, and mint fresh Series/SOP UIDs so the burned volume is a
distinct series *under the same study* rather than a clobber of the original.
Rescale + pixel-format tags are deliberately NOT copied — SimpleITK writes those
from the image buffer, and a stale ``RescaleIntercept`` would corrupt the HU.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Iterable

# Patient/study-level tags carried onto the burned series so a viewer/PACS still
# ties it to the same patient + study (as a new *series*). Frame-of-reference is
# kept so it stays spatially registered to the source acquisition.
_COPY_TAGS: tuple[str, ...] = (
    "0010|0010",  # Patient Name
    "0010|0020",  # Patient ID
    "0010|0030",  # Patient Birth Date
    "0010|0040",  # Patient Sex
    "0008|0020",  # Study Date
    "0008|0030",  # Study Time
    "0008|0050",  # Accession Number
    "0008|1030",  # Study Description
    "0020|0010",  # Study ID
    "0020|000d",  # Study Instance UID
    "0020|0052",  # Frame of Reference UID
    "0008|0060",  # Modality
    "0008|0070",  # Manufacturer
    "0018|0050",  # Slice Thickness
    "0028|0030",  # Pixel Spacing
)


def _uid() -> str:
    """A fresh, valid DICOM UID via the 2.25 (UUID-derived) arc — globally unique
    with no registered root and always ≤64 chars."""
    return f"2.25.{uuid.uuid4().int}"


def list_series(dicom_dir: str | Path) -> list[dict[str, Any]]:
    """``[{uid, n_slices}]`` for every DICOM series under ``dicom_dir``,
    largest-first."""
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    out = []
    for uid in reader.GetGDCMSeriesIDs(str(dicom_dir)):
        files = reader.GetGDCMSeriesFileNames(str(dicom_dir), uid)
        out.append({"uid": uid, "n_slices": len(files)})
    out.sort(key=lambda s: s["n_slices"], reverse=True)
    return out


def read_series(dicom_dir: str | Path, series_uid: str | None = None):
    """Read one DICOM series → ``(image, reader)``.

    ``image`` is the volume in modality units (HU for CT); ``reader`` retains the
    per-slice tag dictionaries (``MetaDataDictionaryArrayUpdate``) so the write
    can carry identity forward. Picks the largest series unless ``series_uid``
    pins one.
    """
    import SimpleITK as sitk

    series = list_series(dicom_dir)
    if not series:
        raise ValueError(f"no DICOM series found under {dicom_dir}")
    uids = {s["uid"] for s in series}
    chosen = series_uid or series[0]["uid"]           # series is largest-first
    if chosen not in uids:
        raise ValueError(f"series {chosen!r} not found; available: {sorted(uids)}")

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(dicom_dir), chosen))
    reader.MetaDataDictionaryArrayUpdateOn()          # keep per-slice tag dicts
    reader.LoadPrivateTagsOn()
    image = reader.Execute()
    return image, reader


def burn_labels(image, labelmap, labels: Iterable[int], fill_value: float):
    """Return ``(burned_image, n_voxels)``: a copy of ``image`` with every voxel
    whose ``labelmap`` value is in ``labels`` set to ``fill_value``.

    ``image`` and ``labelmap`` must share a grid (same size/spacing/origin — the
    labelmap is expected to have been resampled onto the image already). The burn
    is a straight intensity overwrite (like the Slicer path), so the structure
    becomes visible in the exported image itself.
    """
    import numpy as np
    import SimpleITK as sitk

    labels = list(dict.fromkeys(int(v) for v in labels))
    img_arr = sitk.GetArrayFromImage(image)           # [z, y, x], native dtype
    lab_arr = sitk.GetArrayFromImage(labelmap)
    if img_arr.shape != lab_arr.shape:
        raise ValueError(
            f"image {img_arr.shape} and labelmap {lab_arr.shape} are not on the "
            "same grid — resample the labelmap onto the image first")

    mask = np.isin(lab_arr, labels) if labels else np.zeros(lab_arr.shape, bool)
    n = int(mask.sum())
    if n:
        info = np.iinfo(img_arr.dtype) if np.issubdtype(img_arr.dtype, np.integer) else None
        if info is not None and not (info.min <= fill_value <= info.max):
            raise ValueError(
                f"fill value {fill_value} is outside the image dtype "
                f"{img_arr.dtype} range [{info.min}, {info.max}]")
        img_arr[mask] = fill_value

    burned = sitk.GetImageFromArray(img_arr)
    burned.CopyInformation(image)                     # preserve geometry exactly
    return burned, n


def burn_label_map(image, labelmap, fills):
    """Burn each label to its OWN intensity — for distinguishing multiple
    structures in one grayscale series (DICOM pixels can't carry color, so
    different HU is how a navigation station can window/LUT them apart).

    ``fills`` maps ``label → fill_value``. Returns ``(burned_image, {label:
    n_voxels})``. ``image`` and ``labelmap`` must share a grid.
    """
    import numpy as np
    import SimpleITK as sitk

    img_arr = sitk.GetArrayFromImage(image)
    lab_arr = sitk.GetArrayFromImage(labelmap)
    if img_arr.shape != lab_arr.shape:
        raise ValueError(
            f"image {img_arr.shape} and labelmap {lab_arr.shape} are not on the same grid")
    info = np.iinfo(img_arr.dtype) if np.issubdtype(img_arr.dtype, np.integer) else None
    counts: dict[int, int] = {}
    for label, fill in fills.items():
        if info is not None and not (info.min <= fill <= info.max):
            raise ValueError(
                f"fill {fill} for label {label} is outside the image dtype "
                f"{img_arr.dtype} range [{info.min}, {info.max}]")
        mask = lab_arr == int(label)
        counts[int(label)] = int(mask.sum())
        if counts[int(label)]:
            img_arr[mask] = fill
    burned = sitk.GetImageFromArray(img_arr)
    burned.CopyInformation(image)
    return burned, counts


def write_series(
    burned,
    reader,
    out_dir: str | Path,
    *,
    series_description: str = "THOMAS_BURNED",
    series_number: int | str = 9001,
) -> tuple[list[Path], str]:
    """Write ``burned`` as a new DICOM series into ``out_dir``.

    Carries the source study/patient identity + frame of reference forward (from
    ``reader``), derives per-slice ImagePositionPatient/Orientation from the
    ``burned`` geometry, and mints fresh Series/SOP UIDs. Returns
    ``(written_files, new_series_uid)``.
    """
    import SimpleITK as sitk

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()                   # honor the UIDs we set below

    series_uid = _uid()
    copied = [(t, reader.GetMetaData(0, t))
              for t in _COPY_TAGS if reader.HasMetaDataKey(0, t)]
    d = burned.GetDirection()
    orient = "\\".join(
        str(v) for v in (d[0], d[3], d[6], d[1], d[4], d[7]))

    written: list[Path] = []
    for i in range(burned.GetDepth()):
        sl = burned[:, :, i]
        for tag, val in copied:
            sl.SetMetaData(tag, val)
        sl.SetMetaData("0020|000e", series_uid)                     # Series Instance UID (new)
        sl.SetMetaData("0008|0018", _uid())                         # SOP Instance UID (new per slice)
        sl.SetMetaData("0008|103e", series_description)             # Series Description
        sl.SetMetaData("0020|0011", str(series_number))             # Series Number
        sl.SetMetaData("0008|0008", "DERIVED\\SECONDARY")           # Image Type
        sl.SetMetaData("0028|1052", "0")                            # Rescale Intercept
        sl.SetMetaData("0028|1053", "1")                            # Rescale Slope
        sl.SetMetaData("0020|0037", orient)                         # Image Orientation (Patient)
        sl.SetMetaData(
            "0020|0032",
            "\\".join(str(v) for v in burned.TransformIndexToPhysicalPoint((0, 0, i))))  # Image Position (Patient)
        sl.SetMetaData("0020|0013", str(i + 1))                     # Instance Number
        fn = out_dir / f"{i + 1:04d}.dcm"
        writer.SetFileName(str(fn))
        writer.Execute(sl)
        written.append(fn)

    return written, series_uid
