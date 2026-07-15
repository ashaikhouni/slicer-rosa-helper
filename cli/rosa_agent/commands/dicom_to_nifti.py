"""rosa-agent dicom-to-nifti — convert a DICOM series to a de-identified NIfTI.

Reads a DICOM directory (a CT/MRI acquisition) with SimpleITK's GDCM series
reader and writes a ``.nii.gz``. Converting to NIfTI **drops the DICOM header
PHI** — patient name / ID / DOB, study/series dates, and the UID chain — so the
output is de-identified and safe to store + distribute while the source DICOM
stays put. This is the DICOM analogue of ``deidentify-ros``.

When a folder holds several series (e.g. localizer + CT), the largest (most
slices) is chosen unless ``--series-uid`` pins one. ``--list`` just reports the
series without converting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def list_series(dicom_dir: Path) -> list[dict[str, Any]]:
    """Return ``[{uid, n_slices}]`` for every DICOM series under ``dicom_dir``
    (recursively), largest first."""
    import SimpleITK as sitk
    reader = sitk.ImageSeriesReader()
    ids = list(reader.GetGDCMSeriesIDs(str(dicom_dir)))
    out = []
    for uid in ids:
        files = reader.GetGDCMSeriesFileNames(str(dicom_dir), uid)
        out.append({"uid": uid, "n_slices": len(files)})
    out.sort(key=lambda s: s["n_slices"], reverse=True)
    return out


def convert(dicom_dir: Path, out_path: Path, series_uid: str | None = None) -> Path:
    """Convert one DICOM series → NIfTI (de-identified). Picks the largest series
    unless ``series_uid`` is given."""
    import SimpleITK as sitk
    reader = sitk.ImageSeriesReader()
    series = list_series(dicom_dir)
    if not series:
        raise ValueError(
            f"no DICOM series found under {dicom_dir} "
            "(need a folder of .dcm slices)")
    uids = {s["uid"] for s in series}
    chosen = series_uid or series[0]["uid"]   # series[] is largest-first
    if chosen not in uids:
        raise ValueError(f"series {chosen!r} not found; available: {sorted(uids)}")
    files = reader.GetGDCMSeriesFileNames(str(dicom_dir), chosen)
    _stderr(f"[dicom-to-nifti] {len(series)} series found; converting "
            f"{chosen} ({len(files)} slices)")
    reader.SetFileNames(files)
    img = reader.Execute()
    # NIfTI carries no DICOM tags, so the write itself de-identifies. We do NOT
    # copy any DICOM metadata onto the image.
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_path))
    sz, sp = img.GetSize(), img.GetSpacing()
    _stderr(f"[dicom-to-nifti] wrote {out_path} "
            f"(size={tuple(sz)} spacing={tuple(round(s, 3) for s in sp)} — de-identified)")
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="rosa-agent dicom-to-nifti",
        description="Convert a DICOM series to a de-identified NIfTI.")
    p.add_argument("dicom_dir", help="directory containing a DICOM series")
    p.add_argument("--out", "-o", default="", help="output .nii.gz (required unless --list)")
    p.add_argument("--series-uid", default=None,
                   help="convert this specific series UID (default: the largest)")
    p.add_argument("--list", action="store_true",
                   help="list the series found (uid + slice count) and exit")
    args = p.parse_args(argv)

    d = Path(args.dicom_dir)
    if not d.is_dir():
        _stderr(f"error: not a directory: {d}")
        return 2
    try:
        if args.list:
            for s in list_series(d):
                print(f"{s['uid']}\t{s['n_slices']}")
            return 0
        if not args.out:
            _stderr("error: --out/-o is required (or pass --list)")
            return 2
        convert(d, Path(args.out), args.series_uid)
    except Exception as exc:  # noqa: BLE001 — surface a clean one-liner
        _stderr(f"error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
