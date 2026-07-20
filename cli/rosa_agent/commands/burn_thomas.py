"""rosa-agent burn-thomas — burn a THOMAS thalamic structure into a DICOM's pixel
data and export it as a new DICOM series (for surgical navigation).

This is the headless/CLI equivalent of the Slicer **NavigationBurn** module. The
user supplies (1) a DICOM series to burn into (their CT or MRI) and (2) a THOMAS
output dir; the command registers the THOMAS T1 to the DICOM, warps the nucleus
labelmap onto the DICOM grid, overwrites those voxels' intensity, and writes a
new DICOM series that keeps the source patient/study identity — so the thalamic
target is visible in the image itself on a navigation station.

    rosa-agent burn-thomas ./ct_dicom ./THOMAS --out-dir ./ct_burned \\
        --nucleus VA --nucleus MD-Pf --side both --fill 1200

Burn one or more named nuclei (``--nucleus``, repeatable), or the whole thalamus
(``--all``). ``--side {left,right,both}`` picks hemispheres. Registration is rigid
Mattes MI (cross-modality); pass ``--transform`` to reuse a cached ``.tfm`` or
``--no-register`` when the THOMAS T1 already shares the DICOM's frame.

Unlike ``dicom-to-nifti``, the exported series deliberately KEEPS the source
patient/study identity (it is meant to go back to that patient's navigation
station, spatially co-registered under the same study) — so it is NOT
de-identified. Only the Series/SOP UIDs are freshly minted.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _resolve_labels(nuclei, side: str, want_all: bool):
    """Map requested nucleus names/numbers + side → the set of labelmap codes
    (left = THOMAS#, right = THOMAS# + RIGHT_OFFSET). Returns ``(labels, names)``
    or raises ``ValueError`` naming the valid nuclei."""
    from rosa_core.thomas_import import THOMAS_NUCLEI, RIGHT_OFFSET

    offsets = {"left": (0,), "right": (RIGHT_OFFSET,),
               "both": (0, RIGHT_OFFSET)}[side]
    name_to_num = {name.upper(): num for num, (name, _rgb) in THOMAS_NUCLEI.items()}

    if want_all:
        labels = {num + off for num in THOMAS_NUCLEI for off in offsets}
        return labels, ["<all>"]

    labels: set[int] = set()
    names: list[str] = []
    for token in nuclei:
        key = token.strip().upper()
        if key.isdigit() and int(key) in THOMAS_NUCLEI:
            num = int(key)
        elif key in name_to_num:
            num = name_to_num[key]
        else:
            valid = ", ".join(f"{n}({THOMAS_NUCLEI[n][0]})" for n in THOMAS_NUCLEI)
            raise ValueError(f"unknown nucleus {token!r}. Valid: {valid}")
        names.append(THOMAS_NUCLEI[num][0])
        for off in offsets:
            labels.add(num + off)
    return labels, names


def _write_legend(out_dir, legend, side: str) -> None:
    """Write ``burn_legend.tsv`` mapping each burned nucleus → its intensity, so a
    navigation station (or window/level, or a color LUT) can tell them apart."""
    lines = ["nucleus\tside\tfill"]
    lines += [f"{nm}\t{side}\t{f:g}" for nm, f in legend]
    (Path(out_dir) / "burn_legend.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="rosa-agent burn-thomas",
        description="Burn a THOMAS thalamic structure into a DICOM series' pixel "
                    "data and export it as a new DICOM series.")
    ap.add_argument("dicom_dir", help="DICOM series to burn into (CT or MRI)")
    ap.add_argument("thomas_dir", help="THOMAS output dir (has left/ + right/)")
    ap.add_argument("--out-dir", "-o", required=True, help="output dir for the burned DICOM series")
    ap.add_argument("--nucleus", "-n", action="append", default=[], metavar="NAME",
                    help="nucleus to burn, by name (VA, MD-Pf, …) or THOMAS number; repeatable")
    ap.add_argument("--all", action="store_true", help="burn every nucleus (whole thalamus)")
    ap.add_argument("--side", choices=("left", "right", "both"), default="both",
                    help="hemisphere(s) to burn (default: both)")
    ap.add_argument("--fill", type=float, default=1200.0,
                    help="intensity written into the structure (default 1200, e.g. HU for CT)")
    ap.add_argument("--distinct", action="store_true",
                    help="give each nucleus its own intensity (fill, fill+step, …) so multiple "
                         "structures stay separable in the grayscale series; writes burn_legend.tsv")
    ap.add_argument("--distinct-step", type=float, default=400.0,
                    help="intensity increment between nuclei when --distinct (default 400)")
    ap.add_argument("--series-description", default="THOMAS_BURNED",
                    help="SeriesDescription for the exported series")
    ap.add_argument("--series-uid", default=None,
                    help="input series UID to burn into (default: the largest)")
    ap.add_argument("--t1", default="",
                    help="reference image THOMAS ran in (T1 / FGATIR / WMnMPRAGE) to register onto "
                         "the DICOM; omit (with --no-register) to burn without registration")
    ap.add_argument("--transform", default="", help="cached reference→DICOM transform (.tfm) to reuse")
    ap.add_argument("--save-transform", default="", help="write the reference→DICOM transform here")
    ap.add_argument("--no-register", action="store_true",
                    help="skip registration (THOMAS already shares the DICOM frame)")
    ap.add_argument("--metal-clip-hu", type=float, default=1500.0,
                    help="clip the DICOM above this value during registration (metal bias)")
    args = ap.parse_args(argv)

    if not args.nucleus and not args.all:
        _stderr("error: pass at least one --nucleus, or --all")
        return 2
    dicom_dir = Path(args.dicom_dir)
    thomas_dir = Path(args.thomas_dir)
    if not dicom_dir.is_dir():
        _stderr(f"error: DICOM dir not found: {dicom_dir}")
        return 2
    if not thomas_dir.is_dir():
        _stderr(f"error: THOMAS dir not found: {thomas_dir}")
        return 2

    try:
        import SimpleITK as sitk
        from rosa_core.thomas_import import build_thomas_labelmap
        from rosa_core.registration import (
            register_rigid_mi, resample_volume, load_transform)
        from rosa_core import dicom_burn
    except ImportError as exc:
        _stderr(f"error: rosa_core unavailable ({exc})")
        return 2

    try:
        labels, names = _resolve_labels(args.nucleus, args.side, args.all)
    except ValueError as exc:
        _stderr(f"error: {exc}")
        return 2

    try:
        labelmap_img, _lut, _ref = build_thomas_labelmap(thomas_dir)
    except FileNotFoundError as exc:
        _stderr(f"error: {exc}")
        return 2
    except Exception as exc:  # noqa: BLE001 — clean message, not a stack trace
        _stderr(f"error: could not read THOMAS folder {thomas_dir}: {exc}")
        return 1
    _stderr(f"[burn-thomas] built THOMAS labelmap; burning {', '.join(names)} "
            f"({args.side}) → {len(labels)} label(s) at fill={args.fill}")

    # Read the target DICOM (keeps per-slice tags for the write-back).
    try:
        target, reader = dicom_burn.read_series(dicom_dir, args.series_uid)
    except (ValueError, RuntimeError) as exc:
        _stderr(f"error: reading DICOM: {exc}")
        return 1

    # THOMAS labelmap → SITK (via a temp NIfTI so its spatial metadata is
    # unambiguous). ReadImage loads it into memory, so the temp is discarded at
    # once — nothing is left behind in the user's output folder.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as _td:
        tmp = Path(_td) / "thomas_t1frame.nii.gz"
        labelmap_img.to_filename(str(tmp))
        lab_sitk = sitk.ReadImage(str(tmp))

    if args.no_register:
        transform = sitk.Transform(3, sitk.sitkIdentity)
        _stderr("[burn-thomas] --no-register: resampling labelmap onto the DICOM grid as-is")
    elif args.transform and Path(args.transform).is_file():
        transform = load_transform(args.transform)
        _stderr(f"[burn-thomas] reused cached transform {Path(args.transform).name}")
    else:
        # The reference image is provided explicitly (--t1); we do NOT look inside
        # the THOMAS folder, since it doesn't standardly hold the reference.
        if not args.t1:
            _stderr("error: no reference image for registration — pass --t1 with the "
                    "image THOMAS ran in (T1 / FGATIR / WMnMPRAGE), or use --no-register "
                    "if THOMAS already shares the DICOM's frame")
            return 2
        t1_path = Path(args.t1)
        if not t1_path.is_file():
            _stderr(f"error: reference image not found: {t1_path}")
            return 2
        t1_sitk = sitk.ReadImage(str(t1_path))
        _stderr(f"[burn-thomas] registering reference ({t1_path.name}) → DICOM (rigid MI)…")
        result = register_rigid_mi(
            fixed=target, moving=t1_sitk,
            metal_clip_hu=args.metal_clip_hu, logger=_stderr)
        transform = result.transform
        if args.save_transform:
            Path(args.save_transform).parent.mkdir(parents=True, exist_ok=True)
            sitk.WriteTransform(transform, str(args.save_transform))
            _stderr(f"[burn-thomas] saved transform → {args.save_transform}")

    warped = resample_volume(lab_sitk, transform, reference=target, interp="nearest")

    if args.distinct:
        # One intensity per nucleus (both hemispheres of a nucleus share it), so
        # multiple structures stay apart in the grayscale series. Legend → TSV.
        from rosa_core.thomas_import import THOMAS_NUCLEI, RIGHT_OFFSET
        offsets = {"left": (0,), "right": (RIGHT_OFFSET,),
                   "both": (0, RIGHT_OFFSET)}[args.side]
        name_to_num = {nm.upper(): num for num, (nm, _rgb) in THOMAS_NUCLEI.items()}
        burn_names = ([THOMAS_NUCLEI[num][0] for num in THOMAS_NUCLEI]
                      if args.all else names)
        fills, legend = {}, []
        for i, nm in enumerate(burn_names):
            f = args.fill + i * args.distinct_step
            for off in offsets:
                fills[name_to_num[nm.upper()] + off] = f
            legend.append((nm, f))
        burned, counts = dicom_burn.burn_label_map(target, warped, fills)
        n_vox = sum(counts.values())
        _write_legend(out_dir, legend, args.side)
        _stderr("[burn-thomas] distinct intensities → burn_legend.tsv:")
        for nm, f in legend:
            _stderr(f"[burn-thomas]   {nm} = {f:g}")
    else:
        burned, n_vox = dicom_burn.burn_labels(target, warped, labels, args.fill)
    if n_vox == 0:
        _stderr("warning: 0 voxels burned — the structure did not overlap the "
                "DICOM field of view (check the reference/registration or --side/--nucleus)")

    written, series_uid = dicom_burn.write_series(
        burned, reader, out_dir,
        series_description=args.series_description)
    _stderr(f"[burn-thomas] burned {n_vox} voxels; wrote {len(written)} slices → "
            f"{out_dir} (series {series_uid}, '{args.series_description}')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
