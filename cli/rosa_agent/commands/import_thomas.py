"""rosa-agent import-thomas — bring an EXTERNAL THOMAS thalamic segmentation into
a case's CT frame as a deep-structure labelmap + color LUT the viewer can mesh.

THOMAS runs outside the app on the patient T1; this command reads its per-nucleus
masks (``rosa_core.thomas_import``), registers the THOMAS T1 to the case CT
(rigid Mattes MI, cross-modality), and warps the combined labelmap through that
transform onto the CT grid — so the nuclei land in the same frame as the
localized electrodes. ``view-results --structure-meshes`` then renders them.

    rosa-agent import-thomas /path/THOMAS --ct case_ct.nii.gz \\
        -o thomas_in_ct.nii.gz --lut-out thomas_lut.json

With no ``--ct`` it writes the labelmap in the THOMAS T1 frame unchanged (for a
segmentation that already shares the contact frame). ``--t1-to-ct`` reuses a
cached transform (skip registration); ``--save-transform`` caches a fresh one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="rosa-agent import-thomas",
        description="Import a THOMAS thalamic segmentation into a case's CT frame "
                    "as a deep-structure labelmap + LUT.")
    ap.add_argument("thomas_dir", help="THOMAS output dir (has left/ + right/ + T1.nii.gz)")
    ap.add_argument("--output", "-o", required=True, help="output labelmap NIfTI (in CT frame)")
    ap.add_argument("--lut-out", required=True, help="output LUT JSON {label:[name,[r,g,b]]}")
    ap.add_argument("--ct", default="", help="case CT (contact frame). Omit to keep the T1 frame.")
    ap.add_argument("--t1", default="", help="reference T1 (overrides THOMAS dir's T1.nii.gz)")
    ap.add_argument("--t1-to-ct", default="", help="cached T1→CT transform (.tfm) to reuse")
    ap.add_argument("--save-transform", default="", help="write the T1→CT transform here")
    ap.add_argument("--metal-clip-hu", type=float, default=1500.0,
                    help="clip CT above this HU during registration so metal doesn't bias MI")
    args = ap.parse_args(argv)

    try:
        from rosa_core.thomas_import import build_thomas_labelmap
    except ImportError as exc:
        _stderr(f"error: rosa_core unavailable ({exc})")
        return 2

    thomas_dir = Path(args.thomas_dir)
    if not thomas_dir.is_dir():
        _stderr(f"error: THOMAS dir not found: {thomas_dir}")
        return 2

    try:
        labelmap_img, lut, ref_t1 = build_thomas_labelmap(thomas_dir)
    except FileNotFoundError as exc:
        _stderr(f"error: {exc}")
        return 2

    n = len([v for v in lut])
    _stderr(f"[import-thomas] built labelmap: {n} structures from {thomas_dir.name}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.ct:
        import nibabel as nib  # noqa: F401 (labelmap_img is a nibabel image)
        import SimpleITK as sitk
        from rosa_core.registration import (
            register_rigid_mi, resample_volume, load_transform)

        t1_path = Path(args.t1) if args.t1 else ref_t1
        if t1_path is None or not Path(t1_path).is_file():
            _stderr("error: no reference T1 for registration (pass --t1 or ensure "
                    f"{thomas_dir}/T1.nii.gz exists)")
            return 2

        # Persist the labelmap in the T1 frame, then reload via SITK so its spatial
        # metadata is unambiguous for resampling.
        tmp_t1frame = out.with_name(out.stem.replace(".nii", "") + "_t1frame.nii.gz")
        labelmap_img.to_filename(str(tmp_t1frame))
        lab_sitk = sitk.ReadImage(str(tmp_t1frame))
        ct_sitk = sitk.ReadImage(str(args.ct))

        if args.t1_to_ct and Path(args.t1_to_ct).is_file():
            transform = load_transform(args.t1_to_ct)
            _stderr(f"[import-thomas] reused cached transform {Path(args.t1_to_ct).name}")
        else:
            t1_sitk = sitk.ReadImage(str(t1_path))
            _stderr(f"[import-thomas] registering T1 → CT (rigid MI)…")
            result = register_rigid_mi(
                fixed=ct_sitk, moving=t1_sitk,
                metal_clip_hu=args.metal_clip_hu, logger=_stderr)
            transform = result.transform
            if args.save_transform:
                Path(args.save_transform).parent.mkdir(parents=True, exist_ok=True)
                sitk.WriteTransform(transform, str(args.save_transform))
                _stderr(f"[import-thomas] saved transform → {args.save_transform}")

        warped = resample_volume(lab_sitk, transform, reference=ct_sitk, interp="nearest")
        sitk.WriteImage(warped, str(out))
        try:
            tmp_t1frame.unlink()
        except OSError:
            pass
        _stderr(f"[import-thomas] wrote {out} (labelmap warped into CT frame)")
    else:
        labelmap_img.to_filename(str(out))
        _stderr(f"[import-thomas] wrote {out} (T1 frame, no --ct given)")

    Path(args.lut_out).write_text(
        json.dumps({str(k): v for k, v in lut.items()}, indent=0), encoding="utf-8")
    _stderr(f"[import-thomas] wrote {args.lut_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
