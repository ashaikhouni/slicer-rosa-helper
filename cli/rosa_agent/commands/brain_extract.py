"""rosa-agent brain-extract — produce a brain mask from a CT or MRI volume.

Three backends with different modality coverage — the user must pass the
volume that matches the chosen backend:

* **synthstrip** — FreeSurfer's deep-learning binary ``mri_synthstrip``.
  Works on **CT and MRI** (T1, T2, FLAIR, ...). Requires the
  binary on disk (auto-detected; see ``--synthstrip``). Ships with
  FreeSurfer 7.3+. CUDA-or-CPU only (no Metal), so ~100 s on a Mac.
* **deepbet** — fast MIT-licensed CNN skull-strip, **T1 MRI only**. Runs on
  Apple Metal (MPS) in well under a second. Refuses nothing at the CLI
  level, but is trained on T1 and does NOT reliably strip CT — pass a T1.
  Needs a python that can ``import deepbet`` (see ``--deepbet-python`` /
  ``$ROSA_DEEPBET_PYTHON``); runs as a subprocess (torch isolation).
* **log-watershed** — pure-Python **CT-only** fallback. Calibrated on
  HU intensity (bone band, metal floor, hull distance from −500 HU).
  Refuses to run on MRI / PET / non-HU modalities. Validated on a
  10-subject cohort at ~97 % GT contact recall, ~0.5 % leak; ~3 s runtime.
  No external dependencies beyond what ``rosa_core`` already imports.

Usage::

    rosa-agent brain-extract postop_ct.nii.gz -o brain_mask.nii.gz
    rosa-agent brain-extract postop_ct.nii.gz -o mask.nii.gz --backend log-watershed
    rosa-agent brain-extract postop_ct.nii.gz -o mask.nii.gz --backend auto   # default

``--backend auto`` (default) picks SynthStrip when available and falls
through to the LoG watershed otherwise.

Exit codes:
* 0 — mask written
* 2 — input missing or argument error
* 3 — chosen backend not available (only fatal with ``--backend synthstrip``)
* 4 — backend ran but failed
* 5 — modality mismatch (log-watershed asked to run on a non-CT volume)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from rosa_detect.services.synthstrip import (
    SynthStripNotFound,
    find_synthstrip,
    run_synthstrip,
)
from rosa_detect.services.log_watershed import (
    LogWatershedFailed,
    NotACTVolume,
    run_log_watershed,
)


VALID_BACKENDS = ("auto", "synthstrip", "log-watershed", "deepbet")


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _run_synthstrip_backend(args, input_path: Path, log) -> tuple[int, Path | None]:
    """Returns (exit_code, mask_path_or_None). exit_code 3 = not-found."""
    try:
        mask = run_synthstrip(
            input_path=input_path,
            mask_path=args.mask,
            synthstrip_path=args.synthstrip,
            stripped_path=args.stripped,
            no_csf=args.no_csf,
            threads=args.threads,
            timeout=args.timeout,
        )
        return 0, mask
    except SynthStripNotFound as exc:
        _stderr(f"error: {exc}")
        return 3, None
    except subprocess.TimeoutExpired as exc:
        _stderr(f"error: mri_synthstrip timed out after {exc.timeout:.0f}s")
        return 4, None
    except subprocess.CalledProcessError as exc:
        _stderr(f"error: mri_synthstrip failed (exit {exc.returncode}):")
        if exc.stderr:
            _stderr(exc.stderr.decode("utf-8", errors="replace"))
        return 4, None


def _run_deepbet_backend(args, input_path: Path, log) -> tuple[int, Path | None]:
    """Returns (exit_code, mask_path_or_None). exit_code 3 = deepbet not found."""
    from rosa_detect.services.deepbet_strip import run_deepbet, DeepbetNotFound
    try:
        mask = run_deepbet(
            input_path=input_path,
            mask_path=args.mask,
            deepbet_python=args.deepbet_python,
            no_gpu=args.deepbet_cpu,
            timeout=args.timeout,
            log=log,
        )
        return 0, mask
    except DeepbetNotFound as exc:
        _stderr(f"error: {exc}")
        return 3, None
    except subprocess.TimeoutExpired as exc:
        _stderr(f"error: deepbet timed out after {exc.timeout:.0f}s")
        return 4, None
    except subprocess.CalledProcessError as exc:
        _stderr(f"error: deepbet failed (exit {exc.returncode}):")
        if exc.stderr:
            _stderr(exc.stderr.decode("utf-8", errors="replace")[:800])
        return 4, None


def _run_log_watershed_backend(args, input_path: Path, log) -> tuple[int, Path | None]:
    try:
        mask = run_log_watershed(input_path=input_path, mask_path=args.mask)
        return 0, mask
    except NotACTVolume as exc:
        _stderr(f"error: {exc}")
        # Specific exit code so callers can distinguish "wrong modality"
        # from "algorithm degenerate" or "synthstrip missing".
        return 5, None
    except LogWatershedFailed as exc:
        _stderr(f"error: {exc}")
        return 4, None
    except Exception as exc:  # noqa: BLE001
        _stderr(f"error: log-watershed backend crashed: {exc!r}")
        return 4, None


def _register_and_emit(args, input_path: Path, mask: Path, log) -> int:
    """When ``--register-to`` is set, register the INPUT to that target (rigid MI)
    and emit the ``--save-transform`` (.tfm) and/or ``--mask-in-target`` (mask
    resampled into the target's grid). Returns 0 on success / nothing to do, or a
    non-zero exit code on error. Runs in-process (SITK); deepbet's torch stays in
    its own subprocess, so there's no libomp clash here.
    """
    if not args.register_to:
        if args.save_transform or args.mask_in_target:
            _stderr("error: --save-transform / --mask-in-target require --register-to")
            return 2
        return 0
    target = Path(args.register_to).expanduser()
    if not target.is_file():
        _stderr(f"error: --register-to volume not found: {target}")
        return 2
    try:
        import SimpleITK as sitk
        from rosa_core.registration import (
            register_rigid_mi, resample_volume, save_transform,
        )
        fixed = sitk.ReadImage(str(target))        # target frame (e.g. the CT)
        moving = sitk.ReadImage(str(input_path))   # what we stripped (e.g. the T1)
        log(f"[brain-extract] registering {input_path.name} → {target.name} (rigid MI)…")
        reg = register_rigid_mi(fixed=fixed, moving=moving, logger=log)
        if args.save_transform:
            tp = Path(args.save_transform).expanduser()
            tp.parent.mkdir(parents=True, exist_ok=True)
            save_transform(reg.transform, str(tp))
            log(f"[brain-extract] wrote transform {tp}")
        if args.mask_in_target:
            mp = Path(args.mask_in_target).expanduser()
            mp.parent.mkdir(parents=True, exist_ok=True)
            # The native mask shares the input's grid, so resample it through the
            # input→target transform onto the target grid (nearest = keep 0/1).
            warped = resample_volume(
                sitk.ReadImage(str(mask)), reg.transform,
                reference=fixed, interp="nearest",
            )
            sitk.WriteImage(warped, str(mp))
            log(f"[brain-extract] wrote mask in target frame {mp}")
        return 0
    except Exception as exc:  # noqa: BLE001
        _stderr(f"error: register/warp failed: {exc!r}")
        return 4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rosa-agent brain-extract",
        description=(
            "Produce a brain mask from a CT or MRI volume. Default backend is "
            "SynthStrip when available, falling through to a CT-only LoG-watershed."
        ),
    )
    parser.add_argument("input", nargs="?", default=None,
                        help="input volume (.nii/.nii.gz/.mgz); omit with --check")
    parser.add_argument(
        "-o", "--mask", default=None,
        help="output brain-mask path (NIfTI); required unless --check",
    )
    parser.add_argument(
        "--backend", choices=VALID_BACKENDS, default="auto",
        help="which backend to use. 'synthstrip' handles CT and MRI; "
             "'deepbet' is a fast T1-MRI-only CNN (Metal-capable); "
             "'log-watershed' handles CT only (refuses non-CT input). "
             "'auto' (default) uses synthstrip when available and falls "
             "through to log-watershed for CT volumes.",
    )
    parser.add_argument(
        "--stripped", default=None,
        help="optional output for the skull-stripped volume "
             "(SynthStrip backend only)",
    )
    parser.add_argument(
        "--synthstrip", default=None,
        help="explicit path to mri_synthstrip binary "
             "(otherwise probed via $ROSA_SYNTHSTRIP, PATH, "
             "$FREESURFER_HOME/bin/, or common install paths)",
    )
    parser.add_argument(
        "--no-csf", action="store_true",
        help="pass --no-csf to SynthStrip (SynthStrip backend only)",
    )
    parser.add_argument(
        "--threads", type=int, default=None,
        help="thread count passed to SynthStrip's -t flag (SynthStrip backend only)",
    )
    parser.add_argument(
        "--timeout", type=float, default=None,
        help="seconds before killing the subprocess (SynthStrip / deepbet backends)",
    )
    parser.add_argument(
        "--deepbet-python", default=None,
        help="python interpreter that can `import deepbet` (deepbet backend); "
             "otherwise probed via $ROSA_DEEPBET_PYTHON then the current python",
    )
    parser.add_argument(
        "--deepbet-cpu", action="store_true",
        help="force deepbet onto CPU instead of Metal/CUDA (deepbet backend only)",
    )
    # --- optional: register the input to a target and emit the mask in its frame ---
    parser.add_argument(
        "--register-to", default=None,
        help="after extracting the mask, rigidly register the INPUT to this target "
             "volume (Mattes MI) so the mask can be expressed in the target's frame. "
             "Enables --save-transform / --mask-in-target. Typical use: T1 mask "
             "-> CT frame for the placement anchor.",
    )
    parser.add_argument(
        "--save-transform", default=None,
        help="write the input->target rigid transform here (.tfm); requires "
             "--register-to. Reusable downstream (e.g. view-results "
             "--brain-to-ct-transform) so the registration runs once per case.",
    )
    parser.add_argument(
        "--mask-in-target", default=None,
        help="write the brain mask resampled into the target's grid here "
             "(nearest-neighbour); requires --register-to.",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="check backend availability without running. With "
             "--backend synthstrip: exit 0 if found, 3 otherwise. "
             "With --backend auto: prints which backend would run.",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress info prints to stderr")
    args = parser.parse_args(argv)

    log = (lambda _msg: None) if args.quiet else _stderr

    # ----- `--check` short-circuit -----
    if args.check:
        binary = find_synthstrip(args.synthstrip)
        if args.backend == "synthstrip":
            if binary is None:
                _stderr("mri_synthstrip not found.")
                return 3
            print(binary)
            return 0
        if args.backend == "log-watershed":
            print("log-watershed (always available)")
            return 0
        if args.backend == "deepbet":
            from rosa_detect.services.deepbet_strip import find_deepbet
            py = find_deepbet(args.deepbet_python)
            if py is None:
                _stderr("deepbet not found (no python can `import deepbet`).")
                return 3
            print(py)
            return 0
        # auto
        if binary is not None:
            print(f"auto -> synthstrip ({binary})")
        else:
            print("auto -> log-watershed (SynthStrip not found)")
        return 0

    # ----- run -----
    if args.input is None or args.mask is None:
        _stderr("error: both <input> and -o/--mask are required (unless --check)")
        return 2

    input_path = Path(args.input).expanduser()
    if not input_path.is_file():
        _stderr(f"error: input not found: {input_path}")
        return 2

    # Pick backend
    chosen = args.backend
    if chosen == "auto":
        chosen = "synthstrip" if find_synthstrip(args.synthstrip) is not None else "log-watershed"

    log(f"[brain-extract] backend={chosen}  {input_path}  ->  {args.mask}")

    if chosen == "deepbet":
        code, mask = _run_deepbet_backend(args, input_path, log)
        if code != 0:
            return code

    if chosen == "synthstrip":
        code, mask = _run_synthstrip_backend(args, input_path, log)
        # auto-mode fallback: if synthstrip is not found AND the user said
        # "auto", drop to the watershed backend.
        if code == 3 and args.backend == "auto":
            log("[brain-extract] SynthStrip not found, falling back to log-watershed")
            chosen = "log-watershed"
        elif code != 0:
            return code

    if chosen == "log-watershed":
        code, mask = _run_log_watershed_backend(args, input_path, log)
        if code != 0:
            return code

    log(f"[brain-extract] wrote {mask}")
    # Optional: register the input to a target and emit the transform / mask-in-frame.
    rc = _register_and_emit(args, input_path, Path(mask), log)
    if rc != 0:
        return rc
    print(mask)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
