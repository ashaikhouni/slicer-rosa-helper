"""rosa-agent fit-rosa — load a ROSA folder, fit electrode models per
planned trajectory, write contact coordinates.

Pipeline:
    .ros + postop CT  → snap (LoG-neg) → arbitrate → bolt-outer walker
                      → per-trajectory landmarks (bone / ring / Frangi)
                      → peak features  → pitch-match picker
                      → place contacts from tip
                      → optional registration to reference image
                      → manifest.json + contacts.tsv + figures/

Two modes:

* **Forced** (``--electrodes electrode_models.tsv``): use the
  user-supplied electrode model per trajectory. TSV columns:
  ``name``, ``electrode_model``.
* **Auto** (no ``--electrodes``): matched-filter library picker
  (``rosa_core.model_pick.pick_electrode_model``).

Mask backends:

* ``--mask-backend auto`` (default): try SynthStrip first, fall through
  to LoG-watershed when SynthStrip is unavailable.
* ``--mask-backend synthstrip``: hard requirement (errors if not found).
* ``--mask-backend log-watershed``: CT-only pure-Python backend.

Output (``--output DIR``):

    DIR/
        manifest.json       — reference frame stamp, picker config, per-trajectory
                              prediction + branch + score components
        contacts.tsv        — header comment '# reference_frame: <name> sha256:<hex>'
                              then standard columns (trajectory, label, contact_index,
                              x, y, z, peak_detected, electrode_model)
        trajectories.tsv    — one row per fitted shank with picked model + landmarks
        qc.tsv              — plan-vs-placement deviation per shank (entry/target
                              3-D error, contact lateral deviation from the planned
                              line, axis angle); cohort summary in manifest.qc_summary
        figures/            — per-trajectory PNG (when matplotlib available)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------
# Top-level orchestrator.
# ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rosa-agent fit-rosa",
        description=(
            "Fit electrode models per planned trajectory (snap → matched_filter "
            "+ no_metal_rerank). Three input modes:\n"
            "  ROS:    rosa-agent fit-rosa <rosa_folder> ...\n"
            "  Seeds:  rosa-agent fit-rosa --seeds traj.tsv --postop-ct-path ct.nii.gz ...\n"
            "  Auto:   rosa-agent fit-rosa --auto --postop-ct-path ct.nii.gz ..."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "rosa_folder", nargs="?", default=None,
        help="ROSA case folder (contains .ros + DICOM/Analyze tree). "
             "Optional when --seeds or --auto is used.",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="output directory (created if missing)",
    )
    parser.add_argument(
        "--postop-ct", default=None,
        help="ROSA display volume name for the postop CT (ROS mode only; "
             "defaults to the first display whose name contains 'post' or "
             "'ct'; falls back to the first display when neither matches)",
    )
    parser.add_argument(
        "--postop-ct-path", default=None,
        help="raw NIfTI path for the postop CT (used with --seeds or "
             "--auto; mutually exclusive with rosa_folder positional)",
    )
    parser.add_argument(
        "--seeds", default=None,
        help="trajectory seeds TSV (start_x/y/z, end_x/y/z, optional name + "
             "electrode_model columns — same grammar as 'rosa-agent place "
             "--seeds'). Requires --postop-ct-path.",
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="run Stage-1 detector (run_contact_pitch_v1) inline on "
             "--postop-ct-path to produce seeds, then snap + matched_filter "
             "+ verifier. End-to-end CT-only mode.",
    )
    parser.add_argument(
        "--reference-image", default=None,
        help="optional NIfTI to register the postop CT to; final contact "
             "coords are output in this volume's RAS frame. Default: stay in "
             "the postop CT frame.",
    )
    parser.add_argument(
        "--electrodes", default=None,
        help="electrode_models.tsv (columns: name, electrode_model). When "
             "given, force these models per trajectory. Without this flag, "
             "auto mode runs the pitch-match decision-tree picker.",
    )
    parser.add_argument(
        "--mask-backend", default="auto",
        choices=("auto", "hull", "synthstrip", "log-watershed"),
        help="brain-mask backend; 'auto' = SynthStrip → LoG-watershed fallback; "
             "'hull' = fast head-distance approximation (no deps, coarse)",
    )
    parser.add_argument(
        "--synthstrip", default=None,
        help="explicit path to mri_synthstrip (defaults to PATH lookup)",
    )
    parser.add_argument(
        "--library", default="auto",
        help="electrode-library strategy key (e.g. 'auto', 'dixi', 'dixi_all'). "
             "Default 'auto' = all vendors. See "
             "rosa_core.electrode_classifier.PITCH_STRATEGY_OPTIONS.",
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="skip per-trajectory PNG rendering",
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="directory for cached brain mask + LoG/Frangi/metal_evidence "
             "volumes. Default: <rosa_folder>/.rosa_cache. Hits cut runtime "
             "from ~2 min to ~20 s on re-runs.",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="disable disk caching of derived volumes (always recompute).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="suppress progress prints",
    )
    args = parser.parse_args(argv)

    log = (lambda _m: None) if args.quiet else _stderr

    # ---------------------------------------------------------------
    # Input-mode resolution: exactly one of {rosa_folder, --seeds, --auto}.
    # ---------------------------------------------------------------
    mode_flags = sum([bool(args.rosa_folder), bool(args.seeds), bool(args.auto)])
    if mode_flags == 0:
        _stderr("error: must provide one of: rosa_folder, --seeds, --auto")
        return 2
    if mode_flags > 1:
        _stderr("error: rosa_folder / --seeds / --auto are mutually exclusive")
        return 2
    if (args.seeds or args.auto) and not args.postop_ct_path:
        _stderr("error: --seeds and --auto require --postop-ct-path")
        return 2
    mode = ("ros" if args.rosa_folder
            else "seeds" if args.seeds
            else "auto")

    out_dir = Path(args.output).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # ---------------------------------------------------------------
    # 1 + 2. Load CT + planned trajectories per input mode.
    # ROS:    parse .ros, load display volume, lift planned segments.
    # Seeds:  load NIfTI directly, read seeds TSV.
    # Auto:   load NIfTI directly, run Stage-1 detector inline.
    # ---------------------------------------------------------------
    if mode == "ros":
        sitk_img, planned, postop_name, rosa_dir = _load_ros_input(args, log)
    elif mode == "seeds":
        sitk_img, planned, postop_name, rosa_dir = _load_seeds_input(args, log)
    else:  # mode == "auto"
        sitk_img, planned, postop_name, rosa_dir = _load_auto_input(args, log)
    if sitk_img is None or planned is None:
        return 2
    log(f"[fit-rosa] planned trajectories: {len(planned)}")

    # ---------------------------------------------------------------
    # 3 + 4. Brain mask + canonical CT + LoG/Frangi feature volumes.
    # Cached on disk keyed by (CT array sha256, postop display name,
    # mask backend, library version). On cache miss, recomputes all of
    # mask + canonicalize + LoG/Frangi/metal_evidence and writes them
    # to the cache directory.
    # ---------------------------------------------------------------
    work_dir = out_dir / "work"
    work_dir.mkdir(exist_ok=True)
    ct_path = work_dir / "postop_ct.nii.gz"
    import SimpleITK as sitk
    sitk.WriteImage(sitk_img, str(ct_path))

    cache_dir = (
        None if args.no_cache
        else Path(args.cache_dir).expanduser() if args.cache_dir
        else (rosa_dir / ".rosa_cache") if rosa_dir is not None
        else (out_dir / ".rosa_cache")
    )

    feats = _load_or_compute_features(
        sitk_img=sitk_img, ct_path=ct_path, postop_name=postop_name,
        cache_dir=cache_dir,
        mask_backend=args.mask_backend,
        synthstrip_path=args.synthstrip,
        log=log,
    )
    if feats is None:
        _stderr("error: brain-mask backend failed")
        return 4
    img_canon            = feats["img_canon"]
    ct_arr               = feats["ct_arr"]
    log_arr              = feats["log_arr"]
    frangi_arr           = feats["frangi_arr"]
    metal_evidence       = feats["metal_evidence"]
    intracranial_mask_arr= feats["intracranial_mask"]
    ijk_to_ras_mat       = feats["ijk_to_ras_mat"]
    ras_to_ijk_mat       = feats["ras_to_ijk_mat"]
    mask_backend_used    = feats["mask_backend_used"]
    cache_hit            = feats["cache_hit"]

    log(f"[fit-rosa] brain mask: {mask_backend_used} (cache_hit={cache_hit})")

    import numpy as np
    # Clip at zero to match the notebook's cell-8 definition. Positive
    # LoG values are bright structures (bone, etc.) — keeping them would
    # let the snap walker's find_peaks fire on them and the chain to
    # over-extend, which knocks the bone landmark out of place
    # downstream (LSFG on S54: bone_arc=4.9 mm vs notebook 26 mm).
    log_neg = np.clip(-log_arr, 0.0, None).astype("float32", copy=False)

    # ---------------------------------------------------------------
    # 5. Snap pipeline.
    # ---------------------------------------------------------------
    from rosa_core.seeded_fit import (
        run_seeded_fit, estimate_landmarks,
    )

    LOG_NEG_THRESHOLD = 300.0  # notebook default — matches _build_starter.py cell-1

    log("[fit-rosa] running seeded fit…")
    chains = run_seeded_fit(
        planned,
        signal_vol=log_neg, threshold=LOG_NEG_THRESHOLD,
        ras_to_ijk=ras_to_ijk_mat,
        bolt_signal_vol=metal_evidence,
        log=log,
    )

    # ---------------------------------------------------------------
    # 6. Per-trajectory landmarks + picker + contact placement.
    # ---------------------------------------------------------------
    from rosa_core.centerline import (
        sample_trajectory_profile,
        collect_intracranial_hu_along_trajectory,
        otsu_threshold,
        unit,
    )
    from rosa_core.electrode_classifier import (
        trajectory_peak_features,
        filter_models_for_strategy,
        place_contacts_from_tip,
    )
    from rosa_core.model_pick import pick_electrode_model

    forced_models = _load_electrodes_tsv(args.electrodes) if args.electrodes else None
    if args.electrodes and forced_models is None:
        _stderr(f"error: failed to read --electrodes {args.electrodes}")
        return 2

    lib = json.loads(
        (Path(__file__).resolve().parents[3]
         / "CommonLib/rosa_core/resources/electrodes/electrode_models.json"
        ).read_text()
    )
    all_lib_models = lib["models"]
    candidate_models = filter_models_for_strategy(all_lib_models, args.library)
    candidate_ids = [m["id"] for m in candidate_models]
    models_dict = {m["id"]: m for m in all_lib_models}
    log(f"[fit-rosa] picker candidates: {len(candidate_ids)} models "
        f"(strategy={args.library!r})")

    per_trajectory: list[dict[str, Any]] = []
    for planned_i, chain in zip(planned, chains):
        name = planned_i["name"]
        if chain is None:
            per_trajectory.append({
                "name": name,
                "status": "snap_failed",
                "predicted_model": None,
                "branch": None,
            })
            log(f"[fit-rosa] {name}: snap failed; skipping")
            continue

        axis_unit = unit(np.asarray(chain["axis"], dtype=float))
        entry_ras = np.asarray(chain["entry_ras"], dtype=float)
        tip_ras   = np.asarray(chain["tip_ras"], dtype=float)

        landmarks = estimate_landmarks(
            chain, name=name,
            ct_arr=ct_arr, ras_to_ijk=ras_to_ijk_mat,
            intracranial_mask_arr=intracranial_mask_arr,
            frangi_arr=frangi_arr,
        )

        intra_hu = collect_intracranial_hu_along_trajectory(
            entry_ras, tip_ras, axis_unit,
            ct_vol=ct_arr, ras_to_ijk_mat=ras_to_ijk_mat,
            intracranial_mask_arr=intracranial_mask_arr,
        )
        metal_thr = otsu_threshold(intra_hu)
        arc_rel, prof = sample_trajectory_profile(
            entry_ras, tip_ras, axis_unit,
            ct_vol=ct_arr, ras_to_ijk_mat=ras_to_ijk_mat,
            metal_hu=metal_thr,
        )

        win_lo = landmarks.win_lo
        win_hi = landmarks.win_hi
        arc_length = max(0.0, win_hi - win_lo)

        features = trajectory_peak_features(
            chain, prof, arc_rel, win_lo, win_hi,
        )

        # ---- Picker: matched_filter + extent-aware + no_metal_rerank.
        # Replaced the old decision-tree picker (retired 2026-05-30).
        # Benchmarked on SEEG T1-T25 (308 traj): 84% all-exact / 81%
        # AM / 98% CM / 100% MM, vs decision-tree 9% all-exact /
        # 4% AM / 30% CM / 0% MM. Also AMC PMT 19/20 (was 16/20).
        mf_length = float(np.linalg.norm(tip_ras - entry_ras))
        mf_arcs, mf_sig = _sample_log_neg_tube(
            log_neg, ras_to_ijk_mat, entry_ras, axis_unit, mf_length,
        )
        # Matched-filter proximal anchor = bolt→contact transition arc. The CT
        # brain mask under-segments at the skull base, so its intracranial
        # entry can sit DEEPER than the most-superficial real contacts; those
        # contacts then fall in the matcher's unscored bolt zone and the
        # coverage tie-break picks a model one contact too short (the proximal
        # AM under-count). The bone inner-edge landmark is metal/bone-derived
        # and mask-independent, so it catches the transition where the mask
        # fails — take whichever is more proximal. Validated on T1-T25
        # (snap-seeded, DIXI): in-brain recall 97.0→97.4%, in-brain precision
        # held 98.1%, CM unchanged 59/60.
        mask_anchor = _intra_mask_anchor(
            intracranial_mask_arr, ras_to_ijk_mat,
            entry_ras, axis_unit, mf_length,
        )
        bone_anchor = landmarks.bone_arc_mm
        mf_anchor = (float(min(mask_anchor, bone_anchor))
                     if bone_anchor is not None else float(mask_anchor))

        # Forced model vs auto picker.
        forced = (forced_models or {}).get(name)
        if forced is not None and forced not in models_dict:
            _stderr(f"warning: {name} forced model {forced!r} not in library; "
                    f"falling back to auto picker")
            forced = None
        picker_diag: dict[str, Any] = {"votes": {}}
        if forced is not None:
            predicted_id = forced
            branch = "forced"
            picker_diag["matched_filter_corr"] = None
        else:
            # Shared model pick (rosa_core.model_pick) — the single source of
            # truth used by both the CLI and the Slicer-side placement, so the
            # matcher can't diverge between them. Symmetric ±~2 mm tip search
            # (max_extend_tip_mm / max_tip_short_mm) absorbs the snap-tip vs
            # physical-tip offset; the call runs matched_filter -> extent-aware
            # -> no_metal_rerank -> covering-floor.
            predicted_id, branch, mf_res, pick_diag = pick_electrode_model(
                mf_arcs, mf_sig, candidate_models, models_dict,
                bolt_end_arc=mf_anchor, profile_end_arc=mf_length, chain=chain,
            )
            picker_diag.update(pick_diag)

        # Place contacts at the picked model's offsets from the fitted tip.
        contacts_ras: list[list[float]] = []
        if predicted_id is not None:
            model = models_dict[predicted_id]
            contacts = place_contacts_from_tip(tip_ras, axis_unit, model)
            contacts_ras = [[float(c[0]), float(c[1]), float(c[2])] for c in contacts]

        per_trajectory.append({
            "name": name,
            "status": "ok" if predicted_id else "no_pick",
            "predicted_model": predicted_id,
            "branch": branch,
            "n_peaks": int(features.get("n_peaks") or 0),
            "n_clear": int(features.get("n_clear") or 0),
            "pitch_cv": float(features.get("pitch_cv") or float("inf")),
            "peak_span_mm": float(features.get("peak_span_mm") or 0.0),
            "snap_span_mm": float(features.get("snap_span_mm") or 0.0),
            "metal_otsu_hu": float(metal_thr),
            "arc_length_mm": float(arc_length),
            "bone_arc_mm": landmarks.bone_arc_mm,
            "bolt_combined_arc_mm": landmarks.bolt_combined_arc_mm,
            "entry_ras": [float(v) for v in entry_ras],
            "tip_ras":   [float(v) for v in tip_ras],
            "axis":      [float(v) for v in axis_unit],
            "mf_anchor_mm": float(mf_anchor),
            "votes":     picker_diag.get("votes") or {},
            "contacts_postop_ct_ras": contacts_ras,
            "snap_peaks_ras": [
                [float(p[0]), float(p[1]), float(p[2])]
                for p in np.asarray(chain["kept_pts"], dtype=float)
            ],
        })
        if predicted_id is None:
            log(f"[fit-rosa] {name}: no pick")
        else:
            log(f"[fit-rosa] {name}: {predicted_id} ({branch}, "
                f"{len(contacts_ras)} contacts)")

    # ---------------------------------------------------------------
    # 7. Optional registration to reference image.
    # ---------------------------------------------------------------
    reference_volume_name = postop_name
    reference_image_path = ct_path  # default: postop CT frame
    ref_hash = None
    if args.reference_image:
        ref_path = Path(args.reference_image).expanduser()
        if not ref_path.is_file():
            _stderr(f"error: --reference-image not found: {ref_path}")
            return 2
        log(f"[fit-rosa] registering postop CT → {ref_path.name}…")
        from rosa_core.registration import register_rigid_mi, apply_transform_to_points_ras
        ref_img = sitk.ReadImage(str(ref_path))
        reg = register_rigid_mi(
            fixed=ref_img, moving=sitk_img,
            metal_clip_hu=1500.0,
            logger=log,
        )
        # Push contacts from postop CT RAS → reference RAS.
        for row in per_trajectory:
            if row.get("contacts_postop_ct_ras"):
                pushed = apply_transform_to_points_ras(
                    row["contacts_postop_ct_ras"], reg.matrix_ras_4x4,
                )
                row["contacts_output_ras"] = [
                    [float(p[0]), float(p[1]), float(p[2])] for p in pushed
                ]
            else:
                row["contacts_output_ras"] = []
        reference_image_path = ref_path
        reference_volume_name = ref_path.name
        ref_hash = _sha256_of_image_array(ref_img)
    else:
        for row in per_trajectory:
            row["contacts_output_ras"] = list(row.get("contacts_postop_ct_ras") or [])
        ref_hash = _sha256_of_image_array(sitk_img)
        reference_volume_name = f"ROSA::{postop_name}"

    # ---------------------------------------------------------------
    # 7.5 Plan-vs-placement QC (computed in the postop-CT frame, where
    # both the imported plan seeds and the fitted result live, so the
    # comparison is well-posed — entry/target error + contact lateral
    # deviation from the planned line + axis angle).
    # ---------------------------------------------------------------
    from rosa_core.qc import (
        compute_plan_vs_placement_qc, summarize_plan_vs_placement_qc,
    )
    qc_rows = compute_plan_vs_placement_qc(
        planned_by_name={p["name"]: p for p in planned},
        fitted_by_name={
            r["name"]: {
                "start": r["entry_ras"], "end": r["tip_ras"],
                "contacts": r.get("contacts_postop_ct_ras") or [],
            }
            for r in per_trajectory if r.get("status") == "ok"
        },
    )
    qc_summary = summarize_plan_vs_placement_qc(qc_rows)

    # ---------------------------------------------------------------
    # 8. Write outputs.
    # ---------------------------------------------------------------
    runtime_sec = time.perf_counter() - t0
    manifest = {
        "command": "rosa-agent fit-rosa",
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_sec": round(runtime_sec, 2),
        "inputs": {
            "mode": mode,
            "rosa_folder": str(rosa_dir) if rosa_dir is not None else None,
            "postop_ct_display": postop_name,
            "postop_ct_path": (str(args.postop_ct_path)
                               if args.postop_ct_path else None),
            "seeds_tsv": str(args.seeds) if args.seeds else None,
            "reference_image": str(reference_image_path) if args.reference_image else None,
            "electrodes_tsv": str(args.electrodes) if args.electrodes else None,
            "mask_backend": args.mask_backend,
            "library_strategy": args.library,
        },
        "reference_frame": {
            "name": reference_volume_name,
            "sha256": ref_hash,
        },
        "mask_backend_used": mask_backend_used,
        "n_planned": len(planned),
        "n_fit": sum(1 for r in per_trajectory if r["status"] == "ok"),
        "qc_summary": qc_summary,
        "trajectories": per_trajectory,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=float))

    _write_contacts_tsv(out_dir / "contacts.tsv", per_trajectory,
                        reference_volume_name=reference_volume_name,
                        ref_hash=ref_hash)
    _write_trajectories_tsv(out_dir / "trajectories.tsv", per_trajectory)
    _write_qc_tsv(out_dir / "qc.tsv", qc_rows)
    _em = qc_summary["entry_error_mm"]["median"]
    _tm = qc_summary["target_error_mm"]["median"]
    _cm = qc_summary["mean_contact_radial_mm"]["median"]
    log("[fit-rosa] QC plan-vs-fit (median): "
        f"entry={'NA' if _em is None else f'{_em:.2f}'} "
        f"target={'NA' if _tm is None else f'{_tm:.2f}'} "
        f"contact-radial={'NA' if _cm is None else f'{_cm:.2f}'} mm → qc.tsv")

    if not args.no_figures:
        n_fig = _render_figures(
            out_dir / "figures", per_trajectory, planned,
            ct_arr=ct_arr, ras_to_ijk=ras_to_ijk_mat,
            postop_name=postop_name, log=log,
        )
        log(f"[fit-rosa] rendered {n_fig} figure(s) → {out_dir / 'figures'}")

    log(f"[fit-rosa] wrote {out_dir} in {runtime_sec:.1f}s")

    n_ok = manifest["n_fit"]
    n_total = manifest["n_planned"]
    print(
        f"fit-rosa: {n_ok}/{n_total} fit  "
        f"mask={mask_backend_used}  "
        f"runtime={runtime_sec:.1f}s  "
        f"output={out_dir}"
    )
    return 0


# ---------------------------------------------------------------------
# Cached mask + feature computation.
# ---------------------------------------------------------------------


_FEATURE_CACHE_VERSION = "4"  # bump when the cache schema or feature math changes
# v4 (2026-05-19): mask backend runs on the canonical CT (1 mm grid)
# instead of native — both cheaper and aligned to the features pipeline
# so the picker's Otsu HU samples come from the same grid as the
# bone-width disc samples.


def _load_or_compute_features(
    *, sitk_img, ct_path: Path, postop_name: str,
    cache_dir: Path | None,
    mask_backend: str, synthstrip_path: str | None,
    log,
) -> dict | None:
    """Load cached features for this CT, or compute + cache them.

    Cache key includes the CT array sha256, the postop display name, the
    mask backend, the LoG/Frangi sigmas, and a schema version. On hit,
    every required volume is loaded from disk in ~1-2 s. On miss the
    pipeline runs (synthstrip 30-60 s, LoG + Frangi 20-30 s) and writes
    the results so the next invocation is fast.

    Pass ``cache_dir=None`` to disable caching entirely (always
    recompute, never write). Returns ``None`` when the brain-mask backend
    fails; raises on other unrecoverable errors.

    Returned dict keys:
        img_canon, ct_arr, log_arr, frangi_arr, metal_evidence,
        intracranial_mask, ijk_to_ras_mat, ras_to_ijk_mat,
        mask_backend_used, cache_hit
    """
    import json
    import numpy as np
    import SimpleITK as sitk

    config = {
        "version":        _FEATURE_CACHE_VERSION,
        "postop_display": postop_name,
        "mask_backend":   mask_backend,
        "log_sigma_mm":   0.8,
        "frangi_sigma":   1.0,
    }
    cache_key = _feature_cache_key(sitk_img, config)
    cache_subdir = (cache_dir / cache_key) if cache_dir is not None else None

    if cache_subdir is not None and (cache_subdir / "meta.json").is_file():
        try:
            meta = json.loads((cache_subdir / "meta.json").read_text())
            if meta.get("cache_key") == cache_key:
                log(f"[fit-rosa] cache hit at {cache_subdir}")
                img_canon = sitk.ReadImage(str(cache_subdir / "ct_canon.nii.gz"))
                from shank_core.io import image_ijk_ras_matrices
                ijk_to_ras_mat, ras_to_ijk_mat = image_ijk_ras_matrices(img_canon)
                return dict(
                    img_canon=img_canon,
                    ct_arr=sitk.GetArrayFromImage(img_canon).astype("float32", copy=False),
                    log_arr=np.load(cache_subdir / "log_sigma1.npy"),
                    frangi_arr=np.load(cache_subdir / "frangi_sigma1.npy"),
                    metal_evidence=np.load(cache_subdir / "metal_evidence.npy"),
                    intracranial_mask=np.load(cache_subdir / "intracranial_mask.npy"),
                    ijk_to_ras_mat=np.asarray(ijk_to_ras_mat, dtype=float),
                    ras_to_ijk_mat=np.asarray(ras_to_ijk_mat, dtype=float),
                    mask_backend_used=str(meta.get("mask_backend_used") or mask_backend),
                    cache_hit=True,
                )
        except Exception as exc:  # noqa: BLE001
            log(f"[fit-rosa] cache load failed ({exc}); recomputing")

    # ---- Cache miss: full compute. ----
    log("[fit-rosa] computing canonical CT + features + brain mask…")

    # Canonicalize first, then run the mask backend on the 1 mm canonical
    # grid. Two wins: (1) matches `load_features_and_bolts` /
    # `guided_fit_engine.compute_features` exactly — the same CT goes
    # into the mask backend and the LoG/Frangi pipeline, so the mask's
    # Otsu-derived HU samples line up with the bone-width disc samples
    # the picker scores against. (2) cheaper — synthstrip + log-watershed
    # both scale with voxel count, so a 1 mm grid is ~6× faster than a
    # 0.42 mm native grid.
    from rosa_detect.primitives.preprocessing import (
        prepare_volume, log_sigma, frangi_single,
    )
    from shank_core.io import image_ijk_ras_matrices
    ijk_to_ras_mat_native, ras_to_ijk_mat_native = image_ijk_ras_matrices(sitk_img)
    img_canon, ijk_to_ras_mat, ras_to_ijk_mat = prepare_volume(
        sitk_img, ijk_to_ras_mat_native, ras_to_ijk_mat=ras_to_ijk_mat_native,
    )
    canon_ct_path = ct_path.with_name("postop_ct_canon.nii.gz")
    sitk.WriteImage(img_canon, str(canon_ct_path))

    mask_path = ct_path.with_name("brain_mask.nii.gz")
    mask_backend_used = _compute_brain_mask(
        canon_ct_path, mask_path, backend=mask_backend,
        synthstrip_path=synthstrip_path, log=log,
    )
    if mask_backend_used is None:
        return None
    mask_img = sitk.ReadImage(str(mask_path))

    ct_arr = sitk.GetArrayFromImage(img_canon).astype("float32", copy=False)
    log_arr = log_sigma(img_canon, sigma_mm=float(config["log_sigma_mm"]))
    frangi_arr = frangi_single(img_canon, sigma=float(config["frangi_sigma"]))
    ct_clip = np.clip(ct_arr, -1024.0, 1500.0)
    metal_evidence = np.maximum(
        np.abs(log_arr) / 800.0,
        np.maximum(ct_clip, 0.0) / 2000.0,
    ).astype("float32", copy=False)

    # Mask was already computed on the canonical grid; should match
    # img_canon exactly. The resample branch stays as defense-in-depth
    # in case a future backend produces a different grid.
    if mask_img.GetSize() != img_canon.GetSize():
        resampled_mask = sitk.Resample(
            mask_img, img_canon, sitk.Transform(),
            sitk.sitkNearestNeighbor, 0.0,
        )
        intracranial_mask = sitk.GetArrayFromImage(resampled_mask).astype(bool)
    else:
        intracranial_mask = sitk.GetArrayFromImage(mask_img).astype(bool)

    if cache_subdir is not None:
        try:
            cache_subdir.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(img_canon, str(cache_subdir / "ct_canon.nii.gz"))
            np.save(cache_subdir / "log_sigma1.npy",       log_arr.astype("float32", copy=False))
            np.save(cache_subdir / "frangi_sigma1.npy",    frangi_arr.astype("float32", copy=False))
            np.save(cache_subdir / "metal_evidence.npy",   metal_evidence.astype("float32", copy=False))
            np.save(cache_subdir / "intracranial_mask.npy", intracranial_mask)
            meta = {
                "cache_key":          cache_key,
                "config":             config,
                "mask_backend_used":  mask_backend_used,
                "created_at":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            (cache_subdir / "meta.json").write_text(json.dumps(meta, indent=2))
            log(f"[fit-rosa] wrote cache to {cache_subdir}")
        except Exception as exc:  # noqa: BLE001
            log(f"[fit-rosa] cache write failed ({exc}); continuing")

    return dict(
        img_canon=img_canon,
        ct_arr=ct_arr,
        log_arr=log_arr,
        frangi_arr=frangi_arr,
        metal_evidence=metal_evidence,
        intracranial_mask=intracranial_mask,
        ijk_to_ras_mat=np.asarray(ijk_to_ras_mat, dtype=float),
        ras_to_ijk_mat=np.asarray(ras_to_ijk_mat, dtype=float),
        mask_backend_used=mask_backend_used,
        cache_hit=False,
    )


def _feature_cache_key(sitk_img, config: dict) -> str:
    """SHA256 of (CT voxel data, config dict) — stable hex key for the
    cache subdirectory."""
    import SimpleITK as sitk
    arr = sitk.GetArrayFromImage(sitk_img)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    h.update(json.dumps(config, sort_keys=True).encode())
    return h.hexdigest()[:24]


# ---------------------------------------------------------------------
# Brain-mask backend dispatcher (mirrors brain-extract command).
# ---------------------------------------------------------------------


def _compute_brain_mask(
    ct_path: Path, mask_path: Path,
    *, backend: str = "auto",
    synthstrip_path: str | None = None,
    log=lambda _m: None,
) -> str | None:
    """Thin delegate to the shared brain-mask selector
    (``rosa_detect.services.mask_backend.select_brain_mask_to_path``) — the CLI
    and the Slicer/place_seeg path share ONE mask backend. Returns the backend
    name used (``"synthstrip"`` / ``"log-watershed"``) or ``None`` on failure."""
    from rosa_detect.services.mask_backend import select_brain_mask_to_path
    return select_brain_mask_to_path(
        ct_path, mask_path, backend=backend,
        synthstrip_path=synthstrip_path, log=log,
    )


# ---------------------------------------------------------------------
# IO helpers.
# ---------------------------------------------------------------------


def _load_electrodes_tsv(path: str) -> dict[str, str] | None:
    """Read ``electrode_models.tsv`` (columns: name, electrode_model)."""
    p = Path(path).expanduser()
    if not p.is_file():
        return None
    out: dict[str, str] = {}
    with open(p, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            name = (row.get("name") or row.get("trajectory") or "").strip()
            model = (row.get("electrode_model") or row.get("model_id") or "").strip()
            if name and model:
                out[name] = model
    return out


def _sha256_of_image_array(sitk_img) -> str:
    """Hash the SITK image's voxel data after load (insensitive to file
    format / compression)."""
    import SimpleITK as sitk
    arr = sitk.GetArrayFromImage(sitk_img)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def _write_contacts_tsv(
    path: Path, per_trajectory: list[dict[str, Any]],
    *, reference_volume_name: str, ref_hash: str,
) -> None:
    """Write contacts TSV with a leading header comment that stamps the
    reference frame.

    Comment line:
        # reference_frame: <name> sha256:<hex>

    Then the standard contacts.tsv columns. The shared reader
    (``trajectory_io.read_tsv_rows``) skips leading ``#`` comment lines, so
    ``rosa-agent label`` / ``export-view`` / ``place`` read this file correctly.
    Other tools that don't tolerate comments can ``grep -v '^#'`` before reading.
    """
    cols = ["trajectory", "label", "contact_index", "x", "y", "z",
            "peak_detected", "electrode_model"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"# reference_frame: {reference_volume_name} sha256:{ref_hash}\n")
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        writer.writeheader()
        for row in per_trajectory:
            name = row["name"]
            model = row.get("predicted_model") or ""
            for idx, c in enumerate(row.get("contacts_output_ras") or [], start=1):
                writer.writerow({
                    "trajectory": name,
                    "label": f"{name}{idx}",
                    "contact_index": str(idx),
                    "x": f"{float(c[0]):.6f}",
                    "y": f"{float(c[1]):.6f}",
                    "z": f"{float(c[2]):.6f}",
                    "peak_detected": "1",
                    "electrode_model": model,
                })


def _write_trajectories_tsv(path: Path, per_trajectory: list[dict[str, Any]]) -> None:
    """Per-trajectory pick summary."""
    cols = [
        "name", "status", "predicted_model", "branch",
        "n_peaks", "n_clear", "pitch_cv", "peak_span_mm", "snap_span_mm",
        "metal_otsu_hu", "arc_length_mm",
        "bone_arc_mm", "bolt_combined_arc_mm",
        "entry_x", "entry_y", "entry_z",
        "tip_x",   "tip_y",   "tip_z",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        writer.writeheader()
        for row in per_trajectory:
            entry = row.get("entry_ras") or [0.0, 0.0, 0.0]
            tip   = row.get("tip_ras")   or [0.0, 0.0, 0.0]
            writer.writerow({
                "name": row["name"],
                "status": row.get("status") or "",
                "predicted_model": row.get("predicted_model") or "",
                "branch": row.get("branch") or "",
                "n_peaks": row.get("n_peaks") or "",
                "n_clear": row.get("n_clear") or "",
                "pitch_cv": f"{row.get('pitch_cv', 0.0):.4f}" if row.get("pitch_cv") is not None else "",
                "peak_span_mm": f"{row.get('peak_span_mm', 0.0):.3f}",
                "snap_span_mm": f"{row.get('snap_span_mm', 0.0):.3f}",
                "metal_otsu_hu": f"{row.get('metal_otsu_hu', 0.0):.1f}",
                "arc_length_mm": f"{row.get('arc_length_mm', 0.0):.3f}",
                "bone_arc_mm": "" if row.get("bone_arc_mm") is None else f"{row['bone_arc_mm']:.3f}",
                "bolt_combined_arc_mm": "" if row.get("bolt_combined_arc_mm") is None else f"{row['bolt_combined_arc_mm']:.3f}",
                "entry_x": f"{float(entry[0]):.4f}",
                "entry_y": f"{float(entry[1]):.4f}",
                "entry_z": f"{float(entry[2]):.4f}",
                "tip_x":   f"{float(tip[0]):.4f}",
                "tip_y":   f"{float(tip[1]):.4f}",
                "tip_z":   f"{float(tip[2]):.4f}",
            })


def _write_qc_tsv(path: Path, qc_rows: list[dict[str, Any]]) -> None:
    """Plan-vs-placement QC: one row per planned shank (entry/target 3-D error,
    contact lateral deviation from the planned line, axis angle). Empty cells for
    shanks that didn't fit."""
    cols = [
        "trajectory", "entry_error_mm", "target_error_mm",
        "mean_contact_radial_mm", "max_contact_radial_mm", "rms_contact_radial_mm",
        "angle_deg", "n_contacts",
    ]

    def _fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        writer.writeheader()
        for row in qc_rows:
            writer.writerow({c: _fmt(row.get(c)) for c in cols})


# ---------------------------------------------------------------------
# QC figures — per-trajectory slab views.
# ---------------------------------------------------------------------

_SLAB_PAD_ENTRY_MM = 20.0
_SLAB_PAD_TIP_MM = 20.0
_SLAB_HALF_WIDTH_MM = 8.0
_SLAB_STEP_MM = 0.5


def _slab_along_axis(ct_arr, ras_to_ijk, entry, tip, axis, perp):
    """2-D slab sampled on the (axis, perp) plane through ``entry``.

    Returns ``(slab, arcs)`` — ``slab`` is perp-rows × along-cols, ``arcs``
    is the along-axis coordinate (mm, relative to ``entry``).
    """
    import numpy as np
    from rosa_core.volume_sampling import sample_trilinear_batch

    L = float(np.linalg.norm(tip - entry))
    arcs = np.linspace(-_SLAB_PAD_ENTRY_MM, L + _SLAB_PAD_TIP_MM,
                       int((L + _SLAB_PAD_ENTRY_MM + _SLAB_PAD_TIP_MM)
                           / _SLAB_STEP_MM) + 1)
    perps = np.linspace(-_SLAB_HALF_WIDTH_MM, _SLAB_HALF_WIDTH_MM,
                        int(2 * _SLAB_HALF_WIDTH_MM / _SLAB_STEP_MM) + 1)
    pts = (entry[None, None, :]
           + arcs[:, None, None] * axis[None, None, :]
           + perps[None, :, None] * perp[None, None, :])
    vals = sample_trilinear_batch(
        ct_arr.astype("float32", copy=False), ras_to_ijk,
        pts.reshape(-1, 3),
    ).reshape(len(arcs), len(perps))
    return vals.T, arcs


def _render_figures(out_dir: Path, per_trajectory: list[dict[str, Any]],
                    planned: list[dict[str, Any]], *,
                    ct_arr, ras_to_ijk, postop_name: str, log) -> int:
    """One QC PNG per trajectory: two slab views (axis × perp1/perp2) with
    snap peaks + placed contacts overlaid. ``snap_failed`` rows render the
    planned-axis slab so the user can confirm there is no electrode there.

    Returns the number of figures written; 0 (with a warning) if matplotlib
    is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("[fit-rosa] matplotlib not available — skipping figures")
        return 0

    import numpy as np
    from rosa_core.centerline import unit, orthonormal_basis_for_axis

    out_dir.mkdir(parents=True, exist_ok=True)
    planned_by_name = {p["name"]: p for p in planned}
    count = 0
    for idx, row in enumerate(per_trajectory, start=1):
        name = row["name"]
        status = row.get("status") or ""
        try:
            if status == "snap_failed":
                pl = planned_by_name.get(name)
                if pl is None:
                    continue
                entry = np.asarray(pl["start"], dtype=float)
                tip = np.asarray(pl["end"], dtype=float)
                axis = unit(tip - entry)
                peaks = np.empty((0, 3))
                contacts = np.empty((0, 3))
            else:
                entry = np.asarray(row["entry_ras"], dtype=float)
                tip = np.asarray(row["tip_ras"], dtype=float)
                axis = unit(np.asarray(row["axis"], dtype=float))
                peaks = np.asarray(row.get("snap_peaks_ras") or [],
                                   dtype=float).reshape(-1, 3)
                contacts = np.asarray(row.get("contacts_postop_ct_ras") or [],
                                      dtype=float).reshape(-1, 3)

            p1, p2 = orthonormal_basis_for_axis(axis)
            L = float(np.linalg.norm(tip - entry))

            fig, axes = plt.subplots(2, 1, figsize=(12, 5.0))
            for ax, perp, plabel in ((axes[0], p1, "perp1"),
                                     (axes[1], p2, "perp2")):
                slab, arcs = _slab_along_axis(ct_arr, ras_to_ijk,
                                              entry, tip, axis, perp)
                ax.imshow(slab, cmap="gray", origin="lower", aspect="equal",
                          extent=[arcs[0], arcs[-1],
                                  -_SLAB_HALF_WIDTH_MM, _SLAB_HALF_WIDTH_MM],
                          vmin=-200, vmax=1500)

                def _proj(pp):
                    rel = pp - entry
                    return rel @ axis, rel @ perp

                if len(peaks):
                    a, p = _proj(peaks)
                    ax.scatter(a, p, s=64, facecolors="none",
                               edgecolors="red", linewidths=1.4,
                               label=f"snap peaks (n={len(peaks)})")
                if len(contacts):
                    a, p = _proj(contacts)
                    ax.scatter(a, p, s=26, c="lime", marker="o",
                               edgecolors="black", linewidths=0.5,
                               label=f"contacts (n={len(contacts)})")
                    # First + last contact get an explicit halo so the
                    # contact-array start/end is easy to compare against
                    # snap/seed endpoints.
                    ax.scatter([a[0]], [p[0]], s=140, marker="o",
                               facecolors="none", edgecolors="lime",
                               linewidths=1.6,
                               label="1st contact" if ax is axes[0] else None)
                    ax.scatter([a[-1]], [p[-1]], s=140, marker="o",
                               facecolors="none", edgecolors="seagreen",
                               linewidths=1.6,
                               label="last contact" if ax is axes[0] else None)
                ax.scatter([0.0], [0.0], s=80, marker="s", facecolors="none",
                           edgecolors="gold", linewidths=2.0, label="snap entry")
                ax.scatter([L], [0.0], s=80, marker="s", facecolors="none",
                           edgecolors="deepskyblue", linewidths=2.0,
                           label="snap tip")
                # Planned (seed) endpoints projected onto the snap axis.
                # Useful when snap drifted laterally / shortened — the
                # planned start/end show where the user/seed expected the
                # array to live.
                pl = planned_by_name.get(name)
                if pl is not None and status != "snap_failed":
                    pln_s = np.asarray(pl["start"], dtype=float)
                    pln_e = np.asarray(pl["end"],   dtype=float)
                    a_ps, p_ps = _proj(pln_s[None])[0][0], _proj(pln_s[None])[1][0]
                    a_pe, p_pe = _proj(pln_e[None])[0][0], _proj(pln_e[None])[1][0]
                    ax.scatter([a_ps], [p_ps], s=90, marker="D",
                               c="gold", edgecolors="black", linewidths=0.6,
                               label="planned start" if ax is axes[0] else None)
                    ax.scatter([a_pe], [p_pe], s=90, marker="D",
                               c="deepskyblue", edgecolors="black",
                               linewidths=0.6,
                               label="planned end" if ax is axes[0] else None)
                # mf_anchor (matched_filter proximal floor reference).
                mf = row.get("mf_anchor_mm")
                if mf is not None:
                    ax.axvline(float(mf), color="cyan", ls="--", lw=1.0,
                               alpha=0.8,
                               label="mf_anchor" if ax is axes[0] else None)
                ax.set_ylabel(f"{plabel} (mm)")
                ax.set_xlim(arcs[0], arcs[-1])
            axes[1].set_xlabel("along axis (mm)")
            axes[0].legend(loc="upper right", fontsize=6, ncol=5)
            title = (f"{name}  [{status}]"
                     f"  {row.get('predicted_model') or ''}"
                     f"  {row.get('branch') or ''}".rstrip())
            axes[0].set_title(title, fontsize=10)
            fig.suptitle(f"{postop_name}", fontsize=8, y=0.99)
            fig.tight_layout()
            fig.savefig(out_dir / f"{idx:03d}_{_safe_fig_name(name)}.png",
                        dpi=110, bbox_inches="tight")
            plt.close(fig)
            count += 1
        except Exception as exc:  # noqa: BLE001
            (out_dir / f"{idx:03d}_{_safe_fig_name(name)}.error.txt").write_text(
                f"figure render failed: {type(exc).__name__}: {exc}\n"
            )
    return count


def _safe_fig_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


# --------------------------------------------------------------------
# Helpers for the matched-filter picker (sample LoG-neg along the snap
# axis; locate the proximal in-brain anchor).
# --------------------------------------------------------------------
_MF_TUBE_RADIUS_MM = 4.0
_MF_STEP_MM        = 0.3
_MF_PAD_MM         = 2.0


def _sample_log_neg_tube(log_neg, ras_to_ijk, entry_ras, axis_unit, length_mm):
    """Tube-max LoG-neg profile along (entry_ras, axis_unit) from
    -pad to length+pad, in 0.3 mm steps. Same shape as the snap-side
    profile, so the no_metal_rerank threshold (frac * max sig)
    behaves the same way matched_filter_pick was validated against
    in notebooks/seeded_fit/_score_seeded.py.
    """
    import numpy as np
    from rosa_core.centerline import orthonormal_basis_for_axis
    from rosa_core.volume_sampling import sample_trilinear_batch

    perp1, perp2 = orthonormal_basis_for_axis(axis_unit)
    n = int(2 * _MF_TUBE_RADIUS_MM / 0.5) + 1
    ua = np.linspace(-_MF_TUBE_RADIUS_MM, _MF_TUBE_RADIUS_MM, n)
    UU, VV = np.meshgrid(ua, ua, indexing="ij")
    disk = (UU ** 2 + VV ** 2) <= _MF_TUBE_RADIUS_MM ** 2
    du, dv = UU[disk], VV[disk]

    arcs = np.arange(-_MF_PAD_MM, length_mm + _MF_PAD_MM, _MF_STEP_MM)
    sig = np.zeros(len(arcs), dtype=float)
    log_neg_f32 = log_neg.astype("float32", copy=False)
    for i, t in enumerate(arcs):
        c = entry_ras + t * axis_unit
        pts = c[None] + du[:, None] * perp1[None] + dv[:, None] * perp2[None]
        v = sample_trilinear_batch(log_neg_f32, ras_to_ijk, pts)
        finite = v[np.isfinite(v)] if v.size else v
        sig[i] = float(np.max(finite)) if finite.size else 0.0
    return arcs, sig


def _intra_mask_anchor(intracranial_mask, ras_to_ijk, entry_ras, axis_unit,
                       length_mm):
    """Arc (mm) along (entry_ras, axis_unit) where the trajectory
    first crosses into the intracranial mask. Returns 0.0 if already
    in-brain at entry (common — snap's entry_ras is the most proximal
    detected contact, which often sits in cortex)."""
    import numpy as np
    from rosa_core.volume_sampling import sample_trilinear_batch

    mask = intracranial_mask.astype("float32", copy=False)
    ts = np.arange(0.0, length_mm + 1.0, 0.5)
    inb = np.zeros(len(ts), dtype=float)
    for i, t in enumerate(ts):
        v = sample_trilinear_batch(mask, ras_to_ijk, (entry_ras + t * axis_unit)[None])
        inb[i] = float(v[0]) if v.size and np.isfinite(v[0]) else 0.0
    inside = np.where(inb > 0.5)[0]
    return float(ts[inside[0]]) if inside.size else 0.0


# --------------------------------------------------------------------
# Input-mode loaders. Each returns (sitk_img, planned, postop_name,
# rosa_dir-or-None). planned[i] = {"name": str, "start": [x,y,z],
# "end": [x,y,z]} in the RAS frame matching sitk_img. Loaders return
# (None, None, _, _) on error after logging a message.
# --------------------------------------------------------------------
def _load_ros_input(args, log):
    """ROS mode: rosa_folder positional → parse .ros, lift planned
    segments, load chosen display volume as SimpleITK image."""
    from rosa_core.case_loader import load_rosa_volume_as_sitk, find_ros_file
    from rosa_core.ros_parser import parse_ros_file
    from rosa_core.transforms import lps_to_ras_point

    rosa_dir = Path(args.rosa_folder).expanduser()
    if not rosa_dir.is_dir():
        _stderr(f"error: ROSA folder not found: {rosa_dir}")
        return None, None, None, None
    ros_file = find_ros_file(str(rosa_dir))
    parsed = parse_ros_file(ros_file)
    displays = parsed.get("displays") or []
    if not displays:
        _stderr(f"error: no display volumes in {ros_file}")
        return None, None, None, None
    display_names = [d["volume"] for d in displays]
    log(f"[fit-rosa] ROSA file: {ros_file}")
    log(f"[fit-rosa] available displays: {display_names}")

    postop_name = args.postop_ct
    if postop_name is None:
        for d in displays:
            lc = d["volume"].lower()
            if "post" in lc or lc.startswith("ct") or lc == "post":
                postop_name = d["volume"]; break
        if postop_name is None:
            postop_name = displays[0]["volume"]
    if postop_name not in display_names:
        _stderr(f"error: --postop-ct {postop_name!r} not in displays "
                f"{display_names}")
        return None, None, None, None
    log(f"[fit-rosa] postop CT display: {postop_name!r}")

    sitk_img, _ = load_rosa_volume_as_sitk(
        str(rosa_dir), volume_name=postop_name,
    )

    planned: list[dict[str, Any]] = []
    for t in parsed.get("trajectories") or []:
        try:
            s_ras = [float(v) for v in lps_to_ras_point(list(t["start"]))]
            e_ras = [float(v) for v in lps_to_ras_point(list(t["end"]))]
        except (KeyError, TypeError, ValueError):
            continue
        planned.append({"name": str(t.get("name") or ""),
                         "start": s_ras, "end": e_ras})
    if not planned:
        _stderr(f"error: no planned trajectories in {ros_file}")
        return None, None, None, None
    return sitk_img, planned, postop_name, rosa_dir


def _load_postop_sitk_nifti(path_str: str):
    """Shared helper: load a raw NIfTI for the non-ROS modes."""
    import SimpleITK as sitk
    p = Path(path_str).expanduser()
    if not p.is_file():
        _stderr(f"error: --postop-ct-path not found: {p}")
        return None, None
    return sitk.ReadImage(str(p)), p


def _load_seeds_input(args, log):
    """Seeds mode: --seeds traj.tsv + --postop-ct-path ct.nii.gz."""
    from rosa_agent.io.trajectory_io import read_seeds_tsv

    sitk_img, ct_path = _load_postop_sitk_nifti(args.postop_ct_path)
    if sitk_img is None:
        return None, None, None, None
    seeds_path = Path(args.seeds).expanduser()
    if not seeds_path.is_file():
        _stderr(f"error: --seeds not found: {seeds_path}")
        return None, None, None, None
    try:
        seeds = read_seeds_tsv(seeds_path)
    except (ValueError, OSError) as exc:
        _stderr(f"error: failed to read --seeds {seeds_path}: {exc}")
        return None, None, None, None
    planned: list[dict[str, Any]] = []
    for i, s in enumerate(seeds):
        try:
            start = [float(v) for v in s["start_ras"]]
            end   = [float(v) for v in s["end_ras"]]
        except (KeyError, TypeError, ValueError):
            continue
        planned.append({
            "name": str(s.get("name") or f"S{i+1:02d}"),
            "start": start, "end": end,
        })
    if not planned:
        _stderr(f"error: --seeds {seeds_path} produced no valid trajectories")
        return None, None, None, None
    log(f"[fit-rosa] CT: {ct_path}  (mode: seeds, n={len(planned)})")
    return sitk_img, planned, ct_path.stem, None


def _load_auto_input(args, log):
    """Auto mode: --auto + --postop-ct-path. Run Stage-1 detector
    inline to produce seeds, then run the same downstream pipeline.
    Mirrors the loop in notebooks/seeded_fit/_amc_auto.py."""
    from rosa_detect.service import run_contact_pitch_v1
    from rosa_detect.contracts import VolumeRef

    sitk_img, ct_path = _load_postop_sitk_nifti(args.postop_ct_path)
    if sitk_img is None:
        return None, None, None, None
    log(f"[fit-rosa] CT: {ct_path}  (mode: auto — running Stage-1 detector)")
    ctx = {
        "run_id": f"fit-rosa-auto-{ct_path.stem}",
        "ct": VolumeRef(volume_id=ct_path.stem, path=str(ct_path)),
        "logger": (lambda msg: log(f"  [stage1] {msg}")),
    }
    try:
        result = run_contact_pitch_v1(ctx)
    except Exception as exc:  # noqa: BLE001
        _stderr(f"error: Stage-1 detector failed: {type(exc).__name__}: {exc}")
        return None, None, None, None
    trajs = result.get("trajectories") or []
    planned: list[dict[str, Any]] = []
    for i, t in enumerate(trajs):
        try:
            start = [float(v) for v in t["start_ras"]]
            end   = [float(v) for v in t["end_ras"]]
        except (KeyError, TypeError, ValueError):
            continue
        planned.append({
            "name": str(t.get("name") or f"D{i+1:02d}"),
            "start": start, "end": end,
        })
    if not planned:
        _stderr("error: Stage-1 detector emitted no trajectories")
        return None, None, None, None
    log(f"[fit-rosa] Stage-1 emitted {len(planned)} trajectories")
    return sitk_img, planned, ct_path.stem, None


if __name__ == "__main__":
    raise SystemExit(main())
