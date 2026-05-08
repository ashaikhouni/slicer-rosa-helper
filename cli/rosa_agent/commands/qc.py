"""rosa-agent qc — per-emission diagnostic dump for the unified pipeline.

Runs ``detect_and_place_unified`` on a CT, then for every emitted
trajectory builds an ``EmissionQC`` (slab-ready data, walker profile,
per-library-model corr breakdown, placed-contact HU). Writes:

  * ``<output>/qc.json``          — list of per-emission QC dicts.
  * ``<output>/emission_NNN.png`` — 3-panel figure per emission.

Both use ``rosa_core.emission_qc`` so the same code paths feed
notebooks and the CLI.

Usage:

    rosa-agent qc --ct ct.nii.gz --output qc_dir/

Options:

    --strategy <name>    library subset (default ``pmt_35``)
    --no-figures         JSON only — skip PNG rendering (faster).
    --quiet              suppress progress prints.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rosa-agent qc",
        description="Per-emission QC dump for the unified pipeline.")
    parser.add_argument("--ct", required=True, help="path to CT NIfTI")
    parser.add_argument("--output", required=True,
        help="output directory (created if missing)")
    parser.add_argument("--strategy", default="pmt_35",
        help="electrode-library strategy filter (default pmt_35)")
    parser.add_argument("--no-figures", action="store_true",
        help="skip PNG rendering; only write qc.json")
    parser.add_argument("--quiet", action="store_true",
        help="suppress progress prints")
    args = parser.parse_args(argv)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = (lambda _msg: None) if args.quiet else _stderr

    log(f"[qc] loading {args.ct}")
    import numpy as np
    import SimpleITK as sitk

    from shank_core.io import image_ijk_ras_matrices
    from rosa_detect import guided_fit_engine as gfe
    from rosa_detect import contact_pitch_v1_fit as f1
    from rosa_detect.primitives.bolt_anchor import (
        extract_bolt_candidates, METAL_BOLT_THRESHOLD, BOLT_HULL_PROXIMITY_MM,
    )
    from rosa_core import (
        load_electrode_library, detect_and_place_unified,
        build_subject_qc, emission_qc_to_dict,
    )
    from rosa_core.electrode_classifier import filter_models_for_strategy

    img = sitk.ReadImage(str(args.ct))
    i2r_in, r2i_in = image_ijk_ras_matrices(img)
    features = gfe.compute_features(img, np.asarray(i2r_in), np.asarray(r2i_in))
    log_arr = features["log"]
    ct_arr = features["ct_arr_kji"]
    dist_arr = features["head_distance"]
    i2r = np.asarray(features["ijk_to_ras_mat"])
    r2i = np.asarray(features["ras_to_ijk_mat"])

    log("[qc] extracting bolt CCs (HU-mode metal_evidence)")
    metal_evidence = f1.compute_metal_evidence_volume(log_arr, ct_arr)
    bolts, _ = extract_bolt_candidates(
        log_arr, dist_arr, i2r, img.GetSpacing(),
        ras_to_ijk_mat=r2i,
        ct_arr=metal_evidence, hu_threshold=METAL_BOLT_THRESHOLD,
        hull_proximity_mm=BOLT_HULL_PROXIMITY_MM,
    )

    library = load_electrode_library()
    lib_models = filter_models_for_strategy(library["models"], args.strategy)
    log(f"[qc] strategy={args.strategy!r} → {len(lib_models)} library models")

    log("[qc] running detect_and_place_unified")
    emissions = detect_and_place_unified(features, lib_models)
    log(f"[qc] emitted {len(emissions)} trajectories")

    log("[qc] building per-emission QC (this re-runs the centerline + walker per emission)")
    qc_list = build_subject_qc(emissions, features=features, library_models=lib_models)

    qc_json = [emission_qc_to_dict(qc) for qc in qc_list]
    qc_json_path = out_dir / "qc.json"
    with open(qc_json_path, "w") as f:
        json.dump({
            "subject": Path(args.ct).stem,
            "strategy": args.strategy,
            "n_emissions": len(qc_list),
            "emissions": qc_json,
        }, f, indent=2)
    log(f"[qc] wrote {qc_json_path}")

    if args.no_figures:
        log("[qc] --no-figures: skipping PNG rendering")
    else:
        log(f"[qc] rendering {len(qc_list)} figures")
        # Lazy matplotlib import — only when actually rendering.
        import matplotlib
        matplotlib.use("Agg")  # no GUI
        import matplotlib.pyplot as plt
        from rosa_core import render_emission_figure

        for qc in qc_list:
            try:
                fig = render_emission_figure(qc, features=features, bolts=bolts)
                png = out_dir / f"emission_{qc.emission_idx:03d}.png"
                fig.savefig(png, dpi=120, bbox_inches="tight")
                plt.close(fig)
            except Exception as exc:
                log(f"[qc] emission {qc.emission_idx}: render failed ({exc})")
                continue
        log(f"[qc] wrote {len(qc_list)} figures to {out_dir}/")

    log(f"[qc] done. {len(qc_list)} emissions QC'd.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
