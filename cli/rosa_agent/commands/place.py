"""rosa-agent place — single CLI for the 5-mode staged placement pipeline.

Wraps ``rosa_core.placement_modes.place_seeg`` and writes a standardized
QC directory via ``rosa_core.qc_output.write_qc_directory``.

For ROSA-folder input (.ros + Analyze volumes), use ``rosa-agent fit-rosa``
instead — it resolves the .ros file, loads the chosen display volume,
and imports planned trajectories as seeds before delegating to the
same staged-placement core this command uses.

Usage::

    rosa-agent place --ct ct.nii.gz --output qc/                  # mode 1 (auto)
    rosa-agent place --ct ct.nii.gz --output qc/ --n-expected 8   # mode 2 (count)
    rosa-agent place --ct ct.nii.gz --output qc/ --expected E.tsv # mode 3 (named)
    rosa-agent place --ct ct.nii.gz --output qc/ --seeds S.tsv    # mode 4 (with model_id) or 5 (without)

Mode is implied by which optional flags are passed (mirrors ``place_seeg``):

  None of seeds/expected/n_expected → mode 1
  --n-expected N                    → mode 2
  --expected file.tsv               → mode 3
  --seeds file.tsv (with model_id)  → mode 4
  --seeds file.tsv (no model_id)    → mode 5

Seed TSV format: same as the existing ``--seeds`` for ``rosa-agent detect``
— accepted columns include ``name`` + ``start_x/y/z`` + ``end_x/y/z``,
optionally ``electrode_model``. (See ``cli/rosa_agent/io/trajectory_io.py``
for the full grammar.)

Expected TSV format: simple two-column ``name<TAB>model_id``.

Output directory layout (from ``rosa_core.qc_output``):

    output/
      manifest.json
      trajectories.tsv
      contacts.tsv
      figures/                  (PNG per trajectory; needs matplotlib)
      diagnostics/cmp.tsv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rosa-agent place",
        description="Place SEEG contacts via the 5-mode staged pipeline.",
    )
    parser.add_argument("--ct", required=True,
                        help="path to CT NIfTI / NRRD")
    parser.add_argument("--output", required=True,
                        help="output directory (created if missing)")
    parser.add_argument("--seeds", default=None,
                        help="seeds TSV (modes 4 + 5)")
    parser.add_argument("--expected", default=None,
                        help="expected TSV (mode 3): name<TAB>model_id")
    parser.add_argument("--n-expected", type=int, default=None,
                        help="expected shank count (mode 2)")
    parser.add_argument("--library", default=None,
                        help="electrode-library strategy key "
                             "(e.g. 'pmt_35', 'dixi'). Default: full library.")
    parser.add_argument("--subject-id", default=None,
                        help="subject identifier stamped into manifest.json")
    parser.add_argument("--no-figures", action="store_true",
                        help="skip per-trajectory PNG rendering "
                             "(matplotlib not required)")
    parser.add_argument("--sampler", choices=("log", "hu"), default="log",
                        help="walker signal source (default 'log' — LoG-side "
                             "dominates per the 2026-05-09 sweep)")
    parser.add_argument("--band-floor", default=None,
                        help="filter mode-1 emissions by min band "
                             "('high', 'medium', 'low'); default 'medium'")
    parser.add_argument("--snap-angle-deg", type=float, default=12.0,
                        help="mode 4/5 snap-to-v1 angle tolerance (default 12)")
    parser.add_argument("--snap-perp-mm", type=float, default=8.0,
                        help="mode 4/5 snap-to-v1 perp tolerance (default 8)")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress progress prints")
    args = parser.parse_args(argv)

    log = (lambda _msg: None) if args.quiet else _stderr

    # --------------------------------------------------------------
    # Mode-arg validation: at most one of seeds/expected/n_expected.
    # --------------------------------------------------------------
    n_modeargs = sum(
        1 for x in (args.seeds, args.expected, args.n_expected) if x is not None
    )
    if n_modeargs > 1:
        _stderr("error: pass at most one of --seeds / --expected / --n-expected")
        return 2

    ct_path = Path(args.ct)
    if not ct_path.exists():
        _stderr(f"error: CT not found: {ct_path}")
        return 2

    # --------------------------------------------------------------
    # Read seeds / expected when supplied.
    # --------------------------------------------------------------
    seeds = None
    expected = None
    seed_source_path = None
    expected_source_path = None
    if args.seeds:
        seed_source_path = Path(args.seeds)
        if not seed_source_path.exists():
            _stderr(f"error: seeds TSV not found: {seed_source_path}")
            return 2
        seeds = _load_seeds(seed_source_path)
        log(f"[place] loaded {len(seeds)} seeds from {seed_source_path}")
    if args.expected:
        expected_source_path = Path(args.expected)
        if not expected_source_path.exists():
            _stderr(f"error: expected TSV not found: {expected_source_path}")
            return 2
        expected = _load_expected(expected_source_path)
        log(f"[place] loaded {len(expected)} expected entries from {expected_source_path}")

    # --------------------------------------------------------------
    # Sampler swap (lazy import — keep CLI startup fast).
    # --------------------------------------------------------------
    from rosa_core.contact_placement import sample_hu_max, sample_neg_log_max
    sample_fn = sample_neg_log_max if args.sampler == "log" else sample_hu_max

    # --------------------------------------------------------------
    # Run the pipeline.
    # --------------------------------------------------------------
    from rosa_core.placement_modes import place_seeg

    t0 = time.perf_counter()
    log(f"[place] running place_seeg on {ct_path}")
    try:
        batch = place_seeg(
            str(ct_path),
            seeds=seeds,
            expected=expected,
            n_expected=args.n_expected,
            library=args.library,
            sample_fn=sample_fn,
            band_floor=args.band_floor,
            snap_angle_tol_deg=args.snap_angle_deg,
            snap_perp_tol_mm=args.snap_perp_mm,
            progress_logger=log,
        )
    except ValueError as exc:
        # place_seeg raises ValueError on input-validation failures:
        # empty seeds=[] / expected=[] (often an upstream filter dropped
        # everything), n_expected <= 0, mode-arg combinations that
        # contradict each other, mode-4 vouched model not in the active
        # library, etc. Surface a clean one-liner instead of letting
        # the traceback escape — same exit code (2) the argparse layer
        # uses for its own usage errors.
        _stderr(f"error: place_seeg rejected the inputs: {exc}")
        return 2
    runtime_sec = time.perf_counter() - t0
    log(f"[place] mode {batch.diagnostics['mode']}: emitted "
        f"{len(batch.trajectories)} trajectories in {runtime_sec:.1f}s")

    # --------------------------------------------------------------
    # Write the QC directory.
    # --------------------------------------------------------------
    from rosa_core.qc_output import write_qc_directory

    # Reuse the (features, bolts) the placer already computed. Skips
    # the ~5-10 s second LoG/Frangi/hull pass that the prior
    # "re-load for figure rendering" path incurred. Falls back to a
    # fresh load only if the batch didn't carry them (older cached
    # PlacementBatch from a stale install).
    features = batch.features
    bolts = batch.bolts
    if not args.no_figures and features is None:
        log("[place] features not on batch; loading fresh for figure rendering")
        try:
            from rosa_core.volume_loader import load_features_and_bolts
            features, bolts = load_features_and_bolts(str(ct_path))
        except Exception as exc:  # noqa: BLE001
            _stderr(f"warning: failed to load features for figures ({exc}); "
                    f"writing TSVs only")
            features = bolts = None

    out_dir = Path(args.output)
    write_qc_directory(
        batch, out_dir,
        ct_path=ct_path,
        subject_id=args.subject_id,
        library_id=args.library or "full",
        mode_args={
            "seeds_path": str(seed_source_path) if seed_source_path else None,
            "expected_path": str(expected_source_path) if expected_source_path else None,
            "n_expected": args.n_expected,
            "library": args.library,
            "sampler": args.sampler,
            "band_floor": args.band_floor,
            "snap_angle_tol_deg": args.snap_angle_deg,
            "snap_perp_tol_mm": args.snap_perp_mm,
        },
        runtime_seconds=runtime_sec,
        write_figures=not args.no_figures,
        features=features,
        bolts=bolts,
    )
    log(f"[place] wrote {out_dir}")

    # Brief summary on stdout (machine-readable).
    bands = [t.band for t in batch.trajectories]
    print(
        f"mode={batch.diagnostics['mode']} "
        f"n={len(batch.trajectories)} "
        f"high={bands.count('high')} "
        f"medium={bands.count('medium')} "
        f"low={bands.count('low')} "
        f"runtime={runtime_sec:.1f}s "
        f"output={out_dir}"
    )
    return 0


# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------


def _load_seeds(path: Path) -> list:
    """Read a seeds TSV and return ``list[Seed]``.

    Reuses the existing ``trajectory_io.read_seeds_tsv`` parser
    (multi-format aware: trajectory-row, ex/tx, label+xyz pair).
    Sets ``Seed.model_id`` from the row's ``electrode_model`` column when
    present (so the dispatcher routes mode 4 vs 5 automatically).
    """
    from rosa_agent.io.trajectory_io import read_seeds_tsv
    from rosa_core.placement_modes import Seed

    rows = read_seeds_tsv(path)
    out = []
    for r in rows:
        model = r.get("electrode_model") or None
        out.append(Seed(
            name=str(r["name"]),
            start_ras=r["start_ras"],
            end_ras=r["end_ras"],
            model_id=str(model) if model else None,
        ))
    return out


def _load_expected(path: Path) -> list[tuple[str, str]]:
    """Read an expected TSV: ``name<TAB>model_id`` per row.

    Lenient: any 2+ column TSV/CSV with columns named ``name`` (or
    ``trajectory``, ``label``) and ``model_id`` (or ``electrode_model``)
    is accepted.
    """
    from rosa_agent.io.trajectory_io import read_tsv_rows
    rows = read_tsv_rows(path)
    out = []
    for r in rows:
        rl = {k.lower(): v for k, v in r.items()}
        name = rl.get("name") or rl.get("trajectory") or rl.get("label")
        model = rl.get("model_id") or rl.get("electrode_model")
        if not name or not model:
            raise ValueError(
                f"expected TSV row missing name/model_id: {r!r} "
                f"(needs columns 'name' + 'model_id')"
            )
        out.append((str(name), str(model)))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
