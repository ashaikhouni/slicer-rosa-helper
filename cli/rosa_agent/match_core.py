"""Shared core for the trajectory-naming commands.

Both ``match-trajectories`` (a generic named-trajectory file) and ``match-ros``
(a ``.ros`` plan) reduce to the *same* operation:

    detect electrodes on a CT  →  name each detection by matching it to a named
    PLAN trajectory bundle that may live in a *different* RAS frame, using line
    geometry alone (RANSAC over the line bundle — no image registration, no
    reference volume).

This module is that operation. The two CLI commands are thin adapters that only
differ in how they build ``plan_trajs`` (read a TSV vs. parse a ``.ros``); both
then call :func:`run_trajectory_match`. See
:mod:`rosa_core.cross_volume_match` for why line geometry alone suffices.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def add_match_args(parser) -> None:
    """Add the CT / output / detector / matcher args shared by both commands.

    The plan-source args differ per command and are added by each ``main``.
    """
    parser.add_argument("--ct", required=True,
                        help="path to CT NIfTI / NRRD (any RAS frame)")
    parser.add_argument("--output", required=True,
                        help="output directory (created if missing)")
    parser.add_argument("--library", default="dixi",
                        help="electrode-library strategy key (default 'dixi'). "
                             "Pass an empty string for the full library.")
    parser.add_argument("--subject-id", default=None,
                        help="subject identifier stamped into manifest.json")
    parser.add_argument("--no-figures", action="store_true",
                        help="skip per-trajectory PNG rendering")
    parser.add_argument("--sampler", choices=("log", "hu"), default="log",
                        help="walker signal source (default 'log')")
    parser.add_argument("--band-floor", default=None,
                        help="filter detector emissions by min band "
                             "('high', 'medium', 'low'); default 'medium'")
    # Matcher knobs — defaults validated on s57.ros + T24 CT (16/17 plans
    # named, all matched pairs <= 11 deg axis-angle and <= 6 mm perp).
    parser.add_argument("--angle-tol-deg", type=float, default=15.0,
                        help="axis-angle tolerance for RANSAC + greedy match (default 15)")
    parser.add_argument("--ransac-perp-mm", type=float, default=8.0,
                        help="perp line-to-line tolerance for RANSAC inliers (default 8)")
    parser.add_argument("--match-perp-mm", type=float, default=12.0,
                        help="perp tolerance for greedy plan<->det assignment (default 12)")
    parser.add_argument("--ransac-iter", type=int, default=2000,
                        help="RANSAC iteration budget (default 2000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RANSAC RNG seed (default 42)")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress progress prints")


def _write_match_tsv(path: Path, pairs) -> None:
    """One row per planned trajectory.

    ``det_name`` is empty for plans the matcher couldn't pair; ``angle_deg`` /
    ``perp_mm`` are empty in that case too.
    """
    cols = ["plan_name", "det_name", "angle_deg", "perp_mm"]
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for plan, det, ang, perp in pairs:
            f.write("\t".join([
                plan or "",
                det or "",
                f"{ang:.4f}" if ang is not None else "",
                f"{perp:.4f}" if perp is not None else "",
            ]) + "\n")


def run_trajectory_match(
    plan_trajs: list[dict[str, Any]],
    ct_path: Path,
    output: Path,
    *,
    plan_label: str,
    library: str | None = "dixi",
    sampler: str = "log",
    band_floor: str | None = None,
    subject_id: str | None = None,
    no_figures: bool = False,
    angle_tol_deg: float = 15.0,
    ransac_perp_mm: float = 8.0,
    match_perp_mm: float = 12.0,
    ransac_iter: int = 2000,
    seed: int = 42,
    log: Callable[[str], None] = _stderr,
) -> int:
    """Detect on ``ct_path``, name detections from ``plan_trajs``, write a QC dir.

    ``plan_trajs`` is a list of ``{name, start, end}`` (``start_ras``/``end_ras``
    also accepted) RAS dicts. The frame need NOT match the CT — the matcher
    recovers the rigid transform from the line bundle. Returns a process exit
    code (0 ok, 2 on bad input).

    Writes ``output/`` with the same shape as ``rosa-agent place`` plus
    ``match.tsv`` (per-plan match) and ``cross_volume_match.json`` (recovered
    transform + pairs). Detector emissions matched to a plan are RENAMED to the
    plan name; unmatched ones keep their ``CAND-NNN`` names.
    """
    if len(plan_trajs) < 3:
        _stderr(
            f"error: plan {plan_label} declared only {len(plan_trajs)} named "
            f"trajectories — RANSAC needs at least 3 to estimate a rigid transform"
        )
        return 2
    log(f"[match] {len(plan_trajs)} planned trajectories from {plan_label}")

    # ------------------------------------------------------------------
    # Detect on the CT (mode 1).
    # ------------------------------------------------------------------
    library_key = library if library else None
    from rosa_core.contact_placement import sample_hu_max, sample_neg_log_max
    sample_fn = sample_neg_log_max if sampler == "log" else sample_hu_max
    from rosa_core.placement_modes import place_seeg

    t0 = time.perf_counter()
    log(f"[match] running place_seeg on {ct_path} (library={library_key or 'full'})")
    try:
        batch = place_seeg(
            str(ct_path),
            library=library_key,
            sample_fn=sample_fn,
            band_floor=band_floor,
            progress_logger=log,
        )
    except ValueError as exc:
        _stderr(f"error: place_seeg rejected the inputs: {exc}")
        return 2
    runtime_sec = time.perf_counter() - t0
    log(f"[match] mode {batch.diagnostics['mode']}: emitted "
        f"{len(batch.trajectories)} trajectories in {runtime_sec:.1f}s")

    # ------------------------------------------------------------------
    # Cross-volume line match.
    # ------------------------------------------------------------------
    from rosa_core.cross_volume_match import cross_volume_match

    det_dicts = [
        {"name": t.name, "start": t.start_ras, "end": t.end_ras}
        for t in batch.trajectories
    ]
    result = cross_volume_match(
        plan_trajs, det_dicts,
        angle_tol_deg=angle_tol_deg,
        ransac_perp_tol_mm=ransac_perp_mm,
        match_perp_tol_mm=match_perp_mm,
        n_iter=ransac_iter,
        seed=seed,
    )

    n_named = sum(1 for _, det, _, _ in result.pairs if det is not None)
    log(
        f"[match] RANSAC inliers={result.ransac_inliers}, "
        f"refined inliers={result.refined_inliers}, "
        f"named {n_named}/{len(result.pairs)} plans"
    )

    # Build det_name -> plan_name lookup, then rename PlacedTrajectory in place.
    det_to_plan: dict[str, str] = {
        det: plan for plan, det, _, _ in result.pairs if det
    }
    n_renamed = 0
    for traj in batch.trajectories:
        new_name = det_to_plan.get(traj.name)
        if new_name and new_name != traj.name:
            traj.diagnostics["original_detector_name"] = traj.name
            traj.name = new_name
            n_renamed += 1
    log(f"[match] renamed {n_renamed} detector emissions to plan names")

    # ------------------------------------------------------------------
    # Write QC directory + match.tsv + augmented manifest.
    # ------------------------------------------------------------------
    from rosa_core.qc_output import write_qc_directory
    output.mkdir(parents=True, exist_ok=True)

    features = batch.features
    bolts = batch.bolts
    if not no_figures and features is None:
        log("[match] features not on batch; loading fresh for figure rendering")
        try:
            from rosa_core.volume_loader import load_features_and_bolts
            features, bolts = load_features_and_bolts(str(ct_path))
        except Exception as exc:  # noqa: BLE001
            _stderr(f"warning: failed to load features for figures ({exc}); "
                    f"writing TSVs only")
            features = bolts = None

    write_qc_directory(
        batch, output,
        ct_path=ct_path,
        subject_id=subject_id,
        library_id=library or "full",
        mode_args={
            "plan_source": plan_label,
            "library": library,
            "sampler": sampler,
            "band_floor": band_floor,
            "matcher": {
                "angle_tol_deg": angle_tol_deg,
                "ransac_perp_mm": ransac_perp_mm,
                "match_perp_mm": match_perp_mm,
                "ransac_iter": ransac_iter,
                "seed": seed,
            },
        },
        runtime_seconds=runtime_sec,
        write_figures=not no_figures,
        features=features,
        bolts=bolts,
    )

    match_tsv = output / "match.tsv"
    _write_match_tsv(match_tsv, result.pairs)
    log(f"[match] wrote {match_tsv.name} ({len(result.pairs)} rows)")

    # Drop the recovered transform + match summary into a sidecar JSON the
    # caller can pick up programmatically.
    sidecar = output / "cross_volume_match.json"
    sidecar.write_text(json.dumps({
        "plan_source": plan_label,
        "ct": str(ct_path),
        "n_plan_trajectories": len(plan_trajs),
        "n_detector_emissions": len(batch.trajectories),
        "n_named": int(n_named),
        "ransac_inliers": int(result.ransac_inliers),
        "refined_inliers": int(result.refined_inliers),
        "transform_det_to_plan_4x4": result.transform_4x4.tolist(),
        "matcher_diagnostics": result.diagnostics,
        "pairs": [
            {
                "plan_name": plan, "det_name": det,
                "angle_deg": ang, "perp_mm": perp,
            }
            for plan, det, ang, perp in result.pairs
        ],
    }, indent=2) + "\n")
    log(f"[match] wrote {sidecar.name}")

    print(
        f"plan={plan_label} "
        f"ct={ct_path.name} "
        f"n_planned={len(plan_trajs)} "
        f"n_detected={len(batch.trajectories)} "
        f"n_named={n_named} "
        f"refined_inliers={result.refined_inliers} "
        f"runtime={runtime_sec:.1f}s "
        f"output={output}"
    )
    return 0
