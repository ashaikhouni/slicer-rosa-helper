"""rosa-agent match-ros — name detector emissions from a .ros file's plan.

Use case: same patient, two coordinate frames. Given a ROSA case folder
(or a bare .ros file) and a CT in some *other* RAS frame (e.g. a
post-registered CT aligned to a different MRI atlas, or a routine
clinical CT in scanner coordinates), match each detector-emitted
trajectory to a ROS-planned trajectory and propagate the surgeon's
naming. No reference volume from the .ros side is needed.

The match is line-geometry only — see
:mod:`rosa_core.cross_volume_match` for the why. The detector runs first
on the CT (mode 1, ``place_seeg``), then the matcher recovers the rigid
transform from the line bundles and pairs each ROS trajectory with the
detector emission whose infinite line is closest under that transform.

Usage::

    rosa-agent match-ros \\
        --rosa-folder s57_rosa/ \\
        --ct path/to/post_registered_t24.nii.gz \\
        --output match_qc/

    # or pass a bare .ros file directly
    rosa-agent match-ros --ros-file plan.ros --ct ct.nii.gz --output qc/

Output directory layout (same shape as ``rosa-agent place``):

    output/
      manifest.json              ← input provenance + recovered transform + match table
      trajectories.tsv           ← detector emissions, RENAMED to ROS plan names where matched
      contacts.tsv
      figures/                   ← per-trajectory PNGs (filenames use the ROS plan names)
      diagnostics/cmp.tsv
      match.tsv                  ← per-ROS-line match (ros_name, det_name, angle°, perp_mm)

Unmatched detector emissions keep their original ``CAND-NNN`` names.
Unmatched ROS plans appear in ``match.tsv`` with empty det_name.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rosa-agent match-ros",
        description=(
            "Name detector emissions on a CT using planned trajectories "
            "from a .ros file in a different coordinate frame."
        ),
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--rosa-folder",
                     help="ROSA case folder (auto-locates the .ros file inside)")
    src.add_argument("--ros-file",
                     help="path to a .ros file directly (skip folder search)")
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
    # Matcher knobs — defaults validated on s57.ros + T24 CT (16/17 ROS
    # named, all matched pairs ≤ 11° axis-angle and ≤ 6 mm perp).
    parser.add_argument("--angle-tol-deg", type=float, default=15.0,
                        help="axis-angle tolerance for RANSAC + greedy match (default 15)")
    parser.add_argument("--ransac-perp-mm", type=float, default=8.0,
                        help="perp line-to-line tolerance for RANSAC inliers (default 8)")
    parser.add_argument("--match-perp-mm", type=float, default=12.0,
                        help="perp tolerance for greedy ROS↔det assignment (default 12)")
    parser.add_argument("--ransac-iter", type=int, default=2000,
                        help="RANSAC iteration budget (default 2000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RANSAC RNG seed (default 42)")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress progress prints")
    args = parser.parse_args(argv)

    log = (lambda _msg: None) if args.quiet else _stderr

    ct_path = Path(args.ct)
    if not ct_path.exists():
        _stderr(f"error: CT not found: {ct_path}")
        return 2

    # ------------------------------------------------------------------
    # Locate + parse the .ros file (no image read). Shares
    # `cli.rosa_agent.io.ros_input` with `rosa-to-nifti` so any future
    # ros-aware CLI command can drop in via the same helper.
    # ------------------------------------------------------------------
    try:
        from rosa_agent.io.ros_input import load_ros_planned_trajectories
    except ImportError as exc:
        _stderr(f"error: rosa_core / rosa_agent unavailable ({exc})")
        return 2
    try:
        plan = load_ros_planned_trajectories(
            folder=args.rosa_folder, file=args.ros_file,
        )
    except (FileNotFoundError, ValueError) as exc:
        _stderr(f"error: {exc}")
        return 2
    ros_path = plan.ros_file
    ros_trajs = plan.trajectories
    if len(ros_trajs) < 3:
        _stderr(
            f"error: .ros at {ros_path} declared only {len(ros_trajs)} planned "
            f"trajectories — RANSAC needs at least 3 to estimate a rigid transform"
        )
        return 2
    log(f"[match-ros] {len(ros_trajs)} planned trajectories from {ros_path.name}")

    # ------------------------------------------------------------------
    # Detect on the CT (mode 1).
    # ------------------------------------------------------------------
    library_key = args.library if args.library else None
    from rosa_core.contact_placement import sample_hu_max, sample_neg_log_max
    sample_fn = sample_neg_log_max if args.sampler == "log" else sample_hu_max
    from rosa_core.placement_modes import place_seeg

    t0 = time.perf_counter()
    log(f"[match-ros] running place_seeg on {ct_path} (library={library_key or 'full'})")
    try:
        batch = place_seeg(
            str(ct_path),
            library=library_key,
            sample_fn=sample_fn,
            band_floor=args.band_floor,
            progress_logger=log,
        )
    except ValueError as exc:
        _stderr(f"error: place_seeg rejected the inputs: {exc}")
        return 2
    runtime_sec = time.perf_counter() - t0
    log(f"[match-ros] mode {batch.diagnostics['mode']}: emitted "
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
        ros_trajs, det_dicts,
        angle_tol_deg=args.angle_tol_deg,
        ransac_perp_tol_mm=args.ransac_perp_mm,
        match_perp_tol_mm=args.match_perp_mm,
        n_iter=args.ransac_iter,
        seed=args.seed,
    )

    n_named = sum(1 for _, det, _, _ in result.pairs if det is not None)
    log(
        f"[match-ros] RANSAC inliers={result.ransac_inliers}, "
        f"refined inliers={result.refined_inliers}, "
        f"named {n_named}/{len(result.pairs)} ROS plans"
    )

    # Build det_name → ros_name lookup, then rename PlacedTrajectory in place.
    det_to_ros: dict[str, str] = {
        det: ros for ros, det, _, _ in result.pairs if det
    }
    n_renamed = 0
    for traj in batch.trajectories:
        new_name = det_to_ros.get(traj.name)
        if new_name and new_name != traj.name:
            traj.diagnostics["original_detector_name"] = traj.name
            traj.name = new_name
            n_renamed += 1
    log(f"[match-ros] renamed {n_renamed} detector emissions to ROS plan names")

    # ------------------------------------------------------------------
    # Write QC directory + match.tsv + augmented manifest.
    # ------------------------------------------------------------------
    from rosa_core.qc_output import write_qc_directory
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    features = batch.features
    bolts = batch.bolts
    if not args.no_figures and features is None:
        log("[match-ros] features not on batch; loading fresh for figure rendering")
        try:
            from rosa_core.volume_loader import load_features_and_bolts
            features, bolts = load_features_and_bolts(str(ct_path))
        except Exception as exc:  # noqa: BLE001
            _stderr(f"warning: failed to load features for figures ({exc}); "
                    f"writing TSVs only")
            features = bolts = None

    write_qc_directory(
        batch, out_dir,
        ct_path=ct_path,
        subject_id=args.subject_id,
        library_id=args.library or "full",
        mode_args={
            "ros_file": str(ros_path),
            "library": args.library,
            "sampler": args.sampler,
            "band_floor": args.band_floor,
            "matcher": {
                "angle_tol_deg": args.angle_tol_deg,
                "ransac_perp_mm": args.ransac_perp_mm,
                "match_perp_mm": args.match_perp_mm,
                "ransac_iter": args.ransac_iter,
                "seed": args.seed,
            },
        },
        runtime_seconds=runtime_sec,
        write_figures=not args.no_figures,
        features=features,
        bolts=bolts,
    )

    match_tsv = out_dir / "match.tsv"
    _write_match_tsv(match_tsv, result.pairs)
    log(f"[match-ros] wrote {match_tsv.name} ({len(result.pairs)} rows)")

    # Drop the recovered transform + match summary into a sidecar JSON
    # the caller can pick up programmatically.
    sidecar = out_dir / "cross_volume_match.json"
    sidecar.write_text(json.dumps({
        "ros_file": str(ros_path),
        "ct": str(ct_path),
        "n_ros_trajectories": len(ros_trajs),
        "n_detector_emissions": len(batch.trajectories),
        "n_named": int(n_named),
        "ransac_inliers": int(result.ransac_inliers),
        "refined_inliers": int(result.refined_inliers),
        "transform_det_to_ros_4x4": result.transform_4x4.tolist(),
        "matcher_diagnostics": result.diagnostics,
        "pairs": [
            {
                "ros_name": ros, "det_name": det,
                "angle_deg": ang, "perp_mm": perp,
            }
            for ros, det, ang, perp in result.pairs
        ],
    }, indent=2) + "\n")
    log(f"[match-ros] wrote {sidecar.name}")

    print(
        f"ros={ros_path.name} "
        f"ct={ct_path.name} "
        f"n_planned={len(ros_trajs)} "
        f"n_detected={len(batch.trajectories)} "
        f"n_named={n_named} "
        f"refined_inliers={result.refined_inliers} "
        f"runtime={runtime_sec:.1f}s "
        f"output={out_dir}"
    )
    return 0


def _write_match_tsv(path: Path, pairs) -> None:
    """One row per ROS-planned trajectory.

    ``det_name`` is empty for ROS plans the matcher couldn't pair;
    ``angle_deg`` / ``perp_mm`` are empty in that case too.
    """
    cols = ["ros_name", "det_name", "angle_deg", "perp_mm"]
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for ros, det, ang, perp in pairs:
            f.write("\t".join([
                ros or "",
                det or "",
                f"{ang:.4f}" if ang is not None else "",
                f"{perp:.4f}" if perp is not None else "",
            ]) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
