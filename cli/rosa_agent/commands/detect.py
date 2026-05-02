"""rosa-agent detect — run contact_pitch_v1 detection on a CT volume.

Two modes:

* Auto-fit (default): pure ``rosa_detect.service.run_contact_pitch_v1``.
* Guided-fit (``--seeds <tsv>``): one-time feature compute via
  ``rosa_detect.guided_fit_engine.compute_features``, then per-seed
  ``fit_trajectory``.

Outputs a TSV with the stable trajectory columns (see
``rosa_agent.io.trajectory_io.TRAJECTORY_COLUMNS``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from rosa_detect.contracts import VolumeRef
from rosa_detect.service import run_contact_pitch_v1

from ..io.trajectory_io import (
    read_seeds_tsv,
    write_trajectories_tsv,
)


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def _build_ctx(ct_path: str | Path, *, run_id: str) -> dict[str, Any]:
    p = Path(ct_path).expanduser().resolve()
    return {
        "run_id": run_id,
        "ct": VolumeRef(volume_id=p.stem, path=str(p)),
        "config": {},
        "extras": {},
        "logger": _stderr,
    }


def run_auto_detect(ct_path: str | Path, *, run_id: str = "rosa_agent_detect") -> dict[str, Any]:
    """Auto Fit on a CT volume. Returns the full DetectionResult."""
    ctx = _build_ctx(ct_path, run_id=run_id)
    result = run_contact_pitch_v1(ctx)
    return result


def run_guided_detect(
    ct_path: str | Path,
    seeds: list[dict[str, Any]],
    *,
    run_id: str = "rosa_agent_guided",
    roi_radius_mm: float = 5.0,
    max_angle_deg: float = 12.0,
    max_lateral_shift_mm: float = 6.0,
) -> list[dict[str, Any]]:
    """Per-seed guided fit using a single shared feature volume.

    Returns a list of trajectory dicts shaped like ``DetectedTrajectory``
    so the same writer can serialize them.
    """
    import SimpleITK as sitk
    from shank_core.io import image_ijk_ras_matrices
    from rosa_detect import guided_fit_engine as gfe

    p = Path(ct_path).expanduser().resolve()
    img = sitk.ReadImage(str(p))
    ijk_to_ras, ras_to_ijk = image_ijk_ras_matrices(img)
    _stderr(f"[detect] computing guided-fit features on {p.name}…")
    features = gfe.compute_features(img, ijk_to_ras, ras_to_ijk)
    img = features["img"]
    ijk_to_ras = features["ijk_to_ras_mat"]
    ras_to_ijk = features["ras_to_ijk_mat"]

    out: list[dict[str, Any]] = []
    for seed in seeds:
        name = str(seed.get("name") or f"seed_{len(out)+1}")
        fit = gfe.fit_trajectory(
            seed["start_ras"], seed["end_ras"], features,
            ijk_to_ras, ras_to_ijk,
            roi_radius_mm=roi_radius_mm,
            max_angle_deg=max_angle_deg,
            max_lateral_shift_mm=max_lateral_shift_mm,
        )
        if not fit.get("success"):
            _stderr(f"[detect] guided fit failed for {name}: {fit.get('reason')}")
            # Keep the seed in the output so downstream stages still see
            # the planned trajectory; mark it low/zero confidence.
            out.append({
                "name": name,
                "start_ras": list(seed["start_ras"]),
                "end_ras": list(seed["end_ras"]),
                "confidence": 0.0,
                "confidence_label": "low",
                "electrode_model": seed.get("electrode_model") or "",
                "bolt": {"source": "none"},
                "guided_fit_failed": True,
                "guided_fit_reason": fit.get("reason"),
            })
            continue
        traj: dict[str, Any] = {
            "name": name,
            "start_ras": list(fit["start_ras"]),
            "end_ras": list(fit["end_ras"]),
            "confidence": float(fit.get("confidence", 0.0)),
            "confidence_label": str(fit.get("confidence_label") or ""),
            "electrode_model": seed.get("electrode_model") or "",
            "bolt": {"source": str(fit.get("bolt_source") or "none")},
        }
        if fit.get("skull_entry_ras"):
            traj["skull_entry_ras"] = list(fit["skull_entry_ras"])
        if fit.get("bolt_tip_ras"):
            traj["bolt_tip_ras"] = list(fit["bolt_tip_ras"])
        out.append(traj)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rosa-agent detect",
        description="Run SEEG shank detection on a postop CT volume.",
    )
    parser.add_argument("ct_path", help="Path to the CT NIfTI/NRRD")
    parser.add_argument("--seeds", default="", help="Optional seed TSV → guided fit")
    parser.add_argument("--out", "-o", required=True, help="Output trajectory TSV")
    parser.add_argument("--roi-radius-mm", type=float, default=5.0)
    parser.add_argument("--max-angle-deg", type=float, default=12.0)
    parser.add_argument("--max-lateral-shift-mm", type=float, default=6.0)
    args = parser.parse_args(argv)

    if args.seeds:
        seeds = read_seeds_tsv(args.seeds)
        _stderr(f"[detect] {len(seeds)} seed(s) loaded from {args.seeds}")
        trajs = run_guided_detect(
            args.ct_path, seeds,
            roi_radius_mm=args.roi_radius_mm,
            max_angle_deg=args.max_angle_deg,
            max_lateral_shift_mm=args.max_lateral_shift_mm,
        )
    else:
        result = run_auto_detect(args.ct_path)
        if str(result.get("status")) == "error":
            err = dict(result.get("error") or {})
            print(
                f"[detect] pipeline error: {err.get('message')} (stage={err.get('stage')})",
                file=sys.stderr,
            )
            return 1
        trajs = list(result.get("trajectories") or [])
        _stderr(f"[detect] auto-fit produced {len(trajs)} trajectories")

    n = write_trajectories_tsv(args.out, trajs)
    _stderr(f"[detect] wrote {args.out} ({n} trajectories)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
