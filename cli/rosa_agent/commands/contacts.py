"""rosa-agent contacts — place contacts on each trajectory using LoG-driven peaks.

Reads the trajectory TSV emitted by ``rosa-agent detect``, computes
LoG σ=1 once for the CT, then runs ``contact_peak_fit.detect_contacts_on_axis``
per trajectory. Output: contacts TSV with stable columns.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ..io.trajectory_io import (
    read_seeds_tsv,
    write_contacts_tsv,
)


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def _compute_log_volume(ct_path: str | Path):
    """Return (log_kji_float32, ras_to_ijk_4x4)."""
    import SimpleITK as sitk
    from rosa_core.contact_peak_fit import compute_log_sigma1_volume
    from shank_core.io import image_ijk_ras_matrices

    img = sitk.ReadImage(str(ct_path))
    log_arr = compute_log_sigma1_volume(img)
    _, ras_to_ijk = image_ijk_ras_matrices(img)
    return log_arr, ras_to_ijk


def place_contacts(
    ct_path: str | Path,
    trajectories: list[dict[str, Any]],
    *,
    pitch_strategy: str | None = None,
) -> list[dict[str, Any]]:
    """Run peak-driven contact placement for each trajectory.

    Returns a list of contact-group dicts in the format expected by
    ``write_contacts_tsv``:

        {
            "trajectory": <name>,
            "electrode_model": <id>,
            "positions_ras": [...],
            "peak_detected": [...],
        }
    """
    from rosa_core.contact_peak_fit import detect_contacts_on_axis
    from rosa_core.electrode_models import load_electrode_library, model_map

    library = load_electrode_library()
    models_by_id = model_map(library)

    log_arr, ras_to_ijk = _compute_log_volume(ct_path)

    out: list[dict[str, Any]] = []
    for traj in trajectories:
        name = str(traj.get("name") or "")
        start = traj["start_ras"]
        end = traj["end_ras"]
        restrict = str(traj.get("electrode_model") or "") or None
        result = detect_contacts_on_axis(
            start, end,
            log_volume_kji=log_arr,
            ras_to_ijk_mat=ras_to_ijk,
            models_by_id=models_by_id,
            restrict_to_model_id=restrict,
        )
        if result.rejected_reason:
            _stderr(f"[contacts] {name}: rejected ({result.rejected_reason})")
            out.append({
                "trajectory": name,
                "electrode_model": restrict or "",
                "positions_ras": [],
                "peak_detected": [],
            })
            continue
        out.append({
            "trajectory": name,
            "electrode_model": result.model_id,
            "positions_ras": [list(p) for p in result.positions_ras],
            "peak_detected": list(result.peak_detected),
        })
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rosa-agent contacts",
        description="Place contacts along each trajectory using peak-driven LoG sampling.",
    )
    parser.add_argument("trajectories_tsv", help="Trajectory TSV (rosa-agent detect output)")
    parser.add_argument("ct_path", help="CT NIfTI/NRRD")
    parser.add_argument("--out", "-o", required=True, help="Output contacts TSV")
    args = parser.parse_args(argv)

    trajs = read_seeds_tsv(args.trajectories_tsv)
    _stderr(f"[contacts] {len(trajs)} trajectories from {args.trajectories_tsv}")
    groups = place_contacts(args.ct_path, trajs)
    n = write_contacts_tsv(args.out, groups)
    _stderr(f"[contacts] wrote {args.out} ({n} contacts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
