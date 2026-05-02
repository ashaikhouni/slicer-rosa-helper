"""rosa-agent label — assign atlas labels to a contacts TSV.

Wraps the headless atlas providers + ``rosa_core.atlas_assignment_policy``
(pure-Python policy file) to produce a per-contact label TSV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from rosa_core.atlas_assignment_policy import (
    build_assignment_row,
    choose_closest_sample,
    collect_provider_samples,
)

from ..io.trajectory_io import (
    read_tsv_rows,
    write_tsv_rows,
)
from ..services.atlas_provider_headless import (
    LabelmapAtlasProvider,
    ThomasAtlasProvider,
)


LABEL_COLUMNS: tuple[str, ...] = (
    "trajectory",
    "contact_label",
    "contact_index",
    "contact_x", "contact_y", "contact_z",
    "closest_source",
    "closest_label",
    "closest_label_value",
    "closest_distance_to_voxel_mm",
    "thomas_label",
    "thomas_distance_to_voxel_mm",
    "freesurfer_label",
    "freesurfer_distance_to_voxel_mm",
    "wm_label",
    "wm_distance_to_voxel_mm",
)


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def _build_providers(
    *,
    thomas_dir: str | None,
    freesurfer_path: str | None,
    freesurfer_lut: str | None,
    wm_path: str | None,
    wm_lut: str | None,
    atlas_base_path: str | None = None,
    target_volume_path: str | None = None,
) -> dict[str, Any]:
    """Construct the headless atlas providers.

    When ``atlas_base_path`` and ``target_volume_path`` are both passed,
    the FreeSurfer / WM labelmaps are rigidly registered + resampled
    onto the target volume's grid (so contacts in target RAS align with
    atlas labels). THOMAS skips this — its segmentations are typically
    already in the same frame as the labelmap they're paired with.
    """
    providers: dict[str, Any] = {}

    if thomas_dir:
        try:
            providers["thomas"] = ThomasAtlasProvider(thomas_dir)
            _stderr(f"[label] thomas: ready ({len(providers['thomas']._labels)} voxels)")
        except Exception as exc:
            _stderr(f"[label] thomas provider failed: {exc}")
            providers["thomas"] = None
    else:
        providers["thomas"] = None

    if freesurfer_path:
        try:
            providers["freesurfer"] = LabelmapAtlasProvider(
                source_id="freesurfer",
                display_name="FreeSurfer",
                label_path=freesurfer_path,
                lut_path=freesurfer_lut,
                atlas_base_path=atlas_base_path,
                target_volume_path=target_volume_path,
                logger=_stderr,
            )
            _stderr(
                f"[label] freesurfer: ready "
                f"({len(providers['freesurfer']._labels)} voxels)"
                f"{' (registered)' if atlas_base_path else ''}"
            )
        except Exception as exc:
            _stderr(f"[label] freesurfer provider failed: {exc}")
            providers["freesurfer"] = None
    else:
        providers["freesurfer"] = None

    if wm_path:
        try:
            providers["wm"] = LabelmapAtlasProvider(
                source_id="wm",
                display_name="WM",
                label_path=wm_path,
                lut_path=wm_lut,
                atlas_base_path=atlas_base_path,
                target_volume_path=target_volume_path,
                logger=_stderr,
            )
            _stderr(f"[label] wm: ready ({len(providers['wm']._labels)} voxels)")
        except Exception as exc:
            _stderr(f"[label] wm provider failed: {exc}")
            providers["wm"] = None
    else:
        providers["wm"] = None

    return providers


def label_contacts(
    contacts: list[dict[str, Any]],
    providers: dict[str, Any],
) -> list[dict[str, Any]]:
    """Apply the providers to each contact row and return assignment rows."""
    out: list[dict[str, Any]] = []
    for contact in contacts:
        try:
            point_ras = (
                float(contact["x"]),
                float(contact["y"]),
                float(contact["z"]),
            )
        except (KeyError, ValueError) as exc:
            _stderr(f"[label] skipping malformed contact row: {exc}")
            continue
        contact_dict = {
            "trajectory": contact.get("trajectory", ""),
            "label": contact.get("label", ""),
            "index": int(contact.get("contact_index", 0) or 0),
        }
        samples = collect_provider_samples(point_ras, providers)
        closest_source, closest = choose_closest_sample(samples)
        row = build_assignment_row(
            contact_dict, point_ras, samples, closest_source, closest,
        )
        # Project to the flat TSV column set.
        out.append({
            "trajectory": row["trajectory"],
            "contact_label": row["contact_label"],
            "contact_index": row["contact_index"],
            "contact_x": row["contact_ras"][0],
            "contact_y": row["contact_ras"][1],
            "contact_z": row["contact_ras"][2],
            "closest_source": row["closest_source"],
            "closest_label": row["closest_label"],
            "closest_label_value": row["closest_label_value"],
            "closest_distance_to_voxel_mm": row["closest_distance_to_voxel_mm"],
            "thomas_label": row["thomas_label"],
            "thomas_distance_to_voxel_mm": row["thomas_distance_to_voxel_mm"],
            "freesurfer_label": row["freesurfer_label"],
            "freesurfer_distance_to_voxel_mm": row["freesurfer_distance_to_voxel_mm"],
            "wm_label": row["wm_label"],
            "wm_distance_to_voxel_mm": row["wm_distance_to_voxel_mm"],
        })
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rosa-agent label",
        description="Assign atlas labels to a contacts TSV via headless providers.",
    )
    parser.add_argument("contacts_tsv", help="Input contacts TSV (rosa-agent contacts output)")
    parser.add_argument("--thomas", default="", help="THOMAS segmentation directory (optional)")
    parser.add_argument("--freesurfer", default="", help="FreeSurfer parcellation labelmap (optional)")
    parser.add_argument("--freesurfer-lut", default="", help="FreeSurfer color LUT (optional)")
    parser.add_argument("--wm", default="", help="White-matter labelmap (optional)")
    parser.add_argument("--wm-lut", default="", help="White-matter LUT (optional)")
    parser.add_argument(
        "--atlas-base", default="",
        help="T1 / base volume the FreeSurfer / WM atlases were reconned on. "
             "When passed alongside --target-volume, the atlases are rigidly "
             "registered to the target before sampling.",
    )
    parser.add_argument(
        "--target-volume", default="",
        help="Volume the contacts live in (typically the postop CT). "
             "Required when --atlas-base is passed.",
    )
    parser.add_argument("--out", "-o", required=True, help="Output labels TSV")
    args = parser.parse_args(argv)

    if bool(args.atlas_base) != bool(args.target_volume):
        parser.error("--atlas-base and --target-volume must be passed together")

    contacts = read_tsv_rows(args.contacts_tsv)
    if not contacts:
        _stderr("[label] no contacts to label")
        write_tsv_rows(args.out, [], LABEL_COLUMNS)
        return 0

    providers = _build_providers(
        thomas_dir=args.thomas or None,
        freesurfer_path=args.freesurfer or None,
        freesurfer_lut=args.freesurfer_lut or None,
        wm_path=args.wm or None,
        wm_lut=args.wm_lut or None,
        atlas_base_path=args.atlas_base or None,
        target_volume_path=args.target_volume or None,
    )
    if not any(p is not None and p.is_ready() for p in providers.values()):
        _stderr("[label] no atlas providers configured — nothing to label")
        write_tsv_rows(args.out, [], LABEL_COLUMNS)
        return 0

    rows = label_contacts(contacts, providers)
    write_tsv_rows(args.out, rows, LABEL_COLUMNS)
    _stderr(f"[label] wrote {args.out} ({len(rows)} contacts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
