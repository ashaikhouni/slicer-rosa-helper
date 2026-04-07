#!/usr/bin/env python3
"""Export SEEG dataset contacts/shanks into rosa_helper ContactImport TSV format.

Produces, for each subject:
- `contacts.tsv` with columns `trajectory_name,index,x,y,z,label`
- `trajectories.tsv` with columns `name,ex,ey,ez,tx,ty,tz`

Coordinates are written in world RAS, which matches the default import mode in
`ContactImport`.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIB_DIR = os.path.join(REPO_ROOT, "CommonLib")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

from shank_core.io import image_ijk_ras_matrices, ras_to_ijk_float_matrix, read_volume  # noqa: E402
from shank_core.masking import build_preview_masks  # noqa: E402


def _read_tsv_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _write_tsv_rows(path: str | Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _to_float(row: dict[str, str], key: str) -> float:
    return float(str(row.get(key, "")).strip())


def _depth_at_ras(point_ras: np.ndarray, depth_map_kji: np.ndarray | None, ras_to_ijk) -> float | None:
    if depth_map_kji is None:
        return None
    ijk = ras_to_ijk(point_ras)
    i = int(round(float(ijk[0])))
    j = int(round(float(ijk[1])))
    k = int(round(float(ijk[2])))
    if k < 0 or j < 0 or i < 0:
        return None
    if k >= depth_map_kji.shape[0] or j >= depth_map_kji.shape[1] or i >= depth_map_kji.shape[2]:
        return None
    value = float(depth_map_kji[k, j, i])
    if not math.isfinite(value):
        return None
    return value


def _orient_entry_target(
    start_ras: np.ndarray,
    end_ras: np.ndarray,
    depth_map_kji: np.ndarray | None,
    ras_to_ijk,
) -> tuple[np.ndarray, np.ndarray, str]:
    d0 = _depth_at_ras(start_ras, depth_map_kji, ras_to_ijk)
    d1 = _depth_at_ras(end_ras, depth_map_kji, ras_to_ijk)
    if d0 is not None and d1 is not None and abs(d0 - d1) > 1e-3:
        if d0 <= d1:
            return start_ras, end_ras, "depth_map_start_is_entry"
        return end_ras, start_ras, "depth_map_end_is_entry"
    return start_ras, end_ras, "original_order"


def _subject_depth_map(
    ct_path: str,
    *,
    head_mask_method: str,
    head_mask_threshold_hu: float,
    min_metal_depth_mm: float,
    max_metal_depth_mm: float,
) -> tuple[np.ndarray | None, callable]:
    img, arr_kji, spacing_xyz = read_volume(ct_path)
    _, ras_to_ijk = image_ijk_ras_matrices(img)
    ras_to_ijk_fn = lambda ras_xyz: ras_to_ijk_float_matrix(ras_xyz, ras_to_ijk)
    preview = build_preview_masks(
        arr_kji=np.asarray(arr_kji, dtype=float),
        spacing_xyz=tuple(spacing_xyz),
        threshold=1800.0,
        use_head_mask=True,
        build_head_mask=True,
        head_mask_threshold_hu=float(head_mask_threshold_hu),
        head_mask_method=str(head_mask_method),
        head_gate_erode_vox=1,
        head_gate_dilate_vox=1,
        head_gate_margin_mm=0.0,
        min_metal_depth_mm=float(min_metal_depth_mm),
        max_metal_depth_mm=float(max_metal_depth_mm),
        include_debug_masks=False,
    )
    depth_map = preview.get("head_distance_map_kji")
    return (np.asarray(depth_map, dtype=np.float32) if depth_map is not None else None), ras_to_ijk_fn


def _build_contacts_rows(subject_id: str, label_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in sorted(label_rows, key=lambda item: (str(item.get("shank", "")), int(float(item.get("contact_index", 0) or 0)))):
        rows.append(
            {
                "trajectory_name": str(row.get("shank", "")).strip(),
                "index": int(float(row.get("contact_index", 0) or 0)),
                "x": f"{_to_float(row, 'x'):.6f}",
                "y": f"{_to_float(row, 'y'):.6f}",
                "z": f"{_to_float(row, 'z'):.6f}",
                "label": str(row.get("channel", "")).strip() or f"{row.get('shank', '')}{row.get('contact_index', '')}",
                "subject_id": subject_id,
                "channel": str(row.get("channel", "")).strip(),
                "coord_source": str(row.get("coord_source", "")).strip(),
                "snap_status": str(row.get("snap_status", "")).strip(),
                "source_contacts_file": str(row.get("source_contacts_file", "")).strip(),
            }
        )
    return rows


def _build_trajectories_rows(
    subject_id: str,
    shank_rows: list[dict[str, str]],
    depth_map_kji: np.ndarray | None,
    ras_to_ijk,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in sorted(shank_rows, key=lambda item: str(item.get("shank", ""))):
        start_ras = np.asarray([_to_float(row, "start_x"), _to_float(row, "start_y"), _to_float(row, "start_z")], dtype=float)
        end_ras = np.asarray([_to_float(row, "end_x"), _to_float(row, "end_y"), _to_float(row, "end_z")], dtype=float)
        entry_ras, target_ras, orientation_method = _orient_entry_target(start_ras, end_ras, depth_map_kji, ras_to_ijk)
        rows.append(
            {
                "name": str(row.get("shank", "")).strip(),
                "ex": f"{float(entry_ras[0]):.6f}",
                "ey": f"{float(entry_ras[1]):.6f}",
                "ez": f"{float(entry_ras[2]):.6f}",
                "tx": f"{float(target_ras[0]):.6f}",
                "ty": f"{float(target_ras[1]):.6f}",
                "tz": f"{float(target_ras[2]):.6f}",
                "subject_id": subject_id,
                "contact_count": int(float(row.get("contact_count", 0) or 0)),
                "span_mm": f"{float(row.get('span_mm', 0.0) or 0.0):.6f}",
                "mean_intercontact_mm": f"{float(row.get('mean_intercontact_mm', 0.0) or 0.0):.6f}",
                "fit_method": str(row.get("fit_method", "")).strip(),
                "orientation_method": orientation_method,
                "source_labels_path": str(row.get("source_labels_path", "")).strip(),
            }
        )
    return rows


def export_subject(
    *,
    subject_row: dict[str, str],
    out_dir: Path,
    head_mask_method: str,
    head_mask_threshold_hu: float,
    min_metal_depth_mm: float,
    max_metal_depth_mm: float,
) -> dict[str, object]:
    subject_id = str(subject_row["subject_id"]).strip()
    label_rows = _read_tsv_rows(subject_row["labels_path"])
    shank_rows = _read_tsv_rows(subject_row["shanks_path"])
    depth_map_kji, ras_to_ijk = _subject_depth_map(
        subject_row["ct_path"],
        head_mask_method=head_mask_method,
        head_mask_threshold_hu=head_mask_threshold_hu,
        min_metal_depth_mm=min_metal_depth_mm,
        max_metal_depth_mm=max_metal_depth_mm,
    )

    contacts_rows = _build_contacts_rows(subject_id, label_rows)
    trajectories_rows = _build_trajectories_rows(subject_id, shank_rows, depth_map_kji, ras_to_ijk)

    subject_dir = out_dir / subject_id
    contacts_path = subject_dir / "contacts.tsv"
    trajectories_path = subject_dir / "trajectories.tsv"
    _write_tsv_rows(
        contacts_path,
        ["trajectory_name", "index", "x", "y", "z", "label", "subject_id", "channel", "coord_source", "snap_status", "source_contacts_file"],
        contacts_rows,
    )
    _write_tsv_rows(
        trajectories_path,
        ["name", "ex", "ey", "ez", "tx", "ty", "tz", "subject_id", "contact_count", "span_mm", "mean_intercontact_mm", "fit_method", "orientation_method", "source_labels_path"],
        trajectories_rows,
    )
    return {
        "subject_id": subject_id,
        "ct_path": subject_row["ct_path"],
        "contacts_path": str(contacts_path),
        "trajectories_path": str(trajectories_path),
        "n_contacts": len(contacts_rows),
        "n_trajectories": len(trajectories_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export SEEG subject contacts/shanks to rosa_helper ContactImport TSV format")
    parser.add_argument("--dataset-root", required=True, help="Root of the seeg_localization dataset")
    parser.add_argument("--out-dir", required=True, help="Output directory for per-subject rosa_helper import files")
    parser.add_argument("--subjects", default="", help="Comma-separated subject ids; default exports all subjects in manifest")
    parser.add_argument("--head-mask-method", default="outside_air", choices=["outside_air", "not_air_lcc"])
    parser.add_argument("--head-mask-threshold-hu", type=float, default=-500.0)
    parser.add_argument("--min-metal-depth-mm", type=float, default=0.0)
    parser.add_argument("--max-metal-depth-mm", type=float, default=220.0)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    subjects_manifest = _read_tsv_rows(dataset_root / "contact_label_dataset" / "subjects.tsv")
    requested = {item.strip() for item in str(args.subjects).split(",") if item.strip()}
    if requested:
        subjects_manifest = [row for row in subjects_manifest if str(row.get("subject_id", "")).strip() in requested]
    if not subjects_manifest:
        raise SystemExit("No matching subjects found in manifest.")

    subject_summary_rows: list[dict[str, object]] = []
    all_contacts_rows: list[dict[str, object]] = []
    all_trajectories_rows: list[dict[str, object]] = []
    for subject_row in subjects_manifest:
        summary = export_subject(
            subject_row=subject_row,
            out_dir=out_dir,
            head_mask_method=args.head_mask_method,
            head_mask_threshold_hu=args.head_mask_threshold_hu,
            min_metal_depth_mm=args.min_metal_depth_mm,
            max_metal_depth_mm=args.max_metal_depth_mm,
        )
        subject_summary_rows.append(summary)
        all_contacts_rows.extend(_read_tsv_rows(summary["contacts_path"]))
        all_trajectories_rows.extend(_read_tsv_rows(summary["trajectories_path"]))
        print(
            f"[export] {summary['subject_id']}: "
            f"contacts={summary['n_contacts']} trajectories={summary['n_trajectories']}"
        )

    _write_tsv_rows(
        out_dir / "subjects.tsv",
        ["subject_id", "ct_path", "contacts_path", "trajectories_path", "n_contacts", "n_trajectories"],
        subject_summary_rows,
    )
    _write_tsv_rows(
        out_dir / "all_contacts.tsv",
        ["trajectory_name", "index", "x", "y", "z", "label", "subject_id", "channel", "coord_source", "snap_status", "source_contacts_file"],
        all_contacts_rows,
    )
    _write_tsv_rows(
        out_dir / "all_trajectories.tsv",
        ["name", "ex", "ey", "ez", "tx", "ty", "tz", "subject_id", "contact_count", "span_mm", "mean_intercontact_mm", "fit_method", "orientation_method", "source_labels_path"],
        all_trajectories_rows,
    )
    print(f"[done] wrote rosa_helper import bundle to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
