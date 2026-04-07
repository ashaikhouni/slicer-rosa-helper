#!/usr/bin/env python3
"""Build cached shank-level ground truth files from per-contact SEEG labels."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def read_tsv_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv_rows(path: str | Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return vec / norm


def fit_line_from_contacts(points_ras: np.ndarray, contact_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    center = points_ras.mean(axis=0)
    centered = points_ras - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = _unit(vh[0])
    proj = centered @ direction
    if len(points_ras) > 1:
        corr = np.corrcoef(proj, contact_indices.astype(float))[0, 1]
        if np.isfinite(corr) and corr < 0.0:
            direction = -direction
            proj = -proj
    start = center + direction * float(np.min(proj))
    end = center + direction * float(np.max(proj))
    span = float(np.max(proj) - np.min(proj))
    return start, end, direction, span


def mean_intercontact_distance(points_ras: np.ndarray) -> float:
    if len(points_ras) < 2:
        return 0.0
    diffs = np.linalg.norm(points_ras[1:] - points_ras[:-1], axis=1)
    return float(np.mean(diffs))


def build_subject_shanks(subject_id: str, labels_path: str) -> list[dict[str, object]]:
    rows = read_tsv_rows(labels_path)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["shank"]), []).append(row)

    out: list[dict[str, object]] = []
    for shank_name, shank_rows in sorted(grouped.items()):
        shank_rows = sorted(shank_rows, key=lambda r: int(r["contact_index"]))
        pts = np.asarray([[float(r["x"]), float(r["y"]), float(r["z"])] for r in shank_rows], dtype=float)
        idx = np.asarray([int(r["contact_index"]) for r in shank_rows], dtype=int)
        start, end, direction, span = fit_line_from_contacts(pts, idx)
        out.append(
            {
                "subject_id": subject_id,
                "shank": shank_name,
                "contact_count": len(shank_rows),
                "start_x": f"{float(start[0]):.6f}",
                "start_y": f"{float(start[1]):.6f}",
                "start_z": f"{float(start[2]):.6f}",
                "end_x": f"{float(end[0]):.6f}",
                "end_y": f"{float(end[1]):.6f}",
                "end_z": f"{float(end[2]):.6f}",
                "dir_x": f"{float(direction[0]):.6f}",
                "dir_y": f"{float(direction[1]):.6f}",
                "dir_z": f"{float(direction[2]):.6f}",
                "span_mm": f"{float(span):.6f}",
                "mean_intercontact_mm": f"{mean_intercontact_distance(pts):.6f}",
                "fit_method": "pca_line_from_contacts",
                "source_labels_path": labels_path,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cached shank-level SEEG ground truth tables")
    parser.add_argument("--dataset-root", required=True, help="Path to seeg_localization root")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    manifest_path = dataset_root / "contact_label_dataset" / "subjects.tsv"
    shanks_dir = dataset_root / "contact_label_dataset" / "shanks"
    all_shanks_path = dataset_root / "contact_label_dataset" / "all_shanks.tsv"

    manifest_rows = read_tsv_rows(manifest_path)
    subject_rows_out: list[dict[str, str]] = []
    all_shanks_rows: list[dict[str, object]] = []

    for row in manifest_rows:
        subject_id = str(row["subject_id"])
        labels_path = str(row["labels_path"])
        subject_shanks = build_subject_shanks(subject_id, labels_path)
        shank_file = shanks_dir / f"{subject_id}_shanks.tsv"
        if subject_shanks:
            fieldnames = list(subject_shanks[0].keys())
            write_tsv_rows(shank_file, subject_shanks, fieldnames)
            all_shanks_rows.extend(subject_shanks)
        row_out = dict(row)
        row_out["shanks_path"] = str(shank_file)
        row_out["n_shanks"] = str(len(subject_shanks))
        subject_rows_out.append(row_out)
        print(f"[build-shanks] {subject_id}: {len(subject_shanks)} shanks")

    if all_shanks_rows:
        write_tsv_rows(all_shanks_path, all_shanks_rows, list(all_shanks_rows[0].keys()))
    write_tsv_rows(manifest_path, subject_rows_out, list(subject_rows_out[0].keys()))
    print(f"[build-shanks] wrote {all_shanks_path}")


if __name__ == "__main__":
    main()
