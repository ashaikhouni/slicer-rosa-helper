"""Write a standardized QC directory for ``place_seeg`` outputs.

Schema (per the 2026-05-09 design memo, handoff_v3_production_lift):

    output_dir/
      manifest.json          — run metadata + counts + per-mode diag
      trajectories.tsv       — one row per emitted trajectory (rich QC schema,
                                strict superset of the existing public
                                TRAJECTORY_COLUMNS contract)
      contacts.tsv           — one row per placed contact (matches the
                                existing CONTACT_COLUMNS contract)
      figures/               — per-trajectory PNG (optional; needs matplotlib)
        001_<name>.png
        002_<name>.png
      diagnostics/
        cmp.tsv              — full per-emission score-component dump
                                (same as the notebook's cmp_df)

Used by ``rosa-agent place`` (Session 3 CLI subcommand) and callable from
Slicer / notebooks for one-shot QC dumps.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


# ---------------------------------------------------------------------
# Public column set — strict superset of TRAJECTORY_COLUMNS.
# First 12 columns match cli.rosa_agent.io.trajectory_io for back-compat;
# remaining columns are the QC-specific score breakdown.
# ---------------------------------------------------------------------


QC_TRAJECTORY_COLUMNS: tuple[str, ...] = (
    # --- Public TRAJECTORY_COLUMNS (CLI contract) ---
    "name",
    "start_x", "start_y", "start_z",
    "end_x", "end_y", "end_z",
    "confidence",          # = compound_score (back-compat alias)
    "confidence_label",    # = band (back-compat alias)
    "electrode_model",     # = model_id (back-compat alias)
    "bolt_source",
    "length_mm",
    # --- QC additions ---
    "model_id",
    "band",
    "compound_score",
    "bolt_end_arc_mm",
    "signal_kind",
    "n_placed",
    "n_slots",
    "n_covered",
    "n_covered_frac",
    # Sub-scores (the seven compound components + bolt-only penalty).
    "s_corr",
    "s_fft",
    "s_tube",
    "s_margin",
    "s_walker",
    "s_cc_overlap",
    "s_seeder",
    "bolt_only_penalty",
    # Raw measurements.
    "corr",
    "tube_like_frac",
    "model_corr_margin",
    "model_corr_uniformity",
    "zone_cv",
    "zone_ptp_mod",
    "bolt_zone_frac",
    "pitch_power_frac",
    "fft_n_segments",
    "fft_n_reliable_segments",
    "fft_subject_norm",
    "pitch_mm",
    "model_uniform_pitch",
    "placed_hu_mean",
    "placed_hu_min",
    # CC overlap.
    "cc_match",
    "cc_dist_mm",
    "cc_arc_along_mm",
    "cc_overlap_score",
    "cc_n_voxels",
)


CONTACT_COLUMNS: tuple[str, ...] = (
    "trajectory",
    "label",
    "contact_index",
    "x", "y", "z",
    "peak_detected",
    "electrode_model",
)


PIPELINE_VERSION = "v3-staged-1.0.0"


# ---------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------


def write_qc_directory(
    batch,
    output_dir: str | Path,
    *,
    ct_path: str | Path | None = None,
    subject_id: str | None = None,
    library_id: str | None = None,
    mode_args: dict | None = None,
    runtime_seconds: float | None = None,
    write_figures: bool = True,
    features: dict | None = None,
    bolts: list[dict] | None = None,
    extra: dict | None = None,
) -> Path:
    """Write a standardized QC directory for a ``PlacementBatch``.

    Args:
        batch: ``PlacementBatch`` from ``rosa_core.placement_modes.place_seeg``.
        output_dir: where to write. Created (parents=True) if missing.
        ct_path, subject_id, library_id: optional metadata stamped into
            ``manifest.json``.
        mode_args: dict echoing the dispatcher args (seeds names, expected
            list, n_expected) for reproducibility.
        runtime_seconds: end-to-end runtime if the caller measured it.
        write_figures: render per-trajectory PNG via ``qc_figures``.
            Skipped silently when matplotlib isn't available.
        features, bolts: needed to render figures; optional otherwise.
        extra: additional fields to include in manifest.json.

    Returns:
        Path to ``output_dir`` (so the caller can chain).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    write_trajectories_tsv(out / "trajectories.tsv", batch.trajectories)
    write_contacts_tsv(out / "contacts.tsv", batch.trajectories)
    write_diagnostics_cmp_tsv(out / "diagnostics" / "cmp.tsv", batch.trajectories)
    write_manifest_json(
        out / "manifest.json", batch,
        ct_path=ct_path, subject_id=subject_id, library_id=library_id,
        mode_args=mode_args, runtime_seconds=runtime_seconds,
        extra=extra,
    )

    if write_figures and features is not None:
        try:
            from .qc_figures import render_all_figures
            render_all_figures(
                batch.trajectories, out / "figures",
                features=features, bolts=bolts,
            )
        except ImportError:
            # matplotlib not available — silently skip; QC TSVs are still
            # the primary artifact.
            pass

    return out


# ---------------------------------------------------------------------
# trajectories.tsv
# ---------------------------------------------------------------------


def trajectory_to_qc_row(traj) -> dict[str, str]:
    """Format one ``PlacedTrajectory`` as a ``QC_TRAJECTORY_COLUMNS`` row.

    The first 12 columns mirror the existing TRAJECTORY_COLUMNS contract
    (so other tools reading old-style TSV keep working); the remaining
    columns surface the staged-pipeline score breakdown.
    """
    sc = traj.score_components or {}
    sub = sc.get("subscores", {}) or {}
    sx, sy, sz = float(traj.start_ras[0]), float(traj.start_ras[1]), float(traj.start_ras[2])
    ex, ey, ez = float(traj.end_ras[0]), float(traj.end_ras[1]), float(traj.end_ras[2])
    length = ((ex - sx) ** 2 + (ey - sy) ** 2 + (ez - sz) ** 2) ** 0.5

    return {
        # Back-compat columns.
        "name": str(traj.name),
        "start_x": _fmt_float(sx),
        "start_y": _fmt_float(sy),
        "start_z": _fmt_float(sz),
        "end_x": _fmt_float(ex),
        "end_y": _fmt_float(ey),
        "end_z": _fmt_float(ez),
        "confidence": _fmt_float(traj.compound_score, precision=4),
        "confidence_label": str(traj.band or ""),
        "electrode_model": str(traj.model_id or ""),
        "bolt_source": str(traj.bolt_source or ""),
        "length_mm": _fmt_float(length, precision=3),
        # QC additions.
        "model_id": str(traj.model_id or ""),
        "band": str(traj.band or ""),
        "compound_score": _fmt_float(traj.compound_score, precision=4),
        "bolt_end_arc_mm": _fmt_float(traj.bolt_end_arc_mm, precision=3),
        "signal_kind": str((traj.diagnostics or {}).get("signal_kind", "")),
        "n_placed": str(int(sc.get("n_placed", 0))),
        "n_slots": str(int(sc.get("n_slots", 0))),
        "n_covered": str(int(sc.get("n_covered", 0))),
        "n_covered_frac": _fmt_float(sc.get("n_covered_frac"), precision=3),
        # Sub-scores.
        "s_corr": _fmt_float(sub.get("s_corr"), precision=4),
        "s_fft": _fmt_float(sub.get("s_fft"), precision=4),
        "s_tube": _fmt_float(sub.get("s_tube"), precision=4),
        "s_margin": _fmt_float(sub.get("s_margin"), precision=4),
        "s_walker": _fmt_float(sub.get("s_walker"), precision=4),
        "s_cc_overlap": _fmt_float(sub.get("s_cc_overlap"), precision=4),
        "s_seeder": _fmt_float(sub.get("s_seeder"), precision=4),
        "bolt_only_penalty": _fmt_float(sub.get("bolt_only_penalty"), precision=4),
        # Raw.
        "corr": _fmt_float(sc.get("corr"), precision=4),
        "tube_like_frac": _fmt_float(sc.get("tube_like_frac"), precision=4),
        "model_corr_margin": _fmt_float(sc.get("model_corr_margin"), precision=4),
        "model_corr_uniformity": _fmt_float(sc.get("model_corr_uniformity"), precision=4),
        "zone_cv": _fmt_float(sc.get("zone_cv"), precision=4),
        "zone_ptp_mod": _fmt_float(sc.get("zone_ptp_mod"), precision=4),
        "bolt_zone_frac": _fmt_float(sc.get("bolt_zone_frac"), precision=4),
        "pitch_power_frac": _fmt_float(sc.get("pitch_power_frac"), precision=4),
        "fft_n_segments": str(int(sc.get("fft_n_segments", 0))),
        "fft_n_reliable_segments": str(int(sc.get("fft_n_reliable_segments", 0))),
        "fft_subject_norm": _fmt_float(sc.get("fft_subject_norm"), precision=4),
        "pitch_mm": _fmt_float(sc.get("pitch_mm"), precision=3),
        "model_uniform_pitch": "1" if sc.get("model_uniform_pitch") else "0",
        "placed_hu_mean": _fmt_float(sc.get("placed_hu_mean"), precision=1),
        "placed_hu_min": _fmt_float(sc.get("placed_hu_min"), precision=1),
        # CC overlap.
        "cc_match": str(sc.get("cc_match") or ""),
        "cc_dist_mm": _fmt_float(sc.get("cc_dist_mm"), precision=3),
        "cc_arc_along_mm": _fmt_float(sc.get("cc_arc_along_mm"), precision=3),
        "cc_overlap_score": _fmt_float(sc.get("cc_overlap_score"), precision=4),
        "cc_n_voxels": str(int(sc.get("cc_n_voxels", 0))),
    }


def write_trajectories_tsv(path: str | Path, trajectories: Iterable) -> int:
    """Write trajectories.tsv with the rich QC schema. Returns count."""
    rows = [trajectory_to_qc_row(t) for t in trajectories]
    _write_tsv(path, rows, QC_TRAJECTORY_COLUMNS)
    return len(rows)


# ---------------------------------------------------------------------
# contacts.tsv
# ---------------------------------------------------------------------


def write_contacts_tsv(path: str | Path, trajectories: Iterable) -> int:
    """Write contacts.tsv (one row per contact) matching the existing
    CONTACT_COLUMNS contract."""
    rows: list[dict[str, str]] = []
    for traj in trajectories:
        for idx, pt in enumerate(traj.contacts_ras or [], start=1):
            rows.append({
                "trajectory": str(traj.name),
                "label": f"{traj.name}{idx}",
                "contact_index": str(idx),
                "x": _fmt_float(pt[0]),
                "y": _fmt_float(pt[1]),
                "z": _fmt_float(pt[2]),
                # The placer always emits library-template-anchored contacts;
                # there isn't a peak-detection flag to surface, so 1.
                "peak_detected": "1",
                "electrode_model": str(traj.model_id or ""),
            })
    _write_tsv(path, rows, CONTACT_COLUMNS)
    return len(rows)


# ---------------------------------------------------------------------
# manifest.json
# ---------------------------------------------------------------------


def write_manifest_json(
    path: str | Path, batch, *,
    ct_path=None, subject_id=None, library_id=None,
    mode_args=None, runtime_seconds=None, extra=None,
) -> None:
    """Write manifest.json — run metadata + band counts + per-mode diag."""
    bands = [t.band for t in batch.trajectories]
    diag = dict(batch.diagnostics or {})
    manifest = {
        "subject_id": subject_id or "",
        "ct_path": str(ct_path) if ct_path else "",
        "pipeline_version": PIPELINE_VERSION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "library_id": library_id or "",
        "mode": int(diag.get("mode", 0)),
        "mode_args": mode_args or {},
        "n_trajectories": len(batch.trajectories),
        "n_high": bands.count("high"),
        "n_medium": bands.count("medium"),
        "n_low": bands.count("low"),
        "n_input_seeds": int(diag.get("n_input_seeds", 0)),
        "n_emitted": int(diag.get("n_emitted", len(batch.trajectories))),
        "n_library_models": int(diag.get("n_library_models", 0)),
        "subject_fft_normalized": bool(diag.get("subject_fft_normalized", False)),
        "band_floor": diag.get("band_floor"),
        "runtime_seconds": runtime_seconds,
        # Per-mode diagnostic blocks (only the active one will be present).
        "mode1": diag.get("mode1"),
        "mode3": diag.get("mode3"),
        "mode4": diag.get("mode4"),
        "mode5": diag.get("mode5"),
    }
    if extra:
        manifest.update(extra)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, default=_json_default))


# ---------------------------------------------------------------------
# diagnostics/cmp.tsv — full score-component dump
# ---------------------------------------------------------------------


def write_diagnostics_cmp_tsv(path: str | Path, trajectories: Iterable) -> int:
    """Write the full per-emission diagnostic dump.

    Same content as the notebook's ``cmp_df`` — every score component
    flattened into one TSV. Useful for downstream analysis without parsing
    JSON.
    """
    rows = []
    for traj in trajectories:
        sc = traj.score_components or {}
        sub = sc.get("subscores", {}) or {}
        row = {
            "name": traj.name,
            "model_id": traj.model_id or "",
            "band": traj.band or "",
            "compound_score": _fmt_float(traj.compound_score, precision=6),
            "bolt_source": traj.bolt_source or "",
            "bolt_end_arc_mm": _fmt_float(traj.bolt_end_arc_mm, precision=4),
        }
        # Score components — all of them, with stable keys.
        for k, v in sorted(sc.items()):
            if k == "subscores" or k == "per_model_corr":
                continue
            row[k] = _fmt_value(v)
        for k, v in sorted(sub.items()):
            row[f"sub_{k}"] = _fmt_value(v)
        rows.append(row)
    if not rows:
        _write_tsv(path, rows, ("name",))
        return 0
    # Use union of all keys across rows for stable columns.
    keys = set()
    for r in rows:
        keys.update(r.keys())
    # Lead with the canonical columns, append the rest sorted.
    leading = [
        "name", "model_id", "band", "compound_score", "bolt_source",
        "bolt_end_arc_mm",
    ]
    rest = sorted(k for k in keys if k not in leading)
    columns = tuple(leading + rest)
    _write_tsv(path, rows, columns)
    return len(rows)


# ---------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------


def _fmt_float(value: Any, *, precision: int = 6) -> str:
    if value is None or value == "":
        return ""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return ""
    if f != f:  # NaN
        return ""
    return f"{f:.{precision}f}"


def _fmt_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        if value != value:
            return ""
        return f"{value:.6f}"
    return str(value)


def _json_default(obj):
    """Custom JSON encoder for numpy / pathlib / sets / np.ndarray."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"object of type {type(obj).__name__} is not JSON-serializable")


def _write_tsv(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    import csv
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


__all__ = [
    "QC_TRAJECTORY_COLUMNS",
    "CONTACT_COLUMNS",
    "PIPELINE_VERSION",
    "write_qc_directory",
    "write_trajectories_tsv",
    "write_contacts_tsv",
    "write_manifest_json",
    "write_diagnostics_cmp_tsv",
    "trajectory_to_qc_row",
]
