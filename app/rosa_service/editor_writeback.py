"""Persist an edited trajectory plan back to the case TSVs.

The editor works in crop-index space (``world_RAS = crop_index + plan.origin``) and
models each electrode as a rigid comb: a straight entry→target axis + per-contact
offsets from the tip + a tip-depth. Saving therefore does two things by design:

  * rewrites ``trajectories.tsv`` from the (possibly moved / re-modelled / renamed /
    added / removed) endpoints, and
  * regenerates ``contacts.tsv`` from each comb — which is exactly the linear-
    trajectory constraint the editor enforces (contacts snap onto the fitted line).

Reproduces the editor's own contact math (editor/index.html ``planContacts``):
``tip = target + axis·tipOffset``; ``contact[i] = tip − axis·offset[i]``.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

TRAJ_COLS = ["name", "start_x", "start_y", "start_z", "end_x", "end_y", "end_z",
             "confidence", "confidence_label", "electrode_model", "bolt_source", "length_mm"]
CONTACT_COLS = ["trajectory", "label", "contact_index", "x", "y", "z",
                "peak_detected", "electrode_model"]


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _write_tsv(path: Path, cols: list[str], rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_plan(job_dir: str | Path, plan: dict) -> dict:
    """Rewrite ``trajectories.tsv`` + ``contacts.tsv`` from an edited plan.

    Returns ``{n_trajectories, n_contacts}``. Raises ValueError on a malformed
    plan (missing origin / models / a trajectory's model).
    """
    job_dir = Path(job_dir)
    if "origin" not in plan or "trajectories" not in plan or "models" not in plan:
        raise ValueError("plan must carry origin, models, trajectories")
    origin = np.asarray(plan["origin"], dtype=float)
    models = plan["models"]

    # Preserve per-shank provenance columns (confidence / bolt_source) from the
    # original TSV where the name still exists; new shanks get sensible defaults.
    orig = {r["name"]: r for r in _rows(job_dir / "trajectories.tsv")}

    traj_rows, contact_rows, renames = [], [], {}
    for t in plan["trajectories"]:
        name = str(t["name"])
        old = str(t.get("name0") or name)
        if old != name:
            renames[name] = old              # so labels can follow a rename
        model = str(t["model"])
        if model not in models:
            raise ValueError(f"trajectory {name}: unknown model {model!r}")
        m = models[model]
        entry = np.asarray(t["entry"], dtype=float) + origin        # crop-index → world
        target = np.asarray(t["target"], dtype=float) + origin
        axis = _unit(target - entry)
        length = float(np.linalg.norm(target - entry))
        o = orig.get(name, {})
        traj_rows.append({
            "name": name,
            "start_x": f"{entry[0]:.6f}", "start_y": f"{entry[1]:.6f}", "start_z": f"{entry[2]:.6f}",
            "end_x": f"{target[0]:.6f}", "end_y": f"{target[1]:.6f}", "end_z": f"{target[2]:.6f}",
            "confidence": o.get("confidence", ""),
            "confidence_label": t.get("confidence_label") or o.get("confidence_label", "edited"),
            "electrode_model": model,
            "bolt_source": o.get("bolt_source", ""),
            "length_mm": f"{length:.3f}",
        })
        # rigid comb: tip = target + axis·tipOffset; contact[i] = tip − axis·offset[i]
        tip = target + axis * float(t.get("tipOffset", 0.0))
        for i, off in enumerate(m["offsets"]):
            p = tip - axis * float(off)
            contact_rows.append({
                "trajectory": name, "label": f"{name}{i + 1}", "contact_index": i + 1,
                "x": f"{p[0]:.6f}", "y": f"{p[1]:.6f}", "z": f"{p[2]:.6f}",
                "peak_detected": 1, "electrode_model": model,
            })

    _write_tsv(job_dir / "trajectories.tsv", TRAJ_COLS, traj_rows)
    _write_tsv(job_dir / "contacts.tsv", CONTACT_COLS, contact_rows)
    # Geometry changed → the editor + review caches are stale; drop them so they
    # rebuild from the new TSVs on next access.
    for f in ("editor_plan.json", "editor_ct.i16"):
        (job_dir / f).unlink(missing_ok=True)
    return {"n_trajectories": len(traj_rows), "n_contacts": len(contact_rows), "renames": renames}


__all__ = ["write_plan"]
