"""Validate that an imported localization TSV actually belongs to a given CT.

When a batch CLI run is imported for review (CT + ``contacts.tsv``), the one way
it can silently break is a **wrong pairing**: a TSV whose world coordinates were
computed against a *different* CT. Those contacts then land off the electrodes —
or outside the volume entirely.

We catch that at the content level, needing no metadata on the TSV (so it works
on TSVs produced before any provenance sidecar exists): map each contact's world
coordinate into the CT and check it (a) lands inside the volume and (b) sits on
high-HU metal. A right pairing puts (nearly) every contact on an electrode; a
wrong one throws them out of bounds or into air.

The world->voxel mapping mirrors ``editor_payload._build`` EXACTLY (``nib.load``
+ ``inv(affine)``) so the check and the editor never disagree about where a
contact is — the CLI/Slicer parity rule applies here too: one loader, no fork.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

# HU at/above which a sampled voxel counts as electrode metal. Contacts saturate
# CT HU; 800 clears bone (~<600 clamped) with margin. We take the max over a
# 3x3x3 window so a contact centred a fraction of a voxel off a peak still reads
# as on-metal (matches the editor's "peak HU" feel, forgivingly).
METAL_HU = 800.0
CT_CLAMP = (-1024, 3071)

# Verdict bands. RED is a hard block (clearly the wrong CT); YELLOW imports but
# asks the user to confirm; GREEN is a clean pairing.
MIN_INBOUNDS_OK = 0.95      # below -> at best YELLOW
INBOUNDS_RED = 0.50         # contacts falling outside the volume -> wrong CT
ONMETAL_RED = 0.20          # almost nothing on metal -> wrong CT / wrong coords
ONMETAL_GREEN = 0.70        # most contacts on metal -> trust the pairing


def _contact_world_points(contacts_tsv: Path) -> np.ndarray:
    """(N,3) world (x,y,z) for every contact row; skips malformed rows."""
    pts: list[list[float]] = []
    with open(contacts_tsv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                pts.append([float(row["x"]), float(row["y"]), float(row["z"])])
            except (KeyError, ValueError, TypeError):
                continue
    return np.asarray(pts, dtype=float).reshape(-1, 3)


def check_localization(ct_path: str | Path, contacts_tsv: str | Path) -> dict[str, Any]:
    """Score how well ``contacts_tsv``'s coordinates fit ``ct_path``.

    Returns a summary the import UI shows verbatim::

        {n, n_in_bounds, n_on_metal, frac_in_bounds, frac_on_metal,
         verdict: "green"|"yellow"|"red", reason}

    Raises FileNotFoundError if either file is missing, ValueError if the TSV has
    no usable contact rows.
    """
    ct_path, contacts_tsv = Path(ct_path), Path(contacts_tsv)
    if not ct_path.is_file():
        raise FileNotFoundError(f"CT not found: {ct_path}")
    if not contacts_tsv.is_file():
        raise FileNotFoundError(f"contacts TSV not found: {contacts_tsv}")

    pts = _contact_world_points(contacts_tsv)
    n = int(pts.shape[0])
    if n == 0:
        raise ValueError(f"no usable contact rows (need x/y/z) in {contacts_tsv.name}")

    import nibabel as nib
    img = nib.load(str(ct_path))
    arr = np.asanyarray(img.dataobj).astype(np.float32)
    dims = np.array(arr.shape[:3])
    inv = np.linalg.inv(img.affine.astype(float))

    # world -> voxel, identical convention to editor_payload._build
    homog = np.concatenate([pts, np.ones((n, 1))], axis=1)   # (N,4)
    vox = (homog @ inv.T)[:, :3]                              # (N,3) fractional voxel

    inb = np.all((vox >= 0) & (vox <= dims - 1), axis=1)      # inside the volume
    n_in = int(inb.sum())

    # On-metal only makes sense for in-bounds contacts; sample a 3x3x3 max at the
    # nearest voxel (bounds-clamped) and clip to the shipped HU range.
    n_metal = 0
    if n_in:
        centers = np.rint(vox[inb]).astype(int)
        centers = np.clip(centers, 0, dims - 1)
        for cx, cy, cz in centers:
            x0, x1 = max(cx - 1, 0), min(cx + 2, dims[0])
            y0, y1 = max(cy - 1, 0), min(cy + 2, dims[1])
            z0, z1 = max(cz - 1, 0), min(cz + 2, dims[2])
            peak = float(arr[x0:x1, y0:y1, z0:z1].max())
            peak = min(max(peak, CT_CLAMP[0]), CT_CLAMP[1])
            if peak >= METAL_HU:
                n_metal += 1

    frac_in = n_in / n
    frac_metal = n_metal / n

    if frac_in < INBOUNDS_RED:
        verdict, reason = "red", (
            f"{n - n_in} of {n} contacts fall outside this CT — the TSV was "
            "almost certainly built against a different image.")
    elif frac_metal < ONMETAL_RED:
        verdict, reason = "red", (
            f"only {n_metal} of {n} contacts land on metal in this CT — the "
            "coordinates don't match this image.")
    elif frac_in >= MIN_INBOUNDS_OK and frac_metal >= ONMETAL_GREEN:
        verdict, reason = "green", (
            f"{n_metal} of {n} contacts land on electrode metal — the TSV "
            "matches this CT.")
    else:
        verdict, reason = "yellow", (
            f"{n_metal} of {n} contacts land on metal ({n_in}/{n} in bounds) — "
            "usable, but lower than a clean match; confirm the CT is right.")

    return {
        "n": n, "n_in_bounds": n_in, "n_on_metal": n_metal,
        "frac_in_bounds": round(frac_in, 4), "frac_on_metal": round(frac_metal, 4),
        "verdict": verdict, "reason": reason,
    }


__all__ = ["check_localization", "METAL_HU"]
