"""Build the trajectory-editor payload for a pipeline job.

The editor is a client-side reslicer (no GPU, no per-interaction server round
trips): it needs a small **cropped, iso, int16 CT** around the electrodes plus
the plan geometry (entry/target/model per shank) and the canonical electrode
library. This module reads a job's ``trajectories.tsv`` / ``contacts.tsv`` / CT
and caches two artifacts next to them:

  * ``editor_ct.i16``   — cropped 1 mm-iso CT, int16 HU, x-fastest (Fortran) order
  * ``editor_plan.json``— ``{dims, trajectories, models}`` in crop-index space
                          (index == mm from the crop origin, RAS-aligned)

Both are rebuilt when the plan TSV is newer than the cache. The CT never leaves
the machine — the local app serves these to the local browser.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any

import numpy as np

CT_CLAMP = (-1024, 3071)                  # int16 HU range we ship
MARGIN_MM = 14.0
PALETTE = ["#38d2e6", "#8b7be0", "#e08b5b", "#5bd08b", "#e3b23c", "#e86f9e",
           "#6fa8e8", "#c7d24a", "#e8635f", "#4ac7b8", "#b98be0", "#d0a24a"]


def _rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _ct_path(job_dir: Path) -> Path:
    manifest = json.loads((job_dir / "manifest.json").read_text())
    ct = manifest.get("params", {}).get("ct")
    if not ct or not Path(ct).is_file():
        raise FileNotFoundError(f"job {job_dir.name}: CT not found ({ct!r})")
    return Path(ct)


def electrode_library() -> dict[str, dict]:
    """The canonical models (datasheet offsets) from rosa_core resources."""
    import rosa_core
    lib = json.loads((Path(rosa_core.__file__).parent / "resources" / "electrodes"
                      / "electrode_models.json").read_text())["models"]
    lib = {m["id"]: m for m in lib} if isinstance(lib, list) else lib
    out = {}
    for mid, m in lib.items():
        groups = m.get("groups") or [m["contact_count"]]
        uni = len(groups) == 1
        desc = f"{m.get('pitch_mm', 3.5)} mm" if uni else "grouped " + "/".join(map(str, groups))
        out[mid] = {"n": m["contact_count"], "offsets": m["contact_center_offsets_from_tip_mm"],
                    "uniform": uni, "pitch": m.get("pitch_mm", 3.5), "groups": groups,
                    "label": f"{mid} · {m['contact_count']} × {desc}"}
    return out


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n else v


def _geometry(job_dir: Path):
    """Parse trajectories + contacts into RAS geometry; return (shanks, all_pts)."""
    trajs = _rows(job_dir / "trajectories.tsv")
    contacts = _rows(job_dir / "contacts.tsv")
    pts_by, mod_by = defaultdict(list), defaultdict(list)
    for c in contacts:
        pts_by[c["trajectory"]].append([float(c["x"]), float(c["y"]), float(c["z"])])
        mod_by[c["trajectory"]].append((c.get("electrode_model") or "").strip())

    shanks, allpts = [], []
    for tr in trajs:
        name = tr["name"]
        entry = np.array([float(tr["start_x"]), float(tr["start_y"]), float(tr["start_z"])])
        target = np.array([float(tr["end_x"]), float(tr["end_y"]), float(tr["end_z"])])
        cs = np.array(pts_by.get(name, []))
        mods = [m for m in mod_by.get(name, []) if m]
        model = Counter(mods).most_common(1)[0][0] if mods else (tr.get("electrode_model") or "electrode")
        # detected offsets (fallback for a model missing from the library)
        if len(cs) >= 2:
            s = np.sort(cs @ _unit(target - entry))[::-1]
            offsets = [round(float(s[0] - x), 2) for x in s]
            n = len(cs)
        else:
            n, offsets = 15, [round(i * 3.5, 2) for i in range(15)]
        shanks.append(dict(name=name, entry=entry, target=target, model=model, offsets=offsets, n=n,
                           length_mm=float(tr.get("length_mm") or 0),
                           confidence_label=(tr.get("confidence_label") or "medium").strip() or "medium"))
        allpts += [entry, target] + list(cs)
    return shanks, np.array(allpts)


def _build(job_dir: Path) -> dict:
    shanks, allpts = _geometry(job_dir)
    lo = np.floor(allpts.min(0) - MARGIN_MM)
    hi = np.ceil(allpts.max(0) + MARGIN_MM)
    dims = (hi - lo).astype(int)
    nx, ny, nz = (int(x) for x in dims)

    import nibabel as nib
    from scipy.ndimage import map_coordinates
    img = nib.load(str(_ct_path(job_dir)))
    arr = np.asanyarray(img.dataobj).astype(np.float32)
    inv = np.linalg.inv(img.affine.astype(float))
    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    ras = np.stack([ii + lo[0], jj + lo[1], kk + lo[2], np.ones_like(ii)], -1).astype(float)
    vox = ras @ inv.T
    sampled = map_coordinates(arr, [vox[..., 0], vox[..., 1], vox[..., 2]],
                              order=1, mode="constant", cval=float(arr.min()))
    vol = np.clip(sampled, *CT_CLAMP).astype(np.int16).flatten(order="F")   # x-fastest for the JS indexer
    (job_dir / "editor_ct.i16").write_bytes(vol.tobytes())

    models = electrode_library()
    for s in shanks:                          # a case model absent from the library -> detected offsets
        if s["model"] not in models:
            sp = np.diff(s["offsets"])
            uni = len(sp) == 0 or (sp.max() - sp.min() < 0.6)
            models[s["model"]] = {"n": s["n"], "offsets": s["offsets"], "uniform": bool(uni),
                                  "pitch": round(float(np.median(sp)), 2) if len(sp) else 3.5,
                                  "groups": [s["n"]], "label": f"{s['model']} · {s['n']} (detected)"}

    def idx(p):
        return [round(float(p[k] - lo[k]), 2) for k in range(3)]
    plan = dict(
        name=job_dir.name[:8], dims=[nx, ny, nz], models=models,
        trajectories=[dict(name=s["name"], model=s["model"], color=PALETTE[k % len(PALETTE)],
                           entry=idx(s["entry"]), target=idx(s["target"]),
                           entry0=idx(s["entry"]), target0=idx(s["target"]), model0=s["model"],
                           tipOffset=0.0, length_mm=s["length_mm"], confidence_label=s["confidence_label"])
                      for k, s in enumerate(shanks)],
    )
    (job_dir / "editor_plan.json").write_text(json.dumps(plan))
    return plan


def ensure_cache(job_dir: str | Path) -> Path:
    """Build ``editor_plan.json`` + ``editor_ct.i16`` if missing/stale; return the job dir."""
    job_dir = Path(job_dir)
    plan_p, vol_p, src = job_dir / "editor_plan.json", job_dir / "editor_ct.i16", job_dir / "trajectories.tsv"
    if not src.is_file():
        raise FileNotFoundError(f"job {job_dir.name}: no trajectories.tsv (run a pipeline job first)")
    fresh = (plan_p.is_file() and vol_p.is_file()
             and plan_p.stat().st_mtime >= src.stat().st_mtime)
    if not fresh:
        _build(job_dir)
    return job_dir


__all__ = ["ensure_cache", "electrode_library"]
