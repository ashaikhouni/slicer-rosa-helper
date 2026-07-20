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
import os
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any

import numpy as np


def _atomic_write(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (temp + os.replace) so a crash or an
    interrupted write never leaves a truncated file behind for the next reader."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)

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
    _atomic_write(job_dir / "editor_ct.i16", vol.tobytes())

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
        name=job_dir.name[:8], dims=[nx, ny, nz],
        origin=[round(float(lo[k]), 4) for k in range(3)],   # crop origin in RAS world: world = crop_index + origin
        models=models,
        trajectories=[dict(name=s["name"], name0=s["name"], model=s["model"], color=PALETTE[k % len(PALETTE)],
                           entry=idx(s["entry"]), target=idx(s["target"]),
                           entry0=idx(s["entry"]), target0=idx(s["target"]), model0=s["model"],
                           tipOffset=0.0, length_mm=s["length_mm"], confidence_label=s["confidence_label"])
                      for k, s in enumerate(shanks)],
    )
    # Volume is written first (above), then the plan LAST — the freshness check
    # keys on the plan's mtime, so the plan existing implies the volume is done.
    _atomic_write(job_dir / "editor_plan.json", json.dumps(plan).encode())
    return plan


# ---- native-resolution probe's-eye patch (server samples the FULL-res CT) ----
# The editor reslices a 1 mm crop for speed; the probe's-eye (a single-contact
# zoom, where sub-mm centre accuracy matters) is sampled from the native CT here
# so the metal is sharp. Only a small patch travels per request. The native CT +
# its RAS→voxel affine are cached (keyed by path+mtime) so it's read once.
_PROBE_CACHE: dict[str, Any] = {"key": None, "arr": None, "inv": None}


def _probe_ct(ct_path: Path):
    import numpy as np
    import nibabel as nib
    key = f"{ct_path}:{ct_path.stat().st_mtime_ns}"
    if _PROBE_CACHE["key"] != key:
        img = nib.load(str(ct_path))
        arr = np.asanyarray(img.dataobj)                      # keep native dtype (int16
        while arr.ndim > 3:                                   # → half the RAM/load of float32)
            arr = arr[..., 0]
        _PROBE_CACHE.update(key=key, arr=np.ascontiguousarray(arr),
                            inv=np.linalg.inv(img.affine.astype(float)))
    return _PROBE_CACHE["arr"], _PROBE_CACHE["inv"]


def probe_patch(ct_path, center_ras, u, v, ext_u: float, ext_v: float,
                size_u: int, size_v: int) -> bytes:
    """A ``size_v × size_u`` int16 patch of the native CT on the plane through
    ``center_ras`` spanned by RAS dirs ``u`` (columns, ``ext_u`` mm wide) and
    ``v`` (rows, ``ext_v`` mm tall). Row-major bytes, clamped to the SAME range as
    the 1 mm crop (CT_CLAMP) so the client windows it identically — only sharpness
    differs, not brightness (matters for CTs on a non-standard HU scale). Serves
    both the square probe's-eye and the rectangular in-line reformat."""
    import numpy as np
    from scipy.ndimage import map_coordinates
    arr, inv = _probe_ct(Path(ct_path))
    tu = (np.arange(size_u) - size_u / 2.0) * (float(ext_u) / size_u)   # cols → u
    tv = (np.arange(size_v) - size_v / 2.0) * (float(ext_v) / size_v)   # rows → v
    uu, vv = np.meshgrid(tu, tv, indexing="xy")               # [size_v, size_u]
    c = np.asarray(center_ras, float); U = np.asarray(u, float); V = np.asarray(v, float)
    ras = c[None, None, :] + uu[..., None] * U[None, None, :] + vv[..., None] * V[None, None, :]
    ras_h = np.concatenate([ras, np.ones((size_v, size_u, 1))], axis=-1)
    vox = ras_h @ inv.T                                       # RAS → native voxel index
    sampled = map_coordinates(arr, [vox[..., 0], vox[..., 1], vox[..., 2]],
                              order=1, mode="constant", cval=-1024.0)
    lo, hi = CT_CLAMP
    return np.clip(np.rint(sampled), lo, hi).astype("<i2").tobytes()


def ensure_cache(job_dir: str | Path) -> Path:
    """Build ``editor_plan.json`` + ``editor_ct.i16`` if missing/stale; return the job dir."""
    job_dir = Path(job_dir)
    plan_p, vol_p, src = job_dir / "editor_plan.json", job_dir / "editor_ct.i16", job_dir / "trajectories.tsv"
    if not src.is_file():
        raise FileNotFoundError(f"job {job_dir.name}: no trajectories.tsv (run a pipeline job first)")
    fresh = (plan_p.is_file() and vol_p.is_file()
             and plan_p.stat().st_mtime >= src.stat().st_mtime)
    if fresh:
        # Guard against a truncated/partial editor_ct.i16 — an interrupted build,
        # or a case copied to another machine mid-transfer. It must be exactly
        # dims_x*dims_y*dims_z * 2 bytes (int16); a mismatch (e.g. an odd length →
        # the browser's `new Int16Array` throws) forces a rebuild.
        try:
            dims = json.loads(plan_p.read_text()).get("dims") or []
            expected = int(dims[0]) * int(dims[1]) * int(dims[2]) * 2 if len(dims) == 3 else -1
        except (ValueError, OSError, TypeError, IndexError):
            expected = -1
        if expected <= 0 or vol_p.stat().st_size != expected:
            fresh = False
    if not fresh:
        _build(job_dir)
    return job_dir


__all__ = ["ensure_cache", "electrode_library"]
