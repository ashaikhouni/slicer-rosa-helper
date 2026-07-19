"""Warp a case's SEEG contacts into MNI for cohort-level pooling.

A case labeled with CerebrA (through its MRI) caches two ITK affines in its
``regcache/``:

* ``t1_to_ct.tfm``  — ``register_rigid_mi(fixed=CT, moving=T1)`` → as a POINT
  transform it maps **CT-RAS → T1-RAS**.
* ``mni_MNI152NLin2009cSym_to_t1.tfm`` — ``register_affine_mi(fixed=T1,
  moving=MNI)`` → maps **T1-RAS → MNI-RAS**.

Composing them (``B @ A``, i.e. apply the rigid CT→T1 first, then the affine
T1→MNI) warps the CT-frame contacts into ``MNI152NLin2009cSym`` — the shared
space the cohort viewer pools every subject in. The registration is **affine
only** (no deformable warp anywhere in the toolkit), so pooled positions are a
*coverage overview*, not millimetric cross-subject localization.

Pure geometry: depends only on ``rosa_core.registration`` + numpy + the stdlib
(no CLI/app coupling). The eager path (``rosa-agent cohort-export``) and the
lazy path (the ``/cohort/contacts`` endpoint) both call :func:`ensure_contacts_mni`.
"""
from __future__ import annotations

import csv
from pathlib import Path

# The shared pool space is CerebrA's; only a CerebrA-through-T1 label run caches
# the MNI leg, so only those cases are poolable (see docs/design/cohort_view.md §4.2).
POOL_SPACE = "MNI152NLin2009cSym"
CT_TO_T1_TFM = "t1_to_ct.tfm"                     # maps CT-RAS → T1-RAS
T1_TO_MNI_TFM = f"mni_{POOL_SPACE}_to_t1.tfm"     # maps T1-RAS → MNI-RAS
COLUMNS = ["trajectory", "contact_index", "name", "mni_x", "mni_y", "mni_z", "hemisphere"]


def mni_transforms_present(regcache) -> bool:
    """True when both legs of CT→T1→MNI are cached (the case is poolable)."""
    regcache = Path(regcache)
    return (regcache / CT_TO_T1_TFM).is_file() and (regcache / T1_TO_MNI_TFM).is_file()


def ct_to_mni_matrix(regcache):
    """The 4×4 RAS matrix mapping CT-frame points → MNI (``B @ A``).

    Affine-only fast path. Raises if the T1→MNI leg is a nonlinear (refined)
    transform — use :func:`warp_points_ct_to_mni` for the general case.
    """
    from rosa_core.registration import load_transform, transform_to_4x4_ras
    regcache = Path(regcache)
    a = transform_to_4x4_ras(load_transform(str(regcache / CT_TO_T1_TFM)))    # CT → T1
    b = transform_to_4x4_ras(load_transform(str(regcache / T1_TO_MNI_TFM)))   # T1 → MNI
    return b @ a                                                              # apply A first


def warp_points_ct_to_mni(points_ras, regcache):
    """Warp N×3 CT-frame RAS points → MNI, via SITK ``TransformPoint``.

    General: works whether the cached T1→MNI leg is an affine or a nonlinear
    (B-spline composite) transform. Applies CT→T1 (rigid) then T1→MNI, with
    RAS↔LPS conversion at the boundaries (ITK transforms operate in LPS)."""
    import numpy as np
    from rosa_core.registration import load_transform
    regcache = Path(regcache)
    a = load_transform(str(regcache / CT_TO_T1_TFM))     # CT → T1
    b = load_transform(str(regcache / T1_TO_MNI_TFM))    # T1 → MNI (affine or composite)
    out = []
    for p in np.asarray(points_ras, dtype=float).reshape(-1, 3):
        lps = (-float(p[0]), -float(p[1]), float(p[2]))          # RAS → LPS
        q = b.TransformPoint(a.TransformPoint(lps))              # CT → T1 → MNI (LPS)
        out.append((-q[0], -q[1], q[2]))                         # LPS → RAS
    return np.asarray(out, dtype=float)


def ensure_mni_transform(regcache, t1_path, *, refine=False, root=None, log=lambda _m: None) -> bool:
    """Register the patient T1 → MNI (2009cSym) and cache ``regcache/mni_*.tfm``.

    The T1→MNI leg the warp needs, produced exactly the way the label job does
    (``register_affine_mi(fixed=T1, moving=CerebrA template)``). With
    ``refine=True`` it adds a **B-spline nonlinear refinement** on top of the
    affine (SimpleITK, torch-free) and caches the composite instead — better
    subcortical accuracy at the cost of ~30 s. Idempotent for the affine case (a
    no-op if already cached); ``refine=True`` always (re)computes so the nonlinear
    warp is applied. Returns True once the transform is present. The rigid CT→T1
    leg (``t1_to_ct.tfm``) is produced separately (the pipeline's brain-extract).
    """
    regcache = Path(regcache)
    out = regcache / T1_TO_MNI_TFM
    if out.is_file() and not refine:
        return True
    if not t1_path or not Path(t1_path).is_file():
        return out.is_file()
    import SimpleITK as sitk
    from rosa_core import bundled_atlases
    from rosa_core.registration import register_affine_mi, register_bspline_mi, save_transform
    template = bundled_atlases.resolve("cerebra", root).template_path
    t1_img = sitk.ReadImage(str(t1_path))
    mni_img = sitk.ReadImage(str(template))
    log(f"[mni] registering T1 → {POOL_SPACE} (affine MI)…")
    aff = register_affine_mi(fixed=t1_img, moving=mni_img, logger=log)
    tx = aff.transform
    if refine:
        log("[mni] refining T1 → MNI with a B-spline (nonlinear)…")
        tx = register_bspline_mi(fixed=t1_img, moving=mni_img,
                                 initial_transform=aff.transform, logger=log)
    regcache.mkdir(parents=True, exist_ok=True)
    save_transform(tx, str(out))
    return True


def _read_delimited(path) -> list[dict]:
    """Read a TSV/CSV as dicts, skipping leading ``#`` provenance comments."""
    p = Path(path)
    if not p.is_file():
        return []
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines()
             if not ln.lstrip().startswith("#")]
    if not lines:
        return []
    delim = "\t" if "\t" in lines[0] else ","
    return list(csv.DictReader(lines, delimiter=delim))


def contacts_mni_rows(case_dir) -> list[dict]:
    """Warp ``<case>/contacts.tsv`` into MNI; return geometry rows (no region).

    Rows carry ``trajectory, contact_index, name, mni_x/y/z, hemisphere``.
    Returns ``[]`` if the case isn't poolable or has no contacts. Region /
    accepted state are joined by the caller (app-side, from review.json).
    """
    import numpy as np

    case_dir = Path(case_dir)
    regcache = case_dir / "regcache"
    contacts = case_dir / "contacts.tsv"
    if not mni_transforms_present(regcache) or not contacts.is_file():
        return []
    rows = _read_delimited(contacts)
    if not rows:
        return []
    try:
        pts = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows], dtype=float)
    except (KeyError, ValueError):
        return []
    mni = warp_points_ct_to_mni(pts, regcache)     # affine or nonlinear (refined)
    out = []
    for r, p in zip(rows, mni):
        out.append({
            "trajectory": (r.get("trajectory") or "").strip(),
            "contact_index": (r.get("contact_index") or "").strip(),
            "name": (r.get("label") or "").strip(),
            "mni_x": round(float(p[0]), 3),
            "mni_y": round(float(p[1]), 3),
            "mni_z": round(float(p[2]), 3),
            "hemisphere": "R" if p[0] > 0 else "L",     # +x = Right about the AC midline
        })
    return out


def read_tsv(path) -> list[dict]:
    """Read a cached ``contacts_mni.tsv`` back, coercing the MNI coords to float."""
    rows = _read_delimited(path)
    for r in rows:
        for k in ("mni_x", "mni_y", "mni_z"):
            try:
                r[k] = float(r[k])
            except (KeyError, ValueError, TypeError):
                pass
    return rows


def write_tsv(rows, out_path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = [f"# reference_frame: {POOL_SPACE}", "\t".join(COLUMNS)]
    for r in rows:
        body.append("\t".join(str(r.get(c, "")) for c in COLUMNS))
    out_path.write_text("\n".join(body) + "\n", encoding="utf-8")


def _cache_fresh(case_dir) -> bool:
    """True when ``regcache/contacts_mni.tsv`` is newer than every input."""
    case_dir = Path(case_dir)
    out = case_dir / "regcache" / "contacts_mni.tsv"
    if not out.is_file():
        return False
    deps = [case_dir / "contacts.tsv",
            case_dir / "regcache" / CT_TO_T1_TFM,
            case_dir / "regcache" / T1_TO_MNI_TFM]
    if not all(d.is_file() for d in deps):
        return False
    return out.stat().st_mtime >= max(d.stat().st_mtime for d in deps)


def ensure_contacts_mni(case_dir, *, force: bool = False) -> list[dict]:
    """Return the case's MNI contact rows, (re)computing + caching on stale/miss.

    The shared entry point for both the eager CLI step and the lazy endpoint:
    reads the cached ``regcache/contacts_mni.tsv`` when fresh, else recomputes
    from ``contacts.tsv`` + the tfms and writes it. Returns ``[]`` (and clears a
    stale cache) when the case is no longer poolable.
    """
    case_dir = Path(case_dir)
    out = case_dir / "regcache" / "contacts_mni.tsv"
    if not force and _cache_fresh(case_dir):
        return read_tsv(out)
    rows = contacts_mni_rows(case_dir)
    if rows:
        write_tsv(rows, out)
    elif out.is_file():
        try:
            out.unlink()
        except OSError:
            pass
    return rows


def export_case(case_dir) -> int:
    """Force-recompute + write ``contacts_mni.tsv``; return the contact count."""
    return len(ensure_contacts_mni(case_dir, force=True))


__all__ = [
    "POOL_SPACE", "COLUMNS", "mni_transforms_present", "ct_to_mni_matrix",
    "contacts_mni_rows", "read_tsv", "write_tsv", "ensure_contacts_mni", "export_case",
]
