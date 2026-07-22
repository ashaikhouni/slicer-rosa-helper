"""Label MNI-stamped contacts against every bundled MNI-native atlas.

Given a case's ``contacts_mni.tsv`` (contacts in MNI152NLin2009cSym, produced by
:mod:`rosa_core.cohort`), sample each bundled atlas that lives in the **same** MNI
space at each contact's MNI coordinate — a nearest-labeled-voxel lookup, no
registration. Writes a long-form ``contacts_labels_mni.tsv`` (one row per contact
per atlas), the substrate for "label a case against many atlases" without running
a label job per atlas.

Only atlases native to the pool space (``MNI152NLin2009cSym``) are sampled here —
today CerebrA (whole-brain) + Iglesias (thalamic nuclei). Off-variant atlases
(NLin6Asym, 2009aSym) need a one-time template-to-template transform (a later
tier) and are skipped. Pure geometry: numpy + scipy + rosa_core, no cli/app
coupling; the same nearest-labeled-voxel semantics the label job uses.
"""
from __future__ import annotations

from pathlib import Path

from . import cohort

POOL_SPACE = cohort.POOL_SPACE                      # MNI152NLin2009cSym
LABEL_COLUMNS = ["trajectory", "contact_index", "name", "atlas", "region", "distance_mm"]


def native_pool_atlases(root=None) -> list[str]:
    """Ids of bundled, available atlases native to the pool space (zero-transform)."""
    from . import bundled_atlases
    out = []
    for a in bundled_atlases.list_atlases(root):
        if a.get("space") == POOL_SPACE and a.get("available") and a.get("bundled"):
            out.append(a["id"])
    return out


class _AtlasSampler:
    """Nearest-labeled-voxel lookup of one labelmap at RAS-mm points.

    Direct voxel read first (distance 0 when the point lands in a label); a KDTree
    over labeled voxels resolves points that fall on background, gated by
    ``max_distance_mm`` (None = no gate, matching the whole-brain atlas). Built
    lazily so whole-brain atlases that hit directly never pay for the tree.
    """

    def __init__(self, labelmap_path, label_names, max_distance_mm):
        import numpy as np
        import nibabel as nib
        img = nib.load(str(labelmap_path))
        self._lab = np.asarray(img.dataobj)
        self._affine = np.asarray(img.affine, dtype=float)
        self._inv = np.linalg.inv(self._affine)
        self._names = label_names or {}
        self._max = float(max_distance_mm) if max_distance_mm is not None else None
        self._tree = None
        self._tree_labels = None

    def _build_tree(self):
        import numpy as np
        from scipy.spatial import cKDTree
        ijk = np.argwhere(self._lab > 0)
        if ijk.size == 0:
            self._tree, self._tree_labels = False, None
            return
        homog = np.c_[ijk, np.ones(len(ijk))]
        ras = (self._affine @ homog.T).T[:, :3]
        self._tree = cKDTree(ras)
        self._tree_labels = self._lab[ijk[:, 0], ijk[:, 1], ijk[:, 2]].astype(int)

    def sample(self, points_ras):
        """Return (region_name|None, distance_mm) for each Nx3 RAS point."""
        import numpy as np
        pts = np.asarray(points_ras, dtype=float).reshape(-1, 3)
        out = []
        shape = self._lab.shape
        for p in pts:
            v = self._inv @ np.array([p[0], p[1], p[2], 1.0])
            ijk = np.round(v[:3]).astype(int)
            lab = 0
            if (0 <= ijk[0] < shape[0]) and (0 <= ijk[1] < shape[1]) and (0 <= ijk[2] < shape[2]):
                lab = int(self._lab[ijk[0], ijk[1], ijk[2]])
            if lab > 0:
                out.append((self._names.get(lab, f"Label_{lab}"), 0.0))
                continue
            # background at the point → nearest labeled voxel (gated)
            if self._tree is None:
                self._build_tree()
            if not self._tree:
                out.append((None, float("inf")))
                continue
            d, idx = self._tree.query(p)
            if self._max is not None and d > self._max:
                out.append((None, float(d)))
            else:
                lab = int(self._tree_labels[idx])
                out.append((self._names.get(lab, f"Label_{lab}"), float(d)))
        return out


def label_rows(mni_rows, root=None) -> list[dict]:
    """Sample every pool-native atlas at the contacts; return long-form rows."""
    from . import bundled_atlases
    if not mni_rows:
        return []
    pts = [[float(r["mni_x"]), float(r["mni_y"]), float(r["mni_z"])] for r in mni_rows]
    out = []
    for atlas_id in native_pool_atlases(root):
        assets = bundled_atlases.resolve(atlas_id, root)
        names = bundled_atlases.parse_lut(assets.lut_path, assets.lut_format)
        sampler = _AtlasSampler(assets.labelmap_path, names, assets.max_label_distance_mm)
        for r, (region, dist) in zip(mni_rows, sampler.sample(pts)):
            if region is None:
                continue                       # outside this atlas's coverage
            out.append({
                "trajectory": r.get("trajectory", ""),
                "contact_index": r.get("contact_index", ""),
                "name": r.get("name", ""),
                "atlas": atlas_id,
                "region": region,
                "distance_mm": round(float(dist), 2),
            })
    return out


def _labels_path(case_dir) -> Path:
    return Path(case_dir) / "regcache" / "contacts_labels_mni.tsv"


def write_tsv(rows, out_path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = [f"# space: {POOL_SPACE}", "\t".join(LABEL_COLUMNS)]
    for r in rows:
        body.append("\t".join(str(r.get(c, "")) for c in LABEL_COLUMNS))
    out_path.write_text("\n".join(body) + "\n", encoding="utf-8")


def read_tsv(path) -> list[dict]:
    rows = cohort._read_delimited(path)
    for r in rows:
        try:
            r["distance_mm"] = float(r["distance_mm"])
        except (KeyError, ValueError, TypeError):
            pass
    return rows


def _cache_fresh(case_dir) -> bool:
    out = _labels_path(case_dir)
    mni = Path(case_dir) / "regcache" / "contacts_mni.tsv"
    return out.is_file() and mni.is_file() and out.stat().st_mtime >= mni.stat().st_mtime


def ensure_contacts_labels_mni(case_dir, *, force=False, root=None) -> list[dict]:
    """Return the case's per-atlas MNI labels, (re)computing + caching on stale.

    Chains off :func:`cohort.ensure_contacts_mni` (so the MNI coords are fresh
    first), then samples the pool-native atlases. Returns ``[]`` (and clears the
    cache) when the case isn't MNI-poolable or has no contacts.
    """
    case_dir = Path(case_dir)
    out = _labels_path(case_dir)
    mni_rows = cohort.ensure_contacts_mni(case_dir)          # refreshes contacts_mni.tsv
    if not mni_rows:
        if out.is_file():
            try:
                out.unlink()
            except OSError:
                pass
        return []
    if not force and _cache_fresh(case_dir):
        return read_tsv(out)
    rows = label_rows(mni_rows, root)
    write_tsv(rows, out)
    return rows


def stamp_case(case_dir, root=None) -> int:
    """Force-recompute contacts_mni.tsv + contacts_labels_mni.tsv; return label count."""
    cohort.ensure_contacts_mni(case_dir, force=True)
    return len(ensure_contacts_labels_mni(case_dir, force=True, root=root))


__all__ = [
    "POOL_SPACE", "LABEL_COLUMNS", "native_pool_atlases", "label_rows",
    "read_tsv", "write_tsv", "ensure_contacts_labels_mni", "stamp_case",
]
