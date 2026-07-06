"""Bundled MNI-space atlas registry — resolve, verify, parse LUTs, fetch.

The engine-side half of atlas-based contact labeling. This module is pure
Python (json / csv / hashlib / urllib) and holds **no** SimpleITK, nibabel,
or provider dependency, so it imports cheaply and stays usable from both the
CLI and the Slicer side. The heavy lifting — affine registration, labelmap
warping, nearest-voxel sampling — lives in
``rosa_core.registration.register_affine_mi`` and the
``LabelmapAtlasProvider`` (affine path); this module just tells those
consumers *where the files are*, *whether they are intact*, and *what each
label value means*.

Design:

* ``atlases.json`` (alongside this package's ``resources/atlases/``) is the
  manifest — the single source of truth for which atlases exist, their
  license/attribution, the template each registers through, and pinned
  sha256s. Only permissively-licensed, redistributable atlases are listed;
  non-commercial / all-rights-reserved atlases stay bring-your-own.
* The default atlas (``cerebra``) is **bundled** in-tree (CC0, ~0.4 MB) so
  the app labels whole-brain out of the box, offline. Additional atlases can
  be marked ``bundled: false`` and pulled by :func:`ensure_available` at
  install/setup time (never at clinical runtime — the shipped app makes no
  network calls).
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


def default_resource_root() -> Path:
    """Return the bundled ``resources/atlases`` directory."""
    return Path(__file__).resolve().parent / "resources" / "atlases"


@dataclass(frozen=True)
class AtlasAssets:
    """Resolved, on-disk paths + metadata for one atlas."""
    atlas_id: str
    name: str
    labelmap_path: Path
    template_path: Path
    lut_path: Path
    lut_format: str
    transform_kind: str
    space: str
    coverage: str
    license: str
    attribution: str
    license_tier: str = "permissive"   # permissive | noncommercial | fetch
    max_label_distance_mm: float | None = None


# ---------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------


def load_manifest(root: str | Path | None = None) -> dict:
    """Load and lightly validate ``atlases.json``."""
    root = Path(root) if root is not None else default_resource_root()
    manifest_path = root / "atlases.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"atlas manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("atlases", "templates", "default"):
        if key not in manifest:
            raise ValueError(f"atlases.json missing required key {key!r}")
    if manifest["default"] not in manifest["atlases"]:
        raise ValueError(
            f"default atlas {manifest['default']!r} is not in the atlases table")
    return manifest


def list_atlases(root: str | Path | None = None) -> list[dict]:
    """Return a UI-facing summary of every atlas in the manifest.

    Each entry: ``{id, name, coverage, license, space, bundled, available}``.
    ``available`` is True when the atlas's files are present on disk (bundled,
    or already fetched).
    """
    root = Path(root) if root is not None else default_resource_root()
    manifest = load_manifest(root)
    default = manifest["default"]
    out = []
    for atlas_id, entry in manifest["atlases"].items():
        out.append({
            "id": atlas_id,
            "name": entry.get("long_name") or entry.get("name") or atlas_id,
            "coverage": entry.get("coverage", ""),
            "license": entry.get("license", ""),
            "license_tier": entry.get("license_tier", "permissive"),
            "space": entry.get("space", ""),
            "bundled": bool(entry.get("bundled", False)),
            "available": _atlas_files_present(root, manifest, atlas_id),
            "is_default": atlas_id == default,
        })
    return out


def _atlas_files_present(root: Path, manifest: dict, atlas_id: str) -> bool:
    try:
        entry = manifest["atlases"][atlas_id]
        tmpl = manifest["templates"][entry["template"]]
    except KeyError:
        return False
    return all((root / p).is_file()
               for p in (entry["labelmap"], entry["lut"], tmpl["file"]))


# ---------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------


def sha256_of(path: str | Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_checksum(path: str | Path, expected: str | None) -> None:
    """Raise ``ValueError`` if ``path``'s sha256 != ``expected``.

    A ``None``/empty ``expected`` is treated as "not yet pinned" and skipped
    (with no error) — used for atlases whose checksum is pinned on first
    fetch rather than in the shipped manifest.
    """
    if not expected:
        return
    actual = sha256_of(path)
    if actual != expected:
        raise ValueError(
            f"checksum mismatch for {Path(path).name}: "
            f"expected {expected[:16]}…, got {actual[:16]}…")


# ---------------------------------------------------------------------
# LUT parsing  (value -> region name)
# ---------------------------------------------------------------------


def parse_cerebra_lut(path: str | Path) -> dict[int, str]:
    """Parse CerebrA_LabelDetails.csv into ``{label_value: region_name}``.

    CerebrA's CSV lists one row per structure with *two* label columns —
    "RH Labels" and "LH Labels" — so each structure maps two integer values
    to the same name, which we disambiguate by hemisphere:

        2002,Caudal Anterior Cingulate,30,81,...
          -> {30: "Right Caudal Anterior Cingulate",
              81: "Left Caudal Anterior Cingulate"}

    Value 0 is background/unknown. Rows for midline/vermal structures still
    carry two ids in the source (the atlas splits them L/R), so we keep the
    hemisphere prefix uniformly.
    """
    names: dict[int, str] = {0: "Unknown"}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    # Row 0 = header ("Mindboggle ID, Label Name, CerebrA ID, ...");
    # row 1 = sub-header ("...,RH Labels,LH Labels,..."). Data starts at 2.
    for row in rows[2:]:
        if len(row) < 4:
            continue
        name = (row[1] or "").strip()
        if not name:
            continue
        for col, hemi in ((2, "Right"), (3, "Left")):
            raw = (row[col] or "").strip()
            if not raw:
                continue
            try:
                value = int(float(raw))
            except ValueError:
                continue
            names[value] = f"{hemi} {name}"
    return names


def _parse_freesurfer_lut(path: str | Path) -> dict[int, str]:
    # Delegate to the canonical FreeSurfer LUT parser already in the engine.
    from rosa_core.atlas_index import parse_freesurfer_lut
    return parse_freesurfer_lut(path)


def parse_index_name_tsv(path: str | Path) -> dict[int, str]:
    """Parse a two-column ``index<TAB>name`` TSV into ``{value: name}``.

    The BIDS-style LUT the nilearn-derived atlases (Schaefer, Harvard-Oxford)
    ship with here. A header row and blank/comment lines are skipped; value 0
    is background/unknown.
    """
    names: dict[int, str] = {0: "Unknown"}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            try:
                value = int(parts[0])
            except ValueError:
                continue          # header / comment
            name = parts[1].strip()
            if name:
                names[value] = name
    return names


_LUT_PARSERS: dict[str, Callable[[str | Path], dict[int, str]]] = {
    "cerebra_csv": parse_cerebra_lut,
    "freesurfer": _parse_freesurfer_lut,
    "tsv": parse_index_name_tsv,
}


def parse_lut(path: str | Path, lut_format: str) -> dict[int, str]:
    try:
        parser = _LUT_PARSERS[lut_format]
    except KeyError as exc:
        raise ValueError(
            f"unknown lut_format {lut_format!r}; known: "
            f"{sorted(_LUT_PARSERS)}") from exc
    return parser(path)


# ---------------------------------------------------------------------
# Resolve / ensure availability
# ---------------------------------------------------------------------


def resolve(atlas_id: str | None = None,
            root: str | Path | None = None) -> AtlasAssets:
    """Resolve an atlas id to concrete on-disk asset paths + metadata.

    ``atlas_id=None`` resolves the manifest's default atlas. Raises
    ``KeyError`` for an unknown id and ``FileNotFoundError`` if any of the
    atlas's files are absent (e.g. a non-bundled atlas not yet fetched — call
    :func:`ensure_available` first).
    """
    root = Path(root) if root is not None else default_resource_root()
    manifest = load_manifest(root)
    if atlas_id is None:
        atlas_id = manifest["default"]
    if atlas_id not in manifest["atlases"]:
        raise KeyError(
            f"unknown atlas {atlas_id!r}; available: "
            f"{sorted(manifest['atlases'])}")
    entry = manifest["atlases"][atlas_id]
    tmpl = manifest["templates"][entry["template"]]

    labelmap = root / entry["labelmap"]
    template = root / tmpl["file"]
    lut = root / entry["lut"]
    for p in (labelmap, template, lut):
        if not p.is_file():
            raise FileNotFoundError(
                f"atlas {atlas_id!r} asset missing: {p} "
                f"(bundled={entry.get('bundled', False)}; "
                f"call ensure_available() to fetch)")
    return AtlasAssets(
        atlas_id=atlas_id,
        name=entry.get("long_name") or entry.get("name") or atlas_id,
        labelmap_path=labelmap,
        template_path=template,
        lut_path=lut,
        lut_format=entry.get("lut_format", "freesurfer"),
        transform_kind=entry.get("transform_kind", "affine"),
        space=entry.get("space", ""),
        coverage=entry.get("coverage", ""),
        license=entry.get("license", ""),
        attribution=entry.get("attribution", ""),
        license_tier=entry.get("license_tier", "permissive"),
        max_label_distance_mm=entry.get("max_label_distance_mm"),
    )


def ensure_available(atlas_id: str | None = None,
                     root: str | Path | None = None,
                     *,
                     allow_download: bool = False,
                     logger: Optional[Callable[[str], None]] = None) -> AtlasAssets:
    """Ensure an atlas's files are present and intact, then :func:`resolve`.

    Bundled atlases: verify pinned checksums (raises on corruption).
    Non-bundled atlases: if files are missing and ``allow_download`` is set,
    fetch labelmap + template from their manifest ``source_url`` into the
    resource tree, verifying checksums. Downloads are an install/setup-time
    action — never call this with ``allow_download=True`` on the clinical
    path of the shipped app.
    """
    root = Path(root) if root is not None else default_resource_root()
    manifest = load_manifest(root)
    if atlas_id is None:
        atlas_id = manifest["default"]
    if atlas_id not in manifest["atlases"]:
        raise KeyError(f"unknown atlas {atlas_id!r}")
    entry = manifest["atlases"][atlas_id]
    tmpl = manifest["templates"][entry["template"]]

    wants = [
        (root / entry["labelmap"], entry.get("source_url"), entry.get("labelmap_sha256")),
        (root / tmpl["file"], tmpl.get("source_url"), tmpl.get("sha256")),
    ]
    for path, url, sha in wants:
        if path.is_file():
            verify_checksum(path, sha)
            continue
        if not allow_download:
            raise FileNotFoundError(
                f"atlas {atlas_id!r} asset {path.name} absent and "
                f"allow_download=False")
        if not url:
            raise FileNotFoundError(
                f"atlas {atlas_id!r} asset {path.name} absent and has no "
                f"source_url to fetch it from")
        _download(url, path, logger=logger)
        verify_checksum(path, sha)
    return resolve(atlas_id, root)


def _download(url: str, dest: str | Path,
              *, logger: Optional[Callable[[str], None]] = None) -> None:
    import urllib.request
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if logger is not None:
        logger(f"[atlas-fetch] downloading {url} -> {dest.name}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:  # noqa: S310
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            out.write(block)
    tmp.replace(dest)


__all__ = [
    "AtlasAssets",
    "default_resource_root",
    "load_manifest",
    "list_atlases",
    "resolve",
    "ensure_available",
    "parse_lut",
    "parse_cerebra_lut",
    "sha256_of",
    "verify_checksum",
]
