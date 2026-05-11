"""Pure-Python FreeSurfer surface + annotation reader for the CLI agent.

Reads what ``export-view`` needs from a ``recon-all`` subject directory:

* surface geometry (``surf/lh.pial``, ``surf/rh.pial``, etc.) — vertices
  in FreeSurfer *surface RAS* (a.k.a. tkrRAS), faces as triangle indices.
* annotation labels (``label/lh.aparc.annot``, ``rh.aparc.annot``) —
  per-vertex int label + the embedded colortable.
* the ``mri/T1.mgz`` base volume — exposes both the vox→scanner-RAS and
  vox→tkrRAS matrices so we can convert surface vertices from tkrRAS to
  scanner RAS (the frame downstream registration uses).

No Slicer / VTK / Qt imports — sits alongside the other ``cli/rosa_agent/io``
modules so the headless agent stays headless.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------
# Dataclasses returned to callers
# ---------------------------------------------------------------------


@dataclass
class FSSurface:
    """One FreeSurfer surface, normalized into scanner RAS."""

    name: str                  # e.g. "lh.pial"
    hemi: str                  # "lh" | "rh"
    kind: str                  # "pial" | "white" | "inflated" | ...
    vertices_ras: np.ndarray   # (N, 3) float32, scanner RAS mm
    faces: np.ndarray          # (M, 3) int32
    vertex_normals: np.ndarray | None = None     # (N, 3) float32 unit normals
    vertex_labels: np.ndarray | None = None      # (N,) int32 or None
    vertex_colors_rgba: np.ndarray | None = None # (N, 4) uint8 or None
    annotation_name: str | None = None           # e.g. "aparc.annot"


@dataclass
class FSSubject:
    """Resolved FreeSurfer subject directory + its key files."""

    subject_dir: Path
    surf_dir: Path
    mri_dir: Path
    label_dir: Path | None
    t1_path: Path                              # mri/T1.mgz (or orig.mgz fallback)
    vox_to_scanner_ras: np.ndarray             # 4x4 (Norig)
    vox_to_tkr_ras: np.ndarray                 # 4x4 (Torig)
    tkr_to_scanner_ras: np.ndarray             # Norig @ inv(Torig)
    available_surfaces: list[str] = field(default_factory=list)
    available_annotations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------


_KNOWN_SURFACES = (
    "lh.pial", "rh.pial",
    "lh.white", "rh.white",
    "lh.inflated", "rh.inflated",
    "lh.smoothwm", "rh.smoothwm",
)


def _resolve_t1_path(mri_dir: Path) -> Path:
    """Prefer T1.mgz; fall back to orig.mgz / brain.mgz when missing."""
    for name in ("T1.mgz", "orig.mgz", "brain.mgz", "rawavg.mgz"):
        p = mri_dir / name
        if p.is_file():
            return p
        # Some recon-alls keep orig/ as a subdir of mri/.
        p_alt = mri_dir / "orig" / name
        if p_alt.is_file():
            return p_alt
    raise FileNotFoundError(
        f"FreeSurfer base volume not found under {mri_dir}; expected one of "
        f"T1.mgz / orig.mgz / brain.mgz / rawavg.mgz"
    )


def resolve_fs_subject(subject_dir: str | Path) -> FSSubject:
    """Resolve a FreeSurfer recon-all subject directory.

    Accepts either the subject root (``Recon/``) or any of its standard
    children (``surf/``, ``mri/``, ``label/``) — same lookup the Slicer
    side does, just file-system level.
    """
    import nibabel as nib

    root = Path(subject_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"FreeSurfer path not found: {subject_dir}")

    base = root.name
    if base in ("surf", "label", "mri"):
        root = root.parent

    surf_dir = root / "surf"
    mri_dir = root / "mri"
    label_dir = root / "label"

    if not surf_dir.is_dir():
        raise FileNotFoundError(f"No surf/ directory under {root}")
    if not mri_dir.is_dir():
        raise FileNotFoundError(f"No mri/ directory under {root}")

    t1_path = _resolve_t1_path(mri_dir)
    mgh = nib.load(str(t1_path))
    # nibabel.MGHImage exposes both matrices we need:
    vox_to_scanner_ras = np.asarray(mgh.header.get_affine(), dtype=float)
    vox_to_tkr_ras = np.asarray(mgh.header.get_vox2ras_tkr(), dtype=float)
    tkr_to_scanner_ras = vox_to_scanner_ras @ np.linalg.inv(vox_to_tkr_ras)

    available_surfaces = [
        name for name in _KNOWN_SURFACES if (surf_dir / name).is_file()
    ]
    available_annotations: list[str] = []
    if label_dir.is_dir():
        for entry in sorted(label_dir.iterdir()):
            if entry.suffix == ".annot" and entry.is_file():
                available_annotations.append(entry.name)

    return FSSubject(
        subject_dir=root,
        surf_dir=surf_dir,
        mri_dir=mri_dir,
        label_dir=label_dir if label_dir.is_dir() else None,
        t1_path=t1_path,
        vox_to_scanner_ras=vox_to_scanner_ras,
        vox_to_tkr_ras=vox_to_tkr_ras,
        tkr_to_scanner_ras=tkr_to_scanner_ras,
        available_surfaces=available_surfaces,
        available_annotations=available_annotations,
    )


# ---------------------------------------------------------------------
# Surface + annotation loading
# ---------------------------------------------------------------------


def _annot_path_for_surface(
    surface_name: str, annotation_name: str, label_dir: Path | None,
) -> Path | None:
    """Resolve the .annot file matching one surface + an annotation key.

    ``annotation_name`` may be:
      - a bare key (``aparc``, ``aparc.a2009s``, ``aparc.DKTatlas``)
      - a full filename (``lh.aparc.annot``)
      - an absolute path
    """
    if not annotation_name or label_dir is None:
        return None
    name = annotation_name.strip()
    if not name:
        return None
    if Path(name).is_absolute() and Path(name).is_file():
        return Path(name)
    hemi = surface_name.split(".", 1)[0].lower()
    if hemi not in ("lh", "rh"):
        return None
    if name.startswith("lh.") or name.startswith("rh."):
        # Caller passed an explicit hemi prefix; swap to match this surface.
        rest = name.split(".", 1)[1]
        candidate = label_dir / f"{hemi}.{rest}"
    elif name.endswith(".annot"):
        candidate = label_dir / f"{hemi}.{name}"
    else:
        candidate = label_dir / f"{hemi}.{name}.annot"
    return candidate if candidate.is_file() else None


def load_fs_surfaces(
    subject: FSSubject,
    *,
    surface_kinds: tuple[str, ...] = ("pial",),
    annotation: str | None = None,
    transform_4x4: np.ndarray | None = None,
) -> list[FSSurface]:
    """Load FreeSurfer surfaces, push to scanner RAS, optionally tag with annot.

    Parameters
    ----------
    surface_kinds : tuple
        Subset of ``("pial", "white", "inflated", "smoothwm")``. Each
        emits both hemispheres when files exist; missing ones are silently
        skipped.
    annotation : str | None
        Annotation key (e.g. ``"aparc"``). When present, every loaded
        surface gets a vertex-label + per-vertex RGBA color array.
    transform_4x4 : np.ndarray | None
        Optional additional RAS→RAS 4×4 to compose on top of the
        tkrRAS→scannerRAS conversion (e.g. an FS→CT rigid registration).
    """
    import nibabel as nib

    out: list[FSSurface] = []
    extra = np.eye(4) if transform_4x4 is None else np.asarray(transform_4x4, dtype=float)
    combined = extra @ subject.tkr_to_scanner_ras

    for kind in surface_kinds:
        for hemi in ("lh", "rh"):
            name = f"{hemi}.{kind}"
            path = subject.surf_dir / name
            if not path.is_file():
                continue
            verts_tkr, faces = nib.freesurfer.io.read_geometry(str(path))
            verts_tkr = np.asarray(verts_tkr, dtype=float)
            faces = np.asarray(faces, dtype=np.int32)
            h = np.ones((verts_tkr.shape[0], 4), dtype=float)
            h[:, :3] = verts_tkr
            verts_ras = (combined @ h.T).T[:, :3].astype(np.float32)

            # Smooth (area-weighted) per-vertex normals. Without these the
            # glTF receivers fall back to flat-shading the triangles,
            # which makes a folded cortical surface look like a chunky
            # mosaic in any viewer (the v2 GLB had this look).
            v0 = verts_ras[faces[:, 0]]
            v1 = verts_ras[faces[:, 1]]
            v2 = verts_ras[faces[:, 2]]
            face_normals = np.cross(v1 - v0, v2 - v0).astype(np.float64)
            vertex_normals = np.zeros_like(verts_ras, dtype=np.float64)
            np.add.at(vertex_normals, faces[:, 0], face_normals)
            np.add.at(vertex_normals, faces[:, 1], face_normals)
            np.add.at(vertex_normals, faces[:, 2], face_normals)
            n_norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
            n_norms = np.where(n_norms < 1e-12, 1.0, n_norms)
            normals_ras = (vertex_normals / n_norms).astype(np.float32)

            labels = colors = None
            annot_used = None
            if annotation:
                annot_path = _annot_path_for_surface(
                    name, annotation, subject.label_dir,
                )
                if annot_path is not None:
                    labels_raw, ctab, label_names = nib.freesurfer.io.read_annot(
                        str(annot_path), orig_ids=False,
                    )
                    labels = np.asarray(labels_raw, dtype=np.int32)
                    ctab = np.asarray(ctab)
                    if labels.shape[0] != verts_ras.shape[0]:
                        # Refuse to ship a vertex-mismatched coloring — would
                        # paint nonsense onto the surface. The annotation
                        # column simply stays None.
                        labels = None
                    elif ctab.size and ctab.shape[1] >= 4:
                        # FreeSurfer ctab column 3 is "transparency" (0 = opaque,
                        # 255 = fully transparent — opposite of GLB alpha).
                        rgba = np.full((labels.shape[0], 4), 200, dtype=np.uint8)
                        for idx in range(labels.shape[0]):
                            row = int(labels[idx])
                            if 0 <= row < ctab.shape[0]:
                                rgba[idx, 0] = int(ctab[row, 0])
                                rgba[idx, 1] = int(ctab[row, 1])
                                rgba[idx, 2] = int(ctab[row, 2])
                                rgba[idx, 3] = max(0, 255 - int(ctab[row, 3]))
                        colors = rgba
                        annot_used = annot_path.name
                    # Suppress label_names: not needed for vertex coloring; the
                    # caller can re-open the .annot if it wants the legend.

            out.append(FSSurface(
                name=name, hemi=hemi, kind=kind,
                vertices_ras=verts_ras, faces=faces,
                vertex_normals=normals_ras,
                vertex_labels=labels, vertex_colors_rgba=colors,
                annotation_name=annot_used,
            ))
    return out


# ---------------------------------------------------------------------
# LUT loading (lazy import — only used by callers that want labels)
# ---------------------------------------------------------------------


def parse_freesurfer_lut(path: str | Path) -> dict[int, dict[str, Any]]:
    """Parse a FreeSurferColorLUT.txt into ``{label_value: {...}}``.

    Each entry has ``name``, ``rgba`` (4-tuple uint8). Comments and blank
    lines are skipped. Returns ``{}`` when the file is missing — callers
    decide whether that's fatal.
    """
    p = Path(path).expanduser()
    if not p.is_file():
        return {}
    out: dict[int, dict[str, Any]] = {}
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                value = int(parts[0])
                r, g, b, a = (int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5]))
            except ValueError:
                continue
            out[value] = {"name": parts[1], "rgba": (r, g, b, a)}
    return out
