"""Brain surface mesh from a binary mask — pure imaging, no FreeSurfer.

Generates a smooth triangle surface (in RAS mm — the contact frame) from an
intracranial brain mask via marching cubes, for use as anatomical *context* in
the 3D viewer. This is deliberately a smoothed context hull, **not** an
anatomically-exact pial surface: a clean brain surface is all the viewer needs,
and it comes essentially free from the mask the pipeline already produces
(SynthStrip / log-watershed / hull).

No FreeSurfer, no trimesh — only ``numpy`` + ``scipy.ndimage`` +
``skimage.measure`` + ``nibabel``, all already dependencies. Vertices are
returned in **RAS mm**, so a CT-space mask drops straight into the same GLB
scene frame as the contacts and trajectories (its own vox→RAS affine maps to
the contact frame — no registration needed, unlike a FreeSurfer surface).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class BrainSurface:
    """A triangle surface in RAS mm, ready for ``GLBScene.add_surface``."""

    vertices_ras: np.ndarray    # (N, 3) float32, RAS mm
    faces: np.ndarray           # (M, 3) int32
    vertex_normals: np.ndarray  # (N, 3) float32, unit

    @property
    def n_vertices(self) -> int:
        return int(self.vertices_ras.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])


def _load_mask(mask: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(binary_ijk_bool, vox_to_ras_affine_4x4)``.

    Accepts a path or a nibabel image. nibabel presents the affine in RAS+
    regardless of on-disk storage, so ``apply_affine`` yields RAS mm directly.
    """
    import nibabel as nib

    if isinstance(mask, (str, Path)):
        img = nib.load(str(mask))
    elif hasattr(mask, "affine") and hasattr(mask, "dataobj"):
        img = mask
    else:
        raise TypeError("mask must be a filesystem path or a nibabel image")
    arr = np.asanyarray(img.dataobj)
    affine = np.asarray(img.affine, dtype=float)
    return arr > 0, affine


def _largest_component(binary: np.ndarray) -> np.ndarray:
    """Keep only the largest 3D connected component — drops mask speckle."""
    from scipy import ndimage

    lbl, n = ndimage.label(binary)
    if n <= 1:
        return binary
    counts = np.bincount(lbl.ravel())
    counts[0] = 0  # ignore background
    return lbl == int(counts.argmax())


def _taubin_smooth(
    verts: np.ndarray, faces: np.ndarray,
    *, iterations: int, lamb: float = 0.5, mu: float = -0.53,
) -> np.ndarray:
    """Taubin (λ|μ) Laplacian smoothing — smooths without net shrinkage.

    Pure numpy: neighbour sums via ``bincount`` over the (symmetric) edge list.
    Plain Laplacian shrinks the brain inward each pass; the negative μ step
    inflates it back, so the surface de-staircases without collapsing.
    """
    n = verts.shape[0]
    und = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.vstack([und, und[:, ::-1]])  # both directions
    rows, cols = edges[:, 0], edges[:, 1]
    deg = np.bincount(rows, minlength=n).astype(np.float64)
    deg[deg == 0] = 1.0
    v = verts.astype(np.float64)
    for it in range(iterations):
        factor = lamb if (it % 2 == 0) else mu
        nbr = np.empty_like(v)
        for k in range(3):
            nbr[:, k] = np.bincount(rows, weights=v[cols, k], minlength=n)
        v = v + factor * (nbr / deg[:, None] - v)
    return v.astype(np.float32)


def _vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals, computed in RAS (frame-correct)."""
    v = verts.astype(np.float64)
    tris = v[faces]
    face_n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, faces[:, k], face_n)
    norms = np.linalg.norm(vn, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vn / norms).astype(np.float32)


def gyral_mask_from_mri(volume: Any, brain_mask: Any):
    """Gray+white-matter mask (drops sulcal/ventricular CSF) from a T1 + brain mask.

    Otsu-splits the in-brain T1 into CSF/GM/WM and keeps **GM+WM**, so meshing
    this mask dips the surface into the sulci → a folded (pial-ish) surface,
    unlike the filled brain mask's smooth envelope. Cleaner than a raw intensity
    threshold — Otsu adapts to the histogram, ventricles are filled, and
    ``surface_from_mask`` keeps the largest component. No new dependency
    (skimage/scipy). Returns a nibabel image ready for :func:`surface_from_mask`
    (use a low ``smooth_sigma`` there so the folds survive).

    This is the bundleable "gyri without FreeSurfer" path; a real recon
    (FreeSurfer/FastSurfer pial) is cleaner when available.
    """
    import nibabel as nib
    from scipy import ndimage
    from skimage.filters import threshold_multiotsu

    vimg = volume if hasattr(volume, "dataobj") else nib.load(str(volume))
    mimg = brain_mask if hasattr(brain_mask, "dataobj") else nib.load(str(brain_mask))
    t1 = np.asanyarray(vimg.dataobj).astype(np.float32)
    m = np.asanyarray(mimg.dataobj) > 0
    inb = t1[m]
    if inb.size == 0:
        raise ValueError("empty brain mask")
    th = threshold_multiotsu(inb, classes=3)     # [CSF|GM, GM|WM]
    gmwm = (t1 >= float(th[0])) & m               # drop the darkest (CSF) class
    gmwm = ndimage.binary_closing(gmwm, iterations=1)
    gmwm = ndimage.binary_fill_holes(gmwm)        # fill ventricles → only the pial surface
    return nib.Nifti1Image(gmwm.astype(np.uint8), vimg.affine, vimg.header)


def surface_from_mask(
    mask: Any,
    *,
    smooth_sigma: float = 1.2,
    level: float = 0.5,
    step_size: int = 2,
    taubin_iterations: int = 12,
    largest_component: bool = True,
) -> BrainSurface:
    """Build a smooth brain-surface mesh (RAS mm) from a binary mask.

    Args:
        mask: path or nibabel image of a binary brain mask, in the frame the
            contacts live in (a CT-space mask → the contact frame directly).
        smooth_sigma: Gaussian blur (voxels) applied to the binary mask before
            marching cubes — the primary de-staircasing knob (0 to disable).
        level: iso-level for marching cubes on the smoothed mask (0.5 = surface).
        step_size: marching-cubes stride; >1 coarsens the mesh (fewer triangles,
            faster) — the main triangle-budget control without a decimator.
        taubin_iterations: no-shrink mesh smoothing passes (0 to disable).
        largest_component: keep only the largest connected component (drop
            speckle islands from mask noise).

    Returns:
        A :class:`BrainSurface` (vertices_ras, faces, vertex_normals).

    Raises:
        ValueError: the mask is empty (no surface to extract).
    """
    from scipy import ndimage
    try:
        from skimage import measure
    except ImportError as exc:  # scikit-image is the optional [mesh] extra
        raise ImportError(
            "brain_mesh needs scikit-image — install with: "
            "pip install 'rosa-agent[mesh]'"
        ) from exc

    binary_ijk, affine = _load_mask(mask)
    if not binary_ijk.any():
        raise ValueError("brain mask is empty — no surface to extract")

    vol = binary_ijk
    if largest_component:
        vol = _largest_component(vol)
    vol = vol.astype(np.float32)
    if smooth_sigma and smooth_sigma > 0:
        vol = ndimage.gaussian_filter(vol, sigma=float(smooth_sigma))

    verts_ijk, faces, _n, _v = measure.marching_cubes(
        vol, level=float(level), step_size=int(max(1, step_size)),
    )
    faces = np.ascontiguousarray(faces, dtype=np.int32)

    import nibabel as nib
    verts_ras = nib.affines.apply_affine(affine, verts_ijk).astype(np.float32)

    if taubin_iterations and taubin_iterations > 0:
        verts_ras = _taubin_smooth(verts_ras, faces, iterations=int(taubin_iterations))

    normals = _vertex_normals(verts_ras, faces)
    return BrainSurface(
        vertices_ras=np.ascontiguousarray(verts_ras, dtype=np.float32),
        faces=faces,
        vertex_normals=normals,
    )


__all__ = ["BrainSurface", "surface_from_mask"]
