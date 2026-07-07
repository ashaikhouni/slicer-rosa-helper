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


def _largest_mesh_component(verts: np.ndarray, faces: np.ndarray):
    """Keep the largest connected component of a triangle mesh (by vertex count).

    A grayscale iso-surface over the whole brain yields the outer pial surface
    PLUS separate closed shells for internal isocontours (ventricle walls). Those
    are disconnected components; the pial surface is overwhelmingly the largest,
    so keeping it drops the internal shells. Union-find over the edge list.
    """
    n = int(verts.shape[0])
    parent = np.arange(n)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for tri in faces:
        a = find(int(tri[0])); b = find(int(tri[1])); c = find(int(tri[2]))
        parent[b] = a
        parent[c] = a
    roots = np.array([find(i) for i in range(n)])
    keep_root = int(np.bincount(roots).argmax())
    vmask = roots == keep_root
    remap = -np.ones(n, dtype=np.int64)
    remap[vmask] = np.arange(int(vmask.sum()))
    keep_faces = faces[vmask[faces[:, 0]] & vmask[faces[:, 1]] & vmask[faces[:, 2]]]
    return verts[vmask], remap[keep_faces].astype(np.int32)


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


def transform_surface(surface: "BrainSurface", ras_4x4: np.ndarray) -> "BrainSurface":
    """Return a copy of ``surface`` with vertices mapped through a 4×4 RAS
    transform (normals recomputed in the new frame).

    Used to push a surface meshed in the native MRI frame into the CT/contact
    frame via the MRI→CT rigid transform — the same trick FreeSurfer surfaces
    use. Normals are recomputed rather than rotated so the call is correct for
    any affine (rigid included).
    """
    v = np.asarray(surface.vertices_ras, dtype=float)
    h = np.ones((v.shape[0], 4), dtype=float)
    h[:, :3] = v
    moved = (np.asarray(ras_4x4, dtype=float) @ h.T).T[:, :3].astype(np.float32)
    return BrainSurface(
        vertices_ras=np.ascontiguousarray(moved, dtype=np.float32),
        faces=surface.faces,
        vertex_normals=_vertex_normals(moved, surface.faces),
    )


def _n4_bias_correct(t1: np.ndarray, m: np.ndarray, *, shrink: int = 4) -> np.ndarray:
    """N4 bias-field correction of an in-brain T1 (SITK). Returns raw t1 on any
    failure (warns) so the caller degrades gracefully.

    MRI RF inhomogeneity makes WM/GM dimmer in some regions, so a *global* Otsu
    threshold separates tissue inconsistently — gyri stay crisp where the
    intensity is nominal and smear where it's dim. N4 flattens the field, so the
    threshold behaves the same everywhere → uniform gyral resolution. Estimated
    on a shrunk volume (fast: ~1 s) then applied at full resolution.
    """
    try:
        import SimpleITK as sitk
    except ImportError:
        return t1
    try:
        img = sitk.Cast(sitk.GetImageFromArray(t1), sitk.sitkFloat32)
        msk = sitk.GetImageFromArray(m.astype(np.uint8))
        img_s = sitk.Shrink(img, [int(shrink)] * 3)
        msk_s = sitk.Shrink(msk, [int(shrink)] * 3)
        n4 = sitk.N4BiasFieldCorrectionImageFilter()
        n4.SetMaximumNumberOfIterations([50, 50, 50, 30])
        n4.Execute(img_s, msk_s)                      # estimate on the shrunk grid
        bias = n4.GetLogBiasFieldAsImage(img)         # evaluate at full resolution
        return sitk.GetArrayFromImage(img / sitk.Exp(bias)).astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        import warnings
        warnings.warn(f"N4 bias correction failed ({exc}); using raw intensities")
        return t1


def gyral_mask_from_mri(volume: Any, brain_mask: Any, *,
                        open_iterations: int = 1, bias_correct: bool = True):
    """Gray+white-matter mask (drops sulcal/ventricular CSF) from a T1 + brain mask.

    Otsu-splits the in-brain T1 into CSF/GM/WM and keeps **GM+WM**, so meshing
    this mask dips the surface into the sulci → a folded (pial-ish) surface,
    unlike the filled brain mask's smooth envelope. Cleaner than a raw intensity
    threshold — Otsu adapts to the histogram, ventricles are filled, and
    ``surface_from_mask`` keeps the largest component. No new dependency
    (skimage/scipy). Returns a nibabel image ready for :func:`surface_from_mask`
    (use a low ``smooth_sigma`` there so the folds survive).

    ``open_iterations`` binary-opens the GM+WM mask before keeping the largest
    component: opening erodes then dilates, so it strips the small bright
    inclusions the threshold pulls in near the cortex (vessels, dura,
    partial-volume voxels) **and** the thin bridges that attach them — otherwise
    they survive as blisters sitting on the gyri (mesh smoothing only rounds
    them, it can't remove attached geometry). Higher = cleaner but eats thin
    gyri; 0 disables.

    ``bias_correct`` N4-corrects the T1 first so the Otsu threshold gives even
    gyral resolution across the whole brain (see :func:`_n4_bias_correct`).

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
    if m.sum() == 0:
        raise ValueError("empty brain mask")
    if bias_correct:
        # Flatten RF inhomogeneity so a single Otsu threshold gives uniform
        # gyral resolution across the whole brain (not crisp here, mushy there).
        t1 = _n4_bias_correct(t1, m)
    inb = t1[m]
    th = threshold_multiotsu(inb, classes=3)     # [CSF|GM, GM|WM]
    gmwm = (t1 >= float(th[0])) & m               # drop the darkest (CSF) class
    if open_iterations and open_iterations > 0:
        # Strip surface nodules + the thin bridges attaching them, then keep the
        # largest component so the detached blobs drop out (not just rounded).
        gmwm = ndimage.binary_opening(gmwm, iterations=int(open_iterations))
        gmwm = _largest_component(gmwm)
    gmwm = ndimage.binary_closing(gmwm, iterations=1)
    gmwm = ndimage.binary_fill_holes(gmwm)        # fill ventricles → only the pial surface
    return nib.Nifti1Image(gmwm.astype(np.uint8), vimg.affine, vimg.header)


def brain_tissue_from_fastsurfer_aseg(aseg: Any, reference: Any, *,
                                      drop_cerebellum: bool = False) -> np.ndarray:
    """Binary brain-tissue mask (cortex + WM + subcortical GM) from a FastSurfer /
    FreeSurfer ``aparc+aseg`` segmentation, on the ``reference`` volume's grid.

    Drops background, CSF and ventricle labels — so the resulting surface follows
    the *learned* cortical boundary and never includes dura/vessels (they are
    simply unlabelled). ``drop_cerebellum`` additionally removes cerebellum +
    brainstem for a cerebrum-only surface (FastSurfer labels them separately, so
    this is clean — unlike a threshold, which can't). Nearest-neighbour resampled
    onto the reference grid when they differ.
    """
    import nibabel as nib
    import nibabel.processing

    aimg = aseg if hasattr(aseg, "dataobj") else nib.load(str(aseg))
    rimg = reference if hasattr(reference, "dataobj") else nib.load(str(reference))
    if aimg.shape != rimg.shape or not np.allclose(aimg.affine, rimg.affine, atol=1e-3):
        aimg = nib.processing.resample_from_to(aimg, (rimg.shape, rimg.affine), order=0)
    lab = np.asanyarray(aimg.dataobj).astype(np.int32)
    # FreeSurfer LUT: background/CSF/ventricles/choroid/vessels.
    exclude = {0, 24, 4, 43, 5, 44, 14, 15, 72, 31, 63, 30, 62}
    if drop_cerebellum:
        exclude |= {7, 8, 46, 47, 16}   # cerebellum WM/cortex (L/R) + brainstem
    return np.isin(lab, list(exclude), invert=True) & (lab > 0)


def gyral_surface_from_mri(
    volume: Any, brain_mask: Any, *,
    step_size: int = 1,
    smooth_sigma: float = 0.4,
    taubin_iterations: int = 3,
    open_iterations: int = 1,
    nodule_size: int = 3,
    mask_erode: int = 1,
    bias_correct: bool = True,
    brain_tissue: "np.ndarray | None" = None,
) -> "BrainSurface":
    """Crisp gyral (pial-ish) surface from a T1, without FreeSurfer.

    Meshes the **grayscale intensity iso-surface** at the Otsu CSF/GM threshold
    rather than a binary GM+WM mask: the isocontour is sub-voxel and follows the
    true tissue boundary, so gyri resolve far better than a staircased binary
    mesh (whose blur also fills thin sulci). ``step_size=1`` samples every voxel
    for maximum fold detail; ``2`` is ~4× lighter.

    Pipeline: N4 bias-correct (even resolution) → grayscale-open (``nodule_size``,
    flattens bright vessel/dura specks so they don't bulge the surface) → Otsu →
    build a *cleaned* GM+WM
    support (opening strips nodules, largest component, fill) → mesh the T1
    isocontour restricted to that support (so removed nodules can't reappear) →
    keep the largest mesh component (drops ventricle-wall shells) → Taubin. Verts
    are returned in the MRI's RAS; the caller transforms them into CT.

    ``brain_tissue`` (optional) supplies the GM+WM support from an external
    segmentation — e.g. FastSurfer's ``aparc+aseg`` via
    :func:`brain_tissue_from_fastsurfer_aseg`. When given, the Otsu/opening/erode
    support-building is skipped and the iso-surface is meshed inside that
    (learned) region, so dura never enters. When ``None``, the Otsu path is the
    zero-dependency fallback. A real recon pial is still the accuracy tier.
    """
    import nibabel as nib
    from scipy import ndimage
    from skimage import measure
    from skimage.filters import threshold_multiotsu

    vimg = volume if hasattr(volume, "dataobj") else nib.load(str(volume))
    mimg = brain_mask if hasattr(brain_mask, "dataobj") else nib.load(str(brain_mask))
    t1 = np.asanyarray(vimg.dataobj).astype(np.float32)
    m = np.asanyarray(mimg.dataobj) > 0
    if m.sum() == 0:
        raise ValueError("empty brain mask")
    if bias_correct:
        t1 = _n4_bias_correct(t1, m)
    if nodule_size and nodule_size > 0:
        # Grayscale opening flattens bright blobs smaller than the element
        # (vessels/dura specks) while preserving the wider cortical ribbon, so
        # the iso-surface doesn't bulge out into nodules on the gyri.
        t1 = ndimage.grey_opening(t1, size=int(nodule_size))
    th = threshold_multiotsu(t1[m], classes=3)     # [CSF|GM, GM|WM]
    if brain_tissue is not None:
        # External (learned) GM+WM support — FastSurfer/FreeSurfer aseg. Dilate
        # by 1 so the pial iso-contour sits just inside it; no Otsu/opening/erode.
        gmwm = _largest_component(np.asarray(brain_tissue).astype(bool) & m)
        gmwm = ndimage.binary_fill_holes(gmwm)
        support = ndimage.binary_dilation(gmwm, iterations=1)
    else:
        # Erode the brain mask before meshing: the dura / pial vessels the strip
        # leaves live in the outermost 1–2 mm rim, and post-mesh smoothing can't
        # remove them selectively (it blurs the gyri too). Eroding the rim drops
        # that material at the source while the fold structure (defined deeper, at
        # the GM/WM interface) is preserved. 0 disables.
        m_surf = ndimage.binary_erosion(m, iterations=int(mask_erode)) if (mask_erode and mask_erode > 0) else m
        # Cleaned GM+WM support: opening strips bright nodules (vessels/dura) + the
        # thin bridges attaching them; dilate so the pial isocontour is inside it.
        gmwm = (t1 >= float(th[0])) & m_surf
        if open_iterations and open_iterations > 0:
            gmwm = _largest_component(ndimage.binary_opening(gmwm, iterations=int(open_iterations)))
        gmwm = ndimage.binary_fill_holes(ndimage.binary_closing(gmwm, iterations=1))
        support = ndimage.binary_dilation(gmwm, iterations=2)
    # Grayscale iso-surface of the T1 at the CSF/GM threshold, within the support.
    vol = np.where(support, t1, 0.0).astype(np.float32)
    if smooth_sigma and smooth_sigma > 0:
        vol = ndimage.gaussian_filter(vol, sigma=float(smooth_sigma))
    verts_ijk, faces, _n, _v = measure.marching_cubes(
        vol, level=float(th[0]), step_size=int(max(1, step_size)))
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    verts_ijk, faces = _largest_mesh_component(verts_ijk, faces)
    verts_ras = nib.affines.apply_affine(
        np.asarray(vimg.affine, dtype=float), verts_ijk).astype(np.float32)
    if taubin_iterations and taubin_iterations > 0:
        verts_ras = _taubin_smooth(verts_ras, faces, iterations=int(taubin_iterations))
    return BrainSurface(
        vertices_ras=np.ascontiguousarray(verts_ras, dtype=np.float32),
        faces=faces,
        vertex_normals=_vertex_normals(verts_ras, faces),
    )


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


__all__ = [
    "BrainSurface", "surface_from_mask", "gyral_mask_from_mri",
    "gyral_surface_from_mri", "brain_tissue_from_fastsurfer_aseg", "transform_surface",
]
