"""Headless atlas providers for rosa_agent contact labeling.

Conforms to the same Provider duck-type the Slicer-side
``atlas_assignment_policy.collect_provider_samples`` expects:

    provider.is_ready() -> bool
    provider.sample_contact(point_world_ras) -> AtlasSampleResult | None

Without Slicer / VTK we use ``nibabel`` to load the labelmap NIfTI and a
``scipy.spatial.cKDTree`` (when scipy is available) to answer
nearest-labeled-voxel queries. Brute-force NumPy is the fallback.

Centroid math, sample-result formatting, and FreeSurfer LUT parsing
delegate to ``rosa_core.atlas_index`` so this side and the Slicer side
share one implementation of those concerns.

Two concrete providers:

* ``LabelmapAtlasProvider`` — single labelmap volume + LUT (FreeSurfer
  parcellation, white-matter mask, etc.).
* ``ThomasAtlasProvider`` — multiple per-segment labelmaps from a THOMAS
  output directory; each segment's foreground voxels become a labeled
  point cloud the same way the Slicer ``ThomasSegmentationAtlasProvider``
  does after ``ExportSegmentsToLabelmapNode``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from rosa_core.atlas_index import (
    compute_label_centroids,
    distance_to_centroid_mm,
    format_atlas_sample,
    parse_freesurfer_lut,
)


def _require_numpy():
    import numpy as np
    return np


def _load_label_volume(path: str | Path):
    """Return (label_array, affine_xyz_to_ras_4x4, shape_kji).

    nibabel's ``affine`` maps voxel ``(i, j, k, 1)`` to world coordinates
    in **RAS** (NIfTI's native frame), which is exactly what we need —
    no LPS↔RAS flip required.
    """
    import nibabel as nib
    np = _require_numpy()

    img = nib.load(str(path))
    arr = np.asarray(img.get_fdata(caching="unchanged"), dtype=np.int32)
    affine = np.asarray(img.affine, dtype=float)
    return arr, affine, arr.shape


def _voxel_centers_ras(arr_ijk, affine_ijk_to_ras, label_filter=None):
    """Return (RAS Nx3, label values) for foreground voxels.

    ``arr_ijk`` is indexed ``[i, j, k]`` per nibabel's ``get_fdata``.
    ``label_filter`` (optional set) keeps only voxels with those labels.
    """
    np = _require_numpy()
    if label_filter is None:
        mask = arr_ijk > 0
    else:
        mask = np.isin(arr_ijk, np.asarray(list(label_filter), dtype=arr_ijk.dtype))
    ijk = np.argwhere(mask)
    if ijk.size == 0:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=np.int32)
    labels = arr_ijk[ijk[:, 0], ijk[:, 1], ijk[:, 2]].astype(np.int32)
    h = np.ones((ijk.shape[0], 4), dtype=float)
    h[:, :3] = ijk.astype(float)
    ras = (h @ affine_ijk_to_ras.T)[:, :3]
    return ras, labels


def _build_kdtree(points_ras):
    """Return a callable ``query(point_ras) -> (idx, distance_mm)``."""
    np = _require_numpy()
    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(points_ras)

        def _query(point):
            d, idx = tree.query(np.asarray(point, dtype=float).reshape(3), k=1)
            return int(idx), float(d)

        return _query
    except ImportError:
        pts = np.asarray(points_ras, dtype=float)

        def _query(point):
            p = np.asarray(point, dtype=float).reshape(3)
            d2 = ((pts - p) ** 2).sum(axis=1)
            idx = int(np.argmin(d2))
            # ``np.sqrt`` over ``math.sqrt``: keeps the import surface
            # to (numpy,) — used to import ``math`` here, but the
            # 2026-05-02 atlas_index extraction moved the centroid /
            # distance helpers to rosa_core and the local ``import math``
            # went with them. Using np.sqrt sidesteps re-adding it.
            return idx, float(np.sqrt(float(d2[idx])))

        return _query


# ---------------------------------------------------------------------
# Single-labelmap provider.
# ---------------------------------------------------------------------


def _register_and_resample_labelmap_to_temp(
    label_path: str | Path,
    atlas_base_path: str | Path,
    target_volume_path: str | Path,
    *,
    transform_kind: str = "rigid",
    intermediate_volume_path: str | Path | None = None,
    save_intermediate_in_target: str | Path | None = None,
    save_target_in_atlas: str | Path | None = None,
    save_intermediate_in_atlas: str | Path | None = None,
    logger=None,
) -> Path:
    """Register ``atlas_base`` to ``target_volume`` (Mattes MI), resample
    ``label_path`` through the same transform onto the target volume's
    grid (nearest-neighbor), and write the result to a NamedTemp NIfTI.
    Returns that path.

    ``transform_kind`` selects the registration model:

    * ``"rigid"`` (default) — for a same-subject atlas (FreeSurfer / THOMAS
      reconned on the patient's own T1). Only pose differs from the CT, so
      6-DOF is right and safest.
    * ``"affine"`` — for a **population/MNI** atlas template. A standard
      brain differs from the patient in size and shape, so 12-DOF (scale +
      shear) is required; rigid would leave the atlas grossly misaligned.

    ``intermediate_volume_path`` (the patient's own T1) turns on the
    **through-T1** path, the accurate route for an MNI atlas onto a CT:

        MNI template --affine--> T1 --rigid--> CT

    Cross-modal MNI→CT affine (skull, metal, no cortex contrast) is noisier
    than MNI→T1 (MR↔MR); registering to the patient's T1 first, then rigidly
    to the CT, is more robust. The two transforms are composed so the
    labelmap warps straight onto the CT grid. ``save_intermediate_in_target``
    additionally writes the T1 resampled into CT space (registration QC — the
    MRI backdrop the clinician approves against).

    This is the bridge between an atlas (in its own template frame) and
    SEEG contacts (which live in the postop CT's frame). Without it,
    sampling the raw parcellation at CT-frame contact RAS coords gives
    nonsense.
    """
    import tempfile
    import SimpleITK as sitk
    from rosa_core.registration import (
        register_affine_mi,
        register_rigid_mi,
        resample_volume,
    )

    kind = str(transform_kind).lower()
    if kind not in ("rigid", "affine"):
        raise ValueError(f"transform_kind must be 'rigid' or 'affine', got {transform_kind!r}")

    target_img = sitk.ReadImage(str(target_volume_path))
    base_img = sitk.ReadImage(str(atlas_base_path))
    label_img = sitk.ReadImage(str(label_path))

    if intermediate_volume_path is not None:
        # Through-T1: MNI --affine--> T1 --rigid--> CT, composed.
        t1_img = sitk.ReadImage(str(intermediate_volume_path))
        if logger is not None:
            logger(f"[atlas-reg] through-T1: {Path(atlas_base_path).name} "
                   f"--affine--> {Path(intermediate_volume_path).name} "
                   f"--rigid--> {Path(target_volume_path).name}")
        reg_a = register_affine_mi(fixed=t1_img, moving=base_img, logger=logger)   # T1<-MNI
        reg_r = register_rigid_mi(fixed=target_img, moving=t1_img, logger=logger)  # CT<-T1
        if logger is not None:
            logger(f"[atlas-reg] MNI→T1 metric={reg_a.final_metric:+.5f}; "
                   f"T1→CT metric={reg_r.final_metric:+.5f}")
        # Resample maps CT (reference) points → moving space. We need
        # CT→MNI = affine ∘ rigid, i.e. apply rigid (CT→T1) first. ITK
        # CompositeTransform applies the LAST-added transform first, so add
        # the affine (T1→MNI) first and the rigid (CT→T1) last.
        composite = sitk.CompositeTransform(3)
        composite.AddTransform(reg_a.transform)
        composite.AddTransform(reg_r.transform)
        resampled_label = resample_volume(
            label_img, composite, reference=target_img, interp="nearest")
        if save_intermediate_in_target is not None:
            t1_in_ct = resample_volume(
                t1_img, reg_r.transform, reference=target_img, interp="linear")
            sitk.WriteImage(t1_in_ct, str(save_intermediate_in_target))
            if logger is not None:
                logger(f"[atlas-reg] wrote MRI-in-CT QC → {save_intermediate_in_target}")
        # Atlas-space (AC-PC aligned MNI grid) copies of CT + MRI for QC in
        # standard neuroanatomical planes. Invert the two registrations:
        #   MRI→MNI = resample(T1, ref=MNI, MNI→T1)
        #   CT →MNI = resample(CT, ref=MNI, MNI→T1→CT)
        if save_target_in_atlas is not None or save_intermediate_in_atlas is not None:
            a_inv = reg_a.transform.GetInverse()   # MNI→T1
            r_inv = reg_r.transform.GetInverse()   # T1→CT
            if save_intermediate_in_atlas is not None:
                mri_in_mni = resample_volume(
                    t1_img, a_inv, reference=base_img, interp="linear")
                sitk.WriteImage(mri_in_mni, str(save_intermediate_in_atlas))
            if save_target_in_atlas is not None:
                mni_to_ct = sitk.CompositeTransform(3)
                mni_to_ct.AddTransform(r_inv)   # applied second: T1→CT
                mni_to_ct.AddTransform(a_inv)   # applied first:  MNI→T1
                ct_in_mni = resample_volume(
                    target_img, mni_to_ct, reference=base_img, interp="linear")
                sitk.WriteImage(ct_in_mni, str(save_target_in_atlas))
            if logger is not None:
                logger("[atlas-reg] wrote AC-PC (MNI-space) CT/MRI QC copies")
    else:
        if logger is not None:
            logger(
                f"[atlas-reg] registering {Path(atlas_base_path).name} -> "
                f"{Path(target_volume_path).name} ({kind} + MI)…"
            )
        register = register_affine_mi if kind == "affine" else register_rigid_mi
        reg_result = register(
            fixed=target_img, moving=base_img,
            logger=logger,
        )
        if logger is not None:
            logger(
                f"[atlas-reg] done: metric={reg_result.final_metric:+.5f} "
                f"iters={reg_result.n_iterations} "
                f"({reg_result.converged_reason})"
            )
        resampled_label = resample_volume(
            label_img, reg_result.transform,
            reference=target_img,
            interp="nearest",
        )
    # NamedTemporaryFile delete=False so the path survives until the
    # caller (Provider.__init__) can read it back via nibabel; the
    # provider keeps a reference so the file is preserved for the
    # provider's lifetime.
    tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    tmp.close()
    sitk.WriteImage(resampled_label, tmp.name)
    if logger is not None:
        logger(
            f"[atlas-reg] resampled labelmap written to {tmp.name} "
            f"(target grid, nearest-neighbor)"
        )
    return Path(tmp.name)


class LabelmapAtlasProvider:
    """Atlas provider for one label volume (FreeSurfer parcellation, WM, etc.).

    Optional ``atlas_base_path`` + ``target_volume_path`` arguments
    trigger an in-line rigid-registration pass at construction time:
    the atlas base (e.g. the T1 FreeSurfer's recon was run on) is
    registered to the target volume (typically the postop CT the
    contacts live in), the labelmap is resampled through that
    transform onto the target grid (nearest-neighbor), and the
    resampled NIfTI is what the provider then queries. This is the
    standard fix for "FreeSurfer parcellation is in T1 RAS but my
    contacts are in CT RAS".

    When both atlas_base / target paths are omitted (default), the
    labelmap is loaded as-is and assumed to already be in the
    contact-frame RAS.
    """

    def __init__(
        self,
        source_id: str,
        display_name: str,
        label_path: str | Path,
        *,
        lut_path: str | Path | None = None,
        label_names: dict[int, str] | None = None,
        atlas_base_path: str | Path | None = None,
        target_volume_path: str | Path | None = None,
        transform_kind: str = "rigid",
        intermediate_volume_path: str | Path | None = None,
        save_intermediate_in_target: str | Path | None = None,
        save_target_in_atlas: str | Path | None = None,
        save_intermediate_in_atlas: str | Path | None = None,
        logger=None,
    ) -> None:
        self.source_id = str(source_id)
        self.display_name = str(display_name)
        self.label_path = str(label_path)
        # Holds the resampled labelmap temp file (when registration ran)
        # so it survives for the provider's lifetime.
        self._registered_labelmap_path: Path | None = None

        if atlas_base_path is not None and target_volume_path is not None:
            self._registered_labelmap_path = _register_and_resample_labelmap_to_temp(
                label_path=label_path,
                atlas_base_path=atlas_base_path,
                target_volume_path=target_volume_path,
                transform_kind=transform_kind,
                intermediate_volume_path=intermediate_volume_path,
                save_intermediate_in_target=save_intermediate_in_target,
                save_target_in_atlas=save_target_in_atlas,
                save_intermediate_in_atlas=save_intermediate_in_atlas,
                logger=logger,
            )
            label_path = self._registered_labelmap_path
        elif (atlas_base_path is None) != (target_volume_path is None):
            raise ValueError(
                "atlas_base_path and target_volume_path must be passed "
                "together (both, or neither)."
            )

        np = _require_numpy()
        self._np = np

        self._arr, self._affine, self._shape = _load_label_volume(label_path)
        self._affine_inv = np.linalg.inv(self._affine)

        ras, labels = _voxel_centers_ras(self._arr, self._affine)
        self._points_ras = ras
        self._labels = labels
        self._query = _build_kdtree(ras) if ras.shape[0] else None

        if label_names is not None:
            self._label_names = dict(label_names)
        elif lut_path is not None:
            self._label_names = parse_freesurfer_lut(lut_path)
        else:
            self._label_names = {}

        self._centroids = compute_label_centroids(ras, labels)

    def is_ready(self) -> bool:
        return self._query is not None

    def sample_contact(self, point_world_ras: Sequence[float]) -> dict[str, Any] | None:
        if not self.is_ready():
            return None
        idx, distance_mm = self._query(point_world_ras)
        label_value = int(self._labels[idx])
        label_name = self._label_names.get(label_value, f"Label_{label_value}")
        centroid = self._centroids.get(label_value)
        return format_atlas_sample(
            source_id=self.source_id,
            label_value=label_value,
            label_name=label_name,
            distance_to_voxel_mm=float(distance_mm),
            distance_to_centroid_mm=distance_to_centroid_mm(
                point_world_ras, centroid,
            ),
            # Headless agent: native and world coordinates coincide
            # (no per-volume Slicer transform stack).
            native_ras=point_world_ras,
        )


# ---------------------------------------------------------------------
# THOMAS provider — directory of per-segment labelmaps.
# ---------------------------------------------------------------------


class ThomasAtlasProvider:
    """Atlas provider for THOMAS segmentation output directory.

    Each ``*.nii.gz`` (excluding ``thalamus*``) under ``segmentation_dir``
    is treated as one segment. Segment name = file stem; label value is
    auto-assigned starting from 1 in stable filename order. This matches
    the Slicer-side provider's behavior where each segment becomes one
    set of foreground voxels merged into a single nearest-neighbor
    index.
    """

    SKIP_PATTERNS = ("thalamus",)

    def __init__(
        self,
        segmentation_dir: str | Path,
        *,
        source_id: str = "thomas",
        display_name: str = "THOMAS",
    ) -> None:
        self.source_id = str(source_id)
        self.display_name = str(display_name)
        self.segmentation_dir = str(segmentation_dir)

        np = _require_numpy()
        seg_paths = sorted(
            p for p in Path(segmentation_dir).glob("*.nii*")
            if not any(skip in p.name.lower() for skip in self.SKIP_PATTERNS)
        )

        all_pts = []
        all_labels = []
        label_names: dict[int, str] = {}
        next_label = 1
        for path in seg_paths:
            arr, affine, _ = _load_label_volume(path)
            ras, _labels = _voxel_centers_ras(arr, affine)
            if ras.shape[0] == 0:
                continue
            all_pts.append(ras)
            all_labels.append(np.full((ras.shape[0],), next_label, dtype=np.int32))
            label_names[next_label] = path.stem.replace(".nii", "")
            next_label += 1

        if not all_pts:
            self._points_ras = np.empty((0, 3), dtype=float)
            self._labels = np.empty((0,), dtype=np.int32)
            self._query = None
        else:
            self._points_ras = np.vstack(all_pts)
            self._labels = np.concatenate(all_labels)
            self._query = _build_kdtree(self._points_ras)

        self._label_names = label_names
        self._centroids = compute_label_centroids(self._points_ras, self._labels)

    def is_ready(self) -> bool:
        return self._query is not None

    def sample_contact(self, point_world_ras: Sequence[float]) -> dict[str, Any] | None:
        if not self.is_ready():
            return None
        idx, distance_mm = self._query(point_world_ras)
        label_value = int(self._labels[idx])
        label_name = self._label_names.get(label_value, f"Label_{label_value}")
        centroid = self._centroids.get(label_value)
        return format_atlas_sample(
            source_id=self.source_id,
            label_value=label_value,
            label_name=label_name,
            distance_to_voxel_mm=float(distance_mm),
            distance_to_centroid_mm=distance_to_centroid_mm(
                point_world_ras, centroid,
            ),
            native_ras=point_world_ras,
        )
