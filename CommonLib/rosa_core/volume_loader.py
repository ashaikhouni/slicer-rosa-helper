"""CT loading + feature extraction for the placement pipeline.

Encapsulates the boilerplate currently duplicated across the QC notebook,
``rosa_detect.service``, and various test scripts:

  1. Load CT (path / sitk.Image) → SITK image + IJK↔RAS matrices.
  2. Run ``guided_fit_engine.compute_features`` → feature dict
     (ct_arr_kji, log, frangi, head_distance, etc.).
  3. Run ``primitives.bolt_anchor.extract_bolt_candidates`` → bolt CC list.

Returned tuple ``(features, bolts)`` is exactly what
``rosa_core.contact_placement.run_two_pass`` consumes.

Slicer paths (vtkMRMLScalarVolumeNode → SITK) keep using
``rosa_scene.sitk_volume_adapter.prepare_detection_context``; this module
serves CLI / standalone callers and the dispatcher's lazy-load path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_features_and_bolts(ct):
    """Load CT (path / SITK image) and compute features + bolt candidates.

    Args:
        ct: ``str | Path | sitk.Image``. Anything ``SimpleITK.ReadImage``
            accepts — usually a NIfTI or NRRD path; SITK images pass through.

    Returns:
        ``(features, bolts)``:
          * ``features`` — dict from ``guided_fit_engine.compute_features``
            with keys ``ct_arr_kji``, ``log``, ``frangi``, ``head_distance``,
            ``ras_to_ijk_mat``, ``ijk_to_ras_mat``, ``img``.
          * ``bolts`` — list of bolt-CC dicts from ``extract_bolt_candidates``
            (centroid, voxel-count, hull-proximity).

    Raises:
        FileNotFoundError: when ``ct`` is a path that doesn't exist.
        RuntimeError: when feature computation fails.
    """
    img = _to_sitk_image(ct)
    return _compute_features_and_bolts(img)


def _to_sitk_image(ct):
    """Coerce ``ct`` to a SITK image."""
    import SimpleITK as sitk
    if isinstance(ct, sitk.Image):
        return ct
    if isinstance(ct, (str, Path)):
        path = Path(ct)
        if not path.exists():
            raise FileNotFoundError(f"CT path does not exist: {path}")
        return sitk.ReadImage(str(path))
    raise TypeError(
        f"ct must be str | Path | SimpleITK.Image, got {type(ct).__name__}"
    )


def _compute_features_and_bolts(img) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute the features dict + bolt CCs from a SITK image.

    Mirrors the notebook's exact compute-features + extract-bolts cell so
    the placement pipeline gets bit-identical inputs to what the QC
    notebook uses.
    """
    from shank_core.io import image_ijk_ras_matrices
    from rosa_detect import contact_pitch_v1_fit as cpfit
    from rosa_detect import guided_fit_engine as gfe
    from rosa_detect.primitives.bolt_anchor import (
        BOLT_HULL_PROXIMITY_MM, METAL_BOLT_THRESHOLD, extract_bolt_candidates,
    )

    i2r_in, r2i_in = image_ijk_ras_matrices(img)
    features = gfe.compute_features(img, np.asarray(i2r_in), np.asarray(r2i_in))
    i2r = np.asarray(features["ijk_to_ras_mat"])
    r2i = np.asarray(features["ras_to_ijk_mat"])

    metal_evidence = cpfit.compute_metal_evidence_volume(
        features["log"], features["ct_arr_kji"],
    )
    bolts, _bolt_mask = extract_bolt_candidates(
        features["log"], features["head_distance"], i2r, img.GetSpacing(),
        ras_to_ijk_mat=r2i, ct_arr=metal_evidence,
        hu_threshold=METAL_BOLT_THRESHOLD,
        hull_proximity_mm=BOLT_HULL_PROXIMITY_MM,
    )
    return features, bolts


__all__ = ["load_features_and_bolts"]
