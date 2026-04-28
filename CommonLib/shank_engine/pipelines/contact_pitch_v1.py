"""Contact-pitch v1: LoG-blob + library-pitch SEEG detection pipeline.

Stages:
  1. ``detect`` — runs ``contact_pitch_v1_fit.run_two_stage_detection``.
     One stage end-to-end: preprocessing (hull mask + intracranial +
     hull distance + LoG + Frangi), blob-pitch walker, bolt anchoring,
     deep-end refinement, and per-trajectory confidence scoring.

Locates shanks directly from the CT using a library-pitch prior on
LoG regional-minima blobs.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..contracts import DetectionContext, DetectionResult
from .base import BaseDetectionPipeline


def _vtk_matrix_to_numpy(m_vtk):
    mat = np.eye(4, dtype=float)
    for r in range(4):
        for c in range(4):
            mat[r, c] = float(m_vtk.GetElement(r, c))
    return mat


def _volume_node_ijk_to_ras_matrix(volume_node):
    try:
        from __main__ import vtk
    except ImportError:
        import vtk
    m = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(m)
    return _vtk_matrix_to_numpy(m)


def _volume_node_ras_to_ijk_matrix(volume_node):
    try:
        from __main__ import vtk
    except ImportError:
        import vtk
    m = vtk.vtkMatrix4x4()
    volume_node.GetRASToIJKMatrix(m)
    return _vtk_matrix_to_numpy(m)


class ContactPitchV1Pipeline(BaseDetectionPipeline):
    pipeline_id = "contact_pitch_v1"
    pipeline_version = "1.0.18"

    def _load_image_and_matrices(self, ctx: DetectionContext):
        """Return (sitk_image, ijk_to_ras_4x4, ras_to_ijk_4x4).

        When a Slicer ``volume_node`` is present in ``ctx.extras``,
        use Slicer's arr_kji and Slicer's IJK→RAS matrix. Slicer's
        NIfTI reader reorients the voxel array based on the file's
        direction cosines (specifically sform when both sform and
        qform are present), so building SITK from the raw file path
        yields a different voxel layout than what Slicer displays —
        feature volumes registered back to the scene end up rotated
        (subject-137 sform/qform mismatch).

        Only the path-only branch (CLI, or Slicer volumes without a
        storage node) reads via ``sitk.ReadImage`` and takes the
        file's own matrix.
        """
        import SimpleITK as sitk
        from shank_core.io import image_ijk_ras_matrices

        volume_node = (ctx.get("extras") or {}).get("volume_node")

        if volume_node is not None and hasattr(volume_node, "GetIJKToRASMatrix"):
            arr = np.asarray(ctx["arr_kji"], dtype=np.float32)
            img = sitk.GetImageFromArray(arr)
            img.SetSpacing(tuple(float(v) for v in ctx["spacing_xyz"]))
            ijk_to_ras = _volume_node_ijk_to_ras_matrix(volume_node)
            ras_to_ijk = _volume_node_ras_to_ijk_matrix(volume_node)
            self._apply_slicer_geometry_to_sitk(img, ijk_to_ras)
            return (
                img,
                np.asarray(ijk_to_ras, dtype=float),
                np.asarray(ras_to_ijk, dtype=float),
            )

        ct = ctx.get("ct")
        ct_path = getattr(ct, "path", None) if ct is not None else None
        if ct_path:
            img = sitk.ReadImage(str(ct_path))
        else:
            arr = np.asarray(ctx["arr_kji"], dtype=np.float32)
            img = sitk.GetImageFromArray(arr)
            img.SetSpacing(tuple(float(v) for v in ctx["spacing_xyz"]))
        ijk_to_ras, ras_to_ijk = image_ijk_ras_matrices(img)
        return (
            img,
            np.asarray(ijk_to_ras, dtype=float),
            np.asarray(ras_to_ijk, dtype=float),
        )

    @staticmethod
    def _apply_slicer_geometry_to_sitk(img, ijk_to_ras):
        """Overwrite SITK image's origin + direction so they match the
        Slicer volume node's IJK→RAS matrix (converted to SITK's LPS
        convention). Does not resample — just rewrites metadata.
        """
        try:
            spacing = np.array(img.GetSpacing(), dtype=float)
            m = np.asarray(ijk_to_ras, dtype=float)
            origin_ras = m[:3, 3].copy()
            dir_ras = np.zeros((3, 3), dtype=float)
            for k in range(3):
                dir_ras[:, k] = m[:3, k] / max(1e-9, float(spacing[k]))
            # RAS → LPS: flip X and Y components.
            origin_lps = np.array([-origin_ras[0], -origin_ras[1],
                                    origin_ras[2]], dtype=float)
            dir_lps = dir_ras.copy()
            dir_lps[0, :] *= -1.0
            dir_lps[1, :] *= -1.0
            img.SetOrigin(tuple(float(v) for v in origin_lps.tolist()))
            img.SetDirection(tuple(float(v) for v in dir_lps.flatten().tolist()))
        except Exception:
            pass

    def _run_detect(self, ctx: DetectionContext) -> dict[str, Any]:
        from postop_ct_localization import contact_pitch_v1_fit as cpfit

        img, ijk_to_ras, ras_to_ijk = self._load_image_and_matrices(ctx)
        # Forward ctx['logger'] (set by Slicer widget) so the detector
        # can report stage progress; otherwise the call blocks for 10–20 s
        # with no feedback and the UI looks frozen.
        progress_logger = ctx.get("logger")
        # Pitch strategy + manufacturer filter from the Slicer widget.
        # ``pitch_strategy`` drives both the set of walker pitches and
        # the suggestion vendor filter; ``vendors`` can override the
        # latter. Both are optional — CLI / tests don't set them, so
        # the detector defaults to the legacy Dixi-only walker and the
        # full-library suggestion, preserving the regression baseline.
        vendors = ctx.get("contact_pitch_v1_vendors")
        pitch_strategy = ctx.get("contact_pitch_v1_pitch_strategy")
        trajectories, features = cpfit.run_two_stage_detection(
            img, ijk_to_ras, ras_to_ijk, return_features=True,
            progress_logger=progress_logger,
            suggestion_vendors=vendors,
            pitch_strategy=pitch_strategy,
        )
        self._last_feature_arrays = features
        return {
            "trajectories": trajectories,
            "stats": {},
        }

    def run(self, ctx: DetectionContext) -> DetectionResult:
        t_start = time.perf_counter()
        result = self.make_result(ctx)
        diag = self.diagnostics(result)

        try:
            output = self.run_stage(
                ctx=ctx, result=result, diagnostics=diag,
                stage_name="detect",
                fn=lambda: self._run_detect(ctx),
            )
            result["trajectories"] = list(output["trajectories"])
            diag.set_count("proposal_count", len(result["trajectories"]))
        except Exception as exc:
            self.fail(
                ctx=ctx, result=result, diagnostics=diag,
                stage="detect", exc=exc,
            )

        return self.finalize(result, diag, t_start)
