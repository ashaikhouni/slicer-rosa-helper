"""Contact-pitch v1: two-stage LoG+Frangi SEEG detection pipeline.

Stages:
  1. ``detect`` — runs ``contact_pitch_v1_fit.run_two_stage_detection``.
     This single stage covers preprocessing (hull mask + intracranial +
     hull distance + LoG + Frangi), stage-1 blob-pitch, and stage-2
     Frangi shaft fallback.

Orthogonal to the bolt-first ``deep_core_v2`` pipeline: this detector
locates shanks directly from the CT using a Dixi 3.5 mm pitch prior on
LoG regional-minima blobs (stage 1) with a Frangi shaft fallback for
pitch-unresolved shanks (stage 2). Every trajectory is emitted with a
``source`` tag ("stage1" or "stage2") for easy debugging.
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
    pipeline_version = "1.0.0"

    def _load_image_and_matrices(self, ctx: DetectionContext):
        """Return (sitk_image, ijk_to_ras_4x4, ras_to_ijk_4x4).

        In CLI runs ``ctx.ct.path`` is set and SITK reads the file with
        real origin/direction. In Slicer the path may be absent; build
        the SITK image from ``arr_kji`` (spacing is all the physical
        filters need) but pull the true IJK↔RAS matrices from the
        Slicer volume node in ``ctx.extras.volume_node`` so trajectories
        come out in patient RAS instead of the volume-grid frame.
        """
        import SimpleITK as sitk
        from shank_core.io import image_ijk_ras_matrices

        ct = ctx.get("ct")
        ct_path = getattr(ct, "path", None) if ct is not None else None
        if ct_path:
            img = sitk.ReadImage(str(ct_path))
        else:
            arr = np.asarray(ctx["arr_kji"], dtype=np.float32)
            img = sitk.GetImageFromArray(arr)
            img.SetSpacing(tuple(float(v) for v in ctx["spacing_xyz"]))

        volume_node = (ctx.get("extras") or {}).get("volume_node")
        if volume_node is not None and hasattr(volume_node, "GetIJKToRASMatrix"):
            ijk_to_ras = _volume_node_ijk_to_ras_matrix(volume_node)
            ras_to_ijk = _volume_node_ras_to_ijk_matrix(volume_node)
        else:
            ijk_to_ras, ras_to_ijk = image_ijk_ras_matrices(img)
        return (
            img,
            np.asarray(ijk_to_ras, dtype=float),
            np.asarray(ras_to_ijk, dtype=float),
        )

    def _run_detect(self, ctx: DetectionContext) -> dict[str, Any]:
        from postop_ct_localization import contact_pitch_v1_fit as cpfit

        img, ijk_to_ras, ras_to_ijk = self._load_image_and_matrices(ctx)
        # Forward ctx['logger'] (set by Slicer widget) so the detector
        # can report stage progress; otherwise the call blocks for 10–20 s
        # with no feedback and the UI looks frozen.
        progress_logger = ctx.get("logger")
        trajectories, features = cpfit.run_two_stage_detection(
            img, ijk_to_ras, ras_to_ijk, return_features=True,
            progress_logger=progress_logger,
        )
        self._last_feature_arrays = features
        stage1 = sum(1 for t in trajectories if t.get("source") == "stage1")
        stage2 = sum(1 for t in trajectories if t.get("source") == "stage2")
        return {
            "trajectories": trajectories,
            "stats": {"stage1_count": stage1, "stage2_count": stage2},
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
            stats = output.get("stats") or {}
            diag.set_count("proposal_count", len(result["trajectories"]))
            diag.set_count("stage1_count", int(stats.get("stage1_count", 0)))
            diag.set_count("stage2_count", int(stats.get("stage2_count", 0)))
        except Exception as exc:
            self.fail(
                ctx=ctx, result=result, diagnostics=diag,
                stage="detect", exc=exc,
            )

        return self.finalize(result, diag, t_start)
