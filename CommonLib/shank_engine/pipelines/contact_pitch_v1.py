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
    pipeline_version = "1.0.29"

    def _load_image_and_matrices(self, ctx: DetectionContext):
        """Return (sitk_image, ijk_to_ras_4x4, ras_to_ijk_4x4).

        Voxel data: when ``ctx['ct'].path`` exists on disk we load via
        ``sitk.ReadImage`` regardless of whether a Slicer ``volume_node``
        is also present. CLI tests and the Slicer widget therefore see
        the same voxel array — fixes the ``arrayFromVolume`` vs
        ``sitk.ReadImage`` data fork that hit S56 / AMC099 (see
        ``feedback_cli_slicer_parity.md``).

        IJK→RAS matrix: prefer the Slicer volume node's CURRENT matrix
        when one is present, falling back to the on-disk header
        otherwise.

        Why the matrix and the data come from different sources: ROSA-
        imported volumes have the registration BAKED into the volume
        node's ``IJKToRASMatrix`` (no parent transform node). The
        on-disk NIfTI header still carries the un-registered matrix.
        Reading the header for both data + matrix runs the entire
        detection in un-registered RAS while Slicer renders the CT
        post-registration, so trajectories + LoG + Frangi all align
        with each other but not with the CT volume the user sees.
        Using the node's matrix while keeping the on-disk array
        produces detection results in the SAME RAS frame Slicer is
        displaying.

        CLI parity is preserved: CLI calls don't put a volume_node in
        ``ctx['extras']``, so the path-only branch returns the on-disk
        matrix exactly as before. The dataset regression test
        (``test_dataset_full``) loads CTs without registration, so the
        node matrix and disk matrix are identical when a node IS
        present — no behavior drift on the gated subjects.
        """
        import SimpleITK as sitk
        from shank_core.io import image_ijk_ras_matrices

        volume_node = (ctx.get("extras") or {}).get("volume_node")

        ct = ctx.get("ct")
        ct_path = getattr(ct, "path", None) if ct is not None else None
        if ct_path:
            img = sitk.ReadImage(str(ct_path))
            if volume_node is not None and hasattr(volume_node, "GetIJKToRASMatrix"):
                ijk_to_ras = _volume_node_ijk_to_ras_matrix(volume_node)
                ras_to_ijk = _volume_node_ras_to_ijk_matrix(volume_node)
                # Rewrite SITK image origin/direction to match the node
                # matrix; voxel data is unchanged. Without this the
                # canonical-resample step in prepare_volume inherits
                # the on-disk header geometry and produces feature
                # volumes offset from the displayed CT.
                self._apply_slicer_geometry_to_sitk(img, ijk_to_ras)
            else:
                ijk_to_ras, ras_to_ijk = image_ijk_ras_matrices(img)
            return (
                img,
                np.asarray(ijk_to_ras, dtype=float),
                np.asarray(ras_to_ijk, dtype=float),
            )

        arr = np.asarray(ctx["arr_kji"], dtype=np.float32)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing(tuple(float(v) for v in ctx["spacing_xyz"]))

        if volume_node is not None and hasattr(volume_node, "GetIJKToRASMatrix"):
            ijk_to_ras = _volume_node_ijk_to_ras_matrix(volume_node)
            ras_to_ijk = _volume_node_ras_to_ijk_matrix(volume_node)
            self._apply_slicer_geometry_to_sitk(img, ijk_to_ras)
        else:
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
