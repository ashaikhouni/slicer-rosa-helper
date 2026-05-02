"""Detection service — the public entry point for all callers.

The rest of the codebase (Slicer modules, CLI, tests) calls one of:

    from rosa_detect.service import run_contact_pitch_v1
    result = run_contact_pitch_v1(ctx)

Swapping the detection algorithm later means changing the body of this
file — nothing outside ``rosa_detect/`` should need to be touched as
long as the new algorithm produces the same ``DetectionResult`` shape.

Inputs (``DetectionContext``):
    ct.path or arr_kji + spacing_xyz   - voxel data
    extras['volume_node']               - optional Slicer node (matrix source)
    config / params                     - algorithm tuning
    logger                              - optional progress callable

Outputs (``DetectionResult``):
    trajectories[]   - public list of {start_ras, end_ras, confidence,
                       confidence_label, model_id, ...}
    contacts[]       - reserved (currently unused; algorithm-side)
    diagnostics      - timing/counts/notes (extras is opaque to callers)
"""

from __future__ import annotations

import time
import traceback
import uuid
from typing import Any

import numpy as np

from . import contact_pitch_v1_fit as cpfit
from .contracts import (
    DetectionContext,
    DetectionResult,
    default_result,
    sanitize_result,
)
from .diagnostics import DiagnosticsCollector, StageExecutionError


PIPELINE_ID = "contact_pitch_v1"
PIPELINE_VERSION = "1.0.29"


# ---------------------------------------------------------------------
# Image + matrix loading
#
# The split between voxel data (always from disk when path is given) and
# IJK->RAS matrix (Slicer node when present, header otherwise) is the
# CLI/Slicer parity invariant: ROSA-imported volumes have registration
# baked into the node matrix that's not in the on-disk header. See
# feedback_cli_slicer_parity.md.
# ---------------------------------------------------------------------


def _vtk_matrix_to_numpy(m_vtk) -> np.ndarray:
    mat = np.eye(4, dtype=float)
    for r in range(4):
        for c in range(4):
            mat[r, c] = float(m_vtk.GetElement(r, c))
    return mat


def _volume_node_ijk_to_ras_matrix(volume_node) -> np.ndarray:
    try:
        from __main__ import vtk
    except ImportError:
        import vtk
    m = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(m)
    return _vtk_matrix_to_numpy(m)


def _volume_node_ras_to_ijk_matrix(volume_node) -> np.ndarray:
    try:
        from __main__ import vtk
    except ImportError:
        import vtk
    m = vtk.vtkMatrix4x4()
    volume_node.GetRASToIJKMatrix(m)
    return _vtk_matrix_to_numpy(m)


def _apply_slicer_geometry_to_sitk(img, ijk_to_ras: np.ndarray) -> None:
    """Overwrite SITK image origin + direction so they match the Slicer
    volume node's IJK->RAS matrix (converted to LPS). Does not resample.
    Without this, ``prepare_volume`` inherits the on-disk header geometry
    and the feature volumes end up offset from the displayed CT.
    """
    try:
        spacing = np.array(img.GetSpacing(), dtype=float)
        m = np.asarray(ijk_to_ras, dtype=float)
        origin_ras = m[:3, 3].copy()
        dir_ras = np.zeros((3, 3), dtype=float)
        for k in range(3):
            dir_ras[:, k] = m[:3, k] / max(1e-9, float(spacing[k]))
        # RAS -> LPS: flip X and Y components.
        origin_lps = np.array(
            [-origin_ras[0], -origin_ras[1], origin_ras[2]],
            dtype=float,
        )
        dir_lps = dir_ras.copy()
        dir_lps[0, :] *= -1.0
        dir_lps[1, :] *= -1.0
        img.SetOrigin(tuple(float(v) for v in origin_lps.tolist()))
        img.SetDirection(tuple(float(v) for v in dir_lps.flatten().tolist()))
    except Exception:
        pass


def _load_image_and_matrices(ctx: DetectionContext):
    """Return (sitk_image, ijk_to_ras_4x4, ras_to_ijk_4x4) from ``ctx``."""

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
            _apply_slicer_geometry_to_sitk(img, ijk_to_ras)
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
        _apply_slicer_geometry_to_sitk(img, ijk_to_ras)
    else:
        ijk_to_ras, ras_to_ijk = image_ijk_ras_matrices(img)

    return (
        img,
        np.asarray(ijk_to_ras, dtype=float),
        np.asarray(ras_to_ijk, dtype=float),
    )


# ---------------------------------------------------------------------
# Service entry point
# ---------------------------------------------------------------------


def run_contact_pitch_v1(ctx: DetectionContext) -> DetectionResult:
    """Run contact_pitch_v1 detection end-to-end. Returns the public,
    JSON-safe ``DetectionResult``. Use ``run_contact_pitch_v1_with_features``
    when the caller (Slicer visualization) also needs the raw feature
    volumes (LoG, Frangi, hull mask, etc.) — those are algorithm-private
    and not part of the public contract.
    """

    result, _features = run_contact_pitch_v1_with_features(ctx)
    return result


def run_contact_pitch_v1_with_features(
    ctx: DetectionContext,
) -> tuple[DetectionResult, dict[str, Any]]:
    """Same as ``run_contact_pitch_v1`` but also returns the algorithm's
    raw feature arrays (numpy / SimpleITK objects, not JSON-safe).

    The features dict is algorithm-specific and may change without
    notice when the detection algorithm changes. Treat it as opaque.
    Returns ``({}, {})``-shaped fallback if the run errored.
    """

    t_start = time.perf_counter()
    run_id = str(ctx.get("run_id") or uuid.uuid4())
    config = dict(ctx.get("config") or ctx.get("params") or {})

    result = default_result(
        pipeline_id=PIPELINE_ID,
        pipeline_version=PIPELINE_VERSION,
        run_id=run_id,
        params=config,
    )
    diag = DiagnosticsCollector(result["diagnostics"])
    logger = ctx.get("logger")
    features: dict[str, Any] = {}

    def _log(msg: str) -> None:
        if logger is None:
            return
        try:
            logger(str(msg))
        except Exception:
            pass

    try:
        img, ijk_to_ras, ras_to_ijk = _load_image_and_matrices(ctx)
        # Optional Slicer-tab parameters; CLI/tests leave them unset and
        # the algorithm defaults to the Dixi-only walker + full-library
        # suggestion (the regression baseline).
        vendors = ctx.get("contact_pitch_v1_vendors")
        pitch_strategy = ctx.get("contact_pitch_v1_pitch_strategy")

        def _run_detect():
            return cpfit.run_two_stage_detection(
                img, ijk_to_ras, ras_to_ijk,
                return_features=True,
                progress_logger=logger,
                suggestion_vendors=vendors,
                pitch_strategy=pitch_strategy,
            )

        trajectories, features = diag.run_stage("detect", _run_detect)
        result["trajectories"] = list(trajectories)
        diag.set_count("proposal_count", len(result["trajectories"]))
    except StageExecutionError as exc:
        result["status"] = "error"
        result["error"] = {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "stage": exc.stage,
        }
        diag.add_reason("pipeline_error", 1)
        diag.add_reason(f"stage:{exc.stage}", 1)
        diag.note(f"stage '{exc.stage}' failed: {exc}")
        if config.get("debug_traceback"):
            diag.set_extra("traceback", traceback.format_exc())
    except Exception as exc:
        # Anything raised outside ``run_stage`` (e.g. image load failure).
        result["status"] = "error"
        result["error"] = {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "stage": "load_image",
        }
        diag.add_reason("pipeline_error", 1)
        diag.note(f"image load failed: {exc}")
        if config.get("debug_traceback"):
            diag.set_extra("traceback", traceback.format_exc())
        _log(f"detection error: {exc}")

    diag.set_timing("total_ms", (time.perf_counter() - t_start) * 1000.0)
    return sanitize_result(result), features
