"""Detection service — the public entry point for all callers.

The rest of the codebase (Slicer modules, CLI, tests) calls one of:

    from rosa_detect.service import run_contact_pitch_v1
    result = run_contact_pitch_v1(ctx)

Swapping the detection algorithm later means changing the body of this
file — nothing outside ``rosa_detect/`` should need to be touched as
long as the new algorithm produces the same ``DetectionResult`` shape
(see ``rosa_detect.contracts.DetectedTrajectory`` for the trajectory
contract; in particular, future curved-shank algorithms can populate
the optional ``path_ras`` polyline without breaking callers that read
``start_ras`` / ``end_ras`` today).

Inputs (``DetectionContext``):
    ct.path or arr_kji + spacing_xyz   - voxel data
    extras['volume_node']               - optional Slicer node (matrix source)
    config / params                     - algorithm tuning
    logger                              - optional progress callable

Outputs (``DetectionResult``):
    trajectories[]   - list of ``DetectedTrajectory`` dicts (see
                       contracts.py for the public field set; algorithm-
                       private fields stamped on the same dict ARE NOT
                       part of the contract).
    contacts[]       - reserved (currently unused; algorithm-side).
    diagnostics      - timing/counts/notes; ``extras`` is opaque.
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
# Feature-volume publication spec
#
# Slicer publishes the algorithm's intermediate feature volumes (LoG,
# Frangi, masks, ...) into the scene under canonical names so they can
# be inspected in 3D / slice views and reused by later modules
# (ContactsTrajectoryView caches LoG by name).
#
# Per-pipeline spec keeps both the names and the field set in this
# package — the day a v2 lands with a different feature set, only this
# table changes; the Slicer-side publisher and the LoG cache lookup in
# ContactsTrajectoryView don't.
# ---------------------------------------------------------------------


FEATURE_VOLUME_SPECS: dict[str, dict[str, Any]] = {
    "contact_pitch_v1": {
        # Slicer node name = f"{ct_volume_name}_{name_prefix}_{label}".
        "name_prefix": "ContactPitch",
        # (features_dict_key, label, use_percentile_window_level).
        "volumes": [
            ("log_sigma1",    "LoG_sigma1",       True),
            ("frangi_sigma1", "Frangi_sigma1",    True),
            ("head_distance", "HeadDistance_mm",  True),
            ("intracranial",  "IntracranialMask", False),
            ("hull",          "HullMask",         False),
            ("bolt_mask",     "BoltMask",         False),
        ],
        # Convenience: the LoG label callers can reuse instead of
        # hardcoding "LoG_sigma1" themselves (see
        # ContactsTrajectoryView._resolve_log_volume_node).
        "log_label": "LoG_sigma1",
    },
}


def feature_volume_spec(pipeline_id: str = PIPELINE_ID) -> dict[str, Any]:
    """Return the feature-volume publication spec for one pipeline.

    Falls back to the default pipeline's spec for unknown ids so older
    Slicer code that doesn't know about a new pipeline still publishes
    something sensible.
    """
    return FEATURE_VOLUME_SPECS.get(pipeline_id, FEATURE_VOLUME_SPECS[PIPELINE_ID])


def feature_volume_node_name(
    base_volume_name: str,
    label: str,
    pipeline_id: str = PIPELINE_ID,
) -> str:
    """Canonical Slicer node name for one published feature volume.

    Single source of truth for the naming convention so publishers and
    consumers (e.g. cached-LoG lookup) can't drift.
    """
    spec = feature_volume_spec(pipeline_id)
    prefix = str(spec.get("name_prefix") or "Detect")
    return f"{base_volume_name}_{prefix}_{label}"


# ---------------------------------------------------------------------
# Image + matrix loading
#
# The split between voxel data (always from disk when path is given) and
# IJK->RAS matrix (Slicer node when present, header otherwise) is the
# CLI/Slicer parity invariant: ROSA-imported volumes have registration
# baked into the node matrix that's not in the on-disk header. See
# feedback_cli_slicer_parity.md.
# ---------------------------------------------------------------------


def vtk_matrix4x4_to_numpy(m_vtk) -> np.ndarray:
    """Convert a vtkMatrix4x4 (or any object with ``GetElement(r, c)``)
    to a numpy 4x4 float matrix. Public — Slicer modules and the CLI
    use the same pattern; keep one implementation.
    """
    mat = np.eye(4, dtype=float)
    for r in range(4):
        for c in range(4):
            mat[r, c] = float(m_vtk.GetElement(r, c))
    return mat


def volume_node_geometry(volume_node) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ijk_to_ras_4x4, ras_to_ijk_4x4)`` for a Slicer volume node.

    Lazy-imports vtk, preferring the Slicer-injected ``__main__.vtk``
    over a pip ``vtk`` install. Public seam used by Auto Fit, Guided
    Fit, and ContactsTrajectoryView; never call vtk directly from
    Slicer modules — go through this helper so any future migration to
    a non-vtk matrix source happens in one place.
    """
    try:
        from __main__ import vtk
    except ImportError:
        import vtk
    m_ijk = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(m_ijk)
    m_ras = vtk.vtkMatrix4x4()
    volume_node.GetRASToIJKMatrix(m_ras)
    return vtk_matrix4x4_to_numpy(m_ijk), vtk_matrix4x4_to_numpy(m_ras)


def apply_slicer_geometry_to_sitk(img, ijk_to_ras: np.ndarray) -> None:
    """Overwrite SITK image origin + direction so they match the Slicer
    volume node's IJK->RAS matrix (converted to LPS). Does not resample.

    Without this, downstream resample/canonicalization inherits the
    on-disk header geometry and the feature volumes end up offset from
    the displayed CT — for ROSA-imported volumes the header origin and
    the registered node origin differ, and Auto Fit / Guided Fit /
    LoG-cache reads ALL silently mis-localize otherwise.

    Public seam — all three callers (rosa_detect.service load path,
    PostopCT guided_fit, ContactsTrajectoryView LoG compute) use this
    one implementation. See feedback_cli_slicer_parity.md.
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


def image_from_volume_node(volume_node) -> tuple[Any, np.ndarray, np.ndarray]:
    """Build an SITK image from a Slicer volume node, with the node's
    IJK->RAS matrix stamped into origin/direction (LPS-flipped).

    Returns ``(sitk_image, ijk_to_ras, ras_to_ijk)`` ready to feed into
    ``guided_fit_engine.compute_features`` or any other rosa_detect
    entry point. Single source of truth for the Slicer-side parity
    invariant; callers must not roll their own copy.
    """
    import SimpleITK as sitk
    # Local import keeps slicer.* off the CLI path — this function is
    # only called from Slicer modules where ``slicer`` is already in
    # ``__main__``.
    try:
        from __main__ import slicer
    except ImportError:
        import slicer  # type: ignore[no-redef]

    arr_kji = np.asarray(
        slicer.util.arrayFromVolume(volume_node), dtype=np.float32,
    )
    img = sitk.GetImageFromArray(arr_kji)
    img.SetSpacing(tuple(float(v) for v in volume_node.GetSpacing()))
    ijk_to_ras, ras_to_ijk = volume_node_geometry(volume_node)
    apply_slicer_geometry_to_sitk(img, ijk_to_ras)
    return img, ijk_to_ras, ras_to_ijk


# ---------------------------------------------------------------------
# Internal aliases (the original underscore names) — kept so callers
# inside this file don't churn. New code should call the public names.
# ---------------------------------------------------------------------

_vtk_matrix_to_numpy = vtk_matrix4x4_to_numpy
_apply_slicer_geometry_to_sitk = apply_slicer_geometry_to_sitk


def _volume_node_ijk_to_ras_matrix(volume_node) -> np.ndarray:
    return volume_node_geometry(volume_node)[0]


def _volume_node_ras_to_ijk_matrix(volume_node) -> np.ndarray:
    return volume_node_geometry(volume_node)[1]


def load_image_and_matrices(ctx: DetectionContext):
    """Return ``(sitk_image, ijk_to_ras_4x4, ras_to_ijk_4x4)`` from ``ctx``.

    Public entry point. ``ctx`` is the same shape as ``DetectionContext``:
    either ``ctx['ct'].path`` (preferred) or ``ctx['arr_kji']`` +
    ``ctx['spacing_xyz']``. When ``ctx['extras']['volume_node']`` is
    set, the Slicer node's IJK->RAS overrides the header.

    The split between voxel data (always from disk when path is given)
    and IJK->RAS matrix (Slicer node when present, header otherwise) is
    the CLI/Slicer parity invariant — see feedback_cli_slicer_parity.md.
    """
    import SimpleITK as sitk
    from shank_core.io import image_ijk_ras_matrices

    volume_node = (ctx.get("extras") or {}).get("volume_node")

    ct = ctx.get("ct")
    ct_path = getattr(ct, "path", None) if ct is not None else None
    if ct_path:
        img = sitk.ReadImage(str(ct_path))
        if volume_node is not None and hasattr(volume_node, "GetIJKToRASMatrix"):
            ijk_to_ras, ras_to_ijk = volume_node_geometry(volume_node)
            apply_slicer_geometry_to_sitk(img, ijk_to_ras)
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
        ijk_to_ras, ras_to_ijk = volume_node_geometry(volume_node)
        apply_slicer_geometry_to_sitk(img, ijk_to_ras)
    else:
        ijk_to_ras, ras_to_ijk = image_ijk_ras_matrices(img)

    return (
        img,
        np.asarray(ijk_to_ras, dtype=float),
        np.asarray(ras_to_ijk, dtype=float),
    )


_load_image_and_matrices = load_image_and_matrices


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
