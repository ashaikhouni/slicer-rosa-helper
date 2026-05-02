"""SEEG trajectory and contact detection — pure-Python algorithm layer.

Public surface for callers (Slicer modules, CLI, tests):

    from rosa_detect.service import run_contact_pitch_v1
    result = run_contact_pitch_v1(ctx)

Everything else in this package is implementation detail. Algorithm
swaps replace the body of ``service.run_contact_pitch_v1`` (or add a
sibling ``run_contact_pitch_v2``); callers don't need to change as long
as the ``DetectionResult`` shape is preserved.

No Slicer / VTK / Qt dependencies in any module here — only NumPy,
SciPy, SimpleITK.
"""

from . import contact_pitch_v1_fit, guided_fit_engine  # noqa: F401
from .contracts import (
    Bolt,
    BoltSource,
    DetectedTrajectory,
    DetectionContext,
    DetectionDiagnostics,
    DetectionError,
    DetectionPipeline,
    DetectionResult,
    MaskRef,
    VolumeRef,
    default_diagnostics,
    default_result,
    is_straight_trajectory,
    sanitize_result,
    to_jsonable,
    trajectory_arc_length_mm,
    trajectory_path_points,
)
from .diagnostics import DiagnosticsCollector, StageExecutionError
from .service import run_contact_pitch_v1, run_contact_pitch_v1_with_features

__all__ = [
    "VolumeRef",
    "MaskRef",
    "DetectionContext",
    "DetectionDiagnostics",
    "DetectionResult",
    "DetectionError",
    "DetectionPipeline",
    "DetectedTrajectory",
    "Bolt",
    "BoltSource",
    "trajectory_path_points",
    "trajectory_arc_length_mm",
    "is_straight_trajectory",
    "DiagnosticsCollector",
    "StageExecutionError",
    "default_diagnostics",
    "default_result",
    "sanitize_result",
    "to_jsonable",
    "run_contact_pitch_v1",
    "run_contact_pitch_v1_with_features",
    "contact_pitch_v1_fit",
    "guided_fit_engine",
]
