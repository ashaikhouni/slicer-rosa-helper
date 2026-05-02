"""SEEG trajectory and contact detection — pure-Python algorithm layer.

Public surface for callers (Slicer modules, CLI, tests):

    from rosa_detect.service import run_contact_pitch_v1
    result = run_contact_pitch_v1(ctx)

Lazy re-exports (PEP 562). Importing this package runs only this file
plus the pure-stdlib contracts + diagnostics modules. The heavy
algorithm modules (``service``, ``contact_pitch_v1_fit``,
``guided_fit_engine``) — which require NumPy, SciPy, SimpleITK — are
imported on first attribute access via ``__getattr__``.

Why: callers reaching pure-Python types (``DetectionContext``,
``DetectedTrajectory``, etc.) shouldn't pull NumPy as a side effect of
``import rosa_detect``. This mirrors the lazy pattern in
``rosa_core/__init__.py`` and keeps the boundary the package's own
docstring claims: NO Slicer / VTK / Qt deps, AND minimal-environment
imports work without the heavy numerical stack.

Algorithm swaps replace the body of ``service.run_contact_pitch_v1``
(or add a sibling ``run_contact_pitch_v2``); callers don't need to
change as long as the ``DetectionResult`` shape is preserved.
"""

from __future__ import annotations

# Eager — these are pure stdlib (typing, dataclasses, math, contextlib).
# Importing them costs nothing and keeps the public type surface
# accessible without triggering the heavy lazy load.
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

# Lazy — heavy modules (numpy / scipy / SimpleITK consumers).
# Map public attribute name -> (submodule, attribute or None).
# attribute=None means "the submodule itself" (for ``from rosa_detect import
# contact_pitch_v1_fit``-style consumers in tests / probes).
_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    # service entry points
    "run_contact_pitch_v1":              ("service", "run_contact_pitch_v1"),
    "run_contact_pitch_v1_with_features": ("service", "run_contact_pitch_v1_with_features"),
    "feature_volume_spec":               ("service", "feature_volume_spec"),
    "feature_volume_node_name":          ("service", "feature_volume_node_name"),
    # whole-module access (legacy probe / test usage)
    "service":                           ("service", None),
    "contact_pitch_v1_fit":              ("contact_pitch_v1_fit", None),
    "guided_fit_engine":                 ("guided_fit_engine", None),
}


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
    "feature_volume_spec",
    "feature_volume_node_name",
    "contact_pitch_v1_fit",
    "guided_fit_engine",
]


def __getattr__(name: str):
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module 'rosa_detect' has no attribute {name!r}")
    module_name, attr = spec
    import importlib
    module = importlib.import_module(f".{module_name}", __name__)
    value = module if attr is None else getattr(module, attr)
    # Cache so subsequent lookups skip __getattr__.
    globals()[name] = value
    return value


def __dir__():
    return list(__all__)
