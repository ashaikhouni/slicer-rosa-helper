"""Lightweight input/output contracts for the detection service.

This is the only file the rest of the codebase needs to know about when
calling detection. ``DetectionContext`` is the input shape;
``DetectionResult`` is the output shape. Algorithm internals live under
``DetectionResult['diagnostics']['extras']`` and are not part of the
stable contract — callers who reach in there are explicitly opting into
algorithm-specific debugging detail.

No Slicer / VTK / Qt dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol, TypedDict


@dataclass(frozen=True)
class VolumeRef:
    """Reference metadata for an image volume used by detection."""

    volume_id: str
    path: str | None = None
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction: tuple[float, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MaskRef:
    """Reference metadata for a binary mask used by detection."""

    mask_id: str
    shape_kji: tuple[int, int, int] | None = None
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class DetectionDiagnostics(TypedDict, total=False):
    """Per-run diagnostics. ``extras`` is the algorithm-private grab bag."""

    pipeline_id: str
    run_id: str
    counts: dict[str, int]
    timing: dict[str, float]
    reason_codes: dict[str, int]
    params: dict[str, Any]
    notes: list[str]
    extras: dict[str, Any]


class DetectionResult(TypedDict, total=False):
    """Output shape of any detection service call.

    ``trajectories`` is the public surface. Each entry MUST carry at
    least: ``start_ras``, ``end_ras``, ``confidence``, ``confidence_label``,
    ``model_id``. Algorithm-specific fields go on the same dict but
    callers must treat them as opaque.
    """

    pipeline_id: str
    pipeline_version: str
    run_id: str
    status: Literal["ok", "error"]
    trajectories: list[dict[str, Any]]
    contacts: list[dict[str, Any]]
    diagnostics: DetectionDiagnostics
    warnings: list[str]
    error: dict[str, Any] | None
    meta: dict[str, Any]


class DetectionContext(TypedDict, total=False):
    """Input shape for the detection service.

    Two ways to provide pixel data: ``ct.path`` (preferred, loaded via
    SimpleITK) or ``arr_kji + spacing_xyz`` (in-memory fallback for tests
    and Slicer's pre-loaded volume node).

    ``extras['volume_node']`` (Slicer) is honoured for IJK→RAS matrix
    extraction. ``logger`` (callable) receives stage progress strings.
    """

    run_id: str
    ct: VolumeRef
    arr_kji: Any
    spacing_xyz: tuple[float, float, float]
    config: dict[str, Any]
    params: dict[str, Any]
    extras: dict[str, Any]
    logger: Any


class DetectionPipeline(Protocol):
    """Anything callable as ``pipe.run(ctx) -> DetectionResult``.

    Kept for documentation: a future second pipeline (or test fake)
    matching this Protocol can be plugged into the service without
    changing callers.
    """

    pipeline_id: str
    pipeline_version: str

    def run(self, ctx: DetectionContext) -> DetectionResult: ...


@dataclass
class DetectionError(Exception):
    """Structured error helper for pipelines."""

    message: str

    def __str__(self) -> str:
        return self.message


def default_diagnostics(
    pipeline_id: str,
    run_id: str,
    params: dict[str, Any] | None = None,
) -> DetectionDiagnostics:
    return {
        "pipeline_id": str(pipeline_id),
        "run_id": str(run_id),
        "counts": {},
        "timing": {"total_ms": 0.0},
        "reason_codes": {},
        "params": dict(params or {}),
        "notes": [],
        "extras": {},
    }


def default_result(
    *,
    pipeline_id: str,
    pipeline_version: str,
    run_id: str,
    status: Literal["ok", "error"] = "ok",
    params: dict[str, Any] | None = None,
) -> DetectionResult:
    return {
        "pipeline_id": str(pipeline_id),
        "pipeline_version": str(pipeline_version),
        "run_id": str(run_id),
        "status": status,
        "trajectories": [],
        "contacts": [],
        "diagnostics": default_diagnostics(
            pipeline_id=str(pipeline_id),
            run_id=str(run_id),
            params=params,
        ),
        "warnings": [],
        "error": None,
        "meta": {"extras": {}},
    }


def _jsonable_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def to_jsonable(value: Any) -> Any:
    """Recursively convert structures to JSON-serializable builtins."""

    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return to_jsonable(value.tolist())
        except Exception:
            pass
    if hasattr(value, "__dataclass_fields__"):
        try:
            return to_jsonable(asdict(value))
        except Exception:
            pass
    return _jsonable_scalar(value)


def sanitize_result(result: DetectionResult) -> DetectionResult:
    """Normalize result to JSON-safe builtins with stable keys."""

    payload = to_jsonable(result or {})
    diagnostics = payload.get("diagnostics") or {}
    pipeline_id = str(payload.get("pipeline_id") or "")
    run_id = str(payload.get("run_id") or "")

    normalized: DetectionResult = {
        "pipeline_id": pipeline_id,
        "pipeline_version": str(payload.get("pipeline_version") or ""),
        "run_id": run_id,
        "status": "error" if str(payload.get("status") or "ok") == "error" else "ok",
        "trajectories": list(payload.get("trajectories") or []),
        "contacts": list(payload.get("contacts") or []),
        "diagnostics": {
            "pipeline_id": str(diagnostics.get("pipeline_id") or pipeline_id),
            "run_id": str(diagnostics.get("run_id") or run_id),
            "counts": dict(diagnostics.get("counts") or {}),
            "timing": dict(diagnostics.get("timing") or {"total_ms": 0.0}),
            "reason_codes": dict(diagnostics.get("reason_codes") or {}),
            "params": dict(diagnostics.get("params") or {}),
            "notes": list(diagnostics.get("notes") or []),
            "extras": dict(diagnostics.get("extras") or {}),
        },
        "warnings": list(payload.get("warnings") or []),
        "error": payload.get("error") if payload.get("error") is None else dict(payload.get("error") or {}),
        "meta": dict(payload.get("meta") or {"extras": {}}),
    }

    if "extras" not in normalized["meta"]:
        normalized["meta"]["extras"] = {}
    if "total_ms" not in normalized["diagnostics"]["timing"]:
        normalized["diagnostics"]["timing"]["total_ms"] = 0.0
    return normalized
