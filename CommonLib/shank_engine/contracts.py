"""Stable data contracts for the pure-Python SEEG detection engine.

This module intentionally has no Slicer dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TypedDict


RESULT_SCHEMA_VERSION = "seeg_detection_result.v1"
DIAGNOSTICS_SCHEMA_VERSION = "seeg_detection_diagnostics.v1"


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


@dataclass(frozen=True)
class BlobRecord:
    """One candidate metal blob."""

    blob_id: int
    centroid_ras: tuple[float, float, float]
    centroid_kji: tuple[float, float, float]
    voxel_count: int
    peak_hu: float = 0.0
    q95_hu: float = 0.0
    mean_hu: float = 0.0
    pca_axis_ras: tuple[float, float, float] = (0.0, 0.0, 1.0)
    pca_evals: tuple[float, float, float] = (0.0, 0.0, 0.0)
    length_mm: float = 0.0
    diameter_mm: float = 0.0
    elongation: float = 0.0
    depth_min_mm: float = 0.0
    depth_max_mm: float = 0.0
    depth_mean_mm: float = 0.0
    scores: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ShankModel:
    """One detected shank model (line/polyline/spline)."""

    shank_id: str
    kind: Literal["line", "polyline", "spline", "unknown"]
    params: dict[str, Any] = field(default_factory=dict)
    support: dict[str, float] = field(default_factory=dict)
    assigned_blob_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ContactRecord:
    """One detected contact center."""

    shank_id: str
    contact_id: str
    position_ras: tuple[float, float, float]
    confidence: float = 0.0


class ArtifactRecord(TypedDict, total=False):
    """One output artifact produced during detection."""

    kind: str
    path: str
    description: str
    stage: str


class DetectionDiagnostics(TypedDict, total=False):
    """Structured diagnostics emitted by any pipeline."""

    schema_version: str
    pipeline_id: str
    run_id: str
    counts: dict[str, int]
    timing: dict[str, float]
    reason_codes: dict[str, int]
    params: dict[str, Any]
    notes: list[str]
    extras: dict[str, Any]


class DetectionResult(TypedDict, total=False):
    """Canonical JSON-serializable output from a pipeline run."""

    schema_version: str
    pipeline_id: str
    pipeline_version: str
    run_id: str
    status: Literal["ok", "error"]
    trajectories: list[dict[str, Any]]
    contacts: list[dict[str, Any]]
    diagnostics: DetectionDiagnostics
    artifacts: list[ArtifactRecord]
    warnings: list[str]
    error: dict[str, Any] | None
    meta: dict[str, Any]


class DetectionContext(TypedDict, total=False):
    """Input context for all pipelines."""

    run_id: str
    ct: VolumeRef
    arr_kji: Any
    spacing_xyz: tuple[float, float, float]
    config: dict[str, Any]
    params: dict[str, Any]  # legacy alias for config
    ijk_kji_to_ras_fn: Any
    ras_to_ijk_fn: Any
    center_ras: list[float]
    components: dict[str, Any]
    extras: dict[str, Any]
    artifact_writer: Any
    logger: Any


@dataclass
class DetectionError(Exception):
    """Structured error helper for pipelines."""

    message: str

    def __str__(self) -> str:
        return self.message


def default_diagnostics(pipeline_id: str, run_id: str, params: dict[str, Any] | None = None) -> DetectionDiagnostics:
    """Create diagnostics object with required sections."""

    return {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
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
    """Create result object with stable keys and empty collections."""

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "pipeline_id": str(pipeline_id),
        "pipeline_version": str(pipeline_version),
        "run_id": str(run_id),
        "status": status,
        "trajectories": [],
        "contacts": [],
        "diagnostics": default_diagnostics(pipeline_id=str(pipeline_id), run_id=str(run_id), params=params),
        "artifacts": [],
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


def _sanitize_diagnostics(
    diagnostics: dict[str, Any] | None,
    *,
    pipeline_id: str,
    run_id: str,
    warnings: list[str],
    strict_schema: bool,
) -> DetectionDiagnostics:
    payload = to_jsonable(diagnostics or {})
    schema_version = str(payload.get("schema_version") or DIAGNOSTICS_SCHEMA_VERSION)
    if schema_version != DIAGNOSTICS_SCHEMA_VERSION:
        if strict_schema:
            raise DetectionError(
                f"diagnostics schema mismatch: expected {DIAGNOSTICS_SCHEMA_VERSION}, got {schema_version}"
            )
        warnings.append(
            f"diagnostics schema coerced from {schema_version} to {DIAGNOSTICS_SCHEMA_VERSION}"
        )
    return {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "pipeline_id": str(payload.get("pipeline_id") or pipeline_id),
        "run_id": str(payload.get("run_id") or run_id),
        "counts": dict(payload.get("counts") or {}),
        "timing": dict(payload.get("timing") or {"total_ms": 0.0}),
        "reason_codes": dict(payload.get("reason_codes") or {}),
        "params": dict(payload.get("params") or {}),
        "notes": list(payload.get("notes") or []),
        "extras": dict(payload.get("extras") or {}),
    }


def sanitize_result(result: DetectionResult, *, strict_schema: bool = False) -> DetectionResult:
    """Normalize a result to the canonical schema and JSON-safe values."""

    payload = to_jsonable(result or {})
    warnings = list(payload.get("warnings") or [])

    schema_version = str(payload.get("schema_version") or RESULT_SCHEMA_VERSION)
    if schema_version != RESULT_SCHEMA_VERSION:
        if strict_schema:
            raise DetectionError(
                f"result schema mismatch: expected {RESULT_SCHEMA_VERSION}, got {schema_version}"
            )
        warnings.append(f"result schema coerced from {schema_version} to {RESULT_SCHEMA_VERSION}")

    pipeline_id = str(payload.get("pipeline_id") or "")
    run_id = str(payload.get("run_id") or "")
    diagnostics = _sanitize_diagnostics(
        payload.get("diagnostics"),
        pipeline_id=pipeline_id,
        run_id=run_id,
        warnings=warnings,
        strict_schema=strict_schema,
    )

    normalized: DetectionResult = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "pipeline_id": pipeline_id,
        "pipeline_version": str(payload.get("pipeline_version") or ""),
        "run_id": run_id,
        "status": "error" if str(payload.get("status") or "ok") == "error" else "ok",
        "trajectories": list(payload.get("trajectories") or []),
        "contacts": list(payload.get("contacts") or []),
        "diagnostics": diagnostics,
        "artifacts": list(payload.get("artifacts") or []),
        "warnings": warnings,
        "error": payload.get("error") if payload.get("error") is None else dict(payload.get("error") or {}),
        "meta": dict(payload.get("meta") or {"extras": {}}),
    }

    if "extras" not in normalized["meta"]:
        normalized["meta"]["extras"] = {}
    if "total_ms" not in normalized["diagnostics"]["timing"]:
        normalized["diagnostics"]["timing"]["total_ms"] = 0.0
    return normalized
