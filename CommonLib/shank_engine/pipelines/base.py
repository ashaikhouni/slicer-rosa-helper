"""Base helpers for detection pipeline implementations."""

from __future__ import annotations

import time
import traceback
import uuid
from typing import Any

from ..artifacts import FileArtifactWriter, NullArtifactWriter, add_artifact
from ..contracts import DetectionContext, DetectionResult, default_result, sanitize_result
from ..diagnostics import DiagnosticsCollector, StageExecutionError


class BaseDetectionPipeline:
    """Shared helpers for concrete pipeline implementations."""

    pipeline_id = "base"
    pipeline_version = "0.0.0"

    # Key -> default component instance.
    default_components: dict[str, Any] = {}

    def _run_id(self, ctx: DetectionContext) -> str:
        rid = ctx.get("run_id")
        if rid:
            return str(rid)
        return str(uuid.uuid4())

    def _config(self, ctx: DetectionContext) -> dict[str, Any]:
        cfg = ctx.get("config")
        if cfg is None:
            cfg = ctx.get("params")
        return dict(cfg or {})

    def _logger(self, ctx: DetectionContext) -> Any:
        return ctx.get("logger")

    def _log(self, ctx: DetectionContext, message: str) -> None:
        logger = self._logger(ctx)
        if logger is None:
            return
        try:
            logger(str(message))
        except Exception:
            pass

    def make_result(self, ctx: DetectionContext, status: str = "ok") -> DetectionResult:
        return default_result(
            pipeline_id=self.pipeline_id,
            pipeline_version=self.pipeline_version,
            run_id=self._run_id(ctx),
            status="error" if status == "error" else "ok",
            params=self._config(ctx),
        )

    def diagnostics(self, result: DetectionResult) -> DiagnosticsCollector:
        return DiagnosticsCollector(result["diagnostics"])

    def require(self, ctx: DetectionContext, key: str) -> Any:
        if key not in ctx:
            raise ValueError(f"missing required context key: {key}")
        return ctx[key]

    def resolve_component(self, ctx: DetectionContext, key: str, *, required: bool = True) -> Any:
        """Resolve a component using deterministic precedence.

        Order:
        1) ctx.components[key] if present and not None
        2) pipeline default_components[key]
        """

        overrides = dict(ctx.get("components") or {})
        if key in overrides and overrides[key] is not None:
            return overrides[key]
        if key in self.default_components and self.default_components[key] is not None:
            return self.default_components[key]
        if required:
            raise StageExecutionError("component_resolution", f"missing required component '{key}'")
        return None

    def get_artifact_writer(self, ctx: DetectionContext, result: DetectionResult) -> NullArtifactWriter | FileArtifactWriter | Any:
        writer = ctx.get("artifact_writer")
        if writer is not None:
            return writer
        artifact_root = self._config(ctx).get("artifact_root")
        if artifact_root:
            return FileArtifactWriter(str(artifact_root), run_id=str(result.get("run_id") or "run"))
        return NullArtifactWriter(run_id=str(result.get("run_id") or ""))

    def run_stage(
        self,
        *,
        ctx: DetectionContext,
        result: DetectionResult,
        diagnostics: DiagnosticsCollector,
        stage_name: str,
        fn,
    ) -> Any:
        """Run one stage and map all errors to a standard payload."""

        try:
            return diagnostics.run_stage(stage_name, fn)
        except StageExecutionError as exc:
            self.fail(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage=exc.stage,
                exc=exc,
            )
            raise

    def fail(
        self,
        *,
        ctx: DetectionContext,
        result: DetectionResult,
        diagnostics: DiagnosticsCollector,
        stage: str,
        exc: Exception,
    ) -> None:
        result["status"] = "error"
        result["error"] = {"type": exc.__class__.__name__, "message": str(exc), "stage": str(stage)}
        diagnostics.add_reason("pipeline_error", 1)
        diagnostics.add_reason(f"stage:{stage}", 1)
        diagnostics.note(f"stage '{stage}' failed: {exc}")

        if bool(self._config(ctx).get("debug_traceback", False)):
            writer = self.get_artifact_writer(ctx, result)
            try:
                trace_path = writer.write_text("error_traceback.txt", traceback.format_exc())
                add_artifact(
                    result["artifacts"],
                    kind="traceback_txt",
                    path=trace_path,
                    description="pipeline exception traceback",
                    stage=stage,
                )
            except Exception:
                diagnostics.note("failed to write traceback artifact")

    def finalize(self, result: DetectionResult, diagnostics: DiagnosticsCollector, t_start: float) -> DetectionResult:
        diagnostics.set_timing("total_ms", (time.perf_counter() - t_start) * 1000.0)
        return sanitize_result(result)
