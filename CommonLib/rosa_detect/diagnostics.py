"""Diagnostics helpers for pipeline instrumentation."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from .contracts import DetectionDiagnostics, DetectionError


class StageExecutionError(DetectionError):
    """Raised when a pipeline stage fails."""

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = str(stage)


class DiagnosticsCollector:
    """Collect counters/timings/reasons using the canonical diagnostics schema."""

    def __init__(self, diagnostics: DetectionDiagnostics):
        self.diagnostics = diagnostics
        self._ensure_sections()

    def _ensure_sections(self) -> None:
        self.diagnostics.setdefault("counts", {})
        self.diagnostics.setdefault("timing", {"total_ms": 0.0})
        self.diagnostics.setdefault("reason_codes", {})
        self.diagnostics.setdefault("params", {})
        self.diagnostics.setdefault("notes", [])
        self.diagnostics.setdefault("extras", {})

    def inc(self, key: str, amount: int = 1) -> None:
        counts = self.diagnostics["counts"]
        counts[str(key)] = int(counts.get(str(key), 0)) + int(amount)

    def set_count(self, key: str, value: int) -> None:
        self.diagnostics["counts"][str(key)] = int(value)

    def set_timing(self, key: str, value_ms: float) -> None:
        self.diagnostics["timing"][str(key)] = float(value_ms)

    def add_reason(self, code: str, amount: int = 1) -> None:
        reasons = self.diagnostics["reason_codes"]
        reasons[str(code)] = int(reasons.get(str(code), 0)) + int(amount)

    def note(self, message: str) -> None:
        self.diagnostics["notes"].append(str(message))

    def set_param(self, key: str, value: Any) -> None:
        self.diagnostics["params"][str(key)] = value

    def set_extra(self, key: str, value: Any) -> None:
        self.diagnostics["extras"][str(key)] = value

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.set_timing(f"stage.{name}.ms", elapsed_ms)

    def run_stage(self, name: str, fn: Callable[[], Any]) -> Any:
        """Execute one stage with timing and standardized error mapping."""

        start = time.perf_counter()
        try:
            return fn()
        except StageExecutionError:
            raise
        except Exception as exc:
            self.add_reason("stage_exception", 1)
            raise StageExecutionError(stage=name, message=str(exc)) from exc
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.set_timing(f"stage.{name}.ms", elapsed_ms)
