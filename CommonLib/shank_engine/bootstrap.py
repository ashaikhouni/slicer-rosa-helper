"""Bootstrap helpers for built-in shank detection registry entries."""

from __future__ import annotations

from .registry import PipelineRegistry


def register_builtin_pipelines(registry: PipelineRegistry) -> None:
    """Register built-in pipelines in deterministic order.

    Imports are lazy so shank_engine contracts/tests can load without pulling
    optional heavy dependencies until pipeline registration is requested.
    """

    from .pipelines.contact_pitch_v1 import ContactPitchV1Pipeline

    registry.register_pipeline("contact_pitch_v1", ContactPitchV1Pipeline, overwrite=True)
